use super::materials::imp_material_params;
use super::{ImpostorSource, PreparedBake};
use crate::impostor::{FallbackReason, ImpostorBounds};
use crate::model::render::PbrVertex;
use crate::model::{AlphaMode, Material, Model};
use glam::{Mat4, Vec3};
use std::collections::{BTreeSet, HashMap};

/// Validate and merge the selected sources by material. Node transforms are
/// applied to positions, normals and tangents so the bake sees shared object
/// space; each selected mesh keeps an index into the deduplicated material
/// list returned in [`PreparedBake`].
pub(super) fn prepare_sources<'a>(
    sources: &[ImpostorSource<'a>],
    pad_frac: f32,
) -> Result<PreparedBake<'a>, FallbackReason> {
    if sources.is_empty() {
        return Err(FallbackReason::EmptyGeometry);
    }
    let mut batches: Vec<(Vec<PbrVertex>, Vec<u32>)> = Vec::new();
    let mut materials: Vec<&'a Material> = Vec::new();
    let mut material_ids: HashMap<(usize, usize), usize> = HashMap::new();
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    let mut has_geometry = false;

    for source in sources {
        // Bake-space transforms may be non-uniform or reflected (they bake
        // into geometry), but must be finite, affine and invertible — a
        // singular transform maps the whole model onto a plane/point and
        // poisons normals and bounds, not just one vertex.
        let m = source.transform.to_cols_array_2d();
        if !m.iter().flatten().all(|v| v.is_finite()) {
            return Err(FallbackReason::InvalidTransform);
        }
        let affine = [m[0][3], m[1][3], m[2][3], m[3][3]]
            .iter()
            .zip([0.0f32, 0.0, 0.0, 1.0])
            .all(|(a, b)| (a - b).abs() <= 1e-6);
        let inv = source.transform.inverse();
        if !affine || !inv.to_cols_array().iter().all(|v| v.is_finite()) {
            return Err(FallbackReason::InvalidTransform);
        }
        if !source.model.clips.is_empty() {
            return Err(FallbackReason::SkinnedOrAnimated);
        }
        let globals = source.model.rest_globals();
        let selected: BTreeSet<usize> = match source.meshes {
            Some(indices) => indices.iter().copied().collect(),
            None => (0..source.model.meshes.len()).collect(),
        };
        for mesh_index in selected {
            let mesh = source
                .model
                .meshes
                .get(mesh_index)
                .ok_or(FallbackReason::InvalidSelection { mesh: mesh_index })?;
            if mesh.is_skinned() {
                return Err(FallbackReason::SkinnedOrAnimated);
            }
            if mesh.positions.is_empty() || mesh.indices.is_empty() {
                continue;
            }
            if mesh.indices.len() % 3 != 0
                || mesh
                    .indices
                    .iter()
                    .any(|&i| i as usize >= mesh.positions.len())
            {
                return Err(FallbackReason::MalformedGeometry { mesh: mesh_index });
            }
            let material = source
                .model
                .materials
                .get(mesh.material)
                .ok_or(FallbackReason::MalformedGeometry { mesh: mesh_index })?;
            if material.alpha_mode == AlphaMode::Blend {
                return Err(FallbackReason::BlendMaterial {
                    material: material.name.clone(),
                });
            }
            if let Some(extension) = material.unsupported_reason() {
                return Err(FallbackReason::UnsupportedMaterial { extension });
            }
            // Non-finite material data — including UV transform fields —
            // would become shader NaNs; fail typed before any GPU upload.
            {
                let params = imp_material_params(material);
                if !bytemuck::bytes_of(&params)
                    .chunks_exact(4)
                    .all(|c| f32::from_le_bytes(c.try_into().unwrap()).is_finite())
                {
                    return Err(FallbackReason::InvalidSettings(format!(
                        "material {} has non-finite parameters",
                        material.name
                    )));
                }
            }
            let node = mesh
                .node
                .and_then(|n| globals.get(n).copied())
                .unwrap_or(Mat4::IDENTITY);
            let transform = source.transform * node;
            // The COMPOSED transform (source * rest-node) must be finite,
            // affine and invertible — a singular node transform bakes a
            // collapsed copy of the mesh, not an error path.
            {
                let m = transform.to_cols_array_2d();
                let affine = [m[0][3], m[1][3], m[2][3], m[3][3]]
                    .iter()
                    .zip([0.0f32, 0.0, 0.0, 1.0])
                    .all(|(a, b)| (a - b).abs() <= 1e-6);
                if !m.iter().flatten().all(|v| v.is_finite())
                    || !affine
                    || !transform
                        .inverse()
                        .to_cols_array()
                        .iter()
                        .all(|v| v.is_finite())
                {
                    return Err(FallbackReason::InvalidTransform);
                }
            }
            let normal_mat = transform.inverse().transpose();
            let handed = transform.determinant().signum();
            let material_key = (source.model as *const Model as usize, mesh.material);
            let material_index = *material_ids.entry(material_key).or_insert_with(|| {
                materials.push(material);
                batches.push((Vec::new(), Vec::new()));
                materials.len() - 1
            });
            let (verts, idxs) = &mut batches[material_index];
            let base = verts.len() as u32;
            for (i, &position) in mesh.positions.iter().enumerate() {
                let local = Vec3::from(position);
                if !local.is_finite() {
                    return Err(FallbackReason::MalformedGeometry { mesh: mesh_index });
                }
                let p = transform.transform_point3(local);
                if !p.is_finite() {
                    return Err(FallbackReason::MalformedGeometry { mesh: mesh_index });
                }
                let raw_n = mesh.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                if !raw_n.iter().all(|v| v.is_finite()) {
                    return Err(FallbackReason::MalformedGeometry { mesh: mesh_index });
                }
                let n = normal_mat
                    .transform_vector3(Vec3::from(raw_n))
                    .normalize_or_zero();
                let t = mesh.tangents.get(i).copied().unwrap_or([0.0; 4]);
                if !t.iter().all(|v| v.is_finite()) {
                    return Err(FallbackReason::MalformedGeometry { mesh: mesh_index });
                }
                // Tangents are LINEAR-space vectors (unlike normals): the
                // plain transform is correct, then Gram-Schmidt against the
                // inverse-transpose normal. Handedness flips under
                // reflection.
                let t_lin = transform.transform_vector3(Vec3::from([t[0], t[1], t[2]]));
                let t3 = (t_lin - n * t_lin.dot(n)).normalize_or_zero();
                let uv0 = mesh.uvs.get(i).copied().unwrap_or([0.0, 0.0]);
                let uv1 = mesh.uvs1.get(i).copied().unwrap_or([0.0, 0.0]);
                // Malformed source data must fail typed here, not turn into
                // shader NaNs downstream.
                if !n.is_finite()
                    || !t3.is_finite()
                    || !uv0.iter().all(|v| v.is_finite())
                    || !uv1.iter().all(|v| v.is_finite())
                {
                    return Err(FallbackReason::MalformedGeometry { mesh: mesh_index });
                }
                verts.push(PbrVertex {
                    position: p.to_array(),
                    normal: n.to_array(),
                    uv0,
                    uv1,
                    tangent: [t3.x, t3.y, t3.z, t[3] * handed],
                });
                mn = mn.min(p);
                mx = mx.max(p);
            }
            idxs.extend(mesh.indices.iter().map(|i| i + base));
            has_geometry = true;
        }
    }
    if !has_geometry {
        return Err(FallbackReason::EmptyGeometry);
    }
    let pad = (mx - mn) * pad_frac;
    Ok(PreparedBake {
        batches,
        materials,
        bounds: ImpostorBounds {
            min: mn - pad,
            max: mx + pad,
        },
    })
}

/// Test helper: all meshes in one model, merged by material.
#[cfg(test)]
pub(super) fn merge_by_material(model: &Model) -> Vec<(Vec<PbrVertex>, Vec<u32>)> {
    prepare_sources(&[ImpostorSource::model(model)], 0.0)
        .map(|prepared| prepared.batches)
        .unwrap_or_default()
}
