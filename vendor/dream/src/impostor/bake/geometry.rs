use crate::impostor::{FallbackReason, ImpostorBounds};
use crate::model::render::PbrVertex;
use glam::{IVec3, Vec2, Vec3};
use std::collections::{BTreeMap, HashMap};

fn axis_overlap(axis: Vec3, half: Vec3, v: [Vec3; 3]) -> bool {
    if axis.length_squared() < 1e-12 {
        return true;
    }
    let p0 = v[0].dot(axis);
    let p1 = v[1].dot(axis);
    let p2 = v[2].dot(axis);
    let (lo, hi) = (p0.min(p1).min(p2), p0.max(p1).max(p2));
    let r = half.x * axis.x.abs() + half.y * axis.y.abs() + half.z * axis.z.abs();
    lo <= r && hi >= -r
}

fn tri_box_overlap(center: Vec3, half: Vec3, tri: [Vec3; 3]) -> bool {
    let v = [tri[0] - center, tri[1] - center, tri[2] - center];
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        if !axis_overlap(axis, half, v) {
            return false;
        }
    }
    let e0 = v[1] - v[0];
    let e1 = v[2] - v[1];
    let e2 = v[0] - v[2];
    let normal = e0.cross(e1);
    if !axis_overlap(normal, half, v) {
        return false;
    }
    for edge in [e0, e1, e2] {
        for base in [Vec3::X, Vec3::Y, Vec3::Z] {
            if !axis_overlap(edge.cross(base), half, v) {
                return false;
            }
        }
    }
    true
}

fn closest_triangle_point(p: Vec3, tri: [Vec3; 3]) -> Vec3 {
    let (a, b, c) = (tri[0], tri[1], tri[2]);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let wa = d3 * d6 - d5 * d4;
    if wa <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (vc + vb + wa).max(1e-12);
    a + ab * (vb * denom) + ac * (wa * denom)
}

fn triangle_barycentric(p: Vec3, tri: [Vec3; 3]) -> Vec3 {
    let v0 = tri[1] - tri[0];
    let v1 = tri[2] - tri[0];
    let v2 = p - tri[0];
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return Vec3::X;
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    Vec3::new(1.0 - v - w, v, w)
}

fn triangle_tangent(vertices: &[PbrVertex], tri_idx: &[u32]) -> [f32; 4] {
    let v = [
        &vertices[tri_idx[0] as usize],
        &vertices[tri_idx[1] as usize],
        &vertices[tri_idx[2] as usize],
    ];
    let uv = v.map(|p| Vec2::from(p.uv0));
    let e1 = Vec3::from(v[1].position) - Vec3::from(v[0].position);
    let e2 = Vec3::from(v[2].position) - Vec3::from(v[0].position);
    let duv1 = uv[1] - uv[0];
    let duv2 = uv[2] - uv[0];
    let det = duv1.x * duv2.y - duv2.x * duv1.y;
    if det.abs() < 1e-8 {
        return [0.0; 4];
    }
    let t = (e1 * duv2.y - e2 * duv1.y) / det;
    let mut t3 = (t - Vec3::from(v[0].normal) * t.dot(Vec3::from(v[0].normal))).normalize_or_zero();
    if t3.length_squared() < 0.5 {
        let n = (Vec3::from(v[1].position) - Vec3::from(v[0].position))
            .cross(Vec3::from(v[2].position) - Vec3::from(v[0].position))
            .normalize_or_zero();
        t3 = n.any_orthonormal_vector();
    }
    [t3.x, t3.y, t3.z, 1.0]
}

pub(super) const TRIANGLE_WORDS: usize = 48;

fn pack_f32s(words: &mut [u32], offset: usize, values: &[f32]) {
    for (word, value) in words[offset..].iter_mut().zip(values) {
        *word = value.to_bits();
    }
}

/// One source triangle in object space. Voxel cells point here so the shader
/// can interpolate UVs/normals at the actual ray/plane hit instead of using
/// material sampled at the voxel centre.
pub(super) fn triangle_record(
    vertices: &[PbrVertex],
    tri_idx: &[u32],
    tri: [Vec3; 3],
    tri_uv0: [Vec2; 3],
    tri_uv1: [Vec2; 3],
    material: usize,
) -> [u32; TRIANGLE_WORDS] {
    let v = [
        &vertices[tri_idx[0] as usize],
        &vertices[tri_idx[1] as usize],
        &vertices[tri_idx[2] as usize],
    ];
    let fallback_tangent = triangle_tangent(vertices, tri_idx);
    let tangents: [[f32; 4]; 3] = if v.iter().all(|p| p.tangent[3].abs() > 0.5) {
        [v[0].tangent, v[1].tangent, v[2].tangent]
    } else {
        [fallback_tangent; 3]
    };
    let mut out = [0u32; TRIANGLE_WORDS];
    pack_f32s(&mut out, 0, &tri[0].to_array());
    pack_f32s(&mut out, 3, &(tri[1] - tri[0]).to_array());
    pack_f32s(&mut out, 6, &(tri[2] - tri[0]).to_array());
    for (i, vertex) in v.iter().enumerate() {
        pack_f32s(&mut out, 9 + i * 3, &vertex.normal);
        pack_f32s(&mut out, 18 + i * 2, &tri_uv0[i].to_array());
        pack_f32s(&mut out, 24 + i * 2, &tri_uv1[i].to_array());
        pack_f32s(&mut out, 30 + i * 4, &tangents[i]);
    }
    out[42] = material as u32;
    out
}

/// Conservative triangle-surface voxels. Each occupied cell packs a signed
/// byte offset to the closest triangle point, that surface normal, and a
/// variable-length list of all overlapping source triangles.
pub(super) fn voxelize_batches(
    batches: &[(Vec<PbrVertex>, Vec<u32>)],
    bounds: ImpostorBounds,
    res: u32,
) -> (Vec<u32>, u32, Vec<u32>, Vec<u32>) {
    if res == 0 {
        return (vec![0], 0, Vec::new(), Vec::new());
    }
    let res_us = res as usize;
    let cell = bounds.extent() / res as f32;
    let half = cell * 0.5;
    let mut cells = vec![0u32; res_us * res_us * res_us * 3];
    let mut cell_triangles = BTreeMap::<usize, Vec<u32>>::new();
    let mut triangles = Vec::<u32>::new();
    let mut triangle_ids = HashMap::<(usize, usize), u32>::new();
    for (material, (vertices, indices)) in batches.iter().enumerate() {
        for (tri_ordinal, tri_idx) in indices.chunks_exact(3).enumerate() {
            let tri = [
                Vec3::from(vertices[tri_idx[0] as usize].position),
                Vec3::from(vertices[tri_idx[1] as usize].position),
                Vec3::from(vertices[tri_idx[2] as usize].position),
            ];
            let normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
            let tri_uv0 = [
                Vec2::from(vertices[tri_idx[0] as usize].uv0),
                Vec2::from(vertices[tri_idx[1] as usize].uv0),
                Vec2::from(vertices[tri_idx[2] as usize].uv0),
            ];
            let tri_uv1 = [
                Vec2::from(vertices[tri_idx[0] as usize].uv1),
                Vec2::from(vertices[tri_idx[1] as usize].uv1),
                Vec2::from(vertices[tri_idx[2] as usize].uv1),
            ];
            let triangle = *triangle_ids
                .entry((material, tri_ordinal))
                .or_insert_with(|| {
                    let id = (triangles.len() / TRIANGLE_WORDS) as u32;
                    triangles.extend_from_slice(&triangle_record(
                        vertices, tri_idx, tri, tri_uv0, tri_uv1, material,
                    ));
                    id
                });
            let mn = tri[0].min(tri[1]).min(tri[2]);
            let mx = tri[0].max(tri[1]).max(tri[2]);
            let lo = ((mn - bounds.min) / cell)
                .floor()
                .as_ivec3()
                .clamp(IVec3::ZERO, IVec3::splat(res as i32 - 1));
            let hi = ((mx - bounds.min) / cell)
                .floor()
                .as_ivec3()
                .clamp(IVec3::ZERO, IVec3::splat(res as i32 - 1));
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let vc = bounds.min + Vec3::new(x as f32, y as f32, z as f32) * cell + half;
                        if !tri_box_overlap(vc, half, tri) {
                            continue;
                        }
                        let cell_index = (z as usize * res_us + y as usize) * res_us + x as usize;
                        let i = cell_index * 3;
                        // Retain every overlapping source triangle. The packed
                        // point/normal below are only a traversal seed; runtime
                        // chooses the nearest true ray/triangle hit from this list.
                        cell_triangles
                            .entry(cell_index)
                            .or_default()
                            .push(triangle + 1);

                        let point = closest_triangle_point(vc, tri);
                        let old = cells[i];
                        if old != 0 {
                            let old_off = Vec3::new(
                                (((old >> 16) & 0xff) as i32 - 128) as f32 / 255.0,
                                (((old >> 8) & 0xff) as i32 - 128) as f32 / 255.0,
                                ((old & 0xff) as i32 - 128) as f32 / 255.0,
                            ) * cell;
                            if old_off.length_squared() <= vc.distance_squared(point) {
                                continue;
                            }
                        }
                        let bary = triangle_barycentric(point, tri);
                        let v = [
                            &vertices[tri_idx[0] as usize],
                            &vertices[tri_idx[1] as usize],
                            &vertices[tri_idx[2] as usize],
                        ];
                        let shading_normal = (Vec3::from(v[0].normal) * bary.x
                            + Vec3::from(v[1].normal) * bary.y
                            + Vec3::from(v[2].normal) * bary.z)
                            .normalize_or(normal);
                        let off = (point - vc) / cell * 255.0;
                        let qx = (off.x.round() as i32 + 128).clamp(0, 255) as u32;
                        let qy = (off.y.round() as i32 + 128).clamp(0, 255) as u32;
                        let qz = (off.z.round() as i32 + 128).clamp(0, 255) as u32;
                        let qn = |v: f32| (v * 127.0).round() as i32 + 128;
                        cells[i] = 0xFF00_0000 | (qx << 16) | (qy << 8) | qz;
                        cells[i + 1] = ((qn(shading_normal.x).clamp(0, 255) as u32) << 16)
                            | ((qn(shading_normal.y).clamp(0, 255) as u32) << 8)
                            | (qn(shading_normal.z).clamp(0, 255) as u32);
                    }
                }
            }
        }
    }
    // Variable-length triangle lists live after the dense cells. Cell word 2
    // stores a one-based word offset into this region; 0 means empty.
    let surface_count = cell_triangles.len() as u32;
    let mut refs = Vec::new();
    for (cell_index, ids) in cell_triangles {
        let offset = refs.len() as u32;
        refs.push(ids.len() as u32);
        refs.extend(ids);
        cells[cell_index * 3 + 2] = offset + 1;
    }
    (cells, surface_count, triangles, refs)
}

/// Per-cell ref-list bound over the packed region. `refs` is the packed
/// [dense, triangle_base) region; returns the max list length checked.
pub(super) fn check_ref_lists(
    refs: &[u32],
    dense: usize,
    triangle_base: usize,
    max_refs: u32,
) -> Result<u64, FallbackReason> {
    let mut cur = dense;
    let mut max_seen = 0u64;
    while cur < triangle_base {
        let n = refs[cur - dense] as u64;
        if n == 0 {
            return Err(FallbackReason::Gpu("voxel ref list is empty".into()));
        }
        max_seen = max_seen.max(n);
        if n > u64::from(max_refs) {
            return Err(FallbackReason::ResourceLimit {
                requested: n,
                limit: u64::from(max_refs),
            });
        }
        cur += 1 + n as usize;
    }
    Ok(max_seen)
}
