//! Impostors: baked card shells that stand in for developer-selected meshes.
//!
//! [`ImpostorBaker`] turns a whole [`Model`](crate::model::Model), an explicit
//! mesh subset, or a grouped [`ImpostorSource`] list into one
//! [`ImpostorAsset`]. The asset stores 26 orthographic captures of
//! albedo+coverage, object-space normal+AO, metallic-roughness, emissive and
//! slab depth, plus a source-surface voxel field and material-map arrays for
//! far-field reconstruction.
//!
//! Rendering stays manual: game code decides when to submit instances through
//! [`ImpostorRenderer::draw`], and controls crossfade opacity through
//! [`ImpostorInstance`]. [`LodBatch`] is an optional helper
//! for the common hysteresis + fade policy, not an automatic engine override.
//! Each submitted instance becomes a 12-triangle bounding-box shell whose
//! fragment shader
//!
//! 1. finds the ray's surface: opaque assets with a voxel grid take a
//!    triangle DDA through the baked source cells (authoritative geometry —
//!    real ray-triangle hits), while MASK-material or voxel-less assets
//!    march the intersection of the *valid* axis slabs (the depth-aware
//!    visual hull);
//! 2. shades the hit point from registered capture views, refined by exact
//!    source-triangle material data when the screen footprint is tiny;
//! 3. feeds albedo/normal/metal/rough/AO/emissive into the shared
//!    [`lighting_eval`](crate::lighting) — the same function
//!    `model_pbr.wgsl` calls, so mesh and card are lit identically;
//! 4. writes the geometry point's depth and its motion vector, preserving
//!    depth sorting and temporal-AA correctness during LoD crossfades.
//!
//! What a card cannot be: inside-out views are wrong (keep the camera
//! outside the box), it cannot represent
//! [`Material::unsupported_reason`](crate::model::Material) materials or
//! animated/skinned geometry, and concavities deeper than the depth resolution
//! may fill in. Unsupported selections return [`FallbackReason`] so callers
//! can keep rendering the mesh.

mod atlas;
mod bake;
pub mod lod;
mod render;
pub mod scene;
mod storage;

pub use atlas::{AssetDebug, ImpostorAsset, ImpostorAssetLimits};
pub use bake::{BakeSettings, ImpostorBaker, ImpostorSource};
pub use lod::{LodChoice, LodOwner, LodPolicy, LodStats};
pub use render::{ImpostorDrawError, ImpostorInstance, ImpostorInstanceError, ImpostorRenderer};
pub use scene::{LodBatch, LodEntry};
pub use storage::ImpostorAssetError;

#[cfg(test)]
use glam::IVec3;
use glam::{Mat4, Vec3};

/// The fourteen views an asset captures: three axis-aligned slab pairs plus
/// four body-diagonal slab pairs. `dir` is the view direction the
/// orthographic camera sits on, `u`/`v` the axes spanning the tile. Order is
/// fixed — layer `f` of every channel image is `VIEWS[f]` — and mirrored
/// verbatim in `impostor.wgsl` (the consistency test parses the shader).
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub dir: Vec3,
    pub u: Vec3,
    pub v: Vec3,
}

pub const VIEW_COUNT: usize = 26;
pub const AXIS_COUNT: usize = 7;

const S3: f32 = 0.577_350_26; // 1/sqrt(3): body-diagonal axes
const S2: f32 = 0.707_106_77; // 1/sqrt(2)
const S6: f32 = 0.408_248_29; // 1/sqrt(6)

pub const VIEWS: [View; VIEW_COUNT] = [
    // +X: look down -X, tile u = -Z, v = +Y
    View {
        dir: Vec3::X,
        u: Vec3::NEG_Z,
        v: Vec3::Y,
    },
    // -X: look down +X, tile u = +Z, v = +Y
    View {
        dir: Vec3::NEG_X,
        u: Vec3::Z,
        v: Vec3::Y,
    },
    // +Y: look down -Y, tile u = +X, v = -Z
    View {
        dir: Vec3::Y,
        u: Vec3::X,
        v: Vec3::NEG_Z,
    },
    // -Y: look down +Y, tile u = +X, v = +Z
    View {
        dir: Vec3::NEG_Y,
        u: Vec3::X,
        v: Vec3::Z,
    },
    // +Z: look down -Z, tile u = +X, v = +Y
    View {
        dir: Vec3::Z,
        u: Vec3::X,
        v: Vec3::Y,
    },
    // -Z: look down +Z, tile u = -X, v = +Y
    View {
        dir: Vec3::NEG_Z,
        u: Vec3::NEG_X,
        v: Vec3::Y,
    },
    // Four body-diagonal slab pairs. These are what let the hull prove
    // oblique holes a six-axis box visual hull fills with synthetic surface.
    View {
        dir: Vec3::new(S3, S3, S3),
        u: Vec3::new(-S2, 0.0, S2),
        v: Vec3::new(S6, -2.0 * S6, S6),
    },
    View {
        dir: Vec3::new(-S3, -S3, -S3),
        u: Vec3::new(S2, 0.0, -S2),
        v: Vec3::new(S6, -2.0 * S6, S6),
    },
    View {
        dir: Vec3::new(S3, S3, -S3),
        u: Vec3::new(S2, 0.0, S2),
        v: Vec3::new(S6, -2.0 * S6, -S6),
    },
    View {
        dir: Vec3::new(-S3, -S3, S3),
        u: Vec3::new(-S2, 0.0, -S2),
        v: Vec3::new(S6, -2.0 * S6, -S6),
    },
    View {
        dir: Vec3::new(S3, -S3, S3),
        u: Vec3::new(-S2, 0.0, S2),
        v: Vec3::new(-S6, -2.0 * S6, -S6),
    },
    View {
        dir: Vec3::new(-S3, S3, -S3),
        u: Vec3::new(S2, 0.0, -S2),
        v: Vec3::new(-S6, -2.0 * S6, -S6),
    },
    View {
        dir: Vec3::new(-S3, S3, S3),
        u: Vec3::new(0.0, S2, -S2),
        v: Vec3::new(-2.0 * S6, -S6, -S6),
    },
    View {
        dir: Vec3::new(S3, -S3, -S3),
        u: Vec3::new(0.0, -S2, S2),
        v: Vec3::new(-2.0 * S6, -S6, -S6),
    },
    // Twelve edge-direction captures for shading only. They are not hull
    // axes: the seven slab pairs above remain the geometry constraint, while
    // these denser directions cut view-boundary material error.
    View {
        dir: Vec3::new(S2, S2, 0.0),
        u: Vec3::new(S2, -S2, 0.0),
        v: Vec3::new(0.0, 0.0, -1.0),
    },
    View {
        dir: Vec3::new(-S2, -S2, 0.0),
        u: Vec3::new(S2, -S2, 0.0),
        v: Vec3::new(0.0, 0.0, 1.0),
    },
    View {
        dir: Vec3::new(S2, -S2, 0.0),
        u: Vec3::new(S2, S2, 0.0),
        v: Vec3::new(0.0, 0.0, 1.0),
    },
    View {
        dir: Vec3::new(-S2, S2, 0.0),
        u: Vec3::new(S2, S2, 0.0),
        v: Vec3::new(0.0, 0.0, -1.0),
    },
    View {
        dir: Vec3::new(S2, 0.0, S2),
        u: Vec3::new(0.0, 1.0, 0.0),
        v: Vec3::new(-S2, 0.0, S2),
    },
    View {
        dir: Vec3::new(-S2, 0.0, -S2),
        u: Vec3::new(0.0, -1.0, 0.0),
        v: Vec3::new(-S2, 0.0, S2),
    },
    View {
        dir: Vec3::new(S2, 0.0, -S2),
        u: Vec3::new(0.0, 1.0, 0.0),
        v: Vec3::new(S2, 0.0, S2),
    },
    View {
        dir: Vec3::new(-S2, 0.0, S2),
        u: Vec3::new(0.0, -1.0, 0.0),
        v: Vec3::new(S2, 0.0, S2),
    },
    View {
        dir: Vec3::new(0.0, S2, S2),
        u: Vec3::new(1.0, 0.0, 0.0),
        v: Vec3::new(0.0, S2, -S2),
    },
    View {
        dir: Vec3::new(0.0, -S2, -S2),
        u: Vec3::new(1.0, 0.0, 0.0),
        v: Vec3::new(0.0, -S2, S2),
    },
    View {
        dir: Vec3::new(0.0, S2, -S2),
        u: Vec3::new(1.0, 0.0, 0.0),
        v: Vec3::new(0.0, -S2, -S2),
    },
    View {
        dir: Vec3::new(0.0, -S2, S2),
        u: Vec3::new(1.0, 0.0, 0.0),
        v: Vec3::new(0.0, S2, S2),
    },
];

/// Projected box size along an arbitrary basis vector. Unlike
/// `abs(extent.dot(axis))`, component signs cannot cancel here.
#[inline]
pub fn projected_extent(extent: glam::Vec3, axis: glam::Vec3) -> f32 {
    extent.abs().dot(axis.abs()).max(1e-3)
}

/// Object-space point -> face tile uv in `[0, 1]`, the same convention as
/// `impostor.wgsl::face_uv`. `extent` is the padded bounds size.
#[inline]
pub fn face_uv(view: &View, extent: glam::Vec3, p: glam::Vec3, c: glam::Vec3) -> glam::Vec2 {
    let e_u = projected_extent(extent, view.u);
    let e_v = projected_extent(extent, view.v);
    glam::Vec2::new(
        (p - c).dot(view.u) / e_u + 0.5,
        0.5 - (p - c).dot(view.v) / e_v,
    )
}

/// Slab pair a hull view belongs to (0..7) and which sign it is. Views 14+
/// are shading captures and do not constrain the hull.
pub fn view_axis(view: usize) -> (usize, f32) {
    assert!(view < AXIS_COUNT * 2, "view {view} is not a hull slab view");
    (view / 2, if view % 2 == 0 { 1.0 } else { -1.0 })
}

/// Why a model cannot be impostor-baked (returned by the baker and
/// [`static_capture_supported`]).
#[derive(Clone, Debug, PartialEq)]
pub enum FallbackReason {
    /// A material uses an extension the card channels cannot represent.
    UnsupportedMaterial { extension: String },
    /// Alpha-mode BLEND materials need order-dependent blending the bake
    /// collapses into one coverage bit.
    BlendMaterial { material: String },
    /// Skinned/animated meshes move; the bake is a static capture.
    SkinnedOrAnimated,
    /// A selected mesh index is not present in its model.
    InvalidSelection { mesh: usize },
    /// A selected mesh has indices outside its vertex buffer or non-finite data.
    MalformedGeometry { mesh: usize },
    /// A source transform is not finite.
    InvalidTransform,
    /// Nothing visible was captured — degenerate geometry.
    EmptyGeometry,
    /// Bake settings or asset limits are out of range.
    InvalidSettings(String),
    /// Estimated or exact resource use exceeds the caller's limit.
    ResourceLimit { requested: u64, limit: u64 },
    /// A GPU/readback operation failed.
    Gpu(String),
}

impl std::fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FallbackReason::UnsupportedMaterial { extension } => {
                write!(f, "unsupported material extension {extension}")
            }
            FallbackReason::BlendMaterial { material } => {
                write!(f, "BLEND material {material} cannot be captured")
            }
            FallbackReason::SkinnedOrAnimated => {
                write!(f, "skinned/animated models are not static")
            }
            FallbackReason::InvalidSelection { mesh } => {
                write!(f, "selected mesh index {mesh} does not exist")
            }
            FallbackReason::MalformedGeometry { mesh } => {
                write!(f, "selected mesh {mesh} has malformed geometry")
            }
            FallbackReason::InvalidTransform => {
                write!(
                    f,
                    "impostor source transform is not finite, affine or invertible"
                )
            }
            FallbackReason::EmptyGeometry => write!(f, "nothing was captured"),
            FallbackReason::InvalidSettings(m) => write!(f, "invalid bake settings: {m}"),
            FallbackReason::ResourceLimit { requested, limit } => {
                write!(f, "bake needs {requested} bytes over limit {limit}")
            }
            FallbackReason::Gpu(message) => write!(f, "gpu bake failed: {message}"),
        }
    }
}

/// Bake/render-time eligibility: static geometry, every material
/// representable, no BLEND.
pub fn static_capture_supported(model: &crate::model::Model) -> Result<(), FallbackReason> {
    if model.meshes.iter().any(|m| m.is_skinned()) || !model.clips.is_empty() {
        return Err(FallbackReason::SkinnedOrAnimated);
    }
    if model.meshes.is_empty() {
        return Err(FallbackReason::EmptyGeometry);
    }
    for material in &model.materials {
        if material.alpha_mode == crate::model::AlphaMode::Blend {
            return Err(FallbackReason::BlendMaterial {
                material: material.name.clone(),
            });
        }
        if let Some(reason) = material.unsupported_reason() {
            return Err(FallbackReason::UnsupportedMaterial { extension: reason });
        }
    }
    Ok(())
}

/// Object-space bounding box used by the shell, the slabs and the camera
/// "inside box" guard. Padded slightly so silhouettes never clip the box.
#[derive(Clone, Copy, Debug)]
pub struct ImpostorBounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl ImpostorBounds {
    /// Pad `pad_frac` of the extent on every side (a few-texel margin).
    pub fn from_model(model: &crate::model::Model, pad_frac: f32) -> Self {
        let (mn, mx) = model.static_bounds();
        let ext = (mx - mn) * pad_frac;
        ImpostorBounds {
            min: mn - ext,
            max: mx + ext,
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    /// Radius used by [`LodPolicy::screen_coverage`].
    pub fn radius(&self) -> f32 {
        self.extent().length() * 0.5
    }

    /// World-space box under `transform`.
    pub fn transformed(&self, transform: Mat4) -> Self {
        let c = self.center();
        let e = self.extent();
        let mut mn = Vec3::splat(f32::INFINITY);
        let mut mx = Vec3::splat(f32::NEG_INFINITY);
        for corner in 0..8 {
            let p = c
                + (Vec3::from([
                    (corner & 1) as f32,
                    ((corner >> 1) & 1) as f32,
                    ((corner >> 2) & 1) as f32,
                ]) * 2.0
                    - Vec3::ONE)
                    * e
                    * 0.5;
            let w = transform.transform_point3(p);
            mn = mn.min(w);
            mx = mx.max(w);
        }
        ImpostorBounds { min: mn, max: mx }
    }

    /// Does `point` lie inside the (padded) box? The LoD guard uses the
    /// world-space transformed box; cards must never render from inside.
    pub fn contains(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// Rust mirror of the WGSL `leaf_hull_hit` segmented lattice scan — same
/// 16-point lattice, same segment/boundary math using the real `VIEWS` /
/// `face_uv` tables. `slab(a, p)` mirrors `axis_slab` at the leaf mip.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn leaf_traverse(
    ro: Vec3,
    rd: Vec3,
    c: Vec3,
    extent: Vec3,
    n0: f32,
    n1: f32,
    res_l: u32,
    valid: u32,
    slab: &dyn Fn(usize, Vec3) -> Option<(f32, f32)>,
) -> Option<Vec3> {
    let res_f = res_l as f32;
    let step = (n1 - n0) / 16.0;
    let rel_ro = ro - c;
    let inside = |p: Vec3| -> bool {
        for a in 0..AXIS_COUNT {
            if valid & (1 << a) == 0 {
                continue;
            }
            let Some((lo_s, hi_s)) = slab(a, p) else {
                return false;
            };
            let dir = VIEWS[2 * a].dir;
            let ext_n = projected_extent(extent, dir);
            let tol = ext_n * (0.5 / res_f);
            let q = (p - c).dot(dir);
            if q < lo_s - tol || q > hi_s + tol {
                return false;
            }
        }
        true
    };
    let mut k = 0i32;
    for _seg in 0..16 {
        if k > 15 {
            break;
        }
        let t_k = n0 + (k as f32 + 0.5) * step;
        let pk = ro + rd * t_k;
        let mut v_lo = n0;
        let mut v_hi = n1;
        let mut next_b = n1;
        let mut ok = true;
        // Boundaries for ALL valid axes first — an invalid slab's region
        // still ends at a texel boundary we must step to.
        for a in 0..AXIS_COUNT {
            if valid & (1 << a) == 0 {
                continue;
            }
            for s in 0..2usize {
                let f = 2 * a + s;
                let view = &VIEWS[f];
                let e_u = projected_extent(extent, view.u);
                let e_v = projected_extent(extent, view.v);
                let uvc = face_uv(view, extent, pk, c) * res_f;
                let du = rd.dot(view.u) / e_u * res_f;
                let dv = -rd.dot(view.v) / e_v * res_f;
                for comp in 0..2 {
                    let coord = if comp == 1 { uvc.y } else { uvc.x };
                    let slope = if comp == 1 { dv } else { du };
                    if slope.abs() < 1e-9 {
                        continue;
                    }
                    // Negative slope: floor(coord) — an exact integer coord
                    // ends the segment immediately (boundary at the point).
                    let nxt = if slope > 0.0 {
                        coord.floor() + 1.0
                    } else {
                        coord.floor()
                    };
                    let dt = (nxt - coord) / slope;
                    if dt >= 0.0 {
                        next_b = next_b.min(t_k + dt);
                    }
                }
            }
        }
        for a in 0..AXIS_COUNT {
            if valid & (1 << a) == 0 {
                continue;
            }
            let Some((lo_s, hi_s)) = slab(a, pk) else {
                ok = false;
                break;
            };
            let dir = VIEWS[2 * a].dir;
            let ext_n = projected_extent(extent, dir);
            let tol = ext_n * (0.5 / res_f);
            let lo = lo_s - tol;
            let hi = hi_s + tol;
            let q0 = rel_ro.dot(dir);
            let qa = rd.dot(dir);
            if qa.abs() < 1e-9 {
                if q0 < lo || q0 > hi {
                    ok = false;
                    break;
                }
            } else {
                let ta = (lo - q0) / qa;
                let tb = (hi - q0) / qa;
                v_lo = v_lo.max(ta.min(tb));
                v_hi = v_hi.min(ta.max(tb));
                if v_lo > v_hi {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let first = k.max(((v_lo - n0) / step - 0.5).ceil() as i32);
            if (0..=15).contains(&first) {
                let t_f = n0 + (first as f32 + 0.5) * step;
                if t_f <= next_b && t_f <= v_hi {
                    let h = ro + rd * t_f;
                    if inside(h) {
                        return Some(h);
                    }
                    k = first + 1;
                    continue;
                }
            }
        }
        let jump = ((next_b - n0) / step - 0.5).ceil() as i32;
        k = (k + 1).max(jump);
    }
    None
}

/// Reference dense 16-lattice scan (the oracle the leaf mirror must match).
#[cfg(test)]
fn reference_16(
    ro: Vec3,
    rd: Vec3,
    c: Vec3,
    extent: Vec3,
    n0: f32,
    n1: f32,
    res_l: u32,
    valid: u32,
    slab: &dyn Fn(usize, Vec3) -> Option<(f32, f32)>,
) -> Option<Vec3> {
    let res_f = res_l as f32;
    for k in 0..16u32 {
        let t = n0 + (k as f32 + 0.5) * (n1 - n0) / 16.0;
        let p = ro + rd * t;
        let mut ok = true;
        for a in 0..AXIS_COUNT {
            if valid & (1 << a) == 0 {
                continue;
            }
            let Some((lo_s, hi_s)) = slab(a, p) else {
                ok = false;
                break;
            };
            let dir = VIEWS[2 * a].dir;
            let ext_n = projected_extent(extent, dir);
            let tol = ext_n * (0.5 / res_f);
            let q = (p - c).dot(dir);
            if q < lo_s - tol || q > hi_s + tol {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(p);
        }
    }
    None
}

/// Rust mirror of `source_ray_hit`: voxel DDA, each occupied cell's whole
/// triangle-ref list intersected within that cell's traversal interval —
/// nearest real hit wins. `bits` is the voxel word buffer and `tri_base` the
/// word offset of the packed triangle records (shader `material.x`). Returns
/// (t, record word base, bary) like the shader's TriHit.
#[cfg(test)]
pub(crate) fn source_ray_hit_mirror(
    ro: Vec3,
    rd: Vec3,
    gmin: Vec3,
    cell: Vec3,
    res: u32,
    bits: &[u32],
    tri_base: u32,
) -> Option<(f32, u32, Vec3)> {
    if res == 0 {
        return None;
    }
    // ray-box interval over the voxel grid
    let gmax = gmin + cell * res as f32;
    let mut lo = f32::NEG_INFINITY;
    let mut hi = f32::INFINITY;
    for a in 0..3 {
        if rd[a].abs() < 1e-9 {
            if ro[a] < gmin[a] || ro[a] > gmax[a] {
                return None;
            }
        } else {
            let ta = (gmin[a] - ro[a]) / rd[a];
            let tb = (gmax[a] - ro[a]) / rd[a];
            lo = lo.max(ta.min(tb));
            hi = hi.min(ta.max(tb));
        }
    }
    if hi <= lo.max(0.0) {
        return None;
    }
    let res_i = res as i32;
    let dense_words = res * res * res * 3;
    let cell_min = cell.min_element();
    let t = lo.max(0.0);
    let mut v = ((ro + rd * t - gmin) / cell)
        .floor()
        .as_ivec3()
        .clamp(IVec3::ZERO, IVec3::splat(res_i - 1));
    let mut step = [0i32; 3];
    let mut t_max = [f32::MAX; 3];
    let mut t_delta = [f32::MAX; 3];
    for a in 0..3 {
        if rd[a] > 1e-9 {
            step[a] = 1;
            t_max[a] = (gmin[a] + (v[a] + 1) as f32 * cell[a] - ro[a]) / rd[a];
            t_delta[a] = cell[a] / rd[a];
        } else if rd[a] < -1e-9 {
            step[a] = -1;
            t_max[a] = (gmin[a] + v[a] as f32 * cell[a] - ro[a]) / rd[a];
            t_delta[a] = -cell[a] / rd[a];
        }
    }
    let cell_of = |v: IVec3| -> usize { ((v.z * res_i + v.y) * res_i + v.x) as usize * 3 };
    let mut entry_t = t;
    for _ in 0..res * 3 + 4 {
        let base = cell_of(v);
        let occ = bits[base];
        let attr = bits[base + 2];
        if occ != 0 && attr != 0 {
            let cell_exit = t_max[0].min(t_max[1]).min(t_max[2]).min(hi);
            let tol = cell_min * 1e-3;
            let t_lo = entry_t.max(lo) - tol;
            let t_hi = cell_exit + tol;
            let ref_base = dense_words + attr - 1;
            let ref_count = bits[ref_base as usize];
            let mut best_t = f32::MAX;
            let mut best_base = 0u32;
            let mut best_bary = Vec3::ZERO;
            for k in 0..ref_count {
                let tri = bits[(ref_base + 1 + k) as usize];
                if tri == 0 {
                    continue;
                }
                let tb2 = tri_base + (tri - 1) * 48;
                let f32at = |i: u32| -> f32 { f32::from_bits(bits[(tb2 + i) as usize]) };
                let v0 = Vec3::new(f32at(0), f32at(1), f32at(2));
                let e1 = Vec3::new(f32at(3), f32at(4), f32at(5));
                let e2 = Vec3::new(f32at(6), f32at(7), f32at(8));
                let gn = e1.cross(e2);
                let n2 = gn.dot(gn);
                if n2 <= 0.0 {
                    continue;
                }
                let denom = rd.dot(gn);
                if denom.abs() <= 1e-7 * n2.sqrt() {
                    continue;
                }
                let t_hit = (v0 - ro).dot(gn) / denom;
                if t_hit < t_lo || t_hit > t_hi || t_hit >= best_t {
                    continue;
                }
                let q = ro + rd * t_hit - v0;
                let u = q.cross(e2).dot(gn) / n2;
                let v2 = e1.cross(q).dot(gn) / n2;
                let w = 1.0 - u - v2;
                if u < -1e-5 || v2 < -1e-5 || w < -1e-5 {
                    continue;
                }
                best_t = t_hit;
                best_base = tb2;
                best_bary = Vec3::new(w, u, v2);
            }
            if best_base != 0 {
                return Some((best_t, best_base, best_bary));
            }
        }
        // advance DDA
        if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            v.x += step[0];
            entry_t = t_max[0];
            t_max[0] += t_delta[0];
        } else if t_max[1] <= t_max[2] {
            v.y += step[1];
            entry_t = t_max[1];
            t_max[1] += t_delta[1];
        } else {
            v.z += step[2];
            entry_t = t_max[2];
            t_max[2] += t_delta[2];
        }
        if entry_t > hi
            || v.x < 0
            || v.y < 0
            || v.z < 0
            || v.x >= res_i
            || v.y >= res_i
            || v.z >= res_i
        {
            break;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each view's (dir, u, v) must be orthonormal so the face_uv projection
    /// in the shader is consistent with the bake's ortho matrix.
    #[test]
    fn view_table_is_orthonormal() {
        for (i, view) in VIEWS.iter().enumerate() {
            assert!(
                view.u.cross(view.v).distance(view.dir) < 1e-5,
                "view {i} u x v != dir"
            );
            assert!((view.u.length() - 1.0).abs() < 1e-6);
            assert!((view.v.length() - 1.0).abs() < 1e-6);
            if i < AXIS_COUNT * 2 {
                assert_eq!(view_axis(i), (i / 2, if i % 2 == 0 { 1.0 } else { -1.0 }));
            }
        }
    }

    #[test]
    fn bounds_pad_and_contain() {
        let b = ImpostorBounds {
            min: Vec3::new(-1.0, 0.0, -2.0),
            max: Vec3::new(1.0, 2.0, 2.0),
        };
        assert!(b.contains(Vec3::ZERO));
        assert!(!b.contains(Vec3::new(0.0, 3.0, 0.0)));
        assert_eq!(b.extent(), Vec3::new(2.0, 2.0, 4.0));
    }
}

/// Pure-Rust mirror of the WGSL hierarchical hull traversal in
/// `impostor.wgsl` (`hull_hit`). `slab(axis, point, level)` returns the
/// sampled (lo, hi) interval for the point's texel at that mip or None
/// when uncovered — the tests feed synthetic min-depth mips.
#[cfg(test)]
/// Traverses to mip 0 (the exact hull level used by tests and closest-range
/// rendering). `hull_traverse_mip` is the level-selectable core.
pub(crate) fn hull_traverse(
    ro: Vec3,
    rd: Vec3,
    c: Vec3,
    gmin: Vec3,
    exts: Vec3,
    res: u32,
    max_mip: u32,
    valid: u32,
    slab: &dyn Fn(usize, Vec3, u32) -> Option<(f32, f32)>,
) -> Option<Vec3> {
    hull_traverse_mip(ro, rd, c, gmin, exts, res, max_mip, 0, valid, slab)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hull_traverse_mip(
    ro: Vec3,
    rd: Vec3,
    c: Vec3,
    gmin: Vec3,
    exts: Vec3,
    res: u32,
    max_mip: u32,
    target_mip: u32,
    valid: u32,
    slab: &dyn Fn(usize, Vec3, u32) -> Option<(f32, f32)>,
) -> Option<Vec3> {
    let target_mip = target_mip.min(max_mip);
    // ray-box interval — per-axis parallel branch (matches WGSL ray_box)
    let mut lo = f32::NEG_INFINITY;
    let mut hi = f32::INFINITY;
    for a in 0..3 {
        if rd[a].abs() < 1e-9 {
            if ro[a] < gmin[a] || ro[a] > gmin[a] + exts[a] {
                return None;
            }
        } else {
            let t_a = (gmin[a] - ro[a]) / rd[a];
            let t_b = (gmin[a] + exts[a] - ro[a]) / rd[a];
            lo = lo.max(t_a.min(t_b));
            hi = hi.min(t_a.max(t_b));
        }
    }
    let tb = (lo, hi);
    if tb.1 <= tb.0.max(0.0) {
        return None;
    }
    let mut cursor = [0.0f32; 12];
    let mut ends = [0.0f32; 12];
    let mut level = max_mip as usize;
    cursor[level] = tb.0.max(0.0);
    ends[level] = tb.1;
    let mut budget = 6 * res + 64;
    loop {
        if budget == 0 {
            return None;
        }
        budget -= 1;
        if cursor[level] >= ends[level] {
            if level == max_mip as usize {
                return None;
            }
            level += 1;
            continue;
        }
        let res_l = res >> level;
        let cell = exts / res_l as f32;
        // Grid-coordinate directional bias (matches the shader).
        let pgrid = (ro + rd * cursor[level] - gmin) / cell + rd.signum() * 1e-5;
        let mut cell_exit = ends[level];
        for a in 0..3 {
            if rd[a].abs() < 1e-9 {
                if ro[a] < gmin[a] || ro[a] > gmin[a] + exts[a] {
                    return None;
                }
                continue;
            }
            let gi = (pgrid[a].floor() as i32).clamp(0, res_l as i32 - 1);
            let side = if rd[a] > 0.0 { gi + 1 } else { gi };
            let t_b = (gmin[a] + side as f32 * cell[a] - ro[a]) / rd[a];
            cell_exit = cell_exit.min(t_b);
        }
        if cell_exit <= cursor[level] {
            cursor[level] += cell.min_element() * 1e-3;
            continue;
        }
        let pm = ro + rd * ((cursor[level] + cell_exit) * 0.5);
        let mut n0 = cursor[level];
        let mut n1 = cell_exit;
        let mut ok = true;
        for a in 0..AXIS_COUNT {
            if valid & (1 << a) == 0 {
                continue;
            }
            // Diagonal slab uvs are not cell-aligned, even at mip 0. Verify
            // those axes in the finest-cell interval scan below rather than
            // trusting one midpoint sample.
            if a >= 3 {
                continue;
            }
            let Some((lo_s, hi_s)) = slab(a, pm, level as u32) else {
                ok = false;
                break;
            };
            let axis = VIEWS[a * 2].dir;
            let tol = exts.dot(axis.abs()) * (0.5 / res_l as f32);
            let lo = lo_s - tol;
            let hi = hi_s + tol;
            let q0 = (ro - c).dot(axis);
            let qa = rd.dot(axis);
            if qa.abs() < 1e-9 {
                if q0 < lo || q0 > hi {
                    ok = false;
                    break;
                }
            } else {
                let t_a = (lo - q0) / qa;
                let t_b = (hi - q0) / qa;
                n0 = n0.max(t_a.min(t_b));
                n1 = n1.min(t_a.max(t_b));
                if n0 > n1 {
                    ok = false;
                    break;
                }
            }
        }
        // Clamp the slab interval to the cell it was sampled in; a coarse
        // accept must not leak traversal into an unexamined neighbor cell.
        n0 = n0.max(cursor[level]);
        n1 = n1.min(cell_exit);
        cursor[level] = cell_exit;
        if !ok || n0 >= n1 {
            continue;
        }
        if level == target_mip as usize {
            // The interval is bounded by the three box-aligned slabs. Diagonal
            // texels can change inside the cell, so scan the interval densely
            // and verify all axes at each candidate point.
            let span = n1 - n0;
            for k in 0..16 {
                let h = ro + rd * (n0 + span * (k as f32 + 0.5) / 16.0);
                let mut inside = true;
                for a in 0..AXIS_COUNT {
                    if valid & (1 << a) == 0 {
                        continue;
                    }
                    let Some((lo_s, hi_s)) = slab(a, h, target_mip) else {
                        inside = false;
                        break;
                    };
                    let axis = VIEWS[a * 2].dir;
                    let q = (h - c).dot(axis);
                    let tol = exts.dot(axis.abs()) * (0.5 / res_l as f32);
                    if q < lo_s - tol || q > hi_s + tol {
                        inside = false;
                        break;
                    }
                }
                if inside {
                    return Some(h);
                }
            }
            continue;
        }
        level -= 1;
        cursor[level] = n0;
        ends[level] = n1;
    }
}

#[test]
fn hull_traverse_solid_box() {
    let c = Vec3::ZERO;
    let gmin = Vec3::splat(-1.0);
    let exts = Vec3::splat(2.0);
    let slab = |_: usize, _: Vec3, _: u32| Some((-0.9, 0.9));
    // +X ray: hit near x = -0.9
    let hit = hull_traverse(
        Vec3::new(-4.0, 0.0, 0.0),
        Vec3::X,
        c,
        gmin,
        exts,
        128,
        7,
        7,
        &slab,
    )
    .expect("solid hull");
    assert!((hit.x - -0.9).abs() < 0.05, "hit {hit:?}");
    // -Z direction
    let hit2 = hull_traverse(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::NEG_Z,
        c,
        gmin,
        exts,
        128,
        7,
        7,
        &slab,
    )
    .expect("neg dir");
    assert!((hit2.z - 0.9).abs() < 0.05, "hit {hit2:?}");
}

#[test]
fn hull_traverse_coarse_only_never_accepted() {
    // Coarse mips report a fat solid; mip0 has the +X slab uncovered
    // for |y| < 0.5 — a hollow middle that must NOT produce a hit there.
    let c = Vec3::ZERO;
    let gmin = Vec3::splat(-1.0);
    let exts = Vec3::splat(2.0);
    let slab = |a: usize, p: Vec3, l: u32| {
        if l > 0 {
            Some((-0.9, 0.9))
        } else if a == 0 && p.y.abs() < 0.5 {
            None
        } else {
            Some((-0.9, 0.9))
        }
    };
    // Ray into the hole on x: must miss (all fine slabs fail on axis 0).
    assert!(
        hull_traverse(
            Vec3::new(-4.0, 0.0, 0.0),
            Vec3::X,
            c,
            gmin,
            exts,
            128,
            7,
            7,
            &slab
        )
        .is_none(),
        "coarse-only hit must not surface"
    );
    // Ray above the hole still hits.
    let hit = hull_traverse(
        Vec3::new(-4.0, 0.8, 0.0),
        Vec3::X,
        c,
        gmin,
        exts,
        128,
        7,
        7,
        &slab,
    );
    assert!(hit.is_some());
}

#[test]
fn hull_traverse_diagonal_gap_and_budget() {
    let c = Vec3::ZERO;
    let gmin = Vec3::splat(-1.0);
    let exts = Vec3::splat(2.0);
    // Thin diagonal gap in the +X slab at mip0 only.
    let slab = |a: usize, p: Vec3, l: u32| {
        if l == 0 && a == 0 && (p.y - p.z).abs() < 0.05 {
            None
        } else {
            Some((-0.9, 0.9))
        }
    };
    // Ray exactly through the gap: must not return a coarse hit point.
    let d = Vec3::new(1.0, 0.02, 0.02).normalize();
    let hit = hull_traverse(
        Vec3::new(-4.0, -0.02, -0.02),
        d,
        c,
        gmin,
        exts,
        128,
        7,
        7,
        &slab,
    );
    // Either a clean miss or a point satisfying all mip-0 slabs.
    if let Some(h) = hit {
        for a in 0..3 {
            let s = slab(a, h, 0);
            assert!(s.is_some(), "hit {h:?} outside mip0 slab {a}");
            let (lo, hi) = s.unwrap();
            let q = h[a] - c[a];
            assert!(q >= lo - 1e-3 && q <= hi + 1e-3);
        }
    }
}

#[test]
fn hull_traverse_excluded_axis_flat() {
    // Flat quad: only axis 1 (Y) is a real slab, thin in y.
    let c = Vec3::ZERO;
    let gmin = Vec3::splat(-1.0);
    let exts = Vec3::splat(2.0);
    let slab = |a: usize, _: Vec3, _: u32| {
        assert_eq!(a, 1);
        Some((-0.05, 0.05))
    };
    // Parallel ray along the slab plane (exercises the rd~0 clip path).
    let hit = hull_traverse(
        Vec3::new(-4.0, 0.0, 0.0),
        Vec3::X,
        c,
        gmin,
        exts,
        128,
        7,
        1 << 1, // only Y valid
        &slab,
    )
    .expect("flat slab hit");
    assert!(hit.y.abs() < 0.06 && hit.x < -0.8, "hit {hit:?}");
    // Angled ray crossing the slab mid-box.
    let hit = hull_traverse(
        Vec3::new(-4.0, -1.2, 0.0),
        Vec3::new(1.0, 0.3, 0.0).normalize(),
        c,
        gmin,
        exts,
        128,
        7,
        1 << 1,
        &slab,
    )
    .expect("angled flat slab hit");
    assert!(hit.y.abs() < 0.06, "hit {hit:?}");
    // Angled ray staying above the slab: miss.
    assert!(
        hull_traverse(
            Vec3::new(-4.0, 0.5, 0.0),
            Vec3::X,
            c,
            gmin,
            exts,
            128,
            7,
            1 << 1,
            &slab,
        )
        .is_none(),
        "parallel ray above the flat slab must miss"
    );
}

#[test]
fn hull_traverse_scale_extremes() {
    // Tiny model (1e-3 extent): the grid-coordinate bias must still hit.
    let c = Vec3::ZERO;
    for scale in [1e-3f32, 1e3f32] {
        let gmin = Vec3::splat(-scale);
        let exts = Vec3::splat(2.0 * scale);
        let slab = |_: usize, _: Vec3, _: u32| Some((-0.9 * scale, 0.9 * scale));
        let hit = hull_traverse(
            Vec3::new(-4.0 * scale, 0.0, 0.0),
            Vec3::X,
            c,
            gmin,
            exts,
            128,
            7,
            7,
            &slab,
        )
        .unwrap_or_else(|| panic!("scale {scale} missed"));
        assert!(
            (hit.x - -0.9 * scale).abs() < 0.05 * scale,
            "scale {scale} hit {hit:?}"
        );
    }
}

#[test]
fn hull_traverse_parallel_on_slab_boundary() {
    // Ray exactly on the slab min/max boundary along a parallel-ish axis.
    let c = Vec3::ZERO;
    let gmin = Vec3::splat(-1.0);
    let exts = Vec3::splat(2.0);
    let slab = |_: usize, _: Vec3, _: u32| Some((-0.9, 0.9));
    // Ray exactly on the slab min/max boundary along the parallel-ish axis:
    // the slab is closed (within extent*1e-4 tolerance) so these must HIT.
    for q in [-0.9f32, 0.9] {
        let hit = hull_traverse(
            Vec3::new(-4.0, q, 0.0),
            Vec3::X,
            c,
            gmin,
            exts,
            128,
            7,
            7,
            &slab,
        )
        .unwrap_or_else(|| panic!("boundary ray q={q} missed"));
        assert!(hit.y.is_finite() && hit.x.is_finite(), "hit {hit:?}");
    }
    // Parallel ray outside the slab entirely: miss.
    assert!(
        hull_traverse(
            Vec3::new(-4.0, 1.5, 0.0),
            Vec3::X,
            c,
            gmin,
            exts,
            128,
            7,
            7,
            &slab
        )
        .is_none()
    );
}

#[test]
fn box_vertex_all_outward_winding() {
    // Parse the ACTUAL shader table — a duplicated Rust copy could not catch
    // a winding bug reintroduced into impostor.wgsl.
    let wgsl = include_str!("impostor.wgsl");
    let table =
        &wgsl[wgsl.find("var verts = array").unwrap()..wgsl.find("return verts[i]").unwrap()];
    let mut verts = Vec::new();
    for m in table.match_indices("vec3<f32>(").map(|(i, _)| i) {
        let rest = &table[m + 10..];
        let nums: Vec<f32> = rest
            .split(')')
            .next()
            .unwrap()
            .split(',')
            .map(|v| v.trim().parse::<f32>().unwrap())
            .collect();
        verts.push([nums[0], nums[1], nums[2]]);
    }
    assert_eq!(verts.len(), 36, "shader box must be 12 triangles");
    let out = [
        Vec3::NEG_X,
        Vec3::X,
        Vec3::NEG_Y,
        Vec3::Y,
        Vec3::NEG_Z,
        Vec3::Z,
    ];
    for f in 0..6 {
        for t in 0..2 {
            let a = Vec3::from(verts[f * 6 + t * 3]);
            let b = Vec3::from(verts[f * 6 + t * 3 + 1]);
            let d = Vec3::from(verts[f * 6 + t * 3 + 2]);
            let n = (b - a).cross(d - a);
            assert!(n.dot(out[f]) > 0.9, "face {f} tri {t} normal {n:?}");
        }
    }
}

// ---- leaf lattice scan (segmented vs reference 16-probe) -----------

#[cfg(test)]
fn leaf_match(
    name: &str,
    ro: Vec3,
    rd: Vec3,
    c: Vec3,
    extent: Vec3,
    n0: f32,
    n1: f32,
    res_l: u32,
    valid: u32,
    slab: &dyn Fn(usize, Vec3) -> Option<(f32, f32)>,
) {
    let acc = leaf_traverse(ro, rd, c, extent, n0, n1, res_l, valid, slab);
    let ref16 = reference_16(ro, rd, c, extent, n0, n1, res_l, valid, slab);
    match (acc, ref16) {
        (None, None) => {}
        (Some(a), Some(b)) => assert!(
            (a - b).length() < 1e-4,
            "{name}: accelerated {a:?} != reference {b:?}"
        ),
        _ => panic!("{name}: accelerated {acc:?} vs reference {ref16:?}"),
    }
}

#[test]
fn leaf_scan_solid_and_empty() {
    let c = Vec3::ZERO;
    let ext = Vec3::splat(2.0);
    let solid = |_: usize, _: Vec3| Some((-0.9, 0.9));
    let empty = |_: usize, _: Vec3| None;
    let rays = [
        (Vec3::new(-4.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(0.0, -4.0, 0.0), Vec3::Y),
        (
            Vec3::new(4.0, 4.0, 4.0),
            Vec3::new(-1.0, -1.0, -1.0).normalize(),
        ),
        (
            Vec3::new(-4.0, 0.3, 0.7),
            Vec3::new(1.0, -0.1, 0.2).normalize(),
        ),
    ];
    for (ro, rd) in rays {
        for (n0, n1) in [(2.0f32, 4.0), (2.5, 3.5), (2.0, 2.3)] {
            leaf_match("solid", ro, rd, c, ext, n0, n1, 512, 0x7f, &solid);
            leaf_match("empty", ro, rd, c, ext, n0, n1, 512, 0x7f, &empty);
        }
    }
}

#[test]
fn leaf_scan_thin_quad_parallel_boundary() {
    let c = Vec3::ZERO;
    let ext = Vec3::splat(2.0);
    // Only axis 1 (Y) valid, thin slab [0.1, 0.15].
    let slab = |a: usize, _: Vec3| {
        if a == 1 {
            Some((0.1, 0.15))
        } else {
            Some((-1e9, 1e9))
        }
    };
    // Ray along +x at y inside the slab -> hit; y outside -> miss.
    leaf_match(
        "quad-in",
        Vec3::new(-4.0, 0.12, 0.0),
        Vec3::X,
        c,
        ext,
        2.0,
        4.0,
        512,
        2,
        &slab,
    );
    leaf_match(
        "quad-out",
        Vec3::new(-4.0, 0.5, 0.0),
        Vec3::X,
        c,
        ext,
        2.0,
        4.0,
        512,
        2,
        &slab,
    );
    // Ray EXACTLY on the slab lo/hi boundary (within tolerance: hits).
    leaf_match(
        "quad-lo",
        Vec3::new(-4.0, 0.1, 0.0),
        Vec3::X,
        c,
        ext,
        2.0,
        4.0,
        512,
        2,
        &slab,
    );
    leaf_match(
        "quad-hi",
        Vec3::new(-4.0, 0.15, 0.0),
        Vec3::X,
        c,
        ext,
        2.0,
        4.0,
        512,
        2,
        &slab,
    );
    // Short interval containing no lattice point inside the slab.
    leaf_match(
        "quad-short",
        Vec3::new(-4.0, -0.5, 0.0),
        Vec3::X,
        c,
        ext,
        2.0,
        2.02,
        512,
        2,
        &slab,
    );
}

#[test]
fn leaf_scan_texel_boundary_directions() {
    // Slab depends on the texel coord of the +X face uv so segment
    // boundaries actually cut the ray; start exactly on an integer texel
    // boundary in both directions.
    let c = Vec3::ZERO;
    let ext = Vec3::splat(2.0);
    let res_l = 8u32;
    // +X face uv.x = (-z)/2+0.5; slab covered only in texel column 6.
    let slab = move |a: usize, p: Vec3| {
        if a != 0 {
            return Some((-1e9, 1e9));
        }
        let uvp = face_uv(&VIEWS[0], ext, p, c);
        let uvm = face_uv(&VIEWS[1], ext, p, c);
        let col_p = (uvp.x * res_l as f32).floor();
        let col_m = (uvm.x * res_l as f32).floor();
        if col_p == 6.0 && col_m == 1.0 {
            Some((-0.9, 0.9))
        } else {
            None
        }
    };
    // Column 6 of +X u=-Z: u>0.75 -> z < -0.5; -X u=+Z col1 -> z < -0.5.
    // Ray along -z through z=-0.7 (in column), y=0.
    leaf_match(
        "boundary-neg",
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::NEG_Z,
        c,
        ext,
        3.5,
        5.0,
        res_l,
        1,
        &slab,
    );
    leaf_match(
        "boundary-pos",
        Vec3::new(0.0, 0.0, -4.0),
        Vec3::Z,
        c,
        ext,
        3.5,
        5.0,
        res_l,
        1,
        &slab,
    );
    // Ray starting exactly on a texel boundary: z such that uvx integer.
    // uvx*8 integer -> -z/2+0.5 = k/8 -> z = 1 - k/4; pick z=0.5 (k=2).
    leaf_match(
        "boundary-exact",
        Vec3::new(0.0, 0.0, 4.5),
        Vec3::NEG_Z,
        c,
        ext,
        3.5,
        5.5,
        res_l,
        1,
        &slab,
    );
}

#[test]
fn leaf_scan_late_segment_hit() {
    // Diagonal axis texel changes midway through the interval: only a
    // later segment's texel is covered.
    let c = Vec3::ZERO;
    let ext = Vec3::splat(2.0);
    let res_l = 8u32;
    // Axis 3 (+dir face 6, -dir face 7): covered only at v-row >= 5 of +f.
    let slab = move |a: usize, p: Vec3| {
        if a != 3 {
            return Some((-0.9, 0.9));
        }
        let uvp = face_uv(&VIEWS[6], ext, p, c);
        let uvm = face_uv(&VIEWS[7], ext, p, c);
        let row_p = (uvp.y * res_l as f32).floor();
        let row_m = (uvm.y * res_l as f32).floor();
        if row_p >= 5.0 && row_m >= 5.0 {
            Some((-0.9, 0.9))
        } else {
            None
        }
    };
    // Ray along +x with y chosen so the v rows are covered late/early.
    for y in [-0.9f32, -0.2, 0.5, 0.9] {
        leaf_match(
            "late-seg",
            Vec3::new(-4.0, y, 0.3),
            Vec3::new(1.0, 0.05, 0.0).normalize(),
            c,
            ext,
            2.0,
            4.5,
            res_l,
            0x7f,
            &slab,
        );
    }
}

#[test]
fn leaf_scan_scale_extremes() {
    for scale in [1e-3f32, 1e3] {
        let c = Vec3::ZERO;
        let ext = Vec3::splat(2.0 * scale);
        let slab = move |_: usize, _: Vec3| Some((-0.9 * scale, 0.9 * scale));
        leaf_match(
            "scale",
            Vec3::new(-4.0 * scale, 0.0, 0.0),
            Vec3::X,
            c,
            ext,
            2.0 * scale,
            4.0 * scale,
            512,
            0x7f,
            &slab,
        );
    }
}

// ---- source_ray_hit DDA mirror over packed voxel buffers ------------

/// Build a packed voxel buffer: dense 3-word cells, then 1-based ref
/// lists, then 48-word triangle records. `cells[i]` = (x,y,z, [tri ids]).
/// `tris[i]` = (v0, e1, e2) for record i.
#[cfg(test)]
fn packed_voxels(
    res: u32,
    cells: &[(usize, usize, usize, Vec<u32>)],
    tris: &[(Vec3, Vec3, Vec3)],
) -> (Vec<u32>, u32) {
    let n = (res * res * res * 3) as usize;
    let mut bits = vec![0u32; n];
    let mut refs: Vec<u32> = Vec::new();
    for &(x, y, z, ref list) in cells {
        let idx = ((z * res as usize + y) * res as usize + x) * 3;
        bits[idx] = 1; // occupied seed word
        bits[idx + 1] = 0x808080; // packed normal
        bits[idx + 2] = (refs.len() + 1) as u32; // 1-based attr
        refs.push(list.len() as u32);
        refs.extend(list.iter());
    }
    let tri_base = (n + refs.len()) as u32;
    bits.extend(refs);
    for &(v0, e1, e2) in tris {
        let mut rec = [0u32; 48];
        for (i, f) in v0
            .to_array()
            .iter()
            .chain(e1.to_array().iter())
            .chain(e2.to_array().iter())
            .enumerate()
        {
            rec[i] = f.to_bits();
        }
        bits.extend_from_slice(&rec);
    }
    (bits, tri_base)
}

/// Flat z=0.4 quad covering a patch: v0=(-1,-1,.4), e1=(2,0,0), e2=(0,2,0).
#[cfg(test)]
fn quad_tri(z: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> (Vec3, Vec3, Vec3) {
    (
        Vec3::new(x0, y0, z),
        Vec3::new(x1 - x0, 0.0, 0.0),
        Vec3::new(0.0, y1 - y0, 0.0),
    )
}

#[test]
fn source_dda_packed_cases() {
    let res = 4u32;
    let gmin = Vec3::splat(-1.0);
    let cell = Vec3::splat(0.5);
    // Two cells along x: cell (0,0,0) [x -1..-0.5] refs a triangle the
    // ray misses; cell (1,0,0) refs a quad it hits.
    // near_miss lies off the ray; hit_tri is a z=-0.2 x-facing quad —
    // wait: the ray runs along -x so faces must be x-normal. Use
    // x-planes: v0=(x0,y,z) with e1,e2 spanning y/z.
    let near_miss = (
        Vec3::new(-0.5, 0.05, 0.05),
        Vec3::new(0.0, 0.02, 0.0),
        Vec3::new(0.0, 0.0, 0.02),
    ); // tiny quad off the ray in cell (2,2,2)
    // x=-0.2 triangle containing the ray point (y=0.1, z=0):
    // u=0.1, v=0, w=0.9.
    let hit_tri = (
        Vec3::new(-0.2, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 0.5),
    );
    let (bits, tri_base) = packed_voxels(
        res,
        &[
            (2, 2, 2, vec![1]), // first occupied cell: ref misses the ray
            (1, 2, 2, vec![2]), // next cell: real hit
        ],
        &[near_miss, hit_tri],
    );
    // Ray along -x through y=0.1, z=0.
    let ro = Vec3::new(4.0, 0.1, 0.0);
    let rd = Vec3::NEG_X;
    let hit = source_ray_hit_mirror(ro, rd, gmin, cell, res, &bits, tri_base);
    assert!(hit.is_some(), "occupied miss cell must not stop the DDA");
    let (t, base, _) = hit.unwrap();
    assert_eq!(base, tri_base + 48, "second triangle should win");
    assert!((ro + rd * t - Vec3::new(-0.2, 0.1, 0.0)).length() < 1e-4);
    assert!((t - 4.2).abs() < 1e-4);

    // Reverse ref order in one cell: nearest must still win.
    let far_quad = (
        Vec3::new(-0.8, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 0.5),
    );
    let near_quad = (
        Vec3::new(-0.3, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 0.5),
    );
    let (bits2, tb2) = packed_voxels(res, &[(1, 2, 2, vec![1, 2])], &[far_quad, near_quad]);
    let h = source_ray_hit_mirror(
        Vec3::new(4.0, 0.1, 0.0),
        Vec3::NEG_X,
        gmin,
        cell,
        res,
        &bits2,
        tb2,
    )
    .expect("hit");
    assert_eq!(h.1, tb2 + 48, "nearest ref wins regardless of list order");
    assert!((h.0 - 4.3).abs() < 1e-4, "t={}", h.0);

    // Disjoint components: gap between occupied cells -> rays through the
    // gap miss.
    let (bits3, tb3) = packed_voxels(
        res,
        &[(3, 3, 3, vec![1]), (0, 0, 0, vec![1])],
        &[quad_tri(-0.1, -1.0, -0.9, -1.0, -0.9)],
    );
    let h3 = source_ray_hit_mirror(
        Vec3::new(4.0, 0.5, 0.5),
        Vec3::NEG_X,
        gmin,
        cell,
        res,
        &bits3,
        tb3,
    );
    assert!(h3.is_none(), "gap ray must miss");

    // Axis-parallel boundary ray on a cell plane: still correct.
    let h4 = source_ray_hit_mirror(
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::NEG_X,
        gmin,
        cell,
        res,
        &bits,
        tri_base,
    );
    assert!(h4.is_some(), "boundary ray through shared cell hits");

    // >32 refs: the true nearest at index > 32 wins.
    let mut refs_far = vec![1u32; 40]; // 40 refs to the far triangle
    refs_far.push(2); // last ref is the near one
    let (bits5, tb5) = packed_voxels(res, &[(1, 2, 2, refs_far)], &[far_quad, near_quad]);
    let h5 = source_ray_hit_mirror(
        Vec3::new(4.0, 0.1, 0.0),
        Vec3::NEG_X,
        gmin,
        cell,
        res,
        &bits5,
        tb5,
    )
    .expect("hit past index 32");
    assert_eq!(h5.1, tb5 + 48, "ref index 41 (nearest) must win");
}

#[test]
fn source_dda_scale_invariance() {
    // Same triangle at 1e-3 / 1 / 1e3 world scale must intersect with the
    // same barycentrics; no size-dependent floors.
    for scale in [1e-3f32, 1.0, 1e3] {
        let res = 4u32;
        let gmin = Vec3::splat(-1.0 * scale);
        let cell = Vec3::splat(0.5 * scale);
        // x-plane triangle at x=-0.2s; ray hits (y,z)=(0.1s,0). With
        // v0=(-0.2s,0,-0.05s), e1=(0,s,0), e2=(0,0,0.5s): u=0.1, v=0.1, w=0.8.
        let tri = (
            Vec3::new(-0.2 * scale, 0.0, -0.05 * scale),
            Vec3::new(0.0, 1.0 * scale, 0.0),
            Vec3::new(0.0, 0.0, 0.5 * scale),
        );
        let (bits, tb) = packed_voxels(res, &[(1, 2, 2, vec![1])], &[tri]);
        let ro = Vec3::new(4.0 * scale, 0.1 * scale, 0.0);
        let hit = source_ray_hit_mirror(ro, Vec3::NEG_X, gmin, cell, res, &bits, tb)
            .unwrap_or_else(|| panic!("scale {scale} must hit"));
        assert!(
            (hit.0 - 4.2 * scale).abs() < 1e-3 * scale.max(1.0),
            "t at {scale}"
        );
        let (w, u, v) = (hit.2.x, hit.2.y, hit.2.z);
        assert!(
            (w - 0.8).abs() < 1e-4 && (u - 0.1).abs() < 1e-4 && (v - 0.1).abs() < 1e-4,
            "bary ({w},{u},{v}) at scale {scale}"
        );
        // Inside the occupied cell's column but with true bary
        // (w,u,v)=(-0.25,0.45,0.8): the ray is outside the triangle and
        // must be rejected by the barycentric test, not by bounds alone.
        let miss = source_ray_hit_mirror(
            Vec3::new(4.0 * scale, 0.45 * scale, 0.35 * scale),
            Vec3::NEG_X,
            gmin,
            cell,
            res,
            &bits,
            tb,
        );
        assert!(miss.is_none(), "bary-rejected ray must miss at {scale}");
    }
    // Degenerate record (zero area) must never hit.
    let res = 4u32;
    let gmin = Vec3::splat(-1.0);
    let cell = Vec3::splat(0.5);
    let degen = (Vec3::ZERO, Vec3::X, Vec3::X * 2.0);
    let (bits, tb) = packed_voxels(res, &[(1, 2, 2, vec![1])], &[degen]);
    assert!(
        source_ray_hit_mirror(
            Vec3::new(4.0, 0.1, 0.0),
            Vec3::NEG_X,
            gmin,
            cell,
            res,
            &bits,
            tb
        )
        .is_none()
    );
}

#[test]
fn source_fast_shader_gate_contract() {
    // material.w > 0.5 selects the triangle-DDA path; flag semantics are
    // covered by render.rs::source_geometry_flag_matrix.
    let wgsl = include_str!("impostor.wgsl");
    assert!(wgsl.contains("asset.material.w > 0.5"), "flag gate present");
    assert!(wgsl.contains("source_ray_hit"), "DDA path present");
}
