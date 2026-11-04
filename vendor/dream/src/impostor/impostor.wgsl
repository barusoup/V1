// Impostor card: instanced bounding-box shells shaded as the baked mesh.
// Composed after lighting.wgsl (set 3 = lighting_env) at load.
//
// The shader keeps two guarantees the whole system leans on:
//  - geometry is authoritative: opaque assets with a voxel grid take the
//    source-triangle DDA (real ray-triangle hits own coverage, depth and
//    motion); MASK/voxel-less assets march the depth-slab hull instead,
//    which is continuous across face boundaries in both cases;
//  - shading is visibility-weighted projective texturing into the shared
//    lighting_eval with the geometry world position, so the card and the
//    mesh run byte-identical lighting.

struct Camera {
    view_projection: mat4x4<f32>,
    curr_view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Instance {
    model: mat4x4<f32>,
    prev_model: mat4x4<f32>,
    inv_model: mat4x4<f32>,
    misc: vec4<f32>,     // alpha, -, -, -
};

struct Instances { items: array<Instance> };
@group(1) @binding(0) var<storage, read> instances: Instances;

// Per-asset parameters: the box the shell/hull/slabs live in, the 26 view
// axes (u,v span each tile; dir = u x v is implicit), coverage/debug knobs.
struct Asset {
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    // x: coverage cutoff  y: tile resolution  z: slab_valid bitmask (bit a
    // set -> axis a contributes its slab)  w: debug mode
    params: vec4<f32>,
    // x: visibility tolerance in world units  y: tan(fov/2)*2/height (world
    // units per pixel per unit distance)  z: grad scale  w: voxel res
    screen: vec4<f32>,
    // x: triangle-record word offset  y: material-map resolution  z: layers
    material: vec4<f32>,
    u_axes: array<vec4<f32>, 26>,
    v_axes: array<vec4<f32>, 26>,
    extents: array<vec4<f32>, 26>, // per-view ext_u, ext_v, ext_n, -
    views: array<vec4<f32>, 26>,   // per-view dir (points from model to camera side)
};
@group(2) @binding(0) var<uniform> asset: Asset;
@group(2) @binding(1) var t_albedo: texture_2d_array<f32>;
@group(2) @binding(2) var t_normal_ao: texture_2d_array<f32>;
@group(2) @binding(3) var t_mr: texture_2d_array<f32>;
@group(2) @binding(4) var t_emissive: texture_2d_array<f32>;
@group(2) @binding(5) var t_depth: texture_2d_array<f32>;
@group(2) @binding(6) var s_tile: sampler;
struct VoxelBits { bits: array<u32> };
@group(2) @binding(7) var<storage, read> voxel_bits: VoxelBits;

// Source-material maps, indexed by material layer. These let the voxel path
// evaluate the exact triangle material instead of re-projecting a capture.
@group(2) @binding(8) var t_src_base: texture_2d_array<f32>;
@group(2) @binding(9) var t_src_mr: texture_2d_array<f32>;
@group(2) @binding(10) var t_src_normal: texture_2d_array<f32>;
@group(2) @binding(11) var t_src_ao: texture_2d_array<f32>;
@group(2) @binding(12) var t_src_emissive: texture_2d_array<f32>;
@group(2) @binding(13) var s_src: sampler;
struct MaterialParams {
    base_color: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    emissive: vec4<f32>,
    uv0: vec4<f32>, uv1: vec4<f32>, uv2: vec4<f32>, uv3: vec4<f32>, uv4: vec4<f32>,
    tf0: vec4<f32>, tf1: vec4<f32>, tf2: vec4<f32>, tf3: vec4<f32>, tf4: vec4<f32>,
};
struct MaterialTable { items: array<MaterialParams> };
@group(2) @binding(14) var<storage, read> material_table: MaterialTable;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) obj_pos: vec3<f32>,
    @location(1) @interpolate(flat) instance: u32,
};

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
    @builtin(frag_depth) depth: f32,
};

// 12-triangle unit-box vertex stream (positions in [0,1]).
fn box_vertex(i: u32) -> vec3<f32> {
    // 12 triangles, outward-wound (CCW seen from outside the box).
    var verts = array<vec3<f32>, 36>(
        // -x
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
        // +x
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
        // -y
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
        // +y
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0),
        // -z
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        // +z
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    );
    return verts[i];
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    let model = instances.items[ii].model;
    let local = box_vertex(vi % 36u);
    let pos = mix(asset.box_min.xyz, asset.box_max.xyz, local);
    let world = model * vec4<f32>(pos, 1.0);
    var out: VertexOutput;
    out.clip = camera.view_projection * world;
    out.obj_pos = pos;
    out.instance = ii;
    return out;
}

// ---------------------------------------------------------------- hull

fn face_uv(f: u32, p: vec3<f32>, c: vec3<f32>) -> vec2<f32> {
    let e = asset.extents[f];
    // The bake writes image row 0 from framebuffer top, and the engine's
    // convention puts +v there (verified by bake_consistency_gpu).
    return vec2<f32>(
        dot(p - c, asset.u_axes[f].xyz) / e.x + 0.5,
        0.5 - dot(p - c, asset.v_axes[f].xyz) / e.y,
    );
}

fn axis_vec(a: u32) -> vec3<f32> {
    return asset.views[a * 2u].xyz;
}

// Coverage of one face tile at a point (alpha of the albedo layer, texel fetch).
fn face_coverage(f: u32, uv: vec2<f32>) -> f32 {
    let res = i32(asset.params.y);
    let tc = clamp(vec2<i32>(uv * f32(res)), vec2<i32>(0), vec2<i32>(res - 1));
    return textureLoad(t_albedo, tc, i32(f), 0).a;
}

// Slab bounds of axis `a` at point p: view 2a is the + side, 2a+1 the - side.
// Returns (lo, hi, valid): p inside the slab iff valid && lo <= q <= hi.
fn axis_slab(a: u32, p: vec3<f32>, c: vec3<f32>, mip: u32) -> vec3<f32> {
    if (u32(asset.params.z) & (1u << a)) == 0u {
        return vec3<f32>(0.0, 0.0, 1.0); // axis excluded: always inside
    }
    let axis = axis_vec(a);
    let ext_n = asset.extents[a * 2u].z;
    let res = i32(asset.params.y) >> mip;
    let fp = 2u * a;
    let fm = fp + 1u;
    let uv_p = face_uv(fp, p, c);
    let uv_m = face_uv(fm, p, c);
    let t_p = clamp(vec2<i32>(uv_p * f32(res)), vec2<i32>(0), vec2<i32>(res - 1));
    let t_m = clamp(vec2<i32>(uv_m * f32(res)), vec2<i32>(0), vec2<i32>(res - 1));
    // Depth carries the sentinel (1.0 = uncovered), so the depth pair alone
    // decides coverage: an uncovered texel yields lo > hi.
    let d_p = textureLoad(t_depth, t_p, i32(fp), i32(mip)).r;
    let d_m = textureLoad(t_depth, t_m, i32(fm), i32(mip)).r;
    if d_p >= 1.0 || d_m >= 1.0 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    // The depth pyramid is already the occupancy oracle: 1.0 means no
    // captured surface in that texel. Do not additionally threshold averaged
    // albedo coverage here — partial-coverage edge texels are still real
    // surface, and rejecting them erodes each slab silhouette into a facet.
    let hi = ext_n * 0.5 - d_p * ext_n;
    let lo = -ext_n * 0.5 + d_m * ext_n;
    return vec3<f32>(lo, hi, 1.0);
}

// Conservative slab bounds for a WHOLE box-grid cell under axis `a` (diagonal
// views): the cell projects to a uv AABB around the cell centre on each face,
// and at mip min(level+1, max_mip) that AABB is under one texel wide per axis,
// so the four AABB corner texels cover every texel the cell touches. Returns
// (lo, hi, valid) like axis_slab; an entirely uncovered footprint is invalid,
// which is a safe cell rejection (MIN mips can only over-report coverage).
fn cell_axis_slab(
    a: u32,
    center: vec3<f32>,
    half: vec3<f32>,
    c: vec3<f32>,
    level: u32,
    max_mip: u32,
) -> vec3<f32> {
    let ext_n = asset.extents[a * 2u].z;
    let mip = min(level + 1u, max_mip);
    let res_m = i32(asset.params.y) >> mip;
    var d_front = 1.0;
    var d_back = 1.0;
    for (var s = 0u; s < 2u; s = s + 1u) {
        let f = 2u * a + s;
        let e = asset.extents[f];
        let uvc = face_uv(f, center, c);
        let uvh = vec2<f32>(
            dot(abs(asset.u_axes[f].xyz), half) / e.x,
            dot(abs(asset.v_axes[f].xyz), half) / e.y,
        );
        let uv_lo = uvc - uvh;
        let uv_hi = uvc + uvh;
        let t00 = clamp(vec2<i32>(uv_lo * f32(res_m)), vec2<i32>(0), vec2<i32>(res_m - 1));
        let t11 = clamp(vec2<i32>(uv_hi * f32(res_m)), vec2<i32>(0), vec2<i32>(res_m - 1));
        var d = textureLoad(t_depth, t00, i32(f), i32(mip)).r;
        if t11.x != t00.x || t11.y != t00.y {
            d = min(d, textureLoad(t_depth, vec2<i32>(t11.x, t00.y), i32(f), i32(mip)).r);
            d = min(d, textureLoad(t_depth, vec2<i32>(t00.x, t11.y), i32(f), i32(mip)).r);
            d = min(d, textureLoad(t_depth, t11, i32(f), i32(mip)).r);
        }
        if s == 0u {
            d_front = d;
        } else {
            d_back = d;
        }
    }
    if d_front >= 1.0 || d_back >= 1.0 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let hi = ext_n * 0.5 - d_front * ext_n;
    let lo = -ext_n * 0.5 + d_back * ext_n;
    return vec3<f32>(lo, hi, 1.0);
}

fn inside_hull(p: vec3<f32>, c: vec3<f32>, mip: u32) -> bool {
    let res_l = max(i32(asset.params.y) >> mip, 1);
    for (var a = 0u; a < 7u; a = a + 1u) {
        if (u32(asset.params.z) & (1u << a)) == 0u {
            continue; // excluded axis carries no slab constraint
        }
        let slab = axis_slab(a, p, c, mip);
        if slab.z < 0.5 {
            return false;
        }
        let q = dot(p - c, axis_vec(a));
        let tol = asset.extents[a * 2u].z * (0.5 / f32(res_l));
        if q < slab.x - tol || q > slab.y + tol {
            return false;
        }
    }
    return true;
}

// Slab-intersect the ray with the object-space box.
fn ray_box(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    var lo = -3.4e38;
    var hi = 3.4e38;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if abs(rd[a]) < 1e-9 {
            // Parallel axis: inside the box slab → unconstrained; outside → miss.
            if ro[a] < asset.box_min[a] || ro[a] > asset.box_max[a] {
                return vec2<f32>(1.0, -1.0);
            }
        } else {
            let t0 = (asset.box_min[a] - ro[a]) / rd[a];
            let t1 = (asset.box_max[a] - ro[a]) / rd[a];
            lo = max(lo, min(t0, t1));
            hi = min(hi, max(t0, t1));
        }
    }
    return vec2<f32>(lo, hi);
}

fn voxel_cell(v: vec3<i32>, res: u32) -> vec3<u32> {
    if v.x < 0 || v.y < 0 || v.z < 0
        || v.x >= i32(res) || v.y >= i32(res) || v.z >= i32(res)
    {
        return vec3<u32>(0u, 0u, 0u);
    }
    let idx = u32(v.z * i32(res) * i32(res) + v.y * i32(res) + v.x) * 3u;
    return vec3<u32>(
        voxel_bits.bits[idx],
        voxel_bits.bits[idx + 1u],
        voxel_bits.bits[idx + 2u],
    );
}

struct VoxelHit {
    p: vec3<f32>,
    attr: u32,
};

// DDA through the triangle-surface voxel mask. This returns the ray's first
// occupied surface voxel, a much tighter geometric seed than the visual hull.
fn voxel_surface_hit(ro: vec3<f32>, rd: vec3<f32>) -> VoxelHit {
    let res = u32(asset.screen.w + 0.5);
    if res == 0u {
        return VoxelHit(vec3<f32>(0.0), 0u);
    }
    let tb = ray_box(ro, rd);
    if tb.y <= max(tb.x, 0.0) {
        return VoxelHit(vec3<f32>(0.0), 0u);
    }
    let res_i = i32(res);
    let res_f = f32(res);
    let gmin = asset.box_min.xyz;
    let cell = (asset.box_max.xyz - gmin) / res_f;
    var t = max(tb.x, 0.0);
    var v = vec3<i32>(floor((ro + rd * t - gmin) / cell));
    v = clamp(v, vec3<i32>(0), vec3<i32>(res_i - 1));
    var step = vec3<i32>(0);
    var t_max = vec3<f32>(3.4e38);
    var t_delta = vec3<f32>(3.4e38);
    for (var a = 0u; a < 3u; a = a + 1u) {
        if rd[a] > 1e-9 {
            step[a] = 1;
            let boundary = gmin[a] + f32(v[a] + 1) * cell[a];
            t_max[a] = (boundary - ro[a]) / rd[a];
            t_delta[a] = cell[a] / rd[a];
        } else if rd[a] < -1e-9 {
            step[a] = -1;
            let boundary = gmin[a] + f32(v[a]) * cell[a];
            t_max[a] = (boundary - ro[a]) / rd[a];
            t_delta[a] = -cell[a] / rd[a];
        }
    }
    var entry_t = t;
    for (var i = 0u; i < res * 3u + 4u; i = i + 1u) {
        let cell_data = voxel_cell(v, res);
        if cell_data.x != 0u {
            let center = gmin + (vec3<f32>(v) + vec3<f32>(0.5)) * cell;
            let off = vec3<f32>(
                f32(i32((cell_data.x >> 16u) & 0xffu) - 128),
                f32(i32((cell_data.x >> 8u) & 0xffu) - 128),
                f32(i32(cell_data.x & 0xffu) - 128),
            ) / 255.0;
            let normal = vec3<f32>(
                f32(i32((cell_data.y >> 16u) & 0xffu) - 128),
                f32(i32((cell_data.y >> 8u) & 0xffu) - 128),
                f32(i32(cell_data.y & 0xffu) - 128),
            ) / 127.0;
            let plane_p = center + off * cell;
            let denom = dot(rd, normal);
            if abs(denom) > 1e-4 {
                let t_hit = dot(plane_p - ro, normal) / denom;
                let p = ro + rd * t_hit;
                let q = (p - center) / cell;
                // Only accept a local plane hit inside this voxel. Otherwise
                // the packed surface belongs to a different ray segment.
                if t_hit >= tb.x && t_hit <= tb.y
                    && all(q >= vec3<f32>(-0.75)) && all(q <= vec3<f32>(0.75))
                {
                    return VoxelHit(p, cell_data.z);
                }
            }
        }
        if t_max.x <= t_max.y && t_max.x <= t_max.z {
            v.x += step.x;
            entry_t = t_max.x;
            t_max.x += t_delta.x;
        } else if t_max.y <= t_max.z {
            v.y += step.y;
            entry_t = t_max.y;
            t_max.y += t_delta.y;
        } else {
            v.z += step.z;
            entry_t = t_max.z;
            t_max.z += t_delta.z;
        }
        if entry_t > tb.y
            || v.x < 0 || v.y < 0 || v.z < 0
            || v.x >= res_i || v.y >= res_i || v.z >= res_i
        {
            break;
        }
    }
    return VoxelHit(vec3<f32>(0.0), 0u);
}

// -------- exact source surface records -------------------------------------

const TRIANGLE_WORDS: u32 = 48u;

fn tri_f32(base: u32, index: u32) -> f32 {
    return bitcast<f32>(voxel_bits.bits[base + index]);
}

fn tri_vec2(base: u32, index: u32) -> vec2<f32> {
    return vec2<f32>(tri_f32(base, index), tri_f32(base, index + 1u));
}

fn tri_vec3(base: u32, index: u32) -> vec3<f32> {
    return vec3<f32>(
        tri_f32(base, index),
        tri_f32(base, index + 1u),
        tri_f32(base, index + 2u),
    );
}

struct SourceSurface {
    valid: bool,
    p: vec3<f32>,
    n: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    tangent: vec4<f32>,
    material: u32,
    // Triangle data needed to turn a neighbouring screen ray into a source UV
    // derivative. Keeping the vertices here avoids depending on derivative
    // instructions across a reconstructed hull surface.
    v0: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
    uv00: vec2<f32>,
    uv01: vec2<f32>,
    uv02: vec2<f32>,
    uv10: vec2<f32>,
    uv11: vec2<f32>,
    uv12: vec2<f32>,
    geom_n: vec3<f32>,
};

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn empty_source_surface(p: vec3<f32>) -> SourceSurface {
    var out: SourceSurface;
    out.valid = false;
    out.p = p;
    out.n = vec3<f32>(0.0, 0.0, 1.0);
    out.uv0 = vec2<f32>(0.0);
    out.uv1 = vec2<f32>(0.0);
    out.tangent = vec4<f32>(0.0);
    out.material = 0u;
    out.v0 = vec3<f32>(0.0);
    out.e1 = vec3<f32>(0.0);
    out.e2 = vec3<f32>(0.0);
    out.uv00 = vec2<f32>(0.0);
    out.uv01 = vec2<f32>(0.0);
    out.uv02 = vec2<f32>(0.0);
    out.uv10 = vec2<f32>(0.0);
    out.uv11 = vec2<f32>(0.0);
    out.uv12 = vec2<f32>(0.0);
    out.geom_n = vec3<f32>(0.0, 0.0, 1.0);
    return out;
}

// Intersect the pixel ray with the source triangles referenced by the
// first occupied voxel (nearest of the cell's whole list). Barycentrics
// recover exact mesh UVs instead of choosing a material through a
// reconstructed capture view.
fn source_surface(voxel: VoxelHit, ro: vec3<f32>, rd: vec3<f32>) -> SourceSurface {
    var out = empty_source_surface(voxel.p);
    if voxel.attr == 0u || bitcast<u32>(asset.material.x) == 0u {
        return out;
    }
    let voxel_res = u32(max(asset.screen.w, 1.0));
    let dense_words = voxel_res * voxel_res * voxel_res * 3u;
    let ref_base = dense_words + voxel.attr - 1u;
    let ref_count = voxel_bits.bits[ref_base];
    if ref_count == 0u {
        return out;
    }
    let cell_limit = max(
        asset.box_max.x - asset.box_min.x,
        max(asset.box_max.y - asset.box_min.y, asset.box_max.z - asset.box_min.z),
    ) / f32(voxel_res) * 1.5;
    var best_base = 0u;
    var best_t = 3.4e38;
    var best_bary = vec3<f32>(0.0);
    for (var k = 0u; k < ref_count; k = k + 1u) {
        let triangle = voxel_bits.bits[ref_base + 1u + k];
        if triangle == 0u {
            continue;
        }
        let base = bitcast<u32>(asset.material.x) + (triangle - 1u) * TRIANGLE_WORDS;
        let v0 = tri_vec3(base, 0u);
        let e1 = tri_vec3(base, 3u);
        let e2 = tri_vec3(base, 6u);
        let geom_n = cross(e1, e2);
        let denom = dot(rd, geom_n);
        if abs(denom) < 1e-9 {
            continue;
        }
        let t_hit = dot(v0 - ro, geom_n) / denom;
        if t_hit <= 0.0 || t_hit >= best_t {
            continue;
        }
        let p = ro + rd * t_hit;
        if distance(p, voxel.p) > cell_limit {
            continue;
        }
        let q = p - v0;
        let d00 = dot(e1, e1);
        let d01 = dot(e1, e2);
        let d11 = dot(e2, e2);
        let inv_den = 1.0 / max(d00 * d11 - d01 * d01, 1e-12);
        let u = (d11 * dot(q, e1) - d01 * dot(q, e2)) * inv_den;
        let v = (d00 * dot(q, e2) - d01 * dot(q, e1)) * inv_den;
        let w = 1.0 - u - v;
        if u < -0.025 || v < -0.025 || w < -0.025 {
            continue;
        }
        best_base = base;
        best_t = t_hit;
        best_bary = vec3<f32>(w, u, v);
        out.p = p;
    }
    if best_base == 0u {
        return out;
    }
    out = unpack_source(best_base, best_bary, rd, out.p);
    return out;
}

// Unpack the winning triangle record into a SourceSurface. Record word 42 is
// the material INDEX stored as a raw u32 — `tri_f32` would bitcast it to a
// denormal float and every material would read as 0.
fn unpack_source(base: u32, bary: vec3<f32>, rd: vec3<f32>, p: vec3<f32>) -> SourceSurface {
    var out = empty_source_surface(p);
    let w = bary.x;
    let u = bary.y;
    let v = bary.z;
    out.n = normalize(
        tri_vec3(base, 9u) * w + tri_vec3(base, 12u) * u + tri_vec3(base, 15u) * v,
    );
    if dot(out.n, rd) > 0.0 {
        out.n = -out.n;
    }
    out.uv00 = tri_vec2(base, 18u);
    out.uv01 = tri_vec2(base, 20u);
    out.uv02 = tri_vec2(base, 22u);
    out.uv10 = tri_vec2(base, 24u);
    out.uv11 = tri_vec2(base, 26u);
    out.uv12 = tri_vec2(base, 28u);
    out.uv0 = out.uv00 * w + out.uv01 * u + out.uv02 * v;
    out.uv1 = out.uv10 * w + out.uv11 * u + out.uv12 * v;
    out.v0 = tri_vec3(base, 0u);
    out.e1 = tri_vec3(base, 3u);
    out.e2 = tri_vec3(base, 6u);
    out.geom_n = cross(out.e1, out.e2);
    let t0 = vec4<f32>(tri_vec3(base, 30u), tri_f32(base, 33u));
    let t1 = vec4<f32>(tri_vec3(base, 34u), tri_f32(base, 37u));
    let t2 = vec4<f32>(tri_vec3(base, 38u), tri_f32(base, 41u));
    out.tangent = t0 * w + t1 * u + t2 * v;
    out.material = voxel_bits.bits[base + 42u];
    out.valid = true;
    return out;
}

struct TriHit {
    t: f32,
    base: u32,
    bary: vec3<f32>,
};

// Intersect the ray with every triangle referenced by one voxel cell, in the
// t range [t_lo, t_hi] covering exactly that cell's traversal interval. No
// seed-plane distance gate — cell bounds are the only acceptance region.
fn cell_triangle_hit(
    ref_base: u32,
    ro: vec3<f32>,
    rd: vec3<f32>,
    t_lo: f32,
    t_hi: f32,
) -> TriHit {
    var best = TriHit(3.4e38, 0u, vec3<f32>(0.0));
    let ref_count = voxel_bits.bits[ref_base];
    for (var k = 0u; k < ref_count; k = k + 1u) {
        let triangle = voxel_bits.bits[ref_base + 1u + k];
        if triangle == 0u {
            continue;
        }
        let base = bitcast<u32>(asset.material.x) + (triangle - 1u) * TRIANGLE_WORDS;
        let v0 = tri_vec3(base, 0u);
        let e1 = tri_vec3(base, 3u);
        let e2 = tri_vec3(base, 6u);
        // Scale-free intersection: everything divides through |cross|^2 so
        // tiny and huge triangles behave identically (rd is normalized).
        let geom_n = cross(e1, e2);
        let n2 = dot(geom_n, geom_n);
        if n2 <= 0.0 {
            continue; // degenerate record
        }
        let denom = dot(rd, geom_n);
        if abs(denom) <= 1e-7 * sqrt(n2) {
            continue; // parallel to the triangle plane
        }
        let t_hit = dot(v0 - ro, geom_n) / denom;
        if t_hit < t_lo || t_hit > t_hi || t_hit >= best.t {
            continue;
        }
        let q = ro + rd * t_hit - v0;
        let u = dot(cross(q, e2), geom_n) / n2;
        let v = dot(cross(e1, q), geom_n) / n2;
        let w = 1.0 - u - v;
        if u < -1e-5 || v < -1e-5 || w < -1e-5 {
            continue;
        }
        best = TriHit(t_hit, base, vec3<f32>(w, u, v));
    }
    return best;
}

// DDA through the triangle-surface voxel grid, intersecting the REAL triangle
// references in each occupied cell. This is the authoritative geometry path
// for opaque assets: the first real ray-triangle hit along the ray wins —
// no hull approximation, no seed-plane heuristic.
fn source_ray_hit(ro: vec3<f32>, rd: vec3<f32>) -> SourceSurface {
    var out = empty_source_surface(vec3<f32>(0.0));
    let res = u32(asset.screen.w + 0.5);
    if res == 0u {
        return out;
    }
    let tb = ray_box(ro, rd);
    if tb.y <= max(tb.x, 0.0) {
        return out;
    }
    let res_i = i32(res);
    let res_f = f32(res);
    let gmin = asset.box_min.xyz;
    let cell = (asset.box_max.xyz - gmin) / res_f;
    let cell_min = min(cell.x, min(cell.y, cell.z));
    let dense_words = res * res * res * 3u;
    var t = max(tb.x, 0.0);
    var v = vec3<i32>(floor((ro + rd * t - gmin) / cell));
    v = clamp(v, vec3<i32>(0), vec3<i32>(res_i - 1));
    var step = vec3<i32>(0);
    var t_max = vec3<f32>(3.4e38);
    var t_delta = vec3<f32>(3.4e38);
    for (var a = 0u; a < 3u; a = a + 1u) {
        if rd[a] > 1e-9 {
            step[a] = 1;
            let boundary = gmin[a] + f32(v[a] + 1) * cell[a];
            t_max[a] = (boundary - ro[a]) / rd[a];
            t_delta[a] = cell[a] / rd[a];
        } else if rd[a] < -1e-9 {
            step[a] = -1;
            let boundary = gmin[a] + f32(v[a]) * cell[a];
            t_max[a] = (boundary - ro[a]) / rd[a];
            t_delta[a] = -cell[a] / rd[a];
        }
    }
    var entry_t = t;
    for (var i = 0u; i < res * 3u + 4u; i = i + 1u) {
        let cell_data = voxel_cell(v, res);
        if cell_data.x != 0u && cell_data.z != 0u {
            // Occupied cell with triangle references: intersect the WHOLE
            // list, keeping only hits inside this cell's traversal interval
            // (ties on a shared boundary resolve to the first visit — the
            // nearest t along the ray).
            let cell_exit = min(min(t_max.x, t_max.y), min(t_max.z, tb.y));
            let tol = cell_min * 1e-3;
            let hit = cell_triangle_hit(
                dense_words + cell_data.z - 1u,
                ro,
                rd,
                max(entry_t, tb.x) - tol,
                cell_exit + tol,
            );
            if hit.base != 0u {
                return unpack_source(hit.base, hit.bary, rd, ro + rd * hit.t);
            }
            // Occupied but no real triangle crosses this ray segment:
            // continue into the next cell rather than fabricating a hit.
        }
        if t_max.x <= t_max.y && t_max.x <= t_max.z {
            v.x += step.x;
            entry_t = t_max.x;
            t_max.x += t_delta.x;
        } else if t_max.y <= t_max.z {
            v.y += step.y;
            entry_t = t_max.y;
            t_max.y += t_delta.y;
        } else {
            v.z += step.z;
            entry_t = t_max.z;
            t_max.z += t_delta.z;
        }
        if entry_t > tb.y
            || v.x < 0 || v.y < 0 || v.z < 0
            || v.x >= res_i || v.y >= res_i || v.z >= res_i
        {
            break;
        }
    }
    return out;
}

/// Scan the leaf interval [n0,n1] on the 16-point lattice used by the
/// original probe loop, but evaluate each axis slab ONCE per constant-texel
/// segment instead of once per lattice point: texel indices only change at
/// uv-grid boundaries along rd, so the next boundary across all valid faces
/// bounds how far the sampled slabs stay valid. Returns the same earliest
/// lattice candidate the dense scan would accept; on boundary roundoff the
/// inside_hull verify is the authority — never fabricate.
fn leaf_hull_hit(
    ro: vec3<f32>,
    rd: vec3<f32>,
    c: vec3<f32>,
    n0: f32,
    n1: f32,
    mip: u32,
    missed: ptr<function, bool>,
) -> vec3<f32> {
    *missed = false;
    let res_l = i32(asset.params.y) >> mip;
    let res_f = f32(res_l);
    let step = (n1 - n0) / 16.0;
    let rel_ro = ro - c;
    var k = 0u;
    for (var seg = 0u; seg < 16u; seg = seg + 1u) {
        if k > 15u {
            break;
        }
        let t_k = n0 + (f32(k) + 0.5) * step;
        let pk = ro + rd * t_k;
        // Slabs at this candidate + the ray's valid t-range + the next uv
        // boundary on every valid face pair.
        var v_lo = n0;
        var v_hi = n1;
        var next_b = n1;
        var ok = true;
        // Next uv-grid boundary on EVERY valid face pair (must run even when
        // a slab is invalid — the invalid region ends at a texel boundary).
        for (var a = 0u; a < 7u; a = a + 1u) {
            if (u32(asset.params.z) & (1u << a)) == 0u {
                continue;
            }
            for (var s = 0u; s < 2u; s = s + 1u) {
                let f = 2u * a + s;
                let e = asset.extents[f];
                let uvc = face_uv(f, pk, c) * res_f;
                let slopes = vec2<f32>(
                    dot(rd, asset.u_axes[f].xyz) / e.x * res_f,
                    -dot(rd, asset.v_axes[f].xyz) / e.y * res_f,
                );
                for (var comp = 0u; comp < 2u; comp = comp + 1u) {
                    let coord = select(uvc.x, uvc.y, comp == 1u);
                    let slope = select(slopes.x, slopes.y, comp == 1u);
                    if abs(slope) < 1e-9 {
                        continue;
                    }
                    // Next grid line strictly in the travel direction. For
                    // negative slope use floor(coord): at an exact integer
                    // coord the boundary is the point itself (dt=0), so the
                    // candidate is evaluated once and the next lattice point
                    // is sampled in the new texel.
                    let nxt = select(
                        floor(coord),
                        floor(coord) + 1.0,
                        slope > 0.0,
                    );
                    let dt = (nxt - coord) / slope;
                    if dt >= 0.0 {
                        next_b = min(next_b, t_k + dt);
                    }
                }
            }
        }
        for (var a = 0u; a < 7u; a = a + 1u) {
            if (u32(asset.params.z) & (1u << a)) == 0u {
                continue;
            }
            let slab = axis_slab(a, pk, c, mip);
            if slab.z < 0.5 {
                ok = false;
                break;
            }
            let dir = axis_vec(a);
            let ext_n = asset.extents[a * 2u].z;
            let tol = ext_n * (0.5 / res_f);
            let lo = slab.x - tol;
            let hi = slab.y + tol;
            let q0 = dot(rel_ro, dir);
            let qa = dot(rd, dir);
            if abs(qa) < 1e-9 {
                if q0 < lo || q0 > hi {
                    ok = false;
                    break;
                }
            } else {
                let ta = (lo - q0) / qa;
                let tb2 = (hi - q0) / qa;
                v_lo = max(v_lo, min(ta, tb2));
                v_hi = min(v_hi, max(ta, tb2));
                if v_lo > v_hi {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            // First lattice point inside the valid interval and this segment.
            let first = max(
                i32(k),
                i32(ceil((v_lo - n0) / step - 0.5)),
            );
            if first >= 0 && first <= 15 {
                let t_f = n0 + (f32(first) + 0.5) * step;
                if t_f <= next_b && t_f <= v_hi {
                    let h = ro + rd * t_f;
                    // Roundoff guard: the dense check is the authority.
                    if inside_hull(h, c, mip) {
                        return h;
                    }
                    k = u32(first) + 1u;
                    continue;
                }
            }
        }
        // No accepted candidate in this segment: jump to the first lattice
        // point at or after the next texel boundary (at least k+1).
        let jump = i32(ceil((next_b - n0) / step - 0.5));
        k = u32(max(i32(k) + 1, jump));
    }
    *missed = true;
    return vec3<f32>(0.0);
}

// `missed` tells the caller to discard (WGSL has no NaN literals).
/// Hierarchical traversal: depth MIN-mips bound each cell conservatively
/// (a coarse rejection is safe; a coarse acceptance descends). Every
/// returned point lies inside all valid mip-0 slabs — no fabricated hits.
fn hull_hit(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, debug_shell: bool, hull_mip: u32, missed: ptr<function, bool>) -> vec3<f32> {
    *missed = false;
    if debug_shell {
        // Intentional "decal" comparison: shade the shell surface itself.
        let t = ray_box(ro, rd);
        return ro + rd * t.x;
    }
    let tb = ray_box(ro, rd);
    if tb.y <= max(tb.x, 0.0) {
        *missed = true;
        return vec3<f32>(0.0);
    }
    let res_f = asset.params.y;
    let res0 = u32(res_f);
    let max_mip = u32(log2(res_f) + 0.5);
    // Face-uv grid origin: uv=0 sits at c - ext/2 along each axis.
    let gmin = vec3<f32>(
        c.x - asset.extents[0].z * 0.5,
        c.y - asset.extents[2].z * 0.5,
        c.z - asset.extents[4].z * 0.5,
    );
    let exts = vec3<f32>(
        asset.extents[0].z,
        asset.extents[2].z,
        asset.extents[4].z,
    );
    let rel_ro = ro - c; // slab coords are relative to c

    // Traversal stack indexed by mip level (res <= 2048 -> <= 12 levels).
    var cursor: array<f32, 12>;
    var ends: array<f32, 12>;
    var level = max_mip;
    cursor[level] = max(tb.x, 0.0);
    ends[level] = tb.y;
    var budget = 6u * res0 + 64u;

    for (var step_i = 0u; step_i < 65536u; step_i += 1u) {
        if budget == 0u {
            // Never fabricate a hit: budget exhaustion is a clean miss.
            *missed = true;
            return vec3<f32>(0.0);
        }
        budget -= 1u;
        if cursor[level] >= ends[level] {
            if level == max_mip {
                *missed = true;
                return vec3<f32>(0.0);
            }
            // Child interval exhausted; the parent's cursor already sits at
            // the consumed cell's exit — just pop back up.
            level += 1u;
            continue;
        }
        let res_l = res0 >> level;
        let cell = exts / f32(res_l);
        // Cell index in GRID coordinates with a directional epsilon — the
        // bias picks which cell contains the cursor, it never perturbs the
        // intersection times. Parallel axes contribute no boundary (the
        // ray-box test already proved the origin is inside the box).
        let pgrid = (ro + rd * cursor[level] - gmin) / cell + sign(rd) * vec3<f32>(1e-5);
        var cell_exit = ends[level];
        for (var a = 0u; a < 3u; a = a + 1u) {
            if abs(rd[a]) < 1e-9 {
                // Parallel axis: ray never leaves the cell row on this axis.
                if ro[a] < gmin[a] || ro[a] > gmin[a] + exts[a] {
                    *missed = true; // outside the box slab on this axis
                    return vec3<f32>(0.0);
                }
                continue;
            }
            let gi = clamp(i32(floor(pgrid[a])), 0, i32(res_l) - 1);
            let side = select(gi, gi + 1, rd[a] > 0.0);
            let t_b = (gmin[a] + f32(side) * cell[a] - ro[a]) / rd[a];
            cell_exit = min(cell_exit, t_b);
        }
        if cell_exit <= cursor[level] {
            // Degenerate cell on a boundary: step a small grid fraction.
            cursor[level] = cursor[level] + min(cell.x, min(cell.y, cell.z)) * 1e-3;
            continue;
        }
        // Slab values are constant in the cell for the box-aligned axes;
        // sample the midpoint. Diagonal axes use a conservative whole-cell
        // depth envelope (see cell_axis_slab) so coarse empty space is still
        // rejected early; the leaf scan below remains authoritative.
        let pm = ro + rd * ((cursor[level] + cell_exit) * 0.5);
        let gi3 = clamp(
            vec3<i32>(floor(pgrid)),
            vec3<i32>(0),
            vec3<i32>(i32(res_l) - 1),
        );
        let cell_center = gmin + (vec3<f32>(gi3) + vec3<f32>(0.5)) * cell;
        let cell_half = cell * 0.5;
        var n0 = cursor[level];
        var n1 = cell_exit;
        var ok = true;
        for (var a = 0u; a < 7u; a = a + 1u) {
            if (u32(asset.params.z) & (1u << a)) == 0u {
                continue; // excluded axis: no slab constraint
            }
            var slab: vec3<f32>;
            if a < 3u {
                slab = axis_slab(a, pm, c, level);
            } else if level > hull_mip {
                // Diagonal slab, conservative envelope over the cell.
                slab = cell_axis_slab(a, cell_center, cell_half, c, level, max_mip);
            } else {
                continue; // leaf cell: the inside_hull interval scan covers it
            }
            if slab.z < 0.5 {
                ok = false;
                break;
            }
            let q0 = dot(rel_ro, axis_vec(a));
            let qa = dot(rd, axis_vec(a));
            let tol = asset.extents[a * 2u].z * (0.5 / f32(res_l));
            let lo = slab.x - tol;
            let hi = slab.y + tol;
            if abs(qa) < 1e-9 {
                if q0 < lo || q0 > hi {
                    ok = false;
                    break;
                }
            } else {
                let t_a = (lo - q0) / qa;
                let t_b = (hi - q0) / qa;
                n0 = max(n0, min(t_a, t_b));
                n1 = min(n1, max(t_a, t_b));
                if n0 > n1 {
                    ok = false;
                    break;
                }
            }
        }
        // Slab intervals are only valid for the cell they were sampled in.
        // Clamp them to this cell before descending so a coarse accept cannot
        // carry traversal into an unexamined neighbor.
        n0 = max(n0, cursor[level]);
        n1 = min(n1, cell_exit);
        // Parent consumes this cell regardless of the outcome.
        cursor[level] = cell_exit;
        if !ok || n0 >= n1 {
            continue;
        }
        if level == hull_mip {
            // Constant-texel segmented scan over the same 16-point lattice —
            // identical first accepted candidate as the dense inside_hull
            // scan it replaces (see leaf_hull_hit).
            var leaf_miss = false;
            let h = leaf_hull_hit(ro, rd, c, n0, n1, hull_mip, &leaf_miss);
            if !leaf_miss {
                return h;
            }
            continue;
        }
        level -= 1u;
        cursor[level] = n0;
        ends[level] = n1;
    }
    *missed = true;
    return vec3<f32>(0.0);
}
// ---------------------------------------------------------------- shading

fn screen_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * 0.5 + vec2<f32>(0.5);
}

struct FaceData {
    albedo: vec4<f32>,
    normal_ao: vec4<f32>,
    mr: vec4<f32>,
    emissive: vec4<f32>,
};

fn sample_face(f: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> FaceData {
    var s: FaceData;
    s.albedo = textureSampleGrad(t_albedo, s_tile, uv, i32(f), ddx, ddy);
    s.normal_ao = textureSampleGrad(t_normal_ao, s_tile, uv, i32(f), ddx, ddy);
    s.mr = textureSampleGrad(t_mr, s_tile, uv, i32(f), ddx, ddy);
    s.emissive = textureSampleGrad(t_emissive, s_tile, uv, i32(f), ddx, ddy);
    return s;
}

fn sample_face_level(f: u32, uv: vec2<f32>, mip: f32) -> FaceData {
    var s: FaceData;
    s.albedo = textureSampleLevel(t_albedo, s_tile, uv, i32(f), mip);
    s.normal_ao = textureSampleLevel(t_normal_ao, s_tile, uv, i32(f), mip);
    s.mr = textureSampleLevel(t_mr, s_tile, uv, i32(f), mip);
    s.emissive = textureSampleLevel(t_emissive, s_tile, uv, i32(f), mip);
    return s;
}

// Footprint of a neighbouring pixel in one capture's uv space. Instead of the
// raw dpdx of the hull point (which is discontinuous across surface choices),
// advance the screen-space ray derivative across the tangent plane implied by
// the baked normal. This gives correct anisotropic filtering on oblique views.
fn tangent_uv_grad(
    f: u32,
    p: vec3<f32>,
    ro: vec3<f32>,
    rd: vec3<f32>,
    n_obj: vec3<f32>,
    fallback: vec2<f32>,
) -> mat2x2<f32> {
    let e = asset.extents[f];
    let u = asset.u_axes[f].xyz;
    let v = asset.v_axes[f].xyz;
    let rdx = dpdx(rd);
    let rdy = dpdy(rd);
    let t = max(dot(p - ro, rd), 1e-4);
    let ox = rdx * t;
    let oy = rdy * t;
    let denom = dot(rd, n_obj);
    if abs(denom) < 0.15 {
        return mat2x2<f32>(vec2<f32>(fallback.x, 0.0), vec2<f32>(0.0, fallback.y));
    }
    let px_t = ox - rd * (dot(ox, n_obj) / denom);
    let py_t = oy - rd * (dot(oy, n_obj) / denom);
    var gx = vec2<f32>(dot(px_t, u) / e.x, -dot(px_t, v) / e.y);
    var gy = vec2<f32>(dot(py_t, u) / e.x, -dot(py_t, v) / e.y);
    let lim = max(fallback * 8.0, vec2<f32>(4.0 / asset.params.y));
    gx = mix(vec2<f32>(fallback.x, 0.0), clamp(gx, -lim, lim), 0.50);
    gy = mix(vec2<f32>(0.0, fallback.y), clamp(gy, -lim, lim), 0.50);
    return mat2x2<f32>(gx, gy);
}

struct MaterialSample {
    p: vec3<f32>,
    lit: vec3<f32>,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    rough: f32,
    metal: f32,
    ao: f32,
    emissive: vec3<f32>,
    coverage: f32,
};

fn material_uv_select(m: MaterialParams, index: u32) -> vec4<f32> {
    if index == 0u { return m.uv0; }
    if index == 1u { return m.uv1; }
    if index == 2u { return m.uv2; }
    if index == 3u { return m.uv3; }
    return m.uv4;
}

fn material_tf_select(m: MaterialParams, index: u32) -> vec4<f32> {
    if index == 0u { return m.tf0; }
    if index == 1u { return m.tf1; }
    if index == 2u { return m.tf2; }
    if index == 3u { return m.tf3; }
    return m.tf4;
}

fn wrap_coord(x: f32, mode: u32) -> f32 {
    if mode == 1u {
        return clamp(x, 0.0, 1.0);
    }
    if mode == 2u {
        return abs(fract(x * 0.5) * 2.0 - 1.0);
    }
    return fract(x);
}

fn material_map_uv(m: MaterialParams, index: u32, uv0: vec2<f32>, uv1: vec2<f32>) -> vec4<f32> {
    let uvm = material_uv_select(m, index);
    let tf = material_tf_select(m, index);
    let base = select(uv0, uv1, tf.z > 0.5);
    let scaled = base * uvm.xy;
    // Same row-vector math as model_pbr.wgsl: [cos -sin; sin cos] * uv.
    let uv = vec2<f32>(
        tf.x * scaled.x - tf.y * scaled.y,
        tf.y * scaled.x + tf.x * scaled.y,
    ) + uvm.zw;
    let packed = u32(tf.w + 0.5);
    let wrapped = vec2<f32>(
        wrap_coord(uv.x, packed & 3u),
        wrap_coord(uv.y, (packed >> 2u) & 3u),
    );
    if (packed & 16u) != 0u {
        let res = max(asset.material.y, 1.0);
        let nearest = (floor(wrapped * res) + vec2<f32>(0.5)) / res;
        return vec4<f32>(nearest.x, nearest.y, tf.w, 0.0);
    }
    return vec4<f32>(wrapped.x, wrapped.y, tf.w, 0.0);
}

fn source_tangent_frame(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>, t: vec4<f32>) -> mat3x3<f32> {
    if dot(t.xyz, t.xyz) > 0.5 {
        let tangent = normalize(t.xyz);
        return mat3x3<f32>(tangent, normalize(sign(t.w) * cross(n, tangent)), n);
    }
    let duv_dx = dpdx(uv);
    let duv_dy = dpdy(uv);
    let dp_dx = dpdx(p);
    let dp_dy = dpdy(p);
    let tangent = (dp_dx * duv_dy.y - dp_dy * duv_dx.y)
        / max(duv_dx.x * duv_dy.y - duv_dy.x * duv_dx.y, 1e-7);
    let t3 = normalize(tangent - n * dot(n, tangent));
    return mat3x3<f32>(t3, normalize(cross(n, t3)), n);
}

// Evaluate the original material at the exact triangle hit. This removes the
// far-field residual caused by selecting a coherent but different surface
// through the projected atlas views.
fn source_uv_delta(surface: SourceSurface, index: u32, dp: vec3<f32>) -> vec2<f32> {
    let d00 = dot(surface.e1, surface.e1);
    let d01 = dot(surface.e1, surface.e2);
    let d11 = dot(surface.e2, surface.e2);
    let inv_den = 1.0 / max(d00 * d11 - d01 * d01, 1e-12);
    let du = (d11 * dot(dp, surface.e1) - d01 * dot(dp, surface.e2)) * inv_den;
    let dv = (d00 * dot(dp, surface.e2) - d01 * dot(dp, surface.e1)) * inv_den;
    if index == 1u {
        let a = surface.uv10;
        let b = surface.uv11;
        let c = surface.uv12;
        return a * (-du - dv) + b * du + c * dv;
    }
    let a = surface.uv00;
    let b = surface.uv01;
    let c = surface.uv02;
    return a * (-du - dv) + b * du + c * dv;
}

// Exact UV derivatives for the material array. Neighbour rays are intersected
// with the selected source triangle's plane, then transformed through the same
// KHR_texture_transform as the sampled UV. This gives the texture unit the
// same footprint information the mesh rasterizer has.
fn source_map_grad(
    surface: SourceSurface,
    index: u32,
    ro: vec3<f32>,
    rd: vec3<f32>,
    m: MaterialParams,
) -> mat2x2<f32> {
    let tf = material_tf_select(m, index);
    let transform = material_uv_select(m, index);
    let plane_n = normalize(surface.geom_n);
    let plane_q = surface.v0 - ro;
    let fallback = vec2<f32>(1.0 / max(asset.material.y, 1.0), 0.0);
    var cols: array<vec2<f32>, 2>;
    for (var axis = 0u; axis < 2u; axis = axis + 1u) {
        let ray = select(rd + dpdy(rd), rd + dpdx(rd), axis == 0u);
        let denom = dot(ray, plane_n);
        if abs(denom) < 1e-5 {
            cols[axis] = select(fallback.yx, fallback, axis == 0u);
            continue;
        }
        let q = ro + ray * (dot(plane_q, plane_n) / denom) - surface.p;
        let base = source_uv_delta(surface, u32(tf.z + 0.5), q) * transform.xy;
        cols[axis] = vec2<f32>(
            tf.x * base.x - tf.y * base.y,
            tf.y * base.x + tf.x * base.y,
        );
    }
    return mat2x2<f32>(cols[0], cols[1]);
}

// Conservative scalar LOD: the largest screen-axis UV footprint. A scalar is
// preferable here because the voxel can select a different source triangle in
// neighbouring pixels; anisotropic hardware footprints would make that seam
// noisier than the mesh itself.
fn source_map_lod(
    surface: SourceSurface,
    index: u32,
    ro: vec3<f32>,
    rd: vec3<f32>,
    m: MaterialParams,
) -> f32 {
    let grad = source_map_grad(surface, index, ro, rd, m);
    let texels = max(
        max(length(grad[0]), length(grad[1])) * max(asset.material.y, 1.0),
        1.0,
    );
    return log2(texels);
}

// Evaluate the original material at the exact triangle hit. This removes the
// far-field residual caused by selecting a coherent but different surface
// through the projected atlas views.
fn source_material(
    surface: SourceSurface,
    anchor: vec3<f32>,
    ro: vec3<f32>,
    rd: vec3<f32>,
    model: mat4x4<f32>,
) -> MaterialSample {
    var out: MaterialSample;
    out.p = anchor;
    out.lit = vec3<f32>(0.0);
    out.albedo = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.rough = 0.8;
    out.metal = 0.0;
    out.ao = 1.0;
    out.emissive = vec3<f32>(0.0);
    out.coverage = 0.0;
    if !surface.valid {
        return out;
    }
    let material_count = max(i32(asset.material.z + 0.5), 1);
    let layer = clamp(i32(surface.material), 0, material_count - 1);
    let m = material_table.items[layer];

    let base_t = material_map_uv(m, 0u, surface.uv0, surface.uv1);
    let base = textureSampleLevel(
        t_src_base,
        s_src,
        base_t.xy,
        layer,
        source_map_lod(surface, 0u, ro, rd, m),
    );
    var coverage = base.a * m.base_color.a;
    if m.params.w > 0.5 && m.params.w < 1.5 && coverage < m.params.z {
        return out;
    }
    // OPAQUE/MASK hits that survive the cutoff are fully covered — the
    // material alpha is surface shading, not card opacity (final alpha is
    // the instance opacity). BLEND never reaches this path.
    coverage = select(1.0, coverage, m.params.w > 1.5);

    var n = normalize(surface.n);
    let normal_t = material_map_uv(m, 2u, surface.uv0, surface.uv1);
    let normal_tex = textureSampleLevel(
        t_src_normal,
        s_src,
        normal_t.xy,
        layer,
        source_map_lod(surface, 2u, ro, rd, m),
    ).xyz * 2.0 - vec3<f32>(1.0);
    if dot(normal_tex, normal_tex) > 0.01 {
        let tbn = source_tangent_frame(n, surface.p, normal_t.xy, surface.tangent);
        n = normalize(tbn * (normal_tex * vec3<f32>(m.params2.y, m.params2.y, 1.0)));
    }
    if dot(n, rd) > 0.0 {
        n = -n;
    }

    let mr_t = material_map_uv(m, 1u, surface.uv0, surface.uv1);
    let mr = textureSampleLevel(
        t_src_mr,
        s_src,
        mr_t.xy,
        layer,
        source_map_lod(surface, 1u, ro, rd, m),
    );
    let ao_t = material_map_uv(m, 3u, surface.uv0, surface.uv1);
    let ao_tex = textureSampleLevel(
        t_src_ao,
        s_src,
        ao_t.xy,
        layer,
        source_map_lod(surface, 3u, ro, rd, m),
    ).r;
    let emissive_t = material_map_uv(m, 4u, surface.uv0, surface.uv1);
    let emissive_tex = textureSampleLevel(
        t_src_emissive,
        s_src,
        emissive_t.xy,
        layer,
        source_map_lod(surface, 4u, ro, rd, m),
    ).rgb;

    out.albedo = base.rgb * m.base_color.rgb;
    out.normal = n;
    out.rough = clamp(mr.g * m.params.y, 0.03, 1.0);
    out.metal = clamp(mr.b * m.params.x, 0.0, 1.0);
    out.ao = clamp(mix(1.0, ao_tex, m.params2.x), 0.0, 1.0);
    out.emissive = emissive_tex * m.emissive.rgb * m.emissive.w;
    out.coverage = coverage;

    let world = (model * vec4<f32>(surface.p, 1.0)).xyz;
    let n_world = normalize(mat3x3<f32>(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz,
    ) * n);
    let v = normalize(lighting_env.cam_pos.xyz - world);
    out.lit = lighting_eval(out.albedo, out.metal, out.rough, n_world, v, world, out.ao)
        + out.emissive;
    return out;
}

fn projective_material(
    hit_in: vec3<f32>,
    ro: vec3<f32>,
    rd: vec3<f32>,
    px: f32,
    cells_per_px: f32,
    material_lod: f32,
    c: vec3<f32>,
    model: mat4x4<f32>,
) -> MaterialSample {
    var hit = hit_in;
    let to_cam_obj = normalize(-rd);
    var albedo = vec3<f32>(0.0);
    var normal_sum = vec3<f32>(0.0);
    var rough = 0.0;
    var metal = 0.0;
    var ao = 0.0;
    var emissive = vec3<f32>(0.0);
    var lit_sum = vec3<f32>(0.0);
    var coverage = 0.0;
    var wsum = 0.0;
    var best_face = 0u;
    var best_err = 1e9;
    var f0 = 0u;
    var f1 = 0u;
    var s0 = 0.0;
    var s1 = 0.0;
    var best_n = vec3<f32>(0.0, 0.0, 1.0);
    var best_delta = 0.0;
    // First pass: find the closest captured surface along its own view axis.
    for (var f = 0u; f < 26u; f = f + 1u) {
        let e = asset.extents[f];
        let n_f = asset.views[f].xyz;
        let uv_f = face_uv(f, hit, c);
        let res = i32(asset.params.y);
        let tc = clamp(vec2<i32>(uv_f * f32(res)), vec2<i32>(0), vec2<i32>(res - 1));
        let d_f = select(
            textureLoad(t_depth, tc, i32(f), 0).r,
            textureSampleLevel(t_depth, s_tile, uv_f, i32(f), material_lod).r,
            cells_per_px > 12.0,
        );
        let expected = (e.z * 0.5 - dot(hit - c, n_f)) / e.z;
        let err = abs(d_f - expected) * e.z;
        if err < best_err && d_f < 1.0 {
            best_err = err;
            best_face = f;
            best_n = n_f;
            best_delta = e.z * (0.5 - d_f) - dot(hit - c, n_f);
        }
    }

    // Snap the shading point onto that stored capture surface. Selection happens
    // again below at the snapped point; otherwise the chosen view can disagree
    // with the uv/depth used for shading by a sub-texel registration shift.
    if best_err < 1e8 {
        hit += best_n * best_delta;
    }
    let shade_p = hit;

    // Second pass at the corrected point: choose the two captures that can
    // actually see this surface. Depth agreement is the primary cue; camera
    // facing is only a tie-breaker.
    for (var f = 0u; f < 26u; f = f + 1u) {
        let e = asset.extents[f];
        let n_f = asset.views[f].xyz;
        let uv_f = face_uv(f, shade_p, c);
        let cov = face_coverage(f, uv_f);
        let res = i32(asset.params.y);
        let tc = clamp(vec2<i32>(uv_f * f32(res)), vec2<i32>(0), vec2<i32>(res - 1));
        let d_f = select(
            textureLoad(t_depth, tc, i32(f), 0).r,
            textureSampleLevel(t_depth, s_tile, uv_f, i32(f), material_lod).r,
            cells_per_px > 12.0,
        );
        let expected = (e.z * 0.5 - dot(shade_p - c, n_f)) / e.z;
        let err = abs(d_f - expected) * e.z;
        let tol = asset.screen.x;
        let vis = select(0.0, cov, cov >= asset.params.x)
            * (1.0 - smoothstep(tol, 3.0 * tol, err));
        let facing = max(dot(n_f, to_cam_obj), 0.0) + 0.1;
        let score = vis * (facing * facing + 0.05);
        if score > s0 {
            f1 = f0;
            s1 = s0;
            f0 = f;
            s0 = score;
        } else if score > s1 {
            f1 = f;
            s1 = score;
        }
    }

    // Depth/facing blend between the two captures that actually cover this
    // hull point. At subpixel range use an equal two-source integral: score
    // ratios there are measuring texel noise more than surface probability.
    let st = max(s0 + s1, 1e-6);
    let sample_faces = array<u32, 2>(f0, select(f0, f1, s1 > 1e-5));
    let sample_scores = array<f32, 2>(
        s0 / st,
        s1 / st,
    );
    for (var k = 0u; k < 2u; k = k + 1u) {
        let w = sample_scores[k];
        if w <= 1e-4 {
            continue;
        }
        let f = sample_faces[k];
        let e = asset.extents[f];
        var p_sample = shade_p;
        if cells_per_px > 12.0 {
            let uv_depth = face_uv(f, shade_p, c);
            let d_s = textureSampleLevel(t_depth, s_tile, uv_depth, i32(f), material_lod).r;
            let n_f = asset.views[f].xyz;
            p_sample = shade_p + n_f * (e.z * (0.5 - d_s) - dot(shade_p - c, n_f));
        }
        let uv_f = face_uv(f, p_sample, c);
        var tex: FaceData;
        if cells_per_px > 12.0 {
            tex = sample_face_level(f, uv_f, material_lod);
        } else {
            let guv = vec2<f32>(px / e.x, px / e.y) * select(0.5, asset.screen.z, asset.screen.z > 0.0);
            let ddx0 = vec2<f32>(max(guv.x, 1.0 / asset.params.y), 0.0);
            let ddy0 = vec2<f32>(0.0, max(guv.y, 1.0 / asset.params.y));
            let normal_ao0 = textureSampleGrad(t_normal_ao, s_tile, uv_f, i32(f), ddx0, ddy0);
            let n_obj = normalize(normal_ao0.xyz * 2.0 - vec3<f32>(1.0));
            let grad = tangent_uv_grad(f, shade_p, ro, rd, n_obj, guv);
            tex.albedo = textureSampleGrad(t_albedo, s_tile, uv_f, i32(f), grad[0], grad[1]);
            tex.normal_ao = textureSampleGrad(t_normal_ao, s_tile, uv_f, i32(f), grad[0], grad[1]);
            tex.mr = textureSampleGrad(t_mr, s_tile, uv_f, i32(f), grad[0], grad[1]);
            tex.emissive = textureSampleGrad(t_emissive, s_tile, uv_f, i32(f), grad[0], grad[1]);
        }
        let n_obj_s = tex.normal_ao.xyz * 2.0 - vec3<f32>(1.0);
        albedo += w * tex.albedo.rgb;
        normal_sum += w * n_obj_s;
        rough += w * tex.mr.r;
        metal += w * tex.mr.g;
        ao += w * tex.normal_ao.a;
        emissive += w * tex.emissive.rgb;
        coverage += w * tex.albedo.a;
        if cells_per_px > 12.0 {
            let world_s = (model * vec4<f32>(p_sample, 1.0)).xyz;
            let v_s = normalize(lighting_env.cam_pos.xyz - world_s);
            let n_world_s = normalize(mat3x3<f32>(
                model[0].xyz, model[1].xyz, model[2].xyz,
            ) * normalize(n_obj_s));
            let n_len = clamp(length(n_obj_s), 0.0, 1.0);
            let rough_eff = clamp(
                sqrt(tex.mr.r * tex.mr.r + (1.0 - n_len) * 2.0),
                0.03,
                1.0,
            );
            let lit_i = lighting_eval(
                tex.albedo.rgb,
                clamp(tex.mr.g, 0.0, 1.0),
                rough_eff,
                n_world_s,
                v_s,
                world_s,
                clamp(tex.normal_ao.a, 0.0, 1.0),
            ) + tex.emissive.rgb;
            lit_sum += w * lit_i;
        }
        wsum += w;
    }

    if wsum < 1e-4 {
        let f = best_face;
        let e = asset.extents[f];
        let uv_f = face_uv(f, shade_p, c);
        let guv = vec2<f32>(px / e.x, px / e.y) * select(0.5, asset.screen.z, asset.screen.z > 0.0);
        let ddx_f = vec2<f32>(max(guv.x, 1.0 / asset.params.y), 0.0);
        let ddy_f = vec2<f32>(0.0, max(guv.y, 1.0 / asset.params.y));
        let tex = sample_face(f, uv_f, ddx_f, ddy_f);
        albedo = tex.albedo.rgb;
        normal_sum = tex.normal_ao.xyz * 2.0 - vec3<f32>(1.0);
        rough = tex.mr.r;
        metal = tex.mr.g;
        ao = tex.normal_ao.a;
        emissive = tex.emissive.rgb;
        coverage = tex.albedo.a;
        if cells_per_px > 12.0 {
            let world_s = (model * vec4<f32>(shade_p, 1.0)).xyz;
            let v_s = normalize(lighting_env.cam_pos.xyz - world_s);
            let n_world_s = normalize(mat3x3<f32>(
                model[0].xyz, model[1].xyz, model[2].xyz,
            ) * normalize(normal_sum));
            lit_sum = lighting_eval(
                albedo,
                clamp(metal, 0.0, 1.0),
                clamp(rough, 0.03, 1.0),
                n_world_s,
                v_s,
                world_s,
                clamp(ao, 0.0, 1.0),
            ) + emissive;
        }
        wsum = 1.0;
    }

    let inv = 1.0 / wsum;
    return MaterialSample(
        shade_p,
        lit_sum * inv,
        albedo * inv,
        normal_sum * inv,
        clamp(rough * inv, 0.03, 1.0),
        clamp(metal * inv, 0.0, 1.0),
        clamp(ao * inv, 0.0, 1.0),
        emissive * inv,
        coverage * inv,
    );
}

@fragment
fn fs_main(input: VertexOutput) -> FragOutput {
    let inst = instances.items[input.instance];
    if inst.misc.x <= 0.0 {
        discard; // zero-opacity instance: no ray work
    }
    let c = (asset.box_min.xyz + asset.box_max.xyz) * 0.5;

    // Object-space ray: camera position through the inverse model.
    let cam_obj = (inst.inv_model * vec4<f32>(lighting_env.cam_pos.xyz, 1.0)).xyz;
    let debug_mode = u32(asset.params.w);
    let ro = cam_obj;
    let rd = normalize(input.obj_pos - ro);

    // Screen footprint in object units. LOD selection is per-instance, not
    // per-pixel: using the shell point distance lets neighboring fragments
    // choose different hull mips and turn the silhouette into confetti.
    let center_world = (inst.model * vec4<f32>(c, 1.0)).xyz;
    let center_ray_t = length(center_world - lighting_env.cam_pos.xyz);
    let obj_scale = max(length(inst.model[0].xyz), 1e-6);
    var px = center_ray_t * asset.screen.y / obj_scale;
    let res0f = asset.params.y;
    let cell0 = min(asset.extents[0].z, min(asset.extents[2].z, asset.extents[4].z)) / res0f;
    let max_hull_mip = u32(floor(log2(res0f) + 0.5));
    let cells_per_px = px / max(cell0, 1e-6);
    var hull_mip = 0u;
    if cells_per_px > 40.0 {
        // Once a screen pixel covers many atlas texels, mip-0 exposes gaps the
        // real rasterizer can no longer resolve. Aggregate at ~1 px/cell.
        hull_mip = min(u32(floor(log2(cells_per_px))) - 4u, max_hull_mip);
    }
    let material_lod = clamp(log2(cells_per_px) - 3.5, 0.0, f32(max_hull_mip));

    // Opaque assets with a voxel grid take the authoritative geometry path:
    // real ray-triangle hits through the DDA grid, no hull reconstruction.
    // debug 7 keeps the shell-decal diagnostic; MASK/voxel-less assets keep
    // the legacy hull so cutout visibility stays capture-driven.
    let exact_geom = asset.material.w > 0.5 && debug_mode != 7u;
    var hit = vec3<f32>(0.0);
    var source: SourceSurface;
    if exact_geom {
        source = source_ray_hit(ro, rd);
        if !source.valid {
            discard;
        }
        hit = source.p;
    } else {
        var missed = false;
        hit = hull_hit(ro, rd, c, debug_mode == 7u, hull_mip, &missed);
        if missed {
            discard;
        }
        source = empty_source_surface(hit);
        if cells_per_px > 8.0 {
            source = source_surface(voxel_surface_hit(ro, rd), ro, rd);
        }
    }

    if debug_mode == 1u {
        let world0 = (inst.model * vec4<f32>(hit, 1.0)).xyz;
        let clip0 = camera.view_projection * vec4<f32>(world0, 1.0);
        var out1: FragOutput;
        out1.color = vec4<f32>(1.0, 0.15, 0.1, 1.0);
        out1.velocity = vec2<f32>(0.0);
        out1.depth = clip0.z / clip0.w;
        return out1;
    }

    // Refine the object-space pixel footprint at the actual hit distance.
    let hit_ray_t = length((inst.model * vec4<f32>(hit, 1.0)).xyz - lighting_env.cam_pos.xyz);
    px = hit_ray_t * asset.screen.y / obj_scale;

    var sample = projective_material(hit, ro, rd, px, cells_per_px, material_lod, c, inst.model);
    if !exact_geom {
        // Legacy: depth/motion follow the capture-snapped shading point. The
        // fast path keeps the true triangle hit for both.
        hit = sample.p;
    }
    var albedo = sample.albedo;
    var normal_sum = sample.normal;
    var rough = sample.rough;
    var metal = sample.metal;
    var ao = sample.ao;
    var emissive = sample.emissive;
    var integrated_lit = sample.lit;
    var coverage = sample.coverage;
    if cells_per_px > 8.0 && source.valid {
        // Far field: keep hull coverage/depth, but recover material from the
        // exact source triangle instead of a capture-space reconstruction.
        let exact = source_material(source, hit, ro, rd, inst.model);
        if exact.coverage <= 0.0 {
            discard;
        }
        let w = 0.90;
        albedo = mix(albedo, exact.albedo, 0.85);
        normal_sum = normalize(mix(normal_sum, exact.normal, w));
        rough = mix(rough, exact.rough, w);
        metal = mix(metal, exact.metal, w);
        ao = mix(ao, exact.ao, w);
        emissive = mix(emissive, exact.emissive, w);
        coverage = mix(coverage, exact.coverage, w);

        let lit_p = mix(hit, source.p, w);
        let lit_world = (inst.model * vec4<f32>(lit_p, 1.0)).xyz;
        let lit_n = normalize(mat3x3<f32>(
            inst.model[0].xyz,
            inst.model[1].xyz,
            inst.model[2].xyz,
        ) * normal_sum);
        let lit_v = normalize(lighting_env.cam_pos.xyz - lit_world);
        integrated_lit = lighting_eval(albedo, metal, rough, lit_n, lit_v, lit_world, ao)
            + emissive;
    }

    // Depth of the geometry point: projected and written so the card
    // occludes exactly like the surface it stands in for.
    let world = (inst.model * vec4<f32>(hit, 1.0)).xyz;
    let clip_pos = camera.view_projection * vec4<f32>(world, 1.0);
    let hit_depth = clip_pos.z / clip_pos.w;
    // Motion vector from the shading point (not the shell): same screen_uv
    // contract as model.wgsl — prev projection minus current.
    let prev_world = (inst.prev_model * vec4<f32>(hit, 1.0)).xyz;
    let curr_ndc = camera.curr_view_projection * vec4<f32>(world, 1.0);
    let prev_ndc = camera.prev_view_projection * vec4<f32>(prev_world, 1.0);
    let vel = screen_uv(prev_ndc.xy / prev_ndc.w) - screen_uv(curr_ndc.xy / curr_ndc.w);
    let weights_dbg = vec3<f32>(cells_per_px / 64.0, material_lod / 8.0, f32(hull_mip) / 8.0);

    // Shared per-channel parity output with model_pbr.wgsl (10-15).
    if debug_mode >= 10u {
        let n_obj_dbg = normalize(normal_sum);
        let n_w_dbg = normalize(mat3x3<f32>(
            inst.model[0].xyz, inst.model[1].xyz, inst.model[2].xyz,
        ) * n_obj_dbg) * 0.5 + vec3<f32>(0.5);
        var ch = albedo;
        if debug_mode == 11u { ch = vec3<f32>(rough); }
        if debug_mode == 12u { ch = vec3<f32>(metal); }
        if debug_mode == 13u { ch = vec3<f32>(ao); }
        if debug_mode == 14u { ch = emissive; }
        if debug_mode == 15u { ch = n_w_dbg; }
        if debug_mode == 16u {
            let world_v = (inst.model * vec4<f32>(source.p, 1.0)).xyz;
            ch = select(world, world_v, source.valid) * 0.25 + vec3<f32>(0.5);
        }
        var outc: FragOutput;
        outc.color = vec4<f32>(ch, 1.0);
        outc.velocity = vec2<f32>(0.0);
        outc.depth = hit_depth;
        return outc;
    }

    if debug_mode == 2u {
        var out2: FragOutput;
        out2.color = vec4<f32>(0.0, coverage, 0.0, 1.0);
        out2.velocity = vec2<f32>(0.0);
        out2.depth = hit_depth;
        return out2;
    }
    if debug_mode == 3u {
        var out3: FragOutput;
        out3.color = vec4<f32>(albedo, 1.0);
        out3.velocity = vec2<f32>(0.0);
        out3.depth = hit_depth;
        return out3;
    }

    let n_obj = normalize(normal_sum);
    // Object-space normal -> world (the shell is uniform-scaled in practice,
    // so mat3 normalizes fine).
    let n_world = normalize(mat3x3<f32>(inst.model[0].xyz, inst.model[1].xyz, inst.model[2].xyz) * n_obj);

    if debug_mode == 4u {
        var out4: FragOutput;
        out4.color = vec4<f32>(n_world * 0.5 + vec3<f32>(0.5), 1.0);
        out4.velocity = vec2<f32>(0.0);
        out4.depth = hit_depth;
        return out4;
    }
    if debug_mode == 5u {
        var out5: FragOutput;
        out5.color = vec4<f32>(vec3<f32>((ray_box(ro, rd)).x * 0.1), 1.0);
        out5.velocity = vec2<f32>(0.0);
        out5.depth = hit_depth;
        return out5;
    }
    if debug_mode == 6u {
        var out6: FragOutput;
        out6.color = vec4<f32>(weights_dbg, 1.0);
        out6.velocity = vec2<f32>(0.0);
        out6.depth = hit_depth;
        return out6;
    }

    if debug_mode == 8u {
        var out8: FragOutput;
        out8.color = vec4<f32>(metal, rough, ao, 1.0);
        out8.velocity = vec2<f32>(0.0);
        out8.depth = hit_depth;
        return out8;
    }

    // Opaque/MASK-baked card: alpha is exactly the instance crossfade weight.
    // Averaged projected coverage is NOT camera-space opacity — multiplying
    // it in made fully-selected cards dither into transparency. The hull
    // (depth sentinel) carries alpha-test holes; temporal AA handles the
    // silhouette edge.
    let alpha = clamp(inst.misc.x, 0.0, 1.0);

    // Shared lighting at the geometry world position: identical inputs to
    // model_pbr.wgsl -> colour parity with the mesh.
    let v = normalize(lighting_env.cam_pos.xyz - world);
    let lit = select(
        lighting_eval(albedo, metal, rough, n_world, v, world, ao) + emissive,
        integrated_lit,
        cells_per_px > 12.0,
    );

    var out: FragOutput;
    out.color = vec4<f32>(lit, alpha);
    out.velocity = vel;
    out.depth = hit_depth;
    return out;
}
