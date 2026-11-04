// GPU post-bake: downsample / dilate / mipgen / pack over the six capture
// layers. All buffers are layer-major (layer * res*res + y * res + x); the
// depth channel stores one f32 per texel with 1.0 = uncovered sentinel.

struct DParams {
    res: u32,      // output resolution for down/dilate/mip; level res for pack
    ss: u32,       // supersample factor (downsample) or mip level (pack)
    src_off: u32,  // texel offset of src level region in the chain buffers
    dst_off: u32,  // texel offset of dst level region
}
var<push_constant> pcv: DParams;
fn p() -> DParams { return pcv; }

// --------------------------------------------------------------- downsample
@group(0) @binding(1) var t_alb: texture_2d_array<f32>;
@group(0) @binding(2) var t_nrm: texture_2d_array<f32>;
@group(0) @binding(3) var t_mrv: texture_2d_array<f32>;
@group(0) @binding(4) var t_emi: texture_2d_array<f32>;
@group(0) @binding(5) var t_dep: texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read_write> b_alb: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> b_nrm: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> b_mrv: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> b_emi: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> b_dep: array<f32>;
@group(0) @binding(11) var<storage, read_write> b_cov: array<atomic<u32>>;

fn idx(x: u32, y: u32, z: u32, res: u32) -> u32 {
    return (z * res + y) * res + x;
}

@compute @workgroup_size(8, 8, 1)
fn cs_down(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = p().res;
    if id.x >= res || id.y >= res {
        return;
    }
    let ss = p().ss;
    var wa = 0.0;
    var alb = vec3<f32>(0.0);
    var nrm = vec4<f32>(0.0);
    var mrv = vec3<f32>(0.0);
    var emi = vec3<f32>(0.0);
    var dmin = 1.0;
    var covered = 0.0;
    for (var dy = 0u; dy < ss; dy = dy + 1u) {
        for (var dx = 0u; dx < ss; dx = dx + 1u) {
            let tc = vec2<i32>(i32(id.x * ss + dx), i32(id.y * ss + dy));
            let a = textureLoad(t_alb, tc, i32(id.z), 0);
            let w = a.a;
            wa += w;
            covered += select(0.0, 1.0, w > 0.5);
            alb = alb + a.rgb * w;
            let n = textureLoad(t_nrm, tc, i32(id.z), 0);
            nrm = nrm + n * w;
            let m = textureLoad(t_mrv, tc, i32(id.z), 0);
            mrv = mrv + m.rgb * w;
            let e = textureLoad(t_emi, tc, i32(id.z), 0);
            emi = emi + e.rgb * w;
            let d = textureLoad(t_dep, tc, i32(id.z), 0).r;
            if w > 0.5 && d < dmin {
                dmin = d;
            }
        }
    }
    let o = idx(id.x, id.y, id.z, res);
    let cov = wa / f32(ss * ss);
    let norm = select(0.0, 1.0 / wa, wa > 0.0);
    // normals: decode the biased mean, renormalize signed, re-encode.
    let ns = nrm.xyz * norm * 2.0 - vec3<f32>(1.0);
    let l = length(ns);
    let nn = select(vec3<f32>(0.0, 1.0, 0.0), ns / l, l > 1e-4);
    b_alb[o] = vec4<f32>(alb * norm, cov);
    b_nrm[o] = vec4<f32>(nn * 0.5 + vec3<f32>(0.5), nrm.a * norm);
    b_mrv[o] = vec4<f32>(mrv * norm, cov);
    b_emi[o] = vec4<f32>(emi * norm, cov);
    b_dep[o] = dmin;
    atomicAdd(&b_cov[id.z], u32(covered));
}

// ------------------------------------------------------------------- dilate
// propagate covered texels into uncovered ones (mask = albedo alpha > 0).
fn dp() -> DParams { return pcv; }
@group(0) @binding(1) var<storage, read> s_alb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_nrm: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> s_mrv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> s_emi: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> d_alb: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> d_nrm: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> d_mrv: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> d_emi: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn cs_dilate(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = dp().res;
    if id.x >= res || id.y >= res {
        return;
    }
    let o = idx(id.x, id.y, id.z, res);
    let a = s_alb[o];
    if a.a > 0.0 {
        d_alb[o] = a;
        d_nrm[o] = s_nrm[o];
        d_mrv[o] = s_mrv[o];
        d_emi[o] = s_emi[o];
        return;
    }
    var acc_alb = vec3<f32>(0.0);
    var acc_nrm = vec4<f32>(0.0);
    var acc_mrv = vec3<f32>(0.0);
    var acc_emi = vec3<f32>(0.0);
    var n = 0.0;
    let offs = array<vec2<i32>, 4>(
        vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
    for (var k = 0u; k < 4u; k = k + 1u) {
        let off = offs[k];
        let xx = i32(id.x) + off.x;
        let yy = i32(id.y) + off.y;
        if xx < 0 || yy < 0 || xx >= i32(res) || yy >= i32(res) {
            continue;
        }
        let oi = idx(u32(xx), u32(yy), id.z, res);
        if s_alb[oi].a > 0.0 {
            acc_alb += s_alb[oi].rgb;
            acc_nrm += s_nrm[oi];
            acc_mrv += s_mrv[oi].rgb;
            acc_emi += s_emi[oi].rgb;
            n += 1.0;
        }
    }
    let norm = select(0.0, 1.0 / n, n > 0.0);
    d_alb[o] = vec4<f32>(acc_alb * norm, a.a);
    d_nrm[o] = acc_nrm * norm;
    d_mrv[o] = vec4<f32>(acc_mrv * norm, s_mrv[o].a);
    d_emi[o] = vec4<f32>(acc_emi * norm, s_emi[o].a);
}

// ------------------------------------------------------------------- mipgen
// src f32 level -> dst f32 level (band1 normals decode/renorm/encode,
// depth min-reduce). Offsets into the chain buffers are passed via params.
fn mp() -> DParams { return pcv; }
@group(0) @binding(1) var<storage, read> m_alb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> m_nrm: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> m_mrv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> m_emi: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> m_dep: array<f32>;
@group(0) @binding(6) var<storage, read_write> g_alb: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> g_nrm: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> g_mrv: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> g_emi: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> g_dep: array<f32>;

fn src_alb_at(x: u32, y: u32, z: u32, sr: u32) -> vec4<f32> {
    return m_alb[idx(min(x, sr - 1u), min(y, sr - 1u), z, sr)];
}

@compute @workgroup_size(8, 8, 1)
fn cs_mip(@builtin(global_invocation_id) id: vec3<u32>) {
    let dr = mp().res;        // dst res
    let sr = mp().ss;          // src res
    if id.x >= dr || id.y >= dr {
        return;
    }
    let o = mp().dst_off + idx(id.x, id.y, id.z, dr);
    var acc_a = vec4<f32>(0.0);
    var acc_n = vec4<f32>(0.0);
    var acc_m = vec4<f32>(0.0);
    var acc_e = vec4<f32>(0.0);
    var wa = 0.0;
    var wn = 0.0;
    var wm = 0.0;
    var we = 0.0;
    var dmin = 1.0;
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx = 0u; dx < 2u; dx = dx + 1u) {
            let xx = id.x * 2u + dx;
            let yy = id.y * 2u + dy;
            let i = mp().src_off + idx(min(xx, sr - 1u), min(yy, sr - 1u), id.z, sr);
            let a = m_alb[i];
            let n = m_nrm[i];
            let m = m_mrv[i];
            let e = m_emi[i];
            // Albedo alpha is the coverage channel. Normals carry AO in
            // alpha, so they must still be weighted by coverage (a dark AO
            // texel is not "less present" than a bright one).
            acc_a += a * a.a;
            acc_n += n * a.a;
            acc_m += m * m.a;
            acc_e += e * e.a;
            wa += a.a;
            wn += a.a;
            wm += m.a;
            we += e.a;
            dmin = min(dmin, m_dep[i]);
        }
    }
    let na = select(0.0, 1.0 / wa, wa > 0.0);
    let nn_ = select(0.0, 1.0 / wn, wn > 0.0);
    let nm = select(0.0, 1.0 / wm, wm > 0.0);
    let ne = select(0.0, 1.0 / we, we > 0.0);
    g_alb[o] = vec4<f32>(acc_a.rgb * na, wa * 0.25);
    g_mrv[o] = vec4<f32>(acc_m.rgb * nm, wm * 0.25);
    g_emi[o] = vec4<f32>(acc_e.rgb * ne, we * 0.25);
    // Normals stay biased through the chain (the packer renormalizes) —
    // matches the CPU reference, which defers decode/renorm/encode to pack.
    g_nrm[o] = vec4<f32>(acc_n.xyz * nn_, acc_n.w * nn_);
    g_dep[o] = dmin;
}

// --------------------------------------------------------------------- pack
// f32 chain level -> packed output (u32 rgba8 / depth u16 in low bits).
// mode: 0 albedo(srgb), 1 normal(renorm)+ao, 2 mr(unorm), 3 emissive(srgb), 4 depth.
fn pp() -> DParams { return pcv; }
@group(0) @binding(1) var<storage, read> c_alb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> c_nrm: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> c_mrv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> c_emi: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> c_dep: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_alb: array<u32>;
@group(0) @binding(7) var<storage, read_write> k_nrm: array<u32>;
@group(0) @binding(8) var<storage, read_write> k_mrv: array<u32>;
@group(0) @binding(9) var<storage, read_write> k_emi: array<u32>;
@group(0) @binding(10) var<storage, read_write> k_dep: array<u32>;

fn srgb_ch(v: f32) -> u32 {
    return u32(clamp(v, 0.0, 1.0) * 255.0 + 0.5) & 0xFFu;
}
fn srgb_px(v: f32) -> u32 {
    return u32(pow(clamp(v, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5) & 0xFFu;
}

@compute @workgroup_size(8, 8, 1)
fn cs_pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = pp().res;
    if id.x >= res || id.y >= res {
        return;
    }
    let so = pp().src_off + idx(id.x, id.y, id.z, res);
    let o = pp().dst_off + idx(id.x, id.y, id.z, res);
    let a = c_alb[so];
    k_alb[o] = srgb_px(a.r) | (srgb_px(a.g) << 8u) | (srgb_px(a.b) << 16u) | (srgb_ch(a.a) << 24u);
    let v = c_nrm[so].xyz * 2.0 - vec3<f32>(1.0);
    let l = length(v);
    let nn = select(vec3<f32>(0.0, 1.0, 0.0), v / l, l > 1e-4) * 0.5 + vec3<f32>(0.5);
    k_nrm[o] = srgb_ch(nn.x) | (srgb_ch(nn.y) << 8u) | (srgb_ch(nn.z) << 16u) | (srgb_ch(c_nrm[so].a) << 24u);
    let m = c_mrv[so];
    k_mrv[o] = srgb_ch(m.r) | (srgb_ch(m.g) << 8u) | (255u << 24u);
    let e = c_emi[so];
    k_emi[o] = srgb_px(e.r) | (srgb_px(e.g) << 8u) | (srgb_px(e.b) << 16u) | (srgb_ch(e.a) << 24u);
    // Depth packs pairs of u16 per u32 so the buffer can be copied straight
    // into the R16_UNORM image (id.x covers ceil(res/2) texels).
    if id.x * 2u >= res {
        return;
    }
    let x0 = id.x * 2u;
    let o0 = (pp().dst_off + idx(x0, id.y, id.z, res)) / 2u;
    let i0 = pp().src_off + idx(min(x0, res - 1u), id.y, id.z, res);
    let i1 = pp().src_off + idx(min(x0 + 1u, res - 1u), id.y, id.z, res);
    let d0 = u32(clamp(c_dep[i0], 0.0, 1.0) * 65535.0) & 0xFFFFu;
    let d1 = u32(clamp(c_dep[i1], 0.0, 1.0) * 65535.0) & 0xFFFFu;
    k_dep[o0] = d0 | (d1 << 16u);
}
