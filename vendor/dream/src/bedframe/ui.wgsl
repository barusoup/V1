// UI overlay: textured, vertex-coloured, alpha-blended quads in pixel space.
//
// Coordinates: positions are pixels with the origin at the screen's top-left
// (push.screen is the target size). The negative viewport used across the
// engine leaves NDC +y at increasing target rows, and the display target's
// row 0 is the top of the frame, so pixel y=0 must land at NDC -y.
//
// Colours arrive as sRGB bytes (UNORM attribute) and are converted to linear
// here; the target is an sRGB surface, so the hardware encodes on write.
//
// One vertex format serves both textured quads and anti-aliased rounded
// rectangles. For a quad, `params.w == 0` and `uv` is the texture sample
// coordinate. For a shape, `uv` is the pixel offset from the rect's centre
// and `params = [half_w, half_h, radius, 1]`; the fragment shader evaluates a
// rounded-box signed distance and uses it as coverage, so corners are smooth
// without any corner geometry.

struct Push {
    screen: vec2<f32>,
}

var<push_constant> push: Push;

@group(0) @binding(0) var t_tex: texture_2d<f32>;
@group(0) @binding(1) var s_tex: sampler;

struct Vs_out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) params: vec4<f32>,
    // Crop path: the quad's top-left in pixel space, bitpacked as u32
    // (WGSL only allows @interpolate(flat) on integer types). Provoking
    // vertex semantics: quads emit V0 = top-left first, so every
    // fragment sees the same origin.
    @location(3) @interpolate(flat) quad_origin_bits: vec2<u32>,
    // This fragment's pixel-space position (interpolated).
    @location(4) pix: vec2<f32>,
    // Extra shape data (the rounded clip's half extent for mode -1;
    // zero otherwise). Uniform per quad.
    @location(5) extra: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) in_pos: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_color: vec4<f32>,
    @location(3) in_params: vec4<f32>,
    @location(4) in_extra: vec2<f32>,
) -> Vs_out {
    var out: Vs_out;
    let ndc = vec2<f32>(
        in_pos.x / max(push.screen.x, 1.0) * 2.0 - 1.0,
        in_pos.y / max(push.screen.y, 1.0) * 2.0 - 1.0,
    );
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = in_uv;
    out.color = in_color;
    out.params = in_params;
    out.extra = in_extra;
    out.quad_origin_bits = vec2<u32>(bitcast<u32>(in_pos.x), bitcast<u32>(in_pos.y));
    out.pix = in_pos;
    return out;
}

/// Instanced-quad entry: one 52-byte instance record per quad, six
/// corners expanded from `vertex_index` — the legacy indexed path is
/// untouched. Instance colour arrives already-sRGB-unorm like the
/// legacy attribute.
@vertex
fn vs_instance(
    @location(0) inst_rect: vec4<f32>,
    @location(1) inst_uv: vec4<f32>,
    @location(2) inst_color: vec4<f32>,
    @location(3) inst_params: vec4<f32>,
    @location(4) inst_extra: vec2<f32>,
    @builtin(vertex_index) vid: u32,
) -> Vs_out {
    var out = instance_out(inst_rect, inst_uv, inst_color, inst_params, inst_extra, vid);
    // Instance colour arrives as linear already (the VS converts, so
    // the fragment can skip the pow per pixel via fs_linear).
    out.color = vec4<f32>(srgb_to_linear(inst_color.rgb), inst_color.a);
    return out;
}

/// Gamma-blend instanced entry (see `fs_main_gamma`): the colour stays
/// sRGB-encoded, so blending happens on encoded values as in a browser.
@vertex
fn vs_instance_gamma(
    @location(0) inst_rect: vec4<f32>,
    @location(1) inst_uv: vec4<f32>,
    @location(2) inst_color: vec4<f32>,
    @location(3) inst_params: vec4<f32>,
    @location(4) inst_extra: vec2<f32>,
    @builtin(vertex_index) vid: u32,
) -> Vs_out {
    return instance_out(inst_rect, inst_uv, inst_color, inst_params, inst_extra, vid);
}

/// Shared instanced-quad expansion; `color` is left as it arrived.
fn instance_out(
    inst_rect: vec4<f32>,
    inst_uv: vec4<f32>,
    inst_color: vec4<f32>,
    inst_params: vec4<f32>,
    inst_extra: vec2<f32>,
    vid: u32,
) -> Vs_out {
    // [00,10,11, 00,11,01] — two triangles, V0 = top-left so the flat
    // quad origin is stable under the same provoking-vertex rule.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = corners[vid];
    let pos = inst_rect.xy + c * inst_rect.zw;
    var out: Vs_out;
    let ndc = vec2<f32>(
        pos.x / max(push.screen.x, 1.0) * 2.0 - 1.0,
        pos.y / max(push.screen.y, 1.0) * 2.0 - 1.0,
    );
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = inst_uv.xy + c * (inst_uv.zw - inst_uv.xy);
    out.color = inst_color;
    out.params = inst_params;
    out.extra = inst_extra;
    out.quad_origin_bits = vec2<u32>(bitcast<u32>(inst_rect.x), bitcast<u32>(inst_rect.y));
    out.pix = pos;
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, c <= vec3<f32>(0.0031308));
}

/// Rounded-box signed distance from a centre-relative position. The CPU
/// clamps the radius before it gets here; clamping again keeps the helper
/// honest for callers that compose their own.
fn rounded_box_d(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let rr = min(r, min(half.x, half.y));
    let q = abs(p) - (half - vec2<f32>(rr));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

/// 1 when the quad is a radial gradient (mode -2), else 0. Computed with
/// `select`/`mix` rather than control flow so the texture sample below
/// stays in uniform control flow.
fn radial_mask(in: Vs_out) -> f32 {
    return select(0.0, 1.0, in.params.w > -2.5 && in.params.w < -1.5);
}

/// RGBA byte-unpack of a colour packed into a float's bits (radial
/// gradients carry their "to" colour in uv[0]); bytes always make a
/// finite sRGB triple.
fn unpack_srgb8(bits: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(bits & 0xffu),
        f32((bits >> 8u) & 0xffu),
        f32((bits >> 16u) & 0xffu),
    ) / 255.0;
}

/// A full turn, in radians.
const TAU: f32 = 6.283185307179586;

/// Anti-aliased arc stroke (mode `-3`): the band between
/// `mid_radius - half_thickness` and `mid_radius + half_thickness`,
/// sweeping `sweep` radians clockwise on screen from `start`, with round
/// caps of `half_thickness` at both ends. Angles are measured from `+x`
/// (the top is `-PI/2`). `params` carries `[mid_radius, half_thickness,
/// start, -3]`, `extra.x` the sweep, and `uv` the centre-relative pixel
/// position, as for the other SDF modes.
fn arc_coverage(in: Vs_out) -> f32 {
    let r_mid = in.params.x;
    let half_t = in.params.y;
    let start = in.params.z;
    let sweep = in.extra.x;
    let p = in.uv;
    var delta = atan2(p.y, p.x) - start;
    // Normalize the angle from the arc's start into 0..TAU (a full
    // sweep keeps every angle inside).
    delta = delta - floor(delta / TAU) * TAU;
    // Inside the sweep the nearest boundary is the centreline circle;
    // past either end it is the nearer round cap's centre.
    let cap0 = vec2<f32>(cos(start), sin(start)) * r_mid;
    let cap1 = vec2<f32>(cos(start + sweep), sin(start + sweep)) * r_mid;
    let radial = abs(length(p) - r_mid);
    let capped = min(length(p - cap0), length(p - cap1));
    let d = select(capped, radial, delta <= sweep) - half_t;
    return clamp(0.5 - d, 0.0, 1.0);
}

/// Error function (Abramowitz-Stegun 7.1.27 form, max error ~5e-4).
fn erf2(x: vec2<f32>) -> vec2<f32> {
    let s = sign(x);
    let a = abs(x);
    var t = vec2<f32>(1.0) + (vec2<f32>(0.278393) + (vec2<f32>(0.230389) + 0.078108 * (a * a)) * a) * a;
    t = t * t;
    return s - s / (t * t);
}

/// The 1D Gaussian of standard deviation `sigma` at `x`.
fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (2.5066282746 * sigma);
}

/// The Gaussian blur of a rounded box, integrated exactly along x for one
/// row `y` (both relative to the box centre).
fn blurred_row(x: f32, y: f32, sigma: f32, corner: f32, half: vec2<f32>) -> f32 {
    let delta = min(half.y - corner - abs(y), 0.0);
    let curved = half.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
    let integral = 0.5 + 0.5 * erf2((x + vec2<f32>(-curved, curved)) * (0.70710678 / sigma));
    return integral.y - integral.x;
}

/// Gaussian box shadow (mode `-4`): the coverage a CSS `box-shadow` blur
/// gives a rounded box — `params = [half_w, half_h, radius, -4]`,
/// `extra.x` the standard deviation (CSS blur radius / 2), `uv` the
/// centre-relative pixel position. Exact along x, sampled along y.
fn gaussian_shadow_coverage(in: Vs_out) -> f32 {
    let sigma = max(in.extra.x, 0.25);
    let half = in.params.xy;
    let corner = min(in.params.z, min(half.x, half.y));
    let p = in.uv;
    let low = p.y - half.y;
    let high = p.y + half.y;
    let start = clamp(-3.0 * sigma, low, high);
    let end = clamp(3.0 * sigma, low, high);
    let step = (end - start) / 4.0;
    var y = start + step * 0.5;
    var value = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        value = value + blurred_row(p.x, p.y - y, sigma, corner, half) * gaussian(y, sigma) * step;
        y = y + step;
    }
    return clamp(value, 0.0, 1.0);
}

/// Coverage (0..1) for every shape mode, `alpha` untouched for a plain
/// textured quad. `params.w` is the mode: `1` rounded fill, `2` rounded
/// texture, `3 + thickness` ring, `-1` rounded-clip textured quad,
/// `-2` radial gradient, `-3` arc stroke, `-4` Gaussian box shadow,
/// `-(5 + softness)` soft shadow/glow.
fn coverage(in: Vs_out, alpha: f32) -> f32 {
    if (in.params.w < -2.5 && in.params.w > -3.5) {
        return alpha * arc_coverage(in);
    } else if (in.params.w < -3.5 && in.params.w > -4.5) {
        return alpha * gaussian_shadow_coverage(in);
    } else if (in.params.w > 2.5) {
        // Ring: params.w = 3 + thickness. Coverage is the band between the
        // outer edge (d = 0) and the inner edge (d = -thickness), so a
        // hollow outline is one shape, not a filled shape with a hole.
        let t = in.params.w - 3.0;
        let q = abs(in.uv) - (in.params.xy - vec2<f32>(in.params.z));
        let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - in.params.z;
        return alpha * clamp(0.5 - d, 0.0, 1.0) * clamp(0.5 + d + t, 0.0, 1.0);
    } else if (in.params.w > 0.5) {
        // Rounded-box SDF: p = |uv| - (half - r); d = len(max(p,0) +
        // min(max(p.x,p.y),0)) - r. Coverage is a 1px band around d = 0.
        var p = in.uv;
        var half = in.params.xy;
        if (in.params.w > 1.5) {
            // Rounded texture: params.xy carry the FULL pixel size, uv
            // stays a pure sample coordinate (a crop may span any subrect).
            // The SDF position is recovered from the flat quad origin —
            // V0 top-left under the provoking-vertex assumption.
            let origin = vec2<f32>(
                bitcast<f32>(in.quad_origin_bits.x),
                bitcast<f32>(in.quad_origin_bits.y),
            );
            // params.xy is the FULL rect size; p is relative to the
            // rect's centre, so subtract the half-size.
            p = in.pix - origin - half * 0.5;
            half = half * 0.5;
        }
        let d = rounded_box_d(p, half, in.params.z);
        return alpha * clamp(0.5 - d, 0.0, 1.0);
    } else if (in.params.w < -0.5 && in.params.w > -1.5) {
        // Rounded clip (mode -1): carve the corners of a plain textured
        // quad against the clip rect — centre in params.xy, radius in
        // params.z, half extent in `extra`.
        let d = rounded_box_d(in.pix - in.params.xy, in.extra, in.params.z);
        return alpha * clamp(0.5 - d, 0.0, 1.0);
    } else if (in.params.w <= -4.5) {
        // Soft shadow (mode -(5 + softness)): full coverage inside the
        // silhouette, fading over `softness` pixels across the edge. The
        // clamp at 1.0 keeps even a hard shadow anti-aliased.
        let soft = max(-in.params.w - 5.0, 1.0);
        let d = rounded_box_d(in.uv, in.params.xy, in.params.z);
        return alpha * clamp(0.5 - d / soft, 0.0, 1.0);
    }
    // Radial gradients and plain quads are fully covered here; the radial
    // edge band is applied by the colour stage below.
    return alpha;
}

/// Radial gradient progress (0 inside the inner radius, 1 at the outer),
/// 0.0 for every other mode. Also derives the one-pixel AA band at the
/// outer edge so the caller can fold it into alpha.
fn radial_t(in: Vs_out) -> f32 {
    let origin = vec2<f32>(
        bitcast<f32>(in.quad_origin_bits.x),
        bitcast<f32>(in.quad_origin_bits.y),
    );
    let half = in.params.xy * 0.5;
    let lenp = length(in.pix - origin - half);
    let outer = max(min(in.params.z, min(half.x, half.y)), 1e-4);
    let inner = clamp(in.uv.y, 0.0, outer);
    return clamp((lenp - inner) / max(outer - inner, 1e-4), 0.0, 1.0);
}

/// AA band at a radial gradient's outer edge (0 for every other mode).
fn radial_edge(in: Vs_out) -> f32 {
    let origin = vec2<f32>(
        bitcast<f32>(in.quad_origin_bits.x),
        bitcast<f32>(in.quad_origin_bits.y),
    );
    let half = in.params.xy * 0.5;
    let lenp = length(in.pix - origin - half);
    let outer = max(min(in.params.z, min(half.x, half.y)), 1e-4);
    return clamp(0.5 - (lenp - outer), 0.0, 1.0);
}

@fragment
fn fs_main(in: Vs_out) -> @location(0) vec4<f32> {
    return shade(in, false);
}

/// Gamma-blend fragment entry ([`UiBlend::Gamma`]): colours stay
/// sRGB-encoded and the target is a UNORM image, so translucent layers,
/// gradients and anti-aliased edges blend on encoded values the way a web
/// browser composites. Texels are re-encoded after the hardware's sRGB
/// decode so images keep their stored values.
@fragment
fn fs_main_gamma(in: Vs_out) -> @location(0) vec4<f32> {
    return shade(in, true);
}

fn shade(in: Vs_out, gamma: bool) -> vec4<f32> {
    // Radial gradients carry their "to" colour and inner radius in the UV
    // channel — not texture coordinates — so the sample point is steered
    // to a safe value for them (mix, no control flow).
    let mask = radial_mask(in);
    let sample_uv = mix(in.uv, vec2<f32>(0.5, 0.5), mask);
    var tex = textureSample(t_tex, s_tex, sample_uv);
    let packed = bitcast<u32>(in.uv.x);
    let t = radial_t(in) * mask;
    var rgb0 = in.color.rgb;
    var to_rgb = unpack_srgb8(packed);
    if (gamma) {
        tex = vec4<f32>(linear_to_srgb(tex.rgb), tex.a);
    } else {
        rgb0 = srgb_to_linear(rgb0);
        to_rgb = srgb_to_linear(to_rgb);
    }
    // Premultiplied mix: a fade toward a transparent colour keeps its
    // hue and only ramps coverage — mixing straight channels would drag
    // the midtones through black (the rule `Color::lerp_value` follows).
    let to_a = f32((packed >> 24u) & 0xffu) / 255.0;
    let a0 = in.color.a;
    let mixed_a = mix(a0, to_a, t);
    let rgb = mix(rgb0 * a0, to_rgb * to_a, t) / max(mixed_a, 1e-4);
    let alpha = coverage(in, 1.0) * mixed_a * mix(1.0, radial_edge(in), mask);
    // Straight alpha: the blend state interpolates onto the target.
    return vec4<f32>(rgb, alpha) * tex;
}

/// Instance-path fragment: the vertex stage already converted the tint
/// to linear, so this entry skips the per-pixel pow entirely. Texture
/// texels keep their GPU-side sRGB decode either way.
@fragment
fn fs_linear(in: Vs_out) -> @location(0) vec4<f32> {
    let mask = radial_mask(in);
    let sample_uv = mix(in.uv, vec2<f32>(0.5, 0.5), mask);
    let tex = textureSample(t_tex, s_tex, sample_uv);
    let packed = bitcast<u32>(in.uv.x);
    let t = radial_t(in) * mask;
    // Premultiplied mix (see `fs_main`): hue-preserving fade to alpha.
    let rgb0 = in.color.rgb;
    let to_rgb = srgb_to_linear(unpack_srgb8(packed));
    let to_a = f32((packed >> 24u) & 0xffu) / 255.0;
    let a0 = in.color.a;
    let mixed_a = mix(a0, to_a, t);
    let rgb = mix(rgb0 * a0, to_rgb * to_a, t) / max(mixed_a, 1e-4);
    let alpha = coverage(in, 1.0) * mixed_a * mix(1.0, radial_edge(in), mask);
    return vec4<f32>(rgb, alpha) * tex;
}
