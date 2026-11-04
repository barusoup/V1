// Final present: draw the composed display image onto the swapchain.
// Plus-shaped invert crosshair: the pixels inside a centered plus zone
// return 1 - colour, so the crosshair is never a solid UI element and
// always shows inverted scene/UI behind it.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct PresentPush {
    screen: vec2<f32>,
    // 1.0 when the app asked for the crosshair (see Renderer::set_show_crosshair).
    crosshair: f32,
}

var<push_constant> push: PresentPush;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    o.uv = vec2<f32>(p[i].x * 0.5 + 0.5, p[i].y * 0.5 + 0.5);
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return present(in, textureSample(src_tex, samp, in.uv));
}

/// Gamma-blend display target (`UiBlend::Gamma`): the display image holds
/// sRGB-encoded values in a UNORM format; decode them so the sRGB
/// swapchain's encode restores the same bytes.
@fragment
fn fs_gamma(in: VsOut) -> @location(0) vec4<f32> {
    let e = textureSample(src_tex, samp, in.uv);
    let low = e.rgb / 12.92;
    let high = pow((e.rgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return present(in, vec4<f32>(select(high, low, e.rgb <= vec3<f32>(0.04045)), e.a));
}

fn present(in: VsOut, c: vec4<f32>) -> vec4<f32> {
    let px = in.uv * push.screen;
    let center = push.screen * 0.5;
    let d = abs(px - center);
    let arm = 11.0;
    let half_thick = 2.0;
    let inside = (d.x < half_thick && d.y < arm) || (d.y < half_thick && d.x < arm);
    if (push.crosshair > 0.5 && inside) {
        return vec4<f32>(vec3<f32>(1.0) - c.rgb, c.a);
    }
    return c;
}