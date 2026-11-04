@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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
    o.uv = vec2<f32>(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src_tex, samp, in.uv);
}

/// Gamma-blend display target (`UiBlend::Gamma`): the display image is
/// UNORM holding sRGB-encoded values, so the linear scene is encoded here.
@fragment
fn fs_gamma(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, samp, in.uv);
    let low = c.rgb * 12.92;
    let high = 1.055 * pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return vec4<f32>(select(high, low, c.rgb <= vec3<f32>(0.0031308)), c.a);
}
