// Separable Gaussian blur for the UI's frosted-glass backdrop. Two entry points share one
// fullscreen triangle: `fs_h` blurs along x, `fs_v` along y. Taps are spaced two texels apart,
// so a pair of passes covers a wide soft radius at half the display resolution.

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
    // Not flipped: texel row y of the source lands on row y of the target.
    o.uv = vec2<f32>(p[i].x * 0.5 + 0.5, p[i].y * 0.5 + 0.5);
    return o;
}

fn gauss(uv: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    // Taps one texel apart: wider spacing puts the taps on separate copies of a feature, which reads
    // as a lattice of ghosts. The softness comes from more rounds, not wider taps.
    let step = dir * 1.0 / vec2<f32>(textureDimensions(src_tex));
    var c = textureSample(src_tex, samp, uv) * 0.2270270;
    c += (textureSample(src_tex, samp, uv + step * 1.0) + textureSample(src_tex, samp, uv - step * 1.0)) * 0.1945946;
    c += (textureSample(src_tex, samp, uv + step * 2.0) + textureSample(src_tex, samp, uv - step * 2.0)) * 0.1216216;
    c += (textureSample(src_tex, samp, uv + step * 3.0) + textureSample(src_tex, samp, uv - step * 3.0)) * 0.0540541;
    c += (textureSample(src_tex, samp, uv + step * 4.0) + textureSample(src_tex, samp, uv - step * 4.0)) * 0.0162162;
    return c;
}

@fragment
fn fs_h(in: VsOut) -> @location(0) vec4<f32> {
    return gauss(in.uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_v(in: VsOut) -> @location(0) vec4<f32> {
    return gauss(in.uv, vec2<f32>(0.0, 1.0));
}
