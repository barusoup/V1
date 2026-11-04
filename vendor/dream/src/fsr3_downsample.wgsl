// FSR 3.1 helper passes: plain 2x2 box downsampling.
//
// AMD uses the single-pass downsampler (SPD) for the shading-change pyramid and
// for the farthest-depth mip. SPD's reduction is a 2x2 box average, so these
// entry points reproduce the same values a level at a time. Plain box filtering
// over these small levels is what the SDK's SpdReduce4 computes.

// Shading-change pyramid level 0 (rg) -> level 1.
@compute @workgroup_size(8, 8, 1)
fn downsample_mip1(@builtin(global_invocation_id) gid: vec3<u32>) {
    textureStore(w_spd_mip1, vec2<i32>(gid.xy), Box4RG(vec2<i32>(gid.xy)));
}

// Shading-change pyramid level 1 (rg) -> level 2.
@compute @workgroup_size(8, 8, 1)
fn downsample_mip2(@builtin(global_invocation_id) gid: vec3<u32>) {
    textureStore(w_spd_mip2, vec2<i32>(gid.xy), Box4RGFrom1(vec2<i32>(gid.xy)));
}

// Farthest-depth (r) -> farthest-depth mip 1 (r).
@compute @workgroup_size(8, 8, 1)
fn downsample_farthest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iPx = vec2<i32>(gid.xy);
    var acc = 0.0;
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            acc += textureLoad(t_downsample_src_depth, iPx * 2 + vec2<i32>(dx, dy), 0).r;
        }
    }
    textureStore(w_downsample_dst_r, iPx, vec4<f32>(acc * 0.25, 0.0, 0.0, 0.0));
}

fn Box4RG(iPxDst: vec2<i32>) -> vec4<f32> {
    var acc = vec2<f32>(0.0, 0.0);
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            acc += textureLoad(t_spd_mip0, iPxDst * 2 + vec2<i32>(dx, dy), 0).rg;
        }
    }
    return vec4<f32>(acc * 0.25, 0.0, 0.0);
}

fn Box4RGFrom1(iPxDst: vec2<i32>) -> vec4<f32> {
    var acc = vec2<f32>(0.0, 0.0);
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            acc += textureLoad(t_spd_mip1, iPxDst * 2 + vec2<i32>(dx, dy), 0).rg;
        }
    }
    return vec4<f32>(acc * 0.25, 0.0, 0.0);
}
