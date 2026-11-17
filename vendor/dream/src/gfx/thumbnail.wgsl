// Scene card thumbnail: box-downsamples a centre-cropped square of the linear
// scene colour target into sRGB-encoded RGBA8 texels, one invocation per texel.
//
// `crop` is the crop origin in render texels in the *scene target's* row order,
// which differs per backend: Vulkan's scene pass writes bottom-up rows (its
// negative-height viewport flips y), so `main` mirrors each row as it reads it.
// Metal writes top-first rows like its own framebuffer, so `main_topdown`
// reads them straight. Both emit output row 0 = the top of the crop.

struct Params {
    crop: vec2<i32>,
    extent: vec2<i32>,
    side: i32,
    size: i32,
}

var<push_constant> params: Params;
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_px: array<u32>;

/// One scene texel, mirrored into Vulkan's bottom-up row order.
fn texel(x: i32, y: i32) -> vec3<f32> {
    let sx = clamp(params.crop.x + x, 0, params.extent.x - 1);
    let sy = clamp(params.extent.y - 1 - (params.crop.y + y), 0, params.extent.y - 1);
    return textureLoad(scene, vec2<i32>(sx, sy), 0).rgb;
}

/// One scene texel straight from a top-first target (Metal).
fn texel_topdown(x: i32, y: i32) -> vec3<f32> {
    let sx = clamp(params.crop.x + x, 0, params.extent.x - 1);
    let sy = clamp(params.crop.y + y, 0, params.extent.y - 1);
    return textureLoad(scene, vec2<i32>(sx, sy), 0).rgb;
}

fn srgb_encode(value: f32) -> f32 {
    let v = clamp(value, 0.0, 1.0);
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn capture(id: vec3<u32>, flip_y: bool) {
    if i32(id.x) >= params.size || i32(id.y) >= params.size {
        return;
    }
    let side = f32(params.side);
    let size = f32(params.size);
    let fx0 = f32(id.x) * side / size;
    let fx1 = f32(id.x + 1u) * side / size;
    let fy0 = f32(id.y) * side / size;
    let fy1 = f32(id.y + 1u) * side / size;
    let ix0 = clamp(i32(floor(fx0)), 0, params.side - 1);
    let ix1 = clamp(max(i32(ceil(fx1)), i32(floor(fx0)) + 1), 1, params.side);
    let iy0 = clamp(i32(floor(fy0)), 0, params.side - 1);
    let iy1 = clamp(max(i32(ceil(fy1)), i32(floor(fy0)) + 1), 1, params.side);

    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var iy = iy0; iy < iy1; iy += 1) {
        for (var ix = ix0; ix < ix1; ix += 1) {
            if flip_y {
                sum += texel(ix, iy);
            } else {
                sum += texel_topdown(ix, iy);
            }
            n += 1.0;
        }
    }
    let avg = sum / max(n, 1.0);
    let enc = vec3<f32>(srgb_encode(avg.r), srgb_encode(avg.g), srgb_encode(avg.b));
    let px = vec3<u32>(min(enc * 255.0 + 0.5, vec3<f32>(255.0)));
    out_px[u32(params.size) * id.y + id.x] = px.r | (px.g << 8u) | (px.b << 16u) | 0xff000000u;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    capture(id, true);
}

@compute @workgroup_size(8, 8)
fn main_topdown(@builtin(global_invocation_id) id: vec3<u32>) {
    capture(id, false);
}
