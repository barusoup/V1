// Dream app scene shader.
//
// The engine's scene-pass contract: camera block at set 0 (jittered render
// projection plus the unjittered pair for motion vectors), colour and
// velocity attachments, linear-light colour out.
//
// One addition: a screen-space path for the themed sky. Its vertices are
// marked with a sentinel `position.x` and carry NDC in `(y, z)`, so the
// app can paint a flat gradient clear colour (a static themed backdrop —
// not procedural art) without a second pipeline. The quads sit just inside
// the far plane, so depth testing still lets nearer geometry through, and
// they report zero velocity: a screen-locked backdrop reprojects exactly.

struct Camera {
    view_projection: mat4x4<f32>,
    curr_view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
};
struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
};

const SKY_SENTINEL: f32 = 1.0e30;

fn screen_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    if input.position.x >= SKY_SENTINEL {
        // Screen-locked sky: NDC packed into (y, z), pushed just inside
        // the far plane so it never occludes nearer geometry.
        output.clip_position = vec4<f32>(input.position.y, input.position.z, 0.99999, 1.0);
        output.color = input.color;
        output.velocity = vec2<f32>(0.0, 0.0);
        return output;
    }
    let world = vec4<f32>(input.position, 1.0);
    output.clip_position = camera.view_projection * world;
    let curr_clip = camera.curr_view_projection * world;
    let prev_clip = camera.prev_view_projection * world;
    let curr_uv = screen_uv(curr_clip.xy / curr_clip.w);
    let prev_uv = screen_uv(prev_clip.xy / prev_clip.w);
    output.velocity = prev_uv - curr_uv;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;
    output.color = vec4<f32>(input.color);
    output.velocity = input.velocity;
    return output;
}
