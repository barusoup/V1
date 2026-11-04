// Model forward pass: static and skinned meshes with per-object motion
// vectors.
//
// Contract (matches terrain.wgsl / packed.wgsl so FSR reprojects every
// scene-pass target uniformly):
// - set 0 is the engine camera block: jittered view_projection for the
//   rendered position, unjittered curr/prev for motion.
// - velocity = prev_uv - curr_uv in the offscreen targets' texture space
//   (that space is bottom-up; the sign here is deliberate).
// - Per-object motion uses the previous model matrix and previous joint
//   palette, so moving or animating geometry vectors are correct, not just
//   camera motion.

struct Camera {
    view_projection: mat4x4<f32>,
    curr_view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Joint {
    curr: mat4x4<f32>,
    prev: mat4x4<f32>,
};
struct DrawData {
    base_color: vec4<f32>,
    // Runtime-sized: skinned draws bind as many joints as they need.
    joints: array<Joint>,
};
@group(1) @binding(0) var<storage, read> draw_data: DrawData;

@group(2) @binding(0) var t_base: texture_2d<f32>;
@group(2) @binding(1) var s_base: sampler;

struct Push {
    model: mat4x4<f32>,
    prev_model: mat4x4<f32>,
};
var<push_constant> push: Push;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) velocity: vec2<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
};

// Texture-space uv of the offscreen targets; see terrain.wgsl for why the
// sign is what it is.
fn screen_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5);
}

fn velocity_of(world: vec4<f32>, prev_world: vec4<f32>) -> vec2<f32> {
    let curr_clip = camera.curr_view_projection * world;
    let prev_clip = camera.prev_view_projection * prev_world;
    let curr_uv = screen_uv(curr_clip.xy / curr_clip.w);
    let prev_uv = screen_uv(prev_clip.xy / prev_clip.w);
    return prev_uv - curr_uv;
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    let world = push.model * vec4<f32>(position, 1.0);
    let prev_world = push.prev_model * vec4<f32>(position, 1.0);
    output.clip_position = camera.view_projection * world;
    output.uv = uv;
    output.normal = (push.model * vec4<f32>(normal, 0.0)).xyz;
    output.velocity = velocity_of(world, prev_world);
    return output;
}

@vertex
fn vs_skin(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    // Four-joint linear blend skinning; weights are normalized at load.
    var skin_curr = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var skin_prev = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    skin_curr += weights.x * draw_data.joints[joints.x].curr;
    skin_curr += weights.y * draw_data.joints[joints.y].curr;
    skin_curr += weights.z * draw_data.joints[joints.z].curr;
    skin_curr += weights.w * draw_data.joints[joints.w].curr;
    skin_prev += weights.x * draw_data.joints[joints.x].prev;
    skin_prev += weights.y * draw_data.joints[joints.y].prev;
    skin_prev += weights.z * draw_data.joints[joints.z].prev;
    skin_prev += weights.w * draw_data.joints[joints.w].prev;

    let world = push.model * skin_curr * vec4<f32>(position, 1.0);
    let prev_world = push.prev_model * skin_prev * vec4<f32>(position, 1.0);
    output.clip_position = camera.view_projection * world;
    output.uv = uv;
    output.normal = (push.model * skin_curr * vec4<f32>(normal, 0.0)).xyz;
    output.velocity = velocity_of(world, prev_world);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;
    let base = textureSample(t_base, s_base, input.uv);
    // Simple directional wrap-lighting so unlit models read as 3D without a
    // material system; games replace this with their own pipeline.
    let light = normalize(vec3<f32>(0.35, 0.9, 0.25));
    let n = normalize(input.normal);
    let diffuse = 0.65 + 0.35 * max(dot(n, light), 0.0);
    output.color = vec4<f32>(base.rgb * draw_data.base_color.rgb * diffuse, base.a * draw_data.base_color.a);
    output.velocity = input.velocity;
    return output;
}
