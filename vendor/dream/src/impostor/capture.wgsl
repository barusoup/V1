// Impostor capture: one orthographic view of the model in object space,
// writing the five channel attachments the asset keeps as array layers.
// The material bindings mirror model_pbr.wgsl set 2 exactly so the same
// PbrMaterialGpu descriptor sets drive both pipelines.

struct Factors {
    params: vec4<f32>,   // metallic, roughness, alpha_cutoff, alpha_mode
    params2: vec4<f32>,  // ao strength, normal scale, -, -
    emissive: vec4<f32>, // rgb, strength
    uv: array<vec4<f32>, 5>,   // base, mr, normal, ao, emissive: scale.xy, offset.xy
    tf: array<vec4<f32>, 5>,   // cos, sin, uv_set, -
};
@group(0) @binding(0) var<uniform> factors: Factors;
@group(0) @binding(1) var t_base: texture_2d<f32>;
@group(0) @binding(2) var s_base: sampler;
@group(0) @binding(3) var t_mr: texture_2d<f32>;
@group(0) @binding(4) var s_mr: sampler;
@group(0) @binding(5) var t_normal: texture_2d<f32>;
@group(0) @binding(6) var s_normal: sampler;
@group(0) @binding(7) var t_ao: texture_2d<f32>;
@group(0) @binding(8) var s_ao: sampler;
@group(0) @binding(9) var t_emissive: texture_2d<f32>;
@group(0) @binding(10) var s_emissive: sampler;

// set 1: base colour factor (same role as model_pbr's draw data, uniform so
// each material carries it through the bake).
struct Bake { base_color: vec4<f32> };
@group(1) @binding(0) var<uniform> bake: Bake;

struct Push {
    mvp: mat4x4<f32>,
    face_n: vec4<f32>,      // outward normal of the captured face (toward the camera)
    box_center: vec4<f32>,
    extents: vec4<f32>,     // ext_u, ext_v, ext_n, base_alpha
};
var<push_constant> push: Push;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,   // object space; the "world" is the model
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) tangent: vec4<f32>,
    @location(5) @interpolate(flat) material: u32,
};

struct FragOutput {
    @location(0) albedo: vec4<f32>,      // linear rgb, a = coverage
    @location(1) normal_ao: vec4<f32>,   // object-space normal *0.5+0.5, a = AO
    @location(2) mr: vec4<f32>,          // r = roughness, g = metalness
    @location(3) emissive: vec4<f32>,    // linear rgb
    @location(4) depth: f32,             // normalized distance into the box
};

// Per-map transform: scale, rotate, translate then pick the uv set.
fn map_uv(index: u32, uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    let base = select(uv0, uv1, factors.tf[index].z > 0.5);
    let rotated = mat2x2<f32>(
        vec2<f32>(factors.tf[index].x, -factors.tf[index].y),
        vec2<f32>(factors.tf[index].y, factors.tf[index].x),
    ) * (base * factors.uv[index].xy);
    return rotated + factors.uv[index].zw;
}

// Tangent frame: vertex tangents when present, else the derivative frame
// (same contract as model_pbr.wgsl).
fn tangent_frame(n: vec3<f32>, world_pos: vec3<f32>, uv: vec2<f32>, t: vec4<f32>) -> mat3x3<f32> {
    if dot(t.xyz, t.xyz) > 0.5 {
        return mat3x3<f32>(normalize(t.xyz), normalize(t.w * cross(n, t.xyz)), n);
    }
    let duv_dx = dpdx(uv);
    let duv_dy = dpdy(uv);
    let dp_dx = dpdx(world_pos);
    let dp_dy = dpdy(world_pos);
    let tangent = (dp_dx * duv_dy.y - dp_dy * duv_dx.y) / max(duv_dx.x * duv_dy.y - duv_dy.x * duv_dx.y, 1e-7);
    let t3 = normalize(tangent - n * dot(n, tangent));
    return mat3x3<f32>(t3, normalize(cross(n, t3)), n);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = vec4<f32>(input.position, 1.0);
    out.world_pos = pos.xyz;
    out.clip = push.mvp * pos;
    out.normal = input.normal;
    out.uv0 = input.uv0;
    out.uv1 = input.uv1;
    out.tangent = input.tangent;
    out.material = 0u;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> FragOutput {
    var out: FragOutput;

    let base = textureSample(t_base, s_base, map_uv(0u, input.uv0, input.uv1));
    let alpha = base.a * bake.base_color.a;
    // MASK: discard under the cutoff so coverage mirrors the material.
    if factors.params.w > 0.5 && factors.params.w < 1.5 && alpha < factors.params.z {
        discard;
    }

    // Normals always face the capture camera (same effect as
    // model_pbr.wgsl's front-face flip, but independent of viewport winding —
    // this pass uses a positive-height viewport while the scene uses a
    // negative one, so `front_facing` has the opposite meaning here).
    var n = normalize(input.normal);
    if dot(n, push.face_n.xyz) < 0.0 { n = -n; }
    let uv_n = map_uv(2u, input.uv0, input.uv1);
    let normal_tex = textureSample(t_normal, s_normal, uv_n).xyz * 2.0 - vec3<f32>(1.0);
    if dot(normal_tex, normal_tex) > 0.01 {
        let tbn = tangent_frame(n, input.world_pos, uv_n, input.tangent);
        n = normalize(tbn * (normal_tex * vec3<f32>(factors.params2.y, factors.params2.y, 1.0)));
    }

    let mr = textureSample(t_mr, s_mr, map_uv(1u, input.uv0, input.uv1));
    let ao_tex = textureSample(t_ao, s_ao, map_uv(3u, input.uv0, input.uv1)).r;
    let ao = mix(1.0, ao_tex, factors.params2.x);
    let em = textureSample(t_emissive, s_emissive, map_uv(4u, input.uv0, input.uv1)).rgb
        * factors.emissive.rgb * factors.emissive.w;

    // Normalized distance from the capture plane into the box along face_n.
    let d = (push.extents.z * 0.5 - dot(input.world_pos - push.box_center.xyz, push.face_n.xyz))
        / max(push.extents.z, 1e-6);

    out.albedo = vec4<f32>(base.rgb * bake.base_color.rgb, 1.0);
    out.normal_ao = vec4<f32>(n * 0.5 + vec3<f32>(0.5), ao);
    out.mr = vec4<f32>(mr.g * factors.params.y, mr.b * factors.params.x, 0.0, 1.0);
    out.emissive = vec4<f32>(em, 1.0);
    out.depth = clamp(d, 0.0, 1.0);
    return out;
}
