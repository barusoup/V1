// Shared PBR lighting — function-only snippet, concatenated by the Rust side
// (see crate::lighting) ahead of the shader that consumes it. Contract: the
// environment uniform lives at set 3 binding 0 for every pipeline that uses
// this module, so one descriptor set serves models and impostors alike.

struct LightingEnv {
    sun_direction: vec4<f32>,   // xyz: direction TO the sun (normalized)
    sun_color: vec4<f32>,       // rgb: colour * intensity
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    sky_ground: vec4<f32>,
    cam_pos: vec4<f32>,         // xyz: world-space camera position
    params: vec4<f32>,          // x: exposure
};
@group(3) @binding(0) var<uniform> lighting_env: LightingEnv;

const PI: f32 = 3.14159265359;

// Phase-4 hook: one shadow map lives here later; stubbed fully lit.
fn shadow_visibility(world_pos: vec3<f32>) -> f32 {
    return 1.0;
}

// Analytic sky: gradient hemisphere. dir is world-space (unit).
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y, -1.0, 1.0);
    let sky = mix(lighting_env.sky_horizon.rgb, lighting_env.sky_zenith.rgb,
        clamp(up * 1.6, 0.0, 1.0));
    let ground = mix(lighting_env.sky_horizon.rgb, lighting_env.sky_ground.rgb,
        clamp(-up * 1.6, 0.0, 1.0));
    return select(ground, sky, up >= 0.0);
}

// Hemisphere average for rough reflections.
fn sky_average() -> vec3<f32> {
    return (lighting_env.sky_zenith.rgb + lighting_env.sky_horizon.rgb * 2.0
        + lighting_env.sky_ground.rgb) * 0.25;
}

fn f_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn d_ggx(noh: f32, rough: f32) -> f32 {
    let a = max(rough * rough, 1e-4);
    let a2 = a * a;
    let d = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith height-correlated G2/GGX (Schlick approximation).
fn g_smith(nov: f32, nol: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let ggx_l = nol / (nol * (1.0 - a) + a);
    let ggx_v = nov / (nov * (1.0 - a) + a);
    return ggx_l * ggx_v / (4.0 * nov * nol + 1e-6);
}

// Karis's analytic EnvBRDF split-sum approximation.
fn env_brdf(rough: f32, nov: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// Full BRDF for one point: sun + analytic sky (diffuse irradiance, split-sum
// specular). n, v are world-space and normalized; ao multiplies ambient only.
fn lighting_eval(
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    ao: f32,
) -> vec3<f32> {
    let rough = clamp(roughness, 0.02, 1.0);
    let metal = clamp(metallic, 0.0, 1.0);
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metal);
    let l = normalize(lighting_env.sun_direction.xyz);
    let h = normalize(v + l);
    let nov = clamp(dot(n, v), 1e-4, 1.0);
    let nol = clamp(dot(n, l), 0.0, 1.0);
    let noh = clamp(dot(n, h), 0.0, 1.0);

    var color = vec3<f32>(0.0);
    if nol > 0.0 {
        let f = f_schlick(f0, clamp(dot(h, v), 0.0, 1.0));
        let spec = d_ggx(noh, rough) * g_smith(nov, nol, rough) * f;
        let diffuse = albedo * (vec3<f32>(1.0) - f) * (1.0 - metal) / PI;
        color += (diffuse + spec) * lighting_env.sun_color.rgb * nol
            * shadow_visibility(world_pos);
    }

    // Ambient: irradiance approximated by the sky gradient at the normal;
    // specular by a roughness-blurred lookup of the reflection direction.
    let ambient_diffuse = sky_radiance(n) * albedo * (1.0 - metal);
    let reflect_dir = reflect(-v, n);
    let ambient_spec = mix(sky_radiance(reflect_dir), sky_average(),
        rough * rough * 0.8);
    let ab = env_brdf(rough, nov);
    let ambient_specular = ambient_spec * (f0 * ab.x + ab.y);
    color += (ambient_diffuse + ambient_specular) * ao;
    return color * lighting_env.params.x;
}
