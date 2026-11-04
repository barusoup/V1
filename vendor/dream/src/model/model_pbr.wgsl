// PBR model forward pass — static meshes with full glTF material maps,
// shaded by the shared lighting module (prepended at compile time by
// ModelRenderer; it provides lighting_env at set 3 and lighting_eval()).
//
// Same contracts as model.wgsl: set 0 engine camera block, velocity =
// prev_uv - curr_uv in offscreen texture space; push constants carry
// model + prev_model. Set 1 is the per-draw block (colour/opacity), set 2 the
// material (factors uniform + five map/sampler pairs), set 3 lighting.

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
    joints: array<Joint>,
};
@group(1) @binding(0) var<storage, read> draw_data: DrawData;

// Per-material factors and per-map KHR_texture_transform data.
struct MatFactors {
    // x metallic, y roughness, z alpha_cutoff, w alpha_mode (0 opaque 1 mask 2 blend)
    params: vec4<f32>,
    // x ao_strength, y normal_scale, z unused, w unused
    params2: vec4<f32>,
    // rgb emissive factor, a emissive strength
    emissive: vec4<f32>,
    // per map {base, mr, normal, ao, emissive}: (scale.xy, offset.xy)
    uv0: vec4<f32>, uv1: vec4<f32>, uv2: vec4<f32>, uv3: vec4<f32>, uv4: vec4<f32>,
    // per map: (cos rot, sin rot, uv set, unused)
    tf0: vec4<f32>, tf1: vec4<f32>, tf2: vec4<f32>, tf3: vec4<f32>, tf4: vec4<f32>,
};
@group(2) @binding(0) var<uniform> material: MatFactors;
@group(2) @binding(1) var t_base: texture_2d<f32>;
@group(2) @binding(2) var s_base: sampler;
@group(2) @binding(3) var t_mr: texture_2d<f32>;
@group(2) @binding(4) var s_mr: sampler;
@group(2) @binding(5) var t_normal: texture_2d<f32>;
@group(2) @binding(6) var s_normal: sampler;
@group(2) @binding(7) var t_ao: texture_2d<f32>;
@group(2) @binding(8) var s_ao: sampler;
@group(2) @binding(9) var t_emissive: texture_2d<f32>;
@group(2) @binding(10) var s_emissive: sampler;

struct Push {
    model: mat4x4<f32>,
    prev_model: mat4x4<f32>,
    // x: channel-debug code shared with impostor.wgsl
    // (10 albedo, 11 rough, 12 metal, 13 ao, 14 emissive, 15 world normal)
    debug: vec4<f32>,
};
var<push_constant> push: Push;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv0: vec2<f32>,
    @location(1) uv1: vec2<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_tangent: vec4<f32>,  // xyz tangent, w bitangent sign
    @location(4) world_pos: vec3<f32>,
    @location(5) prev_world: vec3<f32>,
    // sign of det(model) — mirrors flip which face is logically front.
    @location(6) det_sign: f32,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
};

fn screen_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5);
}

fn velocity_of(world: vec4<f32>, prev_world: vec4<f32>) -> vec2<f32> {
    let curr_clip = camera.curr_view_projection * world;
    let prev_clip = camera.prev_view_projection * prev_world;
    return screen_uv(prev_clip.xy / prev_clip.w) - screen_uv(curr_clip.xy / curr_clip.w);
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) tangent: vec4<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    let world = push.model * vec4<f32>(position, 1.0);
    let prev_world = push.prev_model * vec4<f32>(position, 1.0);
    output.clip_position = camera.view_projection * world;
    output.uv0 = uv0;
    output.uv1 = uv1;
    // Normals transform by the inverse-transpose (cofactor matrix) — the
    // plain model matrix skews them under non-uniform scale/shear, which the
    // source bake already corrects. The determinant is evaluated on the
    // unit basis: cofactor cols are cross(nb,nc)/sa etc., det is the
    // unit-scale det — valid to the full 1e-6..1e6 scale contract.
    let a = push.model[0].xyz;
    let b = push.model[1].xyz;
    let c = push.model[2].xyz;
    let sa = max(length(a), 1e-30);
    let sb = max(length(b), 1e-30);
    let sc = max(length(c), 1e-30);
    let na = a / sa;
    let nb = b / sb;
    let nc = c / sc;
    let det_u = dot(na, cross(nb, nc));
    let det_ok = abs(det_u) > 1e-8;
    let inv_d = select(0.0, 1.0 / det_u, det_ok);
    let nmat = mat3x3<f32>(
        (cross(nb, nc) / sa) * inv_d,
        (cross(nc, na) / sb) * inv_d,
        (cross(na, nb) / sc) * inv_d,
    );
    let n_raw = nmat * normal;
    output.world_normal =
        normalize(select(vec3<f32>(0.0, 1.0, 0.0), n_raw, det_ok));
    output.world_tangent =
        vec4<f32>((push.model * vec4<f32>(tangent.xyz, 0.0)).xyz, tangent.w * sign(det_u));
    output.det_sign = sign(det_u);
    output.world_pos = world.xyz;
    // Velocity is computed in the fragment: perspective division makes the
    // screen-space delta non-linear, so it can't be interpolated.
    output.prev_world = prev_world.xyz;
    return output;
}

// Apply KHR_texture_transform: pick the UV set then uv' = R(rot) * (uv*scale) + offset.
fn map_uv(uv_pair: vec4<f32>, tf: vec4<f32>, uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    var uv = uv0;
    if tf.z > 0.5 { uv = uv1; }
    let u = uv * uv_pair.xy;
    return vec2<f32>(tf.x * u.x - tf.y * u.y, tf.y * u.x + tf.x * u.y) + uv_pair.zw;
}

// Tangent frame from the interpolated attribute when present, else a
// cotangent frame built from screen-space derivatives (standard when the
// file omits tangents).
fn tangent_frame(input: VertexOutput, uv: vec2<f32>, n: vec3<f32>) -> mat3x3<f32> {
    if dot(input.world_tangent.xyz, input.world_tangent.xyz) > 1e-8 {
        // Gram-Schmidt the interpolated tangent against the (already
        // inverse-transposed) normal so the frame stays orthonormal.
        let t0 = input.world_tangent.xyz;
        let t = normalize(t0 - n * dot(n, t0));
        let b = cross(n, t) * input.world_tangent.w;
        return mat3x3<f32>(t, normalize(b), n);
    }
    let dpdx = dpdx(input.world_pos);
    let dpdy = dpdy(input.world_pos);
    let duvdx = dpdx(uv);
    let duvdy = dpdy(uv);
    let det = duvdx.x * duvdy.y - duvdy.x * duvdx.y;
    let sc = select(-1.0, 1.0, det < 0.0);
    let t = normalize((dpdx * duvdy.y - dpdy * duvdx.y) * sc);
    let b = normalize((-dpdx * duvdy.x + dpdy * duvdx.x) * sc);
    return mat3x3<f32>(t, b, n);
}

@fragment
fn fs_main(
    input: VertexOutput,
    @builtin(front_facing) front_facing: bool,
) -> FragmentOutput {
    var output: FragmentOutput;

    let uv_base = map_uv(material.uv0, material.tf0, input.uv0, input.uv1);
    let uv_mr = map_uv(material.uv1, material.tf1, input.uv0, input.uv1);
    let uv_nrm = map_uv(material.uv2, material.tf2, input.uv0, input.uv1);
    let uv_ao = map_uv(material.uv3, material.tf3, input.uv0, input.uv1);
    let uv_emi = map_uv(material.uv4, material.tf4, input.uv0, input.uv1);

    let base_tex = textureSample(t_base, s_base, uv_base);
    let albedo = base_tex.rgb * draw_data.base_color.rgb;
    // draw_data.base_color.a is the CROSSFADE opacity (draw_pbr_with_opacity
    // writes it directly); the material's own alpha factor rides params2.w.
    let source_alpha = base_tex.a * material.params2.w;
    if material.params.w == 1.0 && source_alpha < material.params.z {
        discard;
    }

    var n = normalize(input.world_normal);
    // Mirrored transforms (det<0) reverse triangle winding — the logical
    // front face is the physical back face.
    let logical_front = select(front_facing, !front_facing, input.det_sign < 0.0);
    if !logical_front { n = -n; }
    let nmap = textureSample(t_normal, s_normal, uv_nrm).xyz * 2.0 - 1.0;
    let tbn = tangent_frame(input, uv_nrm, n);
    n = normalize(tbn * (nmap * vec3<f32>(material.params2.y, material.params2.y, 1.0)));

    let mr_tex = textureSample(t_mr, s_mr, uv_mr);
    let roughness = mr_tex.g * material.params.y;
    let metallic = mr_tex.b * material.params.x;
    let ao = mix(1.0, textureSample(t_ao, s_ao, uv_ao).r, material.params2.x);
    let emissive = textureSample(t_emissive, s_emissive, uv_emi).rgb
        * material.emissive.rgb * material.emissive.a;

    let dbg = u32(push.debug.x);
    if dbg >= 10u {
        var ch = albedo;
        if dbg == 11u { ch = vec3<f32>(roughness); }
        if dbg == 12u { ch = vec3<f32>(metallic); }
        if dbg == 13u { ch = vec3<f32>(ao); }
        if dbg == 14u { ch = emissive; }
        if dbg == 15u { ch = vec3<f32>(n.x, n.y, n.z) * 0.5 + vec3<f32>(0.5); }
        if dbg == 16u { ch = input.world_pos * 0.25 + vec3<f32>(0.5); }
        output.color = vec4<f32>(ch, 1.0);
        output.velocity = vec2<f32>(0.0);
        return output;
    }

    let v = normalize(lighting_env.cam_pos.xyz - input.world_pos);
    let lit = lighting_eval(albedo, metallic, roughness, n, v, input.world_pos, ao);
    // OPAQUE/MASK survivors carry only the crossfade opacity; BLEND keeps
    // the real source alpha under it.
    let out_alpha =
        draw_data.base_color.a * select(1.0, source_alpha, material.params.w > 1.5);
    output.color = vec4<f32>(lit + emissive, out_alpha);
    output.velocity = velocity_of(
        vec4<f32>(input.world_pos, 1.0),
        vec4<f32>(input.prev_world, 1.0),
    );
    return output;
}
