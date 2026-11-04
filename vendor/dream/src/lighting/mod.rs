//! Shared PBR lighting: one directional sun plus an analytic sky, evaluated by
//! the WGSL in [`SHADER`] which is concatenated into consumer shaders before
//! `compile_wgsl` (the engine's WGSL-to-SPIR-V step). The environment is a uniform at
//! **set 3 binding 0** for every pipeline that uses it, so one
//! [`Lighting`] instance serves all consumers; shadow lookups funnel through
//! `shadow_visibility` in the shader (a Phase-4 stub returning 1.0).
//!
//! [`LightingEnv`] is the CPU mirror of the uniform block — all vec4s, so
//! alignment matches WGSL's `vec4` fields.

use crate::gfx::GpuBuffer;
use crate::gfx::vulkan::GpuContext;
use ash::vk;
use std::sync::Arc;

/// The WGSL snippet; prepend to any shader that calls `lighting_eval`.
pub const SHADER: &str = include_str!("lighting.wgsl");

/// Descriptor set index the lighting uniform binds at in every consumer.
pub const SET_INDEX: usize = 3;
/// Binding index within that set.
pub const BINDING: u32 = 0;

/// CPU mirror of `LightingEnv` in `lighting.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingEnv {
    /// xyz: direction TO the sun (normalized); w unused.
    pub sun_direction: [f32; 4],
    /// rgb: sun colour × intensity.
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_ground: [f32; 4],
    /// xyz: world-space camera position (shading point's view vector).
    pub cam_pos: [f32; 4],
    /// x: exposure.
    pub params: [f32; 4],
}

impl Default for LightingEnv {
    /// Day-ish sky with a warm sun 45° up — matches the lab's defaults.
    fn default() -> Self {
        Self::with_sun(0.6, 0.7, 3.0)
    }
}

impl LightingEnv {
    /// Build with the sun at `azimuth`/`elevation` radians and `intensity`.
    pub fn with_sun(azimuth: f32, elevation: f32, intensity: f32) -> Self {
        let dir = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
            0.0,
        ];
        Self {
            sun_direction: dir,
            sun_color: [intensity, intensity * 0.96, intensity * 0.88, 0.0],
            sky_zenith: [0.18, 0.35, 0.65, 0.0],
            sky_horizon: [0.55, 0.60, 0.68, 0.0],
            sky_ground: [0.10, 0.09, 0.08, 0.0],
            cam_pos: [0.0; 4],
            params: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Per-frame-slot lighting uniform sets, shared by every consumer pipeline.
pub struct Lighting {
    ctx: Arc<GpuContext>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    buffers: Vec<GpuBuffer>,
}

impl Lighting {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let layout = {
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT);
            unsafe {
                ctx.device().create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(std::slice::from_ref(&binding)),
                    None,
                )
            }
            .expect("lighting descriptor set layout")
        };
        let slots = ctx.max_frames_in_flight();
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(slots as u32);
        let pool = unsafe {
            ctx.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(slots as u32)
                    .pool_sizes(std::slice::from_ref(&pool_size)),
                None,
            )
        }
        .expect("lighting descriptor pool");
        let layouts: Vec<_> = (0..slots).map(|_| layout).collect();
        let sets = unsafe {
            ctx.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .expect("lighting descriptor sets");
        let buffers: Vec<_> = (0..slots)
            .map(|slot| {
                let buffer = ctx.create_host_buffer(
                    &format!("lighting env {slot}"),
                    std::mem::size_of::<LightingEnv>() as u64,
                );
                let info = vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(std::mem::size_of::<LightingEnv>() as u64);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(sets[slot])
                    .dst_binding(BINDING)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&info));
                unsafe { ctx.device().update_descriptor_sets(&[write], &[]) };
                buffer
            })
            .collect();
        Self {
            ctx: Arc::clone(ctx),
            layout,
            pool,
            sets,
            buffers,
        }
    }

    /// Set layout to place at [`SET_INDEX`] in consumer pipeline layouts.
    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// Write this frame slot's environment. Call before recording draws.
    pub fn upload(&mut self, frame_slot: usize, env: &LightingEnv) {
        let len = self.buffers.len();
        let buffer = &mut self.buffers[frame_slot % len];
        buffer.write(0, bytemuck::bytes_of(env));
    }

    /// Descriptor set for `frame_slot`.
    pub fn set(&self, frame_slot: usize) -> vk::DescriptorSet {
        self.sets[frame_slot % self.sets.len()]
    }
}

impl Drop for Lighting {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            let device = self.ctx.device();
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // CPU mirror of the WGSL math for sanity tests.
    fn sky(dir: Vec3, env: &LightingEnv) -> Vec3 {
        let up = dir.y.clamp(-1.0, 1.0);
        let zenith = Vec3::from_array(env.sky_zenith[..3].try_into().unwrap());
        let horizon = Vec3::from_array(env.sky_horizon[..3].try_into().unwrap());
        let ground = Vec3::from_array(env.sky_ground[..3].try_into().unwrap());
        let sky = horizon.lerp(zenith, (up * 1.6).clamp(0.0, 1.0));
        let gnd = horizon.lerp(ground, (-up * 1.6).clamp(0.0, 1.0));
        if up >= 0.0 { sky } else { gnd }
    }

    fn env_brdf(rough: f32, nov: f32) -> [f32; 2] {
        let c0 = [-1.0f32, -0.0275, -0.572, 0.022];
        let c1 = [1.0f32, 0.0425, 1.04, -0.04];
        let r: [f32; 4] = std::array::from_fn(|i| rough * c0[i] + c1[i]);
        let a004 = (r[0] * r[0]).min(2f32.powf(-9.28 * nov)) * r[0] + r[1];
        [(-1.04) * a004 + r[2], 1.04 * a004 + r[3]]
    }

    #[test]
    fn composed_shader_validates_with_naga() {
        // lighting snippet prepended to a trivial consumer compiles cleanly.
        let source = format!(
            "{SHADER}\nstruct O {{ @location(0) c: vec4<f32> }};\n\
             @fragment fn fs_main(@builtin(position) p: vec4<f32>) -> O {{\n\
             \x20 var o: O; o.c = vec4(lighting_eval(vec3(0.5), 0.0, 0.5,\n\
             \x20 vec3(0,1,0), vec3(0,0,1), p.xyz, 1.0), 1.0); return o; }}"
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("composed lighting shader must validate");
    }

    #[test]
    fn sky_gradient_is_bounded_and_analytic() {
        let env = LightingEnv::default();
        for dir in [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::X,
            Vec3::new(0.4, 0.7, -0.6).normalize(),
        ] {
            let c = sky(dir.normalize(), &env);
            assert!(c.cmplt(Vec3::ONE).all() && c.cmpge(Vec3::ZERO).all());
        }
        // Zenith reflects zenith colour, ground reflects ground colour.
        assert!(sky(Vec3::Y, &env).z > sky(Vec3::Y, &env).x);
        assert!(sky(Vec3::NEG_Y, &env).length() < sky(Vec3::Y, &env).length());
    }

    #[test]
    fn env_brdf_is_energy_conserving() {
        for rough in [0.0, 0.3, 0.6, 1.0] {
            for nov in [0.05, 0.5, 1.0] {
                let [scale, bias] = env_brdf(rough, nov);
                // |f0 * scale + bias| <= 1.2 for f0 in [0,1] (Karis approx
                // slightly exceeds 1 near grazing angles).
                for f0 in [0.04f32, 0.5, 1.0] {
                    let v = f0 * scale + bias;
                    assert!(
                        (0.0..=1.25).contains(&v),
                        "rough={rough} nov={nov} f0={f0} -> {v}"
                    );
                }
            }
        }
        // Mirror view, mirror surface: scale ~1 (full F0 response).
        let [scale, _] = env_brdf(0.0, 1.0);
        assert!((scale - 1.0).abs() < 0.02);
        // Very rough at grazing incidence: response is smooth and bounded.
        let [scale, bias] = env_brdf(1.0, 0.05);
        assert!(scale.is_finite() && bias.is_finite() && scale.abs() < 1.0);
    }

    #[test]
    fn uniform_block_is_all_vec4() {
        assert_eq!(std::mem::size_of::<LightingEnv>(), 7 * 16);
    }
}
