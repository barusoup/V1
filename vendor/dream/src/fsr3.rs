//! AMD FidelityFX Super Resolution 3.1 temporal upscaler (our own port).
//!
//! This module owns every internal surface the algorithm needs, one compute
//! pipeline per pass, and the dispatch chain. It is a hand port of AMD's MIT
//! licensed FSR 3.1 GPU code to WGSL for Dream's Vulkan renderer; see
//! `fsr3_common.wgsl` for the porting notes and deviations.
//!
//! Pass order matches `ffx_fsr3upscaler.cpp`:
//!   prepare inputs -> farthest-depth mip -> shading-change pyramid ->
//!   shading change -> prepare reactivity -> luma instability -> accumulate ->
//!   RCAS.
//!
//! The single-pass downsampler is replaced by a plain 2x2 box pyramid (its
//! reduction is the same box average) and auto exposure is not used: this
//! pipeline feeds a fixed exposure of 1 and no exposure texture.

use crate::camera::Camera;

pub use crate::temporal::JitterSequence;

mod pipeline;
mod targets;

use crate::gfx::{GpuBuffer, GpuContext};
use ash::vk;
use pipeline::{
    barrier, create_compute_pipelines, descriptor_bindings, image, image_binding, sampler_write,
    storage_buffer, storage_image, uniform,
};
use std::sync::Arc;
use targets::FsrTargets;

const FMT_RG16F: vk::Format = vk::Format::R16G16_SFLOAT;
const FMT_R32F: vk::Format = vk::Format::R32_SFLOAT;
const FMT_R16F: vk::Format = vk::Format::R16_SFLOAT;
const FMT_RGBA16F: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const FMT_RGBA8: vk::Format = vk::Format::R8G8B8A8_UNORM;
const FMT_R8: vk::Format = vk::Format::R8_UNORM;

const PRELUDE: &str = include_str!("fsr3_common.wgsl");

fn pass_source(pass: &str) -> String {
    format!("{PRELUDE}\n{pass}")
}

/// Uniform block shared by every pass. Field order and offsets match
/// `Fsr3Constants` in `fsr3_common.wgsl` (and AMD's `Fsr3UpscalerConstants`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Fsr3Constants {
    render_size: [i32; 2],
    previous_frame_render_size: [i32; 2],
    upscale_size: [i32; 2],
    previous_frame_upscale_size: [i32; 2],
    max_render_size: [i32; 2],
    max_upscale_size: [i32; 2],
    device_to_view_depth: [f32; 4],
    jitter_offset: [f32; 2],
    previous_frame_jitter_offset: [f32; 2],
    motion_vector_scale: [f32; 2],
    downscale_factor: [f32; 2],
    motion_vector_jitter_cancellation: [f32; 2],
    tan_half_fov: f32,
    jitter_phase_count: f32,
    delta_time: f32,
    delta_pre_exposure: f32,
    view_space_to_meters_factor: f32,
    frame_index: f32,
    velocity_factor: f32,
    reactiveness_scale: f32,
    shading_change_scale: f32,
    accumulation_added_per_frame: f32,
    min_disocclusion_accumulation: f32,
    _padding: [f32; 3],
}

const _: () = assert!(std::mem::size_of::<Fsr3Constants>() == 160);

pub struct Fsr3 {
    ctx: Arc<GpuContext>,
    layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    /// One set per frame slot: writing descriptors for the slot whose fence
    /// just recycled is safe, while the other slot may still be in flight.
    sets: Vec<vk::DescriptorSet>,
    pipelines: Pipelines,
    targets: FsrTargets,
    reconstructed: GpuBuffer,
    constants: Vec<GpuBuffer>,
    rcas: GpuBuffer,
    linear_sampler: vk::Sampler,
    point_sampler: vk::Sampler,
    render_size: (u32, u32),
    display_size: (u32, u32),
    prev_render_size: (u32, u32),
    prev_display_size: (u32, u32),
    prev_jitter: [f32; 2],
    frame_index: f32,
    ping: usize,
    inputs: [vk::ImageView; 3],
}

#[allow(clippy::struct_field_names)]
struct Pipelines {
    prepare_inputs: vk::Pipeline,
    shading_change: vk::Pipeline,
    downsample_mip1: vk::Pipeline,
    downsample_mip2: vk::Pipeline,
    downsample_farthest: vk::Pipeline,
    shading_change_pass: vk::Pipeline,
    prepare_reactivity: vk::Pipeline,
    luma_instability: vk::Pipeline,
    accumulate: vk::Pipeline,
    rcas: vk::Pipeline,
}

impl Fsr3 {
    pub fn new(
        ctx: &Arc<GpuContext>,
        render_size: (u32, u32),
        display_size: (u32, u32),
        inputs: [vk::ImageView; 3],
    ) -> Self {
        let targets = FsrTargets::build(ctx, render_size, display_size);
        let reconstructed = ctx.create_storage_buffer(
            "fsr reconstructed depth",
            (render_size.0 as u64) * (render_size.1 as u64) * 4,
        );
        let frame_slots = ctx.max_frames_in_flight();
        let constants: Vec<_> = (0..frame_slots)
            .map(|_| {
                ctx.create_host_buffer("fsr constants", std::mem::size_of::<Fsr3Constants>() as u64)
            })
            .collect();
        let rcas = ctx.create_host_buffer("fsr rcas", 16);
        let linear_sampler = ctx.create_sampler("fsr linear");
        let point_sampler = ctx.create_sampler("fsr point");

        let bindings = descriptor_bindings();
        let dsl = {
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe { ctx.device().create_descriptor_set_layout(&info, None) }
                .expect("fsr3 descriptor set layout")
        };
        let layout = {
            let sets = [dsl];
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
            unsafe { ctx.device().create_pipeline_layout(&info, None) }
                .expect("fsr3 pipeline layout")
        };

        let pool = {
            let slots = frame_slots as u32;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(2 * slots),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(2 * slots),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(20 * slots),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(16 * slots),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(2 * slots),
            ];
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(slots)
                .pool_sizes(&sizes);
            unsafe { ctx.device().create_descriptor_pool(&info, None) }
                .expect("fsr3 descriptor pool")
        };
        let layouts: Vec<_> = (0..frame_slots).map(|_| dsl).collect();
        let sets = unsafe {
            ctx.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .expect("fsr3 descriptor sets");

        let modules = [
            (
                "prepare_inputs",
                pass_source(include_str!("fsr3_prepare_inputs.wgsl")),
            ),
            (
                "shading_change",
                pass_source(include_str!("fsr3_shading_change.wgsl")),
            ),
            (
                "downsample",
                pass_source(include_str!("fsr3_downsample.wgsl")),
            ),
            (
                "shading_change_pass",
                pass_source(include_str!("fsr3_shading_change_pass.wgsl")),
            ),
            (
                "prepare_reactivity",
                pass_source(include_str!("fsr3_prepare_reactivity.wgsl")),
            ),
            (
                "luma_instability",
                pass_source(include_str!("fsr3_luma_instability.wgsl")),
            ),
            (
                "accumulate",
                pass_source(include_str!("fsr3_accumulate.wgsl")),
            ),
            ("rcas", pass_source(include_str!("fsr3_rcas.wgsl"))),
        ];
        let m_prepare = ctx.compile_shader(&modules[0].1, "fsr3 prepare inputs");
        let m_shading = ctx.compile_shader(&modules[1].1, "fsr3 shading change");
        let m_downsample = ctx.compile_shader(&modules[2].1, "fsr3 downsample");
        let m_shading_pass = ctx.compile_shader(&modules[3].1, "fsr3 shading change pass");
        let m_reactivity = ctx.compile_shader(&modules[4].1, "fsr3 prepare reactivity");
        let m_instability = ctx.compile_shader(&modules[5].1, "fsr3 luma instability");
        let m_accumulate = ctx.compile_shader(&modules[6].1, "fsr3 accumulate");
        let m_rcas = ctx.compile_shader(&modules[7].1, "fsr3 rcas");

        let [
            prepare_inputs,
            shading_change,
            downsample_mip1,
            downsample_mip2,
            downsample_farthest,
            shading_change_pass,
            prepare_reactivity,
            luma_instability,
            accumulate,
            p_rcas,
        ] = create_compute_pipelines(
            ctx,
            [
                (m_prepare, c"main"),
                (m_shading, c"main"),
                (m_downsample, c"downsample_mip1"),
                (m_downsample, c"downsample_mip2"),
                (m_downsample, c"downsample_farthest"),
                (m_shading_pass, c"main"),
                (m_reactivity, c"main"),
                (m_instability, c"main"),
                (m_accumulate, c"main"),
                (m_rcas, c"main"),
            ],
            layout,
        );
        let pipelines = Pipelines {
            prepare_inputs,
            shading_change,
            downsample_mip1,
            downsample_mip2,
            downsample_farthest,
            shading_change_pass,
            prepare_reactivity,
            luma_instability,
            accumulate,
            rcas: p_rcas,
        };
        for module in [
            m_prepare,
            m_shading,
            m_shading_pass,
            m_reactivity,
            m_instability,
            m_accumulate,
            m_rcas,
            m_downsample,
        ] {
            unsafe { ctx.device().destroy_shader_module(module, None) };
        }

        let mut fsr = Fsr3 {
            ctx: Arc::clone(ctx),
            layout,
            dsl,
            pool,
            sets,
            pipelines,
            targets,
            reconstructed,
            constants,
            rcas,
            linear_sampler,
            point_sampler,
            render_size,
            display_size,
            prev_render_size: render_size,
            prev_display_size: display_size,
            prev_jitter: [0.0, 0.0],
            frame_index: 0.0,
            ping: 0,
            inputs,
        };
        fsr.write_rcas_config(0.6);
        for slot in 0..frame_slots {
            fsr.write_descriptors(slot);
        }
        fsr
    }

    pub fn output_view(&self) -> vk::ImageView {
        self.targets.upscaled_output.view
    }

    pub fn output_image(&self) -> vk::Image {
        self.targets.upscaled_output.image
    }

    /// Render-resolution dilated reactive masks (R8G8B8A8_UNORM, layout
    /// GENERAL) and their size, for diagnostics. The channels are REACTIVE,
    /// DISOCCLUSION, SHADING_CHANGE and ACCUMULATION; a non-zero channel that
    /// survives a stationary camera is a permanent bias on the upscaler's
    /// output, because it feeds the Lanczos kernel bias.
    pub fn reactive_masks(&self) -> (vk::Image, (u32, u32)) {
        (self.targets.dilated_reactive_masks.image, self.render_size)
    }

    pub fn display_size(&self) -> (u32, u32) {
        self.display_size
    }

    /// Reset temporal history (after a resize, or when re-enabling).
    pub fn reset(&mut self) {
        self.frame_index = 0.0;
        self.prev_jitter = [0.0, 0.0];
        self.ping = 0;
    }

    pub fn set_inputs(&mut self, inputs: [vk::ImageView; 3]) {
        self.inputs = inputs;
    }

    pub fn resize(&mut self, render_size: (u32, u32), display_size: (u32, u32)) {
        self.render_size = render_size;
        self.display_size = display_size;
        self.targets = FsrTargets::build(&self.ctx, render_size, display_size);
        if (render_size.0 as u64) * (render_size.1 as u64) * 4 != self.reconstructed.size {
            self.reconstructed = self.ctx.create_storage_buffer(
                "fsr reconstructed depth",
                (render_size.0 as u64) * (render_size.1 as u64) * 4,
            );
        }
        self.reset();
    }

    pub fn write_rcas_config(&mut self, sharpness: f32) {
        // FsrRcasCon: sharpness is in stops; remap matches the FSDK dispatch.
        let stops = (-2.0 * sharpness) + 2.0;
        let linear = (-stops).exp2();
        let config = [linear.to_bits(), 0, 0, 0];
        let bytes = bytemuck::cast_slice(&config);
        self.rcas.write(0, bytes);
    }

    /// Fill the uniform block for this frame slot. `frame` must be the
    /// slot whose fence just recycled — the other slot may still be in
    /// flight and is left untouched.
    pub fn update(
        &mut self,
        camera: &Camera,
        jitter: [f32; 2],
        dt: f32,
        reset: bool,
        frame: usize,
    ) {
        if reset {
            self.reset();
            self.prev_render_size = self.render_size;
            self.prev_display_size = self.display_size;
        }
        let (near, far) = camera.clip_planes();
        let aspect = camera.aspect();
        let tan_half_fov_y = (camera.fov_y() * 0.5).tan();
        let fq = far / (near - far);
        let device_to_view_depth = [-fq, fq * near, tan_half_fov_y * aspect, tan_half_fov_y];
        let jitter_phase_count = {
            let ratio = self.display_size.0 as f32 / self.render_size.0.max(1) as f32;
            (8.0 * ratio * ratio).max(1.0)
        };

        let constants = Fsr3Constants {
            render_size: [self.render_size.0 as i32, self.render_size.1 as i32],
            previous_frame_render_size: [
                self.prev_render_size.0 as i32,
                self.prev_render_size.1 as i32,
            ],
            upscale_size: [self.display_size.0 as i32, self.display_size.1 as i32],
            previous_frame_upscale_size: [
                self.prev_display_size.0 as i32,
                self.prev_display_size.1 as i32,
            ],
            max_render_size: [self.render_size.0 as i32, self.render_size.1 as i32],
            max_upscale_size: [self.display_size.0 as i32, self.display_size.1 as i32],
            device_to_view_depth,
            jitter_offset: jitter,
            previous_frame_jitter_offset: self.prev_jitter,
            motion_vector_scale: [1.0, 1.0],
            downscale_factor: [
                self.render_size.0 as f32 / self.display_size.0.max(1) as f32,
                self.render_size.1 as f32 / self.display_size.1.max(1) as f32,
            ],
            motion_vector_jitter_cancellation: [0.0, 0.0],
            tan_half_fov: tan_half_fov_y * aspect,
            jitter_phase_count,
            delta_time: dt.clamp(0.0, 1.0),
            delta_pre_exposure: 1.0,
            view_space_to_meters_factor: 1.0,
            frame_index: self.frame_index,
            velocity_factor: 1.0,
            reactiveness_scale: 1.0,
            shading_change_scale: 1.0,
            accumulation_added_per_frame: 1.0 / 3.0,
            min_disocclusion_accumulation: -1.0 / 3.0,
            _padding: [0.0; 3],
        };
        self.constants[frame].write(0, bytemuck::bytes_of(&constants));

        self.prev_jitter = jitter;
        self.prev_render_size = self.render_size;
        self.prev_display_size = self.display_size;
        self.frame_index += 1.0;
    }

    fn write_descriptors(&self, frame: usize) {
        let read = self.ping;
        let write = 1 - self.ping;

        let sampled = |view: vk::ImageView| {
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
        };
        let sampler = |s: vk::Sampler| vk::DescriptorImageInfo::default().sampler(s);
        let buffer = |b: &GpuBuffer, size: u64| {
            vk::DescriptorBufferInfo::default()
                .buffer(b.buffer)
                .range(size)
        };

        let set = self.sets[frame];
        let constants_info = buffer(
            &self.constants[frame],
            std::mem::size_of::<Fsr3Constants>() as u64,
        );
        let rcas_info = buffer(&self.rcas, 16);
        let recon_size = self.reconstructed.size;

        let image_info = [
            sampled(self.inputs[0]), // 10 input color
            vk::DescriptorImageInfo::default()
                .image_view(self.inputs[2])
                .image_layout(vk::ImageLayout::GENERAL), // 11 input depth
            sampled(self.inputs[1]), // 12 input motion vectors
            sampled(self.targets.dilated_motion_vectors.view),
            sampled(self.targets.dilated_depth.view),
            sampled(self.targets.farthest_depth_mip1.view),
            // Current luma is the image prepare_inputs writes *this* frame, so
            // its read side is the same resource as binding 44 (`w_current_luma`);
            // the previous frame's luma is the other image of the pair. AMD's
            // host sets `currentLumaUavResourceIndex = currentLumaSrvResourceIndex`
            // and points `PREVIOUS_LUMA` at the other side
            // (host/ffx_fsr3upscaler.cpp, the per-frame descriptor table setup),
            // and `luma_1`/`luma_2` are only cleared on reset. Binding these the
            // other way round makes `shading_change` compare frame N-1 against
            // frame N reprojected, i.e. a two-motion-vector separation instead of
            // zero, which inflates shading change for the whole time the camera
            // moves and collapses accumulation with it.
            sampled(self.targets.current_luma[write].view),
            sampled(self.targets.current_luma[read].view),
            sampled(self.targets.luma_history[read].view),
            sampled(self.targets.luma_instability.view),
            sampled(self.targets.shading_change.view),
            sampled(self.targets.spd_mips[0].view),
            sampled(self.targets.spd_mips[1].view),
            sampled(self.targets.spd_mips[2].view),
            sampled(self.targets.dilated_reactive_masks.view),
            sampled(self.targets.accumulation[read].view),
            sampled(self.targets.internal_upscaled_color[read].view),
            sampled(self.targets.new_locks.view),
            sampled(self.targets.internal_upscaled_color[write].view), // 28 rcas input
            sampled(self.targets.farthest_depth.view),                 // 60 downsample src depth
        ];
        let write_info = [
            sampled(self.targets.dilated_motion_vectors.view),
            sampled(self.targets.dilated_depth.view),
            sampled(self.targets.farthest_depth.view),
            sampled(self.targets.farthest_depth_mip1.view),
            sampled(self.targets.current_luma[write].view),
            sampled(self.targets.luma_history[write].view),
            sampled(self.targets.luma_instability.view),
            sampled(self.targets.shading_change.view),
            sampled(self.targets.spd_mips[0].view),
            sampled(self.targets.spd_mips[1].view),
            sampled(self.targets.spd_mips[2].view),
            sampled(self.targets.dilated_reactive_masks.view),
            sampled(self.targets.accumulation[write].view),
            sampled(self.targets.internal_upscaled_color[write].view),
            sampled(self.targets.new_locks.view),
            sampled(self.targets.upscaled_output.view),
        ];

        let point_info = sampler(self.point_sampler);
        let linear_info = sampler(self.linear_sampler);
        let recon_info = buffer(&self.reconstructed, recon_size);
        let mut writes: Vec<vk::WriteDescriptorSet<'_>> = vec![
            uniform(set, 0, &constants_info),
            uniform(set, 1, &rcas_info),
            sampler_write(set, 2, &point_info),
            sampler_write(set, 3, &linear_info),
        ];
        for (i, info) in image_info.iter().enumerate() {
            writes.push(image(set, image_binding(i as u32), info));
        }
        for (i, info) in write_info.iter().enumerate() {
            writes.push(storage_image(set, 40 + i as u32, info));
        }
        writes.push(storage_buffer(set, 30, &recon_info));
        writes.push(storage_buffer(set, 31, &recon_info));
        unsafe { self.ctx.device().update_descriptor_sets(&writes, &[]) };
    }

    fn dispatch(
        cmd: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
        x: u32,
        y: u32,
        device: &ash::Device,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(cmd, x, y, 1);
        }
    }

    /// Record the whole FSR chain for frame slot `frame`. `history_valid`
    /// false forces a restart frame.
    pub fn record(&mut self, cmd: vk::CommandBuffer, history_valid: bool, frame: usize) {
        let device = self.ctx.device().clone();
        self.write_descriptors(frame);

        let (rw, rh) = self.render_size;
        let (dw, dh) = self.display_size;
        let groups = |w: u32, h: u32| (w.div_ceil(8), h.div_ceil(8));

        // Reconstructed depth is a min-accumulator: start it at +infinity.
        unsafe {
            device.cmd_fill_buffer(
                cmd,
                self.reconstructed.buffer,
                0,
                vk::WHOLE_SIZE,
                0xFFFF_FFFF,
            );
        }
        barrier(&device, cmd);

        let set = self.sets[frame];
        let layout = self.layout;
        let pass = |pipeline: vk::Pipeline, w: u32, h: u32| {
            let (x, y) = groups(w, h);
            Self::dispatch(cmd, pipeline, layout, set, x, y, &device);
            barrier(&device, cmd);
        };

        // Restart frame: clear the persistent state exactly like AMD's host
        // does on firstExecution (ACCUMULATION_1/2 and LUMA_1/2). Without
        // this the initial history garbage survives into the first frames and
        // smears for seconds after start-up, a resize, or re-enabling the
        // upscaler.
        if !history_valid {
            let clears = [
                self.targets.accumulation[0].image,
                self.targets.accumulation[1].image,
                self.targets.luma_history[0].image,
                self.targets.luma_history[1].image,
            ];
            unsafe {
                for &image in &clears {
                    crate::gfx::vulkan::image_barrier(
                        &device,
                        cmd,
                        image,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageAspectFlags::COLOR,
                    );
                }
                for &image in &clears {
                    device.cmd_clear_color_image(
                        cmd,
                        image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                        &[vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }],
                    );
                }
                for &image in &clears {
                    crate::gfx::vulkan::image_barrier(
                        &device,
                        cmd,
                        image,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::MEMORY_READ,
                        vk::ImageAspectFlags::COLOR,
                    );
                }
            }
        }
        pass(self.pipelines.prepare_inputs, rw, rh);
        pass(
            self.pipelines.downsample_farthest,
            rw.div_ceil(2),
            rh.div_ceil(2),
        );
        pass(
            self.pipelines.shading_change,
            rw.div_ceil(2),
            rh.div_ceil(2),
        );
        pass(
            self.pipelines.downsample_mip1,
            rw.div_ceil(4),
            rh.div_ceil(4),
        );
        pass(
            self.pipelines.downsample_mip2,
            rw.div_ceil(8),
            rh.div_ceil(8),
        );
        pass(
            self.pipelines.shading_change_pass,
            rw.div_ceil(2),
            rh.div_ceil(2),
        );
        pass(self.pipelines.prepare_reactivity, rw, rh);
        pass(self.pipelines.luma_instability, rw, rh);
        pass(self.pipelines.accumulate, dw, dh);
        pass(self.pipelines.rcas, dw, dh);

        self.ping = 1 - self.ping;
    }
}

impl Drop for Fsr3 {
    fn drop(&mut self) {
        unsafe {
            let device = self.ctx.device();
            let _ = device.device_wait_idle();
            for pipeline in [
                self.pipelines.prepare_inputs,
                self.pipelines.shading_change,
                self.pipelines.downsample_mip1,
                self.pipelines.downsample_mip2,
                self.pipelines.downsample_farthest,
                self.pipelines.shading_change_pass,
                self.pipelines.prepare_reactivity,
                self.pipelines.luma_instability,
                self.pipelines.accumulate,
                self.pipelines.rcas,
            ] {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.dsl, None);
            device.destroy_sampler(self.linear_sampler, None);
            device.destroy_sampler(self.point_sampler, None);
        }
    }
}
