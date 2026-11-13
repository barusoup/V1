//! Native Metal implementation of Dream's FSR 3.1 temporal upscaler.

use crate::camera::Camera;
use crate::gfx::metal::pipeline::create_compute_pipeline;
use crate::gfx::metal::shader::MslBindings;
use crate::gfx::metal::{GpuBuffer, GpuContext, GpuImage};
use crate::texture::{Filter, MipFilter, SamplerDesc};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLLoadAction, MTLPixelFormat,
    MTLRenderPassDescriptor, MTLSamplerState, MTLSize, MTLStoreAction, MTLTexture, MTLTextureUsage,
};
use std::sync::Arc;

pub use crate::temporal::JitterSequence;

const PRELUDE: &str = include_str!("fsr3_common.wgsl");

fn pass_source(pass: &str) -> String {
    format!("{PRELUDE}\n{pass}")
}

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

struct FsrTargets {
    dilated_motion_vectors: GpuImage,
    dilated_depth: GpuImage,
    farthest_depth: GpuImage,
    current_luma: [GpuImage; 2],
    luma_history: [GpuImage; 2],
    luma_instability: GpuImage,
    dilated_reactive_masks: GpuImage,
    accumulation: [GpuImage; 2],
    farthest_depth_mip1: GpuImage,
    shading_change: GpuImage,
    spd_mips: [GpuImage; 3],
    internal_upscaled_color: [GpuImage; 2],
    new_locks: GpuImage,
    upscaled_output: GpuImage,
}

impl FsrTargets {
    fn build(ctx: &Arc<GpuContext>, render: (u32, u32), display: (u32, u32)) -> Self {
        let (rw, rh) = (render.0.max(1), render.1.max(1));
        let (dw, dh) = (display.0.max(1), display.1.max(1));
        let half = (rw.div_ceil(2), rh.div_ceil(2));
        let quarter = (rw.div_ceil(4), rh.div_ceil(4));
        let eighth = (rw.div_ceil(8), rh.div_ceil(8));

        let usage = MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::RenderTarget;
        let make = |label: &str, (w, h): (u32, u32), fmt: MTLPixelFormat| {
            ctx.create_image(label, w, h, fmt, usage)
        };

        Self {
            dilated_motion_vectors: make("fsr dl mv", (rw, rh), MTLPixelFormat::RG16Float),
            dilated_depth: make("fsr dl depth", (rw, rh), MTLPixelFormat::R32Float),
            farthest_depth: make("fsr farthest depth", (rw, rh), MTLPixelFormat::R16Float),
            current_luma: [
                make("fsr luma 0", (rw, rh), MTLPixelFormat::R16Float),
                make("fsr luma 1", (rw, rh), MTLPixelFormat::R16Float),
            ],
            luma_history: [
                make("fsr luma history 0", (rw, rh), MTLPixelFormat::RGBA16Float),
                make("fsr luma history 1", (rw, rh), MTLPixelFormat::RGBA16Float),
            ],
            luma_instability: make("fsr luma instability", (rw, rh), MTLPixelFormat::R16Float),
            dilated_reactive_masks: make("fsr reactive", (rw, rh), MTLPixelFormat::RGBA8Unorm),
            accumulation: [
                make("fsr accumulation 0", (rw, rh), MTLPixelFormat::R8Unorm),
                make("fsr accumulation 1", (rw, rh), MTLPixelFormat::R8Unorm),
            ],
            farthest_depth_mip1: make("fsr farthest depth mip1", half, MTLPixelFormat::R16Float),
            shading_change: make("fsr shading change", half, MTLPixelFormat::R8Unorm),
            spd_mips: [
                make("fsr spd mip0", half, MTLPixelFormat::RG16Float),
                make("fsr spd mip1", quarter, MTLPixelFormat::RG16Float),
                make("fsr spd mip2", eighth, MTLPixelFormat::RG16Float),
            ],
            internal_upscaled_color: [
                make("fsr upscaled 0", (dw, dh), MTLPixelFormat::RGBA16Float),
                make("fsr upscaled 1", (dw, dh), MTLPixelFormat::RGBA16Float),
            ],
            new_locks: make("fsr new locks", (dw, dh), MTLPixelFormat::R8Unorm),
            upscaled_output: make("fsr output", (dw, dh), MTLPixelFormat::RGBA16Float),
        }
    }
}

struct FsrPipelines {
    prepare_inputs: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    shading_change: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    downsample_mip1: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    downsample_mip2: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    shading_change_pass: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    prepare_reactivity: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    luma_instability: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    accumulate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rcas: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct Fsr3 {
    ctx: Arc<GpuContext>,
    pipelines: FsrPipelines,
    point_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    linear_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    targets: FsrTargets,
    reconstructed: GpuBuffer,
    constants: Vec<GpuBuffer>,
    rcas: GpuBuffer,
    inputs: [Option<Retained<ProtocolObject<dyn MTLTexture>>>; 3],
    render_size: (u32, u32),
    display_size: (u32, u32),
    prev_render_size: (u32, u32),
    prev_display_size: (u32, u32),
    prev_jitter: [f32; 2],
    frame_index: f32,
    ping: usize,
}

fn fsr_bindings() -> MslBindings {
    let mut b = MslBindings::new()
        .buffer(0, 0, 0, false)
        .buffer(0, 1, 1, false)
        .buffer(0, 30, 2, false)
        .buffer(0, 31, 3, true)
        .sampler(0, 2, 0)
        .sampler(0, 3, 1);
    for i in 0..19u32 {
        b = b.texture(0, 10 + i, i as u8);
    }
    for i in 0..16u32 {
        b = b.texture(0, 40 + i, (19 + i) as u8);
    }
    b
}

impl Fsr3 {
    pub fn new(ctx: &Arc<GpuContext>, render_size: (u32, u32), display_size: (u32, u32)) -> Self {
        let bindings = fsr_bindings();
        let make_pipe = |label: &str, src: &str, entry: &str| {
            create_compute_pipeline(ctx, label, &pass_source(src), &bindings, entry)
        };
        let pipelines = FsrPipelines {
            prepare_inputs: make_pipe(
                "fsr prepare_inputs",
                include_str!("fsr3_prepare_inputs.wgsl"),
                "main",
            ),
            shading_change: make_pipe(
                "fsr shading_change",
                include_str!("fsr3_shading_change.wgsl"),
                "main",
            ),
            downsample_mip1: make_pipe(
                "fsr downsample mip1",
                include_str!("fsr3_downsample.wgsl"),
                "downsample_mip1",
            ),
            downsample_mip2: make_pipe(
                "fsr downsample mip2",
                include_str!("fsr3_downsample.wgsl"),
                "downsample_mip2",
            ),
            shading_change_pass: make_pipe(
                "fsr shading_change_pass",
                include_str!("fsr3_shading_change_pass.wgsl"),
                "main",
            ),
            prepare_reactivity: make_pipe(
                "fsr prepare_reactivity",
                include_str!("fsr3_prepare_reactivity.wgsl"),
                "main",
            ),
            luma_instability: make_pipe(
                "fsr luma_instability",
                include_str!("fsr3_luma_instability.wgsl"),
                "main",
            ),
            accumulate: make_pipe(
                "fsr accumulate",
                include_str!("fsr3_accumulate.wgsl"),
                "main",
            ),
            rcas: make_pipe("fsr rcas", include_str!("fsr3_rcas.wgsl"), "main"),
        };

        let point_sampler = ctx.get_or_create_sampler(&SamplerDesc::PIXEL);
        let linear_sampler = ctx.get_or_create_sampler(&SamplerDesc {
            filter: Filter::Linear,
            mip: MipFilter::Nearest,
            ..Default::default()
        });

        let targets = FsrTargets::build(ctx, render_size, display_size);
        let recon_bytes = (render_size.0.max(1) as u64) * (render_size.1.max(1) as u64) * 4;
        let reconstructed = ctx.create_device_buffer("fsr reconstructed depth", recon_bytes);
        let constants = (0..ctx.max_frames_in_flight())
            .map(|_| {
                ctx.create_host_buffer("fsr constants", std::mem::size_of::<Fsr3Constants>() as u64)
            })
            .collect();
        let rcas = ctx.create_host_buffer("fsr rcas", 16);

        let mut fsr = Self {
            ctx: Arc::clone(ctx),
            pipelines,
            point_sampler,
            linear_sampler,
            targets,
            reconstructed,
            constants,
            rcas,
            inputs: [None, None, None],
            render_size,
            display_size,
            prev_render_size: render_size,
            prev_display_size: display_size,
            prev_jitter: [0.0, 0.0],
            frame_index: 0.0,
            ping: 0,
        };
        fsr.write_rcas_config(0.25);
        fsr
    }

    pub fn output_texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.targets.upscaled_output.texture
    }

    pub fn reactive_masks(&self) -> (&ProtocolObject<dyn MTLTexture>, (u32, u32)) {
        (
            &self.targets.dilated_reactive_masks.texture,
            self.render_size,
        )
    }

    pub fn set_inputs(&mut self, inputs: [&ProtocolObject<dyn MTLTexture>; 3]) {
        self.inputs = [
            Some(Retained::from(inputs[0])),
            Some(Retained::from(inputs[1])),
            Some(Retained::from(inputs[2])),
        ];
    }

    pub fn resize(&mut self, render_size: (u32, u32), display_size: (u32, u32)) {
        self.render_size = render_size;
        self.display_size = display_size;
        self.targets = FsrTargets::build(&self.ctx, render_size, display_size);
        let recon_bytes = (render_size.0.max(1) as u64) * (render_size.1.max(1) as u64) * 4;
        self.reconstructed = self
            .ctx
            .create_device_buffer("fsr reconstructed depth", recon_bytes);
        self.ping = 0;
        self.frame_index = 0.0;
    }

    pub fn write_rcas_config(&mut self, sharpness: f32) {
        let stops = (-2.0 * sharpness) + 2.0;
        let linear = (-stops).exp2();
        let config = [linear.to_bits(), 0, 0, 0];
        self.rcas.write_bytes(0, bytemuck::cast_slice(&config));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_constants(
        &mut self,
        camera: &Camera,
        render_size: (u32, u32),
        display_size: (u32, u32),
        jitter: [f32; 2],
        dt: f32,
        sharpness: f32,
        reset: bool,
        frame: usize,
    ) {
        self.render_size = render_size;
        self.display_size = display_size;
        self.write_rcas_config(sharpness);
        if reset {
            self.prev_render_size = render_size;
            self.prev_display_size = display_size;
            self.frame_index = 0.0;
            self.ping = 0;
        }
        let (near, far) = camera.clip_planes();
        let aspect = camera.aspect();
        let tan_half_fov_y = (camera.fov_y() * 0.5).tan();
        let fq = far / (near - far);
        let device_to_view_depth = [-fq, fq * near, tan_half_fov_y * aspect, tan_half_fov_y];
        let jitter_phase_count = {
            let ratio = display_size.0 as f32 / render_size.0.max(1) as f32;
            (8.0 * ratio * ratio).max(1.0)
        };

        let constants = Fsr3Constants {
            render_size: [render_size.0 as i32, render_size.1 as i32],
            previous_frame_render_size: [
                self.prev_render_size.0 as i32,
                self.prev_render_size.1 as i32,
            ],
            upscale_size: [display_size.0 as i32, display_size.1 as i32],
            previous_frame_upscale_size: [
                self.prev_display_size.0 as i32,
                self.prev_display_size.1 as i32,
            ],
            max_render_size: [render_size.0 as i32, render_size.1 as i32],
            max_upscale_size: [display_size.0 as i32, display_size.1 as i32],
            device_to_view_depth,
            jitter_offset: jitter,
            previous_frame_jitter_offset: self.prev_jitter,
            motion_vector_scale: [1.0, 1.0],
            downscale_factor: [
                render_size.0 as f32 / display_size.0.max(1) as f32,
                render_size.1 as f32 / display_size.1.max(1) as f32,
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
        let slot = frame % self.constants.len();
        self.constants[slot].write_bytes(0, bytemuck::bytes_of(&constants));

        self.prev_jitter = jitter;
        self.prev_render_size = render_size;
        self.prev_display_size = display_size;
        self.frame_index += 1.0;
    }

    pub fn record(
        &mut self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        history_valid: bool,
        frame: usize,
    ) {
        let (Some(in_color), Some(in_vel), Some(in_depth)) = (
            self.inputs[0].as_ref(),
            self.inputs[1].as_ref(),
            self.inputs[2].as_ref(),
        ) else {
            return;
        };

        // Fill reconstructed depth buffer with 0xFF (+infinity).
        let blit = cmd.blitCommandEncoder().expect("fsr blit encoder");
        blit.fillBuffer_range_value(
            &self.reconstructed.buffer,
            NSRange::new(0, self.reconstructed.size as usize),
            0xFF,
        );
        blit.endEncoding();

        if !history_valid {
            for tex in [
                &self.targets.accumulation[0].texture,
                &self.targets.accumulation[1].texture,
                &self.targets.luma_history[0].texture,
                &self.targets.luma_history[1].texture,
            ] {
                let rpd = MTLRenderPassDescriptor::new();
                let att = unsafe { rpd.colorAttachments().objectAtIndexedSubscript(0) };
                att.setTexture(Some(tex));
                att.setLoadAction(MTLLoadAction::Clear);
                att.setStoreAction(MTLStoreAction::Store);
                att.setClearColor(MTLClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 0.0,
                });
                let enc = cmd
                    .renderCommandEncoderWithDescriptor(&rpd)
                    .expect("fsr clear");
                enc.endEncoding();
            }
        }

        let read = self.ping;
        let write = 1 - self.ping;
        let const_slot = frame % self.constants.len();

        let enc = cmd.computeCommandEncoder().expect("fsr compute encoder");
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.constants[const_slot].buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.rcas.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.reconstructed.buffer), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.reconstructed.buffer), 0, 3);
            enc.setSamplerState_atIndex(Some(&self.point_sampler), 0);
            enc.setSamplerState_atIndex(Some(&self.linear_sampler), 1);

            let sampled: [&ProtocolObject<dyn MTLTexture>; 19] = [
                in_color,
                in_depth,
                in_vel,
                &self.targets.dilated_motion_vectors.texture,
                &self.targets.dilated_depth.texture,
                &self.targets.farthest_depth_mip1.texture,
                &self.targets.current_luma[write].texture,
                &self.targets.current_luma[read].texture,
                &self.targets.luma_history[read].texture,
                &self.targets.luma_instability.texture,
                &self.targets.shading_change.texture,
                &self.targets.spd_mips[0].texture,
                &self.targets.spd_mips[1].texture,
                &self.targets.spd_mips[2].texture,
                &self.targets.dilated_reactive_masks.texture,
                &self.targets.accumulation[read].texture,
                &self.targets.internal_upscaled_color[read].texture,
                &self.targets.new_locks.texture,
                &self.targets.internal_upscaled_color[write].texture,
            ];
            for (i, tex) in sampled.iter().enumerate() {
                enc.setTexture_atIndex(Some(*tex), i);
            }

            let storage: [&ProtocolObject<dyn MTLTexture>; 16] = [
                &self.targets.dilated_motion_vectors.texture,
                &self.targets.dilated_depth.texture,
                &self.targets.farthest_depth.texture,
                &self.targets.farthest_depth_mip1.texture,
                &self.targets.current_luma[write].texture,
                &self.targets.luma_history[write].texture,
                &self.targets.luma_instability.texture,
                &self.targets.shading_change.texture,
                &self.targets.spd_mips[0].texture,
                &self.targets.spd_mips[1].texture,
                &self.targets.spd_mips[2].texture,
                &self.targets.dilated_reactive_masks.texture,
                &self.targets.accumulation[write].texture,
                &self.targets.internal_upscaled_color[write].texture,
                &self.targets.new_locks.texture,
                &self.targets.upscaled_output.texture,
            ];
            for (i, tex) in storage.iter().enumerate() {
                enc.setTexture_atIndex(Some(*tex), 19 + i);
            }
        }

        let dispatch = |pipe: &ProtocolObject<dyn MTLComputePipelineState>, w: u32, h: u32| {
            enc.setComputePipelineState(pipe);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: w.div_ceil(8).max(1) as usize,
                    height: h.div_ceil(8).max(1) as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 8,
                    height: 8,
                    depth: 1,
                },
            );
        };

        let (rw, rh) = self.render_size;
        let (dw, dh) = self.display_size;
        dispatch(&self.pipelines.prepare_inputs, rw, rh);
        dispatch(
            &self.pipelines.shading_change,
            rw.div_ceil(2),
            rh.div_ceil(2),
        );
        dispatch(
            &self.pipelines.downsample_mip1,
            rw.div_ceil(4),
            rh.div_ceil(4),
        );
        dispatch(
            &self.pipelines.downsample_mip2,
            rw.div_ceil(8),
            rh.div_ceil(8),
        );
        dispatch(
            &self.pipelines.shading_change_pass,
            rw.div_ceil(2),
            rh.div_ceil(2),
        );
        dispatch(&self.pipelines.prepare_reactivity, rw, rh);
        dispatch(&self.pipelines.luma_instability, rw, rh);
        dispatch(&self.pipelines.accumulate, dw, dh);
        dispatch(&self.pipelines.rcas, dw, dh);
        enc.endEncoding();

        self.ping = 1 - self.ping;
    }
}
