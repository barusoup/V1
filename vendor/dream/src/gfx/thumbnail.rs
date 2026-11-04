//! Scene card thumbnails.
//!
//! A compute pass box-filters a centre-cropped square of the scene colour
//! target into sRGB RGBA8 texels, recorded into the frame that rendered the
//! scene. The result is copied into a cached readback buffer, and the CPU
//! picks it up once that frame's fence signals, so no capture ever waits on
//! the GPU or touches the full-resolution target on the CPU.

use super::pipeline::{
    binding, create_dsl, create_pipeline_layout, image_write, sampled, write_sets,
};
use super::{GpuBuffer, GpuContext, Renderer};
use crate::ui::Rect;
use ash::vk;

const WORKGROUP: u32 = 8;

/// Crop and output geometry, passed to the shader as push constants.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    /// Crop origin in render texels, measured from the bottom row (Vulkan row order).
    crop: [i32; 2],
    /// Render target size in texels.
    extent: [i32; 2],
    /// Crop edge in texels.
    side: i32,
    /// Output edge in texels.
    size: i32,
}

/// One capture: its geometry and the buffers it writes.
struct Job {
    params: Params,
    /// Device-local output the compute pass writes.
    storage: GpuBuffer,
    /// Cached host-visible copy the CPU reads once the frame completes.
    readback: GpuBuffer,
    bytes: usize,
}

/// Compute pipeline plus the capture currently queued or in flight.
pub(super) struct ThumbnailPass {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Queued or recorded capture; `None` when idle.
    job: Option<Job>,
    /// Frame slot the job was recorded into, once it has been.
    slot: Option<usize>,
    /// Finished RGBA8 pixels, top row first, waiting to be taken.
    ready: Option<Vec<u8>>,
}

impl ThumbnailPass {
    pub(super) fn new(ctx: &GpuContext) -> Self {
        let dsl = create_dsl(
            ctx,
            "thumbnail layout",
            &[
                binding(
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    vk::ShaderStageFlags::COMPUTE,
                ),
                binding(
                    1,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::COMPUTE,
                ),
            ],
        );
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<Params>() as u32);
        let layout = create_pipeline_layout(ctx, "thumbnail layout", &[dsl], &[push]);
        let module = ctx.compile_shader(include_str!("thumbnail.wgsl"), "scene thumbnail");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipeline = unsafe {
            ctx.device()
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, err)| err)
                .expect("create thumbnail pipeline")[0]
        };
        unsafe { ctx.device().destroy_shader_module(module, None) };

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        ];
        let pool = unsafe {
            ctx.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .expect("create thumbnail pool");
        let set = unsafe {
            ctx.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&dsl)),
            )
        }
        .expect("allocate thumbnail set")[0];

        Self {
            pipeline,
            layout,
            dsl,
            pool,
            set,
            job: None,
            slot: None,
            ready: None,
        }
    }

    /// Point the scene-colour binding at the current target (rebuilt on resize).
    pub(super) fn bind_color(&self, ctx: &GpuContext, view: vk::ImageView) {
        let info = sampled(view);
        write_sets(ctx, &[image_write(self.set, 0, &info)]);
    }

    pub(super) fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.dsl, None);
            device.destroy_descriptor_pool(self.pool, None);
        }
    }

    fn bind_output(&self, ctx: &GpuContext, buffer: &GpuBuffer, bytes: u64) {
        let info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .range(bytes);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&info));
        write_sets(ctx, &[write]);
    }

    /// Record the dispatch and the copy of its output into the readback buffer.
    fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, job: &Job) {
        let groups = (job.params.size as u32).div_ceil(WORKGROUP);
        let copy = vk::BufferCopy::default().size(job.bytes as u64);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&job.params),
            );
            device.cmd_dispatch(cmd, groups, groups, 1);
            let written = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
            device.cmd_copy_buffer(cmd, job.storage.buffer, job.readback.buffer, &[copy]);
        }
    }
}

impl Renderer {
    /// Queue a thumbnail of the scene pass that the next [`Self::render_ui`]
    /// records: the centre-cropped square of `display_rect`, `size` texels
    /// square. Poll [`Self::take_scene_thumbnail`] for the pixels. Returns
    /// `false` while a previous capture is still pending, or when the rect
    /// misses the render target.
    fn ensure_thumb(&mut self) {
        self.ensure_scene_targets();
        if self.thumb.is_none() {
            let pass = ThumbnailPass::new(&self.ctx);
            pass.bind_color(&self.ctx, self.targets.color.view);
            self.thumb = Some(pass);
        }
    }

    pub fn request_scene_thumbnail(&mut self, display_rect: Rect, size: u32) -> bool {
        if self.thumb.as_ref().is_some_and(|t| t.job.is_some()) {
            return false;
        }
        let Some(params) = crop_params(self.render_size, self.display_size, display_rect, size)
        else {
            return false;
        };
        self.ensure_thumb();
        let job = self.new_thumbnail_job(params);
        self.thumb.as_mut().expect("thumb initialized").job = Some(job);
        true
    }

    /// Whether a capture is queued or in flight.
    pub fn scene_thumbnail_busy(&self) -> bool {
        self.thumb.as_ref().is_some_and(|t| t.job.is_some())
    }

    /// The finished thumbnail (RGBA8, top row first), once its frame has
    /// completed. `None` while pending or when nothing finished since the last take.
    pub fn take_scene_thumbnail(&mut self) -> Option<Vec<u8>> {
        self.thumb.as_mut()?.ready.take()
    }

    /// Synchronous variant of [`Self::request_scene_thumbnail`] for callers that
    /// need the pixels now: runs the same downsample in a one-shot submission
    /// and waits for it, reading the scene target as of the last submitted frame.
    pub fn capture_scene_thumbnail(
        &mut self,
        display_rect: Rect,
        size: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        if self.thumb.as_ref().is_some_and(|t| t.job.is_some()) {
            return None;
        }
        let params = crop_params(self.render_size, self.display_size, display_rect, size)?;
        self.ensure_thumb();
        let job = self.new_thumbnail_job(params);
        let thumb = self.thumb.as_ref().expect("thumb initialized");
        self.ctx
            .one_shot(|cmd| thumb.record(self.ctx.device(), cmd, &job));
        let pixels = job.readback.mapped()?[..job.bytes].to_vec();
        let edge = params.size as u32;
        Some((pixels, edge, edge))
    }

    fn new_thumbnail_job(&self, params: Params) -> Job {
        let bytes = params.size as usize * params.size as usize * 4;
        let storage = self.ctx.create_gpu_buffer(
            "scene thumbnail",
            bytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        self.thumb.as_ref().expect("thumb initialized").bind_output(
            &self.ctx,
            &storage,
            bytes as u64,
        );
        let readback = self
            .ctx
            .create_cached_readback_buffer("scene thumbnail readback", bytes as u64);
        Job {
            params,
            storage,
            readback,
            bytes,
        }
    }

    /// Frame recording hook: dispatch a queued capture into this frame.
    pub(super) fn record_scene_thumbnail(&mut self, cmd: vk::CommandBuffer, slot: usize) {
        let Some(thumb) = self.thumb.as_mut() else {
            return;
        };
        if thumb.slot.is_some() {
            return;
        }
        let Some(job) = thumb.job.as_ref() else {
            return;
        };
        thumb.record(self.ctx.device(), cmd, job);
        thumb.slot = Some(slot);
    }

    pub(super) fn finish_scene_thumbnail(&mut self, slot: usize) {
        let Some(thumb) = self.thumb.as_mut() else {
            return;
        };
        if thumb.slot != Some(slot) {
            return;
        }
        thumb.slot = None;
        if let Some(job) = thumb.job.take() {
            thumb.ready = job.readback.mapped().map(|px| px[..job.bytes].to_vec());
        }
    }
}

/// Map a display-space rect to the centre-cropped square of the render target
/// it covers, in the scene pass's Vulkan row order.
fn crop_params(render: (u32, u32), display: (u32, u32), rect: Rect, size: u32) -> Option<Params> {
    if render.0 == 0 || render.1 == 0 || display.0 == 0 || display.1 == 0 {
        return None;
    }
    let (rw, rh) = (render.0 as i32, render.1 as i32);
    // Mirror the scene pass scissor: Vulkan row zero is the display's bottom row.
    let sx = render.0 as f32 / display.0.max(1) as f32;
    let sy = render.1 as f32 / display.1.max(1) as f32;
    let x0 = (rect.x * sx).floor() as i32;
    let y0 = (render.1 as f32 - (rect.y + rect.h) * sy).floor() as i32;
    let x1 = ((rect.x + rect.w) * sx).ceil() as i32;
    let y1 = (render.1 as f32 - rect.y * sy).ceil() as i32;
    let (x0, y0, x1, y1) = (
        x0.clamp(0, rw),
        y0.clamp(0, rh),
        x1.clamp(0, rw),
        y1.clamp(0, rh),
    );
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let side = cw.min(ch);
    Some(Params {
        crop: [x0 + (cw - side) / 2, y0 + (ch - side) / 2],
        extent: [rw, rh],
        side,
        size: size.max(1) as i32,
    })
}
