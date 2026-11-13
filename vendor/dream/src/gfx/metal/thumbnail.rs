use super::metal::shader::{MslBindings, PUSH_CONSTANT_BUFFER_SLOT};
use super::pipeline::create_compute_pipeline;
use super::{GpuBuffer, GpuContext, Renderer};
use crate::ui::Rect;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLSize, MTLTexture,
};
use std::ptr::NonNull;

const WORKGROUP: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    crop: [i32; 2],
    extent: [i32; 2],
    side: i32,
    size: i32,
}

struct Job {
    params: Params,
    readback: GpuBuffer,
    bytes: usize,
}

pub(crate) struct ThumbnailPass {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    color: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    job: Option<Job>,
    slot: Option<usize>,
    ready: Option<Vec<u8>>,
}

impl ThumbnailPass {
    pub(crate) fn new(ctx: &GpuContext) -> Self {
        let bindings = MslBindings::new()
            .texture(0, 0, 0)
            .buffer(0, 1, 0, true)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);
        let pipeline = create_compute_pipeline(
            ctx,
            "scene thumbnail",
            include_str!("../thumbnail.wgsl"),
            &bindings,
            "main",
        );
        Self {
            pipeline,
            color: None,
            job: None,
            slot: None,
            ready: None,
        }
    }

    pub(crate) fn bind_color(&mut self, tex: Retained<ProtocolObject<dyn MTLTexture>>) {
        self.color = Some(tex);
    }

    pub(crate) fn busy(&self) -> bool {
        self.job.is_some() || self.ready.is_some() || self.slot.is_some()
    }

    fn record(&self, cmd: &ProtocolObject<dyn MTLCommandBuffer>, job: &Job) {
        let Some(color) = self.color.as_ref() else {
            return;
        };
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.pipeline);
        unsafe {
            enc.setTexture_atIndex(Some(color), 0);
            enc.setBuffer_offset_atIndex(Some(&job.readback.buffer), 0, 0);
            let bytes = bytemuck::bytes_of(&job.params);
            enc.setBytes_length_atIndex(
                NonNull::new_unchecked(bytes.as_ptr() as *mut _),
                bytes.len(),
                PUSH_CONSTANT_BUFFER_SLOT as usize,
            );
        }
        let groups = (job.params.size as usize).div_ceil(WORKGROUP);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: groups,
                depth: 1,
            },
            MTLSize {
                width: WORKGROUP,
                height: WORKGROUP,
                depth: 1,
            },
        );
        enc.endEncoding();
    }
}

impl Renderer {
    fn ensure_thumb(&mut self) {
        self.ensure_scene_targets();
        if self.thumb.is_none() {
            let mut pass = ThumbnailPass::new(&self.ctx);
            pass.bind_color(self.targets.color.texture.clone());
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

    pub fn scene_thumbnail_busy(&self) -> bool {
        self.thumb.as_ref().is_some_and(|t| t.job.is_some())
    }

    pub fn take_scene_thumbnail(&mut self) -> Option<Vec<u8>> {
        self.thumb.as_mut()?.ready.take()
    }

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
        self.ctx.one_shot(|cmd| thumb.record(cmd, &job));
        let pixels = job.readback.mapped()?[..job.bytes].to_vec();
        let edge = params.size as u32;
        Some((pixels, edge, edge))
    }

    fn new_thumbnail_job(&self, params: Params) -> Job {
        let bytes = params.size as usize * params.size as usize * 4;
        // On Apple Silicon UMA, compute writes directly to a shared readback buffer!
        let readback = self
            .ctx
            .create_readback_buffer("scene thumbnail readback", bytes as u64);
        Job {
            params,
            readback,
            bytes,
        }
    }

    pub(crate) fn record_scene_thumbnail(
        &mut self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        slot: usize,
    ) {
        let Some(thumb) = self.thumb.as_mut() else {
            return;
        };
        if thumb.slot.is_some() {
            return;
        }
        let Some(job) = thumb.job.as_ref() else {
            return;
        };
        thumb.record(cmd, job);
        thumb.slot = Some(slot);
    }

    pub(crate) fn finish_scene_thumbnail(&mut self, slot: usize) {
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

fn crop_params(render: (u32, u32), display: (u32, u32), rect: Rect, size: u32) -> Option<Params> {
    if render.0 == 0 || render.1 == 0 || display.0 == 0 || display.1 == 0 {
        return None;
    }
    let (rw, rh) = (render.0 as i32, render.1 as i32);
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
