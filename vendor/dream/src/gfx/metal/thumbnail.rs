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
        // `main_topdown`: Metal's scene target holds its rows top-first (the
        // world-space shaders are not y-flipped), so the crop reads rows
        // straight; Vulkan's bottom-up target is why the shared `main` mirrors.
        let pipeline = create_compute_pipeline(
            ctx,
            "scene thumbnail",
            include_str!("../thumbnail.wgsl"),
            &bindings,
            "main_topdown",
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
    // The scene target's row 0 is the top row on Metal, so the display rect's
    // top-down y is the crop's y: no flip.
    let y0 = (rect.y * sy).floor() as i32;
    let x1 = ((rect.x + rect.w) * sx).ceil() as i32;
    let y1 = ((rect.y + rect.h) * sy).ceil() as i32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::GpuContext;
    use crate::ui::Rect;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLCommandEncoder, MTLOrigin, MTLPixelFormat, MTLSize,
        MTLTextureUsage,
    };

    /// The crop origin is in the scene target's own row order: Metal's scene
    /// pass writes top-first rows, so a display rect keeps its y.
    #[test]
    fn metal_thumbnail_crop_is_top_down() {
        let params = crop_params((64, 64), (64, 64), Rect::new(0.0, 10.0, 40.0, 20.0), 8)
            .expect("a non-empty rect yields a crop");
        // Centre crop inside the 40x20 rect: 10 texels off its left edge.
        assert_eq!(params.crop, [10, 10], "the crop starts at the rect's top");
        assert_eq!(params.side, 20, "centre-cropped to the shorter edge");
        assert_eq!(params.extent, [64, 64]);

        // A square at the bottom of a tall target: the crop follows the rect.
        let params = crop_params((40, 100), (40, 100), Rect::new(0.0, 70.0, 40.0, 20.0), 8)
            .expect("a non-empty rect yields a crop");
        assert_eq!(params.crop, [10, 70]);
    }

    /// A capture reads the scene target's rows straight: the bright top half
    /// must land in the thumbnail's top rows.
    #[test]
    fn metal_thumbnail_keeps_the_scene_upright() {
        let Some(ctx) = GpuContext::new_headless() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        const S: u32 = 64;
        let scene = ctx.create_image(
            "thumbnail test scene",
            S,
            S,
            MTLPixelFormat::RGBA8Unorm,
            MTLTextureUsage::ShaderRead,
        );
        // Top half white, bottom half black, uploaded row for row.
        let mut pixels = vec![0u8; (S * S * 4) as usize];
        for y in 0..S as usize {
            let v = if y < S as usize / 2 { 255 } else { 0 };
            for x in 0..S as usize {
                let at = (y * S as usize + x) * 4;
                pixels[at..at + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let staging = ctx.upload_buffer("thumbnail test staging", &pixels);
        ctx.one_shot(|cmd| {
            let blit = cmd.blitCommandEncoder().expect("blit encoder");
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &staging.buffer,
                    0,
                    (S * 4) as usize,
                    (S * S * 4) as usize,
                    MTLSize {
                        width: S as usize,
                        height: S as usize,
                        depth: 1,
                    },
                    &scene.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            blit.endEncoding();
        });

        let mut pass = ThumbnailPass::new(&ctx);
        pass.bind_color(scene.texture.clone());
        let params = crop_params((S, S), (S, S), Rect::new(0.0, 0.0, S as f32, S as f32), 8)
            .expect("full-rect crop");
        let job = Job {
            params,
            readback: ctx.create_readback_buffer(
                "thumbnail test readback",
                (params.size as usize * params.size as usize * 4) as u64,
            ),
            bytes: params.size as usize * params.size as usize * 4,
        };
        ctx.one_shot(|cmd| pass.record(cmd, &job));
        let px = job.readback.mapped().expect("host-visible readback");
        let at = |x: usize, y: usize| px[(y * params.size as usize + x) * 4] as i32;
        let top = at(4, 1);
        let bottom = at(4, 6);
        assert!(
            top > 200,
            "the scene's top half must be in the thumbnail's top rows: top={top} bottom={bottom}"
        );
        assert!(
            bottom < 60,
            "the scene's bottom half must be in the thumbnail's bottom rows: top={top} bottom={bottom}"
        );
    }
}
