//! Frosted-glass backdrop: a blurred copy of the UI frame drawn so far.
//!
//! When a UI pass reaches its first `CmdTexture::Backdrop` command, the pass is paused: the
//! target image is copied at reduced resolution, blurred with horizontal and vertical passes, and
//! the pass resumes with the blurred result available to glass draws. Everything drawn before the
//! glass is therefore what the glass blurs.
//!
//! The copy needs a target that can be blitted from. The renderer composes frames either into its
//! composed display image (kept in `GENERAL`) or directly into the swapchain image
//! (`COLOR_ATTACHMENT_OPTIMAL`); [`UiRenderer::capture_backdrop`] transitions whichever layout it
//! is given and restores it afterwards.
//!
//! Each frame slot owns its own backdrop textures, so a slot still in flight on the GPU is never
//! written by the next frame.

use super::{UiRenderer, create_ui_pipeline};
use crate::gfx::vulkan::{self, GpuContext};
use crate::texture::GpuTexture;
use ash::vk;
use std::sync::Arc;

const SHADER: &str = include_str!("../backdrop.wgsl");

/// Reduced-resolution textures for one frame slot: the copy of the frame, the blur scratch, and the
/// blurred result that glass samples.
pub(super) struct Backdrop {
    /// Full display size, so a resize is noticed.
    pub size: (u32, u32),
    /// Half-size intermediate for the two-step downsample (a single 4x linear blit would alias).
    pub mid: Arc<GpuTexture>,
    /// Quarter-size copy of the frame, the first blur input.
    pub source: Arc<GpuTexture>,
    pub scratch: Arc<GpuTexture>,
    pub out: Arc<GpuTexture>,
}

impl Backdrop {
    /// Allocate the four textures for a `size` display. Their clears are recorded into `cmd`.
    pub(super) fn new(
        ctx: &Arc<GpuContext>,
        cmd: vk::CommandBuffer,
        format: vk::Format,
        size: (u32, u32),
    ) -> Self {
        // Quarter resolution: the blur radius is then four times wider in display pixels, so a
        // bright feature spreads into a broad wash instead of a small spot.
        let quarter = |n: u32| n.div_ceil(4).max(1);
        let half = |n: u32| n.div_ceil(2).max(1);
        let make = |label: &str, w: u32, h: u32| {
            Arc::new(GpuTexture::render_target(ctx, cmd, label, w, h, format))
        };
        Self {
            size,
            mid: make("backdrop mid", half(size.0), half(size.1)),
            source: make("backdrop source", quarter(size.0), quarter(size.1)),
            scratch: make("backdrop scratch", quarter(size.0), quarter(size.1)),
            out: make("backdrop out", quarter(size.0), quarter(size.1)),
        }
    }
}

/// The two blur pipelines, shared by every slot. Both write `format`, the display's format.
pub(super) struct BlurPipelines {
    layout: vk::PipelineLayout,
    horizontal: vk::Pipeline,
    vertical: vk::Pipeline,
}

impl BlurPipelines {
    /// `dsl` is the UI's sampled-texture set layout, shared with the UI pipelines.
    pub(super) fn new(ctx: &GpuContext, dsl: vk::DescriptorSetLayout, format: vk::Format) -> Self {
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&dsl));
        let layout = unsafe { ctx.device().create_pipeline_layout(&info, None) }
            .expect("create backdrop blur layout");
        let shader = ctx.compile_shader(SHADER, "backdrop blur");
        // A fullscreen triangle generated in the vertex shader: no vertex input at all.
        let binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(0)
            .input_rate(vk::VertexInputRate::VERTEX);
        let horizontal = create_ui_pipeline(
            ctx,
            shader,
            &binding,
            &[],
            layout,
            format,
            c"vs_main",
            c"fs_h",
        );
        let vertical = create_ui_pipeline(
            ctx,
            shader,
            &binding,
            &[],
            layout,
            format,
            c"vs_main",
            c"fs_v",
        );
        unsafe { ctx.device().destroy_shader_module(shader, None) };
        Self {
            layout,
            horizontal,
            vertical,
        }
    }

    /// Destroy the pipelines and their layout. The caller has idled the GPU.
    pub(super) fn destroy(&self, ctx: &GpuContext) {
        let device = ctx.device();
        unsafe {
            device.destroy_pipeline(self.horizontal, None);
            device.destroy_pipeline(self.vertical, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Full memory barrier between the blur's own passes and the UI draws around them. The images
/// stay in `GENERAL`, so only ordering is needed.
unsafe fn full_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

impl UiRenderer {
    /// Make sure `slot` has backdrop textures for a display of `extent`. New textures are cleared
    /// by commands recorded into `cmd`, the frame's own command buffer.
    pub(super) fn ensure_backdrop(
        &mut self,
        slot: usize,
        cmd: vk::CommandBuffer,
        extent: (u32, u32),
    ) {
        let stale = self.backdrop[slot]
            .as_ref()
            .is_none_or(|backdrop| backdrop.size != extent);
        if stale {
            // The previous textures are dropped here; the slot's fence has recycled, and any
            // textures the last submission used were retained against it.
            self.backdrop[slot] = Some(Backdrop::new(&self.ctx, cmd, self.backdrop_format, extent));
        }
    }

    /// Copy the target into the slot's backdrop and blur it. `target` is the image the UI pass draws
    /// into, currently in `target_layout`, and `extent` its size. The layout is restored when the
    /// copy is done, so the caller can resume its pass. Records only; the caller is inside no
    /// rendering pass.
    pub(super) fn capture_backdrop(
        &mut self,
        slot: usize,
        cmd: vk::CommandBuffer,
        target: vk::Image,
        target_layout: vk::ImageLayout,
        extent: (u32, u32),
    ) {
        let (mid, source, scratch, out) = {
            let backdrop = self.backdrop[slot]
                .as_ref()
                .expect("ensure_backdrop ran before a backdrop capture");
            (
                Arc::clone(&backdrop.mid),
                Arc::clone(&backdrop.source),
                Arc::clone(&backdrop.scratch),
                Arc::clone(&backdrop.out),
            )
        };
        self.slots[slot].retain(&mid);
        self.slots[slot].retain(&source);
        self.slots[slot].retain(&scratch);
        self.slots[slot].retain(&out);

        let device = self.ctx.device().clone();
        let (half_w, half_h) = source.size();
        let full = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: half_w,
                height: half_h,
            },
        };
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        // A `GENERAL` target is read in place. Any other layout (the swapchain image, drawn in
        // `COLOR_ATTACHMENT_OPTIMAL`) is moved to `TRANSFER_SRC_OPTIMAL` for the one blit that reads
        // it, then moved back.
        let general = target_layout == vk::ImageLayout::GENERAL;
        let read_layout = if general {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };
        let blur = &self.blur;
        unsafe {
            if !general {
                vulkan::image_barrier(
                    &device,
                    cmd,
                    target,
                    target_layout,
                    read_layout,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::ImageAspectFlags::COLOR,
                );
            }
            full_barrier(&device, cmd);
            // Two-step downsample: target -> half -> quarter. Each linear blit averages 2x2 texels,
            // so nothing aliases; one 4x blit would sample only four texels per output pixel.
            let blit = |from: vk::Image,
                        from_layout: vk::ImageLayout,
                        from_size: (u32, u32),
                        to: vk::Image,
                        to_size: (u32, u32)| {
                let region = vk::ImageBlit::default()
                    .src_subresource(layers)
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: from_size.0 as i32,
                            y: from_size.1 as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(layers)
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: to_size.0 as i32,
                            y: to_size.1 as i32,
                            z: 1,
                        },
                    ]);
                device.cmd_blit_image(
                    cmd,
                    from,
                    from_layout,
                    to,
                    vk::ImageLayout::GENERAL,
                    &[region],
                    vk::Filter::LINEAR,
                );
            };
            blit(target, read_layout, extent, mid.image(), mid.size());
            if !general {
                // The target only has to be readable for the one blit above; hand it back to the
                // UI pass in the layout it was drawn in.
                vulkan::image_barrier(
                    &device,
                    cmd,
                    target,
                    read_layout,
                    target_layout,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::ImageAspectFlags::COLOR,
                );
            }
            full_barrier(&device, cmd);
            blit(
                mid.image(),
                vk::ImageLayout::GENERAL,
                mid.size(),
                source.image(),
                source.size(),
            );
            full_barrier(&device, cmd);

            // Four horizontal and four vertical passes, ping-ponging scratch and out. Each small pass
            // adds only a little blur, so the result is smooth; the radius comes from the rounds.
            let steps = [
                (&source, &scratch, blur.horizontal),
                (&scratch, &out, blur.vertical),
                (&out, &scratch, blur.horizontal),
                (&scratch, &out, blur.vertical),
                (&out, &scratch, blur.horizontal),
                (&scratch, &out, blur.vertical),
                (&out, &scratch, blur.horizontal),
                (&scratch, &out, blur.vertical),
            ];
            for (from, to, pipeline) in steps {
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(to.view())
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE);
                device.cmd_begin_rendering(
                    cmd,
                    &vk::RenderingInfo::default()
                        .render_area(full)
                        .layer_count(1)
                        .color_attachments(std::slice::from_ref(&attachment)),
                );
                device.cmd_set_viewport(
                    cmd,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: half_w as f32,
                        height: half_h as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                device.cmd_set_scissor(cmd, 0, &[full]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    blur.layout,
                    0,
                    &[from.sampled_descriptor_set()],
                    &[],
                );
                device.cmd_draw(cmd, 3, 1, 0, 0);
                device.cmd_end_rendering(cmd);
                full_barrier(&device, cmd);
            }
        }
    }

    /// The descriptor set glass samples for `slot`. `ensure_backdrop` must have run.
    pub(super) fn backdrop_set(&self, slot: usize) -> vk::DescriptorSet {
        self.backdrop[slot]
            .as_ref()
            .expect("ensure_backdrop ran before a backdrop draw")
            .out
            .sampled_descriptor_set()
    }

    /// The blurred result `slot`'s glass samples, for retention.
    pub(super) fn backdrop_texture(&self, slot: usize) -> Arc<GpuTexture> {
        Arc::clone(
            &self.backdrop[slot]
                .as_ref()
                .expect("ensure_backdrop ran before a backdrop draw")
                .out,
        )
    }
}
