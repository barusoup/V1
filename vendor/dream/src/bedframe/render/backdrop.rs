//! Frosted-glass backdrop: a blurred copy of the UI frame drawn so far.
//!
//! When a UI pass reaches its first `CmdTexture::Backdrop` command, the pass is paused: the
//! display image is copied at half resolution, blurred with two horizontal and two vertical
//! passes, and the pass resumes with the blurred result available to glass draws. Everything
//! drawn before the glass is therefore what the glass blurs.
//!
//! Each frame slot owns its own backdrop textures, so a slot still in flight on the GPU is never
//! written by the next frame.

use super::{UiRenderer, create_ui_pipeline};
use crate::gfx::vulkan::GpuContext;
use crate::texture::{self, GpuTexture};
use ash::vk;
use std::sync::Arc;

const SHADER: &str = include_str!("../backdrop.wgsl");

/// Half-resolution textures for one frame slot: the copy of the frame, the blur scratch, and the
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
    pub(super) fn new(ctx: &Arc<GpuContext>, format: vk::Format, size: (u32, u32)) -> Self {
        // Quarter resolution: the blur radius is then four times wider in display pixels, so a
        // bright feature spreads into a broad wash instead of a small spot.
        let quarter = |n: u32| ((n + 3) / 4).max(1);
        let half = |n: u32| ((n + 1) / 2).max(1);
        let make = |label: &str, w: u32, h: u32| {
            Arc::new(GpuTexture::render_target(ctx, label, w, h, format))
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
    pub(super) fn new(ctx: &GpuContext, format: vk::Format) -> Self {
        let dsl = texture::create_sampled_texture_set_layout(ctx);
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
    /// Make sure `slot` has backdrop textures for a display of `extent`.
    pub(super) fn ensure_backdrop(&mut self, slot: usize, extent: (u32, u32)) {
        let stale = self.backdrop[slot]
            .as_ref()
            .is_none_or(|backdrop| backdrop.size != extent);
        if stale {
            // The previous textures are dropped here; the slot's fence has recycled, and any
            // textures the last submission used were retained against it.
            self.backdrop[slot] = Some(Backdrop::new(&self.ctx, self.backdrop_format, extent));
        }
    }

    /// Copy the display into the slot's backdrop and blur it. `target` is the display image, in
    /// `GENERAL`, and `extent` its size. Records only; the caller is inside no rendering pass.
    pub(super) fn capture_backdrop(
        &mut self,
        slot: usize,
        cmd: vk::CommandBuffer,
        target: vk::Image,
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
        let blur = &self.blur;
        unsafe {
            full_barrier(&device, cmd);
            // Two-step downsample: display -> half -> quarter. Each linear blit averages 2x2 texels, so
            // nothing aliases; one 4x blit would sample only four texels per output pixel.
            let blit =
                |from: vk::Image, from_size: (u32, u32), to: vk::Image, to_size: (u32, u32)| {
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
                        vk::ImageLayout::GENERAL,
                        to,
                        vk::ImageLayout::GENERAL,
                        &[region],
                        vk::Filter::LINEAR,
                    );
                };
            blit(target, extent, mid.image(), mid.size());
            full_barrier(&device, cmd);
            blit(mid.image(), mid.size(), source.image(), source.size());
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
