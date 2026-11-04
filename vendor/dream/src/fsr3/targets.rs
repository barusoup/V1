use crate::gfx::{GpuContext, GpuImage};
use ash::vk;
use std::sync::Arc;

use super::{FMT_R8, FMT_R16F, FMT_R32F, FMT_RG16F, FMT_RGBA8, FMT_RGBA16F};

pub(super) struct FsrTargets {
    // Render resolution.
    pub(super) dilated_motion_vectors: GpuImage,
    pub(super) dilated_depth: GpuImage,
    pub(super) farthest_depth: GpuImage,
    pub(super) current_luma: [GpuImage; 2],
    pub(super) luma_history: [GpuImage; 2],
    pub(super) luma_instability: GpuImage,
    pub(super) dilated_reactive_masks: GpuImage,
    pub(super) accumulation: [GpuImage; 2],
    // Half render resolution.
    pub(super) farthest_depth_mip1: GpuImage,
    pub(super) shading_change: GpuImage,
    pub(super) spd_mips: [GpuImage; 3],
    // Display resolution.
    pub(super) internal_upscaled_color: [GpuImage; 2],
    pub(super) new_locks: GpuImage,
    pub(super) upscaled_output: GpuImage,
}

impl FsrTargets {
    pub(super) fn build(ctx: &Arc<GpuContext>, render: (u32, u32), display: (u32, u32)) -> Self {
        let (rw, rh) = render;
        let (dw, dh) = display;
        let half = (rw.div_ceil(2), rh.div_ceil(2));
        let quarter = (rw.div_ceil(4), rh.div_ceil(4));
        let eighth = (rw.div_ceil(8), rh.div_ceil(8));

        // Storage images must also be sampleable (history is read back) and
        // clearable (history restarts wipe accumulation state).
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_DST;
        let make = |label: &str, (w, h): (u32, u32), format: vk::Format| {
            ctx.create_image(label, w, h, format, usage)
        };

        let targets = FsrTargets {
            dilated_motion_vectors: make("fsr dl mv", (rw, rh), FMT_RG16F),
            dilated_depth: make("fsr dl depth", (rw, rh), FMT_R32F),
            farthest_depth: make("fsr farthest depth", (rw, rh), FMT_R16F),
            current_luma: [
                make("fsr luma 0", (rw, rh), FMT_R16F),
                make("fsr luma 1", (rw, rh), FMT_R16F),
            ],
            luma_history: [
                make("fsr luma history 0", (rw, rh), FMT_RGBA16F),
                make("fsr luma history 1", (rw, rh), FMT_RGBA16F),
            ],
            luma_instability: make("fsr luma instability", (rw, rh), FMT_R16F),
            dilated_reactive_masks: make("fsr reactive", (rw, rh), FMT_RGBA8),
            accumulation: [
                make("fsr accumulation 0", (rw, rh), FMT_R8),
                make("fsr accumulation 1", (rw, rh), FMT_R8),
            ],
            farthest_depth_mip1: make("fsr farthest depth mip1", half, FMT_R16F),
            shading_change: make("fsr shading change", half, FMT_R8),
            spd_mips: [
                make("fsr spd mip0", half, FMT_RG16F),
                make("fsr spd mip1", quarter, FMT_RG16F),
                make("fsr spd mip2", eighth, FMT_RG16F),
            ],
            internal_upscaled_color: [
                make("fsr upscaled 0", (dw, dh), FMT_RGBA16F),
                make("fsr upscaled 1", (dw, dh), FMT_RGBA16F),
            ],
            new_locks: make("fsr new locks", (dw, dh), FMT_R8),
            upscaled_output: make("fsr output", (dw, dh), FMT_RGBA16F),
        };

        let images = [
            targets.dilated_motion_vectors.image,
            targets.dilated_depth.image,
            targets.farthest_depth.image,
            targets.current_luma[0].image,
            targets.current_luma[1].image,
            targets.luma_history[0].image,
            targets.luma_history[1].image,
            targets.luma_instability.image,
            targets.dilated_reactive_masks.image,
            targets.accumulation[0].image,
            targets.accumulation[1].image,
            targets.farthest_depth_mip1.image,
            targets.shading_change.image,
            targets.spd_mips[0].image,
            targets.spd_mips[1].image,
            targets.spd_mips[2].image,
            targets.internal_upscaled_color[0].image,
            targets.internal_upscaled_color[1].image,
            targets.new_locks.image,
            targets.upscaled_output.image,
        ];
        ctx.one_shot(|cmd| unsafe {
            for image in images {
                crate::gfx::vulkan::image_barrier(
                    ctx.device(),
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    vk::ImageAspectFlags::COLOR,
                );
            }
        });

        targets
    }
}
