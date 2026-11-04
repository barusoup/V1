use crate::gfx::GpuImage;
use crate::gfx::vulkan::GpuContext;
use ash::vk;
use std::sync::Arc;

/// Total tight bytes for `layers` layers of a res^2 image over all mips.
pub(super) fn image_bytes(res: u32, mips: u32, layers: u64, bpp: u64) -> u64 {
    let mut out = 0u64;
    for _ in 0..layers {
        for m in 0..mips {
            let r = (res >> m).max(1) as u64;
            out += r * r * bpp;
        }
    }
    out
}

pub(super) fn upload_image_layers(
    ctx: &Arc<GpuContext>,
    label: &str,
    res: u32,
    layers: &[Vec<u8>],
    format: vk::Format,
) -> GpuImage {
    let layer_count = layers.len().max(1) as u32;
    let mips = crate::texture::mip_levels_for(res, res);
    let bytes: Vec<u8> = layers.iter().flatten().copied().collect();
    let mut staging =
        ctx.create_transfer_buffer("material array staging", bytes.len().max(4) as u64);
    staging.write(0, &bytes);
    let image = ctx.create_image_array_mips(
        label,
        res,
        res,
        format,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST,
        layer_count,
        mips,
    );
    let device = ctx.device().clone();
    ctx.one_shot(|cmd| unsafe {
        let full_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mips,
            base_array_layer: 0,
            layer_count,
        };
        let to_dst = vk::ImageMemoryBarrier::default()
            .image(image.image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .subresource_range(full_range);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        );
        let mut offset = 0u64;
        let mut regions = Vec::new();
        for (layer, pixels) in layers.iter().enumerate() {
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: layer as u32,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: res,
                        height: res,
                        depth: 1,
                    }),
            );
            offset += pixels.len() as u64;
        }
        device.cmd_copy_buffer_to_image(
            cmd,
            staging.buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        let to_general = vk::ImageMemoryBarrier::default()
            .image(image.image)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
            .subresource_range(full_range);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_general],
        );

        // Use the same hardware blit path as GpuTexture::from_rgba8 so source
        // arrays get identical sRGB/filter semantics to mesh material maps.
        let mut src_w = res;
        let mut src_h = res;
        for level in 1..mips {
            let dst_w = (res >> level).max(1);
            let dst_h = (res >> level).max(1);
            for layer in 0..layer_count {
                let blit = vk::ImageBlit::default()
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level - 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    })
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: src_w as i32,
                            y: src_h as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level,
                        base_array_layer: layer,
                        layer_count: 1,
                    })
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: dst_w as i32,
                            y: dst_h as i32,
                            z: 1,
                        },
                    ]);
                device.cmd_blit_image(
                    cmd,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }
            let barrier = vk::ImageMemoryBarrier::default()
                .image(image.image)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: level,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count,
                });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            src_w = dst_w;
            src_h = dst_h;
        }
    });
    image
}
