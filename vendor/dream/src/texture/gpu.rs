use super::decode::mip_levels_for;
use super::sampler::SamplerDesc;
use super::{TextureFormat, TextureOptions};
use crate::gfx::vulkan::{self, GpuContext, GpuImage};
use ash::vk;
use std::sync::Arc;

/// The standard texture descriptor set: binding 0, combined image sampler,
/// visible to the fragment stage. Any pipeline whose set layout matches this
/// structure can bind [`GpuTexture::descriptor_set`] at that set index.
pub const TEXTURE_SET_BINDINGS: [vk::DescriptorSetLayoutBinding<'static>; 1] =
    [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: std::ptr::null(),
        _marker: std::marker::PhantomData,
    }];

/// Create a descriptor set layout structurally identical to the one every
/// [`GpuTexture`] is allocated from, for use in custom pipelines.
pub fn create_texture_set_layout(ctx: &GpuContext) -> vk::DescriptorSetLayout {
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&TEXTURE_SET_BINDINGS);
    unsafe { ctx.device().create_descriptor_set_layout(&info, None) }
        .expect("create texture descriptor set layout")
}

/// The engine's WGSL texture contract: naga emits a WGSL `texture_2d` and
/// `sampler` as *separate* SPIR-V resources — a sampled image at binding 0
/// and a sampler at binding 1. Pipelines compiled from the engine's own
/// shaders bind [`GpuTexture::sampled_descriptor_set`], whose sets are
/// allocated against this layout.
pub const SAMPLED_TEXTURE_SET_BINDINGS: [vk::DescriptorSetLayoutBinding<'static>; 2] = [
    vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: std::ptr::null(),
        _marker: std::marker::PhantomData,
    },
    vk::DescriptorSetLayoutBinding {
        binding: 1,
        descriptor_type: vk::DescriptorType::SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: std::ptr::null(),
        _marker: std::marker::PhantomData,
    },
];

/// The layout matching [`SAMPLED_TEXTURE_SET_BINDINGS`], for engine
/// pipelines (`ui.wgsl`, `model.wgsl`) built on the texture set.
pub fn create_sampled_texture_set_layout(ctx: &GpuContext) -> vk::DescriptorSetLayout {
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&SAMPLED_TEXTURE_SET_BINDINGS);
    unsafe { ctx.device().create_descriptor_set_layout(&info, None) }
        .expect("create sampled texture descriptor set layout")
}

/// A GPU texture with its sampler and descriptor set. See the module docs for
/// the lifetime and layout contract.
pub struct GpuTexture {
    pub(super) ctx: Arc<GpuContext>,
    pub(super) image: GpuImage,
    /// Shared with every texture of the same description; owned by the context.
    sampler: vk::Sampler,
    /// Descriptor pool chunk the two sets below came from; needed to free them.
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Engine-WGSL set: `SAMPLED_IMAGE` at 0, `SAMPLER` at 1 — see
    /// [`SAMPLED_TEXTURE_SET_BINDINGS`].
    sampled_set: vk::DescriptorSet,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_levels: u32,
    format: TextureFormat,
    /// The actual Vulkan format (e.g. R8 for coverage pages) — `format`
    /// is only the publicly reported bucket.
    pub(super) vk_format: vk::Format,
}

impl GpuTexture {
    /// Upload RGBA8 pixels. `pixels.len()` must equal `width * height * 4`.
    pub fn from_rgba8(
        ctx: &Arc<GpuContext>,
        label: &str,
        pixels: &[u8],
        width: u32,
        height: u32,
        options: TextureOptions,
    ) -> Self {
        assert!(
            pixels.len() == width as usize * height as usize * 4,
            "{label}: {} bytes for {width}x{height} RGBA8",
            pixels.len()
        );
        let mip_levels = options
            .mip_levels
            .unwrap_or_else(|| mip_levels_for(width, height))
            .min(mip_levels_for(width, height))
            .max(1);
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let image = ctx.create_image_mips(
            label,
            width,
            height,
            options.format.vk_format(),
            usage,
            mip_levels,
        );

        // Staging upload into level 0, then the mip chain by blit.
        let mut staging = ctx.create_transfer_buffer(label, (pixels.len() as u64).max(4));
        staging.write(0, pixels);
        let device = ctx.device().clone();
        let raw = image.image;
        let format = options.format.vk_format();
        ctx.one_shot(|cmd| unsafe {
            vulkan::image_barrier_mips(
                &device,
                cmd,
                raw,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::ImageAspectFlags::COLOR,
                mip_levels,
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer,
                raw,
                vk::ImageLayout::GENERAL,
                &[region],
            );
            generate_mips(&device, cmd, raw, format, width, height, mip_levels);
            vulkan::image_barrier_mips(
                &device,
                cmd,
                raw,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::ImageAspectFlags::COLOR,
                mip_levels,
            );
        });

        Self::finish(
            ctx,
            image,
            options.sampler,
            (width, height),
            mip_levels,
            (options.format, options.format.vk_format()),
        )
    }

    /// An all-zero texture, cleared on the GPU (no staging upload). Used for
    /// atlas pages that fill in lazily and for game render targets.
    pub fn blank(
        ctx: &Arc<GpuContext>,
        label: &str,
        width: u32,
        height: u32,
        options: TextureOptions,
    ) -> Self {
        let mip_levels = 1;
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let image = ctx.create_image_mips(
            label,
            width,
            height,
            options.format.vk_format(),
            usage,
            mip_levels,
        );
        let device = ctx.device().clone();
        let raw = image.image;
        ctx.one_shot(|cmd| unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                raw,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::ImageAspectFlags::COLOR,
            );
            device.cmd_clear_color_image(
                cmd,
                raw,
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
            vulkan::image_barrier(
                &device,
                cmd,
                raw,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::ImageAspectFlags::COLOR,
            );
        });
        Self::finish(
            ctx,
            image,
            options.sampler,
            (width, height),
            mip_levels,
            (options.format, options.format.vk_format()),
        )
    }

    /// 1x1 opaque white texture; the natural bind for untextured quads.
    pub fn white(ctx: &Arc<GpuContext>) -> Self {
        Self::from_rgba8(
            ctx,
            "white texture",
            &[255, 255, 255, 255],
            1,
            1,
            TextureOptions {
                mip_levels: Some(1),
                ..Default::default()
            },
        )
    }

    /// A blank, linearly sampled texture that can also be drawn into and copied
    /// to or from, in the caller's exact Vulkan format. Used for the UI's
    /// frosted-glass backdrop, whose targets must match the display image's
    /// format.
    ///
    /// The clear to transparent black is recorded into `cmd`, not submitted on
    /// its own, so creating one never stalls the frame it is made in. The
    /// texture is defined once `cmd` has been submitted, and it must be used
    /// only by work submitted after that.
    pub fn render_target(
        ctx: &Arc<GpuContext>,
        cmd: vk::CommandBuffer,
        label: &str,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Self {
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let image = ctx.create_image_mips(label, width, height, format, usage, 1);
        let device = ctx.device();
        let raw = image.image;
        unsafe {
            vulkan::image_barrier(
                device,
                cmd,
                raw,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::ImageAspectFlags::COLOR,
            );
            device.cmd_clear_color_image(
                cmd,
                raw,
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
            vulkan::image_barrier(
                device,
                cmd,
                raw,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::ImageAspectFlags::COLOR,
            );
        }
        // Linear and clamped: the blur reads neighbours, and the glass is
        // sampled at fractional UVs.
        let sampler = SamplerDesc {
            filter: super::sampler::Filter::Linear,
            mip: super::sampler::MipFilter::Linear,
            address: super::sampler::AddressMode::ClampToEdge,
            anisotropy: false,
        };
        Self::finish(
            ctx,
            image,
            sampler,
            (width, height),
            1,
            (TextureFormat::Linear, format),
        )
    }

    /// An image with sampler and descriptor sets but no contents and no
    /// upload: the caller transitions UNDEFINED itself and records the
    /// first copy. `format` is the *actual* Vulkan format (coverage pages
    /// are R8, not RGBA); `public_format` is the reported [`TextureFormat`]
    /// for API compat. Internal — atlas pages only.
    pub(crate) fn create_uninitialized(
        ctx: &Arc<GpuContext>,
        label: &str,
        size: (u32, u32),
        format: vk::Format,
        swizzle: vk::ComponentMapping,
        sampler_desc: SamplerDesc,
        public_format: TextureFormat,
    ) -> Self {
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let image =
            ctx.create_image_swizzled(label, size.0.max(1), size.1.max(1), format, usage, swizzle);
        Self::finish(ctx, image, sampler_desc, size, 1, (public_format, format))
    }

    fn finish(
        ctx: &Arc<GpuContext>,
        image: GpuImage,
        sampler_desc: SamplerDesc,
        size: (u32, u32),
        mip_levels: u32,
        formats: (TextureFormat, vk::Format),
    ) -> Self {
        let (width, height) = size;
        let (format, vk_format) = formats;
        let sampler = ctx.shared_sampler(sampler_desc, mip_levels);
        let (pool, [set, sampled_set]) = ctx.alloc_texture_sets();
        let combined = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let image_only = vk::DescriptorImageInfo::default()
            .image_view(image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let sampler_only = vk::DescriptorImageInfo::default().sampler(sampler);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&combined)),
            vk::WriteDescriptorSet::default()
                .dst_set(sampled_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&image_only)),
            vk::WriteDescriptorSet::default()
                .dst_set(sampled_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&sampler_only)),
        ];
        unsafe { ctx.device().update_descriptor_sets(&writes, &[]) };
        Self {
            ctx: Arc::clone(ctx),
            image,
            sampler,
            pool,
            set,
            sampled_set,
            width,
            height,
            mip_levels,
            format,
            vk_format,
        }
    }

    /// Descriptor set binding the texture and its sampler; bind it at the
    /// texture set of any pipeline built with [`create_texture_set_layout`].
    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// Descriptor set for engine WGSL pipelines: sampled image at binding 0,
    /// sampler at binding 1 ([`create_sampled_texture_set_layout`]).
    pub fn sampled_descriptor_set(&self) -> vk::DescriptorSet {
        self.sampled_set
    }

    pub fn view(&self) -> vk::ImageView {
        self.image.view
    }

    /// Bytes backing the GPU image allocation (diagnostics).
    pub fn image_allocation_size(&self) -> u64 {
        self.image.allocation_size()
    }

    pub fn image(&self) -> vk::Image {
        self.image.image
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Approximate GPU payload in bytes, mip chain included.
    pub fn estimated_bytes(&self) -> u64 {
        let bytes_per_texel: u64 = match self.vk_format {
            vk::Format::R8_UNORM | vk::Format::R8_SRGB => 1,
            vk::Format::R8G8_UNORM | vk::Format::R8G8_SRGB => 2,
            _ => 4,
        };
        let mut bytes = 0u64;
        let mut w = self.width as u64;
        let mut h = self.height as u64;
        for _ in 0..self.mip_levels {
            bytes += w * h * bytes_per_texel;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        bytes
    }
}

impl Drop for GpuTexture {
    fn drop(&mut self) {
        // The sampler, layouts and pool chunk are shared: the context owns them.
        self.ctx
            .free_texture_sets(self.pool, [self.set, self.sampled_set]);
    }
}

/// Blit the mip chain for `image` (level 0 must hold the uploaded pixels).
/// `GENERAL` in, `GENERAL` out: legal for both transfer sides.
#[allow(clippy::too_many_arguments)]
unsafe fn generate_mips(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) {
    // Level 0 was written by the buffer copy. Nothing above orders that write
    // before the first blit reads it (SYNC-HAZARD-READ-AFTER-WRITE on
    // vkCmdBlitImage), so make it visible to transfer reads first.
    if mip_levels > 1 {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
    let mut src_w = width;
    let mut src_h = height;
    for level in 1..mip_levels {
        let dst_w = (width >> level).max(1);
        let dst_h = (height >> level).max(1);
        let blit = vk::ImageBlit::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: level - 1,
                base_array_layer: 0,
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
                base_array_layer: 0,
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
        unsafe {
            device.cmd_blit_image(
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                image,
                vk::ImageLayout::GENERAL,
                &[blit],
                vk::Filter::LINEAR,
            );
        }
        // Make level `level`'s writes visible before it becomes the next
        // blit's source and before shaders sample it.
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: level,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        src_w = dst_w;
        src_h = dst_h;
    }
    let _ = format;
}
