use super::decode::mip_levels_for;
use super::sampler::SamplerDesc;
use super::{TextureFormat, TextureOptions};
use crate::gfx::metal::{GpuContext, GpuImage};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLOrigin, MTLPixelFormat,
    MTLSamplerState, MTLSize, MTLTexture, MTLTextureSwizzle, MTLTextureSwizzleChannels,
    MTLTextureType, MTLTextureUsage,
};
use std::ops::Range;
use std::sync::Arc;

pub struct GpuTexture {
    pub(crate) ctx: Arc<GpuContext>,
    pub(crate) image: GpuImage,
    /// Swizzled view (or same as `image.texture` when identity swizzle).
    pub(crate) view: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(crate) sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_levels: u32,
    format: TextureFormat,
    pub(super) mtl_format: MTLPixelFormat,
}

unsafe impl Send for GpuTexture {}
unsafe impl Sync for GpuTexture {}

impl GpuTexture {
    pub fn from_rgba8(
        ctx: &Arc<GpuContext>,
        label: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
        options: TextureOptions,
    ) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        assert_eq!(
            rgba.len(),
            (w as usize) * (h as usize) * 4,
            "{label}: expected {} bytes for {w}x{h} RGBA8, got {}",
            (w as usize) * (h as usize) * 4,
            rgba.len()
        );
        let mip_levels = match options.mip_levels {
            Some(n) => n.clamp(1, mip_levels_for(w, h)),
            None => mip_levels_for(w, h),
        };
        let mtl_format = options.format.mtl_format();
        let image = ctx.create_image_mipmapped(
            label,
            w,
            h,
            mip_levels,
            mtl_format,
            MTLTextureUsage::ShaderRead,
        );
        let staging = ctx.upload_buffer(&format!("{label} staging"), rgba);
        let row_bytes = (w as usize) * 4;
        ctx.one_shot(|cmd| {
            let blit = cmd.blitCommandEncoder().expect("blit encoder");
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &staging.buffer,
                    0,
                    row_bytes,
                    row_bytes * (h as usize),
                    MTLSize {
                        width: w as usize,
                        height: h as usize,
                        depth: 1,
                    },
                    &image.texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            if mip_levels > 1 {
                blit.generateMipmapsForTexture(&image.texture);
            }
            blit.endEncoding();
        });
        Self::finish(
            ctx,
            image,
            None,
            options.sampler,
            (w, h),
            mip_levels,
            options.format,
        )
    }

    pub fn blank(
        ctx: &Arc<GpuContext>,
        label: &str,
        width: u32,
        height: u32,
        options: TextureOptions,
    ) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let mip_levels = match options.mip_levels {
            Some(n) => n.clamp(1, mip_levels_for(w, h)),
            None => mip_levels_for(w, h),
        };
        let mtl_format = options.format.mtl_format();
        let image = ctx.create_image_mipmapped(
            label,
            w,
            h,
            mip_levels,
            mtl_format,
            MTLTextureUsage::ShaderRead,
        );
        Self::finish(
            ctx,
            image,
            None,
            options.sampler,
            (w, h),
            mip_levels,
            options.format,
        )
    }

    pub fn white(ctx: &Arc<GpuContext>) -> Self {
        Self::from_rgba8(
            ctx,
            "default white",
            &[255, 255, 255, 255],
            1,
            1,
            TextureOptions {
                format: TextureFormat::Srgb,
                sampler: SamplerDesc::default(),
                mip_levels: Some(1),
            },
        )
    }

    pub fn render_target(
        ctx: &Arc<GpuContext>,
        label: &str,
        width: u32,
        height: u32,
        format: MTLPixelFormat,
    ) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let image = ctx.create_image(
            label,
            w,
            h,
            format,
            MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
        );
        Self::finish(
            ctx,
            image,
            None,
            SamplerDesc::default(),
            (w, h),
            1,
            TextureFormat::Linear,
        )
    }

    /// Create an uninitialized texture with optional swizzle (e.g. `(1, 1, 1, R)`
    /// for R8 glyph coverage pages).
    pub(crate) fn create_uninitialized(
        ctx: &Arc<GpuContext>,
        label: &str,
        size: (u32, u32),
        format: MTLPixelFormat,
        swizzle: Option<MTLTextureSwizzleChannels>,
        sampler_desc: SamplerDesc,
        public_format: TextureFormat,
    ) -> Self {
        let w = size.0.max(1);
        let h = size.1.max(1);
        let image = ctx.create_image(label, w, h, format, MTLTextureUsage::ShaderRead);
        Self::finish(ctx, image, swizzle, sampler_desc, (w, h), 1, public_format)
    }

    fn finish(
        ctx: &Arc<GpuContext>,
        image: GpuImage,
        swizzle: Option<MTLTextureSwizzleChannels>,
        sampler_desc: SamplerDesc,
        size: (u32, u32),
        mip_levels: u32,
        format: TextureFormat,
    ) -> Self {
        let mtl_format = image.format;
        let view = if let Some(sw) = swizzle {
            unsafe {
                image
                    .texture
                    .newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                        mtl_format,
                        MTLTextureType::Type2D,
                        objc2_foundation::NSRange::new(0, mip_levels as usize),
                        objc2_foundation::NSRange::new(0, 1),
                        sw,
                    )
            }
            .expect("create swizzled Metal texture view")
        } else {
            image.texture.clone()
        };
        let sampler = ctx.get_or_create_sampler(&sampler_desc);
        ctx.live_textures
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            ctx: Arc::clone(ctx),
            image,
            view,
            sampler,
            width: size.0,
            height: size.1,
            mip_levels,
            format,
            mtl_format,
        }
    }

    pub fn texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.view
    }

    pub fn raw_texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.image.texture
    }

    pub fn sampler(&self) -> &ProtocolObject<dyn MTLSamplerState> {
        &self.sampler
    }

    pub fn image_allocation_size(&self) -> u64 {
        self.image.allocation_size()
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

    pub fn estimated_bytes(&self) -> u64 {
        let bpp = match self.mtl_format {
            MTLPixelFormat::R8Unorm => 1u64,
            _ => 4u64,
        };
        let base = (self.width as u64) * (self.height as u64) * bpp;
        if self.mip_levels <= 1 {
            base
        } else {
            (base * 4) / 3
        }
    }

    #[allow(dead_code)]
    pub(crate) fn coverage_swizzle() -> MTLTextureSwizzleChannels {
        MTLTextureSwizzleChannels {
            red: MTLTextureSwizzle::One,
            green: MTLTextureSwizzle::One,
            blue: MTLTextureSwizzle::One,
            alpha: MTLTextureSwizzle::Red,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn _mip_range(&self) -> Range<usize> {
        0..self.mip_levels as usize
    }
}

impl Drop for GpuTexture {
    fn drop(&mut self) {
        self.ctx
            .live_textures
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}
