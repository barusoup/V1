use super::GpuContext;
pub use crate::gfx::present::PresentMode;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSSize as CGSize;
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SwapchainOptions {
    pub present_mode: PresentMode,
    pub prefer_unorm: bool,
}

pub struct Swapchain {
    pub format: MTLPixelFormat,
    pub width: u32,
    pub height: u32,
    pub present_mode: PresentMode,
    pub prefer_unorm: bool,
}

impl Swapchain {
    pub fn new(ctx: &GpuContext, width: u32, height: u32, options: SwapchainOptions) -> Self {
        let format = if options.prefer_unorm {
            MTLPixelFormat::BGRA8Unorm
        } else {
            MTLPixelFormat::BGRA8Unorm_sRGB
        };
        let width = width.max(1);
        let height = height.max(1);
        configure_layer(ctx.layer(), format, width, height, options.present_mode);
        Self {
            format,
            width,
            height,
            present_mode: options.present_mode,
            prefer_unorm: options.prefer_unorm,
        }
    }

    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        configure_layer(
            ctx.layer(),
            self.format,
            self.width,
            self.height,
            self.present_mode,
        );
    }

    pub fn set_present_mode(&mut self, ctx: &GpuContext, mode: PresentMode) {
        self.present_mode = mode;
        configure_layer(ctx.layer(), self.format, self.width, self.height, mode);
    }

    pub fn next_drawable(
        &self,
        ctx: &GpuContext,
    ) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        ctx.layer().nextDrawable()
    }
}

fn configure_layer(
    layer: &CAMetalLayer,
    format: MTLPixelFormat,
    width: u32,
    height: u32,
    present_mode: PresentMode,
) {
    layer.setPixelFormat(format);
    layer.setDrawableSize(CGSize::new(width as f64, height as f64));
    layer.setDisplaySyncEnabled(present_mode == PresentMode::Fifo);
}
