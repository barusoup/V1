use super::GpuContext;
use super::barriers::is_srgb;
use ash::vk;
use ash::vk::Handle;
use std::sync::Arc;

// --- Swapchain -------------------------------------------------------------

pub struct Swapchain {
    ctx: Arc<GpuContext>,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    /// `old` is the outgoing swapchain for this surface; pass
    /// `vk::SwapchainKHR::null()` on first creation. Recreating without handing
    /// the old handle back fails with `ERROR_NATIVE_WINDOW_IN_USE_KHR`.
    pub fn new(ctx: &Arc<GpuContext>, width: u32, height: u32, old: vk::SwapchainKHR) -> Self {
        let caps = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)
        }
        .expect("query surface capabilities");
        let formats = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_formats(ctx.physical_device, ctx.surface)
        }
        .expect("query surface formats");
        let format = formats
            .iter()
            .find(|f| is_srgb(f.format))
            .or_else(|| formats.first())
            .copied()
            .expect("surface reports at least one format");
        let modes = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_present_modes(ctx.physical_device, ctx.surface)
        }
        .unwrap_or_default();
        let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        // One image more than the effective depth (frames in flight, at
        // least the surface minimum), so the acquire-ahead request never
        // finds every image held by frames or the presentation engine.
        let wanted = (super::MAX_FRAMES_IN_FLIGHT as u32).max(caps.min_image_count) + 1;
        let image_count = wanted.min(if caps.max_image_count == 0 {
            wanted
        } else {
            caps.max_image_count
        });

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(ctx.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(caps.current_transform)
            // The compositor decides which alpha modes it honours: take
            // OPAQUE when offered, otherwise the first supported flag.
            .composite_alpha(
                [
                    vk::CompositeAlphaFlagsKHR::OPAQUE,
                    vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                    vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
                    vk::CompositeAlphaFlagsKHR::INHERIT,
                ]
                .into_iter()
                .find(|flag| caps.supported_composite_alpha.contains(*flag))
                .expect("surface supports at least one composite alpha mode"),
            )
            .present_mode(present_mode)
            .old_swapchain(old)
            .clipped(true);

        let swapchain = unsafe { ctx.swapchain_loader.create_swapchain(&info, None) }
            .expect("create swapchain");
        if old.is_null() {
            // Present mode decides how the app is paced and how it stalls;
            // print it once so frame-timing reports can be read with it.
            eprintln!(
                "swapchain: {present_mode:?}, {image_count} images, {:?}",
                format.format
            );
        }
        let images = unsafe { ctx.swapchain_loader.get_swapchain_images(swapchain) }
            .expect("get swapchain images");
        let views = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { ctx.device.create_image_view(&view_info, None) }.expect("swapchain view")
            })
            .collect();

        Self {
            ctx: Arc::clone(ctx),
            swapchain,
            images,
            views,
            format: format.format,
            extent,
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for &view in &self.views {
                self.ctx.device.destroy_image_view(view, None);
            }
            self.ctx
                .swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
