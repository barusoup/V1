use super::GpuContext;
use super::barriers::is_srgb;
use ash::vk;
use ash::vk::Handle;
use std::sync::Arc;

// --- Present mode and options ----------------------------------------------

/// How finished frames reach the screen.
///
/// - [`Self::Fifo`] is vsync: frames are queued and shown in order, the
///   renderer never draws a frame that will not be shown, the loop blocks
///   on the display and the GPU idles between refreshes. Lowest power. This
///   is the default and is always supported.
/// - [`Self::Mailbox`] is latest-frame-wins: lower latency, but frames that
///   are replaced before they are shown were still rendered, so an app that
///   never waits on the display must pace itself or it pins the GPU.
/// - [`Self::Immediate`] presents without waiting for vsync and may tear.
///
/// A mode the surface does not offer falls back to [`Self::Fifo`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PresentMode {
    /// Vertical sync (`FIFO`). The default.
    #[default]
    Fifo,
    /// Latest-frame-wins (`MAILBOX`), falling back to FIFO.
    Mailbox,
    /// No vsync (`IMMEDIATE`), falling back to FIFO. May tear.
    Immediate,
}

/// Swapchain choices made when the swapchain is created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SwapchainOptions {
    /// Requested present mode; see [`PresentMode`].
    pub present_mode: PresentMode,
    /// Prefer a UNORM surface format (`B8G8R8A8_UNORM` or `R8G8B8A8_UNORM`,
    /// SRGB_NONLINEAR colour space) over an sRGB one, so the presented bytes
    /// are exactly what the renderer wrote. Falls back to the sRGB-first
    /// choice when no such format is offered.
    pub prefer_unorm: bool,
}

/// Pick the surface format. With `prefer_unorm`, a UNORM format in the
/// SRGB_NONLINEAR colour space wins if the surface offers one; otherwise the
/// first sRGB format, then the first format reported.
///
/// # Panics
///
/// Panics when `formats` is empty: a surface always reports at least one.
pub fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    prefer_unorm: bool,
) -> vk::SurfaceFormatKHR {
    if prefer_unorm
        && let Some(format) = formats.iter().find(|f| {
            matches!(
                f.format,
                vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
            ) && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
    {
        return *format;
    }
    formats
        .iter()
        .find(|f| is_srgb(f.format))
        .or_else(|| formats.first())
        .copied()
        .expect("surface reports at least one format")
}

/// Map the requested [`PresentMode`] onto what the surface offers. FIFO is
/// mandatory in Vulkan, so it is the fallback for every unsupported mode.
pub fn choose_present_mode(
    modes: &[vk::PresentModeKHR],
    wanted: PresentMode,
) -> vk::PresentModeKHR {
    let candidate = match wanted {
        PresentMode::Fifo => return vk::PresentModeKHR::FIFO,
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
    };
    if modes.contains(&candidate) {
        candidate
    } else {
        vk::PresentModeKHR::FIFO
    }
}

// --- Swapchain -------------------------------------------------------------

pub struct Swapchain {
    ctx: Arc<GpuContext>,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    /// Whether the images can be blitted from (`TRANSFER_SRC` usage and a
    /// format with `BLIT_SRC` optimal tiling). See [`Self::supports_transfer_src`].
    transfer_src: bool,
}

impl Swapchain {
    /// Whether the images can be read by a blit or copy. The UI's frosted-glass
    /// backdrop copies the frame out of the swapchain image when a frame is
    /// composed directly into it; without this, that capture is skipped.
    pub fn supports_transfer_src(&self) -> bool {
        self.transfer_src
    }

    /// `old` is the outgoing swapchain for this surface; pass
    /// `vk::SwapchainKHR::null()` on first creation. Recreating without handing
    /// the old handle back fails with `ERROR_NATIVE_WINDOW_IN_USE_KHR`.
    /// Uses [`SwapchainOptions::default`]; see [`Self::new_with`].
    pub fn new(ctx: &Arc<GpuContext>, width: u32, height: u32, old: vk::SwapchainKHR) -> Self {
        Self::new_with(ctx, width, height, old, SwapchainOptions::default())
    }

    /// [`Self::new`] with explicit present mode and format preferences.
    pub fn new_with(
        ctx: &Arc<GpuContext>,
        width: u32,
        height: u32,
        old: vk::SwapchainKHR,
        options: SwapchainOptions,
    ) -> Self {
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
        let format = choose_surface_format(&formats, options.prefer_unorm);
        let modes = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_present_modes(ctx.physical_device, ctx.surface)
        }
        .unwrap_or_default();
        let present_mode = choose_present_mode(&modes, options.present_mode);

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

        // Blitting out of the swapchain (the glass backdrop capture) needs
        // TRANSFER_SRC usage and a format the driver can blit from. Ask for the
        // usage only when the surface offers it.
        let blit_format = unsafe {
            ctx.instance()
                .get_physical_device_format_properties(ctx.physical_device, format.format)
        }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::BLIT_SRC);
        let transfer_src = blit_format
            && caps
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::TRANSFER_SRC);
        let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if transfer_src {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(ctx.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
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
            transfer_src,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(format: vk::Format, color_space: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space,
        }
    }

    #[test]
    fn swapchain_present_mode_default_is_fifo() {
        assert_eq!(PresentMode::default(), PresentMode::Fifo);
        assert_eq!(SwapchainOptions::default().present_mode, PresentMode::Fifo);
        assert!(!SwapchainOptions::default().prefer_unorm);
    }

    #[test]
    fn swapchain_present_mode_uses_requested_mode_when_offered() {
        let modes = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];
        assert_eq!(
            choose_present_mode(&modes, PresentMode::Mailbox),
            vk::PresentModeKHR::MAILBOX
        );
        assert_eq!(
            choose_present_mode(&modes, PresentMode::Fifo),
            vk::PresentModeKHR::FIFO
        );
    }

    #[test]
    fn swapchain_present_mode_falls_back_to_fifo() {
        let fifo_only = [vk::PresentModeKHR::FIFO];
        assert_eq!(
            choose_present_mode(&fifo_only, PresentMode::Mailbox),
            vk::PresentModeKHR::FIFO
        );
        assert_eq!(
            choose_present_mode(&fifo_only, PresentMode::Immediate),
            vk::PresentModeKHR::FIFO
        );
        let immediate = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE];
        assert_eq!(
            choose_present_mode(&immediate, PresentMode::Immediate),
            vk::PresentModeKHR::IMMEDIATE
        );
    }

    #[test]
    fn swapchain_format_default_prefers_srgb() {
        let formats = [
            surface(
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
            surface(vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
        ];
        assert_eq!(
            choose_surface_format(&formats, false).format,
            vk::Format::B8G8R8A8_SRGB
        );
    }

    #[test]
    fn swapchain_format_prefer_unorm_picks_unorm_when_offered() {
        let formats = [
            surface(vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            surface(
                vk::Format::R8G8B8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
        ];
        assert_eq!(
            choose_surface_format(&formats, true).format,
            vk::Format::R8G8B8A8_UNORM
        );
    }

    #[test]
    fn swapchain_format_prefer_unorm_needs_srgb_nonlinear_space() {
        // A UNORM format in another colour space is not the encoded-bytes
        // target the gamma blend needs; the sRGB fallback wins instead.
        let formats = [
            surface(
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            ),
            surface(vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
        ];
        assert_eq!(
            choose_surface_format(&formats, true).format,
            vk::Format::B8G8R8A8_SRGB
        );
    }

    #[test]
    fn swapchain_format_prefer_unorm_without_unorm_falls_back() {
        let formats = [surface(
            vk::Format::B8G8R8A8_SRGB,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        )];
        assert_eq!(
            choose_surface_format(&formats, true).format,
            vk::Format::B8G8R8A8_SRGB
        );
        // With no sRGB format either, the first reported format is taken.
        let linear = [surface(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        )];
        assert_eq!(
            choose_surface_format(&linear, true).format,
            vk::Format::R16G16B16A16_SFLOAT
        );
    }
}
