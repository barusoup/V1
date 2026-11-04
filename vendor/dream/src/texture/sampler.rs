use crate::gfx::vulkan::GpuContext;
use ash::vk;

/// Magnification/minification filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    #[default]
    Linear,
    Nearest,
}

/// Filter between mip levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MipFilter {
    #[default]
    Linear,
    Nearest,
}

/// Out-of-range UV behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// Sampler state. Hashed and cached: equal descriptions share one
/// `vk::Sampler`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct SamplerDesc {
    pub filter: Filter,
    pub mip: MipFilter,
    pub address: AddressMode,
    /// Anisotropic filtering (clamped to the device limit). Costs nothing when
    /// it is not needed; enable it for oblique 3D surfaces, not for UI glyphs.
    pub anisotropy: bool,
}

impl SamplerDesc {
    /// Pixel-art sampling: hard edges, no mip smoothing.
    pub const PIXEL: Self = Self {
        filter: Filter::Nearest,
        mip: MipFilter::Nearest,
        address: AddressMode::ClampToEdge,
        anisotropy: false,
    };

    /// Smooth sampling with full anisotropy: the default for 3D materials.
    pub const ANISOTROPIC: Self = Self {
        filter: Filter::Linear,
        mip: MipFilter::Linear,
        address: AddressMode::ClampToEdge,
        anisotropy: true,
    };
}

/// Create a `vk::Sampler` from a [`SamplerDesc`]. Shared by textures and any
/// custom sampling the game needs.
pub fn create_sampler(
    ctx: &GpuContext,
    label: &str,
    desc: SamplerDesc,
    mip_levels: u32,
) -> vk::Sampler {
    let filter = match desc.filter {
        Filter::Linear => vk::Filter::LINEAR,
        Filter::Nearest => vk::Filter::NEAREST,
    };
    let mipmap = match desc.mip {
        MipFilter::Linear => vk::SamplerMipmapMode::LINEAR,
        MipFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
    };
    let address = match desc.address {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    };
    let mut info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(mipmap)
        .address_mode_u(address)
        .address_mode_v(address)
        .address_mode_w(address)
        .min_lod(0.0)
        .max_lod(if mip_levels > 1 {
            vk::LOD_CLAMP_NONE
        } else {
            0.0
        });
    if desc.anisotropy {
        let props = unsafe {
            ctx.instance()
                .get_physical_device_properties(ctx.physical_device())
        };
        info = info
            .anisotropy_enable(true)
            .max_anisotropy(props.limits.max_sampler_anisotropy.min(16.0));
    }
    let _ = label;
    unsafe { ctx.device().create_sampler(&info, None) }.expect("create sampler")
}
