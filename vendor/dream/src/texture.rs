//! Textures: decode, upload, sample, manage.
//!
//! A [`GpuTexture`] is self-contained: image, view, sampler and a descriptor
//! set allocated from its own layout. The layout is structurally the standard
//! texture set (binding 0 = combined image sampler, fragment stage), so a
//! pipeline built with [`create_texture_set_layout`] can bind any
//! [`GpuTexture`]'s set directly — see that function's docs for the contract.
//!
//! Images stay in `GENERAL` layout after upload, matching the engine-wide
//! convention (see `gfx`): `GENERAL` is legal for transfer and sampling, so no
//! per-texture layout bookkeeping exists.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use dream::gfx::vulkan::GpuContext;
//! # use dream::texture::{TextureFormat, TextureManager, TextureOptions};
//! # fn example(ctx: Arc<GpuContext>) -> Result<(), String> {
//! let mut textures = TextureManager::new(ctx.clone());
//! let stone = textures.load("assets/stone.png")?;          // sRGB, mipmapped
//! let mask = textures.load_with("assets/mask.png", TextureOptions {
//!     format: TextureFormat::Linear,                        // data, not colour
//!     ..Default::default()
//! })?;
//! // In a draw: bind stone.descriptor_set() at the pipeline's texture set.
//! # let _ = (stone, mask);
//! # Ok(())
//! # }
//! ```

mod decode;
mod gpu;
mod manager;
mod sampler;
mod upload;

pub use decode::{DecodedTexture, decode_rgba8, mip_levels_for};
pub use gpu::{
    GpuTexture, SAMPLED_TEXTURE_SET_BINDINGS, TEXTURE_SET_BINDINGS,
    create_sampled_texture_set_layout, create_texture_set_layout,
};
pub use manager::{TextureId, TextureManager, TextureStats};
pub use sampler::{AddressMode, Filter, MipFilter, SamplerDesc, create_sampler};
pub use upload::TextureUploader;

use ash::vk;

/// How decoded texel bytes are interpreted by the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TextureFormat {
    /// Colour data (PNG/JPEG photos, albedo maps). Sampling returns linear
    /// light; author bytes in sRGB as usual.
    #[default]
    Srgb,
    /// Raw data (masks, glyph coverage, normal maps). Sampling returns the
    /// bytes unchanged.
    Linear,
}

impl TextureFormat {
    pub fn vk_format(self) -> vk::Format {
        match self {
            TextureFormat::Srgb => vk::Format::R8G8B8A8_SRGB,
            TextureFormat::Linear => vk::Format::R8G8B8A8_UNORM,
        }
    }
}

/// Options for loading or creating a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureOptions {
    pub format: TextureFormat,
    pub sampler: SamplerDesc,
    /// `None` builds the full mip chain (recommended for 3D); `Some(1)` or
    /// `Some(levels)` builds exactly that many levels.
    pub mip_levels: Option<u32>,
}

impl Default for TextureOptions {
    fn default() -> Self {
        Self {
            format: TextureFormat::Srgb,
            sampler: SamplerDesc::default(),
            mip_levels: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_length() {
        assert_eq!(mip_levels_for(1, 1), 1);
        assert_eq!(mip_levels_for(1, 256), 9);
        assert_eq!(mip_levels_for(256, 256), 9);
        assert_eq!(mip_levels_for(255, 128), 8);
        assert_eq!(mip_levels_for(1000, 600), 10);
    }

    #[test]
    fn sampler_desc_dedup() {
        let mut seen = std::collections::HashSet::new();
        assert!(seen.insert(SamplerDesc::default()));
        assert!(!seen.insert(SamplerDesc::default()));
        assert!(seen.insert(SamplerDesc::PIXEL));
        assert!(seen.insert(SamplerDesc::ANISOTROPIC));
    }

    #[test]
    fn format_mapping() {
        assert_eq!(TextureFormat::Srgb.vk_format(), vk::Format::R8G8B8A8_SRGB);
        assert_eq!(
            TextureFormat::Linear.vk_format(),
            vk::Format::R8G8B8A8_UNORM
        );
    }

    #[test]
    fn decode_png_round_trip() {
        // Encode with the engine's writer, decode with the image crate: the
        // two sides of the texture pipeline must agree byte-for-byte.
        let pixels = [
            255, 0, 0, 255, //
            0, 255, 0, 128, //
            0, 0, 255, 0, //
            1, 2, 3, 4,
        ];
        let png = crate::png::encode(2, 2, &pixels);
        let decoded = decode_rgba8(&png).expect("decode our own PNG");
        assert_eq!((decoded.width, decoded.height), (2, 2));
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn decode_rejects_garbage() {
        let err = decode_rgba8(b"not an image").expect_err("garbage must fail");
        assert!(err.contains("decode image"), "unexpected error: {err}");
    }

    #[test]
    fn options_defaults_are_srgb_linear_clamp() {
        let options = TextureOptions::default();
        assert_eq!(options.format, TextureFormat::Srgb);
        assert_eq!(options.sampler, SamplerDesc::default());
        assert_eq!(options.mip_levels, None);
    }
}

/// Every WGSL shader in the engine declares textures as
/// `texture_2d` at binding 0 and `sampler` at binding 1 of the texture
/// set — matching [`SAMPLED_TEXTURE_SET_BINDINGS`]. Parsing the real
/// shaders here keeps the descriptor layout honest: a new binding
/// order breaks this test, not the GPU frame.
#[test]
fn engine_wgsl_matches_the_sampled_texture_contract() {
    // (source, group index holding the texture pair)
    let shaders: &[(&str, u32)] = &[
        (include_str!("bedframe/ui.wgsl"), 0),
        (include_str!("model/model.wgsl"), 2),
        (include_str!("present.wgsl"), 0),
        (include_str!("blit.wgsl"), 0),
    ];
    for (source, group) in shaders {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL parses");
        let mut image = false;
        let mut sampler = false;
        for (_, var) in module.global_variables.iter() {
            let Some(binding) = &var.binding else {
                continue;
            };
            if binding.group != *group {
                continue;
            }
            match (&module.types[var.ty].inner, binding.binding) {
                (naga::TypeInner::Image { .. }, 0) => image = true,
                (naga::TypeInner::Sampler { .. }, 1) => sampler = true,
                _ => {}
            }
        }
        assert!(
            image && sampler,
            "shader set {group} has texture@0 + sampler@1 (sampled contract)"
        );
        // And the layout declares exactly those two slots.
        let layout: Vec<_> = SAMPLED_TEXTURE_SET_BINDINGS
            .iter()
            .map(|b| (b.binding, b.descriptor_type))
            .collect();
        assert_eq!(
            layout,
            vec![
                (0, vk::DescriptorType::SAMPLED_IMAGE),
                (1, vk::DescriptorType::SAMPLER)
            ]
        );
    }
}
