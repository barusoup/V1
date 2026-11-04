/// How a material's alpha is intended to be used.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AlphaMode {
    #[default]
    Opaque,
    /// `discard` below the cutoff.
    Mask,
    /// Alpha blending; cannot be captured in an impostor.
    Blend,
}

/// Texture wrap mode requested by the file's sampler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wrap {
    #[default]
    Repeat,
    ClampToEdge,
    MirroredRepeat,
}

/// Texel filtering requested by the file's sampler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TexelFilter {
    /// File leaves it to the implementation — we take linear + mipmaps.
    #[default]
    Linear,
    Nearest,
}

/// `KHR_texture_transform` data attached to a texture reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct UvTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    /// When present, overrides the texture's `tex_coord` UV set.
    pub tex_coord: Option<u32>,
}

/// One material map: decoded pixels plus how the mesh addresses them.
#[derive(Clone, Debug)]
pub struct TextureRef {
    pub texture: crate::texture::DecodedTexture,
    /// `TEXCOORD_n` set index (before any transform override).
    pub tex_coord: u32,
    pub transform: Option<UvTransform>,
    pub wrap_s: Wrap,
    pub wrap_t: Wrap,
    pub mag_filter: TexelFilter,
    pub min_filter: TexelFilter,
}

impl TextureRef {
    /// Effective UV set: the transform's override wins.
    pub fn uv_set(&self) -> u32 {
        self.transform
            .and_then(|t| t.tex_coord)
            .unwrap_or(self.tex_coord)
    }
}

/// Material properties relevant to a forward renderer. The original simple
/// fields (`base_color`, `base_color_texture`) remain the source of truth for
/// the wrap-lighting path; the PBR fields extend it without changing that
/// meaning.
#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    /// sRGB base colour factor, linear multiply on the texture.
    pub base_color: [f32; 4],
    /// Decoded base colour texture (sRGB), uploaded lazily by the renderer.
    pub base_color_texture: Option<crate::texture::DecodedTexture>,
    pub double_sided: bool,
    /// Metallic (b channel) / roughness (g channel) map.
    pub metallic_roughness_texture: Option<TextureRef>,
    /// Tangent-space normal map.
    pub normal_texture: Option<TextureRef>,
    pub normal_scale: f32,
    /// Ambient occlusion map (r channel).
    pub occlusion_texture: Option<TextureRef>,
    pub occlusion_strength: f32,
    /// sRGB emissive map.
    pub emissive_texture: Option<TextureRef>,
    pub emissive_factor: [f32; 3],
    /// `KHR_materials_emissive_strength` multiplier.
    pub emissive_strength: f32,
    /// Base colour map with its sampler/transform info; pixels mirror
    /// `base_color_texture` so existing callers keep working.
    pub base_color_map: Option<TextureRef>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_mode: AlphaMode,
    /// Cutoff used when `alpha_mode` is `Mask`.
    pub alpha_cutoff: f32,
    /// Material extensions present that this engine cannot represent.
    pub unsupported: Vec<String>,
}

/// Material extensions the loaders/shaders understand — anything else on a
/// material lands in [`Material::unsupported`].
pub(super) const SUPPORTED_MATERIAL_EXTENSIONS: &[&str] = &["KHR_materials_emissive_strength"];

impl Material {
    /// Why this material cannot be faithfully rendered/captured, or `None`.
    /// `Blend` alpha and unrepresentable extensions both count.
    pub fn unsupported_reason(&self) -> Option<String> {
        let mut reasons = self.unsupported.clone();
        if self.alpha_mode == AlphaMode::Blend {
            reasons.push("alpha_mode=Blend".to_string());
        }
        if reasons.is_empty() {
            None
        } else {
            Some(reasons.join(", "))
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            double_sided: false,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive_texture: None,
            emissive_factor: [0.0; 3],
            emissive_strength: 1.0,
            base_color_map: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            unsupported: Vec::new(),
        }
    }
}
