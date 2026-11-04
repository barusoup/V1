#[cfg(doc)]
use super::TextureManager;

/// Mip levels for a full chain: floor(log2(max(w, h))) + 1.
pub fn mip_levels_for(width: u32, height: u32) -> u32 {
    let max = width.max(height).max(1);
    32 - max.leading_zeros()
}

/// CPU-side RGBA8 image with its dimensions.
#[derive(Clone, Debug)]
pub struct DecodedTexture {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Decode PNG or JPEG bytes to RGBA8. Purely CPU-side; see
/// [`TextureManager::from_pixels`] for the GPU upload.
pub fn decode_rgba8(bytes: &[u8]) -> Result<DecodedTexture, String> {
    let image = image::load_from_memory(bytes).map_err(|err| format!("decode image: {err}"))?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(DecodedTexture {
        pixels: rgba.into_raw(),
        width,
        height,
    })
}
