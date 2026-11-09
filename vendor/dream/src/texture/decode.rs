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
    let rgba = image.into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(DecodedTexture {
        pixels: rgba.into_raw(),
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::decode_rgba8;

    fn encode_png(image: image::DynamicImage) -> Vec<u8> {
        let mut buf = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("encode png");
        buf
    }

    #[test]
    fn texture_decode_rgba8_expands_rgb_with_opaque_alpha() {
        let source = image::RgbImage::from_fn(3, 2, |x, y| {
            image::Rgb([10 + x as u8 * 40, 20 + y as u8 * 90, 200 - x as u8 * 7])
        });
        let bytes = encode_png(image::DynamicImage::ImageRgb8(source.clone()));

        let decoded = decode_rgba8(&bytes).expect("decode rgb png");

        assert_eq!((decoded.width, decoded.height), (3, 2));
        assert_eq!(decoded.pixels.len(), 3 * 2 * 4);
        for (i, px) in decoded.pixels.chunks_exact(4).enumerate() {
            let (x, y) = ((i % 3) as u32, (i / 3) as u32);
            let expected = source.get_pixel(x, y).0;
            assert_eq!(&px[..3], &expected, "rgb at ({x}, {y})");
            assert_eq!(px[3], 255, "alpha at ({x}, {y})");
        }
    }

    #[test]
    fn texture_decode_rgba8_preserves_rgba_pixels() {
        let source = image::RgbaImage::from_fn(3, 2, |x, y| {
            image::Rgba([
                5 + x as u8 * 50,
                100 + y as u8 * 30,
                220 - x as u8 * 11,
                40 + x as u8 * 60 + y as u8 * 15,
            ])
        });
        let bytes = encode_png(image::DynamicImage::ImageRgba8(source.clone()));

        let decoded = decode_rgba8(&bytes).expect("decode rgba png");

        assert_eq!((decoded.width, decoded.height), (3, 2));
        assert_eq!(decoded.pixels.len(), 3 * 2 * 4);
        assert_eq!(decoded.pixels, source.into_raw());
    }
}
