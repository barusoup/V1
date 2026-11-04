//! Still images: PNG/JPEG in process, everything else through FFmpeg's raw
//! RGBA pipe.

use crate::error::{Error, Result};
use crate::geom::fit_side;
use crate::process;
use crate::tools::{TOOL_DEADLINE, Tools, input_options, is_networked};
use image::metadata::Orientation;
use image::{DynamicImage, GenericImageView, ImageDecoder, ImageFormat, ImageReader};
use std::path::Path;

/// Largest side a decoder accepts before it is refused outright.
const MAX_SIDE: u32 = 32_768;
/// Largest decoded pixel count accepted from the in-process decoder.
const MAX_PIXELS: u64 = 150_000_000;
/// Largest allocation the in-process decoder may reserve.
const MAX_ALLOC: u64 = 512 * 1024 * 1024;
/// Above this pixel count a JPEG is decoded by FFmpeg instead.
///
/// Measured, each image scaled to 1920 wide, best of five, process spawn
/// included: 720p JPEG 1.7 ms in process against ~20 ms through the pipe;
/// 2.5K JPEG 55 ms against 54 ms, a dead heat; 4K JPEG 77 ms against 58 ms.
/// PNG measures the other way at every size (4K: 120 ms in process, 173 ms
/// through the pipe), so it never leaves the process.
const PIPE_ABOVE_PIXELS: u64 = 3_000_000;

/// A decoded still image as tightly packed, top-row-first RGBA8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl Image {
    /// Decode `path`, scaling the longest side down to `max_side`. Zero means
    /// no scaling. PNG and JPEG use the in-process decoder; other formats go
    /// through FFmpeg. `ffmpeg`/`ffprobe` are only required for the fallback.
    pub fn open(path: impl AsRef<Path>, max_side: u32) -> Result<Self> {
        let path = path.as_ref();
        if let Some(image) = decode_in_process(path, max_side)? {
            return Ok(image);
        }
        Self::open_with(&crate::tools::Tools::discover()?, path, max_side)
    }

    /// Decode with explicit FFmpeg tool paths, for applications that bundle
    /// their own build.
    pub fn open_with(
        tools: &crate::tools::Tools,
        path: impl AsRef<Path>,
        max_side: u32,
    ) -> Result<Self> {
        let path = path.as_ref();
        if let Some(image) = decode_in_process(path, max_side)? {
            return Ok(image);
        }
        decode_ffmpeg(tools, path, max_side)
    }

    /// Build from pixels the caller already decoded or generated.
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::msg("image needs a non-zero width and height"));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| Error::msg("image dimensions overflow addressable memory"))?;
        if rgba.len() != expected {
            return Err(Error::msg(format!(
                "image is {} bytes for {width}x{height} RGBA8 ({expected} expected)",
                rgba.len()
            )));
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Decoded width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Decoded height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `(width, height)` in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Tightly packed, top-row-first RGBA8 pixels.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Upload as a full-mip sRGB texture. This is a one-shot upload: still
    /// images rarely change, so it uses Dream's normal synchronous path.
    #[cfg(feature = "gpu")]
    pub fn upload(
        &self,
        ctx: &std::sync::Arc<dream::gfx::vulkan::GpuContext>,
    ) -> std::sync::Arc<dream::texture::GpuTexture> {
        std::sync::Arc::new(dream::texture::GpuTexture::from_rgba8(
            ctx,
            "media image",
            &self.rgba,
            self.width,
            self.height,
            dream::texture::TextureOptions::default(),
        ))
    }
}

fn decode_in_process(path: &Path, max_side: u32) -> Result<Option<Image>> {
    let mut reader = ImageReader::open(path)?.with_guessed_format()?;
    let format = reader.format();
    if !matches!(format, Some(ImageFormat::Png | ImageFormat::Jpeg)) {
        return Ok(None);
    }
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(MAX_ALLOC);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| Error::msg(format!("decode {}: {error}", path.display())))?;
    let (width, height) = decoder.dimensions();
    if u64::from(width).saturating_mul(u64::from(height)) > MAX_PIXELS {
        return Err(Error::msg(format!(
            "{} is {width}x{height}, larger than the {} megapixel limit",
            path.display(),
            MAX_PIXELS / 1_000_000
        )));
    }
    // The header already gave us the size, so a big JPEG can be routed to the
    // decoder that handles it faster before any pixels are touched.
    if format == Some(ImageFormat::Jpeg)
        && u64::from(width).saturating_mul(u64::from(height)) > PIPE_ABOVE_PIXELS
    {
        return Ok(None);
    }
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = DynamicImage::from_decoder(decoder)
        .map_err(|error| Error::msg(format!("decode {}: {error}", path.display())))?;
    image.apply_orientation(orientation);
    let (width, height) = image.dimensions();
    let target = fit_side(width, height, max_side);
    if target != (width, height) {
        // Integer box downsampling: far cheaper than a windowed filter and
        // visually fine at the sizes a Dream app displays.
        image = image.thumbnail_exact(target.0, target.1);
    }
    // `into_` not `to_`: an RGBA source hands over its buffer instead of
    // copying the whole image a second time.
    let rgba = image.into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(Some(Image {
        width,
        height,
        rgba: rgba.into_raw(),
    }))
}

/// Cap for a still decoded without a size request. 8K RGBA is 256 MiB.
const MAX_STILL_BYTES: usize = 256 * 1024 * 1024;

fn decode_ffmpeg(tools: &Tools, path: &Path, max_side: u32) -> Result<Image> {
    let filter = if max_side == 0 {
        "format=rgba,setsar=1".to_string()
    } else {
        // The PAM header tells us the final size, so no separate ffprobe pass
        // is needed. `decrease` never enlarges; `setsar` squares the pixels.
        format!(
            "scale=w='min(iw,{max_side})':h='min(ih,{max_side})':\
             force_original_aspect_ratio=decrease,setsar=1,format=rgba"
        )
    };
    let cap = if max_side == 0 {
        MAX_STILL_BYTES
    } else {
        let side = u64::from(max_side);
        (side * side * 4) as usize + 128 * 1024
    };

    let first = run_still(tools, path, &["-vf", &filter], cap)?;
    if first.status.success() {
        return finish_pam(path, first.stdout, max_side);
    }
    // A tiled HEIF/AVIF is a stream group that FFmpeg stitches in a filter
    // graph of its own; `-vf` cannot attach to it. Scale the stitched group
    // inside the same graph instead.
    let complex = format!("[0:g:0]{filter}[out]");
    let retry = run_still(
        tools,
        path,
        &["-filter_complex", &complex, "-map", "[out]"],
        cap,
    )?;
    if retry.status.success() {
        return finish_pam(path, retry.stdout, max_side);
    }
    Err(process::failure("ffmpeg image decode", path, &first))
}

fn run_still(
    tools: &crate::tools::Tools,
    path: &Path,
    filter: &[&str],
    cap: usize,
) -> Result<process::Run> {
    let mut cmd = tools.ffmpeg_command();
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    input_options(&mut cmd, "4", "4", is_networked(path));
    cmd.arg("-i").arg(path).args(filter).args([
        // Lanczos here is only for chroma and colour conversion; at these
        // sizes its cost is inside the decoder's noise.
        "-sws_flags",
        "lanczos+accurate_rnd+full_chroma_int",
        "-threads",
        "1",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-c:v",
        "pam",
        "-pix_fmt",
        "rgba",
        "pipe:1",
    ]);
    Ok(process::run(cmd, TOOL_DEADLINE, cap)?)
}

/// Parse the PAM header, verify the payload and hand the pixels to [`Image`].
/// PAM is a tiny uncompressed container: an ASCII header, then rows.
fn finish_pam(path: &Path, mut stdout: Vec<u8>, max_side: u32) -> Result<Image> {
    let (width, height, offset) = parse_pam(path, &stdout)?;
    if max_side > 0 && (width > max_side || height > max_side) {
        return Err(Error::msg(format!(
            "{} decoded to {width}x{height}, over the {max_side} px bound",
            path.display()
        )));
    }
    let bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| Error::msg("image dimensions overflow addressable memory"))?;
    if stdout.len() != offset + bytes {
        return Err(Error::msg(format!(
            "{} decoded to {} bytes; expected {bytes} after the PAM header",
            path.display(),
            stdout.len() - offset
        )));
    }
    stdout.drain(..offset);
    Image::from_rgba(width, height, stdout)
}

fn parse_pam(path: &Path, bytes: &[u8]) -> Result<(u32, u32, usize)> {
    if !bytes.starts_with(b"P7\n") {
        return Err(Error::msg(format!(
            "{} did not decode to PAM data",
            path.display()
        )));
    }
    let end = bytes
        .windows(7)
        .position(|window| window == b"ENDHDR\n")
        .ok_or_else(|| Error::msg(format!("{} has no PAM header end", path.display())))?;
    let header = std::str::from_utf8(&bytes[..end])
        .map_err(|_| Error::msg(format!("{} has a non-UTF8 PAM header", path.display())))?;
    let (mut width, mut height, mut depth, mut maxval) = (None, None, None, None);
    for line in header.lines().skip(1) {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        let value = parts.next().and_then(|value| value.parse::<u32>().ok());
        match key {
            "WIDTH" => width = value,
            "HEIGHT" => height = value,
            "DEPTH" => depth = value,
            "MAXVAL" => maxval = value,
            _ => {}
        }
    }
    let (Some(width), Some(height)) = (width, height) else {
        return Err(Error::msg(format!(
            "{} has a PAM header without a size",
            path.display()
        )));
    };
    if width == 0 || height == 0 || depth != Some(4) || maxval != Some(255) {
        return Err(Error::msg(format!(
            "{} decoded to an unsupported PAM shape ({width}x{height}, depth {depth:?})",
            path.display()
        )));
    }
    Ok((width, height, end + 7))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Temp(PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn temp(name: &str) -> Temp {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        Temp(std::env::temp_dir().join(format!(
            "dream-media-image-{}-{id}-{name}",
            std::process::id()
        )))
    }

    fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
        let file = std::fs::File::create(path).unwrap();
        image::codecs::png::PngEncoder::new(file)
            .write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
            .unwrap();
    }

    #[test]
    fn from_rgba_checks_the_buffer() {
        assert!(Image::from_rgba(2, 2, vec![0; 16]).is_ok());
        assert!(Image::from_rgba(0, 2, vec![]).is_err());
        assert!(Image::from_rgba(2, 2, vec![0; 15]).is_err());
    }

    #[test]
    fn png_takes_the_in_process_path_with_alpha() {
        let file = temp("alpha.png");
        let pixels: Vec<u8> = (0..16).map(|i| i as u8 * 16).collect();
        write_png(&file.0, 2, 2, &pixels);
        let image = Image::open(&file.0, 0).unwrap();
        assert_eq!(image.size(), (2, 2));
        assert_eq!(image.rgba(), pixels);
    }

    #[test]
    fn max_side_downscales_without_upscaling() {
        let file = temp("wide.png");
        let pixels = vec![200u8; 16 * 8 * 4];
        write_png(&file.0, 16, 8, &pixels);
        let image = Image::open(&file.0, 8).unwrap();
        assert_eq!(image.size(), (8, 4));
        let same = Image::open(&file.0, 100).unwrap();
        assert_eq!(same.size(), (16, 8));
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let path = std::env::temp_dir().join("dream-media-definitely-missing.png");
        assert!(Image::open(&path, 0).is_err());
    }

    #[test]
    fn jpeg_takes_the_in_process_path() {
        let file = temp("photo.jpg");
        let pixels = vec![120u8; 8 * 8 * 3];
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
            std::fs::File::create(&file.0).unwrap(),
            90,
        );
        image::ImageEncoder::write_image(encoder, &pixels, 8, 8, image::ExtendedColorType::Rgb8)
            .unwrap();
        let image = Image::open(&file.0, 0).unwrap();
        assert_eq!(image.size(), (8, 8));
        assert_eq!(image.rgba().len(), 8 * 8 * 4);
        assert!(image.rgba().chunks_exact(4).all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn an_animated_gif_returns_its_first_frame_through_ffmpeg() {
        let Ok(tools) = crate::tools::Tools::discover() else {
            return;
        };
        let file = temp("animated.gif");
        let status = std::process::Command::new(&tools.ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"])
            .args(["-i", "testsrc=size=64x48:rate=10", "-t", "0.3"])
            .arg(&file.0)
            .status();
        if !status.is_ok_and(|status| status.success()) {
            return;
        }
        let image = Image::open_with(&tools, &file.0, 32).unwrap();
        assert_eq!(image.size(), (32, 24));
        assert_eq!(image.rgba().len(), 32 * 24 * 4);
    }

    #[test]
    fn a_non_png_format_falls_back_to_ffmpeg() {
        let Ok(tools) = crate::tools::Tools::discover() else {
            return;
        };
        let file = temp("still.bmp");
        let status = std::process::Command::new(&tools.ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"])
            .args(["-i", "color=c=red:size=64x48", "-frames:v", "1"])
            .arg(&file.0)
            .status();
        if !status.is_ok_and(|status| status.success()) {
            return;
        }
        let image = Image::open_with(&tools, &file.0, 32).unwrap();
        assert_eq!(image.size(), (32, 24));
        assert_eq!(image.rgba().len(), 32 * 24 * 4);
        assert!(image.rgba()[0] > 200);
    }
}

