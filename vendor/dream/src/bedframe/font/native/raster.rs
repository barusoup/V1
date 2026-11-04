//! Native glyph rasterization: `CTFontDrawGlyphs` into a bounded CG
//! bitmap — coverage for ordinary faces, straight-alpha sRGB for colour.
//! Row 0 of a CG bitmap is the image's TOP row already, so no flip.

use super::face::{Coverage, NativeFace};
use super::ffi::*;
use std::ffi::c_void;

impl NativeFace {
    /// Rasterize `glyph` at `px` into an 8-bit coverage bitmap:
    /// `(width, height, offset_from_pen_x, offset_above_baseline_y, alpha)`.
    /// The bitmap's top-left in output coordinates is
    /// `(pen + ox, baseline - oy)`. Rasterized through the RGBA context
    /// path (which CoreText draws reliably) — coverage is its alpha.
    pub fn rasterize_coverage(&self, glyph: u16, px: f32) -> Coverage {
        self.rasterize_coverage_bounded(glyph, px, self.max_bitmap)
    }

    /// [`rasterize_coverage`] bounded to `max_side` per dimension.
    pub fn rasterize_coverage_bounded(&self, glyph: u16, px: f32, max_side: usize) -> Coverage {
        let mut rgba = if max_side == self.max_bitmap {
            self.rasterize_rgba(glyph, px)
        } else {
            self.rasterize_rgba_inner(glyph, px, max_side)
        };
        let pixels: Vec<u8> = rgba.pixels.chunks_exact(4).map(|px| px[3]).collect();
        rgba.pixels = pixels;
        rgba
    }

    /// Rasterize a colour glyph at `px` into straight-alpha RGBA bounded
    /// to `max_side` per dimension — the atlas page bound, so nothing
    /// oversized allocates before the preflight can reject it.
    pub fn rasterize_rgba_bounded(&self, glyph: u16, px: f32, max_side: usize) -> Coverage {
        self.rasterize_rgba_inner(glyph, px, max_side)
    }

    /// Rasterize a colour glyph at `px` into straight-alpha RGBA. `ox/oy`
    /// are the same pen-relative bitmap offsets as coverage.
    pub fn rasterize_rgba(&self, glyph: u16, px: f32) -> Coverage {
        self.rasterize_rgba_inner(glyph, px, self.max_bitmap)
    }

    fn rasterize_rgba_inner(&self, glyph: u16, px: f32, max_side: usize) -> Coverage {
        let Some(sized) = self.sized(px) else {
            return Coverage::default();
        };
        let glyphs = [glyph];
        let mut rect = CGRect::default();
        unsafe {
            CTFontGetBoundingRectsForGlyphs(
                sized.raw(),
                K_CT_FONT_ORIENTATION_HORIZONTAL,
                glyphs.as_ptr(),
                &mut rect,
                1,
            );
        }
        // Preflight BEFORE allocating: bounds must be finite and the
        // padded bitmap must fit inside a page (the atlas rejects
        // anything larger anyway — don't build a giant Vec first).
        if !rect.size.width.is_finite()
            || !rect.size.height.is_finite()
            || rect.size.width <= 0.0
            || rect.size.height <= 0.0
        {
            return Coverage::default();
        }
        let ox = rect.origin.x.floor() - 1.0;
        let oy = (rect.origin.y + rect.size.height).ceil() + 1.0;
        let w = ((rect.size.width + 4.0).ceil()).max(1.0) as usize;
        let h = ((rect.size.height + 4.0).ceil()).max(1.0) as usize;
        if w > max_side || h > max_side || w.checked_mul(h).is_none_or(|p| p > max_side * max_side)
        {
            return Coverage {
                too_large: true,
                ..Coverage::default()
            };
        }
        let mut rgba = vec![0u8; w * h * 4];
        // Explicit sRGB — DeviceRGB is not guaranteed sRGB.
        let color_space = unsafe { CGColorSpaceCreateWithName(kCGColorSpaceSRGB) };
        if color_space.is_null() {
            return Coverage::default();
        }
        let color_space = CFRef(color_space);
        let ctx = unsafe {
            CGBitmapContextCreate(
                rgba.as_mut_ptr() as *mut c_void,
                w,
                h,
                8,
                w * 4,
                color_space.raw(),
                K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST | K_CG_BITMAP_BYTE_ORDER_32_BIG,
            )
        };
        if ctx.is_null() {
            return Coverage::default();
        }
        let ctx = CFRef(ctx);
        let position = CGPoint {
            x: -ox,
            y: h as f64 - oy,
        };
        unsafe {
            CTFontDrawGlyphs(sized.raw(), glyphs.as_ptr(), &position, 1, ctx.raw());
        }
        // Row 0 is the image's top already; just un-premultiply.
        for px in rgba.chunks_exact_mut(4) {
            let a = px[3];
            if a == 0 {
                px[0] = 0;
                px[1] = 0;
                px[2] = 0;
            } else {
                px[0] = ((u32::from(px[0]) * 255 / u32::from(a)).min(255)) as u8;
                px[1] = ((u32::from(px[1]) * 255 / u32::from(a)).min(255)) as u8;
                px[2] = ((u32::from(px[2]) * 255 / u32::from(a)).min(255)) as u8;
            }
        }
        Coverage {
            width: w as u32,
            height: h as u32,
            ox: ox as f32,
            oy: oy as f32,
            too_large: false,
            pixels: rgba,
        }
    }
}
