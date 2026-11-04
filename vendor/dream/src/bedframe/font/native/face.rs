//! A CoreText face: one retained `CTFont`, validated at load, sized per
//! layout. Immutable handles — `Send`/`Sync` per the documented
//! immutable-CTFont thread-safety contract.

use super::ffi::*;

static NEXT_FACE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

// --- NativeFace -----------------------------------------------------------

/// A CoreText face, at unit size for metrics calls — sized copies are
/// produced per layout (`CTFontCreateFontCopyWithSize` is cheap; the
/// layout cache makes it rare).
pub(crate) struct NativeFace {
    /// Largest bitmap we'll allocate for one glyph (px side). Atlas
    /// pages are 1024 by default; a glyph bigger than this is hidden,
    /// never allocated.
    pub(crate) max_bitmap: usize,
    font: CFRef,
    /// PostScript name (identity for diagnostics and glyph-cache keys).
    name: String,
    /// Unique per-process face identity (glyph-cache key component).
    id: u64,
    /// `kCTFontTraitColorGlyphs` (AppleColorEmoji) — RGBA page material.
    color: bool,
}

// Immutable retained handle; CTFont is documented thread-safe.
unsafe impl Send for NativeFace {}
unsafe impl Sync for NativeFace {}

impl NativeFace {
    /// From TTF/OTF bytes: CFData -> provider -> CGFont -> CTFont.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        unsafe {
            let cf_data = CFDataCreate(std::ptr::null(), data.as_ptr(), data.len() as isize);
            if cf_data.is_null() {
                return Err("native: CFDataCreate failed".to_string());
            }
            let cf_data = CFRef(cf_data);
            let provider = CGDataProviderCreateWithCFData(cf_data.raw());
            if provider.is_null() {
                return Err("native: CGDataProviderCreateWithCFData failed".to_string());
            }
            let provider = CFRef(provider);
            let cg_font = CGFontCreateWithDataProvider(provider.raw());
            if cg_font.is_null() {
                return Err("native: CGFontCreateWithDataProvider failed".to_string());
            }
            let cg_font = CFRef(cg_font);
            let ct_font = CTFontCreateWithGraphicsFont(
                cg_font.raw(),
                1.0,
                std::ptr::null(),
                std::ptr::null(),
            );
            if ct_font.is_null() {
                return Err("native: CTFontCreateWithGraphicsFont failed".to_string());
            }
            Ok(Self::finish(ct_font))
        }
    }

    /// From a PostScript name (system faces; exact resolution only).
    pub fn from_postscript_name(name: &str) -> Result<Self, String> {
        let cf_name = cf_string(name).ok_or("native: CFString create failed")?;
        let font = unsafe { CTFontCreateWithName(cf_name.raw(), 1.0, std::ptr::null()) };
        if font.is_null() {
            return Err(format!("native: no face named {name}"));
        }
        Ok(Self::finish(font))
    }

    fn finish(font: CTFontRef) -> Self {
        let name = unsafe {
            let ps = CTFontCopyPostScriptName(font);
            if ps.is_null() {
                String::new()
            } else {
                let s = cf_to_string(ps).unwrap_or_default();
                CFRelease(ps);
                s
            }
        };
        let color = unsafe { CTFontGetSymbolicTraits(font) & K_CT_FONT_TRAIT_COLOR_GLYPHS != 0 };
        Self {
            font: CFRef(font),
            name,
            id: NEXT_FACE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            max_bitmap: 2048,
            color,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_color(&self) -> bool {
        self.color
    }

    /// Stable identity for glyph-cache keys (unique per loaded face).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// A sized copy of this face for a layout pass. `px` must be finite
    /// and positive — anything else, or a failed copy, is `None` (the
    /// caller hides the cluster rather than guess a size).
    pub(crate) fn sized(&self, px: f32) -> Option<CFRef> {
        if !px.is_finite() || px <= 0.0 || px > 4096.0 {
            return None;
        }
        let copy = unsafe {
            CTFontCreateCopyWithAttributes(
                self.font.raw(),
                px as f64,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        (!copy.is_null()).then(|| CFRef(copy))
    }

    pub fn ascent(&self, px: f32) -> f32 {
        unsafe { CTFontGetAscent(self.font.raw()) as f32 * px }
    }

    pub fn descent(&self, px: f32) -> f32 {
        unsafe { CTFontGetDescent(self.font.raw()) as f32 * px }
    }

    /// Baseline-to-baseline (ascent + descent + leading), scaled.
    pub fn line_height(&self, px: f32) -> f32 {
        unsafe {
            let raw = CTFontGetAscent(self.font.raw())
                + CTFontGetDescent(self.font.raw())
                + CTFontGetLeading(self.font.raw());
            (raw as f32 * px).max(px * 1.0)
        }
    }

    /// Glyph id for a Unicode scalar (u16 pairs handled by shaping; this
    /// is the single-scalar probe used by `has_glyph`).
    pub fn glyph_id(&self, ch: char) -> Option<u16> {
        let mut utf16 = [0u16; 2];
        let mut glyphs = [0u16; 2];
        let count = ch.encode_utf16(&mut utf16).len();
        let ok = unsafe {
            CTFontGetGlyphsForCharacters(
                self.font.raw(),
                utf16.as_ptr(),
                glyphs.as_mut_ptr(),
                count as isize,
            )
        };
        if ok != 0 && glyphs[0] != 0 {
            Some(glyphs[0])
        } else {
            None
        }
    }

    /// Horizontal advance of `glyph` at `px` (0 on any failure).
    pub fn advance(&self, px: f32, glyph: u16) -> f32 {
        let Some(sized) = self.sized(px) else {
            return 0.0;
        };
        let mut size = CGSize::default();
        let glyphs = [glyph];
        unsafe {
            CTFontGetAdvancesForGlyphs(
                sized.raw(),
                K_CT_FONT_ORIENTATION_HORIZONTAL,
                glyphs.as_ptr(),
                &mut size,
                1,
            );
        }
        size.width as f32
    }

    /// Shape `text` (one grapheme cluster or a run) into glyph runs with
    /// this face: `(glyph_ids, positions, string_indices, run_font_ok)`.
    /// `run_font_ok` is false when CoreText substituted another face —
    /// the caller hides the cluster, it never draws fallback boxes.
    pub fn shape(&self, text: &str, px: f32) -> ShapedRun {
        let (Some(cf_text), Some(sized)) = (cf_text(text), self.sized(px)) else {
            return ShapedRun::default();
        };
        // NULL dictionary callbacks are deliberate: the entries are
        // non-retaining, which is safe because `sized` (a retained CFRef)
        // outlives the dict, the attr string AND the CTLine built below.
        let dict = unsafe {
            let keys = [kCTFontAttributeName];
            let values = [sized.raw()];
            CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if dict.is_null() {
            return ShapedRun::default();
        }
        let dict = CFRef(dict);
        let attr = unsafe { CFAttributedStringCreate(std::ptr::null(), cf_text.raw(), dict.raw()) };
        if attr.is_null() {
            return ShapedRun::default();
        }
        let attr = CFRef(attr);
        let line = unsafe { CTLineCreateWithAttributedString(attr.raw()) };
        if line.is_null() {
            return ShapedRun::default();
        }
        let line = CFRef(line);
        let runs = unsafe { CTLineGetGlyphRuns(line.raw()) };
        let run_count = unsafe { CFArrayGetCount(runs) } as usize;
        let mut out = ShapedRun::default();
        for i in 0..run_count {
            let run = unsafe { CFArrayGetValueAtIndex(runs, i as isize) };
            let count = unsafe { CTRunGetGlyphCount(run) } as usize;
            if count == 0 {
                continue;
            }
            let mut glyphs = vec![0u16; count];
            let mut positions = vec![CGPoint::default(); count];
            let mut indices = vec![0isize; count];
            let range = CFRange {
                location: 0,
                length: 0,
            };
            unsafe {
                CTRunGetGlyphs(run, range, glyphs.as_mut_ptr());
                CTRunGetPositions(run, range, positions.as_mut_ptr());
                CTRunGetStringIndices(run, range, indices.as_mut_ptr());
            }
            let attrs = unsafe { CTRunGetAttributes(run) };
            let run_font = if attrs.is_null() {
                std::ptr::null()
            } else {
                unsafe { CFDictionaryGetValue(attrs, kCTFontAttributeName) }
            };
            // A null attr font = our own (we set the run's font on the
            // whole string). Any other face is a substitution — rejected.
            let same_font = run_font.is_null() || run_font == sized.raw();
            out.glyphs.extend_from_slice(&glyphs);
            out.positions
                .extend(positions.iter().map(|p| (p.x as f32, p.y as f32)));
            out.indices.extend(indices.iter().map(|&i| i as usize));
            if !same_font {
                out.substituted = true;
            }
        }
        let mut ascent = 0.0f64;
        let mut descent = 0.0f64;
        let mut leading = 0.0f64;
        let width = unsafe {
            CTLineGetTypographicBounds(line.raw(), &mut ascent, &mut descent, &mut leading)
        } as f32;
        out.advance = width;
        out.ascent = ascent as f32;
        out.descent = descent as f32;
        out
    }
}

/// Output of [`NativeFace::shape`].
#[derive(Default)]
pub(crate) struct ShapedRun {
    pub glyphs: Vec<u16>,
    /// Positions in the line's space (pen start = 0).
    pub positions: Vec<(f32, f32)>,
    /// UTF-16 string index each glyph came from (cluster mapping).
    pub indices: Vec<usize>,
    /// CoreText substituted another face — the caller hides the cluster.
    pub substituted: bool,
    pub advance: f32,
    pub ascent: f32,
    pub descent: f32,
}

/// A rasterized glyph bitmap (coverage or RGBA) and its pen offsets.
#[derive(Default)]
pub(crate) struct Coverage {
    pub width: u32,
    pub height: u32,
    /// Left edge of the bitmap relative to the pen.
    pub ox: f32,
    /// Top edge of the bitmap *above* the baseline (y up).
    pub oy: f32,
    /// Preflight rejected the glyph (bounds too large) — warn, don't
    /// silently blank.
    pub too_large: bool,
    pub pixels: Vec<u8>,
}
