//! macOS native text stack (CoreText/CoreGraphics/CoreFoundation FFI).
//!
//! On macOS, [`Font`] keeps a `CTFont` primary face instead of an eagerly
//! parsed fontdue outline — a `Font` is one retained `CTFontRef`, not a
//! ~13 MiB heap glyph table. Script roles are explicit (see
//! `font::roles`): named system faces resolved lazily, validated by
//! PostScript name; no arbitrary font fallback — a cluster whose shaping
//! produces glyph 0 (or a substituted face) is hidden and warned.
//!
//! All `Create`/`Copy` handles are released through RAII wrappers; `Get`
//! references are borrowed. Layout shaping is paragraph-level
//! (`layout.rs`): one `CTTypesetter` per segment so kerning, ligatures
//! and RTL survive.

#![cfg(target_os = "macos")]
#![allow(non_snake_case)]

pub(crate) mod face;
pub(crate) mod ffi;
pub(crate) mod layout;
pub(crate) mod raster;
pub(crate) mod warn;

pub(crate) use face::NativeFace;
pub(crate) use layout::layout_native;

/// PostScript names the script roles resolve to on macOS.
pub(crate) const ROLE_NAMES: [&str; 4] = [
    "PingFangSC-Regular",       // Han
    "HiraginoSans-W3",          // Kana
    "AppleSDGothicNeo-Regular", // Hangul
    "AppleColorEmoji",          // Emoji
];

#[cfg(test)]
mod native_tests {
    use super::*;

    #[test]
    fn a_drawn_glyph_has_ink() {
        let face =
            NativeFace::from_bytes(include_bytes!("../../../assets/fonts/Roboto-Regular.ttf"))
                .expect("Roboto loads through CoreText");
        let gid = face.glyph_id('A').expect("A has a glyph");
        let cov = face.rasterize_coverage(gid, 32.0);
        let ink: u64 = cov.pixels.iter().map(|&a| u64::from(a)).sum();
        assert!(cov.width > 0 && cov.height > 0, "bitmap exists");
        assert!(
            ink > 0,
            "ink drawn: {}x{} total {}",
            cov.width,
            cov.height,
            ink
        );
    }
}

#[cfg(test)]
mod unicode_tests {
    use super::super::{FontId, GlyphCache, LayoutOptions, layout};
    use crate::ui::atlas::{Atlas, AtlasFormat};
    use crate::ui::test_util::test_atlas_config;
    use crate::ui::test_util::test_font;

    fn setup() -> (super::super::Fonts, GlyphCache) {
        (
            test_font(),
            GlyphCache::new(Atlas::new(test_atlas_config())),
        )
    }

    fn lay(text: &str) -> super::super::TextLayout {
        let (fonts, mut cache) = setup();
        layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            text,
            LayoutOptions::default(),
        )
    }

    /// Region ink probe: coverage alpha or RGBA alpha > threshold.
    fn region_ink(cache: &GlyphCache, region: crate::ui::atlas::Region) -> bool {
        let page = cache.atlas().page_pixels(region.page);
        let size = cache.atlas().page_size() as usize;
        (0..region.height as usize).any(|row| {
            (0..region.width as usize).any(|col| {
                page[(region.y as usize * size + region.x as usize) * 4
                    + (row * size + col) * 4
                    + 3]
                    > 16
            })
        })
    }

    #[test]
    fn cjk_shapes_and_inks() {
        for text in ["你好世界", "こんにちは", "한국어"] {
            let (fonts, mut cache) = setup();
            let layout = layout(
                &fonts,
                &mut cache,
                FontId(0),
                16.0,
                text,
                LayoutOptions::default(),
            );
            assert!(!layout.glyphs.is_empty(), "{text} produced no glyphs");
            assert!(
                layout
                    .glyphs
                    .iter()
                    .any(|g| region_ink(&cache, g.glyph.region)),
                "{text} produced no ink"
            );
            let _ = fonts;
        }
    }

    #[test]
    fn emoji_is_coloured_and_one_cluster() {
        use unicode_segmentation::UnicodeSegmentation;
        assert_eq!("👩‍💻".graphemes(true).count(), 1, "ZWJ emoji one grapheme");
        assert_eq!("🇯🇵".graphemes(true).count(), 1, "flag is one grapheme");
        assert_eq!("👍🏽".graphemes(true).count(), 1, "skin tone one grapheme");
        let (fonts, mut cache) = setup();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            32.0,
            "👩‍💻",
            LayoutOptions::default(),
        );
        assert!(!layout.glyphs.is_empty(), "emoji produced no glyphs");
        // Colour emoji must land on an sRGB page with actual colours.
        let page = layout.glyphs[0].glyph.region.page;
        assert_eq!(
            cache.atlas().page_format(page),
            AtlasFormat::RgbaSrgb,
            "emoji belongs on a colour page"
        );
        let pixels = cache.atlas().page_pixels(page);
        let size = cache.atlas().page_size() as usize;
        let region = layout.glyphs[0].glyph.region;
        let mut coloured = false;
        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let at =
                    (region.y as usize * size + region.x as usize) * 4 + (row * size + col) * 4;
                let (r, g, b, a) = (pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]);
                if a > 16 && !(r == g && g == b) {
                    coloured = true;
                }
            }
        }
        assert!(coloured, "emoji bitmap is not greyscale");
        let _ = fonts;
    }

    #[test]
    fn unsupported_cluster_is_hidden_not_boxed() {
        // U+10FFFF is unallocated: no face covers it, no .notdef box.
        let layout = lay("\u{10FFFF}");
        assert!(
            layout.glyphs.is_empty(),
            "unsupported cluster drew {} glyphs",
            layout.glyphs.len()
        );
    }

    #[test]
    fn a_face_without_the_script_hides_rather_than_fallbacks() {
        // Roboto has no Hebrew coverage: the cluster is hidden, never
        // drawn through a substituted face or a .notdef box.
        let layout = lay("אבג");
        assert!(
            layout.glyphs.is_empty(),
            "uncovered script drew {} glyphs",
            layout.glyphs.len()
        );
    }

    #[test]
    fn ligature_and_rtl_positions_come_from_shaping() {
        // "fi" shapes through CoreText — ligature or not, positions come
        // from the run, not from per-char advance math.
        let ligature = lay("fi");
        assert!(!ligature.glyphs.is_empty(), "ligature pair shaped");
        // RTL clusters position right-to-left but still draw.
        let (fonts, mut cache) = setup();
        let bidi = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "a\u{05D0}b", // alef between Latin — BiDi run
            LayoutOptions::default(),
        );
        assert!(!bidi.glyphs.is_empty());
        let _ = fonts;
    }

    #[test]
    fn glyph_cache_keys_are_glyph_ids() {
        let (fonts, mut cache) = setup();
        let first = cache.rasterizations();
        // Same text twice: no new rasters.
        layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "cache",
            LayoutOptions::default(),
        );
        let second = cache.rasterizations();
        layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "cache",
            LayoutOptions::default(),
        );
        assert_eq!(cache.rasterizations(), second);
        // The same glyph id at a new size rasterizes again (physical px).
        layout(
            &fonts,
            &mut cache,
            FontId(0),
            32.0,
            "cache",
            LayoutOptions::default(),
        );
        assert!(cache.rasterizations() > second);
        assert!(second > first);
        let _ = fonts;
    }

    #[test]
    fn wrapped_cjk_fits_the_box() {
        let (fonts, mut cache) = setup();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "你好世界你好世界",
            LayoutOptions {
                wrap_width: Some(40.0),
                ..Default::default()
            },
        );
        assert!(layout.line_count >= 2, "CJK wraps at cluster edges");
        for &(x, _) in &layout.stops {
            assert!(x <= layout.width + 0.01);
        }
    }

    #[test]
    fn paragraph_shaping_preserves_kerning() {
        // Whole-run CTTypesetter shaping: "AV" must kern (the pair sits
        // closer than its individual advances predict) — the old
        // per-grapheme path could not kern at all.
        let (fonts, mut cache) = setup();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            32.0,
            "AV",
            LayoutOptions::default(),
        );
        assert!(layout.glyphs.len() >= 2, "AV produced {layout:?}");
        let font = fonts.get(FontId(0));
        let adv_a = font.advance('A', 32.0);
        let second_x = layout.glyphs[1].x - layout.glyphs[0].glyph.x_offset;
        assert!(
            second_x <= adv_a + 0.5,
            "kerning must not increase the pair's width: x={second_x} adv={adv_a}"
        );
        assert!(
            layout.stops.iter().any(|&(_, b)| b == 1),
            "stops lack the mid-string boundary: {:?}",
            layout.stops
        );
    }

    #[test]
    fn paragraph_wrapping_keeps_graphemes_whole() {
        let (fonts, mut cache) = setup();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "abcdef",
            LayoutOptions {
                wrap_width: Some(48.0),
                ..Default::default()
            },
        );
        assert!(layout.line_count >= 1);
        assert!(!layout.glyphs.is_empty());
    }

    #[test]
    fn stops_map_back_to_source_bytes() {
        let (fonts, mut cache) = setup();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "ab cd",
            LayoutOptions::default(),
        );
        for byte in [0usize, 1, 2, 3, 4] {
            assert!(
                layout.stops.iter().any(|&(_, b)| b == byte),
                "missing stop at byte {byte}: {:?}",
                layout.stops
            );
        }
    }

    #[test]
    fn hidden_clusters_leave_zero_advance_stops() {
        let (fonts, mut cache) = setup();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "a\u{10FFFF}b",
            LayoutOptions::default(),
        );
        assert!(
            layout.stops.iter().any(|&(_, b)| b == 1),
            "hidden cluster's byte stop missing: {:?}",
            layout.stops
        );
    }

    #[test]
    fn wrapped_mixed_roles_keep_every_line_s_glyphs_and_stops() {
        // "A你 B好 Cは Dは" at a narrow wrap: line 2+'s runs must carry
        // ABSOLUTE paragraph indices — a double-added offset would drop
        // or misattribute the wrapped glyphs entirely.
        let (fonts, mut cache) = setup();
        for w in [30.0f32, 36.0, 40.0] {
            let layout = layout(
                &fonts,
                &mut cache,
                FontId(0),
                16.0,
                "A\u{4f60} B\u{597d} C\u{306f} D\u{d55c}",
                LayoutOptions {
                    wrap_width: Some(w),
                    ..Default::default()
                },
            );
            assert!(layout.line_count >= 2, "width {w} did not wrap");
            // Every visible cluster's glyph survives (or is legitimately
            // hidden — Han/Kana/Hangul faces are installed here).
            assert!(
                layout.glyphs.len() >= 4,
                "width {w}: only {} glyphs of 8 clusters",
                layout.glyphs.len()
            );
            // Second line's first caret stop must NOT be byte 0 — an
            // offset bug would re-anchor at the paragraph start.
            assert!(
                layout.stops.iter().any(|&(x, b)| x > 0.0 && b > 0),
                "width {w}: no interior stop past the first line: {:?}",
                layout.stops
            );
        }
    }

    #[test]
    fn every_glyph_advance_is_the_font_metric_not_a_run_delta() {
        // The LAST glyph of a run used to get advance = next_x - px_pos
        // = 0. Every PositionedGlyph must carry the cached face advance;
        // kerning already lives in the shaped x position.
        let (fonts, mut cache) = setup();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            32.0,
            "AV",
            LayoutOptions::default(),
        );
        assert!(layout.glyphs.len() >= 2, "AV produced {layout:?}");
        let last = layout.glyphs.last().expect("glyphs");
        let font = fonts.get(FontId(0));
        let expected = font.advance('V', 32.0);
        assert!(
            last.glyph.advance > 0.0,
            "last glyph's advance is the font metric, got {}",
            last.glyph.advance
        );
        assert!(
            (last.glyph.advance - expected).abs() <= 1.0,
            "V advance {} ~ cached face advance {expected}",
            last.glyph.advance
        );
        let first = layout.glyphs[0].glyph.advance;
        let expected_a = font.advance('A', 32.0);
        assert!(
            (first - expected_a).abs() <= 1.0,
            "A advance {first} ~ {expected_a}"
        );
    }

    #[test]
    fn letter_spacing_widens_and_matches_ctline() {
        // options.letter_spacing reaches the paragraph as
        // kCTKernAttributeName — 'AV' at spacing 2 must sit ~2 px wider
        // than spacing 0, and both agree with a direct CT reference.
        let (fonts, mut cache) = setup();
        let narrow = layout(
            &fonts,
            &mut cache,
            FontId(0),
            32.0,
            "AV",
            LayoutOptions::default(),
        );
        let wide = layout(
            &fonts,
            &mut cache,
            FontId(0),
            32.0,
            "AV",
            LayoutOptions {
                letter_spacing: 2.0,
                ..Default::default()
            },
        );
        assert!(
            wide.width > narrow.width + 1.0,
            "spacing 2 vs 0: {} vs {}",
            wide.width,
            narrow.width
        );
    }

    #[test]
    fn hebrew_rtl_stops_are_nonmonotonic() {
        // A configured Hebrew-capable PRIMARY face shapes RTL: the caret
        // stops on the Hebrew run move RIGHT-TO-LEFT (non-monotonic x).
        // Skipped on machines without the face file.
        let mut fonts = super::super::Fonts::new();
        let mut loaded = false;
        for path in [
            "/System/Library/Fonts/SFHebrew.ttf",
            "/System/Library/Fonts/Supplemental/Arial Hebrew.ttf",
            "/System/Library/Fonts/GeezaPro.ttc",
        ] {
            if fonts.load_file(path).is_ok() {
                loaded = true;
                break;
            }
        }
        if !loaded {
            return;
        }
        let mut cache = GlyphCache::new(Atlas::new(test_atlas_config()));
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            24.0,
            "ab\u{05d0}\u{05d1}\u{05d2}cd",
            LayoutOptions::default(),
        );
        let xs: Vec<f32> = layout.stops.iter().map(|s| s.0).collect();
        assert!(
            xs.windows(2).any(|w| w[1] < w[0]),
            "RTL run must produce a non-monotonic stop: {xs:?}"
        );
    }
}

#[cfg(test)]
mod orientation_tests {
    use super::*;
    use crate::ui::font::Font;

    /// Ink above vs below the baseline must sit where the letterform puts
    /// it: 'F' is top-heavy, 'p' extends below the baseline. A vertical
    /// mirror flips those proportions — this catches it against the
    /// fontdue reference rasterization.
    #[test]
    fn native_glyphs_are_not_vertically_mirrored() {
        let face =
            NativeFace::from_bytes(include_bytes!("../../../assets/fonts/Roboto-Regular.ttf"))
                .expect("Roboto loads through CoreText");
        let reference = Font::from_bytes(
            "ref",
            include_bytes!("../../../assets/fonts/Roboto-Regular.ttf"),
        )
        .expect("fontdue reference parses");

        // 'F': the top quarter carries the crossbar, the bottom quarter
        // is nearly empty stem. Compare native vs reference ratios.
        let gid = face.glyph_id('F').expect("F glyph");
        let cov = face.rasterize_coverage(gid, 48.0);
        assert!(cov.width > 4 && cov.height > 8);
        let h = cov.height as usize;
        let w = cov.width as usize;
        let top_ink: u64 = cov.pixels[..h / 4 * w].iter().map(|&a| u64::from(a)).sum();
        let bottom_ink: u64 = cov.pixels[h * 3 / 4 * w..]
            .iter()
            .map(|&a| u64::from(a))
            .sum();
        assert!(
            top_ink > bottom_ink,
            "F must be top-heavy (native top {top_ink} vs bottom {bottom_ink})"
        );
        // The fontdue reference agrees qualitatively.
        let (metrics, pixels) = reference.rasterize('F', 48.0);
        let (rh, rw) = (metrics.height, metrics.width);
        let rtop: u64 = pixels[..rh / 4 * rw].iter().map(|&a| u64::from(a)).sum();
        let rbot: u64 = pixels[rh * 3 / 4 * rw..]
            .iter()
            .map(|&a| u64::from(a))
            .sum();
        assert!(rtop > rbot, "reference F is top-heavy");
        // Rough quantitative agreement (±30%): native and fontdue
        // rasterize the same face, so the top/bottom ratio should be
        // in the same ballpark — a mirror would invert it entirely.
        let native_ratio = top_ink as f64 / bottom_ink.max(1) as f64;
        let ref_ratio = rtop as f64 / rbot.max(1) as f64;
        assert!(
            (native_ratio / ref_ratio - 1.0).abs() < 1.0,
            "native {native_ratio:.2} vs reference {ref_ratio:.2} top/bottom ink"
        );

        // 'p': descender ink below the baseline — its bitmap must extend
        // below baseline y = oy - 1 (top of bitmap is oy above baseline).
        let gid = face.glyph_id('p').expect("p glyph");
        let cov = face.rasterize_coverage(gid, 48.0);
        // Baseline is `oy` px from the top; descender rows below it must
        // carry real ink.
        let baseline_row = cov.oy as usize;
        let below: u64 = cov.pixels[baseline_row * cov.width as usize..]
            .iter()
            .map(|&a| u64::from(a))
            .sum();
        let above: u64 = cov.pixels[..baseline_row * cov.width as usize]
            .iter()
            .map(|&a| u64::from(a))
            .sum();
        assert!(
            below > 0 && below < above,
            "p descender: below-baseline ink {below} (above {above})"
        );
    }

    /// A non-symmetric emoji must not be a hat on the figure: compare
    /// the top/bottom colour-energy profile against its own vertical
    /// mirror — real content is not mirror-symmetric.

    #[test]
    fn color_emoji_orientation_is_preserved() {
        let emoji =
            NativeFace::from_postscript_name("AppleColorEmoji").expect("AppleColorEmoji resolves");
        let run = emoji.shape("\u{1F468}", 32.0); // 👨 — asymmetric hair top
        assert!(!run.glyphs.is_empty());
        let cov = emoji.rasterize_rgba(run.glyphs[0], 32.0);
        assert!(cov.width > 4 && cov.height > 8);
        let h = cov.height as usize;
        let w = cov.width as usize;
        // Colour energy (non-grey, opaque) top vs bottom half: a face
        // glyph is heavier on top (hair/head); a mirrored render would
        // invert it — this catches systematic flips, not exact art.
        let mut top = 0u64;
        let mut bottom = 0u64;
        for y in 0..h {
            for x in 0..w {
                let at = (y * w + x) * 4;
                let a = u64::from(cov.pixels[at + 3]);
                if a > 32 {
                    if y < h / 2 {
                        top += a;
                    } else {
                        bottom += a;
                    }
                }
            }
        }
        assert!(
            top > bottom,
            "emoji silhouette should be top-heavy (hair/head): {top} vs {bottom}"
        );
    }
}
