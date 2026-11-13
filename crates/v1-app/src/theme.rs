// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Design tokens: colours, metrics and the few colour helpers the whole UI shares.
//!
//! The palette is true black with tonal steps, never lines. Depth comes from three surfaces
//! (page, panel, raised) and from white washes at low alpha, so any surface can sit on any other
//! and still read.
//!
//! Colours are read through [`Palette`] rather than written as constants, so Settings can swap a
//! theme in at runtime. The default palette is the original black and grey one, value for value.

use std::collections::HashMap;
use std::sync::RwLock;

use dream::ui::Color;

/// How the audio visualizer behind the lyrics page is coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisualizerColor {
    /// Extracted dynamically from the currently playing song's album cover.
    #[default]
    AlbumArt,
    /// Uses the active theme's accent colour.
    Accent,
    /// Uses a fixed custom colour chosen by the user.
    Custom(Color),
}

/// Every colour the interface uses. `Default` is the black-and-grey theme the app shipped with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette {
    pub bg: Color,
    pub panel: Color,
    pub raised: Color,
    pub hover: Color,
    pub select: Color,
    pub press: Color,
    pub panel_hover: Color,
    pub panel_select: Color,
    pub panel_press: Color,
    pub raised_top: Color,
    pub track: Color,
    pub thumb: Color,
    pub thumb_hot: Color,
    pub fg: Color,
    pub muted: Color,
    pub faint: Color,
    /// The one accent: the play disc, the slider fill, a chosen theme.
    pub accent: Color,
    /// Ink on top of the accent.
    pub on_accent: Color,
    /// Lyrics text fill colour.
    pub lyrics_fg: Color,
    /// Lyrics text outline colour.
    pub lyrics_outline: Color,
    /// Audio visualizer colour mode on the lyrics page.
    pub visualizer: VisualizerColor,
}

impl Default for Palette {
    fn default() -> Palette {
        Palette {
            bg: Color::rgb(0x00, 0x00, 0x00),
            panel: Color::rgb(0x0e, 0x0e, 0x0e),
            raised: Color::rgb(0x1a, 0x1a, 0x1a),
            hover: Color::rgb(0x14, 0x14, 0x14),
            select: Color::rgb(0x1c, 0x1c, 0x1c),
            press: Color::rgb(0x24, 0x24, 0x24),
            panel_hover: Color::rgb(0x1a, 0x1a, 0x1a),
            panel_select: Color::rgb(0x22, 0x22, 0x22),
            panel_press: Color::rgb(0x2c, 0x2c, 0x2c),
            raised_top: Color::rgb(0x33, 0x33, 0x33),
            track: Color::rgb(0x2e, 0x2e, 0x2e),
            thumb: Color::rgb(0x34, 0x34, 0x34),
            thumb_hot: Color::rgb(0x5c, 0x5c, 0x5c),
            fg: Color::rgb(0xec, 0xec, 0xec),
            muted: Color::rgb(0x8a, 0x8a, 0x8a),
            faint: Color::rgb(0x52, 0x52, 0x52),
            accent: Color::rgb(0xec, 0xec, 0xec),
            on_accent: Color::rgb(0x00, 0x00, 0x00),
            lyrics_fg: Color::rgb(0xec, 0xec, 0xec),
            lyrics_outline: Color::rgb(0x00, 0x00, 0x00),
            visualizer: VisualizerColor::AlbumArt,
        }
    }
}

impl Palette {
    /// A palette built from a theme's three colours. `primary` sets the surface palette's hue and
    /// depth, `secondary` sets the ink, and `accent` lights interactive highlights and controls.
    pub fn from_colors(primary: Color, secondary: Color, accent: Color) -> Palette {
        Self::from_theme(
            primary,
            secondary,
            accent,
            None,
            None,
            VisualizerColor::AlbumArt,
        )
    }

    /// A palette built from all theme colours, including direct lyrics fill/outline and visualizer
    /// colour controls.
    pub fn from_theme(
        primary: Color,
        secondary: Color,
        accent: Color,
        lyrics_fg: Option<Color>,
        lyrics_outline: Option<Color>,
        visualizer: VisualizerColor,
    ) -> Palette {
        let (h, s, l) = rgb_to_hsl(primary);
        let s_surf = if s < 0.06 { s } else { s.clamp(0.26, 0.72) };
        let bg_l = if primary.r == 0 && primary.g == 0 && primary.b == 0 {
            0.0
        } else {
            l.clamp(0.085, 0.165)
        };
        let surf = |lightness: f32, sat_scale: f32| {
            hsl_to_rgb(
                h,
                (s_surf * sat_scale).clamp(0.0, 0.85),
                lightness.clamp(0.0, 0.92),
            )
        };

        let bg = surf(bg_l, 1.0);
        let panel = surf(bg_l + 0.055, 0.96);
        let raised = surf(bg_l + 0.105, 0.90);

        // Keep ink crisp and readable over rich coloured surfaces while preserving secondary's hue.
        let fg = if luminance(secondary) < 0.45 {
            mix(secondary, Color::rgb(0xf4, 0xf4, 0xf6), 0.78)
        } else {
            secondary
        };

        Palette {
            bg,
            panel,
            raised,
            hover: surf(bg_l + 0.038, 0.96),
            select: mix(surf(bg_l + 0.075, 0.92), accent, 0.14),
            press: surf(bg_l + 0.120, 0.88),
            panel_hover: surf(bg_l + 0.098, 0.92),
            panel_select: mix(surf(bg_l + 0.090, 0.90), accent, 0.24),
            panel_press: surf(bg_l + 0.150, 0.86),
            raised_top: surf(bg_l + 0.190, 0.82),
            track: surf(bg_l + 0.155, 0.75),
            thumb: surf(bg_l + 0.195, 0.70),
            thumb_hot: mix(surf(bg_l + 0.310, 0.65), accent, 0.28),
            fg,
            muted: mix(fg, panel, 0.40),
            faint: mix(fg, panel, 0.64),
            accent,
            on_accent: if luminance(accent) > 0.52 {
                Color::rgb(0x0a, 0x0a, 0x0a)
            } else {
                Color::rgb(0xf4, 0xf4, 0xf4)
            },
            lyrics_fg: lyrics_fg.unwrap_or(fg),
            lyrics_outline: lyrics_outline.unwrap_or(Color::rgb(0x00, 0x00, 0x00)),
            visualizer,
        }
    }
}

/// Convert an sRGB [`Color`] to `(hue 0..1, saturation 0..1, lightness 0..1)`.
pub fn rgb_to_hsl(c: Color) -> (f32, f32, f32) {
    let r = c.r as f32 / 255.0;
    let g = c.g as f32 / 255.0;
    let b = c.b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let d = max - min;
    if d < 1e-5 {
        return (0.0, 0.0, l);
    }
    let s = d / (1.0 - (2.0 * l - 1.0).abs()).max(1e-5);
    let mut h = if (max - r).abs() < 1e-5 {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if (max - g).abs() < 1e-5 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h /= 6.0;
    (h.rem_euclid(1.0), s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
}

/// Convert `(hue 0..1, saturation 0..1, lightness 0..1)` to an opaque [`Color`].
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Color {
    let h = h.rem_euclid(1.0);
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);
    if s < 1e-5 {
        let v = (l * 255.0).round() as u8;
        return Color::rgb(v, v, v);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let chan = |t: f32| {
        let t = t.rem_euclid(1.0);
        let c = if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        };
        (c * 255.0).round().clamp(0.0, 255.0) as u8
    };
    Color::rgb(chan(h + 1.0 / 3.0), chan(h), chan(h - 1.0 / 3.0))
}

/// How bright a colour reads, 0 to 1.
pub fn luminance(color: Color) -> f32 {
    (0.2126 * color.r as f32 + 0.7152 * color.g as f32 + 0.0722 * color.b as f32) / 255.0
}

static CURRENT: RwLock<Palette> = RwLock::new(Palette {
    bg: Color::rgb(0x00, 0x00, 0x00),
    panel: Color::rgb(0x0e, 0x0e, 0x0e),
    raised: Color::rgb(0x1a, 0x1a, 0x1a),
    hover: Color::rgb(0x14, 0x14, 0x14),
    select: Color::rgb(0x1c, 0x1c, 0x1c),
    press: Color::rgb(0x24, 0x24, 0x24),
    panel_hover: Color::rgb(0x1a, 0x1a, 0x1a),
    panel_select: Color::rgb(0x22, 0x22, 0x22),
    panel_press: Color::rgb(0x2c, 0x2c, 0x2c),
    raised_top: Color::rgb(0x33, 0x33, 0x33),
    track: Color::rgb(0x2e, 0x2e, 0x2e),
    thumb: Color::rgb(0x34, 0x34, 0x34),
    thumb_hot: Color::rgb(0x5c, 0x5c, 0x5c),
    fg: Color::rgb(0xec, 0xec, 0xec),
    muted: Color::rgb(0x8a, 0x8a, 0x8a),
    faint: Color::rgb(0x52, 0x52, 0x52),
    accent: Color::rgb(0xec, 0xec, 0xec),
    on_accent: Color::rgb(0x00, 0x00, 0x00),
    lyrics_fg: Color::rgb(0xec, 0xec, 0xec),
    lyrics_outline: Color::rgb(0x00, 0x00, 0x00),
    visualizer: VisualizerColor::AlbumArt,
});

/// The palette every widget reads this frame.
pub fn palette() -> Palette {
    *CURRENT.read().unwrap_or_else(|poison| poison.into_inner())
}

/// Put a palette in force. The next frame is drawn with it.
pub fn set_palette(palette: Palette) {
    *CURRENT.write().unwrap_or_else(|poison| poison.into_inner()) = palette;
}

// Surfaces, darkest to lightest.
/// The page: content sits directly on it.
pub fn bg() -> Color {
    palette().bg
}
/// Sidebar and player bar: a neutral grey lifted just off the black.
pub fn panel() -> Color {
    palette().panel
}
/// Search field, art tile, other small raised shapes.
pub fn raised() -> Color {
    palette().raised
}

// Interaction states are opaque tonal steps, not white-at-low-alpha washes. The swapchain is sRGB
// and blends in linear light, so a 6% white wash lands nearer 30% grey on screen; opaque colours
// land exactly where they are written.
//
// On the page:
pub fn hover() -> Color {
    palette().hover
}
pub fn select() -> Color {
    palette().select
}
pub fn press() -> Color {
    palette().press
}
// On a panel (sidebar, player bar), one step lighter to stay visible against the lighter ground:
pub fn panel_hover() -> Color {
    palette().panel_hover
}
pub fn panel_select() -> Color {
    palette().panel_select
}
pub fn panel_press() -> Color {
    palette().panel_press
}
// On a raised shape (the search field):
pub fn raised_top() -> Color {
    palette().raised_top
}
/// Slider track, on a panel.
pub fn track() -> Color {
    palette().track
}
/// Scrollbar thumb, at rest and while the list is hovered.
pub fn thumb() -> Color {
    palette().thumb
}
pub fn thumb_hot() -> Color {
    palette().thumb_hot
}

// Text.
pub fn fg() -> Color {
    palette().fg
}
pub fn muted() -> Color {
    palette().muted
}
pub fn faint() -> Color {
    palette().faint
}
/// The one accent: the play disc, the slider fill.
pub fn accent() -> Color {
    palette().accent
}
/// Ink on top of the accent (the play button's glyph).
pub fn on_accent() -> Color {
    palette().on_accent
}
/// Lyrics text fill colour.
pub fn lyrics_fg() -> Color {
    palette().lyrics_fg
}
/// Lyrics text outline colour.
pub fn lyrics_outline() -> Color {
    palette().lyrics_outline
}
/// Audio visualizer colour mode on the lyrics page.
pub fn visualizer() -> VisualizerColor {
    palette().visualizer
}

/// Convert an sRGB [`Color`] to `(hue 0..1, saturation 0..1, value 0..1)`.
pub fn rgb_to_hsv(c: Color) -> (f32, f32, f32) {
    let r = c.r as f32 / 255.0;
    let g = c.g as f32 / 255.0;
    let b = c.b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max < 1e-5 { 0.0 } else { d / max };
    let h = if d < 1e-5 {
        0.0
    } else if (max - r).abs() < 1e-5 {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if (max - g).abs() < 1e-5 {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };
    (h.rem_euclid(1.0), s.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

/// Convert `(hue 0..1, saturation 0..1, value 0..1)` to an opaque [`Color`].
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let h = h.rem_euclid(1.0) * 6.0;
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color::rgb(
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

/// The close button's glyph on hover. A system warning colour, not a theme colour.
pub const CLOSE_RED: Color = Color::rgb(0xe8, 0x4b, 0x4b);

// Metrics, in logical pixels.
pub const SIDEBAR_OPEN: f32 = 220.0;
pub const SIDEBAR_SHUT: f32 = 68.0;
/// Below this window width the sidebar folds itself away.
pub const SIDEBAR_AUTO_SHUT: f32 = 860.0;
pub const BAR_H: f32 = 72.0;
/// Radius of the concave fillet where the sidebar meets the player bar.
pub const JOIN_R: f32 = 12.0;
/// The strip along the top of the page where the search field lives.
pub const TOP_H: f32 = 76.0;
/// Each caption button is a square of this size; the cluster is one button tall.
pub const CAPTION_BTN_W: f32 = 36.0;
pub const CAPTION_H: f32 = 36.0;
pub const CAPTION_INSET: f32 = 10.0;
/// The search field's height, and its resting distance from the window's top edge. The caption
/// buttons are centred on the same line, so the two sit level.
pub const SEARCH_H: f32 = 46.0;
pub const SEARCH_TOP: f32 = 16.0;
pub const PAGE_PAD: f32 = 36.0;
pub const ROW_H: f32 = 64.0;

/// The same colour at `amount` of its alpha. `amount` is clamped to 0..=1.
pub fn fade(color: Color, amount: f32) -> Color {
    let a = (color.a as f32 * amount.clamp(0.0, 1.0)).round() as u8;
    Color::rgba(color.r, color.g, color.b, a)
}

/// Linear blend between two colours.
pub fn mix(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    Color::rgba(
        lerp(from.r, to.r),
        lerp(from.g, to.g),
        lerp(from.b, to.b),
        lerp(from.a, to.a),
    )
}

/// Ease-out cubic: fast start, long soft landing. For time-driven (not spring) motion.
pub fn ease_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

/// Extract the most visually "interesting" (vibrant, non-extreme, well-represented) colour from an
/// RGBA8 image, following the downscale -> quantize -> vibrancy/population scoring -> dark-UI
/// readability pipeline used by music players such as Spotify.
///
/// Intended to be called once per image and cached; subsamples on a <=48x48 grid so even large
/// covers finish in microseconds.
pub fn interesting_color(rgba: &[u8], width: u32, height: u32) -> Option<Color> {
    if width == 0 || height == 0 || rgba.len() < (width as usize) * (height as usize) * 4 {
        return None;
    }
    let step = (width.max(height) / 48).max(1) as usize;
    let w = width as usize;
    let h = height as usize;

    // 4-bit-per-channel quantization (up to 4096 buckets) accumulating centroid sums.
    let mut buckets: HashMap<u16, (u32, u64, u64, u64)> = HashMap::new();
    let mut total = 0u32;
    for y in (0..h).step_by(step) {
        let row = y * w * 4;
        for x in (0..w).step_by(step) {
            let i = row + x * 4;
            let a = rgba[i + 3];
            if a < 128 {
                continue;
            }
            let r = rgba[i];
            let g = rgba[i + 1];
            let b = rgba[i + 2];
            let key = ((u16::from(r >> 4)) << 8) | ((u16::from(g >> 4)) << 4) | u16::from(b >> 4);
            let entry = buckets.entry(key).or_insert((0, 0, 0, 0));
            entry.0 += 1;
            entry.1 += u64::from(r);
            entry.2 += u64::from(g);
            entry.3 += u64::from(b);
            total += 1;
        }
    }
    if total == 0 {
        return None;
    }

    let mut best_vibrant: Option<(f32, f32, f32, f32)> = None;
    let mut best_fallback: Option<(u32, f32, f32, f32)> = None;

    for (count, sr, sg, sb) in buckets.into_values() {
        let n = u64::from(count);
        let centroid = Color::rgb((sr / n) as u8, (sg / n) as u8, (sb / n) as u8);
        let (hue, sat, lit) = rgb_to_hsl(centroid);
        let pop = count as f32 / total as f32;

        if best_fallback.is_none_or(|(best_n, _, _, _)| count > best_n) {
            best_fallback = Some((count, hue, sat, lit));
        }

        // Skip near-black, near-white, desaturated greys, and sub-0.4% compression specks.
        if sat < 0.16 || !(0.14..=0.88).contains(&lit) || pop < 0.004 {
            continue;
        }
        let sat_score = ((sat - 0.12) / 0.88).clamp(0.0, 1.0).powf(1.25);
        let light_score = (1.0 - ((lit - 0.52).abs() / 0.40).powi(2)).max(0.0);
        let pop_weight = pop.sqrt().max(0.08);
        let score = (0.65 * sat_score + 0.35 * light_score) * pop_weight;

        if best_vibrant.is_none_or(|(best_s, _, _, _)| score > best_s) {
            best_vibrant = Some((score, hue, sat, lit));
        }
    }

    let (_, hue, sat, lit) =
        best_vibrant.or_else(|| best_fallback.map(|(_, h, s, l)| (0.0, h, s, l)))?;

    // Keep the chosen hue vivid and readable against dark UI surfaces.
    let tuned_l = lit.clamp(0.48, 0.72);
    let tuned_s = if sat >= 0.16 { sat.max(0.42) } else { sat };
    Some(hsl_to_rgb(hue, tuned_s, tuned_l))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_palette_is_the_shipped_black_and_grey() {
        let palette = Palette::default();
        assert_eq!(palette.bg, Color::rgb(0x00, 0x00, 0x00));
        assert_eq!(palette.panel, Color::rgb(0x0e, 0x0e, 0x0e));
        assert_eq!(palette.fg, Color::rgb(0xec, 0xec, 0xec));
        assert_eq!(palette.muted, Color::rgb(0x8a, 0x8a, 0x8a));
    }

    #[test]
    fn a_theme_keeps_its_surfaces_in_order_and_its_ink_readable() {
        let palette = Palette::from_colors(
            Color::rgb(0x2b, 0x33, 0x50),
            Color::rgb(0xe8, 0xed, 0xff),
            Color::rgb(0x8f, 0xb3, 0xff),
        );
        assert!(luminance(palette.bg) < luminance(palette.panel));
        assert!(luminance(palette.panel) < luminance(palette.raised));
        assert!(luminance(palette.fg) > luminance(palette.muted));
        assert!(luminance(palette.muted) > luminance(palette.faint));
        assert!(luminance(palette.fg) > 0.7, "the text must stay readable");
        assert_eq!(palette.accent, Color::rgb(0x8f, 0xb3, 0xff));
    }

    #[test]
    fn near_black_text_is_lifted_so_it_can_be_seen_at_all() {
        let palette = Palette::from_colors(
            Color::rgb(0xff, 0xff, 0xff),
            Color::rgb(0x05, 0x05, 0x05),
            Color::rgb(0xff, 0xff, 0xff),
        );
        assert!(luminance(palette.fg) > 0.5);
        assert_eq!(palette.on_accent, Color::rgb(0x0a, 0x0a, 0x0a));
    }

    #[test]
    fn interesting_color_prefers_vibrant_accent_over_large_grey_background() {
        let (w, h) = (48u32, 48u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                // 85% dull grey background, 15% vivid coral-pink accent block.
                if x < 18 && y < 18 {
                    rgba.extend_from_slice(&[240, 72, 120, 255]);
                } else {
                    rgba.extend_from_slice(&[42, 44, 48, 255]);
                }
            }
        }
        let picked = interesting_color(&rgba, w, h).expect("extracts color");
        let (_, sat, lit) = rgb_to_hsl(picked);
        assert!(sat > 0.5, "expected vibrant color, got {:?}", picked);
        assert!((0.45..=0.75).contains(&lit));
    }

    #[test]
    fn interesting_color_handles_monochrome_and_empty_inputs() {
        assert_eq!(interesting_color(&[], 0, 0), None);
        let grey = [90u8, 90, 90, 255].repeat(16 * 16);
        let picked = interesting_color(&grey, 16, 16).expect("fallback color");
        let (_, _, lit) = rgb_to_hsl(picked);
        assert!((0.45..=0.75).contains(&lit));
    }

    #[test]
    fn hsv_round_trip_preserves_colors() {
        for color in [
            Color::rgb(0xff, 0x00, 0x00),
            Color::rgb(0x7a, 0xa2, 0xff),
            Color::rgb(0x2a, 0x8c, 0x55),
            Color::rgb(0xec, 0xec, 0xec),
            Color::rgb(0x00, 0x00, 0x00),
        ] {
            let (h, s, v) = rgb_to_hsv(color);
            let back = hsv_to_rgb(h, s, v);
            assert!(
                (color.r as i16 - back.r as i16).abs() <= 1
                    && (color.g as i16 - back.g as i16).abs() <= 1
                    && (color.b as i16 - back.b as i16).abs() <= 1,
                "round-trip mismatch for {color:?} -> {back:?}"
            );
        }
    }
}
