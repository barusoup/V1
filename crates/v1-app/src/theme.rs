// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Design tokens: colours, metrics and the few colour helpers the whole UI shares.
//!
//! The palette is true black with tonal steps, never lines. Depth comes from three surfaces
//! (page, panel, raised) and from white washes at low alpha, so any surface can sit on any other
//! and still read.

use dream::ui::Color;

// Surfaces, darkest to lightest.
/// The page: content sits directly on it.
pub const BG: Color = Color::rgb(0x00, 0x00, 0x00);
/// Sidebar and player bar: a neutral grey lifted just off the black.
pub const PANEL: Color = Color::rgb(0x0e, 0x0e, 0x0e);
/// Search field, art tile, other small raised shapes.
pub const RAISED: Color = Color::rgb(0x1a, 0x1a, 0x1a);

// Interaction states are opaque tonal steps, not white-at-low-alpha washes. The swapchain is sRGB
// and blends in linear light, so a 6% white wash lands nearer 30% grey on screen; opaque colours
// land exactly where they are written.
//
// On the page:
pub const HOVER: Color = Color::rgb(0x14, 0x14, 0x14);
pub const SELECT: Color = Color::rgb(0x1c, 0x1c, 0x1c);
pub const PRESS: Color = Color::rgb(0x24, 0x24, 0x24);
// On a panel (sidebar, player bar), one step lighter to stay visible against the lighter ground:
pub const PANEL_HOVER: Color = Color::rgb(0x1a, 0x1a, 0x1a);
pub const PANEL_SELECT: Color = Color::rgb(0x22, 0x22, 0x22);
pub const PANEL_PRESS: Color = Color::rgb(0x2c, 0x2c, 0x2c);
// On a raised shape (the search field):
pub const RAISED_TOP: Color = Color::rgb(0x33, 0x33, 0x33);
/// Slider track, on a panel.
pub const TRACK: Color = Color::rgb(0x2e, 0x2e, 0x2e);
/// Scrollbar thumb, at rest and while the list is hovered.
pub const THUMB: Color = Color::rgb(0x34, 0x34, 0x34);
pub const THUMB_HOT: Color = Color::rgb(0x5c, 0x5c, 0x5c);

// Text.
pub const FG: Color = Color::rgb(0xec, 0xec, 0xec);
pub const MUTED: Color = Color::rgb(0x8a, 0x8a, 0x8a);
pub const FAINT: Color = Color::rgb(0x52, 0x52, 0x52);
/// Ink on top of the accent (the play button's glyph).
pub const ON_ACCENT: Color = Color::rgb(0x00, 0x00, 0x00);

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
/// The window's own caption buttons (minimise, close) in the top-right corner: each
/// button's width and the cluster's height, and the gap from the window's top and right edges.
/// The close button's glyph on hover.
pub const CLOSE_RED: Color = Color::rgb(0xe8, 0x4b, 0x4b);
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
