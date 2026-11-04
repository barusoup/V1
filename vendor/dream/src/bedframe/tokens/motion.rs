//! Motion tokens: the canonical durations and easings widgets animate with.
//!
//! Durations are seconds at `motion_scale` 1.0 — the anim helpers apply the
//! scale themselves, so widgets pass these authored values through.

use super::super::tween::Ease;

/// The stock motion table. Values are the durations BedFrame's widgets
/// have always used, gathered into one place so a theme can slow the whole
/// UI down (or a test can reason about it) without touching call sites.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionTokens {
    /// Snap with no animation.
    pub instant: f32,
    /// Small state changes (0.09 s).
    pub fast: f32,
    /// The default transition (0.14 s).
    pub base: f32,
    /// Deliberate, larger moves (0.22 s).
    pub slow: f32,
    /// Hover fade in (snappier) and out (longer — leaving should linger).
    pub hover_in: f32,
    pub hover_out: f32,
    /// Switch/toggle travel.
    pub toggle: f32,
    /// Checkbox and radio fill.
    pub check: f32,
    /// Segmented-control pill travel.
    pub segmented: f32,
    /// Popup dismiss duration — popups only travel on the way out.
    pub popup_close: f32,
    /// Distance a popup slides back out of its anchor corner, in pixels.
    pub popup_slide: f32,
    /// Popup opacity: near-instant in, quicker out than the slide it rides.
    pub popup_fade_in: f32,
    pub popup_fade_out: f32,
    /// Toast entrance/exit.
    pub toast: f32,
    /// Collapsible-section unroll/dismiss: open is snappier than close.
    pub section_open: f32,
    pub section_close: f32,
    /// Seconds the pointer must rest before a tooltip appears.
    pub tooltip_delay: f32,
    /// Standard ease for state changes; enter/exit for appearing and
    /// leaving surfaces.
    pub ease_standard: Ease,
    pub ease_enter: Ease,
    pub ease_exit: Ease,
}

impl Default for MotionTokens {
    fn default() -> Self {
        Self {
            instant: 0.0,
            fast: 0.09,
            base: 0.14,
            slow: 0.22,
            hover_in: 0.12,
            hover_out: 0.18,
            toggle: 0.16,
            check: 0.14,
            segmented: 0.18,
            popup_close: 0.12,
            popup_slide: 8.0,
            popup_fade_in: 0.05,
            popup_fade_out: 0.09,
            toast: 0.22,
            section_open: 0.13,
            section_close: 0.10,
            tooltip_delay: 0.6,
            ease_standard: Ease::CubicOut,
            ease_enter: Ease::ExpoOut,
            ease_exit: Ease::CubicIn,
        }
    }
}
