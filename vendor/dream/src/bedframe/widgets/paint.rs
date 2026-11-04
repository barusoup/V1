//! Painters: the draw half of the stock widgets. Each takes a
//! [`State`] from a headless behaviour (or a
//! hand-built one) and emits exactly what the L1 widget draws — this is
//! the code custom widgets reuse to look native.
//!
//! ```ignore
//! let (resp, state) = headless::ButtonBehavior::new("ok").update(ui, rect);
//! paint::button(ui, rect, &style, state, "ok");
//! ```

use super::headless::State;
use super::styles::{ButtonStyle, LabelStyle, SliderStyle};
use super::{Ui, display_label, mix_color};
use crate::ui::Rect;
use crate::ui::font::Align;
use crate::ui::layout::hairline;
use crate::ui::tokens::{ControlSize, Radius, Space};

/// The button chrome (bg, border, centred label) — no interaction.
pub fn button(ui: &mut Ui, rect: Rect, style: &ButtonStyle, state: State, label: &str) {
    let hover = state.hover;
    let bg = mix_color(
        mix_color(style.normal, style.hover, hover),
        style.pressed,
        if state.pressed { 1.0 } else { 0.0 },
    );
    ui.draw.round_rect(rect, style.corner_radius, bg);
    if style.normal.a > 0 && style.corner_radius > 0.0 && style.normal != style.hover {
        let border = mix_color(style.normal, style.hover, 0.55 + hover * 0.45);
        ui.draw
            .round_ring(rect, style.corner_radius, hairline(ui.scale), border);
    }
    if let Some(border) = style.border {
        ui.draw
            .round_ring(rect, style.corner_radius, hairline(ui.scale), border);
    }
    let text = if style.disabled {
        style.text_disabled
    } else {
        mix_color(style.text, style.text_hover, hover)
    };
    ui.label(
        rect,
        display_label(label),
        LabelStyle::new(style.font)
            .size(style.size)
            .color(text)
            .align(Align::Center)
            .vcenter(true),
    );
}

/// [`button`] with an icon+label pair centred together.
pub fn button_with_icon(
    ui: &mut Ui,
    rect: Rect,
    style: &ButtonStyle,
    state: State,
    icon: crate::ui::icons::Icon,
    icon_size: u32,
    label: &str,
) {
    let hover = state.hover;
    let bg = mix_color(
        mix_color(style.normal, style.hover, hover),
        style.pressed,
        if state.pressed { 1.0 } else { 0.0 },
    );
    ui.draw.round_rect(rect, style.corner_radius, bg);
    if style.normal.a > 0 && style.corner_radius > 0.0 && style.normal != style.hover {
        let border = mix_color(style.normal, style.hover, 0.55 + hover * 0.45);
        ui.draw
            .round_ring(rect, style.corner_radius, hairline(ui.scale), border);
    }
    let tint = if style.disabled {
        style.text_disabled
    } else {
        mix_color(style.text, style.text_hover, hover)
    };
    let text_str = display_label(label);
    let (tw, _) = ui.measure(text_str, style.font, style.size);
    let gap = if text_str.is_empty() {
        0.0
    } else {
        ui.theme.tokens.control(ControlSize::for_height(rect.h)).gap
    };
    let icon_w = icon_size as f32;
    let total_w = icon_w + gap + tw;
    let start_x = rect.center().0 - total_w * 0.5;
    let icon_rect = Rect::new(start_x, rect.y, icon_w, rect.h);
    ui.icon(icon_rect, icon, icon_size, tint);
    if !text_str.is_empty() {
        let label_rect = Rect::new(start_x + icon_w + gap, rect.y, tw + 4.0, rect.h);
        ui.label(
            label_rect,
            text_str,
            LabelStyle::new(style.font)
                .size(style.size)
                .color(tint)
                .align(Align::Left)
                .vcenter(true),
        );
    }
}

/// Icon button: tinted icon, optional badge in the bottom-right.
#[allow(clippy::too_many_arguments)]
pub fn icon_button(
    ui: &mut Ui,
    rect: Rect,
    style: &ButtonStyle,
    state: State,
    icon: crate::ui::icons::Icon,
    icon_size: u32,
    on: bool,
    badge: Option<&str>,
) {
    let hover = state.hover;
    // `state.pressed` mirrors `active_id == Some(id)` from the behaviour.
    let bg = if on {
        style.text
    } else if state.pressed {
        style.pressed
    } else {
        mix_color(style.normal, style.hover, hover)
    };
    ui.draw.round_rect(rect, style.corner_radius, bg);
    let tint = if on {
        style.on_icon
    } else if style.disabled {
        style.text_disabled
    } else {
        mix_color(style.text, style.text_hover, hover)
    };
    ui.icon(rect, icon, icon_size, tint);
    let badge_size = ui.text_metrics().badge;
    if let Some(badge) = badge {
        let (bw, bh) = ui.measure(badge, style.font, badge_size);
        ui.label(
            Rect::new(
                rect.max().0 - bw - Space::S0_5,
                rect.max().1 - bh - Space::S0_5,
                bw,
                bh,
            ),
            badge,
            LabelStyle::new(style.font)
                .size(badge_size)
                .color(if on {
                    style.on_icon
                } else {
                    style.text_disabled
                })
                .align(Align::Right),
        );
    }
}

/// Pill track + sliding knob; `state.value` is the animated on level.
/// Returns the track's right edge.
pub fn switch(ui: &mut Ui, rect: Rect, style: &ButtonStyle, state: State) -> f32 {
    let control = ui.theme.tokens.control(ControlSize::for_height(rect.h));
    let track_h = control.icon + 2.0;
    let track = Rect::new(
        rect.x,
        rect.center().1 - track_h * 0.5,
        (track_h * 1.9).round().min(rect.w),
        track_h,
    );
    let on_t = state.value;
    let track_color = mix_color(
        mix_color(style.normal, style.hover, state.hover),
        style.accent,
        on_t,
    );
    ui.draw.round_rect(track, track_h * 0.5, track_color);
    if on_t < 0.99 && style.normal.a > 0 {
        let ring = mix_color(style.normal, style.hover, 0.65).fade(1.0 - on_t);
        ui.draw
            .round_ring(track, track_h * 0.5, hairline(ui.scale), ring);
    }
    let knob_d = track_h - 4.0;
    let knob = Rect::new(
        track.x + 2.0 + (track.w - knob_d - 4.0) * on_t,
        track.center().1 - knob_d * 0.5,
        knob_d,
        knob_d,
    );
    let knob_color = mix_color(style.text_disabled, style.on_icon, on_t);
    ui.draw.round_rect(knob, knob_d * 0.5, knob_color);
    track.max().0
}

/// [`switch`] plus a right-hand label.
pub fn toggle(ui: &mut Ui, rect: Rect, style: &ButtonStyle, state: State, label: &str) {
    let track_right = switch(ui, rect, style, state);
    let label_color = if style.disabled {
        style.text_disabled
    } else {
        style.text
    };
    let gap = ui.theme.tokens.control(ControlSize::for_height(rect.h)).gap;
    ui.label(
        Rect::new(
            track_right + gap + 2.0,
            rect.y,
            (rect.max().0 - track_right - gap - 2.0).max(0.0),
            rect.h,
        ),
        display_label(label),
        LabelStyle::new(style.font)
            .size(style.size)
            .color(label_color)
            .align(Align::Left)
            .vcenter(true),
    );
}

/// Box, check mark and label; `state.value` is the check anim level.
pub fn checkbox(ui: &mut Ui, rect: Rect, style: &ButtonStyle, state: State, label: &str) {
    let on_t = state.value;
    let control = ui.theme.tokens.control(ControlSize::for_height(rect.h));
    let box_s = (control.icon + 2.0).min(rect.h);
    let box_rect = Rect::new(rect.x, rect.center().1 - box_s * 0.5, box_s, box_s);
    let bg = mix_color(
        mix_color(style.normal, style.hover, state.hover),
        style.accent,
        on_t,
    );
    let border = mix_color(
        mix_color(style.normal, style.hover, 0.7 + state.hover * 0.3),
        style.accent,
        on_t,
    );
    ui.draw.rounded_frame(
        box_rect,
        ui.theme.tokens.radius(Radius::Xs),
        hairline(ui.scale),
        bg,
        border,
    );
    if on_t > 0.05 {
        ui.icon(
            box_rect,
            crate::ui::icons::Icon::Check,
            (box_s - 4.0).max(10.0) as u32,
            style.on_icon.fade(on_t),
        );
    }
    let text_str = display_label(label);
    if !text_str.is_empty() {
        let text_color = if style.disabled {
            style.text_disabled
        } else {
            style.text
        };
        ui.label(
            Rect::new(
                box_rect.max().0 + control.gap + 2.0,
                rect.y,
                (rect.max().0 - box_rect.max().0 - 10.0).max(0.0),
                rect.h,
            ),
            text_str,
            LabelStyle::new(style.font)
                .size(style.size)
                .color(text_color)
                .align(Align::Left)
                .vcenter(true),
        );
    }
}

/// Circle, dot and label; `state.value` is the select anim level.
pub fn radio(ui: &mut Ui, rect: Rect, style: &ButtonStyle, state: State, label: &str) {
    let on_t = state.value;
    let control = ui.theme.tokens.control(ControlSize::for_height(rect.h));
    let d = (control.icon + 2.0).min(rect.h);
    let circle = Rect::new(rect.x, rect.center().1 - d * 0.5, d, d);
    let bg = mix_color(style.normal, style.hover, state.hover);
    let border = mix_color(
        mix_color(style.normal, style.hover, 0.7 + state.hover * 0.3),
        style.accent,
        on_t,
    );
    ui.draw.rounded_frame(circle, d * 0.5, 1.25, bg, border);
    if on_t > 0.02 {
        let dot_d = (d - 8.0).max(4.0) * on_t;
        let c = circle.center();
        ui.draw.round_rect(
            Rect::new(c.0 - dot_d * 0.5, c.1 - dot_d * 0.5, dot_d, dot_d),
            dot_d * 0.5,
            style.accent.fade(on_t),
        );
    }
    let text_str = display_label(label);
    if !text_str.is_empty() {
        let text_color = if style.disabled {
            style.text_disabled
        } else {
            style.text
        };
        ui.label(
            Rect::new(
                circle.max().0 + control.gap + 2.0,
                rect.y,
                (rect.max().0 - circle.max().0 - 10.0).max(0.0),
                rect.h,
            ),
            text_str,
            LabelStyle::new(style.font)
                .size(style.size)
                .color(text_color)
                .align(Align::Left)
                .vcenter(true),
        );
    }
}

/// Slider track/fill/knob; `state.value` is the normalized t.
pub fn slider(ui: &mut Ui, rect: Rect, style: &SliderStyle, state: State, label: &str) {
    let track = Rect::new(rect.x, rect.center().1 - 3.0, rect.w, 6.0);
    ui.draw.round_rect(track, track.h * 0.5, style.track);
    let t = state.value;
    if track.w * t > 0.0 {
        ui.draw.round_rect(
            Rect::new(track.x, track.y, track.w * t, track.h),
            track.h * 0.5,
            style.fill,
        );
    }
    let knob_size = rect.h.min(22.0);
    let knob = Rect::new(
        track.x + track.w * t - knob_size * 0.5,
        rect.center().1 - knob_size * 0.5,
        knob_size,
        knob_size,
    );
    let knob_color = mix_color(style.knob, style.fill, state.hover * 0.35);
    ui.draw.round_rect(knob, knob_size * 0.5, knob_color);
    ui.draw.round_ring(
        knob,
        knob_size * 0.5,
        1.0,
        mix_color(style.track, style.knob, 0.35),
    );
    if style.show_value {
        ui.label(
            Rect::new(rect.x, rect.y - 18.0, rect.w, 14.0),
            display_label(label),
            LabelStyle::new(style.font)
                .size(ui.text_metrics().body)
                .color(style.knob)
                .align(Align::Left)
                .vcenter(true),
        );
    }
}

/// The segmented pill track: outer frame + gliding pill. Item content is
/// drawn by the caller (see [`Ui::segmented_with`]).
pub fn segmented(
    ui: &mut Ui,
    rect: Rect,
    style: &super::SegmentedStyle,
    pill_at: f32,
    cells: &[(f32, f32)],
) {
    // Track: fill + optional border.
    match style.track_border {
        Some((w, c)) => {
            ui.draw
                .rounded_frame(rect, style.button.corner_radius, w, style.track, c);
        }
        None => ui
            .draw
            .round_rect(rect, style.button.corner_radius, style.track),
    }
    let inset = style.inset;
    let n = cells.len();
    if n == 0 {
        return;
    }
    let roll = (std::f32::consts::PI * (pill_at - pill_at.floor())).sin();
    let (i0, i1) = (
        (pill_at.floor() as usize).min(n - 1),
        (pill_at.ceil() as usize).min(n - 1),
    );
    let t = pill_at - i0 as f32;
    let cell_w = cells[i0].1 + (cells[i1].1 - cells[i0].1) * t;
    let cell_x = cells[i0].0 + (cells[i1].0 - cells[i0].0) * t;
    let width = (cell_w - inset * 2.0).max(1.0) * (1.0 - style.squash * roll);
    let center_x = cell_x + cell_w * 0.5;
    let pill = Rect::new(
        center_x - width * 0.5,
        rect.y + inset,
        width,
        (rect.h - inset * 2.0).max(1.0),
    );
    ui.draw.round_rect(
        pill,
        Radius::inner(style.button.corner_radius, 2.0),
        style.pill,
    );
}

/// drag_field chrome: tinted box, dim label, right-aligned value.
pub fn drag_field(
    ui: &mut Ui,
    rect: Rect,
    font: crate::ui::font::FontId,
    state: State,
    label: &str,
    value: f32,
) {
    let theme = ui.theme;
    let bg = mix_color(theme.g2, theme.g3, state.hover);
    ui.draw.round_rect(rect, 6.0, bg);
    ui.label(
        Rect::new(rect.x + 8.0, rect.y, rect.w * 0.55, rect.h),
        display_label(label),
        LabelStyle::new(font)
            .size(ui.text_metrics().body)
            .color(theme.muted)
            .vcenter(true),
    );
    ui.label(
        Rect::new(rect.x + rect.w * 0.5, rect.y, rect.w * 0.5 - 8.0, rect.h),
        &format!("{value:.2}"),
        LabelStyle::new(font)
            .size(ui.text_metrics().body)
            .color(theme.fg)
            .align(Align::Right)
            .vcenter(true),
    );
}
