//! The right-hand inspector: the editor's own render and AA settings.
//!
//! One column, generous spacing, hairline separators and sentence-case
//! headings. Every row is on whole pixels and every label is centred with the
//! font's real metrics rather than a hand-tuned offset.
//!
//! There is deliberately no scene section: the app boots with no scene
//! loaded, and what sits in the viewport is the user's business, not the
//! inspector's.

use super::App;
use crate::ui::Rect;
use crate::ui::font::Align;
use crate::ui::widgets::LabelStyle;

/// Total scrollable inspector height.
const CONTENT_H: f32 = 190.0;
/// Height of the render section: header plus body.
const RENDER_H: f32 = 176.0;
/// Height of a labelled row.
const ROW_H: f32 = 28.0;
/// Vertical gap between rows.
const ROW_GAP: f32 = 8.0;

/// Centre a single-line label in `rect`.
fn row_label(app: &mut App, rect: Rect, text: &str, color: crate::ui::Color, size: f32) {
    app.ui.label(
        rect,
        text,
        LabelStyle::new(app.font())
            .size(size)
            .color(color)
            .vcenter(true),
    );
}

pub fn inspector(app: &mut App) {
    let panel = app.layout.inspector;
    let theme = app.theme;
    let m = theme.metrics;
    app.ui.draw_list_mut().rect(panel, theme.g1);
    app.ui
        .draw_list_mut()
        .rect(Rect::new(panel.x, panel.y, m.border, panel.h), theme.border);

    let x = panel.x + m.pad;
    let width = panel.w - 2.0 * m.pad;
    let inner = Rect::new(x, panel.y + m.pad * 0.5, width, panel.h - m.pad);
    let content = app.ui.scroll_begin("inspector", inner, CONTENT_H);
    let top = content.y;

    let mut open = app.inspector.render_open;
    render_section(app, x, width, top, &mut open);
    app.inspector.render_open = open;

    app.ui.scroll_end();
}

/// A section header with a chevron. Returns the body rect and whether the body
/// is currently drawn behind a clip (which the caller pops with
/// [`crate::ui::widgets::Ui::collapsible_end`]).
fn section(app: &mut App, top: f32, height: f32, open: &mut bool, title: &str) -> (Rect, bool) {
    let panel = app.layout.inspector;
    let m = app.theme.metrics;
    let x = panel.x + m.pad;
    let width = panel.w - 2.0 * m.pad;
    let rect = Rect::new(x, top, width, height);
    let (body, clipped) = app.ui.collapsible(title, rect, open, app.font());
    if clipped {
        let theme = app.theme;
        app.ui.draw_list_mut().rect(
            Rect::new(x, top + 28.0, width, theme.metrics.border),
            theme.border,
        );
    }
    (body, clipped)
}

fn render_section(app: &mut App, x: f32, width: f32, top: f32, open: &mut bool) {
    let (body, clipped) = section(app, top, RENDER_H, open, "Render");
    if !clipped {
        return;
    }
    // While the section animates shut it is still drawn, but it stops taking
    // input: a control you cannot see must not be clickable.
    let live = *open;
    let theme = app.theme;
    let font = app.font();
    let mut y = body.y + 6.0;

    // Anti-aliasing (FSR 3.1) toggle.
    row_label(
        app,
        Rect::new(x, y, width * 0.6, ROW_H),
        "FSR 3.1",
        theme.fg,
        13.0,
    );
    let mut aa = app.settings.antialias;
    let toggled = app
        .ui
        .switch(
            "aa",
            Rect::new(x + width * 0.5, y + 3.0, width * 0.5, 22.0),
            &mut aa,
            theme.button(font),
        )
        .clicked;
    if toggled && live {
        app.toggle_antialias();
    }
    y += ROW_H + ROW_GAP;

    // Upscale preset is fixed at startup; shown as a read-only value.
    row_label(
        app,
        Rect::new(x, y, width * 0.5, ROW_H),
        "Upscale",
        theme.muted,
        13.0,
    );
    app.ui.label(
        Rect::new(x + width * 0.5, y, width * 0.5, ROW_H),
        &format!("{:?}", app.settings.upscale),
        LabelStyle::new(font)
            .size(13.0)
            .color(theme.fg)
            .align(Align::Right)
            .vcenter(true),
    );
    y += ROW_H + ROW_GAP;

    // Sharpness: a label row, then the slider across the full column.
    row_label(
        app,
        Rect::new(x, y, width, 16.0),
        "Sharpness",
        theme.muted,
        12.0,
    );
    y += 18.0;
    let slider = Rect::new(x, y, width, 20.0);
    // Published so shortcuts and tests key off the real geometry.
    app.sharpness_rect = slider;
    let mut sharpness = app.settings.sharpness;
    let held = app
        .ui
        .slider(
            "sharpness",
            slider,
            &mut sharpness,
            0.0..=1.0,
            theme.slider(font),
        )
        .held;
    if held && live {
        app.settings.sharpness = sharpness;
        app.dirty = true;
    }
    y += 20.0 + 6.0;

    // Footer: the render size the engine is working at.
    row_label(
        app,
        Rect::new(x, y, width, 16.0),
        &format!(
            "Display {} × {} · scale {:.2}",
            app.screen.0 as u32,
            app.screen.1 as u32,
            app.engine_settings.render_scale()
        ),
        theme.faint,
        12.0,
    );

    app.ui.collapsible_end();
}
