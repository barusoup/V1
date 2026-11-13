//! Shapes on `dream::ui::app`: solid discs, rings, an animated arc, pills, a
//! bordered disc and an avatar.
//!
//! ```text
//! cargo run --release --example shapes
//! ```
//!
//! The arc is driven by [`Ui::anim_loop_f32`]. While an animation is in
//! flight the window redraws every refresh, and the loop returns to 0 Hz as
//! soon as nothing moves.

use std::f32::consts::{FRAC_PI_2, TAU};

use dream::ui::app::{AppContext, AppOptions, RunError, UiApp, run};
use dream::ui::layout::{Justify, LayoutAlign as Align, Style};
use dream::ui::tokens::TypeRole;
use dream::ui::view::Paint;
use dream::ui::widgets::Ui;
use dream::ui::widgets::adapters::{Shape, Text};

struct Shapes;

impl UiApp for Shapes {
    fn view(&mut self, ui: &mut Ui, cx: &mut AppContext<'_>) {
        // A looping 0..1 phase that advances with frame time.
        let progress = ui.anim_loop_f32("progress", 3.0);
        let theme = *cx.theme();
        ui.layout_root(cx.screen(), Style::column().pad(24.0).gap(16.0), |ui| {
            ui.add(Text::new("Shapes").role(TypeRole::Title));
            ui.row(Style::row().gap(16.0).align(Align::Center), |ui| {
                ui.add(Shape::circle(28.0).fill(theme.accent));
                ui.add(Shape::ring(48.0, 4.0));
                ui.add(Shape::arc(48.0, 4.0, -FRAC_PI_2, progress * TAU).fill(theme.accent));
                ui.add(Shape::circle(40.0).fill(theme.bg).stroke(2.0, theme.accent));
            });
            ui.row(Style::row().gap(12.0).align(Align::Center), |ui| {
                ui.add(Shape::pill(96.0, 28.0).fill(theme.g2));
                ui.add(Shape::rounded(96.0, 28.0, 6.0).fill(theme.g2));
            });
            ui.container(
                Style::row()
                    .size_px(48.0, 48.0)
                    .align(Align::Center)
                    .justify(Justify::Center),
                Paint::surface(&theme).radius_full(),
                |ui| {
                    ui.add(Text::new("JD"));
                },
            );
        });
    }
}

fn main() -> Result<(), RunError> {
    run(AppOptions::new("Shapes").size(520.0, 420.0), Shapes)
}
