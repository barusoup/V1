//! A complete Dream app in one file: a counter, a toggle, a slider, a text
//! field and a row of status dots.
//!
//! ```text
//! cargo run --release --example hello
//! ```
//!
//! Start new apps from this file. [`run`] owns the window, renderer, input
//! and scheduling. The app only builds its frame, and with nothing to do the
//! window idles at 0 Hz.

use dream::ui::app::{AppContext, AppOptions, RunError, UiApp, run};
use dream::ui::layout::{LayoutAlign as Align, Style};
use dream::ui::tokens::TypeRole;
use dream::ui::widgets::Ui;
use dream::ui::widgets::adapters::{Button, Shape, Slider, Text, TextField, Toggle};

struct Hello {
    count: u32,
    enabled: bool,
    volume: f32,
    name: String,
}

impl UiApp for Hello {
    fn view(&mut self, ui: &mut Ui, cx: &mut AppContext<'_>) {
        let accent = cx.theme().accent;
        let muted = cx.theme().muted;
        ui.layout_root(cx.screen(), Style::column().pad(24.0).gap(12.0), |ui| {
            ui.add(Text::new("Hello, Dream").role(TypeRole::Title));
            ui.row(Style::row().gap(8.0).align(Align::Center), |ui| {
                for dot in 0..5 {
                    let lit = dot < self.count.min(5);
                    ui.add(Shape::circle(12.0).fill(if lit { accent } else { muted }));
                }
                ui.add(Text::new(&format!("Count {}", self.count)));
            });
            // Events set state; the widgets draw it back. `clicked` is only true
            // on the frame of the click, so counting here is safe.
            if ui.add(Button::new("Count up")).clicked {
                self.count += 1;
            }
            ui.add(Toggle::new("Enabled", &mut self.enabled));
            ui.add(Slider::new("Volume", &mut self.volume, 0.0..=1.0));
            ui.add(
                Text::new(&format!("Volume {:.0}%", self.volume * 100.0)).role(TypeRole::Caption),
            );
            ui.add(TextField::new("Name", &mut self.name).placeholder("Your name"));
            let greeting = if self.name.is_empty() {
                "Type your name above".to_string()
            } else {
                format!("Hi {}", self.name)
            };
            ui.add(Text::new(&greeting).role(TypeRole::Body));
        });
    }
}

fn main() -> Result<(), RunError> {
    let app = Hello {
        count: 0,
        enabled: true,
        volume: 0.5,
        name: String::new(),
    };
    run(AppOptions::new("Hello, Dream").size(420.0, 420.0), app)
}
