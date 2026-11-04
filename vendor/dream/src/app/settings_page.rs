//! The Settings page: the app's own preferences, on a page in the main pane.
//!
//! One left-aligned column, the same language as the inspector: sentence-case
//! headings, hairline rules under each section and controls that take their
//! colours from the theme. The page never touches the scene or the viewport,
//! so it can be opened from any page.
//!
//! Built on the L3 view layer: `layout_root` + containers solve every rect;
//! no hand `y +=` bookkeeping.

use super::App;
use crate::ui::Rect;
use crate::ui::layout::{Justify, LayoutAlign, Style};
use crate::ui::theme::Theme;
use crate::ui::tokens::{ControlSize, Space, TypeRole};
use crate::ui::view::Paint;
use crate::ui::widgets::adapters::{Button, Segmented, Separator, Text};
use crate::ui::widgets::headless;

/// Left inset of the content column, matching the scene manager's rows.
const PAD: f32 = Space::S6;
/// Widest the column grows: the page is a column of settings, not a table.
const COL_W: f32 = 440.0;
/// Width of the Theme row's three-way selector.
const SEG_W: f32 = 240.0;

/// Which palette the app should use: the Settings page's Theme row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThemeChoice {
    /// Follow the OS appearance reported by the window.
    #[default]
    System,
    Light,
    Dark,
}

impl ThemeChoice {
    /// Every choice, in selector order.
    pub const ALL: [ThemeChoice; 3] = [ThemeChoice::System, ThemeChoice::Light, ThemeChoice::Dark];

    /// Position in [`Self::ALL`], and in the selector's row of segments.
    pub const fn index(self) -> usize {
        match self {
            ThemeChoice::System => 0,
            ThemeChoice::Light => 1,
            ThemeChoice::Dark => 2,
        }
    }

    /// Selector label. Also the widget id, so it must stay unique.
    pub fn label(self) -> &'static str {
        match self {
            ThemeChoice::System => "System",
            ThemeChoice::Light => "Light",
            ThemeChoice::Dark => "Dark",
        }
    }

    /// The palette this choice resolves to. `system_dark` is the OS
    /// preference the window reported.
    pub fn theme(self, system_dark: bool) -> Theme {
        match self {
            ThemeChoice::Light => Theme::light(),
            ThemeChoice::System if !system_dark => Theme::light(),
            ThemeChoice::System | ThemeChoice::Dark => Theme::dark_theme(),
        }
    }
}

/// Draw the Settings page into the main pane's content area.
pub fn draw_settings_page(app: &mut App, area: Rect) {
    let theme = app.theme;
    let font = app.font();
    // What the closures report back; applied after layout_root returns.
    let mut done = false;
    let mut chosen = app.theme_choice.index();

    // Opaque: the viewport's 3D pass runs under every main-pane page.
    app.ui.layout_root(area, Style::column().fill(), |ui| {
        ui.container(Style::column().fill(), Paint::new().bg(theme.bg), |ui| {
            ui.container(Style::column().fill().scroll_y(), Paint::none(), |ui| {
                ui.column(
                    Style::column()
                        .pad(PAD)
                        .gap(Space::S5)
                        .max_w(COL_W + PAD * 2.0)
                        .fill_w(),
                    |ui| {
                        // The page hides the tab strip while it is
                        // open, so it carries its own way back:
                        // Done returns to the tab page Settings
                        // was opened from (the toolbar gear
                        // toggles the same way).
                        ui.row(
                            Style::default()
                                .fill_w()
                                .h_px(Space::S8)
                                .gap(Space::S2)
                                .justify(Justify::SpaceBetween)
                                .align(LayoutAlign::Center),
                            |ui| {
                                ui.add(Text::new("Settings").role(TypeRole::Title));
                                if ui.add(Button::new("Done").size(ControlSize::Sm)).clicked {
                                    done = true;
                                }
                            },
                        );

                        ui.add(Text::new("Appearance").role(TypeRole::Heading));
                        ui.add(Separator::horizontal());

                        // Theme row: label on the left, the
                        // three-way selector on the right.
                        ui.row(
                            Style::default()
                                .fill_w()
                                .h_px(Space::S8)
                                .gap(Space::S2)
                                .justify(Justify::SpaceBetween)
                                .align(LayoutAlign::Center),
                            |ui| {
                                ui.add(Text::new("Theme").role(TypeRole::Body));
                                let mut sel = chosen;
                                let labels: [&str; 3] = [
                                    ThemeChoice::System.label(),
                                    ThemeChoice::Light.label(),
                                    ThemeChoice::Dark.label(),
                                ];
                                ui.add_styled(
                                    Segmented::new("settings_theme", &labels, &mut sel)
                                        .track_bg(theme.g1)
                                        .pill_bg(theme.g2)
                                        .text(theme.muted)
                                        .text_selected(theme.fg),
                                    Style::default().w_px(SEG_W).h_px(Space::S6),
                                );
                                chosen = sel;
                            },
                        );
                        ui.add(
                            Text::new("System follows the OS appearance setting.")
                                .role(TypeRole::Caption),
                        );
                    },
                );
            });
        });
    });

    // Publish the real segment rects (solved) for the scripted drive phase —
    // read AFTER layout_root so a resized page reports current geometry.
    // Same cell math the segmented behaviour uses.
    if let Some(track) = app.ui.rect_of("settings_theme") {
        headless::segmented_cells(
            track,
            ThemeChoice::ALL.len(),
            app.ui.scale(),
            theme.segmented(font).inset,
            &mut app.theme_cells,
        );
    }
    if done {
        app.close_settings();
    }
    if chosen != app.theme_choice.index() {
        app.set_theme_choice(ThemeChoice::ALL[chosen]);
    }
}
