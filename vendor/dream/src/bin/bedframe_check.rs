//! `bedframe_check` — native GPU regression harness for BedFrame.
//!
//! Builds the real renderer on a hidden window (main thread, as macOS
//! requires), draws a themed UI — panel, text, button, image, a cropped
//! image and deliberately interleaved draw layers — captures the actual
//! RGBA output, and asserts on pixels: panel corners, image texels and
//! glyph ink, at UI scale 1 then 2, across a resize and across newly
//! rasterized characters landing after the first frame.
//!
//! It is a correctness harness, not a benchmark: timings are ignored,
//! pixels are not. Screenshots land in `target/bedframe-check/`.
//!
//! Run (MoltenVK):
//! ```text
//! VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
//! DYLD_LIBRARY_PATH=/opt/homebrew/lib \
//! cargo run --bin bedframe_check
//! ```

use dream::camera::Camera;
use dream::gfx::{GpuContext, Renderer};
use dream::settings::{AaMode, Settings, UpscalePreset};
use dream::texture::{GpuTexture, TextureManager, TextureOptions};
use dream::ui::atlas::AtlasConfig;
use dream::ui::draw::UiFrame;
use dream::ui::font::{FontId, Fonts};
use dream::ui::icons::Icon;
use dream::ui::theme::Theme;
use dream::ui::tokens::{Elevation, TypeRole};
use dream::ui::widgets::{
    ButtonStyle, LAYER_TOOLTIP, LabelStyle, MenuItem, PanelStyle, Ui, UiInput,
};
use dream::ui::{Color, Rect};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

/// One scripted step: a name, the UI scale to render at, an optional
/// physical resize before drawing, and the text to draw.
/// Beyond the scripted frames: retention and raw-upload edge cases.
#[derive(Clone, Copy, PartialEq)]
enum Extra {
    /// A frame whose custom texture is dropped the instant `render_ui`
    /// returns — the slot's retained list must keep it alive.
    DropFrame,
    /// A raw `UiFrame` with a legacy `atlas_updates` upload and no
    /// snapshot — the small legacy path (4-byte texel) must upload.
    LegacyUpload,
    /// CJK + colour emoji through the native text path: the ink assert
    /// covers the CJK; the extra check demands real colour texels.
    Unicode,
    /// `"Fp g"` label — structural orientation asserts (F's top bar, p's
    /// descender) prove native rasters are not vertically mirrored.
    Orient,
    /// Frame A kept while the atlas resets (new epoch): the retained
    /// frame's old page pixels must still sample correctly.
    AtlasReset,
    /// Same frame through the compact and legacy paths — captures must
    /// match within 1 LSB.
    CompactVsLegacy,
    /// The paint primitives (shadow, gradients, radial, rounded clip):
    /// dedicated pixel probes over the scene every step already draws.
    Primitives,
    /// Widget gallery pass: every stock widget at rest on a 640x480
    /// logical canvas; `true` renders the light palette.
    Gallery(bool),
    /// Same gallery with devtools outlines on — sanity assert only.
    GalleryDevtools,
    /// White discs on black at fractional centres, checked pixel by pixel
    /// against the analytic anti-aliasing coverage (see `check_circle_edges`).
    CircleEdges,
    /// Frosted glass over 8 px black/white stripes, captured on the display
    /// path (capture on, as the harness runs). See `check_glass`.
    Glass,
    /// The same glass rendered first for several frames with capture off, so
    /// it takes the direct (swapchain) path; then one capture-on frame is read
    /// back and checked like `Glass`.
    GlassDirect,
    UiOnlyBeforeScene,
    UiOnlyAfterScene,
}

struct Step {
    name: &'static str,
    scale: f32,
    resize_to: Option<(u32, u32)>,
    /// Text drawn this step: later steps add unseen characters to prove
    /// on-demand rasterization lands in the atlas mid-session.
    text: &'static str,
    extra: Option<Extra>,
}

const STEPS: &[Step] = &[
    Step {
        name: "ui-only-first",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::UiOnlyBeforeScene),
    },
    Step {
        name: "scale1",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: None,
    },
    Step {
        name: "scale2",
        scale: 2.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: None,
    },
    Step {
        name: "primitives",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::Primitives),
    },
    Step {
        name: "circles",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::CircleEdges),
    },
    Step {
        name: "circles-1.25",
        scale: 1.25,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::CircleEdges),
    },
    Step {
        name: "glass",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::Glass),
    },
    Step {
        name: "glass-direct",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::GlassDirect),
    },
    Step {
        name: "new-glyphs",
        scale: 2.0,
        resize_to: None,
        // Characters not rasterized in the earlier frames.
        text: "Zx9@#~",
        extra: None,
    },
    Step {
        name: "resized",
        scale: 1.0,
        resize_to: Some((480, 360)),
        text: "BedFrame OK",
        extra: None,
    },
    Step {
        name: "drop-frame",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::DropFrame),
    },
    Step {
        name: "legacy-upload",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::LegacyUpload),
    },
    Step {
        name: "unicode",
        scale: 1.0,
        resize_to: None,
        // CJK + a ZWJ emoji + a flag: Han/Kana roles shape, the emoji
        // lands coloured on an sRGB atlas page.
        text: "你好こんにちは👩‍💻🇯🇵",
        extra: Some(Extra::Unicode),
    },
    Step {
        name: "orient",
        scale: 1.0,
        resize_to: None,
        text: "Fp g",
        extra: Some(Extra::Orient),
    },
    Step {
        name: "atlas-reset",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::AtlasReset),
    },
    Step {
        name: "compact-vs-legacy",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::CompactVsLegacy),
    },
    Step {
        name: "unicode-scale2",
        scale: 2.0,
        resize_to: None,
        text: "你好こんにちは👩‍💻🇯🇵",
        extra: Some(Extra::Unicode),
    },
    Step {
        name: "ui-only-last",
        scale: 1.0,
        resize_to: None,
        text: "BedFrame OK",
        extra: Some(Extra::UiOnlyAfterScene),
    },
    // Widget gallery: one 640x480 logical canvas per palette and scale,
    // each needing a real window resize to 640*s x 480*s physical.
    Step {
        name: "gallery/dark-1x",
        scale: 1.0,
        resize_to: Some((640, 480)),
        text: "BedFrame OK",
        extra: Some(Extra::Gallery(false)),
    },
    Step {
        name: "gallery/dark-2x",
        scale: 2.0,
        resize_to: Some((1280, 960)),
        text: "BedFrame OK",
        extra: Some(Extra::Gallery(false)),
    },
    Step {
        name: "gallery/light-1x",
        scale: 1.0,
        resize_to: Some((640, 480)),
        text: "BedFrame OK",
        extra: Some(Extra::Gallery(true)),
    },
    Step {
        name: "gallery/light-2x",
        scale: 2.0,
        resize_to: Some((1280, 960)),
        text: "BedFrame OK",
        extra: Some(Extra::Gallery(true)),
    },
    Step {
        name: "gallery/devtools-outlines",
        scale: 1.0,
        resize_to: Some((640, 480)),
        text: "BedFrame OK",
        extra: Some(Extra::GalleryDevtools),
    },
];

struct Check {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    ui: Option<Ui>,
    texture: Option<Arc<GpuTexture>>,
    step: usize,
    done: bool,
    /// A redraw was requested last pass; cleared when the platform
    /// delivers `RedrawRequested`. Still set at the next `about_to_wait`
    /// means the platform never delivers redraws for hidden windows
    /// (Windows does not) — the loop then drives the step directly.
    redraw_pending: bool,
    /// Set while a real window resize is in flight — the step renders
    /// only after `WindowEvent::Resized` lands.
    resize_pending: Option<(u32, u32)>,
    /// Pixels captured by the compact pass of `compact-vs-legacy`.
    compact_capture: Option<(Vec<u8>, u32, u32)>,
    /// Retained frame A for the atlas-reset step.
    retained_frame: Option<dream::ui::draw::UiFrame>,
}

/// Build the scripted UI frame in logical units: a panel, text, a button,
/// an image, a cropped image, and a layer sandwich (a tooltip-layer rect
/// submitted before base-layer rects, so `finish` must sort).
fn build_frame(
    ui: &mut Ui,
    scale: f32,
    screen: (f32, f32),
    texture: &Arc<GpuTexture>,
    text: &str,
) -> UiFrame {
    ui.set_scale(scale);
    ui.begin(UiInput::default(), screen);
    let panel = Rect::new(10.0, 10.0, 160.0, 90.0);
    ui.panel(panel, PanelStyle::default());
    ui.label(
        Rect::new(20.0, 20.0, 140.0, 20.0),
        text,
        LabelStyle::new(FontId(0)).size(16.0).color(Color::WHITE),
    );
    ui.button(
        "ok",
        Rect::new(20.0, 60.0, 80.0, 28.0),
        ButtonStyle::new(FontId(0)),
    );
    ui.end_clip();
    // Image + cropped rounded image (UV crop (0.5,0.25), 32x16, r6 —
    // the shader evaluates the SDF against the flat quad origin, not the
    // crop UV). A half-alpha tint variant proves the tint path survives.
    ui.image(Rect::new(190.0, 10.0, 32.0, 32.0), texture, Color::WHITE);
    // Both crops sit inside the logical frame at every scale (the scale-2
    // frame is 200 logical units wide — rects at x=230 would cull).
    ui.draw_list_mut().image_rounded_uv(
        Rect::new(170.0, 104.0, 28.0, 14.0),
        texture,
        Color::WHITE,
        6.0,
        (0.5, 0.25),
    );
    ui.draw_list_mut().image_rounded_uv(
        Rect::new(170.0, 120.0, 28.0, 14.0),
        texture,
        Color::rgba(255, 255, 255, 128),
        6.0,
        (0.5, 0.25),
    );
    // A raw varying-colour triangle and a trapezoid — neither fits the
    // instanced-quad contract, so the compact-vs-legacy A/B really does
    // exercise the indexed fallback path at the GPU, not only in unit
    // tests. Inside the 200x150 logical frame at scale 2.
    {
        use dream::ui::draw::{CmdTexture, UiVertex};
        let v = |pos: [f32; 2], color: [u8; 4]| UiVertex {
            pos,
            uv: [0.0, 0.0],
            color,
            params: [0.0; 4],
            extra: [0.0; 2],
        };
        ui.draw_list_mut().tris(
            &[
                v([10.0, 140.0], [255, 0, 0, 255]),
                v([50.0, 140.0], [0, 0, 255, 255]),
                v([30.0, 148.0], [0, 255, 0, 255]),
            ],
            &[0, 1, 2],
            CmdTexture::White,
        );
        // Trapezoid: four distinct corners, [a,b,c,a,c,d] winding, but
        // NOT a rectangle — must never become an instance.
        let w = |pos: [f32; 2]| v(pos, [255, 255, 0, 255]);
        ui.draw_list_mut().tris(
            &[
                w([60.0, 140.0]),
                w([100.0, 140.0]),
                w([90.0, 148.0]),
                w([60.0, 148.0]),
            ],
            &[0, 1, 2, 0, 2, 3],
            CmdTexture::White,
        );
    }
    // Interleaved layers: the top strip is submitted first but sorts last.
    ui.draw_list_mut().set_layer(LAYER_TOOLTIP);
    ui.rect(Rect::new(10.0, 0.0, 40.0, 6.0), Color::rgb(0, 200, 0));
    ui.draw_list_mut().set_layer(0);
    ui.rect(Rect::new(10.0, 120.0, 40.0, 6.0), Color::rgb(0, 0, 220));
    // A clip that slices a rect in half.
    ui.draw_list_mut()
        .push_clip(Rect::new(60.0, 120.0, 20.0, 20.0));
    ui.rect(Rect::new(60.0, 120.0, 60.0, 20.0), Color::rgb(200, 0, 0));
    ui.draw_list_mut().pop_clip();
    // The paint primitives, in the free right/bottom regions of the 400x300
    // scale-1 canvas. Every step renders them (scale 2 exercises the
    // physical-size branches too); the `primitives` step probes pixels.
    {
        let draw = ui.draw_list_mut();
        // Linear ramp red -> blue across 120px.
        draw.gradient(
            Rect::new(200.0, 60.0, 120.0, 16.0),
            Color::rgb(255, 0, 0),
            Color::rgb(0, 0, 255),
            dream::ui::draw::GradientAxis::LeftToRight,
        );
        // Soft shadow under a small raised card.
        let card = Rect::new(210.0, 90.0, 70.0, 24.0);
        draw.shadow(card, 6.0, (0.0, 6.0), 2.0, 8.0, Color::rgba(0, 0, 0, 190));
        draw.round_rect(card, 6.0, Color::rgb(90, 90, 100));
        // Radial: white core fading out by radius 24.
        draw.radial_gradient(
            Rect::new(330.0, 8.0, 60.0, 60.0),
            4.0,
            24.0,
            Color::WHITE,
            Color::rgba(0, 0, 0, 0),
        );
        // A green fill under a rounded clip: the clip's corners must stay
        // background while its centre paints.
        draw.push_clip_rounded(Rect::new(300.0, 95.0, 60.0, 60.0), 12.0);
        draw.rect(Rect::new(300.0, 95.0, 60.0, 60.0), Color::rgb(0, 210, 90));
        draw.pop_clip();
        // Rounded rect with a white->black vertical sheen.
        draw.round_rect_gradient(
            Rect::new(200.0, 150.0, 90.0, 40.0),
            8.0,
            [Color::WHITE, Color::WHITE, Color::BLACK, Color::BLACK],
        );
        // Four-corner gradient: red/green/blue/yellow.
        draw.gradient_corners(
            Rect::new(300.0, 170.0, 80.0, 30.0),
            [
                Color::rgb(255, 0, 0),
                Color::rgb(0, 255, 0),
                Color::rgb(0, 0, 255),
                Color::rgb(255, 255, 0),
            ],
        );
        // Quarter arc inside the 200x150 logical frame at scale 2:
        // 40x40 at (120,105), radius 20, 6px band from the top sweeping
        // clockwise to 3 o'clock. Centre (140,125), mid radius 17.
        draw.ring_arc(
            Rect::new(120.0, 105.0, 40.0, 40.0),
            20.0,
            6.0,
            -std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_2,
            Color::rgb(255, 140, 0),
        );
    }
    ui.end()
}

/// Logical rect of each gallery "band" — the sanity check asserts every
/// band draws something (pixels differing from the page background).
/// Gallery probe bands: the solved rect of each keyed node, looked up
/// through `rect_of` after the frame is built — bands track the layout.
fn gallery_bands(ui: &Ui) -> Vec<Rect> {
    const KEYS: &[&str] = &[
        "gal_panel",
        "gal_row1",
        "gal_row2",
        "gal_row3",
        "gal_seg",
        "gal_slider",
        "gal_prog",
        "gal_tf1",
        "gal_tf2",
        "gal_sep",
        "gal_collapsible",
        "gal_badge",
        "gal_card",
        "gal_menu",
        "gal_icons",
        "gal_hxg",
        "gal_find",
        "gal_elev_row",
        "gal_types",
        "gal_close",
        "gal_focus",
    ];
    KEYS.iter()
        .map(|k| {
            ui.rect_of(k)
                .unwrap_or_else(|| panic!("gallery band {k} missing"))
        })
        .collect()
}

/// The gallery scene: every stock widget at rest, drawn with theme
/// builders only — a skin diff is a data diff, so this frame is where a
/// wrong role or token shows up first. 640x480 logical units.
fn build_gallery_frame(ui: &mut Ui, scale: f32, light: bool) -> UiFrame {
    let theme = if light {
        Theme::light()
    } else {
        Theme::dark_theme()
    };
    ui.set_theme(&theme);
    ui.set_scale(scale);
    ui.begin(UiInput::default(), (640.0, 480.0));
    ui.rect(Rect::new(0.0, 0.0, 640.0, 480.0), theme.bg);
    let f = FontId(0);
    use dream::ui::layout::{LayoutAlign, Style};
    use dream::ui::view::Paint;
    use dream::ui::widgets::adapters::{
        Badge, Button, ButtonVariant, Checkbox, IconButton, IconView, Progress, Radio, Segmented,
        Separator, Slider, Switch, Text, TextField, Toggle,
    };
    let mut on = true;
    let mut checked = true;
    let mut selected = 1usize;
    let mut value = 0.6f32;
    let mut filled = String::from("Editable text");
    let mut empty = String::new();
    let mut focus = 0usize;
    let mut open = true;
    let items = [
        MenuItem::new("Open"),
        MenuItem::new("Rename"),
        MenuItem::new("Delete").disabled(),
    ];
    // Focus-ring sample inside the left panel: rings enabled + focused.
    ui.set_focus_rings(true);
    ui.set_focus("focusme");
    ui.set_focus_visible(true);
    ui.layout_root(
        Rect::new(0.0, 0.0, 640.0, 480.0),
        Style::row().pad(16.0).gap(16.0).fill(),
        |ui| {
            // Left column: controls on a panel.
            ui.container(
                Style::column()
                    .pad(16.0)
                    .gap(10.0)
                    .w_px(296.0)
                    .fill_h()
                    .key("gal_panel"),
                Paint::surface(&theme),
                |ui| {
                    ui.add(Text::new("Controls").role(TypeRole::Heading));
                    ui.row(Style::row().gap(8.0).h_px(28.0).key("gal_row1"), |ui| {
                        ui.add_styled(Button::new("ok"), Style::default().w_px(88.0).fill_h());
                        ui.add_styled(
                            Button::new("go").variant(ButtonVariant::Primary),
                            Style::default().w_px(88.0).fill_h(),
                        );
                        ui.add_styled(
                            Button::new("del").variant(ButtonVariant::Danger),
                            Style::default().w_px(72.0).fill_h(),
                        );
                    });
                    ui.row(Style::row().gap(8.0).h_px(28.0).key("gal_row2"), |ui| {
                        ui.add_styled(
                            Button::new("ghost").variant(ButtonVariant::Ghost),
                            Style::default().w_px(88.0).fill_h(),
                        );
                        ui.add_styled(
                            IconButton::new("tools", Icon::Settings),
                            Style::default().w_px(28.0).fill_h(),
                        );
                        ui.add_styled(
                            Toggle::new("enabled", &mut on),
                            Style::default().fill_w().fill_h(),
                        );
                    });
                    ui.row(Style::row().gap(8.0).h_px(28.0).key("gal_row3"), |ui| {
                        ui.add_styled(
                            Switch::new("sw", &mut on),
                            Style::default().w_px(40.0).fill_h(),
                        );
                        ui.add_styled(
                            Checkbox::new("agree", &mut checked),
                            Style::default().fill_w().fill_h(),
                        );
                        ui.add_styled(
                            Radio::new("rad", true),
                            Style::default().w_px(96.0).fill_h(),
                        );
                    });
                    ui.add_styled(
                        Segmented::new("seg", &["Day", "Week", "Month"], &mut selected),
                        Style::default().fill_w().h_px(28.0).key("gal_seg"),
                    );
                    ui.add_styled(
                        Slider::new("vol", &mut value, 0.0..=1.0),
                        Style::default().fill_w().h_px(20.0).key("gal_slider"),
                    );
                    ui.add_styled(
                        Progress::new(0.65),
                        Style::default().fill_w().h_px(8.0).key("gal_prog"),
                    );
                    ui.add_styled(
                        TextField::new("name", &mut filled),
                        Style::default().fill_w().h_px(28.0).key("gal_tf1"),
                    );
                    ui.add_styled(
                        TextField::new("mail", &mut empty).placeholder("Email"),
                        Style::default().fill_w().h_px(28.0).key("gal_tf2"),
                    );
                    ui.add_styled(
                        Separator::horizontal(),
                        Style::default().fill_w().h_px(6.0).key("gal_sep"),
                    );
                    ui.slot(
                        Style::default().fill_w().h_px(84.0).key("gal_collapsible"),
                        move |ui, r| {
                            let (body, open_now) = ui.collapsible("advanced", r, &mut open, f);
                            if open_now {
                                ui.label(
                                    body.inset(8.0),
                                    "Section body",
                                    theme.label(f).size(12.0).color(theme.muted),
                                );
                            }
                            ui.collapsible_end();
                        },
                    );
                    ui.add_styled(
                        Badge::new("3 new"),
                        Style::default().w_px(72.0).h_px(18.0).key("gal_badge"),
                    );
                    ui.spacer();
                    ui.add_styled(
                        Button::new("focusme"),
                        Style::default().w_px(96.0).h_px(20.0).key("gal_focus"),
                    );
                },
            );
            // Right column: a card, an inline menu, icons, and samples.
            ui.column(Style::column().gap(8.0).w_px(296.0).fill_h(), |ui| {
                ui.container(
                    Style::column()
                        .pad(16.0)
                        .gap(4.0)
                        .fill_w()
                        .h_px(100.0)
                        .key("gal_card"),
                    Paint::surface(&theme),
                    |ui| {
                        ui.add(Text::new("Card").role(TypeRole::Heading));
                        ui.add(
                            Text::new("Panels and cards read the same roles.")
                                .role(TypeRole::Caption)
                                .color(theme.muted),
                        );
                    },
                );
                let items = items.clone();
                ui.slot(
                    Style::default().fill_w().h_px(104.0).key("gal_menu"),
                    move |ui, r| {
                        let ms = theme.menu(f);
                        ui.draw_list_mut()
                            .rounded_frame(r, ms.radius, 1.0, theme.g1, theme.border);
                        ui.menu("edit", r, &items, &mut focus, ms);
                    },
                );
                // Icons: the whole set at 16 logical px in the default
                // Ink fit — the caption trio (Minus/Maximize/Close)
                // sits among them at one size.
                ui.slot(
                    Style::default().fill_w().h_px(54.0).key("gal_icons"),
                    move |ui, r| {
                        for (i, icon) in Icon::ALL.iter().enumerate() {
                            let (col, row) = (i % 11, i / 11);
                            ui.icon(
                                Rect::new(
                                    r.x + col as f32 * 25.0,
                                    r.y + row as f32 * 26.0,
                                    16.0,
                                    16.0,
                                ),
                                *icon,
                                16,
                                theme.fg,
                            );
                        }
                    },
                );
                // Optical centring: "Hxg" at five sizes vcentred in
                // outlined rects — the outline frames the box the cap
                // span is centred in.
                ui.row(Style::row().gap(6.0).h_px(40.0).key("gal_hxg"), |ui| {
                    for size in [11.0f32, 13.0, 15.0, 18.0, 24.0] {
                        ui.slot(Style::default().w_px(52.0).fill_h(), move |ui, r| {
                            ui.draw_list_mut().round_ring(r, 4.0, 1.0, theme.border);
                            ui.label(r, "Hxg", theme.label(f).size(size).vcenter(true));
                        });
                    }
                });
                ui.add_styled(
                    Button::new("find").icon(Icon::Settings),
                    Style::default().w_px(120.0).h_px(28.0).key("gal_find"),
                );
                // Elevation 2x2 grid beside the type-scale column.
                ui.row(Style::row().gap(12.0).fill_w().key("gal_elev_row"), |ui| {
                    ui.column(Style::column().gap(10.0).w_px(172.0), |ui| {
                        for pair in [
                            [(Elevation::Flat, "Flat"), (Elevation::Raised, "Raised")],
                            [(Elevation::Overlay, "Overlay"), (Elevation::Modal, "Modal")],
                        ] {
                            ui.row(Style::row().gap(12.0), |ui| {
                                for (e, name) in pair {
                                    ui.slot(
                                        Style::default().w_px(80.0).h_px(56.0),
                                        // Caption inside the card: a Modal
                                        // shadow is wider than any gap below.
                                        move |ui, r| {
                                            ui.elevation_shadow(r, 4.0, e);
                                            ui.draw_list_mut().round_rect(r, 4.0, theme.g1);
                                            ui.label(
                                                Rect::new(
                                                    r.x + 6.0,
                                                    r.max().1 - 16.0,
                                                    r.w - 12.0,
                                                    12.0,
                                                ),
                                                name,
                                                theme.label(f).size(9.0).color(theme.muted),
                                            );
                                        },
                                    );
                                }
                            });
                        }
                    });
                    // Type-scale column beside the elevation grid.
                    ui.column(Style::column().gap(2.0).fill_w().key("gal_types"), |ui| {
                        for (role, name) in [
                            (TypeRole::Heading, "Heading"),
                            (TypeRole::Body, "Body"),
                            (TypeRole::Caption, "Caption"),
                            (TypeRole::Mono, "Mono"),
                        ] {
                            ui.add(Text::new(name).role(role));
                        }
                    });
                });
                ui.spacer();
                ui.add_styled(
                    IconView::new(Icon::Close).size(16).tint(theme.fg),
                    Style::default()
                        .w_px(28.0)
                        .h_px(28.0)
                        .align_self(LayoutAlign::End)
                        .key("gal_close"),
                );
            });
        },
    );
    ui.end()
}

/// Discs `(x, y, radius)` in logical points for the circle-edge check. The
/// fractional centres put each extreme at a different place against the pixel
/// grid, which is where a bounding quad without the anti-aliasing pad clips
/// the outer half of the edge band.
const CIRCLE_PROBES: &[(f32, f32, f32)] = &[
    (40.0, 40.0, 6.0),
    (80.3, 40.3, 6.0),
    (120.5, 40.5, 6.0),
    (160.7, 40.2, 6.0),
    (40.0, 120.0, 15.0),
    (80.3, 120.3, 15.0),
    (120.5, 120.5, 15.0),
    (160.7, 120.2, 15.0),
];

/// Largest allowed difference between a pixel's measured coverage and the
/// analytic coverage: 5/255, the rounding of an 8-bit target plus slack.
const CIRCLE_COVERAGE_TOL: f32 = 0.02;

/// Largest allowed relative error of a disc's total ink against `pi r^2`.
const CIRCLE_INK_TOL: f32 = 0.03;

/// White discs on black, drawn through the public `DrawList` path.
fn build_circle_frame(ui: &mut Ui, scale: f32, screen: (f32, f32)) -> UiFrame {
    ui.set_scale(scale);
    ui.begin(UiInput::default(), screen);
    let (width, height) = screen;
    ui.draw_list_mut()
        .round_rect(Rect::new(0.0, 0.0, width, height), 0.0, Color::BLACK);
    for &(cx, cy, r) in CIRCLE_PROBES {
        ui.draw_list_mut().fill_circle((cx, cy), r, Color::WHITE);
    }
    ui.end()
}

/// The coverage the shader computes for a disc at pixel `(x, y)`:
/// `clamp(0.5 - d, 0, 1)`, with `d` the distance from the pixel centre to the
/// edge, in physical pixels (`ui.wgsl`, the SDF modes).
fn disc_coverage(x: u32, y: u32, centre: (f32, f32), radius: f32) -> f32 {
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let d = (px - centre.0).hypot(py - centre.1) - radius;
    (0.5 - d).clamp(0.0, 1.0)
}

/// Check the discs drawn by [`build_circle_frame`] at UI scale `scale`.
///
/// Every pixel on the horizontal line through the centre row and the vertical
/// line through the centre column must match the analytic coverage within
/// [`CIRCLE_COVERAGE_TOL`]. That includes the four extremes, which are also
/// reported on their own. The disc's total ink, the sum of coverage over its
/// bounding box, must match `pi r^2` within [`CIRCLE_INK_TOL`]. Panics on the
/// first mismatch.
fn check_circle_edges(pixels: &[u8], width: u32, height: u32, scale: f32, name: &str) {
    for &(cx, cy, r) in CIRCLE_PROBES {
        let centre = (cx * scale, cy * scale);
        let radius = r * scale;
        let x_lo = ((centre.0 - radius).floor() as i64 - 2).max(0);
        let x_hi = ((centre.0 + radius).ceil() as i64 + 2).min(i64::from(width) - 1);
        let y_lo = ((centre.1 - radius).floor() as i64 - 2).max(0);
        let y_hi = ((centre.1 + radius).ceil() as i64 + 2).min(i64::from(height) - 1);
        // (analytic, measured) for one pixel; greys only, since the disc is
        // white on black and the blend is gamma (encoded values averaged).
        let sample = |x: u32, y: u32| -> (f32, f32) {
            let px = rgba_at(pixels, width, x as f32, y as f32);
            assert!(
                px[0] == px[1] && px[1] == px[2],
                "[{name}] disc ({cx},{cy}) r={r} pixel ({x},{y}) is not grey: {px:?}"
            );
            let measured = f32::from(px[0]) / 255.0;
            (disc_coverage(x, y, centre, radius), measured)
        };
        let row = centre.1.floor() as u32;
        let col = centre.0.floor() as u32;
        let horizontal: Vec<(f32, f32)> = (x_lo..=x_hi).map(|x| sample(x as u32, row)).collect();
        let vertical: Vec<(f32, f32)> = (y_lo..=y_hi).map(|y| sample(col, y as u32)).collect();

        let mut worst = 0.0f32;
        for (axis, line) in [("horizontal", &horizontal), ("vertical", &vertical)] {
            for (i, &(analytic, measured)) in line.iter().enumerate() {
                let err = (measured - analytic).abs();
                worst = worst.max(err);
                assert!(
                    err <= CIRCLE_COVERAGE_TOL,
                    "[{name}] disc ({cx},{cy}) r={r} {axis} sample {i}: measured {measured:.4}, analytic {analytic:.4}"
                );
            }
        }
        // The first and last sample with any analytic coverage are the four
        // extremes, one per side.
        let extreme = |line: &[(f32, f32)], last: bool| -> (f32, f32) {
            let mut hits = line.iter().filter(|s| s.0 > 0.0);
            let hit = if last { hits.next_back() } else { hits.next() };
            *hit.expect("disc has coverage on this line")
        };
        let (left, right) = (extreme(&horizontal, false), extreme(&horizontal, true));
        let (top, bottom) = (extreme(&vertical, false), extreme(&vertical, true));

        let (mut ink, mut ink_analytic) = (0.0f32, 0.0f32);
        for y in y_lo..=y_hi {
            for x in x_lo..=x_hi {
                let px = rgba_at(pixels, width, x as f32, y as f32);
                ink += f32::from(px[0]) / 255.0;
                ink_analytic += disc_coverage(x as u32, y as u32, centre, radius);
            }
        }
        let expected = std::f32::consts::PI * radius * radius;
        let ink_err = (ink - expected).abs() / expected;
        assert!(
            ink_err <= CIRCLE_INK_TOL,
            "[{name}] disc ({cx},{cy}) r={r} ink {ink:.2} vs pi r^2 {expected:.2} ({:.2}% off)",
            ink_err * 100.0
        );
        println!(
            "[circle-edges] {name} disc ({cx},{cy}) r={r} R={radius:.3}px: max|err|={worst:.4}; \
             extremes L/R/T/B measured {:.3}/{:.3}/{:.3}/{:.3} analytic {:.3}/{:.3}/{:.3}/{:.3}; \
             ink {ink:.2} (analytic {ink_analytic:.2}) vs pi R^2 {expected:.2}, err {:.2}%",
            left.1,
            right.1,
            top.1,
            bottom.1,
            left.0,
            right.0,
            top.0,
            bottom.0,
            ink_err * 100.0
        );
    }
}

/// One pixel from the captured frame (top-left origin).
fn rgba_at(pixels: &[u8], width: u32, x: f32, y: f32) -> [u8; 4] {
    let x = (x.max(0.0) as u32).min(width - 1);
    let i = ((y as u32 * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// Bounding box of pixels darker than the panel background in `region`
/// (glyph ink reads dark here — the label is white text on a dark card,
/// so instead find pixels *not equal* to the card fill).
fn ink_bounds(
    pixels: &[u8],
    width: u32,
    region: (u32, u32, u32, u32),
    bg: [u8; 4],
) -> Option<(u32, u32, u32, u32)> {
    let (x0, y0, x1, y1) = region;
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (u32::MAX, 0, u32::MAX, 0);
    for y in y0..y1 {
        for x in x0..x1 {
            let px = rgba_at(pixels, width, x as f32, y as f32);
            if px[..3] != bg[..3] {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    (min_x != u32::MAX).then_some((min_x, max_x, min_y, max_y))
}

/// Width of the glass check's black and white bands, in logical px.
const GLASS_STRIPE: f32 = 8.0;
/// The glass panel in logical px: x, y, width, height, corner radius.
const GLASS_PANEL: (f32, f32, f32, f32, f32) = (100.0, 60.0, 200.0, 120.0, 12.0);
/// Glass frames rendered with capture off before the read-back frame.
const GLASS_DIRECT_FRAMES: usize = 8;

/// 8 px black and white bands over the whole frame, with one opaque glass
/// panel drawn over them. The backdrop is everything drawn before the panel.
fn build_glass_frame(ui: &mut Ui, scale: f32, screen: (f32, f32)) -> UiFrame {
    ui.set_scale(scale);
    ui.begin(UiInput::default(), screen);
    let (width, height) = screen;
    let bands = (width / GLASS_STRIPE).ceil() as u32;
    for band in 0..bands {
        let x = band as f32 * GLASS_STRIPE;
        let color = if band.is_multiple_of(2) {
            Color::BLACK
        } else {
            Color::WHITE
        };
        ui.draw_list_mut()
            .rect(Rect::new(x, 0.0, GLASS_STRIPE, height), color);
    }
    let (x, y, w, h, radius) = GLASS_PANEL;
    ui.draw_list_mut()
        .backdrop_rounded(Rect::new(x, y, w, h), Color::WHITE, radius);
    ui.end()
}

/// Glass pixel check. Outside the panel each pixel must still be its stripe
/// colour exactly. Inside, the stripes average away: the spread must be far
/// below the stripes' own (about 127), and the mean must sit at the midpoint
/// of black and white. Prints the numbers it measured.
fn check_glass(pixels: &[u8], width: u32, height: u32, scale: f32, name: &str) {
    let red = |x: u32, y: u32| pixels[((y * width + x) * 4) as usize];
    let (x, y, w, h, radius) = GLASS_PANEL;
    let (px0, py0) = (x * scale, y * scale);
    let (px1, py1) = ((x + w) * scale, (y + h) * scale);
    // Stay clear of the rounded corners and the glass's own edge.
    let inset = radius * scale + 2.0;
    let stripe_at = |px: u32| -> u8 {
        let band = (px as f32 / (GLASS_STRIPE * scale)).floor() as u32;
        if band.is_multiple_of(2) { 0 } else { 255 }
    };

    let mut outside = Vec::new();
    let mut mismatches = 0usize;
    let mut first_bad = None;
    for py in 0..height {
        for px in 0..width {
            let in_panel = (px as f32) >= px0 - 2.0
                && (px as f32) < px1 + 2.0
                && (py as f32) >= py0 - 2.0
                && (py as f32) < py1 + 2.0;
            if in_panel {
                continue;
            }
            let base = ((py * width + px) * 4) as usize;
            let rgb = [pixels[base], pixels[base + 1], pixels[base + 2]];
            let expected = stripe_at(px);
            if rgb != [expected; 3] {
                mismatches += 1;
                first_bad.get_or_insert((px, py, rgb, expected));
            }
            outside.push(red(px, py) as f64);
        }
    }

    let mut inner = Vec::new();
    let (ix0, iy0) = ((px0 + inset) as u32, (py0 + inset) as u32);
    let (ix1, iy1) = ((px1 - inset) as u32, (py1 - inset) as u32);
    for py in iy0..iy1 {
        for px in ix0..ix1 {
            inner.push(red(px, py) as f64);
        }
    }

    let stats = |values: &[f64]| -> (f64, f64, f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, var.sqrt(), min, max)
    };
    let (out_mean, out_std, _, _) = stats(&outside);
    let (in_mean, in_std, in_min, in_max) = stats(&inner);
    println!(
        "[glass] step={name} outside: n={} mean={out_mean:.2} std={out_std:.2} mismatches={mismatches} inside: n={} mean={in_mean:.2} std={in_std:.2} min={in_min} max={in_max}",
        outside.len(),
        inner.len()
    );
    if let Some((px, py, rgb, expected)) = first_bad {
        panic!(
            "[{name}] outside the glass pixel ({px},{py}) = {rgb:?}, expected stripe {expected}"
        );
    }
    assert!(
        in_std * 8.0 < out_std,
        "[{name}] glass spread {in_std:.2} is not far below the stripes' {out_std:.2}"
    );
    assert!(
        (in_mean - 127.5).abs() <= 6.0,
        "[{name}] glass mean {in_mean:.2} is not at the black/white midpoint 127.5"
    );
}

/// Point the swapchain at the window's current size, if it has moved on.
fn resize_to_window(renderer: &mut Renderer, window: Option<&Window>, label: &str) {
    let Some(window) = window else {
        panic!("{label}: swapchain went stale with no window to resize from");
    };
    let size = window.inner_size();
    let (width, height) = renderer.display_size();
    eprintln!(
        "[check] {label}: swapchain {width}x{height} stale, rebuilding at {}x{}",
        size.width, size.height
    );
    renderer.resize(size);
}

/// Render `frame`, rebuilding the swapchain if the surface reported that it
/// went stale. A resize can land after recording started — X11 applies one
/// asynchronously, and a hidden window gets no chance to react first — so a
/// present at the old extent reports `Outdated`. One retry at the window's
/// actual size settles it; any other error, or a second stale report, is real.
fn render_settled<F>(
    renderer: &mut Renderer,
    window: Option<&Window>,
    label: &str,
    draw: F,
    frame: &UiFrame,
) -> usize
where
    F: for<'a> Fn(&mut dream::gfx::DrawCtx<'a>) -> usize + Copy,
{
    let camera = Camera::default();
    match renderer.render_ui(&camera, None, draw, Some(frame)) {
        Ok(visible) => visible,
        Err(dream::gfx::SurfaceError::Outdated) => {
            resize_to_window(renderer, window, label);
            renderer
                .render_ui(&camera, None, draw, Some(frame))
                .unwrap_or_else(|err| panic!("{label}: {err:?} after rebuilding the swapchain"))
        }
        Err(err) => panic!("{label}: {err:?}"),
    }
}

/// [`Renderer::render_ui_only`] with the same stale-swapchain retry.
fn render_only_settled(
    renderer: &mut Renderer,
    window: Option<&Window>,
    label: &str,
    frame: &UiFrame,
) {
    match renderer.render_ui_only(Some(frame)) {
        Ok(()) => {}
        Err(dream::gfx::SurfaceError::Outdated) => {
            resize_to_window(renderer, window, label);
            renderer
                .render_ui_only(Some(frame))
                .unwrap_or_else(|err| panic!("{label}: {err:?} after rebuilding the swapchain"));
        }
        Err(err) => panic!("{label}: {err:?}"),
    }
}

impl Check {
    /// Run one scripted step: build, render, capture, assert, screenshot.
    fn run_step(&mut self, event_loop: &ActiveEventLoop) {
        let step = &STEPS[self.step];
        let mut blocking_before: u64;
        let renderer = self.renderer.as_mut().expect("renderer");
        let ui = self.ui.as_mut().expect("ui");
        let texture = self.texture.as_ref().expect("texture");
        if let Some((w, h)) = step.resize_to
            && self.resize_pending.is_none()
        {
            // Real resize: ask the window, then park this step until the
            // actual `Resized` event lands — no racing a capture against a
            // swapchain still being rebuilt.
            self.resize_pending = Some((w, h));
            let window = self.window.as_ref().expect("window");
            let _ = window.request_inner_size(PhysicalSize::new(w, h));
            window.request_redraw();
            return;
        }
        if self.resize_pending.is_some() {
            // The resize arrived and `resize` already ran on the event.
            self.resize_pending = None;
        }
        // An unsolicited or late resize (a window manager, or X11 applying
        // the request after this step began) must not leave the frame built
        // for an extent the surface no longer has: rebuild before the frame
        // is built, so every assert below probes the pixels just presented.
        if let Some(window) = self.window.as_deref() {
            let size = window.inner_size();
            if size.width > 0
                && size.height > 0
                && (size.width, size.height) != renderer.display_size()
            {
                resize_to_window(renderer, Some(window), "step start");
            }
        }
        let display = renderer.display_size();
        let logical = (display.0 as f32 / step.scale, display.1 as f32 / step.scale);
        match step.extra {
            Some(Extra::Unicode) => {
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render unicode frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::DropFrame) => {
                // A custom texture (same magenta as the check texel, so
                // the image asserts still hold) whose last `Arc` dies the
                // instant `render_ui` returns — the slot's retained list
                // must keep the Vulkan image alive past the queued draws.
                let temporary = Arc::new(GpuTexture::from_rgba8(
                    renderer.context(),
                    "drop me",
                    &[255u8, 0, 255, 255]
                        .into_iter()
                        .cycle()
                        .take(8 * 8 * 4)
                        .collect::<Vec<u8>>(),
                    8,
                    8,
                    TextureOptions::default(),
                ));
                let mut frame = build_frame(ui, step.scale, logical, texture, step.text);
                for cmd in &mut frame.cmds {
                    if let dream::ui::draw::CmdTexture::Custom(_) = cmd.texture {
                        cmd.texture = dream::ui::draw::CmdTexture::Custom(Arc::clone(&temporary));
                    }
                }
                // The deliberate one-shot texture upload above already
                // counted; measure only the render itself.
                blocking_before = renderer.context().blocking_submissions();
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render drop frame",
                    |_| 0,
                    &frame,
                );
                drop(frame);
                drop(temporary);
            }
            Some(Extra::LegacyUpload) => {
                blocking_before = renderer.context().blocking_submissions();
                // A raw `UiFrame`: `atlas_snapshot = None` forces the legacy
                // dirty-rect path (a snapshot would take precedence and the
                // legacy data would be ignored). RGBA page 0, a 1x1 magenta
                // texel, and a quad sampling its centre.
                let page_size = AtlasConfig::default().page_size;
                let mut draw = dream::ui::draw::DrawList::new();
                let texel_uv = {
                    let s = page_size as f32;
                    [(4.5 / s, 4.5 / s, 4.5 / s, 4.5 / s)]
                };
                draw.quad(
                    Rect::new(190.0, 10.0, 32.0, 32.0),
                    Some([texel_uv[0].0, texel_uv[0].1, texel_uv[0].2, texel_uv[0].3]),
                    Color::WHITE,
                    dream::ui::draw::CmdTexture::AtlasPage(0),
                );
                // A second quad sampling untouched padding (uv 0.9) —
                // the transparent page shows the background through.
                draw.quad(
                    Rect::new(230.0, 10.0, 32.0, 32.0),
                    Some([0.9, 0.9, 0.9, 0.9]),
                    Color::WHITE,
                    dream::ui::draw::CmdTexture::AtlasPage(0),
                );
                let mut frame = draw.finish(logical);
                frame.atlas_updates.push(dream::ui::draw::AtlasUpdate {
                    rect: dream::ui::atlas::DirtyRect {
                        page: 0,
                        x: 4,
                        y: 4,
                        width: 1,
                        height: 1,
                    },
                    pixels: vec![255, 0, 255, 255],
                });
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render raw legacy upload",
                    |_| 0,
                    &frame,
                );
            }
            Some(Extra::Orient) => {
                blocking_before = renderer.context().blocking_submissions();
                // The label "Fp g" — orientation asserts below compare
                // ink structure inside the glyph region.
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render orientation frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::AtlasReset) => {
                blocking_before = renderer.context().blocking_submissions();
                // Frame A drawn once, RETAINED while a second render
                // introduces new glyphs (frame B) — then the atlas
                // resets (overflow path or explicit), bumping the epoch.
                // The retained frame's old pages must still be valid.
                let frame_a = build_frame(ui, step.scale, logical, texture, "BedFrame OK");
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render frame A",
                    |_| 0,
                    &frame_a,
                );
                self.retained_frame = Some(frame_a);
                // Frame B introduces unseen glyphs and renders over the
                // retained page textures.
                let frame_b = build_frame(ui, step.scale, logical, texture, "Zx9@#~");
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render frame B",
                    |_| 0,
                    &frame_b,
                );
                ui.recycle_frame(frame_b);
                // Reset through the Ui path: new epoch, both glyph maps
                // cleared, layout cache invalidated — then re-render the
                // retained frame: its old snapshot must still upload
                // (epoch change) and draw correctly.
                ui.cache_mut().reset_atlas();
                let frame_a = self.retained_frame.take().expect("retained frame A");
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "re-render retained frame A after reset",
                    |_| 0,
                    &frame_a,
                );
            }
            Some(Extra::CompactVsLegacy) => {
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render compact frame",
                    |_| 0,
                    &frame,
                );
                assert_eq!(
                    renderer.context().blocking_submissions(),
                    blocking_before,
                    "compact render must not block"
                );
                let (px, w, h) = renderer.capture_rgba8();
                self.compact_capture = Some((px, w, h));
                let before_legacy = renderer.context().blocking_submissions();
                renderer.set_ui_instancing(false);
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render legacy-geometry frame",
                    |_| 0,
                    &frame,
                );
                assert_eq!(
                    renderer.context().blocking_submissions(),
                    before_legacy,
                    "legacy-geometry render must not block"
                );
                renderer.set_ui_instancing(true);
                ui.recycle_frame(frame);
                // The comparison capture above legitimately costs one
                // readback submission — the outer assertion measures
                // from after it.
                blocking_before = renderer.context().blocking_submissions();
            }
            Some(Extra::UiOnlyBeforeScene) => {
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_only_settled(
                    renderer,
                    self.window.as_deref(),
                    "render ui-only frame",
                    &frame,
                );
                assert!(
                    !renderer.has_temporal(),
                    "AaMode::None builds no temporal upscaler"
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::UiOnlyAfterScene) => {
                let warm = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "warm scene frame",
                    |draw| {
                        let _ = draw.raw();
                        0
                    },
                    &warm,
                );
                ui.recycle_frame(warm);
                let warmed = renderer.context().blocking_submissions();
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_only_settled(
                    renderer,
                    self.window.as_deref(),
                    "ui-only after scene",
                    &frame,
                );
                assert_eq!(
                    renderer.context().blocking_submissions(),
                    warmed,
                    "ui-only render must not block"
                );
                assert!(
                    !renderer.history_valid(),
                    "ui-only frame must invalidate scene history"
                );
                let (ui_px, uw, _uh) = renderer.capture_rgba8();
                let corner = rgba_at(&ui_px, uw, 4.0, 4.0);
                assert_eq!(
                    corner,
                    [0, 0, 0, 255],
                    "ui-only shows opaque black outside the UI, got {corner:?}"
                );
                let after_ui_capture = renderer.context().blocking_submissions();
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "scene resume frame",
                    |draw| {
                        let _ = draw.raw();
                        0
                    },
                    &frame,
                );
                assert_eq!(
                    renderer.context().blocking_submissions(),
                    after_ui_capture,
                    "scene resume must not block"
                );
                assert!(
                    renderer.history_valid(),
                    "a scene frame re-establishes temporal history"
                );
                let opaque_frame = |ui: &mut Ui| {
                    ui.set_scale(step.scale);
                    ui.begin(UiInput::default(), logical);
                    ui.rect(
                        Rect::new(0.0, 0.0, logical.0, logical.1),
                        Color::rgb(20, 24, 30),
                    );
                    ui.label(
                        Rect::new(20.0, 20.0, 140.0, 20.0),
                        step.text,
                        LabelStyle::new(FontId(0)).size(16.0).color(Color::WHITE),
                    );
                    ui.end()
                };
                let frame = opaque_frame(ui);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "legacy opaque frame",
                    |draw| {
                        let _ = draw.raw();
                        0
                    },
                    &frame,
                );
                let (legacy_px, lw, lh) = renderer.capture_rgba8();
                render_only_settled(
                    renderer,
                    self.window.as_deref(),
                    "ui-only opaque frame",
                    &frame,
                );
                let (only_px, ow, oh) = renderer.capture_rgba8();
                assert_eq!((lw, lh), (ow, oh));
                let worst = legacy_px
                    .iter()
                    .zip(&only_px)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap_or(0);
                assert!(
                    worst <= 1,
                    "opaque UI: legacy vs ui-only differ by {worst} LSB"
                );
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_only_settled(
                    renderer,
                    self.window.as_deref(),
                    "final ui-only frame",
                    &frame,
                );
                ui.recycle_frame(frame);
                blocking_before = renderer.context().blocking_submissions();
            }
            Some(Extra::Glass) => {
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_glass_frame(ui, step.scale, logical);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render glass frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::GlassDirect) => {
                // Capture off: frames compose straight into the swapchain and
                // cannot be read back. Render several glass frames that way
                // (they must not fault), then capture one frame with capture on.
                renderer.set_frame_capture(false);
                for _ in 0..GLASS_DIRECT_FRAMES {
                    let frame = build_glass_frame(ui, step.scale, logical);
                    render_settled(
                        renderer,
                        self.window.as_deref(),
                        "render direct glass frame",
                        |_| 0,
                        &frame,
                    );
                    ui.recycle_frame(frame);
                }
                renderer.set_frame_capture(true);
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_glass_frame(ui, step.scale, logical);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render glass frame after direct frames",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::CircleEdges) => {
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_circle_frame(ui, step.scale, logical);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render circle frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::Primitives) => {
                // The standard scene already carries the paint primitives
                // (see `build_frame`); this step only changes which pixels
                // get probed after the capture.
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render primitives frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::Gallery(light)) => {
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_gallery_frame(ui, step.scale, light);
                let _ = &frame;
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render gallery frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            Some(Extra::GalleryDevtools) => {
                blocking_before = renderer.context().blocking_submissions();
                ui.set_devtools(dream::ui::devtools::DevtoolsMode::Outlines);
                let frame = build_gallery_frame(ui, step.scale, false);
                ui.set_devtools(dream::ui::devtools::DevtoolsMode::Off);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render devtools frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
            None => {
                blocking_before = renderer.context().blocking_submissions();
                let frame = build_frame(ui, step.scale, logical, texture, step.text);
                render_settled(
                    renderer,
                    self.window.as_deref(),
                    "render ui frame",
                    |_| 0,
                    &frame,
                );
                ui.recycle_frame(frame);
            }
        }
        // Raw diagnostics (lead instrumentation — read exactly as
        // emitted; no derived totals are produced here).
        for (name, bytes) in renderer.ui_allocation_sizes() {
            println!(
                "[ui-resource] step={} name={name:?} bytes={bytes}",
                step.name
            );
        }
        let (v, i, q) = renderer.ui_last_geometry_upload_sizes();
        println!(
            "[ui-upload] step={} vertex_bytes={v} index_bytes={i} instance_bytes={q}",
            step.name
        );
        assert_eq!(
            renderer.context().blocking_submissions(),
            blocking_before,
            "a UI frame must never pay a blocking submission"
        );
        let unicode_extra = step.extra == Some(Extra::Unicode);
        // CJK/emoji shape through the native (CoreText) text path, which
        // resolves system script faces; off macOS no role faces exist and
        // clusters are hidden by design (never boxes) — so the expected
        // result differs per platform.
        let unicode_native = unicode_extra && cfg!(target_os = "macos");
        let unicode_hidden = unicode_extra && !cfg!(target_os = "macos");
        let (pixels, width, height) = renderer.capture_rgba8();
        let dir = std::path::Path::new("target/bedframe-check");
        let path = dir.join(format!("{}.png", step.name));
        std::fs::create_dir_all(path.parent().expect("png parent dir")).expect("create output dir");
        dream::png::write(&path, width, height, &pixels).expect("write png");
        assert_eq!(
            (width, height),
            (display.0, display.1),
            "captured frame matches the display size"
        );
        let s = step.scale;
        // Panel corner (inside, logical 14,14) is the panel fill; a point
        // just outside the panel is not. The raw legacy-upload frame and
        // the gallery have no standard scene — only dedicated asserts.
        let standard = !matches!(
            step.extra,
            Some(Extra::LegacyUpload)
                | Some(Extra::Gallery(_))
                | Some(Extra::GalleryDevtools)
                | Some(Extra::CircleEdges)
                | Some(Extra::Glass)
                | Some(Extra::GlassDirect)
        );
        let theme = Theme::dark_theme();
        let panel_style = PanelStyle::default();
        let (mut ink_w, mut ink_h) = (0.0f32, 0.0f32);
        if standard {
            let inside = rgba_at(&pixels, width, 14.0 * s, 14.0 * s);
            let outside = rgba_at(&pixels, width, 4.0 * s, 4.0 * s);
            assert_eq!(
                &inside[..3],
                &[
                    panel_style.background.r,
                    panel_style.background.g,
                    panel_style.background.b
                ],
                "[{}] panel fill at its corner, got {inside:?}",
                step.name
            );
            assert_ne!(
                inside, outside,
                "[{}] panel edge distinguishes inside from outside",
                step.name
            );
            if matches!(
                step.extra,
                Some(Extra::UiOnlyBeforeScene) | Some(Extra::UiOnlyAfterScene)
            ) {
                assert_eq!(
                    outside,
                    [0, 0, 0, 255],
                    "[{}] ui-only background outside the UI is opaque black, got {outside:?}",
                    step.name
                );
            }
            // Glyph ink in the text region: pixels that are not the card fill.
            let bounds = ink_bounds(
                &pixels,
                width,
                (
                    (20.0 * s) as u32,
                    (20.0 * s) as u32,
                    (160.0 * s) as u32,
                    (40.0 * s) as u32,
                ),
                [
                    panel_style.background.r,
                    panel_style.background.g,
                    panel_style.background.b,
                    panel_style.background.a,
                ],
            );
            if unicode_native {
                // A colour emoji's texels are non-grey (r,g,b not all equal)
                // and non-transparent.
                let bg = [
                    panel_style.background.r,
                    panel_style.background.g,
                    panel_style.background.b,
                ];
                let mut coloured = false;
                for y in (20.0 * s) as u32..(40.0 * s) as u32 {
                    for x in (20.0 * s) as u32..(160.0 * s) as u32 {
                        let px = rgba_at(&pixels, width, x as f32, y as f32);
                        if px[..3] != bg && !(px[0] == px[1] && px[1] == px[2]) {
                            coloured = true;
                        }
                    }
                }
                assert!(
                    coloured,
                    "[{}] no coloured emoji texels in the text region",
                    step.name
                );
            }
            if unicode_hidden {
                // No script faces on this platform: every cluster hides
                // (never a fallback box), so the text region draws
                // nothing at all.
                assert!(
                    bounds.is_none(),
                    "[{}] unresolvable script must hide, not draw boxes",
                    step.name
                );
            } else {
                let (x0, x1, y0, y1) = bounds
                    .unwrap_or_else(|| panic!("[{}] no glyph ink in the text region", step.name));
                ink_w = (x1 - x0) as f32;
                ink_h = (y1 - y0) as f32;
                // 16 logical px of text rastered at 16*scale: the ink is
                // roughly 16*scale tall (cap height ~0.7em) — a stretched
                // low-res raster would show half that at scale 2.
                assert!(
                    ink_h > 8.0 * s,
                    "[{}] glyph ink {ink_w}x{ink_h} is not ~{:.0}px tall",
                    step.name,
                    16.0 * s
                );
            }
            // The solid image texel at its centre is the uploaded colour.
            let texel = rgba_at(&pixels, width, 206.0 * s, 26.0 * s);
            assert_eq!(
                &texel[..3],
                &[255, 0, 255],
                "[{}] image texel, got {texel:?}",
                step.name
            );
            // The sorted top strip drew on top at layer 3000.
            let strip = rgba_at(&pixels, width, 30.0 * s, 3.0 * s);
            assert_eq!(
                &strip[..3],
                &[0, 200, 0],
                "[{}] interleaved layer sort, got {strip:?}",
                step.name
            );
            // The clipped rect stops at its clip edge (physical 80*s..120*s).
            let clipped_out = rgba_at(&pixels, width, 100.0 * s, 130.0 * s);
            let clipped_in = rgba_at(&pixels, width, 70.0 * s, 130.0 * s);
            assert_eq!(
                &clipped_in[..3],
                &[200, 0, 0],
                "[{}] clipped draw inside the clip, got {clipped_in:?}",
                step.name
            );
            assert_ne!(
                &clipped_out[..3],
                &[200, 0, 0],
                "[{}] clipped draw outside the clip, got {clipped_out:?}",
                step.name
            );
            // Crop-path SDF: a corner of the rounded 32x16 crop is outside
            // the radius (shows through) while its centre samples magenta.
            // The UV-cropped half must not pull SDF coordinates from uv.
            let crop_corner = rgba_at(&pixels, width, 171.0 * s, 105.0 * s);
            let outside_bg = rgba_at(&pixels, width, 4.0 * s, 4.0 * s);
            let crop_center = rgba_at(&pixels, width, 184.0 * s, 111.0 * s);
            assert_eq!(
                &crop_center[..3],
                &[255, 0, 255],
                "[{}] cropped rounded image centre, got {crop_center:?}",
                step.name
            );
            // Corner pixels: inside the rect but outside the r=6 corner —
            // the SDF must clear them (bg through), not the raw quad.
            assert!(
                crop_corner[..3]
                    .iter()
                    .zip(outside_bg[..3].iter())
                    .all(|(a, b)| a.abs_diff(*b) <= 4)
                    || crop_corner[3] <= outside_bg[3].saturating_add(4),
                "[{}] rounded crop corner is not cleared, got {crop_corner:?} bg {outside_bg:?}",
                step.name
            );
            // Half-alpha tint: centre of the tinted crop blends magenta over
            // the background at ~50% — not full magenta, not transparent.
            // The default UI blend is `UiBlend::Gamma`: sRGB-encoded values
            // are averaged directly, as browsers do, so the expected red is
            // (255 + bg) / 2. (Linear-light blending would give ~188 here.)
            let tinted = rgba_at(&pixels, width, 184.0 * s, 127.0 * s);
            let blend = |a: u8, b: u8| -> bool { a.abs_diff(b) <= 24 };
            let expected_r = ((255.0 + outside_bg[0] as f32) * 0.5).round() as u8;
            assert!(
                blend(tinted[0], expected_r),
                "[{}] tinted crop ~half magenta, got {tinted:?} bg {outside_bg:?}",
                step.name
            );
        }
        match step.extra {
            Some(Extra::CircleEdges) => check_circle_edges(&pixels, width, height, s, step.name),
            Some(Extra::Glass) | Some(Extra::GlassDirect) => {
                check_glass(&pixels, width, height, s, step.name)
            }
            Some(Extra::Primitives) => {
                // Pixel probes over the paint primitives. Background is
                // the clear colour sampled from an empty corner.
                let name = step.name;
                let bg = rgba_at(&pixels, width, 396.0 * s, 296.0 * s);
                let near = |a: [u8; 4], b: [u8; 4]| {
                    a[..3]
                        .iter()
                        .zip(b[..3].iter())
                        .all(|(x, y)| x.abs_diff(*y) <= 6)
                };
                // 1. Linear ramp: red end, blue end, and a purple middle.
                let left = rgba_at(&pixels, width, 210.0 * s, 68.0 * s);
                let mid = rgba_at(&pixels, width, 260.0 * s, 68.0 * s);
                let right = rgba_at(&pixels, width, 310.0 * s, 68.0 * s);
                assert!(
                    left[0] > left[2] + 120,
                    "[{name}] gradient red end, got {left:?}"
                );
                assert!(
                    right[2] > right[0] + 120,
                    "[{name}] gradient blue end, got {right:?}"
                );
                assert!(
                    (mid[0] as i32 - mid[2] as i32).abs() <= 24,
                    "[{name}] gradient middle blends, got {mid:?}"
                );
                // 2. Radial: white core, faded ring, untouched corner.
                let core = rgba_at(&pixels, width, 360.0 * s, 38.0 * s);
                assert!(
                    near(core, Color::WHITE.to_array()),
                    "[{name}] radial core is the inner colour, got {core:?}"
                );
                // 8px out: t = 0.2 on the way to transparent — the blend
                // must lift toward white, not sink toward black.
                let ring = rgba_at(&pixels, width, 360.0 * s, 30.0 * s);
                assert!(
                    ring[0] > bg[0] + 40,
                    "[{name}] radial ring blends outward, got {ring:?} bg {bg:?}"
                );
                let outer = rgba_at(&pixels, width, 333.0 * s, 11.0 * s);
                assert!(
                    near(outer, bg),
                    "[{name}] past the outer radius stays background, got {outer:?}"
                );
                // 3. Shadow: solid under the card, a partial fade below the
                // silhouette edge, background well beyond it — and the card
                // itself paints over the shadow's core.
                let card = rgba_at(&pixels, width, 245.0 * s, 102.0 * s);
                assert!(
                    near(card, Color::rgb(90, 90, 100).to_array()),
                    "[{name}] card fill over its shadow, got {card:?}"
                );
                let under = rgba_at(&pixels, width, 245.0 * s, 118.0 * s);
                assert!(
                    (under[0] as i32) < bg[0] as i32 - 60,
                    "[{name}] shadow core darkens the background, got {under:?} bg {bg:?}"
                );
                let fade = rgba_at(&pixels, width, 245.0 * s, 124.0 * s);
                assert!(
                    fade[0] < bg[0] - 8 && fade[0] > bg[0].saturating_sub(70),
                    "[{name}] shadow fades between core and background, got {fade:?} bg {bg:?}"
                );
                // y=135: past the shadow quad (ends ~127), above the
                // gradient fill that starts at y=150.
                let far = rgba_at(&pixels, width, 245.0 * s, 135.0 * s);
                assert!(
                    near(far, bg),
                    "[{name}] no shadow beyond its quad, got {far:?}"
                );
                // 4. Rounded clip: green centre, background in the carved
                // corner (inside the square scissor, outside the rounding).
                let clip_in = rgba_at(&pixels, width, 330.0 * s, 125.0 * s);
                assert!(
                    near(clip_in, Color::rgb(0, 210, 90).to_array()),
                    "[{name}] clipped fill at its centre, got {clip_in:?}"
                );
                let clip_corner = rgba_at(&pixels, width, 302.0 * s, 97.0 * s);
                assert!(
                    near(clip_corner, bg),
                    "[{name}] rounded clip carves the corner, got {clip_corner:?} bg {bg:?}"
                );
                // 5. Rounded gradient fill: bright top, dark bottom, and a
                // carved corner outside the radius.
                let sheen_top = rgba_at(&pixels, width, 245.0 * s, 156.0 * s);
                let sheen_bottom = rgba_at(&pixels, width, 245.0 * s, 184.0 * s);
                assert!(
                    sheen_top[0] > 190 && sheen_bottom[0] < 90,
                    "[{name}] rounded gradient ramps top-to-bottom, got {sheen_top:?} / {sheen_bottom:?}"
                );
                let sheen_corner = rgba_at(&pixels, width, 201.0 * s, 151.0 * s);
                assert!(
                    near(sheen_corner, bg),
                    "[{name}] rounded gradient masks its corner, got {sheen_corner:?}"
                );
                // 6. Four-corner gradient: red TL, green TR, blue BR.
                let c_tl = rgba_at(&pixels, width, 304.0 * s, 174.0 * s);
                let c_tr = rgba_at(&pixels, width, 376.0 * s, 174.0 * s);
                let c_br = rgba_at(&pixels, width, 376.0 * s, 196.0 * s);
                assert!(
                    c_tl[0] > 200 && c_tl[2] < 80,
                    "[{name}] corner gradient top-left, got {c_tl:?}"
                );
                assert!(
                    c_tr[1] > 150 && c_tr[0] < 120,
                    "[{name}] corner gradient top-right, got {c_tr:?}"
                );
                assert!(
                    c_br[2] > 150,
                    "[{name}] corner gradient bottom-right, got {c_br:?}"
                );
                // 7. Quarter arc: the band paints mid-sweep and at the
                // start cap, and the region past the sweep stays
                // background. Centre (140,125), mid radius 17.
                let arc_mid = rgba_at(&pixels, width, 152.0 * s, 113.0 * s);
                assert!(
                    near(arc_mid, Color::rgb(255, 140, 0).to_array()),
                    "[{name}] arc band paints mid-sweep, got {arc_mid:?}"
                );
                let arc_start = rgba_at(&pixels, width, 140.0 * s, 108.0 * s);
                assert!(
                    near(arc_start, Color::rgb(255, 140, 0).to_array()),
                    "[{name}] arc band covers the start cap, got {arc_start:?}"
                );
                let arc_past = rgba_at(&pixels, width, 140.0 * s, 142.0 * s);
                assert!(
                    near(arc_past, bg),
                    "[{name}] past the sweep stays background, got {arc_past:?}"
                );
                println!(
                    "[check] {name}: shadow, gradients, radial and rounded clip verified on the GPU"
                );
            }
            Some(Extra::Orient) => {
                // Column-level ink structure of "Fp g": F's top bar is
                // wider than its bottom (stem only); p's descender dips
                // below the shared baseline.
                let bg = [
                    panel_style.background.r,
                    panel_style.background.g,
                    panel_style.background.b,
                    panel_style.background.a,
                ];
                let mut col_top = vec![None::<u32>; 170];
                let mut col_bottom = vec![None::<u32>; 170];
                for x in 20u32..170 {
                    for y in 20u32..45 {
                        let px = rgba_at(&pixels, width, x as f32, y as f32);
                        if px[..3] != bg[..3] {
                            let xi = x as usize;
                            if col_top[xi].is_none() {
                                col_top[xi] = Some(y);
                            }
                            col_bottom[xi] = Some(y);
                        }
                    }
                }
                // Split clusters on fully-empty columns.
                let mut clusters: Vec<(u32, u32)> = Vec::new();
                let mut start = None;
                for x in 20u32..170 {
                    if col_top[x as usize].is_some() {
                        start.get_or_insert(x);
                    } else if let Some(s0) = start.take() {
                        clusters.push((s0, x));
                    }
                }
                if let Some(s0) = start.take() {
                    clusters.push((s0, 170));
                }
                assert!(
                    clusters.len() >= 2,
                    "expected at least two glyph clusters in 'Fp g', got {clusters:?}"
                );
                let (fx0, fx1) = clusters[0];
                let f_top: Vec<u32> = col_top[fx0 as usize..fx1 as usize]
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                let f_bottom: Vec<u32> = col_bottom[fx0 as usize..fx1 as usize]
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                let f_y0 = *f_top.iter().min().expect("F ink");
                let f_y1 = *f_bottom.iter().max().expect("F ink");
                let f_h = f_y1 - f_y0 + 1;
                // Top-third ink span vs bottom-third: F's top bar spans
                // nearly the full bbox; the bottom holds only the stem.
                let band = (f_h / 3).max(1);
                let top_ink = (fx0..fx1)
                    .filter(|&x| col_top[x as usize].is_some_and(|t| t <= f_y0 + band))
                    .count();
                let bottom_start = (fx0..fx1)
                    .filter(|&x| {
                        col_bottom[x as usize].is_some_and(|b| b >= f_y1.saturating_sub(band))
                    })
                    .count();
                assert!(
                    top_ink > bottom_start,
                    "F's top bar ({top_ink} columns) must exceed its bottom ({bottom_start})"
                );
                // p's cluster descends below F's baseline bottom.
                if let Some(&(px0, px1)) = clusters.get(1) {
                    let p_y1 = col_bottom[px0 as usize..px1 as usize]
                        .iter()
                        .flatten()
                        .copied()
                        .max()
                        .expect("p ink");
                    assert!(
                        p_y1 > f_y1,
                        "p's descender ({p_y1}) must dip below F's baseline ({f_y1})"
                    );
                }
            }
            Some(Extra::LegacyUpload) => {
                // The quad samples only the written texel: magenta. An
                // untouched region of the page stays transparent — the
                // second quad (uv 0.9) shows the scene background through.
                let bg = rgba_at(&pixels, width, 4.0 * s, 4.0 * s);
                let untouched = rgba_at(&pixels, width, 246.0 * s, 26.0 * s);
                assert_eq!(
                    untouched, bg,
                    "untouched page padding stays transparent, got {untouched:?}"
                );
            }
            Some(Extra::GalleryDevtools) => {
                // Sanity only: something was drawn (the page bg).
                let bg = rgba_at(&pixels, width, 4.0 * s, 4.0 * s);
                assert_ne!(
                    bg,
                    [0, 0, 0, 0],
                    "[{}] devtools gallery frame is blank",
                    step.name
                );
            }
            Some(Extra::Gallery(_)) => {
                let bg = rgba_at(&pixels, width, 4.0 * s, 4.0 * s);
                assert!(
                    bg.iter()
                        .zip(ui.theme().bg.to_array())
                        .all(|(a, b)| a.abs_diff(b) <= 1),
                    "[{}] gallery page background differs from the theme: {bg:?}",
                    step.name
                );
                // Every widget band drew something onto whatever surface
                // sits behind it (sampled just left of the band — the page
                // or the panel), so a blank widget on a panel still fails.
                let name = step.name;
                let px = |r: Rect| {
                    (
                        (r.x * s) as u32,
                        (r.y * s) as u32,
                        ((r.x + r.w) * s) as u32,
                        ((r.y + r.h) * s) as u32,
                    )
                };
                for (i, band) in gallery_bands(ui).iter().enumerate() {
                    let behind = rgba_at(
                        &pixels,
                        width,
                        (band.x - 2.0) * s,
                        (band.y + band.h * 0.5) * s,
                    );
                    assert!(
                        ink_bounds(&pixels, width, px(*band), behind).is_some(),
                        "[{name}] gallery band {i} {band:?} drew nothing",
                    );
                }
                // Segmented: every cell carries label ink over the track,
                // and the selected pill differs from the track.
                let seg = ui.rect_of("gal_seg").expect("gal_seg rect");
                let track = rgba_at(&pixels, width, (seg.x + 2.0) * s, (seg.y + 2.0) * s);
                let cell_w = seg.w / 3.0;
                for c in 0..3 {
                    let cell = Rect::new(
                        seg.x + cell_w * c as f32 + 4.0,
                        seg.y + 4.0,
                        cell_w - 8.0,
                        seg.h - 8.0,
                    );
                    let corner = rgba_at(&pixels, width, cell.x * s, cell.y * s);
                    assert!(
                        ink_bounds(&pixels, width, px(cell), corner).is_some(),
                        "[{name}] segmented cell {c} has no label ink",
                    );
                    if c == 1 {
                        assert_ne!(corner, track, "[{name}] segmented selected pill invisible");
                    }
                }
                // Text fields: border differs from the panel, and value /
                // placeholder text puts ink inside the field.
                for key in ["gal_tf1", "gal_tf2"] {
                    let tf = ui.rect_of(key).expect("text field rect");
                    let panel = rgba_at(&pixels, width, (tf.x - 2.0) * s, (tf.y + tf.h * 0.5) * s);
                    let edge = rgba_at(&pixels, width, tf.x * s + 0.5, (tf.y + tf.h * 0.5) * s);
                    assert_ne!(edge, panel, "[{name}] {key} border invisible");
                    let inner = Rect::new(tf.x + 3.0, tf.y + 3.0, tf.w - 6.0, tf.h - 6.0);
                    let fill = rgba_at(&pixels, width, inner.x * s, inner.y * s);
                    assert!(
                        ink_bounds(&pixels, width, px(inner), fill).is_some(),
                        "[{name}] {key} shows no text",
                    );
                }
                let light = matches!(step.extra, Some(Extra::Gallery(true)));
                let gtheme = if light {
                    Theme::light()
                } else {
                    Theme::dark_theme()
                };
                // (a) Cap-band centring: in the 28-high "ok" button, the
                // 'H' column's ink rows centre on the button centre.
                let btn = ui.rect_of("ok").expect("ok button rect");
                let fill = gtheme.button(FontId(0)).normal;
                let fill_px = [fill.r, fill.g, fill.b, fill.a];
                let interior = (
                    (btn.x * s) as u32 + 1,
                    (btn.y * s) as u32 + 1,
                    ((btn.x + btn.w) * s) as u32 - 1,
                    ((btn.y + btn.h) * s) as u32 - 1,
                );
                let ink = ink_bounds(&pixels, width, interior, fill_px)
                    .expect("[gallery] button label ink");
                // 'H' is the first ink run: columns from min_x to the
                // first empty column.
                let mut h_end = ink.0;
                'scan: for x in ink.0..=ink.1 {
                    for y in ink.2..=ink.3 {
                        if rgba_at(&pixels, width, x as f32, y as f32)[..3] != fill_px[..3] {
                            h_end = x;
                            continue 'scan;
                        }
                    }
                    break;
                }
                let (mut h_top, mut h_bot) = (u32::MAX, 0u32);
                for y in ink.2..=ink.3 {
                    for x in ink.0..=h_end {
                        if rgba_at(&pixels, width, x as f32, y as f32)[..3] != fill_px[..3] {
                            h_top = h_top.min(y);
                            h_bot = h_bot.max(y);
                            break;
                        }
                    }
                }
                let band_c = (h_top + h_bot) as f32 * 0.5;
                let btn_c = (btn.y + btn.h * 0.5) * s;
                assert!(
                    (band_c - btn_c).abs() <= 1.5,
                    "[{name}] cap band centre {band_c} vs button centre {btn_c}"
                );
                // (b) The panel border is exactly `max(1, round(scale))`
                // physical px — probe across the left edge mid-height.
                let border = gtheme.border;
                let border_px = [border.r, border.g, border.b, border.a];
                let panel = ui.rect_of("gal_panel").expect("gal_panel rect");
                let row = (panel.y + panel.h * 0.5) * s;
                let edge_x = (panel.x * s).round() as u32;
                let want = s.round().max(1.0) as usize;
                let mut border_cols = 0;
                for x in edge_x.saturating_sub(2)..edge_x + want as u32 + 3 {
                    if rgba_at(&pixels, width, x as f32, row)[..3] == border_px[..3] {
                        border_cols += 1;
                    }
                }
                assert_eq!(
                    border_cols, want,
                    "[{name}] panel hairline is {want} physical px, got {border_cols}"
                );
                // (c) Icon ink centring: Icon::Close's ink bbox centres on
                // its 28x28 cell.
                let cell = ui.rect_of("gal_close").expect("gal_close rect");
                let icon_bg = gtheme.roles.surface.base;
                let icon_bg_px = [icon_bg.r, icon_bg.g, icon_bg.b, icon_bg.a];
                let region = (
                    (cell.x * s) as u32,
                    (cell.y * s) as u32,
                    ((cell.x + cell.w) * s) as u32,
                    ((cell.y + cell.h) * s) as u32,
                );
                let ib = ink_bounds(&pixels, width, region, icon_bg_px)
                    .expect("[gallery] close icon ink");
                let icx = (ib.0 + ib.1 + 1) as f32 * 0.5;
                let icy = (ib.2 + ib.3 + 1) as f32 * 0.5;
                let ccx = (cell.x + cell.w * 0.5) * s;
                let ccy = (cell.y + cell.h * 0.5) * s;
                assert!(
                    (icx - ccx).abs() <= 1.0 && (icy - ccy).abs() <= 1.0,
                    "[{name}] close icon ink centre ({icx},{icy}) vs ({ccx},{ccy})"
                );
                println!("[check] {name}: widget gallery verified on the GPU");
            }
            Some(Extra::CompactVsLegacy) => {
                let (compact_px, cw, ch) = self.compact_capture.take().expect("compact capture");
                assert_eq!((cw, ch), (width, height));
                let mut worst = 0u8;
                for (a, b) in compact_px.iter().zip(pixels.iter()) {
                    worst = worst.max(a.abs_diff(*b));
                }
                assert!(worst <= 1, "compact vs legacy frames differ by {worst} LSB");
            }
            _ => {}
        }
        if step.resize_to.is_some() {
            // Every corner of the resized capture is initialised — the
            // scene background must reach all four, no stale tail.
            let bg = rgba_at(&pixels, width, 4.0 * s, 4.0 * s);
            for &(cx, cy) in &[
                (2.0f32, 2.0f32),
                (width as f32 - 2.0, 2.0),
                (2.0, height as f32 - 2.0),
                (width as f32 - 2.0, height as f32 - 2.0),
            ] {
                let px = rgba_at(&pixels, width, cx, cy);
                assert_eq!(
                    px, bg,
                    "[resized] corner ({cx},{cy}) initialised to background, got {px:?}"
                );
            }
        }
        println!(
            "[check] {}: {}x{} (scale {}) ink {}x{} — {}",
            step.name,
            width,
            height,
            s,
            ink_w as u32,
            ink_h as u32,
            path.display()
        );
        self.step += 1;
        if self.step >= STEPS.len() {
            self.done = true;
            event_loop.exit();
        }
        let _ = theme;
    }
}

/// Textures share one sampler per description and descriptor pool chunks,
/// instead of each owning a sampler, two set layouts and a pool.
fn check_texture_objects_are_shared(ctx: &Arc<GpuContext>) {
    const COUNT: usize = 600;
    const SETS_PER_CHUNK_TEXTURES: usize = 128;
    let before = ctx.texture_object_stats();
    let pixels: Vec<u8> = [32u8, 64, 128, 255]
        .into_iter()
        .cycle()
        .take(4 * 4 * 4)
        .collect();
    let mut manager = TextureManager::new(Arc::clone(ctx));
    let mut textures: Vec<Arc<GpuTexture>> = Vec::with_capacity(COUNT);
    for _ in 0..COUNT {
        let (_, texture) =
            manager.from_pixels("shared texture", &pixels, 4, 4, TextureOptions::default());
        textures.push(texture);
    }
    // The manager's own references go too; only `textures` keeps them alive.
    drop(manager);
    let during = ctx.texture_object_stats();
    assert!(
        during.samplers <= 3 && during.samplers <= before.samplers + 1,
        "[texture_objects_are_shared] samplers {} -> {} for {COUNT} default textures",
        before.samplers,
        during.samplers
    );
    let max_new_pools = COUNT.div_ceil(SETS_PER_CHUNK_TEXTURES);
    assert!(
        during.descriptor_pools <= before.descriptor_pools + max_new_pools,
        "[texture_objects_are_shared] pools {} -> {}, expected at most {max_new_pools} new",
        before.descriptor_pools,
        during.descriptor_pools
    );
    assert_eq!(
        during.live_texture_sets,
        before.live_texture_sets + COUNT,
        "[texture_objects_are_shared] live texture sets"
    );
    drop(textures);
    let after = ctx.texture_object_stats();
    assert_eq!(
        after.live_texture_sets, before.live_texture_sets,
        "[texture_objects_are_shared] live texture sets after drop"
    );
    assert!(
        after.descriptor_pools <= before.descriptor_pools,
        "[texture_objects_are_shared] pools {} after drop, was {}",
        after.descriptor_pools,
        before.descriptor_pools
    );
    println!(
        "[check] texture_objects_are_shared: {COUNT} textures, samplers {} -> {}, pools {} -> {} -> {}",
        before.samplers,
        during.samplers,
        before.descriptor_pools,
        during.descriptor_pools,
        after.descriptor_pools
    );
}

impl ApplicationHandler for Check {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("bedframe check")
                        .with_visible(false)
                        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
                )
                .expect("create hidden window"),
        );
        self.window = Some(Arc::clone(&window));
        let settings = Settings {
            aa: AaMode::None,
            upscale: UpscalePreset::Native,
            sharpness: 0.6,
        };
        let mut renderer = Renderer::new(
            Arc::clone(&window),
            include_str!("../app/scene.wgsl"),
            settings,
        );
        // Every check reads frames back, so keep the composed frame.
        renderer.set_frame_capture(true);
        let ctx = Arc::clone(renderer.context());
        // Solid magenta 8x8: unmistakable texel to assert on.
        let fonts = Fonts::bundled();
        let mut ui = Ui::new(fonts, AtlasConfig::default());
        ui.register_vector_icons(&Icon::ALL);
        ui.set_theme(&Theme::dark_theme());
        let texture = Arc::new(GpuTexture::from_rgba8(
            &ctx,
            "check texel",
            &[255u8, 0, 255, 255]
                .into_iter()
                .cycle()
                .take(8 * 8 * 4)
                .collect::<Vec<u8>>(),
            8,
            8,
            TextureOptions::default(),
        ));
        check_texture_objects_are_shared(&ctx);
        self.renderer = Some(renderer);
        self.ui = Some(ui);
        self.texture = Some(texture);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                // Every resize, not just one this harness asked for: the
                // window manager can change the size at any time, and the
                // swapchain must follow before the next present.
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested if !self.done => {
                self.redraw_pending = false;
                self.run_step(event_loop);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            return;
        }
        if let Some(window) = &self.window {
            window.request_redraw();
            if self.redraw_pending {
                // The previous request never produced a `RedrawRequested`
                // (hidden windows on some platforms): drive the step
                // directly and wake the loop again, or `Wait` would park
                // the harness forever.
                self.redraw_pending = false;
                self.run_step(event_loop);
                if !self.done {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(std::time::Instant::now()));
                }
            }
            self.redraw_pending = true;
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut check = Check {
        window: None,
        renderer: None,
        ui: None,
        texture: None,
        step: 0,
        done: false,
        redraw_pending: false,
        resize_pending: None,
        compact_capture: None,
        retained_frame: None,
    };
    event_loop.run_app(&mut check).expect("run bedframe check");
    assert!(check.done, "bedframe check did not finish its steps");
}
