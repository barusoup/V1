//! The main pane: the central workspace of Dream Studio.
//!
//! Rather than a fixed viewport, the main pane is a dynamic display area with a
//! tab system that hosts different views:
//! - [`MainPanePage::SceneManager`]: The boot and management page for creating,
//!   managing, and selecting 3D scenes, as well as booting the debug scene.
//! - [`MainPanePage::Viewport`]: The active 3D camera and geometry viewport.
//!
//! [`MainPanePage::Settings`] is not a tab: it is a dedicated page that hides
//! the tab strip and page system while open, returning to the page the user
//! came from when closed.

use super::App;
use super::settings_page::draw_settings_page;
use crate::ui::icons::Icon;
use crate::ui::tween::Ease;
use crate::ui::widgets::{ButtonStyle, LabelStyle};
use crate::ui::{Color, Rect};

/// Default ID for the built-in debug scene.
pub const DEBUG_SCENE_ID: u64 = 1;
/// Reserved ID for the retired empty-studio starter (kept for path compat;
/// new scenes start from it via "New Scene" instead).
pub const EMPTY_STUDIO_ID: u64 = 2;

/// Which page currently occupies the main pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MainPanePage {
    /// Boot menu and scene list / creation UI.
    SceneManager,
    /// Active 3D viewport canvas.
    Viewport,
    /// The app's own preferences (theme and friends).
    Settings,
}

/// Metadata and state for an existing scene in the studio.
#[derive(Clone, Debug, PartialEq)]
pub struct ManagedScene {
    pub id: u64,
    pub name: String,
    pub description: String,
    pub object_count: usize,
    pub is_debug: bool,
    /// Orbit the viewport had when this scene was last exited, as
    /// `(yaw, pitch)`. Recorded for a future reopen-at-view; card previews
    /// are real 3D captures (see [`PreviewImage`]).
    pub last_view: Option<(f32, f32)>,
}

/// Square edge of a scene card preview, in pixels.
pub const THUMB_SIZE: u32 = 224;

/// Scene card preview height, in pixels; the strip below it holds the name.
pub const CARD_H: f32 = 184.0;

/// A real 3D capture of a scene: square sRGB RGBA8, [`THUMB_SIZE`] wide.
#[derive(Clone, Debug, PartialEq)]
pub struct PreviewImage {
    pub pixels: Vec<u8>,
}

/// Hidden one-frame shoot for scenes with no exit capture yet: the state to
/// restore once the capture lands, so the shoot never leaks into the app.
#[derive(Clone, Debug)]
pub struct ThumbWarmup {
    pub scene_id: u64,
    pub prev_orbit: super::scene::OrbitCamera,
    pub debug_was_loaded: bool,
}

/// Inertial hold-and-drag state for one horizontal scene row.
///
/// No visible scrollbar: the user presses inside the strip, holds, and drags
/// left/right. Small press-release gestures stay clicks; only movement past a
/// threshold becomes a scroll, and release flings with decaying velocity.
#[derive(Clone, Debug, Default)]
pub struct SceneRowScroll {
    /// Current scroll offset in pixels (>= 0).
    pub offset: f32,
    /// Fling velocity in pixels/second (applied once released).
    pub velocity: f32,
    anchor_x: Option<f32>,
    anchor_offset: f32,
    last_x: f32,
    dragging: bool,
}

/// Outcome of one row-scroll update.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowUpdate {
    /// True on the release frame of a drag: child button clicks must be ignored.
    pub suppress_click: bool,
    /// True while the row is dragging or flinging (the app owes another frame).
    pub animating: bool,
}

impl SceneRowScroll {
    /// Press-drag threshold in pixels before a press becomes a scroll.
    const DRAG_THRESHOLD: f32 = 6.0;
    /// Friction for the fling, per second.
    const FRICTION: f32 = 3.5;

    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        strip: Rect,
        content_w: f32,
        mouse: (f32, f32),
        down: bool,
        pressed: bool,
        released: bool,
        dt: f32,
    ) -> RowUpdate {
        let max_offset = (content_w - strip.w).max(0.0);
        // Keep a stale offset valid when scenes are added/removed.
        self.offset = self.offset.clamp(0.0, max_offset);
        let mut out = RowUpdate::default();

        if pressed && strip.contains(mouse) {
            self.anchor_x = Some(mouse.0);
            self.anchor_offset = self.offset;
            self.last_x = mouse.0;
            self.dragging = false;
            self.velocity = 0.0;
        }

        if down && let Some(anchor) = self.anchor_x {
            let total = mouse.0 - anchor;
            if !self.dragging && total.abs() > Self::DRAG_THRESHOLD {
                self.dragging = true;
            }
            if self.dragging {
                self.offset = (self.anchor_offset - total).clamp(0.0, max_offset);
                let dx = mouse.0 - self.last_x;
                if dt > 1e-6 {
                    let instant = -dx / dt;
                    self.velocity = self.velocity * 0.7 + instant * 0.3;
                }
                self.last_x = mouse.0;
                out.animating = true;
            }
        }

        if released && self.anchor_x.is_some() {
            let was_drag = self.dragging;
            self.anchor_x = None;
            self.dragging = false;
            if was_drag {
                // This release ends a scroll, not a click.
                out.suppress_click = true;
                out.animating = self.velocity.abs() > 1.0;
            } else {
                self.velocity = 0.0;
            }
        }

        if !down && self.anchor_x.is_none() && self.velocity.abs() > 1.0 && dt > 1e-6 {
            self.offset += self.velocity * dt;
            self.velocity *= (-Self::FRICTION * dt).exp();
            if self.velocity.abs() < 20.0 {
                self.velocity = 0.0;
            }
            let clamped = self.offset.clamp(0.0, max_offset);
            if clamped != self.offset {
                self.offset = clamped;
                self.velocity = 0.0;
            } else {
                self.offset = clamped;
                out.animating = true;
            }
        }

        out
    }
}

/// State for the scene manager page and main pane tabs.
#[derive(Clone, Debug)]
pub struct SceneManagerState {
    pub scenes: Vec<ManagedScene>,
    pub active_scene_id: Option<u64>,
    pub new_scene_name: String,
    pub next_scene_id: u64,
    /// Most-interacted-with scene ids, most recent first (capped).
    pub recent_scene_ids: Vec<u64>,
    /// Hold-and-drag scroll state for the "Scenes" row.
    pub scenes_row: SceneRowScroll,
    /// Hold-and-drag scroll state for the "Recent" row.
    pub recent_row: SceneRowScroll,
    /// Published rect of the "Boot Debug Scene" button on the management page.
    pub boot_debug_rect: Rect,
    /// Published rect of the "Create Scene" button.
    pub create_scene_rect: Rect,
    /// Published rect of the Scene Manager tab.
    pub tab_manager_rect: Rect,
    /// Published rect of the Viewport tab.
    pub tab_viewport_rect: Rect,
    /// Published rect of the Settings tab. Kept for path compatibility:
    /// Settings is a dedicated page now, so this stays zeroed.
    pub tab_settings_rect: Rect,
}

impl SceneManagerState {
    pub fn new(main_pane: Rect) -> Self {
        // Only the debug scene ships: empty canvases come from "New Scene".
        let initial_scenes = vec![ManagedScene {
            id: DEBUG_SCENE_ID,
            name: "Debug Scene".to_string(),
            description: "Interactive test scene with checker floor, static geometry, and spinning hero cube."
                .to_string(),
            object_count: 4,
            is_debug: true,
            last_view: None,
        }];

        Self {
            scenes: initial_scenes,
            active_scene_id: None,
            new_scene_name: String::new(),
            next_scene_id: 3,
            recent_scene_ids: Vec::new(),
            scenes_row: SceneRowScroll::default(),
            recent_row: SceneRowScroll::default(),
            boot_debug_rect: Rect::default(),
            create_scene_rect: Rect::default(),
            tab_manager_rect: Rect::new(main_pane.x + 8.0, main_pane.y + 4.0, 130.0, 28.0),
            tab_viewport_rect: Rect::new(main_pane.x + 144.0, main_pane.y + 4.0, 130.0, 28.0),
            tab_settings_rect: Rect::new(main_pane.x + 280.0, main_pane.y + 4.0, 130.0, 28.0),
        }
    }

    /// Record an interaction (open/create/duplicate) for "Recent".
    pub fn touch_scene(&mut self, id: u64) {
        self.recent_scene_ids.retain(|&other| other != id);
        self.recent_scene_ids.insert(0, id);
        self.recent_scene_ids.truncate(20);
    }

    /// Recent scenes, most-interacted-with first.
    pub fn recent_scenes(&self) -> Vec<ManagedScene> {
        let mut out = Vec::new();
        for id in &self.recent_scene_ids {
            if let Some(scene) = self.scenes.iter().find(|s| s.id == *id) {
                out.push(scene.clone());
            }
        }
        out
    }

    /// Next unused `scene<number>` placeholder: one past the highest number
    /// already taken (case-insensitive, `scene3` and `Scene 3` both count), or
    /// `scene1` when none is taken.
    pub fn next_placeholder_name(&self) -> String {
        let mut highest = 0u64;
        for scene in &self.scenes {
            let lower = scene.name.trim().to_lowercase();
            if let Some(rest) = lower.strip_prefix("scene") {
                let digits = rest.trim_start_matches(' ').trim();
                if !digits.is_empty()
                    && digits.chars().all(|c| c.is_ascii_digit())
                    && let Ok(n) = digits.parse::<u64>()
                {
                    highest = highest.max(n);
                }
            }
        }
        format!("scene{}", highest + 1)
    }
}

/// The debug-scene boot button rect on the management page.
pub fn debug_button_rect(app: &App) -> Rect {
    app.scene_manager.boot_debug_rect
}

/// Draw the main pane: the top tab strip and the active page (Scenes or Viewport).
pub fn draw_main_pane(app: &mut App) {
    let main_pane = app.layout.main_pane;
    let theme = app.theme;
    let font = app.font();
    let m = theme.metrics;

    // Settings is a dedicated page, not a tab: while it shows, the whole
    // tab system is hidden and the page owns the full pane.
    if app.main_pane_page == MainPanePage::Settings {
        app.scene_manager.tab_manager_rect = Rect::default();
        app.scene_manager.tab_viewport_rect = Rect::default();
        app.scene_manager.tab_settings_rect = Rect::default();
        draw_settings_page(app, main_pane);
        return;
    }

    const TAB_BAR_H: f32 = 36.0;
    let tab_bar_rect = Rect::new(main_pane.x, main_pane.y, main_pane.w, TAB_BAR_H);

    // Tab strip background
    app.ui.draw_list_mut().rect(tab_bar_rect, theme.g1);
    app.ui.draw_list_mut().rect(
        Rect::new(
            tab_bar_rect.x,
            tab_bar_rect.max().1 - m.border,
            tab_bar_rect.w,
            m.border,
        ),
        theme.border,
    );

    // Tab 1: Scenes
    let tab_h = 28.0;
    let tab_y = tab_bar_rect.center().1 - tab_h * 0.5;
    let tab1_rect = Rect::new(main_pane.x + m.pad, tab_y, 130.0, tab_h);
    app.scene_manager.tab_manager_rect = tab1_rect;
    let is_mgr_active = app.main_pane_page == MainPanePage::SceneManager;

    let tab1_style = ButtonStyle {
        font,
        size: 13.0,
        text: if is_mgr_active { theme.fg } else { theme.muted },
        text_hover: theme.fg,
        text_disabled: theme.faint,
        normal: if is_mgr_active {
            theme.g2
        } else {
            Color::TRANSPARENT
        },
        hover: theme.g3,
        pressed: theme.pressed,
        accent: theme.accent,
        corner_radius: m.radius,
        disabled: false,
        on_icon: theme.accent_text,
        border: None,
    };

    if app.ui.button("Scenes", tab1_rect, tab1_style).clicked {
        app.show_scene_manager();
    }

    // Tab 2: Viewport. A viewport only exists while it is bound to something
    // that uses it (the debug scene or an open managed scene), so the tab is
    // hidden whenever nothing is bound.
    let bound = viewport_bound(app);
    let next_tab_x = tab1_rect.max().0 + m.gap;
    if bound {
        let tab2_rect = Rect::new(next_tab_x, tab_y, 130.0, tab_h);
        app.scene_manager.tab_viewport_rect = tab2_rect;
        let is_vp_active = app.main_pane_page == MainPanePage::Viewport;

        let tab2_label = if app.scene.debug_loaded() {
            "Viewport (Debug)"
        } else if let Some(active_id) = app.scene_manager.active_scene_id {
            if let Some(scene) = app.scene_manager.scenes.iter().find(|s| s.id == active_id) {
                match scene.name.as_str() {
                    "Debug Scene" => "Viewport (Debug)",
                    name if name.len() > 10 => "Viewport (Active)",
                    name => name,
                }
            } else {
                "Viewport"
            }
        } else {
            "Viewport"
        };

        let tab2_style = ButtonStyle {
            font,
            size: 13.0,
            text: if is_vp_active { theme.fg } else { theme.muted },
            text_hover: theme.fg,
            text_disabled: theme.faint,
            normal: if is_vp_active {
                theme.g2
            } else {
                Color::TRANSPARENT
            },
            hover: theme.g3,
            pressed: theme.pressed,
            accent: theme.accent,
            corner_radius: m.radius,
            disabled: false,
            on_icon: theme.accent_text,
            border: None,
        };

        if app.ui.button(tab2_label, tab2_rect, tab2_style).clicked {
            app.main_pane_page = MainPanePage::Viewport;
            app.dirty = true;
        }
    } else {
        app.scene_manager.tab_viewport_rect = Rect::default();
    }
    // No Settings tab: Settings is a dedicated page that hides the tab
    // system while it is open (the toolbar gear opens it).
    app.scene_manager.tab_settings_rect = Rect::default();

    // Content area below tab bar
    let content_rect = Rect::new(
        main_pane.x,
        main_pane.y + TAB_BAR_H,
        main_pane.w,
        (main_pane.h - TAB_BAR_H).max(0.0),
    );

    // The viewport page is unreachable while its viewport is unbound; fall
    // back to the scene manager so an unbound viewport can never be shown.
    if app.main_pane_page == MainPanePage::Viewport && !bound {
        app.main_pane_page = MainPanePage::SceneManager;
    }

    match app.main_pane_page {
        MainPanePage::SceneManager => {
            draw_scene_manager_page(app, content_rect);
        }
        MainPanePage::Viewport => {
            draw_viewport_overlay(app, content_rect);
        }
        // Settings returns early above: it replaces the tabbed pane.
        MainPanePage::Settings => {}
    }
}

/// A viewport exists only while it is bound to something that uses it: the
/// debug scene or an open managed scene.
fn viewport_bound(app: &App) -> bool {
    app.scene.debug_loaded() || app.scene_manager.active_scene_id.is_some()
}

/// Draw the Scenes page: two hold-and-drag horizontal rows.
///
/// - Top row "Scenes": every scene plus a leading "New Scene" card.
/// - Bottom row "Recent": most-interacted-with scenes.
///
/// Rows are left-aligned to the pane, have no visible scrollbar;
/// press-hold-drag scrolls with inertia, while a plain press-release
/// still clicks.
fn draw_scene_manager_page(app: &mut App, area: Rect) {
    const CARD: f32 = 224.0;
    const GAP: f32 = 16.0;
    const LABEL_H: f32 = 26.0;

    let theme = app.theme;
    let font = app.font();

    // Fill the background so 3D geometry underneath does not peek through
    app.ui.draw_list_mut().rect(area, theme.bg);

    let content_h = 24.0 + LABEL_H + CARD + 28.0 + LABEL_H + CARD + 24.0;
    let body = app.ui.scroll_begin("scene_manager_scroll", area, content_h);

    // Left-aligned rows: no centered container, so wide panes use the room
    // instead of parking a narrow column in the middle.
    let x = body.x + 24.0;
    let container_w = (body.w - 48.0).max(320.0);
    let mut y = body.y + 24.0;

    // Snapshot input once: row scroll updates and card buttons share it.
    let (mouse, down, pressed, released, dt) = {
        let input = app.ui.input();
        (
            input.mouse,
            input.mouse_down,
            input.mouse_pressed,
            input.mouse_released,
            input.dt,
        )
    };

    let scenes_snapshot = app.scene_manager.scenes.clone();
    let recent_snapshot = app.scene_manager.recent_scenes();

    // --- Row 1: Scenes (New Scene card first, then every scene) ---
    app.ui.label(
        Rect::new(x, y, container_w, LABEL_H),
        "Scenes",
        LabelStyle::new(font)
            .size(14.0)
            .color(theme.fg)
            .vcenter(true),
    );
    y += LABEL_H;
    let strip = Rect::new(x, y, container_w, CARD);
    let content_w = (1 + scenes_snapshot.len()) as f32 * (CARD + GAP);

    let row_update = app
        .scene_manager
        .scenes_row
        .update(strip, content_w, mouse, down, pressed, released, dt);
    if row_update.animating {
        app.dirty = true;
    }
    let offset = app.scene_manager.scenes_row.offset;

    let mut create_clicked = false;
    let mut scene_to_open: Option<u64> = None;

    app.ui.draw_list_mut().push_clip(strip);

    // Leading New Scene card (always cell 0): no placeholder name shown.
    {
        let card = Rect::new(strip.x - offset, strip.y, CARD, CARD_H);
        app.scene_manager.create_scene_rect = card;
        app.ui.push_id("scenes_create");
        let mut style = theme.button(font);
        style.normal = theme.g1;
        style.hover = theme.g2;
        style.size = 13.0;
        let clicked = app.ui.button("", card, style).clicked;
        app.ui.pop_id();
        // Centred affordance: the name sits *below* the card like scene
        // cards, so the icon owns the card's middle (muted, not fg — the
        // card is an empty slot, not a label).
        let icon_side = 28.0;
        app.ui.icon(
            Rect::new(
                card.center().0 - icon_side * 0.5,
                card.center().1 - icon_side * 0.5,
                icon_side,
                icon_side,
            ),
            Icon::Plus,
            20,
            theme.muted,
        );
        draw_card_name(app, card, "New Scene");
        if clicked && !row_update.suppress_click {
            create_clicked = true;
        }
    }

    for (i, scene) in scenes_snapshot.iter().enumerate() {
        let card = Rect::new(
            strip.x - offset + (i as f32 + 1.0) * (CARD + GAP),
            strip.y,
            CARD,
            CARD_H,
        );
        // Publish the debug scene's card for the scripted boot button.
        if scene.is_debug {
            app.scene_manager.boot_debug_rect = card;
        }
        app.ui.push_id(&format!("scenes_card_{}", scene.id));
        let mut style = theme.button(font);
        style.normal = theme.g1;
        style.hover = theme.g2;
        style.size = 13.0;
        let response = app.ui.button("", card, style);
        // The capture covers the button's own fill, so the same hover the
        // button would show is carried out and painted over the preview.
        let hover = card_hover(app, "hover", response.hovered);
        app.ui.pop_id();

        draw_scene_card(app, card, scene, hover);
        if response.clicked && !row_update.suppress_click {
            scene_to_open = Some(scene.id);
        }
    }

    app.ui.draw_list_mut().pop_clip();
    y += CARD + 28.0;

    // --- Row 2: Recent ---
    app.ui.label(
        Rect::new(x, y, container_w, LABEL_H),
        "Recent",
        LabelStyle::new(font)
            .size(14.0)
            .color(theme.fg)
            .vcenter(true),
    );
    y += LABEL_H;
    let recent_strip = Rect::new(x, y, container_w, CARD);
    let recent_content_w = (recent_snapshot.len().max(1)) as f32 * (CARD + GAP);

    let recent_update = app.scene_manager.recent_row.update(
        recent_strip,
        recent_content_w,
        mouse,
        down,
        pressed,
        released,
        dt,
    );
    if recent_update.animating {
        app.dirty = true;
    }
    let recent_offset = app.scene_manager.recent_row.offset;

    let mut recent_to_open: Option<u64> = None;
    app.ui.draw_list_mut().push_clip(recent_strip);
    if recent_snapshot.is_empty() {
        app.ui.label(
            Rect::new(
                recent_strip.x - recent_offset,
                recent_strip.y + 8.0,
                recent_strip.w,
                20.0,
            ),
            "Nothing to show right now.",
            LabelStyle::new(font)
                .size(13.0)
                .color(theme.muted)
                .vcenter(true),
        );
    } else {
        for (i, scene) in recent_snapshot.iter().enumerate() {
            let card = Rect::new(
                recent_strip.x - recent_offset + i as f32 * (CARD + GAP),
                recent_strip.y,
                CARD,
                CARD_H,
            );
            app.ui.push_id(&format!("recent_card_{}", scene.id));
            let mut style = theme.button(font);
            style.normal = theme.g1;
            style.hover = theme.g2;
            style.size = 13.0;
            let response = app.ui.button("", card, style);
            let hover = card_hover(app, "hover", response.hovered);
            app.ui.pop_id();

            draw_scene_card(app, card, scene, hover);
            if response.clicked && !recent_update.suppress_click {
                recent_to_open = Some(scene.id);
            }
        }
    }
    app.ui.draw_list_mut().pop_clip();

    if create_clicked {
        app.create_new_scene();
    } else if let Some(id) = scene_to_open.or(recent_to_open) {
        app.open_managed_scene(id);
    } else if app.main_pane_page == MainPanePage::SceneManager {
        // Staying put with no navigation: shoot first previews in the
        // background instead of leaving blank cards.
        app.maybe_arm_thumb_warmup();
    }

    app.ui.scroll_end();
}

/// Name shared by New Scene and scene cards, drawn in the strip beneath the
/// preview (`card` is the preview rect).
fn draw_card_name(app: &mut App, card: Rect, name: &str) {
    let font = app.font();
    let label_rect = Rect::new(card.x, card.max().1 + 10.0, card.w, 20.0);
    app.ui.label(
        label_rect,
        name,
        LabelStyle::new(font)
            .size(14.0)
            .color(Color::WHITE)
            .vcenter(true),
    );
}

/// Hover fade for one card, matching the button widget's timing: quicker in
/// than out. Called inside the card's id scope, so rows and cards keep
/// separate fades.
fn card_hover(app: &mut App, key: &str, hovered: bool) -> f32 {
    let (duration, ease) = if hovered {
        (0.12, Ease::CubicOut)
    } else {
        (0.18, Ease::CubicOut)
    };
    let target = if hovered { 1.0 } else { 0.0 };
    app.ui.anim_f32(key, target, duration, ease)
}

/// One scene card: a real 3D capture of the scene plus its name beneath.
/// Exits capture the viewpoint left behind; scenes never opened get a hidden
/// one-frame top-down shoot on the Scenes page.
///
/// `hover` is the card button's 0..1 fade. The capture covers the button's
/// own fill, so the lift is painted over the preview instead of under it.
fn draw_scene_card(app: &mut App, card: Rect, scene: &ManagedScene, hover: f32) {
    /// Peak alpha of the hover lift, 0..1: the theme's ink (near-white in
    /// the dark theme, near-black in the light one) laid over the capture at
    /// a fraction of full strength, so a thumbnail reads as lifted without
    /// washing out — the preview's counterpart to the plain card's fill bump.
    const HOVER_LIFT: f32 = 0.14;

    let theme = app.theme;
    let m = theme.metrics;
    draw_scene_preview(app, card, scene.id);
    app.ui
        .draw_list_mut()
        .round_rect(card, m.radius, Color::BLACK.with_alpha(110));
    if hover > 0.0 {
        let lift = theme
            .fg
            .with_alpha((HOVER_LIFT * hover * 255.0).round() as u8);
        app.ui.draw_list_mut().round_rect(card, m.radius, lift);
    }
    draw_card_name(app, card, &scene.name);
}

/// Real 3D card preview: the scene's exit capture, or the flat canvas until
/// its first shoot lands.
fn draw_scene_preview(app: &mut App, card: Rect, scene_id: u64) {
    // The strip already clips to its own rect; the card clip keeps a
    // half-scrolled preview out of the neighbouring cell.
    let radius = app.theme.metrics.radius;
    app.ui.draw_list_mut().push_clip(card);
    if let Some(texture) = app.preview_textures.get(&scene_id).cloned() {
        // The texture is already cropped to the card's aspect (see
        // `upload_thumbnail`), so its UVs span the card exactly and the
        // rounded corners land on the card's own bottom edge.
        app.ui
            .draw_list_mut()
            .image_rounded(card, &texture, Color::WHITE, radius);
    } else {
        // The viewport's own backdrop: blank canvas, not a placeholder.
        app.ui
            .draw_list_mut()
            .round_rect(card, radius, Color::rgb(0x1e, 0x1e, 0x1e));
    }
    app.ui.draw_list_mut().pop_clip();
}

/// Subtle overlay drawn over the viewport when it is active. Only reachable
/// while the viewport is bound to a scene, so there is no empty state.
fn draw_viewport_overlay(app: &mut App, area: Rect) {
    let theme = app.theme;
    let font = app.font();
    let m = theme.metrics;

    // Small HUD chip in top-left corner, only for the debug scene.
    if app.scene.debug_loaded() {
        let hud_rect = Rect::new(area.x + 12.0, area.y + 12.0, 200.0, 26.0);
        app.ui
            .draw_list_mut()
            .round_rect(hud_rect, m.radius, theme.g1);
        app.ui
            .draw_list_mut()
            .rounded_frame(hud_rect, m.radius, m.border, theme.g1, theme.border);

        app.ui.label(
            Rect::new(hud_rect.x + 8.0, hud_rect.y, hud_rect.w - 16.0, hud_rect.h),
            "Debug Scene (4 Objects)",
            LabelStyle::new(font)
                .size(12.0)
                .color(theme.muted)
                .vcenter(true),
        );
    }
}
