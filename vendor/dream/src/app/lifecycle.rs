use super::main_pane::{MainPanePage, SceneManagerState, draw_main_pane};
use super::{
    App, AppSettings, FramePacing, InspectorState, Layout, ThemeChoice, WindowCommand, chrome,
    inspector, scene,
};
use crate::camera::Camera;
use crate::gfx::Renderer;
use crate::settings::Settings;
use crate::ui::Rect;
use crate::ui::draw::UiFrame;
use crate::ui::font::Fonts;
use crate::ui::icons::Icon;
use crate::ui::theme::Theme;
use crate::ui::widgets::{KeyFeed, Ui, UiInput};
use std::time::Instant;

impl App {
    /// The window's UI scale, sanitized for [`Ui::set_scale`].
    fn sanitize_scale(scale: f64) -> f32 {
        let scale = scale as f32;
        if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        }
    }

    pub fn new(window: std::sync::Arc<winit::window::Window>, engine_settings: Settings) -> Self {
        // The scale is read before the renderer takes ownership of the
        // window: everything the app lays out is logical from here on.
        let scale = Self::sanitize_scale(window.scale_factor());
        let renderer = Renderer::new(window, include_str!("scene.wgsl"), engine_settings);
        let fonts = Fonts::bundled();
        let mut ui = Ui::new(fonts, crate::ui::atlas::AtlasConfig::default());
        ui.register_vector_icons(&Icon::ALL);
        // Studio keeps the keyboard-nav focus ring; BedFrame's default is off.
        ui.set_focus_rings(true);
        ui.set_scale(scale);
        let theme = Theme::dark_theme();
        let screen = {
            let size = renderer.display_size();
            (size.0 as f32 / scale, size.1 as f32 / scale)
        };
        let layout = Layout::compute(screen, &theme.metrics);
        let scene_manager = SceneManagerState::new(layout.main_pane);
        let scene = scene::Scene::new(&renderer);
        let mut camera = Camera::default();
        camera.set_aspect(screen.0 / screen.1.max(1.0));
        scene.orbit.apply(&mut camera);
        Self {
            renderer,
            ui,
            camera,
            theme,
            scene,
            settings: AppSettings::default(),
            theme_choice: ThemeChoice::default(),
            system_dark: true,
            theme_cells: [Rect::default(); 3],
            layout,
            main_pane_page: MainPanePage::SceneManager,
            settings_return: MainPanePage::SceneManager,
            scene_manager,
            preview_pixels: std::collections::HashMap::new(),
            preview_textures: std::collections::HashMap::new(),
            retired_previews: Vec::new(),
            pending_thumb_capture: None,
            thumb_job: None,
            thumb_warmup: None,
            thumb_failed: std::collections::HashSet::new(),
            inspector: InspectorState::default(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            engine_settings,
            input: UiInput::default(),
            mouse: (0.0, 0.0),
            screen,
            scale,
            ui_wake: None,
            ime_preedit: None,
            dirty: true,
            last_input: Instant::now(),
            last_frame: Instant::now(),
            frames_rendered: 0,
            frames_skipped: 0,
            frames_immediate: 0,
            captures: 0,
            drag: None,
            drag_start_orbit: None,
            viewport_cursor: None,
            fixed_dt: None,
            animating: false,
            pacing: FramePacing {
                cap_hz: std::env::var("DREAM_FPS_CAP")
                    .ok()
                    .and_then(|value| value.trim().parse::<f32>().ok())
                    .filter(|hz| *hz > 0.0),
                ..FramePacing::default()
            },
            window_commands: Vec::new(),
            resize_pending: None,
            window_maximized: false,
            titlebar_drag: false,
            pan_mode: false,
            menu_open: false,
            menu_at: (0.0, 0.0),
            menu_focus: 0,
            sharpness_rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            left_down: false,
            right_down: false,
            shift_held: false,
            ctrl_held: false,
            alt_held: false,
            keys: KeyFeed::default(),
            quit: false,
            hidden: false,
        }
    }

    // --- Scale / screen ------------------------------------------------------

    /// Adopt a new UI scale (the window moved to a different-display-density
    /// monitor): rebuilds the logical screen and layout from the physical
    /// display size. Settings and widget state survive — only the geometry
    /// is rederived.
    pub fn set_scale(&mut self, scale: f32) {
        let scale = Self::sanitize_scale(scale as f64);
        if (scale - self.scale).abs() < f32::EPSILON {
            return;
        }
        self.scale = scale;
        self.ui.set_scale(scale);
        self.refresh_screen();
        self.dirty = true;
    }

    /// Recompute the logical screen, layout and camera aspect from the
    /// renderer's physical display size and the current scale. Call after
    /// `Renderer::resize` and [`Self::set_scale`].
    pub fn refresh_screen(&mut self) {
        let size = self.renderer.display_size();
        self.screen = (size.0 as f32 / self.scale, size.1 as f32 / self.scale);
        self.layout = Layout::compute(self.screen, &self.theme.metrics);
        self.camera
            .set_aspect(self.screen.0 / self.screen.1.max(1.0));
    }

    /// `rect` in physical pixels, for APIs that take display space
    /// (`render_ui`'s scene rect, thumbnail captures, IME carets).
    pub fn physical_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            rect.x * self.scale,
            rect.y * self.scale,
            rect.w * self.scale,
            rect.h * self.scale,
        )
    }

    // --- Input edges ---------------------------------------------------------

    /// The window lost focus: drop every held button, key, capture and
    /// pending text — a defocused app must not keep drags running or let a
    /// stale Shift/Control latch change the next click's meaning.
    pub fn cancel_input(&mut self) {
        self.input = UiInput::default();
        self.input.mouse = self.mouse;
        self.left_down = false;
        self.right_down = false;
        self.shift_held = false;
        self.ctrl_held = false;
        self.alt_held = false;
        self.ime_preedit = None;
        self.end_drag();
        self.ui.cancel_input();
        self.dirty = true;
    }

    /// A whole committed string (IME `Ime::Commit` delivers text in one
    /// event): queued for the focused field verbatim — no first-char
    /// truncation.
    pub fn text_input(&mut self, text: &str) {
        self.last_input = Instant::now();
        self.dirty = true;
        self.input.text.push_str(text);
    }

    /// Track an IME preedit (composition in flight). The text is kept, not
    /// committed — richer inline display lands with the editing stage.
    pub fn ime_preedit(&mut self, text: String, cursor: Option<(usize, usize)>) {
        self.last_input = Instant::now();
        self.dirty = true;
        self.ime_preedit = if text.is_empty() {
            None
        } else {
            Some((text, cursor))
        };
    }

    /// Where the IME candidate window should appear: the focused *editor's*
    /// rect in logical units (the host converts to winit logical coords).
    /// A focused button is not a composition target.
    pub fn ime_cursor_rect(&self) -> Option<Rect> {
        self.ui.editing_rect()
    }

    /// Whether an IME composition is in flight: while it is, character
    /// `KeyboardInput`s belong to the composition and must not also
    /// commit as text.
    pub fn composing(&self) -> bool {
        self.ime_preedit.is_some()
    }

    // --- Theme -------------------------------------------------------------

    /// Adopt a palette choice and apply it. Instant by design: a colour
    /// change is not a thing that moves.
    pub fn set_theme_choice(&mut self, choice: ThemeChoice) {
        self.theme_choice = choice;
        self.theme = choice.theme(self.system_dark);
        self.dirty = true;
    }

    /// Record the OS appearance preference the window reported, and follow it
    /// when (and only when) the choice is [`ThemeChoice::System`]. `None`
    /// means the platform did not say: the app keeps its default dark, so a
    /// headless or unsupported platform still has a defined answer.
    pub fn set_system_dark(&mut self, dark: Option<bool>) {
        let dark = dark.unwrap_or(true);
        if dark == self.system_dark {
            return;
        }
        self.system_dark = dark;
        if self.theme_choice == ThemeChoice::System {
            self.set_theme_choice(ThemeChoice::System);
        }
    }

    /// Swap between the dark and light palettes, as an explicit choice: the
    /// Settings page's Theme row also offers following the OS. Instant by
    /// design: a colour change is not a thing that moves.
    pub fn flip_theme(&mut self) {
        let next = if self.theme.dark {
            ThemeChoice::Light
        } else {
            ThemeChoice::Dark
        };
        self.set_theme_choice(next);
    }

    // --- Frame loop ----------------------------------------------------------

    /// Report whether the window is completely hidden — minimised, or fully
    /// occluded by other windows — from the host's platform events and
    /// queries. While it is, no frame is built and no wake-up is scheduled:
    /// the loop parks at zero cost however much motion is in flight, and a
    /// queued thumbnail capture simply waits for the window to show. Showing
    /// it again marks the screen dirty, so whatever changed while hidden
    /// lands on a single frame.
    pub fn set_hidden(&mut self, hidden: bool) {
        if self.hidden == hidden {
            return;
        }
        self.hidden = hidden;
        if !hidden {
            self.dirty = true;
        }
    }

    /// Advance simulation: hero spin, camera easing.
    pub fn update(&mut self, dt: f32) {
        if self.settings.spin > 0.0 && self.scene.debug_loaded() {
            self.scene.advance(dt * self.settings.spin / 0.4);
        }
        self.scene.orbit.ease_zoom(dt);
        self.scene.orbit.apply(&mut self.camera);
    }

    /// Whether anything is still moving on its own: tweens, popups, the
    /// spinning cube, a camera ease, or the short grace window after input.
    /// This is what separates "a frame is owed" from "the screen is still and
    /// the loop may park at 0 Hz".
    pub fn motion_in_flight(&self, now: Instant) -> bool {
        now.duration_since(self.last_input).as_secs_f32() < 0.35
            || self.animating
            || scene_motion_in_flight(
                self.main_pane_page,
                self.scene.orbit.zooming(),
                self.settings.spin > 0.0 && self.scene.debug_loaded(),
            )
    }

    /// True when a frame can show something new. This is the whole
    /// on-demand contract: still screen -> false -> the loop stops asking
    /// for frames (0 Hz) until an event sets `dirty`.
    ///
    /// Input answers immediately (no added latency); self-sustaining motion
    /// (tweens, the spinning cube) is paced to the display's refresh
    /// rate, so continuous animation costs one frame per refresh period
    /// instead of thousands.
    ///
    /// **Ask this once per frame, at the moment the frame is built** — it
    /// spends the frame it grants by moving the pacing grid on, so asking
    /// earlier (to decide whether to *schedule* a frame) and then asking again
    /// in the frame itself answers "nothing due" the second time and drops the
    /// frame. [`Self::next_frame_at`] is the scheduling question; it does not
    /// spend anything.
    pub fn needs_render(&mut self, now: Instant) -> bool {
        // A completely hidden window owes nothing: no frame it builds can
        // be seen, so none is built — 0 Hz, no GPU work — until it shows.
        if self.hidden {
            return false;
        }
        // Input and one-shot state answer at once, and re-anchor the grid to
        // the frame they get.
        if self.dirty || self.drag.is_some() {
            self.pacing.sync(now);
            self.frames_immediate += 1;
            return true;
        }
        // A pending UI clock arriving (a tooltip mid-dwell, a toast nearing
        // expiry) wakes the loop once — like input, it is not paced.
        if self.ui_wake.is_some_and(|wake| wake <= now) {
            self.pacing.sync(now);
            return true;
        }
        self.motion_in_flight(now) && self.pacing.due_now(now)
    }

    /// When the next frame becomes due, if the loop should wake for one.
    ///
    /// A pure query for the host loop: `Some(instant)` means "ask again at
    /// `instant`", and an instant that has already arrived means "show a frame
    /// now". Nothing moves, so the host may call it as often as it likes;
    /// [`Self::needs_render`] makes the actual decision. Self-sustaining motion
    /// is paced to the display, so a deferred frame needs a timed wake —
    /// otherwise the app would park forever after the first paced frame.
    pub fn next_frame_at(&self, now: Instant) -> Option<Instant> {
        // Hidden: no deadline worth waking for — the loop parks until an OS
        // event arrives, the window showing again among them.
        if self.hidden {
            return None;
        }
        earliest_frame_deadline(
            now,
            self.dirty || self.drag.is_some(),
            self.motion_in_flight(now)
                .then(|| self.pacing.next_due(now))
                .flatten(),
            self.ui_wake,
        )
    }

    /// Build one UI frame from current state (no GPU work). `dt` is the
    /// real wall-clock step for the UI's clocks — not the scene's clamped
    /// step — so a deadline-fed frame measures the time it actually waited.
    pub fn build_ui_frame(&mut self, dt: f32) -> UiFrame {
        let screen = {
            let size = self.renderer.display_size();
            (size.0 as f32 / self.scale, size.1 as f32 / self.scale)
        };
        if screen != self.screen {
            self.screen = screen;
            self.layout = Layout::compute(screen, &self.theme.metrics);
            self.camera.set_aspect(screen.0 / screen.1.max(1.0));
            self.dirty = true;
        }
        let mut input = std::mem::take(&mut self.input);
        input.dt = dt;
        // Window commands are edge-triggered: the host drains them once
        // per frame; anything left over is stale and would fire late.
        self.window_commands.clear();
        // A resize-frame press was latched at event time; re-queue it here
        // so it rides the same post-frame drain as the title bar's Move.
        if let Some(edge) = self.resize_pending.take() {
            self.window_commands.push(WindowCommand::Resize(edge));
        }
        // The pointer position is persistent state; edges are not.
        input.mouse = self.mouse;
        // So is a held button: re-apply it over the per-frame edges.
        input.mouse_down = self.left_down;
        input.mouse_right_down = self.right_down;
        // Modifier mirrors: held state survives the per-frame input take.
        self.keys.apply(&mut input);
        // A live IME composition is drawn (underlined) by the focused
        // field without touching its document.
        input.preedit = self
            .ime_preedit
            .clone()
            .map(|(text, cursor)| crate::ui::widgets::Composition { text, cursor });
        self.ui.begin(input, self.screen);
        // One palette for the whole frame: styled widgets get theirs from the
        // caller, the rest (sections, tooltips, scrollbars) from here.
        self.ui.set_theme(&self.theme);
        // An open context menu owns the pointer: everything else still draws,
        // but nothing behind the popup can be clicked through it. The surface
        // is predeclared (layered, not modal) so coverage is live on the
        // menu's first frame, and input confines to its rect.
        if let Some(popup) = chrome::context_menu_rect(self) {
            self.ui
                .declare_input_surface(popup, crate::ui::widgets::LAYER_POPUP, false);
            self.ui.set_input_rect(Some(popup));
        }
        chrome::toolbar(self);
        chrome::rail(self);
        draw_main_pane(self);
        inspector::inspector(self);
        // The window bar floats above the panels, and the context menu above
        // that: both draw late so they get first say over the pointer.
        chrome::window_bar(self);
        chrome::context_menu(self);
        let frame = self.ui.end();
        // A field that copied/cut hands its text to the host clipboard.
        self.ui.flush_clipboard();
        frame
    }

    /// Build the UI, then composite one frame.
    pub fn render(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        // The scene's clamped step protects it from a long stall; the UI's
        // clocks measure real time — a tooltip woken after a 0 Hz park must
        // see the seconds that passed, or `next_wake_in` just reschedules.
        self.update(self.fixed_dt.unwrap_or(elapsed.min(0.1)));
        let frame = self.build_ui_frame(self.fixed_dt.unwrap_or(elapsed));
        self.animating = self.ui.animating();
        self.ui_wake = self.ui.next_wake_in().map(|wake| now + wake);
        self.queue_pending_thumbnail();
        let viewport = self.physical_rect(self.layout.viewport);
        let need_scene = self.main_pane_page == MainPanePage::Viewport
            || self.thumb_warmup.is_some()
            || self.thumb_job.is_some()
            || self.pending_thumb_capture.is_some();
        let App {
            renderer,
            scene,
            camera,
            ..
        } = self;
        if need_scene {
            renderer
                .render_ui(
                    camera,
                    Some(viewport),
                    |draw| {
                        scene.draw(draw);
                        1
                    },
                    Some(&frame),
                )
                .expect("render frame");
        } else {
            renderer.render_ui_only(Some(&frame)).expect("render frame");
        }
        // The submitted geometry's buffers go straight back to the next
        // frame's draw list instead of the allocator.
        self.ui.recycle_frame(frame);
        self.capture_pending_thumbnail();
        self.frames_rendered += 1;
        // Input is consumed by the frame it was rendered in; self-
        // sustaining motion re-arms through `animating` / the spin
        // setting, paced by `needs_render`. A queued thumbnail is read back
        // a couple of frames later, so keep frames coming until it lands.
        self.dirty = self.thumb_job.is_some();
    }

    /// One loop iteration: render if needed, otherwise record the skip.
    pub fn tick(&mut self, now: Instant) {
        if self.needs_render(now) {
            self.render(now);
        } else {
            self.frames_skipped += 1;
        }
    }
}

pub(crate) fn scene_motion_in_flight(page: MainPanePage, zooming: bool, spinning: bool) -> bool {
    page == MainPanePage::Viewport && (zooming || spinning)
}

/// Earliest frame deadline across the scene and UI clocks: `immediate`
/// (dirty/drag) wins outright, otherwise the earlier of the animation
/// and UI wakes, `None` when nothing is due and the host can park.
pub(crate) fn earliest_frame_deadline(
    now: Instant,
    immediate: bool,
    animation_due: Option<Instant>,
    ui_due: Option<Instant>,
) -> Option<Instant> {
    if immediate {
        return Some(now);
    }
    match (animation_due, ui_due) {
        (Some(a), Some(u)) => Some(a.min(u)),
        (Some(a), None) => Some(a),
        (None, Some(u)) => Some(u),
        (None, None) => None,
    }
}

#[cfg(test)]
mod deadline_tests {
    use super::earliest_frame_deadline;
    use std::time::{Duration, Instant};

    #[test]
    fn deadline_picks_earliest_or_parks() {
        let now = Instant::now();
        let a = now + Duration::from_millis(30);
        let u = now + Duration::from_millis(10);
        assert_eq!(earliest_frame_deadline(now, false, None, None), None);
        assert_eq!(earliest_frame_deadline(now, false, Some(a), None), Some(a));
        assert_eq!(earliest_frame_deadline(now, false, None, Some(u)), Some(u));
        assert_eq!(
            earliest_frame_deadline(now, false, Some(a), Some(u)),
            Some(u),
            "earlier UI wake wins"
        );
        assert_eq!(
            earliest_frame_deadline(now, true, Some(a), Some(u)),
            Some(now),
            "immediate returns now"
        );
    }

    #[test]
    fn scene_motion_counts_only_on_the_viewport_page() {
        use super::super::main_pane::MainPanePage;
        use super::scene_motion_in_flight;
        for zooming in [false, true] {
            for spinning in [false, true] {
                let moving = zooming || spinning;
                assert_eq!(
                    scene_motion_in_flight(MainPanePage::Viewport, zooming, spinning),
                    moving
                );
                assert!(!scene_motion_in_flight(
                    MainPanePage::SceneManager,
                    zooming,
                    spinning
                ));
                assert!(!scene_motion_in_flight(
                    MainPanePage::Settings,
                    zooming,
                    spinning
                ));
            }
        }
        assert!(!scene_motion_in_flight(
            MainPanePage::Viewport,
            false,
            false
        ));
    }
}
