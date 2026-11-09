//! Run a BedFrame app without writing a window loop.
//!
//! [`run`] owns everything a Dream program needs around its UI: the window,
//! the Vulkan [`Renderer`], the [`Ui`], the mapping from winit events to
//! [`UiInput`], DPI scale, theme, clipboard, IME, and the scheduling that
//! keeps an idle app at 0 Hz. Your program supplies a [`UiApp`] whose `view`
//! builds one frame:
//!
//! ```no_run
//! use dream::ui::app::{AppContext, AppOptions, RunError, UiApp, run};
//! use dream::ui::layout::Style;
//! use dream::ui::tokens::TypeRole;
//! use dream::ui::widgets::Ui;
//! use dream::ui::widgets::adapters::{Button, Text};
//!
//! struct Counter {
//!     count: u32,
//! }
//!
//! impl UiApp for Counter {
//!     fn view(&mut self, ui: &mut Ui, cx: &mut AppContext) {
//!         ui.layout_root(cx.screen(), Style::column().pad(24.0).gap(12.0), |ui| {
//!             ui.add(Text::new(&format!("Count: {}", self.count)).role(TypeRole::Title));
//!             if ui.add(Button::new("Increment")).clicked {
//!                 self.count += 1;
//!             }
//!         });
//!     }
//! }
//!
//! fn main() -> Result<(), RunError> {
//!     run(AppOptions::new("Counter"), Counter { count: 0 })
//! }
//! ```
//!
//! `examples/hello.rs` is a fuller version of this, and new apps should start
//! from it rather than from a hand-written winit loop.
//!
//! # What the runner does
//!
//! - **Idle is 0 Hz.** A frame is built only when something can have
//!   changed: input, a resize, a scale or theme change, a
//!   [`AppContext::request_frame`], an [`AppWaker::wake`] from another thread,
//!   an animation in flight ([`Ui::animating`]) or a UI timer
//!   ([`Ui::next_wake_in`], such as a tooltip's dwell). Otherwise the loop
//!   parks in `ControlFlow::Wait` and uses no CPU or GPU.
//! - **Other threads can wake it.** Data that arrives on a network, file or
//!   timer thread is shown by calling [`AppWaker::wake`] on a handle from
//!   [`AppContext::waker`] once the data is stored. A burst of wakes builds
//!   one frame, and a wake while the window is hidden waits until it shows.
//! - **Paced animation.** An animation is paced in software to the
//!   monitor's refresh rate, so the loop sleeps between frames instead of
//!   blocking inside the driver (some drivers busy-wait in
//!   `vkAcquireNextImageKHR`/`vkWaitForFences`, which burns a CPU core at
//!   high refresh rates). [`PresentMode::Fifo`] (the default) adds vsync on
//!   top, so no frame is ever rendered that the display will not show.
//! - **Hidden windows cost nothing.** A minimised or occluded window builds
//!   no frames. On the transition to hidden the runner releases the scene
//!   resources, trims the renderer's memory and, with
//!   [`AppOptions::idle_trim`], returns the process working set to the OS.
//! - **Idle memory trims.** While the window is shown and quiet, the working
//!   set is trimmed once after start-up and once after each quiet period. See
//!   [`crate::memory::IdleTrim`].
//! - **Input.** Mouse, wheel, keyboard, IME and clipboard are translated for
//!   you. Keyboard focus rings are on.
//!
//! Cursor icons are not handled: BedFrame has no cursor-request API yet. Set
//! one through [`AppContext::window`] if you need it.
//!
//! # Environment
//!
//! `DREAM_APP_AUTOCLOSE_MS=<ms>` makes [`run`] return after that many
//! milliseconds. Tests and automation use it to smoke-test an example
//! deterministically. An unparseable value is ignored.
//!
//! # Fatal GPU errors
//!
//! A lost device, out-of-memory or other unrecoverable presentation error is
//! printed to standard error and ends the loop, and [`run`] then returns
//! `Ok(())`. A timed-out frame is retried, and an outdated swapchain is
//! recreated.

use super::Rect;
use super::atlas::AtlasConfig;
use super::font::Fonts;
use super::render::UiBlend;
use super::theme::Theme;
use super::widgets::{Composition, Key, KeyFeed, Ui, UiInput};
use crate::gfx::{PresentMode, Renderer, SurfaceError};
use crate::memory::{IdleTrim, trim_working_set};
use crate::settings::Settings;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::event::{ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Theme as WinitTheme, Window, WindowId};

/// Window, renderer and UI settings for [`run`].
#[derive(Clone, Debug)]
pub struct AppOptions {
    /// Window title.
    pub title: String,
    /// Logical inner size in points. Default `(960, 640)`.
    pub size: (f32, f32),
    /// Smallest logical inner size the user may resize to. Default `None`.
    pub min_size: Option<(f32, f32)>,
    /// Whether the user may resize the window. Default `true`.
    pub resizable: bool,
    /// The colour palette. Default [`ThemeMode::System`], which follows the
    /// OS dark or light setting and keeps following it.
    pub theme: ThemeMode,
    /// Swapchain present mode. Default [`PresentMode::Fifo`] (vsync).
    pub present_mode: PresentMode,
    /// The colour space the UI blends in. Default [`UiBlend::default`].
    pub ui_blend: UiBlend,
    /// Return the process working set to the OS while the app is quiet and
    /// when it is hidden. Default `true`. See [`crate::memory::IdleTrim`].
    pub idle_trim: bool,
}

impl Default for AppOptions {
    fn default() -> Self {
        Self {
            title: String::from("Dream app"),
            size: (960.0, 640.0),
            min_size: None,
            resizable: true,
            theme: ThemeMode::System,
            present_mode: PresentMode::Fifo,
            ui_blend: UiBlend::default(),
            idle_trim: true,
        }
    }
}

impl AppOptions {
    /// The default options with the given window title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }

    /// Set the logical inner size.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.size = (width, height);
        self
    }

    /// Set the colour palette.
    pub fn theme(mut self, theme: ThemeMode) -> Self {
        self.theme = theme;
        self
    }
}

/// Which palette the window uses.
// `Custom` carries a whole `Theme` (about 900 bytes), which is what makes the
// enum large. It is built once per app and copied at most once per frame, and
// boxing it would drop `Copy` and change the public `Custom(Theme)` shape.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, Default)]
pub enum ThemeMode {
    /// Dark or light, following the OS appearance and tracking changes to it
    /// while the app runs. Dark when the OS does not say.
    #[default]
    System,
    /// [`Theme::dark_theme`].
    Dark,
    /// [`Theme::light`].
    Light,
    /// A palette of your own.
    Custom(Theme),
}

/// Application code driven by [`run`].
pub trait UiApp {
    /// Called once, after the window, renderer and [`Ui`] exist. Use it to
    /// register fonts, load textures or set up the UI.
    fn init(&mut self, _ui: &mut Ui, _cx: &mut AppContext<'_>) {}

    /// Build one frame. Called only when something changed: input, an
    /// animation or a UI timer, or [`AppContext::request_frame`].
    ///
    /// The build closure passed to [`Ui::layout_root`] may run twice in a
    /// frame, so record state from events (`if resp.clicked { … }`) rather
    /// than assigning it from a response.
    fn view(&mut self, ui: &mut Ui, cx: &mut AppContext<'_>);

    /// The window's close button was pressed. Return `false` to keep the app
    /// running. The default closes the window.
    fn close_requested(&mut self, _cx: &mut AppContext<'_>) -> bool {
        true
    }
}

/// The app's handle on the window, renderer and frame loop, passed to every
/// [`UiApp`] callback.
pub struct AppContext<'a> {
    window: &'a Window,
    renderer: &'a mut Renderer,
    theme: &'a Theme,
    waker: &'a AppWaker,
    scale: f32,
    exit: bool,
    frame: bool,
}

impl AppContext<'_> {
    /// The logical screen, from the origin: the size [`Ui::layout_root`]
    /// lays out in.
    pub fn screen(&self) -> Rect {
        let (width, height) = logical_size(self.renderer.display_size(), self.scale);
        Rect::new(0.0, 0.0, width, height)
    }

    /// The UI scale: physical pixels per logical point.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The palette in force for the frame.
    pub fn theme(&self) -> &Theme {
        self.theme
    }

    /// Exit once the current frame has been presented.
    pub fn exit(&mut self) {
        self.exit = true;
    }

    /// Build another frame after this one, even with no input. Use it when
    /// your own state changed and the UI must show it.
    pub fn request_frame(&mut self) {
        self.frame = true;
    }

    /// A handle that wakes the app from any thread, for data that arrives
    /// outside the UI callbacks. See [`AppWaker`].
    pub fn waker(&self) -> AppWaker {
        self.waker.clone()
    }

    /// Change the window title.
    pub fn set_title(&self, title: &str) {
        self.window.set_title(title);
    }

    /// The renderer, for anything the runner does not wrap: textures,
    /// scene resources and so on.
    pub fn renderer(&mut self) -> &mut Renderer {
        &mut *self.renderer
    }

    /// The window, for anything the runner does not wrap, such as cursor
    /// icons.
    pub fn window(&self) -> &Window {
        self.window
    }
}

/// Wakes the app from any thread: it schedules one frame, as
/// [`AppContext::request_frame`] does on the UI thread.
///
/// Get one from [`AppContext::waker`] and move the clone to the thread that
/// produces the data. Store the data first, then call [`wake`](Self::wake), so
/// the frame it schedules reads the new state. Wakes coalesce: while one is
/// queued, further calls add nothing, so calling `wake` once per item is cheap.
/// A wake while the window is hidden marks the screen dirty, and the frame
/// comes when the window shows.
#[derive(Clone)]
pub struct AppWaker {
    proxy: EventLoopProxy<WakeEvent>,
    gate: WakeGate,
}

impl AppWaker {
    /// Schedule one frame. Safe from any thread and cheap. After the app has
    /// exited it does nothing.
    pub fn wake(&self) {
        if self.gate.arm() && self.proxy.send_event(WakeEvent).is_err() {
            // The loop has ended, so nothing will handle the event.
            self.gate.disarm();
        }
    }
}

/// The user event a wake posts to the loop.
#[derive(Clone, Copy, Debug)]
struct WakeEvent;

/// Coalesces wakes: at most one [`WakeEvent`] is queued at a time.
#[derive(Clone, Debug, Default)]
struct WakeGate(Arc<AtomicBool>);

impl WakeGate {
    /// Mark an event as queued. Returns `true` when the caller must send one,
    /// and `false` when one is already queued.
    fn arm(&self) -> bool {
        !self.0.swap(true, Ordering::AcqRel)
    }

    /// The queued event has been handled, so the next wake sends again.
    fn disarm(&self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Why [`run`] could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// The event loop or the window could not be created.
    EventLoop(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventLoop(message) => write!(f, "event loop error: {message}"),
        }
    }
}

impl std::error::Error for RunError {}

/// Open a window and run `app` in it until the window closes or the app
/// calls [`AppContext::exit`].
///
/// This blocks the calling thread. It returns `Ok(())` when the loop ends.
/// The app's [`UiApp::view`] runs only when the frame needs one (see the
/// module docs), so a still screen costs nothing.
pub fn run<A: UiApp + 'static>(options: AppOptions, app: A) -> Result<(), RunError> {
    let event_loop = EventLoop::<WakeEvent>::with_user_event()
        .build()
        .map_err(|err| RunError::EventLoop(err.to_string()))?;
    let exit_at = autoclose_deadline(
        Instant::now(),
        std::env::var("DREAM_APP_AUTOCLOSE_MS").ok().as_deref(),
    );
    let mut runner = Runner::new(options, app, exit_at, event_loop.create_proxy());
    event_loop
        .run_app(&mut runner)
        .map_err(|err| RunError::EventLoop(err.to_string()))?;
    if let Some(error) = runner.error.take() {
        return Err(error);
    }
    // Tear down in dependency order: the UI first, then the renderer (whose
    // `Drop` waits for the GPU to go idle), then the window it presents to.
    if let Some(live) = runner.live.take() {
        let Live {
            ui,
            renderer,
            window,
            ..
        } = live;
        drop(ui);
        drop(renderer);
        drop(window);
    }
    Ok(())
}

/// The window, renderer and UI that exist once the window does. Fields drop
/// in declaration order, which is the teardown order `run` relies on.
struct Live {
    ui: Ui,
    renderer: Renderer,
    window: Arc<Window>,
    theme: Theme,
}

struct Runner<A> {
    options: AppOptions,
    app: A,
    live: Option<Live>,
    error: Option<RunError>,
    /// `DREAM_APP_AUTOCLOSE_MS`, if set.
    exit_at: Option<Instant>,
    /// Input accumulated since the last frame. Edges live here until a frame
    /// takes them; held state is re-applied to every frame.
    pending: UiInput,
    keys: KeyFeed,
    mouse: (f32, f32),
    left_down: bool,
    right_down: bool,
    preedit: Option<Composition>,
    /// Physical pixels per logical point.
    scale: f32,
    /// The OS reported the window's appearance as dark.
    system_dark: bool,
    /// `WindowEvent::Occluded`: the window is covered by other windows.
    occluded: bool,
    /// Whether the window is hidden (minimised or occluded) right now.
    hidden: bool,
    /// A frame must be built at the next opportunity.
    dirty: bool,
    /// When the last presented frame finished.
    last_frame: Option<Instant>,
    /// The last frame's animations are still moving.
    animating: bool,
    /// The monitor's refresh period and when it was read (see `pacing_period`).
    refresh_cache: Option<(Instant, Duration)>,
    /// When the UI's next timer (`Ui::next_wake_in`) is due.
    ui_wake: Option<Instant>,
    exit_requested: bool,
    memory: IdleTrim,
    /// The handle [`AppContext::waker`] clones from.
    waker: AppWaker,
}

impl<A: UiApp> Runner<A> {
    fn new(
        options: AppOptions,
        app: A,
        exit_at: Option<Instant>,
        proxy: EventLoopProxy<WakeEvent>,
    ) -> Self {
        Self {
            options,
            app,
            live: None,
            error: None,
            exit_at,
            waker: AppWaker {
                proxy,
                gate: WakeGate::default(),
            },
            pending: UiInput::default(),
            keys: KeyFeed::default(),
            mouse: (-1.0, -1.0),
            left_down: false,
            right_down: false,
            preedit: None,
            scale: 1.0,
            system_dark: true,
            occluded: false,
            hidden: false,
            dirty: true,
            last_frame: None,
            animating: false,
            refresh_cache: None,
            ui_wake: None,
            exit_requested: false,
            memory: IdleTrim::new(),
        }
    }

    /// Run one app callback with a fresh [`AppContext`]. Returns `None` before
    /// the window exists.
    fn with_app<R>(
        &mut self,
        callback: impl FnOnce(&mut A, &mut Ui, &mut AppContext<'_>) -> R,
    ) -> Option<R> {
        let scale = self.scale;
        let waker = &self.waker;
        let live = self.live.as_mut()?;
        let mut cx = AppContext {
            window: &live.window,
            renderer: &mut live.renderer,
            theme: &live.theme,
            waker,
            scale,
            exit: false,
            frame: false,
        };
        let out = callback(&mut self.app, &mut live.ui, &mut cx);
        let (exit, frame) = (cx.exit, cx.frame);
        self.exit_requested |= exit;
        self.dirty |= frame;
        Some(out)
    }

    /// Enter or leave the hidden state. Entering releases what a hidden window
    /// cannot show and trims the working set. Leaving marks the screen dirty,
    /// so whatever changed while hidden lands on one frame.
    fn set_hidden(&mut self, hidden: bool) {
        if self.hidden == hidden {
            return;
        }
        self.hidden = hidden;
        if !hidden {
            self.dirty = true;
            self.last_frame = None;
            return;
        }
        if let Some(live) = self.live.as_mut() {
            live.renderer.release_scene_resources();
            live.renderer.trim_memory();
        }
        if self.options.idle_trim {
            trim_working_set();
        }
        self.memory = IdleTrim::new();
        self.animating = false;
        self.ui_wake = None;
    }

    /// The refresh period of the monitor the window is on, for software pacing.
    fn refresh_period(&self) -> Duration {
        let millihertz = self
            .live
            .as_ref()
            .and_then(|live| live.window.current_monitor())
            .and_then(|monitor| monitor.refresh_rate_millihertz());
        frame_period(millihertz)
    }

    /// Build one frame from the accumulated input, present it, and record
    /// what the next schedule needs.
    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        if self.hidden || self.live.is_none() {
            return;
        }
        let now = Instant::now();
        let mut input = std::mem::take(&mut self.pending);
        input.dt = frame_dt(self.last_frame.map(|at| now.duration_since(at)));
        // Pointer position and held buttons persist across frames; edges do
        // not. The keyboard latches are re-applied here for the same reason.
        input.mouse = self.mouse;
        input.mouse_down = self.left_down;
        input.mouse_right_down = self.right_down;
        input.preedit = self.preedit.clone();
        self.keys.apply(&mut input);
        self.dirty = false;

        let scale = self.scale;
        let Some(live) = self.live.as_mut() else {
            return;
        };
        let screen = logical_size(live.renderer.display_size(), scale);
        live.ui.begin(input, screen);
        live.ui.set_theme(&live.theme);
        self.with_app(|app, ui, cx| app.view(ui, cx));

        let Some(live) = self.live.as_mut() else {
            return;
        };
        let frame = live.ui.end();
        live.ui.flush_clipboard();
        if let Some(rect) = live.ui.editing_rect() {
            live.window.set_ime_cursor_area(
                LogicalPosition::new(f64::from(rect.x), f64::from(rect.y)),
                LogicalSize::new(f64::from(rect.w), f64::from(rect.h)),
            );
        }
        self.animating = live.ui.animating();
        let wake = live.ui.next_wake_in();
        let presented = live.renderer.render_ui_only(Some(&frame));
        live.ui.recycle_frame(frame);
        if let Err(err) = presented {
            match classify_frame_failure(err) {
                FrameFailure::Retry => {
                    self.dirty = true;
                    return;
                }
                FrameFailure::Recreate => {
                    live.renderer.recreate_swapchain();
                    self.dirty = true;
                    return;
                }
                FrameFailure::Fatal => {
                    eprintln!("[app] frame presentation failed ({err:?}); closing the app");
                    event_loop.exit();
                    return;
                }
            }
        }
        let done = Instant::now();
        self.last_frame = Some(done);
        self.ui_wake = wake.map(|after| done + after);
        self.memory.activity(done);
        if self.exit_requested {
            event_loop.exit();
        }
    }

    /// The close button: the app decides, and either answer ends the loop.
    fn close(&mut self, event_loop: &ActiveEventLoop) {
        let close = self
            .with_app(|app, _ui, cx| app.close_requested(cx))
            .unwrap_or(true);
        if close || self.exit_requested {
            event_loop.exit();
        }
    }

    /// Translate one key event into the feed: typed text first, then the
    /// physical control keys. A composition in flight owns the characters.
    fn keyboard(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        if self.preedit.is_none()
            && let Some(text) = event.logical_key.to_text()
        {
            let mut forwarded = false;
            for ch in text.chars() {
                if ch.is_control() {
                    continue;
                }
                self.keys.key(&mut self.pending, Key::Char(ch), pressed);
                forwarded = true;
            }
            if forwarded {
                return;
            }
        }
        if let Some(key) = map_physical(&event.physical_key) {
            self.keys.key(&mut self.pending, key, pressed);
        }
    }

    /// The period an animation is paced to: the monitor's refresh period
    /// while animating (every present mode, see the module docs), zero
    /// otherwise. The monitor query is cached for half a second.
    fn pacing_period(&mut self) -> Duration {
        if !self.animating {
            return Duration::ZERO;
        }
        let now = Instant::now();
        match self.refresh_cache {
            Some((at, period)) if now.duration_since(at) < Duration::from_millis(500) => period,
            _ => {
                let period = self.refresh_period();
                self.refresh_cache = Some((now, period));
                period
            }
        }
    }
}

impl<A: UiApp> ApplicationHandler<WakeEvent> for Runner<A> {
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: WakeEvent) {
        // Clear the gate before the frame is built, so a wake that lands while
        // the app is still reading its state queues a fresh event.
        self.waker.gate.disarm();
        self.dirty = true;
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.live.is_some() || self.error.is_some() {
            return;
        }
        let (width, height) = self.options.size;
        let mut attributes = Window::default_attributes()
            .with_title(self.options.title.as_str())
            .with_inner_size(LogicalSize::new(f64::from(width), f64::from(height)))
            .with_resizable(self.options.resizable)
            .with_visible(true);
        if let Some((width, height)) = self.options.min_size {
            attributes = attributes
                .with_min_inner_size(LogicalSize::new(f64::from(width), f64::from(height)));
        }
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.error = Some(RunError::EventLoop(format!("create window: {err}")));
                event_loop.exit();
                return;
            }
        };
        // Composition is opt-in per window. The runner keeps it on so IME works
        // in any field, and places the candidate window at the focused editor.
        window.set_ime_allowed(true);
        self.scale = sanitize_scale(window.scale_factor());
        self.system_dark = window.theme().is_none_or(|theme| theme == WinitTheme::Dark);

        let mut renderer = Renderer::new_with_ui_blend(
            Arc::clone(&window),
            crate::DEFAULT_SCENE_WGSL,
            Settings::default(),
            self.options.ui_blend,
        );
        renderer.set_present_mode(self.options.present_mode);
        let mut ui = Ui::new(Fonts::bundled(), AtlasConfig::default());
        ui.set_scale(self.scale);
        ui.set_focus_rings(true);
        let theme = resolve_theme(self.options.theme, self.system_dark);
        self.live = Some(Live {
            ui,
            renderer,
            window,
            theme,
        });
        self.dirty = true;
        self.with_app(|app, ui, cx| app.init(ui, cx));
        if self.exit_requested {
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                self.redraw(event_loop);
                return;
            }
            WindowEvent::CloseRequested => {
                self.close(event_loop);
                return;
            }
            _ => {}
        }
        if self.live.is_none() {
            return;
        }
        // Any event but a redraw is user activity, which restarts the quiet
        // period before the next idle trim.
        self.memory.activity(Instant::now());
        match event {
            WindowEvent::Resized(size) => {
                // Minimising reports 0x0. Keep the last real size; the window
                // comes back with a real one.
                if size.width > 0 && size.height > 0 {
                    if let Some(live) = self.live.as_mut() {
                        live.renderer.resize(size);
                    }
                    self.dirty = true;
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale = sanitize_scale(scale_factor);
                if let Some(live) = self.live.as_mut() {
                    live.ui.set_scale(self.scale);
                    live.renderer.resize(live.window.inner_size());
                }
                self.dirty = true;
            }
            WindowEvent::ThemeChanged(theme) => {
                self.system_dark = theme == WinitTheme::Dark;
                if matches!(self.options.theme, ThemeMode::System) {
                    let palette = resolve_theme(self.options.theme, self.system_dark);
                    if let Some(live) = self.live.as_mut() {
                        live.theme = palette;
                    }
                    self.dirty = true;
                }
            }
            WindowEvent::Occluded(occluded) => {
                // Handled by `about_to_wait`, which hides or shows the window.
                self.occluded = occluded;
            }
            WindowEvent::Focused(focused) => {
                if !focused {
                    // Keys and buttons held on the way out are gone: drop every
                    // held edge, drag, latch and composition.
                    if let Some(live) = self.live.as_mut() {
                        live.ui.cancel_input();
                    }
                    self.pending = UiInput::default();
                    self.left_down = false;
                    self.right_down = false;
                    self.keys.modifiers(false, false, false);
                    self.preedit = None;
                }
                self.dirty = true;
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Physical pixels become logical exactly once, here.
                let at = position.to_logical::<f64>(f64::from(self.scale));
                self.mouse = (at.x as f32, at.y as f32);
                self.dirty = true;
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse = (-1.0, -1.0);
                self.dirty = true;
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if pressed {
                            self.pending.mouse_pressed = true;
                        } else {
                            self.pending.mouse_released = true;
                        }
                        self.left_down = pressed;
                    }
                    MouseButton::Right => {
                        if pressed {
                            self.pending.mouse_right_pressed = true;
                        }
                        self.right_down = pressed;
                    }
                    _ => return,
                }
                self.dirty = true;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let line_height = self
                    .live
                    .as_ref()
                    .map_or(40.0, |live| live.ui.scroll_config.line_height);
                let (wheel, precise) = wheel_deltas(delta, line_height, f64::from(self.scale));
                self.pending.wheel += wheel;
                self.pending.wheel_precise += precise;
                self.dirty = true;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                // The platform shortcut modifier: Command on macOS, Control
                // elsewhere.
                let command = if cfg!(target_os = "macos") {
                    state.super_key()
                } else {
                    state.control_key()
                };
                self.keys
                    .modifiers(state.shift_key(), command, state.alt_key());
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.keyboard(&event);
                self.dirty = true;
            }
            WindowEvent::Ime(ime) => {
                match ime {
                    Ime::Commit(text) => {
                        self.preedit = None;
                        self.pending.text.push_str(&text);
                    }
                    Ime::Preedit(text, cursor) => self.preedit = composition(text, cursor),
                    Ime::Disabled => self.preedit = None,
                    Ime::Enabled => {}
                }
                self.dirty = true;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.live.is_none() {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        // Minimisation has no event on most platforms, so it is polled here.
        let minimised = self
            .live
            .as_ref()
            .is_some_and(|live| live.window.is_minimized() == Some(true));
        self.set_hidden(self.occluded || minimised);

        let now = Instant::now();
        let trim_at = if self.hidden || !self.options.idle_trim {
            None
        } else {
            self.memory.trim_if_due(now);
            self.memory.next_deadline()
        };
        let decision = schedule(ScheduleInput {
            now,
            hidden: self.hidden,
            dirty: self.dirty,
            animating: self.animating,
            last_frame: self.last_frame,
            frame_period: self.pacing_period(),
            ui_wake: self.ui_wake,
            trim_at,
            exit_at: self.exit_at,
        });
        match decision {
            Schedule::Exit => event_loop.exit(),
            Schedule::Redraw => {
                if let Some(live) = self.live.as_ref() {
                    live.window.request_redraw();
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Schedule::WaitUntil(at) => event_loop.set_control_flow(ControlFlow::WaitUntil(at)),
            Schedule::Park => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}

/// What the loop does next, decided by [`schedule`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Schedule {
    /// `DREAM_APP_AUTOCLOSE_MS` has elapsed.
    Exit,
    /// Request a redraw now.
    Redraw,
    /// Wake at this instant, then decide again.
    WaitUntil(Instant),
    /// Nothing is owed: wait for an OS event (0 Hz).
    Park,
}

/// Everything [`schedule`] reads, gathered from the runner's state.
#[derive(Clone, Copy, Debug)]
struct ScheduleInput {
    now: Instant,
    hidden: bool,
    dirty: bool,
    animating: bool,
    /// When the last presented frame finished.
    last_frame: Option<Instant>,
    /// The monitor's refresh period, the pace of a running animation.
    frame_period: Duration,
    ui_wake: Option<Instant>,
    trim_at: Option<Instant>,
    exit_at: Option<Instant>,
}

/// The loop's decision for `input`. Pure, so the scheduling rules can be
/// tested without a window.
///
/// In order:
/// 1. A reached exit deadline ends the loop.
/// 2. A hidden window owes no frames, so it parks, waking only for the exit.
/// 3. Input, a resize or a request is a frame now.
/// 4. Animation waits one refresh period after the last frame, whatever the
///    present mode: sleeping here instead of blocking in the driver keeps a
///    high-refresh animation cheap.
/// 5. A UI timer that is due is a frame; a future one is a wake-up.
/// 6. Otherwise the earliest of the trim and exit deadlines is a wake-up,
///    and nothing at all parks the loop.
fn schedule(input: ScheduleInput) -> Schedule {
    let ScheduleInput {
        now,
        hidden,
        dirty,
        animating,
        last_frame,
        frame_period,
        ui_wake,
        trim_at,
        exit_at,
    } = input;
    if exit_at.is_some_and(|at| now >= at) {
        return Schedule::Exit;
    }
    if hidden {
        return exit_at.map_or(Schedule::Park, Schedule::WaitUntil);
    }
    if dirty {
        return Schedule::Redraw;
    }
    let mut due = None;
    if animating {
        match last_frame {
            None => return Schedule::Redraw,
            Some(last) => {
                let at = last + frame_period;
                if now >= at {
                    return Schedule::Redraw;
                }
                due = earliest(due, Some(at));
            }
        }
    }
    if let Some(at) = ui_wake {
        if at <= now {
            return Schedule::Redraw;
        }
        due = earliest(due, Some(at));
    }
    due = earliest(due, trim_at);
    due = earliest(due, exit_at);
    due.map_or(Schedule::Park, Schedule::WaitUntil)
}

/// The earlier of two optional instants, `None` only when both are `None`.
fn earliest(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// A physical size as logical points at `scale`.
fn logical_size(physical: (u32, u32), scale: f32) -> (f32, f32) {
    (physical.0 as f32 / scale, physical.1 as f32 / scale)
}

/// A scale factor the UI can use: finite and positive, else 1.
fn sanitize_scale(scale: f64) -> f32 {
    let scale = scale as f32;
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

/// The seconds the frame's UI clock advances by: the real time since the last
/// frame, zero for the first.
///
/// This is deliberately not capped. UI timers such as tooltips and toasts are
/// armed relative to real time, so a woken frame must advance the clock by the
/// full wait, or the timer would fire late. Animations are safe with a large
/// step: a settled animation stays settled, and springs take a bounded number
/// of sub-steps.
fn frame_dt(since_last: Option<Duration>) -> f32 {
    since_last.map_or(0.0, |elapsed| elapsed.as_secs_f32())
}

/// The refresh period of a monitor in millihertz, or 60 Hz when unknown.
fn frame_period(millihertz: Option<u32>) -> Duration {
    match millihertz {
        Some(mhz) if mhz > 0 => Duration::from_secs_f64(1000.0 / f64::from(mhz)),
        _ => Duration::from_secs_f64(1.0 / 60.0),
    }
}

/// The palette for `mode`, given the OS appearance.
fn resolve_theme(mode: ThemeMode, system_dark: bool) -> Theme {
    match mode {
        ThemeMode::System if system_dark => Theme::dark_theme(),
        ThemeMode::System => Theme::light(),
        ThemeMode::Dark => Theme::dark_theme(),
        ThemeMode::Light => Theme::light(),
        ThemeMode::Custom(theme) => theme,
    }
}

/// A wheel event as `(wheel, wheel_precise)` in logical pixels. A line delta
/// is scaled by the scroll line height. A pixel delta is already a distance,
/// so it is only converted to logical points. Positive is up, as in
/// [`UiInput::wheel`].
fn wheel_deltas(delta: MouseScrollDelta, line_height: f32, scale: f64) -> (f32, f32) {
    match delta {
        MouseScrollDelta::LineDelta(_, y) => (y * line_height, 0.0),
        MouseScrollDelta::PixelDelta(pixels) => (0.0, pixels.to_logical::<f64>(scale).y as f32),
    }
}

/// The composition shown for an IME preedit: empty text means none.
fn composition(text: String, cursor: Option<(usize, usize)>) -> Option<Composition> {
    if text.is_empty() {
        None
    } else {
        Some(Composition { text, cursor })
    }
}

/// The instant `DREAM_APP_AUTOCLOSE_MS` closes the app, if it is set to a
/// number of milliseconds.
fn autoclose_deadline(now: Instant, value: Option<&str>) -> Option<Instant> {
    let millis = value?.trim().parse::<u64>().ok()?;
    Some(now + Duration::from_millis(millis))
}

/// Layout-independent control keys. Text comes from `logical_key` in
/// [`Runner::keyboard`], so this covers only the keys that have no text.
fn map_physical(key: &PhysicalKey) -> Option<Key> {
    let PhysicalKey::Code(code) = key else {
        return None;
    };
    Some(match code {
        KeyCode::ShiftLeft | KeyCode::ShiftRight => Key::Shift,
        KeyCode::ControlLeft | KeyCode::ControlRight => Key::Control,
        KeyCode::AltLeft | KeyCode::AltRight => Key::Alt,
        KeyCode::SuperLeft | KeyCode::SuperRight => Key::Super,
        KeyCode::Escape => Key::Escape,
        KeyCode::Enter | KeyCode::NumpadEnter => Key::Enter,
        KeyCode::ArrowUp => Key::Up,
        KeyCode::ArrowDown => Key::Down,
        KeyCode::ArrowLeft => Key::Left,
        KeyCode::ArrowRight => Key::Right,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::Delete => Key::Delete,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Tab => Key::Tab,
        _ => return None,
    })
}

/// What the runner does when presenting a frame fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameFailure {
    /// Drop this frame and build another on the next pass.
    Retry,
    /// Rebuild the swapchain, then build another frame.
    Recreate,
    /// Nothing the runner can do: end the loop.
    Fatal,
}

fn classify_frame_failure(err: SurfaceError) -> FrameFailure {
    match err {
        SurfaceError::Timeout => FrameFailure::Retry,
        SurfaceError::Outdated => FrameFailure::Recreate,
        SurfaceError::Lost | SurfaceError::OutOfMemory | SurfaceError::Other => FrameFailure::Fatal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalPosition;

    /// A schedule input with nothing pending: idle, visible, FIFO.
    fn idle_at(now: Instant) -> ScheduleInput {
        ScheduleInput {
            now,
            hidden: false,
            dirty: false,
            animating: false,
            last_frame: None,
            frame_period: Duration::from_millis(16),
            ui_wake: None,
            trim_at: None,
            exit_at: None,
        }
    }

    #[test]
    fn app_runner_defaults_match_the_spec() {
        let options = AppOptions::default();
        assert_eq!(options.size, (960.0, 640.0));
        assert_eq!(options.min_size, None);
        assert!(options.resizable);
        assert!(matches!(options.theme, ThemeMode::System));
        assert_eq!(options.present_mode, PresentMode::Fifo);
        assert_eq!(options.ui_blend, UiBlend::default());
        assert!(options.idle_trim);
        let named = AppOptions::new("Notes").size(800.0, 500.0);
        assert_eq!(named.title, "Notes");
        assert_eq!(named.size, (800.0, 500.0));
    }

    #[test]
    fn app_runner_key_mapping_table() {
        let table = [
            (KeyCode::ShiftLeft, Key::Shift),
            (KeyCode::ShiftRight, Key::Shift),
            (KeyCode::ControlLeft, Key::Control),
            (KeyCode::AltRight, Key::Alt),
            (KeyCode::SuperLeft, Key::Super),
            (KeyCode::Escape, Key::Escape),
            (KeyCode::Enter, Key::Enter),
            (KeyCode::NumpadEnter, Key::Enter),
            (KeyCode::ArrowUp, Key::Up),
            (KeyCode::ArrowDown, Key::Down),
            (KeyCode::ArrowLeft, Key::Left),
            (KeyCode::ArrowRight, Key::Right),
            (KeyCode::Home, Key::Home),
            (KeyCode::End, Key::End),
            (KeyCode::PageUp, Key::PageUp),
            (KeyCode::PageDown, Key::PageDown),
            (KeyCode::Delete, Key::Delete),
            (KeyCode::Backspace, Key::Backspace),
            (KeyCode::Tab, Key::Tab),
        ];
        for (code, key) in table {
            assert_eq!(
                map_physical(&PhysicalKey::Code(code)),
                Some(key),
                "{code:?}"
            );
        }
    }

    #[test]
    fn app_runner_key_mapping_ignores_text_and_unidentified_keys() {
        // Letters and the space bar arrive as text, not as physical keys.
        assert_eq!(map_physical(&PhysicalKey::Code(KeyCode::KeyA)), None);
        assert_eq!(map_physical(&PhysicalKey::Code(KeyCode::Space)), None);
        assert_eq!(
            map_physical(&PhysicalKey::Unidentified(
                winit::keyboard::NativeKeyCode::Unidentified
            )),
            None
        );
    }

    #[test]
    fn app_runner_wheel_conversion() {
        // One notch of a line wheel is one scroll line height, upward positive.
        assert_eq!(
            wheel_deltas(MouseScrollDelta::LineDelta(0.0, 1.0), 40.0, 2.0),
            (40.0, 0.0)
        );
        assert_eq!(
            wheel_deltas(MouseScrollDelta::LineDelta(0.0, -2.0), 20.0, 1.0),
            (-40.0, 0.0)
        );
        // A pixel delta is physical: at 2x, 30 physical pixels is 15 points.
        let pixels = PhysicalPosition::new(0.0, -30.0);
        assert_eq!(
            wheel_deltas(MouseScrollDelta::PixelDelta(pixels), 40.0, 2.0),
            (0.0, -15.0)
        );
    }

    #[test]
    fn app_runner_frame_dt_keeps_real_time() {
        assert_eq!(frame_dt(None), 0.0, "the first frame has no interval");
        let short = frame_dt(Some(Duration::from_millis(16)));
        assert!((short - 0.016).abs() < 1e-6, "got {short}");
        // Long waits are not clipped: a woken UI timer must see the full wait.
        let long = frame_dt(Some(Duration::from_secs(3)));
        assert!((long - 3.0).abs() < 1e-6, "got {long}");
    }

    #[test]
    fn app_runner_schedule_exits_at_the_deadline() {
        let now = Instant::now();
        let exit = ScheduleInput {
            exit_at: Some(now),
            dirty: true,
            ..idle_at(now)
        };
        assert_eq!(
            schedule(exit),
            Schedule::Exit,
            "exit beats every other wake"
        );
        let before = ScheduleInput {
            exit_at: Some(now + Duration::from_secs(1)),
            ..idle_at(now)
        };
        assert_eq!(
            schedule(before),
            Schedule::WaitUntil(now + Duration::from_secs(1))
        );
    }

    #[test]
    fn app_runner_schedule_hidden_parks_and_ignores_trims() {
        let now = Instant::now();
        let hidden = ScheduleInput {
            hidden: true,
            dirty: true,
            animating: true,
            ui_wake: Some(now),
            trim_at: Some(now),
            ..idle_at(now)
        };
        assert_eq!(schedule(hidden), Schedule::Park, "hidden owes nothing");
    }

    #[test]
    fn app_runner_schedule_dirty_redraws_now() {
        let now = Instant::now();
        let dirty = ScheduleInput {
            dirty: true,
            ..idle_at(now)
        };
        assert_eq!(schedule(dirty), Schedule::Redraw);
    }

    #[test]
    fn app_runner_schedule_animating_first_frame_is_immediate() {
        let now = Instant::now();
        let first = ScheduleInput {
            animating: true,
            last_frame: None,
            ..idle_at(now)
        };
        assert_eq!(schedule(first), Schedule::Redraw);
    }

    #[test]
    fn app_runner_schedule_animating_paces_in_software() {
        let now = Instant::now();
        let last = now - Duration::from_millis(5);
        let period = Duration::from_millis(16);
        let mailbox = ScheduleInput {
            animating: true,
            last_frame: Some(last),
            frame_period: period,
            ..idle_at(now)
        };
        assert_eq!(
            schedule(mailbox),
            Schedule::WaitUntil(last + period),
            "wait for the next refresh, never spin"
        );
        let late = ScheduleInput {
            last_frame: Some(now - Duration::from_millis(20)),
            ..mailbox
        };
        assert_eq!(
            schedule(late),
            Schedule::Redraw,
            "a late refresh is due now"
        );
        let immediate = ScheduleInput {
            last_frame: None,
            ..mailbox
        };
        assert_eq!(schedule(immediate), Schedule::Redraw);
    }

    #[test]
    fn app_runner_schedule_ui_timer_wakes_or_redraws() {
        let now = Instant::now();
        let later = now + Duration::from_millis(300);
        let future = ScheduleInput {
            ui_wake: Some(later),
            ..idle_at(now)
        };
        assert_eq!(schedule(future), Schedule::WaitUntil(later));
        let due = ScheduleInput {
            ui_wake: Some(now),
            ..idle_at(now)
        };
        assert_eq!(schedule(due), Schedule::Redraw);
    }

    #[test]
    fn app_runner_schedule_idle_trim_deadline_wakes_the_loop() {
        let now = Instant::now();
        let trim = now + Duration::from_millis(1500);
        let pending = ScheduleInput {
            trim_at: Some(trim),
            ..idle_at(now)
        };
        assert_eq!(schedule(pending), Schedule::WaitUntil(trim));
    }

    #[test]
    fn app_runner_schedule_fully_idle_parks() {
        let now = Instant::now();
        assert_eq!(schedule(idle_at(now)), Schedule::Park);
    }

    #[test]
    fn app_runner_schedule_earliest_deadline_wins() {
        let now = Instant::now();
        let ui = now + Duration::from_millis(40);
        let trim = now + Duration::from_millis(900);
        let exit = now + Duration::from_secs(8);
        let input = ScheduleInput {
            ui_wake: Some(ui),
            trim_at: Some(trim),
            exit_at: Some(exit),
            ..idle_at(now)
        };
        assert_eq!(schedule(input), Schedule::WaitUntil(ui));
        let animating = ScheduleInput {
            animating: true,
            last_frame: Some(now),
            frame_period: Duration::from_millis(16),
            ..input
        };
        assert_eq!(
            schedule(animating),
            Schedule::WaitUntil(now + Duration::from_millis(16)),
            "the frame pace beats the later UI timer"
        );
    }

    #[test]
    fn app_runner_frame_period_from_refresh() {
        let at_60 = frame_period(Some(60_000)).as_secs_f64();
        assert!((at_60 - 1.0 / 60.0).abs() < 1e-9, "got {at_60}");
        let at_144 = frame_period(Some(144_000)).as_secs_f64();
        assert!((at_144 - 1.0 / 144.0).abs() < 1e-9, "got {at_144}");
        // An unknown or nonsensical rate paces at 60 Hz.
        assert!((frame_period(None).as_secs_f64() - 1.0 / 60.0).abs() < 1e-9);
        assert!((frame_period(Some(0)).as_secs_f64() - 1.0 / 60.0).abs() < 1e-9);
    }

    #[test]
    fn app_runner_theme_resolution() {
        assert!(resolve_theme(ThemeMode::System, true).dark);
        assert!(!resolve_theme(ThemeMode::System, false).dark);
        assert!(resolve_theme(ThemeMode::Dark, false).dark);
        assert!(!resolve_theme(ThemeMode::Light, true).dark);
        let custom = Theme::light();
        assert!(!resolve_theme(ThemeMode::Custom(custom), true).dark);
    }

    #[test]
    fn app_runner_failure_classification() {
        assert_eq!(
            classify_frame_failure(SurfaceError::Timeout),
            FrameFailure::Retry
        );
        assert_eq!(
            classify_frame_failure(SurfaceError::Outdated),
            FrameFailure::Recreate
        );
        assert_eq!(
            classify_frame_failure(SurfaceError::Lost),
            FrameFailure::Fatal
        );
        assert_eq!(
            classify_frame_failure(SurfaceError::OutOfMemory),
            FrameFailure::Fatal
        );
        assert_eq!(
            classify_frame_failure(SurfaceError::Other),
            FrameFailure::Fatal
        );
    }

    #[test]
    fn app_runner_ime_empty_preedit_clears_the_composition() {
        assert_eq!(composition(String::new(), None), None);
        assert_eq!(
            composition("ni".into(), Some((2, 2))),
            Some(Composition {
                text: "ni".into(),
                cursor: Some((2, 2)),
            })
        );
    }

    #[test]
    fn app_runner_autoclose_env_parses_milliseconds() {
        let now = Instant::now();
        assert_eq!(autoclose_deadline(now, None), None);
        assert_eq!(autoclose_deadline(now, Some("soon")), None);
        assert_eq!(
            autoclose_deadline(now, Some(" 4000 ")),
            Some(now + Duration::from_millis(4000))
        );
    }

    #[test]
    fn app_waker_is_send_sync_and_clone() {
        fn assert_handle<T: Send + Sync + Clone>() {}
        assert_handle::<AppWaker>();
    }

    #[test]
    fn app_waker_gate_coalesces_until_handled() {
        let gate = WakeGate::default();
        assert!(gate.arm(), "the first wake sends an event");
        assert!(!gate.arm(), "a wake while one is queued adds nothing");
        assert!(!gate.clone().arm(), "clones share the same gate");
        gate.disarm();
        assert!(gate.arm(), "once the event is handled, the next wake sends");
    }

    #[test]
    fn app_waker_hidden_window_only_marks_dirty() {
        // A hidden window parks even when dirty; showing it again redraws.
        let now = Instant::now();
        let hidden = ScheduleInput {
            hidden: true,
            dirty: true,
            ..idle_at(now)
        };
        assert_eq!(schedule(hidden), Schedule::Park);
        let shown = ScheduleInput {
            hidden: false,
            ..hidden
        };
        assert_eq!(schedule(shown), Schedule::Redraw);
    }

    #[test]
    fn app_runner_scale_and_logical_size_are_sanitised() {
        assert_eq!(sanitize_scale(2.0), 2.0);
        assert_eq!(sanitize_scale(0.0), 1.0);
        assert_eq!(sanitize_scale(f64::NAN), 1.0);
        assert_eq!(logical_size((1920, 1080), 2.0), (960.0, 540.0));
    }
}
