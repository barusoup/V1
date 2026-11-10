// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Root and host loop for v1.
//!
//! The host owns the window, the renderer and the BedFrame UI; the state machine in [`ui`] owns
//! what is on screen. Input is latched into plain fields and handed to the UI once per frame,
//! because `UiInput` edges are single-use.

// Release builds are GUI-subsystem so no console window opens with the app. Debug builds keep
// the console so tracing output stays visible.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod anim;
mod backend;
mod clock;
mod gpu;
mod icons;
mod identity;
mod instance;
mod library;
mod media;
mod playlist;
mod spectrum;
mod store;
mod theme;
mod ui;
mod update;
mod vault;
mod widgets;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use dream::gfx::Renderer;
use dream::settings::Settings;
use dream::ui::atlas::AtlasConfig;
use dream::ui::font::Fonts;
use dream::ui::widgets::{Key, KeyFeed, Ui, UiInput};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key as WinitKey, NamedKey};
use winit::window::{CursorIcon, Window};
#[cfg(windows)]
use winit::platform::windows::{CornerPreference, WindowAttributesExtWindows};

/// `V1_BOOT=1` prints when each step of starting up finished, in milliseconds since `main` began.
/// Off by default, and free then: the switch is read once.
pub fn boot_mark(label: &str) {
    static START: std::sync::OnceLock<Option<Instant>> = std::sync::OnceLock::new();
    let start = START.get_or_init(|| std::env::var_os("V1_BOOT").map(|_| Instant::now()));
    if let Some(start) = start {
        eprintln!("boot {:>7.1} ms  {label}", start.elapsed().as_secs_f64() * 1000.0);
    }
}

/// `V1_MEM=1` prints GPU allocations and working-set trims, to see the app's footprint.
fn mem_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("V1_MEM").is_some())
}

/// `V1_FPS=1` prints how many frames were presented each second, to see what the pacing costs.
fn fps_tick() {
    use std::cell::Cell;
    thread_local! {
        static WINDOW: Cell<Option<(Instant, u32)>> = const { Cell::new(None) };
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("V1_FPS").is_some()) {
        return;
    }
    WINDOW.with(|window| {
        let now = Instant::now();
        let (since, count) = window.get().unwrap_or((now, 0));
        if now.duration_since(since).as_secs_f32() >= 1.0 {
            eprintln!("fps {count}");
            window.set(Some((now, 0)));
        } else {
            window.set(Some((since, count + 1)));
        }
    });
}

fn data_dir() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("v1")
}

/// Events outside winit's own, delivered to the UI thread from background callbacks. The media
/// controls send their transport buttons this way, since SMTC raises them on a COM thread.
#[derive(Debug, Clone, Copy, PartialEq)]
enum UserEvent {
    Media(media::Command),
}

/// Everything the UI thread owns.
struct Host {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    ui: Option<Ui>,
    state: ui::State,
    /// The collection being read from disk on its own thread, while the window and the GPU start.
    /// Taken (and joined) before the first frame.
    library_task: Option<std::thread::JoinHandle<library::Library>>,
    /// The window opens hidden and is shown once its first frame has been drawn, so the user never
    /// sees an empty window while the interface warms up. False in `--shot`, which has its own idea
    /// of when the window shows.
    reveal: bool,
    keys: KeyFeed,
    /// Keyboard edges (typed text, navigation, clipboard chords) gathered since the last frame.
    pending: UiInput,

    /// The display's scale factor (device pixels per OS logical pixel).
    scale: f32,
    /// How much the whole UI is enlarged for the window it is in; see [`zoom_for`].
    zoom: f32,
    mouse: (f32, f32),
    down: bool,
    pressed: bool,
    released: bool,
    /// The right button was pressed since the last frame (it opens menus, see `ui::TrackMenu`).
    secondary_pressed: bool,
    wheel: f32,
    clock: clock::FrameClock,
    /// When the last frame began, for pacing the next one (see [`Host::pace`]).
    last_frame: Instant,
    /// The monitor the window is on, so a move is noticed without asking Windows for the refresh
    /// rate every frame: that query is too slow to run hundreds of times a second.
    display_monitor: Option<winit::monitor::MonitorHandle>,
    /// The refresh rate of [`display_monitor`](Self::display_monitor), in Hz. `None` when the
    /// platform does not report one.
    display_hz: Option<f32>,
    /// When the refresh rate was last read, so a changed display mode is still picked up.
    display_hz_at: Instant,
    hand_cursor: bool,
    /// The window is occluded (another window covers it fully) or minimized. Nothing visible needs
    /// drawing then, and drawing would only animate panels the user cannot see.
    occluded: bool,
    /// Working-set trim schedule: once after warm-up, then after each quiet period.
    memory: dream::memory::IdleTrim,
    /// The hidden state last acted on, so hiding/showing trims and re-arms the timer only once.
    hidden_applied: bool,
    /// Wakes the event loop when a background callback has work for the UI (SMTC buttons).
    proxy: EventLoopProxy<UserEvent>,
    /// The Windows media-controls session, once the window exists.
    media: Option<media::MediaControls>,
    /// A window drag in progress, if the host is moving the window (see [`Host::start_drag`]).
    drag: Option<WindowDrag>,
    /// `--shot <path>`: render one frame, save it, exit. With `--delay`, run first.
    shot: Option<PathBuf>,
    shot_delay: std::time::Duration,
    shoot_at: Option<Instant>,
}

/// A window drag the host carries out itself. The window keeps the offset it had from the pointer
/// when the drag began, so it moves by however far the pointer has travelled since.
#[derive(Clone, Copy)]
struct WindowDrag {
    /// Pointer position in screen pixels when the drag began.
    cursor: (i32, i32),
    /// Window position in screen pixels when the drag began.
    origin: (i32, i32),
}

/// The window is designed at 1280x800 logical pixels. Larger windows enlarge the whole UI (text,
/// icons, spacing) in proportion, so it keeps its balance instead of shrinking into a corner;
/// smaller ones shrink it a little before the layout starts to give way. The steps are 2% so a
/// resize does not churn the glyph cache with a new size every pixel.
fn zoom_for(logical_w: f32, logical_h: f32) -> f32 {
    let fit = (logical_w / 1280.0).min(logical_h / 800.0);
    ((fit.clamp(0.85, 1.75) * 50.0).round()) / 50.0
}

/// Build a frame, making any textures it asked for first. A frame that finds its icons missing
/// draws stand-ins, so when it asked for something the textures are made and it is built again,
/// with the one-shot input spent, so what is presented never has a hole.
fn make_frame(
    ui: &mut Ui,
    renderer: &mut Renderer,
    state: &mut ui::State,
    input: UiInput,
    pointer: ui::Pointer,
    size: (f32, f32),
    scale: f32,
) -> dream::ui::draw::UiFrame {
    state.gpu.sync(renderer.context());
    let frame = ui::build(ui, state, input.clone(), pointer, size.0, size.1, scale);
    if state.gpu.wanted_icons.is_empty() {
        return frame;
    }
    state.gpu.sync(renderer.context());
    ui.recycle_frame(frame);
    let again = UiInput {
        dt: 0.0001,
        mouse: input.mouse,
        mouse_down: input.mouse_down,
        ..Default::default()
    };
    let pointer = ui::Pointer { pressed: false, secondary: false, ..pointer };
    ui::build(ui, state, again, pointer, size.0, size.1, scale)
}

/// Translate one winit key into the UI's vocabulary.
fn map_key(key: &WinitKey) -> Vec<Key> {
    match key {
        WinitKey::Named(named) => match named {
            NamedKey::Enter => vec![Key::Enter],
            NamedKey::Escape => vec![Key::Escape],
            NamedKey::ArrowUp => vec![Key::Up],
            NamedKey::ArrowDown => vec![Key::Down],
            NamedKey::ArrowLeft => vec![Key::Left],
            NamedKey::ArrowRight => vec![Key::Right],
            NamedKey::Home => vec![Key::Home],
            NamedKey::End => vec![Key::End],
            NamedKey::PageUp => vec![Key::PageUp],
            NamedKey::PageDown => vec![Key::PageDown],
            NamedKey::Delete => vec![Key::Delete],
            NamedKey::Backspace => vec![Key::Backspace],
            NamedKey::Space => vec![Key::Char(' ')],
            NamedKey::Shift => vec![Key::Shift],
            NamedKey::Control => vec![Key::Control],
            NamedKey::Alt => vec![Key::Alt],
            NamedKey::Super => vec![Key::Super],
            _ => Vec::new(),
        },
        WinitKey::Character(text) => text.chars().map(Key::Char).collect(),
        _ => Vec::new(),
    }
}

impl ApplicationHandler<UserEvent> for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // `V1_SHOT_SIZE=1920x1080` sizes the window for screenshots.
        let (start_w, start_h) = std::env::var("V1_SHOT_SIZE")
            .ok()
            .and_then(|value| {
                let (w, h) = value.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?))
            })
            .unwrap_or((1280.0, 800.0));
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("V1")
                        .with_window_icon(app_icon())
                        .with_taskbar_icon(taskbar_icon())
                        .with_theme(Some(winit::window::Theme::Dark))
                        // The UI draws its own caption buttons and handles moving the window (see
                        // `ui::WindowRequest`), so the OS title bar goes. Resizing still works from
                        // the window edges.
                        .with_decorations(false)
                        // Windows 11 rounds the corners of the window itself; no shader work needed.
                        .with_corner_preference(CornerPreference::Round)
                        .with_visible(false)
                        .with_min_inner_size(winit::dpi::LogicalSize::new(720.0, 480.0))
                        .with_inner_size(winit::dpi::LogicalSize::new(start_w, start_h)),
                )
                .expect("create window"),
        );
        boot_mark("window created (hidden)");
        self.scale = window.scale_factor() as f32;
        // This app draws no 3D scene, so FSR has nothing to upscale and its sharpening only
        // fights the UI shader's own edge smoothing. Native resolution, no post AA.
        let renderer = Renderer::new(
            Arc::clone(&window),
            dream::DEFAULT_SCENE_WGSL,
            dream::settings::Settings {
                aa: dream::settings::AaMode::None,
                ..Settings::default()
            },
        );
        boot_mark("renderer ready");
        // `V1_MEM=1` prints the UI renderer's own GPU allocations, to keep an eye on the app-side
        // share of the process footprint (the Vulkan driver's own allocations are not visible here).
        if mem_trace() {
            let ui: u64 = renderer.ui_allocation_sizes().iter().map(|(_, bytes)| *bytes).sum();
            eprintln!("ui gpu allocations {:.1} MB", ui as f64 / (1024.0 * 1024.0));
        }
        self.renderer = Some(renderer);
        self.ui = Some(Ui::new(Fonts::bundled(), AtlasConfig::default()));
        boot_mark("fonts and ui ready");
        self.window = Some(window);
        // Know the refresh rate before the first frame, so startup motion is paced correctly.
        self.display_monitor = self.window.as_ref().and_then(|window| window.current_monitor());
        self.display_hz = self.display_monitor.as_ref().and_then(monitor_hz);
        // Windows routes earbud and keyboard transport buttons through SMTC, not through a Win32
        // window message, so a session is what makes them reach the app at all.
        let proxy = self.proxy.clone();
        self.media = self.window.as_ref().and_then(|window| {
            media::MediaControls::new(window, move |command| {
                let _ = proxy.send_event(UserEvent::Media(command));
            })
        });
        boot_mark("media controls attached");
        self.update_zoom();
        self.attach_library();
        boot_mark("collection attached");
        self.reveal = self.shot.is_none() || !self.shot_delay.is_zero();

        if let Some(path) = self.shot.clone() {
            if self.shot_delay.is_zero() {
                self.state.demo();
                self.shot_frame(event_loop, &path);
                return;
            }
            // A delayed shot runs the real loop first, so the frame can show a live search or
            // playback rather than the sample list.
            if let Ok(query) = std::env::var("V1_AUTOPLAY") {
                self.state.query = query;
                self.state.submit = true;
                self.state.autoplay_first();
            }
            self.shoot_at = Some(Instant::now() + self.shot_delay);
        }
        self.redraw(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                // Minimizing sends a zero-sized resize on Windows. It is not a layout the app should
                // adopt: at zero width the sidebar would animate shut, and restoring would replay
                // its opening. The real size arrives with the restore.
                if size.width == 0 || size.height == 0 || self.hidden() {
                    return;
                }
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                }
                self.update_zoom();
                self.redraw_request();
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    self.update_zoom();
                    self.redraw_request();
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if self.hidden() {
                    return;
                }
                self.scale = scale_factor as f32;
                if let Some(window) = &self.window {
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.resize(window.inner_size());
                    }
                }
                self.update_zoom();
                self.redraw_request();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale = self.ui_scale();
                self.mouse = (position.x as f32 / scale, position.y as f32 / scale);
                self.redraw_request();
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse = (-1.0, -1.0);
                self.redraw_request();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.down = true;
                        self.pressed = true;
                    }
                    ElementState::Released => {
                        self.down = false;
                        // The release that ends a window drag is the drag's, not a click's.
                        if self.drag.take().is_none() {
                            self.released = true;
                        }
                    }
                }
                self.redraw_request();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                self.secondary_pressed = true;
                self.redraw_request();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.wheel += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 3.0 * 40.0,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / self.ui_scale(),
                };
                self.redraw_request();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                self.keys
                    .modifiers(state.shift_key(), state.control_key(), state.alt_key());
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if pressed && self.app_shortcut(&event.logical_key) {
                    self.redraw_request();
                    return;
                }
                for key in map_key(&event.logical_key) {
                    // Enter submits the search only while a field is being edited.
                    if pressed && key == Key::Enter && self.editing() {
                        self.state.submit = true;
                    }
                    if pressed && key == Key::Escape {
                        self.state.close_menu();
                    }
                    if key != Key::Tab {
                        self.keys.key(&mut self.pending, key, pressed);
                    }
                }
                self.redraw_request();
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Media(command) => {
                self.apply_media_command(command);
                // The next frame shows the change, even when the window is hidden (the player
                // state still has to be applied); the hidden tick will present nothing.
                self.redraw_request();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let hidden = self.hidden();
        self.apply_hidden(hidden);
        // Nothing in the window can be seen while it is minimized or covered. Tick the player now
        // and then so the queue still advances, but do not build, animate or present frames: doing
        // that is what replayed the sidebar's opening when the window came back.
        if hidden {
            if self.state.fading() {
                // A transport fade survives the window being hidden; finish it without drawing.
                self.pace(event_loop, self.frame_period());
            } else if self.state.is_playing() || self.state.busy() || self.state.awaiting_covers() {
                self.pace(event_loop, HIDDEN_TICK);
            } else {
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            return;
        }
        if let Some(window) = &self.window {
            // `MonitorFromWindow` is cheap; `EnumDisplaySettingsExW`, which has the refresh rate, is
            // not. Only ask for it when the window has actually moved to another monitor.
            let monitor = window.current_monitor();
            let stale = now.duration_since(self.display_hz_at).as_secs_f32() >= 1.0;
            if monitor != self.display_monitor || stale {
                self.display_hz = monitor.as_ref().and_then(monitor_hz);
                self.display_monitor = monitor;
                self.display_hz_at = now;
            }
        }
        if let (Some(path), Some(at)) = (self.shot.clone(), self.shoot_at) {
            if now >= at {
                self.shot_frame(event_loop, &path);
                return;
            }
        }
        // Return the working set to the OS when the warm-up or a quiet period has elapsed. Run it
        // before choosing the wait so the deadline below is the next pending trim, not this one.
        self.idle_maintenance(now);
        // Motion (a transition, a spring, a window drag) is drawn at the monitor's refresh rate, so
        // it follows the pointer and the animations without a lag. Playback moves the progress bar,
        // the clock and the equalizer, which are motion too, so it gets the same rate.
        if self.state.animating() || self.drag.is_some() || self.state.is_playing() {
            self.pace(event_loop, self.frame_period());
        } else if self.state.busy() || self.state.awaiting_covers() {
            self.pace(event_loop, WAITING_FRAME);
        } else {
            // The lyric page may be waiting out a reader's scroll, the scrollbar may be about to
            // fade, the collection may have a file to clear or a download to retry, or a working-set
            // trim may be pending. Wake for the earliest of them; only the first three need a frame.
            let frame_wake = self
                .state
                .lyrics_resume_at()
                .into_iter()
                .chain(self.state.bars_fade_at())
                .chain(self.shoot_at)
                .chain(self.state.library_wake_at())
                .min();
            if frame_wake.is_some_and(|at| at <= now) {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            let wake = frame_wake.into_iter().chain(self.memory.next_deadline()).min();
            match wake {
                Some(at) => event_loop.set_control_flow(ControlFlow::WaitUntil(at)),
                None => event_loop.set_control_flow(ControlFlow::Wait),
            }
        }
    }
}

/// How often a frame is drawn while the app waits on the network. A floor on the time between
/// frames, not a target to reach. (Motion and playback use [`Host::frame_period`] instead.)
const WAITING_FRAME: std::time::Duration = std::time::Duration::from_millis(80);

/// How often the app ticks while its window is minimized or fully covered. Nothing is drawn; the
/// player still needs a poll now and then so a finished song moves on and timers fire.
const HIDDEN_TICK: std::time::Duration = std::time::Duration::from_millis(500);

/// The rate used when the platform does not report the monitor's refresh rate.
const FALLBACK_HZ: f32 = 60.0;

impl Host {
    /// Time between frames while something is moving or playing: one frame per refresh of the
    /// monitor the window is on, so motion follows the display instead of a fixed ceiling.
    fn frame_period(&self) -> std::time::Duration {
        let hz = self
            .display_hz
            .filter(|hz| hz.is_finite() && *hz > 1.0)
            .unwrap_or(FALLBACK_HZ);
        std::time::Duration::from_secs_f32(1.0 / hz.clamp(10.0, 1000.0))
    }

    /// Whether the window is minimized or fully occluded, so a frame would not be seen.
    fn hidden(&self) -> bool {
        self.occluded
            || self
                .window
                .as_ref()
                .and_then(|window| window.is_minimized())
                .unwrap_or(false)
    }

    /// Return the process working set to the OS, logging the change under `V1_MEM`.
    fn trim_memory(&self, label: &str) {
        let before = mem_trace().then(dream::memory::resident_bytes);
        dream::memory::trim_working_set();
        if let Some(before) = before {
            let after = dream::memory::resident_bytes();
            eprintln!(
                "memory {label} trim {:.1} -> {:.1} MB",
                before as f64 / (1024.0 * 1024.0),
                after as f64 / (1024.0 * 1024.0),
            );
        }
    }

    /// Act on a hide/show transition once. A hidden window shows nothing, so its cold pages go
    /// back to the OS now and the warm-up restarts when it shows again.
    fn apply_hidden(&mut self, hidden: bool) {
        if self.hidden_applied == hidden {
            return;
        }
        self.hidden_applied = hidden;
        if hidden {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.release_scene_resources();
                // The UI's upload buffers and glass backdrop textures go back too; the next
                // visible frame makes what it needs again.
                renderer.trim_memory();
            }
            self.trim_memory("hidden");
            self.memory = dream::memory::IdleTrim::new();
        } else {
            self.memory.activity(Instant::now());
            self.redraw_request();
        }
    }

    /// Run a scheduled working-set trim when one is due. The loop calls this as it parks, and
    /// schedules its next wake from `self.memory.next_deadline()`.
    fn idle_maintenance(&mut self, now: Instant) {
        if self.memory.trim_if_due(now) {
            self.trim_memory("idle");
        }
    }

    /// Ask for the next frame once `interval` has passed since the last began, and otherwise sleep
    /// until then. Requesting a redraw on every turn of the loop (even while also setting a wait)
    /// renders one frame per turn, because the request is answered at once.
    fn pace(&self, event_loop: &ActiveEventLoop, interval: std::time::Duration) {
        let now = Instant::now();
        let due = self.last_frame + interval;
        if now >= due {
            self.redraw_request();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(due.max(now)));
    }

    /// Push the current track to the Windows media controls. Does nothing when SMTC is unavailable.
    fn update_media(&mut self) {
        if let Some(media) = self.media.as_mut() {
            media.update(self.state.media_info().as_ref());
        }
    }

    /// Apply a transport command from SMTC or the fallback media keys. A hidden window cannot run
    /// the gain fade (it builds no frames), so there the change lands at once; the status is also
    /// pushed immediately so the flyout's next button press is based on the new state.
    fn apply_media_command(&mut self, command: media::Command) {
        let immediate = self.hidden();
        self.state.media_command(command, immediate);
        self.update_media();
    }

    /// Device pixels per UI unit: the display's scale times the window zoom.
    fn ui_scale(&self) -> f32 {
        self.scale * self.zoom
    }

    fn update_zoom(&mut self) {
        if let Some(window) = &self.window {
            let size = window.inner_size();
            // A minimized window reports a zero size; `zoom_for(0, 0)` would shrink the whole UI
            // and start every width-dependent animation over on restore.
            if size.width == 0 || size.height == 0 {
                return;
            }
            self.zoom = zoom_for(size.width as f32 / self.scale, size.height as f32 / self.scale);
        }
    }

    /// Take the collection the loading thread read, and hand it to the state. The state starts with
    /// an empty one so that nothing waited for the disk before the window could open.
    fn attach_library(&mut self) {
        let Some(task) = self.library_task.take() else { return };
        match task.join() {
            Ok(library) => self.state.attach_library(library),
            Err(_) => eprintln!("could not read the collection; starting with an empty one"),
        }
    }

    fn editing(&self) -> bool {
        self.ui.as_ref().is_some_and(|ui| ui.editing())
    }

    /// The app's own shortcuts. Returns true if the key was one, so it is not typed anywhere.
    fn app_shortcut(&mut self, key: &WinitKey) -> bool {
        let command = self.keys.command();
        match key {
            WinitKey::Character(text) if command && text.eq_ignore_ascii_case("k") => {
                self.state.focus_search = true;
                true
            }
            WinitKey::Character(text) if command && text.eq_ignore_ascii_case("b") => {
                self.state.sidebar_open = !self.state.sidebar_open;
                true
            }
            WinitKey::Character(text) if !command && !self.editing() && text == "/" => {
                self.state.focus_search = true;
                true
            }
            WinitKey::Named(NamedKey::Space) if !command && !self.editing() => {
                self.state.toggle_request = true;
                true
            }
            // Keyboard media keys that Windows delivers as ordinary key events. Most earbuds go
            // through SMTC instead, and when an SMTC session exists the OS may deliver the key both
            // ways; guarding on SMTC avoids toggling the song twice for one press.
            WinitKey::Named(NamedKey::MediaPlayPause) if self.media.is_none() => {
                self.apply_media_command(media::Command::Toggle);
                true
            }
            WinitKey::Named(NamedKey::MediaTrackNext) if self.media.is_none() => {
                self.apply_media_command(media::Command::Next);
                true
            }
            WinitKey::Named(NamedKey::MediaTrackPrevious) if self.media.is_none() => {
                self.apply_media_command(media::Command::Previous);
                true
            }
            WinitKey::Named(NamedKey::MediaStop) if self.media.is_none() => {
                self.apply_media_command(media::Command::Stop);
                true
            }
            _ => false,
        }
    }

    /// `--shot`: build one frame, read it back and write a PNG, then exit.
    ///
    /// For checking motion, `V1_SHOT_SEQ=1,4,10` instead builds consecutive 60 Hz frames, writes
    /// the listed ones as `<path>-001.png` and so on, and `V1_SHOT_HOVER=x,y` is where the pointer
    /// moves to after a settled first frame. `V1_SHOT_COLLAPSE=1` folds the sidebar at that moment.
    fn shot_frame(&mut self, event_loop: &ActiveEventLoop, path: &Path) {
        // Once is enough: `exit` only takes effect on the next loop turn.
        self.shoot_at = None;
        self.shot = None;
        let scale = self.ui_scale();
        let (Some(ui), Some(renderer)) = (self.ui.as_mut(), self.renderer.as_mut()) else {
            event_loop.exit();
            return;
        };
        // Frames are composed straight into the swapchain unless capture is on, and a shot reads
        // the frame back, so keep them.
        renderer.set_frame_capture(true);
        let (physical_w, physical_h) = renderer.display_size();
        let width = physical_w as f32 / scale;
        let height = physical_h as f32 / scale;
        ui.set_scale(scale);

        let hover = std::env::var("V1_SHOT_HOVER").ok().and_then(|value| {
            let (x, y) = value.split_once(',')?;
            Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
        });
        let sequence: Option<Vec<usize>> = std::env::var("V1_SHOT_SEQ").ok().map(|list| {
            list.split(',')
                .filter_map(|part| part.trim().parse().ok())
                .collect()
        });

        match sequence {
            None => {
                let pos = hover.unwrap_or(self.mouse);
                shot_step(ui, renderer, &mut self.state, pos, 0.016, (width, height), scale, Some(path));
            }
            Some(frames) => {
                // Frame 0 settles every spring at its resting value with the pointer parked.
                shot_step(ui, renderer, &mut self.state, (-1.0, -1.0), 1.0 / 60.0, (width, height), scale, None);
                if std::env::var_os("V1_SHOT_COLLAPSE").is_some() {
                    self.state.sidebar_open = false;
                }
                if let Ok(tab) = std::env::var("V1_SHOT_SWITCH") {
                    self.state.shot_switch_tab(&tab);
                }
                let last = frames.iter().copied().max().unwrap_or(0);
                let stem = path.with_extension("");
                for frame in 1..=last {
                    let out = frames.contains(&frame).then(|| {
                        PathBuf::from(format!("{}-{frame:03}.png", stem.display()))
                    });
                    shot_step(
                        ui,
                        renderer,
                        &mut self.state,
                        hover.unwrap_or((-1.0, -1.0)),
                        1.0 / 60.0,
                        (width, height),
                        scale,
                        out.as_deref(),
                    );
                }
            }
        }
        event_loop.exit();
    }

    fn redraw_request(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    /// Do what the UI asked of the window (see [`ui::WindowRequest`]).
    fn window_request(&mut self, event_loop: &ActiveEventLoop, request: ui::WindowRequest) {
        let Some(window) = self.window.clone() else { return };
        match request {
            ui::WindowRequest::Minimize => window.set_minimized(true),
            ui::WindowRequest::Close => event_loop.exit(),
            ui::WindowRequest::Drag => {
                self.start_drag(&window);
                // The drag takes the button's release for itself (see `ElementState::Released`),
                // so the UI must not see the button as held.
                self.down = false;
            }
        }
    }

    /// Begin moving the window with the pointer.
    ///
    /// The host does the move itself rather than calling `drag_window`. That hands the move to the
    /// OS, whose modal loop takes over the thread: winit's loop stops running, so no frame is drawn
    /// until the button comes up, and every animation freezes for the length of the drag.
    fn start_drag(&mut self, window: &Window) {
        match (cursor_screen_pos(), window.outer_position()) {
            (Some(cursor), Ok(origin)) => {
                self.drag = Some(WindowDrag { cursor, origin: (origin.x, origin.y) });
            }
            _ => {
                let _ = window.drag_window();
            }
        }
    }

    /// Move the window to where the pointer's travel since the drag began puts it. Ends the drag
    /// once no mouse button is held, in case the release was missed.
    fn follow_drag(&mut self) {
        let Some(drag) = self.drag else { return };
        if !any_mouse_button_down() {
            self.drag = None;
            return;
        }
        let (Some(window), Some((x, y))) = (self.window.as_ref(), cursor_screen_pos()) else {
            return;
        };
        window.set_outer_position(PhysicalPosition::new(
            drag.origin.0 + x - drag.cursor.0,
            drag.origin.1 + y - drag.cursor.1,
        ));
    }

    /// One frame: pump the backend, advance the player, build and present.
    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        self.last_frame = Instant::now();
        let dt = self.clock.tick(self.last_frame);

        self.follow_drag();
        self.state.pump();
        self.state.tick();
        self.update_media();
        // Hidden: the player and the backend still advance, but there is no reason to lay out,
        // animate or present anything the user cannot see.
        if self.hidden() {
            self.pressed = false;
            self.released = false;
            self.secondary_pressed = false;
            self.wheel = 0.0;
            return;
        }

        self.update_zoom();
        let scale = self.ui_scale();
        let (Some(ui), Some(renderer)) = (self.ui.as_mut(), self.renderer.as_mut()) else {
            return;
        };
        let (physical_w, physical_h) = renderer.display_size();
        let width = physical_w as f32 / scale;
        let height = physical_h as f32 / scale;

        let mut input = std::mem::take(&mut self.pending);
        input.dt = dt;
        input.mouse = self.mouse;
        input.mouse_down = self.down;
        input.mouse_pressed = self.pressed;
        input.mouse_released = self.released;
        input.wheel = std::mem::take(&mut self.wheel);
        // Held modifiers must be re-applied each frame; edges were gathered between frames.
        self.keys.apply(&mut input);

        ui.set_scale(scale);
        let pointer = ui::Pointer {
            pos: self.mouse,
            down: self.down,
            pressed: self.pressed,
            secondary: self.secondary_pressed,
        };
        let frame = make_frame(ui, renderer, &mut self.state, input, pointer, (width, height), scale);
        if let Err(error) = renderer.render_ui_only(Some(&frame)) {
            match error {
                // A stale swapchain is normal after minimize/restore or a display change. Rebuild
                // it and ask for one more frame; a timed-out frame is simply dropped.
                dream::gfx::SurfaceError::Outdated => {
                    renderer.recreate_swapchain();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                dream::gfx::SurfaceError::Timeout => {}
                error => eprintln!("render failed: {error:?}"),
            }
        }
        fps_tick();
        ui.recycle_frame(frame);
        ui.flush_clipboard();
        // A frame counts as activity: the working set is only trimmed once the screen goes quiet.
        self.memory.activity(Instant::now());

        self.pressed = false;
        self.released = false;
        self.secondary_pressed = false;

        if self.state.hand != self.hand_cursor {
            self.hand_cursor = self.state.hand;
            if let Some(window) = &self.window {
                window.set_cursor(if self.hand_cursor {
                    CursorIcon::Pointer
                } else {
                    CursorIcon::Default
                });
            }
        }

        // The first frame is on the swapchain: now the window can appear with something in it. A
        // second frame is asked for, since presenting to a hidden window may not have reached the
        // screen.
        if std::mem::take(&mut self.reveal) {
            boot_mark("first frame presented, window shown");
            if let Some(window) = &self.window {
                window.set_visible(true);
                window.request_redraw();
            }
        }

        // After the frame is presented, so a drag starts from a settled picture.
        for request in std::mem::take(&mut self.state.window_requests) {
            self.window_request(event_loop, request);
        }
    }
}

/// A monitor's refresh rate in Hz, if the platform reports one.
fn monitor_hz(monitor: &winit::monitor::MonitorHandle) -> Option<f32> {
    monitor
        .refresh_rate_millihertz()
        .map(|millihertz| millihertz as f32 / 1000.0)
}

/// Build and present one frame for `--shot`, optionally saving it.
#[allow(clippy::too_many_arguments)]
fn shot_step(
    ui: &mut Ui,
    renderer: &mut Renderer,
    state: &mut ui::State,
    pos: (f32, f32),
    dt: f32,
    size: (f32, f32),
    scale: f32,
    out: Option<&Path>,
) {
    let input = UiInput {
        dt,
        mouse: pos,
        ..Default::default()
    };
    let pointer = ui::Pointer {
        pos,
        down: false,
        pressed: false,
        secondary: false,
    };
    let frame = make_frame(ui, renderer, state, input, pointer, size, scale);
    let _ = renderer.render_ui_only(Some(&frame));
    if let Some(path) = out {
        let (pixels, width, height) = renderer.capture_rgba8();
        match dream::png::write(path, width, height, &pixels) {
            Ok(()) => println!("wrote {} ({width}x{height})", path.display()),
            Err(error) => eprintln!("could not write {}: {error}", path.display()),
        }
    }
    ui.recycle_frame(frame);
}

/// The taskbar's big icon, at the size Windows draws it.
fn taskbar_icon() -> Option<winit::window::Icon> {
    let png = include_bytes!("../assets/icon/app-48.png");
    let rgba = image::load_from_memory_with_format(png, image::ImageFormat::Png).ok()?.to_rgba8();
    let (w, h) = rgba.dimensions();
    winit::window::Icon::from_rgba(rgba.into_raw(), w, h).ok()
}

/// The window and taskbar icon: the 256 px render of the app icon, decoded once per window.
fn app_icon() -> Option<winit::window::Icon> {
    let png = include_bytes!("../assets/icon/app-256.png");
    let rgba = image::load_from_memory_with_format(png, image::ImageFormat::Png).ok()?.to_rgba8();
    let (w, h) = rgba.dimensions();
    winit::window::Icon::from_rgba(rgba.into_raw(), w, h).ok()
}

#[cfg(windows)]
#[repr(C)]
struct NativePoint {
    x: i32,
    y: i32,
}

#[cfg(windows)]
#[link(name = "user32")]
extern "system" {
    fn GetCursorPos(point: *mut NativePoint) -> i32;
    fn GetAsyncKeyState(key: i32) -> i16;
}

/// The pointer in screen pixels. The process is per-monitor DPI aware (winit sets it), so these
/// match the physical positions winit uses for the window.
#[cfg(windows)]
fn cursor_screen_pos() -> Option<(i32, i32)> {
    let mut point = NativePoint { x: 0, y: 0 };
    // SAFETY: GetCursorPos writes one POINT through the pointer, which lives for the call.
    let ok = unsafe { GetCursorPos(&mut point) };
    (ok != 0).then_some((point.x, point.y))
}

/// Whether either mouse button is down. Both are checked because swapped buttons move the
/// logical left button to the physical right one.
#[cfg(windows)]
fn any_mouse_button_down() -> bool {
    const VK_LBUTTON: i32 = 0x01;
    const VK_RBUTTON: i32 = 0x02;
    // SAFETY: GetAsyncKeyState takes a virtual-key code and no pointers. The high bit is set
    // while the key is down, so the OR of two results is negative if either is down.
    let held = unsafe { GetAsyncKeyState(VK_LBUTTON) | GetAsyncKeyState(VK_RBUTTON) };
    held < 0
}

#[cfg(not(windows))]
fn cursor_screen_pos() -> Option<(i32, i32)> {
    None
}

#[cfg(not(windows))]
fn any_mouse_button_down() -> bool {
    false
}

fn main() {
    boot_mark("main");
    #[cfg(windows)]
    identity::apply();
    let arguments: Vec<String> = std::env::args().collect();
    let smoke_mode = arguments.iter().any(|argument| argument == "--smoke");
    let shot_mode = arguments.iter().any(|argument| argument == "--shot");
    let updated = arguments.iter().any(|argument| argument == update::UPDATED_FLAG);
    if smoke_mode {
        smoke();
        return;
    }
    // One running copy per session. A second launch exits silently. `--shot` builds one frame and
    // exits by itself, so it stays outside the guard and can be used for screenshots.
    let _instance = if shot_mode {
        None
    } else {
        match instance::acquire() {
            instance::Acquire::First(guard) => Some(guard),
            instance::Acquire::AlreadyRunning => return,
            instance::Acquire::Unavailable => None,
        }
    };
    // A handoff marker covers the gap while the installer replaces this build. Launches of the old
    // build exit silently; the installer's `--updated` launch claims the handoff and continues.
    if !shot_mode && update::blocked_by_update(updated) {
        return;
    }
    // An available update is installed and the app restarts, so the user never sees the old
    // version. `--shot` renders a frame, so it skips it. The check runs on its own thread: a slow
    // release lookup must not delay the window opening.
    if !shot_mode {
        update::check_in_background();
    }
    let shot = arguments
        .iter()
        .position(|argument| argument == "--shot")
        .and_then(|index| arguments.get(index + 1))
        .map(PathBuf::from);
    let shot_delay = arguments
        .iter()
        .position(|argument| argument == "--delay")
        .and_then(|index| arguments.get(index + 1))
        .and_then(|seconds| seconds.parse::<f32>().ok())
        .map(std::time::Duration::from_secs_f32)
        .unwrap_or_default();
    // The collection is read (and its folders swept) on a thread of its own, beside the window and
    // the GPU coming up. The network thread starts inside `State::new`, so all three run at once.
    let library_task = std::thread::Builder::new()
        .name("v1-library".to_owned())
        .spawn(|| library::Library::open(data_dir()));
    let mut state = ui::State::new(data_dir());
    let library_task = match library_task {
        Ok(task) => Some(task),
        Err(_) => {
            // No thread to spare: read it here, as before. A collection must never be lost to this.
            state.attach_library(library::Library::open(data_dir()));
            None
        }
    };
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("create event loop");
    let proxy = event_loop.create_proxy();
    let host = Host {
        window: None,
        renderer: None,
        ui: None,
        state,
        library_task,
        reveal: false,
        keys: KeyFeed::default(),
        pending: UiInput::default(),
        scale: 1.0,
        zoom: 1.0,
        mouse: (-1.0, -1.0),
        down: false,
        pressed: false,
        released: false,
        secondary_pressed: false,
        wheel: 0.0,
        clock: clock::FrameClock::new(Instant::now()),
        last_frame: Instant::now(),
        display_monitor: None,
        display_hz: None,
        display_hz_at: Instant::now(),
        hand_cursor: false,
        occluded: false,
        memory: dream::memory::IdleTrim::new(),
        hidden_applied: false,
        proxy,
        media: None,
        drag: None,
        shot,
        shot_delay,
        shoot_at: None,
    };
    event_loop.run_app(&mut { host }).expect("run the app");
}

/// `--smoke`: run search → resolve → proxy → decode → play without a window, then exit.
///
/// This is the end-to-end check that the UI cannot make on its own: it proves the network path,
/// the loopback proxy and the decoder all agree, and it prints where the time goes.
fn smoke() {
    use std::time::Duration;
    use yt_client::TransportConfig;
    use yt_js::Player;
    use yt_proxy::{Config as ProxyConfig, Proxy};
    use yt_stream::{FileCache, HttpProber, Resolver, ResolverConfig, YtScripts};

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let started = Instant::now();
        let (tube, web) = yt_client::connect(TransportConfig::default(), Default::default())
            .await
            .expect("connect");
        println!("connect          {:?}", started.elapsed());

        let http = reqwest::Client::builder().build().expect("http");
        let player = Player::load(
            &http,
            &data_dir().join("player"),
            web.player_js_path.as_deref().expect("player script"),
        )
        .await
        .expect("player script");
        println!("player script    {:?}", started.elapsed());

        let proxy = Proxy::start(
            reqwest::Client::builder()
                .no_proxy()
                .http1_only()
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("upstream"),
            ProxyConfig::default(),
        )
        .await
        .expect("proxy");

        let resolver = Resolver::new(
            tube.clone(),
            YtScripts::new(player),
            HttpProber::new(
                reqwest::Client::builder().no_proxy().http1_only().build().expect("prober"),
            ),
            FileCache::open(data_dir().join("streams.json")),
            ResolverConfig::default(),
        );

        let hits = tube.search_songs("Daft Punk", false).await.expect("search");
        let hit = hits.first().expect("a result");
        println!("search           {:?}  {}", started.elapsed(), hit.title);

        let resolved = resolver.resolve(&hit.video_id).await.expect("resolve");
        println!(
            "resolve          {:?}  {:?} itag {}",
            started.elapsed(),
            resolved.client,
            resolved.itag
        );

        let url = proxy.register(&resolved.url, resolved.headers.clone());
        let opening = Instant::now();
        // The same open the app uses for a streamed track: no probe, the stream is known to be audio.
        // `V1_SMOKE_PROBE=1` opens it the old way (ffprobe first), for comparing the two.
        let options = dream_media::PlayerOptions {
            audio: true,
            max_size: (0, 0),
            ..Default::default()
        };
        let mut track = if std::env::var_os("V1_SMOKE_PROBE").is_some() {
            dream_media::Player::open(&url, options)
        } else {
            dream_media::Player::open_audio_unprobed(&url, options, None)
        }
        .expect("open stream");
        println!("open stream      {:?}", opening.elapsed());

        let playing = Instant::now();
        let mut first_sound = None;
        while playing.elapsed() < Duration::from_secs(5) {
            let _ = track.poll(Instant::now());
            if first_sound.is_none() && track.position_ms(Instant::now()) > 0 {
                first_sound = Some(opening.elapsed());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        println!("first sound      {:?} after the open began", first_sound);
        println!(
            "after 5 s        status {:?}  position {:.2}s of {:?}s",
            track.status(),
            track.position_ms(Instant::now()) as f64 / 1000.0,
            track.duration_ms().map(|ms| ms as f64 / 1000.0)
        );
        if let Some(warning) = track.warning() {
            println!("warning          {warning}");
        }
    });
}
