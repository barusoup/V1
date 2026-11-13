//! Dream Studio — the engine's editor app.
//!
//! ```text
//! cargo run --release --bin dream_app                    # the interactive app
//! cargo run --release --bin dream_app -- --selfcheck     # scripted shots + soak
//! cargo run --release --bin dream_app -- --pace-probe 3  # measure frame cadence
//! ```
//!
//! `DREAM_FPS_CAP=<hz>` caps the presented rate app-wide.
//!
//! The window is borderless: the app draws its own title bar and caption
//! buttons (see `app::chrome::window_bar`) and asks this layer to move,
//! resize, minimise, maximise or close through [`WindowCommand`]s. The window
//! itself is rounded by the compositor where the platform supports it. The
//! frame loop is on-demand — a frame is requested only when something can
//! have changed, so a still screen renders at 0 Hz and any input wakes it.
//! A completely hidden window — minimised, or fully occluded by other
//! windows where the platform reports either — renders nothing at all and
//! parks the loop until it shows again.

use dream::app::{App, Key, PointerButton, ResizeEdge, WindowCommand, WindowCursor};
use dream::settings::Settings;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key as WinitKey, KeyCode, PhysicalKey};
use winit::window::{CursorIcon, ResizeDirection, Theme, Window, WindowId};

struct Studio {
    app: Option<App>,
    window: Option<Arc<Window>>,
    selfcheck: bool,
    /// `--pace-probe [seconds]`: run the real loop with continuous motion,
    /// measure the cadence it actually achieves, print it and exit. This is
    /// the only way to measure pacing honestly — the loop's rate depends on
    /// the OS wake-up timer and on how long a frame takes, neither of which a
    /// scripted soak can stand in for.
    pace_probe: Option<f32>,
    probe: Probe,
    last_cursor: (f32, f32),
    cursor_present: bool,
    /// The platform says the window is fully covered by other windows
    /// (`WindowEvent::Occluded`; macOS and X11 report it). Folded into the
    /// app's hidden state together with minimisation.
    occluded: bool,
    last_cursor_icon: Option<CursorIcon>,
    /// Cached monitor refresh rate, so cursor-driven wake-ups do not query
    /// the display settings on every pass.
    monitor: MonitorRate,
}

/// Inter-frame interval and frame-cost samples from `--pace-probe`.
struct Probe {
    started: Option<std::time::Instant>,
    last_present: std::time::Instant,
    intervals: Vec<f32>,
    costs: Vec<f32>,
}

impl Default for Probe {
    fn default() -> Self {
        Self {
            started: None,
            // Stands in as the origin for the first interval only, which
            // `report` discards (it spans event-loop creation to first frame).
            last_present: std::time::Instant::now(),
            intervals: Vec::new(),
            costs: Vec::new(),
        }
    }
}

impl Probe {
    /// Record one presented frame and report whether the probe is done.
    fn sample(&mut self, now: std::time::Instant, cost: std::time::Duration, seconds: f32) -> bool {
        let started = *self.started.get_or_insert(now);
        self.intervals
            .push(now.duration_since(self.last_present).as_secs_f32());
        self.last_present = now;
        self.costs.push(cost.as_secs_f32());
        now.duration_since(started).as_secs_f32() >= seconds
    }

    fn report(&self, summary: &str, target: Option<std::time::Duration>, immediate: u64) {
        // The first interval spans the event loop's creation to the first
        // frame, not two frames, so it is not a sample.
        let mut intervals = self.intervals.iter().copied().skip(1).collect::<Vec<_>>();
        if intervals.is_empty() {
            println!("[pace] {summary}: no frames sampled");
            return;
        }
        let span = intervals.iter().sum::<f32>();
        let frames = intervals.len() as f32;
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |q: f32| intervals[((intervals.len() - 1) as f32 * q) as usize];
        let mut costs = self.costs.iter().copied().skip(1).collect::<Vec<_>>();
        costs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cost_p50 = costs.get(costs.len() / 2).copied().unwrap_or(0.0);
        let cost_max = costs.last().copied().unwrap_or(0.0);
        println!("[pace] {summary}");
        println!(
            "[pace] {frames:.0} frames in {span:.2} s -> {:.1} fps (mean {:.2} ms)",
            frames / span,
            span / frames * 1000.0,
        );
        // The median is the honest "how long is a frame" number: input
        // answers the instant it arrives (a frame far shorter than the
        // period), and a stall shows up as one long interval.
        let (mut early, mut dropped) = (0, 0);
        if let Some(period) = target.map(|period| period.as_secs_f32()) {
            for ms in &intervals {
                if *ms < period * 0.9 {
                    early += 1;
                } else if *ms > period * 1.5 {
                    dropped += 1;
                }
            }
        }
        println!(
            "[pace] interval ms: min {:.2} p50 {:.2} p95 {:.2} max {:.2} | off-cadence {early} dropped {dropped} | immediate {immediate}",
            intervals[0] * 1000.0,
            at(0.5) * 1000.0,
            at(0.95) * 1000.0,
            intervals[intervals.len() - 1] * 1000.0,
        );
        println!(
            "[pace] frame cost ms: p50 {:.2} max {:.2}",
            cost_p50 * 1000.0,
            cost_max * 1000.0,
        );
    }
}

impl Studio {
    fn new(selfcheck: bool, pace_probe: Option<f32>) -> Self {
        Self {
            app: None,
            window: None,
            selfcheck,
            pace_probe,
            probe: Probe::default(),
            last_cursor: (0.0, 0.0),
            cursor_present: false,
            occluded: false,
            last_cursor_icon: None,
            monitor: MonitorRate::new(),
        }
    }
}

/// The Dream logo as the window icon (taskbar and task switcher). `None`
/// if the embedded PNG fails to decode, which leaves the OS default in place.
fn app_icon() -> Option<winit::window::Icon> {
    let logo = image::load_from_memory(include_bytes!("../../assets/branding/dream_icon.png"))
        .ok()?
        .into_rgba8();
    let (width, height) = logo.dimensions();
    winit::window::Icon::from_rgba(logo.into_raw(), width, height).ok()
}

impl ApplicationHandler for Studio {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Dream Studio")
                        .with_window_icon(app_icon())
                        // The app owns its chrome: no OS decorations.
                        .with_decorations(false)
                        // Logical units: on a HiDPI display this asks for a
                        // 1600x900 point window (3200x1800 physical at 2x).
                        .with_inner_size(winit::dpi::LogicalSize::new(1600.0f64, 900.0f64))
                        .with_visible(!self.selfcheck),
                )
                .expect("create window"),
        );
        // Text composition is opt-in per window; the app keeps preedit
        // uncommitted and places the candidate window at the focused field.
        window.set_ime_allowed(true);
        self.window = Some(Arc::clone(&window));
        let rounded = round_window_corners(&window);
        if self.selfcheck {
            println!(
                "[selfcheck] window corners: {}",
                if rounded {
                    "rounded (DWM accepted the preference)"
                } else {
                    "square (not supported on this OS)"
                }
            );
        }
        let system_theme = window.theme();
        let mut app = App::new(window, Settings::load());
        // The window reports the OS appearance; the app's `System` theme
        // choice follows it, and keeps following it through `ThemeChanged`.
        app.set_system_dark(system_theme.map(|theme| theme == Theme::Dark));
        if self.selfcheck {
            app.pacing
                .set_display_hz(self.window.as_deref().and_then(monitor_hz));
            println!("[selfcheck] pacing: {}", app.pacing.summary());
            app.selfcheck();
            #[cfg(target_os = "macos")]
            app.renderer.context().wait_idle();
            #[cfg(not(target_os = "macos"))]
            unsafe {
                app.renderer.context().device().device_wait_idle().unwrap();
            }
            event_loop.exit();
            return;
        }
        if self.pace_probe.is_some() {
            // Continuous motion: the debug scene spins, so the loop is never allowed
            // to park and every frame is owed to the display.
            app.toggle_debug();
        }
        self.app = Some(app);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ThemeChanged(theme) => {
                // Only visible while the theme choice is `System`.
                app.set_system_dark(Some(theme == Theme::Dark));
            }
            WindowEvent::Resized(size) => {
                app.renderer.resize(size);
                // Layout and pointer space are logical: rebuild them from
                // the new physical size at the current scale — and drop the
                // remembered cursor so the next delta is not a size jump.
                app.refresh_screen();
                app.dirty = true;
                self.cursor_present = false;
                self.monitor.invalidate();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                app.set_scale(scale_factor as f32);
                // A density change arrives with a matching Resized on most
                // platforms; rebuilding now also covers the ones that only
                // report the factor.
                app.renderer.resize(
                    self.window
                        .as_ref()
                        .map(|w| w.inner_size())
                        .unwrap_or_default(),
                );
                app.refresh_screen();
                app.dirty = true;
                self.cursor_present = false;
                self.monitor.invalidate();
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Physical pixels -> logical, exactly once.
                let position = position.to_logical::<f64>(app.scale as f64);
                let cursor = (position.x as f32, position.y as f32);
                if self.cursor_present {
                    let dx = cursor.0 - self.last_cursor.0;
                    let dy = cursor.1 - self.last_cursor.1;
                    app.viewport_drag(dx, dy);
                }
                self.last_cursor = cursor;
                self.cursor_present = true;
                app.pointer_moved(cursor);
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor_present = false;
                app.pointer_moved((-1.0, -1.0));
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let button = match button {
                    MouseButton::Left => PointerButton::Left,
                    MouseButton::Right => PointerButton::Right,
                    MouseButton::Middle => PointerButton::Middle,
                    _ => return,
                };
                app.pointer_button(button, state == ElementState::Pressed);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Positive scrolls up; the app's consumers each apply the
                // standard convention (zoom in / scroll to earlier content).
                // Pixel deltas are physical — converted to logical once.
                //
                // Scroll views get real pixels: a line delta is scaled by
                // the scroll config's line height (one notch is three lines
                // on most platforms), and a pixel delta is already the
                // gesture's own distance, marked precise so it tracks the
                // device 1:1 instead of gliding to a target.
                let line_height = app.ui.scroll_config.line_height;
                let (lines, pixels, precise) = match delta {
                    MouseScrollDelta::LineDelta(_, y) => (y * 3.0, y * line_height, false),
                    MouseScrollDelta::PixelDelta(px) => {
                        let dy = px.to_logical::<f64>(app.scale as f64).y as f32;
                        (dy / line_height, dy, true)
                    }
                };
                app.wheel_scroll(lines, pixels, precise);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                // The platform shortcut modifier: Command on macOS,
                // Control elsewhere. App shortcuts only see `ctrl_held`.
                #[cfg(target_os = "macos")]
                let command = state.super_key();
                #[cfg(not(target_os = "macos"))]
                let command = state.control_key();
                app.modifiers(state.shift_key(), command, state.alt_key());
            }
            WindowEvent::Occluded(occluded) => {
                // Fully covered by other windows (macOS and X11 report
                // this; Windows and Wayland have no such event). The app
                // folds it into its hidden state in `about_to_wait`, which
                // parks the loop until the window shows again.
                self.occluded = occluded;
            }
            WindowEvent::Focused(focused) => {
                if !focused {
                    // Keys/buttons held on the way out are gone: drop every
                    // held edge, drag and composition.
                    app.cancel_input();
                    self.cursor_present = false;
                }
            }
            WindowEvent::Ime(ime) => match ime {
                winit::event::Ime::Commit(text) => {
                    app.ime_preedit(String::new(), None);
                    app.text_input(&text);
                }
                winit::event::Ime::Preedit(text, cursor) => {
                    app.ime_preedit(text, cursor);
                }
                winit::event::Ime::Disabled => {
                    app.ime_preedit(String::new(), None);
                }
                _ => {}
            },
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                // While a composition is in flight, character events belong
                // to it — committing them too would double the text.
                // The whole string commits (keyboard layouts produce
                // multi-codepoint strings); key_event routes each char to
                // shortcuts or text as its modifiers demand.
                //
                // The text comes from `logical_key.to_text()`, not a
                // `Key::Character` match: winit reports the space bar as
                // `Named(NamedKey::Space)` on every platform, so a
                // Character-only match never types a space — it falls
                // through to `map_physical`, which knows no `Space`, and
                // dies there. Named control keys (Enter/Tab/Backspace/
                // Escape) spell out control characters, are skipped below
                // and still reach `map_physical` exactly as before.
                if !app.composing()
                    && let Some(text) = typed_text(&event.logical_key)
                {
                    let mut forwarded = false;
                    for ch in text.chars() {
                        if ch.is_control() {
                            continue;
                        }
                        app.key_event(Key::Char(ch), pressed);
                        forwarded = true;
                    }
                    if forwarded {
                        return;
                    }
                }
                if let Some(key) = map_physical(&event.physical_key) {
                    app.key_event(key, pressed);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let started = std::time::Instant::now();
                let rendered_before = app.frames_rendered;
                app.tick(now);
                drain_window_commands(
                    app,
                    self.window.as_ref(),
                    event_loop,
                    &mut self.last_cursor_icon,
                );
                if let (Some(seconds), true) =
                    (self.pace_probe, app.frames_rendered > rendered_before)
                {
                    // Only frames the loop actually presented are intervals.
                    let done = self.probe.sample(now, started.elapsed(), seconds);
                    if done {
                        self.probe.report(
                            &app.pacing.summary(),
                            app.pacing.period(),
                            app.frames_immediate,
                        );
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::Moved(_) => self.monitor.invalidate(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(app) = self.app.as_mut() else {
            return;
        };
        if let Some(window) = &self.window {
            // The window bar's maximize glyph mirrors the real window
            // state, and continuous animation paces to the monitor's
            // refresh (never rendering faster than the display). The rate is
            // re-read here every pass, so moving the window to another
            // monitor, changing a monitor's rate or unplugging one is picked
            // up on the next frame rather than at startup.
            app.window_maximized = window.is_maximized();
            app.pacing
                .set_display_hz(self.monitor.get(window, std::time::Instant::now()));
            // A completely hidden window owes no frames: none it builds can
            // be seen. Minimisation is polled here — the platform has no
            // event for it, and Wayland never answers (`None` keeps the
            // normal on-demand behaviour) — and full occlusion arrives as
            // `Occluded` where the platform can see it. `set_hidden` acts on
            // the transition only, so the show repaints immediately.
            app.set_hidden(self.occluded || window.is_minimized().unwrap_or(false));
        }
        // Schedule only: the frame itself decides. Asking the app whether a
        // frame is due here *and* again inside `tick` would spend the frame
        // twice — the schedule question is a pure query, and `needs_render`
        // moves the pacing grid when it answers yes. The wake may also be a
        // pending memory trim, which runs here rather than in a frame.
        let now = std::time::Instant::now();
        // Probe and selfcheck runs skip trims, so they must not wake for one
        // either: an unclaimed trim deadline would keep the loop redrawing.
        let maintain = self.pace_probe.is_none() && !self.selfcheck;
        if maintain {
            app.idle_maintenance(now);
        }
        let wake = if maintain {
            app.next_wake_at(now)
        } else {
            app.next_frame_at(now)
        };
        match wake {
            // Due (or overdue): get a frame built now.
            Some(due) if due <= now => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            // Self-sustaining motion still owes the display its next paced
            // frame: wake exactly when it is due.
            Some(due) => event_loop.set_control_flow(ControlFlow::WaitUntil(due)),
            // A truly still screen parks until an event arrives (0 Hz).
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}

/// The monitor refresh rate, re-read at most every `MonitorRate::MAX_AGE` or when invalidated (the window moved, resized or changed scale).
struct MonitorRate {
    hz: Option<f32>,
    read_at: Option<std::time::Instant>,
}

impl MonitorRate {
    const MAX_AGE: std::time::Duration = std::time::Duration::from_millis(500);

    fn new() -> Self {
        Self {
            hz: None,
            read_at: None,
        }
    }

    /// Forget the last read so the next `get` re-queries the window.
    fn invalidate(&mut self) {
        self.read_at = None;
    }

    /// True when the rate was never read or the last read is older than `MAX_AGE`.
    fn stale(&self, now: std::time::Instant) -> bool {
        match self.read_at {
            None => true,
            Some(read_at) => now.duration_since(read_at) > Self::MAX_AGE,
        }
    }

    /// The cached rate, re-reading it through `monitor_hz` when stale.
    fn get(&mut self, window: &Window, now: std::time::Instant) -> Option<f32> {
        if self.stale(now) {
            self.hz = monitor_hz(window);
            self.read_at = Some(now);
        }
        self.hz
    }
}

/// The refresh rate of the monitor the window is on, in Hz, if the platform
/// reports one. `None` is a real answer, not an error: the loop then simply
/// answers every wake-up instead of pacing.
fn monitor_hz(window: &Window) -> Option<f32> {
    window
        .current_monitor()
        .and_then(|monitor| monitor.refresh_rate_millihertz())
        .map(|millihertz| millihertz as f32 / 1000.0)
}

/// Round the whole app window where the platform supports it.
///
/// The window is borderless, and on Windows 11 DWM composites borderless
/// windows too — asking it for a round corner preference rounds the app's own
/// chrome (title bar and client area alike), so the surface presents into a
/// rounded window without the app masking anything itself. Elsewhere this is a
/// no-op and the window keeps square corners. Returns whether the compositor
/// took the preference (the selfcheck reports it).
#[cfg(windows)]
fn round_window_corners(window: &Window) -> bool {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    #[link(name = "dwmapi")]
    unsafe extern "system" {
        fn DwmSetWindowAttribute(
            hwnd: *mut core::ffi::c_void,
            attribute: u32,
            value: *const core::ffi::c_void,
            size: u32,
        ) -> i32;
        fn DwmGetWindowAttribute(
            hwnd: *mut core::ffi::c_void,
            attribute: u32,
            value: *mut core::ffi::c_void,
            size: u32,
        ) -> i32;
    }

    /// `DWMWA_WINDOW_CORNER_PREFERENCE`.
    const CORNER_PREFERENCE: u32 = 33;
    /// `DWMWCP_ROUND`.
    const ROUND: i32 = 2;

    let Ok(handle) = window.window_handle() else {
        return false;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return false;
    };
    let hwnd = win32.hwnd.get() as *mut core::ffi::c_void;
    let preference = ROUND;
    let size = std::mem::size_of::<i32>() as u32;
    let mut read_back: i32 = -1;
    // Best effort: this attribute is Windows 11 only and simply fails
    // elsewhere, which is fine — the window is still fully usable.
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            CORNER_PREFERENCE,
            &preference as *const i32 as *const core::ffi::c_void,
            size,
        );
        // Read the preference back so the selfcheck can report whether the
        // compositor actually accepted it.
        DwmGetWindowAttribute(
            hwnd,
            CORNER_PREFERENCE,
            &mut read_back as *mut i32 as *mut core::ffi::c_void,
            size,
        );
    }
    read_back == ROUND
}

#[cfg(not(windows))]
fn round_window_corners(_window: &Window) -> bool {
    false
}

/// Apply the window commands the app queued while building its frame.
fn drain_window_commands(
    app: &mut App,
    window: Option<&Arc<Window>>,
    event_loop: &ActiveEventLoop,
    last_cursor_icon: &mut Option<CursorIcon>,
) {
    let Some(window) = window else { return };
    // The IME candidate window rides on the focused text field, which the
    // app reports in the same logical coordinates the window takes — and
    // composition is offered only while an editor is actually focused.
    let editing = app.ime_cursor_rect();
    window.set_ime_allowed(editing.is_some());
    if let Some(rect) = editing {
        window.set_ime_cursor_area(
            winit::dpi::LogicalPosition::new(rect.x as f64, rect.y as f64),
            winit::dpi::LogicalSize::new(rect.w as f64, rect.h as f64),
        );
    }
    for command in app.window_commands.drain(..) {
        match command {
            WindowCommand::Move => {
                // Borderless move: the OS takes over the drag loop while
                // the button is held. Errors here are non-fatal (a
                // double-press between frames, say).
                let _ = window.drag_window();
            }
            WindowCommand::Resize(edge) => {
                let direction = match edge {
                    ResizeEdge::Left => ResizeDirection::West,
                    ResizeEdge::Right => ResizeDirection::East,
                    ResizeEdge::Top => ResizeDirection::North,
                    ResizeEdge::Bottom => ResizeDirection::South,
                    ResizeEdge::TopLeft => ResizeDirection::NorthWest,
                    ResizeEdge::TopRight => ResizeDirection::NorthEast,
                    ResizeEdge::BottomLeft => ResizeDirection::SouthWest,
                    ResizeEdge::BottomRight => ResizeDirection::SouthEast,
                };
                let _ = window.drag_resize_window(direction);
            }
            WindowCommand::Minimize => window.set_minimized(true),
            WindowCommand::ToggleMaximize => window.set_maximized(!window.is_maximized()),
            WindowCommand::Close => event_loop.exit(),
            WindowCommand::SetCursor(cursor) => {
                let icon = match cursor {
                    WindowCursor::Default => CursorIcon::Default,
                    WindowCursor::Move => CursorIcon::Move,
                    WindowCursor::ResizeHorizontal => CursorIcon::EwResize,
                    WindowCursor::ResizeVertical => CursorIcon::NsResize,
                    WindowCursor::ResizeDiagonal => CursorIcon::NwseResize,
                    WindowCursor::ResizeDiagonalAnti => CursorIcon::NeswResize,
                };
                if *last_cursor_icon != Some(icon) {
                    window.set_cursor(icon);
                    *last_cursor_icon = Some(icon);
                }
            }
        }
    }
}

/// The printable text one logical key types, if any — `winit`'s
/// textual view of the key (`Key::Character`, plus the space bar, which
/// arrives as `Named(NamedKey::Space)` rather than `Character(" ")`).
/// Control characters stay in the string so the caller can filter them
/// out and let `map_physical` route those keys by physical code.
fn typed_text(key: &WinitKey) -> Option<&str> {
    key.to_text()
}

/// Layout-independent control keys (text comes through [`typed_text`]).
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

fn main() {
    let mut args = std::env::args().skip(1);
    let mut selfcheck = false;
    let mut pace_probe = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--selfcheck" => selfcheck = true,
            "--pace-probe" => {
                // `--pace-probe` alone probes for three seconds.
                let seconds = args
                    .next()
                    .and_then(|value| value.parse::<f32>().ok())
                    .filter(|seconds| *seconds > 0.1)
                    .unwrap_or(3.0);
                pace_probe = Some(seconds);
            }
            _ => {}
        }
    }
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop
        .run_app(&mut Studio::new(selfcheck, pace_probe))
        .expect("run studio");
}

#[cfg(test)]
mod tests {
    use super::{MonitorRate, typed_text};
    use dream::ui::Rect;
    use dream::ui::atlas::AtlasConfig;
    use dream::ui::font::{FontId, Fonts};
    use dream::ui::widgets::{ButtonStyle, Key, KeyFeed, Ui, UiInput};
    use winit::keyboard::{Key as WinitKey, NamedKey};

    #[test]
    fn monitor_rate_is_stale_until_read_and_after_max_age() {
        use std::time::{Duration, Instant};
        let t0 = Instant::now();
        let mut rate = MonitorRate::new();
        assert!(rate.stale(t0), "a fresh cache has never been read");
        rate.read_at = Some(t0);
        assert!(!rate.stale(t0 + Duration::from_millis(100)));
        assert!(rate.stale(t0 + Duration::from_millis(600)));
        rate.invalidate();
        assert!(rate.stale(t0), "invalidate forgets the last read");
    }

    /// The space bar must read as printable text. winit hands it over as
    /// `Named(NamedKey::Space)` on every platform — never
    /// `Character(" ")` — so a `Key::Character`-only match dropped it
    /// into `map_physical`, which knows no `Space`, and it never typed.
    #[test]
    fn space_key_reads_as_printable_text() {
        assert_eq!(
            typed_text(&WinitKey::Named(NamedKey::Space)),
            Some(" "),
            "space must reach text fields"
        );
        assert_eq!(typed_text(&WinitKey::Character("x".into())), Some("x"));
    }

    /// Named control keys keep their old route: their text is control
    /// only, so the caller's `is_control` filter skips it and
    /// `map_physical` still decides what the key meant.
    #[test]
    fn named_control_keys_still_fall_through_to_the_physical_map() {
        for key in [
            NamedKey::Enter,
            NamedKey::Tab,
            NamedKey::Backspace,
            NamedKey::Escape,
        ] {
            let key = WinitKey::Named(key);
            let text = typed_text(&key);
            assert!(
                text.is_some_and(|text| text.chars().all(char::is_control)),
                "{key:?} reads as control text, not typed text"
            );
        }
        assert_eq!(typed_text(&WinitKey::Named(NamedKey::ArrowUp)), None);
        assert_eq!(typed_text(&WinitKey::Named(NamedKey::Shift)), None);
    }

    /// End to end through the host's own path: what winit reports for a
    /// space-bar press, decoded by `typed_text` and the `KeyFeed` this
    /// host drives, must reach a focused field as a typed space. The
    /// widget layer always accepted spaces; the winit → `Key::Char`
    /// translation is what dropped them.
    #[test]
    fn space_bar_press_types_a_space_into_a_focused_field() {
        let fonts = Fonts::bundled();
        let mut ui = Ui::new(fonts, AtlasConfig::default());
        let mut feed = KeyFeed::default();
        let mut input = UiInput::default();
        // One key event, forwarded exactly as `window_event` does it.
        let press = WinitKey::Named(NamedKey::Space);
        if let Some(text) = typed_text(&press) {
            for ch in text.chars() {
                if !ch.is_control() {
                    feed.key(&mut input, Key::Char(ch), true);
                }
            }
        }
        // Click-to-focus and the keystroke arrive together.
        input.mouse = (20.0, 15.0);
        input.mouse_pressed = true;
        let rect = Rect::new(0.0, 0.0, 200.0, 30.0);
        let mut text = String::from("hi");
        ui.begin(input, (800.0, 600.0));
        ui.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
        assert_eq!(text, "hi ", "space must type, not fall through");
    }
}
