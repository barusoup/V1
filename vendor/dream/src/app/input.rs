use super::main_pane::MainPanePage;
use super::{App, Command, Key, PointerButton, ViewportDrag, chrome};
use std::time::Instant;

impl App {
    // --- Input ---------------------------------------------------------------

    pub fn pointer_moved(&mut self, position: (f32, f32)) {
        self.input.mouse = position;
        self.mouse = position;
        self.last_input = Instant::now();
        self.dirty = true;
        if self.layout.viewport.contains(position) {
            self.viewport_cursor = Some(position);
        } else {
            self.viewport_cursor = None;
        }
    }

    pub fn pointer_button(&mut self, button: PointerButton, pressed: bool) {
        self.last_input = Instant::now();
        // The resize frame owns presses on the window's outer 6px (and
        // corners); the UI never sees them.
        if pressed
            && button == PointerButton::Left
            && let Some(edge) = chrome::resize_edge_under(self)
        {
            // Latch the edge for the next built frame (commands pushed here
            // at event time are cleared as stale before the host drains
            // them), and owe that frame even on a still screen — otherwise
            // the loop parks in Wait and the press never reaches the host.
            self.resize_pending = Some(edge);
            self.dirty = true;
            return;
        }
        self.dirty = true;
        let in_viewport = self
            .viewport_cursor
            .is_some_and(|p| self.layout.viewport.contains(p));
        match button {
            PointerButton::Left => {
                self.left_down = pressed;
                self.input.mouse_down = pressed;
                // Latch the edges: a press and release inside one frame (a
                // fast click) must not lose the press.
                self.input.mouse_pressed |= pressed;
                self.input.mouse_released |= !pressed;
                if pressed
                    && in_viewport
                    && self.main_pane_page == MainPanePage::Viewport
                    && self.drag.is_none()
                {
                    // The camera takes the drag only when the viewport page is active.
                    self.drag = Some(if self.shift_held {
                        ViewportDrag::Pan
                    } else {
                        ViewportDrag::Orbit
                    });
                    self.drag_start_orbit = Some(self.scene.orbit.clone());
                }
                if !pressed {
                    self.end_drag();
                }
            }
            PointerButton::Right => {
                self.right_down = pressed;
                self.input.mouse_right_down = pressed;
                self.input.mouse_right_pressed |= pressed;
                // Right-click opens the viewport's context menu, but only
                // while the viewport page is showing: every row acts on the
                // view, so the menu has no business over the other pages.
                if pressed
                    && self.main_pane_page == MainPanePage::Viewport
                    && self.layout.viewport.contains(self.mouse)
                {
                    self.menu_open = true;
                    self.menu_at = self.mouse;
                    self.menu_focus = 0;
                }
            }
            PointerButton::Middle => {
                if pressed
                    && in_viewport
                    && self.main_pane_page == MainPanePage::Viewport
                    && self.drag.is_none()
                {
                    self.drag = Some(ViewportDrag::Pan);
                }
                if !pressed {
                    self.end_drag();
                }
            }
        }
    }

    pub(crate) fn end_drag(&mut self) {
        if let Some(drag) = self.drag.take()
            && drag == ViewportDrag::Orbit
            && let Some(before) = self.drag_start_orbit.take()
        {
            // Undo restores the pre-drag orbit, not the post-drag one.
            if before != self.scene.orbit {
                self.undo_stack.push(Command::Camera(before));
                self.redo_stack.clear();
            }
        }
        self.drag_start_orbit = None;
    }

    /// Camera drag, in viewport pixels.
    pub fn viewport_drag(&mut self, dx: f32, dy: f32) {
        match self.drag {
            Some(ViewportDrag::Orbit) => self.scene.orbit.orbit(dx, dy),
            Some(ViewportDrag::Pan) => self.scene.orbit.pan(dx, dy),
            None => {}
        }
        if self.drag.is_some() {
            self.dirty = true;
        }
    }

    /// Wheel scroll. Scroll up (positive) zooms the camera in when the
    /// pointer is over the active 3D viewport; UI scroll views see the same
    /// sign. `delta` is the legacy line unit the camera zooms by; the UI is
    /// fed the same gesture in logical pixels.
    pub fn wheel(&mut self, delta: f32) {
        let pixels = delta * self.ui.scroll_config.line_height;
        self.wheel_scroll(delta, pixels, false);
    }

    /// One wheel event split into its two consumers: `lines` is the legacy
    /// unit (camera zoom), `pixels` is the same gesture in logical pixels
    /// for scroll views. `precise` marks a pixel-accurate device (trackpad,
    /// touch): its deltas are applied 1:1 instead of gliding to a target.
    pub fn wheel_scroll(&mut self, lines: f32, pixels: f32, precise: bool) {
        if precise {
            self.input.wheel_precise += pixels;
        } else {
            self.input.wheel += pixels;
        }
        self.last_input = Instant::now();
        self.dirty = true;
        if self.main_pane_page == MainPanePage::Viewport
            && let Some(cursor) = self.viewport_cursor
            && self.layout.viewport.contains(cursor)
        {
            self.scene.orbit.zoom(lines);
        }
    }

    pub fn key_event(&mut self, key: Key, pressed: bool) {
        self.last_input = Instant::now();
        match (key, pressed) {
            // Character presses route through the app's shortcut layers
            // before reaching a field: under the platform modifier they
            // are chords (the focused editor's, or the scene's undo/redo
            // while nothing is editing), otherwise typed text.
            (Key::Char(ch), true) => {
                if self.keys.command() {
                    if self.ui.editing() {
                        self.keys.key(&mut self.input, key, true);
                    } else {
                        self.shortcut(ch);
                    }
                } else if !self.app_shortcut(ch) {
                    self.keys.key(&mut self.input, key, true);
                }
            }
            _ => self.keys.key(&mut self.input, key, pressed),
        }
        self.sync_modifiers();
        if !pressed {
            return;
        }
        self.dirty = true;
        match key {
            Key::Escape => {
                // Settings is a dedicated page: Escape leaves it, unless a
                // focused field takes the cancel for itself.
                if self.main_pane_page == MainPanePage::Settings && self.ui.focus_id().is_none() {
                    self.close_settings();
                }
            }
            Key::Enter => {
                // If creating a scene while field is focused, submit it —
                // never mid-composition: Enter belongs to the IME.
                if self.main_pane_page == MainPanePage::SceneManager
                    && self.ui.focused("new_scene_input")
                    && !self.composing()
                {
                    self.create_new_scene();
                }
            }
            _ => {}
        }
    }

    /// Mirror the feed's latches into the app's held-modifier fields.
    fn sync_modifiers(&mut self) {
        (self.shift_held, self.ctrl_held, self.alt_held) =
            (self.keys.shift(), self.keys.command(), self.keys.alt());
    }

    /// Track the modifier state winit reports (ModifiersChanged carries the
    /// ground truth — per-key events alone miss a modifier released outside
    /// the window). `command` is the platform shortcut key: Super on macOS,
    /// Control elsewhere — the host resolves it, the app only ever sees
    /// `ctrl_held`.
    pub fn modifiers(&mut self, shift: bool, command: bool, alt: bool) {
        if (self.shift_held, self.ctrl_held, self.alt_held) == (shift, command, alt) {
            return;
        }
        self.keys.modifiers(shift, command, alt);
        self.keys.apply(&mut self.input);
        self.sync_modifiers();
        self.dirty = true;
    }

    /// Ctrl/Meta shortcuts while no field is editing: the scene's
    /// undo/redo. A focused editor's chords were decoded by the `KeyFeed`
    /// already.
    fn shortcut(&mut self, ch: char) {
        match ch {
            'z' | 'Z' => {
                if self.keys.shift() {
                    self.redo();
                } else {
                    self.undo();
                }
            }
            'y' | 'Y' => self.redo(),
            _ => {}
        }
    }

    /// Plain-key shortcuts (the rail badges). Returns true when consumed;
    /// while a text field has focus, every printable key belongs to it.
    fn app_shortcut(&mut self, ch: char) -> bool {
        if self.ui.focus_id().is_some() {
            return false;
        }
        match ch {
            '1' => {
                self.pan_mode = false;
                self.dirty = true;
            }
            '2' => {
                self.pan_mode = true;
                self.dirty = true;
            }
            '7' if self.scene.debug_loaded() => {
                self.settings.spin = if self.settings.spin > 0.0 { 0.0 } else { 0.4 };
                self.dirty = true;
            }
            'd' | 'D' => self.toggle_debug(),
            't' | 'T' => self.flip_theme(),
            _ => return false,
        }
        true
    }
}
