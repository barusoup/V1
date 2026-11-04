//! The default key bindings: a platform-neutral [`Key`] vocabulary plus
//! [`KeyFeed`], the translator that turns raw OS key events into
//! [`UiInput`] — nav edges, typed text, and the standard editing chords
//! (select-all, copy, cut, paste, undo, redo) including the OS clipboard
//! seam. Widgets never see keys; a host forwards events to a `KeyFeed`
//! and the decoded flags ride `Ui::begin` like any other input.

use super::UiInput;

/// Logical keys a host maps OS key events into (`winit`: `Key::Character`
/// produces one [`Key::Char`] per codepoint, `PhysicalKey::Code` the
/// rest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    /// A text-producing key. Under the platform shortcut modifier it is a
    /// chord, not text.
    Char(char),
    Shift,
    Control,
    Alt,
    Super,
    Escape,
    Enter,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    /// Page keys: scroll the view under the pointer a viewport at a time
    /// while no widget owns focus.
    PageUp,
    PageDown,
    Delete,
    Backspace,
    /// Tab focus navigation (forwarded through for the focus stage).
    Tab,
}

/// Held-modifier state plus the key → [`UiInput`] translation behind the
/// built-in editing bindings. The host keeps one, forwards every key
/// event and `ModifiersChanged` ground truth, and calls
/// [`apply`](Self::apply) on each frame's input before `Ui::begin` —
/// edge flags accumulate in the input between frames, but held flags
/// must be re-applied because `begin` takes the input:
///
/// ```ignore
/// feed.modifiers(mods.shift_key(), command, mods.alt_key()); // ModifiersChanged
/// feed.key(&mut input, key, pressed);                        // KeyboardInput
/// feed.apply(&mut input);                                    // each frame
/// ```
///
/// `command` is the platform shortcut modifier — Super on macOS, Control
/// elsewhere — resolved by the host, exactly like word movement is.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyFeed {
    shift: bool,
    ctrl: bool,
    alt: bool,
    meta: bool,
}

impl KeyFeed {
    /// `ModifiersChanged` ground truth (`command` = Super on macOS,
    /// Control elsewhere). Per-key events alone miss a modifier released
    /// outside the window; hosts that forward them too get belt and
    /// suspenders — [`Self::key`] updates the same latches.
    pub fn modifiers(&mut self, shift: bool, command: bool, alt: bool) {
        self.shift = shift;
        self.alt = alt;
        if cfg!(target_os = "macos") {
            self.meta = command;
        } else {
            self.ctrl = command;
        }
    }

    /// Shift held (selection extend, reverse Tab).
    pub fn shift(&self) -> bool {
        self.shift
    }

    /// Alt/Option held (the word-movement modifier on macOS).
    pub fn alt(&self) -> bool {
        self.alt
    }

    /// The platform shortcut modifier: Super on macOS, Control elsewhere.
    pub fn command(&self) -> bool {
        if cfg!(target_os = "macos") {
            self.meta
        } else {
            self.ctrl
        }
    }

    /// The word-movement modifier: Option on macOS, Control elsewhere.
    fn word(&self) -> bool {
        if cfg!(target_os = "macos") {
            self.alt
        } else {
            self.ctrl
        }
    }

    /// Write the held modifiers onto a frame's input (`nav.shift`,
    /// `nav.word`). [`Self::key`] does the same on every event; call this
    /// once per frame so a modifier held across event-free frames still
    /// reaches the widget after `begin` takes the input.
    pub fn apply(&self, input: &mut UiInput) {
        input.nav.shift = self.shift;
        input.nav.word = self.word();
    }

    /// One key event: modifier keys only update the latches (press and
    /// release alike); everything else writes its edge on press —
    /// arrows/Home/End/Delete/Backspace/Tab, Enter/Escape, typed text,
    /// and the editing chords via [`Self::chord`]. A `Char` under the
    /// shortcut modifier is a chord or nothing, never text.
    pub fn key(&mut self, input: &mut UiInput, key: Key, pressed: bool) {
        match key {
            Key::Shift => self.shift = pressed,
            Key::Control => self.ctrl = pressed,
            Key::Alt => self.alt = pressed,
            Key::Super => self.meta = pressed,
            _ => {}
        }
        self.apply(input);
        if !pressed {
            return;
        }
        match key {
            Key::Char(ch) => {
                if self.command() {
                    self.chord(input, ch);
                } else if !ch.is_control() {
                    input.text.push(ch);
                }
            }
            Key::Escape => input.nav.cancel = true,
            Key::Enter => input.nav.confirm = true,
            Key::Up => input.nav.up = true,
            Key::Down => input.nav.down = true,
            Key::Left => input.nav.left = true,
            Key::Right => input.nav.right = true,
            Key::Home => input.nav.home = true,
            Key::End => input.nav.end = true,
            Key::PageUp => input.nav.page_up = true,
            Key::PageDown => input.nav.page_down = true,
            Key::Delete => input.nav.delete = true,
            Key::Backspace => input.backspace += 1,
            Key::Tab => input.nav.tab = true,
            _ => {}
        }
    }

    /// The editing chords under the shortcut modifier: `a` select-all,
    /// `c` copy, `x` cut, `v` paste (read from the OS clipboard),
    /// `z`/`shift+z` undo/redo, `y` redo. Returns whether `ch` named one —
    /// a host with its own shortcut layer routes the misses itself.
    pub fn chord(&self, input: &mut UiInput, ch: char) -> bool {
        match ch.to_ascii_lowercase() {
            'a' => input.nav.select_all = true,
            'c' => input.nav.copy = true,
            'x' => input.nav.cut = true,
            'v' => {
                if let Some(text) = crate::ui::clipboard::get_text() {
                    input.paste = Some(text);
                }
            }
            'z' => {
                if self.shift {
                    input.nav.redo = true;
                } else {
                    input.nav.undo = true;
                }
            }
            'y' => input.nav.redo = true,
            _ => return false,
        }
        true
    }
}
