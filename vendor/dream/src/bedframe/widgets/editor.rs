//! Per-field text editing core: grapheme-aware caret/selection, word
//! movement, a bounded undo ring and host-seam clipboard. Pure `String`
//! operations — no drawing, no GPU — so the whole editor is unit-testable.
//!
//! Carets and anchors are *byte* offsets snapped to extended-grapheme
//! boundaries: a caret never lands inside a ZWJ emoji, a flag, or a
//! combining mark. All offsets come from `unicode_segmentation`, never
//! `Vec<char>` arithmetic that would split clusters.

use super::UiInput;
use crate::ui::tween::Anim;
use unicode_segmentation::UnicodeSegmentation;

/// Byte stops a caret may occupy: `0`, every grapheme boundary, `len`.
#[allow(dead_code)]
pub(crate) fn caret_stops(text: &str) -> Vec<usize> {
    let mut stops = vec![0usize];
    stops.extend(
        text.grapheme_indices(true)
            .map(|(index, cluster)| index + cluster.len()),
    );
    stops
}

/// The grapheme stop at or before `byte` (any byte offset, even inside a
/// cluster or UTF-8 codepoint).
pub(crate) fn snap_before(text: &str, byte: usize) -> usize {
    let mut stop = 0;
    for (index, cluster) in text.grapheme_indices(true) {
        if index > byte {
            break;
        }
        let end = index + cluster.len();
        stop = if end <= byte { end } else { index };
        if end > byte {
            break;
        }
    }
    stop
}

/// The grapheme stop at or after `byte`.
pub(crate) fn snap_after(text: &str, byte: usize) -> usize {
    for (index, cluster) in text.grapheme_indices(true) {
        if index >= byte {
            return index;
        }
        let end = index + cluster.len();
        if end >= byte {
            return end;
        }
    }
    text.len()
}

/// Word boundary left/right of `byte` via `unicode_word_indices`.
pub(crate) fn word_left(text: &str, byte: usize) -> usize {
    let byte = byte.min(text.len());
    text[..byte]
        .unicode_word_indices()
        .next_back()
        .map(|(index, _)| index)
        .unwrap_or(0)
}

pub(crate) fn word_right(text: &str, byte: usize) -> usize {
    let byte = byte.min(text.len());
    text[byte..]
        .unicode_word_indices()
        .nth(1)
        .map(|(index, _)| byte + index)
        .unwrap_or(text.len())
}

/// Per-field editing state: caret + selection anchor (byte offsets) and
/// a bounded undo ring of `(text, caret)` snapshots.
#[derive(Clone, Debug)]
pub struct EditorState {
    /// Byte caret, always a grapheme stop.
    pub caret: usize,
    /// Retirement bookkeeping (stale-state sweep).
    pub last_frame: u64,
    /// Caret-blink wall clock (seconds); driven by `text_field`, reset
    /// on real edits, caret moves and the frame focus lands. Kept here
    /// so the blink survives scroll/view adjustments untouched.
    pub(crate) blink_elapsed: f32,
    /// Whether the field held focus last frame (focus-gain resets the
    /// blink to the visible phase).
    pub(crate) blink_focused: bool,
    /// Set by `text_field` when an edit or caret move happened this
    /// frame — the blink restarts so typing never hides the caret.
    pub(crate) touched: bool,
    /// Horizontal scroll the field last settled on (persistent).
    pub scroll: f32,
    /// Selection anchor (equal to caret = collapsed).
    pub anchor: usize,
    /// Past states, oldest last: `undo_stack.back()` is the state one
    /// undo restores — `(text, caret, anchor)`. The live text is
    /// implicit (it is `*text`).
    undo_stack: std::collections::VecDeque<(String, usize, usize)>,
    /// States undo walked back over, newest last — redo replays them.
    redo_stack: std::collections::VecDeque<(String, usize, usize)>,
    /// Selection-highlight visuals: opacity plus the last drawn edge
    /// positions (text-space logical px). The highlight appears and moves
    /// instantly; only its disappearance animates — a fast fade-out that
    /// still needs the geometry it last had.
    pub sel_fade: Anim<f32>,
    pub sel_last: (f32, f32),
    /// Caret visuals: 1.0 while focused, fading out fast on blur at its
    /// last text-space position — the ghost the fade-out draws.
    pub caret_fade: Anim<f32>,
    pub caret_last_x: f32,
    /// Byte total of both stacks' strings (the global cap subtracts from
    /// this field first when over).
    history_bytes: usize,
    /// What the field's `*text` looked like last frame (len + FNV hash):
    /// external mutation invalidates selection and undo.
    known_len: usize,
    known_hash: u64,
}

/// Snapshots kept per field; the whole UI also honours a 2 MiB retained
/// total (enforced by `Ui`).
const FIELD_HISTORY_MAX: usize = 32;

fn hash_text(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl EditorState {
    /// Open a field whose text is `text` (caret at end, no selection).
    pub fn new(text: &str) -> Self {
        Self {
            caret: text.len(),
            anchor: text.len(),
            last_frame: 0,
            blink_elapsed: 0.0,
            blink_focused: false,
            touched: false,
            scroll: 0.0,
            sel_fade: Anim::new(0.0),
            sel_last: (0.0, 0.0),
            caret_fade: Anim::new(0.0),
            caret_last_x: 0.0,
            undo_stack: std::collections::VecDeque::new(),
            redo_stack: std::collections::VecDeque::new(),
            history_bytes: 0,
            known_len: text.len(),
            known_hash: hash_text(text),
        }
    }

    /// Reconcile with `*text` from the caller: an external mutation loses
    /// its undo trail and selection rather than corrupting offsets.
    pub fn sync(&mut self, text: &str) {
        if text.len() == self.known_len && hash_text(text) == self.known_hash {
            return;
        }
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.history_bytes = 0;
        self.collapse(text.len().min(self.caret));
        self.caret = snap_before(text, self.caret);
        self.anchor = self.caret;
        self.mark_known(text);
    }

    /// Record the current text as the one we last owned.
    fn mark_known(&mut self, text: &str) {
        self.known_len = text.len();
        self.known_hash = hash_text(text);
    }

    /// Byte range selected (`(anchor, caret)` ordered).
    pub fn selection(&self) -> Option<std::ops::Range<usize>> {
        if self.caret == self.anchor {
            None
        } else {
            Some(self.caret.min(self.anchor)..self.caret.max(self.anchor))
        }
    }

    /// Collapse any selection at `byte` (a grapheme stop).
    pub fn collapse(&mut self, byte: usize) {
        self.caret = byte;
        self.anchor = byte;
    }

    /// Push an undo snapshot BEFORE mutating: `(old_text, caret_to_restore)`.
    fn snapshot(&mut self, text: &str) {
        // A real edit kills the redo tail.
        while let Some((dropped, _, _)) = self.redo_stack.pop_back() {
            self.history_bytes -= dropped.len();
        }
        self.undo_stack
            .push_back((text.to_string(), self.caret, self.anchor));
        self.history_bytes += text.len();
        if self.undo_stack.len() > FIELD_HISTORY_MAX
            && let Some((dropped, _, _)) = self.undo_stack.pop_front()
        {
            self.history_bytes -= dropped.len();
        }
    }

    /// Undo: restore the previous snapshot; the caret it saved returns.
    pub fn undo(&mut self, text: &mut String) {
        let Some((old, caret, anchor)) = self.undo_stack.pop_back() else {
            return;
        };
        self.history_bytes -= old.len();
        self.history_bytes += text.len();
        self.redo_stack
            .push_back((std::mem::take(text), self.caret, self.anchor));
        *text = old;
        self.caret = snap_before(text, caret);
        self.anchor = snap_before(text, anchor);
        self.mark_known(text);
    }

    /// Redo: walk forward over states recorded before the undo.
    pub fn redo(&mut self, text: &mut String) {
        let Some((old, caret, anchor)) = self.redo_stack.pop_back() else {
            return;
        };
        self.history_bytes -= old.len();
        self.history_bytes += text.len();
        self.undo_stack
            .push_back((std::mem::take(text), self.caret, self.anchor));
        *text = old;
        self.caret = snap_before(text, caret);
        self.anchor = snap_before(text, anchor);
        self.mark_known(text);
    }

    /// True while a fade-out (selection highlight or caret ghost) is
    /// still moving — `Ui` folds this into `animating()` so hosts keep
    /// presenting the motion.
    pub(crate) fn vis_animating(&self) -> bool {
        !self.sel_fade.finished() || !self.caret_fade.finished()
    }

    /// The snapshot count this state retains and their byte payload —
    /// `Ui` enforces the global retained-history bound with it.
    pub fn history_bytes(&self) -> usize {
        self.history_bytes
    }

    /// Retire the oldest snapshots until `budget` bytes remain.
    pub fn trim_history(&mut self, budget: usize) {
        while self.history_bytes > budget && !self.undo_stack.is_empty() {
            if let Some((dropped, _, _)) = self.undo_stack.pop_front() {
                self.history_bytes -= dropped.len();
            }
        }
        while self.history_bytes > budget && !self.redo_stack.is_empty() {
            if let Some((dropped, _, _)) = self.redo_stack.pop_front() {
                self.history_bytes -= dropped.len();
            }
        }
    }

    /// Insert `insert` at the caret, replacing the selection.
    pub fn insert(&mut self, text: &mut String, insert: &str) {
        self.snapshot(text);
        if let Some(range) = self.selection() {
            text.replace_range(range.clone(), insert);
            self.caret = range.start + insert.len();
        } else {
            let at = snap_before(text, self.caret.min(text.len()));
            text.insert_str(at, insert);
            self.caret = at + insert.len();
        }
        self.caret = snap_before(text, self.caret.min(text.len()));
        self.anchor = self.caret;
        self.mark_known(text);
    }

    /// Backspace: delete the selection, else the grapheme before caret.
    pub fn backspace(&mut self, text: &mut String) {
        if self.selection().is_none() && self.caret == 0 {
            return;
        }
        self.snapshot(text);
        if let Some(range) = self.selection() {
            text.replace_range(range.clone(), "");
            self.collapse(range.start);
        } else {
            let before = snap_before(text, self.caret - 1);
            text.replace_range(before..self.caret, "");
            self.collapse(before);
        }
        self.mark_known(text);
    }

    /// Delete-forward: selection, else the grapheme after caret.
    pub fn delete(&mut self, text: &mut String) {
        if self.selection().is_none() && self.caret >= text.len() {
            return;
        }
        self.snapshot(text);
        if let Some(range) = self.selection() {
            text.replace_range(range.clone(), "");
            self.collapse(range.start);
        } else {
            let after = snap_after(text, self.caret + 1);
            text.replace_range(self.caret..after, "");
        }
        self.mark_known(text);
    }

    /// Move the caret (or extend the selection when `shift`).
    pub fn move_to(&mut self, text: &str, byte: usize, shift: bool) {
        self.caret = snap_before(text, byte.min(text.len()));
        if !shift {
            self.anchor = self.caret;
        }
    }

    pub fn move_left(&mut self, text: &str, shift: bool) {
        if self.selection().is_some() && !shift {
            self.collapse(self.caret.min(self.anchor));
            return;
        }
        let byte = if self.caret == 0 {
            0
        } else {
            snap_before(text, self.caret - 1)
        };
        self.move_to(text, byte, shift);
    }

    pub fn move_right(&mut self, text: &str, shift: bool) {
        if self.selection().is_some() && !shift {
            self.collapse(self.caret.max(self.anchor));
            return;
        }
        let byte = snap_after(text, self.caret + 1);
        self.move_to(text, byte, shift);
    }

    pub fn move_word_left(&mut self, text: &str, shift: bool) {
        self.move_to(text, word_left(text, self.caret), shift);
    }

    pub fn move_word_right(&mut self, text: &str, shift: bool) {
        self.move_to(text, word_right(text, self.caret), shift);
    }

    pub fn select_all(&mut self, text: &str) {
        self.anchor = 0;
        self.caret = text.len();
    }

    /// Apply one frame's editing input to `text`: typed and pasted text, backspace, forward delete,
    /// select-all, copy and cut, undo and redo, and the caret keys. Returns the text a copy or a cut
    /// put on the clipboard, for the host to write out.
    ///
    /// `multiline` keeps line breaks in the text; a single-line field turns them into spaces. This is
    /// the editing half of [`Ui::text_field`](super::Ui::text_field), so a host that lays a field out
    /// itself (a wrapped prompt, a document) edits exactly as the widget does. Enter, Escape and
    /// focus are the host's to handle.
    pub fn apply(&mut self, text: &mut String, input: &UiInput, multiline: bool) -> Option<String> {
        let nav = input.nav;
        if !input.text.is_empty() {
            self.insert_clean(text, &input.text, multiline);
        }
        if let Some(paste) = input.paste.as_deref() {
            self.insert_clean(text, paste, multiline);
        }
        for _ in 0..input.backspace {
            self.backspace(text);
        }
        if nav.delete {
            self.delete(text);
        }
        if nav.select_all {
            self.select_all(text);
        }
        let mut clipboard = None;
        if (nav.copy || nav.cut)
            && let Some(range) = self.selection()
        {
            clipboard = Some(text[range.clone()].to_string());
            if nav.cut {
                self.delete(text);
            }
        }
        if nav.undo {
            self.undo(text);
        }
        if nav.redo {
            self.redo(text);
        }
        if nav.left {
            if nav.word {
                self.move_word_left(text, nav.shift);
            } else {
                self.move_left(text, nav.shift);
            }
        }
        if nav.right {
            if nav.word {
                self.move_word_right(text, nav.shift);
            } else {
                self.move_right(text, nav.shift);
            }
        }
        if nav.home {
            self.move_to(text, 0, nav.shift);
        }
        if nav.end {
            self.move_to(text, text.len(), nav.shift);
        }
        clipboard
    }

    /// Insert typed or pasted text, cleaned for the field, as one edit. Nothing left after cleaning
    /// is no edit at all, so it leaves no undo step.
    fn insert_clean(&mut self, text: &mut String, raw: &str, multiline: bool) {
        let clean = clean_input(raw, multiline);
        if !clean.is_empty() {
            self.insert(text, &clean);
        }
    }
}

/// What a field keeps of typed or pasted text. A single-line field turns line breaks and tabs into
/// spaces; a multi-line one keeps line breaks (CRLF and CR become LF) and tabs. Every other control
/// character is dropped.
fn clean_input(raw: &str, multiline: bool) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push(if multiline { '\n' } else { ' ' });
            }
            '\n' => out.push(if multiline { '\n' } else { ' ' }),
            '\t' => out.push(if multiline { '\t' } else { ' ' }),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::widgets::NavInput;

    #[test]
    fn graphemes_are_one_caret_unit() {
        // ZWJ family emoji: three scalars, one grapheme, one backspace.
        let mut text = "a👩‍💻b".to_string();
        let mut state = EditorState::new(&text);
        state.caret = text.len();
        state.backspace(&mut text);
        assert_eq!(text, "a👩‍💻");
        state.backspace(&mut text);
        assert_eq!(text, "a", "the whole ZWJ cluster deleted at once");
    }

    #[test]
    fn regional_indicators_pair_off() {
        let text = "🇯🇵x".to_string();
        let mut state = EditorState::new(&text);
        state.collapse(0);
        state.move_right(&text, false);
        assert_eq!(state.caret, 8, "flag is one grapheme");
        state.move_right(&text, false);
        assert_eq!(state.caret, 9);
    }

    #[test]
    fn selection_replaces_and_undo_restores() {
        let mut text = "hello world".to_string();
        let mut state = EditorState::new(&text);
        state.select_all(&text);
        state.insert(&mut text, "bye");
        assert_eq!(text, "bye");
        state.undo(&mut text);
        assert_eq!(text, "hello world");
        state.redo(&mut text);
        assert_eq!(text, "bye");
    }

    #[test]
    fn word_moves_stop_at_boundaries() {
        let text = "foo bar baz";
        let mut state = EditorState::new(text);
        state.caret = 0;
        state.move_word_right(text, false);
        assert_eq!(state.caret, 4);
        state.move_word_right(text, false);
        assert_eq!(state.caret, 8);
        state.move_word_left(text, false);
        assert_eq!(state.caret, 4);
    }

    #[test]
    fn shift_extends_the_selection() {
        let text = "abcdef";
        let mut state = EditorState::new(text);
        state.caret = 0;
        state.anchor = 0;
        state.move_right(text, true);
        state.move_right(text, true);
        assert_eq!(state.selection(), Some(0..2));
        state.move_left(text, true);
        assert_eq!(state.selection(), Some(0..1));
        state.collapse(3);
        assert_eq!(state.selection(), None);
    }

    #[test]
    fn text_field_snap_matches_caret_stops_oracle() {
        for text in ["", "abc", "e\u{301}x", "a👩‍💻b", "🇯🇵ok", "x🇯🇵y🇧🇷"] {
            let stops = caret_stops(text);
            for byte in 0..=text.len() + 2 {
                let want_before = stops
                    .iter()
                    .take_while(|&&s| s <= byte)
                    .last()
                    .copied()
                    .unwrap_or(0);
                let want_after = stops
                    .iter()
                    .find(|&&s| s >= byte)
                    .copied()
                    .unwrap_or(text.len());
                assert_eq!(
                    snap_before(text, byte),
                    want_before,
                    "{text:?} byte {byte} snap_before"
                );
                assert_eq!(
                    snap_after(text, byte),
                    want_after,
                    "{text:?} byte {byte} snap_after"
                );
            }
        }
    }

    #[test]
    fn external_mutation_invalidates_history_and_offsets() {
        let mut text = "start".to_string();
        let mut state = EditorState::new(&text);
        state.insert(&mut text, "!");
        assert!(state.history_bytes() > 0);
        // Host rewrites the text directly.
        text.replace_range(.., "reset");
        state.sync(&text);
        assert_eq!(state.history_bytes(), 0, "undo trail invalidated");
        assert_eq!(state.selection(), None);
    }

    #[test]
    fn apply_keeps_line_breaks_only_in_a_multi_line_field() {
        let mut single = String::new();
        let mut state = EditorState::new(&single);
        let input = UiInput {
            text: "a\r\nb\tc".into(),
            ..UiInput::default()
        };
        state.apply(&mut single, &input, false);
        assert_eq!(
            single, "a b c",
            "a single-line field turns breaks and tabs into spaces"
        );

        let mut multi = String::new();
        let mut state = EditorState::new(&multi);
        let input = UiInput {
            paste: Some("a\r\nb\x07".into()),
            ..UiInput::default()
        };
        state.apply(&mut multi, &input, true);
        assert_eq!(
            multi, "a\nb",
            "a multi-line field keeps breaks and drops other controls"
        );
    }

    #[test]
    fn apply_copy_and_cut_hand_back_the_selection() {
        let mut text = "hello world".to_string();
        let mut state = EditorState::new(&text);
        state.select_all(&text);
        let copy = UiInput {
            nav: NavInput {
                copy: true,
                ..NavInput::default()
            },
            ..UiInput::default()
        };
        assert_eq!(
            state.apply(&mut text, &copy, true).as_deref(),
            Some("hello world")
        );
        assert_eq!(text, "hello world", "copy leaves the text alone");

        let cut = UiInput {
            nav: NavInput {
                cut: true,
                ..NavInput::default()
            },
            ..UiInput::default()
        };
        assert_eq!(
            state.apply(&mut text, &cut, true).as_deref(),
            Some("hello world")
        );
        assert_eq!(text, "", "cut removes the selection");
    }
}
