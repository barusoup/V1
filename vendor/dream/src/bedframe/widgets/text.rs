use super::{LabelStyle, Response, TextFieldStyle, Ui, editor, mix_color};
use crate::ui::font::{Align, FontId, LayoutOptions};
use crate::ui::tween::Ease;
use crate::ui::{Color, Rect};

impl Ui {
    /// Single-line text entry with a real editor: click to place the
    /// caret, drag or Shift to select, Option/Ctrl word moves, clipboard
    /// and undo all work at grapheme granularity (a ZWJ emoji, a flag or
    /// a combining mark is one caret unit, one deletion). `*text` is
    /// edited in place and byte-safe — every mutation goes through
    /// `replace_range` on grapheme boundaries.
    ///
    /// The standard keybindings are decoded by [`KeyFeed`](super::KeyFeed):
    /// forward the host's key events to it and `Ctrl`/`Cmd`+A/C/X/V/Z/Y,
    /// arrows, Home/End and friends just work. The copy/cut text leaves
    /// through [`Ui::take_clipboard_output`] / [`Ui::flush_clipboard`].
    ///
    /// `style` is a [`TextFieldStyle`]; a `ButtonStyle` converts into it
    /// with the built-in caret/selection defaults, so existing callers
    /// need no changes. [`TextFieldStyle::password`] draws one bullet
    /// per character while the field keeps editing the real string.
    pub fn text_field(
        &mut self,
        label: &str,
        rect: Rect,
        text: &mut String,
        style: impl Into<TextFieldStyle>,
    ) -> Response {
        let style = style.into();
        let id = self.id(label);
        // A disabled field neither takes the pointer nor keeps a focus it
        // was given while still enabled.
        let enabled = !style.disabled;
        self.blur_if_disabled(id, enabled);
        let hovered = enabled && self.hit(rect);
        let mut clicked = false;
        // Focus follows the pointer's single capture: overlapping fields
        // cannot both claim the same press.
        if !self.view.replay && self.input.mouse_pressed && hovered && self.try_capture(id) {
            self.focus_id = Some(id);
            clicked = true;
        } else if self.input.mouse_pressed && !hovered && self.focus_id == Some(id) {
            // Clicking anywhere else blurs the field (buttons, viewport,
            // empty space) — and drops the selection with it, so a dead
            // highlight can't linger on a field nobody is editing.
            // Scoped to our own id so a press that focuses a sibling
            // field is not stolen back by draw order.
            if let Some(state) = self.editors.get_mut(&id) {
                state.collapse(state.caret);
            }
            self.focus_id = None;
        }
        let focused = enabled && self.focus_id == Some(id) && self.input_allowed(id, rect);
        if focused {
            // A text field in focus is *editing*: the host may offer IME
            // composition here that a focused button must not get.
            self.editing_id = Some(id);
        }
        self.note_focusable(id, rect, enabled, true);

        // The editor state lives across frames; external text mutation
        // invalidates its offsets and undo inside `sync`.
        self.editors
            .entry(id)
            .or_insert_with(|| editor::EditorState::new(text));
        let (caret_now, pre_input) = {
            let state = self.editors.get_mut(&id).expect("inserted above");
            state.sync(text);
            state.last_frame = self.frame;
            (
                state.caret.min(text.len()),
                (state.caret, state.anchor, text.len()),
            )
        };

        // Pointer caret placement + drag selection. The pointer lands in
        // the field's *scrolled* text space — the scroll the field settled
        // on LAST frame, not a prefix-derived guess. Stops only resolve
        // on an actual pointer edge (no per-frame Vec for idle fields).
        let (pointer_byte, pointer_shift) = if !self.view.replay
            && ((self.input.mouse_pressed && hovered)
                || (self.active_id == Some(id) && self.input.mouse_down))
        {
            let scroll = self.editors.get(&id).expect("inserted above").scroll;
            let target = self.input.mouse.0 - (rect.x + style.text_inset) + scroll;
            let display = password_display(text, style.password);
            (
                password_real_byte(
                    text,
                    self.byte_at_x(&display, target, style.font, style.size),
                    style.password,
                ),
                self.input.nav.shift,
            )
        } else {
            (caret_now, false)
        };
        // Capture legality is a Ui-level check; resolve it BEFORE the
        // editor-state borrow.
        let drag_ok = !self.view.replay
            && self.active_id == Some(id)
            && self.input.mouse_down
            && self.capture_allowed(id, rect);
        {
            let state = self.editors.get_mut(&id).expect("inserted above");
            if self.input.mouse_pressed && hovered && self.active_id == Some(id) {
                if pointer_shift {
                    // Shift-click extends the selection from the anchor.
                    state.move_to(text, pointer_byte, true);
                } else {
                    state.collapse(pointer_byte);
                }
            } else if drag_ok {
                // Pointer-drag extends the selection — every frame, so a
                // drag passing behind a modal can't keep mutating a
                // covered field.
                state.move_to(text, pointer_byte, true);
            }
        }

        if focused && !self.view.replay {
            let nav = self.input.nav;
            let state = self.editors.get_mut(&id).expect("inserted above");
            // The editing itself is the editor's: one frame's typed, pasted and navigation input.
            // A single-line field turns line breaks into spaces.
            if let Some(copied) = state.apply(text, &self.input, false) {
                self.clipboard_out = Some(copied);
            }
            if nav.cancel {
                state.collapse(state.caret);
                self.focus_id = None;
            }
        }

        let hover = self.hover_level(id, hovered || focused);

        let bg = mix_color(
            style.normal,
            style.hover,
            if focused { 1.0 } else { hover * 0.5 },
        );
        let border = if focused { style.accent } else { style.hover };
        // A stock 1px border is a hairline: one physical pixel.
        let border_width = if style.border_width == 1.0 {
            crate::ui::layout::hairline(self.scale)
        } else {
            style.border_width
        };
        self.draw
            .rounded_frame(rect, style.corner_radius, border_width, bg, border);
        // Clip so long text scrolls instead of spilling over neighbours.
        self.draw.push_clip(rect);
        {
            let state = self.editors.get_mut(&id).expect("inserted above");
            if (state.caret, state.anchor, text.len()) != pre_input {
                state.touched = true;
            }
        }
        let caret_px = {
            let state = self.editors.get(&id).expect("inserted above");
            self.stop_x(
                &password_display(text, style.password),
                password_display_byte(text, state.caret.min(text.len()), style.password),
                style.font,
                style.size,
            )
        };
        // Persistent scroll: only adjust when the caret leaves the
        // visible window — typing mid-field doesn't teleport the view.
        let window = (rect.w - 20.0).max(1.0);
        let scroll = {
            let state = self.editors.get_mut(&id).expect("inserted above");
            if caret_px < state.scroll {
                state.scroll = caret_px;
            } else if caret_px > state.scroll + window {
                state.scroll = caret_px - window;
            }
            state.scroll
        };
        let state = self.editors.get(&id).expect("inserted above");
        let caret = state.caret.min(text.len());
        let selection = state.selection();
        let shown_x = rect.x + style.text_inset - scroll;
        // Selection highlight + caret fade-out. Geometry is instant —
        // the highlight is exactly the live range and the caret is
        // exactly the caret. Only disappearance animates: both fade out
        // fast at the geometry they last had (text space, so a scroll
        // change can't retarget them).
        let sel_targets = selection.clone().map(|range| {
            let display = password_display(text, style.password);
            (
                self.stop_x(
                    &display,
                    password_display_byte(text, range.start, style.password),
                    style.font,
                    style.size,
                ),
                self.stop_x(
                    &display,
                    password_display_byte(text, range.end, style.password),
                    style.font,
                    style.size,
                ),
            )
        });
        let sel_dt = self.anim_dt();
        let (sx1, sx2, sa, cfade, caret_last_x) = {
            let state = self.editors.get_mut(&id).expect("inserted above");
            match sel_targets {
                Some((tx1, tx2)) => {
                    state.sel_last = (tx1.min(tx2), tx1.max(tx2));
                    state.sel_fade.set(1.0);
                }
                None => {
                    if state.sel_fade.target() != 0.0 {
                        state.sel_fade.animate_to(0.0, style.fade_out, Ease::QuadIn);
                    }
                }
            }
            if focused {
                state.caret_fade.set(1.0);
                state.caret_last_x = caret_px;
            } else if state.caret_fade.target() != 0.0 {
                state
                    .caret_fade
                    .animate_to(0.0, style.fade_out, Ease::QuadIn);
            }
            (
                state.sel_last.0,
                state.sel_last.1,
                state.sel_fade.update(sel_dt),
                state.caret_fade.update(sel_dt),
                state.caret_last_x,
            )
        };
        if sa > 0.002 && (sx2 - sx1).abs() > 0.5 {
            // Selection sits `selection_inset` inside the field, so its
            // radius is the concentric inner of the field's corner.
            self.draw.round_rect(
                Rect::new(
                    shown_x + sx1,
                    rect.y + style.selection_inset,
                    sx2 - sx1,
                    rect.h - style.selection_inset * 2.0,
                ),
                crate::ui::tokens::Radius::inner(style.corner_radius, style.selection_inset),
                Color::rgba(
                    style.accent.r,
                    style.accent.g,
                    style.accent.b,
                    (f32::from(style.selection_alpha) * sa.min(1.0)) as u8,
                ),
            );
        }
        let composing = focused && self.input.preedit.is_some();
        if focused {
            // Caret blink runs on the field's own wall-clock timer — NOT
            // anim_loop_f32: a loop entry keeps `animating` true forever
            // and would pin an idle editor's host at full frame rate.
            // The timer resets on any real edit or caret move (tracked
            // inside the focused-input block above via `state.touched`),
            // and on the frame focus lands.
            let state = self.editors.get_mut(&id).expect("inserted above");
            // A reset frame (focus just gained, or an edit/caret move)
            // starts the VISIBLE phase fresh — it must NOT also swallow
            // the wall-clock dt that accumulated while the host was
            // parked: a 10 s dt would land the caret mid-hidden-phase.
            if !state.blink_focused || state.touched {
                state.blink_elapsed = 0.0;
                state.touched = false;
            } else {
                state.blink_elapsed += self.input.dt.max(0.0);
            }
            state.blink_focused = true;
            // Half of `caret_blink` visible, half hidden — wake the host
            // exactly at the next boundary, then go back to sleep. A
            // non-positive period keeps the caret solid and never wakes.
            let blink = style.caret_blink.max(0.0);
            let phase = if blink > 0.0 {
                state.blink_elapsed % blink
            } else {
                0.0
            };
            if blink > 0.0 {
                let until_next = if phase < blink * 0.5 {
                    blink * 0.5 - phase
                } else {
                    blink - phase
                };
                self.request_wake(until_next);
            }
            let caret_rect = Rect::new(
                shown_x + caret_px,
                rect.y + style.caret_inset,
                style.caret_width,
                rect.h - style.caret_inset * 2.0,
            );
            if !composing && (blink <= 0.0 || phase < blink * 0.5) {
                self.draw.rect(caret_rect, style.caret);
            }
            if !composing {
                // The IME candidate window wants the caret's rect, not
                // the field's whole box.
                self.note_editing_caret(caret_rect);
            }
        } else if let Some(state) = self.editors.get_mut(&id) {
            state.blink_focused = false;
        }
        // Blur leaves a fading ghost of the caret at its last position —
        // same fade-out the selection highlight gets.
        if !focused && cfade > 0.002 {
            self.draw.rect(
                Rect::new(
                    shown_x + caret_last_x,
                    rect.y + style.caret_inset,
                    style.caret_width,
                    rect.h - style.caret_inset * 2.0,
                ),
                Color::rgba(
                    style.caret.r,
                    style.caret.g,
                    style.caret.b,
                    (f32::from(style.caret.a) * cfade.min(1.0)) as u8,
                ),
            );
        }
        // An empty field draws no text at all: the caller owns the
        // placeholder (see the scene manager's "Enter scene name..." hint).
        // While a composition preview replaces the selection visually,
        // the preview draw below renders the WHOLE document with the
        // preview spliced in — the prefix stays visible.
        if !text.is_empty() && !composing {
            self.label(
                Rect::new(shown_x, rect.y, rect.w - 12.0, rect.h),
                &password_display(text, style.password),
                LabelStyle::new(style.font)
                    .size(style.size)
                    .color(if enabled {
                        style.text
                    } else {
                        style.text_disabled
                    })
                    .align(Align::Left)
                    .vcenter(true),
            );
        }
        // IME preedit: the underlined composition preview is the WHOLE
        // document with the preview spliced over the current selection —
        // the prefix stays shaped in context (kerning/ligatures intact)
        // and the document is never mutated until Commit lands.
        if focused && let Some(preedit) = self.input.preedit.clone() {
            let preview = preedit.text.clone();
            let sel = selection.clone().unwrap_or(caret..caret);
            let mut full = String::with_capacity(text.len() + preview.len());
            full.push_str(&text[..sel.start]);
            full.push_str(&preview);
            full.push_str(&text[sel.end..]);
            let full_display = password_display(&full, style.password);
            self.label(
                Rect::new(shown_x, rect.y, rect.w - 12.0, rect.h),
                &full_display,
                LabelStyle::new(style.font)
                    .size(style.size)
                    .color(style.text)
                    .align(Align::Left)
                    .vcenter(true),
            );
            // Underline spans exactly the preview's bytes, resolved
            // through the full preview's own shaped stops.
            let preview_x = shown_x
                + self.stop_x(
                    &full_display,
                    password_display_byte(&full, sel.start, style.password),
                    style.font,
                    style.size,
                );
            let preview_end = shown_x
                + self.stop_x(
                    &full_display,
                    password_display_byte(&full, sel.start + preview.len(), style.password),
                    style.font,
                    style.size,
                );
            self.draw.rect(
                Rect::new(
                    preview_x,
                    rect.y + rect.h - 7.0,
                    (preview_end - preview_x).max(1.0),
                    1.0,
                ),
                style.text,
            );
            // The composition's caret range (winit UTF-8 byte offsets
            // inside the preview): a bar or a highlighted span at
            // `sel.start + cursor`. The host's IME candidate window
            // follows the preview caret, not the document caret.
            if let Some((start, end)) = preedit.cursor {
                let start = start.min(preview.len());
                let end = end.min(preview.len()).max(start);
                let cx = shown_x
                    + self.stop_x(
                        &full_display,
                        password_display_byte(&full, sel.start + start, style.password),
                        style.font,
                        style.size,
                    );
                if end > start {
                    let ex = shown_x
                        + self.stop_x(
                            &full_display,
                            password_display_byte(&full, sel.start + end, style.password),
                            style.font,
                            style.size,
                        );
                    self.draw.rect(
                        Rect::new(cx, rect.y + 4.0, ex - cx, rect.h - 8.0),
                        Color::rgba(
                            style.accent.r,
                            style.accent.g,
                            style.accent.b,
                            style.selection_alpha,
                        ),
                    );
                }
                self.draw.rect(
                    Rect::new(cx, rect.y + 6.0, 1.5, rect.h - 12.0),
                    style.accent,
                );
                self.note_editing_caret(Rect::new(cx, rect.y + 4.0, 2.0, rect.h - 8.0));
            }
        }
        self.draw.pop_clip();
        Response::build(rect, self, id, clicked, style.corner_radius)
    }

    /// Single-line text input with a built-in placeholder hint shown while
    /// the field is empty and unfocused.
    pub fn text_field_with_placeholder(
        &mut self,
        label: &str,
        rect: Rect,
        text: &mut String,
        placeholder: &str,
        style: impl Into<TextFieldStyle>,
    ) -> Response {
        let style: TextFieldStyle = style.into();
        let response = self.text_field(label, rect, text, style);
        if text.is_empty() && !self.focused(label) && !placeholder.is_empty() {
            let inset = if style.text_inset.is_finite() {
                style.text_inset.max(0.0)
            } else {
                8.0
            };
            self.draw.push_clip(rect.inset(2.0));
            self.label(
                Rect::new(
                    rect.x + inset,
                    rect.y,
                    (rect.w - inset - 4.0).max(0.0),
                    rect.h,
                ),
                placeholder,
                LabelStyle::new(style.font)
                    .size(style.size)
                    .color(style.text_disabled)
                    .align(Align::Left)
                    .vcenter(true),
            );
            self.draw.pop_clip();
        }
        response
    }

    /// Logical x of a grapheme-stop caret — resolved through the SAME
    /// cached, shaped layout the draw pass uses (kerning/ligatures
    /// included), then un-scaled so it lands where the glyphs sit.
    fn stop_x(&mut self, text: &str, byte: usize, font: FontId, size: f32) -> f32 {
        let scale = self.scale;
        let layout = self.cached_layout(
            text,
            font,
            size * scale,
            LayoutOptions {
                wrap_width: None,
                line_height: LabelStyle::new(font).line_height,
                ..Default::default()
            },
        );
        let stops = &layout.stops;
        match stops.iter().position(|&(_, b)| b == byte) {
            Some(index) => stops[index].0 / scale,
            None => layout.width / scale,
        }
    }

    /// Byte offset (a grapheme stop) nearest the LOGICAL pointer x — the
    /// same `stops` geometry `stop_x` uses, so click and caret agree
    /// pixel-for-pixel even on non-monotonic (RTL) runs.
    fn byte_at_x(&mut self, text: &str, x: f32, font: FontId, size: f32) -> usize {
        let scale = self.scale;
        let layout = self.cached_layout(
            text,
            font,
            size * scale,
            LayoutOptions {
                wrap_width: None,
                line_height: LabelStyle::new(font).line_height,
                ..Default::default()
            },
        );
        let target = x * scale;
        layout
            .stops
            .iter()
            .min_by(|&&(ax, _), &&(bx, _)| {
                (ax - target)
                    .abs()
                    .partial_cmp(&(bx - target).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|&(_, b)| b)
            .unwrap_or(text.len())
    }

    /// Retained undo history across every field (bytes).
    pub(crate) fn enforce_edit_history_bound(&mut self) {
        const MAX_HISTORY: usize = 2 * 1024 * 1024;
        self.edit_history_bytes = self.editors.values().map(|s| s.history_bytes()).sum();
        while self.edit_history_bytes > MAX_HISTORY {
            // Trim the whole oldest state's history front until the
            // global bound holds — pathological multi-field edits share
            // the budget fairly by age.
            let before = self.edit_history_bytes;
            for state in self.editors.values_mut() {
                if self.edit_history_bytes <= MAX_HISTORY {
                    break;
                }
                state.trim_history(state.history_bytes() / 2);
            }
            self.edit_history_bytes = self.editors.values().map(|s| s.history_bytes()).sum();
            if self.edit_history_bytes == before {
                break;
            }
        }
    }
}

/// What a `password` field draws and measures: one bullet per character
/// of the real text, so caret, selection and pointer geometry stay
/// consistent while the string stays hidden. Editing — and what
/// copy/cut put on the clipboard — still sees the real string.
fn password_display(text: &str, enabled: bool) -> std::borrow::Cow<'_, str> {
    if enabled {
        std::borrow::Cow::Owned("\u{2022}".repeat(text.chars().count()))
    } else {
        std::borrow::Cow::Borrowed(text)
    }
}

/// Byte offset in the masked display string for a real byte offset —
/// a bullet is `'\u{2022}'` (3 bytes) per source char.
fn password_display_byte(text: &str, byte: usize, enabled: bool) -> usize {
    if enabled {
        text[..byte].chars().count() * '\u{2022}'.len_utf8()
    } else {
        byte
    }
}

/// Real byte offset for a byte offset into the masked display string.
fn password_real_byte(text: &str, byte: usize, enabled: bool) -> usize {
    if enabled {
        text.char_indices()
            .nth(byte / '\u{2022}'.len_utf8())
            .map(|(b, _)| b)
            .unwrap_or(text.len())
    } else {
        byte
    }
}
