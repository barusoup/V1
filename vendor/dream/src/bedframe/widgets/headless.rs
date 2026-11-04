//! Headless widget behaviours: input, capture, focus and animation state,
//! no drawing. Pair one with a [`super::paint`] painter to build a widget
//! that looks stock, or supply your own painter:
//!
//! ```ignore
//! let (resp, state) = ButtonBehavior::new("apply").update(ui, rect);
//! paint::button(ui, rect, &theme.button(font), state, "apply");
//! ```
//!
//! Behaviours keep the same ids, anim keys and input semantics as the L1
//! widget functions, so animation state survives a refactor.

use super::{Response, Ui};
use crate::ui::Rect;
use crate::ui::tween::Ease;
use std::ops::RangeInclusive;

/// Everything a painter needs to know about a widget's live state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct State {
    pub hovered: bool,
    /// Captured by this widget and the pointer is still down.
    pub pressed: bool,
    pub focused: bool,
    /// Focus-visible (keyboard modality) — painters gate the ring on it.
    pub focus_visible: bool,
    pub selected: bool,
    pub disabled: bool,
    /// Animated hover level 0..1.
    pub hover: f32,
    /// Animated widget value: on/check level, slider t, pill position.
    pub value: f32,
}

fn snapshot(ui: &Ui, id: u64, enabled: bool, hovered: bool, hover: f32) -> State {
    State {
        hovered,
        pressed: ui.active_id == Some(id) && ui.input.mouse_down,
        focused: ui.focus_id == Some(id),
        focus_visible: ui.focus_visible,
        selected: false,
        disabled: !enabled,
        hover,
        value: 0.0,
    }
}

/// Press/release click + confirm-key + focus for a button-like widget.
pub struct ButtonBehavior<'a> {
    label: &'a str,
    enabled: bool,
}

impl<'a> ButtonBehavior<'a> {
    pub fn new(label: &'a str) -> Self {
        Self {
            label,
            enabled: true,
        }
    }

    pub fn enabled(mut self, e: bool) -> Self {
        self.enabled = e;
        self
    }

    /// Run the frame's interaction; returns the Response and State.
    pub fn update(self, ui: &mut Ui, rect: Rect) -> (Response, State) {
        let id = ui.id(self.label);
        self.update_id(ui, rect, id)
    }

    /// Same as [`Self::update`] with the widget id already hashed —
    /// callers generating ids without strings (segmented) use this.
    pub(crate) fn update_id(self, ui: &mut Ui, rect: Rect, id: u64) -> (Response, State) {
        let enabled = self.enabled;
        ui.blur_if_disabled(id, enabled);
        let hovered = enabled && ui.hit(rect);
        let mut clicked = enabled && ui.click(id, hovered);
        // A focused widget activates on confirm; a disabled one never does.
        if !clicked
            && enabled
            && ui.focus_id == Some(id)
            && ui.input.nav.confirm
            && ui.input_allowed(id, rect)
        {
            clicked = true;
        }
        ui.note_focusable(id, rect, enabled, false);
        let hover = ui.hover_level(id, hovered);
        let resp = Response::plain(rect, ui, id, clicked);
        (resp, snapshot(ui, id, enabled, hovered, hover))
    }
}

/// Click/confirm toggling with an animated on-level (`State.value`).
/// `anim_suffix` is the widget's existing anim key tail ("##switch",
/// "##chk") so the anim state is unchanged by the split.
pub struct ToggleBehavior<'a> {
    label: &'a str,
    value: &'a mut bool,
    enabled: bool,
    duration: f32,
    anim_suffix: &'static str,
}

impl<'a> ToggleBehavior<'a> {
    pub fn new(label: &'a str, value: &'a mut bool) -> Self {
        Self {
            label,
            value,
            enabled: true,
            duration: 0.15,
            anim_suffix: "",
        }
    }

    pub fn enabled(mut self, e: bool) -> Self {
        self.enabled = e;
        self
    }

    pub fn duration(mut self, d: f32) -> Self {
        self.duration = d;
        self
    }

    pub fn anim_suffix(mut self, s: &'static str) -> Self {
        self.anim_suffix = s;
        self
    }

    pub fn update(self, ui: &mut Ui, rect: Rect) -> (Response, State) {
        let id = ui.id(self.label);
        let enabled = self.enabled;
        ui.blur_if_disabled(id, enabled);
        let hovered = enabled && ui.hit(rect);
        let mut clicked = enabled && ui.click(id, hovered);
        if !clicked
            && enabled
            && ui.focus_id == Some(id)
            && ui.input.nav.confirm
            && ui.input_allowed(id, rect)
        {
            clicked = true;
        }
        if clicked {
            *self.value = !*self.value;
        }
        ui.note_focusable(id, rect, enabled, false);
        let hover = ui.hover_level(id, hovered);
        let on_t = ui.anim_f32_id(
            crate::ui::widgets::fnv_push2(ui.id(self.label), "", self.anim_suffix),
            if *self.value { 1.0 } else { 0.0 },
            self.duration,
            Ease::CubicOut,
        );
        let resp = Response::plain(rect, ui, id, clicked);
        let mut st = snapshot(ui, id, enabled, hovered, hover);
        st.selected = *self.value;
        st.value = on_t;
        (resp, st)
    }
}

/// Radio: click reports (never flips); `selected` comes from the caller.
pub struct RadioBehavior<'a> {
    label: &'a str,
    selected: bool,
    enabled: bool,
    duration: f32,
    anim_suffix: &'static str,
}

impl<'a> RadioBehavior<'a> {
    pub fn new(label: &'a str, selected: bool) -> Self {
        Self {
            label,
            selected,
            enabled: true,
            duration: 0.15,
            anim_suffix: "##rad",
        }
    }

    pub fn enabled(mut self, e: bool) -> Self {
        self.enabled = e;
        self
    }

    pub fn duration(mut self, d: f32) -> Self {
        self.duration = d;
        self
    }

    pub fn anim_suffix(mut self, s: &'static str) -> Self {
        self.anim_suffix = s;
        self
    }

    pub fn update(self, ui: &mut Ui, rect: Rect) -> (Response, State) {
        let id = ui.id(self.label);
        let enabled = self.enabled;
        ui.blur_if_disabled(id, enabled);
        let hovered = enabled && ui.hit(rect);
        let mut clicked = enabled && ui.click(id, hovered);
        if !clicked
            && enabled
            && ui.focus_id == Some(id)
            && ui.input.nav.confirm
            && ui.input_allowed(id, rect)
        {
            clicked = true;
        }
        ui.note_focusable(id, rect, enabled, false);
        let hover = ui.hover_level(id, hovered);
        let on_t = ui.anim_f32_id(
            crate::ui::widgets::fnv_push2(ui.id(self.label), "", self.anim_suffix),
            if self.selected { 1.0 } else { 0.0 },
            self.duration,
            Ease::CubicOut,
        );
        let resp = Response::plain(rect, ui, id, clicked);
        let mut st = snapshot(ui, id, enabled, hovered, hover);
        st.selected = self.selected;
        st.value = on_t;
        (resp, st)
    }
}

/// Pointer drag + arrow-key nudge for the slider.
pub struct SliderBehavior<'a> {
    label: &'a str,
    value: &'a mut f32,
    range: RangeInclusive<f32>,
}

impl<'a> SliderBehavior<'a> {
    pub fn new(label: &'a str, value: &'a mut f32, range: RangeInclusive<f32>) -> Self {
        Self {
            label,
            value,
            range,
        }
    }

    pub fn update(self, ui: &mut Ui, rect: Rect) -> (Response, State) {
        let id = ui.id(self.label);
        let hovered = ui.hit(rect);
        let mut clicked = false;
        // Paint-only replay: register/focus visuals run, but the value,
        // capture state and responses stay owned by the live pass.
        if !ui.view.replay && ui.input.mouse_pressed && hovered && ui.try_capture(id) {
            ui.focus_id = Some(id);
            // A pointer press steals the focus ring's visibility.
            ui.focus_visible = false;
        }
        let held = ui.active_id == Some(id) && (ui.view.replay || ui.capture_allowed(id, rect));
        if held && !ui.view.replay {
            if ui.input.mouse_down {
                let t = ((ui.input.mouse.0 - rect.x) / rect.w.max(1.0)).clamp(0.0, 1.0);
                *self.value = self.range.start() + (self.range.end() - self.range.start()) * t;
            } else {
                ui.active_id = None;
                clicked = true;
            }
        }
        ui.note_focusable(id, rect, true, false);
        let focused = ui.focus_id == Some(id) && ui.input_allowed(id, rect);
        if focused {
            let step = (self.range.end() - self.range.start()) * 0.1;
            if ui.input.nav.left {
                *self.value =
                    (*self.value - self.range.start() * 0.0 - step).max(*self.range.start());
            }
            if ui.input.nav.right {
                *self.value = (*self.value + step).min(*self.range.end());
            }
        }
        let hover = ui.hover_level(id, hovered || focused);
        let t = ((*self.value - self.range.start())
            / (self.range.end() - self.range.start()).max(1e-6))
        .clamp(0.0, 1.0);
        let resp = Response::plain(rect, ui, id, clicked);
        let mut st = snapshot(ui, id, true, hovered, hover);
        st.value = t;
        st.focused = focused;
        (resp, st)
    }
}

/// Click/drag scrub for `drag_field` (240 px sweeps the range).
pub struct DragBehavior<'a> {
    label: &'a str,
    value: &'a mut f32,
    range: RangeInclusive<f32>,
    speed: f32,
}

impl<'a> DragBehavior<'a> {
    pub fn new(label: &'a str, value: &'a mut f32, range: RangeInclusive<f32>) -> Self {
        Self {
            label,
            value,
            range,
            speed: 240.0,
        }
    }

    /// Pixels for a full-range sweep (default 240).
    pub fn speed(mut self, px: f32) -> Self {
        self.speed = px;
        self
    }

    pub fn update(self, ui: &mut Ui, rect: Rect) -> (Response, State) {
        let id = ui.id(self.label);
        let hovered = ui.hit(rect);
        let mut changed = false;
        // Paint-only replay: no capture, anchor writes or value mutation —
        // the live pass owns interaction state and responses.
        if !ui.view.replay && ui.input.mouse_pressed && hovered && ui.try_capture(id) {
            ui.drags.insert(
                id,
                super::DragAnchor {
                    pos: ui.input.mouse.0,
                    value: *self.value,
                    last_frame: ui.frame,
                },
            );
        }
        if !ui.view.replay && ui.active_id == Some(id) && ui.capture_allowed(id, rect) {
            if let Some(anchor) = ui.drags.get_mut(&id) {
                anchor.last_frame = ui.frame;
                if ui.input.mouse_down {
                    let span = (self.range.end() - self.range.start()).max(1e-6);
                    let dragged = (ui.input.mouse.0 - anchor.pos) / self.speed;
                    *self.value = (anchor.value + dragged * span)
                        .clamp(*self.range.start(), *self.range.end());
                    changed = true;
                } else {
                    ui.active_id = None;
                    ui.drags.remove(&id);
                }
            } else {
                ui.active_id = None;
            }
        }
        let hover = ui.hover_level(id, hovered);
        let resp = Response::plain(rect, ui, id, changed);
        (resp, snapshot(ui, id, true, hovered, hover))
    }
}

/// Cell geometry for [`Ui::segmented_with`]: one hit zone per option.
#[derive(Clone, Debug)]
pub struct SegmentedCells {
    /// `(x, width)` per segment, sharing the container's y/height.
    pub cells: Vec<(f32, f32)>,
}

/// Per-segment hit/click for `Ui::segmented_with`: ids stay
/// `format!("{opt}##seg_{i}")` under `push_id(label)`.
pub struct SegmentedBehavior<'a> {
    label: &'a str,
    options: &'a [&'a str],
    enabled: bool,
}

impl<'a> SegmentedBehavior<'a> {
    pub fn new(label: &'a str, options: &'a [&'a str]) -> Self {
        Self {
            label,
            options,
            enabled: true,
        }
    }

    pub fn enabled(mut self, e: bool) -> Self {
        self.enabled = e;
        self
    }

    /// Runs the per-segment button behaviours; returns (picked, cells,
    /// responses) — the caller animates the pill and paints.
    /// Compute the segment rects with `ui.seg_cells` scratch and run the
    /// behaviour — the public entry point.
    pub fn update(
        self,
        ui: &mut Ui,
        rect: Rect,
        selected: &mut usize,
        scale: f32,
        inset: f32,
        out: &mut Vec<(Rect, Response, State)>,
    ) -> Option<usize> {
        let mut cells = std::mem::take(&mut ui.seg_cells);
        crate::ui::layout::divide_into(&mut cells, rect.x, rect.w, self.options.len(), 0.0, scale);
        let r = self.update_with_cells(ui, rect, selected, &cells, inset, out);
        ui.seg_cells = cells;
        r
    }

    /// Internal entry: callers that already hold a cells slice (the view
    /// layer's pooled adapter path) use this to skip re-deriving them.
    pub(crate) fn update_with_cells(
        self,
        ui: &mut Ui,
        rect: Rect,
        selected: &mut usize,
        cells: &[(f32, f32)],
        inset: f32,
        out: &mut Vec<(Rect, Response, State)>,
    ) -> Option<usize> {
        out.clear();
        if self.options.is_empty() {
            return None;
        }
        *selected = (*selected).min(self.options.len() - 1);
        let mut picked = None;
        out.reserve(self.options.len());
        // The caller owns the id scope (push_id(label)) so segment ids
        // stay identical to the L1 path.
        let _ = self.label;
        for (i, &opt) in self.options.iter().enumerate() {
            let cell = Rect::new(cells[i].0, rect.y, cells[i].1, rect.h).inset(inset);
            // id = fnv_push(scope, "{opt}##seg_{i}") — hashed byte-by-
            // byte with no string allocation.
            let seg_id = seg_id(ui, opt, i);
            let (resp, st) = ButtonBehavior::new(opt)
                .enabled(self.enabled)
                .update_id(ui, cell, seg_id);
            if resp.clicked {
                *selected = i;
                picked = Some(i);
            }
            out.push((cell, resp, st));
        }
        picked
    }
}

/// Segment rects for a segmented control at `rect`: `divide_into` on the
/// track plus the per-cell `inset` — the same math [`SegmentedBehavior`]
/// uses for hit-testing, so callers publishing cell rects (selfcheck)
/// share it.
pub fn segmented_cells(rect: Rect, count: usize, scale: f32, inset: f32, out: &mut [Rect]) {
    let mut cells = Vec::new();
    crate::ui::layout::divide_into(&mut cells, rect.x, rect.w, count, 0.0, scale);
    for (dst, (x, w)) in out.iter_mut().zip(cells) {
        *dst = Rect::new(x, rect.y, w, rect.h).inset(inset);
    }
}

/// `fnv_push(scope, "{opt}##seg_{i}")` without the string.
pub(crate) fn seg_id(ui: &Ui, opt: &str, i: usize) -> u64 {
    use crate::ui::widgets::FNV_PRIME;
    let mut digits = [0u8; 20];
    let mut n = i;
    let mut d = 20;
    loop {
        d -= 1;
        digits[d] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    let digs = &digits[d..];
    let total = opt.len() + 6 + digs.len();
    let mut hash = ui.scope_seed; // current scope seed
    for b in total.to_le_bytes() {
        hash = (hash ^ u64::from(b)).wrapping_mul(FNV_PRIME);
    }
    for b in opt
        .bytes()
        .chain(b"##seg_".iter().copied())
        .chain(digs.iter().copied())
    {
        hash = (hash ^ u64::from(b)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// One segment passed to a [`Ui::segmented_with`] item closure.
pub struct SegmentItem<'a> {
    pub index: usize,
    pub label: &'a str,
    /// The segment's hit zone (inset inside the cell).
    pub rect: Rect,
    pub state: State,
    /// True the frame this segment was clicked (its Response.clicked).
    pub clicked: bool,
    /// 0..1 closeness of the pill to this segment (text tint driver).
    pub near: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `seg_id` must equal the legacy `"{opt}##seg_{i}"` label id under
    /// any scope — the allocator-free hashing is a byte-for-byte match.
    #[test]
    fn segmented_id_matches_legacy_label() {
        let mut ui = Ui::new(
            crate::ui::font::Fonts::new(),
            crate::ui::atlas::AtlasConfig::default(),
        );
        ui.begin(crate::ui::widgets::UiInput::default(), (400.0, 300.0));
        for opt in ["a", "Unicode é", "a##x"] {
            for i in [0usize, 1, 10, usize::MAX] {
                let want = ui.id(&format!("{opt}##seg_{i}"));
                assert_eq!(seg_id(&ui, opt, i), want, "root scope {opt} {i}");
            }
        }
        ui.push_id("outer");
        ui.push_id("inner");
        for opt in ["a", "Unicode é", "a##x"] {
            for i in [0usize, 1, 10, usize::MAX] {
                let want = ui.id(&format!("{opt}##seg_{i}"));
                assert_eq!(seg_id(&ui, opt, i), want, "nested scope {opt} {i}");
            }
        }
        ui.pop_id();
        ui.pop_id();
        ui.end();
    }
}
