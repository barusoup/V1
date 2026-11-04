use super::{ClipKind, ClipScope, Response, Ui, display_label, mix_color};
use crate::ui::font::FontId;
use crate::ui::tween::Ease;
use crate::ui::{Color, Rect};

impl Ui {
    /// Collapsible section: a header row with a chevron and a body that
    /// unrolls open or rolls shut over a tenth of a second.
    ///
    /// Returns the body rect and whether a clip is open. Draw the body's
    /// contents at that rect's own coordinates and pop the clip with
    /// [`Self::collapsible_end`]: the clip grows from the header downwards, so
    /// the contents are revealed in place rather than sliding, and dismissing
    /// a section is a motion rather than a pop. The body rect is identical
    /// whether the section is open, shut or mid-stroke — only the clip moves,
    /// so rows never shift under the pointer — and the chevron turns over at
    /// half travel instead of snapping on the click.
    pub fn collapsible(
        &mut self,
        label: &str,
        rect: Rect,
        open: &mut bool,
        font: FontId,
    ) -> (Rect, bool) {
        let control = self
            .theme
            .tokens
            .control(crate::ui::tokens::ControlSize::Md);
        let header_h = control.height;
        let header = Rect::new(rect.x, rect.y, rect.w, header_h);
        let id = self.id(label);
        // The header is the container: `rect` also covers the body, so testing
        // it would light the header while the pointer is down among the rows
        // (or over a shut section's hidden body).
        let hovered = self.hit(header);
        let clicked = self.click(id, hovered);
        if clicked {
            *open = !*open;
        }
        let theme = self.theme;
        let hover = self.hover_level(id, hovered);
        let bg = mix_color(Color::TRANSPARENT, theme.g2, hover);
        self.draw.round_rect(header, 6.0, bg);
        // Advance the motion before the chevron so the two share one progress
        // value: an `Expo` start makes the open decisive, and the close
        // accelerates away rather than easing out.
        self.push_id(label);
        let t = self.anim_f32(
            "##section",
            if *open { 1.0 } else { 0.0 },
            if *open {
                theme.tokens.motion.section_open
            } else {
                theme.tokens.motion.section_close
            },
            if *open { Ease::ExpoOut } else { Ease::CubicIn },
        );
        self.pop_id();
        let chevron = if t > 0.5 {
            crate::ui::icons::Icon::ChevronDown
        } else {
            crate::ui::icons::Icon::ChevronRight
        };
        // Chevron in a square slot of side `control.icon`, ink-centred;
        // the label starts one `gap` after the slot.
        self.icon(
            Rect::new(
                header.x + control.pad_x * 0.5,
                header.y + (header_h - control.icon) * 0.5,
                control.icon,
                control.icon,
            ),
            chevron,
            control.icon as u32,
            theme.muted,
        );
        let text_x = header.x + control.pad_x * 0.5 + control.icon + control.gap;
        self.label(
            Rect::new(
                text_x,
                header.y,
                (header.max().0 - text_x).max(0.0),
                header_h,
            ),
            display_label(label),
            // Section headers read the Label role (medium weight).
            {
                let mut st = self.type_style(crate::ui::tokens::TypeRole::Label);
                if font != crate::ui::font::FontId(0) {
                    st.font = font;
                }
                st
            },
        );
        let body = Rect::new(
            rect.x,
            header.max().1 + 4.0,
            rect.w,
            (rect.h - header_h - 4.0).max(0.0),
        );
        let clipped = t > 0.002;
        if clipped {
            self.draw
                .push_clip(Rect::new(body.x, body.y, body.w, body.h * t));
            self.push_clip_scope(ClipScope::Section);
        }
        (body, clipped)
    }

    /// Pop the clip opened by [`Self::collapsible`].
    pub fn collapsible_end(&mut self) {
        self.end_clip_scope(&[ClipKind::Section]);
    }

    /// Scroll view: clip to `rect`, offset content by its scroll offset,
    /// draw a slim scrollbar over it. Returns the content rect for children.
    ///
    /// The wheel goes to the deepest scroll view the pointer can reach, so
    /// a nested list scrolls inside its page instead of the page stealing
    /// the wheel — one tick is consumed once, and the leftover continues
    /// the enclosing view when this one is at its end (scroll chaining).
    /// Notches glide (see [`ScrollConfig`](super::ScrollConfig)); the thumb
    /// drags, and clicking the track pages toward the click.
    pub fn scroll_begin(&mut self, label: &str, rect: Rect, content_h: f32) -> Rect {
        let id = self.id(label);
        let offset = self.scroll_viewport(id, rect, content_h);
        self.register_scroll_view(id, rect);
        self.draw.push_clip(rect);
        self.push_clip_scope(ClipScope::Scroll);
        // The thumb belongs over the content, so it is painted when the
        // view closes (and stays under the enclosing clip, which is what
        // keeps a half-covered view's bar inside the part that shows).
        self.scroll_thumbs.push((id, rect, offset, content_h));
        Rect::new(rect.x, rect.y - offset, rect.w, content_h)
    }

    /// End a scroll view opened with [`Self::scroll_begin`].
    pub fn scroll_end(&mut self) {
        // Only a call that actually closes a scroll scope consumes its
        // thumb — an unbalanced end must not steal the enclosing view's.
        let closes = matches!(self.open_clip_kind(), Some(ClipKind::Scroll) | None);
        self.end_clip_scope(&[ClipKind::Scroll]);
        if closes && let Some((id, rect, offset, content_h)) = self.scroll_thumbs.pop() {
            self.draw_scrollbar(id, rect, offset, content_h);
        }
    }

    /// Draggable splitter between two panels. Drag changes `fraction`
    /// (0..1 of the available length) with clamping.
    pub fn splitter(&mut self, label: &str, rect: Rect, fraction: &mut f32) -> Response {
        let id = self.id(label);
        let hovered = self.hit(rect);
        let mut changed = false;
        if self.input.mouse_pressed && hovered && self.try_capture(id) {
            self.focus_id = Some(id);
        }
        if self.active_id == Some(id) && self.capture_allowed(id, rect) {
            if self.input.mouse_down {
                if rect.w >= rect.h {
                    *fraction = ((self.input.mouse.0 - rect.x) / rect.w.max(1.0)).clamp(0.1, 0.9);
                } else {
                    *fraction = ((self.input.mouse.1 - rect.y) / rect.h.max(1.0)).clamp(0.1, 0.9);
                }
                changed = true;
            } else {
                self.active_id = None;
            }
        }
        self.note_focusable(id, rect, true, false);
        let hover = self.hover_level(id, hovered || self.active_id == Some(id));
        let (w, h) = (rect.w, rect.h);
        let grab = Rect::new(
            rect.center().0 - (w.max(h) * 0.5 + 3.0) * if w >= h { 1.0 } else { 0.0 },
            rect.center().1 - (w.max(h) * 0.5 + 3.0) * if w < h { 1.0 } else { 0.0 },
            if w >= h { w + 6.0 } else { 6.0 },
            if w >= h { 6.0 } else { h + 6.0 },
        );
        let t = hover.max(if self.active_id == Some(id) { 0.6 } else { 0.0 });
        let theme = self.theme;
        let color = mix_color(theme.g2, theme.g3, t);
        self.draw.round_rect(grab, 3.0, color);
        Response::build(rect, self, id, changed, 6.0)
    }

    /// Numeric drag field: click-drag horizontally to scrub the value;
    /// shows the label and a formatted readout. Engine inspector staple.
    pub fn drag_field(
        &mut self,
        label: &str,
        rect: Rect,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        font: FontId,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) =
            super::headless::DragBehavior::new(label, value, range).update(self, rect);
        super::paint::drag_field(self, rect, font, state, label, *value);
        self.focus_ring(rect, 6.0, id);
        resp
    }
    /// Virtualized scroll list: builds only the rows the view can show
    /// (plus one row of overscan each side), identified by their stable
    /// index — a 100k-row list costs a handful of widgets per frame.
    /// Scrolls by the same wheel/thumb rules as [`Self::scroll_begin`];
    /// `f` receives each visible row's index and rect inside the scrolled
    /// clip. Rows are balanced automatically even if `f` returns early —
    /// an early `return` inside the closure exits only the closure.
    pub fn virtual_list(
        &mut self,
        label: &str,
        rect: Rect,
        count: usize,
        row_height: f32,
        gap: f32,
        mut row_fn: impl FnMut(&mut Self, usize, Rect),
    ) {
        if !rect.w.is_finite() || !rect.h.is_finite() || rect.w <= 0.0 || rect.h <= 0.0 {
            return;
        }
        let row = if row_height.is_finite() && row_height > 0.0 {
            row_height
        } else {
            return;
        };
        let gap = if gap.is_finite() { gap.max(0.0) } else { 0.0 };
        let stride = row + gap;
        if !stride.is_finite() || stride <= 0.0 {
            return;
        }
        // `count * stride` can overflow f32 cheaply — go through f64 and
        // saturate at the scroll content bound.
        let content_h = (count as f64 * f64::from(stride) - f64::from(gap))
            .max(0.0)
            .min(f32::MAX as f64) as f32;
        // Rows live under the label's own scope — two virtual lists with
        // the same row indices can't collide.
        self.push_id(label);
        let content = self.scroll_begin(label, rect, content_h.max(0.0));
        // content.y is rect.y - offset; derive the scrolled amount.
        let offset = rect.y - content.y;
        let first = ((offset / stride).floor().max(0.0) as usize).saturating_sub(1);
        let last = (((offset + rect.h) / stride).ceil() as usize)
            .saturating_add(1)
            .min(count);
        for index in first..last {
            self.push_id_num(index as u64);
            let row_rect = Rect::new(content.x, content.y + index as f32 * stride, content.w, row);
            row_fn(self, index, row_rect);
            self.pop_id();
        }
        self.pop_id();
        self.scroll_end();
    }
}
