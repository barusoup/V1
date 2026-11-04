//! Anchored panels that escape parent clips and stay inside the viewport.

use super::{LAYER_POPUP, Ui};
use crate::ui::Rect;
use crate::ui::tokens::Elevation;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FloatingSide {
    Top,
    #[default]
    Bottom,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FloatingAlign {
    #[default]
    Start,
    Center,
    End,
}

/// Placement in logical points. Panels flip to the opposite side when it has
/// more room, then shift and shrink to fit `boundary`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floating {
    pub side: FloatingSide,
    pub align: FloatingAlign,
    pub gap: f32,
}

impl Default for Floating {
    fn default() -> Self {
        Self {
            side: FloatingSide::Bottom,
            align: FloatingAlign::Start,
            gap: 4.0,
        }
    }
}

impl Floating {
    pub fn rect(self, anchor: Rect, size: (f32, f32), boundary: Rect) -> Rect {
        let finite = |n: f32| if n.is_finite() { n } else { 0.0 };
        let boundary = Rect::new(
            finite(boundary.x),
            finite(boundary.y),
            finite(boundary.w).max(0.0),
            finite(boundary.h).max(0.0),
        );
        let anchor = Rect::new(
            finite(anchor.x),
            finite(anchor.y),
            finite(anchor.w).max(0.0),
            finite(anchor.h).max(0.0),
        );
        let (w, h) = (
            finite(size.0).clamp(0.0, boundary.w),
            finite(size.1).clamp(0.0, boundary.h),
        );
        let gap = finite(self.gap).max(0.0);
        let (before, after, needed) = match self.side {
            FloatingSide::Top | FloatingSide::Bottom => (
                anchor.y - boundary.y,
                boundary.max().1 - anchor.max().1,
                h + gap,
            ),
            FloatingSide::Left | FloatingSide::Right => (
                anchor.x - boundary.x,
                boundary.max().0 - anchor.max().0,
                w + gap,
            ),
        };
        let backwards = matches!(self.side, FloatingSide::Top | FloatingSide::Left);
        let (preferred, opposite) = if backwards {
            (before, after)
        } else {
            (after, before)
        };
        let backwards = if preferred < needed && opposite > preferred {
            !backwards
        } else {
            backwards
        };
        let align = |start: f32, extent: f32, panel: f32| match self.align {
            FloatingAlign::Start => start,
            FloatingAlign::Center => start + (extent - panel) * 0.5,
            FloatingAlign::End => start + extent - panel,
        };
        let (x, y) = match self.side {
            FloatingSide::Top | FloatingSide::Bottom => (
                align(anchor.x, anchor.w, w),
                if backwards {
                    anchor.y - gap - h
                } else {
                    anchor.max().1 + gap
                },
            ),
            FloatingSide::Left | FloatingSide::Right => (
                if backwards {
                    anchor.x - gap - w
                } else {
                    anchor.max().0 + gap
                },
                align(anchor.y, anchor.h, h),
            ),
        };
        Rect::new(
            x.clamp(boundary.x, boundary.max().0 - w),
            y.clamp(boundary.y, boundary.max().1 - h),
            w,
            h,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopoverResponse {
    pub rect: Rect,
    /// Escape or a press outside both the panel and its anchor. The host owns
    /// `open`; close it after reading this response.
    pub dismissed: bool,
}

impl Ui {
    /// Build an anchored panel on an overlay layer. Call after its anchor's
    /// `layout_root` returns; the body may start its own layout root. The body
    /// is clipped to the panel and receives its padded content rect.
    ///
    /// As with other immediate-mode overlays, opening-frame input behind the
    /// panel must be predeclared with [`Self::declare_input_surface`] when
    /// widgets built earlier could already have responded to that input.
    pub fn popover(
        &mut self,
        key: &str,
        anchor: Rect,
        size: (f32, f32),
        placement: Floating,
        build: impl FnOnce(&mut Self, Rect),
    ) -> PopoverResponse {
        let id = self.id(key);
        let boundary = Rect::new(0.0, 0.0, self.screen.0, self.screen.1).inset(4.0);
        let rect = placement.rect(anchor, size, boundary);
        let layer = LAYER_POPUP.max(self.draw.layer().saturating_add(1));
        let save = self.overlay_begin(layer);
        let allowed = self.input_allowed(id, rect);
        let dismissed = allowed
            && (self.input.nav.cancel
                || (self.input.mouse_pressed
                    && !rect.contains(self.input.mouse)
                    && !anchor.contains(self.input.mouse)));
        self.register_surface(id, rect, layer, false);
        let theme = *self.theme();
        self.elevation_shadow(rect, 8.0, Elevation::Overlay);
        self.draw.rounded_frame(
            rect,
            8.0,
            crate::ui::layout::hairline(self.scale()),
            theme.g1,
            theme.border,
        );
        self.draw.push_clip(rect);
        self.push_id(key);
        build(self, rect.inset(8.0));
        self.pop_id();
        self.draw.pop_clip();
        self.overlay_end(save);
        PopoverResponse { rect, dismissed }
    }
}
