use super::{
    ClipScope, LAYER_MODAL, LAYER_POPUP, LAYER_TOAST, LAYER_TOOLTIP, LabelStyle, ModalResponse,
    Toast, ToastKind, Toasts, TooltipDwell, Ui,
};
use crate::ui::font::FontId;
use crate::ui::tween::Ease;
use crate::ui::{Color, Rect};

impl Ui {
    /// A modal dialog: dim backdrop, rounded card, scale/fade entrance.
    /// Draw card content into the returned rect, then `ui.end_clip()` —
    /// which also restores the layer and enclosing clips the modal
    /// suspended. `dismissed` reports backdrop click or Escape.
    ///
    /// The card draws on [`LAYER_MODAL`] and registers a modal input
    /// surface: widgets beneath it stop receiving pointer and keyboard
    /// input from the next frame on (a host can cover the opening frame
    /// with [`Ui::declare_input_surface`]).
    pub fn modal(&mut self, label: &str, open: bool, size: (f32, f32)) -> Option<ModalResponse> {
        let id = self.id(label);
        self.push_id(label);
        let a = self.anim_f32("##open", if open { 1.0 } else { 0.0 }, 0.18, Ease::BackOut);
        self.pop_id();
        if a < 0.01 && !open {
            return None;
        }
        let screen = self.screen;
        let card = Rect::new(
            (screen.0 - size.0) * 0.5,
            (screen.1 - size.1) * 0.5 + 8.0 * (1.0 - a),
            size.0,
            size.1,
        );
        // Scale from the centre: inset by the inverse of the anim.
        let scale = 0.86 + 0.14 * a.min(1.0);
        let center = card.center();
        let card = Rect::new(
            center.0 - card.w * scale * 0.5,
            center.1 - card.h * scale * 0.5,
            card.w * scale,
            card.h * scale,
        );
        // Escape enclosing clips and rise to the modal layer; `end_clip`
        // puts it all back. A modal opened from an overlay rises one
        // layer above it so the inner modal really owns input.
        let current = self.draw.layer();
        let layer = if current >= LAYER_POPUP {
            LAYER_MODAL.max(current + 1)
        } else {
            LAYER_MODAL
        };
        let saved_layer = self.draw.set_layer(layer);
        let saved_clips = self.draw.suspend_clip();
        // Backdrop: dim everything, block input outside the card.
        self.draw.rect(
            Rect::new(0.0, 0.0, screen.0, screen.1),
            Color::rgba(0x0e, 0x0e, 0x10, (150.0 * a) as u8),
        );
        let theme = self.theme;
        self.elevation_shadow_faded(card, 10.0, crate::ui::tokens::Elevation::Modal, a);
        self.draw.rounded_frame(
            card,
            10.0,
            crate::ui::layout::hairline(self.scale),
            theme.g1,
            theme.border,
        );
        self.draw.push_clip(card.inset(2.0));
        self.push_clip_scope(ClipScope::Overlay {
            saved_layer,
            saved_clips,
        });
        self.register_surface(id, card, self.draw.layer(), true);
        let dismissed = (self.input.mouse_pressed && !card.contains(self.input.mouse))
            || (open && self.input.nav.cancel && a > 0.5);
        Some(ModalResponse {
            card,
            dismissed,
            alpha: a,
        })
    }

    /// A hover tooltip above `rect` after `tokens.motion.tooltip_delay`
    /// seconds of uninterrupted hover. Call right after the widget it
    /// describes.
    ///
    /// The dwell belongs to the widget's rect: leaving it, or a frame where
    /// the widget never ran, resets the clock — unrelated hovers never pool
    /// their time. The tip itself is noninteractive and draws on
    /// [`LAYER_TOOLTIP`], escaping the widget's clip.
    pub fn tooltip(&mut self, rect: Rect, text: &str, font: FontId) {
        let frame = self.frame;
        let mut shown = false;
        if self.hit(rect) {
            // The clock only continues when this is the same widget, still
            // hovered, frame on frame. Entry — and reset after a skip —
            // start at zero on this frame: the entry frame's own `dt`
            // measures time that ran *before* the pointer arrived, so a
            // long idle gap must not count as dwell.
            let continuous = self.tooltip_dwell.is_some_and(|dwell| {
                dwell.rect == rect && frame.saturating_sub(dwell.last_frame) <= 1
            });
            if continuous {
                let dwell = self.tooltip_dwell.as_mut().expect("checked above");
                dwell.elapsed += self.input.dt;
                dwell.last_frame = frame;
            } else {
                self.tooltip_dwell = Some(TooltipDwell {
                    rect,
                    elapsed: 0.0,
                    last_frame: frame,
                });
            }
            let elapsed = self.tooltip_dwell.expect("set above").elapsed;
            let delay = self.theme.tokens.motion.tooltip_delay;
            shown = elapsed >= delay;
            if !shown {
                // The tip is a pending wall clock, not motion: the host may
                // park, but must wake in time to show it.
                self.request_wake(delay - elapsed);
            }
        } else if self.tooltip_dwell.is_some_and(|dwell| dwell.rect == rect) {
            self.tooltip_dwell = None;
        }
        if !shown {
            return;
        }
        let metrics = self.text_metrics();
        let (w, _) = self.measure(text, font, metrics.tooltip);
        let pad = self.text_metrics().menu_pad;
        let tip = Rect::new(
            rect.center().0 - (w + pad * 2.0) * 0.5,
            rect.y - 30.0,
            w + pad * 2.0,
            24.0,
        );
        let save = self.overlay_begin(LAYER_TOOLTIP);
        let theme = self.theme;
        self.elevation_shadow(tip, 8.0, crate::ui::tokens::Elevation::Overlay);
        self.draw.rounded_frame(
            tip,
            8.0,
            crate::ui::layout::hairline(self.scale),
            theme.g2,
            theme.border,
        );
        self.label(
            Rect::new(tip.x + pad, tip.y, w, tip.h),
            text,
            LabelStyle::new(font)
                .size(metrics.tooltip)
                .color(theme.fg)
                .vcenter(true),
        );
        self.overlay_end(save);
    }

    // --- Toasts -------------------------------------------------------------

    /// Render the toast stack (bottom-right). Toasts age out on their own;
    /// call once per frame after all panels.
    ///
    /// Motion carries the whole lifecycle: each card slides in from the
    /// window edge, holds, then slides back out as it expires. Cards are
    /// solid, borderless and monochrome, drawn from the theme's surfaces.
    pub fn toasts(&mut self, toasts: &mut Toasts) {
        if toasts.items.is_empty() {
            return;
        }
        let dt = self.input.dt;
        let screen = self.screen;
        let theme = self.theme;
        // Toasts float over everything and never take input: they escape
        // the caller's clips on their own layer, then restore it all.
        let save = self.overlay_begin(LAYER_TOAST);
        let metrics = self.text_metrics();
        let font = metrics.font;
        let text_size = metrics.body;
        let pad_x = metrics.toast_pad_x;
        let pad_y = metrics.toast_pad_y;
        let gap = self.text_metrics().menu_pad;
        let mut bottom = screen.1 - 14.0;
        for toast in toasts.items.iter_mut() {
            // A toast's clock starts on its first *rendered* frame: one
            // pushed during a 20 s idle park shows fresh instead of
            // arriving expired. Rendered ids persist so toasts age only
            // while actually on screen.
            let seen_before = self.toasts_prev.contains(&toast.id);
            self.toasts_cur.insert(toast.id);
            if seen_before {
                toast.age += dt;
            }
            if toast.age >= toast.ttl {
                continue;
            }
            // Size the card from the measured text so padding is even on all
            // four sides and the label centres against the card's own box.
            let (text_w, text_h) = self.measure(&toast.text, font, text_size);
            let card = Rect::new(
                0.0,
                0.0,
                (text_w + pad_x * 2.0).round(),
                (text_h + pad_y * 2.0).max(32.0).round(),
            );
            self.push_id_num(toast.id);
            // A resting toast asks for one frame when it is time to leave;
            // it is a wall-clock wait, not animation. The scaled motion
            // duration is what the fade actually takes, so the hand-off
            // waits for the fade's real length rather than the authored
            // one (at scale 0 both are zero and nothing changes).
            let motion = self.motion_time(self.theme.tokens.motion.toast);
            let until_leave = toast.ttl - motion - toast.age;
            if until_leave > 0.0 {
                self.request_wake(until_leave);
            }
            let leaving = toast.age + motion >= toast.ttl;
            // `anim_f32` scales the duration itself — pass the authored
            // value so the fade and the wall-clock hand-off agree.
            let visible = self.anim_f32(
                "vis",
                if leaving { 0.0 } else { 1.0 },
                self.theme.tokens.motion.toast,
                if leaving {
                    Ease::CubicIn
                } else {
                    Ease::CubicOut
                },
            );
            self.pop_id();
            let travel = card.w + 28.0;
            let rect = Rect::new(
                screen.0 - 16.0 - card.w + (1.0 - visible) * travel,
                bottom - card.h,
                card.w,
                card.h,
            );
            // Monochrome raised surface with a soft drop shadow and hairline border.
            self.elevation_shadow_faded(
                rect,
                6.0,
                crate::ui::tokens::Elevation::Overlay,
                visible.clamp(0.0, 1.0),
            );
            self.draw.rounded_frame(
                rect,
                6.0,
                crate::ui::layout::hairline(self.scale),
                theme.g2,
                theme.border,
            );
            self.label(
                Rect::new(rect.x + pad_x, rect.y, text_w, rect.h),
                &toast.text,
                LabelStyle::new(font)
                    .size(text_size)
                    .color(theme.fg)
                    .vcenter(true),
            );
            bottom -= card.h + gap;
        }
        self.overlay_end(save);
        toasts.items.retain(|toast| toast.age < toast.ttl);
    }
}

impl Toasts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, text: impl Into<String>, kind: ToastKind) {
        // Bounded: a burst of events must not stack an unbounded number of
        // toasts, which would needlessly enlarge the per-frame UI buffers.
        const MAX_TOASTS: usize = 5;
        if self.items.len() == MAX_TOASTS {
            self.items.remove(0);
        }
        self.items.push(Toast {
            id: self.next_id,
            text: text.into(),
            age: 0.0,
            ttl: 3.2,
            kind,
        });
        self.next_id += 1;
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
