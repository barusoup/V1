use super::{
    CONTEXT_MENU_W, LAYER_POPUP, LabelStyle, MENU_TEXT_INSET, MenuItem, MenuStyle, Ui,
    display_label, mix_color,
};
use crate::ui::Rect;
use crate::ui::layout::Cursor;
use crate::ui::tokens::Radius;
use crate::ui::tween::Ease;

impl Ui {
    /// A vertical menu that navigates with `input.nav` and selects with
    /// `nav.confirm` or a pointer click. Returns the chosen item index.
    pub fn menu(
        &mut self,
        label: &str,
        rect: Rect,
        items: &[MenuItem],
        focus: &mut usize,
        style: MenuStyle,
    ) -> Option<usize> {
        self.menu_inner(label, rect, items, focus, style, true, 1.0)
    }

    /// [`Self::menu`] with input optionally suppressed. An animated popup keeps
    /// drawing itself (and keeps its focus highlight) while it plays its exit,
    /// which is why the input gate is a parameter rather than a separate path.
    fn menu_inner(
        &mut self,
        label: &str,
        rect: Rect,
        items: &[MenuItem],
        focus: &mut usize,
        style: MenuStyle,
        interactive: bool,
        opacity: f32,
    ) -> Option<usize> {
        // The menu's own id, captured before the scope it pushes below.
        let menu_id = self.id(label);
        self.push_id(label);
        let mut selection = None;
        if items.is_empty() {
            self.pop_id();
            return None;
        }
        *focus = (*focus).min(items.len() - 1);
        // A row is dimmed down to whether the *pointer* is on it or the
        // *keyboard* put focus there — tracked separately, because focus starts
        // at row 0 and lighting it regardless is what made a freshly opened
        // menu look like it was hovering itself.
        let pointer_in_menu = rect.contains(self.input.mouse);
        let keyboard_allowed = interactive && self.input_allowed(menu_id, rect);
        if interactive {
            if keyboard_allowed && (self.input.nav.up || self.input.nav.down) {
                self.menu_keyboard.insert(menu_id, self.frame);
            } else if pointer_in_menu {
                self.menu_keyboard.remove(&menu_id);
            }
        } else {
            // The popup stops taking input on the way out; the next opening
            // starts from the pointer, not from the last keyboard row.
            self.menu_keyboard.remove(&menu_id);
        }
        let keyboard = keyboard_allowed && self.menu_keyboard.contains_key(&menu_id);
        // Navigation skips disabled entries.
        let step = |from: usize, dir: i32| -> usize {
            let mut index = from as i32;
            for _ in 0..items.len() {
                index = (index + dir).rem_euclid(items.len() as i32);
                if items[index as usize].enabled {
                    return index as usize;
                }
            }
            from
        };
        if keyboard_allowed && self.input.nav.up {
            *focus = step(*focus, -1);
        }
        if keyboard_allowed && self.input.nav.down {
            *focus = step(*focus, 1);
        }

        let mut cursor = Cursor::column(rect.inset(style.padding), style.gap, style.item_height);
        for (index, item) in items.iter().enumerate() {
            let Some(item_rect) = cursor.next() else {
                break;
            };
            self.push_id(&item.label);
            let id = self.id("##menu_item");
            let hovered = interactive && item.enabled && self.hit(item_rect);
            let is_focus = index == *focus;
            let clicked = (item.enabled && interactive && self.click(id, hovered))
                || (keyboard_allowed && is_focus && self.input.nav.confirm && item.enabled);
            if clicked {
                selection = Some(index);
            }
            if hovered {
                *focus = index;
            }
            let hover = self.hover_level(id, hovered || (keyboard && is_focus));
            let bg = mix_color(style.item, style.item_hover, hover).fade(opacity);
            // Rows are concentric with the popup: inner radius = menu
            // radius minus the padding ring around them.
            self.draw
                .round_rect(item_rect, Radius::inner(style.radius, style.padding), bg);
            let text = if item.enabled {
                style.text
            } else {
                self.theme.faint
            }
            .fade(opacity);
            // Inset the label from both edges of the row: the hover card is
            // the row, and text sitting on its edge reads as misaligned.
            let text_rect = Rect::new(
                item_rect.x + MENU_TEXT_INSET,
                item_rect.y,
                (item_rect.w - MENU_TEXT_INSET * 2.0).max(0.0),
                item_rect.h,
            );
            self.label(
                text_rect,
                display_label(&item.label),
                LabelStyle::new(style.font)
                    .size(style.size)
                    .color(text)
                    .align(style.align)
                    .vcenter(true),
            );
            self.pop_id();
        }
        self.pop_id();
        selection
    }

    /// The rect a [`Self::context_menu`] occupies at rest, clamped to stay on
    /// screen. Callers use it to confine pointer input while the menu is up.
    pub fn context_menu_rect(&self, at: (f32, f32), items: &[MenuItem], style: &MenuStyle) -> Rect {
        let count = items.len().max(1) as f32;
        let height =
            style.padding * 2.0 + count * style.item_height + (count - 1.0).max(0.0) * style.gap;
        let mut rect = Rect::new(at.0, at.1, CONTEXT_MENU_W, height);
        let screen = self.screen;
        rect.x = rect.x.min(screen.0 - 4.0 - rect.w).max(4.0);
        rect.y = rect.y.min(screen.1 - 4.0 - rect.h).max(4.0);
        rect
    }

    /// Animated context menu anchored at `at`.
    ///
    /// `open` is the caller's latch: the menu clears it when an item is chosen
    /// or the pointer clicks away. While it animates shut it keeps drawing —
    /// sliding back toward its anchor — and stops taking input, so dismissing
    /// is a motion rather than a pop. Returns the chosen item index, if any.
    ///
    /// The caller keeps calling this every frame: it costs nothing while
    /// closed, and it is what drives the exit animation.
    pub fn context_menu(
        &mut self,
        label: &str,
        open: &mut bool,
        at: (f32, f32),
        items: &[MenuItem],
        focus: &mut usize,
        style: MenuStyle,
    ) -> Option<usize> {
        self.push_id(label);
        // Travel is the exit only, so a freshly opened popup rests at 1.0 with
        // nothing to animate: closing always starts from the resting position,
        // even when the menu is dismissed the instant it appears.
        let t = self.anim_f32(
            "##popup",
            if *open { 1.0 } else { 0.0 },
            if *open {
                0.0
            } else {
                self.theme.tokens.motion.popup_close
            },
            if *open { Ease::Linear } else { Ease::CubicIn },
        );
        if !*open && t <= 0.002 {
            self.pop_id();
            return None;
        }
        // Opacity rides its own channel so the card is gone before it stops
        // moving: quick to arrive, quicker to leave than the slide.
        let fade = self.anim_f32(
            "##popup_fade",
            if *open { 1.0 } else { 0.0 },
            if *open {
                self.theme.tokens.motion.popup_fade_in
            } else {
                self.theme.tokens.motion.popup_fade_out
            },
            if *open { Ease::QuadOut } else { Ease::ExpoOut },
        );
        let rest = self.context_menu_rect(at, items, &style);
        // Arrive in place; leave by travelling back out of the corner the menu
        // was opened from.
        let direction = if rest.y >= at.1 { -1.0 } else { 1.0 };
        let animated = if *open {
            rest
        } else {
            rest.translate(
                0.0,
                (1.0 - t) * self.theme.tokens.motion.popup_slide * direction,
            )
        };
        // The popup escapes the caller's clips and draws on its own layer;
        // registering it as an input surface keeps clicks landing on the
        // menu from reaching whatever sits behind it (from the next frame
        // on — the host can predeclare with `declare_input_surface` to cover
        // the opening frame too).
        let save = self.overlay_begin(LAYER_POPUP);
        let layer = self.draw.layer();
        let surface_id = self.id("##surface");
        self.register_surface(surface_id, animated, layer, false);
        self.elevation_shadow_faded(
            animated,
            style.radius,
            crate::ui::tokens::Elevation::Overlay,
            fade,
        );
        self.draw.rounded_frame(
            animated,
            style.radius,
            1.0,
            style.background.fade(fade),
            style.border.fade(fade),
        );
        self.draw.push_clip(animated.inset(1.0));
        let chosen = self.menu_inner(label, animated, items, focus, style, *open, fade);
        self.draw.pop_clip();
        self.overlay_end(save);
        self.pop_id();
        if chosen.is_some() {
            *open = false;
            return chosen;
        }
        let outside = *open && self.input.mouse_pressed && !animated.contains(self.input.mouse);
        if outside || (*open && self.input.nav.cancel) {
            *open = false;
        }
        None
    }
}
