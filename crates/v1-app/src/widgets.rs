// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The small kit every screen is built from: a per-frame context, text that fits, icons, covers
//! and the slider.
//!
//! Widgets here are drawn by hand on top of Dream's draw list rather than taken from its stock
//! set, because the look (flat, borderless, spring-driven) is the whole point and the stock ones
//! paint their own chrome. Dream's text field and scroll view are still used for what they do well:
//! editing and scrolling.

use std::cell::Cell;
use std::sync::Arc;

use dream::texture::GpuTexture;
use dream::ui::draw::DrawList;
use dream::ui::font::{Align, FontId};
use dream::ui::widgets::{LabelStyle, Ui};
use dream::ui::{Color, Rect};

use crate::anim::{rate, Anims};
use crate::gpu::Gpu;
use crate::icons::{self, Icon};
use crate::theme::{self, fade};

/// Pointer state for one frame, in logical units.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pointer {
    pub pos: (f32, f32),
    pub down: bool,
    /// The press edge for this frame.
    pub pressed: bool,
    /// The press edge of the secondary (right) button for this frame.
    pub secondary: bool,
}

/// Which slider owns the drag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliderId {
    Seek,
    Volume,
}

/// A queue row being dragged by its handle. `from` is the row's place in the queue when the drag
/// began, `to` the place it is over now, and `grab` how far below the row's top the handle was held.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueueDrag {
    pub from: usize,
    pub to: usize,
    pub grab: f32,
}

impl QueueDrag {
    /// Where row `i` is drawn while the drag is on: the dragged row sits at `to`, and the rows it
    /// passes move one place to make room.
    pub fn slot(&self, i: usize) -> usize {
        if i == self.from {
            self.to
        } else if self.from < self.to && (self.from + 1..=self.to).contains(&i) {
            i - 1
        } else if self.to < self.from && (self.to..self.from).contains(&i) {
            i + 1
        } else {
            i
        }
    }
}

/// Everything a widget needs for one frame.
pub struct Cx<'a> {
    pub ui: &'a mut Ui,
    pub anims: &'a mut Anims,
    pub gpu: &'a Gpu,
    pub pointer: Pointer,
    pub font: FontId,
    /// Device pixels per logical pixel, for choosing icon texture sizes and snapping to the grid.
    pub scale: f32,
    /// True while an overlay covers the page, so what is beneath ignores the pointer.
    pub blocked: bool,
    /// While `covering` is set, anything inside one of `covers` ignores the pointer. The page is
    /// drawn first, so the surfaces drawn over it (the bar, the sidebar, the caption) are listed
    /// here up front; without this, a click on them would also reach whatever is beneath.
    pub covering: bool,
    pub covers: Vec<Rect>,
    /// The angle of the loading spinner this frame, in radians.
    pub spin: f32,
    /// Set when the pointer is over something clickable; the host turns it into a hand cursor.
    pub hand: bool,
    pub drag: Option<SliderId>,
    /// The queue row being dragged by its handle, if any.
    pub queue_drag: Option<QueueDrag>,
    /// Textures this frame drew a stand-in for.
    pub wanted_icons: Vec<(Icon, u32)>,
    pub wanted_covers: Vec<String>,
    /// Set when a hit test has landed under the pointer this frame, so a press that a widget took
    /// is not also taken as a grab of the window.
    pub touched: Cell<bool>,
    /// Which songs, albums and artists are saved, for the buttons that show it.
    pub saved: Arc<crate::library::SavedIds>,
}

impl Cx<'_> {
    pub fn draw(&mut self) -> &mut DrawList {
        self.ui.draw_list_mut()
    }

    /// Whether the pointer is over `rect` and nothing covers it.
    pub fn over(&self, rect: Rect) -> bool {
        let covered = self.covering && self.covers.iter().any(|cover| cover.contains(self.pointer.pos));
        let over = !self.blocked && !covered && rect.contains(self.pointer.pos);
        if over {
            self.touched.set(true);
        }
        over
    }

    /// Whether the pointer is over `rect`, ignoring anything that covers it.
    pub fn pointer_in(&self, rect: Rect) -> bool {
        rect.contains(self.pointer.pos)
    }

    /// Like [`Self::over`], and records that the pointer is on something clickable.
    pub fn over_clickable(&mut self, rect: Rect) -> bool {
        let over = self.over(rect);
        if over {
            self.hand = true;
        }
        over
    }

    pub fn clicked(&self, rect: Rect) -> bool {
        self.over(rect) && self.pointer.pressed
    }

    /// Like [`Self::clicked`], for the secondary (right) button.
    pub fn right_clicked(&self, rect: Rect) -> bool {
        self.over(rect) && self.pointer.secondary
    }

    /// Snap a logical coordinate to the device pixel grid.
    pub fn snap(&self, value: f32) -> f32 {
        (value * self.scale).round() / self.scale
    }

    /// A single line of text, vertically centred in `rect`.
    pub fn text(&mut self, rect: Rect, text: &str, size: f32, color: Color, align: Align) {
        if color.a == 0 || text.is_empty() {
            return;
        }
        let style = LabelStyle::new(self.font)
            .size(size)
            .color(color)
            .align(align)
            .vcenter(true);
        self.ui.label(rect, text, style);
    }

    /// Text shortened with an ellipsis so it fits `rect`'s width.
    pub fn text_fit(&mut self, rect: Rect, text: &str, size: f32, color: Color, align: Align) {
        if color.a == 0 || text.is_empty() {
            return;
        }
        let fitted = self.fit(text, size, rect.w);
        self.text(rect, &fitted, size, color, align);
    }

    pub fn text_width(&mut self, text: &str, size: f32) -> f32 {
        self.ui.measure(text, self.font, size).0
    }

    /// Text wrapped to the width of `rect`, top-aligned, with lines `leading` times the face's
    /// natural height apart. Measure it with [`Self::text_block_height`] first to lay out what
    /// follows it.
    pub fn text_block(
        &mut self,
        rect: Rect,
        text: &str,
        size: f32,
        color: Color,
        leading: f32,
        align: Align,
    ) {
        if color.a == 0 || text.is_empty() {
            return;
        }
        let style = LabelStyle::new(self.font)
            .size(size)
            .color(color)
            .align(align)
            .line_height(leading);
        self.ui.label_wrapped(rect, text, style);
    }

    /// The height [`Self::text_block`] takes for `text` wrapped to `width`.
    pub fn text_block_height(&mut self, width: f32, text: &str, size: f32, leading: f32) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        self.ui.measure_wrapped(text, self.font, size, width, leading).1
    }

    pub fn fit(&mut self, text: &str, size: f32, max_w: f32) -> String {
        if self.text_width(text, size) <= max_w {
            return text.to_owned();
        }
        let chars: Vec<char> = text.chars().collect();
        let (mut lo, mut hi) = (0usize, chars.len());
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let candidate: String = chars[..mid].iter().collect::<String>() + "…";
            if self.text_width(&candidate, size) <= max_w {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let head: String = chars[..lo].iter().collect();
        format!("{}…", head.trim_end())
    }

    /// Keep painting inside `rect` until [`Self::unclip`].
    pub fn clip(&mut self, rect: Rect) {
        self.ui.paint_push_clip(rect);
    }

    pub fn unclip(&mut self) {
        self.ui.paint_pop_clip();
    }

    /// A Lucide icon centred on `center`, `size` logical pixels square, tinted `color`.
    ///
    /// The texture is rasterized at exactly the icon's device size and drawn 1:1 onto a whole device
    /// pixel, so it stays sharp. Until that texture exists, the nearest size that does is used for
    /// the frame and the exact one is requested. Colours are opaque: state is shown by colour, never
    /// by alpha, which blurs strokes.
    pub fn icon(&mut self, icon: Icon, center: (f32, f32), size: f32, color: Color) {
        self.icon_scaled(icon, center, size, 1.0, color);
    }

    /// [`Self::icon`] drawn `factor` times its `size`. The texture is chosen by `size` alone, so an
    /// icon that scales every frame (a hover, a pop) is never re-rasterized or swapped for a
    /// neighbouring size; the change is only in how big the texture is drawn.
    pub fn icon_scaled(&mut self, icon: Icon, center: (f32, f32), size: f32, factor: f32, color: Color) {
        if color.a == 0 || size * factor <= 0.5 {
            return;
        }
        let px = icons::bucket(size * self.scale);
        let gpu = self.gpu;
        let texture = gpu.icons.get(&(icon, px)).or_else(|| {
            self.wanted_icons.push((icon, px));
            icons::nearest_buckets(px).find_map(|b| gpu.icons.get(&(icon, b)))
        });
        let Some(texture) = texture else { return };
        // Top-left corner on a whole device pixel, extent = the texture's size times `factor`.
        let dev = self.scale;
        let drawn = px as f32 * factor;
        let extent = drawn / dev;
        let x = (center.0 * dev - drawn * 0.5).round() / dev;
        let y = (center.1 * dev - drawn * 0.5).round() / dev;
        self.ui.draw_list_mut().image(Rect::new(x, y, extent, extent), texture, color);
    }

    /// The backgroundless V1 mark centred on `center`, scaled to fit inside `max` (logical pixels)
    /// with its aspect kept, tinted `color`. Nothing is drawn until its texture is ready.
    pub fn wordmark(&mut self, center: (f32, f32), max: (f32, f32), color: Color) {
        if color.a == 0 {
            return;
        }
        let Some(mark) = self.gpu.wordmark.as_ref() else { return };
        let scale = (max.0 / mark.width as f32).min(max.1 / mark.height as f32);
        // Snap the size and position to whole device pixels, so the texture is sampled 1:1 with
        // no half-pixel offset blurring its edges.
        let dev = self.scale;
        let w = (mark.width as f32 * scale * dev).round() / dev;
        let h = (mark.height as f32 * scale * dev).round() / dev;
        let x = ((center.0 * dev - w * dev * 0.5).round()) / dev;
        let y = ((center.1 * dev - h * dev * 0.5).round()) / dev;
        self.ui.draw_list_mut().image(Rect::new(x, y, w, h), &mark.texture, color);
    }

    /// One of several icons for a state. The current one is drawn; the rest are not. There is no
    /// crossfade: two half-transparent glyphs on top of each other read as blur.
    pub fn icon_state(
        &mut self,
        icons: &[Icon],
        current: usize,
        center: (f32, f32),
        size: f32,
        color: Color,
    ) {
        if let Some(icon) = icons.get(current).copied() {
            self.icon(icon, center, size, color);
        }
    }

    /// A cover, rounded, fading in when it arrives. Without it, a plain grey tile; either way the
    /// cover is requested once.
    pub fn cover(&mut self, url: Option<&str>, rect: Rect, radius: f32, alpha: f32) {
        let texture: Option<Arc<GpuTexture>> = url.and_then(|url| {
            let found = self.gpu.covers.get(url).cloned();
            if found.is_none() && !self.wanted_covers.iter().any(|u| u == url) {
                self.wanted_covers.push(url.to_owned());
            }
            found
        });
        let shown = {
            let key = url.map_or(0, hash);
            self.anims.toggle("cover", key, texture.is_some(), rate::QUICK)
        };
        self.draw().round_rect(rect, radius, fade(theme::RAISED, alpha));
        if let Some(texture) = texture {
            self.draw()
                .image_rounded(rect, &texture, fade(Color::WHITE, alpha * shown), radius);
        }
    }

    /// A seek or volume slider: a thin flat track whose fill brightens under the pointer. Returns
    /// whether the value moved this frame.
    pub fn slider(
        &mut self,
        id: SliderId,
        rect: Rect,
        value: &mut f32,
        enabled: bool,
        opacity: f32,
    ) -> bool {
        let index = id as u32;
        let hit = Rect::new(rect.x - 8.0, rect.y - 6.0, rect.w + 16.0, rect.h + 12.0);
        let inside = enabled && self.over(hit);
        if inside {
            self.hand = true;
        }

        let mut changed = false;
        if enabled {
            if self.pointer.pressed && inside {
                self.drag = Some(id);
            }
            if self.drag == Some(id) {
                if self.pointer.down {
                    let t = ((self.pointer.pos.0 - rect.x) / rect.w.max(1.0)).clamp(0.0, 1.0);
                    if (*value - t).abs() > 1e-4 {
                        *value = t;
                        changed = true;
                    }
                } else {
                    self.drag = None;
                }
            }
        } else if self.drag == Some(id) {
            self.drag = None;
        }

        let active = enabled && (inside || self.drag == Some(id));
        let glow = self.anims.toggle("slider", index, active, rate::QUICK);
        let live = self.anims.toggle("slider_on", index, enabled, rate::QUICK);
        let t = value.clamp(0.0, 1.0);

        // The track is a fixed 4 px tall, on the device grid. Its width and fill are not snapped,
        // so a slider that is still unfolding moves smoothly instead of stepping a pixel at a time.
        // Only its colour responds to the pointer.
        let height = 4.0;
        let top = self.snap(rect.y + rect.h * 0.5 - height * 0.5);
        let fill_w = rect.w * t;
        let fill = fade(
            theme::mix(Color::rgb(0xb4, 0xb4, 0xbc), theme::FG, glow),
            (0.4 + 0.6 * live) * opacity,
        );
        let draw = self.draw();
        draw.round_rect(
            Rect::new(rect.x, top, rect.w, height),
            height * 0.5,
            fade(theme::TRACK, (0.55 + 0.45 * live) * opacity),
        );
        if fill_w > 0.01 {
            // A fill narrower than the track's cap shrinks its end radius rather than growing to it.
            draw.round_rect(
                Rect::new(rect.x, top, fill_w, height),
                (height * 0.5).min(fill_w * 0.5),
                fill,
            );
        }
        changed
    }
}

pub fn hash(text: &str) -> u32 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish() as u32
}
