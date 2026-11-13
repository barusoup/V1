//! BedFrame: Dream Engine's UI, text, HUD and menus framework.
//!
//! See `docs/bedframe-cookbook.md` for task-oriented L3 recipes (toolbar,
//! inspector form, card grid, settings page, HUD overlay).
//!
//! Layers, top to bottom:
//!
//! - [`tween`]: easing, tweens and animated values. Standalone; usable for
//!   gameplay animation too.
//! - [`atlas`]: growable paged texture atlas (glyph pages, icon sheets).
//! - [`font`]: font loading, glyph caching and text layout (wrap, align).
//! - [`draw`]: the raw draw list — quads, rounded rectangles, text, images,
//!   scissors, layers. This is the "extreme developer control" surface:
//!   anything the widgets can draw, you can draw directly. Rounded shapes are
//!   resolved by a signed distance term in the UI shader, so a rounded
//!   rectangle is still one quad and batches with text.
//! - [`icons`]: the icon set, copied verbatim from [Lucide](https://lucide.dev)
//!   (stroke-only SVG on a 24x24 grid). Curves and arcs are flattened and
//!   rasterized with analytic anti-aliasing into the atlas, so icons tint like
//!   text and never change weight with state. [`pixel`] retains the legacy
//!   hand-drawn pixel icon set for tools that want it.
//! - [`layout`]: rows, columns and anchors for HUD placement.
//! - [`theme`]: the neutral dark/light palette and the sizing metrics. Themes
//!   swap instantly — motion belongs to things that move.
//! - [`tokens`]: the resolved design tokens on top of a theme — spacing,
//!   radius, density, type scale, elevation, motion and colour roles.
//! - [`widgets`]: immediate-mode widgets with persistent state and built-in
//!   animation (buttons, switches, sliders, panels, text fields, icon buttons,
//!   modals, toasts, tooltips, animated popups and context menus, collapsible
//!   sections, scroll views, splitters, colour pickers) plus
//!   [`widgets::NavInput`]-driven keyboard navigation. Core controls are a
//!   [`widgets::headless`] behaviour (input/state, no drawing) plus a
//!   [`widgets::paint`] painter (draw only), so custom widgets reuse both;
//!   the text/drag fields are the recorded exception — their editing state
//!   is a single L1 block spliced by the view layer, not split yet.
//!   [`widgets::adapters`] wraps each stock widget as an L3
//!   [`view::Widget`].
//!   [`widgets::State`] handles support keyed component state and scoped
//!   context; [`widgets::Presence`] keeps an animated exit mounted.
//! - [`view`]: the L3 declarative layer — [`Ui::layout_root`](crate::ui::widgets::Ui::layout_root) solves a
//!   flexbox-style node tree and paints it in order; interaction is
//!   synchronous against the previous frame's rects.
//! - [`devtools`]: layout outlines and a node inspector overlay for the
//!   view layer.
//! - [`render`]: [`render::UiRenderer`], which turns a [`draw::UiFrame`] into
//!   GPU draws.
//!
//! Typical frame (L3 declarative):
//!
//! ```ignore
//! ui.begin(UiInput { dt, .. }, screen);
//! ui.layout_root(area, Style::column().pad(8.0).gap(8.0), |ui| {
//!     ui.row(Style::row().fill_w().h_px(28.0).justify(Justify::SpaceBetween), |ui| {
//!         ui.text("Settings");
//!         if ui.add(Button::new("Done")).clicked { close(); }
//!     });
//!     ui.add(Slider::new("vol", &mut vol, 0.0..=1.0));
//! });
//! let frame = ui.end();
//! ```
//!
//! Or manual placement (L1 immediate mode — every `Ui::widget(rect,..)`
//! call still works unchanged):
//!
//! ```ignore
//! ui.begin(UiInput { dt, .. }, screen);
//! ui.panel(rect, PanelStyle::default());
//! if ui.button("play", rect, "Play", ButtonStyle::default()).clicked {
//!     start_game();
//! }
//! ui.label(rect, "Hello", LabelStyle::new(font).size(24.0));
//! let frame = ui.end();
//! // `Some(rect)` confines the scene pass to the viewport's own region, so
//! // the panels only ever cover pixels nobody shaded.
//! renderer.render_ui(&camera, Some(viewport), |draw| { /* scene draws */ 3 }, Some(&frame))?;
//! ```
//!
//! Colours are straight-alpha sRGB bytes, the format artists and CSS use.
//! By default ([`UiBlend::Gamma`](render::UiBlend::Gamma)) they blend as
//! encoded values, as in browsers; [`UiBlend::Linear`](render::UiBlend::Linear)
//! converts them to linear light on the way to the sRGB target instead.

pub mod app;
pub mod atlas;
pub mod clipboard;
pub mod devtools;
pub mod draw;
pub mod font;
pub mod icons;
pub mod layout;
pub mod pixel;
#[cfg_attr(target_os = "macos", path = "render_metal.rs")]
pub mod render;
pub mod theme;
pub mod tokens;
pub mod tween;
pub mod view;
pub mod widgets;

/// Test helpers shared across UI tests (real fonts, standard atlases).
#[cfg(test)]
pub(crate) mod test_util {
    use super::atlas::AtlasConfig;
    use super::font::Fonts;

    /// The font bundled with the engine; rasterization and metrics are
    /// meaningless for a synthetic font, and a bundled face keeps tests
    /// identical on every platform.
    pub fn test_font() -> Fonts {
        let mut fonts = Fonts::new();
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/fonts/Roboto-Regular.ttf"
        );
        fonts
            .load_file(path)
            .expect("bundled Roboto-Regular.ttf must load");
        fonts
    }

    pub fn test_atlas_config() -> AtlasConfig {
        AtlasConfig {
            page_size: 512,
            padding: 1,
        }
    }
}

/// sRGB colour with straight alpha, 0..255 per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 255)
    }

    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const TRANSPARENT: Self = Self::rgba(0, 0, 0, 0);

    /// Same colour with a different alpha.
    pub const fn with_alpha(self, a: u8) -> Self {
        Self { a, ..self }
    }

    /// Scale the alpha channel (for fades); clamps to 0..255.
    pub fn fade(self, factor: f32) -> Self {
        Self {
            a: (self.a as f32 * factor.clamp(0.0, 1.0)) as u8,
            ..self
        }
    }

    /// Pack as the vertex-shader attribute order.
    pub const fn to_array(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Construct an opaque sRGB colour from a 24-bit `0xRRGGBB` literal.
    pub const fn hex(rgb: u32) -> Self {
        Self::rgb(
            ((rgb >> 16) & 0xff) as u8,
            ((rgb >> 8) & 0xff) as u8,
            (rgb & 0xff) as u8,
        )
    }

    /// Construct a colour from a 32-bit `0xRRGGBBAA` literal.
    pub const fn hex_rgba(rgba: u32) -> Self {
        Self::rgba(
            ((rgba >> 24) & 0xff) as u8,
            ((rgba >> 16) & 0xff) as u8,
            ((rgba >> 8) & 0xff) as u8,
            (rgba & 0xff) as u8,
        )
    }

    /// Parse a CSS-style hex colour string (`#rgb`, `#rrggbb`, `#rrggbbaa`).
    pub fn from_hex(text: &str) -> Option<Self> {
        let s = text.trim().strip_prefix('#').unwrap_or(text.trim());
        match s.len() {
            3 => {
                let r = u8::from_str_radix(&s[0..1], 16).ok()?;
                let g = u8::from_str_radix(&s[1..2], 16).ok()?;
                let b = u8::from_str_radix(&s[2..3], 16).ok()?;
                Some(Self::rgb(r * 17, g * 17, b * 17))
            }
            6 => {
                let v = u32::from_str_radix(s, 16).ok()?;
                Some(Self::hex(v))
            }
            8 => {
                let v = u32::from_str_radix(s, 16).ok()?;
                Some(Self::hex_rgba(v))
            }
            _ => None,
        }
    }

    /// Premultiplied interpolation toward `other` by `t` in `0..=1`.
    pub fn mix(self, other: Self, t: f32) -> Self {
        <Self as tween::LerpValue>::lerp_value(self, other, t)
    }

    /// Lift toward white by `amount` (`0..=1`).
    pub fn lighten(self, amount: f32) -> Self {
        self.mix(Self::WHITE.with_alpha(self.a), amount)
    }

    /// Shade toward black by `amount` (`0..=1`).
    pub fn darken(self, amount: f32) -> Self {
        self.mix(Self::BLACK.with_alpha(self.a), amount)
    }
}

impl tween::LerpValue for Color {
    /// Interpolate in premultiplied alpha.
    ///
    /// A straight per-channel lerp between a transparent colour and an opaque
    /// one darkens the whole path: at half time it holds *half* of the target
    /// colour at *half* alpha, and the renderer's `src_alpha` blend multiplies
    /// by that alpha a second time, so a fade from nothing composites darker
    /// than either endpoint before it arrives. Weighting each endpoint by its
    /// own alpha and dividing out the interpolated alpha avoids the dip: the
    /// colour stays at its final value and only its coverage ramps.
    fn lerp_value(a: Self, b: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mix = |x: f32, y: f32| x + (y - x) * t;
        let alpha = mix(a.a as f32, b.a as f32);
        let channel = |x: u8, y: u8| {
            if alpha <= 0.0 {
                return 0;
            }
            // Premultiplied endpoints, divided back out by the mixed alpha.
            let premultiplied = mix(x as f32 * a.a as f32, y as f32 * b.a as f32);
            (premultiplied / alpha).round().clamp(0.0, 255.0) as u8
        };
        Self {
            r: channel(a.r, b.r),
            g: channel(a.g, b.g),
            b: channel(a.b, b.b),
            a: alpha.round().clamp(0.0, 255.0) as u8,
        }
    }
}

/// Axis-aligned rectangle in pixel space (origin top-left, y down).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub const fn from_min_max(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            x: min_x,
            y: min_y,
            w: max_x - min_x,
            h: max_y - min_y,
        }
    }

    pub fn min(self) -> (f32, f32) {
        (self.x, self.y)
    }

    pub fn max(self) -> (f32, f32) {
        (self.x + self.w, self.y + self.h)
    }

    pub fn center(self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }

    pub fn size(self) -> (f32, f32) {
        (self.w, self.h)
    }

    pub fn contains(self, point: (f32, f32)) -> bool {
        point.0 >= self.x
            && point.1 >= self.y
            && point.0 < self.x + self.w
            && point.1 < self.y + self.h
    }

    /// Grow or shrink on every side; negative values inset.
    pub fn expand(self, amount: f32) -> Self {
        Self {
            x: self.x - amount,
            y: self.y - amount,
            w: self.w + amount * 2.0,
            h: self.h + amount * 2.0,
        }
    }

    /// Shrink on every side.
    pub fn inset(self, amount: f32) -> Self {
        self.expand(-amount)
    }

    pub fn translate(self, dx: f32, dy: f32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            ..self
        }
    }

    /// Shrink horizontally by `dx` on each side and vertically by `dy` on each side.
    pub fn inset_xy(self, dx: f32, dy: f32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            w: (self.w - dx * 2.0).max(0.0),
            h: (self.h - dy * 2.0).max(0.0),
        }
    }

    /// Inset by explicit CSS-order `(top, right, bottom, left)` padding.
    pub fn pad(self, top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self {
            x: self.x + left,
            y: self.y + top,
            w: (self.w - left - right).max(0.0),
            h: (self.h - top - bottom).max(0.0),
        }
    }

    /// Split off a left strip of `width` pixels, returning `(left, remainder)`.
    pub fn split_left(self, width: f32) -> (Self, Self) {
        let w = width.clamp(0.0, self.w.max(0.0));
        (
            Self::new(self.x, self.y, w, self.h),
            Self::new(self.x + w, self.y, (self.w - w).max(0.0), self.h),
        )
    }

    /// Split off a right strip of `width` pixels, returning `(remainder, right)`.
    pub fn split_right(self, width: f32) -> (Self, Self) {
        let w = width.clamp(0.0, self.w.max(0.0));
        let rem = (self.w - w).max(0.0);
        (
            Self::new(self.x, self.y, rem, self.h),
            Self::new(self.x + rem, self.y, w, self.h),
        )
    }

    /// Split off a top strip of `height` pixels, returning `(top, remainder)`.
    pub fn split_top(self, height: f32) -> (Self, Self) {
        let h = height.clamp(0.0, self.h.max(0.0));
        (
            Self::new(self.x, self.y, self.w, h),
            Self::new(self.x, self.y + h, self.w, (self.h - h).max(0.0)),
        )
    }

    /// Split off a bottom strip of `height` pixels, returning `(remainder, bottom)`.
    pub fn split_bottom(self, height: f32) -> (Self, Self) {
        let h = height.clamp(0.0, self.h.max(0.0));
        let rem = (self.h - h).max(0.0);
        (
            Self::new(self.x, self.y, self.w, rem),
            Self::new(self.x, self.y + rem, self.w, h),
        )
    }

    pub const fn with_x(self, x: f32) -> Self {
        Self { x, ..self }
    }

    pub const fn with_y(self, y: f32) -> Self {
        Self { y, ..self }
    }

    pub const fn with_w(self, w: f32) -> Self {
        Self { w, ..self }
    }

    pub const fn with_h(self, h: f32) -> Self {
        Self { h, ..self }
    }

    /// Centre a `size`-sized rect inside `self` on both axes.
    pub fn center_in(self, size: (f32, f32)) -> Self {
        let c = self.center();
        Rect::new(c.0 - size.0 * 0.5, c.1 - size.1 * 0.5, size.0, size.1)
    }

    /// Align a `size`-sized rect inside `self` per axis.
    pub fn align_in(self, size: (f32, f32), ax: layout::AlignX, ay: layout::AlignY) -> Self {
        let x = match ax {
            layout::AlignX::Start => self.x,
            layout::AlignX::Center => self.center().0 - size.0 * 0.5,
            layout::AlignX::End => self.max().0 - size.0,
        };
        let y = match ay {
            layout::AlignY::Start => self.y,
            layout::AlignY::Center => self.center().1 - size.1 * 0.5,
            layout::AlignY::End => self.max().1 - size.1,
        };
        Rect::new(x, y, size.0, size.1)
    }

    /// The largest rect of aspect `ratio` (w/h) centred inside `self`.
    pub fn fit_aspect(self, ratio: f32) -> Self {
        let ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio
        } else {
            1.0
        };
        let (mut w, mut h) = (self.w, self.w / ratio);
        if h > self.h {
            h = self.h;
            w = h * ratio;
        }
        self.center_in((w, h))
    }

    /// Snap all four edges to the physical grid at `scale`.
    pub fn snap(self, scale: f32) -> Self {
        layout::snap_rect(self, scale)
    }

    /// Intersection of two rects, or `None` when they do not overlap.
    pub fn intersect(self, other: Self) -> Option<Self> {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = (self.x + self.w).min(other.x + other.w);
        let y1 = (self.y + self.h).min(other.y + other.h);
        if x1 > x0 && y1 > y0 {
            Some(Self::from_min_max(x0, y0, x1, y1))
        } else {
            None
        }
    }

    /// Union of two rects.
    pub fn union(self, other: Self) -> Self {
        if self.w <= 0.0 || self.h <= 0.0 {
            return other;
        }
        if other.w <= 0.0 || other.h <= 0.0 {
            return self;
        }
        Self::from_min_max(
            self.x.min(other.x),
            self.y.min(other.y),
            (self.x + self.w).max(other.x + other.w),
            (self.y + self.h).max(other.y + other.h),
        )
    }
}

impl tween::LerpValue for Rect {
    fn lerp_value(a: Self, b: Self, t: f32) -> Self {
        let mix = |x: f32, y: f32| x + (y - x) * t;
        Self {
            x: mix(a.x, b.x),
            y: mix(a.y, b.y),
            w: mix(a.w, b.w),
            h: mix(a.h, b.h),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_ops() {
        let r = Rect::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(r.max(), (40.0, 60.0));
        assert_eq!(r.center(), (25.0, 40.0));
        assert!(r.contains((10.0, 20.0)));
        assert!(r.contains((39.9, 59.9)));
        assert!(!r.contains((40.0, 60.0)));
        assert_eq!(r.inset(5.0), Rect::new(15.0, 25.0, 20.0, 30.0));
        assert_eq!(r.translate(1.0, 2.0), Rect::new(11.0, 22.0, 30.0, 40.0));
    }

    #[test]
    fn rect_intersect_union() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(a.intersect(b), Some(Rect::new(5.0, 5.0, 5.0, 5.0)));
        assert_eq!(a.intersect(Rect::new(20.0, 20.0, 1.0, 1.0)), None);
        assert_eq!(a.union(b), Rect::new(0.0, 0.0, 15.0, 15.0));
    }

    #[test]
    fn color_fade_and_pack() {
        let c = Color::rgba(10, 20, 30, 100);
        assert_eq!(c.fade(0.5).a, 50);
        assert_eq!(c.with_alpha(7).to_array(), [10, 20, 30, 7]);
        assert_eq!(Color::WHITE.fade(2.0).a, 255);
    }

    /// A fade out of nothing must arrive at its colour, never dip below the
    /// surface it is fading in over. Mixing straight channels put half the
    /// target colour at half alpha, which the `src_alpha` blend then darkened
    /// a second time — the hover cards, section headers and menu rows visibly
    /// went *darker than the panel* on the way in before correcting.
    #[test]
    fn color_fade_through_transparency_never_dips() {
        let panel = Color::rgb(0x1e, 0x1e, 0x1e);
        let target = Color::rgb(0x33, 0x33, 0x33);
        // What the GPU outputs for `mixed` drawn over `panel`.
        let composite = |mixed: Color| {
            let a = mixed.a as f32 / 255.0;
            (mixed.r as f32 * a + panel.r as f32 * (1.0 - a)).round() as u8
        };
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            let mixed = <Color as tween::LerpValue>::lerp_value(Color::TRANSPARENT, target, t);
            let lit = composite(mixed);
            assert!(
                lit >= panel.r,
                "fade dips below the panel at t={t}: {lit:#04x} < {:#04x}",
                panel.r
            );
            assert!(
                lit <= target.r,
                "fade overshoots its target at t={t}: {lit:#04x} > {:#04x}",
                target.r
            );
        }
        // Hue is constant along the way: only the coverage ramps.
        let half = <Color as tween::LerpValue>::lerp_value(Color::TRANSPARENT, target, 0.5);
        assert_eq!(
            (half.r, half.g, half.b),
            (target.r, target.g, target.b),
            "half-way colour is the target, not a darkened version of it"
        );
        assert_eq!(half.a, 128, "and its coverage is half");
    }
}
