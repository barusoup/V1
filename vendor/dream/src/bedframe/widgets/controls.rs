use super::{
    ButtonStyle, ClipKind, ClipScope, LabelStyle, PanelStyle, Pulse, Response, SliderStyle, Ui,
    display_label, mix_color,
};
use super::{headless, paint};
use crate::ui::font::{Align, Font, FontId, LayoutOptions, TextLayout};
use crate::ui::tokens::Radius;
use crate::ui::tween::Ease;
use crate::ui::{Color, Rect};

/// Where a label's layout is drawn inside `rect`, decided by the style's
/// vertical rule (`vcenter` + `vcenter_mode`). Returns the draw origin and
/// the first baseline (both logical): the baseline is snapped to a whole
/// *physical* pixel row, and the origin hangs the layout off that baseline
/// (`origin.y = baseline − first_baseline`) so the text never rides a half
/// pixel vertically. Horizontal origin rounding is unchanged.
pub(crate) fn label_origin(
    face: Option<&Font>,
    rect: Rect,
    layout: &TextLayout,
    size: f32,
    scale: f32,
    style: &super::LabelStyle,
) -> ((f32, f32), f32) {
    let baseline = if !style.vcenter {
        rect.y + layout.first_baseline / scale
    } else {
        match style.vcenter_mode {
            super::VCenter::Baseline(y) => rect.y + y,
            super::VCenter::Box => {
                let block_h = if let Some(face) = face {
                    let px = size * scale;
                    let asc = face.ascent(px);
                    let desc = face.descent(px);
                    let lines = layout.line_count.max(1);
                    let extra = (lines - 1) as f32 * face.line_height(px) * style.line_height;
                    (asc - desc + extra) / scale
                } else {
                    layout.height / scale
                };
                rect.center().1 - block_h * 0.5 + layout.first_baseline / scale
            }
            super::VCenter::Cap | super::VCenter::XHeight => {
                // The optical span runs from the first line's cap (or x)
                // top to the last line's baseline — centre that span and
                // the last baseline is its bottom edge.
                let lines = layout.line_count.max(1);
                let span = if let Some(face) = face {
                    let px = size * scale;
                    let h = if style.vcenter_mode == super::VCenter::XHeight {
                        face.x_height(px)
                    } else {
                        face.cap_height(px)
                    };
                    (h + (lines - 1) as f32 * layout.line_advance) / scale
                } else {
                    layout.height / scale
                };
                rect.center().1 + span * 0.5 - (lines - 1) as f32 * layout.line_advance / scale
            }
        }
    };
    let baseline = (baseline * scale).round() / scale;
    let origin_x = match style.align {
        Align::Left => rect.x,
        Align::Center => rect.center().0,
        Align::Right => rect.max().0,
    };
    (
        (
            (origin_x * scale).round() / scale,
            baseline - layout.first_baseline / scale,
        ),
        baseline,
    )
}

/// Snap `byte` to the nearest UTF-8 char boundary at or below it.
fn clamp_boundary(text: &str, byte: usize) -> usize {
    let mut byte = byte.min(text.len());
    while byte > 0 && !text.is_char_boundary(byte) {
        byte -= 1;
    }
    byte
}

impl Ui {
    /// Static text at `rect`'s top-left (respecting the style's alignment:
    /// center/right align inside `rect`).
    ///
    /// Glyphs rasterize at the *physical* font size (`size * scale`) so
    /// text is true-resolution at every scale, then the layout is
    /// normalized back to logical units — the draw list's finish pass
    /// multiplies it back out.
    pub fn label(&mut self, rect: Rect, text: &str, style: LabelStyle) {
        // A label wholly clipped or offscreen still costs a layout pass —
        // the clip/screen test here is cheaper than kern+wrap+bitmap.
        if rect.w <= 0.0 || rect.h <= 0.0 {
            return;
        }
        let screen = Rect::new(0.0, 0.0, self.screen.0, self.screen.1);
        if screen.intersect(rect).is_none()
            || self
                .draw
                .clip_rect()
                .is_some_and(|clip| clip.intersect(rect).is_none())
        {
            return;
        }
        let scale = self.scale;
        let options = LayoutOptions {
            align: style.align,
            line_height: style.line_height,
            ..Default::default()
        };
        let layout = self.cached_layout(text, style.font, style.size * scale, options);
        self.draw_label(rect, &layout, scale, style);
    }

    /// The shared tail of [`Self::label`]: position the physical layout
    /// inside `rect` by the style and draw it scaled back to logical.
    fn draw_label(
        &mut self,
        rect: Rect,
        layout: &std::sync::Arc<super::super::font::TextLayout>,
        scale: f32,
        style: LabelStyle,
    ) {
        let face = self.fonts.get_opt(style.font);
        let (origin, _baseline) = label_origin(face, rect, layout, style.size, scale, &style);
        self.draw
            .text_layout_scaled(origin, layout, scale, style.color, &self.cache);
    }

    /// The size `text` occupies when wrapped to `width` logical pixels —
    /// the height-for-width companion to [`Self::measure`]. Shares the
    /// layout cache with [`Self::label_wrapped`], so measuring a paragraph
    /// and then drawing it costs one layout.
    ///
    /// `line_height` is a multiplier on the face's natural line height, the
    /// same field [`LabelStyle`] carries.
    pub fn measure_wrapped(
        &mut self,
        text: &str,
        font: FontId,
        size: f32,
        width: f32,
        line_height: f32,
    ) -> (f32, f32) {
        let scale = self.scale;
        let options = LayoutOptions {
            line_height,
            wrap_width: Some((width * scale).max(1.0)),
            ..Default::default()
        };
        let layout = self.cached_layout(text, font, size * scale, options);
        (layout.width / scale, layout.height / scale)
    }

    /// The baseline-to-baseline distance `measure_wrapped` lays out with.
    /// Editors use it to turn a caret's block height into a line index.
    pub fn wrapped_line_advance(&mut self, font: FontId, size: f32, line_height: f32) -> f32 {
        let scale = self.scale;
        let options = LayoutOptions {
            line_height,
            ..Default::default()
        };
        let layout = self.cached_layout("", font, size * scale, options);
        (layout.line_advance / scale).max(1.0)
    }

    /// Offset (logical px, from the text's top) of the caret at byte `byte`
    /// when `text` is wrapped to `width` with `line_height`. Returns
    /// `(x, top, line_index)`.
    ///
    /// The prefix laid out to the caret ends exactly where the caret sits;
    /// its line count names the caret's line and its final stop names the
    /// x. The prefix wraps identically to the full text because wrapping is
    /// greedy left-to-right, so no full-document pass is needed.
    pub fn wrapped_caret(
        &mut self,
        text: &str,
        byte: usize,
        font: FontId,
        size: f32,
        width: f32,
        line_height: f32,
    ) -> (f32, f32, usize) {
        let byte = clamp_boundary(text, byte);
        let scale = self.scale;
        let advance = self.wrapped_line_advance(font, size, line_height);
        let prefix = &text[..byte];
        let options = LayoutOptions {
            line_height,
            wrap_width: Some((width * scale).max(1.0)),
            ..Default::default()
        };
        let prefix_layout = self.cached_layout(prefix, font, size * scale, options);
        let trailing_break = prefix.ends_with('\n');
        let line = if trailing_break {
            prefix_layout.line_count
        } else {
            prefix_layout.line_count.saturating_sub(1)
        };
        let x = if trailing_break {
            0.0
        } else {
            prefix_layout
                .stops
                .last()
                .map(|(x, _)| x / scale)
                .unwrap_or(0.0)
        };
        (x, line as f32 * advance, line)
    }

    /// Byte offset nearest to the point `(x, y)` inside text wrapped to
    /// `width` — click-to-place for a wrapped editor.
    #[allow(clippy::too_many_arguments)] // Keep the established rendering API.
    pub fn wrapped_byte_at(
        &mut self,
        text: &str,
        font: FontId,
        size: f32,
        width: f32,
        line_height: f32,
        x: f32,
        y: f32,
    ) -> usize {
        if text.is_empty() {
            return 0;
        }
        let advance = self.wrapped_line_advance(font, size, line_height);
        let last_line = self
            .wrapped_caret(text, text.len(), font, size, width, line_height)
            .2;
        let line = ((y / advance).floor() as usize).min(last_line);

        // Binary-search the first byte on `line`, then walk that line's
        // boundaries taking the nearest stop. Caret lines are monotonic in
        // byte, and the layouts are cached across the walk.
        let boundaries: Vec<usize> = std::iter::once(0)
            .chain(text.char_indices().map(|(index, ch)| index + ch.len_utf8()))
            .collect();
        let mut lo = 0usize;
        let mut hi = boundaries.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let mid_line = self
                .wrapped_caret(text, boundaries[mid], font, size, width, line_height)
                .2;
            if mid_line < line {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut best = (f32::INFINITY, boundaries[lo]);
        for &byte in &boundaries[lo..] {
            let (cx, _, cx_line) = self.wrapped_caret(text, byte, font, size, width, line_height);
            if cx_line != line {
                break;
            }
            let dx = (cx - x).abs();
            if dx < best.0 {
                best = (dx, byte);
            }
        }
        best.1
    }

    /// `label` that wraps at `rect.w`: same cached-layout pipeline, the
    /// wrap width simply joins the cache key. `rect.h` grows with the
    /// wrapped height only in the caller's layout — drawn text still tops
    /// at `rect.y` (or centres when `vcenter`).
    pub fn label_wrapped(&mut self, rect: Rect, text: &str, style: LabelStyle) {
        if rect.w <= 0.0 || rect.h <= 0.0 {
            return;
        }
        let screen = Rect::new(0.0, 0.0, self.screen.0, self.screen.1);
        if screen.intersect(rect).is_none()
            || self
                .draw
                .clip_rect()
                .is_some_and(|clip| clip.intersect(rect).is_none())
        {
            return;
        }
        let scale = self.scale;
        let options = LayoutOptions {
            align: style.align,
            line_height: style.line_height,
            wrap_width: Some(rect.w * scale),
            ..Default::default()
        };
        let layout = self.cached_layout(text, style.font, style.size * scale, options);
        self.draw_label(rect, &layout, scale, style);
    }

    /// Like [`Self::label_wrapped`], but lays out `text` at `style.size` (wrapping to `rect.w`)
    /// and draws the resulting layout scaled by `zoom` centred in `rect`. Animating `zoom`
    /// keeps glyph shapes, line breaks, and relative positions locked without re-rasterizing.
    pub fn label_wrapped_scaled(&mut self, rect: Rect, text: &str, style: LabelStyle, zoom: f32) {
        if rect.w <= 0.0 || rect.h <= 0.0 || zoom <= 0.0 {
            return;
        }
        let screen = Rect::new(0.0, 0.0, self.screen.0, self.screen.1);
        if screen.intersect(rect).is_none()
            || self
                .draw
                .clip_rect()
                .is_some_and(|clip| clip.intersect(rect).is_none())
        {
            return;
        }
        let scale = self.scale;
        let options = LayoutOptions {
            align: style.align,
            line_height: style.line_height,
            wrap_width: Some(rect.w * scale),
            ..Default::default()
        };
        let layout = self.cached_layout(text, style.font, style.size * scale, options);
        let origin_x = match style.align {
            Align::Left => rect.x,
            Align::Center => rect.center().0,
            Align::Right => rect.max().0,
        };
        let scaled_h = (layout.height / scale) * zoom;
        let origin_y = rect.center().1 - scaled_h * 0.5;
        self.draw.text_layout_scaled(
            (origin_x, origin_y),
            &layout,
            scale / zoom,
            style.color,
            &self.cache,
        );
    }

    /// `label` truncated to one line with an ellipsis: the longest whole
    /// graphemes of `text` that, followed by "…", still fit `rect.w`.
    /// Truncates at extended grapheme boundaries — never inside a ZWJ
    /// emoji, a combining mark, or a surrogate pair.
    pub fn label_ellipsis(&mut self, rect: Rect, text: &str, style: LabelStyle) {
        let scale = self.scale;
        let width = rect.w;
        if width <= 0.0 || rect.h <= 0.0 {
            return;
        }
        // Fast path: the whole string fits as-is.
        let mut fits = |candidate: &str| {
            self.cached_layout(
                candidate,
                style.font,
                style.size * scale,
                LayoutOptions {
                    align: style.align,
                    line_height: style.line_height,
                    ..Default::default()
                },
            )
            .width
                <= width * scale
        };
        if fits(text) {
            self.label(rect, text, style);
            return;
        }
        const ELLIPSIS: &str = "…";
        if !fits(ELLIPSIS) {
            return; // narrower than an ellipsis: draw nothing.
        }
        // Binary-search the longest fitting grapheme prefix. Ends are
        // extended-grapheme boundaries, so a cut never splits a cluster.
        let ends: Vec<usize> =
            unicode_segmentation::UnicodeSegmentation::grapheme_indices(text, true)
                .map(|(start, cluster)| start + cluster.len())
                .collect();
        let mut lo = 0usize;
        let mut hi = ends.len();
        let mut candidate = String::new();
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            candidate.clear();
            candidate.push_str(&text[..ends[mid - 1]]);
            candidate.push_str(ELLIPSIS);
            if fits(&candidate) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        candidate.clear();
        if lo > 0 {
            candidate.push_str(&text[..ends[lo - 1]]);
        }
        candidate.push_str(ELLIPSIS);
        self.label(rect, &candidate, style);
    }

    /// Background panel with optional border; pushes a clip rect that
    /// [`Self::end_clip`] pops.
    pub fn panel(&mut self, rect: Rect, style: PanelStyle) {
        match style.border {
            Some((thickness, color)) => {
                // A stock 1px border is a hairline: one physical pixel.
                let thickness = if thickness == 1.0 {
                    crate::ui::layout::hairline(self.scale)
                } else {
                    thickness
                };
                self.draw.rounded_frame(
                    rect,
                    style.corner_radius,
                    thickness,
                    style.background,
                    color,
                );
            }
            None => self
                .draw
                .round_rect(rect, style.corner_radius, style.background),
        }
        self.draw
            .push_clip(rect.inset(crate::ui::layout::hairline(self.scale)));
        self.push_clip_scope(ClipScope::Panel);
    }

    /// Pop the innermost panel/modal clip scope opened by [`Self::panel`],
    /// [`Self::pixel_panel`] or [`Self::modal`]. A top of another kind (a
    /// section's reveal clip, a scroll view) is left alone — it is not
    /// ours to pop.
    pub fn end_clip(&mut self) {
        self.end_clip_scope(&[ClipKind::Panel, ClipKind::Overlay]);
    }

    // --- Widgets ------------------------------------------------------------

    /// Push button with animated hover. See [`ButtonStyle::disabled`].
    pub fn button(&mut self, label: &str, rect: Rect, style: ButtonStyle) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ButtonBehavior::new(label)
            .enabled(!style.disabled)
            .update(self, rect);
        paint::button(self, rect, &style, state, label);
        self.focus_ring(rect, style.corner_radius, id);
        resp
    }

    /// Button with caller-drawn content: the painter draws the chrome,
    /// `content` draws inside `rect` reading the live [`State`](headless::State).
    pub fn button_with(
        &mut self,
        label: &str,
        rect: Rect,
        style: &ButtonStyle,
        content: impl FnOnce(&mut Ui, Rect, headless::State),
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ButtonBehavior::new(label)
            .enabled(!style.disabled)
            .update(self, rect);
        // The chrome without its label: same bg + border as paint::button.
        let hover = state.hover;
        let bg = mix_color(
            mix_color(style.normal, style.hover, hover),
            style.pressed,
            if state.pressed { 1.0 } else { 0.0 },
        );
        self.draw.round_rect(rect, style.corner_radius, bg);
        if style.normal.a > 0 && style.corner_radius > 0.0 && style.normal != style.hover {
            let border = mix_color(style.normal, style.hover, 0.55 + hover * 0.45);
            self.draw.round_ring(
                rect,
                style.corner_radius,
                crate::ui::layout::hairline(self.scale),
                border,
            );
        }
        content(self, rect, state);
        self.focus_ring(rect, style.corner_radius, id);
        resp
    }

    /// Checkable toggle with a label to its right. Flips `value` on click.
    pub fn toggle(
        &mut self,
        label: &str,
        rect: Rect,
        value: &mut bool,
        style: ButtonStyle,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ToggleBehavior::new(label, value)
            .enabled(!style.disabled)
            .duration(self.theme.tokens.motion.toggle)
            .anim_suffix("##switch")
            .update(self, rect);
        paint::toggle(self, rect, &style, state, label);
        self.focus_ring(rect, rect.h * 0.5, id);
        resp
    }

    /// A bare switch with no label: the pill track and knob only, flipping
    /// `value` on click. Use [`Self::toggle`] when there is a label to show —
    /// `toggle`'s label doubles as its drawn text.
    pub fn switch(
        &mut self,
        label: &str,
        rect: Rect,
        value: &mut bool,
        style: ButtonStyle,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ToggleBehavior::new(label, value)
            .enabled(!style.disabled)
            .duration(self.theme.tokens.motion.toggle)
            .anim_suffix("##switch")
            .update(self, rect);
        paint::switch(self, rect, &style, state);
        self.focus_ring(rect, rect.h * 0.5, id);
        resp
    }

    /// Horizontal slider over `range`. Drag with the pointer; `nav.left`/
    /// `nav.right` nudge it in tenths while focused.
    pub fn slider(
        &mut self,
        label: &str,
        rect: Rect,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        style: SliderStyle,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::SliderBehavior::new(label, value, range).update(self, rect);
        paint::slider(self, rect, &style, state, label);
        self.focus_ring(rect, rect.h * 0.5, id);
        resp
    }

    /// Read-only progress bar over `value` in 0..1.
    pub fn progress(&mut self, rect: Rect, value: f32, fill: Color) {
        let theme = self.theme;
        let value = value.clamp(0.0, 1.0);
        let radius = rect.h * 0.5;
        self.draw.round_rect(rect, radius, theme.g2);
        if value > 0.0 {
            self.draw.round_rect(
                Rect::new(rect.x, rect.y, rect.w * value, rect.h),
                radius,
                fill,
            );
        }
    }

    // --- Icons -------------------------------------------------------------

    /// Register icons with the UI. Icons bake lazily on demand at the exact
    /// physical pixel size when drawn (`Ui::icon`, `Ui::icon_region`), so
    /// startup pays zero upfront rasterization or atlas space for unused sizes.
    pub fn register_vector_icons(&mut self, icons: &[crate::ui::icons::Icon]) {
        let _ = icons;
    }

    /// Atlas region for an icon at `size` pixels, baking on demand.
    pub fn icon_region(
        &mut self,
        icon: crate::ui::icons::Icon,
        size: u32,
    ) -> Option<crate::ui::atlas::Region> {
        self.icons.region(icon, size, &mut self.cache)
    }

    /// Draw `icon` centred in `rect` at `size` logical pixels, tinted
    /// `tint`. The raster is baked at `round(size * scale)` physical pixels
    /// so the icon is true-resolution at every scale.
    pub fn icon(&mut self, rect: Rect, icon: crate::ui::icons::Icon, size: u32, tint: Color) {
        self.icon_fit(rect, icon, size, tint, super::IconFit::Ink);
    }

    /// `icon` at `size` logical pixels under `fit`. [`super::IconFit::Grid`]
    /// centres the whole raster square; [`super::IconFit::Ink`] (the default via
    /// [`Self::icon`]) centres and normalises the ink — the target ink
    /// major extent is `size · (20/24) · optical_k` logical pixels, so a
    /// well-behaved Lucide icon keeps roughly its old size while outliers
    /// stop reading oversized.
    pub fn icon_fit(
        &mut self,
        rect: Rect,
        icon: crate::ui::icons::Icon,
        size: u32,
        tint: Color,
        fit: super::IconFit,
    ) {
        let scale = self.scale;
        let center = rect.center();
        match fit {
            super::IconFit::Grid => {
                let raster = ((size as f32) * scale).round().max(1.0) as u32;
                let Some(region) = self.icons.region(icon, raster, &mut self.cache) else {
                    return;
                };
                let (w, h) = (region.width as f32 / scale, region.height as f32 / scale);
                // Every Lucide icon is authored on the whole 24x24 grid, so
                // centring the rasterized square in the control is exact
                // optical alignment — and snapping the quad to whole
                // *physical* pixels keeps the ink crisp.
                let target = Rect::new(
                    ((center.0 - w * 0.5) * scale).round() / scale,
                    ((center.1 - h * 0.5) * scale).round() / scale,
                    w,
                    h,
                );
                self.draw.atlas_region(target, &self.cache, region, tint);
            }
            super::IconFit::Ink => {
                let ink = self.icons.ink_bounds(icon);
                let quad = self.ink_icon_quad(rect, size, ink, icon.optical_k(), scale);
                let Some((quad, raster)) = quad else { return };
                let Some(region) = self.icons.region(icon, raster, &mut self.cache) else {
                    return;
                };
                self.draw.atlas_region(quad, &self.cache, region, tint);
            }
        }
    }

    /// The shared ink-fit placement: raster side `S = round(T·scale · 24 /
    /// ink_major)` physical px (T = `size · 20/24 · k` logical), and a
    /// quad whose origin puts the ink centre on the rect centre, snapped
    /// to whole physical pixels. `ink_fn` resolves the ink box lazily —
    /// custom icons flatten their paths, stock icons read the cache.
    pub(crate) fn ink_icon_quad(
        &mut self,
        rect: Rect,
        size: u32,
        ink: crate::ui::icons::InkBox,
        k: f32,
        scale: f32,
    ) -> Option<(Rect, u32)> {
        let center = rect.center();
        let target = size as f32 * (20.0 / 24.0) * k;
        // Lucide is already optically balanced on its 24 grid, so the
        // ink-major normalisation is clamped: it only evens out outliers
        // (like the caption glyphs), never resizes a well-proportioned
        // icon more than ~8% smaller or ~10% larger.
        let s_grid = (size as f32 * scale).round().max(1.0);
        let s_ink = (target * scale * 24.0 / ink.major()).round().max(1.0);
        let raster = (s_grid * (s_ink / s_grid).clamp(0.92, 1.10))
            .round()
            .max(1.0) as u32;
        let (icx, icy) = ink.center();
        let icx = icx / 24.0 * raster as f32;
        let icy = icy / 24.0 * raster as f32;
        let side = raster as f32 / scale;
        let quad = Rect::new(
            (center.0 * scale - icx).round() / scale,
            (center.1 * scale - icy).round() / scale,
            side,
            side,
        );
        Some((quad, raster))
    }

    // --- Panels ------------------------------------------------------------

    /// A rounded panel fill with a hairline border. Pushes a clip rect; pop
    /// with [`Self::end_clip`]. (`shadow`/`notch` are accepted for source
    /// compatibility and ignored in the modern flat style.)
    pub fn pixel_panel(
        &mut self,
        rect: Rect,
        fill: Color,
        border: Color,
        shadow: Color,
        notch: f32,
    ) {
        let rect = crate::ui::theme::snap(rect);
        let _ = (shadow, notch);
        let hairline = crate::ui::layout::hairline(self.scale);
        self.draw.rounded_frame(rect, 8.0, hairline, fill, border);
        self.draw.push_clip(rect.inset(hairline));
        self.push_clip_scope(ClipScope::Panel);
    }

    /// Square icon button: a rounded surface tinted by hover, inverted when
    /// `on`. The icon is drawn at `icon_size` pixels and its tint changes by
    /// state — never its weight or geometry. `badge` prints a small key hint
    /// at the bottom-right corner.
    // One more parameter than clippy likes: label/id, rect, icon, style,
    // latched state, badge and icon size.
    #[allow(clippy::too_many_arguments)]
    pub fn icon_button(
        &mut self,
        label: &str,
        rect: Rect,
        icon: crate::ui::icons::Icon,
        style: ButtonStyle,
        on: bool,
        badge: Option<&str>,
        icon_size: u32,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ButtonBehavior::new(label)
            .enabled(!style.disabled)
            .update(self, rect);
        paint::icon_button(self, rect, &style, state, icon, icon_size, on, badge);
        self.focus_ring(rect, style.corner_radius, id);
        resp
    }

    /// Latching icon button (tool/toggle rail). Flips `*value` and returns
    /// the same click that flipped it.
    #[allow(clippy::too_many_arguments)]
    pub fn icon_toggle(
        &mut self,
        label: &str,
        rect: Rect,
        icon: crate::ui::icons::Icon,
        value: &mut bool,
        style: ButtonStyle,
        badge: Option<&str>,
        icon_size: u32,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ButtonBehavior::new(label)
            .enabled(!style.disabled)
            .update(self, rect);
        if resp.clicked {
            *value = !*value;
            self.pulses.insert(id, Pulse::new(0.2, self.frame));
        }
        let _pop = self.pulse(id);
        paint::icon_button(self, rect, &style, state, icon, icon_size, *value, badge);
        self.focus_ring(rect, style.corner_radius, id);
        resp
    }

    /// Text button: fixed size from the caller, rounded and hover-tinted.
    /// The label doubles as the button text.
    pub fn text_button(&mut self, label: &str, rect: Rect, style: ButtonStyle) -> Response {
        self.button(label, rect, style)
    }

    /// Elevated card surface: draws a soft ambient drop shadow under `rect`
    /// and then opens a [`Self::panel`] clip scope (pop with [`Self::end_clip`]).
    pub fn card(&mut self, rect: Rect, style: PanelStyle) {
        self.elevation_shadow(
            rect,
            style.corner_radius,
            crate::ui::tokens::Elevation::Raised,
        );
        self.panel(rect, style);
    }

    /// Hairline separator rule in the active theme's border colour.
    pub fn separator(&mut self, rect: Rect) {
        let color = self.theme.border;
        self.draw.rect(rect, color);
    }

    /// Hairline divider rule in an explicit colour.
    pub fn divider(&mut self, rect: Rect, color: Color) {
        self.draw.rect(rect, color);
    }

    /// Pill-shaped status/tag badge with a subtle hairline border and
    /// optically centred label.
    pub fn badge(
        &mut self,
        rect: Rect,
        text: &str,
        bg: Color,
        fg: Color,
        font: crate::ui::font::FontId,
        size: f32,
    ) {
        let radius = rect.h * 0.5;
        let border = mix_color(bg, fg, 0.22);
        self.draw.rounded_frame(
            rect,
            radius,
            crate::ui::layout::hairline(self.scale),
            bg,
            border,
        );
        self.label(
            rect,
            text,
            LabelStyle::new(font)
                .size(size)
                .color(fg)
                .align(Align::Center)
                .vcenter(true),
        );
    }

    /// Interactive checkbox with an animated check fill, `Icon::Check` mark
    /// and a right-hand label.
    pub fn checkbox(
        &mut self,
        label: &str,
        rect: Rect,
        checked: &mut bool,
        style: ButtonStyle,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ToggleBehavior::new(label, checked)
            .enabled(!style.disabled)
            .duration(self.theme.tokens.motion.check)
            .anim_suffix("##chk")
            .update(self, rect);
        paint::checkbox(self, rect, &style, state, label);
        self.focus_ring(rect, self.theme.tokens.radius(Radius::Xs), id);
        resp
    }

    /// Interactive radio button with an animated inner dot and right-hand label.
    pub fn radio(
        &mut self,
        label: &str,
        rect: Rect,
        selected: bool,
        style: ButtonStyle,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::RadioBehavior::new(label, selected)
            .enabled(!style.disabled)
            .duration(self.theme.tokens.motion.check)
            .anim_suffix("##rad")
            .update(self, rect);
        paint::radio(self, rect, &style, state, label);
        let control = self
            .theme
            .tokens
            .control(crate::ui::tokens::ControlSize::for_height(rect.h));
        let d = (control.icon + 2.0).min(rect.h);
        self.focus_ring(rect, d * 0.5, id);
        resp
    }

    /// Gliding-pill segmented selector / tab bar over `options`. Updates
    /// `*selected` and returns `Some(index)` when the user picks a segment.
    pub fn segmented(
        &mut self,
        label: &str,
        rect: Rect,
        options: &[&str],
        selected: &mut usize,
        style: ButtonStyle,
    ) -> Option<usize> {
        // Legacy entry: a SegmentedStyle derived from the theme + the
        // caller's ButtonStyle, identical to the old baked-in look.
        let theme = self.theme;
        let seg = super::SegmentedStyle {
            button: style,
            track: theme.g1,
            track_border: Some((crate::ui::layout::hairline(self.scale), theme.border)),
            pill: style.normal,
            inset: crate::ui::tokens::Space::S0_5,
            text: style.text_disabled,
            text_selected: style.text,
            glide: self.theme.tokens.motion.segmented,
            squash: 0.06,
        };
        self.segmented_styled(label, rect, options, selected, &seg)
    }

    /// Segmented selector with a full [`SegmentedStyle`](super::SegmentedStyle): track, pill and
    /// text colours come from the style, geometry identical to
    /// [`Self::segmented`].
    pub fn segmented_styled(
        &mut self,
        label: &str,
        rect: Rect,
        options: &[&str],
        selected: &mut usize,
        style: &super::SegmentedStyle,
    ) -> Option<usize> {
        let mut chosen = None;
        self.segmented_with(label, rect, options, selected, style, |ui, item| {
            let st = super::ButtonStyle {
                normal: Color::TRANSPARENT,
                hover: Color::TRANSPARENT,
                pressed: Color::TRANSPARENT,
                text: mix_color(style.text, style.text_selected, item.near),
                text_hover: style.button.text_hover,
                ..style.button
            };
            // Item label, same as the old inner buttons drew.
            let text = display_label(item.label);
            ui.label(
                item.rect,
                text,
                LabelStyle::new(st.font)
                    .size(st.size)
                    .color(st.text)
                    .align(Align::Center)
                    .vcenter(true),
            );
            if item.clicked {
                chosen = Some(item.index);
            }
        });
        chosen
    }

    /// Segmented selector with per-item content: track + pill come from
    /// the painter; `item` draws each segment's content and sees its
    /// interaction state. Returns `Some(index)` when the user picks a
    /// segment.
    pub fn segmented_with(
        &mut self,
        label: &str,
        rect: Rect,
        options: &[&str],
        selected: &mut usize,
        style: &super::SegmentedStyle,
        mut item: impl FnMut(&mut Ui, headless::SegmentItem<'_>),
    ) -> Option<usize> {
        if options.is_empty() {
            return None;
        }
        *selected = (*selected).min(options.len() - 1);
        self.push_id(label);
        let mut cells = std::mem::take(&mut self.seg_cells);
        crate::ui::layout::divide_into(&mut cells, rect.x, rect.w, options.len(), 0.0, self.scale);
        let at = self.anim_f32_seeded("##pill", *selected as f32, style.glide, Ease::CubicOut);
        paint::segmented(self, rect, style, at, &cells);
        let inset = style.inset;
        let mut chosen = None;
        let mut items = std::mem::take(&mut self.seg_scratch);
        headless::SegmentedBehavior::new(label, options)
            .enabled(!style.button.disabled)
            .update_with_cells(self, rect, selected, &cells, inset, &mut items);
        for (i, &(cell, ref resp, state)) in items.iter().enumerate() {
            let near = (1.0 - (at - i as f32).abs()).clamp(0.0, 1.0);
            if resp.clicked {
                *selected = i;
                chosen = Some(i);
            }
            item(
                self,
                headless::SegmentItem {
                    index: i,
                    label: options[i],
                    rect: cell,
                    state,
                    clicked: resp.clicked,
                    near,
                },
            );
            self.focus_ring(
                cell,
                style.button.corner_radius,
                headless::seg_id(self, options[i], i),
            );
        }
        self.pop_id();
        self.seg_cells = cells;
        self.seg_scratch = items;
        chosen
    }

    /// Push button with a leading vector icon and label centred together.
    pub fn button_with_icon(
        &mut self,
        label: &str,
        rect: Rect,
        icon: crate::ui::icons::Icon,
        icon_size: u32,
        style: ButtonStyle,
    ) -> Response {
        let id = self.id(label);
        let (resp, state) = headless::ButtonBehavior::new(label)
            .enabled(!style.disabled)
            .update(self, rect);
        paint::button_with_icon(self, rect, &style, state, icon, icon_size, label);
        self.focus_ring(rect, style.corner_radius, id);
        resp
    }

    /// Draw an arbitrary custom SVG `<path d="...">` icon (24x24 grid)
    /// centred in `rect` at `size` logical pixels, rasterized and cached in
    /// the GPU atlas on demand.
    pub fn custom_icon(
        &mut self,
        name: &str,
        rect: Rect,
        svg_paths: &[&str],
        size: u32,
        tint: Color,
    ) {
        let id = self.id(name);
        let scale = self.scale;
        let ink = self.icons.custom_ink_bounds(id, svg_paths, 2.0);
        let Some((quad, raster)) = self.ink_icon_quad(rect, size, ink, 1.0, scale) else {
            return;
        };
        let Some(region) = self
            .icons
            .custom_region(id, svg_paths, raster, 2.0, &mut self.cache)
        else {
            return;
        };
        self.draw.atlas_region(quad, &self.cache, region, tint);
    }

    /// Draw a custom SVG icon on its raw 24x24 grid: `rect` is where the grid
    /// lands, with no ink fitting, so icons that share a grid line up and can
    /// be layered. `bake` is the logical side the glyph is rasterized at, kept
    /// apart from `rect` so a scale animation reuses one atlas entry. `stroke`
    /// is the line weight on the grid (Lucide's is 2). With
    /// `fill`, closed subpaths are filled as well as stroked.
    #[allow(clippy::too_many_arguments)] // Keep the established rendering API.
    pub fn grid_icon(
        &mut self,
        name: &str,
        rect: Rect,
        bake: f32,
        svg_paths: &[&str],
        fill: bool,
        stroke: f32,
        tint: Color,
    ) {
        let id = self.id(name);
        let scale = self.scale;
        let raster = (bake * scale).round().max(1.0) as u32;
        let Some(region) =
            self.icons
                .custom_region_filled(id, svg_paths, raster, stroke, fill, &mut self.cache)
        else {
            return;
        };
        self.draw.atlas_region(rect, &self.cache, region, tint);
    }
}
