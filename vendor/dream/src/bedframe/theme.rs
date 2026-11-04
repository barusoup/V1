//! The engine's UI theme: a restrained, neutral palette, modern metrics and
//! style builders.
//!
//! The visual language is deliberately quiet: solid neutral surfaces, hairline
//! borders, small corner radii, one text colour per role and colour-based
//! state changes (never opacity or weight). Widgets read styles from a
//! [`Theme`] every frame, so a theme swap re-skins the whole UI with no widget
//! changes.
//!
//! Two palettes ship: [`Theme::dark_theme`] (the default, neutral charcoal)
//! and [`Theme::light`] (neutral off-white). Switching between them is an
//! instant swap — no transition, no dither.

use super::tokens::{ColorRoles, ControlSize, Radius, Space, StateLayers, Tokens};
use super::widgets::{ButtonStyle, LabelStyle, MenuStyle, PanelStyle, ScrollbarStyle, SliderStyle};
use super::{Color, Rect};

/// Metrics: the sizing language of the UI. Everything is plain pixels at a
/// 1.0 device scale, so panels and controls land on whole pixels.
#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    /// Default corner radius for controls (buttons, fields, tabs).
    pub radius: f32,
    /// Corner radius for large surfaces (panels, cards, menus).
    pub radius_lg: f32,
    /// Hairline border thickness.
    pub border: f32,
    /// Default gap between sibling controls.
    pub gap: f32,
    /// Default panel padding.
    pub pad: f32,
    /// Custom window bar height (the app draws its own title bar).
    pub titlebar_h: f32,
    /// Toolbar height.
    pub toolbar_h: f32,
    /// Left tool-rail width.
    pub rail_w: f32,
    /// Right inspector width.
    pub inspector_w: f32,
    /// Square icon-button size.
    pub button: f32,
    /// Text button width.
    pub text_button_w: f32,
    /// Height of a standard control (fields, tabs, list rows).
    pub control_h: f32,
    /// Vertical spacing between inspector rows.
    pub row_gap: f32,
    /// Legacy field kept for source compatibility with the old pixel
    /// chrome; unused.
    pub notch: f32,
    /// Legacy field kept for source compatibility with the old pixel
    /// chrome; unused.
    pub shadow: f32,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            radius: 6.0,
            radius_lg: 8.0,
            border: 1.0,
            gap: 8.0,
            pad: 12.0,
            titlebar_h: 36.0,
            toolbar_h: 48.0,
            rail_w: 48.0,
            inspector_w: 260.0,
            button: 32.0,
            text_button_w: 44.0,
            control_h: 28.0,
            row_gap: 10.0,
            notch: 0.0,
            shadow: 0.0,
        }
    }
}

/// One full theme: every colour and metric the UI needs.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub dark: bool,
    /// Page background behind everything.
    pub bg: Color,
    /// Primary text.
    pub fg: Color,
    /// Secondary text (labels, captions).
    pub muted: Color,
    /// Tertiary text (hints, disabled).
    pub faint: Color,
    /// Hairline border colour.
    pub border: Color,
    /// Panel / bar surface.
    pub g1: Color,
    /// Control surface.
    pub g2: Color,
    /// Control hover surface.
    pub g3: Color,
    /// Elevated / pressed surface.
    pub pressed: Color,
    /// Opaque panel fill (kept as `pane` for source compatibility).
    pub pane: Color,
    /// Legacy translucent pane variants; kept for compatibility.
    pub pane_alpha_bg: Color,
    pub pane_alpha_g2: Color,
    /// Legacy ambient wave ramp; kept for compatibility.
    pub wave: [Color; 4],
    /// Accent used for "on" states, selection and focus (monochrome).
    pub accent: Color,
    /// Dimmer accent variant.
    pub accent_dim: Color,
    /// Text/icon colour drawn on top of `accent`.
    pub accent_text: Color,
    /// Checkerboard behind canvases.
    pub chk: [Color; 2],
    /// Status colours.
    pub danger: Color,
    pub warn: Color,
    pub ok: Color,
    pub info: Color,
    pub metrics: Metrics,
    /// Semantic colour roles derived from the base fields above — surfaces,
    /// foregrounds, borders, accent states, intent ramps and translucent
    /// state layers. Filled by `derive_roles`; mutate base fields and call
    /// [`Self::derive_roles`] to refresh it.
    pub roles: ColorRoles,
    /// Resolved design tokens: spacing, radius, density, type scale,
    /// elevation and motion. Filled by the palette constructors and
    /// [`Self::with_metrics`]; replace wholesale with [`Self::with_tokens`].
    pub tokens: Tokens,
}

impl Theme {
    /// Neutral off-white.
    pub fn light() -> Self {
        Self {
            dark: false,
            bg: Color::rgb(0xf4, 0xf4, 0xf4),
            fg: Color::rgb(0x18, 0x18, 0x18),
            muted: Color::rgb(0x52, 0x52, 0x52),
            faint: Color::rgb(0x70, 0x70, 0x70),
            border: Color::rgb(0xd8, 0xd8, 0xd8),
            g1: Color::rgb(0xff, 0xff, 0xff),
            g2: Color::rgb(0xf0, 0xf0, 0xf0),
            g3: Color::rgb(0xe4, 0xe4, 0xe4),
            pressed: Color::rgb(0xdc, 0xdc, 0xdc),
            pane: Color::rgb(0xff, 0xff, 0xff),
            pane_alpha_bg: Color::rgb(0xff, 0xff, 0xff),
            pane_alpha_g2: Color::rgb(0xf0, 0xf0, 0xf0),
            wave: [
                Color::rgb(0xf4, 0xf4, 0xf4),
                Color::rgb(0xec, 0xec, 0xec),
                Color::rgb(0xe4, 0xe4, 0xe4),
                Color::rgb(0xdb, 0xdb, 0xdb),
            ],
            accent: Color::rgb(0x3c, 0x3c, 0x3c),
            accent_dim: Color::rgb(0x70, 0x70, 0x70),
            accent_text: Color::rgb(0xff, 0xff, 0xff),
            chk: [Color::rgb(0xf0, 0xf0, 0xf0), Color::rgb(0xe4, 0xe4, 0xe4)],
            danger: Color::rgb(0xdc, 0x26, 0x26),
            warn: Color::rgb(0xd9, 0x77, 0x06),
            ok: Color::rgb(0x16, 0xa3, 0x4a),
            info: Color::rgb(0x52, 0x52, 0x52),
            metrics: Metrics::default(),
            roles: ColorRoles::PLACEHOLDER,
            tokens: Tokens::new(false, Color::TRANSPARENT),
        }
        .finish()
    }

    /// Neutral charcoal — the default engine theme.
    pub fn dark_theme() -> Self {
        Self {
            dark: true,
            bg: Color::rgb(0x18, 0x18, 0x18),
            fg: Color::rgb(0xe8, 0xe8, 0xe8),
            muted: Color::rgb(0x9a, 0x9a, 0x9a),
            faint: Color::rgb(0x8a, 0x8a, 0x8a),
            border: Color::rgb(0x34, 0x34, 0x34),
            g1: Color::rgb(0x1e, 0x1e, 0x1e),
            g2: Color::rgb(0x28, 0x28, 0x28),
            g3: Color::rgb(0x33, 0x33, 0x33),
            pressed: Color::rgb(0x23, 0x23, 0x23),
            pane: Color::rgb(0x1e, 0x1e, 0x1e),
            pane_alpha_bg: Color::rgb(0x1e, 0x1e, 0x1e),
            pane_alpha_g2: Color::rgb(0x28, 0x28, 0x28),
            wave: [
                Color::rgb(0x18, 0x18, 0x18),
                Color::rgb(0x1c, 0x1c, 0x1c),
                Color::rgb(0x20, 0x20, 0x20),
                Color::rgb(0x26, 0x26, 0x26),
            ],
            accent: Color::rgb(0xcc, 0xcc, 0xcc),
            accent_dim: Color::rgb(0x80, 0x80, 0x80),
            accent_text: Color::rgb(0x18, 0x18, 0x18),
            chk: [Color::rgb(0x20, 0x20, 0x20), Color::rgb(0x28, 0x28, 0x28)],
            danger: Color::rgb(0xe5, 0x48, 0x4d),
            warn: Color::rgb(0xe2, 0xa3, 0x36),
            ok: Color::rgb(0x46, 0xa7, 0x58),
            info: Color::rgb(0x9a, 0x9a, 0x9a),
            metrics: Metrics::default(),
            roles: ColorRoles::PLACEHOLDER,
            tokens: Tokens::new(false, Color::TRANSPARENT),
        }
        .finish()
    }

    /// Fill `roles` and `tokens` from the base fields — every palette
    /// constructor ends with this.
    fn finish(mut self) -> Self {
        self.tokens = Tokens::new(self.dark, self.border);
        self.tokens.radius.md = self.metrics.radius;
        self.tokens.radius.lg = self.metrics.radius_lg;
        self.roles = ColorRoles::derive(&self);
        self
    }

    /// Recompute `roles` from the base colour fields. Call this after
    /// mutating `bg`/`fg`/`accent`/… directly; `tokens` are untouched.
    pub fn derive_roles(mut self) -> Self {
        self.roles = ColorRoles::derive(&self);
        self
    }

    pub fn other(&self) -> Self {
        if self.dark {
            Self::light()
        } else {
            Self::dark_theme()
        }
    }

    /// Panel / large-surface style.
    pub fn panel(&self) -> PanelStyle {
        PanelStyle {
            background: self.g1,
            border: Some((self.metrics.border, self.border)),
            corner_radius: self.tokens.radius(Radius::Lg),
        }
    }

    pub fn label(&self, font: super::font::FontId) -> LabelStyle {
        LabelStyle::new(font).color(self.fg)
    }

    pub fn button(&self, font: super::font::FontId) -> ButtonStyle {
        let layers = self.roles.state_layer;
        ButtonStyle {
            font,
            size: self.tokens.control(ControlSize::Md).text_size,
            text: self.fg,
            text_hover: self.fg,
            text_disabled: self.roles.fg.disabled,
            normal: self.g2,
            hover: StateLayers::over(layers.hover, self.g2),
            pressed: StateLayers::over(layers.pressed, self.g2),
            accent: self.accent,
            corner_radius: self
                .tokens
                .radius(self.tokens.control(ControlSize::Md).radius),
            disabled: false,
            on_icon: self.accent_text,
            border: None,
        }
    }

    /// [`crate::ui::widgets::adapters::ButtonVariant`] → style. `Secondary` is
    /// [`Self::button`]; the others derive from the primary/ghost/danger
    /// surfaces; `Outline` is transparent chrome with the role border;
    /// `Link` is accent-coloured text with no fill or border.
    pub fn button_variant(
        &self,
        font: super::font::FontId,
        variant: crate::ui::widgets::adapters::ButtonVariant,
    ) -> ButtonStyle {
        use crate::ui::widgets::adapters::ButtonVariant as V;
        match variant {
            V::Primary => self.primary_button(font),
            V::Secondary => self.button(font),
            V::Outline => {
                let layers = self.roles.state_layer;
                ButtonStyle {
                    normal: Color::TRANSPARENT,
                    hover: StateLayers::over(layers.hover, Color::TRANSPARENT),
                    pressed: StateLayers::over(layers.pressed, Color::TRANSPARENT),
                    border: Some(self.roles.border.default),
                    ..self.button(font)
                }
            }
            V::Ghost => self.ghost_button(font),
            V::Danger => self.danger_button(font),
            V::Link => ButtonStyle {
                normal: Color::TRANSPARENT,
                hover: Color::TRANSPARENT,
                pressed: Color::TRANSPARENT,
                text: self.roles.accent.default,
                text_hover: self.roles.accent.hover,
                border: None,
                ..self.button(font)
            },
        }
    }

    /// Segmented selector: quiet `g1` track with a hairline border, the
    /// pill in the button's `normal` colour. The track border width is a
    /// runtime hairline — [`Ui::segmented`](crate::ui::widgets::Ui::segmented)
    /// fills it in from the current scale when it builds this style.
    pub fn segmented(&self, font: super::font::FontId) -> super::widgets::SegmentedStyle {
        let button = self.button(font);
        super::widgets::SegmentedStyle {
            track: self.g1,
            track_border: Some((0.0, self.border)),
            pill: button.normal,
            inset: super::tokens::Space::S0_5,
            text: button.text_disabled,
            text_selected: button.text,
            glide: self.tokens.motion.segmented,
            squash: 0.06,
            button,
        }
    }

    /// Slider: a quiet track, a mid-grey travelled part and a bright knob, so
    /// the knob stands out against both ends of the bar and the fill reads
    /// against the track.
    pub fn slider(&self, font: super::font::FontId) -> SliderStyle {
        SliderStyle {
            font,
            track: self.g2,
            fill: self.accent_dim,
            knob: self.fg,
            show_value: false,
        }
    }

    pub fn menu(&self, font: super::font::FontId) -> MenuStyle {
        MenuStyle {
            font,
            size: 13.0,
            text: self.fg,
            item: Color::TRANSPARENT,
            item_hover: self.roles.state_layer.hover,
            item_height: self.tokens.control(ControlSize::Md).height,
            gap: 2.0,
            padding: Space::S1,
            align: super::font::Align::Left,
            background: self.g1,
            border: self.border,
            radius: self.tokens.radius(Radius::Lg),
        }
    }

    /// Icon-button style: quiet at rest, a soft surface on hover, inverted
    /// when latched on.
    pub fn icon_button(&self, font: super::font::FontId) -> ButtonStyle {
        let layers = self.roles.state_layer;
        ButtonStyle {
            font,
            size: self.tokens.control(ControlSize::Md).text_size,
            text: self.muted,
            text_hover: self.fg,
            text_disabled: self.roles.fg.disabled,
            normal: Color::TRANSPARENT,
            hover: StateLayers::over(layers.hover, Color::TRANSPARENT),
            pressed: StateLayers::over(layers.pressed, Color::TRANSPARENT),
            accent: self.accent,
            corner_radius: self
                .tokens
                .radius(self.tokens.control(ControlSize::Md).radius),
            disabled: false,
            on_icon: self.accent_text,
            border: None,
        }
    }

    /// Dimmed text (secondary labels).
    pub fn dim(&self) -> Color {
        self.muted
    }

    /// Inverted foreground/background for "on" states.
    pub fn invert(&self) -> Color {
        self.fg
    }

    /// Tailwind Slate dark palette with a Blue-500 (`#3b82f6`) accent.
    pub fn slate_dark() -> Self {
        Self {
            dark: true,
            bg: Color::rgb(0x02, 0x06, 0x17),
            fg: Color::rgb(0xf8, 0xfa, 0xfc),
            muted: Color::rgb(0x94, 0xa3, 0xb8),
            faint: Color::rgb(0x76, 0x87, 0x9f),
            border: Color::rgb(0x23, 0x31, 0x4a),
            g1: Color::rgb(0x0f, 0x17, 0x2a),
            g2: Color::rgb(0x1e, 0x29, 0x3b),
            g3: Color::rgb(0x33, 0x41, 0x55),
            pressed: Color::rgb(0x17, 0x20, 0x33),
            pane: Color::rgb(0x0f, 0x17, 0x2a),
            pane_alpha_bg: Color::rgb(0x0f, 0x17, 0x2a),
            pane_alpha_g2: Color::rgb(0x1e, 0x29, 0x3b),
            wave: [
                Color::rgb(0x02, 0x06, 0x17),
                Color::rgb(0x09, 0x10, 0x22),
                Color::rgb(0x0f, 0x17, 0x2a),
                Color::rgb(0x1e, 0x29, 0x3b),
            ],
            accent: Color::rgb(0x25, 0x63, 0xeb),
            accent_dim: Color::rgb(0x1d, 0x4e, 0xd8),
            accent_text: Color::WHITE,
            chk: [Color::rgb(0x0f, 0x17, 0x2a), Color::rgb(0x1e, 0x29, 0x3b)],
            danger: Color::rgb(0xef, 0x44, 0x44),
            warn: Color::rgb(0xf5, 0x9e, 0x0b),
            ok: Color::rgb(0x22, 0xc5, 0x5e),
            info: Color::rgb(0x38, 0xbd, 0xf8),
            metrics: Metrics::default(),
            roles: ColorRoles::PLACEHOLDER,
            tokens: Tokens::new(false, Color::TRANSPARENT),
        }
        .finish()
    }

    /// Tailwind Slate light palette with a Blue-600 (`#2563eb`) accent.
    pub fn slate_light() -> Self {
        Self {
            dark: false,
            bg: Color::rgb(0xf8, 0xfa, 0xfc),
            fg: Color::rgb(0x0f, 0x17, 0x2a),
            muted: Color::rgb(0x47, 0x55, 0x69),
            faint: Color::rgb(0x64, 0x74, 0x8b),
            border: Color::rgb(0xd2, 0xda, 0xe4),
            g1: Color::WHITE,
            g2: Color::rgb(0xf1, 0xf5, 0xf9),
            g3: Color::rgb(0xe2, 0xe8, 0xf0),
            pressed: Color::rgb(0xcb, 0xd5, 0xe1),
            pane: Color::WHITE,
            pane_alpha_bg: Color::WHITE,
            pane_alpha_g2: Color::rgb(0xf1, 0xf5, 0xf9),
            wave: [
                Color::rgb(0xf8, 0xfa, 0xfc),
                Color::rgb(0xf1, 0xf5, 0xf9),
                Color::rgb(0xe2, 0xe8, 0xf0),
                Color::rgb(0xcb, 0xd5, 0xe1),
            ],
            accent: Color::rgb(0x25, 0x63, 0xeb),
            accent_dim: Color::rgb(0x3b, 0x82, 0xf6),
            accent_text: Color::WHITE,
            chk: [Color::rgb(0xf1, 0xf5, 0xf9), Color::rgb(0xe2, 0xe8, 0xf0)],
            danger: Color::rgb(0xdc, 0x26, 0x26),
            warn: Color::rgb(0xd9, 0x77, 0x06),
            ok: Color::rgb(0x16, 0xa3, 0x4a),
            info: Color::rgb(0x02, 0x84, 0xc7),
            metrics: Metrics::default(),
            roles: ColorRoles::PLACEHOLDER,
            tokens: Tokens::new(false, Color::TRANSPARENT),
        }
        .finish()
    }

    /// Return a copy of this theme with a custom accent colour and on-accent text colour.
    pub fn with_accent(mut self, accent: Color, accent_text: Color) -> Self {
        self.accent = accent;
        self.accent_dim = mix(accent, self.g2, 0.25);
        self.accent_text = accent_text;
        self.derive_roles()
    }

    /// Return a copy of this theme with custom sizing metrics. The radius
    /// tokens track the metric radii so styles stay consistent.
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = metrics;
        self.tokens.radius.md = metrics.radius;
        self.tokens.radius.lg = metrics.radius_lg;
        self
    }

    /// Return a copy of this theme with a custom token set (density,
    /// radius ramp, type scale, elevation, motion).
    pub fn with_tokens(mut self, tokens: Tokens) -> Self {
        self.tokens = tokens;
        self
    }

    /// Scrollbar look derived from the palette: quiet mid-grey thumb that
    /// lifts toward the foreground while hovered or dragged.
    pub fn scrollbar(&self) -> ScrollbarStyle {
        ScrollbarStyle {
            width: 5.0,
            inset: 2.0,
            min_thumb: 24.0,
            idle: mix(self.g1, self.fg, 0.18),
            active: mix(self.g1, self.fg, 0.45),
        }
    }

    /// Primary call-to-action button style: filled with `accent` and `accent_text`.
    pub fn primary_button(&self, font: super::font::FontId) -> ButtonStyle {
        ButtonStyle {
            font,
            size: self.tokens.control(ControlSize::Md).text_size,
            text: self.accent_text,
            text_hover: self.accent_text,
            text_disabled: self.roles.fg.disabled,
            normal: self.accent,
            hover: self.roles.accent.hover,
            pressed: self.roles.accent.pressed,
            accent: self.accent,
            corner_radius: self
                .tokens
                .radius(self.tokens.control(ControlSize::Md).radius),
            disabled: false,
            on_icon: self.accent_text,
            border: None,
        }
    }

    /// Ghost button style: transparent at rest, subtle surface on hover.
    pub fn ghost_button(&self, font: super::font::FontId) -> ButtonStyle {
        let layers = self.roles.state_layer;
        ButtonStyle {
            font,
            size: self.tokens.control(ControlSize::Md).text_size,
            text: self.muted,
            text_hover: self.fg,
            text_disabled: self.roles.fg.disabled,
            normal: Color::TRANSPARENT,
            hover: StateLayers::over(layers.hover, Color::TRANSPARENT),
            pressed: StateLayers::over(layers.pressed, Color::TRANSPARENT),
            accent: self.accent,
            corner_radius: self
                .tokens
                .radius(self.tokens.control(ControlSize::Md).radius),
            disabled: false,
            on_icon: self.accent_text,
            border: None,
        }
    }

    /// Destructive / danger button style.
    pub fn danger_button(&self, font: super::font::FontId) -> ButtonStyle {
        let solid = self.roles.intent.danger.solid;
        ButtonStyle {
            font,
            size: self.tokens.control(ControlSize::Md).text_size,
            text: Color::WHITE,
            text_hover: Color::WHITE,
            text_disabled: self.roles.fg.disabled,
            normal: solid,
            hover: solid.lighten(0.12),
            pressed: solid.darken(0.12),
            accent: self.danger,
            corner_radius: self
                .tokens
                .radius(self.tokens.control(ControlSize::Md).radius),
            disabled: false,
            on_icon: Color::WHITE,
            border: None,
        }
    }

    /// Themed text input style (recessed fill, hairline border, accent focus ring).
    pub fn text_field(&self, font: super::font::FontId) -> super::widgets::TextFieldStyle {
        super::widgets::TextFieldStyle {
            font,
            size: self.tokens.control(ControlSize::Md).text_size,
            text: self.fg,
            text_disabled: self.roles.fg.disabled,
            normal: self.g1,
            hover: self.g3,
            accent: self.accent,
            corner_radius: self
                .tokens
                .radius(self.tokens.control(ControlSize::Md).radius),
            border_width: self.metrics.border.max(1.0),
            caret: self.muted,
            caret_width: 1.0,
            caret_inset: 7.0,
            caret_blink: 1.06,
            selection_alpha: 75,
            selection_inset: 5.0,
            text_inset: 10.0,
            fade_out: 0.09,
            disabled: false,
            password: false,
        }
    }

    /// Themed text input that masks its contents — one bullet per
    /// character. Same chrome as [`Self::text_field`]; the field still
    /// edits the real string and copy/cut emits it unmasked.
    pub fn password_field(&self, font: super::font::FontId) -> super::widgets::TextFieldStyle {
        self.text_field(font).password(true)
    }
}

impl Default for Theme {
    /// The default engine theme: neutral charcoal.
    fn default() -> Self {
        Self::dark_theme()
    }
}

/// 4x4 ordered-dither threshold in 0..1, used by [`crate::ui::draw::DrawList::dither`]
/// for canvas checkerboards. (The old radial theme flood used it too; the
/// theme now crossfades smoothly instead.)
pub fn bayer2(x: f32, y: f32) -> f32 {
    const M: [[u32; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    let xi = (x.floor() as i64).rem_euclid(4) as usize;
    let yi = (y.floor() as i64).rem_euclid(4) as usize;
    (M[yi][xi] as f32 + 0.5) / 16.0
}

/// Snap a rect to whole pixels so edges stay crisp.
pub fn snap(rect: Rect) -> Rect {
    let r = super::layout::snap_rect(rect, 1.0);
    Rect::new(r.x, r.y, r.w.max(1.0), r.h.max(1.0))
}

/// Same colour at a fractional alpha (CSS `rgba(.74)`).
pub fn alpha(color: Color, a: f32) -> Color {
    color.with_alpha((color.a as f32 * a.clamp(0.0, 1.0)).round() as u8)
}

/// Perceptual mix.
pub fn mix(a: Color, b: Color, t: f32) -> Color {
    super::tween::LerpValue::lerp_value(a, b, t.clamp(0.0, 1.0))
}

/// Named hue helper: the inspector reads colour names from these buckets.
pub fn color_name(color: Color) -> String {
    let (r, g, b) = (
        color.r as f32 / 255.0,
        color.g as f32 / 255.0,
        color.b as f32 / 255.0,
    );
    let hsv = rgb_to_hsv((color.r, color.g, color.b));
    let h = hsv.0;
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let l = (max + min) * 0.5;
    let s = if max == min {
        0.0
    } else {
        (max - min) / (1.0 - (2.0 * l - 1.0).abs())
    };
    if l < 0.1 {
        return "Black".into();
    }
    if l > 0.93 {
        return "White".into();
    }
    if s < 0.12 {
        return if l < 0.3 {
            "Dark Gray"
        } else if l > 0.7 {
            "Light Gray"
        } else {
            "Gray"
        }
        .into();
    }
    let base = if !(15.0..345.0).contains(&h) {
        "Red"
    } else if h < 45.0 {
        "Orange"
    } else if h < 70.0 {
        "Yellow"
    } else if h < 160.0 {
        "Green"
    } else if h < 190.0 {
        "Teal"
    } else if h < 215.0 {
        "Cyan"
    } else if h < 255.0 {
        "Blue"
    } else if h < 290.0 {
        "Purple"
    } else {
        "Pink"
    };
    let prefix = if l > 0.75 {
        "Light "
    } else if l < 0.22 {
        "Dark "
    } else if s < 0.35 {
        "Muted "
    } else {
        ""
    };
    let mut name = format!("{prefix}{base}");
    if base == "Orange" && l < 0.4 {
        name = "Brown".into();
    }
    if base == "Red" && l > 0.78 {
        name = "Pink".into();
    }
    if base == "Red" && s > 0.5 && l < 0.2 {
        name = "Dark Red".into();
    }
    name
}

/// sRGB bytes -> HSV (h in 0..360, s/v in 0..1).
pub fn rgb_to_hsv(rgb: (u8, u8, u8)) -> (f32, f32, f32) {
    let (r, g, b) = (
        rgb.0 as f32 / 255.0,
        rgb.1 as f32 / 255.0,
        rgb.2 as f32 / 255.0,
    );
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let d = max - min;
    let mut h = 0.0;
    if d > 0.0 {
        h = if max == r {
            ((g - b) / d) % 6.0
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        h = (h * 60.0 + 360.0) % 360.0;
    }
    (h, if max > 0.0 { d / max } else { 0.0 }, max)
}

/// HSV (h 0..360, s/v 0..1) -> sRGB bytes.
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let f = |n: f32| {
        let k = (n + h / 60.0) % 6.0;
        let channel = v - v * s * k.min(4.0 - k).clamp(0.0, 1.0);
        (channel * 255.0).round().clamp(0.0, 255.0) as u8
    };
    (f(5.0), f(3.0), f(1.0))
}

/// `#rrggbb` for a colour.
pub fn hex(color: Color) -> String {
    format!("#{:02X}{:02X}{:02X}", color.r, color.g, color.b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palettes_are_neutral_and_ranked() {
        let dark = Theme::dark_theme();
        assert!(dark.dark);
        // Neutral: the surface ramp is greyscale (r == g == b).
        for surface in [dark.bg, dark.g1, dark.g2, dark.g3] {
            assert!(
                surface.r == surface.g && surface.g == surface.b,
                "surface {surface:?} is not neutral"
            );
        }
        // Surfaces brighten from page to control.
        assert!(dark.bg.r < dark.g1.r && dark.g1.r < dark.g2.r && dark.g2.r < dark.g3.r);
        // Text is brighter than every surface it sits on.
        assert!(dark.fg.r > dark.g3.r);
        assert!(dark.muted.r > dark.g2.r && dark.muted.r < dark.fg.r);
    }

    #[test]
    fn light_and_dark_are_opposite() {
        let light = Theme::light();
        let dark = Theme::dark_theme();
        assert!(!light.dark && dark.dark);
        assert!(light.other().dark);
        assert!(!dark.other().dark);
        // The light theme's page is brighter than its text and vice versa.
        assert!(light.bg.r > light.fg.r);
        assert!(dark.bg.r < dark.fg.r);
    }

    #[test]
    fn menu_style_is_monochrome_and_sized() {
        let theme = Theme::dark_theme();
        let menu = theme.menu(super::super::font::FontId(0));
        for color in [menu.text, menu.background, menu.item_hover] {
            assert!(color.r == color.g && color.g == color.b, "menu is neutral");
        }
        assert!(menu.item_height >= 24.0 && menu.radius > 0.0);
    }

    #[test]
    fn hsv_round_trip() {
        for rgb in [
            (0x2fu8, 0x80u8, 0xedu8),
            (0xc0u8, 0x39u8, 0x2bu8),
            (0xf7u8, 0xf0u8, 0xddu8),
            (0x18u8, 0x18u8, 0x1au8),
            (0x00u8, 0x00u8, 0x00u8),
            (0xffu8, 0xffu8, 0xffu8),
        ] {
            let (h, s, v) = rgb_to_hsv(rgb);
            let back = hsv_to_rgb(h, s, v);
            for i in 0..3 {
                let a = [rgb.0, rgb.1, rgb.2][i];
                let b = [back.0, back.1, back.2][i];
                assert!((a as i32 - b as i32).abs() <= 1, "{rgb:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn color_names_match_buckets() {
        assert_eq!(color_name(Color::rgb(0, 0, 0)), "Black");
        assert_eq!(color_name(Color::rgb(255, 255, 255)), "White");
        assert_eq!(color_name(Color::rgb(0x2f, 0x80, 0xed)), "Cyan");
        assert_eq!(color_name(Color::rgb(0x2e, 0x7d, 0x32)), "Green");
    }

    /// The contrast contract every shipped palette must meet: body and
    /// muted text are AA-legible on every surface; subtle text on panels;
    /// accent text on the accent; borders separate without shouting, and
    /// the focus ring is unmistakable.
    fn check_contrast(theme: &Theme, name: &str) {
        use super::super::tokens::contrast_ratio;
        let roles = &theme.roles;
        let surfaces = [
            roles.surface.base,
            roles.surface.panel,
            roles.surface.raised,
            roles.surface.overlay,
            roles.surface.sunken,
        ];
        for (label, fg) in [
            ("fg.default", roles.fg.default),
            ("fg.muted", roles.fg.muted),
        ] {
            for surface in surfaces {
                assert!(
                    contrast_ratio(fg, surface) >= 4.5,
                    "[{name}] {label} vs surface {surface:?}: {:.2} < 4.5",
                    contrast_ratio(fg, surface)
                );
            }
        }
        assert!(
            contrast_ratio(roles.fg.subtle, roles.surface.panel) >= 4.5,
            "[{name}] fg.subtle vs panel: {:.2} < 4.5",
            contrast_ratio(roles.fg.subtle, roles.surface.panel)
        );
        assert!(
            contrast_ratio(roles.accent.fg, roles.accent.default) >= 4.5,
            "[{name}] accent.fg vs accent: {:.2} < 4.5",
            contrast_ratio(roles.accent.fg, roles.accent.default)
        );
        assert!(
            contrast_ratio(roles.border.default, roles.surface.panel) >= 1.3,
            "[{name}] border vs panel: {:.2} < 1.3",
            contrast_ratio(roles.border.default, roles.surface.panel)
        );
        assert!(
            contrast_ratio(roles.border.focus, roles.surface.panel) >= 3.0,
            "[{name}] focus ring vs panel: {:.2} < 3.0",
            contrast_ratio(roles.border.focus, roles.surface.panel)
        );
    }

    #[test]
    fn theme_contrast_light() {
        check_contrast(&Theme::light(), "light");
    }

    #[test]
    fn theme_contrast_dark() {
        check_contrast(&Theme::dark_theme(), "dark_theme");
    }

    #[test]
    fn theme_contrast_slate_dark() {
        check_contrast(&Theme::slate_dark(), "slate_dark");
    }

    #[test]
    fn theme_contrast_slate_light() {
        check_contrast(&Theme::slate_light(), "slate_light");
    }

    /// Every `new`/`default` style ctor is exactly `Theme::default()`'s
    /// builder — no drift between stock styles and the default theme.
    #[test]
    fn theme_defaults_match_styles() {
        use super::super::widgets::TextFieldStyle;
        let theme = Theme::default();
        let font = super::super::font::FontId(0);
        assert_eq!(ButtonStyle::new(font), theme.button(font));
        assert_eq!(PanelStyle::default(), theme.panel());
        assert_eq!(SliderStyle::new(font), theme.slider(font));
        assert_eq!(MenuStyle::new(font), theme.menu(font));
        assert_eq!(TextFieldStyle::new(font), theme.text_field(font));
        assert_eq!(ScrollbarStyle::default(), theme.scrollbar());
    }

    /// Mutating a base field then `derive_roles` re-derives every role.
    #[test]
    fn derive_roles_refreshes_after_mutation() {
        let theme = Theme::dark_theme().with_accent(Color::rgb(0x40, 0x90, 0xe0), Color::WHITE);
        assert_eq!(theme.roles.accent.default, theme.accent);
        assert_eq!(theme.roles.accent.fg, Color::WHITE);
        assert_eq!(theme.roles.border.focus, theme.accent);
        assert_eq!(theme.roles.state_layer.selected.a, 36);
    }

    #[test]
    fn snap_rounds_to_pixels() {
        // Edges snap, sizes follow: 10.4/20.6 snap to 10/21 and the far
        // edges 43.6/65.1 snap to 44/65.
        let rect = snap(Rect::new(10.4, 20.6, 33.2, 44.5));
        assert_eq!(rect, Rect::new(10.0, 21.0, 34.0, 44.0));
    }
}
