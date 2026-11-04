//! Colour roles: the semantic palette derived from a [`Theme`]'s base
//! fields. Widgets read roles (`roles.fg.muted`, `roles.intent.danger`)
//! instead of raw greys, so a palette change stays a data change.

use super::super::Color;
use super::super::theme::{self, Theme};

/// Every semantic colour the UI needs, derived from a [`Theme`]. Build
/// with [`ColorRoles::derive`]; kept inside the theme so a theme swap
/// re-skins everything at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorRoles {
    pub surface: SurfaceRoles,
    pub fg: FgRoles,
    pub border: BorderRoles,
    pub accent: AccentRoles,
    pub intent: IntentRoles,
    pub state_layer: StateLayers,
}

/// The surfaces content sits on, from the page up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceRoles {
    /// Page background.
    pub base: Color,
    /// Cards, panels, bars.
    pub panel: Color,
    /// Panels floating above the page.
    pub raised: Color,
    /// Popups and menus.
    pub overlay: Color,
    /// Recessed wells (inputs, tracks).
    pub sunken: Color,
}

/// Foreground (text and icon) colours by emphasis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FgRoles {
    pub default: Color,
    pub muted: Color,
    pub subtle: Color,
    pub disabled: Color,
    /// Text drawn on `accent` fills.
    pub on_accent: Color,
}

/// Border colours by weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BorderRoles {
    /// Barely-there separators.
    pub subtle: Color,
    pub default: Color,
    /// Emphasis borders (selected cards).
    pub strong: Color,
    /// The focus ring.
    pub focus: Color,
}

/// Accent ramp: the brand/action colour and its interaction states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccentRoles {
    pub default: Color,
    pub hover: Color,
    pub pressed: Color,
    /// Tinted background for selected items.
    pub subtle_bg: Color,
    /// Text/icons that sit on the accent fills.
    pub fg: Color,
}

/// Status/feedback colours. Each intent is an [`IntentColors`] — a solid
/// colour plus the surfaces and text tints it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntentRoles {
    pub danger: IntentColors,
    pub warn: IntentColors,
    pub ok: IntentColors,
    pub info: IntentColors,
}

/// One intent's full ramp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntentColors {
    /// The saturated status colour (badges, danger buttons).
    pub solid: Color,
    /// Tinted background (callouts, inline alerts).
    pub subtle_bg: Color,
    /// Text on neutral surfaces.
    pub fg: Color,
    /// Hairline border around the subtle background.
    pub border: Color,
}

impl IntentColors {
    const PLACEHOLDER: Self = Self {
        solid: Color::TRANSPARENT,
        subtle_bg: Color::TRANSPARENT,
        fg: Color::TRANSPARENT,
        border: Color::TRANSPARENT,
    };
}

/// Translucent overlays for interaction states. These are *layers*, not
/// surface colours: draw them over whatever is underneath with
/// [`StateLayers::over`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateLayers {
    /// ~6% foreground.
    pub hover: Color,
    /// ~10% foreground.
    pub pressed: Color,
    /// ~14% accent.
    pub selected: Color,
}

impl StateLayers {
    /// The colour a state layer produces over `base`. Over nothing the
    /// layer stays translucent — it tints whatever surface it is drawn on.
    pub fn over(layer: Color, base: Color) -> Color {
        if base.a == 0 {
            layer
        } else {
            composite(layer, base)
        }
    }
}

impl ColorRoles {
    /// Transparent placeholder so palette constructors can build the
    /// struct before `finish` fills it from the real fields.
    pub(crate) const PLACEHOLDER: Self = Self {
        surface: SurfaceRoles {
            base: Color::TRANSPARENT,
            panel: Color::TRANSPARENT,
            raised: Color::TRANSPARENT,
            overlay: Color::TRANSPARENT,
            sunken: Color::TRANSPARENT,
        },
        fg: FgRoles {
            default: Color::TRANSPARENT,
            muted: Color::TRANSPARENT,
            subtle: Color::TRANSPARENT,
            disabled: Color::TRANSPARENT,
            on_accent: Color::TRANSPARENT,
        },
        border: BorderRoles {
            subtle: Color::TRANSPARENT,
            default: Color::TRANSPARENT,
            strong: Color::TRANSPARENT,
            focus: Color::TRANSPARENT,
        },
        accent: AccentRoles {
            default: Color::TRANSPARENT,
            hover: Color::TRANSPARENT,
            pressed: Color::TRANSPARENT,
            subtle_bg: Color::TRANSPARENT,
            fg: Color::TRANSPARENT,
        },
        intent: IntentRoles {
            danger: IntentColors::PLACEHOLDER,
            warn: IntentColors::PLACEHOLDER,
            ok: IntentColors::PLACEHOLDER,
            info: IntentColors::PLACEHOLDER,
        },
        state_layer: StateLayers {
            hover: Color::TRANSPARENT,
            pressed: Color::TRANSPARENT,
            selected: Color::TRANSPARENT,
        },
    };

    /// Derive every role from a theme's base fields. Called by the palette
    /// constructors and by [`Theme::derive_roles`] after the base fields
    /// change.
    pub fn derive(theme: &Theme) -> Self {
        let mix = theme::mix;
        let intent = |solid: Color| IntentColors {
            solid,
            subtle_bg: mix(theme.g1, solid, 0.14),
            fg: if theme.dark {
                solid.lighten(0.15)
            } else {
                solid.darken(0.15)
            },
            border: mix(theme.g1, solid, 0.4),
        };
        Self {
            surface: SurfaceRoles {
                base: theme.bg,
                panel: theme.g1,
                raised: theme.g1,
                overlay: if theme.dark {
                    mix(theme.g1, theme.g2, 0.5)
                } else {
                    theme.g1
                },
                sunken: if theme.dark { theme.bg } else { theme.g2 },
            },
            fg: FgRoles {
                default: theme.fg,
                muted: theme.muted,
                subtle: theme.faint,
                disabled: mix(theme.faint, theme.g1, 0.35),
                on_accent: theme.accent_text,
            },
            border: BorderRoles {
                subtle: mix(theme.border, theme.g1, 0.5),
                default: theme.border,
                strong: mix(theme.border, theme.fg, 0.25),
                focus: theme.accent,
            },
            accent: AccentRoles {
                default: theme.accent,
                hover: if theme.dark {
                    theme.accent.lighten(0.08)
                } else {
                    theme.accent.darken(0.08)
                },
                pressed: if theme.dark {
                    theme.accent.darken(0.08)
                } else {
                    theme.accent.lighten(0.08)
                },
                subtle_bg: mix(theme.g1, theme.accent, 0.14),
                fg: theme.accent_text,
            },
            intent: IntentRoles {
                danger: intent(theme.danger),
                warn: intent(theme.warn),
                ok: intent(theme.ok),
                info: intent(theme.info),
            },
            state_layer: StateLayers {
                hover: theme.fg.with_alpha(15),
                pressed: theme.fg.with_alpha(26),
                selected: theme.accent.with_alpha(36),
            },
        }
    }
}

/// Straight-alpha source-over compositing in sRGB bytes: what the GPU
/// produces when `top` is drawn over `under`.
pub fn composite(top: Color, under: Color) -> Color {
    let a_t = top.a as f32 / 255.0;
    let a_u = under.a as f32 / 255.0;
    let a = a_t + a_u * (1.0 - a_t);
    if a <= 0.0 {
        return Color::TRANSPARENT;
    }
    let channel = |t: u8, u: u8| {
        ((t as f32 * a_t + u as f32 * a_u * (1.0 - a_t)) / a)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color::rgba(
        channel(top.r, under.r),
        channel(top.g, under.g),
        channel(top.b, under.b),
        (a * 255.0).round() as u8,
    )
}

/// WCAG 2.x relative luminance of an sRGB colour (alpha ignored).
pub fn relative_luminance(color: Color) -> f32 {
    let channel = |v: u8| {
        let c = v as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b)
}

/// WCAG 2.x contrast ratio between two colours, 1..21.
pub fn contrast_ratio(a: Color, b: Color) -> f32 {
    let (l1, l2) = (relative_luminance(a), relative_luminance(b));
    let (hi, lo) = (l1.max(l2), l1.min(l2));
    (hi + 0.05) / (lo + 0.05)
}
