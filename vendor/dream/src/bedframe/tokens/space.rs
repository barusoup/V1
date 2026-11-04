//! Spacing, radius, density and control-size tokens.

use super::type_scale::TypeRole;

/// The spacing scale: a 4 px base with half and one-and-a-half steps where
/// controls need them. Tokens, not a function — layouts read the constant
/// that matches the visual distance they want.
pub struct Space;

impl Space {
    pub const S0: f32 = 0.0;
    pub const S0_5: f32 = 2.0;
    pub const S1: f32 = 4.0;
    pub const S1_5: f32 = 6.0;
    pub const S2: f32 = 8.0;
    pub const S3: f32 = 12.0;
    pub const S4: f32 = 16.0;
    pub const S5: f32 = 20.0;
    pub const S6: f32 = 24.0;
    pub const S8: f32 = 32.0;
    pub const S10: f32 = 40.0;
    pub const S12: f32 = 48.0;
}

/// Corner-radius roles, smallest to largest. `Full` is the pill/circle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Radius {
    None,
    Xs,
    Sm,
    Md,
    Lg,
    Xl,
    Full,
}

impl Radius {
    /// The concentric-radius rule: a nested shape's radius is its parent's
    /// minus the gap between them, so nested curves stay parallel instead
    /// of pinching at the corners. Clamps at 0 — an inset larger than the
    /// outer radius leaves a square.
    pub fn inner(outer: f32, inset: f32) -> f32 {
        (outer - inset).max(0.0)
    }
}

/// The radius ramp in pixels. `Full` is deliberately finite — the draw
/// path clamps any radius to half the shape's extent, so a huge value is
/// what makes a stadium, not a sentinel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiusScale {
    pub xs: f32,
    pub sm: f32,
    pub md: f32,
    pub lg: f32,
    pub xl: f32,
}

impl Default for RadiusScale {
    fn default() -> Self {
        Self {
            xs: 3.0,
            sm: 4.0,
            md: 6.0,
            lg: 8.0,
            xl: 12.0,
        }
    }
}

impl RadiusScale {
    pub fn get(&self, radius: Radius) -> f32 {
        match radius {
            Radius::None => 0.0,
            Radius::Xs => self.xs,
            Radius::Sm => self.sm,
            Radius::Md => self.md,
            Radius::Lg => self.lg,
            Radius::Xl => self.xl,
            Radius::Full => 9999.0,
        }
    }
}

/// Layout density. `Compact` scales lengths down and snaps to whole pixels;
/// type sizes are never scaled (a smaller font is a different type role,
/// not a shrunken one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Density {
    #[default]
    Comfortable,
    Compact,
}

impl Density {
    /// Scale one length (height, padding, gap) for this density.
    pub fn scale(self, v: f32) -> f32 {
        match self {
            Density::Comfortable => v,
            Density::Compact => (v * 0.85).round(),
        }
    }
}

/// Control size steps, smallest to largest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ControlSize {
    Xs,
    Sm,
    #[default]
    Md,
    Lg,
    Xl,
}

/// The resolved metrics of one control size. `text` is the type role a
/// label would ask for; `text_size` is the pixel size to actually render
/// (Lg pairs the Body role's style with a larger size, so the two fields
/// can disagree).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlMetrics {
    pub height: f32,
    pub pad_x: f32,
    pub gap: f32,
    pub icon: f32,
    pub text: TypeRole,
    pub text_size: f32,
    pub radius: Radius,
}

impl ControlSize {
    /// The stock metrics per size (index = `size as usize`); `Tokens`
    /// clones this table verbatim.
    pub(crate) const STOCK: [ControlMetrics; 5] = [
        ControlMetrics {
            height: 20.0,
            pad_x: 6.0,
            gap: 4.0,
            icon: 12.0,
            text: TypeRole::Caption,
            text_size: 11.0,
            radius: Radius::Sm,
        },
        ControlMetrics {
            height: 24.0,
            pad_x: 8.0,
            gap: 6.0,
            icon: 14.0,
            text: TypeRole::Small,
            text_size: 12.0,
            radius: Radius::Md,
        },
        ControlMetrics {
            height: 28.0,
            pad_x: 10.0,
            gap: 6.0,
            icon: 16.0,
            text: TypeRole::Body,
            text_size: 13.0,
            radius: Radius::Md,
        },
        ControlMetrics {
            height: 32.0,
            pad_x: 12.0,
            gap: 8.0,
            icon: 16.0,
            text: TypeRole::Body,
            text_size: 14.0,
            radius: Radius::Md,
        },
        ControlMetrics {
            height: 40.0,
            pad_x: 16.0,
            gap: 8.0,
            icon: 20.0,
            text: TypeRole::BodyLg,
            text_size: 15.0,
            radius: Radius::Lg,
        },
    ];

    /// The largest stock size whose height fits `h` (`Xs` when nothing
    /// does — the smallest control is still a control).
    pub fn for_height(h: f32) -> ControlSize {
        const SIZES: [ControlSize; 5] = [
            ControlSize::Xs,
            ControlSize::Sm,
            ControlSize::Md,
            ControlSize::Lg,
            ControlSize::Xl,
        ];
        let mut best = ControlSize::Xs;
        for (i, m) in Self::STOCK.iter().enumerate() {
            if m.height <= h {
                best = SIZES[i];
            }
        }
        best
    }
}
