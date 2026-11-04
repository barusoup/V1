//! Design tokens: the resolved layer between the raw [`Theme`] palette and
//! the widget styles — spacing, radius, density, control metrics, type
//! scale, elevation, motion and the derived colour roles.
//!
//! A theme carries one [`Tokens`] value; style builders read it so a token
//! change (density, radius ramp, motion) re-skins every widget without new
//! code paths.

mod color;
mod elevation;
mod motion;
mod space;
mod type_scale;

pub use color::{
    AccentRoles, BorderRoles, ColorRoles, FgRoles, IntentColors, IntentRoles, StateLayers,
    SurfaceRoles, composite, contrast_ratio, relative_luminance,
};
pub use elevation::{Elevation, ElevationScale, ElevationStyle, Shadow};
pub use motion::MotionTokens;
pub use space::{ControlMetrics, ControlSize, Density, Radius, RadiusScale, Space};
pub use type_scale::{FontWeight, TypeRole, TypeScale, TypeStyle};

use super::Color;
#[cfg(doc)]
use super::theme::Theme;

/// Focus-ring geometry (the colour is `roles.border.focus`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocusRing {
    /// Ring thickness in px.
    pub width: f32,
    /// Gap between a control's edge and the ring in px.
    pub offset: f32,
}

/// The full resolved token set for one theme. `Copy` so [`Theme`] stays
/// `Copy` and a theme swap remains a bitwise hand-off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tokens {
    pub density: Density,
    pub radius: RadiusScale,
    /// Control metrics indexed by [`ControlSize`] (`Xs` .. `Xl`).
    pub controls: [ControlMetrics; 5],
    pub type_scale: TypeScale,
    pub elevation: ElevationScale,
    pub motion: MotionTokens,
    pub focus: FocusRing,
}

impl Tokens {
    /// The stock token set for a dark or light UI. `border` feeds the
    /// elevation ramp's hairlines.
    pub fn new(dark: bool, border: Color) -> Self {
        Self {
            density: Density::Comfortable,
            radius: RadiusScale::default(),
            controls: ControlSize::STOCK,
            type_scale: TypeScale::default(),
            elevation: ElevationScale::new(dark, border),
            motion: MotionTokens::default(),
            focus: FocusRing {
                width: 2.0,
                offset: 2.0,
            },
        }
    }

    /// Pixels for a radius role.
    pub fn radius(&self, radius: Radius) -> f32 {
        self.radius.get(radius)
    }

    /// Metrics for a control size, with density applied to the layout
    /// lengths (height, padding, gap — never type or icon size).
    pub fn control(&self, size: ControlSize) -> ControlMetrics {
        let mut metrics = self.controls[size as usize];
        metrics.height = self.density.scale(metrics.height);
        metrics.pad_x = self.density.scale(metrics.pad_x);
        metrics.gap = self.density.scale(metrics.gap);
        metrics
    }

    /// A spacing/length under this density (already-on-scale values pass
    /// through the same function).
    pub fn space(&self, v: f32) -> f32 {
        self.density.scale(v)
    }

    pub fn type_style(&self, role: TypeRole) -> TypeStyle {
        self.type_scale.get(role)
    }

    pub fn elevation(&self, elevation: Elevation) -> &ElevationStyle {
        self.elevation.get(elevation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_space_scale_is_monotonic() {
        let steps = [
            Space::S0,
            Space::S0_5,
            Space::S1,
            Space::S1_5,
            Space::S2,
            Space::S3,
            Space::S4,
            Space::S5,
            Space::S6,
            Space::S8,
            Space::S10,
            Space::S12,
        ];
        for pair in steps.windows(2) {
            assert!(pair[0] < pair[1], "space scale must strictly increase");
        }
    }

    #[test]
    fn tokens_radius_scale_is_monotonic() {
        let t = Tokens::new(true, Color::WHITE);
        for pair in [
            Radius::None,
            Radius::Xs,
            Radius::Sm,
            Radius::Md,
            Radius::Lg,
            Radius::Xl,
        ]
        .windows(2)
        {
            assert!(
                t.radius(pair[0]) < t.radius(pair[1]),
                "{:?} must be smaller than {:?}",
                pair[0],
                pair[1]
            );
        }
        assert!(
            t.radius(Radius::Full).is_finite(),
            "Full is finite on purpose — draw clamps it to half the extent"
        );
    }

    #[test]
    fn tokens_control_sizes_are_monotonic() {
        let t = Tokens::new(true, Color::WHITE);
        for pair in [
            ControlSize::Xs,
            ControlSize::Sm,
            ControlSize::Md,
            ControlSize::Lg,
            ControlSize::Xl,
        ]
        .windows(2)
        {
            let (a, b) = (t.control(pair[0]), t.control(pair[1]));
            assert!(a.height < b.height, "control heights must grow");
            assert!(a.pad_x < b.pad_x, "control padding must grow");
            assert!(a.text_size < b.text_size, "control text must grow");
        }
    }

    #[test]
    fn tokens_compact_density_scales_and_snaps() {
        let mut t = Tokens::new(true, Color::WHITE);
        t.density = Density::Compact;
        let md = t.control(ControlSize::Md);
        assert_eq!(md.height, 24.0, "28 * 0.85 snaps to 24");
        assert_eq!(md.pad_x, 9.0, "10 * 0.85 snaps to 9 (8.5 rounds up)");
        // Type sizes are not layout lengths — density leaves them alone.
        assert_eq!(md.text_size, 13.0);
        assert_eq!(t.type_style(TypeRole::Body).size, 13.0);
    }

    #[test]
    fn tokens_radius_inner_is_concentric_and_clamps() {
        assert_eq!(Radius::inner(8.0, 4.0), 4.0, "menu rows sit inside padding");
        assert_eq!(Radius::inner(6.0, 2.0), 4.0);
        assert_eq!(Radius::inner(2.0, 4.0), 0.0, "over-inset clamps to square");
    }

    #[test]
    fn tokens_state_layer_over_transparent_is_translucent() {
        let roles = ColorRoles::derive(&super::super::theme::Theme::dark_theme());
        let layer = roles.state_layer.hover;
        let over_nothing = StateLayers::over(layer, Color::TRANSPARENT);
        assert_eq!(
            over_nothing, layer,
            "over nothing the layer stays the translucent tint"
        );
        let over_panel = StateLayers::over(layer, roles.surface.panel);
        assert_eq!(over_panel.a, 255, "over a surface it composites opaque");
        assert!(
            over_panel.r > roles.surface.panel.r,
            "a hover over a dark panel lifts it"
        );
    }

    #[test]
    fn tokens_contrast_ratio_reference() {
        assert!(
            (contrast_ratio(Color::BLACK, Color::WHITE) - 21.0).abs() < 0.05,
            "black on white is 21:1"
        );
        assert!(
            (contrast_ratio(Color::WHITE, Color::WHITE) - 1.0).abs() < 1e-4,
            "a colour on itself is 1:1"
        );
        // Composited layer math sanity: half-black over white greys.
        let mid = composite(Color::rgba(0, 0, 0, 128), Color::WHITE);
        assert!((mid.r as i32 - 128).abs() <= 1 && mid.a == 255);
    }
}
