//! Elevation tokens: the shadow/border treatment for each surface level.
//!
//! Data only — wiring these into the shadow call sites is a later phase.

use super::super::Color;

/// Surface elevation levels, lowest to highest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Elevation {
    #[default]
    Flat,
    Raised,
    Overlay,
    Modal,
}

/// One drop shadow: a pixel offset, a blur radius, a spread and a colour
/// (conventionally translucent black).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadow {
    pub offset: (f32, f32),
    pub blur: f32,
    pub spread: f32,
    pub color: Color,
}

/// The chrome for one elevation level: up to two stacked shadows plus an
/// optional hairline border.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElevationStyle {
    pub shadows: [Option<Shadow>; 2],
    pub border: Option<Color>,
}

/// The four-level elevation ramp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElevationScale {
    pub flat: ElevationStyle,
    pub raised: ElevationStyle,
    pub overlay: ElevationStyle,
    pub modal: ElevationStyle,
}

impl ElevationScale {
    /// The stock ramp for a dark or light UI. Shadow alphas are tuned per
    /// mode — light shadows need far less alpha on a white page.
    pub fn new(dark: bool, border: Color) -> Self {
        let shadow = |dy: f32, blur: f32, alpha: f32| {
            Some(Shadow {
                offset: (0.0, dy),
                blur,
                spread: 0.0,
                color: Color::rgba(0, 0, 0, (alpha * 255.0).round() as u8),
            })
        };
        let (raised, overlay, modal) = if dark {
            (
                [shadow(1.0, 3.0, 0.40), None],
                [shadow(4.0, 14.0, 0.45), shadow(1.0, 3.0, 0.30)],
                [shadow(12.0, 32.0, 0.55), shadow(2.0, 6.0, 0.30)],
            )
        } else {
            (
                [shadow(1.0, 3.0, 0.08), shadow(1.0, 2.0, 0.06)],
                [shadow(4.0, 14.0, 0.12), shadow(1.0, 3.0, 0.08)],
                [shadow(12.0, 32.0, 0.18), shadow(2.0, 6.0, 0.10)],
            )
        };
        Self {
            flat: ElevationStyle {
                shadows: [None; 2],
                border: Some(border),
            },
            raised: ElevationStyle {
                shadows: raised,
                border: Some(border),
            },
            overlay: ElevationStyle {
                shadows: overlay,
                border: Some(border),
            },
            modal: ElevationStyle {
                shadows: modal,
                border: Some(border),
            },
        }
    }

    pub fn get(&self, elevation: Elevation) -> &ElevationStyle {
        match elevation {
            Elevation::Flat => &self.flat,
            Elevation::Raised => &self.raised,
            Elevation::Overlay => &self.overlay,
            Elevation::Modal => &self.modal,
        }
    }
}
