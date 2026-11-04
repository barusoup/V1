//! Type scale: named text roles with a size, line height and weight each.
//!
//! This phase carries data only — font resolution (which family and weight
//! a [`FontWeight`] maps to) is wired in a later phase.

/// Font weight a [`TypeStyle`] asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FontWeight {
    #[default]
    Regular,
    Medium,
    SemiBold,
}

/// A named slot in the type scale. Roles are semantic — pick the role,
/// not the pixel size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TypeRole {
    Caption,
    Small,
    #[default]
    Body,
    BodyLg,
    Label,
    Heading,
    Title,
    Display,
    Mono,
}

/// One resolved type style. `line_height` is in pixels (not an em ratio)
/// so rows and controls can share it; `tracking` is letterspacing in px.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TypeStyle {
    pub weight: FontWeight,
    pub size: f32,
    pub line_height: f32,
    pub tracking: f32,
    pub mono: bool,
}

/// The nine-role type scale. Index by [`TypeRole`]; `set` overwrites one
/// slot so a theme can restyle a role without rebuilding the table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TypeScale {
    roles: [TypeStyle; 9],
}

impl Default for TypeScale {
    fn default() -> Self {
        let style = |size: f32, line_height: f32, weight: FontWeight, mono: bool| TypeStyle {
            weight,
            size,
            line_height,
            tracking: 0.0,
            mono,
        };
        Self {
            roles: [
                style(11.0, 14.0, FontWeight::Regular, false), // Caption
                style(12.0, 16.0, FontWeight::Regular, false), // Small
                style(13.0, 18.0, FontWeight::Regular, false), // Body
                style(15.0, 22.0, FontWeight::Regular, false), // BodyLg
                style(13.0, 18.0, FontWeight::Medium, false),  // Label
                style(15.0, 20.0, FontWeight::SemiBold, false), // Heading
                style(18.0, 24.0, FontWeight::SemiBold, false), // Title
                style(24.0, 30.0, FontWeight::SemiBold, false), // Display
                style(12.0, 16.0, FontWeight::Regular, true),  // Mono
            ],
        }
    }
}

impl TypeScale {
    pub fn get(&self, role: TypeRole) -> TypeStyle {
        self.roles[role as usize]
    }

    pub fn set(&mut self, role: TypeRole, style: TypeStyle) {
        self.roles[role as usize] = style;
    }
}
