//! Face-level metrics parsed once at load: the OS/2 heights optical
//! centring needs. Both raster backends read the same bytes, so the
//! numbers are identical whether glyphs come from `fontdue` or CoreText.

/// The face's `head`/`OS/2` numbers, in font units.
pub(crate) struct FaceMetrics {
    units_per_em: f32,
    cap_height: Option<f32>,
    x_height: Option<f32>,
}

impl FaceMetrics {
    /// Parse `data` (the same bytes the raster backend loads). A face that
    /// does not parse gets the fallback-emitted numbers (1 upem, no OS/2
    /// heights) rather than failing — the backend reports parse errors.
    pub fn parse(data: &[u8]) -> Self {
        let face = ttf_parser::Face::parse(data, 0);
        let (units, cap, x) = face
            .map(|f| (f.units_per_em(), f.capital_height(), f.x_height()))
            .unwrap_or((1000, None, None));
        Self {
            units_per_em: f32::from(units.max(1)),
            cap_height: cap.map(f32::from),
            x_height: x.map(f32::from),
        }
    }

    fn scale(&self, units: Option<f32>, px: f32, fallback: f32) -> f32 {
        match units {
            Some(v) if v > 0.0 => v / self.units_per_em * px,
            _ => fallback,
        }
    }

    /// Cap height in pixels (flat-top capital letter height); 0.7 em when
    /// the face does not declare one.
    pub fn cap_height(&self, px: f32) -> f32 {
        self.scale(self.cap_height, px, 0.7 * px)
    }

    /// x-height in pixels; 0.5 em when the face does not declare one.
    pub fn x_height(&self, px: f32) -> f32 {
        self.scale(self.x_height, px, 0.5 * px)
    }
}
