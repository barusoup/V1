use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaMode {
    None,
    Fsr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpscalePreset {
    Native,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl UpscalePreset {
    pub fn scale(self) -> f32 {
        match self {
            UpscalePreset::Native => 1.0,
            UpscalePreset::Quality => 1.5,
            UpscalePreset::Balanced => 1.7,
            UpscalePreset::Performance => 2.0,
            UpscalePreset::UltraPerformance => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub aa: AaMode,
    pub upscale: UpscalePreset,
    pub sharpness: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            aa: AaMode::Fsr,
            upscale: UpscalePreset::Native,
            sharpness: 0.6,
        }
    }
}

impl Settings {
    pub fn file_path() -> PathBuf {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
            .join("settings.toml")
    }

    pub fn load() -> Self {
        let path = Self::file_path();
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        Self::parse(&text)
    }

    fn parse(text: &str) -> Self {
        let mut out = Self::default();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("aa") {
                if line.contains("none") {
                    out.aa = AaMode::None;
                } else if line.contains("fsr") {
                    out.aa = AaMode::Fsr;
                }
            } else if line.starts_with("upscale") {
                if line.contains("quality") {
                    out.upscale = UpscalePreset::Quality;
                } else if line.contains("balanced") {
                    out.upscale = UpscalePreset::Balanced;
                } else if line.contains("performance") && !line.contains("ultra") {
                    out.upscale = UpscalePreset::Performance;
                } else if line.contains("ultra") {
                    out.upscale = UpscalePreset::UltraPerformance;
                } else if line.contains("native") {
                    out.upscale = UpscalePreset::Native;
                }
            } else if line.starts_with("sharpness")
                && let Some(value) = line.split('=').nth(1)
                && let Ok(value) = value.trim().parse::<f32>()
            {
                out.sharpness = value.clamp(0.0, 1.0);
            }
        }
        out
    }

    pub fn render_scale(self) -> f32 {
        self.upscale.scale().recip()
    }
}
