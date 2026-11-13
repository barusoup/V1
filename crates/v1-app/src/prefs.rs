// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Preferences that outlive a run.
//!
//! Kept apart from the collection: these are choices about the app itself, not about the music in
//! it, and losing the file must never cost the listener a saved song. The file is small and read
//! once at startup; changes go through the same background writer as everything else on disk.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::store::Writer;

/// What the volume is on a first run, before the listener has chosen a level.
pub const DEFAULT_VOLUME: f32 = 0.8;
/// How soft a new background is, 0 to 1.
pub const DEFAULT_BLUR: f32 = 0.55;
/// How transparent the UI is over a background, 0 (opaque) to 1 (transparent).
pub const DEFAULT_TRANSPARENCY: f32 = 0.38;
/// The equalizer's ten ISO bands, in Hz: bass to treble.
pub const EQ_BANDS: [f32; 10] = [
    31.0, 62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16_000.0,
];
/// How a band's frequency is written on the Settings page.
pub fn eq_band_label(index: usize) -> String {
    match EQ_BANDS.get(index).copied() {
        Some(hz) if hz >= 1000.0 => format!("{}k", (hz / 1000.0) as u32),
        Some(hz) => format!("{hz:.0}"),
        None => String::new(),
    }
}
/// The equalizer's range each way.
pub const EQ_RANGE_DB: f32 = 12.0;

/// How the audio visualizer behind the lyrics page is coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualizerColorRec {
    /// Dynamically extracted from the currently playing song's album cover.
    #[default]
    AlbumArt,
    /// Uses the theme's accent colour.
    Accent,
    /// Uses a fixed custom colour (`visualizer_color`).
    Custom,
}

/// A theme the listener made: colours and a name, plus whether it is pinned to the top of
/// the manager. Colours are `0xRRGGBB`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThemeRec {
    pub name: String,
    pub primary: u32,
    pub secondary: u32,
    pub accent: u32,
    #[serde(default)]
    pub favorite: bool,
    #[serde(default)]
    pub lyrics_fill: Option<u32>,
    #[serde(default)]
    pub lyrics_outline: Option<u32>,
    #[serde(default)]
    pub visualizer_mode: VisualizerColorRec,
    #[serde(default)]
    pub visualizer_color: Option<u32>,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default = "default_blur")]
    pub background_blur: f32,
    #[serde(default = "default_transparency")]
    pub background_transparency: f32,
}

impl ThemeRec {
    pub fn new(name: impl Into<String>, primary: u32, secondary: u32, accent: u32) -> ThemeRec {
        ThemeRec {
            name: name.into(),
            primary,
            secondary,
            accent,
            favorite: false,
            lyrics_fill: None,
            lyrics_outline: None,
            visualizer_mode: VisualizerColorRec::AlbumArt,
            visualizer_color: None,
            background: None,
            background_blur: DEFAULT_BLUR,
            background_transparency: DEFAULT_TRANSPARENCY,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Stored {
    #[serde(default = "default_volume")]
    volume: f32,
    /// The background picture's path, if the listener chose one.
    #[serde(default)]
    background: Option<String>,
    /// Whether the global background overrides per-theme backgrounds.
    #[serde(default = "default_true")]
    background_enabled: bool,
    #[serde(default = "default_blur")]
    background_blur: f32,
    #[serde(default = "default_transparency")]
    background_transparency: f32,
    /// Whether saved songs are downloaded to this device. On unless the listener turned it off.
    #[serde(default = "default_true")]
    downloads: bool,
    /// Five band gains in dB. Missing (or short) lists fall back to flat.
    #[serde(default)]
    eq: Vec<f32>,
    #[serde(default)]
    themes: Vec<ThemeRec>,
    /// The name of the theme in force, or `None` for Default.
    #[serde(default)]
    active_theme: Option<String>,
}

impl Default for Stored {
    fn default() -> Stored {
        Stored {
            volume: DEFAULT_VOLUME,
            background: None,
            background_enabled: true,
            background_blur: DEFAULT_BLUR,
            background_transparency: DEFAULT_TRANSPARENCY,
            downloads: true,
            eq: Vec::new(),
            themes: Vec::new(),
            active_theme: None,
        }
    }
}

fn default_volume() -> f32 {
    DEFAULT_VOLUME
}

fn default_blur() -> f32 {
    DEFAULT_BLUR
}

fn default_transparency() -> f32 {
    DEFAULT_TRANSPARENCY
}

fn default_true() -> bool {
    true
}

/// The preferences, with the writer that saves them. Dropping it finishes any queued save.
pub struct Prefs {
    path: Option<PathBuf>,
    writer: Option<Writer>,
    stored: Stored,
}

impl Prefs {
    /// Read the preferences under `root`. A missing or unreadable file is simply the defaults: a
    /// preference is never worth stopping the app for.
    pub fn open(root: &Path) -> Prefs {
        let path = root.join("settings.json");
        let stored = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Stored>(&bytes).ok())
            .unwrap_or_default();
        let mut prefs = Prefs {
            path: Some(path),
            writer: Some(Writer::start()),
            stored,
        };
        prefs.stored.volume = prefs.stored.volume.clamp(0.0, 1.0);
        prefs.stored.background_blur = prefs.stored.background_blur.clamp(0.0, 1.0);
        prefs.stored.background_transparency = prefs.stored.background_transparency.clamp(0.0, 1.0);
        prefs.stored.eq = normalise_eq(&prefs.stored.eq).to_vec();
        prefs
            .stored
            .themes
            .retain(|theme| !theme.name.trim().is_empty());
        prefs
    }

    fn save(&self) {
        if let (Some(path), Some(writer)) = (&self.path, &self.writer) {
            if let Ok(bytes) = serde_json::to_vec(&self.stored) {
                writer.put(path.clone(), bytes);
            }
        }
    }

    /// The volume the listener left the app at.
    pub fn volume(&self) -> f32 {
        self.stored.volume
    }

    /// Remember a new volume. Repeated values, as a drag produces, do not touch the disk.
    pub fn set_volume(&mut self, volume: f32) {
        let volume = volume.clamp(0.0, 1.0);
        if (volume - self.stored.volume).abs() < 1e-4 {
            return;
        }
        self.stored.volume = volume;
        self.save();
    }

    /// The picture behind the app, if one was chosen.
    pub fn background(&self) -> Option<&Path> {
        self.stored.background.as_deref().map(Path::new)
    }

    pub fn background_enabled(&self) -> bool {
        self.stored.background_enabled
    }

    pub fn set_background_enabled(&mut self, enabled: bool) {
        if enabled == self.stored.background_enabled {
            return;
        }
        self.stored.background_enabled = enabled;
        self.save();
    }

    pub fn set_background(&mut self, path: Option<&Path>) {
        let value = path.map(|path| path.to_string_lossy().into_owned());
        if value == self.stored.background {
            return;
        }
        self.stored.background = value;
        self.save();
    }

    /// How soft the background is, 0 to 1.
    pub fn background_blur(&self) -> f32 {
        self.stored.background_blur
    }

    pub fn set_background_blur(&mut self, blur: f32) {
        let blur = blur.clamp(0.0, 1.0);
        if (blur - self.stored.background_blur).abs() < 1e-4 {
            return;
        }
        self.stored.background_blur = blur;
        self.save();
    }

    /// How transparent the UI is over the background, 0 to 1.
    pub fn background_transparency(&self) -> f32 {
        self.stored.background_transparency
    }

    pub fn set_background_transparency(&mut self, transparency: f32) {
        let transparency = transparency.clamp(0.0, 1.0);
        if (transparency - self.stored.background_transparency).abs() < 1e-4 {
            return;
        }
        self.stored.background_transparency = transparency;
        self.save();
    }

    /// Whether saved songs are kept on disk. On by default.
    pub fn downloads(&self) -> bool {
        self.stored.downloads
    }

    pub fn set_downloads(&mut self, enabled: bool) {
        if enabled == self.stored.downloads {
            return;
        }
        self.stored.downloads = enabled;
        self.save();
    }

    /// The equalizer's ten band gains, in dB.
    pub fn eq(&self) -> [f32; 10] {
        normalise_eq(&self.stored.eq)
    }

    pub fn set_eq(&mut self, gains: [f32; 10]) {
        let gains = normalise_eq(&gains);
        if self.stored.eq.len() == gains.len()
            && gains
                .iter()
                .zip(&self.stored.eq)
                .all(|(a, b)| (a - b).abs() < 1e-3)
        {
            return;
        }
        self.stored.eq = gains.to_vec();
        self.save();
    }

    /// The themes the listener made, in the order they were saved.
    pub fn themes(&self) -> &[ThemeRec] {
        &self.stored.themes
    }

    pub fn set_themes(&mut self, themes: &[ThemeRec]) {
        if themes == self.stored.themes.as_slice() {
            return;
        }
        self.stored.themes = themes.to_vec();
        self.save();
    }

    /// The name of the theme in force, or `None` for Default.
    pub fn active_theme(&self) -> Option<&str> {
        self.stored.active_theme.as_deref()
    }

    pub fn set_active_theme(&mut self, name: Option<&str>) {
        if name == self.stored.active_theme.as_deref() {
            return;
        }
        self.stored.active_theme = name.map(str::to_owned);
        self.save();
    }
}

/// Ten gains in dB, clamped to the equalizer's range. A missing or short list is flat.
fn normalise_eq(values: &[f32]) -> [f32; 10] {
    let mut gains = [0.0; 10];
    for (slot, value) in gains.iter_mut().zip(values) {
        *slot = if value.is_finite() {
            value.clamp(-EQ_RANGE_DB, EQ_RANGE_DB)
        } else {
            0.0
        };
    }
    gains
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder under the system temp directory, named for this process so tests cannot
    /// collide with one another.
    fn scratch(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root =
            std::env::temp_dir().join(format!("v1-prefs-{}-{nanos}-{name}", std::process::id()));
        std::fs::create_dir_all(&root).expect("create the test folder");
        root
    }

    #[test]
    fn the_volume_survives_a_restart() {
        let root = scratch("restart");
        let mut prefs = Prefs::open(&root);
        assert_eq!(prefs.volume(), DEFAULT_VOLUME);
        prefs.set_volume(0.35);
        drop(prefs);

        let prefs = Prefs::open(&root);
        assert!((prefs.volume() - 0.35).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_broken_file_is_just_the_defaults() {
        let root = scratch("broken");
        std::fs::write(root.join("settings.json"), b"not json").expect("write the broken file");
        let prefs = Prefs::open(&root);
        assert_eq!(prefs.volume(), DEFAULT_VOLUME);
        assert!(prefs.downloads(), "downloads are on by default");
        assert_eq!(prefs.eq(), [0.0; 10]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn themes_background_and_gains_survive_a_restart() {
        let root = scratch("themes");
        let mut prefs = Prefs::open(&root);
        prefs.set_background(Some(Path::new("C:/pictures/night.png")));
        prefs.set_background_blur(0.8);
        prefs.set_downloads(false);
        prefs.set_eq([6.0, -3.0, 0.0, 1.5, 9.0, 2.0, -1.5, 0.0, 3.0, 4.5]);
        let themes = [
            ThemeRec::new("Night", 0x1b2a4a, 0xc9d6ff, 0x7aa2ff),
            ThemeRec {
                favorite: true,
                ..ThemeRec::new("Rose", 0x2a0e18, 0xffd9e2, 0xff5d8f)
            },
        ];
        prefs.set_themes(&themes);
        prefs.set_active_theme(Some("Night"));
        drop(prefs);

        let prefs = Prefs::open(&root);
        assert_eq!(prefs.background(), Some(Path::new("C:/pictures/night.png")));
        assert!((prefs.background_blur() - 0.8).abs() < 1e-6);
        assert!(!prefs.downloads());
        assert_eq!(
            prefs.eq(),
            [6.0, -3.0, 0.0, 1.5, 9.0, 2.0, -1.5, 0.0, 3.0, 4.5]
        );
        assert_eq!(prefs.themes(), themes);
        assert_eq!(prefs.active_theme(), Some("Night"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn equalizer_values_are_clamped_and_padded() {
        assert_eq!(normalise_eq(&[]), [0.0; 10]);
        assert_eq!(
            normalise_eq(&[40.0]),
            [EQ_RANGE_DB, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            normalise_eq(&[f32::NAN, -40.0]),
            [0.0, -EQ_RANGE_DB, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn legacy_theme_json_deserializes_with_defaults() {
        let json = r#"{"name":"Old","primary":1779274,"secondary":13227775,"accent":8037119}"#;
        let rec: ThemeRec = serde_json::from_str(json).expect("deserialize legacy theme");
        assert_eq!(rec.name, "Old");
        assert_eq!(rec.lyrics_fill, None);
        assert_eq!(rec.lyrics_outline, None);
        assert_eq!(rec.visualizer_mode, VisualizerColorRec::AlbumArt);
        assert_eq!(rec.visualizer_color, None);
    }
}
