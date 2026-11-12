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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Stored {
    #[serde(default = "default_volume")]
    volume: f32,
}

impl Default for Stored {
    fn default() -> Stored {
        Stored {
            volume: DEFAULT_VOLUME,
        }
    }
}

fn default_volume() -> f32 {
    DEFAULT_VOLUME
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
        Prefs {
            path: Some(path),
            writer: Some(Writer::start()),
            stored: Stored {
                volume: stored.volume.clamp(0.0, 1.0),
            },
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
        if let (Some(path), Some(writer)) = (&self.path, &self.writer) {
            if let Ok(bytes) = serde_json::to_vec(&self.stored) {
                writer.put(path.clone(), bytes);
            }
        }
    }
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
        let _ = std::fs::remove_dir_all(&root);
    }
}
