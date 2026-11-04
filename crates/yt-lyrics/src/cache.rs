// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Remembering lookups across plays and launches. One small file per video id, so writing one song
//! never rewrites the others.
//!
//! What is stored: a found lyric for as long as the file exists, and a "not found" for a day. Only
//! lookups made with a known length are written (see `Service::lookup`), since a fuzzy match
//! without one can be a different cut of the song. Errors are never written, so an offline run
//! does not turn into a day of "no lyrics".

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::model::Lyrics;

/// How long a "not found" holds before the providers are asked again.
pub const MISS_TTL_SECS: u64 = 24 * 3600;

#[derive(Serialize, Deserialize)]
struct Entry {
    stored_at_unix: u64,
    /// `None` is a stored "not found".
    lyrics: Option<Lyrics>,
}

#[derive(Clone)]
pub struct Cache {
    dir: PathBuf,
}

impl Cache {
    pub fn new(dir: impl Into<PathBuf>) -> Cache {
        Cache { dir: dir.into() }
    }

    /// `Some(Some(lyrics))` found, `Some(None)` a fresh "not found", `None` nothing usable.
    pub fn get(&self, video_id: &str, now_unix: u64) -> Option<Option<Lyrics>> {
        let text = fs::read_to_string(self.path(video_id)?).ok()?;
        let entry: Entry = serde_json::from_str(&text).ok()?;
        match entry.lyrics {
            Some(lyrics) => Some(Some(lyrics)),
            None if now_unix.saturating_sub(entry.stored_at_unix) < MISS_TTL_SECS => Some(None),
            None => None,
        }
    }

    /// Store an outcome. Failures are only logged: the cache is an optimisation, never a reason to
    /// fail a lookup.
    pub fn put(&self, video_id: &str, lyrics: Option<&Lyrics>, now_unix: u64) {
        let Some(path) = self.path(video_id) else {
            return;
        };
        let entry = Entry {
            stored_at_unix: now_unix,
            lyrics: lyrics.cloned(),
        };
        if let Err(error) = write_atomically(&path, &entry) {
            tracing::debug!(%error, "lyrics: cache write failed");
        }
    }

    /// The file for a video id. Only the characters YouTube ids use are allowed, so an id can never
    /// name a path outside the cache directory.
    fn path(&self, video_id: &str) -> Option<PathBuf> {
        let valid = !video_id.is_empty()
            && video_id.len() <= 32
            && video_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        valid.then(|| self.dir.join(format!("{video_id}.json")))
    }
}

/// Write to a temp file, then rename over the target, so a crash mid-write leaves the old file.
fn write_atomically(path: &Path, entry: &Entry) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    {
        let mut file = fs::File::create(&tmp)?;
        let bytes = serde_json::to_vec(entry).map_err(std::io::Error::other)?;
        // Not fsynced: a lyric is fetched again if a crash loses it, and an fsync per lookup is a
        // pause for no gain.
        file.write_all(&bytes)?;
    }
    fs::rename(&tmp, path)
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Body, Source, TimedLine};

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("yt-lyrics-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn sample() -> Lyrics {
        Lyrics {
            source: Source::Boidu,
            body: Body::Synced(vec![TimedLine {
                time_ms: 1000,
                text: "hi".into(),
            }]),
        }
    }

    #[test]
    fn found_lyrics_round_trip_and_never_expire() {
        let cache = Cache::new(scratch("found"));
        cache.put("abcDEF_123-", Some(&sample()), 1_000);
        assert_eq!(
            cache.get("abcDEF_123-", 1_000 + 365 * 24 * 3600),
            Some(Some(sample()))
        );
    }

    #[test]
    fn a_miss_is_remembered_for_a_day_then_forgotten() {
        let cache = Cache::new(scratch("miss"));
        cache.put("abcDEF_123-", None, 1_000);
        assert_eq!(cache.get("abcDEF_123-", 1_000 + 60), Some(None));
        assert_eq!(cache.get("abcDEF_123-", 1_000 + MISS_TTL_SECS), None);
    }

    #[test]
    fn unknown_or_unsafe_ids_are_not_cached() {
        let cache = Cache::new(scratch("ids"));
        cache.put("../escape", Some(&sample()), 1);
        cache.put("", Some(&sample()), 1);
        assert_eq!(cache.get("../escape", 1), None);
        assert_eq!(cache.get("missing_id", 1), None);
    }

    #[test]
    fn a_corrupt_file_reads_as_nothing_cached() {
        let dir = scratch("corrupt");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("zzzzzzzzzzz.json"), "{not json").unwrap();
        assert_eq!(Cache::new(&dir).get("zzzzzzzzzzz", 1), None);
    }
}
