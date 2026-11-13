// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Remembering resolved stream URLs, across calls and across launches.
//!
//! A cache hit skips the whole `/player` chain. The entry carries everything the caller needs
//! (loudness, the play-tracking ping, the headers), so a hit behaves exactly like a fresh resolve.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::resolve::Resolved;

/// Where resolved URLs are kept.
pub trait UrlCache: Send + Sync {
    /// The entry for `video_id`, if it exists. The caller checks expiry against its own clock.
    fn get(&self, video_id: &str) -> Option<Resolved>;
    fn put(&self, entry: Resolved);
    fn remove(&self, video_id: &str);
    /// Drop entries that have expired at `now_unix`.
    fn sweep(&self, now_unix: u64);
}

/// A cache that lives only as long as the process.
#[derive(Default)]
pub struct MemoryCache {
    map: Mutex<HashMap<String, Resolved>>,
}

impl UrlCache for MemoryCache {
    fn get(&self, video_id: &str) -> Option<Resolved> {
        self.map.lock().unwrap().get(video_id).cloned()
    }

    fn put(&self, entry: Resolved) {
        self.map
            .lock()
            .unwrap()
            .insert(entry.video_id.clone(), entry);
    }

    fn remove(&self, video_id: &str) {
        self.map.lock().unwrap().remove(video_id);
    }

    fn sweep(&self, now_unix: u64) {
        self.map
            .lock()
            .unwrap()
            .retain(|_, e| e.expires_at_unix > now_unix);
    }
}

#[derive(Serialize, Deserialize, Default)]
struct OnDisk {
    entries: Vec<Resolved>,
}

/// The most entries a [`FileCache`] keeps. Past it, the ones nearest to expiry go first.
pub const FILE_CACHE_MAX: usize = 256;

/// A cache kept in a JSON file, so a replay after a restart skips the network.
///
/// The whole file is rewritten on each change, so it is kept small: expired entries are dropped when
/// the file is opened and whenever an entry is added, and the count is capped at
/// [`FILE_CACHE_MAX`]. The write goes to a temp file first so a crash cannot leave half a file. It
/// is not fsynced: the cache holds only what can be fetched again, and an fsync on every resolve
/// would stall the async worker thread that made it.
pub struct FileCache {
    path: PathBuf,
    map: Mutex<HashMap<String, Resolved>>,
}

impl FileCache {
    /// Open (or create) the cache at `path`. A missing or unreadable file starts empty: the cache
    /// holds only data that can be fetched again.
    pub fn open(path: impl AsRef<Path>) -> FileCache {
        let path = path.as_ref().to_path_buf();
        let loaded: OnDisk = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        let now = crate::resolve::now_unix();
        let map: HashMap<String, Resolved> = loaded
            .entries
            .into_iter()
            .filter(|e| e.expires_at_unix > now)
            .map(|e| (e.video_id.clone(), e))
            .collect();
        FileCache {
            path,
            map: Mutex::new(map),
        }
    }

    fn flush(&self, map: &HashMap<String, Resolved>) {
        let doc = OnDisk {
            entries: map.values().cloned().collect(),
        };
        let Ok(bytes) = serde_json::to_vec(&doc) else {
            return;
        };
        if let Some(dir) = self.path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("json.tmp");
        let written = fs::File::create(&tmp)
            .and_then(|mut f| f.write_all(&bytes))
            .and_then(|_| fs::rename(&tmp, &self.path));
        if let Err(e) = written {
            tracing::warn!(error = %e, "could not write the stream cache");
        }
    }
}

impl UrlCache for FileCache {
    fn get(&self, video_id: &str) -> Option<Resolved> {
        self.map.lock().unwrap().get(video_id).cloned()
    }

    fn put(&self, entry: Resolved) {
        let mut map = self.map.lock().unwrap();
        let now = crate::resolve::now_unix();
        map.retain(|_, e| e.expires_at_unix > now);
        map.insert(entry.video_id.clone(), entry);
        // Over the cap: the entries closest to expiring are the least valuable.
        while map.len() > FILE_CACHE_MAX {
            let Some(oldest) = map
                .iter()
                .min_by_key(|(_, e)| e.expires_at_unix)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            map.remove(&oldest);
        }
        self.flush(&map);
    }

    fn remove(&self, video_id: &str) {
        let mut map = self.map.lock().unwrap();
        if map.remove(video_id).is_some() {
            self.flush(&map);
        }
    }

    fn sweep(&self, now_unix: u64) {
        let mut map = self.map.lock().unwrap();
        let before = map.len();
        map.retain(|_, e| e.expires_at_unix > now_unix);
        if map.len() != before {
            self.flush(&map);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yt_client::ClientKind;

    fn entry(id: &str, expires_at_unix: u64) -> Resolved {
        Resolved {
            video_id: id.to_owned(),
            url: format!("https://example.test/{id}"),
            headers: Vec::new(),
            client: ClientKind::Ios,
            itag: 140,
            mime_type: "audio/mp4".to_owned(),
            content_length: None,
            expires_at_unix,
            loudness_db: None,
            tracking_ping: None,
            title: None,
            author: None,
            length_seconds: None,
            thumbnail: None,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("yt-stream-cache-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("streams.json")
    }

    #[test]
    fn expired_entries_are_dropped_when_the_file_is_opened() {
        let path = scratch("open");
        let now = crate::resolve::now_unix();
        let cache = FileCache::open(&path);
        cache.put(entry("alive", now + 3600));
        // An entry that was alive when written but has expired by the next launch.
        {
            let mut map = cache.map.lock().unwrap();
            map.insert("stale".to_owned(), entry("stale", now.saturating_sub(10)));
            cache.flush(&map);
        }
        let reopened = FileCache::open(&path);
        assert!(reopened.get("alive").is_some());
        assert!(reopened.get("stale").is_none());
    }

    #[test]
    fn a_put_sweeps_what_has_expired_and_caps_the_count() {
        let path = scratch("cap");
        let now = crate::resolve::now_unix();
        let cache = FileCache::open(&path);
        {
            let mut map = cache.map.lock().unwrap();
            map.insert("dead".to_owned(), entry("dead", now.saturating_sub(10)));
        }
        for n in 0..(FILE_CACHE_MAX + 10) {
            cache.put(entry(&format!("song{n}"), now + 1000 + n as u64));
        }
        let map = cache.map.lock().unwrap();
        assert!(!map.contains_key("dead"));
        assert_eq!(map.len(), FILE_CACHE_MAX);
        // The ones nearest to expiry went first.
        assert!(!map.contains_key("song0"));
        assert!(map.contains_key(&format!("song{}", FILE_CACHE_MAX + 9)));
    }
}
