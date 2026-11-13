// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Saved artists.
//!
//! An artist is not downloaded: there is no audio to keep. What is kept is the page YouTube Music
//! would otherwise have to be asked for again, written to disk the moment the artist is saved, so
//! opening a saved artist is instant and works offline.
//!
//! The vault is built to stay cheap with hundreds of artists saved:
//!
//! - The **index** (name, picture, dates) is small and always in memory. The Artists list is drawn
//!   from it alone, and never loads a page.
//! - A page lives in its own file and is read only when the artist is opened. The **warm set**
//!   keeps the last [`WARM_MAX`] used in memory, so going back and forth between a few artists
//!   never touches the disk, and the rest are cold until asked for.
//! - Pages go **stale**. A few are quietly refreshed on each launch, the oldest first, so the whole
//!   vault stays current over time without a burst of requests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use yt_client::ArtistPage;

use crate::store::{now_secs, safe_id, Writer};

/// How many artist pages are held in memory at once.
pub const WARM_MAX: usize = 24;
/// How many artists are read ahead at launch: the ones saved or refreshed most recently.
pub const PREWARM: usize = 8;
/// A page older than this is due a refresh.
const STALE_SECS: u64 = 14 * 24 * 60 * 60;
/// How many stale pages are refreshed per launch.
pub const REFRESH_PER_LAUNCH: usize = 3;

/// What the list needs to know about a saved artist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtistEntry {
    pub id: String,
    pub name: String,
    pub thumbnail: Option<String>,
    pub saved_at: u64,
    pub fetched_at: u64,
}

#[derive(Serialize, Deserialize)]
struct Stored {
    page: ArtistPage,
}

struct Warm {
    page: ArtistPage,
    used: u64,
}

pub struct ArtistVault {
    /// Where pages live. `None` keeps everything in memory (for `--shot`).
    dir: Option<PathBuf>,
    entries: HashMap<String, ArtistEntry>,
    warm: HashMap<String, Warm>,
    /// Counts uses, so the least recently used page is the lowest.
    clock: u64,
    /// The entries, newest saved first. Rebuilt when they change.
    order: Vec<ArtistEntry>,
    order_dirty: bool,
}

impl ArtistVault {
    pub fn in_memory() -> ArtistVault {
        ArtistVault {
            dir: None,
            entries: HashMap::new(),
            warm: HashMap::new(),
            clock: 0,
            order: Vec::new(),
            order_dirty: true,
        }
    }

    /// Open the vault in `dir`: read the index, drop files that belong to no saved artist, and read
    /// the most recent few pages ahead of need.
    pub fn open(dir: PathBuf) -> ArtistVault {
        let mut vault = ArtistVault::in_memory();
        let index: Vec<ArtistEntry> = std::fs::read(dir.join("index.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        for entry in index {
            if safe_id(&entry.id) {
                vault.entries.insert(entry.id.clone(), entry);
            }
        }
        if let Ok(files) = std::fs::read_dir(&dir) {
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                let keep = name == "index.json"
                    || name
                        .strip_suffix(".json")
                        .is_some_and(|id| vault.entries.contains_key(id));
                if !keep {
                    let _ = std::fs::remove_file(file.path());
                }
            }
        }
        vault.dir = Some(dir);
        let recent: Vec<String> = vault
            .list()
            .iter()
            .take(PREWARM)
            .map(|e| e.id.clone())
            .collect();
        for id in recent {
            vault.get(&id);
        }
        vault
    }

    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn ids(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Every saved artist, newest saved first.
    pub fn list(&mut self) -> &[ArtistEntry] {
        if self.order_dirty {
            self.order = self.entries.values().cloned().collect();
            self.order.sort_by(|a, b| {
                b.saved_at
                    .cmp(&a.saved_at)
                    .then_with(|| a.name.cmp(&b.name))
            });
            self.order_dirty = false;
        }
        &self.order
    }

    fn page_path(dir: &Path, id: &str) -> PathBuf {
        dir.join(format!("{id}.json"))
    }

    fn write_index(&self, writer: Option<&Writer>) {
        let (Some(dir), Some(writer)) = (&self.dir, writer) else {
            return;
        };
        let mut entries: Vec<&ArtistEntry> = self.entries.values().collect();
        entries.sort_by(|a, b| a.id.cmp(&b.id));
        if let Ok(bytes) = serde_json::to_vec(&entries) {
            writer.put(dir.join("index.json"), bytes);
        }
    }

    fn write_page(&self, page: &ArtistPage, writer: Option<&Writer>) {
        let (Some(dir), Some(writer)) = (&self.dir, writer) else {
            return;
        };
        let stored = Stored { page: page.clone() };
        if let Ok(bytes) = serde_json::to_vec(&stored) {
            writer.put(Self::page_path(dir, &page.browse_id), bytes);
        }
    }

    /// Save an artist: the index entry, the page on disk, and the page kept warm. Returns the
    /// picture's URL, for the caller to keep.
    pub fn save(&mut self, page: &ArtistPage, writer: Option<&Writer>) -> Option<String> {
        if !safe_id(&page.browse_id) {
            return None;
        }
        let now = now_secs();
        self.entries.insert(
            page.browse_id.clone(),
            ArtistEntry {
                id: page.browse_id.clone(),
                name: page.name.clone(),
                thumbnail: page.thumbnail.clone(),
                saved_at: now,
                fetched_at: now,
            },
        );
        self.order_dirty = true;
        self.write_page(page, writer);
        self.write_index(writer);
        self.warm_page(page.clone());
        page.thumbnail.clone()
    }

    /// A fresh copy of a saved artist's page has arrived: keep it in place of the old one.
    pub fn refresh(&mut self, page: &ArtistPage, writer: Option<&Writer>) {
        let Some(entry) = self.entries.get_mut(&page.browse_id) else {
            return;
        };
        entry.fetched_at = now_secs();
        if !page.name.is_empty() {
            entry.name = page.name.clone();
        }
        if page.thumbnail.is_some() {
            entry.thumbnail = page.thumbnail.clone();
        }
        self.order_dirty = true;
        self.write_page(page, writer);
        self.write_index(writer);
        self.warm_page(page.clone());
    }

    pub fn remove(&mut self, id: &str, writer: Option<&Writer>) {
        if self.entries.remove(id).is_none() {
            return;
        }
        self.warm.remove(id);
        self.order_dirty = true;
        if let (Some(dir), Some(writer)) = (&self.dir, writer) {
            writer.remove(Self::page_path(dir, id));
        }
        self.write_index(writer);
    }

    /// A saved artist's page, from memory if it is warm, else from disk (and then warm).
    pub fn get(&mut self, id: &str) -> Option<ArtistPage> {
        if !self.entries.contains_key(id) {
            return None;
        }
        self.clock += 1;
        if let Some(warm) = self.warm.get_mut(id) {
            warm.used = self.clock;
            return Some(warm.page.clone());
        }
        let dir = self.dir.as_ref()?;
        let bytes = std::fs::read(Self::page_path(dir, id)).ok()?;
        let stored: Stored = serde_json::from_slice(&bytes).ok()?;
        self.warm_page(stored.page.clone());
        Some(stored.page)
    }

    /// Hold `page` in memory, pushing out the least recently used one when the set is full. A vault
    /// with no disk behind it keeps everything: there would be nothing to read back.
    fn warm_page(&mut self, page: ArtistPage) {
        self.clock += 1;
        self.warm.insert(
            page.browse_id.clone(),
            Warm {
                page,
                used: self.clock,
            },
        );
        while self.dir.is_some() && self.warm.len() > WARM_MAX {
            let coldest = self
                .warm
                .iter()
                .min_by_key(|(_, w)| w.used)
                .map(|(id, _)| id.clone());
            match coldest {
                Some(id) => {
                    self.warm.remove(&id);
                }
                None => break,
            }
        }
    }

    /// The artists whose pages are oldest past the stale mark, at most `limit`, oldest first.
    pub fn stale(&self, limit: usize) -> Vec<String> {
        let now = now_secs();
        let mut due: Vec<&ArtistEntry> = self
            .entries
            .values()
            .filter(|e| now.saturating_sub(e.fetched_at) > STALE_SECS)
            .collect();
        due.sort_by_key(|e| e.fetched_at);
        due.into_iter().take(limit).map(|e| e.id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(id: &str) -> ArtistPage {
        ArtistPage {
            browse_id: id.to_owned(),
            name: format!("Artist {id}"),
            ..ArtistPage::default()
        }
    }

    #[test]
    fn only_a_few_pages_stay_warm_and_the_rest_come_back_from_disk() {
        let dir = std::env::temp_dir().join(format!("v1-vault-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let writer = Writer::start();
            let mut vault = ArtistVault::open(dir.clone());
            for n in 0..(WARM_MAX + 10) {
                vault.save(&page(&format!("UC{n}")), Some(&writer));
            }
            assert!(vault.warm.len() <= WARM_MAX, "warm set is bounded");
            assert_eq!(vault.len(), WARM_MAX + 10);
            drop(writer); // finishes the queue
        }
        let mut vault = ArtistVault::open(dir.clone());
        assert_eq!(vault.len(), WARM_MAX + 10, "the index survived");
        assert!(
            vault.warm.len() <= PREWARM,
            "only the newest few are read ahead"
        );
        let cold = vault
            .get("UC0")
            .expect("a cold page is read back from disk");
        assert_eq!(cold.name, "Artist UC0");
        assert!(vault.get("UCnone").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn removing_an_artist_forgets_it() {
        let mut vault = ArtistVault::in_memory();
        vault.save(&page("UCa"), None);
        assert!(vault.contains("UCa"));
        vault.remove("UCa", None);
        assert!(!vault.contains("UCa") && vault.get("UCa").is_none());
        assert!(vault.list().is_empty());
    }
}
