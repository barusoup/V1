// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The collection: the songs, albums and artists the listener saved, and the files behind them.
//!
//! - **Songs and albums are downloaded**, quietly, as 128 kbps Opus under `downloads/`. Saving an
//!   album downloads every song on it, but does not like those songs: they are kept for the album
//!   alone and stay out of the Songs list. A song is liked only when saved on its own. A song held
//!   by several things (liked, and on one or more saved albums) is one file, and goes only when the
//!   last of them lets go: unliking a song keeps it while an album still holds it, and unliking an
//!   album keeps the songs that are liked.
//! - **Unsaving is forgiving.** The song leaves the collection at once, but its file waits a minute
//!   in limbo: saved again inside that minute, it simply comes back, with no second download. The
//!   wait is invisible, and closing the app skips it. If the app dies first, the next launch sweeps
//!   any file that no saved song owns.
//! - **Artists** keep their page metadata instead of audio; see [`crate::vault`].
//!
//! Nothing here touches the network or the disk on the UI thread: it decides, and hands back
//! [`Effects`] (downloads to start, covers to keep) for the caller to carry out.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use yt_client::{AlbumPage, ArtistPage, SongHit};

use crate::store::{cover_key, now_secs, safe_id, Writer};
use crate::vault::ArtistVault;

/// How long an unsaved song's file is kept in case it is saved again.
pub const GRACE: Duration = Duration::from_secs(60);
/// A failed download is tried again after this long, doubling each time up to [`RETRY_MAX`].
const RETRY_FIRST: Duration = Duration::from_secs(120);
const RETRY_MAX: Duration = Duration::from_secs(3600);

/// A download for the backend to make.
#[derive(Debug, Clone)]
pub struct Job {
    pub video_id: String,
    pub title: String,
    pub artist: String,
    pub album: Option<String>,
}

/// What a change to the collection asks the caller to do.
#[derive(Debug, Default)]
pub struct Effects {
    pub downloads: Vec<Job>,
    /// Cover URLs to keep on disk, so the saved item looks right offline.
    pub covers: Vec<String>,
}

impl Effects {
    fn merge(&mut self, other: Effects) {
        self.downloads.extend(other.downloads);
        self.covers.extend(other.covers);
    }
}

/// Which ids are saved, cheap to share with the frame being built.
#[derive(Debug, Default, Clone)]
pub struct SavedIds {
    pub songs: HashSet<String>,
    pub albums: HashSet<String>,
    pub artists: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongRec {
    pub hit: SongHit,
    pub saved_at: u64,
    /// Saved on its own.
    pub direct: bool,
    /// Saved because these albums are.
    pub albums: Vec<String>,
    /// The Opus file is complete.
    pub downloaded: bool,
    pub loudness_db: Option<f64>,
    pub duration_secs: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlbumRec {
    pub page: AlbumPage,
    pub saved_at: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct Index {
    #[serde(default)]
    songs: Vec<SongRec>,
    #[serde(default)]
    albums: Vec<AlbumRec>,
}

/// A saved song's file, as the player needs it.
#[derive(Debug, Clone)]
pub struct Local {
    pub path: PathBuf,
    pub loudness_db: Option<f64>,
    pub duration_secs: Option<f64>,
}

pub struct Library {
    /// The app's data folder. Empty for an in-memory library.
    root: PathBuf,
    writer: Option<Writer>,
    songs: HashMap<String, SongRec>,
    albums: HashMap<String, AlbumRec>,
    /// Unsaved songs whose files are waiting out [`GRACE`], with when they go.
    limbo: HashMap<String, (SongRec, Instant)>,
    pub vault: ArtistVault,
    ids: Arc<SavedIds>,
    /// Songs with a download under way.
    inflight: HashSet<String>,
    /// Whether saved songs are kept on this device. Off means no new downloads, and the files
    /// already here were removed when it was turned off.
    downloading: bool,
    /// Failed downloads: how many times, and when to try again.
    retry: HashMap<String, (u32, Instant)>,
    /// Bumped on every change, so cached orderings know when to rebuild.
    version: u64,
    song_order: Vec<String>,
    song_order_key: Option<(u64, usize)>,
    /// The saved albums' ids, newest first, valid for the `version` it was built at.
    album_order: Vec<String>,
    album_order_key: Option<u64>,
}

impl Library {
    /// A library with nothing on disk behind it, for `--shot`.
    pub fn in_memory() -> Library {
        Library {
            root: PathBuf::new(),
            writer: None,
            songs: HashMap::new(),
            albums: HashMap::new(),
            limbo: HashMap::new(),
            vault: ArtistVault::in_memory(),
            ids: Arc::new(SavedIds::default()),
            inflight: HashSet::new(),
            downloading: true,
            retry: HashMap::new(),
            version: 0,
            song_order: Vec::new(),
            song_order_key: None,
            album_order: Vec::new(),
            album_order_key: None,
        }
    }

    /// Open the collection stored under `root`. Anything on disk that no saved item owns (the files
    /// of songs unsaved when the app last stopped, half-made downloads) is cleared away.
    pub fn open(root: PathBuf) -> Library {
        let mut library = Library::in_memory();
        let index_path = root.join("library.json");
        let (index, readable): (Index, bool) = match std::fs::read(&index_path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(index) => (index, true),
                Err(_) => {
                    // Keep an unreadable index for inspection rather than overwrite it, and do not
                    // sweep the disk against an index that says nothing: the downloads are the
                    // listener's, and are only cleared when we know what owns them.
                    let _ = std::fs::rename(&index_path, root.join("library.json.bad"));
                    (Index::default(), false)
                }
            },
            Err(_) => (Index::default(), true),
        };
        library.songs = index
            .songs
            .into_iter()
            .filter(|rec| safe_id(&rec.hit.video_id))
            .map(|mut rec| {
                // A song saved from a list that gave no length still knows it from its file.
                if rec.hit.duration.is_none() {
                    rec.hit.duration = rec.duration_secs.map(length_text);
                }
                (rec.hit.video_id.clone(), rec)
            })
            .collect();
        library.albums = index
            .albums
            .into_iter()
            .map(|rec| (rec.page.browse_id.clone(), rec))
            .collect();
        library.root = root.clone();
        if readable {
            library.sweep_disk();
        }
        library.vault = ArtistVault::open(root.join("artists"));
        library.writer = Some(Writer::start());
        library.refresh_ids();
        library
    }

    fn downloads_dir(&self) -> PathBuf {
        self.root.join("downloads")
    }

    fn song_path(&self, id: &str) -> PathBuf {
        self.downloads_dir().join(format!("{id}.opus"))
    }

    /// Startup housekeeping: drop files no saved song owns, forget downloads whose file is gone,
    /// and drop kept covers nothing refers to.
    fn sweep_disk(&mut self) {
        // One listing answers both questions below (which files are unowned, and which saved songs
        // have lost theirs), where a stat per saved song would cost a system call each.
        let mut present: HashSet<String> = HashSet::new();
        if let Ok(files) = std::fs::read_dir(self.downloads_dir()) {
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                let owned = name
                    .strip_suffix(".opus")
                    .and_then(|id| self.songs.get(id))
                    .is_some_and(|rec| rec.downloaded);
                if owned {
                    present.insert(name);
                } else {
                    let _ = std::fs::remove_file(file.path());
                }
            }
        }
        let missing: Vec<String> = self
            .songs
            .values()
            .filter(|rec| {
                rec.downloaded && !present.contains(&format!("{}.opus", rec.hit.video_id))
            })
            .map(|rec| rec.hit.video_id.clone())
            .collect();
        for id in missing {
            if let Some(rec) = self.songs.get_mut(&id) {
                rec.downloaded = false;
            }
        }
        // Covers: keep those the collection uses.
        let mut used: HashSet<String> = HashSet::new();
        for rec in self.songs.values() {
            used.extend(rec.hit.thumbnail.iter().map(|u| cover_key(u)));
        }
        for rec in self.albums.values() {
            used.extend(rec.page.thumbnail.iter().map(|u| cover_key(u)));
        }
        // The vault is opened after this, so read its index directly.
        if let Ok(bytes) = std::fs::read(self.root.join("artists").join("index.json")) {
            if let Ok(entries) = serde_json::from_slice::<Vec<crate::vault::ArtistEntry>>(&bytes) {
                for entry in entries {
                    used.extend(entry.thumbnail.iter().map(|u| cover_key(u)));
                }
            }
        }
        if let Ok(files) = std::fs::read_dir(self.root.join("covers")) {
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                if !used.contains(&name) {
                    let _ = std::fs::remove_file(file.path());
                }
            }
        }
    }

    // ---- Reading ------------------------------------------------------------------------------

    /// Which ids are saved. Cloning this is cheap; it changes only when the collection does.
    pub fn ids(&self) -> Arc<SavedIds> {
        self.ids.clone()
    }

    /// Whether the listener has liked the song itself. A song held only by a saved album is downloaded
    /// but not liked.
    pub fn song_saved(&self, id: &str) -> bool {
        self.songs.get(id).is_some_and(|rec| rec.direct)
    }

    pub fn album_count(&self) -> usize {
        self.albums.len()
    }

    /// The liked songs, newest first, plus any in `ghosts` (songs just unliked from the list on
    /// screen, which stay put until the reader leaves it). Songs held only by albums are left out.
    pub fn song_list(&mut self, ghosts: &HashSet<String>) -> &[String] {
        let key = (self.version, ghosts.len());
        if self.song_order_key != Some(key) {
            let mut recs: Vec<&SongRec> = self
                .songs
                .values()
                .filter(|rec| rec.direct || ghosts.contains(&rec.hit.video_id))
                .collect();
            recs.extend(
                self.limbo
                    .values()
                    .map(|(rec, _)| rec)
                    .filter(|rec| ghosts.contains(&rec.hit.video_id)),
            );
            recs.sort_by(|a, b| {
                b.saved_at
                    .cmp(&a.saved_at)
                    .then_with(|| a.hit.title.cmp(&b.hit.title))
                    .then_with(|| a.hit.video_id.cmp(&b.hit.video_id))
            });
            self.song_order = recs
                .into_iter()
                .map(|rec| rec.hit.video_id.clone())
                .collect();
            self.song_order_key = Some(key);
        }
        &self.song_order
    }

    /// A song of the collection, or one still in limbo.
    pub fn song_hit(&self, id: &str) -> Option<&SongHit> {
        self.songs
            .get(id)
            .or_else(|| self.limbo.get(id).map(|(rec, _)| rec))
            .map(|rec| &rec.hit)
    }

    /// Every saved song.
    pub fn song_hits(&self) -> impl Iterator<Item = &SongHit> {
        self.songs.values().map(|rec| &rec.hit)
    }

    /// Fill in what saved songs are missing (album, length, artist link) from fresher copies of
    /// the same songs, as pages and searches bring them in. Writes the index only on a change.
    pub fn fill_in<'a>(&mut self, hits: impl IntoIterator<Item = &'a SongHit>) {
        let mut changed = false;
        for hit in hits {
            if let Some(rec) = self.songs.get_mut(&hit.video_id) {
                changed |= fill_gaps(&mut rec.hit, hit);
            }
            if let Some((rec, _)) = self.limbo.get_mut(&hit.video_id) {
                fill_gaps(&mut rec.hit, hit);
            }
        }
        if changed {
            self.version += 1;
            self.write_index();
        }
    }

    /// A length fetched for a song: kept on a saved song whose list gave none.
    pub fn set_length(&mut self, id: &str, text: &str) {
        let Some(rec) = self
            .songs
            .get_mut(id)
            .filter(|rec| rec.hit.duration.is_none())
        else {
            return;
        };
        rec.hit.duration = Some(text.to_owned());
        self.version += 1;
        self.write_index();
    }

    /// The saved albums, newest first.
    pub fn albums(&self) -> Vec<&AlbumRec> {
        let mut list: Vec<&AlbumRec> = self.albums.values().collect();
        list.sort_by(|a, b| {
            b.saved_at
                .cmp(&a.saved_at)
                .then_with(|| a.page.title.cmp(&b.page.title))
        });
        list
    }

    /// `albums()[from..to]`, from an ordering that is only rebuilt when the collection changes. The
    /// Albums list asks every frame, and sorting the lot every frame is wasted work.
    pub fn albums_between(&mut self, from: usize, to: usize) -> Vec<&AlbumRec> {
        if self.album_order_key != Some(self.version) {
            self.album_order = self
                .albums()
                .into_iter()
                .map(|rec| rec.page.browse_id.clone())
                .collect();
            self.album_order_key = Some(self.version);
        }
        let to = to.min(self.album_order.len());
        let from = from.min(to);
        self.album_order[from..to]
            .iter()
            .filter_map(|id| self.albums.get(id))
            .collect()
    }

    pub fn album_page(&self, id: &str) -> Option<&AlbumPage> {
        self.albums.get(id).map(|rec| &rec.page)
    }

    /// The downloaded file for a saved song, when it is there to play.
    pub fn local(&self, id: &str) -> Option<Local> {
        let rec = self.songs.get(id).filter(|rec| rec.downloaded)?;
        let path = self.song_path(id);
        let there = path.exists();
        there.then_some(Local {
            path,
            loudness_db: rec.loudness_db,
            duration_secs: rec.duration_secs,
        })
    }

    // ---- Changing -----------------------------------------------------------------------------

    fn changed(&mut self) {
        self.version += 1;
        self.refresh_ids();
        self.write_index();
    }

    fn refresh_ids(&mut self) {
        self.ids = Arc::new(SavedIds {
            songs: self
                .songs
                .values()
                .filter(|rec| rec.direct)
                .map(|rec| rec.hit.video_id.clone())
                .collect(),
            albums: self.albums.keys().cloned().collect(),
            artists: self.vault.ids().cloned().collect(),
        });
    }

    fn write_index(&self) {
        let Some(writer) = &self.writer else { return };
        let index = Index {
            songs: self.songs.values().cloned().collect(),
            albums: self.albums.values().cloned().collect(),
        };
        if let Ok(bytes) = serde_json::to_vec(&index) {
            writer.put(self.root.join("library.json"), bytes);
        }
    }

    fn remove_file(&self, id: &str) {
        if let Some(writer) = &self.writer {
            writer.remove(self.song_path(id));
        }
    }

    /// Put `hit` in the collection (not persisted; callers do that once per action).
    fn insert_song(&mut self, hit: &SongHit, direct: bool, album: Option<&str>) -> Effects {
        let id = hit.video_id.clone();
        let mut fx = Effects::default();
        if !safe_id(&id) {
            return fx;
        }
        if let Some(rec) = self.songs.get_mut(&id) {
            rec.direct |= direct;
            if let Some(album) = album {
                if !rec.albums.iter().any(|a| a == album) {
                    rec.albums.push(album.to_owned());
                }
            }
            return fx;
        }
        let mut rec = match self.limbo.remove(&id) {
            // Saved again inside the minute: the file is still there, so there is nothing to fetch.
            Some((mut rec, _)) => {
                rec.direct = direct;
                rec.albums.clear();
                rec.saved_at = now_secs();
                rec
            }
            None => SongRec {
                hit: hit.clone(),
                saved_at: now_secs(),
                direct,
                albums: Vec::new(),
                downloaded: false,
                loudness_db: None,
                duration_secs: None,
            },
        };
        if let Some(album) = album {
            rec.albums.push(album.to_owned());
        }
        // (A library with no disk behind it has no files to look for.)
        if rec.downloaded && self.writer.is_some() && !self.song_path(&id).exists() {
            rec.downloaded = false;
        }
        if !rec.downloaded {
            fx.downloads.extend(self.job_for(&rec));
        }
        fx.covers.extend(hit.thumbnail.clone());
        self.songs.insert(id, rec);
        fx
    }

    fn job_for(&mut self, rec: &SongRec) -> Option<Job> {
        let id = &rec.hit.video_id;
        if !self.downloading || self.inflight.contains(id) || self.root.as_os_str().is_empty() {
            return None;
        }
        self.inflight.insert(id.clone());
        Some(Job {
            video_id: id.clone(),
            title: rec.hit.title.clone(),
            artist: rec.hit.artists.join(", "),
            album: rec.hit.album.clone(),
        })
    }

    /// Move a song out of the collection. Its file waits in limbo.
    fn doom(&mut self, id: &str, now: Instant) {
        if let Some(rec) = self.songs.remove(id) {
            self.limbo.insert(id.to_owned(), (rec, now + GRACE));
        }
    }

    pub fn save_song(&mut self, hit: &SongHit, now: Instant) -> Effects {
        self.sweep(now);
        let fx = self.insert_song(hit, true, None);
        self.changed();
        fx
    }

    /// Unlike a song. Its file stays while a saved album still holds it, and goes (after the minute)
    /// once nothing does.
    pub fn unsave_song(&mut self, id: &str, now: Instant) {
        let held = match self.songs.get_mut(id) {
            Some(rec) => {
                rec.direct = false;
                !rec.albums.is_empty()
            }
            None => false,
        };
        if !held {
            self.doom(id, now);
        }
        self.changed();
    }

    pub fn save_album(&mut self, page: &AlbumPage, now: Instant) -> Effects {
        self.sweep(now);
        let mut fx = Effects::default();
        for track in &page.tracks {
            // A track on an album page may not name its own album or cover; the page knows both.
            let mut hit = track.clone();
            hit.album.get_or_insert_with(|| page.title.clone());
            hit.album_id.get_or_insert_with(|| page.browse_id.clone());
            if hit.thumbnail.is_none() {
                hit.thumbnail = page.thumbnail.clone();
            }
            fx.merge(self.insert_song(&hit, false, Some(&page.browse_id)));
        }
        fx.covers.extend(page.thumbnail.clone());
        // Saving again (to fill in a page that was stored without its tracks) keeps its place.
        let saved_at = self
            .albums
            .get(&page.browse_id)
            .map_or_else(now_secs, |rec| rec.saved_at);
        self.albums.insert(
            page.browse_id.clone(),
            AlbumRec {
                page: page.clone(),
                saved_at,
            },
        );
        self.changed();
        fx
    }

    /// Unsave an album, and with it the songs that were saved only for its sake.
    pub fn unsave_album(&mut self, id: &str, now: Instant) {
        let Some(rec) = self.albums.remove(id) else {
            return;
        };
        for track in &rec.page.tracks {
            let orphan = match self.songs.get_mut(&track.video_id) {
                Some(song) => {
                    song.albums.retain(|a| a != id);
                    song.albums.is_empty() && !song.direct
                }
                None => false,
            };
            if orphan {
                self.doom(&track.video_id, now);
            }
        }
        self.changed();
    }

    pub fn save_artist(&mut self, page: &ArtistPage) -> Effects {
        let cover = self.vault.save(page, self.writer.as_ref());
        self.changed();
        Effects {
            downloads: Vec::new(),
            covers: cover.into_iter().collect(),
        }
    }

    pub fn unsave_artist(&mut self, id: &str) {
        self.vault.remove(id, self.writer.as_ref());
        self.changed();
    }

    /// A fresh artist page arrived: if the artist is saved, keep the new copy.
    pub fn refresh_artist(&mut self, page: &ArtistPage) {
        if !self.vault.contains(&page.browse_id) {
            return;
        }
        self.vault.refresh(page, self.writer.as_ref());
        // The picture or name may have changed; the id set has not, so no `changed`.
        self.version += 1;
    }

    // ---- Downloads ----------------------------------------------------------------------------

    /// Turn local downloads on or off. Turning it on does not fetch anything by itself: the caller
    /// asks for [`Self::resume`] once the backend is ready.
    pub fn set_downloading(&mut self, enabled: bool) {
        if self.downloading == enabled {
            return;
        }
        self.downloading = enabled;
        if !enabled {
            self.inflight.clear();
            self.retry.clear();
        }
    }

    /// Delete every downloaded file. The songs stay in the collection, and are downloaded again if
    /// local downloads are turned back on. This is final: there is no undo.
    pub fn clear_downloads(&mut self) {
        self.inflight.clear();
        self.retry.clear();
        // Songs waiting out the grace minute go too, with their files.
        self.limbo.clear();
        // A library with no disk behind it has no folder to clear (and must never sweep the
        // working directory, which an empty root would resolve to).
        if !self.root.as_os_str().is_empty() {
            let _ = std::fs::remove_dir_all(self.downloads_dir());
        }
        for rec in self.songs.values_mut() {
            rec.downloaded = false;
        }
        self.changed();
    }

    /// Downloads for every saved song that has no file yet, for when the backend is ready.
    pub fn resume(&mut self) -> Effects {
        if !self.downloading {
            return Effects::default();
        }
        let pending: Vec<SongRec> = self
            .songs
            .values()
            .filter(|r| !r.downloaded)
            .cloned()
            .collect();
        let mut fx = Effects::default();
        for rec in pending {
            fx.downloads.extend(self.job_for(&rec));
        }
        fx
    }

    /// A download finished.
    pub fn downloaded(&mut self, id: &str, loudness_db: Option<f64>, duration_secs: Option<f64>) {
        self.inflight.remove(id);
        self.retry.remove(id);
        // A download that was already in the backend when the listener turned local downloads off
        // lands after the folder was cleared. It is not wanted: delete it rather than keep a file
        // that the setting says does not exist.
        if !self.downloading {
            self.remove_file(id);
            return;
        }
        if let Some(rec) = self.songs.get_mut(id) {
            rec.downloaded = true;
            rec.loudness_db = loudness_db.or(rec.loudness_db);
            rec.duration_secs = duration_secs.or(rec.duration_secs);
            if rec.hit.duration.is_none() {
                rec.hit.duration = rec.duration_secs.map(length_text);
            }
            self.changed();
        } else if let Some((rec, _)) = self.limbo.get_mut(id) {
            // Unsaved while it downloaded, and still inside the minute: if it is saved again the file
            // is ready, and if not, the sweep takes it.
            rec.downloaded = true;
            rec.loudness_db = loudness_db.or(rec.loudness_db);
            rec.duration_secs = duration_secs.or(rec.duration_secs);
        } else {
            // Nobody wants it any more.
            self.remove_file(id);
        }
    }

    /// A download failed; try again later, a little less eagerly each time.
    pub fn download_failed(&mut self, id: &str, now: Instant) {
        self.inflight.remove(id);
        if !self.songs.contains_key(id) {
            return;
        }
        let tries = self.retry.get(id).map_or(0, |(n, _)| *n);
        let wait = (RETRY_FIRST * 2u32.saturating_pow(tries.min(6))).min(RETRY_MAX);
        self.retry.insert(id.to_owned(), (tries + 1, now + wait));
    }

    // ---- Time ---------------------------------------------------------------------------------

    /// Remove the files of songs whose minute is up.
    fn sweep(&mut self, now: Instant) {
        let expired: Vec<String> = self
            .limbo
            .iter()
            .filter(|(_, (_, until))| *until <= now)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            self.limbo.remove(&id);
            self.remove_file(&id);
        }
    }

    /// Once per frame: clear what has expired, and say what is due to be tried again.
    pub fn tick(&mut self, now: Instant) -> Effects {
        if !self.limbo.is_empty() {
            self.sweep(now);
        }
        let mut fx = Effects::default();
        if !self.retry.is_empty() {
            let due: Vec<String> = self
                .retry
                .iter()
                .filter(|(id, (_, at))| *at <= now && !self.inflight.contains(*id))
                .map(|(id, _)| id.clone())
                .collect();
            for id in due {
                match self.songs.get(&id).cloned() {
                    Some(rec) => fx.downloads.extend(self.job_for(&rec)),
                    None => {
                        self.retry.remove(&id);
                    }
                }
            }
        }
        fx
    }

    /// When [`Self::tick`] next has something to do, so the host can sleep until then.
    pub fn next_wake(&self) -> Option<Instant> {
        self.limbo
            .values()
            .map(|(_, until)| *until)
            .chain(
                self.retry
                    .iter()
                    .filter(|(id, _)| !self.inflight.contains(*id))
                    .map(|(_, (_, at))| *at),
            )
            .min()
    }

    /// Skip every wait: files in limbo go now. Called as the app closes.
    pub fn settle(&mut self) {
        let doomed: Vec<String> = self.limbo.keys().cloned().collect();
        self.limbo.clear();
        for id in doomed {
            self.remove_file(&id);
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // Closing the app skips the wait. The writer finishes its queue as it drops.
        self.settle();
    }
}

/// Copy into `song` what `other` (the same song, seen elsewhere) knows and `song` does not: the
/// album and its link, the length, the artist link and the cover. Never overwrites. Says whether
/// anything changed.
pub fn fill_gaps(song: &mut SongHit, other: &SongHit) -> bool {
    if song.video_id != other.video_id {
        return false;
    }
    let before = song.clone();
    if song.album_id.is_none() && other.album_id.is_some() {
        song.album_id = other.album_id.clone();
        song.album = other.album.clone().or(song.album.take());
    }
    if song.album.is_none() {
        song.album = other.album.clone();
    }
    if song.duration.is_none() {
        song.duration = other.duration.clone();
    }
    if song.artist_id.is_none() {
        song.artist_id = other.artist_id.clone();
    }
    if song.artists.is_empty() {
        song.artists = other.artists.clone();
    }
    // Songs saved before each artist's link was kept pick the links up from a fresher copy.
    if song.artist_ids.is_empty() && !other.artist_ids.is_empty() && song.artists == other.artists {
        song.artist_ids = other.artist_ids.clone();
    }
    if song.thumbnail.is_none() {
        song.thumbnail = other.thumbnail.clone();
    }
    *song != before
}

/// A length in seconds as a list shows it: `m:ss`, or `h:mm:ss` past the hour.
pub fn length_text(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (hours, minutes, seconds) = (total / 3600, total / 60 % 60, total % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_are_filled_from_a_fresher_copy_and_nothing_is_overwritten() {
        let mut song = hit("a");
        song.album = None;
        song.album_id = None;
        song.duration = None;
        let mut fresher = hit("a");
        fresher.album = Some("Discovery".to_owned());
        fresher.album_id = Some("MPREb_disc".to_owned());
        fresher.duration = Some("5:20".to_owned());
        fresher.title = "Another title".to_owned();
        assert!(fill_gaps(&mut song, &fresher));
        assert_eq!(song.album.as_deref(), Some("Discovery"));
        assert_eq!(song.album_id.as_deref(), Some("MPREb_disc"));
        assert_eq!(song.duration.as_deref(), Some("5:20"));
        assert_eq!(song.title, "Song a");
        assert!(
            !fill_gaps(&mut song, &fresher),
            "a second pass changes nothing"
        );
        assert!(
            !fill_gaps(&mut song, &hit("b")),
            "another song is never used"
        );
    }

    #[test]
    fn a_saved_song_learns_its_album_and_length() {
        let mut library = Library::in_memory();
        let mut bare = hit("a");
        bare.album = None;
        bare.album_id = None;
        bare.duration = None;
        let _ = library.save_song(&bare, Instant::now());
        let mut fresher = bare.clone();
        fresher.album = Some("Discovery".to_owned());
        fresher.album_id = Some("MPREb_disc".to_owned());
        library.fill_in([&fresher]);
        library.set_length("a", "4:01");
        let saved = library.song_hit("a").unwrap();
        assert_eq!(saved.album_id.as_deref(), Some("MPREb_disc"));
        assert_eq!(saved.duration.as_deref(), Some("4:01"));
    }

    #[test]
    fn lengths_read_like_the_lists() {
        assert_eq!(length_text(338.0), "5:38");
        assert_eq!(length_text(59.6), "1:00");
        assert_eq!(length_text(3723.0), "1:02:03");
    }

    fn hit(id: &str) -> SongHit {
        SongHit {
            video_id: id.to_owned(),
            title: format!("Song {id}"),
            artists: vec!["Someone".to_owned()],
            album: None,
            duration: Some("3:00".to_owned()),
            thumbnail: Some(format!("https://example.test/{id}")),
            artist_id: None,
            album_id: None,
            artist_ids: Vec::new(),
        }
    }

    fn album(id: &str, tracks: &[&str]) -> AlbumPage {
        AlbumPage {
            browse_id: id.to_owned(),
            title: "An Album".to_owned(),
            tracks: tracks.iter().map(|t| hit(t)).collect(),
            ..AlbumPage::default()
        }
    }

    /// A library that behaves as if it had a disk, without one: an empty `root` would stop jobs
    /// from being made, so give it a name that is never touched (no writer is attached).
    fn library() -> Library {
        let mut library = Library::in_memory();
        library.root = PathBuf::from("v1-test-root-never-written");
        library
    }

    #[test]
    fn saving_a_song_starts_one_download_and_marks_it_saved() {
        let mut lib = library();
        let now = Instant::now();
        let fx = lib.save_song(&hit("aaa"), now);
        assert_eq!(fx.downloads.len(), 1);
        assert!(lib.song_saved("aaa") && lib.ids().songs.contains("aaa"));
        let again = lib.save_song(&hit("aaa"), now);
        assert!(
            again.downloads.is_empty(),
            "already saved, already downloading"
        );
    }

    #[test]
    fn saving_again_within_a_minute_restores_without_a_second_download() {
        let mut lib = library();
        let t0 = Instant::now();
        lib.save_song(&hit("aaa"), t0);
        lib.downloaded("aaa", Some(1.0), Some(180.0));
        lib.unsave_song("aaa", t0);
        assert!(!lib.song_saved("aaa"), "gone from the collection at once");
        assert!(lib.next_wake().is_some(), "but its file is waiting");
        let fx = lib.save_song(&hit("aaa"), t0 + Duration::from_secs(59));
        assert!(
            fx.downloads.is_empty(),
            "the file was kept, so nothing is fetched"
        );
        assert!(lib.songs["aaa"].downloaded);
        assert!(lib.next_wake().is_none());
    }

    #[test]
    fn saving_after_the_minute_downloads_afresh() {
        let mut lib = library();
        let t0 = Instant::now();
        lib.save_song(&hit("aaa"), t0);
        lib.downloaded("aaa", None, None);
        lib.unsave_song("aaa", t0);
        // No tick ran in between: saving still honours the deadline.
        let fx = lib.save_song(&hit("aaa"), t0 + Duration::from_secs(61));
        assert_eq!(fx.downloads.len(), 1);
        assert!(!lib.songs["aaa"].downloaded);
    }

    #[test]
    fn closing_skips_the_wait() {
        let mut lib = library();
        let now = Instant::now();
        lib.save_song(&hit("aaa"), now);
        lib.unsave_song("aaa", now);
        lib.settle();
        assert!(lib.next_wake().is_none());
    }

    #[test]
    fn an_album_saves_its_songs_and_unsaving_it_takes_back_only_those() {
        let mut lib = library();
        let now = Instant::now();
        lib.save_song(&hit("b"), now); // saved on its own as well
        let fx = lib.save_album(&album("MPREb_x", &["a", "b", "c"]), now);
        assert_eq!(fx.downloads.len(), 2, "b was already on its way");
        assert!(
            ["a", "b", "c"].iter().all(|id| lib.songs.contains_key(*id)),
            "all downloaded"
        );
        assert!(
            !lib.song_saved("a") && !lib.song_saved("c"),
            "but not liked"
        );
        assert!(lib.song_saved("b"), "b was liked on its own too");
        assert!(lib.ids().albums.contains("MPREb_x"));

        lib.unsave_album("MPREb_x", now);
        assert!(!lib.songs.contains_key("a") && !lib.songs.contains_key("c"));
        assert!(lib.song_saved("b"), "b was liked on its own too");
        assert!(!lib.ids().albums.contains("MPREb_x"));
    }

    #[test]
    fn album_songs_are_not_in_the_songs_list() {
        let mut lib = library();
        let now = Instant::now();
        lib.save_album(&album("one", &["a", "b"]), now);
        assert!(lib.song_list(&HashSet::new()).is_empty());
        assert!(lib.ids().songs.is_empty());
        lib.save_song(&hit("a"), now);
        assert_eq!(lib.song_list(&HashSet::new()), ["a".to_owned()]);
    }

    #[test]
    fn a_song_on_two_saved_albums_stays_until_both_go() {
        let mut lib = library();
        let now = Instant::now();
        lib.save_album(&album("one", &["a"]), now);
        lib.save_album(&album("two", &["a"]), now);
        lib.unsave_album("one", now);
        assert!(lib.songs.contains_key("a"));
        lib.unsave_album("two", now);
        assert!(!lib.songs.contains_key("a"));
    }

    #[test]
    fn unliking_a_song_keeps_it_while_a_saved_album_holds_it() {
        let mut lib = library();
        let t0 = Instant::now();
        lib.save_album(&album("one", &["a"]), t0);
        lib.save_song(&hit("a"), t0);
        lib.downloaded("a", None, None);
        lib.unsave_song("a", t0);
        assert!(!lib.song_saved("a"), "no longer liked");
        assert!(lib.songs["a"].downloaded, "file kept for the album");
        assert!(lib.next_wake().is_none(), "nothing waiting to be removed");
        lib.sweep(t0 + Duration::from_secs(61));
        assert!(lib.songs.contains_key("a"));
        lib.unsave_album("one", t0);
        assert!(
            lib.next_wake().is_some(),
            "now nothing holds it: it waits to go"
        );
    }

    #[test]
    fn unliking_an_album_keeps_songs_that_are_liked() {
        let mut lib = library();
        let now = Instant::now();
        lib.save_song(&hit("a"), now);
        lib.save_album(&album("one", &["a"]), now);
        lib.unsave_album("one", now);
        assert!(lib.song_saved("a") && lib.songs.contains_key("a"));
    }

    #[test]
    fn songs_just_unsaved_stay_listed_only_for_the_reader_who_asks_for_them() {
        let mut lib = library();
        let now = Instant::now();
        lib.save_song(&hit("a"), now);
        lib.save_song(&hit("b"), now);
        lib.unsave_song("a", now);
        assert_eq!(lib.song_list(&HashSet::new()).len(), 1);
        let ghosts: HashSet<String> = ["a".to_owned()].into();
        assert_eq!(lib.song_list(&ghosts).len(), 2);
    }

    #[test]
    fn a_failed_download_is_retried_later_and_a_finished_one_is_not() {
        let mut lib = library();
        let t0 = Instant::now();
        lib.save_song(&hit("a"), t0);
        lib.download_failed("a", t0);
        assert!(
            lib.tick(t0 + Duration::from_secs(10)).downloads.is_empty(),
            "too soon"
        );
        assert_eq!(
            lib.tick(t0 + RETRY_FIRST + Duration::from_secs(1))
                .downloads
                .len(),
            1
        );
        lib.downloaded("a", None, None);
        assert!(lib
            .tick(t0 + Duration::from_secs(99_999))
            .downloads
            .is_empty());
    }

    #[test]
    fn a_download_finishing_for_a_song_nobody_wants_is_not_kept() {
        let mut lib = library();
        let t0 = Instant::now();
        lib.save_song(&hit("a"), t0);
        lib.unsave_song("a", t0);
        lib.sweep(t0 + Duration::from_secs(61));
        lib.downloaded("a", None, None);
        assert!(!lib.song_saved("a"));
    }
}
