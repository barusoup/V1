// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The order of songs: what is playing, what is up next, and what played before it.
//!
//! This holds the rules and no playback, so every transition can be tested without audio or the
//! network. The screen calls into it and then starts or stops the player to match.
//!
//! The rules:
//! - Moving forward (Next, a pick from a list, or the end of a song) files the song that was
//!   playing into the history. Only a song that actually started is filed.
//! - Going back takes the newest history entry and files the song being left at the front of the
//!   queue, so Next walks back to it and the queue keeps its order.
//! - Back restarts the current song instead when it is past its first few seconds, or when there
//!   is no history.
//! - A load can be superseded before it starts. A skipped song is dropped. A song that was being
//!   loaded by going back returns to the history, and one loaded forward returns to the queue
//!   when a back takes over, so nothing is lost by pressing buttons quickly.
//! - A song that fails to load is dropped without being filed, because it never played.

use std::collections::VecDeque;

use yt_client::SongHit;

/// Songs kept in the history. Older ones fall off the front.
pub const HISTORY_LIMIT: usize = 100;

/// Which way a song is being loaded. It decides where the song that was playing goes once the
/// new one starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Back,
}

#[derive(Debug, Default)]
pub struct Playlist {
    /// Up next, in order.
    pub queue: VecDeque<SongHit>,
    /// What played before the current song, oldest first. Back takes from the end.
    pub history: VecDeque<SongHit>,
    /// The song that is audible, once it has started.
    pub current: Option<SongHit>,
    /// A song asked for and not yet started.
    loading: Option<(SongHit, Direction)>,
}

impl Playlist {
    /// The video id of the song being loaded, so a backend reply can be matched to it.
    pub fn loading_id(&self) -> Option<&str> {
        self.loading.as_ref().map(|(hit, _)| hit.video_id.as_str())
    }

    pub fn is_loading(&self) -> bool {
        self.loading.is_some()
    }

    /// Fill in what the songs here are missing from fresher copies of them (see
    /// [`crate::library::fill_gaps`]): the queue, the history, and the current and loading song.
    pub fn fill_in<'a>(&mut self, hits: impl IntoIterator<Item = &'a SongHit>) {
        let by_id: std::collections::HashMap<&str, &SongHit> =
            hits.into_iter().map(|hit| (hit.video_id.as_str(), hit)).collect();
        if by_id.is_empty() {
            return;
        }
        let songs = self
            .queue
            .iter_mut()
            .chain(self.history.iter_mut())
            .chain(self.current.as_mut())
            .chain(self.loading.as_mut().map(|(hit, _)| hit));
        for song in songs {
            if let Some(other) = by_id.get(song.video_id.as_str()) {
                crate::library::fill_gaps(song, other);
            }
        }
    }

    /// The song being loaded, so the bar can show its details before it starts.
    pub fn loading_song(&self) -> Option<&SongHit> {
        self.loading.as_ref().map(|(hit, _)| hit)
    }

    /// Play `results[index]`. The songs after it become the queue. Returns the song to load.
    pub fn play_from(&mut self, results: &[SongHit], index: usize) -> Option<SongHit> {
        let hit = results.get(index)?.clone();
        self.cancel(Direction::Forward);
        self.queue = results[index + 1..].iter().cloned().collect();
        Some(self.begin(hit, Direction::Forward))
    }

    /// Play one song on its own, as a single pick from a search: the queue is left as it is.
    pub fn play_one(&mut self, hit: SongHit) -> SongHit {
        self.cancel(Direction::Forward);
        self.begin(hit, Direction::Forward)
    }

    /// Next: skip whatever is loading and take the head of the queue.
    pub fn next(&mut self) -> Option<SongHit> {
        self.cancel(Direction::Forward);
        self.advance()
    }

    /// Pick the queue entry at `index`; the entries before it are skipped.
    pub fn skip_to(&mut self, index: usize) -> Option<SongHit> {
        for _ in 0..index.min(self.queue.len()) {
            self.queue.pop_front();
        }
        self.next()
    }

    /// Add a song to the end of the queue. Nothing starts: the queue only waits for Next.
    pub fn enqueue(&mut self, hit: SongHit) {
        self.queue.push_back(hit);
    }

    /// Take the queue entry at `index` out of the queue, so it will not play.
    pub fn remove_queued(&mut self, index: usize) -> Option<SongHit> {
        self.queue.remove(index)
    }

    /// Move the queue entry at `from` so it ends up at `to`; the others keep their order.
    pub fn move_queued(&mut self, from: usize, to: usize) {
        let len = self.queue.len();
        if from == to || from >= len || to >= len {
            return;
        }
        if let Some(hit) = self.queue.remove(from) {
            self.queue.insert(to, hit);
        }
    }

    /// Back to the previous song. `None` means there is nothing to go back to, or the current song
    /// is far enough in that it should restart: the caller restarts it.
    pub fn back(&mut self, past_start: bool) -> Option<SongHit> {
        self.cancel(Direction::Back);
        if past_start || self.history.is_empty() {
            return None;
        }
        let hit = self.history.pop_back()?;
        Some(self.begin(hit, Direction::Back))
    }

    /// The song at `video_id` has started, so it becomes current and the song it replaces is filed.
    /// Returns the song, or `None` when that load was superseded.
    pub fn started(&mut self, video_id: &str) -> Option<SongHit> {
        if self.loading_id() != Some(video_id) {
            return None;
        }
        let (hit, direction) = self.loading.take()?;
        if let Some(old) = self.current.replace(hit.clone()) {
            match direction {
                Direction::Forward => self.file(old),
                Direction::Back => self.queue.push_front(old),
            }
        }
        Some(hit)
    }

    /// The load for `video_id` failed, so the song is dropped. Returns whether it was the one
    /// loading; a reply for a superseded load is ignored.
    pub fn load_failed(&mut self, video_id: &str) -> bool {
        if self.loading_id() == Some(video_id) {
            self.loading = None;
            true
        } else {
            false
        }
    }

    /// The current song ended and nothing is loading: file it and move on to the queue head.
    pub fn finished(&mut self) -> Option<SongHit> {
        if let Some(old) = self.current.take() {
            self.file(old);
        }
        self.next()
    }

    /// Playback stopped on its own (the stream failed mid-song). The song is not filed.
    pub fn stopped(&mut self) {
        self.current = None;
    }

    fn begin(&mut self, hit: SongHit, direction: Direction) -> SongHit {
        self.loading = Some((hit.clone(), direction));
        hit
    }

    fn advance(&mut self) -> Option<SongHit> {
        let hit = self.queue.pop_front()?;
        Some(self.begin(hit, Direction::Forward))
    }

    /// Undo a load that has not started, because the user moved on before it did.
    fn cancel(&mut self, incoming: Direction) {
        let Some((hit, direction)) = self.loading.take() else {
            return;
        };
        match (direction, incoming) {
            (Direction::Forward, Direction::Forward) => {}
            (Direction::Forward, Direction::Back) => self.queue.push_front(hit),
            (Direction::Back, _) => self.history.push_back(hit),
        }
    }

    fn file(&mut self, hit: SongHit) {
        self.history.push_back(hit);
        while self.history.len() > HISTORY_LIMIT {
            self.history.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn song(id: &str) -> SongHit {
        SongHit {
            video_id: id.to_owned(),
            title: id.to_owned(),
            artists: Vec::new(),
            album: None,
            duration: None,
            thumbnail: None,
            artist_id: None,
            album_id: None,
            artist_ids: Vec::new(),
        }
    }

    fn ids(songs: &VecDeque<SongHit>) -> Vec<&str> {
        songs.iter().map(|hit| hit.video_id.as_str()).collect()
    }

    /// Play a list of songs starting at `a`, and let it start: the usual opening.
    fn playing(list: &[&str]) -> Playlist {
        let results: Vec<SongHit> = list.iter().map(|id| song(id)).collect();
        let mut playlist = Playlist::default();
        let first = playlist.play_from(&results, 0).expect("first song");
        assert!(playlist.started(&first.video_id).is_some());
        playlist
    }

    fn load(playlist: &mut Playlist, hit: Option<SongHit>) {
        let hit = hit.expect("a song to load");
        assert!(playlist.started(&hit.video_id).is_some());
    }

    #[test]
    fn next_files_the_song_that_was_playing() {
        let mut playlist = playing(&["a", "b", "c"]);
        let hit = playlist.next();
        load(&mut playlist, hit);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "b");
        assert_eq!(ids(&playlist.history), ["a"]);
        assert_eq!(ids(&playlist.queue), ["c"]);
    }

    #[test]
    fn back_goes_to_the_previous_song_and_next_returns_to_the_one_left() {
        let mut playlist = playing(&["a", "b", "c"]);
        let hit = playlist.next();
        load(&mut playlist, hit);

        let hit = playlist.back(false);
        load(&mut playlist, hit);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
        assert!(playlist.history.is_empty());
        assert_eq!(ids(&playlist.queue), ["b", "c"]);

        let hit = playlist.next();
        load(&mut playlist, hit);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "b");
        assert_eq!(ids(&playlist.history), ["a"]);
        assert_eq!(ids(&playlist.queue), ["c"]);
    }

    #[test]
    fn back_restarts_when_the_song_is_past_its_start() {
        let mut playlist = playing(&["a", "b"]);
        let hit = playlist.next();
        load(&mut playlist, hit);
        assert!(playlist.back(true).is_none());
        assert!(!playlist.is_loading());
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "b");
        assert_eq!(ids(&playlist.history), ["a"]);
    }

    #[test]
    fn back_with_no_history_restarts() {
        let mut playlist = playing(&["a", "b"]);
        assert!(playlist.back(false).is_none());
        assert!(!playlist.is_loading());
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
        assert_eq!(ids(&playlist.queue), ["b"]);
    }

    #[test]
    fn a_playlist_pick_files_the_current_song_too() {
        let results: Vec<SongHit> = ["a", "b", "c", "d"].iter().map(|id| song(id)).collect();
        let mut playlist = playing(&["a"]);
        let hit = playlist.play_from(&results, 2).expect("c");
        load(&mut playlist, Some(hit));
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "c");
        assert_eq!(ids(&playlist.history), ["a"]);
        assert_eq!(ids(&playlist.queue), ["d"]);
    }

    #[test]
    fn skipping_a_queue_entry_drops_the_ones_before_it() {
        let mut playlist = playing(&["a", "b", "c", "d"]);
        let hit = playlist.skip_to(1);
        load(&mut playlist, hit);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "c");
        assert_eq!(ids(&playlist.history), ["a"]);
        assert_eq!(ids(&playlist.queue), ["d"]);
    }

    #[test]
    fn next_while_loading_skips_the_loading_song() {
        let mut playlist = playing(&["a", "b", "c"]);
        let hit = playlist.next();
        assert_eq!(hit.map(|hit| hit.video_id), Some("b".to_owned()));
        let hit = playlist.next();
        load(&mut playlist, hit);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "c");
        assert_eq!(ids(&playlist.history), ["a"]);
        assert!(playlist.queue.is_empty());
    }

    #[test]
    fn back_while_loading_forward_returns_the_pending_song_to_the_queue() {
        let mut playlist = playing(&["a", "b", "c"]);
        let hit = playlist.next();
        load(&mut playlist, hit);
        // `c` is loading forward: it has left the queue but has not started.
        let hit = playlist.next();
        assert_eq!(hit.map(|hit| hit.video_id), Some("c".to_owned()));
        // Going back takes over: `c` returns to the queue and `a` loads from the history.
        let hit = playlist.back(false);
        assert_eq!(hit.map(|hit| hit.video_id), Some("a".to_owned()));
        assert_eq!(ids(&playlist.queue), ["c"]);
        load(&mut playlist, Some(song("a")));
        // `b`, which was playing, goes to the front of the queue so Next returns to it.
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
        assert_eq!(ids(&playlist.queue), ["b", "c"]);
        assert!(playlist.history.is_empty());
    }

    #[test]
    fn repeated_back_while_loading_asks_for_the_same_song_again() {
        let mut playlist = playing(&["a", "b", "c"]);
        let hit = playlist.next();
        load(&mut playlist, hit);
        let hit = playlist.next();
        load(&mut playlist, hit);
        // History is a and b, c is playing. Back asks for b.
        let hit = playlist.back(false);
        assert_eq!(hit.map(|hit| hit.video_id), Some("b".to_owned()));
        // Back again before b has started: b returns to the history and is asked for again,
        // so the same song is loaded and nothing is lost.
        let hit = playlist.back(false);
        assert_eq!(hit.map(|hit| hit.video_id), Some("b".to_owned()));
        assert_eq!(ids(&playlist.history), ["a"]);
        assert_eq!(playlist.loading_id(), Some("b"));
    }

    #[test]
    fn a_reply_for_a_superseded_load_is_ignored() {
        let mut playlist = playing(&["a", "b", "c"]);
        let _ = playlist.next(); // `b` loading
        let _ = playlist.next(); // `c` loading, `b` skipped
        assert!(playlist.started("b").is_none());
        assert!(!playlist.load_failed("b"));
        assert_eq!(playlist.loading_id(), Some("c"));
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
    }

    #[test]
    fn a_failed_load_is_dropped_and_not_filed() {
        let mut playlist = playing(&["a", "b", "c"]);
        let _ = playlist.next(); // `b` loading
        assert!(playlist.load_failed("b"));
        assert!(!playlist.is_loading());
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
        assert!(playlist.history.is_empty());
        assert_eq!(ids(&playlist.queue), ["c"]);
    }

    #[test]
    fn moving_a_queue_entry_reorders_the_queue_and_keeps_the_rest() {
        let mut playlist = playing(&["a", "b", "c", "d"]);
        playlist.move_queued(0, 2);
        assert_eq!(ids(&playlist.queue), ["c", "d", "b"]);
        playlist.move_queued(2, 0);
        assert_eq!(ids(&playlist.queue), ["b", "c", "d"]);
        // Out of range, or no move: nothing changes.
        playlist.move_queued(0, 9);
        playlist.move_queued(1, 1);
        assert_eq!(ids(&playlist.queue), ["b", "c", "d"]);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
    }

    #[test]
    fn enqueueing_appends_and_leaves_the_playing_song_alone() {
        let mut playlist = playing(&["a", "b"]);
        playlist.enqueue(song("c"));
        assert_eq!(ids(&playlist.queue), ["b", "c"]);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
        assert!(!playlist.is_loading());
    }

    #[test]
    fn removing_a_queue_entry_leaves_the_playing_song_alone() {
        let mut playlist = playing(&["a", "b", "c"]);
        let removed = playlist.remove_queued(0).expect("b");
        assert_eq!(removed.video_id, "b");
        assert_eq!(ids(&playlist.queue), ["c"]);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
        assert!(playlist.history.is_empty());
        assert!(playlist.remove_queued(5).is_none());
    }

    #[test]
    fn finishing_with_an_empty_queue_files_the_song_and_stops() {
        let mut playlist = playing(&["a"]);
        assert!(playlist.finished().is_none());
        assert!(playlist.current.is_none());
        assert_eq!(ids(&playlist.history), ["a"]);
        // Back can still reach it.
        let hit = playlist.back(false);
        load(&mut playlist, hit);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "a");
    }

    #[test]
    fn playing_one_search_result_leaves_the_queue_alone() {
        let mut playlist = playing(&["a", "b"]);
        playlist.enqueue(song("q"));
        let hit = playlist.play_one(song("s"));
        assert!(playlist.started(&hit.video_id).is_some());
        assert_eq!(ids(&playlist.queue), ["b", "q"]);
    }

    #[test]
    fn finishing_moves_on_to_the_queue() {
        let mut playlist = playing(&["a", "b"]);
        let hit = playlist.finished();
        load(&mut playlist, hit);
        assert_eq!(playlist.current.as_ref().unwrap().video_id, "b");
        assert_eq!(ids(&playlist.history), ["a"]);
    }

    #[test]
    fn a_failure_mid_song_does_not_file_it() {
        let mut playlist = playing(&["a", "b"]);
        playlist.stopped();
        assert!(playlist.current.is_none());
        assert!(playlist.history.is_empty());
    }

    #[test]
    fn history_is_capped() {
        let list: Vec<String> = (0..HISTORY_LIMIT + 5).map(|i| format!("s{i}")).collect();
        let refs: Vec<&str> = list.iter().map(String::as_str).collect();
        let mut playlist = playing(&refs[..1]);
        for id in &refs[1..] {
            playlist.queue.push_back(song(id));
        }
        for _ in 1..refs.len() {
            let hit = playlist.next();
            load(&mut playlist, hit);
        }
        // 104 songs were filed; the oldest four fell off.
        assert_eq!(playlist.history.len(), HISTORY_LIMIT);
        assert_eq!(playlist.history.front().unwrap().video_id, "s4");
    }
}
