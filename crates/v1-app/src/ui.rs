// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The screen: state, and one function that turns it into a frame.
//!
//! Layout is four surfaces on a black page:
//!
//! ```text
//! ┌─────────┬───────────────────────────────┐
//! │ sidebar │  ·  search (pops in on hover) │
//! │ (folds) │  page: title, then content    │
//! ├─────────┴───────────────────────────────┤
//! │ player bar                              │
//! └─────────────────────────────────────────┘
//! ```
//!
//! Widgets are placed directly (no layout solver) so the geometry stays obvious. Everything that
//! moves is a spring from [`crate::anim`], so interrupting any motion turns it around smoothly.
//! All sizes are logical; the host scales the whole UI with the window (see `main.rs`).

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use dream::ui::draw::UiFrame;
use dream::ui::font::{Align, FontId};
use dream::ui::theme::Theme;
use dream::ui::widgets::{
    ScrollBehavior, ScrollToOptions, ScrollbarStyle, TextFieldStyle, Ui, UiInput,
};
use dream::ui::{Color, Rect};

use yt_client::{AlbumCard, AlbumPage, ArtistCard, ArtistPage, TopResult};
use yt_lyrics::{active_line, Body, Lookup, Query, Source, TimedLine};

use crate::anim::{rate, Anims};
use crate::backend::{Backend, Command, Event, Ready};
use crate::gpu::{CoverImage, Gpu};
use crate::icons::{self, Icon};
use crate::library::{Effects, Library};
use crate::playlist::Playlist;
use crate::spectrum::{self, Spectrum};
use crate::theme::{self, ease_out, fade, mix};
use crate::widgets::{hash, Cx, QueueDrag, SliderId};
pub use crate::widgets::Pointer;

type P = (f32, f32);

/// The destinations in the sidebar. Adding a tab is: a variant here, an icon, and a `match` arm in
/// [`page`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Library,
    Queue,
}

impl Tab {
    const ALL: [Tab; 2] = [Tab::Library, Tab::Queue];

    fn label(self) -> &'static str {
        match self {
            Tab::Library => "Collection",
            Tab::Queue => "Queue",
        }
    }

    fn icon(self) -> Icon {
        match self {
            Tab::Library => Icon::Book,
            Tab::Queue => Icon::Menu,
        }
    }
}

/// The three lists inside the Collection page, switched from the tabs at its top left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionTab {
    Songs,
    Artists,
    Albums,
}

impl CollectionTab {
    const ALL: [CollectionTab; 3] = [CollectionTab::Songs, CollectionTab::Albums, CollectionTab::Artists];

    fn icon(self) -> Icon {
        match self {
            CollectionTab::Songs => Icon::Music,
            CollectionTab::Artists => Icon::User,
            CollectionTab::Albums => Icon::Disc,
        }
    }

    fn index(self) -> usize {
        match self {
            CollectionTab::Songs => 0,
            CollectionTab::Albums => 1,
            CollectionTab::Artists => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            CollectionTab::Songs => "Songs",
            CollectionTab::Artists => "Artists",
            CollectionTab::Albums => "Albums",
        }
    }
}

/// The three kinds of result a search shows, switched from the tabs under its top result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultsTab {
    Songs,
    Artists,
    Albums,
}

impl ResultsTab {
    const ALL: [ResultsTab; 3] = [ResultsTab::Songs, ResultsTab::Albums, ResultsTab::Artists];

    fn icon(self) -> Icon {
        match self {
            ResultsTab::Songs => Icon::Music,
            ResultsTab::Artists => Icon::User,
            ResultsTab::Albums => Icon::Disc,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ResultsTab::Songs => "Songs",
            ResultsTab::Artists => "Artists",
            ResultsTab::Albums => "Albums",
        }
    }
}

/// A page a link opened: an artist's channel or an album. It is shown over the results or the tab,
/// and the back button returns to whatever was underneath.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Detail {
    Artist(String),
    Album(String),
}

/// The pages opened over the results or the tab. `top` is the one showing (`None` is the results or
/// the tab themselves). `below` holds what was showing before each open, so back returns to it.
#[derive(Default)]
struct Pages {
    top: Option<Detail>,
    below: Vec<Option<Detail>>,
}

impl Pages {
    /// Open `detail` over what is showing. Opening the page already on top changes nothing. Returns
    /// whether the stack changed.
    fn open(&mut self, detail: Detail) -> bool {
        if self.top.as_ref() == Some(&detail) {
            return false;
        }
        self.below.push(self.top.take());
        self.top = Some(detail);
        true
    }

    /// Back to the page before the one on top. Returns false when there is nothing to go back to.
    fn back(&mut self) -> bool {
        if self.top.is_none() {
            return false;
        }
        self.top = self.below.pop().flatten();
        true
    }

    /// Drop every page: the results or the tab, with nothing to go back to.
    fn clear(&mut self) {
        self.top = None;
        self.below.clear();
    }
}

/// The search results as they were when the reader left them for a page, so the back button can
/// return to them, at the same scroll position.
struct ResultsReturn {
    query: String,
    results_query: String,
    scroll: f32,
}

/// A fetched page as the UI holds it: asked for, there, or failed with why.
enum Loaded<T> {
    Loading,
    Ready(T),
    Failed(String),
}

/// A fetched page as the UI holds it.
fn loaded<T>(result: Result<T, String>) -> Loaded<T> {
    match result {
        Ok(page) => Loaded::Ready(page),
        Err(message) => Loaded::Failed(message),
    }
}

/// How many times a song's length is asked for before giving up on it for the session.
const LENGTH_TRIES: u32 = 4;

/// How many artist or album pages are kept fetched at once. Each is a few tens of kilobytes.
const PAGE_CACHE: usize = 48;

/// The back button's width, and the gap after it: the search field gives up this much of its left
/// side to make room for it.
const BACK_STEP: f32 = 54.0;

/// The lyrics for the track that is playing, once a lookup has been sent for it. `lookup` is `None`
/// while the providers are still being asked.
struct LyricsView {
    video_id: String,
    lookup: Option<Lookup>,
    /// Counts the sets shown this session, so each one's springs start fresh.
    gen: u32,
    /// When `lookup` arrived. Drives the entrance.
    arrived: Instant,
}

/// How the lyric page scrolls. It keeps the line being sung in the middle. Once the reader scrolls
/// it, the page stays where they left it, and goes back to following once they have been still for
/// [`LYRIC_IDLE_SECS`].
#[derive(Default)]
struct LyricsFollow {
    /// The lyric set this belongs to. A new set starts fresh.
    set: Option<u32>,
    /// The reader has scrolled, so the page is holding still for them.
    reader: bool,
    /// When the reader last scrolled the page.
    scrolled_at: Option<Instant>,
    /// Whether the page has been placed yet. The first placement is instant, not a glide.
    placed: bool,
}

/// Back restarts the current song when it is further in than this, the way most players do.
const RESTART_AFTER_SECS: f32 = 3.0;

/// One playing track.
pub struct Now {
    pub video_id: String,
    /// The song as the lists know it, for saving it from the bar.
    pub hit: yt_client::SongHit,
    pub title: String,
    pub artist: String,
    /// The browse id of the artist, when known: the artist name links to their page.
    pub artist_id: Option<String>,
    pub thumbnail: Option<String>,
    pub player: dream_media::Player,
    pub position: f32,
    pub duration: f32,
    pub paused: bool,
    /// What the listener asked for. Pausing is a fade: this goes false at once, the gain falls to
    /// silence, and only then is the player actually paused.
    pub want_playing: bool,
    /// Output gain, 0 to 1, moved toward `want_playing` over [`FADE_SECS`].
    pub gain: f32,
    /// Volume normalisation: a fixed per-track factor in 0 to 1, from YouTube's loudness figure.
    pub norm: f32,
    /// The player has been paused for good, so the pause is not repeated every frame.
    pub held: bool,
}

/// The factor that brings a track to YouTube's reference loudness. A positive `loudness_db` means
/// the track is louder than the reference, so it is turned down by that much. Quiet tracks are
/// left alone, as YouTube does, since boosting could clip. Unknown loudness is left alone too.
fn normalisation(loudness_db: Option<f64>) -> f32 {
    match loudness_db {
        Some(db) if db.is_finite() && db > 0.0 => 10f64.powf(-db / 20.0) as f32,
        _ => 1.0,
    }
}

/// How long a pause or a start takes to fade. Brief enough to feel immediate, long enough to
/// avoid a click.
const FADE_SECS: f32 = 0.15;

/// How many upcoming songs are resolved ahead of time, so skipping to them starts at once.
const PRELOAD_AHEAD: usize = 3;

/// How long a cover that failed to arrive waits before it may be asked for again.
const COVER_RETRY: Duration = Duration::from_secs(20);

/// How often the lists on screen are scanned for songs with no length, when no news has come.
const LENGTHS_EVERY: Duration = Duration::from_millis(250);

/// What the screen wants done, applied after the frame has been built.
enum Action {
    Search,
    Play(usize),
    PlayQueued(usize),
    Toggle,
    Seek(f32),
    Next,
    Back,
    Tab(Tab),
    ToggleSidebar,
    ToggleLyrics,
    Mute,
    Window(WindowRequest),
    /// Open an artist's or album's page.
    Open(Detail),
    /// Back from a page: close the lyric page, or return to the page before.
    PageBack,
    /// Play a track from the page on screen.
    PlayDetail(usize),
    /// Play the search's top result, when it is a song.
    PlayTop,
    /// Save a song, or unsave it if it is saved.
    SaveSong(yt_client::SongHit),
    /// Save an album (and every song on it), or unsave it.
    SaveAlbum(AlbumPage),
    /// Save an artist's page, or unsave it.
    SaveArtist(ArtistPage),
    /// Play from the Collection's songs, starting at this position in the list.
    PlayLibrary(usize),
    /// Switch to another list inside the Collection page.
    CollectionTab(CollectionTab),
    /// Switch to another kind of result (songs, artists, albums) under the search's top result.
    ResultsTab(ResultsTab),
    /// Move a queue entry to another place in the queue.
    MoveQueued { from: usize, to: usize },
    /// Take a song out of the queue.
    RemoveQueued(usize),
    /// Add a song to the end of the queue.
    Enqueue(yt_client::SongHit),
    /// Open the menu for a track, at the pointer.
    OpenMenu(TrackMenu),
}

/// Something the window itself has to do, asked for by the caption buttons or by a press on the
/// top strip (the window has no OS title bar, so the app does these). The host takes them after
/// each frame, see `main.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRequest {
    Minimize,
    Close,
    /// Start moving the window: the press was on empty space in the top strip.
    Drag,
}

pub struct State {
    pub backend: Backend,
    pub query: String,
    /// The songs of the search on screen.
    pub results: Vec<yt_client::SongHit>,
    /// The search's top result, shown above the songs, and the artists and albums it matched.
    pub top: Option<TopResult>,
    pub result_artists: Vec<ArtistCard>,
    pub result_albums: Vec<AlbumCard>,
    /// Up next, and what played before: see [`crate::playlist`].
    pub playlist: Playlist,
    /// The lyrics of the track that is playing.
    lyrics: Option<LyricsView>,
    pub now: Option<Now>,
    pub status: String,
    pub booted: bool,
    /// 0..=1 of the current track, bound to the seek bar.
    pub seek: f32,
    pub volume: f32,
    /// Set by the host when Enter is pressed.
    pub submit: bool,
    /// Set by the host for Ctrl+K and `/`: reveal the search field and focus it.
    pub focus_search: bool,
    /// Set by the host for Space while no field is being edited.
    pub toggle_request: bool,
    pub sidebar_open: bool,
    /// The pointer is over something clickable; the host shows a hand cursor.
    pub hand: bool,
    /// Icon and cover textures; the host makes the ones a frame asked for.
    pub gpu: Gpu,
    /// Window actions asked for since the host last took them (see [`WindowRequest`]).
    pub window_requests: Vec<WindowRequest>,

    tab: Tab,
    /// Which list the Collection page is showing.
    collection_tab: CollectionTab,
    /// Which kind of result the search page is showing: songs, artists or albums.
    results_tab: ResultsTab,
    /// What the listener saved, and the files behind it.
    library: Library,
    /// Songs unsaved from the Collection's song list since it was opened. They stay in the list,
    /// showing the plus again, until the reader leaves it, so an unsave can be undone in place.
    ghosts: HashSet<String>,
    anims: Anims,
    /// A search is in flight; the host keeps polling for the reply.
    searching: bool,
    /// Play the first result as soon as it arrives (`V1_AUTOPLAY`).
    autoplay: bool,
    /// Players are opened on loading threads and reported back here; see [`State::start`].
    opened_tx: Sender<Opened>,
    opened_rx: Receiver<Opened>,
    drag: Option<SliderId>,
    /// The queue row being dragged by its handle, if any.
    queue_drag: Option<QueueDrag>,
    results_query: String,
    results_at: Instant,
    /// The results to return to from the page opened over them, if the reader stayed on them for a
    /// while or clicked into something.
    results_return: Option<ResultsReturn>,
    /// Where the results list was scrolled to, as of the last frame it was on screen.
    results_scroll: f32,
    /// The reader clicked a song on the results since they last left them.
    results_clicked: bool,
    /// A scroll position to restore on the results, once they are back on screen.
    pending_results_scroll: Option<f32>,
    /// Search-as-you-type: the text last seen, when it changed, and whether a search is owed.
    query_seen: String,
    edited_at: Instant,
    query_dirty: bool,
    /// The track the bar is showing, and when it changed (drives the bar's crossfade).
    shown_id: Option<String>,
    track_at: Instant,
    /// The spinner's angle, in radians, while a song loads.
    spin: f32,
    /// Whether the bar drew the spinner last frame, so leaving it can land the glyph at once.
    glyph_loading: bool,
    search_wanted: bool,
    /// The text the bar held when a page change cleared it, kept on screen while the bar slides away.
    search_ghost: Option<String>,
    volume_open: bool,
    /// The lyric page is open. It is a page of its own: it takes the place of the tab's content.
    lyrics_open: bool,
    lyrics_follow: LyricsFollow,
    /// The pages a link opened. The one on top takes the place of the results or the tab.
    pages: Pages,
    /// Artist and album pages fetched so far, by browse id. Kept, so going back is instant.
    artists: HashMap<String, Loaded<ArtistPage>>,
    albums: HashMap<String, Loaded<AlbumPage>>,
    /// The spectrum behind the lyric page, and whether it is still moving (rising or settling).
    spectrum: Spectrum,
    spectrum_live: bool,
    volume_before_mute: Option<f32>,
    cover_requested: HashSet<String>,
    /// Covers asked for whose answer (a picture, or a failure) has not come back yet. The host keeps
    /// polling only while this is not empty, so a cover that can never arrive cannot hold it awake.
    cover_inflight: HashSet<String>,
    /// Covers that failed, and when they may be asked for again.
    cover_retry: HashMap<String, Instant>,
    /// The query of the search most recently sent, so an answer to an older one is not shown.
    last_search: String,
    /// When the lengths still missing were last looked for.
    lengths_at: Instant,
    /// Lengths fetched for tracks whose list row gave none, as `m:ss`, keyed by video id.
    lengths: HashMap<String, String>,
    length_requested: HashSet<String>,
    /// Length lookups that came back empty: how many times, and when to try again.
    length_failures: HashMap<String, (u32, Instant)>,
    /// Upcoming songs already asked to be resolved, so each is asked for once.
    preloaded: HashSet<String>,
    /// When the fade last moved, so each frame steps by its own length.
    tick_at: Instant,
    /// True while the bar's gain is still moving, so the host keeps presenting frames.
    fading: bool,
    animating: bool,
    /// A stand-in track for `--shot`, so the bar can be photographed without a network.
    demo_track: Option<TrackView>,
    /// The track menu, while it is open (see [`TrackMenu`]).
    menu: Option<TrackMenu>,
}

/// A player that finished opening on a loading thread, with the track it was opened for.
struct Opened {
    ready: Ready,
    player: Result<dream_media::Player, String>,
}

/// What the bar needs to know about the current track.
#[derive(Clone)]
struct TrackView {
    video_id: String,
    /// The song as the lists know it, when it is known (the bar's save button needs it).
    hit: Option<yt_client::SongHit>,
    title: String,
    artist: String,
    artist_id: Option<String>,
    thumbnail: Option<String>,
    position: f32,
    duration: f32,
    paused: bool,
    /// The song is still loading: the bar shows its details, and the play button is held.
    loading: bool,
    /// Low, mid and high band levels of the sound being output, 0 to 1.
    levels: [f32; 3],
}

/// One row of a track list. It borrows from the song it shows: rows are rebuilt every frame, and
/// copying every string of every song each time would be most of what a frame costs.
struct Row<'a> {
    /// The song itself, whole, for saving: the page a row is shown on may leave parts out.
    hit: &'a yt_client::SongHit,
    /// The album page this row is shown on, if any. A song saved from it takes what the page knows
    /// and the song left out (see [`Row::song`]).
    page: Option<&'a AlbumPage>,
    video_id: &'a str,
    title: &'a str,
    artist: std::borrow::Cow<'a, str>,
    album: &'a str,
    duration: &'a str,
    thumbnail: Option<&'a str>,
    artist_id: Option<&'a str>,
    album_id: Option<&'a str>,
}

/// The menu open over a track, from a right-click on its row. It keeps the song itself, so it still
/// works after the list it came from has scrolled or changed.
#[derive(Clone)]
struct TrackMenu {
    hit: yt_client::SongHit,
    /// Where the pointer was when the menu opened. The menu's top-left corner goes here, pulled
    /// back inside the window if it would run off the edge.
    at: P,
}

impl<'a> Row<'a> {
    /// A row for `hit`. Its length is the one the list gave, else one fetched for it (`lengths`).
    fn from(hit: &'a yt_client::SongHit, lengths: &'a HashMap<String, String>) -> Row<'a> {
        Row {
            hit,
            page: None,
            video_id: &hit.video_id,
            title: &hit.title,
            artist: match hit.artists.as_slice() {
                [] => std::borrow::Cow::Borrowed(""),
                [only] => std::borrow::Cow::Borrowed(only.as_str()),
                several => std::borrow::Cow::Owned(several.join(", ")),
            },
            album: hit.album.as_deref().unwrap_or_default(),
            duration: hit
                .duration
                .as_deref()
                .or_else(|| lengths.get(&hit.video_id).map(String::as_str))
                .unwrap_or_default(),
            thumbnail: hit.thumbnail.as_deref(),
            artist_id: hit.artist_id.as_deref(),
            album_id: hit.album_id.as_deref(),
        }
    }

    /// The song as it is saved or queued from this row: a copy, with what the album page knows
    /// filled in where the song left it out.
    fn song(&self) -> yt_client::SongHit {
        let mut hit = self.hit.clone();
        if let Some(page) = self.page {
            hit.album.get_or_insert_with(|| page.title.clone());
            hit.album_id.get_or_insert_with(|| page.browse_id.clone());
            if hit.thumbnail.is_none() {
                hit.thumbnail = page.thumbnail.clone();
            }
        }
        hit
    }
}

impl State {
    /// A state with an empty collection. The host opens the real one on another thread, while the
    /// window and the GPU come up, and hands it over with [`State::attach_library`] before the
    /// first frame.
    pub fn new(data_dir: std::path::PathBuf) -> State {
        let library = Library::in_memory();
        let backend = Backend::start(data_dir);
        let (opened_tx, opened_rx) = mpsc::channel();
        let now = Instant::now();
        State {
            backend,
            query: String::new(),
            results: Vec::new(),
            top: None,
            result_artists: Vec::new(),
            result_albums: Vec::new(),
            playlist: Playlist::default(),
            lyrics: None,
            now: None,
            status: String::new(),
            booted: false,
            seek: 0.0,
            volume: 0.8,
            submit: false,
            focus_search: false,
            toggle_request: false,
            sidebar_open: true,
            hand: false,
            gpu: Gpu::default(),
            window_requests: Vec::new(),
            tab: Tab::Library,
            collection_tab: CollectionTab::Songs,
            results_tab: ResultsTab::Songs,
            library,
            ghosts: HashSet::new(),
            anims: Anims::default(),
            searching: false,
            autoplay: false,
            opened_tx,
            opened_rx,
            drag: None,
            queue_drag: None,
            results_query: String::new(),
            results_at: now,
            results_return: None,
            results_scroll: 0.0,
            results_clicked: false,
            pending_results_scroll: None,
            query_seen: String::new(),
            edited_at: now,
            query_dirty: false,
            shown_id: None,
            track_at: now,
            spin: 0.0,
            glyph_loading: false,
            search_wanted: false,
            search_ghost: None,
            volume_open: false,
            lyrics_open: false,
            lyrics_follow: LyricsFollow::default(),
            pages: Pages::default(),
            artists: HashMap::new(),
            albums: HashMap::new(),
            volume_before_mute: None,
            cover_requested: HashSet::new(),
            cover_inflight: HashSet::new(),
            cover_retry: HashMap::new(),
            last_search: String::new(),
            lengths_at: now,
            lengths: HashMap::new(),
            length_requested: HashSet::new(),
            length_failures: HashMap::new(),
            preloaded: HashSet::new(),
            tick_at: now,
            fading: false,
            animating: false,
            spectrum: Spectrum::new(),
            spectrum_live: false,
            demo_track: None,
            menu: None,
        }
    }

    /// Take up the collection read from disk. Nothing has used the empty one yet: this runs before
    /// the first frame, so the backend's `Ready` (which resumes unfinished downloads) finds it.
    pub fn attach_library(&mut self, library: Library) {
        self.library = library;
    }

    /// Close the track menu, if one is open. Escape does this.
    pub fn close_menu(&mut self) {
        self.menu = None;
    }

    /// Drain backend events. Called once per frame, before the build.
    pub fn pump(&mut self) {
        // Covers are a few hundred KB each on the GPU. Past a few hundred, start over: frames in
        // flight hold their own references, and whatever is on screen is simply fetched again.
        if self.gpu.covers.len() > 400 {
            self.gpu.covers.clear();
            // Covers still on their way are kept: their answers will arrive and are uploaded as
            // usual, and asking for them again would only fetch them twice.
            self.cover_requested.clone_from(&self.cover_inflight);
            self.cover_retry.clear();
        }
        if self.artists.len() > PAGE_CACHE || self.albums.len() > PAGE_CACHE {
            self.trim_pages();
        }
        if self.preloaded.len() > 512 {
            self.preloaded.clear();
        }
        let mut heard = false;
        while let Ok(event) = self.backend.events.try_recv() {
            heard = true;
            match event {
                Event::Ready => {
                    crate::boot_mark("network ready (search works)");
                    self.booted = true;
                    self.status.clear();
                    self.resume_library();
                }
                Event::Fatal(message) => self.status = message,
                Event::Results(query, found) => {
                    // Only the newest search counts. An older one that was already on its way
                    // would otherwise replace the results of what the reader typed since.
                    if query != self.last_search {
                        continue;
                    }
                    self.searching = false;
                    if query == self.query {
                        self.status.clear();
                    }
                    let mut seen: Vec<yt_client::SongHit> = found.songs.clone();
                    if let Some(TopResult::Song(hit)) = &found.top {
                        seen.push(hit.clone());
                    }
                    self.learn(&seen);
                    self.results = found.songs;
                    self.top = found.top;
                    self.result_artists = found.artists;
                    self.result_albums = found.albums;
                    self.results_query = query;
                    self.results_tab = ResultsTab::Songs;
                    self.results_at = Instant::now();
                    if self.autoplay {
                        // The first thing the search shows: the top song, else the first song.
                        if let Some(TopResult::Song(_)) = &self.top {
                            self.autoplay = false;
                            self.play_top();
                        } else if !self.results.is_empty() {
                            self.autoplay = false;
                            self.play_index(0);
                        }
                    }
                }
                Event::SearchFailed(query, message) => {
                    if query == self.last_search {
                        self.searching = false;
                        self.status = message;
                    }
                }
                // A load that was superseded still reports back; only the one being waited on counts.
                Event::Playable(ready) => {
                    if self.playlist.loading_id() == Some(ready.video_id.as_str()) {
                        self.start(ready);
                    }
                }
                Event::PlayFailed(video_id, message) => {
                    if self.playlist.load_failed(&video_id) {
                        self.status = message;
                    }
                }
                Event::Cover { url, width, height, rgba } => {
                    self.cover_inflight.remove(&url);
                    self.gpu.incoming.push(CoverImage { url, width, height, rgba });
                }
                Event::CoverFailed(url) => {
                    // Forgotten as requested, but not for a while: the next frame that wants it
                    // would ask again at once, and a cover that cannot be had would be asked for
                    // every frame. Asking again after the wait is how it recovers from a network
                    // that was down.
                    self.cover_inflight.remove(&url);
                    self.cover_requested.remove(&url);
                    self.cover_retry.insert(url, Instant::now() + COVER_RETRY);
                }
                // Only the answer for the track on screen counts; an older one is dropped.
                Event::Lyrics { video_id, lookup } => {
                    if let Some(view) = self.lyrics.as_mut().filter(|v| v.video_id == video_id) {
                        view.lookup = Some(lookup);
                        view.arrived = Instant::now();
                    }
                }
                Event::Artist { id, result } => {
                    // A saved artist's stored page is kept current with whatever arrives.
                    if let Ok(page) = &result {
                        self.library.refresh_artist(page);
                        self.learn(&page.songs);
                    }
                    // A quiet refresh that fails leaves the page already on screen alone.
                    let showing = matches!(self.artists.get(&id), Some(Loaded::Ready(_)));
                    if result.is_ok() || !showing {
                        self.artists.insert(id, loaded(result));
                    }
                }
                Event::Album { id, result } => {
                    // A saved album stored without its tracks (pages once came back empty) is
                    // filled in, and its songs saved, once a full page arrives.
                    if let Ok(page) = &result {
                        self.learn(&page.tracks);
                        let stored_empty =
                            self.library.album_page(&id).is_some_and(|p| p.tracks.is_empty());
                        if stored_empty && !page.tracks.is_empty() {
                            let effects = self.library.save_album(page, Instant::now());
                            self.run_effects(effects);
                        }
                    }
                    self.albums.insert(id, loaded(result));
                }
                Event::Length { video_id, seconds } => match seconds {
                    Some(seconds) => {
                        let text = crate::library::length_text(seconds as f64);
                        self.length_failures.remove(&video_id);
                        self.library.set_length(&video_id, &text);
                        if let Some(track) = self.now.as_mut().filter(|t| t.video_id == video_id) {
                            if track.duration <= 0.0 {
                                track.duration = seconds as f32;
                            }
                        }
                        self.lengths.insert(video_id, text);
                    }
                    None => {
                        // Throttled or refused: asked for again later, a few times at most.
                        self.length_requested.remove(&video_id);
                        let tries = self.length_failures.get(&video_id).map_or(0, |(n, _)| *n) + 1;
                        let wait = Duration::from_secs(5u64 << tries.min(6));
                        self.length_failures.insert(video_id, (tries, Instant::now() + wait));
                    }
                },
                Event::Downloaded { video_id, loudness_db, duration_secs } => {
                    self.library.downloaded(&video_id, loudness_db, duration_secs);
                }
                // Saving is quiet: a failed download is not announced, only tried again later.
                Event::DownloadFailed { video_id, message } => {
                    tracing::debug!("download of {video_id} failed: {message}");
                    self.library.download_failed(&video_id, Instant::now());
                }
            }
        }
        // The scan for missing lengths walks every list on screen and the whole collection, so it
        // runs when something arrived that could add to them, or a few times a second otherwise
        // (a page opened from memory brings no event), not on every frame.
        if heard || self.lengths_at.elapsed() >= LENGTHS_EVERY {
            self.lengths_at = Instant::now();
            self.fill_lengths();
        }
        self.preload_upcoming();
        while let Ok(opened) = self.opened_rx.try_recv() {
            self.finish_start(opened);
        }
    }

    /// Resolve the next few songs in the queue, one request each, so skipping to them starts
    /// without the wait. Only the queue's head is looked at, so a song is asked for once and the
    /// list never grows past what is about to play.
    fn preload_upcoming(&mut self) {
        for index in 0..PRELOAD_AHEAD.min(self.playlist.queue.len()) {
            let video_id = &self.playlist.queue[index].video_id;
            // Asked about once. (This runs every frame, so the common case, a song already dealt
            // with, must cost a lookup and nothing more.)
            if self.preloaded.contains(video_id) {
                continue;
            }
            let video_id = video_id.clone();
            self.preloaded.insert(video_id.clone());
            // A saved song plays from its file, so there is nothing to resolve ahead of time.
            if self.library.local(&video_id).is_some() {
                continue;
            }
            let _ = self.backend.commands.send(Command::Preload(video_id));
        }
    }

    /// Carry out what a change to the collection asked for: downloads and kept covers.
    fn run_effects(&mut self, effects: Effects) {
        for job in effects.downloads {
            let _ = self.backend.commands.send(Command::Download(job));
        }
        for url in effects.covers {
            let _ = self.backend.commands.send(Command::KeepCover(url));
        }
    }

    /// The backend is up: finish any download that was left undone, and quietly refresh a few of
    /// the saved artists whose stored pages have gone stale.
    fn resume_library(&mut self) {
        let effects = self.library.resume();
        self.run_effects(effects);
        for id in self.library.vault.stale(crate::vault::REFRESH_PER_LAUNCH) {
            let _ = self.backend.commands.send(Command::Artist(id));
        }
    }

    /// When the collection next has something to do on its own, so the host wakes for it.
    pub fn library_wake_at(&self) -> Option<Instant> {
        self.library.next_wake()
    }

    /// Fill in what the songs we hold are missing (album, length, artist link) from fresher copies
    /// that a search or a page just brought in: the saved songs, the queue, and the song playing.
    fn learn(&mut self, hits: &[yt_client::SongHit]) {
        if hits.is_empty() {
            return;
        }
        self.library.fill_in(hits);
        self.playlist.fill_in(hits);
        if let Some(track) = self.now.as_mut() {
            if let Some(other) = hits.iter().find(|hit| hit.video_id == track.video_id) {
                crate::library::fill_gaps(&mut track.hit, other);
                if track.artist_id.is_none() {
                    track.artist_id = track.hit.artist_id.clone();
                }
                if track.duration <= 0.0 {
                    track.duration = hit_seconds(track.hit.duration.as_deref());
                }
            }
        }
    }

    /// Ask for the length of every track on screen that has none yet, once each. Covers the
    /// results, the queue and the open artist or album page.
    fn fill_lengths(&mut self) {
        let mut hits: Vec<&yt_client::SongHit> = self.results.iter().collect();
        hits.extend(self.playlist.queue.iter());
        match &self.pages.top {
            Some(Detail::Artist(id)) => {
                if let Some(Loaded::Ready(page)) = self.artists.get(id) {
                    hits.extend(page.songs.iter());
                }
            }
            Some(Detail::Album(id)) => {
                if let Some(Loaded::Ready(page)) = self.albums.get(id) {
                    hits.extend(page.tracks.iter());
                }
            }
            None => {}
        }
        hits.extend(self.playlist.loading_song());
        hits.extend(self.now.as_ref().map(|track| &track.hit));
        hits.extend(self.library.song_hits());
        let now = Instant::now();
        let missing: Vec<String> = hits
            .into_iter()
            .filter(|hit| hit.duration.is_none() && !self.lengths.contains_key(&hit.video_id))
            .filter(|hit| {
                self.length_failures
                    .get(&hit.video_id)
                    .is_none_or(|(tries, retry_at)| *tries < LENGTH_TRIES && now >= *retry_at)
            })
            .map(|hit| hit.video_id.clone())
            .collect();
        for video_id in missing {
            if self.length_requested.insert(video_id.clone()) {
                let _ = self.backend.commands.send(Command::Length(video_id));
            }
        }
    }

    /// Advance the player. Called once per frame, before the build.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let effects = self.library.tick(now);
        self.run_effects(effects);
        let mut finished = false;
        let mut failed = None;
        let step = now.duration_since(self.tick_at).as_secs_f32().min(0.1) / FADE_SECS;
        self.tick_at = now;
        self.fading = false;
        if let Some(track) = self.now.as_mut() {
            let _ = track.player.poll(now);
            track.position = (track.player.position_ms(now) as f32 / 1000.0).max(0.0);
            if let Some(duration) = track.player.duration_ms() {
                track.duration = (duration as f32 / 1000.0).max(0.0);
            }
            match track.player.status() {
                dream_media::Status::Ended => finished = true,
                dream_media::Status::Failed(message) => failed = Some(message),
                dream_media::Status::Playing | dream_media::Status::Paused | dream_media::Status::Loading => {}
            }
            // The fade: the gain moves toward what was asked for, and a pause lands only once it is
            // silent. The bar shows the request at once, not the player's lagging state.
            let goal = if track.want_playing { 1.0 } else { 0.0 };
            if track.gain != goal {
                track.gain = if track.gain < goal {
                    (track.gain + step).min(goal)
                } else {
                    (track.gain - step).max(goal)
                };
                self.fading = true;
            }
            track.player.set_volume(self.volume * track.gain * track.norm);
            if !track.want_playing && track.gain <= 0.0 && !track.held {
                track.player.pause(now);
                track.held = true;
            }
            track.paused = !track.want_playing;
        }
        if let Some(message) = failed {
            self.status = message;
            self.now = None;
            self.playlist.stopped();
        }
        // While a load is pending, the ended track stays put: the load replaces it and files it
        // in the history. Otherwise the ended song is filed now and the queue moves on.
        if finished && !self.playlist.is_loading() {
            self.now = None;
            match self.playlist.finished() {
                Some(hit) => self.send_play(hit),
                None => self.status.clear(),
            }
        }
    }

    pub fn is_playing(&self) -> bool {
        self.now.as_ref().is_some_and(|track| !track.paused)
    }

    /// Play the first search result automatically; used by the screenshot harness.
    pub fn autoplay_first(&mut self) {
        self.autoplay = true;
    }

    /// True while the UI is waiting on the backend, so the host keeps polling for the reply.
    pub fn busy(&self) -> bool {
        !self.booted || self.searching || self.playlist.is_loading()
    }

    /// True while something on screen is still moving, so the host keeps presenting frames.
    pub fn animating(&self) -> bool {
        self.animating
    }

    /// When the lyric page will take its place back after a reader's scroll, so the host wakes then
    /// even if nothing else is moving. `None` when no such wait is pending.
    pub fn lyrics_resume_at(&self) -> Option<Instant> {
        let follow = &self.lyrics_follow;
        if !self.lyrics_open || !follow.reader {
            return None;
        }
        follow
            .scrolled_at
            .map(|at| at + std::time::Duration::from_secs_f32(LYRIC_IDLE_SECS))
    }

    /// Switch tabs as if the user had clicked one; used by the screenshot harness to film the
    /// sidebar's selection.
    pub fn shot_switch_tab(&mut self, name: &str) {
        if name == "lyrics" {
            self.set_lyrics_open(true);
            return;
        }
        let tab = match name {
            "library" => Tab::Library,
            "queue" => Tab::Queue,
            _ => Tab::Library,
        };
        self.apply(Action::Tab(tab));
    }

    /// True while covers are on their way, so the host keeps polling for them.
    pub fn awaiting_covers(&self) -> bool {
        !self.cover_inflight.is_empty() || !self.gpu.incoming.is_empty()
    }

    /// Sample content for `--shot`, so a screenshot shows the real layout. Timestamps are aged so
    /// entrance animations have already finished.
    pub fn demo(&mut self) {
        let long_ago = Instant::now()
            .checked_sub(Duration::from_secs(30))
            .unwrap_or_else(Instant::now);
        self.results_at = long_ago;
        self.track_at = long_ago;
        self.query = "daft punk".to_owned();
        self.query_seen = self.query.clone();
        self.status.clear();
        self.booted = true;
        let songs = [
            ("Around the World", "Daft Punk", "Homework", "7:09"),
            ("Instant Crush", "Daft Punk, Julian Casablancas", "Random Access Memories", "5:38"),
            ("One More Time", "Daft Punk", "Discovery", "5:20"),
            ("Harder, Better, Faster, Stronger", "Daft Punk", "Discovery", "3:45"),
            ("Something About Us", "Daft Punk", "Discovery", "3:52"),
            ("Veridis Quo", "Daft Punk", "Discovery", "5:45"),
            ("Digital Love", "Daft Punk", "Discovery", "5:01"),
            ("Aerodynamic", "Daft Punk", "Discovery", "3:27"),
            ("Get Lucky", "Daft Punk, Pharrell Williams, Nile Rodgers", "Random Access Memories", "6:10"),
            ("Da Funk", "Daft Punk", "Homework", "5:28"),
        ];
        self.results = songs
            .iter()
            .enumerate()
            .map(|(index, (title, artist, album, duration))| yt_client::SongHit {
                video_id: format!("demo{index}"),
                title: (*title).to_owned(),
                artists: artist.split(", ").map(str::to_owned).collect(),
                album: Some((*album).to_owned()),
                duration: Some((*duration).to_owned()),
                // `V1_SHOT_COVERS` stands in plain tiles for real covers, to check layout offline.
                thumbnail: std::env::var_os("V1_SHOT_COVERS").map(|_| format!("demo://{index}")),
                artist_id: None,
                album_id: None,
            })
            .collect();
        if std::env::var_os("V1_SHOT_COVERS").is_some() {
            for index in 0..songs.len() {
                let hue = index as f32 / songs.len() as f32;
                let channel = |offset: f32| {
                    let t = (hue + offset).fract();
                    (80.0 + 140.0 * (0.5 + 0.5 * (t * std::f32::consts::TAU).sin())) as u8
                };
                let (r, g, b) = (channel(0.0), channel(0.33), channel(0.66));
                let rgba = (0..64 * 64).flat_map(|_| [r, g, b, 255]).collect();
                self.gpu.incoming.push(CoverImage {
                    url: format!("demo://{index}"),
                    width: 64,
                    height: 64,
                    rgba,
                });
                self.cover_requested.insert(format!("demo://{index}"));
            }
        }
        // The search's top result and the artists and albums it matched, invented to match the songs.
        let thumb = |index: usize| std::env::var_os("V1_SHOT_COVERS").map(|_| format!("demo://{index}"));
        self.top = Some(TopResult::Artist(ArtistCard {
            browse_id: "UCdemo-top".to_owned(),
            name: "Daft Punk".to_owned(),
            meta: "68.2M monthly audience".to_owned(),
            thumbnail: thumb(0),
        }));
        self.result_artists = (1..=4)
            .map(|index| ArtistCard {
                browse_id: format!("UCdemo{index}"),
                name: format!("Demo Artist {index}"),
                meta: "1.2M monthly audience".to_owned(),
                thumbnail: thumb(index),
            })
            .collect();
        self.result_albums = demo_cards("Album", 5);
        self.results_query = self.query.clone();
        self.playlist.queue = self.results.iter().skip(3).cloned().collect();
        if std::env::var_os("V1_SHOT_EMPTY").is_some() {
            self.results.clear();
            self.top = None;
            self.result_artists.clear();
            self.result_albums.clear();
            self.query.clear();
            self.query_seen.clear();
        }
        if std::env::var_os("V1_SHOT_PLAYING").is_some() {
            self.shown_id = Some("demo1".to_owned());
            self.demo_track = Some(TrackView {
                video_id: "demo1".to_owned(),
                hit: self.results.get(1).cloned(),
                title: "Instant Crush (feat. Julian Casablancas)".to_owned(),
                artist: "Daft Punk, Julian Casablancas".to_owned(),
                artist_id: None,
                thumbnail: None,
                position: 71.0,
                duration: 338.0,
                paused: std::env::var_os("V1_SHOT_PAUSED").is_some(),
                loading: false,
                levels: [0.8, 0.95, 0.6],
            });
        }
        if std::env::var_os("V1_SHOT_COLLAPSED").is_some() {
            self.sidebar_open = false;
        }
        // `V1_SHOT_SAVED=1` marks a few of the songs as saved, to see both states of the button.
        // `V1_SHOT_COLLECTION=songs|artists|albums` fills an in-memory collection and opens that
        // list. Nothing touches the disk.
        let collection = std::env::var("V1_SHOT_COLLECTION").ok();
        if collection.is_some() || std::env::var_os("V1_SHOT_SAVED").is_some() {
            self.library = Library::in_memory();
            let now = Instant::now();
            for index in [1, 3, 4] {
                if let Some(hit) = self.results.get(index).cloned() {
                    let _ = self.library.save_song(&hit, now);
                }
            }
            if collection.is_some() {
                for index in 0..14 {
                    let _ = self.library.save_artist(&ArtistPage {
                        browse_id: format!("UCdemo{index}"),
                        name: format!("Demo Artist {}", index + 1),
                        thumbnail: std::env::var_os("V1_SHOT_COVERS").map(|_| format!("demo://{}", index % 10)),
                        ..ArtistPage::default()
                    });
                    let _ = self.library.save_album(
                        &AlbumPage {
                            browse_id: format!("MPREb_demo{index}"),
                            title: format!("Demo Album {}", index + 1),
                            artist: format!("Demo Artist {}", index + 1),
                            thumbnail: std::env::var_os("V1_SHOT_COVERS").map(|_| format!("demo://{}", index % 10)),
                            ..AlbumPage::default()
                        },
                        now,
                    );
                }
                self.collection_tab = match collection.as_deref() {
                    Some("artists") => CollectionTab::Artists,
                    Some("albums") => CollectionTab::Albums,
                    _ => CollectionTab::Songs,
                };
                self.tab = Tab::Library;
                self.results.clear();
                self.top = None;
                self.result_artists.clear();
                self.result_albums.clear();
                self.results_query.clear();
                self.query.clear();
                self.query_seen.clear();
            }
        }
        match std::env::var("V1_SHOT_TAB").as_deref() {
            Ok("library") => self.tab = Tab::Library,
            Ok("queue") => self.tab = Tab::Queue,
            Ok("lyrics") => self.set_lyrics_open(true),
            _ => {}
        }
        // `V1_SHOT_PAGE=artist|album` opens a stand-in artist or album page over the results, to check
        // its layout. The names are invented.
        match std::env::var("V1_SHOT_PAGE").as_deref() {
            Ok("artist") => self.show_demo_detail(Detail::Artist("demo-artist".to_owned())),
            Ok("album") => self.show_demo_detail(Detail::Album("demo-album".to_owned())),
            _ => {}
        }
        // `V1_SHOT_LYRICS=synced|plain|...` stands in lyrics for the demo track, to check the layout.
        if let Ok(kind) = std::env::var("V1_SHOT_LYRICS") {
            self.lyrics = Some(LyricsView {
                video_id: "demo1".to_owned(),
                lookup: Some(demo_lookup(&kind)),
                gen: 0,
                // Backdated, so the entrance has already finished in a still.
                arrived: Instant::now()
                    .checked_sub(Duration::from_secs(30))
                    .unwrap_or_else(Instant::now),
            });
        }
    }

    /// Put a stand-in page over the demo results, with the results underneath it.
    fn show_demo_detail(&mut self, detail: Detail) {
        let thumbnail = std::env::var_os("V1_SHOT_COVERS").map(|_| "demo://0".to_owned());
        let tracks: Vec<yt_client::SongHit> = self.results.iter().take(8).cloned().collect();
        match &detail {
            Detail::Artist(id) => {
                self.artists.insert(
                    id.clone(),
                    Loaded::Ready(ArtistPage {
                        browse_id: id.clone(),
                        name: "Demo Artist".to_owned(),
                        thumbnail,
                        songs_title: "Top songs".to_owned(),
                        songs: tracks,
                        shelves: vec![
                            yt_client::Shelf { title: "Albums".to_owned(), albums: demo_cards("Album", 7) },
                            yt_client::Shelf {
                                title: "Singles & EPs".to_owned(),
                                albums: demo_cards("Single", 3),
                            },
                        ],
                    }),
                );
            }
            Detail::Album(id) => {
                self.albums.insert(
                    id.clone(),
                    Loaded::Ready(AlbumPage {
                        browse_id: id.clone(),
                        title: "Demo Album".to_owned(),
                        artist: "Demo Artist".to_owned(),
                        artist_id: Some("demo-artist".to_owned()),
                        meta: "Album  ·  2001  ·  8 songs  ·  40 minutes".to_owned(),
                        thumbnail,
                        tracks,
                    }),
                );
            }
        }
        self.pages.open(detail);
    }

    fn track_view(&self) -> Option<TrackView> {
        // A song being loaded shows its own details at once. The audible song keeps playing
        // underneath until the new one starts, and the play button is held until then.
        if let Some(hit) = self.playlist.loading_song() {
            return Some(TrackView {
                video_id: hit.video_id.clone(),
                hit: Some(hit.clone()),
                title: hit.title.clone(),
                artist: hit.artists.join(", "),
                artist_id: hit.artist_id.clone(),
                thumbnail: hit.thumbnail.clone(),
                position: 0.0,
                duration: self.known_length(hit),
                paused: true,
                loading: true,
                levels: [0.0; 3],
            });
        }
        match &self.now {
            Some(track) => Some(TrackView {
                video_id: track.video_id.clone(),
                hit: Some(track.hit.clone()),
                title: track.title.clone(),
                artist: track.artist.clone(),
                artist_id: track.artist_id.clone(),
                thumbnail: track.thumbnail.clone(),
                position: track.position,
                duration: track.duration,
                paused: track.paused,
                loading: false,
                levels: track.player.levels(),
            }),
            None => self.demo_track.clone(),
        }
    }

    /// A song's length in seconds as far as the app knows it: from its list, else fetched. 0 when
    /// unknown.
    fn known_length(&self, hit: &yt_client::SongHit) -> f32 {
        let listed = hit_seconds(hit.duration.as_deref());
        if listed > 0.0 {
            return listed;
        }
        hit_seconds(self.lengths.get(&hit.video_id).map(String::as_str))
    }

    fn position_fraction(&self) -> f32 {
        let (position, duration) = match (&self.now, &self.demo_track) {
            (Some(track), _) => (track.position, track.duration),
            (None, Some(track)) => (track.position, track.duration),
            _ => return 0.0,
        };
        if duration <= 0.0 {
            return 0.0;
        }
        (position / duration).clamp(0.0, 1.0)
    }

    /// Play a song picked from the search results. Only that song plays; the queue is left alone.
    fn play_index(&mut self, index: usize) {
        if let Some(hit) = self.results.get(index).cloned() {
            let hit = self.playlist.play_one(hit);
            self.send_play(hit);
        }
    }

    /// Play the top result, when it is a song. Only that song plays; the queue is left alone.
    fn play_top(&mut self) {
        let Some(TopResult::Song(hit)) = self.top.clone() else {
            return;
        };
        let hit = self.playlist.play_one(hit);
        self.send_play(hit);
    }

    fn send_play(&mut self, hit: yt_client::SongHit) {
        // The song that was playing is no longer the active one: it fades out now, and the new
        // song takes over when it opens.
        if let Some(track) = self.now.as_mut() {
            track.want_playing = false;
            track.paused = true;
        }
        self.status = format!("Loading {}", hit.title);
        // A saved song plays from its own file: no resolving, and no network.
        if let Some(local) = self.library.local(&hit.video_id) {
            let known = self.known_length(&hit);
            self.start(Ready {
                video_id: hit.video_id,
                url: local.path.to_string_lossy().into_owned(),
                artist: hit.artists.join(", "),
                duration_secs: local.duration_secs.or((known > 0.0).then_some(known as f64)),
                loudness_db: local.loudness_db,
            });
            return;
        }
        let _ = self.backend.commands.send(Command::Play(hit.video_id));
    }

    /// Next: skip whatever is loading and play the head of the queue.
    fn next(&mut self) {
        match self.playlist.next() {
            Some(hit) => self.send_play(hit),
            None => self.status.clear(),
        }
    }

    /// Back: go to the previous song, or restart the current one when it is well in or there is
    /// nothing before it.
    fn back(&mut self) {
        let past_start = self
            .now
            .as_ref()
            .is_some_and(|track| track.position > RESTART_AFTER_SECS);
        match self.playlist.back(past_start) {
            Some(hit) => self.send_play(hit),
            None => {
                self.status.clear();
                if let Some(track) = self.now.as_mut() {
                    track.player.seek(0);
                    track.position = 0.0;
                }
            }
        }
    }

    /// The requested song has resolved: open it. The playlist decides what the song that was
    /// playing becomes, and it is only filed once this one has actually started.
    /// Open the track's player on a loading thread. Opening probes the stream and starts the audio
    /// device, which takes a few hundred milliseconds; done on the frame loop it froze the window at
    /// every track change. The result comes back through [`Self::pump`], and the playlist keeps the
    /// track in its loading state until then.
    fn start(&mut self, ready: Ready) {
        let tx = self.opened_tx.clone();
        std::thread::spawn(move || {
            let options = dream_media::PlayerOptions {
                max_size: (0, 0),
                audio: true,
                looping: false,
                hwaccel: false,
            };
            // A stream from the loopback proxy is audio, and the backend already told us its length,
            // so there is nothing for a probe to find out: opening skips the ffprobe launch and the
            // second trip to the CDN that goes with it. A saved file is probed as before, since its
            // exact length is the decoder's.
            let streamed = ready.url.starts_with("http://127.0.0.1:");
            let player = if streamed {
                dream_media::Player::open_audio_unprobed(&ready.url, options, None)
            } else {
                dream_media::Player::open(&ready.url, options)
            }
            .map_err(|error| error.to_string());
            // Only lost if the app is closing, and then nothing needs the player.
            let _ = tx.send(Opened { ready, player });
        });
    }

    /// Take up a player that finished opening. A load that was superseded in the meantime is
    /// dropped here, as before.
    fn finish_start(&mut self, opened: Opened) {
        let Opened { ready, player } = opened;
        match player {
            Ok(mut player) => {
                // Starts silent: `tick` fades it up to the volume.
                player.set_volume(0.0);
                self.status.clear();
                let Some(hit) = self.playlist.started(&ready.video_id) else {
                    return;
                };
                // `/player` usually gives the length; when it does not, the list's or a fetched one
                // stands in until the decoder reads it from the stream.
                let known = self.known_length(&hit);
                let duration_secs =
                    ready.duration_secs.or((known > 0.0).then_some(known as f64));
                // Search results carry richer metadata than the player response: full artist
                // credits and the cover.
                let artist = if hit.artists.is_empty() {
                    ready.artist.clone()
                } else {
                    hit.artists.join(", ")
                };
                // Lyrics look the song up by its own title and primary artist, and by the exact
                // length this video plays, which is what keeps them off another cut of it.
                let _ = self.backend.commands.send(Command::Lyrics(Query {
                    video_id: ready.video_id.clone(),
                    title: hit.title.clone(),
                    artist: hit
                        .artists
                        .first()
                        .cloned()
                        .unwrap_or_else(|| ready.artist.clone()),
                    album: hit.album.clone(),
                    duration_secs,
                }));
                let gen = self.lyrics.as_ref().map_or(0, |v| v.gen.wrapping_add(1));
                self.lyrics = Some(LyricsView {
                    video_id: ready.video_id.clone(),
                    lookup: None,
                    gen,
                    arrived: Instant::now(),
                });
                self.now = Some(Now {
                    video_id: ready.video_id,
                    hit: hit.clone(),
                    title: hit.title,
                    artist,
                    artist_id: hit.artist_id,
                    thumbnail: hit.thumbnail,
                    position: 0.0,
                    duration: duration_secs.map_or(0.0, |secs| secs as f32),
                    paused: false,
                    want_playing: true,
                    gain: 0.0,
                    norm: normalisation(ready.loudness_db),
                    held: false,
                    player,
                });
            }
            Err(error) => {
                // A load that was superseded while it opened is not worth a message.
                if self.playlist.load_failed(&ready.video_id) {
                    self.status = format!("Could not open the stream: {error}");
                }
            }
        }
    }

    fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(track) = self.now.as_mut() {
            track.player.set_volume(self.volume * track.gain * track.norm);
        }
    }

    /// Going to another page (a tab, an artist or album, the lyric page) takes the reader off the
    /// search results, and clears the search bar with them. Search-as-you-type is stopped too, so
    /// the cleared text does not search again.
    fn leave_results(&mut self) {
        // Remember the results when the reader stayed on them for a while or clicked into them.
        let showing = !self.results_query.is_empty();
        let stayed = self.results_at.elapsed() >= RESULTS_STAY;
        self.results_return = (showing && (self.results_clicked || stayed)).then(|| ResultsReturn {
            query: self.query.clone(),
            results_query: self.results_query.clone(),
            scroll: self.results_scroll,
        });
        self.results_clicked = false;
        self.results_query.clear();
        // The bar slides away as it always does; its text goes with it instead of vanishing first.
        self.search_ghost = (!self.query.is_empty()).then(|| self.query.clone());
        self.query.clear();
        self.query_seen.clear();
        self.query_dirty = false;
        self.search_wanted = false;
    }

    /// Show or hide the lyric page. Opening it starts the page's placement afresh, so it centres on
    /// the current line at once, and forgets any scrolling the reader did before.
    fn set_lyrics_open(&mut self, open: bool) {
        if open && !self.lyrics_open {
            self.lyrics_follow = LyricsFollow::default();
        }
        self.lyrics_open = open;
    }

    /// Open an artist's or album's page over whatever is showing. Opening one leaves the lyric page,
    /// and back returns to what was showing before.
    fn open(&mut self, detail: Detail) {
        // Only leaving the results or the tab counts; a page opened over another page keeps the
        // results remembered beneath it.
        if self.pages.top.is_none() {
            self.leave_results();
        }
        self.lyrics_open = false;
        if self.pages.open(detail.clone()) {
            self.request_detail(&detail);
        }
    }

    /// Back: close the lyric page if it is showing, else return to the page before this one.
    fn page_back(&mut self) {
        if self.lyrics_open {
            self.lyrics_open = false;
        } else if self.pages.back() && self.pages.top.is_none() {
            self.restore_results();
        }
    }

    /// Back on the results (or the tab) from a page: bring the remembered results back, if any.
    fn restore_results(&mut self) {
        if let Some(ret) = self.results_return.take() {
            self.query_seen = ret.query.clone();
            self.query = ret.query;
            self.query_dirty = false;
            self.results_query = ret.results_query;
            self.pending_results_scroll = Some(ret.scroll);
        }
    }

    /// Ask for a page unless it is already here or on its way. A page that failed is asked for again.
    fn request_detail(&mut self, detail: &Detail) {
        match detail {
            Detail::Artist(id) => {
                if matches!(self.artists.get(id), None | Some(Loaded::Failed(_))) {
                    // A saved artist opens from its stored page: instant, and it works offline.
                    // It is still fetched quietly behind it, so a page stored by an older version
                    // (or one gone stale) is brought up to date while it is on screen.
                    if let Some(page) = self.library.vault.get(id) {
                        self.artists.insert(id.clone(), Loaded::Ready(page));
                        let _ = self.backend.commands.send(Command::Artist(id.clone()));
                        return;
                    }
                    self.artists.insert(id.clone(), Loaded::Loading);
                    let _ = self.backend.commands.send(Command::Artist(id.clone()));
                }
            }
            Detail::Album(id) => {
                if matches!(self.albums.get(id), None | Some(Loaded::Failed(_))) {
                    // A stored page with no tracks is fetched again rather than shown empty.
                    if let Some(page) =
                        self.library.album_page(id).filter(|p| !p.tracks.is_empty()).cloned()
                    {
                        self.albums.insert(id.clone(), Loaded::Ready(page));
                        return;
                    }
                    self.albums.insert(id.clone(), Loaded::Loading);
                    let _ = self.backend.commands.send(Command::Album(id.clone()));
                }
            }
        }
    }

    /// Forget fetched pages once there are too many, keeping the one on screen. Going back to a page
    /// that was forgotten fetches it again.
    fn trim_pages(&mut self) {
        let current = match &self.pages.top {
            Some(Detail::Artist(id)) => Some(id.clone()),
            _ => None,
        };
        self.artists.retain(|id, _| current.as_deref() == Some(id.as_str()));
        let current = match &self.pages.top {
            Some(Detail::Album(id)) => Some(id.clone()),
            _ => None,
        };
        self.albums.retain(|id, _| current.as_deref() == Some(id.as_str()));
    }

    /// The tracks of the page on screen, to play from. Empty while it loads.
    fn detail_tracks(&self) -> Vec<yt_client::SongHit> {
        match &self.pages.top {
            Some(Detail::Artist(id)) => match self.artists.get(id) {
                Some(Loaded::Ready(page)) => page.songs.clone(),
                _ => Vec::new(),
            },
            Some(Detail::Album(id)) => match self.albums.get(id) {
                Some(Loaded::Ready(page)) => page.tracks.clone(),
                _ => Vec::new(),
            },
            None => Vec::new(),
        }
    }

    fn request_cover(&mut self, url: String) {
        if self.gpu.covers.contains_key(&url) {
            return;
        }
        if self.cover_retry.get(&url).is_some_and(|retry_at| Instant::now() < *retry_at) {
            return;
        }
        if !self.cover_requested.insert(url.clone()) {
            return;
        }
        self.cover_retry.remove(&url);
        self.cover_inflight.insert(url.clone());
        let _ = self.backend.commands.send(Command::Cover(url));
    }

    fn apply(&mut self, action: Action) {
        // What an action changes is drawn on the next frame, which the host has to be asked for: the
        // frame that asked for it was already built.
        self.animating = true;
        // A song picked from the results counts as clicking into them.
        if matches!(action, Action::Play(_)) && self.pages.top.is_none() && !self.lyrics_open {
            self.results_clicked = true;
        }
        match action {
            Action::Window(request) => self.window_requests.push(request),
            Action::OpenMenu(menu) => self.menu = Some(menu),
            Action::Enqueue(hit) => self.playlist.enqueue(hit),
            Action::Search => {
                if !self.query.trim().is_empty() {
                    // A search is for the results page, so it takes over from the lyric page.
                    self.set_lyrics_open(false);
                    self.pages.clear();
                    self.results_return = None;
                    self.results_clicked = false;
                    self.status = "Searching…".to_owned();
                    self.searching = true;
                    self.last_search.clone_from(&self.query);
                    let _ = self
                        .backend
                        .commands
                        .send(Command::Search(self.query.clone()));
                }
            }
            Action::Play(index) => self.play_index(index),
            Action::PlayQueued(index) => {
                if let Some(hit) = self.playlist.skip_to(index) {
                    self.send_play(hit);
                }
            }
            Action::Toggle => {
                // Held while a song loads: the button is disabled, so the audible song is left alone.
                let loading = self.playlist.is_loading();
                let now = Instant::now();
                if let Some(track) = self.now.as_mut().filter(|_| !loading) {
                    track.want_playing = !track.want_playing;
                    track.paused = !track.want_playing;
                    if track.want_playing {
                        // Resume at once; the gain fades back up from wherever it is.
                        track.held = false;
                        track.player.play(now);
                    }
                    // A pause is applied by `tick`, once the fade has taken the sound to silence.
                }
            }
            Action::Seek(fraction) => {
                if let Some(track) = self.now.as_mut() {
                    let ms = (fraction as f64 * track.duration as f64 * 1000.0) as i64;
                    track.player.seek(ms);
                    track.position = fraction * track.duration;
                }
            }
            Action::Next => self.next(),
            Action::Back => self.back(),
            Action::Open(detail) => {
                self.ghosts.clear();
                self.open(detail)
            }
            Action::PageBack => self.page_back(),
            Action::PlayDetail(index) => {
                let tracks = self.detail_tracks();
                if let Some(hit) = self.playlist.play_from(&tracks, index) {
                    self.send_play(hit);
                }
            }
            Action::PlayTop => self.play_top(),
            Action::SaveSong(hit) => {
                let now = Instant::now();
                if self.library.song_saved(&hit.video_id) {
                    self.library.unsave_song(&hit.video_id, now);
                    self.ghosts.insert(hit.video_id);
                } else {
                    self.ghosts.remove(&hit.video_id);
                    let effects = self.library.save_song(&hit, now);
                    self.run_effects(effects);
                }
            }
            Action::SaveAlbum(page) => {
                let now = Instant::now();
                if self.library.ids().albums.contains(&page.browse_id) {
                    self.library.unsave_album(&page.browse_id, now);
                } else {
                    let effects = self.library.save_album(&page, now);
                    self.run_effects(effects);
                }
            }
            Action::SaveArtist(page) => {
                if self.library.vault.contains(&page.browse_id) {
                    self.library.unsave_artist(&page.browse_id);
                } else {
                    let effects = self.library.save_artist(&page);
                    self.run_effects(effects);
                }
            }
            Action::PlayLibrary(index) => {
                let ids: Vec<String> = self.library.song_list(&self.ghosts).to_vec();
                let hits: Vec<yt_client::SongHit> =
                    ids.iter().filter_map(|id| self.library.song_hit(id).cloned()).collect();
                if let Some(hit) = self.playlist.play_from(&hits, index) {
                    self.send_play(hit);
                }
            }
            Action::CollectionTab(tab) => {
                self.collection_tab = tab;
                self.ghosts.clear();
            }
            Action::ResultsTab(tab) => self.results_tab = tab,
            Action::MoveQueued { from, to } => self.playlist.move_queued(from, to),
            Action::RemoveQueued(index) => {
                self.playlist.remove_queued(index);
            }
            Action::Tab(tab) => {
                // Choosing a tab leaves the lyric page, any page, and the search results, for that tab.
                self.ghosts.clear();
                self.leave_results();
                self.lyrics_open = false;
                self.pages.clear();
                if tab != self.tab {
                    self.tab = tab;
                }
            }
            Action::ToggleSidebar => self.sidebar_open = !self.sidebar_open,
            Action::ToggleLyrics => self.set_lyrics_open(!self.lyrics_open),
            Action::Mute => {
                if self.volume > 0.0 {
                    self.volume_before_mute = Some(self.volume);
                    self.set_volume(0.0);
                } else {
                    let restored = self.volume_before_mute.take().unwrap_or(0.8);
                    self.set_volume(restored);
                }
            }
        }
    }
}

/// Stand-in lyrics for `--shot`. Invented text, so no real song's words go into the repository.
fn demo_lookup(kind: &str) -> Lookup {
    let lines = [
        "Lights come up on the empty street",
        "Every window keeps a secret",
        "",
        "We walk where the river bends",
        "Counting the stars we never named",
        "Slow down, the night is long enough",
        "",
        "Hold the thread and let it run",
        "Somewhere a radio is playing",
        "The sky is ink, the air is warm, and every streetlight hums a little louder than the one before it",
        "We kept our voices low and gold",
        "",
        "And morning finds us where we were",
    ];
    let timed = lines
        .iter()
        .enumerate()
        .map(|(i, text)| TimedLine {
            time_ms: 1_000 + i as u64 * 3_200,
            text: (*text).to_owned(),
        })
        .collect();
    let source = Source::Boidu;
    match kind {
        "plain" => Lookup::Found(
            yt_lyrics::Lyrics::plain(source, lines.iter().map(|l| (*l).to_owned()).collect())
                .expect("the stand-in has text"),
        ),
        "instrumental" => Lookup::Found(yt_lyrics::Lyrics::instrumental(source)),
        "none" => Lookup::NotFound,
        "unavailable" => Lookup::Unavailable("offline".to_owned()),
        _ => Lookup::Found(
            yt_lyrics::Lyrics::synced(source, timed).expect("the stand-in has text"),
        ),
    }
}

/// Seconds from a row's `m:ss` or `h:mm:ss` length; 0 when there is none or it does not parse.
fn hit_seconds(duration: Option<&str>) -> f32 {
    duration
        .and_then(|text| {
            text.split(':')
                .try_fold(0.0_f32, |total, part| part.parse::<f32>().ok().map(|p| total * 60.0 + p))
        })
        .unwrap_or(0.0)
}

/// `m:ss`, or `h:mm:ss`.
fn clock(seconds: f32) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "0:00".to_owned();
    }
    let total = seconds as u64;
    let (minutes, seconds) = (total / 60, total % 60);
    if minutes >= 60 {
        format!("{}:{:02}:{:02}", minutes / 60, minutes % 60, seconds)
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::clock;
    use super::{SEARCH_CLOSE_RATE, SEARCH_OPEN_RATE};
    use crate::anim::Anims;

    /// The search bar's reveal is a spring, so how quick it feels is entirely its two rates. These
    /// pin them to 60 Hz frames, so a softer rate cannot creep back in unnoticed: eight frames is
    /// 133 ms, and the bar's whole travel is 82 px.
    #[test]
    fn the_search_bar_opens_and_closes_within_a_few_frames() {
        let step = 1.0 / 60.0;

        let mut anims = Anims::default();
        anims.begin(step);
        anims.toggle("search", 0, false, SEARCH_OPEN_RATE); // seed it at rest, hidden
        let opening: Vec<f32> = (0..24)
            .map(|_| {
                anims.begin(step);
                anims.toggle("search", 0, true, SEARCH_OPEN_RATE)
            })
            .collect();
        assert!(opening[7] > 0.95, "open too slow: {}", opening[7]);
        assert_eq!(*opening.last().unwrap(), 1.0);

        let mut anims = Anims::default();
        anims.begin(step);
        anims.toggle("search", 0, true, SEARCH_CLOSE_RATE); // seed it at rest, shown
        let closing: Vec<f32> = (0..24)
            .map(|_| {
                anims.begin(step);
                anims.toggle("search", 0, false, SEARCH_CLOSE_RATE)
            })
            .collect();
        assert!(closing[7] < 0.05, "close too slow: {}", closing[7]);
        assert_eq!(*closing.last().unwrap(), 0.0);
    }

    #[test]
    fn back_returns_through_each_page_to_the_root() {
        use super::{Detail, Pages};
        let mut pages = Pages::default();
        assert!(pages.open(Detail::Artist("a".to_owned())));
        assert!(pages.open(Detail::Album("b".to_owned())));
        assert!(pages.back());
        assert_eq!(pages.top, Some(Detail::Artist("a".to_owned())));
        assert!(pages.back());
        assert_eq!(pages.top, None, "back from the first page is the results or the tab");
        assert!(!pages.back(), "nothing left to go back to");
    }

    #[test]
    fn opening_the_page_already_on_top_changes_nothing() {
        use super::{Detail, Pages};
        let mut pages = Pages::default();
        assert!(pages.open(Detail::Album("b".to_owned())));
        assert!(!pages.open(Detail::Album("b".to_owned())));
        assert!(pages.back());
        assert_eq!(pages.top, None, "one open, one back: the repeat did not add a step");
    }

    #[test]
    fn clearing_drops_every_page() {
        use super::{Detail, Pages};
        let mut pages = Pages::default();
        pages.open(Detail::Artist("a".to_owned()));
        pages.open(Detail::Album("b".to_owned()));
        pages.clear();
        assert_eq!(pages.top, None);
        assert!(!pages.back());
    }

    #[test]
    fn clock_formats_minutes_and_hours() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(71.4), "1:11");
        assert_eq!(clock(3599.0), "59:59");
        assert_eq!(clock(3661.0), "1:01:01");
        assert_eq!(clock(f32::NAN), "0:00");
        assert_eq!(clock(-3.0), "0:00");
    }
}

/// Seconds since `then`.
fn since(now: Instant, then: Instant) -> f32 {
    now.saturating_duration_since(then).as_secs_f32()
}

/// Build one frame. The caller owns the renderer; this only produces geometry.
///
/// `width` and `height` are in logical units; `scale` is device pixels per logical unit (the
/// display's scale times the window zoom), used to pick icon sizes and snap to the pixel grid.
pub fn build(
    ui: &mut Ui,
    state: &mut State,
    input: UiInput,
    pointer: Pointer,
    width: f32,
    height: f32,
    scale: f32,
) -> UiFrame {
    let now = Instant::now();
    let dt = input.dt;
    let mut actions: Vec<Action> = Vec::new();

    let submitted = std::mem::take(&mut state.submit);
    if std::mem::take(&mut state.toggle_request) {
        actions.push(Action::Toggle);
    }
    state.seek = state.position_fraction();
    let track = state.track_view();

    // The bar crossfades whenever the track changes.
    let shown = track.as_ref().map(|track| track.video_id.clone());
    if shown != state.shown_id {
        state.shown_id = shown;
        state.track_at = now;
    }

    // Search as you type: wait for a pause, then ask. Enter asks immediately.
    if state.query != state.query_seen {
        state.query_seen = state.query.clone();
        state.edited_at = now;
        state.query_dirty = true;
    }
    if submitted && !state.query.trim().is_empty() {
        state.query_dirty = false;
        actions.push(Action::Search);
    } else if state.query_dirty && since(now, state.edited_at) > 0.45 {
        state.query_dirty = false;
        if state.query.trim().chars().count() >= 2 {
            actions.push(Action::Search);
        }
    }

    let dark = Theme::dark_theme();
    let mut anims = std::mem::take(&mut state.anims);
    let gpu = std::mem::take(&mut state.gpu);
    anims.begin(dt);

    // Wheel input over the lyric page is the reader taking it over (see `LyricsFollow`). Read it
    // before `input` moves into the UI.
    let wheel = input.wheel + input.wheel_precise;

    // The track menu closes on a scroll, and on a press outside it. A right press outside it closes it
    // and then reaches the page, so right-clicking another track opens that track's menu. A left press
    // outside it only closes it: it is used up here, so it does not also press what is underneath.
    let menu_entries = match state.menu.as_ref() {
        Some(menu) => track_menu_entries(state, menu),
        None => Vec::new(),
    };
    let mut left_dismissed = false;
    if let Some(rect) = state.menu.as_ref().map(|menu| menu_rect(menu, menu_entries.len(), width, height)) {
        let inside = rect.contains(pointer.pos);
        if !inside && (pointer.pressed || pointer.secondary || wheel != 0.0) {
            left_dismissed = pointer.pressed;
            state.menu = None;
        }
    }

    ui.begin(input, (width, height));
    ui.set_theme(&dark);
    ui.set_focus_rings(false);
    ui.scrollbar = ScrollbarStyle {
        width: 4.0,
        inset: 14.0,
        min_thumb: 40.0,
        idle: theme::THUMB,
        active: theme::THUMB_HOT,
    };
    ui.rect(Rect::new(0.0, 0.0, width, height), theme::BG);

    // Layout.
    let body_h = height - theme::BAR_H;
    let want_open = state.sidebar_open && width >= theme::SIDEBAR_AUTO_SHUT;
    let open_t = anims.toggle("sidebar", 0, want_open, rate::PANEL);
    let sidebar_w = (theme::SIDEBAR_SHUT + (theme::SIDEBAR_OPEN - theme::SIDEBAR_SHUT) * open_t).round();
    let sidebar = Rect::new(0.0, 0.0, sidebar_w, body_h);
    let main = Rect::new(sidebar_w, 0.0, width - sidebar_w, body_h);
    let bar = Rect::new(0.0, body_h, width, theme::BAR_H);
    // The window's own caption buttons, in the top-right corner over the page, centred on the
    // search field's line so the two sit level.
    let caption = Rect::new(
        width - theme::CAPTION_INSET - theme::CAPTION_BTN_W * 2.0,
        theme::SEARCH_TOP + (theme::SEARCH_H - theme::CAPTION_H) * 0.5,
        theme::CAPTION_BTN_W * 2.0,
        theme::CAPTION_H,
    );
    if state.lyrics_open && wheel != 0.0 && main.contains(pointer.pos) {
        state.lyrics_follow.reader = true;
        state.lyrics_follow.scrolled_at = Some(now);
    }

    // The search field is hidden until the pointer nears the top of the page, or it has text or
    // focus. Once shown, the zone that keeps it shown is taller, so it does not flicker at the
    // edge, and taller still on the results page, so reaching for the first rows does not dismiss
    // it.
    let typed = !state.query.is_empty();
    let focused = ui.focused("search");
    let on_detail = !state.lyrics_open && state.pages.top.is_some();
    let results_page = !state.lyrics_open && !on_detail && !state.results_query.is_empty();
    let open_zone = if results_page || on_detail { theme::TOP_H + 150.0 } else { 124.0 };
    let zone_h = if state.search_wanted { open_zone } else { theme::TOP_H };
    let zone = Rect::new(main.x, 0.0, main.w, zone_h);
    // On the results page the bar stays up while the list is at its top; once scrolled, it is hidden
    // again and behaves like everywhere else.
    // The scroll area of whatever list or page is on screen.
    let list_id = match &state.pages.top {
        Some(detail) => detail_scroll_id(detail),
        None => "search_results".to_owned(),
    };
    // Back on the results: bring the list back to where the reader left it. Otherwise remember where
    // it is, so leaving the results can return to it.
    if results_page {
        if let Some(target) = state.pending_results_scroll.take() {
            ui.scroll_by(&list_id, target - ui.scroll_state(&list_id).0, dream::ui::widgets::ScrollBehavior::Instant);
        }
        state.results_scroll = ui.scroll_state(&list_id).0;
    }
    // `V1_SHOT_SCROLL=<px>` scrolls the list on screen to that offset, to check the glass over rows.
    if let Some(target) = std::env::var("V1_SHOT_SCROLL").ok().and_then(|v| v.parse::<f32>().ok()) {
        let current = ui.scroll_state(&list_id).0;
        if (target - current).abs() > 0.5 {
            ui.scroll_by(&list_id, target - current, dream::ui::widgets::ScrollBehavior::Instant);
        }
    }
    // A detail page is pinned like the results: the bar, and the back button with it, stay up while
    // the page is at its top.
    let at_top = ui.scroll_state(&list_id).0 < 0.5;
    let pinned = (results_page || on_detail) && at_top;
    // The caption buttons are not part of the reveal zone: reaching for them does not pop the field.
    let wanted = pinned
        || typed
        || focused
        || state.focus_search
        || (zone.contains(pointer.pos) && !caption.contains(pointer.pos));
    state.search_wanted = wanted;
    // It drops in from above the window and settles without overshoot; leaving, it springs back up,
    // faster still, so the page is clear the moment the pointer leaves the zone.
    let stiffness = if wanted { SEARCH_OPEN_RATE } else { SEARCH_CLOSE_RATE };
    let reveal = anims.toggle("search", 0, wanted, stiffness);
    if wanted || reveal <= 0.001 {
        state.search_ghost = None;
    }
    let search_h = theme::SEARCH_H;
    let hidden_y = -(search_h + 20.0);
    let search_y = hidden_y + (theme::SEARCH_TOP - hidden_y) * reveal;
    // Centred on the page. On a narrow window that would run under the caption buttons, so it is
    // centred in the room left of them instead.
    let caption_edge = caption.x - theme::CAPTION_INSET;
    let mut search_w = (main.w - theme::PAGE_PAD * 2.0).clamp(220.0, 620.0);
    let mut search_x = main.x + (main.w - search_w) * 0.5;
    if search_x + search_w > caption_edge {
        let room = (caption_edge - main.x).max(0.0);
        search_w = (room - theme::PAGE_PAD * 2.0).clamp(220.0, 620.0);
        search_x = main.x + (room - search_w) * 0.5;
    }
    // The back button's room is kept on the bar's left always, so the bar never moves when the button
    // appears. The button has its own spring; as it opens, the field gives up that much of its left
    // side and the button slides into the gap.
    let left_room = main.x + BACK_STEP;
    if search_x < left_room {
        search_x = left_room;
        search_w = (caption_edge - search_x).min(search_w).max(0.0);
    }
    let back_wanted = state.pages.top.is_some() || state.lyrics_open;
    let back_stiffness = if back_wanted { SEARCH_OPEN_RATE } else { SEARCH_CLOSE_RATE };
    let back_t = anims.toggle("back", 0, back_wanted, back_stiffness);
    let shave = BACK_STEP * back_t;
    let search = Rect::new(
        (search_x + shave).round(),
        search_y.round(),
        (search_w - shave).round(),
        search_h,
    );
    let back = Rect::new(search.x - BACK_STEP, search.y, search_h, search_h);
    let back_visible = back_t > 0.001 && back.y + back.h > 0.0;
    let covered = (search.y + search.h > 0.0 && search.contains(pointer.pos))
        || (back_visible && back.contains(pointer.pos));
    let over_caption = caption.contains(pointer.pos);

    // The spinner turns while a song loads, the same for the bar and the rows.
    if state.playlist.is_loading() {
        state.spin = (state.spin + anims.dt() * SPINNER_TURN) % std::f32::consts::TAU;
    }
    // Everything drawn over the page. The page is drawn first, so these are known up front and a
    // page widget under one of them does not take the press.
    let mut covers = vec![bar, sidebar, caption];
    if search.y + search.h > 0.0 {
        covers.push(search);
    }
    if back_visible {
        covers.push(back);
    }
    // While the track menu is open, the page under it takes no presses, and a press outside it (which
    // closes it, above) takes none either.
    if let Some(menu) = state.menu.as_ref() {
        covers.push(menu_rect(menu, menu_entries.len(), width, height));
    }
    if left_dismissed {
        covers.push(Rect::new(0.0, 0.0, width, height));
    }

    let mut cx = Cx {
        ui: &mut *ui,
        anims: &mut anims,
        gpu: &gpu,
        pointer,
        font: FontId(0),
        scale,
        blocked: covered || over_caption,
        hand: false,
        drag: state.drag,
        queue_drag: state.queue_drag,
        wanted_icons: Vec::new(),
        wanted_covers: Vec::new(),
        touched: std::cell::Cell::new(false),
        covering: true,
        covers,
        spin: state.spin,
        saved: state.library.ids(),
    };

    // Page first, then the surfaces that sit over it.
    page(&mut cx, state, track.as_ref(), main, now, &mut actions);
    cx.covering = false;
    // Over the caption buttons, everything beneath them ignores the pointer, the page included.
    cx.blocked = over_caption;
    draw_sidebar(&mut cx, state, sidebar, open_t, &mut actions);
    draw_join_crescent(&mut cx, sidebar);
    // The chrome floats over the content, which now scrolls up under it. Its layer sorts it above
    // the page's rows, so a row's text never paints over the search bar or the caption icons.
    let page_layer = cx.draw().set_layer(CHROME_LAYER);
    draw_search(&mut cx, state, search);
    draw_back(&mut cx, back, back_t, &mut actions);
    draw_bar(&mut cx, state, track.as_ref(), bar, &mut actions);
    cx.blocked = false;
    draw_caption(&mut cx, caption, &mut actions);
    cx.draw().set_layer(page_layer);

    // The track menu sits over everything, the caption buttons included.
    draw_track_menu(&mut cx, state, &menu_entries, width, height, &mut actions);

    // A press on empty space in the top strip grabs the window. Anything that took the press (a
    // row, the sidebar toggle), the search field, and the caption buttons keep it for themselves.
    let press = cx.pointer.pressed;
    if press
        && cx.pointer.pos.1 < theme::TOP_H
        && !cx.touched.get()
        && !left_dismissed
        && !search.contains(cx.pointer.pos)
        && !back.contains(cx.pointer.pos)
        && !caption.contains(cx.pointer.pos)
    {
        actions.push(Action::Window(WindowRequest::Drag));
    }

    let hand = cx.hand;
    state.drag = cx.drag;
    state.queue_drag = cx.queue_drag;
    let wanted_icons = std::mem::take(&mut cx.wanted_icons);
    let wanted_covers = std::mem::take(&mut cx.wanted_covers);
    // A scroll glide in flight (the lyric page centring on its line) keeps the host presenting too.
    let scrolling = ui.animating();
    let frame = ui.end();

    state.hand = hand;
    let entering = since(now, state.results_at) < 1.0
        || since(now, state.track_at) < 0.5
        || state
            .lyrics
            .as_ref()
            .is_some_and(|view| since(now, view.arrived) < 1.5);
    let visualizing = state.lyrics_open && state.spectrum_live;
    // A song that is loading keeps the spinner turning.
    let loading = state.playlist.is_loading();
    state.animating = anims.moving()
        || entering
        || state.query_dirty
        || scrolling
        || visualizing
        || loading
        || state.fading;
    state.anims = anims;
    state.gpu = gpu;
    state.gpu.wanted_icons.extend(wanted_icons);
    for url in wanted_covers {
        state.request_cover(url);
    }

    for action in actions {
        state.apply(action);
    }
    frame
}

// ---------------------------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------------------------

/// The crescent that rounds the sidebar's bottom-right corner into the player bar. The sidebar and
/// the bar are the same colour, so the join reads as one panel: the page corner is cut by a
/// concave quarter-circle, and the sliver outside that arc is painted in the panel colour.
/// Drawn as thin horizontal strips, each as wide as the sliver is at its height.
fn draw_join_crescent(cx: &mut Cx, sidebar: Rect) {
    let r = theme::JOIN_R;
    let right = sidebar.x + sidebar.w;
    let bottom = sidebar.y + sidebar.h;
    // Centre of the fillet circle: `r` in from the sidebar edge and `r` up from the bar.
    let (cx_, cy) = (right + r, bottom - r);
    let step = 0.5;
    let rows = (r / step).ceil() as i32;
    for i in 0..rows {
        let y0 = cy + i as f32 * step;
        let y1 = (y0 + step).min(bottom);
        let dy = (y0 + y1) * 0.5 - cy;
        // Where the circle's edge sits at this height; the sliver runs from the sidebar to there.
        let edge = cx_ - (r * r - dy * dy).max(0.0).sqrt();
        let w = edge - right;
        if w > 0.0 && y1 > y0 {
            cx.draw().rect(Rect::new(right, y0, w, y1 - y0), theme::PANEL);
        }
    }
}

fn draw_sidebar(cx: &mut Cx, state: &State, rect: Rect, open_t: f32, actions: &mut Vec<Action>) {
    cx.draw().rect(rect, theme::PANEL);
    // Labels appear in the second half of the opening, and leave in the first half of closing, so
    // they are never squeezed against a narrow edge.
    let label_a = ((open_t - 0.55) / 0.45).clamp(0.0, 1.0);

    // Both the logo and the toggle live in the same 43 × 40 box, centred on x = 34 like the tab
    // icons. Shut, the box holds the logo, and hovering it swaps in the toggle. Open, the logo
    // stays on the left and the toggle moves to the top right. At the shut width (68) the two
    // boxes are the same, so the toggle slides across as the sidebar animates.
    let logo = Rect::new(12.5, 16.0, 43.0, 40.0);
    let toggle = Rect::new(rect.w - 56.0, 16.0, 43.0, 40.0);
    let shut = open_t < 0.5;
    let logo_hover = shut && cx.over_clickable(logo);
    let swap = cx.anims.toggle("logo_swap", 0, logo_hover, rate::QUICKER);
    // Shut, the toggle shows only while the logo is hovered; open, it is always there.
    let toggle_a = if shut { swap } else { 1.0 };
    let logo_a = if shut { 1.0 - swap } else { 1.0 };

    if logo_a > 0.0 {
        cx.wordmark((34.0, 36.0), (32.0, 30.0), fade(Color::WHITE, logo_a));
    }

    if toggle_a > 0.0 {
        let hover = toggle_a > 0.5 && cx.over_clickable(toggle);
        let hv = cx.anims.toggle("toggle_hv", 0, hover, rate::SNAP);
        let center = (toggle.x + 21.5, toggle.y + 20.0);
        cx.icon_state(
            &[Icon::PanelLeft, Icon::PanelRight],
            if open_t > 0.5 { 0 } else { 1 },
            center,
            22.0,
            fade(mix(theme::MUTED, theme::FG, hv), toggle_a),
        );
        if toggle_a > 0.5 && cx.clicked(toggle) {
            actions.push(Action::ToggleSidebar);
        }
    }

    // Tabs. The selected tab is a plain pill with no pop; hovering washes the others.
    let item_w = rect.w - 24.0;
    let top = 76.0;
    let pitch = 40.0;
    let item_h = 36.0;
    for (index, tab) in Tab::ALL.iter().copied().enumerate() {
        let item = Rect::new(12.0, top + index as f32 * pitch, item_w, item_h);
        let hover = cx.over_clickable(item);
        let hv = cx.anims.toggle("tab_hv", index as u32, hover, rate::SNAP);
        // While the lyric page, a detail page (artist, album, ...) or the search results are up,
        // they are the page: no tab is selected.
        let selected = tab == state.tab
            && !state.lyrics_open
            && state.pages.top.is_none()
            && state.results_query.is_empty();

        if selected {
            cx.draw().round_rect(item, 11.0, theme::PANEL_SELECT);
        } else {
            cx.draw().round_rect(item, 11.0, mix(theme::PANEL, theme::PANEL_HOVER, hv));
        }

        let color = mix(theme::MUTED, theme::FG, if selected { 1.0 } else { hv });
        cx.icon(tab.icon(), (34.0, item.y + item_h * 0.5), 19.0, color);
        cx.clip(item);
        cx.ui.label(
            Rect::new(52.0, item.y, 150.0, item.h),
            tab.label(),
            dream::ui::widgets::LabelStyle::new(FontId(2))
                .size(13.0)
                .color(fade(color, label_a))
                .align(Align::Left)
                .vcenter(true),
        );
        cx.unclip();
        if cx.clicked(item) {
            actions.push(Action::Tab(tab));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Caption
// ---------------------------------------------------------------------------------------------

/// The window's own minimise and close buttons, top right. There is no OS title bar, so the app
/// draws them in its own greys. No shared container or group hover: each button only lifts a step
/// under the pointer.
fn draw_caption(cx: &mut Cx, rect: Rect, actions: &mut Vec<Action>) {
    let buttons = [
        (Icon::Minus, WindowRequest::Minimize),
        (Icon::X, WindowRequest::Close),
    ];
    for (index, (icon, request)) in buttons.into_iter().enumerate() {
        let button = Rect::new(
            rect.x + index as f32 * theme::CAPTION_BTN_W,
            rect.y,
            theme::CAPTION_BTN_W,
            rect.h,
        );
        let hover = cx.over_clickable(button);
        let hv = cx.anims.toggle("caption_btn", index as u32, hover, rate::SNAP);
        // Close turns its glyph red on hover; both buttons keep the grey container behind them.
        let is_close = matches!(request, WindowRequest::Close);
        if hv > 0.0 {
            let lit = Rect::new(button.x + 3.0, button.y + 3.0, button.w - 6.0, button.h - 6.0);
            cx.draw().round_rect(lit, 8.0, fade(theme::PANEL_PRESS, hv));
        }
        let hot = if is_close { theme::CLOSE_RED } else { theme::FG };
        let color = mix(theme::MUTED, hot, hv);
        let center = (button.x + button.w * 0.5, button.y + button.h * 0.5);
        cx.icon(icon, center, 16.0, color);
        if cx.clicked(button) {
            actions.push(Action::Window(request));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------------------------

fn draw_search(cx: &mut Cx, state: &mut State, rect: Rect) {
    // Fully above the window: nothing to draw, and nothing to focus.
    if rect.y + rect.h <= 0.0 && !state.focus_search {
        return;
    }
    let typed = !state.query.is_empty();
    let focused = cx.ui.focused("search");
    let hover = cx.pointer_in(rect);
    let focus_t = cx.anims.toggle("search_focus", 0, focused, rate::QUICK);
    let hover_t = cx.anims.toggle("search_hover", 0, hover, rate::QUICK);
    let lift = focus_t.max(hover_t * 0.6);

    // Frosted glass: the page behind the bar, blurred, darkened by a grey tint that brightens a
    // little on focus. Text and icons on it stay opaque.
    // Frosted glass: the blurred page goes through untinted, then a translucent milky grey sits over
    // it. The wash is kept well under opaque so the blur stays readable; the grey is dark enough to
    // sit on black. (The wash is a palette grey: white at low alpha blends in linear light and reads
    // far brighter on screen than its alpha suggests.)
    cx.draw().backdrop_rounded(rect, Color::rgb(0xff, 0xff, 0xff), rect.h * 0.5);
    // The tint is the sidebar's selected-pill grey, brightening one step toward its press grey on focus.
    cx.draw().round_rect(
        rect,
        rect.h * 0.5,
        fade(mix(theme::PANEL_SELECT, theme::PANEL_PRESS, lift), 0.5 + 0.1 * lift),
    );
    cx.icon(
        Icon::Search,
        (rect.x + 28.0, rect.y + rect.h * 0.5),
        19.0,
        mix(theme::MUTED, theme::FG, focus_t),
    );

    let clear = Rect::new(rect.x + rect.w - 42.0, rect.y + 9.0, 28.0, 28.0);
    let field = Rect::new(rect.x + 54.0, rect.y, rect.w - 54.0 - 48.0, rect.h);

    if state.focus_search {
        state.focus_search = false;
        cx.ui.set_focus("search");
    }
    let mut style = TextFieldStyle::new(cx.font);
    style.size = 15.5;
    style.text = theme::FG;
    style.normal = Color::TRANSPARENT;
    style.hover = Color::TRANSPARENT;
    style.accent = theme::FG;
    style.selection_alpha = 56;
    style.corner_radius = 0.0;
    style.border_width = 0.0;
    style.caret = theme::FG;
    style.caret_width = 1.5;
    style.caret_inset = 13.0;
    // Selection highlight spans the same height as the caret, so the two line up.
    style.selection_inset = style.caret_inset;
    style.text_inset = 0.0;
    cx.ui.text_field("search", field, &mut state.query, style);

    if let (false, Some(ghost)) = (typed, state.search_ghost.as_deref()) {
        cx.text(field, ghost, 15.5, theme::FG, Align::Left);
    } else if !typed {
        cx.text(
            field,
            "Search songs and artists",
            15.5,
            fade(theme::MUTED, 1.0 - 0.35 * focus_t),
            Align::Left,
        );
    }

    // The clear button appears instantly once there is something to clear. No motion: it is
    // drawn at a fixed size, and hover only switches its colour.
    if typed {
        let hovered = cx.pointer_in(clear);
        if hovered {
            cx.hand = true;
        }
        let center = (clear.x + clear.w * 0.5, clear.y + clear.h * 0.5);
        if hovered {
            cx.draw().round_rect(
                Rect::new(center.0 - 14.0, center.1 - 14.0, 28.0, 28.0),
                14.0,
                theme::RAISED_TOP,
            );
        }
        cx.icon(Icon::X, center, 16.0, if hovered { theme::FG } else { theme::MUTED });
        if hovered && cx.pointer.pressed {
            state.query.clear();
            cx.ui.set_focus("search");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------------------------

fn page(
    cx: &mut Cx,
    state: &mut State,
    track: Option<&TrackView>,
    main: Rect,
    now: Instant,
    actions: &mut Vec<Action>,
) {
    cx.clip(main);

    if state.lyrics_open {
        lyrics_page(cx, state, track, main, now, actions);
        cx.unclip();
        return;
    }

    // The list reaches the window's right edge, so its scrollbar floats in the margin beyond the
    // rows instead of on top of them.
    // The list fills the page. Its top spacing is part of the scrolling content (see `track_list`),
    // so nothing is reserved above it and nothing sits fixed over what scrolls beneath.
    let list_top = 0.0;
    let left = theme::PAGE_PAD - 12.0;
    let view = Rect::new(
        main.x + left,
        list_top,
        (main.w - left).max(0.0),
        (main.h - list_top).max(0.0),
    );

    if let Some(detail) = state.pages.top.clone() {
        detail_page(cx, state, track, &detail, main, view, actions);
        cx.unclip();
        return;
    }

    if !state.results_query.is_empty() {
        // Results are a page-like search state, not a sidebar destination.
        let list_age = since(now, state.results_at);
        search_results(cx, state, track, view, list_age, actions);
        cx.unclip();
        return;
    }

    match state.tab {
        Tab::Library => collection_page(cx, state, track, main, view, actions),
        Tab::Queue => {
            if state.playlist.queue.is_empty() {
                empty_state(
                    cx,
                    main,
                    Icon::Menu,
                    "Nothing in the queue",
                    "Songs you add to the queue, and the rest of an album or playlist you play, show up here.",
                );
            } else {
                // Queue rows are always fully shown: no stagger on tab switch.
                let rows: Vec<Row> =
                    state.playlist.queue.iter().map(|hit| Row::from(hit, &state.lengths)).collect();
                track_list(cx, track, &rows, view, "queue_list", 4000, f32::INFINITY, true, actions);
            }
        }
    }
    cx.unclip();
}

// ---------------------------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------------------------

/// The tabs along the Collection page's top left: their height, and the room under them.
const COLLECTION_TAB_H: f32 = 34.0;
const COLLECTION_TAB_GAP: f32 = 14.0;
const COLLECTION_TAB_ICON: f32 = 16.0;

/// The Collection page: Songs, Artists and Albums, switched from the tabs at its top left. A long
/// list builds only the rows or tiles that are on screen, so a collection of thousands costs what a
/// screenful does.
fn collection_page(
    cx: &mut Cx,
    state: &mut State,
    track: Option<&TrackView>,
    main: Rect,
    view: Rect,
    actions: &mut Vec<Action>,
) {
    // The lists sit side by side and the page slides between them: `at` is which one is centred,
    // moving on a spring toward the chosen tab, so switching again mid-slide turns around smoothly.
    let target = state.collection_tab;
    let at = cx.anims.spring("collection_slide", 0, target.index() as f32, 380.0);
    // The scrollbars are hidden for the whole slide, then fade back in quickly as it lands. The
    // spring always tracks "not sliding", so it is already rising when the slide ends.
    let sliding = (at - target.index() as f32).abs() > 0.01;
    let bar_a = cx.anims.toggle("collection_bar", 0, !sliding, rate::QUICKER);
    let bar_a = if sliding { 0.0 } else { bar_a };
    for tab in CollectionTab::ALL {
        let dx = (tab.index() as f32 - at) * view.w;
        if dx.abs() >= view.w {
            continue;
        }
        let shift = |r: Rect| Rect::new(r.x + dx, r.y, r.w, r.h);
        collection_list(cx, state, track, tab, shift(main), shift(view), bar_a, actions);
    }
    // The pills are drawn once, apart from the lists that slide, so they never move or clip with
    // them. They ride the chosen list's scroll, as they did when they were part of it.
    let id = match target {
        CollectionTab::Songs => "collection_songs",
        CollectionTab::Artists => "collection_artists",
        CollectionTab::Albums => "collection_albums",
    };
    let offset = cx.ui.scroll_state(id).0;
    let labels = CollectionTab::ALL.map(|tab| (tab.label(), tab.icon()));
    if let Some(index) = tab_pills(cx, &labels, target.index(), view.x + 8.0, PAGE_TOP - offset, 0) {
        actions.push(Action::CollectionTab(CollectionTab::ALL[index]));
    }
}

/// One list of the Collection page, in the rectangles it is given (shifted to the side while the
/// page slides).
#[allow(clippy::too_many_arguments)]
fn collection_list(
    cx: &mut Cx,
    state: &mut State,
    track: Option<&TrackView>,
    tab: CollectionTab,
    main: Rect,
    view: Rect,
    bar_alpha: f32,
    actions: &mut Vec<Action>,
) {
    // The lists scroll below the caption buttons, as the detail pages do, so their scrollbar never
    // runs up behind them. Their content is measured from that scroll area's top.
    let view = below_caption(view);
    let content_top = PAGE_TOP + COLLECTION_TAB_H + COLLECTION_TAB_GAP - SCROLL_TOP;

    // How many entries the list has, and what to say when it has none.
    let count = match tab {
        CollectionTab::Songs => state.library.song_list(&state.ghosts).len(),
        CollectionTab::Artists => state.library.vault.len(),
        CollectionTab::Albums => state.library.album_count(),
    };
    if count == 0 {
        let (icon, title, hint) = match tab {
            CollectionTab::Songs => (Icon::Music, "No saved songs", "Press + on a song to keep it here. It is downloaded to this device."),
            CollectionTab::Artists => (Icon::Mic, "No saved artists", "Press + on an artist's page to keep them here."),
            CollectionTab::Albums => (Icon::Disc, "No saved albums", "Press + on an album to save every song on it."),
        };
        empty_state(cx, main, icon, title, hint);
        return;
    }

    let scroll_id = match tab {
        CollectionTab::Songs => "collection_songs",
        CollectionTab::Artists => "collection_artists",
        CollectionTab::Albums => "collection_albums",
    };
    let offset = cx.ui.scroll_state(scroll_id).0;
    // The thumb is painted when each scroll view closes, from the style in force then. Fade it by
    // `bar_alpha` for these lists only; the page's own style is put back once they are drawn.
    let bar = cx.ui.scrollbar;
    cx.ui.scrollbar.idle = fade(bar.idle, bar_alpha);
    cx.ui.scrollbar.active = fade(bar.active, bar_alpha);

    match tab {
        CollectionTab::Songs => {
            let first = (((offset - content_top) / theme::ROW_H).floor().max(0.0) as usize).saturating_sub(2);
            let last = (first + (view.h / theme::ROW_H).ceil() as usize + 6).min(count);
            let ids: Vec<String> = state.library.song_list(&state.ghosts)[first..last].to_vec();
            let rows: Vec<Row> = ids
                .iter()
                .filter_map(|id| state.library.song_hit(id))
                .map(|hit| Row::from(hit, &state.lengths))
                .collect();
            let content_h = content_top + count as f32 * theme::ROW_H + 24.0;
            let body = cx.ui.scroll_begin(scroll_id, view, content_h);
            track_rows(
                cx,
                track,
                &rows,
                view,
                body,
                body.y + content_top,
                6000,
                f32::INFINITY,
                false,
                Action::PlayLibrary,
                first,
                actions,
            );
            cx.ui.scroll_end();
        }
        CollectionTab::Artists | CollectionTab::Albums => {
            let columns = tile_columns(view.w - 44.0);
            let step = TILE + TILE_TEXT_H + TILE_GAP;
            let first_row = (((offset - content_top) / step).floor().max(0.0) as usize).saturating_sub(1);
            let rows_shown = (view.h / step).ceil() as usize + 3;
            let from = (first_row * columns).min(count);
            let to = ((first_row + rows_shown) * columns).min(count);
            // (title, meta, picture, where it opens), for just the tiles in view.
            let tiles: Vec<(String, String, Option<String>, Detail)> = match tab {
                CollectionTab::Artists => state.library.vault.list()[from..to]
                    .iter()
                    .map(|a| (a.name.clone(), "Artist".to_owned(), a.thumbnail.clone(), Detail::Artist(a.id.clone())))
                    .collect(),
                _ => state.library.albums_between(from, to)
                    .into_iter()
                    .map(|a| {
                        (
                            a.page.title.clone(),
                            a.page.artist.clone(),
                            a.page.thumbnail.clone(),
                            Detail::Album(a.page.browse_id.clone()),
                        )
                    })
                    .collect(),
            };
            let content_h = content_top + count.div_ceil(columns) as f32 * step + 24.0;
            let body = cx.ui.scroll_begin(scroll_id, view, content_h);
            let left = body.x + 8.0;
            let artists = tab == CollectionTab::Artists;
            for (n, (title, meta, picture, target)) in tiles.into_iter().enumerate() {
                let k = from + n;
                let tx = left + (k % columns) as f32 * (TILE + TILE_GAP);
                let ty = body.y + content_top + (k / columns) as f32 * step;
                media_tile(
                    cx,
                    view,
                    actions,
                    (tx, ty),
                    (title.as_str(), meta.as_str(), picture.as_deref()),
                    artists,
                    target,
                    if artists { 20_000 } else { 30_000 } + k as u32,
                );
            }
            cx.ui.scroll_end();
        }
    }
    cx.ui.scrollbar = bar;
}

/// A row of pill tabs (icon and label each), left-aligned at `(x, y)`. The `current` one is lit and
/// its pill glides between tabs; the others wash under the pointer. Returns the index of a tab
/// clicked this frame, never the chosen one. `key` offsets the animation keys, so two rows of pills
/// on screen keep their own springs.
fn tab_pills(
    cx: &mut Cx,
    tabs: &[(&'static str, Icon)],
    current: usize,
    x: f32,
    y: f32,
    key: u32,
) -> Option<usize> {
    let widths: Vec<f32> = tabs
        .iter()
        .map(|(label, _)| cx.text_width(label, 13.5) + 34.0 + COLLECTION_TAB_ICON + 8.0)
        .collect();
    let mut lefts = Vec::with_capacity(widths.len());
    let mut left = x;
    for width in &widths {
        lefts.push(left);
        left += width + 8.0;
    }
    // The chosen pill is a single shape that glides from tab to tab: position and width are springs.
    let pill_x = cx.anims.spring("tab_pill_x", key, lefts[current] - x, 700.0) + x;
    let pill_w = cx.anims.spring("tab_pill_w", key, widths[current], 700.0);
    cx.draw()
        .round_rect(Rect::new(pill_x, y, pill_w, COLLECTION_TAB_H), COLLECTION_TAB_H * 0.5, theme::PRESS);

    let mut picked = None;
    for (index, (label, icon)) in tabs.iter().enumerate() {
        let rect = Rect::new(lefts[index], y, widths[index], COLLECTION_TAB_H);
        let selected = index == current;
        let hover = cx.over_clickable(rect);
        let slot = key + index as u32;
        let hv = cx.anims.toggle("tab_hv", slot, hover, rate::SNAP);
        let on = cx.anims.toggle("tab_on", slot, selected, rate::SNAP);
        // Hover only warms the text and icon; the container belongs to the chosen tab alone.
        let ink = mix(theme::MUTED, theme::FG, hv.max(on));
        cx.icon(*icon, (rect.x + 17.0 + COLLECTION_TAB_ICON * 0.5, rect.y + rect.h * 0.5), COLLECTION_TAB_ICON, ink);
        cx.text(
            Rect::new(rect.x + 17.0 + COLLECTION_TAB_ICON + 8.0, rect.y, rect.w, rect.h),
            label,
            13.5,
            ink,
            Align::Left,
        );
        if !selected && cx.clicked(rect) {
            picked = Some(index);
        }
    }
    picked
}

/// Type size for every synced line. It never changes: the line being sung is lit by colour alone, so
/// its wrapping and size stay put as it becomes active.
const LYRIC_SIZE: f32 = 22.0;
/// The emphasis (a line lighting up as it is sung) is a spring like the rest of the UI, a little
/// quicker than the `PANEL` preset.
const LYRIC_RATE: f32 = 640.0;
const LYRIC_LEADING: f32 = 1.2;
const LYRIC_GAP: f32 = 18.0;
const PAUSE_H: f32 = 30.0;
const PLAIN_SIZE: f32 = 18.0;
const PLAIN_LEADING: f32 = 1.4;
const PLAIN_GAP: f32 = 8.0;
const STANZA_H: f32 = 24.0;
/// The lyric column is centred, never wider than this, and keeps this much room at each side.
const LYRIC_COLUMN_MAX: f32 = 640.0;
const LYRIC_COLUMN_PAD: f32 = 40.0;
/// Once the reader has scrolled the lyrics, the page keeps its place until they have been still this
/// long, then goes back to the line being sung.
const LYRIC_IDLE_SECS: f32 = 3.0;
/// Entrance: each line starts this much after the one above it and takes `ENTER_TIME`. It rises
/// `ENTER_RISE` px as it fades in.
const ENTER_STAGGER: f32 = 0.04;
const ENTER_TIME: f32 = 0.55;
const ENTER_RISE: f32 = 12.0;
/// Spring keys for one lyric set are spaced this far apart, so a new set starts with fresh springs.
const KEY_SPAN: u32 = 4096;
/// The visualizer behind the lyric page: the spectrum as bars, mirrored about the page's centre so
/// it sits under the centred lyrics, lows in the middle and highs out at the edges.
/// The tallest bar, as a fraction of the page height.
const VISUALIZER_HEIGHT: f32 = 0.34;
/// The bars stand this far above the bottom of the page.
const VISUALIZER_BASE: f32 = 28.0;
/// Each bar's share of its slot; the rest is the gap.
const VISUALIZER_FILL: f32 = 0.5;
/// Bar tones, as how far they are mixed from the page black toward the text colour: a quiet band and
/// a full one. Both opaque and all well under the faint lyric tone, so the words
/// always read first.
const VISUALIZER_QUIET: f32 = 0.06;
const VISUALIZER_LOUD: f32 = 0.17;
/// Below this band level the output counts as silent. The meter's floor is -55 dB, so this is about
/// -54 dB; it is the line between audible and not.
const VISUALIZER_AUDIBLE: f32 = 0.02;

/// A colour as it looks `amount` of the way out of the page. It is mixed from black, not blended
/// with alpha, so it stays opaque and sharp.
fn revealed(base: Color, amount: f32) -> Color {
    mix(theme::BG, base, amount)
}

/// How far a line has arrived `age` seconds after its set appeared, 0 to 1.
fn arrival(age: f32, index: usize) -> f32 {
    ease_out((age - index as f32 * ENTER_STAGGER) / ENTER_TIME)
}

/// The centred column the lyrics are set in: its left edge and width.
fn lyric_column(view: Rect) -> (f32, f32) {
    let width = (view.w - 2.0 * LYRIC_COLUMN_PAD).clamp(120.0, LYRIC_COLUMN_MAX);
    (view.x + (view.w - width) * 0.5, width)
}

/// The sound behind the lyric page: the spectrum of what is playing as rounded bars along the bottom,
/// mirrored about the centre. Loud bands are
/// a step brighter than quiet ones. It fades in with the sound and out with it (silence, zero volume,
/// mute, or paused), and the bars settle rather than vanish.
fn lyric_visualizer(cx: &mut Cx, state: &mut State, track: Option<&TrackView>, main: Rect) {
    let dt = cx.anims.dt();
    let levels = track.map_or([0.0; 3], |t| t.levels);
    let loudest = levels.iter().copied().fold(0.0_f32, f32::max);
    let audible = track.is_some_and(|t| !t.paused) && loudest > VISUALIZER_AUDIBLE;

    // Feed the analyser: the real output when there is one, a still shape for `--shot`.
    let rate = state
        .now
        .as_ref()
        .and_then(|now| now.player.scope(&mut state.spectrum.samples));
    match rate {
        Some(rate) if audible => state.spectrum.analyse(rate, dt),
        None if audible && state.demo_track.is_some() => {
            state.spectrum.set(demo_spectrum());
        }
        _ => state.spectrum.decay(dt),
    }
    state.spectrum_live = state.spectrum.active();

    let show = cx.anims.toggle("lyric_visualizer", 0, audible, rate::PANEL);
    if show <= 0.002 {
        return;
    }

    let bands = spectrum::BANDS;
    let slots = bands * 2;
    let span = main.w - 2.0 * theme::PAGE_PAD;
    let slot = span / slots as f32;
    let bar_w = (slot * VISUALIZER_FILL).round().max(2.0);
    let radius = bar_w * 0.5;
    let max_h = main.h * VISUALIZER_HEIGHT;
    let base = (main.y + main.h - VISUALIZER_BASE).round();
    let tone = |amount: f32| mix(theme::BG, theme::FG, amount * show);

    let draw = cx.draw();
    for side in [-1.0_f32, 1.0] {
        for band in 0..bands {
            let level = state.spectrum.levels()[band];
            // Band 0 sits either side of the centre line; higher bands step outward.
            let centre = main.x + main.w * 0.5 + side * slot * (band as f32 + 0.5);
            let x = (centre - bar_w * 0.5).round();

            // Never shorter than a dot, so a quiet band still holds its place in the row.
            let h = (max_h * level).max(bar_w).round();
            // The outer bands dim toward the page edges, so the row has no hard ends.
            let reach = band as f32 / (bands - 1) as f32;
            let edge = 1.0 - 0.55 * ease_out(((reach - 0.55) / 0.45).max(0.0)).powi(2);
            let amount = VISUALIZER_QUIET + (VISUALIZER_LOUD - VISUALIZER_QUIET) * level;
            let colour = tone(amount * edge);
            draw.round_rect(Rect::new(x, base - h, bar_w, h), radius, colour);

        }
    }
}

/// A plausible still spectrum for `--shot`: strong bass, a lively middle, highs tailing off.
fn demo_spectrum() -> [f32; spectrum::BANDS] {
    let mut levels = [0.0; spectrum::BANDS];
    for (band, level) in levels.iter_mut().enumerate() {
        let t = band as f32 / (spectrum::BANDS - 1) as f32;
        let kick = 0.3 * (-((t - 0.08) / 0.07).powi(2)).exp();
        let body = 0.62 - 0.42 * t + 0.1 * (t * 19.0).sin() + 0.06 * (t * 47.0).cos();
        *level = (body + kick).clamp(0.04, 0.95);
    }
    levels
}

/// The lyric page. It is a page of its own, in place of the tab's content: no scrim, no lift, and
/// nothing fades in or out when it is opened or closed.
fn lyrics_page(
    cx: &mut Cx,
    state: &mut State,
    track: Option<&TrackView>,
    main: Rect,
    now: Instant,
    actions: &mut Vec<Action>,
) {
    // Behind everything else on the page: the visualizer and its tint go down first.
    lyric_visualizer(cx, state, track, main);

    let shown = state
        .lyrics
        .as_ref()
        .filter(|v| track.is_some_and(|t| t.video_id == v.video_id));
    match track {
        None => empty_state(cx, main, Icon::Music, "Nothing playing", "Play a song to see its lyrics."),
        Some(track) => match shown {
            // Still asking the providers: nothing yet, rather than a placeholder that flickers.
            None => {}
            Some(shown) => {
                let age = since(now, shown.arrived);
                let key = shown.gen.wrapping_mul(KEY_SPAN);
                match shown.lookup.as_ref() {
                    None => {}
                    Some(Lookup::Found(lyrics)) => match &lyrics.body {
                        Body::Synced(lines) => {
                            let position_ms = (track.position.max(0.0) * 1000.0) as u64;
                            lyrics_synced(
                                cx,
                                lines,
                                Playhead {
                                    position_ms,
                                    duration: track.duration,
                                },
                                &mut state.lyrics_follow,
                                Reveal { age, key },
                                main,
                                now,
                                actions,
                            );
                        }
                        Body::Plain(lines) => lyrics_plain(cx, lines, Reveal { age, key }, main),
                        Body::Instrumental => {
                            empty_state(cx, main, Icon::Music, "Instrumental", "This song has no words.")
                        }
                    },
                    Some(Lookup::NotFound) => empty_state(
                        cx,
                        main,
                        Icon::Music,
                        "No lyrics for this song",
                        "Neither Boidu nor LRCLIB has them.",
                    ),
                    Some(Lookup::Unavailable(_)) => empty_state(
                        cx,
                        main,
                        Icon::Music,
                        "Lyrics are unavailable",
                        "The lyric services could not be reached. Playing the song again tries once more.",
                    ),
                }
            }
        },
    }
}

/// Where the playhead is, for the lyrics that follow it.
struct Playhead {
    position_ms: u64,
    /// Seconds, as the bar knows it.
    duration: f32,
}

/// How long the current lyric set has been there (`age`), and the spring key base for that set.
#[derive(Clone, Copy)]
struct Reveal {
    age: f32,
    key: u32,
}

/// Synced lyrics. Every line is set at the same size, so the line being sung never changes size or
/// wrapping. It brightens while the rest sit in the faint tone, and the page keeps that line in the
/// middle (see [`LyricsFollow`]). A click seeks.
#[allow(clippy::too_many_arguments)]
fn lyrics_synced(
    cx: &mut Cx,
    lines: &[TimedLine],
    playhead: Playhead,
    follow: &mut LyricsFollow,
    reveal: Reveal,
    view: Rect,
    now: Instant,
    actions: &mut Vec<Action>,
) {
    let (x, width) = lyric_column(view);
    let active = active_line(lines, playhead.position_ms);

    // A new set starts from scratch: the page centres on its first line without a glide.
    if follow.set != Some(reveal.key) {
        *follow = LyricsFollow {
            set: Some(reveal.key),
            ..LyricsFollow::default()
        };
    }

    let slots: Vec<f32> = lines
        .iter()
        .map(|line| {
            if line.text.is_empty() {
                PAUSE_H
            } else {
                cx.text_block_height(width, &line.text, LYRIC_SIZE, LYRIC_LEADING)
            }
        })
        .collect();
    let mut tops = Vec::with_capacity(lines.len());
    // Half a page of space above the first line, so it can sit in the middle like any other.
    let mut y = view.h * 0.5;
    for slot in &slots {
        tops.push(y);
        y += slot + LYRIC_GAP;
    }
    let content_h = y + view.h * 0.5;

    // The page keeps the line being sung in the middle.
    let target = match active {
        Some(i) => tops[i] + slots[i] * 0.5 - view.h * 0.5,
        None => 0.0,
    };

    // Once the reader has been still for a while, the page takes its place back on the line being sung.
    if follow.reader && follow.scrolled_at.is_none_or(|at| since(now, at) >= LYRIC_IDLE_SECS) {
        follow.reader = false;
    }
    if !follow.reader {
        let behavior = if follow.placed {
            ScrollBehavior::Smooth
        } else {
            ScrollBehavior::Instant
        };
        follow.placed = true;
        cx.ui.scroll_to("lyrics", ScrollToOptions { top: target, behavior });
    }
    let body = cx.ui.scroll_begin("lyrics", view, content_h);

    // Every line's spring runs each frame, even one that is off screen, so it is right when it returns.
    let emphasis: Vec<f32> = (0..lines.len())
        .map(|i| {
            cx.anims
                .toggle("lyric_on", reveal.key + i as u32, active == Some(i), LYRIC_RATE)
        })
        .collect();

    for (i, line) in lines.iter().enumerate() {
        if line.text.is_empty() {
            continue;
        }
        let is_active = active == Some(i);
        let text_h = cx.text_block_height(width, &line.text, LYRIC_SIZE, LYRIC_LEADING);
        let arrived = arrival(reveal.age, i);
        // Centred in its slot, so a line that grows or shrinks stays on the middle of its place.
        let top = body.y + tops[i] + (slots[i] - text_h) * 0.5 + (1.0 - arrived) * ENTER_RISE;
        if top + text_h < view.y || top > view.y + view.h {
            continue;
        }
        let rect = Rect::new(x, top, width, text_h);
        let over = cx.over_clickable(rect);
        let hv = cx
            .anims
            .toggle("lyric_hover", reveal.key + i as u32, over && !is_active, rate::SNAP);
        let rest = mix(theme::FAINT, theme::FG, emphasis[i]);
        let colour = revealed(mix(rest, theme::MUTED, hv), arrived);
        cx.text_block(rect, &line.text, LYRIC_SIZE, colour, LYRIC_LEADING, Align::Center);
        if over && cx.clicked(rect) && playhead.duration > 0.0 {
            let fraction = (line.time_ms as f32 / 1000.0 / playhead.duration).clamp(0.0, 1.0);
            actions.push(Action::Seek(fraction));
            // A click asks to be synced to this line, so the page takes its place back at once.
            follow.reader = false;
            follow.scrolled_at = None;
        }
    }
    cx.ui.scroll_end();
}

/// Plain lyrics: untimed, so nothing is highlighted, and the page scrolls by hand. Lines arrive
/// with the same stagger as synced ones.
fn lyrics_plain(cx: &mut Cx, lines: &[String], reveal: Reveal, view: Rect) {
    let (x, width) = lyric_column(view);
    let heights: Vec<f32> = lines
        .iter()
        .map(|line| {
            if line.is_empty() {
                STANZA_H
            } else {
                cx.text_block_height(width, line, PLAIN_SIZE, PLAIN_LEADING)
            }
        })
        .collect();
    let mut tops = Vec::with_capacity(lines.len());
    let mut y = PAGE_TOP;
    for height in &heights {
        tops.push(y);
        y += height + PLAIN_GAP;
    }
    let body = cx.ui.scroll_begin("lyrics_plain", view, y + STANZA_H);
    for (i, line) in lines.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        let arrived = arrival(reveal.age, i);
        let rect = Rect::new(
            x,
            body.y + tops[i] + (1.0 - arrived) * ENTER_RISE,
            width,
            heights[i],
        );
        let colour = revealed(theme::FG, arrived);
        cx.text_block(rect, line, PLAIN_SIZE, colour, PLAIN_LEADING, Align::Center);
    }
    cx.ui.scroll_end();
}

/// An icon, title and hint, centred in `main`, for when there are no lyrics to show.
fn empty_state(cx: &mut Cx, main: Rect, icon: Icon, title: &str, hint: &str) {
    let center_x = main.x + main.w * 0.5;
    let y = main.h * 0.5 - 8.0;
    cx.icon(icon, (center_x, y - 54.0), 38.0, theme::FAINT);
    cx.text(
        Rect::new(main.x, y - 16.0, main.w, 28.0),
        title,
        18.0,
        theme::FG,
        Align::Center,
    );
    cx.text_fit(
        Rect::new(main.x + theme::PAGE_PAD, y + 14.0, main.w - theme::PAGE_PAD * 2.0, 22.0),
        hint,
        13.0,
        theme::MUTED,
        Align::Center,
    );
}

/// Between the parts of a row's meta line: artist, then album.
const ROW_SEP: &str = "  ·  ";

/// The geometry of the pages a link opens. A tile is an album on an artist's page; the picture is an
/// artist's, round; the cover is an album's. Each section has a heading, then its content.
const TILE: f32 = 156.0;
const TILE_GAP: f32 = 24.0;
/// Under a tile: its title, then its meta line.
const TILE_TEXT_H: f32 = 52.0;
const PICTURE: f32 = 120.0;
const COVER_BIG: f32 = 176.0;
const SECTION_H: f32 = 40.0;
/// A queue row's drag handle (its width, on the left) and the radius of its remove button (right).
const HANDLE_W: f32 = 26.0;
const TRASH_R: f32 = 14.0;
/// The draw layer for a queue row being dragged: above the page's rows, below the chrome.
const DRAG_LAYER: i32 = 1;
/// The draw layer for the floating chrome (search bar, back button, caption buttons): above the page.
const CHROME_LAYER: i32 = 2;

/// Where a page's content starts inside its scroll area: below the search bar's row, so the content
/// rests clear of it. Scrolled, the content runs up under the floating bar and the window edge.
const PAGE_TOP: f32 = theme::TOP_H + 8.0;

/// A scrolling list of tracks: a cover, the title and artist, the album, the length. Rows fade in
/// in a short stagger, wash on hover, and show an equalizer over their cover while they play.
#[allow(clippy::too_many_arguments)]
fn track_list(
    cx: &mut Cx,
    track: Option<&TrackView>,
    rows: &[Row],
    view: Rect,
    scroll_id: &str,
    base: u32,
    age: f32,
    queued: bool,
    actions: &mut Vec<Action>,
) {
    // The rows start below the floating search bar's row when at rest; scrolled, they run under it.
    // The scroll area starts below the caption buttons, so its scrollbar never runs up behind them.
    // The rows still rest at the same place: their top spacing is shortened by that same amount.
    let view = below_caption(view);
    let top = theme::TOP_H - SCROLL_TOP;
    let content_h = top + rows.len() as f32 * theme::ROW_H + 24.0;
    let body = cx.ui.scroll_begin(scroll_id, view, content_h);
    let play: fn(usize) -> Action = if queued { Action::PlayQueued } else { Action::Play };
    track_rows(cx, track, rows, view, body, body.y + top, base, age, queued, play, 0, actions);
    cx.ui.scroll_end();
}

/// The rows of a track list, from `top` down, inside the open scroll area `body`. `view` is the
/// visible part of the page: a row takes the pointer only inside it. A press on a row plays it through
/// `play`, unless it landed on the artist or the album, which opens that page instead, or on the save
/// button, which saves the song.
///
/// `rows` may be only a window of a longer list (a big collection builds just what is on screen):
/// `first` is the position of `rows[0]` in the whole list, which places the rows and numbers them
/// for `play`.
#[allow(clippy::too_many_arguments)]
fn track_rows(
    cx: &mut Cx,
    track: Option<&TrackView>,
    rows: &[Row],
    view: Rect,
    body: Rect,
    top: f32,
    base: u32,
    age: f32,
    queued: bool,
    play: fn(usize) -> Action,
    first: usize,
    actions: &mut Vec<Action>,
) {
    let playing_id = track.map(|track| track.video_id.as_str());
    let wide = view.w > 760.0;
    let pointer_in_view = cx.pointer_in(view);

    // A queue row is moved by its handle. Letting go ends the drag and asks for the move to the
    // place the row was last over, so the list lands where the reader left it.
    if queued {
        if let Some(done) = cx.queue_drag.filter(|_| !cx.pointer.down) {
            cx.queue_drag = None;
            if done.to != done.from {
                actions.push(Action::MoveQueued { from: done.from, to: done.to });
            }
        }
    }
    // While the drag is on, the dragged row follows the pointer and records where it would land.
    let mut drag = if queued { cx.queue_drag } else { None };
    if let Some(current) = drag.as_mut() {
        let row_top = cx.pointer.pos.1 - current.grab;
        let last = rows.len().saturating_sub(1);
        current.to = (((row_top - top) / theme::ROW_H).round().max(0.0) as usize).min(last);
        cx.queue_drag = drag;
    }

    for (n, row) in rows.iter().enumerate() {
        let i = first + n;
        let dragged = drag.is_some_and(|d| d.from == i);
        let y0 = match drag {
            Some(d) if dragged => cx.pointer.pos.1 - d.grab,
            Some(d) => top + d.slot(i) as f32 * theme::ROW_H,
            None => top + i as f32 * theme::ROW_H,
        };
        if y0 + theme::ROW_H < view.y - 4.0 || y0 > view.y + view.h + 4.0 {
            continue;
        }
        let enter = ease_out((age - i.min(14) as f32 * 0.02) / 0.24);
        let a = enter;
        if a <= 0.002 {
            continue;
        }
        let y = y0 + (1.0 - enter) * 6.0;
        // 28 px of margin on the right keeps the scrollbar clear of the rows.
        let pill = Rect::new(body.x, y + 4.0, (body.w - 28.0).max(0.0), theme::ROW_H - 8.0);
        let key = base + i as u32;

        let hovered = pointer_in_view && cx.over_clickable(pill);
        let hv = cx.anims.toggle("row", key, hovered, rate::SNAP);
        let press = cx.anims.toggle("row_press", key, hovered && cx.pointer.down, rate::SNAP);
        let playing = !queued && playing_id == Some(row.video_id);

        // The playing row keeps a resting fill that brightens under the pointer.
        let wash = if playing {
            fade(mix(theme::SELECT, theme::PRESS, hv * (0.5 + 0.5 * press)), a)
        } else {
            fade(mix(theme::HOVER, theme::PRESS, press), hv * a)
        };
        // A dragged row is lifted: drawn in its own layer, over an opaque ground so the rows it
        // passes do not show through.
        let lift = dragged.then(|| cx.draw().set_layer(DRAG_LAYER));
        if dragged {
            cx.draw().round_rect(pill, 10.0, fade(theme::RAISED, a));
        }
        cx.draw().round_rect(pill, 10.0, wash);

        // Cover, with the play glyph (hovered) or the equalizer (playing) over it.
        let handle_w = if queued { HANDLE_W } else { 0.0 };
        let cover = Rect::new(pill.x + 8.0 + handle_w, pill.y + 6.0, 44.0, 44.0);
        cx.cover(row.thumbnail, cover, 7.0, a);
        cover_overlay(cx, track, row.video_id, playing, cover, 7.0, a, hv);

        // A queue row's grip on the left: grabbing it moves the row. The remove button sits on the right.
        let mut handle_pressed = false;
        let mut trash_clicked = false;
        let trash_center = (pill.x + pill.w - 16.0 - TRASH_R, pill.y + pill.h * 0.5);
        let trash = Rect::new(
            trash_center.0 - TRASH_R,
            trash_center.1 - TRASH_R,
            TRASH_R * 2.0,
            TRASH_R * 2.0,
        );
        if queued {
            let handle = Rect::new(pill.x, pill.y, HANDLE_W, pill.h);
            let grip_hover = pointer_in_view && cx.over_clickable(handle);
            let gv = cx.anims.toggle("queue_grip", key, grip_hover, rate::SNAP);
            let grip = mix(mix(theme::FAINT, theme::MUTED, hv), theme::FG, gv);
            let grip_center = (handle.x + handle.w * 0.5, handle.y + handle.h * 0.5);
            cx.icon(Icon::GripVertical, grip_center, 16.0, fade(grip, a));
            handle_pressed = pointer_in_view && cx.clicked(handle);
            if handle_pressed && cx.queue_drag.is_none() {
                cx.queue_drag = Some(QueueDrag { from: i, to: i, grab: cx.pointer.pos.1 - y0 });
            }

            let trash_hover = pointer_in_view && cx.over_clickable(trash);
            let tv = cx.anims.toggle("queue_trash", key, trash_hover, rate::SNAP);
            if tv > 0.002 {
                cx.draw().round_rect(trash, TRASH_R, fade(theme::CLOSE_RED, 0.16 * tv * a));
            }
            cx.icon(Icon::Trash2, trash_center, 16.0, fade(theme::CLOSE_RED, a));
            trash_clicked = pointer_in_view && cx.clicked(trash);
        }

        let text_x = cover.x + cover.w + 14.0;
        // Clear of the scrollbar, which floats in the right-hand margin. A queue row leaves room
        // for its remove button.
        let end = if queued { trash.x - 12.0 } else { pill.x + pill.w - 16.0 };
        let duration = Rect::new(end - 56.0, pill.y, 56.0, pill.h);
        cx.text(duration, row.duration, 12.5, fade(theme::MUTED, a), Align::Right);

        let tail = duration.x;

        // Wide rows put the album in a column of its own; narrow ones run the artist and album on
        // one line. Either way the title takes what is left.
        let album_w = (pill.w * 0.26).clamp(120.0, 300.0);
        let album_x = tail - 20.0 - album_w;
        // The save button sits right of the names, so the names stop short of it by its width and gap.
        let save_space = 14.0 + SAVE_ROW_R * 2.0;
        let full_w = (tail - 16.0 - text_x - save_space).max(40.0);
        let title_w = if wide { (album_x - 20.0 - text_x - save_space).max(40.0) } else { full_w };

        // The save button sits just right of the song and artist names, not out at the far edge.
        let title_text_w = cx.text_width(row.title, 14.5);
        let meta_text_w = if wide {
            cx.text_width(&row.artist, 12.5)
        } else {
            let sep = if row.artist.is_empty() || row.album.is_empty() { 0.0 } else { cx.text_width(ROW_SEP, 12.5) };
            cx.text_width(&row.artist, 12.5) + sep + cx.text_width(row.album, 12.5)
        };
        let names_w = title_text_w.max(meta_text_w).min(title_w);
        let save_x = text_x + names_w + 14.0 + SAVE_ROW_R;
        let saved = cx.saved.songs.contains(row.video_id);
        let save_clicked = save_button(
            cx,
            hash(row.video_id),
            (save_x, pill.y + pill.h * 0.5),
            SAVE_ROW_R,
            saved,
            pointer_in_view,
            hv,
            a,
        );
        // The title links to the song's album, when the row knows it: a press on it opens the album.
        let title_opened = title_link(
            cx,
            Rect::new(text_x, pill.y + 8.0, title_w, 22.0),
            row.title,
            14.5,
            fade(theme::FG, a),
            row.album_id.map(|id| Detail::Album(id.to_owned())),
        );

        // The artist and the album are links: each brightens under the pointer, and a press on one
        // opens its page rather than playing the row.
        let meta_y = pill.y + 28.0;
        let opened = if wide {
            let artist = row_text(
                cx,
                "link_artist",
                key,
                Rect::new(text_x, meta_y, title_w, 18.0),
                &row.artist,
                12.5,
                a,
                row.artist_id.map(|id| Detail::Artist(id.to_owned())),
            );
            let album = row_text(
                cx,
                "link_album",
                key,
                Rect::new(album_x, pill.y, album_w, pill.h),
                row.album,
                13.0,
                a,
                row.album_id.map(|id| Detail::Album(id.to_owned())),
            );
            artist.or(album)
        } else {
            let artist_w = if row.artist.is_empty() {
                0.0
            } else {
                cx.text_width(&row.artist, 12.5).min(full_w)
            };
            let sep_w = if row.artist.is_empty() || row.album.is_empty() {
                0.0
            } else {
                cx.text_width(ROW_SEP, 12.5)
            };
            let artist = row_text(
                cx,
                "link_artist",
                key,
                Rect::new(text_x, meta_y, full_w, 18.0),
                &row.artist,
                12.5,
                a,
                row.artist_id.map(|id| Detail::Artist(id.to_owned())),
            );
            if sep_w > 0.0 {
                cx.text_fit(
                    Rect::new(text_x + artist_w, meta_y, sep_w, 18.0),
                    ROW_SEP,
                    12.5,
                    fade(theme::MUTED, a),
                    Align::Left,
                );
            }
            let album_text_x = text_x + artist_w + sep_w;
            let album = row_text(
                cx,
                "link_album",
                key,
                Rect::new(album_text_x, meta_y, (text_x + full_w - album_text_x).max(0.0), 18.0),
                row.album,
                12.5,
                a,
                row.album_id.map(|id| Detail::Album(id.to_owned())),
            );
            artist.or(album)
        };

        if let Some(layer) = lift {
            cx.draw().set_layer(layer);
        }

        // A right press on a row opens the track menu for it, at the pointer.
        if pointer_in_view && cx.right_clicked(pill) {
            actions.push(Action::OpenMenu(TrackMenu { hit: row.song(), at: cx.pointer.pos }));
        }

        if save_clicked {
            actions.push(Action::SaveSong(row.song()));
        } else if let Some(target) = title_opened.or(opened) {
            actions.push(Action::Open(target));
        } else if trash_clicked {
            actions.push(Action::RemoveQueued(i));
        } else if handle_pressed {
            // A press on the handle grabs the row; it does not play it.
        } else if pointer_in_view && cx.clicked(pill) {
            actions.push(play(i));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------------------------

/// The save button's green. Opaque, like every colour here.
const SAVE_GREEN: Color = Color::rgb(0x1e, 0xd7, 0x60);
/// Radii of the save button where it appears: on a row, on the player bar, and on a page's header.
const SAVE_ROW_R: f32 = 9.0;
const SAVE_BAR_R: f32 = 12.0;
const SAVE_HEAD_R: f32 = 16.0;
/// The artist page's is a little smaller than the album page's.
const SAVE_ARTIST_R: f32 = 13.0;
/// The green fill's spring: it grows from the centre in about a fifth of a second.
const SAVE_FILL_RATE: f32 = 620.0;
/// Keeps the same song's button on the bar and on a row from sharing springs, which would step them
/// twice a frame when both are on screen.
const SALT_BAR: u32 = 0x5bd1_e995;
const SALT_HEAD: u32 = 0x1b87_3593;

/// A circular save button: a ring with a plus, which fills green from the centre while the plus
/// gives way to a check when the thing is saved, and does the same backwards when it is unsaved.
/// Returns whether it was pressed this frame.
///
/// `live` says whether the pointer may use it at all (it is on the visible page), `warm` is how lit
/// the surface it sits on is (a hovered row), and `alpha` fades the whole button with its row.
#[allow(clippy::too_many_arguments)]
fn save_button(
    cx: &mut Cx,
    key: u32,
    center: P,
    radius: f32,
    saved: bool,
    live: bool,
    warm: f32,
    alpha: f32,
) -> bool {
    // Everything sits on the device pixel grid, so the ring and the glyphs are drawn sharp.
    let center = (cx.snap(center.0), cx.snap(center.1));
    let reach = radius + 6.0;
    let hit = Rect::new(center.0 - reach, center.1 - reach, reach * 2.0, reach * 2.0);
    let hover = live && cx.over_clickable(hit);
    let hv = cx.anims.toggle("save_hv", key, hover, rate::SNAP);
    // One critically damped spring drives the whole change, so nothing overshoots or fights. The
    // glyphs are never scaled (a scaled texture is a blurred one): they hand over by opacity while
    // the green grows.
    let t = cx
        .anims
        .spring("save_fill", key, if saved { 1.0 } else { 0.0 }, SAVE_FILL_RATE)
        .clamp(0.0, 1.0);

    let idle = mix(mix(theme::FAINT, theme::MUTED, warm), theme::FG, hv);
    let square = Rect::new(center.0 - radius, center.1 - radius, radius * 2.0, radius * 2.0);
    cx.draw().ring_arc(
        square,
        radius,
        1.5,
        -std::f32::consts::FRAC_PI_2,
        std::f32::consts::TAU,
        fade(mix(idle, SAVE_GREEN, t), alpha),
    );
    if t > 0.003 {
        let r = radius * t;
        // Drawn as a full-thickness arc, like the ring: its quad has a pixel of room for the
        // anti-aliased edge, whereas a rounded rect's quad is exactly the square and clips the
        // edge flat where the circle touches it, which made the filled button look blobby.
        cx.draw().ring_arc(
            Rect::new(center.0 - r, center.1 - r, r * 2.0, r * 2.0),
            r,
            r,
            -std::f32::consts::FRAC_PI_2,
            std::f32::consts::TAU,
            fade(SAVE_GREEN, alpha),
        );
    }
    let glyph = ((radius * 1.3) / 2.0).round() * 2.0;
    let plus = (1.0 - t * 2.5).clamp(0.0, 1.0);
    let check = ((t - 0.4) * 2.5).clamp(0.0, 1.0);
    if plus > 0.0 {
        cx.icon(Icon::Plus, center, glyph, fade(idle, alpha * plus));
    }
    if check > 0.0 {
        cx.icon(Icon::Check, center, glyph, fade(theme::ON_ACCENT, alpha * check));
    }
    live && cx.clicked(hit)
}

// ---------------------------------------------------------------------------------------------
// The track menu
// ---------------------------------------------------------------------------------------------

/// The track menu's width, its rows and the padding round them. Fixed, so its place is known before
/// anything is drawn.
const MENU_W: f32 = 212.0;
const MENU_ITEM_H: f32 = 34.0;
const MENU_PAD: f32 = 6.0;
/// How far the menu keeps from the window's edges when it would otherwise run off them.
const MENU_MARGIN: f32 = 8.0;
/// The menu is drawn above the bar, the sidebar and the caption buttons.
const MENU_LAYER: i32 = 3;

/// What a line of the track menu does when pressed.
enum MenuCommand {
    Album(String),
    Artist(String),
    /// Save the song, or unsave it. The menu stays open.
    Save,
    /// Take the song out of the queue, at this place in it.
    Remove(usize),
    /// Add the song to the end of the queue.
    Add,
}

/// One line of the track menu.
struct MenuEntry {
    label: &'static str,
    icon: Icon,
    command: MenuCommand,
}

/// The lines the track menu shows for its song right now. A line that cannot do anything at the
/// moment is left out altogether: no album or artist is known for the song, or the page on screen
/// already is that album or artist.
fn track_menu_entries(state: &State, menu: &TrackMenu) -> Vec<MenuEntry> {
    let on_screen = |detail: &Detail| state.pages.top.as_ref() == Some(detail);
    let mut entries = Vec::new();
    if let Some(id) = &menu.hit.album_id {
        if !on_screen(&Detail::Album(id.clone())) {
            entries.push(MenuEntry {
                label: "Go to album",
                icon: Icon::Disc,
                command: MenuCommand::Album(id.clone()),
            });
        }
    }
    if let Some(id) = &menu.hit.artist_id {
        if !on_screen(&Detail::Artist(id.clone())) {
            entries.push(MenuEntry {
                label: "Go to artist",
                icon: Icon::User,
                command: MenuCommand::Artist(id.clone()),
            });
        }
    }
    if state.library.song_saved(&menu.hit.video_id) {
        entries.push(MenuEntry { label: "Unsave", icon: Icon::Check, command: MenuCommand::Save });
    } else {
        entries.push(MenuEntry { label: "Save", icon: Icon::Plus, command: MenuCommand::Save });
    }
    match state.playlist.queue.iter().position(|hit| hit.video_id == menu.hit.video_id) {
        Some(index) => entries.push(MenuEntry {
            label: "Remove from queue",
            icon: Icon::Trash2,
            command: MenuCommand::Remove(index),
        }),
        None => entries.push(MenuEntry {
            label: "Add to queue",
            icon: Icon::ListMusic,
            command: MenuCommand::Add,
        }),
    }
    entries
}

/// Where the track menu sits, with `rows` lines: its top-left corner at the pointer, pulled back inside
/// the window if it would run off an edge.
fn menu_rect(menu: &TrackMenu, rows: usize, width: f32, height: f32) -> Rect {
    let h = MENU_PAD * 2.0 + MENU_ITEM_H * rows as f32;
    let x = menu.at.0.min(width - MENU_W - MENU_MARGIN).max(MENU_MARGIN);
    let y = menu.at.1.min(height - h - MENU_MARGIN).max(MENU_MARGIN);
    Rect::new(x, y, MENU_W, h)
}

/// The track menu, drawn over the whole window and acting on its song. Saving keeps the menu open, so
/// several songs can be saved in a row; every other line closes it.
fn draw_track_menu(
    cx: &mut Cx,
    state: &mut State,
    entries: &[MenuEntry],
    width: f32,
    height: f32,
    actions: &mut Vec<Action>,
) {
    let Some(menu) = state.menu.clone() else { return };
    let rect = menu_rect(&menu, entries.len(), width, height);

    let previous = cx.draw().set_layer(MENU_LAYER);
    // A thin edge in a lighter tone, then the face inside it.
    cx.draw().round_rect(rect, 10.0, theme::TRACK);
    cx.draw().round_rect(
        Rect::new(rect.x + 1.0, rect.y + 1.0, rect.w - 2.0, rect.h - 2.0),
        9.0,
        theme::RAISED,
    );

    for (i, entry) in entries.iter().enumerate() {
        let row = Rect::new(
            rect.x + MENU_PAD,
            rect.y + MENU_PAD + i as f32 * MENU_ITEM_H,
            rect.w - MENU_PAD * 2.0,
            MENU_ITEM_H,
        );
        let hovered = cx.over_clickable(row);
        let hv = cx.anims.toggle("menu_item", i as u32, hovered, rate::SNAP);
        if hv > 0.002 {
            cx.draw().round_rect(row, 7.0, mix(theme::RAISED, theme::PRESS, hv));
        }
        let ink = mix(theme::MUTED, theme::FG, hv);
        cx.icon(entry.icon, (row.x + 18.0, row.y + row.h * 0.5), 16.0, ink);
        cx.text(
            Rect::new(row.x + 36.0, row.y, row.w - 44.0, row.h),
            entry.label,
            13.5,
            ink,
            Align::Left,
        );

        if cx.clicked(row) {
            let (action, close) = match &entry.command {
                MenuCommand::Album(id) => (Action::Open(Detail::Album(id.clone())), true),
                MenuCommand::Artist(id) => (Action::Open(Detail::Artist(id.clone())), true),
                MenuCommand::Save => (Action::SaveSong(menu.hit.clone()), false),
                MenuCommand::Remove(index) => (Action::RemoveQueued(*index), true),
                MenuCommand::Add => (Action::Enqueue(menu.hit.clone()), true),
            };
            actions.push(action);
            if close {
                state.menu = None;
            }
        }
    }
    cx.draw().set_layer(previous);
}

/// One piece of a row's text. Plain text stays muted. A link (`target` is set) brightens to `FG`
/// under the pointer, and a press on it returns its target. The hit area is the text itself.
#[allow(clippy::too_many_arguments)]
fn row_text(
    cx: &mut Cx,
    name: &'static str,
    key: u32,
    rect: Rect,
    text: &str,
    size: f32,
    alpha: f32,
    target: Option<Detail>,
) -> Option<Detail> {
    if text.is_empty() {
        return None;
    }
    let Some(target) = target else {
        cx.text_fit(rect, text, size, fade(theme::MUTED, alpha), Align::Left);
        return None;
    };
    let hit = Rect::new(rect.x, rect.y, cx.text_width(text, size).min(rect.w), rect.h);
    let hovered = cx.over_clickable(hit);
    let hv = cx.anims.toggle(name, key, hovered, rate::SNAP);
    cx.text_fit(rect, text, size, fade(mix(theme::MUTED, theme::FG, hv), alpha), Align::Left);
    cx.clicked(hit).then_some(target)
}

/// A song's title, when it names the album the song is from: drawn in `color`, and a press on it
/// returns the album.
#[allow(clippy::too_many_arguments)]
fn title_link(
    cx: &mut Cx,
    rect: Rect,
    text: &str,
    size: f32,
    color: Color,
    target: Option<Detail>,
) -> Option<Detail> {
    if text.is_empty() {
        return None;
    }
    let Some(target) = target else {
        cx.text_fit(rect, text, size, color, Align::Left);
        return None;
    };
    let hit = Rect::new(rect.x, rect.y, cx.text_width(text, size).min(rect.w), rect.h);
    cx.over_clickable(hit);
    cx.text_fit(rect, text, size, color, Align::Left);
    cx.clicked(hit).then_some(target)
}

/// The scroll area of a detail page. Each page keeps its own place, so back returns to where it was.
fn detail_scroll_id(detail: &Detail) -> String {
    match detail {
        Detail::Artist(id) => format!("artist:{id}"),
        Detail::Album(id) => format!("album:{id}"),
    }
}

/// The page a link opened, in the space the results or the tab would fill.
fn detail_page(
    cx: &mut Cx,
    state: &State,
    track: Option<&TrackView>,
    detail: &Detail,
    main: Rect,
    view: Rect,
    actions: &mut Vec<Action>,
) {
    let scroll_id = detail_scroll_id(detail);
    match detail {
        Detail::Artist(id) => match state.artists.get(id) {
            Some(Loaded::Ready(page)) => {
                artist_page(cx, track, page, &state.lengths, &scroll_id, view, actions)
            }
            Some(Loaded::Failed(message)) => failed_page(cx, main, message),
            Some(Loaded::Loading) | None => loading_page(cx, main),
        },
        Detail::Album(id) => match state.albums.get(id) {
            Some(Loaded::Ready(page)) => {
                album_page(cx, track, page, &state.lengths, &scroll_id, view, actions)
            }
            Some(Loaded::Failed(message)) => failed_page(cx, main, message),
            Some(Loaded::Loading) | None => loading_page(cx, main),
        },
    }
}

fn loading_page(cx: &mut Cx, main: Rect) {
    cx.text(
        Rect::new(main.x, main.h * 0.5 - 14.0, main.w, 28.0),
        "Loading…",
        14.5,
        theme::MUTED,
        Align::Center,
    );
}

fn failed_page(cx: &mut Cx, main: Rect, message: &str) {
    empty_state(cx, main, Icon::Disc, "Couldn't open this page", message);
}

/// How many album tiles fit across `width`.
fn tile_columns(width: f32) -> usize {
    (((width + TILE_GAP) / (TILE + TILE_GAP)).floor() as usize).max(1)
}

/// Where a detail page's scroll area starts: just under the caption buttons, so its scrollbar
/// never runs up behind them. The page's own top spacing shrinks by the same amount.
const SCROLL_TOP: f32 = theme::SEARCH_TOP + (theme::SEARCH_H + theme::CAPTION_H) * 0.5 + 8.0;
const DETAIL_TOP: f32 = PAGE_TOP - SCROLL_TOP;

fn below_caption(view: Rect) -> Rect {
    Rect::new(view.x, view.y + SCROLL_TOP, view.w, (view.h - SCROLL_TOP).max(0.0))
}

/// A section heading on a page, in the text colour, as the page words it.
fn section_heading(cx: &mut Cx, rect: Rect, text: &str) {
    cx.text_fit(rect, text, 16.0, theme::FG, Align::Left);
}

/// An artist's channel: the round picture and name, the songs, then each shelf of albums as a grid.
fn artist_page(
    cx: &mut Cx,
    track: Option<&TrackView>,
    page: &ArtistPage,
    lengths: &HashMap<String, String>,
    scroll_id: &str,
    view: Rect,
    actions: &mut Vec<Action>,
) {
    let view = below_caption(view);
    let columns = tile_columns(view.w - 44.0);
    // The songs are rows of this artist's own page, so they show their album, not the artist.
    let songs: Vec<Row> = page
        .songs
        .iter()
        .map(|hit| {
            let mut row = Row::from(hit, lengths);
            row.artist_id = None;
            row
        })
        .collect();

    let mut content_h = DETAIL_TOP + PICTURE + 40.0;
    if !songs.is_empty() {
        content_h += SECTION_H + songs.len() as f32 * theme::ROW_H + 24.0;
    }
    for shelf in &page.shelves {
        let rows = shelf.albums.len().div_ceil(columns) as f32;
        content_h += SECTION_H + rows * (TILE + TILE_TEXT_H + TILE_GAP) + 24.0;
    }
    let body = cx.ui.scroll_begin(scroll_id, view, content_h);
    let left = body.x + 8.0;

    let picture = Rect::new(left, body.y + DETAIL_TOP, PICTURE, PICTURE);
    cx.cover(page.thumbnail.as_deref(), picture, PICTURE * 0.5, 1.0);
    let name_x = picture.x + PICTURE + 28.0;
    let name_w = (body.x + body.w - 36.0 - name_x).max(40.0);
    cx.text_fit(
        Rect::new(name_x, picture.y + PICTURE * 0.5 - 20.0, name_w, 40.0),
        &page.name,
        30.0,
        theme::FG,
        Align::Left,
    );

    // Saving the artist keeps their page on this device (there is no audio to download).
    let saved = cx.saved.artists.contains(&page.browse_id);
    let button = (name_x + SAVE_ARTIST_R, picture.y + PICTURE - SAVE_ARTIST_R - 2.0);
    let pointer_in_view = cx.pointer_in(view);
    if save_button(
        cx,
        hash(&page.browse_id) ^ SALT_HEAD,
        button,
        SAVE_ARTIST_R,
        saved,
        pointer_in_view,
        1.0,
        1.0,
    ) {
        actions.push(Action::SaveArtist(page.clone()));
    }
    cx.text(
        Rect::new(button.0 + SAVE_ARTIST_R + 12.0, button.1 - 10.0, 160.0, 20.0),
        if saved { "Saved" } else { "Save" },
        13.0,
        if saved { theme::FG } else { theme::MUTED },
        Align::Left,
    );

    let mut y = body.y + DETAIL_TOP + PICTURE + 40.0;
    if !songs.is_empty() {
        section_heading(cx, Rect::new(left, y, body.w - 16.0, 28.0), &page.songs_title);
        let rows_top = y + SECTION_H;
        track_rows(
            cx,
            track,
            &songs,
            view,
            body,
            rows_top,
            8000,
            f32::INFINITY,
            false,
            Action::PlayDetail,
            0,
            actions,
        );
        y = rows_top + songs.len() as f32 * theme::ROW_H + 24.0;
    }
    for (shelf_index, shelf) in page.shelves.iter().enumerate() {
        section_heading(cx, Rect::new(left, y, body.w - 16.0, 28.0), &shelf.title);
        let grid_top = y + SECTION_H;
        for (k, card) in shelf.albums.iter().enumerate() {
            let x = left + (k % columns) as f32 * (TILE + TILE_GAP);
            let ty = grid_top + (k / columns) as f32 * (TILE + TILE_TEXT_H + TILE_GAP);
            album_tile(cx, view, actions, (x, ty), card, 9000 + (shelf_index * 100 + k) as u32);
        }
        let rows = shelf.albums.len().div_ceil(columns) as f32;
        y = grid_top + rows * (TILE + TILE_TEXT_H + TILE_GAP) + 24.0;
    }
    cx.ui.scroll_end();
}

/// An album on an artist's page: a square cover, then its title and meta. The whole tile opens the
/// album, and washes on hover like a row does.
fn album_tile(
    cx: &mut Cx,
    view: Rect,
    actions: &mut Vec<Action>,
    origin: (f32, f32),
    card: &AlbumCard,
    key: u32,
) {
    let target = Detail::Album(card.browse_id.clone());
    media_tile(
        cx,
        view,
        actions,
        origin,
        (card.title.as_str(), card.meta.as_str(), card.thumbnail.as_deref()),
        false,
        target,
        key,
    );
}

/// A tile for a page: a square cover (round for an artist), then its title and meta. The whole tile
/// opens `target`, and washes on hover like a row does.
#[allow(clippy::too_many_arguments)]
fn media_tile(
    cx: &mut Cx,
    view: Rect,
    actions: &mut Vec<Action>,
    origin: (f32, f32),
    (title, meta, thumbnail): (&str, &str, Option<&str>),
    round: bool,
    target: Detail,
    key: u32,
) {
    let (x, y) = origin;
    let pill = Rect::new(x - 10.0, y - 10.0, TILE + 20.0, TILE + TILE_TEXT_H + 8.0);
    let pointer_in_view = cx.pointer_in(view);
    let hovered = pointer_in_view && cx.over_clickable(pill);
    let hv = cx.anims.toggle("tile", key, hovered, rate::SNAP);
    cx.draw().round_rect(pill, 12.0, mix(theme::BG, theme::HOVER, hv));
    let radius = if round { TILE * 0.5 } else { 10.0 };
    cx.cover(thumbnail, Rect::new(x, y, TILE, TILE), radius, 1.0);
    cx.text_fit(
        Rect::new(x, y + TILE + 12.0, TILE, 20.0),
        title,
        14.5,
        theme::FG,
        Align::Left,
    );
    cx.text_fit(
        Rect::new(x, y + TILE + 32.0, TILE, 18.0),
        meta,
        12.5,
        theme::MUTED,
        Align::Left,
    );
    if pointer_in_view && cx.clicked(pill) {
        actions.push(Action::Open(target));
    }
}

/// The search results, in one scroll area: the top result as a block of its own, then the tabs
/// (Songs, Artists, Albums), then the kind of result the chosen tab shows. A tab with nothing in it
/// says so in a line. The scroll area keeps the id `search_results`, which the results' scroll
/// position is kept under.
fn search_results(
    cx: &mut Cx,
    state: &State,
    track: Option<&TrackView>,
    view: Rect,
    age: f32,
    actions: &mut Vec<Action>,
) {
    let rows: Vec<Row> = state.results.iter().map(|hit| Row::from(hit, &state.lengths)).collect();
    let columns = tile_columns(view.w - 44.0);
    let grid_h = |count: usize| count.div_ceil(columns) as f32 * (TILE + TILE_TEXT_H + TILE_GAP);
    let tab = state.results_tab;

    // The chosen tab's part: its height, or a line of 'nothing here' when it is empty.
    let part_h = match tab {
        ResultsTab::Songs if !rows.is_empty() => rows.len() as f32 * theme::ROW_H + 24.0,
        ResultsTab::Artists if !state.result_artists.is_empty() => grid_h(state.result_artists.len()) + 24.0,
        ResultsTab::Albums if !state.result_albums.is_empty() => grid_h(state.result_albums.len()) + 24.0,
        _ => EMPTY_PART_H,
    };
    let mut content_h = theme::TOP_H + TAB_ROW_H + part_h;
    if state.top.is_some() {
        content_h += TOP_PAD + TOP_CARD_H + 24.0;
    }

    let body = cx.ui.scroll_begin("search_results", view, content_h);
    let left = body.x + 8.0;
    let mut y = body.y + theme::TOP_H;

    if let Some(top) = &state.top {
        let pill = Rect::new(body.x, y + TOP_PAD, (body.w - 28.0).max(0.0), TOP_CARD_H);
        top_card(cx, view, track, top, pill, actions);
        y += TOP_PAD + TOP_CARD_H + 24.0;
    }

    let labels = ResultsTab::ALL.map(|kind| (kind.label(), kind.icon()));
    let current = ResultsTab::ALL.iter().position(|kind| *kind == tab).unwrap_or(0);
    if let Some(index) = tab_pills(cx, &labels, current, left, y, 8) {
        actions.push(Action::ResultsTab(ResultsTab::ALL[index]));
    }
    y += TAB_ROW_H;

    match tab {
        ResultsTab::Songs if !rows.is_empty() => {
            track_rows(cx, track, &rows, view, body, y, 0, age, false, Action::Play, 0, actions);
        }
        ResultsTab::Artists if !state.result_artists.is_empty() => {
            for (k, artist) in state.result_artists.iter().enumerate() {
                let x = left + (k % columns) as f32 * (TILE + TILE_GAP);
                let ty = y + (k / columns) as f32 * (TILE + TILE_TEXT_H + TILE_GAP);
                media_tile(
                    cx,
                    view,
                    actions,
                    (x, ty),
                    (artist.name.as_str(), artist.meta.as_str(), artist.thumbnail.as_deref()),
                    true,
                    Detail::Artist(artist.browse_id.clone()),
                    7000 + k as u32,
                );
            }
        }
        ResultsTab::Albums if !state.result_albums.is_empty() => {
            for (k, album) in state.result_albums.iter().enumerate() {
                let x = left + (k % columns) as f32 * (TILE + TILE_GAP);
                let ty = y + (k / columns) as f32 * (TILE + TILE_TEXT_H + TILE_GAP);
                album_tile(cx, view, actions, (x, ty), album, 9000 + k as u32);
            }
        }
        _ => {
            let hint = match tab {
                ResultsTab::Songs => "No songs matched.",
                ResultsTab::Artists => "No artists matched.",
                ResultsTab::Albums => "No albums matched.",
            };
            cx.text(Rect::new(left, y + 12.0, body.w - 16.0, 24.0), hint, 14.0, theme::MUTED, Align::Left);
        }
    }
    cx.ui.scroll_end();
}

/// The overlay over a song's cover, as the song's row draws it: a dim with the spinner while the song
/// loads, the equalizer while it sounds, or nothing while it is playing silently; otherwise the dim
/// and play glyph appear under the pointer (`hv`). `playing` says whether this song is the one that
/// is playing, and `a` fades it in with its row.
#[allow(clippy::too_many_arguments)]
fn cover_overlay(
    cx: &mut Cx,
    track: Option<&TrackView>,
    video_id: &str,
    playing: bool,
    cover: Rect,
    radius: f32,
    a: f32,
    hv: f32,
) {
    // The visualizer exists only while sound is audibly coming out; otherwise the cover shows
    // no dim and no bars, as if it were not playing at all.
    const SILENT_LEVEL: f32 = 0.02;
    let paused = track.is_none_or(|track| track.paused);
    let levels = track.map_or([0.0; 3], |track| track.levels);
    let audible = playing && !paused && levels.iter().any(|level| *level > SILENT_LEVEL);
    // The song being loaded shows the spinner, over its cover, as the bar does.
    let loading = track.is_some_and(|track| track.loading && track.video_id == video_id);
    let dim = if audible || loading {
        1.0
    } else if playing {
        0.0
    } else {
        hv
    };
    cx.draw()
        .round_rect(cover, radius, Color::rgba(0, 0, 0, (130.0 * dim * a) as u8));
    let center = (cover.x + cover.w * 0.5, cover.y + cover.h * 0.5);
    if loading {
        draw_spinner(cx, center, 12.0, fade(theme::FG, a));
    } else if audible {
        icons::equalizer(cx.draw(), center, 16.0, mix(theme::BG, theme::FG, a), levels, true);
    } else if playing {
        // Playing but silent (or paused): nothing over the cover.
    } else if hv > 0.5 {
        // Opaque and fixed size; shown once hover is well under way, so no dark ghost lingers.
        cx.icon(Icon::Play, center, 18.0, theme::FG);
    }
}

/// The search's top result, as a block of its own above the songs: its cover, a "Top result" label,
/// its title and its meta. A song plays, with the same cover overlay as its row; an artist or an
/// album opens its page.
fn top_card(
    cx: &mut Cx,
    view: Rect,
    track: Option<&TrackView>,
    top: &TopResult,
    pill: Rect,
    actions: &mut Vec<Action>,
) {
    let pointer_in_view = cx.pointer_in(view);
    let hovered = pointer_in_view && cx.over_clickable(pill);
    let hv = cx.anims.toggle("top_card", 0, hovered, rate::SNAP);
    cx.draw().round_rect(pill, 14.0, mix(theme::PANEL, theme::SELECT, hv));

    let (title, meta, thumbnail, round, target): (&str, String, Option<&str>, bool, Option<Detail>) =
        match top {
            TopResult::Song(hit) => {
                let mut parts = Vec::new();
                if !hit.artists.is_empty() {
                    parts.push(hit.artists.join(", "));
                }
                if let Some(album) = hit.album.as_deref().filter(|album| !album.is_empty()) {
                    parts.push(album.to_owned());
                }
                (hit.title.as_str(), parts.join(ROW_SEP), hit.thumbnail.as_deref(), false, None)
            }
            TopResult::Artist(artist) => (
                artist.name.as_str(),
                artist.meta.clone(),
                artist.thumbnail.as_deref(),
                true,
                Some(Detail::Artist(artist.browse_id.clone())),
            ),
            TopResult::Album(album) => (
                album.title.as_str(),
                album.meta.clone(),
                album.thumbnail.as_deref(),
                false,
                Some(Detail::Album(album.browse_id.clone())),
            ),
        };

    let cover = Rect::new(pill.x + 16.0, pill.y + 16.0, TOP_COVER, TOP_COVER);
    let radius = if round { TOP_COVER * 0.5 } else { 12.0 };
    cx.cover(thumbnail, cover, radius, 1.0);
    if let TopResult::Song(hit) = top {
        let playing = track.is_some_and(|track| track.video_id == hit.video_id);
        cover_overlay(cx, track, &hit.video_id, playing, cover, radius, 1.0, hv);
    }
    let text_x = cover.x + TOP_COVER + 24.0;
    let text_w = (pill.x + pill.w - 24.0 - text_x).max(40.0);
    cx.text_fit(
        Rect::new(text_x, pill.y + 34.0, text_w, 18.0),
        "Top result",
        12.5,
        theme::MUTED,
        Align::Left,
    );
    cx.text_fit(
        Rect::new(text_x, pill.y + 56.0, text_w, 36.0),
        title,
        26.0,
        theme::FG,
        Align::Left,
    );
    cx.text_fit(
        Rect::new(text_x, pill.y + 98.0, text_w, 20.0),
        &meta,
        13.0,
        theme::MUTED,
        Align::Left,
    );
    if pointer_in_view && cx.clicked(pill) {
        actions.push(match target {
            Some(detail) => Action::Open(detail),
            None => Action::PlayTop,
        });
    }
}

/// Above the search's top card, and the card's height and cover, in pixels.
const TOP_PAD: f32 = 8.0;
const TOP_CARD_H: f32 = 152.0;
/// The search's tab row under the top result: the pills' height, and the room after them.
const TAB_ROW_H: f32 = COLLECTION_TAB_H + 20.0;
/// The room a tab with nothing in it takes, for its line of 'no ... matched'.
const EMPTY_PART_H: f32 = 64.0;
const TOP_COVER: f32 = 120.0;

/// An album: its cover, its title, the artist (a link), and its meta line; then its tracks. The
/// rows do not repeat the album's name, since the page is the album.
fn album_page(
    cx: &mut Cx,
    track: Option<&TrackView>,
    page: &AlbumPage,
    lengths: &HashMap<String, String>,
    scroll_id: &str,
    view: Rect,
    actions: &mut Vec<Action>,
) {
    let tracks: Vec<Row> = page
        .tracks
        .iter()
        .map(|hit| {
            let mut row = Row::from(hit, lengths);
            row.album = "";
            row.album_id = None;
            // Saved from here, a track keeps what the page knows and the row left out.
            row.page = Some(page);
            row
        })
        .collect();
    let view = below_caption(view);
    let rows_top_off = DETAIL_TOP + COVER_BIG + 32.0;
    let content_h = rows_top_off + tracks.len() as f32 * theme::ROW_H + 24.0;
    let body = cx.ui.scroll_begin(scroll_id, view, content_h);
    let left = body.x + 8.0;

    let cover = Rect::new(left, body.y + DETAIL_TOP, COVER_BIG, COVER_BIG);
    cx.cover(page.thumbnail.as_deref(), cover, 12.0, 1.0);

    // Title, artist and meta, standing centred on the height of the cover.
    let text_x = cover.x + COVER_BIG + 28.0;
    let text_w = (body.x + body.w - 36.0 - text_x).max(40.0);
    let block_y = cover.y + (COVER_BIG - 92.0) * 0.5;
    cx.text_fit(
        Rect::new(text_x, block_y, text_w, 34.0),
        &page.title,
        26.0,
        theme::FG,
        Align::Left,
    );
    let artist = row_text(
        cx,
        "link_album_artist",
        0,
        Rect::new(text_x, block_y + 42.0, text_w, 22.0),
        &page.artist,
        15.0,
        1.0,
        page.artist_id.clone().map(Detail::Artist),
    );
    cx.text_fit(
        Rect::new(text_x, block_y + 74.0, text_w, 18.0),
        &page.meta,
        13.0,
        theme::MUTED,
        Align::Left,
    );
    if let Some(target) = artist {
        actions.push(Action::Open(target));
    }

    // Saving the album saves every song on it.
    let saved = cx.saved.albums.contains(&page.browse_id);
    let button = (text_x + SAVE_HEAD_R, cover.y + COVER_BIG - SAVE_HEAD_R - 2.0);
    let pointer_in_view = cx.pointer_in(view);
    if save_button(
        cx,
        hash(&page.browse_id) ^ SALT_HEAD,
        button,
        SAVE_HEAD_R,
        saved,
        pointer_in_view,
        1.0,
        1.0,
    ) {
        actions.push(Action::SaveAlbum(page.clone()));
    }
    cx.text(
        Rect::new(button.0 + SAVE_HEAD_R + 12.0, button.1 - 10.0, 160.0, 20.0),
        if saved { "Saved" } else { "Save" },
        13.0,
        if saved { theme::FG } else { theme::MUTED },
        Align::Left,
    );

    if !tracks.is_empty() {
        track_rows(
            cx,
            track,
            &tracks,
            view,
            body,
            body.y + rows_top_off,
            8000,
            f32::INFINITY,
            false,
            Action::PlayDetail,
            0,
            actions,
        );
    }
    cx.ui.scroll_end();
}

/// The back button: a circle the height of the search field, in the same glass, with an arrow. It
/// sits on the field's left and slides in as the field gives up that room (see `build`).
///
/// `reveal` runs 0 to 1 on the button's own spring. It drives the whole appearance, not just whether
/// the button is drawn: the button fades in, grows from a little smaller, and slides in from the
/// left, and it reverses the same way on the way out. The field's shave (see `build`) moves with
/// the same spring, so the two read as one motion.
fn draw_back(cx: &mut Cx, rect: Rect, reveal: f32, actions: &mut Vec<Action>) {
    let t = reveal.clamp(0.0, 1.0);
    if t <= 0.001 || rect.y + rect.h <= 0.0 {
        return;
    }
    // Only a button that is mostly in place takes the pointer; a fading one is not clickable.
    let live = t > 0.5;
    let hover = live && cx.over_clickable(rect);
    let lift = cx.anims.toggle("back_hover", 0, hover, rate::QUICK);

    // The drawn rect: grows from 70% to full size around its centre, and comes in from the left.
    let grow = 0.7 + 0.3 * t;
    let slide = -(1.0 - t) * rect.h * 0.35;
    let w = rect.w * grow;
    let h = rect.h * grow;
    let cxm = rect.x + rect.w * 0.5 + slide;
    let cym = rect.y + rect.h * 0.5;
    let drawn = Rect::new(cxm - w * 0.5, cym - h * 0.5, w, h);

    // The glass and the tint are the search field's, so the two read as one bar. Everything that
    // carries colour is faded by `t` so the button arrives and leaves without a visible edge.
    cx.draw()
        .backdrop_rounded(drawn, fade(Color::rgb(0xff, 0xff, 0xff), t), drawn.h * 0.5);
    cx.draw().round_rect(
        drawn,
        drawn.h * 0.5,
        fade(
            mix(theme::PANEL_SELECT, theme::PANEL_PRESS, lift),
            t * (0.5 + 0.1 * lift),
        ),
    );
    cx.icon(
        Icon::ArrowLeft,
        (cxm, cym),
        19.0 * grow,
        fade(mix(theme::MUTED, theme::FG, lift), t),
    );
    if live && cx.clicked(rect) {
        actions.push(Action::PageBack);
    }
}

/// Stand-in album cards for `V1_SHOT_PAGE`. Invented names.
fn demo_cards(kind: &str, count: usize) -> Vec<AlbumCard> {
    (0..count)
        .map(|index| AlbumCard {
            browse_id: format!("MPREb_demo{kind}{index}"),
            title: format!("Demo {kind} {}", index + 1),
            meta: format!("{kind}  ·  {}", 1990 + index),
            thumbnail: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Player bar
// ---------------------------------------------------------------------------------------------

/// A bare icon button, matching the volume icon: no container, no size change. Hover brightens the
/// glyph from `MUTED` to `FG`; a disabled button dims toward the panel.
#[allow(clippy::too_many_arguments)]
fn icon_button(
    cx: &mut Cx,
    key: u32,
    center: P,
    radius: f32,
    enabled: bool,
    icons: &[Icon],
    current: usize,
    size: f32,
) -> bool {
    let hit = Rect::new(center.0 - radius, center.1 - radius, radius * 2.0, radius * 2.0);
    let hover = enabled && cx.over_clickable(hit);
    let hv = cx.anims.toggle("btn_hv", key, hover, rate::SNAP);
    let live = cx.anims.toggle("btn_on", key, enabled, rate::QUICK);
    cx.icon_state(
        icons,
        current,
        center,
        size,
        mix(theme::PANEL, mix(theme::MUTED, theme::FG, hv), 0.35 + 0.65 * live),
    );
    enabled && cx.clicked(hit)
}

/// How long the reader must stay on the search results before going to a page remembers them.
const RESULTS_STAY: std::time::Duration = std::time::Duration::from_secs(2);

/// The search field's spring. It travels a long way in from above the window, so it sits just
/// under the volume slider's stiffness rather than matching it; closing is stiffer again, so the
/// field is gone as soon as the pointer leaves the zone.
const SEARCH_OPEN_RATE: f32 = 1400.0;
const SEARCH_CLOSE_RATE: f32 = 2100.0;

/// The volume slider's spring. Opening is a little slower than closing, so a pointer that leaves
/// is answered at once and the slider never lingers half-folded. Both are far stiffer than the
/// `PANEL` preset: the whole unfold is over in about a tenth of a second, so reaching for the icon
/// gets a flick rather than a drift, and it is still a spring, so a pointer that reverses midway
/// turns around instead of restarting.
const VOLUME_OPEN_RATE: f32 = 1600.0;
const VOLUME_CLOSE_RATE: f32 = 2400.0;

/// True when the playing track has words to show: synced or plain lyrics that have arrived for it.
/// Instrumentals, misses, and lookups still in flight do not count.
fn has_lyrics(state: &State, track: Option<&TrackView>) -> bool {
    let Some(track) = track else { return false };
    state
        .lyrics
        .as_ref()
        .filter(|view| view.video_id == track.video_id)
        .and_then(|view| view.lookup.as_ref())
        .is_some_and(|lookup| match lookup {
            Lookup::Found(lyrics) => matches!(lyrics.body, Body::Synced(_) | Body::Plain(_)),
            _ => false,
        })
}

/// The spinner that replaces the play glyph while a song loads: its radius, stroke width, arc
/// length, and turn rate in radians per second.
const SPINNER_RADIUS: f32 = 11.0;
const SPINNER_THICKNESS: f32 = 2.5;
const SPINNER_SWEEP: f32 = std::f32::consts::PI * 1.4;
const SPINNER_TURN: f32 = std::f32::consts::TAU * 1.1;

/// The loading spinner: an arc of [`SPINNER_SWEEP`] turning at `cx.spin`, `radius` out from
/// `center`. The bar's play disc and each row's cover both draw it.
fn draw_spinner(cx: &mut Cx, center: (f32, f32), radius: f32, color: Color) {
    let rect = Rect::new(center.0 - radius, center.1 - radius, radius * 2.0, radius * 2.0);
    let angle = cx.spin;
    cx.draw().ring_arc(rect, radius, SPINNER_THICKNESS, angle, SPINNER_SWEEP, color);
}

fn draw_bar(
    cx: &mut Cx,
    state: &mut State,
    track: Option<&TrackView>,
    bar: Rect,
    actions: &mut Vec<Action>,
) {
    cx.draw().rect(bar, theme::PANEL);
    let width = bar.w;
    let has = track.is_some();

    let left_w = (width * 0.26).clamp(220.0, 340.0);
    let right_w = (width * 0.2).clamp(150.0, 230.0);
    let mid_l = 24.0 + left_w + 16.0;
    let mid_r = width - 24.0 - right_w - 16.0;
    let mid_w = (mid_r - mid_l).max(200.0);
    let mid_x = (mid_l + mid_r) * 0.5;

    // ---- Track: cover, title, artist. Keep the metadata steady when the track changes.
    let tile_size = 56.0;
    // The tile sits 8 px in from the window edge; the text follows 10 px after it.
    let tile_left = 8.0;
    let text_gap = 10.0;
    let tile_c = (tile_left + 28.0, bar.y + 36.0);
    cx.cover(
        track.and_then(|track| track.thumbnail.as_deref()),
        Rect::new(tile_c.0 - tile_size * 0.5, tile_c.1 - tile_size * 0.5, tile_size, tile_size),
        8.0,
        1.0,
    );

    let text_x = tile_left + 56.0 + text_gap;
    let text_w = left_w - 56.0 - 16.0;
    // The bar shows only what is playing. App status ("Loading…", "Searching…") never goes here.
    let (title, sub, title_color) = match track {
        Some(track) => (track.title.as_str(), track.artist.as_str(), theme::FG),
        None => ("Nothing playing", "", theme::MUTED),
    };
    cx.clip(Rect::new(tile_left + 56.0 + text_gap, bar.y, text_w + 4.0, bar.h));
    // The title names the track's album: a press on it opens the album.
    let album_link = track
        .and_then(|track| track.hit.as_ref())
        .and_then(|hit| hit.album_id.clone())
        .map(Detail::Album);
    if let Some(target) = title_link(
        cx,
        Rect::new(text_x, bar.y + 15.0, text_w, 22.0),
        title,
        14.5,
        title_color,
        album_link,
    ) {
        actions.push(Action::Open(target));
    }
    // The artist names the track's page: a press on it opens the artist.
    let artist_link = track.and_then(|track| track.artist_id.clone()).map(Detail::Artist);
    if let Some(target) = row_text(
        cx,
        "bar_artist",
        0,
        Rect::new(text_x, bar.y + 35.0, text_w, 18.0),
        sub,
        12.5,
        1.0,
        artist_link,
    ) {
        actions.push(Action::Open(target));
    }
    cx.unclip();

    // A right press on the track in the bar opens its menu, as a right press on a row does.
    if let Some(hit) = track.and_then(|track| track.hit.as_ref()) {
        if cx.right_clicked(Rect::new(0.0, bar.y, left_w + 24.0, bar.h)) {
            actions.push(Action::OpenMenu(TrackMenu { hit: hit.clone(), at: cx.pointer.pos }));
        }
    }

    // ---- Transport.
    let paused = track.is_none_or(|track| track.paused);
    // While a song loads the play button is held: it does not respond, and shows a spinner.
    let loading = track.is_some_and(|track| track.loading);
    let playable = has && !loading;
    let y = bar.y + 28.0;
    // Every transport glyph shares one device-pixel row: the play disc's centre.
    let row = cx.snap(y - 20.0) + 20.0;
    // Back is live while a track is loaded or there is something in the history to go back to.
    let has_back = has || !state.playlist.history.is_empty();
    if icon_button(cx, 1, (mid_x - 52.0, row), 18.0, has_back, &[Icon::SkipBack], 0, 26.0) {
        actions.push(Action::Back);
    }
    // Next is only live when something is queued.
    let has_next = !state.playlist.queue.is_empty();
    if icon_button(cx, 2, (mid_x + 52.0, row), 18.0, has_next, &[Icon::SkipForward], 0, 26.0) {
        actions.push(Action::Next);
    }

    // The disc stays centred on its fixed hit target and is never resized; hover only tints it.
    let c = (cx.snap(mid_x - 20.0) + 20.0, cx.snap(y - 20.0) + 20.0);
    let play_rect = Rect::new(c.0 - 20.0, c.1 - 20.0, 40.0, 40.0);
    let hover = playable && cx.over_clickable(play_rect);
    let hover_t = cx.anims.spring("play_hover", 0, if hover { 1.0 } else { 0.0 }, rate::SNAP).clamp(0.0, 1.0);
    let live = cx.anims.toggle("play_on", 0, has, rate::QUICK);
    let visual = Rect::new(c.0 - 20.0, c.1 - 20.0, 40.0, 40.0);
    // Opaque throughout: the disc's brightness and the glyph's colour carry the state, never alpha.
    // Hover changes the colour in both states, in opposite directions so each stays visible: an idle
    // disc (dark panel tone) lifts toward white, a live one (already white) eases toward grey.
    let disc = mix(theme::PANEL, theme::FG, 0.22 + 0.78 * live);
    let idle_hover = mix(disc, theme::FG, hover_t * 0.24);
    let live_hover = mix(disc, theme::MUTED, hover_t * 0.2);
    cx.draw().round_rect(visual, 20.0, mix(idle_hover, live_hover, live));
    // Play and pause swap by scale: the outgoing glyph shrinks to nothing while the incoming one
    // grows from nothing with a slight spring overshoot. Both are opaque, so they never blend.
    let showing_pause = playable && !paused;
    let ink = mix(theme::MUTED, theme::ON_ACCENT, live);
    // Glyph and spinner swap with no motion in either direction: the spinner appears and goes at
    // once. Play and pause still swap by scale, but only when neither is a spinner.
    let glyph_targets = [if showing_pause { 0.0 } else { 1.0 }, if showing_pause { 1.0 } else { 0.0 }];
    if loading {
        // The springs are held at the state the song will start in, so the pause glyph is already
        // in place when the spinner gives way to it.
        cx.anims.snap("play_glyph", 0, 0.0);
        cx.anims.snap("play_glyph", 1, 1.0);
        draw_spinner(cx, c, SPINNER_RADIUS, ink);
    } else {
        if state.glyph_loading {
            // Leaving the spinner: land on the target at once instead of popping in.
            cx.anims.snap("play_glyph", 0, glyph_targets[0]);
            cx.anims.snap("play_glyph", 1, glyph_targets[1]);
        }
        let play_t = cx.anims.pop("play_glyph", 0, !showing_pause).max(0.0);
        let pause_t = cx.anims.pop("play_glyph", 1, showing_pause).max(0.0);
        if play_t > 0.01 {
            cx.icon_scaled(Icon::Play, c, 22.0, play_t, ink);
        }
        if pause_t > 0.01 {
            cx.icon_scaled(Icon::Pause, c, 22.0, pause_t, ink);
        }
    }
    state.glyph_loading = loading;
    if playable && cx.clicked(play_rect) {
        actions.push(Action::Toggle);
    }

    // ---- Seek.
    let seek_w = mid_w.min(560.0);
    let seek_x = mid_x - seek_w * 0.5;
    let seek_y = bar.y + 60.0;
    let (position, duration) = track.map_or((0.0, 0.0), |track| (track.position, track.duration));
    let times = if has { theme::MUTED } else { theme::FAINT };
    cx.text(
        Rect::new(seek_x, seek_y - 10.0, 40.0, 20.0),
        &clock(position),
        12.0,
        times,
        Align::Right,
    );
    cx.text(
        Rect::new(seek_x + seek_w - 40.0, seek_y - 10.0, 40.0, 20.0),
        &if duration > 0.0 { clock(duration) } else { "0:00".to_owned() },
        12.0,
        times,
        Align::Left,
    );
    let mut seek = state.seek;
    let seekable = has && duration > 0.0;
    if cx.slider(
        SliderId::Seek,
        Rect::new(seek_x + 52.0, seek_y - 8.0, seek_w - 104.0, 16.0),
        &mut seek,
        seekable,
        1.0,
    ) {
        state.seek = seek;
        actions.push(Action::Seek(seek));
    }

    // ---- Volume: just the icon until the pointer comes near, then the slider unfolds beside it.
    let icon_c = (width - 24.0 - 20.0, bar.y + 36.0);
    let icon_hit = Rect::new(icon_c.0 - 20.0, icon_c.1 - 20.0, 40.0, 40.0);
    let full_w = 112.0;
    let slider_right = icon_hit.x - 4.0;
    let region_x = slider_right - full_w - 14.0;
    let region = Rect::new(region_x, icon_hit.y - 4.0, icon_hit.x + icon_hit.w - region_x, 48.0);
    let near = cx.pointer_in(if state.volume_open { region } else { icon_hit });
    let wanted = near || cx.drag == Some(SliderId::Volume);
    state.volume_open = wanted;
    let stiffness = if wanted { VOLUME_OPEN_RATE } else { VOLUME_CLOSE_RATE };
    let open = cx
        .anims
        .spring("volume_open", 0, if wanted { 1.0 } else { 0.0 }, stiffness)
        .clamp(0.0, 1.0);
    // Everything about the slider is a function of that one value, so it cannot drift out of step:
    // the width is the spring itself, the opacity ramps over the first half of the travel (so it
    // is never a visible sliver, going out or coming in), and its far end starts at the icon's
    // centre, behind it, so it slides out from there.
    let reveal = (open * 2.0).min(1.0);
    let alpha = reveal * reveal * (3.0 - 2.0 * reveal);
    let slide = (1.0 - open) * (icon_c.0 - slider_right);

    let level = state.volume;
    let volume_state = if level <= 0.001 {
        0
    } else if level < 0.5 {
        1
    } else {
        2
    };
    // Muted from the icon: the track still unfolds with the pointer, so the state stays readable,
    // but it is dim and inert. The icon is the only way back, because a drag on a muted slider
    // would either fight the mute or land on a level the icon is no longer showing.
    let muted = state.volume_before_mute.is_some();
    // The extra dimming rides the same spring as the slider's own disabled tone, so going muted is
    // one fade rather than a snap.
    let dim = cx.anims.toggle("volume_muted", 0, muted, rate::QUICK);

    // ---- Lyrics: a microphone just left of the volume icon. It stays put: while the volume slider is
    // showing, it fades out underneath it and cannot be clicked, so nothing else moves. It is lit
    // while the lyric menu is open.
    let mic_c = (icon_c.0 - 44.0, icon_c.1);
    let mic_hit = Rect::new(mic_c.0 - 20.0, mic_c.1 - 20.0, 40.0, 40.0);
    // Greyed out and unclickable unless the current song has lyrics to show. An open page can always
    // be closed.
    let mic_enabled = state.lyrics_open || has_lyrics(state, track);
    // Hidden, not just dimmed, whenever the volume slider has any presence.
    let mic_active = mic_enabled && alpha <= 0.002;
    let mic_hover = mic_active && cx.over_clickable(mic_hit);
    let mic_hv = cx.anims.toggle("mic_hv", 0, mic_hover, rate::SNAP);
    let lit = cx.anims.toggle("mic_lit", 0, state.lyrics_open, rate::QUICK);
    let mic_colour = if mic_enabled {
        mix(theme::MUTED, theme::FG, mic_hv.max(lit))
    } else {
        theme::FAINT
    };
    // Fade toward the panel as the slider comes in, so the microphone is gone by the time the
    // slider is fully open.
    cx.icon(Icon::Mic, mic_c, 22.0, mix(theme::PANEL, mic_colour, 1.0 - alpha));
    if mic_active && cx.clicked(mic_hit) {
        actions.push(Action::ToggleLyrics);
    }

    // ---- Save: the current song, just left of the lyrics button. It fades with that button under the
    // volume slider, and cannot be pressed while the slider is showing.
    if let Some(hit) = track.and_then(|track| track.hit.as_ref()) {
        let saved = cx.saved.songs.contains(&hit.video_id);
        let at = (mic_c.0 - 44.0, icon_c.1);
        if save_button(
            cx,
            hash(&hit.video_id) ^ SALT_BAR,
            at,
            SAVE_BAR_R,
            saved,
            alpha <= 0.002,
            1.0,
            1.0 - alpha,
        ) {
            actions.push(Action::SaveSong(hit.clone()));
        }
    }

    // Bare glyph: no wash and no size change; hovering only warms its colour.
    let hover = cx.over_clickable(icon_hit);
    let hv = cx.anims.toggle("volume_hv", 0, hover, rate::SNAP);
    cx.icon_state(
        &[Icon::VolumeX, Icon::Volume1, Icon::Volume2],
        volume_state,
        icon_c,
        22.0,
        mix(theme::MUTED, theme::FG, hv),
    );
    if cx.clicked(icon_hit) {
        actions.push(Action::Mute);
    }
    if alpha > 0.002 {
        let right = slider_right + slide;
        let w = full_w * open;
        let mut volume = state.volume;
        // Clipped at the icon's hover circle, so the track is never drawn over the icon: it
        // emerges from behind the circle's left edge.
        let edge = icon_c.0 - 20.0;
        cx.clip(Rect::new(0.0, icon_hit.y - 4.0, edge, 48.0));
        let changed = cx.slider(
            SliderId::Volume,
            Rect::new(right - w, icon_c.1 - 8.0, w, 16.0),
            &mut volume,
            !muted,
            alpha * (1.0 - 0.45 * dim),
        );
        cx.unclip();
        if changed {
            state.set_volume(volume);
            state.volume_before_mute = None;
        }
    }
}
