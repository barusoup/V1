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
use crate::background::Background;
use crate::gpu::{CoverImage, Gpu};
use crate::icons::{self, Icon};
use crate::library::{Effects, Library};
use crate::playlist::Playlist;
use crate::prefs::{Prefs, ThemeRec, VisualizerColorRec, DEFAULT_VOLUME, EQ_RANGE_DB};
use crate::spectrum::{self, Spectrum};
use crate::theme::{self, ease_out, fade, mix};
pub use crate::widgets::Pointer;
use crate::widgets::{hash, Cx, QueueDrag, SliderId};

type P = (f32, f32);

/// The destinations in the sidebar. Adding a tab is: a variant here, an icon, and a `match` arm in
/// [`page`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Library,
    Queue,
    /// Reached from the button at the bottom of the sidebar, not from the tab list.
    Settings,
}

impl Tab {
    const ALL: [Tab; 2] = [Tab::Library, Tab::Queue];

    fn label(self) -> &'static str {
        match self {
            Tab::Library => "Collection",
            Tab::Queue => "Queue",
            Tab::Settings => "Settings",
        }
    }

    fn icon(self) -> Icon {
        match self {
            Tab::Library => Icon::Book,
            Tab::Queue => Icon::Menu,
            Tab::Settings => Icon::Settings,
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
    const ALL: [CollectionTab; 3] = [
        CollectionTab::Songs,
        CollectionTab::Albums,
        CollectionTab::Artists,
    ];

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

/// How many times a song's missing metadata (its album and its link) is looked up. A lookup that
/// finds nothing is tried again later, a little less eagerly each time.
const META_TRIES: u32 = 4;
const META_FIRST_RETRY: Duration = Duration::from_secs(30);
const META_MAX_RETRY: Duration = Duration::from_secs(600);

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
    /// The last active lyric index, so a new lyric can re-sync after a short scroll pause.
    last_active: Option<usize>,
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
    /// The equalizer badge should be showing. It is latched: a level crossing above the on threshold
    /// sets it, and only a sustained quiet spell clears it, so a volume change or a quiet passage
    /// cannot make the badge flicker.
    audible: bool,
    /// When the output first fell below the off threshold while `audible` was set.
    quiet_since: Option<Instant>,
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

/// The equalizer badge over a playing song's cover: it comes on once the output rises above the
/// first level, and only goes away after the output has stayed below the second for a while. The
/// gap and the hold are what keep a volume change, or a quiet passage in the music, from blinking
/// the badge in and out.
const BADGE_ON_LEVEL: f32 = 0.02;
const BADGE_OFF_LEVEL: f32 = 0.004;
const BADGE_QUIET_HOLD: Duration = Duration::from_millis(800);

/// Move the equalizer badge's latch for one frame. Returns whether it is showing.
///
/// It comes on the first time the output is clearly audible and goes off only after a sustained
/// quiet spell (below the lower threshold for the hold), so the badge cannot be blinked by the
/// meter's release, a volume drag, or the gaps between songs in an album.
fn advance_badge(
    audible: bool,
    quiet_since: &mut Option<Instant>,
    levels: [f32; 3],
    now: Instant,
) -> bool {
    if levels.iter().any(|level| *level > BADGE_ON_LEVEL) {
        *quiet_since = None;
        return true;
    }
    if !audible {
        return false;
    }
    let quiet = *quiet_since.get_or_insert(now);
    if now.duration_since(quiet) >= BADGE_QUIET_HOLD
        && levels.iter().all(|level| *level < BADGE_OFF_LEVEL)
    {
        *quiet_since = None;
        return false;
    }
    true
}

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
    MoveQueued {
        from: usize,
        to: usize,
    },
    /// Take a song out of the queue.
    RemoveQueued(usize),
    /// Add a song to the end of the queue.
    Enqueue(yt_client::SongHit),
    /// Open the menu for a track, at the pointer.
    OpenMenu(TrackMenu),
    /// Switch to a theme by name, or to Default with `None`.
    SetTheme(Option<String>),
    /// Pin or unpin one of the listener's themes.
    FavoriteTheme(String),
    /// Delete one of the listener's themes.
    DeleteTheme(String),
    /// Load a saved theme into the theme builder for editing.
    EditTheme(String),
    /// Save or update a theme from the colours chosen on the Settings page.
    CreateTheme,
    /// Open the picture picker for the global background.
    PickBackground,
    RemoveBackground,
    /// Toggle whether the global background overrides per-theme backgrounds.
    ToggleGlobalBackground(bool),
    /// Open the picture picker for the theme currently in the builder.
    PickThemeBackground,
    /// How soft the background is, 0 to 1.
    BlurStrength(f32),
    ThemeBlurStrength(f32),
    BackgroundTransparency(f32),
    ThemeBackgroundTransparency(f32),
    /// Open or close the final confirmation about deleting downloads.
    ConfirmDialog(Option<ConfirmDialog>),
    /// Turn local downloads on or off. Off deletes what is already here.
    SetDownloads(bool),
    /// The equalizer's ten band gains, in dB.
    SetEq([f32; 10]),
    ResetEq,
}

/// Something the window itself has to do, asked for by the caption buttons or by a press on the
/// top strip (the window has no OS title bar, so the app does these). The host takes them after
/// each frame, see `main.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRequest {
    Minimize,
    ToggleFullscreen,
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
    /// The preferences behind the volume, and whatever else outlives a run.
    prefs: Prefs,
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
    /// Current native fullscreen state, supplied by the host before building a frame.
    pub fullscreen: bool,

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
    background_prep_pending: std::sync::Arc<(
        std::sync::Mutex<Option<BackgroundPrepJob>>,
        std::sync::Condvar,
    )>,
    background_prep_rx: Receiver<BackgroundPrepResult>,
    background_prep_stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    background_prep_id: u64,
    background_editor_generation: u64,
    background_prep_busy: bool,
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
    /// When the reader last scrolled the page, so its scrollbar can stay for a moment and then fade
    /// (see [`build`]). `None` until the first scroll, and again once the bar is fully away.
    bar_at: Option<Instant>,
    /// The pointer is on the page's right edge, where the scrollbar floats.
    bar_hover: bool,
    /// The pointer holds the scrollbar, so it stays up for as long as the hold lasts.
    bar_grab: bool,
    volume_open: bool,
    /// The lyric page is open. It is a page of its own: it takes the place of the tab's content.
    lyrics_open: bool,
    theater: bool,
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
    /// Songs whose metadata lookup has been sent and not answered yet.
    meta_requested: HashSet<String>,
    /// Metadata lookups that found nothing: how many times, and when to try again.
    meta_failures: HashMap<String, (u32, Instant)>,
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
    /// The themes the listener made, mirroring what is saved.
    themes: Vec<ThemeRec>,
    data_dir: std::path::PathBuf,
    /// What the Settings page is holding that does not outlive the run.
    settings: Settings,
    /// The custom background, decoded and softened off the UI thread.
    background: Background,
    /// Error from stripping audio / preparing a chosen background video.
    background_prep_error: Option<String>,
    /// The equalizer's ten band gains, in dB.
    eq: [f32; 10],
}

/// A player that finished opening on a loading thread, with the track it was opened for.
struct Opened {
    ready: Ready,
    player: Result<dream_media::Player, String>,
}

impl Drop for State {
    fn drop(&mut self) {
        let (slot, ready) = &*self.background_prep_pending;
        let mut pending = slot.lock().unwrap_or_else(|error| error.into_inner());
        self.background_prep_stop
            .store(true, std::sync::atomic::Ordering::Release);
        *pending = None;
        ready.notify_one();
    }
}

#[derive(Clone)]
enum BackgroundPrepTarget {
    Global,
    Theme(Option<String>, u64),
}

fn background_theme_target_is_current(
    creator_open: bool,
    editing_theme: &Option<String>,
    editor_generation: u64,
    target_theme: &Option<String>,
    target_generation: u64,
) -> bool {
    creator_open && editing_theme == target_theme && editor_generation == target_generation
}

fn background_prep_failure_updates_global_error(target: &BackgroundPrepTarget) -> bool {
    matches!(target, BackgroundPrepTarget::Global)
}

struct BackgroundPrepJob {
    id: u64,
    source: std::path::PathBuf,
    data_dir: std::path::PathBuf,
    target: BackgroundPrepTarget,
}

struct BackgroundPrepResult {
    id: u64,
    target: BackgroundPrepTarget,
    result: Result<std::path::PathBuf, String>,
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
    /// Whether the equalizer badge over this song's cover is showing. Latched with hysteresis, so
    /// a quiet moment (or a volume change) does not flicker it in and out (see [`Now::audible`]).
    audible: bool,
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
    /// Each artist credit and the page it links to, if any.
    artists: Vec<(&'a str, Option<&'a str>)>,
    album_id: Option<&'a str>,
}

/// Which category tab is open on the Settings page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SettingsTab {
    #[default]
    Appearance,
    Audio,
    General,
}

impl SettingsTab {
    const ALL: [SettingsTab; 3] = [
        SettingsTab::Appearance,
        SettingsTab::Audio,
        SettingsTab::General,
    ];

    fn index(self) -> usize {
        match self {
            SettingsTab::Appearance => 0,
            SettingsTab::Audio => 1,
            SettingsTab::General => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            SettingsTab::Appearance => "Appearance",
            SettingsTab::Audio => "Audio",
            SettingsTab::General => "General",
        }
    }

    fn icon(self) -> Icon {
        match self {
            SettingsTab::Appearance => Icon::Image,
            SettingsTab::Audio => Icon::Volume2,
            SettingsTab::General => Icon::Settings,
        }
    }
}

/// Which colour slot in the theme builder the in-app picker is currently editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorSlot {
    Primary,
    Secondary,
    Accent,
    LyricsFill,
    LyricsOutline,
    Visualizer,
}

impl ColorSlot {
    fn label(self) -> &'static str {
        match self {
            ColorSlot::Primary => "Background",
            ColorSlot::Secondary => "Text",
            ColorSlot::Accent => "Accent",
            ColorSlot::LyricsFill => "Lyrics Text",
            ColorSlot::LyricsOutline => "Lyrics Outline",
            ColorSlot::Visualizer => "Visualizer",
        }
    }
}

/// Live state for the floating in-app colour picker popup.
struct ColorPickerState {
    slot: ColorSlot,
    anchor: Rect,
    hsv: (f32, f32, f32),
    hex: String,
    last_color: Color,
    /// Prevents the click that opened the popup from immediately closing it on the same frame.
    just_opened: bool,
}

/// The Settings page's working state: the theme being made, and whether the download confirmation
/// is up. Saved choices live in [`Prefs`]; this is only what has not been committed yet.
struct Settings {
    /// Active category tab on the Settings page.
    tab: SettingsTab,
    /// Filter query for the theme manager.
    theme_query: String,
    /// Whether the full theme list is expanded when there are many themes.
    show_all_themes: bool,
    /// Whether the inline "New theme" / "Edit theme" builder is expanded.
    creator_open: bool,
    /// Name of the saved theme currently being edited, or `None` when creating a new theme.
    editing_theme: Option<String>,
    /// The name for the theme being made.
    name: String,
    primary: Color,
    secondary: Color,
    accent: Color,
    lyrics_fill: Color,
    lyrics_outline: Color,
    visualizer_mode: VisualizerColorRec,
    visualizer_color: Color,
    theme_background: Option<String>,
    theme_background_blur: f32,
    theme_background_transparency: f32,
    /// Floating in-app colour picker popup, when open.
    color_picker: Option<ColorPickerState>,
    /// Which confirmation dialog is showing, if any.
    confirm_dialog: Option<ConfirmDialog>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ConfirmDialog {
    Downloads,
    DeleteTheme(String),
    RemoveBackground,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            tab: SettingsTab::Appearance,
            theme_query: String::new(),
            show_all_themes: false,
            creator_open: false,
            editing_theme: None,
            name: String::new(),
            primary: Color::rgb(0x00, 0x00, 0x00),
            secondary: Color::rgb(0xec, 0xec, 0xec),
            accent: Color::rgb(0xec, 0xec, 0xec),
            lyrics_fill: Color::rgb(0xec, 0xec, 0xec),
            lyrics_outline: Color::rgb(0x00, 0x00, 0x00),
            visualizer_mode: VisualizerColorRec::AlbumArt,
            visualizer_color: Color::rgb(0xec, 0xec, 0xec),
            theme_background: None,
            theme_background_blur: crate::prefs::DEFAULT_BLUR,
            theme_background_transparency: crate::prefs::DEFAULT_TRANSPARENCY,
            color_picker: None,
            confirm_dialog: None,
        }
    }
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
            artists: hit.artist_links().collect(),
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

/// What a press on a track row should do, given what is already playing or loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackPress {
    /// Nothing is playing this song: start it.
    Start,
    /// This song is already loading: leave that load alone.
    Ignore,
    /// This song is the one playing: pause or resume it.
    Toggle,
}

fn same_track_press(current: Option<&str>, loading: Option<&str>, pressed: &str) -> TrackPress {
    if loading == Some(pressed) {
        TrackPress::Ignore
    } else if loading.is_some() {
        // The song that was playing is already fading out for another load; a press on it starts
        // it again, which cancels that load, rather than pausing a song on its way out.
        TrackPress::Start
    } else if current == Some(pressed) {
        TrackPress::Toggle
    } else {
        TrackPress::Start
    }
}

impl State {
    /// A state with an empty collection. The host opens the real one on another thread, while the
    /// window and the GPU come up, and hands it over with [`State::attach_library`] before the
    /// first frame.
    pub fn new(data_dir: std::path::PathBuf) -> State {
        let library = Library::in_memory();
        let prefs = Prefs::open(&data_dir);
        let volume = prefs.volume();
        // The saved theme goes on before the first frame, so the window never flashes the default.
        let themes = prefs.themes().to_vec();
        theme::set_palette(theme_palette(prefs.active_theme(), &themes));
        let background = Background::new(
            effective_background_path(&prefs, &themes),
            prefs.background_blur(),
        );
        let eq = prefs.eq();
        let backend = Backend::start(data_dir.clone());
        let (opened_tx, opened_rx) = mpsc::channel();
        let background_prep_pending = std::sync::Arc::new((
            std::sync::Mutex::new(None::<BackgroundPrepJob>),
            std::sync::Condvar::new(),
        ));
        let worker_pending = std::sync::Arc::clone(&background_prep_pending);
        let background_prep_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_stop = std::sync::Arc::clone(&background_prep_stop);
        let (background_prep_results, background_prep_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("v1-background-prep".to_owned())
            .spawn(move || loop {
                let job = {
                    let (slot, ready) = &*worker_pending;
                    let mut slot = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    while slot.is_none() && !worker_stop.load(std::sync::atomic::Ordering::Acquire)
                    {
                        slot = ready
                            .wait(slot)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    if worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                        return;
                    }
                    slot.take().expect("job became available")
                };
                let result =
                    crate::background::prepare_background_media(&job.source, &job.data_dir);
                if background_prep_results
                    .send(BackgroundPrepResult {
                        id: job.id,
                        target: job.target,
                        result,
                    })
                    .is_err()
                {
                    break;
                }
            })
            .expect("background preparation worker");
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
            volume,
            prefs,
            submit: false,
            focus_search: false,
            toggle_request: false,
            sidebar_open: true,
            hand: false,
            gpu: Gpu::default(),
            window_requests: Vec::new(),
            fullscreen: false,
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
            background_prep_pending,
            background_prep_rx,
            background_prep_stop,
            background_prep_id: 0,
            background_editor_generation: 0,
            background_prep_busy: false,
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
            bar_at: None,
            bar_hover: false,
            bar_grab: false,
            volume_open: false,
            lyrics_open: false,
            theater: false,
            lyrics_follow: LyricsFollow::default(),
            pages: Pages::default(),
            artists: HashMap::new(),
            albums: HashMap::new(),
            volume_before_mute: None,
            cover_inflight: HashSet::new(),
            cover_retry: HashMap::new(),
            last_search: String::new(),
            lengths_at: now,
            lengths: HashMap::new(),
            length_requested: HashSet::new(),
            length_failures: HashMap::new(),
            meta_requested: HashSet::new(),
            meta_failures: HashMap::new(),
            preloaded: HashSet::new(),
            tick_at: now,
            fading: false,
            animating: false,
            spectrum: Spectrum::new(),
            spectrum_live: false,
            demo_track: None,
            menu: None,
            themes,
            data_dir,
            settings: Settings::default(),
            background,
            background_prep_error: None,
            eq,
        }
    }

    /// Take up the collection read from disk. Nothing has used the empty one yet: this runs before
    /// the first frame, so the backend's `Ready` (which resumes unfinished downloads) finds it.
    pub fn attach_library(&mut self, library: Library) {
        self.library = library;
        self.library.set_downloading(self.prefs.downloads());
    }

    /// Close the track menu, if one is open. Escape does this, along with the download
    /// confirmation, which is the other thing that sits over the page.
    pub fn close_menu(&mut self) {
        self.menu = None;
        self.settings.confirm_dialog = None;
        self.settings.color_picker = None;
    }

    /// Drain backend events. Called once per frame, before the build.
    pub fn pump(&mut self) {
        while let Ok(done) = self.background_prep_rx.try_recv() {
            if done.id != self.background_prep_id {
                continue;
            }
            self.background_prep_busy = false;
            match done.result {
                Ok(path) => match done.target {
                    BackgroundPrepTarget::Global => {
                        self.background_prep_error = None;
                        self.prefs.set_background(Some(&path));
                        self.prefs.set_background_enabled(true);
                        sync_background(self);
                    }
                    BackgroundPrepTarget::Theme(editing, session) => {
                        if background_theme_target_is_current(
                            self.settings.creator_open,
                            &self.settings.editing_theme,
                            self.background_editor_generation,
                            &editing,
                            session,
                        ) {
                            self.settings.theme_background =
                                Some(path.to_string_lossy().into_owned());
                        }
                    }
                },
                Err(error) => {
                    if background_prep_failure_updates_global_error(&done.target) {
                        self.background_prep_error = Some(error);
                    }
                }
            }
        }
        // A softened background that finished on its thread is handed to the GPU for the next frame.
        if let Some(image) = self.background.poll() {
            self.gpu.incoming_background = Some(image);
        }
        // A picture that could not be read must not leave the previous one on screen under its
        // name: the Settings page is showing why it failed.
        if self.background.error.is_some() && !self.background.loading() {
            self.gpu.clear_background();
        }
        self.background.refresh();
        if self.artists.len() > PAGE_CACHE || self.albums.len() > PAGE_CACHE {
            self.trim_pages();
        }
        if self.preloaded.len() > 512 {
            self.preloaded.clear();
        }
        if self.meta_failures.len() > 512 {
            self.meta_failures.clear();
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
                Event::Cover {
                    url,
                    width,
                    height,
                    rgba,
                } => {
                    self.cover_inflight.remove(&url);
                    self.gpu.queue_cover(CoverImage {
                        url,
                        width,
                        height,
                        rgba,
                    });
                }
                Event::CoverFailed(url) => {
                    // Forgotten as requested, but not for a while: the next frame that wants it
                    // would ask again at once, and a cover that cannot be had would be asked for
                    // every frame. Asking again after the wait is how it recovers from a network
                    // that was down.
                    self.cover_inflight.remove(&url);
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
                        let stored_empty = self
                            .library
                            .album_page(&id)
                            .is_some_and(|p| p.tracks.is_empty());
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
                        self.length_failures
                            .insert(video_id, (tries, Instant::now() + wait));
                    }
                },
                Event::SongMeta { video_id, hit } => {
                    self.meta_requested.remove(&video_id);
                    let found_album = hit.as_ref().is_some_and(|hit| hit.album_id.is_some());
                    if let Some(updated) = hit {
                        self.learn(std::slice::from_ref(&updated));
                    }
                    if found_album {
                        self.meta_failures.remove(&video_id);
                    } else {
                        // Nothing more to find this time: try again later, a little less often.
                        let tries = self.meta_failures.get(&video_id).map_or(0, |(n, _)| *n) + 1;
                        let wait = (META_FIRST_RETRY * 2u32.saturating_pow(tries.min(5)))
                            .min(META_MAX_RETRY);
                        self.meta_failures
                            .insert(video_id, (tries, Instant::now() + wait));
                    }
                }
                Event::Downloaded {
                    video_id,
                    loudness_db,
                    duration_secs,
                } => {
                    self.library
                        .downloaded(&video_id, loudness_db, duration_secs);
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

    fn cancel_background_prep(&mut self) {
        self.background_prep_id = self.background_prep_id.wrapping_add(1);
        self.background_prep_busy = false;
        let (slot, _) = &*self.background_prep_pending;
        *slot.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }

    fn queue_background_prep(&mut self, source: std::path::PathBuf, target: BackgroundPrepTarget) {
        self.background_prep_id = self.background_prep_id.wrapping_add(1);
        self.background_prep_busy = true;
        let job = BackgroundPrepJob {
            id: self.background_prep_id,
            source,
            data_dir: self.data_dir.clone(),
            target,
        };
        let (slot, ready) = &*self.background_prep_pending;
        *slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(job);
        ready.notify_one();
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
        // The lists on screen are rebuilt from their own copies: fill those too, so a link that
        // arrived after the results were drawn becomes live without another search.
        for song in &mut self.results {
            for other in hits {
                crate::library::fill_gaps(song, other);
            }
        }
        if let Some(TopResult::Song(song)) = &mut self.top {
            for other in hits {
                crate::library::fill_gaps(song, other);
            }
        }
        for entry in self.artists.values_mut() {
            if let Loaded::Ready(page) = entry {
                for song in &mut page.songs {
                    for other in hits {
                        crate::library::fill_gaps(song, other);
                    }
                }
            }
        }
        for entry in self.albums.values_mut() {
            if let Loaded::Ready(page) = entry {
                for song in &mut page.tracks {
                    for other in hits {
                        crate::library::fill_gaps(song, other);
                    }
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
                dream_media::Status::Playing
                | dream_media::Status::Paused
                | dream_media::Status::Loading => {}
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
            track.player.set_volume(
                self.volume_before_mute.unwrap_or(self.volume) * track.gain * track.norm,
            );
            track.player.set_muted(self.volume_before_mute.is_some());
            track.player.set_equalizer(self.eq);
            if !track.want_playing && track.gain <= 0.0 && !track.held {
                track.player.pause(now);
                track.held = true;
            }
            track.paused = !track.want_playing;
            // Latch the equalizer badge: on as soon as the output is clearly audible, off only
            // after it has been quiet for a while. A slider drag changes the gain in steps and the
            // meters follow with their own ballistics; a plain threshold would cross both ways
            // between frames and blink the badge.
            let levels = track.player.levels();
            track.audible = advance_badge(track.audible, &mut track.quiet_since, levels, now);
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

    /// True while a play/pause fade is still moving the gain, so the host keeps ticking.
    pub fn fading(&self) -> bool {
        self.fading
    }

    /// What the Windows media controls should show, or `None` when nothing is loaded. Borrows the
    /// current track rather than cloning it: this runs every frame.
    pub fn media_info(&self) -> Option<crate::media::MediaInfo<'_>> {
        if let Some(hit) = self.playlist.loading_song() {
            return Some(crate::media::MediaInfo {
                title: &hit.title,
                artist: hit.artists.first().map_or("", String::as_str),
                album: hit.album.as_deref(),
                position: 0.0,
                duration: self.known_length(hit) as f64,
                playing: false,
            });
        }
        if let Some(track) = &self.now {
            return Some(crate::media::MediaInfo {
                title: &track.title,
                artist: &track.artist,
                album: track.hit.album.as_deref(),
                position: track.position as f64,
                duration: track.duration as f64,
                playing: !track.paused,
            });
        }
        self.demo_track
            .as_ref()
            .map(|track| crate::media::MediaInfo {
                title: &track.title,
                artist: &track.artist,
                album: track.hit.as_ref().and_then(|hit| hit.album.as_deref()),
                position: track.position as f64,
                duration: track.duration as f64,
                playing: !track.paused,
            })
    }

    /// Apply a transport button from the OS (earbuds, media keys, or the Windows flyout).
    /// `immediate` skips the gain fade: a hidden window builds no frames, and a fade part-way
    /// through would leave the output quiet but not paused until the window is shown again.
    pub fn media_command(&mut self, command: crate::media::Command, immediate: bool) {
        use crate::media::Command;
        let transport = matches!(
            command,
            Command::Play | Command::Pause | Command::Stop | Command::Toggle
        );
        match command {
            Command::Play => self.set_playing(true),
            Command::Pause => self.set_playing(false),
            Command::Toggle => self.toggle_playback(),
            Command::Next => self.next(),
            Command::Previous => self.back(),
            Command::Stop => self.set_playing(false),
            Command::Seek(seconds) => {
                if let Some(track) = self.now.as_mut() {
                    let upper = if track.duration > 0.0 {
                        track.duration as f64
                    } else {
                        seconds.max(0.0)
                    };
                    let seconds = seconds.clamp(0.0, upper);
                    track.player.seek((seconds * 1000.0).round() as i64);
                    track.position = seconds as f32;
                }
            }
        }
        if immediate && transport {
            self.settle_transport();
        }
        // The frame that is due should show the result even if no input asked for one.
        self.animating = true;
    }

    /// Make the current play/pause request take effect now, without waiting for fade frames.
    fn settle_transport(&mut self) {
        let now = Instant::now();
        let Some(track) = self.now.as_mut() else {
            return;
        };
        let goal = if track.want_playing { 1.0 } else { 0.0 };
        track.gain = goal;
        track
            .player
            .set_volume(self.volume_before_mute.unwrap_or(self.volume) * goal * track.norm);
        if track.want_playing {
            track.held = false;
        } else if !track.held {
            track.player.pause(now);
            track.held = true;
        }
        track.paused = !track.want_playing;
        self.fading = false;
    }

    /// Start or pause the loaded song, as an explicit Play or Pause asks. Unlike a toggle, a
    /// repeated Play or Pause is a no-op.
    fn set_playing(&mut self, play: bool) {
        // Held while a song loads: the transport is disabled until it starts.
        let loading = self.playlist.is_loading();
        let now = Instant::now();
        if let Some(track) = self.now.as_mut().filter(|_| !loading) {
            if track.want_playing != play {
                track.want_playing = play;
                track.paused = !play;
                if play {
                    track.held = false;
                    track.player.play(now);
                }
            }
        }
    }

    /// Play the first search result automatically; used by the screenshot harness.
    pub fn autoplay_first(&mut self) {
        self.autoplay = true;
    }

    /// True while the UI is waiting on the backend, so the host keeps polling for the reply.
    pub fn busy(&self) -> bool {
        !self.booted || self.searching || self.playlist.is_loading() || self.background_prep_busy
    }

    /// True while something on screen is still moving, so the host keeps presenting frames.
    pub fn animating(&self) -> bool {
        self.animating || self.background.is_animated()
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

    /// When the scrollbar starts to fade, so the host wakes for it even when nothing else is moving.
    /// `None` while none is up, or while the pointer keeps one up.
    pub fn bars_fade_at(&self) -> Option<Instant> {
        if self.bar_hover || self.bar_grab {
            return None;
        }
        self.bar_at
            .map(|at| at + std::time::Duration::from_secs_f32(BAR_LINGER))
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
            (
                "Instant Crush",
                "Daft Punk, Julian Casablancas",
                "Random Access Memories",
                "5:38",
            ),
            ("One More Time", "Daft Punk", "Discovery", "5:20"),
            (
                "Harder, Better, Faster, Stronger",
                "Daft Punk",
                "Discovery",
                "3:45",
            ),
            ("Something About Us", "Daft Punk", "Discovery", "3:52"),
            ("Veridis Quo", "Daft Punk", "Discovery", "5:45"),
            ("Digital Love", "Daft Punk", "Discovery", "5:01"),
            ("Aerodynamic", "Daft Punk", "Discovery", "3:27"),
            (
                "Get Lucky",
                "Daft Punk, Pharrell Williams, Nile Rodgers",
                "Random Access Memories",
                "6:10",
            ),
            ("Da Funk", "Daft Punk", "Homework", "5:28"),
        ];
        self.results = songs
            .iter()
            .enumerate()
            .map(
                |(index, (title, artist, album, duration))| yt_client::SongHit {
                    video_id: format!("demo{index}"),
                    title: (*title).to_owned(),
                    artists: artist.split(", ").map(str::to_owned).collect(),
                    album: Some((*album).to_owned()),
                    duration: Some((*duration).to_owned()),
                    // `V1_SHOT_COVERS` stands in plain tiles for real covers, to check layout offline.
                    thumbnail: std::env::var_os("V1_SHOT_COVERS")
                        .map(|_| format!("demo://{index}")),
                    artist_id: None,
                    album_id: None,
                    artist_ids: Vec::new(),
                },
            )
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
                self.gpu.queue_cover(CoverImage {
                    url: format!("demo://{index}"),
                    width: 64,
                    height: 64,
                    rgba,
                });
            }
        }
        // The search's top result and the artists and albums it matched, invented to match the songs.
        let thumb =
            |index: usize| std::env::var_os("V1_SHOT_COVERS").map(|_| format!("demo://{index}"));
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
                audible: true,
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
                        thumbnail: std::env::var_os("V1_SHOT_COVERS")
                            .map(|_| format!("demo://{}", index % 10)),
                        ..ArtistPage::default()
                    });
                    let _ = self.library.save_album(
                        &AlbumPage {
                            browse_id: format!("MPREb_demo{index}"),
                            title: format!("Demo Album {}", index + 1),
                            artist: format!("Demo Artist {}", index + 1),
                            thumbnail: std::env::var_os("V1_SHOT_COVERS")
                                .map(|_| format!("demo://{}", index % 10)),
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
            Ok("settings") => {
                self.tab = Tab::Settings;
                self.results.clear();
                self.top = None;
                self.result_artists.clear();
                self.result_albums.clear();
                self.results_query.clear();
                self.query.clear();
                self.query_seen.clear();
            }
            Ok("lyrics") => self.set_lyrics_open(true),
            _ => {}
        }
        // `V1_SHOT_SETTINGS=1` seeds the theme manager, a half-made theme and an equalizer curve, to
        // check the Settings page's layout.
        if std::env::var_os("V1_SHOT_SETTINGS").is_some() {
            self.themes = vec![
                ThemeRec {
                    favorite: true,
                    ..ThemeRec::new("Midnight", 0x002b_3350, 0x00e8_edff, 0x008f_b3ff)
                },
                ThemeRec::new("Ember", 0x003a_231c, 0x00ff_e3d1, 0x00ff_8a4c),
            ];
            self.settings.name = "Storm".to_owned();
            self.eq = [4.0, 3.0, 1.5, 0.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5];
        }
        // `V1_SHOT_THEME=Midnight` draws the frame in a saved theme, to check a custom palette.
        if let Ok(name) = std::env::var("V1_SHOT_THEME") {
            theme::set_palette(theme_palette(Some(&name), &self.themes));
        }
        // `V1_SHOT_DIALOG=1` shows the final confirmation about deleting downloads.
        if std::env::var_os("V1_SHOT_DIALOG").is_some() {
            self.settings.confirm_dialog = Some(ConfirmDialog::Downloads);
        }
        // `V1_SHOT_BACKGROUND=<path>` puts a picture behind the demo, decoded on this thread so a
        // still can show it without waiting for the loader.
        if let Ok(path) = std::env::var("V1_SHOT_BACKGROUND") {
            if let Some(image) =
                crate::background::preview(std::path::Path::new(&path), self.background.strength)
            {
                self.gpu.incoming_background = Some(image);
            }
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
                            yt_client::Shelf {
                                title: "Albums".to_owned(),
                                albums: demo_cards("Album", 7),
                            },
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
                audible: false,
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
                audible: track.audible,
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

    /// Play a song picked from the search results. If it is already the song that is playing,
    /// pause or resume it instead of loading it again from the start.
    fn play_index(&mut self, index: usize) {
        if let Some(hit) = self.results.get(index).cloned() {
            if self.clicked_current(&hit.video_id) {
                return;
            }
            let hit = self.playlist.play_one(hit);
            self.send_play(hit);
        }
    }

    /// Play the top result, when it is a song. Only that song plays; the queue is left alone.
    fn play_top(&mut self) {
        let Some(TopResult::Song(hit)) = self.top.clone() else {
            return;
        };
        if self.clicked_current(&hit.video_id) {
            return;
        }
        let hit = self.playlist.play_one(hit);
        self.send_play(hit);
    }

    /// A press on the song that is already up: pause or resume it, rather than starting it over.
    /// A press on the song that is still loading is ignored, as its own load is already under way.
    /// Returns whether the press was taken care of here.
    fn clicked_current(&mut self, video_id: &str) -> bool {
        let current = self.now.as_ref().map(|track| track.video_id.as_str());
        match same_track_press(current, self.playlist.loading_id(), video_id) {
            TrackPress::Toggle => {
                self.toggle_playback();
                true
            }
            TrackPress::Ignore => true,
            TrackPress::Start => false,
        }
    }

    /// Pause the song that is playing, or resume it. A pause is applied by `tick`, once the fade
    /// has taken the sound to silence.
    fn toggle_playback(&mut self) {
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
                duration_secs: local
                    .duration_secs
                    .or((known > 0.0).then_some(known as f64)),
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
                let duration_secs = ready
                    .duration_secs
                    .or((known > 0.0).then_some(known as f64));
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
                    audible: false,
                    quiet_since: None,
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
        self.apply_volume(volume);
        self.prefs.set_volume(self.volume);
    }

    /// Move the output without changing what the next run starts at. Muting goes through here:
    /// the muted level is a moment, not the listener's chosen volume.
    fn apply_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(track) = self.now.as_mut() {
            track.player.set_volume(
                self.volume_before_mute.unwrap_or(self.volume) * track.gain * track.norm,
            );
            track.player.set_muted(self.volume_before_mute.is_some());
        }
    }

    /// Going to another page (a tab, an artist or album, the lyric page) takes the reader off the
    /// search results, and clears the search bar with them. Search-as-you-type is stopped too, so
    /// the cleared text does not search again.
    fn leave_results(&mut self) {
        // Remember the results when the reader stayed on them for a while or clicked into them.
        let showing = !self.results_query.is_empty();
        let stayed = self.results_at.elapsed() >= RESULTS_STAY;
        self.results_return =
            (showing && (self.results_clicked || stayed)).then(|| ResultsReturn {
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
                    if let Some(page) = self
                        .library
                        .album_page(id)
                        .filter(|p| !p.tracks.is_empty())
                        .cloned()
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
        self.artists
            .retain(|id, _| current.as_deref() == Some(id.as_str()));
        let current = match &self.pages.top {
            Some(Detail::Album(id)) => Some(id.clone()),
            _ => None,
        };
        self.albums
            .retain(|id, _| current.as_deref() == Some(id.as_str()));
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
        if self.gpu.covers.contains_key(&url) || self.cover_inflight.contains(&url) {
            return;
        }
        if self
            .cover_retry
            .get(&url)
            .is_some_and(|retry_at| Instant::now() < *retry_at)
        {
            return;
        }
        self.cover_retry.remove(&url);
        self.cover_inflight.insert(url.clone());
        let _ = self.backend.commands.send(Command::Cover(url));
    }

    /// Look up what a row left out (its album, and the album's link), unless a lookup is already on
    /// its way or the last few found nothing. A row is redrawn every frame, so this is the guard
    /// that keeps a song with no findable album from turning into a search every frame.
    fn request_metadata(&mut self, hit: yt_client::SongHit) {
        if hit.album_id.is_some() || self.meta_requested.contains(&hit.video_id) {
            return;
        }
        let now = Instant::now();
        if self
            .meta_failures
            .get(&hit.video_id)
            .is_some_and(|(tries, retry_at)| *tries >= META_TRIES || now < *retry_at)
        {
            return;
        }
        self.meta_requested.insert(hit.video_id.clone());
        let _ = self.backend.commands.send(Command::SongMeta(hit));
    }

    pub fn seek_relative(&mut self, seconds: f64) {
        if let Some(track) = &self.now {
            let position = track.player.position_ms(Instant::now()) as f64 / 1000.0;
            self.media_command(crate::media::Command::Seek(position + seconds), false);
        }
    }

    pub fn adjust_volume(&mut self, delta: f32) {
        self.set_volume(self.volume + delta);
    }

    pub fn toggle_mute(&mut self) {
        self.apply(Action::Mute);
    }
    pub fn toggle_theater(&mut self) {
        if self.lyrics_open {
            self.theater = !self.theater;
        }
    }
    pub fn toggle_lyrics(&mut self) {
        self.apply(Action::ToggleLyrics);
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
                if let Some(hit) = self.playlist.queue.get(index).cloned() {
                    if self.clicked_current(&hit.video_id) {
                        return;
                    }
                }
                if let Some(hit) = self.playlist.skip_to(index) {
                    self.send_play(hit);
                }
            }
            Action::Toggle => self.toggle_playback(),
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
                if let Some(hit) = tracks.get(index) {
                    if self.clicked_current(&hit.video_id) {
                        return;
                    }
                }
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
                let hits: Vec<yt_client::SongHit> = ids
                    .iter()
                    .filter_map(|id| self.library.song_hit(id).cloned())
                    .collect();
                if let Some(hit) = hits.get(index) {
                    if self.clicked_current(&hit.video_id) {
                        return;
                    }
                }
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
            Action::SetTheme(name) => set_theme(self, name),
            Action::FavoriteTheme(name) => {
                if let Some(theme) = self.themes.iter_mut().find(|theme| theme.name == name) {
                    theme.favorite = !theme.favorite;
                    self.prefs.set_themes(&self.themes);
                }
            }
            Action::DeleteTheme(name) => {
                self.settings.confirm_dialog = None;
                self.themes.retain(|theme| theme.name != name);
                self.prefs.set_themes(&self.themes);
                if self.settings.editing_theme.as_deref() == Some(name.as_str()) {
                    self.settings.editing_theme = None;
                    self.settings.creator_open = false;
                    self.settings.color_picker = None;
                }
                if self.prefs.active_theme() == Some(name.as_str()) {
                    set_theme(self, None);
                }
            }
            Action::EditTheme(name) => {
                if self.settings.creator_open
                    && self.settings.editing_theme.as_deref() == Some(name.as_str())
                {
                    self.settings.creator_open = false;
                    self.settings.editing_theme = None;
                    self.settings.name.clear();
                    self.settings.color_picker = None;
                } else if let Some(theme) = self.themes.iter().find(|t| t.name == name).cloned() {
                    self.settings.editing_theme = Some(theme.name.clone());
                    self.settings.name = theme.name;
                    self.settings.primary = rgb_color(theme.primary);
                    self.settings.secondary = rgb_color(theme.secondary);
                    self.settings.accent = rgb_color(theme.accent);
                    self.settings.lyrics_fill =
                        theme.lyrics_fill.map(rgb_color).unwrap_or_else(|| {
                            theme::Palette::from_colors(
                                rgb_color(theme.primary),
                                rgb_color(theme.secondary),
                                rgb_color(theme.accent),
                            )
                            .lyrics_fg
                        });
                    self.settings.lyrics_outline = theme
                        .lyrics_outline
                        .map(rgb_color)
                        .unwrap_or(Color::rgb(0x00, 0x00, 0x00));
                    self.settings.visualizer_mode = theme.visualizer_mode;
                    self.settings.visualizer_color = theme
                        .visualizer_color
                        .map(rgb_color)
                        .unwrap_or(self.settings.accent);
                    self.settings.theme_background = theme.background;
                    self.settings.theme_background_blur = theme.background_blur;
                    self.settings.theme_background_transparency = theme.background_transparency;
                    self.background_editor_generation =
                        self.background_editor_generation.wrapping_add(1);
                    self.settings.creator_open = true;
                    self.settings.color_picker = None;
                }
            }
            Action::CreateTheme => {
                let raw_name = self.settings.name.trim();
                let editing = self.settings.editing_theme.take();
                let was_active = editing
                    .as_deref()
                    .is_some_and(|old| self.prefs.active_theme() == Some(old));
                let final_name = match editing.as_deref() {
                    Some(old) if raw_name == old => old.to_owned(),
                    Some(old) => {
                        let others: Vec<ThemeRec> = self
                            .themes
                            .iter()
                            .filter(|t| t.name != old)
                            .cloned()
                            .collect();
                        unique_theme_name(raw_name, &others)
                    }
                    None => unique_theme_name(raw_name, &self.themes),
                };
                let existing_fav = editing
                    .as_deref()
                    .and_then(|old| self.themes.iter().find(|t| t.name == old))
                    .is_some_and(|t| t.favorite);
                let theme = ThemeRec {
                    name: final_name.clone(),
                    primary: rgb_value(self.settings.primary),
                    secondary: rgb_value(self.settings.secondary),
                    accent: rgb_value(self.settings.accent),
                    favorite: existing_fav,
                    lyrics_fill: Some(rgb_value(self.settings.lyrics_fill)),
                    lyrics_outline: Some(rgb_value(self.settings.lyrics_outline)),
                    visualizer_mode: self.settings.visualizer_mode,
                    visualizer_color: Some(rgb_value(self.settings.visualizer_color)),
                    background: self.settings.theme_background.clone(),
                    background_blur: self.settings.theme_background_blur,
                    background_transparency: self.settings.theme_background_transparency,
                };
                if let Some(old) = editing.as_deref() {
                    if let Some(slot) = self.themes.iter_mut().find(|t| t.name == old) {
                        *slot = theme;
                    } else {
                        self.themes.push(theme);
                    }
                } else {
                    self.themes.push(theme);
                }
                self.prefs.set_themes(&self.themes);
                self.settings.name.clear();
                self.settings.creator_open = false;
                self.settings.color_picker = None;
                if editing.is_none() || was_active {
                    set_theme(self, Some(final_name));
                } else if self.prefs.active_theme().is_some() {
                    set_theme(self, self.prefs.active_theme().map(str::to_owned));
                }
            }
            Action::PickBackground => {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(
                        "Images & Video",
                        &[
                            "png", "jpg", "jpeg", "webp", "jxl", "avif", "gif", "bmp", "tif",
                            "tiff", "ico", "tga", "pnm", "qoi", "dds", "hdr", "exr", "ff", "mp4",
                            "webm", "mov", "mkv", "avi", "m4v", "ogv",
                        ],
                    )
                    .pick_file()
                {
                    self.background_prep_error = None;
                    self.queue_background_prep(path, BackgroundPrepTarget::Global);
                }
            }
            Action::RemoveBackground => {
                self.cancel_background_prep();
                self.settings.confirm_dialog = None;
                self.background_prep_error = None;
                self.prefs.set_background(None);
                sync_background(self);
            }
            Action::ToggleGlobalBackground(enabled) => {
                self.cancel_background_prep();
                self.prefs.set_background_enabled(enabled);
                sync_background(self);
            }
            Action::PickThemeBackground => {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(
                        "Images & Video",
                        &[
                            "png", "jpg", "jpeg", "webp", "jxl", "avif", "gif", "bmp", "tif",
                            "tiff", "ico", "tga", "pnm", "qoi", "dds", "hdr", "exr", "ff", "mp4",
                            "webm", "mov", "mkv", "avi", "m4v", "ogv",
                        ],
                    )
                    .pick_file()
                {
                    let editing = self.settings.editing_theme.clone();
                    self.queue_background_prep(
                        path,
                        BackgroundPrepTarget::Theme(editing, self.background_editor_generation),
                    );
                }
            }
            Action::BlurStrength(strength) => {
                if self.prefs.background_enabled() && self.prefs.background().is_some() {
                    self.background.set_strength(strength);
                }
                self.prefs.set_background_blur(strength);
            }
            Action::ThemeBlurStrength(strength) => {
                self.settings.theme_background_blur = strength;
                let global_active =
                    self.prefs.background_enabled() && self.prefs.background().is_some();
                let editing_active = self.settings.editing_theme.as_deref().is_some()
                    && self.settings.editing_theme.as_deref() == self.prefs.active_theme();
                if !global_active && editing_active {
                    self.background.set_strength(strength);
                }
            }
            Action::BackgroundTransparency(transparency) => {
                self.prefs.set_background_transparency(transparency);
            }
            Action::ThemeBackgroundTransparency(transparency) => {
                self.settings.theme_background_transparency = transparency;
            }
            Action::ConfirmDialog(dialog) => self.settings.confirm_dialog = dialog,
            Action::SetDownloads(enabled) => {
                if !enabled {
                    // Final, as the dialog says: every downloaded file goes now.
                    self.library.clear_downloads();
                }
                self.prefs.set_downloads(enabled);
                self.library.set_downloading(enabled);
                if enabled {
                    let effects = self.library.resume();
                    self.run_effects(effects);
                }
                self.settings.confirm_dialog = None;
            }
            Action::SetEq(gains) => {
                self.eq = gains;
                self.prefs.set_eq(gains);
            }
            Action::ResetEq => {
                self.eq = [0.0; 10];
                self.prefs.set_eq(self.eq);
            }
            Action::ToggleSidebar => self.sidebar_open = !self.sidebar_open,
            Action::ToggleLyrics => self.set_lyrics_open(!self.lyrics_open),
            Action::Mute => {
                if self.volume > 0.0 {
                    self.volume_before_mute = Some(self.volume);
                    self.apply_volume(0.0);
                } else {
                    let restored = self.volume_before_mute.take().unwrap_or(DEFAULT_VOLUME);
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
        _ => {
            Lookup::Found(yt_lyrics::Lyrics::synced(source, timed).expect("the stand-in has text"))
        }
    }
}

/// Seconds from a row's `m:ss` or `h:mm:ss` length; 0 when there is none or it does not parse.
fn hit_seconds(duration: Option<&str>) -> f32 {
    duration
        .and_then(|text| {
            text.split(':').try_fold(0.0_f32, |total, part| {
                part.parse::<f32>().ok().map(|p| total * 60.0 + p)
            })
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
    use super::{
        advance_badge, background_prep_failure_updates_global_error,
        background_theme_target_is_current, bars_shown, BackgroundPrepTarget, BADGE_QUIET_HOLD,
        BAR_LINGER, SEARCH_CLOSE_RATE, SEARCH_OPEN_RATE,
    };
    use crate::anim::{rate, Anims};
    use std::time::{Duration, Instant};

    #[test]
    fn background_theme_result_requires_the_same_open_editor_session() {
        let theme = Some("theme-1".to_owned());
        assert!(background_theme_target_is_current(
            true, &theme, 7, &theme, 7
        ));
        assert!(!background_theme_target_is_current(
            false, &theme, 7, &theme, 7
        ));
        assert!(!background_theme_target_is_current(
            true, &theme, 8, &theme, 7
        ));
        assert!(!background_theme_target_is_current(
            true,
            &Some("theme-2".to_owned()),
            7,
            &theme,
            7
        ));
    }

    #[test]
    fn only_global_background_failures_replace_the_global_error() {
        assert!(background_prep_failure_updates_global_error(
            &BackgroundPrepTarget::Global
        ));
        assert!(!background_prep_failure_updates_global_error(
            &BackgroundPrepTarget::Theme(None, 1)
        ));
        assert!(!background_prep_failure_updates_global_error(
            &BackgroundPrepTarget::Theme(Some("theme".to_owned()), 2)
        ));
    }

    /// The scrollbar stays up for a moment after the last scroll, then goes. Scrolling, and reaching
    /// for the bar, bring it straight back.
    #[test]
    fn the_scrollbar_lingers_after_the_last_scroll_then_goes() {
        let now = Instant::now();
        let just_scrolled = now - Duration::from_secs_f32(BAR_LINGER * 0.5);
        let long_ago = now - Duration::from_secs_f32(BAR_LINGER + 0.5);
        assert!(
            bars_shown(now, Some(just_scrolled), false, false),
            "it should still be up during the linger"
        );
        assert!(
            !bars_shown(now, Some(long_ago), false, false),
            "it should be gone after the linger"
        );
        assert!(
            bars_shown(now, Some(long_ago), true, false),
            "the pointer on it should bring it back"
        );
        assert!(
            bars_shown(now, None, false, true),
            "scrolling should bring it up"
        );
        assert!(
            !bars_shown(now, None, false, false),
            "it should stay hidden before anything scrolls"
        );
    }

    /// Played out frame by frame: the bar holds for the whole linger, then fades out within a quarter
    /// of a second, and the motion stops so the host can park.
    #[test]
    fn the_scrollbar_holds_its_linger_then_fades_out() {
        let start = Instant::now();
        let step = 1.0 / 60.0;
        let mut anims = Anims::default();
        let mut alpha = 1.0;
        for frame in 0..120 {
            let elapsed = frame as f32 * step;
            let now = start + Duration::from_secs_f32(elapsed);
            let shown = bars_shown(now, Some(start), false, false);
            anims.begin(step);
            let bar_rate = if shown { rate::QUICKER } else { rate::QUICK };
            alpha = anims.toggle("bar", 0, shown, bar_rate);
            if elapsed < BAR_LINGER - 0.05 {
                assert!(
                    alpha > 0.99,
                    "it faded during the linger, at {elapsed:.2}s: {alpha}"
                );
            }
        }
        // Two seconds after the last scroll it is gone, and nothing is left moving.
        assert_eq!(alpha, 0.0, "it never faded out");
        assert!(!anims.moving(), "the host should be free to park");
    }

    /// The badge over a playing cover must not blink while the volume slider moves the meters
    /// across the threshold. It latches on, tolerates short dips, and only clears after a pause.
    #[test]
    fn the_cover_equalizer_badge_latches_through_a_volume_change() {
        let start = Instant::now();
        let mut quiet = None;
        // Silence before playback: no badge.
        assert!(!advance_badge(false, &mut quiet, [0.0; 3], start));
        // The first clearly audible block turns it on.
        assert!(advance_badge(false, &mut quiet, [0.09, 0.2, 0.1], start));
        // Meter readings dipping under the on-level for a few frames do not blink it.
        for frame in 0..12 {
            assert!(advance_badge(
                true,
                &mut quiet,
                [0.018, 0.012, 0.015],
                start + Duration::from_millis(16 * frame)
            ));
        }
        // Nor does a drag whose readings cross the on-level back and forth.
        for (frame, level) in [0.021, 0.019, 0.03, 0.012, 0.025, 0.01]
            .into_iter()
            .enumerate()
        {
            let now = start + Duration::from_millis(300 + 16 * frame as u64);
            assert!(advance_badge(true, &mut quiet, [level; 3], now));
        }
        // A long quiet spell clears it.
        let quiet_at = start + Duration::from_secs(1);
        let after = quiet_at + BADGE_QUIET_HOLD + Duration::from_millis(1);
        assert!(
            advance_badge(true, &mut quiet, [0.001; 3], quiet_at),
            "the hold starts now"
        );
        assert!(!advance_badge(true, &mut quiet, [0.001; 3], after));
    }

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
    fn pressing_the_song_that_is_already_playing_toggles_it() {
        use super::{same_track_press, TrackPress};
        assert_eq!(same_track_press(Some("a"), None, "a"), TrackPress::Toggle);
        assert_eq!(same_track_press(Some("a"), None, "b"), TrackPress::Start);
        // A second press while it is still loading must not start it over.
        assert_eq!(same_track_press(None, Some("a"), "a"), TrackPress::Ignore);
        assert_eq!(
            same_track_press(Some("a"), Some("a"), "a"),
            TrackPress::Ignore
        );
        // Pressing the fading song while another loads starts it again.
        assert_eq!(
            same_track_press(Some("a"), Some("b"), "a"),
            TrackPress::Start
        );
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
        assert_eq!(
            pages.top, None,
            "back from the first page is the results or the tab"
        );
        assert!(!pages.back(), "nothing left to go back to");
    }

    #[test]
    fn opening_the_page_already_on_top_changes_nothing() {
        use super::{Detail, Pages};
        let mut pages = Pages::default();
        assert!(pages.open(Detail::Album("b".to_owned())));
        assert!(!pages.open(Detail::Album("b".to_owned())));
        assert!(pages.back());
        assert_eq!(
            pages.top, None,
            "one open, one back: the repeat did not add a step"
        );
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

/// How long a scrollbar stays up after the reader stops scrolling, before it fades away.
const BAR_LINGER: f32 = 1.2;
/// How close to the page's right edge the pointer brings the scrollbar back up, in pixels. The bar
/// floats in the margin there, and the rows keep clear of it.
const BAR_REACH: f32 = 30.0;

/// Whether the scrollbar shows at `now`: while the reader is scrolling, while the pointer is on it,
/// and for [`BAR_LINGER`] after the last scroll.
fn bars_shown(now: Instant, last: Option<Instant>, hot: bool, scrolling: bool) -> bool {
    scrolling || hot || last.is_some_and(|at| since(now, at) < BAR_LINGER)
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
    let mut input = input;
    if state.theater && state.lyrics_open {
        // The UI scroll engine receives no reader input; programmatic following still runs.
        input.wheel = 0.0;
        input.wheel_precise = 0.0;
        input.nav = Default::default();
        input.mouse = (-1.0, -1.0);
        input.mouse_down = false;
        input.mouse_pressed = false;
        input.mouse_released = false;
        state.lyrics_follow.reader = false;
        state.lyrics_follow.scrolled_at = None;
    }
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
    // before `input` moves into the UI, along with the keys, which scroll the view under the pointer.
    let wheel = input.wheel + input.wheel_precise;
    let keys = input.nav;

    // The track menu closes on a scroll, and on a press outside it. A right press outside it closes it
    // and then reaches the page, so right-clicking another track opens that track's menu. A left press
    // outside it only closes it: it is used up here, so it does not also press what is underneath.
    let menu_entries = match state.menu.as_ref() {
        Some(menu) => track_menu_entries(state, menu),
        None => Vec::new(),
    };
    let mut left_dismissed = false;
    if let Some(rect) = state
        .menu
        .as_ref()
        .map(|menu| menu_rect(menu, menu_entries.len(), width, height))
    {
        let inside = rect.contains(pointer.pos);
        if !inside && (pointer.pressed || pointer.secondary || wheel != 0.0) {
            left_dismissed = pointer.pressed;
            state.menu = None;
        }
    }

    ui.begin(input, (width, height));
    ui.set_theme(&dark);
    ui.set_focus_rings(false);
    ui.rect(Rect::new(0.0, 0.0, width, height), theme::bg());
    draw_background(ui, &gpu, width, height);

    // Layout.
    let theater = state.theater && state.lyrics_open;
    let t = anims.toggle("theater", 0, theater, rate::PANEL);
    let body_h = height - theme::BAR_H * (1.0 - t);
    let want_open = state.sidebar_open && width >= theme::SIDEBAR_AUTO_SHUT;
    let open_t = anims.toggle("sidebar", 0, want_open, rate::PANEL);
    let sidebar_w =
        (theme::SIDEBAR_SHUT + (theme::SIDEBAR_OPEN - theme::SIDEBAR_SHUT) * open_t).round();
    let full_w = sidebar_w;
    let sidebar_w = sidebar_w * (1.0 - t);
    let sidebar = Rect::new(sidebar_w - full_w, 0.0, full_w, body_h);
    let main = Rect::new(sidebar_w, 0.0, width - sidebar_w, body_h);
    let bar = Rect::new(0.0, body_h, width, theme::BAR_H);
    // The window's own caption buttons, in the top-right corner over the page, centred on the
    // search field's line so the two sit level.
    let caption = Rect::new(
        width - theme::CAPTION_INSET - theme::CAPTION_BTN_W * 3.0,
        theme::SEARCH_TOP + (theme::SEARCH_H - theme::CAPTION_H) * 0.5,
        theme::CAPTION_BTN_W * 3.0,
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
    let open_zone = if results_page || on_detail {
        theme::TOP_H + 150.0
    } else {
        124.0
    };
    let zone_h = if state.search_wanted {
        open_zone
    } else {
        theme::TOP_H
    };
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
            ui.scroll_by(
                &list_id,
                target - ui.scroll_state(&list_id).0,
                dream::ui::widgets::ScrollBehavior::Instant,
            );
        }
        state.results_scroll = ui.scroll_state(&list_id).0;
    }
    // `V1_SHOT_SCROLL=<px>` scrolls the list on screen to that offset, to check the glass over rows.
    // It counts as the reader scrolling, so the bar is up in the shot as it would be for them.
    if let Some(target) = std::env::var("V1_SHOT_SCROLL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        state.bar_at = Some(now);
        let current = ui.scroll_state(&list_id).0;
        if (target - current).abs() > 0.5 {
            ui.scroll_by(
                &list_id,
                target - current,
                dream::ui::widgets::ScrollBehavior::Instant,
            );
        }
    }

    // The scrollbar hides itself when the reader is not scrolling. A wheel turn, a key or a press on
    // the bar brings it up at once; it lingers for a moment after the last of that, then fades away.
    // Reaching for the page's right edge, where the bar floats, or holding it keeps it up.
    let over_bar = !theater
        && pointer.pos.0 > width - BAR_REACH
        && pointer.pos.1 > theme::SEARCH_TOP + theme::CAPTION_H
        && pointer.pos.1 < body_h;
    if pointer.pressed && over_bar {
        state.bar_grab = true;
    } else if !pointer.down {
        state.bar_grab = false;
    }
    let keys_scrolled = !focused
        && (keys.up || keys.down || keys.page_up || keys.page_down || keys.home || keys.end);
    let scrolling = state.bar_grab || keys_scrolled || (wheel != 0.0 && main.contains(pointer.pos));
    state.bar_hover = over_bar;
    if scrolling {
        state.bar_at = Some(now);
    }
    let bar_up = bars_shown(now, state.bar_at, over_bar, scrolling);
    // It comes back quickly, and goes a little more gently.
    let bar_rate = if bar_up { rate::QUICKER } else { rate::QUICK };
    let bar_alpha = anims.toggle("bar", 0, bar_up, bar_rate);
    if !bar_up && bar_alpha <= 0.0 {
        // Fully away: nothing left for the host to wake for.
        state.bar_at = None;
    }
    ui.scrollbar = ScrollbarStyle {
        width: 4.0,
        inset: 14.0,
        min_thumb: 40.0,
        idle: fade(theme::thumb(), bar_alpha),
        active: fade(theme::thumb_hot(), bar_alpha),
    };

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
    let search_h = theme::SEARCH_H;
    let hidden_y = -(search_h + 20.0);
    let previous_reveal = anims.value("search", 0).unwrap_or(0.0);
    let was_hidden = hidden_y + (theme::SEARCH_TOP - hidden_y) * previous_reveal + search_h <= 0.0;
    let stiffness = if wanted {
        SEARCH_OPEN_RATE
    } else {
        SEARCH_CLOSE_RATE
    };
    let reveal = anims.toggle("search", 0, wanted, stiffness);
    if wanted || reveal <= 0.001 {
        state.search_ghost = None;
    }
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
    // Prepare the horizontal layout offscreen, before the vertical reveal advances. Otherwise
    // opening a detail page from a hidden bar animates both axes and makes it drop diagonally.
    if was_hidden {
        anims.snap("back", 0, if back_wanted { 1.0 } else { 0.0 });
    }
    let back_stiffness = if back_wanted {
        SEARCH_OPEN_RATE
    } else {
        SEARCH_CLOSE_RATE
    };
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
    let covered = !theater
        && ((search.y + search.h > 0.0 && search.contains(pointer.pos))
            || (back_visible && back.contains(pointer.pos)));
    let over_caption = !theater && caption.contains(pointer.pos);

    // The spinner turns while a song loads, the same for the bar and the rows.
    if state.playlist.is_loading() {
        state.spin = (state.spin + anims.dt() * SPINNER_TURN) % std::f32::consts::TAU;
    }
    // Everything drawn over the page. The page is drawn first, so these are known up front and a
    // page widget under one of them does not take the press.
    let mut covers = vec![bar, sidebar];
    if !theater {
        covers.push(caption);
    }
    if !theater && search.y + search.h > 0.0 {
        covers.push(search);
    }
    if !theater && back_visible {
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
    // The download confirmation and colour picker cover the window: nothing beneath them takes a press.
    if state.settings.confirm_dialog.is_some() || state.settings.color_picker.is_some() {
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
        wanted_meta: Vec::new(),
        touched: std::cell::Cell::new(false),
        covering: true,
        covers,
        spin: state.spin,
        saved: state.library.ids(),
    };

    // Exiting Settings cancels any in-progress theme builder or colour picker.
    if (state.tab != Tab::Settings || state.lyrics_open || state.pages.top.is_some())
        && (state.settings.creator_open || state.settings.color_picker.is_some())
    {
        state.settings.creator_open = false;
        state.settings.editing_theme = None;
        state.settings.name.clear();
        state.settings.color_picker = None;
    }

    // Page first, then the surfaces that sit over it.
    page(&mut cx, state, track.as_ref(), main, now, &mut actions);
    cx.covering = false;
    // Over the caption buttons, everything beneath them ignores the pointer, the page included.
    // The confirmation dialog does the same to everything it covers: only its own buttons are
    // live while it is up.
    cx.blocked = over_caption
        || state.settings.confirm_dialog.is_some()
        || state.settings.color_picker.is_some();
    if sidebar_w > 0.001 {
        cx.blocked |= theater;
        cx.clip(Rect::new(0.0, 0.0, sidebar_w, body_h));
        draw_sidebar(&mut cx, state, sidebar, open_t, &mut actions);
        cx.unclip();
        // The join extends to the right of the sidebar, outside its content clip.
        let panel_alpha = ui_panel_alpha(state);
        draw_join_crescent(&mut cx, sidebar, panel_alpha);
    }
    // The chrome floats over the content, which now scrolls up under it. Its layer sorts it above
    // the page's rows, so a row's text never paints over the search bar or the caption icons.
    let page_layer = cx.draw().set_layer(CHROME_LAYER);
    if !theater {
        draw_search(&mut cx, state, search);
        draw_back(&mut cx, back, back_t, &mut actions);
    }
    draw_bar(&mut cx, state, track.as_ref(), bar, &mut actions);
    cx.blocked = false;
    if !theater {
        draw_caption(&mut cx, caption, state.fullscreen, &mut actions);
    }
    cx.blocked = false;
    cx.draw().set_layer(page_layer);

    if state.lyrics_open {
        cx.blocked = state.menu.is_some()
            || state.settings.confirm_dialog.is_some()
            || state.settings.color_picker.is_some();
        let hit = Rect::new(main.x + 16.0, 16.0, 32.0, 32.0);
        let hot = hit.contains(pointer.pos);
        let shown = !theater || hot;
        let fade_rate = if shown { rate::QUICK } else { rate::QUICKER };
        let a = cx.anims.toggle("theater_button", 0, shown, fade_rate);
        let hover = a > 0.5 && cx.over_clickable(hit);
        let hv = cx.anims.toggle("theater_hover", 0, hover, rate::QUICK);
        if a > 0.001 {
            cx.draw()
                .round_rect(hit, 8.0, fade(theme::panel_hover(), hv * a));
            let c = fade(mix(theme::muted(), theme::fg(), hv), a);
            // Transparent outline: every stroke fades together over any background.
            cx.draw()
                .round_ring(Rect::new(hit.x + 7.0, hit.y + 8.0, 18.0, 13.0), 2.5, 1.5, c);
            cx.draw().line(
                (hit.x + 16.0, hit.y + 21.0),
                (hit.x + 16.0, hit.y + 24.0),
                1.5,
                c,
            );
            cx.draw().line(
                (hit.x + 12.0, hit.y + 24.0),
                (hit.x + 20.0, hit.y + 24.0),
                1.5,
                c,
            );
            if a > 0.5 && cx.clicked(hit) {
                state.toggle_theater();
            }
        }
    }
    // The track menu sits over everything, the caption buttons included.
    draw_track_menu(&mut cx, state, &menu_entries, width, height, &mut actions);
    if state.settings.color_picker.is_some() {
        draw_color_picker(&mut cx, state, width, height);
    }
    if let Some(dialog) = state.settings.confirm_dialog.as_ref() {
        draw_confirm_dialog(&mut cx, dialog, width, height, &mut actions);
    }

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
        && state.settings.confirm_dialog.is_none()
        && state.settings.color_picker.is_none()
    {
        actions.push(Action::Window(WindowRequest::Drag));
    }

    let hand = cx.hand;
    state.drag = cx.drag;
    state.queue_drag = cx.queue_drag;
    let wanted_icons = std::mem::take(&mut cx.wanted_icons);
    let wanted_covers = std::mem::take(&mut cx.wanted_covers);
    let wanted_meta = std::mem::take(&mut cx.wanted_meta);
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
    for hit in wanted_meta {
        state.request_metadata(hit);
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
fn draw_join_crescent(cx: &mut Cx, sidebar: Rect, panel_alpha: f32) {
    let r = theme::JOIN_R;
    let right = sidebar.x + sidebar.w;
    let bottom = sidebar.y + sidebar.h;
    let corner = Rect::new(right, bottom - r, r, r);
    // Centre of the fillet circle: `r` in from the sidebar edge and `r` up from the bar.
    let center = (right + r, bottom - r);
    if cx.gpu.background.is_some() {
        cx.draw()
            .concave_corner(corner, center, r, Color::WHITE, true);
        cx.draw()
            .concave_corner(corner, center, r, fade(theme::panel(), panel_alpha), false);
    } else {
        cx.draw()
            .concave_corner(corner, center, r, theme::panel(), false);
    }
}

fn draw_sidebar(cx: &mut Cx, state: &State, rect: Rect, open_t: f32, actions: &mut Vec<Action>) {
    if cx.gpu.background.is_some() {
        cx.draw().backdrop_rect(rect, Color::WHITE);
        cx.draw()
            .rect(rect, fade(theme::panel(), ui_panel_alpha(state)));
    } else {
        cx.draw().rect(rect, theme::panel());
    }
    // Labels appear in the second half of the opening, and leave in the first half of closing, so
    // they are never squeezed against a narrow edge.
    let label_a = ((open_t - 0.55) / 0.45).clamp(0.0, 1.0);

    // Both the logo and the toggle live in the same 43 × 40 box, centred on x = 34 like the tab
    // icons. Shut, the box holds the logo, and hovering it swaps in the toggle. Open, the logo
    // stays on the left and the toggle moves to the top right. At the shut width (68) the two
    // boxes are the same, so the toggle slides across as the sidebar animates.
    let logo = Rect::new(rect.x + 12.5, 16.0, 43.0, 40.0);
    let toggle = Rect::new(rect.x + rect.w - 56.0, 16.0, 43.0, 40.0);
    let shut = open_t < 0.5;
    let logo_hover = shut && cx.over_clickable(logo);
    let swap = cx.anims.toggle("logo_swap", 0, logo_hover, rate::QUICKER);
    // Shut, the toggle shows only while the logo is hovered; open, it is always there.
    let toggle_a = if shut { swap } else { 1.0 };
    let logo_a = if shut { 1.0 - swap } else { 1.0 };

    if logo_a > 0.0 {
        cx.wordmark(
            (rect.x + 34.0, 36.0),
            (32.0, 30.0),
            fade(Color::WHITE, logo_a),
        );
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
            fade(mix(theme::muted(), theme::fg(), hv), toggle_a),
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
    // While the lyric page, a detail page (artist, album, ...) or the search results are up, they
    // are the page: no tab is selected.
    let on_tab = !state.lyrics_open && state.pages.top.is_none() && state.results_query.is_empty();
    for (index, tab) in Tab::ALL.iter().copied().enumerate() {
        let item = Rect::new(rect.x + 12.0, top + index as f32 * pitch, item_w, item_h);
        let selected = on_tab && tab == state.tab;
        if sidebar_item(cx, item, tab, selected, label_a, index as u32) {
            actions.push(Action::Tab(tab));
        }
    }
    // Settings lives at the foot of the sidebar, above the player bar, away from the destinations.
    let item = Rect::new(rect.x + 12.0, rect.h - 48.0, item_w, item_h);
    let selected = on_tab && state.tab == Tab::Settings;
    if sidebar_item(cx, item, Tab::Settings, selected, label_a, 2) {
        actions.push(Action::Tab(Tab::Settings));
    }
}

/// One sidebar destination: a pill when chosen, a wash under the pointer, the icon fixed at x = 34
/// so it never moves while the sidebar folds. Returns whether it was pressed.
fn sidebar_item(cx: &mut Cx, item: Rect, tab: Tab, selected: bool, label_a: f32, key: u32) -> bool {
    let has_bg = cx.gpu.background.is_some();
    let hover = cx.over_clickable(item);
    let hv = cx.anims.toggle("tab_hv", key, hover, rate::SNAP);
    if selected {
        let fill = if has_bg {
            fade(theme::panel_select(), 0.72)
        } else {
            theme::panel_select()
        };
        cx.draw().round_rect(item, 11.0, fill);
    } else if has_bg {
        if hv > 0.005 {
            cx.draw()
                .round_rect(item, 11.0, fade(theme::panel_hover(), hv * 0.65));
        }
    } else {
        cx.draw()
            .round_rect(item, 11.0, mix(theme::panel(), theme::panel_hover(), hv));
    }
    let color = mix(theme::muted(), theme::fg(), if selected { 1.0 } else { hv });
    cx.icon(
        tab.icon(),
        (item.x + 22.0, item.y + item.h * 0.5),
        19.0,
        color,
    );
    cx.clip(item);
    cx.ui.label(
        Rect::new(item.x + 40.0, item.y, 150.0, item.h),
        tab.label(),
        dream::ui::widgets::LabelStyle::new(FontId(2))
            .size(13.0)
            .color(fade(color, label_a))
            .align(Align::Left)
            .vcenter(true),
    );
    cx.unclip();
    cx.clicked(item)
}

// ---------------------------------------------------------------------------------------------
// Caption
// ---------------------------------------------------------------------------------------------

/// The window's own minimise, fullscreen and close buttons, top right. There is no OS title bar, so the app
/// draws them in its own greys. No shared container or group hover: each button only lifts a step
/// under the pointer.
fn draw_caption(cx: &mut Cx, rect: Rect, fullscreen: bool, actions: &mut Vec<Action>) {
    let buttons = [
        (Icon::Minus, WindowRequest::Minimize),
        (
            if fullscreen {
                Icon::ExitFullscreen
            } else {
                Icon::Fullscreen
            },
            WindowRequest::ToggleFullscreen,
        ),
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
        let hv = cx
            .anims
            .toggle("caption_btn", index as u32, hover, rate::SNAP);
        // Close turns its glyph red on hover; the buttons keep the grey container behind them.
        let is_close = matches!(request, WindowRequest::Close);
        if hv > 0.0 {
            let lit = Rect::new(
                button.x + 3.0,
                button.y + 3.0,
                button.w - 6.0,
                button.h - 6.0,
            );
            cx.draw()
                .round_rect(lit, 8.0, fade(theme::panel_press(), hv));
        }
        let hot = if is_close {
            theme::CLOSE_RED
        } else {
            theme::fg()
        };
        let color = mix(theme::muted(), hot, hv);
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
    cx.draw()
        .backdrop_rounded(rect, Color::rgb(0xff, 0xff, 0xff), rect.h * 0.5);
    // The tint matches the sidebar and player bar (`theme::panel()`), brightening on hover/focus,
    // without using the accent colour.
    cx.draw().round_rect(
        rect,
        rect.h * 0.5,
        fade(
            mix(theme::panel(), theme::panel_press(), lift),
            0.62 + 0.12 * lift,
        ),
    );
    cx.icon(
        Icon::Search,
        (rect.x + 28.0, rect.y + rect.h * 0.5),
        19.0,
        mix(theme::muted(), theme::fg(), focus_t),
    );

    let clear = Rect::new(rect.x + rect.w - 42.0, rect.y + 9.0, 28.0, 28.0);
    let field = Rect::new(rect.x + 54.0, rect.y, rect.w - 54.0 - 48.0, rect.h);

    if state.focus_search {
        state.focus_search = false;
        cx.ui.set_focus("search");
    }
    let mut style = TextFieldStyle::new(cx.font);
    style.size = 15.5;
    style.text = theme::fg();
    style.normal = Color::TRANSPARENT;
    style.hover = Color::TRANSPARENT;
    style.accent = theme::fg();
    style.selection_alpha = 56;
    style.corner_radius = 0.0;
    style.border_width = 0.0;
    style.caret = theme::fg();
    style.caret_width = 1.5;
    style.caret_inset = 13.0;
    // Selection highlight spans the same height as the caret, so the two line up.
    style.selection_inset = style.caret_inset;
    style.text_inset = 0.0;
    cx.ui.text_field("search", field, &mut state.query, style);

    if let (false, Some(ghost)) = (typed, state.search_ghost.as_deref()) {
        cx.text(field, ghost, 15.5, theme::fg(), Align::Left);
    } else if !typed {
        cx.text(
            field,
            "Search songs and artists",
            15.5,
            fade(theme::muted(), 1.0 - 0.35 * focus_t),
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
                theme::raised_top(),
            );
        }
        cx.icon(
            Icon::X,
            center,
            16.0,
            if hovered { theme::fg() } else { theme::muted() },
        );
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
        let blocked = cx.blocked;
        cx.blocked |= state.theater;
        lyrics_page(cx, state, track, main, now, actions);
        cx.blocked = blocked;
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
        Tab::Settings => settings_page(cx, state, view, actions),
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
                let rows: Vec<Row> = state
                    .playlist
                    .queue
                    .iter()
                    .map(|hit| Row::from(hit, &state.lengths))
                    .collect();
                track_list(
                    cx,
                    track,
                    &rows,
                    view,
                    "queue_list",
                    4000,
                    f32::INFINITY,
                    true,
                    actions,
                );
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
    let at = cx
        .anims
        .spring("collection_slide", 0, target.index() as f32, 380.0);
    // The scrollbars are hidden for the whole slide, then fade back in quickly as it lands. The
    // spring always tracks "not sliding", so it is already rising when the slide ends.
    let sliding = (at - target.index() as f32).abs() > 0.01;
    let bar_a = cx
        .anims
        .toggle("collection_bar", 0, !sliding, rate::QUICKER);
    let bar_a = if sliding { 0.0 } else { bar_a };
    for tab in CollectionTab::ALL {
        let dx = (tab.index() as f32 - at) * view.w;
        if dx.abs() >= view.w {
            continue;
        }
        let shift = |r: Rect| Rect::new(r.x + dx, r.y, r.w, r.h);
        collection_list(
            cx,
            state,
            track,
            tab,
            shift(main),
            shift(view),
            bar_a,
            actions,
        );
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
    if let Some(index) = tab_pills(
        cx,
        &labels,
        target.index(),
        view.x + 8.0,
        PAGE_TOP - offset,
        &COLLECTION_PILLS,
    ) {
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
    // Keep the initial clearance inside the content so it scrolls away with the list.
    let content_top = PAGE_TOP + COLLECTION_TAB_H + COLLECTION_TAB_GAP;

    // How many entries the list has, and what to say when it has none.
    let count = match tab {
        CollectionTab::Songs => state.library.song_list(&state.ghosts).len(),
        CollectionTab::Artists => state.library.vault.len(),
        CollectionTab::Albums => state.library.album_count(),
    };
    if count == 0 {
        let (icon, title, hint) = match tab {
            CollectionTab::Songs => (
                Icon::Music,
                "No saved songs",
                "Press + on a song to keep it here. It is downloaded to this device.",
            ),
            CollectionTab::Artists => (
                Icon::Mic,
                "No saved artists",
                "Press + on an artist's page to keep them here.",
            ),
            CollectionTab::Albums => (
                Icon::Disc,
                "No saved albums",
                "Press + on an album to save every song on it.",
            ),
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
            let first = (((offset - content_top) / theme::ROW_H).floor().max(0.0) as usize)
                .saturating_sub(2);
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
                true,
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
            let first_row =
                (((offset - content_top) / step).floor().max(0.0) as usize).saturating_sub(1);
            let rows_shown = (view.h / step).ceil() as usize + 3;
            let from = (first_row * columns).min(count);
            let to = ((first_row + rows_shown) * columns).min(count);
            // (title, meta, picture, where it opens), for just the tiles in view.
            let tiles: Vec<(String, String, Option<String>, Detail)> = match tab {
                CollectionTab::Artists => state.library.vault.list()[from..to]
                    .iter()
                    .map(|a| {
                        (
                            a.name.clone(),
                            "Artist".to_owned(),
                            a.thumbnail.clone(),
                            Detail::Artist(a.id.clone()),
                        )
                    })
                    .collect(),
                _ => state
                    .library
                    .albums_between(from, to)
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

/// The animation slots one row of pill tabs owns. Springs are keyed by `(name, index)`, so two rows
/// on screen at once must not share a name: the sidebar's own tabs use `tab_hv`, and each pill row
/// brings names of its own. Sharing one made hovering a pill move a sidebar tab.
struct PillKeys {
    hover: &'static str,
    on: &'static str,
    glide_x: &'static str,
    glide_w: &'static str,
}

/// The Collection page's tabs (Songs, Albums, Artists).
const COLLECTION_PILLS: PillKeys = PillKeys {
    hover: "collection_tab_hv",
    on: "collection_tab_on",
    glide_x: "collection_tab_glide_x",
    glide_w: "collection_tab_glide_w",
};

/// The search results' tabs (Songs, Albums, Artists).
const RESULT_PILLS: PillKeys = PillKeys {
    hover: "result_tab_hv",
    on: "result_tab_on",
    glide_x: "result_tab_glide_x",
    glide_w: "result_tab_glide_w",
};

/// A row of pill tabs (icon and label each), left-aligned at `(x, y)`. The `current` one is lit and
/// its pill glides between tabs; the others wash under the pointer. Returns the index of a tab
/// clicked this frame, never the chosen one. `keys` names this row's animation slots, so every row
/// on screen keeps its own.
fn tab_pills(
    cx: &mut Cx,
    tabs: &[(&'static str, Icon)],
    current: usize,
    x: f32,
    y: f32,
    keys: &PillKeys,
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
    let pill_x = cx.anims.spring(keys.glide_x, 0, lefts[current] - x, 700.0) + x;
    let pill_w = cx.anims.spring(keys.glide_w, 0, widths[current], 700.0);
    cx.draw().round_rect(
        Rect::new(pill_x, y, pill_w, COLLECTION_TAB_H),
        COLLECTION_TAB_H * 0.5,
        theme::press(),
    );

    let mut picked = None;
    for (index, (label, icon)) in tabs.iter().enumerate() {
        let rect = Rect::new(lefts[index], y, widths[index], COLLECTION_TAB_H);
        let selected = index == current;
        let hover = cx.over_clickable(rect);
        let hv = cx.anims.toggle(keys.hover, index as u32, hover, rate::SNAP);
        let on = cx.anims.toggle(keys.on, index as u32, selected, rate::SNAP);
        // Hover only warms the text and icon; the container belongs to the chosen tab alone.
        let ink = mix(theme::muted(), theme::fg(), hv.max(on));
        cx.icon(
            *icon,
            (
                rect.x + 17.0 + COLLECTION_TAB_ICON * 0.5,
                rect.y + rect.h * 0.5,
            ),
            COLLECTION_TAB_ICON,
            ink,
        );
        cx.text(
            Rect::new(
                rect.x + 17.0 + COLLECTION_TAB_ICON + 8.0,
                rect.y,
                rect.w,
                rect.h,
            ),
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
const LYRIC_ACTIVE_SIZE: f32 = 26.0;
/// The emphasis (a line lighting up as it is sung) is a spring like the rest of the UI, a little
/// quicker than the `PANEL` preset.
const LYRIC_RATE: f32 = 360.0;
const LYRIC_LEADING: f32 = 1.2;
const LYRIC_GAP: f32 = 18.0;
const PAUSE_H: f32 = 30.0;
const PLAIN_SIZE: f32 = 18.0;
const PLAIN_LEADING: f32 = 1.4;
const PLAIN_GAP: f32 = 8.0;
const STANZA_H: f32 = 24.0;
/// The lyric column is centred, never wider than this, and keeps this much room at each side.
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
    mix(theme::bg(), base, amount)
}

/// How far a line has arrived `age` seconds after its set appeared, 0 to 1.
fn arrival(age: f32, index: usize) -> f32 {
    ease_out((age - index as f32 * ENTER_STAGGER) / ENTER_TIME)
}

/// The centred column the lyrics are set in: its left edge and width.
fn lyric_column(view: Rect) -> (f32, f32) {
    let width = (view.w - 2.0 * LYRIC_COLUMN_PAD).max(120.0);
    (view.x + (view.w - width) * 0.5, width)
}

/// Pre-wrap a lyric line at `size` for `max_w` using explicit newlines so it keeps the same
/// line-by-line breaks regardless of animated font size.
fn wrap_lyric_lines(cx: &mut Cx, text: &str, size: f32, max_w: f32) -> String {
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut current = String::new();
        for word in para.split_whitespace() {
            if current.is_empty() {
                current.push_str(word);
            } else {
                let candidate = format!("{current} {word}");
                if cx.text_width(&candidate, size) <= max_w {
                    current = candidate;
                } else {
                    out.push(current);
                    current = word.to_owned();
                }
            }
        }
        out.push(current);
    }
    out.join("\n")
}

/// The sound behind the lyric page: the spectrum of what is playing as rounded bars along the bottom,
/// mirrored about the centre. Loud bands are
/// a step brighter than quiet ones. It fades in with the sound and out with it (silence, zero volume,
/// mute, or paused), and the bars settle rather than vanish.
fn lyric_visualizer(cx: &mut Cx, state: &mut State, track: Option<&TrackView>, main: Rect) {
    let dt = cx.anims.dt();
    let levels = track.map_or([0.0; 3], |t| t.levels);
    let loudest = levels.iter().copied().fold(0.0_f32, f32::max);
    let paused = track.is_some_and(|t| t.paused);
    let muted = state.volume_before_mute.is_some();
    let audible = track.is_some_and(|t| !t.paused) && (muted || loudest > VISUALIZER_AUDIBLE);

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
        _ if paused => {}
        _ => state.spectrum.decay(dt),
    }
    state.spectrum_live = !paused && (audible || state.spectrum.active());

    let show = cx
        .anims
        .toggle("lyric_visualizer", 0, audible || paused, rate::PANEL);
    let tint = cx
        .anims
        .toggle("lyric_visualizer_quiet", 0, paused || muted, rate::PANEL);
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
    let cover_color = track
        .and_then(|t| t.thumbnail.as_deref())
        .and_then(|url| cx.gpu.cover_color(url));
    let viz_color = match theme::visualizer() {
        theme::VisualizerColor::AlbumArt => cover_color,
        theme::VisualizerColor::Accent => Some(theme::accent()),
        theme::VisualizerColor::Custom(c) => Some(c),
    };
    let bar_colour = |level: f32, edge: f32| -> Color {
        if let Some(c) = viz_color {
            let lit = mix(mix(c, Color::WHITE, 0.14 * level), theme::bg(), tint);
            let alpha = ((0.42 + 0.46 * level) * edge * show * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            Color::rgba(lit.r, lit.g, lit.b, alpha)
        } else {
            let amount = VISUALIZER_QUIET + (VISUALIZER_LOUD - VISUALIZER_QUIET) * level;
            mix(
                theme::bg(),
                theme::fg(),
                amount * edge * show * (1.0 - tint),
            )
        }
    };

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
            let colour = bar_colour(level, edge);
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
        None => empty_state(
            cx,
            main,
            Icon::Music,
            "Nothing playing",
            "Play a song to see its lyrics.",
        ),
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

    let wrapped: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.text.is_empty() {
                String::new()
            } else {
                wrap_lyric_lines(cx, &line.text, LYRIC_ACTIVE_SIZE, width)
            }
        })
        .collect();
    let slots: Vec<f32> = wrapped
        .iter()
        .map(|text| {
            if text.is_empty() {
                PAUSE_H
            } else {
                cx.text_block_height(width, text, LYRIC_ACTIVE_SIZE, LYRIC_LEADING)
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

    let line_changed = active != follow.last_active;
    follow.last_active = active;

    // Once the reader has been still for a while, or when a new lyric becomes active after a
    // short pause since the last scroll, the page takes its place back on the line being sung.
    if follow.reader
        && (follow
            .scrolled_at
            .is_none_or(|at| since(now, at) >= LYRIC_IDLE_SECS)
            || (line_changed && follow.scrolled_at.is_none_or(|at| since(now, at) >= 1.5)))
    {
        follow.reader = false;
    }
    if !follow.reader {
        let behavior = if follow.placed {
            ScrollBehavior::Smooth
        } else {
            ScrollBehavior::Instant
        };
        follow.placed = true;
        cx.ui.scroll_to(
            "lyrics",
            ScrollToOptions {
                top: target,
                behavior,
            },
        );
    }
    let body = cx.ui.scroll_begin("lyrics", view, content_h);

    // Every line's spring runs each frame, even one that is off screen, so it is right when it returns.
    let emphasis: Vec<f32> = (0..lines.len())
        .map(|i| {
            cx.anims.toggle(
                "lyric_on",
                reveal.key + i as u32,
                active == Some(i),
                LYRIC_RATE,
            )
        })
        .collect();

    for (i, line) in lines.iter().enumerate() {
        if line.text.is_empty() {
            continue;
        }
        let is_active = active == Some(i);
        let size = LYRIC_SIZE + (LYRIC_ACTIVE_SIZE - LYRIC_SIZE) * emphasis[i];
        let zoom = size / LYRIC_ACTIVE_SIZE;
        let text = &wrapped[i];
        let arrived = arrival(reveal.age, i);
        let slot_top = body.y + tops[i] + (1.0 - arrived) * ENTER_RISE;
        if slot_top + slots[i] < view.y || slot_top > view.y + view.h {
            continue;
        }
        let rect = Rect::new(x, slot_top, width, slots[i]);
        let text_w = (text
            .lines()
            .map(|l| cx.text_width(l, LYRIC_ACTIVE_SIZE))
            .fold(0.0_f32, f32::max)
            * zoom)
            .min(width);
        let text_h = slots[i] * zoom;
        let hit_top = slot_top + (slots[i] - text_h) * 0.5;
        let hit_rect = Rect::new(x + (width - text_w) * 0.5, hit_top, text_w, text_h);
        let edge_fade = if is_active {
            1.0
        } else {
            let view_mid = view.y + view.h * 0.5;
            let line_mid = slot_top + slots[i] * 0.5;
            let fade_radius = (view.h * 0.32).max(1.0);
            let dist = ((line_mid - view_mid).abs() / fade_radius).clamp(0.0, 1.0);
            1.0 - dist * dist
        };
        let in_view = view.contains(cx.pointer.pos) && cx.pointer.pos.1 >= theme::TOP_H;
        let over = edge_fade > 0.0 && in_view && cx.over_clickable(hit_rect);
        let hv = cx.anims.toggle(
            "lyric_hover",
            reveal.key + i as u32,
            over && !is_active,
            rate::SNAP,
        );
        let lyrics_fg = theme::lyrics_fg();
        let lyrics_dim = mix(lyrics_fg, theme::bg(), 0.62);
        let rest = mix(lyrics_dim, lyrics_fg, emphasis[i]);
        let colour = fade(
            revealed(mix(rest, mix(lyrics_fg, theme::muted(), 0.45), hv), arrived),
            edge_fade,
        );
        lyric_text_outlined(
            cx,
            rect,
            text,
            LYRIC_ACTIVE_SIZE,
            colour,
            LYRIC_LEADING,
            emphasis[i],
            zoom,
        );
        if over && cx.clicked(hit_rect) && playhead.duration > 0.0 {
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
        let colour = revealed(theme::lyrics_fg(), arrived);
        lyric_text_outlined(cx, rect, line, PLAIN_SIZE, colour, PLAIN_LEADING, 1.0, 1.0);
    }
    cx.ui.scroll_end();
}

#[allow(clippy::too_many_arguments)]
fn lyric_text_outlined(
    cx: &mut Cx,
    rect: Rect,
    text: &str,
    size: f32,
    color: Color,
    leading: f32,
    outline_strength: f32,
    zoom: f32,
) {
    let alpha = (color.a as f32 / 255.0) * outline_strength.clamp(0.0, 1.0) * 0.55;
    if alpha > 0.01 {
        let outline = fade(theme::lyrics_outline(), alpha);
        for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
            let offset = Rect::new(rect.x + dx, rect.y + dy, rect.w, rect.h);
            if (zoom - 1.0).abs() > 1e-4 {
                cx.text_block_scaled(offset, text, size, outline, leading, Align::Center, zoom);
            } else {
                cx.text_block(offset, text, size, outline, leading, Align::Center);
            }
        }
    }
    if (zoom - 1.0).abs() > 1e-4 {
        cx.text_block_scaled(rect, text, size, color, leading, Align::Center, zoom);
    } else {
        cx.text_block(rect, text, size, color, leading, Align::Center);
    }
}

/// An icon, title and hint, centred in `main`, for when there are no lyrics to show.
fn empty_state(cx: &mut Cx, main: Rect, icon: Icon, title: &str, hint: &str) {
    let center_x = main.x + main.w * 0.5;
    let y = main.h * 0.5 - 8.0;
    cx.icon(icon, (center_x, y - 54.0), 38.0, theme::faint());
    cx.text(
        Rect::new(main.x, y - 16.0, main.w, 28.0),
        title,
        18.0,
        theme::fg(),
        Align::Center,
    );
    cx.text_fit(
        Rect::new(
            main.x + theme::PAGE_PAD,
            y + 14.0,
            main.w - theme::PAGE_PAD * 2.0,
            22.0,
        ),
        hint,
        13.0,
        theme::muted(),
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
    let top = theme::TOP_H;
    let content_h = top + rows.len() as f32 * theme::ROW_H + 24.0;
    let body = cx.ui.scroll_begin(scroll_id, view, content_h);
    let play: fn(usize) -> Action = if queued {
        Action::PlayQueued
    } else {
        Action::Play
    };
    track_rows(
        cx,
        track,
        rows,
        view,
        body,
        body.y + top,
        base,
        age,
        true,
        queued,
        play,
        0,
        actions,
    );
    cx.ui.scroll_end();
}

/// The rows of a track list, from `top` down, inside the open scroll area `body`. `view` is the
/// visible part of the page: a row takes the pointer only inside it. A press on a row plays it through
/// `play`, unless it landed on the artist or the album, which opens that page instead, or on the save
/// button, which saves the song.
///
/// `covers` draws each row's cover, with the play glyph or equalizer over it. A page that already
/// shows one cover for the whole list (an album's) passes `false`, and the rows start at their text.
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
    covers: bool,
    queued: bool,
    play: fn(usize) -> Action,
    first: usize,
    actions: &mut Vec<Action>,
) {
    let playing_id = track.map(|track| track.video_id.as_str());
    // A row wide enough for a column of albums gives the album one; one with no album to show uses
    // the whole width for its names.
    let columns = view.w > 760.0;
    let pointer_in_view = cx.pointer_in(view);

    // A queue row is moved by its handle. Letting go ends the drag and asks for the move to the
    // place the row was last over, so the list lands where the reader left it.
    if queued {
        if let Some(done) = cx.queue_drag.filter(|_| !cx.pointer.down) {
            cx.queue_drag = None;
            if done.to != done.from {
                actions.push(Action::MoveQueued {
                    from: done.from,
                    to: done.to,
                });
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
        let pill = Rect::new(
            body.x,
            y + 4.0,
            (body.w - 28.0).max(0.0),
            theme::ROW_H - 8.0,
        );
        let key = base + i as u32;

        let hovered = pointer_in_view && cx.over_clickable(pill);
        let hv = cx.anims.toggle("row", key, hovered, rate::SNAP);
        let press = cx
            .anims
            .toggle("row_press", key, hovered && cx.pointer.down, rate::SNAP);
        let playing = !queued && playing_id == Some(row.video_id);
        // A row from a list that does not carry the album's link can still fill it in later; a
        // row on an album page already knows its album from the page itself.
        if row.page.is_none() {
            cx.want_meta(row.hit);
        }

        // The playing row keeps a resting fill that brightens under the pointer.
        let wash = if playing {
            fade(
                mix(theme::select(), theme::press(), hv * (0.5 + 0.5 * press)),
                a,
            )
        } else {
            fade(mix(theme::hover(), theme::press(), press), hv * a)
        };
        // A dragged row is lifted: drawn in its own layer, over an opaque ground so the rows it
        // passes do not show through.
        let lift = dragged.then(|| cx.draw().set_layer(DRAG_LAYER));
        if dragged {
            cx.draw().round_rect(pill, 10.0, fade(theme::raised(), a));
        }
        cx.draw().round_rect(pill, 10.0, wash);

        // Cover, with the play glyph (hovered) or the equalizer (playing) over it. A list that
        // already has one cover at its head leaves the columns out, and the names start there.
        let handle_w = if queued { HANDLE_W } else { 0.0 };
        let cover_w = if covers { 44.0 } else { 0.0 };
        let cover = Rect::new(pill.x + 8.0 + handle_w, pill.y + 6.0, cover_w, 44.0);
        if covers {
            cx.cover(row.thumbnail, cover, 7.0, a);
            cover_overlay(cx, track, row.video_id, playing, cover, 7.0, a, hv);
        }

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
            let grip = mix(mix(theme::faint(), theme::muted(), hv), theme::fg(), gv);
            let grip_center = (handle.x + handle.w * 0.5, handle.y + handle.h * 0.5);
            cx.icon(Icon::GripVertical, grip_center, 16.0, fade(grip, a));
            handle_pressed = pointer_in_view && cx.clicked(handle);
            if handle_pressed && cx.queue_drag.is_none() {
                cx.queue_drag = Some(QueueDrag {
                    from: i,
                    to: i,
                    grab: cx.pointer.pos.1 - y0,
                });
            }

            let trash_hover = pointer_in_view && cx.over_clickable(trash);
            let tv = cx.anims.toggle("queue_trash", key, trash_hover, rate::SNAP);
            if tv > 0.002 {
                cx.draw()
                    .round_rect(trash, TRASH_R, fade(theme::CLOSE_RED, 0.16 * tv * a));
            }
            cx.icon(Icon::Trash2, trash_center, 16.0, fade(theme::CLOSE_RED, a));
            trash_clicked = pointer_in_view && cx.clicked(trash);
        }

        // With no cover the names start where one would have: on the page's left edge, level with
        // the album's own cover above them.
        let text_x = if covers {
            cover.x + cover.w + 14.0
        } else {
            cover.x
        };
        // Clear of the scrollbar, which floats in the right-hand margin. A queue row leaves room
        // for its remove button.
        let end = if queued {
            trash.x - 12.0
        } else {
            pill.x + pill.w - 16.0
        };
        let duration = Rect::new(end - 56.0, pill.y, 56.0, pill.h);
        cx.text(
            duration,
            row.duration,
            12.5,
            fade(theme::muted(), a),
            Align::Right,
        );

        let tail = duration.x;

        // Wide rows put the album in a column of its own; narrow ones run the artist and album on
        // one line. Either way the title takes what is left.
        let wide = columns && !row.album.is_empty();
        let album_w = (pill.w * 0.26).clamp(120.0, 300.0);
        let album_x = tail - 20.0 - album_w;
        // The save button sits right of the names, so the names stop short of it by its width and gap.
        let save_space = 14.0 + SAVE_ROW_R * 2.0;
        let full_w = (tail - 16.0 - text_x - save_space).max(40.0);
        let title_w = if wide {
            (album_x - 20.0 - text_x - save_space).max(40.0)
        } else {
            full_w
        };

        // The save button sits just right of the song and artist names, not out at the far edge.
        let title_text_w = cx.text_width(row.title, 14.5);
        let meta_text_w = if wide {
            cx.text_width(&row.artist, 12.5)
        } else {
            let sep = if row.artist.is_empty() || row.album.is_empty() {
                0.0
            } else {
                cx.text_width(ROW_SEP, 12.5)
            };
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
            fade(theme::fg(), a),
            row.album_id.map(|id| Detail::Album(id.to_owned())),
        );

        // The artist and the album are links: each brightens under the pointer, and a press on one
        // opens its page rather than playing the row.
        let meta_y = pill.y + 28.0;
        let opened = if wide {
            let artist = artist_text(
                cx,
                "link_artist",
                key,
                Rect::new(text_x, meta_y, title_w, 18.0),
                &row.artists,
                12.5,
                a,
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
            let artist = artist_text(
                cx,
                "link_artist",
                key,
                Rect::new(text_x, meta_y, full_w, 18.0),
                &row.artists,
                12.5,
                a,
            );
            if sep_w > 0.0 {
                cx.text_fit(
                    Rect::new(text_x + artist_w, meta_y, sep_w, 18.0),
                    ROW_SEP,
                    12.5,
                    fade(theme::muted(), a),
                    Align::Left,
                );
            }
            let album_text_x = text_x + artist_w + sep_w;
            let album = row_text(
                cx,
                "link_album",
                key,
                Rect::new(
                    album_text_x,
                    meta_y,
                    (text_x + full_w - album_text_x).max(0.0),
                    18.0,
                ),
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
            actions.push(Action::OpenMenu(TrackMenu {
                hit: row.song(),
                at: cx.pointer.pos,
            }));
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
        .spring(
            "save_fill",
            key,
            if saved { 1.0 } else { 0.0 },
            SAVE_FILL_RATE,
        )
        .clamp(0.0, 1.0);

    let idle = mix(mix(theme::faint(), theme::muted(), warm), theme::fg(), hv);
    let def_p = theme::Palette::default();
    let custom_theme = theme::accent() != def_p.accent || theme::bg() != def_p.bg;
    let save_color = if custom_theme {
        theme::accent()
    } else {
        SAVE_GREEN
    };
    let square = Rect::new(
        center.0 - radius,
        center.1 - radius,
        radius * 2.0,
        radius * 2.0,
    );
    cx.draw().ring_arc(
        square,
        radius,
        1.5,
        -std::f32::consts::FRAC_PI_2,
        std::f32::consts::TAU,
        fade(mix(idle, save_color, t), alpha),
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
            fade(save_color, alpha),
        );
    }
    let glyph = ((radius * 1.3) / 2.0).round() * 2.0;
    let plus = (1.0 - t * 2.5).clamp(0.0, 1.0);
    let check = ((t - 0.4) * 2.5).clamp(0.0, 1.0);
    if plus > 0.0 {
        cx.icon(Icon::Plus, center, glyph, fade(idle, alpha * plus));
    }
    if check > 0.0 {
        cx.icon(
            Icon::Check,
            center,
            glyph,
            fade(theme::on_accent(), alpha * check),
        );
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
        entries.push(MenuEntry {
            label: "Unsave",
            icon: Icon::Check,
            command: MenuCommand::Save,
        });
    } else {
        entries.push(MenuEntry {
            label: "Save",
            icon: Icon::Plus,
            command: MenuCommand::Save,
        });
    }
    match state
        .playlist
        .queue
        .iter()
        .position(|hit| hit.video_id == menu.hit.video_id)
    {
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
    let Some(menu) = state.menu.clone() else {
        return;
    };
    let rect = menu_rect(&menu, entries.len(), width, height);

    let previous = cx.draw().set_layer(MENU_LAYER);
    // A thin edge in a lighter tone, then the face inside it.
    cx.draw().round_rect(rect, 10.0, theme::track());
    cx.draw().round_rect(
        Rect::new(rect.x + 1.0, rect.y + 1.0, rect.w - 2.0, rect.h - 2.0),
        9.0,
        theme::raised(),
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
            cx.draw()
                .round_rect(row, 7.0, mix(theme::raised(), theme::press(), hv));
        }
        let ink = mix(theme::muted(), theme::fg(), hv);
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
        cx.text_fit(rect, text, size, fade(theme::muted(), alpha), Align::Left);
        return None;
    };
    let hit = Rect::new(
        rect.x,
        rect.y,
        cx.text_width(text, size).min(rect.w),
        rect.h,
    );
    let hovered = cx.over_clickable(hit);
    let hv = cx.anims.toggle(name, key, hovered, rate::SNAP);
    cx.text_fit(
        rect,
        text,
        size,
        fade(mix(theme::muted(), theme::fg(), hv), alpha),
        Align::Left,
    );
    cx.clicked(hit).then_some(target)
}

/// Between a song's artist credits, as `artists.join(", ")` puts it.
const ARTIST_SEP: &str = ", ";

/// A song's artists, side by side and comma-separated, each its own [`row_text`] link to that
/// artist's page (a credit without a page stays plain). Returns the artist pressed, if any.
#[allow(clippy::too_many_arguments)]
fn artist_text(
    cx: &mut Cx,
    name: &'static str,
    key: u32,
    rect: Rect,
    artists: &[(&str, Option<&str>)],
    size: f32,
    alpha: f32,
) -> Option<Detail> {
    let right = rect.x + rect.w;
    let mut x = rect.x;
    let mut opened = None;
    for (i, (artist, id)) in artists.iter().enumerate() {
        if i > 0 {
            if x >= right {
                break;
            }
            cx.text_fit(
                Rect::new(x, rect.y, right - x, rect.h),
                ARTIST_SEP,
                size,
                fade(theme::muted(), alpha),
                Align::Left,
            );
            x += cx.text_width(ARTIST_SEP, size);
        }
        if x >= right {
            break;
        }
        // Each credit hovers on its own; the first keeps the key a lone artist always had.
        let link_key = key.wrapping_add((i as u32).wrapping_mul(0x9E37_79B9));
        let target = id.map(|id| Detail::Artist(id.to_owned()));
        let pressed = row_text(
            cx,
            name,
            link_key,
            Rect::new(x, rect.y, right - x, rect.h),
            artist,
            size,
            alpha,
            target,
        );
        opened = opened.or(pressed);
        x += cx.text_width(artist, size);
    }
    opened
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
    let hit = Rect::new(
        rect.x,
        rect.y,
        cx.text_width(text, size).min(rect.w),
        rect.h,
    );
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
        theme::muted(),
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

/// A section heading on a page, in the text colour, as the page words it.
fn section_heading(cx: &mut Cx, rect: Rect, text: &str) {
    cx.text_fit(rect, text, 16.0, theme::fg(), Align::Left);
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
    let columns = tile_columns(view.w - 44.0);
    // The songs are rows of this artist's own page, so they show their album, not the artist.
    let songs: Vec<Row> = page
        .songs
        .iter()
        .map(|hit| {
            let mut row = Row::from(hit, lengths);
            for (_, id) in &mut row.artists {
                if *id == Some(page.browse_id.as_str()) {
                    *id = None;
                }
            }
            row
        })
        .collect();

    let mut content_h = PAGE_TOP + PICTURE + 40.0;
    if !songs.is_empty() {
        content_h += SECTION_H + songs.len() as f32 * theme::ROW_H + 24.0;
    }
    for shelf in &page.shelves {
        let rows = shelf.albums.len().div_ceil(columns) as f32;
        content_h += SECTION_H + rows * (TILE + TILE_TEXT_H + TILE_GAP) + 24.0;
    }
    let body = cx.ui.scroll_begin(scroll_id, view, content_h);
    let left = body.x + 8.0;

    let picture = Rect::new(left, body.y + PAGE_TOP, PICTURE, PICTURE);
    cx.cover(page.thumbnail.as_deref(), picture, PICTURE * 0.5, 1.0);
    let name_x = picture.x + PICTURE + 28.0;
    let name_w = (body.x + body.w - 36.0 - name_x).max(40.0);
    cx.text_fit(
        Rect::new(name_x, picture.y + PICTURE * 0.5 - 20.0, name_w, 40.0),
        &page.name,
        30.0,
        theme::fg(),
        Align::Left,
    );

    // Saving the artist keeps their page on this device (there is no audio to download).
    let saved = cx.saved.artists.contains(&page.browse_id);
    let button = (
        name_x + SAVE_ARTIST_R,
        picture.y + PICTURE - SAVE_ARTIST_R - 2.0,
    );
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
        Rect::new(
            button.0 + SAVE_ARTIST_R + 12.0,
            button.1 - 10.0,
            160.0,
            20.0,
        ),
        if saved { "Saved" } else { "Save" },
        13.0,
        if saved { theme::fg() } else { theme::muted() },
        Align::Left,
    );

    let mut y = body.y + PAGE_TOP + PICTURE + 40.0;
    if !songs.is_empty() {
        section_heading(
            cx,
            Rect::new(left, y, body.w - 16.0, 28.0),
            &page.songs_title,
        );
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
            true,
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
            album_tile(
                cx,
                view,
                actions,
                (x, ty),
                card,
                9000 + (shelf_index * 100 + k) as u32,
            );
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
        (
            card.title.as_str(),
            card.meta.as_str(),
            card.thumbnail.as_deref(),
        ),
        false,
        target,
        key,
    );
}

/// Draws a rounded card background, using frosted-glass translucency (matching the sidebar and
/// player bar) when a custom background is active.
fn draw_card_surface(
    cx: &mut Cx,
    rect: Rect,
    radius: f32,
    opaque_rest: Color,
    opaque_hover: Color,
    glass_hover: Color,
    hv: f32,
) {
    if cx.gpu.background.is_some() {
        cx.draw().backdrop_rounded(rect, Color::WHITE, radius);
        cx.draw().round_rect(
            rect,
            radius,
            fade(mix(theme::panel(), glass_hover, hv), 0.62 + 0.10 * hv),
        );
    } else {
        cx.draw()
            .round_rect(rect, radius, mix(opaque_rest, opaque_hover, hv));
    }
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
    let pill = Rect::new(x - 10.0, y - 10.0, TILE + 20.0, TILE + TILE_TEXT_H + 10.0);
    let pointer_in_view = cx.pointer_in(view);
    let hovered = pointer_in_view && cx.over_clickable(pill);
    let hv = cx.anims.toggle("tile", key, hovered, rate::SNAP);
    draw_card_surface(
        cx,
        pill,
        12.0,
        theme::bg(),
        theme::hover(),
        theme::panel_hover(),
        hv,
    );
    let radius = if round { TILE * 0.5 } else { 10.0 };
    cx.cover(thumbnail, Rect::new(x, y, TILE, TILE), radius, 1.0);
    cx.text_fit(
        Rect::new(x, y + TILE + 6.0, TILE, 20.0),
        title,
        14.5,
        theme::fg(),
        Align::Left,
    );
    cx.text_fit(
        Rect::new(x, y + TILE + 26.0, TILE, 18.0),
        meta,
        12.5,
        theme::muted(),
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
    let rows: Vec<Row> = state
        .results
        .iter()
        .map(|hit| Row::from(hit, &state.lengths))
        .collect();
    let columns = tile_columns(view.w - 44.0);
    let grid_h = |count: usize| count.div_ceil(columns) as f32 * (TILE + TILE_TEXT_H + TILE_GAP);
    let tab = state.results_tab;

    // The chosen tab's part: its height, or a line of 'nothing here' when it is empty.
    let part_h = match tab {
        ResultsTab::Songs if !rows.is_empty() => rows.len() as f32 * theme::ROW_H + 24.0,
        ResultsTab::Artists if !state.result_artists.is_empty() => {
            grid_h(state.result_artists.len()) + 24.0
        }
        ResultsTab::Albums if !state.result_albums.is_empty() => {
            grid_h(state.result_albums.len()) + 24.0
        }
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
    let current = ResultsTab::ALL
        .iter()
        .position(|kind| *kind == tab)
        .unwrap_or(0);
    if let Some(index) = tab_pills(cx, &labels, current, left, y, &RESULT_PILLS) {
        actions.push(Action::ResultsTab(ResultsTab::ALL[index]));
    }
    y += TAB_ROW_H;

    match tab {
        ResultsTab::Songs if !rows.is_empty() => {
            track_rows(
                cx,
                track,
                &rows,
                view,
                body,
                y,
                0,
                age,
                true,
                false,
                Action::Play,
                0,
                actions,
            );
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
                    (
                        artist.name.as_str(),
                        artist.meta.as_str(),
                        artist.thumbnail.as_deref(),
                    ),
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
            cx.text(
                Rect::new(left, y + 12.0, body.w - 16.0, 24.0),
                hint,
                14.0,
                theme::muted(),
                Align::Left,
            );
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
    // no dim and no bars, as if it were not playing at all. `audible` is latched by `State::tick`,
    // so this is not re-decided from a momentary meter reading here.
    let paused = track.is_none_or(|track| track.paused);
    let levels = track.map_or([0.0; 3], |track| track.levels);
    let audible = playing && !paused && track.is_some_and(|track| track.audible);
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
        draw_spinner(cx, center, 12.0, fade(theme::fg(), a));
    } else if audible {
        icons::equalizer(
            cx.draw(),
            center,
            16.0,
            mix(theme::bg(), theme::fg(), a),
            levels,
            true,
        );
    } else if playing {
        // Playing but silent (or paused): nothing over the cover.
    } else if hv > 0.5 {
        // Opaque and fixed size; shown once hover is well under way, so no dark ghost lingers.
        cx.icon(Icon::Play, center, 18.0, theme::fg());
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
    draw_card_surface(
        cx,
        pill,
        14.0,
        theme::panel(),
        theme::select(),
        theme::select(),
        hv,
    );

    let (title, meta, thumbnail, round, target): (
        &str,
        String,
        Option<&str>,
        bool,
        Option<Detail>,
    ) = match top {
        // A song's artists and album are drawn below as links of their own, not as one string.
        TopResult::Song(hit) => (
            hit.title.as_str(),
            String::new(),
            hit.thumbnail.as_deref(),
            false,
            None,
        ),
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
        theme::muted(),
        Align::Left,
    );
    cx.text_fit(
        Rect::new(text_x, pill.y + 56.0, text_w, 36.0),
        title,
        26.0,
        theme::fg(),
        Align::Left,
    );
    // The meta line. A song's artist credits are links of their own, as a row's are, followed by
    // its album; an artist's or an album's meta is plain.
    let meta_rect = Rect::new(text_x, pill.y + 98.0, text_w, 20.0);
    let opened = if let TopResult::Song(hit) = top {
        let credits: Vec<(&str, Option<&str>)> = hit.artist_links().collect();
        let credits_w = if credits.is_empty() {
            0.0
        } else {
            cx.text_width(&hit.artists.join(ARTIST_SEP), 13.0)
                .min(text_w)
        };
        let mut opened = artist_text(
            cx,
            "link_top_artist",
            0,
            Rect::new(text_x, meta_rect.y, credits_w, meta_rect.h),
            &credits,
            13.0,
            1.0,
        );
        let mut x = text_x + credits_w;
        if let Some(album) = hit.album.as_deref().filter(|album| !album.is_empty()) {
            let sep_w = if credits.is_empty() {
                0.0
            } else {
                cx.text_width(ROW_SEP, 13.0)
            };
            if sep_w > 0.0 {
                cx.text_fit(
                    Rect::new(x, meta_rect.y, sep_w, meta_rect.h),
                    ROW_SEP,
                    13.0,
                    theme::muted(),
                    Align::Left,
                );
            }
            x += sep_w;
            let room = Rect::new(
                x,
                meta_rect.y,
                (meta_rect.x + meta_rect.w - x).max(0.0),
                meta_rect.h,
            );
            match hit
                .album_id
                .as_deref()
                .map(|id| Detail::Album(id.to_owned()))
            {
                Some(target) => {
                    opened = opened.or(row_text(
                        cx,
                        "link_top_album",
                        0,
                        room,
                        album,
                        13.0,
                        1.0,
                        Some(target),
                    ));
                }
                None => cx.text_fit(room, album, 13.0, theme::muted(), Align::Left),
            }
        }
        opened
    } else {
        cx.text_fit(meta_rect, &meta, 13.0, theme::muted(), Align::Left);
        None
    };
    if let Some(detail) = opened {
        actions.push(Action::Open(detail));
    } else if pointer_in_view && cx.clicked(pill) {
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
/// rows repeat neither the album's name nor its cover, since the page already shows both.
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
    let rows_top_off = PAGE_TOP + COVER_BIG + 32.0;
    let content_h = rows_top_off + tracks.len() as f32 * theme::ROW_H + 24.0;
    let body = cx.ui.scroll_begin(scroll_id, view, content_h);
    let left = body.x + 8.0;

    let cover = Rect::new(left, body.y + PAGE_TOP, COVER_BIG, COVER_BIG);
    cx.cover(page.thumbnail.as_deref(), cover, 12.0, 1.0);

    // Title, artist and meta, standing centred on the height of the cover.
    let text_x = cover.x + COVER_BIG + 28.0;
    let text_w = (body.x + body.w - 36.0 - text_x).max(40.0);
    let block_y = cover.y + (COVER_BIG - 92.0) * 0.5;
    cx.text_fit(
        Rect::new(text_x, block_y, text_w, 34.0),
        &page.title,
        26.0,
        theme::fg(),
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
        theme::muted(),
        Align::Left,
    );
    if let Some(target) = artist {
        actions.push(Action::Open(target));
    }

    // Saving the album saves every song on it.
    let saved = cx.saved.albums.contains(&page.browse_id);
    let button = (
        text_x + SAVE_HEAD_R,
        cover.y + COVER_BIG - SAVE_HEAD_R - 2.0,
    );
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
        if saved { theme::fg() } else { theme::muted() },
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

    // The glass and the tint are the search field's (matching the sidebar and player bar without
    // accent), so the two read as one bar. Everything that carries colour is faded by `t` so the
    // button arrives and leaves without a visible edge.
    cx.draw()
        .backdrop_rounded(drawn, fade(Color::rgb(0xff, 0xff, 0xff), t), drawn.h * 0.5);
    cx.draw().round_rect(
        drawn,
        drawn.h * 0.5,
        fade(
            mix(theme::panel(), theme::panel_press(), lift),
            t * (0.62 + 0.12 * lift),
        ),
    );
    cx.icon(
        Icon::ArrowLeft,
        (cxm, cym),
        19.0 * grow,
        fade(mix(theme::muted(), theme::fg(), lift), t),
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
    let hit = Rect::new(
        center.0 - radius,
        center.1 - radius,
        radius * 2.0,
        radius * 2.0,
    );
    let hover = enabled && cx.over_clickable(hit);
    let hv = cx.anims.toggle("btn_hv", key, hover, rate::SNAP);
    let live = cx.anims.toggle("btn_on", key, enabled, rate::QUICK);
    cx.icon_state(
        icons,
        current,
        center,
        size,
        mix(
            theme::panel(),
            mix(theme::muted(), theme::fg(), hv),
            0.35 + 0.65 * live,
        ),
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
    let rect = Rect::new(
        center.0 - radius,
        center.1 - radius,
        radius * 2.0,
        radius * 2.0,
    );
    let angle = cx.spin;
    cx.draw()
        .ring_arc(rect, radius, SPINNER_THICKNESS, angle, SPINNER_SWEEP, color);
}

fn draw_bar(
    cx: &mut Cx,
    state: &mut State,
    track: Option<&TrackView>,
    bar: Rect,
    actions: &mut Vec<Action>,
) {
    if cx.gpu.background.is_some() {
        cx.draw().backdrop_rect(bar, Color::WHITE);
        cx.draw()
            .rect(bar, fade(theme::panel(), ui_panel_alpha(state)));
    } else {
        cx.draw().rect(bar, theme::panel());
    }
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
        Rect::new(
            tile_c.0 - tile_size * 0.5,
            tile_c.1 - tile_size * 0.5,
            tile_size,
            tile_size,
        ),
        8.0,
        1.0,
    );

    let text_x = tile_left + 56.0 + text_gap;
    let text_w = left_w - 56.0 - 16.0;
    // The bar shows only what is playing. App status ("Loading…", "Searching…") never goes here.
    let (title, sub, title_color) = match track {
        Some(track) => (track.title.as_str(), track.artist.as_str(), theme::fg()),
        None => ("Nothing playing", "", theme::muted()),
    };
    cx.clip(Rect::new(
        tile_left + 56.0 + text_gap,
        bar.y,
        text_w + 4.0,
        bar.h,
    ));
    // The title names the track's album: a press on it opens the album. When the song row did not
    // carry the album's link, ask for it so the link appears once it is found.
    if let Some(hit) = track.and_then(|track| track.hit.as_ref()) {
        cx.want_meta(hit);
    }
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
    // Each artist names their page: a press on one opens that artist.
    let artists: Vec<(&str, Option<&str>)> = match track {
        Some(track) => match track.hit.as_ref().filter(|hit| !hit.artists.is_empty()) {
            Some(hit) => hit
                .artist_links()
                .enumerate()
                .map(|(i, (name, id))| {
                    (
                        name,
                        id.or(if i == 0 {
                            track.artist_id.as_deref()
                        } else {
                            None
                        }),
                    )
                })
                .collect(),
            None => vec![(sub, track.artist_id.as_deref())],
        },
        None => Vec::new(),
    };
    if let Some(target) = artist_text(
        cx,
        "bar_artist",
        0,
        Rect::new(text_x, bar.y + 35.0, text_w, 18.0),
        &artists,
        12.5,
        1.0,
    ) {
        actions.push(Action::Open(target));
    }
    cx.unclip();

    // A right press on the track in the bar opens its menu, as a right press on a row does.
    if let Some(hit) = track.and_then(|track| track.hit.as_ref()) {
        if cx.right_clicked(Rect::new(0.0, bar.y, left_w + 24.0, bar.h)) {
            actions.push(Action::OpenMenu(TrackMenu {
                hit: hit.clone(),
                at: cx.pointer.pos,
            }));
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
    if icon_button(
        cx,
        1,
        (mid_x - 52.0, row),
        18.0,
        has_back,
        &[Icon::SkipBack],
        0,
        26.0,
    ) {
        actions.push(Action::Back);
    }
    // Next is only live when something is queued.
    let has_next = !state.playlist.queue.is_empty();
    if icon_button(
        cx,
        2,
        (mid_x + 52.0, row),
        18.0,
        has_next,
        &[Icon::SkipForward],
        0,
        26.0,
    ) {
        actions.push(Action::Next);
    }

    // The disc stays centred on its fixed hit target and is never resized; hover only tints it.
    let c = (cx.snap(mid_x - 20.0) + 20.0, cx.snap(y - 20.0) + 20.0);
    let play_rect = Rect::new(c.0 - 20.0, c.1 - 20.0, 40.0, 40.0);
    let hover = playable && cx.over_clickable(play_rect);
    let hover_t = cx
        .anims
        .spring("play_hover", 0, if hover { 1.0 } else { 0.0 }, rate::SNAP)
        .clamp(0.0, 1.0);
    let live = cx.anims.toggle("play_on", 0, has, rate::QUICK);
    let visual = Rect::new(c.0 - 20.0, c.1 - 20.0, 40.0, 40.0);
    // Opaque throughout: the disc's brightness and the glyph's colour carry the state, never alpha.
    // Hover changes the colour in both states, in opposite directions so each stays visible: an idle
    // disc (dark panel tone) lifts toward white, a live one (already white) eases toward grey.
    let disc = mix(theme::panel(), theme::accent(), 0.22 + 0.78 * live);
    let idle_hover = mix(disc, theme::fg(), hover_t * 0.24);
    let live_hover = mix(disc, theme::muted(), hover_t * 0.2);
    cx.draw()
        .round_rect(visual, 20.0, mix(idle_hover, live_hover, live));
    // Play and pause swap by scale: the outgoing glyph shrinks to nothing while the incoming one
    // grows from nothing with a slight spring overshoot. Both are opaque, so they never blend.
    let showing_pause = playable && !paused;
    let ink = mix(theme::muted(), theme::on_accent(), live);
    // Glyph and spinner swap with no motion in either direction: the spinner appears and goes at
    // once. Play and pause still swap by scale, but only when neither is a spinner.
    let glyph_targets = [
        if showing_pause { 0.0 } else { 1.0 },
        if showing_pause { 1.0 } else { 0.0 },
    ];
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
    let times = if has { theme::muted() } else { theme::faint() };
    cx.text(
        Rect::new(seek_x, seek_y - 10.0, 40.0, 20.0),
        &clock(position),
        12.0,
        times,
        Align::Right,
    );
    cx.text(
        Rect::new(seek_x + seek_w - 40.0, seek_y - 10.0, 40.0, 20.0),
        &if duration > 0.0 {
            clock(duration)
        } else {
            "0:00".to_owned()
        },
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
    let region = Rect::new(
        region_x,
        icon_hit.y - 4.0,
        icon_hit.x + icon_hit.w - region_x,
        48.0,
    );
    let near = cx.pointer_in(if state.volume_open { region } else { icon_hit });
    let wanted = near || cx.drag == Some(SliderId::Volume);
    state.volume_open = wanted;
    let stiffness = if wanted {
        VOLUME_OPEN_RATE
    } else {
        VOLUME_CLOSE_RATE
    };
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

    // ---- Lyrics: a microphone just left of the volume icon. It stays put and is absent while the
    // volume slider is showing. It is lit while the lyric menu is open.
    let mic_c = (icon_c.0 - 44.0, icon_c.1);
    let mic_hit = Rect::new(mic_c.0 - 20.0, mic_c.1 - 20.0, 40.0, 40.0);
    // Greyed out and unclickable unless the current song has lyrics to show. An open page can always
    // be closed.
    let mic_enabled = state.lyrics_open || has_lyrics(state, track);
    // Hidden, not just dimmed, whenever the volume slider has any presence.
    let mic_active = mic_enabled && alpha <= 0.002;
    let mic_hover = mic_active && cx.over_clickable(mic_hit);
    let mic_hv = cx.anims.toggle("mic_hv", 0, mic_hover, rate::SNAP);
    let lit = cx
        .anims
        .toggle("mic_lit", 0, state.lyrics_open, rate::QUICK);
    let mic_colour = if mic_enabled {
        mix(theme::muted(), theme::fg(), mic_hv.max(lit))
    } else {
        theme::faint()
    };
    // Skip drawing it altogether while the slider is visible. Recolouring it toward the panel
    // leaves a visible glyph over a custom background because the bar is translucent there.
    if alpha <= 0.002 {
        cx.icon(Icon::Mic, mic_c, 22.0, mic_colour);
    }
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
        mix(theme::muted(), theme::fg(), hv),
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

// ---------------------------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------------------------

/// The tint over the picture. Kept subtle so the background's colours and frosted glass shine
/// through while keeping text over it crisp and readable.
const BACKGROUND_DIM: f32 = 0.22;

/// The chosen picture or video behind the whole window, aspect-filled (zoomed to fill with no empty
/// bars at any window size or aspect ratio).
fn draw_background(ui: &mut Ui, gpu: &Gpu, width: f32, height: f32) {
    let (Some(texture), (tw, th)) = (gpu.background.as_ref(), gpu.background_size) else {
        return;
    };
    if tw == 0 || th == 0 || width <= 0.0 || height <= 0.0 {
        return;
    }
    let scale = (width / tw as f32).max(height / th as f32);
    let vis_u = (width / (tw as f32 * scale)).clamp(0.0, 1.0);
    let vis_v = (height / (th as f32 * scale)).clamp(0.0, 1.0);
    let u0 = (1.0 - vis_u) * 0.5;
    let v0 = (1.0 - vis_v) * 0.5;
    let full = Rect::new(0.0, 0.0, width, height);
    ui.draw_list_mut().image_uv_rect(
        full,
        texture,
        Color::WHITE,
        0.0,
        (u0, v0, 1.0 - u0, 1.0 - v0),
    );
    ui.draw_list_mut()
        .rect(full, fade(theme::bg(), BACKGROUND_DIM));
}

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------

/// Quick-access preset dots inside the full-range colour picker (starts with Default's black,
/// grey, and off-white so they are always one click away).
const PICKER_PRESETS: [Color; 8] = [
    Color::rgb(0x00, 0x00, 0x00),
    Color::rgb(0x1a, 0x1a, 0x1a),
    Color::rgb(0x8a, 0x8a, 0x8a),
    Color::rgb(0xec, 0xec, 0xec),
    Color::rgb(0xff, 0x5d, 0x8f),
    Color::rgb(0x7a, 0xa2, 0xff),
    Color::rgb(0x22, 0xc5, 0x5e),
    Color::rgb(0xf9, 0x73, 0x16),
];

const SET_MAX_W: f32 = 700.0;
const THEME_ROW_H: f32 = 42.0;
const THEME_GRID_GAP: f32 = 8.0;
const THEME_PREVIEW_CAP: usize = 8;

fn theme_creator_box_height(width: f32, has_tbg: bool) -> f32 {
    // Wide layout places Interface and Lyrics groups side-by-side (3 rows each) + theme background row;
    // narrow layout stacks them (6 rows) + theme background row.
    let frost_h = if has_tbg { 72.0 } else { 0.0 };
    if width >= 600.0 {
        16.0 + 34.0 + 16.0 + 22.0 + 3.0 * 44.0 + 44.0 + frost_h + 12.0
    } else {
        16.0 + 34.0 + 16.0 + (22.0 + 3.0 * 44.0) * 2.0 + 14.0 + 44.0 + frost_h + 12.0
    }
}

const EQ_PRESETS: [(&str, [f32; 10]); 4] = [
    ("Flat", [0.0; 10]),
    ("Bass", [5.5, 4.5, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0]),
    (
        "Bright",
        [-1.0, -0.5, 0.0, 0.0, 0.5, 1.5, 3.0, 4.5, 5.0, 4.5],
    ),
    (
        "Vocal",
        [-2.0, -1.5, -0.5, 1.5, 3.5, 4.0, 3.0, 1.5, 0.5, 0.0],
    ),
];

fn filtered_theme_indices(state: &State) -> Vec<usize> {
    let q = state.settings.theme_query.trim().to_ascii_lowercase();
    theme_order(&state.themes)
        .into_iter()
        .filter(|&i| q.is_empty() || state.themes[i].name.to_ascii_lowercase().contains(&q))
        .collect()
}

fn surface_color(cx: &Cx) -> Color {
    if cx.gpu.background.is_some() {
        fade(theme::panel(), 0.54)
    } else {
        theme::panel()
    }
}

/// The Settings page's category tabs (Appearance, Audio, General).
const SETTINGS_PILLS: PillKeys = PillKeys {
    hover: "settings_tab_hv",
    on: "settings_tab_on",
    glide_x: "settings_tab_glide_x",
    glide_w: "settings_tab_glide_w",
};

fn settings_page(cx: &mut Cx, state: &mut State, view: Rect, actions: &mut Vec<Action>) {
    let left = view.x + 8.0;
    let width = (view.w - 32.0).clamp(280.0, SET_MAX_W);
    let top = PAGE_TOP;
    let body = cx
        .ui
        .scroll_begin("settings", view, settings_height(state, width).max(view.h));
    let mut y = body.y + top;

    cx.text(
        Rect::new(left, y, width, 32.0),
        "Settings",
        24.0,
        theme::fg(),
        Align::Left,
    );
    y += 42.0;

    let labels = SettingsTab::ALL.map(|tab| (tab.label(), tab.icon()));
    if let Some(index) = tab_pills(
        cx,
        &labels,
        state.settings.tab.index(),
        left,
        y,
        &SETTINGS_PILLS,
    ) {
        state.settings.tab = SettingsTab::ALL[index];
        state.settings.color_picker = None;
    }
    y += COLLECTION_TAB_H + 22.0;

    match state.settings.tab {
        SettingsTab::Appearance => {
            // ---- Global Background.
            let global_bg = state.prefs.background();
            let has_bg = global_bg.is_some();
            let bg_enabled = state.prefs.background_enabled();
            section_heading(
                cx,
                Rect::new(left, y, width - 64.0, 30.0),
                "Global Background",
            );
            if switch(
                cx,
                1,
                Rect::new(left + width - 46.0, y + 3.0, 46.0, 24.0),
                bg_enabled,
            ) {
                actions.push(Action::ToggleGlobalBackground(!bg_enabled));
            }
            y += 30.0;
            cx.text_fit(
                Rect::new(left, y, width, 18.0),
                "Supports image and video (loops). Overrides theme backgrounds when on.",
                13.0,
                theme::muted(),
                Align::Left,
            );
            y += 24.0;

            let card_h = if has_bg { 128.0 } else { 56.0 };
            let card = Rect::new(left, y, width, card_h);
            let surf = surface_color(cx);
            cx.draw().round_rect(card, 12.0, surf);

            let pad = 10.0;
            let right_reserve = if has_bg { 48.0 } else { 14.0 };
            let pick_row = Rect::new(
                card.x + pad,
                card.y + pad,
                card.w - pad - right_reserve,
                36.0,
            );
            let pick_hover = cx.over_clickable(pick_row);
            let pick_hv = cx.anims.toggle("bg_pick", 0, pick_hover, rate::SNAP);
            if pick_hv > 0.01 {
                cx.draw()
                    .round_rect(pick_row, 8.0, fade(theme::panel_hover(), pick_hv));
            }
            cx.icon(
                Icon::Folder,
                (pick_row.x + 16.0, pick_row.y + 18.0),
                15.5,
                mix(theme::muted(), theme::fg(), pick_hv.max(0.45)),
            );
            if cx.clicked(pick_row) {
                actions.push(Action::PickBackground);
            }

            let text_rect = Rect::new(pick_row.x + 34.0, pick_row.y, pick_row.w - 40.0, 36.0);
            if let Some(err) = state
                .background_prep_error
                .as_deref()
                .or(state.background.error.as_deref())
            {
                cx.text_fit(text_rect, err, 12.5, theme::CLOSE_RED, Align::Left);
            } else if let Some(path) = global_bg {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("Background");
                cx.text_fit(
                    text_rect,
                    name,
                    13.0,
                    if bg_enabled {
                        theme::fg()
                    } else {
                        theme::muted()
                    },
                    Align::Left,
                );
            } else {
                cx.text_fit(
                    text_rect,
                    "No background selected",
                    13.0,
                    mix(theme::muted(), theme::fg(), pick_hv),
                    Align::Left,
                );
            }

            if has_bg {
                let trash_rect = Rect::new(card.x + card.w - 42.0, pick_row.y + 2.0, 32.0, 32.0);
                let trash_hover = cx.over_clickable(trash_rect);
                let trash_hv = cx.anims.toggle("bg_trash", 0, trash_hover, rate::SNAP);
                if trash_hv > 0.01 {
                    cx.draw()
                        .round_rect(trash_rect, 8.0, fade(theme::panel_hover(), trash_hv));
                }
                cx.icon(
                    Icon::Trash2,
                    (trash_rect.x + 16.0, trash_rect.y + 16.0),
                    15.0,
                    mix(theme::muted(), theme::CLOSE_RED, trash_hv),
                );
                if cx.clicked(trash_rect) {
                    actions.push(Action::ConfirmDialog(Some(ConfirmDialog::RemoveBackground)));
                }

                let fy = pick_row.y + pick_row.h + 10.0;
                cx.text(
                    Rect::new(card.x + 14.0, fy, 46.0, 28.0),
                    "Frost",
                    13.0,
                    theme::muted(),
                    Align::Left,
                );
                let mut strength = state.background.strength;
                if cx.slider(
                    SliderId::Blur,
                    Rect::new(card.x + 64.0, fy + 6.0, (card.w - 128.0).max(100.0), 16.0),
                    &mut strength,
                    true,
                    1.0,
                ) {
                    actions.push(Action::BlurStrength(strength));
                }
                cx.text(
                    Rect::new(card.x + card.w - 56.0, fy, 42.0, 28.0),
                    &format!("{}%", (strength * 100.0).round() as i32),
                    12.0,
                    theme::muted(),
                    Align::Right,
                );

                let ty = fy + 34.0;
                cx.text(
                    Rect::new(card.x + 14.0, ty, 88.0, 28.0),
                    "Transparency",
                    13.0,
                    theme::muted(),
                    Align::Left,
                );
                let mut transparency = state.prefs.background_transparency();
                if cx.slider(
                    SliderId::Transparency,
                    Rect::new(card.x + 106.0, ty + 6.0, (card.w - 170.0).max(100.0), 16.0),
                    &mut transparency,
                    true,
                    1.0,
                ) {
                    actions.push(Action::BackgroundTransparency(transparency));
                }
                cx.text(
                    Rect::new(card.x + card.w - 56.0, ty, 42.0, 28.0),
                    &format!("{}%", (transparency * 100.0).round() as i32),
                    12.0,
                    theme::muted(),
                    Align::Right,
                );
            }
            y += card_h + 28.0;

            // ---- Themes.
            section_heading(cx, Rect::new(left, y, width - 124.0, 30.0), "Themes");
            let creator_label = if state.settings.creator_open {
                "Cancel"
            } else {
                "New theme"
            };
            let creator_icon = if state.settings.creator_open {
                Icon::X
            } else {
                Icon::Plus
            };
            if pill_button(
                cx,
                Rect::new(left + width - 112.0, y, 112.0, 30.0),
                creator_label,
                Some(creator_icon),
                true,
                10,
            ) {
                if state.settings.creator_open {
                    state.settings.creator_open = false;
                    state.settings.editing_theme = None;
                    state.settings.color_picker = None;
                } else {
                    state.settings.editing_theme = None;
                    state.settings.name.clear();
                    state.background_editor_generation =
                        state.background_editor_generation.wrapping_add(1);
                    state.settings.creator_open = true;
                }
            }
            y += 38.0;

            let show_search = state.themes.len() > 5;
            if show_search {
                let mut s_style = TextFieldStyle::new(cx.font);
                s_style.size = 13.0;
                s_style.text = theme::fg();
                s_style.normal = surface_color(cx);
                s_style.hover = mix(surface_color(cx), theme::panel_hover(), 0.65);
                s_style.accent = theme::accent();
                s_style.selection_alpha = 56;
                s_style.corner_radius = 10.0;
                s_style.border_width = 0.0;
                s_style.caret = theme::fg();
                s_style.caret_width = 1.5;
                s_style.caret_inset = 9.0;
                s_style.selection_inset = 9.0;
                s_style.text_inset = 34.0;
                let filter_rect = Rect::new(left, y, width, 34.0);
                cx.ui.text_field(
                    "theme_filter",
                    filter_rect,
                    &mut state.settings.theme_query,
                    s_style,
                );
                cx.icon(
                    Icon::Search,
                    (filter_rect.x + 17.0, filter_rect.y + 17.0),
                    14.0,
                    theme::muted(),
                );
                if state.settings.theme_query.is_empty() {
                    cx.text(
                        Rect::new(
                            filter_rect.x + 34.0,
                            filter_rect.y,
                            filter_rect.w - 46.0,
                            filter_rect.h,
                        ),
                        "Filter themes",
                        12.5,
                        theme::muted(),
                        Align::Left,
                    );
                }
                y += 44.0;
            }

            let raw_active = state.prefs.active_theme().map(str::to_owned);
            let active = match raw_active.as_deref() {
                Some(name) if state.themes.iter().any(|t| t.name == name) => Some(name.to_owned()),
                _ => None,
            };

            let filtered = filtered_theme_indices(state);
            let searching = !state.settings.theme_query.trim().is_empty();
            let total_entries = 1 + filtered.len();
            let capped =
                !searching && !state.settings.show_all_themes && total_entries > THEME_PREVIEW_CAP;
            let visible_entries = if capped {
                THEME_PREVIEW_CAP
            } else {
                total_entries
            };
            let cols = if width >= 460.0 { 2 } else { 1 };
            let grid_rows = visible_entries.div_ceil(cols);
            let cell_w = (width - THEME_GRID_GAP * (cols.saturating_sub(1) as f32)) / cols as f32;
            let place = |slot: usize, base_y: f32| -> Rect {
                let col = slot % cols;
                let row = slot / cols;
                Rect::new(
                    left + col as f32 * (cell_w + THEME_GRID_GAP),
                    base_y + row as f32 * (THEME_ROW_H + THEME_GRID_GAP),
                    cell_w,
                    THEME_ROW_H,
                )
            };

            let def_p = theme::Palette::default();
            if theme_row(
                cx,
                place(0, y),
                "Default",
                (def_p.bg, def_p.panel, def_p.accent),
                active.is_none(),
                None,
                false,
                0,
            ) == ThemePress::Apply
            {
                actions.push(Action::SetTheme(None));
            }

            let visible_saved = visible_entries.saturating_sub(1);
            for (offset, &index) in filtered.iter().take(visible_saved).enumerate() {
                let slot = offset + 1;
                let (name, colors, favorite) = {
                    let t = &state.themes[index];
                    let pal = theme::Palette::from_colors(
                        rgb_color(t.primary),
                        rgb_color(t.secondary),
                        rgb_color(t.accent),
                    );
                    (t.name.clone(), (pal.bg, pal.panel, pal.accent), t.favorite)
                };
                let chosen = active.as_deref() == Some(name.as_str());
                match theme_row(
                    cx,
                    place(slot, y),
                    &name,
                    colors,
                    chosen,
                    Some(favorite),
                    true,
                    index as u32 + 1,
                ) {
                    ThemePress::Apply => actions.push(Action::SetTheme(Some(name))),
                    ThemePress::Favorite => actions.push(Action::FavoriteTheme(name)),
                    ThemePress::Edit => actions.push(Action::EditTheme(name)),
                    ThemePress::Delete => actions.push(Action::ConfirmDialog(Some(
                        ConfirmDialog::DeleteTheme(name),
                    ))),
                    ThemePress::None => {}
                }
            }
            y += grid_rows as f32 * (THEME_ROW_H + THEME_GRID_GAP);

            if !searching && total_entries > THEME_PREVIEW_CAP {
                let label = if state.settings.show_all_themes {
                    "Show fewer".to_owned()
                } else {
                    format!("Show all ({total_entries})")
                };
                if text_button(cx, Rect::new(left, y, 120.0, 26.0), &label, false) {
                    state.settings.show_all_themes = !state.settings.show_all_themes;
                }
                y += 30.0;
            }

            if state.settings.creator_open {
                y += 8.0;
                draw_theme_builder(cx, state, left, y, width, actions);
            }
        }
        SettingsTab::Audio => {
            // ---- Equalizer.
            section_heading(cx, Rect::new(left, y, width, 30.0), "Equalizer");
            y += 36.0;

            let mut px = left;
            let reset_w = 88.0;
            for (idx, (label, preset)) in EQ_PRESETS.iter().enumerate() {
                let is_active = state
                    .eq
                    .iter()
                    .zip(preset.iter())
                    .all(|(a, b)| (a - b).abs() < 0.25);
                let pw = (label.len() as f32 * 7.2 + 24.0).max(52.0);
                if px + pw <= left + width - reset_w - 8.0 {
                    let rect = Rect::new(px, y, pw, 28.0);
                    let hover = cx.over_clickable(rect);
                    let hv = cx.anims.toggle("eq_preset", idx as u32, hover, rate::SNAP);
                    let bg = if is_active {
                        theme::accent()
                    } else {
                        mix(surface_color(cx), theme::panel_hover(), hv)
                    };
                    let fg = if is_active {
                        theme::on_accent()
                    } else {
                        mix(theme::muted(), theme::fg(), hv)
                    };
                    cx.draw().round_rect(rect, 14.0, bg);
                    cx.text(rect, label, 12.0, fg, Align::Center);
                    if cx.clicked(rect) {
                        actions.push(Action::SetEq(*preset));
                    }
                    px += pw + 8.0;
                }
            }
            if pill_button(
                cx,
                Rect::new(left + width - reset_w, y - 1.0, reset_w, 30.0),
                "Reset",
                Some(Icon::Reset),
                true,
                12,
            ) {
                actions.push(Action::ResetEq);
            }
            y += 38.0;

            let deck = Rect::new(left, y, width, 162.0);
            let surf = surface_color(cx);
            cx.draw().round_rect(deck, 12.0, surf);
            let bands = crate::prefs::EQ_BANDS.len();
            let pad_x = 14.0;
            let step_w = (deck.w - pad_x * 2.0) / bands as f32;
            let fader_top = deck.y + 30.0;
            let fader_h = 98.0;

            for index in 0..bands {
                let cx_band = deck.x + pad_x + step_w * (index as f32 + 0.5);
                let gain = state.eq[index];
                let db_text = if gain.abs() < 0.25 {
                    "0".to_owned()
                } else {
                    format!("{gain:+.1}")
                };
                cx.text(
                    Rect::new(cx_band - step_w * 0.5, deck.y + 10.0, step_w, 16.0),
                    &db_text,
                    10.5,
                    if gain.abs() >= 0.25 {
                        theme::accent()
                    } else {
                        theme::faint()
                    },
                    Align::Center,
                );

                let fader_rect = Rect::new(cx_band - 12.0, fader_top, 24.0, fader_h);
                let mut value = (gain + EQ_RANGE_DB) / (2.0 * EQ_RANGE_DB);
                if cx.vertical_slider(SliderId::Eq(index), fader_rect, &mut value, true) {
                    let mut gains = state.eq;
                    let db = value * 2.0 * EQ_RANGE_DB - EQ_RANGE_DB;
                    gains[index] = (db * 2.0).round() / 2.0;
                    actions.push(Action::SetEq(gains));
                }

                let label = crate::prefs::eq_band_label(index);
                cx.text(
                    Rect::new(
                        cx_band - step_w * 0.5,
                        fader_top + fader_h + 8.0,
                        step_w,
                        16.0,
                    ),
                    &label,
                    11.0,
                    theme::muted(),
                    Align::Center,
                );
            }
        }
        SettingsTab::General => {
            // ---- Downloads.
            let enabled = state.prefs.downloads();
            section_heading(cx, Rect::new(left, y, width - 64.0, 26.0), "Downloads");
            if switch(
                cx,
                0,
                Rect::new(left + width - 46.0, y + 4.0, 46.0, 24.0),
                enabled,
            ) {
                if enabled {
                    let has_saved =
                        !state.library.ids().songs.is_empty() || state.library.album_count() > 0;
                    actions.push(if has_saved {
                        Action::ConfirmDialog(Some(ConfirmDialog::Downloads))
                    } else {
                        Action::SetDownloads(false)
                    });
                } else {
                    actions.push(Action::SetDownloads(true));
                }
            }
            y += 28.0;
            cx.text_fit(
                Rect::new(left, y, width - 64.0, 18.0),
                "Download songs in collections locally. Turning this off deletes local copies.",
                13.0,
                theme::muted(),
                Align::Left,
            );
        }
    }

    cx.ui.scroll_end();
}

/// The settings column's height, matching [`settings_page`].
fn settings_height(state: &State, width: f32) -> f32 {
    let top = PAGE_TOP;
    let header_h = 42.0 + COLLECTION_TAB_H + 22.0;
    let body_h = match state.settings.tab {
        SettingsTab::Appearance => {
            let has_bg = state.background.path.is_some();
            let bg_h = 30.0 + 24.0 + if has_bg { 94.0 } else { 56.0 } + 28.0;

            let show_search = state.themes.len() > 5;
            let total_entries = 1 + filtered_theme_indices(state).len();
            let searching = !state.settings.theme_query.trim().is_empty();
            let capped =
                !searching && !state.settings.show_all_themes && total_entries > THEME_PREVIEW_CAP;
            let visible_entries = if capped {
                THEME_PREVIEW_CAP
            } else {
                total_entries
            };
            let cols = if width >= 460.0 { 2 } else { 1 };
            let grid_rows = visible_entries.div_ceil(cols);
            let show_all_h = if !searching && total_entries > THEME_PREVIEW_CAP {
                30.0
            } else {
                0.0
            };
            let creator_h = if state.settings.creator_open {
                theme_creator_box_height(width, state.settings.theme_background.is_some()) + 16.0
            } else {
                0.0
            };
            let themes_h = 38.0
                + if show_search { 44.0 } else { 0.0 }
                + grid_rows as f32 * (THEME_ROW_H + THEME_GRID_GAP)
                + show_all_h
                + creator_h
                + 32.0;
            bg_h + themes_h
        }
        SettingsTab::Audio => 36.0 + 38.0 + 162.0 + 36.0,
        SettingsTab::General => 28.0 + 48.0,
    };

    top + header_h + body_h
}

/// What a press on a theme row asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThemePress {
    None,
    Apply,
    Favorite,
    Edit,
    Delete,
}

/// One compact borderless row in the theme grid: subtle 3-dot palette preview, name, and star/delete.
#[allow(clippy::too_many_arguments)]
fn theme_row(
    cx: &mut Cx,
    row: Rect,
    name: &str,
    swatches: (Color, Color, Color),
    active: bool,
    favorite: Option<bool>,
    deletable: bool,
    key: u32,
) -> ThemePress {
    let hover = cx.over_clickable(row);
    let hv = cx.anims.toggle("theme_row", key, hover, rate::SNAP);
    let has_bg = cx.gpu.background.is_some();
    let bg = if active {
        if has_bg {
            fade(theme::panel_select(), 0.78)
        } else {
            theme::panel_select()
        }
    } else {
        mix(surface_color(cx), theme::panel_hover(), hv)
    };
    cx.draw().round_rect(row, 10.0, bg);

    // 3 compact 10px colour dots instead of oversized bordered tiles.
    let dot_y = row.y + (row.h - 10.0) * 0.5;
    let mut dx = row.x + 12.0;
    for c in [swatches.0, swatches.1, swatches.2] {
        cx.draw()
            .round_rect(Rect::new(dx, dot_y, 10.0, 10.0), 5.0, c);
        dx += 13.0;
    }

    let text_x = dx + 6.0;
    let actions_w = match (favorite.is_some(), deletable) {
        (true, true) => 86.0,
        (true, false) | (false, true) => 34.0,
        (false, false) => 12.0,
    };
    let ink = mix(
        theme::muted(),
        theme::fg(),
        if active { 1.0 } else { hv.max(0.62) },
    );
    cx.text_fit(
        Rect::new(
            text_x,
            row.y,
            (row.x + row.w - actions_w - text_x).max(20.0),
            row.h,
        ),
        name,
        13.5,
        ink,
        Align::Left,
    );

    let mut ax = row.x + row.w - 32.0;
    if deletable {
        let bin = Rect::new(ax, row.y + row.h * 0.5 - 14.0, 28.0, 28.0);
        let b_hover = cx.over_clickable(bin);
        let bhv = cx.anims.toggle("theme_bin", key, b_hover, rate::SNAP);
        cx.icon(
            Icon::Trash2,
            (bin.x + 14.0, bin.y + 14.0),
            13.5,
            mix(theme::faint(), theme::CLOSE_RED, bhv),
        );
        if cx.clicked(bin) {
            return ThemePress::Delete;
        }
        ax -= 26.0;

        let edit = Rect::new(ax, row.y + row.h * 0.5 - 14.0, 28.0, 28.0);
        let e_hover = cx.over_clickable(edit);
        let ehv = cx.anims.toggle("theme_edit", key, e_hover, rate::SNAP);
        cx.icon(
            Icon::Settings,
            (edit.x + 14.0, edit.y + 14.0),
            13.5,
            mix(theme::faint(), theme::fg(), ehv),
        );
        if cx.clicked(edit) {
            return ThemePress::Edit;
        }
        ax -= 26.0;
    }
    if let Some(favorite) = favorite {
        let star = Rect::new(ax, row.y + row.h * 0.5 - 14.0, 28.0, 28.0);
        let s_hover = cx.over_clickable(star);
        let shv = cx.anims.toggle("theme_star", key, s_hover, rate::SNAP);
        let color = if favorite {
            Color::rgb(0xfb, 0xbf, 0x24)
        } else {
            mix(theme::faint(), theme::fg(), shv)
        };
        let icon = if favorite {
            Icon::StarFilled
        } else {
            Icon::Star
        };
        cx.icon(icon, (star.x + 14.0, star.y + 14.0), 14.5, color);
        if cx.clicked(star) {
            return ThemePress::Favorite;
        }
    }
    if cx.clicked(row) {
        ThemePress::Apply
    } else {
        ThemePress::None
    }
}

fn format_hex(c: Color) -> String {
    format!("#{:02X}{:02X}{:02X}", c.r, c.g, c.b)
}

fn parse_hex(input: &str) -> Option<Color> {
    let s = input.trim().strip_prefix('#').unwrap_or(input.trim());
    if s.len() != 6 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some(Color::rgb(r, g, b))
}

fn settings_slot_color(settings: &Settings, slot: ColorSlot) -> Color {
    match slot {
        ColorSlot::Primary => settings.primary,
        ColorSlot::Secondary => settings.secondary,
        ColorSlot::Accent => settings.accent,
        ColorSlot::LyricsFill => settings.lyrics_fill,
        ColorSlot::LyricsOutline => settings.lyrics_outline,
        ColorSlot::Visualizer => settings.visualizer_color,
    }
}

fn set_settings_slot_color(settings: &mut Settings, slot: ColorSlot, color: Color) {
    match slot {
        ColorSlot::Primary => settings.primary = color,
        ColorSlot::Secondary => settings.secondary = color,
        ColorSlot::Accent => settings.accent = color,
        ColorSlot::LyricsFill => settings.lyrics_fill = color,
        ColorSlot::LyricsOutline => settings.lyrics_outline = color,
        ColorSlot::Visualizer => settings.visualizer_color = color,
    }
}

fn open_color_picker(settings: &mut Settings, slot: ColorSlot, anchor: Rect) {
    let color = settings_slot_color(settings, slot);
    let toggle_off = settings
        .color_picker
        .as_ref()
        .is_some_and(|p| p.slot == slot);
    if toggle_off {
        settings.color_picker = None;
    } else {
        settings.color_picker = Some(ColorPickerState {
            slot,
            anchor,
            hsv: theme::rgb_to_hsv(color),
            hex: format_hex(color),
            last_color: color,
            just_opened: true,
        });
    }
}

/// The theme builder card: name + Reset/Save, and two clear groups (Interface | Lyrics)
/// side-by-side on wide windows or stacked on narrow ones.
fn draw_theme_builder(
    cx: &mut Cx,
    state: &mut State,
    left: f32,
    y: f32,
    width: f32,
    actions: &mut Vec<Action>,
) {
    let box_h = theme_creator_box_height(width, state.settings.theme_background.is_some());
    let box_rect = Rect::new(left, y, width, box_h);
    let surf = surface_color(cx);
    cx.draw().round_rect(box_rect, 12.0, surf);
    let pad = 16.0;
    let cx0 = box_rect.x + pad;
    let cw = box_rect.w - pad * 2.0;
    let mut cy = box_rect.y + pad;

    // ---- Row 1: Name field + Default reset + Save/Update.
    let reset_w = 88.0;
    let save_w = 98.0;
    let field_w = (cw - reset_w - save_w - 18.0).max(120.0);
    let mut style = TextFieldStyle::new(cx.font);
    style.size = 13.5;
    style.text = theme::fg();
    style.normal = theme::raised();
    style.hover = mix(theme::raised(), theme::panel_press(), 0.5);
    style.accent = theme::accent();
    style.selection_alpha = 56;
    style.corner_radius = 9.0;
    style.border_width = 0.0;
    style.caret = theme::fg();
    style.caret_width = 1.5;
    style.caret_inset = 10.0;
    style.selection_inset = 10.0;
    style.text_inset = 12.0;
    let field = Rect::new(cx0, cy, field_w, 34.0);
    cx.ui
        .text_field("theme_name", field, &mut state.settings.name, style);
    if state.settings.name.is_empty() {
        cx.text(
            Rect::new(field.x + 12.0, field.y, field.w - 24.0, field.h),
            "Theme name",
            13.0,
            theme::muted(),
            Align::Left,
        );
    }

    if pill_button(
        cx,
        Rect::new(field.x + field.w + 9.0, cy, reset_w, 34.0),
        "Default",
        Some(Icon::Reset),
        true,
        12,
    ) {
        state.settings.primary = Color::rgb(0x00, 0x00, 0x00);
        state.settings.secondary = Color::rgb(0xec, 0xec, 0xec);
        state.settings.accent = Color::rgb(0xec, 0xec, 0xec);
        state.settings.lyrics_fill = Color::rgb(0xec, 0xec, 0xec);
        state.settings.lyrics_outline = Color::rgb(0x00, 0x00, 0x00);
        state.settings.visualizer_mode = VisualizerColorRec::AlbumArt;
        state.settings.visualizer_color = Color::rgb(0xec, 0xec, 0xec);
        state.settings.color_picker = None;
    }

    let ready = !state.settings.name.trim().is_empty();
    let save_label = if state.settings.editing_theme.is_some() {
        "Update"
    } else {
        "Save"
    };
    if pill_button(
        cx,
        Rect::new(field.x + field.w + 9.0 + reset_w + 9.0, cy, save_w, 34.0),
        save_label,
        Some(Icon::Check),
        ready,
        11,
    ) && ready
    {
        actions.push(Action::CreateTheme);
    }
    cy += 50.0;

    // ---- Two clean groups: Interface & Lyrics.
    let side_by_side = width >= 600.0;
    let group_gap = 24.0;
    let group_w = if side_by_side {
        (cw - group_gap) * 0.5
    } else {
        cw
    };
    let left_x = cx0;
    let right_x = if side_by_side {
        cx0 + group_w + group_gap
    } else {
        cx0
    };
    let left_y = cy;
    let right_y = if side_by_side {
        cy
    } else {
        cy + 22.0 + 3.0 * 44.0 + 14.0
    };

    // Left group: Interface
    cx.text(
        Rect::new(left_x, left_y, group_w, 18.0),
        "Interface",
        11.5,
        theme::muted(),
        Align::Left,
    );
    let iface_slots = [
        (ColorSlot::Primary, "Background"),
        (ColorSlot::Secondary, "Text"),
        (ColorSlot::Accent, "Accent"),
    ];
    for (idx, (slot, label)) in iface_slots.into_iter().enumerate() {
        let ry = left_y + 22.0 + idx as f32 * 44.0;
        let rect = Rect::new(left_x, ry, group_w, 36.0);
        let current = settings_slot_color(&state.settings, slot);
        let active = state
            .settings
            .color_picker
            .as_ref()
            .is_some_and(|p| p.slot == slot);
        if color_slot_button(cx, rect, label, current, active, slot as u32) {
            open_color_picker(&mut state.settings, slot, rect);
        }
    }

    // Right group: Lyrics
    cx.text(
        Rect::new(right_x, right_y, group_w, 18.0),
        "Lyrics",
        11.5,
        theme::muted(),
        Align::Left,
    );
    let lyric_slots = [
        (ColorSlot::LyricsFill, "Text"),
        (ColorSlot::LyricsOutline, "Outline"),
    ];
    for (idx, (slot, label)) in lyric_slots.into_iter().enumerate() {
        let ry = right_y + 22.0 + idx as f32 * 44.0;
        let rect = Rect::new(right_x, ry, group_w, 36.0);
        let current = settings_slot_color(&state.settings, slot);
        let active = state
            .settings
            .color_picker
            .as_ref()
            .is_some_and(|p| p.slot == slot);
        if color_slot_button(cx, rect, label, current, active, slot as u32) {
            open_color_picker(&mut state.settings, slot, rect);
        }
    }

    // Visualizer row inside Lyrics group: label + mode pills (+ swatch when Custom).
    let viz_y = right_y + 22.0 + 2.0 * 44.0;
    let viz_row = Rect::new(right_x, viz_y, group_w, 36.0);
    cx.draw().round_rect(viz_row, 9.0, theme::raised());
    let label_w = 72.0;
    cx.text(
        Rect::new(viz_row.x + 10.0, viz_row.y, label_w, viz_row.h),
        "Visualizer",
        12.5,
        theme::muted(),
        Align::Left,
    );

    let custom_swatch_w = if state.settings.visualizer_mode == VisualizerColorRec::Custom {
        34.0
    } else {
        0.0
    };
    let pills_x = viz_row.x + label_w + 10.0;
    let pills_w = (viz_row.w - label_w - 18.0 - custom_swatch_w).max(120.0);
    let pill_gap = 4.0;
    let each_w = (pills_w - pill_gap * 2.0) / 3.0;
    let modes = [
        (VisualizerColorRec::AlbumArt, "Album Cover"),
        (VisualizerColorRec::Accent, "Accent"),
        (VisualizerColorRec::Custom, "Custom"),
    ];
    for (idx, (mode, label)) in modes.into_iter().enumerate() {
        let pill = Rect::new(
            pills_x + idx as f32 * (each_w + pill_gap),
            viz_row.y + 4.0,
            each_w,
            28.0,
        );
        let selected = state.settings.visualizer_mode == mode;
        let hover = cx.over_clickable(pill);
        let hv = cx.anims.toggle("viz_mode", idx as u32, hover, rate::SNAP);
        let bg = if selected {
            theme::accent()
        } else {
            mix(theme::panel(), theme::panel_hover(), hv)
        };
        let fg = if selected {
            theme::on_accent()
        } else {
            mix(theme::muted(), theme::fg(), hv)
        };
        cx.draw().round_rect(pill, 7.0, bg);
        cx.text_fit(pill, label, 11.5, fg, Align::Center);
        if cx.clicked(pill) {
            state.settings.visualizer_mode = mode;
            if mode != VisualizerColorRec::Custom
                && state
                    .settings
                    .color_picker
                    .as_ref()
                    .is_some_and(|p| p.slot == ColorSlot::Visualizer)
            {
                state.settings.color_picker = None;
            }
        }
    }

    if state.settings.visualizer_mode == VisualizerColorRec::Custom {
        let sw_rect = Rect::new(viz_row.x + viz_row.w - 32.0, viz_row.y + 4.0, 28.0, 28.0);
        let active = state
            .settings
            .color_picker
            .as_ref()
            .is_some_and(|p| p.slot == ColorSlot::Visualizer);
        let sw_hover = cx.over_clickable(sw_rect);
        cx.draw()
            .round_rect(sw_rect, 7.0, state.settings.visualizer_color);
        cx.draw().round_ring(
            sw_rect,
            7.0,
            if active { 1.5 } else { 1.0 },
            if active {
                theme::accent()
            } else {
                fade(theme::fg(), if sw_hover { 0.55 } else { 0.28 })
            },
        );
        if cx.clicked(sw_rect) {
            open_color_picker(&mut state.settings, ColorSlot::Visualizer, sw_rect);
        }
    }

    // ---- Per-theme Background row.
    let bg_y = right_y + 22.0 + 3.0 * 44.0 + 4.0;
    let bg_row = Rect::new(cx0, bg_y, cw, 36.0);
    cx.draw().round_rect(bg_row, 9.0, theme::raised());
    let has_tbg = state.settings.theme_background.is_some();
    let tbg_hit = Rect::new(
        bg_row.x,
        bg_row.y,
        bg_row.w - if has_tbg { 38.0 } else { 0.0 },
        bg_row.h,
    );
    let tbg_hover = cx.over_clickable(tbg_hit);
    let tbg_hv = cx.anims.toggle("theme_bg_pick", 0, tbg_hover, rate::SNAP);
    if tbg_hv > 0.01 {
        cx.draw()
            .round_rect(tbg_hit, 9.0, fade(theme::panel_hover(), tbg_hv));
    }
    cx.icon(
        Icon::Folder,
        (bg_row.x + 16.0, bg_row.y + 18.0),
        15.0,
        mix(theme::muted(), theme::fg(), tbg_hv.max(0.45)),
    );
    cx.text(
        Rect::new(bg_row.x + 34.0, bg_row.y, 82.0, bg_row.h),
        "Background",
        12.5,
        theme::muted(),
        Align::Left,
    );
    let tbg_label = state
        .settings
        .theme_background
        .as_deref()
        .and_then(|p| std::path::Path::new(p).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("None (click to choose)");
    cx.text_fit(
        Rect::new(
            bg_row.x + 120.0,
            bg_row.y,
            (tbg_hit.w - 132.0).max(40.0),
            bg_row.h,
        ),
        tbg_label,
        12.5,
        if has_tbg { theme::fg() } else { theme::faint() },
        Align::Left,
    );
    if cx.clicked(tbg_hit) {
        actions.push(Action::PickThemeBackground);
    }
    if has_tbg {
        let trash = Rect::new(bg_row.x + bg_row.w - 34.0, bg_row.y + 4.0, 28.0, 28.0);
        let thover = cx.over_clickable(trash);
        let thv = cx.anims.toggle("theme_bg_trash", 0, thover, rate::SNAP);
        cx.icon(
            Icon::Trash2,
            (trash.x + 14.0, trash.y + 14.0),
            13.5,
            mix(theme::faint(), theme::CLOSE_RED, thv),
        );
        if cx.clicked(trash) {
            state.cancel_background_prep();
            state.settings.theme_background = None;
        }
    }
    if has_tbg {
        let fy = bg_row.y + bg_row.h + 10.0;
        cx.text(
            Rect::new(cx0, fy, 46.0, 28.0),
            "Frost",
            13.0,
            theme::muted(),
            Align::Left,
        );
        let mut strength = state.settings.theme_background_blur;
        if cx.slider(
            SliderId::ThemeBlur,
            Rect::new(cx0 + 50.0, fy + 6.0, (cw - 106.0).max(100.0), 16.0),
            &mut strength,
            true,
            1.0,
        ) {
            actions.push(Action::ThemeBlurStrength(strength));
        }
        cx.text(
            Rect::new(cx0 + cw - 42.0, fy, 42.0, 28.0),
            &format!("{}%", (strength * 100.0).round() as i32),
            12.0,
            theme::muted(),
            Align::Right,
        );

        let ty = fy + 34.0;
        cx.text(
            Rect::new(cx0, ty, 88.0, 28.0),
            "Transparency",
            13.0,
            theme::muted(),
            Align::Left,
        );
        let mut transparency = state.settings.theme_background_transparency;
        if cx.slider(
            SliderId::ThemeTransparency,
            Rect::new(cx0 + 92.0, ty + 6.0, (cw - 148.0).max(100.0), 16.0),
            &mut transparency,
            true,
            1.0,
        ) {
            actions.push(Action::ThemeBackgroundTransparency(transparency));
        }
        cx.text(
            Rect::new(cx0 + cw - 42.0, ty, 42.0, 28.0),
            &format!("{}%", (transparency * 100.0).round() as i32),
            12.0,
            theme::muted(),
            Align::Right,
        );
    }
}

/// A compact clickable colour button showing label, live swatch dot with border, and hex code.
fn color_slot_button(
    cx: &mut Cx,
    rect: Rect,
    label: &str,
    color: Color,
    active: bool,
    key: u32,
) -> bool {
    let hover = cx.over_clickable(rect);
    let hv = cx.anims.toggle("color_slot", key, hover, rate::SNAP);
    let bg = if active {
        theme::panel_select()
    } else {
        mix(theme::raised(), theme::panel_hover(), hv)
    };
    cx.draw().round_rect(rect, 9.0, bg);
    if active {
        cx.draw().round_ring(rect, 9.0, 1.5, theme::accent());
    }

    let swatch = Rect::new(rect.x + 8.0, rect.y + (rect.h - 20.0) * 0.5, 20.0, 20.0);
    cx.draw().round_rect(swatch, 6.0, color);
    // Subtle 1px ring so #000000 or #ECECEC always has a crisp edge against any surface.
    cx.draw()
        .round_ring(swatch, 6.0, 1.0, fade(theme::fg(), 0.28));

    let hex = format_hex(color);
    let hex_w = 62.0;
    cx.text_fit(
        Rect::new(
            rect.x + 36.0,
            rect.y,
            (rect.w - 44.0 - hex_w).max(20.0),
            rect.h,
        ),
        label,
        12.5,
        mix(
            theme::muted(),
            theme::fg(),
            if active { 1.0 } else { hv.max(0.65) },
        ),
        Align::Left,
    );
    cx.text(
        Rect::new(rect.x + rect.w - hex_w - 8.0, rect.y, hex_w, rect.h),
        &hex,
        11.5,
        theme::faint(),
        Align::Right,
    );
    cx.clicked(rect)
}

/// Floating in-app full-range HSV + Hex colour picker popup.
fn draw_color_picker(cx: &mut Cx, state: &mut State, width: f32, height: f32) {
    let Some(picker) = state.settings.color_picker.as_mut() else {
        return;
    };
    let pop_w = 256.0;
    let pop_h = 312.0;
    let margin = 14.0;
    let px = (picker.anchor.x).clamp(margin, (width - pop_w - margin).max(margin));
    let below = picker.anchor.y + picker.anchor.h + 8.0;
    let py = if below + pop_h <= height - margin {
        below
    } else {
        (picker.anchor.y - pop_h - 8.0).max(margin)
    };
    let popup = Rect::new(px.round(), py.round(), pop_w, pop_h);

    // Click outside the popup closes it (skip on the exact frame it was opened).
    let just_opened = picker.just_opened;
    picker.just_opened = false;
    if !just_opened && cx.pointer.pressed && !popup.contains(cx.pointer.pos) && cx.drag.is_none() {
        state.settings.color_picker = None;
        return;
    }

    let previous = cx.draw().set_layer(MENU_LAYER + 1);
    cx.draw().box_shadow(
        popup,
        12.0,
        (0.0, 8.0),
        22.0,
        0.0,
        Color::rgba(0, 0, 0, 130),
    );
    cx.draw().round_rect(popup, 12.0, theme::track());
    cx.draw()
        .round_rect(popup.inset(1.0), 11.0, theme::raised());

    let pad = 16.0;
    let ix = popup.x + pad;
    let iw = popup.w - pad * 2.0;
    let mut iy = popup.y + 12.0;

    // Header: preview dot + slot title + close button.
    let current_color = theme::hsv_to_rgb(picker.hsv.0, picker.hsv.1, picker.hsv.2);
    cx.draw()
        .round_rect(Rect::new(ix, iy + 5.0, 14.0, 14.0), 4.0, current_color);
    cx.draw().round_ring(
        Rect::new(ix, iy + 5.0, 14.0, 14.0),
        4.0,
        1.0,
        fade(theme::fg(), 0.25),
    );
    cx.text(
        Rect::new(ix + 22.0, iy, iw - 50.0, 24.0),
        picker.slot.label(),
        13.0,
        theme::fg(),
        Align::Left,
    );
    let close_rect = Rect::new(popup.x + popup.w - 36.0, iy - 2.0, 26.0, 26.0);
    let close_hover = cx.over_clickable(close_rect);
    cx.icon(
        Icon::X,
        (close_rect.x + 13.0, close_rect.y + 13.0),
        14.0,
        if close_hover {
            theme::fg()
        } else {
            theme::muted()
        },
    );
    let close_clicked = cx.clicked(close_rect);
    iy += 30.0;

    // 2D Saturation / Value square.
    let sv_rect = Rect::new(ix, iy, iw, 144.0);
    let pure_hue = theme::hsv_to_rgb(picker.hsv.0, 1.0, 1.0);
    cx.draw().round_rect_gradient(
        sv_rect,
        8.0,
        [Color::WHITE, pure_hue, pure_hue, Color::WHITE],
    );
    cx.draw().round_rect_gradient(
        sv_rect,
        8.0,
        [
            Color::TRANSPARENT,
            Color::TRANSPARENT,
            Color::BLACK,
            Color::BLACK,
        ],
    );
    cx.draw()
        .round_ring(sv_rect, 8.0, 1.0, fade(theme::fg(), 0.14));

    if sv_rect.contains(cx.pointer.pos) {
        cx.hand = true;
        if cx.pointer.pressed {
            cx.drag = Some(SliderId::ColorSv);
        }
    }
    let mut hsv_changed = false;
    if cx.drag == Some(SliderId::ColorSv) {
        if cx.pointer.down {
            let s = ((cx.pointer.pos.0 - sv_rect.x) / sv_rect.w.max(1.0)).clamp(0.0, 1.0);
            let v = 1.0 - ((cx.pointer.pos.1 - sv_rect.y) / sv_rect.h.max(1.0)).clamp(0.0, 1.0);
            picker.hsv.1 = s;
            picker.hsv.2 = v;
            hsv_changed = true;
        } else {
            cx.drag = None;
        }
    }

    let sv_thumb = (
        sv_rect.x + picker.hsv.1 * sv_rect.w,
        sv_rect.y + (1.0 - picker.hsv.2) * sv_rect.h,
    );
    cx.draw().fill_circle(sv_thumb, 7.0, Color::WHITE);
    cx.draw().stroke_circle(sv_thumb, 7.0, 1.5, Color::BLACK);
    cx.draw().fill_circle(
        sv_thumb,
        4.5,
        theme::hsv_to_rgb(picker.hsv.0, picker.hsv.1, picker.hsv.2),
    );
    iy += 156.0;

    // Hue bar (6 smooth rainbow segments).
    let hue_rect = Rect::new(ix, iy, iw, 14.0);
    let stops = [
        Color::rgb(255, 0, 0),
        Color::rgb(255, 255, 0),
        Color::rgb(0, 255, 0),
        Color::rgb(0, 255, 255),
        Color::rgb(0, 0, 255),
        Color::rgb(255, 0, 255),
        Color::rgb(255, 0, 0),
    ];
    cx.draw().push_clip_rounded(hue_rect, 7.0);
    let seg_w = hue_rect.w / 6.0;
    for i in 0..6 {
        let seg = Rect::new(
            hue_rect.x + i as f32 * seg_w,
            hue_rect.y,
            seg_w + 0.5,
            hue_rect.h,
        );
        cx.draw()
            .gradient_corners(seg, [stops[i], stops[i + 1], stops[i + 1], stops[i]]);
    }
    cx.draw().pop_clip();
    cx.draw()
        .round_ring(hue_rect, 7.0, 1.0, fade(theme::fg(), 0.14));

    let hue_hit = hue_rect.expand(4.0);
    if hue_hit.contains(cx.pointer.pos) {
        cx.hand = true;
        if cx.pointer.pressed {
            cx.drag = Some(SliderId::ColorHue);
        }
    }
    if cx.drag == Some(SliderId::ColorHue) {
        if cx.pointer.down {
            let h = ((cx.pointer.pos.0 - hue_rect.x) / hue_rect.w.max(1.0)).clamp(0.0, 0.9999);
            picker.hsv.0 = h;
            hsv_changed = true;
        } else {
            cx.drag = None;
        }
    }
    let hue_thumb = (
        hue_rect.x + picker.hsv.0 * hue_rect.w,
        hue_rect.y + hue_rect.h * 0.5,
    );
    cx.draw().fill_circle(hue_thumb, 8.0, Color::WHITE);
    cx.draw().stroke_circle(hue_thumb, 8.0, 1.5, Color::BLACK);
    cx.draw()
        .fill_circle(hue_thumb, 5.0, theme::hsv_to_rgb(picker.hsv.0, 1.0, 1.0));
    iy += 26.0;

    let mut pending_color: Option<(ColorSlot, Color)> = None;

    // Quick preset dots (including Default #000000, #1A1A1A, #8A8A8A, #ECECEC).
    let dot_step = (iw - 20.0) / (PICKER_PRESETS.len() - 1) as f32;
    for (idx, preset) in PICKER_PRESETS.into_iter().enumerate() {
        let dc = (ix + 10.0 + idx as f32 * dot_step, iy + 10.0);
        let drect = Rect::new(dc.0 - 10.0, dc.1 - 10.0, 20.0, 20.0);
        if cx.over_clickable(drect) {
            cx.draw().fill_circle(dc, 10.0, fade(theme::fg(), 0.25));
        }
        cx.draw().fill_circle(dc, 8.0, preset);
        cx.draw()
            .stroke_circle(dc, 8.0, 1.0, fade(theme::fg(), 0.25));
        if cx.clicked(drect) {
            picker.hsv = theme::rgb_to_hsv(preset);
            picker.hex = format_hex(preset);
            picker.last_color = preset;
            hsv_changed = false;
            pending_color = Some((picker.slot, preset));
        }
    }
    iy += 30.0;

    if hsv_changed {
        let c = theme::hsv_to_rgb(picker.hsv.0, picker.hsv.1, picker.hsv.2);
        picker.hex = format_hex(c);
        picker.last_color = c;
        pending_color = Some((picker.slot, c));
    }

    // Editable #RRGGBB hex field + Done button.
    let mut style = TextFieldStyle::new(cx.font);
    style.size = 13.0;
    style.text = theme::fg();
    style.normal = theme::panel();
    style.hover = theme::panel_hover();
    style.accent = theme::accent();
    style.selection_alpha = 56;
    style.corner_radius = 8.0;
    style.border_width = 0.0;
    style.caret = theme::fg();
    style.caret_width = 1.5;
    style.caret_inset = 8.0;
    style.selection_inset = 8.0;
    style.text_inset = 10.0;
    let hex_rect = Rect::new(ix, iy, iw - 76.0, 32.0);
    cx.ui
        .text_field("color_hex", hex_rect, &mut picker.hex, style);
    if let Some(parsed) = parse_hex(&picker.hex) {
        if parsed != picker.last_color {
            picker.last_color = parsed;
            picker.hsv = theme::rgb_to_hsv(parsed);
            pending_color = Some((picker.slot, parsed));
        }
    }

    let done_clicked = pill_button(
        cx,
        Rect::new(ix + iw - 68.0, iy, 68.0, 32.0),
        "Done",
        None,
        true,
        13,
    );

    cx.draw().set_layer(previous);
    if let Some((slot, color)) = pending_color {
        set_settings_slot_color(&mut state.settings, slot, color);
    }
    if close_clicked || done_clicked {
        state.settings.color_picker = None;
    }
}

/// A small filled pill with an optional icon. Dimmed and inert when `enabled` is false.
fn pill_button(
    cx: &mut Cx,
    rect: Rect,
    label: &str,
    icon: Option<Icon>,
    enabled: bool,
    key: u32,
) -> bool {
    let hover = enabled && cx.over_clickable(rect);
    let hv = cx.anims.toggle("pill_hv", key, hover, rate::SNAP);
    cx.draw().round_rect(
        rect,
        rect.h * 0.5,
        mix(theme::raised(), theme::panel_press(), hv),
    );
    let color = if enabled {
        mix(theme::muted(), theme::fg(), hv.max(0.35))
    } else {
        theme::faint()
    };
    let mut text = rect;
    if let Some(icon) = icon {
        cx.icon(icon, (rect.x + 18.0, rect.y + rect.h * 0.5), 15.0, color);
        text = Rect::new(rect.x + 30.0, rect.y, rect.w - 36.0, rect.h);
    }
    cx.text(text, label, 12.5, color, Align::Center);
    enabled && cx.clicked(rect)
}

/// A switch: the knob slides and the track lights as it is turned on.
fn switch(cx: &mut Cx, key: u32, rect: Rect, on: bool) -> bool {
    let t = cx
        .anims
        .spring("switch", key, if on { 1.0 } else { 0.0 }, 620.0)
        .clamp(0.0, 1.0);
    let hit = Rect::new(rect.x - 6.0, rect.y - 6.0, rect.w + 12.0, rect.h + 12.0);
    let hover = cx.over_clickable(hit);
    let hv = cx.anims.toggle("switch_hv", key, hover, rate::SNAP);
    let off = mix(theme::raised_top(), theme::fg(), 0.12 * hv);
    cx.draw()
        .round_rect(rect, rect.h * 0.5, mix(off, theme::accent(), t));
    let r = rect.h * 0.5 - 3.0;
    let x = rect.x + r + 3.0 + t * (rect.w - 2.0 * (r + 3.0));
    let knob = if t > 0.5 {
        theme::on_accent()
    } else {
        mix(theme::muted(), theme::fg(), hv)
    };
    cx.draw()
        .round_rect(Rect::new(x - r, rect.y + 3.0, r * 2.0, r * 2.0), r, knob);
    cx.clicked(hit)
}

/// Confirmation dialog for destructive actions. Its buttons are live while the page under it is not.
fn draw_confirm_dialog(
    cx: &mut Cx,
    dialog: &ConfirmDialog,
    width: f32,
    height: f32,
    actions: &mut Vec<Action>,
) {
    let (title, body) = match dialog {
        ConfirmDialog::Downloads => (
            "Delete downloaded songs?",
            "This will delete all of the local copies of your albums and songs saved to Collections.",
        ),
        ConfirmDialog::DeleteTheme(_) => (
            "Delete theme?",
            "This will permanently delete this theme.",
        ),
        ConfirmDialog::RemoveBackground => (
            "Delete background?",
            "This will permanently delete this background.",
        ),
    };
    cx.draw().rect(
        Rect::new(0.0, 0.0, width, height),
        Color::rgba(0, 0, 0, 150),
    );
    let w = 420.0f32.min(width - 40.0);
    let h = 184.0;
    let panel = Rect::new(
        ((width - w) * 0.5).round(),
        ((height - h) * 0.5).round(),
        w,
        h,
    );
    let layer = cx.draw().set_layer(MENU_LAYER + 1);
    cx.draw().backdrop_rounded(panel, Color::WHITE, 16.0);
    cx.draw().round_rect(panel, 16.0, theme::panel());
    cx.text(
        Rect::new(panel.x + 24.0, panel.y + 22.0, w - 48.0, 26.0),
        title,
        17.0,
        theme::fg(),
        Align::Center,
    );
    let body_h = cx.text_block_height(w - 48.0, body, 13.0, 1.35);
    cx.text_block(
        Rect::new(panel.x + 24.0, panel.y + 56.0, w - 48.0, body_h),
        body,
        13.0,
        theme::muted(),
        1.35,
        Align::Center,
    );
    let by = panel.y + h - 54.0;
    let cancel_w = 92.0;
    let delete_w = 92.0;
    let gap = 10.0;
    let bx = panel.x + (w - (cancel_w + gap + delete_w)) * 0.5;
    if dialog_button(cx, Rect::new(bx, by, cancel_w, 34.0), "Cancel", None, false) {
        actions.push(Action::ConfirmDialog(None));
    }
    if dialog_button(
        cx,
        Rect::new(bx + cancel_w + gap, by, delete_w, 34.0),
        "Delete",
        None,
        true,
    ) {
        match dialog {
            ConfirmDialog::Downloads => actions.push(Action::SetDownloads(false)),
            ConfirmDialog::DeleteTheme(name) => actions.push(Action::DeleteTheme(name.clone())),
            ConfirmDialog::RemoveBackground => actions.push(Action::RemoveBackground),
        }
    }
    cx.draw().set_layer(layer);
}

fn dialog_button(cx: &mut Cx, rect: Rect, label: &str, icon: Option<Icon>, danger: bool) -> bool {
    let hover = cx.over_clickable(rect);
    let hv = cx
        .anims
        .toggle("dlg_btn", hash(label) % 512, hover, rate::SNAP);
    cx.draw()
        .round_rect(rect, 9.0, mix(theme::raised(), theme::panel_hover(), hv));
    let base = if danger {
        theme::CLOSE_RED
    } else {
        theme::muted()
    };
    let hot = if danger {
        mix(theme::CLOSE_RED, theme::fg(), 0.3)
    } else {
        theme::fg()
    };
    let ink = mix(base, hot, hv);
    if let Some(icon) = icon {
        cx.icon(icon, (rect.x + 18.0, rect.y + rect.h * 0.5), 14.0, ink);
        cx.text(
            Rect::new(rect.x + 28.0, rect.y, rect.w - 34.0, rect.h),
            label,
            13.0,
            ink,
            Align::Center,
        );
    } else {
        cx.text(rect, label, 13.0, ink, Align::Center);
    }
    cx.clicked(rect)
}

fn text_button(cx: &mut Cx, rect: Rect, label: &str, danger: bool) -> bool {
    let hover = cx.over_clickable(rect);
    let hv = cx
        .anims
        .toggle("text_btn", hash(label) % 512, hover, rate::SNAP);
    if hv > 0.0 {
        cx.draw()
            .round_rect(rect, 8.0, fade(theme::panel_hover(), hv));
    }
    let base = if danger {
        theme::CLOSE_RED
    } else {
        theme::muted()
    };
    let hot = if danger {
        mix(theme::CLOSE_RED, theme::fg(), 0.3)
    } else {
        theme::fg()
    };
    cx.text(rect, label, 13.0, mix(base, hot, hv), Align::Center);
    cx.clicked(rect)
}

/// Saved themes, favourites first, each group in the order it was made.
fn theme_order(themes: &[ThemeRec]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..themes.len()).collect();
    order.sort_by_key(|index| (!themes[*index].favorite, *index));
    order
}

/// A name not already taken, so two themes never collide in the list.
fn unique_theme_name(base: &str, themes: &[ThemeRec]) -> String {
    let base = if base.is_empty() { "My theme" } else { base };
    // `Default` belongs to the built-in theme, so saved ones may not take it.
    let taken = |name: &str| {
        name.eq_ignore_ascii_case("Default") || themes.iter().any(|theme| theme.name == name)
    };
    if !taken(base) {
        return base.to_owned();
    }
    let mut number = 2;
    loop {
        let candidate = format!("{base} {number}");
        if !taken(&candidate) {
            return candidate;
        }
        number += 1;
    }
}

fn rgb_color(value: u32) -> Color {
    Color::rgb((value >> 16) as u8, (value >> 8) as u8, value as u8)
}

fn rgb_value(color: Color) -> u32 {
    (u32::from(color.r) << 16) | (u32::from(color.g) << 8) | u32::from(color.b)
}

/// The palette a theme name stands for.
fn theme_palette(active: Option<&str>, themes: &[ThemeRec]) -> theme::Palette {
    match active.and_then(|name| themes.iter().find(|theme| theme.name == name)) {
        Some(theme) => {
            let viz = match theme.visualizer_mode {
                VisualizerColorRec::AlbumArt => theme::VisualizerColor::AlbumArt,
                VisualizerColorRec::Accent => theme::VisualizerColor::Accent,
                VisualizerColorRec::Custom => theme::VisualizerColor::Custom(
                    theme
                        .visualizer_color
                        .map(rgb_color)
                        .unwrap_or_else(|| rgb_color(theme.accent)),
                ),
            };
            theme::Palette::from_theme(
                rgb_color(theme.primary),
                rgb_color(theme.secondary),
                rgb_color(theme.accent),
                theme.lyrics_fill.map(rgb_color),
                theme.lyrics_outline.map(rgb_color),
                viz,
            )
        }
        None => theme::Palette::default(),
    }
}

fn effective_background_path(prefs: &Prefs, themes: &[ThemeRec]) -> Option<std::path::PathBuf> {
    if prefs.background_enabled() {
        if let Some(global) = prefs.background() {
            return Some(global.to_path_buf());
        }
    }
    prefs
        .active_theme()
        .and_then(|name| themes.iter().find(|t| t.name == name))
        .and_then(|t| t.background.as_deref())
        .map(std::path::PathBuf::from)
}

fn effective_background_transparency(state: &State) -> f32 {
    let global_active = state.prefs.background_enabled() && state.prefs.background().is_some();
    if global_active {
        return state.prefs.background_transparency();
    }
    if state.settings.creator_open
        && state.settings.editing_theme.as_deref().is_some()
        && state.settings.editing_theme.as_deref() == state.prefs.active_theme()
    {
        return state.settings.theme_background_transparency;
    }
    state
        .prefs
        .active_theme()
        .and_then(|name| state.themes.iter().find(|t| t.name == name))
        .filter(|t| t.background.is_some())
        .map(|t| t.background_transparency)
        .unwrap_or_else(|| state.prefs.background_transparency())
}

fn ui_panel_alpha(state: &State) -> f32 {
    (1.0 - effective_background_transparency(state)).clamp(0.0, 1.0)
}

fn effective_background_blur(prefs: &Prefs, themes: &[crate::prefs::ThemeRec]) -> f32 {
    if prefs.background_enabled() && prefs.background().is_some() {
        return prefs.background_blur();
    }
    prefs
        .active_theme()
        .and_then(|name| themes.iter().find(|t| t.name == name))
        .filter(|t| t.background.is_some())
        .map(|t| t.background_blur)
        .unwrap_or_else(|| prefs.background_blur())
}

fn sync_background(state: &mut State) {
    state
        .background
        .set_strength(effective_background_blur(&state.prefs, &state.themes));
    let target = effective_background_path(&state.prefs, &state.themes);
    if state.background.path == target {
        return;
    }
    match target {
        Some(path) => state.background.choose(path),
        None => {
            state.background.clear();
            state.gpu.clear_background();
        }
    }
}

fn set_theme(state: &mut State, name: Option<String>) {
    state.prefs.set_active_theme(name.as_deref());
    theme::set_palette(theme_palette(name.as_deref(), &state.themes));
    sync_background(state);
}
