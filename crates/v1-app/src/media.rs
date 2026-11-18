// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! System media controls.
//!
//! This is what makes earbuds, keyboard media keys, and the desktop's media flyout drive the app.
//! Windows uses System Media Transport Controls (SMTC); macOS uses MPRemoteCommandCenter and
//! MPNowPlayingInfoCenter; Linux exports an MPRIS player on the session bus, which is what the
//! desktop's media keys are usually bound to (through `playerctl` or a panel widget).
//!
//! Transport buttons are forwarded to the UI thread and applied like a press on the bar. Display
//! metadata and the timeline are pushed back so the flyout shows the track and seeks land in the
//! right place. Everything here is best effort: if the OS refuses a session, the app still runs.

/// A transport action requested by the OS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Command {
    Play,
    Pause,
    Toggle,
    Next,
    Previous,
    Stop,
    /// Seek to this position, in seconds.
    Seek(f64),
    /// Set the output volume, 0 to 1. Only MPRIS carries a volume knob.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Volume(f64),
}

/// What the SMTC session should show. Borrowed from the track view, so building it every frame
/// copies nothing.
#[derive(Debug, Clone, Copy)]
pub struct MediaInfo<'a> {
    pub title: &'a str,
    pub artist: &'a str,
    pub album: Option<&'a str>,
    /// Playback position and length, in seconds.
    pub position: f64,
    pub duration: f64,
    pub playing: bool,
    /// The output volume, 0 to 1. MPRIS exposes this to the desktop.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub volume: f32,
}

#[cfg(any(windows, target_os = "macos", target_os = "linux"))]
pub use imp::MediaControls;

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
pub struct MediaControls;

#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
impl MediaControls {
    pub fn new<F>(_window: &winit::window::Window, _on_command: F) -> Option<Self>
    where
        F: Fn(Command) + Send + 'static,
    {
        None
    }

    pub fn update(&mut self, _info: Option<&MediaInfo<'_>>) {}
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use block2::RcBlock;
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, AnyObject};

    use super::{Command, MediaInfo};

    #[link(name = "Foundation", kind = "framework")]
    extern "C" {}

    #[link(name = "MediaPlayer", kind = "framework")]
    extern "C" {
        static MPMediaItemPropertyTitle: *mut AnyObject;
        static MPMediaItemPropertyArtist: *mut AnyObject;
        static MPMediaItemPropertyAlbumTitle: *mut AnyObject;
        static MPMediaItemPropertyPlaybackDuration: *mut AnyObject;
        static MPNowPlayingInfoPropertyElapsedPlaybackTime: *mut AnyObject;
        static MPNowPlayingInfoPropertyPlaybackRate: *mut AnyObject;
    }

    const MP_REMOTE_COMMAND_HANDLER_STATUS_SUCCESS: isize = 0;
    const MP_NOW_PLAYING_PLAYBACK_STATE_STOPPED: usize = 2;
    const MP_NOW_PLAYING_PLAYBACK_STATE_PLAYING: usize = 1;
    const MP_NOW_PLAYING_PLAYBACK_STATE_PAUSED: usize = 2;

    const TIMELINE_THROTTLE: Duration = Duration::from_secs(2);

    pub struct MediaControls {
        last_title: String,
        last_artist: String,
        last_album: Option<String>,
        last_playing: Option<bool>,
        last_timeline: Option<Instant>,
        last_position: f64,
        last_duration: f64,
        targets: Vec<(*mut AnyObject, *mut AnyObject)>,
    }

    // SAFETY: `targets` are retained Objective-C command/token pointers only touched on creation
    // and drop, and `MediaControls` lives on the main UI thread.
    unsafe impl Send for MediaControls {}

    unsafe fn nsstring(s: &str) -> *mut AnyObject {
        let cls = AnyClass::get("NSString").expect("NSString class");
        let obj: *mut AnyObject = msg_send![cls, alloc];
        let bytes = s.as_ptr() as *const c_void;
        let len = s.len();
        // NSUTF8StringEncoding = 4
        msg_send![obj, initWithBytes: bytes, length: len, encoding: 4usize]
    }

    unsafe fn nsnumber_f64(v: f64) -> *mut AnyObject {
        let cls = AnyClass::get("NSNumber").expect("NSNumber class");
        msg_send![cls, numberWithDouble: v]
    }

    impl MediaControls {
        pub fn new<F>(_window: &winit::window::Window, on_command: F) -> Option<Self>
        where
            F: Fn(Command) + Send + 'static,
        {
            // Wrap in Arc; `RcBlock` requires `Fn` + `'static`.
            let cb = Arc::new(on_command);
            let mut targets = Vec::new();
            unsafe {
                let cls = AnyClass::get("MPRemoteCommandCenter")?;
                let center: *mut AnyObject = msg_send![cls, sharedCommandCenter];
                if center.is_null() {
                    return None;
                }

                // Helper closure-free registration using a thin Sync wrapper.
                struct SyncCb<F>(F);
                // SAFETY: `F: Send + 'static`, only called from MPRemoteCommandCenter handler.
                unsafe impl<F: Send> Sync for SyncCb<F> {}
                let cb = Arc::new(SyncCb(cb));

                let mut reg = |selector: &str, cmd: Command| {
                    let sel = objc2::runtime::Sel::register(selector);
                    let command: *mut AnyObject = msg_send![center, performSelector: sel];
                    if command.is_null() {
                        return;
                    }
                    let _: () = msg_send![command, setEnabled: true];
                    let cb = Arc::clone(&cb);
                    let block = RcBlock::new(move |_event: NonNull<AnyObject>| -> isize {
                        (cb.0)(cmd);
                        MP_REMOTE_COMMAND_HANDLER_STATUS_SUCCESS
                    });
                    let token: *mut AnyObject = msg_send![command, addTargetWithHandler: &*block];
                    if !token.is_null() {
                        let _: *mut AnyObject = msg_send![command, retain];
                        let _: *mut AnyObject = msg_send![token, retain];
                        targets.push((command, token));
                    }
                };

                reg("playCommand", Command::Play);
                reg("pauseCommand", Command::Pause);
                reg("togglePlayPauseCommand", Command::Toggle);
                reg("nextTrackCommand", Command::Next);
                reg("previousTrackCommand", Command::Previous);
                reg("stopCommand", Command::Stop);

                // Seek command
                let seek_sel = objc2::runtime::Sel::register("changePlaybackPositionCommand");
                let seek_cmd: *mut AnyObject = msg_send![center, performSelector: seek_sel];
                if !seek_cmd.is_null() {
                    let _: () = msg_send![seek_cmd, setEnabled: true];
                    let cb = Arc::clone(&cb);
                    let block = RcBlock::new(move |event: NonNull<AnyObject>| -> isize {
                        let pos: f64 = msg_send![event.as_ptr(), positionTime];
                        (cb.0)(Command::Seek(pos.max(0.0)));
                        MP_REMOTE_COMMAND_HANDLER_STATUS_SUCCESS
                    });
                    let token: *mut AnyObject = msg_send![seek_cmd, addTargetWithHandler: &*block];
                    if !token.is_null() {
                        let _: *mut AnyObject = msg_send![seek_cmd, retain];
                        let _: *mut AnyObject = msg_send![token, retain];
                        targets.push((seek_cmd, token));
                    }
                }
            }

            Some(Self {
                last_title: String::new(),
                last_artist: String::new(),
                last_album: None,
                last_playing: None,
                last_timeline: None,
                last_position: 0.0,
                last_duration: 0.0,
                targets,
            })
        }

        pub fn update(&mut self, info: Option<&MediaInfo<'_>>) {
            unsafe {
                let Some(cls) = AnyClass::get("MPNowPlayingInfoCenter") else {
                    return;
                };
                let center: *mut AnyObject = msg_send![cls, defaultCenter];
                if center.is_null() {
                    return;
                }

                let Some(info) = info else {
                    if self.last_playing != Some(false) || !self.last_title.is_empty() {
                        let null: *mut AnyObject = std::ptr::null_mut();
                        let _: () = msg_send![center, setNowPlayingInfo: null];
                        let _: () = msg_send![center, setPlaybackState: MP_NOW_PLAYING_PLAYBACK_STATE_STOPPED];
                        self.last_title.clear();
                        self.last_artist.clear();
                        self.last_album = None;
                        self.last_playing = Some(false);
                    }
                    return;
                };

                let meta_changed = self.last_title != info.title
                    || self.last_artist != info.artist
                    || self.last_album.as_deref() != info.album;
                let state_changed = self.last_playing != Some(info.playing);
                let now = Instant::now();
                let jumped = (info.position - self.last_position).abs() > 2.5
                    || (info.duration - self.last_duration).abs() > 0.5;
                let due = self
                    .last_timeline
                    .is_none_or(|t| now.duration_since(t) >= TIMELINE_THROTTLE);

                if !meta_changed && !state_changed && !jumped && !due {
                    return;
                }

                let dict_cls = AnyClass::get("NSMutableDictionary").expect("NSMutableDictionary");
                let dict: *mut AnyObject = msg_send![dict_cls, dictionary];

                let title = nsstring(info.title);
                let artist = nsstring(info.artist);
                let _: () = msg_send![dict, setObject: title, forKey: MPMediaItemPropertyTitle];
                let _: () = msg_send![dict, setObject: artist, forKey: MPMediaItemPropertyArtist];
                let _: () = msg_send![title, release];
                let _: () = msg_send![artist, release];

                if let Some(album) = info.album {
                    let ns_album = nsstring(album);
                    let _: () =
                        msg_send![dict, setObject: ns_album, forKey: MPMediaItemPropertyAlbumTitle];
                    let _: () = msg_send![ns_album, release];
                }

                let dur = nsnumber_f64(info.duration.max(0.0));
                let pos = nsnumber_f64(info.position.max(0.0));
                let rate = nsnumber_f64(if info.playing { 1.0 } else { 0.0 });
                let _: () =
                    msg_send![dict, setObject: dur, forKey: MPMediaItemPropertyPlaybackDuration];
                let _: () = msg_send![dict, setObject: pos, forKey: MPNowPlayingInfoPropertyElapsedPlaybackTime];
                let _: () =
                    msg_send![dict, setObject: rate, forKey: MPNowPlayingInfoPropertyPlaybackRate];

                let _: () = msg_send![center, setNowPlayingInfo: dict];
                let state = if info.playing {
                    MP_NOW_PLAYING_PLAYBACK_STATE_PLAYING
                } else {
                    MP_NOW_PLAYING_PLAYBACK_STATE_PAUSED
                };
                let _: () = msg_send![center, setPlaybackState: state];

                if meta_changed {
                    self.last_title.clear();
                    self.last_title.push_str(info.title);
                    self.last_artist.clear();
                    self.last_artist.push_str(info.artist);
                    self.last_album = info.album.map(str::to_owned);
                }
                self.last_playing = Some(info.playing);
                self.last_timeline = Some(now);
                self.last_position = info.position;
                self.last_duration = info.duration;
            }
        }
    }

    impl Drop for MediaControls {
        fn drop(&mut self) {
            unsafe {
                if let Some(cls) = AnyClass::get("MPNowPlayingInfoCenter") {
                    let center: *mut AnyObject = msg_send![cls, defaultCenter];
                    if !center.is_null() {
                        let null: *mut AnyObject = std::ptr::null_mut();
                        let _: () = msg_send![center, setNowPlayingInfo: null];
                        let _: () = msg_send![center, setPlaybackState: MP_NOW_PLAYING_PLAYBACK_STATE_STOPPED];
                    }
                }
                for (command, token) in self.targets.drain(..) {
                    let _: () = msg_send![command, removeTarget: token];
                    let _: () = msg_send![token, release];
                    let _: () = msg_send![command, release];
                }
            }
        }
    }
}

/// Linux media controls: an MPRIS player on the session bus.
///
/// Desktop media keys and panel widgets do not talk to the focused window; they talk to
/// `org.mpris.MediaPlayer2` services on the session bus (usually through `playerctl`). Exporting
/// that interface is therefore what makes earbuds, keyboard transport buttons and the desktop's
/// media widget drive the app.
///
/// The session lives on its own thread, because the zbus blocking connection runs its own
/// executor and must not block the event loop. The UI thread pushes a snapshot when the track,
/// playback state or volume changes (and once a second while playing, for the timeline); the
/// D-Bus thread turns those into `PropertiesChanged` signals. Transport calls come back over a
/// second channel, which the UI thread drains as ordinary media commands.
#[cfg(target_os = "linux")]
mod imp {
    use std::collections::HashMap;
    use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::time::Instant;

    use zbus::blocking::Connection;
    use zbus::object_server::SignalEmitter;
    use zbus::zvariant::{ObjectPath, OwnedObjectPath, OwnedValue, Value};

    use super::{Command, MediaInfo};

    /// MPRIS requires the bus name to be `org.mpris.MediaPlayer2.<player>`. The suffix also has to
    /// match the desktop entry's base name, so it is `v1`.
    const BUS_NAME: &str = "org.mpris.MediaPlayer2.v1";
    const OBJECT_PATH: &str = "/org/mpris/MediaPlayer2";
    const PLAYER_INTERFACE: &str = "org.mpris.MediaPlayer2.Player";
    /// The track-id object path root. Each new track gets a fresh id from a counter.
    const TRACK_ROOT: &str = "/org/v1/track";
    const NO_TRACK: &str = "/org/mpris/MediaPlayer2/TrackList/NoTrack";
    /// A position jump bigger than this looks like a seek rather than normal progress. Clients
    /// interpolate from the last position, so ordinary playback needs no periodic update.
    const SEEK_JUMP: f64 = 2.5;
    /// How far a position may drift from where playback should have reached before it counts as a
    /// seek. Slightly wider than a dropped frame's worth of progress.
    const DRIFT: f64 = 1.0;

    /// The track as the D-Bus thread keeps it.
    struct Track {
        title: String,
        artist: String,
        album: Option<String>,
        duration: f64,
        position: f64,
        playing: bool,
        /// When `position` was last written, for interpolating while playing.
        at: Instant,
        /// Changes when the track does, so clients see a new `mpris:trackid`.
        id: u64,
    }

    impl Track {
        fn position_now(&self) -> f64 {
            if self.playing {
                self.position + self.at.elapsed().as_secs_f64()
            } else {
                self.position
            }
        }

        fn path(&self) -> String {
            format!("{TRACK_ROOT}/{}", self.id)
        }
    }

    #[derive(Default)]
    struct Shared {
        track: Option<Track>,
        volume: f64,
        next_id: u64,
    }

    /// A copy of the player state sent from the UI thread.
    struct Snapshot {
        title: String,
        artist: String,
        album: Option<String>,
        position: f64,
        duration: f64,
        playing: bool,
        volume: f32,
    }

    enum Update {
        Snapshot(Snapshot),
        Clear,
        Stop,
    }

    struct Player {
        shared: Arc<Mutex<Shared>>,
        commands: SyncSender<Command>,
    }

    impl Player {
        fn send(&self, command: Command) {
            // The UI thread drains this promptly; a full queue means the app is shutting down.
            let _ = self.commands.try_send(command);
        }

        fn shared(&self) -> MutexGuard<'_, Shared> {
            self.shared
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }

    #[zbus::interface(name = "org.mpris.MediaPlayer2.Player")]
    impl Player {
        fn next(&self) {
            self.send(Command::Next);
        }

        fn previous(&self) {
            self.send(Command::Previous);
        }

        fn pause(&self) {
            self.send(Command::Pause);
        }

        fn play(&self) {
            self.send(Command::Play);
        }

        fn play_pause(&self) {
            self.send(Command::Toggle);
        }

        fn stop(&self) {
            self.send(Command::Stop);
        }

        /// `Seek` is relative, in microseconds, as MPRIS specifies.
        fn seek(&self, offset: i64) {
            let position = self
                .shared()
                .track
                .as_ref()
                .map(Track::position_now)
                .unwrap_or(0.0);
            self.send(Command::Seek(
                (position + offset as f64 / 1_000_000.0).max(0.0),
            ));
        }

        /// `SetPosition` addresses a specific track; ignore it when the track has moved on.
        fn set_position(&self, track_id: OwnedObjectPath, position: i64) {
            let current = self.shared().track.as_ref().map(Track::path);
            if current.as_deref() != Some(track_id.as_str()) {
                return;
            }
            self.send(Command::Seek((position as f64 / 1_000_000.0).max(0.0)));
        }

        fn open_uri(&self, _uri: String) -> zbus::fdo::Result<()> {
            Err(zbus::fdo::Error::NotSupported(
                "V1 cannot open URIs".to_owned(),
            ))
        }

        #[zbus(property)]
        fn playback_status(&self) -> String {
            match &self.shared().track {
                Some(track) if track.playing => "Playing",
                Some(_) => "Paused",
                None => "Stopped",
            }
            .to_owned()
        }

        #[zbus(property)]
        fn metadata(&self) -> HashMap<String, OwnedValue> {
            let shared = self.shared();
            metadata_of(&shared)
        }

        #[zbus(property)]
        fn position(&self) -> i64 {
            self.shared()
                .track
                .as_ref()
                .map_or(0, |track| seconds_to_micros(track.position_now()))
        }

        #[zbus(property)]
        fn volume(&self) -> f64 {
            self.shared().volume
        }

        #[zbus(property)]
        fn set_volume(&self, volume: f64) {
            self.send(Command::Volume(volume.clamp(0.0, 1.0)));
        }

        #[zbus(property)]
        fn rate(&self) -> f64 {
            1.0
        }

        #[zbus(property)]
        fn minimum_rate(&self) -> f64 {
            1.0
        }

        #[zbus(property)]
        fn maximum_rate(&self) -> f64 {
            1.0
        }

        #[zbus(property)]
        fn can_go_next(&self) -> bool {
            true
        }

        #[zbus(property)]
        fn can_go_previous(&self) -> bool {
            true
        }

        #[zbus(property)]
        fn can_play(&self) -> bool {
            true
        }

        #[zbus(property)]
        fn can_pause(&self) -> bool {
            true
        }

        #[zbus(property)]
        fn can_seek(&self) -> bool {
            true
        }

        #[zbus(property)]
        fn can_control(&self) -> bool {
            true
        }

        #[zbus(property)]
        fn loop_status(&self) -> String {
            "None".to_owned()
        }

        #[zbus(property)]
        fn shuffle(&self) -> bool {
            false
        }
    }

    /// The `org.mpris.MediaPlayer2` root interface. V1 cannot be raised or quit through D-Bus,
    /// so those capabilities are reported false.
    struct Root;

    #[zbus::interface(name = "org.mpris.MediaPlayer2")]
    impl Root {
        fn raise(&self) {}
        fn quit(&self) {}

        #[zbus(property)]
        fn identity(&self) -> String {
            "V1".to_owned()
        }

        #[zbus(property)]
        fn desktop_entry(&self) -> String {
            "v1".to_owned()
        }

        #[zbus(property)]
        fn supported_uri_schemes(&self) -> Vec<String> {
            Vec::new()
        }

        #[zbus(property)]
        fn supported_mime_types(&self) -> Vec<String> {
            Vec::new()
        }

        #[zbus(property)]
        fn can_quit(&self) -> bool {
            false
        }

        #[zbus(property)]
        fn can_raise(&self) -> bool {
            false
        }

        #[zbus(property)]
        fn has_track_list(&self) -> bool {
            false
        }

        #[zbus(property)]
        fn can_set_fullscreen(&self) -> bool {
            false
        }

        #[zbus(property)]
        fn fullscreen(&self) -> bool {
            false
        }
    }

    /// The MPRIS metadata dictionary for the current track.
    fn metadata_of(shared: &Shared) -> HashMap<String, OwnedValue> {
        let mut metadata = HashMap::new();
        let Some(track) = &shared.track else {
            metadata.insert("mpris:trackid".to_owned(), path_value(NO_TRACK));
            return metadata;
        };
        metadata.insert("mpris:trackid".to_owned(), path_value(&track.path()));
        metadata.insert(
            "mpris:length".to_owned(),
            OwnedValue::from(seconds_to_micros(track.duration)),
        );
        metadata.insert(
            "xesam:title".to_owned(),
            OwnedValue::from(zbus::zvariant::Str::from(track.title.as_str())),
        );
        metadata.insert(
            "xesam:artist".to_owned(),
            OwnedValue::try_from(Value::from(vec![track.artist.clone()]))
                .expect("an artist array is a D-Bus value"),
        );
        if let Some(album) = track.album.as_deref().filter(|album| !album.is_empty()) {
            metadata.insert(
                "xesam:album".to_owned(),
                OwnedValue::from(zbus::zvariant::Str::from(album)),
            );
        }
        metadata
    }

    fn path_value(path: &str) -> OwnedValue {
        let path = ObjectPath::try_from(path).expect("a well-formed MPRIS track path");
        OwnedValue::from(path)
    }

    fn seconds_to_micros(seconds: f64) -> i64 {
        if !seconds.is_finite() || seconds <= 0.0 {
            return 0;
        }
        (seconds * 1_000_000.0).round().clamp(0.0, i64::MAX as f64) as i64
    }

    /// Announce changed player properties. An empty map is not worth a signal.
    fn announce(emitter: &SignalEmitter<'_>, changed: HashMap<String, Value<'static>>) {
        if changed.is_empty() {
            return;
        }
        let body = (PLAYER_INTERFACE, changed, Vec::<String>::new());
        let _ = zbus::block_on(emitter.emit(
            "org.freedesktop.DBus.Properties",
            "PropertiesChanged",
            &body,
        ));
    }

    fn announce_seeked(emitter: &SignalEmitter<'_>, position: f64) {
        let _ =
            zbus::block_on(emitter.emit(PLAYER_INTERFACE, "Seeked", &seconds_to_micros(position)));
    }

    /// Serve the MPRIS interfaces until the UI thread closes the channel.
    fn serve(updates: Receiver<Update>, commands: SyncSender<Command>) {
        let Ok(connection) = Connection::session() else {
            return;
        };
        // Another copy of V1 (or a stale name owner) wins the name; stay quiet then.
        if connection.request_name(BUS_NAME).is_err() {
            return;
        }
        let shared = Arc::new(Mutex::new(Shared::default()));
        let player = Player {
            shared: Arc::clone(&shared),
            commands,
        };
        if connection.object_server().at(OBJECT_PATH, Root).is_err() {
            return;
        }
        if connection.object_server().at(OBJECT_PATH, player).is_err() {
            return;
        }
        let emitter = match connection
            .object_server()
            .interface::<_, Player>(OBJECT_PATH)
        {
            Ok(reference) => reference.signal_emitter().clone(),
            Err(_) => return,
        };

        let mut announced_volume = -1.0;
        loop {
            // Nothing here is periodic: the player's position is interpolated from the last
            // snapshot, and every real change arrives on the channel. Block until one does.
            let update = match updates.recv() {
                Ok(update) => update,
                Err(_) => return,
            };
            let mut changed: HashMap<String, Value<'static>> = HashMap::new();
            match update {
                Update::Stop => return,
                Update::Clear => {
                    let mut shared = shared
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if shared.track.take().is_some() {
                        changed.insert("Metadata".to_owned(), Value::from(metadata_of(&shared)));
                        changed.insert("PlaybackStatus".to_owned(), Value::from("Stopped"));
                    }
                }
                Update::Snapshot(snapshot) => {
                    let mut shared = shared
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let same_track = shared.track.as_ref().is_some_and(|track| {
                        track.title == snapshot.title
                            && track.artist == snapshot.artist
                            && track.album == snapshot.album
                    });
                    let mut seeked = None;
                    let metadata_changed = if same_track {
                        let track = shared.track.as_mut().expect("same track exists");
                        let playing_changed = track.playing != snapshot.playing;
                        let length_changed = (track.duration - snapshot.duration).abs() > 0.5;
                        // Ordinary playback has moved the position since the last snapshot, so
                        // compare against where it should have reached, not where it was then.
                        let expected = track.position_now();
                        if (snapshot.position - expected).abs() > SEEK_JUMP {
                            seeked = Some(snapshot.position);
                        }
                        track.position = snapshot.position;
                        track.at = Instant::now();
                        track.playing = snapshot.playing;
                        track.duration = snapshot.duration;
                        if playing_changed {
                            let status = if snapshot.playing {
                                "Playing"
                            } else {
                                "Paused"
                            };
                            changed.insert("PlaybackStatus".to_owned(), Value::from(status));
                        }
                        length_changed
                    } else {
                        shared.next_id += 1;
                        let id = shared.next_id;
                        shared.track = Some(Track {
                            title: snapshot.title,
                            artist: snapshot.artist,
                            album: snapshot.album,
                            duration: snapshot.duration,
                            position: snapshot.position,
                            playing: snapshot.playing,
                            at: Instant::now(),
                            id,
                        });
                        changed.insert(
                            "PlaybackStatus".to_owned(),
                            Value::from(if snapshot.playing {
                                "Playing"
                            } else {
                                "Paused"
                            }),
                        );
                        true
                    };
                    if metadata_changed {
                        changed.insert("Metadata".to_owned(), Value::from(metadata_of(&shared)));
                    }
                    let volume = f64::from(snapshot.volume);
                    if (volume - announced_volume).abs() > 0.005 {
                        shared.volume = volume;
                        announced_volume = volume;
                        changed.insert("Volume".to_owned(), Value::from(volume));
                    }
                    if let Some(position) = seeked {
                        announce_seeked(&emitter, position);
                    }
                }
            }
            announce(&emitter, changed);
        }
    }

    /// The identity and state last sent to D-Bus, so a frame that changes nothing allocates
    /// nothing and crosses no thread boundary.
    struct Shown {
        title: String,
        artist: String,
        album: Option<String>,
        playing: bool,
        duration: f64,
        volume: f32,
    }

    impl Shown {
        fn matches(&self, info: &MediaInfo<'_>) -> bool {
            self.title == info.title
                && self.artist == info.artist
                && self.album.as_deref() == info.album
                && self.playing == info.playing
                && (self.duration - info.duration).abs() < 0.5
                && (self.volume - info.volume).abs() < 0.005
        }
    }

    pub struct MediaControls {
        updates: Sender<Update>,
        shown: Option<Shown>,
        last_sent: Option<Instant>,
        /// The position in the last snapshot, for telling a seek from normal playback.
        last_position: f64,
    }

    impl MediaControls {
        pub fn new<F>(_window: &winit::window::Window, on_command: F) -> Option<Self>
        where
            F: Fn(Command) + Send + 'static,
        {
            let (updates, receiver) = mpsc::channel();
            let (commands, command_receiver) = mpsc::sync_channel(64);
            std::thread::Builder::new()
                .name("v1-mpris".to_owned())
                .spawn(move || serve(receiver, commands))
                .ok()?;
            // Transport calls arrive on zbus threads; apply them on the UI's own thread through
            // the callback the host installs.
            std::thread::Builder::new()
                .name("v1-mpris-commands".to_owned())
                .spawn(move || {
                    while let Ok(command) = command_receiver.recv() {
                        on_command(command);
                    }
                })
                .ok()?;
            Some(Self {
                updates,
                shown: None,
                last_sent: None,
                last_position: 0.0,
            })
        }

        pub fn update(&mut self, info: Option<&MediaInfo<'_>>) {
            let Some(info) = info else {
                if self.shown.take().is_some() {
                    let _ = self.updates.send(Update::Clear);
                }
                return;
            };
            // Borrowed comparison: an unchanged frame costs no allocation at all.
            let changed = self.shown.as_ref().is_none_or(|shown| !shown.matches(info));
            let moved = match (self.shown.as_ref(), self.last_sent) {
                (Some(shown), Some(sent)) => {
                    let expected = self.last_position
                        + if shown.playing {
                            sent.elapsed().as_secs_f64()
                        } else {
                            0.0
                        };
                    (info.position - expected).abs() > DRIFT
                }
                _ => true,
            };
            if !changed && !moved {
                return;
            }
            let snapshot = Snapshot {
                title: info.title.to_owned(),
                artist: info.artist.to_owned(),
                album: info.album.map(str::to_owned),
                position: info.position,
                duration: info.duration,
                playing: info.playing,
                volume: info.volume,
            };
            if self.updates.send(Update::Snapshot(snapshot)).is_ok() {
                self.shown = Some(Shown {
                    title: info.title.to_owned(),
                    artist: info.artist.to_owned(),
                    album: info.album.map(str::to_owned),
                    playing: info.playing,
                    duration: info.duration,
                    volume: info.volume,
                });
                self.last_sent = Some(Instant::now());
                self.last_position = info.position;
            }
        }
    }

    impl Drop for MediaControls {
        fn drop(&mut self) {
            let _ = self.updates.send(Update::Stop);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn positions_convert_to_microseconds() {
            assert_eq!(seconds_to_micros(0.0), 0);
            assert_eq!(seconds_to_micros(-1.0), 0);
            assert_eq!(seconds_to_micros(f64::NAN), 0);
            assert_eq!(seconds_to_micros(1.5), 1_500_000);
        }

        #[test]
        fn metadata_carries_the_track_fields_mpris_clients_read() {
            let shared = Shared {
                track: Some(Track {
                    title: "Around the World".to_owned(),
                    artist: "Daft Punk".to_owned(),
                    album: Some("Homework".to_owned()),
                    duration: 429.0,
                    position: 0.0,
                    playing: true,
                    at: Instant::now(),
                    id: 7,
                }),
                volume: 0.8,
                next_id: 7,
            };
            let metadata = metadata_of(&shared);
            assert_eq!(
                metadata.get("mpris:length"),
                Some(&OwnedValue::from(429_000_000i64))
            );
            let title: String = metadata["xesam:title"].clone().try_into().unwrap();
            assert_eq!(title, "Around the World");
            let artists: Vec<String> = metadata["xesam:artist"].clone().try_into().unwrap();
            assert_eq!(artists, ["Daft Punk"]);
            let track_id: OwnedObjectPath = metadata["mpris:trackid"].clone().try_into().unwrap();
            assert_eq!(track_id.as_str(), "/org/v1/track/7");
        }

        #[test]
        fn stopped_metadata_names_the_no_track_path() {
            let metadata = metadata_of(&Shared::default());
            let track_id: OwnedObjectPath = metadata["mpris:trackid"].clone().try_into().unwrap();
            assert_eq!(track_id.as_str(), NO_TRACK);
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use windows::core::{Result as WinResult, HSTRING};
    use windows::Foundation::{EventRegistrationToken, TimeSpan, TypedEventHandler};
    use windows::Media::{
        MediaPlaybackStatus, MediaPlaybackType, PlaybackPositionChangeRequestedEventArgs,
        SystemMediaTransportControls, SystemMediaTransportControlsButton,
        SystemMediaTransportControlsButtonPressedEventArgs,
        SystemMediaTransportControlsTimelineProperties,
    };
    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::WinRT::{
        ISystemMediaTransportControlsInterop, RoGetActivationFactory, RoInitialize,
        RO_INIT_SINGLETHREADED,
    };
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    use super::{Command, MediaInfo};

    /// How often the timeline is refreshed while playing. The flyout interpolates between updates,
    /// so once a second keeps the scrubber right without crossing into COM on every frame.
    const TIMELINE_EVERY: Duration = Duration::from_secs(1);

    pub struct MediaControls {
        controls: SystemMediaTransportControls,
        button_token: EventRegistrationToken,
        seek_token: Option<EventRegistrationToken>,
        /// The title/artist/album currently pushed to the flyout.
        shown: Option<(String, String, Option<String>)>,
        status: Option<MediaPlaybackStatus>,
        last_timeline: Option<Instant>,
        cleared: bool,
    }

    impl MediaControls {
        /// Attach an SMTC session to `window`. `on_command` runs when the OS sends a transport
        /// button; it may be called from a COM thread, so it must be thread-safe.
        pub fn new<F>(window: &winit::window::Window, on_command: F) -> Option<Self>
        where
            F: Fn(Command) + Send + Sync + 'static,
        {
            let hwnd = window_hwnd(window)?;
            // Winit has already made this thread a single-threaded COM apartment. Matching the
            // apartment keeps RoInitialize from changing it under winit; an error (including
            // "already initialized") is harmless.
            unsafe {
                let _ = RoInitialize(RO_INIT_SINGLETHREADED);
            }
            let class = HSTRING::from("Windows.Media.SystemMediaTransportControls");
            let interop: ISystemMediaTransportControlsInterop =
                unsafe { RoGetActivationFactory(&class) }.ok()?;
            let controls: SystemMediaTransportControls =
                unsafe { interop.GetForWindow(hwnd) }.ok()?;

            let _ = controls.SetIsEnabled(true);
            let _ = controls.SetIsPlayEnabled(true);
            let _ = controls.SetIsPauseEnabled(true);
            let _ = controls.SetIsStopEnabled(true);
            let _ = controls.SetIsNextEnabled(true);
            let _ = controls.SetIsPreviousEnabled(true);
            let _ = controls.SetIsFastForwardEnabled(false);
            let _ = controls.SetIsRewindEnabled(false);
            // Do not take over channel or record buttons.
            let _ = controls.SetIsChannelUpEnabled(false);
            let _ = controls.SetIsChannelDownEnabled(false);
            let _ = controls.SetIsRecordEnabled(false);

            let callback: Arc<dyn Fn(Command) + Send + Sync> = Arc::new(on_command);

            let button_callback = Arc::clone(&callback);
            let handler = TypedEventHandler::<
                SystemMediaTransportControls,
                SystemMediaTransportControlsButtonPressedEventArgs,
            >::new(move |_sender, args| {
                if let Some(args) = args.as_ref() {
                    if let WinResult::Ok(button) = args.Button() {
                        if let Some(command) = command_for(button) {
                            button_callback(command);
                        }
                    }
                }
                Ok(())
            });
            let button_token = controls.ButtonPressed(&handler).ok()?;

            // Seek requests from the media flyout, if the interface is available on this Windows.
            let seek_callback = Arc::clone(&callback);
            let seek_handler = TypedEventHandler::<
                SystemMediaTransportControls,
                PlaybackPositionChangeRequestedEventArgs,
            >::new(move |_sender, args| {
                if let Some(args) = args.as_ref() {
                    if let Ok(position) = args.RequestedPlaybackPosition() {
                        seek_callback(Command::Seek(ticks_to_seconds(position)));
                    }
                }
                Ok(())
            });
            let seek_token = controls.PlaybackPositionChangeRequested(&seek_handler).ok();

            Some(Self {
                controls,
                button_token,
                seek_token,
                shown: None,
                status: None,
                last_timeline: None,
                cleared: false,
            })
        }

        /// Push the current track and position to the media flyout. Cheap to call every frame: only
        /// a track change, a play/pause change, and the once-a-second timeline actually cross COM.
        pub fn update(&mut self, info: Option<&MediaInfo<'_>>) {
            let Some(info) = info else {
                self.clear();
                return;
            };
            self.cleared = false;

            let key = (
                info.title.to_owned(),
                info.artist.to_owned(),
                info.album.map(str::to_owned),
            );
            if self.shown.as_ref() != Some(&key) {
                let mut updated = false;
                if let Ok(updater) = self.controls.DisplayUpdater() {
                    let _ = updater.SetType(MediaPlaybackType::Music);
                    if let Ok(music) = updater.MusicProperties() {
                        let _ = music.SetTitle(&HSTRING::from(info.title));
                        let _ = music.SetArtist(&HSTRING::from(info.artist));
                        if let Some(album) = info.album.filter(|album| !album.is_empty()) {
                            let _ = music.SetAlbumTitle(&HSTRING::from(album));
                        }
                    }
                    updated = updater.Update().is_ok();
                }
                if updated {
                    self.shown = Some(key);
                    // Force a timeline push with the new duration.
                    self.last_timeline = None;
                }
            }

            let status = if info.playing {
                MediaPlaybackStatus::Playing
            } else {
                MediaPlaybackStatus::Paused
            };
            if self.status != Some(status) && self.controls.SetPlaybackStatus(status).is_ok() {
                self.status = Some(status);
            }

            if self
                .last_timeline
                .is_none_or(|at| at.elapsed() >= TIMELINE_EVERY)
            {
                self.set_timeline(info);
                self.last_timeline = Some(Instant::now());
            }
        }

        /// Say that nothing is playing, so the flyout stops showing the last song.
        fn clear(&mut self) {
            if self.cleared {
                return;
            }
            let _ = self
                .controls
                .SetPlaybackStatus(MediaPlaybackStatus::Stopped);
            let mut cleared = true;
            if let Ok(updater) = self.controls.DisplayUpdater() {
                let _ = updater.ClearAll();
                cleared = updater.Update().is_ok();
            }
            if cleared {
                self.shown = None;
                self.status = None;
                self.last_timeline = None;
                self.cleared = true;
            }
        }

        fn set_timeline(&self, info: &MediaInfo<'_>) {
            let Ok(timeline) = SystemMediaTransportControlsTimelineProperties::new() else {
                return;
            };
            let duration = seconds_to_ticks(info.duration);
            let position = seconds_to_ticks(info.position).clamp(0, duration);
            let _ = timeline.SetStartTime(TimeSpan { Duration: 0 });
            let _ = timeline.SetEndTime(TimeSpan { Duration: duration });
            let _ = timeline.SetMinSeekTime(TimeSpan { Duration: 0 });
            let _ = timeline.SetMaxSeekTime(TimeSpan { Duration: duration });
            let _ = timeline.SetPosition(TimeSpan { Duration: position });
            let _ = self.controls.UpdateTimelineProperties(&timeline);
        }
    }

    impl Drop for MediaControls {
        fn drop(&mut self) {
            if let Some(token) = self.seek_token.take() {
                let _ = self.controls.RemovePlaybackPositionChangeRequested(token);
            }
            let _ = self.controls.RemoveButtonPressed(self.button_token);
            let _ = self.controls.SetIsEnabled(false);
        }
    }

    /// The window's native handle, as the interop interface wants it.
    fn window_hwnd(window: &winit::window::Window) -> Option<HWND> {
        let handle = window.window_handle().ok()?;
        match handle.as_raw() {
            RawWindowHandle::Win32(win32) => Some(HWND(win32.hwnd.get() as *mut core::ffi::c_void)),
            _ => None,
        }
    }

    fn command_for(button: SystemMediaTransportControlsButton) -> Option<Command> {
        // The constants are plain i32 wrappers, so compare by value.
        if button == SystemMediaTransportControlsButton::Play {
            Some(Command::Play)
        } else if button == SystemMediaTransportControlsButton::Pause {
            Some(Command::Pause)
        } else if button == SystemMediaTransportControlsButton::Stop {
            Some(Command::Stop)
        } else if button == SystemMediaTransportControlsButton::Next {
            Some(Command::Next)
        } else if button == SystemMediaTransportControlsButton::Previous {
            Some(Command::Previous)
        } else {
            None
        }
    }

    /// 100-nanosecond ticks to seconds.
    fn ticks_to_seconds(time: TimeSpan) -> f64 {
        time.Duration as f64 / 10_000_000.0
    }

    fn seconds_to_ticks(seconds: f64) -> i64 {
        if !seconds.is_finite() || seconds <= 0.0 {
            return 0;
        }
        (seconds * 10_000_000.0).round().clamp(0.0, i64::MAX as f64) as i64
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn transport_buttons_map_to_the_app_s_commands() {
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Play),
                Some(Command::Play)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Pause),
                Some(Command::Pause)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Next),
                Some(Command::Next)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Previous),
                Some(Command::Previous)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Stop),
                Some(Command::Stop)
            );
            // Record, fast-forward, rewind and channel buttons are not ours.
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Record),
                None
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::FastForward),
                None
            );
        }

        #[test]
        fn timeline_positions_convert_between_seconds_and_ticks() {
            assert_eq!(seconds_to_ticks(0.0), 0);
            assert_eq!(seconds_to_ticks(-3.0), 0);
            assert_eq!(seconds_to_ticks(f64::NAN), 0);
            assert_eq!(seconds_to_ticks(1.5), 15_000_000);
            assert!(
                (ticks_to_seconds(TimeSpan {
                    Duration: 15_000_000
                }) - 1.5)
                    .abs()
                    < 1e-9
            );
        }
    }
}
