//! The playback engine: session lifecycle, clock, frame selection, audio sync
//! and seeks. It has no GPU dependency; [`crate::VideoPlayer`] is the thin GPU
//! wrapper.

use crate::audio::AudioOutput;
use crate::clock::Clock;
use crate::error::{Error, Result};
use crate::geom::fit_box;
use crate::probe::MediaInfo;
use crate::process::Stream;
use crate::stream::{
    VideoConfig, VideoMsg, VideoSession, feed_audio, spawn_audio, spawn_video, stream_failure,
};
use crate::tools::Tools;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::mpsc::{self, Sender, SyncSender, TryRecvError};
use std::time::{Duration, Instant};

/// Largest decoded frame, whatever the caller asks for.
const MAX_DECODE_W: u32 = 3840;
const MAX_DECODE_H: u32 = 2160;
/// Seeks closer together than this are coalesced, so scrubbing does not start
/// a decoder for every pointer move.
const SEEK_COALESCE: Duration = Duration::from_millis(120);
/// Audio that must be buffered before a session starts its clock.
const PRIME_AUDIO_MS: usize = 60;
/// Longest a session waits for audio before starting anyway.
const PRIME_TIMEOUT: Duration = Duration::from_millis(750);
/// A session that produces no frame for this long has failed.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(15);
/// An exact seek decodes everything before the position.
const EXACT_FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(60);
/// Poll cadence while audio is still priming.
const AUDIO_POLL: Duration = Duration::from_millis(10);
/// Poll cadence for an audio-only tail.
const AUDIO_TAIL_POLL: Duration = Duration::from_millis(50);

const NO_FRAME: &str = "This video could not be decoded";
const NO_AUDIO_DEVICE: &str = "No audio output device is available";
const NO_AUDIO_DECODED: &str = "The audio stream gave no sound";

/// Options for [`Player::open`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerOptions {
    /// Bounding box for decoded video. It never enlarges, and is clamped to
    /// 3840x2160. Zero for either side means the cap for that side.
    pub max_size: (u32, u32),
    /// Decode and play the audio track. A video still plays silently when no
    /// output device exists.
    pub audio: bool,
    /// Restart at zero when the file ends.
    pub looping: bool,
    /// Let FFmpeg try a hardware decoder, with a software retry if it produces
    /// no frame. Off by default: on the development machine an `auto`
    /// hardware decoder added ~120 ms to the first frame and decoded the same
    /// 1080p H.264 clip at half the software rate. Enable it for 4K or a
    /// CPU-constrained machine where the trade-off goes the other way.
    pub hwaccel: bool,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        Self {
            max_size: (1920, 1080),
            audio: true,
            looping: false,
            hwaccel: false,
        }
    }
}

/// Current playback state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Probing is done but the first frame (or audio prime) has not arrived.
    Loading,
    /// The clock is advancing.
    Playing,
    /// The clock is held, with the current frame on screen.
    Paused,
    /// The file reached its end and is not looping.
    Ended,
    /// Decoding could not continue; the string is user-facing.
    Failed(String),
}

/// One decoded RGBA frame. Dropping it hands its pixel buffer back to the
/// decoder thread for reuse; do not copy it into long-lived state without
/// cloning the pixels you need.
pub struct Frame {
    width: u32,
    height: u32,
    pts_ms: i64,
    rgba: Vec<u8>,
    recycle: Option<SyncSender<Vec<u8>>>,
}

impl Frame {
    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `(width, height)` in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Timestamp relative to the start of the file, in milliseconds.
    pub fn pts_ms(&self) -> i64 {
        self.pts_ms
    }

    /// Tightly packed, top-row-first RGBA8.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pts_ms", &self.pts_ms)
            .field("bytes", &self.rgba.len())
            .finish()
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if let Some(recycle) = self.recycle.take() {
            // A full pool means the reader already has enough buffers; this
            // one can simply be freed. `try_send` keeps the drop allocation-free.
            let _ = recycle.try_send(std::mem::take(&mut self.rgba));
        }
    }
}

struct Session {
    start_ms: i64,
    video: Option<VideoSession>,
    /// Held to keep the audio decoder alive; its stdout belongs to the feeder.
    audio: Option<Stream>,
    /// The next frame, held until its timestamp comes due.
    pending: Option<(i64, Vec<u8>)>,
    got_frame: bool,
    /// The video reader reached end of stream, or there is no video.
    ended: bool,
    hwaccel: bool,
    exact: bool,
    started_at: Instant,
    primed: bool,
    audio_running: bool,
    last_played: u64,
}

impl Session {
    fn first_frame_timeout(&self) -> Duration {
        if self.exact {
            EXACT_FIRST_FRAME_TIMEOUT
        } else {
            FIRST_FRAME_TIMEOUT
        }
    }

    fn pull(&mut self) -> Option<(i64, Vec<u8>)> {
        let message = self.video.as_ref().map(|video| video.frames.try_recv());
        match message {
            Some(Ok(VideoMsg::Frame { pts_ms, rgba })) => {
                self.got_frame = true;
                Some((pts_ms, rgba))
            }
            Some(Ok(VideoMsg::End)) | Some(Err(TryRecvError::Disconnected)) => {
                self.ended = true;
                None
            }
            Some(Err(TryRecvError::Empty)) | None => None,
        }
    }
}

/// A media player for video with optional sound, or audio alone.
///
/// Opening probes the file and starts its decoders on the calling thread, which
/// takes tens of milliseconds and, for the first file with sound in a process,
/// the one-time cost of starting the audio device. Open on a loading thread if
/// that budget matters. `poll` is cheap and non-blocking.
pub struct Player {
    source: PathBuf,
    tools: Tools,
    info: MediaInfo,
    options: PlayerOptions,
    decode: (u32, u32),
    clock: Clock,
    user_playing: bool,
    ended: bool,
    failure: Option<String>,
    warning: Option<String>,
    session: Option<Session>,
    want: Option<i64>,
    hwaccel: bool,
    exact_once: bool,
    audio: Option<AudioOutput>,
    /// The device could not be opened once; do not pay for the attempt again
    /// on every seek.
    audio_unavailable: bool,
    volume: f32,
    muted: bool,
}

impl Player {
    /// Probe `source` and start decoding. Returns an error when the file has no
    /// playable stream, the tools are missing, or an audio-only file has no
    /// output device.
    pub fn open(source: impl AsRef<Path>, options: PlayerOptions) -> Result<Self> {
        Self::open_with(&Tools::discover()?, source, options)
    }

    /// Open with explicit FFmpeg paths.
    pub fn open_with(
        tools: &Tools,
        source: impl AsRef<Path>,
        options: PlayerOptions,
    ) -> Result<Self> {
        let source = source.as_ref().to_path_buf();
        let info = crate::probe::probe_with(tools, &source)?;
        Self::from_info(tools, source, options, info)
    }

    /// Open a source the caller already knows is a single audio stream, without probing it.
    ///
    /// The probe is a process launch and a full open of the stream before the decoder opens it
    /// again, which for a network source is a whole extra round trip. A caller that has just
    /// resolved the stream itself (v1 knows it asked for audio) has nothing left to learn from it.
    /// `duration_ms` is reported by [`Player::duration_ms`] as given, `None` when unknown. A source
    /// that turns out not to be playable fails through [`Player::status`] instead of here.
    pub fn open_audio_unprobed(
        source: impl AsRef<Path>,
        options: PlayerOptions,
        duration_ms: Option<i64>,
    ) -> Result<Self> {
        let info = MediaInfo {
            width: 0,
            height: 0,
            fps: 0.0,
            duration_ms,
            has_video: false,
            has_audio: true,
            format: String::new(),
        };
        Self::from_info(&Tools::discover()?, source.as_ref().to_path_buf(), options, info)
    }

    fn from_info(
        tools: &Tools,
        source: PathBuf,
        options: PlayerOptions,
        info: MediaInfo,
    ) -> Result<Self> {
        let cap = (
            bounded(options.max_size.0, MAX_DECODE_W),
            bounded(options.max_size.1, MAX_DECODE_H),
        );
        let decode = if info.has_video {
            fit_box(info.width, info.height, cap.0, cap.1)
        } else {
            (0, 0)
        };
        if !info.has_video && !options.audio {
            return Err(Error::msg(
                "the file has no video and audio playback is disabled",
            ));
        }
        let volume = 1.0;
        let hwaccel = options.hwaccel;
        let mut player = Self {
            source,
            tools: tools.clone(),
            info,
            options,
            decode,
            clock: Clock::new(0),
            user_playing: true,
            ended: false,
            failure: None,
            warning: None,
            session: None,
            want: None,
            hwaccel,
            exact_once: false,
            audio: None,
            audio_unavailable: false,
            volume,
            muted: false,
        };
        player.begin(0, Instant::now());
        if let Some(message) = player.failure.clone() {
            return Err(Error::msg(message));
        }
        Ok(player)
    }

    /// The probed metadata.
    pub fn info(&self) -> &MediaInfo {
        &self.info
    }

    /// A non-fatal problem, such as an audio track that could not be decoded
    /// while the video still plays. `poll` keeps working; a host can surface
    /// this once. Cleared when a new session starts cleanly.
    pub fn warning(&self) -> Option<&str> {
        self.warning.as_deref()
    }

    /// Duration in milliseconds, when the container reports it.
    pub fn duration_ms(&self) -> Option<i64> {
        self.info.duration_ms
    }

    /// The requested position while a seek is pending, so a scrub bar does not
    /// snap back to the old session.
    pub fn position_ms(&self, now: Instant) -> i64 {
        self.want.unwrap_or_else(|| self.clock.position(now))
    }

    /// Current playback state.
    pub fn status(&self) -> Status {
        if let Some(message) = &self.failure {
            return Status::Failed(message.clone());
        }
        if self.ended {
            return Status::Ended;
        }
        match &self.session {
            Some(session) if session.primed && self.user_playing => Status::Playing,
            Some(session) if session.primed => Status::Paused,
            _ => Status::Loading,
        }
    }

    /// Resume, or restart from zero after the end.
    pub fn play(&mut self, now: Instant) {
        if self.failure.is_some() || self.user_playing {
            return;
        }
        if self.ended {
            self.ended = false;
            self.user_playing = true;
            self.want = None;
            self.begin(0, now);
            return;
        }
        self.user_playing = true;
        if self.session.as_ref().is_some_and(|session| session.primed) {
            self.clock.start(now);
            self.set_audio_running(true);
        }
    }

    /// Hold the current frame and stop audio output.
    pub fn pause(&mut self, now: Instant) {
        self.user_playing = false;
        if self.failure.is_some() {
            return;
        }
        // Take the audio position first, so the held frame is the one the
        // listener last heard.
        self.sync_audio_clock(now);
        self.clock.stop(now);
        self.set_audio_running(false);
    }

    /// Pause if playing, play if paused.
    pub fn toggle(&mut self, now: Instant) {
        if self.user_playing {
            self.pause(now);
        } else {
            self.play(now);
        }
    }

    /// Clamped to the file's duration. Cheap to call while dragging: the last
    /// request wins and only one decoder is restarted per coalescing window.
    pub fn seek(&mut self, ms: i64) {
        if self.failure.is_some() {
            return;
        }
        let upper = self.duration_ms().unwrap_or(i64::MAX);
        self.want = Some(ms.clamp(0, upper));
    }

    /// Set output gain, clamped to `0..=1`.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = if volume.is_finite() {
            volume.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if let Some(audio) = &self.audio {
            audio.shared().set_volume(self.volume);
        }
    }

    /// Current output gain.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Silence output without stopping the clock or the decode stream.
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        if let Some(audio) = &self.audio {
            audio.shared().set_muted(muted);
        }
    }

    /// Whether output is muted.
    pub fn muted(&self) -> bool {
        self.muted
    }

    /// Level of the low, mid and high bands of the sound being output right
    /// now, each 0 to 1. All zero when there is no audio output, or while
    /// paused, muted or silent.
    pub fn levels(&self) -> [f32; 3] {
        self.audio
            .as_ref()
            .map_or([0.0; 3], |audio| audio.shared().levels())
    }

    /// Copy the newest `out.len()` mono samples of the sound being output
    /// (up to 4096), oldest first, for a spectrum or waveform display, and
    /// return the output's sample rate. Returns `None`, leaving `out`
    /// untouched, when there is no audio output.
    pub fn scope(&self, out: &mut [f32]) -> Option<u32> {
        let audio = self.audio.as_ref()?;
        audio.shared().scope(out);
        Some(audio.rate())
    }

    /// Change the decode bounding box. Restarts the current position when the
    /// decoded size actually changes; a repeated call with the same size is a
    /// no-op.
    pub fn set_max_size(&mut self, max_size: (u32, u32), now: Instant) {
        if !self.info.has_video {
            return;
        }
        let cap = (
            bounded(max_size.0, MAX_DECODE_W),
            bounded(max_size.1, MAX_DECODE_H),
        );
        let decode = fit_box(self.info.width, self.info.height, cap.0, cap.1);
        if decode == self.decode {
            return;
        }
        self.decode = decode;
        if self.ended || self.failure.is_some() {
            return;
        }
        let position = self.want.take().unwrap_or_else(|| self.clock.position(now));
        self.begin(position.max(0), now);
    }

    /// Advance to `now`, returning a frame when a new one is due.
    pub fn poll(&mut self, now: Instant) -> Option<Frame> {
        if self.failure.is_some() {
            return None;
        }
        self.apply_seek(now);
        self.sync_audio_clock(now);
        let frame = self.select_frame(now);
        self.check_priming(now);
        self.check_end(now);
        frame
    }

    /// When `poll` should run next, if time alone will produce work. `None`
    /// means only a decoder wake-up (which this API does not expose) or a new
    /// interaction can change anything; a normal frame loop can just poll
    /// every frame.
    pub fn next_wake(&self, now: Instant) -> Option<Instant> {
        if self.failure.is_some() || self.ended {
            return None;
        }
        let session = self.session.as_ref()?;
        let mut soonest: Option<Instant> = None;
        let mut consider = |at: Instant| {
            let at = at.max(now);
            soonest = Some(soonest.map_or(at, |current| current.min(at)));
        };
        if self.want.is_some() {
            let at = if session.got_frame {
                now
            } else {
                session.started_at + SEEK_COALESCE
            };
            consider(at);
        }
        if !session.primed {
            consider(now + AUDIO_POLL);
        } else if session.audio_running && (session.video.is_none() || session.ended) {
            consider(now + AUDIO_TAIL_POLL);
        }
        if session.video.is_some() && !session.got_frame {
            consider(session.started_at + session.first_frame_timeout());
        }
        if let Some((pts_ms, _)) = &session.pending
            && self.clock.running()
        {
            let wait = (*pts_ms - self.clock.position(now)).max(0);
            consider(now + Duration::from_millis(wait as u64));
        }
        soonest
    }

    fn begin(&mut self, start_ms: i64, now: Instant) {
        self.end_session();
        self.warning = None;
        self.ended = false;
        self.clock = Clock::new(start_ms);
        let exact = std::mem::take(&mut self.exact_once);
        let has_video = self.info.has_video;
        let has_audio = self.info.has_audio;

        let mut video = None;
        if has_video {
            let cfg = VideoConfig {
                width: self.decode.0,
                height: self.decode.1,
                fps: self.info.fps,
                hwaccel: self.hwaccel,
                exact_seek: exact,
            };
            match spawn_video(&self.tools, &self.source, start_ms, &cfg) {
                Ok(session) => video = Some(session),
                Err(error) => {
                    self.fail(error.to_string());
                    return;
                }
            }
        }

        // The output device is opened here, after the video decoder has
        // started, rather than before it: the first CoreAudio call on macOS
        // costs ~90 ms and the picture can be decoding under it. An audio-only
        // file still fails through `self.fail` and `open` still returns Err.
        if has_audio && self.options.audio && self.audio.is_none() {
            if self.audio_unavailable {
                if !has_video {
                    self.fail(NO_AUDIO_DEVICE.into());
                    return;
                }
            } else {
                self.audio = AudioOutput::open();
                if self.audio.is_none() {
                    self.audio_unavailable = true;
                    if !has_video {
                        self.fail(NO_AUDIO_DEVICE.into());
                        return;
                    }
                }
            }
        }
        if let Some(audio) = &self.audio {
            audio.shared().set_volume(self.volume);
            audio.shared().set_muted(self.muted);
        }

        let mut audio = None;
        if has_audio && let Some(out) = &self.audio {
            let shared = out.shared();
            shared.set_running(false);
            let generation = shared.reset();
            let channels = out.channels();
            match spawn_audio(&self.tools, &self.source, start_ms, out.rate(), channels) {
                Ok(mut stream) => {
                    if let Some(stdout) = stream.stdout() {
                        let channels = out.channels();
                        std::thread::spawn(move || {
                            feed_audio(stdout, shared, generation, channels);
                        });
                        audio = Some(stream);
                    } else if !has_video {
                        self.fail("audio decoder has no output pipe".into());
                        return;
                    }
                }
                Err(error) => {
                    if !has_video {
                        self.fail(error.to_string());
                        return;
                    }
                    // Video still plays on the wall clock; a library has no
                    // business writing to stderr, so the host reports this.
                    self.warning = Some(format!("audio decoder did not start: {error}"));
                }
            }
        }

        self.session = Some(Session {
            start_ms,
            video,
            audio,
            pending: None,
            got_frame: false,
            ended: !has_video,
            hwaccel: self.hwaccel,
            exact,
            started_at: now,
            primed: false,
            audio_running: false,
            last_played: 0,
        });
    }

    fn end_session(&mut self) {
        if let Some(session) = self.session.take() {
            retire(DeadStreams {
                video: session.video,
                audio: session.audio,
            });
        }
        if let Some(audio) = &self.audio {
            let shared = audio.shared();
            shared.set_running(false);
            shared.reset();
        }
    }

    fn fail(&mut self, message: String) {
        self.warning = None;
        self.end_session();
        self.user_playing = false;
        self.ended = false;
        self.failure = Some(message);
    }

    fn set_audio_running(&mut self, on: bool) {
        let has_stream = self
            .session
            .as_ref()
            .is_some_and(|session| session.audio.is_some());
        if let Some(session) = self.session.as_mut() {
            session.audio_running = on && has_stream;
        }
        if let Some(audio) = &self.audio {
            audio.shared().set_running(on && has_stream);
        }
    }

    fn sync_audio_clock(&mut self, now: Instant) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if !session.audio_running {
            return;
        }
        let Some(audio) = &self.audio else {
            return;
        };
        let shared = audio.shared();
        let played = shared.played_frames();
        if played == session.last_played {
            return;
        }
        session.last_played = played;
        let rate = u64::from(audio.rate());
        let at_ms = session.start_ms + (played * 1000 / rate.max(1)) as i64;
        self.clock.set(at_ms, now);
    }

    fn apply_seek(&mut self, now: Instant) {
        let Some(ms) = self.want else {
            return;
        };
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let coalesced = now.saturating_duration_since(session.started_at) >= SEEK_COALESCE;
        if !(session.got_frame || coalesced) {
            return;
        }
        self.want = None;
        self.begin(ms, now);
    }

    fn check_priming(&mut self, now: Instant) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        if session.primed {
            return;
        }
        let video_ready = session.video.is_none() || session.got_frame;
        let elapsed = now.saturating_duration_since(session.started_at);
        let audio_ready = match (&session.audio, &self.audio) {
            (Some(_), Some(audio)) => {
                let needed = audio.rate() as usize * PRIME_AUDIO_MS / 1000;
                audio.shared().buffered_frames() >= needed || elapsed >= PRIME_TIMEOUT
            }
            _ => true,
        };
        if !(video_ready && audio_ready) {
            return;
        }
        if let Some(session) = self.session.as_mut() {
            session.primed = true;
        }
        if self.user_playing {
            self.clock.start(now);
            self.set_audio_running(true);
        }
    }

    /// Drain due frames. The newest due frame wins; passed-over frames go
    /// straight back to the reader, which is how a slow caller drops frames
    /// instead of falling behind.
    fn select_frame(&mut self, now: Instant) -> Option<Frame> {
        let position = self.clock.position(now);
        let (width, height) = self.decode;
        let session = self.session.as_mut()?;
        let sender = session.video.as_ref().map(|video| video.pool.clone());
        let mut winner: Option<(i64, Vec<u8>)> = None;
        loop {
            if session.pending.is_none() {
                session.pending = session.pull();
            }
            let due = matches!(&session.pending, Some((pts, _)) if *pts <= position);
            if !due {
                break;
            }
            let Some(frame) = session.pending.take() else {
                break;
            };
            if let Some((_, skipped)) = winner.replace(frame)
                && let Some(sender) = &sender
            {
                let _ = sender.try_send(skipped);
            }
        }
        winner.map(|(pts_ms, rgba)| Frame {
            width,
            height,
            pts_ms,
            rgba,
            recycle: sender,
        })
    }

    fn check_end(&mut self, now: Instant) {
        enum Next {
            Nothing,
            Restart(i64),
            Exact(i64),
            Fail(String),
            End,
        }
        let next = {
            let Some(session) = self.session.as_ref() else {
                return;
            };
            let has_video = session.video.is_some();
            let timed_out = has_video
                && !session.got_frame
                && now.saturating_duration_since(session.started_at)
                    >= session.first_frame_timeout();
            let decoded_nothing = has_video && !session.got_frame && session.ended;
            if timed_out || decoded_nothing {
                if session.hwaccel {
                    Next::Restart(session.start_ms)
                } else if decoded_nothing && session.start_ms > 0 && !session.exact {
                    Next::Exact(session.start_ms)
                } else if decoded_nothing && session.start_ms > 0 {
                    Next::End
                } else {
                    Next::Fail(stream_failure(
                        session.video.as_ref().map(|video| &video.stream),
                        NO_FRAME,
                    ))
                }
            } else {
                let audio_drained = match (&session.audio, &self.audio) {
                    (Some(_), Some(audio)) => {
                        let shared = audio.shared();
                        shared.finished() && shared.buffered_frames() == 0
                    }
                    _ => true,
                };
                let clock_settled = self.clock.running() || !self.user_playing;
                if session.ended && session.pending.is_none() && clock_settled && audio_drained {
                    // An audio-only session from the start that drained without one frame ever
                    // played was a source that gave nothing (a refused or dead stream), not a
                    // track that ended. Say so, instead of letting a caller read it as a song
                    // that finished at once.
                    let nothing_decoded = !has_video
                        && session.start_ms == 0
                        && self
                            .audio
                            .as_ref()
                            .is_some_and(|audio| session.audio.is_some() && audio.shared().played_frames() == 0);
                    if nothing_decoded {
                        Next::Fail(stream_failure(session.audio.as_ref(), NO_AUDIO_DECODED))
                    } else {
                        Next::End
                    }
                } else {
                    Next::Nothing
                }
            }
        };
        match next {
            Next::Nothing => {}
            Next::Restart(start_ms) => {
                self.hwaccel = false;
                self.begin(start_ms, now);
            }
            Next::Exact(start_ms) => {
                self.exact_once = true;
                self.begin(start_ms, now);
            }
            Next::Fail(message) => self.fail(message),
            Next::End => {
                self.clock.stop(now);
                self.set_audio_running(false);
                if self.options.looping {
                    self.begin(0, now);
                } else {
                    self.ended = true;
                    self.user_playing = false;
                }
            }
        }
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.end_session();
    }
}

/// Move retired decoder processes off the UI thread: dropping a `Child` waits
/// for it to exit, and a seek should not.
struct DeadStreams {
    video: Option<VideoSession>,
    audio: Option<Stream>,
}

/// One reaper for the whole process rather than one per player: killing and
/// reaping FFmpeg children is not player state, and a media service may hold
/// dozens of players at once.
fn retire(dead: DeadStreams) {
    static REAPER: OnceLock<Sender<DeadStreams>> = OnceLock::new();
    let sender = REAPER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<DeadStreams>();
        std::thread::Builder::new()
            .name("dream-media-reaper".into())
            .spawn(move || {
                while let Ok(dead) = receiver.recv() {
                    let DeadStreams { video, audio } = dead;
                    drop(video);
                    drop(audio);
                }
            })
            .expect("spawn media reaper");
        sender
    });
    if let Err(error) = sender.send(dead) {
        drop(error.0);
    }
}

fn bounded(value: u32, cap: u32) -> u32 {
    if value == 0 { cap } else { value.min(cap) }
}
