// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The network side of the app.
//!
//! One tokio runtime on its own thread owns everything that talks to YouTube: the InnerTube
//! client, the player script, the resolver and the loopback range proxy. The UI thread never
//! awaits; it sends [`Command`]s and drains [`Event`]s once per frame.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{watch, Semaphore};
use yt_client::{connect, AlbumPage, ArtistPage, InnerTube, SearchResults, TransportConfig, WebConfig};
use yt_js::Player;
use yt_lyrics::{Lookup, Query, Service as LyricsService};
use yt_proxy::{Config as ProxyConfig, Proxy};
use yt_stream::{FileCache, HttpProber, Resolver, ResolverConfig, YtScripts};

use crate::library::Job;
use crate::store::{cover_key, safe_id};

/// What the UI asks for.
#[derive(Debug, Clone)]
pub enum Command {
    Search(String),
    /// Resolve this video id and hand back a loopback URL ready for playback.
    Play(String),
    /// Resolve this video id ahead of time, so a later [`Command::Play`] finds it cached. Quiet: no
    /// reply is sent, and one at a time, well behind anything the user is waiting for.
    Preload(String),
    /// Cover art, decoded for the rows and the bar.
    Cover(String),
    /// Synced lyrics for a track that has started playing.
    Lyrics(Query),
    /// An artist's channel, by its `UC…` browse id.
    Artist(String),
    /// An album, by its `MPREb_…` browse id.
    Album(String),
    /// The length of a track whose list row gave none.
    Length(String),
    /// Make a saved song's offline copy: 128 kbps Opus under `downloads/`. Quiet and unhurried.
    Download(Job),
    /// Keep a cover's bytes on disk, so a saved item still has its picture offline.
    KeepCover(String),
    Shutdown,
}

/// A resolved track, already registered with the proxy.
#[derive(Debug, Clone)]
pub struct Ready {
    /// Which track this is, so the list can mark it while it plays.
    pub video_id: String,
    /// The loopback URL the player should open. The CDN URL never leaves the backend.
    pub url: String,
    pub artist: String,
    pub duration_secs: Option<f64>,
    /// YouTube's measured loudness relative to its reference level, for volume normalisation.
    pub loudness_db: Option<f64>,
}

/// What the backend reports.
#[derive(Debug)]
pub enum Event {
    /// Bootstrapping finished; search works from here, and play once the player script has loaded.
    Ready,
    /// Something is wrong with the network side. Search or play may still come right later.
    Fatal(String),
    Results(String, SearchResults),
    /// A search failed: the query it was for, and why.
    SearchFailed(String, String),
    Playable(Ready),
    /// A resolve failed: the video id it was for, and why.
    PlayFailed(String, String),
    /// A cover, decoded to square RGBA8 and ready to upload. `url` is the key it was asked for by.
    Cover {
        url: String,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
    /// A cover could not be fetched or decoded. Without this the UI would wait for it for good.
    CoverFailed(String),
    /// The answer to a [`Command::Lyrics`], keyed by the track it was asked for.
    Lyrics { video_id: String, lookup: Lookup },
    /// The answer to a [`Command::Artist`]: the page, or why it could not be read.
    Artist { id: String, result: Result<ArtistPage, String> },
    /// The answer to a [`Command::Album`].
    Album { id: String, result: Result<AlbumPage, String> },
    /// The answer to a [`Command::Length`]: the length in seconds, if YouTube gave one.
    Length { video_id: String, seconds: Option<u64> },
    /// A [`Command::Download`] finished: the file is in place.
    Downloaded {
        video_id: String,
        loudness_db: Option<f64>,
        duration_secs: Option<f64>,
    },
    /// A [`Command::Download`] failed. The library tries again later.
    DownloadFailed { video_id: String, message: String },
}

/// How many length lookups run at once.
const LENGTHS_AT_ONCE: usize = 4;
/// How many covers are fetched and decoded at once. A page of results asks for dozens together;
/// all of them in flight at once only fights the song being loaded for the connection.
const COVERS_AT_ONCE: usize = 8;

/// The bitrate saved songs are stored at.
const DOWNLOAD_BITRATE: &str = "128k";
/// How many songs download at once. Few, and at a low process priority, so it stays unnoticed.
const DOWNLOADS_AT_ONCE: usize = 2;

/// Where a kept cover lives.
fn kept_cover_path(data_dir: &Path, url: &str) -> PathBuf {
    data_dir.join("covers").join(cover_key(url))
}

/// Re-encode `url` (a loopback stream) to Opus at [`DOWNLOAD_BITRATE`] in `part`, then move it to
/// `done`. Runs FFmpeg with no window and at below-normal priority.
fn encode(ffmpeg: &Path, url: &str, part: &Path, done: &Path, job: &Job) -> Result<(), String> {
    let mut command = std::process::Command::new(ffmpeg);
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"])
        .args(["-protocol_whitelist", "file,http,https,tcp,tls"])
        .args(["-reconnect", "1", "-reconnect_streamed", "1"])
        .args(["-reconnect_on_network_error", "1", "-reconnect_delay_max", "2"])
        .args(["-i", url, "-vn", "-map", "0:a:0", "-map_metadata", "-1"])
        .args(["-metadata", &format!("title={}", job.title)])
        .args(["-metadata", &format!("artist={}", job.artist)]);
    if let Some(album) = &job.album {
        command.args(["-metadata", &format!("album={album}")]);
    }
    command
        .args(["-c:a", "libopus", "-b:a", DOWNLOAD_BITRATE, "-vbr", "on", "-application", "audio"])
        .args(["-f", "ogg"])
        .arg(part)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
        command.creation_flags(0x0800_0000 | 0x0000_4000);
    }
    let output = command.output().map_err(|error| format!("could not run ffmpeg: {error}"))?;
    if !output.status.success() {
        let _ = std::fs::remove_file(part);
        let text = String::from_utf8_lossy(&output.stderr);
        let last = text.lines().last().unwrap_or("ffmpeg failed").trim().to_owned();
        return Err(last);
    }
    let _ = std::fs::remove_file(done);
    std::fs::rename(part, done).map_err(|error| error.to_string())
}

/// Cover side in pixels. Rows show it at about 48 logical pixels, the bar at 56, so this leaves
/// room for a 4x display and still keeps a screenful of covers to a few megabytes.
const COVER_SIZE: u32 = 256;

/// The URL to actually fetch: Google's image host takes the size in the URL, and search results
/// carry a tiny one.
fn cover_url(url: &str) -> String {
    let sized_host = url.contains("googleusercontent.com") || url.contains("ggpht.com");
    match url.rfind('=') {
        Some(at) if sized_host && url[at + 1..].starts_with('w') => {
            format!("{}=w{COVER_SIZE}-h{COVER_SIZE}-l90-rj", &url[..at])
        }
        _ => url.to_owned(),
    }
}

async fn fetch_cover(http: &reqwest::Client, data_dir: &Path, url: &str) -> Option<(u32, u32, Vec<u8>)> {
    // A cover kept for a saved item is read from disk, so it shows offline and without a request.
    let bytes = match tokio::fs::read(kept_cover_path(data_dir, url)).await {
        Ok(bytes) => bytes,
        Err(_) => http
            .get(cover_url(url))
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .bytes()
            .await
            .ok()?
            .to_vec(),
    };
    tokio::task::spawn_blocking(move || {
        let image = image::load_from_memory(&bytes).ok()?;
        let square = image.resize_to_fill(COVER_SIZE, COVER_SIZE, image::imageops::FilterType::Triangle);
        Some((COVER_SIZE, COVER_SIZE, square.to_rgba8().into_raw()))
    })
    .await
    .ok()?
}

/// Handle to the backend thread.
pub struct Backend {
    pub commands: UnboundedSender<Command>,
    pub events: Receiver<Event>,
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
    }
}

impl Backend {
    /// Start the runtime. `data_dir` holds the player-script and stream-URL caches.
    pub fn start(data_dir: PathBuf) -> Backend {
        let (commands, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (events, event_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("v1-net".to_owned())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = events.send(Event::Fatal(format!("no async runtime: {error}")));
                        return;
                    }
                };
                runtime.block_on(async move {
                    if let Err(error) = serve(&mut rx, events.clone(), data_dir).await {
                        let _ = events.send(Event::Fatal(error.to_string()));
                    }
                });
            })
            .expect("spawn the network thread");
        Backend {
            commands,
            events: event_rx,
        }
    }
}

/// The resolver as the app builds it.
type AppResolver = Resolver<YtScripts, HttpProber, FileCache>;

/// Where the player script's load stands. Search and covers do not need it, so they are served as
/// soon as the bootstrap is through; only resolving a stream waits for it.
#[derive(Clone)]
enum PlayerState {
    Loading,
    Ready(Arc<AppResolver>),
    Failed(String),
}

/// Everything the tasks share.
struct Net {
    tube: InnerTube,
    http: reqwest::Client,
    proxy: Proxy,
    lyrics: LyricsService,
    player: watch::Receiver<PlayerState>,
    data_dir: PathBuf,
    events: Sender<Event>,
    /// Preloads share one gate, so at most one is in flight at a time, and each waits a moment
    /// before it starts so it never competes with the song that is being loaded now.
    preload_gate: Semaphore,
    /// Length lookups come in bursts (a page of songs at once); a few at a time keeps YouTube from
    /// throttling them into empty answers.
    length_gate: Semaphore,
    download_gate: Semaphore,
    cover_gate: Semaphore,
}

impl Net {
    /// The resolver, waiting for the player script if it is still loading.
    async fn resolver(&self) -> Result<Arc<AppResolver>, String> {
        let mut watcher = self.player.clone();
        let state = watcher
            .wait_for(|state| !matches!(state, PlayerState::Loading))
            .await
            .map_err(|_| "the network thread stopped".to_owned())?
            .clone();
        match state {
            PlayerState::Ready(resolver) => Ok(resolver),
            PlayerState::Failed(message) => Err(message),
            PlayerState::Loading => Err("the player script is not ready".to_owned()),
        }
    }

    async fn search(&self, query: String) {
        let event = match self.tube.search(&query, false).await {
            Ok(hits) => Event::Results(query, hits),
            Err(error) => Event::SearchFailed(query, error.to_string()),
        };
        let _ = self.events.send(event);
    }

    async fn play(&self, video_id: String) {
        let resolved = match self.resolver().await {
            Ok(resolver) => resolver.resolve(&video_id).await.map_err(|e| e.to_string()),
            Err(message) => Err(message),
        };
        let event = match resolved {
            Ok(resolved) => {
                let url = self.proxy.register(&resolved.url, resolved.headers.clone());
                Event::Playable(Ready {
                    // Auto-generated artist channels are named `<artist> - Topic`.
                    artist: resolved
                        .author
                        .unwrap_or_default()
                        .trim_end_matches(" - Topic")
                        .to_owned(),
                    video_id,
                    loudness_db: resolved.loudness_db,
                    // When `/player` leaves the length out, the app falls back to the list's
                    // length and then to the decoder's, both exact. A guess from the file
                    // size is not used: it can be a minute off, and lyrics match on it.
                    duration_secs: resolved.length_seconds.map(|seconds| seconds as f64),
                    url,
                })
            }
            Err(message) => Event::PlayFailed(video_id, message),
        };
        let _ = self.events.send(event);
    }

    async fn preload(&self, video_id: String) {
        tokio::time::sleep(Duration::from_millis(800)).await;
        let Ok(_turn) = self.preload_gate.acquire().await else { return };
        let Ok(resolver) = self.resolver().await else { return };
        // The result lands in the resolver's cache; nothing is sent back.
        let _ = resolver.resolve(&video_id).await;
    }

    async fn cover(&self, url: String) {
        let Ok(_turn) = self.cover_gate.acquire().await else { return };
        let event = match fetch_cover(&self.http, &self.data_dir, &url).await {
            Some((width, height, rgba)) => Event::Cover { url, width, height, rgba },
            None => Event::CoverFailed(url),
        };
        let _ = self.events.send(event);
    }

    async fn lyrics(&self, query: Query) {
        let video_id = query.video_id.clone();
        let lookup = self.lyrics.lookup(&query).await;
        let _ = self.events.send(Event::Lyrics { video_id, lookup });
    }

    async fn length(&self, video_id: String) {
        let Ok(_turn) = self.length_gate.acquire().await else { return };
        let seconds = self.tube.track_length(&video_id).await.ok().flatten();
        let _ = self.events.send(Event::Length { video_id, seconds });
    }

    async fn keep_cover(&self, url: String) {
        let path = kept_cover_path(&self.data_dir, &url);
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            return;
        }
        let Ok(response) = self.http.get(cover_url(&url)).send().await else { return };
        let Ok(response) = response.error_for_status() else { return };
        let Ok(bytes) = response.bytes().await else { return };
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::write(&path, &bytes).await;
    }

    async fn download(&self, job: Job) {
        let video_id = job.video_id.clone();
        let Ok(turn) = self.download_gate.acquire().await else { return };
        let for_lyrics = job.clone();
        let outcome: Result<(Option<f64>, Option<f64>), String> = async {
            if !safe_id(&video_id) {
                return Err("unusable video id".to_owned());
            }
            let resolver = self.resolver().await?;
            let resolved = resolver.resolve(&video_id).await.map_err(|e| e.to_string())?;
            let url = self.proxy.register(&resolved.url, resolved.headers.clone());
            let dir = self.data_dir.join("downloads");
            tokio::fs::create_dir_all(&dir).await.map_err(|e| e.to_string())?;
            let done = dir.join(format!("{video_id}.opus"));
            let part = dir.join(format!("{video_id}.opus.part"));
            let ffmpeg = dream_media::Tools::discover().map_err(|e| e.to_string())?.ffmpeg;
            tokio::task::spawn_blocking(move || encode(&ffmpeg, &url, &part, &done, &job))
                .await
                .map_err(|e| e.to_string())??;
            Ok((resolved.loudness_db, resolved.length_seconds.map(|s| s as f64)))
        }
        .await;
        // The download slot is free before the lyrics go out, so a lyric lookup never holds it.
        drop(turn);
        let _ = self.events.send(match &outcome {
            Ok((loudness_db, duration_secs)) => Event::Downloaded {
                video_id: video_id.clone(),
                loudness_db: *loudness_db,
                duration_secs: *duration_secs,
            },
            Err(message) => Event::DownloadFailed { video_id: video_id.clone(), message: message.clone() },
        });
        // A saved song keeps its lyrics with it: fetch them now, while the length is known, so they
        // play offline. The lookup caches what it finds under the video id (see `yt_lyrics::Cache`).
        if let Ok((_, duration_secs)) = outcome {
            self.keep_lyrics(for_lyrics, duration_secs).await;
        }
    }

    /// Fetch and keep a saved song's lyrics. Playback reads the cache first, so these load with no
    /// network. Failures are ignored: the song still plays, and its lyrics are asked for again then.
    async fn keep_lyrics(&self, job: Job, duration_secs: Option<f64>) {
        let query = Query {
            video_id: job.video_id,
            title: job.title,
            // Playback looks up by the primary artist, so this does too.
            artist: job.artist.split(", ").next().unwrap_or_default().to_owned(),
            album: job.album,
            duration_secs,
        };
        let _ = self.lyrics.lookup(&query).await;
    }
}

/// Answer a command that cannot be served (the bootstrap has not got through), so nothing waits on
/// a reply that will never come.
fn refuse(command: Command, events: &Sender<Event>, message: &str) {
    let message = message.to_owned();
    let _ = match command {
        Command::Search(query) => events.send(Event::SearchFailed(query, message)),
        Command::Play(video_id) => events.send(Event::PlayFailed(video_id, message)),
        Command::Lyrics(query) => events.send(Event::Lyrics {
            video_id: query.video_id,
            lookup: Lookup::Unavailable(message),
        }),
        Command::Artist(id) => events.send(Event::Artist { id, result: Err(message) }),
        Command::Album(id) => events.send(Event::Album { id, result: Err(message) }),
        Command::Length(video_id) => events.send(Event::Length { video_id, seconds: None }),
        Command::Download(job) => events.send(Event::DownloadFailed { video_id: job.video_id, message }),
        Command::Cover(url) => events.send(Event::CoverFailed(url)),
        Command::Preload(_) | Command::KeepCover(_) | Command::Shutdown => Ok(()),
    };
}

/// Bootstrap against YouTube Music, trying again with a growing wait until it works. Starting the
/// app with no network (or just before it comes up) must not leave search dead for the whole run.
/// Commands that arrive meanwhile are refused at once. `None` when the app closed first.
async fn connect_with_retry(
    commands: &mut UnboundedReceiver<Command>,
    events: &Sender<Event>,
) -> Option<(InnerTube, WebConfig)> {
    let mut wait = Duration::from_secs(1);
    loop {
        let error = match connect(TransportConfig::default(), Default::default()).await {
            Ok(connected) => return Some(connected),
            Err(error) => error,
        };
        let message = format!("Could not reach YouTube Music: {error}");
        let _ = events.send(Event::Fatal(message.clone()));
        let retry = tokio::time::sleep(wait);
        tokio::pin!(retry);
        loop {
            tokio::select! {
                () = &mut retry => break,
                command = commands.recv() => match command {
                    None | Some(Command::Shutdown) => return None,
                    Some(command) => refuse(command, events, &message),
                },
            }
        }
        wait = (wait * 2).min(Duration::from_secs(15));
    }
}

/// Load the player script and build the resolver around it.
async fn build_resolver(
    http: &reqwest::Client,
    data_dir: &Path,
    tube: InnerTube,
    player_path: Option<String>,
) -> Result<AppResolver, String> {
    let player_path = player_path.ok_or("the homepage named no player script")?;
    let player = Player::load(http, &data_dir.join("player"), &player_path)
        .await
        .map_err(|e| e.to_string())?;
    // The stream probe checks what the range proxy will later fetch, so it speaks the same
    // HTTP/1.1 (see the proxy's upstream client).
    let prober = HttpProber::new(
        reqwest::Client::builder()
            .no_proxy()
            .http1_only()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| e.to_string())?,
    );
    // Reading the stream-URL cache is file I/O; keep it off the async workers.
    let cache_path = data_dir.join("streams.json");
    let cache = tokio::task::spawn_blocking(move || FileCache::open(cache_path))
        .await
        .map_err(|e| e.to_string())?;
    Ok(Resolver::new(
        tube,
        YtScripts::new(player),
        prober,
        cache,
        ResolverConfig::default(),
    ))
}

async fn serve(
    commands: &mut UnboundedReceiver<Command>,
    events: Sender<Event>,
    data_dir: PathBuf,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // The proxy needs a client with no total timeout; it enforces its own per-read stall limit. It
    // also speaks HTTP/1.1 only: bounded ranges at full speed are what was measured against
    // YouTube's servers, and nothing says the same holds over HTTP/2.
    let upstream = reqwest::Client::builder()
        .no_proxy()
        .http1_only()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    // The proxy binds a loopback port and needs nothing from the network, so it starts beside the
    // bootstrap instead of after it.
    let (proxy, connected) = tokio::join!(
        Proxy::start(upstream, ProxyConfig::default()),
        connect_with_retry(commands, &events)
    );
    let proxy = proxy?;
    let Some((tube, web)) = connected else { return Ok(()) };

    // For covers and lyrics. Their hosts take HTTP/2, so a screenful of covers shares a connection.
    // A total timeout: a cover or a lyric that stalls would otherwise hold its task, and whatever
    // on screen waits for it, for good.
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build()?;
    // The player script comes from www.youtube.com, which resets HTTP/2 requests from this client.
    let script_http = reqwest::Client::builder()
        .http1_only()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .build()?;

    let (player_tx, player_rx) = watch::channel(PlayerState::Loading);
    let net = Arc::new(Net {
        tube: tube.clone(),
        // Lyrics keep their own cache next to the others. Lookups run as tasks, so a slow provider
        // can never hold up a search or a resolve.
        lyrics: LyricsService::new(http.clone(), Some(data_dir.join("lyrics"))),
        http: http.clone(),
        proxy,
        player: player_rx,
        data_dir: data_dir.clone(),
        events: events.clone(),
        preload_gate: Semaphore::new(1),
        length_gate: Semaphore::new(LENGTHS_AT_ONCE),
        download_gate: Semaphore::new(DOWNLOADS_AT_ONCE),
        cover_gate: Semaphore::new(COVERS_AT_ONCE),
    });

    // Search, pages, covers and lyrics work from here. The player script (a download on a first
    // run, then a parse and an engine start) loads behind them, and only a play waits for it.
    let _ = events.send(Event::Ready);
    tokio::spawn({
        let events = events.clone();
        async move {
            let state = match build_resolver(&script_http, &data_dir, tube, web.player_js_path).await {
                Ok(resolver) => PlayerState::Ready(Arc::new(resolver)),
                Err(message) => {
                    let _ = events.send(Event::Fatal(message.clone()));
                    PlayerState::Failed(message)
                }
            };
            let _ = player_tx.send(state);
        }
    });

    // Every command runs as a task of its own, so a slow search never holds up a play, a cover or
    // a length. The newest search and the newest play replace the one before: the older answer
    // could only be thrown away, and typing ahead or skipping fast would otherwise queue them all.
    let mut search_task: Option<tokio::task::AbortHandle> = None;
    let mut play_task: Option<tokio::task::AbortHandle> = None;
    while let Some(command) = commands.recv().await {
        let net = net.clone();
        match command {
            Command::Search(query) => {
                if let Some(old) = search_task.take() {
                    old.abort();
                }
                search_task = Some(tokio::spawn(async move { net.search(query).await }).abort_handle());
            }
            Command::Play(video_id) => {
                if let Some(old) = play_task.take() {
                    old.abort();
                }
                play_task = Some(tokio::spawn(async move { net.play(video_id).await }).abort_handle());
            }
            Command::Preload(video_id) => {
                tokio::spawn(async move { net.preload(video_id).await });
            }
            // Covers must not queue behind a search or a resolve, so each gets its own task.
            Command::Cover(url) => {
                tokio::spawn(async move { net.cover(url).await });
            }
            Command::Lyrics(query) => {
                tokio::spawn(async move { net.lyrics(query).await });
            }
            // Pages are fetched alongside whatever else is happening, like covers and lyrics.
            Command::Artist(id) => {
                tokio::spawn(async move {
                    let result = net.tube.artist_page(&id).await.map_err(|error| error.to_string());
                    let _ = net.events.send(Event::Artist { id, result });
                });
            }
            Command::Album(id) => {
                tokio::spawn(async move {
                    let result = net.tube.album_page(&id).await.map_err(|error| error.to_string());
                    let _ = net.events.send(Event::Album { id, result });
                });
            }
            Command::Length(video_id) => {
                tokio::spawn(async move { net.length(video_id).await });
            }
            Command::KeepCover(url) => {
                tokio::spawn(async move { net.keep_cover(url).await });
            }
            Command::Download(job) => {
                tokio::spawn(async move { net.download(job).await });
            }
            Command::Shutdown => break,
        }
    }
    Ok(())
}
