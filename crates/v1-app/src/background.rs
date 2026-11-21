// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The custom background: a picture, GIF, or looping video behind the whole app, softened into
//! frosted glass.
//!
//! Decoding and blurring happen on a thread of their own, so opening a large photo or playing a
//! looping video never stalls a frame. Static pictures are decoded once and kept, downscaled, so
//! moving the strength slider only re-blurs what is already in memory. Animated GIFs and videos
//! stream frames from FFmpeg (with audio stripped on upload) and apply live Gaussian frost.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The working size of a static picture: long edge, in pixels.
const MAX_SIDE: u32 = 1280;
/// The working size of an animated GIF or video background: long edge, in pixels.
const MAX_VIDEO_SIDE: u32 = 720;
/// Maximum frame rate of an animated background.
const MAX_VIDEO_FPS: f32 = 24.0;
/// Maximum Gaussian sigma at full frost strength, in working pixels.
const MAX_SIGMA: f32 = 26.0;

const ANIMATED_EXTS: &[&str] = &[
    "gif", "apng", "mp4", "webm", "mov", "mkv", "avi", "m4v", "ogv",
];

/// Whether `path` is an animated background (GIF, APNG, or video).
pub fn is_animated_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ANIMATED_EXTS.iter().any(|v| v.eq_ignore_ascii_case(ext)))
}

#[derive(Default)]
struct FfmpegCodecs {
    encoders: HashSet<String>,
    decoders: HashSet<String>,
}

impl FfmpegCodecs {
    fn discover(ffmpeg: &Path) -> &'static FfmpegCodecs {
        static CACHE: OnceLock<FfmpegCodecs> = OnceLock::new();
        CACHE.get_or_init(|| {
            let encoders = query_ffmpeg_codecs(ffmpeg, "-encoders");
            let decoders = query_ffmpeg_codecs(ffmpeg, "-decoders");
            FfmpegCodecs { encoders, decoders }
        })
    }

    fn has_enc(&self, name: &str) -> bool {
        self.encoders.contains(name)
    }

    fn has_dec(&self, name: &str) -> bool {
        self.decoders.contains(name)
    }

    fn has_av1_dec(&self) -> bool {
        self.has_dec("libdav1d") || self.has_dec("libaom-av1") || self.has_dec("av1")
    }
}

fn query_ffmpeg_codecs(ffmpeg: &Path, flag: &str) -> HashSet<String> {
    let Ok(out) = hidden_command(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", flag])
        .output()
    else {
        return HashSet::new();
    };
    let mut set = HashSet::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let trimmed = line.trim_start();
        // Format: " V..... libsvtav1            SVT-AV1..."
        let mut parts = trimmed.split_whitespace();
        if let (Some(flags), Some(name)) = (parts.next(), parts.next()) {
            if flags.starts_with('V') && flags.len() >= 6 {
                set.insert(name.to_owned());
            }
        }
    }
    set
}

fn unique_stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Prepare a chosen background file for use by transcoding/optimizing it into a managed copy
/// under `data_dir/backgrounds/`. Never returns a reference to the original user file.
pub fn prepare_background_media(src: &Path, data_dir: &Path) -> Result<PathBuf, String> {
    let bg_dir = data_dir.join("backgrounds");
    std::fs::create_dir_all(&bg_dir).map_err(|e| e.to_string())?;
    let stamp = unique_stamp();

    if is_animated_path(src) {
        return prepare_animated_background(src, &bg_dir, stamp);
    }

    // Check if an image container (e.g. animated WebP) actually has multiple frames.
    if let Ok(tools) = dream_media::Tools::discover() {
        if is_multi_frame_image(&tools, src) {
            return prepare_animated_background(src, &bg_dir, stamp);
        }
    }

    prepare_still_background(src, &bg_dir, stamp)
}

fn is_multi_frame_image(tools: &dream_media::Tools, path: &Path) -> bool {
    let Ok(out) = hidden_command(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
    else {
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u32>()
        .is_ok_and(|n| n > 1)
}

/// Transcode a still image to a downscaled, metadata-free, storage-optimized managed copy.
fn prepare_still_background(src: &Path, bg_dir: &Path, stamp: u128) -> Result<PathBuf, String> {
    let image = dream_media::Image::open(src, MAX_SIDE).map_err(|e| e.to_string())?;
    let (w, h) = image.size();
    if w == 0 || h == 0 {
        return Err("That image has no pixels.".to_owned());
    }

    if let Ok(tools) = dream_media::Tools::discover() {
        let codecs = FfmpegCodecs::discover(&tools.ffmpeg);

        // 1. AVIF (smallest storage, modern codec) if FFmpeg can both encode and decode it.
        if codecs.has_av1_dec() {
            let avif_encoders: &[(&str, &[&str])] = &[
                ("libsvtav1", &["-crf", "32", "-preset", "6"]),
                (
                    "libaom-av1",
                    &["-crf", "32", "-cpu-used", "6", "-still-picture", "1"],
                ),
            ];
            for (enc, args) in avif_encoders {
                if codecs.has_enc(enc) {
                    let dst = bg_dir.join(format!("bg-{stamp}.avif"));
                    if encode_still_ffmpeg(&tools.ffmpeg, image.rgba(), w, h, enc, args, &dst) {
                        return Ok(dst);
                    }
                    let _ = std::fs::remove_file(&dst);
                }
            }
        }

        // 2. Lossy WebP if FFmpeg has both libwebp encoder and native webp decoder.
        if codecs.has_enc("libwebp") && codecs.has_dec("webp") {
            let dst = bg_dir.join(format!("bg-{stamp}.webp"));
            if encode_still_ffmpeg(
                &tools.ffmpeg,
                image.rgba(),
                w,
                h,
                "libwebp",
                &["-quality", "80", "-compression_level", "6"],
                &dst,
            ) {
                return Ok(dst);
            }
            let _ = std::fs::remove_file(&dst);
        }
    }

    // 3. Universal in-process JPEG fallback (opaque RGB8, quality 82, zero metadata).
    let dst = bg_dir.join(format!("bg-{stamp}.jpg"));
    let mut rgb = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for px in image.rgba().chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    let mut file = std::fs::File::create(&dst).map_err(|e| e.to_string())?;
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, 82);
    enc.encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
        .map_err(|e| {
            let _ = std::fs::remove_file(&dst);
            e.to_string()
        })?;
    Ok(dst)
}

fn encode_still_ffmpeg(
    ffmpeg: &Path,
    rgba: &[u8],
    w: u32,
    h: u32,
    encoder: &str,
    extra_args: &[&str],
    dst: &Path,
) -> bool {
    let size_str = format!("{w}x{h}");
    // Even dimensions for yuv420p.
    let ew = (w & !1).max(2);
    let eh = (h & !1).max(2);
    let vf = format!("scale={ew}:{eh}");
    let mut cmd = hidden_command(ffmpeg);
    cmd.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        &size_str,
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-vf",
        &vf,
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
    ])
    .args(extra_args)
    .arg(dst)
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::null());

    let Ok(mut child) = cmd.spawn() else {
        return false;
    };
    if let Some(mut stdin) = child.stdin.take() {
        if stdin.write_all(rgba).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return false;
        }
    }
    child.wait().is_ok_and(|s| s.success()) && dst.exists()
}

/// Transcode an animated GIF/WebP/video into a downscaled, audio-free, 8-bit 4:2:0,
/// capped-FPS, modern-codec managed copy under `data_dir/backgrounds/`.
fn prepare_animated_background(src: &Path, bg_dir: &Path, stamp: u128) -> Result<PathBuf, String> {
    let tools = dream_media::Tools::discover().map_err(|e| e.to_string())?;
    let codecs = FfmpegCodecs::discover(&tools.ffmpeg);
    let (w, h, fps) = probe_video(&tools, src)?;
    // Ensure even dimensions for yuv420p.
    let w = (w & !1).max(2);
    let h = (h & !1).max(2);
    let vf = format!("scale={w}:{h}:flags=lanczos,fps={fps:.2}");

    struct Candidate<'a> {
        ext: &'a str,
        encoder: &'a str,
        usable: bool,
        args: &'a [&'a str],
        faststart: bool,
    }

    let candidates = [
        Candidate {
            ext: "mp4",
            encoder: "libsvtav1",
            usable: codecs.has_enc("libsvtav1") && codecs.has_av1_dec(),
            args: &["-crf", "36", "-preset", "6"],
            faststart: true,
        },
        Candidate {
            ext: "mp4",
            encoder: "libaom-av1",
            usable: codecs.has_enc("libaom-av1") && codecs.has_av1_dec(),
            args: &["-crf", "36", "-cpu-used", "6"],
            faststart: true,
        },
        Candidate {
            ext: "mp4",
            encoder: "libx265",
            usable: codecs.has_enc("libx265") && codecs.has_dec("hevc"),
            args: &["-crf", "28", "-preset", "medium", "-tag:v", "hvc1"],
            faststart: true,
        },
        Candidate {
            ext: "webm",
            encoder: "libvpx-vp9",
            usable: codecs.has_enc("libvpx-vp9") && codecs.has_dec("vp9"),
            args: &[
                "-crf",
                "34",
                "-b:v",
                "0",
                "-deadline",
                "good",
                "-cpu-used",
                "2",
            ],
            faststart: false,
        },
        Candidate {
            ext: "mp4",
            encoder: "libx264",
            usable: codecs.has_enc("libx264") && codecs.has_dec("h264"),
            args: &["-crf", "26", "-preset", "medium"],
            faststart: true,
        },
        Candidate {
            ext: "mp4",
            encoder: "mpeg4",
            usable: true,
            args: &["-q:v", "6"],
            faststart: true,
        },
    ];

    for c in candidates {
        if !c.usable {
            continue;
        }
        let dst = bg_dir.join(format!("bg-{stamp}.{}", c.ext));
        let mut cmd = hidden_command(&tools.ffmpeg);
        cmd.args(["-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
            .arg(src)
            .args([
                "-an",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-vf",
                &vf,
                "-c:v",
                c.encoder,
                "-pix_fmt",
                "yuv420p",
            ])
            .args(c.args);
        if c.faststart {
            cmd.args(["-movflags", "+faststart"]);
        }
        cmd.arg(&dst);
        if cmd.status().is_ok_and(|s| s.success()) && dst.exists() {
            return Ok(dst);
        }
        let _ = std::fs::remove_file(&dst);
    }

    Err("Could not optimize the selected animated background.".to_owned())
}

/// Remove any managed background files in `data_dir/backgrounds/` that are no longer referenced.
pub fn prune_unused_backgrounds(data_dir: &Path, keep: &[&Path]) {
    let bg_dir = data_dir.join("backgrounds");
    let Ok(entries) = std::fs::read_dir(&bg_dir) else {
        return;
    };
    let keep_set: HashSet<PathBuf> = keep
        .iter()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| (*p).to_path_buf()))
        .collect();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let canon = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if !keep_set.contains(&canon) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

fn hidden_command(program: &Path) -> Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let mut cmd = Command::new(program);
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

type StaticTask = Box<dyn FnOnce() + Send + 'static>;

struct StaticWorker {
    pending: Arc<(Mutex<Option<StaticTask>>, Condvar)>,
}

/// One latest-wins worker for still-image decode and blur jobs. Rapid file changes or slider
/// movements can replace the queued work, while an already-running decode finishes off the UI
/// thread. The worker never waits for the UI: each job sends through a one-item channel with
/// `try_send`.
fn queue_static(task: StaticTask) -> Result<(), String> {
    static WORKER: OnceLock<Result<StaticWorker, String>> = OnceLock::new();
    let worker = WORKER
        .get_or_init(|| {
            let pending = Arc::new((Mutex::new(None::<StaticTask>), Condvar::new()));
            let worker_pending = Arc::clone(&pending);
            std::thread::Builder::new()
                .name("v1-background".to_owned())
                .spawn(move || {
                    loop {
                        let task = {
                            let (slot, ready) = &*worker_pending;
                            let mut slot =
                                slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                            while slot.is_none() {
                                slot = ready
                                    .wait(slot)
                                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                            }
                            slot.take().expect("worker woke with a queued task")
                        };
                        // A malformed file must not permanently stop this shared worker.
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
                    }
                })
                .map_err(|error| error.to_string())?;
            Ok(StaticWorker { pending })
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let (slot, ready) = &*worker.pending;
    *slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(task);
    ready.notify_one();
    Ok(())
}

#[derive(Clone)]
struct Source {
    width: u32,
    height: u32,
    // Strength changes only need a new mutable output buffer. Sharing the
    // immutable decode avoids cloning the full source on every slider update.
    rgba: Arc<Vec<u8>>,
}

/// Ensure a streaming FFmpeg child is reaped even if frame processing unwinds.
struct ChildGuard {
    child: Arc<Mutex<std::process::Child>>,
    finished: Arc<AtomicBool>,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub enum BackgroundPixels {
    Bc7(Vec<u8>),
    Rgba(Vec<u8>),
}

pub struct BackgroundImage {
    pub animated: bool,
    pub width: u32,
    pub height: u32,
    pub pixels: BackgroundPixels,
}

struct Msg {
    generation: u64,
    image: Result<(u32, u32, BackgroundPixels), String>,
    /// The unblurred decoded picture, when this message finished decoding one.
    source: Option<Source>,
}

/// The chosen picture/video, its frost strength, and the background decoder/blurrer.
pub struct Background {
    pub path: Option<PathBuf>,
    /// The file name, for the Settings line.
    pub name: String,
    /// How soft the picture is, 0 to 1.
    pub strength: f32,
    /// Why the last load did not work, if it did not.
    pub error: Option<String>,
    animated: bool,
    source: Option<Source>,
    rx: Option<Receiver<Msg>>,
    generation: u64,
    /// The strength the static picture in flight (or on screen) was softened with.
    blurred: f32,
    /// Shared strength for live animated video/GIF blur.
    live_strength: Arc<AtomicU32>,
    /// Whether the GPU supports BC7 block compression (4:1 VRAM savings).
    bc7_enabled: Arc<AtomicBool>,
    /// Stop signal for an active animated decoder thread.
    stop_flag: Option<Arc<AtomicBool>>,
}

impl Drop for Background {
    fn drop(&mut self) {
        self.stop_animation();
    }
}

impl Background {
    /// A background for a remembered path, if any. The load starts on a thread at once.
    pub fn new(path: Option<PathBuf>, strength: f32) -> Background {
        let strength = strength.clamp(0.0, 1.0);
        let mut background = Background {
            path: None,
            name: String::new(),
            strength,
            error: None,
            animated: false,
            source: None,
            rx: None,
            generation: 0,
            blurred: -1.0,
            live_strength: Arc::new(AtomicU32::new(strength.to_bits())),
            bc7_enabled: Arc::new(AtomicBool::new(true)),
            stop_flag: None,
        };
        if let Some(path) = path {
            background.choose(path);
        }
        background
    }

    /// Inform the background workers whether the GPU supports BC7 textures.
    pub fn set_bc7(&self, enabled: bool) {
        self.bc7_enabled.store(enabled, Ordering::Release);
    }

    fn stop_animation(&mut self) {
        if let Some(flag) = self.stop_flag.take() {
            flag.store(true, Ordering::Release);
        }
    }

    /// Put `path` behind the app, replacing whatever was there. The current picture stays on screen
    /// until the new one is ready.
    pub fn choose(&mut self, path: PathBuf) {
        self.stop_animation();
        self.name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        self.path = Some(path.clone());
        self.source = None;
        self.error = None;
        self.generation += 1;
        self.blurred = self.strength;
        self.live_strength
            .store(self.strength.to_bits(), Ordering::Release);

        let generation = self.generation;
        if is_animated_path(&path) {
            self.animated = true;
            let (tx, rx) = mpsc::sync_channel(1);
            self.rx = Some(rx);
            let stop = Arc::new(AtomicBool::new(false));
            self.stop_flag = Some(Arc::clone(&stop));
            let live_strength = Arc::clone(&self.live_strength);
            let bc7 = Arc::clone(&self.bc7_enabled);
            std::thread::Builder::new()
                .name("v1-background-video".to_owned())
                .spawn(move || {
                    stream_animated(generation, &path, live_strength, bc7, stop, tx);
                })
                .ok();
        } else {
            self.animated = false;
            let strength = self.strength;
            let bc7 = Arc::clone(&self.bc7_enabled);
            let (tx, rx) = mpsc::sync_channel(1);
            self.rx = Some(rx);
            if let Err(error) = queue_static(Box::new(move || {
                let _ = tx.try_send(decode(
                    generation,
                    &path,
                    strength,
                    bc7.load(Ordering::Acquire),
                ));
            })) {
                self.rx = None;
                self.error = Some(error);
            }
        }
    }

    /// Take the picture or video away.
    pub fn clear(&mut self) {
        self.stop_animation();
        self.generation += 1;
        self.path = None;
        self.name.clear();
        self.animated = false;
        self.source = None;
        self.error = None;
        self.rx = None;
    }

    /// Whether the current background is a looping GIF or video.
    pub fn is_animated(&self) -> bool {
        self.animated && self.error.is_none()
    }

    /// Change how soft the background is.
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 1.0);
        self.live_strength
            .store(self.strength.to_bits(), Ordering::Release);
    }

    /// Soften a static picture again if the strength has moved since the copy on screen was made.
    pub fn refresh(&mut self) {
        if self.animated || self.loading() {
            return;
        }
        if (self.strength - self.blurred).abs() < 0.005 {
            return;
        }
        if let Some(source) = self.source.clone() {
            self.blurred = self.strength;
            self.blur(source);
        }
    }

    /// Whether a static picture is still being decoded or softened.
    pub fn loading(&self) -> bool {
        !self.animated && self.rx.is_some()
    }

    /// The softened frame, once it is ready, for the host to upload.
    pub fn poll(&mut self) -> Option<BackgroundImage> {
        let rx = self.rx.as_ref()?;
        match rx.try_recv() {
            Ok(msg) => {
                if !self.animated {
                    self.rx = None;
                }
                if msg.generation != self.generation {
                    return None;
                }
                match msg.image {
                    Ok((width, height, pixels)) => {
                        if let Some(source) = msg.source {
                            self.source = Some(source);
                        }
                        self.error = None;
                        Some(BackgroundImage {
                            animated: self.animated,
                            width,
                            height,
                            pixels,
                        })
                    }
                    Err(error) => {
                        self.rx = None;
                        self.animated = false;
                        self.error = Some(error);
                        None
                    }
                }
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.rx = None;
                self.animated = false;
                self.error =
                    Some("The background decoder stopped before completing the image.".to_owned());
                None
            }
        }
    }

    fn blur(&mut self, source: Source) {
        self.generation += 1;
        let generation = self.generation;
        let strength = self.strength;
        let bc7 = Arc::clone(&self.bc7_enabled);
        let (tx, rx) = mpsc::sync_channel(1);
        self.rx = Some(rx);
        if let Err(error) = queue_static(Box::new(move || {
            let mut rgba = source.rgba.as_slice().to_vec();
            frost(&mut rgba, source.width, source.height, strength);
            let pixels = if bc7.load(Ordering::Acquire) {
                BackgroundPixels::Bc7(encode_bc7_rgba(&rgba, source.width, source.height))
            } else {
                BackgroundPixels::Rgba(rgba)
            };
            let _ = tx.try_send(Msg {
                generation,
                image: Ok((source.width, source.height, pixels)),
                source: None,
            });
        })) {
            self.rx = None;
            self.error = Some(error);
        }
    }
}

/// Decode and soften a picture on this thread, for `--shot`.
pub fn preview(path: &Path, strength: f32) -> Option<BackgroundImage> {
    let msg = decode(0, path, strength, false);
    let (width, height, pixels) = msg.image.ok()?;
    Some(BackgroundImage {
        animated: false,
        width,
        height,
        pixels,
    })
}

/// Decode a still through dream-media.
fn decode_file(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let image = dream_media::Image::open(path, MAX_SIDE).map_err(|error| error.to_string())?;
    let (width, height) = image.size();
    Ok((width, height, image.rgba().to_vec()))
}

/// Decode a picture and soften it.
fn decode(generation: u64, path: &Path, strength: f32, use_bc7: bool) -> Msg {
    let failed = |message: String| Msg {
        generation,
        image: Err(message),
        source: None,
    };
    let (width, height, rgba) = match decode_file(path) {
        Ok(image) => image,
        Err(message) => return failed(message),
    };
    if width == 0 || height == 0 {
        return failed("That image has no pixels.".to_owned());
    }
    let source = Source {
        width,
        height,
        rgba: Arc::new(rgba),
    };
    let mut rgba = source.rgba.as_slice().to_vec();
    frost(&mut rgba, width, height, strength);
    let pixels = if use_bc7 {
        BackgroundPixels::Bc7(encode_bc7_rgba(&rgba, width, height))
    } else {
        BackgroundPixels::Rgba(rgba)
    };
    Msg {
        generation,
        image: Ok((width, height, pixels)),
        source: Some(source),
    }
}

/// Probe `(width, height, fps)` of a GIF or video using `ffprobe`.
fn probe_video(tools: &dream_media::Tools, path: &Path) -> Result<(u32, u32, f32), String> {
    let mut cmd = hidden_command(&tools.ffprobe);
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0:s=x",
    ])
    .arg(path);
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err("Could not read video dimensions.".to_owned());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next().unwrap_or("").trim();
    let mut parts = line.split('x');
    let orig_w: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let orig_h: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    if orig_w == 0 || orig_h == 0 {
        return Err("Video has invalid dimensions.".to_owned());
    }
    let fps_str = parts.next().unwrap_or("24");
    let fps = if let Some((num, den)) = fps_str.split_once('/') {
        let n: f32 = num.parse().unwrap_or(24.0);
        let d: f32 = den.parse().unwrap_or(1.0);
        if d > 0.0 {
            n / d
        } else {
            24.0
        }
    } else {
        fps_str.parse().unwrap_or(24.0)
    };
    let fps = if fps.is_finite() {
        fps.clamp(10.0, MAX_VIDEO_FPS)
    } else {
        MAX_VIDEO_FPS
    };

    let long = orig_w.max(orig_h);
    let (w, h) = if long > MAX_VIDEO_SIDE {
        let scale = MAX_VIDEO_SIDE as f32 / long as f32;
        (
            ((orig_w as f32 * scale).round() as u32).max(2),
            ((orig_h as f32 * scale).round() as u32).max(2),
        )
    } else {
        (orig_w, orig_h)
    };
    Ok((w, h, fps))
}

fn stream_animated(
    generation: u64,
    path: &Path,
    live_strength: Arc<AtomicU32>,
    bc7: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    tx: SyncSender<Msg>,
) {
    let failed = |message: String| {
        let _ = tx.try_send(Msg {
            generation,
            image: Err(message),
            source: None,
        });
    };
    let tools = match dream_media::Tools::discover() {
        Ok(tools) => tools,
        Err(e) => return failed(e.to_string()),
    };
    let (w, h, fps) = match probe_video(&tools, path) {
        Ok(v) => v,
        Err(e) => return failed(e),
    };
    let frame_bytes = (w as usize) * (h as usize) * 4;
    let frame_period = Duration::from_secs_f32(1.0 / fps);
    let vf = format!("scale={w}:{h},fps={fps:.2}");

    let mut cmd = hidden_command(&tools.ffmpeg);
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "2",
        "-stream_loop",
        "-1",
        "-i",
    ])
    .arg(path)
    .args([
        "-an", "-vf", &vf, "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::null());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return failed(e.to_string()),
    };
    let Some(mut stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return failed("Failed to open video stream.".to_owned());
    };
    let child = Arc::new(Mutex::new(child));
    let finished = Arc::new(AtomicBool::new(false));
    let watcher_child = Arc::clone(&child);
    let watcher_stop = Arc::clone(&stop);
    let watcher_finished = Arc::clone(&finished);
    let watcher = match std::thread::Builder::new()
        .name("v1-background-stop".to_owned())
        .spawn(move || {
            while !watcher_stop.load(Ordering::Acquire) && !watcher_finished.load(Ordering::Acquire)
            {
                std::thread::sleep(Duration::from_millis(10));
            }
            if watcher_stop.load(Ordering::Acquire) && !watcher_finished.load(Ordering::Acquire) {
                if let Ok(mut child) = watcher_child.lock() {
                    let _ = child.kill();
                }
            }
        }) {
        Ok(watcher) => watcher,
        Err(error) => {
            drop(stdout);
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
                let _ = child.wait();
            }
            return failed(error.to_string());
        }
    };
    let child_guard = ChildGuard {
        child: Arc::clone(&child),
        finished: Arc::clone(&finished),
    };

    let mut raw = vec![0u8; frame_bytes];
    let mut next_frame = Instant::now();

    while !stop.load(Ordering::Acquire) {
        if stdout.read_exact(&mut raw).is_err() {
            break;
        }
        if stop.load(Ordering::Acquire) {
            break;
        }
        let strength = f32::from_bits(live_strength.load(Ordering::Relaxed)).clamp(0.0, 1.0);
        let mut frame = raw.clone();
        frost(&mut frame, w, h, strength);
        let pixels = if bc7.load(Ordering::Acquire) {
            BackgroundPixels::Bc7(encode_bc7_rgba(&frame, w, h))
        } else {
            BackgroundPixels::Rgba(frame)
        };

        let _ = tx.try_send(Msg {
            generation,
            image: Ok((w, h, pixels)),
            source: None,
        });

        next_frame += frame_period;
        let now = Instant::now();
        if next_frame > now {
            std::thread::sleep(next_frame - now);
        } else {
            next_frame = now;
        }
    }
    drop(child_guard);
    let _ = watcher.join();
}

/// Three-pass separable box blur (converges to a smooth Gaussian blur with zero boxy artifacts)
/// plus adaptive luminance shading so text over the background always stays readable.
fn frost(rgba: &mut [u8], width: u32, height: u32, strength: f32) {
    if width == 0 || height == 0 {
        return;
    }
    let strength = strength.clamp(0.0, 1.0);
    let sigma = strength * MAX_SIGMA;
    if sigma >= 0.6 {
        let radii = boxes_for_gauss(sigma, 3);
        let mut temp = vec![0u8; rgba.len()];
        for r in radii {
            if r > 0 {
                box_blur_pass(rgba, &mut temp, width, height, r);
            }
        }
    }

    // Adaptive shade: keep dark backgrounds rich and bright backgrounds comfortably dimmed.
    let mut luma_sum = 0u64;
    let mut samples = 0u64;
    for pixel in rgba.chunks_exact(4).step_by(16) {
        let l = 2126 * u64::from(pixel[0]) + 7152 * u64::from(pixel[1]) + 722 * u64::from(pixel[2]);
        luma_sum += l;
        samples += 1;
    }
    let avg_luma = if samples > 0 {
        (luma_sum as f32) / (samples as f32 * 2_550_000.0)
    } else {
        0.5
    };
    let shade = (0.68 - avg_luma * 0.30).clamp(0.38, 0.68);
    for pixel in rgba.chunks_exact_mut(4) {
        for channel in &mut pixel[..3] {
            *channel = (f32::from(*channel) * shade).round().min(255.0) as u8;
        }
        pixel[3] = 255;
    }
}

/// Kovesi's box-blur radii for `n` passes approximating a Gaussian of standard deviation `sigma`.
fn boxes_for_gauss(sigma: f32, n: usize) -> Vec<u32> {
    let w_ideal = ((12.0 * sigma * sigma / n as f32) + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i32;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wu = wl + 2;
    let m_ideal = (12.0 * sigma * sigma
        - (n as f32) * (wl as f32) * (wl as f32)
        - 4.0 * (n as f32) * (wl as f32)
        - 3.0 * (n as f32))
        / (-4.0 * (wl as f32) - 4.0);
    let m = m_ideal.round() as i32;
    (0..n)
        .map(|i| {
            let w = if (i as i32) < m { wl } else { wu };
            ((w - 1) / 2).max(0) as u32
        })
        .collect()
}

fn box_blur_pass(rgba: &mut [u8], temp: &mut [u8], width: u32, height: u32, radius: u32) {
    let (w, h) = (width as usize, height as usize);
    let r = radius as i32;
    let span = (2 * r + 1) as u32;
    let half = span / 2;
    // Horizontal.
    for y in 0..h {
        let row = y * w * 4;
        for c in 0..3 {
            let mut sum: u32 = 0;
            for i in -r..=r {
                let x = i.clamp(0, w as i32 - 1) as usize;
                sum += u32::from(rgba[row + x * 4 + c]);
            }
            for x in 0..w {
                temp[row + x * 4 + c] = ((sum + half) / span) as u8;
                let out = (x as i32 - r).clamp(0, w as i32 - 1) as usize;
                let inn = (x as i32 + r + 1).clamp(0, w as i32 - 1) as usize;
                sum += u32::from(rgba[row + inn * 4 + c]);
                sum -= u32::from(rgba[row + out * 4 + c]);
            }
        }
        for x in 0..w {
            temp[row + x * 4 + 3] = 255;
        }
    }
    // Vertical.
    let stride = w * 4;
    for x in 0..w {
        for c in 0..3 {
            let mut sum: u32 = 0;
            for i in -r..=r {
                let y = i.clamp(0, h as i32 - 1) as usize;
                sum += u32::from(temp[y * stride + x * 4 + c]);
            }
            for y in 0..h {
                rgba[y * stride + x * 4 + c] = ((sum + half) / span) as u8;
                let out = (y as i32 - r).clamp(0, h as i32 - 1) as usize;
                let inn = (y as i32 + r + 1).clamp(0, h as i32 - 1) as usize;
                sum += u32::from(temp[inn * stride + x * 4 + c]);
                sum -= u32::from(temp[out * stride + x * 4 + c]);
            }
        }
    }
}

/// Compress an RGBA8 image into BC7 Mode 6 (1 byte/pixel, 4:1 VRAM compression).
/// Uses 7-bit endpoints + shared P-bit (effective 8-bit precision) and 4-bit indices per 4x4 block.
pub fn encode_bc7_rgba(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bw = width.div_ceil(4) as usize;
    let bh = height.div_ceil(4) as usize;
    let mut out = vec![0u8; bw * bh * 16];
    let w = width as usize;
    let h = height as usize;

    for by in 0..bh {
        for bx in 0..bw {
            let mut block = [[0f32; 3]; 16];
            let mut mean = [0f32; 3];
            for py in 0..4 {
                let sy = (by * 4 + py).min(h - 1);
                for px in 0..4 {
                    let sx = (bx * 4 + px).min(w - 1);
                    let off = (sy * w + sx) * 4;
                    let rgb = [
                        f32::from(rgba[off]),
                        f32::from(rgba[off + 1]),
                        f32::from(rgba[off + 2]),
                    ];
                    block[py * 4 + px] = rgb;
                    mean[0] += rgb[0];
                    mean[1] += rgb[1];
                    mean[2] += rgb[2];
                }
            }
            mean[0] /= 16.0;
            mean[1] /= 16.0;
            mean[2] /= 16.0;

            // Find principal axis via 2-step farthest-pair + projection.
            let mut far0 = block[0];
            let mut max_d = -1.0f32;
            for c in &block {
                let d =
                    (c[0] - mean[0]).powi(2) + (c[1] - mean[1]).powi(2) + (c[2] - mean[2]).powi(2);
                if d > max_d {
                    max_d = d;
                    far0 = *c;
                }
            }
            let mut far1 = far0;
            max_d = -1.0;
            for c in &block {
                let d =
                    (c[0] - far0[0]).powi(2) + (c[1] - far0[1]).powi(2) + (c[2] - far0[2]).powi(2);
                if d > max_d {
                    max_d = d;
                    far1 = *c;
                }
            }

            let dir = [far1[0] - far0[0], far1[1] - far0[1], far1[2] - far0[2]];
            let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];

            let mut ep0 = far0;
            let mut ep1 = far1;
            let mut indices = [0u8; 16];

            if len_sq > 1e-4 {
                let mut min_t = f32::INFINITY;
                let mut max_t = f32::NEG_INFINITY;
                let mut ts = [0.0f32; 16];
                for (i, c) in block.iter().enumerate() {
                    let t = ((c[0] - mean[0]) * dir[0]
                        + (c[1] - mean[1]) * dir[1]
                        + (c[2] - mean[2]) * dir[2])
                        / len_sq;
                    ts[i] = t;
                    min_t = min_t.min(t);
                    max_t = max_t.max(t);
                }
                for ch in 0..3 {
                    ep0[ch] = (mean[ch] + min_t * dir[ch]).clamp(0.0, 255.0);
                    ep1[ch] = (mean[ch] + max_t * dir[ch]).clamp(0.0, 255.0);
                }
                let span = (max_t - min_t).max(1e-5);
                for i in 0..16 {
                    let q = (((ts[i] - min_t) / span) * 15.0).round() as i32;
                    indices[i] = q.clamp(0, 15) as u8;
                }
            }

            // Anchor pixel 0 must have MSB == 0 (index < 8). Swap endpoints if needed.
            if indices[0] >= 8 {
                std::mem::swap(&mut ep0, &mut ep1);
                for idx in &mut indices {
                    *idx = 15 - *idx;
                }
            }

            let q7 = |v: f32| -> (u128, u128) {
                let v8 = (v.round() as u32).clamp(0, 255);
                (u128::from(v8 >> 1), u128::from(v8 & 1))
            };
            let (r0, rp0) = q7(ep0[0]);
            let (g0, _) = q7(ep0[1]);
            let (b0, _) = q7(ep0[2]);
            let (r1, rp1) = q7(ep1[0]);
            let (g1, _) = q7(ep1[1]);
            let (b1, _) = q7(ep1[2]);
            // Opaque alpha (A0 = 127, A1 = 127 -> (127 << 1) | 1 = 255 if p=1, or 254 if p=0).
            let a0 = 127u128;
            let a1 = 127u128;

            // Pack BC7 Mode 6 (128 bits, little-endian):
            // [0..7): mode 0b1000000 (0x40)
            // [7..63): R0(7), R1(7), G0(7), G1(7), B0(7), B1(7), A0(7), A1(7)
            // [63..65): P0(1), P1(1)
            // [65..128): idx0(3), idx1..idx15(4 each)
            let mut bits: u128 = 0x40;
            bits |= r0 << 7;
            bits |= r1 << 14;
            bits |= g0 << 21;
            bits |= g1 << 28;
            bits |= b0 << 35;
            bits |= b1 << 42;
            bits |= a0 << 49;
            bits |= a1 << 56;
            bits |= rp0 << 63;
            bits |= rp1 << 64;
            bits |= u128::from(indices[0] & 0x7) << 65;
            for (i, &idx) in indices.iter().enumerate().skip(1) {
                bits |= u128::from(idx & 0xf) << (68 + (i - 1) * 4);
            }

            let dst_off = (by * bw + bx) * 16;
            out[dst_off..dst_off + 16].copy_from_slice(&bits.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_static_worker_keeps_only_the_newest_waiting_job() {
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let (active_tx, active_rx) = mpsc::channel();
        queue_static(Box::new(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            active_tx.send(()).unwrap();
        }))
        .unwrap();
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let (obsolete_tx, obsolete_rx) = mpsc::channel();
        queue_static(Box::new(move || obsolete_tx.send(()).unwrap())).unwrap();
        let (latest_tx, latest_rx) = mpsc::channel();
        queue_static(Box::new(move || latest_tx.send(()).unwrap())).unwrap();
        release_tx.send(()).unwrap();
        active_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        latest_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(obsolete_rx.recv_timeout(Duration::from_millis(20)).is_err());
    }

    #[test]
    fn a_png_decodes_with_its_pixels_intact() {
        let path =
            std::env::temp_dir().join(format!("v1-background-decode-{}.png", std::process::id()));
        let pixels: Vec<u8> = [255, 40, 10, 255].repeat(4);
        image::RgbaImage::from_raw(2, 2, pixels.clone())
            .unwrap()
            .save(&path)
            .unwrap();
        let decoded = decode_file(&path).expect("a written PNG decodes");
        let _ = std::fs::remove_file(&path);
        assert_eq!((decoded.0, decoded.1), (2, 2));
        assert_eq!(decoded.2, pixels);
    }

    #[test]
    fn a_missing_picture_is_an_error() {
        assert!(decode_file(Path::new("no-such-background.png")).is_err());
    }

    #[test]
    fn bc7_encoder_packs_valid_mode6_blocks_for_odd_dimensions() {
        let rgba = vec![120u8, 180, 240, 255].repeat(7 * 5);
        let bc7 = encode_bc7_rgba(&rgba, 7, 5);
        // 7x5 -> 2x2 blocks -> 64 bytes.
        assert_eq!(bc7.len(), 64);
        for chunk in bc7.chunks_exact(16) {
            assert_eq!(chunk[0] & 0x7f, 0x40);
        }
    }

    #[test]
    fn prepare_background_media_copies_and_optimizes_still_images() {
        let dir = std::env::temp_dir().join(format!("v1-bg-prep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("big.png");
        let rgba = vec![80u8, 120, 200, 255].repeat(64 * 64);
        image::RgbaImage::from_raw(64, 64, rgba)
            .unwrap()
            .save(&src)
            .unwrap();

        let managed = prepare_background_media(&src, &dir).expect("still image should optimize");
        assert!(managed.starts_with(dir.join("backgrounds")));
        assert_ne!(managed, src);
        assert!(managed.exists());

        // Removing the original file must not break decoding of the managed copy.
        std::fs::remove_file(&src).unwrap();
        let decoded = decode_file(&managed).expect("managed copy decodes independently");
        assert_eq!((decoded.0, decoded.1), (64, 64));

        // Pruning should remove unreferenced files.
        prune_unused_backgrounds(&dir, &[]);
        assert!(!managed.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
