// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The custom background: a picture, GIF, or looping video behind the whole app, softened into
//! frosted glass.
//!
//! Decoding and blurring happen on a thread of their own, so opening a large photo or playing a
//! looping video never stalls a frame. Static pictures are decoded once and kept, downscaled, so
//! moving the strength slider only re-blurs what is already in memory. Animated GIFs and videos
//! stream frames from FFmpeg (with audio stripped on upload) and apply live Gaussian frost.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::gpu::CoverImage;

/// The working size of a static picture: long edge, in pixels.
const MAX_SIDE: u32 = 1280;
/// The working size of an animated GIF or video background: long edge, in pixels.
const MAX_VIDEO_SIDE: u32 = 720;
/// Maximum Gaussian sigma at full frost strength, in working pixels.
const MAX_SIGMA: f32 = 26.0;

const VIDEO_EXTS: &[&str] = &["mp4", "webm", "mov", "mkv", "avi", "m4v", "ogv"];
const ANIMATED_EXTS: &[&str] = &["gif", "mp4", "webm", "mov", "mkv", "avi", "m4v", "ogv"];

/// Whether `path` is a video format whose audio should be stripped on upload.
pub fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| VIDEO_EXTS.iter().any(|v| v.eq_ignore_ascii_case(ext)))
}

/// Whether `path` is an animated background (GIF or video).
pub fn is_animated_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ANIMATED_EXTS.iter().any(|v| v.eq_ignore_ascii_case(ext)))
}

/// Prepare a chosen background file for use. For video files, strips any audio track into a
/// managed copy under `data_dir/backgrounds/` so the background is guaranteed silent.
pub fn prepare_background_media(src: &Path, data_dir: &Path) -> Result<PathBuf, String> {
    if !is_video_path(src) {
        return Ok(src.to_path_buf());
    }
    let tools = dream_media::Tools::discover().map_err(|e| e.to_string())?;
    let bg_dir = data_dir.join("backgrounds");
    std::fs::create_dir_all(&bg_dir).map_err(|e| e.to_string())?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let ext = src
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4")
        .to_ascii_lowercase();
    let dst = bg_dir.join(format!("bg-{stamp}.{ext}"));

    // Fast lossless remux with audio stripped (`-an`).
    let mut cmd = hidden_command(&tools.ffmpeg);
    cmd.args(["-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(src)
        .args(["-an", "-c:v", "copy"])
        .arg(&dst);
    let status = cmd.status().map_err(|e| e.to_string())?;
    if status.success() && dst.exists() {
        return Ok(dst);
    }

    // Use encoders available in the bundled LGPL builds. Apple provides hardware
    // H.264; other platforms use FFmpeg's native MPEG-4 encoder.
    let _ = std::fs::remove_file(&dst);
    let fallback_dst = bg_dir.join(format!("bg-{stamp}.mp4"));
    let mut cmd = hidden_command(&tools.ffmpeg);
    cmd.args(["-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(src)
        .arg("-an");
    #[cfg(target_os = "macos")]
    cmd.args(["-c:v", "h264_videotoolbox", "-b:v", "6000k"]);
    #[cfg(not(target_os = "macos"))]
    cmd.args(["-c:v", "mpeg4", "-q:v", "3"]);
    cmd.args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        .arg(&fallback_dst);
    let status = cmd.status().map_err(|e| e.to_string())?;
    if status.success() && fallback_dst.exists() {
        Ok(fallback_dst)
    } else {
        let _ = std::fs::remove_file(&fallback_dst);
        Err("Could not strip audio from the selected video.".to_owned())
    }
}

fn hidden_command(program: &Path) -> Command {
    #[cfg(windows)]
    let mut cmd = Command::new(program);
    #[cfg(not(windows))]
    let cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
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

struct Msg {
    generation: u64,
    image: Result<(u32, u32, Vec<u8>), String>,
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
            stop_flag: None,
        };
        if let Some(path) = path {
            background.choose(path);
        }
        background
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
            std::thread::Builder::new()
                .name("v1-background-video".to_owned())
                .spawn(move || {
                    stream_animated(generation, &path, live_strength, stop, tx);
                })
                .ok();
        } else {
            self.animated = false;
            let strength = self.strength;
            let (tx, rx) = mpsc::sync_channel(1);
            self.rx = Some(rx);
            if let Err(error) = queue_static(Box::new(move || {
                let _ = tx.try_send(decode(generation, &path, strength));
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
    pub fn poll(&mut self) -> Option<CoverImage> {
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
                    Ok((width, height, rgba)) => {
                        if let Some(source) = msg.source {
                            self.source = Some(source);
                        }
                        self.error = None;
                        Some(CoverImage {
                            url: if self.animated {
                                format!("background:anim:{}", msg.generation)
                            } else {
                                format!("background:{}", msg.generation)
                            },
                            width,
                            height,
                            rgba,
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
        let (tx, rx) = mpsc::sync_channel(1);
        self.rx = Some(rx);
        if let Err(error) = queue_static(Box::new(move || {
            let mut rgba = source.rgba.as_slice().to_vec();
            frost(&mut rgba, source.width, source.height, strength);
            let _ = tx.try_send(Msg {
                generation,
                image: Ok((source.width, source.height, rgba)),
                source: None,
            });
        })) {
            self.rx = None;
            self.error = Some(error);
        }
    }
}

/// Decode and soften a picture on this thread, for `--shot`.
pub fn preview(path: &Path, strength: f32) -> Option<CoverImage> {
    let msg = decode(0, path, strength);
    let (width, height, rgba) = msg.image.ok()?;
    Some(CoverImage {
        url: "background:preview".to_owned(),
        width,
        height,
        rgba,
    })
}

/// Decode a still through dream-media.
fn decode_file(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let image = dream_media::Image::open(path, MAX_SIDE).map_err(|error| error.to_string())?;
    let (width, height) = image.size();
    Ok((width, height, image.rgba().to_vec()))
}

/// Decode a picture and soften it.
fn decode(generation: u64, path: &Path, strength: f32) -> Msg {
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
    Msg {
        generation,
        image: Ok((width, height, rgba)),
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
        fps.clamp(10.0, 30.0)
    } else {
        24.0
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

        let _ = tx.try_send(Msg {
            generation,
            image: Ok((w, h, frame)),
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
}
