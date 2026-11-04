//! Low-level decoder streams: FFmpeg commands and the reader/feeder threads
//! that turn their stdout into frames or ring samples.

use crate::audio::AudioShared;
use crate::error::{Error, Result};
use crate::process::{Stream, stderr_line};
use crate::tools::{Tools, input_options, is_networked};
use std::io::{ErrorKind, Read};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::time::Duration;

/// Total decoded-frame memory per video session. Every frame buffer is
/// allocated once at open and then cycled, so this is the session's entire
/// pixel cost rather than a per-queue limit.
const BUFFER_BYTES: usize = 48 * 1024 * 1024;
const BUFFER_DEPTH_MIN: usize = 2;
const BUFFER_DEPTH_MAX: usize = 6;
/// How long the reader waits for a free buffer before re-checking whether its
/// session has been dropped.
const POOL_WAIT: Duration = Duration::from_millis(50);

/// What a video reader sends. `End` counts as a queue slot too, so the viewer
/// never waits for a wake that cannot come.
pub(crate) enum VideoMsg {
    Frame { pts_ms: i64, rgba: Vec<u8> },
    End,
}

/// A live video decoder and its reader channel.
pub(crate) struct VideoSession {
    pub stream: Stream,
    pub frames: Receiver<VideoMsg>,
    /// Handed to every [`Frame`](crate::Frame); dropping a frame returns its
    /// pixel buffer here instead of freeing it. Exactly `depth` buffers exist,
    /// so a return always fits.
    pub pool: SyncSender<Vec<u8>>,
    /// Set when the session is dropped, so a reader parked on an empty pool
    /// exits instead of waiting for a frame that will never be freed.
    cancel: Arc<AtomicBool>,
}

impl Drop for VideoSession {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Everything the video command needs to know.
pub(crate) struct VideoConfig {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub hwaccel: bool,
    /// Decode from the start and discard up to the seek point, for streams
    /// whose demuxer seek lands past the last keyframe.
    pub exact_seek: bool,
}

pub(crate) fn spawn_video(
    tools: &Tools,
    source: &Path,
    start_ms: i64,
    cfg: &VideoConfig,
) -> Result<VideoSession> {
    if cfg.width == 0 || cfg.height == 0 || !(cfg.fps.is_finite() && cfg.fps > 0.0) {
        return Err(Error::msg(
            "video output needs a non-zero size and a positive frame rate",
        ));
    }
    let mut cmd = tools.ffmpeg_command();
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    if cfg.hwaccel {
        cmd.args(["-hwaccel", "auto"]);
    }
    input_options(&mut cmd, "4", "2", is_networked(source));
    if cfg.exact_seek {
        cmd.arg("-i").arg(source);
        seek_arg(&mut cmd, start_ms);
    } else {
        seek_arg(&mut cmd, start_ms);
        cmd.arg("-i").arg(source);
    }
    let filter = format!(
        "scale={}:{}:flags=bilinear,setsar=1,format=rgba",
        cfg.width, cfg.height
    );
    cmd.args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf"])
        .arg(filter)
        .args(["-fps_mode", "cfr", "-r"])
        .arg(frame_rate_arg(cfg.fps))
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]);
    let mut stream = Stream::spawn(cmd)?;
    let stdout = stream
        .stdout()
        .ok_or_else(|| Error::msg("video decoder has no output pipe"))?;
    let bytes = (cfg.width as usize)
        .checked_mul(cfg.height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| Error::msg("video dimensions overflow addressable memory"))?;
    let depth = buffer_depth(bytes);
    // Allocate the pool up front: one zeroing per buffer for the whole
    // session, and a read that never has to reserve or grow.
    let (pool, free) = mpsc::sync_channel(depth);
    for _ in 0..depth {
        pool.try_send(vec![0u8; bytes])
            .map_err(|_| Error::msg("video buffer pool overflow"))?;
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let frames = spawn_reader(stdout, start_ms, cfg.fps, free, Arc::clone(&cancel), depth);
    Ok(VideoSession {
        stream,
        frames,
        pool,
        cancel,
    })
}

fn buffer_depth(bytes: usize) -> usize {
    (BUFFER_BYTES / bytes.max(1)).clamp(BUFFER_DEPTH_MIN, BUFFER_DEPTH_MAX)
}

fn spawn_reader(
    mut stdout: impl Read + Send + 'static,
    start_ms: i64,
    fps: f64,
    free: Receiver<Vec<u8>>,
    cancel: Arc<AtomicBool>,
    depth: usize,
) -> Receiver<VideoMsg> {
    let (sender, receiver) = mpsc::sync_channel(depth);
    std::thread::spawn(move || {
        let mut index: u64 = 0;
        loop {
            // Waiting here is the backpressure: when the viewer is paused or
            // slow, decoding stops instead of filling memory.
            let mut rgba = match free.recv_timeout(POOL_WAIT) {
                Ok(buffer) => buffer,
                Err(RecvTimeoutError::Timeout) => {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => return,
            };
            // `read_exact` into the whole buffer: never a `clear` + `resize`,
            // which would zero several megabytes per frame for nothing.
            debug_assert!(rgba.capacity() >= rgba.len());
            if stdout.read_exact(&mut rgba).is_err() {
                let _ = sender.send(VideoMsg::End);
                return;
            }
            // Frame i is at start + i/fps. Computed from the index rather than
            // summed, so rounding never accumulates.
            let pts_ms = start_ms + (index as f64 * 1000.0 / fps).round() as i64;
            index += 1;
            if sender.send(VideoMsg::Frame { pts_ms, rgba }).is_err() {
                return;
            }
        }
    });
    receiver
}

/// Start the audio decoder for one session.
pub(crate) fn spawn_audio(
    tools: &Tools,
    source: &Path,
    start_ms: i64,
    rate: u32,
    channels: u16,
) -> Result<Stream> {
    if rate == 0 || channels == 0 {
        return Err(Error::msg(
            "audio output needs a sample rate and a channel count",
        ));
    }
    let mut cmd = tools.ffmpeg_command();
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    input_options(&mut cmd, "1", "1", is_networked(source));
    seek_arg(&mut cmd, start_ms);
    cmd.arg("-i")
        .arg(source)
        .args(["-map", "0:a:0", "-vn", "-sn", "-dn", "-ac"])
        .arg(channels.to_string())
        .arg("-ar")
        .arg(rate.to_string())
        .args(["-threads", "1", "-f", "f32le", "pipe:1"]);
    Stream::spawn(cmd)
}

/// `-ss` before `-i` seeks in the demuxer. Zero means the beginning and adds
/// no argument.
fn seek_arg(cmd: &mut std::process::Command, start_ms: i64) {
    if start_ms > 0 {
        cmd.args(["-ss", &format!("{:.3}", start_ms as f64 / 1000.0)]);
    }
}

fn frame_rate_arg(fps: f64) -> String {
    let text = format!("{fps:.6}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Read f32le samples from the decoder into the shared ring. Partial samples
/// and partial frames are carried to the next read. Returns when the stream
/// ends or a seek makes this generation stale.
pub(crate) fn feed_audio(
    mut stdout: impl Read,
    shared: Arc<AudioShared>,
    generation: u64,
    channels: u16,
) {
    let frame_bytes = 4 * usize::from(channels.max(1));
    let mut carry = Vec::with_capacity(64 * 1024);
    let mut chunk = vec![0u8; 32 * 1024];
    let mut samples: Vec<f32> = Vec::with_capacity(16 * 1024);
    loop {
        let read = match stdout.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        carry.extend_from_slice(&chunk[..read]);
        let whole = carry.len() / frame_bytes * frame_bytes;
        if whole == 0 {
            continue;
        }
        samples.clear();
        samples.extend(
            carry[..whole]
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        );
        carry.copy_within(whole.., 0);
        carry.truncate(carry.len() - whole);
        if !shared.push_wait(generation, &samples) {
            return;
        }
    }
    shared.set_finished(generation);
}

/// First line of a decoder's diagnostics, or a fallback, for a failure status.
pub(crate) fn stream_failure(stream: Option<&Stream>, fallback: &str) -> String {
    stream
        .map(Stream::stderr_text)
        .filter(|text| !text.is_empty())
        .map(|text| stderr_line(text.as_bytes()))
        .unwrap_or_else(|| fallback.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_fits_in_a_predictable_amount_of_memory() {
        // Depth times frame size is the session's whole pixel cost, because
        // every buffer is allocated once and then recycled.
        for (width, height) in [
            (640u32, 360u32),
            (1280, 720),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
        ] {
            let bytes = width as usize * height as usize * 4;
            let depth = buffer_depth(bytes);
            assert!((BUFFER_DEPTH_MIN..=BUFFER_DEPTH_MAX).contains(&depth));
            assert!(
                depth * bytes <= 68 * 1024 * 1024,
                "{width}x{height} would hold {} MiB",
                depth * bytes / (1024 * 1024)
            );
        }
    }
}

