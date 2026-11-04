//! FFmpeg/ffprobe discovery and the bounded input arguments shared by every
//! decoder invocation.

use crate::error::{Error, Result};
use crate::process;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

/// Longest probe or still decode. Playback streams have no fixed deadline:
/// they live until their stream is dropped.
pub(crate) const TOOL_DEADLINE: Duration = Duration::from_secs(20);

/// Transient allocation ceiling for one FFmpeg process.
const MAX_ALLOC: &str = "268435456";

/// Demuxers this crate accepts. Playlists, concat and screen-capture formats
/// are absent: a media file is a container, not an instruction to open
/// something else.
const FORMATS: &str = "mov,matroska,webm,avi,mpeg,mpegts,flv,ogg,asf,m4v,h264,hevc,av1,ivf,gif,apng,webp,image2,png_pipe,jpeg_pipe,jpegls_pipe,jpegxl_pipe,bmp_pipe,tiff_pipe,webp_pipe,qoi_pipe,exr_pipe,dds_pipe,ico,pnm_pipe,ppm_pipe,pgm_pipe,pam_pipe,mp3,flac,wav,aac,aiff,ape,wv,tta,caf,w64,opus";

/// Protocols this crate accepts. `file` covers local media; `http`, `https`,
/// `tcp` and `tls` cover a stream served over loopback by an app's own range
/// proxy. Everything else (concat, subfile, crypto, data) stays refused.
const PROTOCOLS: &str = "file,http,https,tcp,tls";

static DISCOVERED: OnceLock<Option<Tools>> = OnceLock::new();

/// The two FFmpeg programs the handler needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tools {
    /// Path to the `ffmpeg` executable.
    pub ffmpeg: PathBuf,
    /// Path to the matching `ffprobe` executable.
    pub ffprobe: PathBuf,
}

impl Tools {
    /// Find the tools once for the process and reuse the answer. Search order:
    /// `DREAM_FFMPEG` / `DREAM_FFPROBE`, `DREAM_FFMPEG_DIR`, an `ffmpeg/`
    /// folder beside the executable, the executable's own folder, then
    /// absolute directories on `PATH`.
    pub fn discover() -> Result<Self> {
        let found = DISCOVERED.get_or_init(|| {
            let ffmpeg = locate("DREAM_FFMPEG", "DREAM_FFMPEG_DIR", "ffmpeg")?;
            let ffprobe = locate("DREAM_FFPROBE", "DREAM_FFMPEG_DIR", "ffprobe")?;
            Some(Self { ffmpeg, ffprobe })
        });
        found.clone().ok_or_else(|| {
            Error::msg(
                "ffmpeg and ffprobe were not found; put them on PATH, beside the executable, \
                 in an ffmpeg/ folder beside it, or set DREAM_FFMPEG/DREAM_FFPROBE",
            )
        })
    }

    /// Explicit paths, for tests and apps that bundle their own build.
    pub fn from_paths(ffmpeg: impl Into<PathBuf>, ffprobe: impl Into<PathBuf>) -> Self {
        Self {
            ffmpeg: ffmpeg.into(),
            ffprobe: ffprobe.into(),
        }
    }

    pub(crate) fn ffmpeg_command(&self) -> Command {
        process::command(&self.ffmpeg)
    }

    /// Probe a file to flat key/value metadata.
    pub(crate) fn ffprobe_flat(&self, source: &Path) -> Result<String> {
        let mut cmd = process::command(&self.ffprobe);
        cmd.args(["-v", "error", "-max_alloc", MAX_ALLOC, "-threads", "1"]);
        cmd.args(["-protocol_whitelist", PROTOCOLS, "-format_whitelist", FORMATS]);
        cmd.args(["-probesize", "8388608", "-analyzeduration", "5000000"]);
        cmd.args(["-print_format", "flat", "-show_format", "-show_streams"]);
        cmd.arg(source);
        let run = process::run(cmd, TOOL_DEADLINE, 1024 * 1024)?;
        if !run.status.success() {
            return Err(process::failure("ffprobe", source, &run));
        }
        Ok(String::from_utf8_lossy(&run.stdout).into_owned())
    }
}

/// Input options shared by probe and decode: an allowlist of media demuxers and
/// protocols, bounded probing/allocation, and reconnect for a networked input.
/// True when the source is a URL rather than a path, which decides whether FFmpeg's
/// HTTP-only options may be passed.
pub(crate) fn is_networked(source: &Path) -> bool {
    let text = source.to_string_lossy();
    text.starts_with("http://") || text.starts_with("https://")
}

pub(crate) fn input_options(cmd: &mut Command, threads: &str, filter_threads: &str, networked: bool) {
    cmd.args([
        "-max_alloc",
        MAX_ALLOC,
        "-threads",
        threads,
        "-filter_threads",
        filter_threads,
        "-protocol_whitelist",
        PROTOCOLS,
        "-format_whitelist",
        FORMATS,
        "-probesize",
        "8388608",
        "-analyzeduration",
        "5000000",
    ]);
    if networked {
        // HTTP-only options: FFmpeg rejects them outright for a local file.
        cmd.args([
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_on_network_error",
            "1",
            "-reconnect_delay_max",
            "2",
        ]);
    }
}

fn locate(env_file: &str, env_dir: &str, name: &str) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(env_file) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let mut dirs = Vec::new();
    if let Some(dir) = std::env::var_os(env_dir) {
        dirs.push(PathBuf::from(dir));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.join("ffmpeg"));
        dirs.push(parent.to_path_buf());
    }
    if let Some(found) = dirs.iter().find_map(|dir| in_dir(dir, name)) {
        return Some(found);
    }
    // A relative path entry such as "." would let the current working
    // directory stand in for FFmpeg; only absolute PATH entries qualify.
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .filter(|dir| dir.is_absolute())
        .find_map(|dir| in_dir(&dir, name))
}

fn in_dir(dir: &Path, name: &str) -> Option<PathBuf> {
    for candidate in [format!("{name}.exe"), name.to_string()] {
        let path = dir.join(candidate);
        if path.is_file() {
            return Some(path.canonicalize().unwrap_or(path));
        }
    }
    None
}


