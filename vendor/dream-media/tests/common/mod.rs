#![allow(dead_code)]

use dream_media::Tools;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub struct FixtureDir(pub PathBuf);

impl FixtureDir {
    pub fn new(name: &str) -> Self {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dream-media-test-{}-{id}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn tools() -> Option<Tools> {
    Tools::discover().ok()
}

pub fn ffmpeg(tools: &Tools, args: &[&str]) -> bool {
    Command::new(&tools.ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .status()
        .is_ok_and(|status| status.success())
}

/// A 2 second 320x240 25 fps clip with a 440 Hz tone, H.264 when this build
/// has libx264 and MPEG-4 part 2 otherwise.
pub fn sample_clip(tools: &Tools, dir: &FixtureDir) -> PathBuf {
    let dest = dir.path("clip.mp4");
    for codec in ["libx264", "mpeg4"] {
        let path = dest.to_str().unwrap();
        if ffmpeg(
            tools,
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=25",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                codec,
                "-c:a",
                "aac",
                path,
            ],
        ) {
            return dest;
        }
    }
    panic!("ffmpeg could not write the sample clip");
}

pub fn rotated_clip(tools: &Tools, dir: &FixtureDir) -> PathBuf {
    let source = sample_clip(tools, dir);
    let dest = dir.path("rotated.mp4");
    assert!(
        ffmpeg(
            tools,
            &[
                "-display_rotation",
                "90",
                "-i",
                path_str(&source),
                "-c",
                "copy",
                path_str(&dest),
            ],
        ),
        "ffmpeg could not write the rotated clip"
    );
    dest
}

pub fn long_gop_clip(tools: &Tools, dir: &FixtureDir) -> PathBuf {
    let dest = dir.path("long-gop.ts");
    assert!(
        ffmpeg(
            tools,
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=25",
                "-t",
                "3",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "250",
                "-c:v",
                "mpeg2video",
                dest.to_str().unwrap(),
            ],
        ),
        "ffmpeg could not write the transport stream"
    );
    dest
}

pub fn sample_audio(tools: &Tools, dir: &FixtureDir) -> PathBuf {
    let dest = dir.path("tone.wav");
    assert!(
        ffmpeg(
            tools,
            &[
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1.5",
                "-c:a",
                "pcm_s16le",
                dest.to_str().unwrap(),
            ],
        ),
        "ffmpeg could not write the sample audio"
    );
    dest
}

pub fn path_str(path: &Path) -> &str {
    path.to_str().unwrap()
}
