//! Dream Media Handler.
//!
//! A small media service for Dream apps. It decodes still images, video and
//! audio through FFmpeg and hands video frames to Dream's custom graphics
//! pipeline as GPU textures, without the storage, import, database, encryption
//! or thumbnail layers of a full media application.
//!
//! # The three entry points
//!
//! - [`Image`] decodes a still to RGBA on the CPU. PNG always decodes in
//!   process; JPEG does too up to 3 megapixels; everything else, and larger
//!   JPEGs, go through a raw RGBA pipe. `Image::upload` turns it into a
//!   texture.
//! - [`Player`] plays video and/or audio. `poll` returns [`Frame`]s of RGBA
//!   pixels, scheduled against an audio-clocked timeline, with dropped frames
//!   and recycled buffers.
//! - `VideoPlayer` (feature `gpu`, on by default) wraps a [`Player`] with a
//!   `VideoSurface`: a two-texture ring uploaded asynchronously through
//!   `dream::texture::TextureUploader`, ready to bind in Dream's UI or a custom
//!   pipeline.
//!
//! # Cost, in numbers
//!
//! A playing video costs one decode thread, one FFmpeg process (two with
//! sound), a fixed pool of frame buffers sized to at most 68 MiB, and two
//! GPU textures. Once warmed, `poll` allocates nothing: buffers cycle back
//! from [`Frame`] to the decoder, the audio ring is preallocated, and the
//! feeder sleeps on a condition variable instead of polling. Dropped frames
//! are recycled rather than freed, so a slow UI never grows memory.
//!
//! ```no_run
//! # #[cfg(feature = "gpu")] {
//! use std::sync::Arc;
//! use std::time::Instant;
//! use dream::gfx::vulkan::GpuContext;
//! use dream_media::{PlayerOptions, VideoPlayer};
//!
//! fn open(ctx: Arc<GpuContext>) -> dream_media::Result<VideoPlayer> {
//!     let mut video = VideoPlayer::open(&ctx, "intro.mp4", PlayerOptions::default())?;
//!     if let Some(texture) = video.poll(Instant::now()) {
//!         // ui.image(rect, texture, Color::WHITE);
//!         let _ = texture;
//!     }
//!     Ok(video)
//! }
//! # }
//! ```
//!
//! # Why FFmpeg, and what is bounded
//!
//! FFmpeg does the container and codec work; this crate does the scheduling,
//! resizing, buffer reuse and GPU handoff. Every non-streaming process has a
//! deadline and output caps, decodes are scaled to the caller's target instead
//! of the source, and playback processes are killed when their stream is
//! dropped. `ffmpeg` and `ffprobe` must be on `PATH`, beside the executable, in
//! an `ffmpeg/` folder beside it, or named by `DREAM_FFMPEG` /
//! `DREAM_FFPROBE` / `DREAM_FFMPEG_DIR`.
//!
//! Audio uses its own CPAL output stream per player. Subtitle tracks, playlists,
//! network protocols and frame-accurate editing are deliberately unsupported:
//! this is a player, not a pipeline.
#![warn(missing_docs)]

mod audio;
mod clock;
mod error;
mod geom;
mod image;
mod player;
mod probe;
mod process;
mod stream;
mod tools;

#[cfg(feature = "gpu")]
mod gpu;

pub use error::{Error, Result};
pub use image::Image;
pub use player::{Frame, Player, PlayerOptions, Status};
pub use probe::{MediaInfo, probe, probe_with};
pub use tools::Tools;

#[cfg(feature = "gpu")]
pub use gpu::{VideoPlayer, VideoSurface};
