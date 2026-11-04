//! Dream GPU bridge: a small ring of textures uploaded asynchronously from
//! video frames, and a convenience wrapper around [`Player`].

use crate::Result;
use crate::player::{Frame, Player, PlayerOptions, Status};
use crate::probe::MediaInfo;
use crate::tools::Tools;
use dream::gfx::vulkan::GpuContext;
use dream::texture::{GpuTexture, SamplerDesc, TextureFormat, TextureOptions, TextureUploader};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Textures two frames alternate between. One is being sampled while the
/// other receives the next frame; queue order makes the update safe without a
/// CPU wait. Dream keeps at most three frames in flight, and an update is
/// always submitted after the earlier frames that sampled that image.
const RING: usize = 2;

/// A GPU texture ring fed by [`Frame`]s.
///
/// Uploads are asynchronous. Draw the texture returned by [`VideoSurface::upload`] in the
/// frame you submit next; the copy is already queued ahead of that frame.
pub struct VideoSurface {
    uploader: TextureUploader,
    textures: Vec<Arc<GpuTexture>>,
    next: usize,
    size: (u32, u32),
    ready: Option<usize>,
}

impl VideoSurface {
    /// Create an empty ring on `ctx`. Textures are allocated on the first
    /// frame, once its size is known.
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        Self {
            uploader: TextureUploader::new(ctx),
            textures: Vec::with_capacity(RING),
            next: 0,
            size: (0, 0),
            ready: None,
        }
    }

    /// Upload one frame into the ring and return the texture to draw.
    pub fn upload(&mut self, frame: &Frame) -> Arc<GpuTexture> {
        let size = frame.size();
        if self.textures.len() != RING || self.size != size {
            self.rebuild(size);
        }
        let index = self.next;
        self.next = (self.next + 1) % RING;
        let texture = Arc::clone(&self.textures[index]);
        self.uploader.upload(&texture, frame.rgba());
        self.ready = Some(index);
        Arc::clone(&texture)
    }

    /// The most recently uploaded texture, if a frame has arrived.
    pub fn texture(&self) -> Option<&Arc<GpuTexture>> {
        self.ready.and_then(|index| self.textures.get(index))
    }

    /// Size of the current texture, if a frame has arrived.
    pub fn size(&self) -> Option<(u32, u32)> {
        self.ready.map(|_| self.size)
    }

    fn rebuild(&mut self, size: (u32, u32)) {
        // Old textures may still be retained by UI frame slots; dropping our
        // Arcs simply lets those slots finish. Before freeing the ones nobody
        // retained, wait for the queue so no in-flight draw can still read
        // them. This only happens on a size change.
        if !self.textures.is_empty() {
            self.uploader.wait_idle();
            self.uploader.context().one_shot(|_| {});
        }
        self.textures.clear();
        self.size = size;
        self.next = 0;
        self.ready = None;
        for index in 0..RING {
            self.textures.push(Arc::new(GpuTexture::blank(
                self.uploader_context(),
                &format!("media video {index}"),
                size.0,
                size.1,
                TextureOptions {
                    format: TextureFormat::Srgb,
                    sampler: SamplerDesc::default(),
                    mip_levels: Some(1),
                },
            )));
        }
    }

    fn uploader_context(&self) -> &Arc<GpuContext> {
        self.uploader.context()
    }
}

/// A [`Player`] plus a [`VideoSurface`], ready to hand a Dream UI the current
/// video texture.
pub struct VideoPlayer {
    player: Player,
    surface: VideoSurface,
}

impl VideoPlayer {
    /// Open with the process-wide FFmpeg discovery and a Dream GPU context.
    pub fn open(
        ctx: &Arc<GpuContext>,
        source: impl AsRef<Path>,
        options: PlayerOptions,
    ) -> Result<Self> {
        Self::open_with(&Tools::discover()?, ctx, source, options)
    }

    /// Open with explicit FFmpeg paths.
    pub fn open_with(
        tools: &Tools,
        ctx: &Arc<GpuContext>,
        source: impl AsRef<Path>,
        options: PlayerOptions,
    ) -> Result<Self> {
        let player = Player::open_with(tools, source, options)?;
        Ok(Self {
            player,
            surface: VideoSurface::new(ctx),
        })
    }

    /// Advance playback; returns a freshly uploaded texture when a new frame
    /// is due.
    pub fn poll(&mut self, now: Instant) -> Option<Arc<GpuTexture>> {
        let frame = self.player.poll(now)?;
        Some(self.surface.upload(&frame))
    }

    /// The texture of the last frame uploaded by [`VideoPlayer::poll`].
    pub fn texture(&self) -> Option<&Arc<GpuTexture>> {
        self.surface.texture()
    }

    /// The texture ring, for apps that want to upload or inspect it directly.
    pub fn surface(&self) -> &VideoSurface {
        &self.surface
    }

    /// The underlying decoder/clock engine.
    pub fn player(&self) -> &Player {
        &self.player
    }

    /// The underlying decoder/clock engine, mutably for advanced control.
    pub fn player_mut(&mut self) -> &mut Player {
        &mut self.player
    }

    /// The probed metadata.
    pub fn info(&self) -> &MediaInfo {
        self.player.info()
    }

    /// Current playback state.
    pub fn status(&self) -> Status {
        self.player.status()
    }

    /// Duration in milliseconds, when known.
    pub fn duration_ms(&self) -> Option<i64> {
        self.player.duration_ms()
    }

    /// Position at `now`, including a pending seek target.
    pub fn position_ms(&self, now: Instant) -> i64 {
        self.player.position_ms(now)
    }

    /// Resume or restart after the end.
    pub fn play(&mut self, now: Instant) {
        self.player.play(now);
    }

    /// Hold the current frame and stop audio.
    pub fn pause(&mut self, now: Instant) {
        self.player.pause(now);
    }

    /// Pause if playing, play if paused.
    pub fn toggle(&mut self, now: Instant) {
        self.player.toggle(now);
    }

    /// Seek to `ms`, coalesced while dragging.
    pub fn seek(&mut self, ms: i64) {
        self.player.seek(ms);
    }

    /// Set output gain, clamped to `0..=1`.
    pub fn set_volume(&mut self, volume: f32) {
        self.player.set_volume(volume);
    }

    /// Current output gain.
    pub fn volume(&self) -> f32 {
        self.player.volume()
    }

    /// Silence output without stopping playback.
    pub fn set_muted(&mut self, muted: bool) {
        self.player.set_muted(muted);
    }

    /// Whether output is muted.
    pub fn muted(&self) -> bool {
        self.player.muted()
    }

    /// Change the decode bounding box; restarts at the current position only
    /// when the decoded size changes.
    pub fn set_max_size(&mut self, max_size: (u32, u32), now: Instant) {
        self.player.set_max_size(max_size, now);
    }

    /// When `poll` should run next if only time will produce work.
    pub fn next_wake(&self, now: Instant) -> Option<Instant> {
        self.player.next_wake(now)
    }
}

impl Drop for VideoSurface {
    fn drop(&mut self) {
        if !self.textures.is_empty() {
            self.uploader.wait_idle();
            self.uploader.context().one_shot(|_| {});
        }
    }
}
