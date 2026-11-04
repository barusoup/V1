//! Native Vulkan renderer.
//!
//! Chain: scene pass -> FSR 3.1 temporal upscale (see `fsr3.rs`) -> blit to the
//! swapchain.
//!
//! All offscreen images live in `GENERAL` layout, which is legal for every use
//! and lets us skip per-image layout bookkeeping. The swapchain image is the
//! only one that gets explicit transitions.
//!
//! Coordinate convention: shaders author to wgpu-style NDC (y up, depth 0..1),
//! which we preserve in Vulkan by using a negative viewport height.

mod acquire;
mod draw;
mod frame;
mod pipeline;
mod readback;
mod renderer;
mod shader;
mod thumbnail;
mod timing;
pub mod vulkan;

pub use vulkan::{GpuBuffer, GpuContext, GpuImage, PulledMesh};

use crate::fsr3::{Fsr3, JitterSequence};
use ash::vk;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const SCENE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
pub const VELOCITY_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;

/// Mirror of the errors the swapchain path can report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceError {
    Lost,
    Outdated,
    OutOfMemory,
    Timeout,
    Other,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorVertex {
    pub position: [f32; 3],
    pub color: [u8; 4],
}

/// Scene-pass uniform block. The scene renders with the jittered projection;
/// motion vectors use the two unjittered projections.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewUniform {
    view_projection: [[f32; 4]; 4],
    curr_view_projection: [[f32; 4]; 4],
    prev_view_projection: [[f32; 4]; 4],
}

/// Summary of one rendered frame, read back from the GPU. Used by the game's
/// `DREAM_FRAME_STATS` hook and by scripted capture runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMetrics {
    pub size: (u32, u32),
    pub mean: [f32; 3],
    pub sky_pct: f32,
    pub sky_top_pct: f32,
    pub sky_bottom_pct: f32,
    pub nonblack_pct: f32,
    /// Largest motion-vector magnitude in the render-resolution target, in
    /// normalized render-target units.
    pub mv_max: f32,
    /// Mean motion-vector magnitude.
    pub mv_mean: f32,
    /// Fraction of motion vectors pointing more than a quarter of the screen in
    /// one frame, which for this scene means a broken projection.
    pub mv_large_pct: f32,
    /// Fraction of motion vectors with a non-negligible magnitude.
    pub mv_nonzero_pct: f32,
    /// Mean of each channel of the FSR dilated reactive masks, at render
    /// resolution: REACTIVE, DISOCCLUSION, SHADING_CHANGE, ACCUMULATION. All
    /// four should fall back to ~0 once the camera stops; a channel that stays
    /// high is a permanent bias on the upscaled output rather than a transient.
    pub reactive_mean: [f32; 4],
}

impl FrameMetrics {
    pub fn summary(&self) -> String {
        format!(
            "{}x{} mean=({:.3},{:.3},{:.3}) sky={:.1}% (top={:.1}% bottom={:.1}%) nonblack={:.1}% mv_max={:.4} mv_mean={:.5} mv_large={:.2}% mv_nonzero={:.1}% reactive={:.4} disocclusion={:.4} shading={:.4} accumulation={:.4}",
            self.size.0,
            self.size.1,
            self.mean[0],
            self.mean[1],
            self.mean[2],
            self.sky_pct,
            self.sky_top_pct,
            self.sky_bottom_pct,
            self.nonblack_pct,
            self.mv_max,
            self.mv_mean,
            self.mv_large_pct,
            self.mv_nonzero_pct,
            self.reactive_mean[0],
            self.reactive_mean[1],
            self.reactive_mean[2],
            self.reactive_mean[3],
        )
    }
}

/// CPU-side time the last `render` call spent at each engine checkpoint, in
/// microseconds. A frame whose wall time is large while every game-side timer
/// is small blocked inside the renderer: `token_wait` is the frame-start stall
/// the app feels, `fence_wait` means the GPU was behind, `acquire` or `present`
/// means the presentation engine (compositor) held the swapchain (both run
/// off the frame's critical path or after it). `record` includes the caller's
/// draw callback.
#[derive(Clone, Copy, Debug, Default)]
pub struct FramePhases {
    /// Time the frame's slot fence took to signal, measured on the
    /// acquire-ahead worker (the acquire is gated on that fence).
    pub fence_wait_us: f32,
    /// Time the worker spent inside `vkAcquireNextImageKHR` for this frame's
    /// image -- the presentation engine holding the swapchain. Off the
    /// frame's critical path; the frame only pays the residual in
    /// [`Self::token_wait_us`].
    pub acquire_us: f32,
    /// Time the frame itself blocked waiting for its token: the frame-start
    /// stall the app feels. Near zero when the acquire-ahead kept up; a
    /// skipped frame reports its whole wait budget here.
    pub token_wait_us: f32,
    pub record_us: f32,
    pub submit_us: f32,
    pub present_us: f32,
}

struct SceneTargets {
    color: GpuImage,
    velocity: GpuImage,
    depth: GpuImage,
}

/// Draw commands handed to the game for the scene pass.
pub struct DrawCtx<'a> {
    device: &'a ash::Device,
    cmd: vk::CommandBuffer,
    scene_pipeline: vk::Pipeline,
    scene_layout: vk::PipelineLayout,
    scene_set: vk::DescriptorSet,
    pulling: Option<(vk::Pipeline, vk::PipelineLayout)>,
    frame_slot: usize,
    frame_id: u64,
    starter: std::cell::RefCell<Option<&'a mut dyn FnMut()>>,
}

pub struct Renderer {
    ctx: Arc<GpuContext>,
    swapchain: vulkan::Swapchain,

    scene_pipeline: vk::Pipeline,
    scene_layout: vk::PipelineLayout,
    scene_dsl: vk::DescriptorSetLayout,
    /// Vertex-pulling pipeline variant: zero vertex input, quad storage at
    /// set 1, a 16-byte vertex-stage origin push constant. Present only when
    /// the renderer was built by [`Self::new_with_pulling`].
    pulling: Option<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)>,
    blit_pipeline: vk::Pipeline,
    blit_layout: vk::PipelineLayout,
    blit_dsl: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,

    /// Per frame slot: slot N's set/uniform are only touched after its
    /// fence recycled, so a submitted frame never sees them mutate.
    scene_sets: Vec<vk::DescriptorSet>,
    blit_set: vk::DescriptorSet,

    sampler: vk::Sampler,
    scene_uniforms: Vec<GpuBuffer>,

    // Per-frame resources.
    frame_pool: vk::CommandPool,
    cmd: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    /// Serial carried by each slot's fence; confirmed after the wait.
    frame_serials: Vec<u64>,
    frame: usize,
    /// Monotonic frame counter handed to [`DrawCtx::frame_id`]; bumped only
    /// after a submission succeeds, so a failed frame does not retire slot
    /// resources it never used.
    frame_id: u64,
    /// Per-slot timestamp query pair bracketing the scene pass (0 = before
    /// begin_rendering, 1 = after end_rendering). `scene_ms[slot]` holds the
    /// measured GPU time of the last completed frame on that slot.
    gpu_pool: vk::QueryPool,
    scene_ms: Vec<f32>,
    scene_sample: Option<(u64, f32)>,
    scene_recorded: Vec<bool>,

    targets: SceneTargets,
    targets_ready: bool,
    /// GPU downsample of the scene target for card thumbnails (built on demand).
    thumb: Option<thumbnail::ThumbnailPass>,
    fsr: Option<Fsr3>,
    fsr_enabled: bool,
    /// Composed display image (swapchain format, screen-row order): the blit
    /// writes it, the UI pass draws over it, the present pass copies it to
    /// the swapchain. Captures read it, so screenshots include the UI.
    display: GpuImage,
    present_pipeline: vk::Pipeline,
    present_layout: vk::PipelineLayout,
    present_set: vk::DescriptorSet,
    /// UI overlay pass; recorded between the blit and the present pass.
    ui: crate::ui::render::UiRenderer,
    /// Colour space the UI pass blends in (fixed at construction: it
    /// decides the display image's format and the pipelines' entries).
    ui_blend: crate::ui::render::UiBlend,
    display_size: (u32, u32),
    render_size: (u32, u32),
    jitter: JitterSequence,
    prev_view_projection: glam::Mat4,
    history_valid: bool,
    temporal: bool,
    /// Draw the centred plus crosshair in the present pass. Off by default.
    show_crosshair: bool,
    sharpness: f32,
    render_scale: f32,
    last_frame: Instant,
    /// Segment timings of the most recent completed `render*` call.
    phases: FramePhases,
    /// Managed acquire-ahead worker: resolves the next frame's swapchain
    /// image (gated on the slot's fence) while the app keeps running.
    acquire: acquire::AcquireWorker,
    /// Wall-clock bound on the frame-start wait for the token. `None` -- the
    /// default -- waits as long as the presentation engine needs, capped at
    /// the liveness bound (`acquire::UNBUDGETED_WAIT_CAP`).
    frame_wait_budget: Option<Duration>,
}
