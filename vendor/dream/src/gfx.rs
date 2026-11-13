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

#[cfg(not(target_os = "macos"))]
mod acquire;
#[cfg_attr(target_os = "macos", path = "gfx/metal/draw.rs")]
mod draw;
#[cfg_attr(target_os = "macos", path = "gfx/metal/frame.rs")]
mod frame;
#[cfg(not(target_os = "macos"))]
mod pipeline;
pub mod present;
#[cfg(target_os = "macos")]
use metal::pipeline;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg_attr(target_os = "macos", path = "gfx/metal/readback.rs")]
mod readback;
#[cfg_attr(target_os = "macos", path = "gfx/metal/renderer.rs")]
pub(crate) mod renderer;
mod shader;
#[cfg_attr(target_os = "macos", path = "gfx/metal/thumbnail.rs")]
mod thumbnail;
#[cfg(not(target_os = "macos"))]
mod timing;
pub mod vulkan;

#[cfg(target_os = "macos")]
pub(crate) use metal::align_up;
#[cfg(target_os = "macos")]
pub use metal::{DynamicBuffers, GpuBuffer, GpuContext, GpuImage, PulledMesh};
pub use present::PresentMode;
#[cfg(not(target_os = "macos"))]
pub use vulkan::{DynamicBuffers, GpuBuffer, GpuContext, GpuImage, PulledMesh};

use crate::fsr3::{Fsr3, JitterSequence};
#[cfg(not(target_os = "macos"))]
use ash::vk;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(not(target_os = "macos"))]
pub const SCENE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
#[cfg(not(target_os = "macos"))]
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
#[cfg(not(target_os = "macos"))]
pub const VELOCITY_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;

#[cfg(target_os = "macos")]
pub const SCENE_FORMAT: metal::PixelFormat = metal::PixelFormat::RGBA16Float;
#[cfg(target_os = "macos")]
pub const DEPTH_FORMAT: metal::PixelFormat = metal::PixelFormat::Depth32Float;
#[cfg(target_os = "macos")]
pub const VELOCITY_FORMAT: metal::PixelFormat = metal::PixelFormat::RG16Float;

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

#[cfg(not(target_os = "macos"))]
struct SceneTargets {
    color: GpuImage,
    velocity: GpuImage,
    depth: GpuImage,
}

#[cfg(target_os = "macos")]
pub(crate) struct SceneTargets {
    pub(crate) color: GpuImage,
    pub(crate) velocity: GpuImage,
    pub(crate) depth: GpuImage,
}

/// Draw commands handed to the game for the scene pass.
#[cfg(not(target_os = "macos"))]
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

/// Draw commands handed to the game for the scene pass (native Metal).
#[cfg(target_os = "macos")]
pub struct DrawCtx<'a> {
    pub(crate) encoder:
        &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
    pub(crate) scene_pipeline:
        &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    pub(crate) scene_depth_stencil:
        &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDepthStencilState>,
    pub(crate) scene_uniform: &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    pub(crate) pulling:
        Option<&'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>,
    pub(crate) index_buffer: Option<(
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        u64,
    )>,
    pub(crate) frame_slot: usize,
    pub(crate) frame_id: u64,
    pub(crate) starter: std::cell::RefCell<Option<&'a mut dyn FnMut()>>,
}

#[cfg(not(target_os = "macos"))]
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
    /// One per SWAPCHAIN IMAGE (indexed by the acquired image index), not per
    /// frame slot: a presented image's semaphore stays referenced by the
    /// presentation engine until that image is re-acquired, so reusing one per
    /// slot violates VUID-vkQueueSubmit-pSignalSemaphores-00067. Rebuilt with
    /// the swapchain in `Renderer::resize`.
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
    /// Composed display image (swapchain format, screen-row order), used only
    /// by frames that cannot be composed straight into the swapchain: a
    /// crosshair, frame capture, or a colour-space change. There the blit
    /// writes it, the UI pass draws over it and the present pass copies it to
    /// the swapchain. Otherwise it is a 1x1 placeholder that the present set
    /// can still sample.
    display: GpuImage,
    /// Whether `display` is the full-size image rather than the placeholder.
    display_live: bool,
    /// Whether a frame has been composed into `display` since it was built:
    /// the only state in which the display holds something to read back.
    display_written: bool,
    /// Keep the composed frame for CPU readbacks (see
    /// [`Renderer::set_frame_capture`]).
    frame_capture: bool,
    /// Requested swapchain present mode (see [`Renderer::set_present_mode`]).
    present_mode: vulkan::PresentMode,
    /// Set once a readback has found no retained frame and said so.
    capture_notice: bool,
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

#[cfg(target_os = "macos")]
pub struct Renderer {
    pub(crate) ctx: Arc<GpuContext>,
    pub(crate) swapchain: metal::Swapchain,
    pub(crate) scene_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
    pub(crate) scene_depth_stencil:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>,
    pub(crate) no_depth_stencil:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>,
    pub(crate) pulling: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
        >,
    >,
    pub(crate) blit_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
    pub(crate) present_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
    pub(crate) scene_uniforms: Vec<GpuBuffer>,
    pub(crate) slots: Vec<Arc<renderer::SlotSync>>,
    pub(crate) frame: usize,
    pub(crate) frame_id: u64,
    pub(crate) scene_ms: Vec<f32>,
    pub(crate) scene_sample: Option<(u64, f32)>,
    pub(crate) scene_recorded: Vec<bool>,
    pub(crate) targets: SceneTargets,
    pub(crate) targets_ready: bool,
    pub(crate) thumb: Option<thumbnail::ThumbnailPass>,
    pub(crate) fsr: Option<Fsr3>,
    pub(crate) fsr_enabled: bool,
    pub(crate) display: GpuImage,
    pub(crate) display_live: bool,
    pub(crate) display_written: bool,
    pub(crate) frame_capture: bool,
    pub(crate) present_mode: PresentMode,
    pub(crate) capture_notice: bool,
    pub(crate) ui: crate::ui::render::UiRenderer,
    pub(crate) ui_blend: crate::ui::render::UiBlend,
    pub(crate) display_size: (u32, u32),
    pub(crate) render_size: (u32, u32),
    pub(crate) jitter: JitterSequence,
    pub(crate) prev_view_projection: glam::Mat4,
    pub(crate) history_valid: bool,
    pub(crate) temporal: bool,
    pub(crate) show_crosshair: bool,
    pub(crate) sharpness: f32,
    pub(crate) render_scale: f32,
    pub(crate) last_frame: Instant,
    pub(crate) phases: FramePhases,
    pub(crate) frame_wait_budget: Option<Duration>,
}
