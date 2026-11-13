use super::metal::pipeline::{
    VertexAttr, create_depth_stencil_state, create_render_pipeline, display_format,
    make_vertex_descriptor, present_fragment_entry, scaled_size,
};
use super::metal::shader::{MslBindings, PUSH_CONSTANT_BUFFER_SLOT};
use super::metal::swapchain::{PresentMode, Swapchain, SwapchainOptions};
use super::{
    ColorVertex, DEPTH_FORMAT, FramePhases, GpuBuffer, GpuContext, GpuImage, Renderer,
    SCENE_FORMAT, SceneTargets, VELOCITY_FORMAT, ViewUniform,
};
use crate::fsr3::{Fsr3, JitterSequence};
use crate::settings::{AaMode, Settings};
use crate::ui::render::UiBlend;
use objc2_metal::{MTLCompareFunction, MTLPixelFormat, MTLTextureUsage, MTLVertexFormat};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use winit::{dpi::PhysicalSize, window::Window};

const MAX_FRAMES_IN_FLIGHT: usize = 3;

pub(crate) struct SlotSync {
    pub(crate) state: Mutex<SlotState>,
    pub(crate) cv: Condvar,
}

pub(crate) struct SlotState {
    pub(crate) done: bool,
    pub(crate) serial: u64,
    pub(crate) gpu_ms: f32,
}

impl SlotSync {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SlotState {
                done: true,
                serial: 0,
                gpu_ms: 0.0,
            }),
            cv: Condvar::new(),
        })
    }

    pub(crate) fn wait(&self, budget: Option<Duration>) -> Result<(u64, f32), ()> {
        let mut guard = self.state.lock().unwrap();
        let cap = budget.unwrap_or(Duration::from_millis(250));
        while !guard.done {
            let (g, res) = self.cv.wait_timeout(guard, cap).unwrap();
            guard = g;
            if res.timed_out() && !guard.done {
                return Err(());
            }
        }
        Ok((guard.serial, guard.gpu_ms))
    }

    pub(crate) fn mark_pending(&self, serial: u64) {
        let mut guard = self.state.lock().unwrap();
        guard.done = false;
        guard.serial = serial;
        guard.gpu_ms = 0.0;
    }

    pub(crate) fn mark_done(&self, gpu_ms: f32) {
        let mut guard = self.state.lock().unwrap();
        guard.done = true;
        guard.gpu_ms = gpu_ms;
        self.cv.notify_all();
    }
}

fn build_targets(ctx: &Arc<GpuContext>, size: (u32, u32)) -> SceneTargets {
    let (w, h) = (size.0.max(1), size.1.max(1));
    let color_usage = MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead;
    let depth_usage = MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead;
    SceneTargets {
        color: ctx.create_image("scene color", w, h, SCENE_FORMAT, color_usage),
        velocity: ctx.create_image("scene velocity", w, h, VELOCITY_FORMAT, color_usage),
        depth: ctx.create_image("scene depth", w, h, DEPTH_FORMAT, depth_usage),
    }
}

fn build_display(ctx: &Arc<GpuContext>, size: (u32, u32), format: MTLPixelFormat) -> GpuImage {
    ctx.create_image(
        "display target",
        size.0.max(1),
        size.1.max(1),
        format,
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
    )
}

fn swapchain_options(present_mode: PresentMode, blend: UiBlend) -> SwapchainOptions {
    SwapchainOptions {
        present_mode,
        prefer_unorm: blend == UiBlend::Gamma,
    }
}

impl Renderer {
    pub fn new(window: Arc<Window>, shader_src: &str, settings: Settings) -> Self {
        Self::new_inner(window, shader_src, None, settings, UiBlend::default())
    }

    pub fn new_with_ui_blend(
        window: Arc<Window>,
        shader_src: &str,
        settings: Settings,
        blend: UiBlend,
    ) -> Self {
        Self::new_inner(window, shader_src, None, settings, blend)
    }

    pub fn new_with_pulling(
        window: Arc<Window>,
        shader_src: &str,
        pulling_shader_src: &str,
        settings: Settings,
    ) -> Self {
        Self::new_inner(
            window,
            shader_src,
            Some(pulling_shader_src),
            settings,
            UiBlend::default(),
        )
    }

    fn new_inner(
        window: Arc<Window>,
        shader_src: &str,
        pulling_shader_src: Option<&str>,
        settings: Settings,
        ui_blend: UiBlend,
    ) -> Self {
        let size = window.inner_size();
        let display_size = (size.width.max(1), size.height.max(1));
        let ctx = GpuContext::new_owned(window);

        let temporal = matches!(settings.aa, AaMode::Fsr);
        let render_scale = if temporal {
            settings.render_scale().clamp(0.1, 1.0)
        } else {
            1.0
        };
        let present_mode = PresentMode::default();
        let swapchain = Swapchain::new(
            &ctx,
            display_size.0,
            display_size.1,
            swapchain_options(present_mode, ui_blend),
        );
        let display_size = (swapchain.width, swapchain.height);
        let render_size = scaled_size(display_size, render_scale);
        let display_fmt = display_format(swapchain.format, ui_blend);

        let scene_uniforms: Vec<GpuBuffer> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                ctx.create_host_buffer("camera uniform", std::mem::size_of::<ViewUniform>() as u64)
            })
            .collect();

        let scene_bindings = MslBindings::new().buffer(0, 0, 0, false);
        let vertex_desc = make_vertex_descriptor(
            std::mem::size_of::<ColorVertex>(),
            &[
                VertexAttr {
                    location: 0,
                    format: MTLVertexFormat::Float3,
                    offset: 0,
                },
                VertexAttr {
                    location: 1,
                    format: MTLVertexFormat::UChar4Normalized,
                    offset: 12,
                },
            ],
        );
        let scene_pipeline = create_render_pipeline(
            &ctx,
            "scene pipeline",
            shader_src,
            &scene_bindings,
            "vs_main",
            "fs_main",
            &[SCENE_FORMAT, VELOCITY_FORMAT],
            Some(DEPTH_FORMAT),
            Some(&vertex_desc),
            false,
        );
        let scene_depth_stencil =
            create_depth_stencil_state(&ctx, true, true, MTLCompareFunction::Less);
        let no_depth_stencil =
            create_depth_stencil_state(&ctx, false, false, MTLCompareFunction::Always);

        let pulling = pulling_shader_src.map(|src| {
            let pull_bindings = MslBindings::new()
                .buffer(0, 0, 0, false)
                .buffer(1, 0, 1, false)
                .push_constants(PUSH_CONSTANT_BUFFER_SLOT);
            create_render_pipeline(
                &ctx,
                "scene pulling pipeline",
                src,
                &pull_bindings,
                "vs_main",
                "fs_main",
                &[SCENE_FORMAT, VELOCITY_FORMAT],
                Some(DEPTH_FORMAT),
                None,
                false,
            )
        });

        let blit_bindings = MslBindings::new().texture(0, 0, 0).sampler(0, 1, 0);
        let blit_fs = if ui_blend == UiBlend::Gamma {
            "fs_gamma"
        } else {
            "fs_main"
        };
        let blit_pipeline = create_render_pipeline(
            &ctx,
            "blit pipeline",
            include_str!("../../blit.wgsl"),
            &blit_bindings,
            "vs_main",
            blit_fs,
            &[display_fmt],
            None,
            None,
            false,
        );

        let present_bindings = MslBindings::new()
            .texture(0, 0, 0)
            .sampler(0, 1, 0)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);
        let present_fs = present_fragment_entry(ui_blend, display_fmt);
        let present_pipeline = create_render_pipeline(
            &ctx,
            "present pipeline",
            include_str!("../../present.wgsl"),
            &present_bindings,
            "vs_main",
            present_fs,
            &[swapchain.format],
            None,
            None,
            false,
        );

        let targets = build_targets(&ctx, (1, 1));
        let display = build_display(&ctx, (1, 1), display_fmt);
        let ui = crate::ui::render::UiRenderer::new(&ctx, display_fmt, 512);
        let slots = (0..MAX_FRAMES_IN_FLIGHT).map(|_| SlotSync::new()).collect();

        Self {
            ctx,
            swapchain,
            scene_pipeline,
            scene_depth_stencil,
            no_depth_stencil,
            pulling,
            blit_pipeline,
            present_pipeline,
            scene_uniforms,
            slots,
            frame: 0,
            frame_id: 0,
            scene_ms: vec![0.0; MAX_FRAMES_IN_FLIGHT],
            scene_sample: None,
            scene_recorded: vec![false; MAX_FRAMES_IN_FLIGHT],
            targets,
            targets_ready: false,
            thumb: None,
            fsr: None,
            fsr_enabled: temporal,
            display,
            display_live: false,
            display_written: false,
            frame_capture: false,
            present_mode,
            capture_notice: false,
            ui,
            ui_blend,
            display_size,
            render_size,
            jitter: JitterSequence::default(),
            prev_view_projection: glam::Mat4::IDENTITY,
            history_valid: false,
            temporal,
            show_crosshair: false,
            sharpness: settings.sharpness,
            render_scale,
            last_frame: Instant::now(),
            phases: FramePhases::default(),
            frame_wait_budget: None,
        }
    }

    pub(crate) fn needs_display(&self) -> bool {
        let display_fmt = display_format(self.swapchain.format, self.ui_blend);
        !super::pipeline::composes_direct(
            self.swapchain.format,
            display_fmt,
            self.show_crosshair,
            self.frame_capture,
        )
    }

    pub(crate) fn ensure_display(&mut self) {
        if self.display_live {
            return;
        }
        let display_fmt = display_format(self.swapchain.format, self.ui_blend);
        self.display = build_display(&self.ctx, self.display_size, display_fmt);
        self.display_live = true;
        self.display_written = false;
    }

    pub(crate) fn release_display(&mut self) {
        if !self.display_live {
            return;
        }
        let display_fmt = display_format(self.swapchain.format, self.ui_blend);
        self.display = build_display(&self.ctx, (1, 1), display_fmt);
        self.display_live = false;
        self.display_written = false;
    }

    pub(crate) fn ensure_scene_targets(&mut self) {
        if self.targets_ready {
            return;
        }
        self.targets = build_targets(&self.ctx, self.render_size);
        self.targets_ready = true;
        if let Some(thumb) = self.thumb.as_mut() {
            thumb.bind_color(self.targets.color.texture.clone());
        }
        if self.fsr_enabled {
            let mut fsr = Fsr3::new(&self.ctx, self.render_size, self.display_size);
            fsr.set_inputs([
                &self.targets.color.texture,
                &self.targets.depth.texture,
                &self.targets.velocity.texture,
            ]);
            self.fsr = Some(fsr);
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.ctx.wait_idle();
        self.swapchain.resize(&self.ctx, size.width, size.height);
        self.display_size = (self.swapchain.width, self.swapchain.height);
        self.render_size = scaled_size(self.display_size, self.render_scale);
        if self.targets_ready {
            self.targets = build_targets(&self.ctx, self.render_size);
            if let Some(thumb) = self.thumb.as_mut() {
                thumb.bind_color(self.targets.color.texture.clone());
            }
            if let Some(fsr) = self.fsr.as_mut() {
                fsr.resize(self.render_size, self.display_size);
                fsr.set_inputs([
                    &self.targets.color.texture,
                    &self.targets.depth.texture,
                    &self.targets.velocity.texture,
                ]);
            }
        }
        if self.needs_display() {
            let display_fmt = display_format(self.swapchain.format, self.ui_blend);
            self.display = build_display(&self.ctx, self.display_size, display_fmt);
            self.display_live = true;
            self.display_written = false;
        }
        self.history_valid = false;
        self.jitter.reset();
    }

    pub fn recreate_swapchain(&mut self) {
        self.resize(PhysicalSize::new(self.display_size.0, self.display_size.1));
    }

    pub fn set_frame_wait_budget(&mut self, budget: Option<Duration>) {
        self.frame_wait_budget = budget;
    }

    pub fn frame_wait_budget(&self) -> Option<Duration> {
        self.frame_wait_budget
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    pub fn set_show_crosshair(&mut self, show: bool) {
        if self.show_crosshair == show {
            return;
        }
        self.show_crosshair = show;
        if self.needs_display() {
            self.ensure_display();
        } else {
            self.release_display();
        }
    }

    pub fn set_frame_capture(&mut self, enabled: bool) {
        if self.frame_capture == enabled {
            return;
        }
        self.frame_capture = enabled;
        self.capture_notice = false;
        if self.needs_display() {
            self.ensure_display();
        } else {
            self.release_display();
        }
    }

    pub fn frame_capture(&self) -> bool {
        self.frame_capture
    }

    pub fn set_present_mode(&mut self, mode: PresentMode) {
        if self.present_mode == mode {
            return;
        }
        self.present_mode = mode;
        self.swapchain.set_present_mode(&self.ctx, mode);
    }

    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    pub fn trim_memory(&mut self) {
        self.ctx.wait_idle();
        self.ui.trim();
        self.ctx.collect_retired_buffers();
        crate::memory::trim_working_set();
    }

    pub fn set_ui_instancing(&mut self, enabled: bool) {
        self.ui.set_instancing(enabled);
    }

    pub fn ui_allocation_sizes(&self) -> Vec<(String, u64)> {
        self.ui.allocation_sizes()
    }

    pub fn ui_last_geometry_upload_sizes(&self) -> (u64, u64, u64) {
        self.ui.last_geometry_upload_sizes()
    }

    pub fn ui_last_prepared_counts(&self) -> (u64, u64, u64) {
        self.ui.last_prepared_counts()
    }

    pub fn set_temporal(&mut self, enabled: bool) {
        if self.temporal == enabled {
            return;
        }
        self.temporal = enabled;
        self.fsr_enabled = enabled;
        self.history_valid = false;
        self.jitter.reset();
        if enabled && self.targets_ready && self.fsr.is_none() {
            let mut fsr = Fsr3::new(&self.ctx, self.render_size, self.display_size);
            fsr.set_inputs([
                &self.targets.color.texture,
                &self.targets.depth.texture,
                &self.targets.velocity.texture,
            ]);
            self.fsr = Some(fsr);
        }
    }

    pub fn toggle_temporal(&mut self) {
        self.set_temporal(!self.temporal);
    }

    pub fn reset_history(&mut self) {
        self.history_valid = false;
        self.jitter.reset();
    }

    pub fn has_temporal(&self) -> bool {
        self.fsr_enabled
    }

    pub fn scene_resources_loaded(&self) -> bool {
        self.targets_ready
    }

    pub fn release_scene_resources(&mut self) -> bool {
        if !self.targets_ready {
            return false;
        }
        if self.thumb.as_ref().is_some_and(|t| t.busy()) {
            return false;
        }
        self.ctx.wait_idle();
        self.targets = build_targets(&self.ctx, (1, 1));
        self.targets_ready = false;
        self.fsr = None;
        self.history_valid = false;
        self.jitter.reset();
        true
    }

    pub fn reset_temporal_history(&mut self) {
        self.history_valid = false;
        self.jitter.reset();
    }

    pub fn history_valid(&self) -> bool {
        self.history_valid
    }

    pub fn display_size(&self) -> (u32, u32) {
        self.display_size
    }

    pub fn render_size(&self) -> (u32, u32) {
        self.render_size
    }

    pub fn temporal(&self) -> bool {
        self.temporal
    }
}
