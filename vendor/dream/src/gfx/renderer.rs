#[cfg(doc)]
use super::PulledMesh;
use super::pipeline::{
    allocate_sets, binding, build_display, build_targets, create_descriptor_pool, create_dsl,
    create_graphics_pipeline, create_graphics_pipeline_fs, create_pipeline_layout, display_format,
    image_write, new_fence, new_semaphore, sampled, sampler_info, sampler_write, scaled_size,
    uniform_write, write_sets,
};
use super::vulkan;
use super::{
    ColorVertex, DEPTH_FORMAT, FramePhases, GpuContext, Renderer, SCENE_FORMAT, VELOCITY_FORMAT,
    ViewUniform,
};
use crate::camera::Camera;
use crate::fsr3::{Fsr3, JitterSequence};
use crate::settings::{AaMode, Settings};
use crate::ui::render::UiBlend;
use ash::vk;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{dpi::PhysicalSize, window::Window};

impl Renderer {
    pub fn new(window: Arc<Window>, shader_src: &str, settings: Settings) -> Self {
        Self::new_inner(window, shader_src, None, settings, UiBlend::Linear)
    }

    /// Like [`Self::new`], with the UI pass blending in `blend`'s colour
    /// space. [`UiBlend::Gamma`] composites like a web browser (the display
    /// image becomes UNORM holding sRGB-encoded values); scene output is
    /// encoded into it by the blit, so 3D content still shows correctly.
    pub fn new_with_ui_blend(
        window: Arc<Window>,
        shader_src: &str,
        settings: Settings,
        blend: UiBlend,
    ) -> Self {
        Self::new_inner(window, shader_src, None, settings, blend)
    }

    /// Like [`Self::new`], but additionally compiles `pulling_shader_src` as
    /// a zero-vertex-input pipeline that fetches packed quads from a set-1
    /// storage buffer and takes a 16-byte vertex-stage origin push constant.
    /// `DrawCtx::draw_pulled*` then draws [`PulledMesh`]es through it.
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
            UiBlend::Linear,
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
        let display_size = (size.width.max(1), size.height.max(1)); // requested
        // The context keeps the window alive until its own teardown destroys
        // the surface: the caller's Arc may drop as soon as `new` returns.
        let ctx = GpuContext::new_owned(window.clone());

        let temporal = matches!(settings.aa, AaMode::Fsr);
        let render_scale = if temporal {
            settings.render_scale().clamp(0.1, 1.0)
        } else {
            1.0
        };
        let swapchain = vulkan::Swapchain::new(
            &ctx,
            display_size.0,
            display_size.1,
            vk::SwapchainKHR::null(),
        );
        // The surface may clamp the requested extent — derive display
        // size from the swapchain that was actually created.
        let display_size = (
            swapchain.extent.width.max(1),
            swapchain.extent.height.max(1),
        );
        let render_size = scaled_size(display_size, render_scale);
        let display_fmt = display_format(swapchain.format, ui_blend);
        let gamma = ui_blend == UiBlend::Gamma;
        let sampler = ctx.create_sampler("scene sampler");
        let max_frames_init = ctx.max_frames_in_flight();
        let scene_uniforms: Vec<_> = (0..max_frames_init)
            .map(|_| {
                ctx.create_host_buffer("camera uniform", std::mem::size_of::<ViewUniform>() as u64)
            })
            .collect();

        let scene_dsl = create_dsl(
            &ctx,
            "scene layout",
            // Camera uniform: VERTEX|FRAGMENT — vertex shaders transform, and
            // the impostor fragment shader needs the matrices for frag_depth
            // and motion vectors (the ray itself comes from the lighting env).
            &[binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            )],
        );
        let blit_dsl = create_dsl(
            &ctx,
            "blit layout",
            &[
                binding(
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                binding(
                    1,
                    vk::DescriptorType::SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
            ],
        );

        let scene_shader = ctx.compile_shader(shader_src, "game shader");
        let blit_shader = ctx.compile_shader(include_str!("../blit.wgsl"), "blit");

        let scene_layout = create_pipeline_layout(&ctx, "scene layout", &[scene_dsl], &[]);
        let vertex_binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<ColorVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let vertex_attrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(12),
        ];
        let scene_pipeline = create_graphics_pipeline(
            &ctx,
            scene_shader,
            &[vertex_binding],
            &vertex_attrs,
            scene_layout,
            &[SCENE_FORMAT, VELOCITY_FORMAT],
            Some(DEPTH_FORMAT),
        );
        let blit_layout = create_pipeline_layout(&ctx, "blit layout", &[blit_dsl], &[]);
        let blit_pipeline = create_graphics_pipeline_fs(
            &ctx,
            blit_shader,
            &[],
            &[],
            blit_layout,
            &[display_fmt],
            None,
            if gamma { c"fs_gamma" } else { c"fs_main" },
        );

        let pulling = pulling_shader_src.map(|source| {
            // Set 1 binds one packed-quad storage buffer per mesh; the
            // 16-byte origin rides along as a vertex-stage push constant.
            let pulled_dsl = create_dsl(
                &ctx,
                "pulled quads",
                &[binding(
                    0,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::VERTEX,
                )],
            );
            let push = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(16);
            let layout =
                create_pipeline_layout(&ctx, "pulling layout", &[scene_dsl, pulled_dsl], &[push]);
            let shader = ctx.compile_shader(source, "pulling shader");
            let pipeline = create_graphics_pipeline(
                &ctx,
                shader,
                &[],
                &[],
                layout,
                &[SCENE_FORMAT, VELOCITY_FORMAT],
                Some(DEPTH_FORMAT),
            );
            unsafe { ctx.device().destroy_shader_module(shader, None) };
            (pipeline, layout, pulled_dsl)
        });

        unsafe {
            ctx.device().destroy_shader_module(scene_shader, None);
            ctx.device().destroy_shader_module(blit_shader, None);
        }

        let descriptor_pool = create_descriptor_pool(&ctx, ctx.max_frames_in_flight());
        let (scene_sets, blit_set, present_set) =
            allocate_sets(&ctx, descriptor_pool, &scene_dsl, &blit_dsl);

        let max_frames = ctx.max_frames_in_flight();
        let frame_pool = unsafe {
            ctx.device().create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family())
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .expect("create frame command pool");
        let cmd = unsafe {
            ctx.device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(frame_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(max_frames as u32),
            )
        }
        .expect("allocate frame command buffers");
        let image_available = (0..max_frames).map(|_| new_semaphore(&ctx)).collect();
        let render_finished = (0..max_frames).map(|_| new_semaphore(&ctx)).collect();
        let in_flight = (0..max_frames).map(|_| new_fence(&ctx)).collect();

        // Allocate a 1x1 placeholder until a frame actually records 3D scene
        // draws (or requests a scene readback/thumbnail): UI-only apps never
        // pay for full-resolution HDR color/velocity/depth or FSR 3.1 targets.
        let targets = build_targets(&ctx, (1, 1));
        let thumb = None;
        let fsr = None;

        // Composed display image + present pass + UI overlay. The blit writes
        // the display image, UI draws over it, the present pass copies it to
        // the swapchain.
        let display = build_display(&ctx, display_size, display_fmt);
        let present_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(12);
        let present_layout =
            create_pipeline_layout(&ctx, "present layout", &[blit_dsl], &[present_push]);
        let present_shader = ctx.compile_shader(include_str!("../present.wgsl"), "present");
        let present_pipeline = create_graphics_pipeline_fs(
            &ctx,
            present_shader,
            &[],
            &[],
            present_layout,
            &[swapchain.format],
            None,
            if gamma { c"fs_gamma" } else { c"fs_main" },
        );
        unsafe { ctx.device().destroy_shader_module(present_shader, None) };
        let ui = crate::ui::render::UiRenderer::new_with_blend(
            &ctx,
            display_fmt,
            crate::ui::atlas::AtlasConfig::default().page_size,
            ui_blend,
        );

        let gpu_pool = unsafe {
            ctx.device()
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2 * max_frames as u32),
                    None,
                )
                .expect("renderer timestamp pool")
        };
        let renderer = Self {
            ctx,
            swapchain,
            scene_pipeline,
            scene_layout,
            scene_dsl,
            pulling,
            blit_pipeline,
            blit_layout,
            blit_dsl,
            descriptor_pool,
            scene_sets,
            blit_set,
            sampler,
            scene_uniforms,
            frame_pool,
            cmd,
            image_available,
            render_finished,
            in_flight,
            frame_serials: vec![0; max_frames],
            frame: 0,
            frame_id: 0,
            gpu_pool,
            scene_ms: vec![0.0; max_frames],
            scene_sample: None,
            scene_recorded: vec![false; max_frames],
            targets,
            targets_ready: false,
            thumb,
            fsr,
            fsr_enabled: temporal,
            display,
            present_pipeline,
            present_layout,
            present_set,
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
            acquire: super::acquire::AcquireWorker::new(),
            frame_wait_budget: None,
        };
        renderer.write_sets();
        renderer
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        // Quiesce the acquire-ahead worker before the swapchain goes away: a
        // request in flight would otherwise touch a destroyed swapchain, and
        // a token it already delivered carries a dead image index. Its slot's
        // semaphore is signalled (the image was acquired), so wait it out.
        if let Some(Ok(token)) = self.acquire.shutdown() {
            self.reclaim_slot(token.slot);
        }
        unsafe { self.ctx.device().device_wait_idle() }.expect("wait idle on resize");

        // Hand the retired swapchain back so the driver can reuse the window.
        let old = self.swapchain.swapchain;
        self.swapchain = vulkan::Swapchain::new(&self.ctx, size.width, size.height, old);
        // The surface may clamp or refuse the requested extent — every
        // downstream size derives from the extent actually chosen.
        let actual = self.swapchain.extent;
        self.display_size = (actual.width, actual.height);
        self.render_size = scaled_size(self.display_size, self.render_scale);
        if self.targets_ready {
            self.targets = build_targets(&self.ctx, self.render_size);
        }
        self.display = build_display(
            &self.ctx,
            self.display_size,
            display_format(self.swapchain.format, self.ui_blend),
        );
        if let Some(fsr) = self.fsr.as_mut() {
            fsr.resize(self.render_size, self.display_size);
            fsr.set_inputs([
                self.targets.color.view,
                self.targets.velocity.view,
                self.targets.depth.view,
            ]);
        }
        self.history_valid = false;
        self.jitter.reset();
        self.write_sets();
        self.acquire.respawn();
    }

    /// Wall-clock bound on the frame-start wait for the frame's swapchain
    /// image (slot fence + acquire, resolved on the acquire-ahead worker).
    /// `None` -- the default -- waits as long as the presentation engine
    /// needs, up to the engine's liveness cap (`acquire::UNBUDGETED_WAIT_CAP`,
    /// 250 ms): scripted capture runs render every requested frame without
    /// ever freezing behind a dead presentation engine. Interactive apps pass
    /// a frame-scale budget so a stall skips the frame
    /// (`SurfaceError::Timeout`) instead of freezing the app loop while the
    /// simulation falls behind.
    pub fn set_frame_wait_budget(&mut self, budget: Option<Duration>) {
        self.frame_wait_budget = budget;
    }

    /// The current [`Self::set_frame_wait_budget`] value.
    pub fn frame_wait_budget(&self) -> Option<Duration> {
        self.frame_wait_budget
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Show or hide the centred crosshair drawn by the present pass. Off by
    /// default; apps that want it (a game's aim reticle) turn it on.
    pub fn set_show_crosshair(&mut self, show: bool) {
        self.show_crosshair = show;
    }

    /// BedFrame's private instanced-quad path (default on). The legacy
    /// indexed path renders identically — a pure A/B switch.
    pub fn set_ui_instancing(&mut self, enabled: bool) {
        self.ui.set_instancing(enabled);
    }

    /// UI-owned GPU resource sizes, raw per resource
    /// (`(name, allocation bytes)`) — the bench-side accounting reads
    /// these directly; nothing here derives a total.
    pub fn ui_allocation_sizes(&self) -> Vec<(String, u64)> {
        self.ui.allocation_sizes()
    }

    /// Bytes uploaded into the UI slot's geometry buffers by the last
    /// record: `(legacy_vertex_bytes, legacy_index_bytes, instance_bytes)`.
    pub fn ui_last_geometry_upload_sizes(&self) -> (u64, u64, u64) {
        self.ui.last_geometry_upload_sizes()
    }

    /// Primitive counts of the last UI record: `(instances, raw_tris,
    /// draws)`.
    pub fn ui_last_prepared_counts(&self) -> (u64, u64, u64) {
        self.ui.last_prepared_counts()
    }

    /// Enable or disable the temporal upscaler. Without a renderer that was
    /// built for it, enabling does nothing; see [`Self::toggle_temporal`].
    pub fn set_temporal(&mut self, enabled: bool) {
        if self.temporal != enabled {
            self.toggle_temporal();
        }
    }

    /// Turn the temporal upscaler on or off at runtime. Off blits the scene
    /// colour straight to the swapchain, which is the reference image scripted
    /// capture runs compare against.
    pub fn toggle_temporal(&mut self) {
        if !self.fsr_enabled && self.fsr.is_none() {
            eprintln!("FSR 3.1: unavailable, `aa = \"none\"` in settings");
            return;
        }
        self.temporal = !self.temporal;
        self.history_valid = false;
        if self.temporal {
            self.render_scale = 1.0;
        }
        self.write_sets();
        eprintln!(
            "FSR 3.1: {}",
            if self.temporal {
                "on"
            } else if self.render_size == self.display_size {
                "off (direct blit)"
            } else {
                "off (direct blit, upscaled from a smaller render target)"
            }
        );
    }

    fn write_sets(&self) {
        for (set, uniform) in self.scene_sets.iter().zip(self.scene_uniforms.iter()) {
            let scene_buffer = vk::DescriptorBufferInfo::default()
                .buffer(uniform.buffer)
                .range(std::mem::size_of::<ViewUniform>() as u64);
            write_sets(&self.ctx, &[uniform_write(*set, 0, &scene_buffer)]);
        }

        let source = match self
            .fsr
            .as_ref()
            .filter(|_| self.temporal && self.targets_ready)
        {
            Some(fsr) => fsr.output_view(),
            None => self.targets.color.view,
        };
        let image = sampled(source);
        let sampler = sampler_info(self.sampler);
        let writes = [
            image_write(self.blit_set, 0, &image),
            sampler_write(self.blit_set, 1, &sampler),
        ];
        write_sets(&self.ctx, &writes);

        let display = sampled(self.display.view);
        let sampler = sampler_info(self.sampler);
        let writes = [
            image_write(self.present_set, 0, &display),
            sampler_write(self.present_set, 1, &sampler),
        ];
        write_sets(&self.ctx, &writes);

        if let Some(thumb) = self.thumb.as_ref() {
            thumb.bind_color(&self.ctx, self.targets.color.view);
        }
    }

    /// Invalidate the temporal history (teleport, cut, level change). The
    /// next frame renders as a restart and clears the upscaler's persistent
    /// accumulation state, so no stale image smears across the discontinuity.
    pub fn reset_history(&mut self) {
        self.history_valid = false;
        self.jitter.reset();
    }

    /// Whether the temporal upscaler exists (built with `aa = "fsr"`).
    pub fn has_temporal(&self) -> bool {
        self.fsr_enabled || self.fsr.is_some()
    }

    pub(super) fn ensure_scene_targets(&mut self) {
        if self.targets_ready {
            return;
        }
        self.targets_ready = true;
        self.targets = build_targets(&self.ctx, self.render_size);
        if self.fsr_enabled && self.fsr.is_none() {
            let mut fsr = Fsr3::new(
                &self.ctx,
                self.render_size,
                self.display_size,
                [
                    self.targets.color.view,
                    self.targets.velocity.view,
                    self.targets.depth.view,
                ],
            );
            fsr.write_rcas_config(self.sharpness);
            self.fsr = Some(fsr);
        }
        self.write_sets();
    }

    /// Drop accumulated temporal history (scene/object changes between
    /// captures) without toggling the pipeline.
    pub fn reset_temporal_history(&mut self) {
        self.history_valid = false;
    }

    /// Whether the temporal history is currently usable (false after a
    /// restart/reset until the next frame completes).
    pub fn history_valid(&self) -> bool {
        self.history_valid
    }

    pub fn display_size(&self) -> (u32, u32) {
        self.display_size
    }

    pub fn render_size(&self) -> (u32, u32) {
        self.render_size
    }

    /// Whether the temporal upscaler is currently running.
    pub fn temporal(&self) -> bool {
        self.temporal
    }

    pub(super) fn write_uniforms(&mut self, camera: &Camera, jp: [f32; 2]) {
        let rw = self.render_size.0 as f32;
        let rh = self.render_size.1 as f32;
        let jittered = camera.jittered_view_projection(jp, [rw, rh]);
        let curr_vp = camera.view_projection();
        // Without valid history, motion is meaningless: report zero motion.
        let prev_vp = if self.history_valid {
            self.prev_view_projection
        } else {
            curr_vp
        };
        self.scene_uniforms[self.frame].write(
            0,
            bytemuck::bytes_of(&ViewUniform {
                view_projection: jittered.to_cols_array_2d(),
                curr_view_projection: curr_vp.to_cols_array_2d(),
                prev_view_projection: prev_vp.to_cols_array_2d(),
            }),
        );
        let _ = self.sharpness;
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // The worker may be inside a swapchain call; join it before the
        // device and the swapchain go away.
        self.acquire.shutdown();
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            let device = self.ctx.device();
            device.destroy_pipeline(self.scene_pipeline, None);
            device.destroy_pipeline(self.blit_pipeline, None);
            device.destroy_pipeline(self.present_pipeline, None);
            if let Some(thumb) = self.thumb.as_ref() {
                thumb.destroy(device);
            }
            device.destroy_pipeline_layout(self.scene_layout, None);
            device.destroy_pipeline_layout(self.blit_layout, None);
            device.destroy_pipeline_layout(self.present_layout, None);
            if let Some((pipeline, layout, dsl)) = self.pulling {
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(dsl, None);
            }
            device.destroy_descriptor_set_layout(self.scene_dsl, None);
            device.destroy_query_pool(self.gpu_pool, None);
            device.destroy_descriptor_set_layout(self.blit_dsl, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_command_pool(self.frame_pool, None);
            for &semaphore in &self.image_available {
                device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished {
                device.destroy_semaphore(semaphore, None);
            }
            for &fence in &self.in_flight {
                device.destroy_fence(fence, None);
            }
        }
    }
}
