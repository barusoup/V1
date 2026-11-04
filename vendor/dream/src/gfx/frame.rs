use super::acquire::AcquireOutcome;
use super::pipeline::{
    attachment, depth_attachment, general_barrier, make_viewport, map_surface_error, scene_region,
};
use super::timing::elapsed_timestamp_ticks;
use super::vulkan;
use super::{DrawCtx, FramePhases, Renderer, SurfaceError};
use crate::camera::Camera;
use ash::vk;
use std::sync::Arc;
use std::time::Instant;

impl Renderer {
    /// Render one frame with no UI overlay.
    pub fn render<F>(&mut self, camera: &Camera, draw: F) -> Result<usize, SurfaceError>
    where
        F: FnOnce(&mut DrawCtx<'_>) -> usize,
    {
        self.render_inner(camera, None, draw, None, true)
    }

    /// Render one frame with an optional [`crate::ui::draw::UiFrame`] drawn
    /// over the composed image. The overlay renders at display resolution
    /// *after* upscaling, so text stays crisp at every `upscale` preset and
    /// is immune to temporal reprojection.
    pub fn render_ui<F>(
        &mut self,
        camera: &Camera,
        scene_rect: Option<crate::ui::Rect>,
        draw: F,
        ui: Option<&crate::ui::draw::UiFrame>,
    ) -> Result<usize, SurfaceError>
    where
        F: FnOnce(&mut DrawCtx<'_>) -> usize,
    {
        self.render_inner(camera, scene_rect, draw, ui, true)
    }

    pub fn render_ui_only(
        &mut self,
        ui: Option<&crate::ui::draw::UiFrame>,
    ) -> Result<(), SurfaceError> {
        self.render_inner(&Camera::default(), None, |_| 0, ui, false)
            .map(|_| ())
    }

    fn render_inner<F>(
        &mut self,
        camera: &Camera,
        scene_rect: Option<crate::ui::Rect>,
        draw: F,
        ui: Option<&crate::ui::draw::UiFrame>,
        scene_enabled: bool,
    ) -> Result<usize, SurfaceError>
    where
        F: FnOnce(&mut DrawCtx<'_>) -> usize,
    {
        self.scene_sample = None;
        self.phases = FramePhases::default();
        // Frame start. The acquire-ahead worker already waited this slot's
        // fence and reserved its swapchain image, so a ready token means the
        // slot is free and the frame renders without blocking. Within the
        // wait budget a not-yet-ready token skips the frame
        // (`SurfaceError::Timeout`) instead of freezing the app loop behind a
        // stalled presentation engine; the request stays outstanding and a
        // later frame picks the image up.
        if !self.acquire.has_request() {
            self.request_acquire();
        }
        let wait_start = Instant::now();
        let token = match self.acquire.wait(self.frame_wait_budget) {
            AcquireOutcome::Ready(token) => token,
            AcquireOutcome::Pending => {
                self.phases.token_wait_us = wait_start.elapsed().as_secs_f32() * 1e6;
                return Err(SurfaceError::Timeout);
            }
            AcquireOutcome::Failed(err) => return Err(err),
            AcquireOutcome::Lost => return Err(SurfaceError::Other),
        };
        self.phases.fence_wait_us = token.fence_wait_us;
        self.phases.acquire_us = token.acquire_us;
        self.phases.token_wait_us = wait_start.elapsed().as_secs_f32() * 1e6;
        let device = self.ctx.device().clone();
        let frame = token.slot;
        let image_index = token.image_index;
        let suboptimal = token.suboptimal;
        // This slot's fence signalled its serial; the in-order queue makes
        // every earlier submission complete as well.
        self.ctx.confirm_completed(self.frame_serials[frame]);
        self.finish_scene_thumbnail(frame);
        if self.frame_serials[frame] > 0
            && self.scene_recorded[frame]
            && self.ctx.timestamp_valid_bits() > 0
        {
            let mut ticks = [0u64; 2];
            let got = unsafe {
                device.get_query_pool_results(
                    self.gpu_pool,
                    frame as u32 * 2,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if got.is_ok()
                && let Some(elapsed) =
                    elapsed_timestamp_ticks(ticks[0], ticks[1], self.ctx.timestamp_valid_bits())
            {
                let ms = elapsed as f32 * self.ctx.timestamp_period() * 1e-6;
                if ms.is_finite() && ms > 0.0 {
                    self.scene_ms[frame] = ms;
                    self.scene_sample = Some((self.frame_serials[frame], ms));
                }
            }
        } else if self.frame_serials[frame] > 0 {
            self.scene_ms[frame] = 0.0;
        }

        unsafe {
            device
                .reset_fences(&[self.in_flight[frame]])
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
            device
                .reset_command_buffer(self.cmd[frame], vk::CommandBufferResetFlags::empty())
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
        }

        let cmd = self.cmd[frame];
        let render_size = self.render_size;
        let display_size = self.display_size;
        let swapchain_extent = (self.swapchain.extent.width, self.swapchain.extent.height);
        let jp = if scene_enabled && self.temporal {
            self.jitter.next(render_size.0, display_size.0)
        } else {
            [0.0, 0.0]
        };
        let curr_view_projection = camera.view_projection();

        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();
        let reset = !self.history_valid;
        if scene_enabled {
            self.write_uniforms(camera, jp);
            if self.temporal
                && let Some(fsr) = self.fsr.as_mut()
            {
                fsr.update(camera, jp, dt, reset, frame);
            }
        }

        let mark = Instant::now();
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
        }

        // The scene owns one region of the render target - the app's viewport.
        // Outside it are its panels, which are drawn later at display
        // resolution: shading those pixels would be work nobody sees. The
        // render area stays the whole target so every pixel is still cleared;
        // only the scissor is narrowed.
        let (scene_viewport, scene_scissor) = scene_region(scene_rect, render_size, display_size);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: render_size.0,
                height: render_size.1,
            },
        };
        let swap_viewport = make_viewport(swapchain_extent);
        // The display pass owns its own extent — it can differ from the
        // swapchain's when a resize request wasn't fully honored.
        let display_viewport = make_viewport(display_size);
        let swap_scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: swapchain_extent.0,
                height: swapchain_extent.1,
            },
        };

        // Swapchain image: undefined -> color attachment (the blit overwrites
        // every pixel, so the previous contents do not matter).
        let swapchain_image = self.swapchain.images[image_index as usize];
        let swapchain_view = self.swapchain.views[image_index as usize];
        unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                swapchain_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::ImageAspectFlags::COLOR,
            );
        }

        // --- Scene pass ---
        if self.ctx.timestamp_valid_bits() > 0 {
            unsafe {
                device.cmd_reset_query_pool(cmd, self.gpu_pool, frame as u32 * 2, 2);
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 2,
                );
            }
        }
        let scene_pipeline = self.scene_pipeline;
        let scene_layout = self.scene_layout;
        let scene_set = self.scene_sets[frame];
        let pulling = self.pulling.map(|(pipeline, layout, _)| (pipeline, layout));
        let frame_id = self.frame_id;
        let already_ready = self.targets_ready;
        let mut scene_pass_begun = false;

        let begin_scene = |this: &mut Self, begun: &mut bool| {
            if *begun {
                return;
            }
            *begun = true;
            if !this.targets_ready {
                this.ensure_scene_targets();
                if this.temporal
                    && let Some(fsr) = this.fsr.as_mut()
                {
                    fsr.update(camera, jp, dt, reset, frame);
                }
            }
            let color = attachment(
                this.targets.color.view,
                vk::AttachmentLoadOp::CLEAR,
                [0.48, 0.69, 0.89, 1.0],
            );
            let velocity = attachment(
                this.targets.velocity.view,
                vk::AttachmentLoadOp::CLEAR,
                [0.0, 0.0, 0.0, 0.0],
            );
            let colors = [color, velocity];
            let depth = depth_attachment(this.targets.depth.view, 1.0);
            let info = vk::RenderingInfo::default()
                .render_area(scissor)
                .layer_count(1)
                .color_attachments(&colors)
                .depth_attachment(&depth);
            unsafe {
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[scene_viewport]);
                device.cmd_set_scissor(cmd, 0, &[scene_scissor]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, scene_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    scene_layout,
                    0,
                    &[scene_set],
                    &[],
                );
            }
        };

        let visible = if scene_enabled {
            if already_ready {
                begin_scene(self, &mut scene_pass_begun);
            }
            let mut starter = || {
                begin_scene(self, &mut scene_pass_begun);
            };
            let mut ctx = DrawCtx {
                device: &device,
                cmd,
                scene_pipeline,
                scene_layout,
                scene_set,
                pulling,
                frame_slot: frame,
                frame_id,
                starter: std::cell::RefCell::new(if already_ready {
                    None
                } else {
                    Some(&mut starter)
                }),
            };
            let visible = draw(&mut ctx);
            let _ = ctx.starter.borrow_mut().take();
            if scene_pass_begun {
                unsafe { device.cmd_end_rendering(cmd) };
            }
            visible
        } else {
            0
        };
        if self.ctx.timestamp_valid_bits() > 0 {
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.gpu_pool,
                    frame as u32 * 2 + 1,
                );
            }
        }

        if scene_pass_begun {
            // Scene colour, depth and motion vectors become readable.
            unsafe {
                let barriers = [
                    general_barrier(
                        self.targets.color.image,
                        vk::ImageAspectFlags::COLOR,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    ),
                    general_barrier(
                        self.targets.depth.image,
                        vk::ImageAspectFlags::DEPTH,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    ),
                    general_barrier(
                        self.targets.velocity.image,
                        vk::ImageAspectFlags::COLOR,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    ),
                ];
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }

            self.record_scene_thumbnail(cmd, frame);

            // --- FSR 3.1 temporal upscale ---
            if self.temporal
                && let Some(fsr) = self.fsr.as_mut()
            {
                fsr.record(cmd, !reset, frame);
            }
        }

        if scene_enabled {
            unsafe {
                vulkan::image_barrier(
                    &device,
                    cmd,
                    self.blit_source_image(),
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::ImageAspectFlags::COLOR,
                );
            }
        }

        // --- Blit pass (FSR output -> composed display image) ---
        {
            let load = if scene_enabled {
                vk::AttachmentLoadOp::DONT_CARE
            } else {
                vk::AttachmentLoadOp::CLEAR
            };
            let color = attachment(self.display.view, load, [0.0, 0.0, 0.0, 1.0]);
            let display_scissor = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: display_size.0,
                    height: display_size.1,
                },
            };
            let info = vk::RenderingInfo::default()
                .render_area(display_scissor)
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color));
            unsafe {
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[display_viewport]);
                device.cmd_set_scissor(cmd, 0, &[display_scissor]);
                if scene_enabled {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.blit_pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.blit_layout,
                        0,
                        &[self.blit_set],
                        &[],
                    );
                    device.cmd_draw(cmd, 3, 1, 0, 0);
                }
                device.cmd_end_rendering(cmd);
            }
        }

        // --- UI overlay (display resolution, over the composed image) ---
        if let Some(ui) = ui {
            self.ui
                .record(cmd, frame, self.display.image, self.display.view, display_size, ui);
        }

        // Composed display image becomes readable for the present pass.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[general_barrier(
                    self.display.image,
                    vk::ImageAspectFlags::COLOR,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::AccessFlags::SHADER_READ,
                )],
            );
        }

        // --- Present pass (display image -> swapchain) ---
        {
            let color = attachment(
                swapchain_view,
                vk::AttachmentLoadOp::DONT_CARE,
                [0.0, 0.0, 0.0, 1.0],
            );
            let info = vk::RenderingInfo::default()
                .render_area(swap_scissor)
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color));
            unsafe {
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[swap_viewport]);
                device.cmd_set_scissor(cmd, 0, &[swap_scissor]);
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.present_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.present_layout,
                    0,
                    &[self.present_set],
                    &[],
                );
                let crosshair = if self.show_crosshair { 1.0 } else { 0.0 };
                let push = [display_size.0 as f32, display_size.1 as f32, crosshair];
                device.cmd_push_constants(
                    cmd,
                    self.present_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_draw(cmd, 3, 1, 0, 0);
                device.cmd_end_rendering(cmd);
            }
        }

        unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                swapchain_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::empty(),
                vk::ImageAspectFlags::COLOR,
            );
            device
                .end_command_buffer(cmd)
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
            self.phases.record_us = mark.elapsed().as_secs_f32() * 1e6;

            let wait_semaphores = [self.image_available[frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished[frame]];
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signal_semaphores);
            // A failed submit gets no serial; the frame's serial slot only
            // records submissions that actually entered the queue.
            let mark = Instant::now();
            let serial = self
                .ctx
                .try_submit_tracked(&submit, self.in_flight[frame])
                .map_err(|_| self.frame_abort(frame, SurfaceError::Other))?;
            self.phases.submit_us = mark.elapsed().as_secs_f32() * 1e6;
            self.frame_serials[frame] = serial;
            self.scene_recorded[frame] = scene_pass_begun;
            // The submitted frame owns this id now; a failed present below
            // must not reuse it for slot bookkeeping.
            self.frame_id += 1;
            self.frame = (frame + 1) % self.ctx.max_frames_in_flight();

            let swapchains = [self.swapchain.swapchain];
            let indices = [image_index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&indices);
            // Present on the same queue must be serialized with submits so
            // the serial tracker mirrors real queue order.
            {
                let mark = Instant::now();
                let _queue = self.ctx.queue_lock.lock().unwrap();
                // Present reports the same resize/loss conditions as
                // acquire: `OUT_OF_DATE` means the surface moved on while
                // this frame was recorded, so callers must be able to resize
                // and retry rather than see an undiagnosable `Other`.
                self.ctx
                    .swapchain_loader
                    .queue_present(self.ctx.queue(), &present)
                    .map_err(map_surface_error)?;
                self.phases.present_us = mark.elapsed().as_secs_f32() * 1e6;
            }
            // Request the next acquire only AFTER presenting. The image this
            // frame just presented is what frees the swapchain image the next
            // acquire needs: requesting earlier over-commits the pool (this
            // frame still holds its recording image, the presentation engine
            // holds the queued ones) and the acquire then throttles to the
            // compositor's latch cadence, dragging the whole loop with it.
            // The request still resolves on the worker while the app runs its
            // next update.
            self.request_acquire();
        }

        if scene_enabled {
            self.prev_view_projection = curr_view_projection;
            self.history_valid = true;
        } else {
            self.history_valid = false;
        }

        if suboptimal {
            Err(SurfaceError::Outdated)
        } else {
            Ok(visible)
        }
    }

    /// Segment timings of the most recent `render*` call, in microseconds --
    /// including a skipped one, whose phases show where its wait budget went.
    /// See [`FramePhases`] for how to read them.
    pub fn frame_phases(&self) -> FramePhases {
        self.phases
    }

    /// Post the acquire-ahead request for the next frame slot. Only called
    /// with no request outstanding; the worker gates the acquire on the
    /// slot's fence, so a delivered token proves the slot is free.
    fn request_acquire(&mut self) {
        let frame = self.frame;
        self.acquire.request(
            Arc::clone(&self.ctx),
            self.swapchain.swapchain,
            frame,
            self.in_flight[frame],
            self.image_available[frame],
        );
    }

    /// A frame that aborts before its submission entered the queue has
    /// already acquired its swapchain image: `image_available[frame]` is
    /// signalled but unwaited, and a binary semaphore must be waited before
    /// it can be signalled again. [`Self::reclaim_slot`] waits it out.
    fn frame_abort(&mut self, frame: usize, err: SurfaceError) -> SurfaceError {
        self.reclaim_slot(frame);
        err
    }

    /// Return an aborted frame's slot to a reusable state with an empty batch
    /// that waits `image_available[frame]` out. The batch carries the slot's
    /// fence -- reset first: the aborted frame may have reset it without
    /// submitting -- so the acquire-ahead worker's fence gate covers this
    /// wait before it can signal the semaphore again. Safe to call for any
    /// slot not currently in flight; a failed submit means none is.
    pub(super) fn reclaim_slot(&mut self, frame: usize) {
        let device = self.ctx.device().clone();
        if unsafe { device.reset_fences(&[self.in_flight[frame]]) }.is_err() {
            return; // device lost; nothing here can be salvaged
        }
        let wait = [self.image_available[frame]];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages);
        if let Ok(serial) = self.ctx.try_submit_tracked(&submit, self.in_flight[frame]) {
            self.frame_serials[frame] = serial;
            self.scene_recorded[frame] = false;
        }
    }

    fn blit_source_image(&self) -> vk::Image {
        match self
            .fsr
            .as_ref()
            .filter(|_| self.temporal && self.targets_ready)
        {
            Some(fsr) => fsr.output_image(),
            None => self.targets.color.image,
        }
    }
}
