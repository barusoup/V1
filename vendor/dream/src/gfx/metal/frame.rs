use super::metal::shader::PUSH_CONSTANT_BUFFER_SLOT;
use super::pipeline::{make_viewport, scene_region};
use super::{DrawCtx, FramePhases, Renderer, SurfaceError, ViewUniform};
use crate::camera::Camera;
use block2::RcBlock;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLLoadAction,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLStoreAction,
};
use objc2_quartz_core::CAMetalDrawable;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

impl Renderer {
    pub fn render<F>(&mut self, camera: &Camera, draw: F) -> Result<usize, SurfaceError>
    where
        F: FnOnce(&mut DrawCtx<'_>) -> usize,
    {
        self.render_inner(camera, None, draw, None, true)
    }

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
        objc2::rc::autoreleasepool(|_| {
            self.render_inner_pooled(camera, scene_rect, draw, ui, scene_enabled)
        })
    }

    fn render_inner_pooled<F>(
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
        if self.needs_display() && !self.display_live {
            self.ensure_display();
        }

        let frame = self.frame;
        let wait_start = Instant::now();
        let (prev_serial, prev_gpu_ms) = match self.slots[frame].wait(self.frame_wait_budget) {
            Ok(res) => res,
            Err(()) => {
                self.phases.token_wait_us = wait_start.elapsed().as_secs_f32() * 1e6;
                return Err(SurfaceError::Timeout);
            }
        };
        self.phases.fence_wait_us = wait_start.elapsed().as_secs_f32() * 1e6;
        self.phases.token_wait_us = self.phases.fence_wait_us;

        if prev_serial > 0 {
            self.ctx.confirm_completed(prev_serial);
            self.ctx.collect_retired_buffers();
            self.finish_scene_thumbnail(frame);
            if self.scene_recorded[frame] && prev_gpu_ms > 0.0 {
                self.scene_ms[frame] = prev_gpu_ms;
                self.scene_sample = Some((prev_serial, prev_gpu_ms));
            } else {
                self.scene_ms[frame] = 0.0;
            }
        }

        let acq_start = Instant::now();
        let Some(drawable) = self.swapchain.next_drawable(&self.ctx) else {
            self.phases.acquire_us = acq_start.elapsed().as_secs_f32() * 1e6;
            return Err(SurfaceError::Timeout);
        };
        self.phases.acquire_us = acq_start.elapsed().as_secs_f32() * 1e6;

        let record_start = Instant::now();
        let render_size = self.render_size;
        let display_size = self.display_size;
        let swapchain_extent = (self.swapchain.width, self.swapchain.height);

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
            self.ensure_scene_targets();
            let view_projection = camera.jittered_view_projection(
                jp,
                [self.render_size.0 as f32, self.render_size.1 as f32],
            );
            let prev_view_projection = if self.history_valid {
                self.prev_view_projection
            } else {
                curr_view_projection
            };
            let uniform = ViewUniform {
                view_projection: view_projection.to_cols_array_2d(),
                curr_view_projection: curr_view_projection.to_cols_array_2d(),
                prev_view_projection: prev_view_projection.to_cols_array_2d(),
            };
            self.scene_uniforms[frame].write_bytes(0, bytemuck::bytes_of(&uniform));
        }

        let cmd = self
            .ctx
            .queue()
            .commandBuffer()
            .expect("create Metal command buffer");

        let direct = !self.needs_display();
        let drawable_tex = drawable.texture();
        let target_size: (u32, u32) = if direct {
            swapchain_extent
        } else {
            display_size
        };

        let mut drawn = 0usize;
        if scene_enabled {
            let (scene_viewport, scene_scissor) =
                scene_region(scene_rect, render_size, display_size);
            let rpd = MTLRenderPassDescriptor::new();
            let ca = rpd.colorAttachments();
            let c0 = unsafe { ca.objectAtIndexedSubscript(0) };
            c0.setTexture(Some(&self.targets.color.texture));
            c0.setLoadAction(MTLLoadAction::Clear);
            c0.setStoreAction(MTLStoreAction::Store);
            c0.setClearColor(MTLClearColor {
                red: 0.48,
                green: 0.69,
                blue: 0.89,
                alpha: 1.0,
            });

            let c1 = unsafe { ca.objectAtIndexedSubscript(1) };
            c1.setTexture(Some(&self.targets.velocity.texture));
            c1.setLoadAction(MTLLoadAction::Clear);
            c1.setStoreAction(MTLStoreAction::Store);
            c1.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });

            let da = rpd.depthAttachment();
            da.setTexture(Some(&self.targets.depth.texture));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setStoreAction(MTLStoreAction::Store);
            da.setClearDepth(1.0);

            let enc = cmd
                .renderCommandEncoderWithDescriptor(&rpd)
                .expect("scene render encoder");
            enc.setViewport(scene_viewport);
            enc.setScissorRect(scene_scissor);

            let mut noop_starter = || {};
            let mut draw_ctx = DrawCtx {
                encoder: &enc,
                scene_pipeline: &self.scene_pipeline,
                scene_depth_stencil: &self.scene_depth_stencil,
                scene_uniform: &self.scene_uniforms[frame].buffer,
                pulling: self.pulling.as_deref(),
                index_buffer: None,
                frame_slot: frame,
                frame_id: self.frame_id,
                starter: std::cell::RefCell::new(Some(&mut noop_starter)),
            };
            drawn = draw(&mut draw_ctx);
            enc.endEncoding();

            // FSR 3.1 temporal upscale
            if self.fsr_enabled
                && let Some(fsr) = self.fsr.as_mut()
            {
                let f = fsr;
                f.update_constants(
                    camera,
                    render_size,
                    display_size,
                    jp,
                    dt,
                    self.sharpness,
                    reset,
                    frame,
                );
                f.record(&cmd, self.history_valid, frame);
            }

            self.record_scene_thumbnail(&cmd, frame);
            let scene_target_tex: &ProtocolObject<dyn objc2_metal::MTLTexture> = if direct {
                &drawable_tex
            } else {
                &self.display.texture
            };

            // Blit scene / upscaled output into target_tex
            let blit_src = if self.fsr_enabled
                && let Some(fsr) = self.fsr.as_ref()
            {
                fsr.output_texture()
            } else {
                &self.targets.color.texture
            };

            let blit_rpd = MTLRenderPassDescriptor::new();
            let b0 = unsafe { blit_rpd.colorAttachments().objectAtIndexedSubscript(0) };
            b0.setTexture(Some(scene_target_tex));
            b0.setLoadAction(MTLLoadAction::DontCare);
            b0.setStoreAction(MTLStoreAction::Store);

            let blit_viewport = make_viewport(target_size);
            let blit_scissor = objc2_metal::MTLScissorRect {
                x: 0,
                y: 0,
                width: target_size.0.max(1) as usize,
                height: target_size.1.max(1) as usize,
            };
            let blit_enc = cmd
                .renderCommandEncoderWithDescriptor(&blit_rpd)
                .expect("blit render encoder");
            blit_enc.setViewport(blit_viewport);
            blit_enc.setScissorRect(blit_scissor);
            blit_enc.setRenderPipelineState(&self.blit_pipeline);
            blit_enc.setDepthStencilState(Some(&self.no_depth_stencil));
            unsafe {
                blit_enc.setFragmentTexture_atIndex(Some(blit_src), 0);
                blit_enc.setFragmentSamplerState_atIndex(Some(self.ctx.default_sampler()), 0);
                blit_enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            blit_enc.endEncoding();
        } else {
            let target_tex: &ProtocolObject<dyn objc2_metal::MTLTexture> = if direct {
                &drawable_tex
            } else {
                &self.display.texture
            };
            // UI-only: clear target_tex if no UI frame will clear it.
            if ui.is_none() {
                let clear_rpd = MTLRenderPassDescriptor::new();
                let c0 = unsafe { clear_rpd.colorAttachments().objectAtIndexedSubscript(0) };
                c0.setTexture(Some(target_tex));
                c0.setLoadAction(MTLLoadAction::Clear);
                c0.setStoreAction(MTLStoreAction::Store);
                c0.setClearColor(MTLClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                });
                let enc = cmd
                    .renderCommandEncoderWithDescriptor(&clear_rpd)
                    .expect("clear encoder");
                enc.endEncoding();
            }
        }

        let target_tex: &ProtocolObject<dyn objc2_metal::MTLTexture> = if direct {
            &drawable_tex
        } else {
            &self.display.texture
        };

        // UI overlay pass
        if let Some(ui_frame) = ui {
            self.ui.record(
                &cmd,
                target_tex,
                target_size,
                ui_frame,
                frame,
                !scene_enabled,
            );
        }

        // Present pass (only when indirect composition is active)
        if !direct {
            let pres_rpd = MTLRenderPassDescriptor::new();
            let p0 = unsafe { pres_rpd.colorAttachments().objectAtIndexedSubscript(0) };
            p0.setTexture(Some(&drawable_tex));
            p0.setLoadAction(MTLLoadAction::DontCare);
            p0.setStoreAction(MTLStoreAction::Store);

            let pres_enc = cmd
                .renderCommandEncoderWithDescriptor(&pres_rpd)
                .expect("present render encoder");
            pres_enc.setViewport(make_viewport(swapchain_extent));
            pres_enc.setRenderPipelineState(&self.present_pipeline);
            pres_enc.setDepthStencilState(Some(&self.no_depth_stencil));
            let push = [
                swapchain_extent.0 as f32,
                swapchain_extent.1 as f32,
                if self.show_crosshair { 1.0 } else { 0.0 },
                0.0f32,
            ];
            unsafe {
                pres_enc.setFragmentTexture_atIndex(Some(&self.display.texture), 0);
                pres_enc.setFragmentSamplerState_atIndex(Some(self.ctx.default_sampler()), 0);
                let bytes = bytemuck::bytes_of(&push);
                pres_enc.setFragmentBytes_length_atIndex(
                    NonNull::new_unchecked(bytes.as_ptr() as *mut _),
                    bytes.len(),
                    PUSH_CONSTANT_BUFFER_SLOT as usize,
                );
                pres_enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            pres_enc.endEncoding();
            self.display_written = true;
        }

        self.phases.record_us = record_start.elapsed().as_secs_f32() * 1e6;

        let submit_start = Instant::now();
        let serial = self.ctx.next_serial();
        let slot_sync = Arc::clone(&self.slots[frame]);
        slot_sync.mark_pending(serial);
        let handler = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let cb_ref = unsafe { cb.as_ref() };
            let start = cb_ref.GPUStartTime();
            let end = cb_ref.GPUEndTime();
            let gpu_ms = if end > start {
                ((end - start) * 1000.0) as f32
            } else {
                0.0
            };
            slot_sync.mark_done(gpu_ms);
        });
        unsafe {
            cmd.addCompletedHandler(RcBlock::as_ptr(&handler));
        }

        let drawable_obj: &ProtocolObject<dyn objc2_metal::MTLDrawable> =
            ProtocolObject::from_ref(&*drawable);
        cmd.presentDrawable(drawable_obj);
        cmd.commit();
        self.phases.submit_us = submit_start.elapsed().as_secs_f32() * 1e6;

        self.scene_recorded[frame] = scene_enabled;
        if scene_enabled {
            self.prev_view_projection = curr_view_projection;
            self.history_valid = true;
        } else {
            self.history_valid = false;
        }
        self.frame_id += 1;
        self.frame = (frame + 1) % self.slots.len();

        Ok(drawn)
    }

    pub fn frame_phases(&self) -> FramePhases {
        self.phases
    }
}
