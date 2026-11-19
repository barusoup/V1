//! GPU layer for UI frames.
//!
//! [`UiRenderer`] owns the pipeline, per-frame upload rings and atlas page
//! textures, and records a [`UiFrame`] into a command buffer as one pass of
//! alpha-blended, scissored draws. The renderer calls it inside the frame
//! chain (see `Renderer::render_ui`); games normally never touch this module
//! directly, but everything here is public for custom composite passes.
//!
//! Frame-in-flight discipline: vertex/index uploads go to a host buffer per
//! frame slot and are only written after that slot's fence recycled, matching
//! the engine's scene-uniform convention.

mod backdrop;
mod prepare;

use super::draw::{DrawCmd, UiFrame, UiVertex};
use crate::gfx::vulkan::{self, GpuContext};
use crate::texture::{self, GpuTexture, SamplerDesc, TextureFormat};
use ash::vk;
use prepare::{DrawKind, Prepared, UiInstance};
use std::sync::Arc;

pub use super::draw::AtlasUpdate;

/// Capacity policy for a repeatedly reused per-frame buffer.
///
/// Growth is immediate: a frame's geometry must always fit. Shrinking is
/// deliberately lazy — a buffer only releases memory after
/// [`Self::SHRINK_AFTER`] consecutive frames that use less than a quarter of
/// it, so a panel that opens and closes, or a transient burst of toasts, does
/// not churn allocations. Capacities round up to powers of two, which both
/// bounds the waste and keeps an oscillating frame size from reaching the
/// shrink threshold (see [`Self::resolve`]).
struct ResizePolicy {
    capacity: u64,
    /// Capacity never drops below the size the buffer was created with, so a
    /// normal UI frame never triggers a resize after start-up.
    floor: u64,
    /// Consecutive frames seen using far less than `capacity`.
    low_frames: u32,
}

impl ResizePolicy {
    /// Only count a frame as under-used when `capacity` exceeds this multiple
    /// of what the frame needed.
    const SHRINK_HEADROOM: u64 = 4;
    /// Under-used frames in a row before memory is released. Counted per slot,
    /// so a few seconds of idle at 60 Hz with frames in flight.
    const SHRINK_AFTER: u32 = 120;

    fn new(capacity: u64) -> Self {
        let capacity = capacity.max(4);
        Self {
            capacity,
            floor: capacity,
            low_frames: 0,
        }
    }

    /// Current capacity in bytes.
    fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Back to the creation size, with the shrink count restarted.
    fn reset_to_floor(&mut self) {
        self.capacity = self.floor;
        self.low_frames = 0;
    }

    /// Account for a frame needing `needed` bytes and return the new capacity
    /// when the buffer must be recreated, or `None` to keep the existing one.
    fn resolve(&mut self, needed: u64) -> Option<u64> {
        let needed = needed.max(4);
        if needed > self.capacity {
            // Round up so a slowly growing frame does not reallocate every
            // frame, and so capacity is always on the power-of-two grid.
            self.capacity = needed.next_power_of_two().max(self.floor);
            self.low_frames = 0;
            return Some(self.capacity);
        }
        if self.capacity > self.floor
            && needed.saturating_mul(Self::SHRINK_HEADROOM) < self.capacity
        {
            self.low_frames += 1;
            if self.low_frames >= Self::SHRINK_AFTER {
                let target = needed.next_power_of_two().max(self.floor);
                self.capacity = target;
                self.low_frames = 0;
                return Some(target);
            }
        } else {
            self.low_frames = 0;
        }
        None
    }
}

struct FrameSlot {
    vertices: Option<vulkan::GpuBuffer>,
    indices: Option<vulkan::GpuBuffer>,
    vertex_policy: ResizePolicy,
    index_policy: ResizePolicy,
    /// Transfer-staging buffer for this slot's atlas uploads (grows with
    /// the same lazy-shrink policy, allocated on first atlas upload).
    staging: Option<vulkan::GpuBuffer>,
    staging_policy: ResizePolicy,
    /// Every texture the slot's last-recorded commands can touch — atlas
    /// pages (current and retired epochs), custom textures, the white
    /// fallback. Cleared only when the slot records again, which is after
    /// its fence recycled, so a retained GPU texture can never be
    /// destroyed under a queued draw. `retained_seen` dedups the same
    /// texture bound by many draws.
    retained: Vec<Arc<GpuTexture>>,
    retained_seen: std::collections::HashSet<vk::Image>,
    /// Host vertex buffer for converted quads (instance lane); same
    /// lazy-shrink policy, 4 KiB floor.
    instances: vulkan::GpuBuffer,
    instances_policy: ResizePolicy,
    /// Reusable preparation vectors — the buffers' `Vec` capacities are
    /// owned per slot so a busy frame doesn't re-allocate every record.
    prepared: Prepared,
}

impl FrameSlot {
    /// Keep `texture` alive until this slot's next record (fence-safe);
    /// deduplicates by image handle so a texture bound by many draws
    /// costs one `Arc`.
    fn retain(&mut self, texture: &Arc<GpuTexture>) {
        if self.retained_seen.insert(texture.image()) {
            self.retained.push(Arc::clone(texture));
        }
    }
}

/// A page's GPU texture plus whether it has ever held pixels (the first
/// upload transitions UNDEFINED; later ones transition from GENERAL).
struct PageTex {
    texture: Arc<GpuTexture>,
    written: bool,
}

/// GPU memory of one atlas identity: page textures and the last uploaded
/// revision per page — the renderer-side of [`super::atlas::upload_plan`].
struct AtlasGpu {
    id: u64,
    epoch: u64,
    revisions: std::collections::HashMap<usize, u64>,
    pages: Vec<Option<PageTex>>,
}

/// Records [`UiFrame`]s onto a render target. See the module docs.
pub struct UiRenderer {
    ctx: Arc<GpuContext>,
    /// Legacy indexed pipeline — built lazily if a draw falls back from instancing.
    pipeline: Option<vk::Pipeline>,
    /// Private instanced-quad pipeline (same layout/dsl/push).
    instance_pipeline: vk::Pipeline,
    /// Frosted-glass variants in gamma mode — built lazily on first glass draw.
    backdrop_pipeline: Option<vk::Pipeline>,
    backdrop_instance_pipeline: Option<vk::Pipeline>,
    /// Compact instancing path — `set_instancing(false)` renders the old
    /// raw-indexed path byte-for-byte.
    instancing: bool,
    layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
    white: Arc<GpuTexture>,
    slots: Vec<FrameSlot>,
    /// Frosted-glass backdrop textures, one set per frame slot, made on first use.
    backdrop: Vec<Option<backdrop::Backdrop>>,
    backdrop_format: vk::Format,
    blend: UiBlend,
    blur: Option<backdrop::BlurPipelines>,
    /// GPU memory of the atlas identity the last frame carried.
    atlas: Option<AtlasGpu>,
    page_size: u32,
    draws: u64,
    /// Raw upload diagnostics from the last record (bytes written into
    /// the slot buffers; instance lane added by the compact path).
    last_vertex_bytes: u64,
    last_index_bytes: u64,
    last_instance_bytes: u64,
    last_instances: u64,
    last_raw_tris: u64,
    last_draws: u64,
}

/// Colour space the UI pass blends in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiBlend {
    /// sRGB-encoded values, the way web browsers composite: translucent
    /// layers, gradients and anti-aliased edges match CSS pixel for pixel.
    /// Needs a UNORM target holding encoded values. The engine default.
    #[default]
    Gamma,
    /// Physically linear light: colours are decoded to linear and the sRGB
    /// target encodes on write. Light-on-dark edges look heavier and dark
    /// text on light backgrounds thinner than in browsers. Opt in with
    /// [`Renderer::new_with_ui_blend`](crate::gfx::Renderer::new_with_ui_blend).
    Linear,
}

impl UiRenderer {
    /// Build for targets of `format` (the swapchain format), blending in
    /// [`UiBlend::default()`]. With the default [`UiBlend::Gamma`], `format`
    /// must be a UNORM target that stores sRGB-encoded values.
    pub fn new(ctx: &Arc<GpuContext>, format: vk::Format, page_size: u32) -> Self {
        Self::new_with_blend(ctx, format, page_size, UiBlend::default())
    }

    /// Like [`Self::new`], with an explicit [`UiBlend`]. For
    /// [`UiBlend::Gamma`], `format` must be a UNORM target that stores
    /// sRGB-encoded values.
    pub fn new_with_blend(
        ctx: &Arc<GpuContext>,
        format: vk::Format,
        page_size: u32,
        blend: UiBlend,
    ) -> Self {
        let gamma = blend == UiBlend::Gamma;
        let dsl = texture::create_sampled_texture_set_layout(ctx);
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8);
        let layout = {
            let set_layouts = [dsl];
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push));
            unsafe { ctx.device().create_pipeline_layout(&info, None) }.expect("ui pipeline layout")
        };
        let shader = ctx.compile_shader(include_str!("ui.wgsl"), "ui");
        // Instance records arrive on binding 1 (rate INSTANCE) so the
        // shared pass can keep both buffers bound — indexed draws use
        // binding 0, instanced draws use binding 1.
        let (instance_binding, instance_attrs) = instance_vertex_desc();
        let instance_pipeline = create_ui_pipeline(
            ctx,
            shader,
            &instance_binding,
            &instance_attrs,
            layout,
            format,
            if gamma {
                c"vs_instance_gamma"
            } else {
                c"vs_instance"
            },
            if gamma {
                c"fs_main_gamma"
            } else {
                c"fs_linear"
            },
        );
        unsafe { ctx.device().destroy_shader_module(shader, None) };

        let white = Arc::new(GpuTexture::white(ctx));
        let slots: Vec<FrameSlot> = (0..ctx.max_frames_in_flight())
            .map(|_| new_slot(ctx, 8 * 1024, 4 * 1024))
            .collect();
        let backdrop = (0..slots.len()).map(|_| None).collect();

        Self {
            ctx: Arc::clone(ctx),
            pipeline: None,
            instance_pipeline,
            backdrop_pipeline: None,
            backdrop_instance_pipeline: None,
            instancing: true,
            layout,
            dsl,
            white,
            slots,
            backdrop,
            backdrop_format: format,
            blend,
            blur: None,
            atlas: None,
            page_size,
            draws: 0,
            last_vertex_bytes: 0,
            last_index_bytes: 0,
            last_instance_bytes: 0,
            last_instances: 0,
            last_raw_tris: 0,
            last_draws: 0,
        }
    }

    fn ensure_legacy_pipeline(&mut self) -> vk::Pipeline {
        if let Some(p) = self.pipeline {
            return p;
        }
        let gamma = self.blend == UiBlend::Gamma;
        let shader = self.ctx.compile_shader(include_str!("ui.wgsl"), "ui");
        let (binding, attrs) = raw_vertex_desc();
        let p = create_ui_pipeline(
            &self.ctx,
            shader,
            &binding,
            &attrs,
            self.layout,
            self.backdrop_format,
            c"vs_main",
            if gamma { c"fs_main_gamma" } else { c"fs_main" },
        );
        unsafe { self.ctx.device().destroy_shader_module(shader, None) };
        self.pipeline = Some(p);
        p
    }

    fn ensure_backdrop_pipeline(&mut self) -> vk::Pipeline {
        if self.blend != UiBlend::Gamma {
            return self.ensure_legacy_pipeline();
        }
        if let Some(p) = self.backdrop_pipeline {
            return p;
        }
        let shader = self.ctx.compile_shader(include_str!("ui.wgsl"), "ui");
        let (binding, attrs) = raw_vertex_desc();
        let p = create_ui_pipeline(
            &self.ctx,
            shader,
            &binding,
            &attrs,
            self.layout,
            self.backdrop_format,
            c"vs_main",
            c"fs_backdrop",
        );
        unsafe { self.ctx.device().destroy_shader_module(shader, None) };
        self.backdrop_pipeline = Some(p);
        p
    }

    fn ensure_backdrop_instance_pipeline(&mut self) -> vk::Pipeline {
        if self.blend != UiBlend::Gamma {
            return self.instance_pipeline;
        }
        if let Some(p) = self.backdrop_instance_pipeline {
            return p;
        }
        let shader = self.ctx.compile_shader(include_str!("ui.wgsl"), "ui");
        let (binding, attrs) = instance_vertex_desc();
        let p = create_ui_pipeline(
            &self.ctx,
            shader,
            &binding,
            &attrs,
            self.layout,
            self.backdrop_format,
            c"vs_instance_gamma",
            c"fs_backdrop",
        );
        unsafe { self.ctx.device().destroy_shader_module(shader, None) };
        self.backdrop_instance_pipeline = Some(p);
        p
    }

    /// Compact instanced-quad path (default on). `false` renders the
    /// legacy raw-indexed pipeline byte-for-byte — for A/B comparison.
    pub fn set_instancing(&mut self, enabled: bool) {
        self.instancing = enabled;
    }

    /// Draw calls recorded since construction (diagnostics).
    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// Total bytes reserved by the per-frame-slot upload buffers, summed
    /// across all slots in flight (diagnostics).
    pub fn buffer_capacity(&self) -> u64 {
        self.slots
            .iter()
            .map(|slot| {
                slot.vertices.as_ref().map_or(0, |b| b.size)
                    + slot.indices.as_ref().map_or(0, |b| b.size)
                    + slot.instances.size
                    + slot.staging.as_ref().map_or(0, |b| b.size)
            })
            .sum()
    }

    /// Give every frame slot's upload buffers back at their policy floor, so
    /// the peak of a busy spell is released at once instead of after a run of
    /// quiet frames. An idle UI (hidden
    /// window, minimised) calls this so it holds only its start-up footprint.
    ///
    /// The caller guarantees the GPU is idle (`device_wait_idle`): the old
    /// buffers are freed here, and so are the frosted-glass backdrop textures
    /// (they are remade on the next glass frame). Each slot's retained list is
    /// cleared too, since it would otherwise keep those textures alive until
    /// the slot's next record. The current atlas pages are not touched.
    pub fn trim(&mut self) {
        for backdrop in &mut self.backdrop {
            *backdrop = None;
        }
        for slot in &mut self.slots {
            slot.retained.clear();
            slot.retained_seen.clear();
            slot.vertex_policy.reset_to_floor();
            slot.index_policy.reset_to_floor();
            slot.instances_policy.reset_to_floor();
            slot.staging_policy.reset_to_floor();
            slot.vertices = None;
            slot.indices = None;
            slot.instances = self
                .ctx
                .create_host_vertex_buffer("ui instances", slot.instances_policy.capacity());
            slot.staging = None;
        }
    }

    /// Record the UI pass onto `target_view` (`extent` in pixels). The target
    /// must already hold the frame underneath (load, don't clear). The target
    /// is expected in `GENERAL`; see [`Self::record_in_layout`].
    ///
    /// Frosted-glass ([`CmdTexture::Backdrop`](super::draw::CmdTexture::Backdrop))
    /// draws sample their backdrop textures, but this entry point has no image
    /// to copy from, so it does not capture one: the glass shows the last
    /// backdrop this slot captured (blank until the first capture). Use
    /// [`Self::record_in_layout`] with the target image to get a live backdrop.
    pub fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        target_view: vk::ImageView,
        extent: (u32, u32),
        frame: &UiFrame,
    ) {
        self.record_in_layout(
            cmd,
            frame_slot,
            vk::Image::null(),
            target_view,
            extent,
            frame,
            vk::ImageLayout::GENERAL,
        );
    }

    /// [`Self::record`] for a target held in `layout` (for example the
    /// swapchain image, `COLOR_ATTACHMENT_OPTIMAL`, when the frame is composed
    /// straight into it). The layout only changes the attachment's declared
    /// layout; the pipelines and their blending are the same.
    ///
    /// `target_image` is the image behind `target_view`, in `layout`. It is
    /// copied for frosted-glass backdrops, and the copy needs it to be
    /// blittable (`TRANSFER_SRC`). Pass `vk::Image::null()` when it is not
    /// available: the backdrop capture is then skipped and glass shows the last
    /// backdrop this slot captured.
    #[allow(clippy::too_many_arguments)]
    pub fn record_in_layout(
        &mut self,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        target_image: vk::Image,
        target_view: vk::ImageView,
        extent: (u32, u32),
        frame: &UiFrame,
        layout: vk::ImageLayout,
    ) {
        self.record_in_layout_with_clear(
            cmd,
            frame_slot,
            target_image,
            target_view,
            extent,
            frame,
            layout,
            None,
        );
    }

    /// [`Self::record_in_layout`], optionally merging an opaque `clear`
    /// colour into the initial UI render pass so UI-only frames skip an extra
    /// empty clear pass.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_in_layout_with_clear(
        &mut self,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        target_image: vk::Image,
        target_view: vk::ImageView,
        extent: (u32, u32),
        frame: &UiFrame,
        layout: vk::ImageLayout,
        clear: Option<[f32; 4]>,
    ) {
        let device = self.ctx.device().clone();
        // The *last-record* counters reset every call — an empty-upload
        // or no-cmd frame reports zeros, never a stale previous frame.
        self.last_draws = 0;
        self.last_instances = 0;
        self.last_raw_tris = 0;
        self.last_vertex_bytes = 0;
        self.last_index_bytes = 0;
        self.last_instance_bytes = 0;

        // 1. Size this frame slot's upload buffers. Growth is immediate so the
        //    geometry always fits; shrink is lazy so an idle UI releases memory
        //    without a resize thrash. This runs before the empty-frame return:
        //    an idle UI is exactly when buffers should shrink back. The slot's
        //    fence has recycled by now (see the module docs), so replacing its
        //    buffer cannot disturb work still in flight.
        let slot_index = frame_slot % self.slots.len();
        // Backdrop textures are made before the slot's retained list is cleared below. Old ones
        // stay alive through that list until the slot's last submission has completed.
        if frame
            .cmds
            .iter()
            .any(|draw| matches!(draw.texture, super::draw::CmdTexture::Backdrop))
        {
            self.ensure_backdrop(slot_index, cmd, extent);
        }
        {
            // The slot is about to be re-recorded: whatever its previous
            // submission referenced has completed by fence order.
            self.slots[slot_index].retained.clear();
            self.slots[slot_index].retained_seen.clear();
            // Compact geometry: quads become 60-byte instances, arbitrary
            // triangles stay indexed — converted quads' legacy vertices
            // are NOT uploaded at all.
            prepare::prepare_into(
                &mut self.slots[slot_index].prepared,
                &frame.cmds,
                &frame.vertices,
                &frame.indices,
                self.instancing,
            );
            let slot = &mut self.slots[slot_index];
            let vertex_bytes =
                (slot.prepared.vertices.len() * std::mem::size_of::<UiVertex>()) as u64;
            let index_bytes = (slot.prepared.indices.len() * 4) as u64;
            let instance_bytes =
                (slot.prepared.instances.len() * std::mem::size_of::<UiInstance>()) as u64;
            let ctx = &self.ctx;
            resize_optional_buffer(
                &mut slot.vertices,
                &mut slot.vertex_policy,
                vertex_bytes,
                |cap| ctx.create_host_vertex_buffer("ui vertices", cap),
            );
            resize_optional_buffer(
                &mut slot.indices,
                &mut slot.index_policy,
                index_bytes,
                |cap| ctx.create_host_index_buffer("ui indices", cap),
            );
            // Staging sees idle frames too when already allocated.
            let uploads_due = frame.atlas_snapshot.is_some() || !frame.atlas_updates.is_empty();
            if !uploads_due && slot.staging.is_some() {
                resize_optional_buffer(&mut slot.staging, &mut slot.staging_policy, 0, |cap| {
                    ctx.create_transfer_buffer("ui atlas staging", cap)
                });
            }
            if let Some(capacity) = slot.instances_policy.resolve(instance_bytes) {
                slot.instances = ctx.create_host_vertex_buffer("ui instances", capacity);
            }
        }

        // 2. Atlas uploads — recorded even when the frame has no draw
        //    commands (a pure-upload frame still owes the GPU the pixels).
        if let Some(snapshot) = &frame.atlas_snapshot {
            self.upload_snapshot(cmd, slot_index, snapshot);
        } else if !frame.atlas_updates.is_empty() {
            self.upload_legacy(cmd, slot_index, frame);
        }

        let scissor_full = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent.0,
                height: extent.1,
            },
        };
        if frame.cmds.is_empty() {
            if let Some(clear_color) = clear {
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(target_view)
                    .image_layout(layout)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: clear_color,
                        },
                    });
                unsafe {
                    device.cmd_begin_rendering(
                        cmd,
                        &vk::RenderingInfo::default()
                            .render_area(scissor_full)
                            .layer_count(1)
                            .color_attachments(std::slice::from_ref(&attachment)),
                    );
                    device.cmd_end_rendering(cmd);
                }
            }
            return;
        }

        // 3. Write this frame's prepared geometry into the slot's buffers.
        let slot = &mut self.slots[slot_index];
        self.last_vertex_bytes =
            (slot.prepared.vertices.len() * std::mem::size_of::<UiVertex>()) as u64;
        self.last_index_bytes = (slot.prepared.indices.len() * 4) as u64;
        self.last_raw_tris = (slot.prepared.indices.len() / 3) as u64;
        self.last_instances = slot.prepared.instances.len() as u64;
        self.last_instance_bytes =
            (slot.prepared.instances.len() * std::mem::size_of::<UiInstance>()) as u64;
        let prepared = std::mem::take(&mut slot.prepared);
        if !prepared.vertices.is_empty()
            && let Some(vertices) = slot.vertices.as_mut()
        {
            vertices.write(0, bytemuck::cast_slice(&prepared.vertices));
        }
        if !prepared.indices.is_empty()
            && let Some(indices) = slot.indices.as_mut()
        {
            indices.write(0, bytemuck::cast_slice(&prepared.indices));
        }
        if !prepared.instances.is_empty() {
            slot.instances
                .write(0, bytemuck::cast_slice(&prepared.instances));
        }

        let vertex_buffer = slot
            .vertices
            .as_ref()
            .map_or(vk::Buffer::null(), |b| b.buffer);
        let index_buffer = slot
            .indices
            .as_ref()
            .map_or(vk::Buffer::null(), |b| b.buffer);
        let instance_buffer = slot.instances.buffer;

        // Lazily compile only the pipeline variants this frame actually uses.
        let mut need_legacy = false;
        let mut need_backdrop_indexed = false;
        let mut need_backdrop_instanced = false;
        for prepared_draw in &prepared.draws {
            let glass = matches!(
                frame.cmds[prepared_draw.cmd_index].texture,
                super::draw::CmdTexture::Backdrop
            );
            match prepared_draw.kind {
                DrawKind::Indexed { .. } => {
                    if glass {
                        need_backdrop_indexed = true;
                    } else {
                        need_legacy = true;
                    }
                }
                DrawKind::Instanced { .. } => {
                    if glass {
                        need_backdrop_instanced = true;
                    }
                }
            }
        }
        let legacy_pipeline = if need_legacy {
            self.ensure_legacy_pipeline()
        } else {
            vk::Pipeline::null()
        };
        let backdrop_pipeline = if need_backdrop_indexed {
            self.ensure_backdrop_pipeline()
        } else {
            vk::Pipeline::null()
        };
        let backdrop_instance_pipeline = if need_backdrop_instanced {
            self.ensure_backdrop_instance_pipeline()
        } else {
            self.instance_pipeline
        };
        let instance_pipeline = self.instance_pipeline;
        let pipeline_layout = self.layout;
        let push = [frame.size.0, frame.size.1];

        let begin_pass = |cmd: vk::CommandBuffer, initial_clear: Option<[f32; 4]>| unsafe {
            let (load_op, clear_val) = match initial_clear {
                Some(c) => (
                    vk::AttachmentLoadOp::CLEAR,
                    vk::ClearValue {
                        color: vk::ClearColorValue { float32: c },
                    },
                ),
                None => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
            };
            device.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(scissor_full)
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(target_view)
                        .image_layout(layout)
                        .load_op(load_op)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(clear_val)]),
            );
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: extent.1 as f32,
                    width: extent.0 as f32,
                    height: -(extent.1 as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_bind_vertex_buffers(cmd, 1, &[instance_buffer], &[0]);
            device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );
        };
        begin_pass(cmd, clear);
        let mut captured = false;

        let mut bound_set = vk::DescriptorSet::null();
        let mut bound_scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: i32::MIN,
                y: i32::MIN,
            },
            extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
        };
        let mut bound_pipeline = vk::Pipeline::null();
        let mut bound_indexed_vb = false;
        let mut last_retained_image = vk::Image::null();
        for prepared_draw in &prepared.draws {
            let draw = &frame.cmds[prepared_draw.cmd_index];
            // The first glass draw needs everything beneath it drawn. End the pass, blur the target so
            // far, and begin again; cached bindings are stale after that and are re-established.
            if matches!(draw.texture, super::draw::CmdTexture::Backdrop) && !captured {
                captured = true;
                if target_image != vk::Image::null() {
                    unsafe { device.cmd_end_rendering(cmd) };
                    self.capture_backdrop(slot_index, cmd, target_image, layout, extent);
                    begin_pass(cmd, None);
                    bound_set = vk::DescriptorSet::null();
                    bound_scissor = vk::Rect2D {
                        offset: vk::Offset2D {
                            x: i32::MIN,
                            y: i32::MIN,
                        },
                        extent: vk::Extent2D {
                            width: u32::MAX,
                            height: u32::MAX,
                        },
                    };
                    bound_pipeline = vk::Pipeline::null();
                    bound_indexed_vb = false;
                }
            }
            let glass = matches!(draw.texture, super::draw::CmdTexture::Backdrop);
            let backdrop_tex = glass.then(|| self.backdrop_texture(slot_index));
            let tex: &Arc<GpuTexture> = if let Some(ref b) = backdrop_tex {
                b
            } else {
                match &draw.texture {
                    super::draw::CmdTexture::Backdrop => unreachable!("handled above"),
                    super::draw::CmdTexture::White => &self.white,
                    super::draw::CmdTexture::AtlasPage(page) => {
                        &self
                            .atlas
                            .as_ref()
                            .and_then(|atlas| atlas.pages.get(*page))
                            .and_then(|p| p.as_ref())
                            .expect("atlas page referenced by a draw exists")
                            .texture
                    }
                    super::draw::CmdTexture::Custom(texture) => texture,
                }
            };
            let img = tex.image();
            if img != last_retained_image {
                last_retained_image = img;
                self.slots[slot_index].retain(tex);
            }
            let set = tex.sampled_descriptor_set();
            unsafe {
                if bound_set != set {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.layout,
                        0,
                        &[set],
                        &[],
                    );
                    bound_set = set;
                }
                let mut scissor = to_scissor(draw, frame.size, extent);
                if let Some(d) = frame.dirty_rect {
                    let (sx, sy) = scissor_scale(frame.size, extent);
                    let dx0 = (d.x * sx).floor().clamp(0.0, extent.0 as f32) as i32;
                    let dy0 = (d.y * sy).floor().clamp(0.0, extent.1 as f32) as i32;
                    let dx1 = ((d.x + d.w) * sx).ceil().clamp(0.0, extent.0 as f32) as i32;
                    let dy1 = ((d.y + d.h) * sy).ceil().clamp(0.0, extent.1 as f32) as i32;
                    let sx0 = scissor.offset.x.max(dx0);
                    let sy0 = scissor.offset.y.max(dy0);
                    let sx1 = (scissor.offset.x + scissor.extent.width as i32).min(dx1);
                    let sy1 = (scissor.offset.y + scissor.extent.height as i32).min(dy1);
                    scissor = vk::Rect2D {
                        offset: vk::Offset2D { x: sx0, y: sy0 },
                        extent: vk::Extent2D {
                            width: (sx1 - sx0).max(0) as u32,
                            height: (sy1 - sy0).max(0) as u32,
                        },
                    };
                    if scissor.extent.width == 0 || scissor.extent.height == 0 {
                        continue;
                    }
                }
                if bound_scissor != scissor {
                    device.cmd_set_scissor(cmd, 0, &[scissor]);
                    bound_scissor = scissor;
                }
                match prepared_draw.kind {
                    DrawKind::Indexed { first, count } => {
                        if !bound_indexed_vb {
                            device.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
                            device.cmd_bind_index_buffer(
                                cmd,
                                index_buffer,
                                0,
                                vk::IndexType::UINT32,
                            );
                            bound_indexed_vb = true;
                        }
                        let wanted = if glass {
                            backdrop_pipeline
                        } else {
                            legacy_pipeline
                        };
                        if bound_pipeline != wanted {
                            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, wanted);
                            bound_pipeline = wanted;
                        }
                        device.cmd_draw_indexed(cmd, count, 1, first, 0, 0);
                    }
                    DrawKind::Instanced { first, count } => {
                        let wanted = if glass {
                            backdrop_instance_pipeline
                        } else {
                            instance_pipeline
                        };
                        if bound_pipeline != wanted {
                            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, wanted);
                            bound_pipeline = wanted;
                        }
                        device.cmd_draw(cmd, 6, count, 0, first);
                    }
                }
            }
            self.draws += 1;
            self.last_draws += 1;
        }
        self.slots[slot_index].prepared = prepared;

        unsafe { device.cmd_end_rendering(cmd) };
    }

    /// Record the transfers an [`AtlasSnapshot`] needs into the caller's
    /// command buffer (before any `begin_rendering`), updating this
    /// renderer's page-revision memory. Ordering proof: a slot records
    /// only after its fence recycled, the command buffer submits in order
    /// behind prior work, and page transitions keep every copy inside the
    /// same GENERAL layout — a later frame's copy therefore always sees
    /// the earlier frame's writes, and a stale snapshot restores its
    /// older pixels via a full-page copy.
    fn upload_snapshot(
        &mut self,
        cmd: vk::CommandBuffer,
        slot_index: usize,
        snapshot: &super::atlas::AtlasSnapshot,
    ) {
        // Identity change (new atlas or a post-overflow reset): the new
        // epoch starts a clean revision memory. Old page textures need
        // no blanket retention here — every slot already retains the
        // textures its own queued draws bound and its own queued uploads
        // wrote, so nothing still in flight loses its backing.
        let identity = self.atlas.as_ref().map(|a| (a.id, a.epoch));
        if identity != Some((snapshot.atlas_id, snapshot.epoch)) {
            self.atlas = Some(AtlasGpu {
                id: snapshot.atlas_id,
                epoch: snapshot.epoch,
                revisions: std::collections::HashMap::new(),
                pages: Vec::new(),
            });
        }
        let atlas = self.atlas.as_mut().expect("set above");
        while atlas.pages.len() < snapshot.pages.len() {
            atlas.pages.push(None);
        }
        let plan =
            super::atlas::upload_plan(Some((atlas.id, atlas.epoch, &atlas.revisions)), snapshot);

        let active_rows = |page: &super::atlas::SnapshotPage| -> u32 {
            let bpp = page.format.bytes_per_texel();
            let row_bytes = snapshot.page_size as usize * bpp;
            if row_bytes == 0 {
                return 0;
            }
            match page.pixels.iter().rposition(|&b| b != 0) {
                Some(last) => ((last / row_bytes) as u32 + 2).min(snapshot.page_size),
                None => 0,
            }
        };

        // Pass 1: sizes for the staging buffer.
        let mut needed: u64 = 0;
        let mut any_work = false;
        for (index, page) in snapshot.pages.iter().enumerate() {
            match plan[index] {
                super::atlas::PageUpload::Skip => {}
                super::atlas::PageUpload::Full => {
                    any_work = true;
                    let rows = active_rows(page) as u64;
                    let bpp = page.format.bytes_per_texel() as u64;
                    needed += (rows * snapshot.page_size as u64 * bpp + 3) & !3;
                }
                super::atlas::PageUpload::Dirty => {
                    any_work = true;
                    let rect = page.last_dirty.expect("dirty plan has a rect");
                    let bpp = page.format.bytes_per_texel() as u64;
                    needed += (rect.width as u64 * rect.height as u64 * bpp + 3) & !3;
                }
            }
        }
        if !any_work {
            if self.slots[slot_index].staging.is_some() {
                let ctx = &self.ctx;
                let slot = &mut self.slots[slot_index];
                resize_optional_buffer(&mut slot.staging, &mut slot.staging_policy, 0, |cap| {
                    ctx.create_transfer_buffer("ui atlas staging", cap)
                });
            }
            return;
        }
        let ctx = &self.ctx;
        let slot = &mut self.slots[slot_index];
        resize_optional_buffer(&mut slot.staging, &mut slot.staging_policy, needed, |cap| {
            ctx.create_transfer_buffer("ui atlas staging", cap)
        });
        let staging = slot.staging.as_mut().expect("staging allocated above");
        let device = self.ctx.device().clone();
        let mut offset: u64 = 0;
        for (index, page) in snapshot.pages.iter().enumerate() {
            let is_full = matches!(plan[index], super::atlas::PageUpload::Full);
            let (rect, bpp) = match plan[index] {
                super::atlas::PageUpload::Skip => continue,
                super::atlas::PageUpload::Full => (
                    super::atlas::DirtyRect {
                        page: index,
                        x: 0,
                        y: 0,
                        width: snapshot.page_size,
                        height: active_rows(page),
                    },
                    page.format.bytes_per_texel(),
                ),
                super::atlas::PageUpload::Dirty => (
                    page.last_dirty.expect("dirty plan has a rect"),
                    page.format.bytes_per_texel(),
                ),
            };
            let atlas_page = atlas.pages[index].get_or_insert_with(|| PageTex {
                texture: Arc::new(GpuTexture::create_uninitialized(
                    &self.ctx,
                    "ui atlas page",
                    (snapshot.page_size, snapshot.page_size),
                    match page.format {
                        super::atlas::AtlasFormat::Coverage => vk::Format::R8_UNORM,
                        super::atlas::AtlasFormat::RgbaLinear => vk::Format::R8G8B8A8_UNORM,
                        super::atlas::AtlasFormat::RgbaSrgb => vk::Format::R8G8B8A8_SRGB,
                    },
                    match page.format {
                        super::atlas::AtlasFormat::Coverage => vk::ComponentMapping {
                            r: vk::ComponentSwizzle::ONE,
                            g: vk::ComponentSwizzle::ONE,
                            b: vk::ComponentSwizzle::ONE,
                            a: vk::ComponentSwizzle::R,
                        },
                        _ => vk::ComponentMapping::default(),
                    },
                    SamplerDesc {
                        filter: texture::Filter::Linear,
                        mip: texture::MipFilter::Nearest,
                        address: texture::AddressMode::ClampToEdge,
                        anisotropy: false,
                    },
                    TextureFormat::Linear,
                )),
                written: false,
            });
            let image = atlas_page.texture.image();
            // Upload rows tightly packed at a 4-aligned staging offset.
            let row_bytes = rect.width as usize * bpp;
            let page_row = snapshot.page_size as usize * bpp;
            for row in 0..rect.height as usize {
                let src = ((rect.y as usize + row) * page_row + rect.x as usize * bpp)
                    ..((rect.y as usize + row) * page_row + rect.x as usize * bpp + row_bytes);
                staging.write(offset + (row * row_bytes) as u64, &page.pixels[src]);
            }
            let buffer_offset = offset;
            offset += (row_bytes * rect.height as usize) as u64;
            offset = (offset + 3) & !3;
            let first_write = !atlas_page.written;
            atlas_page.written = true;
            unsafe {
                if first_write {
                    // Fresh page: nothing readable existed; also clears the
                    // transparent padding the shader's linear filter needs.
                    vulkan::image_barrier(
                        &device,
                        cmd,
                        image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageAspectFlags::COLOR,
                    );
                } else {
                    // An older snapshot may restore pixels: wait on prior
                    // fragment reads AND prior transfer writes.
                    vulkan::image_barrier(
                        &device,
                        cmd,
                        image,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageAspectFlags::COLOR,
                    );
                }
                if is_full && rect.height < snapshot.page_size {
                    device.cmd_clear_color_image(
                        cmd,
                        image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                        &[vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }],
                    );
                    if rect.height > 0 {
                        vulkan::image_barrier(
                            &device,
                            cmd,
                            image,
                            vk::ImageLayout::GENERAL,
                            vk::ImageLayout::GENERAL,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::ImageAspectFlags::COLOR,
                        );
                    }
                }
                if rect.height > 0 && rect.width > 0 {
                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(buffer_offset)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D {
                            x: rect.x as i32,
                            y: rect.y as i32,
                            z: 0,
                        })
                        .image_extent(vk::Extent3D {
                            width: rect.width.max(1),
                            height: rect.height.max(1),
                            depth: 1,
                        });
                    device.cmd_copy_buffer_to_image(
                        cmd,
                        staging.buffer,
                        image,
                        vk::ImageLayout::GENERAL,
                        &[region],
                    );
                }
                vulkan::image_barrier(
                    &device,
                    cmd,
                    image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::ImageAspectFlags::COLOR,
                );
            }
            atlas.revisions.insert(index, page.revision);
        }
        for page in atlas.pages.iter().flatten() {
            self.slots[slot_index].retain(&page.texture);
        }
    }

    /// Legacy path for standalone/raw frames: RGBA dirty rectangles without
    /// a snapshot. Behaves like the pre-snapshot renderer (full pages are
    /// RGBA-UNORM), recording into the caller's command buffer rather than
    /// a one-shot submission.
    fn upload_legacy(&mut self, cmd: vk::CommandBuffer, slot_index: usize, frame: &UiFrame) {
        // No snapshot to diff against: every update is a dirty upload.
        let total: u64 = frame
            .atlas_updates
            .iter()
            .map(|u| (u.pixels.len() as u64 + 3) & !3)
            .sum();
        let ctx = &self.ctx;
        let slot = &mut self.slots[slot_index];
        resize_optional_buffer(
            &mut slot.staging,
            &mut slot.staging_policy,
            total.max(4),
            |cap| ctx.create_transfer_buffer("ui atlas staging", cap),
        );
        let staging = slot.staging.as_mut().expect("staging allocated above");
        let device = self.ctx.device().clone();
        // Legacy pages are RGBA-UNORM under a synthetic identity — an
        // R8 snapshot atlas must not be reused for them (wrong format
        // AND wrong revision memory). Origin-slot retention covers the
        // replaced textures' queued uses, same as an epoch change.
        if self.atlas.as_ref().map(|a| (a.id, a.epoch)) != Some((0, 0)) {
            self.atlas = Some(AtlasGpu {
                id: 0,
                epoch: 0,
                revisions: std::collections::HashMap::new(),
                pages: Vec::new(),
            });
        }
        let atlas = self.atlas.as_mut().expect("set above");
        let mut offset: u64 = 0;
        for update in &frame.atlas_updates {
            let index = update.rect.page;
            while atlas.pages.len() <= index {
                atlas.pages.push(None);
            }
            let atlas_page = atlas.pages[index].get_or_insert_with(|| PageTex {
                texture: Arc::new(GpuTexture::create_uninitialized(
                    &self.ctx,
                    "ui atlas page",
                    (self.page_size, self.page_size),
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ComponentMapping::default(),
                    SamplerDesc {
                        filter: texture::Filter::Linear,
                        mip: texture::MipFilter::Nearest,
                        address: texture::AddressMode::ClampToEdge,
                        anisotropy: false,
                    },
                    TextureFormat::Linear,
                )),
                written: false,
            });
            let image = atlas_page.texture.image();
            staging.write(offset, &update.pixels);
            let buffer_offset = offset;
            offset += (update.pixels.len() as u64 + 3) & !3;
            let first_write = !atlas_page.written;
            atlas_page.written = true;
            unsafe {
                vulkan::image_barrier(
                    &device,
                    cmd,
                    image,
                    if first_write {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::GENERAL
                    },
                    vk::ImageLayout::GENERAL,
                    if first_write {
                        vk::PipelineStageFlags::TOP_OF_PIPE
                    } else {
                        vk::PipelineStageFlags::ALL_COMMANDS
                    },
                    vk::PipelineStageFlags::TRANSFER,
                    if first_write {
                        vk::AccessFlags::empty()
                    } else {
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_WRITE
                    },
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::ImageAspectFlags::COLOR,
                );
                if first_write {
                    // A fresh page with only a partial dirty rect still
                    // reads its padding: clear the whole page first.
                    device.cmd_clear_color_image(
                        cmd,
                        image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                        &[vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }],
                    );
                    // The partial copy below writes the same page: order it
                    // after the clear (SYNC-HAZARD-WRITE-AFTER-WRITE).
                    let order = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[order],
                        &[],
                        &[],
                    );
                }
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D {
                        x: update.rect.x as i32,
                        y: update.rect.y as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: update.rect.width.max(1),
                        height: update.rect.height.max(1),
                        depth: 1,
                    });
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    &[region],
                );
                vulkan::image_barrier(
                    &device,
                    cmd,
                    image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::ImageAspectFlags::COLOR,
                );
            }
        }
    }

    /// UI-owned GPU allocation sizes, one entry per resource — staging,
    /// vertex and index buffers, atlas pages, the white fallback. Raw
    /// numbers for the bench harness, not a memory claim.
    pub fn allocation_sizes(&self) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        for (slot, frame_slot) in self.slots.iter().enumerate() {
            out.push((
                format!("ui vertices[{slot}]"),
                frame_slot
                    .vertices
                    .as_ref()
                    .map_or(0, |b| b.allocation_size()),
            ));
            out.push((
                format!("ui indices[{slot}]"),
                frame_slot
                    .indices
                    .as_ref()
                    .map_or(0, |b| b.allocation_size()),
            ));
            out.push((
                format!("ui instances[{slot}]"),
                frame_slot.instances.allocation_size(),
            ));
            out.push((
                format!("ui staging[{slot}]"),
                frame_slot
                    .staging
                    .as_ref()
                    .map_or(0, |b| b.allocation_size()),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        let mut push_tex = |name: String, texture: &GpuTexture| {
            if seen.insert(texture.image()) {
                out.push((name, texture.image_allocation_size()));
            }
        };
        push_tex("ui white".to_string(), &self.white);
        if let Some(atlas) = &self.atlas {
            for (index, page) in atlas.pages.iter().enumerate() {
                if let Some(page) = page {
                    push_tex(format!("ui atlas[{index}]"), &page.texture);
                }
            }
        }
        // Retained-by-slot textures (a retired epoch still in flight) count
        // too — they are real allocations owned by this renderer's usage.
        for (slot, frame_slot) in self.slots.iter().enumerate() {
            for (i, texture) in frame_slot.retained.iter().enumerate() {
                push_tex(format!("ui retained[{slot}][{i}]"), texture);
            }
        }
        out
    }

    /// Bytes uploaded into the slot's geometry buffers by the last
    /// `record`: `(legacy_vertex_bytes, legacy_index_bytes,
    /// instance_bytes)`. Raw counts only — no derived totals.
    pub fn last_geometry_upload_sizes(&self) -> (u64, u64, u64) {
        (
            self.last_vertex_bytes,
            self.last_index_bytes,
            self.last_instance_bytes,
        )
    }

    /// Primitive counts of the last record: `(instances, raw_triangles,
    /// draws)`.
    pub fn last_prepared_counts(&self) -> (u64, u64, u64) {
        (self.last_instances, self.last_raw_tris, self.last_draws)
    }
}

impl Drop for UiRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            let device = self.ctx.device();
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline(self.instance_pipeline, None);
            if let Some(p) = self.backdrop_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.backdrop_instance_pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            if let Some(ref blur) = self.blur {
                blur.destroy(&self.ctx);
            }
            device.destroy_descriptor_set_layout(self.dsl, None);
        }
    }
}

fn scissor_scale(frame_size: (f32, f32), extent: (u32, u32)) -> (f32, f32) {
    if frame_size.0 > 0.0 && frame_size.1 > 0.0 {
        let sx = extent.0 as f32 / frame_size.0;
        let sy = extent.1 as f32 / frame_size.1;
        let sx = if (sx - 1.0).abs() > 1e-5 { sx } else { 1.0 };
        let sy = if (sy - 1.0).abs() > 1e-5 { sy } else { 1.0 };
        (sx, sy)
    } else {
        (1.0, 1.0)
    }
}

fn to_scissor(draw: &DrawCmd, frame_size: (f32, f32), extent: (u32, u32)) -> vk::Rect2D {
    let Some(rect) = draw.scissor else {
        return vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent.0,
                height: extent.1,
            },
        };
    };
    let (sx, sy) = scissor_scale(frame_size, extent);
    let x0 = (rect.x * sx).floor().clamp(0.0, extent.0 as f32) as i32;
    let y0 = (rect.y * sy).floor().clamp(0.0, extent.1 as f32) as i32;
    let x1 = ((rect.x + rect.w) * sx).ceil().clamp(0.0, extent.0 as f32) as i32;
    let y1 = ((rect.y + rect.h) * sy).ceil().clamp(0.0, extent.1 as f32) as i32;
    vk::Rect2D {
        offset: vk::Offset2D { x: x0, y: y0 },
        extent: vk::Extent2D {
            width: (x1 - x0).max(0) as u32,
            height: (y1 - y0).max(0) as u32,
        },
    }
}

fn raw_vertex_desc() -> (
    vk::VertexInputBindingDescription,
    [vk::VertexInputAttributeDescription; 5],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<UiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(16),
        // Shape parameters: half-size, corner radius, mode.
        vk::VertexInputAttributeDescription::default()
            .location(3)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(20),
        // Extra shape data (the rounded clip's half extent; zero for
        // every other mode).
        vk::VertexInputAttributeDescription::default()
            .location(4)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(36),
    ];
    (binding, attrs)
}

fn instance_vertex_desc() -> (
    vk::VertexInputBindingDescription,
    [vk::VertexInputAttributeDescription; 5],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(1)
        .stride(std::mem::size_of::<UiInstance>() as u32)
        .input_rate(vk::VertexInputRate::INSTANCE);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(1)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(1)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(16),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(32),
        vk::VertexInputAttributeDescription::default()
            .location(3)
            .binding(1)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(36),
        vk::VertexInputAttributeDescription::default()
            .location(4)
            .binding(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(52),
    ];
    (binding, attrs)
}

fn resize_optional_buffer(
    buf: &mut Option<vulkan::GpuBuffer>,
    policy: &mut ResizePolicy,
    needed: u64,
    make: impl FnOnce(u64) -> vulkan::GpuBuffer,
) {
    if needed == 0 {
        if let Some(cap) = buf.as_ref().and_then(|_| policy.resolve(0)) {
            *buf = Some(make(cap));
        }
        return;
    }
    let alloc_cap = match buf {
        None => Some(policy.resolve(needed).unwrap_or_else(|| policy.capacity())),
        Some(_) => policy.resolve(needed),
    };
    if let Some(cap) = alloc_cap {
        *buf = Some(make(cap));
    }
}

fn new_slot(ctx: &Arc<GpuContext>, vertex_capacity: u64, index_capacity: u64) -> FrameSlot {
    FrameSlot {
        vertices: None,
        indices: None,
        vertex_policy: ResizePolicy::new(vertex_capacity),
        index_policy: ResizePolicy::new(index_capacity),
        staging: None,
        staging_policy: ResizePolicy::new(16 * 1024),
        retained: Vec::new(),
        retained_seen: std::collections::HashSet::new(),
        instances: ctx.create_host_vertex_buffer("ui instances", 4 * 1024),
        instances_policy: ResizePolicy::new(4 * 1024),
        prepared: Prepared::default(),
    }
}

#[allow(clippy::too_many_arguments)]
fn create_ui_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    binding: &vk::VertexInputBindingDescription,
    attrs: &[vk::VertexInputAttributeDescription],
    layout: vk::PipelineLayout,
    format: vk::Format,
    vs: &std::ffi::CStr,
    fs: &std::ffi::CStr,
) -> vk::Pipeline {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(vs),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
            .name(fs),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(binding))
        .vertex_attribute_descriptions(attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // UI geometry is authored clockwise in pixel space; with the negative
    // viewport's winding flip that is the front face. No culling either way:
    // quads are cheap and text runs must not vanish on a winding surprise.
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Straight-alpha blending over the composed frame.
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD);
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(vk::Format::UNDEFINED);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    unsafe {
        ctx.device()
            .create_graphics_pipelines(ctx.pipeline_cache(), &[info], None)
            .map_err(|(_, err)| err)
            .expect("create ui pipeline")[0]
    }
}

/// Convert an sRGB colour to linear light — mirrors `ui.wgsl`'s vertex path.
/// Exposed for tests that check CPU and shader agree.
pub fn srgb_to_linear(value: u8) -> f32 {
    let c = value as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::Color;
    use crate::ui::atlas::DirtyRect;

    #[test]
    fn srgb_conversion_endpoints() {
        assert_eq!(srgb_to_linear(0), 0.0);
        assert!((srgb_to_linear(255) - 1.0).abs() < 1e-6);
        assert!((srgb_to_linear(128) - 0.2158).abs() < 1e-3, "mid-grey");
    }

    #[test]
    fn scissor_clamps_to_target() {
        let draw = DrawCmd {
            texture: super::super::draw::CmdTexture::White,
            scissor: Some(crate::ui::Rect::new(-10.0, -5.0, 50.0, 40.0)),
            first_index: 0,
            index_count: 0,
            layer: 0,
        };
        let scissor = to_scissor(&draw, (100.0, 100.0), (100, 100));
        assert_eq!(scissor.offset, vk::Offset2D { x: 0, y: 0 });
        assert_eq!(scissor.extent.width, 40);
        assert_eq!(scissor.extent.height, 35);

        let draw = DrawCmd {
            scissor: Some(crate::ui::Rect::new(80.0, 80.0, 50.0, 50.0)),
            ..draw
        };
        let scissor = to_scissor(&draw, (100.0, 100.0), (100, 100));
        assert_eq!(scissor.extent.width, 20);
        assert_eq!(scissor.extent.height, 20);

        let scaled_draw = DrawCmd {
            scissor: Some(crate::ui::Rect::new(100.0, 50.0, 200.0, 100.0)),
            ..draw
        };
        let scaled = to_scissor(&scaled_draw, (640.0, 480.0), (1280, 960));
        assert_eq!(scaled.offset, vk::Offset2D { x: 200, y: 100 });
        assert_eq!(scaled.extent.width, 400);
        assert_eq!(scaled.extent.height, 200);
    }

    #[test]
    fn scissor_none_is_full_target() {
        let draw = DrawCmd {
            texture: super::super::draw::CmdTexture::White,
            scissor: None,
            first_index: 0,
            index_count: 0,
            layer: 0,
        };
        let scissor = to_scissor(&draw, (640.0, 480.0), (640, 480));
        assert_eq!(scissor.extent.width, 640);
        assert_eq!(scissor.extent.height, 480);
    }

    #[test]
    fn atlas_update_carries_pixels() {
        let update = AtlasUpdate {
            rect: DirtyRect {
                page: 2,
                x: 1,
                y: 2,
                width: 3,
                height: 4,
            },
            pixels: vec![0; 3 * 4 * 4],
        };
        assert_eq!(update.rect.page, 2);
        assert_eq!(update.pixels.len(), 3 * 4 * 4);
    }

    #[test]
    fn resize_grows_immediately_to_fit() {
        let mut policy = ResizePolicy::new(64 * 1024);
        assert_eq!(policy.capacity(), 64 * 1024);
        // A frame bigger than capacity resizes right away, rounded up.
        assert_eq!(policy.resolve(100 * 1024), Some(128 * 1024));
        assert_eq!(policy.capacity(), 128 * 1024);
        // A frame that still fits does not resize.
        assert_eq!(policy.resolve(128 * 1024), None);
    }

    #[test]
    fn resize_holds_capacity_until_sustained_idle() {
        let mut policy = ResizePolicy::new(64 * 1024);
        policy.resolve(256 * 1024);
        // One frame short of the threshold: nothing released yet.
        for _ in 0..(ResizePolicy::SHRINK_AFTER - 1) {
            assert_eq!(policy.resolve(8 * 1024), None);
        }
        assert_eq!(policy.capacity(), 256 * 1024);
        // Sustained under-use trips the shrink, down to the floor (small
        // frames never need less, since the floor is the starting size).
        assert_eq!(policy.resolve(8 * 1024), Some(64 * 1024));
        assert_eq!(policy.capacity(), 64 * 1024);
    }

    #[test]
    fn resize_never_shrinks_below_the_floor() {
        let mut policy = ResizePolicy::new(64 * 1024);
        policy.resolve(512 * 1024);
        for _ in 0..ResizePolicy::SHRINK_AFTER {
            policy.resolve(4);
        }
        assert_eq!(policy.capacity(), 64 * 1024, "returns to the initial floor");
    }

    #[test]
    fn resize_does_not_thrash_on_oscillation() {
        let mut policy = ResizePolicy::new(64 * 1024);
        // Alternating large/small frames: the small frames never stay under the
        // threshold long enough to shrink, so only the first grow reallocates.
        let mut resizes = 0;
        for i in 0..1000 {
            let needed = if i % 2 == 0 { 200 * 1024 } else { 8 * 1024 };
            if policy.resolve(needed).is_some() {
                resizes += 1;
            }
        }
        assert_eq!(resizes, 1, "alternating sizes must not reallocate");
    }

    #[test]
    fn vertex_format_shape() {
        // The pipeline's stride comes from this size; the shape attribute
        // sits at offset 20, the clip/extra attribute at 36, and the
        // struct stays tightly packed.
        assert_eq!(std::mem::size_of::<UiVertex>(), 44);
        let _ = Color::WHITE;
    }
}
