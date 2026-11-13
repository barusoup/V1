//! Native Metal GPU layer for UI frames (`UiRenderer`).

#[path = "render/backdrop_metal.rs"]
mod backdrop;
#[path = "render/prepare.rs"]
mod prepare;

use super::draw::{CmdTexture, UiFrame, UiVertex};
use crate::gfx::metal::pipeline::{
    VertexAttr, create_depth_stencil_state, create_render_pipeline, make_vertex_descriptor,
    make_viewport,
};
use crate::gfx::metal::shader::{MslBindings, PUSH_CONSTANT_BUFFER_SLOT, VERTEX_BUFFER_SLOT};
use crate::gfx::metal::{GpuBuffer, GpuContext};
use crate::texture::{self, GpuTexture, SamplerDesc, TextureFormat};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCompareFunction,
    MTLDepthStencilState, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineState, MTLScissorRect,
    MTLSize, MTLStoreAction, MTLTexture, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};
use prepare::{DrawKind, Prepared, UiInstance};
use std::ptr::NonNull;
use std::sync::Arc;

pub use super::draw::AtlasUpdate;

const SHADER: &str = include_str!("ui.wgsl");

struct ResizePolicy {
    capacity: u64,
    floor: u64,
    low_frames: u32,
}

impl ResizePolicy {
    const SHRINK_HEADROOM: u64 = 4;
    const SHRINK_AFTER: u32 = 120;

    fn new(capacity: u64) -> Self {
        let capacity = capacity.max(4);
        Self {
            capacity,
            floor: capacity,
            low_frames: 0,
        }
    }

    fn reset_to_floor(&mut self) {
        self.capacity = self.floor;
        self.low_frames = 0;
    }

    fn resolve(&mut self, needed: u64) -> Option<u64> {
        let needed = needed.max(4);
        if needed > self.capacity {
            self.capacity = needed.next_power_of_two().max(self.floor);
            self.low_frames = 0;
            return Some(self.capacity);
        }
        if needed * Self::SHRINK_HEADROOM < self.capacity && self.capacity > self.floor {
            self.low_frames += 1;
            if self.low_frames >= Self::SHRINK_AFTER {
                let target = (needed * 2).next_power_of_two().max(self.floor);
                if target < self.capacity {
                    self.capacity = target;
                    self.low_frames = 0;
                    return Some(target);
                }
            }
        } else {
            self.low_frames = 0;
        }
        None
    }
}

struct FrameSlot {
    vertices: GpuBuffer,
    indices: GpuBuffer,
    instances: GpuBuffer,
    staging: GpuBuffer,
    vertex_policy: ResizePolicy,
    index_policy: ResizePolicy,
    instance_policy: ResizePolicy,
    staging_policy: ResizePolicy,
    retained_textures: Vec<Arc<GpuTexture>>,
}

struct PageTex {
    texture: Arc<GpuTexture>,
    #[allow(dead_code)]
    written: bool,
}

struct AtlasGpu {
    id: u64,
    epoch: u64,
    revisions: std::collections::HashMap<usize, u64>,
    pages: Vec<Option<PageTex>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiBlend {
    #[default]
    Gamma,
    Linear,
}

pub struct UiRenderer {
    ctx: Arc<GpuContext>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    instance_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    backdrop_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    backdrop_instance_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth_stencil: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    instancing: bool,
    white: Arc<GpuTexture>,
    slots: Vec<FrameSlot>,
    backdrop: Vec<Option<backdrop::Backdrop>>,
    backdrop_format: MTLPixelFormat,
    blur: backdrop::BlurPipelines,
    atlas: Option<AtlasGpu>,
    page_size: u32,
    draws: u64,
    last_vertex_bytes: u64,
    last_index_bytes: u64,
    last_instance_bytes: u64,
    last_instances: u64,
    last_raw_tris: u64,
    last_draws: u64,
}

fn make_instance_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    let desc = MTLVertexDescriptor::vertexDescriptor();
    let layouts = desc.layouts();
    let layout = unsafe { layouts.objectAtIndexedSubscript(VERTEX_BUFFER_SLOT as usize) };
    unsafe {
        layout.setStride(std::mem::size_of::<UiInstance>());
        layout.setStepFunction(MTLVertexStepFunction::PerInstance);
        layout.setStepRate(1);
    }
    let attrs = [
        (0, MTLVertexFormat::Float4, 0),
        (1, MTLVertexFormat::Float4, 16),
        (2, MTLVertexFormat::UChar4Normalized, 32),
        (3, MTLVertexFormat::Float4, 36),
        (4, MTLVertexFormat::Float2, 52),
    ];
    let attributes = desc.attributes();
    for (loc, fmt, off) in attrs {
        let a = unsafe { attributes.objectAtIndexedSubscript(loc) };
        unsafe {
            a.setFormat(fmt);
            a.setOffset(off);
            a.setBufferIndex(VERTEX_BUFFER_SLOT as usize);
        }
    }
    desc
}

impl UiRenderer {
    pub fn new(ctx: &Arc<GpuContext>, format: MTLPixelFormat, page_size: u32) -> Self {
        Self::new_with_blend(ctx, format, page_size, UiBlend::default())
    }

    pub fn new_with_blend(
        ctx: &Arc<GpuContext>,
        format: MTLPixelFormat,
        page_size: u32,
        blend: UiBlend,
    ) -> Self {
        let gamma = blend == UiBlend::Gamma;
        let bindings = MslBindings::new()
            .texture(0, 0, 0)
            .sampler(0, 1, 0)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);

        let raw_vd = make_vertex_descriptor(
            std::mem::size_of::<UiVertex>(),
            &[
                VertexAttr {
                    location: 0,
                    format: MTLVertexFormat::Float2,
                    offset: 0,
                },
                VertexAttr {
                    location: 1,
                    format: MTLVertexFormat::Float2,
                    offset: 8,
                },
                VertexAttr {
                    location: 2,
                    format: MTLVertexFormat::UChar4Normalized,
                    offset: 16,
                },
                VertexAttr {
                    location: 3,
                    format: MTLVertexFormat::Float4,
                    offset: 20,
                },
                VertexAttr {
                    location: 4,
                    format: MTLVertexFormat::Float2,
                    offset: 36,
                },
            ],
        );
        let inst_vd = make_instance_vertex_descriptor();

        let (vs_inst, fs_main, fs_backdrop) = if gamma {
            ("vs_instance_gamma", "fs_main_gamma", "fs_backdrop")
        } else {
            ("vs_instance", "fs_linear", "fs_main")
        };

        let pipeline = create_render_pipeline(
            ctx,
            "ui pipeline",
            SHADER,
            &bindings,
            "vs_main",
            if gamma { "fs_main_gamma" } else { "fs_main" },
            &[format],
            None,
            Some(&raw_vd),
            true,
        );
        let instance_pipeline = create_render_pipeline(
            ctx,
            "ui instanced pipeline",
            SHADER,
            &bindings,
            vs_inst,
            fs_main,
            &[format],
            None,
            Some(&inst_vd),
            true,
        );
        let backdrop_pipeline = if gamma {
            create_render_pipeline(
                ctx,
                "ui backdrop pipeline",
                SHADER,
                &bindings,
                "vs_main",
                "fs_backdrop",
                &[format],
                None,
                Some(&raw_vd),
                true,
            )
        } else {
            pipeline.clone()
        };
        let backdrop_instance_pipeline = if gamma {
            create_render_pipeline(
                ctx,
                "ui instanced backdrop pipeline",
                SHADER,
                &bindings,
                vs_inst,
                fs_backdrop,
                &[format],
                None,
                Some(&inst_vd),
                true,
            )
        } else {
            instance_pipeline.clone()
        };

        let depth_stencil =
            create_depth_stencil_state(ctx, false, false, MTLCompareFunction::Always);
        let white = Arc::new(GpuTexture::white(ctx));
        let blur = backdrop::BlurPipelines::new(ctx, format);

        let frames = ctx.max_frames_in_flight();
        let slots = (0..frames)
            .map(|_| {
                let v_cap = 64 * 1024;
                let i_cap = 32 * 1024;
                let inst_cap = 32 * 1024;
                let st_cap = 64 * 1024;
                FrameSlot {
                    vertices: ctx.create_host_buffer("ui vertices", v_cap),
                    indices: ctx.create_host_buffer("ui indices", i_cap),
                    instances: ctx.create_host_buffer("ui instances", inst_cap),
                    staging: ctx.create_host_buffer("ui atlas staging", st_cap),
                    vertex_policy: ResizePolicy::new(v_cap),
                    index_policy: ResizePolicy::new(i_cap),
                    instance_policy: ResizePolicy::new(inst_cap),
                    staging_policy: ResizePolicy::new(st_cap),
                    retained_textures: Vec::new(),
                }
            })
            .collect();
        let backdrop = (0..frames).map(|_| None).collect();

        Self {
            ctx: Arc::clone(ctx),
            pipeline,
            instance_pipeline,
            backdrop_pipeline,
            backdrop_instance_pipeline,
            depth_stencil,
            instancing: true,
            white,
            slots,
            backdrop,
            backdrop_format: format,
            blur,
            atlas: None,
            page_size: page_size.max(1),
            draws: 0,
            last_vertex_bytes: 0,
            last_index_bytes: 0,
            last_instance_bytes: 0,
            last_instances: 0,
            last_raw_tris: 0,
            last_draws: 0,
        }
    }

    pub fn set_instancing(&mut self, enabled: bool) {
        self.instancing = enabled;
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    pub fn buffer_capacity(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| s.vertices.size + s.indices.size + s.instances.size + s.staging.size)
            .sum()
    }

    pub fn trim(&mut self) {
        for slot in &mut self.slots {
            slot.retained_textures.clear();
            slot.vertex_policy.reset_to_floor();
            slot.index_policy.reset_to_floor();
            slot.instance_policy.reset_to_floor();
            slot.staging_policy.reset_to_floor();
        }
        for b in &mut self.backdrop {
            *b = None;
        }
    }

    pub fn record(
        &mut self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        target: &ProtocolObject<dyn MTLTexture>,
        extent: (u32, u32),
        frame: &UiFrame,
        slot_index: usize,
        clear: bool,
    ) {
        let slot_index = slot_index % self.slots.len();
        self.slots[slot_index].retained_textures.clear();
        self.sync_atlas(cmd, frame, slot_index);

        let mut prepared = Prepared::default();
        prepare::prepare_into(
            &mut prepared,
            &frame.cmds,
            &frame.vertices,
            &frame.indices,
            self.instancing,
        );
        let v_bytes = bytemuck::cast_slice::<UiVertex, u8>(&prepared.vertices);
        let i_bytes = bytemuck::cast_slice::<u32, u8>(&prepared.indices);
        let inst_bytes = bytemuck::cast_slice::<UiInstance, u8>(&prepared.instances);

        self.last_vertex_bytes = v_bytes.len() as u64;
        self.last_index_bytes = i_bytes.len() as u64;
        self.last_instance_bytes = inst_bytes.len() as u64;
        self.last_instances = prepared.instances.len() as u64;
        self.last_raw_tris = (prepared.indices.len() / 3) as u64;
        self.last_draws = prepared.draws.len() as u64;

        {
            let slot = &mut self.slots[slot_index];
            if let Some(cap) = slot.vertex_policy.resolve(v_bytes.len() as u64) {
                slot.vertices = self.ctx.create_host_buffer("ui vertices", cap);
            }
            if let Some(cap) = slot.index_policy.resolve(i_bytes.len() as u64) {
                slot.indices = self.ctx.create_host_buffer("ui indices", cap);
            }
            if let Some(cap) = slot.instance_policy.resolve(inst_bytes.len() as u64) {
                slot.instances = self.ctx.create_host_buffer("ui instances", cap);
            }
            if !v_bytes.is_empty() {
                slot.vertices.write_bytes(0, v_bytes);
            }
            if !i_bytes.is_empty() {
                slot.indices.write_bytes(0, i_bytes);
            }
            if !inst_bytes.is_empty() {
                slot.instances.write_bytes(0, inst_bytes);
            }
        }

        if prepared.draws.is_empty() {
            if clear {
                let rpd = MTLRenderPassDescriptor::new();
                let att = unsafe { rpd.colorAttachments().objectAtIndexedSubscript(0) };
                att.setTexture(Some(target));
                att.setLoadAction(MTLLoadAction::Clear);
                att.setStoreAction(MTLStoreAction::Store);
                let enc = cmd
                    .renderCommandEncoderWithDescriptor(&rpd)
                    .expect("ui clear");
                enc.endEncoding();
            }
            return;
        }

        let screen = [extent.0.max(1) as f32, extent.1.max(1) as f32];
        let scale = (1.0f32, 1.0f32);
        let viewport = make_viewport(extent);
        let full_scissor = MTLScissorRect {
            x: 0,
            y: 0,
            width: extent.0.max(1) as usize,
            height: extent.1.max(1) as usize,
        };
        let dirty_scissor = if clear {
            None
        } else {
            frame.dirty_rect.map(|d| {
                let x0 = ((d.x * scale.0).floor().max(0.0) as usize).min(extent.0 as usize);
                let y0 = ((d.y * scale.1).floor().max(0.0) as usize).min(extent.1 as usize);
                let x1 = (((d.x + d.w) * scale.0).ceil().max(0.0) as usize).min(extent.0 as usize);
                let y1 = (((d.y + d.h) * scale.1).ceil().max(0.0) as usize).min(extent.1 as usize);
                MTLScissorRect {
                    x: x0,
                    y: y0,
                    width: x1.saturating_sub(x0),
                    height: y1.saturating_sub(y0),
                }
            })
        };

        let begin_ui_pass = |load: MTLLoadAction| {
            let rpd = MTLRenderPassDescriptor::new();
            let att = unsafe { rpd.colorAttachments().objectAtIndexedSubscript(0) };
            att.setTexture(Some(target));
            att.setLoadAction(load);
            att.setStoreAction(MTLStoreAction::Store);
            let enc = cmd
                .renderCommandEncoderWithDescriptor(&rpd)
                .expect("ui render encoder");
            enc.setViewport(viewport);
            enc.setDepthStencilState(Some(&self.depth_stencil));
            enc
        };

        let mut enc = begin_ui_pass(if clear {
            MTLLoadAction::Clear
        } else {
            MTLLoadAction::Load
        });
        let mut backdrop_captured = false;
        let mut draw_count = 0u64;

        for draw in &prepared.draws {
            let cmd_info = &frame.cmds[draw.cmd_index];
            let is_backdrop = matches!(cmd_info.texture, CmdTexture::Backdrop);
            if is_backdrop && !backdrop_captured {
                enc.endEncoding();
                let slot_bd = &mut self.backdrop[slot_index];
                let need_new = slot_bd.as_ref().is_none_or(|b| b.size != extent);
                if need_new {
                    *slot_bd = Some(backdrop::Backdrop::new(
                        &self.ctx,
                        self.backdrop_format,
                        extent,
                    ));
                }
                let bd = slot_bd.as_ref().unwrap();
                backdrop::capture_backdrop_pass(&self.ctx, cmd, &self.blur, bd, target);
                backdrop_captured = true;
                enc = begin_ui_pass(MTLLoadAction::Load);
            }

            let tex: &GpuTexture = match &cmd_info.texture {
                CmdTexture::White => &self.white,
                CmdTexture::AtlasPage(idx) => self
                    .atlas
                    .as_ref()
                    .and_then(|a| a.pages.get(*idx))
                    .and_then(|p| p.as_ref())
                    .map(|p| p.texture.as_ref())
                    .unwrap_or(&self.white),
                CmdTexture::Custom(t) => t.as_ref(),
                CmdTexture::Backdrop => self.backdrop[slot_index]
                    .as_ref()
                    .expect("backdrop captured")
                    .out
                    .as_ref(),
            };

            let mut scissor = if let Some(c) = cmd_info.scissor {
                let x0 = ((c.x * scale.0).floor().max(0.0) as usize).min(extent.0 as usize);
                let y0 = ((c.y * scale.1).floor().max(0.0) as usize).min(extent.1 as usize);
                let x1 = (((c.x + c.w) * scale.0).ceil().max(0.0) as usize).min(extent.0 as usize);
                let y1 = (((c.y + c.h) * scale.1).ceil().max(0.0) as usize).min(extent.1 as usize);
                MTLScissorRect {
                    x: x0,
                    y: y0,
                    width: x1.saturating_sub(x0),
                    height: y1.saturating_sub(y0),
                }
            } else {
                full_scissor
            };
            if let Some(d) = dirty_scissor {
                let x0 = scissor.x.max(d.x);
                let y0 = scissor.y.max(d.y);
                let x1 = (scissor.x + scissor.width).min(d.x + d.width);
                let y1 = (scissor.y + scissor.height).min(d.y + d.height);
                scissor = MTLScissorRect {
                    x: x0,
                    y: y0,
                    width: x1.saturating_sub(x0),
                    height: y1.saturating_sub(y0),
                };
            }
            if scissor.width == 0 || scissor.height == 0 {
                continue;
            }
            enc.setScissorRect(scissor);

            let slot = &self.slots[slot_index];
            match draw.kind {
                DrawKind::Indexed {
                    first: first_index,
                    count: index_count,
                } => {
                    let pipe = if is_backdrop {
                        &self.backdrop_pipeline
                    } else {
                        &self.pipeline
                    };
                    enc.setRenderPipelineState(pipe);
                    unsafe {
                        enc.setVertexBuffer_offset_atIndex(
                            Some(&slot.vertices.buffer),
                            0,
                            VERTEX_BUFFER_SLOT as usize,
                        );
                        let push_bytes = bytemuck::bytes_of(&screen);
                        enc.setVertexBytes_length_atIndex(
                            NonNull::new_unchecked(push_bytes.as_ptr() as *mut _),
                            push_bytes.len(),
                            PUSH_CONSTANT_BUFFER_SLOT as usize,
                        );
                        enc.setFragmentTexture_atIndex(Some(tex.texture()), 0);
                        enc.setFragmentSamplerState_atIndex(Some(tex.sampler()), 0);
                        enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            MTLPrimitiveType::Triangle,
                            index_count as usize,
                            MTLIndexType::UInt32,
                            &slot.indices.buffer,
                            first_index as usize * std::mem::size_of::<u32>(),
                        );
                    }
                }
                DrawKind::Instanced {
                    first: first_instance,
                    count: instance_count,
                } => {
                    let pipe = if is_backdrop {
                        &self.backdrop_instance_pipeline
                    } else {
                        &self.instance_pipeline
                    };
                    enc.setRenderPipelineState(pipe);
                    unsafe {
                        enc.setVertexBuffer_offset_atIndex(
                            Some(&slot.instances.buffer),
                            0,
                            VERTEX_BUFFER_SLOT as usize,
                        );
                        let push_bytes = bytemuck::bytes_of(&screen);
                        enc.setVertexBytes_length_atIndex(
                            NonNull::new_unchecked(push_bytes.as_ptr() as *mut _),
                            push_bytes.len(),
                            PUSH_CONSTANT_BUFFER_SLOT as usize,
                        );
                        enc.setFragmentTexture_atIndex(Some(tex.texture()), 0);
                        enc.setFragmentSamplerState_atIndex(Some(tex.sampler()), 0);
                        enc.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                            MTLPrimitiveType::Triangle,
                            0,
                            6,
                            instance_count as usize,
                            first_instance as usize,
                        );
                    }
                }
            }
            draw_count += 1;
        }

        enc.endEncoding();
        self.draws = draw_count;
    }

    fn sync_atlas(
        &mut self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        frame: &UiFrame,
        slot_index: usize,
    ) {
        if let Some(snapshot) = frame.atlas_snapshot.as_ref() {
            let need_reset = self
                .atlas
                .as_ref()
                .is_none_or(|a| a.id != snapshot.atlas_id || a.epoch != snapshot.epoch);
            if need_reset {
                if let Some(old) = self.atlas.take() {
                    for p in old.pages.into_iter().flatten() {
                        self.slots[slot_index].retained_textures.push(p.texture);
                    }
                }
                self.atlas = Some(AtlasGpu {
                    id: snapshot.atlas_id,
                    epoch: snapshot.epoch,
                    revisions: std::collections::HashMap::new(),
                    pages: Vec::new(),
                });
            }
            let atlas = self.atlas.as_mut().unwrap();
            while atlas.pages.len() < snapshot.pages.len() {
                atlas.pages.push(None);
            }

            let mut total_bytes = 0u64;
            let mut plan = Vec::with_capacity(snapshot.pages.len());
            for (index, page) in snapshot.pages.iter().enumerate() {
                let prev_rev = atlas.revisions.get(&index).copied();
                if prev_rev == Some(page.revision) {
                    plan.push(super::atlas::PageUpload::Skip);
                    continue;
                }
                let p = if atlas.pages[index].is_none() || prev_rev.is_none() {
                    super::atlas::PageUpload::Full
                } else if page.last_dirty.is_some() {
                    super::atlas::PageUpload::Dirty
                } else {
                    super::atlas::PageUpload::Full
                };
                let bpp = page.format.bytes_per_texel() as u64;
                let sz = match p {
                    super::atlas::PageUpload::Skip => 0,
                    super::atlas::PageUpload::Full => {
                        snapshot.page_size as u64 * snapshot.page_size as u64 * bpp
                    }
                    super::atlas::PageUpload::Dirty => {
                        let r = page.last_dirty.unwrap();
                        r.width as u64 * r.height as u64 * bpp
                    }
                };
                total_bytes += (sz + 3) & !3;
                plan.push(p);
            }
            if total_bytes == 0 {
                return;
            }
            let slot = &mut self.slots[slot_index];
            if let Some(cap) = slot.staging_policy.resolve(total_bytes) {
                slot.staging = self.ctx.create_host_buffer("ui atlas staging", cap);
            }
            let blit = cmd.blitCommandEncoder().expect("atlas blit encoder");
            let mut offset = 0u64;
            for (index, page) in snapshot.pages.iter().enumerate() {
                let (rect, bpp) = match plan[index] {
                    super::atlas::PageUpload::Skip => continue,
                    super::atlas::PageUpload::Full => (
                        super::atlas::DirtyRect {
                            page: index,
                            x: 0,
                            y: 0,
                            width: snapshot.page_size,
                            height: snapshot.page_size,
                        },
                        page.format.bytes_per_texel(),
                    ),
                    super::atlas::PageUpload::Dirty => {
                        (page.last_dirty.unwrap(), page.format.bytes_per_texel())
                    }
                };
                let atlas_page = atlas.pages[index].get_or_insert_with(|| PageTex {
                    texture: Arc::new(GpuTexture::create_uninitialized(
                        &self.ctx,
                        "ui atlas page",
                        (snapshot.page_size, snapshot.page_size),
                        match page.format {
                            super::atlas::AtlasFormat::Coverage => MTLPixelFormat::R8Unorm,
                            super::atlas::AtlasFormat::RgbaLinear => MTLPixelFormat::RGBA8Unorm,
                            super::atlas::AtlasFormat::RgbaSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
                        },
                        match page.format {
                            super::atlas::AtlasFormat::Coverage => {
                                Some(GpuTexture::coverage_swizzle())
                            }
                            _ => None,
                        },
                        SamplerDesc {
                            filter: texture::Filter::Linear,
                            mip: texture::MipFilter::Nearest,
                            address: texture::AddressMode::ClampToEdge,
                            anisotropy: false,
                        },
                        TextureFormat::Linear,
                    )),
                    written: true,
                });
                let row_bytes = rect.width as usize * bpp;
                let total_rect_bytes = row_bytes * rect.height as usize;
                let mut tmp = vec![0u8; total_rect_bytes];
                let stride = snapshot.page_size as usize * bpp;
                for row in 0..rect.height as usize {
                    let src_off = (rect.y as usize + row) * stride + rect.x as usize * bpp;
                    let dst_off = row * row_bytes;
                    tmp[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&page.pixels[src_off..src_off + row_bytes]);
                }
                slot.staging.write_bytes(offset, &tmp);
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &slot.staging.buffer,
                        offset as usize,
                        row_bytes,
                        total_rect_bytes,
                        MTLSize {
                            width: rect.width.max(1) as usize,
                            height: rect.height.max(1) as usize,
                            depth: 1,
                        },
                        atlas_page.texture.raw_texture(),
                        0,
                        0,
                        MTLOrigin {
                            x: rect.x as usize,
                            y: rect.y as usize,
                            z: 0,
                        },
                    );
                }
                offset += (total_rect_bytes as u64 + 3) & !3;
                atlas.revisions.insert(index, page.revision);
            }
            blit.endEncoding();
            return;
        }

        if frame.atlas_updates.is_empty() {
            return;
        }
        let total: u64 = frame
            .atlas_updates
            .iter()
            .map(|u| (u.pixels.len() as u64 + 3) & !3)
            .sum();
        let slot = &mut self.slots[slot_index];
        if let Some(cap) = slot.staging_policy.resolve(total.max(4)) {
            slot.staging = self.ctx.create_host_buffer("ui atlas staging", cap);
        }
        if self.atlas.as_ref().map(|a| (a.id, a.epoch)) != Some((0, 0)) {
            self.atlas = Some(AtlasGpu {
                id: 0,
                epoch: 0,
                revisions: std::collections::HashMap::new(),
                pages: Vec::new(),
            });
        }
        let atlas = self.atlas.as_mut().unwrap();
        let blit = cmd.blitCommandEncoder().expect("legacy atlas blit encoder");
        let mut offset = 0u64;
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
                    MTLPixelFormat::RGBA8Unorm,
                    None,
                    SamplerDesc {
                        filter: texture::Filter::Linear,
                        mip: texture::MipFilter::Nearest,
                        address: texture::AddressMode::ClampToEdge,
                        anisotropy: false,
                    },
                    TextureFormat::Linear,
                )),
                written: true,
            });
            slot.staging.write_bytes(offset, &update.pixels);
            let row_bytes = update.rect.width as usize * 4;
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &slot.staging.buffer,
                    offset as usize,
                    row_bytes,
                    update.pixels.len(),
                    MTLSize {
                        width: update.rect.width.max(1) as usize,
                        height: update.rect.height.max(1) as usize,
                        depth: 1,
                    },
                    atlas_page.texture.raw_texture(),
                    0,
                    0,
                    MTLOrigin {
                        x: update.rect.x as usize,
                        y: update.rect.y as usize,
                        z: 0,
                    },
                );
            }
            offset += (update.pixels.len() as u64 + 3) & !3;
        }
        blit.endEncoding();
    }

    pub fn allocation_sizes(&self) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            out.push((format!("ui slot {i} vertices"), slot.vertices.size));
            out.push((format!("ui slot {i} indices"), slot.indices.size));
            out.push((format!("ui slot {i} instances"), slot.instances.size));
            out.push((format!("ui slot {i} staging"), slot.staging.size));
        }
        if let Some(atlas) = self.atlas.as_ref() {
            for (i, page) in atlas.pages.iter().enumerate() {
                if let Some(p) = page {
                    out.push((
                        format!("ui atlas page {i}"),
                        p.texture.image_allocation_size(),
                    ));
                }
            }
        }
        out
    }

    pub fn last_geometry_upload_sizes(&self) -> (u64, u64, u64) {
        (
            self.last_vertex_bytes,
            self.last_index_bytes,
            self.last_instance_bytes,
        )
    }

    pub fn last_prepared_counts(&self) -> (u64, u64, u64) {
        (self.last_instances, self.last_raw_tris, self.last_draws)
    }
}

pub fn srgb_to_linear(value: u8) -> f32 {
    let s = value as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}
