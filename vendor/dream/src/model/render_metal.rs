//! Native Metal GPU rendering for [`Model`]s: upload, material textures, draw recording.

use super::{AlphaMode, Material, Model, ModelInstance, TexelFilter, TextureRef, Wrap};
use crate::gfx::DrawCtx;
use crate::gfx::metal::pipeline::{
    VertexAttr, create_depth_stencil_state, create_render_pipeline, make_vertex_descriptor,
};
use crate::gfx::metal::shader::{MslBindings, PUSH_CONSTANT_BUFFER_SLOT, VERTEX_BUFFER_SLOT};
use crate::gfx::metal::{GpuBuffer, GpuContext, align_up};
use crate::lighting::Lighting;
use crate::texture::{self, GpuTexture, TextureFormat, TextureOptions};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCompareFunction, MTLCullMode, MTLDepthStencilState, MTLIndexType, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPipelineState, MTLVertexFormat, MTLWinding,
};
use std::ptr::NonNull;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub tangent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawHeader {
    base_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct JointMatrices {
    curr: [[f32; 4]; 4],
    prev: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialFactors {
    params: [f32; 4],
    params2: [f32; 4],
    emissive: [f32; 4],
    uv: [[f32; 4]; 5],
    tf: [[f32; 4]; 5],
}

pub(crate) fn pack_transform(map: Option<&TextureRef>) -> ([f32; 4], [f32; 4]) {
    let Some(map) = map else {
        return ([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    };
    let uv_set = map.uv_set() as f32;
    let Some(t) = map.transform else {
        return ([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, uv_set, 0.0]);
    };
    let (s, c) = t.rotation.sin_cos();
    (
        [t.scale[0], t.scale[1], t.offset[0], t.offset[1]],
        [c, s, uv_set, 0.0],
    )
}

pub struct MeshGpu {
    vertices: GpuBuffer,
    indices: GpuBuffer,
    index_count: u32,
    material: usize,
    skinned: bool,
    node_transform: glam::Mat4,
    pbr_vertices: Option<GpuBuffer>,
}

struct PbrMaterialGpu {
    factors: GpuBuffer,
    base: Arc<GpuTexture>,
    mr: Arc<GpuTexture>,
    normal: Arc<GpuTexture>,
    ao: Arc<GpuTexture>,
    emissive: Arc<GpuTexture>,
}

pub struct ModelGpu {
    ctx: Arc<GpuContext>,
    model: Arc<Model>,
    meshes: Vec<MeshGpu>,
    textures: Vec<Arc<GpuTexture>>,
    pbr_materials: Vec<PbrMaterialGpu>,
}

impl ModelGpu {
    pub fn upload(ctx: &Arc<GpuContext>, model: &Arc<Model>) -> Self {
        let mut meshes = Vec::with_capacity(model.meshes.len());
        for (i, mesh) in model.meshes.iter().enumerate() {
            let skinned = !mesh.joints.is_empty();
            let vertices = if skinned {
                let verts: Vec<SkinnedVertex> = (0..mesh.positions.len())
                    .map(|i| SkinnedVertex {
                        position: mesh.positions[i],
                        normal: mesh.normals[i],
                        uv: mesh.uvs[i],
                        joints: mesh.joints[i],
                        weights: mesh.weights[i],
                    })
                    .collect();
                ctx.upload_buffer(
                    &format!("model skinned {}", mesh.name),
                    bytemuck::cast_slice(&verts),
                )
            } else {
                let verts: Vec<ModelVertex> = (0..mesh.positions.len())
                    .map(|i| ModelVertex {
                        position: mesh.positions[i],
                        normal: mesh.normals[i],
                        uv: mesh.uvs[i],
                    })
                    .collect();
                ctx.upload_buffer(
                    &format!("model mesh {}", mesh.name),
                    bytemuck::cast_slice(&verts),
                )
            };
            let indices = ctx.upload_buffer(
                &format!("model indices {}", mesh.name),
                bytemuck::cast_slice(&mesh.indices),
            );
            meshes.push(MeshGpu {
                vertices,
                indices,
                index_count: mesh.indices.len() as u32,
                material: mesh.material,
                skinned,
                node_transform: model.mesh_node_transform(i),
                pbr_vertices: None,
            });
        }

        let textures = model
            .materials
            .iter()
            .map(|mat| upload_material_texture(ctx, mat.base_color_texture.as_ref()))
            .collect();

        Self {
            ctx: Arc::clone(ctx),
            model: Arc::clone(model),
            meshes,
            textures,
            pbr_materials: Vec::new(),
        }
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    fn ensure_pbr(&mut self) {
        if self.pbr_materials.is_empty() && !self.model.materials.is_empty() {
            let ctx = &self.ctx;
            self.pbr_materials = self
                .model
                .materials
                .iter()
                .map(|m| build_pbr_material(ctx, m))
                .collect();
        }
        for (i, mesh_gpu) in self.meshes.iter_mut().enumerate() {
            if !mesh_gpu.skinned && mesh_gpu.pbr_vertices.is_none() {
                let mesh = &self.model.meshes[i];
                let verts: Vec<PbrVertex> = (0..mesh.positions.len())
                    .map(|vi| PbrVertex {
                        position: mesh.positions[vi],
                        normal: mesh.normals[vi],
                        uv0: mesh.uvs[vi],
                        uv1: mesh.uvs1.get(vi).copied().unwrap_or(mesh.uvs[vi]),
                        tangent: mesh.tangents.get(vi).copied().unwrap_or([0.0; 4]),
                    })
                    .collect();
                mesh_gpu.pbr_vertices = Some(self.ctx.upload_buffer(
                    &format!("model pbr {}", mesh.name),
                    bytemuck::cast_slice(&verts),
                ));
            }
        }
    }
}

fn upload_material_texture(
    ctx: &Arc<GpuContext>,
    tex: Option<&texture::DecodedTexture>,
) -> Arc<GpuTexture> {
    match tex {
        Some(decoded) => Arc::new(GpuTexture::from_rgba8(
            ctx,
            "model baseColor",
            &decoded.pixels,
            decoded.width,
            decoded.height,
            TextureOptions {
                format: TextureFormat::Srgb,
                sampler: texture::SamplerDesc::ANISOTROPIC,
                mip_levels: None,
            },
        )),
        None => Arc::new(GpuTexture::white(ctx)),
    }
}

fn map_texture(
    ctx: &Arc<GpuContext>,
    label: &str,
    map: Option<&TextureRef>,
    format: TextureFormat,
    fallback: [u8; 4],
) -> Arc<GpuTexture> {
    match map {
        Some(m) => Arc::new(GpuTexture::from_rgba8(
            ctx,
            label,
            &m.texture.pixels,
            m.texture.width,
            m.texture.height,
            TextureOptions {
                format,
                sampler: texture::SamplerDesc {
                    filter: match m.mag_filter {
                        TexelFilter::Nearest => texture::Filter::Nearest,
                        TexelFilter::Linear => texture::Filter::Linear,
                    },
                    mip: texture::MipFilter::Linear,
                    address: match m.wrap_s {
                        Wrap::Repeat => texture::AddressMode::Repeat,
                        Wrap::ClampToEdge => texture::AddressMode::ClampToEdge,
                        Wrap::MirroredRepeat => texture::AddressMode::MirrorRepeat,
                    },
                    anisotropy: true,
                },
                mip_levels: None,
            },
        )),
        None => Arc::new(GpuTexture::from_rgba8(
            ctx,
            label,
            &fallback,
            1,
            1,
            TextureOptions {
                format,
                sampler: texture::SamplerDesc::ANISOTROPIC,
                mip_levels: Some(1),
            },
        )),
    }
}

fn build_pbr_material(ctx: &Arc<GpuContext>, mat: &Material) -> PbrMaterialGpu {
    let base = map_texture(
        ctx,
        "pbr base",
        mat.base_color_map.as_ref(),
        TextureFormat::Srgb,
        [255, 255, 255, 255],
    );
    let mr = map_texture(
        ctx,
        "pbr mr",
        mat.metallic_roughness_texture.as_ref(),
        TextureFormat::Linear,
        [255, 255, 255, 255],
    );
    let normal = map_texture(
        ctx,
        "pbr normal",
        mat.normal_texture.as_ref(),
        TextureFormat::Linear,
        [128, 128, 255, 255],
    );
    let ao = map_texture(
        ctx,
        "pbr ao",
        mat.occlusion_texture.as_ref(),
        TextureFormat::Linear,
        [255, 255, 255, 255],
    );
    let emissive = map_texture(
        ctx,
        "pbr emissive",
        mat.emissive_texture.as_ref(),
        TextureFormat::Srgb,
        [0, 0, 0, 255],
    );
    let alpha_mode = match mat.alpha_mode {
        AlphaMode::Opaque => 0.0,
        AlphaMode::Mask => 1.0,
        AlphaMode::Blend => 2.0,
    };
    let tfs = [
        pack_transform(mat.base_color_map.as_ref()),
        pack_transform(mat.metallic_roughness_texture.as_ref()),
        pack_transform(mat.normal_texture.as_ref()),
        pack_transform(mat.occlusion_texture.as_ref()),
        pack_transform(mat.emissive_texture.as_ref()),
    ];
    let factors = MaterialFactors {
        params: [
            mat.metallic_factor,
            mat.roughness_factor,
            mat.alpha_cutoff,
            alpha_mode,
        ],
        params2: [
            mat.occlusion_strength,
            mat.normal_scale,
            mat.base_color[3],
            0.0,
        ],
        emissive: [
            mat.emissive_factor[0],
            mat.emissive_factor[1],
            mat.emissive_factor[2],
            mat.emissive_strength,
        ],
        uv: [tfs[0].0, tfs[1].0, tfs[2].0, tfs[3].0, tfs[4].0],
        tf: [tfs[0].1, tfs[1].1, tfs[2].1, tfs[3].1, tfs[4].1],
    };
    let factors_buf = ctx.upload_buffer("pbr material factors", bytemuck::bytes_of(&factors));
    PbrMaterialGpu {
        factors: factors_buf,
        base,
        mr,
        normal,
        ao,
        emissive,
    }
}

struct JointRing {
    buffer: GpuBuffer,
    capacity: u64,
    used: u64,
    last_frame_id: u64,
    retired: Vec<GpuBuffer>,
}

struct PbrState {
    opaque: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    blended: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl PbrState {
    fn new(
        ctx: &GpuContext,
        color_formats: &[MTLPixelFormat],
        depth_format: MTLPixelFormat,
    ) -> Self {
        let src = format!(
            "{}\n{}",
            crate::lighting::SHADER,
            include_str!("model_pbr.wgsl")
        );
        let bindings = MslBindings::new()
            .buffer(0, 0, 0, false)
            .buffer(1, 0, 1, false)
            .buffer(2, 0, 2, false)
            .buffer(3, 0, 3, false)
            .texture(2, 1, 0)
            .sampler(2, 2, 0)
            .texture(2, 3, 1)
            .sampler(2, 4, 1)
            .texture(2, 5, 2)
            .sampler(2, 6, 2)
            .texture(2, 7, 3)
            .sampler(2, 8, 3)
            .texture(2, 9, 4)
            .sampler(2, 10, 4)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);

        let vd = make_vertex_descriptor(
            std::mem::size_of::<PbrVertex>(),
            &[
                VertexAttr {
                    location: 0,
                    format: MTLVertexFormat::Float3,
                    offset: 0,
                },
                VertexAttr {
                    location: 1,
                    format: MTLVertexFormat::Float3,
                    offset: 12,
                },
                VertexAttr {
                    location: 2,
                    format: MTLVertexFormat::Float2,
                    offset: 24,
                },
                VertexAttr {
                    location: 3,
                    format: MTLVertexFormat::Float2,
                    offset: 32,
                },
                VertexAttr {
                    location: 4,
                    format: MTLVertexFormat::Float4,
                    offset: 40,
                },
            ],
        );

        let opaque = create_render_pipeline(
            ctx,
            "model pbr opaque",
            &src,
            &bindings,
            "vs_main",
            "fs_main",
            color_formats,
            Some(depth_format),
            Some(&vd),
            false,
        );
        let blended = create_render_pipeline(
            ctx,
            "model pbr blended",
            &src,
            &bindings,
            "vs_main",
            "fs_main",
            color_formats,
            Some(depth_format),
            Some(&vd),
            true,
        );
        Self { opaque, blended }
    }
}

pub struct ModelRenderer {
    ctx: Arc<GpuContext>,
    static_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    skinned_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    static_blended: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    skinned_blended: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth_opaque: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    depth_blended: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    color_formats: Vec<MTLPixelFormat>,
    depth_format: MTLPixelFormat,
    pbr: Option<PbrState>,
    rings: Vec<JointRing>,
    draws: u64,
    pub debug_channel: u32,
}

impl ModelRenderer {
    pub fn new(
        ctx: &Arc<GpuContext>,
        color_formats: &[MTLPixelFormat],
        depth_format: MTLPixelFormat,
    ) -> Self {
        let src = include_str!("model.wgsl");
        let bindings = MslBindings::new()
            .buffer(0, 0, 0, false)
            .buffer(1, 0, 1, false)
            .texture(2, 0, 0)
            .sampler(2, 1, 0)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);

        let static_vd = make_vertex_descriptor(
            std::mem::size_of::<ModelVertex>(),
            &[
                VertexAttr {
                    location: 0,
                    format: MTLVertexFormat::Float3,
                    offset: 0,
                },
                VertexAttr {
                    location: 1,
                    format: MTLVertexFormat::Float3,
                    offset: 12,
                },
                VertexAttr {
                    location: 2,
                    format: MTLVertexFormat::Float2,
                    offset: 24,
                },
            ],
        );
        let skinned_vd = make_vertex_descriptor(
            std::mem::size_of::<SkinnedVertex>(),
            &[
                VertexAttr {
                    location: 0,
                    format: MTLVertexFormat::Float3,
                    offset: 0,
                },
                VertexAttr {
                    location: 1,
                    format: MTLVertexFormat::Float3,
                    offset: 12,
                },
                VertexAttr {
                    location: 2,
                    format: MTLVertexFormat::Float2,
                    offset: 24,
                },
                VertexAttr {
                    location: 3,
                    format: MTLVertexFormat::UShort4,
                    offset: 32,
                },
                VertexAttr {
                    location: 4,
                    format: MTLVertexFormat::Float4,
                    offset: 40,
                },
            ],
        );

        let static_pipeline = create_render_pipeline(
            ctx,
            "model static",
            src,
            &bindings,
            "vs_main",
            "fs_main",
            color_formats,
            Some(depth_format),
            Some(&static_vd),
            false,
        );
        let skinned_pipeline = create_render_pipeline(
            ctx,
            "model skinned",
            src,
            &bindings,
            "vs_skin",
            "fs_main",
            color_formats,
            Some(depth_format),
            Some(&skinned_vd),
            false,
        );
        let static_blended = create_render_pipeline(
            ctx,
            "model static blended",
            src,
            &bindings,
            "vs_main",
            "fs_main",
            color_formats,
            Some(depth_format),
            Some(&static_vd),
            true,
        );
        let skinned_blended = create_render_pipeline(
            ctx,
            "model skinned blended",
            src,
            &bindings,
            "vs_skin",
            "fs_main",
            color_formats,
            Some(depth_format),
            Some(&skinned_vd),
            true,
        );

        let depth_opaque = create_depth_stencil_state(ctx, true, true, MTLCompareFunction::Less);
        let depth_blended =
            create_depth_stencil_state(ctx, true, false, MTLCompareFunction::LessEqual);

        let initial_cap = 64 * 1024;
        let rings = (0..ctx.max_frames_in_flight())
            .map(|_| JointRing {
                buffer: ctx.create_host_buffer("model joint ring", initial_cap),
                capacity: initial_cap,
                used: 0,
                last_frame_id: u64::MAX,
                retired: Vec::new(),
            })
            .collect();

        Self {
            ctx: Arc::clone(ctx),
            static_pipeline,
            skinned_pipeline,
            static_blended,
            skinned_blended,
            depth_opaque,
            depth_blended,
            color_formats: color_formats.to_vec(),
            depth_format,
            pbr: None,
            rings,
            draws: 0,
            debug_channel: 0,
        }
    }

    fn recycle_slot(&mut self, ring_index: usize, frame_id: u64) {
        let ring = &mut self.rings[ring_index];
        if ring.last_frame_id != frame_id {
            ring.used = 0;
            ring.last_frame_id = frame_id;
            ring.retired.clear();
        }
    }

    pub fn draw(&mut self, ctx: &mut DrawCtx<'_>, model: &mut ModelGpu, instance: &ModelInstance) {
        self.draw_with_opacity(ctx, model, instance, 1.0);
    }

    pub fn draw_with_opacity(
        &mut self,
        ctx: &mut DrawCtx<'_>,
        model: &mut ModelGpu,
        instance: &ModelInstance,
        opacity: f32,
    ) {
        let opacity = if opacity.is_finite() {
            opacity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if opacity <= 0.0 {
            return;
        }
        let ring_index = ctx.frame_slot() % self.rings.len();
        self.recycle_slot(ring_index, ctx.frame_id());
        let (model_matrix, prev_model) = instance.motion_matrices();
        let blended = opacity < 1.0;

        for mesh_index in 0..model.meshes.len() {
            let mesh = &model.meshes[mesh_index];
            let material_index = mesh.material;
            let (joint_curr, joint_prev) = if mesh.skinned {
                instance.skinning_matrices(0)
            } else {
                (Vec::new(), Vec::new())
            };
            let (model_mat, prev_mat) = if mesh.skinned {
                (model_matrix, prev_model)
            } else {
                (
                    model_matrix * mesh.node_transform,
                    prev_model * mesh.node_transform,
                )
            };
            let mut push = [0u8; 128];
            push[..64].copy_from_slice(bytemuck::bytes_of(&model_mat.to_cols_array_2d()));
            push[64..].copy_from_slice(bytemuck::bytes_of(&prev_mat.to_cols_array_2d()));

            let mut base_color = model
                .model()
                .materials
                .get(material_index)
                .map(|m| m.base_color)
                .unwrap_or([1.0; 4]);
            base_color[3] *= opacity;

            let joints: Vec<JointMatrices> = joint_curr
                .iter()
                .zip(joint_prev.iter())
                .map(|(curr, prev)| JointMatrices {
                    curr: curr.to_cols_array_2d(),
                    prev: prev.to_cols_array_2d(),
                })
                .collect();
            let block_size = std::mem::size_of::<DrawHeader>()
                + joints.len() * std::mem::size_of::<JointMatrices>();
            let offset = {
                let ring = &mut self.rings[ring_index];
                let offset = align_up(ring.used, 256);
                if offset + block_size as u64 > ring.capacity {
                    let grown = self
                        .ctx
                        .create_host_buffer("model joint ring", (offset + block_size as u64) * 2);
                    ring.retired
                        .push(std::mem::replace(&mut ring.buffer, grown));
                    ring.capacity = (offset + block_size as u64) * 2;
                }
                ring.buffer
                    .write_bytes(offset, bytemuck::bytes_of(&DrawHeader { base_color }));
                if !joints.is_empty() {
                    ring.buffer.write_bytes(
                        offset + std::mem::size_of::<DrawHeader>() as u64,
                        bytemuck::cast_slice(&joints),
                    );
                }
                ring.used = offset + block_size as u64;
                offset
            };

            let pipe = match (mesh.skinned, blended) {
                (false, false) => &self.static_pipeline,
                (true, false) => &self.skinned_pipeline,
                (false, true) => &self.static_blended,
                (true, true) => &self.skinned_blended,
            };
            let tex = &model.textures[material_index];
            let cam_buf = ctx.scene_uniform_buffer().as_ptr();
            let enc = ctx.raw();
            enc.setRenderPipelineState(pipe);
            enc.setDepthStencilState(Some(if blended {
                &self.depth_blended
            } else {
                &self.depth_opaque
            }));
            enc.setFrontFacingWinding(MTLWinding::Clockwise);
            enc.setCullMode(MTLCullMode::Back);
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(&*cam_buf), 0, 0);
                enc.setFragmentBuffer_offset_atIndex(Some(&*cam_buf), 0, 0);
                enc.setVertexBuffer_offset_atIndex(
                    Some(&self.rings[ring_index].buffer.buffer),
                    offset as usize,
                    1,
                );
                enc.setFragmentBuffer_offset_atIndex(
                    Some(&self.rings[ring_index].buffer.buffer),
                    offset as usize,
                    1,
                );
                enc.setVertexBuffer_offset_atIndex(
                    Some(&mesh.vertices.buffer),
                    0,
                    VERTEX_BUFFER_SLOT as usize,
                );
                enc.setVertexBytes_length_atIndex(
                    NonNull::new_unchecked(push.as_ptr() as *mut _),
                    push.len(),
                    PUSH_CONSTANT_BUFFER_SLOT as usize,
                );
                enc.setFragmentTexture_atIndex(Some(tex.texture()), 0);
                enc.setFragmentSamplerState_atIndex(Some(tex.sampler()), 0);
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    mesh.index_count as usize,
                    MTLIndexType::UInt32,
                    &mesh.indices.buffer,
                    0,
                );
            }
            self.draws += 1;
        }
    }

    pub fn draw_pbr(
        &mut self,
        ctx: &mut DrawCtx<'_>,
        model: &mut ModelGpu,
        instance: &ModelInstance,
        lighting: &Lighting,
    ) {
        self.draw_pbr_with_opacity(ctx, model, instance, lighting, 1.0);
    }

    pub fn draw_pbr_with_opacity(
        &mut self,
        ctx: &mut DrawCtx<'_>,
        model: &mut ModelGpu,
        instance: &ModelInstance,
        lighting: &Lighting,
        opacity: f32,
    ) {
        let opacity = if opacity.is_finite() {
            opacity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if opacity <= 0.0 {
            return;
        }
        model.ensure_pbr();
        if self.pbr.is_none() {
            self.pbr = Some(PbrState::new(
                &self.ctx,
                &self.color_formats,
                self.depth_format,
            ));
        }
        let frame_slot = ctx.frame_slot();
        let ring_index = frame_slot % self.rings.len();
        self.recycle_slot(ring_index, ctx.frame_id());
        let (model_matrix, prev_model) = instance.motion_matrices();
        let lighting_buf = lighting.buffer(frame_slot);

        for mesh_index in 0..model.meshes.len() {
            let mesh = &model.meshes[mesh_index];
            if mesh.skinned {
                continue;
            }
            let Some(pbr_vb) = mesh.pbr_vertices.as_ref() else {
                continue;
            };
            let model_mat = model_matrix * mesh.node_transform;
            let prev_mat = prev_model * mesh.node_transform;
            let mut push = [0u8; 144];
            push[..64].copy_from_slice(bytemuck::bytes_of(&model_mat.to_cols_array_2d()));
            push[64..128].copy_from_slice(bytemuck::bytes_of(&prev_mat.to_cols_array_2d()));
            push[128..132].copy_from_slice(&(self.debug_channel as f32).to_le_bytes());

            let material_index = mesh.material;
            let material = &model.model().materials[material_index];
            let blended = opacity < 1.0 || material.alpha_mode == AlphaMode::Blend;
            let mut base_color = material.base_color;
            base_color[3] = opacity;

            let offset = {
                let ring = &mut self.rings[ring_index];
                let block_size = std::mem::size_of::<DrawHeader>() as u64;
                let required_size = block_size.max(144);
                let offset = align_up(ring.used, 256);
                if offset + required_size > ring.capacity {
                    let grown = self
                        .ctx
                        .create_host_buffer("model joint ring", (offset + required_size) * 2);
                    ring.retired
                        .push(std::mem::replace(&mut ring.buffer, grown));
                    ring.capacity = (offset + required_size) * 2;
                }
                ring.buffer
                    .write_bytes(offset, bytemuck::bytes_of(&DrawHeader { base_color }));
                ring.used = offset + block_size;
                offset
            };

            let pbr_state = self.pbr.as_ref().unwrap();
            let pipe = if blended {
                &pbr_state.blended
            } else {
                &pbr_state.opaque
            };
            let mat_gpu = &model.pbr_materials[material_index];
            let cam_buf = ctx.scene_uniform_buffer().as_ptr();
            let enc = ctx.raw();
            enc.setRenderPipelineState(pipe);
            enc.setDepthStencilState(Some(if blended {
                &self.depth_blended
            } else {
                &self.depth_opaque
            }));
            enc.setFrontFacingWinding(MTLWinding::Clockwise);
            enc.setCullMode(if material.double_sided {
                MTLCullMode::None
            } else {
                MTLCullMode::Back
            });
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(&*cam_buf), 0, 0);
                enc.setFragmentBuffer_offset_atIndex(Some(&*cam_buf), 0, 0);
                enc.setVertexBuffer_offset_atIndex(
                    Some(&self.rings[ring_index].buffer.buffer),
                    offset as usize,
                    1,
                );
                enc.setFragmentBuffer_offset_atIndex(
                    Some(&self.rings[ring_index].buffer.buffer),
                    offset as usize,
                    1,
                );
                enc.setFragmentBuffer_offset_atIndex(Some(&mat_gpu.factors.buffer), 0, 2);
                enc.setFragmentBuffer_offset_atIndex(Some(lighting_buf), 0, 3);
                enc.setVertexBuffer_offset_atIndex(
                    Some(&pbr_vb.buffer),
                    0,
                    VERTEX_BUFFER_SLOT as usize,
                );
                enc.setVertexBytes_length_atIndex(
                    NonNull::new_unchecked(push.as_ptr() as *mut _),
                    push.len(),
                    PUSH_CONSTANT_BUFFER_SLOT as usize,
                );
                enc.setFragmentBytes_length_atIndex(
                    NonNull::new_unchecked(push.as_ptr() as *mut _),
                    push.len(),
                    PUSH_CONSTANT_BUFFER_SLOT as usize,
                );
                let maps = [
                    &mat_gpu.base,
                    &mat_gpu.mr,
                    &mat_gpu.normal,
                    &mat_gpu.ao,
                    &mat_gpu.emissive,
                ];
                for (i, tex) in maps.iter().enumerate() {
                    enc.setFragmentTexture_atIndex(Some(tex.texture()), i);
                    enc.setFragmentSamplerState_atIndex(Some(tex.sampler()), i);
                }
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    mesh.index_count as usize,
                    MTLIndexType::UInt32,
                    &mesh.indices.buffer,
                    0,
                );
            }
            self.draws += 1;
        }
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }
}

trait ProtocolPtr {
    fn as_ptr(&self) -> *const ProtocolObject<dyn objc2_metal::MTLBuffer>;
}
impl ProtocolPtr for ProtocolObject<dyn objc2_metal::MTLBuffer> {
    fn as_ptr(&self) -> *const ProtocolObject<dyn objc2_metal::MTLBuffer> {
        self as *const _
    }
}
