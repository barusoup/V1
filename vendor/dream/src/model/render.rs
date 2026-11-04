//! GPU rendering for [`Model`]s: upload, material textures, draw recording.
//!
//! [`ModelGpu`] owns the uploaded buffers for one model; [`ModelRenderer`]
//! owns the pipelines and the per-frame joint rings and records draws inside
//! the game's scene pass through [`crate::gfx::DrawCtx`]. Set 0 is the
//! engine's camera block (shared with every scene pipeline), set 1 is the
//! per-draw joint/material block, set 2 is the material texture.
//!
//! Motion vectors: the shader writes per-object velocity from the previous
//! model matrix and previous joint palette carried by
//! [`crate::model::ModelInstance`], matching the FSR texture-space contract.

use super::{Material, Model, ModelInstance};

mod material;
mod pipeline;

use crate::gfx::DrawCtx;
use crate::gfx::vulkan::{GpuContext, align_up};
use crate::texture::{self, GpuTexture, TextureFormat, TextureOptions};
use ash::vk;
pub(crate) use material::{pack_transform, pbr_vertex_buffer};
use pipeline::create_model_pipeline;
use std::sync::Arc;

/// Interleaved vertex for static meshes (pos, normal, uv).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Interleaved vertex for skinned meshes (adds joints + weights).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

/// Interleaved vertex for the PBR path (adds uv1 + tangent).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    /// xyz tangent, w bitangent sign; zero vector when the file has none.
    pub tangent: [f32; 4],
}

/// Per-draw block written to the joint ring: material colour then the joint
/// palette. Layout must match `model.wgsl`'s `DrawData`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawHeader {
    base_color: [f32; 4],
}

/// Per-material uniform block, layout must match `model_pbr.wgsl`'s
/// `MatFactors`: factors then, per map (base, mr, normal, ao, emissive),
/// transform (scale.xy, offset.xy) and (cos rot, sin rot, uv set, _).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialFactors {
    /// x metallic, y roughness, z alpha cutoff, w alpha mode (0/1/2).
    params: [f32; 4],
    /// x AO strength, y normal scale.
    params2: [f32; 4],
    /// rgb emissive factor, a emissive strength.
    emissive: [f32; 4],
    uv: [[f32; 4]; 5],
    tf: [[f32; 4]; 5],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct JointMatrices {
    curr: [[f32; 4]; 4],
    prev: [[f32; 4]; 4],
}

/// One uploaded mesh primitive.
pub struct MeshGpu {
    vertices: crate::gfx::GpuBuffer,
    indices: crate::gfx::GpuBuffer,
    index_count: u32,
    material: usize,
    skinned: bool,
    /// Rest-pose node global — static meshes are drawn at
    /// `model * node_transform` (glTF node transforms apply to static
    /// geometry; skinned meshes transform via the joint palette instead).
    node_transform: glam::Mat4,
    /// PBR vertex stream (uv1 + tangents), built lazily on first PBR draw.
    pbr_vertices: Option<crate::gfx::GpuBuffer>,
}

/// GPU resources for one PBR material: the five maps (falling back to
/// white/flat-normal/black 1x1s), its factor uniform and a persistent
/// descriptor set. Owned by the material, freed with it.
pub(crate) struct PbrMaterialGpu {
    ctx: Arc<GpuContext>,
    pub(crate) _textures: Vec<Arc<GpuTexture>>,
    pub(crate) _factors: crate::gfx::GpuBuffer,
    pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
}

impl Drop for PbrMaterialGpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// GPU-side resources for one [`Model`]: buffers uploaded once, textures
/// created lazily on first draw.
pub struct ModelGpu {
    meshes: Vec<MeshGpu>,
    textures: Vec<Option<Arc<GpuTexture>>>,
    /// PBR material resources, built lazily on first `draw_pbr`.
    pbr_materials: Vec<Option<Arc<PbrMaterialGpu>>>,
    model: Arc<Model>,
}

impl ModelGpu {
    /// Upload mesh data. Textures wait until [`ModelRenderer::draw`], which
    /// needs a live texture manager anyway.
    pub fn upload(ctx: &Arc<GpuContext>, model: &Arc<Model>) -> Self {
        let mut meshes = Vec::with_capacity(model.meshes.len());
        for (mesh_index, mesh) in model.meshes.iter().enumerate() {
            let skinned = mesh.is_skinned();
            let label = format!("model mesh {}", mesh.name);
            // One interleaved buffer, cast once. The per-vertex
            // `bytes_of(..).to_vec()` this replaced allocated a Vec for
            // every vertex in the mesh.
            let vertex_buffer = if skinned {
                let vertices = skinned_vertices(mesh);
                ctx.create_vertex_buffer(&label, bytemuck::cast_slice(&vertices))
            } else {
                let vertices = static_vertices(mesh);
                ctx.create_vertex_buffer(&label, bytemuck::cast_slice(&vertices))
            };
            let index_bytes: Vec<u8> = mesh
                .indices
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect();
            let index_buffer =
                ctx.create_index_buffer(&format!("model indices {}", mesh.name), &index_bytes);
            meshes.push(MeshGpu {
                vertices: vertex_buffer,
                indices: index_buffer,
                index_count: mesh.indices.len() as u32,
                material: mesh.material,
                skinned,
                node_transform: model.mesh_node_transform(mesh_index),
                pbr_vertices: None,
            });
        }
        let textures = (0..model.materials.len()).map(|_| None).collect();
        let pbr_materials = (0..model.materials.len()).map(|_| None).collect();
        Self {
            meshes,
            textures,
            pbr_materials,
            model: Arc::clone(model),
        }
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }
}

/// Interleaved static vertices for `mesh` (missing normal/uv take defaults).
fn static_vertices(mesh: &super::Mesh) -> Vec<ModelVertex> {
    let mut vertices = Vec::with_capacity(mesh.positions.len());
    for (index, &position) in mesh.positions.iter().enumerate() {
        vertices.push(ModelVertex {
            position,
            normal: mesh.normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: mesh.uvs.get(index).copied().unwrap_or([0.0, 0.0]),
        });
    }
    vertices
}

/// Interleaved skinned vertices. Weights are normalized so the shader's
/// blend is well-defined; a zero-weight vertex falls back to joint 0.
fn skinned_vertices(mesh: &super::Mesh) -> Vec<SkinnedVertex> {
    let mut vertices = Vec::with_capacity(mesh.positions.len());
    for (index, &position) in mesh.positions.iter().enumerate() {
        let joints = mesh.joints.get(index).copied().unwrap_or([0; 4]);
        let mut weights = mesh
            .weights
            .get(index)
            .copied()
            .unwrap_or([1.0, 0.0, 0.0, 0.0]);
        let sum: f32 = weights.iter().sum();
        if sum > 1e-6 {
            for weight in &mut weights {
                *weight /= sum;
            }
        } else {
            weights = [1.0, 0.0, 0.0, 0.0];
        }
        vertices.push(SkinnedVertex {
            position,
            normal: mesh.normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: mesh.uvs.get(index).copied().unwrap_or([0.0, 0.0]),
            joints,
            weights,
        });
    }
    vertices
}

/// Interleaved PBR vertices (uv1 + tangents; missing attributes emit the
/// fallbacks the shader understands).
fn pbr_vertices(mesh: &super::Mesh) -> Vec<PbrVertex> {
    let mut vertices = Vec::with_capacity(mesh.positions.len());
    for (index, &position) in mesh.positions.iter().enumerate() {
        vertices.push(PbrVertex {
            position,
            normal: mesh.normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv0: mesh.uvs.get(index).copied().unwrap_or([0.0, 0.0]),
            uv1: mesh.uvs1.get(index).copied().unwrap_or([0.0, 0.0]),
            tangent: mesh.tangents.get(index).copied().unwrap_or([0.0; 4]),
        });
    }
    vertices
}

/// Per-frame-slot joint ring: one host buffer, dynamic offsets per draw.
/// `retired` keeps replaced buffers alive until the slot's frame fence has
/// recycled — a mid-frame growth must not free memory an earlier draw in the
/// same submission still references.
struct JointRing {
    buffer: crate::gfx::GpuBuffer,
    capacity: u64,
    used: u64,
    retired: Vec<crate::gfx::GpuBuffer>,
    /// Frame id this slot was last recycled for (`u64::MAX` = never).
    frame_id: u64,
}

/// Owns the model pipelines and per-frame joint rings. See the module docs.
pub struct ModelRenderer {
    /// Channel-debug code written into the PBR push block (0 = lit).
    pub debug_channel: u32,
    ctx: Arc<GpuContext>,
    pipeline_static: vk::Pipeline,
    pipeline_skinned: vk::Pipeline,
    pipeline_static_blended: vk::Pipeline,
    pipeline_skinned_blended: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl_joints: vk::DescriptorSetLayout,
    dsl_texture: vk::DescriptorSetLayout,
    /// Per frame slot: descriptor pool chunks (64 draws each), grown on
    /// exhaustion. Per-slot because a reset would invalidate sets an
    /// in-flight frame still references — a slot's chunks reset only when
    /// its fence has recycled.
    pools: Vec<SlotPools>,
    /// Kept so grown chunks match the initial one.
    pool_sizes: [vk::DescriptorPoolSize; 3],
    joint_align: u64,
    rings: Vec<JointRing>,
    draws: u64,
    /// Scene formats kept for lazily-built pipelines (PBR).
    color_formats: Vec<vk::Format>,
    depth_format: vk::Format,
    /// PBR pipeline state, built on first [`Self::draw_pbr`] because its
    /// layout includes the caller's lighting set layout.
    pbr: Option<PbrState>,
}

/// Lazily-created PBR pipelines and the material descriptor layout.
struct PbrState {
    pipeline: vk::Pipeline,
    pipeline_blended: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl_material: vk::DescriptorSetLayout,
}

/// A frame slot's descriptor pools: chunks of 64 draws, grown on demand.
/// `cursor` walks chunks during a frame; recycling the slot rewinds it and
/// resets every chunk.
struct SlotPools {
    chunks: Vec<vk::DescriptorPool>,
    cursor: usize,
}

impl ModelRenderer {
    /// Build pipelines for scene targets (`color_formats` are the scene pass's
    /// colour + velocity formats; `depth_format` its depth).
    pub fn new(
        ctx: &Arc<GpuContext>,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
    ) -> Self {
        let dsl_joints = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe { ctx.device().create_descriptor_set_layout(&info, None) }
                .expect("model joint set layout")
        };
        let dsl_texture = texture::create_sampled_texture_set_layout(ctx);
        let camera_layout = scene_set_layout(ctx);

        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(128);
        let layout = {
            let set_layouts = [camera_layout, dsl_joints, dsl_texture];
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push));
            unsafe { ctx.device().create_pipeline_layout(&info, None) }
                .expect("model pipeline layout")
        };

        let shader = ctx.compile_shader(include_str!("model.wgsl"), "model");
        let pipeline_static = create_model_pipeline(
            ctx,
            shader,
            layout,
            color_formats,
            depth_format,
            c"vs_main",
            false,
            false,
        );
        let pipeline_skinned = create_model_pipeline(
            ctx,
            shader,
            layout,
            color_formats,
            depth_format,
            c"vs_skin",
            true,
            false,
        );
        let pipeline_static_blended = create_model_pipeline(
            ctx,
            shader,
            layout,
            color_formats,
            depth_format,
            c"vs_main",
            false,
            true,
        );
        let pipeline_skinned_blended = create_model_pipeline(
            ctx,
            shader,
            layout,
            color_formats,
            depth_format,
            c"vs_skin",
            true,
            true,
        );
        unsafe { ctx.device().destroy_shader_module(shader, None) };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                .descriptor_count(64),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(64),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(64),
        ];
        let pools = (0..ctx.max_frames_in_flight())
            .map(|_| SlotPools {
                chunks: vec![new_draw_pool(ctx, &pool_sizes)],
                cursor: 0,
            })
            .collect();

        let joint_align = unsafe {
            ctx.instance()
                .get_physical_device_properties(ctx.physical_device())
                .limits
                .min_storage_buffer_offset_alignment
                .max(16)
        };
        let rings = (0..ctx.max_frames_in_flight())
            .map(|_| JointRing {
                buffer: ctx.create_host_buffer("model joint ring", 256 * 1024),
                capacity: 256 * 1024,
                used: 0,
                retired: Vec::new(),
                frame_id: u64::MAX,
            })
            .collect();

        Self {
            debug_channel: 0,
            ctx: Arc::clone(ctx),
            pipeline_static,
            pipeline_skinned,
            pipeline_static_blended,
            pipeline_skinned_blended,
            layout,
            dsl_joints,
            dsl_texture,
            pools,
            pool_sizes,
            joint_align,
            rings,
            draws: 0,
            color_formats: color_formats.to_vec(),
            depth_format,
            pbr: None,
        }
    }

    /// Recycle a frame slot on first use: rewind the ring, reset every pool
    /// chunk, free buffers retired by mid-frame growth.
    fn recycle_slot(&mut self, ring_index: usize, frame_id: u64) {
        if self.rings[ring_index].frame_id == frame_id {
            return;
        }
        self.rings[ring_index].frame_id = frame_id;
        self.rings[ring_index].used = 0;
        self.rings[ring_index].retired.clear();
        let pools = &mut self.pools[ring_index];
        pools.cursor = 0;
        for chunk in &pools.chunks {
            unsafe {
                self.ctx
                    .device()
                    .reset_descriptor_pool(*chunk, vk::DescriptorPoolResetFlags::empty())
            }
            .expect("reset model descriptor pool");
        }
    }

    /// Draw one instance's meshes. Call inside the scene pass callback with
    /// the same `DrawCtx` the game draws its terrain with.
    pub fn draw(&mut self, ctx: &mut DrawCtx<'_>, model: &mut ModelGpu, instance: &ModelInstance) {
        self.draw_with_opacity(ctx, model, instance, 1.0);
    }

    /// Draw the model with a uniform opacity in `0..=1`, using alpha blending.
    /// Intended for LoD transitions; ordinary scene draws should use `draw`.
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
        let frame_slot = ctx.frame_slot();
        let ring_index = frame_slot % self.rings.len();
        // A new frame id on this slot means its fence recycled: the ring may
        // be rewound, its pool reset, and buffers retired by a mid-frame grow
        // finally freed. Same-id draws just keep bumping the cursor.
        self.recycle_slot(ring_index, ctx.frame_id());
        let (model_matrix, prev_model) = instance.motion_matrices();
        for mesh_index in 0..model.meshes.len() {
            let material_index = model.meshes[mesh_index].material;
            let (joint_curr, joint_prev) = if model.meshes[mesh_index].skinned {
                instance.skinning_matrices(0)
            } else {
                (Vec::new(), Vec::new())
            };
            // Static meshes honour their glTF node transform; skinned meshes
            // transform through the joint palette instead.
            let (model_mat, prev_mat) = if model.meshes[mesh_index].skinned {
                (model_matrix, prev_model)
            } else {
                (
                    model_matrix * model.meshes[mesh_index].node_transform,
                    prev_model * model.meshes[mesh_index].node_transform,
                )
            };
            let push = {
                let mut bytes = [0u8; 128];
                bytes[..64].copy_from_slice(bytemuck::bytes_of(&model_mat.to_cols_array_2d()));
                bytes[64..].copy_from_slice(bytemuck::bytes_of(&prev_mat.to_cols_array_2d()));
                bytes
            };
            let mut base_color = model
                .model()
                .materials
                .get(material_index)
                .map(|m| m.base_color)
                .unwrap_or([1.0; 4]);
            base_color[3] *= opacity;

            // Upload the per-draw block (material + joint palette).
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
                let offset = align_up(ring.used, self.joint_align);
                if offset + block_size as u64 > ring.capacity {
                    let grown = self
                        .ctx
                        .create_host_buffer("model joint ring", (offset + block_size as u64) * 2);
                    // Keep the old buffer: draws already recorded this frame
                    // still reference it. Freed when the slot recycles.
                    ring.retired
                        .push(std::mem::replace(&mut ring.buffer, grown));
                    ring.capacity = (offset + block_size as u64) * 2;
                }
                ring.buffer
                    .write(offset, bytemuck::bytes_of(&DrawHeader { base_color }));
                if !joints.is_empty() {
                    ring.buffer.write(
                        offset + std::mem::size_of::<DrawHeader>() as u64,
                        bytemuck::cast_slice(&joints),
                    );
                }
                ring.used = offset + block_size as u64;
                offset
            };

            // Texture: create lazily from the material's decoded pixels.
            if model.textures[material_index].is_none() {
                model.textures[material_index] = Some(Arc::new(material_texture(
                    &self.ctx,
                    &model.model().materials[material_index],
                )));
            }
            let texture = model.textures[material_index]
                .clone()
                .expect("created above");

            // Descriptor set for this draw; a chunk that cannot satisfy it
            // is exhausted, so the cursor moves on (growing a chunk if all
            // are full) rather than failing a valid frame.
            let set = self.alloc_draw_sets(ring_index);
            let (joint_set, texture_set) = (set[0], set[1]);
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(self.rings[ring_index].buffer.buffer)
                .offset(0)
                .range((block_size as u64).max(16));
            // The texture set follows the engine's WGSL contract: sampled
            // image at 0, sampler at 1.
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(texture.view())
                .image_layout(vk::ImageLayout::GENERAL);
            let sampler_info = vk::DescriptorImageInfo::default().sampler(texture.sampler());
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(joint_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .buffer_info(std::slice::from_ref(&buffer_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(texture_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&image_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(texture_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(&sampler_info)),
            ];
            unsafe { self.ctx.device().update_descriptor_sets(&writes, &[]) };

            // Record the draw against the game's command buffer.
            let (device, cmd) = ctx.raw();
            let pipeline = match (model.meshes[mesh_index].skinned, opacity < 1.0) {
                (false, false) => self.pipeline_static,
                (true, false) => self.pipeline_skinned,
                (false, true) => self.pipeline_static_blended,
                (true, true) => self.pipeline_skinned_blended,
            };
            let dynamic_offset = offset as u32;
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[ctx.scene_descriptor_set(), joint_set, texture_set],
                    &[dynamic_offset],
                );
                device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX, 0, &push);
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[model.meshes[mesh_index].vertices.buffer],
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    cmd,
                    model.meshes[mesh_index].indices.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(cmd, model.meshes[mesh_index].index_count, 1, 0, 0, 0);
            }
            self.draws += 1;
        }
    }

    /// Draw one instance's static meshes through the full PBR path
    /// (`model_pbr.wgsl`): all material maps, KHR_texture_transform, MASK
    /// cutoffs, shared [`crate::lighting::Lighting`] environment. Skinned
    /// meshes are skipped — they stay on the simple path.
    pub fn draw_pbr(
        &mut self,
        ctx: &mut DrawCtx<'_>,
        model: &mut ModelGpu,
        instance: &ModelInstance,
        lighting: &crate::lighting::Lighting,
    ) {
        self.draw_pbr_with_opacity(ctx, model, instance, lighting, 1.0);
    }

    /// PBR equivalent of [`Self::draw_with_opacity`]: alpha blending, no
    /// depth writes, `LESS_OR_EQUAL` — the velocity attachment is not
    /// blended (same contract as the simple blended pipeline).
    pub fn draw_pbr_with_opacity(
        &mut self,
        ctx: &mut DrawCtx<'_>,
        model: &mut ModelGpu,
        instance: &ModelInstance,
        lighting: &crate::lighting::Lighting,
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
        if self.pbr.is_none() {
            self.pbr = Some(PbrState::new(
                &self.ctx,
                &self.color_formats,
                self.depth_format,
                self.dsl_joints,
                lighting,
            ));
        }
        let frame_slot = ctx.frame_slot();
        let ring_index = frame_slot % self.rings.len();
        self.recycle_slot(ring_index, ctx.frame_id());
        let (model_matrix, prev_model) = instance.motion_matrices();
        let lighting_set = lighting.set(frame_slot);
        let pbr_layout = self.pbr.as_ref().unwrap().layout;
        let dsl_material = self.pbr.as_ref().unwrap().dsl_material;

        for mesh_index in 0..model.meshes.len() {
            if model.meshes[mesh_index].skinned {
                continue; // PBR path is static-only this phase
            }
            let model_mat = model_matrix * model.meshes[mesh_index].node_transform;
            let prev_mat = prev_model * model.meshes[mesh_index].node_transform;
            let push = {
                let mut bytes = [0u8; 144];
                bytes[..64].copy_from_slice(bytemuck::bytes_of(&model_mat.to_cols_array_2d()));
                bytes[64..128].copy_from_slice(bytemuck::bytes_of(&prev_mat.to_cols_array_2d()));
                bytes[128..132].copy_from_slice(&(self.debug_channel as f32).to_le_bytes());
                bytes
            };
            let material_index = model.meshes[mesh_index].material;
            let material = &model.model().materials[material_index];
            let blended = opacity < 1.0 || material.alpha_mode == super::AlphaMode::Blend;
            // base_color.a is the crossfade opacity only — the material's
            // own alpha factor stays in params2.w so OPAQUE texels can't be
            // faded out by a factor below 1.
            let mut base_color = material.base_color;
            base_color[3] = opacity;

            let offset = {
                let ring = &mut self.rings[ring_index];
                let block_size = std::mem::size_of::<DrawHeader>() as u64;
                let offset = align_up(ring.used, self.joint_align);
                if offset + block_size > ring.capacity {
                    let grown = self
                        .ctx
                        .create_host_buffer("model joint ring", (offset + block_size) * 2);
                    ring.retired
                        .push(std::mem::replace(&mut ring.buffer, grown));
                    ring.capacity = (offset + block_size) * 2;
                }
                ring.buffer
                    .write(offset, bytemuck::bytes_of(&DrawHeader { base_color }));
                ring.used = offset + block_size;
                offset
            };

            if model.meshes[mesh_index].pbr_vertices.is_none() {
                model.meshes[mesh_index].pbr_vertices = Some(pbr_vertex_buffer(
                    &self.ctx,
                    &model.model().meshes[mesh_index],
                ));
            }
            if model.pbr_materials[material_index].is_none() {
                model.pbr_materials[material_index] = Some(Arc::new(PbrMaterialGpu::new(
                    &self.ctx,
                    dsl_material,
                    &model.model().materials[material_index],
                )));
            }
            let material_gpu = model.pbr_materials[material_index].clone().unwrap();

            let set = self.alloc_draw_sets(ring_index);
            let joint_set = set[0];
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(self.rings[ring_index].buffer.buffer)
                .offset(0)
                .range(std::mem::size_of::<DrawHeader>() as u64);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(joint_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                .buffer_info(std::slice::from_ref(&buffer_info));
            unsafe { self.ctx.device().update_descriptor_sets(&[write], &[]) };

            let pbr = self.pbr.as_ref().unwrap();
            let pipeline = if blended {
                pbr.pipeline_blended
            } else {
                pbr.pipeline
            };
            let (device, cmd) = ctx.raw();
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pbr_layout,
                    0,
                    &[
                        ctx.scene_descriptor_set(),
                        joint_set,
                        material_gpu.set,
                        lighting_set,
                    ],
                    &[offset as u32],
                );
                device.cmd_push_constants(
                    cmd,
                    pbr_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &push,
                );
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[model.meshes[mesh_index]
                        .pbr_vertices
                        .as_ref()
                        .unwrap()
                        .buffer],
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    cmd,
                    model.meshes[mesh_index].indices.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(cmd, model.meshes[mesh_index].index_count, 1, 0, 0, 0);
            }
            self.draws += 1;
        }
    }

    /// Two draw sets (joint ring + texture) from `slot`'s pool chunks.
    fn alloc_draw_sets(&mut self, slot: usize) -> [vk::DescriptorSet; 2] {
        let layouts = [self.dsl_joints, self.dsl_texture];
        loop {
            let pools = &mut self.pools[slot];
            if pools.cursor == pools.chunks.len() {
                let chunk = new_draw_pool(&self.ctx, &self.pool_sizes);
                pools.chunks.push(chunk);
            }
            let result = unsafe {
                self.ctx.device().allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pools.chunks[pools.cursor])
                        .set_layouts(&layouts),
                )
            };
            match result {
                Ok(sets) => return [sets[0], sets[1]],
                Err(err) if crate::gfx::vulkan::pool_exhaustion_is_growable(err) => {
                    pools.cursor += 1;
                }
                Err(err) => panic!("model draw set allocation: {err}"),
            }
        }
    }

    /// Draws recorded since construction (diagnostics).
    pub fn draws(&self) -> u64 {
        self.draws
    }
}

impl Drop for ModelRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            let device = self.ctx.device();
            device.destroy_pipeline(self.pipeline_static, None);
            device.destroy_pipeline(self.pipeline_skinned, None);
            device.destroy_pipeline(self.pipeline_static_blended, None);
            device.destroy_pipeline(self.pipeline_skinned_blended, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.dsl_joints, None);
            device.destroy_descriptor_set_layout(self.dsl_texture, None);
            if let Some(pbr) = &self.pbr {
                device.destroy_pipeline(pbr.pipeline, None);
                device.destroy_pipeline(pbr.pipeline_blended, None);
                device.destroy_pipeline_layout(pbr.layout, None);
                device.destroy_descriptor_set_layout(pbr.dsl_material, None);
            }
            for pools in &self.pools {
                for chunk in &pools.chunks {
                    device.destroy_descriptor_pool(*chunk, None);
                }
            }
        }
    }
}

fn material_texture(ctx: &Arc<GpuContext>, material: &Material) -> GpuTexture {
    match &material.base_color_texture {
        Some(decoded) => GpuTexture::from_rgba8(
            ctx,
            &format!("material {}", material.name),
            &decoded.pixels,
            decoded.width,
            decoded.height,
            TextureOptions {
                format: TextureFormat::Srgb,
                ..Default::default()
            },
        ),
        None => GpuTexture::white(ctx),
    }
}

/// The engine camera set layout (binding 0: uniform, vertex+fragment) — must
/// stay structurally compatible with the scene pass's set 0. Fragment access
/// is required: impostor.wgsl reads the camera matrices in fs_main for
/// frag_depth and motion vectors.
pub(crate) fn scene_set_layout(ctx: &Arc<GpuContext>) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { ctx.device().create_descriptor_set_layout(&info, None) }.expect("camera set layout")
}

/// One 64-draw pool chunk (two sets per draw: joint ring + texture).
fn new_draw_pool(ctx: &GpuContext, pool_sizes: &[vk::DescriptorPoolSize]) -> vk::DescriptorPool {
    unsafe {
        ctx.device().create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(128)
                .pool_sizes(pool_sizes),
            None,
        )
    }
    .expect("model descriptor pool")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_layouts_are_compact() {
        assert_eq!(std::mem::size_of::<ModelVertex>(), 32);
        // 32 + joints (4 x u16 = 8) + weights (4 x f32 = 16).
        assert_eq!(std::mem::size_of::<SkinnedVertex>(), 56);
        assert_eq!(std::mem::size_of::<JointMatrices>(), 128);
        // 32 + uv1 (8) + tangent (16).
        assert_eq!(std::mem::size_of::<PbrVertex>(), 56);
        assert_eq!(std::mem::size_of::<MaterialFactors>(), 13 * 16);
    }

    /// The three interleaved builders share the old per-vertex output but
    /// fill defaults without allocating per vertex.
    #[test]
    fn vertex_packers_fill_defaults_and_normalize_weights() {
        let mesh = crate::model::Mesh {
            name: "pack".into(),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            uvs: Vec::new(),
            uvs1: Vec::new(),
            tangents: Vec::new(),
            joints: vec![[1, 2, 3, 4]],
            weights: vec![[2.0, 2.0, 0.0, 0.0], [0.0; 4]],
            indices: vec![0, 1, 2],
            material: 0,
            node: None,
        };

        let statics = static_vertices(&mesh);
        assert_eq!(statics.len(), 3);
        assert_eq!(statics[0].normal, [0.0, 0.0, 1.0]);
        assert_eq!(statics[1].normal, [0.0, 1.0, 0.0], "default normal");
        assert_eq!(statics[1].uv, [0.0, 0.0], "default uv");
        assert_eq!(
            bytemuck::cast_slice::<ModelVertex, u8>(&statics).len(),
            3 * 32
        );

        let skinned = skinned_vertices(&mesh);
        assert_eq!(skinned[0].joints, [1, 2, 3, 4]);
        assert_eq!(skinned[0].weights, [0.5, 0.5, 0.0, 0.0]);
        assert_eq!(
            skinned[1].weights,
            [1.0, 0.0, 0.0, 0.0],
            "zero weights fall back to joint 0"
        );
        assert_eq!(skinned[2].joints, [0; 4], "missing joints default");
        assert_eq!(
            bytemuck::cast_slice::<SkinnedVertex, u8>(&skinned).len(),
            3 * 56
        );

        let pbr = pbr_vertices(&mesh);
        assert_eq!(pbr[0].uv1, [0.0, 0.0]);
        assert_eq!(pbr[0].tangent, [0.0; 4]);
        assert_eq!(bytemuck::cast_slice::<PbrVertex, u8>(&pbr).len(), 3 * 56);
    }

    #[test]
    fn align_up_rounds() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 16), 272);
    }

    /// Regression: the camera set must expose FRAGMENT — impostor.wgsl reads
    /// `camera` in fs_main (object ray + frag_depth). A VERTEX-only layout
    /// silently produced garbage fragment inputs (bogus card depths).
    #[test]
    fn camera_layout_exposes_fragment_stage() {
        let src = include_str!("render.rs");
        let dsl = &src[src.find("fn scene_set_layout").unwrap()..];
        assert!(
            dsl.contains("VERTEX | vk::ShaderStageFlags::FRAGMENT"),
            "scene_set_layout camera binding must include FRAGMENT"
        );
        // And the engine scene builder in gfx.rs must match.
        let gfx = include_str!("../gfx/renderer.rs");
        let idx = gfx.find("\"scene layout\"").unwrap();
        let scene = &gfx[idx..idx + 600];
        assert!(
            scene.contains("VERTEX | vk::ShaderStageFlags::FRAGMENT"),
            "gfx scene_dsl camera binding must include FRAGMENT"
        );
    }
}

#[cfg(test)]
mod normal_matrix_tests {
    use glam::{Mat4, Vec3};

    /// CPU mirror of the unit-basis cofactor normal matrix in
    /// model_pbr.wgsl (scale-safe to 1e-6..1e6).
    fn normal_matrix(m: Mat4) -> Mat4 {
        let a = m.col(0).truncate();
        let b = m.col(1).truncate();
        let c = m.col(2).truncate();
        let sa = a.length().max(1e-30);
        let sb = b.length().max(1e-30);
        let sc = c.length().max(1e-30);
        let (na, nb, nc) = (a / sa, b / sb, c / sc);
        let det_u = na.dot(nb.cross(nc));
        let cols = [
            (nb.cross(nc) / sa) / det_u,
            (nc.cross(na) / sb) / det_u,
            (na.cross(nb) / sc) / det_u,
        ];
        Mat4::from_cols(
            cols[0].extend(0.0),
            cols[1].extend(0.0),
            cols[2].extend(0.0),
            Vec3::ZERO.extend(1.0),
        )
    }

    #[test]
    fn cofactor_matches_glam_and_orthonormalises() {
        for (m, check_glam) in [
            (Mat4::IDENTITY, true),
            (
                Mat4::from_scale(Vec3::new(2.0, 1.0, 4.0))
                    * Mat4::from_rotation_y(0.6)
                    * Mat4::from_rotation_x(0.3),
                true,
            ),
            (Mat4::from_scale(Vec3::new(-1.0, 2.0, 1.0)), true),
            // glam's inverse() itself loses precision at extreme scales —
            // our unit-det formula stays exact; verify orthogonality only.
            (
                Mat4::from_scale(Vec3::splat(1e-6)) * Mat4::from_rotation_z(0.9),
                false,
            ),
            (
                Mat4::from_scale(Vec3::splat(1e6)) * Mat4::from_rotation_x(0.2),
                false,
            ),
        ] {
            let ours = normal_matrix(m);
            if check_glam {
                let glam_inv_t = m.inverse().transpose();
                for c in 0..3 {
                    let d = (ours.col(c).truncate() - glam_inv_t.col(c).truncate()).length();
                    assert!(d < 1e-5, "col {c} differs: {d}");
                }
            }
            // Orthogonality: transformed normal stays perpendicular to the
            // transformed tangent under non-uniform scale.
            let n = Vec3::new(0.0, 1.0, 0.0);
            let t = Vec3::new(1.0, 0.0, 0.0);
            let n_w = (ours * n.extend(0.0)).truncate();
            let t_w = m.transform_vector3(t);
            assert!(
                n_w.normalize().dot(t_w.normalize()).abs() < 1e-3,
                "normal not orthogonal to transformed tangent"
            );
            // And M^T * N = I — the defining property of the normal matrix.
            let back = m.transpose() * ours;
            for c in 0..3 {
                let d = (back.col(c).truncate() - Mat4::IDENTITY.col(c).truncate()).length();
                assert!(d < 1e-4, "M^T*N != I at col {c}: {d}");
            }
        }
        // Reflection: det<0 flips tangent handedness.
        let m = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        assert!(
            m.col(0)
                .truncate()
                .dot(m.col(1).truncate().cross(m.col(2).truncate()))
                < 0.0
        );
        // Singular: det=0 → shader falls back, no NaN mirror needed here.
    }
}
