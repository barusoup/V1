//! Native Metal impostor baker: 26 orthographic captures + GPU compute post-bake.

use super::atlas::{ImpostorAsset, ImpostorAssetLimits};
use super::{FallbackReason, ImpostorBounds, VIEW_COUNT, VIEWS, projected_extent};
use crate::gfx::metal::pipeline::{
    VertexAttr, create_compute_pipeline, create_depth_stencil_state, make_vertex_descriptor,
};
use crate::gfx::metal::shader::{MslBindings, PUSH_CONSTANT_BUFFER_SLOT, VERTEX_BUFFER_SLOT};
use crate::gfx::{GpuBuffer, GpuContext, GpuImage};
use crate::model::render::{PbrVertex, pack_transform};
use crate::model::{AlphaMode, Material, Model, TexelFilter, TextureRef, Wrap};
use crate::texture::{GpuTexture, TextureFormat, TextureOptions};
use glam::{Mat4, Vec2, Vec3};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCompareFunction,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCullMode, MTLDepthStencilState,
    MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineState, MTLSize,
    MTLStoreAction, MTLVertexFormat, MTLViewport, MTLWinding,
};
use std::ptr::NonNull;
use std::sync::Arc;

mod materials {
    use super::{
        AlphaMode, ImpMaterialParams, Material, TexelFilter, TextureRef, Wrap, pack_transform,
    };

    pub(crate) fn imp_material_params(m: &Material) -> ImpMaterialParams {
        let pack_wf = |tex: Option<&TextureRef>| -> f32 {
            let Some(t) = tex else {
                return 0.0;
            };
            let w = |wrap: Wrap| -> u32 {
                match wrap {
                    Wrap::Repeat => 0,
                    Wrap::ClampToEdge => 1,
                    Wrap::MirroredRepeat => 2,
                }
            };
            let f = match t.min_filter {
                TexelFilter::Linear => 0,
                TexelFilter::Nearest => 1,
            };
            (w(t.wrap_s) | (w(t.wrap_t) << 2) | (f << 4)) as f32
        };
        let (uv0, mut tf0) = pack_transform(m.base_color_map.as_ref());
        let (uv1, mut tf1) = pack_transform(m.metallic_roughness_texture.as_ref());
        let (uv2, mut tf2) = pack_transform(m.normal_texture.as_ref());
        let (uv3, mut tf3) = pack_transform(m.occlusion_texture.as_ref());
        let (uv4, mut tf4) = pack_transform(m.emissive_texture.as_ref());
        tf0[3] = pack_wf(m.base_color_map.as_ref());
        tf1[3] = pack_wf(m.metallic_roughness_texture.as_ref());
        tf2[3] = pack_wf(m.normal_texture.as_ref());
        tf3[3] = pack_wf(m.occlusion_texture.as_ref());
        tf4[3] = pack_wf(m.emissive_texture.as_ref());
        ImpMaterialParams {
            base_color: m.base_color,
            params: [
                m.metallic_factor,
                m.roughness_factor,
                m.alpha_cutoff,
                match m.alpha_mode {
                    AlphaMode::Opaque => 0.0,
                    AlphaMode::Mask => 1.0,
                    AlphaMode::Blend => 2.0,
                },
            ],
            params2: [
                m.occlusion_strength,
                m.normal_scale,
                if m.double_sided { 1.0 } else { 0.0 },
                0.0,
            ],
            emissive: [
                m.emissive_factor[0],
                m.emissive_factor[1],
                m.emissive_factor[2],
                m.emissive_strength,
            ],
            uv: [uv0, uv1, uv2, uv3, uv4],
            tf: [tf0, tf1, tf2, tf3, tf4],
        }
    }
}

#[path = "bake/geometry.rs"]
mod geometry;
#[path = "bake/previews.rs"]
mod previews;
#[path = "bake/sources.rs"]
mod sources;

use geometry::{check_ref_lists, voxelize_batches};
use previews::build_previews;
use sources::prepare_sources;

#[derive(Clone, Copy, Debug)]
pub struct BakeSettings {
    pub resolution: u32,
    pub supersample: u32,
    pub voxel_resolution: u32,
    pub material_resolution: u32,
}

impl BakeSettings {
    pub fn compact() -> Self {
        Self {
            resolution: 256,
            supersample: 2,
            voxel_resolution: 64,
            material_resolution: 256,
        }
    }
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            resolution: 512,
            supersample: 2,
            voxel_resolution: 128,
            material_resolution: 512,
        }
    }
}

pub struct ImpostorSource<'a> {
    model: &'a Model,
    meshes: Option<&'a [usize]>,
    transform: Mat4,
}

impl<'a> ImpostorSource<'a> {
    pub fn model(model: &'a Model) -> Self {
        Self {
            model,
            meshes: None,
            transform: Mat4::IDENTITY,
        }
    }

    pub fn meshes(model: &'a Model, mesh_indices: &'a [usize]) -> Self {
        Self {
            model,
            meshes: Some(mesh_indices),
            transform: Mat4::IDENTITY,
        }
    }

    pub fn transformed(mut self, transform: Mat4) -> Self {
        self.transform = transform;
        self
    }
}

struct PreparedBake<'a> {
    batches: Vec<(Vec<PbrVertex>, Vec<u32>)>,
    materials: Vec<&'a Material>,
    bounds: ImpostorBounds,
}

#[cfg(test)]
pub struct BakeTestLayers {
    pub albedo: Vec<Vec<u8>>,
    pub normal_ao: Vec<Vec<u8>>,
    pub depth: Vec<Vec<u16>>,
}

#[cfg(test)]
#[derive(Debug)]
pub struct BakeTestStats {
    pub coverage: [f32; VIEW_COUNT],
    pub center_depth: [f32; VIEW_COUNT],
    pub dilated: bool,
    pub mip_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CaptureFactors {
    params: [f32; 4],
    params2: [f32; 4],
    emissive: [f32; 4],
    uv: [[f32; 4]; 5],
    tf: [[f32; 4]; 5],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CapturePush {
    mvp: [[f32; 4]; 4],
    face_n: [f32; 4],
    box_center: [f32; 4],
    extents: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DParams {
    res: u32,
    ss: u32,
    src_off: u32,
    dst_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ImpMaterialParams {
    base_color: [f32; 4],
    params: [f32; 4],
    params2: [f32; 4],
    emissive: [f32; 4],
    uv: [[f32; 4]; 5],
    tf: [[f32; 4]; 5],
}

struct BakeMaterial {
    factors: GpuBuffer,
    base_color: GpuBuffer,
    base: Arc<GpuTexture>,
    mr: Arc<GpuTexture>,
    normal: Arc<GpuTexture>,
    ao: Arc<GpuTexture>,
    emissive: Arc<GpuTexture>,
    base_alpha: f32,
}

struct BakeBatch {
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    index_count: u32,
    material: BakeMaterial,
}

pub struct ImpostorBaker {
    ctx: Arc<GpuContext>,
    capture_pipe: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    capture_depth_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    cs_down: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cs_dilate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cs_mip: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cs_pack: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ImpostorBaker {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let cap_bindings = MslBindings::new()
            .buffer(0, 0, 0, false)
            .texture(0, 1, 0)
            .sampler(0, 2, 0)
            .texture(0, 3, 1)
            .sampler(0, 4, 1)
            .texture(0, 5, 2)
            .sampler(0, 6, 2)
            .texture(0, 7, 3)
            .sampler(0, 8, 3)
            .texture(0, 9, 4)
            .sampler(0, 10, 4)
            .buffer(1, 0, 1, false)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);

        let cap_src = include_str!("capture.wgsl");
        let vdesc = make_vertex_descriptor(
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
        let capture_pipe = crate::gfx::metal::pipeline::create_render_pipeline_unflipped(
            ctx,
            "impostor_capture",
            cap_src,
            &cap_bindings,
            "vs_main",
            "fs_main",
            &[
                MTLPixelFormat::RGBA16Float,
                MTLPixelFormat::RGBA16Float,
                MTLPixelFormat::RGBA16Float,
                MTLPixelFormat::RGBA16Float,
                MTLPixelFormat::R16Float,
            ],
            Some(MTLPixelFormat::Depth32Float),
            Some(&vdesc),
            false,
        );
        let capture_depth_state =
            create_depth_stencil_state(ctx, true, true, MTLCompareFunction::LessEqual);

        let post_src = include_str!("bake_post.wgsl");
        let down_bindings = MslBindings::new()
            .texture(0, 1, 0)
            .texture(0, 2, 1)
            .texture(0, 3, 2)
            .texture(0, 4, 3)
            .texture(0, 5, 4)
            .buffer(0, 6, 0, true)
            .buffer(0, 7, 1, true)
            .buffer(0, 8, 2, true)
            .buffer(0, 9, 3, true)
            .buffer(0, 10, 4, true)
            .buffer(0, 11, 5, true)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);
        let cs_down =
            create_compute_pipeline(ctx, "impostor_down", post_src, &down_bindings, "cs_down");

        let dilate_bindings = MslBindings::new()
            .buffer(0, 1, 0, false)
            .buffer(0, 2, 1, false)
            .buffer(0, 3, 2, false)
            .buffer(0, 4, 3, false)
            .buffer(0, 5, 4, true)
            .buffer(0, 6, 5, true)
            .buffer(0, 7, 6, true)
            .buffer(0, 8, 7, true)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);
        let cs_dilate = create_compute_pipeline(
            ctx,
            "impostor_dilate",
            post_src,
            &dilate_bindings,
            "cs_dilate",
        );

        let mip_bindings = MslBindings::new()
            .buffer(0, 1, 0, false)
            .buffer(0, 2, 1, false)
            .buffer(0, 3, 2, false)
            .buffer(0, 4, 3, false)
            .buffer(0, 5, 4, false)
            .buffer(0, 6, 5, true)
            .buffer(0, 7, 6, true)
            .buffer(0, 8, 7, true)
            .buffer(0, 9, 8, true)
            .buffer(0, 10, 9, true)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);
        let cs_mip =
            create_compute_pipeline(ctx, "impostor_mip", post_src, &mip_bindings, "cs_mip");

        let pack_bindings = MslBindings::new()
            .buffer(0, 1, 0, false)
            .buffer(0, 2, 1, false)
            .buffer(0, 3, 2, false)
            .buffer(0, 4, 3, false)
            .buffer(0, 5, 4, false)
            .buffer(0, 6, 5, true)
            .buffer(0, 7, 6, true)
            .buffer(0, 8, 7, true)
            .buffer(0, 9, 8, true)
            .buffer(0, 10, 9, true)
            .push_constants(PUSH_CONSTANT_BUFFER_SLOT);
        let cs_pack =
            create_compute_pipeline(ctx, "impostor_pack", post_src, &pack_bindings, "cs_pack");

        Self {
            ctx: Arc::clone(ctx),
            capture_pipe,
            capture_depth_state,
            cs_down,
            cs_dilate,
            cs_mip,
            cs_pack,
        }
    }

    pub fn bake(
        &self,
        model: &Arc<Model>,
        settings: BakeSettings,
    ) -> Result<ImpostorAsset, FallbackReason> {
        super::static_capture_supported(model)?;
        self.bake_sources(&[ImpostorSource::model(model)], settings)
    }

    pub fn bake_with_limits(
        &self,
        model: &Arc<Model>,
        settings: BakeSettings,
        limits: ImpostorAssetLimits,
    ) -> Result<ImpostorAsset, FallbackReason> {
        super::static_capture_supported(model)?;
        self.bake_sources_with_limits(&[ImpostorSource::model(model)], settings, limits)
    }

    pub fn bake_meshes(
        &self,
        model: &Arc<Model>,
        mesh_indices: &[usize],
        settings: BakeSettings,
    ) -> Result<ImpostorAsset, FallbackReason> {
        super::static_capture_supported(model)?;
        self.bake_sources(&[ImpostorSource::meshes(model, mesh_indices)], settings)
    }

    pub fn bake_meshes_with_limits(
        &self,
        model: &Arc<Model>,
        mesh_indices: &[usize],
        settings: BakeSettings,
        limits: ImpostorAssetLimits,
    ) -> Result<ImpostorAsset, FallbackReason> {
        super::static_capture_supported(model)?;
        self.bake_sources_with_limits(
            &[ImpostorSource::meshes(model, mesh_indices)],
            settings,
            limits,
        )
    }

    pub fn bake_sources(
        &self,
        sources: &[ImpostorSource<'_>],
        settings: BakeSettings,
    ) -> Result<ImpostorAsset, FallbackReason> {
        self.bake_sources_with_limits(sources, settings, ImpostorAssetLimits::default())
    }

    pub fn bake_sources_with_limits(
        &self,
        sources: &[ImpostorSource<'_>],
        settings: BakeSettings,
        limits: ImpostorAssetLimits,
    ) -> Result<ImpostorAsset, FallbackReason> {
        let res = settings.resolution;
        let mat = settings.material_resolution;
        let vox = settings.voxel_resolution;
        let cap_dim = 16384u32;
        if !res.is_power_of_two() || !(16..=1024).contains(&res) {
            return Err(FallbackReason::InvalidSettings(format!(
                "atlas resolution {res} must be a power of two in 16..=1024"
            )));
        }
        if res > limits.max_texture_resolution || res > cap_dim {
            return Err(FallbackReason::InvalidSettings(format!(
                "atlas resolution {res} exceeds limit {} / device {cap_dim}",
                limits.max_texture_resolution
            )));
        }
        if !(1..=4).contains(&settings.supersample) {
            return Err(FallbackReason::InvalidSettings(format!(
                "supersample {} must be 1..=4",
                settings.supersample
            )));
        }
        if vox != 0
            && (!vox.is_power_of_two()
                || !(16..=256).contains(&vox)
                || vox > limits.max_voxel_resolution)
        {
            return Err(FallbackReason::InvalidSettings(format!(
                "voxel resolution {vox} must be 0 or a power of two in 16..=256 within the limit"
            )));
        }
        if !mat.is_power_of_two() || mat > 1024 || mat > limits.max_texture_resolution {
            return Err(FallbackReason::InvalidSettings(format!(
                "material resolution {mat} must be a power of two <= 1024 and within the limit"
            )));
        }
        if res * settings.supersample > cap_dim || mat > cap_dim {
            return Err(FallbackReason::InvalidSettings(format!(
                "capture edge {} / material res {mat} exceeds device {cap_dim}",
                res * settings.supersample
            )));
        }
        let res = res.max(16);
        let prepared = prepare_sources(sources, 4.0 / res as f32)?;
        if prepared.materials.len() > limits.max_material_count as usize {
            return Err(FallbackReason::ResourceLimit {
                requested: prepared.materials.len() as u64,
                limit: u64::from(limits.max_material_count),
            });
        }

        let mips = 32 - res.leading_zeros();
        let atlas_bytes: u64 = [4u64, 4, 4, 4, 2]
            .iter()
            .map(|&bpp| {
                let mut e = res as u64;
                (0..mips)
                    .map(|_| {
                        let b = e * e * bpp * VIEW_COUNT as u64;
                        e = (e / 2).max(1);
                        b
                    })
                    .sum::<u64>()
            })
            .sum();
        let mat_layers = prepared.materials.len().max(1) as u64;
        let mat_bytes = 5 * (mat as u64).pow(2) * 4 * mat_layers;
        let table_bytes = (mat_layers * std::mem::size_of::<ImpMaterialParams>() as u64).max(4);
        let dense_bytes = if vox > 0 { (vox as u64).pow(3) * 4 } else { 0 };
        let estimate = atlas_bytes + mat_bytes + table_bytes + dense_bytes;
        if estimate > limits.max_gpu_bytes {
            return Err(FallbackReason::ResourceLimit {
                requested: estimate,
                limit: limits.max_gpu_bytes,
            });
        }

        let (voxel_cells, surface_count, triangle_records, triangle_refs) =
            voxelize_batches(&prepared.batches, prepared.bounds, vox);
        let triangle_count = (triangle_records.len() / 48) as u32;
        let triangle_base = voxel_cells.len() as u32 + triangle_refs.len() as u32;
        let max_cell_refs = check_ref_lists(
            &triangle_refs,
            voxel_cells.len(),
            triangle_base as usize,
            limits.max_cell_references,
        )?;
        let voxel_bytes_exact =
            (voxel_cells.len() + triangle_refs.len() + triangle_records.len()) * 4;
        let exact = estimate - dense_bytes + voxel_bytes_exact as u64;
        if exact > limits.max_gpu_bytes {
            return Err(FallbackReason::ResourceLimit {
                requested: exact,
                limit: limits.max_gpu_bytes,
            });
        }
        let sb_limit = u64::from(self.ctx.max_storage_buffer_range());
        if voxel_bytes_exact as u64 > sb_limit || table_bytes > sb_limit {
            return Err(FallbackReason::ResourceLimit {
                requested: voxel_bytes_exact as u64,
                limit: sb_limit,
            });
        }

        self.bake_prepared_with(
            prepared,
            settings,
            limits,
            voxel_cells,
            surface_count,
            triangle_records,
            triangle_refs,
            triangle_base,
            triangle_count,
            max_cell_refs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bake_prepared_with(
        &self,
        prepared: PreparedBake<'_>,
        settings: BakeSettings,
        _limits: ImpostorAssetLimits,
        voxel_cells: Vec<u32>,
        surface_count: u32,
        triangle_records: Vec<u32>,
        triangle_refs: Vec<u32>,
        triangle_base: u32,
        triangle_count: u32,
        max_cell_refs: u64,
    ) -> Result<ImpostorAsset, FallbackReason> {
        let start = std::time::Instant::now();
        let res = settings.resolution.max(16);
        let ss = settings.supersample.max(1);
        let cap = res * ss;
        let bounds = prepared.bounds;
        let center = bounds.center();
        let extent = bounds.extent();

        let batches = prepared.batches;
        let materials = prepared.materials;
        let has_mask = materials.iter().any(|m| m.alpha_mode == AlphaMode::Mask);
        let mut baked = Vec::with_capacity(batches.len());
        for (material_index, (vertices, indices)) in batches.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }
            let material = materials[material_index];
            baked.push(BakeBatch {
                vertex_buffer: self
                    .ctx
                    .create_vertex_buffer("bake verts", bytemuck::cast_slice(vertices)),
                index_buffer: self
                    .ctx
                    .create_index_buffer("bake idx", bytemuck::cast_slice(indices)),
                index_count: indices.len() as u32,
                material: self.bake_material(material),
            });
        }
        if baked.is_empty() {
            return Err(FallbackReason::EmptyGeometry);
        }

        let mut voxel_cells = voxel_cells;
        voxel_cells.extend(triangle_refs);
        voxel_cells.extend(triangle_records);

        let mut mvps = [Mat4::IDENTITY; VIEW_COUNT];
        let mut face_ns = [Vec3::ZERO; VIEW_COUNT];
        let mut exts = [[0.0f32; 3]; VIEW_COUNT];
        for (f, view) in VIEWS.iter().enumerate() {
            let ext_u = projected_extent(extent, view.u);
            let ext_v = projected_extent(extent, view.v);
            let ext_n = projected_extent(extent, view.dir);
            let eye = center + view.dir * (ext_n * 0.5 + 1.0);
            let view_mat = Mat4::look_to_rh(eye, -view.dir, view.v);
            let proj = Mat4::orthographic_rh(
                -ext_u * 0.5,
                ext_u * 0.5,
                -ext_v * 0.5,
                ext_v * 0.5,
                0.01,
                ext_n + 2.0,
            );
            mvps[f] = proj * view_mat;
            face_ns[f] = view.dir;
            exts[f] = [ext_u, ext_v, ext_n];
        }

        let caps = self.capture_views_gpu(&baked, cap, &mvps, &face_ns, center, &exts);
        let (images, tiles, dep_tiles, coverage_frac) = self.gpu_post(&caps, res, ss);

        let voxel_bytes = voxel_cells.len() * std::mem::size_of::<u32>();
        let voxels = if voxel_cells.is_empty() {
            self.ctx.create_storage_buffer("impostor voxels", &[0u8; 4])
        } else {
            self.ctx
                .create_storage_buffer("impostor voxels", bytemuck::cast_slice(&voxel_cells))
        };
        let _ = max_cell_refs;
        let mips = 32 - res.leading_zeros();

        let mut slab_valid = [true; super::AXIS_COUNT];
        let mut any_valid = false;
        for a in 0..super::AXIS_COUNT {
            slab_valid[a] = coverage_frac[2 * a] > 0.005 || coverage_frac[2 * a + 1] > 0.005;
            any_valid |= slab_valid[a];
        }
        if !any_valid {
            return Err(FallbackReason::EmptyGeometry);
        }

        #[cfg(test)]
        let test_stats_inner = {
            let mut stats = BakeTestStats {
                coverage: [0.0; VIEW_COUNT],
                center_depth: [0.0; VIEW_COUNT],
                dilated: false,
                mip_count: 0,
            };
            for f in 0..VIEW_COUNT {
                let alb = &tiles[0][f];
                let mut covered = 0usize;
                for px in alb.chunks_exact(4) {
                    if px[3] > 0 {
                        covered += 1;
                    } else if px[0] > 0 || px[1] > 0 || px[2] > 0 {
                        stats.dilated = true;
                    }
                }
                stats.coverage[f] = covered as f32 / (res * res) as f32;
                let c = ((res / 2 * res + res / 2) as usize).min(dep_tiles[f].len() - 1);
                stats.center_depth[f] =
                    u16::from_le_bytes([dep_tiles[f][c], dep_tiles[f][c + 1]]) as f32 / 65535.0;
            }
            stats.mip_count = (res as f32).log2() as u32 + 1;
            Some(stats)
        };
        #[cfg(test)]
        let test_layers = {
            Some(BakeTestLayers {
                albedo: tiles[0].clone(),
                normal_ao: tiles[1].clone(),
                depth: dep_tiles
                    .iter()
                    .map(|t| {
                        t.chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect()
                    })
                    .collect(),
            })
        };

        let mut it = images.into_iter();
        let albedo = it.next().unwrap();
        let normal_ao = it.next().unwrap();
        let mr = it.next().unwrap();
        let emissive = it.next().unwrap();
        let depth = it.next().unwrap();
        let material_res = settings.material_resolution.clamp(1, 1024);
        let (material_maps, material_table, material_count, material_bytes) =
            material_gpu_data(&self.ctx, &materials, material_res);

        let bytes = [4usize, 4, 4, 4, 2]
            .iter()
            .map(|b| {
                let mut level = res;
                (0..mips)
                    .map(|_| {
                        let v = (level * level) as usize * b * VIEW_COUNT;
                        level = (level / 2).max(1);
                        v
                    })
                    .sum::<usize>()
            })
            .sum::<usize>()
            + voxel_bytes
            + material_bytes;

        let dep_u16: Vec<Vec<u16>> = dep_tiles
            .iter()
            .map(|t| {
                t.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            })
            .collect();
        let previews = build_previews(&tiles[0], &tiles[1], &tiles[2], &tiles[3], &dep_u16, res);

        Ok(ImpostorAsset {
            bounds,
            resolution: res,
            slab_valid,
            voxel_resolution: settings.voxel_resolution,
            triangle_base,
            has_mask_materials: has_mask,
            triangle_count,
            surface_count,
            material_resolution: material_res,
            material_count,
            bytes,
            bake_ms: start.elapsed().as_secs_f32() * 1000.0,
            previews,
            #[cfg(test)]
            test_stats: test_stats_inner,
            #[cfg(test)]
            test_layers,
            albedo,
            normal_ao,
            metallic_rough: mr,
            emissive,
            depth,
            voxels,
            material_maps,
            material_table,
            ctx: Arc::clone(&self.ctx),
            drop_without_wait: false,
        })
    }

    fn bake_material(&self, m: &Material) -> BakeMaterial {
        let tex_or_default = |t: Option<&TextureRef>,
                              srgb: bool,
                              default_px: [u8; 4]|
         -> Arc<GpuTexture> {
            let fmt = if srgb {
                TextureFormat::Srgb
            } else {
                TextureFormat::Linear
            };
            match t {
                Some(r) => {
                    let d = &r.texture;
                    let opts = TextureOptions {
                        format: fmt,
                        sampler: crate::texture::SamplerDesc {
                            filter: match r.mag_filter {
                                TexelFilter::Nearest => crate::texture::Filter::Nearest,
                                TexelFilter::Linear => crate::texture::Filter::Linear,
                            },
                            mip: crate::texture::MipFilter::Linear,
                            address: match r.wrap_s {
                                Wrap::Repeat => crate::texture::AddressMode::Repeat,
                                Wrap::ClampToEdge => crate::texture::AddressMode::ClampToEdge,
                                Wrap::MirroredRepeat => crate::texture::AddressMode::MirrorRepeat,
                            },
                            anisotropy: true,
                        },
                        ..TextureOptions::default()
                    };
                    Arc::new(GpuTexture::from_rgba8(
                        &self.ctx, "bake tex", &d.pixels, d.width, d.height, opts,
                    ))
                }
                None => {
                    let mut opts = TextureOptions::default();
                    opts.format = fmt;
                    Arc::new(GpuTexture::from_rgba8(
                        &self.ctx,
                        "bake default",
                        &default_px,
                        1,
                        1,
                        opts,
                    ))
                }
            }
        };
        let (uv0, tf0) = pack_transform(m.base_color_map.as_ref());
        let (uv1, tf1) = pack_transform(m.metallic_roughness_texture.as_ref());
        let (uv2, tf2) = pack_transform(m.normal_texture.as_ref());
        let (uv3, tf3) = pack_transform(m.occlusion_texture.as_ref());
        let (uv4, tf4) = pack_transform(m.emissive_texture.as_ref());
        let factors_data = CaptureFactors {
            params: [
                m.metallic_factor,
                m.roughness_factor,
                m.alpha_cutoff,
                match m.alpha_mode {
                    AlphaMode::Opaque => 0.0,
                    AlphaMode::Mask => 1.0,
                    AlphaMode::Blend => 2.0,
                },
            ],
            params2: [m.occlusion_strength, m.normal_scale, 0.0, 0.0],
            emissive: [
                m.emissive_factor[0],
                m.emissive_factor[1],
                m.emissive_factor[2],
                m.emissive_strength,
            ],
            uv: [uv0, uv1, uv2, uv3, uv4],
            tf: [tf0, tf1, tf2, tf3, tf4],
        };
        let factors = self
            .ctx
            .create_uniform_buffer("bake factors", bytemuck::bytes_of(&factors_data));
        let base_color = self
            .ctx
            .create_uniform_buffer("bake base_color", bytemuck::bytes_of(&m.base_color));
        BakeMaterial {
            factors,
            base_color,
            base: tex_or_default(m.base_color_map.as_ref(), true, [255, 255, 255, 255]),
            mr: tex_or_default(
                m.metallic_roughness_texture.as_ref(),
                false,
                [255, 255, 255, 255],
            ),
            normal: tex_or_default(m.normal_texture.as_ref(), false, [128, 128, 255, 255]),
            ao: tex_or_default(m.occlusion_texture.as_ref(), false, [255, 255, 255, 255]),
            emissive: tex_or_default(m.emissive_texture.as_ref(), true, [0, 0, 0, 255]),
            base_alpha: m.base_color[3],
        }
    }

    fn capture_views_gpu(
        &self,
        batches: &[BakeBatch],
        cap: u32,
        mvps: &[Mat4; VIEW_COUNT],
        face_ns: &[Vec3; VIEW_COUNT],
        center: Vec3,
        exts: &[[f32; 3]; VIEW_COUNT],
    ) -> [GpuImage; 5] {
        let caps = [
            self.ctx.create_texture_2d_array(
                "cap alb",
                cap,
                cap,
                VIEW_COUNT as u32,
                1,
                MTLPixelFormat::RGBA16Float,
                true,
            ),
            self.ctx.create_texture_2d_array(
                "cap nrm",
                cap,
                cap,
                VIEW_COUNT as u32,
                1,
                MTLPixelFormat::RGBA16Float,
                true,
            ),
            self.ctx.create_texture_2d_array(
                "cap mrv",
                cap,
                cap,
                VIEW_COUNT as u32,
                1,
                MTLPixelFormat::RGBA16Float,
                true,
            ),
            self.ctx.create_texture_2d_array(
                "cap emi",
                cap,
                cap,
                VIEW_COUNT as u32,
                1,
                MTLPixelFormat::RGBA16Float,
                true,
            ),
            self.ctx.create_texture_2d_array(
                "cap dep",
                cap,
                cap,
                VIEW_COUNT as u32,
                1,
                MTLPixelFormat::R16Float,
                true,
            ),
        ];
        let depth_tex = self
            .ctx
            .create_depth_image(cap, cap, MTLPixelFormat::Depth32Float);

        self.ctx.one_shot(|cmd| {
            for f in 0..VIEW_COUNT {
                let rpd = MTLRenderPassDescriptor::new();
                for (i, c) in caps.iter().enumerate() {
                    let att = unsafe { rpd.colorAttachments().objectAtIndexedSubscript(i) };
                    att.setTexture(Some(&c.texture));
                    att.setSlice(f);
                    att.setLoadAction(MTLLoadAction::Clear);
                    att.setStoreAction(MTLStoreAction::Store);
                    if i == 4 {
                        att.setClearColor(objc2_metal::MTLClearColor {
                            red: 1.0,
                            green: 1.0,
                            blue: 1.0,
                            alpha: 1.0,
                        });
                    } else {
                        att.setClearColor(objc2_metal::MTLClearColor {
                            red: 0.0,
                            green: 0.0,
                            blue: 0.0,
                            alpha: 0.0,
                        });
                    }
                }
                let d_att = rpd.depthAttachment();
                d_att.setTexture(Some(&depth_tex.texture));
                d_att.setLoadAction(MTLLoadAction::Clear);
                d_att.setStoreAction(MTLStoreAction::DontCare);
                d_att.setClearDepth(1.0);

                let enc = cmd.renderCommandEncoderWithDescriptor(&rpd).expect("capture enc");
                enc.setRenderPipelineState(&self.capture_pipe);
                enc.setDepthStencilState(Some(&self.capture_depth_state));
                enc.setViewport(MTLViewport {
                    originX: 0.0,
                    originY: 0.0,
                    width: cap as f64,
                    height: cap as f64,
                    znear: 0.0,
                    zfar: 1.0,
                });
                enc.setFrontFacingWinding(MTLWinding::Clockwise);

                for batch in batches {
                    let m = &batch.material;
                    enc.setCullMode(MTLCullMode::None);
                    let push = CapturePush {
                        mvp: mvps[f].to_cols_array_2d(),
                        face_n: face_ns[f].extend(0.0).to_array(),
                        box_center: center.extend(0.0).to_array(),
                        extents: [exts[f][0], exts[f][1], exts[f][2], m.base_alpha],
                    };
                    let push_bytes = bytemuck::bytes_of(&push);
                    unsafe {
                        enc.setVertexBuffer_offset_atIndex(Some(&batch.vertex_buffer.buffer), 0, VERTEX_BUFFER_SLOT as usize);
                        enc.setVertexBytes_length_atIndex(
                            NonNull::new_unchecked(push_bytes.as_ptr() as *mut _),
                            push_bytes.len(),
                            PUSH_CONSTANT_BUFFER_SLOT as usize,
                        );
                        enc.setFragmentBytes_length_atIndex(
                            NonNull::new_unchecked(push_bytes.as_ptr() as *mut _),
                            push_bytes.len(),
                            PUSH_CONSTANT_BUFFER_SLOT as usize,
                        );
                        enc.setFragmentBuffer_offset_atIndex(Some(&m.factors.buffer), 0, 0);
                        enc.setFragmentBuffer_offset_atIndex(Some(&m.base_color.buffer), 0, 1);
                        enc.setFragmentTexture_atIndex(Some(&m.base.view), 0);
                        enc.setFragmentTexture_atIndex(Some(&m.mr.view), 1);
                        enc.setFragmentTexture_atIndex(Some(&m.normal.view), 2);
                        enc.setFragmentTexture_atIndex(Some(&m.ao.view), 3);
                        enc.setFragmentTexture_atIndex(Some(&m.emissive.view), 4);
                        enc.setFragmentSamplerState_atIndex(Some(&m.base.sampler), 0);
                        enc.setFragmentSamplerState_atIndex(Some(&m.mr.sampler), 1);
                        enc.setFragmentSamplerState_atIndex(Some(&m.normal.sampler), 2);
                        enc.setFragmentSamplerState_atIndex(Some(&m.ao.sampler), 3);
                        enc.setFragmentSamplerState_atIndex(Some(&m.emissive.sampler), 4);
                        enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            MTLPrimitiveType::Triangle,
                            batch.index_count as usize,
                            MTLIndexType::UInt32,
                            &batch.index_buffer.buffer,
                            0,
                        );
                    }
                }
                enc.endEncoding();
            }
        });

        caps
    }

    fn gpu_post(
        &self,
        caps: &[GpuImage; 5],
        res: u32,
        ss: u32,
    ) -> (
        [GpuImage; 5],
        [Vec<Vec<u8>>; 4],
        Vec<Vec<u8>>,
        [f32; VIEW_COUNT],
    ) {
        let mips = 32 - res.leading_zeros();
        let mut level_edges = Vec::with_capacity(mips as usize);
        let mut level_texels = Vec::with_capacity(mips as usize);
        let mut total_texels = 0u32;
        let mut edge = res;
        for _ in 0..mips {
            level_edges.push(edge);
            level_texels.push(total_texels);
            total_texels += VIEW_COUNT as u32 * edge * edge;
            edge = (edge / 2).max(1);
        }

        let chain_vec4_bytes = (total_texels as u64 * 16).max(16);
        let chain_f32_bytes = (total_texels as u64 * 4).max(16);
        let l0_vec4_bytes = (VIEW_COUNT as u64 * res as u64 * res as u64 * 16).max(16);

        let chain_alb = self
            .ctx
            .create_host_storage_buffer("post chain alb", chain_vec4_bytes);
        let chain_nrm = self
            .ctx
            .create_host_storage_buffer("post chain nrm", chain_vec4_bytes);
        let chain_mrv = self
            .ctx
            .create_host_storage_buffer("post chain mrv", chain_vec4_bytes);
        let chain_emi = self
            .ctx
            .create_host_storage_buffer("post chain emi", chain_vec4_bytes);
        let chain_dep = self
            .ctx
            .create_host_storage_buffer("post chain dep", chain_f32_bytes);
        let tmp_alb = self
            .ctx
            .create_host_storage_buffer("post tmp alb", l0_vec4_bytes);
        let tmp_nrm = self
            .ctx
            .create_host_storage_buffer("post tmp nrm", l0_vec4_bytes);
        let tmp_mrv = self
            .ctx
            .create_host_storage_buffer("post tmp mrv", l0_vec4_bytes);
        let tmp_emi = self
            .ctx
            .create_host_storage_buffer("post tmp emi", l0_vec4_bytes);
        let mut cov_buf = self
            .ctx
            .create_host_storage_buffer("post cov", (VIEW_COUNT * 4) as u64);
        cov_buf.write_bytes(0, &[0u8; VIEW_COUNT * 4]);

        let pack_u32_bytes = (total_texels as u64 * 4).max(16);
        let pack_dep_bytes = (total_texels as u64 * 2).max(16);
        let pack_alb = self.ctx.create_readback_buffer("pack alb", pack_u32_bytes);
        let pack_nrm = self.ctx.create_readback_buffer("pack nrm", pack_u32_bytes);
        let pack_mrv = self.ctx.create_readback_buffer("pack mrv", pack_u32_bytes);
        let pack_emi = self.ctx.create_readback_buffer("pack emi", pack_u32_bytes);
        let pack_dep = self.ctx.create_readback_buffer("pack dep", pack_dep_bytes);

        let out_imgs = [
            self.ctx.create_texture_2d_array(
                "imp alb",
                res,
                res,
                VIEW_COUNT as u32,
                mips,
                MTLPixelFormat::RGBA8Unorm_sRGB,
                false,
            ),
            self.ctx.create_texture_2d_array(
                "imp nrm",
                res,
                res,
                VIEW_COUNT as u32,
                mips,
                MTLPixelFormat::RGBA8Unorm,
                false,
            ),
            self.ctx.create_texture_2d_array(
                "imp mrv",
                res,
                res,
                VIEW_COUNT as u32,
                mips,
                MTLPixelFormat::RGBA8Unorm,
                false,
            ),
            self.ctx.create_texture_2d_array(
                "imp emi",
                res,
                res,
                VIEW_COUNT as u32,
                mips,
                MTLPixelFormat::RGBA8Unorm_sRGB,
                false,
            ),
            self.ctx.create_texture_2d_array(
                "imp dep",
                res,
                res,
                VIEW_COUNT as u32,
                mips,
                MTLPixelFormat::R16Unorm,
                false,
            ),
        ];

        self.ctx.one_shot(|cmd| {
            // 1. Downsample -> chain[0]
            {
                let enc = cmd.computeCommandEncoder().expect("down enc");
                enc.setComputePipelineState(&self.cs_down);
                let p = DParams {
                    res,
                    ss,
                    src_off: 0,
                    dst_off: 0,
                };
                let pb = bytemuck::bytes_of(&p);
                unsafe {
                    enc.setTexture_atIndex(Some(&caps[0].texture), 0);
                    enc.setTexture_atIndex(Some(&caps[1].texture), 1);
                    enc.setTexture_atIndex(Some(&caps[2].texture), 2);
                    enc.setTexture_atIndex(Some(&caps[3].texture), 3);
                    enc.setTexture_atIndex(Some(&caps[4].texture), 4);
                    enc.setBuffer_offset_atIndex(Some(&chain_alb.buffer), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&chain_nrm.buffer), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&chain_mrv.buffer), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&chain_emi.buffer), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&chain_dep.buffer), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&cov_buf.buffer), 0, 5);
                    enc.setBytes_length_atIndex(
                        NonNull::new_unchecked(pb.as_ptr() as *mut _),
                        pb.len(),
                        PUSH_CONSTANT_BUFFER_SLOT as usize,
                    );
                }
                let tg = MTLSize { width: 8, height: 8, depth: 1 };
                let grid = MTLSize {
                    width: res.div_ceil(8) as usize,
                    height: res.div_ceil(8) as usize,
                    depth: VIEW_COUNT,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
            }

            // 2. Dilate (4 passes ping-pong on level 0: even count lands back in chain)
            for pass in 0..4 {
                let (s_a, s_n, s_m, s_e, d_a, d_n, d_m, d_e) = if pass % 2 == 0 {
                    (&chain_alb, &chain_nrm, &chain_mrv, &chain_emi, &tmp_alb, &tmp_nrm, &tmp_mrv, &tmp_emi)
                } else {
                    (&tmp_alb, &tmp_nrm, &tmp_mrv, &tmp_emi, &chain_alb, &chain_nrm, &chain_mrv, &chain_emi)
                };
                let enc = cmd.computeCommandEncoder().expect("dilate enc");
                enc.setComputePipelineState(&self.cs_dilate);
                let p = DParams {
                    res,
                    ss: 0,
                    src_off: 0,
                    dst_off: 0,
                };
                let pb = bytemuck::bytes_of(&p);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&s_a.buffer), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&s_n.buffer), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&s_m.buffer), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&s_e.buffer), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&d_a.buffer), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&d_n.buffer), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(&d_m.buffer), 0, 6);
                    enc.setBuffer_offset_atIndex(Some(&d_e.buffer), 0, 7);
                    enc.setBytes_length_atIndex(
                        NonNull::new_unchecked(pb.as_ptr() as *mut _),
                        pb.len(),
                        PUSH_CONSTANT_BUFFER_SLOT as usize,
                    );
                }
                let tg = MTLSize { width: 8, height: 8, depth: 1 };
                let grid = MTLSize {
                    width: res.div_ceil(8) as usize,
                    height: res.div_ceil(8) as usize,
                    depth: VIEW_COUNT,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
            }

            // 3. Mipgen
            for m in 1..(mips as usize) {
                let dr = level_edges[m];
                let sr = level_edges[m - 1];
                let enc = cmd.computeCommandEncoder().expect("mip enc");
                enc.setComputePipelineState(&self.cs_mip);
                let p = DParams {
                    res: dr,
                    ss: sr,
                    src_off: level_texels[m - 1],
                    dst_off: level_texels[m],
                };
                let pb = bytemuck::bytes_of(&p);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&chain_alb.buffer), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&chain_nrm.buffer), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&chain_mrv.buffer), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&chain_emi.buffer), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&chain_dep.buffer), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&chain_alb.buffer), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(&chain_nrm.buffer), 0, 6);
                    enc.setBuffer_offset_atIndex(Some(&chain_mrv.buffer), 0, 7);
                    enc.setBuffer_offset_atIndex(Some(&chain_emi.buffer), 0, 8);
                    enc.setBuffer_offset_atIndex(Some(&chain_dep.buffer), 0, 9);
                    enc.setBytes_length_atIndex(
                        NonNull::new_unchecked(pb.as_ptr() as *mut _),
                        pb.len(),
                        PUSH_CONSTANT_BUFFER_SLOT as usize,
                    );
                }
                let tg = MTLSize { width: 8, height: 8, depth: 1 };
                let grid = MTLSize {
                    width: dr.div_ceil(8) as usize,
                    height: dr.div_ceil(8) as usize,
                    depth: VIEW_COUNT,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
            }

            // 4. Pack each mip level
            for m in 0..(mips as usize) {
                let r = level_edges[m];
                let off = level_texels[m];
                let enc = cmd.computeCommandEncoder().expect("pack enc");
                enc.setComputePipelineState(&self.cs_pack);
                let p = DParams {
                    res: r,
                    ss: m as u32,
                    src_off: off,
                    dst_off: off,
                };
                let pb = bytemuck::bytes_of(&p);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&chain_alb.buffer), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&chain_nrm.buffer), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&chain_mrv.buffer), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&chain_emi.buffer), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&chain_dep.buffer), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&pack_alb.buffer), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(&pack_nrm.buffer), 0, 6);
                    enc.setBuffer_offset_atIndex(Some(&pack_mrv.buffer), 0, 7);
                    enc.setBuffer_offset_atIndex(Some(&pack_emi.buffer), 0, 8);
                    enc.setBuffer_offset_atIndex(Some(&pack_dep.buffer), 0, 9);
                    enc.setBytes_length_atIndex(
                        NonNull::new_unchecked(pb.as_ptr() as *mut _),
                        pb.len(),
                        PUSH_CONSTANT_BUFFER_SLOT as usize,
                    );
                }
                let tg = MTLSize { width: 8, height: 8, depth: 1 };
                let grid = MTLSize {
                    width: r.div_ceil(8) as usize,
                    height: r.div_ceil(8) as usize,
                    depth: VIEW_COUNT,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
            }

            // 5. Blit packed buffers into 2DArray mip levels
            let blit = cmd.blitCommandEncoder().expect("pack blit");
            let packs = [&pack_alb, &pack_nrm, &pack_mrv, &pack_emi, &pack_dep];
            let bpps = [4usize, 4, 4, 4, 2];
            for (img_idx, (pack_buf, &bpp)) in packs.iter().zip(bpps.iter()).enumerate() {
                for m in 0..(mips as usize) {
                    let r = level_edges[m];
                    let base_texel = level_texels[m] as usize;
                    let layer_bytes = (r * r) as usize * bpp;
                    for layer in 0..VIEW_COUNT {
                        let off = base_texel * bpp + layer * layer_bytes;
                        unsafe {
                            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                                &pack_buf.buffer,
                                off,
                                r as usize * bpp,
                                layer_bytes,
                                MTLSize {
                                    width: r as usize,
                                    height: r as usize,
                                    depth: 1,
                                },
                                &out_imgs[img_idx].texture,
                                layer,
                                m,
                                MTLOrigin { x: 0, y: 0, z: 0 },
                            );
                        }
                    }
                }
            }
            blit.endEncoding();
        });

        let cov_words: &[u32] = bytemuck::cast_slice(cov_buf.read_bytes(VIEW_COUNT * 4));
        let cap_px = (res * ss * res * ss) as f32;
        let mut coverage_frac = [0.0f32; VIEW_COUNT];
        for (i, &c) in cov_words.iter().enumerate() {
            coverage_frac[i] = c as f32 / cap_px.max(1.0);
        }

        let l0_rgba_len = (res * res * 4) as usize;
        let l0_dep_len = (res * res * 2) as usize;
        let alb_bytes = pack_alb.read_bytes(VIEW_COUNT * l0_rgba_len);
        let nrm_bytes = pack_nrm.read_bytes(VIEW_COUNT * l0_rgba_len);
        let mrv_bytes = pack_mrv.read_bytes(VIEW_COUNT * l0_rgba_len);
        let emi_bytes = pack_emi.read_bytes(VIEW_COUNT * l0_rgba_len);
        let dep_bytes = pack_dep.read_bytes(VIEW_COUNT * l0_dep_len);

        let split_layers = |src: &[u8], stride: usize| -> Vec<Vec<u8>> {
            (0..VIEW_COUNT)
                .map(|f| src[f * stride..(f + 1) * stride].to_vec())
                .collect()
        };
        let tiles = [
            split_layers(alb_bytes, l0_rgba_len),
            split_layers(nrm_bytes, l0_rgba_len),
            split_layers(mrv_bytes, l0_rgba_len),
            split_layers(emi_bytes, l0_rgba_len),
        ];
        let dep_tiles = split_layers(dep_bytes, l0_dep_len);

        (out_imgs, tiles, dep_tiles, coverage_frac)
    }
}

fn wrap_index(i: i32, len: u32, wrap: Wrap) -> usize {
    let len = len.max(1) as i32;
    let i = match wrap {
        Wrap::Repeat => i.rem_euclid(len),
        Wrap::ClampToEdge => i.clamp(0, len - 1),
        Wrap::MirroredRepeat => {
            let period = len * 2;
            let r = i.rem_euclid(period);
            if r < len { r } else { period - 1 - r }
        }
    };
    i as usize
}

fn srgb_channel_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_channel_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn sample_decoded(
    texture: &crate::texture::DecodedTexture,
    uv: Vec2,
    wrap_s: Wrap,
    wrap_t: Wrap,
    filter: TexelFilter,
    linear_rgb: bool,
) -> [f32; 4] {
    if texture.width == 0 || texture.height == 0 {
        return [1.0; 4];
    }
    let px_at = |x: i32, y: i32| -> [f32; 4] {
        let xi = wrap_index(x, texture.width, wrap_s);
        let yi = wrap_index(y, texture.height, wrap_t);
        let i = (yi * texture.width as usize + xi) * 4;
        let mut p = [
            texture.pixels[i] as f32 / 255.0,
            texture.pixels[i + 1] as f32 / 255.0,
            texture.pixels[i + 2] as f32 / 255.0,
            texture.pixels[i + 3] as f32 / 255.0,
        ];
        if linear_rgb {
            for c in &mut p[..3] {
                *c = srgb_channel_to_linear(*c);
            }
        }
        p
    };
    let fx = uv.x * texture.width as f32 - 0.5;
    let fy = uv.y * texture.height as f32 - 0.5;
    if matches!(filter, TexelFilter::Nearest) {
        return px_at(fx.round() as i32, fy.round() as i32);
    }
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let p00 = px_at(x0, y0);
    let p10 = px_at(x0 + 1, y0);
    let p01 = px_at(x0, y0 + 1);
    let p11 = px_at(x0 + 1, y0 + 1);
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let a = p00[c] * (1.0 - tx) + p10[c] * tx;
        let b = p01[c] * (1.0 - tx) + p11[c] * tx;
        out[c] = a * (1.0 - ty) + b * ty;
    }
    out
}

fn area_sample_decoded(
    texture: &crate::texture::DecodedTexture,
    x: u32,
    y: u32,
    dst_res: u32,
    wrap_s: Wrap,
    wrap_t: Wrap,
    linear_rgb: bool,
) -> [f32; 4] {
    let sx0 = x as f32 * texture.width as f32 / dst_res as f32;
    let sx1 = (x + 1) as f32 * texture.width as f32 / dst_res as f32;
    let sy0 = y as f32 * texture.height as f32 / dst_res as f32;
    let sy1 = (y + 1) as f32 * texture.height as f32 / dst_res as f32;
    let mut out = [0.0; 4];
    let mut weight = 0.0;
    let ix0 = sx0.floor() as i32;
    let ix1 = sx1.ceil() as i32 - 1;
    let iy0 = sy0.floor() as i32;
    let iy1 = sy1.ceil() as i32 - 1;
    for sy in iy0..=iy1 {
        let wy = (sy as f32 + 1.0).min(sy1) - (sy as f32).max(sy0);
        if wy <= 0.0 {
            continue;
        }
        let yi = wrap_index(sy, texture.height, wrap_t);
        for sx in ix0..=ix1 {
            let wx = (sx as f32 + 1.0).min(sx1) - (sx as f32).max(sx0);
            let w = wx * wy;
            if w <= 0.0 {
                continue;
            }
            let xi = wrap_index(sx, texture.width, wrap_s);
            let i = (yi * texture.width as usize + xi) * 4;
            let mut px = [
                texture.pixels[i] as f32 / 255.0,
                texture.pixels[i + 1] as f32 / 255.0,
                texture.pixels[i + 2] as f32 / 255.0,
                texture.pixels[i + 3] as f32 / 255.0,
            ];
            if linear_rgb {
                for c in &mut px[..3] {
                    *c = srgb_channel_to_linear(*c);
                }
            }
            for c in 0..4 {
                out[c] += px[c] * w;
            }
            weight += w;
        }
    }
    if weight > 0.0 {
        for c in &mut out {
            *c /= weight;
        }
    }
    out
}

fn resample_texture_layer(
    tex: Option<&TextureRef>,
    res: u32,
    srgb: bool,
    default_px: [u8; 4],
) -> Vec<u8> {
    let mut out = vec![0u8; (res * res * 4) as usize];
    let Some(tr) = tex else {
        for px in out.chunks_exact_mut(4) {
            px.copy_from_slice(&default_px);
        }
        return out;
    };
    let dec = &tr.texture;
    let downsample = dec.width > res || dec.height > res;
    for y in 0..res {
        for x in 0..res {
            let mut c = if downsample {
                area_sample_decoded(dec, x, y, res, tr.wrap_s, tr.wrap_t, srgb)
            } else {
                let uv = Vec2::new((x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32);
                sample_decoded(dec, uv, tr.wrap_s, tr.wrap_t, tr.min_filter, srgb)
            };
            if srgb {
                for ch in &mut c[..3] {
                    *ch = linear_channel_to_srgb(*ch);
                }
            }
            let o = ((y * res + x) * 4) as usize;
            out[o] = (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out[o + 1] = (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out[o + 2] = (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out[o + 3] = (c[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    out
}

fn downsample_rgba8_2x2(src: &[u8], src_res: u32, srgb: bool) -> Vec<u8> {
    let dst_res = (src_res / 2).max(1);
    let mut dst = vec![0u8; (dst_res * dst_res * 4) as usize];
    for y in 0..dst_res {
        for x in 0..dst_res {
            let mut sum = [0.0f32; 4];
            for dy in 0..2 {
                let sy = (y * 2 + dy).min(src_res - 1);
                for dx in 0..2 {
                    let sx = (x * 2 + dx).min(src_res - 1);
                    let i = ((sy * src_res + sx) * 4) as usize;
                    for c in 0..4 {
                        let v = src[i + c] as f32 / 255.0;
                        sum[c] += if srgb && c < 3 {
                            srgb_channel_to_linear(v)
                        } else {
                            v
                        };
                    }
                }
            }
            let o = ((y * dst_res + x) * 4) as usize;
            for c in 0..4 {
                let mut v = sum[c] * 0.25;
                if srgb && c < 3 {
                    v = linear_channel_to_srgb(v);
                }
                dst[o + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    dst
}

fn upload_array(
    ctx: &Arc<GpuContext>,
    name: &str,
    format: MTLPixelFormat,
    res: u32,
    layers: &[Vec<u8>],
) -> GpuImage {
    let layer_count = layers.len().max(1) as u32;
    let mips = 32 - res.max(1).leading_zeros();
    let srgb = format == MTLPixelFormat::RGBA8Unorm_sRGB;
    let img = ctx.create_texture_2d_array(name, res, res, layer_count, mips, format, false);

    let mut per_layer_mips: Vec<Vec<(u32, Vec<u8>)>> = Vec::with_capacity(layers.len());
    let mut total = 0usize;
    for l in layers {
        let mut chain = Vec::with_capacity(mips as usize);
        let mut edge = res;
        let mut cur = l.clone();
        total += cur.len();
        chain.push((edge, cur.clone()));
        for _ in 1..mips {
            cur = downsample_rgba8_2x2(&cur, edge, srgb);
            edge = (edge / 2).max(1);
            total += cur.len();
            chain.push((edge, cur.clone()));
        }
        per_layer_mips.push(chain);
    }

    let mut staging = ctx.create_upload_buffer("mat map upload", total.max(4));
    let mut offsets = Vec::new();
    let mut off = 0usize;
    for chain in &per_layer_mips {
        for (_, data) in chain {
            staging.write_bytes(off as u64, data);
            offsets.push(off);
            off += data.len();
        }
    }
    ctx.one_shot(|cmd| {
        let blit = cmd.blitCommandEncoder().expect("mat map blit");
        let mut idx = 0usize;
        for (i, chain) in per_layer_mips.iter().enumerate() {
            for (mip, (edge, data)) in chain.iter().enumerate() {
                let src_off = offsets[idx];
                idx += 1;
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &staging.buffer,
                        src_off,
                        *edge as usize * 4,
                        data.len(),
                        MTLSize {
                            width: *edge as usize,
                            height: *edge as usize,
                            depth: 1,
                        },
                        &img.texture,
                        i,
                        mip,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
                }
            }
        }
        blit.endEncoding();
    });
    img
}

fn pack_wrap_filter(tex: Option<&TextureRef>) -> f32 {
    let Some(t) = tex else {
        return 0.0;
    };
    let w = |wrap: Wrap| -> u32 {
        match wrap {
            Wrap::Repeat => 0,
            Wrap::ClampToEdge => 1,
            Wrap::MirroredRepeat => 2,
        }
    };
    let f = match t.min_filter {
        TexelFilter::Linear => 0,
        TexelFilter::Nearest => 1,
    };
    (w(t.wrap_s) | (w(t.wrap_t) << 2) | (f << 4)) as f32
}

fn material_gpu_data(
    ctx: &Arc<GpuContext>,
    materials: &[&Material],
    mat_res: u32,
) -> ([GpuImage; 5], GpuBuffer, u32, usize) {
    let count = materials.len().max(1);
    let mut layer_data: [Vec<Vec<u8>>; 5] = Default::default();
    let mut params = Vec::with_capacity(count);
    for m in materials {
        layer_data[0].push(resample_texture_layer(
            m.base_color_map.as_ref(),
            mat_res,
            true,
            [255, 255, 255, 255],
        ));
        layer_data[1].push(resample_texture_layer(
            m.metallic_roughness_texture.as_ref(),
            mat_res,
            false,
            [255, 255, 255, 255],
        ));
        layer_data[2].push(resample_texture_layer(
            m.normal_texture.as_ref(),
            mat_res,
            false,
            [128, 128, 255, 255],
        ));
        layer_data[3].push(resample_texture_layer(
            m.occlusion_texture.as_ref(),
            mat_res,
            false,
            [255, 255, 255, 255],
        ));
        layer_data[4].push(resample_texture_layer(
            m.emissive_texture.as_ref(),
            mat_res,
            true,
            [0, 0, 0, 255],
        ));
        let (uv0, mut tf0) = pack_transform(m.base_color_map.as_ref());
        let (uv1, mut tf1) = pack_transform(m.metallic_roughness_texture.as_ref());
        let (uv2, mut tf2) = pack_transform(m.normal_texture.as_ref());
        let (uv3, mut tf3) = pack_transform(m.occlusion_texture.as_ref());
        let (uv4, mut tf4) = pack_transform(m.emissive_texture.as_ref());
        tf0[3] = pack_wrap_filter(m.base_color_map.as_ref());
        tf1[3] = pack_wrap_filter(m.metallic_roughness_texture.as_ref());
        tf2[3] = pack_wrap_filter(m.normal_texture.as_ref());
        tf3[3] = pack_wrap_filter(m.occlusion_texture.as_ref());
        tf4[3] = pack_wrap_filter(m.emissive_texture.as_ref());
        params.push(ImpMaterialParams {
            base_color: m.base_color,
            params: [
                m.metallic_factor,
                m.roughness_factor,
                m.alpha_cutoff,
                match m.alpha_mode {
                    AlphaMode::Opaque => 0.0,
                    AlphaMode::Mask => 1.0,
                    AlphaMode::Blend => 2.0,
                },
            ],
            params2: [
                m.occlusion_strength,
                m.normal_scale,
                if m.double_sided { 1.0 } else { 0.0 },
                0.0,
            ],
            emissive: [
                m.emissive_factor[0],
                m.emissive_factor[1],
                m.emissive_factor[2],
                m.emissive_strength,
            ],
            uv: [uv0, uv1, uv2, uv3, uv4],
            tf: [tf0, tf1, tf2, tf3, tf4],
        });
    }
    if materials.is_empty() {
        let defaults = [
            [255u8, 255, 255, 255],
            [255, 255, 255, 255],
            [128, 128, 255, 255],
            [255, 255, 255, 255],
            [0, 0, 0, 255],
        ];
        for (d, px) in layer_data.iter_mut().zip(defaults) {
            let mut v = vec![0u8; (mat_res * mat_res * 4) as usize];
            for c in v.chunks_exact_mut(4) {
                c.copy_from_slice(&px);
            }
            d.push(v);
        }
        params.push(ImpMaterialParams::default());
    }
    let formats = [
        MTLPixelFormat::RGBA8Unorm_sRGB,
        MTLPixelFormat::RGBA8Unorm,
        MTLPixelFormat::RGBA8Unorm,
        MTLPixelFormat::RGBA8Unorm,
        MTLPixelFormat::RGBA8Unorm_sRGB,
    ];
    let maps = std::array::from_fn(|i| {
        upload_array(ctx, "imp mat map", formats[i], mat_res, &layer_data[i])
    });
    let table_bytes = params.len() * std::mem::size_of::<ImpMaterialParams>();
    let table = ctx.create_storage_buffer("imp mat table", bytemuck::cast_slice(&params));
    let mut mip_extent = mat_res;
    let mut per_layer_bytes = 0usize;
    while mip_extent > 0 {
        per_layer_bytes += mip_extent as usize * mip_extent as usize * 4;
        mip_extent /= 2;
    }
    let map_bytes = 5 * per_layer_bytes * count;
    (maps, table, materials.len() as u32, map_bytes + table_bytes)
}
