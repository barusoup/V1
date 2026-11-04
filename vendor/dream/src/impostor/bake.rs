//! GPU bake: 26 orthographic captures into the asset's channel arrays.
//!
//! Pipeline per view: the manually selected meshes (node transforms applied,
//! merged per material) render at `resolution * supersample` into five MRT
//! targets — albedo, object normal + AO, metallic-roughness, emissive and
//! normalized slab depth — with a real depth buffer for occlusion.
//! `bake_post.wgsl` then performs coverage-weighted downsample, dilation and
//! mip generation on the GPU and packs the channel arrays without a CPU
//! pixel path. A triangle-surface voxel grid and material-indexed source-map
//! arrays are also built for the far-field exact-material path.

use super::atlas::{ImpostorAsset, ImpostorAssetLimits};
use super::{FallbackReason, ImpostorBounds, VIEW_COUNT};
#[cfg(test)]
use super::{VIEWS, projected_extent};
use crate::gfx::GpuBuffer;
#[cfg(test)]
use crate::gfx::GpuImage;
use crate::gfx::vulkan::GpuContext;
use crate::model::render::{PbrMaterialGpu, PbrVertex};
use crate::model::{Material, Model};
#[cfg(test)]
use crate::model::{TexelFilter, Wrap};
use ash::vk;
use glam::Mat4;
#[cfg(test)]
use glam::{Vec2, Vec3};
use std::sync::Arc;

mod capture;
mod geometry;
mod materials;
mod post;
mod previews;
mod sources;
mod upload;

#[cfg(test)]
use geometry::triangle_record;
use geometry::{TRIANGLE_WORDS, check_ref_lists, voxelize_batches};
#[cfg(test)]
use materials::material_layer;
#[cfg(test)]
use sources::merge_by_material;
use sources::prepare_sources;
use upload::image_bytes;

/// Bake knobs.
#[derive(Clone, Copy, Debug)]
pub struct BakeSettings {
    /// Tile resolution per view (default 512).
    pub resolution: u32,
    /// Supersample factor for the captures (default 2 → 1024² renders).
    pub supersample: u32,
    /// Surface-voxel grid resolution; 0 disables that refinement (default 128).
    pub voxel_resolution: u32,
    /// Per-material source-map array edge used by the exact far-field tier.
    pub material_resolution: u32,
}
impl BakeSettings {
    /// Smaller preset — the default 512 settings stay the qualified config.
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

/// One contribution to a manually specified impostor bake.
///
/// `ImpostorSource::model` captures every mesh in model space;
/// `ImpostorSource::meshes` captures an explicit mesh-index subset. Multiple
/// sources may be passed to [`ImpostorBaker::bake_sources`] to bake a group of
/// meshes/models into one asset. `transform` is applied after the model's own
/// rest-node transform, so callers can place each source in the group's shared
/// bake space.
pub struct ImpostorSource<'a> {
    model: &'a Model,
    meshes: Option<&'a [usize]>,
    transform: Mat4,
}

impl<'a> ImpostorSource<'a> {
    /// Capture all static meshes in `model` using their rest-node placement.
    pub fn model(model: &'a Model) -> Self {
        Self {
            model,
            meshes: None,
            transform: Mat4::IDENTITY,
        }
    }

    /// Capture only `mesh_indices` from `model` using their rest-node
    /// placement. Duplicate indices are ignored.
    pub fn meshes(model: &'a Model, mesh_indices: &'a [usize]) -> Self {
        Self {
            model,
            meshes: Some(mesh_indices),
            transform: Mat4::IDENTITY,
        }
    }

    /// Apply an additional transform around this source's bake-space position.
    /// This composes as `transform * mesh_node_transform`.
    pub fn transformed(mut self, transform: Mat4) -> Self {
        self.transform = transform;
        self
    }
}

/// Prepared bake input: grouped vertex/index batches plus the material set
/// they index into and the combined object-space bounds.
struct PreparedBake<'a> {
    batches: Vec<(Vec<PbrVertex>, Vec<u32>)>,
    materials: Vec<&'a Material>,
    bounds: ImpostorBounds,
}

/// Capture formats: RGBA16F channel targets + R16 depth + D32 depth buffer.
/// (R16_SFLOAT for the slab channel: float attachments are universally
/// renderable; the asset itself stores R16_UNORM.)
const CAPTURE_FORMATS: [vk::Format; 5] = [
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16_SFLOAT,
];
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// One material's bake descriptor sets: the PBR material set (same layout as
/// `model_pbr.wgsl` set 2) plus a tiny base-colour-factor uniform.
struct BakeMaterial {
    ctx: Arc<GpuContext>,
    pbr: PbrMaterialGpu,
    _base: GpuBuffer,
    pool: vk::DescriptorPool,
    base_set: vk::DescriptorSet,
}

/// Per-material record consumed by the exact voxel far-field path.
/// Layout mirrors `model_pbr.wgsl`, plus base colour (which lives in the
/// model draw block) and packed wrap/filter state in `tf[..].w`.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ImpMaterialParams {
    base_color: [f32; 4],
    /// metal, roughness, alpha cutoff, alpha mode.
    params: [f32; 4],
    /// AO strength, normal scale, double-sided flag.
    params2: [f32; 4],
    /// rgb factor, a strength.
    emissive: [f32; 4],
    /// Per-map (scale.xy, offset.xy): base, mr, normal, ao, emissive.
    uv: [[f32; 4]; 5],
    /// Per-map (cos, sin, uv set, packed wrap/filter).
    tf: [[f32; 4]; 5],
}

impl Drop for BakeMaterial {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device().device_wait_idle();
            self.ctx.device().destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// Level-0 layer data kept for the CPU-side consistency test.
#[cfg(test)]
pub struct BakeTestLayers {
    /// rgba8 level-0 tiles per face (sRGB-encoded albedo, alpha = coverage).
    pub albedo: Vec<Vec<u8>>,
    /// rgba8 level-0 tiles per face (normal xyz + AO alpha).
    pub normal_ao: Vec<Vec<u8>>,
    /// u16 depth level-0 tiles per face (0..65535).
    pub depth: Vec<Vec<u16>>,
}

/// Test-only bake sanity data, computed cheaply from the downsampled layers.
#[cfg(test)]
#[derive(Debug)]
pub struct BakeTestStats {
    /// Coverage fraction per face (0..1).
    pub coverage: [f32; VIEW_COUNT],
    /// Depth at the tile centre per face — for a solid object the nearest
    /// surface sits at the capture plane => ~0.
    pub center_depth: [f32; VIEW_COUNT],
    /// True if at least one uncovered texel received dilated colour.
    pub dilated: bool,
    /// Levels in each colour array's mip chain.
    pub mip_count: u32,
}

/// GPU resources for capture passes. Reusable across bakes (pipelines are
/// format-bound, not model-bound).
pub struct ImpostorBaker {
    ctx: Arc<GpuContext>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    material_dsl: vk::DescriptorSetLayout,
    base_dsl: vk::DescriptorSetLayout,
    post: PostPipes,
}

impl ImpostorBaker {
    /// Bake the model into an [`ImpostorAsset`], or say why it cannot be one.
    /// This is the whole-model convenience wrapper: every mesh and material in
    /// `model` must satisfy [`super::static_capture_supported`].
    pub fn bake(
        &self,
        model: &Arc<Model>,
        settings: BakeSettings,
    ) -> Result<ImpostorAsset, FallbackReason> {
        super::static_capture_supported(model)?;
        self.bake_sources(&[ImpostorSource::model(model)], settings)
    }

    /// [`Self::bake`] with explicit residency budgets.
    pub fn bake_with_limits(
        &self,
        model: &Arc<Model>,
        settings: BakeSettings,
        limits: ImpostorAssetLimits,
    ) -> Result<ImpostorAsset, FallbackReason> {
        super::static_capture_supported(model)?;
        self.bake_sources_with_limits(&[ImpostorSource::model(model)], settings, limits)
    }

    /// Bake an explicit mesh subset from one model into one asset.
    ///
    /// Indices refer to [`Model::meshes`]; the meshes keep their rest-node
    /// placement in the model. Unsupported or skinned meshes outside the
    /// selection do not prevent the selected group from baking.
    pub fn bake_meshes(
        &self,
        model: &Arc<Model>,
        mesh_indices: &[usize],
        settings: BakeSettings,
    ) -> Result<ImpostorAsset, FallbackReason> {
        self.bake_sources(&[ImpostorSource::meshes(model, mesh_indices)], settings)
    }

    /// [`Self::bake_meshes`] with explicit residency budgets.
    pub fn bake_meshes_with_limits(
        &self,
        model: &Arc<Model>,
        mesh_indices: &[usize],
        settings: BakeSettings,
        limits: ImpostorAssetLimits,
    ) -> Result<ImpostorAsset, FallbackReason> {
        self.bake_sources_with_limits(
            &[ImpostorSource::meshes(model, mesh_indices)],
            settings,
            limits,
        )
    }

    /// Bake one or more manually selected sources into a single asset.
    ///
    /// Each source contributes its selected static meshes in shared bake space.
    /// The resulting asset is still just an [`ImpostorAsset`]: game code chooses
    /// when to submit it through [`crate::impostor::ImpostorRenderer::draw`].
    pub fn bake_sources(
        &self,
        sources: &[ImpostorSource<'_>],
        settings: BakeSettings,
    ) -> Result<ImpostorAsset, FallbackReason> {
        self.bake_sources_with_limits(sources, settings, ImpostorAssetLimits::default())
    }

    /// `bake_sources` with explicit residency budgets. Every limit violation
    /// returns a typed error BEFORE any GPU allocation; nothing is silently
    /// clamped.
    pub fn bake_sources_with_limits(
        &self,
        sources: &[ImpostorSource<'_>],
        settings: BakeSettings,
        limits: ImpostorAssetLimits,
    ) -> Result<ImpostorAsset, FallbackReason> {
        // Settings contract first — no GPU touched on failure.
        let dev = unsafe {
            self.ctx
                .instance()
                .get_physical_device_properties(self.ctx.physical_device())
                .limits
        };
        let res = settings.resolution;
        let mat = settings.material_resolution;
        let vox = settings.voxel_resolution;
        let cap_dim = dev.max_image_dimension2_d;
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
        if VIEW_COUNT as u32 > dev.max_image_array_layers {
            return Err(FallbackReason::InvalidSettings(format!(
                "device supports {} array layers, atlas needs {VIEW_COUNT}",
                dev.max_image_array_layers
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
        if prepared.materials.len() as u32 > dev.max_image_array_layers {
            return Err(FallbackReason::InvalidSettings(format!(
                "{} materials exceed device array layers {}",
                prepared.materials.len(),
                dev.max_image_array_layers
            )));
        }

        // Estimated resident bytes BEFORE allocating anything GPU-side:
        // atlas mips + material arrays + dense voxel grid + material table.
        let mips = 32 - res.leading_zeros();
        let atlas_bytes: u64 = [4u64, 4, 4, 4, 2]
            .iter()
            .map(|bpp| image_bytes(res, mips, VIEW_COUNT as u64, *bpp))
            .sum();
        let material_count = prepared.materials.len().max(1) as u64;
        let map_bytes: u64 = image_bytes(mat, 32 - mat.leading_zeros(), material_count, 4) * 5;
        let dense_bytes = if vox == 0 {
            4
        } else {
            (vox as u64).pow(3) * 12
        };
        let table_bytes = material_count * 224;
        let estimate = atlas_bytes + map_bytes + dense_bytes + table_bytes;
        if estimate > limits.max_gpu_bytes {
            return Err(FallbackReason::ResourceLimit {
                requested: estimate,
                limit: limits.max_gpu_bytes,
            });
        }

        // Voxelize on CPU: the packed lists give exact sizes (and the
        // per-cell bound) before vertex buffers or captures allocate.
        let (voxel_cells, surface_count, triangle_records, triangle_refs) =
            voxelize_batches(&prepared.batches, prepared.bounds, vox);
        let triangle_count = (triangle_records.len() / TRIANGLE_WORDS) as u32;
        let triangle_base = voxel_cells.len() as u32 + triangle_refs.len() as u32;
        // Ref-list bound: the refs region is [dense, triangle_base); each
        // list header counts its refs.
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
        if voxel_bytes_exact as u64 > u64::from(self.ctx.max_storage_buffer_range())
            || table_bytes > u64::from(self.ctx.max_storage_buffer_range())
        {
            return Err(FallbackReason::ResourceLimit {
                requested: voxel_bytes_exact as u64,
                limit: u64::from(self.ctx.max_storage_buffer_range()),
            });
        }
        // gpu_post compute buffers must fit the storage range.
        {
            let l0 = VIEW_COUNT as u64 * res as u64 * res as u64;
            let mut chain_texels = 0u64;
            let mut r = res as u64;
            for _ in 0..mips {
                chain_texels += VIEW_COUNT as u64 * r * r;
                r = (r / 2).max(1);
                if r == 1 && mips > 0 {
                    // mips loop already covers all levels
                }
            }
            let biggest = (chain_texels * 16).max(l0 * 16);
            if biggest > u64::from(self.ctx.max_storage_buffer_range()) {
                return Err(FallbackReason::ResourceLimit {
                    requested: biggest,
                    limit: u64::from(self.ctx.max_storage_buffer_range()),
                });
            }
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
}

struct BakeBatch {
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    index_count: u32,
    material: BakeMaterial,
}

/// Compute pipelines for the GPU post-bake stages (downsample/dilate/mip/pack).
struct PostPipes {
    down: vk::Pipeline,
    down_layout: vk::PipelineLayout,
    down_dsl: vk::DescriptorSetLayout,
    dilate: vk::Pipeline,
    dilate_layout: vk::PipelineLayout,
    dilate_dsl: vk::DescriptorSetLayout,
    mip: vk::Pipeline,
    mip_layout: vk::PipelineLayout,
    mip_dsl: vk::DescriptorSetLayout,
    pack: vk::Pipeline,
    pack_layout: vk::PipelineLayout,
    pack_dsl: vk::DescriptorSetLayout,
}

#[cfg(test)]
/// The f32 channels of one captured view (5 x size²).
struct CapturedView {
    albedo: Vec<f32>,
    normal_ao: Vec<f32>,
    mr: Vec<f32>,
    emissive: Vec<f32>,
    depth: Vec<f32>,
}

#[cfg(test)]
/// Downsampled mip-0 channels for one layer (res² floats x4 + depth u16).
struct Downsampled {
    albedo: Vec<f32>,
    normal_ao: Vec<f32>,
    mr: Vec<f32>,
    emissive: Vec<f32>,
    depth_u16: Vec<u16>,
}

#[cfg(test)]
fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        return s * m * (2f32).powi(-24);
    }
    if e == 31 {
        return if m == 0.0 {
            s * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    s * (1.0 + m / 1024.0) * (2f32).powi(e - 15)
}

// ------------------------------------------------------------------ CPU pack

/// Downsample a 2x capture to res, coverage-weighted for colour channels and
/// min for depth. Alpha of every channel layer = coverage fraction.
#[cfg(test)]
fn downsample_view(view: &CapturedView, res: u32) -> Downsampled {
    // Capture side is res * supersample; derive it from the layer length.
    let cap = ((view.albedo.len() / 4) as f64).sqrt() as u32;
    let ss = (cap / res).max(1);
    let mut albedo = vec![0f32; (res * res * 4) as usize];
    let mut normal_ao = vec![0f32; (res * res * 4) as usize];
    let mut mr = vec![0f32; (res * res * 4) as usize];
    let mut emissive = vec![0f32; (res * res * 4) as usize];
    let mut depth_u16 = vec![0u16; (res * res) as usize];
    for y in 0..res {
        for x in 0..res {
            let mut wa = 0.0;
            let mut alb = [0f32; 3];
            let mut nrm = [0f32; 4];
            let mut mrv = [0f32; 4];
            let mut emi = [0f32; 3];
            let mut dmin = 1.0f32;
            for dy in 0..ss {
                for dx in 0..ss {
                    let i = (((y * ss + dy) * cap + x * ss + dx) * 4) as usize;
                    let w = view.albedo[i + 3];
                    wa += w;
                    for c in 0..3 {
                        alb[c] += view.albedo[i + c] * w;
                        nrm[c] += view.normal_ao[i + c] * w;
                        mrv[c] += view.mr[i + c] * w;
                        emi[c] += view.emissive[i + c] * w;
                    }
                    nrm[3] += view.normal_ao[i + 3] * w;
                    let d = view.depth[i / 4];
                    if w > 0.5 && d < dmin {
                        dmin = d;
                    }
                }
            }
            let o = ((y * res + x) * 4) as usize;
            let cov = wa / (ss * ss) as f32;
            let norm = if wa > 0.0 { 1.0 / wa } else { 0.0 };
            for c in 0..3 {
                albedo[o + c] = alb[c] * norm;
                nrm[c] *= norm;
                mr[o + c] = mrv[c] * norm;
                emissive[o + c] = emi[c] * norm;
            }
            nrm[3] *= norm;
            // normals are stored biased (n*0.5+0.5): decode, renormalize the
            // signed vector, re-encode. Normalizing the biased mean would
            // tilt every normal toward (1,1,1)/sqrt(3).
            let ns = [nrm[0] * 2.0 - 1.0, nrm[1] * 2.0 - 1.0, nrm[2] * 2.0 - 1.0];
            let l = (ns[0] * ns[0] + ns[1] * ns[1] + ns[2] * ns[2]).sqrt();
            let ns = if l > 1e-4 {
                [ns[0] / l, ns[1] / l, ns[2] / l]
            } else {
                [0.0, 1.0, 0.0]
            };
            normal_ao[o] = ns[0] * 0.5 + 0.5;
            normal_ao[o + 1] = ns[1] * 0.5 + 0.5;
            normal_ao[o + 2] = ns[2] * 0.5 + 0.5;
            normal_ao[o + 3] = nrm[3]; // AO, not coverage
            albedo[o + 3] = cov;
            mr[o + 3] = cov;
            emissive[o + 3] = cov;
            depth_u16[(y * res + x) as usize] = (dmin.clamp(0.0, 1.0) * 65535.0) as u16;
        }
    }
    Downsampled {
        albedo,
        normal_ao,
        mr,
        emissive,
        depth_u16,
    }
}

/// Propagate covered channels into uncovered texels (a few passes is enough
/// for mip bleed). `mask` marks covered texels — for the normal+AO layer the
/// mask comes from albedo coverage since its alpha channel is AO.
#[cfg(test)]
fn dilate(data: &mut [f32], res: u32, channels: usize, mask: &[bool]) {
    for _ in 0..6 {
        let src = data.to_vec();
        for y in 0..res {
            for x in 0..res {
                let i = ((y * res + x) * 4) as usize;
                if mask[(y * res + x) as usize] {
                    continue;
                }
                let mut acc = vec![0f32; channels];
                let mut n = 0.0;
                for (dy, dx) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (yy, xx) = (y as i32 + dy, x as i32 + dx);
                    if yy < 0 || xx < 0 || yy >= res as i32 || xx >= res as i32 {
                        continue;
                    }
                    let jj = (yy as u32 * res + xx as u32) as usize;
                    if mask[jj] {
                        let j = jj * 4;
                        for c in 0..channels {
                            acc[c] += src[j + c];
                        }
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    for c in 0..channels {
                        data[i + c] = acc[c] / n;
                    }
                }
            }
        }
    }
}

/// Box-filter mip chain for a colour layer. `coverage` supplies per-texel
/// sample weights for channels whose alpha is data rather than coverage
/// (normal/AO); `pack` turns the f32 rgba into final bytes.
#[cfg(test)]
fn mip_levels_f32(
    base: &[f32],
    res: u32,
    coverage: Option<&Vec<Vec<f32>>>,
    alpha_is_coverage: bool,
) -> Vec<Vec<f32>> {
    let mut levels: Vec<Vec<f32>> = vec![base.to_vec()];
    let mut size = res;
    while size > 1 {
        let half = (size / 2).max(1);
        let prev = levels.last().unwrap();
        let level = levels.len() - 1;
        let mut next = vec![0f32; (half * half * 4) as usize];
        for y in 0..half {
            for x in 0..half {
                let mut acc = [0f32; 4];
                let mut wsum = 0.0;
                for (dy, dx) in [(0u32, 0u32), (0, 1), (1, 0), (1, 1)] {
                    let (yy, xx) = (y * 2 + dy, x * 2 + dx);
                    let (yy, xx) = (yy.min(size - 1), xx.min(size - 1));
                    let texel = (yy * size + xx) as usize;
                    let i = texel * 4;
                    let w = coverage.map_or(prev[i + 3], |c| c[level][texel]);
                    wsum += w;
                    for c in 0..4 {
                        acc[c] += prev[i + c] * w;
                    }
                }
                let o = ((y * half + x) * 4) as usize;
                let cov = wsum / 4.0;
                let norm = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
                for c in 0..3 {
                    next[o + c] = acc[c] * norm;
                }
                // Alpha: coverage fraction for colour channels, AO for band 1.
                next[o + 3] = if alpha_is_coverage {
                    cov
                } else {
                    acc[3] * norm
                };
            }
        }
        levels.push(next);
        size = half;
    }
    levels
}

#[cfg(test)]
#[allow(dead_code)]
fn mip_chain_f32(
    base: &[f32],
    res: u32,
    alpha_is_coverage: bool,
    pack: impl Fn(&[f32]) -> [u8; 4],
) -> Vec<Vec<u8>> {
    mip_levels_f32(base, res, None, alpha_is_coverage)
        .iter()
        .map(|level| {
            let mut out = Vec::with_capacity(level.len());
            for px in level.chunks_exact(4) {
                out.extend_from_slice(&pack(px));
            }
            out
        })
        .collect()
}

#[cfg(test)]
fn normal_rgba8(px: &[f32]) -> [u8; 4] {
    // px rgb is n*0.5+0.5 (biased) — decode, renormalize signed, re-encode.
    let v = [px[0] * 2.0 - 1.0, px[1] * 2.0 - 1.0, px[2] * 2.0 - 1.0];
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (x, y, z) = if l > 1e-4 {
        (v[0] / l, v[1] / l, v[2] / l)
    } else {
        (0.0, 1.0, 0.0)
    };
    [
        ((x * 0.5 + 0.5) * 255.0) as u8,
        ((y * 0.5 + 0.5) * 255.0) as u8,
        ((z * 0.5 + 0.5) * 255.0) as u8,
        (px[3].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

#[cfg(test)]
fn mr_rgba8(px: &[f32]) -> [u8; 4] {
    [
        (px[0].clamp(0.0, 1.0) * 255.0) as u8,
        (px[1].clamp(0.0, 1.0) * 255.0) as u8,
        0,
        255,
    ]
}

#[cfg(test)]
fn rgba_srgb(px: &[f32]) -> [u8; 4] {
    let srgb = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
    [
        srgb(px[0]),
        srgb(px[1]),
        srgb(px[2]),
        (px[3].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

/// Min-reduce depth mip chain (nearest depth wins; sentinel 1.0 preserved).
#[cfg(test)]
#[allow(dead_code)]
fn depth_mip_chain(base: &[u16], res: u32) -> Vec<Vec<u16>> {
    let mut levels = vec![base.to_vec()];
    let mut size = res;
    while size > 1 {
        let half = (size / 2).max(1);
        let prev = levels.last().unwrap();
        let mut next = vec![0u16; (half * half) as usize];
        for y in 0..half {
            for x in 0..half {
                let mut d = u16::MAX;
                for (dy, dx) in [(0u32, 0u32), (0, 1), (1, 0), (1, 1)] {
                    let (yy, xx) = (y * 2 + dy, x * 2 + dx);
                    let (yy, xx) = (yy.min(size - 1), xx.min(size - 1));
                    d = d.min(prev[(yy * size + xx) as usize]);
                }
                next[(y * half + x) as usize] = d;
            }
        }
        levels.push(next);
        size = half;
    }
    levels
}

/// Upload a [6 layers x mips] byte list into an array image. `levels` is
/// layer-major: layers 0..5, each `mips` entries, level sizes shrinking.
#[cfg(test)]
#[allow(dead_code)]
fn upload_array(
    ctx: &Arc<GpuContext>,
    label: &str,
    res: u32,
    levels: &[Vec<u8>],
    format: vk::Format,
) -> crate::gfx::GpuImage {
    let mips = (levels.len() / VIEW_COUNT) as u32;
    let bytes: Vec<u8> = levels.iter().flatten().copied().collect();
    let mut staging = ctx.create_transfer_buffer("atlas staging", bytes.len() as u64);
    staging.write(0, &bytes);
    let image = ctx.create_image_array_mips(
        label,
        res,
        res,
        format,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        VIEW_COUNT as u32,
        mips,
    );
    let device = ctx.device().clone();
    ctx.one_shot(|cmd| unsafe {
        // Transition all layers/mips to TRANSFER_DST, copy, then to GENERAL.
        let full_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mips,
            base_array_layer: 0,
            layer_count: VIEW_COUNT as u32,
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image.image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .subresource_range(full_range);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        let mut offset = 0u64;
        let mut regions = Vec::with_capacity(VIEW_COUNT * mips as usize);
        let mut level = res;
        for layer in 0..VIEW_COUNT {
            for mip in 0..mips {
                let mip_size = (level * level) as usize * bytes_per_texel(format);
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: mip,
                            base_array_layer: layer as u32,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width: level,
                            height: level,
                            depth: 1,
                        }),
                );
                offset += mip_size as u64;
                level = (level / 2).max(1);
            }
            level = res;
        }
        device.cmd_copy_buffer_to_image(
            cmd,
            staging.buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image.image)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .subresource_range(full_range);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    });
    image
}

#[cfg(test)]
#[allow(dead_code)]
fn bytes_per_texel(format: vk::Format) -> usize {
    match format {
        vk::Format::R16_UNORM => 2,
        _ => 4,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::{AlphaMode, Material, Mesh, Model, Node};

    fn flat_quad_model() -> Arc<Model> {
        // A quad in the XZ plane: +/-Y captures are edge-on (empty), so the
        // y slab is invalid — the exact case slab_valid exists for.
        Arc::new(Model {
            meshes: vec![Mesh {
                name: "quad".into(),
                positions: vec![
                    [-1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                normals: vec![[0.0, 1.0, 0.0]; 4],
                uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                uvs1: Vec::new(),
                tangents: Vec::new(),
                joints: Vec::new(),
                weights: Vec::new(),
                indices: vec![0, 2, 1, 0, 3, 2],
                material: 0,
                node: Some(0),
            }],
            materials: vec![Material {
                name: "q".into(),
                base_color: [0.8, 0.3, 0.2, 1.0],
                ..Material::default()
            }],
            nodes: vec![Node {
                name: "root".into(),
                translation: Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::ONE,
                children: Vec::new(),
                mesh: Some(0),
                skin: None,
            }],
            roots: vec![0],
            ..Model::default()
        })
    }

    #[test]
    fn flat_quad_supported_by_static_capture() {
        // The thin axis is a bake-time slab_valid concern, not a rejection.
        assert!(super::super::static_capture_supported(&flat_quad_model()).is_ok());
    }

    #[test]
    fn blend_and_skinned_reject() {
        let mut model = flat_quad_model().as_ref().clone();
        model.materials[0].alpha_mode = AlphaMode::Blend;
        match super::super::static_capture_supported(&model) {
            Err(FallbackReason::BlendMaterial { .. }) => {}
            other => panic!("expected BlendMaterial, got {other:?}"),
        }
        let mut skinned = flat_quad_model().as_ref().clone();
        skinned.meshes[0].joints = vec![[0, 0, 0, 0]; 4];
        skinned.meshes[0].weights = vec![[1.0, 0.0, 0.0, 0.0]; 4];
        match super::super::static_capture_supported(&skinned) {
            Err(FallbackReason::SkinnedOrAnimated) => {}
            other => panic!("expected SkinnedOrAnimated, got {other:?}"),
        }
    }

    #[test]
    fn manual_mesh_selection_is_validated() {
        let model = flat_quad_model();
        let missing = [7usize];
        match prepare_sources(&[ImpostorSource::meshes(&model, &missing)], 0.0) {
            Err(FallbackReason::InvalidSelection { mesh: 7 }) => {}
            Err(other) => panic!("expected InvalidSelection, got {other:?}"),
            Ok(_) => panic!("expected InvalidSelection, got a prepared bake"),
        }

        let mut malformed = model.as_ref().clone();
        malformed.meshes[0].indices[0] = 99;
        let valid = [0usize];
        match prepare_sources(&[ImpostorSource::meshes(&malformed, &valid)], 0.0) {
            Err(FallbackReason::MalformedGeometry { mesh: 0 }) => {}
            Err(other) => panic!("expected MalformedGeometry, got {other:?}"),
            Ok(_) => panic!("expected MalformedGeometry, got a prepared bake"),
        }

        let mut skinned = model.as_ref().clone();
        skinned.meshes[0].joints = vec![[0, 0, 0, 0]; 4];
        skinned.meshes[0].weights = vec![[1.0, 0.0, 0.0, 0.0]; 4];
        match prepare_sources(&[ImpostorSource::meshes(&skinned, &valid)], 0.0) {
            Err(FallbackReason::SkinnedOrAnimated) => {}
            Err(other) => panic!("expected SkinnedOrAnimated, got {other:?}"),
            Ok(_) => panic!("expected SkinnedOrAnimated, got a prepared bake"),
        }

        let empty: [usize; 0] = [];
        match prepare_sources(&[ImpostorSource::meshes(&model, &empty)], 0.0) {
            Err(FallbackReason::EmptyGeometry) => {}
            Err(other) => panic!("expected EmptyGeometry, got {other:?}"),
            Ok(_) => panic!("expected EmptyGeometry, got a prepared bake"),
        }
    }

    #[test]
    fn grouped_sources_share_bounds_and_materials() {
        let model = flat_quad_model();
        let prepared = prepare_sources(
            &[
                ImpostorSource::model(&model),
                ImpostorSource::model(&model)
                    .transformed(Mat4::from_translation(Vec3::new(4.0, 0.0, 0.0))),
            ],
            0.0,
        )
        .expect("group prepares");
        assert_eq!(prepared.materials.len(), 1);
        assert_eq!(prepared.batches.len(), 1);
        assert_eq!(prepared.batches[0].0.len(), 8);
        assert_eq!(prepared.batches[0].1.len(), 12);
        assert!(prepared.bounds.min.x <= -1.0);
        assert!(prepared.bounds.max.x >= 5.0);
    }

    #[test]
    fn selection_can_ignore_unused_unsupported_materials() {
        let mut model = flat_quad_model().as_ref().clone();
        model.materials.push(Material {
            name: "unused blend".into(),
            alpha_mode: AlphaMode::Blend,
            ..Material::default()
        });
        let selected = [0usize];
        let prepared = prepare_sources(&[ImpostorSource::meshes(&model, &selected)], 0.0)
            .expect("selected material is supported");
        assert_eq!(prepared.materials.len(), 1);
    }

    #[test]
    fn normal_ao_mips_are_coverage_weighted() {
        // AO is data, not coverage. A mostly-covered dark-AO texel must beat
        // a barely-covered bright-AO texel in the next mip.
        let albedo = vec![
            1.0, 1.0, 1.0, 0.9, //
            1.0, 1.0, 1.0, 0.1, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        let normal_ao = vec![
            0.5, 0.5, 0.5, 0.1, //
            0.5, 0.5, 0.5, 0.9, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        let albedo_levels = mip_levels_f32(&albedo, 2, None, true);
        let coverage: Vec<Vec<f32>> = albedo_levels
            .iter()
            .map(|level| level.chunks_exact(4).map(|px| px[3]).collect())
            .collect();
        let levels = mip_levels_f32(&normal_ao, 2, Some(&coverage), false);
        assert!(
            (levels[1][3] - 0.18).abs() < 1e-5,
            "coverage-weighted AO {}, AO-weighted would be 0.82",
            levels[1][3]
        );
    }

    /// GPU tests need a window for the Vulkan surface; create a hidden one
    /// and skip gracefully where that's not possible.
    #[allow(deprecated)]
    fn gpu_ctx() -> Option<Arc<GpuContext>> {
        std::panic::catch_unwind(|| {
            #[cfg(windows)]
            let ev = {
                use winit::platform::windows::EventLoopBuilderExtWindows;
                winit::event_loop::EventLoop::builder()
                    .with_any_thread(true)
                    .build()
                    .ok()?
            };
            #[cfg(not(windows))]
            let ev = winit::event_loop::EventLoop::new().ok()?;
            let window = ev
                .create_window(winit::window::Window::default_attributes().with_visible(false))
                .ok()?;
            Some(GpuContext::new_owned(Arc::new(window)))
        })
        .ok()
        .flatten()
    }

    pub(crate) fn cube_model() -> Arc<Model> {
        let mut m = flat_quad_model().as_ref().clone();
        m.meshes[0].positions = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        m.meshes[0].normals = vec![[0.0, 1.0, 0.0]; 8];
        m.meshes[0].uvs = vec![[0.0, 0.0]; 8];
        m.meshes[0].indices = vec![
            0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7, 4, 5, 6, 4, 6,
            7, 1, 0, 3, 1, 3, 2,
        ];
        Arc::new(m)
    }

    /// One GPU context per process (winit allows a single EventLoop), so the
    /// bake assertions share a test.
    #[test]
    fn bake_gpu() {
        let Some(ctx) = gpu_ctx() else {
            eprintln!("skipping bake_gpu: no GPU window");
            return;
        };
        let baker = ImpostorBaker::new(&ctx);
        let settings = BakeSettings {
            resolution: 64,
            supersample: 2,
            voxel_resolution: 0,
            material_resolution: 64,
        };

        let cube = baker.bake(&cube_model(), settings).expect("cube bakes");
        if let Ok(dir) = std::env::var("BAKE_PREVIEW_DIR") {
            for (name, px, w, h) in &cube.previews {
                std::fs::write(
                    format!("{dir}/cube_{name}.png"),
                    crate::png::encode(*w, *h, px),
                )
                .unwrap();
            }
        }
        let stats = cube.test_stats.as_ref().expect("test stats");
        for f in 0..VIEW_COUNT {
            eprintln!(
                "cube view {f}: coverage {:.3} centre_depth {:.3}",
                stats.coverage[f], stats.center_depth[f]
            );
        }
        for f in 0..6 {
            // ~4-texel padding shrinks the projected cube below the tile:
            // expected coverage is (1 - 2*pad/res)^2 ≈ 0.77 at res 64.
            assert!(
                stats.coverage[f] > 0.7,
                "face {f} coverage {}",
                stats.coverage[f]
            );
            // Nearest surface sits pad-depth inside the padded capture
            // plane (~4/64 ≈ 0.06 at res 64), far from the 1.0 sentinel.
            assert!(
                stats.center_depth[f] < 0.15,
                "face {f} centre depth {}",
                stats.center_depth[f]
            );
        }
        assert!(stats.mip_count > 1);
        assert!(stats.dilated, "coverage dilation should fill edges");

        // The manual subset and grouped-source entry points reach the same GPU
        // bake path; the group shares one material layer while its bounds cover
        // both transformed copies.
        let cube_src = cube_model();
        let selected = [0usize];
        let subset = baker
            .bake_meshes(&cube_src, &selected, settings)
            .expect("mesh subset bakes");
        assert_eq!(subset.material_count, 1);
        let grouped = baker
            .bake_sources(
                &[
                    ImpostorSource::model(&cube_src),
                    ImpostorSource::model(&cube_src)
                        .transformed(Mat4::from_translation(Vec3::new(4.0, 0.0, 0.0))),
                ],
                settings,
            )
            .expect("grouped sources bake");
        assert_eq!(grouped.material_count, 1);
        assert!(grouped.bounds.min.x <= -1.0);
        assert!(grouped.bounds.max.x >= 5.0);

        // sRGB encode check: linear 0.8 albedo must land as ~234 in the
        // SRGB array — a double-encode would give ~246 (and dark renders).
        let layers = cube.test_layers.as_ref().expect("test layers");
        let centre = (32 * 64 + 32) as usize * 4;
        let al = layers.albedo[0][centre] as f32 / 255.0;
        let expected = 0.8f32.powf(1.0 / 2.2);
        assert!(
            (al - expected).abs() < 0.03,
            "centre albedo {al:.3} vs srgb(0.8)={expected:.3}"
        );

        // Opaque cube: the triangle-DDA path is eligible (flag asserted in
        // render.rs against voxel_resolution).
        assert!(!cube.has_mask_materials);
        let mut mask_model = cube_model();
        Arc::get_mut(&mut mask_model).unwrap().materials[0].alpha_mode = AlphaMode::Mask;
        let masked = baker.bake(&mask_model, settings).expect("mask bakes");
        assert!(masked.has_mask_materials);

        // A flat quad still bakes — its thin axis leaves the hull.
        let quad = baker
            .bake(&flat_quad_model(), settings)
            .expect("flat quad still bakes");
        assert_eq!(
            quad.slab_valid,
            [false, true, false, true, true, true, true]
        );
    }

    /// Read one (layer, mip) of a baked array image back to the CPU.
    fn read_layer_mip(
        ctx: &Arc<GpuContext>,
        image: &GpuImage,
        layer: u32,
        mip: u32,
        res: u32,
        bpp: u64,
    ) -> Vec<u8> {
        let r = (res >> mip).max(1);
        let buf = ctx.create_readback_buffer("test rb", (r * r) as u64 * bpp);
        let device = ctx.device().clone();
        ctx.one_shot(|cmd| unsafe {
            let sub = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            };
            let to = vk::ImageMemoryBarrier::default()
                .image(image.image)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(sub);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to],
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: layer,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: r,
                    height: r,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cmd,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buf.buffer,
                &[region],
            );
            let back = to
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[back],
            );
        });
        buf.mapped().map(|m| m.to_vec()).unwrap_or_default()
    }

    /// The GPU post chain (downsample/dilate/mips/pack) must match the CPU
    /// reference within tolerance: colour <= 2/255, normals <= 1 degree,
    /// depth exact. Runs on the cube and the helmet at levels 0 and 2.
    #[test]
    fn gpu_post_parity_gpu() {
        let Some(ctx) = gpu_ctx() else {
            eprintln!("skipping gpu_post_parity_gpu: no GPU window");
            return;
        };
        let models = [
            (cube_model(), 64u32),
            (
                Model::load_gltf_file(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/fixtures/DamagedHelmet.glb"
                ))
                .expect("helmet")
                .into(),
                128,
            ),
        ];
        for (model, res) in models {
            let ss = 2u32;
            let baker = ImpostorBaker::new(&ctx);
            let asset = baker
                .bake(
                    &model,
                    BakeSettings {
                        resolution: res,
                        supersample: ss,
                        voxel_resolution: 0,
                        material_resolution: 64,
                    },
                )
                .expect("bake");
            let _layers = asset.test_layers.as_ref().expect("test layers");

            // CPU reference on the same captures.
            let bounds = ImpostorBounds::from_model(&model, 4.0 / res as f32);
            let center = bounds.center();
            let extent = bounds.extent();
            let batches = merge_by_material(&model);
            let baked: Vec<BakeBatch> = batches
                .iter()
                .enumerate()
                .filter(|(_, (_, idx))| !idx.is_empty())
                .map(|(mi, (v, i))| BakeBatch {
                    vertex_buffer: ctx.create_vertex_buffer("t verts", bytemuck::cast_slice(v)),
                    index_buffer: ctx.create_index_buffer("t idx", bytemuck::cast_slice(i)),
                    index_count: i.len() as u32,
                    material: baker.bake_material(&model.materials[mi]),
                })
                .collect();
            let cap = res * ss;
            let mut views = Vec::new();
            for view in VIEWS.iter() {
                let ext_u = projected_extent(extent, view.u);
                let ext_v = projected_extent(extent, view.v);
                let ext_n = projected_extent(extent, view.dir);
                let eye = center + view.dir * (ext_n * 0.5 + 1.0);
                let mvp = Mat4::orthographic_rh(
                    -ext_u * 0.5,
                    ext_u * 0.5,
                    -ext_v * 0.5,
                    ext_v * 0.5,
                    0.01,
                    ext_n + 2.0,
                ) * Mat4::look_to_rh(eye, -view.dir, view.v);
                views.push(
                    baker
                        .capture_view(&baked, cap, mvp, view.dir, center, [ext_u, ext_v, ext_n])
                        .expect("capture"),
                );
            }
            // Raw capture sanity: score surface-point depth agreement before
            // the post chain so bake/pack issues can't masquerade as capture
            // projection issues.
            let surface = surface_samples(&model, 2000);
            for (f, (view, raw)) in VIEWS.iter().zip(&views).enumerate() {
                let ext_n = projected_extent(extent, view.dir);
                let tol = 3.0 * ext_n / (res * ss) as f32;
                let (mut ok, mut seen) = (0usize, 0usize);
                let mut shown = 0usize;
                for &(p, n) in &surface {
                    if n.dot(view.dir) <= 0.3 {
                        continue;
                    }
                    seen += 1;
                    let uv = crate::impostor::face_uv(view, extent, p, center);
                    let cap = res * ss;
                    let tx = ((uv.x * cap as f32) as i32).clamp(0, cap as i32 - 1);
                    let ty = ((uv.y * cap as f32) as i32).clamp(0, cap as i32 - 1);
                    let ti = (ty as u32 * cap + tx as u32) as usize;
                    let expected = (ext_n * 0.5 - (p - center).dot(view.dir)) / ext_n;
                    let baked = raw.depth[ti];
                    let err = (baked - expected).abs() * ext_n;
                    let good = baked <= expected + tol / ext_n || err <= tol;
                    if good {
                        ok += 1;
                    } else if shown < 4 {
                        shown += 1;
                        eprintln!(
                            "  view {f} miss: uv=({:.4},{:.4}) tex=({tx},{ty}) exp={expected:.4} got={baked:.4} cov={:.3} ndot={:.3}",
                            uv.x,
                            uv.y,
                            raw.albedo[ti * 4 + 3],
                            n.dot(view.dir)
                        );
                    }
                }
                eprintln!(
                    "raw view {f}: depth {:.1}% ({ok}/{seen})",
                    ok as f32 * 100.0 / seen.max(1) as f32
                );
            }

            let mut chans: [Vec<Vec<f32>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut deps: Vec<Vec<u16>> = Vec::new();
            for view in &views {
                let ds = downsample_view(view, res);
                chans[0].push(ds.albedo);
                chans[1].push(ds.normal_ao);
                chans[2].push(ds.mr);
                chans[3].push(ds.emissive);
                deps.push(ds.depth_u16);
            }
            for f in 0..VIEW_COUNT {
                let mask: Vec<bool> = chans[0][f].chunks_exact(4).map(|px| px[3] > 0.0).collect();
                dilate(&mut chans[0][f], res, 3, &mask);
                dilate(&mut chans[1][f], res, 4, &mask);
                dilate(&mut chans[2][f], res, 3, &mask);
                dilate(&mut chans[3][f], res, 3, &mask);
            }
            let packers: [fn(&[f32]) -> [u8; 4]; 4] =
                [rgba_srgb, normal_rgba8, mr_rgba8, rgba_srgb];
            let gpu_images = [
                &asset.albedo,
                &asset.normal_ao,
                &asset.metallic_rough,
                &asset.emissive,
            ];
            for f in 0..VIEW_COUNT {
                let albedo_levels = mip_levels_f32(&chans[0][f], res, None, true);
                let coverage_levels: Vec<Vec<f32>> = albedo_levels
                    .iter()
                    .map(|level| level.chunks_exact(4).map(|px| px[3]).collect())
                    .collect();
                for ch in 0..4 {
                    let cpu_levels = if ch == 0 {
                        albedo_levels.clone()
                    } else if ch == 1 {
                        mip_levels_f32(&chans[ch][f], res, Some(&coverage_levels), false)
                    } else {
                        mip_levels_f32(&chans[ch][f], res, None, true)
                    };
                    for level in [0u32, 2] {
                        let r = (res >> level).max(1);
                        let gpu_px = read_layer_mip(&ctx, gpu_images[ch], f as u32, level, res, 4);
                        let mut max_byte = 0i32;
                        let mut max_ang = 0f32;
                        let mut worst = 0usize;
                        for t in 0..(r * r) as usize {
                            let c = packers[ch](&cpu_levels[level as usize][t * 4..t * 4 + 4]);
                            let g = &gpu_px[t * 4..t * 4 + 4];
                            let mut d = 0i32;
                            for k in 0..4 {
                                d = d.max((c[k] as i32 - g[k] as i32).abs());
                            }
                            if d > max_byte {
                                max_byte = d;
                                worst = t;
                            }
                            if ch == 1 {
                                let dec = |b: &[u8]| {
                                    Vec3::new(
                                        b[0] as f32 / 255.0 * 2.0 - 1.0,
                                        b[1] as f32 / 255.0 * 2.0 - 1.0,
                                        b[2] as f32 / 255.0 * 2.0 - 1.0,
                                    )
                                };
                                let (a, b) = (dec(&c), dec(g));
                                if a.length() > 0.5 && b.length() > 0.5 {
                                    let ang =
                                        a.normalize().angle_between(b.normalize()).to_degrees();
                                    max_ang = max_ang.max(ang);
                                }
                            }
                        }
                        eprintln!(
                            "ch{ch} face{f} mip{level}: max byte {max_byte} ang {max_ang:.2} worst {worst} cpu={:?} gpu={:?}",
                            packers[ch](&cpu_levels[level as usize][worst * 4..worst * 4 + 4]),
                            &gpu_px[worst * 4..worst * 4 + 4],
                        );
                        assert!(
                            max_byte <= 2,
                            "ch{ch} face{f} mip{level} max byte diff {max_byte}"
                        );
                        assert!(
                            max_ang <= 1.0,
                            "normal face{f} mip{level} max angle {max_ang:.2} deg"
                        );
                    }
                }
                let gpu_d = read_layer_mip(&ctx, &asset.depth, f as u32, 0, res, 2);
                for t in 0..(res * res) as usize {
                    let g = u16::from_le_bytes([gpu_d[t * 2], gpu_d[t * 2 + 1]]);
                    assert_eq!(
                        g, deps[f][t],
                        "depth face{f} texel {t}: cpu {} gpu {}",
                        deps[f][t], g
                    );
                }
            }
        }
    }

    /// Deterministic xorshift for area-weighted surface sampling.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    /// Sample ~`n` area-weighted points on the model's surfaces (object space,
    /// node rest transforms applied), returning (position, surface normal).
    fn surface_samples(model: &Model, n: usize) -> Vec<(Vec3, Vec3)> {
        let globals = model.rest_globals();
        let mut tris: Vec<(Vec3, Vec3, Vec3, f32)> = Vec::new();
        let mut vnorm: Vec<(Vec3, Vec3, Vec3)> = Vec::new();
        for mesh in &model.meshes {
            let Some(node) = mesh.node else { continue };
            let g = globals[node];
            for t in mesh.indices.chunks_exact(3) {
                let a = g.transform_point3(Vec3::from(mesh.positions[t[0] as usize]));
                let b = g.transform_point3(Vec3::from(mesh.positions[t[1] as usize]));
                let c = g.transform_point3(Vec3::from(mesh.positions[t[2] as usize]));
                let area = (b - a).cross(c - a).length() * 0.5;
                if area > 1e-8 {
                    let na = g.transform_vector3(Vec3::from(mesh.normals[t[0] as usize]));
                    let nb = g.transform_vector3(Vec3::from(mesh.normals[t[1] as usize]));
                    let nc = g.transform_vector3(Vec3::from(mesh.normals[t[2] as usize]));
                    tris.push((a, b, c, area));
                    vnorm.push((na, nb, nc));
                }
            }
        }
        let total: f32 = tris.iter().map(|t| t.3).sum();
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut r = rng.next() * total;
            let mut ti = 0usize;
            for (i, t) in tris.iter().enumerate() {
                r -= t.3;
                if r <= 0.0 {
                    ti = i;
                    break;
                }
            }
            let tri = &tris[ti];
            let (u, v) = (rng.next().sqrt(), rng.next());
            let w0 = 1.0 - u;
            let w1 = u * (1.0 - v);
            let w2 = u * v;
            let p = tri.0 + (tri.1 - tri.0) * w1 + (tri.2 - tri.0) * w2;
            let vn = vnorm[ti];
            let n = (vn.0 * w0 + vn.1 * w1 + vn.2 * w2).normalize();
            out.push((p, n));
        }
        out
    }

    /// Bake-consistency: projected surface points must land on matching
    /// coverage/depth/normal texels in every view. Catches basis/orientation
    /// bugs between the capture and the shader's face_uv.
    #[test]
    fn bake_consistency_gpu() {
        let Some(ctx) = gpu_ctx() else {
            eprintln!("skipping bake_consistency_gpu: no GPU window");
            return;
        };
        // NB: only one GPU context per process — this test owns it.
        let model = Model::load_gltf_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/DamagedHelmet.glb"
        ))
        .expect("helmet");
        let baker = ImpostorBaker::new(&ctx);
        let res = 128u32;
        let asset = baker
            .bake(
                &model,
                BakeSettings {
                    resolution: res,
                    supersample: 1,
                    voxel_resolution: 0,
                    material_resolution: 128,
                },
            )
            .expect("helmet bakes");
        let layers = asset.test_layers.as_ref().expect("test layers");
        let bounds = asset.bounds;
        let center = bounds.center();
        let extent = bounds.extent();
        let samples = surface_samples(&model, 2000);
        if let Ok(dir) = std::env::var("BAKE_PREVIEW_DIR") {
            for (name, px, w, h) in &asset.previews {
                std::fs::write(format!("{dir}/{name}.png"), crate::png::encode(*w, *h, px))
                    .unwrap();
            }
            // Per-view tiles for inspection.
            for f in 0..VIEW_COUNT {
                std::fs::write(
                    format!("{dir}/albedo_v{f}.png"),
                    crate::png::encode(res, res, &layers.albedo[f]),
                )
                .unwrap();
                std::fs::write(
                    format!("{dir}/normal_v{f}.png"),
                    crate::png::encode(res, res, &layers.normal_ao[f]),
                )
                .unwrap();
                let depth: Vec<u8> = layers.depth[f]
                    .iter()
                    .flat_map(|d| {
                        let v = (*d as u32 * 255 / 65535) as u8;
                        [v, v, v, 255]
                    })
                    .collect();
                std::fs::write(
                    format!("{dir}/depth_v{f}.png"),
                    crate::png::encode(res, res, &depth),
                )
                .unwrap();
            }
        }

        let mut bad_depth = Vec::new();
        for f in 0..VIEW_COUNT {
            let view = &VIEWS[f];
            let dir = view.dir;
            let ext_u = projected_extent(extent, view.u);
            let ext_v = projected_extent(extent, view.v);
            let ext_n = projected_extent(extent, dir);
            let eye = center + dir * (ext_n * 0.5 + 1.0);
            let mvp = Mat4::orthographic_rh(
                -ext_u * 0.5,
                ext_u * 0.5,
                -ext_v * 0.5,
                ext_v * 0.5,
                0.01,
                ext_n + 2.0,
            ) * Mat4::look_to_rh(eye, -dir, view.v);
            let tol = 3.0 * ext_n / res as f32; // 3 texels of surface distance
            let (mut ok_d, mut ok_c, mut ok_n, mut seen) = (0usize, 0usize, 0usize, 0usize);
            let mut n_seen = 0usize;
            let mut facing_samples = Vec::new();
            let mut max_uv_err = 0f32;
            for &(p, n) in &samples {
                if n.dot(dir) <= 0.3 {
                    continue;
                }
                seen += 1;
                facing_samples.push(p);
                let uv = crate::impostor::face_uv(view, extent, p, center);
                let clip = mvp * p.extend(1.0);
                let uv_m =
                    glam::Vec2::new(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
                max_uv_err = max_uv_err.max((uv - uv_m).abs().max_element());
                let tx = ((uv.x * res as f32) as i32).clamp(0, res as i32 - 1);
                let ty = ((uv.y * res as f32) as i32).clamp(0, res as i32 - 1);
                let ti = (ty as u32 * res + tx as u32) as usize;
                // Coverage
                let cov = layers.albedo[f][ti * 4 + 3] as f32 / 255.0;
                if cov > 0.5 {
                    ok_c += 1;
                }
                // Depth: expected normalized distance from the capture plane.
                let expected = (ext_n * 0.5 - (p - center).dot(dir)) / ext_n;
                let baked = layers.depth[f][ti] as f32 / 65535.0;
                let err = (baked - expected).abs() * ext_n;
                // Occluded points legitimately read nearer; allow that too
                // (baked < expected is fine — nearer surface).
                if baked <= expected + tol / ext_n || err <= tol {
                    ok_d += 1;
                }
                // Object-space normal within 25 degrees of the surface normal.
                let nb = &layers.normal_ao[f][ti * 4..ti * 4 + 3];
                let nbv = Vec3::new(
                    nb[0] as f32 / 255.0 * 2.0 - 1.0,
                    nb[1] as f32 / 255.0 * 2.0 - 1.0,
                    nb[2] as f32 / 255.0 * 2.0 - 1.0,
                );
                // Normals: only where the baked depth proves this texel is
                // the sampled surface itself (unoccluded). Normal-map detail
                // perturbs off the vertex normal by ~30-50 deg on this asset,
                // so the gate is a 60-degree cone at >=85% — enough to catch
                // wrong bases or space mix-ups, not detail fidelity.
                if err <= tol && cov > 0.5 {
                    n_seen += 1;
                    if nbv.length() > 0.5 && nbv.normalize().dot(n).abs() > 0.5 {
                        ok_n += 1;
                    }
                }
            }
            if f >= 6 {
                let mut rates = [0usize; 4];
                for &p in &facing_samples {
                    let uv0 = crate::impostor::face_uv(view, extent, p, center);
                    for (mode, rate) in rates.iter_mut().enumerate() {
                        let uv = match mode {
                            0 => uv0,
                            1 => glam::Vec2::new(1.0 - uv0.x, uv0.y),
                            2 => glam::Vec2::new(uv0.x, 1.0 - uv0.y),
                            _ => glam::Vec2::new(1.0 - uv0.x, 1.0 - uv0.y),
                        };
                        let tx = ((uv.x * res as f32) as i32).clamp(0, res as i32 - 1);
                        let ty = ((uv.y * res as f32) as i32).clamp(0, res as i32 - 1);
                        let ti = (ty as u32 * res + tx as u32) as usize;
                        let expected = (ext_n * 0.5 - (p - center).dot(dir)) / ext_n;
                        let baked = layers.depth[f][ti] as f32 / 65535.0;
                        let err = (baked - expected).abs() * ext_n;
                        if baked <= expected + tol / ext_n || err <= tol {
                            *rate += 1;
                        }
                    }
                }
                eprintln!(
                    "  uv rates id={:.1}% flip-u={:.1}% flip-v={:.1}% flip-uv={:.1}%",
                    rates[0] as f32 * 100.0 / facing_samples.len().max(1) as f32,
                    rates[1] as f32 * 100.0 / facing_samples.len().max(1) as f32,
                    rates[2] as f32 * 100.0 / facing_samples.len().max(1) as f32,
                    rates[3] as f32 * 100.0 / facing_samples.len().max(1) as f32,
                );
                let mut match_n = 0usize;
                let mut off_sum = glam::Vec2::ZERO;
                let mut off_max = 0f32;
                for &p in facing_samples.iter().take(600) {
                    let uv = crate::impostor::face_uv(view, extent, p, center);
                    let cx = (uv.x * res as f32) as i32;
                    let cy = (uv.y * res as f32) as i32;
                    let expected = (ext_n * 0.5 - (p - center).dot(dir)) / ext_n;
                    let mut best = (i32::MIN, i32::MIN, f32::INFINITY);
                    for y in (cy - 16)..=(cy + 16) {
                        if y < 0 || y >= res as i32 {
                            continue;
                        }
                        for x in (cx - 16)..=(cx + 16) {
                            if x < 0 || x >= res as i32 {
                                continue;
                            }
                            let ti = (y as u32 * res + x as u32) as usize;
                            let err = (layers.depth[f][ti] as f32 / 65535.0 - expected).abs();
                            if err < best.2 {
                                best = (x - cx, y - cy, err);
                            }
                        }
                    }
                    if best.2 * ext_n <= tol {
                        match_n += 1;
                        off_sum += glam::Vec2::new(best.0 as f32, best.1 as f32);
                        off_max = off_max.max(best.0.abs() as f32).max(best.1.abs() as f32);
                    }
                }
                eprintln!(
                    "  matched={match_n}/600 mean_off=({:.2},{:.2}) max_off={off_max:.1}",
                    off_sum.x / match_n.max(1) as f32,
                    off_sum.y / match_n.max(1) as f32,
                );
            }
            let seen = seen.max(1);
            eprintln!(
                "view {f} dir={dir:?} seen={seen} depth={:.1}% cov={:.1}% nrm={:.1}% ({n_seen} aligned) uv_err={max_uv_err:.6}",
                ok_d as f32 * 100.0 / seen as f32,
                ok_c as f32 * 100.0 / seen as f32,
                ok_n as f32 * 100.0 / n_seen.max(1) as f32,
            );
            if (ok_d as f32) < 0.9 * seen as f32 {
                bad_depth.push((f, ok_d, seen));
            }
            assert!(
                ok_c as f32 >= 0.9 * seen as f32,
                "view {f}: coverage ok {ok_c}/{seen}"
            );
            // Normals can't be hard-gated per view: open-shell surfaces
            // legitimately capture inner faces with outward normals (the
            // helmet's rear opening), so the fraction is reported and only
            // a scrambled channel (near-zero agreement) fails.
            assert!(
                ok_n as f32 >= 0.25 * n_seen.max(1) as f32,
                "view {f}: normal ok {ok_n}/{n_seen}"
            );
        }
        assert!(bad_depth.is_empty(), "bad depth views {bad_depth:?}");
    }

    /// Mirror of the WGSL hierarchical traversal against REAL baked depth
    /// (synthesized MIN mips), vs a brute-force mip-0 scan as ground truth.
    #[test]
    fn hull_traverse_mirror_gpu() {
        use crate::impostor::{AXIS_COUNT, VIEWS, face_uv, hull_traverse, view_axis};
        let Some(ctx) = gpu_ctx() else {
            eprintln!("skip hull_traverse_mirror_gpu: no GPU");
            return;
        };
        let model = Model::load_gltf_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/DamagedHelmet.glb"
        ))
        .unwrap();
        let baker = ImpostorBaker::new(&ctx);
        let res = 512u32;
        let asset = baker
            .bake(
                &model,
                BakeSettings {
                    resolution: res,
                    supersample: 1,
                    voxel_resolution: 0,
                    material_resolution: 128,
                },
            )
            .unwrap();
        let layers = asset.test_layers.as_ref().unwrap();
        let bounds = asset.bounds;
        let c = bounds.center();
        let extent = bounds.extent();
        let valid_mask: u32 = asset
            .slab_valid
            .iter()
            .enumerate()
            .map(|(i, v)| (*v as u32) << i)
            .sum();

        let mut depth: Vec<Vec<Vec<f32>>> = vec![Vec::new()];
        depth[0] = layers
            .depth
            .iter()
            .map(|f| f.iter().map(|&v| v as f32 / 65535.0).collect())
            .collect();
        let mut rl = res;
        while rl > 1 {
            let rl2 = rl / 2;
            let prev = depth.last().unwrap();
            let mut next = vec![vec![0f32; (rl2 * rl2) as usize]; VIEW_COUNT];
            for f in 0..VIEW_COUNT {
                for y in 0..rl2 {
                    for x in 0..rl2 {
                        let s =
                            |dx: u32, dy: u32| prev[f][((y * 2 + dy) * rl + x * 2 + dx) as usize];
                        next[f][(y * rl2 + x) as usize] =
                            s(0, 0).min(s(1, 0)).min(s(0, 1)).min(s(1, 1));
                    }
                }
            }
            depth.push(next);
            rl = rl2;
        }
        let max_mip = (depth.len() - 1) as u32;

        let slab = |a: usize, p: glam::Vec3, mip: u32| -> Option<(f32, f32)> {
            let dir = VIEWS[a * 2].dir;
            let ext_n = projected_extent(extent, dir);
            let rl = res >> mip;
            let tex = |uv: glam::Vec2| -> usize {
                let x = (uv.x * rl as f32).clamp(0.0, (rl - 1) as f32) as usize;
                let y = (uv.y * rl as f32).clamp(0.0, (rl - 1) as f32) as usize;
                y * rl as usize + x
            };
            let d_p = depth[mip as usize][2 * a][tex(face_uv(&VIEWS[2 * a], extent, p, c))];
            let d_m = depth[mip as usize][2 * a + 1][tex(face_uv(&VIEWS[2 * a + 1], extent, p, c))];
            if d_p >= 1.0 || d_m >= 1.0 {
                None
            } else {
                Some((-ext_n * 0.5 + d_m * ext_n, ext_n * 0.5 - d_p * ext_n))
            }
        };
        let slabv = |a: usize, p: glam::Vec3, mip: u32| -> Option<(f32, f32)> {
            if valid_mask & (1 << a) == 0 {
                return None;
            }
            slab(a, p, mip)
        };
        // Exact finest-slab check with the SAME relative tolerance the
        // traversal uses (extent * 1e-4 per axis).
        let inside0 = |p: glam::Vec3| -> bool {
            for a in 0..AXIS_COUNT {
                if valid_mask & (1 << a) == 0 {
                    continue;
                }
                let Some((lo, hi)) = slab(a, p, 0) else {
                    return false;
                };
                let axis = VIEWS[a * 2].dir;
                let q = (p - c).dot(axis);
                let tol = extent.dot(axis.abs()) * (0.5 / res as f32);
                if q < lo - tol || q > hi + tol {
                    return false;
                }
            }
            true
        };
        let gmin = c - extent * 0.5;
        let _ = view_axis;

        // Every real surface point must be contained by every valid slab.
        // Count per-axis violations independently of the ray traversal.
        let surface = surface_samples(&model, 5000);
        let mut slab_none = [0usize; AXIS_COUNT];
        let mut slab_out = [0usize; AXIS_COUNT];
        let mut slab_err_sum = [0f32; AXIS_COUNT];
        let mut slab_err_max = [0f32; AXIS_COUNT];
        let mut all_inside = 0usize;
        for &(p, _) in &surface {
            let mut ok = true;
            for a in 0..AXIS_COUNT {
                if valid_mask & (1 << a) == 0 {
                    continue;
                }
                let Some((lo, hi)) = slab(a, p, 0) else {
                    slab_none[a] += 1;
                    ok = false;
                    continue;
                };
                let axis = VIEWS[a * 2].dir;
                let q = (p - c).dot(axis);
                let tol = extent.dot(axis.abs()) * (0.5 / res as f32);
                let err = (lo - q).max(q - hi).max(0.0);
                slab_err_sum[a] += err;
                slab_err_max[a] = slab_err_max[a].max(err);
                if q < lo - tol || q > hi + tol {
                    slab_out[a] += 1;
                    ok = false;
                }
            }
            all_inside += ok as usize;
        }
        let slab_avg: Vec<f32> = slab_err_sum
            .iter()
            .map(|s| s / surface.len() as f32)
            .collect();
        eprintln!(
            "surface containment {:.1}% none={slab_none:?} out={slab_out:?} avg={slab_avg:?} max={slab_err_max:?}",
            all_inside as f32 * 100.0 / surface.len() as f32
        );

        // Dense ray fans from four azimuths — the wedge was at az 2.0.
        let mut worst = Vec::new();
        for (az, el) in [(0.6f32, 0.15f32), (2.0, 0.4), (3.8, -0.1), (5.2, 0.7)] {
            let cam = glam::Vec3::new(
                az.cos() * el.cos() * 8.0,
                el.sin() * 8.0 + 0.9,
                az.sin() * el.cos() * 8.0,
            );
            let fwd = (glam::Vec3::new(0.0, 0.9, 0.0) - cam).normalize();
            let right = fwd.cross(glam::Vec3::Y).normalize();
            let up = right.cross(fwd).normalize();
            let mut misses = 0usize;
            let mut samples = 0usize;
            let mut report = 0usize;
            for iy in -50..50 {
                for ix in -50..50 {
                    let dir = (fwd * 8.0 + right * (ix as f32 * 0.02) + up * (iy as f32 * 0.02))
                        .normalize();
                    let mut brute = None;
                    for i in 0..2000 {
                        let t = 4.0 + i as f32 * 0.004;
                        if inside0(cam + dir * t) {
                            brute = Some(cam + dir * t);
                            break;
                        }
                    }
                    let Some(b) = brute else { continue };
                    samples += 1;
                    match hull_traverse(cam, dir, c, gmin, extent, res, max_mip, valid_mask, &slabv)
                    {
                        None => {
                            misses += 1;
                            if report < 8 {
                                report += 1;
                                eprintln!("az={az} ix={ix} iy={iy} MISS brute={b:?}");
                            }
                        }
                        Some(h) => {
                            // Strict invariant: every returned point satisfies
                            // the exact mip-0 slabs (same relative tolerance).
                            if !inside0(h) {
                                misses += 1;
                                if report < 8 {
                                    report += 1;
                                    eprintln!("az={az} ix={ix} iy={iy} NOT-INSIDE b={b:?} h={h:?}");
                                }
                            }
                        }
                    }
                }
            }
            worst.push((az, misses, samples));
            eprintln!("az={az} brute_hits={samples} traversal_mismatches={misses}");
        }
        assert!(worst.iter().all(|&(_, m, _)| m == 0), "{worst:?}");
    }

    /// Material index is stored as a raw u32 in record word 42 — the shader
    /// must read the raw word (bitcast to f32 would round every index to 0).
    #[test]
    fn triangle_record_material_is_raw_word() {
        let v = [PbrVertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv0: [0.0, 0.0],
            uv1: [0.0, 0.0],
        }; 3];
        let rec = triangle_record(
            &v,
            &[0, 1, 2],
            [Vec3::ZERO, Vec3::X, Vec3::Z],
            [Vec2::ZERO; 3],
            [Vec2::ZERO; 3],
            3,
        );
        assert_eq!(rec[42], 3u32, "material index stored raw");
        let wgsl = include_str!("impostor.wgsl");
        assert!(
            wgsl.contains("voxel_bits.bits[base + 42u]"),
            "shader must read the raw material word"
        );
        assert!(
            !wgsl.contains("u32(tri_f32(base, 42u)"),
            "shader must NOT bitcast the material word as float"
        );
    }

    #[test]
    fn helmet_max_refs_report() {
        let Some(_ctx) = gpu_ctx() else {
            return;
        };
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/DamagedHelmet.glb");
        let Ok(model) = Model::load_gltf_file(path) else {
            return;
        };
        for (name, settings) in [
            ("compact", BakeSettings::compact()),
            ("default", BakeSettings::default()),
        ] {
            let src = ImpostorSource::model(&model);
            let prepared =
                prepare_sources(&[src], 4.0 / settings.resolution as f32).expect("prepare");
            let vox = settings.voxel_resolution;
            let (cells, _sc, records, refs) =
                voxelize_batches(&prepared.batches, prepared.bounds, vox);
            let max = check_ref_lists(&refs, cells.len(), cells.len() + refs.len(), u32::MAX)
                .expect("refs");
            eprintln!(
                "[report] helmet {name}: max_cell_refs={max} tricount={} surfaces={_sc}",
                records.len() / TRIANGLE_WORDS
            );
        }
    }

    #[test]
    fn material_layer_ignores_legacy_texture() {
        // A legacy `base_color_texture` with no `base_color_map` must bake
        // WHITE — the PBR/capture paths never read the legacy field.
        let mut mat = Material::default();
        mat.base_color_texture = Some(crate::texture::DecodedTexture {
            pixels: vec![200, 50, 50, 255],
            width: 1,
            height: 1,
        });
        let layer = material_layer(&mat, 0, 1);
        assert_eq!(layer.pixels, vec![255, 255, 255, 255]);

        // The same texture via `base_color_map` produces matching bytes
        // (sRGB round-trip within 1).
        mat.base_color_map = Some(crate::model::TextureRef {
            texture: mat.base_color_texture.clone().unwrap(),
            tex_coord: 0,
            transform: None,
            wrap_s: Wrap::Repeat,
            wrap_t: Wrap::Repeat,
            mag_filter: TexelFilter::Linear,
            min_filter: TexelFilter::Linear,
        });
        let layer = material_layer(&mat, 0, 1);
        for (a, b) in layer.pixels.iter().zip([200, 50, 50, 255]) {
            assert!((*a as i32 - b as i32).abs() <= 1, "{a} vs {b}");
        }
    }
}
