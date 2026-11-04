//! The baked asset: five channel arrays, one layer for each of the 26 views.

use super::ImpostorBounds;
use crate::gfx::vulkan::GpuContext;
use crate::gfx::{GpuBuffer, GpuImage};
use std::sync::Arc;

/// Which debug view the impostor shader renders (Asset.params.w).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AssetDebug {
    Final = 0,
    HullHit = 1,
    Coverage = 2,
    Albedo = 3,
    Normal = 4,
    Depth = 5,
    Weights = 6,
    /// Shade the shell point with no hull march — the flat "decal" look.
    Shell = 7,
    /// (metal, rough, ao) channel dump.
    MrAo = 8,
    /// Shared per-channel parity output (10 albedo, 11 rough, 12 metal,
    /// 13 ao, 14 emissive, 15 world normal) — same code in model_pbr.wgsl.
    Channel(u32),
}

impl AssetDebug {
    pub fn name(self) -> &'static str {
        match self {
            AssetDebug::Final => "final",
            AssetDebug::HullHit => "hull",
            AssetDebug::Coverage => "coverage",
            AssetDebug::Albedo => "albedo",
            AssetDebug::Normal => "normal",
            AssetDebug::Depth => "depth",
            AssetDebug::Weights => "weights",
            AssetDebug::Shell => "shell",
            AssetDebug::MrAo => "mr+ao",
            AssetDebug::Channel(_) => "channel",
        }
    }

    /// Numeric code pushed to the shader (Asset.params.w).
    pub fn code(self) -> u32 {
        match self {
            AssetDebug::Final => 0,
            AssetDebug::HullHit => 1,
            AssetDebug::Coverage => 2,
            AssetDebug::Albedo => 3,
            AssetDebug::Normal => 4,
            AssetDebug::Depth => 5,
            AssetDebug::Weights => 6,
            AssetDebug::Shell => 7,
            AssetDebug::MrAo => 8,
            AssetDebug::Channel(n) => n,
        }
    }

    pub const COUNT: usize = 8;
}

/// One baked impostor: the images, voxel/source-material tables, bounds and
/// CPU thumbnails for the atlas viewer. Channels are 2D arrays whose layer
/// `f` is `VIEWS[f]`.
///
/// Memory is dominated by the 26-view arrays and source voxel records: the
/// 512² lab asset reports ≈191 MiB, the 1024² asset ≈659 MiB.
pub struct ImpostorAsset {
    /// Padded object-space bounds the shell renders.
    pub bounds: ImpostorBounds,
    /// Tile resolution (pixels per view layer edge).
    pub resolution: u32,
    /// Which slab pairs contribute to the hull; a pair whose two views
    /// captured ~no coverage is excluded (flat quads would vanish otherwise).
    pub slab_valid: [bool; super::AXIS_COUNT],
    /// Surface-voxel bitmask resolution; 0 disables voxel refinement.
    pub voxel_resolution: u32,
    /// Word offset of packed triangle records in the voxel buffer.
    pub triangle_base: u32,
    /// Any baked material uses AlphaMode::Mask — then the triangle DDA path
    /// cannot replace the hull (cutout visibility is coverage-driven).
    pub(crate) has_mask_materials: bool,
    /// Unique source-triangle records in the voxel buffer.
    pub triangle_count: u32,
    /// Occupied triangle-surface voxel count.
    pub surface_count: u32,
    /// Source-material texture-array edge used by the exact voxel path.
    pub material_resolution: u32,
    /// Number of material layers in the source-map arrays.
    pub material_count: u32,
    /// GPU bytes across channel arrays, voxel/triangle data and material maps.
    pub bytes: usize,
    /// Wall-clock bake time.
    pub bake_ms: f32,
    /// CPU thumbnails for the atlas panel: `(channel, rgba8, w, h)` — one row
    /// of view tiles per channel.
    pub previews: Vec<(String, Vec<u8>, u32, u32)>,
    /// GPU-test surface: coverage/depth sanity data from the bake.
    #[cfg(test)]
    pub test_stats: Option<super::bake::BakeTestStats>,
    /// Level-0 pixel data kept for the CPU-side bake-consistency test.
    #[cfg(test)]
    pub test_layers: Option<super::bake::BakeTestLayers>,

    pub(crate) albedo: GpuImage,
    pub(crate) normal_ao: GpuImage,
    pub(crate) metallic_rough: GpuImage,
    pub(crate) emissive: GpuImage,
    pub(crate) depth: GpuImage,
    /// One bit per voxel marking triangle-surface occupancy.
    pub(crate) voxels: GpuBuffer,
    /// Source maps as material-indexed texture arrays: base, MR, normal, AO,
    /// emissive. These keep far-field voxel material exact rather than
    /// re-deriving material from a registered capture view.
    pub(crate) material_maps: [GpuImage; 5],
    /// Per-material factors/transforms used by `impostor.wgsl`.
    pub(crate) material_table: GpuBuffer,
    pub(crate) ctx: Arc<GpuContext>,
    /// Set by `ImpostorRenderer::retire_asset`/`collect_retired` once the
    /// submit serial referencing this asset completed — lets `Drop` skip
    /// the conservative device-idle wait.
    pub(crate) drop_without_wait: bool,
}

/// Residency budgets for bake and archive load. Nothing is silently
/// clamped to these — exceeding a limit returns a typed error.
#[derive(Clone, Copy, Debug)]
pub struct ImpostorAssetLimits {
    /// Total GPU bytes the asset may allocate (default 256 MiB).
    pub max_gpu_bytes: u64,
    /// Maximum archive file size accepted by `load_baked`.
    pub max_file_bytes: u64,
    /// Maximum atlas/source-map array edge (default 1024).
    pub max_texture_resolution: u32,
    /// Maximum voxel grid edge; 0 disables the voxel path (default 128).
    pub max_voxel_resolution: u32,
    /// Maximum materials in the source-map arrays (default 32).
    pub max_material_count: u32,
    /// Maximum triangle references per voxel cell — bounds the per-fragment
    /// DDA loop (default 512). Overlong lists are rejected, never truncated.
    pub max_cell_references: u32,
}

impl Default for ImpostorAssetLimits {
    fn default() -> Self {
        Self {
            max_gpu_bytes: 256 * 1024 * 1024,
            max_file_bytes: 256 * 1024 * 1024 + 248,
            max_texture_resolution: 1024,
            max_voxel_resolution: 128,
            max_material_count: 32,
            max_cell_references: 512,
        }
    }
}

impl ImpostorAsset {
    pub fn summary(&self) -> String {
        let mask = self
            .slab_valid
            .iter()
            .map(|v| if *v { '1' } else { '0' })
            .collect::<String>();
        format!(
            "res={} layers={} slab=[{}] bytes={} ({:.1} MiB) bake={:.0}ms",
            self.resolution,
            super::VIEW_COUNT,
            mask,
            self.bytes,
            self.bytes as f32 / (1024.0 * 1024.0),
            self.bake_ms,
        )
    }

    /// Mip count the channel images were built with.
    pub fn mip_levels(&self) -> u32 {
        32 - self.resolution.leading_zeros()
    }
}

impl Drop for ImpostorAsset {
    /// Safe by default: waits for device idle before the GPU members are
    /// released (streaming unloads go through
    /// `ImpostorRenderer::retire_asset`, which marks the asset after its
    /// submit serial completes so this wait is skipped).
    fn drop(&mut self) {
        if !self.drop_without_wait {
            // Hold the submission lock so a racing queue submit can't use
            // this asset's resources mid-wait.
            let _queue = self.ctx.queue_lock.lock().unwrap();
            unsafe {
                let _ = self.ctx.device().device_wait_idle();
            }
        }
    }
}
