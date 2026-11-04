//! Persisted baked-asset archives (V1).
//!
//! `ImpostorAsset::write_baked` serialises the baked GPU resources exactly —
//! every atlas/source-map layer and mip (including MIN depth), the voxel
//! buffer and the material table — uncompressed. `ImpostorAsset::load_baked`
//! validates the whole file (header, sizes, CRCs, voxel structure) before
//! allocating a single GPU handle.
//!
//! Layout (248-byte header, little-endian):
//!   [0..8)    magic `DREAM\0\0\0`
//!   [8)      file_version u32 = 1
//!   [12)     geometry_layout u32 = 1
//!   [16..40) bounds min.xyz + max.xyz (f32)
//!   [40..88) 12 u32: resolution, voxel_resolution, material_resolution,
//!            material_count, triangle_base, triangle_count, surface_count,
//!            slab_mask, has_mask(0/1), VIEW_COUNT, AXIS_COUNT, blob_count
//!   [88)     gpu_bytes u64 (sum of blob lengths)
//!   [96)     view_signature u64 (FNV-1a of the VIEWS dir/u/v float bits)
//!   [104)    12 blob lengths u64
//!   [200)    12 CRC32 (IEEE) u32
//! Blob order: 0..4 atlas albedo/normal_ao/mr/emissive/depth — all mips,
//! layer-major then mip-minor; bytes-per-texel 4/4/4/4/2. 5..9 source
//! material maps base/mr/normal/ao/emissive (4 B/texel). 10 voxel buffer
//! raw u32 LE. 11 material table (224-byte `ImpMaterialParams` records).

use ash::vk;
use glam::Vec3;
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::gfx::{GpuBuffer, GpuContext, GpuImage};
use crate::impostor::atlas::{ImpostorAsset, ImpostorAssetLimits};
use crate::impostor::{AXIS_COUNT, ImpostorBounds, VIEW_COUNT, VIEWS};

/// 8-byte archive magic, NUL-padded to a fixed width. Archives written before
/// the rename carry an earlier magic and are rejected — re-bake them.
const MAGIC: &[u8; 8] = b"DREAM\0\0\0";
const FILE_VERSION: u32 = 1;
const GEOMETRY_LAYOUT: u32 = 1;
const HEADER_LEN: usize = 248;
const BLOB_COUNT: usize = 12;
const TRIANGLE_WORDS: usize = 48;
const MATERIAL_WORDS: usize = 56; // 224-byte ImpMaterialParams
const ATLAS_BPP: [usize; 5] = [4, 4, 4, 4, 2];
const ATLAS_FORMATS: [vk::Format; 5] = [
    vk::Format::R8G8B8A8_SRGB,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::R8G8B8A8_SRGB,
    vk::Format::R16_UNORM,
];
const MAP_FORMATS: [vk::Format; 5] = [
    vk::Format::R8G8B8A8_SRGB,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::R8G8B8A8_SRGB,
];

/// Archive/load error. No GPU handle is allocated for a file that fails
/// validation.
#[derive(Debug)]
pub enum ImpostorAssetError {
    Io(std::io::Error),
    UnsupportedVersion,
    InvalidData(String),
    BudgetExceeded { requested: u64, limit: u64 },
    DeviceLimit(String),
}

impl fmt::Display for ImpostorAssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "impostor asset io: {e}"),
            Self::UnsupportedVersion => write!(f, "unsupported impostor asset version"),
            Self::InvalidData(m) => write!(f, "invalid impostor asset: {m}"),
            Self::BudgetExceeded { requested, limit } => {
                write!(
                    f,
                    "impostor asset needs {requested} bytes over limit {limit}"
                )
            }
            Self::DeviceLimit(m) => write!(f, "impostor asset exceeds device limits: {m}"),
        }
    }
}

impl std::error::Error for ImpostorAssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ImpostorAssetError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn invalid(msg: impl Into<String>) -> ImpostorAssetError {
    ImpostorAssetError::InvalidData(msg.into())
}

// ---- CRC32 (IEEE, table-driven) -------------------------------------------

fn crc32_table() -> &'static [u32; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB88320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *e = c;
        }
        t
    })
}

fn crc32(data: &[u8]) -> u32 {
    let table = crc32_table();
    let mut c = !0u32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

// ---- little-endian header cursor -------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, ImpostorAssetError> {
        let b = self
            .buf
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| invalid("truncated header"))?;
        self.pos += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ImpostorAssetError> {
        let b = self
            .buf
            .get(self.pos..self.pos + 8)
            .ok_or_else(|| invalid("truncated header"))?;
        self.pos += 8;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, ImpostorAssetError> {
        Ok(f32::from_bits(self.u32()?))
    }
}

/// FNV-1a over the little-endian float bits of every VIEWS dir/u/v.
fn view_signature() -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in VIEWS.iter() {
        for f in [v.dir, v.u, v.v].iter().flat_map(|v| v.to_array()) {
            for b in f.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

/// Tight bytes for `layers` layers of a res^2 image across all mips.
fn image_blob_bytes(res: u32, layers: u64, bpp: u64) -> u64 {
    let mips = 32 - res.leading_zeros();
    let mut out = 0u64;
    for _ in 0..layers {
        for m in 0..mips {
            let r = (res >> m).max(1) as u64;
            out += r * r * bpp;
        }
    }
    out
}

fn mip_extent(res: u32, mip: u32) -> u32 {
    (res >> mip).max(1)
}

// ---- GPU readback / upload ---------------------------------------------------

/// Read one image array back to tight `layer-major, mip-minor` bytes. GPU
/// staging pads each (layer, mip) copy to a 4-byte offset; the pads are
/// stripped when the blob is assembled (matters for tiny R16 depth mips).
fn read_image_blob(
    ctx: &Arc<GpuContext>,
    image: &GpuImage,
    res: u32,
    layers: u32,
    bpp: usize,
) -> Vec<u8> {
    let mips = 32 - res.leading_zeros();
    let mut offsets = Vec::new();
    let mut cursor = 0u64;
    for _layer in 0..layers {
        for m in 0..mips {
            offsets.push(cursor);
            let r = mip_extent(res, m) as u64;
            cursor = (cursor + r * r * bpp as u64 + 3) & !3;
        }
    }
    let staging = ctx.create_cached_readback_buffer("asset archive readback", cursor.max(4));
    let device = ctx.device().clone();
    let aspect = vk::ImageAspectFlags::COLOR;
    ctx.one_shot(|cmd| unsafe {
        crate::gfx::vulkan::image_barrier_full(
            &device,
            cmd,
            image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            aspect,
            mips,
            layers,
        );
        let mut i = 0usize;
        for layer in 0..layers {
            for m in 0..mips {
                let r = mip_extent(res, m);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offsets[i])
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: aspect,
                        mip_level: m,
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
                    vk::ImageLayout::GENERAL,
                    staging.buffer,
                    &[region],
                );
                i += 1;
            }
        }
        crate::gfx::vulkan::image_barrier_full(
            &device,
            cmd,
            image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            aspect,
            mips,
            layers,
        );
    });
    let data = staging.mapped().expect("readback mapped").to_vec();
    let mut out = Vec::with_capacity(image_blob_bytes(res, layers as u64, bpp as u64) as usize);
    let mut i = 0usize;
    for _layer in 0..layers {
        for m in 0..mips {
            let n = (mip_extent(res, m) as usize).pow(2) * bpp;
            let off = offsets[i] as usize;
            out.extend_from_slice(&data[off..off + n]);
            i += 1;
        }
    }
    out
}

fn read_buffer_blob(ctx: &Arc<GpuContext>, buf: &GpuBuffer) -> Vec<u8> {
    let staging = ctx.create_cached_readback_buffer("asset buffer readback", buf.size.max(4));
    let device = ctx.device().clone();
    ctx.one_shot(|cmd| unsafe {
        crate::gfx::vulkan::buffer_barrier(
            &device,
            cmd,
            buf.buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_copy_buffer(
            cmd,
            buf.buffer,
            staging.buffer,
            &[vk::BufferCopy::default().size(buf.size)],
        );
        crate::gfx::vulkan::buffer_barrier(
            &device,
            cmd,
            buf.buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        );
    });
    // mapped() exposes the whole allocation (rounded up) — trim to the
    // logical buffer size so the serialized blob matches the header.
    staging.mapped().expect("readback mapped")[..buf.size as usize].to_vec()
}

/// Upload raw blob bytes into a fresh image array, preserving stored mips.
fn write_image_blob(
    ctx: &Arc<GpuContext>,
    label: &str,
    res: u32,
    layers: u32,
    format: vk::Format,
    bpp: usize,
    blob: &[u8],
) -> GpuImage {
    let mips = 32 - res.leading_zeros();
    let mut offsets = Vec::new();
    let mut tight = 0usize;
    let mut padded = Vec::new();
    for _layer in 0..layers {
        for m in 0..mips {
            let r = mip_extent(res, m);
            let n = r as usize * r as usize * bpp;
            offsets.push(padded.len() as u64);
            padded.extend_from_slice(&blob[tight..tight + n]);
            padded.resize((padded.len() + 3) & !3, 0);
            tight += n;
        }
    }
    let mut staging = ctx.create_transfer_buffer("asset archive staging", padded.len() as u64);
    staging.write(0, &padded);
    let image = ctx.create_image_array_mips(
        label,
        res,
        res,
        format,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        layers,
        mips,
    );
    let device = ctx.device().clone();
    let aspect = vk::ImageAspectFlags::COLOR;
    ctx.one_shot(|cmd| unsafe {
        crate::gfx::vulkan::image_barrier_full(
            &device,
            cmd,
            image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            aspect,
            mips,
            layers,
        );
        let mut i = 0usize;
        for layer in 0..layers {
            for m in 0..mips {
                let r = mip_extent(res, m);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offsets[i])
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: aspect,
                        mip_level: m,
                        base_array_layer: layer,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: r,
                        height: r,
                        depth: 1,
                    });
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    &[region],
                );
                i += 1;
            }
        }
        crate::gfx::vulkan::image_barrier_full(
            &device,
            cmd,
            image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            aspect,
            mips,
            layers,
        );
    });
    image
}

// ---- decode -----------------------------------------------------------------

/// A fully validated archive: metadata + all 12 blobs (CRC-verified).
struct ParsedArchive {
    bounds_min: Vec3,
    bounds_max: Vec3,
    resolution: u32,
    voxel_resolution: u32,
    material_resolution: u32,
    material_count: u32,
    triangle_base: u32,
    triangle_count: u32,
    surface_count: u32,
    slab_mask: u32,
    has_mask: bool,
    gpu_bytes: u64,
    blobs: Vec<Vec<u8>>,
}

/// Read and fully validate an archive. Pure CPU — `load_baked` only uploads
/// to the GPU after this succeeds.
fn read_archive(
    path: &Path,
    limits: &ImpostorAssetLimits,
    dev_max_image_dim: u32,
    dev_max_array_layers: u32,
) -> Result<ParsedArchive, ImpostorAssetError> {
    let meta = std::fs::metadata(path)?;
    let file_len = meta.len();
    if file_len > limits.max_file_bytes {
        return Err(ImpostorAssetError::BudgetExceeded {
            requested: file_len,
            limit: limits.max_file_bytes,
        });
    }
    if file_len < HEADER_LEN as u64 {
        return Err(invalid("truncated header"));
    }
    let mut file = std::fs::File::open(path)?;
    let mut header = vec![0u8; HEADER_LEN];
    file.read_exact(&mut header)?;
    if header[..8] != *MAGIC {
        return Err(invalid(
            "magic mismatch: expected the `DREAM` impostor header",
        ));
    }
    let mut r = Reader {
        buf: &header,
        pos: 8,
    };
    if r.u32()? != FILE_VERSION {
        return Err(ImpostorAssetError::UnsupportedVersion);
    }
    if r.u32()? != GEOMETRY_LAYOUT {
        return Err(invalid("unsupported geometry layout"));
    }
    let bounds_min = Vec3::new(r.f32()?, r.f32()?, r.f32()?);
    let bounds_max = Vec3::new(r.f32()?, r.f32()?, r.f32()?);
    let f12: Vec<u32> = (0..12).map(|_| r.u32()).collect::<Result<_, _>>()?;
    let gpu_bytes = r.u64()?;
    let sig = r.u64()?;
    let lengths: Vec<u64> = (0..BLOB_COUNT).map(|_| r.u64()).collect::<Result<_, _>>()?;
    let crcs: Vec<u32> = (0..BLOB_COUNT).map(|_| r.u32()).collect::<Result<_, _>>()?;

    if f12[9] != VIEW_COUNT as u32 || f12[10] != AXIS_COUNT as u32 || f12[11] != BLOB_COUNT as u32 {
        return Err(invalid("view/axis/blob constants mismatch"));
    }
    if sig != view_signature() {
        return Err(invalid("view signature mismatch"));
    }
    let has_mask = match f12[8] {
        0 => false,
        1 => true,
        _ => return Err(invalid("invalid has_mask flag")),
    };
    let extent = bounds_max - bounds_min;
    let center = bounds_min * 0.5 + bounds_max * 0.5;
    if !bounds_min.is_finite()
        || !bounds_max.is_finite()
        || !extent.is_finite()
        || !extent.cmpgt(Vec3::ZERO).all()
        || !center.is_finite()
    {
        return Err(invalid("impossible bounds"));
    }
    let (resolution, voxel_resolution, material_resolution, material_count) =
        (f12[0], f12[1], f12[2], f12[3]);
    let (triangle_base, triangle_count, surface_count, slab_mask) =
        (f12[4], f12[5], f12[6], f12[7]);
    if slab_mask > (1 << AXIS_COUNT) - 1 {
        return Err(invalid("slab mask out of range"));
    }
    let tex_cap = limits.max_texture_resolution.min(dev_max_image_dim);
    if !resolution.is_power_of_two() || !(16..=1024).contains(&resolution) || resolution > tex_cap {
        return Err(invalid("atlas resolution out of range"));
    }
    if voxel_resolution != 0
        && (!voxel_resolution.is_power_of_two()
            || voxel_resolution > 256
            || voxel_resolution > limits.max_voxel_resolution)
    {
        return Err(invalid("voxel resolution out of range"));
    }
    if !material_resolution.is_power_of_two()
        || material_resolution > 1024
        || material_resolution > tex_cap
    {
        return Err(invalid("material resolution out of range"));
    }
    if material_count == 0
        || material_count > limits.max_material_count
        || material_count > dev_max_array_layers
    {
        return Err(invalid("material count out of range"));
    }
    if VIEW_COUNT as u32 > dev_max_array_layers {
        return Err(invalid("device cannot hold the atlas layers"));
    }

    // Expected blob sizes — checked arithmetic, exact match required.
    let mut expected = Vec::with_capacity(BLOB_COUNT);
    for bpp in ATLAS_BPP {
        expected.push(image_blob_bytes(resolution, VIEW_COUNT as u64, bpp as u64));
    }
    for _ in 0..5 {
        expected.push(image_blob_bytes(
            material_resolution,
            material_count as u64,
            4,
        ));
    }
    let voxel_words = if voxel_resolution == 0 {
        1u64
    } else {
        // triangle_base covers dense cells + packed refs; the buffer is
        // exactly triangle_base + 48 words per unique record.
        u64::from(triangle_base) + TRIANGLE_WORDS as u64 * u64::from(triangle_count)
    };
    expected.push(voxel_words * 4);
    expected.push(u64::from(material_count) * 224);
    for (i, (&len, &exp)) in lengths.iter().zip(&expected).enumerate() {
        if len != exp {
            return Err(invalid(format!("blob {i} length {len} != expected {exp}")));
        }
    }
    let total: u64 = lengths.iter().sum();
    if total != gpu_bytes {
        return Err(invalid("gpu_bytes mismatch"));
    }
    if file_len != HEADER_LEN as u64 + total {
        return Err(invalid("file length mismatch (trailing/truncated)"));
    }
    if gpu_bytes > limits.max_gpu_bytes {
        return Err(ImpostorAssetError::BudgetExceeded {
            requested: gpu_bytes,
            limit: limits.max_gpu_bytes,
        });
    }

    // Read + CRC every blob before any GPU allocation.
    let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(BLOB_COUNT);
    for (i, (&len, &crc)) in lengths.iter().zip(&crcs).enumerate() {
        let mut b = vec![0u8; len as usize];
        file.read_exact(&mut b)
            .map_err(|_| invalid("truncated blob"))?;
        if crc32(&b) != crc {
            return Err(invalid(format!("blob {i} crc mismatch")));
        }
        blobs.push(b);
    }

    // Packed voxel layout: dense 3-word cells, ascending packed ref lists,
    // 48-word triangle records.
    let words: Vec<u32> = blobs[10]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if voxel_resolution == 0 {
        if words != [0] || triangle_base != 1 || triangle_count != 0 || surface_count != 0 {
            return Err(invalid("voxel-disabled layout must be empty"));
        }
    } else {
        let dense = (voxel_resolution as u64).pow(3) * 3;
        if u64::from(triangle_base) < dense
            || words.len() as u64
                != u64::from(triangle_base) + TRIANGLE_WORDS as u64 * u64::from(triangle_count)
        {
            return Err(invalid("voxel buffer layout mismatch"));
        }
        let mut refs_cursor = dense;
        let mut occupied = 0u64;
        for i in 0..voxel_resolution.pow(3) as usize {
            let seed = words[i * 3];
            let attr = words[i * 3 + 2];
            if seed == 0 {
                if attr != 0 {
                    return Err(invalid("empty voxel with reference list"));
                }
                continue;
            }
            occupied += 1;
            if attr == 0 {
                return Err(invalid("occupied voxel without reference list"));
            }
            if dense + u64::from(attr) - 1 != refs_cursor {
                return Err(invalid("reference list order mismatch"));
            }
            // A dangling attr (e.g. triangle_base==dense with an occupied
            // cell) must fail typed, not index out of bounds.
            let list_base = refs_cursor as usize;
            let Some(count) = words.get(list_base).map(|w| u64::from(*w)) else {
                return Err(invalid("reference list header out of bounds"));
            };
            if count == 0 {
                return Err(invalid("empty reference list"));
            }
            if count > u64::from(limits.max_cell_references) {
                return Err(invalid(format!(
                    "reference list length {count} out of range"
                )));
            }
            if refs_cursor + 1 + count > u64::from(triangle_base) {
                return Err(invalid("reference list out of bounds"));
            }
            for k in 0..count as usize {
                let t = words
                    .get(list_base + 1 + k)
                    .copied()
                    .ok_or_else(|| invalid("reference out of bounds"))?;
                if t == 0 || t > triangle_count {
                    return Err(invalid("triangle reference out of range"));
                }
            }
            refs_cursor += 1 + count;
        }
        if occupied != u64::from(surface_count) {
            return Err(invalid("surface_count mismatch"));
        }
        if refs_cursor != u64::from(triangle_base) {
            return Err(invalid("reference region does not end at triangle_base"));
        }
        let rec_base = triangle_base as usize;
        for t in 0..triangle_count as usize {
            let base = rec_base + t * TRIANGLE_WORDS;
            for w in 0..42 {
                let w32 = words
                    .get(base + w)
                    .copied()
                    .ok_or_else(|| invalid("triangle record out of bounds"))?;
                if !f32::from_bits(w32).is_finite() {
                    return Err(invalid("non-finite triangle record word"));
                }
            }
            let mid = words
                .get(base + 42)
                .copied()
                .ok_or_else(|| invalid("triangle record out of bounds"))?;
            if mid >= material_count {
                return Err(invalid("triangle material out of range"));
            }
        }
    }
    // Non-empty content: a legal-but-empty pack must not silently draw a
    // black box — require at least one valid slab axis, any covered albedo
    // texel, and (when voxel-enabled) occupied cells + triangles.
    if slab_mask == 0 {
        return Err(invalid("no valid slab axes"));
    }
    if !blobs[0].chunks_exact(4).any(|p| p[3] != 0) {
        return Err(invalid("albedo atlas has zero coverage"));
    }
    if voxel_resolution != 0 && (surface_count == 0 || triangle_count == 0) {
        return Err(invalid(
            "voxel-enabled asset has no occupied cells/triangles",
        ));
    }
    // Material table: finite floats; alpha mode 0/1 only; derived has_mask
    // must match the header flag.
    {
        let table: Vec<f32> = blobs[11]
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect();
        if table.len() != material_count as usize * MATERIAL_WORDS {
            return Err(invalid("material table size mismatch"));
        }
        if !table.iter().all(|v| v.is_finite()) {
            return Err(invalid("non-finite material table"));
        }
        let mut derived_mask = false;
        for m in 0..material_count as usize {
            let mode = table[m * MATERIAL_WORDS + 7]; // params[3]
            if mode != 0.0 && mode != 1.0 {
                return Err(invalid(
                    "material alpha mode out of range (BLEND not allowed)",
                ));
            }
            derived_mask |= mode == 1.0;
        }
        if derived_mask != has_mask {
            return Err(invalid("has_mask flag mismatches material table"));
        }
    }

    Ok(ParsedArchive {
        bounds_min,
        bounds_max,
        resolution,
        voxel_resolution,
        material_resolution,
        material_count,
        triangle_base,
        triangle_count,
        surface_count,
        slab_mask,
        has_mask,
        gpu_bytes,
        blobs,
    })
}

// ---- ImpostorAsset methods ---------------------------------------------------

impl ImpostorAsset {
    /// Serialise the baked asset to `path` (uncompressed, exact mips).
    /// `create_new` — refuses to overwrite; a partially written file is
    /// removed on error. Blocks on GPU readback.
    pub fn write_baked(&self, path: impl AsRef<Path>) -> Result<(), ImpostorAssetError> {
        let path = path.as_ref();
        // Reserve the file BEFORE the GPU readback: create_new fails fast on
        // an existing path, and the only file we ever remove on error is the
        // one we just created. A GPU-side panic can leave a reserved empty
        // file behind — acceptable and documented here.
        let mut file = std::fs::File::create_new(path)?;
        let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(BLOB_COUNT);
        let atlas_images = [
            &self.albedo,
            &self.normal_ao,
            &self.metallic_rough,
            &self.emissive,
            &self.depth,
        ];
        for (i, image) in atlas_images.iter().enumerate() {
            blobs.push(read_image_blob(
                &self.ctx,
                image,
                self.resolution,
                VIEW_COUNT as u32,
                ATLAS_BPP[i],
            ));
        }
        for image in &self.material_maps {
            blobs.push(read_image_blob(
                &self.ctx,
                image,
                self.material_resolution,
                self.material_count,
                4,
            ));
        }
        blobs.push(read_buffer_blob(&self.ctx, &self.voxels));
        blobs.push(read_buffer_blob(&self.ctx, &self.material_table));

        let gpu_bytes: u64 = blobs.iter().map(|b| b.len() as u64).sum();
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FILE_VERSION.to_le_bytes());
        header.extend_from_slice(&GEOMETRY_LAYOUT.to_le_bytes());
        for v in self
            .bounds
            .min
            .to_array()
            .iter()
            .chain(self.bounds.max.to_array().iter())
        {
            header.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let slab_mask: u32 = self
            .slab_valid
            .iter()
            .enumerate()
            .map(|(i, v)| (*v as u32) << i)
            .sum();
        for v in [
            self.resolution,
            self.voxel_resolution,
            self.material_resolution,
            self.material_count,
            self.triangle_base,
            self.triangle_count,
            self.surface_count,
            slab_mask,
            self.has_mask_materials as u32,
            VIEW_COUNT as u32,
            AXIS_COUNT as u32,
            BLOB_COUNT as u32,
        ] {
            header.extend_from_slice(&v.to_le_bytes());
        }
        header.extend_from_slice(&gpu_bytes.to_le_bytes());
        header.extend_from_slice(&view_signature().to_le_bytes());
        for b in &blobs {
            header.extend_from_slice(&(b.len() as u64).to_le_bytes());
        }
        for b in &blobs {
            header.extend_from_slice(&crc32(b).to_le_bytes());
        }
        debug_assert_eq!(header.len(), HEADER_LEN);

        // Write into the reserved handle; on failure remove ONLY the file
        // this call created (never a pre-existing path).
        let write_result = (|| -> Result<(), ImpostorAssetError> {
            file.write_all(&header)?;
            for b in &blobs {
                file.write_all(b)?;
            }
            Ok(())
        })();
        if write_result.is_err() {
            drop(file);
            let _ = std::fs::remove_file(path);
        }
        write_result
    }

    /// Load a baked archive. Validates header, budgets, sizes, CRCs and the
    /// packed voxel layout BEFORE allocating any GPU resource.
    pub fn load_baked(
        ctx: &Arc<GpuContext>,
        path: impl AsRef<Path>,
        limits: ImpostorAssetLimits,
    ) -> Result<Self, ImpostorAssetError> {
        let dev_limits = unsafe {
            ctx.instance()
                .get_physical_device_properties(ctx.physical_device())
                .limits
        };
        let decoded = read_archive(
            path.as_ref(),
            &limits,
            dev_limits.max_image_dimension2_d,
            dev_limits.max_image_array_layers,
        )?;
        let ParsedArchive {
            bounds_min,
            bounds_max,
            resolution,
            voxel_resolution,
            material_resolution,
            material_count,
            triangle_base,
            triangle_count,
            surface_count,
            slab_mask,
            has_mask,
            gpu_bytes,
            blobs,
        } = decoded;
        let sb_range = u64::from(ctx.max_storage_buffer_range());
        if blobs[10].len() as u64 > sb_range || blobs[11].len() as u64 > sb_range {
            return Err(ImpostorAssetError::DeviceLimit(
                "storage buffer exceeds maxStorageBufferRange".into(),
            ));
        }
        let atlas_images: [GpuImage; 5] = std::array::from_fn(|i| {
            write_image_blob(
                ctx,
                &format!("impostor atlas {i}"),
                resolution,
                VIEW_COUNT as u32,
                ATLAS_FORMATS[i],
                ATLAS_BPP[i],
                &blobs[i],
            )
        });
        let material_maps: [GpuImage; 5] = std::array::from_fn(|i| {
            write_image_blob(
                ctx,
                &format!("impostor material map {i}"),
                material_resolution,
                material_count,
                MAP_FORMATS[i],
                4,
                &blobs[5 + i],
            )
        });
        let voxels = ctx.upload_buffer(
            "impostor surface voxels",
            &blobs[10],
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        let material_table = ctx.upload_buffer(
            "impostor material table",
            &blobs[11],
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        let [albedo, normal_ao, metallic_rough, emissive, depth] = atlas_images;
        Ok(ImpostorAsset {
            bounds: ImpostorBounds {
                min: bounds_min,
                max: bounds_max,
            },
            resolution,
            voxel_resolution,
            slab_valid: std::array::from_fn(|i| slab_mask & (1 << i) != 0),
            material_resolution,
            material_count,
            triangle_base,
            triangle_count,
            surface_count,
            has_mask_materials: has_mask,
            bytes: gpu_bytes as usize,
            bake_ms: 0.0,
            previews: Vec::new(),
            #[cfg(test)]
            test_stats: None,
            #[cfg(test)]
            test_layers: None,
            albedo,
            normal_ao,
            metallic_rough,
            emissive,
            depth,
            voxels,
            material_maps,
            material_table,
            ctx: Arc::clone(ctx),
            drop_without_wait: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    /// Build the smallest legal archive in memory: res 16, voxel 0, one
    /// opaque material at 1x1 maps.
    fn minimal_archive() -> Vec<u8> {
        let res = 16u32;
        let mat_res = 1u32;
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        for bpp in ATLAS_BPP {
            let mut blob =
                vec![0xABu8; image_blob_bytes(res, VIEW_COUNT as u64, bpp as u64) as usize];
            if bpp as usize == ATLAS_BPP[0] {
                blob[3] = 255; // real albedo coverage
            }
            blobs.push(blob);
        }
        for _ in 0..5 {
            blobs.push(vec![0xCDu8; image_blob_bytes(mat_res, 1, 4) as usize]);
        }
        blobs.push(0u32.to_le_bytes().to_vec()); // voxel: [0]
        blobs.push(vec![0u8; 224]); // one opaque material record
        let gpu_bytes: u64 = blobs.iter().map(|b| b.len() as u64).sum();

        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FILE_VERSION.to_le_bytes());
        header.extend_from_slice(&GEOMETRY_LAYOUT.to_le_bytes());
        for v in [-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0] {
            header.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in [
            res,
            0, // voxel_resolution 0
            mat_res,
            1, // material_count
            1, // triangle_base
            0, // triangle_count
            0, // surface_count
            1, // slab_mask (axis 0 valid)
            0, // has_mask
            VIEW_COUNT as u32,
            AXIS_COUNT as u32,
            BLOB_COUNT as u32,
        ] {
            header.extend_from_slice(&v.to_le_bytes());
        }
        header.extend_from_slice(&gpu_bytes.to_le_bytes());
        header.extend_from_slice(&view_signature().to_le_bytes());
        for b in &blobs {
            header.extend_from_slice(&(b.len() as u64).to_le_bytes());
        }
        for b in &blobs {
            header.extend_from_slice(&crc32(b).to_le_bytes());
        }
        assert_eq!(header.len(), HEADER_LEN);
        let mut out = header;
        for b in &blobs {
            out.extend_from_slice(&b);
        }
        out
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "dream_stage2_{}_{}_{}.dreami",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn decode(bytes: &[u8], limits: &ImpostorAssetLimits) -> Result<(), ImpostorAssetError> {
        let path = temp_path("dec");
        std::fs::write(&path, bytes).unwrap();
        let r = read_archive(&path, limits, u32::MAX, u32::MAX).map(|_| ());
        let _ = std::fs::remove_file(&path);
        r
    }

    #[test]
    fn archive_roundtrip_decode() {
        let bytes = minimal_archive();
        decode(&bytes, &ImpostorAssetLimits::default()).expect("minimal archive decodes");
    }

    #[test]
    fn archive_rejects_malformed() {
        use ImpostorAssetError as E;
        let good = minimal_archive();
        let lim = ImpostorAssetLimits::default();
        let mut b = good.clone();
        b[0] = b'X';
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        let mut b = good.clone();
        b[8] = 9;
        assert!(matches!(decode(&b, &lim), Err(E::UnsupportedVersion)));
        let mut b = good.clone();
        b[12] = 9;
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        let mut b = good.clone();
        b[96] ^= 1;
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        let mut b = good.clone();
        b[16..40].copy_from_slice(&[0u8; 24]);
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        let mut b = good.clone();
        b[72] = 7;
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        let mut b = good.clone();
        let last = b.len() - 1;
        b[last] ^= 0xff;
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        assert!(matches!(
            decode(&good[..good.len() - 4], &lim),
            Err(E::InvalidData(_))
        ));
        let mut b = good.clone();
        b.extend_from_slice(&[0u8; 8]);
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        let mut b = good.clone();
        let len_off = 104;
        let l = u64::from_le_bytes(b[len_off..len_off + 8].try_into().unwrap()) + 4;
        b[len_off..len_off + 8].copy_from_slice(&l.to_le_bytes());
        assert!(matches!(decode(&b, &lim), Err(E::InvalidData(_))));
        let tiny = ImpostorAssetLimits {
            max_gpu_bytes: 8,
            ..ImpostorAssetLimits::default()
        };
        assert!(matches!(
            decode(&good, &tiny),
            Err(E::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn archive_voxel_structure_validation() {
        let res = 16u32;
        let vox = 16u32;
        let mat_res = 1u32;
        let dense = vox.pow(3) as usize * 3;
        let mut cells = vec![0u32; dense];
        cells[0] = 1; // occupied seed
        cells[2] = 1; // attr = refs cursor + 1
        let mut words = cells.clone();
        words.push(1); // ref count
        words.push(1); // ref -> triangle 1
        let triangle_base = words.len() as u32;
        words.extend(std::iter::repeat(0u32).take(TRIANGLE_WORDS));
        let voxel_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let build = |voxel: &[u8], mask: u32| -> Vec<u8> {
            let mut blobs: Vec<Vec<u8>> = Vec::new();
            for (i, bpp) in ATLAS_BPP.iter().enumerate() {
                let mut blob =
                    vec![0u8; image_blob_bytes(res, VIEW_COUNT as u64, *bpp as u64) as usize];
                if i == 0 {
                    blob[3] = 255;
                }
                blobs.push(blob);
            }
            for _ in 0..5 {
                blobs.push(vec![0u8; image_blob_bytes(mat_res, 1, 4) as usize]);
            }
            blobs.push(voxel.to_vec());
            let mut table = vec![0u8; 224];
            if mask == 1 {
                let off = 7 * 4; // params[3] alpha mode -> Mask
                table[off..off + 4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
            }
            blobs.push(table);
            let gpu_bytes: u64 = blobs.iter().map(|b| b.len() as u64).sum();
            let mut header = Vec::new();
            header.extend_from_slice(MAGIC);
            header.extend_from_slice(&FILE_VERSION.to_le_bytes());
            header.extend_from_slice(&GEOMETRY_LAYOUT.to_le_bytes());
            for v in [-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0] {
                header.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            for v in [
                res,
                vox,
                mat_res,
                1,
                triangle_base,
                1,
                1,
                1, // slab_mask (axis 0)
                mask,
                VIEW_COUNT as u32,
                AXIS_COUNT as u32,
                BLOB_COUNT as u32,
            ] {
                header.extend_from_slice(&v.to_le_bytes());
            }
            header.extend_from_slice(&gpu_bytes.to_le_bytes());
            header.extend_from_slice(&view_signature().to_le_bytes());
            for b in &blobs {
                header.extend_from_slice(&(b.len() as u64).to_le_bytes());
            }
            for b in &blobs {
                header.extend_from_slice(&crc32(b).to_le_bytes());
            }
            let mut out = header;
            for b in &blobs {
                out.extend_from_slice(&b);
            }
            out
        };

        let lim = ImpostorAssetLimits::default();
        let valid = build(&voxel_bytes, 0);
        decode(&valid, &lim).expect("voxel archive decodes");

        // material id out of range: record word 42 = 5 (> count 1)
        let mut bad = voxel_bytes.clone();
        let w42 = (triangle_base as usize + 42) * 4;
        bad[w42..w42 + 4].copy_from_slice(&5u32.to_le_bytes());
        assert!(matches!(
            decode(&build(&bad, 0), &lim),
            Err(ImpostorAssetError::InvalidData(_))
        ));
        // ref out of range (ref=2 > tricount=1)
        let mut bad = voxel_bytes.clone();
        let roff = (dense + 1) * 4;
        bad[roff..roff + 4].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            decode(&build(&bad, 0), &lim),
            Err(ImpostorAssetError::InvalidData(_))
        ));
        // has_mask flag mismatches material table (header 0, table Mask)
        let mut mism = build(&voxel_bytes, 1);
        mism[72] = 0;
        assert!(matches!(
            decode(&mism, &lim),
            Err(ImpostorAssetError::InvalidData(_))
        ));
        // empty cell with dangling attr
        let mut bad = voxel_bytes.clone();
        bad[(3 + 2) * 4..(3 + 2) * 4 + 4].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            decode(&build(&bad, 0), &lim),
            Err(ImpostorAssetError::InvalidData(_))
        ));
    }

    #[test]
    fn cell_reference_limit_enforced() {
        // A refs list of 513 at default limit 512 must reject; explicit
        // limit 1024 accepts it. Never truncated.
        let res = 16u32;
        let vox = 16u32;
        let mat_res = 1u32;
        let dense = vox.pow(3) as usize * 3;
        let mut cells = vec![0u32; dense];
        cells[0] = 1;
        cells[2] = 1;
        let n = 513u32;
        let mut words = cells.clone();
        words.push(n);
        for t in 1..=n {
            words.push(t % 64 + 1); // any in-range refs
        }
        // triangle_base must cover dense + refs; records follow. Use 64
        // records (refs capped at 64 each).
        let triangle_base = words.len() as u32;
        let tricount = 64u32;
        words.extend(std::iter::repeat(0u32).take(TRIANGLE_WORDS * tricount as usize));
        let voxel_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let build_with = |voxel: &[u8], tb: u32, tc: u32| -> Vec<u8> {
            let mut blobs: Vec<Vec<u8>> = Vec::new();
            for (i, bpp) in ATLAS_BPP.iter().enumerate() {
                let mut blob =
                    vec![0u8; image_blob_bytes(res, VIEW_COUNT as u64, *bpp as u64) as usize];
                if i == 0 {
                    blob[3] = 255;
                }
                blobs.push(blob);
            }
            for _ in 0..5 {
                blobs.push(vec![0u8; image_blob_bytes(mat_res, 1, 4) as usize]);
            }
            blobs.push(voxel.to_vec());
            blobs.push(vec![0u8; 224]);
            let gpu_bytes: u64 = blobs.iter().map(|b| b.len() as u64).sum();
            let mut header = Vec::new();
            header.extend_from_slice(MAGIC);
            header.extend_from_slice(&FILE_VERSION.to_le_bytes());
            header.extend_from_slice(&GEOMETRY_LAYOUT.to_le_bytes());
            for v in [-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0] {
                header.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            for v in [
                res,
                vox,
                mat_res,
                1,
                tb,
                tc,
                1,
                1,
                0,
                VIEW_COUNT as u32,
                AXIS_COUNT as u32,
                BLOB_COUNT as u32,
            ] {
                header.extend_from_slice(&v.to_le_bytes());
            }
            header.extend_from_slice(&gpu_bytes.to_le_bytes());
            header.extend_from_slice(&view_signature().to_le_bytes());
            for b in &blobs {
                header.extend_from_slice(&(b.len() as u64).to_le_bytes());
            }
            for b in &blobs {
                header.extend_from_slice(&crc32(b).to_le_bytes());
            }
            let mut out = header;
            for b in &blobs {
                out.extend_from_slice(&b);
            }
            out
        };

        let bytes = build_with(&voxel_bytes, triangle_base, tricount);
        assert!(matches!(
            decode(&bytes, &ImpostorAssetLimits::default()),
            Err(ImpostorAssetError::InvalidData(_)),
        ));
        let wide = ImpostorAssetLimits {
            max_cell_references: 1024,
            ..ImpostorAssetLimits::default()
        };
        decode(&bytes, &wide).expect("limit 1024 accepts 513 refs");
        // 41 refs always fine at default — rebuild archive with the
        // consistent triangle_base for that refs length.
        let n41 = 41u32;
        let mut w41 = cells.clone();
        w41.push(n41);
        for t in 1..=n41 {
            w41.push(t % 64 + 1);
        }
        let tb41 = w41.len() as u32;
        w41.extend(std::iter::repeat(0u32).take(TRIANGLE_WORDS * tricount as usize));
        let v41: Vec<u8> = w41.iter().flat_map(|w| w.to_le_bytes()).collect();
        let bytes41 = build_with(&v41, tb41, tricount);
        decode(&bytes41, &ImpostorAssetLimits::default()).expect("41 refs allowed");
    }

    #[test]
    fn archive_dangling_attr_no_panic() {
        // triangle_base == dense (no refs region) but an occupied cell with
        // attr=1: must fail typed, never index out of bounds.
        let res = 16u32;
        let vox = 16u32;
        let dense = vox.pow(3) as usize * 3;
        let mut words = vec![0u32; dense];
        words[0] = 1;
        words[2] = 1; // dangling attr into a refs region that doesn't exist
        let voxel: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mat_res = 1u32;
        let triangle_base = dense as u32;
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        for (i, bpp) in ATLAS_BPP.iter().enumerate() {
            let mut blob =
                vec![0u8; image_blob_bytes(res, VIEW_COUNT as u64, *bpp as u64) as usize];
            if i == 0 {
                blob[3] = 255;
            }
            blobs.push(blob);
        }
        for _ in 0..5 {
            blobs.push(vec![0u8; image_blob_bytes(mat_res, 1, 4) as usize]);
        }
        blobs.push(voxel);
        blobs.push(vec![0u8; 224]);
        let gpu_bytes: u64 = blobs.iter().map(|b| b.len() as u64).sum();
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FILE_VERSION.to_le_bytes());
        header.extend_from_slice(&GEOMETRY_LAYOUT.to_le_bytes());
        for v in [-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0] {
            header.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in [
            res,
            vox,
            mat_res,
            1,
            triangle_base,
            0,
            1,
            1,
            0,
            VIEW_COUNT as u32,
            AXIS_COUNT as u32,
            BLOB_COUNT as u32,
        ] {
            header.extend_from_slice(&v.to_le_bytes());
        }
        header.extend_from_slice(&gpu_bytes.to_le_bytes());
        header.extend_from_slice(&view_signature().to_le_bytes());
        for b in &blobs {
            header.extend_from_slice(&(b.len() as u64).to_le_bytes());
        }
        for b in &blobs {
            header.extend_from_slice(&crc32(b).to_le_bytes());
        }
        let mut out = header;
        for b in &blobs {
            out.extend_from_slice(&b);
        }
        let r = decode(&out, &ImpostorAssetLimits::default());
        assert!(matches!(r, Err(ImpostorAssetError::InvalidData(_))));
    }

    #[test]
    fn write_baked_preserves_existing_file() {
        let Some(ctx) = gpu_ctx() else {
            return;
        };
        let baker = crate::impostor::ImpostorBaker::new(&ctx);
        let model = crate::impostor::bake::tests::cube_model();
        let settings = crate::impostor::BakeSettings {
            resolution: 64,
            supersample: 1,
            voxel_resolution: 32,
            material_resolution: 32,
        };
        let asset = baker.bake(&model, settings).expect("bake");
        let path = temp_path("exists");
        std::fs::write(&path, b"USER DATA").unwrap();
        let err = asset.write_baked(&path).unwrap_err();
        assert!(matches!(err, ImpostorAssetError::Io(_)));
        assert_eq!(std::fs::read(&path).unwrap(), b"USER DATA");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_buffer_exact_size() {
        // Allocations are rounded; a 12-byte buffer must serialize as
        // exactly 12 bytes, not the padded allocation.
        let Some(ctx) = gpu_ctx() else {
            return;
        };
        let data: Vec<u8> = (0u8..12).collect();
        let buf = ctx.upload_buffer(
            "test 12B",
            &data,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        assert_eq!(read_buffer_blob(&ctx, &buf), data);
    }

    #[test]
    fn archive_gpu_roundtrip() {
        let Some(ctx) = gpu_ctx() else {
            return;
        };
        let baker = crate::impostor::ImpostorBaker::new(&ctx);
        let model = crate::impostor::bake::tests::cube_model();
        let settings = crate::impostor::BakeSettings {
            resolution: 64,
            supersample: 1,
            voxel_resolution: 32,
            material_resolution: 32,
        };
        let asset = baker.bake(&model, settings).expect("bake");
        let p1 = temp_path("a1");
        asset.write_baked(&p1).expect("write");
        let loaded =
            ImpostorAsset::load_baked(&ctx, &p1, ImpostorAssetLimits::default()).expect("load");
        let p2 = temp_path("a2");
        loaded.write_baked(&p2).expect("rewrite");
        assert_eq!(
            std::fs::read(&p1).unwrap(),
            std::fs::read(&p2).unwrap(),
            "reloaded asset must reserialise byte-identical"
        );
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }
}
