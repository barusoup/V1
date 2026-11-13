//! Persisted baked-asset archives (V1) — Metal backend.
//!
//! Uses the exact same 248-byte little-endian `.dimp` binary layout and CRC32
//! validation as `storage.rs`.

use glam::Vec3;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLOrigin, MTLPixelFormat, MTLSize,
};
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::gfx::{GpuBuffer, GpuContext, GpuImage};
use crate::impostor::atlas::{ImpostorAsset, ImpostorAssetLimits};
use crate::impostor::{AXIS_COUNT, ImpostorBounds, VIEW_COUNT, VIEWS};

const MAGIC: &[u8; 8] = b"DREAM\0\0\0";
const FILE_VERSION: u32 = 1;
const GEOMETRY_LAYOUT: u32 = 1;
const HEADER_LEN: usize = 248;
const BLOB_COUNT: usize = 12;
const TRIANGLE_WORDS: usize = 48;
const MATERIAL_WORDS: usize = 56;
const ATLAS_BPP: [usize; 5] = [4, 4, 4, 4, 2];
const ATLAS_FORMATS: [MTLPixelFormat; 5] = [
    MTLPixelFormat::RGBA8Unorm_sRGB,
    MTLPixelFormat::RGBA8Unorm,
    MTLPixelFormat::RGBA8Unorm,
    MTLPixelFormat::RGBA8Unorm_sRGB,
    MTLPixelFormat::R16Unorm,
];
const MAP_FORMATS: [MTLPixelFormat; 5] = [
    MTLPixelFormat::RGBA8Unorm_sRGB,
    MTLPixelFormat::RGBA8Unorm,
    MTLPixelFormat::RGBA8Unorm,
    MTLPixelFormat::RGBA8Unorm,
    MTLPixelFormat::RGBA8Unorm_sRGB,
];

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

fn atlas_blob_bytes(res: u32, mips: u32, bpp: usize) -> usize {
    let mut total = 0usize;
    let mut edge = res;
    for _ in 0..mips {
        total += (edge * edge) as usize * bpp * VIEW_COUNT;
        edge = (edge / 2).max(1);
    }
    total
}

fn readback_image_blob(
    ctx: &Arc<GpuContext>,
    image: &GpuImage,
    res: u32,
    layers: u32,
    mips: u32,
    bpp: usize,
) -> Vec<u8> {
    let mut offsets = Vec::with_capacity((layers * mips) as usize);
    let mut total = 0usize;
    for _layer in 0..layers {
        let mut edge = res;
        for _mip in 0..mips {
            let len = (edge * edge) as usize * bpp;
            offsets.push((total, edge, len));
            total += len;
            edge = (edge / 2).max(1);
        }
    }
    let staging = ctx.create_readback_buffer("dimp readback", total.max(4) as u64);
    ctx.one_shot(|cmd| {
        let blit = cmd.blitCommandEncoder().expect("dimp readback blit");
        let mut idx = 0usize;
        for layer in 0..layers {
            let mut edge = res;
            for mip in 0..mips {
                let (off, _, _) = offsets[idx];
                idx += 1;
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        &image.texture,
                        layer as usize,
                        mip as usize,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                        MTLSize {
                            width: edge as usize,
                            height: edge as usize,
                            depth: 1,
                        },
                        &staging.buffer,
                        off,
                        edge as usize * bpp,
                        (edge * edge) as usize * bpp,
                    );
                }
                edge = (edge / 2).max(1);
            }
        }
        blit.endEncoding();
    });
    staging.read_bytes(total).to_vec()
}

fn readback_buffer_blob(ctx: &Arc<GpuContext>, buf: &GpuBuffer) -> Vec<u8> {
    if buf.size == 0 {
        return Vec::new();
    }
    let staging = ctx.create_readback_buffer("dimp buf readback", buf.size);
    ctx.one_shot(|cmd| {
        let blit = cmd.blitCommandEncoder().expect("dimp buf readback blit");
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &buf.buffer,
                0,
                &staging.buffer,
                0,
                buf.size as usize,
            );
        }
        blit.endEncoding();
    });
    staging.read_bytes(buf.size as usize).to_vec()
}

fn upload_image_blob(
    ctx: &Arc<GpuContext>,
    name: &str,
    format: MTLPixelFormat,
    res: u32,
    layers: u32,
    mips: u32,
    bpp: usize,
    blob: &[u8],
) -> GpuImage {
    let image = ctx.create_texture_2d_array(name, res, res, layers, mips, format, false);
    let mut staging = ctx.create_upload_buffer("dimp upload", blob.max_len_1());
    if !blob.is_empty() {
        staging.write_bytes(0, blob);
    }
    ctx.one_shot(|cmd| {
        let blit = cmd.blitCommandEncoder().expect("dimp upload blit");
        let mut off = 0usize;
        for layer in 0..layers {
            let mut edge = res;
            for mip in 0..mips {
                let len = (edge * edge) as usize * bpp;
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &staging.buffer,
                        off,
                        edge as usize * bpp,
                        len,
                        MTLSize {
                            width: edge as usize,
                            height: edge as usize,
                            depth: 1,
                        },
                        &image.texture,
                        layer as usize,
                        mip as usize,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
                }
                off += len;
                edge = (edge / 2).max(1);
            }
        }
        blit.endEncoding();
    });
    image
}

trait SliceHelper {
    fn max_len_1(&self) -> usize;
}
impl SliceHelper for [u8] {
    fn max_len_1(&self) -> usize {
        self.len().max(1)
    }
}

impl ImpostorAsset {
    pub fn write_baked(&self, path: impl AsRef<Path>) -> Result<(), ImpostorAssetError> {
        let mips = self.mip_levels();
        let mut blobs = Vec::with_capacity(BLOB_COUNT);
        let imgs = [
            &self.albedo,
            &self.normal_ao,
            &self.metallic_rough,
            &self.emissive,
            &self.depth,
        ];
        for (i, img) in imgs.iter().enumerate() {
            blobs.push(readback_image_blob(
                &self.ctx,
                img,
                self.resolution,
                VIEW_COUNT as u32,
                mips,
                ATLAS_BPP[i],
            ));
        }
        let mat_mips = 32 - self.material_resolution.max(1).leading_zeros();
        for map in &self.material_maps {
            blobs.push(readback_image_blob(
                &self.ctx,
                map,
                self.material_resolution,
                self.material_count.max(1),
                mat_mips,
                4,
            ));
        }
        blobs.push(readback_buffer_blob(&self.ctx, &self.voxels));
        blobs.push(readback_buffer_blob(&self.ctx, &self.material_table));

        let slab_mask: u32 = self
            .slab_valid
            .iter()
            .enumerate()
            .map(|(i, v)| (*v as u32) << i)
            .sum();
        let gpu_bytes: u64 = blobs.iter().map(|b| b.len() as u64).sum();

        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FILE_VERSION.to_le_bytes());
        header.extend_from_slice(&GEOMETRY_LAYOUT.to_le_bytes());
        for f in self
            .bounds
            .min
            .to_array()
            .into_iter()
            .chain(self.bounds.max.to_array())
        {
            header.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        for v in [
            self.resolution,
            self.voxel_resolution,
            self.material_resolution,
            self.material_count,
            self.triangle_base,
            self.triangle_count,
            self.surface_count,
            slab_mask,
            u32::from(self.has_mask_materials),
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

        let mut file = std::fs::File::create(path)?;
        file.write_all(&header)?;
        for b in &blobs {
            file.write_all(b)?;
        }
        Ok(())
    }

    pub fn load_baked(
        ctx: &Arc<GpuContext>,
        path: impl AsRef<Path>,
        limits: ImpostorAssetLimits,
    ) -> Result<Self, ImpostorAssetError> {
        let mut file = std::fs::File::open(path)?;
        let meta_len = file.metadata()?.len();
        if meta_len > limits.max_file_bytes {
            return Err(ImpostorAssetError::BudgetExceeded {
                requested: meta_len,
                limit: limits.max_file_bytes,
            });
        }
        if meta_len < HEADER_LEN as u64 {
            return Err(invalid("file shorter than 248-byte header"));
        }

        let mut header = [0u8; HEADER_LEN];
        file.read_exact(&mut header)?;
        if &header[0..8] != MAGIC {
            return Err(invalid("bad magic"));
        }
        let mut r = Reader {
            buf: &header,
            pos: 8,
        };
        let version = r.u32()?;
        let geom_layout = r.u32()?;
        if version != FILE_VERSION || geom_layout != GEOMETRY_LAYOUT {
            return Err(ImpostorAssetError::UnsupportedVersion);
        }
        let min = Vec3::new(r.f32()?, r.f32()?, r.f32()?);
        let max = Vec3::new(r.f32()?, r.f32()?, r.f32()?);
        if !min.is_finite()
            || !max.is_finite()
            || min.x >= max.x
            || min.y >= max.y
            || min.z >= max.z
        {
            return Err(invalid("invalid bounds"));
        }
        let resolution = r.u32()?;
        let voxel_resolution = r.u32()?;
        let material_resolution = r.u32()?;
        let material_count = r.u32()?;
        let triangle_base = r.u32()?;
        let triangle_count = r.u32()?;
        let surface_count = r.u32()?;
        let slab_mask = r.u32()?;
        let has_mask = r.u32()?;
        let view_count = r.u32()?;
        let axis_count = r.u32()?;
        let blob_count = r.u32()?;
        let gpu_bytes = r.u64()?;
        let sig = r.u64()?;

        if view_count != VIEW_COUNT as u32
            || axis_count != AXIS_COUNT as u32
            || blob_count != BLOB_COUNT as u32
        {
            return Err(invalid("view/axis/blob count mismatch"));
        }
        if sig != view_signature() {
            return Err(invalid("view signature mismatch"));
        }
        if !resolution.is_power_of_two() || !(16..=1024).contains(&resolution) {
            return Err(invalid("invalid resolution"));
        }
        if resolution > limits.max_texture_resolution {
            return Err(ImpostorAssetError::BudgetExceeded {
                requested: u64::from(resolution),
                limit: u64::from(limits.max_texture_resolution),
            });
        }
        if voxel_resolution != 0
            && (!voxel_resolution.is_power_of_two()
                || !(16..=256).contains(&voxel_resolution)
                || voxel_resolution > limits.max_voxel_resolution)
        {
            return Err(ImpostorAssetError::BudgetExceeded {
                requested: u64::from(voxel_resolution),
                limit: u64::from(limits.max_voxel_resolution),
            });
        }
        if !material_resolution.is_power_of_two()
            || material_resolution > limits.max_texture_resolution
        {
            return Err(ImpostorAssetError::BudgetExceeded {
                requested: u64::from(material_resolution),
                limit: u64::from(limits.max_texture_resolution),
            });
        }
        if material_count > limits.max_material_count {
            return Err(ImpostorAssetError::BudgetExceeded {
                requested: u64::from(material_count),
                limit: u64::from(limits.max_material_count),
            });
        }

        let mut blob_lens = [0u64; BLOB_COUNT];
        for l in &mut blob_lens {
            *l = r.u64()?;
        }
        let mut blob_crcs = [0u32; BLOB_COUNT];
        for c in &mut blob_crcs {
            *c = r.u32()?;
        }

        let sum_blobs: u64 = blob_lens.iter().sum();
        if sum_blobs != gpu_bytes || HEADER_LEN as u64 + sum_blobs != meta_len {
            return Err(invalid("blob length sum mismatch"));
        }
        if gpu_bytes > limits.max_gpu_bytes {
            return Err(ImpostorAssetError::BudgetExceeded {
                requested: gpu_bytes,
                limit: limits.max_gpu_bytes,
            });
        }

        let mips = 32 - resolution.leading_zeros();
        for i in 0..5 {
            if blob_lens[i] != atlas_blob_bytes(resolution, mips, ATLAS_BPP[i]) as u64 {
                return Err(invalid("atlas blob size mismatch"));
            }
        }
        let mat_layers = material_count.max(1) as u64;
        let mat_mips = 32 - material_resolution.max(1).leading_zeros();
        let mat_blob_bytes = atlas_blob_bytes(material_resolution, mat_mips, 4) as u64
            / VIEW_COUNT as u64
            * mat_layers;
        for i in 5..10 {
            if blob_lens[i] != mat_blob_bytes {
                return Err(invalid("material map blob size mismatch"));
            }
        }
        let sb_limit = u64::from(ctx.max_storage_buffer_range());
        if blob_lens[10] > sb_limit || blob_lens[11] > sb_limit {
            return Err(ImpostorAssetError::DeviceLimit(
                "voxel or material table exceeds maxStorageBufferRange".into(),
            ));
        }

        let mut blobs = Vec::with_capacity(BLOB_COUNT);
        for (i, &len) in blob_lens.iter().enumerate() {
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf)?;
            if crc32(&buf) != blob_crcs[i] {
                return Err(invalid(format!("blob {i} CRC mismatch")));
            }
            blobs.push(buf);
        }

        let albedo = upload_image_blob(
            ctx,
            "dimp albedo",
            ATLAS_FORMATS[0],
            resolution,
            VIEW_COUNT as u32,
            mips,
            ATLAS_BPP[0],
            &blobs[0],
        );
        let normal_ao = upload_image_blob(
            ctx,
            "dimp normal_ao",
            ATLAS_FORMATS[1],
            resolution,
            VIEW_COUNT as u32,
            mips,
            ATLAS_BPP[1],
            &blobs[1],
        );
        let metallic_rough = upload_image_blob(
            ctx,
            "dimp mr",
            ATLAS_FORMATS[2],
            resolution,
            VIEW_COUNT as u32,
            mips,
            ATLAS_BPP[2],
            &blobs[2],
        );
        let emissive = upload_image_blob(
            ctx,
            "dimp emissive",
            ATLAS_FORMATS[3],
            resolution,
            VIEW_COUNT as u32,
            mips,
            ATLAS_BPP[3],
            &blobs[3],
        );
        let depth = upload_image_blob(
            ctx,
            "dimp depth",
            ATLAS_FORMATS[4],
            resolution,
            VIEW_COUNT as u32,
            mips,
            ATLAS_BPP[4],
            &blobs[4],
        );

        let material_maps = std::array::from_fn(|i| {
            upload_image_blob(
                ctx,
                "dimp mat map",
                MAP_FORMATS[i],
                material_resolution,
                material_count.max(1),
                mat_mips,
                4,
                &blobs[5 + i],
            )
        });

        let voxels = if blobs[10].is_empty() {
            ctx.create_storage_buffer("dimp voxels", &[0u8; 4])
        } else {
            ctx.create_storage_buffer("dimp voxels", &blobs[10])
        };
        let material_table = if blobs[11].is_empty() {
            ctx.create_storage_buffer("dimp mat table", &[0u8; 4])
        } else {
            ctx.create_storage_buffer("dimp mat table", &blobs[11])
        };

        let _ = (TRIANGLE_WORDS, MATERIAL_WORDS);
        let mut slab_valid = [false; AXIS_COUNT];
        for (i, v) in slab_valid.iter_mut().enumerate() {
            *v = (slab_mask >> i) & 1 != 0;
        }

        Ok(ImpostorAsset {
            bounds: ImpostorBounds { min, max },
            resolution,
            slab_valid,
            voxel_resolution,
            triangle_base,
            has_mask_materials: has_mask != 0,
            triangle_count,
            surface_count,
            material_resolution,
            material_count,
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
