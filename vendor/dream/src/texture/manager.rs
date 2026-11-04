#[cfg(doc)]
use super::SamplerDesc;

use super::TextureOptions;
use super::decode::{DecodedTexture, decode_rgba8};
use super::gpu::GpuTexture;
use crate::gfx::vulkan::GpuContext;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// --- Management ------------------------------------------------------------

/// Opaque handle to a texture inside a [`TextureManager`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(u32);

/// Manager statistics, for devtools and leak checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureStats {
    pub textures: usize,
    /// Approximate GPU payload of managed textures in bytes.
    pub gpu_bytes: u64,
    /// Textures loaded more than once that shared one GPU image.
    pub cache_hits: usize,
}

/// Loads, caches and owns textures. File loads are deduplicated by canonical
/// path, so two systems asking for `stone.png` share one GPU image.
///
/// The manager owns sampler caching through [`SamplerDesc`] equality; textures
/// are handed out as `Arc<GpuTexture>` and stay alive while referenced.
pub struct TextureManager {
    ctx: Arc<GpuContext>,
    by_id: Vec<Option<Arc<GpuTexture>>>,
    by_path: HashMap<PathBuf, TextureId>,
    free: Vec<TextureId>,
    cache_hits: usize,
    default_options: TextureOptions,
}

impl TextureManager {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        Self {
            ctx,
            by_id: Vec::new(),
            by_path: HashMap::new(),
            free: Vec::new(),
            cache_hits: 0,
            default_options: TextureOptions::default(),
        }
    }

    /// Options used by [`Self::load`] when no explicit options are given.
    pub fn set_default_options(&mut self, options: TextureOptions) {
        self.default_options = options;
    }

    /// Load a PNG or JPEG from disk with the default options. Repeat loads of
    /// the same path return the same texture.
    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<Arc<GpuTexture>, String> {
        self.load_with(path, self.default_options)
    }

    /// Load a PNG or JPEG from disk with explicit options. Options apply on
    /// first load; later loads of the same path return the cached texture.
    pub fn load_with(
        &mut self,
        path: impl AsRef<Path>,
        options: TextureOptions,
    ) -> Result<Arc<GpuTexture>, String> {
        let path = canonical(path.as_ref());
        if let Some(&id) = self.by_path.get(&path) {
            self.cache_hits += 1;
            return Ok(self.get(id).expect("cached texture present").clone());
        }
        let bytes = std::fs::read(&path).map_err(|err| format!("{}: {err}", path.display()))?;
        let decoded = decode_rgba8(&bytes)?;
        let texture = self.create_decoded(&path.display().to_string(), &decoded, options);
        let id = self.insert(texture.clone());
        self.by_path.insert(path, id);
        Ok(texture)
    }

    /// Upload CPU pixels; used for generated art (UI atlases, gradients) and
    /// by [`Self::load_with`] after decoding.
    pub fn from_pixels(
        &mut self,
        label: &str,
        pixels: &[u8],
        width: u32,
        height: u32,
        options: TextureOptions,
    ) -> (TextureId, Arc<GpuTexture>) {
        let texture = self.create_decoded(
            label,
            &DecodedTexture {
                pixels: pixels.to_vec(),
                width,
                height,
            },
            options,
        );
        let id = self.insert(texture.clone());
        (id, texture)
    }

    /// Decode and upload image bytes (PNG/JPEG) in one step.
    pub fn from_bytes(
        &mut self,
        label: &str,
        bytes: &[u8],
        options: TextureOptions,
    ) -> Result<(TextureId, Arc<GpuTexture>), String> {
        let decoded = decode_rgba8(bytes)?;
        Ok(self.from_pixels(
            label,
            &decoded.pixels,
            decoded.width,
            decoded.height,
            options,
        ))
    }

    /// Insert an externally built texture (e.g. a UI font atlas) and get a
    /// handle for it. The manager does not deduplicate these.
    pub fn insert(&mut self, texture: Arc<GpuTexture>) -> TextureId {
        if let Some(id) = self.free.pop() {
            self.by_id[id.0 as usize] = Some(texture);
            return id;
        }
        let id = TextureId(self.by_id.len() as u32);
        self.by_id.push(Some(texture));
        id
    }

    pub fn get(&self, id: TextureId) -> Option<&Arc<GpuTexture>> {
        self.by_id.get(id.0 as usize)?.as_ref()
    }

    /// Look up a previously loaded file texture.
    pub fn get_path(&self, path: impl AsRef<Path>) -> Option<&Arc<GpuTexture>> {
        let id = *self.by_path.get(&canonical(path.as_ref()))?;
        self.get(id)
    }

    /// Drop the manager's reference. Textures still held elsewhere stay alive.
    pub fn remove(&mut self, id: TextureId) {
        if self
            .by_id
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .is_some()
        {
            self.by_id[id.0 as usize] = None;
            self.free.push(id);
            self.by_path.retain(|_, &mut cached| cached != id);
        }
    }

    pub fn stats(&self) -> TextureStats {
        let mut gpu_bytes = 0;
        let mut textures = 0;
        for texture in self.by_id.iter().flatten() {
            textures += 1;
            gpu_bytes += texture.estimated_bytes();
        }
        TextureStats {
            textures,
            gpu_bytes,
            cache_hits: self.cache_hits,
        }
    }

    /// The context every managed texture was created on.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    fn create_decoded(
        &self,
        label: &str,
        decoded: &DecodedTexture,
        options: TextureOptions,
    ) -> Arc<GpuTexture> {
        Arc::new(GpuTexture::from_rgba8(
            &self.ctx,
            label,
            &decoded.pixels,
            decoded.width,
            decoded.height,
            options,
        ))
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}
