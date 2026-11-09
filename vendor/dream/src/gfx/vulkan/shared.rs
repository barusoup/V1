//! Texture objects shared by every texture of one [`GpuContext`]: samplers,
//! the two texture descriptor set layouts, and descriptor pool chunks that
//! sets are allocated from.
//!
//! Creating these per texture wasted driver memory and creation time, and
//! image-heavy apps hit `maxSamplerAllocationCount`. Samplers are keyed by
//! their description and mipmapped-ness; pools are chunks of
//! [`CHUNK_TEXTURES`] textures that grow on demand and are retired when empty.

use super::GpuContext;
use crate::texture::{
    SamplerDesc, create_sampled_texture_set_layout, create_sampler, create_texture_set_layout,
};
use ash::vk;
use std::collections::HashMap;

/// Textures one pool chunk can back: each texture takes two sets, so
/// `max_sets` is twice this, and each descriptor type is sized to match.
const CHUNK_TEXTURES: u32 = 128;

/// Per-context cache behind `GpuContext::shared_textures`.
#[derive(Default)]
pub(super) struct SharedTextureObjects {
    /// Keyed by `(description, mipmapped)`: the only sampler state that
    /// depends on the mip count is whether LOD clamping is unbounded.
    samplers: HashMap<(SamplerDesc, bool), vk::Sampler>,
    /// `[combined, sampled]`, created on first use.
    layouts: Option<[vk::DescriptorSetLayout; 2]>,
    chunks: Vec<PoolChunk>,
}

/// One descriptor pool and the number of textures whose sets it holds.
struct PoolChunk {
    pool: vk::DescriptorPool,
    live: u32,
}

/// Diagnostic counts of the shared texture objects of one [`GpuContext`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureObjectStats {
    /// Distinct `vk::Sampler`s alive in the shared cache.
    pub samplers: usize,
    /// Descriptor pool chunks alive.
    pub descriptor_pools: usize,
    /// Textures currently holding descriptor sets (one per texture).
    pub live_texture_sets: usize,
}

/// Cache key for a sampler: equal descriptions share one `vk::Sampler`, and
/// all mip counts above one behave identically.
fn sampler_key(desc: SamplerDesc, mip_levels: u32) -> (SamplerDesc, bool) {
    (desc, mip_levels > 1)
}

impl GpuContext {
    /// The shared `[combined, sampled]` texture set layouts, created once.
    /// Allocation takes the same lock and uses the lock-held helper directly,
    /// so this is the entry point for callers outside the texture module.
    #[allow(dead_code)]
    pub(crate) fn shared_texture_layouts(&self) -> [vk::DescriptorSetLayout; 2] {
        let mut shared = self.shared_textures.lock().unwrap();
        layouts_in(self, &mut shared)
    }

    /// A shared sampler for `desc`, created on the first request for each
    /// `(desc, mipmapped)` pair.
    pub(crate) fn shared_sampler(&self, desc: SamplerDesc, mip_levels: u32) -> vk::Sampler {
        let mut shared = self.shared_textures.lock().unwrap();
        sampler_in(self, &mut shared, desc, mip_levels)
    }

    /// Allocate the `[combined, sampled]` set pair for one texture from the
    /// first pool chunk with room, growing a new chunk when all are full.
    pub(crate) fn alloc_texture_sets(&self) -> (vk::DescriptorPool, [vk::DescriptorSet; 2]) {
        let mut shared = self.shared_textures.lock().unwrap();
        alloc_in(self, &mut shared)
    }

    /// Return a texture's sets to `pool`. A chunk that empties is destroyed
    /// unless it is the last one left.
    pub(crate) fn free_texture_sets(&self, pool: vk::DescriptorPool, sets: [vk::DescriptorSet; 2]) {
        let mut shared = self.shared_textures.lock().unwrap();
        unsafe { self.device.free_descriptor_sets(pool, &sets) }
            .expect("free texture descriptor sets");
        let index = shared
            .chunks
            .iter()
            .position(|chunk| chunk.pool == pool)
            .expect("texture descriptor pool chunk");
        let chunk = &mut shared.chunks[index];
        chunk.live -= 1;
        let empty = chunk.live == 0;
        if empty && shared.chunks.len() > 1 {
            let retired = shared.chunks.remove(index);
            unsafe { self.device.destroy_descriptor_pool(retired.pool, None) };
        }
    }

    /// Counts of the shared texture objects, for diagnostics and tests.
    pub fn texture_object_stats(&self) -> TextureObjectStats {
        let shared = self.shared_textures.lock().unwrap();
        TextureObjectStats {
            samplers: shared.samplers.len(),
            descriptor_pools: shared.chunks.len(),
            live_texture_sets: shared.chunks.iter().map(|c| c.live as usize).sum(),
        }
    }

    /// Destroy every shared pool, sampler and layout. Called from `Drop` after
    /// the device is idle, while the device is still alive.
    pub(super) fn destroy_shared_texture_objects(&mut self) {
        let device = &self.device;
        let shared = self.shared_textures.get_mut().expect("shared texture lock");
        unsafe {
            for chunk in shared.chunks.drain(..) {
                device.destroy_descriptor_pool(chunk.pool, None);
            }
            for (_, sampler) in shared.samplers.drain() {
                device.destroy_sampler(sampler, None);
            }
            if let Some(layouts) = shared.layouts.take() {
                for layout in layouts {
                    device.destroy_descriptor_set_layout(layout, None);
                }
            }
        }
    }
}

fn layouts_in(ctx: &GpuContext, shared: &mut SharedTextureObjects) -> [vk::DescriptorSetLayout; 2] {
    *shared.layouts.get_or_insert_with(|| {
        [
            create_texture_set_layout(ctx),
            create_sampled_texture_set_layout(ctx),
        ]
    })
}

fn sampler_in(
    ctx: &GpuContext,
    shared: &mut SharedTextureObjects,
    desc: SamplerDesc,
    mip_levels: u32,
) -> vk::Sampler {
    *shared
        .samplers
        .entry(sampler_key(desc, mip_levels))
        .or_insert_with(|| create_sampler(ctx, "shared texture sampler", desc, mip_levels))
}

fn alloc_in(
    ctx: &GpuContext,
    shared: &mut SharedTextureObjects,
) -> (vk::DescriptorPool, [vk::DescriptorSet; 2]) {
    let layouts = layouts_in(ctx, shared);
    for index in 0..shared.chunks.len() {
        let pool = shared.chunks[index].pool;
        match allocate_pair(ctx, pool, &layouts) {
            Ok(sets) => {
                shared.chunks[index].live += 1;
                return (pool, sets);
            }
            Err(err) if super::pool_exhaustion_is_growable(err) => continue,
            Err(err) => panic!("texture descriptor sets: {err:?}"),
        }
    }
    let pool = create_chunk(ctx);
    let sets = allocate_pair(ctx, pool, &layouts)
        .unwrap_or_else(|err| panic!("texture descriptor sets: {err:?}"));
    shared.chunks.push(PoolChunk { pool, live: 1 });
    (pool, sets)
}

fn allocate_pair(
    ctx: &GpuContext,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout; 2],
) -> Result<[vk::DescriptorSet; 2], vk::Result> {
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    let sets = unsafe { ctx.device().allocate_descriptor_sets(&info) }?;
    match sets.as_slice() {
        &[set, sampled_set] => Ok([set, sampled_set]),
        _ => panic!("texture descriptor allocation returned wrong count"),
    }
}

fn create_chunk(ctx: &GpuContext) -> vk::DescriptorPool {
    // One pool backs both sets of every texture: the legacy combined contract
    // and the engine-WGSL sampled contract.
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(CHUNK_TEXTURES),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(CHUNK_TEXTURES),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(CHUNK_TEXTURES),
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(2 * CHUNK_TEXTURES)
        .pool_sizes(&pool_sizes);
    unsafe { ctx.device().create_descriptor_pool(&info, None) }
        .expect("texture descriptor pool chunk")
}

#[cfg(test)]
mod tests {
    use super::sampler_key;
    use crate::texture::SamplerDesc;

    #[test]
    fn texture_sampler_key_splits_single_and_mipmapped() {
        let desc = SamplerDesc::default();
        assert_ne!(sampler_key(desc, 1), sampler_key(desc, 5));
        assert_eq!(sampler_key(desc, 5), sampler_key(desc, 9));
    }
}
