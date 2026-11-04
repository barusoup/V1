use super::ImpostorBaker;
use crate::gfx::GpuBuffer;
use crate::gfx::GpuImage;
use crate::impostor::VIEW_COUNT;
use ash::vk;

impl ImpostorBaker {
    /// GPU post-bake: downsample -> dilate -> mipgen -> pack -> copy to the
    /// five final array images. One submission; returns the images, the
    /// level-0 RGBA tiles (for previews/test data), and per-layer coverage.
    pub(super) fn gpu_post(
        &self,
        caps: &[GpuImage],
        res: u32,
        ss: u32,
    ) -> (
        Vec<GpuImage>,
        Vec<Vec<Vec<u8>>>,
        Vec<Vec<u8>>,
        [f32; VIEW_COUNT],
    ) {
        let ctx = &self.ctx;
        let device = ctx.device().clone();
        let mips = 32 - res.leading_zeros();
        // Level texel offsets (layer-major per mip).
        let level_off: Vec<u64> = (0..mips)
            .scan(0u64, |acc, l| {
                let r = (res >> l).max(1);
                let out = *acc;
                *acc += VIEW_COUNT as u64 * (r as u64) * (r as u64);
                Some(out)
            })
            .collect();
        let last_res = (res >> (mips - 1)).max(1) as u64;
        let chain_texels = level_off.last().unwrap() + VIEW_COUNT as u64 * last_res * last_res;
        let l0_texels = VIEW_COUNT as u64 * (res as u64) * (res as u64);

        // Work buffers (device-local).
        let chain: Vec<GpuBuffer> = (0..4)
            .map(|i| {
                ctx.create_gpu_buffer(
                    &format!("post chain{i}"),
                    chain_texels * 16,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            })
            .collect();
        let scratch: Vec<GpuBuffer> = (0..4)
            .map(|i| {
                ctx.create_gpu_buffer(
                    &format!("post scratch{i}"),
                    l0_texels * 16,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            })
            .collect();
        let dep_chain = ctx.create_gpu_buffer(
            "post dep",
            chain_texels * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        );
        let packs: Vec<GpuBuffer> = (0..4)
            .map(|i| {
                ctx.create_gpu_buffer(
                    &format!("post pack{i}"),
                    chain_texels * 4,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                )
            })
            .collect();
        let pack_dep = ctx.create_gpu_buffer(
            "post pdep",
            chain_texels * 2,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        let cov_buf = ctx.create_gpu_buffer(
            "post cov",
            (VIEW_COUNT * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let cov_rb = ctx.create_readback_buffer("post cov rb", (VIEW_COUNT * 4) as u64);
        let tile_rb =
            ctx.create_cached_readback_buffer("post tiles rb", 4 * l0_texels * 4 + l0_texels * 2);

        // Final array images.
        let formats = [
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::R16_UNORM,
        ];
        let images: Vec<GpuImage> = formats
            .iter()
            .enumerate()
            .map(|(i, f)| {
                ctx.create_image_array_mips(
                    &format!("imp {i}"),
                    res,
                    res,
                    *f,
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    VIEW_COUNT as u32,
                    mips,
                )
            })
            .collect();

        // Descriptor pool + sets.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(5),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(64),
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(5)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .expect("post pool");
        let alloc = |dsl: vk::DescriptorSetLayout| {
            unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(std::slice::from_ref(&dsl)),
                )
            }
            .expect("post set")[0]
        };
        let set_down = alloc(self.post.down_dsl);
        let set_dilate_a = alloc(self.post.dilate_dsl);
        let set_dilate_b = alloc(self.post.dilate_dsl);
        let set_mip = alloc(self.post.mip_dsl);
        let set_pack = alloc(self.post.pack_dsl);

        let buf = |b: &GpuBuffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(b.buffer)
                .range(vk::WHOLE_SIZE)
        };
        let img = |v: vk::ImageView| {
            vk::DescriptorImageInfo::default()
                .image_view(v)
                .image_layout(vk::ImageLayout::GENERAL)
        };
        // down: 1-5 textures, 6-9 chain, 10 dep, 11 cov
        {
            let img_infos: Vec<_> = caps.iter().map(|c| img(c.view)).collect();
            let infos: Vec<_> = chain
                .iter()
                .chain(std::iter::once(&dep_chain))
                .chain(std::iter::once(&cov_buf))
                .map(&buf)
                .collect();
            let mut writes = Vec::new();
            for (i, info) in img_infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set_down)
                        .dst_binding(i as u32 + 1)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(std::slice::from_ref(info)),
                );
            }
            for (i, info) in infos.iter().enumerate() {
                writes.push(ssbo_write(set_down, i as u32 + 6, info));
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        // dilate A: src chain -> dst scratch; B: reverse.
        for (set, srcs, dsts) in [
            (set_dilate_a, &chain, &scratch),
            (set_dilate_b, &scratch, &chain),
        ] {
            let mut writes = Vec::new();
            let src_infos: Vec<_> = srcs.iter().map(&buf).collect();
            let dst_infos: Vec<_> = dsts.iter().map(&buf).collect();
            for (i, info) in src_infos.iter().enumerate() {
                writes.push(ssbo_write(set, i as u32 + 1, info));
            }
            for (i, info) in dst_infos.iter().enumerate() {
                writes.push(ssbo_write(set, i as u32 + 5, info));
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        // mip: 1-4 chain read, 5 dep read; 6-9 chain write, 10 dep write.
        {
            let mut writes = Vec::new();
            let c_infos: Vec<_> = chain.iter().map(&buf).collect();
            let d_info = buf(&dep_chain);
            for (i, info) in c_infos.iter().enumerate() {
                writes.push(ssbo_write(set_mip, i as u32 + 1, info));
            }
            writes.push(ssbo_write(set_mip, 5, &d_info));
            for (i, info) in c_infos.iter().enumerate() {
                writes.push(ssbo_write(set_mip, i as u32 + 6, info));
            }
            writes.push(ssbo_write(set_mip, 10, &d_info));
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        // pack: 1-4 chain, 5 dep; 6-9 pack, 10 pack_dep.
        {
            let mut writes = Vec::new();
            let c_infos: Vec<_> = chain.iter().map(&buf).collect();
            let d_info = buf(&dep_chain);
            let p_infos: Vec<_> = packs.iter().map(&buf).collect();
            let pd_info = buf(&pack_dep);
            for (i, info) in c_infos.iter().enumerate() {
                writes.push(ssbo_write(set_pack, i as u32 + 1, info));
            }
            writes.push(ssbo_write(set_pack, 5, &d_info));
            for (i, info) in p_infos.iter().enumerate() {
                writes.push(ssbo_write(set_pack, i as u32 + 6, info));
            }
            writes.push(ssbo_write(set_pack, 10, &pd_info));
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        let compute_barrier = |cmd: vk::CommandBuffer| unsafe {
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            );
        };
        let push = |v: [u32; 4]| bytemuck::bytes_of(&v).to_vec();

        let t_submit = std::time::Instant::now();
        let dbg = std::env::var_os("DREAM_BAKE_DEBUG").is_some();
        let t_mark = std::cell::Cell::new(t_submit);
        let mark = |label: &str| {
            if dbg {
                let now = std::time::Instant::now();
                eprintln!(
                    "[bake] {label}: {:.1}ms",
                    (now - t_mark.get()).as_secs_f32() * 1000.0
                );
                t_mark.set(now);
            }
        };
        ctx.one_shot(|cmd| unsafe {
            device.cmd_fill_buffer(cmd, cov_buf.buffer, 0, (VIEW_COUNT * 4) as u64, 0);
            let flush = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[flush],
                &[],
                &[],
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.post.down);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.down_layout,
                0,
                &[set_down],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.post.down_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push([res, ss, 0, 0]),
            );
            device.cmd_dispatch(cmd, res.div_ceil(8), res.div_ceil(8), VIEW_COUNT as u32);
            compute_barrier(cmd);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.post.dilate);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.dilate_layout,
                0,
                &[set_dilate_a],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.post.dilate_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push([res, 0, 0, 0]),
            );
            device.cmd_dispatch(cmd, res.div_ceil(8), res.div_ceil(8), VIEW_COUNT as u32);
            compute_barrier(cmd);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.dilate_layout,
                0,
                &[set_dilate_b],
                &[],
            );
            device.cmd_dispatch(cmd, res.div_ceil(8), res.div_ceil(8), VIEW_COUNT as u32);
            compute_barrier(cmd);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.dilate_layout,
                0,
                &[set_dilate_a],
                &[],
            );
            device.cmd_dispatch(cmd, res.div_ceil(8), res.div_ceil(8), VIEW_COUNT as u32);
            compute_barrier(cmd);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.dilate_layout,
                0,
                &[set_dilate_b],
                &[],
            );
            device.cmd_dispatch(cmd, res.div_ceil(8), res.div_ceil(8), VIEW_COUNT as u32);
            compute_barrier(cmd);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.dilate_layout,
                0,
                &[set_dilate_a],
                &[],
            );
            device.cmd_dispatch(cmd, res.div_ceil(8), res.div_ceil(8), VIEW_COUNT as u32);
            compute_barrier(cmd);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.dilate_layout,
                0,
                &[set_dilate_b],
                &[],
            );
            device.cmd_dispatch(cmd, res.div_ceil(8), res.div_ceil(8), VIEW_COUNT as u32);
            compute_barrier(cmd);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.post.mip);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.mip_layout,
                0,
                &[set_mip],
                &[],
            );
            for l in 1..mips {
                let dr = (res >> l).max(1);
                let sr = (res >> (l - 1)).max(1);
                device.cmd_push_constants(
                    cmd,
                    self.post.mip_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push([
                        dr,
                        sr,
                        level_off[l as usize - 1] as u32,
                        level_off[l as usize] as u32,
                    ]),
                );
                device.cmd_dispatch(cmd, dr.div_ceil(8), dr.div_ceil(8), VIEW_COUNT as u32);
                compute_barrier(cmd);
            }

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.post.pack);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post.pack_layout,
                0,
                &[set_pack],
                &[],
            );
            for l in 0..mips {
                let r = (res >> l).max(1);
                device.cmd_push_constants(
                    cmd,
                    self.post.pack_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push([
                        r,
                        l,
                        level_off[l as usize] as u32,
                        level_off[l as usize] as u32,
                    ]),
                );
                device.cmd_dispatch(cmd, r.div_ceil(8), r.div_ceil(8), VIEW_COUNT as u32);
            }
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            );
        });
        mark("compute dispatches");
        ctx.one_shot(|cmd| unsafe {
            // pack buffers -> final images (level-major regions).
            for (i, image) in images.iter().enumerate() {
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
                let (src, bpt) = if i == 4 {
                    (&pack_dep, 2u64)
                } else {
                    (&packs[i], 4u64)
                };
                let mut regions = Vec::with_capacity(VIEW_COUNT * mips as usize);
                for l in 0..mips as usize {
                    let r = (res >> l).max(1);
                    for f in 0..VIEW_COUNT {
                        let off = (level_off[l] + f as u64 * r as u64 * r as u64) * bpt;
                        regions.push(
                            vk::BufferImageCopy::default()
                                .buffer_offset(off)
                                .image_subresource(vk::ImageSubresourceLayers {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    mip_level: l as u32,
                                    base_array_layer: f as u32,
                                    layer_count: 1,
                                })
                                .image_extent(vk::Extent3D {
                                    width: r,
                                    height: r,
                                    depth: 1,
                                }),
                        );
                    }
                }
                device.cmd_copy_buffer_to_image(
                    cmd,
                    src.buffer,
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
            }
        });
        mark("image copies");
        ctx.one_shot(|cmd| unsafe {
            // coverage + level-0 tiles readback
            device.cmd_copy_buffer(
                cmd,
                cov_buf.buffer,
                cov_rb.buffer,
                &[vk::BufferCopy::default().size((VIEW_COUNT * 4) as u64)],
            );
            let mut off = 0u64;
            for src in packs.iter().chain(std::iter::once(&pack_dep)) {
                let bytes = if src.buffer == pack_dep.buffer {
                    l0_texels * 2
                } else {
                    l0_texels * 4
                };
                device.cmd_copy_buffer(
                    cmd,
                    src.buffer,
                    tile_rb.buffer,
                    &[vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(off)
                        .size(bytes)],
                );
                off += bytes;
            }
        });
        unsafe { device.destroy_descriptor_pool(pool, None) };

        // CPU-side results.
        let cov_raw: Vec<u32> = cov_rb
            .mapped()
            .map(|m| {
                m[..VIEW_COUNT * 4]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .unwrap_or_default();
        let cov: [f32; VIEW_COUNT] = std::array::from_fn(|i| {
            cov_raw.get(i).copied().unwrap_or(0) as f32 / (res * res * ss * ss) as f32
        });
        let rb = tile_rb.mapped().unwrap_or(&[]).to_vec();
        if std::env::var_os("DREAM_BAKE_DEBUG").is_some() {
            eprintln!(
                "[bake] post submit+readback {:.1}ms",
                t_submit.elapsed().as_secs_f32() * 1000.0
            );
        }
        let mut tiles: Vec<Vec<Vec<u8>>> = vec![Vec::new(); 5];
        let mut off = 0usize;
        for i in 0..4 {
            let bytes = l0_texels as usize * 4;
            for f in 0..VIEW_COUNT {
                tiles[i].push(
                    rb[off + f as usize * res as usize * res as usize * 4
                        ..off + (f + 1) as usize * res as usize * res as usize * 4]
                        .to_vec(),
                );
            }
            off += bytes;
        }
        let mut dep_tiles = Vec::new();
        for f in 0..VIEW_COUNT {
            dep_tiles.push(
                rb[off + f * res as usize * res as usize * 2
                    ..off + (f + 1) * res as usize * res as usize * 2]
                    .to_vec(),
            );
        }
        (images, tiles, dep_tiles, cov)
    }
}

/// One descriptor write for a storage buffer.
pub(super) fn ssbo_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}
