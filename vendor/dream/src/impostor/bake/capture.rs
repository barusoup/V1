use super::materials::material_gpu_data;
use super::previews::build_previews;
use super::{
    BakeBatch, BakeMaterial, BakeSettings, CAPTURE_FORMATS, DEPTH_FORMAT, ImpostorBaker, PostPipes,
    PreparedBake,
};
#[cfg(test)]
use super::{BakeTestLayers, BakeTestStats, CapturedView, f16_to_f32};
#[cfg(test)]
use crate::gfx::GpuBuffer;
use crate::gfx::GpuImage;
use crate::gfx::vulkan;
use crate::gfx::vulkan::GpuContext;
use crate::impostor::atlas::ImpostorAsset;
use crate::impostor::atlas::ImpostorAssetLimits;
use crate::impostor::{AXIS_COUNT, FallbackReason, VIEW_COUNT, VIEWS, projected_extent};
use crate::model::render::PbrMaterialGpu;
use crate::model::render::PbrVertex;
use crate::model::{AlphaMode, Material};
use ash::vk;
use glam::{Mat4, Vec3};
use std::sync::Arc;

impl ImpostorBaker {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        // set 0 mirrors model_pbr.wgsl's material layout exactly.
        let material_dsl = pbr_material_dsl(ctx);
        let base_dsl = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            unsafe {
                ctx.device().create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .expect("capture base set layout")
        };
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(128);
        let layout = unsafe {
            ctx.device().create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[material_dsl, base_dsl])
                    .push_constant_ranges(std::slice::from_ref(&push)),
                None,
            )
        }
        .expect("capture pipeline layout");
        let shader = ctx.compile_shader(include_str!("../capture.wgsl"), "impostor_capture");
        let pipeline = create_capture_pipeline(ctx, shader, layout);
        unsafe { ctx.device().destroy_shader_module(shader, None) };

        let module = ctx.compile_shader(include_str!("../bake_post.wgsl"), "impostor_post");
        let dt = vk::DescriptorType::SAMPLED_IMAGE;
        let st = vk::DescriptorType::STORAGE_BUFFER;
        let (down, down_layout, down_dsl) = cs_pipeline(
            ctx,
            module,
            "cs_down",
            &[dt, dt, dt, dt, dt, st, st, st, st, st, st],
            "down",
        );
        let (dilate, dilate_layout, dilate_dsl) =
            cs_pipeline(ctx, module, "cs_dilate", &[st; 8], "dilate");
        let (mip, mip_layout, mip_dsl) = cs_pipeline(ctx, module, "cs_mip", &[st; 10], "mip");
        let (pack, pack_layout, pack_dsl) = cs_pipeline(ctx, module, "cs_pack", &[st; 10], "pack");
        unsafe { ctx.device().destroy_shader_module(module, None) };
        let post = PostPipes {
            down,
            down_layout,
            down_dsl,
            dilate,
            dilate_layout,
            dilate_dsl,
            mip,
            mip_layout,
            mip_dsl,
            pack,
            pack_layout,
            pack_dsl,
        };
        Self {
            ctx: Arc::clone(ctx),
            pipeline,
            layout,
            material_dsl,
            base_dsl,
            post,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn bake_prepared_with(
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

        // Meshes are already grouped by material and transformed into shared
        // object space (world space == object space for the bake). Budget
        // checks already ran; GPU batch creation follows them now.
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

        // All six ortho captures into 6-layer array images (one submission),
        // then the GPU post chain (downsample/dilate/mipgen/pack -> copies).
        let mut mvps = [Mat4::IDENTITY; VIEW_COUNT];
        let mut face_ns = [Vec3::ZERO; VIEW_COUNT];
        let mut exts = [[0.0f32; 3]; VIEW_COUNT];
        for (f, view) in VIEWS.iter().enumerate() {
            let ext_u = projected_extent(extent, view.u);
            let ext_v = projected_extent(extent, view.v);
            let ext_n = projected_extent(extent, view.dir);
            let eye = center + view.dir * (ext_n * 0.5 + 1.0);
            let view_mat = Mat4::look_to_rh(eye, -view.dir, view.v);
            // Vulkan clip z is 0..1 — orthographic_rh, not the GL variant.
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
        let cap_start = std::time::Instant::now();
        let (caps, _capture_depth) =
            self.capture_views_gpu(&baked, cap, &mvps, &face_ns, center, &exts);
        let capture_ms = cap_start.elapsed().as_secs_f32() * 1000.0;
        let post_start = std::time::Instant::now();
        let (images, tiles, dep_tiles, coverage_frac) = self.gpu_post(&caps, res, ss);
        let post_ms = post_start.elapsed().as_secs_f32() * 1000.0;
        let voxel_bytes = voxel_cells.len() * std::mem::size_of::<u32>();
        let voxels = self.ctx.upload_buffer(
            "impostor surface voxels",
            bytemuck::cast_slice(&voxel_cells),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        if std::env::var_os("DREAM_BAKE_DEBUG").is_some() {
            eprintln!(
                "[bake] res={res} ss={ss} extent=({:.4},{:.4},{:.4}) surfaces={surface_count} triangles={triangle_count} maxrefs={max_cell_refs} capture={capture_ms:.1}ms post={post_ms:.1}ms (wall)",
                extent.x, extent.y, extent.z,
            );
        }
        let mips = 32 - res.leading_zeros();

        let mut slab_valid = [true; AXIS_COUNT];
        let mut any_valid = false;
        for a in 0..AXIS_COUNT {
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

        // preview inputs: level-0 packed tiles (albedo/nrm/mr/emi rgba8, dep u16)
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
            triangle_count,
            has_mask_materials: has_mask,
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

    /// One material's texture set + base-colour uniform, for capture draws.
    pub(super) fn bake_material(&self, material: &Material) -> BakeMaterial {
        let pbr = PbrMaterialGpu::new(&self.ctx, self.material_dsl, material);
        let mut base = self.ctx.create_host_buffer("bake base", 16);
        base.write(0, bytemuck::bytes_of(&material.base_color));
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1);
        let pool = unsafe {
            self.ctx.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(std::slice::from_ref(&pool_size)),
                None,
            )
        }
        .expect("bake base pool");
        let set = unsafe {
            self.ctx.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&self.base_dsl)),
            )
        }
        .expect("bake base set")[0];
        let info = vk::DescriptorBufferInfo::default()
            .buffer(base.buffer)
            .offset(0)
            .range(16);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&info));
        unsafe { self.ctx.device().update_descriptor_sets(&[write], &[]) };
        BakeMaterial {
            ctx: Arc::clone(&self.ctx),
            pbr,
            _base: base,
            pool,
            base_set: set,
        }
    }

    /// Render all six views into 6-layer array images in one submission.
    /// Returns (channel images, per-layer attachment views, depth images);
    pub(super) fn capture_views_gpu(
        &self,
        baked: &[BakeBatch],
        size: u32,
        mvps: &[Mat4; VIEW_COUNT],
        face_ns: &[Vec3; VIEW_COUNT],
        center: Vec3,
        extents: &[[f32; 3]; VIEW_COUNT],
    ) -> (Vec<GpuImage>, Vec<GpuImage>) {
        let ctx = &self.ctx;
        let device = ctx.device().clone();
        let mut images = Vec::with_capacity(5);
        for (i, format) in CAPTURE_FORMATS.iter().enumerate() {
            images.push(ctx.create_image_array_mips(
                &format!("bake ch{i}"),
                size,
                size,
                *format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                VIEW_COUNT as u32,
                1,
            ));
        }
        // Per-layer 2D attachment views.
        let layer_views: Vec<Vec<vk::ImageView>> = images
            .iter()
            .enumerate()
            .map(|(i, img)| {
                (0..VIEW_COUNT as u32)
                    .map(|f| unsafe {
                        device
                            .create_image_view(
                                &vk::ImageViewCreateInfo::default()
                                    .image(img.image)
                                    .view_type(vk::ImageViewType::TYPE_2D)
                                    .format(CAPTURE_FORMATS[i])
                                    .subresource_range(vk::ImageSubresourceRange {
                                        aspect_mask: vk::ImageAspectFlags::COLOR,
                                        base_mip_level: 0,
                                        level_count: 1,
                                        base_array_layer: f,
                                        layer_count: 1,
                                    }),
                                None,
                            )
                            .expect("layer view")
                    })
                    .collect()
            })
            .collect();
        let depth: Vec<GpuImage> = (0..VIEW_COUNT)
            .map(|_| {
                ctx.create_image(
                    "bake z",
                    size,
                    size,
                    DEPTH_FORMAT,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                )
            })
            .collect();

        ctx.one_shot(|cmd| unsafe {
            for image in images.iter().map(|i| i.image) {
                vulkan::image_barrier(
                    &device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::ImageAspectFlags::COLOR,
                );
            }
            for d in &depth {
                vulkan::image_barrier(
                    &device,
                    cmd,
                    d.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::ImageAspectFlags::DEPTH,
                );
            }
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            };
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: size as f32,
                height: size as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            for f in 0..VIEW_COUNT {
                let mvp = mvps[f];
                let push = {
                    let mut bytes = [0u8; 128];
                    bytes[..64].copy_from_slice(bytemuck::bytes_of(&mvp.to_cols_array_2d()));
                    bytes[64..80]
                        .copy_from_slice(bytemuck::bytes_of(&face_ns[f].extend(0.0).to_array()));
                    bytes[80..96]
                        .copy_from_slice(bytemuck::bytes_of(&center.extend(0.0).to_array()));
                    bytes[96..112].copy_from_slice(bytemuck::bytes_of(&[
                        extents[f][0],
                        extents[f][1],
                        extents[f][2],
                        0.0,
                    ]));
                    bytes
                };
                let colors: Vec<_> = (0..5)
                    .map(|i| {
                        vk::RenderingAttachmentInfo::default()
                            .image_view(layer_views[i][f])
                            .image_layout(vk::ImageLayout::GENERAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 0.0],
                                },
                            })
                    })
                    .collect();
                let depth_att = vk::RenderingAttachmentInfo::default()
                    .image_view(depth[f].view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    });
                let info = vk::RenderingInfo::default()
                    .render_area(scissor)
                    .layer_count(1)
                    .color_attachments(&colors)
                    .depth_attachment(&depth_att);
                device.cmd_begin_rendering(cmd, &info);
                device.cmd_set_viewport(cmd, 0, &[viewport]);
                device.cmd_set_scissor(cmd, 0, &[scissor]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                for batch in baked {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.layout,
                        0,
                        &[batch.material.pbr.set, batch.material.base_set],
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        self.layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &push,
                    );
                    device.cmd_bind_vertex_buffers(cmd, 0, &[batch.vertex_buffer.buffer], &[0]);
                    device.cmd_bind_index_buffer(
                        cmd,
                        batch.index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(cmd, batch.index_count, 1, 0, 0, 0);
                }
                device.cmd_end_rendering(cmd);
            }
        });
        for set in &layer_views {
            for &v in set {
                unsafe { device.destroy_image_view(v, None) };
            }
        }
        (images, depth)
    }

    /// Render + read back one view's five attachments.
    #[cfg(test)]
    pub(super) fn capture_view(
        &self,
        baked: &[BakeBatch],
        size: u32,
        mvp: Mat4,
        face_n: Vec3,
        center: Vec3,
        extents: [f32; 3],
    ) -> Result<CapturedView, FallbackReason> {
        let ctx = &self.ctx;
        let mut images = Vec::with_capacity(5);
        for (i, format) in CAPTURE_FORMATS.iter().enumerate() {
            images.push(ctx.create_image(
                &format!("bake ch{i}"),
                size,
                size,
                *format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            ));
        }
        let depth = ctx.create_image(
            "bake z",
            size,
            size,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );
        // Readback buffers, one per attachment.
        let bpp = [8u64, 8, 8, 8, 2];
        let buffers: Vec<GpuBuffer> = (0..5)
            .map(|i| {
                ctx.create_readback_buffer(
                    &format!("bake rb{i}"),
                    size as u64 * size as u64 * bpp[i],
                )
            })
            .collect();

        let device = ctx.device().clone();
        let pipeline = self.pipeline;
        let layout = self.layout;
        let push = {
            let mut bytes = [0u8; 128];
            bytes[..64].copy_from_slice(bytemuck::bytes_of(&mvp.to_cols_array_2d()));
            bytes[64..80].copy_from_slice(bytemuck::bytes_of(&face_n.extend(0.0).to_array()));
            bytes[80..96].copy_from_slice(bytemuck::bytes_of(&center.extend(0.0).to_array()));
            bytes[96..112].copy_from_slice(bytemuck::bytes_of(&[
                extents[0], extents[1], extents[2], 0.0,
            ]));
            bytes
        };

        ctx.one_shot(|cmd| unsafe {
            // UNDEFINED -> GENERAL on every target.
            for image in images.iter().map(|i| i.image) {
                vulkan::image_barrier(
                    &device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::ImageAspectFlags::COLOR,
                );
            }
            vulkan::image_barrier(
                &device,
                cmd,
                depth.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::ImageAspectFlags::DEPTH,
            );
            let colors: Vec<_> = images
                .iter()
                .map(|image| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(image.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        })
                })
                .collect();
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            };
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: size as f32,
                height: size as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let info = vk::RenderingInfo::default()
                .render_area(scissor)
                .layer_count(1)
                .color_attachments(&colors)
                .depth_attachment(&depth_att);
            device.cmd_begin_rendering(cmd, &info);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            for batch in baked {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[batch.material.pbr.set, batch.material.base_set],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &push,
                );
                device.cmd_bind_vertex_buffers(cmd, 0, &[batch.vertex_buffer.buffer], &[0]);
                device.cmd_bind_index_buffer(
                    cmd,
                    batch.index_buffer.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(cmd, batch.index_count, 1, 0, 0, 0);
            }
            device.cmd_end_rendering(cmd);
        });
        // Readback copies in a second submission.
        ctx.one_shot(|cmd| unsafe {
            for (i, image) in images.iter().enumerate() {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: size,
                        height: size,
                        depth: 1,
                    });
                device.cmd_copy_image_to_buffer(
                    cmd,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    buffers[i].buffer,
                    &[region],
                );
            }
        });

        let half = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .chunks_exact(8)
                .flat_map(|px| {
                    px.chunks_exact(2)
                        .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
                        .collect::<Vec<f32>>()
                })
                .collect()
        };
        let read = |b: &GpuBuffer| {
            b.mapped()
                .map(|m| m.to_vec())
                .ok_or_else(|| FallbackReason::Gpu("readback not host visible".into()))
        };
        Ok(CapturedView {
            albedo: half(&read(&buffers[0])?),
            normal_ao: half(&read(&buffers[1])?),
            mr: half(&read(&buffers[2])?),
            emissive: half(&read(&buffers[3])?),
            depth: read(&buffers[4])?
                .chunks_exact(2)
                .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
                .collect(),
        })
    }
}

/// Build a compute pipeline from bake_post.wgsl entry `entry` with `types`
/// as the set-0 binding descriptor types (bindings are 1..=n).
fn cs_pipeline(
    ctx: &Arc<GpuContext>,
    module: vk::ShaderModule,
    entry: &str,
    types: &[vk::DescriptorType],
    label: &str,
) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout) {
    let device = ctx.device().clone();
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = types
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32 + 1)
                .descriptor_type(*ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("post dsl");
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(16);
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&dsl))
                .push_constant_ranges(std::slice::from_ref(&push)),
            None,
        )
    }
    .expect("post layout");
    let entry_name = std::ffi::CString::new(entry).unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry_name);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipe = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, err)| err)
            .unwrap_or_else(|err| panic!("{label}: create compute pipeline: {err:?}"))[0]
    };
    (pipe, layout, dsl)
}

/// Build the material descriptor-set layout shared with `model_pbr.wgsl`.
fn pbr_material_dsl(ctx: &Arc<GpuContext>) -> vk::DescriptorSetLayout {
    let mut bindings = vec![
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    for index in 0..5u32 {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(1 + index * 2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(2 + index * 2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
    }
    unsafe {
        ctx.device().create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("capture material set layout")
}

fn create_capture_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
) -> vk::Pipeline {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(c"vs_main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
            .name(c"fs_main"),
    ];
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<PbrVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .location(3)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(32),
        vk::VertexInputAttributeDescription::default()
            .location(4)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(40),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_attachments: Vec<_> = (0..5)
        .map(|_| {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)
        })
        .collect();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&CAPTURE_FORMATS)
        .depth_attachment_format(DEPTH_FORMAT);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_state)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    unsafe {
        ctx.device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, err)| err)
            .expect("create capture pipeline")[0]
    }
}
