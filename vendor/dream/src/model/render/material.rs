use super::pipeline::create_pbr_pipeline;
use super::{MaterialFactors, PbrMaterialGpu, PbrState, scene_set_layout};
use crate::gfx::vulkan::GpuContext;
use crate::model::Material;
use crate::texture::{self, GpuTexture, TextureFormat, TextureOptions};
use ash::vk;
use std::sync::Arc;

// ---------------------------------------------------------------- PBR path

/// Build the interleaved PBR vertex stream (uv1 + tangents; missing tangent
/// attribute emits a zero vector so the shader falls back to a derivative
/// frame).
pub(crate) fn pbr_vertex_buffer(
    ctx: &Arc<GpuContext>,
    mesh: &crate::model::Mesh,
) -> crate::gfx::GpuBuffer {
    // One interleaved buffer, cast once (no per-vertex Vec).
    let vertices = super::pbr_vertices(mesh);
    ctx.create_vertex_buffer(
        &format!("model pbr mesh {}", mesh.name),
        bytemuck::cast_slice(&vertices),
    )
}

/// One material map: real pixels when present, else a neutral 1x1 that is
/// identity under the factor multiply (black for emissive, flat normal for
/// the normal map, white elsewhere).
fn map_texture(
    ctx: &Arc<GpuContext>,
    label: &str,
    map: Option<&crate::model::TextureRef>,
    fallback: [u8; 4],
    format: TextureFormat,
) -> GpuTexture {
    match map {
        Some(reference) => {
            // The file can ask per-axis wrap/filter; the sampler has one
            // shared address mode — take S (the overwhelmingly common case
            // is symmetric anyway).
            let sampler = texture::SamplerDesc {
                filter: match reference.mag_filter {
                    crate::model::TexelFilter::Nearest => texture::Filter::Nearest,
                    crate::model::TexelFilter::Linear => texture::Filter::Linear,
                },
                mip: texture::MipFilter::Linear,
                address: match reference.wrap_s {
                    crate::model::Wrap::Repeat => texture::AddressMode::Repeat,
                    crate::model::Wrap::ClampToEdge => texture::AddressMode::ClampToEdge,
                    crate::model::Wrap::MirroredRepeat => texture::AddressMode::MirrorRepeat,
                },
                anisotropy: true,
            };
            GpuTexture::from_rgba8(
                ctx,
                label,
                &reference.texture.pixels,
                reference.texture.width,
                reference.texture.height,
                TextureOptions {
                    format,
                    sampler,
                    mip_levels: None,
                },
            )
        }
        None => GpuTexture::from_rgba8(
            ctx,
            label,
            &fallback,
            1,
            1,
            TextureOptions {
                format,
                mip_levels: Some(1),
                ..Default::default()
            },
        ),
    }
}

/// Pack a map's uv transform: `(scale.xy, offset.xy)` and `(cos, sin, set, _)`.
pub(crate) fn pack_transform(map: Option<&crate::model::TextureRef>) -> ([f32; 4], [f32; 4]) {
    let Some(map) = map else {
        return ([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    };
    let (offset, rotation, scale) = map
        .transform
        .map(|t| (t.offset, t.rotation, t.scale))
        .unwrap_or(([0.0, 0.0], 0.0, [1.0, 1.0]));
    (
        [scale[0], scale[1], offset[0], offset[1]],
        [rotation.cos(), rotation.sin(), map.uv_set() as f32, 0.0],
    )
}

impl PbrMaterialGpu {
    pub(crate) fn new(
        ctx: &Arc<GpuContext>,
        layout: vk::DescriptorSetLayout,
        material: &Material,
    ) -> Self {
        let name = material.name.as_str();
        let textures: Vec<Arc<GpuTexture>> = [
            map_texture(
                ctx,
                &format!("{name} base"),
                material.base_color_map.as_ref(),
                [255; 4],
                TextureFormat::Srgb,
            ),
            map_texture(
                ctx,
                &format!("{name} mr"),
                material.metallic_roughness_texture.as_ref(),
                [255; 4],
                TextureFormat::Linear,
            ),
            map_texture(
                ctx,
                &format!("{name} normal"),
                material.normal_texture.as_ref(),
                [128, 128, 255, 255],
                TextureFormat::Linear,
            ),
            map_texture(
                ctx,
                &format!("{name} ao"),
                material.occlusion_texture.as_ref(),
                [255; 4],
                TextureFormat::Linear,
            ),
            map_texture(
                ctx,
                &format!("{name} emissive"),
                material.emissive_texture.as_ref(),
                [0; 4],
                TextureFormat::Srgb,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect();

        let maps: [Option<&crate::model::TextureRef>; 5] = [
            material.base_color_map.as_ref(),
            material.metallic_roughness_texture.as_ref(),
            material.normal_texture.as_ref(),
            material.occlusion_texture.as_ref(),
            material.emissive_texture.as_ref(),
        ];
        let mut factors = MaterialFactors {
            params: [
                material.metallic_factor,
                material.roughness_factor,
                material.alpha_cutoff,
                match material.alpha_mode {
                    crate::model::AlphaMode::Opaque => 0.0,
                    crate::model::AlphaMode::Mask => 1.0,
                    crate::model::AlphaMode::Blend => 2.0,
                },
            ],
            params2: [
                material.occlusion_strength,
                material.normal_scale,
                0.0,
                material.base_color[3], // materialAlphaFactor
            ],
            emissive: [
                material.emissive_factor[0],
                material.emissive_factor[1],
                material.emissive_factor[2],
                material.emissive_strength,
            ],
            ..Default::default()
        };
        for (index, map) in maps.iter().enumerate() {
            let (uv, tf) = pack_transform(*map);
            factors.uv[index] = uv;
            factors.tf[index] = tf;
        }
        let mut factor_buffer = ctx.create_host_buffer(
            &format!("{name} factors"),
            std::mem::size_of::<MaterialFactors>() as u64,
        );
        factor_buffer.write(0, bytemuck::bytes_of(&factors));
        {
            let pool_size = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(5),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(5),
            ];
            let pool = unsafe {
                ctx.device().create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_size),
                    None,
                )
            }
            .expect("material descriptor pool");
            let set = unsafe {
                ctx.device().allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(std::slice::from_ref(&layout)),
                )
            }
            .expect("material descriptor set")[0];
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(factor_buffer.buffer)
                .offset(0)
                .range(std::mem::size_of::<MaterialFactors>() as u64);
            let mut writes = vec![
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info)),
            ];
            // Keep the image infos alive until update_descriptor_sets runs.
            let image_infos: Vec<_> = textures
                .iter()
                .flat_map(|texture| {
                    [
                        vk::DescriptorImageInfo::default()
                            .image_view(texture.view())
                            .image_layout(vk::ImageLayout::GENERAL),
                        vk::DescriptorImageInfo::default().sampler(texture.sampler()),
                    ]
                })
                .collect();
            for (index, info) in image_infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1 + index as u32)
                        .descriptor_type(if index % 2 == 0 {
                            vk::DescriptorType::SAMPLED_IMAGE
                        } else {
                            vk::DescriptorType::SAMPLER
                        })
                        .image_info(std::slice::from_ref(info)),
                );
            }
            unsafe { ctx.device().update_descriptor_sets(&writes, &[]) };
            return Self {
                ctx: Arc::clone(ctx),
                _textures: textures,
                _factors: factor_buffer,
                pool,
                set,
            };
        }
    }
}

impl PbrState {
    pub(super) fn new(
        ctx: &Arc<GpuContext>,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
        dsl_joints: vk::DescriptorSetLayout,
        lighting: &crate::lighting::Lighting,
    ) -> Self {
        // set 2 = material: factors uniform + five image/sampler pairs
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        for index in 0..5 {
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
        let dsl_material = unsafe {
            ctx.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .expect("pbr material set layout");
        // set 1 reuses the renderer's joint/draw layout: the PBR shader reads
        // the same DrawData block (base colour carries opacity).
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(144);
        let layout = {
            let set_layouts = [
                scene_set_layout(ctx),
                dsl_joints,
                dsl_material,
                lighting.set_layout(),
            ];
            unsafe {
                ctx.device().create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(std::slice::from_ref(&push)),
                    None,
                )
            }
            .expect("pbr pipeline layout")
        };
        let shader = ctx.compile_shader(
            &format!(
                "{}{}",
                crate::lighting::SHADER,
                include_str!("../model_pbr.wgsl")
            ),
            "model_pbr",
        );
        let pipeline = create_pbr_pipeline(ctx, shader, layout, color_formats, depth_format, false);
        let pipeline_blended =
            create_pbr_pipeline(ctx, shader, layout, color_formats, depth_format, true);
        unsafe { ctx.device().destroy_shader_module(shader, None) };
        Self {
            pipeline,
            pipeline_blended,
            layout,
            dsl_material,
        }
    }
}
