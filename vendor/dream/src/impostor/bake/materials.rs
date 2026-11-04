use super::ImpMaterialParams;
use super::upload::upload_image_layers;
use crate::gfx::GpuBuffer;
use crate::gfx::GpuImage;
use crate::gfx::vulkan::GpuContext;
use crate::model::render::pack_transform;
use crate::model::{AlphaMode, Material, TexelFilter, TextureRef, Wrap};
use ash::vk;
use glam::Vec2;
use std::sync::Arc;

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

fn decoded_texel(
    texture: &crate::texture::DecodedTexture,
    x: i32,
    y: i32,
    wrap_s: Wrap,
    wrap_t: Wrap,
) -> [f32; 4] {
    if texture.width == 0 || texture.height == 0 {
        return [1.0; 4];
    }
    let x = wrap_index(x, texture.width, wrap_s);
    let y = wrap_index(y, texture.height, wrap_t);
    let i = (y * texture.width as usize + x) * 4;
    [
        texture.pixels[i] as f32 / 255.0,
        texture.pixels[i + 1] as f32 / 255.0,
        texture.pixels[i + 2] as f32 / 255.0,
        texture.pixels[i + 3] as f32 / 255.0,
    ]
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

fn decoded_value(
    texture: &crate::texture::DecodedTexture,
    x: i32,
    y: i32,
    wrap_s: Wrap,
    wrap_t: Wrap,
    linear_rgb: bool,
) -> [f32; 4] {
    let mut px = decoded_texel(texture, x, y, wrap_s, wrap_t);
    if linear_rgb {
        for c in &mut px[..3] {
            *c = srgb_channel_to_linear(*c);
        }
    }
    px
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
    if filter == TexelFilter::Nearest {
        return decoded_value(
            texture,
            (uv.x * texture.width as f32).floor() as i32,
            (uv.y * texture.height as f32).floor() as i32,
            wrap_s,
            wrap_t,
            linear_rgb,
        );
    }
    let x = uv.x * texture.width as f32 - 0.5;
    let y = uv.y * texture.height as f32 - 0.5;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0.0; 4];
    for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
        for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
            let s = decoded_value(texture, x0 + dx, y0 + dy, wrap_s, wrap_t, linear_rgb);
            for c in 0..4 {
                out[c] += s[c] * wx * wy;
            }
        }
    }
    out
}

/// Area-weighted sample used when a source map is downsampled into the
/// material array. A plain bilinear read at the destination centre keeps too
/// much source frequency, so the array's level 0 would not correspond to the
/// source mip its UV Jacobian selects.
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
        for sx in ix0..=ix1 {
            let wx = (sx as f32 + 1.0).min(sx1) - (sx as f32).max(sx0);
            let w = wx * wy;
            if w <= 0.0 {
                continue;
            }
            let px = decoded_value(texture, sx, sy, wrap_s, wrap_t, linear_rgb);
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

fn material_map(material: &Material, index: usize) -> Option<&TextureRef> {
    match index {
        0 => material.base_color_map.as_ref(),
        1 => material.metallic_roughness_texture.as_ref(),
        2 => material.normal_texture.as_ref(),
        3 => material.occlusion_texture.as_ref(),
        4 => material.emissive_texture.as_ref(),
        _ => None,
    }
}

/// Resample one source map into the fixed-size texture array used by the
/// far-field voxel shader. Wrapping/filtering matches the source sampler.
pub(super) fn material_layer(
    material: &Material,
    index: usize,
    res: u32,
) -> crate::texture::DecodedTexture {
    let fallback = match index {
        0 => [255, 255, 255, 255],
        1 => [255, 255, 255, 255],
        2 => [128, 128, 255, 255],
        3 => [255, 255, 255, 255],
        _ => [0, 0, 0, 255],
    };
    let srgb = matches!(index, 0 | 4);
    let mut pixels = vec![0u8; res as usize * res as usize * 4];
    for y in 0..res {
        for x in 0..res {
            let uv = Vec2::new((x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32);
            let rgba = if let Some(map) = material_map(material, index) {
                if map.mag_filter == TexelFilter::Linear
                    && (map.texture.width > res || map.texture.height > res)
                {
                    area_sample_decoded(&map.texture, x, y, res, map.wrap_s, map.wrap_t, srgb)
                } else {
                    sample_decoded(
                        &map.texture,
                        uv,
                        map.wrap_s,
                        map.wrap_t,
                        map.mag_filter,
                        srgb,
                    )
                }
            } else {
                fallback.map(|v| v as f32 / 255.0)
            };
            let i = (y * res + x) as usize * 4;
            for c in 0..4 {
                let channel = if srgb && c < 3 {
                    linear_channel_to_srgb(rgba[c])
                } else {
                    rgba[c].clamp(0.0, 1.0)
                };
                pixels[i + c] = (channel * 255.0).round() as u8;
            }
        }
    }
    crate::texture::DecodedTexture {
        pixels,
        width: res,
        height: res,
    }
}

pub(super) fn imp_material_params(material: &Material) -> ImpMaterialParams {
    let maps = (0..5)
        .map(|index| material_map(material, index))
        .collect::<Vec<_>>();
    let mut out = ImpMaterialParams {
        base_color: material.base_color,
        params: [
            material.metallic_factor,
            material.roughness_factor,
            material.alpha_cutoff,
            match material.alpha_mode {
                AlphaMode::Opaque => 0.0,
                AlphaMode::Mask => 1.0,
                AlphaMode::Blend => 2.0,
            },
        ],
        params2: [
            material.occlusion_strength,
            material.normal_scale,
            material.double_sided as u8 as f32,
            0.0,
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
        let (uv, mut tf) = pack_transform(*map);
        if let Some(map) = map {
            let nearest = (map.mag_filter == TexelFilter::Nearest
                || map.min_filter == TexelFilter::Nearest) as u32;
            tf[3] = (map.wrap_s as u32 | ((map.wrap_t as u32) << 2) | (nearest << 4)) as f32;
        }
        out.uv[index] = uv;
        out.tf[index] = tf;
    }
    out
}

pub(super) fn material_gpu_data(
    ctx: &Arc<GpuContext>,
    materials: &[&Material],
    res: u32,
) -> ([GpuImage; 5], GpuBuffer, u32, usize) {
    let count = materials.len().max(1);
    let defaults = Material::default();
    let params: Vec<ImpMaterialParams> = (0..count)
        .map(|i| imp_material_params(materials.get(i).copied().unwrap_or(&defaults)))
        .collect();
    let table = ctx.upload_buffer(
        "impostor material table",
        bytemuck::cast_slice(&params),
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    );
    let formats = [
        vk::Format::R8G8B8A8_SRGB,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8B8A8_SRGB,
    ];
    let mut bytes = params.len() * std::mem::size_of::<ImpMaterialParams>();
    let mut mip_extent = res;
    let mut map_bytes = 0usize;
    while mip_extent > 0 {
        map_bytes += mip_extent as usize * mip_extent as usize * 4;
        mip_extent /= 2;
    }
    let maps: [GpuImage; 5] = std::array::from_fn(|index| {
        let layers: Vec<Vec<u8>> = (0..count)
            .map(|material_index| {
                let material = materials.get(material_index).copied().unwrap_or(&defaults);
                material_layer(material, index, res).pixels
            })
            .collect();
        bytes += map_bytes * layers.len();
        upload_image_layers(
            ctx,
            &format!("impostor material map {index}"),
            res,
            &layers,
            formats[index],
        )
    });
    (maps, table, count as u32, bytes)
}
