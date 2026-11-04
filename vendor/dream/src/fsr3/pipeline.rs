use crate::gfx::GpuContext;
use ash::vk;
use std::sync::Arc;

pub(super) fn descriptor_bindings() -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    let mut bindings = Vec::new();
    let mut add = |binding: u32, ty: vk::DescriptorType| {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
    };
    add(0, vk::DescriptorType::UNIFORM_BUFFER);
    add(1, vk::DescriptorType::UNIFORM_BUFFER);
    add(2, vk::DescriptorType::SAMPLER);
    add(3, vk::DescriptorType::SAMPLER);
    for i in 0..19u32 {
        add(image_binding(i), vk::DescriptorType::SAMPLED_IMAGE);
    }
    add(60, vk::DescriptorType::SAMPLED_IMAGE); // downsample source depth
    add(30, vk::DescriptorType::STORAGE_BUFFER);
    add(31, vk::DescriptorType::STORAGE_BUFFER);
    for i in 0..16u32 {
        add(40 + i, vk::DescriptorType::STORAGE_IMAGE);
    }
    add(63, vk::DescriptorType::STORAGE_IMAGE); // unused r16f downsample target
    add(62, vk::DescriptorType::STORAGE_IMAGE); // unused rg16f downsample target
    bindings
}

/// Sampled-image bindings are 10..=28 for the fsr resources in index order.
pub(super) fn image_binding(index: u32) -> u32 {
    10 + index
}

pub(super) fn uniform<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}

pub(super) fn storage_buffer<'a>(
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

pub(super) fn image<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorImageInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
        .image_info(std::slice::from_ref(info))
}

pub(super) fn storage_image<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorImageInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(std::slice::from_ref(info))
}

pub(super) fn sampler_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorImageInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .image_info(std::slice::from_ref(info))
}

pub(super) fn barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let memory = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &memory,
            &[],
            &[],
        );
    }
}

pub(super) fn create_compute_pipelines<const N: usize>(
    ctx: &Arc<GpuContext>,
    specs: [(vk::ShaderModule, &std::ffi::CStr); N],
    layout: vk::PipelineLayout,
) -> [vk::Pipeline; N] {
    let infos: Vec<vk::ComputePipelineCreateInfo<'_>> = specs
        .iter()
        .map(|&(module, entry)| {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)
        })
        .collect();
    let vec = unsafe {
        ctx.device()
            .create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
            .map_err(|(_, err)| err)
            .expect("create compute pipelines")
    };
    vec.try_into().expect("pipeline count matches specs")
}
