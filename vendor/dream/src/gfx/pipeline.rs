use super::vulkan;
use super::{
    DEPTH_FORMAT, GpuContext, GpuImage, SCENE_FORMAT, SceneTargets, SurfaceError, VELOCITY_FORMAT,
};
use ash::vk;
use std::sync::Arc;

pub(super) fn scaled_size(size: (u32, u32), scale: f32) -> (u32, u32) {
    (
        ((size.0 as f32 * scale).round() as u32).max(1),
        ((size.1 as f32 * scale).round() as u32).max(1),
    )
}

/// The scene pass' viewport and scissor: the region of the render target the
/// app's scene panel occupies, mapped from display pixels into render pixels.
/// The viewport itself stays the full target; the scissor does the confining,
/// so world coordinates/rasterized geometry stay in framebuffer space.
pub(super) fn scene_region(
    display_rect: Option<crate::ui::Rect>,
    render: (u32, u32),
    display: (u32, u32),
) -> (vk::Viewport, vk::Rect2D) {
    let viewport = make_viewport(render);
    let Some(rect) = display_rect else {
        return (
            viewport,
            vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: render.0,
                    height: render.1,
                },
            },
        );
    };
    let sx = render.0 as f32 / display.0.max(1) as f32;
    let sy = render.1 as f32 / display.1.max(1) as f32;
    let x0 = (rect.x * sx).floor().clamp(0.0, render.0 as f32) as i32;
    // Vulkan's framebuffer row zero is the display's bottom row: flip y.
    let y0 = (render.1 as f32 - (rect.y + rect.h) * sy)
        .floor()
        .clamp(0.0, render.1 as f32) as i32;
    let x1 = ((rect.x + rect.w) * sx).ceil().clamp(0.0, render.0 as f32) as i32;
    let y1 = (render.1 as f32 - rect.y * sy)
        .ceil()
        .clamp(0.0, render.1 as f32) as i32;
    (
        viewport,
        vk::Rect2D {
            offset: vk::Offset2D { x: x0, y: y0 },
            extent: vk::Extent2D {
                width: (x1 - x0).max(0) as u32,
                height: (y1 - y0).max(0) as u32,
            },
        },
    )
}

/// Vulkan's default NDC has y pointing down; a negative viewport height flips it
/// back to the y-up, depth-0..1 convention every shader here is written against.
pub(super) fn make_viewport(size: (u32, u32)) -> vk::Viewport {
    vk::Viewport {
        x: 0.0,
        y: size.1 as f32,
        width: size.0 as f32,
        height: -(size.1 as f32),
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

pub(super) fn attachment(
    view: vk::ImageView,
    load: vk::AttachmentLoadOp,
    clear: [f32; 4],
) -> vk::RenderingAttachmentInfo<'static> {
    vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL)
        .load_op(load)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue { float32: clear },
        })
}

pub(super) fn depth_attachment(
    view: vk::ImageView,
    clear: f32,
) -> vk::RenderingAttachmentInfo<'static> {
    vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: clear,
                stencil: 0,
            },
        })
}

// Stage masks live on the barrier command, not the image barrier, so the stage
// parameters are accepted only to keep call sites readable.
pub(super) fn general_barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    _src_stage: vk::PipelineStageFlags,
    _dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
}

pub(super) fn binding(
    binding: u32,
    ty: vk::DescriptorType,
    stage: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(ty)
        .descriptor_count(1)
        .stage_flags(stage)
}

pub(super) fn create_dsl(
    ctx: &GpuContext,
    label: &str,
    bindings: &[vk::DescriptorSetLayoutBinding<'_>],
) -> vk::DescriptorSetLayout {
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
    unsafe { ctx.device().create_descriptor_set_layout(&info, None) }
        .unwrap_or_else(|err| panic!("{label}: create descriptor set layout: {err:?}"))
}

pub(super) fn create_pipeline_layout(
    ctx: &GpuContext,
    label: &str,
    sets: &[vk::DescriptorSetLayout],
    push_constants: &[vk::PushConstantRange],
) -> vk::PipelineLayout {
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(sets)
        .push_constant_ranges(push_constants);
    unsafe { ctx.device().create_pipeline_layout(&info, None) }
        .unwrap_or_else(|err| panic!("{label}: create pipeline layout: {err:?}"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_graphics_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    vertex_bindings: &[vk::VertexInputBindingDescription],
    vertex_attrs: &[vk::VertexInputAttributeDescription],
    layout: vk::PipelineLayout,
    color_formats: &[vk::Format],
    depth_format: Option<vk::Format>,
) -> vk::Pipeline {
    create_graphics_pipeline_fs(
        ctx,
        shader,
        vertex_bindings,
        vertex_attrs,
        layout,
        color_formats,
        depth_format,
        c"fs_main",
    )
}

/// [`create_graphics_pipeline`] with an explicit fragment entry point.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_graphics_pipeline_fs(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    vertex_bindings: &[vk::VertexInputBindingDescription],
    vertex_attrs: &[vk::VertexInputAttributeDescription],
    layout: vk::PipelineLayout,
    color_formats: &[vk::Format],
    depth_format: Option<vk::Format>,
    fragment_entry: &std::ffi::CStr,
) -> vk::Pipeline {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(c"vs_main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
            .name(fragment_entry),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(vertex_bindings)
        .vertex_attribute_descriptions(vertex_attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // Back-face culling with a negative viewport height reverses winding, so
    // front faces are clockwise.
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let mut depth_state = vk::PipelineDepthStencilStateCreateInfo::default();
    if depth_format.is_some() {
        depth_state = depth_state
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
    }
    // Vulkan requires one blend-state entry per colour attachment. Supplying
    // fewer leaves the extra attachments' write mask undefined: the scene pass
    // writes motion vectors into attachment 1, and a missing entry there means
    // the velocity target is never written (upscalers then reproject as if
    // nothing moved: clean while static, smeared on any camera motion).
    let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = color_formats
        .iter()
        .map(|_| {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)
        })
        .collect();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(color_formats)
        .depth_attachment_format(depth_format.unwrap_or(vk::Format::UNDEFINED));

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
            .expect("create graphics pipeline")[0]
    }
}

pub(super) fn create_descriptor_pool(ctx: &GpuContext, slots: usize) -> vk::DescriptorPool {
    let n = slots as u32;
    let sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(n),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(2),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(2),
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(n + 2)
        .pool_sizes(&sizes);
    unsafe { ctx.device().create_descriptor_pool(&info, None) }.expect("create descriptor pool")
}

#[allow(clippy::type_complexity)]
pub(super) fn allocate_sets(
    ctx: &GpuContext,
    pool: vk::DescriptorPool,
    scene_dsl: &vk::DescriptorSetLayout,
    blit_dsl: &vk::DescriptorSetLayout,
) -> (Vec<vk::DescriptorSet>, vk::DescriptorSet, vk::DescriptorSet) {
    let mut layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
    for _ in 0..ctx.max_frames_in_flight() {
        layouts.push(*scene_dsl);
    }
    // Blit source and present source share the sampled-image + sampler set.
    layouts.push(*blit_dsl);
    layouts.push(*blit_dsl);
    let sets = unsafe {
        ctx.device().allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts),
        )
    }
    .expect("allocate descriptor sets");
    let (scene_sets, tail) = sets.split_at(ctx.max_frames_in_flight());
    (scene_sets.to_vec(), tail[0], tail[1])
}

pub(super) fn sampled(view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL)
}

pub(super) fn sampler_info(sampler: vk::Sampler) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default().sampler(sampler)
}

pub(super) fn uniform_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    buffer: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(buffer))
}

pub(super) fn image_write<'a>(
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

pub(super) fn write_sets(ctx: &GpuContext, writes: &[vk::WriteDescriptorSet<'_>]) {
    unsafe { ctx.device().update_descriptor_sets(writes, &[]) };
}

pub(super) fn new_semaphore(ctx: &GpuContext) -> vk::Semaphore {
    unsafe {
        ctx.device()
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
    }
    .expect("create semaphore")
}

pub(super) fn new_fence(ctx: &GpuContext) -> vk::Fence {
    unsafe {
        ctx.device().create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )
    }
    .expect("create fence")
}

/// Classify a swapchain-path `VkResult` (acquire or present) so callers see
/// the resize/loss signals the presentation engine actually reported.
pub(super) fn map_surface_error(err: vk::Result) -> SurfaceError {
    match err {
        vk::Result::ERROR_OUT_OF_DATE_KHR => SurfaceError::Outdated,
        vk::Result::ERROR_SURFACE_LOST_KHR => SurfaceError::Lost,
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            SurfaceError::OutOfMemory
        }
        vk::Result::TIMEOUT => SurfaceError::Timeout,
        _ => SurfaceError::Other,
    }
}

/// The display image's format: the swapchain format, or its UNORM
/// counterpart for [`UiBlend::Gamma`](crate::ui::render::UiBlend::Gamma)
/// (encoded values blended as stored).
pub(super) fn display_format(
    swapchain: vk::Format,
    blend: crate::ui::render::UiBlend,
) -> vk::Format {
    match (blend, swapchain) {
        (crate::ui::render::UiBlend::Gamma, vk::Format::B8G8R8A8_SRGB) => {
            vk::Format::B8G8R8A8_UNORM
        }
        (crate::ui::render::UiBlend::Gamma, vk::Format::R8G8B8A8_SRGB) => {
            vk::Format::R8G8B8A8_UNORM
        }
        _ => swapchain,
    }
}

/// The composed display image: swapchain format, screen-row order.
pub(super) fn build_display(
    ctx: &Arc<GpuContext>,
    display_size: (u32, u32),
    format: vk::Format,
) -> GpuImage {
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC;
    let display = ctx.create_image("display", display_size.0, display_size.1, format, usage);
    let device = ctx.device().clone();
    let image = display.image;
    ctx.one_shot(|cmd| unsafe {
        vulkan::image_barrier(
            &device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            vk::ImageAspectFlags::COLOR,
        );
    });
    display
}

pub(super) fn build_targets(ctx: &Arc<GpuContext>, render_size: (u32, u32)) -> SceneTargets {
    let (w, h) = render_size;
    let color_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let depth_usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;

    let color = ctx.create_image("scene color", w, h, SCENE_FORMAT, color_usage);
    let velocity = ctx.create_image("velocity", w, h, VELOCITY_FORMAT, color_usage);
    let depth = ctx.create_image("scene depth", w, h, DEPTH_FORMAT, depth_usage);

    let images = [
        (color.image, vk::ImageAspectFlags::COLOR),
        (velocity.image, vk::ImageAspectFlags::COLOR),
        (depth.image, vk::ImageAspectFlags::DEPTH),
    ];
    ctx.one_shot(|cmd| unsafe {
        for (image, aspect) in &images {
            vulkan::image_barrier(
                ctx.device(),
                cmd,
                *image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::empty(),
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                *aspect,
            );
        }
        ctx.device().cmd_clear_color_image(
            cmd,
            color.image,
            vk::ImageLayout::GENERAL,
            &vk::ClearColorValue {
                float32: [0.48, 0.69, 0.89, 1.0],
            },
            &[vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        );
    });

    SceneTargets {
        color,
        velocity,
        depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Present reports the same conditions as acquire; the distinction is
    /// what lets a caller rebuild the swapchain instead of treating a resize
    /// as an undiagnosable failure.
    #[test]
    fn swapchain_results_keep_their_resize_signal() {
        assert_eq!(
            map_surface_error(vk::Result::ERROR_OUT_OF_DATE_KHR),
            SurfaceError::Outdated
        );
        assert_eq!(
            map_surface_error(vk::Result::ERROR_SURFACE_LOST_KHR),
            SurfaceError::Lost
        );
        assert_eq!(
            map_surface_error(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            SurfaceError::OutOfMemory
        );
        assert_eq!(
            map_surface_error(vk::Result::ERROR_UNKNOWN),
            SurfaceError::Other
        );
    }
}
