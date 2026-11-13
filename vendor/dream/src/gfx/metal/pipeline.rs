use super::GpuContext;
use super::shader::{MslBindings, VERTEX_BUFFER_SLOT};
use crate::ui::render::UiBlend;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCompareFunction,
    MTLComputePipelineState, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice,
    MTLLibrary, MTLPixelFormat, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLScissorRect, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction, MTLViewport,
};

pub(crate) fn scaled_size(size: (u32, u32), scale: f32) -> (u32, u32) {
    (
        ((size.0 as f32 * scale).round() as u32).max(1),
        ((size.1 as f32 * scale).round() as u32).max(1),
    )
}

pub(crate) fn make_viewport(size: (u32, u32)) -> MTLViewport {
    MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: size.0.max(1) as f64,
        height: size.1.max(1) as f64,
        znear: 0.0,
        zfar: 1.0,
    }
}

pub(crate) fn scene_region(
    display_rect: Option<crate::ui::Rect>,
    render: (u32, u32),
    display: (u32, u32),
) -> (MTLViewport, MTLScissorRect) {
    let viewport = make_viewport(render);
    let Some(rect) = display_rect else {
        return (
            viewport,
            MTLScissorRect {
                x: 0,
                y: 0,
                width: render.0.max(1) as usize,
                height: render.1.max(1) as usize,
            },
        );
    };
    let sx = render.0 as f32 / display.0.max(1) as f32;
    let sy = render.1 as f32 / display.1.max(1) as f32;
    let x0 = (rect.x * sx).floor().clamp(0.0, render.0 as f32) as usize;
    let y0 = (render.1 as f32 - (rect.y + rect.h) * sy)
        .floor()
        .clamp(0.0, render.1 as f32) as usize;
    let x1 = ((rect.x + rect.w) * sx).ceil().clamp(0.0, render.0 as f32) as usize;
    let y1 = (render.1 as f32 - rect.y * sy)
        .ceil()
        .clamp(0.0, render.1 as f32) as usize;
    (
        viewport,
        MTLScissorRect {
            x: x0,
            y: y0,
            width: x1.saturating_sub(x0).max(1),
            height: y1.saturating_sub(y0).max(1),
        },
    )
}

pub(crate) fn display_format(swapchain_format: MTLPixelFormat, blend: UiBlend) -> MTLPixelFormat {
    match blend {
        UiBlend::Gamma => MTLPixelFormat::RGBA8Unorm,
        UiBlend::Linear => swapchain_format,
    }
}

pub(crate) fn present_fragment_entry(blend: UiBlend, display_fmt: MTLPixelFormat) -> &'static str {
    match blend {
        UiBlend::Gamma if super::is_srgb(display_fmt) => "fs_main",
        UiBlend::Gamma => "fs_gamma",
        UiBlend::Linear => "fs_main",
    }
}

pub(crate) fn composes_direct(
    swapchain_format: MTLPixelFormat,
    display_fmt: MTLPixelFormat,
    show_crosshair: bool,
    frame_capture: bool,
) -> bool {
    !show_crosshair && !frame_capture && swapchain_format == display_fmt
}

pub(crate) struct VertexAttr {
    pub location: usize,
    pub format: MTLVertexFormat,
    pub offset: usize,
}

pub(crate) fn make_vertex_descriptor(
    stride: usize,
    attrs: &[VertexAttr],
) -> Retained<MTLVertexDescriptor> {
    let desc = MTLVertexDescriptor::vertexDescriptor();
    let layouts = desc.layouts();
    let layout = unsafe { layouts.objectAtIndexedSubscript(VERTEX_BUFFER_SLOT as usize) };
    unsafe {
        layout.setStride(stride);
        layout.setStepFunction(MTLVertexStepFunction::PerVertex);
        layout.setStepRate(1);
    }
    let attributes = desc.attributes();
    for attr in attrs {
        let a = unsafe { attributes.objectAtIndexedSubscript(attr.location) };
        unsafe {
            a.setFormat(attr.format);
            a.setOffset(attr.offset);
            a.setBufferIndex(VERTEX_BUFFER_SLOT as usize);
        }
    }
    desc
}

#[allow(clippy::too_many_arguments)]
fn get_function(
    lib: &ProtocolObject<dyn MTLLibrary>,
    entry_points: &[(String, String)],
    entry: &str,
    label: &str,
    kind: &str,
) -> Retained<ProtocolObject<dyn objc2_metal::MTLFunction>> {
    let msl_name = entry_points
        .iter()
        .find(|(wgsl, _)| wgsl == entry)
        .map(|(_, msl)| msl.as_str())
        .unwrap_or(entry);
    lib.newFunctionWithName(&NSString::from_str(msl_name))
        .unwrap_or_else(|| panic!("{label}: missing {kind} function '{entry}' (msl '{msl_name}')"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_render_pipeline(
    ctx: &GpuContext,
    label: &str,
    source: &str,
    bindings: &MslBindings,
    vs_entry: &str,
    fs_entry: &str,
    color_formats: &[MTLPixelFormat],
    depth_format: Option<MTLPixelFormat>,
    vertex_desc: Option<&MTLVertexDescriptor>,
    blend_first_attachment: bool,
) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
    create_render_pipeline_opts(
        ctx,
        label,
        source,
        bindings,
        vs_entry,
        fs_entry,
        color_formats,
        depth_format,
        vertex_desc,
        blend_first_attachment,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_render_pipeline_unflipped(
    ctx: &GpuContext,
    label: &str,
    source: &str,
    bindings: &MslBindings,
    vs_entry: &str,
    fs_entry: &str,
    color_formats: &[MTLPixelFormat],
    depth_format: Option<MTLPixelFormat>,
    vertex_desc: Option<&MTLVertexDescriptor>,
    blend_first_attachment: bool,
) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
    create_render_pipeline_opts(
        ctx,
        label,
        source,
        bindings,
        vs_entry,
        fs_entry,
        color_formats,
        depth_format,
        vertex_desc,
        blend_first_attachment,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_render_pipeline_opts(
    ctx: &GpuContext,
    label: &str,
    source: &str,
    bindings: &MslBindings,
    vs_entry: &str,
    fs_entry: &str,
    color_formats: &[MTLPixelFormat],
    depth_format: Option<MTLPixelFormat>,
    vertex_desc: Option<&MTLVertexDescriptor>,
    blend_first_attachment: bool,
    flip_y: bool,
) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
    let (lib, eps) = if flip_y {
        ctx.compile_library(source, label, bindings)
    } else {
        ctx.compile_library_unflipped(source, label, bindings)
    };
    let vs_fn = get_function(&lib, &eps, vs_entry, label, "vertex");
    let fs_fn = get_function(&lib, &eps, fs_entry, label, "fragment");

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setLabel(Some(&NSString::from_str(label)));
    desc.setVertexFunction(Some(&vs_fn));
    desc.setFragmentFunction(Some(&fs_fn));
    if let Some(vd) = vertex_desc {
        desc.setVertexDescriptor(Some(vd));
    }
    let attachments = desc.colorAttachments();
    for (i, &fmt) in color_formats.iter().enumerate() {
        let att = unsafe { attachments.objectAtIndexedSubscript(i) };
        att.setPixelFormat(fmt);
        att.setWriteMask(MTLColorWriteMask::All);
        if i == 0 && blend_first_attachment {
            att.setBlendingEnabled(true);
            att.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
            att.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            att.setRgbBlendOperation(MTLBlendOperation::Add);
            att.setSourceAlphaBlendFactor(MTLBlendFactor::One);
            att.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            att.setAlphaBlendOperation(MTLBlendOperation::Add);
        } else {
            att.setBlendingEnabled(false);
        }
    }
    if let Some(dfmt) = depth_format {
        desc.setDepthAttachmentPixelFormat(dfmt);
    }

    ctx.device()
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .unwrap_or_else(|err| panic!("{label}: create render pipeline failed: {err:?}"))
}

pub(crate) fn create_compute_pipeline(
    ctx: &GpuContext,
    label: &str,
    source: &str,
    bindings: &MslBindings,
    entry: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let (lib, eps) = ctx.compile_library(source, label, bindings);
    let func = get_function(&lib, &eps, entry, label, "compute");
    ctx.device()
        .newComputePipelineStateWithFunction_error(&func)
        .unwrap_or_else(|err| panic!("{label}: create compute pipeline failed: {err:?}"))
}

pub(crate) fn create_depth_stencil_state(
    ctx: &GpuContext,
    depth_test: bool,
    depth_write: bool,
    compare: MTLCompareFunction,
) -> Retained<ProtocolObject<dyn MTLDepthStencilState>> {
    let desc = MTLDepthStencilDescriptor::new();
    desc.setDepthCompareFunction(if depth_test {
        compare
    } else {
        MTLCompareFunction::Always
    });
    desc.setDepthWriteEnabled(depth_write);
    ctx.device()
        .newDepthStencilStateWithDescriptor(&desc)
        .expect("create Metal depth stencil state")
}
