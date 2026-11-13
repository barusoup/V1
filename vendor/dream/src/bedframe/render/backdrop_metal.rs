use crate::gfx::metal::GpuContext;
use crate::gfx::metal::pipeline::{create_render_pipeline, make_viewport};
use crate::gfx::metal::shader::MslBindings;
use crate::texture::GpuTexture;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineState, MTLStoreAction,
    MTLTexture,
};
use std::sync::Arc;

const SHADER: &str = include_str!("../backdrop.wgsl");

pub(super) struct Backdrop {
    pub size: (u32, u32),
    pub mid: Arc<GpuTexture>,
    pub source: Arc<GpuTexture>,
    pub scratch: Arc<GpuTexture>,
    pub out: Arc<GpuTexture>,
}

impl Backdrop {
    pub(super) fn new(ctx: &Arc<GpuContext>, format: MTLPixelFormat, size: (u32, u32)) -> Self {
        let quarter = |n: u32| n.div_ceil(4).max(1);
        let half = |n: u32| n.div_ceil(2).max(1);
        let make = |label: &str, w: u32, h: u32| {
            Arc::new(GpuTexture::render_target(ctx, label, w, h, format))
        };
        Self {
            size,
            mid: make("backdrop mid", half(size.0), half(size.1)),
            source: make("backdrop source", quarter(size.0), quarter(size.1)),
            scratch: make("backdrop scratch", quarter(size.0), quarter(size.1)),
            out: make("backdrop out", quarter(size.0), quarter(size.1)),
        }
    }
}

pub(super) struct BlurPipelines {
    pub downsample: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub horizontal: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub vertical: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl BlurPipelines {
    pub(super) fn new(ctx: &GpuContext, format: MTLPixelFormat) -> Self {
        let bindings = MslBindings::new().texture(0, 0, 0).sampler(0, 1, 0);
        let downsample = create_render_pipeline(
            ctx,
            "backdrop downsample",
            include_str!("../../blit.wgsl"),
            &bindings,
            "vs_main",
            "fs_main",
            &[format],
            None,
            None,
            false,
        );
        let horizontal = create_render_pipeline(
            ctx,
            "backdrop blur H",
            SHADER,
            &bindings,
            "vs_main",
            "fs_h",
            &[format],
            None,
            None,
            false,
        );
        let vertical = create_render_pipeline(
            ctx,
            "backdrop blur V",
            SHADER,
            &bindings,
            "vs_main",
            "fs_v",
            &[format],
            None,
            None,
            false,
        );
        Self {
            downsample,
            horizontal,
            vertical,
        }
    }
}

pub(super) fn capture_backdrop_pass(
    ctx: &GpuContext,
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    blur: &BlurPipelines,
    backdrop: &Backdrop,
    target: &ProtocolObject<dyn MTLTexture>,
) {
    let run_pass = |from: &ProtocolObject<dyn MTLTexture>,
                    to: &ProtocolObject<dyn MTLTexture>,
                    to_size: (u32, u32),
                    pipeline: &ProtocolObject<dyn MTLRenderPipelineState>| {
        let rpd = MTLRenderPassDescriptor::new();
        let att = unsafe { rpd.colorAttachments().objectAtIndexedSubscript(0) };
        att.setTexture(Some(to));
        att.setLoadAction(MTLLoadAction::DontCare);
        att.setStoreAction(MTLStoreAction::Store);
        let enc = cmd
            .renderCommandEncoderWithDescriptor(&rpd)
            .expect("backdrop pass encoder");
        enc.setViewport(make_viewport(to_size));
        enc.setRenderPipelineState(pipeline);
        unsafe {
            enc.setFragmentTexture_atIndex(Some(from), 0);
            enc.setFragmentSamplerState_atIndex(Some(ctx.default_sampler()), 0);
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        enc.endEncoding();
    };

    run_pass(
        target,
        backdrop.mid.texture(),
        backdrop.mid.size(),
        &blur.downsample,
    );
    run_pass(
        backdrop.mid.texture(),
        backdrop.source.texture(),
        backdrop.source.size(),
        &blur.downsample,
    );

    let qsize = backdrop.out.size();
    let steps = [
        (&backdrop.source, &backdrop.scratch, &blur.horizontal),
        (&backdrop.scratch, &backdrop.out, &blur.vertical),
        (&backdrop.out, &backdrop.scratch, &blur.horizontal),
        (&backdrop.scratch, &backdrop.out, &blur.vertical),
        (&backdrop.out, &backdrop.scratch, &blur.horizontal),
        (&backdrop.scratch, &backdrop.out, &blur.vertical),
        (&backdrop.out, &backdrop.scratch, &blur.horizontal),
        (&backdrop.scratch, &backdrop.out, &blur.vertical),
    ];
    for (from, to, pipe) in steps {
        run_pass(from.texture(), to.texture(), qsize, pipe);
    }
}
