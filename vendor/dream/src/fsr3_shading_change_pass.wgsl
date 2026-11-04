// FSR 3.1 pass: shading change (ffx_fsr3upscaler_shading_change.h).
// Combines the first three shading-change pyramid levels into the shading-change
// map the reactivity pass consumes.

struct ShadingChangeLumaInfo {
    samples: array<f32, 3>,
}

fn ComputeShadingChangeLuma(fUv: vec2<f32>) -> ShadingChangeLumaInfo {
    var info: ShadingChangeLumaInfo;
    let fMipUv = ClampUv(fUv, ShadingChangeRenderSize(), ShadingChangeRenderSize());

    for (var iMipLevel = iShadingChangeMipStart; iMipLevel < 3; iMipLevel++) {
        let fSample = SampleSPDMipLevel(fMipUv, iMipLevel);
        info.samples[iMipLevel] = abs(fSample.x * fSample.y);
    }
    return info;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iPxPos = vec2<i32>(gid.xy);
    let size = ShadingChangeRenderSize();
    if (!IsOnScreen(iPxPos, size)) {
        return;
    }

    let fUv = (vec2<f32>(iPxPos) + 0.5) / vec2<f32>(size);
    let fUvJittered = fUv + Jitter() / vec2<f32>(RenderSize());

    let info = ComputeShadingChangeLuma(fUvJittered);

    let fScale = 1.0;
    var fShadingChange = 0.0;
    for (var iMipLevel = iShadingChangeMipStart; iMipLevel < 3; iMipLevel++) {
        if (info.samples[iMipLevel] > 0.0) {
            fShadingChange = max(fShadingChange, info.samples[iMipLevel]) * fScale;
        }
    }

    StoreShadingChange(iPxPos, ffxSaturate(fShadingChange));
}
