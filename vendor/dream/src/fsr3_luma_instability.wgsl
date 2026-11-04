// FSR 3.1 pass: luma instability (ffx_fsr3upscaler_luma_instability.h).
// Maintains the four-frame luma history used to detect unstable (flickering)
// pixels and locks them.

struct LumaInstabilityFactorData {
    luma_history: vec4<f32>,
    luma_instability_factor: f32,
}

fn ComputeLumaInstabilityFactor(data_in: LumaInstabilityFactorData, fCurrentFrameLuma: f32, fFarthestDepthInMeters: f32) -> LumaInstabilityFactorData {
    _ = fFarthestDepthInMeters;
    var data = data_in;
    let N_MINUS_1 = 0;
    let N_MINUS_2 = 1;
    let N_MINUS_3 = 2;
    let N_MINUS_4 = 3;

    var fLumaInstability = 0.0;
    let fDiffs0 = fCurrentFrameLuma - data.luma_history[N_MINUS_1];
    let fSimilarity0 = MinDividedByMaxOr(fCurrentFrameLuma, data.luma_history[N_MINUS_1], 1.0);

    var fMaxSimilarity = fSimilarity0;

    if (fSimilarity0 < 1.0) {
        for (var i = N_MINUS_2; i <= N_MINUS_4; i++) {
            let fDiffs1 = fCurrentFrameLuma - data.luma_history[i];
            let fSimilarity1 = MinDividedByMax(fCurrentFrameLuma, data.luma_history[i]);
            if (sign(fDiffs0) == sign(fDiffs1)) {
                fMaxSimilarity = max(fMaxSimilarity, fSimilarity1);
            }
        }
        fLumaInstability = select(0.0, 1.0, fMaxSimilarity > fSimilarity0);
    }

    // Shift history
    data.luma_history[N_MINUS_4] = data.luma_history[N_MINUS_3];
    data.luma_history[N_MINUS_3] = data.luma_history[N_MINUS_2];
    data.luma_history[N_MINUS_2] = data.luma_history[N_MINUS_1];
    data.luma_history[N_MINUS_1] = fCurrentFrameLuma;

    data.luma_history /= Exposure();

    data.luma_instability_factor = fLumaInstability * select(0.0, 1.0, data.luma_history[N_MINUS_4] != 0.0);

    return data;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iPxPos = vec2<i32>(gid.xy);
    if (!IsOnScreen(iPxPos, RenderSize())) {
        return;
    }

    var data: LumaInstabilityFactorData;
    data.luma_instability_factor = 0.0;
    data.luma_history = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let fDilatedMotionVector = LoadDilatedMotionVector(iPxPos);
    let fUv = (vec2<f32>(iPxPos) + 0.5) / vec2<f32>(RenderSize());
    let fUvCurrFrameJittered = fUv + Jitter() / vec2<f32>(RenderSize());
    let fUvPrevFrameJittered = fUv + PreviousFrameJitter() / vec2<f32>(PreviousFrameRenderSize());
    let fReprojectedUv = fUvPrevFrameJittered + fDilatedMotionVector;

    if (IsUvInside(fReprojectedUv)) {
        let fUvReactive_HW = ClampUv(fUvCurrFrameJittered, RenderSize(), MaxRenderSize());

        let fDilatedReactiveMasks = SampleDilatedReactiveMasks(fUvReactive_HW);
        let fReactiveMask = ffxSaturate(fDilatedReactiveMasks[REACTIVE]);
        let fDisocclusion = ffxSaturate(fDilatedReactiveMasks[DISOCCLUSION]);
        let fShadingChange = ffxSaturate(fDilatedReactiveMasks[SHADING_CHANGE]);
        let fAccumulation = ffxSaturate(fDilatedReactiveMasks[ACCUMULATION]);

        let bAccumulationFactor = fAccumulation > 0.9;
        let bComputeInstability = bAccumulationFactor;

        if (bComputeInstability) {
            let fUv_HW = ClampUv(fUvCurrFrameJittered, RenderSize(), MaxRenderSize());
            let fCurrentFrameLuma = SampleCurrentLuma(fUv_HW) * Exposure();

            let fReprojectedUv_HW = ClampUv(fReprojectedUv, PreviousFrameRenderSize(), MaxRenderSize());
            data.luma_history = SampleLumaHistory(fReprojectedUv_HW) * DeltaPreExposure() * Exposure();

            let fFarthestDepthUv_HW = ClampUv(fUvCurrFrameJittered, RenderSize() / 2, RenderSize() / 2);
            let fFarthestDepthInMeters = SampleFarthestDepthMip1(fFarthestDepthUv_HW);

            data = ComputeLumaInstabilityFactor(data, fCurrentFrameLuma, fFarthestDepthInMeters);

            let fVelocityWeight = 1.0 - ffxSaturate(Get4KVelocity(fDilatedMotionVector) / 20.0);
            data.luma_instability_factor *= fVelocityWeight * (1.0 - fDisocclusion) * (1.0 - fReactiveMask) * (1.0 - fShadingChange);
        }
    }

    StoreLumaHistory(iPxPos, data.luma_history);
    StoreLumaInstability(iPxPos, data.luma_instability_factor);
}
