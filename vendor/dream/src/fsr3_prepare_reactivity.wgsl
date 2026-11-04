// FSR 3.1 pass: prepare reactivity (ffx_fsr3upscaler_prepare_reactivity.h).
// Produces the dilated reactive masks (reactive, disocclusion, shading change,
// accumulation) and the thin-feature locks.

fn ComputeDisocclusions(fUv: vec2<f32>, fMotionVectorIn: vec2<f32>, fCurrentDepthViewSpace: f32) -> f32 {
    let fNearestDepthInMeters = min(fCurrentDepthViewSpace * ViewSpaceToMetersFactor(), FSR3UPSCALER_FP16_MAX);
    let fReconstructedDeptMvThreshold = ReconstructedDepthMvPxThreshold(fNearestDepthInMeters);

    var fMotionVector = fMotionVectorIn;
    fMotionVector *= select(0.0, 1.0, Get4KVelocity(fMotionVector) > fReconstructedDeptMvThreshold);

    let fReprojectedUv = fUv + fMotionVector;
    let bilinearInfo = GetBilinearSamplingData(fReprojectedUv, RenderSize());

    var fDisocclusion = 0.0;
    var fWeightSum = 0.0;
    var bPotentialDisocclusion = true;

    for (var iSampleIndex = 0; iSampleIndex < 4 && bPotentialDisocclusion; iSampleIndex++) {
        let iOffset = bilinearInfo.offsets[iSampleIndex];
        let iSamplePos = ClampLoad(bilinearInfo.base_pos, iOffset, RenderSize());

        if (IsOnScreen(iSamplePos, RenderSize())) {
            let fWeight = bilinearInfo.weights[iSampleIndex];
            if (fWeight > fReconstructedDepthBilinearWeightThreshold) {
                let fPrevNearestDepthViewSpace = GetViewSpaceDepth(LoadReconstructedPrevDepth(iSamplePos));
                let fDepthDifference = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;

                bPotentialDisocclusion = bPotentialDisocclusion && (fDepthDifference > FSR3UPSCALER_FP32_MIN);

                if (bPotentialDisocclusion) {
                    let fHalfViewportWidth = length(vec2<f32>(RenderSize()) * 0.5);
                    let fDepthThreshold = max(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);

                    let Ksep = 1.37e-05;
                    let fRequiredDepthSeparation = Ksep * fHalfViewportWidth * fDepthThreshold;

                    fDisocclusion += ffxSaturate(fRequiredDepthSeparation / fDepthDifference) * fWeight;
                    fWeightSum += fWeight;
                }
            }
        }
    }

    fDisocclusion = select(0.0, ffxSaturate(1.0 - fDisocclusion / fWeightSum), bPotentialDisocclusion && fWeightSum > 0.0);

    return fDisocclusion;
}

fn ComputeMotionDivergence(fUv: vec2<f32>, fMotionVector: vec2<f32>, fCurrentDepthSample: f32) -> f32 {
    let iPxReprojectedPos = vec2<i32>((fUv + fMotionVector) * vec2<f32>(RenderSize()));
    let fReprojectedDepth = LoadDilatedDepth(iPxReprojectedPos);
    let fReprojectedMotionVector = LoadDilatedMotionVector(iPxReprojectedPos);

    let fReprojectedVelocity = Get4KVelocity(fReprojectedMotionVector);
    let f4KVelocity = Get4KVelocity(fMotionVector);

    let fNucleusDepthInMeters = GetViewSpaceDepthInMeters(fReprojectedDepth);
    let fCurrentDepthInMeters = GetViewSpaceDepthInMeters(fCurrentDepthSample);

    let fDistanceFactor = MinDividedByMax(fNucleusDepthInMeters, fCurrentDepthInMeters);
    let fVelocityFactor = ffxSaturate(f4KVelocity / 10.0);
    let fMotionVectorFieldConfidence = (1.0 - ffxSaturate(fReprojectedVelocity / f4KVelocity)) * fDistanceFactor * fVelocityFactor;

    return fMotionVectorFieldConfidence;
}

fn DilateReactiveMasks(iPxPos: vec2<i32>, fUv: vec2<f32>) -> f32 {
    var fDilatedReactiveMasks = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let sampleCoord = ClampLoad(iPxPos, vec2<i32>(x, y), RenderSize());
            fDilatedReactiveMasks = max(fDilatedReactiveMasks, LoadReactiveMask(sampleCoord));
        }
    }
    return fDilatedReactiveMasks;
}

fn DilateTransparencyAndCompositionMasks(iPxPos: vec2<i32>, fUv: vec2<f32>) -> f32 {
    _ = iPxPos;
    let fUvT = ClampUv(fUv, RenderSize(), RenderSize());
    return SampleTransparencyAndCompositionMask(fUvT);
}

fn ComputeThinFeatureConfidence(iPxPos: vec2<i32>) -> f32 {
    // 1 2 3
    // 4 0 5
    // 6 7 8
    let iNucleusIndex = 0;
    let iSampleOffsets = array<vec2<i32>, 9>(
        vec2<i32>(0, 0),
        vec2<i32>(-1, -1),
        vec2<i32>(0, -1),
        vec2<i32>(1, -1),
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(-1, 1),
        vec2<i32>(0, 1),
        vec2<i32>(1, 1));

    var fSamples: array<f32, 9>;
    var fLumaMin = FSR3UPSCALER_FP32_MAX;
    var fLumaMax = FSR3UPSCALER_FP32_MIN;

    for (var i = 0; i < 9; i++) {
        let iPxSamplePos = ClampLoad(iPxPos, iSampleOffsets[i], RenderSize());
        fSamples[i] = LoadCurrentLuma(iPxSamplePos) * Exposure();
        fLumaMin = min(fLumaMin, fSamples[i]);
        fLumaMax = max(fLumaMax, fSamples[i]);
    }

    let fThreshold = 0.9;
    var fDissimilarLumaMin = FSR3UPSCALER_FP32_MAX;
    var fDissimilarLumaMax = 0.0;

    var uPatternMask: u32 = 1u << u32(iNucleusIndex); // Flag nucleus as similar
    let uRejectionMasks = array<u32, 4>(
        (1u << 1u) | (1u << 2u) | (1u << 4u) | (1u << u32(iNucleusIndex)),
        (1u << 2u) | (1u << 3u) | (1u << 5u) | (1u << u32(iNucleusIndex)),
        (1u << 4u) | (1u << 6u) | (1u << 7u) | (1u << u32(iNucleusIndex)),
        (1u << 5u) | (1u << 7u) | (1u << 8u) | (1u << u32(iNucleusIndex)));

    var iBitIndex = 1u;
    for (var i = 1; i < 9; i++) {
        let fDifference = abs(fSamples[i] - fSamples[iNucleusIndex]) / (fLumaMax - fLumaMin);
        if (fDifference < fThreshold) {
            uPatternMask |= (1u << iBitIndex);
        } else {
            fDissimilarLumaMin = min(fDissimilarLumaMin, fSamples[i]);
            fDissimilarLumaMax = max(fDissimilarLumaMax, fSamples[i]);
        }
        iBitIndex += 1u;
    }

    let bIsRidge = fSamples[iNucleusIndex] > fDissimilarLumaMax || fSamples[iNucleusIndex] < fDissimilarLumaMin;
    if (!bIsRidge) {
        return 0.0;
    }

    for (var i = 0; i < 4; i++) {
        if ((uPatternMask & uRejectionMasks[i]) == uRejectionMasks[i]) {
            return 0.0;
        }
    }

    return 1.0 - fLumaMin / fLumaMax;
}

fn UpdateAccumulation(iPxPos: vec2<i32>, fUv: vec2<f32>, fMotionVector: vec2<f32>, fDisocclusion: f32, fShadingChange: f32) -> f32 {
    let fReprojectedUv = fUv + fMotionVector;
    var fAccumulation = 0.0;

    if (IsUvInside(fReprojectedUv)) {
        let fReprojectedUv_HW = ClampUv(fReprojectedUv, PreviousFrameRenderSize(), MaxRenderSize());
        fAccumulation = ffxSaturate(SampleAccumulation(fReprojectedUv_HW));
    }
    let fAccumulationAddedPerFrame = AccumulationAddedPerFrame();

    fAccumulation = ffxLerp(fAccumulation, 0.0, fShadingChange);

    let fMinDisocclusionAccumulation = MinDisocclusionAccumulation();
    fAccumulation = ffxLerp(fAccumulation, min(fMinDisocclusionAccumulation, fAccumulation), fDisocclusion);

    fAccumulation *= select(0.0, 1.0, round(fAccumulation * 100.0) > 1.0);

    let fAccumulatedFramesToStore = ffxSaturate(fAccumulation + fAccumulationAddedPerFrame);
    StoreAccumulation(iPxPos, fAccumulatedFramesToStore);

    return fAccumulation;
}

fn ComputeShadingChange(fUv: vec2<f32>) -> f32 {
    let fShadingChangeUv = ClampUv(fUv - Jitter() / vec2<f32>(RenderSize()), ShadingChangeRenderSize(), ShadingChangeMaxRenderSize());
    return ffxSaturate(SampleShadingChange(fShadingChangeUv));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iPxPos = vec2<i32>(gid.xy);
    if (!IsOnScreen(iPxPos, RenderSize())) {
        return;
    }

    let fUv = (vec2<f32>(iPxPos) + 0.5) / vec2<f32>(RenderSize());
    let fMotionVector = LoadDilatedMotionVector(iPxPos);

    let fDilatedDepth = LoadDilatedDepth(iPxPos);

    let fDisocclusion = ComputeDisocclusions(fUv, fMotionVector, GetViewSpaceDepth(fDilatedDepth));
    let fShadingChange = max(DilateReactiveMasks(iPxPos, fUv), ComputeShadingChange(fUv));

    let fMotionDivergence = ComputeMotionDivergence(fUv, fMotionVector, fDilatedDepth);
    let fDilatedTransparencyAndComposition = DilateTransparencyAndCompositionMasks(iPxPos, fUv);
    let fFinalReactiveness = max(fMotionDivergence, fDilatedTransparencyAndComposition);

    let fAccumulation = UpdateAccumulation(iPxPos, fUv, fMotionVector, fDisocclusion, fShadingChange);

    var fOutput: vec4<f32>;
    fOutput[REACTIVE] = fFinalReactiveness;
    fOutput[DISOCCLUSION] = fDisocclusion;
    fOutput[SHADING_CHANGE] = fShadingChange;
    fOutput[ACCUMULATION] = fAccumulation;

    StoreDilatedReactiveMasks(iPxPos, fOutput);

    let fLockStrength = ComputeThinFeatureConfidence(iPxPos);
    if (fLockStrength > (1.0 / 100.0)) {
        StoreNewLocks(ComputeHrPosFromLrPos(iPxPos), fLockStrength);
    }
}
