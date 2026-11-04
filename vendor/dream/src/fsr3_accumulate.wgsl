// FSR 3.1 pass: accumulate (ffx_fsr3upscaler_accumulate.h, plus the upsample,
// sample and reproject helpers it pulls in).
//
// This is the core temporal upscale: a sliced Lanczos temporal filter with a
// variance clipping box (rectification), pixel locking and reactive-mask
// aware history rejection.

struct AccumulationPassCommonParams {
    iPxHrPos: vec2<i32>,
    fHrUv: vec2<f32>,
    fLrUvJittered: vec2<f32>,
    fLrUv_HwSampler: vec2<f32>,
    fMotionVector: vec2<f32>,
    fReprojectedHrUv: vec2<f32>,
    f4KVelocity: f32,
    fDisocclusion: f32,
    fReactiveMask: f32,
    fShadingChange: f32,
    fAccumulation: f32,
    fLumaInstabilityFactor: f32,
    fFarthestDepthInMeters: f32,
    bIsExistingSample: bool,
    bIsNewSample: bool,
}

struct AccumulationPassData {
    clippingBox: RectificationBox,
    fUpsampledColor: vec3<f32>,
    fUpsampledWeight: f32,
    fHistoryColor: vec3<f32>,
    fHistoryWeight: f32,
    fLock: f32,
    fLockContributionThisFrame: f32,
}

// --- sample.h (reference Lanczos-2, no LUT) ---------------------------------
fn ClampCoord(iPxSample: vec2<i32>, iPxOffset: vec2<i32>, iTextureSize: vec2<i32>) -> vec2<i32> {
    var result = iPxSample + iPxOffset;
    result.x = max(1, min(result.x, iTextureSize.x - 2));
    result.y = max(1, min(result.y, iTextureSize.y - 2));
    return result;
}

fn Lanczos2x4(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, t: f32) -> vec4<f32> {
    let w0 = Lanczos2(-1.0 - t);
    let w1 = Lanczos2(0.0 - t);
    let w2 = Lanczos2(1.0 - t);
    let w3 = Lanczos2(2.0 - t);
    return (w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3) / (w0 + w1 + w2 + w3);
}

// Custom bicubic fetch (DeclareCustomFetchBicubicSamples) over LoadHistory.
fn FetchHistorySamples(iPxSample: vec2<i32>, iTextureSize: vec2<i32>, d: ptr<function, array<vec4<f32>, 16>>) {
    for (var row = 0; row < 4; row++) {
        for (var col = 0; col < 4; col++) {
            let off = vec2<i32>(col - 1, row - 1);
            (*d)[row * 4 + col] = LoadHistory(ClampCoord(iPxSample, off, iTextureSize));
        }
    }
}

fn HistorySample(fUvSample: vec2<f32>, iTextureSize: vec2<i32>) -> vec4<f32> {
    let fPxSample = (fUvSample * vec2<f32>(iTextureSize)) - vec2<f32>(0.5, 0.5);
    let fPxFrac = fract(fPxSample);
    var p = fPxSample;
    p.x = max(0.0, min(f32(iTextureSize.x - 1), p.x));
    p.y = max(0.0, min(f32(iTextureSize.y - 1), p.y));
    let iPxSample = vec2<i32>(floor(p));

    var s: array<vec4<f32>, 16>;
    FetchHistorySamples(iPxSample, iTextureSize, &s);

    let x0 = Lanczos2x4(s[0], s[1], s[2], s[3], fPxFrac.x);
    let x1 = Lanczos2x4(s[4], s[5], s[6], s[7], fPxFrac.x);
    let x2 = Lanczos2x4(s[8], s[9], s[10], s[11], fPxFrac.x);
    let x3 = Lanczos2x4(s[12], s[13], s[14], s[15], fPxFrac.x);
    var fColorXY = Lanczos2x4(x0, x1, x2, x3, fPxFrac.y);

    // Deringing over the four centre taps.
    let s11 = s[5];
    let s21 = s[6];
    let s12 = s[9];
    let s22 = s[10];
    let fDeringingMin = min(min(s11, s21), min(s12, s22));
    let fDeringingMax = max(max(s11, s21), max(s12, s22));
    fColorXY = clamp(fColorXY, fDeringingMin, fDeringingMax);

    return fColorXY;
}

// --- reproject.h ------------------------------------------------------------
// Render-resolution motion vectors (the SDK's LOW_RES_MOTION_VECTORS permu-
// tation): read the dilated motion vectors at the render-resolution position of
// this high-resolution pixel. The app omits the display-resolution MV flag, so
// the input MV texture is render-sized and must not be fetched at HR coords.
fn GetMotionVector(iPxHrPos: vec2<i32>, fHrUv: vec2<f32>) -> vec2<f32> {
    _ = iPxHrPos;
    return LoadDilatedMotionVector(vec2<i32>(fHrUv * vec2<f32>(RenderSize())));
}

fn ReprojectHistoryColor(params: AccumulationPassCommonParams, data: ptr<function, AccumulationPassData>) {
    let fReprojectedHistory = HistorySample(params.fReprojectedHrUv, PreviousFrameUpscaleSize());

    (*data).fHistoryColor = fReprojectedHistory.rgb;
    (*data).fHistoryColor *= DeltaPreExposure() * Exposure();
    (*data).fHistoryColor = RGBToYCoCg((*data).fHistoryColor);
    (*data).fLock = fReprojectedHistory.w;
}

// --- upsample.h -------------------------------------------------------------
fn GetUpsampleLanczosWeight(fSrcSampleOffset: vec2<f32>, fKernelWeight: f32) -> f32 {
    let fSrcSampleOffsetBiased = fSrcSampleOffset * fKernelWeight;
    // LANCZOS_TYPE_APPROXIMATE (FSR1 polynomial), biased term is squared.
    return Lanczos2ApproxSq(dot(fSrcSampleOffsetBiased, fSrcSampleOffsetBiased));
}

fn ComputeMaxKernelWeight(params: AccumulationPassCommonParams) -> f32 {
    _ = params;
    let fKernelSizeBias = 1.0 + (1.0 / DownscaleFactor().x - 1.0);
    return min(1.99, fKernelSizeBias);
}

fn LoadPreparedColor(iSamplePos: vec2<i32>) -> vec3<f32> {
    let fRgb = max(vec3<f32>(0.0, 0.0, 0.0), LoadInputColor(iSamplePos)) * Exposure();
    return RGBToYCoCg(fRgb);
}


fn ComputeUpsampledColorAndWeight(params: AccumulationPassCommonParams, data: ptr<function, AccumulationPassData>) {
    // The clipping box is kept in a local and written back at the end: naga's
    // SPIR-V backend cannot codegen `&(*data).member` passed to a function.
    var box = RectificationBox(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
    let fDstOutputPos = vec2<f32>(params.iPxHrPos) + vec2<f32>(0.5, 0.5);
    let fSrcOutputPos = fDstOutputPos * DownscaleFactor();
    let iSrcInputPos = vec2<i32>(floor(fSrcOutputPos));
    let fSrcUnjitteredPos = (vec2<f32>(iSrcInputPos) + vec2<f32>(0.5, 0.5)) - Jitter();
    let fBaseSampleOffset = fSrcUnjitteredPos - fSrcOutputPos;

    var offsetTL: vec2<i32>;
    offsetTL.x = select(-1, -2, fSrcUnjitteredPos.x > fSrcOutputPos.x);
    offsetTL.y = select(-1, -2, fSrcUnjitteredPos.y > fSrcOutputPos.y);

    let bFlipRow = fSrcUnjitteredPos.y > fSrcOutputPos.y;
    let bFlipCol = fSrcUnjitteredPos.x > fSrcOutputPos.x;

    let bIsInitialSample = params.fAccumulation == 0.0;

    var fSamples: array<vec3<f32>, 9>;
    var iSampleIndex = 0;
    for (var row = 0; row < 3; row++) {
        for (var col = 0; col < 3; col++) {
            let iSampleColRow = vec2<i32>(select(col, 3 - col, bFlipCol), select(row, 3 - row, bFlipRow));
            let iSrcSamplePos = iSrcInputPos + offsetTL + iSampleColRow;
            let iSampleCoord = ClampLoad(iSrcSamplePos, vec2<i32>(0, 0), RenderSize());
            fSamples[iSampleIndex] = LoadPreparedColor(iSampleCoord);
            iSampleIndex++;
        }
    }

    let fKernelBiasMax = ComputeMaxKernelWeight(params);
    let fKernelBiasMin = max(1.0, (1.0 + fKernelBiasMax) * 0.3);
    let fKernelBiasWeight = min(
        1.0 - params.fDisocclusion * 0.5,
        min(1.0 - params.fShadingChange, ffxSaturate((*data).fHistoryWeight * 5.0)));
    let fKernelBias = ffxLerp(fKernelBiasMin, fKernelBiasMax, fKernelBiasWeight);

    let fOffsetTL = vec2<f32>(offsetTL);

    var fUpsampledColor = vec3<f32>(0.0, 0.0, 0.0);
    var fUpsampledWeight = 0.0;

    iSampleIndex = 0;
    for (var row = 0; row < 3; row++) {
        for (var col = 0; col < 3; col++) {
            let sampleColRow = vec2<i32>(select(col, 3 - col, bFlipCol), select(row, 3 - row, bFlipRow));
            let fOffset = fOffsetTL + vec2<f32>(sampleColRow);
            let fSrcSampleOffset = fBaseSampleOffset + fOffset;

            let iSrcSamplePos = iSrcInputPos + offsetTL + sampleColRow;
            let fOnScreenFactor = select(0.0, 1.0, IsOnScreen(iSrcSamplePos, RenderSize()));

            if (!bIsInitialSample) {
                let fSampleWeight = fOnScreenFactor * GetUpsampleLanczosWeight(fSrcSampleOffset, fKernelBias);
                fUpsampledColor += fSamples[iSampleIndex] * fSampleWeight;
                fUpsampledWeight += fSampleWeight;
            }

            {
                let fRectificationCurveBias = -2.3;
                let fSrcSampleOffsetSq = dot(fSrcSampleOffset, fSrcSampleOffset);
                let fBoxSampleWeight = exp(fRectificationCurveBias * fSrcSampleOffsetSq) * fOnScreenFactor;
                let bInitialSample = (row == 0) && (col == 0);
                RectificationBoxAddSample(bInitialSample, &box, fSamples[iSampleIndex], fBoxSampleWeight);
            }
            iSampleIndex++;
        }
    }

    RectificationBoxComputeVarianceBoxData(&box);

    fUpsampledWeight *= select(0.0, 1.0, fUpsampledWeight > FSR3UPSCALER_EPSILON);

    if (fUpsampledWeight > FSR3UPSCALER_EPSILON) {
        fUpsampledColor = fUpsampledColor / fUpsampledWeight;
        fUpsampledWeight *= fAverageLanczosWeightPerFrame;
        fUpsampledColor = clamp(fUpsampledColor, box.aabb_min, box.aabb_max);
    }

    if (bIsInitialSample) {
        fUpsampledColor = box.box_center;
        fUpsampledWeight = 1.0;
        (*data).fHistoryWeight = 0.0;
    }

    (*data).fUpsampledColor = fUpsampledColor;
    (*data).fUpsampledWeight = fUpsampledWeight;
    (*data).clippingBox = box;
}

// --- accumulate.h -----------------------------------------------------------
fn Accumulate(params: AccumulationPassCommonParams, data: ptr<function, AccumulationPassData>) {
    _ = params;
    (*data).fHistoryWeight *= select(0.0, 1.0, (*data).fHistoryWeight > FSR3UPSCALER_FP16_MIN);
    (*data).fHistoryWeight = max(FSR3UPSCALER_EPSILON, (*data).fHistoryWeight + (*data).fUpsampledWeight);

    let fAlpha = ffxSaturate((*data).fUpsampledWeight / (*data).fHistoryWeight);
    (*data).fHistoryColor = ffxLerp3((*data).fHistoryColor, (*data).fUpsampledColor, fAlpha);
    (*data).fHistoryColor = YCoCgToRGB((*data).fHistoryColor);
}

fn RectifyHistory(params: AccumulationPassCommonParams, data: ptr<function, AccumulationPassData>) {
    let f4kVelocityFactor = ffxSaturate(params.f4KVelocity / 20.0);
    let fDistanceFactor = ffxSaturate(0.75 - params.fFarthestDepthInMeters / 20.0);
    let fAccumulationFactor = 1.0 - params.fAccumulation;
    let fReactiveFactor = pow(params.fReactiveMask, 1.0 / 2.0);
    let fShadingChangeFactor = params.fShadingChange;
    let fBoxScaleT = max(f4kVelocityFactor, max(fDistanceFactor, max(fAccumulationFactor, max(fReactiveFactor, fShadingChangeFactor))));

    let fBoxScale = ffxLerp(3.0, 1.0, fBoxScaleT);
    let fScaledBoxVec = (*data).clippingBox.box_vec * vec3<f32>(1.7, 1.0, 1.0) * fBoxScale;

    let fClampedScaledBoxVec = max(fScaledBoxVec, vec3<f32>(1.193e-7, 1.193e-7, 1.193e-7));
    let fTransformedHistoryColor = ((*data).fHistoryColor - (*data).clippingBox.box_center) / fClampedScaledBoxVec;

    if (length(fTransformedHistoryColor) > 1.0) {
        let fClampedHistoryColor = normalize(fTransformedHistoryColor);
        let fFinalClampedHistoryColor = (fClampedHistoryColor * fScaledBoxVec) + (*data).clippingBox.box_center;

        let fHistoryContribution = max(params.fLumaInstabilityFactor, (*data).fLockContributionThisFrame) * params.fAccumulation * (1.0 - params.fDisocclusion);
        (*data).fHistoryColor = ffxLerp3(fFinalClampedHistoryColor, (*data).fHistoryColor, ffxSaturate(fHistoryContribution));
    }
}

fn UpdateLockStatus(params: AccumulationPassCommonParams, data: ptr<function, AccumulationPassData>) {
    (*data).fLock *= select(0.0, 1.0, params.bIsNewSample == false);

    let fLifetimeDecreaseFactor = max(ffxSaturate(params.fShadingChange), max(params.fReactiveMask, params.fDisocclusion));
    (*data).fLock = max(0.0, (*data).fLock - fLifetimeDecreaseFactor * fLockMax);

    (*data).fLockContributionThisFrame = ffxSaturate(ffxSaturate((*data).fLock - fLockThreshold) * (fLockMax - fLockThreshold));

    let fNewLockIntensity = LoadNewLocks(params.iPxHrPos) * (1.0 - max(params.fShadingChange * 0.0, params.fReactiveMask));
    (*data).fLock = max(0.0, min((*data).fLock + fNewLockIntensity, fLockMax));

    let fLifetimeDecrease = (0.1 / JitterSequenceLength()) * (1.0 - fLifetimeDecreaseFactor);
    (*data).fLock = max(0.0, (*data).fLock - fLifetimeDecrease);

    let fEstimatedUvNextFrame = params.fHrUv - params.fMotionVector;
    (*data).fLock *= select(0.0, 1.0, IsUvInside(fEstimatedUvNextFrame) == true);
}

fn ComputeBaseAccumulationWeight(params: AccumulationPassCommonParams, data: ptr<function, AccumulationPassData>) {
    var fBaseAccumulation = params.fAccumulation;
    fBaseAccumulation = min(fBaseAccumulation, ffxLerp(fBaseAccumulation, 0.15, ffxSaturate(max(0.0, (params.f4KVelocity * VelocityFactor()) / 0.5))));
    (*data).fHistoryWeight = fBaseAccumulation;
}

fn InitPassData(iPxHrPos: vec2<i32>, params: ptr<function, AccumulationPassCommonParams>, data: ptr<function, AccumulationPassData>) {
    (*params).iPxHrPos = iPxHrPos;
    let fHrUv = (vec2<f32>(iPxHrPos) + 0.5) / vec2<f32>(UpscaleSize());
    (*params).fHrUv = fHrUv;
    (*params).fLrUvJittered = fHrUv + Jitter() / vec2<f32>(RenderSize());
    (*params).fLrUv_HwSampler = ClampUv((*params).fLrUvJittered, RenderSize(), MaxRenderSize());

    (*params).fMotionVector = GetMotionVector(iPxHrPos, fHrUv);
    (*params).f4KVelocity = Get4KVelocity((*params).fMotionVector);

    (*params).fReprojectedHrUv = (*params).fHrUv + (*params).fMotionVector;
    (*params).bIsExistingSample = IsUvInside((*params).fReprojectedHrUv);

    let fLumaInstabilityUv_HW = ClampUv(fHrUv, RenderSize(), MaxRenderSize());
    (*params).fLumaInstabilityFactor = SampleLumaInstability(fLumaInstabilityUv_HW);

    let fFarthestDepthUv = ClampUv((*params).fLrUvJittered, RenderSize() / 2, RenderSize() / 2);
    (*params).fFarthestDepthInMeters = SampleFarthestDepthMip1(fFarthestDepthUv);
    (*params).bIsNewSample = ((*params).bIsExistingSample == false) || (0.0 == FrameIndex());

    let fDilatedReactiveMasks = SampleDilatedReactiveMasks((*params).fLrUv_HwSampler);
    (*params).fReactiveMask = ffxSaturate(fDilatedReactiveMasks[REACTIVE]);
    (*params).fDisocclusion = ffxSaturate(fDilatedReactiveMasks[DISOCCLUSION]);
    (*params).fShadingChange = ffxSaturate(fDilatedReactiveMasks[SHADING_CHANGE]);
    (*params).fAccumulation = ffxSaturate(fDilatedReactiveMasks[ACCUMULATION]);
    (*params).fAccumulation *= select(0.0, 1.0, round((*params).fAccumulation * 100.0) > 1.0);

    (*data).fUpsampledColor = vec3<f32>(0.0, 0.0, 0.0);
    (*data).fHistoryColor = vec3<f32>(0.0, 0.0, 0.0);
    (*data).fHistoryWeight = 1.0;
    (*data).fUpsampledWeight = 0.0;
    (*data).fLock = 0.0;
    (*data).fLockContributionThisFrame = 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iPxHrPos = vec2<i32>(gid.xy);
    if (!IsOnScreen(iPxHrPos, UpscaleSize())) {
        return;
    }

    var params: AccumulationPassCommonParams;
    var data: AccumulationPassData;
    InitPassData(iPxHrPos, &params, &data);

    if (params.bIsExistingSample && !params.bIsNewSample) {
        ReprojectHistoryColor(params, &data);
    }

    UpdateLockStatus(params, &data);
    ComputeBaseAccumulationWeight(params, &data);
    ComputeUpsampledColorAndWeight(params, &data);
    RectifyHistory(params, &data);
    Accumulate(params, &data);

    data.fHistoryColor /= Exposure();
    data.fHistoryColor = max(data.fHistoryColor, vec3<f32>(0.0, 0.0, 0.0));

    StoreInternalColorAndWeight(iPxHrPos, vec4<f32>(data.fHistoryColor, data.fLock));
    StoreNewLocks(iPxHrPos, 0.0);
}
