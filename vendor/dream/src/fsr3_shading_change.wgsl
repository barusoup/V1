// FSR 3.1 pass: shading-change pyramid level 0 (ffx_fsr3upscaler_shading_change_pyramid.h).
//
// AMD builds this with the single-pass downsampler; level 0 is a plain 2x2 box
// average of the full-resolution signed-luma-difference image, so it is
// evaluated here directly at half render resolution. Channel r holds the
// difference, channel g its sign (the SDK stores and averages both).

struct SampleSet {
    samples: array<f32, 5>,
}

fn CompareSwap(s: ptr<function, SampleSet>, i: i32, j: i32) {
    let lo = min((*s).samples[i], (*s).samples[j]);
    let hi = max((*s).samples[i], (*s).samples[j]);
    (*s).samples[i] = lo;
    (*s).samples[j] = hi;
}

fn SortSet(s: ptr<function, SampleSet>) {
    CompareSwap(s, 0, 3);
    CompareSwap(s, 1, 4);
    CompareSwap(s, 0, 2);
    CompareSwap(s, 1, 3);
    CompareSwap(s, 0, 1);
    CompareSwap(s, 2, 4);
    CompareSwap(s, 1, 2);
    CompareSwap(s, 3, 4);
    CompareSwap(s, 2, 3);
}

fn ComputeMinimumDifference(fSet0: ptr<function, SampleSet>, fSet1: ptr<function, SampleSet>) -> f32 {
    var fMinDiff = FSR3UPSCALER_FP16_MAX - 1.0;
    var a = 0;
    var b = 0;

    SortSet(fSet0);
    SortSet(fSet1);

    let fMax = min((*fSet0).samples[4], (*fSet1).samples[4]);

    if (fMax > FSR3UPSCALER_FP32_MIN) {
        for (var i = 0; i < 5 && fMinDiff < FSR3UPSCALER_FP16_MAX; i++) {
            var fDiff = (*fSet0).samples[a] - (*fSet1).samples[b];
            if (abs(fDiff) > FSR3UPSCALER_FP16_MIN) {
                fDiff = sign(fDiff) * (1.0 - MinDividedByMax((*fSet0).samples[a], (*fSet1).samples[b]));
                fMinDiff = select(fMinDiff, fDiff, abs(fDiff) < abs(fMinDiff));
                a += i32((*fSet0).samples[a] < (*fSet1).samples[b]);
                b += i32((*fSet0).samples[a] >= (*fSet1).samples[b]);
            } else {
                fMinDiff = FSR3UPSCALER_FP16_MAX;
            }
        }
    }

    return fMinDiff * select(0.0, 1.0, fMinDiff < (FSR3UPSCALER_FP16_MAX - 1.0));
}

fn GetCurrentLumaBilinearSamples(fUv: vec2<f32>) -> SampleSet {
    let fUvJittered = fUv + Jitter() / vec2<f32>(RenderSize());
    let iBasePos = vec2<i32>(floor(fUvJittered * vec2<f32>(RenderSize())));
    let iSampleOffsets = array<vec2<i32>, 5>(
        vec2<i32>(0, 0), vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1));

    var fSet: SampleSet;
    for (var i = 0; i < 5; i++) {
        let iSamplePos = ClampLoad(iBasePos, iSampleOffsets[i], RenderSize());
        var v = LoadCurrentLuma(iSamplePos) * Exposure();
        v = pow(v, fShadingChangeSamplePow);
        fSet.samples[i] = max(v, FSR3UPSCALER_EPSILON);
    }
    return fSet;
}

struct PreviousLumaSamples {
    samples_set: SampleSet,
    is_existing: bool,
}

fn GetPreviousLumaBilinearSamples(fUv: vec2<f32>, fMotionVector: vec2<f32>) -> PreviousLumaSamples {
    var data: PreviousLumaSamples;
    data.is_existing = false;
    let fUvJittered = fUv + PreviousFrameJitter() / vec2<f32>(PreviousFrameRenderSize());
    let fReprojectedUv = fUvJittered + fMotionVector;
    data.is_existing = IsUvInside(fReprojectedUv);

    if (data.is_existing) {
        let iBasePos = vec2<i32>(floor(fReprojectedUv * vec2<f32>(PreviousFrameRenderSize())));
        let iSampleOffsets = array<vec2<i32>, 5>(
            vec2<i32>(0, 0), vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1));
        for (var i = 0; i < 5; i++) {
            let iSamplePos = ClampLoad(iBasePos, iSampleOffsets[i], PreviousFrameRenderSize());
            var v = LoadPreviousLuma(iSamplePos) * DeltaPreExposure() * Exposure();
            v = pow(v, fShadingChangeSamplePow);
            data.samples_set.samples[i] = max(v, FSR3UPSCALER_EPSILON);
        }
    }
    return data;
}

fn ComputeDiff(fUv: vec2<f32>, fMotionVector: vec2<f32>) -> f32 {
    var fMinDiff = 0.0;
    var fCurrentSamples = GetCurrentLumaBilinearSamples(fUv);
    let previousData = GetPreviousLumaBilinearSamples(fUv, fMotionVector);
    if (previousData.is_existing) {
        var prevSet = previousData.samples_set;
        fMinDiff = ComputeMinimumDifference(&fCurrentSamples, &prevSet);
    }
    return fMinDiff;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iPxMip0 = vec2<i32>(gid.xy);
    let mip0Size = ShadingChangeRenderSize();
    if (!IsOnScreen(iPxMip0, mip0Size)) {
        return;
    }

    var fSumDiff = 0.0;
    var fSumSign = 0.0;
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let iPxSamplePos = iPxMip0 * 2 + vec2<i32>(dx, dy);
            let fDilatedMotionVector = LoadDilatedMotionVector(iPxSamplePos);
            let fUv = (vec2<f32>(iPxSamplePos) + 0.5) / vec2<f32>(RenderSize());
            let d = ComputeDiff(fUv, fDilatedMotionVector);
            fSumDiff += d;
            fSumSign += select(0.0, sign(d), d != 0.0);
        }
    }

    StoreSPDMipLevel(0, iPxMip0, vec2<f32>(fSumDiff * 0.25, fSumSign * 0.25));
}
