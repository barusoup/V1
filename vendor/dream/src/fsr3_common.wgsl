// FSR 3.1 upscaler - shared prelude.
//
// This is a hand port of AMD's FidelityFX Super Resolution 3.1 temporal
// upscaler GPU code (MIT licensed, FidelityFX SDK 1.1.4) to WGSL for Dream's
// native Vulkan renderer. It is our own implementation of AMD's algorithm, not
// AMD's shipped shaders. Deviations from the reference are noted inline; the
// notable ones are:
//   * auto exposure / the SPD luminance pyramid is omitted (we feed a fixed
//     exposure of 1 and compute the farthest-depth and shading-change pyramids
//     with plain 2x2 box downsampling passes).
//   * the reconstructed-previous-nearest-depth surface is a storage buffer of
//     packed f32 bit patterns rather than an r32ui image, because WGSL exposes
//     no image atomics.
// Every pass in this directory is concatenated after this prelude and shares
// one descriptor set layout, so unused bindings are simply dead in a given pass.

struct Fsr3Constants {
    render_size: vec2<i32>,
    previous_frame_render_size: vec2<i32>,
    upscale_size: vec2<i32>,
    previous_frame_upscale_size: vec2<i32>,
    max_render_size: vec2<i32>,
    max_upscale_size: vec2<i32>,
    device_to_view_depth: vec4<f32>,
    jitter_offset: vec2<f32>,
    previous_frame_jitter_offset: vec2<f32>,
    motion_vector_scale: vec2<f32>,
    downscale_factor: vec2<f32>,
    motion_vector_jitter_cancellation: vec2<f32>,
    tan_half_fov: f32,
    jitter_phase_count: f32,
    delta_time: f32,
    delta_pre_exposure: f32,
    view_space_to_meters_factor: f32,
    frame_index: f32,
    velocity_factor: f32,
    reactiveness_scale: f32,
    shading_change_scale: f32,
    accumulation_added_per_frame: f32,
    min_disocclusion_accumulation: f32,
    _padding: vec3<f32>,
};

@group(0) @binding(0) var<uniform> cb: Fsr3Constants;
// RCAS config: FsrRcasCon output, as four u32 bit patterns.
@group(0) @binding(1) var<uniform> rcas_config: vec4<u32>;
@group(0) @binding(2) var sp_point: sampler;
@group(0) @binding(3) var sp_linear: sampler;

@group(0) @binding(10) var t_input_color: texture_2d<f32>;
@group(0) @binding(11) var t_input_depth: texture_depth_2d;
@group(0) @binding(12) var t_input_motion_vectors: texture_2d<f32>;
@group(0) @binding(13) var t_dilated_motion_vectors: texture_2d<f32>;
@group(0) @binding(14) var t_dilated_depth: texture_2d<f32>;
@group(0) @binding(15) var t_farthest_depth_mip1: texture_2d<f32>;
@group(0) @binding(16) var t_current_luma: texture_2d<f32>;
@group(0) @binding(17) var t_previous_luma: texture_2d<f32>;
@group(0) @binding(18) var t_luma_history: texture_2d<f32>;
@group(0) @binding(19) var t_luma_instability: texture_2d<f32>;
@group(0) @binding(20) var t_shading_change: texture_2d<f32>;
@group(0) @binding(21) var t_spd_mip0: texture_2d<f32>;
@group(0) @binding(22) var t_spd_mip1: texture_2d<f32>;
@group(0) @binding(23) var t_spd_mip2: texture_2d<f32>;
@group(0) @binding(24) var t_dilated_reactive_masks: texture_2d<f32>;
@group(0) @binding(25) var t_accumulation: texture_2d<f32>;
@group(0) @binding(26) var t_internal_upscaled_color: texture_2d<f32>;
@group(0) @binding(27) var t_new_locks: texture_2d<f32>;
@group(0) @binding(28) var t_rcas_input: texture_2d<f32>;

@group(0) @binding(30) var<storage, read> reconstructed_prev_depth: array<u32>;
@group(0) @binding(31) var<storage, read_write> reconstructed_prev_depth_rw: array<atomic<u32>>;

@group(0) @binding(40) var w_dilated_motion_vectors: texture_storage_2d<rg16float, write>;
@group(0) @binding(41) var w_dilated_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(42) var w_farthest_depth: texture_storage_2d<r16float, write>;
@group(0) @binding(43) var w_farthest_depth_mip1: texture_storage_2d<r16float, write>;
@group(0) @binding(44) var w_current_luma: texture_storage_2d<r16float, write>;
@group(0) @binding(45) var w_luma_history: texture_storage_2d<rgba16float, write>;
@group(0) @binding(46) var w_luma_instability: texture_storage_2d<r16float, write>;
@group(0) @binding(47) var w_shading_change: texture_storage_2d<r8unorm, write>;
@group(0) @binding(48) var w_spd_mip0: texture_storage_2d<rg16float, write>;
@group(0) @binding(49) var w_spd_mip1: texture_storage_2d<rg16float, write>;
@group(0) @binding(50) var w_spd_mip2: texture_storage_2d<rg16float, write>;
@group(0) @binding(51) var w_dilated_reactive_masks: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(52) var w_accumulation: texture_storage_2d<r8unorm, write>;
@group(0) @binding(53) var w_internal_upscaled_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(54) var w_new_locks: texture_storage_2d<r8unorm, write>;
@group(0) @binding(55) var w_upscaled_output: texture_storage_2d<rgba16float, write>;

// Render "downsample" planes (used by the box pyramids) are bound at 60+.
@group(0) @binding(60) var t_downsample_src: texture_2d<f32>;
@group(0) @binding(61) var t_downsample_src_depth: texture_2d<f32>;
@group(0) @binding(62) var w_downsample_dst: texture_storage_2d<rg16float, write>;
@group(0) @binding(63) var w_downsample_dst_r: texture_storage_2d<r16float, write>;

// ---------------------------------------------------------------------------
// Constants (ffx_fsr3upscaler_common.h)
// ---------------------------------------------------------------------------
const FSR3UPSCALER_FP16_MIN: f32 = 6.1e-05;
const FSR3UPSCALER_FP16_MAX: f32 = 65504.0;
const FSR3UPSCALER_EPSILON: f32 = 6.1e-05;
const FSR3UPSCALER_TONEMAP_EPSILON: f32 = 6.1e-05;
const FSR3UPSCALER_FP32_MAX: f32 = 3.402823466e+38;
const FSR3UPSCALER_FP32_MIN: f32 = 1.175494351e-38;

const fReconstructedDepthBilinearWeightThreshold: f32 = FSR3UPSCALER_EPSILON * 10.0;

const fUpsampleLanczosWeightScale: f32 = 1.0 / 16.0;
const fAverageLanczosWeightPerFrame: f32 = 0.74 * fUpsampleLanczosWeightScale;

const SS: i32 = 5; // SHADING_CHANGE_SET_SIZE
const iShadingChangeMipStart: i32 = 0;
const fShadingChangeSamplePow: f32 = 1.0;

const fLockThreshold: f32 = 1.0;
const fLockMax: f32 = 2.0;

const REACTIVE: i32 = 0;
const DISOCCLUSION: i32 = 1;
const SHADING_CHANGE: i32 = 2;
const ACCUMULATION: i32 = 3;

fn ffxSaturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn ffxLerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn ffxLerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn ReconstructedDepthMvPxThreshold(fNearestDepthInMeters: f32) -> f32 {
    return ffxLerp(0.25, 0.75, ffxSaturate(fNearestDepthInMeters / 100.0));
}

fn Get4KVelocity(fMotionVector: vec2<f32>) -> f32 {
    return length(fMotionVector * vec2<f32>(3840.0, 2160.0));
}

fn RenderSize() -> vec2<i32> { return cb.render_size; }
fn PreviousFrameRenderSize() -> vec2<i32> { return cb.previous_frame_render_size; }
fn MaxRenderSize() -> vec2<i32> { return cb.max_render_size; }
fn UpscaleSize() -> vec2<i32> { return cb.upscale_size; }
fn PreviousFrameUpscaleSize() -> vec2<i32> { return cb.previous_frame_upscale_size; }
fn MaxUpscaleSize() -> vec2<i32> { return cb.max_upscale_size; }
fn Jitter() -> vec2<f32> { return cb.jitter_offset; }
fn PreviousFrameJitter() -> vec2<f32> { return cb.previous_frame_jitter_offset; }
fn DeviceToViewSpaceTransformFactors() -> vec4<f32> { return cb.device_to_view_depth; }
fn MotionVectorScale() -> vec2<f32> { return cb.motion_vector_scale; }
fn DownscaleFactor() -> vec2<f32> { return cb.downscale_factor; }
fn MotionVectorJitterCancellation() -> vec2<f32> { return cb.motion_vector_jitter_cancellation; }
fn TanHalfFoV() -> f32 { return cb.tan_half_fov; }
fn JitterSequenceLength() -> f32 { return cb.jitter_phase_count; }
fn DeltaTime() -> f32 { return cb.delta_time; }
fn DeltaPreExposure() -> f32 { return cb.delta_pre_exposure; }
fn ViewSpaceToMetersFactor() -> f32 { return cb.view_space_to_meters_factor; }
fn FrameIndex() -> f32 { return cb.frame_index; }
fn VelocityFactor() -> f32 { return cb.velocity_factor; }
fn AccumulationAddedPerFrame() -> f32 { return cb.accumulation_added_per_frame; }
fn MinDisocclusionAccumulation() -> f32 { return cb.min_disocclusion_accumulation; }
fn RCASConfig() -> vec4<u32> { return rcas_config; }

// No exposure input: FSR treats a zero exposure as "1".
fn Exposure() -> f32 { return 1.0; }

fn ShadingChangeRenderSize() -> vec2<i32> {
    return vec2<i32>(vec2<f32>(RenderSize()) * 0.5);
}
fn ShadingChangeMaxRenderSize() -> vec2<i32> {
    return vec2<i32>(vec2<f32>(MaxRenderSize()) * 0.5);
}
fn PreviousFrameShadingChangeRenderSize() -> vec2<i32> {
    return vec2<i32>(vec2<f32>(PreviousFrameRenderSize()) * 0.5);
}

// ---------------------------------------------------------------------------
// Math helpers (ffx_fsr3upscaler_common.h)
// ---------------------------------------------------------------------------
struct RectificationBox {
    box_center: vec3<f32>,
    box_vec: vec3<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    box_center_weight: f32,
}

fn RectificationBoxAddInitialSample(box: ptr<function, RectificationBox>, colorSample: vec3<f32>, fSampleWeight: f32) {
    (*box).aabb_min = colorSample;
    (*box).aabb_max = colorSample;
    let weightedSample = colorSample * fSampleWeight;
    (*box).box_center = weightedSample;
    (*box).box_vec = colorSample * weightedSample;
    (*box).box_center_weight = fSampleWeight;
}

fn RectificationBoxAddSample(bInitialSample: bool, box: ptr<function, RectificationBox>, colorSample: vec3<f32>, fSampleWeight: f32) {
    if (bInitialSample) {
        RectificationBoxAddInitialSample(box, colorSample, fSampleWeight);
    } else {
        (*box).aabb_min = min((*box).aabb_min, colorSample);
        (*box).aabb_max = max((*box).aabb_max, colorSample);
        let weightedSample = colorSample * fSampleWeight;
        (*box).box_center += weightedSample;
        (*box).box_vec += colorSample * weightedSample;
        (*box).box_center_weight += fSampleWeight;
    }
}

fn RectificationBoxComputeVarianceBoxData(box: ptr<function, RectificationBox>) {
    (*box).box_center_weight = select(1.0, (*box).box_center_weight, abs((*box).box_center_weight) > FSR3UPSCALER_FP32_MIN);
    (*box).box_center /= (*box).box_center_weight;
    (*box).box_vec /= (*box).box_center_weight;
    let stdDev = sqrt(abs((*box).box_vec - (*box).box_center * (*box).box_center));
    (*box).box_vec = stdDev;
}

fn MinDividedByMax(v0: f32, v1: f32) -> f32 {
    let m = max(v0, v1);
    return select(0.0, min(v0, v1) / m, m != 0.0);
}

fn MinDividedByMaxOr(v0: f32, v1: f32, fOnZeroReturnValue: f32) -> f32 {
    let m = max(v0, v1);
    return select(fOnZeroReturnValue, min(v0, v1) / m, m != 0.0);
}

fn YCoCgToRGB(fYCoCg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        fYCoCg.x + fYCoCg.y - fYCoCg.z,
        fYCoCg.x + fYCoCg.z,
        fYCoCg.x - fYCoCg.y - fYCoCg.z);
}

fn RGBToYCoCg(fRgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * fRgb.r + 0.5 * fRgb.g + 0.25 * fRgb.b,
        0.5 * fRgb.r - 0.5 * fRgb.b,
        -0.25 * fRgb.r + 0.5 * fRgb.g - 0.25 * fRgb.b);
}

fn RGBToLuma(fLinearRgb: vec3<f32>) -> f32 {
    return dot(fLinearRgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn Tonemap(fRgb: vec3<f32>) -> vec3<f32> {
    let m = max(max(0.0, fRgb.r), max(fRgb.g, fRgb.b)) + 1.0;
    return fRgb / m;
}

fn InverseTonemap(fRgb: vec3<f32>) -> vec3<f32> {
    let d = max(FSR3UPSCALER_TONEMAP_EPSILON, 1.0 - max(fRgb.r, max(fRgb.g, fRgb.b)));
    return fRgb / d;
}

fn IsUvInside(fUv: vec2<f32>) -> bool {
    return (fUv.x >= 0.0 && fUv.x <= 1.0) && (fUv.y >= 0.0 && fUv.y <= 1.0);
}

fn ClampLoad(iPxSample: vec2<i32>, iPxOffset: vec2<i32>, iTextureSize: vec2<i32>) -> vec2<i32> {
    var result = iPxSample + iPxOffset;
    result.x = max(0, min(result.x, iTextureSize.x - 1));
    result.y = max(0, min(result.y, iTextureSize.y - 1));
    return result;
}

fn ClampUv(fUv: vec2<f32>, iTextureSize: vec2<i32>, iResourceSize: vec2<i32>) -> vec2<f32> {
    let fSampleLocation = fUv * vec2<f32>(iTextureSize);
    let fClampedLocation = max(vec2<f32>(0.5, 0.5), min(fSampleLocation, vec2<f32>(iTextureSize) - vec2<f32>(0.5, 0.5)));
    return fClampedLocation / vec2<f32>(iResourceSize);
}

fn IsOnScreen(pos: vec2<i32>, size: vec2<i32>) -> bool {
    return pos.x < size.x && pos.y < size.y && pos.x >= 0 && pos.y >= 0;
}

fn ComputeHrPosFromLrPos(iPxLrPos: vec2<i32>) -> vec2<i32> {
    let fSrcJitteredPos = vec2<f32>(iPxLrPos) + vec2<f32>(0.5, 0.5) - Jitter();
    let fLrPosInHr = (fSrcJitteredPos / vec2<f32>(RenderSize())) * vec2<f32>(UpscaleSize());
    return vec2<i32>(floor(fLrPosInHr));
}

fn GetViewSpaceDepth(fDeviceDepth: f32) -> f32 {
    let f = DeviceToViewSpaceTransformFactors();
    return f.y / (fDeviceDepth - f.x);
}

fn GetViewSpaceDepthInMeters(fDeviceDepth: f32) -> f32 {
    return GetViewSpaceDepth(fDeviceDepth) * ViewSpaceToMetersFactor();
}

struct BilinearSamplingData {
    offsets: array<vec2<i32>, 4>,
    weights: array<f32, 4>,
    base_pos: vec2<i32>,
}

fn GetBilinearSamplingData(fUv: vec2<f32>, iSize: vec2<i32>) -> BilinearSamplingData {
    var data: BilinearSamplingData;
    let fPxSample = (fUv * vec2<f32>(iSize)) - vec2<f32>(0.5, 0.5);
    data.base_pos = vec2<i32>(floor(fPxSample));
    let fPxFrac = fract(fPxSample);
    data.offsets[0] = vec2<i32>(0, 0);
    data.offsets[1] = vec2<i32>(1, 0);
    data.offsets[2] = vec2<i32>(0, 1);
    data.offsets[3] = vec2<i32>(1, 1);
    data.weights[0] = (1.0 - fPxFrac.x) * (1.0 - fPxFrac.y);
    data.weights[1] = fPxFrac.x * (1.0 - fPxFrac.y);
    data.weights[2] = (1.0 - fPxFrac.x) * fPxFrac.y;
    data.weights[3] = fPxFrac.x * fPxFrac.y;
    return data;
}

// ---------------------------------------------------------------------------
// Lanczos-2 (ffx_fsr3upscaler_sample.h). We use the "reference" analytic kernel
// (no 128-entry LUT) as the SDK does when the LUT path is off.
// ---------------------------------------------------------------------------
fn Lanczos2NoClamp(x: f32) -> f32 {
    let PI = 3.141592653589793;
    if abs(x) < FSR3UPSCALER_EPSILON {
        return 1.0;
    }
    return (sin(PI * x) / (PI * x)) * (sin(0.5 * PI * x) / (0.5 * PI * x));
}

fn Lanczos2(x: f32) -> f32 {
    return Lanczos2NoClamp(min(abs(x), 2.0));
}

// FSR1's polynomial approximation of lanczos2. Input is x*x and must be <= 4.
fn Lanczos2ApproxSqNoClamp(x2: f32) -> f32 {
    let a = (2.0 / 5.0) * x2 - 1.0;
    let b = (1.0 / 4.0) * x2 - 1.0;
    return ((25.0 / 16.0) * a * a - (25.0 / 16.0 - 1.0)) * (b * b);
}

fn Lanczos2ApproxSq(x2: f32) -> f32 {
    return Lanczos2ApproxSqNoClamp(min(x2, 4.0));
}

fn Lanczos2Approx(x: f32) -> f32 {
    return Lanczos2ApproxSq(x * x);
}

// ---------------------------------------------------------------------------
// Resource accessors (ffx_fsr3upscaler_callbacks_glsl.h)
// ---------------------------------------------------------------------------
fn LoadInputDepth(iPxPos: vec2<i32>) -> f32 {
    return textureLoad(t_input_depth, iPxPos, 0);
}

fn LoadInputColor(iPxPos: vec2<i32>) -> vec3<f32> {
    return textureLoad(t_input_color, iPxPos, 0).rgb;
}

fn SampleInputColor(fUV: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(t_input_color, sp_linear, fUV, 0.0).rgb;
}

fn LoadInputMotionVector(iPxPos: vec2<i32>) -> vec2<f32> {
    let fSrcMotionVector = textureLoad(t_input_motion_vectors, iPxPos, 0).xy;
    return fSrcMotionVector * MotionVectorScale() - MotionVectorJitterCancellation();
}

fn LoadHistory(iPxHistory: vec2<i32>) -> vec4<f32> {
    return textureLoad(t_internal_upscaled_color, iPxHistory, 0);
}

fn SampleHistory(fUV: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t_internal_upscaled_color, sp_linear, fUV, 0.0);
}

fn StoreLumaHistory(iPxPos: vec2<i32>, fLumaHistory: vec4<f32>) {
    textureStore(w_luma_history, iPxPos, fLumaHistory);
}

fn LoadLumaHistory(iPxPos: vec2<i32>) -> vec4<f32> {
    return textureLoad(t_luma_history, iPxPos, 0);
}

fn SampleLumaHistory(fUV: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t_luma_history, sp_linear, fUV, 0.0);
}

fn LoadRCAS_Input(iPxPos: vec2<i32>) -> vec4<f32> {
    return textureLoad(t_rcas_input, iPxPos, 0);
}

fn StoreInternalColorAndWeight(iPxPos: vec2<i32>, fColorAndWeight: vec4<f32>) {
    textureStore(w_internal_upscaled_color, iPxPos, fColorAndWeight);
}

fn StoreUpscaledOutput(iPxPos: vec2<i32>, fColor: vec3<f32>) {
    textureStore(w_upscaled_output, iPxPos, vec4<f32>(fColor, 1.0));
}

fn SampleAccumulation(fUV: vec2<f32>) -> f32 {
    return textureSampleLevel(t_accumulation, sp_linear, fUV, 0.0).x;
}

fn StoreAccumulation(iPxPos: vec2<i32>, fAccumulation: f32) {
    textureStore(w_accumulation, iPxPos, vec4<f32>(fAccumulation, 0.0, 0.0, 0.0));
}

fn LoadShadingChange(iPxPos: vec2<i32>) -> f32 {
    return textureLoad(t_shading_change, iPxPos, 0).x * cb.shading_change_scale;
}

fn SampleShadingChange(fUV: vec2<f32>) -> f32 {
    return textureSampleLevel(t_shading_change, sp_linear, fUV, 0.0).x * cb.shading_change_scale;
}

fn StoreShadingChange(iPxPos: vec2<i32>, fShadingChange: f32) {
    textureStore(w_shading_change, iPxPos, vec4<f32>(fShadingChange, 0.0, 0.0, 0.0));
}

fn SampleSPDMipLevel(fUV: vec2<f32>, mipLevel: i32) -> vec2<f32> {
    if mipLevel == 0 {
        return textureSampleLevel(t_spd_mip0, sp_linear, fUV, 0.0).rg;
    } else if mipLevel == 1 {
        return textureSampleLevel(t_spd_mip1, sp_linear, fUV, 0.0).rg;
    }
    return textureSampleLevel(t_spd_mip2, sp_linear, fUV, 0.0).rg;
}

fn StoreSPDMipLevel(mipLevel: i32, iPxPos: vec2<i32>, v: vec2<f32>) {
    if mipLevel == 0 {
        textureStore(w_spd_mip0, iPxPos, vec4<f32>(v, 0.0, 0.0));
    } else if mipLevel == 1 {
        textureStore(w_spd_mip1, iPxPos, vec4<f32>(v, 0.0, 0.0));
    } else if mipLevel == 2 {
        textureStore(w_spd_mip2, iPxPos, vec4<f32>(v, 0.0, 0.0));
    }
}

fn LoadFarthestDepthFromMip1(iPxPos: vec2<i32>) -> f32 {
    return textureLoad(t_farthest_depth_mip1, iPxPos, 0).x;
}

fn SampleFarthestDepthMip1(fUV: vec2<f32>) -> f32 {
    return textureSampleLevel(t_farthest_depth_mip1, sp_linear, fUV, 0.0).x;
}

fn StoreFarthestDepth(iPxPos: vec2<i32>, fDepth: f32) {
    textureStore(w_farthest_depth, iPxPos, vec4<f32>(fDepth, 0.0, 0.0, 0.0));
}

fn StoreFarthestDepthMip1(iPxPos: vec2<i32>, fDepth: f32) {
    textureStore(w_farthest_depth_mip1, iPxPos, vec4<f32>(fDepth, 0.0, 0.0, 0.0));
}

fn LoadCurrentLuma(iPxPos: vec2<i32>) -> f32 {
    return textureLoad(t_current_luma, iPxPos, 0).r;
}

fn SampleCurrentLuma(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(t_current_luma, sp_linear, uv, 0.0).r;
}

fn StoreCurrentLuma(iPxPos: vec2<i32>, fLuma: f32) {
    textureStore(w_current_luma, iPxPos, vec4<f32>(fLuma, 0.0, 0.0, 0.0));
}

fn SampleLumaInstability(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(t_luma_instability, sp_linear, uv, 0.0).x;
}

fn StoreLumaInstability(iPxPos: vec2<i32>, fLumaInstability: f32) {
    textureStore(w_luma_instability, iPxPos, vec4<f32>(fLumaInstability, 0.0, 0.0, 0.0));
}

fn LoadPreviousLuma(iPxPos: vec2<i32>) -> f32 {
    return textureLoad(t_previous_luma, iPxPos, 0).r;
}

fn LoadNewLocks(iPxPos: vec2<i32>) -> f32 {
    return textureLoad(t_new_locks, iPxPos, 0).r;
}

fn StoreNewLocks(iPxPos: vec2<i32>, newLock: f32) {
    textureStore(w_new_locks, iPxPos, vec4<f32>(newLock, 0.0, 0.0, 0.0));
}

fn LoadReconstructedPrevDepth(iPxPos: vec2<i32>) -> f32 {
    return bitcast<f32>(reconstructed_prev_depth[u32(iPxPos.y) * u32(RenderSize().x) + u32(iPxPos.x)]);
}

fn StoreReconstructedDepth(iPxPos: vec2<i32>, fDepth: f32) {
    let uDepth = bitcast<u32>(fDepth);
    let index = u32(iPxPos.y) * u32(RenderSize().x) + u32(iPxPos.x);
    atomicMin(&reconstructed_prev_depth_rw[index], uDepth);
}

fn StoreDilatedDepth(iPxPos: vec2<i32>, fDepth: f32) {
    textureStore(w_dilated_depth, iPxPos, vec4<f32>(fDepth, 0.0, 0.0, 0.0));
}

fn StoreDilatedMotionVector(iPxPos: vec2<i32>, fMotionVector: vec2<f32>) {
    textureStore(w_dilated_motion_vectors, iPxPos, vec4<f32>(fMotionVector, 0.0, 0.0));
}

fn LoadDilatedMotionVector(iPxInput: vec2<i32>) -> vec2<f32> {
    return textureLoad(t_dilated_motion_vectors, iPxInput, 0).xy;
}

fn SampleDilatedMotionVector(fUV: vec2<f32>) -> vec2<f32> {
    return textureSampleLevel(t_dilated_motion_vectors, sp_linear, fUV, 0.0).xy;
}

fn LoadDilatedDepth(iPxInput: vec2<i32>) -> f32 {
    return textureLoad(t_dilated_depth, iPxInput, 0).r;
}

fn SampleDilatedDepth(fUV: vec2<f32>) -> f32 {
    return textureSampleLevel(t_dilated_depth, sp_linear, fUV, 0.0).r;
}

fn SampleDilatedReactiveMasks(fUV: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t_dilated_reactive_masks, sp_linear, fUV, 0.0);
}

fn StoreDilatedReactiveMasks(iPxPos: vec2<i32>, fDilatedReactiveMasks: vec4<f32>) {
    textureStore(w_dilated_reactive_masks, iPxPos, fDilatedReactiveMasks);
}

fn SampleTransparencyAndCompositionMask(fUV: vec2<f32>) -> f32 {
    // No transparency-and-composition mask is supplied by this pipeline.
    return 0.0;
}

fn LoadReactiveMask(iPxPos: vec2<i32>) -> f32 {
    // No application-supplied reactive mask exists in this pipeline, so the
    // default (all zero) resource is used and reactivenessScale is 1.
    return 0.0;
}
