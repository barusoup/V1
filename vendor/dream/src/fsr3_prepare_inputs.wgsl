// FSR 3.1 pass: prepare inputs (contains ffx_fsr3upscaler_prepare_inputs.h and
// the reference reprojection helpers from ffx_fsr3upscaler_reproject.h).

struct DepthExtents {
    nearest: f32,
    nearest_coord: vec2<i32>,
    farthest: f32,
}

fn ReconstructPrevDepth(iPxPos: vec2<i32>, fDepth: f32, fMotionVectorIn: vec2<f32>) {
    let fNearestDepthInMeters = min(GetViewSpaceDepthInMeters(fDepth), FSR3UPSCALER_FP16_MAX);
    let fReconstructedDeptMvThreshold = ReconstructedDepthMvPxThreshold(fNearestDepthInMeters);

    var fMotionVector = fMotionVectorIn;
    // Discard small mvs
    fMotionVector *= select(0.0, 1.0, Get4KVelocity(fMotionVector) > fReconstructedDeptMvThreshold);

    let fUv = (vec2<f32>(iPxPos) + 0.5) / vec2<f32>(RenderSize());
    let fReprojectedUv = fUv + fMotionVector;
    let bilinearInfo = GetBilinearSamplingData(fReprojectedUv, RenderSize());

    // Push to all pixels having some contribution if reprojection is bilinear.
    for (var iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++) {
        let iOffset = bilinearInfo.offsets[iSampleIndex];
        let fWeight = bilinearInfo.weights[iSampleIndex];
        if (fWeight > fReconstructedDepthBilinearWeightThreshold) {
            let iStorePos = bilinearInfo.base_pos + iOffset;
            if (IsOnScreen(iStorePos, RenderSize())) {
                StoreReconstructedDepth(iStorePos, fDepth);
            }
        }
    }
}

fn FindDepthExtents(iPxPos: vec2<i32>) -> DepthExtents {
    var extents: DepthExtents;
    let iSampleOffsets = array<vec2<i32>, 9>(
        vec2<i32>(0, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, 1),
        vec2<i32>(0, -1),
        vec2<i32>(-1, 0),
        vec2<i32>(-1, 1),
        vec2<i32>(1, 1),
        vec2<i32>(-1, -1),
        vec2<i32>(1, -1));

    var depth: array<f32, 9>;
    for (var i = 0; i < 9; i++) {
        depth[i] = LoadInputDepth(iPxPos + iSampleOffsets[i]);
    }

    extents.nearest_coord = iPxPos;
    extents.nearest = depth[0];
    extents.farthest = depth[0];
    for (var i = 1; i < 9; i++) {
        let iPos = iPxPos + iSampleOffsets[i];
        if (IsOnScreen(iPos, RenderSize())) {
            let fNdDepth = depth[i];
            if (fNdDepth < extents.nearest) {
                extents.farthest = max(extents.farthest, fNdDepth);
                extents.nearest_coord = iPos;
                extents.nearest = fNdDepth;
            }
        }
    }
    return extents;
}

// Motion vectors are at render resolution (the app does not use
// FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS), which is the
// SDK's FSR3UPSCALER_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS case: sample the
// input motion vectors at the nearest render-resolution position.
fn DilateMotionVector(iPxPos: vec2<i32>, depthExtents: DepthExtents) -> vec2<f32> {
    _ = iPxPos;
    return LoadInputMotionVector(depthExtents.nearest_coord);
}

fn GetCurrentFrameLuma(iPxPos: vec2<i32>) -> f32 {
    let fRgb = max(vec3<f32>(0.0, 0.0, 0.0), LoadInputColor(iPxPos));
    return RGBToLuma(fRgb);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iPxPos = vec2<i32>(gid.xy);
    if (!IsOnScreen(iPxPos, RenderSize())) {
        return;
    }

    let depthExtents = FindDepthExtents(iPxPos);
    let fDilatedMotionVector = DilateMotionVector(iPxPos, depthExtents);

    ReconstructPrevDepth(iPxPos, depthExtents.nearest, fDilatedMotionVector);

    StoreDilatedMotionVector(iPxPos, fDilatedMotionVector);
    StoreDilatedDepth(iPxPos, depthExtents.nearest);

    let fFarthestDepthInMeters = min(GetViewSpaceDepthInMeters(depthExtents.farthest), FSR3UPSCALER_FP16_MAX);
    StoreFarthestDepth(iPxPos, fFarthestDepthInMeters);

    StoreCurrentLuma(iPxPos, GetCurrentFrameLuma(iPxPos));
}
