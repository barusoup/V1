// FSR 3.1 pass: RCAS sharpening (ffx_fsr3upscaler_rcas.h + FSR1's FsrRcasF).
//
// The reference implementation swizzles the workgroup into a quad pattern for
// cache locality; this port dispatches a plain 8x8 tile, which produces the same
// per-pixel result.

const FSR_RCAS_LIMIT: f32 = 0.25 - (1.0 / 16.0);

fn FsrRcasLoadF(p: vec2<i32>) -> vec3<f32> {
    let size = UpscaleSize();
    let clamped = clamp(p, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
    let fColor = LoadRCAS_Input(clamped);
    return fColor.rgb * Exposure();
}

fn WriteUpscaledOutput(iPxHrPos: vec2<i32>, fUpscaledColor: vec3<f32>) {
    StoreUpscaledOutput(iPxHrPos, fUpscaledColor);
}

fn CurrFilter(pos: vec2<i32>) {
    // Algorithm uses a minimal 3x3 pixel neighborhood.
    //     b
    //   d e f
    //     h
    let sp = pos;
    let b = FsrRcasLoadF(sp + vec2<i32>(0, -1));
    let d = FsrRcasLoadF(sp + vec2<i32>(-1, 0));
    let e = FsrRcasLoadF(sp);
    let f = FsrRcasLoadF(sp + vec2<i32>(1, 0));
    let h = FsrRcasLoadF(sp + vec2<i32>(0, 1));

    // Luma times 2.
    let bL = b.b * 0.5 + (b.r * 0.5 + b.g);
    let dL = d.b * 0.5 + (d.r * 0.5 + d.g);
    let eL = e.b * 0.5 + (e.r * 0.5 + e.g);
    let fL = f.b * 0.5 + (f.r * 0.5 + f.g);
    let hL = h.b * 0.5 + (h.r * 0.5 + h.g);

    // Noise detection.
    var nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    let rng = max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL);
    nz = ffxSaturate(abs(nz) * (1.0 / max(rng, 1e-8)));
    nz = -0.5 * nz + 1.0;

    // Min and max of ring.
    let mn4R = min(min(min(b.r, d.r), f.r), h.r);
    let mn4G = min(min(min(b.g, d.g), f.g), h.g);
    let mn4B = min(min(min(b.b, d.b), f.b), h.b);
    let mx4R = max(max(max(b.r, d.r), f.r), h.r);
    let mx4G = max(max(max(b.g, d.g), f.g), h.g);
    let mx4B = max(max(max(b.b, d.b), f.b), h.b);

    // Immediate constants for peak range.
    let peakC = vec2<f32>(1.0, -4.0);

    // Limiters.
    let hitMinR = mn4R / (4.0 * mx4R);
    let hitMinG = mn4G / (4.0 * mx4G);
    let hitMinB = mn4B / (4.0 * mx4B);
    let hitMaxR = (peakC.x - mx4R) / (4.0 * mn4R + peakC.y);
    let hitMaxG = (peakC.x - mx4G) / (4.0 * mn4G + peakC.y);
    let hitMaxB = (peakC.x - mx4B) / (4.0 * mn4B + peakC.y);
    let lobeR = max(-hitMinR, hitMaxR);
    let lobeG = max(-hitMinG, hitMaxG);
    let lobeB = max(-hitMinB, hitMaxB);
    let lobe = max(-FSR_RCAS_LIMIT, min(max(max(lobeR, lobeG), lobeB), 0.0)) * bitcast<f32>(RCASConfig().x);

    // Apply noise removal (FSR_RCAS_DENOISE).
    let lobe2 = lobe * nz;

    // Resolve.
    let rcpL = 1.0 / (4.0 * lobe2 + 1.0);
    var c: vec3<f32>;
    c.r = (lobe2 * b.r + lobe2 * d.r + lobe2 * h.r + lobe2 * f.r + e.r) * rcpL;
    c.g = (lobe2 * b.g + lobe2 * d.g + lobe2 * h.g + lobe2 * f.g + e.g) * rcpL;
    c.b = (lobe2 * b.b + lobe2 * d.b + lobe2 * h.b + lobe2 * f.b + e.b) * rcpL;

    c /= Exposure();

    WriteUpscaledOutput(pos, c);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = vec2<i32>(gid.xy);
    if (!IsOnScreen(pos, UpscaleSize())) {
        return;
    }
    CurrFilter(pos);
}
