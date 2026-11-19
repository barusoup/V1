//! WGSL -> SPIR-V compilation for the Vulkan backend.
//!
//! Built-in shaders are precompiled to stripped SPIR-V at build time by
//! `build.rs` and resolved in O(1) with zero runtime `naga` overhead. Dynamic
//! or custom WGSL strings fall back to runtime `naga` compilation seamlessly.

include!(concat!(env!("OUT_DIR"), "/precompiled_shaders.rs"));

fn fnv1a_64_normalized(bytes: &[u8]) -> (u64, usize) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut len = 0usize;
    for &b in bytes {
        if b == b'\r' {
            continue;
        }
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        len += 1;
    }
    (hash, len)
}

/// Look up a build-time precompiled SPIR-V slice for `source`, if present.
pub(crate) fn precompiled_wgsl(source: &str) -> Option<&'static [u32]> {
    let (hash, len) = fnv1a_64_normalized(source.as_bytes());
    PRECOMPILED_SPV
        .iter()
        .find(|&&(h, l, _)| h == hash && l == len)
        .map(|&(_, _, words)| words)
}

/// Compile a WGSL source string into SPIR-V words.
///
/// Panics with the shader's own diagnostics on failure: a shader that will not
/// compile is a programming error, not a runtime condition to recover from.
pub fn compile_wgsl(source: &str, label: &str) -> Vec<u32> {
    if let Some(words) = precompiled_wgsl(source) {
        return words.to_vec();
    }
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|err| {
        panic!(
            "{label}: WGSL parse failed:
{}",
            err.emit_to_string(source)
        )
    });

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .unwrap_or_else(|err| panic!("{label}: WGSL validation failed: {err:?}"));

    let options = naga::back::spv::Options {
        flags: naga::back::spv::Options::default()
            .flags
            .difference(naga::back::spv::WriterFlags::DEBUG),
        ..naga::back::spv::Options::default()
    };
    naga::back::spv::write_vec(&module, &info, &options, None)
        .unwrap_or_else(|err| panic!("{label}: SPIR-V generation failed: {err:?}"))
}

#[cfg(test)]
mod tests {
    use super::precompiled_wgsl;

    #[test]
    fn builtin_wgsl_shaders_hit_precompiled_cache() {
        assert!(precompiled_wgsl(crate::DEFAULT_SCENE_WGSL).is_some());
        assert!(precompiled_wgsl(include_str!("../app/scene.wgsl")).is_some());
        assert!(precompiled_wgsl(include_str!("../bedframe/ui.wgsl")).is_some());
        assert!(precompiled_wgsl(include_str!("../blit.wgsl")).is_some());
        assert!(precompiled_wgsl(include_str!("../present.wgsl")).is_some());
        let pbr = format!(
            "{}{}",
            crate::lighting::SHADER,
            include_str!("../model/model_pbr.wgsl")
        );
        assert!(
            precompiled_wgsl(&pbr).is_some(),
            "model_pbr.wgsl must hit precompiled SPIR-V"
        );
        let impostor = format!(
            "{}{}",
            crate::lighting::SHADER,
            include_str!("../impostor/impostor.wgsl")
        );
        assert!(
            precompiled_wgsl(&impostor).is_some(),
            "impostor.wgsl must hit precompiled SPIR-V"
        );
        let fsr_prepare = format!(
            "{}\n{}",
            include_str!("../fsr3_common.wgsl"),
            include_str!("../fsr3_prepare_inputs.wgsl")
        );
        assert!(
            precompiled_wgsl(&fsr_prepare).is_some(),
            "fsr3_prepare_inputs.wgsl must hit precompiled SPIR-V"
        );
    }
}
