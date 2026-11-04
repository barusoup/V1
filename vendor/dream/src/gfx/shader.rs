//! WGSL -> SPIR-V compilation for the Vulkan backend.
//!
//! Built-in shaders are precompiled to stripped SPIR-V at build time by
//! `build.rs` and resolved in O(1) with zero runtime `naga` overhead. Dynamic
//! or custom WGSL strings fall back to runtime `naga` compilation seamlessly.

include!(concat!(env!("OUT_DIR"), "/precompiled_shaders.rs"));

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Look up a build-time precompiled SPIR-V slice for `source`, if present.
pub(crate) fn precompiled_wgsl(source: &str) -> Option<&'static [u32]> {
    let hash = fnv1a_64(source.as_bytes());
    let len = source.len();
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
