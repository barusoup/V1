fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut failed = false;
    let entries = wgsl_files("src");

    assert!(
        !entries.is_empty(),
        "no .wgsl files found in src/; build validation skipped?"
    );
    // FSR 3.1 pass files are concatenated after the shared prelude at load, so
    // validate them the same way here.
    let prelude = std::fs::read_to_string("src/fsr3_common.wgsl").unwrap_or_default();
    // PBR shaders get the shared lighting module prepended at load; validate
    // them the same way here.
    let lighting = std::fs::read_to_string("src/lighting/lighting.wgsl").unwrap_or_default();
    let spv_options = naga::back::spv::Options {
        flags: naga::back::spv::Options::default()
            .flags
            .difference(naga::back::spv::WriterFlags::DEBUG),
        ..naga::back::spv::Options::default()
    };
    let mut compiled_entries: Vec<(u64, usize, Vec<u32>)> = Vec::new();
    let mut seen_keys = std::collections::HashSet::new();

    for path in entries {
        println!("cargo:rerun-if-changed={}", path.display());
        let raw_source = std::fs::read_to_string(&path).expect("read wgsl");
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name == "fsr3_common.wgsl" || name == "lighting.wgsl" {
            // Shared preludes are prepended to pass shaders, but also validate them standalone.
        }
        let mut variants = Vec::new();
        for (p_pre, l_pre, src_var) in [
            (prelude.as_str(), lighting.as_str(), raw_source.as_str()),
            (
                &prelude.replace(
                    "

", "
",
                ),
                &lighting.replace(
                    "

", "
",
                ),
                &raw_source.replace(
                    "

", "
",
                ),
            ),
        ] {
            let mut s = src_var.to_string();
            if name.starts_with("fsr3_") && name != "fsr3_common.wgsl" {
                s = format!(
                    "{p_pre}
{s}"
                );
            }
            if s.contains("lighting_eval") && name != "lighting.wgsl" {
                s = format!(
                    "{l_pre}
{s}"
                );
            }
            variants.push(s);
        }
        let source = &variants[0];
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(module) => module,
            Err(error) => {
                eprintln!(
                    "WGSL parse error in {}:
{}",
                    path.display(),
                    error.emit_to_string(source)
                );
                failed = true;
                continue;
            }
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = match validator.validate(&module) {
            Ok(info) => info,
            Err(error) => {
                eprintln!(
                    "WGSL validation error in {}:
{error:?}",
                    path.display()
                );
                failed = true;
                continue;
            }
        };
        match naga::back::spv::write_vec(&module, &info, &spv_options, None) {
            Ok(words) => {
                for v in &variants {
                    let h = fnv1a_64(v.as_bytes());
                    let len = v.len();
                    if seen_keys.insert((h, len)) {
                        compiled_entries.push((h, len, words.clone()));
                    }
                }
            }
            Err(error) => {
                eprintln!("SPIR-V generation failed for {}: {error:?}", path.display());
                failed = true;
            }
        }
    }
    if failed {
        panic!("one or more WGSL shaders failed; see errors above");
    }

    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let dest = std::path::Path::new(&out_dir).join("precompiled_shaders.rs");
        let mut code = String::from(
            "pub static PRECOMPILED_SPV: &[(u64, usize, &[u32])] = &[
",
        );
        for (hash, len, words) in &compiled_entries {
            code.push_str(&format!("    ({hash:#018x}, {len}, &["));
            for w in words {
                code.push_str(&format!("{w:#010x},"));
            }
            code.push_str(
                "]),
",
            );
        }
        code.push_str(
            "];
",
        );
        let _ = std::fs::write(dest, code);
    }
}

/// Every .wgsl under `root`, recursively (submodules keep their shaders
/// beside their code).
fn wgsl_files(root: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut dirs = vec![std::path::PathBuf::from(root)];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).expect("read source dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "wgsl") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}
