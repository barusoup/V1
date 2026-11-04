//! Bounded dedup warnings for unsupported clusters and missing role
//! faces — at most 128 distinct keys are reported, then suppressed
//! (the caller's text stays correct; the console stays readable).

use std::collections::HashSet;
use std::sync::Mutex;

const MAX_WARNINGS: usize = 128;

/// Warning buckets by purpose.
pub(crate) enum WarnKind {
    UnsupportedCluster,
    MissingRole,
    OversizeGlyph,
}

fn bucket(kind: WarnKind) -> &'static Mutex<Option<HashSet<String>>> {
    static UNSUPPORTED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    static MISSING_ROLE: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    static OVERSIZE: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    match kind {
        WarnKind::UnsupportedCluster => &UNSUPPORTED,
        WarnKind::MissingRole => &MISSING_ROLE,
        WarnKind::OversizeGlyph => &OVERSIZE,
    }
}

/// `key` identifies the miss (face/cluster). Bounded preview: never
/// interpolates more than 16 scalars into the log line.
pub(crate) fn warn_once(kind: WarnKind, key: &str, preview: &str) {
    let set = bucket(kind);
    let mut guard = set.lock().unwrap();
    let set = guard.get_or_insert_with(HashSet::new);
    if set.len() >= MAX_WARNINGS {
        if set.len() == MAX_WARNINGS {
            eprintln!("[ui] suppressing further glyph warnings ({MAX_WARNINGS} reached)");
            set.insert(String::new()); // sentinel: don't reprint the cap note
        }
        return;
    }
    let key = preview_codepoints(key, preview);
    if set.insert(key.clone()) {
        eprintln!("[ui] {key}");
    }
}

/// A bounded `(codepoint preview)` for log lines: ≤16 scalars then "…".
fn preview_codepoints(key: &str, preview: &str) -> String {
    let short: String = preview.chars().take(16).collect();
    let ellipsis = if preview.chars().count() > 16 {
        "…"
    } else {
        ""
    };
    format!("{key} {short:?}{ellipsis}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_warning_set_is_hard_capped() {
        for i in 0..300 {
            warn_once(WarnKind::OversizeGlyph, &format!("k{i}"), "x");
        }
        let guard = bucket(WarnKind::OversizeGlyph).lock().unwrap();
        let set = guard.as_ref().unwrap();
        assert!(
            set.len() <= MAX_WARNINGS + 1,
            "cap {} + sentinel, got {}",
            MAX_WARNINGS,
            set.len()
        );
    }
}
