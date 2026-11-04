//! Explicit script roles and grapheme classification — shared by the
//! macOS CoreText path and the portable fontdue path.
//!
//! A grapheme belongs to exactly one role; `Primary` is not a fallback
//! for uncovered text, it is the role for everything not matched.

/// Explicit script role a grapheme resolves to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptRole {
    Han,
    Kana,
    Hangul,
    Emoji,
    /// Everything else: the caller's primary font.
    Primary,
}

/// `script_overrides` slot per role. `Primary` never looks up a role
/// font — passing it to `set_script_font` is a documented no-op.
pub(crate) fn role_index(role: ScriptRole) -> usize {
    match role {
        ScriptRole::Han => 0,
        ScriptRole::Kana => 1,
        ScriptRole::Hangul => 2,
        ScriptRole::Emoji => 3,
        ScriptRole::Primary => usize::MAX,
    }
}

/// `ch` is a Han ideograph (unified, ext A/B, compat).
fn is_han(c: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&c)
        || (0x3400..=0x4DBF).contains(&c)
        || (0x20000..=0x2A6DF).contains(&c)
        || (0x2A700..=0x3134F).contains(&c)
        || (0xF900..=0xFAFF).contains(&c)
}

/// `ch` is Kana (including halfwidth forms).
fn is_kana(c: u32) -> bool {
    (0x3040..=0x30FF).contains(&c)
        || (0x31F0..=0x31FF).contains(&c)
        || (0xFF66..=0xFF9D).contains(&c)
}

/// `ch` is Hangul (syllables, jamo, compat jamo).
fn is_hangul(c: u32) -> bool {
    (0xAC00..=0xD7AF).contains(&c)
        || (0x1100..=0x11FF).contains(&c)
        || (0x3130..=0x318F).contains(&c)
}

/// `ch` is CJK punctuation or fullwidth form — drawn by the East-Asian
/// faces, so it classifies with `cluster_role`'s script context: the
/// caller resolves it to the same role as surrounding CJK.
fn is_cjk_punct(c: u32) -> bool {
    (0x3000..=0x303F).contains(&c) || (0xFF01..=0xFF60).contains(&c)
}

/// Actual emoji-presentation code points (a checked subset — technical
/// characters like ⌘ U+2318 or ⏱ U+23F1 are NOT emoji by default).
fn is_emoji_presentation(c: u32) -> bool {
    matches!(c,
        0x231A..=0x231B | 0x23E9..=0x23EC | 0x23F0 | 0x23F3
        | 0x25FD..=0x25FE | 0x2614..=0x2615 | 0x2648..=0x2653 | 0x267F
        | 0x2693 | 0x26A1 | 0x26AA..=0x26AB | 0x26BD..=0x26BE
        | 0x26C4..=0x26C5 | 0x26CE | 0x26D4 | 0x26EA | 0x26F2..=0x26F3
        | 0x26F5 | 0x26FA | 0x26FD | 0x2705 | 0x270A..=0x270B | 0x2728
        | 0x274C | 0x274E | 0x2753..=0x2755 | 0x2757 | 0x2795..=0x2797
        | 0x27B0 | 0x27BF | 0x2B1B..=0x2B1C | 0x2B50 | 0x2B55
        | 0x1F004 | 0x1F0CF | 0x1F18E | 0x1F191..=0x1F19A
        | 0x1F1E6..=0x1F1FF | 0x1F201 | 0x1F21A | 0x1F22F
        | 0x1F232..=0x1F236 | 0x1F238..=0x1F23A | 0x1F250..=0x1F251
        | 0x1F300..=0x1FAFF)
}

/// Which explicit role one extended grapheme belongs to. Emoji needs a
/// real signal: a pictograph code point, VS16 (U+FE0F), a keycap
/// (U+20E3) or a ZWJ sequence containing a pictograph. A plain ASCII
/// digit, ⌘, © or ™ stays `Primary`; VS15 (U+FE0E) forces text too.
pub(crate) fn script_role(cluster: &str) -> ScriptRole {
    let mut cjk_punct = false;
    let mut pictograph = false;
    let mut vs16 = false;
    let mut keycap = false;
    for ch in cluster.chars() {
        let c = ch as u32;
        if is_han(c) {
            return ScriptRole::Han;
        }
        if is_kana(c) {
            return ScriptRole::Kana;
        }
        if is_hangul(c) {
            return ScriptRole::Hangul;
        }
        if is_cjk_punct(c) {
            cjk_punct = true;
        }
        if c == 0xFE0E {
            // VS15: explicitly text presentation.
            return ScriptRole::Primary;
        }
        if c == 0xFE0F {
            vs16 = true;
        }
        if c == 0x20E3 {
            keycap = true;
        }
        if is_emoji_presentation(c) {
            pictograph = true;
        }
    }
    // Emoji when a pictograph is present, VS16 asks for it, or a keycap
    // modifier is applied.
    if pictograph || vs16 || keycap {
        ScriptRole::Emoji
    } else if cjk_punct {
        // Lone CJK punctuation goes with the Han face (its glyphs carry
        // the fullwidth forms).
        ScriptRole::Han
    } else {
        ScriptRole::Primary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_never_turns_technical_glyphs_emoji() {
        // The spec's hard negatives.
        assert_eq!(script_role("\u{2318}"), ScriptRole::Primary, "⌘ stays text");
        assert_eq!(script_role("©"), ScriptRole::Primary);
        assert_eq!(script_role("™"), ScriptRole::Primary);
        assert_eq!(script_role("1"), ScriptRole::Primary, "ASCII digit");
        assert_eq!(script_role("⏱"), ScriptRole::Primary, "U+23F1 is technical");
        // VS15 forces text even on a pictograph.
        assert_eq!(script_role("❤\u{FE0E}"), ScriptRole::Primary);
        // Real emoji.
        assert_eq!(script_role("😀"), ScriptRole::Emoji);
        assert_eq!(script_role("🇯🇵"), ScriptRole::Emoji, "regional indicators");
        assert_eq!(
            script_role("#\u{FE0F}\u{20E3}"),
            ScriptRole::Emoji,
            "keycap"
        );
        assert_eq!(script_role("❤\u{FE0F}"), ScriptRole::Emoji, "VS16");
        // Scripts.
        assert_eq!(script_role("字"), ScriptRole::Han);
        assert_eq!(script_role("𠀀"), ScriptRole::Han, "ext B");
        assert_eq!(script_role("か"), ScriptRole::Kana);
        assert_eq!(script_role("ｶ"), ScriptRole::Kana, "halfwidth");
        assert_eq!(script_role("한"), ScriptRole::Hangul);
        assert_eq!(script_role("。"), ScriptRole::Han, "CJK punctuation");
        assert_eq!(script_role("Ａ"), ScriptRole::Han, "fullwidth form");
        assert_eq!(script_role("a"), ScriptRole::Primary);
    }
}
