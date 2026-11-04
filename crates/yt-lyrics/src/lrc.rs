// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! LRC text, the format LRCLIB's `syncedLyrics` uses: `[mm:ss.xx] words`.
//!
//! Handled beyond the bare stamp:
//! - A line can carry several stamps (`[00:10.00][01:05.00]chorus`), and it repeats at each.
//! - Metadata tags (`[ar:…]`, `[ti:…]`) are skipped. `[offset:±ms]` shifts every stamp.
//! - Bracketed words that are not a time (`[Chorus] la la`) stay as text.
//! - A stamp with no words is a pause. It is kept, so the panel knows the song has gone quiet.
//! - A byte-order mark, CRLF line ends and 1 to 3 fraction digits are all accepted.

use crate::model::{Lyrics, Source, TimedLine};

/// Parse LRC. `None` when there is no timestamped line at all, so the caller can try the plain text.
pub fn parse(source: Source, text: &str) -> Option<Lyrics> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut offset_ms: i64 = 0;
    let mut stamped: Vec<(i64, String)> = Vec::new();

    for raw in text.lines() {
        let mut rest = raw.trim();
        let mut stamps = Vec::new();
        while let Some(after) = rest.strip_prefix('[') {
            let Some(end) = after.find(']') else { break };
            let tag = &after[..end];
            if let Some(ms) = parse_stamp(tag) {
                stamps.push(ms);
                rest = after[end + 1..].trim_start();
            } else {
                // Not a time: either a metadata tag, or bracketed words that belong to the text.
                if let Some(ms) = tag
                    .strip_prefix("offset:")
                    .and_then(|v| v.trim().parse().ok())
                {
                    offset_ms = ms;
                }
                break;
            }
        }
        for ms in stamps {
            stamped.push((ms as i64, rest.to_owned()));
        }
    }

    if stamped.is_empty() {
        return None;
    }
    // LRC convention: a positive offset makes the lyrics come sooner, so it comes off the stamps.
    let lines = stamped
        .into_iter()
        .map(|(ms, text)| TimedLine {
            time_ms: (ms - offset_ms).max(0) as u64,
            text,
        })
        .collect();
    Lyrics::synced(source, lines)
}

/// `mm:ss`, `mm:ss.x`, `mm:ss.xx` or `mm:ss.xxx` as milliseconds.
fn parse_stamp(tag: &str) -> Option<u64> {
    let (minutes, rest) = tag.split_once(':')?;
    let minutes: u64 = minutes.trim().parse().ok()?;
    let (seconds, fraction) = match rest.split_once('.') {
        Some((s, f)) => (s, Some(f)),
        None => (rest, None),
    };
    let seconds: u64 = seconds.trim().parse().ok()?;
    let millis = match fraction {
        None => 0,
        Some(digits) => {
            let digits: String = digits.chars().take_while(char::is_ascii_digit).collect();
            if digits.is_empty() {
                return None;
            }
            let value: u64 = digits.chars().take(3).collect::<String>().parse().ok()?;
            match digits.len().min(3) {
                1 => value * 100,
                2 => value * 10,
                _ => value,
            }
        }
    };
    Some((minutes * 60 + seconds) * 1000 + millis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Body;

    fn synced(text: &str) -> Vec<(u64, String)> {
        match parse(Source::Lrclib, text).expect("has stamps").body {
            Body::Synced(lines) => lines.into_iter().map(|l| (l.time_ms, l.text)).collect(),
            other => panic!("expected synced, got {other:?}"),
        }
    }

    #[test]
    fn basic_lines_and_fraction_widths() {
        let lines = synced("[00:01.00]one\n[00:03.5]two\n[01:02.345]three");
        assert_eq!(
            lines,
            vec![
                (1_000, "one".into()),
                (3_500, "two".into()),
                (62_345, "three".into())
            ]
        );
    }

    #[test]
    fn repeated_stamps_repeat_the_line() {
        let lines = synced("[00:10.00][01:05.00]chorus\n[00:05.00]verse");
        assert_eq!(
            lines,
            vec![
                (5_000, "verse".into()),
                (10_000, "chorus".into()),
                (65_000, "chorus".into())
            ]
        );
    }

    #[test]
    fn metadata_tags_are_not_lines() {
        let lines = synced("[ar:Queen]\n[ti:Song]\n[by:someone]\n[00:01.00]hi\n[00:02.00]there");
        assert_eq!(lines, vec![(1_000, "hi".into()), (2_000, "there".into())]);
    }

    #[test]
    fn offset_shifts_every_stamp_and_never_goes_negative() {
        let lines = synced("[offset:+500]\n[00:02.00]x\n[00:00.20]early");
        assert_eq!(lines, vec![(0, "early".into()), (1_500, "x".into())]);
    }

    #[test]
    fn bracketed_words_stay_in_the_text() {
        let lines = synced("[00:01.00][Chorus] la la\n[00:03.00]tail");
        assert_eq!(
            lines,
            vec![(1_000, "[Chorus] la la".into()), (3_000, "tail".into())]
        );
    }

    #[test]
    fn bom_and_crlf_are_accepted() {
        let lines = synced("\u{feff}[00:01.00]a\r\n[00:02.00]b\r\n");
        assert_eq!(lines, vec![(1_000, "a".into()), (2_000, "b".into())]);
    }

    #[test]
    fn a_timed_empty_line_is_a_pause() {
        let lines = synced("[00:01.00]a\n[00:04.00]\n[00:06.00]b");
        assert_eq!(
            lines,
            vec![(1_000, "a".into()), (4_000, "".into()), (6_000, "b".into())]
        );
    }

    #[test]
    fn no_stamps_is_not_synced() {
        assert!(parse(Source::Lrclib, "just words\nmore words").is_none());
    }

    #[test]
    fn every_line_at_zero_is_plain() {
        let lyrics = parse(Source::Lrclib, "[00:00.00]a\n[00:00.00]b").unwrap();
        assert_eq!(lyrics.body, Body::Plain(vec!["a".into(), "b".into()]));
    }
}
