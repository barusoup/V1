// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Parsing the client's `Range` header and deciding which bytes to serve.

/// What the client asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// No `Range` header: the whole file.
    Full,
    /// `bytes=start-` or `bytes=start-end`.
    Range { start: u64, end: Option<u64> },
}

/// Parse a `Range` header. Only single-range `bytes=` forms are supported. Anything else is
/// treated as a full request, which is what a player expects when a server ignores ranges.
pub fn parse_range(header: Option<&str>) -> Want {
    let Some(value) = header.and_then(|h| h.trim().strip_prefix("bytes=")) else {
        return Want::Full;
    };
    if value.contains(',') {
        return Want::Full;
    }
    let Some((start, end)) = value.split_once('-') else {
        return Want::Full;
    };
    let Ok(start) = start.trim().parse::<u64>() else {
        return Want::Full;
    };
    let end = match end.trim() {
        "" => None,
        e => match e.parse::<u64>() {
            Ok(e) => Some(e),
            Err(_) => return Want::Full,
        },
    };
    Want::Range { start, end }
}

/// The inclusive byte window to serve out of a file of `total` bytes. `None` when the range starts
/// past the end, which is the one case that answers 416.
pub fn resolve_window(want: Want, total: u64) -> Option<(u64, u64)> {
    if total == 0 {
        return None;
    }
    match want {
        Want::Full => Some((0, total - 1)),
        Want::Range { start, end } => {
            if start >= total {
                return None;
            }
            let last = total - 1;
            let end = end.map_or(last, |e| e.min(last));
            (end >= start).then_some((start, end))
        }
    }
}

/// `(first, last, total)` from a `Content-Range: bytes 0-1023/12345` header. `None` for the forms
/// that do not say all three (`bytes */12345`, an unknown total `*`).
pub fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes")?.trim_start();
    let (span, total) = rest.split_once('/')?;
    let (first, last) = span.split_once('-')?;
    Some((
        first.trim().parse().ok()?,
        last.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

/// `total` from a `Content-Range: bytes 0-0/12345` header.
pub fn total_from_content_range(value: &str) -> Option<u64> {
    value.rsplit_once('/')?.1.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_forms_players_send() {
        assert_eq!(parse_range(None), Want::Full);
        assert_eq!(
            parse_range(Some("bytes=0-")),
            Want::Range {
                start: 0,
                end: None
            }
        );
        assert_eq!(
            parse_range(Some("bytes=10-19")),
            Want::Range {
                start: 10,
                end: Some(19)
            }
        );
        assert_eq!(
            parse_range(Some("bytes=0-1,5-6")),
            Want::Full,
            "multi-range is not served"
        );
        assert_eq!(parse_range(Some("items=0-")), Want::Full);
    }

    #[test]
    fn windows_clamp_to_the_file() {
        assert_eq!(resolve_window(Want::Full, 100), Some((0, 99)));
        assert_eq!(
            resolve_window(
                Want::Range {
                    start: 90,
                    end: None
                },
                100
            ),
            Some((90, 99))
        );
        assert_eq!(
            resolve_window(
                Want::Range {
                    start: 90,
                    end: Some(500)
                },
                100
            ),
            Some((90, 99))
        );
        assert_eq!(
            resolve_window(
                Want::Range {
                    start: 100,
                    end: None
                },
                100
            ),
            None
        );
        assert_eq!(
            resolve_window(
                Want::Range {
                    start: 5,
                    end: Some(2)
                },
                100
            ),
            None
        );
    }

    #[test]
    fn reads_all_three_numbers_from_content_range() {
        assert_eq!(
            parse_content_range("bytes 4194304-8388607/582623249"),
            Some((4_194_304, 8_388_607, 582_623_249))
        );
        assert_eq!(parse_content_range("bytes */1000"), None);
        assert_eq!(parse_content_range("bytes 0-9/*"), None);
        assert_eq!(parse_content_range("items 0-9/10"), None);
    }

    #[test]
    fn reads_the_total_from_content_range() {
        assert_eq!(
            total_from_content_range("bytes 0-0/582623249"),
            Some(582_623_249)
        );
        assert_eq!(total_from_content_range("bytes */*"), None);
    }
}
