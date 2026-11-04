// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! TTML, the XML lyric format Boidu serves (Apple Music's flavour, with word timing).
//!
//! The line is the `<p>`. Its time is the `begin` on the `<p>`, or the nearest enclosing `<div>`'s
//! when the `<p>` has none. The text is every text node inside the `<p>`, at any depth, so
//! background vocals (`<span ttm:role="x-bg">` nested in a span) and the spaces between word spans
//! all come through. Word timing is not kept: the panel moves by line.
//!
//! Entities are resolved by the parser. Only the five predefined ones and numeric references are
//! expected. An unknown one is kept as written, not treated as a fatal error.

use quick_xml::events::{BytesRef, BytesStart, Event};
use quick_xml::Reader;

use crate::model::{Lyrics, Source, TimedLine};

/// Parse a TTML document. `Ok(None)` when it holds no text. `Err` when it is not XML.
pub fn parse(source: Source, xml: &str) -> Result<Option<Lyrics>, String> {
    let mut reader = Reader::from_str(xml);
    // One entry per open `<div>`: its `begin`, if it had one.
    let mut divs: Vec<Option<u64>> = Vec::new();
    // The `<p>` being read: its own begin (or the div's), and its text so far.
    let mut current: Option<(Option<u64>, String)> = None;
    let mut lines: Vec<(Option<u64>, String)> = Vec::new();

    loop {
        match reader.read_event().map_err(|e| e.to_string())? {
            Event::Eof => break,
            Event::Start(e) => match e.local_name().as_ref() {
                b"div" => divs.push(begin_of(&e)),
                b"p" => {
                    let begin = begin_of(&e).or_else(|| divs.iter().rev().find_map(|d| *d));
                    current = Some((begin, String::new()));
                }
                _ => {}
            },
            Event::End(e) => match e.local_name().as_ref() {
                b"div" => {
                    divs.pop();
                }
                b"p" => {
                    if let Some(line) = current.take() {
                        lines.push(line);
                    }
                }
                _ => {}
            },
            Event::Text(text) => {
                if let Some((_, buf)) = current.as_mut() {
                    buf.push_str(&text.xml10_content().map_err(|e| e.to_string())?);
                }
            }
            Event::GeneralRef(entity) => {
                if let Some((_, buf)) = current.as_mut() {
                    buf.push_str(&resolve_entity(&entity)?);
                }
            }
            _ => {}
        }
    }

    let timed: Vec<TimedLine> = lines
        .iter()
        .filter_map(|(begin, text)| {
            let text = squash(text);
            let time_ms = (*begin)?;
            (!text.is_empty()).then_some(TimedLine { time_ms, text })
        })
        .collect();
    if !timed.is_empty() {
        return Ok(Lyrics::synced(source, timed));
    }
    // No timing anywhere: the words are still worth showing as plain text.
    let plain: Vec<String> = lines.into_iter().map(|(_, text)| squash(&text)).collect();
    Ok(Lyrics::plain(source, plain))
}

fn begin_of(element: &BytesStart) -> Option<u64> {
    let attr = element.try_get_attribute("begin").ok()??;
    parse_time(std::str::from_utf8(&attr.value).ok()?)
}

fn resolve_entity(entity: &BytesRef) -> Result<String, String> {
    if let Some(c) = entity.resolve_char_ref().map_err(|e| e.to_string())? {
        return Ok(c.to_string());
    }
    let name = entity.decode().map_err(|e| e.to_string())?;
    Ok(match name.as_ref() {
        "amp" => "&".to_owned(),
        "lt" => "<".to_owned(),
        "gt" => ">".to_owned(),
        "quot" => "\"".to_owned(),
        "apos" => "'".to_owned(),
        other => format!("&{other};"),
    })
}

/// Collapse every run of whitespace to one space, and trim. Line breaks inside a `<p>` are layout,
/// not lyric.
fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A TTML clock value as milliseconds: `hh:mm:ss.fff`, `mm:ss.fff`, `12.5s`, `1500ms` or `12.5`.
fn parse_time(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(ms) = value.strip_suffix("ms") {
        return ms.trim().parse::<f64>().ok().map(|v| v.round() as u64);
    }
    let seconds = if let Some(s) = value.strip_suffix('s') {
        s.trim().parse::<f64>().ok()?
    } else if value.contains(':') {
        let parts: Vec<&str> = value.split(':').collect();
        let (hours, minutes, secs) = match parts.as_slice() {
            [m, s] => (0.0, m.parse::<f64>().ok()?, s.parse::<f64>().ok()?),
            [h, m, s] => (
                h.parse::<f64>().ok()?,
                m.parse::<f64>().ok()?,
                s.parse::<f64>().ok()?,
            ),
            _ => return None,
        };
        hours * 3600.0 + minutes * 60.0 + secs
    } else {
        value.parse::<f64>().ok()?
    };
    (seconds >= 0.0 && seconds.is_finite()).then(|| (seconds * 1000.0).round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Body;

    const SAMPLE: &str = include_str!("../tests/fixtures/boidu_sample.ttml");

    fn synced(xml: &str) -> Vec<(u64, String)> {
        match parse(Source::Boidu, xml).unwrap().expect("has text").body {
            Body::Synced(lines) => lines.into_iter().map(|l| (l.time_ms, l.text)).collect(),
            other => panic!("expected synced, got {other:?}"),
        }
    }

    #[test]
    fn reads_a_real_word_timed_document() {
        let lyrics = parse(Source::Boidu, SAMPLE).unwrap().unwrap();
        assert_eq!(lyrics.source, Source::Boidu);
        let Body::Synced(lines) = lyrics.body else {
            panic!("the sample is word timed");
        };
        assert_eq!(
            lines[0],
            TimedLine {
                time_ms: 0,
                text: "Is this the real life?".into()
            }
        );
        assert_eq!(lines.len(), 7);
        assert!(lines.windows(2).all(|w| w[0].time_ms <= w[1].time_ms));
        // Background vocals are part of the line, parentheses and all.
        assert!(
            lines.iter().any(|l| l.text.contains("(Ooh, poor)")),
            "{lines:?}"
        );
    }

    #[test]
    fn spaces_between_word_spans_survive() {
        let lines = synced(
            r#"<tt><body><div><p begin="1"><span begin="1">Is</span> <span begin="2">this</span></p><p begin="3">ok</p></div></body></tt>"#,
        );
        assert_eq!(lines, vec![(1_000, "Is this".into()), (3_000, "ok".into())]);
    }

    #[test]
    fn entities_are_resolved() {
        let lines = synced(
            r#"<tt><body><div><p begin="1">Rock &amp; roll &#39;n&#39; &apos;x&apos; &lt;3</p><p begin="2">two</p></div></body></tt>"#,
        );
        assert_eq!(
            lines,
            vec![
                (1_000, "Rock & roll 'n' 'x' <3".into()),
                (2_000, "two".into())
            ]
        );
    }

    #[test]
    fn a_line_without_begin_takes_its_div() {
        let lines = synced(
            r#"<tt><body><div begin="10"><p>early</p></div><div begin="20"><p begin="21.5">own</p></div></body></tt>"#,
        );
        assert_eq!(
            lines,
            vec![(10_000, "early".into()), (21_500, "own".into())]
        );
    }

    #[test]
    fn clock_forms() {
        assert_eq!(parse_time("0.000"), Some(0));
        assert_eq!(parse_time("5:54.320"), Some(354_320));
        assert_eq!(parse_time("1:02:03.5"), Some(3_723_500));
        assert_eq!(parse_time("1.5s"), Some(1_500));
        assert_eq!(parse_time("1500ms"), Some(1_500));
        assert_eq!(parse_time("-1"), None);
        assert_eq!(parse_time("soon"), None);
    }

    #[test]
    fn untimed_text_is_plain() {
        let lyrics = parse(
            Source::Boidu,
            r#"<tt><body><div><p>one</p><p>two</p></div></body></tt>"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(lyrics.body, Body::Plain(vec!["one".into(), "two".into()]));
    }

    #[test]
    fn empty_document_has_no_lyrics() {
        assert_eq!(parse(Source::Boidu, "<tt><body/></tt>").unwrap(), None);
    }

    #[test]
    fn malformed_xml_is_an_error_not_a_miss() {
        assert!(parse(Source::Boidu, "<tt><body><p begin=\"1\">oops</div>").is_err());
    }
}
