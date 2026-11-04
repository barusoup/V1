// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! What a lookup gives back, whichever provider answered, and what it is asked for.

use serde::{Deserialize, Serialize};

/// Where a lyric came from. [`Source::name`] is the attribution the UI shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Boidu,
    Lrclib,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Boidu => "Boidu",
            Source::Lrclib => "LRCLIB",
        }
    }
}

/// One line of synced lyrics. An empty `text` is a pause, such as an instrumental break.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimedLine {
    pub time_ms: u64,
    pub text: String,
}

/// The shape of a lyric. Each variant is drawn differently, so the UI matches on this.
// Adjacent tagging: internally tagged enums cannot hold a sequence, and both text variants do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "lines", rename_all = "lowercase")]
pub enum Body {
    /// Sorted by `time_ms`. The panel follows the line the playhead is in.
    Synced(Vec<TimedLine>),
    /// Untimed, top to bottom. A blank entry is a stanza break.
    Plain(Vec<String>),
    /// The song has no words. That is an answer, not a missing lyric.
    Instrumental,
}

/// A lyric and where it came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lyrics {
    pub source: Source,
    pub body: Body,
}

impl Lyrics {
    pub fn is_synced(&self) -> bool {
        matches!(self.body, Body::Synced(_))
    }

    /// A result that no lower-priority provider can beat: synced words, or an instrumental verdict.
    pub fn settles(&self) -> bool {
        matches!(self.body, Body::Synced(_) | Body::Instrumental)
    }

    pub fn instrumental(source: Source) -> Lyrics {
        Lyrics {
            source,
            body: Body::Instrumental,
        }
    }

    /// Timed lines, sorted. `None` when there is no text at all.
    ///
    /// Lines whose text is all at one moment have no timing to follow, so they become plain. That
    /// happens with untimed text that a provider stamps `0` on every line, and with LRC files that
    /// stamp every line `[00:00.00]`. Shown as synced, the panel would never move.
    pub fn synced(source: Source, mut lines: Vec<TimedLine>) -> Option<Lyrics> {
        if lines.iter().all(|line| line.text.trim().is_empty()) {
            return None;
        }
        lines.sort_by_key(|line| line.time_ms);
        let mut sung = lines.iter().filter(|line| !line.text.trim().is_empty());
        let first = sung.next().map(|line| line.time_ms);
        if sung.all(|line| Some(line.time_ms) == first) {
            return Lyrics::plain(source, lines.into_iter().map(|line| line.text).collect());
        }
        Some(Lyrics {
            source,
            body: Body::Synced(lines),
        })
    }

    /// Untimed text. Leading and trailing blank lines are dropped; blank lines in the middle stay
    /// as stanza breaks. `None` when nothing is left.
    pub fn plain(source: Source, lines: Vec<String>) -> Option<Lyrics> {
        let mut lines: Vec<String> = lines.into_iter().map(|l| l.trim_end().to_owned()).collect();
        while lines.last().is_some_and(|l| l.trim().is_empty()) {
            lines.pop();
        }
        let start = lines.iter().position(|l| !l.trim().is_empty())?;
        lines.drain(..start);
        Some(Lyrics {
            source,
            body: Body::Plain(lines),
        })
    }
}

/// The line the playhead is in: the last one that has started. `None` before the first line.
/// `lines` must be sorted by time, which [`Lyrics::synced`] guarantees.
pub fn active_line(lines: &[TimedLine], position_ms: u64) -> Option<usize> {
    lines
        .partition_point(|line| line.time_ms <= position_ms)
        .checked_sub(1)
}

/// What is known about a track. The length is what keeps fuzzy matching honest: without it, a
/// lookup can land on a different cut of the same song, with lyrics seconds off the audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// The YouTube video id. Keys the on-disk cache.
    pub video_id: String,
    pub title: String,
    /// The primary artist. Multi-artist strings like "A, B" miss exact matches elsewhere.
    pub artist: String,
    pub album: Option<String>,
    pub duration_secs: Option<f64>,
}

impl Query {
    /// The track length, when it is known and usable.
    pub fn length(&self) -> Option<f64> {
        self.duration_secs.filter(|d| d.is_finite() && *d > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(time_ms: u64, text: &str) -> TimedLine {
        TimedLine {
            time_ms,
            text: text.to_owned(),
        }
    }

    #[test]
    fn one_cue_is_plain_not_synced() {
        let lyrics = Lyrics::synced(
            Source::Lrclib,
            vec![line(0, "a"), line(0, "b"), line(0, "c")],
        )
        .unwrap();
        assert_eq!(
            lyrics.body,
            Body::Plain(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn synced_sorts_and_keeps_pauses() {
        let lyrics = Lyrics::synced(
            Source::Boidu,
            vec![line(5000, "later"), line(1000, ""), line(2000, "first")],
        )
        .unwrap();
        assert_eq!(
            lyrics.body,
            Body::Synced(vec![
                line(1000, ""),
                line(2000, "first"),
                line(5000, "later")
            ])
        );
    }

    #[test]
    fn no_text_is_no_lyrics() {
        assert!(Lyrics::synced(Source::Boidu, vec![line(1000, "  ")]).is_none());
        assert!(Lyrics::plain(Source::Lrclib, vec!["".into(), " ".into()]).is_none());
    }

    #[test]
    fn plain_trims_outer_blanks_only() {
        let lyrics = Lyrics::plain(
            Source::Lrclib,
            vec![
                "".into(),
                "verse".into(),
                "".into(),
                "chorus".into(),
                "  ".into(),
            ],
        )
        .unwrap();
        assert_eq!(
            lyrics.body,
            Body::Plain(vec!["verse".into(), "".into(), "chorus".into()])
        );
    }

    #[test]
    fn active_line_is_the_last_one_started() {
        let lines = vec![
            line(1000, "a"),
            line(3000, "b"),
            line(3000, "b2"),
            line(9000, "c"),
        ];
        assert_eq!(active_line(&lines, 0), None);
        assert_eq!(active_line(&lines, 999), None);
        assert_eq!(active_line(&lines, 1000), Some(0));
        assert_eq!(active_line(&lines, 3000), Some(2));
        assert_eq!(active_line(&lines, 500_000), Some(3));
    }

    #[test]
    fn length_must_be_positive_and_finite() {
        let mut q = Query {
            video_id: "v".into(),
            title: "t".into(),
            artist: "a".into(),
            album: None,
            duration_secs: Some(0.0),
        };
        assert_eq!(q.length(), None);
        q.duration_secs = Some(f64::NAN);
        assert_eq!(q.length(), None);
        q.duration_secs = Some(215.4);
        assert_eq!(q.length(), Some(215.4));
    }
}
