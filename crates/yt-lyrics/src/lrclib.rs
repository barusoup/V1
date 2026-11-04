// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! LRCLIB: the free, keyless lyrics catalogue (<https://lrclib.net/docs>). The fallback behind Boidu.
//!
//! The order is chosen so a synced answer is found whenever one exists, without ever taking a
//! different cut of the song:
//! 1. `/api/get`, an exact match on title, artist, album and length. Its plain text is kept, but a
//!    synced copy may still exist under another record, so it does not stop here.
//! 2. `/api/search` on title and artist, keeping only candidates within [`DURATION_TOLERANCE`]
//!    seconds of ours. The closest synced one wins.
//! 3. Free-text search, with the title cleaned of YouTube decoration. Only when a length is known,
//!    since free text is the looser match.

use std::time::Duration;

use reqwest::StatusCode;
use serde::Deserialize;

use crate::error::Error;
use crate::lrc;
use crate::model::{Lyrics, Query, Source};

/// How far a candidate's length may differ from ours. Wider and it can be a different cut.
pub const DURATION_TOLERANCE: f64 = 5.0;

const TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Track {
    #[serde(default)]
    pub instrumental: bool,
    #[serde(default)]
    pub plain_lyrics: Option<String>,
    #[serde(default)]
    pub synced_lyrics: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
}

impl Track {
    fn has_synced(&self) -> bool {
        self.synced_lyrics
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
    }

    /// The best lyric this one record holds: instrumental, then synced, then plain.
    pub fn lyrics(&self) -> Option<Lyrics> {
        if self.instrumental {
            return Some(Lyrics::instrumental(Source::Lrclib));
        }
        if self.has_synced() {
            if let Some(lyrics) = lrc::parse(Source::Lrclib, self.synced_lyrics.as_deref()?) {
                return Some(lyrics);
            }
        }
        let plain = self.plain_lyrics.as_deref()?;
        Lyrics::plain(Source::Lrclib, plain.lines().map(str::to_owned).collect())
    }
}

/// Fetch the best lyric LRCLIB has for `query`. `Ok(None)` is a definitive "not in LRCLIB".
/// `Err` means the lookup did not finish, so nothing can be concluded.
pub async fn fetch(
    http: &reqwest::Client,
    root: &str,
    query: &Query,
) -> Result<Option<Lyrics>, Error> {
    let length = query.length();

    let mut exact_params = vec![
        ("track_name", query.title.clone()),
        ("artist_name", query.artist.clone()),
    ];
    if let Some(album) = &query.album {
        exact_params.push(("album_name", album.clone()));
    }
    if let Some(d) = length {
        exact_params.push(("duration", d.round().to_string()));
    }
    let exact = get_exact(http, root, &exact_params).await?;

    let mut plain: Option<Lyrics> = None;
    if let Some(track) = &exact {
        if let Some(l) = track.lyrics() {
            if l.settles() {
                return Ok(Some(l));
            }
            plain = Some(l);
        }
    }

    // Title and artist first. If that search itself fails, an exact plain match is still a
    // real answer; without one there is nothing to conclude.
    let mut candidates = match search(
        http,
        root,
        &[
            ("track_name", query.title.clone()),
            ("artist_name", query.artist.clone()),
        ],
    )
    .await
    {
        Ok(found) => found,
        Err(error) => {
            return match plain {
                Some(l) => Ok(Some(l)),
                None => Err(error),
            }
        }
    };
    if let Some(track) = best_candidate(&candidates, length, true) {
        return Ok(track.lyrics().or(plain));
    }

    // Free text only with a known length: it is the looser match, and the length is what keeps it
    // on the right song. A failed free-text search does not undo what the earlier ones found.
    if length.is_some() {
        let cleaned = clean_title(&query.title);
        if !cleaned.is_empty() {
            for q in [format!("{cleaned} {}", query.artist), cleaned] {
                if let Ok(more) = search(http, root, &[("q", q)]).await {
                    candidates.extend(more);
                }
            }
            if let Some(track) = best_candidate(&candidates, length, true) {
                return Ok(track.lyrics().or(plain));
            }
        }
    }

    if plain.is_some() {
        return Ok(plain);
    }
    // Nothing synced anywhere: a plain text within the length window, if there is one.
    Ok(best_candidate(&candidates, length, false).and_then(Track::lyrics))
}

async fn get_exact(
    http: &reqwest::Client,
    root: &str,
    params: &[(&str, String)],
) -> Result<Option<Track>, Error> {
    let response = http
        .get(format!("{root}/get"))
        .query(params)
        .header(reqwest::header::USER_AGENT, crate::USER_AGENT)
        .timeout(TIMEOUT)
        .send()
        .await?;
    match response.status() {
        StatusCode::NOT_FOUND => Ok(None),
        status if status.is_success() => Ok(Some(response.json().await?)),
        status => Err(Error::Status("LRCLIB", status.as_u16())),
    }
}

async fn search(
    http: &reqwest::Client,
    root: &str,
    params: &[(&str, String)],
) -> Result<Vec<Track>, Error> {
    let response = http
        .get(format!("{root}/search"))
        .query(params)
        .header(reqwest::header::USER_AGENT, crate::USER_AGENT)
        .timeout(TIMEOUT)
        .send()
        .await?;
    match response.status() {
        // No results is an answer, not a failure.
        StatusCode::NOT_FOUND => Ok(Vec::new()),
        status if status.is_success() => Ok(response.json().await?),
        status => Err(Error::Status("LRCLIB", status.as_u16())),
    }
}

/// The candidate to use. With a known length, only ones within tolerance qualify; a candidate with
/// no length is not trusted then. Synced beats plain. `want_synced` restricts the answer to synced
/// candidates, which is how the free-text passes avoid taking a plain match over a synced one.
pub(crate) fn best_candidate(
    candidates: &[Track],
    length: Option<f64>,
    want_synced: bool,
) -> Option<&Track> {
    let close = |t: &&Track| match length {
        None => true,
        Some(ours) => t
            .duration
            .is_some_and(|theirs| (theirs - ours).abs() <= DURATION_TOLERANCE),
    };
    let distance = |t: &Track| length.and_then(|ours| t.duration.map(|d| (d - ours).abs()));

    let mut best_synced: Option<(f64, &Track)> = None;
    let mut first_plain: Option<&Track> = None;
    for track in candidates.iter().filter(close) {
        if track.has_synced() {
            let d = distance(track).unwrap_or(0.0);
            if best_synced.is_none_or(|(bd, _)| d < bd) {
                best_synced = Some((d, track));
            }
        } else if first_plain.is_none() {
            first_plain = Some(track);
        }
    }
    match best_synced {
        Some((_, track)) => Some(track),
        None if want_synced => None,
        None => first_plain,
    }
}

/// A YouTube title without what no catalogue files a song under: everything after a `|`, and every
/// bracketed aside, such as "(Official Video)", "[English]" or "【MV】".
pub fn clean_title(title: &str) -> String {
    let head = title.split('|').next().unwrap_or_default();
    let mut depth = 0u32;
    let mut out = String::new();
    for c in head.chars() {
        match c {
            '(' | '[' | '【' | '（' => depth += 1,
            ')' | ']' | '】' | '）' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Body;

    fn track(duration: f64, synced: Option<&str>, plain: Option<&str>) -> Track {
        Track {
            instrumental: false,
            plain_lyrics: plain.map(str::to_owned),
            synced_lyrics: synced.map(str::to_owned),
            duration: Some(duration),
        }
    }

    #[test]
    fn clean_title_drops_asides_and_suffixes() {
        assert_eq!(
            clean_title("Bad Apple／ Lizz Robinett (English Cover) | Lyrics/Lyric Video [English]"),
            "Bad Apple／ Lizz Robinett"
        );
        assert_eq!(clean_title("Song Name (Official Video)"), "Song Name");
        assert_eq!(clean_title("Title 【MV】 extra"), "Title extra");
        assert_eq!(clean_title("(only an aside)"), "");
    }

    #[test]
    fn the_closest_synced_candidate_wins_over_a_nearer_plain_one() {
        let candidates = vec![
            track(200.0, None, Some("plain at 200")),
            track(203.0, Some("[00:01.00]synced at 203"), None),
            track(199.0, Some("[00:01.00]synced at 199"), None),
        ];
        let picked = best_candidate(&candidates, Some(200.0), true).unwrap();
        assert_eq!(picked.duration, Some(199.0));
    }

    #[test]
    fn candidates_outside_the_window_are_ignored() {
        let candidates = vec![track(230.0, Some("[00:01.00]another cut"), None)];
        assert!(best_candidate(&candidates, Some(200.0), true).is_none());
        assert!(best_candidate(&candidates, Some(200.0), false).is_none());
    }

    #[test]
    fn without_a_length_the_first_synced_candidate_is_taken() {
        let candidates = vec![
            track(10.0, None, Some("plain")),
            track(999.0, Some("[00:01.00]first synced"), None),
        ];
        let picked = best_candidate(&candidates, None, true).unwrap();
        assert_eq!(picked.duration, Some(999.0));
    }

    #[test]
    fn plain_is_used_only_when_allowed() {
        let candidates = vec![track(200.0, None, Some("only plain"))];
        assert!(best_candidate(&candidates, Some(200.0), true).is_none());
        assert_eq!(
            best_candidate(&candidates, Some(200.0), false)
                .unwrap()
                .duration,
            Some(200.0)
        );
    }

    #[test]
    fn a_track_yields_instrumental_then_synced_then_plain() {
        let instrumental = Track {
            instrumental: true,
            plain_lyrics: None,
            synced_lyrics: None,
            duration: None,
        };
        assert_eq!(instrumental.lyrics().unwrap().body, Body::Instrumental);

        let synced = track(1.0, Some("[00:01.00]a\n[00:02.00]b"), Some("a\nb"));
        assert!(synced.lyrics().unwrap().is_synced());

        // A flat synced text is plain, not a timeline that never moves.
        let flat = track(1.0, Some("[00:00.00]a\n[00:00.00]b"), None);
        assert_eq!(
            flat.lyrics().unwrap().body,
            Body::Plain(vec!["a".into(), "b".into()])
        );

        let plain_only = track(1.0, Some("   "), Some("verse\n\nchorus"));
        assert_eq!(
            plain_only.lyrics().unwrap().body,
            Body::Plain(vec!["verse".into(), "".into(), "chorus".into()])
        );
    }
}
