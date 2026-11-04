// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Boidu: the Better Lyrics API (`lyrics-api.boidu.dev`). Apple Music's word-timed lyrics, served
//! from a cache that is filled by the people who run it.
//!
//! Only the public path is used. We have no `X-API-Key`, and the key Better Lyrics itself uses is
//! theirs alone, so it is never sent or requested. Without a key Boidu answers a cache hit and
//! refuses a miss with 401. A 401 therefore means "not cached under this key", not "no lyrics":
//! the caller moves on to LRCLIB. Duration is part of Boidu's cache key, so a query without one
//! cannot hit at all and is not sent.
//!
//! Boidu rate limits by IP (`429`, or `503` in cache-only mode), and says how long to wait in
//! `Retry-After`. That comes back as [`Error::Throttled`] so the caller can stop asking for a while.

use std::time::Duration;

use reqwest::{header::RETRY_AFTER, StatusCode};
use serde_json::Value;

use crate::error::Error;
use crate::model::{Lyrics, Query, Source};
use crate::ttml;

const TIMEOUT: Duration = Duration::from_secs(8);
/// Used when a throttled answer gives no `Retry-After`.
const DEFAULT_WAIT: Duration = Duration::from_secs(60);
/// Never wait longer than this on the server's word alone.
const MAX_WAIT: Duration = Duration::from_secs(300);

/// Fetch Boidu's lyric for `query`. `Ok(None)` when Boidu has none or does not have it cached.
pub async fn fetch(
    http: &reqwest::Client,
    root: &str,
    query: &Query,
) -> Result<Option<Lyrics>, Error> {
    let Some(length) = query.length() else {
        return Ok(None);
    };
    let seconds = length.round() as i64;

    // The cache key is title, artist, album and duration. A track may be cached with its album or
    // without one, so try the album form first and the bare form only after a 401.
    let forms: Vec<Option<&str>> = match query.album.as_deref() {
        Some(album) => vec![Some(album), None],
        None => vec![None],
    };
    for album in forms {
        let mut params = vec![
            ("s", query.title.clone()),
            ("a", query.artist.clone()),
            ("d", seconds.to_string()),
        ];
        if let Some(album) = album {
            params.push(("al", album.to_owned()));
        }
        let response = http
            .get(root)
            .query(&params)
            .header(reqwest::header::USER_AGENT, crate::USER_AGENT)
            .timeout(TIMEOUT)
            .send()
            .await?;
        match response.status() {
            StatusCode::OK => return read(response.json::<Value>().await?),
            StatusCode::UNAUTHORIZED => continue,
            StatusCode::NOT_FOUND => return Ok(None),
            StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE => {
                return Err(Error::Throttled("Boidu", retry_after(&response)));
            }
            status => return Err(Error::Status("Boidu", status.as_u16())),
        }
    }
    Ok(None)
}

fn read(answer: Value) -> Result<Option<Lyrics>, Error> {
    let ttml = answer
        .get("ttml")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Malformed("Boidu", "no ttml field".to_owned()))?;
    ttml::parse(Source::Boidu, ttml).map_err(|e| Error::Malformed("Boidu", e))
}

fn retry_after(response: &reqwest::Response) -> Duration {
    response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_WAIT)
        .min(MAX_WAIT)
}
