// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The lookup: Boidu, then LRCLIB, with the cache and the rate-limit pause around them.
//!
//! The ranking, in order:
//! 1. A synced lyric (or an instrumental verdict) from either provider wins. Boidu is asked first,
//!    so when it has the song it is the one shown.
//! 2. Otherwise plain text, from Boidu's untimed TTML or from LRCLIB.
//! 3. Otherwise "not found". This is only concluded from LRCLIB, since Boidu's refusal can mean
//!    "not cached". And if Boidu failed rather than refused, the "not found" is not cached, because
//!    Boidu may well have the song once it is reachable again.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::boidu;
use crate::cache::{now_unix, Cache};
use crate::error::Error;
use crate::lrclib;
use crate::model::{Lyrics, Query};

/// The whole lookup, both providers together. Past this the answer is "unavailable".
const BUDGET: Duration = Duration::from_secs(25);

#[derive(Debug, Clone, PartialEq)]
pub enum Lookup {
    Found(Lyrics),
    /// Looked up and not there. Remembered for a day when the length was known.
    NotFound,
    /// Could not be looked up: offline, rate limited, a bad answer. Never remembered.
    Unavailable(String),
}

#[derive(Debug, Clone)]
pub struct Endpoints {
    pub boidu: String,
    pub lrclib: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Endpoints {
            boidu: "https://lyrics-api.boidu.dev/getLyrics".to_owned(),
            lrclib: "https://lrclib.net/api".to_owned(),
        }
    }
}

/// What one provider said.
#[derive(Debug)]
pub(crate) enum Attempt {
    Hit(Lyrics),
    /// It answered, and it has nothing.
    Miss,
    /// It did not answer. The message is for logs and the status line.
    Failed(String),
}

/// What the two providers add up to.
#[derive(Debug, PartialEq)]
pub(crate) enum Outcome {
    Found(Lyrics),
    /// `cacheable` is false when Boidu failed: it may still have the song.
    Missing {
        cacheable: bool,
    },
    Failed(String),
}

/// Rank two providers' answers. Boidu's comes first in the list, so it is preferred on a tie.
pub(crate) fn combine(boidu: Attempt, lrclib: Attempt) -> Outcome {
    let boidu_failed = matches!(boidu, Attempt::Failed(_));
    let mut plain = None;
    if let Attempt::Hit(lyrics) = boidu {
        if lyrics.settles() {
            return Outcome::Found(lyrics);
        }
        plain = Some(lyrics);
    }
    match lrclib {
        Attempt::Hit(lyrics) if lyrics.settles() => Outcome::Found(lyrics),
        Attempt::Hit(lyrics) => Outcome::Found(plain.unwrap_or(lyrics)),
        Attempt::Miss => match plain {
            Some(lyrics) => Outcome::Found(lyrics),
            None => Outcome::Missing {
                cacheable: !boidu_failed,
            },
        },
        Attempt::Failed(error) => match plain {
            Some(lyrics) => Outcome::Found(lyrics),
            None => Outcome::Failed(error),
        },
    }
}

/// Write an outcome to the cache on the blocking pool, so the file I/O never holds an async worker.
async fn store(cache: &Cache, video_id: &str, lyrics: Option<Lyrics>) {
    let (cache, video_id) = (cache.clone(), video_id.to_owned());
    let _ = tokio::task::spawn_blocking(move || {
        cache.put(&video_id, lyrics.as_ref(), now_unix());
    })
    .await;
}

pub struct Service {
    http: reqwest::Client,
    endpoints: Endpoints,
    cache: Option<Cache>,
    boidu_paused_until: Mutex<Option<Instant>>,
}

impl Service {
    /// `cache_dir: None` keeps nothing between lookups.
    pub fn new(http: reqwest::Client, cache_dir: Option<PathBuf>) -> Service {
        Service {
            http,
            endpoints: Endpoints::default(),
            cache: cache_dir.map(Cache::new),
            boidu_paused_until: Mutex::new(None),
        }
    }

    /// Point the providers somewhere else. For tests.
    pub fn with_endpoints(mut self, endpoints: Endpoints) -> Service {
        self.endpoints = endpoints;
        self
    }

    pub async fn lookup(&self, query: &Query) -> Lookup {
        let cacheable = query.length().is_some();
        if let Some(cache) = &self.cache {
            // The cache is plain file I/O: it runs on the blocking pool, not on an async worker.
            let (cache, video_id) = (cache.clone(), query.video_id.clone());
            let stored = tokio::task::spawn_blocking(move || cache.get(&video_id, now_unix()))
                .await
                .ok()
                .flatten();
            if let Some(entry) = stored {
                return match entry {
                    Some(lyrics) => Lookup::Found(lyrics),
                    None => Lookup::NotFound,
                };
            }
        }

        let outcome = match tokio::time::timeout(BUDGET, self.ask(query)).await {
            Ok(outcome) => outcome,
            Err(_) => Outcome::Failed("the lookup timed out".to_owned()),
        };
        match outcome {
            Outcome::Found(lyrics) => {
                if let (Some(cache), true) = (&self.cache, cacheable) {
                    store(cache, &query.video_id, Some(lyrics.clone())).await;
                }
                Lookup::Found(lyrics)
            }
            Outcome::Missing { cacheable: safe } => {
                if let (Some(cache), true, true) = (&self.cache, cacheable, safe) {
                    store(cache, &query.video_id, None).await;
                }
                Lookup::NotFound
            }
            Outcome::Failed(error) => Lookup::Unavailable(error),
        }
    }

    async fn ask(&self, query: &Query) -> Outcome {
        let boidu = self.ask_boidu(query).await;
        // Synced words from Boidu are final: asking LRCLIB would only cost a request.
        if let Attempt::Hit(lyrics) = &boidu {
            if lyrics.settles() {
                return Outcome::Found(lyrics.clone());
            }
        }
        let lrclib = match lrclib::fetch(&self.http, &self.endpoints.lrclib, query).await {
            Ok(Some(lyrics)) => Attempt::Hit(lyrics),
            Ok(None) => Attempt::Miss,
            Err(error) => Attempt::Failed(error.to_string()),
        };
        combine(boidu, lrclib)
    }

    async fn ask_boidu(&self, query: &Query) -> Attempt {
        if query.length().is_none() {
            // Boidu keys on length, so without one it cannot answer at all.
            return Attempt::Miss;
        }
        if self.boidu_paused() {
            return Attempt::Failed("Boidu is rate limiting".to_owned());
        }
        match boidu::fetch(&self.http, &self.endpoints.boidu, query).await {
            Ok(Some(lyrics)) => Attempt::Hit(lyrics),
            Ok(None) => Attempt::Miss,
            Err(Error::Throttled(_, wait)) => {
                self.pause_boidu(wait);
                Attempt::Failed(format!("Boidu asked us to wait {}s", wait.as_secs()))
            }
            Err(error) => {
                tracing::debug!(%error, "lyrics: Boidu failed");
                Attempt::Failed(error.to_string())
            }
        }
    }

    fn boidu_paused(&self) -> bool {
        let guard = self.boidu_paused_until.lock().unwrap();
        guard.is_some_and(|until| Instant::now() < until)
    }

    fn pause_boidu(&self, wait: Duration) {
        *self.boidu_paused_until.lock().unwrap() = Some(Instant::now() + wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Body, Source, TimedLine};

    fn synced(source: Source) -> Lyrics {
        Lyrics {
            source,
            body: Body::Synced(vec![TimedLine {
                time_ms: 1,
                text: "x".into(),
            }]),
        }
    }

    fn plain(source: Source) -> Lyrics {
        Lyrics {
            source,
            body: Body::Plain(vec!["x".into()]),
        }
    }

    #[test]
    fn boidu_synced_beats_lrclib_synced() {
        let outcome = combine(
            Attempt::Hit(synced(Source::Boidu)),
            Attempt::Hit(synced(Source::Lrclib)),
        );
        assert_eq!(outcome, Outcome::Found(synced(Source::Boidu)));
    }

    #[test]
    fn lrclib_synced_is_the_fallback_when_boidu_has_nothing() {
        let outcome = combine(Attempt::Miss, Attempt::Hit(synced(Source::Lrclib)));
        assert_eq!(outcome, Outcome::Found(synced(Source::Lrclib)));
    }

    #[test]
    fn lrclib_synced_beats_boidu_plain() {
        let outcome = combine(
            Attempt::Hit(plain(Source::Boidu)),
            Attempt::Hit(synced(Source::Lrclib)),
        );
        assert_eq!(outcome, Outcome::Found(synced(Source::Lrclib)));
    }

    #[test]
    fn plain_is_shown_when_nothing_is_synced() {
        let outcome = combine(Attempt::Hit(plain(Source::Boidu)), Attempt::Miss);
        assert_eq!(outcome, Outcome::Found(plain(Source::Boidu)));
    }

    #[test]
    fn an_instrumental_verdict_settles_the_lookup() {
        let outcome = combine(
            Attempt::Hit(plain(Source::Boidu)),
            Attempt::Hit(Lyrics::instrumental(Source::Lrclib)),
        );
        assert_eq!(
            outcome,
            Outcome::Found(Lyrics::instrumental(Source::Lrclib))
        );
    }

    #[test]
    fn a_definitive_miss_is_cacheable_only_when_boidu_did_not_fail() {
        assert_eq!(
            combine(Attempt::Miss, Attempt::Miss),
            Outcome::Missing { cacheable: true }
        );
        assert_eq!(
            combine(Attempt::Failed("busy".into()), Attempt::Miss),
            Outcome::Missing { cacheable: false }
        );
    }

    #[test]
    fn a_failed_lrclib_is_unavailable_unless_boidu_had_plain() {
        assert_eq!(
            combine(Attempt::Miss, Attempt::Failed("offline".into())),
            Outcome::Failed("offline".into())
        );
        assert_eq!(
            combine(
                Attempt::Hit(plain(Source::Boidu)),
                Attempt::Failed("offline".into())
            ),
            Outcome::Found(plain(Source::Boidu))
        );
    }
}
