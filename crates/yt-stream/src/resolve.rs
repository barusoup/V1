// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Turning a video id into a playable audio URL.
//!
//! Order of work, and why:
//! 1. Cache first. A URL with enough life left needs no network at all.
//! 2. Skip any client that is known to fail for lack of a prerequisite (signature timestamp,
//!    PoToken). Calling it would only return UNPLAYABLE.
//! 3. Try the clients in order. For each one: `/player`, pick a format, make the URL playable,
//!    then HEAD it with the headers the player will send. The first URL that passes wins.
//! 4. Cache the winner with its own expiry, so the next play skips steps 2 to 4.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use yt_client::{Auth, ClientKind, ClientSpec, InnerTube, LoginUse, PlayerRequest, PoTokenNeed};

use crate::cache::UrlCache;
use crate::pick::{pick_audio, Quality};
use crate::probe::Prober;
use crate::scripts::PlayerScripts;

/// What a successful resolve hands to the playback layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resolved {
    pub video_id: String,
    /// The URL to fetch audio from.
    pub url: String,
    /// Headers the URL needs. The player must send these.
    pub headers: Vec<(String, String)>,
    /// The identity that produced the URL.
    pub client: ClientKind,
    pub itag: u32,
    pub mime_type: String,
    /// Total size in bytes, when YouTube says.
    pub content_length: Option<u64>,
    /// After this Unix time (seconds) the URL is dead.
    pub expires_at_unix: u64,
    /// YouTube's loudness measurement, for volume normalisation.
    pub loudness_db: Option<f64>,
    /// URL to ping so the play counts in the account's watch history.
    pub tracking_ping: Option<String>,
    pub title: Option<String>,
    pub author: Option<String>,
    pub length_seconds: Option<u64>,
    pub thumbnail: Option<String>,
}

/// Why one client in the chain did not produce a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attempt {
    pub client: ClientKind,
    pub reason: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("no playable stream for {video_id}")]
    NoStream {
        video_id: String,
        tried: Vec<Attempt>,
    },

    /// `/player` says the id is a video, not a song. The app plays songs only.
    #[error("{video_id} is a video ({kind}), not a song")]
    IsVideo { video_id: String, kind: String },

    #[error(transparent)]
    Client(#[from] yt_client::Error),
}

#[derive(Debug, Clone)]
pub struct ResolverConfig {
    pub quality: Quality,
    /// A cached URL with less life left than this is fetched again.
    pub min_remaining: Duration,
    /// Clients to try, in order.
    pub chain: Vec<ClientKind>,
    /// Whether stream requests carry the account.
    pub auth: Auth,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        ResolverConfig {
            quality: Quality::High,
            min_remaining: Duration::from_secs(120),
            chain: ClientSpec::CHAIN.to_vec(),
            auth: Auth::Anonymous,
        }
    }
}

/// Used when YouTube does not say how long a URL lasts. Real URLs last about six hours.
const FALLBACK_LIFETIME: Duration = Duration::from_secs(6 * 3600);

pub struct Resolver<S, P, C> {
    tube: InnerTube,
    scripts: S,
    prober: P,
    cache: C,
    config: ResolverConfig,
}

impl<S: PlayerScripts, P: Prober, C: UrlCache> Resolver<S, P, C> {
    pub fn new(tube: InnerTube, scripts: S, prober: P, cache: C, config: ResolverConfig) -> Self {
        Resolver {
            tube,
            scripts,
            prober,
            cache,
            config,
        }
    }

    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// A playable stream for `video_id`.
    pub async fn resolve(&self, video_id: &str) -> Result<Resolved, ResolveError> {
        let now = now_unix();
        if let Some(hit) = self.cache.get(video_id) {
            if hit.expires_at_unix > now + self.config.min_remaining.as_secs() {
                tracing::debug!(video_id, "stream URL from cache");
                return Ok(hit);
            }
            self.cache.remove(video_id);
        }

        let session = self.tube.session.snapshot();
        let sts = self.scripts.signature_timestamp().await;
        let pot = match session.visitor_data.as_deref() {
            Some(v) => self.scripts.po_token(v).await,
            None => None,
        };

        let mut tried = Vec::new();
        let chain = self.config.chain.clone();
        for &kind in &chain {
            let spec = ClientSpec::of(kind);
            let needs_pot = match spec.po_token {
                PoTokenNeed::Never => false,
                PoTokenNeed::Always => true,
                // Only a signed-in Premium account is exempt, and this client has no account
                // path yet, so treat it as required.
                PoTokenNeed::UnlessPremium => true,
            };

            if spec.needs_sts && sts.is_none() {
                tried.push(attempt(kind, "no signature timestamp"));
                continue;
            }
            if needs_pot && pot.is_none() {
                tried.push(attempt(kind, "no PoToken"));
                continue;
            }

            let req = PlayerRequest {
                video_id,
                playlist_id: None,
                signature_timestamp: if spec.needs_sts { sts } else { None },
                po_token: if needs_pot { pot.as_deref() } else { None },
            };
            let resp = match self.tube.player(kind, self.config.auth, req).await {
                Ok(r) => r,
                Err(e) => {
                    tried.push(attempt(kind, &format!("request failed: {e}")));
                    continue;
                }
            };

            if !resp.playability_status.is_ok() {
                let status = &resp.playability_status.status;
                let reason = resp.playability_status.reason.clone().unwrap_or_default();
                tried.push(attempt(kind, &format!("{status}: {reason}")));
                continue;
            }

            // A backstop behind the parsers, which never hand out a video id: if `/player` says
            // this is a video, stop here rather than try other clients.
            if let Some(details) = resp.video_details.as_ref().filter(|d| d.is_video()) {
                return Err(ResolveError::IsVideo {
                    video_id: video_id.to_owned(),
                    kind: details.music_video_type.clone().unwrap_or_default(),
                });
            }

            let Some(streaming) = resp.streaming_data.as_ref() else {
                tried.push(attempt(kind, "no streaming data"));
                continue;
            };
            let formats = if streaming.adaptive_formats.is_empty() {
                &streaming.formats
            } else {
                &streaming.adaptive_formats
            };
            let Some(format) = pick_audio(formats, self.config.quality) else {
                tried.push(attempt(kind, "no audio format"));
                continue;
            };
            let Some(mut url) = self.scripts.playable_url(format, kind).await else {
                tried.push(attempt(kind, "URL could not be made playable"));
                continue;
            };
            if needs_pot {
                if let Some(token) = &pot {
                    url = with_query(&url, "pot", token);
                }
            }

            let headers = stream_headers(&spec, &session);
            if !self.prober.probe(&url, &headers).await {
                tried.push(attempt(kind, "URL refused on HEAD"));
                continue;
            }

            let lifetime = streaming
                .expires_in_seconds
                .map(Duration::from_secs)
                .unwrap_or(FALLBACK_LIFETIME);
            let details = resp.video_details.as_ref();
            let resolved = Resolved {
                video_id: video_id.to_owned(),
                url,
                headers,
                client: kind,
                itag: format.itag,
                mime_type: format.mime_type.clone(),
                content_length: format.content_length,
                expires_at_unix: now + lifetime.as_secs(),
                loudness_db: resp.loudness_db(),
                tracking_ping: resp.tracking_ping().map(str::to_owned),
                title: details.and_then(|d| d.title.clone()),
                author: details.and_then(|d| d.author.clone()),
                length_seconds: details.and_then(|d| d.length_seconds),
                thumbnail: details
                    .and_then(|d| d.thumbnail.as_ref())
                    .and_then(|t| t.thumbnails.last())
                    .map(|t| t.url.clone()),
            };
            tracing::info!(video_id, client = ?kind, itag = resolved.itag, "resolved");
            self.cache.put(resolved.clone());
            return Ok(resolved);
        }

        self.cache.sweep(now);
        Err(ResolveError::NoStream {
            video_id: video_id.to_owned(),
            tried,
        })
    }
}

fn attempt(client: ClientKind, reason: &str) -> Attempt {
    Attempt {
        client,
        reason: reason.to_owned(),
    }
}

/// The headers a stream request must carry for `spec`: the identity's user agent, and the cookie
/// only where the identity supports login and the session has one.
fn stream_headers(spec: &ClientSpec, session: &yt_client::Session) -> Vec<(String, String)> {
    let mut headers = vec![("User-Agent".to_owned(), spec.user_agent.to_owned())];
    if spec.login == LoginUse::Supported {
        if let Some(jar) = &session.cookies {
            headers.push(("Cookie".to_owned(), jar.header()));
        }
    }
    headers
}

fn with_query(url: &str, key: &str, value: &str) -> String {
    let sep = if url.contains('?') { '&' } else { '?' };
    format!("{url}{sep}{key}={value}")
}

pub(crate) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
