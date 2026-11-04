// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Checking a stream URL before it is handed to a player.
//!
//! A `HEAD` is not a good predictor any more: googlevideo answers `HEAD` with 403 for a URL whose
//! first MiB is still served, and answers `200` for a URL that dies at byte one million. What the
//! player actually needs is a read *past* the point where googlevideo stops serving un-attested
//! requests, so the probe is a small bounded `GET` at that offset. One round trip, ~64 KiB, and a
//! capped URL turns into a silent fall-through to the next identity.

use std::time::Duration;

use reqwest::StatusCode;

/// Where the probe reads. The un-attested cap sits at the first MiB.
const PROBE_OFFSET: u64 = 2 * 1024 * 1024;
/// How much to ask for. The response body is never read; only the status matters.
const PROBE_LEN: u64 = 64 * 1024;

/// Decides whether a URL is worth handing to a player.
pub trait Prober: Send + Sync {
    fn probe(
        &self,
        url: &str,
        headers: &[(String, String)],
    ) -> impl std::future::Future<Output = bool> + Send;
}

/// Bounded `GET` at [`PROBE_OFFSET`]. Accepts 200, 206, and 416 (the file is shorter than the
/// offset, so there is nothing left to refuse).
#[derive(Clone)]
pub struct HttpProber {
    http: reqwest::Client,
    timeout: Duration,
}

impl HttpProber {
    pub fn new(http: reqwest::Client) -> Self {
        HttpProber {
            http,
            timeout: Duration::from_secs(10),
        }
    }
}

impl Prober for HttpProber {
    async fn probe(&self, url: &str, headers: &[(String, String)]) -> bool {
        let range = format!("bytes={PROBE_OFFSET}-{}", PROBE_OFFSET + PROBE_LEN - 1);
        let mut req = self
            .http
            .get(url)
            .timeout(self.timeout)
            .header("range", range);
        for (name, value) in headers {
            req = req.header(name.as_str(), value.as_str());
        }
        match req.send().await {
            Ok(resp) => {
                let ok = matches!(
                    resp.status(),
                    StatusCode::OK | StatusCode::PARTIAL_CONTENT | StatusCode::RANGE_NOT_SATISFIABLE
                );
                if !ok {
                    tracing::debug!(status = %resp.status(), "stream URL refused past the cap");
                }
                ok
            }
            Err(e) => {
                tracing::debug!(error = %e, "stream URL probe failed");
                false
            }
        }
    }
}
