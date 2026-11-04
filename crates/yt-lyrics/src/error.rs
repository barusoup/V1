// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Duration;

/// Why a provider could not answer. Each is "we do not know", never "there are no lyrics", so none
/// of them is cached as a miss.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("network: {0}")]
    Http(#[from] reqwest::Error),
    #[error("{0} answered {1}")]
    Status(&'static str, u16),
    /// Rate limited or temporarily unavailable. Ask again only after the wait.
    #[error("{0} is busy, retrying in {seconds}s", seconds = .1.as_secs())]
    Throttled(&'static str, Duration),
    #[error("unreadable {0} answer: {1}")]
    Malformed(&'static str, String),
}
