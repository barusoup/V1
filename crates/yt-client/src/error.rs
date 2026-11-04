// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("network: {0}")]
    Http(#[from] reqwest::Error),

    #[error("could not decode response: {0}")]
    Json(#[from] serde_json::Error),

    /// YouTube answered, but refused the request with an HTTP error status.
    #[error("YouTube answered {status} for {endpoint}")]
    Status { endpoint: String, status: u16 },

    /// The account cookie was refused. The caller should ask the user to sign in again.
    #[error("the signed-in session was rejected")]
    SessionRejected,

    /// `sw.js_data` did not contain a visitor id.
    #[error("no visitor id in sw.js_data")]
    NoVisitorData,

    /// The homepage did not contain the web client version or API key.
    #[error("no client config on the music.youtube.com homepage")]
    NoClientConfig,

    /// YouTube answered, but the page had nothing in it that this client reads.
    #[error("YouTube sent no content for {endpoint}")]
    NoContent { endpoint: String },
}

impl Error {
    /// True when retrying the same request might succeed: the network failed, not YouTube.
    pub fn is_transient(&self) -> bool {
        match self {
            Error::Http(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            _ => false,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
