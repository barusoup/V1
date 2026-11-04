// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! `yt-lyrics`: synced lyrics for a track, from Boidu first and LRCLIB as the fallback.
//!
//! - [`Service::lookup`] is the one call the app makes. It checks the cache, asks the providers in
//!   order, and says what it found as a [`Lookup`].
//! - [`model`] holds the result: a [`Lyrics`] is either timed lines, plain text, or "instrumental".
//! - [`lrc`] and [`ttml`] are the two text formats; [`boidu`] and [`lrclib`] fetch from the APIs.
//!
//! The edge cases this is built around: a lookup with no track length never reaches a fuzzy match
//! or the cache, since it can land on another cut of the song; a flat timeline is shown as plain
//! text; a provider that refused or failed never becomes a remembered "no lyrics"; and Boidu is
//! paused for as long as it asks when it rate limits.

pub mod boidu;
pub mod cache;
pub mod error;
pub mod lrc;
pub mod lrclib;
pub mod model;
pub mod service;
pub mod ttml;

pub use cache::Cache;
pub use error::Error;
pub use model::{active_line, Body, Lyrics, Query, Source, TimedLine};
pub use service::{Endpoints, Lookup, Service};

/// Sent to both providers. LRCLIB asks clients to identify themselves.
pub const USER_AGENT: &str = concat!("v1/", env!("CARGO_PKG_VERSION"), " (desktop client)");
