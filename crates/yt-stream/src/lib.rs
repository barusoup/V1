// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! `yt-stream`: from a video id to a playable audio URL.
//!
//! Pieces, each replaceable:
//! - [`Resolver`]: the fallback chain, with cache, prerequisite checks and HEAD validation.
//! - [`PlayerScripts`]: the signature timestamp, PoToken and URL deciphering. `NoScripts` for now.
//! - [`Prober`]: the URL check. [`HttpProber`] sends HEAD with the player's headers.
//! - [`UrlCache`]: [`MemoryCache`] or [`FileCache`] (survives restarts).
//! - [`pick_audio`]: format choice.

pub mod cache;
pub mod pick;
pub mod probe;
pub mod resolve;
pub mod scripts;

pub use cache::{FileCache, MemoryCache, UrlCache};
pub use pick::{pick_audio, Quality, LOW_BITRATE_CAP};
pub use probe::{HttpProber, Prober};
pub use resolve::{Attempt, ResolveError, Resolved, Resolver, ResolverConfig};
pub use scripts::{NoScripts, PlayerScripts, YtScripts};
