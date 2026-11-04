// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! `yt-client`: a client for YouTube Music's private InnerTube API.
//!
//! What it does:
//! - Bootstraps from YouTube itself: visitor id, web client version and API key.
//! - Speaks as a chosen client identity (`ClientKind`), with that identity's headers and context.
//! - Keeps an account session with cookie rotation, and signs account requests with SAPISIDHASH.
//! - Retries only network failures, never an answer from YouTube.
//!
//! It knows nothing about audio, caching or playback. See `yt-stream` for URL resolution.

pub mod bootstrap;
pub mod clients;
pub mod endpoints;
pub mod error;
pub mod models;
pub mod pages;
pub mod session;
pub mod transport;

pub use bootstrap::{apply_web_config, fetch_visitor_data, fetch_web_config, WebConfig};
pub use clients::{ClientKind, ClientSpec, LoginUse, PoTokenNeed, VersionTable};
pub use endpoints::{is_song_type, PlayerRequest, SearchResults, TopResult, SONGS_FILTER};
pub use error::{Error, Result};
pub use models::{Format, PlayerResponse, SongHit, StreamingData};
pub use pages::{AlbumCard, AlbumPage, ArtistCard, ArtistPage, Shelf};
pub use session::{CookieJar, Session, SharedSession};
pub use transport::{Auth, InnerTube, TransportConfig};

/// Build a ready client: fetch the visitor id and the web config, then return a transport that
/// uses them, and the web config itself (for the player script path). Anonymous unless the session
/// already carries cookies.
///
/// The two fetches do not depend on each other, so they run together, and they go through the
/// transport's own connection pool: the first real request then finds a warm connection to the
/// music host instead of paying for another DNS lookup and TLS handshake.
pub async fn connect(
    config: TransportConfig,
    session: Session,
) -> Result<(InnerTube, WebConfig)> {
    let shared = SharedSession::new(session);
    let mut tube = InnerTube::new(config, shared.clone())?;

    let (web, visitor) = tokio::try_join!(
        fetch_web_config(tube.http()),
        fetch_visitor_data(tube.http())
    )?;
    shared.update(|s| s.visitor_data = Some(visitor));
    tube.set_api_key(web.api_key.clone());
    apply_web_config(&tube.versions, &web);
    Ok((tube, web))
}
