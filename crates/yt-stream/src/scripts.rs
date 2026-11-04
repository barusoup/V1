// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The part of stream resolution that needs YouTube's own JavaScript.
//!
//! Two things come from the player script: the signature timestamp (sent with `/player`) and the
//! functions that turn a ciphered URL into a playable one. A PoToken is a further attestation that
//! some clients require. All three sit behind this trait, so the resolver does not depend on how
//! they are produced.
//!
//! [`YtScripts`] runs the player script in an embedded engine (`yt-js`). [`NoScripts`] provides
//! nothing, and the resolver then skips the clients that need scripts. That is the degraded mode.

use std::future::Future;

use yt_client::{ClientKind, Format};
use yt_js::Player;

pub trait PlayerScripts: Send + Sync {
    /// Signature timestamp for `/player`, or `None` if it cannot be provided right now.
    fn signature_timestamp(&self) -> impl Future<Output = Option<u32>> + Send;

    /// A PoToken for this visitor, or `None` if one cannot be provided right now.
    fn po_token(&self, visitor_data: &str) -> impl Future<Output = Option<String>> + Send;

    /// The playable URL for a format of `client`, or `None` if it cannot be made playable.
    fn playable_url(
        &self,
        format: &Format,
        client: ClientKind,
    ) -> impl Future<Output = Option<String>> + Send;
}

/// Provides nothing. Only clients that need no script can succeed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoScripts;

impl PlayerScripts for NoScripts {
    async fn signature_timestamp(&self) -> Option<u32> {
        None
    }

    async fn po_token(&self, _visitor_data: &str) -> Option<String> {
        None
    }

    async fn playable_url(&self, format: &Format, _client: ClientKind) -> Option<String> {
        format.url.clone()
    }
}

/// The scripts, run from YouTube's current player script.
#[derive(Clone)]
pub struct YtScripts {
    player: Player,
}

impl YtScripts {
    pub fn new(player: Player) -> Self {
        YtScripts { player }
    }

    pub fn player(&self) -> &Player {
        &self.player
    }
}

/// Clients that answer with a `signatureCipher` (or a plain URL that still needs the `n`
/// transform) are the ones whose URLs the player's URL class has to build.
fn needs_player_urls(client: ClientKind) -> bool {
    yt_client::ClientSpec::of(client).needs_sts
}

impl PlayerScripts for YtScripts {
    async fn signature_timestamp(&self) -> Option<u32> {
        self.player.sts
    }

    async fn po_token(&self, _visitor_data: &str) -> Option<String> {
        // Not yet: the attestation that produces a PoToken is the next piece to build.
        None
    }

    async fn playable_url(&self, format: &Format, client: ClientKind) -> Option<String> {
        if let Some(cipher) = &format.signature_cipher {
            let parts = parse_cipher(cipher)?;
            return self
                .player
                .build_url(&parts.url, Some((&parts.s, &parts.sp)))
                .await
                .ok();
        }
        let url = format.url.as_deref()?;
        if needs_player_urls(client) {
            // Web and TV clients get their `n` transform from the player's own URL class. Direct
            // URLs from clients that need no script are used exactly as YouTube sent them.
            return self.player.build_url(url, None).await.ok();
        }
        Some(url.to_owned())
    }
}

/// The parts of a `signatureCipher` value: `s` (the ciphered signature), `sp` (the parameter
/// name to set, usually `sig`) and `url` (the base URL).
#[derive(Debug, PartialEq, Eq)]
pub struct CipherParts {
    pub s: String,
    pub sp: String,
    pub url: String,
}

pub fn parse_cipher(cipher: &str) -> Option<CipherParts> {
    let mut s = None;
    let mut sp = None;
    let mut url = None;
    for (k, v) in url::form_urlencoded::parse(cipher.as_bytes()) {
        match k.as_ref() {
            "s" => s = Some(v.into_owned()),
            "sp" => sp = Some(v.into_owned()),
            "url" => url = Some(v.into_owned()),
            _ => {}
        }
    }
    Some(CipherParts {
        s: s?,
        sp: sp.unwrap_or_else(|| "sig".to_owned()),
        url: url?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cipher_string_yields_its_three_parts_decoded() {
        let c = "s=AB%3DC&sp=sig&url=https%3A%2F%2Fx.test%2Fvideoplayback%3Fitag%3D140";
        let parts = parse_cipher(c).expect("parts");
        assert_eq!(parts.s, "AB=C");
        assert_eq!(parts.sp, "sig");
        assert_eq!(parts.url, "https://x.test/videoplayback?itag=140");
    }

    #[test]
    fn a_cipher_without_its_url_is_not_used() {
        assert_eq!(parse_cipher("s=AB&sp=sig"), None);
    }
}
