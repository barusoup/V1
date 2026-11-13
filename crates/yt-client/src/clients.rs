// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Client identities: which YouTube client app a request claims to be.
//!
//! YouTube answers differently depending on the identity in the request. Each identity here was
//! checked against live responses (see the notes on each variant). The web client's version
//! changes with every YouTube deploy, so it is discovered at runtime (`bootstrap`), not hardcoded.
//!
//! Two flags decide whether a YouTube-owned URL is playable at all:
//! - `needs_sts`: the formats arrive ciphered (`signatureCipher`), so `player.js` must sign them.
//! - `po_token`: googlevideo wants a `pot=` query parameter, or it serves only the first MiB.
//!
//! The identities that need neither are the ones worth trying first; they are also the ones
//! YouTube removes first, so the registry stays data, not logic.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// The identities this crate knows how to speak as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ClientKind {
    /// Web music client. Used for search, browse and metadata. Its streams need a signature
    /// timestamp and a PoToken (a Premium account lifts the latter). Verified live: search works
    /// anonymously.
    WebRemix,
    /// visionOS app identity. Direct stream URLs, so no cipher and no PoToken. Verified live:
    /// `/player` returns OK with direct audio URLs for an anonymous request.
    VisionOs,
    /// Desktop TV identity (Cobalt). Ciphered, no PoToken in principle.
    Tv,
    /// TV identity with a Samsung Tizen device. Ciphered, no PoToken in principle. The first
    /// identity yt-dlp tries for a video.
    TvSamsung,
    /// Older TV identity. Ciphered, no PoToken in principle.
    TvDowngraded,
    /// TV identity signed out. Ciphered and needs a PoToken.
    TvHtml5Simply,
    /// Embedded web player, as used by third-party sites. Ciphered, no PoToken in principle.
    WebEmbedded,
    /// Mobile web client. Ciphered; needs a PoToken unless Premium.
    Mweb,
    /// iOS app. Direct URLs, but googlevideo caps them at the first MiB without a PoToken.
    Ios,
    /// Android app. Direct URLs, same first-MiB cap without a PoToken.
    Android,
    /// Android VR app. Direct URLs, same first-MiB cap without a PoToken.
    AndroidVr,
}

/// Whether a client refuses to serve a full stream without a PoToken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoTokenNeed {
    /// URLs work as sent.
    Never,
    /// Required anonymously; a signed-in Premium account is exempt.
    UnlessPremium,
    /// Required whenever the player response carries no player token of its own.
    Always,
}

/// Whether a client may (or must) send the account cookie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoginUse {
    /// Never sends the cookie.
    None,
    /// Sends the cookie when the caller asks for an account request.
    Supported,
}

/// Extra device fields some identities carry in `context.client`.
#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub os_name: &'static str,
    pub os_version: &'static str,
    pub device_make: &'static str,
    pub device_model: &'static str,
}

/// Everything needed to build a request as one identity.
#[derive(Debug, Clone)]
pub struct ClientSpec {
    pub kind: ClientKind,
    /// `context.client.clientName`.
    pub name: &'static str,
    /// Numeric id sent as `X-YouTube-Client-Name`. Must be the number, not the name.
    pub id: u32,
    /// Default version. `WebRemix` is overridden by discovery.
    pub default_version: &'static str,
    pub user_agent: &'static str,
    /// Host for InnerTube calls. The music host is only used by the music/vision identities.
    pub host: &'static str,
    pub device: Option<Device>,
    pub login: LoginUse,
    /// The `/player` request needs a signature timestamp from the player script.
    pub needs_sts: bool,
    /// What this identity needs before googlevideo will serve the whole stream.
    pub po_token: PoTokenNeed,
    /// Embedded players must name the third-party page they are embedded in.
    pub embedded: bool,
}

const FIREFOX_UA: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0";
const SAFARI_MAC_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Safari/605.1.15,gzip(gfe)";
const VISIONOS_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
const TV_UA: &str = "Mozilla/5.0 (ChromiumStylePlatform) Cobalt/25.lts.30.1034943-gold (unlike Gecko), Unknown_TV_Unknown_0/Unknown (Unknown, Unknown)";
const SAMSUNG_TV_UA: &str = "Mozilla/5.0 (SMART-TV; Linux; Tizen 2.4.0) AppleWebKit/538.1 (KHTML, like Gecko) Version/2.4.0 TV Safari/538.1";
const IPAD_UA: &str = "Mozilla/5.0 (iPad; CPU OS 16_7_10 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1,gzip(gfe)";
const IOS_UA: &str =
    "com.google.ios.youtube/21.26.4 (iPhone16,2; U; CPU iOS 18_3_2 like Mac OS X;)";
const ANDROID_UA: &str = "com.google.android.youtube/21.26.364 (Linux; U; Android 11) gzip";
const ANDROID_VR_UA: &str =
    "com.google.android.apps.youtube.vr.oculus/1.65.10 (Linux; U; Android 12L; eureka-user Build/SQ3A.220605.009.A1) gzip";

pub const MUSIC_HOST_CONST: &str = MUSIC_HOST;
pub const VIDEO_HOST_CONST: &str = VIDEO_HOST;

const MUSIC_HOST: &str = "music.youtube.com";
const VIDEO_HOST: &str = "www.youtube.com";

impl ClientSpec {
    pub fn of(kind: ClientKind) -> ClientSpec {
        match kind {
            ClientKind::WebRemix => ClientSpec {
                kind,
                name: "WEB_REMIX",
                id: 67,
                default_version: "1.20261027.10.00",
                user_agent: FIREFOX_UA,
                host: MUSIC_HOST,
                device: None,
                login: LoginUse::Supported,
                needs_sts: true,
                po_token: PoTokenNeed::UnlessPremium,
                embedded: false,
            },
            ClientKind::VisionOs => ClientSpec {
                kind,
                name: "VISIONOS",
                id: 101,
                default_version: "1.02",
                user_agent: VISIONOS_UA,
                host: MUSIC_HOST,
                device: Some(Device {
                    os_name: "visionOS",
                    os_version: "26.5.23O471",
                    device_make: "Apple",
                    device_model: "RealityDevice17,1",
                }),
                login: LoginUse::None,
                needs_sts: false,
                po_token: PoTokenNeed::Never,
                embedded: false,
            },
            ClientKind::Tv => ClientSpec {
                kind,
                name: "TVHTML5",
                id: 7,
                default_version: "7.20260707.07.00",
                user_agent: TV_UA,
                host: VIDEO_HOST,
                device: None,
                login: LoginUse::Supported,
                needs_sts: true,
                po_token: PoTokenNeed::Never,
                embedded: false,
            },
            ClientKind::TvSamsung => ClientSpec {
                kind,
                name: "TVHTML5",
                id: 7,
                default_version: "5.20260707",
                user_agent: SAMSUNG_TV_UA,
                host: VIDEO_HOST,
                device: Some(Device {
                    os_name: "Tizen",
                    os_version: "2.4.0",
                    device_make: "Samsung",
                    device_model: "UKS9800",
                }),
                login: LoginUse::Supported,
                needs_sts: true,
                po_token: PoTokenNeed::Never,
                embedded: false,
            },
            ClientKind::TvDowngraded => ClientSpec {
                kind,
                name: "TVHTML5",
                id: 7,
                default_version: "5.20260707",
                user_agent: "Mozilla/5.0 (ChromiumStylePlatform) Cobalt/Version",
                host: VIDEO_HOST,
                device: None,
                login: LoginUse::Supported,
                needs_sts: true,
                po_token: PoTokenNeed::Never,
                embedded: false,
            },
            ClientKind::TvHtml5Simply => ClientSpec {
                kind,
                name: "TVHTML5_SIMPLY",
                id: 75,
                default_version: "1.0",
                user_agent: TV_UA,
                host: VIDEO_HOST,
                device: None,
                login: LoginUse::None,
                needs_sts: true,
                po_token: PoTokenNeed::Always,
                embedded: false,
            },
            ClientKind::WebEmbedded => ClientSpec {
                kind,
                name: "WEB_EMBEDDED_PLAYER",
                id: 56,
                default_version: "2.20260708.00.00",
                user_agent: SAFARI_MAC_UA,
                host: VIDEO_HOST,
                device: None,
                login: LoginUse::Supported,
                needs_sts: true,
                po_token: PoTokenNeed::Never,
                embedded: true,
            },
            ClientKind::Mweb => ClientSpec {
                kind,
                name: "MWEB",
                id: 2,
                default_version: "2.20260708.05.00",
                user_agent: IPAD_UA,
                host: VIDEO_HOST,
                device: None,
                login: LoginUse::Supported,
                needs_sts: true,
                po_token: PoTokenNeed::UnlessPremium,
                embedded: false,
            },
            ClientKind::Ios => ClientSpec {
                kind,
                name: "IOS",
                id: 5,
                default_version: "21.26.4",
                user_agent: IOS_UA,
                host: VIDEO_HOST,
                device: Some(Device {
                    os_name: "iPhone",
                    os_version: "18.3.2.22D82",
                    device_make: "Apple",
                    device_model: "iPhone16,2",
                }),
                login: LoginUse::Supported,
                needs_sts: false,
                po_token: PoTokenNeed::Always,
                embedded: false,
            },
            ClientKind::Android => ClientSpec {
                kind,
                name: "ANDROID",
                id: 3,
                default_version: "21.26.364",
                user_agent: ANDROID_UA,
                host: VIDEO_HOST,
                device: Some(Device {
                    os_name: "Android",
                    os_version: "11",
                    device_make: "Google",
                    device_model: "Pixel 5",
                }),
                login: LoginUse::Supported,
                needs_sts: false,
                po_token: PoTokenNeed::Always,
                embedded: false,
            },
            ClientKind::AndroidVr => ClientSpec {
                kind,
                name: "ANDROID_VR",
                id: 28,
                default_version: "1.65.10",
                user_agent: ANDROID_VR_UA,
                host: VIDEO_HOST,
                device: Some(Device {
                    os_name: "Android",
                    os_version: "12L",
                    device_make: "Oculus",
                    device_model: "Quest 3",
                }),
                login: LoginUse::Supported,
                needs_sts: false,
                po_token: PoTokenNeed::Always,
                embedded: false,
            },
        }
    }

    /// The order to try identities in when resolving a stream.
    pub const CHAIN: &'static [ClientKind] = &[
        ClientKind::VisionOs,
        ClientKind::TvSamsung,
        ClientKind::Tv,
        ClientKind::WebEmbedded,
        ClientKind::WebRemix,
        ClientKind::TvHtml5Simply,
    ];
}

impl std::fmt::Display for ClientKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let spec = ClientSpec::of(*self);
        write!(f, "{}", spec.name)
    }
}

/// The version of each client that is in use. Starts at each spec's default and can be
/// replaced at runtime by discovery. Shared across clones of the transport.
#[derive(Debug, Clone, Default)]
pub struct VersionTable(Arc<RwLock<HashMap<ClientKind, String>>>);

impl VersionTable {
    pub fn set(&self, kind: ClientKind, version: String) {
        self.0.write().unwrap().insert(kind, version);
    }

    pub fn get(&self, kind: ClientKind) -> String {
        self.0
            .read()
            .unwrap()
            .get(&kind)
            .cloned()
            .unwrap_or_else(|| ClientSpec::of(kind).default_version.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_identity_has_a_distinct_numeric_id_and_name_pair() {
        let all = [
            ClientKind::WebRemix,
            ClientKind::VisionOs,
            ClientKind::Tv,
            ClientKind::TvSamsung,
            ClientKind::TvDowngraded,
            ClientKind::TvHtml5Simply,
            ClientKind::WebEmbedded,
            ClientKind::Mweb,
            ClientKind::Ios,
            ClientKind::Android,
            ClientKind::AndroidVr,
        ];
        for k in all {
            let spec = ClientSpec::of(k);
            assert!(!spec.name.is_empty());
            assert!(!spec.default_version.is_empty());
            assert!(spec.id > 0);
        }
    }

    #[test]
    fn the_chain_starts_with_an_identity_that_needs_no_po_token() {
        let first = ClientSpec::of(ClientSpec::CHAIN[0]);
        assert_eq!(first.po_token, PoTokenNeed::Never);
        assert!(!first.needs_sts);
    }
}
