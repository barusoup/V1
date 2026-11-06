// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Typed views of the InnerTube responses we use. Only the fields the client reads are modelled,
//! so an unknown field never breaks parsing.
//!
//! YouTube sends many numbers as JSON strings (`"contentLength": "219521915"`). The helpers below
//! accept either form.

use serde::{Deserialize, Deserializer};

fn de_opt_u64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NumOrStr {
        Num(u64),
        Str(String),
    }
    Ok(match Option::<NumOrStr>::deserialize(d)? {
        Some(NumOrStr::Num(n)) => Some(n),
        Some(NumOrStr::Str(s)) => s.parse().ok(),
        None => None,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayabilityStatus {
    #[serde(default)]
    pub status: String,
    pub reason: Option<String>,
}

impl PlayabilityStatus {
    pub fn is_ok(&self) -> bool {
        self.status == "OK"
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerResponse {
    #[serde(default)]
    pub playability_status: PlayabilityStatus,
    pub streaming_data: Option<StreamingData>,
    pub video_details: Option<VideoDetails>,
    pub playback_tracking: Option<PlaybackTracking>,
    pub player_config: Option<PlayerConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamingData {
    /// How long the stream URLs stay valid, in seconds (about six hours in practice).
    #[serde(default, deserialize_with = "de_opt_u64")]
    pub expires_in_seconds: Option<u64>,
    #[serde(default)]
    pub adaptive_formats: Vec<Format>,
    #[serde(default)]
    pub formats: Vec<Format>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Format {
    pub itag: u32,
    pub mime_type: String,
    #[serde(default)]
    pub bitrate: u64,
    pub audio_quality: Option<String>,
    pub audio_channels: Option<u32>,
    #[serde(default, deserialize_with = "de_opt_u64")]
    pub content_length: Option<u64>,
    /// A direct URL. Absent when the stream is ciphered.
    pub url: Option<String>,
    /// A ciphered URL: `s=` (signature), `sp=` (param name) and `url=` (base). Needs the player
    /// script to become a playable URL.
    pub signature_cipher: Option<String>,
    pub audio_track: Option<AudioTrack>,
}

impl Format {
    pub fn is_audio(&self) -> bool {
        self.mime_type.starts_with("audio/")
    }

    pub fn is_opus(&self) -> bool {
        self.mime_type.contains("opus")
    }

    /// The default-language track of a dubbed video. `None` when YouTube does not say.
    pub fn is_default_language(&self) -> Option<bool> {
        self.audio_track.as_ref().and_then(|t| t.audio_is_default)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioTrack {
    pub display_name: Option<String>,
    pub audio_is_default: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoDetails {
    pub video_id: Option<String>,
    pub title: Option<String>,
    pub author: Option<String>,
    #[serde(default, deserialize_with = "de_opt_u64")]
    pub length_seconds: Option<u64>,
    pub thumbnail: Option<Thumbnails>,
    /// Song or video (`MUSIC_VIDEO_TYPE_ATV`, `_OMV`, …). Only the music clients send it.
    pub music_video_type: Option<String>,
}

impl VideoDetails {
    /// Whether the answer says this is a video rather than a song. `false` when it does not say.
    pub fn is_video(&self) -> bool {
        self.music_video_type
            .as_deref()
            .is_some_and(|kind| !crate::endpoints::is_song_type(kind))
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Thumbnails {
    #[serde(default)]
    pub thumbnails: Vec<Thumbnail>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Thumbnail {
    pub url: String,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaybackTracking {
    pub videostats_playback_url: Option<UrlBase>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UrlBase {
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerConfig {
    pub audio_config: Option<AudioConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioConfig {
    pub loudness_db: Option<f64>,
}

impl PlayerResponse {
    /// Loudness YouTube measured for this track, for volume normalisation.
    pub fn loudness_db(&self) -> Option<f64> {
        self.player_config
            .as_ref()?
            .audio_config
            .as_ref()?
            .loudness_db
    }

    /// The watch-history ping URL. Pinging it is what registers a play in the account's history.
    pub fn tracking_ping(&self) -> Option<&str> {
        self.playback_tracking
            .as_ref()?
            .videostats_playback_url
            .as_ref()?
            .base_url
            .as_deref()
    }
}

/// One song row from a search, or a track on an artist or album page.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SongHit {
    pub video_id: String,
    pub title: String,
    pub artists: Vec<String>,
    pub album: Option<String>,
    pub duration: Option<String>,
    pub thumbnail: Option<String>,
    /// The browse id (`UC…`) of the first artist credit, when the row links to it.
    pub artist_id: Option<String>,
    /// The browse id (`MPREb_…`) of the album, when the row links to it.
    pub album_id: Option<String>,
    /// The browse id of each artist credit, in the order of `artists`, when the row links to it.
    /// Empty for songs saved before it was kept: then only the first artist links, by `artist_id`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artist_ids: Vec<Option<String>>,
}

impl SongHit {
    /// Each artist credit with the browse id it links to, if any.
    pub fn artist_links(&self) -> impl Iterator<Item = (&str, Option<&str>)> {
        self.artists.iter().enumerate().map(|(i, name)| {
            let id = if self.artist_ids.is_empty() {
                (i == 0).then_some(self.artist_id.as_deref()).flatten()
            } else {
                self.artist_ids.get(i).and_then(Option::as_deref)
            };
            (name.as_str(), id)
        })
    }
}
