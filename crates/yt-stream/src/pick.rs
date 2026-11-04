// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Choosing one audio format out of a `/player` response.

use yt_client::Format;

/// How good a stream to ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Best available.
    #[default]
    High,
    /// At most 128 kbit/s. Smaller downloads on slow links.
    Low,
}

/// Bitrate ceiling for [`Quality::Low`], in bits per second.
pub const LOW_BITRATE_CAP: u64 = 128_000;

/// Pick the audio format to stream.
///
/// Order of preference:
/// 1. The default-language track of a dubbed video. A dub is never the better stream.
/// 2. Quality rank (`AUDIO_QUALITY_HIGH` > `MEDIUM` > `LOW`).
/// 3. Channel count.
/// 4. Codec: Opus over AAC.
/// 5. Bitrate.
///
/// `Low` drops anything above [`LOW_BITRATE_CAP`] first, unless nothing else exists.
pub fn pick_audio(formats: &[Format], quality: Quality) -> Option<&Format> {
    let audio: Vec<&Format> = formats.iter().filter(|f| f.is_audio()).collect();
    if audio.is_empty() {
        return None;
    }

    // Keep only the default-language tracks when YouTube marks any of them.
    let defaults: Vec<&Format> = audio
        .iter()
        .copied()
        .filter(|f| f.is_default_language() == Some(true))
        .collect();
    let pool = if defaults.is_empty() { audio } else { defaults };

    match quality {
        Quality::High => pool.into_iter().max_by_key(|f| rank_key(f)),
        Quality::Low => {
            let capped: Vec<&Format> = pool
                .iter()
                .copied()
                .filter(|f| f.bitrate <= LOW_BITRATE_CAP)
                .collect();
            let pick = if capped.is_empty() { pool } else { capped };
            pick.into_iter().max_by_key(|f| (f.bitrate, rank_key(f)))
        }
    }
}

fn rank_key(f: &Format) -> (u8, u32, u8, u64) {
    let quality = match f.audio_quality.as_deref() {
        Some("AUDIO_QUALITY_HIGH") => 3,
        Some("AUDIO_QUALITY_MEDIUM") => 2,
        Some("AUDIO_QUALITY_LOW") => 1,
        _ => 0,
    };
    let codec = if f.is_opus() {
        2
    } else if f.mime_type.contains("mp4a") {
        1
    } else {
        0
    };
    (quality, f.audio_channels.unwrap_or(2), codec, f.bitrate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(itag: u32, mime: &str, bitrate: u64, quality: &str, default: Option<bool>) -> Format {
        serde_json::from_value(serde_json::json!({
            "itag": itag,
            "mimeType": mime,
            "bitrate": bitrate,
            "audioQuality": quality,
            "audioChannels": 2,
            "url": format!("https://x/{itag}"),
            "audioTrack": default.map(|d| serde_json::json!({ "audioIsDefault": d })),
        }))
        .unwrap()
    }

    #[test]
    fn high_prefers_quality_then_opus_then_bitrate() {
        let formats = vec![
            fmt(
                140,
                "audio/mp4; codecs=\"mp4a.40.2\"",
                131_000,
                "AUDIO_QUALITY_MEDIUM",
                None,
            ),
            fmt(
                251,
                "audio/webm; codecs=\"opus\"",
                150_000,
                "AUDIO_QUALITY_MEDIUM",
                None,
            ),
            fmt(
                249,
                "audio/webm; codecs=\"opus\"",
                55_000,
                "AUDIO_QUALITY_LOW",
                None,
            ),
        ];
        assert_eq!(pick_audio(&formats, Quality::High).unwrap().itag, 251);
    }

    #[test]
    fn dubs_lose_to_the_default_language_at_any_bitrate() {
        let formats = vec![
            fmt(
                251,
                "audio/webm; codecs=\"opus\"",
                160_000,
                "AUDIO_QUALITY_MEDIUM",
                Some(false),
            ),
            fmt(
                250,
                "audio/webm; codecs=\"opus\"",
                70_000,
                "AUDIO_QUALITY_LOW",
                Some(true),
            ),
        ];
        assert_eq!(pick_audio(&formats, Quality::High).unwrap().itag, 250);
    }

    #[test]
    fn low_stays_under_the_cap_when_it_can() {
        let formats = vec![
            fmt(
                251,
                "audio/webm; codecs=\"opus\"",
                150_000,
                "AUDIO_QUALITY_MEDIUM",
                None,
            ),
            fmt(
                249,
                "audio/webm; codecs=\"opus\"",
                55_000,
                "AUDIO_QUALITY_LOW",
                None,
            ),
        ];
        assert_eq!(pick_audio(&formats, Quality::Low).unwrap().itag, 249);
    }

    #[test]
    fn no_audio_means_no_pick() {
        let formats = vec![fmt(137, "video/mp4", 0, "", None)];
        assert!(pick_audio(&formats, Quality::High).is_none());
    }
}
