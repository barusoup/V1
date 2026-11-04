//! What a player needs before it starts decoding: display size, frame rate,
//! stream kinds and length, read from `ffprobe -print_format flat`.
//!
//! Flat output is a few lines of `key=value` pairs, so this crate needs no JSON
//! dependency. Rotation and pixel aspect ratio are applied to the reported
//! size, which means callers can size their texture from [`MediaInfo`] alone.

use crate::error::{Error, Result};
use crate::tools::Tools;
use std::collections::BTreeMap;
use std::path::Path;

const DEFAULT_FPS: f64 = 30.0;

/// Metadata for one media file.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaInfo {
    /// Display width after sample-aspect and rotation are applied. Zero for
    /// audio-only files.
    pub width: u32,
    /// Display height after sample-aspect and rotation are applied. Zero for
    /// audio-only files.
    pub height: u32,
    /// Source frame rate, clamped to 1..=240. Zero when there is no video.
    pub fps: f64,
    /// Container or stream duration in milliseconds.
    pub duration_ms: Option<i64>,
    /// At least one drawable video stream (cover art excluded).
    pub has_video: bool,
    /// At least one audio stream.
    pub has_audio: bool,
    /// `ffprobe`'s container format name, e.g. `mov,mp4,m4a`.
    pub format: String,
}

impl MediaInfo {
    /// No picture; useful for choosing an audio-only UI.
    pub fn is_audio_only(&self) -> bool {
        self.has_audio && !self.has_video
    }

    /// The display size, or `None` for audio-only media.
    pub fn size(&self) -> Option<(u32, u32)> {
        self.has_video.then_some((self.width, self.height))
    }
}

/// Probe `source` using the process-wide tool cache.
pub fn probe(source: impl AsRef<Path>) -> Result<MediaInfo> {
    probe_with(&Tools::discover()?, source)
}

/// Probe `source` with an explicit tool location.
pub fn probe_with(tools: &Tools, source: impl AsRef<Path>) -> Result<MediaInfo> {
    let source = source.as_ref();
    let text = tools.ffprobe_flat(source)?;
    parse_info(&text).map_err(|error| Error::msg(format!("{}: {error}", source.display())))
}

/// Parse flat `ffprobe` output. Pure, so the shape of real files can be tested
/// without spawning a decoder.
pub(crate) fn parse_info(text: &str) -> Result<MediaInfo> {
    let flat = Flat::parse(text);
    let streams = flat.streams;
    let video = streams.iter().find(|stream| {
        field(stream, "codec_type") == Some("video")
            && field(stream, "disposition.attached_pic") != Some("1")
            && stream_size(stream).is_some()
    });
    let has_audio = streams
        .iter()
        .any(|stream| field(stream, "codec_type") == Some("audio"));

    let Some(video) = video else {
        if !has_audio {
            return Err(Error::msg("no playable audio or video stream"));
        }
        return Ok(MediaInfo {
            width: 0,
            height: 0,
            fps: 0.0,
            duration_ms: duration_ms(&flat.format, None),
            has_video: false,
            has_audio: true,
            format: field(&flat.format, "format_name")
                .unwrap_or("unknown")
                .to_string(),
        });
    };

    let (mut width, mut height) =
        stream_size(video).ok_or_else(|| Error::msg("video stream has no size"))?;
    if let Some((num, den)) = sample_aspect(video) {
        width = (f64::from(width) * num / den).round().max(1.0) as u32;
    }
    if rotation(video).is_some_and(|degrees| (degrees.round() as i64).abs() % 180 == 90) {
        std::mem::swap(&mut width, &mut height);
    }

    Ok(MediaInfo {
        width,
        height,
        fps: frame_rate(video),
        duration_ms: duration_ms(&flat.format, Some(video)),
        has_video: true,
        has_audio,
        format: field(&flat.format, "format_name")
            .unwrap_or("unknown")
            .to_string(),
    })
}

struct Flat {
    format: BTreeMap<String, String>,
    streams: Vec<BTreeMap<String, String>>,
}

impl Flat {
    fn parse(text: &str) -> Self {
        let mut format = BTreeMap::new();
        let mut streams: Vec<BTreeMap<String, String>> = Vec::new();
        for line in text.lines() {
            let Some((key, value)) = line.trim().split_once('=') else {
                continue;
            };
            let value = unquote(value);
            if let Some(rest) = key.strip_prefix("format.") {
                format.insert(rest.to_string(), value);
            } else if let Some(rest) = key.strip_prefix("streams.stream.")
                && let Some((index, field)) = rest.split_once('.')
                && let Ok(index) = index.parse::<usize>()
            {
                if streams.len() <= index {
                    streams.resize_with(index + 1, BTreeMap::new);
                }
                streams[index].insert(field.to_string(), value);
            }
        }
        Self { format, streams }
    }
}

fn field<'a>(map: &'a BTreeMap<String, String>, key: &str) -> Option<&'a str> {
    map.get(key).map(String::as_str).filter(|v| *v != "N/A")
}

fn unquote(value: &str) -> String {
    let value = value.trim();
    let Some(inner) = value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    else {
        return value.to_string();
    };
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

fn stream_size(stream: &BTreeMap<String, String>) -> Option<(u32, u32)> {
    let width = field(stream, "width")?.parse().ok()?;
    let height = field(stream, "height")?.parse().ok()?;
    (width > 0 && height > 0).then_some((width, height))
}

fn sample_aspect(stream: &BTreeMap<String, String>) -> Option<(f64, f64)> {
    let (num, den) = field(stream, "sample_aspect_ratio")?.split_once(':')?;
    let num: f64 = num.trim().parse().ok()?;
    let den: f64 = den.trim().parse().ok()?;
    (num > 0.0 && den > 0.0 && (num - den).abs() > f64::EPSILON).then_some((num, den))
}

fn rotation(stream: &BTreeMap<String, String>) -> Option<f64> {
    let side_data = stream.iter().find_map(|(key, value)| {
        (key.starts_with("side_data_list.side_data.") && key.ends_with(".rotation"))
            .then(|| number(value))
            .flatten()
    });
    side_data.or_else(|| field(stream, "tags.rotate").and_then(number))
}

fn frame_rate(stream: &BTreeMap<String, String>) -> f64 {
    ["avg_frame_rate", "r_frame_rate"]
        .iter()
        .find_map(|key| field(stream, key).and_then(parse_rate))
        .map(|fps| fps.clamp(1.0, 240.0))
        .unwrap_or(DEFAULT_FPS)
}

fn parse_rate(text: &str) -> Option<f64> {
    let (num, den) = text.split_once('/')?;
    let num: f64 = num.trim().parse().ok()?;
    let den: f64 = den.trim().parse().ok()?;
    if den == 0.0 {
        return None;
    }
    let value = num / den;
    (value.is_finite() && value > 0.0).then_some(value)
}

fn duration_ms(
    format: &BTreeMap<String, String>,
    video: Option<&BTreeMap<String, String>>,
) -> Option<i64> {
    let container = field(format, "duration");
    let stream = video.and_then(|video| field(video, "duration"));
    [container, stream]
        .into_iter()
        .flatten()
        .filter_map(number)
        .find(|seconds| seconds.is_finite() && *seconds > 0.0)
        .map(|seconds| (seconds * 1000.0).round() as i64)
}

fn number(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(streams: &str, format: &str) -> Result<MediaInfo> {
        parse_info(&format!("{streams}\n{format}"))
    }

    #[test]
    fn a_widescreen_video_with_audio() {
        let info = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=1920
streams.stream.0.height=1080
streams.stream.0.avg_frame_rate="30000/1001"
streams.stream.0.r_frame_rate="30000/1001"
streams.stream.0.duration="10.0"
streams.stream.1.codec_type="audio""#,
            r#"format.format_name="mov,mp4,m4a"
format.duration="10.010000""#,
        )
        .unwrap();
        assert_eq!((info.width, info.height), (1920, 1080));
        assert!((info.fps - 29.97).abs() < 0.01);
        assert_eq!(info.duration_ms, Some(10_010));
        assert!(info.has_video && info.has_audio);
    }

    #[test]
    fn a_quarter_turn_swaps_the_display_size() {
        let info = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=1920
streams.stream.0.height=1080
streams.stream.0.avg_frame_rate="30/1"
streams.stream.0.side_data_list.side_data.0.side_data_type="Display Matrix"
streams.stream.0.side_data_list.side_data.0.rotation=-90"#,
            r#"format.duration="2.0""#,
        )
        .unwrap();
        assert_eq!((info.width, info.height), (1080, 1920));
    }

    #[test]
    fn an_old_rotate_tag_is_read_too() {
        let info = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=1280
streams.stream.0.height=720
streams.stream.0.avg_frame_rate="30/1"
streams.stream.0.tags.rotate="90""#,
            r#"format.duration="2.0""#,
        )
        .unwrap();
        assert_eq!((info.width, info.height), (720, 1280));
    }

    #[test]
    fn a_non_square_pixel_aspect_stretches_width() {
        let info = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=720
streams.stream.0.height=480
streams.stream.0.sample_aspect_ratio="4:3"
streams.stream.0.avg_frame_rate="30/1""#,
            r#"format.duration="2.0""#,
        )
        .unwrap();
        assert_eq!((info.width, info.height), (960, 480));
    }

    #[test]
    fn cover_art_and_audio_only_is_not_video() {
        let info = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=500
streams.stream.0.height=500
streams.stream.0.disposition.attached_pic=1
streams.stream.1.codec_type="audio""#,
            r#"format.duration="215.5""#,
        )
        .unwrap();
        assert!(!info.has_video && info.has_audio);
        assert_eq!(info.size(), None);
        assert_eq!(info.duration_ms, Some(215_500));
        assert!(info.is_audio_only());
    }

    #[test]
    fn a_zero_average_rate_falls_back_to_the_real_rate() {
        let info = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=640
streams.stream.0.height=360
streams.stream.0.avg_frame_rate="0/0"
streams.stream.0.r_frame_rate="25/1""#,
            r#"format.duration="1.0""#,
        )
        .unwrap();
        assert_eq!(info.fps, 25.0);
    }

    #[test]
    fn missing_rate_defaults_to_thirty_and_clamps() {
        let missing = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=64
streams.stream.0.height=64"#,
            "",
        )
        .unwrap();
        assert_eq!(missing.fps, 30.0);
        assert_eq!(missing.duration_ms, None);

        let fast = info(
            r#"streams.stream.0.codec_type="video"
streams.stream.0.width=64
streams.stream.0.height=64
streams.stream.0.avg_frame_rate="1000/1""#,
            "",
        )
        .unwrap();
        assert_eq!(fast.fps, 240.0);
    }

    #[test]
    fn nothing_playable_is_an_error() {
        assert!(info("", "").is_err());
        assert!(info(r#"streams.stream.0.codec_type="subtitle""#, "").is_err());
    }

    #[test]
    fn quoted_values_are_unescaped() {
        let text = r#"format.format_name="mov,mp4"
format.tags.title="A \"quoted\" title with \\ slash""#;
        let flat = Flat::parse(text);
        assert_eq!(flat.format["format_name"], "mov,mp4");
        assert_eq!(
            flat.format["tags.title"],
            "A \"quoted\" title with \\ slash"
        );
    }
}
