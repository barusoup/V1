//! Audio decoding: WAV/FLAC/OGG(Vorbis)/MP3 to interleaved stereo f32.
//!
//! Decoding is pure CPU and fully tested; the mixer resamples to the device
//! rate at playback time, so decoded data keeps its native rate.

use std::io::Cursor;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded clip data: interleaved stereo f32 samples in -1..1.
#[derive(Clone, Debug, Default)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration: f32,
}

impl AudioData {
    /// Frames (stereo sample pairs) in the clip.
    pub fn frames(&self) -> usize {
        self.samples.len() / 2
    }
}

/// Decode compressed/PCM audio bytes. Formats: WAV (PCM + float), FLAC,
/// OGG/Vorbis, MP3. Mono is duplicated to stereo; anything wider is folded
/// to stereo.
pub fn decode(bytes: &[u8]) -> Result<AudioData, String> {
    decode_with_hint(bytes, Hint::new())
}

/// Decode with a filename hint, which helps the probe pick a format for
/// extension-less data.
pub fn decode_with_hint(bytes: &[u8], hint: Hint) -> Result<AudioData, String> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| format!("probe audio: {err}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no decodable audio track")?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| format!("create decoder: {err}"))?;

    let mut samples: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(err) => return Err(format!("read packet: {err}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(err) => return Err(format!("decode packet: {err}")),
        };
        append_decoded(&mut samples, decoded);
    }

    let frames = samples.len() / 2;
    Ok(AudioData {
        duration: frames as f32 / sample_rate.max(1) as f32,
        samples,
        sample_rate,
    })
}

/// Decode a file from disk.
pub fn decode_file(path: impl AsRef<std::path::Path>) -> Result<AudioData, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }
    decode_with_hint(&bytes, hint)
}

fn append_decoded(samples: &mut Vec<f32>, decoded: AudioBufferRef<'_>) {
    // SampleBuffer converts any decoded format to interleaved f32.
    let spec = *decoded.spec();
    let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
    buffer.copy_interleaved_ref(decoded);
    let channels = spec.channels.count().max(1);
    let interleaved = buffer.samples();
    let frames = interleaved.len() / channels;
    samples.reserve(frames * 2);
    for frame in 0..frames {
        // Fold to stereo: left = ch 0, right = ch 1 (or ch 0 when mono).
        let left = interleaved[frame * channels];
        let right = if channels > 1 {
            interleaved[frame * channels + 1]
        } else {
            left
        };
        samples.push(left);
        samples.push(right);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Hand-built 16-bit PCM stereo WAV: no encoder dependency needed to
    /// test the decoder.
    pub fn test_wav(frames: u32, sample_rate: u32, left: i16, right: i16) -> Vec<u8> {
        let data_bytes = frames * 4;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&2u16.to_le_bytes()); // stereo
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
        wav.extend_from_slice(&4u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_bytes.to_le_bytes());
        for _ in 0..frames {
            wav.extend_from_slice(&left.to_le_bytes());
            wav.extend_from_slice(&right.to_le_bytes());
        }
        wav
    }

    #[test]
    fn decodes_wav_stereo() {
        let wav = test_wav(100, 22_050, 16_000, -16_000);
        let data = decode(&wav).expect("decode wav");
        assert_eq!(data.sample_rate, 22_050);
        assert_eq!(data.frames(), 100);
        assert!((data.duration - 100.0 / 22_050.0).abs() < 1e-4);
        assert!((data.samples[0] - 16_000.0 / 32_768.0).abs() < 1e-3);
        assert!((data.samples[1] + 16_000.0 / 32_768.0).abs() < 1e-3);
    }

    #[test]
    fn decodes_wav_from_file() {
        let wav = test_wav(16, 44_100, 0, 0);
        let path = std::env::temp_dir().join("dream_test_decode.wav");
        std::fs::write(&path, &wav).expect("write temp wav");
        let data = decode_file(&path).expect("decode file");
        assert_eq!(data.frames(), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_garbage() {
        let err = decode(b"not audio at all").expect_err("garbage must fail");
        assert!(!err.is_empty());
    }

    #[test]
    fn empty_wav_has_no_frames() {
        let wav = test_wav(0, 44_100, 0, 0);
        let data = decode(&wav).expect("decode empty wav");
        assert_eq!(data.frames(), 0);
    }
}
