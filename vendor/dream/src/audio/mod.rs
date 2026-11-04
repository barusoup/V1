//! Audio: decoding, mixing and device output.
//!
//! - [`decode()`]: WAV/FLAC/OGG/MP3 to stereo f32 on the CPU.
//! - [`mixer`]: the deterministic CPU mixer (voices, buses, spatial, fades).
//! - This module: the output device ([`Audio`]) and clip storage
//!   ([`AudioLibrary`]).
//!
//! ```ignore
//! let mut audio = Audio::open_default()?;         // device output
//! let mut audio = Audio::offline(48_000);         // or silent/deterministic
//! let shot = audio.library().load_file("boom.wav")?;
//! audio.play(shot, PlayParams { bus: Bus::Sfx, ..Default::default() });
//! ```
//!
//! The mixer lives behind `Arc<Mutex<Mixer>>`: the audio thread pumps it from
//! the device callback while the game drives it from frame code. Offline
//! mode runs the same mixer with no device, so scripted captures stay silent
//! and deterministic.

pub mod decode;
pub mod mixer;

pub use decode::{AudioData, decode, decode_file};
pub use mixer::{Bus, Listener, MAX_VOICES, Mixer, PlayParams, VoiceHandle};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Decoded clip storage with path deduplication, mirroring
/// [`crate::texture::TextureManager`].
#[derive(Default)]
pub struct AudioLibrary {
    clips: HashMap<std::path::PathBuf, Arc<AudioData>>,
    by_name: HashMap<String, Arc<AudioData>>,
}

impl AudioLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode and store a file; repeat loads return the same clip.
    pub fn load_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Arc<AudioData>, String> {
        let path =
            std::fs::canonicalize(path.as_ref()).unwrap_or_else(|_| path.as_ref().to_path_buf());
        if let Some(clip) = self.clips.get(&path) {
            return Ok(clip.clone());
        }
        let clip = Arc::new(decode::decode_file(&path)?);
        self.clips.insert(path, clip.clone());
        Ok(clip)
    }

    /// Store decoded bytes under a name (embedded assets).
    pub fn add_bytes(&mut self, name: &str, bytes: &[u8]) -> Result<Arc<AudioData>, String> {
        if let Some(clip) = self.by_name.get(name) {
            return Ok(clip.clone());
        }
        let clip = Arc::new(decode::decode(bytes)?);
        self.by_name.insert(name.to_string(), clip.clone());
        Ok(clip)
    }

    /// Store an already-decoded clip under a name.
    pub fn add(&mut self, name: &str, clip: Arc<AudioData>) -> Arc<AudioData> {
        self.by_name.insert(name.to_string(), clip.clone());
        clip
    }

    pub fn get(&self, name: &str) -> Option<Arc<AudioData>> {
        self.by_name.get(name).cloned()
    }

    pub fn stats(&self) -> (usize, f32) {
        let mut seconds = 0.0;
        let mut count = 0;
        for clip in self.clips.values().chain(self.by_name.values()) {
            count += 1;
            seconds += clip.duration;
        }
        (count, seconds)
    }
}

/// The audio output: a device stream pumping the shared [`Mixer`], or an
/// offline mixer when no output is wanted.
pub struct Audio {
    mixer: Arc<Mutex<Mixer>>,
    library: AudioLibrary,
    _stream: Option<cpal::Stream>,
    device_name: String,
}

impl Audio {
    /// Open the default output device at its preferred configuration.
    pub fn open_default() -> Result<Self, String> {
        use cpal::traits::HostTrait;
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("no default audio output device")?;
        Self::open(device)
    }

    /// Open the first output device whose name contains `name` (case-insensitive), or the default
    /// one when no device matches. A machine can carry virtual sinks (a streaming driver's, a
    /// cable's) that are selectable as the default and heard by nobody; naming a real endpoint is
    /// how a caller gets out of one.
    pub fn open_named(name: &str) -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, HostTrait};
        let wanted = name.to_lowercase();
        let host = cpal::default_host();
        let device = host.output_devices().ok().and_then(|devices| {
            devices.into_iter().find(|device| {
                device
                    .name()
                    .map(|name| name.to_lowercase().contains(&wanted))
                    .unwrap_or(false)
            })
        });
        match device {
            Some(device) => Self::open(device),
            None => Self::open_default(),
        }
    }

    /// Every output device's name, in the host's order.
    pub fn output_devices() -> Vec<String> {
        use cpal::traits::{DeviceTrait, HostTrait};
        cpal::default_host()
            .output_devices()
            .map(|devices| devices.filter_map(|device| device.name().ok()).collect())
            .unwrap_or_default()
    }

    /// Open one device at its preferred configuration.
    fn open(device: cpal::Device) -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, StreamTrait};
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        let config = device
            .default_output_config()
            .map_err(|err| format!("output config: {err}"))?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();
        let config: cpal::StreamConfig = config.into();

        let mixer = Arc::new(Mutex::new(Mixer::new(sample_rate)));
        let stream_mixer = Arc::clone(&mixer);
        let stream = match sample_format {
            cpal::SampleFormat::F32 => build_stream(&device, &config, move |out| {
                fill::<f32>(out, channels, &stream_mixer)
            }),
            cpal::SampleFormat::I16 => build_stream(&device, &config, move |out| {
                fill::<i16>(out, channels, &stream_mixer)
            }),
            cpal::SampleFormat::U16 => build_stream(&device, &config, move |out| {
                fill::<u16>(out, channels, &stream_mixer)
            }),
            format => return Err(format!("unsupported output sample format {format:?}")),
        }?;
        stream
            .play()
            .map_err(|err| format!("start stream: {err}"))?;

        Ok(Self {
            mixer,
            library: AudioLibrary::new(),
            _stream: Some(stream),
            device_name,
        })
    }

    /// No device: mixing only, through [`Self::mixer`] or
    /// [`Self::render_offline`]. Scripted runs use this so they stay silent
    /// and reproducible.
    pub fn offline(sample_rate: u32) -> Self {
        Self {
            mixer: Arc::new(Mutex::new(Mixer::new(sample_rate))),
            library: AudioLibrary::new(),
            _stream: None,
            device_name: "offline".into(),
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Shared mixer: play, stop, move the listener, set bus gains.
    pub fn mixer(&self) -> Arc<Mutex<Mixer>> {
        Arc::clone(&self.mixer)
    }

    pub fn library(&mut self) -> &mut AudioLibrary {
        &mut self.library
    }

    /// Play a clip through the mixer. The handle controls it later.
    pub fn play(&self, clip: Arc<AudioData>, params: PlayParams) -> VoiceHandle {
        self.mixer.lock().unwrap().play(clip, params)
    }

    pub fn stop(&self, handle: VoiceHandle, fade_out: Option<f32>) {
        self.mixer.lock().unwrap().stop(handle, fade_out);
    }

    /// Mix `frames` stereo frames to the CPU (additive), advancing the mixer.
    /// Works in both modes: the primary offline render path.
    pub fn render_offline(&self, out: &mut [f32]) {
        self.mixer.lock().unwrap().mix_into(out);
    }

    pub fn sample_rate(&self) -> u32 {
        self.mixer.lock().unwrap().sample_rate()
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut data_fn: impl FnMut(&mut [T]) + Send + 'static,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample,
{
    use cpal::traits::DeviceTrait;
    device
        .build_output_stream(
            config,
            move |out: &mut [T], _| data_fn(out),
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .map_err(|err| format!("build stream: {err}"))
}

/// Convert mixed f32 stereo into the device's buffer shape.
fn fill<T>(out: &mut [T], channels: usize, mixer: &Arc<Mutex<Mixer>>)
where
    f32: cpal::FromSample<T> + cpal::SizedSample,
    T: cpal::Sample + cpal::FromSample<f32>,
{
    let frames = out.len() / channels.max(1);
    let mut scratch = vec![0.0f32; frames * 2];
    mixer.lock().unwrap().mix_into(&mut scratch);
    for (frame, chunk) in out.chunks_mut(channels).enumerate() {
        for (channel, sample) in chunk.iter_mut().enumerate() {
            let value = match channel {
                0 => scratch[frame * 2],
                1 => scratch[frame * 2 + 1],
                // Extra device channels get the left signal (harmless downmix).
                _ => scratch[frame * 2],
            };
            *sample = T::from_sample(value.clamp(-1.0, 1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frames: usize, sample_rate: u32) -> Arc<AudioData> {
        let samples: Vec<f32> = (0..frames * 2)
            .map(|i| {
                ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / sample_rate as f32).sin() * 0.5
            })
            .collect();
        Arc::new(AudioData {
            samples,
            sample_rate,
            duration: frames as f32 / sample_rate as f32,
        })
    }

    #[test]
    fn offline_audio_mixes_silently() {
        let audio = Audio::offline(48_000);
        audio.play(sine(480, 48_000), PlayParams::default());
        let mut out = vec![0.0; 128 * 2];
        audio.render_offline(&mut out);
        assert!(out.iter().any(|s| s.abs() > 0.01), "mixed audible signal");
    }

    #[test]
    fn library_dedupes_files() {
        let wav = super::decode::tests::test_wav(8, 44_100, 100, 100);
        let path = std::env::temp_dir().join("dream_test_library.wav");
        std::fs::write(&path, &wav).expect("write temp wav");
        let mut library = AudioLibrary::new();
        let a = library.load_file(&path).expect("load");
        let b = library.load_file(&path).expect("load again");
        assert!(Arc::ptr_eq(&a, &b), "second load shares the clip");
        let (count, seconds) = library.stats();
        assert_eq!(count, 1);
        assert!(seconds > 0.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn library_add_by_name() {
        let mut library = AudioLibrary::new();
        let clip = sine(10, 48_000);
        library.add("beep", clip);
        assert!(library.get("beep").is_some());
        assert!(library.get("boop").is_none());
        let bytes = super::decode::tests::test_wav(8, 44_100, 0, 0);
        library.add_bytes("w", &bytes).expect("add bytes");
        assert!(library.get("w").is_some());
    }

    #[test]
    fn sample_conversion_clamps() {
        // fill() is exercised through the offline path above; here check the
        // conversion helper's clamp behaviour directly via a tiny buffer.
        let mixer = Arc::new(Mutex::new(Mixer::new(48_000)));
        mixer.lock().unwrap().play(
            Arc::new(AudioData {
                samples: vec![4.0; 16],
                sample_rate: 48_000,
                duration: 8.0 / 48_000.0,
            }),
            PlayParams::default(),
        );
        let mut out = vec![0.0f32; 8];
        fill::<f32>(&mut out, 2, &mixer);
        assert!(
            out.iter().all(|s| (-1.0..=1.0).contains(s)),
            "clamped: {out:?}"
        );
    }
}
