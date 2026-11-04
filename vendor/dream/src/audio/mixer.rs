//! The CPU mixer: voices, buses, spatialization, fades.
//!
//! Fully deterministic and device-free: [`Mixer::mix_into`] accumulates
//! stereo frames into any buffer, so tests, tools and the audio callback all
//! run the same code. The device (`crate::audio`) only owns a cpal stream that
//! pumps this mixer.
//!
//! Model: a fixed voice pool (oldest stolen under pressure), per-voice
//! gain/pitch/pan/looping, optional 3D positioning against a listener
//! transform, and gain buses (master + music/sfx/ui) that everything sums
//! through.

use super::decode::AudioData;
use std::sync::Arc;

/// Maximum simultaneous voices. The oldest playing voice is stolen when
/// exceeded.
pub const MAX_VOICES: usize = 128;

/// Gain buses every voice sums through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bus {
    Master,
    Music,
    Sfx,
    Ui,
}

impl Bus {
    fn index(self) -> usize {
        match self {
            Bus::Master => 0,
            Bus::Music => 1,
            Bus::Sfx => 2,
            Bus::Ui => 3,
        }
    }
}

/// Listener transform for spatial voices: forward is the facing direction.
#[derive(Clone, Copy, Debug)]
pub struct Listener {
    pub position: glam::Vec3,
    pub forward: glam::Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            forward: glam::Vec3::new(0.0, 0.0, -1.0),
        }
    }
}

/// Per-voice playback parameters for [`Mixer::play`].
#[derive(Clone, Copy, Debug)]
pub struct PlayParams {
    pub gain: f32,
    /// Playback rate multiplier; 2.0 plays twice as fast (and an octave up).
    pub pitch: f32,
    /// Stereo pan for non-spatial voices: -1 hard left, +1 hard right.
    pub pan: f32,
    pub looping: bool,
    /// World position; spatial voices attenuate by distance and pan by
    /// direction relative to the listener.
    pub position: Option<glam::Vec3>,
    pub bus: Bus,
    /// Distance where attenuation starts.
    pub min_distance: f32,
    /// Distance where a spatial voice is silent.
    pub max_distance: f32,
    /// Seconds of fade-in from silence.
    pub fade_in: f32,
}

impl Default for PlayParams {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pitch: 1.0,
            pan: 0.0,
            looping: false,
            position: None,
            bus: Bus::Sfx,
            min_distance: 1.0,
            max_distance: 64.0,
            fade_in: 0.0,
        }
    }
}

/// Handle to a playing voice. Generation-checked: a stale handle never
/// affects a recycled voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoiceHandle {
    slot: u32,
    generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VoiceState {
    Playing,
    Paused,
    Stopping { frames_left: u64 },
    Dead,
}

#[derive(Debug)]
struct Voice {
    clip: Option<Arc<AudioData>>,
    /// Fractional read position in clip frames.
    cursor: f64,
    gain: f32,
    pitch: f32,
    pan: f32,
    looping: bool,
    position: Option<glam::Vec3>,
    bus: Bus,
    min_distance: f32,
    max_distance: f32,
    /// Fade-in progress, seconds remaining.
    fade_in_left: f32,
    fade_in_total: f32,
    /// Length of a running fade-out, in frames. The envelope divides the
    /// remaining frames by this fixed length, so the ramp stays continuous
    /// across `mix_into` blocks instead of restarting at unity each block.
    fade_out_total: u64,
    state: VoiceState,
    generation: u32,
    /// Start order, for oldest-voice stealing.
    order: u64,
}

impl Voice {
    fn dead() -> Self {
        Self {
            clip: None,
            cursor: 0.0,
            gain: 1.0,
            pitch: 1.0,
            pan: 0.0,
            looping: false,
            position: None,
            bus: Bus::Sfx,
            min_distance: 1.0,
            max_distance: 64.0,
            fade_in_left: 0.0,
            fade_in_total: 0.0,
            fade_out_total: 0,
            state: VoiceState::Dead,
            generation: 0,
            order: 0,
        }
    }
}

/// The mixer. See the module docs. `mix_into` *adds* into the output buffer,
/// so several mixers (or a test tone) can sum.
pub struct Mixer {
    sample_rate: u32,
    voices: Vec<Voice>,
    bus_gain: [f32; 4],
    listener: Listener,
    mixed_frames: u64,
    /// Voices stolen by the limiter since creation (diagnostics).
    stolen: u64,
    /// Monotonic play counter for oldest-voice stealing.
    next_order: u64,
}

impl Mixer {
    /// `sample_rate` is the rate `mix_into` produces (the device rate).
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            voices: (0..MAX_VOICES).map(|_| Voice::dead()).collect(),
            bus_gain: [1.0; 4],
            listener: Listener::default(),
            mixed_frames: 0,
            stolen: 0,
            next_order: 1,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn set_listener(&mut self, listener: Listener) {
        self.listener = listener;
    }

    pub fn listener(&self) -> Listener {
        self.listener
    }

    /// Bus gain, 0..1+ (master multiplies every other bus).
    pub fn set_bus_gain(&mut self, bus: Bus, gain: f32) {
        self.bus_gain[bus.index()] = gain.max(0.0);
    }

    pub fn bus_gain(&self, bus: Bus) -> f32 {
        self.bus_gain[bus.index()]
    }

    /// Start a clip. Returns a handle for later control.
    pub fn play(&mut self, clip: Arc<AudioData>, params: PlayParams) -> VoiceHandle {
        let (slot, generation) = self.allocate_slot();
        let fade_in_total = if params.fade_in > 0.0 {
            params.fade_in
        } else {
            0.0
        };
        self.voices[slot] = Voice {
            clip: Some(clip),
            cursor: 0.0,
            gain: params.gain.max(0.0),
            pitch: params.pitch.max(0.001),
            pan: params.pan.clamp(-1.0, 1.0),
            looping: params.looping,
            position: params.position,
            bus: params.bus,
            min_distance: params.min_distance.max(0.01),
            max_distance: params.max_distance.max(0.02),
            fade_in_left: fade_in_total,
            fade_in_total,
            fade_out_total: 0,
            state: VoiceState::Playing,
            generation,
            order: self.next_order,
        };
        self.next_order += 1;
        VoiceHandle {
            slot: slot as u32,
            generation,
        }
    }

    /// Stop immediately, or ramp to silence over `fade_out` seconds before
    /// the voice disappears.
    pub fn stop(&mut self, handle: VoiceHandle, fade_out: Option<f32>) {
        let sample_rate = self.sample_rate;
        if let Some(voice) = self.voice_mut(handle) {
            voice.state = match fade_out {
                Some(seconds) if seconds > 0.0 => {
                    // At least one frame, or the ramp has no length to
                    // measure its envelope against.
                    let frames = ((seconds * sample_rate as f32) as u64).max(1);
                    // Re-stopping a voice that is already fading keeps the
                    // envelope where it is: the remaining time is rescaled
                    // to the new ramp length instead of jumping back to
                    // unity, which would click.
                    let frames_left = match voice.state {
                        VoiceState::Stopping { frames_left } if voice.fade_out_total > 0 => {
                            let fraction = frames_left as f64 / voice.fade_out_total as f64;
                            ((fraction * frames as f64).round() as u64).clamp(1, frames)
                        }
                        _ => frames,
                    };
                    voice.fade_out_total = frames;
                    VoiceState::Stopping { frames_left }
                }
                _ => VoiceState::Dead,
            };
        }
    }

    pub fn pause(&mut self, handle: VoiceHandle) {
        if let Some(voice) = self.voice_mut(handle)
            && voice.state == VoiceState::Playing
        {
            voice.state = VoiceState::Paused;
        }
    }

    pub fn resume(&mut self, handle: VoiceHandle) {
        if let Some(voice) = self.voice_mut(handle)
            && voice.state == VoiceState::Paused
        {
            voice.state = VoiceState::Playing;
        }
    }

    pub fn set_gain(&mut self, handle: VoiceHandle, gain: f32) {
        if let Some(voice) = self.voice_mut(handle) {
            voice.gain = gain.max(0.0);
        }
    }

    pub fn set_pitch(&mut self, handle: VoiceHandle, pitch: f32) {
        if let Some(voice) = self.voice_mut(handle) {
            voice.pitch = pitch.max(0.001);
        }
    }

    pub fn set_position(&mut self, handle: VoiceHandle, position: glam::Vec3) {
        if let Some(voice) = self.voice_mut(handle) {
            voice.position = Some(position);
        }
    }

    pub fn is_active(&self, handle: VoiceHandle) -> bool {
        self.voice(handle)
            .is_some_and(|voice| !matches!(voice.state, VoiceState::Dead))
    }

    /// Voices currently playing, paused or fading out.
    pub fn active_voices(&self) -> usize {
        self.voices
            .iter()
            .filter(|voice| !matches!(voice.state, VoiceState::Dead))
            .count()
    }

    pub fn stolen_voices(&self) -> u64 {
        self.stolen
    }

    /// Accumulate `frames` stereo frames into `out` (interleaved, additive).
    /// The output advances the mixer clock deterministically.
    pub fn mix_into(&mut self, out: &mut [f32]) {
        debug_assert!(out.len().is_multiple_of(2), "stereo output");
        let frames = out.len() / 2;
        let sample_rate = self.sample_rate as f64;
        for voice in self.voices.iter_mut() {
            let (VoiceState::Playing | VoiceState::Stopping { .. }) = voice.state else {
                continue;
            };
            let Some(clip) = voice.clip.clone() else {
                voice.state = VoiceState::Dead;
                continue;
            };
            if clip.samples.is_empty() {
                voice.state = VoiceState::Dead;
                continue;
            }
            let clip_frames = clip.frames() as f64;
            let step = (clip.sample_rate as f64 / sample_rate) * voice.pitch as f64;

            // Per-voice constant factors for this block.
            let (left_pan, right_pan) = stereo_pan(voice.pan);
            let (spatial_gain, spatial_left, spatial_right) = match voice.position {
                Some(position) => spatialize(
                    &self.listener,
                    position,
                    voice.min_distance,
                    voice.max_distance,
                ),
                None => (1.0, 1.0, 1.0),
            };
            let bus = self.bus_gain[voice.bus.index()] * self.bus_gain[Bus::Master.index()];
            let base_gain = voice.gain * spatial_gain * bus;
            let (left_pan, right_pan) = (left_pan * spatial_left, right_pan * spatial_right);

            let mut fade_frames_left = u64::MAX;
            if let VoiceState::Stopping { frames_left } = voice.state {
                fade_frames_left = frames_left;
            }
            // The fixed ramp length (never the remaining frames, which would
            // reset the envelope to 1.0 at every block boundary).
            let fade_total = voice.fade_out_total.max(1) as f32;

            let mut mixed = 0u64;
            for frame in 0..frames {
                // Fade envelope, evaluated per frame.
                let mut envelope = 1.0f32;
                if voice.fade_in_left > 0.0 {
                    let elapsed = voice.fade_in_total - voice.fade_in_left;
                    envelope *= (elapsed / voice.fade_in_total).clamp(0.0, 1.0);
                    voice.fade_in_left =
                        (voice.fade_in_left - 1.0 / self.sample_rate as f32).max(0.0);
                }
                if fade_frames_left != u64::MAX {
                    envelope *= (fade_frames_left as f32 / fade_total).clamp(0.0, 1.0);
                }
                let gain = base_gain * envelope;
                if gain <= 0.0 && fade_frames_left != u64::MAX {
                    // Silent fade tail: stop early.
                    voice.state = VoiceState::Dead;
                    break;
                }

                let (s0, s1) = sample_pair(&clip, voice.cursor);
                out[frame * 2] += s0 * gain * left_pan;
                out[frame * 2 + 1] += s1 * gain * right_pan;
                voice.cursor += step;
                mixed += 1;

                if fade_frames_left != u64::MAX {
                    fade_frames_left -= 1;
                    if fade_frames_left == 0 {
                        voice.state = VoiceState::Dead;
                        break;
                    }
                }
                if voice.cursor >= clip_frames {
                    if voice.looping {
                        voice.cursor %= clip_frames;
                    } else {
                        voice.state = VoiceState::Dead;
                        break;
                    }
                }
            }
            if let VoiceState::Stopping { frames_left } = voice.state {
                let left = frames_left.saturating_sub(mixed.min(frames_left));
                voice.state = VoiceState::Stopping { frames_left: left };
            }
            let _ = mixed;
        }
        self.mixed_frames += frames as u64;
    }

    fn voice(&self, handle: VoiceHandle) -> Option<&Voice> {
        let voice = self.voices.get(handle.slot as usize)?;
        if voice.generation == handle.generation && !matches!(voice.state, VoiceState::Dead) {
            Some(voice)
        } else {
            None
        }
    }

    fn voice_mut(&mut self, handle: VoiceHandle) -> Option<&mut Voice> {
        let voice = self.voices.get_mut(handle.slot as usize)?;
        if voice.generation == handle.generation && !matches!(voice.state, VoiceState::Dead) {
            Some(voice)
        } else {
            None
        }
    }

    fn allocate_slot(&mut self) -> (usize, u32) {
        if let Some(index) = self
            .voices
            .iter()
            .position(|voice| matches!(voice.state, VoiceState::Dead))
        {
            let generation = self.voices[index].generation.wrapping_add(1);
            return (index, generation.max(1));
        }
        // Steal the oldest (earliest started) voice.
        let oldest = self
            .voices
            .iter()
            .enumerate()
            .min_by_key(|(_, voice)| voice.order)
            .map(|(index, _)| index)
            .expect("voice pool is never empty");
        self.stolen += 1;
        let generation = self.voices[oldest].generation.wrapping_add(1);
        (oldest, generation.max(1))
    }
}

/// Stereo pan for `pan` in -1..1. Centre is identity (1, 1): a neutral pan
/// never colours the mix. Only the far side attenuates, on a cosine curve
/// for constant perceived loudness.
fn stereo_pan(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    let left = if pan > 0.0 {
        (pan * std::f32::consts::FRAC_PI_2).cos()
    } else {
        1.0
    };
    let right = if pan < 0.0 {
        (-pan * std::f32::consts::FRAC_PI_2).cos()
    } else {
        1.0
    };
    (left, right)
}

/// Distance attenuation with linear rolloff plus directional stereo pan.
/// Returns (gain, left multiplier, right multiplier).
fn spatialize(
    listener: &Listener,
    position: glam::Vec3,
    min_distance: f32,
    max_distance: f32,
) -> (f32, f32, f32) {
    let offset = position - listener.position;
    let distance = offset.length();
    let gain = if distance <= min_distance {
        1.0
    } else if distance >= max_distance {
        0.0
    } else {
        (max_distance - distance) / (max_distance - min_distance)
    };
    // Pan by lateral direction relative to the listener's facing direction.
    let forward = listener.forward.normalize_or_zero();
    let right = forward.cross(glam::Vec3::Y).normalize_or_zero();
    let lateral = if distance > 1e-4 {
        offset.normalize().dot(right)
    } else {
        0.0
    };
    let (left, right_gain) = stereo_pan(lateral.clamp(-1.0, 1.0));
    (gain, left, right_gain)
}

/// Linear-interpolated stereo sample pair at a fractional clip frame.
fn sample_pair(clip: &AudioData, cursor: f64) -> (f32, f32) {
    let frames = clip.frames();
    if frames == 0 {
        return (0.0, 0.0);
    }
    let index = cursor.floor() as usize % frames;
    let next = (index + 1) % frames;
    let t = (cursor - cursor.floor()) as f32;
    let left = clip.samples[index * 2] * (1.0 - t) + clip.samples[next * 2] * t;
    let right = clip.samples[index * 2 + 1] * (1.0 - t) + clip.samples[next * 2 + 1] * t;
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(frames: usize, sample_rate: u32, value: f32) -> Arc<AudioData> {
        Arc::new(AudioData {
            samples: vec![value; frames * 2],
            sample_rate,
            duration: frames as f32 / sample_rate as f32,
        })
    }

    fn mixer() -> Mixer {
        Mixer::new(48_000)
    }

    #[test]
    fn plays_clip_at_expected_gain() {
        let mut m = mixer();
        m.play(
            clip(100, 48_000, 0.5),
            PlayParams {
                gain: 0.5,
                ..Default::default()
            },
        );
        let mut out = vec![0.0; 20 * 2];
        m.mix_into(&mut out);
        assert!((out[0] - 0.25).abs() < 1e-6, "gain applied, got {}", out[0]);
        assert!((out[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn clip_ends_and_voice_dies() {
        let mut m = mixer();
        let handle = m.play(clip(10, 48_000, 1.0), PlayParams::default());
        let mut out = vec![0.0; 40 * 2];
        m.mix_into(&mut out);
        assert!(m.active_voices() == 0, "clip finished");
        assert!(!m.is_active(handle));
        // Frames past the clip are silent.
        assert_eq!(out[20 * 2], 0.0);
    }

    #[test]
    fn looping_wraps() {
        let mut m = mixer();
        // Clip alternates +1/-1 per frame: looping is visible in the output.
        let data = Arc::new(AudioData {
            samples: vec![1.0, 1.0, -1.0, -1.0],
            sample_rate: 48_000,
            duration: 2.0 / 48_000.0,
        });
        m.play(
            data,
            PlayParams {
                looping: true,
                ..Default::default()
            },
        );
        let mut out = vec![0.0; 8 * 2];
        m.mix_into(&mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[2], -1.0);
        assert_eq!(out[4], 1.0, "loops back to the start");
        assert_eq!(out[6], -1.0);
        assert_eq!(m.active_voices(), 1);
    }

    #[test]
    fn pitch_changes_rate() {
        let mut m = mixer();
        // Frames: +1, -1, +1, -1.
        let data = Arc::new(AudioData {
            samples: vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            sample_rate: 48_000,
            duration: 4.0 / 48_000.0,
        });
        m.play(
            data,
            PlayParams {
                pitch: 2.0,
                ..Default::default()
            },
        );
        let mut out = vec![0.0; 4 * 2];
        m.mix_into(&mut out);
        // At 2x the second output frame samples clip frame 2 (+1) instead of
        // frame 1 (-1), then the clip is consumed.
        assert_eq!(out[2], 1.0); // frame 1, left channel
        assert_eq!(out[2 * 2], 0.0, "clip ended after two output frames");
    }

    #[test]
    fn pan_routes_channels() {
        let mut m = mixer();
        m.play(
            clip(10, 48_000, 1.0),
            PlayParams {
                pan: -1.0,
                ..Default::default()
            },
        );
        let mut out = vec![0.0; 4 * 2];
        m.mix_into(&mut out);
        assert!(out[0] > 0.7, "hard left has left signal: {}", out[0]);
        assert!(
            out[1].abs() < 1e-6,
            "hard left is silent on the right: {}",
            out[1]
        );
    }

    #[test]
    fn spatial_attenuates_with_distance() {
        let mut m = mixer();
        m.set_listener(Listener {
            position: glam::Vec3::ZERO,
            forward: glam::Vec3::new(0.0, 0.0, -1.0),
        });
        let params = |position: glam::Vec3| PlayParams {
            position: Some(position),
            min_distance: 1.0,
            max_distance: 10.0,
            ..Default::default()
        };
        let near_voice = m.play(
            clip(10, 48_000, 1.0),
            params(glam::Vec3::new(0.0, 0.0, -1.0)),
        );
        let mut near = vec![0.0; 2 * 2];
        m.mix_into(&mut near);
        // Stop the first voice: mix_into is additive, so it would otherwise
        // leak into the second measurement.
        m.stop(near_voice, None);
        m.play(
            clip(10, 48_000, 1.0),
            params(glam::Vec3::new(0.0, 0.0, -9.0)),
        );
        let mut far = vec![0.0; 2 * 2];
        m.mix_into(&mut far);
        let near_level = near[0].abs();
        let far_level = far[0].abs();
        assert!(
            near_level > far_level,
            "near {near_level} vs far {far_level}"
        );
        assert!(far_level < 0.3, "distance rolloff, got {far_level}");
    }

    #[test]
    fn bus_gains_multiply() {
        let mut m = mixer();
        m.set_bus_gain(Bus::Sfx, 0.5);
        m.set_bus_gain(Bus::Master, 0.5);
        m.play(
            clip(10, 48_000, 1.0),
            PlayParams {
                bus: Bus::Sfx,
                ..Default::default()
            },
        );
        let mut out = vec![0.0; 2 * 2];
        m.mix_into(&mut out);
        assert!((out[0] - 0.25).abs() < 1e-6, "0.5 * 0.5, got {}", out[0]);
    }

    #[test]
    fn stop_with_fade_ramps_out() {
        let mut m = mixer();
        let handle = m.play(clip(48_000, 48_000, 1.0), PlayParams::default());
        m.stop(handle, Some(0.001)); // 48 frames of fade at 48 kHz
        let mut out = vec![0.0; 64 * 2];
        m.mix_into(&mut out);
        assert!(m.active_voices() == 0, "fade completes within the block");
        assert!(
            out[0] > out[40],
            "fades toward silence: {} vs {}",
            out[0],
            out[40]
        );
    }

    #[test]
    fn stop_fade_keeps_ramping_across_blocks() {
        let mut m = Mixer::new(1_000);
        // 100 frames of fade, mixed in 32-frame blocks: frame i of the ramp
        // carries gain (100 - i) / 100. A per-block denominator would
        // restart block two at unity — an audible click.
        let handle = m.play(clip(4_000, 1_000, 1.0), PlayParams::default());
        m.stop(handle, Some(0.1));
        let mut first = vec![0.0; 32 * 2];
        m.mix_into(&mut first);
        let mut second = vec![0.0; 32 * 2];
        m.mix_into(&mut second);
        assert!((first[0] - 1.0).abs() < 1e-3, "starts at full gain");
        assert!((first[30 * 2] - 0.70).abs() < 1e-3, "ramps down");
        assert!(
            (second[0] - 0.68).abs() < 1e-3,
            "continues the ramp across the block boundary, got {}",
            second[0]
        );
        assert!(second[0] < first[30 * 2], "envelope never jumps back up");
    }

    #[test]
    fn restarting_a_stop_fade_keeps_the_envelope_continuous() {
        let mut m = Mixer::new(1_000);
        let handle = m.play(clip(4_000, 1_000, 1.0), PlayParams::default());
        m.stop(handle, Some(0.1)); // 100 frames
        let mut first = vec![0.0; 64 * 2];
        m.mix_into(&mut first);
        // 64 frames consumed: the envelope sits at 36/100.
        m.stop(handle, Some(0.1));
        let mut second = vec![0.0; 32 * 2];
        m.mix_into(&mut second);
        assert!(
            (second[0] - 0.36).abs() < 1e-3,
            "re-stop keeps the envelope continuous, got {}",
            second[0]
        );
    }

    #[test]
    fn stale_handles_do_not_control_recycled_voices() {
        let mut m = mixer();
        let first = m.play(clip(10, 48_000, 1.0), PlayParams::default());
        // Finish the clip so the slot recycles.
        let mut out = vec![0.0; 20 * 2];
        m.mix_into(&mut out);
        assert!(!m.is_active(first));
        let second = m.play(clip(10, 48_000, 1.0), PlayParams::default());
        assert!(m.is_active(second));
        // The stale handle must not stop the new voice.
        m.stop(first, None);
        assert!(
            m.is_active(second),
            "generation check protects the new voice"
        );
    }

    #[test]
    fn voice_limit_steals_oldest() {
        let mut m = mixer();
        let first = m.play(clip(48_000, 48_000, 1.0), PlayParams::default());
        for _ in 1..MAX_VOICES {
            m.play(clip(48_000, 48_000, 1.0), PlayParams::default());
        }
        assert_eq!(m.active_voices(), MAX_VOICES);
        m.play(clip(48_000, 48_000, 1.0), PlayParams::default());
        assert_eq!(m.active_voices(), MAX_VOICES, "pool stays bounded");
        assert_eq!(m.stolen_voices(), 1);
        assert!(!m.is_active(first), "the oldest voice was stolen");
    }

    #[test]
    fn mix_is_additive() {
        let mut m = mixer();
        m.play(clip(10, 48_000, 0.2), PlayParams::default());
        m.play(clip(10, 48_000, 0.3), PlayParams::default());
        let mut out = vec![0.0; 2 * 2];
        m.mix_into(&mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fade_in_starts_silent() {
        let mut m = mixer();
        m.play(
            clip(48_000, 48_000, 1.0),
            PlayParams {
                fade_in: 0.01,
                ..Default::default()
            },
        );
        let mut out = vec![0.0; 32 * 2];
        m.mix_into(&mut out);
        assert!(out[0] < 0.1, "starts near silence, got {}", out[0]);
        assert!(out[62] > out[0], "ramps up over the fade");
    }

    #[test]
    fn pan_centre_is_identity_and_sides_keep_power() {
        let (l, r) = stereo_pan(0.0);
        assert_eq!((l, r), (1.0, 1.0), "neutral pan never colours the mix");
        let (l, r) = stereo_pan(1.0);
        assert!(l.abs() < 1e-6, "cos(pi/2) rounds to ~0, got {l}");
        assert_eq!(r, 1.0);
        let (l, r) = stereo_pan(-0.5);
        assert_eq!(l, 1.0);
        assert!((r - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    }
}
