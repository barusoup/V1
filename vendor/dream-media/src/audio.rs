//! Audio output: a CPAL device stream fed from a preallocated ring.
//!
//! The decoder thread writes interleaved stereo f32 into the ring; the device
//! callback drains it. A single mutex guards the ring, the played-frame count
//! and the generation counter, so a seek's reset is atomic with respect to the
//! callback. No audio-path operation allocates after construction.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// Decoded audio that may wait in the ring: 0.75 seconds. Small enough that a
/// seek discards little, large enough to ride out a decoder tick.
const RING_NUMER: usize = 3;
const RING_DENOM: usize = 4;

/// The device output. Stays on the thread that opened it; only [`AudioShared`]
/// crosses threads.
pub(crate) struct AudioOutput {
    _stream: Stream,
    shared: Arc<AudioShared>,
    rate: u32,
    channels: u16,
}

impl AudioOutput {
    /// Open the default output device at its preferred configuration.
    /// `None` means there is no usable device; video still plays silently on
    /// the wall clock.
    pub(crate) fn open() -> Option<Self> {
        let device = cpal::default_host().default_output_device()?;
        let supported = device.default_output_config().ok()?;
        let format = supported.sample_format();
        let config = supported.config();
        let rate = config.sample_rate.0;
        let channels = config.channels;
        let device_channels = usize::from(channels.max(1));
        let shared = Arc::new(AudioShared::new(rate, channels));
        let on_error = |error| eprintln!("dream-media audio output: {error}");
        let stream = match format {
            SampleFormat::F32 => {
                let shared = Arc::clone(&shared);
                // Reused by the callback; cpal buffers are normally a few
                // thousand samples, so this never grows in steady state.
                let mut scratch = vec![0.0f32; 16 * 1024];
                device
                    .build_output_stream(
                        &config,
                        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                            if channels == 2 {
                                shared.fill(data);
                            } else {
                                let frames = data.len() / device_channels;
                                let needed = frames * 2;
                                if scratch.len() < needed {
                                    scratch.resize(needed, 0.0);
                                }
                                let scratch = &mut scratch[..needed];
                                shared.fill(scratch);
                                for (frame, chunk) in data.chunks_mut(device_channels).enumerate() {
                                    for (channel, sample) in chunk.iter_mut().enumerate() {
                                        *sample = match channel {
                                            0 => scratch[frame * 2],
                                            1 => scratch[frame * 2 + 1],
                                            _ => scratch[frame * 2],
                                        };
                                    }
                                }
                            }
                        },
                        on_error,
                        None,
                    )
                    .ok()?
            }
            SampleFormat::I16 => {
                let shared = Arc::clone(&shared);
                let mut scratch = vec![0.0f32; 16 * 1024];
                device
                    .build_output_stream(
                        &config,
                        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            let frames = data.len() / device_channels;
                            let needed = frames * 2;
                            if scratch.len() < needed {
                                scratch.resize(needed, 0.0);
                            }
                            let scratch = &mut scratch[..needed];
                            shared.fill(scratch);
                            for (frame, chunk) in data.chunks_mut(device_channels).enumerate() {
                                for (channel, sample) in chunk.iter_mut().enumerate() {
                                    let value = match channel {
                                        0 => scratch[frame * 2],
                                        1 => scratch[frame * 2 + 1],
                                        _ => scratch[frame * 2],
                                    };
                                    *sample = (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
                                }
                            }
                        },
                        on_error,
                        None,
                    )
                    .ok()?
            }
            SampleFormat::U16 => {
                let shared = Arc::clone(&shared);
                let mut scratch = vec![0.0f32; 16 * 1024];
                device
                    .build_output_stream(
                        &config,
                        move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                            let frames = data.len() / device_channels;
                            let needed = frames * 2;
                            if scratch.len() < needed {
                                scratch.resize(needed, 0.0);
                            }
                            let scratch = &mut scratch[..needed];
                            shared.fill(scratch);
                            for (frame, chunk) in data.chunks_mut(device_channels).enumerate() {
                                for (channel, sample) in chunk.iter_mut().enumerate() {
                                    let value = match channel {
                                        0 => scratch[frame * 2],
                                        1 => scratch[frame * 2 + 1],
                                        _ => scratch[frame * 2],
                                    };
                                    let centred = (value.clamp(-1.0, 1.0) + 1.0) * 0.5;
                                    *sample = (centred * f32::from(u16::MAX)) as u16;
                                }
                            }
                        },
                        on_error,
                        None,
                    )
                    .ok()?
            }
            _ => return None,
        };
        stream.play().ok()?;
        Some(Self {
            _stream: stream,
            shared,
            rate,
            channels,
        })
    }

    pub(crate) fn rate(&self) -> u32 {
        self.rate
    }

    pub(crate) fn channels(&self) -> u16 {
        self.channels
    }

    pub(crate) fn shared(&self) -> Arc<AudioShared> {
        Arc::clone(&self.shared)
    }
}

/// Number of level meters: low, mid and high.
pub(crate) const BANDS: usize = 3;
/// Band edges for the meters, in Hz.
const LOW_EDGE_HZ: f32 = 250.0;
const HIGH_EDGE_HZ: f32 = 2500.0;
/// Meter response: dBFS mapped onto 0..1 across this range.
const METER_FLOOR_DB: f32 = -55.0;
const METER_CEIL_DB: f32 = -12.0;
/// Meter ballistics: quick rise, slower fall, so bars track beats without jittering.
const METER_ATTACK_S: f32 = 0.06;
const METER_RELEASE_S: f32 = 0.30;

/// State shared by the feeder thread, the UI thread and the device callback.
pub(crate) struct AudioShared {
    ring: Mutex<Ring>,
    /// Wakes the feeder when the callback drains room into the ring. Without
    /// it, the feeder would have to poll, and a poll loop is a thread waking
    /// hundreds of times a second for the whole of playback.
    space: Condvar,
    channels: u16,
    capacity: usize,
    running: AtomicBool,
    /// `f32` bits, so volume changes never take the ring lock.
    volume: AtomicU32,
    muted: AtomicBool,
    /// Filter state for the meters. Touched only by the device callback.
    meters: Mutex<Meters>,
    /// Latest band levels in 0..=1, as `f32` bits. Written by the callback,
    /// read by the UI, so neither side takes a lock to read or write them.
    levels: [AtomicU32; BANDS],
    /// The most recent output, mixed to mono, for spectrum display. Written by
    /// the callback (which skips a block rather than wait on it), copied out
    /// by the UI.
    scope: Mutex<Scope>,
}

/// Length of the scope: the last this-many mono samples the device played.
pub(crate) const SCOPE_LEN: usize = 4096;

/// A fixed ring of the latest mono output samples.
struct Scope {
    samples: Box<[f32; SCOPE_LEN]>,
    write: usize,
}

impl Scope {
    fn push(&mut self, interleaved: &[f32], channels: usize) {
        let scale = 1.0 / channels as f32;
        for frame in interleaved.chunks_exact(channels) {
            self.samples[self.write] = frame.iter().sum::<f32>() * scale;
            self.write = (self.write + 1) % SCOPE_LEN;
        }
    }

    /// Copy the newest `out.len()` samples, oldest first.
    fn copy_latest(&self, out: &mut [f32]) {
        let n = out.len().min(SCOPE_LEN);
        let start = (self.write + SCOPE_LEN - n) % SCOPE_LEN;
        let first = n.min(SCOPE_LEN - start);
        out[..first].copy_from_slice(&self.samples[start..start + first]);
        out[first..n].copy_from_slice(&self.samples[..n - first]);
        out[n..].fill(0.0);
    }
}

/// One second-order (Butterworth) biquad section, direct form II transposed.
#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// RBJ cookbook low-pass or high-pass at `hz`, Q = 1/sqrt(2).
    fn new(lowpass: bool, hz: f32, rate: f32) -> Self {
        let w0 = std::f32::consts::TAU * hz / rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        let (b0, b1, b2) = if lowpass {
            let b1 = 1.0 - cos;
            (b1 * 0.5, b1, b1 * 0.5)
        } else {
            let b1 = -(1.0 + cos);
            (-b1 * 0.5, b1, -b1 * 0.5)
        };
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Splits the output into low, mid and high bands with cascaded biquads (4th
/// order slopes, so a tone in one band barely shows in the others), and tracks
/// each band's level. Fixed-size, no allocation.
struct Meters {
    rate: f32,
    low: [Biquad; 2],
    mid: [Biquad; 4],
    high: [Biquad; 2],
    smoothed: [f32; BANDS],
}

impl Meters {
    fn new(rate: u32) -> Self {
        let rate = rate.max(1) as f32;
        let lp = |hz| Biquad::new(true, hz, rate);
        let hp = |hz| Biquad::new(false, hz, rate);
        Self {
            rate,
            low: [lp(LOW_EDGE_HZ), lp(LOW_EDGE_HZ)],
            mid: [
                hp(LOW_EDGE_HZ),
                hp(LOW_EDGE_HZ),
                lp(HIGH_EDGE_HZ),
                lp(HIGH_EDGE_HZ),
            ],
            high: [hp(HIGH_EDGE_HZ), hp(HIGH_EDGE_HZ)],
            smoothed: [0.0; BANDS],
        }
    }

    /// Fold one callback's interleaved stereo samples into the meters and
    /// return the smoothed levels. Silence (or an empty buffer) decays them.
    fn analyse(&mut self, stereo: &[f32]) -> [f32; BANDS] {
        let frames = stereo.len() / 2;
        if frames == 0 {
            return self.smoothed;
        }
        let mut sums = [0.0f32; BANDS];
        for frame in stereo.chunks_exact(2) {
            let mono = 0.5 * (frame[0] + frame[1]);
            let low = self.low.iter_mut().fold(mono, |x, f| f.run(x));
            let mid = self.mid.iter_mut().fold(mono, |x, f| f.run(x));
            let high = self.high.iter_mut().fold(mono, |x, f| f.run(x));
            sums[0] += low * low;
            sums[1] += mid * mid;
            sums[2] += high * high;
        }
        let dt = frames as f32 / self.rate;
        let attack = 1.0 - (-dt / METER_ATTACK_S).exp();
        let release = 1.0 - (-dt / METER_RELEASE_S).exp();
        for (band, sum) in sums.iter().enumerate() {
            let rms = (sum / frames as f32).sqrt();
            let db = 20.0 * rms.max(1e-9).log10();
            let target = ((db - METER_FLOOR_DB) / (METER_CEIL_DB - METER_FLOOR_DB)).clamp(0.0, 1.0);
            let k = if target > self.smoothed[band] { attack } else { release };
            self.smoothed[band] += k * (target - self.smoothed[band]);
        }
        self.smoothed
    }
}

struct Ring {
    samples: Vec<f32>,
    read: usize,
    write: usize,
    len: usize,
    played_frames: u64,
    generation: u64,
    finished: bool,
}

impl AudioShared {
    pub(crate) fn new(rate: u32, channels: u16) -> Self {
        let channels = channels.max(1);
        let capacity = (rate.max(1) as usize * usize::from(channels) * RING_NUMER / RING_DENOM)
            .max(usize::from(channels) * 2);
        // Keep the ring a whole number of frames so `room` never splits one.
        let capacity = capacity / usize::from(channels) * usize::from(channels);
        Self {
            ring: Mutex::new(Ring {
                samples: vec![0.0; capacity.max(usize::from(channels))],
                read: 0,
                write: 0,
                len: 0,
                played_frames: 0,
                generation: 0,
                finished: false,
            }),
            space: Condvar::new(),
            channels,
            capacity: capacity.max(usize::from(channels)),
            running: AtomicBool::new(false),
            volume: AtomicU32::new(1.0f32.to_bits()),
            muted: AtomicBool::new(false),
            meters: Mutex::new(Meters::new(rate)),
            levels: std::array::from_fn(|_| AtomicU32::new(0.0f32.to_bits())),
            scope: Mutex::new(Scope {
                samples: Box::new([0.0; SCOPE_LEN]),
                write: 0,
            }),
        }
    }

    /// The newest `out.len()` mono samples the device played (up to
    /// [`SCOPE_LEN`]), oldest first. Anything beyond the scope is zeroed.
    pub(crate) fn scope(&self, out: &mut [f32]) {
        self.scope
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .copy_latest(out);
    }

    /// Band levels (low, mid, high), each 0..=1, of the samples the device
    /// played most recently.
    pub(crate) fn levels(&self) -> [f32; BANDS] {
        std::array::from_fn(|band| f32::from_bits(self.levels[band].load(Ordering::Relaxed)))
    }

    pub(crate) fn set_running(&self, on: bool) {
        self.running.store(on, Ordering::Release);
    }

    pub(crate) fn set_volume(&self, volume: f32) {
        let volume = if volume.is_finite() {
            volume.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.volume.store(volume.to_bits(), Ordering::Relaxed);
    }

    pub(crate) fn set_muted(&self, muted: bool) {
        self.muted.store(muted, Ordering::Relaxed);
    }

    /// Clear the ring, zero the played count and start a new generation.
    /// Everything happens under one lock, so the callback either counted a
    /// frame before the reset or none after it. The samples are not zeroed:
    /// an empty ring is never read past `len`, and a seek should not pay to
    /// clear a buffer it is about to overwrite.
    pub(crate) fn reset(&self) -> u64 {
        let mut ring = self.lock();
        ring.read = 0;
        ring.write = 0;
        ring.len = 0;
        ring.played_frames = 0;
        ring.finished = false;
        ring.generation += 1;
        ring.generation
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.lock().generation
    }

    /// Append samples if they fit and the generation is still current; false
    /// means a seek superseded this feeder. The feeder itself uses
    /// [`Self::push_wait`], which blocks instead of dropping samples.
    #[cfg(test)]
    pub(crate) fn push(&self, generation: u64, samples: &[f32]) -> bool {
        let mut ring = self.lock();
        if ring.generation != generation || samples.len() > self.capacity - ring.len {
            return false;
        }
        ring.push(samples);
        true
    }

    /// Free space in samples (not frames).
    #[cfg(test)]
    pub(crate) fn room(&self) -> usize {
        self.capacity - self.lock().len
    }

    /// Push all of `samples`, sleeping on the ring's condvar when it is full
    /// rather than polling. Returns false once a seek makes this feeder stale.
    /// Only whole frames are pushed, so the callback's frame count is exact.
    pub(crate) fn push_wait(&self, generation: u64, mut samples: &[f32]) -> bool {
        let channels = usize::from(self.channels);
        let mut ring = self.lock();
        while !samples.is_empty() {
            if ring.generation != generation {
                return false;
            }
            let room = (self.capacity - ring.len) / channels * channels;
            if room == 0 {
                // The timeout only guards against a lost wake-up; the
                // callback notifies after every drain.
                ring = self
                    .space
                    .wait_timeout(ring, Duration::from_millis(20))
                    .unwrap_or_else(|poison| poison.into_inner())
                    .0;
                continue;
            }
            let take = room.min(samples.len());
            ring.push(&samples[..take]);
            samples = &samples[take..];
        }
        true
    }

    pub(crate) fn buffered_frames(&self) -> usize {
        self.lock().len / usize::from(self.channels)
    }

    pub(crate) fn played_frames(&self) -> u64 {
        self.lock().played_frames
    }

    pub(crate) fn set_finished(&self, generation: u64) {
        let mut ring = self.lock();
        if ring.generation == generation {
            ring.finished = true;
        }
    }

    pub(crate) fn finished(&self) -> bool {
        self.lock().finished
    }

    /// The device callback's body. Kept here so it can be tested without a
    /// device. Shortfalls are zeroed and only real frames are counted.
    pub(crate) fn fill(&self, out: &mut [f32]) {
        if !self.running.load(Ordering::Acquire) {
            out.fill(0.0);
            self.measure(out);
            return;
        }
        let gain = if self.muted.load(Ordering::Relaxed) {
            0.0
        } else {
            f32::from_bits(self.volume.load(Ordering::Relaxed))
        };
        let mut ring = self.lock();
        let channels = usize::from(self.channels);
        ring.read_into(out, gain, channels);
        drop(ring);
        self.space.notify_all();
        // After gain, so the meters show what is heard: silence when muted.
        self.measure(out);
    }

    /// Feed the samples about to leave the device into the meters and publish
    /// the result. Runs on the device callback; the meters lock has no other user.
    fn measure(&self, out: &[f32]) {
        let mut meters = self
            .meters
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let levels = meters.analyse(out);
        drop(meters);
        for (slot, level) in self.levels.iter().zip(levels) {
            slot.store(level.to_bits(), Ordering::Relaxed);
        }
        // Never wait on the UI from the device callback: if it is mid-copy,
        // this block is simply left out of the scope.
        if let Ok(mut scope) = self.scope.try_lock() {
            scope.push(out, usize::from(self.channels));
        }
    }

    fn lock(&self) -> MutexGuard<'_, Ring> {
        self.ring
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl Ring {
    fn push(&mut self, samples: &[f32]) {
        let capacity = self.samples.len();
        let first = samples.len().min(capacity - self.write);
        self.samples[self.write..self.write + first].copy_from_slice(&samples[..first]);
        let rest = samples.len() - first;
        if rest > 0 {
            self.samples[..rest].copy_from_slice(&samples[first..]);
        }
        self.write = (self.write + samples.len()) % capacity;
        self.len += samples.len();
    }

    fn read_into(&mut self, out: &mut [f32], gain: f32, channels: usize) {
        let capacity = self.samples.len();
        let take = out.len().min(self.len);
        let first = take.min(capacity - self.read);
        for (slot, sample) in out[..first]
            .iter_mut()
            .zip(&self.samples[self.read..self.read + first])
        {
            *slot = *sample * gain;
        }
        let rest = take - first;
        if rest > 0 {
            for (slot, sample) in out[first..take].iter_mut().zip(&self.samples[..rest]) {
                *slot = *sample * gain;
            }
        }
        out[take..].fill(0.0);
        self.read = (self.read + take) % capacity;
        self.len -= take;
        self.played_frames += (take / channels.max(1)) as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1000 Hz stereo: 0.75 s = 750 frames = 1500 samples.
    fn shared() -> AudioShared {
        AudioShared::new(1000, 2)
    }

    #[test]
    fn capacity_is_three_quarters_of_a_second() {
        let shared = shared();
        assert_eq!(shared.room(), 1500);
        assert!(shared.push(shared.generation(), &[0.5; 1000]));
        assert_eq!(shared.room(), 500);
        assert_eq!(shared.buffered_frames(), 500);
    }

    #[test]
    fn reset_bumps_the_generation_and_clears_state() {
        let shared = shared();
        let generation = shared.generation();
        shared.push(generation, &[0.1; 200]);
        shared.set_running(true);
        let mut out = [0.0; 200];
        shared.fill(&mut out);
        shared.set_finished(generation);
        let next = shared.reset();
        assert_eq!(next, generation + 1);
        assert_eq!(shared.buffered_frames(), 0);
        assert_eq!(shared.played_frames(), 0);
        assert!(!shared.finished());
    }

    #[test]
    fn a_stale_generation_cannot_push_or_finish() {
        let shared = shared();
        let old = shared.generation();
        let new = shared.reset();
        assert!(!shared.push(old, &[1.0; 4]));
        shared.set_finished(old);
        assert!(!shared.finished());
        assert!(shared.push(new, &[1.0; 4]));
        shared.set_finished(new);
        assert!(shared.finished());
    }

    #[test]
    fn stopped_output_is_silent_and_consumes_nothing() {
        let shared = shared();
        shared.push(shared.generation(), &[0.5; 8]);
        let mut out = [9.0; 8];
        shared.fill(&mut out);
        assert_eq!(out, [0.0; 8]);
        assert_eq!(shared.buffered_frames(), 4);
        assert_eq!(shared.played_frames(), 0);
    }

    #[test]
    fn running_output_consumes_and_counts_frames() {
        let shared = shared();
        shared.push(shared.generation(), &[0.25; 8]);
        shared.set_running(true);
        let mut out = [0.0; 4];
        shared.fill(&mut out);
        assert_eq!(out, [0.25; 4]);
        assert_eq!(shared.played_frames(), 2);
        assert_eq!(shared.buffered_frames(), 2);
    }

    #[test]
    fn a_shortfall_is_zeroed_and_only_real_frames_count() {
        let shared = shared();
        shared.push(shared.generation(), &[0.5; 4]);
        shared.set_running(true);
        let mut out = [7.0; 8];
        shared.fill(&mut out);
        assert_eq!(out, [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(shared.played_frames(), 2);
    }

    #[test]
    fn volume_and_mute_scale_without_stopping_the_clock() {
        let shared = shared();
        shared.set_running(true);
        shared.push(shared.generation(), &[1.0; 12]);
        shared.set_volume(0.5);
        let mut out = [0.0; 4];
        shared.fill(&mut out);
        assert_eq!(out, [0.5; 4]);
        shared.set_muted(true);
        let mut out = [0.0; 4];
        shared.fill(&mut out);
        assert_eq!(out, [0.0; 4]);
        assert_eq!(shared.played_frames(), 4);
    }

    #[test]
    fn the_ring_wraps_across_reads_and_writes() {
        let shared = shared();
        shared.set_running(true);
        let generation = shared.generation();
        // Consume most of the ring, then push across the wrap point.
        assert!(shared.push(generation, &[1.0; 1400]));
        let mut out = [0.0; 1400];
        shared.fill(&mut out);
        assert_eq!(shared.room(), 1500);
        assert!(shared.push(generation, &[2.0; 1500]));
        let mut out = [0.0; 1500];
        shared.fill(&mut out);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1499], 2.0);
    }

    #[test]
    fn a_full_ring_wakes_the_feeder_when_the_callback_drains_it() {
        let shared = Arc::new(shared());
        let generation = shared.generation();
        assert!(shared.push_wait(generation, &[0.5; 1500]));
        let feeder = {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || shared.push_wait(generation, &[1.0; 500]))
        };
        // The feeder is parked on the condvar; consuming 500 samples makes
        // exactly the room its 500-sample push needs.
        std::thread::sleep(Duration::from_millis(20));
        shared.set_running(true);
        let mut out = [0.0; 500];
        shared.fill(&mut out);
        assert!(feeder.join().unwrap());
        assert_eq!(shared.buffered_frames(), 750);
    }

    #[test]
    fn a_blocked_feeder_gives_up_when_a_seek_resets_the_ring() {
        let shared = Arc::new(shared());
        let generation = shared.generation();
        assert!(shared.push_wait(generation, &[0.5; 1500]));
        let feeder = {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || shared.push_wait(generation, &[1.0; 1500]))
        };
        std::thread::sleep(Duration::from_millis(20));
        assert_ne!(shared.reset(), generation);
        assert!(!feeder.join().unwrap());
        assert_eq!(shared.buffered_frames(), 0);
    }

    #[test]
    fn volume_is_clamped_and_rejects_non_finite_values() {
        let shared = shared();
        shared.set_volume(3.0);
        assert_eq!(f32::from_bits(shared.volume.load(Ordering::Relaxed)), 1.0);
        shared.set_volume(-1.0);
        assert_eq!(f32::from_bits(shared.volume.load(Ordering::Relaxed)), 0.0);
        shared.set_volume(f32::NAN);
        assert_eq!(f32::from_bits(shared.volume.load(Ordering::Relaxed)), 0.0);
    }

    /// Run `seconds` of a stereo sine at `hz` through the meters, one 10 ms
    /// callback at a time, and return the levels at the end.
    fn meter_tone(hz: f32, seconds: f32) -> [f32; BANDS] {
        const RATE: u32 = 48_000;
        let shared = AudioShared::new(RATE, 2);
        let chunk = RATE as usize / 100; // 10 ms of frames
        let mut phase = 0.0f32;
        let step = std::f32::consts::TAU * hz / RATE as f32;
        for _ in 0..(seconds * 100.0) as usize {
            let mut buf = vec![0.0f32; chunk * 2];
            for frame in buf.chunks_exact_mut(2) {
                let s = 0.5 * phase.sin();
                frame[0] = s;
                frame[1] = s;
                phase += step;
            }
            shared.measure(&buf);
        }
        shared.levels()
    }

    #[test]
    fn silence_has_zero_levels() {
        let shared = shared();
        shared.measure(&[0.0; 960]);
        assert_eq!(shared.levels(), [0.0; BANDS]);
    }

    #[test]
    fn a_tone_lights_only_its_own_band() {
        for (hz, band) in [(100.0, 0), (1000.0, 1), (8000.0, 2)] {
            let levels = meter_tone(hz, 0.5);
            assert!(levels[band] > 0.9, "{hz} Hz: {levels:?}");
            for (other, level) in levels.iter().enumerate() {
                if other != band {
                    assert!(*level < levels[band] - 0.5, "{hz} Hz leaks into {other}: {levels:?}");
                }
            }
        }
    }

    #[test]
    fn the_scope_holds_the_latest_mono_output_oldest_first() {
        let shared = shared();
        shared.set_running(true);
        let generation = shared.generation();
        let samples: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert!(shared.push(generation, &samples));
        let mut out = vec![0.0; 8];
        shared.fill(&mut out);
        let mut latest = [9.0f32; 3];
        shared.scope(&mut latest);
        // Frames (0,1) (2,3) (4,5) (6,7) mix to 0.5, 2.5, 4.5, 6.5.
        assert_eq!(latest, [2.5, 4.5, 6.5]);
    }

    #[test]
    fn levels_decay_when_the_sound_stops() {
        let shared = shared();
        shared.measure(&[0.5; 960]);
        let loud = shared.levels();
        assert!(loud.iter().any(|l| *l > 0.0));
        for _ in 0..200 {
            shared.measure(&[0.0; 960]);
        }
        assert!(shared.levels().iter().all(|l| *l < 0.01));
    }
}
