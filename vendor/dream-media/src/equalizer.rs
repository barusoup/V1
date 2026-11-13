//! A ten-band equalizer for the device callback.
//!
//! One RBJ peaking section per band, cascaded, per channel. The UI publishes gains from another
//! thread into atomics; the callback recomputes its coefficients only when they change, and skips
//! the filters entirely while every band is flat.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Bass to treble (standard 10-band ISO graphic equalizer).
pub(crate) const EQ_BANDS: usize = 10;
const FREQS: [f32; EQ_BANDS] = [
    31.0, 62.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0,
];
/// One-octave Q so adjacent bands blend smoothly without resonant peaks.
const Q: f32 = 1.41;
const MAX_DB: f32 = 12.0;

/// The gains, shared with the UI thread.
pub(crate) struct EqGains {
    gains: [AtomicU32; EQ_BANDS],
    dirty: AtomicBool,
    live: AtomicBool,
}

impl EqGains {
    pub(crate) fn new() -> EqGains {
        EqGains {
            gains: std::array::from_fn(|_| AtomicU32::new(0.0f32.to_bits())),
            dirty: AtomicBool::new(false),
            live: AtomicBool::new(false),
        }
    }

    /// Publish new gains, in dB. Out-of-range or non-finite values are made safe here, so the
    /// callback never has to guard them.
    pub(crate) fn store(&self, gains: [f32; EQ_BANDS]) {
        let mut live = false;
        for (slot, gain) in self.gains.iter().zip(gains) {
            let gain = if gain.is_finite() {
                gain.clamp(-MAX_DB, MAX_DB)
            } else {
                0.0
            };
            live |= gain.abs() > 0.01;
            slot.store(gain.to_bits(), Ordering::Relaxed);
        }
        self.live.store(live, Ordering::Release);
        self.dirty.store(true, Ordering::Release);
    }

    /// The gains, if they changed since the last call.
    pub(crate) fn take(&self) -> Option<[f32; EQ_BANDS]> {
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return None;
        }
        Some(std::array::from_fn(|band| {
            f32::from_bits(self.gains[band].load(Ordering::Relaxed))
        }))
    }

    /// Whether any band is doing anything. Flat gains skip the filters altogether.
    pub(crate) fn live(&self) -> bool {
        self.live.load(Ordering::Acquire)
    }
}

/// One peaking section, with a state pair per channel.
#[derive(Clone, Copy)]
struct Peaking {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: [f32; 2],
    z2: [f32; 2],
}

impl Peaking {
    fn flat() -> Peaking {
        Peaking {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: [0.0; 2],
            z2: [0.0; 2],
        }
    }

    /// RBJ cookbook peaking EQ. At 0 dB the coefficients come out as an identity.
    fn set(&mut self, db: f32, hz: f32, rate: f32) {
        let a = 10f32.powf(db / 40.0);
        let nyquist = (rate * 0.5).max(1.0);
        let w0 = std::f32::consts::TAU * hz.clamp(20.0, nyquist * 0.9) / rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * Q);
        let a0 = 1.0 + alpha / a;
        self.b0 = (1.0 + alpha * a) / a0;
        self.b1 = (-2.0 * cos) / a0;
        self.b2 = (1.0 - alpha * a) / a0;
        self.a1 = (-2.0 * cos) / a0;
        self.a2 = (1.0 - alpha / a) / a0;
    }

    fn run(&mut self, x: f32, channel: usize) -> f32 {
        let y = self.b0 * x + self.z1[channel];
        self.z1[channel] = self.b1 * x - self.a1 * y + self.z2[channel];
        self.z2[channel] = self.b2 * x - self.a2 * y;
        y
    }
}

/// The filter bank, owned by the device callback.
pub(crate) struct Eq {
    filters: [Peaking; EQ_BANDS],
    rate: f32,
}

impl Eq {
    pub(crate) fn new(rate: f32) -> Eq {
        Eq {
            filters: [Peaking::flat(); EQ_BANDS],
            rate: rate.max(1.0),
        }
    }

    pub(crate) fn set(&mut self, gains: [f32; EQ_BANDS]) {
        for (band, (filter, gain)) in self.filters.iter_mut().zip(gains).enumerate() {
            filter.set(gain, FREQS[band], self.rate);
        }
    }

    /// Zero the state, so a seek does not carry the old signal's tail into the new position.
    pub(crate) fn clear(&mut self) {
        for filter in &mut self.filters {
            filter.z1 = [0.0; 2];
            filter.z2 = [0.0; 2];
        }
    }

    /// Filter interleaved samples in place. Only the first two channels are shaped; anything more
    /// is left alone, as the app's output is stereo.
    pub(crate) fn apply(&mut self, out: &mut [f32], channels: usize) {
        if channels == 0 {
            return;
        }
        for frame in out.chunks_exact_mut(channels) {
            for (channel, sample) in frame.iter_mut().take(2).enumerate() {
                let mut x = *sample;
                for filter in &mut self.filters {
                    x = filter.run(x, channel);
                }
                *sample = x.clamp(-1.0, 1.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, rate: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let x = (std::f32::consts::TAU * hz * i as f32 / rate).sin() * 0.4;
                [x, x]
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn flat_gains_leave_the_signal_alone() {
        let rate = 48_000.0;
        let input = tone(1_000.0, rate, 4_800);
        let mut eq = Eq::new(rate);
        eq.set([0.0; EQ_BANDS]);
        let mut output = input.clone();
        eq.apply(&mut output, 2);
        for (a, b) in input.iter().zip(&output) {
            assert!(
                (a - b).abs() < 1e-4,
                "flat EQ changed the signal: {a} vs {b}"
            );
        }
    }

    #[test]
    fn boosting_a_band_raises_the_tone_in_it() {
        let rate = 48_000.0;
        let input = tone(1_000.0, rate, 4_800);
        let mut boosted = Eq::new(rate);
        boosted.set([0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0]);
        let mut output = input.clone();
        boosted.apply(&mut output, 2);
        assert!(
            rms(&output) > rms(&input) * 1.5,
            "a +12 dB bell should be clearly louder: {} vs {}",
            rms(&output),
            rms(&input)
        );
    }

    #[test]
    fn gains_are_published_and_taken_once() {
        let gains = EqGains::new();
        assert!(gains.take().is_none());
        gains.store([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(gains.live());
        assert_eq!(
            gains.take(),
            Some([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert!(gains.take().is_none());
    }
}
