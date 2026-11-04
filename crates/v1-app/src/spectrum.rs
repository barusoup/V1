// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The spectrum behind the lyric page: the sound being output, through an FFT, folded into
//! log-spaced bands with meter-like ballistics.
//!
//! Everything is preallocated; a frame's analysis allocates nothing.

use std::f32::consts::TAU;

/// FFT length. At 48 kHz this is 85 ms of sound and 11.7 Hz per bin, fine enough to separate the
/// bass bands without smearing the beat.
pub const FFT_LEN: usize = 4096;
/// Number of bands, low to high.
pub const BANDS: usize = 36;
/// The span the bands cover, in Hz.
const LOW_HZ: f32 = 35.0;
const HIGH_HZ: f32 = 16_000.0;
/// Music falls off at about 3–4 dB per octave; this tilt (dB per octave, pivoting at 1 kHz) levels
/// it, so the highs move as much as the lows instead of lying flat.
const TILT_DB_PER_OCTAVE: f32 = 3.5;
/// Display range: dBFS (after the tilt) mapped onto 0..1.
const FLOOR_DB: f32 = -64.0;
const CEIL_DB: f32 = -14.0;
/// Ballistics: a near-instant rise so hits land, and a short fall that tracks the sound closely.
const ATTACK_S: f32 = 0.008;
const RELEASE_S: f32 = 0.06;

pub struct Spectrum {
    window: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    /// Bit-reversal permutation and twiddles for `FFT_LEN`.
    reversed: Vec<u32>,
    twiddles: Vec<(f32, f32)>,
    /// Smoothed band levels, 0..=1.
    levels: [f32; BANDS],
    /// Raw mono samples, filled by the caller before `analyse`.
    pub samples: Vec<f32>,
}

impl Default for Spectrum {
    fn default() -> Self {
        Self::new()
    }
}

impl Spectrum {
    pub fn new() -> Self {
        let bits = FFT_LEN.trailing_zeros();
        let reversed = (0..FFT_LEN as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        let twiddles = (0..FFT_LEN / 2)
            .map(|k| {
                let (s, c) = (-TAU * k as f32 / FFT_LEN as f32).sin_cos();
                (c, s)
            })
            .collect();
        // Hann window.
        let window = (0..FFT_LEN)
            .map(|i| 0.5 - 0.5 * (TAU * i as f32 / (FFT_LEN - 1) as f32).cos())
            .collect();
        Self {
            window,
            re: vec![0.0; FFT_LEN],
            im: vec![0.0; FFT_LEN],
            reversed,
            twiddles,
            levels: [0.0; BANDS],
            samples: vec![0.0; FFT_LEN],
        }
    }

    pub fn levels(&self) -> &[f32; BANDS] {
        &self.levels
    }

    /// Whether anything is still showing (rising, or falling back to rest).
    pub fn active(&self) -> bool {
        self.levels.iter().any(|v| *v > 0.002)
    }

    /// Analyse `self.samples` (at `rate` Hz) and advance the bands by `dt` seconds.
    pub fn analyse(&mut self, rate: u32, dt: f32) {
        let mut targets = [0.0f32; BANDS];
        self.band_targets(rate.max(1) as f32, &mut targets);
        self.advance(&targets, dt);
    }

    /// Let everything fall back to rest, as if the sound had stopped.
    pub fn decay(&mut self, dt: f32) {
        self.advance(&[0.0; BANDS], dt);
    }

    /// Set the bands directly (for `--shot`, which has no sound).
    pub fn set(&mut self, levels: [f32; BANDS]) {
        self.levels = levels;
    }

    fn advance(&mut self, targets: &[f32; BANDS], dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        let attack = 1.0 - (-dt / ATTACK_S).exp();
        let release = 1.0 - (-dt / RELEASE_S).exp();
        for (level, &target) in self.levels.iter_mut().zip(targets) {
            let k = if target > *level { attack } else { release };
            *level += k * (target - *level);
        }
    }

    /// The instantaneous level of each band, 0..=1, from the current samples.
    fn band_targets(&mut self, rate: f32, out: &mut [f32; BANDS]) {
        // Window, then FFT in place.
        for i in 0..FFT_LEN {
            let j = self.reversed[i] as usize;
            self.re[j] = self.samples[i] * self.window[i];
            self.im[j] = 0.0;
        }
        let mut size = 2;
        while size <= FFT_LEN {
            let half = size / 2;
            let step = FFT_LEN / size;
            for start in (0..FFT_LEN).step_by(size) {
                for k in 0..half {
                    let (wr, wi) = self.twiddles[k * step];
                    let (a, b) = (start + k, start + k + half);
                    let tr = self.re[b] * wr - self.im[b] * wi;
                    let ti = self.re[b] * wi + self.im[b] * wr;
                    self.re[b] = self.re[a] - tr;
                    self.im[b] = self.im[a] - ti;
                    self.re[a] += tr;
                    self.im[a] += ti;
                }
            }
            size *= 2;
        }

        // Amplitude of a full-scale sine reads as 1: the Hann window's coherent gain is 0.5.
        let norm = 2.0 / (FFT_LEN as f32 * 0.5);
        let bin_hz = rate / FFT_LEN as f32;
        let nyquist_bin = FFT_LEN / 2 - 1;
        let magnitude = |re: &[f32], im: &[f32], bin: usize| -> f32 {
            let bin = bin.min(nyquist_bin);
            (re[bin] * re[bin] + im[bin] * im[bin]).sqrt() * norm
        };
        let ratio = HIGH_HZ / LOW_HZ;
        for (band, slot) in out.iter_mut().enumerate() {
            let lo_hz = LOW_HZ * ratio.powf(band as f32 / BANDS as f32);
            let hi_hz = LOW_HZ * ratio.powf((band + 1) as f32 / BANDS as f32);
            let centre_hz = (lo_hz * hi_hz).sqrt();
            let lo = (lo_hz / bin_hz).floor() as usize;
            let hi = (hi_hz / bin_hz).ceil() as usize;
            // A band narrower than a bin takes the bin it falls in, interpolated so neighbouring
            // bass bands don't read as identical steps.
            let amplitude = if hi <= lo + 1 {
                let pos = centre_hz / bin_hz;
                let i = pos.floor() as usize;
                let f = pos - i as f32;
                magnitude(&self.re, &self.im, i) * (1.0 - f)
                    + magnitude(&self.re, &self.im, i + 1) * f
            } else {
                (lo..hi)
                    .map(|bin| magnitude(&self.re, &self.im, bin))
                    .fold(0.0, f32::max)
            };
            let db = 20.0 * amplitude.max(1e-9).log10()
                + TILT_DB_PER_OCTAVE * (centre_hz / 1000.0).log2();
            *slot = ((db - FLOOR_DB) / (CEIL_DB - FLOOR_DB)).clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, rate: u32, amplitude: f32) -> Spectrum {
        let mut s = Spectrum::new();
        for (i, x) in s.samples.iter_mut().enumerate() {
            *x = amplitude * (TAU * hz * i as f32 / rate as f32).sin();
        }
        for _ in 0..60 {
            s.analyse(rate, 1.0 / 60.0);
        }
        s
    }

    #[test]
    fn a_tone_lights_the_band_it_falls_in() {
        for hz in [60.0, 440.0, 5000.0] {
            let s = tone(hz, 48_000, 0.05);
            let loudest = (0..BANDS)
                .max_by(|a, b| s.levels()[*a].total_cmp(&s.levels()[*b]))
                .unwrap();
            // The band the tone falls in, give or take one: a bass band is narrower than the
            // window's main lobe, so a low tone spreads over its neighbours.
            let ratio = HIGH_HZ / LOW_HZ;
            let expected = ((hz / LOW_HZ).ln() / ratio.ln() * BANDS as f32).floor() as usize;
            assert!(loudest.abs_diff(expected) <= 1, "{hz} Hz lit band {loudest}, not {expected}");
            assert!(s.levels()[loudest] > 0.4, "{hz} Hz: {:?}", s.levels());
            // Far bands stay dark.
            let far = if loudest > BANDS / 2 { 0 } else { BANDS - 1 };
            assert!(s.levels()[far] < 0.05, "{hz} Hz leaks: {:?}", s.levels());
        }
    }

    #[test]
    fn silence_is_dark_and_decay_returns_to_rest() {
        let mut s = tone(440.0, 48_000, 0.5);
        assert!(s.active());
        s.samples.fill(0.0);
        for _ in 0..240 {
            s.analyse(48_000, 1.0 / 60.0);
        }
        assert!(!s.active(), "{:?}", s.levels());
    }
}
