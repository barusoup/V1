// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Frame timing for the UI's clocks.
//!
//! Every spring, tween and scroll glide advances by the frame's `dt`. The clock turns the wall-clock
//! gap between frames into that `dt`, smoothed so one slow frame does not make motion stutter.
//!
//! The clock has to tell a steady frame from a wake. While anything is moving the host polls, so
//! frames follow one another within a few tens of milliseconds. A longer gap means the app was
//! parked with nothing in flight, and the frame that ends the pause is not a frame time. Blending
//! that gap into the smoothed value would hand the first frames back a step near twice the nominal
//! one, decaying over the next ten or so frames: the stutter after a pause.

use std::time::Instant;

/// The step a frame takes when there is no history to measure. A wake frame uses it too.
pub const NOMINAL_DT: f32 = 1.0 / 60.0;
/// A gap longer than this is a wake, not a frame.
const WAKE_GAP: f32 = 0.1;
/// Steady gaps are clamped to this before smoothing, so a long stall is never a large step.
const MAX_STEP: f32 = 0.1;
/// Weight of the newest frame in the smoothed step.
const WEIGHT: f32 = 0.2;

/// The UI's frame clock. Owned by the host; one [`FrameClock::tick`] per frame.
pub struct FrameClock {
    last: Instant,
    smoothed: f32,
}

impl FrameClock {
    pub fn new(now: Instant) -> FrameClock {
        FrameClock {
            last: now,
            smoothed: NOMINAL_DT,
        }
    }

    /// Advance to `now` and return this frame's step in seconds. Always positive.
    pub fn tick(&mut self, now: Instant) -> f32 {
        let gap = now.saturating_duration_since(self.last).as_secs_f32();
        self.last = now;
        self.smoothed = if gap > WAKE_GAP {
            NOMINAL_DT
        } else {
            self.smoothed + (gap.min(MAX_STEP) - self.smoothed) * WEIGHT
        };
        self.smoothed.max(0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn ms(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    /// Steady 60 Hz frames settle to the nominal step.
    #[test]
    fn steady_frames_measure_their_own_gap() {
        let start = Instant::now();
        let mut clock = FrameClock::new(start);
        let mut now = start;
        let mut last = 0.0;
        for _ in 0..60 {
            now += ms(17);
            last = clock.tick(now);
        }
        assert!((last - 0.017).abs() < 0.001, "steady step {last}");
    }

    /// The frame that ends a long pause takes the nominal step, not the pause and not a blend of it.
    #[test]
    fn a_wake_takes_the_nominal_step() {
        let start = Instant::now();
        let mut clock = FrameClock::new(start);
        let mut now = start;
        for _ in 0..60 {
            now += ms(17);
            clock.tick(now);
        }
        now += Duration::from_secs(3);
        assert_eq!(clock.tick(now), NOMINAL_DT);
    }

    /// After a wake the next frames run at the normal rate at once. Nothing decays from a spike.
    #[test]
    fn nothing_decays_after_a_wake() {
        let start = Instant::now();
        let mut clock = FrameClock::new(start);
        let mut now = start + Duration::from_secs(5);
        clock.tick(now);
        for _ in 0..12 {
            now += ms(17);
            let step = clock.tick(now);
            assert!((step - 0.017).abs() < 0.0015, "post-wake step {step}");
        }
    }

    /// A slow frame during motion (shorter than a wake) is still smoothed, not reset to nominal.
    #[test]
    fn a_slow_frame_is_blended_not_reset() {
        let start = Instant::now();
        let mut clock = FrameClock::new(start);
        let mut now = start;
        for _ in 0..60 {
            now += ms(17);
            clock.tick(now);
        }
        now += ms(50);
        let step = clock.tick(now);
        let expected = NOMINAL_DT + (0.05 - NOMINAL_DT) * WEIGHT;
        assert!((step - expected).abs() < 0.001, "blended step {step}");
    }

    /// Two ticks at the same instant must not produce a zero or negative step.
    #[test]
    fn the_step_is_never_zero() {
        let start = Instant::now();
        let mut clock = FrameClock::new(start);
        for _ in 0..200 {
            assert!(clock.tick(start) > 0.0);
        }
    }
}
