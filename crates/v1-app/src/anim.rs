// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Motion: one primitive, a critically damped spring.
//!
//! Every animated value in the UI (hover washes, the sidebar's width, the search field's reveal,
//! the play button's glyph) is a spring chasing a target. A spring keeps its velocity when the
//! target changes mid-flight, so a hover that is cancelled halfway turns around smoothly instead of
//! restarting, and there is never a frame where something snaps. Springs are critically damped, so
//! nothing overshoots.
//!
//! Values are keyed by `(name, index)` and created on first use *at their target*, so a fresh
//! widget appears in place instead of animating in from zero.

use std::collections::HashMap;

/// Below this distance and speed a spring is considered at rest.
const REST_DIST: f32 = 0.0008;
const REST_SPEED: f32 = 0.02;
/// Fixed integration step; keeps stiff springs stable whatever the frame time.
const STEP: f32 = 1.0 / 240.0;

/// Stiffness presets. Higher is faster; damping is derived so every one is critically damped.
pub mod rate {
    /// Hover washes and press feedback: settles in about 120 ms.
    pub const SNAP: f32 = 900.0;
    /// Brief hover reveals: settles in about 70 ms.
    pub const QUICKER: f32 = 2400.0;
    /// Small reveals (icons, labels): about 200 ms.
    pub const QUICK: f32 = 420.0;
    /// Panels and large shapes (the sidebar's width): about 240 ms.
    pub const PANEL: f32 = 420.0;
    /// Slower, for anything that should feel weighty.
    #[allow(dead_code)]
    pub const SMOOTH: f32 = 220.0;
    /// Pops: stiff, and paired with a damping ratio below 1 so they overshoot slightly and settle.
    #[allow(dead_code)]
    pub const POP: f32 = 900.0;
    /// Damping ratio for a subtle pop (about 8% overshoot).
    #[allow(dead_code)]
    pub const POP_DAMPING: f32 = 0.62;
}

/// A spring nobody has asked about for this many frames is forgotten (about ten seconds at 60 Hz).
/// Widgets key their springs by what they show (a cover's URL, a row's place), so without this every
/// cover and row ever drawn would leave a spring behind for the life of the process. Forgetting is
/// invisible: a spring is created at its target, which is where a resting one was anyway.
const FORGET_AFTER: u64 = 600;
/// How often the forgetting is done.
const FORGET_EVERY: u64 = 256;

#[derive(Clone, Copy)]
struct Slot {
    x: f32,
    v: f32,
    /// The frame this spring was last asked about.
    seen: u64,
}

#[derive(Default)]
pub struct Anims {
    slots: HashMap<(&'static str, u32), Slot>,
    dt: f32,
    moving: bool,
    frame: u64,
}

impl Anims {
    /// Start a frame. `dt` is the frame time in seconds.
    pub fn begin(&mut self, dt: f32) {
        self.dt = dt.clamp(0.0, 0.05);
        self.moving = false;
        self.frame += 1;
        if self.frame.is_multiple_of(FORGET_EVERY) {
            let now = self.frame;
            self.slots.retain(|_, slot| now - slot.seen < FORGET_AFTER);
        }
    }

    /// How many springs are being kept.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.slots.len()
    }

    /// This frame's time step, in seconds.
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Whether any spring moved this frame, so the host keeps presenting.
    pub fn moving(&self) -> bool {
        self.moving
    }

    /// Chase `target` and return the current value.
    pub fn spring(&mut self, name: &'static str, index: u32, target: f32, stiffness: f32) -> f32 {
        self.spring_damped(name, index, target, stiffness, 1.0)
    }

    /// A pop: chase `target` and overshoot it slightly before settling. 0 and 1 are the usual
    /// rest values, so the result may briefly leave that range; callers clamp what must not.
    #[allow(dead_code)]
    pub fn pop(&mut self, name: &'static str, index: u32, on: bool) -> f32 {
        let target = if on { 1.0 } else { 0.0 };
        self.spring_damped(name, index, target, rate::POP, rate::POP_DAMPING)
    }

    /// A spring with an explicit damping ratio: 1 is critically damped, below 1 overshoots.
    pub fn spring_damped(
        &mut self,
        name: &'static str,
        index: u32,
        target: f32,
        stiffness: f32,
        ratio: f32,
    ) -> f32 {
        let (dt, frame) = (self.dt, self.frame);
        let slot = self.slots.entry((name, index)).or_insert(Slot {
            x: target,
            v: 0.0,
            seen: frame,
        });
        slot.seen = frame;
        let damping = 2.0 * ratio * stiffness.sqrt();
        let mut left = dt;
        while left > 1e-6 {
            let h = left.min(STEP);
            let accel = stiffness * (target - slot.x) - damping * slot.v;
            slot.v += accel * h;
            slot.x += slot.v * h;
            left -= h;
        }
        if (target - slot.x).abs() < REST_DIST && slot.v.abs() < REST_SPEED {
            slot.x = target;
            slot.v = 0.0;
        } else {
            self.moving = true;
        }
        slot.x
    }

    /// A spring toward 1 when `on`, else 0. The common case: hover, press, open/closed.
    pub fn toggle(&mut self, name: &'static str, index: u32, on: bool, stiffness: f32) -> f32 {
        self.spring(name, index, if on { 1.0 } else { 0.0 }, stiffness)
    }

    /// Read a spring's last value without advancing it.
    pub fn value(&self, name: &'static str, index: u32) -> Option<f32> {
        self.slots.get(&(name, index)).map(|slot| slot.x)
    }

    /// Set a value outright, with no motion, for a change that must not animate.
    pub fn snap(&mut self, name: &'static str, index: u32, value: f32) {
        self.slots.insert(
            (name, index),
            Slot {
                x: value,
                v: 0.0,
                seen: self.frame,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `frames` frames of `dt` seconds toward `target`, returning every value seen.
    fn run(anims: &mut Anims, target: f32, stiffness: f32, frames: usize, dt: f32) -> Vec<f32> {
        (0..frames)
            .map(|_| {
                anims.begin(dt);
                anims.spring("t", 0, target, stiffness)
            })
            .collect()
    }

    #[test]
    fn springs_nobody_asks_about_are_forgotten_and_the_ones_in_use_are_kept() {
        let mut anims = Anims::default();
        anims.begin(1.0 / 60.0);
        // A thousand covers drawn once, and one hover that is drawn every frame.
        for index in 0..1000 {
            anims.toggle("cover", index, true, rate::QUICK);
        }
        for _ in 0..(FORGET_AFTER + 2 * FORGET_EVERY) {
            anims.begin(1.0 / 60.0);
            anims.toggle("hover", 0, true, rate::SNAP);
        }
        assert_eq!(anims.len(), 1, "only the spring still in use remains");
        // A forgotten spring comes back at its target, as a new one does.
        anims.begin(1.0 / 60.0);
        assert_eq!(anims.toggle("cover", 5, true, rate::QUICK), 1.0);
    }

    #[test]
    fn a_new_spring_starts_at_its_target() {
        let mut anims = Anims::default();
        anims.begin(1.0 / 60.0);
        assert_eq!(anims.spring("t", 0, 1.0, rate::SMOOTH), 1.0);
        assert!(!anims.moving());
    }

    #[test]
    fn it_settles_exactly_and_never_overshoots() {
        let mut anims = Anims::default();
        anims.begin(1.0 / 60.0);
        anims.spring("t", 0, 0.0, rate::SMOOTH);
        for stiffness in [rate::SNAP, rate::QUICK, rate::SMOOTH] {
            let values = run(&mut anims, 1.0, stiffness, 240, 1.0 / 60.0);
            assert!(values.iter().all(|v| *v <= 1.0), "overshot at {stiffness}");
            assert!(
                values.windows(2).all(|w| w[1] >= w[0] - 1e-6),
                "not monotonic"
            );
            assert_eq!(
                *values.last().unwrap(),
                1.0,
                "did not settle at {stiffness}"
            );
            let back = run(&mut anims, 0.0, stiffness, 240, 1.0 / 60.0);
            assert!(back.iter().all(|v| *v >= 0.0), "undershot at {stiffness}");
            assert_eq!(*back.last().unwrap(), 0.0);
        }
    }

    #[test]
    fn a_pop_overshoots_a_little_then_settles() {
        let mut anims = Anims::default();
        anims.begin(1.0 / 60.0);
        anims.pop("p", 0, false);
        let values: Vec<f32> = (0..120)
            .map(|_| {
                anims.begin(1.0 / 60.0);
                anims.pop("p", 0, true)
            })
            .collect();
        let peak = values.iter().cloned().fold(0.0, f32::max);
        assert!(peak > 1.02 && peak < 1.2, "peak {peak}");
        assert_eq!(*values.last().unwrap(), 1.0);
        // And it is quick: within 10% of rest inside a fifth of a second.
        assert!((values[11] - 1.0).abs() < 0.1, "slow: {}", values[11]);
    }

    #[test]
    fn a_reversal_turns_around_without_a_jump() {
        let mut anims = Anims::default();
        anims.begin(1.0 / 60.0);
        anims.spring("t", 0, 0.0, rate::SMOOTH);
        let up = run(&mut anims, 1.0, rate::SMOOTH, 8, 1.0 / 60.0);
        let turn = run(&mut anims, 0.0, rate::SMOOTH, 1, 1.0 / 60.0)[0];
        // The step across the reversal is as small as a normal frame step: no snap.
        let normal_step = up[7] - up[6];
        assert!((turn - up[7]).abs() <= normal_step.abs() + 1e-4);
    }

    #[test]
    fn frame_rate_does_not_change_the_motion() {
        let (mut a, mut b) = (Anims::default(), Anims::default());
        for anims in [&mut a, &mut b] {
            anims.begin(0.016);
            anims.spring("t", 0, 0.0, rate::SMOOTH);
        }
        let at_60 = run(&mut a, 1.0, rate::SMOOTH, 12, 1.0 / 60.0)[11];
        let at_120 = run(&mut b, 1.0, rate::SMOOTH, 24, 1.0 / 120.0)[23];
        assert!((at_60 - at_120).abs() < 0.01, "{at_60} vs {at_120}");
    }
}
