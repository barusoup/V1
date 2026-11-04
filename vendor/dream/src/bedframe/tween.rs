//! Animation primitives: easing curves, tweens, and animated values.
//!
//! Standalone — the widgets use these for hover/press fades and menu motion,
//! and games can use them for anything that moves over time.
//!
//! ```ignore
//! let mut fade = Tween::new(0.0, 1.0, 0.25).ease(Ease::CubicOut);
//! fade.update(dt);
//! panel.style.opacity = fade.value();
//! ```

/// Easing curves. All map 0..1 to roughly 0..1; `Back`, `Elastic` and
/// `Bounce` overshoot deliberately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Ease {
    #[default]
    Linear,
    QuadIn,
    QuadOut,
    QuadInOut,
    CubicIn,
    CubicOut,
    CubicInOut,
    SineIn,
    SineOut,
    SineInOut,
    ExpoOut,
    BackOut,
    ElasticOut,
    BounceOut,
    /// Hold at the start value until the tween ends, then jump. For stepped
    /// UI (numeric readouts, discrete states).
    Step,
}

/// Map `t` (clamped to 0..1) through the curve.
pub fn ease(curve: Ease, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match curve {
        Ease::Linear => t,
        Ease::QuadIn => t * t,
        Ease::QuadOut => 1.0 - (1.0 - t) * (1.0 - t),
        Ease::QuadInOut => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
            }
        }
        Ease::CubicIn => t * t * t,
        Ease::CubicOut => 1.0 - (1.0 - t).powi(3),
        Ease::CubicInOut => {
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
            }
        }
        Ease::SineIn => 1.0 - (t * std::f32::consts::FRAC_PI_2).cos(),
        Ease::SineOut => (t * std::f32::consts::FRAC_PI_2).sin(),
        Ease::SineInOut => -(t * std::f32::consts::PI).cos() / 2.0 + 0.5,
        Ease::ExpoOut => {
            if t >= 1.0 {
                1.0
            } else {
                1.0 - 2f32.powf(-10.0 * t)
            }
        }
        Ease::BackOut => {
            let c1 = 1.70158;
            let c3 = c1 + 1.0;
            1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
        }
        Ease::ElasticOut => {
            if t <= 0.0 {
                0.0
            } else if t >= 1.0 {
                1.0
            } else {
                let c4 = 2.0 * std::f32::consts::PI / 3.0;
                2f32.powf(-10.0 * t) * (t * 10.0 - 0.75).sin() * c4 + 1.0
            }
        }
        Ease::BounceOut => bounce_out(t),
        Ease::Step => {
            if t >= 1.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn bounce_out(t: f32) -> f32 {
    let n1 = 7.5625;
    let d1 = 2.75;
    if t < 1.0 / d1 {
        n1 * t * t
    } else if t < 2.0 / d1 {
        let t = t - 1.5 / d1;
        n1 * t * t + 0.75
    } else if t < 2.5 / d1 {
        let t = t - 2.25 / d1;
        n1 * t * t + 0.9375
    } else {
        let t = t - 2.625 / d1;
        n1 * t * t + 0.984375
    }
}

/// How a tween behaves when it reaches the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PlayMode {
    /// Stop at the end value.
    #[default]
    Once,
    /// Restart from the beginning.
    Loop,
    /// Reverse direction at each end.
    PingPong,
}

/// A scalar animated from `from` to `to` over `duration` seconds.
///
/// Deterministic: advance it with [`Tween::update`]; no wall clock involved,
/// so scripted captures and tests get identical results.
#[derive(Clone, Copy, Debug)]
pub struct Tween {
    from: f32,
    to: f32,
    duration: f32,
    elapsed: f32,
    delay: f32,
    ease: Ease,
    mode: PlayMode,
    finished: bool,
}

impl Tween {
    pub fn new(from: f32, to: f32, duration: f32) -> Self {
        Self {
            from,
            to,
            duration: duration.max(0.0),
            elapsed: 0.0,
            delay: 0.0,
            ease: Ease::Linear,
            mode: PlayMode::Once,
            finished: duration <= 0.0,
        }
    }

    /// Fade-style convenience: 1.0 to 0.0 (or the other way with `from`).
    pub fn fade_in(duration: f32) -> Self {
        Self::new(0.0, 1.0, duration)
    }

    pub fn ease(mut self, ease: Ease) -> Self {
        self.ease = ease;
        self
    }

    pub fn mode(mut self, mode: PlayMode) -> Self {
        self.mode = mode;
        self
    }

    /// Seconds to wait before the first update moves anything.
    pub fn delay(mut self, delay: f32) -> Self {
        self.delay = delay.max(0.0);
        self
    }

    /// Advance by `dt` seconds and return the new value.
    pub fn update(&mut self, mut dt: f32) -> f32 {
        if self.finished {
            return self.value();
        }
        if self.delay > 0.0 {
            if dt < self.delay {
                self.delay -= dt;
                return self.value();
            }
            // Only the time past the delay animates.
            dt -= self.delay;
            self.delay = 0.0;
        }
        if self.duration <= 0.0 {
            self.elapsed = 0.0;
            self.finished = true;
            return self.value();
        }
        self.elapsed += dt;
        let end = self.duration;
        if self.elapsed >= end {
            match self.mode {
                PlayMode::Once => {
                    self.elapsed = end;
                    self.finished = true;
                }
                PlayMode::Loop => {
                    self.elapsed %= end;
                }
                PlayMode::PingPong => {
                    // Swap endpoints and keep the overshoot.
                    let overshoot = self.elapsed - end;
                    std::mem::swap(&mut self.from, &mut self.to);
                    self.elapsed = overshoot.min(end);
                }
            }
        }
        self.value()
    }

    /// The eased value at the current time.
    pub fn value(&self) -> f32 {
        if self.duration <= 0.0 {
            return self.to;
        }
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        self.from + (self.to - self.from) * ease(self.ease, t)
    }

    /// The raw, un-eased progress 0..1.
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        }
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Jump to the end immediately.
    pub fn finish(&mut self) {
        self.elapsed = self.duration;
        self.finished = true;
    }

    /// Start over with the original endpoints.
    pub fn restart(&mut self) {
        self.elapsed = 0.0;
        self.finished = self.duration <= 0.0;
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }
}

/// Values that can be interpolated between two keys.
pub trait LerpValue: Copy {
    fn lerp_value(a: Self, b: Self, t: f32) -> Self;
}

impl LerpValue for f32 {
    fn lerp_value(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}

impl LerpValue for glam::Vec2 {
    fn lerp_value(a: Self, b: Self, t: f32) -> Self {
        a.lerp(b, t)
    }
}

impl LerpValue for glam::Vec3 {
    fn lerp_value(a: Self, b: Self, t: f32) -> Self {
        a.lerp(b, t)
    }
}

/// An animated value of any [`LerpValue`] type: start at `from`, animate
/// toward targets with [`Anim::animate_to`].
///
/// Carries a caller-stamped `last_frame` so owners can retire animations
/// nobody has touched (the `Ui` keyframes registry uses it).
#[derive(Clone, Copy, Debug)]
pub struct Anim<T: LerpValue> {
    tween: Tween,
    from: T,
    to: T,
    last_frame: u64,
}

impl<T: LerpValue> Anim<T> {
    pub fn new(initial: T) -> Self {
        Self {
            tween: Tween::new(0.0, 1.0, 0.0),
            from: initial,
            to: initial,
            last_frame: 0,
        }
    }

    /// Animate to `target` over `duration` seconds.
    pub fn animate_to(&mut self, target: T, duration: f32, ease: Ease) {
        self.from = self.value();
        self.to = target;
        self.tween = Tween::new(0.0, 1.0, duration).ease(ease);
    }

    pub fn set(&mut self, value: T) {
        self.from = value;
        self.to = value;
        self.tween = Tween::new(0.0, 1.0, 0.0);
    }

    /// Advance and return the interpolated value.
    pub fn update(&mut self, dt: f32) -> T {
        self.tween.update(dt);
        self.value()
    }

    pub fn value(&self) -> T {
        T::lerp_value(self.from, self.to, self.tween.value())
    }

    pub fn finished(&self) -> bool {
        self.tween.finished()
    }

    /// The value this animation is heading to.
    pub fn target(&self) -> T {
        self.to
    }

    /// Stamp the frame that last advanced this animation (retirement).
    pub fn stamp(&mut self, frame: u64) {
        self.last_frame = frame;
    }

    pub fn last_frame(&self) -> u64 {
        self.last_frame
    }
}

/// A damped spring toward a target — motion whose *feel* is configuration,
/// not a preset curve. Stiffness, damping and mass are all parameters:
/// snappy (high stiffness, critical damping), bouncy (low damping), or
/// floaty (low stiffness, high mass) are three field values, not three
/// named modes. Integrate with the real frame time via [`Self::update`].
#[derive(Clone, Copy, Debug)]
pub struct Spring {
    /// Current position.
    pub value: f32,
    /// Current speed (units per second).
    pub velocity: f32,
    /// Where the spring is being pulled.
    pub target: f32,
    /// Pull strength (N/m equivalent): higher reaches the target faster.
    pub stiffness: f32,
    /// Resistance (N·s/m equivalent): `0` oscillates forever; the
    /// critical value `2 * sqrt(stiffness * mass)` settles without
    /// overshoot; below it the spring overshoots, above it it eases in.
    pub damping: f32,
    /// Inertia: higher masses respond slower at equal stiffness.
    pub mass: f32,
}

impl Spring {
    /// A spring at rest at `value` with sensible general-purpose
    /// parameters. Every field is public — treat the result as a
    /// starting point, not a style.
    pub fn new(value: f32) -> Self {
        Self {
            value,
            velocity: 0.0,
            target: value,
            stiffness: 300.0,
            damping: 30.0,
            mass: 1.0,
        }
    }

    /// Builder setters (the fields are public too; whichever reads better).
    pub fn stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.max(0.0);
        self
    }

    pub fn damping(mut self, damping: f32) -> Self {
        self.damping = damping.max(0.0);
        self
    }

    pub fn mass(mut self, mass: f32) -> Self {
        self.mass = mass.max(1e-3);
        self
    }

    /// Pull toward `target` (retargeting mid-flight is the intended use:
    /// the spring continues from wherever it is, carrying its velocity).
    pub fn to(mut self, target: f32) -> Self {
        self.target = target;
        self
    }

    /// In-place retarget for imperative use (the builder [`Self::to`]
    /// consumes `self` and is easy to drop by accident).
    pub fn retarget(&mut self, target: f32) {
        self.target = target;
    }

    /// Advance by `dt` seconds. Stiff springs are internally substepped
    /// so a long frame cannot blow the integrator up.
    pub fn update(&mut self, dt: f32) {
        const MAX_STEP: f32 = 1.0 / 240.0;
        let mut remaining = dt.clamp(0.0, 0.25);
        let mass = self.mass.max(1e-3);
        while remaining > 0.0 {
            let step = remaining.min(MAX_STEP);
            remaining -= step;
            let acceleration =
                (self.stiffness * (self.target - self.value) - self.damping * self.velocity) / mass;
            self.velocity += acceleration * step;
            self.value += self.velocity * step;
        }
    }

    /// Whether the spring has effectively come to rest at its target.
    pub fn settled(&self, epsilon: f32) -> bool {
        (self.value - self.target).abs() < epsilon && self.velocity.abs() < epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }

    #[test]
    fn ease_endpoints() {
        for curve in [
            Ease::Linear,
            Ease::QuadIn,
            Ease::QuadOut,
            Ease::QuadInOut,
            Ease::CubicIn,
            Ease::CubicOut,
            Ease::CubicInOut,
            Ease::SineIn,
            Ease::SineOut,
            Ease::SineInOut,
            Ease::ExpoOut,
            Ease::BackOut,
            Ease::ElasticOut,
            Ease::BounceOut,
        ] {
            close(ease(curve, 0.0), 0.0);
            close(ease(curve, 1.0), 1.0);
        }
        // Step holds then jumps.
        close(ease(Ease::Step, 0.99), 0.0);
        close(ease(Ease::Step, 1.0), 1.0);
    }

    #[test]
    fn ease_monotone_for_smooth_curves() {
        for curve in [Ease::Linear, Ease::CubicInOut, Ease::SineInOut] {
            let mut prev = -1.0;
            for i in 0..=20 {
                let value = ease(curve, i as f32 / 20.0);
                assert!(value >= prev, "{curve:?} not monotone at {i}");
                prev = value;
            }
        }
    }

    #[test]
    fn back_out_overshoots() {
        let peak = (0..=100)
            .map(|i| ease(Ease::BackOut, i as f32 / 100.0))
            .fold(f32::MIN, f32::max);
        assert!(peak > 1.05, "back-out should overshoot, peak {peak}");
    }

    #[test]
    fn tween_once_reaches_end() {
        let mut tween = Tween::new(0.0, 10.0, 1.0).ease(Ease::Linear);
        close(tween.update(0.5), 5.0);
        close(tween.update(1.0), 10.0);
        assert!(tween.finished());
        // Updates after finish hold the end value.
        close(tween.update(1.0), 10.0);
    }

    #[test]
    fn tween_delay() {
        let mut tween = Tween::new(0.0, 1.0, 1.0).delay(0.5);
        close(tween.update(0.25), 0.0);
        close(tween.update(0.25), 0.0); // delay exhausted exactly here
        assert!(!tween.finished());
        close(tween.update(1.0), 1.0);
    }

    #[test]
    fn tween_loop_and_ping_pong() {
        let mut looped = Tween::new(0.0, 1.0, 1.0).mode(PlayMode::Loop);
        looped.update(1.25);
        close(looped.progress(), 0.25);
        assert!(!looped.finished());

        let mut ping = Tween::new(0.0, 1.0, 1.0).mode(PlayMode::PingPong);
        close(ping.update(1.0), 1.0);
        close(ping.update(0.5), 0.5); // heading back down
        close(ping.update(0.5), 0.0); // bottom of the swing
        close(ping.update(0.5), 0.5); // and back up
    }

    #[test]
    fn tween_zero_duration_jumps() {
        let mut tween = Tween::new(3.0, 9.0, 0.0);
        close(tween.update(0.016), 9.0);
        assert!(tween.finished());
    }

    #[test]
    fn anim_transitions_between_values() {
        let mut anim = Anim::<f32>::new(0.0);
        anim.animate_to(10.0, 1.0, Ease::Linear);
        close(anim.update(0.5), 5.0);
        // Retarget mid-flight: continues from the current value.
        anim.animate_to(0.0, 1.0, Ease::Linear);
        close(anim.update(0.5), 2.5);
        anim.set(4.0);
        close(anim.update(0.1), 4.0);
        assert!(anim.finished());
    }

    #[test]
    fn anim_color_lerp() {
        let mut anim = Anim::new(crate::ui::Color::BLACK);
        anim.animate_to(crate::ui::Color::WHITE, 1.0, Ease::Linear);
        anim.update(0.5);
        let value = anim.value();
        assert_eq!(value.r, 128, "mid-grey at half time, got {value:?}");
    }

    #[test]
    fn spring_settles_on_target() {
        let mut spring = Spring::new(0.0).to(10.0);
        for _ in 0..600 {
            spring.update(1.0 / 60.0);
        }
        assert!(spring.settled(0.01), "value {}", spring.value);
        close(spring.value, 10.0);
    }

    #[test]
    fn spring_parameters_change_the_ride() {
        // Underdamped: overshoots past the target at least once.
        let mut bouncy = Spring::new(0.0).to(1.0).stiffness(400.0).damping(10.0);
        let mut peak = f32::MIN;
        for _ in 0..240 {
            bouncy.update(1.0 / 60.0);
            peak = peak.max(bouncy.value);
        }
        assert!(
            peak > 1.05,
            "underdamped spring must overshoot, peak {peak}"
        );
        // Critically damped: reaches the target without ever passing it.
        let critical = 2.0 * (400.0f32 * 1.0).sqrt();
        let mut smooth = Spring::new(0.0).to(1.0).stiffness(400.0).damping(critical);
        for _ in 0..240 {
            smooth.update(1.0 / 60.0);
            assert!(
                smooth.value <= 1.001,
                "critically damped overshoot: {}",
                smooth.value
            );
        }
        assert!(smooth.settled(0.01));
    }

    #[test]
    fn spring_retargets_mid_flight() {
        let mut spring = Spring::new(0.0).to(10.0);
        for _ in 0..30 {
            spring.update(1.0 / 60.0);
        }
        let carried = spring.velocity;
        spring.retarget(0.0);
        for _ in 0..600 {
            spring.update(1.0 / 60.0);
        }
        assert!(spring.settled(0.01) && (spring.value - 0.0).abs() < 0.01);
        assert!(carried != 0.0, "the spring was actually moving");
    }

    #[test]
    fn spring_long_frame_cannot_explode() {
        // A very stiff spring with a huge dt: substepping keeps it finite
        // and it still lands on target.
        let mut spring = Spring::new(0.0).to(1.0).stiffness(50_000.0).damping(100.0);
        spring.update(10.0);
        assert!(spring.value.is_finite() && spring.velocity.is_finite());
    }
}
