use super::{LoopState, RateBucket, SpringAnim, Ui};
use crate::ui::Color;
use crate::ui::tween::{Anim, Ease, Spring};

impl Ui {
    // --- Animation rate control -----------------------------------------

    /// Cap how fast animation advances for a group of widgets, in hertz.
    ///
    /// Everything drawn between this call and [`Self::end_rate_scope`]
    /// advances its animations in whole `1/hz` steps: the app keeps
    /// presenting at the display's rate, but that group only changes `hz`
    /// times a second. That is the per-element (one widget) and per-group
    /// (whatever the scope wraps) dial; the app-wide cap lives on
    /// [`crate::app::App::pacing`]. `None` restores the display rate.
    ///
    /// ```
    /// use dream::bedframe::atlas::AtlasConfig;
    /// use dream::bedframe::tween::Ease;
    /// use dream::bedframe::widgets::Ui;
    ///
    /// let mut ui = Ui::new(Default::default(), AtlasConfig::default());
    /// ui.begin(Default::default(), (800.0, 600.0));
    /// ui.rate_scope("hud", Some(30.0));      // this group eases at 30 Hz
    /// ui.anim_f32("hud.pulse", 1.0, 0.2, Ease::Linear);
    /// ui.end_rate_scope();                   // later widgets are uncapped
    /// ui.anim_f32("menu.panel", 1.0, 0.2, Ease::Linear);
    /// let _ = ui.end();
    /// ```
    pub fn rate_scope(&mut self, key: &str, hz: Option<f32>) {
        let id = self.id(key);
        let frame_dt = self.input.dt;
        let frame = self.frame;
        let dt = match hz {
            Some(hz) if hz.is_finite() && hz > 0.0 => {
                let step = 1.0 / hz.clamp(0.5, 1000.0);
                let bucket = self.rates.entry(id).or_insert(RateBucket {
                    pending: 0.0,
                    last_frame: frame,
                });
                bucket.last_frame = frame;
                bucket.pending += frame_dt;
                let steps = (bucket.pending / step + 4.0 * f32::EPSILON).floor();
                let advanced = (steps * step).max(0.0);
                bucket.pending = (bucket.pending - advanced).max(0.0);
                advanced
            }
            _ => {
                self.rates.insert(
                    id,
                    RateBucket {
                        pending: 0.0,
                        last_frame: frame,
                    },
                );
                frame_dt
            }
        };
        self.rate_stack.push(dt);
    }

    /// Close the innermost [`Self::rate_scope`].
    pub fn end_rate_scope(&mut self) {
        self.rate_stack.pop();
    }

    /// Animation time for the innermost open scope (the real frame time when
    /// none is open). Registry animations — popups, sections, toasts, icons —
    /// advance by this, so a scope caps them for free.
    pub(crate) fn anim_dt(&self) -> f32 {
        // Paint-only replay pass: behaviours run but time must not
        // advance, or anims would step twice on retry frames.
        if self.view.replay {
            return 0.0;
        }
        self.rate_stack.last().copied().unwrap_or(self.input.dt)
    }

    /// An authored animation duration scaled by [`Ui::motion_scale`]:
    /// `1.0` keeps the authored timing, `0.5` completes the same curve in
    /// half the time, `0` resolves instantly (the standard reduced-motion
    /// treatment — the state still changes, it just stops travelling).
    /// Hosts feed the OS preference in here; the framework ships no policy.
    pub fn motion_time(&self, duration: f32) -> f32 {
        (duration * self.motion_scale.clamp(0.0, 100.0)).max(0.0)
    }

    // --- Keyframe animations --------------------------------------------

    /// A persistent animated f32 keyed like a widget id. Retargeting every
    /// frame is the intended use: the value eases from wherever it is now
    /// toward `target` over `duration`, so callers never store tween state.
    /// A key seen for the first time eases up from 0, so widgets and
    /// panels fade in when they appear.
    ///
    /// ```
    /// use dream::bedframe::atlas::AtlasConfig;
    /// use dream::bedframe::tween::Ease;
    /// use dream::bedframe::widgets::{Ui, UiInput};
    /// use dream::bedframe::Rect;
    ///
    /// let mut ui = Ui::new(Default::default(), AtlasConfig::default());
    /// let play = Rect::new(10.0, 10.0, 100.0, 40.0);
    ///
    /// let mut input = UiInput::default();
    /// input.dt = 0.1;                   // a 100 ms step clears the 0.12 s ease
    /// input.mouse = (40.0, 30.0);        // pointing at the button
    ///
    /// let mut lift = 0.0;
    /// for _ in 0..3 {
    ///     ui.begin(input.clone(), (800.0, 600.0));
    ///     let hovered = play.contains(input.mouse);
    ///     // Retarget every frame: the value eases from wherever it is now.
    ///     lift = ui.anim_f32(
    ///         "play.lift",
    ///         if hovered { 1.0 } else { 0.0 },
    ///         0.12,
    ///         Ease::CubicOut,
    ///     );
    ///     let _ = ui.end();
    /// }
    /// // Hovered the whole time, so it settled at the top of its travel.
    /// assert_eq!(lift, 1.0);
    /// ```
    pub fn anim_f32(&mut self, key: &str, target: f32, duration: f32, ease: Ease) -> f32 {
        let id = self.id(key);
        // Read the scoped step and the motion scale before the animation
        // borrows the registry.
        let dt = self.anim_dt();
        let duration = self.motion_time(duration);
        let frame = self.frame;
        let anim = self.anims_f32.entry(id).or_insert_with(|| Anim::new(0.0));
        let fresh = anim.last_frame() == 0;
        if fresh || anim.target() != target {
            anim.animate_to(target, duration, ease);
        }
        anim.stamp(frame);
        anim.update(dt)
    }

    /// [`Self::anim_f32`] for state that already *has* a value.
    ///
    /// The only difference is the first sight of a key: instead of easing up
    /// from zero it starts at `target`, so a control that is re-created after
    /// its state was retired (a switch inside a section that was collapsed
    /// long enough for the registry sweep to drop it) reappears in the state
    /// it belongs in rather than visibly animating to it. Later changes
    /// animate exactly as [`Self::anim_f32`] does.
    pub fn anim_f32_seeded(&mut self, key: &str, target: f32, duration: f32, ease: Ease) -> f32 {
        let id = self.id(key);
        let dt = self.anim_dt();
        let duration = self.motion_time(duration);
        let frame = self.frame;
        let anim = self
            .anims_f32
            .entry(id)
            .or_insert_with(|| Anim::new(target));
        if anim.target() != target {
            anim.animate_to(target, duration, ease);
        }
        anim.stamp(frame);
        anim.update(dt)
    }

    /// [`Self::anim_f32_seeded`] with the id already hashed — headless
    /// behaviours use it to seed `{label}{suffix}` keys without a
    /// `format!` allocation.
    pub(crate) fn anim_f32_id(&mut self, id: u64, target: f32, duration: f32, ease: Ease) -> f32 {
        let dt = self.anim_dt();
        let duration = self.motion_time(duration);
        let frame = self.frame;
        let anim = self
            .anims_f32
            .entry(id)
            .or_insert_with(|| Anim::new(target));
        if anim.target() != target {
            anim.animate_to(target, duration, ease);
        }
        anim.stamp(frame);
        anim.update(dt)
    }

    /// Persistent animated colour, same contract as [`Self::anim_f32`];
    /// a key seen for the first time fades up from transparent.
    pub fn anim_color(&mut self, key: &str, target: Color, duration: f32, ease: Ease) -> Color {
        let id = self.id(key);
        let dt = self.anim_dt();
        let duration = self.motion_time(duration);
        let frame = self.frame;
        let anim = self
            .anims_color
            .entry(id)
            .or_insert_with(|| Anim::new(Color::TRANSPARENT));
        let fresh = anim.last_frame() == 0;
        if fresh || anim.target() != target {
            anim.animate_to(target, duration, ease);
        }
        anim.stamp(frame);
        anim.update(dt)
    }

    /// A looping 0..1 sawtooth (blinks, pulses, spinners) with its own
    /// per-key phase state. The phase advances by `dt / duration` modulo
    /// one — retargeting every frame is unnecessary, so a loop actually
    /// completes its cycle instead of restarting at each call. It never
    /// settles, so while it is touched it keeps [`Self::animating`] true.
    pub fn anim_loop_f32(&mut self, key: &str, duration: f32) -> f32 {
        let id = self.id(key);
        let dt = self.anim_dt();
        // Loops scale by *speed*, not duration: motion at zero freezes the
        // cycle (a stopped blink stays lit) instead of racing it.
        let speed = self.motion_scale.clamp(0.0, 100.0);
        let frame = self.frame;
        let state = self.loops.entry(id).or_insert(LoopState {
            phase: 0.0,
            last_frame: frame,
        });
        state.last_frame = frame;
        state.phase = (state.phase + dt * speed / duration.max(1e-3)) % 1.0;
        state.phase
    }

    /// A persistent spring-driven f32 keyed like a widget id — the
    /// physics-flavoured counterpart to [`Self::anim_f32`]. Retarget
    /// every frame; the value chases `target` with the given stiffness,
    /// damping and mass, carrying its velocity across retargets, and the
    /// tuning stays live (dragging a stiffness slider retunes the running
    /// spring). A first call starts at zero and springs up to the target,
    /// matching [`Self::anim_f32`]'s fade-in convention. Advances by the
    /// scoped animation time, so [`Self::rate_scope`] caps springs too.
    pub fn anim_spring_f32(
        &mut self,
        key: &str,
        target: f32,
        stiffness: f32,
        damping: f32,
        mass: f32,
    ) -> f32 {
        let id = self.id(key);
        let dt = self.anim_dt();
        let scale = self.motion_scale.clamp(0.0, 100.0);
        let frame = self.frame;
        // A key seen for the first time starts at 0 and springs up to the
        // target — matching `anim_f32`'s fade-in-from-zero convention, so
        // a widget's first visible frame animates instead of jumping.
        let entry = self.anims_spring.entry(id).or_insert(SpringAnim {
            spring: Spring::new(0.0).to(target),
            last_frame: frame,
        });
        entry.last_frame = frame;
        entry.spring.target = target;
        entry.spring.stiffness = stiffness.max(0.0);
        entry.spring.damping = damping.max(0.0);
        entry.spring.mass = mass.max(1e-3);
        if scale <= 0.0 {
            // Motion stopped: the spring lands instantly instead of
            // freezing mid-flight.
            entry.spring.velocity = 0.0;
            entry.spring.value = target;
        } else {
            // A spring carries no authored duration for the scale to
            // shrink, so it is scaled as *time*: simulated time runs at
            // `1 / scale`, settling the spring in `scale`× its natural
            // time — the same "half the scale, half the motion" rule the
            // durations follow — while the developer's stiffness, damping
            // and mass stay exactly as tuned. The step is substepped and
            // bounded so a tiny scale or a very stiff spring can never
            // integrate a step explicit Euler cannot survive.
            let mut remaining = dt / scale;
            const SUBSTEP: f32 = 1.0 / 240.0;
            let mut steps = 0;
            while remaining > 1e-6 && steps < 64 {
                let h = remaining.min(SUBSTEP);
                entry.spring.update(h);
                remaining -= h;
                steps += 1;
            }
        }
        entry.spring.value
    }

    /// True while any animation touched *this frame* is still moving — the
    /// app uses this to keep rendering on demand while motion is in flight.
    /// State left behind by a widget that stopped running does not count:
    /// an animation nobody advanced is parked, not in flight.
    pub fn animating(&self) -> bool {
        let frame = self.frame;
        self.anims_f32
            .values()
            .any(|anim| anim.last_frame() == frame && !anim.finished())
            || self
                .anims_color
                .values()
                .any(|anim| anim.last_frame() == frame && !anim.finished())
            || self
                .anims_spring
                .values()
                .any(|anim| anim.last_frame == frame && !anim.spring.settled(0.01))
            || self
                .state
                .values()
                .any(|state| state.last_frame == frame && !state.hover.finished())
            || (self.motion_scale > 0.0
                && self.loops.values().any(|state| state.last_frame == frame))
            || self.pulses.values().any(|pulse| pulse.last_frame == frame)
            || self
                .editors
                .values()
                .any(|state| state.last_frame == frame && state.vis_animating())
            // A scroll view mid-glide or mid-fling is motion: without this
            // the host would park and freeze the scroll between frames.
            || self
                .scrolls
                .values()
                .any(|state| state.last_frame == frame && state.moving())
    }

    /// Live animation count (diagnostics).
    pub fn anim_count(&self) -> usize {
        self.anims_f32.len() + self.anims_color.len() + self.anims_spring.len()
    }

    /// One-shot scale bump (napxl `pop`): 1.0 -> ~1.12 -> 1.0 over `ttl`.
    /// Self-removing; the `last_frame` stamp also retires pulses whose
    /// widget disappeared.
    pub(super) fn pulse(&mut self, id: u64) -> f32 {
        let dt = self.anim_dt();
        if let Some(pulse) = self.pulses.get_mut(&id) {
            pulse.last_frame = self.frame;
            pulse.age += dt;
            if pulse.age >= pulse.ttl {
                self.pulses.remove(&id);
                return 1.0;
            }
            let t = pulse.age / pulse.ttl;
            return 1.0 + 0.12 * (t * std::f32::consts::PI).sin();
        }
        1.0
    }
}
