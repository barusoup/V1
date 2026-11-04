//! LoD selection: when to swap a mesh for its impostor.
//!
//! Distance alone makes impostors pop, because the swap threshold sits inside
//! the range where the two representations differ most. Three things keep the
//! transition boring instead:
//!
//! - **Hysteresis**: the impostor takes over at a farther distance than the
//!   mesh returns at, so a camera hovering on the boundary cannot oscillate
//!   between them frame to frame.
//! - **A crossfade band**: inside it both representations are drawn with
//!   complementary weights ([`LodChoice::Crossfade`]), so the silhouette never
//!   snaps. A fade carries the [`LodOwner`] it is fading from, so the hand-over
//!   happens at a threshold instead of under the camera.
//! - **Screen-space sizing**: the trigger is the asset's projected size, not
//!   raw distance, so a large crate switches later than a small pebble at the
//!   same range. [`LodPolicy::screen_coverage`] is that measure.
//!
//! The policy is pure arithmetic on purpose: `dream_study` drives a camera
//! through the band and asserts no oscillation, which needs no GPU.

/// Which representation currently holds an object: the hysteresis state.
///
/// Ownership moves only when a threshold is crossed — the impostor takes over
/// at [`LodPolicy::impostor_below`], the mesh returns at [`LodPolicy::mesh_above`]
/// — and whoever holds the object keeps it in between. A fade carries the owner
/// it is fading *from*, so a [`LodChoice`] is a complete state: fed back into
/// [`LodPolicy::select`] next frame it cannot invent a transfer that did not
/// happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LodOwner {
    /// The full mesh holds the object and the impostor is fading in.
    Mesh,
    /// The impostor holds the object and is fading out.
    Impostor,
}

/// What to draw for one object this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LodChoice {
    /// Full mesh; the impostor is not drawn.
    Mesh,
    /// Both representations are drawn, weighted `t`. `t = 0` is a pure mesh,
    /// `t = 1` a pure impostor.
    ///
    /// `owner` is the representation the fade is leaving. It is part of the
    /// hysteresis state, not a convenience: recovering the owner from `t`
    /// cannot be done (see [`LodPolicy::select_coverage`]).
    Crossfade { t: f32, owner: LodOwner },
    /// Impostor only.
    Impostor,
}

impl LodChoice {
    /// Mesh weight in the 0..=1 blend, `1.0` meaning draw the mesh fully.
    pub fn mesh_weight(&self) -> f32 {
        match self {
            LodChoice::Mesh => 1.0,
            LodChoice::Impostor => 0.0,
            LodChoice::Crossfade { t, .. } => 1.0 - *t,
        }
    }

    /// Which representation owns the object this frame.
    ///
    /// Unlike [`Self::is_impostor`], which asks what is *drawn* this frame,
    /// this is the hysteresis state: during a fade it is the representation the
    /// object is leaving, whose window the fade is following.
    pub fn owner(&self) -> LodOwner {
        match self {
            LodChoice::Mesh => LodOwner::Mesh,
            LodChoice::Impostor => LodOwner::Impostor,
            LodChoice::Crossfade { owner, .. } => *owner,
        }
    }

    pub fn impostor_weight(&self) -> f32 {
        1.0 - self.mesh_weight()
    }

    /// Whether the full mesh should be submitted this frame.
    pub fn draws_mesh(&self) -> bool {
        self.mesh_weight() > 0.0
    }

    /// Whether the impostor should be submitted this frame.
    pub fn draws_impostor(&self) -> bool {
        self.impostor_weight() > 0.0
    }

    pub fn is_impostor(&self) -> bool {
        matches!(self, LodChoice::Impostor)
    }
}

/// Thresholds for one asset class. Distances are in world units and are
/// compared against [`LodPolicy::screen_coverage`] by default, so the same
/// policy works across asset scales.
#[derive(Clone, Copy, Debug)]
pub struct LodPolicy {
    /// Projected size (radius / distance) at or below which the impostor wins.
    pub impostor_below: f32,
    /// Projected size at or above which the mesh comes back. Must be greater
    /// than `impostor_below`; that gap *is* the hysteresis.
    pub mesh_above: f32,
    /// Part of the hysteresis band used to crossfade. `0.0` disables fading
    /// and snaps at the switch points, `1.0` fades across the whole band.
    pub fade: f32,
    /// Distance (world units) beyond which the impostor is always used
    /// regardless of projected size. Catches huge assets that would otherwise
    /// never switch.
    pub max_distance: f32,
}

impl Default for LodPolicy {
    fn default() -> Self {
        Self {
            impostor_below: 0.02,
            mesh_above: 0.03,
            fade: 0.6,
            max_distance: 4096.0,
        }
    }
}

impl LodPolicy {
    /// `tan(fov_y / 2)` for the engine camera's 70 degree vertical field of
    /// view, which [`Self::screen_coverage`] divides by.
    pub const HALF_FOV_TAN: f32 = 0.7002075;

    /// Screen coverage of a sphere of `radius` at `distance`: the fraction of
    /// the viewport's height it spans.
    ///
    /// Sharing one function across the engine keeps the thresholds tunable in
    /// terms of "how many pixels does it cover" rather than world units, which
    /// is what makes one policy work for a pebble and a barn.
    pub fn screen_coverage(radius: f32, distance: f32) -> f32 {
        let distance = distance.max(1e-3);
        let coverage = (radius.abs() / distance) / Self::HALF_FOV_TAN;
        if coverage.is_finite() {
            coverage.max(0.0)
        } else {
            0.0
        }
    }

    /// The distance at which a sphere of `radius` projects to `coverage`.
    /// Inverse of [`Self::screen_coverage`]; used to place a camera on a
    /// threshold in tests and by callers reasoning in pixels.
    pub fn distance_for(radius: f32, coverage: f32) -> f32 {
        radius.abs() / (coverage.max(1e-6) * Self::HALF_FOV_TAN)
    }

    /// Normalise the thresholds so `mesh_above >= impostor_below`, keeping
    /// explicit values when they are already ordered. Non-finite or
    /// non-positive inputs fall back to the defaults rather than producing a
    /// band that never triggers or always does.
    pub fn sanitised(&self) -> Self {
        let default = Self::default();
        let mut out = *self;
        if !out.impostor_below.is_finite() || out.impostor_below <= 0.0 {
            out.impostor_below = default.impostor_below;
        }
        if !out.mesh_above.is_finite() || out.mesh_above <= 0.0 {
            out.mesh_above = default.mesh_above;
        }
        if out.mesh_above <= out.impostor_below {
            out.mesh_above = out.impostor_below * 1.5 + 1e-4;
        }
        // A huge-but-finite `impostor_below` repairs `mesh_above` to infinity,
        // producing a band whose fade arithmetic turns NaN. When the repair
        // cannot yield a finite, strictly ordered band, fall back wholesale.
        if !(out.mesh_above > out.impostor_below && out.mesh_above.is_finite()) {
            out.impostor_below = default.impostor_below;
            out.mesh_above = default.mesh_above;
        }
        out.fade = if out.fade.is_finite() {
            out.fade.clamp(0.0, 1.0)
        } else {
            default.fade
        };
        out.max_distance = if out.max_distance.is_finite() {
            out.max_distance.max(1.0)
        } else {
            default.max_distance
        };
        out
    }

    /// The choice for this frame from a sphere's `radius` and `distance`.
    ///
    /// `previous` supplies the hysteresis state: whichever representation is
    /// currently in charge stays in charge until the *other* threshold is
    /// crossed. Passing [`LodChoice::Mesh`] for a brand new object biases
    /// toward the accurate representation on its first frame.
    ///
    /// Inputs that make the sphere meaningless — non-finite values, a
    /// non-positive radius, or a camera inside the sphere (`distance <=
    /// radius`) — keep the mesh: no projected size derived from them can be
    /// trusted to justify the cheaper representation.
    pub fn select(&self, radius: f32, distance: f32, previous: LodChoice) -> LodChoice {
        if !radius.is_finite() || !distance.is_finite() || radius <= 0.0 || distance <= radius {
            return LodChoice::Mesh;
        }
        let policy = self.sanitised();
        let coverage = Self::screen_coverage(radius, distance);
        // max_distance used to hard-snap to the impostor, which popped worse
        // than any band crossing. Instead feed a synthetic coverage that
        // equals `impostor_below` exactly at max_distance and shrinks past it:
        // the approach fades through the same thresholds hysteresis already
        // manages, and anything beyond the limit still ends on the impostor.
        // f64 keeps the division exact for very large distances.
        let synthetic =
            (policy.impostor_below as f64 * policy.max_distance as f64 / distance as f64) as f32;
        self.select_coverage(coverage.min(synthetic.max(0.0)), previous)
    }

    /// The choice from an already-computed [`Self::screen_coverage`], for
    /// callers that measure the projected size themselves (screen-space
    /// bounds, a coarser culling pass, a test). Non-finite coverage keeps the
    /// mesh: an unmeasurable silhouette must not leak into the impostor.
    pub fn select_coverage(&self, coverage: f32, previous: LodChoice) -> LodChoice {
        if !coverage.is_finite() {
            return LodChoice::Mesh;
        }
        let policy = self.sanitised();
        let coverage = coverage.max(0.0);

        let band = policy.mesh_above - policy.impostor_below;
        let fade_width = band * policy.fade;

        // Ownership with hysteresis: whichever representation holds the object
        // keeps it until the far threshold for the other one is crossed, which
        // is what stops a camera parked on a boundary from flipping every
        // frame. `previous.owner()` is that state, and because a fade carries
        // its owner, a fade in progress keeps the one it started from.
        //
        // The owner cannot be inferred instead. Off the weight (`t >= 0.5`) it
        // transfers at the fade's *midpoint*, and the next frame runs the other
        // representation's window, whose weight at that coverage is already
        // 1.0: the object skips the upper half of its fade in one frame. Off
        // the coverage it is ambiguous, because the two windows sit at opposite
        // ends of the band and overlap whenever `fade` exceeds half of it. That
        // is the whole reason [`LodChoice::Crossfade`] holds an owner.
        let owned_impostor = match previous.owner() {
            LodOwner::Impostor => coverage <= policy.mesh_above,
            LodOwner::Mesh => coverage < policy.impostor_below,
        };
        // Each owner follows its own window, anchored at the threshold it hands
        // over at, and each window saturates exactly there — so the two
        // formulas agree where ownership moves and the weight is continuous
        // across the hand-over whatever the frame step. (Which is also why the
        // window may be narrower than the band: what is skipped is curvature,
        // never a jump.)
        let t = if fade_width <= 0.0 {
            // Fading disabled: snap at the thresholds.
            if owned_impostor { 1.0 } else { 0.0 }
        } else if owned_impostor {
            // The impostor holds the object and fades back out as coverage
            // rises toward `mesh_above`, where the mesh takes over.
            ((policy.mesh_above - coverage) / fade_width).clamp(0.0, 1.0)
        } else {
            // The mesh holds the object and fades the impostor in as coverage
            // falls toward `impostor_below`, where the impostor takes over.
            ((policy.impostor_below + fade_width - coverage) / fade_width).clamp(0.0, 1.0)
        };

        // A fade this thin is not worth the second draw, and rounding at the
        // exact thresholds must not manufacture one: `mesh_above - coverage`
        // and `fade_width` are equal in real arithmetic but not in f32, so
        // without the epsilon a camera sitting exactly on a threshold would
        // pay for two representations forever.
        const FADE_EPSILON: f32 = 1e-4;
        if t <= FADE_EPSILON {
            LodChoice::Mesh
        } else if t >= 1.0 - FADE_EPSILON {
            LodChoice::Impostor
        } else {
            LodChoice::Crossfade {
                t,
                owner: if owned_impostor {
                    LodOwner::Impostor
                } else {
                    LodOwner::Mesh
                },
            }
        }
    }
}

/// Counters for a run of [`LodPolicy::select`] calls, so a game (or the study)
/// can prove the LoD system is not thrashing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LodStats {
    /// Times the choice changed representation (mesh <-> impostor), counting a
    /// fade's endpoints once each.
    pub switches: u32,
    /// Frames spent drawing both representations.
    pub fade_frames: u32,
    /// Frames spent on the impostor only.
    pub impostor_frames: u32,
    /// Frames spent on the mesh only, or fading into it.
    pub mesh_frames: u32,
}

impl LodStats {
    /// Feed one frame's decision. `previous` is the pre-selection choice.
    pub fn record(&mut self, previous: LodChoice, current: LodChoice) {
        let before = previous.is_impostor();
        let after = current.is_impostor();
        if before != after {
            self.switches += 1;
        }
        if matches!(current, LodChoice::Crossfade { .. }) {
            self.fade_frames += 1;
        } else if after {
            self.impostor_frames += 1;
        } else {
            self.mesh_frames += 1;
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "switches={} fade={} impostor={} mesh={}",
            self.switches, self.fade_frames, self.impostor_frames, self.mesh_frames
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> LodPolicy {
        LodPolicy {
            impostor_below: 0.02,
            mesh_above: 0.03,
            fade: 0.5,
            max_distance: 1000.0,
        }
    }

    #[test]
    fn nearby_objects_keep_their_mesh() {
        let p = policy();
        let choice = p.select(1.0, 10.0, LodChoice::Mesh);
        assert!(matches!(choice, LodChoice::Mesh), "got {choice:?}");
        assert!(choice.draws_mesh() && !choice.draws_impostor());
    }

    #[test]
    fn distant_objects_use_the_impostor() {
        let p = policy();
        let choice = p.select(1.0, 500.0, LodChoice::Mesh);
        assert!(choice.is_impostor(), "got {choice:?}");
        assert!(!choice.draws_mesh() && choice.draws_impostor());
    }

    #[test]
    fn the_switch_needs_the_other_threshold_to_be_crossed() {
        let p = policy();
        // The impostor takes over only below `impostor_below`...
        assert!(
            !p.select_coverage(0.021, LodChoice::Mesh).is_impostor(),
            "above the acquire threshold keeps the mesh"
        );
        assert!(
            p.select_coverage(0.019, LodChoice::Mesh).is_impostor(),
            "below the acquire threshold switches"
        );
        // ...and the mesh only returns above `mesh_above`, so the band in
        // between belongs to whoever already has it.
        assert!(
            p.select_coverage(0.025, LodChoice::Impostor).is_impostor(),
            "the band still belongs to the impostor"
        );
        assert!(
            !p.select_coverage(0.025, LodChoice::Mesh).is_impostor(),
            "and to the mesh when the mesh has it"
        );
        assert!(
            p.select_coverage(0.035, LodChoice::Impostor).draws_mesh(),
            "above mesh_above the mesh returns"
        );
    }

    #[test]
    fn a_camera_dithering_on_the_acquire_threshold_switches_once() {
        let p = policy();
        let radius = 1.0;
        // Exactly on the acquire threshold, jittered by a tenth of a percent:
        // the worst case for a naive comparison.
        let distance = LodPolicy::distance_for(radius, p.impostor_below);
        let mut choice = LodChoice::Mesh;
        let mut stats = LodStats::default();
        for frame in 0..600 {
            let jitter = if frame % 2 == 0 { 0.999 } else { 1.001 };
            let previous = choice;
            choice = p.select(radius, distance * jitter, previous);
            stats.record(previous, choice);
        }
        assert_eq!(
            stats.switches,
            1,
            "an object parked on the threshold must settle, not thrash: {}",
            stats.summary()
        );
        assert_eq!(stats.impostor_frames, 599, "{}", stats.summary());
    }

    #[test]
    fn a_camera_dithering_inside_the_band_never_switches_at_all() {
        let p = policy();
        let radius = 1.0;
        // Mid-band: a naive single-threshold comparison would flip here on
        // every frame of the jitter.
        let distance = LodPolicy::distance_for(radius, 0.025);
        let mut choice = LodChoice::Mesh;
        let mut stats = LodStats::default();
        let mut furthest = 0.0f32;
        for frame in 0..600 {
            let jitter = if frame % 2 == 0 { 0.99 } else { 1.01 };
            let previous = choice;
            choice = p.select(radius, distance * jitter, previous);
            stats.record(previous, choice);
            furthest = furthest.max(choice.impostor_weight());
        }
        assert_eq!(stats.switches, 0, "hysteresis holds: {}", stats.summary());
        assert!(
            furthest < 0.5,
            "the object must never hand over to the impostor mid-band (peaked at {furthest})"
        );
        assert!(!choice.is_impostor(), "got {choice:?}");
    }

    #[test]
    fn a_receding_camera_switches_once_and_fades_through_the_band() {
        let p = policy();
        let mut choice = LodChoice::Mesh;
        let mut stats = LodStats::default();
        let mut saw_fade = false;
        let mut fade_values = Vec::new();
        for step in 0..800 {
            let distance = 1.0 + step as f32 * 0.25;
            let previous = choice;
            choice = p.select(1.0, distance, previous);
            stats.record(previous, choice);
            if let LodChoice::Crossfade { t, .. } = choice {
                if t > 0.0 && t < 1.0 {
                    saw_fade = true;
                    fade_values.push(t);
                }
            }
        }
        assert_eq!(stats.switches, 1, "one transition: {}", stats.summary());
        assert!(saw_fade, "the band must actually fade: {}", stats.summary());
        // The fade must run in one direction, not bounce.
        assert!(
            fade_values.windows(2).all(|pair| pair[0] <= pair[1]),
            "fade values must be monotonic while receding: {fade_values:?}"
        );
        assert!(choice.is_impostor(), "got {choice:?}");
    }

    #[test]
    fn the_crossfade_weight_never_jumps_between_frames() {
        // The pop this guards against lived in the policy, not the shader:
        // ownership used to transfer at the fade's midpoint, so the next frame
        // ran the other direction's window, whose weight at that coverage is
        // already 1.0. A fine sweep out past the impostor and back must move the
        // weight in small steps; anything larger is a frame where the object
        // visibly changes representation.
        for fade in [0.25, 0.6, 1.0] {
            let mut p = policy();
            p.fade = fade;
            let band = p.mesh_above - p.impostor_below;
            let low = p.impostor_below - band;
            let high = p.mesh_above + band;
            let steps = 4000;
            let mut choice = LodChoice::Mesh;
            let mut previous_weight = choice.impostor_weight();
            let mut peak = 0.0f32;
            for index in 0..=steps * 2 {
                // Triangle wave: out to well inside the impostor, then back.
                let t = index as f32 / steps as f32;
                let balance = 1.0 - (t - 1.0).abs();
                let coverage = high - (high - low) * balance;
                choice = p.select_coverage(coverage, choice);
                let weight = choice.impostor_weight();
                peak = peak.max((weight - previous_weight).abs());
                previous_weight = weight;
            }
            assert!(
                peak <= 0.02,
                "the crossfade weight jumped by {peak:.3} in one coverage step at fade={fade}: that is a rendered pop"
            );
        }
    }

    #[test]
    fn a_fade_is_a_complete_state_so_a_still_camera_does_not_drift() {
        // The weight is a function of the coverage and the owner and nothing
        // else, so an object that stops moving stops changing: feeding a choice
        // back at the same coverage is idempotent. Recovering the owner from
        // the weight broke exactly this — a parked fade kept walking toward the
        // impostor one frame at a time.
        let p = policy();
        for coverage in [0.019, 0.0201, 0.0225, 0.025, 0.0275, 0.0299, 0.031] {
            for start in [LodChoice::Mesh, LodChoice::Impostor] {
                let once = p.select_coverage(coverage, start);
                let twice = p.select_coverage(coverage, once);
                assert_eq!(
                    once, twice,
                    "coverage {coverage} is not stable from {start:?}"
                );
                assert_eq!(
                    once.owner(),
                    twice.owner(),
                    "the owner drifted at coverage {coverage}"
                );
            }
        }
    }

    #[test]
    fn a_crossfade_never_pops_even_when_the_camera_jumps() {
        // A fly-through moves coverage smoothly, but a game teleports objects
        // (respawns, cuts, boosts) and the policy must not manufacture a
        // discontinuity out of that: the weight can only travel as far as the
        // window it is inside allows. `|dt| <= |dcoverage| / fade_width` is that
        // bound, and it has to hold across the hand-over too — that is where a
        // fade loses sight of which representation it is blending with.
        for fade in [0.25, 0.6, 1.0] {
            let mut p = policy();
            p.fade = fade;
            let fade_width = (p.mesh_above - p.impostor_below) * fade;
            let mut choice = LodChoice::Mesh;
            let mut previous_coverage = 0.05f32;
            let mut previous_weight = choice.impostor_weight();
            // A deterministic xorshift walk, so a failure is reproducible.
            let mut seed = 0x2f6e2b1u32;
            for step in 0..4000 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                // Deliberately includes jumps clean across the whole band.
                let coverage = (seed % 10_000) as f32 / 10_000.0 * 0.04;
                choice = p.select_coverage(coverage, choice);
                let weight = choice.impostor_weight();
                let moved = (weight - previous_weight).abs();
                let step_size = (coverage - previous_coverage).abs();
                let bound = step_size / fade_width + 1e-3;
                assert!(
                    moved <= bound,
                    "fade={fade} step {step}: the weight moved {moved:.3} while coverage moved {step_size:.4} (bound {bound:.3})"
                );
                previous_coverage = coverage;
                previous_weight = weight;
            }
        }
    }

    #[test]
    fn approaching_and_receding_take_different_paths() {
        let p = policy();
        // Receding through the acquire window is a fade...
        let receding = p.select_coverage(0.0225, LodChoice::Mesh);
        assert!(
            matches!(receding, LodChoice::Crossfade { t, .. } if t > 0.0 && t < 1.0),
            "got {receding:?}"
        );
        // ...while the same coverage with the impostor already in charge is
        // simply the impostor, because ownership only releases at mesh_above.
        let held = p.select_coverage(0.0225, LodChoice::Impostor);
        assert!(held.is_impostor(), "got {held:?}");
        // And approaching, the impostor fades out near mesh_above.
        let approaching = p.select_coverage(0.0275, LodChoice::Impostor);
        assert!(
            matches!(approaching, LodChoice::Crossfade { t, .. } if t > 0.0 && t < 1.0),
            "got {approaching:?}"
        );
        let released = p.select_coverage(0.0275, LodChoice::Mesh);
        assert!(matches!(released, LodChoice::Mesh), "got {released:?}");
    }

    #[test]
    fn a_large_asset_switches_later_than_a_small_one() {
        let p = policy();
        let distance = 60.0;
        let small = p.select(0.5, distance, LodChoice::Mesh);
        let large = p.select(8.0, distance, LodChoice::Mesh);
        assert!(small.is_impostor(), "a pebble is already an impostor");
        assert!(
            !large.is_impostor(),
            "a barn at the same range is still a mesh"
        );
    }

    #[test]
    fn max_distance_forces_the_impostor_regardless_of_size() {
        let p = LodPolicy {
            max_distance: 100.0,
            ..policy()
        };
        let choice = p.select(200.0, 500.0, LodChoice::Mesh);
        assert!(
            choice.is_impostor(),
            "beyond max_distance the impostor wins"
        );
    }

    #[test]
    fn fade_can_be_disabled() {
        let p = LodPolicy {
            fade: 0.0,
            ..policy()
        };
        let mut choice = LodChoice::Mesh;
        let mut switches = 0;
        for step in 0..800 {
            let distance = 1.0 + step as f32 * 0.25;
            let previous = choice;
            choice = p.select(1.0, distance, previous);
            assert!(
                !matches!(choice, LodChoice::Crossfade { .. }),
                "fade=0 must snap: {choice:?}"
            );
            if previous.is_impostor() != choice.is_impostor() {
                switches += 1;
            }
        }
        assert_eq!(switches, 1, "a snap still crosses exactly once");
        assert!(choice.is_impostor());
    }

    #[test]
    fn inverted_thresholds_are_repaired_not_trusted() {
        let broken = LodPolicy {
            impostor_below: 0.05,
            mesh_above: 0.01,
            ..policy()
        };
        let fixed = broken.sanitised();
        assert!(fixed.mesh_above > fixed.impostor_below);
        // Selection still behaves, with the repaired band.
        let far = broken.select(1.0, 1000.0, LodChoice::Mesh);
        assert!(far.is_impostor());
    }

    #[test]
    fn screen_coverage_matches_projection() {
        // A sphere whose radius equals the distance is wider than the
        // half-viewport: coverage = (radius / distance) / tan(fov/2).
        let coverage = LodPolicy::screen_coverage(1.0, 1.0);
        assert!(
            (coverage - 1.0 / LodPolicy::HALF_FOV_TAN).abs() < 1e-4,
            "{coverage}"
        );
        // Doubling the distance halves it.
        let far = LodPolicy::screen_coverage(1.0, 2.0);
        assert!((far - coverage / 2.0).abs() < 1e-4, "{far}");
        // A zero distance cannot divide by zero.
        assert!(LodPolicy::screen_coverage(1.0, 0.0).is_finite());
        assert!(LodPolicy::screen_coverage(f32::NAN, 1.0).is_finite());
        // distance_for inverts it.
        let d = LodPolicy::distance_for(2.0, 0.05);
        assert!((LodPolicy::screen_coverage(2.0, d) - 0.05).abs() < 1e-4);
    }

    #[test]
    fn weights_are_complementary() {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let choice = LodChoice::Crossfade {
                t,
                owner: LodOwner::Mesh,
            };
            assert!((choice.mesh_weight() + choice.impostor_weight() - 1.0).abs() < 1e-6);
            assert!((choice.impostor_weight() - t).abs() < 1e-6);
            assert_eq!(choice.owner(), LodOwner::Mesh);
        }
        assert_eq!(LodChoice::Mesh.impostor_weight(), 0.0);
        assert_eq!(LodChoice::Impostor.mesh_weight(), 0.0);
        assert_eq!(LodChoice::Mesh.owner(), LodOwner::Mesh);
        assert_eq!(LodChoice::Impostor.owner(), LodOwner::Impostor);
    }

    #[test]
    fn invalid_selection_inputs_keep_the_mesh() {
        let p = policy();
        for (radius, distance) in [
            (f32::NAN, 100.0),
            (f32::INFINITY, 100.0),
            (1.0, f32::NAN),
            (1.0, f32::INFINITY),
            (1.0, f32::NEG_INFINITY),
            (0.0, 100.0),
            (-2.0, 100.0),
            (50.0, 50.0), // camera exactly on the sphere
            (50.0, 10.0), // camera inside the sphere
        ] {
            for previous in [LodChoice::Mesh, LodChoice::Impostor] {
                assert_eq!(
                    p.select(radius, distance, previous),
                    LodChoice::Mesh,
                    "radius={radius} distance={distance} must keep the mesh"
                );
            }
        }
        // A tiny positive radius is a real sphere and may still select the
        // impostor: only degenerate inputs are rejected.
        assert!(
            p.select(1e-6, 500.0, LodChoice::Mesh).is_impostor(),
            "a speck far away is still an impostor"
        );
        // select_coverage cannot be handed garbage either.
        for coverage in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                p.select_coverage(coverage, LodChoice::Impostor),
                LodChoice::Mesh
            );
        }
    }

    #[test]
    fn sanitised_repairs_every_degenerate_policy_field() {
        let default = LodPolicy::default();
        for bad in [
            LodPolicy {
                impostor_below: f32::NAN,
                ..policy()
            },
            LodPolicy {
                impostor_below: -1.0,
                ..policy()
            },
            LodPolicy {
                impostor_below: 0.0,
                ..policy()
            },
            LodPolicy {
                impostor_below: f32::INFINITY,
                ..policy()
            },
        ] {
            let fixed = bad.sanitised();
            assert_eq!(fixed.impostor_below, default.impostor_below);
        }
        for bad in [f32::NAN, f32::INFINITY, -0.5, 0.0] {
            let fixed = LodPolicy {
                mesh_above: bad,
                ..policy()
            }
            .sanitised();
            assert!(fixed.mesh_above > fixed.impostor_below);
        }
        let nan_fade = LodPolicy {
            fade: f32::NAN,
            ..policy()
        }
        .sanitised();
        assert_eq!(nan_fade.fade, default.fade);
        let over_fade = LodPolicy {
            fade: 3.0,
            ..policy()
        }
        .sanitised();
        assert_eq!(over_fade.fade, 1.0);
        for bad in [f32::NAN, f32::INFINITY] {
            let fixed = LodPolicy {
                max_distance: bad,
                ..policy()
            }
            .sanitised();
            assert_eq!(fixed.max_distance, default.max_distance);
        }
        assert_eq!(
            LodPolicy {
                max_distance: -10.0,
                ..policy()
            }
            .sanitised()
            .max_distance,
            1.0
        );
        // f32::MAX is finite and positive but the band repair overflows:
        // `impostor_below * 1.5` is infinite, so both thresholds reset.
        let huge = LodPolicy {
            impostor_below: f32::MAX,
            mesh_above: 0.03,
            ..policy()
        }
        .sanitised();
        assert_eq!(huge.impostor_below, default.impostor_below);
        assert_eq!(huge.mesh_above, default.mesh_above);
        // And it must still select sanely.
        let p = LodPolicy {
            impostor_below: f32::MAX,
            mesh_above: 0.03,
            ..policy()
        };
        assert!(p.select(1.0, 1000.0, LodChoice::Mesh).is_impostor());
    }

    #[test]
    fn ownership_boundaries_are_exact() {
        let p = policy();
        // At exactly impostor_below the impostor takes over from a mesh
        // owner; at exactly mesh_above the mesh takes it back.
        assert!(
            p.select_coverage(p.impostor_below, LodChoice::Mesh)
                .is_impostor()
        );
        assert!(
            !p.select_coverage(p.impostor_below + 1e-3, LodChoice::Mesh)
                .is_impostor(),
            "just above the threshold the mesh keeps ownership"
        );
        assert!(
            p.select_coverage(p.mesh_above, LodChoice::Impostor)
                .draws_mesh(),
            "at mesh_above the mesh owns or is fading in"
        );
        assert_eq!(
            p.select_coverage(p.mesh_above - 1e-3, LodChoice::Impostor)
                .owner(),
            LodOwner::Impostor,
            "just below mesh_above the impostor keeps ownership (a fade keeps its owner)"
        );
    }

    #[test]
    fn max_distance_fades_instead_of_snapping() {
        // Beyond the limit the impostor still wins — but the approach crosses
        // the same fade band the coverage thresholds manage, so a flight out
        // and back across max_distance never jumps the weight.
        let p = LodPolicy {
            impostor_below: 0.02,
            mesh_above: 0.03,
            fade: 1.0,
            max_distance: 200.0,
        };
        // The asset is large enough that its real coverage stays above the
        // band at the limit, so the synthetic max_distance coverage — not
        // projected size — is what crosses the thresholds.
        let radius = 10.0;
        let mut choice = LodChoice::Mesh;
        let mut previous_weight = choice.impostor_weight();
        let mut peak = 0.0f32;
        // Out to twice max_distance and back, 4000 steps over a 300-unit
        // range: ~1/1000 of the span per step.
        for index in 0..=4000usize {
            let t = index as f32 / 2000.0;
            let balance = 1.0 - (t - 1.0).abs();
            let distance = p.max_distance * 0.5 + p.max_distance * 1.5 * balance;
            choice = p.select(radius, distance, choice);
            let weight = choice.impostor_weight();
            peak = peak.max((weight - previous_weight).abs());
            previous_weight = weight;
        }
        assert!(
            peak <= 0.01,
            "the weight jumped {peak:.3} crossing max_distance — the old hard snap"
        );
        assert_eq!(choice, LodChoice::Mesh, "the sweep ends near the camera");
    }

    #[test]
    fn far_past_max_distance_is_still_the_impostor() {
        let p = LodPolicy {
            max_distance: 200.0,
            ..policy()
        };
        let mut choice = LodChoice::Mesh;
        // Walk out gradually: ownership transfers before the limit.
        for distance in [150.0, 190.0, 199.0, 201.0, 300.0, 2000.0] {
            choice = p.select(1.0, distance, choice);
        }
        assert!(choice.is_impostor());
        // And a fresh object at 10x the limit is an impostor outright.
        assert!(p.select(1.0, 2000.0, LodChoice::Mesh).is_impostor());
    }
}
