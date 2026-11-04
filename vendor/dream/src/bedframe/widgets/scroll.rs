//! Scroll physics: one model behind every scroll surface.
//!
//! BedFrame opens scroll viewports two ways — [`Ui::scroll_begin`] for an
//! explicit viewport and a layout container with `overflow: scroll` for the
//! view layer — and both run the wheel, keyboard, scrollbar, momentum and
//! programmatic routing implemented here, so a scroll feels identical
//! wherever it is built.
//!
//! The model mirrors the parts of the DOM scrolling surface applications
//! actually use, and defaults to how a modern web page feels:
//!
//! - Wheel notches glide toward a destination instead of jumping, and
//!   repeated notches accumulate into one gesture. Precision devices
//!   (trackpads) apply their pixels 1:1 — they already produce smooth,
//!   inertial deltas — and only coast when such a gesture stops mid-flight.
//! - [`Ui::scroll_to`], [`Ui::scroll_by`] and [`Ui::scroll_into_view`]
//!   take a [`ScrollBehavior`] (`Instant`, `Smooth`, `Auto`) and
//!   [`ScrollAlign`] targeting, like `element.scrollTo({top, behavior})`
//!   and `element.scrollIntoView({block, behavior})`.
//! - Nested views chain: what the inner view cannot use (it is at its end)
//!   continues the enclosing one, the way `overscroll-behavior: auto`
//!   passes a wheel on to the page.
//! - A view under the pointer scrolls from the keyboard (arrows, page
//!   keys, Home/End) while no widget owns focus.
//!
//! Every knob lives on [`ScrollConfig`] (`Ui::scroll_config`) and
//! [`ScrollConfig::no_momentum`] restores the old jump-and-stop feel:
//!
//! ```ignore
//! // A snappier glide, no keyboard scrolling, and a longer wheel notch.
//! ui.scroll_config = ScrollConfig {
//!     smoothness: 0.04,
//!     keyboard: false,
//!     line_height: 56.0,
//!     ..ScrollConfig::default()
//! };
//! ```

use super::{LAYER_POPUP, ScrollbarStyle, Ui};
use crate::ui::Rect;
use crate::ui::tween::{Ease, ease};

/// Highest coasting speed a precision gesture may carry into momentum, in
/// pixels per second. Keeps a stalled frame from flinging the view away.
const MAX_FLING: f32 = 12_000.0;
/// Below this many pixels a glide is considered arrived.
const SETTLE_EPSILON: f32 = 0.25;

/// How a programmatic scroll travels to its destination — the DOM's
/// `ScrollBehavior`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScrollBehavior {
    /// Jump: the offset lands on the destination this frame.
    Instant,
    /// Glide: animate over [`ScrollConfig::smooth_duration`] with
    /// [`ScrollConfig::smooth_ease`].
    Smooth,
    /// Whatever the configuration says ([`ScrollConfig::default_behavior`],
    /// smooth unless [`ScrollConfig::smooth`] is off).
    #[default]
    Auto,
}

impl ScrollBehavior {
    /// The behaviour that actually applies: `Auto` reads the config, and
    /// a config with animation off answers `Instant` to everything.
    fn resolve(self, config: &ScrollConfig) -> ScrollBehavior {
        match self {
            ScrollBehavior::Auto => {
                if config.smooth {
                    match config.default_behavior {
                        ScrollBehavior::Auto => ScrollBehavior::Smooth,
                        other => other,
                    }
                } else {
                    ScrollBehavior::Instant
                }
            }
            other => {
                if config.smooth {
                    other
                } else {
                    ScrollBehavior::Instant
                }
            }
        }
    }
}

/// Where a target sits in the viewport — the DOM's `ScrollLogicalPosition`
/// on the block axis (BedFrame scroll views are vertical).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScrollAlign {
    /// Put the target's top at the viewport's top.
    Start,
    /// Centre the target in the viewport.
    Center,
    /// Put the target's bottom at the viewport's bottom.
    End,
    /// Scroll the shortest distance that reveals the target; a target that
    /// is already fully visible does not move the view.
    #[default]
    Nearest,
}

impl ScrollAlign {
    /// Resolve this alignment into a scroll offset, from the target's
    /// position and height inside the content (content coordinates) plus
    /// the current offset.
    fn offset_for(self, top: f32, height: f32, viewport_h: f32, current: f32) -> f32 {
        match self {
            ScrollAlign::Start => top,
            ScrollAlign::Center => top - (viewport_h - height) * 0.5,
            ScrollAlign::End => top - viewport_h + height,
            ScrollAlign::Nearest => {
                if top < current {
                    top
                } else if top + height > current + viewport_h {
                    top + height - viewport_h
                } else {
                    current
                }
            }
        }
    }
}

/// Options for [`Ui::scroll_to`] — `element.scrollTo({top, behavior})`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScrollToOptions {
    /// Destination offset in pixels from the content's top.
    pub top: f32,
    pub behavior: ScrollBehavior,
}

impl ScrollToOptions {
    /// A destination with the configured default behaviour.
    pub fn new(top: f32) -> Self {
        Self {
            top,
            ..Self::default()
        }
    }

    /// A destination that glides there.
    pub fn smooth(top: f32) -> Self {
        Self {
            top,
            behavior: ScrollBehavior::Smooth,
        }
    }

    /// A destination that jumps there.
    pub fn instant(top: f32) -> Self {
        Self {
            top,
            behavior: ScrollBehavior::Instant,
        }
    }
}

/// Options for [`Ui::scroll_into_view`] —
/// `element.scrollIntoView({block, behavior})`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScrollIntoViewOptions {
    pub block: ScrollAlign,
    pub behavior: ScrollBehavior,
}

/// One scroll view's state after its last build.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScrollMetrics {
    /// Rendered offset in pixels from the content's top.
    pub offset: f32,
    /// Viewport height in pixels.
    pub viewport: f32,
    /// Content height in pixels (as the last build reported it).
    pub content: f32,
    /// Largest offset the current content allows.
    pub max_offset: f32,
    /// Coasting speed in pixels per second; `0` when not coasting.
    pub velocity: f32,
    /// The view is still travelling toward its destination.
    pub animating: bool,
}

impl ScrollMetrics {
    /// Fraction of the content scrolled, 0..1 — a thumb's position, or a
    /// "reading progress" readout.
    pub fn progress(&self) -> f32 {
        if self.max_offset <= 0.0 {
            0.0
        } else {
            (self.offset / self.max_offset).clamp(0.0, 1.0)
        }
    }
}

/// Scroll-view physics and programmatic scrolling. Public fields, no hidden
/// behaviour: set the whole struct on [`Ui::scroll_config`] to taste.
///
/// The default is a modern web feel — wheel notches glide for about an
/// eighth of a second, trackpads track their own pixels, and nested views
/// chain at the ends. [`ScrollConfig::no_momentum`] turns all of it off.
#[derive(Clone, Copy, Debug)]
pub struct ScrollConfig {
    /// Pixels one line-based wheel notch moves per wheel line. Hosts turn
    /// an OS line delta into pixels with this (`winit` reports a notch as
    /// three lines on Windows), so it sets the notch size.
    pub line_height: f32,
    /// Multiplier on top of the incoming wheel pixels (a 1:1 wheel is
    /// `1.0`; precision deltas are scaled the same way).
    pub wheel_scale: f32,
    /// Fraction of a coasting velocity kept after one full second of
    /// momentum. `0` stops within a frame; values near `1` glide for a
    /// long time.
    pub momentum_decay: f32,
    /// Coasting velocities below this many pixels per second stop
    /// immediately.
    pub stop_threshold: f32,
    /// Wheel notches glide to their destination instead of jumping. This
    /// is the master switch: `false` makes every scroll instant, including
    /// [`ScrollBehavior::Smooth`] requests.
    pub smooth: bool,
    /// Glide time constant in seconds: the fraction of the remaining
    /// distance covered per second is `1 - exp(-1 / smoothness)`. Smaller
    /// is snappier (`0.02` barely glides), larger is floatier.
    pub smoothness: f32,
    /// Duration of a programmatic [`ScrollBehavior::Smooth`] scroll over
    /// roughly one viewport; longer jumps take a little longer (up to
    /// 2.5×), short ones a little less.
    pub smooth_duration: f32,
    /// Curve a programmatic smooth scroll follows.
    pub smooth_ease: Ease,
    /// What [`ScrollBehavior::Auto`] means (smooth unless configured
    /// otherwise).
    pub default_behavior: ScrollBehavior,
    /// A view under the pointer scrolls from the keyboard (arrows, page
    /// keys, Home/End) while no widget owns focus and no popup is open.
    pub keyboard: bool,
    /// How much of the viewport a PageUp/PageDown (and a scrollbar track
    /// click) travels, as a fraction.
    pub page_fraction: f32,
    /// How far one arrow key travels, in pixels.
    pub arrow_step: f32,
    /// Pass the leftover of a wheel tick up to the enclosing scroll view
    /// when this one is at its end (DOM scroll chaining).
    pub chain: bool,
}

impl Default for ScrollConfig {
    fn default() -> Self {
        Self {
            line_height: 40.0,
            wheel_scale: 1.0,
            momentum_decay: 0.04,
            stop_threshold: 24.0,
            smooth: true,
            smoothness: 0.06,
            smooth_duration: 0.28,
            smooth_ease: Ease::CubicOut,
            default_behavior: ScrollBehavior::Smooth,
            keyboard: true,
            page_fraction: 0.85,
            arrow_step: 48.0,
            chain: true,
        }
    }
}

impl ScrollConfig {
    /// No coasting and no gliding: a wheel tick jumps and stops, like a
    /// slider — the pre-smoothing behaviour.
    pub fn no_momentum() -> Self {
        Self {
            momentum_decay: 0.0,
            stop_threshold: f32::INFINITY,
            smooth: false,
            default_behavior: ScrollBehavior::Instant,
            ..Self::default()
        }
    }
}

/// How a scroll view's offset is moving this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum ScrollMotion {
    /// Parked (or following the pointer exactly).
    #[default]
    Idle,
    /// Exponential approach to [`ScrollState::target`] — wheel notches and
    /// keyboard steps.
    Smooth,
    /// Inertia: [`ScrollState::velocity`] decays by
    /// [`ScrollConfig::momentum_decay`] per second.
    Fling,
    /// A time-boxed eased scroll (programmatic `behavior: smooth`).
    Tween {
        from: f32,
        to: f32,
        elapsed: f32,
        duration: f32,
        curve: Ease,
    },
}

/// A programmatic scroll request waiting for the view's geometry. Applied
/// on the next [`Ui::scroll_viewport`]: only then are the content extent and
/// viewport height known, so an alignment and a clamp are exact.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScrollGoal {
    /// Target position in content coordinates (its top edge).
    pub top: f32,
    /// Target height in content coordinates (used by [`ScrollAlign`]).
    pub height: f32,
    pub block: ScrollAlign,
    pub behavior: ScrollBehavior,
}

/// Scroll-view scroll offset (pixels, >= 0), where it is heading, and the
/// motion carrying it there.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScrollState {
    pub offset: f32,
    /// Destination of a [`ScrollMotion::Smooth`] glide; equal to `offset`
    /// whenever nothing is gliding.
    pub target: f32,
    /// Coasting velocity in pixels per second.
    pub velocity: f32,
    pub motion: ScrollMotion,
    /// A pending programmatic request, resolved on the next advance.
    pub goal: Option<ScrollGoal>,
    /// Viewport height as of the last advance.
    pub viewport: f32,
    /// Last frame a wheel tick landed here (momentum hand-off).
    pub wheel_frame: u64,
    /// Retirement stamp.
    pub last_frame: u64,
}

impl ScrollState {
    fn new() -> Self {
        Self {
            offset: 0.0,
            target: 0.0,
            velocity: 0.0,
            motion: ScrollMotion::Idle,
            goal: None,
            viewport: 0.0,
            wheel_frame: 0,
            last_frame: 0,
        }
    }

    /// True while the view is still travelling — the host must keep
    /// building frames until this is false.
    pub(crate) fn moving(&self) -> bool {
        !matches!(self.motion, ScrollMotion::Idle) || (self.target - self.offset).abs() > 0.01
    }
}

/// A scroll view registered this frame (wheel ownership and chaining).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScrollRegion {
    pub id: u64,
    /// The view's effective rect: its own rect intersected with the clip it
    /// was drawn under — independent of where the pointer happened to be.
    pub rect: Rect,
    /// The draw layer it was registered on (surface gating uses this).
    pub layer: i32,
    /// Whether the view could take input at all (not confined away, not
    /// clipped to nothing) — again independent of the pointer position.
    pub enabled: bool,
    /// The enclosing scroll view, if this one is nested — where a wheel
    /// tick this view cannot use is passed on.
    pub parent: Option<u64>,
}

/// The track and thumb of a scrollbar laid out inside `viewport`, or `None`
/// when the content fits (no scrollbar to draw or grab).
pub(crate) fn scrollbar_rects(
    bar: &ScrollbarStyle,
    viewport: Rect,
    offset: f32,
    content_h: f32,
) -> Option<(Rect, Rect, f32)> {
    let max_offset = (content_h - viewport.h).max(0.0);
    if max_offset <= 0.0 || viewport.h <= 2.0 {
        return None;
    }
    let track = Rect::new(
        viewport.max().0 - bar.inset - bar.width,
        viewport.y + bar.inset,
        bar.width,
        (viewport.h - bar.inset * 2.0).max(0.0),
    );
    let thumb_h = (viewport.h / content_h.max(1.0) * track.h)
        .max(bar.min_thumb)
        .min(track.h);
    let travel = (track.h - thumb_h).max(0.0);
    let thumb = Rect::new(
        track.x,
        track.y + (offset / max_offset).clamp(0.0, 1.0) * travel,
        track.w,
        thumb_h,
    );
    Some((track, thumb, max_offset))
}

impl Ui {
    /// Register `rect` as a wheel-scrollable viewport: the record is the
    /// rect intersected with the enclosing clip (a fully clipped view
    /// shrinks to nothing rather than silently widening back), flagged
    /// enabled only when interactive at the current layer. Its enclosing
    /// scroll view — if any — is recorded too, for scroll chaining.
    pub(crate) fn register_scroll_view(&mut self, id: u64, rect: Rect) {
        let effective = match self.draw.clip_rect() {
            Some(clip) => clip.intersect(rect).unwrap_or_default(),
            None => rect,
        };
        let enabled = self.interactive(rect) && effective.w > 0.0 && effective.h > 0.0;
        let mouse = self.input.mouse;
        let parent = self
            .scroll_cur
            .iter()
            .rposition(|region| {
                region.enabled
                    && region.id != id
                    && region.rect.contains(mouse)
                    && region.rect.contains(rect.center())
            })
            .map(|index| self.scroll_cur[index].id);
        self.scroll_cur.push(ScrollRegion {
            id,
            rect: effective,
            layer: self.draw.layer(),
            enabled,
            parent,
        });
    }

    /// The view that owns this frame's wheel: the innermost enabled view
    /// under the pointer, resolved against last frame's registry (which is
    /// complete, unlike this frame's building pass). `None` when nothing
    /// scrolls under the pointer — a view appearing for the first time
    /// claims the tick itself.
    fn wheel_owner(&self) -> Option<usize> {
        let mouse = self.input.mouse;
        self.scroll_prev.iter().rposition(|region| {
            region.enabled
                && region.rect.contains(mouse)
                && !self.surface_covers(region.layer, mouse)
        })
    }

    /// The scroll view enclosing `id`, where an unusable remainder of a
    /// wheel tick goes. Read from the registry (last frame's record, or this
    /// frame's already-registered ancestors for a view that just appeared).
    fn scroll_parent_of(&self, id: u64, viewport: Rect, owner: Option<usize>) -> Option<u64> {
        if let Some(region) = owner.and_then(|index| self.scroll_prev.get(index))
            && region.id == id
            && let Some(parent) = region.parent
        {
            return Some(parent);
        }
        let mouse = self.input.mouse;
        self.scroll_cur
            .iter()
            .rev()
            .find(|region| {
                region.id != id
                    && region.enabled
                    && region.rect.contains(mouse)
                    && region.rect.contains(viewport.center())
            })
            .map(|region| region.id)
    }

    /// Whether a view under the pointer may take this frame's keys. Keys
    /// belong to the focused widget, so only a focus-free frame scrolls;
    /// an open popup or modal keeps its own arrow-key navigation.
    fn scroll_keys_allowed(&self) -> bool {
        if self.editing() || self.focus_id.is_some() {
            return false;
        }
        !self
            .surfaces_cur
            .iter()
            .chain(self.surfaces_prev.iter())
            .any(|surface| surface.layer >= LAYER_POPUP)
    }

    /// Advance one scroll viewport by this frame's input, then by its own
    /// motion. Returns the offset the content should be drawn at.
    ///
    /// Shared by [`Ui::scroll_begin`] and the view layer's scroll
    /// containers, so both get the same wheel ownership, gliding,
    /// momentum, chaining, keyboard and scrollbar hand.
    pub(crate) fn scroll_viewport(&mut self, id: u64, viewport: Rect, content_h: f32) -> f32 {
        let config = self.scroll_config;
        let dt = self.input.dt.max(0.0);
        let max_offset = (content_h - viewport.h).max(0.0);
        let mut state = self
            .scrolls
            .get(&id)
            .copied()
            .unwrap_or_else(ScrollState::new);
        state.last_frame = self.frame;
        state.viewport = viewport.h;
        // Shrinking content re-clamps immediately — an offset past the new
        // end would leave dead space at the bottom.
        state.offset = state.offset.clamp(0.0, max_offset);
        state.target = state.target.clamp(0.0, max_offset);
        if matches!(state.motion, ScrollMotion::Idle) {
            state.target = state.offset;
        }
        self.scroll_contents.insert(id, content_h);

        let hovered = self.hit(viewport);
        let owner = self.wheel_owner();
        let owns_wheel = hovered && owner.is_none_or(|index| self.scroll_prev[index].id == id);
        let mut leftover = 0.0f32;

        // A nested view could not use all of last frame's tick: it arrives
        // here, on the widget we handed it to.
        if let Some((delta, _)) = self.scroll_chain.remove(&id)
            && delta != 0.0
        {
            let desired = state.target + delta;
            let clamped = desired.clamp(0.0, max_offset);
            leftover += desired - clamped;
            state.target = clamped;
            state.velocity = 0.0;
            if config.smooth && dt > 0.0 {
                state.motion = ScrollMotion::Smooth;
            } else {
                state.offset = clamped;
                state.motion = ScrollMotion::Idle;
            }
        }

        // --- Wheel -----------------------------------------------------------
        let mut wheel_this_frame = false;
        if owns_wheel && !self.wheel_taken {
            let smooth_px = -self.input.wheel * config.wheel_scale;
            let precise_px = -self.input.wheel_precise * config.wheel_scale;
            if smooth_px != 0.0 || precise_px != 0.0 {
                self.wheel_taken = true;
                state.wheel_frame = self.frame;
                wheel_this_frame = true;
                // Line-based notches glide to a destination; a running
                // programmatic scroll yields to the wheel from where it is.
                if smooth_px != 0.0 {
                    if matches!(state.motion, ScrollMotion::Tween { .. }) {
                        state.target = state.offset;
                    }
                    let desired = state.target + smooth_px;
                    let clamped = desired.clamp(0.0, max_offset);
                    leftover += desired - clamped;
                    state.target = clamped;
                    state.velocity = 0.0;
                    state.motion = if config.smooth && dt > 0.0 {
                        ScrollMotion::Smooth
                    } else {
                        state.offset = clamped;
                        ScrollMotion::Idle
                    };
                }
                // Precision devices are already smooth: track their pixels
                // one to one and remember the gesture's speed.
                if precise_px != 0.0 {
                    let raw = state.offset + precise_px;
                    let clamped = raw.clamp(0.0, max_offset);
                    leftover += raw - clamped;
                    state.offset = clamped;
                    state.target = clamped;
                    state.motion = ScrollMotion::Idle;
                    // Only a gesture that actually moved the view has a
                    // speed worth carrying (a push into the end does not).
                    if dt > 1e-4 && raw == clamped {
                        let instant = (precise_px / dt).clamp(-MAX_FLING, MAX_FLING);
                        state.velocity = state.velocity * 0.35 + instant * 0.65;
                    }
                }
            }
        }
        // The device stopped sending deltas mid-gesture: carry its speed
        // into a decaying glide. Devices that ship their own momentum keep
        // sending deltas, so their final delta is tiny and this adds
        // nothing.
        if !wheel_this_frame
            && state.wheel_frame + 1 == self.frame
            && config.momentum_decay > 0.0
            && state.velocity.abs() >= config.stop_threshold
            && matches!(state.motion, ScrollMotion::Idle)
        {
            state.motion = ScrollMotion::Fling;
        }

        // --- Keyboard --------------------------------------------------------
        if owns_wheel && config.keyboard && self.scroll_keys_allowed() {
            let nav = self.input.nav;
            let mut step = 0.0;
            if nav.down {
                step += config.arrow_step;
            }
            if nav.up {
                step -= config.arrow_step;
            }
            if nav.page_down {
                step += viewport.h * config.page_fraction;
            }
            if nav.page_up {
                step -= viewport.h * config.page_fraction;
            }
            if nav.home {
                step = -max_offset - state.target; // to the very top
            } else if nav.end {
                step = max_offset - state.target; // to the very end
            }
            if step != 0.0 {
                if matches!(state.motion, ScrollMotion::Tween { .. }) {
                    state.target = state.offset;
                }
                // Keys stay inside the view they are aimed at: unlike a
                // wheel tick, a page step does not spill into the parent.
                let clamped = (state.target + step).clamp(0.0, max_offset);
                state.target = clamped;
                state.velocity = 0.0;
                state.motion = if config.smooth && dt > 0.0 {
                    ScrollMotion::Smooth
                } else {
                    state.offset = clamped;
                    ScrollMotion::Idle
                };
            }
        }

        // --- Programmatic request -------------------------------------------
        if let Some(goal) = state.goal.take() {
            let to = goal
                .block
                .offset_for(goal.top, goal.height, viewport.h, state.offset)
                .clamp(0.0, max_offset);
            match goal.behavior.resolve(&config) {
                ScrollBehavior::Instant => {
                    state.offset = to;
                    state.target = to;
                    state.velocity = 0.0;
                    state.motion = ScrollMotion::Idle;
                }
                _ => {
                    let distance = (to - state.offset).abs();
                    // A request for the destination already being travelled
                    // to is the same intent repeated (a follow-tail caller
                    // runs every frame) — do not restart the glide.
                    let already = matches!(
                        state.motion,
                        ScrollMotion::Tween { to: heading, .. } if (heading - to).abs() < 0.5
                    );
                    state.target = to;
                    if distance <= 0.0 {
                        state.offset = to;
                        state.velocity = 0.0;
                        state.motion = ScrollMotion::Idle;
                    } else if already {
                        // Keep the glide that is already heading there.
                    } else if dt > 0.0 {
                        let travel = (distance / viewport.h.max(1.0)).clamp(0.0, 2.0);
                        let duration = config.smooth_duration * (0.6 + 0.2 * travel);
                        state.velocity = 0.0;
                        state.motion = ScrollMotion::Tween {
                            from: state.offset,
                            to,
                            elapsed: 0.0,
                            duration: duration.max(0.01),
                            curve: config.smooth_ease,
                        };
                    } else {
                        // No time this frame: place it, like a wheel tick on
                        // a zero-dt frame.
                        state.offset = to;
                        state.velocity = 0.0;
                        state.motion = ScrollMotion::Idle;
                    }
                }
            }
        }

        // --- Motion ----------------------------------------------------------
        match state.motion {
            ScrollMotion::Idle => {}
            ScrollMotion::Smooth => {
                if dt <= 0.0 {
                    state.offset = state.target;
                    state.motion = ScrollMotion::Idle;
                } else {
                    let k = 1.0 - (-dt / config.smoothness.max(1e-4)).exp();
                    state.offset += (state.target - state.offset) * k;
                    if (state.target - state.offset).abs() <= SETTLE_EPSILON {
                        state.offset = state.target;
                        state.motion = ScrollMotion::Idle;
                    }
                }
            }
            ScrollMotion::Fling => {
                let decay = config.momentum_decay.clamp(0.0, 0.999_9);
                if dt <= 0.0 || decay <= 0.0 {
                    state.velocity = 0.0;
                    state.motion = ScrollMotion::Idle;
                } else {
                    // Exact integral of v0 * decay^t over the frame, so the
                    // glide is identical at any frame rate.
                    let span = (decay.powf(dt) - 1.0) / decay.ln();
                    state.offset += state.velocity * span;
                    state.velocity *= decay.powf(dt);
                    if state.velocity.abs() < config.stop_threshold {
                        state.velocity = 0.0;
                        state.motion = ScrollMotion::Idle;
                    }
                }
            }
            ScrollMotion::Tween {
                from,
                to,
                mut elapsed,
                duration,
                curve,
            } => {
                if dt <= 0.0 {
                    state.offset = to;
                    state.motion = ScrollMotion::Idle;
                } else {
                    elapsed += dt;
                    let t = (elapsed / duration.max(1e-4)).clamp(0.0, 1.0);
                    state.offset = from + (to - from) * ease(curve, t).clamp(0.0, 1.0);
                    if t >= 1.0 {
                        state.offset = to;
                        state.motion = ScrollMotion::Idle;
                    } else {
                        state.motion = ScrollMotion::Tween {
                            from,
                            to,
                            elapsed,
                            duration,
                            curve,
                        };
                    }
                }
            }
        }

        // --- Scrollbar hand --------------------------------------------------
        if let Some((track, thumb, range)) =
            scrollbar_rects(&self.scrollbar, viewport, state.offset, content_h)
        {
            // A 5 px thumb is a miss waiting to happen: the grab area is
            // wider than the painted one.
            let grab = Rect::new(thumb.x - 2.0, thumb.y, thumb.w + 4.0, thumb.h);
            if self.input.mouse_pressed && self.hit(grab) && self.try_capture(id) {
                self.drags.insert(
                    id,
                    super::DragAnchor {
                        pos: self.input.mouse.1,
                        value: state.offset,
                        last_frame: self.frame,
                    },
                );
            }
            if self.active_id == Some(id) && self.capture_allowed(id, grab) {
                // A hand on the scrollbar owns the motion: no coasting.
                state.motion = ScrollMotion::Idle;
                state.velocity = 0.0;
                if let Some(anchor) = self.drags.get_mut(&id) {
                    anchor.last_frame = self.frame;
                    if self.input.mouse_down {
                        let travel = (track.h - thumb.h).max(1.0);
                        let dragged =
                            anchor.value + (self.input.mouse.1 - anchor.pos) / travel * range;
                        state.offset = dragged.clamp(0.0, range);
                        state.target = state.offset;
                    } else {
                        self.active_id = None;
                        self.drags.remove(&id);
                    }
                } else {
                    self.active_id = None;
                }
            } else if self.input.mouse_pressed
                && self.hit(track)
                && !thumb.contains(self.input.mouse)
            {
                // Clicking the track pages toward the click, like a browser.
                let direction = if self.input.mouse.1 < thumb.y {
                    -1.0
                } else {
                    1.0
                };
                let desired = state.offset + direction * viewport.h * config.page_fraction;
                let clamped = desired.clamp(0.0, range);
                state.target = clamped;
                state.velocity = 0.0;
                state.motion = if config.smooth && dt > 0.0 {
                    ScrollMotion::Smooth
                } else {
                    state.offset = clamped;
                    ScrollMotion::Idle
                };
            }
        }

        // --- Commit ----------------------------------------------------------
        // A coast that reached an end is over: keeping its velocity would
        // leave the view "moving" against a wall forever, pinning the
        // host's frame loop on a still screen.
        let clamped = state.offset.clamp(0.0, max_offset);
        if clamped != state.offset {
            state.offset = clamped;
            state.target = clamped;
            state.velocity = 0.0;
            state.motion = ScrollMotion::Idle;
        }
        if matches!(state.motion, ScrollMotion::Idle) {
            state.target = state.offset;
        }
        let offset = state.offset;
        self.scrolls.insert(id, state);

        let parent = (leftover != 0.0 && config.chain)
            .then(|| self.scroll_parent_of(id, viewport, owner))
            .flatten();
        if let Some(parent) = parent {
            self.scroll_chain.insert(parent, (leftover, self.frame));
        }
        offset
    }

    /// Draw the scrollbar for a viewport at `offset` (nothing when the
    /// content fits). Both scroll surfaces share it, so the thumb looks and
    /// reacts the same wherever it is built.
    pub(crate) fn draw_scrollbar(&mut self, id: u64, viewport: Rect, offset: f32, content_h: f32) {
        let bar = self.scrollbar;
        let Some((track, thumb, _)) = scrollbar_rects(&bar, viewport, offset, content_h) else {
            return;
        };
        let hovered = self.hit(viewport) || track.contains(self.input.mouse);
        let color = if hovered || self.active_id == Some(id) {
            bar.active
        } else {
            bar.idle
        };
        self.draw.round_rect(thumb, thumb.w * 0.5, color);
    }

    /// Queue a programmatic scroll for `label`. Resolved on the view's next
    /// advance, where the content extent is known and the clamp is exact.
    fn scroll_goal(
        &mut self,
        label: &str,
        top: f32,
        height: f32,
        block: ScrollAlign,
        behavior: ScrollBehavior,
    ) {
        let id = self.id(label);
        let state = self.scrolls.entry(id).or_insert_with(ScrollState::new);
        state.last_frame = self.frame;
        state.goal = Some(ScrollGoal {
            top: top.max(0.0),
            height: height.max(0.0),
            block,
            behavior,
        });
        // The request needs a frame in which the view is built.
        self.request_wake(0.0);
    }

    /// Scroll the view labelled `label` to a destination —
    /// `element.scrollTo({top, behavior})`.
    ///
    /// The request survives until the view is next built, so it may be made
    /// before or after the view; the clamp against the content happens when
    /// the view knows its extent. Wheel input takes over from a glide in
    /// flight, exactly like a browser.
    ///
    /// Repeating the same smooth request is free — a caller that pins a log
    /// to its newest row every frame re-states one intent, and the glide it
    /// already started is left alone. A view that must sit exactly at an
    /// offset *now* (a re-layout that would otherwise animate) should use
    /// [`Self::set_scroll`] or [`ScrollBehavior::Instant`].
    ///
    /// ```ignore
    /// ui.scroll_to("log", ScrollToOptions::smooth(f32::INFINITY));
    /// ```
    pub fn scroll_to(&mut self, label: &str, options: ScrollToOptions) {
        self.scroll_goal(
            label,
            options.top,
            0.0,
            ScrollAlign::Start,
            options.behavior,
        );
    }

    /// Scroll the view labelled `label` by `delta` pixels —
    /// `element.scrollBy(delta)`; positive travels toward the end.
    pub fn scroll_by(&mut self, label: &str, delta: f32, behavior: ScrollBehavior) {
        let id = self.id(label);
        let current = self.scrolls.get(&id).map_or(0.0, |state| state.offset);
        self.scroll_goal(label, current + delta, 0.0, ScrollAlign::Start, behavior);
    }

    /// Reveal `rect` inside the view labelled `label` —
    /// `element.scrollIntoView({block, behavior})`.
    ///
    /// `rect` is in **content coordinates**: `y` counts from the top of the
    /// scrolled content, not the screen. Inside a view, translate a drawn
    /// rect with `rect.y - content.y` (the content rect
    /// [`Self::scroll_begin`] returned) before passing it here.
    pub fn scroll_into_view(&mut self, label: &str, rect: Rect, options: ScrollIntoViewOptions) {
        self.scroll_goal(label, rect.y, rect.h, options.block, options.behavior);
    }

    /// Move the scroll view labelled `label` to `offset` (logical px from
    /// the top), immediately: a placed view, not a flick. Coasting and any
    /// glide stop. The next build clamps it to the content it measures, so
    /// pinning a transcript to its newest row is `set_scroll(label,
    /// f32::INFINITY)` once, not a layout of its own.
    ///
    /// For an animated or alignment-aware move, use [`Self::scroll_to`] /
    /// [`Self::scroll_into_view`].
    pub fn set_scroll(&mut self, label: &str, offset: f32) {
        let id = self.id(label);
        let state = self.scrolls.entry(id).or_insert_with(ScrollState::new);
        state.goal = None;
        state.offset = offset.max(0.0);
        state.target = state.offset;
        state.velocity = 0.0;
        state.motion = ScrollMotion::Idle;
        state.last_frame = self.frame;
    }

    /// `(offset, content_height)` of the view labelled `label` after its
    /// last build. The rect's own height is the viewport — the caller
    /// already has it when it laid the view out.
    pub fn scroll_state(&self, label: &str) -> (f32, f32) {
        let id = self.id(label);
        let offset = self.scrolls.get(&id).map_or(0.0, |state| state.offset);
        let content = self.scroll_contents.get(&id).copied().unwrap_or(0.0);
        (offset, content)
    }

    /// Full metrics for the view labelled `label` — offset, extents,
    /// coasting speed and whether it is still moving.
    pub fn scroll_metrics(&self, label: &str) -> ScrollMetrics {
        let id = self.id(label);
        let state = self
            .scrolls
            .get(&id)
            .copied()
            .unwrap_or_else(ScrollState::new);
        let content = self.scroll_contents.get(&id).copied().unwrap_or(0.0);
        ScrollMetrics {
            offset: state.offset,
            viewport: state.viewport,
            content,
            max_offset: (content - state.viewport).max(0.0),
            velocity: state.velocity,
            animating: state.moving(),
        }
    }
}
