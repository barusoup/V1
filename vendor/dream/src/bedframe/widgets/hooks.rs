//! Keyed component state, scoped context and enter/exit presence.

use super::Ui;
use crate::ui::tween::{Anim, Ease};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Shared component state, independent of a mutable borrow of [`Ui`].
///
/// Cloning a handle shares its value without allocating. Mutate it from event
/// responses; unconditional mutations in an L3 build would run again on replay.
/// A host changing state outside a frame must also request its next UI frame.
#[derive(Debug)]
pub struct State<T> {
    value: Rc<RefCell<T>>,
}

impl<T> Clone for State<T> {
    fn clone(&self) -> Self {
        Self {
            value: Rc::clone(&self.value),
        }
    }
}

impl<T> State<T> {
    /// Create a host-owned state handle, also usable as a context provider.
    pub fn new(value: T) -> Self {
        Self {
            value: Rc::new(RefCell::new(value)),
        }
    }

    /// Read without cloning the value. Nested mutation of this same state
    /// while `read` runs panics, following [`RefCell`]'s borrowing rules.
    pub fn with<R>(&self, read: impl FnOnce(&T) -> R) -> R {
        read(&self.value.borrow())
    }

    /// Change the shared value. Do not reborrow this state inside `update`.
    pub fn update<R>(&self, update: impl FnOnce(&mut T) -> R) -> R {
        update(&mut self.value.borrow_mut())
    }

    pub fn set(&self, value: T) {
        *self.value.borrow_mut() = value;
    }

    /// Take a value snapshot; the handle itself never needs `T: Clone`.
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.value.borrow().clone()
    }
}

struct HookValue {
    value: Rc<dyn Any>,
    frame: u64,
}

type ContextStack = Rc<RefCell<Vec<Rc<dyn Any>>>>;

#[derive(Default)]
pub(super) struct Hooks {
    states: HashMap<u64, HookValue>,
    contexts: ContextStack,
}

struct ContextScope {
    stack: ContextStack,
    depth: usize,
}

impl Drop for ContextScope {
    fn drop(&mut self) {
        self.stack.borrow_mut().truncate(self.depth);
    }
}

impl Hooks {
    pub(super) fn begin(&mut self) {
        self.contexts.borrow_mut().clear();
    }

    pub(super) fn retire(&mut self, frame: u64) {
        self.states
            .retain(|_, value| frame.saturating_sub(value.frame) < 600);
        // As with widget state, preserve this frame's working set.
        if self.states.len() > 4096 {
            let mut ages: Vec<_> = self
                .states
                .iter()
                .filter(|(_, value)| value.frame != frame)
                .map(|(&id, value)| (value.frame, id))
                .collect();
            ages.sort_unstable();
            let count = self.states.len() - 4096;
            for (_, id) in ages.into_iter().take(count) {
                self.states.remove(&id);
            }
        }
        if self.states.capacity() > 1024 && self.states.len() * 4 < self.states.capacity() {
            self.states.shrink_to_fit();
        }
    }
}

impl Ui {
    /// Persistent state keyed by the current id scope and `key`.
    ///
    /// Initialization runs once, including on a first-frame layout replay.
    /// Keep the value's type stable for a key; changing it panics. Inactive
    /// state follows widget retirement (600 frames, periodic bounded sweep).
    /// Retained external handles stay valid after retirement; mounting the
    /// retired key creates a new value.
    ///
    /// ```
    /// use dream::ui::widgets::Ui;
    /// let mut ui = Ui::new(Default::default(), Default::default());
    /// ui.begin(Default::default(), (200.0, 100.0));
    /// let count = ui.use_state("count", || 0usize);
    /// count.update(|n| *n += 1); // In an app, guard this with a clicked response.
    /// assert_eq!(count.get(), 1);
    /// let _ = ui.end();
    /// ```
    pub fn use_state<T: 'static>(&mut self, key: &str, init: impl FnOnce() -> T) -> State<T> {
        let id = self.id(key);
        let entry = self.hooks.states.entry(id).or_insert_with(|| HookValue {
            value: Rc::new(RefCell::new(init())),
            frame: self.frame,
        });
        entry.frame = self.frame;
        State {
            value: Rc::clone(&entry.value)
                .downcast::<RefCell<T>>()
                .unwrap_or_else(|_| panic!("use_state type changed for key {key:?}")),
        }
    }

    /// Provide a state handle to descendants by its value type. The nearest
    /// provider wins; the previous context is restored even if `build` panics.
    /// Reuse a handle from [`State::new`] or [`Self::use_state`] across frames
    /// so providing/reading context does not allocate in steady state.
    pub fn with_context<T: 'static, R>(
        &mut self,
        value: &State<T>,
        build: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let stack = Rc::clone(&self.hooks.contexts);
        let depth = stack.borrow().len();
        stack.borrow_mut().push(value.value.clone());
        let _scope = ContextScope { stack, depth };
        build(self)
    }

    /// Read the nearest context for `T`. Different value types coexist;
    /// wrap two contexts of the same primitive type in distinct newtypes.
    pub fn use_context<T: 'static>(&self) -> Option<State<T>> {
        self.hooks.contexts.borrow().iter().rev().find_map(|value| {
            Rc::clone(value)
                .downcast::<RefCell<T>>()
                .ok()
                .map(|value| State { value })
        })
    }

    /// Animate mounting/unmounting. Call every frame, then build the content
    /// while [`Presence::is_present`] is true and gate input with
    /// [`Presence::is_interactive`]. Apply `opacity` in the component's painter.
    /// The exit remains mounted until it finishes; reversing an exit continues
    /// from its current opacity. Uses the scoped animation clock, replay freeze
    /// and the host's motion-scale preference.
    pub fn presence(&mut self, key: &str, visible: bool, duration: f32) -> Presence {
        self.push_id(key);
        let id = self.id("##presence");
        self.pop_id();
        let target = if visible { 1.0 } else { 0.0 };
        let dt = self.anim_dt();
        let duration = self.motion_time(duration);
        let anim = self.anims_f32.entry(id).or_insert_with(|| Anim::new(0.0));
        if !duration.is_finite() || duration <= 0.0 {
            anim.set(target);
        } else if anim.target() != target {
            anim.animate_to(target, duration, Ease::CubicOut);
        }
        anim.stamp(self.frame);
        let opacity = anim.update(dt);
        let opacity = opacity.clamp(0.0, 1.0);
        let phase = if visible {
            if opacity == 1.0 {
                PresencePhase::Visible
            } else {
                PresencePhase::Entering
            }
        } else if opacity == 0.0 {
            PresencePhase::Hidden
        } else {
            PresencePhase::Exiting
        };
        Presence { opacity, phase }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresencePhase {
    Hidden,
    Entering,
    Visible,
    Exiting,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Presence {
    pub opacity: f32,
    pub phase: PresencePhase,
}

impl Presence {
    pub fn is_present(self) -> bool {
        self.phase != PresencePhase::Hidden
    }

    /// Exiting content is still painted but must no longer accept input.
    pub fn is_interactive(self) -> bool {
        matches!(self.phase, PresencePhase::Entering | PresencePhase::Visible)
    }
}
