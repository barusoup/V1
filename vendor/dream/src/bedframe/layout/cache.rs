//! Solve caching: a structural hash accumulated in `push` plus the
//! inputs `solve` reads (root rect, scale, scroll offsets). Equal hash →
//! the previous output is still valid and steps 1–3 are skipped.

use super::{Leaf, Rect, Size, Style};
use std::hash::{Hash, Hasher};

#[derive(Default)]
pub(crate) struct SolveHash {
    h: u64,
}

impl SolveHash {
    pub fn new() -> Self {
        Self {
            h: 0xcbf2_9ce4_8422_2325,
        }
    }

    /// Continue hashing from an existing accumulated value.
    pub fn from_seed(h: u64) -> Self {
        Self { h }
    }

    fn bits(&mut self, b: u64) {
        self.h = (self.h ^ b).wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn f32(&mut self, v: f32) {
        self.bits(u64::from(v.to_bits()));
    }

    fn opt_f32(&mut self, v: &Option<f32>) {
        self.bits(v.is_some() as u64);
        if let Some(v) = v {
            self.f32(*v);
        }
    }

    fn size(&mut self, s: &Size) {
        match s {
            Size::Px(v) => {
                self.bits(1);
                self.f32(*v);
            }
            Size::Fit => self.bits(2),
            Size::Fill => self.bits(3),
            Size::Percent(v) => {
                self.bits(4);
                self.f32(*v);
            }
        }
    }

    fn edges_f32(&mut self, e: &super::Edges<f32>) {
        for v in [e.top, e.right, e.bottom, e.left] {
            self.f32(v);
        }
    }

    /// Fold one pushed node into the tree hash.
    pub fn push_node(&mut self, parent: Option<super::solver::NodeId>, s: &Style, leaf: &Leaf) {
        self.bits(parent.map_or(u64::MAX, |p| u64::from(p.0)));
        self.bits(s.display as u64);
        self.bits(s.direction as u64);
        self.bits(s.wrap as u64);
        self.f32(s.gap);
        self.f32(s.cross_gap);
        self.edges_f32(&s.padding);
        self.edges_f32(&s.margin);
        self.bits(s.justify as u64);
        self.bits(s.align_items as u64);
        self.bits(s.align_self.map_or(u64::MAX, |a| a as u64));
        self.f32(s.grow);
        self.f32(s.shrink);
        self.opt_f32(&s.basis);
        self.size(&s.width);
        self.size(&s.height);
        for o in [s.min_w, s.max_w, s.min_h, s.max_h] {
            self.opt_f32(&o);
        }
        self.opt_f32(&s.aspect_ratio);
        self.bits(match s.overflow {
            super::Overflow::Visible => 0,
            super::Overflow::Clip => 1,
            super::Overflow::Scroll { x: false, y: false } => 2,
            super::Overflow::Scroll { x: true, y: false } => 3,
            super::Overflow::Scroll { x: false, y: true } => 4,
            super::Overflow::Scroll { x: true, y: true } => 5,
        });
        match s.position {
            super::Position::Relative => self.bits(0),
            super::Position::Absolute(e) => {
                self.bits(1);
                for v in [e.top, e.right, e.bottom, e.left] {
                    self.opt_f32(&v);
                }
            }
        }
        self.bits(u64::from(s.place.0 as u8) << 8 | u64::from(s.place.1 as u8));
        self.bits(s.key.unwrap_or(u64::MAX));
        match leaf {
            Leaf::None => self.bits(0),
            Leaf::Fixed { w, h, baseline } => {
                self.bits(1);
                self.f32(*w);
                self.f32(*h);
                self.opt_f32(baseline);
            }
            Leaf::Measured { key } => {
                self.bits(2);
                self.bits(*key);
            }
        }
    }

    /// The remaining solve inputs.
    pub fn solve_inputs(&mut self, root: Rect, scale: f32, scrolls: &[(f32, f32)]) {
        for v in [root.x, root.y, root.w, root.h, scale] {
            self.f32(v);
        }
        for &(x, y) in scrolls {
            self.f32(x);
            self.f32(y);
        }
    }

    pub fn value(&self) -> u64 {
        self.h
    }
}

// Silence unused warnings for Hash infra kept minimal.
#[allow(dead_code)]
fn _hash<T: Hash>(t: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}
