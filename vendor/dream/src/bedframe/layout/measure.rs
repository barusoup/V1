//! Leaf measurement: what a node reports when its style says `Fit`.
use super::solver::NodeId;

/// A node's intrinsic content, when it has any.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Leaf {
    /// Pure container — size comes entirely from children/style.
    None,
    /// A leaf with a fixed intrinsic size (icon, checkbox box, image).
    Fixed {
        w: f32,
        h: f32,
        /// Baseline offset from the leaf's top, for Align::Baseline.
        baseline: Option<f32>,
    },
    /// Host-measured leaf (text): the host is asked through [`Measure`].
    Measured { key: u64 },
}

/// A leaf's reported size; `baseline` is the offset from its top edge.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LeafSize {
    pub w: f32,
    pub h: f32,
    pub baseline: Option<f32>,
}

/// Host measurement hook. `avail_w` is the width the leaf must fit
/// (`None` for intrinsic measurement) — a text host returns the height
/// the text needs at that width.
pub trait Measure {
    fn measure(&mut self, node: NodeId, key: u64, avail_w: Option<f32>) -> LeafSize;
}
