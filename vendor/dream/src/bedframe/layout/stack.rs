//! Stack display layout: children overlap in the content box, placed by
//! their `place` (AlignX, AlignY) and sized against the box.

use super::measure::Measure;
use super::solver::LayoutTree;
use super::style::Position;
use super::{AlignX, AlignY, Rect};

/// Lay out `id`'s (non-absolute) children stacked in `content`.
pub(super) fn layout_stack(tree: &mut LayoutTree, id: u32, content: Rect, host: &mut dyn Measure) {
    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;
    for c in tree.child_ids(id) {
        if matches!(tree.node(c).style.position, Position::Absolute(_)) {
            continue;
        }
        let (w, h) = tree.size_in_box_pub(c, content.w, content.h, host);
        let (px, py) = tree.node(c).style.place;
        let x = match px {
            AlignX::Start => content.x,
            AlignX::Center => content.x + (content.w - w) * 0.5,
            AlignX::End => content.x + content.w - w,
        };
        let y = match py {
            AlignY::Start => content.y,
            AlignY::Center => content.y + (content.h - h) * 0.5,
            AlignY::End => content.y + content.h - h,
        };
        tree.node_mut(c).abs = Rect::new(x, y, w, h);
        max_x = max_x.max(x - content.x + w);
        max_y = max_y.max(y - content.y + h);
        tree.layout_children(c, host);
    }
    let p = tree.node(id).style.padding;
    tree.node_mut(id).content = (max_x + p.horizontal(), max_y + p.vertical());
}
