//! Layout helpers for HUD and menu placement.
//!
//! Deliberately small: a cursor that hands out rects (`Row`/`Column`) and an
//! anchor calculator for screen-relative placement. Anything fancier is
//! composed from [`crate::ui::Rect`] arithmetic directly — the draw layer has
//! no layout assumptions.

use super::Rect;

mod cache;
mod flex;
mod measure;
mod snap;
mod solver;
#[cfg(test)]
mod solver_tests;
mod stack;
mod style;
pub use measure::{Leaf, LeafSize, Measure};
pub use snap::*;
pub use solver::{LayoutTree, NodeId, SolveStats};
pub use style::{
    Align as LayoutAlign, Direction, Display, Edges, Justify, Overflow, Position, Size, Style,
};

/// Horizontal alignment inside a parent rect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AlignX {
    #[default]
    Start,
    Center,
    End,
}

/// Vertical alignment inside a parent rect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AlignY {
    #[default]
    Start,
    Center,
    End,
}

/// Where a rect sits inside its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Anchor {
    #[default]
    TopLeft,
    Top,
    TopRight,
    Left,
    Center,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

impl Anchor {
    /// Place a `size`-sized rect at `anchor` inside `parent`, offset by
    /// `margin` pixels from the parent's edges. `margin` is ignored on
    /// centred axes — a centred rect has no edge to measure it from.
    pub fn place_xy(self, parent: Rect, size: (f32, f32), margin_x: f32, margin_y: f32) -> Rect {
        let (w, h) = size;
        let (x, y) = match self {
            Anchor::TopLeft => (parent.x + margin_x, parent.y + margin_y),
            Anchor::Top => (parent.center().0 - w * 0.5, parent.y + margin_y),
            Anchor::TopRight => (parent.max().0 - w - margin_x, parent.y + margin_y),
            Anchor::Left => (parent.x + margin_x, parent.center().1 - h * 0.5),
            Anchor::Center => (parent.center().0 - w * 0.5, parent.center().1 - h * 0.5),
            Anchor::Right => (parent.max().0 - w - margin_x, parent.center().1 - h * 0.5),
            Anchor::BottomLeft => (parent.x + margin_x, parent.max().1 - h - margin_y),
            Anchor::Bottom => (parent.center().0 - w * 0.5, parent.max().1 - h - margin_y),
            Anchor::BottomRight => (parent.max().0 - w - margin_x, parent.max().1 - h - margin_y),
        };
        Rect::new(x, y, w, h)
    }

    /// Place a `size`-sized rect at `anchor` inside `parent`, offset by
    /// `margin` pixels from the parent's edges. `margin` is ignored on
    /// centred axes.
    pub fn place(self, parent: Rect, size: (f32, f32), margin: f32) -> Rect {
        let (w, h) = size;
        let (x, y) = match self {
            Anchor::TopLeft => (parent.x + margin, parent.y + margin),
            Anchor::Top => (parent.center().0 - w * 0.5, parent.y + margin),
            Anchor::TopRight => (parent.max().0 - w - margin, parent.y + margin),
            Anchor::Left => (parent.x + margin, parent.center().1 - h * 0.5),
            Anchor::Center => (parent.center().0 - w * 0.5, parent.center().1 - h * 0.5),
            Anchor::Right => (parent.max().0 - w - margin, parent.center().1 - h * 0.5),
            Anchor::BottomLeft => (parent.x + margin, parent.max().1 - h - margin),
            Anchor::Bottom => (parent.center().0 - w * 0.5, parent.max().1 - h - margin),
            Anchor::BottomRight => (parent.max().0 - w - margin, parent.max().1 - h - margin),
        };
        Rect::new(x, y, w, h)
    }
}

/// Direction a cursor advances.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    Horizontal,
    Vertical,
}

/// Hands out equally sized rects along one axis — buttons in a row, menu
/// entries in a column.
#[derive(Clone, Debug)]
pub struct Cursor {
    area: Rect,
    axis: Axis,
    gap: f32,
    offset: f32,
    per_item: f32,
}

impl Cursor {
    /// `item_size` is the height for columns, the width for rows.
    pub fn new(area: Rect, axis: Axis, gap: f32, item_size: f32) -> Self {
        Self {
            area,
            axis,
            gap,
            offset: 0.0,
            per_item: item_size,
        }
    }

    /// Column of full-width entries.
    pub fn column(area: Rect, gap: f32, item_height: f32) -> Self {
        Self::new(area, Axis::Vertical, gap, item_height)
    }

    /// Row of full-height entries with the given width.
    pub fn row(area: Rect, gap: f32, item_width: f32) -> Self {
        Self::new(area, Axis::Horizontal, gap, item_width)
    }

    /// Hand out a single slot of custom `size` along the cursor's axis,
    /// advancing by `size + gap`.
    pub fn next_sized(&mut self, size: f32) -> Option<Rect> {
        let offset = self.offset;
        let rect = match self.axis {
            Axis::Vertical => {
                let y = self.area.y + offset;
                if y + size > self.area.max().1 {
                    return None;
                }
                Rect::new(self.area.x, y, self.area.w, size)
            }
            Axis::Horizontal => {
                let x = self.area.x + offset;
                if x + size > self.area.max().0 {
                    return None;
                }
                Rect::new(x, self.area.y, size, self.area.h)
            }
        };
        self.offset += size + self.gap;
        Some(rect)
    }

    /// Advance the cursor position by `pixels` along its axis (extra spacing).
    pub fn advance(&mut self, pixels: f32) {
        self.offset = (self.offset + pixels).max(0.0);
    }

    /// Current offset along the cursor's axis.
    pub fn position(&self) -> f32 {
        self.offset
    }

    /// Rects for `count` items, stopping early when the area fills up.
    pub fn take(&mut self, count: usize) -> Vec<Rect> {
        (0..count).map_while(|_| self.next()).collect()
    }

    /// Remaining space after the slots handed out so far.
    pub fn remainder(&self) -> Rect {
        let offset = self.offset;
        match self.axis {
            Axis::Vertical => Rect::new(
                self.area.x,
                self.area.y + offset,
                self.area.w,
                (self.area.h - offset).max(0.0),
            ),
            Axis::Horizontal => Rect::new(
                self.area.x + offset,
                self.area.y,
                (self.area.w - offset).max(0.0),
                self.area.h,
            ),
        }
    }
}

impl Iterator for Cursor {
    type Item = Rect;

    /// Next slot, or `None` when the area is exhausted.
    fn next(&mut self) -> Option<Rect> {
        self.next_sized(self.per_item)
    }
}

/// Split `area` into `count` equal slots along `axis` with `gap` between
/// them. Slots snap to whole logical pixels and tile exactly.
pub fn split(area: Rect, axis: Axis, count: usize, gap: f32) -> Vec<Rect> {
    split_scaled(area, axis, count, gap, 1.0)
}

/// [`split`] at `scale` physical pixels per logical pixel — edges land on
/// the physical grid.
pub fn split_scaled(area: Rect, axis: Axis, count: usize, gap: f32, scale: f32) -> Vec<Rect> {
    if count == 0 {
        return Vec::new();
    }
    let (start, extent) = match axis {
        Axis::Vertical => (area.y, area.h),
        Axis::Horizontal => (area.x, area.w),
    };
    divide(start, extent, count, gap, scale)
        .into_iter()
        .map(|(pos, size)| match axis {
            Axis::Vertical => Rect::new(area.x, pos, area.w, size),
            Axis::Horizontal => Rect::new(pos, area.y, size, area.h),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area() -> Rect {
        Rect::new(0.0, 0.0, 100.0, 100.0)
    }

    #[test]
    fn anchor_places_corners() {
        let parent = area();
        let size = (20.0, 10.0);
        assert_eq!(
            Anchor::TopLeft.place(parent, size, 5.0),
            Rect::new(5.0, 5.0, 20.0, 10.0)
        );
        assert_eq!(
            Anchor::BottomRight.place(parent, size, 5.0),
            Rect::new(75.0, 85.0, 20.0, 10.0)
        );
        assert_eq!(
            Anchor::Center.place(parent, size, 0.0),
            Rect::new(40.0, 45.0, 20.0, 10.0)
        );
    }

    #[test]
    fn anchor_place_xy_margins_per_axis() {
        let parent = area();
        assert_eq!(
            Anchor::BottomRight.place_xy(parent, (20.0, 10.0), 5.0, 9.0),
            Rect::new(75.0, 81.0, 20.0, 10.0)
        );
    }

    #[test]
    fn rect_center_align_fit_and_snap() {
        let parent = area();
        assert_eq!(
            parent.center_in((20.0, 10.0)),
            Rect::new(40.0, 45.0, 20.0, 10.0)
        );
        assert_eq!(
            parent.align_in((20.0, 10.0), AlignX::End, AlignY::Start),
            Rect::new(80.0, 0.0, 20.0, 10.0)
        );
        // 4:3 in a 100x100 box is 100x75, centred.
        assert_eq!(
            parent.fit_aspect(4.0 / 3.0),
            Rect::new(0.0, 12.5, 100.0, 75.0)
        );
        let r = Rect::new(10.4, 20.6, 33.2, 44.5).snap(1.0);
        assert_eq!(r, Rect::new(10.0, 21.0, 34.0, 44.0));
    }

    #[test]
    fn cursor_column_stops_at_the_edge() {
        let mut cursor = Cursor::column(area(), 2.0, 30.0);
        assert!(cursor.next().is_some());
        assert!(cursor.next().is_some());
        assert!(cursor.next().is_some());
        assert!(cursor.next().is_none(), "4 * 30 + gaps exceed 100");
    }

    #[test]
    fn cursor_remainder_tracks_progress() {
        let mut cursor = Cursor::column(Rect::new(0.0, 0.0, 50.0, 100.0), 0.0, 25.0);
        cursor.next();
        cursor.next();
        assert_eq!(cursor.remainder(), Rect::new(0.0, 50.0, 50.0, 50.0));
    }

    #[test]
    fn split_divides_evenly() {
        // 100 wide, 4 slots, 3 gaps of 10: the 70px share tiles exactly
        // in whole-pixel slots of 17/18 with the gaps between them.
        let slots = split(area(), Axis::Horizontal, 4, 10.0);
        assert_eq!(slots.len(), 4);
        assert_eq!(slots[0], Rect::new(0.0, 0.0, 17.0, 100.0));
        assert_eq!(slots[1], Rect::new(27.0, 0.0, 18.0, 100.0));
        assert_eq!(slots[2], Rect::new(55.0, 0.0, 17.0, 100.0));
        assert_eq!(slots[3], Rect::new(82.0, 0.0, 18.0, 100.0));
    }

    #[test]
    fn split_zero_count_is_empty() {
        assert!(split(area(), Axis::Vertical, 0, 1.0).is_empty());
    }
}

/// One item's share of a [`flow`]: fixed size, content-measured size, or
/// a weighted slice of whatever is left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Length {
    /// Exactly this many pixels along the axis.
    Fixed(f32),
    /// A share of the remaining space, proportional to the weight.
    Fill(f32),
    /// The measured content size (caller supplies the number).
    Content(f32),
}

/// Lay out `items` along `axis` inside `area` with `gap` between them:
/// `Fixed` and `Content` keep their own sizes, `Fill`s split whatever
/// space remains in proportion to their weights. Sizes and gaps are
/// sanitised (nonfinite or negative becomes zero), rects never go
/// negative, and an overfull area clamps the later items to zero size
/// rather than pushing them outside.
pub fn flow(area: Rect, axis: Axis, gap: f32, items: &[Length]) -> Vec<Rect> {
    let gap = if gap.is_finite() { gap.max(0.0) } else { 0.0 };
    let (x, y, w, h) = (area.x, area.y, area.w, area.h);
    let extent = match axis {
        Axis::Horizontal => w.max(0.0),
        Axis::Vertical => h.max(0.0),
    };
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    let gaps = gap * n.saturating_sub(1) as f32;
    let mut sizes = vec![0.0f32; n];
    let mut fixed_total = 0.0f32;
    let mut weight_total = 0.0f32;
    for (i, item) in items.iter().enumerate() {
        match *item {
            Length::Fixed(v) | Length::Content(v) => {
                let v = if v.is_finite() { v.max(0.0) } else { 0.0 };
                sizes[i] = v;
                fixed_total += v;
            }
            Length::Fill(weight) => {
                weight_total += if weight.is_finite() {
                    weight.max(0.0)
                } else {
                    0.0
                };
            }
        }
    }
    // Weight math in f64: `weight * remaining` can overflow f32 even
    // with finite inputs (a huge weight times a huge width).
    let remaining = (f64::from(extent) - f64::from(gaps) - f64::from(fixed_total)).max(0.0);
    let weight_total_d = f64::from(weight_total);
    for (i, item) in items.iter().enumerate() {
        if let Length::Fill(weight) = *item {
            let weight = if weight.is_finite() {
                weight.max(0.0)
            } else {
                0.0
            };
            sizes[i] = if weight_total_d > 0.0 {
                ((remaining * f64::from(weight)) / weight_total_d).min(f32::MAX as f64) as f32
            } else {
                0.0
            };
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut cursor = 0.0f32;
    for size in sizes {
        // An overfull area clamps each item to what remains — never a
        // negative or spilling rect.
        let shown = size.min((extent - cursor).max(0.0));
        out.push(place(axis, x, y, cursor, shown, w, h));
        cursor += shown + gap;
    }
    out
}

fn place(axis: Axis, x: f32, y: f32, offset: f32, size: f32, w: f32, h: f32) -> Rect {
    match axis {
        Axis::Horizontal => Rect::new(x + offset, y, size, h),
        Axis::Vertical => Rect::new(x, y + offset, w, size),
    }
}

#[cfg(test)]
mod flex_tests {
    use super::*;

    #[test]
    fn a_row_of_fills_shares_the_remainder() {
        let area = Rect::new(0.0, 0.0, 100.0, 20.0);
        let rects = flow(
            area,
            Axis::Horizontal,
            0.0,
            &[Length::Fixed(20.0), Length::Fill(1.0), Length::Fill(1.0)],
        );
        assert_eq!(rects.len(), 3);
        assert_eq!((rects[0].x, rects[0].w), (0.0, 20.0));
        assert!((rects[1].w - 40.0).abs() < 0.01);
        assert!((rects[2].w - 40.0).abs() < 0.01);
        assert!((rects[2].x - 60.0).abs() < 0.01);
    }

    #[test]
    fn gaps_and_content_and_overfull() {
        let area = Rect::new(0.0, 0.0, 110.0, 10.0);
        let rects = flow(
            area,
            Axis::Horizontal,
            5.0,
            &[Length::Content(30.0), Length::Fill(1.0)],
        );
        assert_eq!(rects[1].x, 35.0);
        assert!((rects[1].w - 75.0).abs() < 0.01, "fill takes the rest");
        // Overfull fixed items clamp to what remains — never spill out.
        let over = flow(
            area,
            Axis::Horizontal,
            0.0,
            &[Length::Fixed(80.0), Length::Fixed(80.0)],
        );
        assert_eq!(over[1].x, 80.0);
        assert_eq!(over[1].w, 30.0, "clamped to remaining space");
        assert!(over.iter().all(|r| r.x + r.w <= area.x + area.w + 0.01));
    }

    #[test]
    fn a_column_flows_vertically_and_sanitizes() {
        let area = Rect::new(0.0, 0.0, 50.0, 90.0);
        let rects = flow(
            area,
            Axis::Vertical,
            f32::NAN,
            &[Length::Fixed(30.0), Length::Fill(1.0)],
        );
        assert_eq!(rects[0].h, 30.0);
        assert!((rects[1].y - 30.0).abs() < 0.01);
        assert!((rects[1].h - 60.0).abs() < 0.01);
        assert!(rects.iter().all(|r| r.w >= 0.0 && r.h >= 0.0));
    }
}

/// Lay out `columns × rows` equal cells inside `area` in row-major order,
/// separated by `col_gap` horizontally and `row_gap` vertically.
pub fn grid(area: Rect, columns: usize, rows: usize, col_gap: f32, row_gap: f32) -> Vec<Rect> {
    grid_scaled(area, columns, rows, col_gap, row_gap, 1.0)
}

/// [`grid`] at `scale` physical pixels per logical pixel.
pub fn grid_scaled(
    area: Rect,
    columns: usize,
    rows: usize,
    col_gap: f32,
    row_gap: f32,
    scale: f32,
) -> Vec<Rect> {
    if columns == 0 || rows == 0 {
        return Vec::new();
    }
    let col_gap = if col_gap.is_finite() {
        col_gap.max(0.0)
    } else {
        0.0
    };
    let row_gap = if row_gap.is_finite() {
        row_gap.max(0.0)
    } else {
        0.0
    };
    let cols = divide(area.x, area.w, columns, col_gap, scale);
    let rows_ = divide(area.y, area.h, rows, row_gap, scale);
    let mut out = Vec::with_capacity(columns * rows);
    for &(y, h) in &rows_ {
        for &(x, w) in &cols {
            out.push(Rect::new(x, y, w, h));
        }
    }
    out
}
