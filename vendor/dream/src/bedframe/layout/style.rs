//! Layout style data for the solver: flex/stack geometry in the CSS
//! flexbox model, plus BedFrame's stacking and absolute positioning.
//! Everything is plain `Copy` data — the tree owns the structure.

use super::{AlignX, AlignY};

/// One axis of a box's size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Size {
    /// Exactly this many logical pixels.
    Px(f32),
    /// Content-measured natural size.
    #[default]
    Fit,
    /// Share of the free space (flex-grow ≥ 1 on the main axis, stretch
    /// on the cross axis).
    Fill,
    /// Fraction of the parent's content box on this axis, /100.
    Percent(f32),
}

/// Four edges of a box (`Option<f32>` variants mean "edge unset").
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Edges<T> {
    pub top: T,
    pub right: T,
    pub bottom: T,
    pub left: T,
}

impl<T: Copy> Edges<T> {
    /// Same value on all four edges.
    pub fn all(v: T) -> Self {
        Self {
            top: v,
            right: v,
            bottom: v,
            left: v,
        }
    }
}

impl Edges<f32> {
    /// Horizontal and vertical values.
    pub fn xy(x: f32, y: f32) -> Self {
        Self {
            top: y,
            right: x,
            bottom: y,
            left: x,
        }
    }

    /// All zeros.
    pub fn zero() -> Self {
        Self::all(0.0)
    }

    /// left + right.
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    /// top + bottom.
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

/// Main axis direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
    Row,
    #[default]
    Column,
}

/// Main-axis distribution of leftover space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Justify {
    #[default]
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

/// Cross-axis alignment within a line (or per-item override).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    Start,
    Center,
    End,
    #[default]
    Stretch,
    Baseline,
}

/// What content that exceeds the content box does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Overflow {
    #[default]
    Visible,
    Clip,
    /// Scrollable on the flagged axes (children lay out unbounded on
    /// those axes; the view layer owns the actual clipping).
    Scroll {
        x: bool,
        y: bool,
    },
}

/// In-flow or out-of-flow placement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Position {
    #[default]
    Relative,
    /// Offset from the parent's *padding* box; `None` edges fall back to
    /// the node's own size.
    Absolute(Edges<Option<f32>>),
}

/// Box layout algorithm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Display {
    #[default]
    Flex,
    /// Children overlap inside the content box, placed by `place`.
    Stack,
}

/// One node's style — the full CSS-flexbox subset the solver reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    pub display: Display,
    pub direction: Direction,
    pub wrap: bool,
    /// Gap between items on the main axis.
    pub gap: f32,
    /// Gap between wrapped lines on the cross axis.
    pub cross_gap: f32,
    pub padding: Edges<f32>,
    pub margin: Edges<f32>,
    pub justify: Justify,
    pub align_items: Align,
    pub align_self: Option<Align>,
    pub grow: f32,
    pub shrink: f32,
    /// Flex basis override (main axis hypothetical size).
    pub basis: Option<f32>,
    pub width: Size,
    pub height: Size,
    pub min_w: Option<f32>,
    pub max_w: Option<f32>,
    pub min_h: Option<f32>,
    pub max_h: Option<f32>,
    /// w/h ratio: derives the missing axis when exactly one is definite.
    pub aspect_ratio: Option<f32>,
    pub overflow: Overflow,
    pub position: Position,
    /// Stack children: where they sit inside the parent's content box.
    pub place: (AlignX, AlignY),
    /// Measured-leaf identity (host-side measure cache key).
    pub key: Option<u64>,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            display: Display::Flex,
            direction: Direction::Column,
            wrap: false,
            gap: 0.0,
            cross_gap: 0.0,
            padding: Edges::zero(),
            margin: Edges::zero(),
            justify: Justify::Start,
            align_items: Align::Stretch,
            align_self: None,
            grow: 0.0,
            shrink: 1.0,
            basis: None,
            width: Size::Fit,
            height: Size::Fit,
            min_w: None,
            max_w: None,
            min_h: None,
            max_h: None,
            aspect_ratio: None,
            overflow: Overflow::Visible,
            position: Position::Relative,
            place: (AlignX::Start, AlignY::Start),
            key: None,
        }
    }
}

impl Style {
    /// A row-direction flex container.
    pub fn row() -> Self {
        Self {
            direction: Direction::Row,
            ..Default::default()
        }
    }

    /// A column-direction flex container (the default).
    pub fn column() -> Self {
        Self::default()
    }

    /// A stacking container (children placed by `place`).
    pub fn stack() -> Self {
        Self {
            display: Display::Stack,
            ..Default::default()
        }
    }

    pub fn pad(mut self, v: f32) -> Self {
        self.padding = Edges::all(v);
        self
    }
    pub fn pad_x(mut self, v: f32) -> Self {
        self.padding.left = v;
        self.padding.right = v;
        self
    }
    pub fn pad_y(mut self, v: f32) -> Self {
        self.padding.top = v;
        self.padding.bottom = v;
        self
    }
    pub fn padding(mut self, e: Edges<f32>) -> Self {
        self.padding = e;
        self
    }
    pub fn margin(mut self, e: Edges<f32>) -> Self {
        self.margin = e;
        self
    }
    pub fn gap(mut self, v: f32) -> Self {
        self.gap = v;
        self
    }
    pub fn cross_gap(mut self, v: f32) -> Self {
        self.cross_gap = v;
        self
    }
    pub fn wrap(mut self, w: bool) -> Self {
        self.wrap = w;
        self
    }
    pub fn justify(mut self, j: Justify) -> Self {
        self.justify = j;
        self
    }
    /// align_items — the line-level cross alignment for children.
    pub fn align(mut self, a: Align) -> Self {
        self.align_items = a;
        self
    }
    pub fn align_self(mut self, a: Align) -> Self {
        self.align_self = Some(a);
        self
    }
    pub fn grow(mut self, g: f32) -> Self {
        self.grow = g;
        self
    }
    pub fn shrink(mut self, s: f32) -> Self {
        self.shrink = s;
        self
    }
    pub fn basis(mut self, px: f32) -> Self {
        self.basis = Some(px);
        self
    }
    pub fn w(mut self, s: Size) -> Self {
        self.width = s;
        self
    }
    pub fn h(mut self, s: Size) -> Self {
        self.height = s;
        self
    }
    pub fn w_px(mut self, f: f32) -> Self {
        self.width = Size::Px(f);
        self
    }
    pub fn h_px(mut self, f: f32) -> Self {
        self.height = Size::Px(f);
        self
    }
    pub fn size_px(mut self, w: f32, h: f32) -> Self {
        self.width = Size::Px(w);
        self.height = Size::Px(h);
        self
    }
    /// Width and height Fill.
    pub fn fill(mut self) -> Self {
        self.width = Size::Fill;
        self.height = Size::Fill;
        self
    }
    pub fn fill_w(mut self) -> Self {
        self.width = Size::Fill;
        self
    }
    pub fn fill_h(mut self) -> Self {
        self.height = Size::Fill;
        self
    }
    pub fn min_w(mut self, v: f32) -> Self {
        self.min_w = Some(v);
        self
    }
    pub fn max_w(mut self, v: f32) -> Self {
        self.max_w = Some(v);
        self
    }
    pub fn min_h(mut self, v: f32) -> Self {
        self.min_h = Some(v);
        self
    }
    pub fn max_h(mut self, v: f32) -> Self {
        self.max_h = Some(v);
        self
    }
    pub fn aspect(mut self, r: f32) -> Self {
        self.aspect_ratio = Some(r);
        self
    }
    pub fn clip(mut self) -> Self {
        self.overflow = Overflow::Clip;
        self
    }
    pub fn scroll_x(mut self) -> Self {
        self.overflow = Overflow::Scroll { x: true, y: false };
        self
    }
    pub fn scroll_y(mut self) -> Self {
        self.overflow = Overflow::Scroll { x: false, y: true };
        self
    }
    pub fn scroll(mut self) -> Self {
        self.overflow = Overflow::Scroll { x: true, y: true };
        self
    }
    pub fn absolute(mut self, e: Edges<Option<f32>>) -> Self {
        self.position = Position::Absolute(e);
        self
    }
    pub fn place(mut self, x: AlignX, y: AlignY) -> Self {
        self.place = (x, y);
        self
    }
    /// Measured-leaf identity (FNV-1a of the string, the same hash family
    /// as widget ids).
    pub fn key(mut self, s: &str) -> Self {
        self.key = Some(crate::ui::widgets::fnv_push(
            crate::ui::widgets::FNV_OFFSET,
            s,
        ));
        self
    }
}
