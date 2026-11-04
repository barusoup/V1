//! View node payloads: [`Paint`] container chrome, [`PaintOp`] leaf
//! ops, and the [`Widget`] trait adapters implement.

use crate::ui::font::FontId;
use crate::ui::layout::Style;
use crate::ui::theme::Theme;
use crate::ui::tokens::{ControlSize, Elevation, TypeRole};
use crate::ui::widgets::Ui;
use crate::ui::widgets::headless::State;
use crate::ui::widgets::{ButtonStyle, LabelStyle, Response, SegmentedStyle, SliderStyle};
use crate::ui::{Color, Rect};

/// Container chrome drawn around children.
#[derive(Clone, Copy, Debug, Default)]
pub struct Paint {
    pub bg: Option<Color>,
    pub border: Option<(f32, Color)>,
    pub radius: f32,
    pub elevation: Elevation,
    /// Clip children to the content box (in addition to scroll axes).
    pub clip: bool,
}

impl Paint {
    pub fn new() -> Self {
        Self::default()
    }

    /// The panel look.
    pub fn surface(theme: &Theme) -> Self {
        Self {
            bg: Some(theme.g1),
            border: Some((1.0, theme.border)),
            radius: theme.tokens.radius(crate::ui::tokens::Radius::Lg),
            elevation: Elevation::Flat,
            clip: true,
        }
    }

    /// No chrome.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn bg(mut self, c: Color) -> Self {
        self.bg = Some(c);
        self
    }
    pub fn border(mut self, w: f32, c: Color) -> Self {
        self.border = Some((w, c));
        self
    }
    pub fn radius(mut self, r: f32) -> Self {
        self.radius = r;
        self
    }
    pub fn elevation(mut self, e: Elevation) -> Self {
        self.elevation = e;
        self
    }
    pub fn clip(mut self, c: bool) -> Self {
        self.clip = c;
        self
    }
}

/// A leaf's intrinsic size report.
#[derive(Clone, Debug, Default)]
pub struct Intrinsic {
    pub w: f32,
    pub h: f32,
    pub baseline: Option<f32>,
    /// Set when the leaf is measured height-for-width (wrapped text).
    pub wrap: Option<TextWrap>,
}

/// What a wrapping leaf needs for height-for-width measuring: the text
/// lives in the view's string arena (`text` = byte range).
#[derive(Clone, Debug)]
pub struct TextWrap {
    pub text: (u32, u32),
    pub font: FontId,
    pub size: f32,
    /// Resolved line height (px); 0 = font natural.
    pub line_height: f32,
    pub baseline_first: f32,
}

/// Measurement context passed to [`Widget::measure`].
pub struct MeasureCx<'a> {
    pub(crate) ui: &'a mut Ui,
}

impl MeasureCx<'_> {
    /// (w, h) of `text` at `size`, unwrapped.
    pub fn measure_text(&mut self, text: &str, font: FontId, size: f32) -> (f32, f32) {
        self.ui.measure(text, font, size)
    }

    pub fn tokens(&self) -> &crate::ui::tokens::Tokens {
        &self.ui.theme().tokens
    }

    /// Themed label style for a type role.
    pub fn type_style(&self, role: TypeRole) -> LabelStyle {
        self.ui.type_style(role)
    }

    /// Cap height (logical px) of a face at `size`.
    pub fn cap_height(&self, font: FontId, size: f32) -> f32 {
        self.ui.fonts().get(font).cap_height(size)
    }

    /// Control metrics for a size tier.
    pub fn control(&self, size: ControlSize) -> crate::ui::tokens::ControlMetrics {
        self.ui.theme().tokens.control(size)
    }

    pub fn scale(&self) -> f32 {
        self.ui.scale()
    }
}

/// A recorded or immediate paint operation (opaque).
pub struct PaintOp(pub(crate) NodeOp);

impl PaintOp {
    /// No painted output (the behaviour already ran).
    pub fn none() -> Self {
        PaintOp(NodeOp::None)
    }

    /// Caller-drawn content at the solved rect.
    pub fn custom(f: impl FnOnce(&mut Ui, Rect) + 'static) -> Self {
        PaintOp(NodeOp::Custom(Box::new(f)))
    }

    /// Run `f` immediately against `prev` with the draw list swapped to
    /// the view's scratch list, inheriting the input clip and layer;
    /// at paint the recorded geometry splices
    /// into the main list translated by (solved − prev).
    pub fn record(
        ui: &mut Ui,
        prev: Rect,
        f: impl FnOnce(&mut Ui) -> Response,
    ) -> (Response, PaintOp) {
        let start = ui.view.scratch.cursor();
        let clip = ui.draw.clip_rect();
        let layer = ui.draw.layer();
        std::mem::swap(&mut ui.draw, &mut ui.view.scratch);
        let scratch_layer = ui.draw.set_layer(layer);
        if let Some(clip) = clip {
            ui.draw.push_clip(clip);
        }
        let resp = f(ui);
        if clip.is_some() {
            ui.draw.pop_clip();
        }
        ui.draw.set_layer(scratch_layer);
        std::mem::swap(&mut ui.draw, &mut ui.view.scratch);
        let end = ui.view.scratch.cursor();
        (resp, PaintOp(NodeOp::Record { start, end, prev }))
    }
}

/// A declarative widget: measures itself, runs its behaviour on the
/// previous frame's rect, and returns a [`PaintOp`] for this frame's
/// solved rect.
pub trait Widget {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic;
    /// L1 immediate path (own rect).
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response
    where
        Self: Sized;
    /// The view path: run the behaviour, return a paint op.
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp)
    where
        Self: Sized,
    {
        (self.ui(ui, rect), PaintOp::none())
    }
    /// Leaf layout style (Fit both axes unless the widget wants Fill).
    fn layout_style(&self) -> Style {
        Style::default()
    }
    /// Widget label → id for prev-rect lookup / `rect_of`.
    fn label(&self) -> Option<&str> {
        None
    }
}

/// A boxed slot painter (exact solved rect, caller-drawn).
type SlotFn = Box<dyn FnOnce(&mut Ui, Rect)>;

/// Per-node paint payload (parallel to tree nodes).
pub(crate) enum NodeOp {
    None,
    Container(Paint),
    Custom(SlotFn),
    /// Button-like: style + live state + label text range into the arena.
    Button {
        behavior_id: u64,
        style: ButtonStyle,
        state: State,
        label: (u32, u32),
        icon: Option<(crate::ui::icons::Icon, u32)>,
    },
    IconButton {
        behavior_id: u64,
        style: ButtonStyle,
        state: State,
        icon: crate::ui::icons::Icon,
        icon_size: u32,
        on: bool,
        badge: Option<(u32, u32)>,
    },
    Toggle {
        behavior_id: u64,
        style: ButtonStyle,
        state: State,
        label: (u32, u32),
        kind: ToggleKind,
    },
    Slider {
        behavior_id: u64,
        style: SliderStyle,
        state: State,
        label: (u32, u32),
    },
    Text {
        style: LabelStyle,
        text: (u32, u32),
    },
    Segmented {
        style: SegmentedStyle,
        at: f32,
        /// Ranges into `ViewState::seg_items` (label/state/near); the
        /// cells are recomputed from the solved rect at paint.
        items: (u32, u32),
    },
    Icon {
        icon: crate::ui::icons::Icon,
        size: u32,
        tint: Color,
    },
    Separator {
        color: Color,
    },
    Progress {
        fill: Color,
        value: f32,
    },
    /// Badge with text in the arena range, flat colors.
    Badge {
        text: (u32, u32),
        bg: crate::ui::Color,
        fg: crate::ui::Color,
        size: f32,
    },
    /// Geometry recorded against the previous frame's rect: the scratch
    /// list's `start..end` cursor range.
    Record {
        start: (usize, usize, usize),
        end: (usize, usize, usize),
        prev: Rect,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum ToggleKind {
    Switch,
    Checkbox,
    Radio,
}

/// A registered measure request for a `Leaf::Measured` node.
pub(crate) enum MeasureReq {
    Text(TextWrap),
}
