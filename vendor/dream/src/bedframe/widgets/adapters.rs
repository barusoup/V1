//! Declarative widget adapters for the view layer ([`crate::ui::view`]).
//! Each adapter's `ui()` calls the L1 widget, so both paths draw the
//! same widget.

use super::Ui;
use super::headless;
use crate::ui::font::FontId;
use crate::ui::icons::Icon;
use crate::ui::layout::Style;
use crate::ui::tokens::{ControlSize, TypeRole};
use crate::ui::view::{Intrinsic, MeasureCx, PaintOp, Widget};
use crate::ui::widgets::Response;
use crate::ui::{Color, Rect};

/// Static text.
pub struct Text<'a> {
    text: &'a str,
    role: TypeRole,
    color: Option<Color>,
    wrap: bool,
    align: Option<crate::ui::font::Align>,
}

impl<'a> Text<'a> {
    pub fn new(t: &'a str) -> Self {
        Self {
            text: t,
            role: TypeRole::Body,
            color: None,
            wrap: false,
            align: None,
        }
    }
    pub fn role(mut self, r: TypeRole) -> Self {
        self.role = r;
        self
    }
    pub fn color(mut self, c: Color) -> Self {
        self.color = Some(c);
        self
    }
    pub fn wrap(mut self, w: bool) -> Self {
        self.wrap = w;
        self
    }
    pub fn align(mut self, a: crate::ui::font::Align) -> Self {
        self.align = Some(a);
        self
    }
}

impl Widget for Text<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        let st = cx.ui.type_style(self.role);
        let (w, h) = cx.ui.measure(self.text, st.font, st.size);
        let baseline = h * 0.5 + cx.cap_height(st.font, st.size) * 0.5;
        if self.wrap {
            let lo = cx.ui.view.arena.len() as u32;
            cx.ui.view.arena.push_str(self.text);
            let hi = cx.ui.view.arena.len() as u32;
            Intrinsic {
                w,
                h,
                baseline: Some(baseline),
                wrap: Some(crate::ui::view::TextWrap {
                    text: (lo, hi),
                    font: st.font,
                    size: st.size,
                    line_height: st.line_height,
                    baseline_first: baseline,
                }),
            }
        } else {
            Intrinsic {
                w,
                h,
                baseline: Some(baseline),
                wrap: None,
            }
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let mut st = ui.type_style(self.role);
        if let Some(c) = self.color {
            st.color = c;
        }
        if let Some(a) = self.align {
            st.align = a;
        }
        ui.label(rect, self.text, st);
        Response::plain(rect, ui, u64::MAX, false)
    }
    fn interact(self, ui: &mut Ui, _rect: Rect) -> (Response, PaintOp) {
        let mut st = ui.type_style(self.role);
        if let Some(c) = self.color {
            st.color = c;
        }
        if let Some(a) = self.align {
            st.align = a;
        }
        let lo = ui.view.arena.len() as u32;
        ui.view.arena.push_str(self.text);
        let hi = ui.view.arena.len() as u32;
        (
            Response::plain(Rect::new(0.0, 0.0, 0.0, 0.0), ui, u64::MAX, false),
            PaintOp(crate::ui::view::NodeOp::Text {
                style: st,
                text: (lo, hi),
            }),
        )
    }
}

/// Push button adapter.
pub struct Button<'a> {
    label: &'a str,
    pub(crate) style: Option<crate::ui::widgets::ButtonStyle>,
    size: ControlSize,
    disabled: bool,
    icon: Option<(Icon, u32)>,
    variant: ButtonVariant,
    full_width: bool,
}

/// Button variants (visual only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ButtonVariant {
    Primary,
    #[default]
    Secondary,
    Outline,
    Ghost,
    Danger,
    Link,
}

impl<'a> Button<'a> {
    pub fn new(label: &'a str) -> Self {
        Self {
            label,
            style: None,
            size: ControlSize::Md,
            disabled: false,
            icon: None,
            variant: ButtonVariant::Secondary,
            full_width: false,
        }
    }
    pub fn variant(mut self, v: ButtonVariant) -> Self {
        self.variant = v;
        self
    }
    pub fn icon(mut self, i: Icon) -> Self {
        self.icon = Some((i, 16));
        self
    }
    pub fn size(mut self, s: ControlSize) -> Self {
        self.size = s;
        self
    }
    pub fn disabled(mut self, d: bool) -> Self {
        self.disabled = d;
        self
    }
    pub fn full_width(mut self) -> Self {
        self.full_width = true;
        self
    }
}

impl Widget for Button<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        let c = cx.control(self.size);
        let st = self
            .style
            .unwrap_or_else(|| cx.ui.theme().button_variant(FontId(0), self.variant));
        let (tw, _) = cx.ui.measure(self.label, st.font, st.size);
        let icon_w = self.icon.map_or(0.0, |(_, s)| s as f32 + c.gap);
        Intrinsic {
            w: c.pad_x * 2.0 + tw + icon_w,
            h: c.height,
            baseline: Some(c.height * 0.5 + cx.cap_height(st.font, st.size) * 0.5),
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let mut style = self
            .style
            .unwrap_or_else(|| ui.theme().button_variant(FontId(0), self.variant));
        style.disabled |= self.disabled;
        match self.icon {
            Some((i, s)) => ui.button_with_icon(self.label, rect, i, s, style),
            None => ui.button(self.label, rect, style),
        }
    }
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
        let mut style = self
            .style
            .unwrap_or_else(|| ui.theme().button_variant(FontId(0), self.variant));
        style.disabled |= self.disabled;
        let (resp, state) = headless::ButtonBehavior::new(self.label)
            .enabled(!style.disabled)
            .update(ui, rect);
        let lo = ui.view.arena.len() as u32;
        ui.view.arena.push_str(self.label);
        let hi = ui.view.arena.len() as u32;
        (
            resp,
            PaintOp(crate::ui::view::NodeOp::Button {
                behavior_id: ui.id(self.label),
                style,
                state,
                label: (lo, hi),
                icon: self.icon,
            }),
        )
    }
    fn layout_style(&self) -> Style {
        if self.full_width {
            Style::default().fill_w()
        } else {
            Style::default()
        }
    }
    fn label(&self) -> Option<&str> {
        Some(self.label)
    }
}

/// Toggle/checkbox share the headless toggle behaviour.
macro_rules! toggler {
    ($name:ident, $kind:ident, $l1:ident) => {
        pub struct $name<'a> {
            label: &'a str,
            value: &'a mut bool,
            disabled: bool,
        }
        impl<'a> $name<'a> {
            pub fn new(label: &'a str, value: &'a mut bool) -> Self {
                Self {
                    label,
                    value,
                    disabled: false,
                }
            }
            pub fn disabled(mut self, d: bool) -> Self {
                self.disabled = d;
                self
            }
        }
        impl Widget for $name<'_> {
            fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
                let c = cx.control(ControlSize::Md);
                Intrinsic {
                    w: c.icon
                        + 2.0
                        + c.gap
                        + cx.ui.measure(self.label, FontId(0), c.text_size).0
                        + 4.0,
                    h: c.height,
                    baseline: None,
                    wrap: None,
                }
            }
            fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
                let mut st = ui.theme().button(FontId(0));
                st.disabled |= self.disabled;
                ui.$l1(self.label, rect, self.value, st)
            }
            fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
                let mut st = ui.theme().button(FontId(0));
                st.disabled |= self.disabled;
                let (resp, state) = headless::ToggleBehavior::new(self.label, self.value)
                    .enabled(!st.disabled)
                    .duration(ui.theme().tokens.motion.toggle)
                    .anim_suffix(if stringify!($kind) == "Checkbox" {
                        "##chk"
                    } else {
                        "##switch"
                    })
                    .update(ui, rect);
                let lo = ui.view.arena.len() as u32;
                ui.view.arena.push_str(self.label);
                let hi = ui.view.arena.len() as u32;
                (
                    resp,
                    PaintOp(crate::ui::view::NodeOp::Toggle {
                        behavior_id: ui.id(self.label),
                        style: st,
                        state,
                        label: (lo, hi),
                        kind: crate::ui::view::ToggleKind::$kind,
                    }),
                )
            }
            fn label(&self) -> Option<&str> {
                Some(self.label)
            }
        }
    };
}
toggler!(Switch, Switch, switch);
toggler!(Checkbox, Checkbox, checkbox);
pub type Toggle<'a> = Switch<'a>;

/// Slider adapter (Fill width, preferred 160).
pub struct Slider<'a> {
    label: &'a str,
    value: &'a mut f32,
    range: std::ops::RangeInclusive<f32>,
}
impl<'a> Slider<'a> {
    pub fn new(label: &'a str, value: &'a mut f32, range: std::ops::RangeInclusive<f32>) -> Self {
        Self {
            label,
            value,
            range,
        }
    }
}
impl Widget for Slider<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        Intrinsic {
            w: 160.0,
            h: cx.control(ControlSize::Md).height,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let st = ui.theme().slider(FontId(0));
        ui.slider(self.label, rect, self.value, self.range.clone(), st)
    }
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
        let style = ui.theme().slider(FontId(0));
        let (resp, state) =
            headless::SliderBehavior::new(self.label, self.value, self.range.clone())
                .update(ui, rect);
        let lo = ui.view.arena.len() as u32;
        ui.view.arena.push_str(self.label);
        let hi = ui.view.arena.len() as u32;
        (
            resp,
            PaintOp(crate::ui::view::NodeOp::Slider {
                behavior_id: ui.id(self.label),
                style,
                state,
                label: (lo, hi),
            }),
        )
    }
    fn layout_style(&self) -> Style {
        Style::default().fill_w()
    }
    fn label(&self) -> Option<&str> {
        Some(self.label)
    }
}

/// Vector icon leaf.
pub struct IconView {
    icon: Icon,
    size: u32,
    tint: Option<Color>,
}
impl IconView {
    pub fn new(icon: Icon) -> Self {
        Self {
            icon,
            size: 16,
            tint: None,
        }
    }
    pub fn size(mut self, s: u32) -> Self {
        self.size = s;
        self
    }
    pub fn tint(mut self, c: Color) -> Self {
        self.tint = Some(c);
        self
    }
}
impl Widget for IconView {
    fn measure(&self, _cx: &mut MeasureCx<'_>) -> Intrinsic {
        Intrinsic {
            w: self.size as f32,
            h: self.size as f32,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let tint = self.tint.unwrap_or(ui.theme().fg);
        ui.icon(rect, self.icon, self.size, tint);
        Response::plain(rect, ui, u64::MAX, false)
    }
    fn interact(self, ui: &mut Ui, _rect: Rect) -> (Response, PaintOp) {
        let tint = self.tint.unwrap_or(ui.theme().fg);
        (
            Response::plain(Rect::new(0.0, 0.0, 0.0, 0.0), ui, u64::MAX, false),
            PaintOp(crate::ui::view::NodeOp::Icon {
                icon: self.icon,
                size: self.size,
                tint,
            }),
        )
    }
}

/// Hairline rule (Fill on its long axis).
pub struct Separator {
    vertical: bool,
}
impl Separator {
    pub fn horizontal() -> Self {
        Self { vertical: false }
    }
    pub fn vertical() -> Self {
        Self { vertical: true }
    }
}
impl Widget for Separator {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        let h = crate::ui::layout::hairline(cx.scale());
        if self.vertical {
            Intrinsic {
                w: h,
                h: 0.0,
                baseline: None,
                wrap: None,
            }
        } else {
            Intrinsic {
                w: 0.0,
                h,
                baseline: None,
                wrap: None,
            }
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        ui.draw.rect(rect, ui.theme().border);
        Response::plain(rect, ui, u64::MAX, false)
    }
    fn interact(self, ui: &mut Ui, _rect: Rect) -> (Response, PaintOp) {
        (
            Response::plain(Rect::new(0.0, 0.0, 0.0, 0.0), ui, u64::MAX, false),
            PaintOp(crate::ui::view::NodeOp::Separator {
                color: ui.theme().border,
            }),
        )
    }
    fn layout_style(&self) -> Style {
        if self.vertical {
            Style::default().fill_h()
        } else {
            Style::default().fill_w()
        }
    }
}

/// Progress bar (Fill width, h 6).
pub struct Progress {
    value: f32,
}
impl Progress {
    pub fn new(v: f32) -> Self {
        Self { value: v }
    }
}
impl Widget for Progress {
    fn measure(&self, _cx: &mut MeasureCx<'_>) -> Intrinsic {
        Intrinsic {
            w: 0.0,
            h: 6.0,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        ui.progress(rect, self.value, ui.theme().accent);
        Response::plain(rect, ui, u64::MAX, false)
    }
    fn interact(self, ui: &mut Ui, _rect: Rect) -> (Response, PaintOp) {
        (
            Response::plain(Rect::new(0.0, 0.0, 0.0, 0.0), ui, u64::MAX, false),
            PaintOp(crate::ui::view::NodeOp::Progress {
                fill: ui.theme().accent,
                value: self.value,
            }),
        )
    }
    fn layout_style(&self) -> Style {
        Style::default().fill_w()
    }
}

/// Icon button adapter.
pub struct IconButton<'a> {
    label: &'a str,
    icon: Icon,
    size: ControlSize,
    on: bool,
    badge: Option<&'a str>,
    disabled: bool,
}

impl<'a> IconButton<'a> {
    pub fn new(label: &'a str, icon: Icon) -> Self {
        Self {
            label,
            icon,
            size: ControlSize::Md,
            on: false,
            badge: None,
            disabled: false,
        }
    }
    pub fn on(mut self, on: bool) -> Self {
        self.on = on;
        self
    }
    pub fn badge(mut self, b: &'a str) -> Self {
        self.badge = Some(b);
        self
    }
    pub fn size(mut self, s: ControlSize) -> Self {
        self.size = s;
        self
    }
    pub fn disabled(mut self, d: bool) -> Self {
        self.disabled = d;
        self
    }
}

impl Widget for IconButton<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        let h = cx.control(self.size).height;
        Intrinsic {
            w: h,
            h,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let mut st = ui.theme().icon_button(FontId(0));
        st.disabled |= self.disabled;
        ui.icon_button(self.label, rect, self.icon, st, self.on, self.badge, 16)
    }
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
        let mut st = ui.theme().icon_button(FontId(0));
        st.disabled |= self.disabled;
        let (resp, state) = headless::ButtonBehavior::new(self.label)
            .enabled(!st.disabled)
            .update(ui, rect);
        let badge = self.badge.map(|b| {
            let lo = ui.view.arena.len() as u32;
            ui.view.arena.push_str(b);
            (lo, ui.view.arena.len() as u32)
        });
        (
            resp,
            PaintOp(crate::ui::view::NodeOp::IconButton {
                behavior_id: ui.id(self.label),
                style: st,
                state,
                icon: self.icon,
                icon_size: (rect.h.min(18.0) as u32).max(10),
                on: self.on,
                badge,
            }),
        )
    }
    fn label(&self) -> Option<&str> {
        Some(self.label)
    }
}

/// Latched-state radio dot adapter (click reports, caller latches).
pub struct Radio<'a> {
    label: &'a str,
    selected: bool,
    disabled: bool,
    _m: std::marker::PhantomData<&'a ()>,
}

impl<'a> Radio<'a> {
    pub fn new(label: &'a str, selected: bool) -> Self {
        Self {
            label,
            selected,
            disabled: false,
            _m: std::marker::PhantomData,
        }
    }
    pub fn disabled(mut self, d: bool) -> Self {
        self.disabled = d;
        self
    }
}

impl Widget for Radio<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        let c = cx.control(ControlSize::Md);
        Intrinsic {
            w: c.icon + 2.0 + c.gap + cx.ui.measure(self.label, FontId(0), c.text_size).0 + 4.0,
            h: c.height,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let mut st = ui.theme().button(FontId(0));
        st.disabled |= self.disabled;
        ui.radio(self.label, rect, self.selected, st)
    }
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
        let mut st = ui.theme().button(FontId(0));
        st.disabled |= self.disabled;
        let (resp, state) = headless::RadioBehavior::new(self.label, self.selected)
            .enabled(!st.disabled)
            .duration(ui.theme().tokens.motion.check)
            .anim_suffix("##rad")
            .update(ui, rect);
        let lo = ui.view.arena.len() as u32;
        ui.view.arena.push_str(self.label);
        let hi = ui.view.arena.len() as u32;
        (
            resp,
            PaintOp(crate::ui::view::NodeOp::Toggle {
                behavior_id: ui.id(self.label),
                style: st,
                state,
                label: (lo, hi),
                kind: crate::ui::view::ToggleKind::Radio,
            }),
        )
    }
    fn label(&self) -> Option<&str> {
        Some(self.label)
    }
}

/// Segmented selector adapter.
pub struct Segmented<'a> {
    label: &'a str,
    options: &'a [&'a str],
    selected: &'a mut usize,
    style: Option<crate::ui::widgets::SegmentedStyle>,
    track: Option<Color>,
    pill: Option<Color>,
    text: Option<Color>,
    text_selected: Option<Color>,
}

impl<'a> Segmented<'a> {
    pub fn new(label: &'a str, options: &'a [&'a str], selected: &'a mut usize) -> Self {
        Self {
            label,
            options,
            selected,
            style: None,
            track: None,
            pill: None,
            text: None,
            text_selected: None,
        }
    }
    pub fn style(mut self, s: crate::ui::widgets::SegmentedStyle) -> Self {
        self.style = Some(s);
        self
    }
    /// Track fill override (applied onto the ambient theme style).
    pub fn track_bg(mut self, c: Color) -> Self {
        self.track = Some(c);
        self
    }
    pub fn pill_bg(mut self, c: Color) -> Self {
        self.pill = Some(c);
        self
    }
    pub fn text(mut self, c: Color) -> Self {
        self.text = Some(c);
        self
    }
    pub fn text_selected(mut self, c: Color) -> Self {
        self.text_selected = Some(c);
        self
    }
    fn resolve(&self, ui: &Ui) -> crate::ui::widgets::SegmentedStyle {
        let mut st = self
            .style
            .unwrap_or_else(|| ui.theme().segmented(FontId(0)));
        if let Some(c) = self.track {
            st.track = c;
        }
        if let Some(c) = self.pill {
            st.pill = c;
        }
        if let Some(c) = self.text {
            st.text = c;
        }
        if let Some(c) = self.text_selected {
            st.text_selected = c;
        }
        st
    }
}

impl Widget for Segmented<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        let c = cx.control(ControlSize::Md);
        Intrinsic {
            w: 120.0,
            h: c.height,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let st = self.resolve(ui);
        let picked = ui.segmented_styled(self.label, rect, self.options, self.selected, &st);
        Response::plain(rect, ui, ui.id(self.label), picked.is_some())
    }
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
        let style = self.resolve(ui);
        let id = ui.id(self.label);
        let mut resp = Response::plain(rect, ui, id, false);
        if self.options.is_empty() {
            return (resp, PaintOp::none());
        }
        // Same id scope as the L1 path: controls with the same label
        // animate their own pill.
        ui.push_id(self.label);
        *self.selected = (*self.selected).min(self.options.len() - 1);
        let at = ui.anim_f32_seeded(
            "##pill",
            *self.selected as f32,
            style.glide,
            crate::ui::tween::Ease::CubicOut,
        );
        let mut cells = std::mem::take(&mut ui.seg_cells);
        crate::ui::layout::divide_into(
            &mut cells,
            rect.x,
            rect.w,
            self.options.len(),
            0.0,
            ui.scale(),
        );
        let mut items = std::mem::take(&mut ui.seg_scratch);
        let picked = headless::SegmentedBehavior::new(self.label, self.options)
            .enabled(!style.button.disabled)
            .update_with_cells(ui, rect, self.selected, &cells, style.inset, &mut items);
        ui.seg_scratch = items;
        let items = &ui.seg_scratch;
        ui.seg_cells = cells;
        let i_lo = ui.view.seg_items.len() as u32;
        for (i, (_cell, _resp, _st)) in items.iter().enumerate() {
            let lo = ui.view.arena.len() as u32;
            ui.view.arena.push_str(self.options[i]);
            // seg_id hashes under the pushed label scope — the same id
            // the behaviour used for hit-testing and the focus ring.
            ui.view.seg_items.push(crate::ui::view::SegItem {
                text: (lo, ui.view.arena.len() as u32),
                id: crate::ui::widgets::headless::seg_id(ui, self.options[i], i),
                near: (1.0 - (at - i as f32).abs()).clamp(0.0, 1.0),
            });
        }
        let i_hi = ui.view.seg_items.len() as u32;
        ui.pop_id();
        resp.clicked = picked.is_some();
        (
            resp,
            PaintOp(crate::ui::view::NodeOp::Segmented {
                style,
                at,
                items: (i_lo, i_hi),
            }),
        )
    }
    fn label(&self) -> Option<&str> {
        Some(self.label)
    }
}

/// Drag-field adapter (record path).
pub struct DragField<'a> {
    label: &'a str,
    value: &'a mut f32,
    range: std::ops::RangeInclusive<f32>,
}

impl<'a> DragField<'a> {
    pub fn new(label: &'a str, value: &'a mut f32, range: std::ops::RangeInclusive<f32>) -> Self {
        Self {
            label,
            value,
            range,
        }
    }
}

impl Widget for DragField<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        Intrinsic {
            w: 120.0,
            h: cx.control(ControlSize::Md).height,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        ui.drag_field(self.label, rect, self.value, self.range.clone(), FontId(0))
    }
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
        let label = self.label;
        let value = self.value;
        let range = self.range.clone();
        PaintOp::record(ui, rect, |ui| {
            ui.drag_field(label, rect, value, range, FontId(0))
        })
    }
    fn layout_style(&self) -> Style {
        Style::default().fill_w()
    }
    fn label(&self) -> Option<&str> {
        Some(self.label)
    }
}

/// Text-field adapter (record path — behaviour/paint split deferred).
pub struct TextField<'a> {
    label: &'a str,
    text: &'a mut String,
    placeholder: &'a str,
    password: bool,
}

impl<'a> TextField<'a> {
    pub fn new(label: &'a str, text: &'a mut String) -> Self {
        Self {
            label,
            text,
            placeholder: "",
            password: false,
        }
    }
    pub fn placeholder(mut self, p: &'a str) -> Self {
        self.placeholder = p;
        self
    }
    pub fn password(mut self, p: bool) -> Self {
        self.password = p;
        self
    }
}

impl Widget for TextField<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        Intrinsic {
            w: 200.0,
            h: cx.control(ControlSize::Md).height,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let mut st = crate::ui::widgets::TextFieldStyle::from(ui.theme().button(FontId(0)));
        st.password = self.password;
        ui.text_field_with_placeholder(self.label, rect, self.text, self.placeholder, st)
    }
    fn interact(self, ui: &mut Ui, rect: Rect) -> (Response, PaintOp) {
        let label = self.label;
        let text = self.text;
        let ph = self.placeholder;
        let pw = self.password;
        PaintOp::record(ui, rect, move |ui| {
            let mut st = crate::ui::widgets::TextFieldStyle::from(ui.theme().button(FontId(0)));
            st.password = pw;
            ui.text_field_with_placeholder(label, rect, text, ph, st)
        })
    }
    fn layout_style(&self) -> Style {
        Style::default().fill_w()
    }
    fn label(&self) -> Option<&str> {
        Some(self.label)
    }
}

/// Inline badge/pill label (record path — no interaction).
pub struct Badge<'a> {
    text: &'a str,
    bg: Option<Color>,
    fg: Option<Color>,
}

impl<'a> Badge<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            bg: None,
            fg: None,
        }
    }
    pub fn bg(mut self, c: Color) -> Self {
        self.bg = Some(c);
        self
    }
    pub fn fg(mut self, c: Color) -> Self {
        self.fg = Some(c);
        self
    }
}

impl Widget for Badge<'_> {
    fn measure(&self, cx: &mut MeasureCx<'_>) -> Intrinsic {
        let size = cx.ui.text_metrics().badge;
        let (w, h) = cx.measure_text(self.text, FontId(0), size);
        Intrinsic {
            w: w + 12.0,
            h: h + 6.0,
            baseline: None,
            wrap: None,
        }
    }
    fn ui(self, ui: &mut Ui, rect: Rect) -> Response {
        let bg = self.bg.unwrap_or(ui.theme().accent_dim);
        let fg = self.fg.unwrap_or(ui.theme().accent_text);
        ui.badge(rect, self.text, bg, fg, FontId(0), ui.text_metrics().badge);
        Response::plain(rect, ui, u64::MAX, false)
    }
    fn interact(self, ui: &mut Ui, _rect: Rect) -> (Response, PaintOp) {
        let bg = self.bg.unwrap_or(ui.theme().accent_dim);
        let fg = self.fg.unwrap_or(ui.theme().accent_text);
        let size = ui.text_metrics().badge;
        let lo = ui.view.arena.len() as u32;
        ui.view.arena.push_str(self.text);
        let hi = ui.view.arena.len() as u32;
        (
            Response::plain(Rect::new(0.0, 0.0, 0.0, 0.0), ui, u64::MAX, false),
            PaintOp(crate::ui::view::NodeOp::Badge {
                text: (lo, hi),
                bg,
                fg,
                size,
            }),
        )
    }
}
