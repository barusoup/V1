//! Devtools overlays for the L3 view layer: layout outlines and a
//! node inspector. Enable per frame with [`Ui::set_devtools`]:
//!
//! ```ignore
//! ui.set_devtools(DevtoolsMode::Outlines);
//! ui.layout_root(area, Style::column(), |ui| { /* … */ });
//! ```
//!
//! `Outlines` draws every node's border box plus padding/gap bands.
//! `Inspect` additionally highlights the deepest node under the pointer
//! (pinned on click) with a small info card.

use crate::ui::layout::{Direction, Display, LayoutTree, NodeId, Size};
use crate::ui::tokens::TypeRole;
use crate::ui::widgets::{LAYER_DEVTOOLS, Ui};
use crate::ui::{Color, Rect};
use std::collections::HashMap;

/// Devtools overlay mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DevtoolsMode {
    /// No overlay.
    #[default]
    Off,
    /// Per-node border outlines + padding/gap bands.
    Outlines,
    /// Outlines plus pointer inspection with pin-on-click.
    Inspect,
}

/// Per-node record kept each frame for the inspector.
#[derive(Clone, Copy, Debug)]
pub struct NodeRec {
    pub id: u64,
    pub rect: Rect,
    /// Padding (l, t, r, b).
    pub padding: (f32, f32, f32, f32),
    pub intrinsic_w: f32,
    pub intrinsic_h: f32,
    pub baseline: Option<f32>,
    pub depth: u16,
    pub display: u8,
    pub direction: u8,
    pub width: u8,
    pub height: u8,
    pub grow: f32,
    pub justify: u8,
    pub align: u8,
}

/// Inspector state: the pinned node id (when Inspect).
#[derive(Default)]
pub(crate) struct InspectState {
    pub pinned: Option<u64>,
    /// Node records of the last painted frame (`node index → rec`).
    pub records: Vec<NodeRec>,
}

const PALETTE: [Color; 6] = [
    Color::rgb(0x3b, 0xd9, 0xff),
    Color::rgb(0xff, 0xb8, 0x6c),
    Color::rgb(0xff, 0x7a, 0xa8),
    Color::rgb(0x95, 0xe6, 0x72),
    Color::rgb(0xc2, 0x98, 0xff),
    Color::rgb(0xff, 0xd8, 0x76),
];

fn size_tag(s: Size) -> u8 {
    match s {
        Size::Px(_) => 0,
        Size::Fit => 1,
        Size::Fill => 2,
        Size::Percent(_) => 3,
    }
}

fn size_txt(s: Size) -> String {
    match s {
        Size::Px(v) => format!("{v:.0}px"),
        Size::Fit => "fit".into(),
        Size::Fill => "fill".into(),
        Size::Percent(p) => format!("{p:.0}%"),
    }
}

/// Record one node's inspector entry (called from `layout_root` after
/// solve; `intrinsic` = leaf size pre-grow).
pub(crate) fn record_node(ui: &mut Ui, tree: &LayoutTree, node: NodeId, id: u64, depth: u16) {
    let style = tree.style(node);
    let rect = tree.rect(node);
    let leaf = tree.leaf_size(node);
    ui.view.devtools.records.push(NodeRec {
        id,
        rect,
        padding: (
            style.padding.left,
            style.padding.top,
            style.padding.right,
            style.padding.bottom,
        ),
        intrinsic_w: leaf.0,
        intrinsic_h: leaf.1,
        baseline: tree.baseline(node).map(|b| b - rect.y),
        depth,
        display: matches!(style.display, Display::Stack) as u8,
        direction: matches!(style.direction, Direction::Column) as u8,
        width: size_tag(style.width),
        height: size_tag(style.height),
        grow: style.grow,
        justify: style.justify as u8,
        align: style.align_items as u8,
    });
}

/// Draw outlines for a whole tree (after its paint pass).
pub(crate) fn draw_outlines(ui: &mut Ui, tree: &LayoutTree) {
    ui.draw.set_layer(LAYER_DEVTOOLS);
    // Index rects by node for the gap/parent bands.
    let n = tree.len();
    let mut prev_child_end: HashMap<u32, f32> = HashMap::new();
    for i in 0..n as u32 {
        let node = NodeId(i);
        let rect = tree.rect(node);
        let depth = tree.depth(node);
        let color = PALETTE[(depth as usize) % PALETTE.len()];
        ui.draw.round_ring(
            rect,
            0.0,
            crate::ui::layout::hairline(ui.scale()),
            color.fade(0.85),
        );
        // Padding band (green 15%).
        let st = tree.style(node);
        let (l, t, r, b) = (
            st.padding.left,
            st.padding.top,
            st.padding.right,
            st.padding.bottom,
        );
        let pad = Color::rgb(0x2e, 0xb8, 0x4c).fade(0.15);
        if l > 0.0 {
            ui.draw.rect(Rect::new(rect.x, rect.y, l, rect.h), pad);
        }
        if r > 0.0 {
            ui.draw
                .rect(Rect::new(rect.max().0 - r, rect.y, r, rect.h), pad);
        }
        if t > 0.0 {
            ui.draw.rect(Rect::new(rect.x, rect.y, rect.w, t), pad);
        }
        if b > 0.0 {
            ui.draw
                .rect(Rect::new(rect.x, rect.max().1 - b, rect.w, b), pad);
        }
        // Gap band before this child on the parent's main axis (purple).
        if let Some(p) = tree.parent(node) {
            let pst = tree.style(p);
            let gap = if pst.direction == Direction::Row {
                rect.y
            } else {
                rect.x
            } * 0.0;
            let _ = gap;
            if let Some(&end) = prev_child_end.get(&p.0) {
                let band = if pst.direction == Direction::Row {
                    Rect::new(end, rect.y, rect.x - end, rect.h)
                } else {
                    Rect::new(rect.x, end, rect.w, rect.y - end)
                };
                if band.w > 0.5 && band.h > 0.5 {
                    ui.draw.rect(band, Color::rgb(0x9c, 0x5c, 0xff).fade(0.15));
                }
            }
            let end = if pst.direction == Direction::Row {
                rect.max().0
            } else {
                rect.max().1
            };
            prev_child_end.insert(p.0, end);
        }
    }
    ui.draw.set_layer(0);
}

/// Inspector pass: pick the deepest node under the pointer, outline it,
/// draw its baseline, and show a compact info card. Mouse press pins.
pub(crate) fn draw_inspect(ui: &mut Ui) {
    let mouse = ui.input.mouse;
    // Deepest record containing the pointer.
    let hit = ui
        .view
        .devtools
        .records
        .iter()
        .filter(|r| r.rect.contains(mouse))
        .max_by_key(|r| r.depth)
        .map(|r| r.id);
    // Pinning is owned by devtools_begin (at Ui::begin); here we only
    // pick the record to display: pinned wins, else hovered.
    let _ = hit;
    let rec = ui
        .view
        .devtools
        .records
        .iter()
        .find(|r| Some(r.id) == ui.view.devtools.pinned)
        .or_else(|| {
            ui.view
                .devtools
                .records
                .iter()
                .filter(|r| r.rect.contains(mouse))
                .max_by_key(|r| r.depth)
        })
        .copied();
    let Some(rec) = rec else { return };
    ui.draw.set_layer(LAYER_DEVTOOLS);
    let accent = ui.theme().accent;
    ui.draw.round_ring(rec.rect, 0.0, 2.0, accent);
    if let Some(base) = rec.baseline {
        let y = rec.rect.y + base;
        ui.draw.rect(
            Rect::new(rec.rect.x, y - 0.5, rec.rect.w, 1.0),
            Color::rgb(0x3b, 0xd9, 0xff),
        );
    }
    // Info card.
    let style_txt = format!(
        "{} {} w={} h={} grow={} gap",
        if rec.display == 0 { "flex" } else { "stack" },
        if rec.direction == 0 { "row" } else { "column" },
        size_txt(match rec.width {
            0 => Size::Px(rec.intrinsic_w),
            1 => Size::Fit,
            2 => Size::Fill,
            _ => Size::Percent(50.0),
        }),
        match rec.height {
            0 => "px".to_string(),
            1 => "fit".to_string(),
            2 => "fill".to_string(),
            _ => "%".to_string(),
        },
        rec.grow,
    );
    let card_w = 260.0;
    let card_h = 92.0;
    let card = Rect::new(
        (mouse.0 + 16.0).min(ui.screen.0 - card_w - 4.0),
        (mouse.1 + 16.0).min(ui.screen.1 - card_h - 4.0),
        card_w,
        card_h,
    );
    ui.draw
        .rounded_frame(card, 6.0, 1.0, ui.theme().g1.fade(0.95), ui.theme().border);
    let ts = ui.type_style(TypeRole::Mono);
    let lines = [
        format!("id #{:08x}  d{}", rec.id as u32, rec.depth),
        format!(
            "rect {} {} × {} {}",
            rec.rect.x, rec.rect.y, rec.rect.w, rec.rect.h
        ),
        format!("intrinsic {:.0} × {:.0}", rec.intrinsic_w, rec.intrinsic_h),
        format!(
            "baseline {}",
            rec.baseline.map_or("-".to_string(), |b| format!("{b:.1}"))
        ),
        style_txt,
    ];
    let mut ty = card.y + 6.0;
    for line in &lines {
        ui.label(Rect::new(card.x + 8.0, ty, card.w - 12.0, 12.0), line, ts);
        ty += 16.0;
    }
    ui.draw.set_layer(0);
}

/// Per-frame inspect wiring: pin handling ran in [`Ui::begin`].
pub(crate) fn devtools_begin(ui: &mut Ui) {
    // Records of the *previous* frame pick the hovered node for pinning;
    // consume the press so widgets under the pointer don't click.
    if ui.devtools != DevtoolsMode::Inspect || !ui.input.mouse_pressed {
        return;
    }
    let mouse = ui.input.mouse;
    let hit = ui
        .view
        .devtools
        .records
        .iter()
        .filter(|r| r.rect.contains(mouse))
        .max_by_key(|r| r.depth)
        .map(|r| r.id);
    let pinned = ui.view.devtools.pinned;
    ui.view.devtools.pinned = if pinned == hit { None } else { hit };
    ui.press_consumed = true;
}
