//! Paint pass: replay node ops in tree order at the solved rects.

use super::node::{MeasureReq, NodeOp, Paint, TextWrap};
use crate::ui::Rect;
use crate::ui::font::LayoutOptions;
use crate::ui::layout::{LayoutTree, NodeId, Overflow};
use crate::ui::tokens::Elevation;
use crate::ui::widgets::{Ui, paint as wpaint};

/// Measure host: text nodes measure height-for-width through the Ui.
pub(crate) struct Host<'a> {
    pub ui: &'a mut Ui,
    pub reqs: &'a mut std::collections::HashMap<u64, MeasureReq>,
}

impl crate::ui::layout::Measure for Host<'_> {
    fn measure(
        &mut self,
        node: NodeId,
        key: u64,
        avail_w: Option<f32>,
    ) -> crate::ui::layout::LeafSize {
        let _ = node;
        match self.reqs.get(&key) {
            Some(MeasureReq::Text(t)) => text_size(self.ui, t, avail_w),
            None => Default::default(),
        }
    }
}

fn text_size(ui: &mut Ui, t: &TextWrap, avail_w: Option<f32>) -> crate::ui::layout::LeafSize {
    let scale = ui.scale();
    let text = {
        let (a, b) = t.text;
        ui.view.arena[a as usize..b as usize].to_owned()
    };
    let layout = ui.cached_layout(
        &text,
        t.font,
        t.size * scale,
        LayoutOptions {
            wrap_width: avail_w.map(|w| (w * scale).max(1.0)),
            line_height: t.line_height.max(1.0),
            ..Default::default()
        },
    );
    let ink = layout.ink_bounds();
    crate::ui::layout::LeafSize {
        w: ink.map_or(0.0, |r| r.w).max(1.0) / scale,
        h: ink.map_or(t.size, |r| r.h).max(1.0) / scale,
        baseline: Some(layout.first_baseline / scale),
    }
}

/// Paint all nodes in tree order at their solved rects; clip containers
/// scope their children.
pub(crate) fn paint_tree(ui: &mut Ui, tree: &mut LayoutTree) {
    let order = std::mem::take(&mut ui.view.order);
    let arena = std::mem::take(&mut ui.view.arena);
    let seg_items = std::mem::take(&mut ui.view.seg_items);
    // Track which nodes opened a clip so Exit can pop it.
    let mut clip_stack = std::mem::take(&mut ui.view.clip_stack);
    clip_stack.clear();
    for &nid in &order {
        if nid.0 == u32::MAX {
            // Exit marker: pop this container's clip and paint its
            // scroll thumb over the children.
            if let Some((container, clipped)) = clip_stack.pop().map(|(n, c)| (NodeId(n), c))
                && clipped
            {
                scroll_thumb(ui, tree, container);
                ui.draw.pop_clip();
            }
            continue;
        }
        let rect = tree.rect(nid);
        let is_parent = tree.children(nid).next().is_some();
        match ui.view.ops.get_mut(nid.0 as usize) {
            Some(NodeOp::Container(p)) if is_parent => {
                let p = *p;
                paint_container(ui, rect, &p);
                let clipped = p.clip
                    || matches!(
                        tree.style(nid).overflow,
                        Overflow::Clip | Overflow::Scroll { .. }
                    );
                if matches!(tree.style(nid).overflow, Overflow::Scroll { .. }) {
                    // Register the wheel region at the solved rect before
                    // the clip is pushed (enclosing clip applies).
                    ui.register_scroll_view(ui.view.node_ids[nid.0 as usize], rect);
                }
                if clipped {
                    ui.draw.push_clip(rect);
                }
                clip_stack.push((nid.0, clipped));
            }
            _ => {
                if is_parent {
                    clip_stack.push((nid.0, false));
                }
                let op = std::mem::replace(&mut ui.view.ops[nid.0 as usize], NodeOp::None);
                paint_op(ui, &arena, &seg_items, rect, op);
            }
        }
    }
    // Exit markers pair every pushed clip; drain any stragglers.
    while let Some((_, clipped)) = clip_stack.pop() {
        if clipped {
            ui.draw.pop_clip();
        }
    }
    ui.view.arena = arena;
    ui.view.seg_items = seg_items;
    ui.view.clip_stack = clip_stack;
    ui.view.order = order;
}

fn arena_str(arena: &str, r: (u32, u32)) -> &str {
    &arena[r.0 as usize..r.1 as usize]
}

fn paint_op(ui: &mut Ui, arena: &str, seg_items: &[super::SegItem], rect: Rect, op: NodeOp) {
    match op {
        NodeOp::None | NodeOp::Container(_) => {}
        NodeOp::Custom(f) => f(ui, rect),
        NodeOp::Button {
            behavior_id,
            style,
            state,
            label,
            icon,
        } => {
            let label = arena_str(arena, label);
            match icon {
                Some((ic, sz)) => wpaint::button_with_icon(ui, rect, &style, state, ic, sz, label),
                None => wpaint::button(ui, rect, &style, state, label),
            }
            ui.focus_ring(rect, style.corner_radius, behavior_id);
        }
        NodeOp::IconButton {
            behavior_id,
            style,
            state,
            icon,
            icon_size,
            on,
            badge,
        } => {
            let badge = badge.map(|r| arena_str(arena, r));
            wpaint::icon_button(ui, rect, &style, state, icon, icon_size, on, badge);
            ui.focus_ring(rect, style.corner_radius, behavior_id);
        }
        NodeOp::Toggle {
            behavior_id,
            style,
            state,
            label,
            kind,
        } => {
            let label = arena_str(arena, label);
            match kind {
                super::node::ToggleKind::Switch => {
                    wpaint::switch(ui, rect, &style, state);
                }
                super::node::ToggleKind::Checkbox => {
                    wpaint::checkbox(ui, rect, &style, state, label);
                }
                super::node::ToggleKind::Radio => {
                    wpaint::radio(ui, rect, &style, state, label);
                }
            }
            ui.focus_ring(rect, rect.h * 0.5, behavior_id);
        }
        NodeOp::Slider {
            behavior_id,
            style,
            state,
            label,
        } => {
            let label = arena_str(arena, label);
            wpaint::slider(ui, rect, &style, state, label);
            ui.focus_ring(rect, rect.h * 0.5, behavior_id);
        }
        NodeOp::Text { style, text } => {
            ui.label(rect, arena_str(arena, text), style);
        }
        NodeOp::Segmented { style, at, items } => {
            let items_v = &seg_items[items.0 as usize..items.1 as usize];
            // Cells come from the SOLVED rect — a moved/resized control
            // must paint its pill and labels there this frame.
            let mut cells_v = std::mem::take(&mut ui.seg_cells);
            crate::ui::layout::divide_into(
                &mut cells_v,
                rect.x,
                rect.w,
                items_v.len(),
                0.0,
                ui.scale(),
            );
            wpaint::segmented(ui, rect, &style, at, &cells_v);
            // Item labels with the pill-nearness text tint, same as the
            // L1 segmented path.
            for (i, it) in items_v.iter().enumerate() {
                let t = arena_str(arena, it.text);
                let cell = Rect::new(cells_v[i].0, rect.y, cells_v[i].1, rect.h).inset(style.inset);
                let text = crate::ui::widgets::mix_color(style.text, style.text_selected, it.near);
                ui.focus_ring(cell, style.button.corner_radius, it.id);
                ui.label(
                    cell,
                    t,
                    crate::ui::widgets::LabelStyle::new(style.button.font)
                        .size(style.button.size)
                        .color(text)
                        .align(crate::ui::font::Align::Center)
                        .vcenter(true),
                );
            }
            ui.seg_cells = cells_v;
        }
        NodeOp::Icon { icon, size, tint } => {
            ui.icon(rect, icon, size, tint);
        }
        NodeOp::Separator { color, .. } => {
            ui.draw.rect(rect, color);
        }
        NodeOp::Progress { fill, value } => {
            ui.progress(rect, value, fill);
        }
        NodeOp::Badge { text, bg, fg, size } => {
            ui.badge(
                rect,
                arena_str(arena, text),
                bg,
                fg,
                crate::ui::font::FontId(0),
                size,
            );
        }
        NodeOp::Record { start, end, prev } => {
            let offset = (rect.x - prev.x, rect.y - prev.y);
            ui.draw.splice_from(&ui.view.scratch, start, end, offset);
            if (rect.w - prev.w).abs() > 0.5 || (rect.h - prev.h).abs() > 0.5 {
                ui.request_wake(0.0);
            }
        }
    }
}

/// Draw a scroll container's thumb over its (already offset) children.
fn scroll_thumb(ui: &mut Ui, tree: &LayoutTree, node: NodeId) {
    let style = tree.style(node);
    if !matches!(style.overflow, Overflow::Scroll { .. }) {
        return;
    }
    let wid = ui.node_id_of(node);
    let rect = tree.rect(node);
    let (_, ch) = tree.content_size(node);
    let offset = ui.view_scroll(wid);
    ui.draw_scrollbar(wid, rect, offset, ch);
}

/// Paint container chrome (called when a container opens).
fn paint_container(ui: &mut Ui, rect: Rect, p: &Paint) {
    if p.elevation != Elevation::Flat {
        ui.elevation_shadow(rect, p.radius, p.elevation);
    }
    match (p.bg, p.border) {
        (Some(bg), Some((w, c))) => {
            let w = if w == 1.0 {
                crate::ui::layout::hairline(ui.scale())
            } else {
                w
            };
            ui.draw.rounded_frame(rect, p.radius, w, bg, c);
        }
        (Some(bg), None) => ui.draw.round_rect(rect, p.radius, bg),
        (None, Some((w, c))) => {
            ui.draw.round_ring(rect, p.radius, w, c);
        }
        (None, None) => {}
    }
}
