//! L3 declarative views: a per-frame [`crate::ui::layout::LayoutTree`]
//! built under [`Ui::layout_root`](crate::ui::widgets::Ui::layout_root), solved once, then painted in tree
//! order. Interactive nodes run their headless behaviour against the
//! previous frame's solved rect, so `ui.add(w).clicked` is synchronous;
//! a brand-new node gets an off-screen rect for this frame and requests
//! an immediate wake so the next frame has it.
//!
//! ```ignore
//! ui.layout_root(area, Style::column().gap(8.0), |ui| {
//!     ui.text("Title");
//!     if ui.add(Button::new("Done")).clicked { /* … */ }
//! });
//! ```

mod node;
pub(crate) mod paint;

use crate::ui::Rect;
use crate::ui::layout::{LayoutTree, Style};
use crate::ui::widgets::Response;
use std::collections::HashMap;

pub use node::{Intrinsic, MeasureCx, Paint, PaintOp, TextWrap, Widget};

pub(crate) use node::{NodeOp, ToggleKind};

/// Per-frame view layer state (a field of `Ui`).
#[derive(Default)]
pub struct ViewState {
    /// Working tree during a build (the pool lives in `trees`).
    pub(crate) tree: LayoutTree,
    /// Open containers (top of the stack = current parent).
    pub(crate) stack: Vec<u32>,
    pub(crate) in_layout: bool,
    /// widget-id → solved rect from the previous frame / this frame.
    pub(crate) prev_rects: HashMap<u64, Rect>,
    pub(crate) cur_rects: HashMap<u64, Rect>,
    /// widget-id → content size from the previous solve (scroll extents).
    pub(crate) prev_content: HashMap<u64, (f32, f32)>,
    /// Double-buffered counterpart (this frame's writes).
    pub(crate) cur_content: HashMap<u64, (f32, f32)>,
    /// Clip-scope stack for paint replay (pooled across frames).
    pub(crate) clip_stack: Vec<(u32, bool)>,
    /// True while `build` replays paint-only (edge input stripped):
    /// anim helpers must not step (`anim_dt` returns 0).
    pub(crate) replay: bool,
    /// Set by add/open paths when a node lacks a previous rect.
    pub(crate) needs_presolve: bool,
    /// Per-open-container next child index (avoids `children().count()`).
    pub(crate) child_idx: std::collections::HashMap<u32, u32>,
    /// One layout tree per `layout_root` call order, so several roots per
    /// frame cache independently.
    pub(crate) trees: Vec<crate::ui::layout::LayoutTree>,
    /// Cursor into `trees`, reset each `begin`.
    pub(crate) tree_cursor: usize,
    /// Paint ops, parallel to the tree's node indices.
    pub(crate) ops: Vec<NodeOp>,
    /// Node ids (widget-id or derived container id), parallel to nodes.
    pub(crate) node_ids: Vec<u64>,
    /// Per-frame string arena for labels/text in ops.
    pub(crate) arena: String,
    /// Measure requests for `Leaf::Measured` nodes, keyed by node id.
    pub(crate) measure_reqs: HashMap<u64, node::MeasureReq>,
    /// Scroll offsets keyed by container id, applied at solve.
    /// `(offset, last_frame)` — stale entries retire like other state.
    pub(crate) scroll: HashMap<u64, (f32, u64)>,
    /// Paint replay order (enter/leaf/exit markers by node index).
    pub(crate) order: Vec<super::layout::NodeId>,
    /// Build-time clip stack (parallel to open containers).
    pub(crate) build_clips: Vec<bool>,
    /// Scratch list for `PaintOp::record` captures.
    /// Shared per-frame segmented cell buffer (`(x, w)` cells).
    /// Shared per-frame segmented item buffer (label range, state, near).
    pub(crate) seg_items: Vec<SegItem>,
    /// Inspector records/state for devtools.
    pub(crate) devtools: crate::ui::devtools::InspectState,
    /// Scratch list for `PaintOp::record` captures.
    pub(crate) scratch: crate::ui::draw::DrawList,
}

/// Frame statistics for benches.
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewStats {
    pub nodes: usize,
    pub solves: u64,
    pub cache_hits: u64,
}

/// One pooled segmented item (label arena range + live state + nearness).
#[derive(Clone, Copy)]
pub(crate) struct SegItem {
    pub text: (u32, u32),
    /// Segment widget id (computed in the label scope) for focus rings.
    pub id: u64,
    pub near: f32,
}

/// Paint-only replay input: level state kept (mouse position and held
/// buttons, nav shift/word modifiers, IME preedit), every edge event
/// stripped and dt frozen so nothing re-fires or double-steps.
fn replay_input(input: &crate::ui::widgets::UiInput) -> crate::ui::widgets::UiInput {
    let mut i = crate::ui::widgets::UiInput {
        dt: 0.0,
        mouse: input.mouse,
        mouse_down: input.mouse_down,
        mouse_right_down: input.mouse_right_down,
        preedit: input.preedit.clone(),
        ..Default::default()
    };
    i.nav.shift = input.nav.shift;
    i.nav.word = input.nav.word;
    i
}

fn pid_salt(ui: &crate::ui::widgets::Ui, parent: u32) -> u64 {
    ui.view.node_ids[parent as usize]
}

impl crate::ui::widgets::Ui {
    /// Open a declarative layout pass covering `rect`. Nested calls panic
    /// in debug builds. On frames where the layout changes `build` runs a
    /// second, paint-only time with edge input stripped — keep it
    /// idempotent and record state from events (`if resp.clicked { … }`),
    /// not by unconditional assignment from responses.
    pub fn layout_root(&mut self, rect: Rect, style: Style, mut build: impl FnMut(&mut Self)) {
        debug_assert!(!self.view.in_layout, "layout_root may not be nested");
        self.view.in_layout = true;
        let cursor = self.view.tree_cursor;
        self.view.tree_cursor += 1;
        while self.view.trees.len() <= cursor {
            self.view.trees.push(crate::ui::layout::LayoutTree::new());
        }
        // Recorded paint ops for this root land in the scratch list; it
        // never culls (a record at last frame's rect may splice on-screen).
        self.view.scratch.begin(self.screen);
        self.view.scratch.cull = false;
        let draw_cp = self.draw.checkpoint();
        let scratch_cp = self.view.scratch.checkpoint();
        let regs = self.regs_checkpoint();
        self.view_open_root(style);
        self.view.needs_presolve = false;
        build(self);
        self.view.stack.pop();
        let mut tree = self.view_solve_publish(rect);
        if self.view.needs_presolve {
            // Retry once: publish these solved rects as "previous", roll
            // the draw lists back, truncate live-pass registrations, and
            // replay the build paint-only (edge input stripped, anim dt
            // frozen, scroll physics skipped) at real rects. Interaction
            // state the live pass produced is authoritative and kept.
            let (hot, active) = (self.hot_id, self.active_id);
            let (focus, focus_vis) = (self.focus_id(), self.focus_visible());
            self.draw.restore(draw_cp);
            self.view.scratch.restore(scratch_cp);
            self.regs_restore(regs);
            // Recycle the first solved tree: its allocation is retained
            // by the pool for the reopened root.
            self.view.trees[cursor] = tree;
            self.view.tree = crate::ui::layout::LayoutTree::new();
            self.view_open_root(style);
            self.view.needs_presolve = false;
            let saved = {
                let r = replay_input(&self.input);
                std::mem::replace(&mut self.input, r)
            };
            self.view.replay = true;
            build(self);
            self.view.replay = false;
            self.input = saved;
            self.hot_id = hot;
            self.active_id = active;
            self.restore_view_focus(focus, focus_vis);
            self.view.stack.pop();
            tree = self.view_solve_publish(rect);
        }
        self.view_paint(&mut tree);
        self.view.trees[cursor] = tree;
        self.view.tree = crate::ui::layout::LayoutTree::new();
        self.view.in_layout = false;
    }

    /// Clear the per-root buffers and push a fresh root node.
    fn view_open_root(&mut self, style: Style) {
        self.view.tree = std::mem::take(&mut self.view.trees[self.view.tree_cursor - 1]);
        self.view.tree.clear();
        self.view.stack.clear();
        self.view.ops.clear();
        self.view.node_ids.clear();
        self.view.order.clear();
        self.view.arena.clear();
        self.view.seg_items.clear();
        self.view.child_idx.clear();
        self.view.measure_reqs.clear();
        self.view.devtools.records.clear();
        let root = self
            .view
            .tree
            .push(None, style, crate::ui::layout::Leaf::None);
        self.view.stack.push(root.0);
        self.view.child_idx.insert(0, 0);
        self.view.ops.push(NodeOp::None);
        self.view.node_ids.push(self.id("##view_root"));
        self.view.order.push(root);
    }

    /// Solve the current tree, publish solved rects/content into the
    /// current maps, set `needs_presolve` when recorded/clip bounds change, then
    /// record + paint + devtools overlays.
    fn view_solve_publish(&mut self, rect: Rect) -> crate::ui::layout::LayoutTree {
        let mut tree = std::mem::take(&mut self.view.tree);
        let mut reqs = std::mem::take(&mut self.view.measure_reqs);
        let scale = self.scale();
        let mut host = paint::Host {
            ui: self,
            reqs: &mut reqs,
        };
        tree.solve(rect, scale, &mut host);
        self.view.measure_reqs = reqs;
        let mut missing = self.view.needs_presolve;
        for i in 0..tree.len() as u32 {
            let r = tree.rect(crate::ui::layout::NodeId(i));
            let id = self.view.node_ids[i as usize];
            self.view.cur_rects.insert(id, r);
            self.view
                .cur_content
                .insert(id, tree.content_size(crate::ui::layout::NodeId(i)));
            match &self.view.ops[i as usize] {
                NodeOp::Record { prev, .. } if r != *prev => missing = true,
                NodeOp::Container(p)
                    if (p.clip
                        || matches!(
                            tree.style(crate::ui::layout::NodeId(i)).overflow,
                            crate::ui::layout::Overflow::Clip
                                | crate::ui::layout::Overflow::Scroll { .. }
                        ))
                        && self.view.prev_rects.get(&id) != Some(&r) =>
                {
                    missing = true;
                }
                _ => {}
            }
        }
        self.view.needs_presolve = missing;
        if missing && !self.view.replay {
            // Publish this solve as "previous" for the replay build so
            // nodes interact/record at real rects on the final pass.
            for i in 0..tree.len() as u32 {
                let r = tree.rect(crate::ui::layout::NodeId(i));
                self.view
                    .prev_rects
                    .insert(self.view.node_ids[i as usize], r);
            }
        }
        tree
    }

    /// Paint the final solved tree once, plus devtools overlays. Runs
    /// after any replay so custom slots execute a single time per frame.
    fn view_paint(&mut self, tree: &mut crate::ui::layout::LayoutTree) {
        let devtools_on = self.devtools != crate::ui::devtools::DevtoolsMode::Off;
        for i in 0..tree.len() as u32 {
            if !devtools_on {
                break;
            }
            let node = crate::ui::layout::NodeId(i);
            crate::ui::devtools::record_node(
                self,
                tree,
                node,
                self.view.node_ids[i as usize],
                tree.depth(node) as u16,
            );
        }
        paint::paint_tree(self, tree);
        if self.devtools != crate::ui::devtools::DevtoolsMode::Off {
            crate::ui::devtools::draw_outlines(self, tree);
            if self.devtools == crate::ui::devtools::DevtoolsMode::Inspect {
                crate::ui::devtools::draw_inspect(self);
            }
        }
    }

    /// A row-direction flex container.
    pub fn row(&mut self, mut style: Style, build: impl FnOnce(&mut Self)) {
        style.display = crate::ui::layout::Display::Flex;
        style.direction = crate::ui::layout::Direction::Row;
        self.open_container(style, Paint::none());
        build(self);
        self.close_container();
    }

    /// A column-direction flex container.
    pub fn column(&mut self, mut style: Style, build: impl FnOnce(&mut Self)) {
        style.display = crate::ui::layout::Display::Flex;
        style.direction = crate::ui::layout::Direction::Column;
        self.open_container(style, Paint::none());
        build(self);
        self.close_container();
    }

    /// A stacking container (children placed by their `place`).
    pub fn stack(&mut self, mut style: Style, build: impl FnOnce(&mut Self)) {
        style.display = crate::ui::layout::Display::Stack;
        self.open_container(style, Paint::none());
        build(self);
        self.close_container();
    }

    /// A container with chrome drawn by `paint`.
    pub fn container(&mut self, style: Style, paint: Paint, build: impl FnOnce(&mut Self)) {
        self.open_container(style, paint);
        build(self);
        self.close_container();
    }

    pub(crate) fn open_container(&mut self, style: Style, paint: Paint) {
        let parent = *self
            .view
            .stack
            .last()
            .expect("container outside layout_root");
        let p = crate::ui::layout::NodeId(parent);
        // Derived id: key, or parent id + child index.
        // Child index from the parent frame — avoids a sibling walk.
        let idx = self.view.child_idx.entry(parent).or_insert(0);
        let idx = std::mem::replace(idx, *idx + 1) as u64;
        let pid = self.view.node_ids[parent as usize];
        let id = style
            .key
            .unwrap_or_else(|| crate::ui::widgets::fnv_push_num(pid, idx));
        if !self.view.prev_rects.contains_key(&id) {
            // Container children may clip to this rect — a first-frame
            // container needs the presolve too.
            self.view.needs_presolve = true;
        }
        // Scroll input and motion against last frame's rect + content
        // extent — the same shared viewport the widget layer runs, so a
        // layout container scrolls exactly like `scroll_begin` (glide,
        // momentum, chaining, keyboard, scrollbar hand).
        if let crate::ui::layout::Overflow::Scroll { .. } = style.overflow
            && !self.view.replay
            && let (Some(&vp), Some(&cs)) = (
                self.view.prev_rects.get(&id),
                self.view.prev_content.get(&id),
            )
        {
            let off = self.scroll_viewport(id, vp, cs.1);
            self.set_view_scroll(id, off);
        }
        let scroll = self.view.scroll.get(&id).map_or(0.0, |&(o, _)| o);
        let node = self
            .view
            .tree
            .push(Some(p), style, crate::ui::layout::Leaf::None);
        self.view.tree.set_scroll_offset(node, (0.0, scroll));
        self.view.stack.push(node.0);
        self.view.ops.push(NodeOp::Container(paint));
        self.view.node_ids.push(id);
        self.view.order.push(node);
        // During build, children hit-test under the container's previous
        // clip rect.
        let clipped = matches!(
            self.view.tree.style(node).overflow,
            crate::ui::layout::Overflow::Clip | crate::ui::layout::Overflow::Scroll { .. }
        ) || paint.clip;
        let clipped = clipped && self.view.prev_rects.contains_key(&id);
        if clipped {
            self.draw.push_clip(self.view.prev_rects[&id]);
        }
        self.view.build_clips.push(clipped);
    }

    pub(crate) fn close_container(&mut self) {
        self.view.stack.pop();
        self.view.order.push(crate::ui::layout::NodeId(u32::MAX)); // Exit
        if self.view.build_clips.pop() == Some(true) {
            self.draw.pop_clip();
        }
    }

    /// Add a widget leaf to the current container.
    pub fn add(&mut self, w: impl Widget) -> Response {
        self.add_styled(w, Style::default())
    }

    /// `add` with a layout-style override for the leaf.
    pub fn add_styled(&mut self, w: impl Widget, mut style: Style) -> Response {
        style = merge_style(w.layout_style(), style);
        let parent = *self.view.stack.last().expect("add outside layout_root");
        let pid = self.view.node_ids[parent as usize];
        // Style::key wins (matches containers); else the widget's label.
        let id = style
            .key
            .or_else(|| w.label().map(|l| self.id(l)))
            .unwrap_or_else(|| {
                crate::ui::widgets::fnv_push_num(
                    pid,
                    self.view.tree.len() as u64 ^ 0x1eaf_0000_0000_0000,
                )
            });
        let prev = self.view.prev_rects.get(&id).copied();
        if prev.is_none() {
            self.view.needs_presolve = true;
            self.request_wake(0.0);
        }
        let intr = {
            let mut cx = MeasureCx { ui: self };
            w.measure(&mut cx)
        };
        // A node with no previous rect skips its behaviour this pass: running
        // it against a placeholder rect would corrupt persistent state (e.g.
        // a text field scrolling to fit a zero-width box). The same-frame
        // replay interacts and paints it at the solved rect.
        let (resp, op) = match prev {
            Some(rect) => w.interact(self, rect),
            None => (crate::ui::widgets::Response::default(), PaintOp::none()),
        };
        let leaf = match intr.wrap {
            Some(t) => {
                self.view.measure_reqs.insert(id, node::MeasureReq::Text(t));
                crate::ui::layout::Leaf::Measured { key: id }
            }
            None => crate::ui::layout::Leaf::Fixed {
                w: intr.w,
                h: intr.h,
                baseline: intr.baseline,
            },
        };
        let node = self
            .view
            .tree
            .push(Some(crate::ui::layout::NodeId(parent)), style, leaf);
        self.view.ops.push(op.0);
        self.view.node_ids.push(id);
        self.view.order.push(node);
        resp
    }

    /// A text leaf (`add(Text::new(text))`).
    pub fn text(&mut self, text: &str) -> Response {
        self.add(crate::ui::widgets::adapters::Text::new(text))
    }

    /// A grow-1 filler node.
    pub fn spacer(&mut self) {
        let parent = *self.view.stack.last().expect("spacer outside layout_root");
        let node = self.view.tree.push(
            Some(crate::ui::layout::NodeId(parent)),
            Style::default().fill(),
            crate::ui::layout::Leaf::Fixed {
                w: 0.0,
                h: 0.0,
                baseline: None,
            },
        );
        let id = crate::ui::widgets::fnv_push_num(
            pid_salt(self, parent),
            0x5_A0CEu64.wrapping_add(node.0 as u64),
        );
        if !self.view.prev_rects.contains_key(&id) {
            self.view.needs_presolve = true;
        }
        self.view.ops.push(NodeOp::None);
        self.view.node_ids.push(id);
        self.view.order.push(node);
    }

    /// A node with an exact solved rect and caller paint (boxed).
    pub fn slot(&mut self, style: Style, paint: impl FnOnce(&mut Self, Rect) + 'static) {
        let parent = *self.view.stack.last().expect("slot outside layout_root");
        let node = self.view.tree.push(
            Some(crate::ui::layout::NodeId(parent)),
            style,
            crate::ui::layout::Leaf::Fixed {
                w: 0.0,
                h: 0.0,
                baseline: None,
            },
        );
        let id = style.key.unwrap_or_else(|| {
            crate::ui::widgets::fnv_push_num(
                pid_salt(self, parent),
                0x5107_u64.wrapping_add(node.0 as u64),
            )
        });
        if !self.view.prev_rects.contains_key(&id) {
            self.view.needs_presolve = true;
        }
        self.view.ops.push(NodeOp::Custom(Box::new(paint)));
        self.view.node_ids.push(id);
        self.view.order.push(node);
    }

    /// Id-scope wrapper — children hash under `key`.
    pub fn keyed(&mut self, key: &str, build: impl FnOnce(&mut Self)) {
        self.push_id(key);
        build(self);
        self.pop_id();
    }

    /// A widget/container's solved rect: this frame's when solved, else
    /// the previous frame's.
    pub fn rect_of(&self, label: &str) -> Option<Rect> {
        let id = self.id(label);
        self.view
            .cur_rects
            .get(&id)
            .or_else(|| self.view.prev_rects.get(&id))
            .copied()
    }

    /// True inside a [`Self::layout_root`] build.
    pub fn in_layout(&self) -> bool {
        self.view.in_layout
    }

    /// Current view-scroll offset for a container id.
    pub(crate) fn view_scroll(&self, id: u64) -> f32 {
        self.view.scroll.get(&id).map_or(0.0, |&(o, _)| o)
    }

    /// Set a scroll container's offset (clamped at solve).
    pub(crate) fn set_view_scroll(&mut self, id: u64, off: f32) {
        self.view.scroll.insert(id, (off, self.frame));
    }

    /// The widget id a view node carries.
    pub(crate) fn node_id_of(&self, node: crate::ui::layout::NodeId) -> u64 {
        self.view.node_ids[node.0 as usize]
    }

    /// Layout stats for benches.
    pub fn view_stats(&self) -> ViewStats {
        let (mut nodes, mut solves, mut hits) = (0usize, 0u64, 0u64);
        for t in &self.view.trees {
            let s = t.stats();
            nodes += t.len();
            solves += s.solves;
            hits += s.cache_hits;
        }
        ViewStats {
            nodes,
            solves,
            cache_hits: hits,
        }
    }
}

/// `add_styled`: fields the caller didn't override keep the widget's own.
fn merge_style(w: Style, o: Style) -> Style {
    Style {
        width: if o.width == crate::ui::layout::Size::Fit {
            w.width
        } else {
            o.width
        },
        height: if o.height == crate::ui::layout::Size::Fit {
            w.height
        } else {
            o.height
        },
        ..o
    }
}

#[cfg(test)]
mod tests;
