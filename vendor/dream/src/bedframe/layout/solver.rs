//! The layout tree: arena of styled nodes + the CSS-flexbox-subset
//! solve. Push nodes, call [`LayoutTree::solve`], read rects.

use super::Rect;
use super::cache::SolveHash;
use super::measure::{Leaf, LeafSize, Measure};
use super::snap_rect;
use super::style::{Direction, Display, Overflow, Position, Size, Style};

/// Stable handle to a node (arena index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

const NIL: u32 = u32::MAX;

#[derive(Clone, Copy)]
pub(crate) struct Node {
    pub style: Style,
    pub leaf: Leaf,
    pub parent: u32,
    pub first_child: u32,
    pub next: u32,
    /// Absolute border box, *unsnapped* — children are positioned from
    /// this so shared edges snap identically.
    pub abs: Rect,
    /// Intrinsic (natural) size from the measure pass.
    pub intrinsic: (f32, f32),
    /// Children extent including padding (scroll extent).
    pub content: (f32, f32),
    /// First baseline, absolute y.
    pub baseline: Option<f32>,
    /// Scroll offset applied to children (clamped at solve).
    pub scroll: (f32, f32),
}

/// Sibling traversal for [`LayoutTree::children`].
pub struct Children<'a> {
    tree: &'a [Node],
    next: u32,
}

impl Iterator for Children<'_> {
    type Item = NodeId;
    fn next(&mut self) -> Option<NodeId> {
        if self.next == NIL {
            return None;
        }
        let id = self.next;
        self.next = self.tree[id as usize].next;
        Some(NodeId(id))
    }
}

/// Solve statistics (`clear` keeps them).
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    pub solves: u64,
    pub cache_hits: u64,
}

/// Main/cross axis values for a direction: `M`/`C` are per-node f32s.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Ax {
    pub main: f32,
    pub cross: f32,
}

impl Ax {
    pub fn of(dir: Direction, w: f32, h: f32) -> Self {
        match dir {
            Direction::Row => Self { main: w, cross: h },
            Direction::Column => Self { main: h, cross: w },
        }
    }

    pub fn wh(&self, dir: Direction) -> (f32, f32) {
        match dir {
            Direction::Row => (self.main, self.cross),
            Direction::Column => (self.cross, self.main),
        }
    }
}

/// Arena of nodes; `push` appends children, `solve` lays the tree out.
pub struct LayoutTree {
    nodes: Vec<Node>,
    scale: f32,
    hash: SolveHash,
    last_hash: Option<u64>,
    /// Outputs of the last real solve, restored on a cache hit.
    last_out: Vec<(Rect, (f32, f32), Option<f32>)>,
    /// Scratch buffers reused across solves (steady state → no allocs).
    kids_scratch: Vec<u32>,
    scrolls_scratch: Vec<(f32, f32)>,
    stats: SolveStats,
}

impl Default for LayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            scale: 1.0,
            hash: SolveHash::new(),
            last_hash: None,
            last_out: Vec::new(),
            kids_scratch: Vec::new(),
            scrolls_scratch: Vec::new(),
            stats: SolveStats::default(),
        }
    }

    /// Reset the arena for the next frame (keeps capacity and the cache).
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.hash = SolveHash::new();
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The node's intrinsic (natural) size from the measure pass.
    pub fn leaf_size(&self, id: NodeId) -> (f32, f32) {
        self.nodes[id.0 as usize].intrinsic
    }

    /// Depth in the tree (root = 0).
    pub fn depth(&self, mut id: NodeId) -> u32 {
        let mut d = 0;
        while let Some(p) = self.parent(id) {
            d += 1;
            id = p;
        }
        d
    }

    /// Append a node as `parent`'s last child (`None` = root).
    pub fn push(&mut self, parent: Option<NodeId>, style: Style, leaf: Leaf) -> NodeId {
        let id = self.nodes.len() as u32;
        self.hash.push_node(parent, &style, &leaf);
        let mut node = Node {
            style,
            leaf,
            parent: NIL,
            first_child: NIL,
            next: NIL,
            abs: Rect::new(0.0, 0.0, 0.0, 0.0),
            intrinsic: (0.0, 0.0),
            content: (0.0, 0.0),
            baseline: None,
            scroll: (0.0, 0.0),
        };
        if let Some(p) = parent {
            node.parent = p.0;
            if self.nodes[p.0 as usize].first_child == NIL {
                self.nodes[p.0 as usize].first_child = id;
            } else {
                let mut c = self.nodes[p.0 as usize].first_child;
                while self.nodes[c as usize].next != NIL {
                    c = self.nodes[c as usize].next;
                }
                self.nodes[c as usize].next = id;
            }
        }
        self.nodes.push(node);
        NodeId(id)
    }

    pub fn style(&self, id: NodeId) -> &Style {
        &self.nodes[id.0 as usize].style
    }

    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        let p = self.nodes[id.0 as usize].parent;
        (p != NIL).then_some(NodeId(p))
    }

    pub fn children(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        Children {
            tree: &self.nodes,
            next: self.nodes[id.0 as usize].first_child,
        }
    }

    pub(crate) fn child_ids(&self, id: u32) -> Vec<u32> {
        self.children(NodeId(id)).map(|c| c.0).collect()
    }

    /// `id`'s child indices pushed into `out` (reused scratch).
    pub(crate) fn child_ids_into(&self, id: u32, out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.children(NodeId(id)).map(|c| c.0));
    }

    /// Absolute border box, edge-snapped to the physical grid.
    pub fn rect(&self, id: NodeId) -> Rect {
        snap_rect(self.nodes[id.0 as usize].abs, self.scale)
    }

    /// [`Self::rect`] minus padding (snapped).
    pub fn content_rect(&self, id: NodeId) -> Rect {
        let n = &self.nodes[id.0 as usize];
        let p = n.style.padding;
        snap_rect(
            Rect::new(
                n.abs.x + p.left,
                n.abs.y + p.top,
                (n.abs.w - p.horizontal()).max(0.0),
                (n.abs.h - p.vertical()).max(0.0),
            ),
            self.scale,
        )
    }

    /// Children extent including padding (the scroll extent).
    pub fn content_size(&self, id: NodeId) -> (f32, f32) {
        self.nodes[id.0 as usize].content
    }

    /// Absolute y of the first baseline.
    pub fn baseline(&self, id: NodeId) -> Option<f32> {
        self.nodes[id.0 as usize].baseline
    }

    /// Set the scroll offset applied to a node's children.
    pub fn set_scroll_offset(&mut self, id: NodeId, off: (f32, f32)) {
        self.nodes[id.0 as usize].scroll = off;
    }

    pub fn scroll_offset(&self, id: NodeId) -> (f32, f32) {
        self.nodes[id.0 as usize].scroll
    }

    pub fn stats(&self) -> SolveStats {
        self.stats
    }

    pub(crate) fn node(&self, id: u32) -> &Node {
        &self.nodes[id as usize]
    }

    pub(crate) fn node_mut(&mut self, id: u32) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    /// Intrinsic (natural) size of a node after its own Size/min/max.
    fn measure_node(&mut self, id: u32, host: &mut dyn Measure) -> (f32, f32) {
        let (style, leaf) = {
            let n = &self.nodes[id as usize];
            (n.style, n.leaf)
        };
        let (mut w, mut h) = match leaf {
            Leaf::None => self.container_intrinsic(id, &style, host),
            Leaf::Fixed { w, h, .. } => (w, h),
            Leaf::Measured { key } => {
                let s = host.measure(NodeId(id), key, None);
                (s.w, s.h)
            }
        };
        if let Size::Px(v) = style.width {
            w = v;
        }
        if let Size::Px(v) = style.height {
            h = v;
        }
        if let Some(r) = style.aspect_ratio {
            match (style.width, style.height) {
                (Size::Px(_), _) if !matches!(style.height, Size::Px(_)) => h = w / r,
                (_, Size::Px(_)) if !matches!(style.width, Size::Px(_)) => w = h * r,
                _ => {}
            }
        }
        w = clamp(w, style.min_w, style.max_w);
        h = clamp(h, style.min_h, style.max_h);
        self.nodes[id as usize].intrinsic = (w, h);
        (w, h)
    }

    /// Container intrinsic: children measured bottom-up; absolute
    /// children excluded.
    fn container_intrinsic(
        &mut self,
        id: u32,
        style: &Style,
        _host: &mut dyn Measure,
    ) -> (f32, f32) {
        let dir = style.direction;
        let mut kids = std::mem::take(&mut self.kids_scratch);
        self.child_ids_into(id, &mut kids);
        kids.retain(|&c| !matches!(self.nodes[c as usize].style.position, Position::Absolute(_)));
        let pad = Ax::of(dir, style.padding.horizontal(), style.padding.vertical());
        let mut main = pad.main;
        let mut cross = 0.0f32;
        let mut first = true;
        for c in &*kids {
            // Children have higher indices, so the reverse pass measured
            // them already — just read their stored intrinsic.
            let (cw, ch) = self.nodes[*c as usize].intrinsic;
            let cm = Ax::of(dir, cw, ch);
            let cs = self.nodes[*c as usize].style;
            let m = cm.main + margin_main(&cs, dir);
            let x = cm.cross + margin_cross(&cs, dir);
            match style.display {
                Display::Stack => {
                    main = main.max(pad.main + m);
                    cross = cross.max(x);
                }
                Display::Flex => {
                    main += m + if first { 0.0 } else { style.gap };
                    cross = cross.max(x);
                }
            }
            first = false;
        }
        self.kids_scratch = kids;
        Ax::of(dir, main, cross + pad.cross).wh(dir)
    }

    /// Lay out the whole tree in `root`.
    pub fn solve(&mut self, root: Rect, scale: f32, host: &mut dyn Measure) {
        let mut scrolls = std::mem::take(&mut self.scrolls_scratch);
        scrolls.clear();
        scrolls.extend(self.nodes.iter().map(|n| n.scroll));
        let mut h = SolveHash::from_seed(self.hash.value());
        h.solve_inputs(root, scale, &scrolls);
        let h = h.value();
        self.scrolls_scratch = scrolls;
        if self.last_hash == Some(h) {
            self.stats.cache_hits += 1;
            // Same structure → the previous outputs are still valid.
            if self.last_out.len() == self.nodes.len() {
                for (n, &(abs, content, baseline)) in
                    self.nodes.iter_mut().zip(self.last_out.iter())
                {
                    n.abs = abs;
                    n.content = content;
                    n.baseline = baseline;
                }
            }
            return;
        }
        self.last_hash = Some(h);
        self.stats.solves += 1;
        self.scale = scale;
        let Some(root_i) = self.nodes.iter().position(|n| n.parent == NIL) else {
            return;
        };
        for i in (0..self.nodes.len() as u32).rev() {
            self.measure_node(i, host);
        }
        self.nodes[root_i].abs = root;
        self.layout_children(root_i as u32, host);
        self.last_out = self
            .nodes
            .iter()
            .map(|n| (n.abs, n.content, n.baseline))
            .collect();
    }

    /// A measured leaf re-measured at a definite width (height-for-width).
    pub(crate) fn measured(&self, id: u32, host: &mut dyn Measure, w: Option<f32>) -> LeafSize {
        match self.nodes[id as usize].leaf {
            Leaf::Measured { key } => host.measure(NodeId(id), key, w),
            Leaf::Fixed { w, h, baseline } => LeafSize { w, h, baseline },
            Leaf::None => {
                let (w, h) = self.nodes[id as usize].intrinsic;
                LeafSize {
                    w,
                    h,
                    baseline: None,
                }
            }
        }
    }

    /// One child's border (w,h) inside a content box — for Stack
    /// children and Absolute nodes.
    pub(crate) fn size_in_box_pub(
        &self,
        id: u32,
        cw: f32,
        ch: f32,
        host: &mut dyn Measure,
    ) -> (f32, f32) {
        let s = &self.nodes[id as usize].style;
        let (iw, ih) = self.nodes[id as usize].intrinsic;
        let mut w = match s.width {
            Size::Px(v) => v,
            Size::Fit => iw,
            Size::Fill => cw.max(0.0),
            Size::Percent(p) => cw.max(0.0) * p / 100.0,
        };
        let mut h = match s.height {
            Size::Px(v) => v,
            Size::Fit => ih,
            Size::Fill => ch.max(0.0),
            Size::Percent(p) => ch.max(0.0) * p / 100.0,
        };
        if matches!(self.nodes[id as usize].leaf, Leaf::Measured { .. }) {
            let definite_w = match s.width {
                Size::Fit => None,
                _ => Some(w),
            };
            let m = self.measured(id, host, definite_w);
            if matches!(s.width, Size::Fit) {
                w = m.w;
            }
            if matches!(s.height, Size::Fit) {
                h = m.h;
            }
        }
        if let Some(r) = s.aspect_ratio {
            if matches!(s.height, Size::Fit | Size::Fill)
                && !matches!(s.width, Size::Fit | Size::Fill)
            {
                h = w / r;
            } else if matches!(s.width, Size::Fit | Size::Fill)
                && !matches!(s.height, Size::Fit | Size::Fill)
            {
                w = h * r;
            }
        }
        (clamp(w, s.min_w, s.max_w), clamp(h, s.min_h, s.max_h))
    }

    /// Lay out `id`'s children inside its content box.
    pub(crate) fn layout_children(&mut self, id: u32, host: &mut dyn Measure) {
        let style = self.nodes[id as usize].style;
        let n_abs = self.nodes[id as usize].abs;
        let pad = style.padding;
        let mut content = Rect::new(
            n_abs.x + pad.left,
            n_abs.y + pad.top,
            (n_abs.w - pad.horizontal()).max(0.0),
            (n_abs.h - pad.vertical()).max(0.0),
        );
        // Scroll axes: children lay out unbounded on them.
        let (sx, sy) = match style.overflow {
            Overflow::Scroll { x, y } => {
                let (ox, oy) = self.nodes[id as usize].scroll;
                (x.then_some(ox), y.then_some(oy))
            }
            _ => (None, None),
        };
        if sx.is_some() {
            content.w = f32::INFINITY;
        }
        if sy.is_some() {
            content.h = f32::INFINITY;
        }
        match style.display {
            Display::Stack => super::stack::layout_stack(self, id, content, host),
            Display::Flex => super::flex::layout_flex(self, id, content, host),
        }
        // Absolute children sit in the padding box.
        for c in self.child_ids(id) {
            if let Position::Absolute(e) = self.nodes[c as usize].style.position {
                let pb = Rect::new(
                    n_abs.x + pad.left,
                    n_abs.y + pad.top,
                    (n_abs.w - pad.horizontal()).max(0.0),
                    (n_abs.h - pad.vertical()).max(0.0),
                );
                let (w, h) = self.size_in_box_pub(c, pb.w, pb.h, host);
                let w = match (e.left, e.right) {
                    (Some(l), Some(r)) => (pb.w - l - r).max(0.0),
                    _ => w,
                };
                let h = match (e.top, e.bottom) {
                    (Some(t), Some(b)) => (pb.h - t - b).max(0.0),
                    _ => h,
                };
                let x = pb.x + e.left.unwrap_or(pb.w - e.right.unwrap_or(0.0) - w);
                let y = pb.y + e.top.unwrap_or(pb.h - e.bottom.unwrap_or(0.0) - h);
                self.nodes[c as usize].abs = Rect::new(x, y, w, h);
                self.layout_children(c, host);
            }
        }
        // Baseline: the node's own leaf, else its first child's.
        if self.nodes[id as usize].baseline.is_none()
            && let Leaf::Fixed { baseline, .. } = self.nodes[id as usize].leaf
        {
            self.nodes[id as usize].baseline = baseline.map(|b| n_abs.y + b);
        }
        if self.nodes[id as usize].baseline.is_none()
            && let Some(c) = self.child_ids(id).into_iter().next()
        {
            self.nodes[id as usize].baseline = self.nodes[c as usize].baseline;
        }
    }

    /// Leaf baseline offset from the leaf's top.
    pub(crate) fn leaf_baseline(&self, id: u32) -> Option<f32> {
        match self.nodes[id as usize].leaf {
            Leaf::Fixed { baseline, .. } => baseline,
            _ => None,
        }
    }
}

pub(crate) fn clamp(v: f32, min: Option<f32>, max: Option<f32>) -> f32 {
    let v = min.map_or(v, |l| v.max(l));
    max.map_or(v, |l| v.min(l))
}

pub(crate) fn margin_main(s: &Style, dir: Direction) -> f32 {
    match dir {
        Direction::Row => s.margin.horizontal(),
        Direction::Column => s.margin.vertical(),
    }
}

pub(crate) fn margin_cross(s: &Style, dir: Direction) -> f32 {
    match dir {
        Direction::Row => s.margin.vertical(),
        Direction::Column => s.margin.horizontal(),
    }
}
