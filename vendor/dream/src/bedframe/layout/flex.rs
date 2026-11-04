//! Flex display layout: lines, grow/shrink, justify, cross align.

use super::Rect;
use super::measure::{Leaf, Measure};
use super::solver::{LayoutTree, NodeId, clamp, margin_cross, margin_main};
use super::style::{Align, Direction, Justify, Position, Size, Style};

fn min_main(s: &Style, dir: Direction) -> Option<f32> {
    match dir {
        Direction::Row => s.min_w,
        Direction::Column => s.min_h,
    }
}
fn max_main(s: &Style, dir: Direction) -> Option<f32> {
    match dir {
        Direction::Row => s.max_w,
        Direction::Column => s.max_h,
    }
}
fn min_cross(s: &Style, dir: Direction) -> Option<f32> {
    match dir {
        Direction::Row => s.min_h,
        Direction::Column => s.min_w,
    }
}
fn max_cross(s: &Style, dir: Direction) -> Option<f32> {
    match dir {
        Direction::Row => s.max_h,
        Direction::Column => s.max_w,
    }
}
fn margin_main_start(s: &Style, dir: Direction) -> f32 {
    match dir {
        Direction::Row => s.margin.left,
        Direction::Column => s.margin.top,
    }
}
fn margin_cross_start(s: &Style, dir: Direction) -> f32 {
    match dir {
        Direction::Row => s.margin.top,
        Direction::Column => s.margin.left,
    }
}
fn main_of(dir: Direction, w: f32, h: f32) -> f32 {
    match dir {
        Direction::Row => w,
        Direction::Column => h,
    }
}
fn cross_of(dir: Direction, w: f32, h: f32) -> f32 {
    match dir {
        Direction::Row => h,
        Direction::Column => w,
    }
}

struct Item {
    id: u32,
    hypo: f32,
    main: f32,
    cross: f32,
    grow: f32,
    shrink: f32,
    basis: f32,
    m_main: f32,
    m_cross_start: f32,
    m_cross: f32,
    baseline: Option<f32>,
    align: Align,
    stretchable: bool,
    /// Cross `Size::Fill`: fills the cross like stretch, still aligned.
    fill_cross: bool,
}

/// Lay out `id`'s (non-absolute) children inside `content`.
pub(super) fn layout_flex(tree: &mut LayoutTree, id: u32, content: Rect, host: &mut dyn Measure) {
    let style = *tree.style(NodeId(id));
    let dir = style.direction;
    let inner_main = main_of(dir, content.w, content.h);
    let inner_cross = cross_of(dir, content.w, content.h);
    let kids: Vec<u32> = tree
        .child_ids(id)
        .into_iter()
        .filter(|&c| !matches!(tree.node(c).style.position, Position::Absolute(_)))
        .collect();

    // Hypothetical main sizes.
    let mut items: Vec<Item> = Vec::with_capacity(kids.len());
    for &c in &kids {
        let s = tree.node(c).style;
        let (iw, ih) = tree.node(c).intrinsic;
        let size_main = match dir {
            Direction::Row => size_v(s.width),
            Direction::Column => size_v(s.height),
        };
        let size_cross = match dir {
            Direction::Row => size_v(s.height),
            Direction::Column => size_v(s.width),
        };
        let (mut grow, mut basis) = (s.grow, s.basis.unwrap_or(main_of(dir, iw, ih)));
        if matches!(size_main, ST::Fill) {
            grow = grow.max(1.0);
            basis = 0.0;
        }
        let hypo = match size_main {
            ST::Px(v) => v,
            ST::Fill => basis,
            ST::Percent(p) if inner_main.is_finite() => inner_main * p / 100.0,
            _ => basis,
        };
        let cross = match size_cross {
            ST::Px(v) => v,
            ST::Percent(p) if inner_cross.is_finite() => inner_cross * p / 100.0,
            _ => cross_of(dir, iw, ih),
        };
        let stretchable = matches!(size_cross, ST::Fit | ST::Fill);
        items.push(Item {
            id: c,
            hypo: clamp(hypo, min_main(&s, dir), max_main(&s, dir)),
            main: 0.0,
            cross: clamp(cross, min_cross(&s, dir), max_cross(&s, dir)),
            grow,
            shrink: s.shrink,
            basis: hypo,
            m_main: margin_main(&s, dir),
            m_cross_start: margin_cross_start(&s, dir),
            m_cross: margin_cross(&s, dir),
            baseline: tree.leaf_baseline(c),
            align: s.align_self.unwrap_or(style.align_items),
            stretchable,
            fill_cross: matches!(size_cross, ST::Fill),
        });
        let last = items.len() - 1;
        items[last].main = items[last].hypo;
    }

    // Lines.
    let gap = style.gap.max(0.0);
    let mut lines: Vec<Vec<usize>> = Vec::new();
    if style.wrap && inner_main.is_finite() {
        let mut line: Vec<usize> = Vec::new();
        let mut used = 0.0f32;
        for (i, it) in items.iter().enumerate() {
            let outer = it.hypo + it.m_main;
            if !line.is_empty() && used + gap + outer > inner_main {
                lines.push(std::mem::take(&mut line));
                used = 0.0;
            }
            used += if line.is_empty() { outer } else { outer + gap };
            line.push(i);
        }
        if !line.is_empty() {
            lines.push(line);
        }
    } else {
        lines.push((0..items.len()).collect());
    }

    let mut cross_cursor = 0.0f32;
    let mut content_main = 0.0f32;
    let mut content_cross = 0.0f32;
    let n_lines = lines.len();
    for line in &lines {
        // Free space.
        let used: f32 = line
            .iter()
            .map(|&i| items[i].hypo + items[i].m_main)
            .sum::<f32>()
            + gap * (line.len().saturating_sub(1) as f32);
        let mut free = if inner_main.is_finite() {
            inner_main - used
        } else {
            0.0
        };
        // Grow.
        let total_grow: f32 = line.iter().map(|&i| items[i].grow.max(0.0)).sum();
        if free > 0.0 && total_grow > 0.0 {
            // Grow on the *basis*; clamp violators freeze and the rest
            // redistributes (CSS §9.7).
            let mut unfrozen: Vec<usize> = line
                .iter()
                .copied()
                .filter(|&i| items[i].grow > 0.0)
                .collect();
            let mut left = free;
            let mut frozen: Vec<usize> = Vec::new();
            for _ in 0..8 {
                if unfrozen.is_empty() || left <= 1e-4 {
                    break;
                }
                let gsum: f32 = unfrozen.iter().map(|&i| items[i].grow).sum();
                let mut froze = false;
                let mut next = Vec::new();
                for &i in &unfrozen {
                    let s = tree.node(items[i].id).style;
                    let target = items[i].basis + left * items[i].grow / gsum;
                    let c = clamp(target, min_main(&s, dir), max_main(&s, dir));
                    if (c - target).abs() > 1e-4 {
                        items[i].main = c;
                        froze = true;
                        frozen.push(i);
                    } else {
                        next.push(i);
                    }
                }
                if !froze {
                    for &i in &unfrozen {
                        items[i].main = items[i].basis + left * items[i].grow / gsum;
                    }
                    break;
                }
                // Refund the space a frozen item's clamped size doesn't
                // consume (its hypo already counted in `free`).
                left = free
                    - frozen
                        .iter()
                        .map(|&i| items[i].main - items[i].hypo)
                        .sum::<f32>();
                unfrozen = next;
            }
            free = 0.0;
        } else {
            for &i in line {
                items[i].main = items[i].hypo;
            }
        }
        // Shrink, freezing violators (CSS §9.7 loop).
        if free < 0.0 {
            let mut deficit = -free;
            let mut unfrozen: Vec<usize> = line
                .iter()
                .copied()
                .filter(|&i| items[i].shrink > 0.0)
                .collect();
            for _ in 0..8 {
                if unfrozen.is_empty() || deficit <= 1e-4 {
                    break;
                }
                let weight: f32 = unfrozen
                    .iter()
                    .map(|&i| items[i].shrink * items[i].basis.max(0.0))
                    .sum();
                if weight <= 0.0 {
                    break;
                }
                let mut froze = false;
                let mut next = Vec::new();
                for &i in &unfrozen {
                    let s = tree.node(items[i].id).style;
                    let take = deficit * items[i].shrink * items[i].basis.max(0.0) / weight;
                    let target = items[i].hypo - take;
                    let c = clamp(target, min_main(&s, dir), None);
                    if c > target + 1e-4 {
                        items[i].main = c;
                        deficit -= items[i].hypo - c;
                        froze = true;
                    } else {
                        next.push(i);
                    }
                }
                if !froze {
                    for &i in &unfrozen {
                        items[i].main = items[i].hypo
                            - deficit * items[i].shrink * items[i].basis.max(0.0) / weight;
                    }
                    break;
                }
                unfrozen = next;
            }
            for &i in line {
                let s = tree.node(items[i].id).style;
                items[i].main = clamp(items[i].main, min_main(&s, dir), max_main(&s, dir));
            }
        }

        // Measured children in a Column parent re-measure their height at
        // the resolved width (height-for-width text).
        for &i in line {
            let c = items[i].id;
            if matches!(dir, Direction::Column)
                && matches!(tree.node(c).leaf, Leaf::Measured { .. })
            {
                let w = match items[i].align {
                    Align::Stretch => inner_cross - items[i].m_cross,
                    _ => items[i].cross,
                }
                .max(0.0);
                let m = tree.measured(c, host, Some(w));
                items[i].main = m.h;
                items[i].hypo = m.h;
            }
        }

        // Line metrics.
        let line_outer: f32 = line
            .iter()
            .map(|&i| items[i].main + items[i].m_main)
            .sum::<f32>()
            + gap * (line.len().saturating_sub(1) as f32);
        let line_cross_max = line
            .iter()
            .map(|&i| items[i].cross + items[i].m_cross)
            .fold(0.0f32, f32::max);
        let line_baseline = line
            .iter()
            .filter_map(|&i| {
                if items[i].align == Align::Baseline {
                    items[i].baseline.or(Some(items[i].cross))
                } else {
                    None
                }
            })
            .fold(None, |acc: Option<f32>, b| {
                Some(acc.map_or(b, |a| a.max(b)))
            });
        content_main = content_main.max(line_outer);

        // Justify.
        let remaining = if inner_main.is_finite() {
            (inner_main - line_outer).max(0.0)
        } else {
            0.0
        };
        let (mut cursor, extra) = match style.justify {
            Justify::Start => (0.0, 0.0),
            Justify::Center => (remaining * 0.5, 0.0),
            Justify::End => (remaining, 0.0),
            Justify::SpaceBetween => (
                0.0,
                if line.len() > 1 {
                    remaining / (line.len() - 1) as f32
                } else {
                    0.0
                },
            ),
            Justify::SpaceAround => (
                remaining / (2.0 * line.len() as f32),
                remaining / line.len() as f32,
            ),
            Justify::SpaceEvenly => {
                let g = remaining / (line.len() + 1) as f32;
                (g, g)
            }
        };

        let line_cross_size = if n_lines == 1 && inner_cross.is_finite() {
            inner_cross
        } else {
            line_cross_max
        };

        for (k, &i) in line.iter().enumerate() {
            let it = &items[i];
            let s = tree.node(it.id).style;
            let main_pos = cursor + margin_main_start(&s, dir);
            // Cross size: Fit/Percent/Px keep the measured value; Stretch
            // (or a `Fill` cross under any align) fills the line, clamped
            // by min/max (CSS clamps the stretched size, never overrides).
            let avail = (line_cross_size - it.m_cross).max(0.0);
            let fills = it.fill_cross
                || (it.align == Align::Stretch && it.stretchable && it.align != Align::Baseline);
            let cross_size = if fills && it.align != Align::Baseline {
                clamp(avail, min_cross(&s, dir), max_cross(&s, dir))
            } else {
                it.cross
            };
            let cross_pos = match it.align {
                Align::Start | Align::Stretch => cross_cursor + it.m_cross_start,
                Align::Center => cross_cursor + it.m_cross_start + (avail - cross_size) * 0.5,
                Align::End => cross_cursor + it.m_cross_start + (avail - cross_size),
                Align::Baseline => {
                    let lb = line_baseline.unwrap_or(it.cross);
                    let ib = it.baseline.unwrap_or(it.cross);
                    cross_cursor + it.m_cross_start + lb - ib
                }
            };
            let (x, y, w, h) = match dir {
                Direction::Row => (
                    content.x + main_pos,
                    content.y + cross_pos,
                    it.main,
                    cross_size,
                ),
                Direction::Column => (
                    content.x + cross_pos,
                    content.y + main_pos,
                    cross_size,
                    it.main,
                ),
            };
            tree.node_mut(it.id).abs = Rect::new(x, y, w, h);
            tree.node_mut(it.id).baseline = it.baseline.map(|b| y + b);
            cursor += it.main + it.m_main + gap + if k + 1 < line.len() { extra } else { 0.0 };
            tree.layout_children(it.id, host);
        }
        cross_cursor += line_cross_size + style.cross_gap;
        content_cross = content_cross.max(line_cross_size);
    }
    if n_lines > 1 {
        content_cross = cross_cursor - style.cross_gap;
    }
    let (cw, ch) = match dir {
        Direction::Row => (content_main, content_cross),
        Direction::Column => (content_cross, content_main),
    };
    let p = style.padding;
    tree.node_mut(id).content = (cw + p.horizontal(), ch + p.vertical());

    // Scroll: clamp offset, shift children along the scrolled axes.
    if let super::Overflow::Scroll { x: sx, y: sy } = style.overflow {
        let n = tree.node_mut(id);
        let (ox, oy) = n.scroll;
        let n_abs = n.abs;
        let pad = n.style.padding;
        let vp_w = (n_abs.w - pad.horizontal()).max(0.0);
        let vp_h = (n_abs.h - pad.vertical()).max(0.0);
        let cx = if sx {
            (n.content.0 - vp_w).max(0.0)
        } else {
            0.0
        };
        let cy = if sy {
            (n.content.1 - vp_h).max(0.0)
        } else {
            0.0
        };
        let (ox, oy) = (ox.clamp(0.0, cx), oy.clamp(0.0, cy));
        n.scroll = (ox, oy);
        if ox != 0.0 || oy != 0.0 {
            for c in tree.child_ids(id) {
                let a = &mut tree.node_mut(c).abs;
                a.x -= if sx { ox } else { 0.0 };
                a.y -= if sy { oy } else { 0.0 };
                if let Some(b) = tree.node_mut(c).baseline {
                    tree.node_mut(c).baseline = Some(b - if sy { oy } else { 0.0 });
                }
            }
        }
    }
}

enum ST {
    Px(f32),
    Fill,
    Percent(f32),
    Fit,
}

fn size_v(s: Size) -> ST {
    match s {
        Size::Px(v) => ST::Px(v),
        Size::Fill => ST::Fill,
        Size::Percent(p) => ST::Percent(p),
        Size::Fit => ST::Fit,
    }
}
