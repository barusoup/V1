//! Solver tests: exact-rect assertions hand-worked from CSS flexbox.
use super::measure::{Leaf, LeafSize, Measure};
use super::solver::{LayoutTree, NodeId};
use super::style::{Align, Edges, Justify, Size, Style};
use super::{AlignX, AlignY, Rect};

struct Null;
impl Measure for Null {
    fn measure(&mut self, _: NodeId, _: u64, _: Option<f32>) -> LeafSize {
        LeafSize::default()
    }
}

fn fixed(w: f32, h: f32) -> Leaf {
    Leaf::Fixed {
        w,
        h,
        baseline: None,
    }
}

fn solve(t: &mut LayoutTree, w: f32, h: f32) {
    t.solve(Rect::new(0.0, 0.0, w, h), 1.0, &mut Null);
}

#[test]
fn layout_flex_fill_splits_free_space() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row().pad(10.0).gap(10.0), Leaf::None);
    let a = t.push(
        Some(r),
        Style::row().w_px(50.0).h_px(20.0),
        fixed(50.0, 20.0),
    );
    let b = t.push(Some(r), Style::row().fill().h_px(20.0), fixed(10.0, 20.0));
    let c = t.push(
        Some(r),
        Style::row().fill().h_px(20.0).grow(2.0),
        fixed(10.0, 20.0),
    );
    solve(&mut t, 300.0, 40.0);
    assert_eq!(t.rect(a), Rect::new(10.0, 10.0, 50.0, 20.0));
    assert_eq!(t.rect(b), Rect::new(70.0, 10.0, 70.0, 20.0));
    assert_eq!(t.rect(c), Rect::new(150.0, 10.0, 140.0, 20.0));
}

#[test]
fn layout_flex_shrink_weighted_by_basis() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row(), Leaf::None);
    let a = t.push(
        Some(r),
        Style::row().w_px(150.0).h_px(10.0),
        fixed(150.0, 10.0),
    );
    let b = t.push(
        Some(r),
        Style::row().w_px(150.0).h_px(10.0),
        fixed(150.0, 10.0),
    );
    solve(&mut t, 200.0, 20.0);
    assert_eq!(t.rect(a).w, 100.0);
    assert_eq!(t.rect(b), Rect::new(100.0, 0.0, 100.0, 10.0));
}

#[test]
fn layout_flex_min_w_freezes_and_redistributes() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row(), Leaf::None);
    let a = t.push(
        Some(r),
        Style::row().fill().min_w(100.0).h_px(10.0),
        fixed(10.0, 10.0),
    );
    let b = t.push(Some(r), Style::row().fill().h_px(10.0), fixed(10.0, 10.0));
    solve(&mut t, 150.0, 20.0);
    assert_eq!(t.rect(a).w, 100.0);
    assert_eq!(t.rect(b), Rect::new(100.0, 0.0, 50.0, 10.0));
}

#[test]
fn layout_flex_justify_variants() {
    for (j, x1) in [
        (Justify::Start, 40.0),
        (Justify::Center, 60.0),
        (Justify::End, 80.0),
        (Justify::SpaceBetween, 80.0),
        (Justify::SpaceAround, 73.4),
        (Justify::SpaceEvenly, 70.0),
    ] {
        let mut t = LayoutTree::new();
        let r = t.push(None, Style::row().justify(j), Leaf::None);
        let mut ids = Vec::new();
        for _ in 0..3 {
            ids.push(t.push(Some(r), Style::row().w_px(20.0), fixed(20.0, 10.0)));
        }
        solve(&mut t, 100.0, 20.0);
        assert!(
            (t.rect(ids[2]).x - x1).abs() < 0.5,
            "{j:?} last x {:?}",
            t.rect(ids[2])
        );
    }
}

#[test]
fn layout_flex_align_items_and_self() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row().align(Align::Center), Leaf::None);
    let a = t.push(
        Some(r),
        Style::row().w_px(20.0).h_px(10.0),
        fixed(20.0, 10.0),
    );
    let b = t.push(
        Some(r),
        Style::row().w_px(20.0).h_px(10.0).align_self(Align::End),
        fixed(20.0, 10.0),
    );
    solve(&mut t, 100.0, 30.0);
    assert_eq!(t.rect(a).y, 10.0);
    assert_eq!(t.rect(b).y, 20.0);
}

#[test]
fn layout_flex_stretch_is_default_cross() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row(), Leaf::None);
    let a = t.push(Some(r), Style::row().w_px(20.0), fixed(20.0, 5.0));
    solve(&mut t, 100.0, 30.0);
    assert_eq!(t.rect(a), Rect::new(0.0, 0.0, 20.0, 30.0));
}

#[test]
fn layout_flex_fit_container_wraps_children() {
    let mut t = LayoutTree::new();
    let outer = t.push(None, Style::column(), Leaf::None);
    let inner = t.push(Some(outer), Style::row().gap(5.0).pad(2.0), Leaf::None);
    t.push(
        Some(inner),
        Style::row().w_px(20.0).h_px(10.0),
        fixed(20.0, 10.0),
    );
    t.push(
        Some(inner),
        Style::row().w_px(30.0).h_px(15.0),
        fixed(30.0, 15.0),
    );
    solve(&mut t, 100.0, 100.0);
    assert_eq!(t.rect(inner), Rect::new(0.0, 0.0, 100.0, 19.0));
}

#[test]
fn layout_flex_percent_and_aspect() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::column(), Leaf::None);
    let a = t.push(
        Some(r),
        Style::column().w(Size::Percent(50.0)).h_px(10.0),
        fixed(5.0, 5.0),
    );
    let b = t.push(
        Some(r),
        Style::column().w_px(40.0).aspect(2.0),
        fixed(40.0, 40.0),
    );
    solve(&mut t, 200.0, 100.0);
    assert_eq!(t.rect(a).w, 100.0);
    assert_eq!(t.rect(b), Rect::new(0.0, 10.0, 40.0, 20.0));
}

struct Text {
    text_w: f32,
    line: f32,
}
impl Measure for Text {
    fn measure(&mut self, _: NodeId, _: u64, avail_w: Option<f32>) -> LeafSize {
        match avail_w {
            None => LeafSize {
                w: self.text_w,
                h: self.line,
                baseline: None,
            },
            Some(w) => LeafSize {
                w,
                h: (self.text_w / w.max(1.0)).ceil() * self.line,
                baseline: None,
            },
        }
    }
}

#[test]
fn layout_flex_height_for_width_measured() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::column(), Leaf::None);
    let txt = t.push(Some(r), Style::column().key("t"), Leaf::Measured { key: 1 });
    t.solve(
        Rect::new(0.0, 0.0, 100.0, 50.0),
        1.0,
        &mut Text {
            text_w: 250.0,
            line: 10.0,
        },
    );
    assert_eq!(t.rect(txt), Rect::new(0.0, 0.0, 100.0, 30.0));
}

#[test]
fn layout_wrap_breaks_and_cross_gaps() {
    let mut t = LayoutTree::new();
    let r = t.push(
        None,
        Style::row().wrap(true).gap(10.0).cross_gap(5.0),
        Leaf::None,
    );
    let mut ids = Vec::new();
    for _ in 0..5 {
        ids.push(t.push(
            Some(r),
            Style::row().w_px(80.0).h_px(20.0),
            fixed(80.0, 20.0),
        ));
    }
    solve(&mut t, 300.0, 100.0);
    assert_eq!(t.rect(ids[3]), Rect::new(0.0, 25.0, 80.0, 20.0));
    assert_eq!(t.rect(ids[4]), Rect::new(90.0, 25.0, 80.0, 20.0));
}

#[test]
fn layout_baseline_aligns_first_baselines() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row().align(Align::Baseline), Leaf::None);
    let a = t.push(
        Some(r),
        Style::row().w_px(20.0).h_px(30.0),
        Leaf::Fixed {
            w: 20.0,
            h: 30.0,
            baseline: Some(20.0),
        },
    );
    let b = t.push(
        Some(r),
        Style::row().w_px(20.0).h_px(15.0),
        Leaf::Fixed {
            w: 20.0,
            h: 15.0,
            baseline: Some(10.0),
        },
    );
    solve(&mut t, 100.0, 40.0);
    assert_eq!(t.rect(a).y, 0.0);
    assert_eq!(t.rect(b).y, 10.0);
    assert_eq!(t.baseline(a), Some(20.0));
    assert_eq!(t.baseline(b), Some(20.0));
}

#[test]
fn layout_stack_places_children() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::stack(), Leaf::None);
    let c = t.push(
        Some(r),
        Style::stack()
            .w_px(20.0)
            .h_px(10.0)
            .place(AlignX::End, AlignY::Center),
        fixed(20.0, 10.0),
    );
    solve(&mut t, 100.0, 50.0);
    assert_eq!(t.rect(c), Rect::new(80.0, 20.0, 20.0, 10.0));
}

#[test]
fn layout_abs_insets() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::column().pad(10.0), Leaf::None);
    let a = t.push(
        Some(r),
        Style::column().absolute(Edges {
            top: None,
            right: Some(0.0),
            bottom: Some(0.0),
            left: Some(0.0),
        }),
        fixed(10.0, 20.0),
    );
    let b = t.push(
        Some(r),
        Style::column().absolute(Edges {
            top: Some(5.0),
            right: Some(0.0),
            bottom: None,
            left: None,
        }),
        fixed(15.0, 10.0),
    );
    solve(&mut t, 100.0, 60.0);
    assert_eq!(t.rect(a), Rect::new(10.0, 30.0, 80.0, 20.0));
    assert_eq!(t.rect(b), Rect::new(75.0, 15.0, 15.0, 10.0));
}

#[test]
fn layout_scroll_extends_and_offsets() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::column().scroll_y(), Leaf::None);
    let mut ids = Vec::new();
    for _ in 0..10 {
        ids.push(t.push(Some(r), Style::column().h_px(40.0), fixed(20.0, 40.0)));
    }
    solve(&mut t, 100.0, 100.0);
    assert_eq!(t.content_size(r).1, 400.0);
    t.set_scroll_offset(r, (0.0, 50.0));
    solve(&mut t, 100.0, 100.0);
    assert_eq!(t.rect(ids[0]).y, -50.0);
    t.set_scroll_offset(r, (0.0, 9999.0));
    solve(&mut t, 100.0, 100.0);
    assert_eq!(t.rect(ids[0]).y, -300.0);
}

#[test]
fn layout_snap_shared_edges_at_fractional_scale() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row(), Leaf::None);
    let mut ids = Vec::new();
    for _ in 0..3 {
        ids.push(t.push(Some(r), Style::row().fill(), fixed(10.0, 10.0)));
    }
    t.solve(Rect::new(0.3, 0.0, 100.4, 10.0), 1.5, &mut Null);
    let (r0, r1, r2) = (t.rect(ids[0]), t.rect(ids[1]), t.rect(ids[2]));
    assert_eq!(r0.max().0, r1.x);
    assert_eq!(r1.max().0, r2.x);
    for r in [r0, r1, r2] {
        for e in [r.x, r.max().0] {
            assert!((e * 1.5).fract().abs() < 1e-4);
        }
    }
}

#[test]
fn layout_cache_hits_and_invalidates() {
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row(), Leaf::None);
    let a = t.push(Some(r), Style::row().fill(), fixed(10.0, 10.0));
    solve(&mut t, 100.0, 20.0);
    let first = t.rect(a);
    t.clear();
    let r = t.push(None, Style::row(), Leaf::None);
    let a2 = t.push(Some(r), Style::row().fill(), fixed(10.0, 10.0));
    solve(&mut t, 100.0, 20.0);
    assert_eq!(t.stats().cache_hits, 1);
    assert_eq!(t.rect(a2), first);
    solve(&mut t, 200.0, 20.0);
    assert_eq!(t.stats().solves, 2);
}

/// 2,000-node tree, cold vs cached solve.
#[test]
#[ignore]
fn layout_solver_profile() {
    let mut t = LayoutTree::new();
    let root = t.push(None, Style::column(), Leaf::None);
    let mut parents = vec![root];
    while t.len() < 2000 {
        let p = parents[t.len() % parents.len()];
        let c = t.push(
            Some(p),
            if t.len().is_multiple_of(2) {
                Style::row()
            } else {
                Style::column()
            },
            Leaf::None,
        );
        parents.push(c);
        t.push(Some(c), Style::row().fill().min_h(8.0), fixed(10.0, 8.0));
    }
    let start = std::time::Instant::now();
    t.solve(Rect::new(0.0, 0.0, 800.0, 600.0), 1.0, &mut Null);
    let cold = start.elapsed();
    let start = std::time::Instant::now();
    t.solve(Rect::new(0.0, 0.0, 800.0, 600.0), 1.0, &mut Null);
    let cached = start.elapsed();
    println!(
        "layout_solver_profile: nodes={} cold={:?} cached={:?} stats={:?}",
        t.len(),
        cold,
        cached,
        t.stats()
    );
}

#[test]
fn layout_flex_stretch_respects_min_max() {
    // Column 1000 wide; a stretched child with max_w clamps to 300 at x 0.
    for (align, want_x) in [(Align::Stretch, 0.0), (Align::Center, 350.0)] {
        let mut t = LayoutTree::new();
        let r = t.push(None, Style::column().pad(0.0).align(align), Leaf::None);
        let c = t.push(
            Some(r),
            Style::default().fill_w().h_px(20.0).max_w(300.0),
            fixed(10.0, 10.0),
        );
        solve(&mut t, 1000.0, 400.0);
        let rect = t.rect(c);
        assert_eq!(rect.w, 300.0, "stretch must clamp to max_w ({align:?})");
        assert_eq!(rect.x, want_x, "{align:?}");
    }
}

#[test]
fn layout_flex_fill_cross_respects_min_max() {
    // Row 1000 high; a fill-height child with max_h clamps to 200.
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row().pad(0.0), Leaf::None);
    let c = t.push(
        Some(r),
        Style::default().w_px(40.0).fill_h().max_h(200.0),
        fixed(10.0, 10.0),
    );
    solve(&mut t, 300.0, 1000.0);
    assert_eq!(t.rect(c).h, 200.0);
    // min_h clamps stretch upward.
    let mut t = LayoutTree::new();
    let r = t.push(None, Style::row().pad(0.0), Leaf::None);
    let c = t.push(
        Some(r),
        Style::default().w_px(40.0).fill_h().min_h(1200.0),
        fixed(10.0, 10.0),
    );
    solve(&mut t, 300.0, 1000.0);
    assert_eq!(t.rect(c).h, 1200.0);
}
