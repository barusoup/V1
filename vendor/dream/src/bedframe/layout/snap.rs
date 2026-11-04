//! Physical-pixel snapping: UI geometry is authored in logical units but
//! rasterized at `scale` physical pixels per logical pixel. Edges that sit
//! off the physical grid smear a hairline into a blur, so layout snaps
//! absolute edges — never sizes — and derives sizes from the snapped edges.

use super::Rect;

/// Snap one absolute coordinate to the physical pixel grid.
pub fn snap_edge(v: f32, scale: f32) -> f32 {
    if scale <= 0.0 || !scale.is_finite() || !v.is_finite() {
        return v;
    }
    (v * scale).round() / scale
}

/// Snap all four absolute edges of `rect` to the physical grid. Sizes are
/// derived from the snapped edges, so a rect that tiles with its neighbours
/// still tiles after snapping.
pub fn snap_rect(rect: Rect, scale: f32) -> Rect {
    let x0 = snap_edge(rect.x, scale);
    let y0 = snap_edge(rect.y, scale);
    let x1 = snap_edge(rect.x + rect.w, scale);
    let y1 = snap_edge(rect.y + rect.h, scale);
    Rect::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
}

/// The logical width of `round(scale)` physical pixels — a hairline that
/// always spans a whole number of physical pixels: 1 px at 1x/1.25x, 2 px
/// at 2x, 3 px at 3x.
pub fn hairline(scale: f32) -> f32 {
    if scale <= 0.0 || !scale.is_finite() {
        return 1.0;
    }
    scale.round().max(1.0) / scale
}

/// Divide `extent` (starting at `start`) into `count` slots separated by
/// `gap`, snapping every absolute edge to the physical grid. Slots tile
/// exactly: each is `(pos, size)` with `pos` on the grid, `pos + size`
/// on the grid, and consecutive slots separated by exactly `gap` physical
/// pixels. Sizes differ by at most one physical pixel.
/// `divide` writing into a reusable buffer (cleared first, same math).
pub fn divide_into(
    out: &mut Vec<(f32, f32)>,
    start: f32,
    extent: f32,
    count: usize,
    gap: f32,
    scale: f32,
) {
    out.clear();
    if count == 0 {
        return;
    }
    let s = f64::from(scale);
    let first = (f64::from(start) * s).round() as i64;
    let last = (f64::from(start + extent) * s).round() as i64;
    let gap_px = (f64::from(gap) * s).round() as i64;
    let n = count as i64;
    let budget = (last - first - gap_px * (n - 1)).max(0);
    out.reserve(count);
    let mut pos = first;
    for i in 0..count {
        let i = i as i64;
        let end = first + (i + 1) * budget / n + gap_px * i;
        out.push(((pos as f64 / s) as f32, ((end - pos) as f64 / s) as f32));
        pos = end + gap_px;
    }
}

/// Evenly divide `extent` into `count` `(pos, size)` slots snapped to the
/// physical pixel grid — same math as [`divide_into`], allocated.
pub fn divide(start: f32, extent: f32, count: usize, gap: f32, scale: f32) -> Vec<(f32, f32)> {
    let mut out = Vec::with_capacity(count);
    divide_into(&mut out, start, extent, count, gap, scale);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }

        fn f32(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        fn usize(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn snap_divide_tiles_exactly_on_the_grid() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for _ in 0..2000 {
            let scale = [1.0f32, 1.25, 1.5, 2.0, 3.0][rng.usize(5)];
            let start = rng.f32(-50.0, 500.0);
            let extent = rng.f32(1.0, 600.0);
            let count = rng.usize(12) + 1;
            let gap = rng.f32(0.0, 8.0);
            // Gaps that alone overflow the extent can't tile inside it.
            if gap * (count - 1) as f32 >= extent {
                continue;
            }
            let slots = divide(start, extent, count, gap, scale);
            assert_eq!(slots.len(), count);
            // Tolerance is f32-combination noise (well under half a px —
            // an off-grid edge would be ~0.5 out).
            let on_grid = |v: f32| ((v * scale) - (v * scale).round()).abs() < 1e-2;
            for &(pos, size) in &slots {
                assert!(on_grid(pos), "pos {pos} off grid at scale {scale}");
                assert!(on_grid(pos + size), "end off grid at scale {scale}");
            }
            // Compare in rounded physical pixels: recombining pos+size in
            // f32 drifts a few ulp off the exact logical edge.
            assert!(
                (slots[0].0 - snap_edge(start, scale)).abs() < 1e-4,
                "first slot starts at snap(start)"
            );
            let last = slots.last().unwrap();
            assert_eq!(
                ((last.0 + last.1) * scale).round(),
                (snap_edge(start + extent, scale) * scale).round(),
                "last slot ends at snap(start+extent)"
            );
            let (mut min, mut max) = (f32::MAX, 0.0f32);
            for w in slots.windows(2) {
                let gap_px = (w[1].0 * scale).round() - ((w[0].0 + w[0].1) * scale).round();
                let want = (gap * scale).round();
                assert_eq!(gap_px, want, "slots separated by exactly gap px");
                min = min.min(w[0].1);
                max = max.max(w[0].1);
            }
            min = min.min(slots.last().unwrap().1);
            max = max.max(slots.last().unwrap().1);
            assert!(
                (max - min) * scale <= 1.0 + 1e-4,
                "sizes differ by <= 1 physical px: {min}..{max} at scale {scale}"
            );
        }
    }

    #[test]
    fn snap_hairline_is_whole_physical_px() {
        for scale in [1.0, 1.25, 1.5, 1.75, 2.0, 3.0] {
            let w = hairline(scale) * scale;
            assert!(
                (w - w.round()).abs() < 1e-4 && w >= 1.0,
                "hairline at scale {scale} is {w} physical px"
            );
        }
    }
}
