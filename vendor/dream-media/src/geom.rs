//! Small integer geometry helpers, kept exact and allocation-free.

/// Fit `width` x `height` inside `max_w` x `max_h` without changing the aspect
/// ratio and without enlarging. A zero bound means "unbounded"; zero input
/// dimensions collapse to 1x1 so no caller can build an empty Vulkan image.
pub(crate) fn fit_box(width: u32, height: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (1, 1);
    }
    if max_w == 0 || max_h == 0 || (width <= max_w && height <= max_h) {
        return (width, height);
    }
    let (mut w, mut h) = (u64::from(width), u64::from(height));
    let (box_w, box_h) = (u64::from(max_w), u64::from(max_h));
    if w * box_h > h * box_w {
        h = (h * box_w / w).max(1);
        w = box_w;
    } else {
        w = (w * box_h / h).max(1);
        h = box_h;
    }
    (w as u32, h as u32)
}

/// Fit a box inside a square maximum side. `max_side == 0` is unbounded.
pub(crate) fn fit_side(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    if max_side == 0 {
        (width.max(1), height.max(1))
    } else {
        fit_box(width, height, max_side, max_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_aspect_and_never_upscales() {
        assert_eq!(fit_box(1920, 1080, 1280, 720), (1280, 720));
        assert_eq!(fit_box(1080, 1920, 1280, 720), (405, 720));
        assert_eq!(fit_box(640, 480, 1920, 1080), (640, 480));
        assert_eq!(fit_box(1921, 1081, 1920, 1080), (1919, 1080));
    }

    #[test]
    fn odd_and_tiny_sizes_stay_valid() {
        assert_eq!(fit_box(1, 1, 1920, 1080), (1, 1));
        assert_eq!(fit_box(1, 10_000, 100, 100), (1, 100));
        assert_eq!(fit_box(0, 0, 100, 100), (1, 1));
        assert_eq!(fit_side(400, 200, 0), (400, 200));
        assert_eq!(fit_side(400, 200, 100), (100, 50));
    }
}
