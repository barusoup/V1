//! Icons: the engine's icon set, sourced verbatim from [Lucide](https://lucide.dev)
//! and rasterized into the UI glyph atlas.
//!
//! Every icon is a stroke-only SVG description on Lucide's 24x24 grid using
//! their defaults: `stroke-width: 2`, round caps and round joins, `fill: none`.
//! [`rasterize`] flattens the curves (cubic, quadratic and elliptical arcs) to
//! polylines and turns distance-to-stroke into pixel coverage, so icons are
//! cleanly anti-aliased at any size and tint like text through the normal atlas
//! path — no per-icon draw state, no weight changes on state.
//!
//! Adding an icon is a copy-paste of Lucide's own element attributes; the
//! parser below handles their full command set (`M L H V C S Q T A Z`, absolute
//! and relative, with implicit repeats), so provenance stays one-to-one with
//! upstream. The design box is the whole grid, which is why every icon centres
//! in a square without per-glyph nudging.

use super::atlas::Region;
use super::font::GlyphCache;
use std::collections::HashMap;

/// The grid every Lucide icon is authored on.
const GRID: f32 = 24.0;
/// Lucide's default stroke weight on that grid.
const STROKE: f32 = 2.0;
/// Distance samples per axis. Distances are averaged before the coverage ramp
/// (a box-filtered distance field), which keeps thin strokes at full opacity
/// and still gives smooth edges.
const SUPERSAMPLE: u32 = 3;
/// Flattening error budget in design units: segments no longer than this.
const FLATNESS: f32 = 0.45;

/// One icon in the engine set. The doc note on each variant is its Lucide slug.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
    /// `undo-2`
    Undo,
    /// `redo-2`
    Redo,
    /// `box`
    Box,
    /// `sun`
    Sun,
    /// `camera`
    Camera,
    /// `move`
    Move,
    /// `play`
    Play,
    /// `pause`
    Pause,
    /// `minus`
    Minus,
    /// `square` (window maximize)
    Maximize,
    /// `copy` (window restore)
    Restore,
    /// `x`
    Close,
    /// `check`
    Check,
    /// `chevron-down`
    ChevronDown,
    /// `chevron-right`
    ChevronRight,
    /// `plus`
    Plus,
    /// `sliders-horizontal`
    Sliders,
    /// `settings` (cogwheel)
    Settings,
    /// `grid-3x3`
    Grid,
    /// `wrench`
    Wrench,
    /// `list`
    List,
}

impl Icon {
    /// Every icon; apps register the whole set once at startup.
    pub const ALL: [Icon; 21] = [
        Icon::Undo,
        Icon::Redo,
        Icon::Box,
        Icon::Sun,
        Icon::Camera,
        Icon::Move,
        Icon::Play,
        Icon::Pause,
        Icon::Minus,
        Icon::Maximize,
        Icon::Restore,
        Icon::Close,
        Icon::Check,
        Icon::ChevronDown,
        Icon::ChevronRight,
        Icon::Plus,
        Icon::Sliders,
        Icon::Settings,
        Icon::Grid,
        Icon::Wrench,
        Icon::List,
    ];

    /// Stable name (atlas key / diagnostics).
    pub fn name(self) -> &'static str {
        match self {
            Icon::Undo => "undo",
            Icon::Redo => "redo",
            Icon::Box => "box",
            Icon::Sun => "sun",
            Icon::Camera => "camera",
            Icon::Move => "move",
            Icon::Play => "play",
            Icon::Pause => "pause",
            Icon::Minus => "minus",
            Icon::Maximize => "maximize",
            Icon::Restore => "restore",
            Icon::Close => "close",
            Icon::Check => "check",
            Icon::ChevronDown => "chevron-down",
            Icon::ChevronRight => "chevron-right",
            Icon::Plus => "plus",
            Icon::Sliders => "sliders",
            Icon::Settings => "settings",
            Icon::Grid => "grid",
            Icon::Wrench => "wrench",
            Icon::List => "list",
        }
    }

    /// Upstream Lucide slug (provenance / update path).
    pub fn slug(self) -> &'static str {
        match self {
            Icon::Undo => "undo-2",
            Icon::Redo => "redo-2",
            Icon::Box => "box",
            Icon::Sun => "sun",
            Icon::Camera => "camera",
            Icon::Move => "move",
            Icon::Play => "play",
            Icon::Pause => "pause",
            Icon::Minus => "minus",
            Icon::Maximize => "square",
            Icon::Restore => "copy",
            Icon::Close => "x",
            Icon::Check => "check",
            Icon::ChevronDown => "chevron-down",
            Icon::ChevronRight => "chevron-right",
            Icon::Plus => "plus",
            Icon::Sliders => "sliders-horizontal",
            Icon::Settings => "settings",
            Icon::Grid => "grid-3x3",
            Icon::Wrench => "wrench",
            Icon::List => "list",
        }
    }

    /// The Lucide elements, copied from upstream (`icons/<slug>.svg`).
    fn elements(self) -> &'static [El] {
        match self {
            Icon::Undo => &[
                El::Path("M9 14 4 9l5-5"),
                El::Path("M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"),
            ],
            Icon::Redo => &[
                El::Path("m15 14 5-5-5-5"),
                El::Path("M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13"),
            ],
            Icon::Box => &[
                El::Path(
                    "M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z",
                ),
                El::Path("m3.3 7 8.7 5 8.7-5"),
                El::Path("M12 22V12"),
            ],
            Icon::Sun => &[
                El::Circle {
                    cx: 12.0,
                    cy: 12.0,
                    r: 4.0,
                },
                El::Path("M12 2v2"),
                El::Path("M12 20v2"),
                El::Path("m4.93 4.93 1.41 1.41"),
                El::Path("m17.66 17.66 1.41 1.41"),
                El::Path("M2 12h2"),
                El::Path("M20 12h2"),
                El::Path("m6.34 17.66-1.41 1.41"),
                El::Path("m19.07 4.93-1.41 1.41"),
            ],
            Icon::Camera => &[
                El::Path(
                    "M13.997 4a2 2 0 0 1 1.76 1.05l.486.9A2 2 0 0 0 18.003 7H20a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1.997a2 2 0 0 0 1.759-1.048l.489-.904A2 2 0 0 1 10.004 4z",
                ),
                El::Circle {
                    cx: 12.0,
                    cy: 13.0,
                    r: 3.0,
                },
            ],
            Icon::Move => &[
                El::Path("M12 2v20"),
                El::Path("m15 19-3 3-3-3"),
                El::Path("m19 9 3 3-3 3"),
                El::Path("M2 12h20"),
                El::Path("m5 9-3 3 3 3"),
                El::Path("m9 5 3-3 3 3"),
            ],
            Icon::Play => &[El::Path(
                "M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z",
            )],
            Icon::Pause => &[
                El::Rect {
                    x: 14.0,
                    y: 3.0,
                    w: 5.0,
                    h: 18.0,
                    rx: 1.0,
                },
                El::Rect {
                    x: 5.0,
                    y: 3.0,
                    w: 5.0,
                    h: 18.0,
                    rx: 1.0,
                },
            ],
            Icon::Minus => &[El::Path("M5 12h14")],
            Icon::Maximize => &[El::Rect {
                x: 3.0,
                y: 3.0,
                w: 18.0,
                h: 18.0,
                rx: 2.0,
            }],
            Icon::Restore => &[
                El::Rect {
                    x: 8.0,
                    y: 8.0,
                    w: 14.0,
                    h: 14.0,
                    rx: 2.0,
                },
                El::Path("M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"),
            ],
            Icon::Close => &[El::Path("M18 6 6 18"), El::Path("m6 6 12 12")],
            Icon::Check => &[El::Path("M20 6 9 17l-5-5")],
            Icon::ChevronDown => &[El::Path("m6 9 6 6 6-6")],
            Icon::ChevronRight => &[El::Path("m9 18 6-6-6-6")],
            Icon::Plus => &[El::Path("M5 12h14"), El::Path("M12 5v14")],
            Icon::Sliders => &[
                El::Path("M10 5H3"),
                El::Path("M12 19H3"),
                El::Path("M14 3v4"),
                El::Path("M16 17v4"),
                El::Path("M21 12h-9"),
                El::Path("M21 19h-5"),
                El::Path("M21 5h-7"),
                El::Path("M8 10v4"),
                El::Path("M8 12H3"),
            ],
            // The rounded-tooth gear Lucide has shipped since 0.534.0, the
            // same revision the rest of this set comes from.
            Icon::Settings => &[
                El::Path(
                    "M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915",
                ),
                El::Circle {
                    cx: 12.0,
                    cy: 12.0,
                    r: 3.0,
                },
            ],
            Icon::Grid => &[
                El::Rect {
                    x: 3.0,
                    y: 3.0,
                    w: 18.0,
                    h: 18.0,
                    rx: 2.0,
                },
                El::Path("M3 9h18"),
                El::Path("M3 15h18"),
                El::Path("M9 3v18"),
                El::Path("M15 3v18"),
            ],
            Icon::Wrench => &[El::Path(
                "M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z",
            )],
            Icon::List => &[
                El::Path("M8 6h13"),
                El::Path("M8 12h13"),
                El::Path("M8 18h13"),
                El::Path("M3 6h.01"),
                El::Path("M3 12h.01"),
                El::Path("M3 18h.01"),
            ],
        }
    }
}

/// One SVG element. Lucide's set uses only these three.
#[derive(Clone, Copy, Debug)]
enum El {
    /// Path data (`d`), stroke-only.
    Path(&'static str),
    /// `<circle cx cy r/>`.
    Circle { cx: f32, cy: f32, r: f32 },
    /// `<rect x y width height rx/>` (square corners when `rx` is 0).
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        rx: f32,
    },
}

const TAU: f32 = std::f32::consts::TAU;

/// Flatten an icon's elements into polylines on the 24x24 grid.
fn polylines(icon: Icon) -> Vec<Vec<(f32, f32)>> {
    let mut out = Vec::new();
    for element in icon.elements() {
        match *element {
            El::Path(d) => flatten_path(d, &mut out),
            El::Circle { cx, cy, r } => {
                let mut ring = Vec::new();
                circle_ring(cx, cy, r, &mut ring);
                out.push(ring);
            }
            El::Rect { x, y, w, h, rx } => {
                let mut ring = Vec::new();
                round_rect_ring(x, y, w, h, rx, &mut ring);
                out.push(ring);
            }
        }
    }
    out
}

/// Apparent a circle outline with segment length near [`FLATNESS`].
fn circle_ring(cx: f32, cy: f32, r: f32, out: &mut Vec<(f32, f32)>) {
    let steps = ((TAU * r / FLATNESS).ceil() as usize).clamp(12, 96);
    for k in 0..steps {
        let t = TAU * k as f32 / steps as f32;
        out.push((cx + r * t.cos(), cy + r * t.sin()));
    }
    out.push((cx + r, cy));
}

/// Append a rounded-rectangle outline (Lucide `<rect rx>`).
fn round_rect_ring(x: f32, y: f32, w: f32, h: f32, rx: f32, out: &mut Vec<(f32, f32)>) {
    let r = rx.clamp(0.0, w.min(h) * 0.5);
    let start = out.len();
    let arc = |cx: f32, cy: f32, a0: f32, a1: f32, out: &mut Vec<(f32, f32)>| {
        let steps = ((r.abs() * (a1 - a0).abs() / FLATNESS).ceil() as usize).max(2);
        for k in 0..=steps {
            let t = a0 + (a1 - a0) * k as f32 / steps as f32;
            out.push((cx + r * t.cos(), cy + r * t.sin()));
        }
    };
    arc(x + w - r, y + r, -TAU * 0.25, 0.0, out);
    arc(x + w - r, y + h - r, 0.0, TAU * 0.25, out);
    arc(x + r, y + h - r, TAU * 0.25, TAU * 0.5, out);
    arc(x + r, y + r, TAU * 0.5, TAU * 0.75, out);
    // Close the ring: the outline runs top-right -> right -> bottom -> left ->
    // top-left, so the top edge is the segment back to the first point.
    out.push(out[start]);
}

/// A number/command scanner over SVG path data.
struct Scan<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Scan<'a> {
    fn new(data: &'a str) -> Self {
        Self {
            bytes: data.as_bytes(),
            at: 0,
        }
    }

    fn skip_separators(&mut self) {
        while matches!(
            self.bytes.get(self.at),
            Some(b' ') | Some(b',') | Some(b'\n') | Some(b'\r') | Some(b'\t')
        ) {
            self.at += 1;
        }
    }

    fn at_end(&mut self) -> bool {
        self.skip_separators();
        self.at >= self.bytes.len()
    }

    fn peek_command(&mut self) -> Option<u8> {
        self.skip_separators();
        match self.bytes.get(self.at) {
            Some(&c) if c.is_ascii_alphabetic() => Some(c),
            _ => None,
        }
    }

    fn take_command(&mut self) -> u8 {
        let c = self.bytes[self.at];
        self.at += 1;
        c
    }

    /// Next number, advancing past it. Garbage parses as 0 so a bad datum
    /// cannot desynchronise the rest of the path.
    fn number(&mut self) -> f32 {
        self.skip_separators();
        let start = self.at;
        if matches!(self.bytes.get(self.at), Some(b'-') | Some(b'+')) {
            self.at += 1;
        }
        while matches!(self.bytes.get(self.at), Some(c) if c.is_ascii_digit()) {
            self.at += 1;
        }
        if matches!(self.bytes.get(self.at), Some(b'.')) {
            self.at += 1;
            while matches!(self.bytes.get(self.at), Some(c) if c.is_ascii_digit()) {
                self.at += 1;
            }
        }
        if matches!(self.bytes.get(self.at), Some(b'e') | Some(b'E')) {
            self.at += 1;
            if matches!(self.bytes.get(self.at), Some(b'-') | Some(b'+')) {
                self.at += 1;
            }
            while matches!(self.bytes.get(self.at), Some(c) if c.is_ascii_digit()) {
                self.at += 1;
            }
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(0.0)
    }

    /// Arc flag: a single `0`/`1` digit.
    fn flag(&mut self) -> bool {
        self.number() != 0.0
    }
}

/// Flatten one `d` string, appending each subpath to `out`.
fn flatten_path(data: &str, out: &mut Vec<Vec<(f32, f32)>>) {
    let mut scan = Scan::new(data);
    let mut current = (0.0f32, 0.0f32);
    let mut subpath_start = current;
    let mut control: Option<(f32, f32)> = None;
    let mut subpath: Vec<(f32, f32)> = Vec::new();
    let mut command = 0u8;

    while !scan.at_end() {
        match scan.peek_command() {
            Some(_) => command = scan.take_command(),
            // Extra coordinate pairs repeat the previous command; after a
            // moveto they are implicit linetos.
            None => match command {
                b'M' => command = b'L',
                b'm' => command = b'l',
                0 => break,
                _ => {}
            },
        }
        let relative = command.is_ascii_lowercase();
        let place = |p: (f32, f32), current: (f32, f32)| {
            if relative {
                (current.0 + p.0, current.1 + p.1)
            } else {
                p
            }
        };
        match command.to_ascii_uppercase() {
            b'M' => {
                flush(&mut subpath, out);
                current = place((scan.number(), scan.number()), current);
                subpath_start = current;
                subpath.push(current);
                control = None;
            }
            b'L' => {
                current = place((scan.number(), scan.number()), current);
                subpath.push(current);
                control = None;
            }
            b'H' => {
                let x = scan.number();
                current = (if relative { current.0 + x } else { x }, current.1);
                subpath.push(current);
                control = None;
            }
            b'V' => {
                let y = scan.number();
                current = (current.0, if relative { current.1 + y } else { y });
                subpath.push(current);
                control = None;
            }
            b'C' => {
                let c1 = place((scan.number(), scan.number()), current);
                let c2 = place((scan.number(), scan.number()), current);
                let end = place((scan.number(), scan.number()), current);
                cubic(current, c1, c2, end, &mut subpath);
                current = end;
                control = Some(c2);
            }
            b'S' => {
                let c1 = reflect(control, current);
                let c2 = place((scan.number(), scan.number()), current);
                let end = place((scan.number(), scan.number()), current);
                cubic(current, c1, c2, end, &mut subpath);
                current = end;
                control = Some(c2);
            }
            b'Q' => {
                let c1 = place((scan.number(), scan.number()), current);
                let end = place((scan.number(), scan.number()), current);
                quadratic(current, c1, end, &mut subpath);
                current = end;
                control = Some(c1);
            }
            b'T' => {
                let c1 = reflect(control, current);
                let end = place((scan.number(), scan.number()), current);
                quadratic(current, c1, end, &mut subpath);
                current = end;
                control = Some(c1);
            }
            b'A' => {
                let rx = scan.number();
                let ry = scan.number();
                let rotation = scan.number();
                let large = scan.flag();
                let sweep = scan.flag();
                let end = place((scan.number(), scan.number()), current);
                elliptical_arc(current, rx, ry, rotation, large, sweep, end, &mut subpath);
                current = end;
                control = None;
            }
            b'Z' => {
                if !subpath.is_empty() {
                    subpath.push(subpath_start);
                    flush(&mut subpath, out);
                }
                current = subpath_start;
                control = None;
            }
            _ => break,
        }
    }
    flush(&mut subpath, out);
}

fn flush(subpath: &mut Vec<(f32, f32)>, out: &mut Vec<Vec<(f32, f32)>>) {
    if subpath.len() > 1 {
        out.push(std::mem::take(subpath));
    } else {
        subpath.clear();
    }
}

/// Mirror `control` through `current` (the `S`/`T` smooth-curve rule).
fn reflect(control: Option<(f32, f32)>, current: (f32, f32)) -> (f32, f32) {
    match control {
        Some(c) => (2.0 * current.0 - c.0, 2.0 * current.1 - c.1),
        None => current,
    }
}

/// Segments to approximate a curve of `rough` control-polygon length.
fn segments(rough: f32) -> usize {
    ((rough / FLATNESS).ceil() as usize).clamp(6, 72)
}

/// Append a cubic Bézier (excluding `p0`, which is already in the subpath).
fn cubic(
    p0: (f32, f32),
    c1: (f32, f32),
    c2: (f32, f32),
    p1: (f32, f32),
    out: &mut Vec<(f32, f32)>,
) {
    let rough = dist(p0, c1) + dist(c1, c2) + dist(c2, p1);
    let n = segments(rough);
    for k in 1..=n {
        let t = k as f32 / n as f32;
        let u = 1.0 - t;
        let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
        out.push((
            a * p0.0 + b * c1.0 + c * c2.0 + d * p1.0,
            a * p0.1 + b * c1.1 + c * c2.1 + d * p1.1,
        ));
    }
}

/// Append a quadratic Bézier as its exactly equivalent cubic.
fn quadratic(p0: (f32, f32), control: (f32, f32), p1: (f32, f32), out: &mut Vec<(f32, f32)>) {
    let c1 = (
        p0.0 + 2.0 / 3.0 * (control.0 - p0.0),
        p0.1 + 2.0 / 3.0 * (control.1 - p0.1),
    );
    let c2 = (
        p1.0 + 2.0 / 3.0 * (control.0 - p1.0),
        p1.1 + 2.0 / 3.0 * (control.1 - p1.1),
    );
    cubic(p0, c1, c2, p1, out);
}

/// Append an SVG elliptical arc (`A`), per the SVG implementation notes.
#[allow(clippy::too_many_arguments)]
fn elliptical_arc(
    from: (f32, f32),
    rx: f32,
    ry: f32,
    rotation: f32,
    large: bool,
    sweep: bool,
    to: (f32, f32),
    out: &mut Vec<(f32, f32)>,
) {
    let (mut rx, mut ry) = (rx.abs(), ry.abs());
    if rx < 1e-4 || ry < 1e-4 || from == to {
        out.push(to);
        return;
    }
    let phi = rotation.to_radians();
    let (sin_phi, cos_phi) = phi.sin_cos();
    let half = ((from.0 - to.0) * 0.5, (from.1 - to.1) * 0.5);
    let x1 = cos_phi * half.0 + sin_phi * half.1;
    let y1 = -sin_phi * half.0 + cos_phi * half.1;
    // Scale up radii that are too small to span the chord.
    let lambda = x1 * x1 / (rx * rx) + y1 * y1 / (ry * ry);
    if lambda > 1.0 {
        let scale = lambda.sqrt();
        rx *= scale;
        ry *= scale;
    }
    let sign = if large != sweep { 1.0 } else { -1.0 };
    let numerator = (rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1).max(0.0);
    let denominator = rx * rx * y1 * y1 + ry * ry * x1 * x1;
    let factor = sign * (numerator / denominator.max(1e-9)).sqrt();
    let center_x = factor * rx * y1 / ry;
    let center_y = -factor * ry * x1 / rx;
    let cx = cos_phi * center_x - sin_phi * center_y + (from.0 + to.0) * 0.5;
    let cy = sin_phi * center_x + cos_phi * center_y + (from.1 + to.1) * 0.5;
    let angle = |ux: f32, uy: f32, vx: f32, vy: f32| {
        let mut a = (ux * vx + uy * vy).clamp(-1.0, 1.0).acos();
        if ux * vy - uy * vx < 0.0 {
            a = -a;
        }
        a
    };
    let (sx, sy) = ((x1 - center_x) / rx, (y1 - center_y) / ry);
    let (ex, ey) = ((-x1 - center_x) / rx, (-y1 - center_y) / ry);
    let start = angle(1.0, 0.0, sx, sy);
    let mut sweep_angle = angle(sx, sy, ex, ey);
    if !sweep && sweep_angle > 0.0 {
        sweep_angle -= TAU;
    }
    if sweep && sweep_angle < 0.0 {
        sweep_angle += TAU;
    }
    let radius = (rx + ry) * 0.5;
    let steps = ((sweep_angle.abs() * radius / FLATNESS).ceil() as usize).clamp(2, 96);
    for k in 1..=steps {
        let t = start + sweep_angle * (k as f32 / steps as f32);
        let (sin_t, cos_t) = t.sin_cos();
        out.push((
            cos_phi * rx * cos_t - sin_phi * ry * sin_t + cx,
            sin_phi * rx * cos_t + cos_phi * ry * sin_t + cy,
        ));
    }
    // Land exactly on the endpoint: the trig above is approximate.
    out.push(to);
}

/// Squared distance between two points.
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Distance from `p` to the segment `a`..`b`.
fn segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (pax, pay) = (p.0 - a.0, p.1 - a.1);
    let (bax, bay) = (b.0 - a.0, b.1 - a.1);
    let len2 = bax * bax + bay * bay;
    let t = if len2 > 1e-9 {
        ((pax * bax + pay * bay) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (dx, dy) = (pax - bax * t, pay - bay * t);
    (dx * dx + dy * dy).sqrt()
}

/// Distance from `p` to the nearest stroke centreline.
fn stroke_distance(polys: &[Vec<(f32, f32)>], p: (f32, f32)) -> f32 {
    let mut best = f32::INFINITY;
    for poly in polys {
        for pair in poly.windows(2) {
            let d = segment_distance(p, pair[0], pair[1]);
            if d < best {
                best = d;
            }
        }
    }
    best
}

/// Rasterize `icon` to `size`x`size` 8-bit coverage, top-left origin.
///
/// Coverage is analytic: the mean distance to the stroke over the pixel
/// footprint is turned into coverage across one pixel of ramp, so edges are
/// smooth and the middle of a stroke stays fully opaque even when the stroke
/// is barely a pixel wide.
pub fn rasterize(icon: Icon, size: u32) -> Vec<u8> {
    let size = size.max(1);
    let polys = polylines(icon);
    let scale = size as f32 / GRID;
    let half_stroke = STROKE * 0.5;
    let pixel = 1.0 / scale;
    let inv_samples = 1.0 / (SUPERSAMPLE * SUPERSAMPLE) as f32;
    let mut out = vec![0u8; (size * size) as usize];
    if polys.is_empty() {
        return out;
    }
    let band = 0.98 * pixel;
    for y in 0..size {
        for x in 0..size {
            let center = ((x as f32 + 0.5) / scale, (y as f32 + 0.5) / scale);
            let d_center = stroke_distance(&polys, center);
            if d_center >= half_stroke + band {
                continue;
            }
            if d_center <= half_stroke - band {
                out[(y * size + x) as usize] = 255;
                continue;
            }
            let mut distance = d_center;
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    if sx == 1 && sy == 1 {
                        continue;
                    }
                    let sample = (
                        (x as f32 + (sx as f32 + 0.5) / SUPERSAMPLE as f32) / scale,
                        (y as f32 + (sy as f32 + 0.5) / SUPERSAMPLE as f32) / scale,
                    );
                    distance += stroke_distance(&polys, sample);
                }
            }
            let coverage = (0.5 + (half_stroke - distance * inv_samples) / pixel).clamp(0.0, 1.0);
            out[(y * size + x) as usize] = (coverage * 255.0).round() as u8;
        }
    }
    out
}

/// The bounding box an icon's ink actually fills, in 24-grid units: the
/// flattened stroke bounds expanded by half a stroke width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InkBox {
    pub min: (f32, f32),
    pub max: (f32, f32),
}

impl InkBox {
    /// Ink centre, grid units.
    pub fn center(&self) -> (f32, f32) {
        (
            (self.min.0 + self.max.0) * 0.5,
            (self.min.1 + self.max.1) * 0.5,
        )
    }

    /// Longest side of the ink box, grid units.
    pub fn major(&self) -> f32 {
        (self.max.0 - self.min.0)
            .max(self.max.1 - self.min.1)
            .max(1e-3)
    }
}

fn polys_ink_box(polys: &[Vec<(f32, f32)>], stroke_width: f32) -> InkBox {
    let half = stroke_width * 0.5;
    let (mut min, mut max) = (
        (f32::INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::NEG_INFINITY),
    );
    for poly in polys {
        for &(x, y) in poly {
            min = (min.0.min(x), min.1.min(y));
            max = (max.0.max(x), max.1.max(y));
        }
    }
    if min.0 > max.0 {
        return InkBox {
            min: (0.0, 0.0),
            max: (GRID, GRID),
        };
    }
    InkBox {
        min: (min.0 - half, min.1 - half),
        max: (max.0 + half, max.1 + half),
    }
}

/// `icon`'s ink box on the 24x24 grid (strokes expanded by `STROKE / 2`).
pub fn ink_box(icon: Icon) -> InkBox {
    polys_ink_box(&polylines(icon), STROKE)
}

/// The ink box of a custom icon's flattened paths, expanded by its own
/// `stroke_width / 2`.
pub fn ink_box_svg_paths(paths: &[&str], stroke_width: f32) -> InkBox {
    let mut polys = Vec::new();
    for &d in paths {
        flatten_path(d, &mut polys);
    }
    polys_ink_box(&polys, stroke_width)
}

impl Icon {
    /// Optical weight correction for [`crate::ui::widgets::IconFit::Ink`]:
    /// shapes that read large for their ink extent get a smaller target.
    pub fn optical_k(self) -> f32 {
        match self {
            Icon::Maximize
            | Icon::Restore
            | Icon::Box
            | Icon::Grid
            | Icon::Settings
            | Icon::Sun
            | Icon::Camera => 0.9,
            _ => 1.0,
        }
    }
}

/// Rasterize arbitrary SVG `<path d="...">` strings authored on the 24x24
/// grid into `size`x`size` 8-bit coverage with `stroke_width` (2.0 is
/// Lucide's standard weight).
pub fn rasterize_svg_paths(paths: &[&str], size: u32, stroke_width: f32) -> Vec<u8> {
    rasterize_svg_paths_with(paths, size, stroke_width, false)
}

/// Whether `p` lies inside the closed polyline `poly` (even-odd).
fn point_in_poly(poly: &[(f32, f32)], p: (f32, f32)) -> bool {
    let mut inside = false;
    for pair in poly.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
            inside = !inside;
        }
    }
    inside
}

/// Like [`rasterize_svg_paths`], and when `fill` is set the closed subpaths
/// (those that end where they began) are filled as well as stroked, as an SVG
/// with `fill="currentColor"` would be. Open subpaths stay stroke-only.
pub fn rasterize_svg_paths_with(
    paths: &[&str],
    size: u32,
    stroke_width: f32,
    fill: bool,
) -> Vec<u8> {
    let size = size.max(1);
    let mut polys = Vec::new();
    for &d in paths {
        flatten_path(d, &mut polys);
    }
    let closed: Vec<&Vec<(f32, f32)>> = if fill {
        polys
            .iter()
            .filter(|poly| {
                poly.len() > 3
                    && poly
                        .first()
                        .zip(poly.last())
                        .is_some_and(|(a, b)| (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-3)
            })
            .collect()
    } else {
        Vec::new()
    };
    let scale = size as f32 / GRID;
    let half_stroke = stroke_width.max(0.25) * 0.5;
    let pixel = 1.0 / scale;
    let inv_samples = 1.0 / (SUPERSAMPLE * SUPERSAMPLE) as f32;
    let mut out = vec![0u8; (size * size) as usize];
    if polys.is_empty() {
        return out;
    }
    let band = 0.98 * pixel;
    for y in 0..size {
        for x in 0..size {
            let center = ((x as f32 + 0.5) / scale, (y as f32 + 0.5) / scale);
            let d_center = stroke_distance(&polys, center);
            // The stroke covers every edge of the fill, so a pixel is either
            // wholly inside the shape or the stroke below antialiases it.
            if closed.iter().any(|poly| point_in_poly(poly, center)) {
                out[(y * size + x) as usize] = 255;
                continue;
            }
            if d_center >= half_stroke + band {
                continue;
            }
            if d_center <= half_stroke - band {
                out[(y * size + x) as usize] = 255;
                continue;
            }
            let mut distance = d_center;
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    if sx == 1 && sy == 1 {
                        continue;
                    }
                    let sample = (
                        (x as f32 + (sx as f32 + 0.5) / SUPERSAMPLE as f32) / scale,
                        (y as f32 + (sy as f32 + 0.5) / SUPERSAMPLE as f32) / scale,
                    );
                    distance += stroke_distance(&polys, sample);
                }
            }
            let coverage = (0.5 + (half_stroke - distance * inv_samples) / pixel).clamp(0.0, 1.0);
            out[(y * size + x) as usize] = (coverage * 255.0).round() as u8;
        }
    }
    out
}

/// Baked icon regions, keyed by (icon, pixel size).
#[derive(Debug, Default)]
pub struct IconSet {
    regions: HashMap<(Icon, u32), Region>,
    custom: HashMap<(u64, u32), Region>,
    ink: HashMap<Icon, InkBox>,
    custom_ink: HashMap<u64, InkBox>,
}

impl IconSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bake `icon` at `size` pixels (idempotent).
    pub fn register(&mut self, icon: Icon, size: u32, cache: &mut GlyphCache) {
        let size = size.max(1);
        let key = (icon, size);
        if self.regions.contains_key(&key) {
            return;
        }
        let coverage = rasterize(icon, size);
        match cache.insert_coverage(&coverage, size, size) {
            Ok(region) => {
                self.regions.insert(key, region);
            }
            Err(crate::ui::atlas::AtlasError::BudgetExceeded { .. }) => {
                // Recorded on the cache; the next frame boundary resets.
            }
            Err(err) => {
                eprintln!("[ui] icon {} too large for the atlas: {err}", icon.name())
            }
        }
    }

    /// Every baked region points into the (old) atlas — after a reset the
    /// set must rebake. The ink boxes are geometry, not atlas data, so
    /// they survive.
    pub fn clear(&mut self) {
        self.regions.clear();
        self.custom.clear();
    }

    /// `icon`'s cached ink box (strokes expanded by `STROKE / 2`).
    pub fn ink_bounds(&mut self, icon: Icon) -> InkBox {
        *self.ink.entry(icon).or_insert_with(|| ink_box(icon))
    }

    /// A custom icon's cached ink box, expanded by its `stroke_width / 2`.
    pub fn custom_ink_bounds(&mut self, id: u64, paths: &[&str], stroke_width: f32) -> InkBox {
        *self
            .custom_ink
            .entry(id)
            .or_insert_with(|| ink_box_svg_paths(paths, stroke_width))
    }

    /// The atlas region for `icon` at `size`, baking on demand.
    pub fn region(&mut self, icon: Icon, size: u32, cache: &mut GlyphCache) -> Option<Region> {
        let size = size.max(1);
        if let Some(region) = self.regions.get(&(icon, size)) {
            return Some(*region);
        }
        self.register(icon, size, cache);
        self.regions.get(&(icon, size)).copied()
    }

    /// The atlas region for a custom SVG path icon identified by `id` at `size` px.
    pub fn custom_region(
        &mut self,
        id: u64,
        paths: &[&str],
        size: u32,
        stroke_width: f32,
        cache: &mut GlyphCache,
    ) -> Option<Region> {
        let size = size.max(1);
        let key = (id, size);
        if let Some(region) = self.custom.get(&key) {
            return Some(*region);
        }
        let coverage = rasterize_svg_paths(paths, size, stroke_width);
        if let Ok(region) = cache.insert_coverage(&coverage, size, size) {
            self.custom.insert(key, region);
            return Some(region);
        }
        None
    }

    /// Like [`Self::custom_region`], optionally filling closed subpaths.
    pub fn custom_region_filled(
        &mut self,
        id: u64,
        paths: &[&str],
        size: u32,
        stroke_width: f32,
        fill: bool,
        cache: &mut GlyphCache,
    ) -> Option<Region> {
        let size = size.max(1);
        let key = (id, size);
        if let Some(region) = self.custom.get(&key) {
            return Some(*region);
        }
        let coverage = rasterize_svg_paths_with(paths, size, stroke_width, fill);
        if let Ok(region) = cache.insert_coverage(&coverage, size, size) {
            self.custom.insert(key, region);
            return Some(region);
        }
        None
    }

    /// Baked (icon, size) pairs — atlas diagnostics.
    pub fn baked(&self) -> usize {
        self.regions.len() + self.custom.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::atlas::AtlasConfig;

    fn cache() -> GlyphCache {
        GlyphCache::new(super::super::atlas::Atlas::new(AtlasConfig {
            page_size: 512,
            padding: 1,
        }))
    }

    #[test]
    fn a_filled_closed_path_is_solid_inside_and_open_paths_stay_hollow() {
        let square = ["M4 4H20V20H4Z"];
        let hollow = rasterize_svg_paths_with(&square, 24, 2.0, false);
        let solid = rasterize_svg_paths_with(&square, 24, 2.0, true);
        assert_eq!(hollow[12 * 24 + 12], 0);
        assert_eq!(solid[12 * 24 + 12], 255);
        let open = ["M4 4H20V20H4"];
        let still_open = rasterize_svg_paths_with(&open, 24, 2.0, true);
        assert_eq!(still_open[12 * 24 + 12], 0);
    }

    #[test]
    fn every_icon_flattens_inside_the_grid() {
        for icon in Icon::ALL {
            let polys = polylines(icon);
            assert!(!polys.is_empty(), "{}: no geometry", icon.name());
            let points: usize = polys.iter().map(Vec::len).sum();
            assert!(points >= 2, "{}: {points} points", icon.name());
            for poly in &polys {
                assert!(poly.len() >= 2, "{}: degenerate subpath", icon.name());
                for &(x, y) in poly {
                    assert!(
                        (-1.0..=25.0).contains(&x) && (-1.0..=25.0).contains(&y),
                        "{}: point ({x}, {y}) off the 24x24 grid",
                        icon.name()
                    );
                }
            }
        }
    }

    #[test]
    fn every_icon_has_ink_and_breathing_room() {
        for icon in Icon::ALL {
            let px = 20u32;
            let cov = rasterize(icon, px);
            assert_eq!(
                cov.len(),
                (px * px) as usize,
                "{}: wrong buffer",
                icon.name()
            );
            let inked = cov.iter().filter(|&&c| c > 16).count();
            let solid = cov.iter().filter(|&&c| c >= 250).count();
            assert!(inked > 12, "{}: too little ink ({inked})", icon.name());
            assert!(
                solid < (px * px) as usize / 2,
                "{}: too solid ({solid}) — likely a fill bug",
                icon.name()
            );
        }
    }

    #[test]
    fn ink_sits_where_lucide_authored_it() {
        // Lucide authors each icon's optical placement inside the 24 box, so
        // centring the box is the correct measured alignment. This measures
        // the rendered ink and asserts no icon drifted: a parser bug (an arc
        // sign, a relative/absolute mix-up) moves the bounding box and fails.
        let px = 24u32;
        for icon in Icon::ALL {
            let cov = rasterize(icon, px);
            let (mut x0, mut y0, mut x1, mut y1) = (px, px, 0u32, 0u32);
            for y in 0..px {
                for x in 0..px {
                    if cov[(y * px + x) as usize] > 8 {
                        x0 = x0.min(x);
                        y0 = y0.min(y);
                        x1 = x1.max(x + 1);
                        y1 = y1.max(y + 1);
                    }
                }
            }
            assert!(x1 > x0 && y1 > y0, "{}: no ink", icon.name());
            let center = ((x0 + x1) as f32 * 0.5, (y0 + y1) as f32 * 0.5);
            let offset = (center.0 - px as f32 * 0.5)
                .abs()
                .max((center.1 - px as f32 * 0.5).abs());
            assert!(
                offset <= 2.5,
                "{}: ink centre {center:?} is {offset:.2}px off the box centre",
                icon.name()
            );
        }
    }

    #[test]
    fn rasterization_is_anti_aliased() {
        // A diagonal stroke must produce partial coverage somewhere.
        let cov = rasterize(Icon::Close, 24);
        assert!(
            cov.iter().any(|&c| c > 8 && c < 248),
            "diagonal edges should be anti-aliased"
        );
    }

    #[test]
    fn outlines_are_closed_rings() {
        // A stroked ring must come back to where it started, or its closing
        // edge (for a rect, the whole top edge) is missing.
        for icon in [Icon::Maximize, Icon::Grid, Icon::Pause, Icon::Sun] {
            // The ring is the first subpath (Grid's inner lines are open).
            let ring = polylines(icon).into_iter().next().expect("geometry");
            assert_eq!(
                ring.first(),
                ring.last(),
                "{}: outline is not closed",
                icon.name()
            );
        }
        // And the ink of a full square reaches its top row as well as its
        // bottom row.
        let px = 16u32;
        let cov = rasterize(Icon::Maximize, px);
        let row_ink = |row: u32| {
            (0..px)
                .filter(|&x| cov[(row * px + x) as usize] > 8)
                .count()
        };
        assert!(row_ink(2) > 6, "top edge inked: {}", row_ink(2));
        assert!(
            row_ink(px - 3) > 6,
            "bottom edge inked: {}",
            row_ink(px - 3)
        );
    }

    #[test]
    fn curves_and_arcs_flatten_smoothly() {
        // A half-circle arc from (0,0) to (20,0), sweep=1 bulges upward
        // (negative y) in SVG's y-down space.
        let mut out = Vec::new();
        flatten_path("M0 0A10 10 0 0 1 20 0", &mut out);
        assert_eq!(out.len(), 1, "one subpath");
        let pts = &out[0];
        assert!(
            pts.len() > 8,
            "arc should be finely segmented, got {}",
            pts.len()
        );
        let min_y = pts.iter().fold(f32::INFINITY, |m, p| m.min(p.1));
        assert!((min_y + 10.0).abs() < 0.4, "arc apex at y=-10, got {min_y}");
        assert_eq!(*pts.last().unwrap(), (20.0, 0.0), "lands on the endpoint");

        // Implicit linetos after a moveto, and relative curves.
        let mut out = Vec::new();
        flatten_path("M9 14 4 9l5-5", &mut out);
        assert_eq!(out[0][0], (9.0, 14.0));
        assert_eq!(out[0][1], (4.0, 9.0));
        assert_eq!(*out[0].last().unwrap(), (9.0, 4.0));

        // Cubic + smooth cubic (`C` then `S`).
        let mut out = Vec::new();
        flatten_path("M0 0C5 0 10 5 10 10S15 20 20 20", &mut out);
        let pts = &out[0];
        assert!(pts.len() > 12, "cubics segmented, got {}", pts.len());
        assert_eq!(*pts.last().unwrap(), (20.0, 20.0));
    }

    #[test]
    fn sizes_scale_and_regions_are_stable() {
        let mut cache = cache();
        let mut icons = IconSet::new();
        let a = icons.region(Icon::Play, 20, &mut cache).expect("baked");
        assert_eq!((a.width, a.height), (20, 20));
        let again = icons.region(Icon::Play, 20, &mut cache).expect("cached");
        assert_eq!(a, again, "region is stable across calls");
        let big = icons.region(Icon::Play, 32, &mut cache).expect("baked");
        assert_eq!((big.width, big.height), (32, 32));
        assert_eq!(icons.baked(), 2);
    }

    #[test]
    fn baked_icon_has_white_coverage_in_atlas() {
        let mut cache = cache();
        let mut icons = IconSet::new();
        let region = icons.region(Icon::Check, 20, &mut cache).unwrap();
        let page = cache.atlas().page_pixels(region.page);
        let size = cache.atlas().page_size() as usize;
        let at = (region.y as usize * size + region.x as usize) * 4;
        // Somewhere in the glyph there is solid white ink.
        let has_solid = (0..region.height as usize).any(|row| {
            (0..region.width as usize).any(|col| page[at + (row * size + col) * 4 + 3] >= 250)
        });
        assert!(has_solid, "icon ink is opaque white coverage");
    }

    #[test]
    fn slugs_are_unique_and_named() {
        let mut slugs: Vec<&str> = Icon::ALL.iter().map(|i| i.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(
            slugs.len(),
            count,
            "lucide slug must identify exactly one icon"
        );
        for icon in Icon::ALL {
            assert!(!icon.slug().is_empty() && !icon.name().is_empty());
        }
    }
}
