// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Icons: the real Lucide set (ISC licence, see `assets/icons/LICENSE-lucide.txt`).
//!
//! Each icon is the unmodified Lucide SVG, parsed once, flattened to polylines on its 24 unit grid
//! and rasterized to a white-on-alpha texture with analytic anti-aliasing (distance to the stroke,
//! signed distance to the outline for filled shapes). Textures are made lazily at a handful of
//! pixel sizes so an icon is always drawn close to 1:1 and stays crisp at any UI scale.
//!
//! Only the shapes Lucide actually uses are handled: `path` (with every command, including arcs),
//! `circle`, `rect` (with `rx`) and `line`.

use dream::ui::draw::DrawList;
use dream::ui::{Color, Rect};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Icon {
    House,
    Book,
    Disc,
    ListMusic,
    Menu,
    Search,
    ArrowLeft,
    X,
    PanelLeft,
    PanelRight,
    Play,
    Pause,
    SkipBack,
    SkipForward,
    Volume1,
    Volume2,
    VolumeX,
    Music,
    Mic,
    Minus,
    Plus,
    Check,
    User,
    GripVertical,
    Trash2,
}

impl Icon {
    /// Every icon, for the tests that check each one parses and rasterizes.
    #[allow(dead_code)]
    pub const ALL: [Icon; 25] = [
        Icon::House,
        Icon::Book,
        Icon::Disc,
        Icon::ListMusic,
        Icon::Menu,
        Icon::Search,
        Icon::ArrowLeft,
        Icon::X,
        Icon::PanelLeft,
        Icon::PanelRight,
        Icon::Play,
        Icon::Pause,
        Icon::SkipBack,
        Icon::SkipForward,
        Icon::Volume1,
        Icon::Volume2,
        Icon::VolumeX,
        Icon::Music,
        Icon::Mic,
        Icon::Minus,
        Icon::Plus,
        Icon::Check,
        Icon::User,
        Icon::GripVertical,
        Icon::Trash2,
    ];

    fn svg(self) -> &'static str {
        match self {
            Icon::House => include_str!("../assets/icons/house.svg"),
            Icon::Book => include_str!("../assets/icons/book.svg"),
            Icon::Disc => include_str!("../assets/icons/disc-3.svg"),
            Icon::ListMusic => include_str!("../assets/icons/list-music.svg"),
            Icon::Menu => include_str!("../assets/icons/menu.svg"),
            Icon::Search => include_str!("../assets/icons/search.svg"),
            Icon::ArrowLeft => include_str!("../assets/icons/arrow-left.svg"),
            Icon::X => include_str!("../assets/icons/x.svg"),
            Icon::PanelLeft => include_str!("../assets/icons/panel-left.svg"),
            Icon::PanelRight => include_str!("../assets/icons/panel-right.svg"),
            Icon::Play => include_str!("../assets/icons/play.svg"),
            Icon::Pause => include_str!("../assets/icons/pause.svg"),
            Icon::SkipBack => include_str!("../assets/icons/skip-back.svg"),
            Icon::SkipForward => include_str!("../assets/icons/skip-forward.svg"),
            Icon::Volume1 => include_str!("../assets/icons/volume-1.svg"),
            Icon::Volume2 => include_str!("../assets/icons/volume-2.svg"),
            Icon::VolumeX => include_str!("../assets/icons/volume-x.svg"),
            Icon::Music => include_str!("../assets/icons/music.svg"),
            Icon::Mic => include_str!("../assets/icons/mic.svg"),
            Icon::Minus => include_str!("../assets/icons/minus.svg"),
            Icon::Plus => include_str!("../assets/icons/plus.svg"),
            Icon::Check => include_str!("../assets/icons/check.svg"),
            Icon::User => include_str!("../assets/icons/user.svg"),
            Icon::GripVertical => include_str!("../assets/icons/grip-vertical.svg"),
            Icon::Trash2 => include_str!("../assets/icons/trash-2.svg"),
        }
    }

    /// Transport glyphs are solid, as in Apple's players; the rest stay outlines.
    fn filled(self) -> bool {
        matches!(
            self,
            Icon::Play | Icon::Pause | Icon::SkipBack | Icon::SkipForward
        )
    }

    /// Stroke width on the 24 grid. Lucide ships 2; a little lighter reads more refined.
    fn stroke(self) -> f32 {
        match self {
            // Drawn small, inside the save button: a touch heavier so the strokes hold up.
            Icon::Plus | Icon::Check => 2.4,
            _ => 1.75,
        }
    }
}

/// The texture size, in device pixels, for an icon drawn at `device_px`. It is exactly that size,
/// so the icon is drawn 1:1 with no resampling (which is what made icons look soft).
pub fn bucket(device_px: f32) -> u32 {
    device_px.round().clamp(8.0, 512.0) as u32
}

/// Sizes near `wanted`, nearest first, for falling back to a similar texture on the frame an exact
/// one is still being rasterized.
pub fn nearest_buckets(wanted: u32) -> impl Iterator<Item = u32> {
    let wanted = wanted as i64;
    (0..=64i64)
        .flat_map(move |d| [wanted - d, wanted + d])
        .filter(|s| (8..=512).contains(s))
        .map(|s| s as u32)
}

type P = (f32, f32);

struct Poly {
    points: Vec<P>,
    closed: bool,
}

/// One SVG element's outlines.
struct Shape {
    polys: Vec<Poly>,
}

// ---------------------------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------------------------

/// The value of attribute `name` on a tag, or `None`. Matches whole attribute names only.
fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!(" {name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

fn number_attr(tag: &str, name: &str) -> f32 {
    attr(tag, name).and_then(|v| v.trim().parse().ok()).unwrap_or(0.0)
}

fn parse_svg(svg: &str) -> Vec<Shape> {
    let mut shapes = Vec::new();
    let mut rest = svg;
    while let Some(open) = rest.find('<') {
        let Some(close) = rest[open..].find('>') else { break };
        let tag = &rest[open + 1..open + close];
        rest = &rest[open + close + 1..];
        let name = tag.split_whitespace().next().unwrap_or("");
        // Normalise whitespace so attribute lookup can rely on single spaces.
        let flat: String = tag.split_whitespace().collect::<Vec<_>>().join(" ");
        let flat = format!(" {}", flat.trim_start_matches(name).trim_start());
        match name {
            "path" => {
                if let Some(d) = attr(&flat, "d") {
                    shapes.push(Shape { polys: flatten_path(d) });
                }
            }
            "circle" => {
                let (cx, cy, r) = (
                    number_attr(&flat, "cx"),
                    number_attr(&flat, "cy"),
                    number_attr(&flat, "r"),
                );
                let steps = 64;
                let points = (0..steps)
                    .map(|i| {
                        let a = i as f32 / steps as f32 * std::f32::consts::TAU;
                        (cx + r * a.cos(), cy + r * a.sin())
                    })
                    .collect();
                shapes.push(Shape { polys: vec![Poly { points, closed: true }] });
            }
            "rect" => {
                let (x, y, w, h) = (
                    number_attr(&flat, "x"),
                    number_attr(&flat, "y"),
                    number_attr(&flat, "width"),
                    number_attr(&flat, "height"),
                );
                let rx = number_attr(&flat, "rx").min(w * 0.5).min(h * 0.5);
                shapes.push(Shape { polys: vec![rounded_rect(x, y, w, h, rx)] });
            }
            "line" => {
                let a = (number_attr(&flat, "x1"), number_attr(&flat, "y1"));
                let b = (number_attr(&flat, "x2"), number_attr(&flat, "y2"));
                shapes.push(Shape { polys: vec![Poly { points: vec![a, b], closed: false }] });
            }
            _ => {}
        }
    }
    shapes
}

fn rounded_rect(x: f32, y: f32, w: f32, h: f32, r: f32) -> Poly {
    let mut points = Vec::new();
    let corners = [
        (x + w - r, y + r, -90.0f32),
        (x + w - r, y + h - r, 0.0),
        (x + r, y + h - r, 90.0),
        (x + r, y + r, 180.0),
    ];
    for (cx, cy, start) in corners {
        if r <= 0.0 {
            points.push((cx, cy));
            continue;
        }
        for i in 0..=8 {
            let a = (start + 90.0 * i as f32 / 8.0).to_radians();
            points.push((cx + r * a.cos(), cy + r * a.sin()));
        }
    }
    Poly { points, closed: true }
}

/// A scanner over SVG path data.
struct Scan<'a> {
    s: &'a [u8],
    i: usize,
}

impl Scan<'_> {
    fn skip(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] == b' ' || self.s[self.i] == b',' || self.s[self.i] == b'\n') {
            self.i += 1;
        }
    }

    fn command(&mut self) -> Option<u8> {
        self.skip();
        let c = *self.s.get(self.i)?;
        if c.is_ascii_alphabetic() {
            self.i += 1;
            Some(c)
        } else {
            None
        }
    }

    fn has_number(&mut self) -> bool {
        self.skip();
        self.s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.'))
    }

    fn number(&mut self) -> f32 {
        self.skip();
        let start = self.i;
        if matches!(self.s.get(self.i), Some(b'-') | Some(b'+')) {
            self.i += 1;
        }
        let mut dot = false;
        while let Some(&c) = self.s.get(self.i) {
            if c.is_ascii_digit() {
                self.i += 1;
            } else if c == b'.' && !dot {
                dot = true;
                self.i += 1;
            } else if (c == b'e' || c == b'E') && self.i > start {
                self.i += 1;
                if matches!(self.s.get(self.i), Some(b'-') | Some(b'+')) {
                    self.i += 1;
                }
            } else {
                break;
            }
        }
        std::str::from_utf8(&self.s[start..self.i])
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(0.0)
    }

    /// An arc flag is one digit and may touch the next number.
    fn flag(&mut self) -> bool {
        self.skip();
        let on = self.s.get(self.i) == Some(&b'1');
        self.i += 1;
        on
    }
}

fn flatten_path(d: &str) -> Vec<Poly> {
    let mut scan = Scan { s: d.as_bytes(), i: 0 };
    let mut polys: Vec<Poly> = Vec::new();
    let mut cur: P = (0.0, 0.0);
    let mut start: P = (0.0, 0.0);
    let mut points: Vec<P> = Vec::new();
    // The last control point, for the smooth commands.
    let mut last_cubic: Option<P> = None;
    let mut last_quad: Option<P> = None;
    let mut cmd = b'M';

    let finish = |points: &mut Vec<P>, polys: &mut Vec<Poly>, closed: bool| {
        if points.len() > 1 || (closed && !points.is_empty()) {
            polys.push(Poly { points: std::mem::take(points), closed });
        } else {
            points.clear();
        }
    };

    loop {
        if let Some(c) = scan.command() {
            cmd = c;
        } else if !scan.has_number() {
            break;
        }
        let rel = cmd.is_ascii_lowercase();
        let (ox, oy) = if rel { cur } else { (0.0, 0.0) };
        match cmd.to_ascii_uppercase() {
            b'M' => {
                finish(&mut points, &mut polys, false);
                let p = (ox + scan.number(), oy + scan.number());
                cur = p;
                start = p;
                points.push(p);
                // Further pairs after a move are implicit line-tos.
                cmd = if rel { b'l' } else { b'L' };
                last_cubic = None;
                last_quad = None;
            }
            b'L' => {
                cur = (ox + scan.number(), oy + scan.number());
                points.push(cur);
                last_cubic = None;
                last_quad = None;
            }
            b'H' => {
                cur = (ox + scan.number(), cur.1);
                points.push(cur);
                last_cubic = None;
                last_quad = None;
            }
            b'V' => {
                cur = (cur.0, oy + scan.number());
                points.push(cur);
                last_cubic = None;
                last_quad = None;
            }
            b'C' | b'S' => {
                let c1 = if cmd.eq_ignore_ascii_case(&b'C') {
                    (ox + scan.number(), oy + scan.number())
                } else {
                    match last_cubic {
                        Some(c) => (2.0 * cur.0 - c.0, 2.0 * cur.1 - c.1),
                        None => cur,
                    }
                };
                let c2 = (ox + scan.number(), oy + scan.number());
                let end = (ox + scan.number(), oy + scan.number());
                cubic(&mut points, cur, c1, c2, end);
                last_cubic = Some(c2);
                last_quad = None;
                cur = end;
            }
            b'Q' | b'T' => {
                let c = if cmd.eq_ignore_ascii_case(&b'Q') {
                    (ox + scan.number(), oy + scan.number())
                } else {
                    match last_quad {
                        Some(c) => (2.0 * cur.0 - c.0, 2.0 * cur.1 - c.1),
                        None => cur,
                    }
                };
                let end = (ox + scan.number(), oy + scan.number());
                // A quadratic is a cubic with these control points.
                let c1 = (cur.0 + 2.0 / 3.0 * (c.0 - cur.0), cur.1 + 2.0 / 3.0 * (c.1 - cur.1));
                let c2 = (end.0 + 2.0 / 3.0 * (c.0 - end.0), end.1 + 2.0 / 3.0 * (c.1 - end.1));
                cubic(&mut points, cur, c1, c2, end);
                last_quad = Some(c);
                last_cubic = None;
                cur = end;
            }
            b'A' => {
                let (rx, ry, rot) = (scan.number(), scan.number(), scan.number());
                let (large, sweep) = (scan.flag(), scan.flag());
                let end = (ox + scan.number(), oy + scan.number());
                arc(&mut points, cur, rx, ry, rot, large, sweep, end);
                cur = end;
                last_cubic = None;
                last_quad = None;
            }
            b'Z' => {
                finish(&mut points, &mut polys, true);
                cur = start;
                points.push(cur);
                last_cubic = None;
                last_quad = None;
            }
            _ => break,
        }
    }
    finish(&mut points, &mut polys, false);
    polys
}

fn cubic(out: &mut Vec<P>, p0: P, p1: P, p2: P, p3: P) {
    let steps = 20;
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        let u = 1.0 - t;
        out.push((
            u * u * u * p0.0 + 3.0 * u * u * t * p1.0 + 3.0 * u * t * t * p2.0 + t * t * t * p3.0,
            u * u * u * p0.1 + 3.0 * u * u * t * p1.1 + 3.0 * u * t * t * p2.1 + t * t * t * p3.1,
        ));
    }
}

/// SVG elliptical arc, endpoint form to centre form (SVG spec, appendix F.6.5).
#[allow(clippy::too_many_arguments)]
fn arc(out: &mut Vec<P>, from: P, rx: f32, ry: f32, rot_deg: f32, large: bool, sweep: bool, to: P) {
    if (from.0 - to.0).abs() < 1e-6 && (from.1 - to.1).abs() < 1e-6 {
        return;
    }
    let (mut rx, mut ry) = (rx.abs(), ry.abs());
    if rx < 1e-6 || ry < 1e-6 {
        out.push(to);
        return;
    }
    let phi = rot_deg.to_radians();
    let (cos, sin) = (phi.cos(), phi.sin());
    let dx = (from.0 - to.0) * 0.5;
    let dy = (from.1 - to.1) * 0.5;
    let x1 = cos * dx + sin * dy;
    let y1 = -sin * dx + cos * dy;
    let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }
    let num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1;
    let den = rx * rx * y1 * y1 + ry * ry * x1 * x1;
    let mut k = (num / den).max(0.0).sqrt();
    if large == sweep {
        k = -k;
    }
    let cxp = k * rx * y1 / ry;
    let cyp = -k * ry * x1 / rx;
    let cx = cos * cxp - sin * cyp + (from.0 + to.0) * 0.5;
    let cy = sin * cxp + cos * cyp + (from.1 + to.1) * 0.5;
    let angle = |ux: f32, uy: f32, vx: f32, vy: f32| {
        let dot = ux * vx + uy * vy;
        let len = (ux * ux + uy * uy).sqrt() * (vx * vx + vy * vy).sqrt();
        let mut a = (dot / len).clamp(-1.0, 1.0).acos();
        if ux * vy - uy * vx < 0.0 {
            a = -a;
        }
        a
    };
    let theta = angle(1.0, 0.0, (x1 - cxp) / rx, (y1 - cyp) / ry);
    let mut delta = angle((x1 - cxp) / rx, (y1 - cyp) / ry, (-x1 - cxp) / rx, (-y1 - cyp) / ry);
    if !sweep && delta > 0.0 {
        delta -= std::f32::consts::TAU;
    } else if sweep && delta < 0.0 {
        delta += std::f32::consts::TAU;
    }
    let steps = ((delta.abs() * rx.max(ry) / 0.1).ceil() as usize).clamp(4, 96);
    for i in 1..=steps {
        let a = theta + delta * i as f32 / steps as f32;
        let (ex, ey) = (rx * a.cos(), ry * a.sin());
        out.push((cos * ex - sin * ey + cx, sin * ex + cos * ey + cy));
    }
    if let Some(last) = out.last_mut() {
        *last = to;
    }
}

// ---------------------------------------------------------------------------------------------
// Rasterizing
// ---------------------------------------------------------------------------------------------

fn dist_sq_to_segment(p: P, a: P, b: P) -> f32 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let len_sq = abx * abx + aby * aby;
    let t = if len_sq < 1e-12 {
        0.0
    } else {
        (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len_sq).clamp(0.0, 1.0)
    };
    let (dx, dy) = (p.0 - (a.0 + abx * t), p.1 - (a.1 + aby * t));
    dx * dx + dy * dy
}

/// Even-odd point-in-polygon over the closed polygons of one shape.
fn inside(p: P, polys: &[Poly]) -> bool {
    let mut crossings = 0;
    for poly in polys.iter().filter(|poly| poly.closed) {
        let n = poly.points.len();
        for i in 0..n {
            let (a, b) = (poly.points[i], poly.points[(i + 1) % n]);
            if (a.1 > p.1) != (b.1 > p.1) {
                let x = a.0 + (p.1 - a.1) / (b.1 - a.1) * (b.0 - a.0);
                if x > p.0 {
                    crossings += 1;
                }
            }
        }
    }
    crossings % 2 == 1
}

/// `px` x `px` RGBA: white, with the icon's coverage in alpha.
pub fn rasterize(icon: Icon, px: u32) -> Vec<u8> {
    let shapes = parse_svg(icon.svg());
    let scale = px as f32 / 24.0;
    let half = icon.stroke() * 0.5;
    let filled = icon.filled();
    let mut out = vec![0u8; (px * px * 4) as usize];
    for y in 0..px {
        for x in 0..px {
            let p = ((x as f32 + 0.5) / scale, (y as f32 + 0.5) / scale);
            let mut cover = 0.0f32;
            for shape in &shapes {
                let mut best = f32::MAX;
                for poly in &shape.polys {
                    let n = poly.points.len();
                    let segments = if poly.closed { n } else { n.saturating_sub(1) };
                    for i in 0..segments {
                        let d = dist_sq_to_segment(p, poly.points[i], poly.points[(i + 1) % n]);
                        best = best.min(d);
                    }
                }
                let d = best.sqrt();
                // One pixel of ramp, centred on the stroke's edge.
                cover = cover.max(((half - d) * scale + 0.5).clamp(0.0, 1.0));
                if filled && shape.polys.iter().any(|poly| poly.closed) {
                    let signed = if inside(p, &shape.polys) { -d } else { d };
                    cover = cover.max((0.5 - signed * scale).clamp(0.0, 1.0));
                }
            }
            let i = ((y * px + x) * 4) as usize;
            out[i..i + 4].copy_from_slice(&[255, 255, 255, (cover * 255.0).round() as u8]);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The equalizer: not an icon but a live drawing
// ---------------------------------------------------------------------------------------------

/// Three bars, low to high, that follow the sound being played: `levels` are the low, mid and high
/// band levels, each 0 to 1. When `live` is false the bars hold a fixed skyline, so a paused track
/// still reads as "this one".
pub fn equalizer(draw: &mut DrawList, center: P, size: f32, color: Color, levels: [f32; 3], live: bool) {
    let bar_w = (size * 0.2).max(2.0);
    let gap = bar_w * 0.8;
    let total = bar_w * 3.0 + gap * 2.0;
    let left = center.0 - total * 0.5;
    let base = center.1 + size * 0.5;
    for i in 0..3 {
        let h = if live {
            size * (0.2 + 0.8 * levels[i].clamp(0.0, 1.0))
        } else {
            size * [0.55, 1.0, 0.72][i]
        };
        let x = left + i as f32 * (bar_w + gap);
        draw.round_rect(Rect::new(x, base - h, bar_w, h), bar_w * 0.5, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_icon_parses_to_geometry_inside_the_grid() {
        for icon in Icon::ALL {
            let shapes = parse_svg(icon.svg());
            assert!(!shapes.is_empty(), "{icon:?} has no shapes");
            for poly in shapes.iter().flat_map(|s| &s.polys) {
                assert!(poly.points.len() >= 2, "{icon:?} has an empty outline");
                for p in &poly.points {
                    assert!(
                        (-0.5..=24.5).contains(&p.0) && (-0.5..=24.5).contains(&p.1),
                        "{icon:?} point {p:?} is off the grid"
                    );
                }
            }
        }
    }

    #[test]
    fn rasterized_icons_have_ink_and_clear_corners() {
        for icon in Icon::ALL {
            let px = 48;
            let pixels = rasterize(icon, px);
            let ink = pixels.chunks(4).filter(|p| p[3] > 128).count();
            assert!(ink > 20, "{icon:?} drew almost nothing ({ink} px)");
            assert_eq!(pixels[3], 0, "{icon:?} touches the top-left corner");
        }
    }

    #[test]
    fn filled_transport_icons_are_solid_in_the_middle() {
        let px = 48u32;
        // Alpha at a point given on the 24 unit grid.
        let alpha = |pixels: &[u8], x: f32, y: f32| {
            let (col, row) = ((x * 2.0) as u32, (y * 2.0) as u32);
            pixels[((row * px + col) * 4 + 3) as usize]
        };
        let play = rasterize(Icon::Play, px);
        assert_eq!(alpha(&play, 10.0, 12.0), 255, "play should be filled");
        // Inside the house's doorway: an outline icon must leave it empty.
        let house = rasterize(Icon::House, px);
        assert_eq!(alpha(&house, 12.0, 16.0), 0, "an outline icon must not fill");
    }

    #[test]
    fn textures_match_device_size() {
        assert_eq!(bucket(15.0), 15);
        assert_eq!(bucket(22.4), 22);
        assert_eq!(bucket(16.5), 17);
        assert_eq!(bucket(300.0), 300);
        assert_eq!(nearest_buckets(22).next(), Some(22));
    }
}
