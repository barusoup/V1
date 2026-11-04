//! Lazy per-glyph outline compiler and scanline coverage rasterizer.
//!
//! Instead of tessellating all thousands of glyphs in a font file upfront at
//! load time (which costs 10+ MB of heap per font even when an app only draws
//! a few dozen characters), this module compiles each glyph outline on demand
//! the first time it is rasterized and caches the compiled segments by glyph id.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

const CURVE_SCALE: f32 = 100.0;

#[derive(Copy, Clone, Default)]
struct Aabb {
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
}

#[derive(Copy, Clone, Default, PartialEq)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Copy, Clone)]
struct QuadCurve {
    a: Point,
    b: Point,
    c: Point,
}

impl QuadCurve {
    fn point(self, t: f32) -> Point {
        let tm = 1.0 - t;
        let a = tm * tm;
        let b = 2.0 * tm * t;
        let c = t * t;
        Point::new(
            a * self.a.x + b * self.b.x + c * self.c.x,
            a * self.a.y + b * self.b.y + c * self.c.y,
        )
    }
}

#[derive(Copy, Clone)]
struct CubeCurve {
    a: Point,
    b: Point,
    c: Point,
    d: Point,
}

impl CubeCurve {
    fn point(self, t: f32) -> Point {
        let tm = 1.0 - t;
        let a = tm * tm * tm;
        let b = 3.0 * (tm * tm) * t;
        let c = 3.0 * tm * (t * t);
        let d = t * t * t;
        Point::new(
            a * self.a.x + b * self.b.x + c * self.c.x + d * self.d.x,
            a * self.a.y + b * self.b.y + c * self.c.y + d * self.d.y,
        )
    }
}

#[derive(Copy, Clone)]
struct Line {
    coords: [f32; 4],
    nudge: [u32; 4],
    adj: [f32; 2],
    params: [f32; 4],
}

impl Line {
    fn new(start: Point, end: Point) -> Self {
        let (x_start_nudge, x_first_adj) = if end.x >= start.x { (0, 1.0) } else { (1, 0.0) };
        let (y_start_nudge, y_first_adj) = if end.y >= start.y { (0, 1.0) } else { (1, 0.0) };
        let x_end_nudge = u32::from(end.x > start.x);
        let y_end_nudge = u32::from(end.y > start.y);
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let tdx = if dx == 0.0 { f32::MAX } else { 1.0 / dx };
        let tdy = 1.0 / dy;
        Self {
            coords: [start.x, start.y, end.x, end.y],
            nudge: [x_start_nudge, y_start_nudge, x_end_nudge, y_end_nudge],
            adj: [x_first_adj, y_first_adj],
            params: [tdx, tdy, dx, dy],
        }
    }

    fn reposition(&mut self, bounds: Aabb, reverse: bool) {
        let [mut x0, mut y0, mut x1, mut y1] = if reverse {
            [
                self.coords[2],
                self.coords[3],
                self.coords[0],
                self.coords[1],
            ]
        } else {
            self.coords
        };
        x0 -= bounds.xmin;
        y0 = (y0 - bounds.ymax).abs();
        x1 -= bounds.xmin;
        y1 = (y1 - bounds.ymax).abs();
        *self = Self::new(Point::new(x0, y0), Point::new(x1, y1));
    }
}

struct Segment {
    a: Point,
    at: f32,
    c: Point,
    ct: f32,
}

struct GeometryBuilder {
    v_lines: Vec<Line>,
    m_lines: Vec<Line>,
    bounds: Aabb,
    start_point: Point,
    previous_point: Point,
    area: f32,
    max_area: f32,
}

impl GeometryBuilder {
    fn new(units_per_em: f32) -> Self {
        Self {
            v_lines: Vec::new(),
            m_lines: Vec::new(),
            bounds: Aabb {
                xmin: f32::MAX,
                xmax: f32::MIN,
                ymin: f32::MAX,
                ymax: f32::MIN,
            },
            start_point: Point::default(),
            previous_point: Point::default(),
            area: 0.0,
            max_area: 6.0 * (units_per_em / CURVE_SCALE),
        }
    }

    fn push(&mut self, start: Point, end: Point) {
        if start.y.to_bits() != end.y.to_bits() {
            self.area += (end.y - start.y) * (end.x + start.x);
            if start.x.to_bits() == end.x.to_bits() {
                self.v_lines.push(Line::new(start, end));
            } else {
                self.m_lines.push(Line::new(start, end));
            }
            for p in [start, end] {
                self.bounds.xmin = self.bounds.xmin.min(p.x);
                self.bounds.xmax = self.bounds.xmax.max(p.x);
                self.bounds.ymin = self.bounds.ymin.min(p.y);
                self.bounds.ymax = self.bounds.ymax.max(p.y);
            }
        }
    }

    fn finalize(mut self, advance_width: f32, advance_height: f32) -> CompiledGlyph {
        let bounds = if self.v_lines.is_empty() && self.m_lines.is_empty() {
            fontdue::OutlineBounds::default()
        } else {
            let reverse = self.area > 0.0;
            for line in self.v_lines.iter_mut().chain(self.m_lines.iter_mut()) {
                line.reposition(self.bounds, reverse);
            }
            self.v_lines.shrink_to_fit();
            self.m_lines.shrink_to_fit();
            fontdue::OutlineBounds {
                xmin: self.bounds.xmin,
                ymin: self.bounds.ymin,
                width: self.bounds.xmax - self.bounds.xmin,
                height: self.bounds.ymax - self.bounds.ymin,
            }
        };
        CompiledGlyph {
            v_lines: self.v_lines,
            m_lines: self.m_lines,
            advance_width,
            advance_height,
            bounds,
        }
    }
}

impl OutlineBuilder for GeometryBuilder {
    fn move_to(&mut self, x0: f32, y0: f32) {
        let next = Point::new(x0, y0);
        self.start_point = next;
        self.previous_point = next;
    }

    fn line_to(&mut self, x0: f32, y0: f32) {
        let next = Point::new(x0, y0);
        self.push(self.previous_point, next);
        self.previous_point = next;
    }

    fn quad_to(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let ctrl = Point::new(x0, y0);
        let next = Point::new(x1, y1);
        let curve = QuadCurve {
            a: self.previous_point,
            b: ctrl,
            c: next,
        };
        let mut stack = vec![Segment {
            a: self.previous_point,
            at: 0.0,
            c: next,
            ct: 1.0,
        }];
        while let Some(seg) = stack.pop() {
            let bt = (seg.at + seg.ct) * 0.5;
            let b = curve.point(bt);
            let area =
                (b.x - seg.a.x) * (seg.c.y - seg.a.y) - (seg.c.x - seg.a.x) * (b.y - seg.a.y);
            if area.abs() > self.max_area {
                stack.push(Segment {
                    a: seg.a,
                    at: seg.at,
                    c: b,
                    ct: bt,
                });
                stack.push(Segment {
                    a: b,
                    at: bt,
                    c: seg.c,
                    ct: seg.ct,
                });
            } else {
                self.push(seg.a, seg.c);
            }
        }
        self.previous_point = next;
    }

    fn curve_to(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) {
        let c1 = Point::new(x0, y0);
        let c2 = Point::new(x1, y1);
        let next = Point::new(x2, y2);
        let curve = CubeCurve {
            a: self.previous_point,
            b: c1,
            c: c2,
            d: next,
        };
        let mut stack = vec![Segment {
            a: self.previous_point,
            at: 0.0,
            c: next,
            ct: 1.0,
        }];
        while let Some(seg) = stack.pop() {
            let bt = (seg.at + seg.ct) * 0.5;
            let b = curve.point(bt);
            let area =
                (b.x - seg.a.x) * (seg.c.y - seg.a.y) - (seg.c.x - seg.a.x) * (b.y - seg.a.y);
            if area.abs() > self.max_area {
                stack.push(Segment {
                    a: seg.a,
                    at: seg.at,
                    c: b,
                    ct: bt,
                });
                stack.push(Segment {
                    a: b,
                    at: bt,
                    c: seg.c,
                    ct: seg.ct,
                });
            } else {
                self.push(seg.a, seg.c);
            }
        }
        self.previous_point = next;
    }

    fn close(&mut self) {
        if self.start_point != self.previous_point {
            self.push(self.previous_point, self.start_point);
        }
        self.previous_point = self.start_point;
    }
}

struct CompiledGlyph {
    v_lines: Vec<Line>,
    m_lines: Vec<Line>,
    advance_width: f32,
    advance_height: f32,
    bounds: fontdue::OutlineBounds,
}

#[inline(always)]
fn sub_nudge(val: f32, nudge: u32) -> f32 {
    f32::from_bits(val.to_bits().wrapping_sub(nudge)).trunc()
}

#[inline(always)]
fn fract_signed(x: f32) -> f32 {
    x - x.trunc()
}

#[inline(always)]
fn is_negative_bits(x: f32) -> bool {
    x.to_bits() >= 0x8000_0000
}

fn rasterize_compiled(glyph: &CompiledGlyph, scale: f32) -> (fontdue::Metrics, Vec<u8>) {
    let bounds = glyph.bounds.scale(scale);
    let mut offset_x = fract_signed(bounds.xmin);
    let mut offset_y = fract_signed(1.0 - fract_signed(bounds.height) - fract_signed(bounds.ymin));
    if is_negative_bits(offset_x) {
        offset_x += 1.0;
    }
    if is_negative_bits(offset_y) {
        offset_y += 1.0;
    }
    let width = (bounds.width + offset_x).ceil() as i32 as usize;
    let height = (bounds.height + offset_y).ceil() as i32 as usize;
    let metrics = fontdue::Metrics {
        xmin: bounds.xmin.floor() as i32,
        ymin: bounds.ymin.floor() as i32,
        width,
        height,
        advance_width: scale * glyph.advance_width,
        advance_height: scale * glyph.advance_height,
        bounds,
    };
    let len = width * height;
    if len == 0 {
        return (metrics, Vec::new());
    }
    let mut a = vec![0.0f32; len + 3];
    let w_f = width as f32;
    let inv_scale = 1.0 / scale;

    for line in &glyph.v_lines {
        let x0 = line.coords[0] * scale + offset_x;
        let y0 = line.coords[1] * scale + offset_y;
        let x1 = line.coords[2] * scale + offset_x;
        let y1 = line.coords[3] * scale + offset_y;
        let start_x = sub_nudge(x0, line.nudge[0]);
        let start_y = sub_nudge(y0, line.nudge[1]);
        let end_x = sub_nudge(x1, line.nudge[2]);
        let end_y = sub_nudge(y1, line.nudge[3]);
        let mut target_y = start_y + line.adj[1];
        let sy = 1.0f32.copysign(y1 - y0);
        let mut y_prev = y0;
        let mut index = (start_x + start_y * w_f) as i32;
        let index_y_inc = w_f.copysign(sy) as i32;
        let mut dist = (start_y - end_y).abs() as i32;
        let mid_x = fract_signed(x0);
        while dist > 0 {
            dist -= 1;
            let h = y_prev - target_y;
            let m = h * mid_x;
            let idx = index as usize;
            a[idx] += h - m;
            a[idx + 1] += m;
            index += index_y_inc;
            y_prev = target_y;
            target_y += sy;
        }
        let h = y_prev - y1;
        let m = h * mid_x;
        let idx = (end_x + end_y * w_f) as i32 as usize;
        a[idx] += h - m;
        a[idx + 1] += m;
    }

    for line in &glyph.m_lines {
        let x0 = line.coords[0] * scale + offset_x;
        let y0 = line.coords[1] * scale + offset_y;
        let x1 = line.coords[2] * scale + offset_x;
        let y1 = line.coords[3] * scale + offset_y;
        let start_x = sub_nudge(x0, line.nudge[0]);
        let start_y = sub_nudge(y0, line.nudge[1]);
        let end_x = sub_nudge(x1, line.nudge[2]);
        let end_y = sub_nudge(y1, line.nudge[3]);
        let tdx_raw = line.params[0] * inv_scale;
        let tdy_raw = line.params[1] * inv_scale;
        let dx = line.params[2] * scale;
        let dy = line.params[3] * scale;
        let mut target_x = start_x + line.adj[0];
        let mut target_y = start_y + line.adj[1];
        let sx = 1.0f32.copysign(tdx_raw);
        let sy = 1.0f32.copysign(tdy_raw);
        let mut tmx = tdx_raw * (target_x - x0);
        let mut tmy = tdy_raw * (target_y - y0);
        let tdx = tdx_raw.abs();
        let tdy = tdy_raw.abs();
        let mut x_prev = x0;
        let mut y_prev = y0;
        let mut index = (start_x + start_y * w_f) as i32;
        let index_x_inc = sx as i32;
        let index_y_inc = w_f.copysign(sy) as i32;
        let mut dist = ((start_x - end_x).abs() + (start_y - end_y).abs()) as i32;
        while dist > 0 {
            dist -= 1;
            let prev_index = index as usize;
            let (x_next, y_next) = if tmx < tmy {
                let yn = tmx * dy + y0;
                let xn = target_x;
                tmx += tdx;
                target_x += sx;
                index += index_x_inc;
                (xn, yn)
            } else {
                let yn = target_y;
                let xn = tmy * dx + x0;
                tmy += tdy;
                target_y += sy;
                index += index_y_inc;
                (xn, yn)
            };
            let h = y_prev - y_next;
            let mid_x = fract_signed((x_prev + x_next) * 0.5);
            let m = h * mid_x;
            a[prev_index] += h - m;
            a[prev_index + 1] += m;
            x_prev = x_next;
            y_prev = y_next;
        }
        let h = y_prev - y1;
        let mid_x = fract_signed((x_prev + x1) * 0.5);
        let m = h * mid_x;
        let idx = (end_x + end_y * w_f) as i32 as usize;
        a[idx] += h - m;
        a[idx + 1] += m;
    }

    let mut out = vec![0u8; len];
    let mut acc = 0.0f32;
    for i in 0..len {
        acc += a[i];
        out[i] = (acc.abs() * 255.9).clamp(0.0, 255.0) as u8;
    }
    (metrics, out)
}

/// On-demand TTF font face: stores the raw font bytes + header metrics, and
/// only compiles glyph outlines for glyphs that actually get rasterized.
pub(crate) struct LazyFont {
    data: Vec<u8>,
    units_per_em: f32,
    line_metrics_unscaled: fontdue::LineMetrics,
    ascii_glyphs: [u16; 128],
    ascii_advances: [u16; 128],
    default_advance: u16,
    has_legacy_kern: bool,
    gpos_first_glyphs: Vec<u64>,
    kern_cache: Mutex<HashMap<u32, i16>>,
    compiled: Mutex<HashMap<u16, Arc<CompiledGlyph>>>,
}

impl LazyFont {
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        let face = Face::parse(data, 0).map_err(|_| "failed to parse TTF/OTF face")?;
        let units_per_em = f32::from(face.units_per_em().max(1));
        let asc = i32::from(face.ascender());
        let desc = i32::from(face.descender());
        let gap = i32::from(face.line_gap());
        let line_metrics_unscaled = fontdue::LineMetrics {
            ascent: asc as f32,
            descent: desc as f32,
            line_gap: gap as f32,
            new_line_size: (asc - desc + gap) as f32,
        };
        let mut ascii_glyphs = [0u16; 128];
        let mut ascii_advances = [0u16; 128];
        let default_advance = face.glyph_hor_advance(GlyphId(0)).unwrap_or(0);
        for b in 0u8..128 {
            let ch = b as char;
            let gid = face.glyph_index(ch).map_or(0, |g| g.0);
            ascii_glyphs[b as usize] = gid;
            let lookup = if gid != 0 { GlyphId(gid) } else { GlyphId(0) };
            ascii_advances[b as usize] = face.glyph_hor_advance(lookup).unwrap_or(0);
        }
        let has_legacy_kern = face.tables().kern.is_some();
        let num_glyphs = face.number_of_glyphs() as usize;
        let mut gpos_first_glyphs = vec![0u64; num_glyphs.div_ceil(64)];
        if let Some(gpos) = face.tables().gpos.as_ref() {
            use ttf_parser::gpos::{PairAdjustment, PositioningSubtable};
            let kern = ttf_parser::Tag::from_bytes(b"kern");
            for feature in gpos.features {
                if feature.tag != kern {
                    continue;
                }
                for lookup_index in feature.lookup_indices {
                    let Some(lookup) = gpos.lookups.get(lookup_index) else {
                        continue;
                    };
                    for subtable in lookup.subtables.into_iter::<PositioningSubtable>() {
                        if let PositioningSubtable::Pair(pair) = subtable {
                            let cov = match pair {
                                PairAdjustment::Format1 { coverage, .. } => coverage,
                                PairAdjustment::Format2 { coverage, .. } => coverage,
                            };
                            for gid in 0..num_glyphs {
                                if cov.get(GlyphId(gid as u16)).is_some() {
                                    gpos_first_glyphs[gid / 64] |= 1u64 << (gid % 64);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            data: data.to_vec(),
            units_per_em,
            line_metrics_unscaled,
            ascii_glyphs,
            ascii_advances,
            default_advance,
            has_legacy_kern,
            gpos_first_glyphs,
            kern_cache: Mutex::new(HashMap::new()),
            compiled: Mutex::new(HashMap::new()),
        })
    }

    #[inline]
    pub fn line_metrics(&self, px: f32) -> fontdue::LineMetrics {
        let s = px / self.units_per_em;
        fontdue::LineMetrics {
            ascent: self.line_metrics_unscaled.ascent * s,
            descent: self.line_metrics_unscaled.descent * s,
            line_gap: self.line_metrics_unscaled.line_gap * s,
            new_line_size: self.line_metrics_unscaled.new_line_size * s,
        }
    }

    #[inline]
    fn glyph_index(&self, ch: char) -> u16 {
        let code = ch as u32;
        if code < 128 {
            return self.ascii_glyphs[code as usize];
        }
        Face::parse(&self.data, 0)
            .ok()
            .and_then(|face| face.glyph_index(ch))
            .map_or(0, |g| g.0)
    }

    pub fn has_glyph(&self, ch: char) -> bool {
        self.glyph_index(ch) != 0
    }

    pub fn advance(&self, ch: char, px: f32) -> f32 {
        let code = ch as u32;
        let adv = if code < 128 {
            f32::from(self.ascii_advances[code as usize])
        } else {
            let Ok(face) = Face::parse(&self.data, 0) else {
                return 0.0;
            };
            let gid = face
                .glyph_index(ch)
                .filter(|g| g.0 != 0)
                .unwrap_or(GlyphId(0));
            f32::from(face.glyph_hor_advance(gid).unwrap_or(self.default_advance))
        };
        (adv * (px / self.units_per_em)).round()
    }

    pub fn kern(&self, left: char, right: char, px: f32) -> f32 {
        let first_id = self.glyph_index(left);
        if first_id == 0 {
            return 0.0;
        }
        let gid = first_id as usize;
        let in_gpos = self
            .gpos_first_glyphs
            .get(gid / 64)
            .is_some_and(|word| (word & (1u64 << (gid % 64))) != 0);
        if !in_gpos && !self.has_legacy_kern {
            return 0.0;
        }
        let second_id = self.glyph_index(right);
        if second_id == 0 {
            return 0.0;
        }
        let key = (u32::from(first_id) << 16) | u32::from(second_id);
        if let Some(&units) = self.kern_cache.lock().unwrap().get(&key) {
            return (f32::from(units) * (px / self.units_per_em)).round();
        }
        let Ok(face) = Face::parse(&self.data, 0) else {
            return 0.0;
        };
        let first = GlyphId(first_id);
        let second = GlyphId(second_id);
        let mut units = if in_gpos {
            super::gpos_pair_kern_units(&face, first, second)
        } else {
            None
        };
        if units.is_none()
            && self.has_legacy_kern
            && let Some(kern) = face.tables().kern
        {
            for subtable in kern.subtables {
                if subtable.horizontal
                    && !subtable.variable
                    && let Some(v) = subtable.glyphs_kerning(first, second)
                {
                    units = Some(v);
                    break;
                }
            }
        }
        let units = units.unwrap_or(0);
        let mut cache = self.kern_cache.lock().unwrap();
        if cache.len() >= 4096 {
            cache.clear();
        }
        cache.insert(key, units);
        (f32::from(units) * (px / self.units_per_em)).round()
    }

    pub fn rasterize(&self, ch: char, px: f32) -> (fontdue::Metrics, Vec<u8>) {
        if px <= 0.0 || !px.is_finite() {
            return (fontdue::Metrics::default(), Vec::new());
        }
        let gid_u16 = self.glyph_index(ch);
        let gid = GlyphId(gid_u16);
        let compiled = {
            let mut map = self.compiled.lock().unwrap();
            if let Some(existing) = map.get(&gid_u16) {
                Arc::clone(existing)
            } else {
                let Ok(face) = Face::parse(&self.data, 0) else {
                    return (fontdue::Metrics::default(), Vec::new());
                };
                if map.len() >= 512 {
                    map.clear();
                }
                let adv_w = face.glyph_hor_advance(gid).unwrap_or(0) as f32;
                let adv_h = face.glyph_ver_advance(gid).unwrap_or(0) as f32;
                let mut builder = GeometryBuilder::new(self.units_per_em);
                face.outline_glyph(gid, &mut builder);
                let built = Arc::new(builder.finalize(adv_w, adv_h));
                map.insert(gid_u16, Arc::clone(&built));
                built
            }
        };
        let scale = px / self.units_per_em;
        rasterize_compiled(&compiled, scale)
    }
}
