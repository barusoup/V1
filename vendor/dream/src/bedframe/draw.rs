//! Raw UI draw lists: quads, text, images, scissors, layers.
//!
//! Everything the widgets draw goes through [`DrawList`]; games can draw the
//! same primitives directly for custom HUDs, debug overlays or editors.
//!
//! Output is a [`UiFrame`]: flat vertex/index buffers plus ordered [`DrawCmd`]s.
//! Commands are merged when their texture and scissor match, and stable-sorted
//! by layer at [`DrawList::finish`], so layering never costs draw calls where
//! it is not used.

use super::atlas::Region;
use super::font::{Fonts, GlyphCache, PositionedGlyph, TextLayout};
use super::{Color, Rect};
use crate::texture::GpuTexture;
use std::sync::Arc;

/// One UI vertex: pixel-space position, atlas/texture UV, sRGB colour and
/// shape parameters.
///
/// For textured quads `params` is all zero and `uv` is the texture sample
/// coordinate. For an anti-aliased rounded rectangle (see
/// [`DrawList::round_rect`]) `uv` instead holds the pixel position relative
/// to the rect's centre and `params` is `[half_w, half_h, radius, 1]`, which
/// the UI shader turns into a signed-distance coverage term. Keeping both in
/// the one vertex format means shapes and glyphs still batch into a single
/// draw call.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
    /// Shape parameters: `[half_w, half_h, radius, mode]`; zero for plain
    /// textured quads. The mode doubles as the sign selector for the
    /// negative modes: `-1` = textured quad under a rounded clip, `-2` =
    /// radial gradient, `-(5 + softness)` = soft shadow/glow. Positive
    /// modes keep their historic meaning (`0` quad, `1` rounded fill,
    /// `2` rounded texture, `3 + thickness` ring).
    pub params: [f32; 4],
    /// Extra shape data. Mode `-1` carries the rounded clip's half extent
    /// (`[half_w, half_h]`). Modes `2` and `-2` carry the anti-aliasing pad
    /// `[pad, pad]`: their quad is grown by it, so the shader steps back
    /// from the quad origin to the shape's own origin. Every other mode
    /// leaves it zero. Kept as a separate attribute so the four historic
    /// channels stay compatible.
    pub extra: [f32; 2],
}

/// Shape parameters for a plain textured quad.
const QUAD_PARAMS: [f32; 4] = [0.0; 4];
/// Extra shape data for a plain textured quad (see [`UiVertex::extra`]).
const QUAD_EXTRA: [f32; 2] = [0.0; 2];
/// Mode selector for [`DrawList::ring_arc`]'s SDF stroke. Negative so it can
/// never collide with the `3 + thickness` ring mode; carries `[mid_radius,
/// half_thickness, start, -3]` in `params` and the sweep in `extra.x`.
const ARC_MODE: f32 = -3.0;

/// Axis a two-colour linear gradient runs along. Explicit directions —
/// a gradient is authored geometry, not a style preset, so the developer
/// names the axis rather than configuring a generic vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientAxis {
    TopToBottom,
    BottomToTop,
    LeftToRight,
    RightToLeft,
}

/// Texture a command samples from.
#[derive(Clone)]
pub enum CmdTexture {
    /// Opaque white: solid colour quads.
    White,
    /// A page of the UI atlas (font glyphs, packed icons).
    AtlasPage(usize),
    /// A game texture (images, portraits, item icons).
    Custom(Arc<GpuTexture>),
    /// The frame so far, blurred: everything drawn before the first
    /// command that uses this is captured and blurred by the renderer, then
    /// this samples the result. Used for frosted-glass surfaces.
    Backdrop,
}

impl std::fmt::Debug for CmdTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CmdTexture::White => write!(f, "White"),
            CmdTexture::AtlasPage(page) => write!(f, "AtlasPage({page})"),
            CmdTexture::Custom(_) => write!(f, "Custom(..)"),
            CmdTexture::Backdrop => write!(f, "Backdrop"),
        }
    }
}

impl PartialEq for CmdTexture {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CmdTexture::White, CmdTexture::White) => true,
            (CmdTexture::AtlasPage(a), CmdTexture::AtlasPage(b)) => a == b,
            (CmdTexture::Custom(a), CmdTexture::Custom(b)) => Arc::ptr_eq(a, b),
            (CmdTexture::Backdrop, CmdTexture::Backdrop) => true,
            _ => false,
        }
    }
}

/// One draw command: a triangle range with fixed texture and scissor.
#[derive(Clone, Debug)]
pub struct DrawCmd {
    pub texture: CmdTexture,
    /// Scissor in pixel space; `None` means the full target.
    pub scissor: Option<Rect>,
    pub first_index: u32,
    pub index_count: u32,
    /// Layer this command sorts into (higher draws later, on top).
    pub layer: i32,
}

/// An atlas page upload carried by a [`UiFrame`]: the pixels of `rect` must
/// be copied into the page's texture before its draws execute.
#[derive(Clone, Debug)]
pub struct AtlasUpdate {
    pub rect: super::atlas::DirtyRect,
    /// Tightly packed RGBA rows (`width * height * 4` bytes).
    pub pixels: Vec<u8>,
}

/// Built per frame; handed to `Renderer::render_ui` (see [`crate::ui::render`]).
#[derive(Clone, Debug, Default)]
pub struct UiFrame {
    pub vertices: Vec<UiVertex>,
    pub indices: Vec<u32>,
    pub cmds: Vec<DrawCmd>,
    /// Atlas pages changed this frame, to upload before drawing (legacy
    /// path for standalone/raw frames — `atlas_snapshot` takes precedence).
    pub atlas_updates: Vec<AtlasUpdate>,
    /// Immutable atlas contents at `end` time: the renderer diffs this
    /// against its own upload record and keeps the pixels alive for as
    /// long as the frame might be retained.
    pub atlas_snapshot: Option<std::sync::Arc<super::atlas::AtlasSnapshot>>,
    /// Screen size the coordinates are in.
    pub size: (f32, f32),
    /// Optional dirty rectangle in UI pixel coordinates. When `Some`,
    /// renderers may clip draws to this region (skipping draws outside it)
    /// when updating a retained UI target. `None` redraws the full frame.
    pub dirty_rect: Option<Rect>,
}

impl UiFrame {
    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    /// Restricts rendering to `rect` (in UI pixel coordinates) when
    /// updating a retained UI target.
    pub fn with_dirty_rect(mut self, rect: Rect) -> Self {
        self.dirty_rect = Some(rect);
        self
    }

    /// Total quads submitted (diagnostics and tests).
    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }
}

/// Accumulates UI geometry for one frame.
///
/// Coordinates are pixels with the origin at the screen's top-left. Draw calls
/// append triangles; `finish` produces the [`UiFrame`].
/// Style for [`DrawList::text`]: which face, how large, what colour.
#[derive(Clone, Copy, Debug)]
pub struct TextParams {
    pub font: super::font::FontId,
    pub px: f32,
    pub color: Color,
}

impl TextParams {
    pub fn new(font: super::font::FontId, px: f32) -> Self {
        Self {
            font,
            px,
            color: Color::WHITE,
        }
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

/// One clip-scope entry: the square scissor every draw respects on the
/// GPU, plus an optional rounded corner shape the fragment stage carves.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Clip {
    /// Plain scissored rectangle.
    Square(Rect),
    /// Scissored rectangle whose corners are rounded in the shader for
    /// the plain-textured quads (fills, glyphs, icons, images) drawn
    /// under it.
    Rounded { rect: Rect, radius: f32 },
}

impl Clip {
    fn rect(self) -> Rect {
        match self {
            Clip::Square(rect) | Clip::Rounded { rect, .. } => rect,
        }
    }
}

pub struct DrawList {
    vertices: Vec<UiVertex>,
    indices: Vec<u32>,
    cmds: Vec<DrawCmd>,
    layer: i32,
    /// When false, `culled` never fires (view-layer scratch records
    /// off-screen content that later splices back on-screen).
    pub(crate) cull: bool,
    scissor_stack: Vec<Clip>,
    size: (f32, f32),
    /// Logical-to-physical multiplier applied by [`Self::finish`]: draw
    /// calls are submitted in logical units, the frame comes out physical.
    scale: f32,
}

impl Default for DrawList {
    fn default() -> Self {
        Self::new()
    }
}

/// Saved draw-list position for the paint-only replay rollback:
/// buffer lengths plus the tail cmd's index count, scissor depth, layer.
pub(crate) struct Checkpoint {
    vertices: usize,
    indices: usize,
    cmds: usize,
    last_indices: usize,
    scissor: usize,
    layer: i32,
}

impl DrawList {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            cmds: Vec::new(),
            layer: 0,
            cull: true,
            scissor_stack: Vec::new(),
            size: (0.0, 0.0),
            scale: 1.0,
        }
    }

    /// Reset for a new frame; keeps allocations.
    pub fn begin(&mut self, size: (f32, f32)) {
        self.vertices.clear();
        self.indices.clear();
        self.cmds.clear();
        self.layer = 0;
        self.scissor_stack.clear();
        self.size = size;
    }

    /// Reclaim a finished frame's buffers: whichever allocation of the
    /// current and the recycled one is larger survives, emptied. Never
    /// called while the list is mid-frame.
    pub(crate) fn recycle(&mut self, mut frame: UiFrame) {
        fn keep<T>(live: &mut Vec<T>, mut back: Vec<T>) {
            back.clear();
            if back.capacity() > live.capacity() {
                *live = back;
            }
        }
        keep(&mut self.vertices, std::mem::take(&mut frame.vertices));
        keep(&mut self.indices, std::mem::take(&mut frame.indices));
        keep(&mut self.cmds, std::mem::take(&mut frame.cmds));
    }

    /// Draw calls after this point sort into `layer` (higher is on top).
    /// Returns the previous layer so callers can restore it.
    pub fn set_layer(&mut self, layer: i32) -> i32 {
        std::mem::replace(&mut self.layer, layer)
    }

    pub fn layer(&self) -> i32 {
        self.layer
    }

    /// Logical-to-physical scale: geometry is accumulated in logical units
    /// and [`Self::finish`] emits it in physical pixels. Defaults to 1.0 —
    /// a bare draw list is always physical. Only positive finite values
    /// mean anything; anything else falls back to 1.0.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
    }

    /// The scale [`Self::finish`] applies (see [`Self::set_scale`]).
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Outward grow of an SDF quad, in logical units: one physical pixel.
    /// The shaders put a 1px anti-aliasing band centred on each shape edge,
    /// so the outer half of that band lies outside the shape's own rect and
    /// would never be rasterized unless the quad grows by this much.
    fn aa_pad(&self) -> f32 {
        1.0 / self.scale
    }

    /// Capture the cursor plus mutable tail state (tail cmd index count,
    /// scissor depth, layer) so `restore` rolls back cleanly.
    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            vertices: self.vertices.len(),
            indices: self.indices.len(),
            cmds: self.cmds.len(),
            last_indices: self.cmds.last().map_or(0, |c| c.index_count as usize),
            scissor: self.scissor_stack.len(),
            layer: self.layer,
        }
    }

    /// Roll back to `cp`, restoring the tail cmd's index count, the
    /// scissor stack and the layer.
    pub(crate) fn restore(&mut self, cp: Checkpoint) {
        self.vertices.truncate(cp.vertices);
        self.indices.truncate(cp.indices);
        self.cmds.truncate(cp.cmds);
        if let Some(c) = self.cmds.last_mut() {
            c.index_count = cp.last_indices as u32;
        }
        self.scissor_stack.truncate(cp.scissor);
        self.layer = cp.layer;
    }

    /// Byte ranges a recorded paint op captured for later splicing.
    pub(crate) fn cursor(&self) -> (usize, usize, usize) {
        (self.vertices.len(), self.indices.len(), self.cmds.len())
    }

    /// Splice the recorded range `start..end` (cursors from
    /// [`Self::cursor`]) of `src` into this list translated by `offset`
    /// (logical px). Indices rebase onto the appended vertex block; the
    /// mode -1 rounded-clip centre in params[0..2] translates too; cmds
    /// re-push at the current layer, recorded scissor translated and
    /// intersected with the current clip. A cmd that straddles the range
    /// edge (recording merged into the batch before it) is clamped.
    pub(crate) fn splice_from(
        &mut self,
        src: &DrawList,
        start: (usize, usize, usize),
        end: (usize, usize, usize),
        offset: (f32, f32),
    ) {
        let ((v0, i0, c0), (v1, i1, c1)) = (start, end);
        let base = self.vertices.len() as u32;
        for v in &src.vertices[v0..v1] {
            let mut v = *v;
            v.pos[0] += offset.0;
            v.pos[1] += offset.1;
            if v.params[3] == -1.0 {
                v.params[0] += offset.0;
                v.params[1] += offset.1;
            }
            // Modes 2 and -2 need no extra work: their shape origin is the
            // quad origin, which moved with `pos`, and `extra` is a relative
            // pad.
            self.vertices.push(v);
        }
        let index_base = self.indices.len() as u32;
        for &i in &src.indices[i0..i1] {
            self.indices.push(base + (i - v0 as u32));
        }
        let (i0, i1) = (i0 as u32, i1 as u32);
        for cmd in &src.cmds[c0.saturating_sub(1)..c1] {
            let lo = cmd.first_index.max(i0);
            let hi = (cmd.first_index + cmd.index_count).min(i1);
            if hi <= lo {
                continue;
            }
            let first_index = index_base + (lo - i0);
            let index_count = hi - lo;
            // Spliced geometry is clipped by BOTH the recorded scissor
            // (translated) and the clip live at splice time; an absent
            // recorded scissor inherits the current clip, and an absent
            // current clip keeps the recorded one.
            let scissor = match (
                cmd.scissor
                    .map(|r| Rect::new(r.x + offset.0, r.y + offset.1, r.w, r.h)),
                self.clip_rect(),
            ) {
                (Some(r), Some(c)) => Some(c.intersect(r).unwrap_or_default()),
                (Some(r), None) => Some(r),
                (None, c) => c,
            };
            if let Some(last) = self.cmds.last_mut()
                && last.texture == cmd.texture
                && last.scissor == scissor
                && last.layer == self.layer
                && last.first_index + last.index_count == first_index
            {
                last.index_count += index_count;
            } else {
                self.cmds.push(DrawCmd {
                    texture: cmd.texture.clone(),
                    scissor,
                    first_index,
                    index_count,
                    layer: self.layer,
                });
            }
        }
    }

    /// Clip draw calls after this point to `rect`, intersected with the
    /// current clip. [`Self::pop_clip`] restores the previous one.
    pub fn push_clip(&mut self, rect: Rect) {
        let rect = match self.scissor_stack.last() {
            Some(prev) => prev.rect().intersect(rect).unwrap_or(Rect::default()),
            None => rect,
        };
        self.scissor_stack.push(Clip::Square(rect));
    }

    /// Clip draw calls after this point to `rect` with rounded corners of
    /// `radius` pixels, intersected with the current clip. The GPU scissor
    /// still applies (so the square bound is hard), and the fragment stage
    /// additionally carves the corners for every plain-textured quad drawn
    /// under the scope: solid fills, glyphs, icons and images all follow
    /// the rounding. A square [`Self::round_rect`] (radius 0) is drawn as a
    /// plain quad under this scope and is carved too. The rounded SDF shape
    /// primitives (`round_rect` with a radius, `round_ring`, `ring_arc`,
    /// rounded images and shadows) keep their own geometry and are clipped
    /// by the square bound only — give such a shape the clip's radius as its
    /// own and it agrees with the rounding by construction.
    pub fn push_clip_rounded(&mut self, rect: Rect, radius: f32) {
        let rect = match self.scissor_stack.last() {
            Some(prev) => prev.rect().intersect(rect).unwrap_or(Rect::default()),
            None => rect,
        };
        let radius = radius.clamp(0.0, rect.w.min(rect.h) * 0.5);
        self.scissor_stack.push(Clip::Rounded { rect, radius });
    }

    pub fn pop_clip(&mut self) {
        self.scissor_stack.pop();
    }

    fn current_scissor(&self) -> Option<Rect> {
        self.scissor_stack
            .last()
            .map(|clip| clip.rect())
            .filter(|r| r.w > 0.0 && r.h > 0.0)
    }

    /// The clip currently in force, including an empty one (widgets use it
    /// to keep input and drawing in agreement). `None` is unclipped.
    pub(crate) fn clip_rect(&self) -> Option<Rect> {
        self.scissor_stack.last().map(|clip| clip.rect())
    }

    /// Lift the whole clip stack out of the way (an overlay that must escape
    /// its enclosing panels). Pair with [`Self::restore_clip`].
    pub(crate) fn suspend_clip(&mut self) -> Vec<Rect> {
        std::mem::take(&mut self.scissor_stack)
            .into_iter()
            .map(Clip::rect)
            .collect()
    }

    /// Put back a stack taken by [`Self::suspend_clip`]. Restored entries
    /// lose their rounding (square scissor only) — an escaped overlay draws
    /// above the panels that owned the rounding anyway.
    pub(crate) fn restore_clip(&mut self, stack: Vec<Rect>) {
        self.scissor_stack = stack.into_iter().map(Clip::Square).collect();
    }

    /// True when the clip stack has an empty top: draw calls must vanish,
    /// not fall back to unclipped.
    fn clipped_away(&self) -> bool {
        self.scissor_stack
            .last()
            .is_some_and(|clip| clip.rect().w <= 0.0 || clip.rect().h <= 0.0)
    }

    /// Coarse cull: a rect that can show no pixels — an empty clip above,
    /// fully outside the current clip or fully outside the screen. Kept
    /// cheap on purpose (axis-aligned tests only); exact raster clipping
    /// still happens in the fragment stage.
    fn culled(&self, rect: Rect) -> bool {
        if self.clipped_away() {
            return true;
        }
        // `cull == false` (view scratch) only bypasses the screen test —
        // a recorded widget may live off-screen yet still be clipped by
        // its own scissor at splice time.
        let screen = Rect::new(0.0, 0.0, self.size.0, self.size.1);
        if self.cull && screen.intersect(rect).is_none() {
            return true;
        }
        self.current_scissor()
            .is_some_and(|clip| clip.intersect(rect).is_none())
    }

    /// Axis-aligned quad. `uv` is `(u0, v0, u1, v1)`; `None` samples white.
    pub fn quad(&mut self, rect: Rect, uv: Option<[f32; 4]>, color: Color, texture: CmdTexture) {
        if color.a == 0 || self.culled(rect) {
            return;
        }
        let [u0, v0, u1, v1] = uv.unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let base = self.vertices.len() as u32;
        let c = color.to_array();
        // A rounded clip in force carves this quad's corners in the shader:
        // the clip's centre/radius ride in params, its half extent in
        // `extra` (mode -1 in `ui.wgsl`). Plain square clips keep the
        // historic all-zero params, so batching and instancing are
        // unchanged when rounding is not in use.
        let (params, extra) = match self.scissor_stack.last() {
            Some(Clip::Rounded { rect: clip, radius }) => (
                [clip.center().0, clip.center().1, *radius, -1.0],
                [clip.w * 0.5, clip.h * 0.5],
            ),
            _ => (QUAD_PARAMS, QUAD_EXTRA),
        };
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [rect.x, rect.y],
                uv: [u0, v0],
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [rect.x + rect.w, rect.y],
                uv: [u1, v0],
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [rect.x + rect.w, rect.y + rect.h],
                uv: [u1, v1],
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [rect.x, rect.y + rect.h],
                uv: [u0, v1],
                color: c,
                params,
                extra,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(texture, 6);
    }

    /// Anti-aliased rounded rectangle. `radius` is clamped to half the
    /// shorter side (a pill when equal). Coverage is resolved in the UI
    /// shader from a signed distance, so corners stay crisp at any size and
    /// the whole shape is still a single quad (one draw call, no corner
    /// geometry).
    ///
    /// A square shape (`radius <= 0`) under a rounded clip is emitted as a
    /// plain quad instead, so the clip's corner carve applies to it; a
    /// rounded square needs no such path because its radius is the clip's
    /// own.
    pub fn round_rect(&mut self, rect: Rect, radius: f32, color: Color) {
        if radius <= 0.0 && matches!(self.scissor_stack.last(), Some(Clip::Rounded { .. })) {
            if rect.w > 0.0 && rect.h > 0.0 {
                self.quad(rect, None, color, CmdTexture::White);
            }
            return;
        }
        self.push_round(rect, radius, color, 1.0);
    }

    /// Hollow rounded outline: `thickness` pixels inside `rect`'s edge, with
    /// the outer corner radius `radius`. Transparent interior, so whatever is
    /// behind shows through.
    pub fn round_ring(&mut self, rect: Rect, radius: f32, thickness: f32, color: Color) {
        self.push_round(rect, radius, color, 3.0 + thickness.max(0.0));
    }

    /// Shared quad emit for filled rounded shapes (`mode` 1) and rings
    /// (`mode` 3 + thickness); see `ui.wgsl`.
    fn push_round(&mut self, rect: Rect, radius: f32, color: Color, mode: f32) {
        let pad = self.aa_pad();
        let quad = rect.expand(pad);
        if color.a == 0 || rect.w <= 0.0 || rect.h <= 0.0 || self.culled(quad) {
            return;
        }
        let (hw, hh) = (rect.w * 0.5, rect.h * 0.5);
        // The quad reaches `pad` past the shape; `uv` is measured from the
        // centre to those padded corners, while `params` keep the shape.
        let (ew, eh) = (hw + pad, hh + pad);
        let r = radius.clamp(0.0, hw.min(hh));
        let c = color.to_array();
        let params = [hw, hh, r, mode];
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [quad.x, quad.y],
                uv: [-ew, -eh],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y],
                uv: [ew, -eh],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y + quad.h],
                uv: [ew, eh],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.x, quad.y + quad.h],
                uv: [-ew, eh],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// Anti-aliased circular arc stroke: the band [`Self::round_ring`]
    /// paints, but only the `sweep` radians sweeping clockwise on screen
    /// from `start`. Angles are in radians measured from `+x`, so the top
    /// of the ring is `-FRAC_PI_2`; ends are round caps of
    /// `thickness / 2`, so a partial sweep reads as a stroke rather than a
    /// cut wedge.
    ///
    /// The arc is always circular, centred in `rect`, with its outer
    /// radius clamped to half the shorter side — it matches
    /// [`Self::round_ring`] exactly only for a square `rect` with
    /// `radius = side / 2`. Coverage comes from the same signed-distance
    /// path as `round_rect`, so the curve stays smooth at any size and
    /// the shape is still one instanced quad.
    pub fn ring_arc(
        &mut self,
        rect: Rect,
        radius: f32,
        thickness: f32,
        start: f32,
        sweep: f32,
        color: Color,
    ) {
        if color.a == 0 || thickness <= 0.0 || sweep <= 0.0 || self.clipped_away() {
            return;
        }
        if ![
            rect.x, rect.y, rect.w, rect.h, radius, thickness, start, sweep,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return;
        }
        let sweep = sweep.min(std::f32::consts::TAU);
        let (hw, hh) = (rect.w * 0.5, rect.h * 0.5);
        let r = radius.clamp(0.0, hw.min(hh));
        if r <= 0.0 {
            return;
        }
        // A thickness past the radius runs the band to the centre instead
        // of dropping the arc.
        let half_t = thickness.min(r) * 0.5;
        let r_mid = r - half_t;
        // The quad carries centre-relative positions in `uv`, so it is the
        // ring's bounds inflated by a pixel: the anti-aliased outer edge
        // needs that room and the centre stays put.
        let quad = Rect::new(rect.x - 1.0, rect.y - 1.0, rect.w + 2.0, rect.h + 2.0);
        if self.culled(quad) {
            return;
        }
        let c = color.to_array();
        let params = [r_mid, half_t, start, ARC_MODE];
        let extra = [sweep, 0.0];
        let centre = (rect.x + hw, rect.y + hh);
        let corner = |pos: (f32, f32)| UiVertex {
            pos: [pos.0, pos.1],
            uv: [pos.0 - centre.0, pos.1 - centre.1],
            color: c,
            params,
            extra,
        };
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            corner((quad.x, quad.y)),
            corner((quad.x + quad.w, quad.y)),
            corner((quad.x + quad.w, quad.y + quad.h)),
            corner((quad.x, quad.y + quad.h)),
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// A rounded rectangle with a 1px (or `border`) stroke: the ring of
    /// `stroke` is drawn on `rect`'s edge, `border` pixels deep, and the
    /// `fill` sits under it. Under an opaque stroke the fill is inset by
    /// `border * 0.5` on every side (its radius shrinks by the same amount,
    /// clamped at 0), so its anti-aliased edge lands under the middle of the
    /// ring rather than on the outer edge where the fill colour would bleed
    /// out as a halo. Otherwise (no stroke, or a translucent one that must
    /// tint over the fill) the fill covers the full `rect`. A transparent `fill` draws nothing, so a field with
    /// only a border (a text box inside a bar) does not paint its stroke
    /// colour across the whole interior.
    pub fn rounded_frame(
        &mut self,
        rect: Rect,
        radius: f32,
        border: f32,
        fill: Color,
        stroke: Color,
    ) {
        let stroked = border > 0.0 && stroke.a > 0;
        if fill.a > 0 {
            // Only an opaque ring can hide the fill's edge; under a
            // translucent ring the fill must reach the outer edge (CSS
            // paints the background under the whole border box).
            if stroked && stroke.a == 255 {
                let inset = border * 0.5;
                self.round_rect(rect.inset(inset), (radius - inset).max(0.0), fill);
            } else {
                self.round_rect(rect, radius, fill);
            }
        }
        if stroked {
            self.round_ring(rect, radius, border, stroke);
        }
    }

    /// Solid colour rectangle.
    pub fn rect(&mut self, rect: Rect, color: Color) {
        self.quad(rect, None, color, CmdTexture::White);
    }

    /// A game texture, optionally tinted.
    pub fn image(&mut self, rect: Rect, texture: &Arc<GpuTexture>, tint: Color) {
        self.quad(rect, None, tint, CmdTexture::Custom(Arc::clone(texture)));
    }

    /// A game texture clipped to a rounded rectangle of `radius`. Corners
    /// are anti-aliased by the same signed-distance path as
    /// [`DrawList::round_rect`]; still a single quad.
    pub fn image_rounded(
        &mut self,
        rect: Rect,
        texture: &Arc<GpuTexture>,
        tint: Color,
        radius: f32,
    ) {
        self.image_rounded_uv(rect, texture, tint, radius, (1.0, 1.0));
    }

    /// Like [`DrawList::image_rounded`], but the texture's `[0, uv_max]`
    /// range fills `rect`, so a non-square `rect` shows a crop of a square
    /// texture (top-left anchored) instead of stretching it. Corners round
    /// against `rect` itself.
    pub fn image_rounded_uv(
        &mut self,
        rect: Rect,
        texture: &Arc<GpuTexture>,
        tint: Color,
        radius: f32,
        uv_max: (f32, f32),
    ) {
        self.image_uv_rect(rect, texture, tint, radius, (0.0, 0.0, uv_max.0, uv_max.1));
    }

    /// Like [`DrawList::image_rounded_uv`], but the texture rectangle
    /// `(u0, v0)..(u1, v1)` fills `rect`. A caller that wants the least
    /// possible crop can centre the rectangle on the source and keep the
    /// quad's aspect, so nothing is stretched.
    pub fn image_uv_rect(
        &mut self,
        rect: Rect,
        texture: &Arc<GpuTexture>,
        tint: Color,
        radius: f32,
        uv: (f32, f32, f32, f32),
    ) {
        self.push_shape_image(
            rect,
            CmdTexture::Custom(Arc::clone(texture)),
            tint,
            radius,
            uv,
        );
    }

    /// Frosted glass: the frame behind `rect`, blurred, tinted by `tint` and
    /// clipped to a rounded rectangle of `radius`. The renderer captures and
    /// blurs everything drawn before the first glass command, so anything
    /// that should sit behind the glass must be drawn first. Corners are
    /// anti-aliased like [`Self::image_rounded`].
    pub fn backdrop_rounded(&mut self, rect: Rect, tint: Color, radius: f32) {
        // The backdrop covers the whole logical frame, so the rect's own
        // position, over the frame size, is its UV range.
        let (w, h) = self.size;
        let (w, h) = (w.max(1.0), h.max(1.0));
        let uv = (
            rect.x / w,
            rect.y / h,
            (rect.x + rect.w) / w,
            (rect.y + rect.h) / h,
        );
        self.push_shape_image(rect, CmdTexture::Backdrop, tint, radius, uv);
    }

    /// Geometry behind [`Self::image_uv_rect`], over any texture so the
    /// vertex layout can be tested without a GPU.
    fn push_shape_image(
        &mut self,
        rect: Rect,
        texture: CmdTexture,
        tint: Color,
        radius: f32,
        uv: (f32, f32, f32, f32),
    ) {
        let pad = self.aa_pad();
        let quad = rect.expand(pad);
        if tint.a == 0 || rect.w <= 0.0 || rect.h <= 0.0 || self.culled(quad) {
            return;
        }
        let r = radius.clamp(0.0, (rect.w * 0.5).min(rect.h * 0.5));
        // `params` keep the unpadded full size; the shader finds the shape
        // from the padded quad origin by stepping back `extra` = pad.
        let params = [rect.w, rect.h, r, 2.0];
        let extra = [pad, pad];
        let c = tint.to_array();
        // Extend the texture range linearly over the pad, so the padded
        // corners sample just past the shape's own UV edges.
        let (u0, v0, u1, v1) = uv;
        let du = pad * (u1 - u0) / rect.w;
        let dv = pad * (v1 - v0) / rect.h;
        let (u0, v0, u1, v1) = (u0 - du, v0 - dv, u1 + du, v1 + dv);
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [quad.x, quad.y],
                uv: [u0, v0],
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y],
                uv: [u1, v0],
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y + quad.h],
                uv: [u1, v1],
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [quad.x, quad.y + quad.h],
                uv: [u0, v1],
                color: c,
                params,
                extra,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(texture, 6);
    }

    /// Soft-edged shadow (or glow) under a rounded rectangle.
    ///
    /// Every aspect is a parameter — this is a primitive, not a preset:
    /// `rect` is the shadowed shape, `radius` its corner rounding, `offset`
    /// shifts the shadow (zero for a centred glow), `spread` grows the
    /// silhouette, and `softness` is the width, in pixels, of the fade
    /// band around the silhouette's edge (`0.0` = a hard edge with normal
    /// anti-aliasing). `color` usually carries its own alpha: a low alpha
    /// over a wide softness reads as a soft ambient shadow, a saturated
    /// accent at high alpha as a glow. Draw it before the shape it shadows;
    /// still one quad, so it batches with everything.
    pub fn shadow(
        &mut self,
        rect: Rect,
        radius: f32,
        offset: (f32, f32),
        spread: f32,
        softness: f32,
        color: Color,
    ) {
        if color.a == 0 {
            return;
        }
        let softness = softness.max(0.0);
        // The silhouette the shadow paints, grown by the spread and
        // shifted by the offset…
        let silhouette = rect.expand(spread).translate(offset.0, offset.1);
        if silhouette.w <= 0.0 || silhouette.h <= 0.0 {
            return;
        }
        // …and the quad that carries it, padded by half the fade band plus
        // a pixel so the coverage tail is never culled mid-fade.
        let quad = silhouette.expand(softness * 0.5 + 1.0);
        if self.culled(quad) {
            return;
        }
        let (hw, hh) = (silhouette.w * 0.5, silhouette.h * 0.5);
        let r = (radius + spread).clamp(0.0, hw.min(hh));
        let params = [hw, hh, r, -(5.0 + softness)];
        let c = color.to_array();
        let (qx, qy) = (quad.w * 0.5, quad.h * 0.5);
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [quad.x, quad.y],
                uv: [-qx, -qy],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.max().0, quad.y],
                uv: [qx, -qy],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.max().0, quad.max().1],
                uv: [qx, qy],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.x, quad.max().1],
                uv: [-qx, qy],
                color: c,
                params,
                extra: QUAD_EXTRA,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// CSS `box-shadow` (outer): `rect` is the element's border box with
    /// corner `radius`; `offset`, `blur` and `spread` are the CSS lengths
    /// (`blur` is the blur radius, a Gaussian of standard deviation
    /// `blur / 2`, as browsers render it). Unlike [`Self::shadow`]'s
    /// linear fade this is a true Gaussian, so layered shadows from a
    /// stylesheet come out with the same falloff. Draw it before the shape
    /// it shadows; one quad.
    pub fn box_shadow(
        &mut self,
        rect: Rect,
        radius: f32,
        offset: (f32, f32),
        blur: f32,
        spread: f32,
        color: Color,
    ) {
        if color.a == 0 {
            return;
        }
        let silhouette = rect.expand(spread).translate(offset.0, offset.1);
        if silhouette.w <= 0.0 || silhouette.h <= 0.0 {
            return;
        }
        let sigma = (blur.max(0.0) * 0.5).max(0.25);
        // Three standard deviations hold all but 0.3% of the blur.
        let quad = silhouette.expand(3.0 * sigma + 1.0);
        if self.culled(quad) {
            return;
        }
        let (hw, hh) = (silhouette.w * 0.5, silhouette.h * 0.5);
        // CSS grows the corner radius with the spread (and never below 0).
        let r = (radius + spread).clamp(0.0, hw.min(hh));
        let params = [hw, hh, r, -4.0];
        let extra = [sigma, 0.0];
        let c = color.to_array();
        let centre = silhouette.center();
        let corner = |x: f32, y: f32| UiVertex {
            pos: [x, y],
            uv: [x - centre.0, y - centre.1],
            color: c,
            params,
            extra,
        };
        let base = self.vertices.len() as u32;
        let (x0, y0, x1, y1) = (quad.x, quad.y, quad.max().0, quad.max().1);
        self.vertices.extend_from_slice(&[
            corner(x0, y0),
            corner(x1, y0),
            corner(x1, y1),
            corner(x0, y1),
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// Anti-aliased rounded rectangle with its own radius per corner, in
    /// CSS order `[top_left, top_right, bottom_right, bottom_left]` (the
    /// `border-radius` longhands). Drawn as four quadrant quads that share
    /// the rect's centre: inside a quadrant only that corner can be the
    /// nearest, so each evaluates the ordinary rounded-box distance with
    /// its own radius and the seams line up exactly.
    pub fn round_rect_corners(&mut self, rect: Rect, radii: [f32; 4], color: Color) {
        self.push_round_corners(rect, radii, color, 1.0);
    }

    /// Hollow outline with per-corner radii (see [`Self::round_rect_corners`]
    /// and [`Self::round_ring`]).
    pub fn round_ring_corners(
        &mut self,
        rect: Rect,
        radii: [f32; 4],
        thickness: f32,
        color: Color,
    ) {
        self.push_round_corners(rect, radii, color, 3.0 + thickness.max(0.0));
    }

    fn push_round_corners(&mut self, rect: Rect, radii: [f32; 4], color: Color, mode: f32) {
        if radii.iter().all(|&r| r == radii[0]) {
            return self.push_round(rect, radii[0], color, mode);
        }
        if color.a == 0 || rect.w <= 0.0 || rect.h <= 0.0 || self.culled(rect.expand(1.0)) {
            return;
        }
        let (hw, hh) = (rect.w * 0.5, rect.h * 0.5);
        let (cx, cy) = rect.center();
        let c = color.to_array();
        // Quadrants in radii order; each reaches one pixel past the rect's
        // outer edges for the anti-aliased band, and stops at the centre.
        let quadrants = [
            (rect.x - 1.0, rect.y - 1.0, cx, cy),
            (cx, rect.y - 1.0, rect.max().0 + 1.0, cy),
            (cx, cy, rect.max().0 + 1.0, rect.max().1 + 1.0),
            (rect.x - 1.0, cy, cx, rect.max().1 + 1.0),
        ];
        for (q, &radius) in quadrants.iter().zip(radii.iter()) {
            let params = [hw, hh, radius.clamp(0.0, hw.min(hh)), mode];
            let corner = |x: f32, y: f32| UiVertex {
                pos: [x, y],
                uv: [x - cx, y - cy],
                color: c,
                params,
                extra: QUAD_EXTRA,
            };
            let base = self.vertices.len() as u32;
            self.vertices.extend_from_slice(&[
                corner(q.0, q.1),
                corner(q.2, q.1),
                corner(q.2, q.3),
                corner(q.0, q.3),
            ]);
            self.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            self.push_cmd(CmdTexture::White, 6);
        }
    }

    /// Two-colour linear gradient across `rect` along `axis`. The colours
    /// ride the quad's corners, so the GPU interpolates them for free —
    /// one quad, batched like any other.
    pub fn gradient(&mut self, rect: Rect, from: Color, to: Color, axis: GradientAxis) {
        let [tl, tr, br, bl] = match axis {
            GradientAxis::TopToBottom => [from, from, to, to],
            GradientAxis::BottomToTop => [to, to, from, from],
            GradientAxis::LeftToRight => [from, to, to, from],
            GradientAxis::RightToLeft => [to, from, from, to],
        };
        self.gradient_corners(rect, [tl, tr, br, bl]);
    }

    /// Four-corner gradient: each corner of `rect` carries its own colour
    /// and the interior interpolates bilinearly, in
    /// `[top-left, top-right, bottom-right, bottom-left]` order. Diagonals,
    /// ramps and any two-axis blend fall out of this without a new
    /// primitive.
    pub fn gradient_corners(&mut self, rect: Rect, corners: [Color; 4]) {
        if rect.w <= 0.0
            || rect.h <= 0.0
            || self.culled(rect)
            || corners.iter().all(|color| color.a == 0)
        {
            return;
        }
        let [tl, tr, br, bl] = corners;
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [rect.x, rect.y],
                uv: [0.0, 0.0],
                color: tl.to_array(),
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [rect.x + rect.w, rect.y],
                uv: [1.0, 0.0],
                color: tr.to_array(),
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [rect.x + rect.w, rect.y + rect.h],
                uv: [1.0, 1.0],
                color: br.to_array(),
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [rect.x, rect.y + rect.h],
                uv: [0.0, 1.0],
                color: bl.to_array(),
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// Rounded rectangle filled with a four-corner gradient: the SDF masks
    /// the shape (mode 1) while the colour interpolates across the corners,
    /// so a rounded panel with a sheen is still one quad.
    pub fn round_rect_gradient(&mut self, rect: Rect, radius: f32, corners: [Color; 4]) {
        let pad = self.aa_pad();
        let quad = rect.expand(pad);
        if rect.w <= 0.0
            || rect.h <= 0.0
            || self.culled(quad)
            || corners.iter().all(|color| color.a == 0)
        {
            return;
        }
        let (hw, hh) = (rect.w * 0.5, rect.h * 0.5);
        // Padded corners carry the outer colours too: the gradient runs a
        // hair past the rect, which the one-pixel band hides.
        let (ew, eh) = (hw + pad, hh + pad);
        let r = radius.clamp(0.0, hw.min(hh));
        let params = [hw, hh, r, 1.0];
        let [tl, tr, br, bl] = corners;
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [quad.x, quad.y],
                uv: [-ew, -eh],
                color: tl.to_array(),
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y],
                uv: [ew, -eh],
                color: tr.to_array(),
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y + quad.h],
                uv: [ew, eh],
                color: br.to_array(),
                params,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [quad.x, quad.y + quad.h],
                uv: [-ew, eh],
                color: bl.to_array(),
                params,
                extra: QUAD_EXTRA,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// Radial gradient centred on `rect`: `from` fills the inner circle of
    /// `inner_radius` pixels and blends to `to` at `outer_radius` (clamped
    /// to fit the rect), with the outer edge anti-aliased over a pixel.
    /// The "to" colour is packed into the quad's UV channel, so this stays
    /// a single quad on the shared vertex format.
    pub fn radial_gradient(
        &mut self,
        rect: Rect,
        inner_radius: f32,
        outer_radius: f32,
        from: Color,
        to: Color,
    ) {
        let pad = self.aa_pad();
        let quad = rect.expand(pad);
        if (from.a == 0 && to.a == 0) || rect.w <= 0.0 || rect.h <= 0.0 || self.culled(quad) {
            return;
        }
        let packed = f32::from_bits(u32::from_le_bytes(to.to_array()));
        let outer = outer_radius.clamp(0.0, (rect.w * 0.5).min(rect.h * 0.5));
        let inner = inner_radius.clamp(0.0, outer);
        // As for mode 2: `params` keep the unpadded size and `extra` steps
        // the shader from the padded quad origin back to the rect's origin.
        let params = [rect.w, rect.h, outer, -2.0];
        let extra = [pad, pad];
        let uv = [packed, inner];
        let c = from.to_array();
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [quad.x, quad.y],
                uv,
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y],
                uv,
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [quad.x + quad.w, quad.y + quad.h],
                uv,
                color: c,
                params,
                extra,
            },
            UiVertex {
                pos: [quad.x, quad.y + quad.h],
                uv,
                color: c,
                params,
                extra,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// An atlas region (icon, packed sprite) at `rect`.
    pub fn atlas_region(&mut self, rect: Rect, cache: &GlyphCache, region: Region, tint: Color) {
        let uv = region.uv(cache.atlas().page_size());
        self.quad(rect, Some(uv), tint, CmdTexture::AtlasPage(region.page));
    }

    /// One glyph from a laid-out text run.
    pub fn positioned_glyph(
        &mut self,
        origin: (f32, f32),
        glyph: PositionedGlyph,
        color: Color,
        cache: &GlyphCache,
    ) {
        if glyph.glyph.width <= 0.0 || glyph.glyph.height <= 0.0 {
            return;
        }
        let uv = glyph.glyph.region.uv(cache.atlas().page_size());
        // A colour (sRGB) page carries real texel colours: tint it white
        // and keep only the caller's alpha, or emoji would draw text-tinted.
        let color = if cache.atlas().page_format(glyph.glyph.region.page)
            == super::atlas::AtlasFormat::RgbaSrgb
        {
            Color::rgba(255, 255, 255, color.a)
        } else {
            color
        };
        // Snap at the *physical* pixel: a half-logical-unit at scale 2 is
        // still a whole device pixel, and landing on one keeps the raster
        // crisp where a logical snap would blur it.
        let scale = self.scale;
        let rect = Rect::new(
            ((origin.0 + glyph.x) * scale).round() / scale,
            ((origin.1 + glyph.y) * scale).round() / scale,
            glyph.glyph.width,
            glyph.glyph.height,
        );
        self.quad(
            rect,
            Some(uv),
            color,
            CmdTexture::AtlasPage(glyph.glyph.region.page),
        );
    }

    /// A whole laid-out text run at `origin` (top-left of the layout box).
    pub fn text_layout(
        &mut self,
        origin: (f32, f32),
        layout: &TextLayout,
        color: Color,
        cache: &GlyphCache,
    ) {
        for &glyph in &layout.glyphs {
            self.positioned_glyph(origin, glyph, color, cache);
        }
    }

    /// Draw a layout produced at *physical* size, scaled back into logical
    /// space — the cached-layout path. Positions land exactly where the
    /// normalized layout would put them, but no glyph vector is copied or
    /// normalized per label.
    pub fn text_layout_scaled(
        &mut self,
        origin: (f32, f32),
        layout: &TextLayout,
        physical_scale: f32,
        color: Color,
        cache: &GlyphCache,
    ) {
        if (physical_scale - 1.0).abs() < 1e-6 {
            self.text_layout(origin, layout, color, cache);
            return;
        }
        let inv = 1.0 / physical_scale;
        for &glyph in &layout.glyphs {
            let scaled = PositionedGlyph {
                x: glyph.x * inv,
                y: glyph.y * inv,
                ch: glyph.ch,
                glyph: super::font::Glyph {
                    advance: glyph.glyph.advance * inv,
                    x_offset: glyph.glyph.x_offset * inv,
                    y_offset: glyph.glyph.y_offset * inv,
                    width: glyph.glyph.width * inv,
                    height: glyph.glyph.height * inv,
                    region: glyph.glyph.region,
                },
            };
            self.positioned_glyph(origin, scaled, color, cache);
        }
    }

    /// Convenience: lay out and draw one line of text. Returns the layout's
    /// width so callers can flow around it. For repeated strings, lay out
    /// once and call [`Self::text_layout`] instead.
    pub fn text(
        &mut self,
        fonts: &Fonts,
        cache: &mut GlyphCache,
        params: TextParams,
        origin: (f32, f32),
        text: &str,
    ) -> f32 {
        let layout = super::font::layout(
            fonts,
            cache,
            params.font,
            params.px,
            text,
            super::font::LayoutOptions::default(),
        );
        let width = layout.width;
        self.text_layout(origin, &layout, params.color, cache);
        width
    }

    /// Straight line as a rotated quad of `thickness` pixels.
    pub fn line(&mut self, from: (f32, f32), to: (f32, f32), thickness: f32, color: Color) {
        let half = thickness * 0.5;
        let bounds = Rect::new(
            from.0.min(to.0) - half,
            from.1.min(to.1) - half,
            (to.0 - from.0).abs() + thickness,
            (to.1 - from.1).abs() + thickness,
        );
        if color.a == 0 || self.culled(bounds) {
            return;
        }
        let dx = to.0 - from.0;
        let dy = to.1 - from.1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            return;
        }
        let nx = -dy / len * thickness * 0.5;
        let ny = dx / len * thickness * 0.5;
        let base = self.vertices.len() as u32;
        let c = color.to_array();
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [from.0 + nx, from.1 + ny],
                uv: [0.0, 0.0],
                color: c,
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [to.0 + nx, to.1 + ny],
                uv: [1.0, 0.0],
                color: c,
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [to.0 - nx, to.1 - ny],
                uv: [1.0, 1.0],
                color: c,
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
            UiVertex {
                pos: [from.0 - nx, from.1 - ny],
                uv: [0.0, 1.0],
                color: c,
                params: QUAD_PARAMS,
                extra: QUAD_EXTRA,
            },
        ]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.push_cmd(CmdTexture::White, 6);
    }

    /// Filled, anti-aliased circle — the same single SDF quad as
    /// [`Self::fill_circle`]. `segments` is ignored (the SDF edge is exact
    /// at any size) and kept only so existing callers still compile.
    pub fn circle(&mut self, center: (f32, f32), radius: f32, color: Color, segments: u32) {
        let _ = segments;
        self.fill_circle(center, radius, color);
    }

    /// Filled, anti-aliased circle of `radius` around `center`, as one SDF
    /// quad. Non-positive radii draw nothing.
    pub fn fill_circle(&mut self, center: (f32, f32), radius: f32, color: Color) {
        let (cx, cy) = center;
        self.round_rect(
            Rect::new(cx - radius, cy - radius, radius * 2.0, radius * 2.0),
            radius,
            color,
        );
    }

    /// Anti-aliased circle outline. `radius` is the outer edge; the band
    /// runs `thickness` pixels inward (a `thickness` at or beyond `radius`
    /// fills the disc). Non-positive thickness draws nothing. Built on
    /// [`Self::round_ring`], so it is one SDF quad.
    pub fn stroke_circle(&mut self, center: (f32, f32), radius: f32, thickness: f32, color: Color) {
        if thickness <= 0.0 {
            return;
        }
        let (cx, cy) = center;
        self.round_ring(
            Rect::new(cx - radius, cy - radius, radius * 2.0, radius * 2.0),
            radius,
            thickness,
            color,
        );
    }

    /// Filled pill: a rounded rectangle whose corner radius is half its
    /// shorter side, so a square becomes a circle and a wide rect becomes a
    /// stadium. True ellipses (unequal radii on both axes) are not
    /// supported — this always draws circular ends.
    pub fn pill(&mut self, rect: Rect, color: Color) {
        self.round_rect(rect, rect.w.min(rect.h) * 0.5, color);
    }

    /// Raw triangles — the escape hatch. Vertex positions are pixel space;
    /// `indices` are relative to `vertices`' first element.
    pub fn tris(&mut self, vertices: &[UiVertex], indices: &[u32], texture: CmdTexture) {
        if self.clipped_away() {
            return;
        }
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(vertices);
        self.indices
            .extend(indices.iter().map(|&index| base + index));
        self.push_cmd(texture, indices.len() as u32);
    }

    /// Rectangle with the four corners cut at 45 degrees (the HTML
    /// `clip-path` notch), in three quads: centre band plus side bands.
    pub fn notched(&mut self, rect: Rect, color: Color, notch: f32) {
        if self.clipped_away() {
            return;
        }
        let n = notch.min(rect.w * 0.5).min(rect.h * 0.5).max(0.0);
        if n <= 0.0 {
            self.rect(rect, color);
            return;
        }
        self.rect(
            Rect::new(rect.x + n, rect.y, rect.w - 2.0 * n, rect.h),
            color,
        );
        self.rect(Rect::new(rect.x, rect.y + n, n, rect.h - 2.0 * n), color);
        self.rect(
            Rect::new(rect.max().0 - n, rect.y + n, n, rect.h - 2.0 * n),
            color,
        );
    }

    /// Notched rectangle with a chunky pixel border: border colour notched
    /// frame, fill inset by `border`. `border` of 0 is a plain fill.
    pub fn notched_frame(
        &mut self,
        rect: Rect,
        fill: Color,
        border: Color,
        notch: f32,
        border_w: f32,
    ) {
        let rect = snap_rect(rect);
        self.notched(rect, border, notch);
        if border_w > 0.0 {
            let inner = Rect::new(
                rect.x + border_w,
                rect.y + border_w,
                rect.w - 2.0 * border_w,
                rect.h - 2.0 * border_w,
            );
            self.notched(inner, fill, (notch - border_w * 0.5).max(0.0));
        }
    }

    /// Hard offset shadow (HTML `box-shadow: 8px 8px 0`): a copy of the
    /// panel shape shifted down-right. Draw before the panel itself.
    pub fn hard_shadow(&mut self, rect: Rect, offset: f32, color: Color, notch: f32) {
        self.notched(rect.translate(offset, offset), color, notch);
    }

    /// Transparency checkerboard as four big quadrants (the HTML `#chk`
    /// stage backdrop): `a` top-left + bottom-right, `b` the rest.
    pub fn checker(&mut self, rect: Rect, a: Color, b: Color) {
        if self.clipped_away() {
            return;
        }
        let (cx, cy) = rect.center();
        self.rect(Rect::new(rect.x, rect.y, cx - rect.x, cy - rect.y), a);
        self.rect(Rect::new(cx, rect.y, rect.max().0 - cx, cy - rect.y), b);
        self.rect(Rect::new(rect.x, cy, cx - rect.x, rect.max().1 - cy), b);
        self.rect(Rect::new(cx, cy, rect.max().0 - cx, rect.max().1 - cy), a);
    }

    /// A field of `cell`-sized quads mixing `a`/`b` by ordered dither:
    /// cell (i, j) shows `b` when `amount >= bayer(i, j)`. One merged draw
    /// — the cheapest way to paint a pixel-art gradient or transition band.
    pub fn dither(&mut self, rect: Rect, cell: f32, a: Color, b: Color, amount: f32) {
        if self.clipped_away() || amount <= 0.0 {
            if self.clipped_away() {
                return;
            }
            self.rect(rect, a);
            return;
        }
        let cell = cell.max(1.0);
        let amount = amount.clamp(0.0, 1.0);
        let cols = (rect.w / cell).ceil().max(1.0) as i32;
        let rows = (rect.h / cell).ceil().max(1.0) as i32;
        for j in 0..rows {
            for i in 0..cols {
                let x = rect.x + i as f32 * cell;
                let y = rect.y + j as f32 * cell;
                let w = (rect.max().0 - x).min(cell);
                let h = (rect.max().1 - y).min(cell);
                if w <= 0.0 || h <= 0.0 {
                    continue;
                }
                let threshold = super::theme::bayer2(x, y);
                let color = if amount >= threshold { b } else { a };
                self.rect(Rect::new(x, y, w, h), color);
            }
        }
    }

    /// Append a command for the last `index_count` indices. `first_index` is
    /// an offset into the index buffer, so it is derived here rather than
    /// passed by callers (who naturally have vertex offsets at hand).
    fn push_cmd(&mut self, texture: CmdTexture, index_count: u32) {
        let first_index = self.indices.len() as u32 - index_count;
        let scissor = self.current_scissor();
        // Merge with the previous command when state matches: consecutive
        // quads of one texture become one draw.
        if let Some(last) = self.cmds.last_mut()
            && last.texture == texture
            && last.scissor == scissor
            && last.layer == self.layer
            && last.first_index + last.index_count == first_index
        {
            last.index_count += index_count;
            return;
        }
        self.cmds.push(DrawCmd {
            texture,
            scissor,
            first_index,
            index_count,
            layer: self.layer,
        });
    }

    /// Produce the frame for the current target `size` (logical units — the
    /// frame's own size comes out physical): commands sorted by layer
    /// (stable, so same-layer order is submission order). Vertices and
    /// indices stay absolute — sorting only permutes command order.
    ///
    /// When a scale is set, every vertex position, scissor, shape parameter
    /// and shape local coordinate is multiplied by it before the frame is
    /// emitted. Only texture UVs remain unscaled; cropped-image
    /// dimensions/radius (`params.w == 2`) are pixel-space and scale too,
    /// and so is the anti-aliasing pad in `extra` for modes `2` and `-2`
    /// (it is a length, the step from the quad origin to the shape).
    ///
    /// Fast path: when the commands were already submitted in layer order
    /// (the common case — layers are pushed deliberately), the buffers move
    /// across untouched. Sorting only costs when layers are interleaved.
    pub fn finish(self, size: (f32, f32)) -> UiFrame {
        let scale = self.scale;
        let mut vertices = self.vertices;
        let mut cmds = self.cmds;
        let mut size = size;
        if scale != 1.0 {
            for vertex in &mut vertices {
                vertex.pos[0] *= scale;
                vertex.pos[1] *= scale;
                // Text and plain quads only scale their positions.
                if vertex.params[3] == 0.0 {
                    continue;
                }
                // Shape params are pixel-space too: [half_w, half_h,
                // radius, is_shape] for flag 1 (plus SDF local coords in
                // uv), [full_w, full_h, radius, 2] for the rounded-
                // texture crop, and [half_w, half_h, radius, 3 + thickness]
                // for a ring, whose thickness is itself a length. Only
                // texture UVs remain unscaled; cropped-image
                // dimensions/radius are pixel-space and scale too.
                if vertex.params[3] == 1.0 {
                    vertex.params[0] *= scale;
                    vertex.params[1] *= scale;
                    vertex.params[2] *= scale;
                    vertex.uv[0] *= scale;
                    vertex.uv[1] *= scale;
                } else if vertex.params[3] == 2.0 {
                    vertex.params[0] *= scale;
                    vertex.params[1] *= scale;
                    vertex.params[2] *= scale;
                    // The pad from the quad origin to the shape: a length.
                    vertex.extra[0] *= scale;
                    vertex.extra[1] *= scale;
                } else if vertex.params[3] > 2.5 {
                    vertex.params[0] *= scale;
                    vertex.params[1] *= scale;
                    vertex.params[2] *= scale;
                    vertex.params[3] = 3.0 + (vertex.params[3] - 3.0) * scale;
                    vertex.uv[0] *= scale;
                    vertex.uv[1] *= scale;
                } else if vertex.params[3] == -1.0 {
                    // Rounded-clip textured quad: the clip's centre, radius
                    // and half extent are all pixel-space lengths.
                    vertex.params[0] *= scale;
                    vertex.params[1] *= scale;
                    vertex.params[2] *= scale;
                    vertex.extra[0] *= scale;
                    vertex.extra[1] *= scale;
                } else if vertex.params[3] == -2.0 {
                    // Radial gradient: full size and both radii are lengths
                    // (the inner rides in uv[1]); uv[0] holds the packed
                    // "to" colour and must not be scaled.
                    vertex.params[0] *= scale;
                    vertex.params[1] *= scale;
                    vertex.params[2] *= scale;
                    vertex.uv[1] *= scale;
                    // Origin pad (see mode 2): a length.
                    vertex.extra[0] *= scale;
                    vertex.extra[1] *= scale;
                } else if vertex.params[3] == ARC_MODE {
                    // Arc stroke: the mid radius and half thickness are
                    // lengths and uv is the centre-relative position. The
                    // start angle (params.z) and the sweep (extra.x) are
                    // angles — scaling them would bend the arc.
                    vertex.params[0] *= scale;
                    vertex.params[1] *= scale;
                    vertex.uv[0] *= scale;
                    vertex.uv[1] *= scale;
                } else if vertex.params[3] < -0.5 {
                    // Soft shadow: half extent, radius and the fade band
                    // (packed into the mode as -(5 + softness)) are lengths.
                    vertex.params[0] *= scale;
                    vertex.params[1] *= scale;
                    vertex.params[2] *= scale;
                    vertex.params[3] = -(5.0 + (-vertex.params[3] - 5.0) * scale);
                    vertex.uv[0] *= scale;
                    vertex.uv[1] *= scale;
                }
            }
            for cmd in &mut cmds {
                if let Some(rect) = &mut cmd.scissor {
                    rect.x *= scale;
                    rect.y *= scale;
                    rect.w *= scale;
                    rect.h *= scale;
                }
            }
            size = (size.0 * scale, size.1 * scale);
        }
        let sorted = cmds.windows(2).any(|pair| pair[0].layer > pair[1].layer);
        if !sorted {
            return UiFrame {
                vertices,
                indices: self.indices,
                cmds,
                atlas_updates: Vec::new(),
                atlas_snapshot: None,
                size,
                dirty_rect: None,
            };
        }
        // Stable sort by layer: later layers draw on top. Indices and
        // vertices are absolute, so reordering the commands alone is enough —
        // no re-indexing pass (the renderer draws every command with a zero
        // vertex offset).
        cmds.sort_by_key(|cmd| cmd.layer);
        UiFrame {
            vertices,
            indices: self.indices,
            cmds,
            atlas_updates: Vec::new(),
            atlas_snapshot: None,
            size,
            dirty_rect: None,
        }
    }
}

/// Snap to whole pixels (pixel art must land on texel boundaries).
fn snap_rect(rect: Rect) -> Rect {
    Rect::new(
        rect.x.round(),
        rect.y.round(),
        rect.w.round(),
        rect.h.round(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_offscreen_quads_still_emit() {
        let mut d = DrawList::new();
        d.begin((240.0, 180.0));
        d.set_scale(2.0);
        // Mostly-offscreen rect: still emits (intersect is partial).
        d.rect(Rect::new(230.0, 10.0, 32.0, 16.0), Color::WHITE);
        let frame = d.finish((240.0, 180.0));
        assert_eq!(frame.cmds.len(), 1, "{:?}", frame.cmds);
        // Fully-offscreen: culled.
        let mut d = DrawList::new();
        d.begin((240.0, 180.0));
        d.rect(Rect::new(240.0, 10.0, 32.0, 16.0), Color::WHITE);
        assert!(d.finish((240.0, 180.0)).cmds.is_empty());
    }

    #[test]
    fn rings_scale_their_thickness_with_the_frame() {
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.set_scale(2.0);
        // 20x10 at radius 4 with a 1.5 px band: mode 3 + thickness.
        d.round_ring(Rect::new(10.0, 10.0, 20.0, 10.0), 4.0, 1.5, Color::WHITE);
        let frame = d.finish((100.0, 100.0));
        let v = frame.vertices[0];
        // The quad is padded by one physical pixel: 10 - 0.5 logical -> 19.
        assert_eq!(v.pos, [19.0, 19.0]);
        assert_eq!(v.uv, [-21.0, -11.0], "padded SDF local coords are physical");
        // half_w, half_h, radius and the band are all physical: 1.5 -> 3.0.
        assert_eq!(v.params, [20.0, 10.0, 8.0, 6.0]);
    }

    #[test]
    fn rounded_frame_fill_sits_inside_the_border() {
        let white = Color::WHITE;
        let black = Color::BLACK;
        // 40x20 at radius 6 with a 2 px border: the fill is inset by 1 px
        // and its radius drops by 1, so it sits under the ring's middle.
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.rounded_frame(Rect::new(10.0, 10.0, 40.0, 20.0), 6.0, 2.0, white, black);
        let frame = d.finish((100.0, 100.0));
        // Fill first (mode 1), then the ring (mode 3 + thickness).
        assert_eq!(frame.vertices[0].params, [19.0, 9.0, 5.0, 1.0]);
        assert_eq!(frame.vertices[4].params[3], 5.0);
        assert_eq!(frame.vertices[4].params[..3], [20.0, 10.0, 6.0]);

        // A radius below the inset clamps the fill's radius at zero.
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.rounded_frame(Rect::new(10.0, 10.0, 40.0, 20.0), 0.5, 2.0, white, black);
        let frame = d.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params[2], 0.0);

        // No border (or a transparent stroke): the fill keeps the full rect.
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.rounded_frame(Rect::new(10.0, 10.0, 40.0, 20.0), 6.0, 0.0, white, black);
        d.rounded_frame(
            Rect::new(10.0, 40.0, 40.0, 20.0),
            6.0,
            2.0,
            white,
            Color::TRANSPARENT,
        );
        let frame = d.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params, [20.0, 10.0, 6.0, 1.0]);
        assert_eq!(frame.quad_count(), 2);

        // A translucent stroke tints over the fill, so the fill keeps the
        // full rect (the CSS border box).
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.rounded_frame(
            Rect::new(10.0, 10.0, 40.0, 20.0),
            6.0,
            2.0,
            white,
            Color::rgba(0, 0, 0, 40),
        );
        let frame = d.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params, [20.0, 10.0, 6.0, 1.0]);
    }

    #[test]
    fn square_round_rect_under_rounded_clip_is_carved() {
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.push_clip_rounded(Rect::new(0.0, 0.0, 50.0, 50.0), 8.0);
        d.round_rect(Rect::new(5.0, 5.0, 20.0, 20.0), 0.0, Color::WHITE);
        let frame = d.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params[3], -1.0);
        assert_eq!(frame.vertices.len(), 4);

        // A rounded shape keeps its own SDF geometry under the same clip.
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.push_clip_rounded(Rect::new(0.0, 0.0, 50.0, 50.0), 8.0);
        d.round_rect(Rect::new(5.0, 5.0, 20.0, 20.0), 4.0, Color::WHITE);
        let frame = d.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params[3], 1.0);

        // And a square shape with no rounded clip stays on the SDF path.
        let mut d = DrawList::new();
        d.begin((100.0, 100.0));
        d.push_clip(Rect::new(0.0, 0.0, 50.0, 50.0));
        d.round_rect(Rect::new(5.0, 5.0, 20.0, 20.0), 0.0, Color::WHITE);
        let frame = d.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params[3], 1.0);
    }

    fn rect() -> Rect {
        Rect::new(0.0, 0.0, 10.0, 10.0)
    }

    #[test]
    fn quads_merge_into_one_command() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.rect(rect(), Color::WHITE);
        list.rect(rect().translate(20.0, 0.0), Color::BLACK);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.cmds.len(), 1, "same texture + scissor merges");
        assert_eq!(frame.quad_count(), 2);
        assert_eq!(frame.cmds[0].index_count, 12);
    }

    #[test]
    fn layers_sort_stably_on_top() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.rect(rect(), Color::WHITE); // layer 0
        list.set_layer(5);
        list.rect(rect(), Color::BLACK);
        list.set_layer(0);
        list.rect(rect(), Color::WHITE); // layer 0 again
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.cmds.len(), 3);
        assert_eq!(frame.cmds[0].layer, 0);
        assert_eq!(frame.cmds[1].layer, 0);
        assert_eq!(frame.cmds[2].layer, 5, "layer 5 draws last (on top)");
    }

    #[test]
    fn texture_change_splits_commands() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.rect(rect(), Color::WHITE);
        list.quad(rect(), None, Color::WHITE, CmdTexture::AtlasPage(1));
        list.rect(rect(), Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.cmds.len(), 3);
    }

    #[test]
    fn clip_stack_intersects() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.push_clip(Rect::new(0.0, 0.0, 50.0, 50.0));
        list.push_clip(Rect::new(25.0, 25.0, 100.0, 100.0));
        list.rect(Rect::new(25.0, 25.0, 10.0, 10.0), Color::WHITE);
        list.pop_clip();
        list.pop_clip();
        let frame = list.finish((100.0, 100.0));
        assert_eq!(
            frame.cmds[0].scissor,
            Some(Rect::new(25.0, 25.0, 25.0, 25.0)),
            "nested clips intersect"
        );
    }

    #[test]
    fn empty_clip_discards_geometry() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.push_clip(Rect::new(0.0, 0.0, 10.0, 10.0));
        list.push_clip(Rect::new(50.0, 50.0, 10.0, 10.0));
        list.rect(rect(), Color::WHITE);
        assert_eq!(list.cmds.len(), 0, "fully clipped quads do not draw");
    }

    #[test]
    fn line_and_circle_generate_geometry() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.line((0.0, 0.0), (50.0, 50.0), 2.0, Color::WHITE);
        list.circle((50.0, 50.0), 10.0, Color::WHITE, 16);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.cmds.len(), 1, "same texture merges into one draw");
        assert_eq!(
            frame.indices.len() / 3,
            2 + 2,
            "line quad + circle SDF quad"
        );
    }

    #[test]
    fn sdf_quads_are_padded_for_antialiasing() {
        for scale in [1.0_f32, 2.0] {
            let mut list = DrawList::new();
            list.begin((100.0, 100.0));
            list.set_scale(scale);
            list.round_rect(Rect::new(10.0, 10.0, 20.0, 20.0), 10.0, Color::WHITE);
            let frame = list.finish((100.0, 100.0));
            let v = &frame.vertices;
            // One physical pixel past every edge, in physical units.
            let pad = 1.0;
            let (lo, hi) = (10.0 * scale - pad, 30.0 * scale + pad);
            assert_eq!(v[0].pos, [lo, lo], "scale {scale}: top-left");
            assert_eq!(v[2].pos, [hi, hi], "scale {scale}: bottom-right");
            // uv is centre-relative over the padded corners ...
            let half = 10.0 * scale + pad;
            assert_eq!(v[0].uv, [-half, -half], "scale {scale}");
            assert_eq!(v[2].uv, [half, half], "scale {scale}");
            // ... while params keep the unpadded half size and radius.
            let shape = 10.0 * scale;
            assert_eq!(v[0].params, [shape, shape, shape, 1.0], "scale {scale}");
        }
    }

    #[test]
    fn rounded_image_pad_rides_in_extra() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.set_scale(2.0);
        // Textures can't be built without a GPU, so test the shared helper
        // with a stand-in texture: the geometry is identical.
        list.push_shape_image(
            Rect::new(10.0, 10.0, 20.0, 10.0),
            CmdTexture::White,
            Color::WHITE,
            4.0,
            (0.0, 0.0, 1.0, 0.5),
        );
        let frame = list.finish((100.0, 100.0));
        let v = &frame.vertices;
        // Physical rect is 20..60 x 20..40; the quad adds one physical pixel.
        assert_eq!(v[0].pos, [19.0, 19.0]);
        assert_eq!(v[2].pos, [61.0, 41.0]);
        // Params keep the unpadded full size and radius (all physical).
        assert_eq!(v[0].params, [40.0, 20.0, 8.0, 2.0]);
        // The pad is a length, so finish scales it to one physical pixel,
        // and the quad origin plus pad lands back on the shape's origin.
        assert_eq!(v[0].extra, [1.0, 1.0]);
        assert_eq!(v[0].pos[0] + v[0].extra[0], 20.0);
        assert_eq!(v[0].pos[1] + v[0].extra[1], 20.0);
        // UVs extend over the pad (0.5 logical on a 20x10 rect): the
        // padded corners sample just outside the 0..1 x 0..0.5 range.
        let eps = 1e-6;
        let du = 0.5 / 20.0;
        let dv = 0.5 * 0.5 / 10.0;
        assert!((v[0].uv[0] + du).abs() < eps && (v[0].uv[1] + dv).abs() < eps);
        assert!((v[2].uv[0] - (1.0 + du)).abs() < eps && (v[2].uv[1] - (0.5 + dv)).abs() < eps);
    }

    #[test]
    fn backdrop_rounded_uses_backdrop_texture_and_screen_uv() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.set_scale(2.0);
        // A 20x10 glass panel at (10, 20) in a 100x100 logical frame, radius 4.
        let rect = Rect::new(10.0, 20.0, 20.0, 10.0);
        list.backdrop_rounded(rect, Color::WHITE, 4.0);
        let frame = list.finish((100.0, 100.0));
        // One command, sampling the captured backdrop rather than an atlas or game texture.
        assert_eq!(frame.cmds.len(), 1, "{:?}", frame.cmds);
        assert_eq!(frame.cmds[0].texture, CmdTexture::Backdrop);
        let v = &frame.vertices;
        // Padded by one physical pixel, so the quad starts at (19, 39) physical.
        assert_eq!(v[0].pos, [19.0, 39.0]);
        assert_eq!(v[2].pos, [61.0, 61.0]);
        // Params keep the unpadded physical size and radius, mode 2 (rounded image).
        assert_eq!(v[0].params, [40.0, 20.0, 8.0, 2.0]);
        assert_eq!(v[0].extra, [1.0, 1.0]);
        // The backdrop spans the whole logical frame, so each corner's UV is its logical position
        // over the frame size. The padded corners extrapolate past the rect by the pad (0.5 logical).
        let eps = 1e-6;
        let pad = 0.5;
        let uv = |x: f32, y: f32| ((x / 100.0), (y / 100.0));
        let (x0, y0) = uv(rect.x - pad, rect.y - pad);
        let (x1, y1) = uv(rect.x + rect.w + pad, rect.y + rect.h + pad);
        assert!(
            (v[0].uv[0] - x0).abs() < eps && (v[0].uv[1] - y0).abs() < eps,
            "{:?}",
            v[0].uv
        );
        assert!(
            (v[2].uv[0] - x1).abs() < eps && (v[2].uv[1] - y1).abs() < eps,
            "{:?}",
            v[2].uv
        );
        // Unscaled UVs: the scale changes positions and params, not texture coordinates.
        assert!((v[0].uv[0] - 0.095).abs() < eps);
    }

    #[test]
    fn circle_uses_sdf_quad() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        // `segments` is ignored: the same SDF quad whatever it is.
        list.circle((50.0, 50.0), 10.0, Color::WHITE, 3);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.vertices.len(), 4);
        assert_eq!(frame.indices.len(), 6);
        // Radius 10 on a 20-square: half size 10, radius 10, mode 1.
        assert_eq!(frame.vertices[0].params, [10.0, 10.0, 10.0, 1.0]);
    }

    #[test]
    fn circle_api_variants_use_sdf_modes() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.fill_circle((20.0, 20.0), 5.0, Color::WHITE);
        list.stroke_circle((60.0, 20.0), 8.0, 2.0, Color::WHITE);
        list.stroke_circle((20.0, 60.0), 8.0, 0.0, Color::WHITE); // nothing
        list.pill(Rect::new(40.0, 50.0, 30.0, 10.0), Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.vertices.len(), 12, "three shapes, four corners each");
        // Filled circle: mode 1, radius = half side.
        assert_eq!(frame.vertices[0].params, [5.0, 5.0, 5.0, 1.0]);
        // Outline: mode 3 + thickness, outer radius 8, band 2 px inward.
        assert_eq!(frame.vertices[4].params, [8.0, 8.0, 8.0, 5.0]);
        // Pill: radius is half the shorter side (5 for a 30x10 rect).
        assert_eq!(frame.vertices[8].params, [15.0, 5.0, 5.0, 1.0]);
    }

    #[test]
    fn finish_reindexes_without_dropping_vertices() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.rect(rect(), Color::WHITE);
        list.set_layer(9);
        list.rect(rect(), Color::BLACK);
        let frame = list.finish((100.0, 100.0));
        // Every index must resolve to a vertex.
        for &index in &frame.indices {
            assert!((index as usize) < frame.vertices.len());
        }
        // Two quads, four vertices each; nothing is compacted away.
        assert_eq!(frame.vertices.len(), 8);
    }

    /// The sorted path used to compact the buffers and write indices local
    /// to each command — but the renderer always draws with vertex_offset 0,
    /// so only the first command landed on its own geometry. Commands now
    /// keep their absolute indices, so every draw must reference the exact
    /// vertices it submitted.
    #[test]
    fn sorted_commands_keep_their_own_vertices() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        let white_rect = Rect::new(1.0, 2.0, 10.0, 10.0);
        let black_rect = Rect::new(50.0, 60.0, 20.0, 20.0);
        let grey_rect = Rect::new(30.0, 30.0, 5.0, 5.0);
        list.rect(white_rect, Color::WHITE); // layer 0
        list.set_layer(9);
        list.rect(black_rect, Color::BLACK); // layer 9
        // A triangle fan on the same layer exercises non-quad indices.
        list.tris(
            &[
                UiVertex {
                    pos: [70.0, 70.0],
                    uv: [0.0; 2],
                    color: Color::BLACK.to_array(),
                    params: QUAD_PARAMS,
                    extra: QUAD_EXTRA,
                },
                UiVertex {
                    pos: [80.0, 70.0],
                    uv: [0.0; 2],
                    color: Color::BLACK.to_array(),
                    params: QUAD_PARAMS,
                    extra: QUAD_EXTRA,
                },
                UiVertex {
                    pos: [75.0, 80.0],
                    uv: [0.0; 2],
                    color: Color::BLACK.to_array(),
                    params: QUAD_PARAMS,
                    extra: QUAD_EXTRA,
                },
            ],
            &[0, 1, 2],
            CmdTexture::AtlasPage(0),
        );
        list.set_layer(0);
        list.rect(grey_rect, Color::rgb(0x80, 0x80, 0x80)); // layer 0 again
        let frame = list.finish((100.0, 100.0));

        let quad_at = |frame: &UiFrame, cmd: &DrawCmd| -> Rect {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for &index in &frame.indices
                [cmd.first_index as usize..(cmd.first_index + cmd.index_count) as usize]
            {
                let v = &frame.vertices[index as usize];
                xs.push(v.pos[0]);
                ys.push(v.pos[1]);
            }
            let x = xs.iter().copied().fold(f32::MAX, f32::min);
            let y = ys.iter().copied().fold(f32::MAX, f32::min);
            let mx = xs.iter().copied().fold(f32::MIN, f32::max);
            let my = ys.iter().copied().fold(f32::MIN, f32::max);
            Rect::from_min_max(x, y, mx, my)
        };

        assert_eq!(frame.cmds.len(), 4);
        // Layer 0 commands come first in submission order, then layer 9.
        assert_eq!(quad_at(&frame, &frame.cmds[0]), white_rect);
        assert_eq!(quad_at(&frame, &frame.cmds[1]), grey_rect);
        assert_eq!(quad_at(&frame, &frame.cmds[2]), black_rect);
        let fan = quad_at(&frame, &frame.cmds[3]);
        assert_eq!(fan, Rect::new(70.0, 70.0, 10.0, 10.0), "tri fan intact");
        // And every index stays inside the vertex buffer.
        for &index in &frame.indices {
            assert!((index as usize) < frame.vertices.len());
        }
    }

    #[test]
    fn begin_clears_between_frames() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.rect(rect(), Color::WHITE);
        list.begin((100.0, 100.0));
        list.rect(rect(), Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.quad_count(), 1);
    }

    // --- Shadows, gradients, rounded clips ---------------------------------

    /// The shadow's quad spans the silhouette grown by half the fade band
    /// plus a pixel; its params carry the silhouette, the grown radius and
    /// the softness packed into the mode.
    #[test]
    fn shadow_pads_the_quad_and_packs_softness() {
        let mut list = DrawList::new();
        list.begin((400.0, 400.0));
        list.shadow(
            Rect::new(50.0, 50.0, 100.0, 60.0),
            6.0,
            (2.0, 3.0),
            4.0,
            8.0,
            Color::rgba(0, 0, 0, 100),
        );
        let frame = list.finish((400.0, 400.0));
        assert_eq!(frame.quad_count(), 1);
        let v = frame.vertices[0];
        // Silhouette: rect + spread 4 -> 108x68 at (48, 47)+offset(2,3).
        assert_eq!(v.params[0], 54.0, "half width of the silhouette");
        assert_eq!(v.params[1], 34.0, "half height of the silhouette");
        assert_eq!(v.params[2], 10.0, "radius grows by the spread");
        assert_eq!(v.params[3], -13.0, "softness 8 packs as -(5+8)");
        // Quad = silhouette (48, 49, 108, 68) padded by softness/2 + 1 = 5.
        assert_eq!(
            Rect::new(v.pos[0], v.pos[1], 108.0 + 10.0, 68.0 + 10.0),
            Rect::new(43.0, 44.0, 118.0, 78.0)
        );
    }

    #[test]
    fn shadow_hard_edge_is_one_pixel_aa() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.shadow(
            Rect::new(10.0, 10.0, 20.0, 20.0),
            0.0,
            (0.0, 0.0),
            0.0,
            0.0,
            Color::WHITE,
        );
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params[3], -5.0);
        // Zero softness still pads by one pixel for the AA band.
        assert_eq!(frame.vertices[0].pos, [9.0, 9.0]);
    }

    #[test]
    fn gradient_axes_map_corners() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        let from = Color::rgb(1, 2, 3);
        let to = Color::rgb(4, 5, 6);
        list.gradient(rect(), from, to, GradientAxis::TopToBottom);
        list.gradient(
            rect().translate(20.0, 0.0),
            from,
            to,
            GradientAxis::LeftToRight,
        );
        let frame = list.finish((100.0, 100.0));
        assert_eq!(
            frame.vertices[0].color,
            from.to_array(),
            "TL of a top-down ramp"
        );
        assert_eq!(
            frame.vertices[2].color,
            to.to_array(),
            "BR of a top-down ramp"
        );
        assert_eq!(
            frame.vertices[4].color,
            from.to_array(),
            "TL of a left-right ramp"
        );
        assert_eq!(
            frame.vertices[5].color,
            to.to_array(),
            "TR of a left-right ramp"
        );
        // Per-vertex colours: the whole quad still one command.
        assert_eq!(frame.cmds.len(), 1);
    }

    #[test]
    fn round_rect_gradient_keeps_shape_mode_with_mixed_colours() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.round_rect_gradient(
            rect(),
            3.0,
            [Color::WHITE, Color::WHITE, Color::BLACK, Color::BLACK],
        );
        let frame = list.finish((100.0, 100.0));
        let v = &frame.vertices;
        assert_eq!(v[0].params, [5.0, 5.0, 3.0, 1.0], "still a rounded fill");
        assert_eq!(v[0].color, Color::WHITE.to_array());
        assert_eq!(v[2].color, Color::BLACK.to_array());
    }

    #[test]
    fn radial_gradient_packs_the_to_colour() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        let to = Color::rgba(0x11, 0x22, 0x33, 0x44);
        list.radial_gradient(rect(), 2.0, 40.0, Color::WHITE, to);
        let frame = list.finish((100.0, 100.0));
        let v = frame.vertices[0];
        assert_eq!(
            v.params,
            [10.0, 10.0, 5.0, -2.0],
            "outer clamped to the rect"
        );
        assert_eq!(v.uv[1], 2.0, "inner radius rides in uv[1]");
        assert_eq!(
            f32::from_bits(v.uv[0].to_bits()).to_bits(),
            f32::from_bits(u32::from_le_bytes(to.to_array())).to_bits(),
            "uv[0] holds the packed to-colour bits"
        );
    }

    #[test]
    fn rounded_clip_marks_plain_quads_only() {
        let mut list = DrawList::new();
        list.begin((200.0, 200.0));
        list.push_clip_rounded(Rect::new(10.0, 10.0, 80.0, 60.0), 8.0);
        list.rect(Rect::new(12.0, 12.0, 4.0, 4.0), Color::WHITE); // mode -1
        list.round_rect(Rect::new(20.0, 20.0, 10.0, 10.0), 2.0, Color::WHITE); // mode 1
        list.pop_clip();
        let frame = list.finish((200.0, 200.0));
        assert_eq!(frame.vertices[0].params, [50.0, 40.0, 8.0, -1.0]);
        assert_eq!(frame.vertices[0].extra, [40.0, 30.0]);
        // The SDF fill keeps its own shape params (clipped by the square
        // scissor only).
        assert_eq!(frame.vertices[4].params, [5.0, 5.0, 2.0, 1.0]);
        assert_eq!(
            frame.cmds[0].scissor,
            Some(Rect::new(10.0, 10.0, 80.0, 60.0))
        );
    }

    #[test]
    fn square_clips_leave_quads_untouched() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.push_clip(Rect::new(0.0, 0.0, 50.0, 50.0));
        list.rect(rect(), Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(frame.vertices[0].params, QUAD_PARAMS);
        assert_eq!(frame.vertices[0].extra, QUAD_EXTRA);
    }

    #[test]
    fn finish_scales_the_new_modes() {
        let mut list = DrawList::new();
        list.begin((400.0, 400.0));
        list.set_scale(2.0);
        list.shadow(
            Rect::new(10.0, 10.0, 20.0, 20.0),
            0.0,
            (0.0, 0.0),
            0.0,
            8.0,
            Color::WHITE,
        );
        list.radial_gradient(rect(), 2.0, 4.0, Color::WHITE, Color::BLACK);
        list.push_clip_rounded(Rect::new(0.0, 0.0, 40.0, 40.0), 4.0);
        list.rect(Rect::new(1.0, 1.0, 2.0, 2.0), Color::WHITE);
        let frame = list.finish((200.0, 200.0));
        let (shadow, radial, clipped) =
            (&frame.vertices[0], &frame.vertices[4], &frame.vertices[8]);
        assert_eq!(shadow.params[0], 20.0, "shadow half extent scales");
        assert_eq!(shadow.params[3], -(5.0 + 16.0), "softness is physical");
        // Vertex 0 is the quad's top-left: uv is -half of the padded quad
        // (a 20px silhouette padded by softness/2 + 1 = 5 -> 30px half).
        assert_eq!(shadow.uv[1], -30.0, "uv offsets scale");
        assert_eq!(radial.params[2], 8.0, "outer radius scales");
        assert_eq!(
            radial.uv[1], 4.0,
            "inner radius scales; packed colour does not"
        );
        assert_eq!(clipped.params, [40.0, 40.0, 8.0, -1.0], "clip rect scales");
        assert_eq!(clipped.extra, [40.0, 40.0], "clip half extent scales");
    }

    #[test]
    fn ring_arc_carries_its_band_and_angles() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.ring_arc(rect(), 5.0, 2.0, -1.5, 3.0, Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        let v = frame.vertices[0];
        assert_eq!(v.params[0], 4.0, "mid radius is outer minus half thickness");
        assert_eq!(v.params[1], 1.0, "half thickness");
        assert_eq!(v.params[2], -1.5, "the start angle rides in params.z");
        assert_eq!(v.params[3], ARC_MODE);
        assert_eq!(v.extra, [3.0, 0.0], "the sweep rides in extra.x");
        // The quad is the ring's bounds inflated by a pixel for the
        // anti-aliased edge, so `uv` is the corner offset from the centre.
        assert_eq!(v.uv, [-6.0, -6.0]);
        assert_eq!(frame.vertices[2].uv, [6.0, 6.0]);
        assert_eq!(frame.vertices[0].params, frame.vertices[3].params);
    }

    #[test]
    fn finish_scales_arc_lengths_not_angles() {
        let mut list = DrawList::new();
        list.begin((400.0, 400.0));
        list.set_scale(2.0);
        list.ring_arc(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            10.0,
            4.0,
            -1.5,
            3.0,
            Color::WHITE,
        );
        let frame = list.finish((400.0, 400.0));
        let v = frame.vertices[0];
        assert_eq!(v.params[0], 16.0, "mid radius is a length and scales");
        assert_eq!(v.params[1], 4.0, "half thickness is a length and scales");
        assert_eq!(v.params[2], -1.5, "the start angle must not scale");
        assert_eq!(v.params[3], ARC_MODE, "the mode survives scaling");
        assert_eq!(v.extra, [3.0, 0.0], "the sweep must not scale");
        assert_eq!(v.uv, [-22.0, -22.0], "centre-relative offsets scale");
    }

    #[test]
    fn ring_arc_clamps_sweep_to_a_full_turn() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.ring_arc(rect(), 5.0, 2.0, 0.0, 9.0, Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        assert_eq!(
            frame.vertices[0].extra[0],
            std::f32::consts::TAU,
            "a sweep past TAU is a full ring"
        );
    }

    #[test]
    fn ring_arc_clamps_thickness_to_the_radius() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        // radius clamps to half the rect side (5); thickness past it runs
        // the band to the centre instead of dropping the arc.
        list.ring_arc(rect(), 5.0, 9.0, 0.0, 1.0, Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        let v = frame.vertices[0];
        assert_eq!(v.params[1], 2.5, "half thickness is half the radius");
        assert_eq!(v.params[0], 2.5, "mid radius reaches the centre");
    }

    #[test]
    fn ring_arc_rejects_non_finite_angles() {
        let mut list = DrawList::new();
        list.begin((100.0, 100.0));
        list.ring_arc(rect(), 5.0, 2.0, f32::NAN, 1.0, Color::WHITE);
        list.ring_arc(rect(), 5.0, 2.0, 0.0, f32::INFINITY, Color::WHITE);
        let frame = list.finish((100.0, 100.0));
        assert!(frame.vertices.is_empty(), "non-finite input emits nothing");
    }
}

#[cfg(test)]
mod splice_tests {
    use super::*;

    /// splice_from translates positions, rebases indices, and moves the
    /// mode −1 rounded-clip centre along.
    #[test]
    fn splice_from_translates_and_rebases() {
        let mut src = DrawList::new();
        src.begin((100.0, 100.0));
        src.rect(Rect::new(10.0, 10.0, 20.0, 20.0), Color::rgb(200, 0, 0));
        // A recorded second list splices it 5px right / 7 down.
        let mut dst = DrawList::new();
        dst.begin((100.0, 100.0));
        dst.rect(Rect::new(0.0, 0.0, 4.0, 4.0), Color::rgb(0, 0, 200));
        dst.splice_from(&src, (0, 0, 0), src.cursor(), (5.0, 7.0));
        // dst own quad (4 verts) then spliced (4).
        assert_eq!(dst.vertices.len(), 8);
        assert_eq!(dst.indices.len(), 12);
        // Spliced indices point at verts 4..8.
        assert_eq!(&dst.indices[6..9], &[4, 5, 6]);
        // Spliced rect's first vertex moved by (5,7).
        let v = &dst.vertices[4];
        assert!((v.pos[0] - 15.0).abs() < 1e-5 && (v.pos[1] - 17.0).abs() < 1e-5);
        let _ = src;
    }

    /// Splicing a middle range copies only that range: later geometry
    /// stays out, indices rebase relative to the range's first vertex,
    /// and a batch the range merged into is clamped to it.
    #[test]
    fn splice_from_middle_range_only() {
        let mut src = DrawList::new();
        src.begin((100.0, 100.0));
        src.rect(Rect::new(0.0, 0.0, 4.0, 4.0), Color::rgb(1, 0, 0));
        let start = src.cursor();
        src.rect(Rect::new(10.0, 10.0, 4.0, 4.0), Color::rgb(2, 0, 0));
        let end = src.cursor();
        src.rect(Rect::new(50.0, 50.0, 4.0, 4.0), Color::rgb(3, 0, 0));
        let mut dst = DrawList::new();
        dst.begin((100.0, 100.0));
        dst.splice_from(&src, start, end, (1.0, 0.0));
        assert_eq!(dst.vertices.len(), 4);
        assert_eq!(&dst.indices[..3], &[0, 1, 2]);
        assert!((dst.vertices[0].pos[0] - 11.0).abs() < 1e-5);
        assert_eq!(dst.cmds.len(), 1);
        assert_eq!(dst.cmds[0].first_index, 0);
        assert_eq!(dst.cmds[0].index_count, 6);
    }
    /// A recorded command with no scissor inherits the clip live at
    /// splice time (recorded content stays inside the parent's clip).
    #[test]
    fn splice_from_no_scissor_inherits_current_clip() {
        let mut src = DrawList::new();
        src.begin((400.0, 300.0));
        src.quad(
            Rect::new(0.0, 0.0, 50.0, 50.0),
            None,
            crate::ui::Color::WHITE,
            crate::ui::draw::CmdTexture::White,
        );
        let start = (0, 0, 0);
        let end = src.cursor();
        let mut dst = DrawList::new();
        dst.begin((400.0, 300.0));
        dst.push_clip(Rect::new(10.0, 10.0, 20.0, 20.0));
        dst.splice_from(&src, start, end, (0.0, 0.0));
        let last = dst.cmds.last().unwrap();
        assert_eq!(last.scissor, Some(Rect::new(10.0, 10.0, 20.0, 20.0)));
    }

    /// Recorded + current clips that don't overlap splice as Some(empty)
    /// — never widened to None.
    #[test]
    fn splice_from_disjoint_clips_stay_empty() {
        let mut src = DrawList::new();
        src.begin((400.0, 300.0));
        src.push_clip(Rect::new(0.0, 0.0, 10.0, 10.0));
        src.quad(
            Rect::new(0.0, 0.0, 5.0, 5.0),
            None,
            crate::ui::Color::WHITE,
            crate::ui::draw::CmdTexture::White,
        );
        let start = (0, 0, 0);
        let end = src.cursor();
        let mut dst = DrawList::new();
        dst.begin((400.0, 300.0));
        dst.push_clip(Rect::new(100.0, 100.0, 10.0, 10.0));
        dst.splice_from(&src, start, end, (0.0, 0.0));
        let last = dst.cmds.last().unwrap();
        assert_eq!(
            last.scissor,
            Some(Rect::default()),
            "disjoint clips must stay Some(empty)"
        );
    }

    /// `cull = false` (the view scratch) only bypasses the screen test —
    /// a fully empty clip still emits nothing.
    #[test]
    fn scratch_culling_keeps_empty_clip() {
        let mut dl = DrawList::new();
        dl.cull = false;
        dl.begin((400.0, 300.0));
        dl.push_clip(Rect::new(50.0, 50.0, 0.0, 0.0));
        dl.quad(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            None,
            crate::ui::Color::WHITE,
            crate::ui::draw::CmdTexture::White,
        );
        assert!(dl.vertices.is_empty());
        assert!(dl.indices.is_empty());
    }
}
