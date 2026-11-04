//! Compact UI geometry preparation — the instancing path.
//!
//! Scans each final sorted `DrawCmd` in paint order: commands whose six
//! indices form the standard `[a,b,c,a,c,d]` quad (four axis-aligned
//! vertices, uniform color/params, UV affine in position) convert to a
//! 60-byte `UiInstance` record. Everything else — arbitrary triangles,
//! rotated lines, per-vertex gradients — falls through to the legacy
//! indexed path unchanged. Mixed `quad, tri, quad` order is preserved:
//! adjacent compatible draws merge, a non-quad never reorders.

use crate::ui::draw::{CmdTexture, DrawCmd, UiVertex};

/// One converted quad, the instance vertex buffer's element.
/// `#[repr(C)]`, 60 bytes: rect[4] @0, uv[4] @16, color[4] @32 (bytes),
/// params[4] @36, extra[2] @52. Private ABI — no public GPU contract.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiInstance {
    /// `[x, y, w, h]` in pixels.
    pub rect: [f32; 4],
    /// `[u0, v0, u1, v1]` — the corner of the top-left position's UV and
    /// the bottom-right's (signs preserve flips).
    pub uv: [f32; 4],
    /// Straight-alpha sRGB tint, packed.
    pub color: [u8; 4],
    /// `[half_w_or_w, half_h_or_h, radius, mode]` — same semantics as the
    /// legacy `params` attribute.
    pub params: [f32; 4],
    /// Same semantics as the legacy `extra` attribute (the rounded clip's
    /// half extent for mode -1; zero otherwise).
    pub extra: [f32; 2],
}

const _: [(); 60] = [(); std::mem::size_of::<UiInstance>()];

/// How a prepared draw reaches the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawKind {
    /// `count` converted quads, `first` into the instance buffer.
    Instanced { first: u32, count: u32 },
    /// `count` raw indices, `first` into the remapped index buffer.
    Indexed { first: u32, count: u32 },
}

/// A prepared draw: the source command plus how it draws.
#[derive(Clone, Debug)]
pub struct PreparedDraw {
    /// Index into `frame.cmds` — texture/scissor lookup.
    pub cmd_index: usize,
    pub kind: DrawKind,
}

/// The whole prepared frame.
#[derive(Default)]
pub struct Prepared {
    pub draws: Vec<PreparedDraw>,
    /// Instance records for converted quads.
    pub instances: Vec<UiInstance>,
    /// Remapped vertices for the indexed fallback (converted quads are
    /// NOT copied — the upload shrinks by exactly the converted bytes).
    pub vertices: Vec<UiVertex>,
    pub indices: Vec<u32>,
    /// Raw-index → compacted-index table (`u32::MAX` = unmapped); reset
    /// lazily at the first indexed fallback, capacity retained.
    remap: Vec<u32>,
}

impl Prepared {
    fn clear(&mut self) {
        self.draws.clear();
        self.instances.clear();
        self.vertices.clear();
        self.indices.clear();
        self.remap.clear();
    }
}

/// `cmd` draws the standard `[a,b,c,a,c,d]` quad pattern at `quad`'s
/// 6-index slot; returns the four distinct vertex indices it uses.
fn quad_verts_at(frame_indices: &[u32], draw: &DrawCmd, quad: usize) -> Option<[u32; 4]> {
    let base = draw.first_index as usize + quad * 6;
    let i = frame_indices.get(base..base + 6)?;
    let (a, b, c, d) = (i[0], i[1], i[2], i[5]);
    // Standard winding: [a,b,c, a,c,d]. Any relabeling of the same quad
    // pattern is rejected — the four vertices must be exactly these.
    if i[3] != a || i[4] != c {
        return None;
    }
    if a == b || a == c || a == d || b == c || b == d || c == d {
        return None;
    }
    Some([a, b, c, d])
}

/// Convert one 6-index slot's four vertices to an instance when they
/// describe an axis-aligned rect with uniform color/params and affine
/// UV in position.
fn convert_quad(vertices: &[UiVertex], verts: [u32; 4]) -> Option<UiInstance> {
    let v = [
        *vertices.get(verts[0] as usize)?,
        *vertices.get(verts[1] as usize)?,
        *vertices.get(verts[2] as usize)?,
        *vertices.get(verts[3] as usize)?,
    ];
    // Uniform color + params on all four corners. Per-vertex colours
    // (gradients) deliberately fail here and stay on the indexed path.
    if v.iter()
        .any(|u| u.color != v[0].color || u.params != v[0].params || u.extra != v[0].extra)
    {
        return None;
    }
    // Non-finite geometry/UV/params must stay on the indexed path —
    // NaN would poison every comparison below into silent acceptance.
    if v.iter().any(|u| {
        !u.pos
            .iter()
            .chain(u.uv.iter())
            .chain(u.params.iter())
            .all(|x| x.is_finite())
    }) {
        return None;
    }
    // Strict geometry: a,b,c,d must be the TL,TR,BR,BL corners of an
    // axis-aligned rectangle, in that order. Trapezoids, bowties,
    // interior points and re-ordered windings raster differently — they
    // are NOT a rectangle and belong to the legacy draw.
    let eps = 1e-4;
    let same = |a: f32, b: f32| (a - b).abs() <= eps;
    let (tl, tr, br, bl) = (v[0].pos, v[1].pos, v[2].pos, v[3].pos);
    let (x, y) = (tl[0], tl[1]);
    if !(same(tr[1], y) && same(bl[0], x)) {
        return None;
    }
    let w = tr[0] - x;
    let h = bl[1] - y;
    if !(w > 0.0 && h > 0.0) || !same(br[0], x + w) || !same(br[1], y + h) {
        return None;
    }
    // UV must match the same corner orientation: TL/TL, TR/TR, BR/BR,
    // BL/BL. Flipped or rotated UVs change the sampled image — keep
    // them on the indexed path instead of re-orienting.
    let (u0, v0) = (v[0].uv[0], v[0].uv[1]);
    let (u1, v1) = (v[2].uv[0], v[2].uv[1]);
    if !(u1 > u0 && v1 > v0)
        || !same(v[1].uv[0], u1)
        || !same(v[1].uv[1], v0)
        || !same(v[3].uv[0], u0)
        || !same(v[3].uv[1], v1)
    {
        return None;
    }
    Some(UiInstance {
        rect: [x, y, w, h],
        uv: [u0, v0, u1, v1],
        color: v[0].color,
        params: v[0].params,
        extra: v[0].extra,
    })
}

/// Convert every 6-index quad inside `cmd` (a `push_cmd`-merged run can
/// carry many) — all-or-nothing: any non-quad sub-command means the
/// whole command falls back to the indexed path unchanged.
///
/// Appends into `instances` starting at a checkpoint; on failure the
/// tail is rolled back so a rejected command never leaves stray
/// instances or extra draws behind. Returns the appended count.
fn try_instances(
    vertices: &[UiVertex],
    frame_indices: &[u32],
    draw: &DrawCmd,
    instances: &mut Vec<UiInstance>,
) -> Option<u32> {
    if draw.index_count == 0 || !draw.index_count.is_multiple_of(6) {
        return None;
    }
    let checkpoint = instances.len();
    let quads = (draw.index_count / 6) as usize;
    for q in 0..quads {
        let ok =
            quad_verts_at(frame_indices, draw, q).and_then(|verts| convert_quad(vertices, verts));
        match ok {
            Some(inst) => instances.push(inst),
            None => {
                instances.truncate(checkpoint);
                return None;
            }
        }
    }
    Some(quads as u32)
}

/// Whether two adjacent draws share the merge-relevant state: the same
/// texture and scissor mean they can record as one call.
fn mergeable(a: &DrawCmd, b: &DrawCmd) -> bool {
    a.scissor == b.scissor && same_texture(&a.texture, &b.texture)
}

fn same_texture(a: &CmdTexture, b: &CmdTexture) -> bool {
    match (a, b) {
        (CmdTexture::White, CmdTexture::White) => true,
        (CmdTexture::AtlasPage(p), CmdTexture::AtlasPage(q)) => p == q,
        (CmdTexture::Custom(t), CmdTexture::Custom(u)) => std::sync::Arc::ptr_eq(t, u),
        _ => false,
    }
}

/// Prepare `frame` for the compact pipeline. `enabled == false` returns a
/// purely-indexed path identical to the old behavior.
#[cfg(test)]
pub fn prepare(
    cmds: &[DrawCmd],
    vertices: &[UiVertex],
    frame_indices: &[u32],
    enabled: bool,
) -> Prepared {
    let mut out = Prepared::default();
    prepare_into(&mut out, cmds, vertices, frame_indices, enabled);
    out
}

/// `prepare` into a caller-owned buffer — per-slot reuse keeps the
/// vectors' capacities across records.
pub(crate) fn prepare_into(
    out: &mut Prepared,
    cmds: &[DrawCmd],
    vertices: &[UiVertex],
    frame_indices: &[u32],
    enabled: bool,
) {
    out.clear();
    // Raw-index remap: converted quads' vertices are dropped; every used
    // index must point into the compacted vertex buffer.
    let mut remap_ready = false;
    let mut i = 0usize;
    while i < cmds.len() {
        let draw = &cmds[i];
        // Probe by appending: a rejected command rolls back its tail so
        // no allocation, instance or draw escapes.
        let first_inst = out.instances.len();
        let inst = enabled
            .then(|| try_instances(vertices, frame_indices, draw, &mut out.instances))
            .flatten();
        match inst {
            Some(mut count) => {
                let first = first_inst as u32;
                // Merge adjacent instanced draws sharing texture+scissor.
                let mut j = i + 1;
                while j < cmds.len()
                    && mergeable(draw, &cmds[j])
                    && let Some(next) =
                        try_instances(vertices, frame_indices, &cmds[j], &mut out.instances)
                {
                    count += next;
                    j += 1;
                }
                out.draws.push(PreparedDraw {
                    cmd_index: i,
                    kind: DrawKind::Instanced { first, count },
                });
                i = j;
            }
            None => {
                let first = out.indices.len() as u32;
                let mut count = 0u32;
                let mut j = i;
                // Merge adjacent indexed draws sharing texture+scissor —
                // never across an instanced draw.
                while j < cmds.len() && mergeable(draw, &cmds[j]) {
                    // Probe appends before it knows — ALWAYS roll the
                    // tail back, whether the next command converted
                    // (merge stop) or not (merge continues).
                    let checkpoint = out.instances.len();
                    let blocked = enabled
                        && try_instances(vertices, frame_indices, &cmds[j], &mut out.instances)
                            .is_some();
                    out.instances.truncate(checkpoint);
                    if blocked {
                        break;
                    }
                    if !remap_ready {
                        out.remap.clear();
                        out.remap.resize(vertices.len(), u32::MAX);
                        remap_ready = true;
                    }
                    let src = &cmds[j];
                    let base = src.first_index as usize;
                    for k in 0..src.index_count as usize {
                        let old = frame_indices[base + k] as usize;
                        let new = if out.remap[old] != u32::MAX {
                            out.remap[old]
                        } else {
                            out.vertices.push(vertices[old]);
                            out.remap[old] = (out.vertices.len() - 1) as u32;
                            out.remap[old]
                        };
                        out.indices.push(new);
                    }
                    count += src.index_count;
                    j += 1;
                }
                out.draws.push(PreparedDraw {
                    cmd_index: i,
                    kind: DrawKind::Indexed { first, count },
                });
                i = j;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::draw::DrawCmd;

    fn quad_cmd(texture_page: usize, first_index: u32) -> DrawCmd {
        DrawCmd {
            texture: CmdTexture::AtlasPage(texture_page),
            scissor: None,
            first_index,
            index_count: 6,
            layer: 0,
        }
    }

    fn quad_vertices(x: f32, y: f32, w: f32, h: f32) -> [UiVertex; 4] {
        let c = [200u8, 100, 50, 255];
        [
            UiVertex {
                pos: [x, y],
                uv: [0.0, 0.0],
                color: c,
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [x + w, y],
                uv: [1.0, 0.0],
                color: c,
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [x + w, y + h],
                uv: [1.0, 1.0],
                color: c,
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [x, y + h],
                uv: [0.0, 1.0],
                color: c,
                params: [0.0; 4],
                extra: [0.0; 2],
            },
        ]
    }

    #[test]
    fn a_standard_quad_becomes_one_60byte_instance() {
        assert_eq!(std::mem::size_of::<UiInstance>(), 60);
        let mut vertices = Vec::new();
        vertices.extend_from_slice(&quad_vertices(10.0, 20.0, 30.0, 40.0));
        let indices = vec![0u32, 1, 2, 0, 2, 3];
        let cmds = vec![quad_cmd(0, 0)];
        let prepared = prepare(&cmds, &vertices, &indices, true);
        assert_eq!(prepared.instances.len(), 1);
        assert!(
            prepared.vertices.is_empty(),
            "converted verts not re-uploaded"
        );
        assert_eq!(prepared.instances[0].rect, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(prepared.instances[0].uv, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn mixed_order_and_merging() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        // quad(0), tri, quad(1), quad(1): quad, raw, merged-instance pair.
        vertices.extend_from_slice(&quad_vertices(0.0, 0.0, 10.0, 10.0));
        indices.extend_from_slice(&[0u32, 1, 2, 0, 2, 3]);
        let tri_base = vertices.len() as u32;
        for i in 0..3 {
            vertices.push(UiVertex {
                pos: [i as f32 * 3.0, i as f32],
                uv: [0.0, 0.0],
                color: [i as u8; 4],
                params: [0.0; 4],
                extra: [0.0; 2],
            });
        }
        indices.extend_from_slice(&[tri_base, tri_base + 1, tri_base + 2]);
        vertices.extend_from_slice(&quad_vertices(50.0, 0.0, 10.0, 10.0));
        indices.extend_from_slice(&[7u32, 8, 9, 7, 9, 10]);
        vertices.extend_from_slice(&quad_vertices(70.0, 0.0, 10.0, 10.0));
        indices.extend_from_slice(&[11u32, 12, 13, 11, 13, 14]);
        let cmds = vec![
            quad_cmd(0, 0),
            DrawCmd {
                texture: CmdTexture::AtlasPage(9),
                scissor: None,
                first_index: 6,
                index_count: 3,
                layer: 0,
            },
            quad_cmd(1, 9),
            quad_cmd(1, 15),
        ];
        let prepared = prepare(&cmds, &vertices, &indices, true);
        assert_eq!(prepared.draws.len(), 3);
        assert_eq!(prepared.instances.len(), 3);
        assert!(matches!(
            prepared.draws[0].kind,
            DrawKind::Instanced { count: 1, .. }
        ));
        assert!(matches!(
            prepared.draws[1].kind,
            DrawKind::Indexed { count: 3, .. }
        ));
        assert!(matches!(
            prepared.draws[2].kind,
            DrawKind::Instanced { count: 2, .. }
        ));
        // The raw tri's vertices were remapped into the compact buffer.
        assert_eq!(prepared.vertices.len(), 3);
        assert_eq!(prepared.indices, vec![0u32, 1, 2]);
    }

    fn verts4(pos: [[f32; 2]; 4], uv: [[f32; 2]; 4]) -> Vec<UiVertex> {
        let c = [10u8, 20, 30, 255];
        pos.iter()
            .zip(uv.iter())
            .map(|(&pos, &uv)| UiVertex {
                pos,
                uv,
                color: c,
                params: [0.0; 4],
                extra: [0.0; 2],
            })
            .collect()
    }

    #[test]
    fn trapezoid_and_bowtie_stay_indexed() {
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        // Trapezoid: BR pulled in — still four distinct affine points.
        let trap = verts4([[0.0, 0.0], [20.0, 0.0], [10.0, 10.0], [0.0, 10.0]], uv);
        let cmds = vec![quad_cmd(0, 0)];
        let indices = vec![0u32, 1, 2, 0, 2, 3];
        let prepared = prepare(&cmds, &trap, &indices, true);
        assert!(prepared.instances.is_empty());
        assert_eq!(prepared.indices, indices);
        // Bowtie: TL,BR,TR,BL — wrong winding, not a rectangle.
        let bow = verts4([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]], uv);
        let prepared = prepare(&cmds, &bow, &indices, true);
        assert!(prepared.instances.is_empty());
        // Flipped UV on a correct rect: stays indexed (don't re-orient).
        let flip_uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let flip = verts4(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            flip_uv,
        );
        let prepared = prepare(&cmds, &flip, &indices, true);
        assert!(prepared.instances.is_empty());
        // NaN position never converts.
        let nan = verts4(
            [[f32::NAN, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            uv,
        );
        let prepared = prepare(&cmds, &nan, &indices, true);
        assert!(prepared.instances.is_empty());
        // A valid quad followed by a trapezoid in the SAME DrawCmd rolls
        // back — the whole command falls back, no stray instance/draw.
        let mut mixed = quad_vertices(0.0, 0.0, 8.0, 8.0).to_vec();
        mixed.extend_from_slice(&[
            UiVertex {
                pos: [0.0, 0.0],
                uv: [0.0, 0.0],
                color: [10, 20, 30, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [20.0, 0.0],
                uv: [1.0, 0.0],
                color: [10, 20, 30, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [10.0, 10.0],
                uv: [1.0, 1.0],
                color: [10, 20, 30, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [0.0, 10.0],
                uv: [0.0, 1.0],
                color: [10, 20, 30, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            },
        ]);
        let indices = vec![0u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
        let cmds = vec![DrawCmd {
            texture: CmdTexture::AtlasPage(0),
            scissor: None,
            first_index: 0,
            index_count: 12,
            layer: 0,
        }];
        let prepared = prepare(&cmds, &mixed, &indices, true);
        assert!(prepared.instances.is_empty(), "whole-cmd fallback");
        assert!(matches!(prepared.draws[0].kind, DrawKind::Indexed { .. }));
        assert_eq!(prepared.draws.len(), 1, "no extra draw left behind");
    }

    #[test]
    fn a_blocked_probe_leaves_no_orphan_instances() {
        // Raw tri + instanced quad, same texture/scissor: the merge's
        // probe converts the quad, sees the merge stop, and must roll
        // it BACK — the quad's own draw then starts at instance 0.
        let tri = vec![
            UiVertex {
                pos: [0.0, 0.0],
                uv: [0.0, 0.0],
                color: [1, 2, 3, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [9.0, 0.0],
                uv: [0.0, 0.0],
                color: [4, 5, 6, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            },
            UiVertex {
                pos: [4.0, 9.0],
                uv: [0.0, 0.0],
                color: [7, 8, 9, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            },
        ];
        let mut vertices = tri;
        vertices.extend_from_slice(&quad_vertices(0.0, 0.0, 10.0, 10.0));
        let indices = vec![0u32, 1, 2, 3, 4, 5, 3, 5, 6];
        let cmds = vec![
            DrawCmd {
                texture: CmdTexture::AtlasPage(0),
                scissor: None,
                first_index: 0,
                index_count: 3,
                layer: 0,
            },
            DrawCmd {
                texture: CmdTexture::AtlasPage(0),
                scissor: None,
                first_index: 3,
                index_count: 6,
                layer: 1,
            },
        ];
        let prepared = prepare(&cmds, &vertices, &indices, true);
        assert_eq!(prepared.instances.len(), 1, "no orphan instances");
        assert_eq!(prepared.indices.len(), 3, "only the raw tri");
        match prepared.draws[1].kind {
            DrawKind::Instanced { first, count } => {
                assert_eq!((first, count), (0, 1), "quad instanced at 0")
            }
            ref other => panic!("expected instanced: {other:?}"),
        }
    }

    #[test]
    fn a_merged_multi_quad_command_converts_all() {
        // push_cmd merges consecutive same-texture quads into one
        // 12-index DrawCmd — every sub-quad must still convert.
        let mut vertices = Vec::new();
        vertices.extend_from_slice(&quad_vertices(0.0, 0.0, 10.0, 10.0));
        vertices.extend_from_slice(&quad_vertices(20.0, 0.0, 10.0, 10.0));
        let indices = vec![0u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
        let cmds = vec![DrawCmd {
            texture: CmdTexture::AtlasPage(0),
            scissor: None,
            first_index: 0,
            index_count: 12,
            layer: 0,
        }];
        let prepared = prepare(&cmds, &vertices, &indices, true);
        assert_eq!(prepared.instances.len(), 2);
        assert!(matches!(
            prepared.draws[0].kind,
            DrawKind::Instanced { count: 2, .. }
        ));
        assert!(prepared.vertices.is_empty());
    }

    #[test]
    fn varying_color_triangles_fall_back() {
        let vertices: Vec<UiVertex> = (0..4)
            .map(|i| UiVertex {
                pos: [0.0, i as f32],
                uv: [0.0, 0.0],
                color: [i as u8; 4],
                params: [0.0; 4],
                extra: [0.0; 2],
            })
            .collect();
        let indices = vec![0u32, 1, 2, 0, 2, 3];
        let cmds = vec![quad_cmd(0, 0)];
        let prepared = prepare(&cmds, &vertices, &indices, true);
        assert!(prepared.instances.is_empty());
        assert!(matches!(prepared.draws[0].kind, DrawKind::Indexed { .. }));
        assert_eq!(prepared.vertices.len(), 4);
    }
    fn tri_vertices(seed: f32) -> [UiVertex; 3] {
        (0..3)
            .map(|i| UiVertex {
                pos: [seed + i as f32 * 4.0, seed + i as f32],
                uv: [0.0, 0.0],
                color: [i as u8 + 1, 2, 3, 255],
                params: [0.0; 4],
                extra: [0.0; 2],
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[test]
    fn preparation_remap_resets_between_records() {
        let mut out = Prepared::default();
        let tris_a = tri_vertices(0.0).to_vec();
        let cmds_a = vec![DrawCmd {
            texture: CmdTexture::AtlasPage(0),
            scissor: None,
            first_index: 0,
            index_count: 3,
            layer: 0,
        }];
        let idx_a = vec![0u32, 1, 2];
        prepare_into(&mut out, &cmds_a, &tris_a, &idx_a, true);
        assert_eq!(out.indices.len(), 3);
        assert_eq!(out.remap.len(), 3, "sized to the source vertex count");

        let mut quads = Vec::new();
        quads.extend_from_slice(&quad_vertices(0.0, 0.0, 10.0, 10.0));
        let cmds_q = vec![quad_cmd(0, 0)];
        let idx_q = vec![0u32, 1, 2, 0, 2, 3];
        prepare_into(&mut out, &cmds_q, &quads, &idx_q, true);
        assert_eq!(out.instances.len(), 1);
        assert!(out.indices.is_empty());

        let tris_b = tri_vertices(100.0).to_vec();
        prepare_into(&mut out, &cmds_a, &tris_b, &idx_a, true);
        assert_eq!(out.vertices.len(), 3);
        for k in 0..3 {
            assert_eq!(
                out.vertices[out.indices[k] as usize].pos,
                tris_b[idx_a[k] as usize].pos
            );
        }
    }

    #[test]
    fn preparation_remap_dedups_across_commands() {
        let vertices = tri_vertices(0.0).to_vec();
        let indices = vec![0u32, 1, 2, 0, 2, 1];
        let cmds = vec![
            DrawCmd {
                texture: CmdTexture::AtlasPage(0),
                scissor: None,
                first_index: 0,
                index_count: 3,
                layer: 0,
            },
            DrawCmd {
                texture: CmdTexture::AtlasPage(1),
                scissor: None,
                first_index: 3,
                index_count: 3,
                layer: 0,
            },
        ];
        let prepared = prepare(&cmds, &vertices, &indices, true);
        assert_eq!(prepared.draws.len(), 2);
        assert_eq!(prepared.vertices.len(), 3, "vertex 0 compacted once");
        for (k, &new) in prepared.indices.iter().enumerate() {
            assert_eq!(
                prepared.vertices[new as usize].pos,
                vertices[indices[k] as usize].pos
            );
        }
    }

    #[test]
    fn preparation_reordered_indices_reconstruct_source() {
        let vertices = tri_vertices(3.0).to_vec();
        let indices = vec![2u32, 0, 2, 1, 2, 0];
        let cmds = vec![DrawCmd {
            texture: CmdTexture::AtlasPage(0),
            scissor: None,
            first_index: 0,
            index_count: 6,
            layer: 0,
        }];
        let prepared = prepare(&cmds, &vertices, &indices, true);
        assert_eq!(prepared.vertices.len(), 3, "three distinct verts");
        for (k, &new) in prepared.indices.iter().enumerate() {
            assert_eq!(
                prepared.vertices[new as usize].pos, vertices[indices[k] as usize].pos,
                "index {k}"
            );
        }
    }

    #[test]
    #[ignore]
    fn preparation_profile() {
        use crate::ui::draw::{DrawList, GradientAxis};
        use crate::ui::{Color, Rect};
        use std::time::Instant;
        for mode in 0..3 {
            let mut draw = DrawList::new();
            draw.begin((2400.0, 1200.0));
            for i in 0..800 {
                let rect = Rect::new((i % 40) as f32 * 60.0, (i / 40) as f32 * 40.0, 58.0, 38.0);
                if mode == 1 || (mode == 2 && i % 4 == 0) {
                    draw.gradient(
                        rect,
                        Color::rgb(10, 20, 30),
                        Color::rgb(80, 90, 100),
                        GradientAxis::TopToBottom,
                    );
                } else {
                    draw.round_rect(rect, 6.0, Color::rgb(10, 20, 30));
                }
            }
            let frame = draw.finish((2400.0, 1200.0));
            let mut out = Prepared::default();
            for _ in 0..30 {
                prepare_into(&mut out, &frame.cmds, &frame.vertices, &frame.indices, true);
            }
            let mut samples = Vec::with_capacity(2000);
            for _ in 0..2000 {
                let start = Instant::now();
                prepare_into(&mut out, &frame.cmds, &frame.vertices, &frame.indices, true);
                std::hint::black_box(&out);
                samples.push(start.elapsed().as_nanos());
            }
            samples.sort_unstable();
            println!(
                "preparation,mode={mode},frames=2000,p50_ns={},p95_ns={},p99_ns={},max_ns={},vertices={},indices={},instances={},draws={}",
                samples[1000],
                samples[1900],
                samples[1980],
                samples[1999],
                out.vertices.len(),
                out.indices.len(),
                out.instances.len(),
                out.draws.len()
            );
        }
    }
}
