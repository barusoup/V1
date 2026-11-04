//! Paragraph-level CoreText shaping: one `CTTypesetter` per text
//! segment, wrapped with `CTTypesetterSuggestLineBreak` — kerning,
//! ligatures and RTL survive because shaping sees the whole run, not
//! one cluster at a time. Unsupported clusters are probed first and
//! removed entirely (hidden, zero advance); runs whose face CoreText
//! substituted away from the selected span face are rejected by
//! PostScript name, never drawn as fallback boxes.

use super::face::{NativeFace, ShapedRun};
use super::ffi::*;
use super::warn::{WarnKind, warn_once};
use crate::ui::font::roles::script_role;
use crate::ui::font::{Align, GlyphCache, LayoutOptions, PositionedGlyph, ScriptRole, TextLayout};
use std::ffi::c_void;
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

/// A supported cluster's span inside the paragraph's UTF-16 space.
struct Span {
    /// Byte range in the ORIGINAL text (caret/ch mapping).
    start_byte: usize,
    /// UTF-16 range inside the paragraph attributed string.
    utf16_start: isize,
    utf16_len: isize,
    face: Arc<NativeFace>,
}

/// A grapheme boundary: (paragraph UTF-16 index, original byte offset).
/// Hidden clusters share the next visible index — their caret sits at
/// the same x, with a distinct byte stop.
struct Stop {
    utf16_index: isize,
    orig_byte: usize,
}

/// Build one paragraph (supported clusters joined, per-span font
/// attributes), or `None` if everything in `text` is hidden.
#[allow(clippy::too_many_arguments)]
fn build_paragraph(
    fonts: &super::super::Fonts,
    primary: &Arc<NativeFace>,
    px: f32,
    text: &str,
    seg_start: usize,
    letter_spacing: f32,
    spans: &mut Vec<Span>,
    stops: &mut Vec<Stop>,
) -> Option<(CFRef, isize)> {
    let mut para = String::new();
    let mut utf16_cursor = 0isize;
    for (index, cluster) in text.grapheme_indices(true) {
        let orig_byte = seg_start + index;
        let role = script_role(cluster);
        let face = match role {
            ScriptRole::Primary => Arc::clone(primary),
            _ => match fonts.role_face(role) {
                Some(face) => face,
                // No role face: hidden + warn — never primary-substitute.
                None => {
                    warn_once(WarnKind::MissingRole, "missing script face", cluster);
                    stops.push(Stop {
                        utf16_index: utf16_cursor,
                        orig_byte,
                    });
                    continue;
                }
            },
        };
        // Probe the cluster alone first: substituted or glyph-0 = hidden.
        let probe: ShapedRun = face.shape(cluster, px);
        if probe.substituted || probe.glyphs.contains(&0) {
            warn_once(WarnKind::UnsupportedCluster, face.name(), cluster);
            stops.push(Stop {
                utf16_index: utf16_cursor,
                orig_byte,
            });
            continue;
        }
        stops.push(Stop {
            utf16_index: utf16_cursor,
            orig_byte,
        });
        let utf16_len = cluster.chars().map(|c| c.len_utf16()).sum::<usize>() as isize;
        para.push_str(cluster);
        spans.push(Span {
            start_byte: orig_byte,
            utf16_start: utf16_cursor,
            utf16_len,
            face,
        });
        utf16_cursor += utf16_len;
    }
    if para.is_empty() {
        return None;
    }
    // One mutable attributed string; the dictionary uses the proper
    // CFType callbacks so the sized faces are retained for the line's
    // lifetime — a sized CFRef may drop while the attr string still
    // points at it.
    let attr = unsafe {
        let mutable = CFAttributedStringCreateMutable(std::ptr::null(), 0);
        if mutable.is_null() {
            return None;
        }
        let mutable = CFRef(mutable);
        let cf_text = cf_text(&para)?;
        CFAttributedStringReplaceString(
            mutable.raw(),
            CFRange {
                location: 0,
                length: 0,
            },
            cf_text.raw(),
        );
        for span in spans.iter() {
            let Some(sized) = span.face.sized(px) else {
                continue;
            };
            // CFAttributedStringSetAttribute retains the value per the
            // CoreFoundation ownership rules — the sized CFRef can drop
            // at the end of this iteration while the attr string keeps
            // it alive for the line/typesetter lifetime.
            CFAttributedStringSetAttribute(
                mutable.raw(),
                CFRange {
                    location: span.utf16_start,
                    length: span.utf16_len,
                },
                kCTFontAttributeName,
                sized.raw(),
            );
        }
        // Letter spacing (physical pixels per grapheme) applies to the
        // whole paragraph — kCTKernAttributeName is additive spacing in
        // points at the font's size; ours is physical px = pt at 1x.
        if letter_spacing != 0.0 && letter_spacing.is_finite() {
            let spacing = letter_spacing as f64;
            let kern = CFRef(CFNumberCreate(
                std::ptr::null(),
                6,
                &spacing as *const f64 as *const c_void,
            ));
            if !kern.raw().is_null() {
                CFAttributedStringSetAttribute(
                    mutable.raw(),
                    CFRange {
                        location: 0,
                        length: utf16_cursor,
                    },
                    kCTKernAttributeName,
                    kern.raw(),
                );
            }
        }
        mutable
    };
    Some((attr, utf16_cursor))
}

/// CTTypesetter-wrap a paragraph into lines: `(line_ref, utf16_range,
/// visible_width)`.
fn wrap_lines(
    attr: CFTypeRef,
    para_len: isize,
    wrap: Option<f32>,
    cluster_boundaries: &[isize],
) -> Vec<(CFRef, CFRange, f32)> {
    let typesetter = unsafe { CTTypesetterCreateWithAttributedString(attr) };
    if typesetter.is_null() {
        return Vec::new();
    }
    let typesetter = CFRef(typesetter);
    let mut lines = Vec::new();
    let mut start = 0isize;
    while start < para_len {
        let break_len = match wrap {
            Some(w) => {
                let len =
                    unsafe { CTTypesetterSuggestLineBreak(typesetter.raw(), start, w as f64) };
                if len <= 0 {
                    // Overfull single cluster: break at the first cluster
                    // boundary after `start`, never mid-grapheme.
                    cluster_boundaries
                        .iter()
                        .copied()
                        .find(|&b| b > start)
                        .map(|b| b - start)
                        .unwrap_or(para_len - start)
                } else {
                    // Snap the suggested break back to a cluster boundary
                    // (the typesetter may cut inside an extended grapheme
                    // when one alone overflows the measure). A break at a
                    // boundary — or at the paragraph end — is already
                    // valid and needs no snap.
                    let end = start + len;
                    let snapped = if end >= para_len || cluster_boundaries.contains(&end) {
                        end
                    } else {
                        cluster_boundaries
                            .iter()
                            .copied()
                            .rfind(|&b| b > start && b <= end)
                            .or_else(|| cluster_boundaries.iter().copied().find(|&b| b > start))
                            .unwrap_or(para_len)
                    };
                    (snapped - start).min(para_len - start)
                }
            }
            None => para_len - start,
        };
        let range = CFRange {
            location: start,
            length: break_len,
        };
        let line = unsafe { CTTypesetterCreateLine(typesetter.raw(), range) };
        if line.is_null() {
            return lines;
        }
        let line = CFRef(line);
        let mut ascent = 0.0;
        let mut descent = 0.0;
        let mut leading = 0.0;
        let width = unsafe {
            CTLineGetTypographicBounds(line.raw(), &mut ascent, &mut descent, &mut leading)
        };
        // Only a real wrap break drops trailing whitespace: it belongs to
        // neither line. The last line (or the unwrapped whole paragraph)
        // keeps it — a trailing blank's caret sits after the space.
        let broken = wrap.is_some() && start + break_len < para_len;
        let visible = if broken {
            let trailing = unsafe { CTLineGetTrailingWhitespaceWidth(line.raw()) };
            (width - trailing) as f32
        } else {
            width as f32
        };
        lines.push((line, range, visible));
        start += break_len;
    }
    lines
}

/// Emit a line's glyph runs. Returns `(positioned glyphs, line width)`.
/// `face_names` is the set of PostScript names allowed by the paragraph's
/// spans — a run on any other face is a rejected substitution.
#[allow(clippy::too_many_arguments)]
fn emit_line(
    line: CFTypeRef,
    cache: &mut GlyphCache,
    faces: &std::collections::BTreeMap<isize, Arc<NativeFace>>,
    utf16_to_byte: &dyn Fn(isize) -> usize,
    text: &str,
    px: f32,
    baseline: f32,
    pen: f32,
    out: &mut Vec<PositionedGlyph>,
) {
    let runs = unsafe { CTLineGetGlyphRuns(line) };
    let run_count = unsafe { CFArrayGetCount(runs) } as usize;
    for i in 0..run_count {
        let run = unsafe { CFArrayGetValueAtIndex(runs, i as isize) };
        let count = unsafe { CTRunGetGlyphCount(run) } as usize;
        if count == 0 {
            continue;
        }
        let mut glyphs = vec![0u16; count];
        let mut positions = vec![CGPoint::default(); count];
        let mut indices = vec![0isize; count];
        let whole = CFRange {
            location: 0,
            length: 0,
        };
        unsafe {
            CTRunGetGlyphs(run, whole, glyphs.as_mut_ptr());
            CTRunGetPositions(run, whole, positions.as_mut_ptr());
            CTRunGetStringIndices(run, whole, indices.as_mut_ptr());
        }
        // The run's face must be one of the paragraph's selected faces —
        // compare PostScript NAME (a sized copy shares the identity but
        // not the pointer).
        let attrs = unsafe { CTRunGetAttributes(run) };
        let run_font = if attrs.is_null() {
            std::ptr::null()
        } else {
            unsafe { CFDictionaryGetValue(attrs, kCTFontAttributeName) }
        };
        let run_name = if run_font.is_null() {
            String::new()
        } else {
            unsafe {
                let ps = CTFontCopyPostScriptName(run_font);
                let name = cf_to_string(ps).unwrap_or_default();
                if !ps.is_null() {
                    CFRelease(ps);
                }
                name
            }
        };
        for (g_index, &gid) in glyphs.iter().enumerate() {
            // CTRunGetStringIndices already returns indices into the
            // (whole) paragraph string — no line-range offset.
            let utf16_index = indices[g_index];
            let face = faces
                .range(..=utf16_index)
                .next_back()
                .map(|(_, f)| f)
                .cloned();
            let Some(face) = face else { continue };
            // Face-name validation: a different PostScript name is a
            // CoreText substitution — hide the glyph, warn, keep going.
            if !run_font.is_null() && !run_name.is_empty() && run_name != face.name() {
                warn_once(WarnKind::UnsupportedCluster, face.name(), "substituted run");
                continue;
            }
            if gid == 0 {
                continue;
            }
            let orig_byte = utf16_to_byte(utf16_index);
            let ch = text[orig_byte..].chars().next().unwrap_or('\u{FFFD}');
            let glyph = cache.native_glyph(&face, face.id(), gid, px);
            if glyph.width <= 0.0 || glyph.height <= 0.0 {
                continue;
            }
            let (px_pos, py_pos) = (positions[g_index].x as f32, positions[g_index].y as f32);
            // The cached glyph's own `advance` is the font metric; kerning
            // already lives in `positions` (shaped x).
            out.push(PositionedGlyph {
                glyph,
                x: pen + px_pos + glyph.x_offset,
                y: baseline - py_pos + glyph.y_offset,
                ch,
            });
        }
    }
}

/// Full text → [`TextLayout`]. The paragraph path preserves pair
/// kerning/ligatures/RTL; stops come from `CTLineGetOffsetForStringIndex`
/// over every grapheme UTF-16 boundary mapped back to source bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layout_native(
    fonts: &super::super::Fonts,
    cache: &mut GlyphCache,
    primary: &Arc<NativeFace>,
    px: f32,
    text: &str,
    options: LayoutOptions,
) -> TextLayout {
    if !px.is_finite() || px <= 0.0 {
        return TextLayout::default();
    }
    let line_advance = primary.line_height(px) * options.line_height;
    let ascent = primary.ascent(px);

    // Per '\n' segment: split on '\t' (a tab gets a manual 4-space stop),
    // then shape each piece as one paragraph.
    let space_advance = primary
        .glyph_id(' ')
        .map(|gid| primary.advance(px, gid))
        .unwrap_or(px * 0.3)
        .max(1.0);
    let tab_stop = space_advance * 4.0;

    /// One emitted line: glyphs + caret stops (x, orig byte).
    #[derive(Default)]
    struct Line {
        glyphs: Vec<PositionedGlyph>,
        stops: Vec<(f32, usize)>,
        width: f32,
        /// Pen position at the line's logical end (tab continuation).
        pen: f32,
    }
    let mut lines: Vec<Line> = Vec::new();
    let mut byte_cursor = 0usize;
    for segment in text.split('\n') {
        let seg_start = byte_cursor;
        byte_cursor += segment.len() + 1;
        // Tab pieces: shape between '\t's; the tab itself is a 4-space
        // stop on the CURRENT line.
        let mut pieces = Vec::new();
        let mut piece_start = 0usize;
        let mut tabs: Vec<usize> = Vec::new();
        for (index, _) in segment.char_indices() {
            if segment.as_bytes()[index] == b'\t' {
                pieces.push((seg_start + piece_start, &segment[piece_start..index]));
                tabs.push(seg_start + index);
                piece_start = index + 1;
            }
        }
        pieces.push((seg_start + piece_start, &segment[piece_start..]));

        // The current line a piece extends (tab continues it).
        let mut current: Option<usize> = None;
        for (piece_index, (piece_start_byte, piece)) in pieces.iter().enumerate() {
            let piece_start_byte = *piece_start_byte;
            if piece.is_empty() && piece_index + 1 == pieces.len() && tabs.is_empty() {
                continue;
            }
            let mut spans = Vec::new();
            let mut stops = Vec::new();
            let paragraph = build_paragraph(
                fonts,
                primary,
                px,
                piece,
                piece_start_byte,
                options.letter_spacing,
                &mut spans,
                &mut stops,
            );
            // UTF-16 → original byte map for run indices and stops.
            let mut utf16_bytes: Vec<usize> = Vec::new();
            for span in &spans {
                let cluster = &text[span.start_byte..];
                let mut u = span.utf16_start;
                for (b, ch) in cluster.char_indices() {
                    let end = (u as usize + ch.len_utf16()) as isize;
                    while utf16_bytes.len() <= (end - 1) as usize {
                        utf16_bytes.push(span.start_byte + b);
                    }
                    u = end;
                }
            }
            let utf16_to_byte = |index: isize| -> usize {
                if index <= 0 {
                    return piece_start_byte;
                }
                utf16_bytes
                    .get(index as usize)
                    .copied()
                    .unwrap_or(piece_start_byte + piece.len())
            };
            let faces: std::collections::BTreeMap<isize, Arc<NativeFace>> = spans
                .iter()
                .map(|s| (s.utf16_start, Arc::clone(&s.face)))
                .collect();
            let boundaries: Vec<isize> = spans.iter().map(|s| s.utf16_start).collect();
            let Some((attr, para_len)) = paragraph else {
                // All hidden: stops still needed (zero advance).
                for stop in stops {
                    let line_index = match current {
                        Some(i) => i,
                        None => {
                            lines.push(Line::default());
                            current = Some(lines.len() - 1);
                            lines.len() - 1
                        }
                    };
                    let pen = lines[line_index].pen;
                    lines[line_index].stops.push((pen, stop.orig_byte));
                }
                continue;
            };
            let wrapped = wrap_lines(attr.raw(), para_len, options.wrap_width, &boundaries);
            for (line_index_in_piece, (line, range, visible)) in wrapped.iter().enumerate() {
                // Each wrapped line after the first begins a new visual
                // line (the first continues the tab-joined line).
                let line_index = if line_index_in_piece == 0 {
                    match current {
                        Some(i) => i,
                        None => {
                            lines.push(Line::default());
                            current = Some(lines.len() - 1);
                            lines.len() - 1
                        }
                    }
                } else {
                    lines.push(Line::default());
                    lines.len() - 1
                };
                let baseline = ascent + line_advance * line_index as f32;
                let pen = lines[line_index].pen;
                emit_line(
                    line.raw(),
                    cache,
                    &faces,
                    &utf16_to_byte,
                    text,
                    px,
                    baseline,
                    pen,
                    &mut lines[line_index].glyphs,
                );
                // Caret stops at every grapheme boundary inside the range.
                for stop in &stops {
                    if stop.utf16_index >= range.location
                        && stop.utf16_index < range.location + range.length
                    {
                        let x = unsafe {
                            CTLineGetOffsetForStringIndex(
                                line.raw(),
                                stop.utf16_index,
                                std::ptr::null_mut(),
                            )
                        } as f32;
                        lines[line_index].stops.push((pen + x, stop.orig_byte));
                    }
                }
                // Line-end stop: the absolute index one past this line's
                // range inside the paragraph.
                let end_index = range.location + range.length;
                let end_x = unsafe {
                    CTLineGetOffsetForStringIndex(line.raw(), end_index, std::ptr::null_mut())
                } as f32;
                // No clamp to `visible`: an RTL line's end offset can
                // legitimately sit at x=0 past trailing whitespace.
                lines[line_index].stops.push((
                    pen + end_x,
                    utf16_to_byte(end_index).min(piece_start_byte + piece.len()),
                ));
                lines[line_index].pen = pen + *visible;
                lines[line_index].width = lines[line_index].pen;
            }
            // `current` tracks the LAST wrapped line: a following tab or
            // piece joins the tail, not the first line.
            if !wrapped.is_empty() {
                current = Some(lines.len() - 1);
            }
            // A tab after this piece advances the pen to the next stop.
            if let Some(&tab_byte) = tabs.get(piece_index) {
                let line_index = current.expect("piece emitted");
                let pen = lines[line_index].pen;
                lines[line_index].stops.push((pen, tab_byte));
                let next = ((pen / tab_stop).floor() + 1.0) * tab_stop;
                lines[line_index].pen = next.max(pen + space_advance);
                lines[line_index].width = lines[line_index].pen;
                let pen_after = lines[line_index].pen;
                lines[line_index].stops.push((pen_after, tab_byte + 1));
            }
        }
        // The next segment's first piece always opens a fresh line; an
        // empty/all-hidden segment still owns one.
        if current.is_none() {
            lines.push(Line::default());
        }
    }
    if lines.is_empty() {
        lines.push(Line::default());
    }

    // Align + assemble.
    let line_count = lines.len();
    let mut out = TextLayout {
        glyphs: Vec::new(),
        width: 0.0,
        height: line_advance * line_count as f32,
        line_count,
        stops: Vec::new(),
        first_baseline: ascent,
        line_advance,
    };
    out.width = lines.iter().map(|l| l.width).fold(0.0, f32::max);
    for line in lines {
        let offset = match options.align {
            Align::Left => 0.0,
            Align::Center => -line.width * 0.5,
            Align::Right => -line.width,
        };
        for mut glyph in line.glyphs {
            glyph.x += offset;
            out.glyphs.push(glyph);
        }
        for (x, byte) in line.stops {
            out.stops.push((x + offset, byte));
        }
    }
    // Final stop: end of the source text.
    out.stops.push((out.width, text.len()));
    out
}
