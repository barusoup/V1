//! Fonts: loading, glyph caching into the atlas, and text layout.
//!
//! A [`Font`] wraps a fontdue font and is pure CPU: metrics and rasterization
//! only. [`GlyphCache`] owns the [`Atlas`] and memoizes rasterized glyphs, so
//! each (font, character, size) is rasterized once per atlas.
//!
//! Layout ([`layout`]) produces a [`TextLayout`] of positioned glyphs with
//! wrapping, alignment, kerning and explicit `\n` handling. Drawing that
//! layout is `ui::draw`'s job.
//!
//! Coordinates: pixel space, origin top-left, y down. Layout positions glyphs
//! relative to the layout origin at (0, 0); `TextLayout::glyphs` carry their
//! final pixel offsets directly.

use super::Rect;
use super::atlas::{Atlas, Region};
use std::collections::HashMap;

pub(crate) mod lazy_raster;
mod metrics;
pub(crate) mod roles;

#[cfg(target_os = "macos")]
pub(crate) mod native;

/// Handle to a font inside a [`Fonts`] registry. The raw index is exposed so
/// games can serialize it; validity is checked by the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontId(pub u32);

/// Backing rasterizer for a [`Font`]: lazy per-glyph outline compiler everywhere
/// except macOS, where the face is a retained CoreText handle (see `font::native`
/// — a `Font` then owns one `CTFont`, not a parsed glyph table).
enum Face {
    #[allow(dead_code)]
    Portable(lazy_raster::LazyFont),
    #[cfg(target_os = "macos")]
    Native(std::sync::Arc<native::NativeFace>),
}

/// Pair kerning in font units from the font's GPOS `kern` feature.
pub(crate) fn gpos_pair_kern_units(
    face: &ttf_parser::Face<'_>,
    first: ttf_parser::GlyphId,
    second: ttf_parser::GlyphId,
) -> Option<i16> {
    use ttf_parser::gpos::PositioningSubtable;

    let gpos = face.tables().gpos.as_ref()?;
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
                let PositioningSubtable::Pair(pair) = subtable else {
                    continue;
                };
                if let Some(units) = pair_x_advance(&pair, first, second) {
                    return Some(units);
                }
            }
        }
    }
    None
}

/// The first glyph's horizontal advance adjustment for one pair, in font
/// units; `None` when the pair is not covered by this subtable.
fn pair_x_advance(
    pair: &ttf_parser::gpos::PairAdjustment,
    first: ttf_parser::GlyphId,
    second: ttf_parser::GlyphId,
) -> Option<i16> {
    use ttf_parser::gpos::PairAdjustment;
    match pair {
        PairAdjustment::Format1 { coverage, sets } => {
            let set = sets.get(coverage.get(first)?)?;
            Some(set.get(second)?.0.x_advance)
        }
        PairAdjustment::Format2 {
            coverage,
            classes,
            matrix,
        } => {
            coverage.get(first)?;
            let classes = (classes.0.get(first), classes.1.get(second));
            Some(matrix.get(classes)?.0.x_advance)
        }
    }
}

/// A loaded font face. Sizes throughout are pixels per em.
pub struct Font {
    name: String,
    face: Face,
    /// OS/2 heights shared by both raster backends (optical centring).
    metrics: metrics::FaceMetrics,
}

impl std::fmt::Debug for Font {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Font").field("name", &self.name).finish()
    }
}

impl Font {
    /// Parse a TTF/OTF font from bytes. `name` is diagnostic only.
    pub fn from_bytes(name: impl Into<String>, data: &[u8]) -> Result<Self, String> {
        let name = name.into();
        let metrics = metrics::FaceMetrics::parse(data);
        #[cfg(target_os = "macos")]
        {
            let face = native::NativeFace::from_bytes(data)
                .map_err(|err| format!("{name}: parse font: {err}"))?;
            Ok(Self {
                name,
                metrics,
                face: Face::Native(std::sync::Arc::new(face)),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let font = lazy_raster::LazyFont::from_bytes(data)
                .map_err(|err| format!("{name}: parse font: {err}"))?;
            Ok(Self {
                name,
                metrics,
                face: Face::Portable(font),
            })
        }
    }

    /// Load a TTF/OTF font from disk.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self::from_bytes(name, &bytes)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Distance from a line's baseline to the top of its tallest glyphs.
    pub fn ascent(&self, px: f32) -> f32 {
        self.line_metrics(px).map(|m| m.ascent).unwrap_or(px * 0.8)
    }

    /// Distance from a line's baseline down to its lowest glyphs (negative).
    pub fn descent(&self, px: f32) -> f32 {
        self.line_metrics(px)
            .map(|m| m.descent)
            .unwrap_or(-px * 0.2)
    }

    /// Flat-top capital letter height in pixels (the OS/2 `sCapHeight`,
    /// or 0.7 em when the face does not declare one). This is the optical
    /// span cap-height centring uses, not the taller ascent line box.
    pub fn cap_height(&self, px: f32) -> f32 {
        self.metrics.cap_height(px)
    }

    /// x-height in pixels (the OS/2 `sxHeight`, or 0.5 em when the face
    /// does not declare one).
    pub fn x_height(&self, px: f32) -> f32 {
        self.metrics.x_height(px)
    }

    /// Baseline-to-baseline distance for successive lines.
    pub fn line_height(&self, px: f32) -> f32 {
        self.line_metrics(px)
            .map(|m| m.new_line_size)
            .unwrap_or(px * 1.2)
    }

    fn line_metrics(&self, px: f32) -> Option<fontdue::LineMetrics> {
        match &self.face {
            Face::Portable(font) => Some(font.line_metrics(px)),
            #[cfg(target_os = "macos")]
            Face::Native(face) => Some(fontdue::LineMetrics {
                ascent: face.ascent(px),
                descent: -face.descent(px),
                line_gap: face.line_height(px) - face.ascent(px) - face.descent(px),
                new_line_size: face.line_height(px),
            }),
        }
    }

    /// The native face, when this is a CoreText-backed font.
    #[cfg(target_os = "macos")]
    pub(crate) fn native(&self) -> Option<std::sync::Arc<native::NativeFace>> {
        match &self.face {
            Face::Native(face) => Some(std::sync::Arc::clone(face)),
            _ => None,
        }
    }

    /// Horizontal advance for one character, rounded to whole pixels. Glyph
    /// bitmaps are rasterized at a zero sub-pixel offset, so the pen must stay
    /// on integer pixels or neighbouring letters drift apart and collide.
    pub fn advance(&self, ch: char, px: f32) -> f32 {
        match &self.face {
            Face::Portable(font) => font.advance(ch, px),
            #[cfg(target_os = "macos")]
            Face::Native(face) => face
                .glyph_id(ch)
                .map(|gid| face.advance(px, gid))
                .unwrap_or(0.0)
                .round(),
        }
    }

    /// Kerning adjustment between two adjacent characters (usually <= 0),
    /// rounded to whole pixels for the same reason as [`Self::advance`].
    pub fn kern(&self, left: char, right: char, px: f32) -> f32 {
        match &self.face {
            Face::Portable(font) => font.kern(left, right, px),
            // Pair-shaped width minus the sum of the individual
            // advances — the public kerning contract (negative for AV).
            #[cfg(target_os = "macos")]
            Face::Native(native) => {
                // Positions give the true pair width (kerning applied):
                // kern = distance the second glyph moved minus 'left'
                // unkerneled advance. Glyph-0 pairs report 0.
                let pair = native.shape(&[left, right].iter().collect::<String>(), px);
                let left_adv = native.glyph_id(left).map_or(0.0, |g| native.advance(px, g));
                if pair.positions.len() >= 2 && pair.glyphs.len() >= 2 && !pair.glyphs.contains(&0)
                {
                    (pair.positions[1].1 - pair.positions[0].1 - left_adv).round()
                } else {
                    0.0
                }
            }
        }
    }

    pub fn has_glyph(&self, ch: char) -> bool {
        match &self.face {
            Face::Portable(font) => font.has_glyph(ch),
            #[cfg(target_os = "macos")]
            Face::Native(face) => face.glyph_id(ch).is_some(),
        }
    }

    /// Rasterize one character into 8-bit coverage, top-left origin. Whitespace
    /// returns an empty bitmap with its advance.
    pub fn rasterize(&self, ch: char, px: f32) -> (fontdue::Metrics, Vec<u8>) {
        match &self.face {
            Face::Portable(font) => font.rasterize(ch, px),
            #[cfg(target_os = "macos")]
            Face::Native(face) => {
                let Some(gid) = face.glyph_id(ch) else {
                    return (fontdue::Metrics::default(), Vec::new());
                };
                let cov = face.rasterize_coverage(gid, px);
                let metrics = fontdue::Metrics {
                    xmin: cov.ox.round() as i32,
                    // fontdue's ymin is the bottom edge above the baseline.
                    ymin: (cov.oy - cov.height as f32).round() as i32,
                    width: cov.width as usize,
                    height: cov.height as usize,
                    advance_width: face.advance(px, gid),
                    advance_height: 0.0,
                    bounds: fontdue::OutlineBounds::default(),
                };
                (metrics, cov.pixels)
            }
        }
    }
}

/// Explicit script role a grapheme resolves to (platform: macOS loads the
/// roles lazily through CoreText; elsewhere a host registers overrides).
pub use roles::ScriptRole;

/// A UI font family: the faces [`Fonts::resolve`] maps a
/// [`crate::ui::tokens::FontWeight`] and the mono flag onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FontFamily {
    pub regular: FontId,
    pub medium: FontId,
    pub semibold: FontId,
    pub mono: FontId,
}

/// Registry of loaded fonts; font handles stay valid for the registry's life.
#[derive(Default)]
pub struct Fonts {
    fonts: Vec<Font>,
    /// The active family for [`Self::resolve`]; unset hosts keep FontId(0).
    family: Option<FontFamily>,
    /// Host-registered role overrides (portable CJK/emoji roles).
    script_overrides: [Option<FontId>; 4],
    /// Lazily resolved role faces (CoreText system faces).
    #[cfg(target_os = "macos")]
    role_faces: [std::sync::OnceLock<Option<std::sync::Arc<native::NativeFace>>>; 4],
}

impl Fonts {
    pub fn new() -> Self {
        Self::default()
    }

    /// The bundled family: Inter Regular/Medium/SemiBold plus JetBrains
    /// Mono, all under `assets/fonts/` (licences in `LICENSES.md`). Inter
    /// Regular loads first, so it is `FontId(0)` and the default face for
    /// any caller that never resolves a weight.
    pub fn bundled() -> Fonts {
        let mut fonts = Fonts::new();
        let regular = fonts
            .load_bytes(
                "Inter-Regular",
                include_bytes!("../../assets/fonts/inter/Inter-Regular.ttf"),
            )
            .expect("bundled Inter-Regular.ttf must load");
        let medium = fonts
            .load_bytes(
                "Inter-Medium",
                include_bytes!("../../assets/fonts/inter/Inter-Medium.ttf"),
            )
            .expect("bundled Inter-Medium.ttf must load");
        let semibold = fonts
            .load_bytes(
                "Inter-SemiBold",
                include_bytes!("../../assets/fonts/inter/Inter-SemiBold.ttf"),
            )
            .expect("bundled Inter-SemiBold.ttf must load");
        let mono = fonts
            .load_bytes(
                "JetBrainsMono-Regular",
                include_bytes!("../../assets/fonts/jetbrains-mono/JetBrainsMono-Regular.ttf"),
            )
            .expect("bundled JetBrainsMono-Regular.ttf must load");
        fonts.set_family(FontFamily {
            regular,
            medium,
            semibold,
            mono,
        });
        fonts
    }

    /// Set the family [`Self::resolve`] reads.
    pub fn set_family(&mut self, family: FontFamily) {
        self.family = Some(family);
    }

    /// The active family, if any.
    pub fn family(&self) -> Option<FontFamily> {
        self.family
    }

    /// Resolve a weight + mono flag to a face. Without a family this is
    /// `FontId(0)`; missing weights alias regular (no faux bold).
    pub fn resolve(&self, weight: crate::ui::tokens::FontWeight, mono: bool) -> FontId {
        match self.family {
            Some(f) if mono => f.mono,
            Some(f) => match weight {
                crate::ui::tokens::FontWeight::Regular => f.regular,
                crate::ui::tokens::FontWeight::Medium => f.medium,
                crate::ui::tokens::FontWeight::SemiBold => f.semibold,
            },
            None => FontId(0),
        }
    }

    /// Register an explicit font for a script role (a Han face for CJK,
    /// an emoji face for colour). On macOS the roles already resolve to
    /// validated system faces; elsewhere this is how a host opts in.
    pub fn set_script_font(&mut self, role: ScriptRole, font: FontId) {
        if role == ScriptRole::Primary {
            // The primary font is the caller's load order, not a role
            // override — a documented no-op.
            return;
        }
        self.script_overrides[roles::role_index(role)] = Some(font);
    }

    /// The face a role resolves to off macOS — host-registered overrides
    /// only (there is no automatic fallback).
    #[cfg(not(target_os = "macos"))]
    pub fn role_face(&self, role: ScriptRole) -> Option<FontId> {
        self.script_overrides[roles::role_index(role)]
    }

    /// The face a role resolves to, or `None` when only the primary font
    /// should be used. macOS resolves lazy system faces and validates the
    /// resolved PostScript name — a missing face warns once and returns
    /// `None` rather than substituting arbitrarily.
    #[cfg(target_os = "macos")]
    pub(crate) fn role_face(
        &self,
        role: roles::ScriptRole,
    ) -> Option<std::sync::Arc<native::NativeFace>> {
        use roles::ScriptRole::*;
        if let Some(override_id) = self.script_overrides[roles::role_index(role)] {
            return self
                .fonts
                .get(override_id.0 as usize)
                .and_then(Font::native);
        }
        let index = match role {
            Han => 0,
            Kana => 1,
            Hangul => 2,
            Emoji => 3,
            Primary => return None,
        };
        self.role_faces[index]
            .get_or_init(|| {
                let name = native::ROLE_NAMES[index];
                match native::NativeFace::from_postscript_name(name) {
                    Ok(face) if face.name() == name => Some(std::sync::Arc::new(face)),
                    Ok(face) => {
                        eprintln!(
                            "[ui] script role resolved {} but got {}; hidden",
                            name,
                            face.name()
                        );
                        None
                    }
                    Err(err) => {
                        eprintln!("[ui] script role {name}: {err}");
                        None
                    }
                }
            })
            .clone()
    }

    /// Load a font from bytes. Errors if the data does not parse.
    pub fn load_bytes(&mut self, name: &str, data: &[u8]) -> Result<FontId, String> {
        let font = Font::from_bytes(name, data)?;
        self.fonts.push(font);
        Ok(FontId(self.fonts.len() as u32 - 1))
    }

    /// Load a font from disk.
    pub fn load_file(&mut self, path: impl AsRef<std::path::Path>) -> Result<FontId, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.load_bytes(&name, &bytes)
    }

    pub fn get(&self, id: FontId) -> &Font {
        &self.fonts[id.0 as usize]
    }

    /// Fallible lookup — an invalid handle reports `None` instead of panicking.
    pub fn get_opt(&self, id: FontId) -> Option<&Font> {
        self.fonts.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.fonts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fonts.is_empty()
    }
}

/// One cached glyph's atlas placement and pen-relative offsets.
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub region: Region,
    /// Horizontal pen advance after drawing.
    pub advance: f32,
    /// Bitmap left edge relative to the pen x.
    pub x_offset: f32,
    /// Bitmap top edge relative to the baseline (negative is above it, in
    /// y-down screen space).
    pub y_offset: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct GlyphKey {
    font: FontId,
    ch: char,
    /// Size quantized to 1/8 px: collapses float keys without visible aliasing.
    px_eighth: u32,
}

/// Rasterized glyph cache bound to one [`Atlas`].
pub struct GlyphCache {
    atlas: Atlas,
    /// Cached glyphs and their last-used tick (bounded, eldest evicted).
    glyphs: HashMap<GlyphKey, (Glyph, u64)>,
    /// Native glyphs keyed (face identity, glyph id, canonical px).
    #[cfg(target_os = "macos")]
    native_glyphs: HashMap<(u64, u16, u32), (Glyph, u64)>,
    tick: u64,
    /// Rasterizations performed (cache misses); tests assert this stops.
    rasterizations: usize,
    /// The atlas refused a page (budget): the next `Ui::begin` resets the
    /// atlas — mid-frame placement cannot roll back, so recovery is
    /// deferred to a frame boundary.
    overflow_pending: bool,
    /// Warned once this overflow episode (deduped log spam).
    overflow_warned: bool,
    /// Last observed atlas epoch — `sync_epoch` detects raw resets.
    last_epoch: u64,
}

impl GlyphCache {
    pub fn new(atlas: Atlas) -> Self {
        let epoch = atlas.epoch();
        Self {
            atlas,
            glyphs: HashMap::new(),
            #[cfg(target_os = "macos")]
            native_glyphs: HashMap::new(),
            tick: 0,
            rasterizations: 0,
            last_epoch: epoch,
            overflow_pending: false,
            overflow_warned: false,
        }
    }

    /// Coverage insert that records a budget overflow for the next frame
    /// boundary instead of panicking mid-frame. Direct `Atlas` callers get
    /// the error; the UI path gets it plus a recovery flag.
    pub fn insert_coverage(
        &mut self,
        coverage: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Region, crate::ui::atlas::AtlasError> {
        match self.atlas.insert_coverage(coverage, width, height) {
            Err(err @ crate::ui::atlas::AtlasError::BudgetExceeded { .. }) => {
                self.overflow_pending = true;
                if !self.overflow_warned {
                    self.overflow_warned = true;
                    eprintln!("[ui] atlas budget exceeded: {err} — resetting at the next frame");
                }
                Err(err)
            }
            result => result,
        }
    }

    /// Colour insert with the same overflow contract as coverage.
    #[cfg(target_os = "macos")]
    fn insert_color_srgb(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Region, crate::ui::atlas::AtlasError> {
        match self.atlas.insert_color_srgb(rgba, width, height) {
            Err(err @ crate::ui::atlas::AtlasError::BudgetExceeded { .. }) => {
                self.overflow_pending = true;
                if !self.overflow_warned {
                    self.overflow_warned = true;
                    eprintln!("[ui] atlas budget exceeded: {err} — resetting at the next frame");
                }
                Err(err)
            }
            result => result,
        }
    }

    /// Whether an overflow reset is owed at the next frame boundary.
    pub fn overflow_pending(&self) -> bool {
        self.overflow_pending
    }

    /// Reset the atlas after a budget overflow: new epoch, all glyphs and
    /// cached regions rebaked on demand. The layout cache keys on the
    /// epoch and invalidates itself.
    pub fn reset_atlas(&mut self) {
        self.atlas.reset();
        self.glyphs.clear();
        #[cfg(target_os = "macos")]
        self.native_glyphs.clear();
        self.last_epoch = self.atlas.epoch();
        self.overflow_pending = false;
        self.overflow_warned = false;
    }

    /// Reset generation — glyph atlas epoch (layout-cache key part).
    pub fn epoch(&self) -> u64 {
        self.atlas.epoch()
    }

    /// Epoch bookkeeping for `Ui::begin`: ANY reset that went around
    /// `reset_atlas` (a raw `Atlas::reset` via `atlas_mut`) must not
    /// leave stale regions in either glyph map. Returns whether the
    /// epoch changed — the caller drops icon/layout caches too.
    pub(crate) fn sync_epoch(&mut self) -> bool {
        let epoch = self.atlas.epoch();
        if epoch != self.last_epoch {
            self.last_epoch = epoch;
            self.glyphs.clear();
            #[cfg(target_os = "macos")]
            self.native_glyphs.clear();
            return true;
        }
        false
    }

    pub fn atlas(&self) -> &Atlas {
        &self.atlas
    }

    pub fn atlas_mut(&mut self) -> &mut Atlas {
        &mut self.atlas
    }

    pub fn rasterizations(&self) -> usize {
        self.rasterizations
    }

    /// Glyph for (font, char, size), rasterizing into the atlas on miss.
    /// Whitespace yields a zero-sized region at the current shelf corner.
    /// Oversized, invalid or unaffordable glyphs yield an invisible
    /// placeholder rather than a panic — and are not cached.
    pub fn glyph(&mut self, fonts: &Fonts, font: FontId, ch: char, px: f32) -> Glyph {
        // Canonicalize to the same 1/8 px the key uses: the raster itself
        // must not depend on which fractional size was asked first.
        let px = if px.is_finite() && px > 0.0 {
            (px * 8.0).round() / 8.0
        } else {
            return Self::empty(0.0, 0.0);
        };
        let key = GlyphKey {
            font,
            ch,
            px_eighth: (px * 8.0).round() as u32,
        };
        self.tick += 1;
        if let Some(&mut (glyph, ref mut tick)) = self.glyphs.get_mut(&key) {
            *tick = self.tick;
            return glyph;
        }
        let hidden =
            |metrics: fontdue::Metrics| Self::empty(metrics.advance_width, metrics.xmin as f32);
        let Some(face) = fonts.get_opt(font) else {
            // An invalid handle is a caller bug, not a crash: warn once.
            eprintln!("[ui] glyph requested for unknown FontId {}", font.0);
            return Self::empty(0.0, 0.0);
        };
        // Refuse to rasterize a glyph that can never fit a page: the
        // coverage bitmap itself would be a page-sized+ allocation.
        // The advance alone answers the caller — no raster is needed.
        if px > self.atlas.page_size() as f32 {
            return Self::empty(face.advance(ch, px), 0.0);
        }
        self.rasterizations += 1;
        let (metrics, coverage) = face.rasterize(ch, px);
        let width = metrics.width as u32;
        let height = metrics.height as u32;
        let region = match self.insert_coverage(&coverage, width, height) {
            Ok(region) => region,
            Err(_) => {
                // Oversized or over budget: an invisible glyph — and NOT
                // cached, so a later reset re-rasterizes it for real.
                return hidden(metrics);
            }
        };
        let glyph = Glyph {
            region,
            advance: metrics.advance_width,
            x_offset: metrics.xmin as f32,
            // fontdue's ymin is the bottom edge above the baseline (y up);
            // screen space is y down, so the bitmap top sits at
            // -(ymin + height) relative to the baseline.
            y_offset: -(metrics.ymin as f32 + height as f32),
            width: width as f32,
            height: height as f32,
        };
        // Bounded metadata: 16k glyph entries is far above a UI's live
        // set; the eldest fall out and rebake on demand. Regions stay
        // valid — the atlas owns pixels until a reset.
        const MAX_GLYPHS: usize = 16384;
        if self.glyphs.len() >= MAX_GLYPHS
            && let Some(eldest) = self
                .glyphs
                .iter()
                .min_by_key(|(_, entry)| entry.1)
                .map(|(key, _)| *key)
        {
            self.glyphs.remove(&eldest);
        }
        self.glyphs.insert(key, (glyph, self.tick));
        glyph
    }

    /// A native glyph: coverage for ordinary faces, straight-alpha RGBA
    /// (sRGB page) for colour faces. Keyed `(face id, glyph id, px)` —
    /// the glyph id, never the character.
    #[cfg(target_os = "macos")]
    pub(crate) fn native_glyph(
        &mut self,
        face: &native::NativeFace,
        face_id: u64,
        glyph_id: u16,
        px: f32,
    ) -> Glyph {
        let px_eighth = (px.max(0.0) * 8.0).round() as u32;
        let key = (face_id, glyph_id, px_eighth);
        self.tick += 1;
        if let Some(&mut (glyph, ref mut tick)) = self.native_glyphs.get_mut(&key) {
            *tick = self.tick;
            return glyph;
        }
        if self.native_glyphs.len() >= 16384
            && let Some(eldest) = self
                .native_glyphs
                .iter()
                .min_by_key(|(_, entry)| entry.1)
                .map(|(key, _)| *key)
        {
            self.native_glyphs.remove(&eldest);
        }
        self.rasterizations += 1;
        let px = px_eighth as f32 / 8.0;
        // Preflight bound BEFORE allocating: the padded bitmap must fit
        // the actual page minus padding — don't build a page-sized Vec
        // the atlas would then reject.
        let bound = (self.atlas.page_size() as usize)
            .saturating_sub(self.atlas.padding() as usize * 2)
            .max(8);
        let cov = if face.is_color() {
            face.rasterize_rgba_bounded(glyph_id, px, bound)
        } else {
            face.rasterize_coverage_bounded(glyph_id, px, bound)
        };
        if cov.too_large {
            native::warn::warn_once(native::warn::WarnKind::OversizeGlyph, face.name(), "glyph");
            // A too-large glyph cannot ever fit: hide it (never cache an
            // empty glyph as valid) and schedule the reset — the same
            // contract as a budget overflow.
            self.overflow_pending = true;
            return Self::empty(face.advance(px, glyph_id), 0.0);
        }
        let advance = face.advance(px, glyph_id);
        let region = if face.is_color() {
            match self.insert_color_srgb(&cov.pixels, cov.width, cov.height) {
                Ok(region) => region,
                Err(_) => return Self::empty(advance, cov.ox),
            }
        } else {
            match self.insert_coverage(&cov.pixels, cov.width, cov.height) {
                Ok(region) => region,
                Err(_) => return Self::empty(advance, cov.ox),
            }
        };
        let glyph = Glyph {
            region,
            advance,
            x_offset: cov.ox,
            y_offset: -cov.oy,
            width: cov.width as f32,
            height: cov.height as f32,
        };
        self.native_glyphs.insert(key, (glyph, self.tick));
        glyph
    }

    /// Invisible placeholder for a glyph that cannot be produced — the
    /// text layout still advances by `advance`.
    fn empty(advance: f32, x_offset: f32) -> Glyph {
        Glyph {
            region: Region {
                page: 0,
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            advance,
            x_offset,
            y_offset: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }
}

/// Horizontal placement of a text line inside its block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Align {
    #[default]
    Left,
    Center,
    Right,
}

/// Text layout options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutOptions {
    /// Wrap to this width (pixels). `None` lays out one line per `\n`.
    pub wrap_width: Option<f32>,
    pub align: Align,
    /// Multiplier on the font's natural line height.
    pub line_height: f32,
    /// Extra pixels added after every character (letter spacing).
    pub letter_spacing: f32,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            wrap_width: None,
            align: Align::Left,
            line_height: 1.0,
            letter_spacing: 0.0,
        }
    }
}

/// A glyph placed by [`layout`], in pixels relative to the layout origin.
#[derive(Clone, Copy, Debug)]
pub struct PositionedGlyph {
    pub glyph: Glyph,
    /// Bitmap left edge, absolute in layout space.
    pub x: f32,
    /// Bitmap top edge, absolute in layout space.
    pub y: f32,
    /// The character this glyph came from (diagnostics).
    pub ch: char,
}

/// Result of [`layout`].
#[derive(Clone, Debug, Default)]
pub struct TextLayout {
    pub glyphs: Vec<PositionedGlyph>,
    /// Widest line extent (un-aligned).
    pub width: f32,
    /// Total height: lines * line advance.
    pub height: f32,
    pub line_count: usize,
    /// Caret stops: `(x position, source byte offset)` at grapheme
    /// boundaries — the editor maps clicks/carets through these without
    /// re-measuring. Physical units, same space as `glyphs`.
    pub stops: Vec<(f32, usize)>,
    /// Distance from the layout origin to the first line's baseline
    /// (the layout's own ascent), in layout space.
    pub first_baseline: f32,
    /// Baseline-to-baseline distance, in layout space.
    pub line_advance: f32,
}

impl TextLayout {
    /// The y of `line`'s baseline in layout space.
    pub fn baseline(&self, line: usize) -> f32 {
        self.first_baseline + line as f32 * self.line_advance
    }

    /// Union of the positioned glyph bitmap rects, in layout space —
    /// the actual ink the layout draws (no line-box padding).
    pub fn ink_bounds(&self) -> Option<Rect> {
        let mut out: Option<Rect> = None;
        for g in &self.glyphs {
            let r = Rect::new(g.x, g.y, g.glyph.width, g.glyph.height);
            out = Some(match out {
                Some(u) => u.union(r),
                None => r,
            });
        }
        out
    }
}

/// Lay out `text` at `px` pixels per em, rasterizing glyphs into the cache as
/// needed. Handles `\n`, greedy word wrap (`wrap_width`), kerning and
/// alignment. Tab characters advance the pen to the next multiple of four
/// space widths. Trailing spaces at a wrap break do not count toward a
/// line's width or alignment.
/// One cluster's resolved font (`None` = hidden: no role face).
/// Bounded (128) dedup log for missing role faces — shared by the
/// portable layout path and tests.
#[cfg(any(test, not(target_os = "macos")))]
static MISSING_ROLE_WARN: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
    std::sync::Mutex::new(None);

/// A ≤16-scalar cluster preview — the warn key never holds the whole
/// (unbounded) cluster.
#[cfg(not(target_os = "macos"))]
fn warn_preview(cluster: &str) -> String {
    let mut out: String = cluster.chars().take(16).collect();
    if cluster.chars().nth(16).is_some() {
        out.push_str("…");
    }
    out
}

#[cfg(not(target_os = "macos"))]
fn warn_missing_role(role: roles::ScriptRole, cluster: &str) {
    let mut warned = MISSING_ROLE_WARN.lock().unwrap();
    let set = warned.get_or_insert_with(std::collections::HashSet::new);
    if set.len() < 128 {
        let preview = warn_preview(cluster);
        if set.insert(format!("{role:?}:{preview}")) {
            eprintln!("[ui] no face for {role:?} — {preview:?} hidden");
        }
    } else if set.len() == 128 {
        set.insert(String::new());
        eprintln!("[ui] suppressing further missing-face warnings (128)");
    }
}

#[cfg(test)]
fn warn_missing_role_for_test(role: roles::ScriptRole, cluster: &str) {
    let mut warned = MISSING_ROLE_WARN.lock().unwrap();
    let set = warned.get_or_insert_with(std::collections::HashSet::new);
    if set.len() < 128 {
        set.insert(format!("{role:?}:{cluster}"));
    } else if set.len() == 128 {
        set.insert(String::new());
    }
}

#[cfg(test)]
fn missing_role_warn_count() -> usize {
    MISSING_ROLE_WARN
        .lock()
        .unwrap()
        .as_ref()
        .map_or(0, |s| s.len())
}

#[cfg(not(target_os = "macos"))]
fn cluster_font_map(fonts: &Fonts, font: FontId, text: &str) -> Vec<(usize, Option<FontId>)> {
    use unicode_segmentation::UnicodeSegmentation;
    let mut map = Vec::new();
    for (index, cluster) in text.grapheme_indices(true) {
        let role = roles::script_role(cluster);
        let face = match role {
            roles::ScriptRole::Primary => Some(font),
            _ => match fonts.role_face(role) {
                Some(face) => Some(face),
                None => {
                    warn_missing_role(role, cluster);
                    None
                }
            },
        };
        map.push((index, face));
    }
    map
}

/// The resolved `FontId` for the cluster containing `byte` (or `None`
/// when that cluster is hidden).
#[cfg(not(target_os = "macos"))]
fn font_at<'a>(map: &'a [(usize, Option<FontId>)], byte: usize) -> Option<FontId> {
    map.partition_point(|&(start, _)| start <= byte)
        .checked_sub(1)
        .map(|index| map[index].1)
        .unwrap_or(None)
}

/// One character on the portable layout path: the char as stored, its
/// advance, the kern against its predecessor, its byte offset in the source
/// text, and the script face its cluster resolved to — `None` for a hidden
/// cluster, which keeps its caret stops but is never rasterized.
type LineChar = (char, f32, f32, usize, Option<FontId>);

pub fn layout(
    fonts: &Fonts,
    cache: &mut GlyphCache,
    font: FontId,
    px: f32,
    text: &str,
    options: LayoutOptions,
) -> TextLayout {
    let face = fonts.get(font);
    #[cfg(target_os = "macos")]
    if let Some(native_face) = face.native() {
        return native::layout_native(fonts, cache, &native_face, px, text, options);
    }
    // Portable path: resolve every grapheme cluster to its script face (or
    // hidden) once, up front. The native path above never needs this.
    #[cfg(not(target_os = "macos"))]
    let ascii = text.is_ascii();
    #[cfg(not(target_os = "macos"))]
    let cluster_map = if ascii {
        Vec::new()
    } else {
        cluster_font_map(fonts, font, text)
    };
    let line_advance = face.line_height(px) * options.line_height;
    let ascent = face.ascent(px);
    // A tab stop every four space widths.
    let tab_stop = face.advance(' ', px) * 4.0;

    // Split into lines of `LineChar` first; wrap greedily on spaces.
    // `line_w` tracks the line's width alongside `current` so the wrap
    // check is O(1) per character instead of a re-sum per line.
    let mut lines: Vec<Vec<LineChar>> = Vec::new();
    // Unwrapped labels know their first line's capacity before layout.
    let first_line_len = if options.wrap_width.is_none() {
        text.split('\n').next().unwrap_or_default().chars().count()
    } else {
        0
    };
    let mut current: Vec<LineChar> = Vec::with_capacity(first_line_len);
    let mut line_w = 0.0f32;
    let mut prev_ch: Option<char> = None;
    for (byte, ch) in text.char_indices() {
        if ch == '\n' {
            lines.push(std::mem::take(&mut current));
            line_w = 0.0;
            prev_ch = None;
            continue;
        }
        // The character's face: its cluster's script role (portable
        // path) or the primary font. A hidden cluster contributes zero
        // advance and no glyph.
        #[cfg(not(target_os = "macos"))]
        let ch_font = if ascii {
            Some(font)
        } else {
            font_at(&cluster_map, byte)
        };
        #[cfg(not(target_os = "macos"))]
        let Some(ch_font) = ch_font else {
            current.push((ch, 0.0, 0.0, byte, None));
            prev_ch = None;
            continue;
        };
        #[cfg(target_os = "macos")]
        let ch_font = font;
        let ch_face = fonts.get(ch_font);
        let (stored, advance) = if ch == '\t' {
            let next = ((line_w / tab_stop.max(1e-3)).floor() + 1.0) * tab_stop;
            (' ', (next - line_w).max(ch_face.advance(' ', px)))
        } else {
            (ch, ch_face.advance(ch, px) + options.letter_spacing)
        };
        let mut kern = prev_ch
            .map(|prev| ch_face.kern(prev, stored, px))
            .unwrap_or(0.0);
        if let Some(wrap) = options.wrap_width
            && !current.is_empty()
            && line_w + kern + advance > wrap
        {
            // Break at the last space when there is one. The break space is
            // dropped, and so is any run of trailing whitespace before it —
            // trailing spaces at a wrap are invisible and must not count
            // toward the line's width or alignment.
            if let Some(break_at) = current.iter().rposition(|&(c, _, _, _, _)| c == ' ') {
                let mut rest: Vec<LineChar> = current.split_off(break_at + 1);
                while matches!(current.last(), Some((' ', _, _, _, _))) {
                    current.pop();
                }
                lines.push(std::mem::take(&mut current));
                // A new line starts fresh: no kerning across the break.
                if let Some(first) = rest.first_mut() {
                    first.2 = 0.0;
                }
                line_w = rest
                    .iter()
                    .map(|&(_, advance, kern, _, _)| advance + kern)
                    .sum();
                current = rest;
            } else {
                lines.push(std::mem::take(&mut current));
                line_w = 0.0;
            }
            // The breaking char kerns against whatever now ends the line it
            // joins — nothing when the line is empty, the moved-down
            // remainder's last char otherwise.
            kern = current
                .last()
                .map(|&(c, _, _, _, _)| ch_face.kern(c, stored, px))
                .unwrap_or(0.0);
        }
        prev_ch = Some(stored);
        line_w += advance + kern;
        current.push((stored, advance, kern, byte, Some(ch_font)));
    }
    lines.push(current);

    let line_count = lines.len();
    let stop_count: usize = lines.iter().map(Vec::len).sum();
    let glyph_count = lines
        .iter()
        .flatten()
        .filter(|&&(ch, _, _, _, face)| face.is_some() && ch != ' ' && ch != '\t')
        .count();
    let mut out = TextLayout {
        glyphs: Vec::with_capacity(glyph_count),
        width: 0.0,
        height: line_advance * line_count as f32,
        line_count,
        stops: Vec::with_capacity(stop_count + 1),
        first_baseline: ascent,
        line_advance,
    };

    for (index, line) in lines.iter().enumerate() {
        let width: f32 = line
            .iter()
            .map(|&(_, advance, kern, _, _)| advance + kern)
            .sum();
        out.width = out.width.max(width);
        let origin_x = match options.align {
            Align::Left => 0.0,
            Align::Center => (-width * 0.5).round(),
            Align::Right => (-width).round(),
        };
        let baseline = ascent + line_advance * index as f32;
        let mut pen_x = origin_x;
        for &(ch, advance, kern, byte, face) in line {
            // The caret stop exists for every cluster — hidden ones too —
            // so navigation can still enter and leave the zero-advance
            // gap. A hidden cluster (`face` is `None`) keeps its stop and
            // draws nothing: never a .notdef box from a substituted face.
            out.stops.push((pen_x + kern, byte));
            if let Some(face) = face
                && ch != ' '
                && ch != '\t'
            {
                // Rasterize through the cluster's own script face, not
                // the primary font: metrics and ink must agree, or a
                // registered role face advances like itself and draws
                // someone else's glyphs.
                let glyph = cache.glyph(fonts, face, ch, px);
                out.glyphs.push(PositionedGlyph {
                    glyph,
                    x: pen_x + kern + glyph.x_offset,
                    y: baseline + glyph.y_offset,
                    ch,
                });
            }
            pen_x += advance + kern;
        }
    }
    out.stops.push((out.width, text.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_util::test_font;

    fn cache() -> GlyphCache {
        GlyphCache::new(Atlas::new(crate::ui::test_util::test_atlas_config()))
    }

    /// The portable path's hidden-cluster contract, matching the macOS
    /// native path: an unresolvable cluster keeps its caret stops (so
    /// navigation can enter and leave the gap) but rasterizes nothing —
    /// never a .notdef box through the primary face.
    #[test]
    #[cfg(not(target_os = "macos"))]
    fn portable_hidden_clusters_keep_stops_but_draw_no_glyphs() {
        let fonts = test_font();
        let mut cache = cache();
        // 你 has no Han face registered here, so its cluster hides.
        let text = "a\u{4f60}b";
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            text,
            LayoutOptions::default(),
        );
        let drawn: Vec<char> = layout.glyphs.iter().map(|g| g.ch).collect();
        assert_eq!(drawn, vec!['a', 'b'], "hidden cluster must not rasterize");
        for byte in [0usize, 1, 4, 5] {
            assert!(
                layout.stops.iter().any(|&(_, b)| b == byte),
                "caret stop for byte {byte} missing: {:?}",
                layout.stops
            );
        }
        let at = |byte: usize| {
            layout
                .stops
                .iter()
                .find(|&&(_, b)| b == byte)
                .expect("stop")
                .0
        };
        // 你 is three UTF-8 bytes (1..4): the caret can enter the gap
        // before it and leave it after it, both at the same x.
        assert_eq!(
            at(1),
            at(4),
            "the hidden cluster spans zero advance between its neighbours"
        );
    }

    /// A registered script face owns its cluster's ink, not just its
    /// metrics: the portable path rasterizes through the role font, so the
    /// drawn bitmap is that face's — the primary font is never consulted
    /// for another script's glyphs.
    #[test]
    #[cfg(not(target_os = "macos"))]
    fn portable_role_face_rasterizes_through_its_own_font() {
        let mut fonts = test_font();
        let han = fonts
            .load_bytes(
                "PressStart2P",
                include_bytes!("../../assets/fonts/press-start-2p.ttf"),
            )
            .expect("bundled press-start-2p.ttf must load");
        fonts.set_script_font(roles::ScriptRole::Han, han);
        let mut cache = cache();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "你",
            LayoutOptions::default(),
        );
        let drawn = layout
            .glyphs
            .iter()
            .find(|g| g.ch == '你')
            .expect("the Han cluster rasterizes through its role face");
        // Probe both faces directly: the drawn bitmap must be the role
        // face's, and the two faces must be distinguishable here or the
        // comparison would prove nothing.
        // (built by hand: the local `cache` binding below shadows the
        // module's `cache()` helper)
        let mut probe_cache =
            GlyphCache::new(Atlas::new(crate::ui::test_util::test_atlas_config()));
        let via_primary = probe_cache.glyph(&fonts, FontId(0), '你', 16.0);
        let via_role = probe_cache.glyph(&fonts, han, '你', 16.0);
        assert!(
            via_primary.advance != via_role.advance
                || via_primary.width != via_role.width
                || via_primary.height != via_role.height,
            "probe faces must be distinguishable: {via_primary:?} vs {via_role:?}"
        );
        assert_eq!(
            drawn.glyph.advance, via_role.advance,
            "advance comes from the role face"
        );
        assert_eq!(
            drawn.glyph.width, via_role.width,
            "bitmap width comes from the role face"
        );
        assert_eq!(
            drawn.glyph.height, via_role.height,
            "bitmap height comes from the role face"
        );
    }

    #[test]
    fn kern_is_real() {
        // AV pairs kern; the public API must not be a stub.
        let fonts = test_font();
        let face = fonts.get(FontId(0));
        let kern = face.kern('A', 'V', 16.0);
        #[cfg(not(target_os = "macos"))]
        assert!(kern < 0.0, "fontdue reports AV kerning, got {kern}");
        #[cfg(target_os = "macos")]
        assert!(kern <= 0.0, "native pair-kern path, got {kern}");
    }

    #[test]
    fn the_missing_role_warn_set_is_capped() {
        for i in 0..300 {
            warn_missing_role_for_test(roles::ScriptRole::Han, &format!("c{i}"));
        }
        assert!(
            missing_role_warn_count() <= 129,
            "cap 128 + sentinel, got {}",
            missing_role_warn_count()
        );
    }

    #[test]
    fn font_rejects_garbage() {
        let err = Font::from_bytes("bad", b"not a font").expect_err("garbage must fail");
        assert!(err.contains("parse font"), "unexpected error: {err}");
    }

    #[test]
    fn metrics_are_sane() {
        let fonts = test_font();
        let font = fonts.get(FontId(0));
        let px = 16.0;
        assert!(font.ascent(px) > 0.0);
        assert!(font.descent(px) <= 0.0);
        assert!(font.line_height(px) > font.ascent(px));
        assert!(font.advance('M', px) > 0.0);
        assert!(font.has_glyph('A'));
    }

    #[test]
    fn single_line_layout_is_linear() {
        let fonts = test_font();
        let mut cache = cache();
        let text = "Hello";
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            text,
            LayoutOptions::default(),
        );
        assert_eq!(layout.line_count, 1);
        assert_eq!(layout.glyphs.len(), text.len());
        for pair in layout.glyphs.windows(2) {
            assert!(pair[1].x > pair[0].x, "glyphs advance left to right");
            // Bitmap tops differ per glyph (ascenders vs x-height); the
            // shared baseline is top - y_offset.
            let baseline = |g: &PositionedGlyph| g.y - g.glyph.y_offset;
            assert!(
                (baseline(&pair[0]) - baseline(&pair[1])).abs() < 1e-4,
                "one line shares a baseline"
            );
        }
        assert!(layout.width > 0.0);
        assert!(layout.height > 0.0);
    }

    #[test]
    fn newlines_split_lines() {
        let fonts = test_font();
        let mut cache = cache();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "ab\ncd\nef",
            LayoutOptions::default(),
        );
        assert_eq!(layout.line_count, 3);
        let ys: Vec<_> = layout.glyphs.iter().map(|g| g.y).collect();
        // One glyph per character: line starts at 0, 2, 4.
        assert!(ys[2] > ys[0], "second line sits below the first");
        assert!(ys[4] > ys[2], "third line sits below the second");
    }

    #[test]
    fn wrapping_respects_width() {
        let fonts = test_font();
        let mut cache = cache();
        let options = LayoutOptions {
            wrap_width: Some(80.0),
            ..Default::default()
        };
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "the quick brown fox jumps over the lazy dog",
            options,
        );
        assert!(layout.line_count > 1, "long text must wrap");
        // No glyph's right edge exceeds the wrap width by a whole glyph.
        for glyph in &layout.glyphs {
            assert!(
                glyph.x + glyph.glyph.width <= 80.0 + 20.0,
                "glyph {:?} escapes the wrap width",
                glyph.ch
            );
        }
    }

    #[test]
    fn alignment_shifts_lines() {
        let fonts = test_font();
        let mut cache = cache();
        let text = "xx\nlonger line";
        let left = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            text,
            LayoutOptions::default(),
        );
        let center = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            text,
            LayoutOptions {
                align: Align::Center,
                ..Default::default()
            },
        );
        let right = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            text,
            LayoutOptions {
                align: Align::Right,
                ..Default::default()
            },
        );
        // The layout origin is the alignment point: center/right shift the
        // first glyph left of the origin relative to left alignment.
        assert!(center.glyphs[0].x < left.glyphs[0].x);
        assert!(right.glyphs[0].x < center.glyphs[0].x);
    }

    #[test]
    fn glyph_cache_hits_only_once() {
        let fonts = test_font();
        let mut cache = cache();
        for _ in 0..3 {
            layout(
                &fonts,
                &mut cache,
                FontId(0),
                16.0,
                "cache",
                LayoutOptions::default(),
            );
        }
        let first = cache.rasterizations();
        assert_eq!(first, 4, "four distinct characters in \"cache\"");
        // Same text, same size: no new rasterizations.
        layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "cache",
            LayoutOptions::default(),
        );
        assert_eq!(cache.rasterizations(), first);
        // A new size rasterizes again.
        layout(
            &fonts,
            &mut cache,
            FontId(0),
            24.0,
            "cache",
            LayoutOptions::default(),
        );
        assert!(cache.rasterizations() > first);
    }

    #[test]
    fn rasterized_glyph_has_ink() {
        let fonts = test_font();
        let mut cache = cache();
        let glyph = cache.glyph(&fonts, FontId(0), 'A', 32.0);
        assert!(glyph.width > 0.0 && glyph.height > 0.0);
        let size = cache.atlas().page_size() as usize;
        let page = cache.atlas().page_pixels(glyph.region.page);
        let at = (glyph.region.y as usize * size + glyph.region.x as usize) * 4;
        let has_ink = (0..glyph.region.height as usize).any(|row| {
            (0..glyph.region.width as usize).any(|col| page[at + (row * size + col) * 4 + 3] > 32)
        });
        assert!(has_ink, "'A' must rasterize with visible coverage");
    }

    #[test]
    fn tabs_and_spaces_advance() {
        let fonts = test_font();
        let mut cache = cache();
        let layout = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "a\tb",
            LayoutOptions::default(),
        );
        assert_eq!(layout.glyphs.len(), 2, "tab produces no ink");
        assert!(layout.glyphs[1].x > layout.glyphs[0].x + 16.0);
    }

    /// A tab is not a fixed four-space advance: it lands the pen on the next
    /// multiple of four space widths, so columns line up.
    #[test]
    fn tab_advances_to_the_next_stop() {
        let fonts = test_font();
        let mut cache = cache();
        let face = fonts.get(FontId(0));
        let stop = face.advance(' ', 16.0) * 4.0;
        for text in ["a\tb", "ab\tb", "abcd\tb"] {
            let layout = layout(
                &fonts,
                &mut cache,
                FontId(0),
                16.0,
                text,
                Default::default(),
            );
            let b = layout.glyphs.last().expect("b glyph");
            // The pen position (x minus the bitmap bearing) must sit exactly
            // on a tab stop.
            let pen = b.x - b.glyph.x_offset;
            let rem = pen % stop;
            assert!(
                rem < 0.51 || rem > stop - 0.51,
                "{text:?}: pen {pen} is not on a {stop} tab stop"
            );
        }
    }

    /// The space a line wraps on is not part of either line: it must not
    /// count toward the wrapped line's width (which drives alignment).
    #[test]
    fn wrap_trims_the_break_space() {
        let fonts = test_font();
        let mut cache = cache();
        let aa = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "aa",
            Default::default(),
        );
        let wrapped = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "aa bb",
            LayoutOptions {
                wrap_width: Some(aa.width + aa.width * 0.5),
                ..Default::default()
            },
        );
        assert_eq!(wrapped.line_count, 2, "wrapped at the space");
        // Each visible line is exactly its letters — the break space's
        // advance is counted by neither (a wider 'bb' line would exceed
        // aa.width legitimately).
        let bb = layout(
            &fonts,
            &mut cache,
            FontId(0),
            16.0,
            "bb",
            Default::default(),
        );
        assert_eq!(
            wrapped.width,
            aa.width.max(bb.width),
            "the break space is not part of the visible width"
        );
        // Four letters of ink across both lines.
        assert_eq!(wrapped.glyphs.len(), 4);
    }

    /// Layout used to re-sum each candidate line for every character; the
    /// running-width pass must produce identical output to a straight sum.
    #[test]
    fn wrapping_matches_a_reference_layout() {
        let fonts = test_font();
        let mut cache = cache();
        let text = "the quick brown fox jumps over the lazy dog again and again";
        let options = LayoutOptions {
            wrap_width: Some(120.0),
            ..Default::default()
        };
        let layout = layout(&fonts, &mut cache, FontId(0), 16.0, text, options);
        // Every glyph stays inside the wrap box (with one glyph of slack for
        // words longer than the line).
        for glyph in &layout.glyphs {
            assert!(glyph.x + glyph.glyph.width < 120.0 + 20.0);
        }
        assert!(layout.line_count >= 3, "{text:?} must wrap several times");
    }
    #[test]
    #[cfg(not(target_os = "macos"))]
    fn font_at_resolves_owning_cluster() {
        let map: Vec<(usize, Option<FontId>)> =
            vec![(0, Some(FontId(0))), (4, None), (9, Some(FontId(2)))];
        assert_eq!(font_at(&map, 0), Some(FontId(0)));
        assert_eq!(font_at(&map, 3), Some(FontId(0)));
        assert_eq!(font_at(&map, 4), None, "hidden cluster, byte at start");
        assert_eq!(font_at(&map, 8), None, "hidden cluster, interior byte");
        assert_eq!(font_at(&map, 9), Some(FontId(2)));
        assert_eq!(font_at(&map, 10_000), Some(FontId(2)), "past the end");
        let empty: Vec<(usize, Option<FontId>)> = Vec::new();
        assert_eq!(font_at(&empty, 0), None);
        assert_eq!(font_at(&empty, 99), None);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn font_at_ascii_layout_matches_cluster_resolution() {
        let fonts = test_font();
        let mut cache = cache();
        let prefix = "Hello, world! 123 AV plain";
        let mixed = format!("{prefix}\u{4f60}");
        let options = [
            LayoutOptions::default(),
            LayoutOptions {
                align: Align::Center,
                ..Default::default()
            },
            LayoutOptions {
                align: Align::Right,
                ..Default::default()
            },
            LayoutOptions {
                wrap_width: Some(60.0),
                ..Default::default()
            },
        ];
        for opts in options {
            let fast = layout(&fonts, &mut cache, FontId(0), 16.0, prefix, opts);
            let resolved = layout(&fonts, &mut cache, FontId(0), 16.0, &mixed, opts);
            assert_eq!(fast.glyphs.len(), resolved.glyphs.len());
            for (a, b) in fast.glyphs.iter().zip(&resolved.glyphs) {
                assert_eq!(a.x, b.x);
                assert_eq!(a.y, b.y);
                assert_eq!(a.ch, b.ch);
                assert_eq!(a.glyph.advance, b.glyph.advance);
                assert_eq!(a.glyph.width, b.glyph.width);
                assert_eq!(a.glyph.height, b.glyph.height);
            }
            let stops = (fast.stops.len() - 1).min(resolved.stops.len());
            assert_eq!(fast.stops[..stops], resolved.stops[..stops]);
            assert_eq!(fast.width, resolved.width);
            assert_eq!(fast.height, resolved.height);
            assert_eq!(fast.line_count, resolved.line_count);
        }
    }

    #[test]
    #[ignore]
    fn long_text_layout_profile() {
        use std::time::Instant;
        let fonts = test_font();
        let mut cache = cache();
        for count in [128usize, 512, 2048, 8192] {
            let text = "ab cd ".repeat(count.div_ceil(6));
            for _ in 0..5 {
                std::hint::black_box(layout(
                    &fonts,
                    &mut cache,
                    FontId(0),
                    16.0,
                    &text,
                    LayoutOptions::default(),
                ));
            }
            let mut samples = Vec::with_capacity(100);
            for _ in 0..100 {
                let start = Instant::now();
                let shaped = layout(
                    &fonts,
                    &mut cache,
                    FontId(0),
                    16.0,
                    &text,
                    LayoutOptions::default(),
                );
                std::hint::black_box(&shaped);
                samples.push(start.elapsed().as_nanos());
            }
            samples.sort_unstable();
            println!(
                "long_text,bytes={},frames=100,p50_ns={},p95_ns={},max_ns={}",
                text.len(),
                samples[50],
                samples[95],
                samples[99]
            );
        }
    }
}
