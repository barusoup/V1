//! Paged shelf-packing texture atlas.
//!
//! Purely CPU-side: pages are native pixel buffers (coverage bytes or RGBA8)
//! that the [`crate::ui::render`] layer uploads to GPU textures. Keeping
//! packing CPU-side means it is unit tested without a GPU, and the upload
//! layer only ever sees dirty rectangles or full-page snapshots.
//!
//! Growth is by *page*: when the current page fills up, a new page is added
//! and existing regions stay valid (they name their page). Pages never mix
//! pixel formats — a coverage page stays coverage. UI batches simply split
//! at page boundaries.
//!
//! Snapshots: [`Atlas::snapshot`] hands the GPU layer an immutable
//! point-in-time copy (shared pixel storage via `Arc`, so a finished frame
//! keeps the pixels it was built against even while the live atlas keeps
//! growing). Writes go through `Arc::make_mut`, so they never disturb a
//! snapshot a retained frame still holds.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique atlas identity: a reset gets a new epoch, a new atlas a new id —
/// either way a GPU uploader knows a snapshot belongs to this atlas.
static NEXT_ATLAS_ID: AtomicU64 = AtomicU64::new(1);

/// A placed rectangle inside a page. UVs are computed against the page size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub page: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    /// UV rectangle `(u0, v0, u1, v1)` for a page of `page_size` pixels.
    pub fn uv(self, page_size: u32) -> [f32; 4] {
        let s = page_size as f32;
        [
            self.x as f32 / s,
            self.y as f32 / s,
            (self.x + self.width) as f32 / s,
            (self.y + self.height) as f32 / s,
        ]
    }
}

/// A dirty rectangle the GPU layer must upload: the pixels of `rect` within
/// the page's buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    pub page: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// The pixel format of a page. Pages never mix formats: an R8 coverage
/// page stays coverage, an RGBA page stays RGBA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasFormat {
    /// One byte per texel (font/icon coverage). On the GPU this is R8_UNORM
    /// swizzled to (1,1,1,a) — white with the coverage as alpha.
    Coverage,
    /// Four bytes per texel, unencoded (linear-light game textures, icons
    /// baked in colour).
    RgbaLinear,
    /// Four bytes per texel, sRGB-encoded.
    RgbaSrgb,
}

impl AtlasFormat {
    /// Bytes per texel in the native page buffer.
    pub fn bytes_per_texel(self) -> usize {
        match self {
            AtlasFormat::Coverage => 1,
            _ => 4,
        }
    }
}

#[derive(Debug)]
struct Page {
    /// Native pixels (`format.bytes_per_texel()` per texel). `Arc` so a
    /// snapshot shares storage instead of copying; `Arc::make_mut` on the
    /// next write protects the frame holding the snapshot.
    pixels: Arc<Vec<u8>>,
    format: AtlasFormat,
    /// Bumped once per *snapshot-sealed* batch of writes; a GPU uploader
    /// keeps the last uploaded revision per page and skips unchanged ones.
    revision: u64,
    /// The dirty union the last snapshot sealed (what a consecutive
    /// upload copies). Retained — an unchanged page keeps reporting it.
    last_dirty: Option<DirtyRect>,
    /// Union of writes since the last snapshot — sealed into
    /// `last_dirty` (and one revision tick) by the next snapshot.
    pending_dirty: Option<DirtyRect>,
    /// Union since the last `take_dirty` — the legacy path's record,
    /// independent of the snapshot one.
    legacy_dirty: Option<DirtyRect>,
    /// Lazily expanded RGBA copy for the legacy `page_pixels` API —
    /// invalidated on every write.
    rgba_expansion: std::sync::OnceLock<Vec<u8>>,
    shelf_x: u32,
    shelf_y: u32,
    shelf_height: u32,
}

impl Page {
    fn new(size: u32, format: AtlasFormat) -> Self {
        let texels = size as usize * size as usize;
        Self {
            pixels: Arc::new(vec![0; texels * format.bytes_per_texel()]),
            format,
            revision: 0,
            last_dirty: None,
            pending_dirty: None,
            legacy_dirty: None,
            rgba_expansion: std::sync::OnceLock::new(),
            shelf_x: 0,
            shelf_y: 0,
            shelf_height: 0,
        }
    }

    fn mark_dirty(&mut self, x: u32, y: u32, width: u32, height: u32) {
        let rect = DirtyRect {
            page: 0, // fixed up by the atlas
            x,
            y,
            width,
            height,
        };
        let union = |slot: &mut Option<DirtyRect>| {
            *slot = Some(match slot.take() {
                Some(existing) => union_dirty(existing, rect),
                None => rect,
            });
        };
        union(&mut self.pending_dirty);
        union(&mut self.legacy_dirty);
        // Any expansion built before this write is stale.
        self.rgba_expansion = std::sync::OnceLock::new();
    }
}

fn union_dirty(a: DirtyRect, b: DirtyRect) -> DirtyRect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.width).max(b.x + b.width);
    let y1 = (a.y + a.height).max(b.y + b.height);
    DirtyRect {
        page: a.page,
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

/// Configuration for [`Atlas::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasConfig {
    /// Edge length of each square page, in pixels. 1024 keeps one page's
    /// CPU payload small while covering full UI sessions.
    pub page_size: u32,
    /// Transparent padding between placements, in pixels. Keeps linear
    /// filtering from bleeding between neighbouring glyphs.
    pub padding: u32,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            page_size: 1024,
            padding: 1,
        }
    }
}

/// One page in an [`AtlasSnapshot`]: shared native pixels plus the
/// revision and dirty union a GPU uploader diffs against its own record.
#[derive(Clone, Debug)]
pub struct SnapshotPage {
    pub format: AtlasFormat,
    /// Monotonic per-page write counter.
    pub revision: u64,
    /// What changed since the last snapshot — `None` means "everything"
    /// (a fresh page, or one whose dirty union was never tracked).
    pub last_dirty: Option<DirtyRect>,
    /// Native pixel buffer (`page_size² × format.bytes_per_texel()`).
    pub pixels: Arc<Vec<u8>>,
}

/// Immutable point-in-time atlas contents. Retained by the frame that
/// produced it, so pixels cannot shift under an in-flight GPU upload.
#[derive(Clone, Debug)]
pub struct AtlasSnapshot {
    /// Which atlas produced this snapshot.
    pub atlas_id: u64,
    /// Which reset generation of that atlas.
    pub epoch: u64,
    pub page_size: u32,
    pub pages: Vec<SnapshotPage>,
}

/// What the GPU layer should do with one snapshot page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageUpload {
    /// Nothing changed — the GPU copy already holds this revision.
    Skip,
    /// Upload only `last_dirty` — the previous upload was the immediately
    /// preceding revision of this page.
    Dirty,
    /// Upload the whole page (new page, or a non-consecutive revision —
    /// an older frame arrived late or the atlas reset in between).
    Full,
}

/// Diff a renderer's memory against a snapshot: which pages exist, and
/// whether each needs a full or dirty upload. `previous` maps page index
/// to the last uploaded revision for the same `atlas_id`/`epoch`; any
/// other identity is a different atlas entirely (every page full).
///
/// This is the pure seam — [`crate::ui::render::UiRenderer`] drives the
/// actual transfers from it.
pub fn upload_plan(
    previous: Option<(u64, u64, &RevisionMap)>,
    snapshot: &AtlasSnapshot,
) -> Vec<PageUpload> {
    let matches = previous
        .as_ref()
        .is_some_and(|(id, epoch, _)| *id == snapshot.atlas_id && *epoch == snapshot.epoch);
    snapshot
        .pages
        .iter()
        .enumerate()
        .map(|(index, page)| {
            if !matches {
                return PageUpload::Full;
            }
            let (_, _, revisions) = previous.expect("matches");
            match revisions.get(&index) {
                Some(&rev) if rev == page.revision => PageUpload::Skip,
                // Consecutive revision: only the dirty union is missing.
                Some(&rev) if rev + 1 == page.revision && page.last_dirty.is_some() => {
                    PageUpload::Dirty
                }
                _ => PageUpload::Full,
            }
        })
        .collect()
}

/// Page index -> last uploaded revision: the renderer-side record
/// [`upload_plan`] diffs a snapshot against.
pub type RevisionMap = std::collections::HashMap<usize, u64>;

/// Growable atlas: shelf packing (first-fit rows), pages on demand.
#[derive(Debug)]
pub struct Atlas {
    config: AtlasConfig,
    /// CPU byte cap on native page payload (default 8 MiB;
    /// [`Atlas::set_byte_budget`]). Admission-side: `insert` fails with
    /// [`AtlasError::BudgetExceeded`] rather than growing past it.
    byte_budget: usize,
    id: u64,
    /// Reset generation: a new epoch invalidates every retained snapshot's
    /// identity check, so the GPU layer does a clean re-upload.
    epoch: u64,
    pages: Vec<Page>,
    /// Total placements accepted (glyphs, icons) — cheap growth diagnostics.
    placements: usize,
    /// Cached snapshot, invalidated on the next write.
    snapshot: Option<Arc<AtlasSnapshot>>,
}

impl Atlas {
    /// Default native-payload budget (8 MiB).
    const DEFAULT_BYTE_BUDGET: usize = 8 * 1024 * 1024;

    pub fn new(config: AtlasConfig) -> Self {
        assert!(
            config.page_size >= 8 && config.page_size <= 4096,
            "page size {} out of range 8..=4096",
            config.page_size
        );
        Self {
            config,
            byte_budget: Self::DEFAULT_BYTE_BUDGET
                .max(config.page_size as usize * config.page_size as usize),
            id: NEXT_ATLAS_ID.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            pages: vec![Page::new(config.page_size, AtlasFormat::Coverage)],
            placements: 0,
            snapshot: None,
        }
    }

    /// Throw every page away and start the next epoch. Retained snapshots
    /// keep their pixels (they share `Arc` storage); the GPU layer sees a
    /// new `atlas_id`/`epoch` and re-uploads everything.
    pub fn reset(&mut self) {
        self.epoch += 1;
        self.pages.clear();
        self.pages
            .push(Page::new(self.config.page_size, AtlasFormat::Coverage));
        self.snapshot = None;
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn config(&self) -> AtlasConfig {
        self.config
    }

    /// New CPU admission budget (native payload bytes). Values below one
    /// page are clamped up to a single coverage page.
    pub fn set_byte_budget(&mut self, bytes: usize) {
        self.byte_budget =
            bytes.max(self.config.page_size as usize * self.config.page_size as usize);
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Transparent padding between placements, in pixels.
    pub fn padding(&self) -> u32 {
        self.config.padding
    }

    pub fn page_size(&self) -> u32 {
        self.config.page_size
    }

    pub fn placements(&self) -> usize {
        self.placements
    }

    /// Insert RGBA8 linear pixels (`width * height * 4` bytes).
    pub fn insert_rgba(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Region, AtlasError> {
        self.insert_with(pixels, width, height, AtlasFormat::RgbaLinear)
    }

    /// Insert sRGB-encoded RGBA8 pixels.
    pub fn insert_color_srgb(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Region, AtlasError> {
        self.insert_with(pixels, width, height, AtlasFormat::RgbaSrgb)
    }

    /// Insert 8-bit coverage pixels (`width * height` bytes) — the glyph
    /// path. Stored natively: one byte per texel.
    pub fn insert_coverage(
        &mut self,
        coverage: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Region, AtlasError> {
        self.insert_with(coverage, width, height, AtlasFormat::Coverage)
    }

    fn insert_with(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        format: AtlasFormat,
    ) -> Result<Region, AtlasError> {
        let bpp = format.bytes_per_texel();
        assert_eq!(
            pixels.len(),
            width as usize * height as usize * bpp,
            "pixel payload does not match dimensions"
        );
        let size = self.config.page_size;
        let padded_w = width.saturating_add(self.config.padding * 2);
        let padded_h = height.saturating_add(self.config.padding * 2);
        if padded_w > size || padded_h > size {
            return Err(AtlasError::TooLarge {
                width,
                height,
                page_size: size,
            });
        }
        // Empty placements (whitespace glyphs) still need a valid region; a
        // 0x0 region at the current shelf corner costs nothing and samples
        // transparent padding.
        if width == 0 || height == 0 {
            // A zero-sized glyph never samples; any page's coordinates do.
            let page = self.pages.len() - 1;
            let corner = (self.pages[page].shelf_x, self.pages[page].shelf_y);
            return Ok(Region {
                page,
                x: corner.0,
                y: corner.1,
                width: 0,
                height: 0,
            });
        }

        let pad = self.config.padding;
        let mut page_index = match self.pages.iter().rposition(|page| page.format == format) {
            Some(index) => index,
            None => {
                // First insertion of this format: its own page.
                self.grow_page(format, size)?;
                self.pages.len() - 1
            }
        };
        loop {
            {
                let page = &mut self.pages[page_index];
                // New shelf when the row has no room left.
                if page.shelf_x + width + pad > size {
                    page.shelf_x = 0;
                    page.shelf_y += page.shelf_height + pad;
                    page.shelf_height = 0;
                }
                if page.shelf_y + height + pad <= size {
                    let x = page.shelf_x + pad;
                    let y = page.shelf_y;
                    page.shelf_height = page.shelf_height.max(height + pad);
                    page.shelf_x += width + pad;
                    let row_bytes = size as usize * bpp;
                    let offset = y as usize * row_bytes + x as usize * bpp;
                    // Drop the atlas's own cached snapshot first so
                    // `make_mut` only copies when an *external* frame still
                    // holds one — otherwise this write is in place.
                    self.snapshot = None;
                    let pixels_arc = Arc::make_mut(&mut page.pixels);
                    for row in 0..height as usize {
                        let dst = offset + row * row_bytes;
                        let src = row * width as usize * bpp;
                        pixels_arc[dst..dst + width as usize * bpp]
                            .copy_from_slice(&pixels[src..src + width as usize * bpp]);
                    }
                    page.mark_dirty(x, y, width, height);
                    self.placements += 1;
                    return Ok(Region {
                        page: page_index,
                        x,
                        y,
                        width,
                        height,
                    });
                }
            }
            // No room: next same-format page, or grow one.
            match self.pages[page_index + 1..]
                .iter()
                .position(|page| page.format == format)
            {
                Some(ahead) => page_index += ahead + 1,
                None => {
                    self.grow_page(format, size)?;
                    page_index = self.pages.len() - 1;
                }
            }
        }
    }

    /// Append a page of `format` (budget-checked).
    fn grow_page(&mut self, format: AtlasFormat, size: u32) -> Result<(), AtlasError> {
        let bpp = format.bytes_per_texel();
        if self.native_bytes() + (size as usize * size as usize * bpp) > self.byte_budget {
            return Err(AtlasError::BudgetExceeded {
                budget: self.byte_budget,
            });
        }
        self.pages.push(Page::new(size, format));
        self.snapshot = None;
        Ok(())
    }

    /// Copy a placed region's RGBA pixels out as tightly packed rows (used
    /// by tests and CPU consumers; the GPU layer reads the page buffers
    /// directly). Coverage pages expand to white-with-alpha.
    pub fn region_pixels(&self, region: Region) -> Vec<u8> {
        self.copy_rows_rgba(region.page, region.x, region.y, region.width, region.height)
    }

    /// Full page buffer as RGBA — a lazily expanded view for coverage
    /// pages (the GPU layer uses `native_pixels` instead). Legacy API:
    /// callers that need RGBA get it; the atlas keeps storing coverage.
    pub fn page_pixels(&self, page_index: usize) -> &[u8] {
        let page = &self.pages[page_index];
        if page.format == AtlasFormat::Coverage {
            page.rgba_expansion
                .get_or_init(|| self.expand_coverage(page_index))
        } else {
            &page.pixels
        }
    }

    /// Expand a coverage page to white-with-alpha RGBA rows.
    fn expand_coverage(&self, page_index: usize) -> Vec<u8> {
        let page = &self.pages[page_index];
        let mut out = Vec::with_capacity(page.pixels.len() * 4);
        for &v in page.pixels.iter() {
            out.extend_from_slice(&[255, 255, 255, v]);
        }
        out
    }

    /// The native (unexpanded) page buffer — the GPU upload source.
    pub fn native_pixels(&self, page: usize) -> &[u8] {
        &self.pages[page].pixels
    }

    /// A page's pixel format and write revision (GPU upload decisions).
    pub fn page_format(&self, page: usize) -> AtlasFormat {
        self.pages[page].format
    }

    pub fn page_revision(&self, page: usize) -> u64 {
        self.pages[page].revision
    }

    /// Take the pending upload rect for `page`, clearing it. The GPU layer
    /// calls this once per frame per page (legacy path — new code uses
    /// [`Atlas::snapshot`] + [`upload_plan`]).
    pub fn take_dirty(&mut self, page: usize) -> Option<DirtyRect> {
        let dirty = self.pages[page].legacy_dirty.take();
        dirty.map(|mut rect| {
            rect.page = page;
            rect
        })
    }

    /// Copy a dirty rectangle out as tightly packed RGBA rows (legacy path).
    pub fn dirty_pixels(&self, rect: DirtyRect) -> Vec<u8> {
        self.copy_rows_rgba(rect.page, rect.x, rect.y, rect.width, rect.height)
    }

    /// Native rows of `rect` in the page's own format (the GPU upload path:
    /// tightly packed rows for `vkCmdCopyBufferToImage`).
    pub fn native_rows(&self, rect: DirtyRect) -> Vec<u8> {
        let page = &self.pages[rect.page];
        let bpp = page.format.bytes_per_texel();
        let row_bytes = self.config.page_size as usize * bpp;
        let mut out = Vec::with_capacity(rect.width as usize * rect.height as usize * bpp);
        for row in 0..rect.height as usize {
            let start = (rect.y as usize + row) * row_bytes + rect.x as usize * bpp;
            out.extend_from_slice(&page.pixels[start..start + rect.width as usize * bpp]);
        }
        out
    }

    /// Copy a rectangle of one page out as tightly packed RGBA rows.
    fn copy_rows_rgba(
        &self,
        page_index: usize,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let page = &self.pages[page_index];
        let size = self.config.page_size as usize;
        match page.format {
            AtlasFormat::Coverage => {
                let mut out = Vec::with_capacity(width as usize * height as usize * 4);
                for row in 0..height as usize {
                    let start = (y as usize + row) * size + x as usize;
                    for &v in &page.pixels[start..start + width as usize] {
                        out.extend_from_slice(&[255, 255, 255, v]);
                    }
                }
                out
            }
            _ => {
                let mut out = Vec::with_capacity(width as usize * height as usize * 4);
                for row in 0..height as usize {
                    let start = ((y as usize + row) * size + x as usize) * 4;
                    out.extend_from_slice(&page.pixels[start..start + width as usize * 4]);
                }
                out
            }
        }
    }

    /// Total CPU footprint of the native page buffers.
    pub fn cpu_bytes(&self) -> usize {
        self.native_bytes()
    }

    /// Native pixel payload across all pages (the budget's subject).
    pub fn native_bytes(&self) -> usize {
        self.pages.iter().map(|page| page.pixels.len()).sum()
    }

    /// Immutable snapshot of the atlas for one frame's upload — cheap
    /// (shared pixel storage), and invalidated on the next write.
    pub fn snapshot(&mut self) -> Arc<AtlasSnapshot> {
        if self.snapshot.is_none() {
            // Seal pending writes: one revision tick per *batch*, so ten
            // glyph inserts between frames are one upload, not ten.
            for page in &mut self.pages {
                if page.pending_dirty.is_some() {
                    page.last_dirty = page.pending_dirty.take();
                    page.revision += 1;
                }
            }
            self.snapshot = Some(Arc::new(AtlasSnapshot {
                atlas_id: self.id,
                epoch: self.epoch,
                page_size: self.config.page_size,
                pages: self
                    .pages
                    .iter()
                    .map(|page| SnapshotPage {
                        format: page.format,
                        revision: page.revision,
                        last_dirty: page.last_dirty,
                        pixels: Arc::clone(&page.pixels),
                    })
                    .collect(),
            }));
        }
        Arc::clone(self.snapshot.as_ref().expect("built above"))
    }
}

/// Placement failure for [`Atlas::insert_rgba`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasError {
    /// The image cannot fit in one page even with an empty atlas.
    TooLarge {
        width: u32,
        height: u32,
        page_size: u32,
    },
    /// A new page would push the atlas's native payload past its budget.
    BudgetExceeded { budget: usize },
}

impl std::fmt::Display for AtlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtlasError::TooLarge {
                width,
                height,
                page_size,
            } => write!(
                f,
                "{width}x{height} does not fit in a {page_size}x{page_size} atlas page"
            ),
            AtlasError::BudgetExceeded { budget } => {
                write!(f, "atlas byte budget {budget} exceeded")
            }
        }
    }
}

impl std::error::Error for AtlasError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Atlas {
        Atlas::new(AtlasConfig {
            page_size: 64,
            padding: 1,
        })
    }

    #[test]
    fn inserts_pack_left_to_right() {
        let mut atlas = small();
        let a = atlas.insert_rgba(&[1u8; 4 * 8 * 8], 8, 8).unwrap();
        let b = atlas.insert_rgba(&[2u8; 4 * 8 * 8], 8, 8).unwrap();
        // RGBA gets its own page (page 0 is coverage); two same-format
        // inserts share it.
        assert_eq!(a.page, b.page, "same-format inserts share a page");
        assert!(b.x > a.x, "second placement packs to the right");
        assert_eq!(a.y, b.y, "same shelf");
    }

    #[test]
    fn regions_do_not_overlap() {
        let mut atlas = small();
        let mut placed = Vec::new();
        for i in 0..16u8 {
            let region = atlas
                .insert_rgba(&[i; 4 * 7 * 7], 7, 7)
                .expect("fits across pages");
            for other in &placed {
                let (a, b): (Region, Region) = (*other, region);
                let overlap = a.page == b.page
                    && a.x < b.x + b.width
                    && b.x < a.x + a.width
                    && a.y < b.y + b.height
                    && b.y < a.y + a.height;
                assert!(!overlap, "regions overlap: {a:?} vs {b:?}");
            }
            placed.push(region);
        }
    }

    #[test]
    fn grows_by_page_and_keeps_regions_valid() {
        let mut atlas = small();
        // 64x64 page minus padding: a handful of 20x20 fills one page.
        let first = atlas.insert_rgba(&[1u8; 4 * 20 * 20], 20, 20).unwrap();
        for _ in 0..8 {
            atlas.insert_rgba(&[1u8; 4 * 20 * 20], 20, 20).unwrap();
        }
        assert!(atlas.page_count() > 1, "should have grown");
        // The early region still reads its original pixels.
        let pixels = atlas.region_pixels(first);
        assert_eq!(pixels[0], 1);
    }

    #[test]
    fn rejects_oversized_images() {
        let mut atlas = small();
        let err = atlas
            .insert_rgba(&vec![0u8; 4 * 64 * 8], 64, 8)
            .expect_err("64 wide + padding must not fit");
        assert!(matches!(err, AtlasError::TooLarge { .. }));
    }

    #[test]
    fn coverage_inserts_as_white_with_alpha() {
        let mut atlas = small();
        let region = atlas.insert_coverage(&[0, 128, 255, 64], 2, 2).unwrap();
        let size = atlas.page_size() as usize;
        // Native coverage: single bytes. The RGBA view expands it.
        let native = atlas.native_pixels(region.page);
        assert_eq!(native[region.y as usize * size + region.x as usize], 0);
        let page = atlas.page_pixels(region.page);
        let at = (region.y as usize * size + region.x as usize) * 4;
        assert_eq!(&page[at..at + 4], &[255, 255, 255, 0]);
        assert_eq!(&page[at + 4..at + 8], &[255, 255, 255, 128]);
    }

    #[test]
    fn formats_never_mix_on_a_page() {
        let mut atlas = small();
        let cov = atlas.insert_coverage(&[9u8; 4], 2, 2).unwrap();
        let rgb = atlas.insert_rgba(&[3u8; 4 * 4], 2, 2).unwrap();
        assert_eq!(
            atlas.page_format(cov.page),
            AtlasFormat::Coverage,
            "coverage page stays coverage"
        );
        assert_eq!(
            atlas.page_format(rgb.page),
            AtlasFormat::RgbaLinear,
            "rgba lands on its own page"
        );
        assert_ne!(cov.page, rgb.page);
    }

    #[test]
    fn dirty_rect_tracks_and_copies() {
        let mut atlas = small();
        let region = atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        let dirty = atlas.take_dirty(0).expect("insert marks dirty");
        assert_eq!((dirty.width, dirty.height), (2, 2));
        assert_eq!(dirty.x, region.x);
        assert!(atlas.take_dirty(0).is_none(), "dirty is consumed");
        let pixels = atlas.dirty_pixels(DirtyRect {
            page: 0,
            x: region.x,
            y: region.y,
            width: 2,
            height: 2,
        });
        assert_eq!(pixels.len(), 2 * 2 * 4);
        assert!(pixels.iter().all(|&b| b == 255 || b == 7));
    }

    #[test]
    fn dirty_rects_union_while_untouched() {
        let mut atlas = small();
        atlas.insert_coverage(&[1u8; 4], 2, 2).unwrap();
        atlas.insert_coverage(&[1u8; 4], 2, 2).unwrap();
        let dirty = atlas.take_dirty(0).unwrap();
        // Two inserts on the same shelf: one unioned rect covering both.
        assert!(dirty.width >= 2);
    }

    #[test]
    fn zero_sized_glyphs_place_somewhere_valid() {
        let mut atlas = small();
        let region = atlas.insert_coverage(&[], 0, 0).unwrap();
        assert_eq!(region.width, 0);
        assert!(region.page < atlas.page_count());
    }

    #[test]
    fn budget_admission_fails_instead_of_resetting() {
        // One 64x64 coverage page = 4 KiB; allow exactly that.
        let mut atlas = Atlas::new(AtlasConfig {
            page_size: 64,
            padding: 1,
        });
        atlas.set_byte_budget(64 * 64);
        // Insert until a new page is needed: the first refusal must be
        // the budget, never a panic or silent reset.
        let mut err = None;
        for _ in 0..80 {
            if let Err(e) = atlas.insert_coverage(&[9u8; 8 * 8], 8, 8) {
                err = Some(e);
                break;
            }
        }
        assert!(
            matches!(err, Some(AtlasError::BudgetExceeded { .. })),
            "expected BudgetExceeded, got {err:?}"
        );
    }

    #[test]
    fn snapshots_are_immutable_under_later_writes() {
        let mut atlas = small();
        let r1 = atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        let first = atlas.snapshot();
        // A second write while `first` is held: the page COWs, so the two
        // snapshots hold *different* buffers and the old pixels survive.
        let r2 = atlas.insert_coverage(&[5u8; 4], 2, 2).unwrap();
        let second = atlas.snapshot();
        assert_ne!(
            first.pages[0].revision, second.pages[0].revision,
            "the write batch ticks once"
        );
        assert!(
            !Arc::ptr_eq(&first.pages[0].pixels, &second.pages[0].pixels),
            "retained snapshot owns its own copy"
        );
        let size = 64usize;
        let at1 = r1.y as usize * size + r1.x as usize;
        let at2 = r2.y as usize * size + r2.x as usize;
        assert_eq!(first.pages[0].pixels[at1], 7);
        assert_eq!(first.pages[0].pixels[at2], 0, "old snapshot lacks it");
        assert_eq!(second.pages[0].pixels[at2], 5);
    }

    #[test]
    fn a_write_with_no_snapshot_held_stays_in_place() {
        let mut atlas = small();
        atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        let snap = atlas.snapshot();
        let shared_ptr = Arc::as_ptr(&snap.pages[0].pixels);
        drop(snap);
        // With no external frame holding the buffer, the next write is
        // in place — same allocation, no copy.
        atlas.insert_coverage(&[5u8; 4], 2, 2).unwrap();
        let second = atlas.snapshot();
        assert_eq!(Arc::as_ptr(&second.pages[0].pixels), shared_ptr);
    }

    #[test]
    fn a_write_batch_is_one_revision() {
        let mut atlas = small();
        for _ in 0..10 {
            atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        }
        let snap = atlas.snapshot();
        // Ten inserts since page creation: one sealed batch.
        assert_eq!(snap.pages[0].revision, 1, "batch ticks once");
        let dirty = snap.pages[0].last_dirty.expect("batch is dirty");
        // The union covers only THIS batch, not the padding row above.
        assert!(dirty.width >= 20 && dirty.height >= 2);
        // An unchanged page keeps its revision and dirty union.
        let again = atlas.snapshot();
        assert_eq!(again.pages[0].revision, 1);
        assert_eq!(again.pages[0].last_dirty, Some(dirty));
        // A second batch ticks once more and re-dirties only its writes.
        atlas.insert_coverage(&[3u8; 4], 2, 2).unwrap();
        let third = atlas.snapshot();
        assert_eq!(third.pages[0].revision, 2);
        let d2 = third.pages[0].last_dirty.unwrap();
        assert!(d2.width <= dirty.width, "only the new write");
    }

    #[test]
    fn upload_plan_diffs_by_revision() {
        let mut atlas = small();
        atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        let snap = atlas.snapshot();
        // Never uploaded anything: the page is full.
        let mut previous = std::collections::HashMap::new();
        assert_eq!(
            upload_plan(None, &snap),
            vec![PageUpload::Full],
            "unknown atlas uploads full"
        );
        // The page was uploaded at revision 0 (page creation): the write
        // is consecutive -> dirty.
        previous.insert(0, 0);
        let id = snap.atlas_id;
        let epoch = snap.epoch;
        let plan = upload_plan(Some((id, epoch, &previous)), &snap);
        assert_eq!(plan, vec![PageUpload::Dirty], "consecutive is dirty");
        // Same revision: nothing new.
        previous.insert(0, snap.pages[0].revision);
        let plan = upload_plan(Some((id, epoch, &previous)), &snap);
        assert_eq!(plan, vec![PageUpload::Skip]);
        // A different epoch: full again.
        let plan = upload_plan(Some((id, epoch + 1, &previous)), &snap);
        assert_eq!(plan, vec![PageUpload::Full]);
    }

    #[test]
    fn a_reset_bumps_the_epoch_and_plans_full() {
        let mut atlas = small();
        atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        let old_epoch = atlas.epoch();
        let _old = atlas.snapshot();
        atlas.reset();
        assert_ne!(atlas.epoch(), old_epoch, "reset is a new epoch");
        atlas.insert_coverage(&[3u8; 4], 2, 2).unwrap();
        let snap = atlas.snapshot();
        // A renderer still holding the old identity uploads everything.
        let mut previous = std::collections::HashMap::new();
        previous.insert(0, 1u64);
        let plan = upload_plan(Some((snap.atlas_id, old_epoch, &previous)), &snap);
        assert_eq!(plan, vec![PageUpload::Full]);
    }

    #[test]
    fn a_retained_old_frame_restores_older_pixels() {
        let mut atlas = small();
        atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        let old_snap = atlas.snapshot();
        // More writes — the GPU now holds a newer revision.
        atlas.insert_coverage(&[3u8; 4], 2, 2).unwrap();
        let _new_snap = atlas.snapshot();
        // The renderer uploaded revision 2 already; the retained older
        // frame arrives now: non-consecutive -> full restore of the OLD
        // pixels.
        let mut previous = std::collections::HashMap::new();
        previous.insert(0, 2u64);
        let plan = upload_plan(
            Some((old_snap.atlas_id, old_snap.epoch, &previous)),
            &old_snap,
        );
        assert_eq!(
            plan,
            vec![PageUpload::Full],
            "a late older frame does a full restore"
        );
    }

    #[test]
    fn a_discarded_frame_does_not_lose_the_change() {
        let mut atlas = small();
        atlas.insert_coverage(&[7u8; 4], 2, 2).unwrap();
        let _dropped = atlas.snapshot(); // built, then the frame is dropped
        atlas.insert_coverage(&[5u8; 4], 2, 2).unwrap();
        let next = atlas.snapshot();
        // Renderer saw rev 1 (the dropped frame never arrived): the next
        // snapshot is consecutive and uploads just its dirty union.
        let mut previous = std::collections::HashMap::new();
        previous.insert(0, 1u64);
        let plan = upload_plan(Some((next.atlas_id, next.epoch, &previous)), &next);
        assert_eq!(plan, vec![PageUpload::Dirty]);
        assert!(next.pages[0].last_dirty.is_some());
    }
}
