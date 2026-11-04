//! Pixel-art icons: hand-drawn bitmaps baked into the UI atlas.
//!
//! Icons are authored as string grids (`#` = ink, anything else = empty),
//! exactly like the sprites they will eventually edit. Baking them as
//! *coverage* into the glyph atlas means every icon is one merged quad that
//! tints like text, costs nothing to draw, and stays crisp: each icon is
//! baked once per integer pixel scale, so a 8x8 icon drawn at 3x renders
//! 24x24 texels with no filtering blur.
//!
//! ```
//! use dream::bedframe::pixel::builtin;
//!
//! // Pick one of the built-in icons and read its art out as a coverage
//! // bitmap: 255 per inked cell, 0 per empty one.
//! let play = *builtin().get("play").expect("play is a built-in icon");
//! assert_eq!((play.width(), play.height()), (8, 8));
//! let coverage = play.coverage();
//! assert_eq!(coverage.len(), 8 * 8);
//! assert!(coverage.iter().all(|&c| c == 0 || c == 255));
//! // Scale 3 keeps the art crisp: 8 icon pixels become 24 device pixels.
//! assert_eq!(play.coverage_scaled(3).len(), 24 * 24);
//! // Names are looked up through the map, so a typo is a plain miss.
//! let icons = builtin();
//! assert!(icons.get("no-such-icon").is_none());
//! ```

use super::atlas::Region;
use super::font::GlyphCache;
use std::collections::HashMap;

/// One pixel-art icon: a grid of rows, `#` for ink. All rows must be the
/// same length; the grid may be empty.
#[derive(Clone, Copy, Debug)]
pub struct PixelIcon {
    rows: &'static [&'static str],
}

impl PixelIcon {
    pub const fn new(rows: &'static [&'static str]) -> Self {
        Self { rows }
    }

    pub fn width(&self) -> u32 {
        self.rows.first().map_or(0, |row| row.len() as u32)
    }

    pub fn height(&self) -> u32 {
        self.rows.len() as u32
    }

    /// Ink coverage at one unit pixel (0 or 255).
    pub fn coverage(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(self.rows.len() * self.rows.first().map_or(0, |r| r.len()));
        for row in self.rows {
            for ch in row.chars() {
                out.push(if ch == '#' { 255 } else { 0 });
            }
        }
        out
    }

    /// Nearest-neighbour upscale to `scale` device pixels per icon pixel.
    /// Integer scales only: fractional scales would blur the art.
    pub fn coverage_scaled(&self, scale: u32) -> Vec<u8> {
        assert!(scale >= 1, "icon scale must be at least 1");
        if scale == 1 {
            return self.coverage();
        }
        let base = self.coverage();
        let w = self.width() as usize;
        let h = self.height() as usize;
        let mut out = vec![0u8; w * h * scale as usize * scale as usize];
        for y in 0..h {
            for x in 0..w {
                let value = base[y * w + x];
                if value == 0 {
                    continue;
                }
                for sy in 0..scale as usize {
                    let row = (y * scale as usize + sy) * w * scale as usize + x * scale as usize;
                    for sx in 0..scale as usize {
                        out[row + sx] = value;
                    }
                }
            }
        }
        out
    }
}

/// Named pixel icons, baked into a [`GlyphCache`] atlas per (icon, scale).
///
/// Register once at startup; afterwards `region` hands out atlas regions
/// for drawing. Baking is idempotent — re-registering a name replaces it.
#[derive(Debug, Default)]
pub struct IconSet {
    regions: HashMap<(String, u32), Region>,
}

impl IconSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bake `icon` under `name` at every scale the UI asks for (lazily).
    pub fn register(&mut self, name: &str, icon: PixelIcon, cache: &mut GlyphCache) {
        let key = (name.to_string(), 1u32);
        if self.regions.contains_key(&key) {
            return;
        }
        let coverage = icon.coverage();
        match cache
            .atlas_mut()
            .insert_coverage(&coverage, icon.width(), icon.height())
        {
            Ok(region) => {
                self.regions.insert(key, region);
            }
            Err(err) => {
                eprintln!("[ui] icon {name} too large for the atlas: {err}");
            }
        }
    }

    /// The atlas region for `name` at `scale` (1 unless already baked).
    /// Bakes on demand; `None` when the name was never registered.
    pub fn region(&mut self, name: &str, scale: u32, cache: &mut GlyphCache) -> Option<Region> {
        let scale = scale.max(1);
        let key = (name.to_string(), scale);
        if let Some(region) = self.regions.get(&key) {
            return Some(*region);
        }
        let icon = *builtin().get(name)?;
        let coverage = icon.coverage_scaled(scale);
        let region = cache
            .atlas_mut()
            .insert_coverage(&coverage, icon.width() * scale, icon.height() * scale)
            .ok()?;
        self.regions.insert(key, region);
        Some(region)
    }

    /// Number of baked (icon, scale) pairs — atlas growth diagnostics.
    pub fn baked(&self) -> usize {
        self.regions.len()
    }
}

/// The built-in editor icon set, drawn on the same 8x8 grid as the HTML
/// napxl app that inspired this UI. Engine-specific icons join the family.
pub fn builtin() -> HashMap<&'static str, PixelIcon> {
    let mut map: HashMap<&'static str, PixelIcon> = HashMap::new();
    let mut add = |name: &'static str, rows: &'static [&'static str]| {
        map.insert(name, PixelIcon::new(rows));
    };
    // Tools (napxl parity).
    add(
        "pencil",
        &[
            ".....##.", "....####", "...####.", "..####..", ".####...", ".###....", "###.....",
            "##......",
        ],
    );
    add(
        "eraser",
        &[
            "...####.", "..#...##", ".#...#.#", "#...#.#.", ".#.#.#..", "..#.#...", "...#....",
            "........",
        ],
    );
    add(
        "fill",
        &[
            "...##...", "..#..#..", ".######.", "#......#", "#......#", ".#....#.", "..####..",
            "........",
        ],
    );
    add(
        "pick",
        &[
            "....###.", "...#####", "....###.", "...##...", "..##....", ".##.....", "#.......",
            "........",
        ],
    );
    add(
        "line",
        &[
            "......##", ".....##.", "....##..", "...##...", "..##....", ".##.....", "##......",
            "#.......",
        ],
    );
    add(
        "rect",
        &[
            "########", "#......#", "#......#", "#......#", "#......#", "#......#", "#......#",
            "########",
        ],
    );
    // Toggles and chrome (napxl parity).
    add(
        "mirror",
        &[
            "..#..#..", ".##..##.", "###..###", "###..###", ".##..##.", "..#..#..", "........",
            "........",
        ],
    );
    add(
        "grid",
        &[
            "#######.", "#..#..#.", "#..#..#.", "#######.", "#..#..#.", "#..#..#.", "#######.",
            "........",
        ],
    );
    add(
        "undo",
        &[
            "..#.....", ".##.....", "#######.", ".##...##", "..#....#", ".......#", "......##",
            ".#####..",
        ],
    );
    add(
        "redo",
        &[
            ".....#..", ".....##.", "#######.", "##...##.", "#....#..", "#.......", "##......",
            ".#####..",
        ],
    );
    add(
        "trash",
        &[
            "..####..", "########", ".#.##.#.", ".#.##.#.", ".#.##.#.", ".#.##.#.", ".######.",
            "........",
        ],
    );
    add(
        "theme",
        &[
            "..####..", ".#..###.", "#...####", "#...####", "#...####", "#...####", ".#..###.",
            "..####..",
        ],
    );
    add(
        "import",
        &[
            "...##...", "...##...", ".######.", "..####..", "...##...", "#......#", "#......#",
            "########",
        ],
    );
    add(
        "export",
        &[
            "...##...", "..####..", ".######.", "...##...", "...##...", "#......#", "#......#",
            "########",
        ],
    );
    add(
        "photo",
        &[
            "..####..", "########", "#......#", "#..##..#", "#..##..#", "#......#", "########",
            "........",
        ],
    );
    add(
        "napxl",
        &[
            "#######.", "#.....#.", "#.###.#.", "#.....#.", "#.###.#.", "#.....#.", "#######.",
            "........",
        ],
    );
    // Engine icons.
    add(
        "camera",
        &[
            "..#####..",
            ".#.....#.",
            "#.......#",
            "#..###..#",
            "#..###..#",
            "#.......#",
            ".#.....#.",
            "..#####..",
        ],
    );
    add(
        "cube",
        &[
            "..####..", ".#....#.", "#......#", "#......#", "#......#", "#......#", ".#....#.",
            "..####..",
        ],
    );
    add(
        "light",
        &[
            "..####..", ".######.", "########", "########", "########", "..####..", "...##...",
            "..####..",
        ],
    );
    add(
        "play",
        &[
            "#.......", "##......", "###.....", "####....", "####....", "###.....", "##......",
            "#.......",
        ],
    );
    add(
        "pause",
        &[
            "###..###", "###..###", "###..###", "###..###", "###..###", "###..###", "###..###",
            "###..###",
        ],
    );
    add(
        "gear",
        &[
            "..####..", ".######.", "##....##", "##.##.##", "##.##.##", "##....##", ".######.",
            "..####..",
        ],
    );
    add(
        "orbit",
        &[
            "..####..", ".#....#.", "#..##..#", "#......#", "##.##.##", "#......#", ".#....#.",
            "..####..",
        ],
    );
    add(
        "wave",
        &[
            "........", "........", "..####..", ".#....#.", "#......#", "##.##.##", "#......#",
            ".######.",
        ],
    );
    add(
        "info",
        &[
            "..####..", ".#....#.", ".#....#.", ".#....#.", "........", "..####..", "..####..",
            "........",
        ],
    );
    add(
        "x",
        &[
            "##....##", "###..###", ".######.", "..####..", "..####..", ".######.", "###..###",
            "##....##",
        ],
    );
    add(
        "check",
        &[
            ".......#", "......##", ".....##.", "#...##..", "##.##...", ".###....", "..#.....",
            "........",
        ],
    );
    add(
        "chevron_down",
        &[
            "........", "........", "########", ".######.", "..####..", "...##...", "........",
            "........",
        ],
    );
    add(
        "chevron_right",
        &[
            "#.......", "##......", "###.....", "####....", "###.....", "##......", "#.......",
            "........",
        ],
    );
    add(
        "speaker",
        &[
            "..#.....", "##.#....", "##.##...", "##.##...", "##.##...", "##.#....", "..#.....",
            "........",
        ],
    );
    add(
        "mute",
        &[
            "..#....#", ".##...#.", "####.#..", "####..#.", ".##..#.#", "..#.#..#", ".....#..",
            "........",
        ],
    );
    add(
        "vol1",
        &[
            "........", "..#.....", "..#.#...", "###..#..", "###..#..", "..#.#...", "..#.....",
            "........",
        ],
    );
    add(
        "vol2",
        &[
            "........", "..#...#.", "..#.#..#", "###..#.#", "###..#.#", "..#.#..#", "..#...#.",
            "........",
        ],
    );
    map
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
    fn coverage_matches_grid() {
        let icon = PixelIcon::new(&["#.", ".#"]);
        assert_eq!(icon.coverage(), vec![255, 0, 0, 255]);
        assert_eq!((icon.width(), icon.height()), (2, 2));
    }

    #[test]
    fn scaled_coverage_is_nearest() {
        let icon = PixelIcon::new(&["#.", ".#"]);
        let scaled = icon.coverage_scaled(2);
        assert_eq!(scaled.len(), 16);
        assert_eq!(scaled[0], 255);
        assert_eq!(scaled[1], 255);
        assert_eq!(scaled[2], 0);
        assert_eq!(scaled[3], 0);
    }

    #[test]
    fn icons_bake_and_region_stable() {
        let mut cache = cache();
        let mut icons = IconSet::new();
        for (name, icon) in builtin() {
            icons.register(name, icon, &mut cache);
        }
        assert_eq!(icons.baked(), builtin().len());
        let first = icons.region("play", 1, &mut cache).expect("baked");
        let second = icons.region("play", 1, &mut cache).expect("baked");
        assert_eq!(first, second, "region is stable across calls");
        assert!(icons.region("nope", 1, &mut cache).is_none());
        // Crisp: scale 3 is 24x24 texels.
        let big = icons
            .region("play", 3, &mut cache)
            .expect("baked on demand");
        assert_eq!((big.width, big.height), (24, 24));
    }

    #[test]
    fn baked_icons_have_ink_in_atlas() {
        let mut cache = cache();
        let mut icons = IconSet::new();
        icons.register("check", PixelIcon::new(&[".......#"]), &mut cache);
        let region = icons.region("check", 1, &mut cache).unwrap();
        let page = cache.atlas().page_pixels(region.page);
        let size = cache.atlas().page_size() as usize;
        let at = (region.y as usize * size + region.x as usize) * 4;
        assert_eq!(&page[at..at + 4], &[255, 255, 255, 0], "first cell empty");
        let last = at + 7 * 4;
        assert_eq!(&page[last..last + 4], &[255, 255, 255, 255], "ink is white");
    }
}
