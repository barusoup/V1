use crate::impostor::VIEW_COUNT;

/// Six-tile RGBA strips (128px per tile) per channel for the atlas panel.
pub(super) fn build_previews(
    albedo: &[Vec<u8>],
    nao: &[Vec<u8>],
    mr: &[Vec<u8>],
    emi: &[Vec<u8>],
    dep: &[Vec<u16>],
    res: u32,
) -> Vec<(String, Vec<u8>, u32, u32)> {
    let tile = 96u32;
    let mip_side = |level: &Vec<u8>| -> u32 { ((level.len() / 4) as f64).sqrt() as u32 };
    let mut out = Vec::new();
    for (name, levels) in [
        ("albedo", albedo),
        ("normal+ao", nao),
        ("mr", mr),
        ("emissive", emi),
    ] {
        let mut strip = vec![0u8; (tile * tile * VIEW_COUNT as u32 * 4) as usize];
        for f in 0..VIEW_COUNT {
            let level0 = &levels[f * levels.len() / VIEW_COUNT];
            let side = mip_side(level0);
            for y in 0..tile {
                for x in 0..tile {
                    let sx = x * side / tile;
                    let sy = y * side / tile;
                    let si = ((sy * side + sx) * 4) as usize;
                    let oi =
                        (((y * tile * VIEW_COUNT as u32) + (f as u32) * tile + x) * 4) as usize;
                    strip[oi..oi + 4].copy_from_slice(&level0[si..si + 4]);
                }
            }
        }
        out.push((name.to_string(), strip, tile * VIEW_COUNT as u32, tile));
    }
    let mut strip = vec![0u8; (tile * tile * VIEW_COUNT as u32 * 4) as usize];
    for f in 0..VIEW_COUNT {
        let level0 = &dep[f * dep.len() / VIEW_COUNT];
        for y in 0..tile {
            for x in 0..tile {
                let sx = x * res / tile;
                let sy = y * res / tile;
                let d = level0[(sy * res + sx) as usize];
                let v = (d as u32 * 255 / 65535) as u8;
                let oi = (((y * tile * VIEW_COUNT as u32) + (f as u32) * tile + x) * 4) as usize;
                strip[oi..oi + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
    }
    out.push(("depth".to_string(), strip, tile * VIEW_COUNT as u32, tile));
    out
}
