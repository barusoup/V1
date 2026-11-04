// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Textures the UI draws: icons and album covers.
//!
//! Building a frame is pure CPU work with no access to the device, so drawing code only *asks* for
//! textures: a missing one is recorded, the frame draws a stand-in (a nearby icon size, or a
//! placeholder tile), and the host makes the texture before the next frame. Covers arrive decoded
//! from the network thread; icons are rasterized here on demand.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dream::gfx::GpuContext;
use dream::texture::{GpuTexture, SamplerDesc, TextureFormat, TextureOptions};

use crate::icons::{self, Icon};

/// A decoded cover waiting to be uploaded.
pub struct CoverImage {
    pub url: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The backgroundless V1 mark for the sidebar, in the app's grey text colour (`theme::MUTED`) with
/// no outline. Rendered from `assets/icon/V1_wordmark.svg`.
const WORDMARK_PNG: &[u8] = include_bytes!("../assets/icon/V1_wordmark_sidebar.png");

/// Covers uploaded per frame. Each upload is a blocking round trip to the GPU; see [`Gpu::sync`].
const COVER_UPLOADS_PER_FRAME: usize = 4;

/// The decoded wordmark texture and its pixel size, so the caller can keep its aspect ratio.
pub struct Wordmark {
    pub texture: Arc<GpuTexture>,
    pub width: u32,
    pub height: u32,
}

#[derive(Default)]
pub struct Gpu {
    pub icons: HashMap<(Icon, u32), Arc<GpuTexture>>,
    pub covers: HashMap<String, Arc<GpuTexture>>,
    pub wordmark: Option<Wordmark>,
    /// Icon sizes the last frame wanted and did not have.
    pub wanted_icons: HashSet<(Icon, u32)>,
    pub incoming: Vec<CoverImage>,
}

impl Gpu {
    /// Make every texture that was asked for. Called by the host between frames.
    pub fn sync(&mut self, ctx: &Arc<GpuContext>) {
        if self.wordmark.is_none() {
            if let Ok(image) = image::load_from_memory_with_format(WORDMARK_PNG, image::ImageFormat::Png) {
                let rgba = image.to_rgba8();
                let (width, height) = rgba.dimensions();
                let texture = GpuTexture::from_rgba8(
                    ctx,
                    "wordmark",
                    rgba.as_raw(),
                    width,
                    height,
                    TextureOptions {
                        format: TextureFormat::Srgb,
                        sampler: SamplerDesc::default(),
                        // Full mip chain: the mark is drawn smaller than its texture, and
                        // without mips the GPU samples one level and aliases.
                        mip_levels: None,
                    },
                );
                self.wordmark = Some(Wordmark { texture: Arc::new(texture), width, height });
            }
        }
        for (icon, size) in self.wanted_icons.drain().collect::<Vec<_>>() {
            if self.icons.contains_key(&(icon, size)) {
                continue;
            }
            let pixels = icons::rasterize(icon, size);
            let texture = GpuTexture::from_rgba8(
                ctx,
                "icon",
                &pixels,
                size,
                size,
                TextureOptions {
                    format: TextureFormat::Srgb,
                    sampler: SamplerDesc::default(),
                    mip_levels: Some(1),
                },
            );
            self.icons.insert((icon, size), Arc::new(texture));
        }
        // Each upload waits for the GPU to drain, so a search result list bringing in dozens of covers
        // at once would stall one frame. The rest wait their turn; the grey tile stands in meanwhile.
        let take = self.incoming.len().min(COVER_UPLOADS_PER_FRAME);
        for cover in self.incoming.drain(..take) {
            let texture = GpuTexture::from_rgba8(
                ctx,
                "cover",
                &cover.rgba,
                cover.width,
                cover.height,
                TextureOptions {
                    format: TextureFormat::Srgb,
                    sampler: SamplerDesc::default(),
                    mip_levels: None,
                },
            );
            self.covers.insert(cover.url, Arc::new(texture));
        }
    }
}
