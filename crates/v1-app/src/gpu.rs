// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Textures the UI draws: icons and album covers.
//!
//! Building a frame is pure CPU work with no access to the device, so drawing code only *asks* for
//! textures: a missing one is recorded, the frame draws a stand-in (a nearby icon size, or a
//! placeholder tile), and the host makes the texture before the next frame. Covers arrive decoded
//! from the network thread; icons are rasterized here on demand.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use dream::gfx::GpuContext;
use dream::texture::{GpuTexture, SamplerDesc, TextureFormat, TextureOptions, TextureUploader};
use dream::ui::Color;

use crate::icons::{self, Icon};
use crate::theme;

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

/// How many cover textures are kept. A 256×256 RGBA cover is 256 KiB, so this bounds the cover
/// cache to 16 MiB; a screenful of rows and pages is far below it. Older covers are dropped first,
/// and a row that still shows one simply asks for it again on the next frame.
const COVER_CACHE: usize = 64;
/// Keep decoded covers waiting for upload within the same 16 MiB budget as the texture cache.
const COVER_PENDING: usize = 64;

/// Session cache cap for extracted album cover accent colours. Colours are 4 bytes each, so
/// keeping hundreds in memory avoids re-extracting even after a cover texture is evicted.
const COVER_COLOR_CACHE: usize = 512;

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
    /// Cover URLs in the order they entered the cache, oldest first. `covers` may lose entries as
    /// they are evicted; stale URLs in here are skipped when the next eviction runs.
    cover_order: VecDeque<String>,
    /// Session cache of each cover URL's extracted "interesting" colour, computed once on upload.
    cover_colors: HashMap<String, Color>,
    cover_color_order: VecDeque<String>,
    pub wordmark: Option<Wordmark>,
    /// The custom background, softened on a loading thread and uploaded here.
    pub background: Option<Arc<GpuTexture>>,
    background_animated: bool,
    /// Reuses staging buffers and the same single-mip texture for decoded
    /// animation frames; dropping a texture per frame otherwise churns GPU
    /// allocations and forces a synchronous upload.
    background_uploader: Option<TextureUploader>,
    /// Its pixel size, for fitting it to the window without stretching it.
    pub background_size: (u32, u32),
    /// A softened background waiting to be uploaded, newest only.
    pub incoming_background: Option<CoverImage>,
    /// Icon sizes the last frame wanted and did not have.
    pub wanted_icons: HashSet<(Icon, u32)>,
    pub incoming: Vec<CoverImage>,
}

impl Gpu {
    /// Clear the background after retiring any upload that still targets its texture.
    pub fn clear_background(&mut self) {
        if let Some(uploader) = self.background_uploader.as_mut() {
            uploader.wait_idle();
        }
        // Return the staging buffers too when no animation needs them.
        self.background_uploader = None;
        self.background = None;
        self.background_animated = false;
        self.background_size = (0, 0);
        self.incoming_background = None;
    }

    /// Queue the newest decoded copy of a cover. Repeated requests for one URL do not retain
    /// multiple 256 KiB buffers, and a burst cannot grow memory without bound while uploads wait.
    pub fn queue_cover(&mut self, cover: CoverImage) {
        if self.covers.contains_key(&cover.url) {
            return;
        }
        self.incoming.retain(|queued| queued.url != cover.url);
        if self.incoming.len() >= COVER_PENDING {
            self.incoming.remove(0);
        }
        self.incoming.push(cover);
    }

    /// Make every texture that was asked for. Called by the host between frames.
    pub fn sync(&mut self, ctx: &Arc<GpuContext>) {
        if self.wordmark.is_none() {
            if let Ok(image) =
                image::load_from_memory_with_format(WORDMARK_PNG, image::ImageFormat::Png)
            {
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
                self.wordmark = Some(Wordmark {
                    texture: Arc::new(texture),
                    width,
                    height,
                });
            }
        }
        // The background is one texture, so it goes first: a frame that asks for it is about to
        // show it.
        if let Some(background) = self.incoming_background.take() {
            let animated = background.url.starts_with("background:anim:");
            let same_size = animated
                && self.background_animated
                && self
                    .background
                    .as_ref()
                    .is_some_and(|texture| texture.size() == (background.width, background.height));
            if same_size {
                let uploader = self
                    .background_uploader
                    .get_or_insert_with(|| TextureUploader::new(ctx));
                uploader.upload(
                    self.background.as_deref().expect("same-size background"),
                    &background.rgba,
                );
                self.background_size = (background.width, background.height);
            } else {
                // An in-flight copy still owns the old texture as its target.
                // Retire it before replacing or dropping that allocation.
                if let Some(uploader) = self.background_uploader.as_mut() {
                    uploader.wait_idle();
                }
                let options = TextureOptions {
                    format: TextureFormat::Srgb,
                    sampler: SamplerDesc::default(),
                    mip_levels: if animated { Some(1) } else { None },
                };
                let texture = if animated {
                    GpuTexture::blank(
                        ctx,
                        "background",
                        background.width,
                        background.height,
                        options,
                    )
                } else {
                    GpuTexture::from_rgba8(
                        ctx,
                        "background",
                        &background.rgba,
                        background.width,
                        background.height,
                        options,
                    )
                };
                self.background_size = (background.width, background.height);
                self.background = Some(Arc::new(texture));
                self.background_animated = animated;
                if !animated {
                    self.background_uploader = None;
                }
                if animated {
                    let uploader = self
                        .background_uploader
                        .get_or_insert_with(|| TextureUploader::new(ctx));
                    uploader.upload(
                        self.background.as_deref().expect("just created background"),
                        &background.rgba,
                    );
                }
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
        let incoming: Vec<_> = self.incoming.drain(..take).collect();
        for cover in incoming {
            self.cache_cover_color(&cover.url, &cover.rgba, cover.width, cover.height);
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
            self.insert_cover(cover.url, Arc::new(texture));
        }
    }

    /// Cached "interesting" colour for `url`, if already extracted this session.
    pub fn cover_color(&self, url: &str) -> Option<Color> {
        self.cover_colors.get(url).copied()
    }

    /// Compute and cache the most interesting colour for `url` once per session.
    pub fn cache_cover_color(&mut self, url: &str, rgba: &[u8], width: u32, height: u32) {
        if self.cover_colors.contains_key(url) {
            return;
        }
        let Some(color) = theme::interesting_color(rgba, width, height) else {
            return;
        };
        if self.cover_colors.insert(url.to_owned(), color).is_none() {
            self.cover_color_order.push_back(url.to_owned());
        }
        while self.cover_colors.len() > COVER_COLOR_CACHE {
            let Some(oldest) = self.cover_color_order.pop_front() else {
                break;
            };
            self.cover_colors.remove(&oldest);
        }
    }

    /// Put a cover in the cache, evicting the oldest ones once it is full. The caller asked for this
    /// URL because a row on screen needs it, so it goes at the newest end.
    fn insert_cover(&mut self, url: String, texture: Arc<GpuTexture>) {
        if self.covers.insert(url.clone(), texture).is_none() {
            self.cover_order.push_back(url);
        }
        while self.covers.len() > COVER_CACHE {
            let Some(oldest) = self.cover_order.pop_front() else {
                break;
            };
            self.covers.remove(&oldest);
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // The uploader's command buffers refer to the background texture; wait
        // before normal field destruction releases that texture.
        if let Some(uploader) = self.background_uploader.as_mut() {
            uploader.wait_idle();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cover(url: &str, byte: u8) -> CoverImage {
        CoverImage {
            url: url.to_owned(),
            width: 1,
            height: 1,
            rgba: vec![byte; 4],
        }
    }

    #[test]
    fn pending_covers_coalesce_and_remain_bounded() {
        let mut gpu = Gpu::default();
        gpu.queue_cover(cover("same", 1));
        gpu.queue_cover(cover("same", 2));
        assert_eq!(gpu.incoming.len(), 1);
        assert_eq!(gpu.incoming[0].rgba[0], 2);

        for index in 0..=COVER_PENDING {
            gpu.queue_cover(cover(&format!("cover-{index}"), 3));
        }
        assert_eq!(gpu.incoming.len(), COVER_PENDING);
        assert!(!gpu.incoming.iter().any(|cover| cover.url == "same"));
        assert_eq!(
            gpu.incoming.last().unwrap().url,
            format!("cover-{COVER_PENDING}")
        );
    }
}
