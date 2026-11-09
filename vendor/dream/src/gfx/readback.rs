use super::vulkan;
use super::{FrameMetrics, Renderer};
use crate::ui::render::UiBlend;
use ash::vk;

impl Renderer {
    /// Whether display-target pixels are stored blue-first (B8G8R8A8).
    fn display_bgra(&self) -> bool {
        matches!(
            self.swapchain.format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
        )
    }

    /// Whether display-target pixels are sRGB-encoded. True for an sRGB
    /// surface, and for the gamma blend, which stores encoded values whatever
    /// the surface format is.
    fn display_srgb(&self) -> bool {
        vulkan::is_srgb(self.swapchain.format) || self.ui_blend == UiBlend::Gamma
    }

    /// Whether the composed display image holds a frame that a readback can
    /// copy. Frames are composed straight into the swapchain unless capture is
    /// on, so when nothing is retained this turns capture on (the next frame
    /// is kept), says so once on stderr, and returns `false`: the readback then
    /// yields an empty frame rather than copying the placeholder or an
    /// unwritten image.
    fn display_readable(&mut self) -> bool {
        if self.display_written {
            return true;
        }
        if self.frame_capture {
            if !self.capture_notice {
                eprintln!(
                    "dream: no frame has been rendered since Renderer::set_frame_capture(true); the readback is empty."
                );
                self.capture_notice = true;
            }
        } else {
            self.set_frame_capture(true);
            if !self.capture_notice {
                eprintln!(
                    "dream: Renderer frame capture was off, so the last frame was not retained; enabled it now — call Renderer::set_frame_capture(true) before rendering frames you read back."
                );
                self.capture_notice = true;
            }
        }
        false
    }

    /// Copy the composed display image to the CPU. The display target is in
    /// the swapchain format — 8-bit BGRA or RGBA, sRGB-encoded when the
    /// surface is sRGB — not the scene pass's RGBA16F. Without a retained
    /// frame (see [`Self::display_readable`]) this returns zeroed texels.
    fn read_display_bytes(&mut self) -> Vec<u8> {
        let (w, h) = self.display_size;
        if !self.display_readable() {
            return vec![0; (w as usize) * (h as usize) * 4];
        }
        let bytes = (w as u64) * (h as u64) * 4;
        let buffer = self.ctx.create_readback_buffer("readback", bytes);
        let image = self.display.image;
        let device = self.ctx.device().clone();
        self.ctx.one_shot(|cmd| unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageAspectFlags::COLOR,
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &[region],
            );
        });
        buffer
            .mapped()
            .expect("readback buffer is host visible")
            .to_vec()
    }

    /// Raw bytes of the motion-vector target (render resolution, RG16
    /// half-float, two channels per texel, 4 bytes per pixel), plus its
    /// dimensions. No conversion: diagnostics comparing velocity output
    /// across submissions should diff these bytes directly.
    pub fn capture_velocity_rg16f(&mut self) -> (Vec<u8>, u32, u32) {
        let (w, h) = self.render_size;
        (self.read_velocity_bytes(), w, h)
    }

    /// Copy the scene depth target (render resolution, `D32_SFLOAT`, top row
    /// first, Vulkan clip depth 0..1 where 1.0 = far/cleared) to the CPU.
    /// Returns raw little-endian `f32` bytes, `render_size` texels.
    pub fn capture_depth_f32(&mut self) -> (Vec<u8>, u32, u32) {
        self.ensure_scene_targets();
        let (w, h) = self.render_size;
        let bytes = (w as u64) * (h as u64) * 4;
        let buffer = self.ctx.create_readback_buffer("depth readback", bytes);
        let image = self.targets.depth.image;
        let device = self.ctx.device().clone();
        self.ctx.one_shot(|cmd| unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageAspectFlags::DEPTH,
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &[region],
            );
            // Later depth writes must wait on this read; queue order alone is
            // not a memory barrier.
            vulkan::image_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::ImageAspectFlags::DEPTH,
            );
        });
        let data = buffer
            .mapped()
            .expect("depth buffer is host visible")
            .to_vec();
        // Vulkan image row 0 is the framebuffer's bottom row; flip so the
        // returned buffer is top-first like the PNG captures.
        let row = (w as usize) * 4;
        let mut flipped = vec![0u8; data.len()];
        for y in 0..h as usize {
            let src = (h as usize - 1 - y) * row;
            flipped[y * row..(y + 1) * row].copy_from_slice(&data[src..src + row]);
        }
        (flipped, w, h)
    }

    /// Copy the motion-vector target (render resolution, f16 two-channel) to the CPU.
    fn read_velocity_bytes(&mut self) -> Vec<u8> {
        self.ensure_scene_targets();
        let (rw, rh) = self.render_size;
        let bytes = (rw as u64) * (rh as u64) * 4;
        let buffer = self.ctx.create_readback_buffer("velocity readback", bytes);
        let image = self.targets.velocity.image;
        let device = self.ctx.device().clone();
        self.ctx.one_shot(|cmd| unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageAspectFlags::COLOR,
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: rw,
                    height: rh,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &[region],
            );
        });
        buffer
            .mapped()
            .expect("velocity buffer is host visible")
            .to_vec()
    }

    /// Copy the FSR dilated reactive masks (render resolution, R8G8B8A8_UNORM)
    /// to the CPU, or `None` when temporal upscaling is off.
    fn read_reactive_bytes(&mut self) -> Option<Vec<u8>> {
        let (image, (rw, rh)) = self.fsr.as_ref().map(|fsr| fsr.reactive_masks())?;
        let bytes = (rw as u64) * (rh as u64) * 4;
        let buffer = self.ctx.create_readback_buffer("reactive readback", bytes);
        let device = self.ctx.device().clone();
        self.ctx.one_shot(|cmd| unsafe {
            vulkan::image_barrier(
                &device,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageAspectFlags::COLOR,
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: rw,
                    height: rh,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                buffer.buffer,
                &[region],
            );
        });
        Some(
            buffer
                .mapped()
                .expect("reactive readback buffer is host visible")
                .to_vec(),
        )
    }

    /// GPU time of the scene pass on the frame that most recently completed on
    /// this slot cycle, in milliseconds (0 when timestamps are unsupported).
    pub fn scene_gpu_ms(&self) -> f32 {
        self.scene_ms.iter().copied().fold(0.0, f32::max)
    }

    /// Fresh completed scene-pass timestamp from the last render call.
    /// The submission serial lets benchmarks reject repeated/stale samples.
    /// `None` means no valid query completed during that call.
    pub fn scene_gpu_sample(&self) -> Option<(u64, f32)> {
        self.scene_sample
    }

    /// Summarise the last completed frame: mean colour, sky coverage by half and
    /// motion-vector activity. A debug aid for verifying rendering without
    /// looking at a window; `DREAM_FRAME_STATS` exposes it to the game. The
    /// colour part needs the composed frame retained: see
    /// [`Self::set_frame_capture`].
    pub fn frame_metrics(&mut self) -> FrameMetrics {
        let (w, h) = self.display_size;
        let (rw, rh) = self.render_size;
        let data = self.read_display_bytes();
        let count = (w as u64) * (h as u64);
        let bgra = self.display_bgra();
        let srgb = self.display_srgb();
        let mut sum = [0.0f64; 3];
        let mut sky = 0u64;
        let mut sky_top = 0u64;
        let mut sky_bottom = 0u64;
        let mut nonblack = 0u64;
        for i in 0..count {
            let [r, g, b] = display_linear(&data, i as usize, bgra, srgb);
            sum[0] += r as f64;
            sum[1] += g as f64;
            sum[2] += b as f64;
            if (r - 0.48).abs() < 0.02 && (g - 0.69).abs() < 0.02 && (b - 0.89).abs() < 0.02 {
                sky += 1;
                // The composed display image is in screen-row order (row 0 is
                // the top of the frame); report halves in screen space.
                let screen_row = i / (w as u64);
                if screen_row < (h as u64) / 2 {
                    sky_top += 1;
                } else {
                    sky_bottom += 1;
                }
            }
            if r + g + b > 0.01 {
                nonblack += 1;
            }
        }
        // Motion vectors are only useful if the scene pass really writes them.
        // A camera that moves while this reports zero means upscalers will
        // reproject as if nothing moved.
        let vdata = self.read_velocity_bytes();
        let vcount = (rw as u64) * (rh as u64);
        let mut mv_max = 0.0f32;
        let mut mv_sum = 0.0f64;
        let mut mv_large = 0u64;
        let mut mv_nonzero = 0u64;
        for i in 0..vcount {
            let base = (i * 4) as usize;
            let vx = f16_to_f32(u16::from_le_bytes([vdata[base], vdata[base + 1]]));
            let vy = f16_to_f32(u16::from_le_bytes([vdata[base + 2], vdata[base + 3]]));
            let mag = (vx * vx + vy * vy).sqrt();
            mv_sum += mag as f64;
            if mag > mv_max {
                mv_max = mag;
            }
            if mag > 0.25 {
                mv_large += 1;
            }
            if mag > 1e-5 {
                mv_nonzero += 1;
            }
        }
        let mut reactive_mean = [0.0f32; 4];
        if let Some(mdata) = self.read_reactive_bytes() {
            let pixels = (mdata.len() / 4).max(1) as f64;
            for i in 0..mdata.len() / 4 {
                let base = i * 4;
                for (channel, value) in reactive_mean.iter_mut().enumerate() {
                    *value += mdata[base + channel] as f32;
                }
            }
            for value in &mut reactive_mean {
                *value = (*value as f64 / pixels / 255.0) as f32;
            }
        }

        let half = (count / 2) as f64;
        FrameMetrics {
            size: (w, h),
            mean: [
                (sum[0] / count as f64) as f32,
                (sum[1] / count as f64) as f32,
                (sum[2] / count as f64) as f32,
            ],
            sky_pct: (100.0 * sky as f64 / count as f64) as f32,
            sky_top_pct: (100.0 * sky_top as f64 / half) as f32,
            sky_bottom_pct: (100.0 * sky_bottom as f64 / half) as f32,
            nonblack_pct: (100.0 * nonblack as f64 / count as f64) as f32,
            mv_max,
            mv_mean: (mv_sum / vcount as f64) as f32,
            mv_large_pct: (100.0 * mv_large as f64 / vcount as f64) as f32,
            mv_nonzero_pct: (100.0 * mv_nonzero as f64 / vcount as f64) as f32,
            reactive_mean,
        }
    }

    /// The composed frame as sRGB-encoded 8-bit RGBA, top row first, ready
    /// for [`crate::png::write`]. The display target is in the swapchain
    /// format: sRGB bytes are copied as-is (they are already display-encoded,
    /// so no re-encoding), and channel order is normalized to RGBA.
    ///
    /// Reads the frame retained by [`Self::set_frame_capture`]. When capture
    /// was off, no frame is retained: the result is black and capture is
    /// switched on for the next frame.
    pub fn capture_rgba8(&mut self) -> (Vec<u8>, u32, u32) {
        let (w, h) = self.display_size;
        let data = self.read_display_bytes();
        let bgra = self.display_bgra();
        let srgb = self.display_srgb();
        let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
        // The display image is in screen-row order; PNG rows run the same way,
        // so no flip here.
        for i in 0..(w as usize) * (h as usize) {
            let src = i * 4;
            let (r, g, b) = if bgra {
                (data[src + 2], data[src + 1], data[src])
            } else {
                (data[src], data[src + 1], data[src + 2])
            };
            let dst = i * 4;
            if srgb {
                out[dst] = r;
                out[dst + 1] = g;
                out[dst + 2] = b;
            } else {
                out[dst] = linear_to_srgb8(r as f32 / 255.0);
                out[dst + 1] = linear_to_srgb8(g as f32 / 255.0);
                out[dst + 2] = linear_to_srgb8(b as f32 / 255.0);
            }
            out[dst + 3] = 255;
        }
        (out, w, h)
    }

    /// Capture the last completed frame to a PNG file.
    pub fn save_png(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let (pixels, w, h) = self.capture_rgba8();
        crate::png::write(path, w, h, &pixels)
    }

    /// One-line summary of the last completed frame, see [`Self::frame_metrics`].
    pub fn readback_stats(&mut self) -> String {
        self.frame_metrics().summary()
    }
}

// --- Helpers ---------------------------------------------------------------

/// One display-target pixel as linear floats. The composed image stores 8-bit
/// pixels in the swapchain format: `bgra` selects channel order, `srgb`
/// whether the stored values are sRGB-encoded (decoded here so metric
/// comparisons work in linear space).
fn display_linear(data: &[u8], index: usize, bgra: bool, srgb: bool) -> [f32; 3] {
    let base = index * 4;
    let (r, g, b) = if bgra {
        (data[base + 2], data[base + 1], data[base])
    } else {
        (data[base], data[base + 1], data[base + 2])
    };
    if srgb {
        [srgb8_to_linear(r), srgb8_to_linear(g), srgb8_to_linear(b)]
    } else {
        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
    }
}

/// 8-bit sRGB to linear float.
fn srgb8_to_linear(value: u8) -> f32 {
    let s = value as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear colour to 8-bit sRGB, matching the swapchain's sRGB transfer function.
fn linear_to_srgb8(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let srgb = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5) as u8
}

/// Minimal IEEE-754 half -> single conversion for the readback diagnostic.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((127 - 15 + e + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}
