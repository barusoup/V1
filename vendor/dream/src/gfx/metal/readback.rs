use super::Renderer;
use crate::gfx::FrameMetrics;
use crate::ui::render::UiBlend;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLOrigin, MTLPixelFormat, MTLSize,
};

impl Renderer {
    fn display_bgra(&self) -> bool {
        matches!(
            self.display.format,
            MTLPixelFormat::BGRA8Unorm_sRGB | MTLPixelFormat::BGRA8Unorm
        )
    }

    fn display_srgb(&self) -> bool {
        super::metal::is_srgb(self.display.format) || self.ui_blend == UiBlend::Gamma
    }

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

    fn read_texture_bytes(
        &self,
        texture: &ProtocolObject<dyn objc2_metal::MTLTexture>,
        w: u32,
        h: u32,
        bpp: u32,
        label: &str,
    ) -> Vec<u8> {
        let row_bytes = (w * bpp) as u64;
        let total_bytes = row_bytes * h as u64;
        let buffer = self.ctx.create_readback_buffer(label, total_bytes);
        self.ctx.one_shot(|cmd| {
            let blit = cmd.blitCommandEncoder().expect("blit encoder");
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize {
                        width: w as usize,
                        height: h as usize,
                        depth: 1,
                    },
                    &buffer.buffer,
                    0,
                    row_bytes as usize,
                    total_bytes as usize,
                );
            }
            blit.endEncoding();
        });
        buffer.mapped().expect("readback buffer is host visible")[..total_bytes as usize].to_vec()
    }

    fn read_display_bytes(&mut self) -> Vec<u8> {
        let (w, h) = self.display_size;
        if !self.display_readable() {
            return vec![0; (w as usize) * (h as usize) * 4];
        }
        self.read_texture_bytes(&self.display.texture, w, h, 4, "readback")
    }

    pub fn capture_velocity_rg16f(&mut self) -> (Vec<u8>, u32, u32) {
        let (w, h) = self.render_size;
        (self.read_velocity_bytes(), w, h)
    }

    pub fn capture_depth_f32(&mut self) -> (Vec<u8>, u32, u32) {
        self.ensure_scene_targets();
        let (w, h) = self.render_size;
        let bytes = self.read_texture_bytes(&self.targets.depth.texture, w, h, 4, "depth readback");
        (bytes, w, h)
    }

    fn read_velocity_bytes(&mut self) -> Vec<u8> {
        self.ensure_scene_targets();
        let (rw, rh) = self.render_size;
        self.read_texture_bytes(
            &self.targets.velocity.texture,
            rw,
            rh,
            4,
            "velocity readback",
        )
    }

    fn read_reactive_bytes(&self) -> Option<Vec<u8>> {
        let fsr = self.fsr.as_ref()?;
        let (tex, (rw, rh)) = fsr.reactive_masks();
        Some(self.read_texture_bytes(tex, rw, rh, 4, "reactive readback"))
    }

    pub fn scene_gpu_ms(&self) -> f32 {
        self.scene_ms.iter().copied().fold(0.0, f32::max)
    }

    pub fn scene_gpu_sample(&self) -> Option<(u64, f32)> {
        self.scene_sample
    }

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
            if r > 0.01 || g > 0.01 || b > 0.01 {
                nonblack += 1;
            }
            if b > 0.25 && b > r * 1.4 {
                sky += 1;
                let y = i / (w as u64);
                if y * 2 < h as u64 {
                    sky_top += 1;
                } else {
                    sky_bottom += 1;
                }
            }
        }
        let fcount = count.max(1) as f64;
        let half = (count / 2).max(1) as f64;

        let mv_bytes = self.read_velocity_bytes();
        let mv_count = (rw as usize) * (rh as usize);
        let mut mv_max = 0.0f32;
        let mut mv_sum = 0.0f64;
        let mut mv_large = 0u64;
        let mut mv_nonzero = 0u64;
        for i in 0..mv_count {
            let b = i * 4;
            let vx = half_to_f32(u16::from_le_bytes([mv_bytes[b], mv_bytes[b + 1]]));
            let vy = half_to_f32(u16::from_le_bytes([mv_bytes[b + 2], mv_bytes[b + 3]]));
            let mag = (vx * vx + vy * vy).sqrt();
            if mag > mv_max {
                mv_max = mag;
            }
            mv_sum += mag as f64;
            if mag > 0.25 {
                mv_large += 1;
            }
            if mag > 1e-4 {
                mv_nonzero += 1;
            }
        }
        let mv_fcount = mv_count.max(1) as f64;

        let mut reactive_mean = [0.0f32; 4];
        if let Some(rx_bytes) = self.read_reactive_bytes() {
            let rx_count = (rw as usize) * (rh as usize);
            let mut rx_sum = [0.0f64; 4];
            for i in 0..rx_count {
                let b = i * 4;
                for c in 0..4 {
                    rx_sum[c] += rx_bytes[b + c] as f64 / 255.0;
                }
            }
            let rx_fcount = rx_count.max(1) as f64;
            for c in 0..4 {
                reactive_mean[c] = (rx_sum[c] / rx_fcount) as f32;
            }
        }

        FrameMetrics {
            size: (w, h),
            mean: [
                (sum[0] / fcount) as f32,
                (sum[1] / fcount) as f32,
                (sum[2] / fcount) as f32,
            ],
            sky_pct: (sky as f64 / fcount * 100.0) as f32,
            sky_top_pct: (sky_top as f64 / half * 100.0) as f32,
            sky_bottom_pct: (sky_bottom as f64 / half * 100.0) as f32,
            nonblack_pct: (nonblack as f64 / fcount * 100.0) as f32,
            mv_max,
            mv_mean: (mv_sum / mv_fcount) as f32,
            mv_large_pct: (mv_large as f64 / mv_fcount * 100.0) as f32,
            mv_nonzero_pct: (mv_nonzero as f64 / mv_fcount * 100.0) as f32,
            reactive_mean,
        }
    }

    pub fn capture_rgba8(&mut self) -> (Vec<u8>, u32, u32) {
        let (w, h) = self.display_size;
        let data = self.read_display_bytes();
        let bgra = self.display_bgra();
        let srgb = self.display_srgb();
        let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
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

    pub fn save_png(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let (pixels, w, h) = self.capture_rgba8();
        crate::png::write(path, w, h, &pixels)
    }

    pub fn readback_stats(&mut self) -> String {
        self.frame_metrics().summary()
    }
}

fn display_linear(data: &[u8], index: usize, bgra: bool, srgb: bool) -> [f32; 3] {
    let base = index * 4;
    let (r, g, b) = if bgra {
        (data[base + 2], data[base + 1], data[base])
    } else {
        (data[base], data[base + 1], data[base + 2])
    };
    let to_linear = |v: u8| -> f32 {
        let f = v as f32 / 255.0;
        if !srgb {
            return f;
        }
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    };
    [to_linear(r), to_linear(g), to_linear(b)]
}

fn linear_to_srgb8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut e = -14i32;
            let mut m = frac;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | (0xff << 23) | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f_bits)
}
