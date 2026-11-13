//! Shared PBR lighting for the native Metal backend.

use crate::gfx::metal::{GpuBuffer, GpuContext};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::sync::Arc;

pub const SHADER: &str = include_str!("lighting.wgsl");
pub const SET_INDEX: usize = 3;
pub const BINDING: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingEnv {
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_ground: [f32; 4],
    pub cam_pos: [f32; 4],
    pub params: [f32; 4],
}

impl Default for LightingEnv {
    fn default() -> Self {
        Self::with_sun(0.6, 0.7, 3.0)
    }
}

impl LightingEnv {
    pub fn with_sun(azimuth: f32, elevation: f32, intensity: f32) -> Self {
        let dir = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
            0.0,
        ];
        Self {
            sun_direction: dir,
            sun_color: [intensity, intensity * 0.96, intensity * 0.88, 0.0],
            sky_zenith: [0.18, 0.35, 0.65, 0.0],
            sky_horizon: [0.55, 0.60, 0.68, 0.0],
            sky_ground: [0.10, 0.09, 0.08, 0.0],
            cam_pos: [0.0; 4],
            params: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

pub struct Lighting {
    buffers: Vec<GpuBuffer>,
}

impl Lighting {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let slots = ctx.max_frames_in_flight();
        let buffers = (0..slots)
            .map(|slot| {
                ctx.create_host_buffer(
                    &format!("lighting env {slot}"),
                    std::mem::size_of::<LightingEnv>() as u64,
                )
            })
            .collect();
        Self { buffers }
    }

    pub fn upload(&mut self, frame_slot: usize, env: &LightingEnv) {
        let len = self.buffers.len();
        self.buffers[frame_slot % len].write_bytes(0, bytemuck::bytes_of(env));
    }

    pub fn buffer(&self, frame_slot: usize) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffers[frame_slot % self.buffers.len()].buffer
    }
}
