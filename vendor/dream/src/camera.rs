use glam::{Mat4, Vec3};
use std::collections::HashSet;
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

pub const DEFAULT_VIEW_DISTANCE_CHUNKS: i32 = 96;
pub const UNITS_PER_CHUNK: f32 = 32.0;

pub struct Camera {
    pub position: Vec3,
    view_distance_chunks: i32,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    keys: HashSet<KeyCode>,
    caps_boost: bool,
    aspect: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 44.0, 70.0),
            view_distance_chunks: DEFAULT_VIEW_DISTANCE_CHUNKS,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.28,
            velocity: Vec3::ZERO,
            keys: HashSet::new(),
            caps_boost: false,
            aspect: 16.0 / 9.0,
        }
    }
}

impl Camera {
    const FOV_Y: f32 = 70_f32.to_radians();

    /// Vertical field of view in radians.
    pub fn fov_y(&self) -> f32 {
        Self::FOV_Y
    }

    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn view_distance(&self) -> i32 {
        self.view_distance_chunks
    }

    pub fn set_view_distance(&mut self, chunks: i32) {
        self.view_distance_chunks = chunks.max(1);
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect.max(0.01);
    }

    pub fn key(&mut self, input: &KeyEvent) {
        let PhysicalKey::Code(code) = input.physical_key else {
            return;
        };
        if code == KeyCode::CapsLock {
            // Caps Lock is an OS-level toggle: flip our boost on each fresh press.
            if input.state == ElementState::Pressed && !input.repeat {
                self.caps_boost = !self.caps_boost;
            }
            return;
        }
        self.set_key(code, input.state == ElementState::Pressed);
    }

    /// Press or release a key without a window event, so scripted capture runs
    /// can drive the camera without injecting OS input.
    pub fn set_key(&mut self, code: KeyCode, down: bool) {
        if code == KeyCode::CapsLock {
            return;
        }
        if down {
            self.keys.insert(code);
        } else {
            self.keys.remove(&code);
        }
    }

    /// Point the camera directly, in the same units [`Self::yaw`] and
    /// [`Self::pitch`] report. Scripted capture uses this for deterministic poses.
    pub fn set_look(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-1.54, 1.54);
    }

    pub fn clear_keys(&mut self) {
        self.keys.clear();
        self.velocity = Vec3::ZERO;
    }

    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    pub fn key_down(&self, code: KeyCode) -> bool {
        self.keys.contains(&code)
    }

    pub fn caps_boost(&self) -> bool {
        self.caps_boost
    }

    pub fn mouse_motion(&mut self, delta: (f64, f64), captured: bool) {
        if !captured {
            return;
        }
        // Mouse X grows right; positive yaw turns the initial -Z view toward +X.
        self.yaw += delta.0 as f32 * 0.0025;
        self.pitch = (self.pitch - delta.1 as f32 * 0.0025).clamp(-1.54, 1.54);
    }

    pub fn device_event(&mut self, event: &DeviceEvent, captured: bool) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_motion(*delta, captured);
        }
    }

    pub fn update_fly(&mut self, dt: f32) {
        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let mut wish = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            wish += forward;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            wish -= forward;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            wish += right;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            wish -= right;
        }
        if self.keys.contains(&KeyCode::Space) {
            wish += Vec3::Y;
        }
        if self.keys.contains(&KeyCode::ControlLeft) || self.keys.contains(&KeyCode::ControlRight) {
            wish -= Vec3::Y;
        }
        if wish.length_squared() > 0.0 {
            wish = wish.normalize();
        }
        let boost = if self.keys.contains(&KeyCode::ShiftLeft)
            || self.keys.contains(&KeyCode::ShiftRight)
        {
            3.0
        } else {
            1.0
        };
        let caps = if self.caps_boost { 10.0 } else { 1.0 };
        let target = wish * (28.0 * boost * caps);
        let response = if wish == Vec3::ZERO { 5.0 } else { 9.0 };
        self.velocity = self.velocity.lerp(target, (response * dt).min(1.0));
        self.position += self.velocity * dt;
    }

    pub fn clip_planes(&self) -> (f32, f32) {
        (
            0.1,
            self.view_distance_chunks as f32 * UNITS_PER_CHUNK * 1.2,
        )
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    pub fn view_projection(&self) -> Mat4 {
        let (near, far) = self.clip_planes();
        Mat4::perspective_rh(Self::FOV_Y, self.aspect, near, far) * self.view()
    }

    pub fn sphere_visible(&self, center: Vec3, radius: f32) -> bool {
        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let offset = center - self.position;
        let depth = offset.dot(forward);
        let (near, far) = self.clip_planes();
        if depth + radius < near || depth - radius > far {
            return false;
        }
        let tan_v = (Self::FOV_Y * 0.5).tan();
        let tan_h = tan_v * self.aspect;
        let depth = depth.max(0.0);
        offset.dot(right).abs() <= depth * tan_h + radius * tan_h.hypot(1.0)
            && offset.dot(up).abs() <= depth * tan_v + radius * tan_v.hypot(1.0)
    }

    pub fn jittered_projection(&self, jitter_pixels: [f32; 2], size_pixels: [f32; 2]) -> Mat4 {
        let (near, far) = self.clip_planes();
        let mut projection = Mat4::perspective_rh(Self::FOV_Y, self.aspect, near, far);
        projection.z_axis.x += 2.0 * jitter_pixels[0] / size_pixels[0].max(1.0);
        projection.z_axis.y += 2.0 * jitter_pixels[1] / size_pixels[1].max(1.0);
        projection
    }

    pub fn jittered_view_projection(&self, jitter_pixels: [f32; 2], size_pixels: [f32; 2]) -> Mat4 {
        self.jittered_projection(jitter_pixels, size_pixels) * self.view()
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }
}
