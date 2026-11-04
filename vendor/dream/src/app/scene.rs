//! The app's viewport: a flat neutral canvas, an orbit camera, and — only
//! once the user asks for it — a small debug scene (checkered floor, boxes,
//! spinning hero).
//!
//! Booting into a debug scene is opt-in: the app boots into the scene manager page
//! in the main pane. The debug scene is booted on demand from the scene manager
//! and unloaded the same way.
//!
//! Scenes are independent of the editor's theme: every colour here is
//! fixed. The theme only styles the editor's own UI.
//!
//! All geometry is CPU data streamed through the engine's vertex path (the
//! editor must run from a bare checkout).

use crate::camera::Camera;
use crate::gfx::vulkan::DynamicBuffers;
use crate::gfx::{ColorVertex, DrawCtx, Renderer};
use crate::ui::Color;
use glam::Vec3;
use std::f32::consts::TAU;
use std::sync::Arc;

/// One box in the debug scene the inspector-adjacent tooling can select.
#[derive(Clone, Debug)]
pub struct SceneObject {
    pub name: &'static str,
    pub position: Vec3,
    pub size: f32,
    pub color: Color,
}

/// Neutral dark grey behind every scene (monochrome, no hue).
const CANVAS: Color = Color::rgb(0x1e, 0x1e, 0x1e);
/// Debug floor squares.
const FLOOR_LIGHT: Color = Color::WHITE;
const FLOOR_DARK: Color = Color::BLACK;
/// Debug box colours (fixed, so they never follow the editor theme).
const CRATE_COLOR: Color = Color::rgb(0x2f, 0x80, 0xed);
const SHELL_COLOR: Color = Color::rgb(0x43, 0xc7, 0x5a);
const SLAB_COLOR: Color = Color::rgb(0xff, 0x9f, 0x1c);
const HERO_COLOR: Color = Color::rgb(0x9b, 0x6b, 0xd6);

/// The manually-loaded debug scene: a checker floor, three boxes and a
/// spinning hero cube.
pub struct DebugScene {
    /// Floor and static boxes.
    statics: DynamicBuffers,
    statics_index_count: u32,
    /// The spinning hero box, rebuilt each animated frame.
    hero: DynamicBuffers,
    hero_index_count: u32,
    pub objects: Vec<SceneObject>,
    pub hero_spin: f32,
}

/// Compatibility alias for the debug scene.
pub type DemoScene = DebugScene;

/// Orbit state driving the engine camera.
#[derive(Clone, Debug, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Where `distance` is easing toward; wheel input moves this, not
    /// `distance`, so each notch glides instead of snapping.
    pub zoom_target: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// Exponential ease rate for zoom (per second); higher is snappier.
const ZOOM_EASE_RATE: f32 = 20.0;

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 1.0, 0.0),
            distance: 18.0,
            zoom_target: 18.0,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.32,
        }
    }
}

impl OrbitCamera {
    /// Camera position for the current orbit angles. Uses the *engine*
    /// camera's yaw/pitch convention (see `Camera::forward`): the eye sits
    /// `distance` behind the target along the view direction.
    pub fn position(&self) -> Vec3 {
        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        self.target - forward * self.distance
    }

    /// Apply to the engine camera; the engine's far plane must reach the
    /// scene (a 16-chunk view distance gives 614 units, plenty).
    pub fn apply(&self, camera: &mut Camera) {
        camera.position = self.position();
        camera.set_look(self.yaw, self.pitch);
        camera.set_view_distance(16);
    }

    /// Orbit with the grab-the-world metaphor: dragging right swings the
    /// camera left around the target, so the scene follows the cursor.
    /// (The engine's camera azimuth is `yaw`, so a rightward drag raises
    /// it — the eye moves to the viewer's left.)
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.008;
        self.pitch = (self.pitch - dy * 0.008).clamp(-1.5, 1.5);
    }

    /// Wheel up zooms in (the standard scrollwheel convention). Only the
    /// target moves here; `ease_zoom` glides the distance toward it.
    pub fn zoom(&mut self, delta: f32) {
        self.zoom_target = (self.zoom_target * (1.0 - delta * 0.06)).clamp(4.0, 90.0);
    }

    /// Frame-rate independent exponential approach of `distance` toward
    /// `zoom_target`. Snaps once the remaining gap is imperceptible.
    pub fn ease_zoom(&mut self, dt: f32) {
        let gap = self.zoom_target - self.distance;
        if gap.abs() < 0.001 {
            self.distance = self.zoom_target;
        } else {
            self.distance += gap * (1.0 - (-ZOOM_EASE_RATE * dt).exp());
        }
    }

    /// True while the distance is still gliding toward its target.
    pub fn zooming(&self) -> bool {
        self.distance != self.zoom_target
    }

    /// Pan follows the drag exactly, in the camera's own screen plane: the
    /// world point under the cursor tracks the cursor. Both axes are
    /// camera-relative — `right` is the horizontal screen axis and `up` is
    /// the screen-up axis (which tilts backward as you look down), so
    /// panning up at an angle moves you up along your view, not along the
    /// world Y axis. Dragging pulls the look-at target against both.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        // cross(forward, world_up), de-homogenised; normalise away the
        // pitch foreshortening.
        let right = Vec3::new(-forward.z, 0.0, forward.x).normalize_or_zero();
        let up = right.cross(forward);
        self.target += right * (-dx * 0.02) + up * (dy * 0.02);
    }
}

pub struct Scene {
    /// Flat backdrop behind everything (screen space).
    canvas: DynamicBuffers,
    /// The manually-loaded debug scene (None until the user asks for it).
    pub debug: Option<DebugScene>,
    pub orbit: OrbitCamera,
    /// Draw calls submitted on the last frame.
    pub draws: u32,
    pub quads: u32,
}

impl Scene {
    pub fn new(renderer: &Renderer) -> Self {
        let ctx = renderer.context();
        let mut canvas = DynamicBuffers::new(
            ctx,
            "canvas",
            (4 * std::mem::size_of::<ColorVertex>()) as u64,
            (6 * 4) as u64,
        );
        let (vertices, indices) = canvas_geometry();
        canvas.update(
            bytemuck::cast_slice(&vertices),
            bytemuck::cast_slice(&indices),
        );
        Self {
            canvas,
            debug: None,
            orbit: OrbitCamera::default(),
            draws: 0,
            quads: 0,
        }
    }

    /// Whether the debug scene is loaded.
    pub fn debug_loaded(&self) -> bool {
        self.debug.is_some()
    }

    /// Compatibility alias for `debug_loaded`.
    pub fn demo_loaded(&self) -> bool {
        self.debug_loaded()
    }

    /// Objects in the debug scene, when it is loaded.
    pub fn objects(&self) -> Option<&[SceneObject]> {
        self.debug.as_ref().map(|debug| debug.objects.as_slice())
    }

    pub fn set_object_color(&mut self, index: usize, color: Color) {
        if let Some(debug) = self.debug.as_mut()
            && let Some(object) = debug.objects.get_mut(index)
        {
            object.color = color;
            debug.rebuild_hero();
        }
    }

    /// Load the debug scene. Idempotent.
    pub fn load_debug(&mut self) {
        if self.debug.is_some() {
            return;
        }
        let ctx = Arc::clone(self.canvas_context());
        let (vertices, indices) = debug_static_geometry();
        let mut statics = DynamicBuffers::new(
            &ctx,
            "debug",
            vertices.len() as u64 * std::mem::size_of::<ColorVertex>() as u64,
            indices.len() as u64 * 4,
        );
        statics.update(
            bytemuck::cast_slice(&vertices),
            bytemuck::cast_slice(&indices),
        );
        let mut debug = DebugScene {
            statics,
            statics_index_count: indices.len() as u32,
            hero: DynamicBuffers::new(&ctx, "debug hero", 36 * 16, 36 * 4),
            hero_index_count: 0,
            objects: vec![
                SceneObject::new("crate", Vec3::new(-3.5, 0.75, 1.5), 1.5, CRATE_COLOR),
                SceneObject::new("shell", Vec3::new(3.0, 0.6, -1.0), 1.2, SHELL_COLOR),
                SceneObject::new("slab", Vec3::new(0.5, 0.4, 3.5), 0.8, SLAB_COLOR),
                SceneObject::new("hero", Vec3::new(0.0, 1.6, 0.0), 1.6, HERO_COLOR),
            ],
            hero_spin: 0.0,
        };
        debug.rebuild_hero();
        self.debug = Some(debug);
    }

    /// Compatibility alias for `load_debug`.
    pub fn load_demo(&mut self) {
        self.load_debug();
    }

    /// Unload the debug scene; the viewport returns to the canvas and camera.
    pub fn unload_debug(&mut self) {
        self.debug = None;
    }

    /// Compatibility alias for `unload_debug`.
    pub fn unload_demo(&mut self) {
        self.unload_debug();
    }

    /// Advance ambient animation (the debug scene's spinning hero cube).
    pub fn advance(&mut self, spin_dt: f32) {
        if let Some(debug) = self.debug.as_mut() {
            debug.hero_spin = (debug.hero_spin + spin_dt) % TAU;
            debug.rebuild_hero();
        }
    }

    /// Draw the viewport: canvas, then the debug scene when it is loaded.
    pub fn draw(&mut self, draw: &mut DrawCtx<'_>) {
        self.draws = 0;
        self.quads = 0;
        draw.bind_vertex_buffer(self.canvas.vertices());
        draw.bind_index_buffer(self.canvas.indices());
        draw.draw_indexed(6);
        self.draws += 1;
        self.quads += 1;
        if let Some(debug) = self.debug.as_mut() {
            draw.bind_vertex_buffer(debug.statics.vertices());
            draw.bind_index_buffer(debug.statics.indices());
            draw.draw_indexed(debug.statics_index_count);
            self.draws += 1;
            self.quads += debug.statics_index_count / 6;
            draw.bind_vertex_buffer(debug.hero.vertices());
            draw.bind_index_buffer(debug.hero.indices());
            if debug.hero_index_count > 0 {
                draw.draw_indexed(debug.hero_index_count);
                self.draws += 1;
                self.quads += debug.hero_index_count / 6;
            }
        }
    }

    /// The GPU context, for building debug geometry on demand.
    fn canvas_context(&self) -> &Arc<crate::gfx::vulkan::GpuContext> {
        // The canvas buffers hold the only long-lived context reference; the
        // renderer's context outlives the scene, so borrow through it.
        self.canvas.context()
    }
}

impl SceneObject {
    fn new(name: &'static str, position: Vec3, size: f32, color: Color) -> Self {
        Self {
            name,
            position,
            size,
            color,
        }
    }
}

impl DebugScene {
    /// Rebuild the spinning hero box (36 vertices; trivial on the CPU).
    fn rebuild_hero(&mut self) {
        let hero = &self.objects[3];
        let rotation = glam::Mat4::from_rotation_y(self.hero_spin)
            * glam::Mat4::from_rotation_x(self.hero_spin * 0.35);
        let (vertices, indices) =
            box_geometry(hero.position, hero.size, hero.color, Some(rotation));
        self.hero.update(
            bytemuck::cast_slice(&vertices),
            bytemuck::cast_slice(&indices),
        );
        self.hero_index_count = indices.len() as u32;
    }
}

/// The canvas: one flat screen-space quad in the neutral backdrop colour.
/// Static — the engine's clear colour is replaced by this fill.
fn canvas_geometry() -> (Vec<ColorVertex>, Vec<u32>) {
    // NDC corners, wound CCW in NDC (the scene viewport flips y, which
    // makes CCW-in-NDC front-facing).
    let color = CANVAS.to_array();
    let ndc = |nx: f32, ny: f32| ColorVertex {
        position: [1.0e30, nx, ny],
        color,
    };
    let vertices = vec![
        ndc(-1.0, 1.0),
        ndc(-1.0, -1.0),
        ndc(1.0, -1.0),
        ndc(1.0, 1.0),
    ];
    (vertices, vec![0, 1, 2, 0, 2, 3])
}

/// The debug scene's world geometry: a black and white checkerboard floor plus
/// three boxes, wound so back-face culling keeps the camera-facing sides.
pub fn debug_static_geometry() -> (Vec<ColorVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let half = 16i32;
    let light = FLOOR_LIGHT;
    let dark = FLOOR_DARK;
    for z in -half..half {
        for x in -half..half {
            let checker = ((x + z).rem_euclid(2) == 0) as usize;
            let color = [light, dark][checker];
            quad(
                &mut vertices,
                &mut indices,
                [
                    [x as f32, 0.0, z as f32 + 1.0],
                    [x as f32 + 1.0, 0.0, z as f32 + 1.0],
                    [x as f32 + 1.0, 0.0, z as f32],
                    [x as f32, 0.0, z as f32],
                ],
                color.to_array(),
            );
        }
    }
    for (center, size, color) in [
        (Vec3::new(-3.5, 0.75, 1.5), 1.5, CRATE_COLOR),
        (Vec3::new(3.0, 0.6, -1.0), 1.2, SHELL_COLOR),
        (Vec3::new(0.5, 0.4, 3.5), 0.8, SLAB_COLOR),
    ] {
        let (box_vertices, box_indices) = box_geometry(center, size, color, None);
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&box_vertices);
        indices.extend(box_indices.iter().map(|i| base + i));
    }
    (vertices, indices)
}

/// Compatibility alias for `debug_static_geometry`.
pub fn demo_static_geometry() -> (Vec<ColorVertex>, Vec<u32>) {
    debug_static_geometry()
}

/// A box with fake per-face lambert shading (flat vertex colours, no
/// lighting in the shader). `spin` rotates the box about its own centre.
pub fn box_geometry(
    center: Vec3,
    size: f32,
    color: Color,
    spin: Option<glam::Mat4>,
) -> (Vec<ColorVertex>, Vec<u32>) {
    let half = size * 0.5;
    // (normal, shade, corners wound CW as seen from outside the box)
    let faces: [([f32; 3], f32, [[f32; 3]; 4]); 6] = [
        (
            [0.0, 0.0, 1.0],
            1.0,
            [
                [-half, -half, half],
                [half, -half, half],
                [half, half, half],
                [-half, half, half],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            0.62,
            [
                [half, -half, -half],
                [-half, -half, -half],
                [-half, half, -half],
                [half, half, -half],
            ],
        ),
        (
            [1.0, 0.0, 0.0],
            0.82,
            [
                [half, -half, half],
                [half, -half, -half],
                [half, half, -half],
                [half, half, half],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            0.7,
            [
                [-half, -half, -half],
                [-half, -half, half],
                [-half, half, half],
                [-half, half, -half],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            1.12,
            [
                [-half, half, half],
                [half, half, half],
                [half, half, -half],
                [-half, half, -half],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            0.45,
            [
                [-half, -half, -half],
                [half, -half, -half],
                [half, -half, half],
                [-half, -half, half],
            ],
        ),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (_, shade, corners) in faces {
        let shaded = crate::ui::theme::mix(color, Color::BLACK, (shade - 1.0).max(0.0) * 0.55);
        let shaded = if shade > 1.0 {
            crate::ui::theme::mix(color, Color::WHITE, (shade - 1.0) * 0.6)
        } else {
            shaded
        };
        let base = vertices.len() as u32;
        for corner in corners {
            let local = Vec3::from(corner);
            let point = match spin {
                Some(rotation) => center + rotation.transform_point3(local),
                None => center + local,
            };
            vertices.push(ColorVertex {
                position: point.to_array(),
                color: shaded.to_array(),
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

fn quad(
    vertices: &mut Vec<ColorVertex>,
    indices: &mut Vec<u32>,
    corners: [[f32; 3]; 4],
    color: [u8; 4],
) {
    let base = vertices.len() as u32;
    for corner in corners {
        vertices.push(ColorVertex {
            position: corner,
            color,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}
