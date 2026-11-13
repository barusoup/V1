//! Engine study: drives every new Dream system through a hidden window and
//! prints structured results. No visible window, no focus, no input.
//!
//! ```text
//! cargo run --release --bin dream_study
//! ```
//!
//! Covers: UI (text, panels, widgets, menus, clipping, atlas growth), textures
//! (procedural checkerboard with mips), models (static + skinned, animated),
//! motion-vector health while the camera pans, the temporal history reset,
//! and an offline audio mix. Screenshots land in `target/dream-study/`.

use dream::audio::{Audio, AudioData, PlayParams};
use dream::camera::Camera;
use dream::gfx::{
    ColorVertex, DEPTH_FORMAT, DrawCtx, GpuBuffer, Renderer, SCENE_FORMAT, VELOCITY_FORMAT,
};
use dream::model::render::{ModelGpu, ModelRenderer};
use dream::model::{
    AnimClip, Channel, Interpolation, Key, Material, Mesh, Model, ModelInstance, Node, PlayMode,
    Skin, Target,
};
use dream::settings::{AaMode, Settings, UpscalePreset};
use dream::temporal::{MotionTracker, ObjectId};
use dream::texture::{DecodedTexture, TextureFormat, TextureManager, TextureOptions};
use dream::ui::atlas::AtlasConfig;
use dream::ui::draw::UiFrame;
use dream::ui::font::{FontId, Fonts};
use dream::ui::widgets::{
    ButtonStyle, LabelStyle, MenuItem, MenuStyle, PanelStyle, SliderStyle, Ui, UiInput,
};
use dream::ui::{Color, Rect};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event_loop::{ActiveEventLoop, EventLoop},
    window::Window,
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

struct Run;

fn checkerboard(size: u32) -> DecodedTexture {
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..size {
            let dark = ((x / 8) + (y / 8)) % 2 == 0;
            let value = if dark { 70 } else { 210 };
            let at = ((y * size + x) * 4) as usize;
            pixels[at..at + 4].copy_from_slice(&[value, value, value, 255]);
        }
    }
    DecodedTexture {
        pixels,
        width: size,
        height: size,
    }
}

/// Ground grid + a back wall, engine `ColorVertex` format.
fn scene_geometry() -> (Vec<ColorVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let quad = |vertices: &mut Vec<ColorVertex>,
                indices: &mut Vec<u32>,
                corners: [[f32; 3]; 4],
                color: [u8; 4]| {
        let base = vertices.len() as u32;
        for corner in corners {
            vertices.push(ColorVertex {
                position: corner,
                color,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    for z in -8..8i32 {
        for x in -8..8i32 {
            let checker = ((x + z).rem_euclid(2) == 0) as u8;
            let value = 90 + checker * 70;
            // Wound so the face normal points up (+y): the scene pass culls
            // back faces.
            quad(
                &mut vertices,
                &mut indices,
                [
                    [x as f32, 0.0, z as f32 + 1.0],
                    [x as f32 + 1.0, 0.0, z as f32 + 1.0],
                    [x as f32 + 1.0, 0.0, z as f32],
                    [x as f32, 0.0, z as f32],
                ],
                [value, value, value, 255],
            );
        }
    }
    quad(
        &mut vertices,
        &mut indices,
        [
            [-8.0, 0.0, -8.0],
            [8.0, 0.0, -8.0],
            [8.0, 6.0, -8.0],
            [-8.0, 6.0, -8.0],
        ],
        [120, 90, 70, 255],
    );
    (vertices, indices)
}

/// A lit unit cube, one textured material.
fn cube_model(base_color: [f32; 4], texture: DecodedTexture) -> Arc<Model> {
    let mut mesh = Mesh {
        name: "cube".into(),
        material: 0,
        ..Default::default()
    };
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [0.0, 0.0, 1.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ],
        ),
        (
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ],
        ),
    ];
    for (normal, corners) in faces {
        let base = mesh.positions.len() as u32;
        for (index, corner) in corners.into_iter().enumerate() {
            mesh.positions.push(corner);
            mesh.normals.push(normal);
            mesh.uvs.push([index as f32 % 2.0, (index / 2) as f32]);
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Arc::new(Model {
        meshes: vec![mesh],
        materials: vec![Material {
            name: "cube".into(),
            base_color,
            base_color_texture: Some(texture),
            double_sided: false,
            ..Material::default()
        }],
        nodes: vec![Node {
            name: "root".into(),
            translation: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            children: Vec::new(),
            mesh: Some(0),
            skin: None,
        }],
        skins: Vec::new(),
        clips: Vec::new(),
        roots: vec![0],
    })
}

fn skinned_strip_model() -> Arc<Model> {
    let mut mesh = Mesh {
        name: "strip".into(),
        material: 0,
        ..Default::default()
    };
    // 2 x 5 vertices from y 0..2; the upper half belongs to joint 1.
    for row in 0..5 {
        let y = row as f32 * 0.5;
        let joint = if row >= 2 { 1u16 } else { 0 };
        for x in [-0.25, 0.25] {
            mesh.positions.push([x, y, 0.0]);
            mesh.normals.push([0.0, 0.0, 1.0]);
            mesh.uvs.push([0.5, y / 2.0]);
            mesh.joints.push([joint, 0, 0, 0]);
            mesh.weights.push([1.0, 0.0, 0.0, 0.0]);
        }
    }
    for row in 0..4 {
        let base = row * 2;
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 3, base, base + 3, base + 2]);
    }
    let clip = AnimClip {
        name: "wave".into(),
        duration: 1.0,
        channels: vec![Channel {
            node: 2,
            target: Target::Translation,
            interpolation: Interpolation::Linear,
            keys: vec![
                Key {
                    time: 0.0,
                    value: [0.0, 1.0, 0.0, 0.0],
                    in_tangent: [0.0; 4],
                    out_tangent: [0.0; 4],
                },
                Key {
                    time: 0.5,
                    value: [1.0, 1.0, 0.0, 0.0],
                    in_tangent: [0.0; 4],
                    out_tangent: [0.0; 4],
                },
                Key {
                    time: 1.0,
                    value: [0.0, 1.0, 0.0, 0.0],
                    in_tangent: [0.0; 4],
                    out_tangent: [0.0; 4],
                },
            ],
        }],
    };
    Arc::new(Model {
        meshes: vec![mesh],
        materials: vec![Material {
            name: "strip".into(),
            base_color: [0.9, 0.6, 0.2, 1.0],
            base_color_texture: None,
            double_sided: true,
            ..Material::default()
        }],
        nodes: vec![
            Node {
                name: "root".into(),
                translation: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
                children: vec![1],
                mesh: Some(0),
                skin: Some(0),
            },
            Node {
                name: "joint0".into(),
                translation: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
                children: vec![2],
                mesh: None,
                skin: None,
            },
            Node {
                name: "joint1".into(),
                translation: glam::Vec3::new(0.0, 1.0, 0.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
                children: Vec::new(),
                mesh: None,
                skin: None,
            },
        ],
        skins: vec![Skin {
            joints: vec![1, 2],
            inverse_binds: vec![
                glam::Mat4::IDENTITY,
                glam::Mat4::from_translation(glam::Vec3::new(0.0, -1.0, 0.0)),
            ],
        }],
        clips: vec![clip],
        roots: vec![0],
    })
}

fn load_font() -> Fonts {
    Fonts::bundled()
}

/// One UI frame: orientation markers, HUD labels, a widget panel, a clipped
/// overflow region, and a navigation menu.
fn build_ui_frame(
    ui: &mut Ui,
    dt: f32,
    texture: &Arc<dream::texture::GpuTexture>,
    slider_value: f32,
    menu_focus: &mut usize,
) -> UiFrame {
    ui.begin(
        UiInput {
            dt,
            ..Default::default()
        },
        (WIDTH as f32, HEIGHT as f32),
    );
    let font = FontId(0);

    // Orientation markers: red across the top edge, blue across the bottom.
    ui.rect(Rect::new(0.0, 0.0, 120.0, 36.0), Color::rgb(220, 40, 40));
    ui.rect(
        Rect::new(0.0, HEIGHT as f32 - 36.0, 120.0, 36.0),
        Color::rgb(40, 80, 220),
    );

    // HUD corner labels.
    ui.label(
        Rect::new(140.0, 8.0, 600.0, 24.0),
        "Dream UI study — HUD label",
        LabelStyle::new(font).size(18.0).color(Color::WHITE),
    );

    // Centre panel with widgets.
    let panel = Rect::new(
        WIDTH as f32 * 0.5 - 260.0,
        HEIGHT as f32 * 0.5 - 190.0,
        520.0,
        380.0,
    );
    ui.panel(panel, PanelStyle::default());
    ui.label(
        Rect::new(panel.x + 24.0, panel.y + 18.0, 400.0, 28.0),
        "Main Menu",
        LabelStyle::new(font).size(28.0),
    );
    ui.button(
        "Play",
        Rect::new(panel.x + 24.0, panel.y + 64.0, 200.0, 44.0),
        ButtonStyle::new(font),
    );
    let mut toggle = true;
    ui.toggle(
        "Music",
        Rect::new(panel.x + 24.0, panel.y + 120.0, 220.0, 32.0),
        &mut toggle,
        ButtonStyle::new(font),
    );
    let mut slider = slider_value;
    ui.slider(
        "Volume",
        Rect::new(panel.x + 24.0, panel.y + 168.0, 300.0, 32.0),
        &mut slider,
        0.0..=1.0,
        SliderStyle::new(font),
    );
    ui.progress(
        Rect::new(panel.x + 24.0, panel.y + 212.0, 300.0, 14.0),
        0.65,
        Color::rgb(90, 200, 120),
    );
    ui.image(
        Rect::new(panel.x + 360.0, panel.y + 64.0, 112.0, 112.0),
        texture,
        Color::WHITE,
    );

    // Clipped scroll region with overflowing text.
    ui.draw_list_mut()
        .push_clip(Rect::new(panel.x + 24.0, panel.y + 240.0, 472.0, 60.0));
    ui.label(
        Rect::new(panel.x + 24.0, panel.y + 240.0, 472.0, 120.0),
        "Clipped region: this long line continues far past the clip rect and must be cut off cleanly at the boundary.",
        LabelStyle::new(font)
            .size(16.0)
            .color(Color::rgb(190, 200, 220)),
    );
    ui.end_clip();

    // Menu with focus navigation state (the `Quit` entry is disabled).
    let items = [
        MenuItem::new("New Game"),
        MenuItem::new("Continue"),
        MenuItem::new("Settings"),
        MenuItem::new("Quit").disabled(),
    ];
    ui.menu(
        "main",
        Rect::new(WIDTH as f32 - 320.0, 80.0, 280.0, 220.0),
        &items,
        menu_focus,
        MenuStyle::new(font),
    );

    ui.end()
}

/// Everything the scene pass draws, bundled so one helper can render a frame.
struct Scene<'a> {
    vertex_buffer: &'a GpuBuffer,
    index_buffer: &'a GpuBuffer,
    index_count: u32,
    models: &'a mut ModelRenderer,
    cube_gpu: &'a mut ModelGpu,
    strip_gpu: &'a mut ModelGpu,
    cube_instance: &'a ModelInstance,
    strip_instance: &'a ModelInstance,
}

fn render_frame(
    renderer: &mut Renderer,
    camera: &Camera,
    scene: Scene<'_>,
    ui_frame: Option<&UiFrame>,
) {
    renderer
        .render_ui(
            camera,
            None,
            |draw: &mut DrawCtx<'_>| {
                draw.bind_vertex_buffer(scene.vertex_buffer);
                draw.bind_index_buffer(scene.index_buffer);
                draw.draw_indexed(scene.index_count);
                scene.models.draw(draw, scene.cube_gpu, scene.cube_instance);
                scene
                    .models
                    .draw(draw, scene.strip_gpu, scene.strip_instance);
                3
            },
            ui_frame,
        )
        .expect("render frame");
}

impl ApplicationHandler for Run {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let out_dir = std::path::Path::new("target/dream-study");
        std::fs::create_dir_all(out_dir).expect("create study output dir");

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_visible(false)
                        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
                )
                .expect("create hidden window"),
        );
        let mut renderer = Renderer::new(
            window,
            include_str!("study_scene.wgsl"),
            Settings {
                aa: AaMode::Fsr,
                upscale: UpscalePreset::Native,
                sharpness: 0.6,
            },
        );

        // The study reads composed frames back (`capture_rgba8`, `save_png`).
        renderer.set_frame_capture(true);

        // --- Scene geometry ----------------------------------------------
        let (vertices, indices) = scene_geometry();
        let vertex_buffer = renderer
            .context()
            .create_vertex_buffer("study scene", bytemuck::cast_slice(&vertices));
        let index_buffer = renderer
            .context()
            .create_index_buffer("study scene indices", bytemuck::cast_slice(&indices));
        let index_count = indices.len() as u32;

        // --- Textures ----------------------------------------------------
        let mut textures = TextureManager::new(Arc::clone(renderer.context()));
        let (checker_id, checker) = textures.from_pixels(
            "checkerboard",
            &checkerboard(64).pixels,
            64,
            64,
            TextureOptions {
                format: TextureFormat::Srgb,
                ..Default::default()
            },
        );
        let stats = textures.stats();
        println!(
            "[study] texture id={:?} size={:?} mips={} textures={} gpu_bytes={}",
            checker_id,
            checker.size(),
            checker.mip_levels(),
            stats.textures,
            stats.gpu_bytes
        );

        // --- Models ------------------------------------------------------
        let cube = cube_model([0.85, 0.25, 0.2, 1.0], checkerboard(64));
        let strip = skinned_strip_model();
        let mut cube_gpu = ModelGpu::upload(renderer.context(), &cube);
        let mut strip_gpu = ModelGpu::upload(renderer.context(), &strip);
        let mut cube_instance = ModelInstance::new(Arc::clone(&cube));
        let mut strip_instance = ModelInstance::new(Arc::clone(&strip));
        strip_instance
            .player_mut()
            .play(Arc::new(strip.clips[0].clone()), PlayMode::Loop, 0.0);
        let mut models = ModelRenderer::new(
            renderer.context(),
            &[SCENE_FORMAT, VELOCITY_FORMAT],
            DEPTH_FORMAT,
        );
        let mut tracker = MotionTracker::new();

        // --- UI ----------------------------------------------------------
        let fonts = load_font();
        let mut ui = Ui::new(fonts, AtlasConfig::default());
        let mut menu_focus = 0usize;

        // --- Audio (offline, silent) -------------------------------------
        let audio = Audio::offline(48_000);
        let samples: Vec<f32> = (0..48_000 / 4)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.4)
            .flat_map(|s| [s, s])
            .collect();
        let clip = Arc::new(AudioData {
            samples,
            sample_rate: 48_000,
            duration: 0.25,
        });
        audio.play(clip, PlayParams::default());
        let mut mixed = vec![0.0f32; 48_000 / 4 * 2];
        audio.render_offline(&mut mixed);
        let peak = mixed.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.1, "offline mix produced no audio (peak {peak})");
        println!(
            "[study] audio device={} peak={:.3} active_voices={}",
            audio.device_name(),
            peak,
            audio.mixer().lock().unwrap().active_voices()
        );

        // --- Camera ------------------------------------------------------
        let mut camera = Camera::default();
        camera.set_aspect(WIDTH as f32 / HEIGHT as f32);
        camera.set_view_distance(16);
        camera.position = glam::Vec3::new(0.0, 3.0, 9.0);
        camera.set_look(-std::f32::consts::FRAC_PI_2, -0.18);

        let mut frame_index = 0u32;

        // Warm up with the camera at rest (FSR history converges).
        for _ in 0..32 {
            cube_instance.set_transform(glam::Mat4::from_translation(glam::Vec3::new(
                -2.0, 1.0, 3.0,
            )));
            strip_instance
                .set_transform(glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.5, 3.0)));
            cube_instance.update(1.0 / 60.0);
            strip_instance.update(1.0 / 60.0);
            tracker.update(ObjectId(1), cube_instance.transform());
            tracker.update(ObjectId(2), strip_instance.transform());
            let slider_value = 0.5 + 0.25 * (frame_index as f32 * 0.05).sin();
            let ui_frame =
                build_ui_frame(&mut ui, 1.0 / 60.0, &checker, slider_value, &mut menu_focus);
            render_frame(
                &mut renderer,
                &camera,
                Scene {
                    vertex_buffer: &vertex_buffer,
                    index_buffer: &index_buffer,
                    index_count,
                    models: &mut models,
                    cube_gpu: &mut cube_gpu,
                    strip_gpu: &mut strip_gpu,
                    cube_instance: &cube_instance,
                    strip_instance: &strip_instance,
                },
                Some(&ui_frame),
            );
            frame_index += 1;
        }

        // Screenshot with UI + models.
        let (pixels, width, height) = renderer.capture_rgba8();
        dream::png::write(&out_dir.join("ui-models.png"), width, height, &pixels)
            .expect("write screenshot");
        let sample = |x: usize, y: usize| {
            let at = (y * width as usize + x) * 4;
            (pixels[at], pixels[at + 1], pixels[at + 2])
        };
        let top = sample(60, 18);
        let bottom = sample(60, height as usize - 18);
        let top_ok = top.0 > 150 && top.1 < 120;
        let bottom_ok = bottom.2 > 150 && bottom.0 < 120;
        println!(
            "[study] orientation top={:?} bottom={:?} top_ok={} bottom_ok={}",
            top, bottom, top_ok, bottom_ok
        );
        assert!(top_ok && bottom_ok, "display readback orientation wrong");
        println!(
            "[study] ui frames_drawn={} atlas_pages={} rasterizations={} tracked_objects={}",
            frame_index,
            ui.cache().atlas().page_count(),
            ui.cache().rasterizations(),
            tracker.len()
        );

        // Motion-vector health while panning.
        for _ in 0..12 {
            camera.set_look(camera.yaw() + 3.0f32.to_radians(), camera.pitch());
            cube_instance.set_transform(glam::Mat4::from_translation(glam::Vec3::new(
                -2.0, 1.0, 3.0,
            )));
            strip_instance
                .set_transform(glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.5, 3.0)));
            cube_instance.update(1.0 / 60.0);
            strip_instance.update(1.0 / 60.0);
            let slider_value = 0.5 + 0.25 * (frame_index as f32 * 0.05).sin();
            let ui_frame =
                build_ui_frame(&mut ui, 1.0 / 60.0, &checker, slider_value, &mut menu_focus);
            render_frame(
                &mut renderer,
                &camera,
                Scene {
                    vertex_buffer: &vertex_buffer,
                    index_buffer: &index_buffer,
                    index_count,
                    models: &mut models,
                    cube_gpu: &mut cube_gpu,
                    strip_gpu: &mut strip_gpu,
                    cube_instance: &cube_instance,
                    strip_instance: &strip_instance,
                },
                Some(&ui_frame),
            );
            frame_index += 1;
        }
        let metrics = renderer.frame_metrics();
        println!("[study] motion {}", metrics.summary());
        // Sky belongs above the horizon: the upper half must be sky-heavier
        // than the lower half. Equal halves means an upside-down render.
        assert!(
            metrics.sky_top_pct > metrics.sky_bottom_pct,
            "sky concentrated in the lower half: {}",
            metrics.summary()
        );
        // Nonzero motion is bounded below by geometry coverage (sky pixels get
        // no vectors) and above by sanity: a pan must move things, and nothing
        // may jump a quarter screen in one frame.
        let motion_ok =
            metrics.mv_nonzero_pct > 5.0 && metrics.mv_max > 0.01 && metrics.mv_large_pct < 1.0;
        println!(
            "[study] motion_ok={} (mv_nonzero={:.1}% mv_max={:.4} mv_large={:.2}%)",
            motion_ok, metrics.mv_nonzero_pct, metrics.mv_max, metrics.mv_large_pct
        );
        assert!(motion_ok, "motion vectors unhealthy: {}", metrics.summary());

        // Temporal history reset then one frame (the invalid-history path).
        renderer.reset_history();
        assert!(!renderer.history_valid(), "reset invalidates history");
        let ui_frame = build_ui_frame(&mut ui, 1.0 / 60.0, &checker, 0.2, &mut menu_focus);
        render_frame(
            &mut renderer,
            &camera,
            Scene {
                vertex_buffer: &vertex_buffer,
                index_buffer: &index_buffer,
                index_count,
                models: &mut models,
                cube_gpu: &mut cube_gpu,
                strip_gpu: &mut strip_gpu,
                cube_instance: &cube_instance,
                strip_instance: &strip_instance,
            },
            Some(&ui_frame),
        );
        assert!(
            renderer.history_valid(),
            "history recovers after a reset frame"
        );
        renderer
            .save_png(&out_dir.join("after-reset.png"))
            .expect("write screenshot");

        // FSR off control: direct blit, UI still composited.
        renderer.toggle_temporal();
        let ui_frame = build_ui_frame(&mut ui, 1.0 / 60.0, &checker, 0.2, &mut menu_focus);
        render_frame(
            &mut renderer,
            &camera,
            Scene {
                vertex_buffer: &vertex_buffer,
                index_buffer: &index_buffer,
                index_count,
                models: &mut models,
                cube_gpu: &mut cube_gpu,
                strip_gpu: &mut strip_gpu,
                cube_instance: &cube_instance,
                strip_instance: &strip_instance,
            },
            Some(&ui_frame),
        );
        renderer
            .save_png(&out_dir.join("direct.png"))
            .expect("write screenshot");

        println!(
            "[study] model_draws={} meshes={} clip={}",
            models.draws(),
            cube.meshes.len() + strip.meshes.len(),
            strip.clips[0].summary()
        );

        println!("[study] done screenshots={}", out_dir.display());

        #[cfg(target_os = "macos")]
        renderer.context().wait_idle();
        #[cfg(not(target_os = "macos"))]
        unsafe {
            renderer.context().device().device_wait_idle().unwrap();
        }
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
    }
}

fn main() {
    EventLoop::new()
        .expect("create event loop")
        .run_app(&mut Run)
        .expect("run study");
}
