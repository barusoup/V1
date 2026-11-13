//! Native qualification for baked-asset shipping and multi-asset draw safety.
//! Run with a fresh output directory, then repeat with `--load-only`.
use dream::camera::Camera;
use dream::gfx::{DEPTH_FORMAT, Renderer, SCENE_FORMAT, VELOCITY_FORMAT};
use dream::impostor::{
    AssetDebug, BakeSettings, ImpostorAsset, ImpostorAssetLimits, ImpostorBaker, ImpostorInstance,
    ImpostorRenderer, ImpostorSource,
};
use dream::lighting::{Lighting, LightingEnv};
use dream::model::render::{ModelGpu, ModelRenderer};
use dream::model::{Material, Mesh, Model, ModelInstance};
use dream::settings::{AaMode, Settings, UpscalePreset};
use glam::{Mat4, Vec3};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

const W: u32 = 1280;
const H: u32 = 800;
const RESIDENT_LIMIT: u64 = 1280 * 1024 * 1024;
const SCENE: &str = r#"
struct In { @location(0) p: vec3<f32>, @location(1) c: vec4<f32> };
struct V { @builtin(position) p: vec4<f32>, @location(0) c: vec4<f32> };
struct F { @location(0) c: vec4<f32>, @location(1) v: vec2<f32> };
@vertex fn vs_main(i: In) -> V { return V(vec4<f32>(i.p, 1.0), i.c); }
@fragment fn fs_main(i: V) -> F { return F(i.c, vec2<f32>(0.0)); }
"#;

fn add_box(m: &mut Model, lo: Vec3, hi: Vec3, material: usize) {
    let p = |x: bool, y: bool, z: bool| {
        Vec3::new(
            if x { hi.x } else { lo.x },
            if y { hi.y } else { lo.y },
            if z { hi.z } else { lo.z },
        )
    };
    let faces = [
        (
            [
                p(false, false, true),
                p(true, false, true),
                p(true, true, true),
                p(false, true, true),
            ],
            Vec3::Z,
        ),
        (
            [
                p(true, false, false),
                p(false, false, false),
                p(false, true, false),
                p(true, true, false),
            ],
            Vec3::NEG_Z,
        ),
        (
            [
                p(true, false, true),
                p(true, false, false),
                p(true, true, false),
                p(true, true, true),
            ],
            Vec3::X,
        ),
        (
            [
                p(false, false, false),
                p(false, false, true),
                p(false, true, true),
                p(false, true, false),
            ],
            Vec3::NEG_X,
        ),
        (
            [
                p(false, true, true),
                p(true, true, true),
                p(true, true, false),
                p(false, true, false),
            ],
            Vec3::Y,
        ),
        (
            [
                p(false, false, false),
                p(true, false, false),
                p(true, false, true),
                p(false, false, true),
            ],
            Vec3::NEG_Y,
        ),
    ];
    let mut mesh = Mesh {
        material,
        ..Mesh::default()
    };
    for (quad, n) in faces {
        let base = mesh.positions.len() as u32;
        let t = (quad[1] - quad[0]).normalize();
        for (q, uv) in quad
            .into_iter()
            .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        {
            mesh.positions.push(q.to_array());
            mesh.normals.push(n.to_array());
            mesh.uvs.push(uv);
            mesh.tangents.push(t.extend(1.0).to_array());
        }
        mesh.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    m.meshes.push(mesh);
}

fn fixture(id: usize) -> Arc<Model> {
    let mut m = Model::default();
    m.materials = vec![
        Material {
            base_color: [0.16 + id as f32 * 0.025, 0.35, 0.12, 1.0],
            metallic_factor: 0.0,
            roughness_factor: 0.85,
            ..Material::default()
        },
        Material {
            base_color: [0.65, 0.12 + id as f32 * 0.015, 0.25, 1.0],
            metallic_factor: 0.0,
            roughness_factor: 0.65,
            ..Material::default()
        },
    ];
    match id % 4 {
        0 => {
            add_box(
                &mut m,
                Vec3::new(-0.8, 0.0, -0.6),
                Vec3::new(0.8, 1.7, 0.6),
                0,
            );
            add_box(
                &mut m,
                Vec3::new(-1.0, 1.7, -0.8),
                Vec3::new(1.0, 2.0, 0.8),
                1,
            );
        }
        1 => {
            add_box(
                &mut m,
                Vec3::new(-1.0, 0.0, -0.25),
                Vec3::new(-0.65, 2.0, 0.25),
                0,
            );
            add_box(
                &mut m,
                Vec3::new(0.65, 0.0, -0.25),
                Vec3::new(1.0, 2.0, 0.25),
                1,
            );
            add_box(
                &mut m,
                Vec3::new(-0.65, 1.65, -0.25),
                Vec3::new(0.65, 2.0, 0.25),
                0,
            );
        }
        2 => {
            for i in 0..5 {
                let x = -0.95 + i as f32 * 0.45;
                add_box(
                    &mut m,
                    Vec3::new(x, 0.0, -0.08),
                    Vec3::new(x + 0.10, 2.0, 0.08),
                    i % 2,
                );
            }
            add_box(
                &mut m,
                Vec3::new(-1.0, 0.5, -0.09),
                Vec3::new(1.0, 0.62, 0.09),
                1,
            );
        }
        _ => {
            add_box(
                &mut m,
                Vec3::new(-1.2, 0.0, -0.5),
                Vec3::new(-0.5, 2.0, 0.5),
                0,
            );
            add_box(
                &mut m,
                Vec3::new(0.5, 0.0, -0.5),
                Vec3::new(1.2, 1.5, 0.5),
                1,
            );
        }
    }
    if id == 14 {
        let tex = dream::texture::DecodedTexture {
            width: 2,
            height: 2,
            pixels: [0, 50, 100, 255]
                .into_iter()
                .flat_map(|a| [200, 120, 30, a])
                .collect(),
        };
        for material in &mut m.materials {
            material.base_color[3] = 0.25;
            material.base_color_texture = Some(tex.clone());
            material.base_color_map = Some(dream::model::TextureRef {
                texture: tex.clone(),
                tex_coord: 0,
                transform: None,
                wrap_s: dream::model::Wrap::Repeat,
                wrap_t: dream::model::Wrap::Repeat,
                mag_filter: dream::model::TexelFilter::Linear,
                min_filter: dream::model::TexelFilter::Linear,
            });
        }
    }
    m.nodes = vec![dream::model::Node {
        name: "fixture-root".into(),
        translation: Vec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        scale: Vec3::ONE,
        children: Vec::new(),
        mesh: Some(0),
        skin: None,
    }];
    m.roots = vec![0];
    for mesh in &mut m.meshes {
        mesh.node = Some(0);
    }
    Arc::new(m)
}

struct Item {
    model: Arc<Model>,
    mesh: ModelGpu,
    asset: ImpostorAsset,
    normalization: Mat4,
}

struct Check {
    renderer: Renderer,
    lighting: Lighting,
    models: ModelRenderer,
    cards: ImpostorRenderer,
    camera: Camera,
    items: Vec<Item>,
    mesh_batches: Vec<Vec<ModelInstance>>,
}

impl Check {
    fn prepare_meshes(&mut self, batches: &[(usize, Vec<ImpostorInstance>)]) {
        self.mesh_batches = batches
            .iter()
            .map(|(id, instances)| {
                instances
                    .iter()
                    .map(|instance| {
                        let mut mesh = ModelInstance::new(self.items[*id].model.clone());
                        mesh.set_transform(Mat4::from_cols_array_2d(&instance.prev_model));
                        mesh.update(0.0);
                        mesh.set_transform(Mat4::from_cols_array_2d(&instance.model));
                        mesh.update(0.0);
                        mesh
                    })
                    .collect()
            })
            .collect();
    }

    fn frame(&mut self, cards: bool, batches: &[(usize, Vec<ImpostorInstance>)], reverse: bool) {
        let camera = &self.camera;
        let env = LightingEnv {
            cam_pos: camera.position.extend(0.0).to_array(),
            ..LightingEnv::default()
        };
        let Check {
            renderer,
            lighting,
            models,
            cards: card_renderer,
            items,
            mesh_batches,
            ..
        } = self;
        renderer
            .render(camera, |draw| {
                lighting.upload(draw.frame_slot(), &env);
                let mut draws = 0;
                for j in 0..batches.len() {
                    let i = if reverse { batches.len() - 1 - j } else { j };
                    let (asset_id, instances) = &batches[i];
                    let item = &mut items[*asset_id];
                    if cards {
                        card_renderer
                            .try_draw(
                                draw,
                                &mut item.asset,
                                instances,
                                lighting,
                                H as f32,
                                (camera.fov_y() * 0.5).tan(),
                                AssetDebug::Final,
                            )
                            .expect("validated batch");
                        draws += usize::from(!instances.is_empty());
                    } else {
                        for mesh in &mesh_batches[i] {
                            models.draw_pbr(draw, &mut item.mesh, mesh, lighting);
                            draws += item.model.meshes.len();
                        }
                    }
                }
                draws
            })
            .expect("render");
    }

    fn capture(
        &mut self,
        cards: bool,
        batches: &[(usize, Vec<ImpostorInstance>)],
        reverse: bool,
    ) -> Vec<u8> {
        self.renderer.reset_history();
        if !cards {
            self.prepare_meshes(batches);
        }
        for _ in 0..4 {
            self.frame(cards, batches, reverse);
        }
        self.renderer.capture_rgba8().0
    }

    fn pose(&mut self, az: f32, el: f32, distance: f32) {
        let target = Vec3::new(0.0, 1.0, 0.0);
        self.camera.position =
            target + Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos()) * distance;
        let f = (target - self.camera.position).normalize();
        self.camera.set_look(f.z.atan2(f.x), f.y.asin());
    }
}

fn run(event_loop: &ActiveEventLoop, dir: &Path, load_only: bool) -> bool {
    std::fs::create_dir_all(dir).expect("evidence directory");
    let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Dream Impostors V1 Check")
                    .with_inner_size(winit::dpi::PhysicalSize::new(W, H)),
            )
            .expect("window"),
    );
    let mut renderer = Renderer::new(
        window,
        SCENE,
        Settings {
            aa: AaMode::None,
            upscale: UpscalePreset::Native,
            sharpness: 0.6,
        },
    );
    // The qualification reads back composed frames (`capture_rgba8`).
    renderer.set_frame_capture(true);
    assert_eq!(renderer.render_size(), (W, H), "qualification resolution");
    let ctx = renderer.context().clone();
    let lighting = Lighting::new(&ctx);
    let mut camera = Camera::default();
    camera.set_aspect(W as f32 / H as f32);
    let cards = ImpostorRenderer::new(
        &ctx,
        &[SCENE_FORMAT, VELOCITY_FORMAT],
        DEPTH_FORMAT,
        &lighting,
    );
    let models = ModelRenderer::new(&ctx, &[SCENE_FORMAT, VELOCITY_FORMAT], DEPTH_FORMAT);
    let baker = (!load_only).then(|| ImpostorBaker::new(&ctx));
    let mut items = Vec::new();
    let mut resident = 0u64;
    let helmet = Model::load_gltf_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/DamagedHelmet.glb"
    ))
    .expect("helmet fixture");
    for id in 0..17 {
        let model = match id {
            12 | 13 => helmet.clone(),
            15 => {
                let mut m = (*helmet).clone();
                m.nodes.first_mut().expect("helmet rest node").scale *= Vec3::new(1.3, 0.8, 0.9);
                Arc::new(m)
            }
            16 => {
                let mut m = (*fixture(id)).clone();
                m.nodes = vec![dream::model::Node {
                    name: "mirrored-rest-node".into(),
                    translation: Vec3::ZERO,
                    rotation: glam::Quat::from_rotation_y(0.3),
                    scale: Vec3::new(-1.0, 1.2, 0.8),
                    children: Vec::new(),
                    mesh: None,
                    skin: None,
                }];
                m.roots = vec![0];
                for mesh in &mut m.meshes {
                    mesh.node = Some(0);
                }
                Arc::new(m)
            }
            _ => fixture(id),
        };
        let path = dir.join(format!("asset-{id}.imp1"));
        let mut limits = ImpostorAssetLimits::default();
        limits.max_gpu_bytes = limits.max_gpu_bytes.min(RESIDENT_LIMIT - resident);
        if !load_only {
            let settings = if id == 12 || id == 15 {
                BakeSettings::default()
            } else {
                BakeSettings::compact()
            };
            let asset = baker
                .as_ref()
                .unwrap()
                .bake_sources_with_limits(&[ImpostorSource::model(&model)], settings, limits)
                .expect("bounded bake");
            println!("[v1-pack] baked id={id} {}", asset.summary());
            asset.write_baked(&path).expect("write new baked artifact");
        }
        let start = Instant::now();
        let asset = ImpostorAsset::load_baked(&ctx, &path, limits).expect("load baked artifact");
        resident += asset.bytes as u64;
        assert!(resident <= RESIDENT_LIMIT);
        println!(
            "[v1-pack] loaded id={id} bytes={} load-ms={:.1} resident={resident}",
            asset.bytes,
            start.elapsed().as_secs_f64() * 1000.0
        );
        let (lo, hi) = model.static_bounds();
        let normalization = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * Mat4::from_scale(Vec3::splat(2.0 / (hi - lo).max_element()))
            * Mat4::from_translation(-(lo + hi) * 0.5);
        let mesh = ModelGpu::upload(&ctx, &model);
        items.push(Item {
            model,
            mesh,
            asset,
            normalization,
        });
    }
    drop(baker);
    let mut check = Check {
        renderer,
        lighting,
        models,
        cards,
        camera,
        items,
        mesh_batches: Vec::new(),
    };
    if std::env::args().any(|arg| arg == "--motion-only") {
        return check.motion_check(dir, "motion");
    }
    let prefix = std::env::args()
        .find_map(|arg| arg.strip_prefix("--evidence-prefix=").map(str::to_owned))
        .unwrap_or_else(|| {
            if load_only {
                "loaded".into()
            } else {
                "first".into()
            }
        });
    let mut ok = check.qualification(dir, &prefix);
    for count in [256, 1024, 4096] {
        ok &= check.benchmark(count);
    }
    let ring_bytes = check.cards.buffer_bytes();
    // Idle submissions must not allocate more draw storage.
    for _ in 0..32 {
        check.frame(true, &[], false);
    }
    check.cards.collect_retired();
    check.pose(0.6, 0.15, 20.0);
    let reload_batch = vec![check.batch(0, Mat4::IDENTITY)];
    let reload_reference = check.capture(true, &reload_batch, false);
    let reload_model = check.items[0].model.clone();
    let reload_normalization = check.items[0].normalization;
    let last_batches: Vec<_> = (0..check.items.len())
        .map(|id| {
            check.batch(
                id,
                Mat4::from_translation(Vec3::new(id as f32 * 0.1, 0.0, 0.0)),
            )
        })
        .collect();
    // Retire immediately after a real asset-using submission, without a
    // screenshot/readback wait that would drain the queue first.
    check.frame(true, &last_batches, false);
    let pending_serial = ctx.submitted_serial();
    let not_yet_collected = ctx.completed_serial() < pending_serial;
    let blocking_before = ctx.blocking_submissions();
    while let Some(item) = check.items.pop() {
        check.cards.retire_asset(item.asset);
    }
    for _ in 0..4 {
        check.frame(true, &[], false);
        check.cards.collect_retired();
    }
    let retired = check.cards.retired_asset_count();
    let no_uploads = ctx.blocking_submissions() == blocking_before;
    let bounded = check.cards.buffer_bytes() <= ring_bytes;
    println!(
        "[v1] unload retired={retired} submission-pending={not_yet_collected} no-blocking-uploads={no_uploads} ring-bytes={} bounded={bounded}",
        check.cards.buffer_bytes()
    );
    ok &= retired == 0 && no_uploads && bounded && not_yet_collected;
    let asset = ImpostorAsset::load_baked(
        &ctx,
        dir.join("asset-0.imp1"),
        ImpostorAssetLimits::default(),
    )
    .expect("reload after retirement");
    check.items.push(Item {
        mesh: ModelGpu::upload(&ctx, &reload_model),
        model: reload_model,
        asset,
        normalization: reload_normalization,
    });
    let reloaded = check.capture(true, &reload_batch, false);
    let reload_exact = reload_reference == reloaded;
    save_png(dir, &format!("{prefix}-reload"), &reloaded);
    println!("[v1] reload-after-retirement exact={reload_exact}");
    ok &= reload_exact;
    println!(
        "[v1] load-only={load_only} asset-resident-peak={resident} result={}",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

struct App {
    dir: PathBuf,
    load_only: bool,
    finished: bool,
    passed: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.passed = run(event_loop, &self.dir, self.load_only);
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::window::WindowId,
        _: winit::event::WindowEvent,
    ) {
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(
        args.iter()
            .find(|s| !s.starts_with("--"))
            .expect("impostor_v1_check OUTPUT_DIR [--load-only]"),
    );
    let mut app = App {
        dir,
        load_only: args.iter().any(|s| s == "--load-only"),
        finished: false,
        passed: false,
    };
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut app)
        .expect("native qualification");
    if !app.passed {
        std::process::exit(1);
    }
}

fn save_png(dir: &Path, name: &str, pixels: &[u8]) {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(format!("{name}.png")))
        .expect("new evidence path");
    f.write_all(&dream::png::encode(W, H, pixels)).expect("PNG");
}

fn parity(a: &[u8], b: &[u8], bg: &[u8]) -> (bool, f64, f64) {
    assert_eq!(a.len(), (W * H * 4) as usize);
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), bg.len());
    let mut union = 0usize;
    let mut intersect = 0usize;
    let mut n_a = 0usize;
    let mut n_b = 0usize;
    let mut abs = 0u64;
    let mut bias = [0i64; 3];
    for ((a, b), bg) in a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .zip(bg.chunks_exact(4))
    {
        let m_a = (0..3).any(|c| a[c].abs_diff(bg[c]) > 3);
        let m_b = (0..3).any(|c| b[c].abs_diff(bg[c]) > 3);
        n_a += usize::from(m_a);
        n_b += usize::from(m_b);
        union += usize::from(m_a || m_b);
        if m_a && m_b {
            intersect += 1;
            for c in 0..3 {
                abs += a[c].abs_diff(b[c]) as u64;
                bias[c] += b[c] as i64 - a[c] as i64;
            }
        }
    }
    let delta = abs as f64 / (3 * intersect).max(1) as f64;
    let iou = intersect as f64 / union.max(1) as f64;
    let ratio = n_b as f64 / n_a.max(1) as f64;
    let passed = n_a >= 100
        && n_b >= 100
        && delta <= 12.0
        && iou >= 0.93
        && (0.8..=1.25).contains(&ratio)
        && bias
            .iter()
            .all(|b| (*b as f64 / intersect.max(1) as f64).abs() <= 4.0);
    (passed, delta, iou)
}

fn half(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = (h >> 10) & 31;
    let frac = (h & 1023) as f32;
    if exp == 0 {
        sign * frac / 16_777_216.0
    } else if exp == 31 {
        if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + frac / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    }
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
fn velocity_top_first(mut bytes: Vec<u8>) -> Vec<u8> {
    assert_eq!(bytes.len(), (W * H * 4) as usize);
    // Velocity readback is raw Vulkan row order; depth readback is top-first.
    let row = W as usize * 4;
    for y in 0..H as usize / 2 {
        let opposite = H as usize - 1 - y;
        for x in 0..row {
            bytes.swap(y * row + x, opposite * row + x);
        }
    }
    bytes
}

fn read_velocity_top_first(bytes: Vec<u8>) -> Vec<u8> {
    #[cfg(target_os = "macos")]
    {
        bytes
    }
    #[cfg(not(target_os = "macos"))]
    {
        velocity_top_first(bytes)
    }
}

impl Check {
    fn batch(&self, id: usize, placement: Mat4) -> (usize, Vec<ImpostorInstance>) {
        let m = placement * self.items[id].normalization;
        (id, vec![ImpostorInstance::new(m, m, 1.0)])
    }

    fn qualification(&mut self, dir: &Path, prefix: &str) -> bool {
        let mut ok = true;
        for id in 0..self.items.len() {
            for distance in [8.0, 20.0, 60.0] {
                // This edge-on thin fixture has only 42 pixels at d60,
                // below the unchanged 100-pixel comparison sample minimum.
                let distance = if id == 6 && distance == 60.0 {
                    40.0
                } else {
                    distance
                };
                self.pose(0.6 + id as f32 * 0.43, 0.15, distance);
                let bg = self.capture(true, &[], false);
                let batch = vec![self.batch(id, Mat4::IDENTITY)];
                let a = self.capture(false, &batch, false);
                let b = self.capture(true, &batch, false);
                let (pass, abs, iou) = parity(&a, &b, &bg);
                println!(
                    "[v1] asset={id} d={distance} abs={abs:.3} IoU={iou:.4} {}",
                    if pass { "ok" } else { "FAIL" }
                );
                save_png(dir, &format!("{prefix}-{id}-d{distance}-mesh"), &a);
                save_png(dir, &format!("{prefix}-{id}-d{distance}-card"), &b);
                ok &= pass;
            }
        }
        self.pose(0.6, 0.15, 8.0);
        let bg = self.capture(true, &[], false);
        let batch = vec![self.batch(0, Mat4::IDENTITY)];
        let a = self.capture(false, &batch, false);
        self.pose(1.2, 0.15, 8.0);
        let wrong = self.capture(true, &batch, false);
        let rejected = !parity(&a, &wrong, &bg).0;
        println!("[v1] pose-negative-control rejected={rejected}");
        ok &= rejected;

        self.pose(std::f32::consts::FRAC_PI_2, 0.15, 12.0);
        let heroes = vec![
            self.batch(0, Mat4::from_translation(Vec3::new(-0.8, 0.0, -0.2))),
            self.batch(3, Mat4::from_translation(Vec3::new(0.8, 0.0, 0.2))),
        ];
        let reference = self.capture(true, &heroes, false);
        // Large offscreen batch forces mid-frame instance-ring growth.
        let filler = vec![
            ImpostorInstance::new(
                Mat4::from_translation(Vec3::new(1000.0, 0.0, 0.0)),
                Mat4::from_translation(Vec3::new(1000.0, 0.0, 0.0)),
                1.0,
            );
            8192
        ];
        let grown = vec![heroes[0].clone(), (2, filler), heroes[1].clone()];
        let after = self.capture(true, &grown, false);
        let growth_ok = reference == after;
        println!("[v1] mid-frame-instance-growth exact={growth_ok}");
        save_png(dir, &format!("{prefix}-growth"), &after);
        ok &= growth_ok;

        self.pose(std::f32::consts::FRAC_PI_2, 0.1, 9.0);
        // >64 draws grows descriptor pools; >64KiB of params grows the UBO ring.
        let split: Vec<_> = (0..80)
            .map(|j| {
                self.batch(
                    j % 12,
                    Mat4::from_translation(Vec3::new(
                        (j % 10) as f32 * 0.52 - 2.34,
                        (j / 10) as f32 * 0.50,
                        -(j as f32) * 0.013,
                    )) * Mat4::from_scale(Vec3::splat(0.22)),
                )
            })
            .collect();
        let mut merged: Vec<_> = (0..12).map(|id| (id, Vec::new())).collect();
        for (id, instances) in &split {
            merged[*id].1.extend_from_slice(instances);
        }
        let reference = self.capture(true, &merged, false);
        let split_image = self.capture(true, &split, false);
        let reverse = self.capture(true, &split, true);
        let pools_ok = reference == split_image && split_image == reverse;
        println!("[v1] multi-asset-pool-params-growth exact={pools_ok}");
        save_png(dir, &format!("{prefix}-pools"), &split_image);
        ok &= pools_ok;
        ok &= self.motion_check(dir, prefix);
        ok
    }

    fn motion_check(&mut self, dir: &Path, prefix: &str) -> bool {
        self.pose(0.6, 0.15, 8.0);
        let n = self.items[1].normalization;
        let prev =
            Mat4::from_translation(Vec3::new(-0.25, 0.0, 0.0)) * Mat4::from_rotation_y(-0.1) * n;
        let curr =
            Mat4::from_translation(Vec3::new(0.25, 0.0, 0.0)) * Mat4::from_rotation_y(0.1) * n;
        let mut batches = vec![(1, vec![ImpostorInstance::new(curr, prev, 1.0)])];
        self.capture(false, &batches, false);
        let mesh_v = read_velocity_top_first(self.renderer.capture_velocity_rg16f().0);
        let mesh_d = self.renderer.capture_depth_f32().0;
        self.capture(true, &batches, false);
        let card_v = read_velocity_top_first(self.renderer.capture_velocity_rg16f().0);
        let card_d = self.renderer.capture_depth_f32().0;
        for (name, bytes) in [
            ("mesh-v", &mesh_v),
            ("card-v", &card_v),
            ("mesh-d", &mesh_d),
            ("card-d", &card_d),
        ] {
            let mut file = std::fs::File::create_new(dir.join(format!("{prefix}-{name}.bin")))
                .expect("new motion evidence");
            file.write_all(bytes).expect("motion evidence");
        }
        assert_eq!(mesh_v.len(), card_v.len());
        let mut count = 0;
        let mut delta = 0.0f64;
        let mut max_delta = 0.0f32;
        let mut max_motion = 0.0f32;
        let mut finite = true;
        for (i, (a, b)) in mesh_d
            .chunks_exact(4)
            .zip(card_d.chunks_exact(4))
            .enumerate()
        {
            let a = f32::from_le_bytes(a.try_into().unwrap());
            let b = f32::from_le_bytes(b.try_into().unwrap());
            if !(a < 1.0 && b < 1.0) {
                continue;
            }
            count += 1;
            for c in 0..2 {
                let off = i * 4 + c * 2;
                let a = half(u16::from_le_bytes(mesh_v[off..off + 2].try_into().unwrap()));
                let b = half(u16::from_le_bytes(card_v[off..off + 2].try_into().unwrap()));
                finite &= a.is_finite() && b.is_finite();
                delta += (a - b).abs() as f64;
                max_delta = max_delta.max((a - b).abs());
                max_motion = max_motion.max(a.abs());
            }
        }
        let mean = delta / (2 * count).max(1) as f64;
        batches[0].1[0].misc[0] = 0.5;
        self.capture(true, &batches, false);
        let fade_v = read_velocity_top_first(self.renderer.capture_velocity_rg16f().0);
        batches[0].1[0].misc[0] = 0.0;
        self.capture(true, &batches, false);
        let zero_v = read_velocity_top_first(self.renderer.capture_velocity_rg16f().0);
        let zero_d = self.renderer.capture_depth_f32().0;
        let zero_clean = zero_v.iter().all(|v| *v == 0)
            && zero_d
                .chunks_exact(4)
                .all(|d| f32::from_le_bytes(d.try_into().unwrap()) == 1.0);
        let fade_exact = fade_v == card_v;
        let ok = finite
            && count >= 100
            && max_motion > 0.001
            && mean <= 0.0005
            && max_delta <= 0.002
            && fade_exact
            && zero_clean;
        println!(
            "[v1] moving-transform velocity mean={mean:.6} max={max_delta:.6} samples={count} fade-exact={fade_exact} zero-clean={zero_clean} {}",
            if ok { "ok" } else { "FAIL" }
        );
        ok
    }

    fn benchmark(&mut self, count: usize) -> bool {
        self.camera.position = Vec3::new(0.0, 9.0, 75.0);
        self.camera.set_look(-std::f32::consts::FRAC_PI_2, -0.10);
        let mut batches: Vec<_> = (0..12).map(|id| (id, Vec::new())).collect();
        for i in 0..count {
            let id = i % 12;
            let p = Vec3::new((i % 64) as f32 * 2.5 - 78.75, 0.0, -((i / 64) as f32) * 2.5);
            batches[id]
                .1
                .push(self.batch(id, Mat4::from_translation(p)).1[0]);
        }
        let mut ok = true;
        for cards in [false, true] {
            self.prepare_meshes(&batches);
            for _ in 0..20 {
                self.frame(cards, &batches, false);
            }
            let ring_before = self.cards.buffer_bytes();
            let before = if cards {
                self.cards.draws()
            } else {
                self.models.draws()
            };
            let mut gpu = Vec::new();
            let mut wall = Vec::new();
            let mut last_serial = 0;
            for _ in 0..60 {
                let start = Instant::now();
                self.frame(cards, &batches, false);
                wall.push(start.elapsed().as_secs_f64() * 1000.0);
                let (serial, ms) = self
                    .renderer
                    .scene_gpu_sample()
                    .expect("fresh GPU timing required");
                assert!(
                    serial > last_serial && ms.is_finite() && ms > 0.0,
                    "stale/invalid timestamp"
                );
                last_serial = serial;
                gpu.push(ms);
            }
            let after = if cards {
                self.cards.draws()
            } else {
                self.models.draws()
            };
            assert_eq!((after - before) % 60, 0);
            gpu.sort_by(f32::total_cmp);
            wall.sort_by(f64::total_cmp);
            let bounded = self.cards.buffer_bytes() <= ring_before;
            let pass = !cards || (gpu[56] <= 5.0 && bounded);
            let (px, _, _) = self.renderer.capture_rgba8();
            let bg = self.capture(true, &[], false);
            let nonempty = parity(&px, &px, &bg).0;
            println!(
                "[v1-perf] objects={count} mode={} draws={} scene-med={:.3} scene-p95={:.3} frame-call-med={:.3} frame-call-p95={:.3} nonempty={nonempty} {}",
                if cards { "card" } else { "mesh" },
                (after - before) / 60,
                gpu[29],
                gpu[56],
                wall[29],
                wall[56],
                if pass && nonempty { "ok" } else { "FAIL" },
            );
            ok &= pass && nonempty;
        }
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::{H, W, half, velocity_top_first};

    #[test]
    fn velocity_readback_row_alignment() {
        let mut bytes = vec![0; (W * H * 4) as usize];
        bytes[0] = 17;
        bytes[((H - 1) * W * 4) as usize] = 23;
        let flipped = velocity_top_first(bytes);
        assert_eq!(flipped[0], 23);
        assert_eq!(flipped[((H - 1) * W * 4) as usize], 17);
    }

    #[test]
    fn half_velocity_decode() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xbc00), -1.0);
        assert_eq!(half(1), 1.0 / 16_777_216.0);
        assert_eq!(half(0x7c00), f32::INFINITY);
        assert!(half(0x7e00).is_nan());
    }
}
