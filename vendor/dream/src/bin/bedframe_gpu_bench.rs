use dream::camera::Camera;
use dream::gfx::Renderer;
use dream::settings::Settings;
use dream::ui::atlas::AtlasConfig;
use dream::ui::draw::{GradientAxis, UiFrame};
use dream::ui::font::{FontId, Fonts};
use dream::ui::widgets::{ButtonStyle, Ui, UiInput};
use dream::ui::{Color, Rect};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

struct Probe;

fn make_frame(size: (u32, u32)) -> UiFrame {
    let fonts = Fonts::bundled();
    let mut ui = Ui::new(fonts, AtlasConfig::default());
    ui.begin(UiInput::default(), (size.0 as f32, size.1 as f32));
    ui.rect(
        Rect::new(0.0, 0.0, size.0 as f32, size.1 as f32),
        Color::rgb(20, 24, 30),
    );
    for i in 0..400 {
        let x = (i % 20) as f32 * 60.0;
        let y = (i / 20) as f32 * 28.0;
        ui.button(
            &format!("Row {i:03}"),
            Rect::new(x, y, 58.0, 26.0),
            ButtonStyle::new(FontId(0)),
        );
        if i % 4 == 0 {
            ui.draw_list_mut().gradient(
                Rect::new(x, y + 24.0, 58.0, 2.0),
                Color::rgb(30, 40, 60),
                Color::rgb(80, 90, 110),
                GradientAxis::LeftToRight,
            );
        }
    }
    ui.end()
}

fn render(renderer: &mut Renderer, frame: &UiFrame, scene: bool) {
    if scene {
        renderer
            .render_ui(
                &Camera::default(),
                None,
                |draw| {
                    let _ = draw.raw();
                    0
                },
                Some(frame),
            )
            .expect("scene frame");
    } else {
        renderer.render_ui_only(Some(frame)).expect("UI-only frame");
    }
}

fn profile(renderer: &mut Renderer, frame: &UiFrame, scene: bool, name: &str) {
    for _ in 0..30 {
        render(renderer, frame, scene);
    }
    let mut total = Vec::with_capacity(1000);
    let mut record = Vec::with_capacity(1000);
    let mut wait = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let started = Instant::now();
        render(renderer, frame, scene);
        total.push(started.elapsed().as_nanos());
        let phases = renderer.frame_phases();
        record.push(phases.record_us);
        wait.push(phases.token_wait_us);
    }
    total.sort_unstable();
    record.sort_by(f32::total_cmp);
    wait.sort_by(f32::total_cmp);
    println!(
        "{name},frames=1000,render_cpu_p50_ns={},render_cpu_p95_ns={},render_cpu_p99_ns={},record_cpu_p50_us={},record_cpu_p95_us={},token_wait_p95_us={}",
        total[500], total[950], total[990], record[500], record[950], wait[950]
    );
}

impl ApplicationHandler for Probe {
    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Dream UI Renderer Probe")
                        .with_inner_size(PhysicalSize::new(1200, 600))
                        .with_visible(false),
                )
                .expect("probe window"),
        );
        let mut renderer = Renderer::new(
            window,
            include_str!("../app/scene.wgsl"),
            Settings::default(),
        );
        let frame = make_frame(renderer.display_size());
        for _ in 0..30 {
            render(&mut renderer, &frame, false);
        }
        let cold = renderer.context().gpu_allocation_stats();
        println!(
            "ui_only_cold,live_allocator_bytes={},live_allocation_count={},vertices={},indices={},commands={}",
            cold.0,
            cold.1,
            frame.vertices.len(),
            frame.indices.len(),
            frame.cmds.len()
        );
        // The pixel checks read composed frames back; the timing loops below
        // run with capture off, the default direct path.
        renderer.set_frame_capture(true);
        render(&mut renderer, &frame, false);
        let (ui_pixels, w, h) = renderer.capture_rgba8();
        render(&mut renderer, &frame, true);
        let (scene_pixels, sw, sh) = renderer.capture_rgba8();
        assert_eq!((w, h), (sw, sh));
        assert_eq!(ui_pixels.len(), scene_pixels.len());
        let worst = ui_pixels
            .iter()
            .zip(&scene_pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            worst <= 1,
            "opaque UI-only vs scene output differs by {worst} LSB"
        );
        let warm = renderer.context().gpu_allocation_stats();
        println!(
            "scene_warm,live_allocator_bytes={},live_allocation_count={},pixel_max_difference_lsb={worst}",
            warm.0, warm.1
        );
        let released = renderer.release_scene_resources();
        let freed = renderer.context().gpu_allocation_stats();
        println!(
            "scene_released,released={released},live_allocator_bytes={},live_allocation_count={}",
            freed.0, freed.1
        );
        assert!(released, "scene resources were not released");
        assert!(
            freed.0 * 10 <= warm.0 * 6,
            "release freed too little: {} -> {} live allocator bytes",
            warm.0,
            freed.0
        );
        // The scene is shown again: the targets, upscaler and blit set rebuild
        // on demand and reproduce the first scene frame.
        render(&mut renderer, &frame, true);
        let (rebuilt_pixels, rw, rh) = renderer.capture_rgba8();
        assert_eq!((w, h), (rw, rh));
        let rebuilt_worst = scene_pixels
            .iter()
            .zip(&rebuilt_pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            rebuilt_worst <= 1,
            "rebuilt scene output differs from the first scene frame by {rebuilt_worst} LSB"
        );
        let rebuilt = renderer.context().gpu_allocation_stats();
        println!(
            "scene_rebuilt,live_allocator_bytes={},pixel_max_difference_lsb={rebuilt_worst}",
            rebuilt.0
        );
        renderer.set_frame_capture(false);
        profile(&mut renderer, &frame, true, "scene_under_opaque_ui");
        profile(&mut renderer, &frame, false, "ui_only_after_scene");
        profile(&mut renderer, &frame, true, "scene_under_opaque_ui_repeat");
        event_loop.exit();
    }
}

fn main() {
    println!(
        "CPU render/record timings, not GPU execution time; allocator bytes exclude driver/swapchain/pipeline overhead."
    );
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut Probe)
        .expect("run probe");
}
