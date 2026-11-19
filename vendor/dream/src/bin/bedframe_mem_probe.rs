//! Staged resident-memory probe for a UI-only Vulkan app.
//!
//! Prints the process resident size after each start-up stage: the working
//! set on Windows, resident set size elsewhere. Resident memory includes cold
//! driver and loader pages that were touched once and are rarely used again.
//! After the start-up stages, the probe calls [`dream::memory::trim_working_set`]
//! and prints again. The trimmed figure shows the hot set the app needs while
//! it renders, and the wake rounds show that a few frames after the trim bring
//! back only the pages that are actually used.
//!
//! ```text
//! cargo run --release --bin bedframe_mem_probe
//! ```

use dream::gfx::{Renderer, SurfaceError};
use dream::memory::{resident_bytes, trim_working_set};
use dream::settings::Settings;
use dream::ui::Rect;
use dream::ui::atlas::AtlasConfig;
use dream::ui::font::{FontId, Fonts};
use dream::ui::widgets::{ButtonStyle, Ui, UiInput};
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 600;
/// Time for the OS to settle a trim before reading the working set.
const TRIM_SETTLE: Duration = Duration::from_millis(1500);
/// Time between a stage and its measurement.
const SETTLE: Duration = Duration::from_millis(200);

fn report(label: &str) {
    sleep(SETTLE);
    let mb = resident_bytes() as f64 / 1048576.0;
    println!("{label:<34} resident {mb:7.1} MB");
}

fn trimmed(label: &str) {
    trim_working_set();
    sleep(TRIM_SETTLE);
    report(label);
}

/// Build one UI frame with a single button and submit it.
fn ui_frame(ui: &mut Ui, renderer: &mut Renderer, window: &Window) {
    let mut timeouts = 0u32;
    loop {
        let (w, h) = renderer.display_size();
        ui.begin(UiInput::default(), (w as f32, h as f32));
        ui.button(
            "Hello world",
            Rect::new(10.0, 10.0, 120.0, 30.0),
            ButtonStyle::new(FontId(0)),
        );
        let frame = ui.end();
        let res = renderer.render_ui_only(Some(&frame));
        ui.recycle_frame(frame);
        match res {
            Ok(()) => return,
            Err(SurfaceError::Outdated) => renderer.resize(window.inner_size()),
            Err(SurfaceError::Timeout) if timeouts < 20 => {
                timeouts += 1;
                sleep(Duration::from_millis(20));
            }
            Err(err) => panic!("render UI frame: {err:?}"),
        }
    }
}

struct Probe;

impl ApplicationHandler for Probe {
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut attrs = Window::default_attributes()
            .with_title("bedframe_mem_probe")
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
            .with_visible(true);
        #[cfg(target_os = "linux")]
        {
            attrs = attrs
                .with_min_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
                .with_max_inner_size(PhysicalSize::new(WIDTH, HEIGHT));
        }
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        report("window created");

        let mut renderer = Renderer::new(
            Arc::clone(&window),
            dream::DEFAULT_SCENE_WGSL,
            Settings::default(),
        );
        report("Renderer::new");

        let mut ui = Ui::new(Fonts::bundled(), AtlasConfig::default());
        report("Ui::new+fonts");

        for _ in 0..3 {
            ui_frame(&mut ui, &mut renderer, &window);
        }
        report("after 3 UI frames");

        for _ in 0..200 {
            ui_frame(&mut ui, &mut renderer, &window);
        }
        report("after 200 UI frames");

        trimmed("after trim_working_set");

        for round in 1..=3 {
            ui_frame(&mut ui, &mut renderer, &window);
            for _ in 0..60 {
                ui_frame(&mut ui, &mut renderer, &window);
            }
            report(&format!("wake round {round}"));
            trimmed(&format!("  trimmed after round {round}"));
        }

        event_loop.exit();
    }
}

fn main() {
    report("process start");
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.run_app(&mut Probe).expect("run event loop");
}
