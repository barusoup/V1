//! Dream Studio: the engine's own editor shell.
//!
//! This is the working app the engine's UI framework (BedFrame) was built for - a
//! visible window with a real viewport, pixel-art chrome, an inspector and
//! driving every subsystem the engine has today (Vulkan scene +
//! UI overlay, FSR, camera, tweens, atlas text).
//!
//! Design notes:
//!
//! - **On-demand rendering.** The app renders a frame only when something
//!   can have changed: input, a UI animation in flight, or a camera move.
//!   A perfectly still screen
//!   requests zero frames (0 Hz, ~0 % GPU), and input wakes it instantly.
//!   No extra latency is added while active: frames are requested the
//!   moment an event lands, ahead of `about_to_wait`.
//! - **Hidden means off.** A completely hidden window — minimised, or fully
//!   occluded by other windows where the platform reports it — owes no
//!   frames at all: the loop parks until the window shows again, whatever
//!   motion is in flight, and repaints on the way back.
//! - **Refresh-rate awareness.** While continuously rendering, the loop
//!   measures its own present cadence and reports the display's refresh
//!   rate in the status bar; `dt` is clamped so a long idle cannot make
//!   animations jump.
//! - **Instant theme swap.** Switching theme is immediate. Motion is reserved
//!   for things that move - panels, popups - not for a colour change.
//! - **No assets.** Scene, palette and icons are all code: a bare checkout
//!   builds a running editor.

pub mod chrome;
pub mod inspector;
pub mod main_pane;
pub mod scene;

use crate::camera::Camera;
use crate::gfx::Renderer;
use crate::settings::{Settings, UpscalePreset};
use crate::texture::GpuTexture;
use crate::ui::font::{Align, FontId};
use crate::ui::theme::{self, Theme};
use crate::ui::widgets::{ButtonStyle, KeyFeed, Ui, UiInput};
use crate::ui::{Color, Rect};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

mod actions;
mod input;
mod lifecycle;
mod pacing;
mod settings_page;

use main_pane::{MainPanePage, PreviewImage, SceneManagerState, ThumbWarmup};
pub use settings_page::ThemeChoice;

/// The napxl 24-swatch palette (identical values to the HTML app).
pub const PALETTE: [Color; 24] = [
    Color::rgb(0x00, 0x00, 0x00),
    Color::rgb(0x3b, 0x3b, 0x3b),
    Color::rgb(0x7a, 0x7a, 0x7a),
    Color::rgb(0xbd, 0xbd, 0xbd),
    Color::rgb(0xff, 0xff, 0xff),
    Color::rgb(0x7b, 0x3f, 0x2a),
    Color::rgb(0xc0, 0x39, 0x2b),
    Color::rgb(0xff, 0x5a, 0x4f),
    Color::rgb(0xff, 0x9f, 0x1c),
    Color::rgb(0xff, 0xd2, 0x3f),
    Color::rgb(0xf6, 0xe7, 0xb4),
    Color::rgb(0x2e, 0x7d, 0x32),
    Color::rgb(0x43, 0xc7, 0x5a),
    Color::rgb(0xb6, 0xe3, 0x6a),
    Color::rgb(0x0b, 0x6e, 0x6e),
    Color::rgb(0x2e, 0xc4, 0xb6),
    Color::rgb(0x0a, 0x3d, 0x91),
    Color::rgb(0x2f, 0x80, 0xed),
    Color::rgb(0x8e, 0xcb, 0xff),
    Color::rgb(0x5b, 0x3f, 0xa0),
    Color::rgb(0x9b, 0x6b, 0xd6),
    Color::rgb(0xd6, 0x33, 0x84),
    Color::rgb(0xff, 0x8f, 0xb8),
    Color::rgb(0xff, 0xc9, 0xa8),
];

/// Editor preferences that live for the session.
#[derive(Clone, Debug)]
pub struct AppSettings {
    pub antialias: bool,
    pub upscale: UpscalePreset,
    pub sharpness: f32,
    pub spin: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            antialias: true,
            upscale: UpscalePreset::Native,
            sharpness: 0.6,
            spin: 0.4,
        }
    }
}

/// Undoable editor actions (camera moves).
#[derive(Clone, Debug)]
pub enum Command {
    Camera(scene::OrbitCamera),
}

/// Panel rectangles for the current screen size. The window bar owns the
/// top strip (it floats above the rest of the UI, but the strip stays
/// reserved so nothing important hides under it).
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub titlebar: Rect,
    pub toolbar: Rect,
    pub rail: Rect,
    pub inspector: Rect,
    /// The main central workspace pane (can display scene manager, viewport, etc.).
    pub main_pane: Rect,
    /// Viewport rectangle alias matching `main_pane`.
    pub viewport: Rect,
}

impl Layout {
    pub fn compute(screen: (f32, f32), metrics: &theme::Metrics) -> Self {
        let titlebar = Rect::new(0.0, 0.0, screen.0, metrics.titlebar_h);
        let toolbar = Rect::new(0.0, titlebar.max().1, screen.0, metrics.toolbar_h);
        let rail = Rect::new(
            0.0,
            toolbar.max().1,
            metrics.rail_w,
            screen.1 - toolbar.max().1,
        );
        let inspector = Rect::new(
            screen.0 - metrics.inspector_w,
            toolbar.max().1,
            metrics.inspector_w,
            rail.h,
        );
        let main_pane = Rect::new(
            rail.max().0,
            toolbar.max().1,
            (screen.0 - rail.w - metrics.inspector_w).max(0.0),
            rail.h,
        );
        let viewport = main_pane;
        Self {
            titlebar,
            toolbar,
            rail,
            inspector,
            main_pane,
            viewport,
        }
    }
}

/// Inspector state: selected object, colour editing, render section.
#[derive(Clone, Debug)]
pub struct InspectorState {
    pub selected: Option<usize>,
    pub hsv: (f32, f32, f32),
    pub palette: Option<usize>,
    /// Colour before the current picker drag (one undo entry per drag).
    pub color_drag: Option<Color>,
    pub scene_open: bool,
    pub render_open: bool,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            selected: Some(3),
            hsv: (30.0, 0.8, 0.9),
            palette: None,
            color_drag: None,
            scene_open: true,
            render_open: true,
        }
    }
}

/// Commands the app queues for the window owning it (the winit layer
/// drains these each frame): the app's custom title bar has no OS chrome
/// to fall back on, so moving, resizing and the caption buttons are all
/// imperative window requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowCommand {
    /// Drag the whole window by the title bar.
    Move,
    /// Resize from a window edge/corner.
    Resize(ResizeEdge),
    Minimize,
    ToggleMaximize,
    Close,
    /// Cursor affordance for the current hover region.
    SetCursor(WindowCursor),
}

/// Edge (or corner) to resize from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeEdge {
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Cursor the window should show for the current hover region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowCursor {
    Default,
    Move,
    ResizeHorizontal,
    ResizeVertical,
    ResizeDiagonal,
    ResizeDiagonalAnti,
}

/// Frame pacing: what rate the loop should present at.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FramePacing {
    pub display_hz: Option<f32>,
    pub cap_hz: Option<f32>,
    due: Option<Instant>,
}

/// One pixel from a captured frame buffer (top-left origin, sRGB bytes).
fn rgba_at(pixels: &[u8], width: u32, at: (f32, f32)) -> (u8, u8, u8) {
    let x = (at.0.max(0.0) as u32).min(width.saturating_sub(1));
    let y = at.1.max(0.0) as u32;
    let i = ((y * width + x) * 4) as usize;
    (pixels[i], pixels[i + 1], pixels[i + 2])
}

/// Resident set size in bytes (leak check in the selfcheck soak).
#[allow(clippy::needless_return)] // cfg'd tail blocks need explicit returns
fn rss_bytes() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                process: *mut core::ffi::c_void,
                counters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        let mut counters = unsafe { core::mem::zeroed::<ProcessMemoryCounters>() };
        counters.cb = core::mem::size_of::<ProcessMemoryCounters>() as u32;
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if ok != 0 {
            return counters.working_set_size as u64;
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        // `ps` reports resident bytes directly (KiB); /proc does not exist
        // on macOS so the Linux path would silently read 0.
        let pid = std::process::id();
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok();
        return output
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map(|kib| kib * 1024)
            .unwrap_or(0);
    }
    #[cfg(target_os = "linux")]
    {
        // statm field 2 is RESIDENT pages; field 1 is total virtual size.
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|text| {
                text.split_whitespace()
                    .nth(1)
                    .and_then(|pages| pages.parse::<u64>().ok())
            })
            .map(|pages| pages * 4096)
            .unwrap_or(0)
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

/// Scripted screenshot states for `--selfcheck`.
#[derive(Clone, Debug)]
pub enum SelfCheckShot {
    Default,
    Dark,
    Light,
    Hover,
    Inspector,
    Menu,
    Settings,
}

pub struct App {
    pub renderer: Renderer,
    pub ui: Ui,
    pub camera: Camera,
    pub theme: Theme,
    pub scene: scene::Scene,
    pub settings: AppSettings,
    /// Which palette the app uses (the Settings page's Theme row).
    pub theme_choice: ThemeChoice,
    /// OS appearance preference reported by the window. `true` until the
    /// platform says otherwise, which is the dark palette the app has always
    /// booted with, so [`ThemeChoice::System`] has a defined answer headless.
    pub system_dark: bool,
    /// Published rects of the Theme selector's three cells, in
    /// [`ThemeChoice::ALL`] order (the scripted drive phase clicks these).
    pub theme_cells: [Rect; 3],
    pub layout: Layout,
    /// Active page displayed in the main pane (Scene Manager, Viewport, or
    /// the dedicated Settings page).
    pub main_pane_page: MainPanePage,
    /// The tab page Settings was opened from — where [`App::close_settings`]
    /// returns. Never `Settings` itself.
    pub settings_return: MainPanePage,
    /// State for the scene manager page.
    pub scene_manager: SceneManagerState,
    /// Real 3D captures per scene (square sRGB RGBA8), by scene id.
    pub preview_pixels: HashMap<u64, PreviewImage>,
    /// Uploaded card previews, by scene id.
    pub preview_textures: HashMap<u64, Arc<GpuTexture>>,
    /// Replaced previews cooling down before their GPU reads complete.
    pub retired_previews: Vec<(Arc<GpuTexture>, u8)>,
    /// Scene id whose exit view the next frame's scene pass captures.
    pub pending_thumb_capture: Option<u64>,
    /// Scene id of a capture queued on the renderer, awaiting its pixels.
    pub thumb_job: Option<u64>,
    /// Hidden one-frame shoot restoring itself after the capture lands.
    pub thumb_warmup: Option<ThumbWarmup>,
    /// Scenes whose capture failed (never retried by warmups).
    pub thumb_failed: HashSet<u64>,
    pub inspector: InspectorState,
    pub undo_stack: Vec<Command>,
    pub redo_stack: Vec<Command>,
    pub engine_settings: Settings,
    /// UI input for the frame being built (event handlers fill this).
    pub input: UiInput,
    /// Persistent pointer position (survives the per-frame input take).
    pub mouse: (f32, f32),
    /// Logical screen size (physical pixels / `scale`): every app rect,
    /// layout and pointer position lives in these units.
    pub screen: (f32, f32),
    /// UI scale factor (window.scale_factor): logical → physical.
    pub scale: f32,
    /// When a pending UI clock (tooltip dwell, toast expiry) next needs a
    /// frame — folded into [`Self::next_frame_at`] so a parked app still
    /// wakes for it. Refreshed every built frame.
    pub ui_wake: Option<Instant>,
    /// Active IME preedit (composition string and its caret range), kept
    /// uncommitted until `Ime::Commit` lands.
    pub ime_preedit: Option<(String, Option<(usize, usize)>)>,
    // Frame-loop bookkeeping.
    /// The window is completely hidden — minimised, or fully occluded by
    /// other windows (where the platform reports either). No frame is built
    /// or presented while it is; showing the window again repaints at once.
    pub hidden: bool,
    pub dirty: bool,
    pub last_input: Instant,
    pub last_frame: Instant,
    pub frames_rendered: u64,
    pub frames_skipped: u64,
    pub frames_immediate: u64,
    pub captures: u32,
    /// Set while the user drags the viewport (camera controls).
    pub drag: Option<ViewportDrag>,
    /// Orbit state before the current viewport drag (undo snapshot).
    drag_start_orbit: Option<scene::OrbitCamera>,
    /// Cursor hidden over the viewport (napxl crosshair stage).
    pub viewport_cursor: Option<(f32, f32)>,
    /// Fixed frame time for scripted runs (selfcheck determinism).
    pub fixed_dt: Option<f32>,
    /// Live UI animation state from the last built frame.
    pub animating: bool,
    /// Frame pacing: the display's refresh rate and the developer cap.
    pub pacing: FramePacing,
    /// Window commands queued for the host loop (title bar, resize frame).
    pub window_commands: Vec<WindowCommand>,
    resize_pending: Option<ResizeEdge>,
    pub window_maximized: bool,
    pub titlebar_drag: bool,
    pub menu_open: bool,
    pub menu_at: (f32, f32),
    pub menu_focus: usize,
    pub sharpness_rect: Rect,
    pub left_down: bool,
    pub right_down: bool,
    pub pan_mode: bool,
    pub shift_held: bool,
    pub ctrl_held: bool,
    /// Alt/Option held (word-modifier, reserved for the focus stage).
    pub alt_held: bool,
    /// The default editing keybinds: latches the modifiers and decodes
    /// key events into `input`'s nav edges, text and clipboard seam.
    keys: KeyFeed,
    pub quit: bool,
}

/// Pointer buttons the app translates into UI input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerButton {
    Left,
    Right,
    Middle,
}

// Logical keys (mapped from winit in the binary): the BedFrame key
// vocabulary, re-exported so hosts drive [`App::key_event`] and the
// widget editing chords through one enum.
pub use crate::ui::widgets::Key;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewportDrag {
    Orbit,
    Pan,
}

impl App {
    pub fn style_button(&self, _center: (f32, f32)) -> ButtonStyle {
        self.theme.button(self.font())
    }

    pub fn font(&self) -> FontId {
        FontId(0)
    }

    pub fn label_color(&self) -> Color {
        self.theme.fg
    }

    pub fn dim_color(&self) -> Color {
        self.theme.g3
    }

    pub fn align_left() -> Align {
        Align::Left
    }

    // --- Selfcheck -------------------------------------------------------------

    /// Scripted screenshot run (see `SelfCheckShot`). Renders each state
    /// and writes `target/dream-app/selfcheck-*.png`, then soaks the
    /// on-demand loop and reports frame/memory stats.
    pub fn selfcheck(&mut self) {
        let dir = std::path::Path::new("target/dream-app");
        std::fs::create_dir_all(dir).expect("create selfcheck dir");
        let states: [(SelfCheckShot, &str); 7] = [
            (SelfCheckShot::Default, "default"),
            (SelfCheckShot::Dark, "dark"),
            (SelfCheckShot::Light, "light"),
            (SelfCheckShot::Hover, "hover"),
            (SelfCheckShot::Inspector, "inspector"),
            (SelfCheckShot::Menu, "menu"),
            (SelfCheckShot::Settings, "settings"),
        ];
        // The scene renders only into the viewport's own region: with no UI
        // drawn over it, the viewport holds the scene's canvas while the rest
        // of the frame keeps the renderer's clear colour.
        let viewport = self.layout.viewport;
        // Pixel reads are physical; layout rects are logical.
        let scale = self.scale;
        let pv = self.physical_rect(viewport);
        let p_inspector = self.physical_rect(self.layout.inspector);
        let p_titlebar = self.physical_rect(self.layout.titlebar);
        self.scene.load_debug();
        for _ in 0..2 {
            let App {
                renderer,
                scene,
                camera,
                ..
            } = self;
            renderer
                .render_ui(
                    camera,
                    Some(pv),
                    |draw| {
                        scene.draw(draw);
                        1
                    },
                    None,
                )
                .expect("render the scene with no UI overlay");
        }
        let (pixels, width, _height) = self.renderer.capture_rgba8();
        let corner = pv.x + 12.0 * scale;
        let inside = rgba_at(&pixels, width, (corner, pv.y + 12.0 * scale));
        let above = rgba_at(&pixels, width, (pv.center().0, pv.y - 20.0 * scale));
        let outside = rgba_at(&pixels, width, p_inspector.center());
        println!(
            "[selfcheck] viewport session: inside {inside:?} above {above:?} outside {outside:?}"
        );
        assert_ne!(inside, above, "the scene drew above its viewport");
        assert_ne!(inside, outside, "the scene drew beside its viewport");
        assert_eq!(
            above, outside,
            "everything outside the viewport should be the clear colour"
        );
        assert!(
            outside.2 > outside.0,
            "the region outside the viewport should hold the clear colour, got {outside:?}"
        );
        self.scene.unload_debug();

        // Deterministic 60 fps while settling scripted states.
        self.fixed_dt = Some(1.0 / 60.0);
        for (shot, name) in states {
            self.selfcheck_apply(&shot);
            for _ in 0..28 {
                self.render(Instant::now());
            }
            let (pixels, width, height) = self.renderer.capture_rgba8();
            let path = dir.join(format!("selfcheck-{name}.png"));
            crate::png::write(&path, width, height, &pixels).expect("write selfcheck png");
            if matches!(&shot, SelfCheckShot::Dark | SelfCheckShot::Light) {
                Self::assert_themed_ink(&pixels, width as usize, &self.theme, p_inspector);
            }
            if matches!(&shot, SelfCheckShot::Default) {
                Self::assert_caption_glyphs(&pixels, width as usize, p_titlebar);
            }
            if matches!(&shot, SelfCheckShot::Settings) {
                Self::assert_theme_selector(&pixels, width as usize, self);
            }
            println!(
                "[selfcheck] {name}: {}x{} saved {} (dark={})",
                width,
                height,
                path.display(),
                self.theme.dark,
            );
        }
        // Reset to the default dark theme for the soak.
        self.set_theme_choice(ThemeChoice::Dark);
        self.selfcheck_apply(&SelfCheckShot::Default);
        // The drive phase ends with a capture and a few live frames.
        self.drive();

        // Soak phase 1 (busy): drive input, drags, theme flips and renders.
        let rss = rss_bytes;
        let rss_before = rss();
        let gpu_before = self.renderer.context().gpu_allocation_stats();
        println!(
            "[selfcheck] gpu before soak: {} bytes in {} allocations",
            gpu_before.0, gpu_before.1
        );
        for _ in 0..200 {
            self.toggle_debug();
            self.tick(Instant::now());
            self.toggle_debug();
        }
        println!(
            "[selfcheck] gpu after 200 debug load/unload cycles: {:?}",
            self.renderer.context().gpu_allocation_stats()
        );
        for frame in 0..1200 {
            let now = Instant::now();
            match frame % 7 {
                0 => self.pointer_moved((200.0 + frame as f32, 300.0)),
                1 => self.pointer_button(PointerButton::Left, true),
                2 => self.pointer_button(PointerButton::Left, false),
                3 => self.viewport_drag(6.0, 2.0),
                4 => self.wheel(0.0),
                5 => self.flip_theme(),
                _ => {}
            }
            self.tick(now);
            if frame % 300 == 0 {
                self.dirty = true;
            }
        }
        let rss_mid = rss();
        println!(
            "[selfcheck] gpu after busy phase: {:?}",
            self.renderer.context().gpu_allocation_stats()
        );

        // Soak phase 2 (idle): nothing moves, nothing animates — the
        // on-demand loop must request zero frames (0 Hz).
        self.settings.spin = 0.0;
        self.dirty = false;
        std::thread::sleep(std::time::Duration::from_millis(400));
        let skipped_before = self.frames_skipped;
        for _ in 0..60 {
            self.tick(Instant::now());
        }
        let idle_skips = self.frames_skipped - skipped_before;
        assert!(
            idle_skips >= 58,
            "idle loop still rendering: only {idle_skips}/60 frames skipped"
        );
        println!("[selfcheck] idle proof: {idle_skips}/60 frames skipped (0 Hz while still)");
        // Recovery: one wake-up event renders again.
        self.settings.spin = 0.4;
        let rendered_before_wake = self.frames_rendered;
        self.pointer_moved((640.0, 450.0));
        self.tick(Instant::now());
        assert!(
            self.frames_rendered > rendered_before_wake,
            "input must wake the renderer"
        );
        // Soak phase 3 (hidden): the window reports itself completely hidden
        // (minimised, or fully covered by other windows). Even a dirty frame
        // — input arrived while hidden — is owed to no one: nothing renders,
        // and showing the window again paints at once.
        self.set_hidden(true);
        let rendered_before_hidden = self.frames_rendered;
        for _ in 0..60 {
            self.dirty = true;
            self.tick(Instant::now());
        }
        assert_eq!(
            self.frames_rendered, rendered_before_hidden,
            "a hidden window must not render while dirty"
        );
        println!(
            "[selfcheck] hidden proof: 0/60 frames rendered while hidden (dirty the whole time)"
        );
        self.set_hidden(false);
        self.tick(Instant::now());
        assert!(
            self.frames_rendered > rendered_before_hidden,
            "showing the window must repaint it immediately"
        );
        println!("[selfcheck] hidden proof: showing the window rendered a frame at once");

        let original_page = self.main_pane_page;
        for page in [MainPanePage::SceneManager, MainPanePage::Settings] {
            self.main_pane_page = page;
            let frame = self.build_ui_frame(1.0 / 60.0);
            let viewport = self.physical_rect(self.layout.viewport);
            let App {
                renderer,
                scene,
                camera,
                ..
            } = self;
            renderer
                .render_ui(
                    camera,
                    Some(viewport),
                    |draw| {
                        scene.draw(draw);
                        1
                    },
                    Some(&frame),
                )
                .expect("legacy composite frame");
            let (legacy_px, lw, lh) = renderer.capture_rgba8();
            renderer
                .render_ui_only(Some(&frame))
                .expect("ui-only frame");
            let (only_px, ow, oh) = renderer.capture_rgba8();
            assert_eq!((lw, lh), (ow, oh));
            let worst = legacy_px
                .iter()
                .zip(&only_px)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            assert!(
                worst <= 1,
                "{page:?}: legacy vs ui-only capture differs by {worst} LSB"
            );
            self.ui.recycle_frame(frame);
            println!("[selfcheck] {page:?}: legacy vs ui-only capture worst {worst} LSB");
        }
        self.main_pane_page = original_page;
        self.dirty = true;
        let rss_after = rss();
        println!(
            "[selfcheck] gpu after idle+wake: {:?}",
            self.renderer.context().gpu_allocation_stats()
        );
        println!(
            "[selfcheck] soak: frames={} rendered={} skipped={} rss {} -> {} -> {} bytes (delta {})",
            self.frames_rendered + self.frames_skipped,
            self.frames_rendered,
            self.frames_skipped,
            rss_before,
            rss_mid,
            rss_after,
            rss_after as i64 - rss_before as i64
        );
        println!(
            "[selfcheck] atlas pages={} anim_count={}",
            self.ui.cache().atlas().page_count(),
            self.ui.anim_count()
        );
        self.render(Instant::now());
        let (pixels, width, height) = self.renderer.capture_rgba8();
        let path = dir.join("selfcheck-after-soak.png");
        crate::png::write(&path, width, height, &pixels).expect("write after-soak png");
        println!("[selfcheck] after-soak saved {}", path.display());
    }

    /// Headless interaction session: drives the same public input API the
    /// winit loop calls, on a hidden window, and asserts the app responds.
    pub fn drive(&mut self) {
        use crate::ui::Rect;
        self.fixed_dt = Some(1.0 / 60.0);
        let mut failures: Vec<String> = Vec::new();
        let check = |ok: bool, what: &str, failures: &mut Vec<String>| {
            if ok {
                println!("[drive] ok   {what}");
            } else {
                println!("[drive] FAIL {what}");
                failures.push(what.to_string());
            }
        };

        // 1. The toolbar settings button opens the Settings page, and its
        // Theme selector drives the palette: an explicit choice sticks, while
        // System follows the OS appearance the window reports.
        self.click(chrome::settings_button_rect(self));
        check(
            self.main_pane_page == MainPanePage::Settings,
            "the settings button opens the Settings page",
            &mut failures,
        );
        check(
            self.theme_cells.iter().all(|cell| cell.w > 1.0),
            "the Settings page publishes its Theme cells",
            &mut failures,
        );
        // An explicit choice sticks, whatever the OS says.
        self.set_system_dark(Some(true));
        self.click(self.theme_cells[ThemeChoice::Light.index()]);
        check(
            self.theme_choice == ThemeChoice::Light && !self.theme.dark,
            "the Theme selector picks light",
            &mut failures,
        );
        self.click(self.theme_cells[ThemeChoice::Dark.index()]);
        check(
            self.theme_choice == ThemeChoice::Dark && self.theme.dark,
            "the Theme selector picks dark",
            &mut failures,
        );
        // System hands the palette back to the OS, and keeps following it.
        self.set_system_dark(Some(false));
        self.click(self.theme_cells[ThemeChoice::System.index()]);
        check(
            self.theme_choice == ThemeChoice::System && !self.theme.dark,
            "System follows the OS appearance",
            &mut failures,
        );
        self.set_system_dark(Some(true));
        check(
            self.theme.dark,
            "an OS appearance change reaches a System choice live",
            &mut failures,
        );

        // The pill is a carousel, not a snap: one frame into a change it sits
        // strictly between the segment it left and the one it is heading for,
        // and it comes to rest on the chosen one. Parking on Light first means
        // the travel measured is exactly one segment wide.
        self.click(self.theme_cells[ThemeChoice::Light.index()]);
        for _ in 0..20 {
            self.render(Instant::now());
        }
        let left = self.theme_cells[ThemeChoice::Light.index()];
        let right = self.theme_cells[ThemeChoice::Dark.index()];
        self.click(right);
        self.render(Instant::now());
        let (pixels, width, _height) = self.renderer.capture_rgba8();
        let glide = Self::pill_span(
            &pixels,
            width as usize,
            &self.theme,
            self.theme_cells.map(|cell| self.physical_rect(cell)),
        );
        println!("[drive] pill one frame in: {glide:?} (left {left:?}, target {right:?})");
        check(
            glide.is_some_and(|(x0, x1)| {
                x0 as f32 > left.x + 1.0 && (x1 as f32) < right.max().0 - 1.0
            }),
            "the Theme pill glides between segments",
            &mut failures,
        );
        for _ in 0..20 {
            self.render(Instant::now());
        }
        let (pixels, width, _height) = self.renderer.capture_rgba8();
        let settled = Self::pill_span(
            &pixels,
            width as usize,
            &self.theme,
            self.theme_cells.map(|cell| self.physical_rect(cell)),
        );
        println!("[drive] pill settled: {settled:?}");
        check(
            settled.is_some_and(|(x0, x1)| {
                x0 as f32 >= right.x - 1.0 && (x1 as f32) <= right.max().0 + 1.0
            }),
            "the Theme pill settles on the chosen segment",
            &mut failures,
        );

        self.show_scene_manager();
        self.set_theme_choice(ThemeChoice::Dark);

        // 1b. A slider keeps tracking while the button is held.
        let slider = self.sharpness_rect;
        check(
            slider.w > 1.0,
            "inspector slider reports its rect",
            &mut failures,
        );
        self.pointer_moved(slider.center());
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        let grabbed = self.settings.sharpness;
        self.pointer_moved((slider.x + slider.w * 0.95, slider.center().1));
        for _ in 0..3 {
            self.tick(std::time::Instant::now());
        }
        let dragged = self.settings.sharpness;
        self.pointer_button(PointerButton::Left, false);
        self.tick(std::time::Instant::now());
        check(
            dragged > grabbed + 0.1,
            "slider keeps dragging while the button is held",
            &mut failures,
        );
        self.settings.sharpness = 0.6;

        // 1c. The viewport context menu: it opens only on the viewport page,
        // choosing an item acts and closes it, and clicking away clears the
        // latch. The app boots into the scene manager page, so the first
        // right-click must not open it.
        let anchor = self.layout.viewport.center();
        self.pointer_moved(anchor);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Right, true);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Right, false);
        self.tick(std::time::Instant::now());
        check(
            !self.menu_open,
            "right-click off the viewport page does not open the context menu",
            &mut failures,
        );
        self.load_debug_scene();
        self.tick(std::time::Instant::now());
        self.scene.orbit.yaw = 1.0;
        self.pointer_moved(anchor);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Right, true);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Right, false);
        self.tick(std::time::Instant::now());
        check(
            self.menu_open,
            "right-click on the viewport opens the context menu",
            &mut failures,
        );
        // First row: "Reset camera".
        self.pointer_moved((anchor.0 + 8.0, anchor.1 + 21.0));
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, false);
        self.tick(std::time::Instant::now());
        check(
            self.scene.orbit == scene::OrbitCamera::default(),
            "context menu item acts",
            &mut failures,
        );
        check(
            !self.menu_open,
            "choosing an item closes the context menu",
            &mut failures,
        );
        self.pointer_button(PointerButton::Right, true);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Right, false);
        self.tick(std::time::Instant::now());
        self.pointer_moved((self.layout.viewport.x + 40.0, self.layout.viewport.y + 40.0));
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, false);
        self.tick(std::time::Instant::now());
        check(
            !self.menu_open,
            "clicking away closes the context menu",
            &mut failures,
        );
        self.toggle_debug();
        self.menu_open = false;

        // 2. Scene manager verification and dedicated boot button.
        check(
            !self.scene.debug_loaded(),
            "app boots with no scene loaded",
            &mut failures,
        );
        check(
            self.main_pane_page == MainPanePage::SceneManager,
            "app boots into the scene manager page in the main pane",
            &mut failures,
        );

        // Test creating a scene in the scene manager
        let scenes_count_before = self.scene_manager.scenes.len();
        self.scene_manager.new_scene_name = "Studio Alpha".to_string();
        self.create_new_scene();
        check(
            self.scene_manager.scenes.len() == scenes_count_before + 1,
            "create scene adds to the scenes list",
            &mut failures,
        );
        check(
            self.main_pane_page == MainPanePage::Viewport,
            "creating a scene opens the viewport",
            &mut failures,
        );

        // Switch back to Scene Manager
        self.main_pane_page = MainPanePage::SceneManager;
        self.tick(std::time::Instant::now());

        // Dedicated boot button for debug scene
        let quads_empty = self.scene.quads;
        self.click(chrome::debug_button_rect(self));
        check(
            self.scene.debug_loaded(),
            "dedicated button boots the debug scene",
            &mut failures,
        );
        check(
            self.main_pane_page == MainPanePage::Viewport,
            "booting debug scene switches to viewport page",
            &mut failures,
        );
        self.tick(std::time::Instant::now());
        check(
            self.scene.quads > quads_empty && self.scene.debug_loaded(),
            "a loaded scene draws geometry",
            &mut failures,
        );

        // 3. Viewport drag orbits and pans in active viewport
        let yaw_before = self.scene.orbit.yaw;
        let pitch_before = self.scene.orbit.pitch;
        self.drag_in_viewport((120.0, 40.0));
        check(
            self.scene.orbit.yaw > yaw_before + 0.1,
            "drag right orbits the scene with the cursor",
            &mut failures,
        );
        check(
            self.scene.orbit.pitch < pitch_before - 0.1,
            "drag down tips the camera up",
            &mut failures,
        );

        // Shift-drag pans in screen plane
        self.scene.orbit = scene::OrbitCamera::default();
        let target_before = self.scene.orbit.target;
        self.drag_in_viewport_shifted((120.0, 60.0));
        let moved = self.scene.orbit.target - target_before;
        check(
            moved.x < -1.0,
            "pan right drags the target along -screen-right",
            &mut failures,
        );
        check(
            moved.y > 0.5 && moved.z < 0.0,
            "pan up moves along camera screen-up, not world Y",
            &mut failures,
        );
        self.key_with(Key::Char('z'), true);
        check(
            (self.scene.orbit.yaw - yaw_before).abs() < 0.01,
            "ctrl+z restores the camera",
            &mut failures,
        );

        // 4. Wheel over the viewport zooms IN for scroll-up; capture still works.
        let zoom_before = self.scene.orbit.distance;
        self.pointer_moved(self.layout.viewport.center());
        self.wheel(3.0);
        self.tick(std::time::Instant::now());
        for _ in 0..60 {
            self.update(1.0 / 60.0);
        }
        check(
            self.scene.orbit.distance < zoom_before - 0.1,
            "wheel up over the viewport zooms in",
            &mut failures,
        );
        let captures_before = self.captures;
        self.capture_png();
        check(
            self.captures > captures_before,
            "capture still writes a PNG",
            &mut failures,
        );

        // 5. The custom window bar's caption buttons and drag region
        let caption = |app: &App, index: usize| {
            let bar = app.layout.titlebar;
            let w = 46.0;
            Rect::new(bar.max().0 - w * (index as f32 + 1.0), bar.y, w, bar.h)
        };
        self.click(caption(self, 0));
        check(
            self.window_commands.contains(&WindowCommand::Close),
            "close button asks the window to close",
            &mut failures,
        );
        self.window_commands.clear();
        self.click(caption(self, 1));
        check(
            self.window_commands
                .contains(&WindowCommand::ToggleMaximize),
            "maximize button toggles maximize",
            &mut failures,
        );
        self.window_commands.clear();
        self.click(caption(self, 2));
        check(
            self.window_commands.contains(&WindowCommand::Minimize),
            "minimize button asks the window to minimize",
            &mut failures,
        );
        self.window_commands.clear();

        let bar_rect = Rect::new(
            self.layout.titlebar.x + 200.0,
            self.layout.titlebar.y + 8.0,
            10.0,
            10.0,
        );
        self.pointer_moved(bar_rect.center());
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        let moved = self.window_commands.contains(&WindowCommand::Move);
        self.pointer_button(PointerButton::Left, false);
        self.tick(std::time::Instant::now());
        check(moved, "title bar drag moves the window", &mut failures);
        self.window_commands.clear();
        self.tick(std::time::Instant::now());
        check(
            self.window_commands
                .iter()
                .filter(|c| **c == WindowCommand::Move)
                .count()
                <= 1,
            "window commands do not repeat across frames",
            &mut failures,
        );
        self.window_commands.clear();

        let edge_press = Rect::new(2.0, self.screen.1 * 0.5, 2.0, 2.0);
        self.pointer_moved(edge_press.center());
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        let resized = self
            .window_commands
            .contains(&WindowCommand::Resize(ResizeEdge::Left));
        self.pointer_button(PointerButton::Left, false);
        self.tick(std::time::Instant::now());
        check(resized, "resize frame resizes the window", &mut failures);
        self.window_commands.clear();

        // Leave the app on the dark palette, on the Scenes page.
        self.set_system_dark(Some(true));
        self.set_theme_choice(ThemeChoice::Dark);
        self.main_pane_page = MainPanePage::SceneManager;
        let checks = failures.len();
        println!("[drive] complete, {checks} failures");
        assert!(failures.is_empty(), "drive failures: {failures:?}");
    }

    /// A click at `rect`: move, press, release (three frames like a user).
    fn click(&mut self, rect: crate::ui::Rect) {
        let center = rect.center();
        self.pointer_moved(center);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, false);
        self.tick(std::time::Instant::now());
    }

    fn key_with(&mut self, key: Key, ctrl: bool) {
        if ctrl {
            self.key_event(Key::Control, true);
        }
        self.key_event(key, true);
        if ctrl {
            self.key_event(Key::Control, false);
        }
    }

    /// Drag inside the viewport: press at the centre, move, release.
    fn drag_in_viewport(&mut self, delta: (f32, f32)) {
        self.main_pane_page = MainPanePage::Viewport;
        let center = self.layout.viewport.center();
        self.pointer_moved(center);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        let target = (center.0 + delta.0, center.1 + delta.1);
        self.input.mouse = target;
        self.viewport_drag(delta.0, delta.1);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, false);
        self.tick(std::time::Instant::now());
    }

    /// Shift-drag (pan) inside the viewport.
    fn drag_in_viewport_shifted(&mut self, delta: (f32, f32)) {
        self.main_pane_page = MainPanePage::Viewport;
        let center = self.layout.viewport.center();
        self.pointer_moved(center);
        self.tick(std::time::Instant::now());
        self.key_event(Key::Shift, true);
        self.pointer_button(PointerButton::Left, true);
        self.tick(std::time::Instant::now());
        let target = (center.0 + delta.0, center.1 + delta.1);
        self.input.mouse = target;
        self.viewport_drag(delta.0, delta.1);
        self.tick(std::time::Instant::now());
        self.pointer_button(PointerButton::Left, false);
        self.key_event(Key::Shift, false);
        self.tick(std::time::Instant::now());
    }

    /// Span of the Theme selector's pill in a captured frame, in pixels: the
    /// widest run of pill-tinted columns across the selector's row. `None`
    /// when there is no pill to find, which is itself a failure worth naming.
    ///
    /// Sampled a few pixels inside the pill's top edge, above the label ink:
    /// a column counts as pill when it reads closer to the pill tint than to
    /// the track under it, which holds in either palette (the pill is *lighter*
    /// than the track in the dark theme and *darker* in the light one).
    fn pill_span(
        pixels: &[u8],
        width: usize,
        theme: &Theme,
        cells: [Rect; 3],
    ) -> Option<(usize, usize)> {
        let row = cells[0].y as usize + 4;
        let (from, to) = (cells[0].x as usize, cells[2].max().0 as usize);
        let pill = theme.g2.r as i32;
        let track = theme.g1.r as i32;
        let mut best: Option<(usize, usize)> = None;
        let mut run = 0usize;
        for x in from..to {
            let pixel = rgba_at(pixels, width as u32, (x as f32 + 0.5, row as f32 + 0.5)).0 as i32;
            let is_pill = (pixel - pill).abs() < (pixel - track).abs();
            if is_pill {
                run += 1;
                continue;
            }
            if run > best.map_or(0, |(x0, x1)| x1 - x0) {
                best = Some((x - run, x));
            }
            run = 0;
        }
        if run > best.map_or(0, |(x0, x1)| x1 - x0) {
            best = Some((to - run, to));
        }
        // Anything narrower than half a segment is label ink, not the pill.
        best.filter(|(x0, x1)| (x1 - x0) as f32 > (cells[1].w * 0.5))
    }

    fn assert_themed_ink(pixels: &[u8], width: usize, theme: &Theme, inspector: Rect) {
        let band_x = inspector.x as usize + 34..inspector.x as usize + 130;
        let band_y = inspector.y as usize + 8..inspector.y as usize + 30;
        let want = theme.fg.r;
        let mut closest = 255u8;
        for y in band_y {
            for x in band_x.clone() {
                let r = pixels[(y * width + x) * 4];
                if r.abs_diff(want) < closest.abs_diff(want) {
                    closest = r;
                }
            }
        }
        let delta = closest.abs_diff(want);
        println!(
            "[selfcheck] theme proof: section header ink {closest:#04x} (theme.fg {want:#04x}, delta {delta})"
        );
        assert!(
            delta <= 8,
            "section header ink {closest:#04x} is not the theme foreground {want:#04x}"
        );
    }

    /// The Theme selector must show exactly one filled segment: the chosen
    /// cell sits on the `g2` pill, the other two only on the `g1` track. Each
    /// cell is sampled left of its label, where nothing but the pill can be.
    fn assert_theme_selector(pixels: &[u8], width: usize, app: &App) {
        let theme = app.theme;
        let chosen = app.theme_choice.index();
        for choice in ThemeChoice::ALL {
            let cell = app.physical_rect(app.theme_cells[choice.index()]);
            assert!(
                cell.w > 1.0,
                "the Settings page did not publish a Theme cell"
            );
            let got = rgba_at(pixels, width as u32, (cell.x + 4.0, cell.center().1)).0;
            let want = if choice.index() == chosen {
                theme.g2
            } else {
                theme.g1
            };
            println!(
                "[selfcheck] theme selector: {} cell {got:#04x} (want {:#04x})",
                choice.label(),
                want.r
            );
            assert!(
                got.abs_diff(want.r) <= 2,
                "the {} Theme cell should read {:#04x}, got {got:#04x}",
                choice.label(),
                want.r
            );
        }
    }

    fn assert_caption_glyphs(pixels: &[u8], width: usize, titlebar: Rect) {
        let slot = |index: usize| -> (usize, usize) {
            let right = titlebar.max().0 as usize - index * crate::app::chrome::CAPTION_W as usize;
            (right - crate::app::chrome::CAPTION_W as usize, right)
        };
        let bar = pixels[(titlebar.y as usize + 4) * width + slot(2).0 - 2];
        let ink = |index: usize| -> (f32, f32, f32, f32) {
            let (x0, x1) = slot(index);
            let (mut min_x, mut max_x, mut min_y, mut max_y) =
                (usize::MAX, 0usize, usize::MAX, 0usize);
            for y in titlebar.y as usize + 1..titlebar.max().1 as usize - 1 {
                for x in x0..x1 {
                    if pixels[(y * width + x) * 4].abs_diff(bar) > 24 {
                        min_x = min_x.min(x);
                        max_x = max_x.max(x);
                        min_y = min_y.min(y);
                        max_y = max_y.max(y);
                    }
                }
            }
            if min_x == usize::MAX {
                return (0.0, 0.0, f32::MAX, f32::MAX);
            }
            let centre_x = (min_x + max_x) as f32 * 0.5;
            let centre_y = (min_y + max_y) as f32 * 0.5;
            let slot_x = (x0 + x1 - 1) as f32 * 0.5;
            let slot_y = (titlebar.y + titlebar.max().1 - 1.0) * 0.5;
            (
                (max_x - min_x + 1) as f32,
                (max_y - min_y + 1) as f32,
                centre_x - slot_x,
                centre_y - slot_y,
            )
        };
        let (close_w, close_h, close_dx, close_dy) = ink(0);
        let (square_w, square_h, square_dx, square_dy) = ink(1);
        let (rule_w, rule_h, rule_dx, rule_dy) = ink(2);
        println!(
            "[selfcheck] caption ink: rule {rule_w}x{rule_h} square {square_w}x{square_h} \
             cross {close_w}x{close_h}; off-centre ({rule_dx:+.1},{rule_dy:+.1}) \
             ({square_dx:+.1},{square_dy:+.1}) ({close_dx:+.1},{close_dy:+.1})"
        );
        // Ink-fit sizes glyphs by their ink box, so the square reads a
        // touch smaller than the cross (optical_k 0.9) — close in both
        // axes, not identical.
        assert!(
            (square_w - close_w).abs() <= 3.0 && (square_h - close_h).abs() <= 3.0,
            "the maximise square ({square_w}x{square_h}) should read as the close cross ({close_w}x{close_h})"
        );
        assert!(
            (rule_w - close_w).abs() <= 2.0,
            "the minimise rule ({rule_w}px) and the cross ({close_w}px) should be one width"
        );
        for (name, dx, dy) in [
            ("rule", rule_dx, rule_dy),
            ("square", square_dx, square_dy),
            ("cross", close_dx, close_dy),
        ] {
            assert!(
                dx.abs() <= 0.75 && dy.abs() <= 0.75,
                "the {name} glyph sits off-centre in its button ({dx:+.1}, {dy:+.1})"
            );
        }
    }

    fn selfcheck_apply(&mut self, shot: &SelfCheckShot) {
        self.menu_open = false;
        self.left_down = false;
        self.right_down = false;
        match shot {
            SelfCheckShot::Default => {
                self.set_theme_choice(ThemeChoice::Dark);
                if self.scene.debug_loaded() {
                    self.toggle_debug();
                }
                self.main_pane_page = MainPanePage::SceneManager;
                self.inspector.selected = Some(3);
                self.pointer_moved((2000.0, 1000.0));
            }
            SelfCheckShot::Dark => {
                self.set_theme_choice(ThemeChoice::Dark);
                self.main_pane_page = MainPanePage::SceneManager;
            }
            SelfCheckShot::Light => {
                self.set_theme_choice(ThemeChoice::Light);
                // The context menu only exists on the viewport page, so the
                // shot needs a bound viewport to show it.
                self.load_debug_scene();
                self.inspector.selected = Some(3);
                self.inspector.render_open = true;
                self.menu_open = true;
                self.menu_at = self.layout.viewport.center();
                self.menu_focus = 1;
                self.pointer_moved((2000.0, 1000.0));
            }
            SelfCheckShot::Hover => {
                self.set_theme_choice(ThemeChoice::Dark);
                if self.scene.debug_loaded() {
                    self.toggle_debug();
                }
                self.main_pane_page = MainPanePage::SceneManager;
                self.pointer_moved(chrome::settings_button_rect(self).center());
                self.left_down = false;
            }
            SelfCheckShot::Inspector => {
                self.set_theme_choice(ThemeChoice::Dark);
                if self.scene.debug_loaded() {
                    self.toggle_debug();
                }
                self.main_pane_page = MainPanePage::SceneManager;
                self.inspector.selected = Some(0);
                self.inspector.palette = Some(7);
                self.pointer_moved((
                    self.layout.inspector.x + 100.0,
                    self.layout.inspector.y + 200.0,
                ));
            }
            SelfCheckShot::Settings => {
                self.set_theme_choice(ThemeChoice::Dark);
                if self.scene.debug_loaded() {
                    self.toggle_debug();
                }
                self.main_pane_page = MainPanePage::Settings;
                self.pointer_moved((2000.0, 1000.0));
            }
            SelfCheckShot::Menu => {
                self.set_theme_choice(ThemeChoice::Dark);
                self.load_debug_scene();
                let at = self.layout.viewport.center();
                self.menu_open = true;
                self.menu_at = at;
                self.menu_focus = 1;
                self.pointer_moved((at.0 + 10.0, at.1 + 52.0));
            }
        }
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn paced_frames_hold_the_display_rate_through_late_wakes() {
        let period = Duration::from_secs_f32(1.0 / 360.0);
        let mut pacing = FramePacing {
            display_hz: Some(360.0),
            ..FramePacing::default()
        };
        let start = Instant::now();
        let mut now = start;
        let mut frames = 0u32;
        let mut arrival = start;
        let mut arrival_frames = 0u32;
        while now < start + Duration::from_secs(1) {
            now += Duration::from_millis(1);
            if pacing.due_now(now) {
                frames += 1;
            }
            if now >= arrival + period {
                arrival = now;
                arrival_frames += 1;
            }
        }
        assert!(
            (frames as f32 - 360.0).abs() < 8.0,
            "paced {frames} frames in a second on a 360 Hz display"
        );
        assert!(
            arrival_frames < frames - 15,
            "the arrival-anchored rule should sag ({arrival_frames} frames against the grid's {frames})"
        );
    }

    #[test]
    fn a_long_stall_re_anchors_instead_of_catching_up() {
        let mut pacing = FramePacing {
            display_hz: Some(240.0),
            ..FramePacing::default()
        };
        let period = pacing.period().expect("a display rate");
        let start = Instant::now();
        assert!(
            pacing.due_now(start),
            "the first paced frame is due at once"
        );
        assert!(!pacing.due_now(start + period / 2), "not due yet");
        let resumed = start + Duration::from_millis(300);
        assert!(pacing.due_now(resumed), "the frame after a stall is due");
        assert!(!pacing.due_now(resumed + period / 2));
        assert!(pacing.due_now(resumed + period));
    }

    #[test]
    fn a_monitor_change_re_anchors_the_grid() {
        let mut pacing = FramePacing::default();
        let start = Instant::now();
        assert!(pacing.due_now(start));
        pacing.set_display_hz(Some(60.0));
        assert!(pacing.due_now(start + Duration::from_millis(2)));
        assert!(!pacing.due_now(start + Duration::from_millis(4)));
        assert!(pacing.due_now(start + Duration::from_millis(20)));
        pacing.set_display_hz(Some(0.0));
        pacing.set_display_hz(Some(f32::NAN));
        assert_eq!(pacing.display_hz, None);
        assert!(pacing.due_now(start + Duration::from_millis(21)));
    }

    #[test]
    fn pacing_sync_moves_the_deadline_one_slot_ahead() {
        let mut pacing = FramePacing {
            display_hz: Some(60.0),
            ..FramePacing::default()
        };
        let period = pacing.period().expect("a display rate");
        let now = Instant::now();
        pacing.sync(now);
        assert!(
            !pacing.due_now(now),
            "the frame just answered cannot be due again"
        );
        assert!(!pacing.due_now(now + period / 2));
        assert!(pacing.due_now(now + period));
    }

    #[test]
    fn pacing_missed_periods_answer_once_and_stay_on_grid() {
        let mut pacing = FramePacing {
            display_hz: Some(120.0),
            ..FramePacing::default()
        };
        let period = pacing.period().expect("a display rate");
        let start = Instant::now();
        assert!(pacing.due_now(start), "first paced frame is due at once");
        let late = start + period * 4 + period / 2;
        assert!(pacing.due_now(late), "the late frame answers once");
        assert!(
            !pacing.due_now(late + Duration::from_millis(1)),
            "missed periods are skipped, not burst-fired"
        );
        let expected = start + period * 5;
        assert_eq!(
            pacing.next_due(late),
            Some(expected),
            "the next deadline is grid-aligned and strictly future"
        );
    }

    #[test]
    fn pacing_nonfinite_public_rates_do_not_panic() {
        for hz in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
            let mut pacing = FramePacing {
                display_hz: Some(hz),
                cap_hz: Some(hz),
                ..FramePacing::default()
            };
            assert_eq!(pacing.period(), None, "{hz} is not a rate");
            let now = Instant::now();
            assert!(pacing.due_now(now), "unpaced answers every wake");
        }
    }

    #[test]
    fn pacing_low_positive_cap_remains_capped() {
        for cap in [0.5, 1.0] {
            let pacing = FramePacing {
                display_hz: Some(360.0),
                cap_hz: Some(cap),
                ..FramePacing::default()
            };
            assert_eq!(pacing.target_hz(), Some(cap));
            assert!(
                (pacing.period().unwrap() - Duration::from_millis(100)) < Duration::from_millis(1),
                "the 10 Hz floor gives ~100ms: {:?}",
                pacing.period()
            );
        }
    }

    #[test]
    fn the_cap_and_the_display_bound_each_other() {
        let with = |display_hz, cap_hz| FramePacing {
            display_hz,
            cap_hz,
            ..FramePacing::default()
        };
        assert_eq!(with(Some(357.0), Some(60.0)).target_hz(), Some(60.0));
        assert_eq!(with(Some(60.0), Some(357.0)).target_hz(), Some(60.0));
        assert_eq!(with(Some(144.0), None).target_hz(), Some(144.0));
        assert_eq!(with(None, Some(30.0)).target_hz(), Some(30.0));
        assert_eq!(with(None, None).target_hz(), None);
        let fast = with(None, Some(1.0e6)).period().expect("a capped rate");
        assert!(fast >= Duration::from_millis(1), "{fast:?}");
        assert!(with(None, Some(1.0e6)).target_hz().is_some());
    }
}
