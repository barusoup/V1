//! CPU allocation and timing probe, separate from the engine/driver process.
//! Requested Rust heap bytes exclude allocator overhead and native font caches.
//! Process RSS is reported separately; it is not labelled "UI memory".

use dream::ui::Rect;
use dream::ui::atlas::AtlasConfig;
use dream::ui::font::{FontId, Fonts};
use dream::ui::icons::Icon;
use dream::ui::tween::Ease;
use dream::ui::widgets::{ButtonStyle, LabelStyle, TextFieldStyle, Ui, UiInput};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct CountingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

fn allocated(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
    CALLS.fetch_add(1, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            allocated(size);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[cfg(target_os = "macos")]
fn resident_bytes() -> Option<(u64, u64)> {
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    // SDK mach/task_info.h: MACH_TASK_BASIC_INFO is 12 natural_t words.
    // The first three fields are virtual, resident and peak resident u64s.
    let mut words = [0u32; 12];
    let mut count = words.len() as u32;
    let result = unsafe { task_info(mach_task_self_, 20, words.as_mut_ptr().cast(), &mut count) };
    if result != 0 || count < 6 {
        return None;
    }
    let wide = |at: usize| {
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&words[at].to_ne_bytes());
        bytes[4..].copy_from_slice(&words[at + 1].to_ne_bytes());
        u64::from_ne_bytes(bytes)
    };
    Some((wide(2), wide(4)))
}

#[cfg(windows)]
fn resident_bytes() -> Option<(u64, u64)> {
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
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (ok != 0).then_some((
        counters.working_set_size as u64,
        counters.peak_working_set_size as u64,
    ))
}

#[cfg(not(any(target_os = "macos", windows)))]
fn resident_bytes() -> Option<(u64, u64)> {
    None
}

fn make_ui(scale: f32) -> Ui {
    let fonts = Fonts::bundled();
    let mut ui = Ui::new(fonts, AtlasConfig::default());
    ui.set_scale(scale);
    ui.register_vector_icons(&Icon::ALL);
    ui
}

fn labels(ui: &mut Ui, texts: &[String]) {
    for (i, text) in texts.iter().enumerate() {
        let rect = Rect::new((i % 20) as f32 * 120.0, (i / 20) as f32 * 28.0, 118.0, 26.0);
        ui.label(rect, text, LabelStyle::new(FontId(0)).size(16.0));
    }
}

fn workload(ui: &mut Ui, texts: &mut [String], pool: &[String], mode: u8, frame_no: usize) {
    match mode {
        0 => labels(ui, &texts[..200]),
        1 => {
            for i in 0..200 {
                let id = format!("Row {}##{}", i, frame_no * 200 + i);
                ui.button(
                    &id,
                    Rect::new((i % 20) as f32 * 120.0, (i / 20) as f32 * 28.0, 118.0, 26.0),
                    ButtonStyle::new(FontId(0)),
                );
            }
        }
        2 => ui.label(
            Rect::new(0.0, 0.0, 2200.0, 100.0),
            "Raster stress 0123456789",
            LabelStyle::new(FontId(0)).size(10.0 + (frame_no % 161) as f32 * 0.125),
        ),
        3 => ui.label(
            Rect::new(0.0, 0.0, 2200.0, 100.0),
            "\u{4e2d}\u{6587} \u{65e5}\u{672c}\u{8a9e} \u{d55c}\u{ad6d}\u{c5b4} \
                 \u{1f600} \u{1f469}\u{200d}\u{1f4bb} \u{1f1ef}\u{1f1f5} \u{1f44d}\u{1f3fd}",
            LabelStyle::new(FontId(0)).size(16.0 + (frame_no % 17) as f32 * 0.5),
        ),
        4 => {
            for (i, text) in texts[..400].iter().enumerate() {
                ui.push_id(text);
                let target = ((frame_no / 30 + i) % 2) as f32;
                let offset = ui.anim_f32("slide", target * 8.0, 0.2, Ease::CubicOut);
                ui.draw_list_mut().push_clip(Rect::new(
                    (i % 20) as f32 * 120.0,
                    (i / 20) as f32 * 28.0,
                    120.0,
                    28.0,
                ));
                ui.button(
                    text,
                    Rect::new(
                        (i % 20) as f32 * 120.0 + offset,
                        (i / 20) as f32 * 28.0,
                        108.0,
                        26.0,
                    ),
                    ButtonStyle::new(FontId(0)),
                );
                ui.draw_list_mut().pop_clip();
                ui.pop_id();
            }
        }
        5 => {
            for (i, text) in texts[..200].iter_mut().enumerate() {
                ui.push_id(&pool[i]);
                ui.text_field(
                    "field",
                    Rect::new((i % 20) as f32 * 120.0, (i / 20) as f32 * 28.0, 118.0, 26.0),
                    text,
                    TextFieldStyle::new(FontId(0)),
                );
                ui.pop_id();
            }
        }
        6 => {
            for i in 0..200 {
                ui.label(
                    Rect::new((i % 20) as f32 * 120.0, (i / 20) as f32 * 28.0, 118.0, 26.0),
                    &pool[(frame_no * 200 + i) % pool.len()],
                    LabelStyle::new(FontId(0)).size(16.0),
                );
            }
        }
        7 => labels(ui, texts),
        // L1 equivalent of the L3 settings form below.
        8 => {
            let mut seg = 1usize;
            let mut on = false;
            let mut v = 0.5f32;
            let theme = *ui.theme();
            let btn = theme.button(FontId(0));
            ui.button("Done", Rect::new(260.0, 8.0, 60.0, 24.0), btn);
            ui.label(
                Rect::new(8.0, 8.0, 200.0, 24.0),
                "Settings",
                LabelStyle::new(FontId(0)).size(16.0),
            );
            ui.label(
                Rect::new(8.0, 44.0, 300.0, 20.0),
                "Section",
                LabelStyle::new(FontId(0)).size(12.0),
            );
            for (i, name) in pool.iter().enumerate().take(3) {
                let y = 72.0 + i as f32 * 40.0;
                ui.label(
                    Rect::new(8.0, y, 100.0, 20.0),
                    name,
                    LabelStyle::new(FontId(0)).size(13.0),
                );
                match i {
                    0 => {
                        let opts = ["A", "B"];
                        let st = theme.segmented(FontId(0));
                        ui.segmented_styled(
                            "seg",
                            Rect::new(120.0, y, 180.0, 24.0),
                            &opts,
                            &mut seg,
                            &st,
                        );
                    }
                    1 => {
                        ui.switch("sw", Rect::new(120.0, y, 180.0, 24.0), &mut on, btn);
                    }
                    _ => {
                        let sst = theme.slider(FontId(0));
                        ui.slider(
                            "sl",
                            Rect::new(120.0, y, 180.0, 24.0),
                            &mut v,
                            0.0..=1.0,
                            sst,
                        );
                    }
                }
            }
            ui.label(
                Rect::new(8.0, 196.0, 300.0, 14.0),
                "caption",
                LabelStyle::new(FontId(0)).size(11.0),
            );
        }
        // Same form through the L3 view layer.
        9 => {
            let mut seg = 1usize;
            let mut on = false;
            let mut v = 0.5f32;
            let opts = ["A", "B"];
            ui.layout_root(
                Rect::new(0.0, 0.0, 400.0, 220.0),
                dream::ui::layout::Style::column().pad(8.0).gap(8.0),
                |ui| {
                    ui.row(
                        dream::ui::layout::Style::default().fill_w().h_px(28.0),
                        |ui| {
                            ui.text("Settings");
                            ui.spacer();
                            ui.add(dream::ui::widgets::adapters::Button::new("Done"));
                        },
                    );
                    ui.text("Section");
                    for (i, name) in pool.iter().enumerate().take(3) {
                        ui.row(
                            dream::ui::layout::Style::default()
                                .fill_w()
                                .h_px(26.0)
                                .gap(8.0),
                            |ui| {
                                ui.text(name);
                                match i {
                                    0 => {
                                        ui.add(dream::ui::widgets::adapters::Segmented::new(
                                            "seg", &opts, &mut seg,
                                        ));
                                    }
                                    1 => {
                                        ui.add(dream::ui::widgets::adapters::Switch::new(
                                            "sw", &mut on,
                                        ));
                                    }
                                    _ => {
                                        ui.add(dream::ui::widgets::adapters::Slider::new(
                                            "sl",
                                            &mut v,
                                            0.0..=1.0,
                                        ));
                                    }
                                }
                            },
                        );
                    }
                    ui.text("caption");
                },
            );
        }
        _ => unreachable!(),
    }
}

fn run(name: &str, scale: f32, mode: u8) {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let mut ui = make_ui(scale);
    let mut texts: Vec<String> = (0..1200).map(|i| format!("Status {i:04}")).collect();
    let pool: Vec<String> = (0..2048).map(|i| format!("Value {i:04}")).collect();
    let screen = if mode == 7 {
        (2400.0, 1800.0)
    } else {
        (2400.0, 600.0)
    };
    for frame_no in 0..30 {
        ui.begin(
            UiInput {
                dt: 1.0 / 60.0,
                ..Default::default()
            },
            screen,
        );
        workload(&mut ui, &mut texts, &pool, mode, frame_no);
        let frame = ui.end();
        ui.recycle_frame(frame);
    }
    let mut samples = Vec::with_capacity(1000);
    let mut begin_samples = Vec::with_capacity(1000);
    let mut widget_samples = Vec::with_capacity(1000);
    let mut end_samples = Vec::with_capacity(1000);
    let calls_before = CALLS.load(Ordering::Relaxed);
    let stable_live = LIVE.load(Ordering::Relaxed);
    let mut retained = std::collections::VecDeque::new();
    let mut steady_allocs: u64 = 0;
    let mut steady_frames: u64 = 0;
    for frame_no in 0..1000 {
        let frame_calls = CALLS.load(Ordering::Relaxed);
        let started = Instant::now();
        ui.begin(
            UiInput {
                dt: 1.0 / 60.0,
                ..Default::default()
            },
            screen,
        );
        let after_begin = Instant::now();
        workload(&mut ui, &mut texts, &pool, mode, frame_no + 30);
        let after_widgets = Instant::now();
        retained.push_back(ui.end());
        std::hint::black_box(ui.animating());
        let after_end = Instant::now();
        begin_samples.push(after_begin.duration_since(started).as_nanos());
        widget_samples.push(after_widgets.duration_since(after_begin).as_nanos());
        end_samples.push(after_end.duration_since(after_widgets).as_nanos());
        // Model two in-flight frame snapshots, not an unbounded archive.
        if retained.len() > 2 {
            ui.recycle_frame(retained.pop_front().expect("three frames"));
        }
        samples.push(started.elapsed().as_nanos());
        if frame_no >= 500 {
            steady_allocs += (CALLS.load(Ordering::Relaxed) - frame_calls) as u64;
            steady_frames += 1;
        }
    }
    let steady = steady_allocs as f64 / steady_frames.max(1) as f64;
    let calls = CALLS.load(Ordering::Relaxed) - calls_before;
    let ending = LIVE.load(Ordering::Relaxed);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
    samples.sort_unstable();
    begin_samples.sort_unstable();
    widget_samples.sort_unstable();
    end_samples.sort_unstable();
    let stats = ui.layout_stats();
    let frame = retained.back().expect("retained frame");
    let p50 = samples[500];
    let p95 = samples[950];
    let rss = resident_bytes();
    println!(
        "{name},scale={scale},frames=1000,p50_ns={p50},p95_ns={p95},p99_ns={},max_ns={},alloc_calls={calls},\
         retained_rust_bytes={},peak_rust_bytes={peak},growth_bytes={},\
         process_rss_bytes={},process_peak_rss_bytes={},begin_p50_ns={},widgets_p50_ns={},end_p50_ns={},end_p99_ns={},steady_allocs_per_frame={steady:.2},layout_hits={},layout_misses={},layout_entries={},layout_bytes={},vertices={},indices={},commands={}",
        samples[990],
        samples[999],
        ending.saturating_sub(baseline),
        ending.saturating_sub(stable_live),
        rss.map_or(0, |value| value.0),
        rss.map_or(0, |value| value.1),
        begin_samples[500],
        widget_samples[500],
        end_samples[500],
        end_samples[990],
        stats.hits,
        stats.misses,
        stats.entries,
        stats.retained_bytes,
        frame.vertices.len(),
        frame.indices.len(),
        frame.cmds.len(),
    );
    assert!(peak < 80_000_000, "{name}: Rust heap exceeds 80 decimal MB");
}

fn main() {
    println!(
        "Requested Rust allocations; RSS includes executable/native caches. GPU not included."
    );
    for scale in [1.0, 2.0] {
        run("stable_labels", scale, 0);
        run("dynamic_widgets", scale, 1);
        run("size_churn", scale, 2);
        if std::env::args().any(|arg| arg == "--apps") {
            run("l1_form", scale, 8);
            run("l3_form", scale, 9);
            run("animated_controls", scale, 4);
            run("idle_text_fields", scale, 5);
            run("layout_churn", scale, 6);
            run("layout_pressure", scale, 7);
        }
        if std::env::args().any(|arg| arg == "--unicode") {
            run("cjk_emoji", scale, 3);
        }
    }
}
