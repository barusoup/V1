# BedFrame cookbook

Recipes for the declarative layer (`dream::ui::view`). Geometry comes from the layout solver.

## Minimal app

Start an app with `dream::ui::app::run`. It opens the window and creates the renderer and the `Ui`, maps input, and handles DPI, theme, clipboard and IME. Your code only builds a frame:

```rust,ignore
use dream::ui::app::{AppContext, AppOptions, RunError, UiApp, run};
use dream::ui::layout::Style;
use dream::ui::tokens::TypeRole;
use dream::ui::widgets::Ui;
use dream::ui::widgets::adapters::{Button, Text};

struct Counter { count: u32 }

impl UiApp for Counter {
    fn view(&mut self, ui: &mut Ui, cx: &mut AppContext) {
        ui.layout_root(cx.screen(), Style::column().pad(24.0).gap(12.0), |ui| {
            ui.add(Text::new(&format!("Count: {}", self.count)).role(TypeRole::Title));
            if ui.add(Button::new("Increment")).clicked { self.count += 1; }
        });
    }
}

fn main() -> Result<(), RunError> {
    run(AppOptions::new("Counter"), Counter { count: 0 })
}
```

`examples/hello.rs` is a fuller starting point. The runner is idle-correct by default:

- It parks at 0 Hz when nothing changes: no input, no animation and no UI timer. A frame is built only when something can have changed.
- It stops rendering while the window is minimised or occluded, and it releases scene resources and trims memory while hidden.
- It uses FIFO vsync (`AppOptions::present_mode` changes this). Animations pace to the display.
- Idle memory is trimmed back to the OS when the app is quiet (`AppOptions::idle_trim`).

Data that arrives on another thread wakes the app through an `AppWaker`. Store the data first, then call `wake()`. Spawn the loader from `init`, which runs once:

```rust,ignore
let (waker, data) = (cx.waker(), Arc::clone(&self.data)); // data: Arc<Mutex<Option<String>>>
std::thread::spawn(move || {
    *data.lock().unwrap() = Some(load_slowly());
    waker.wake(); // builds one frame that shows the data
});
```

## Basics

```rust,ignore
ui.layout_root(area, Style::column().pad(8.0).gap(4.0), |ui| {
    ui.add(Text::new("Title").role(TypeRole::Heading));
    if ui.add(Button::new("Done")).clicked { /* … */ }
});
```

- The build closure is `FnMut` and may run twice per frame. It must be idempotent: record state from events (`if resp.clicked { count += 1 }`), never assign from responses unconditionally.
- `ui.rect_of("key")` returns the solved rect after `layout_root` returns.
- `ui.draw` and `DrawList` work inside `slot` closures.

## Toolbar

```rust,ignore
ui.layout_root(area, Style::row().gap(4.0).align(Align::Center), |ui| {
    ui.add(IconButton::new("##back", Icon::ChevronLeft));
    ui.add_styled(Text::new("Scene").role(TypeRole::Heading), Style::default().grow(1.0));
    ui.add(Button::new("Play").variant(ButtonVariant::Primary));
});
```

## Form rows

There is no `Form` widget yet. Use a column of two-column rows:

```rust,ignore
ui.layout_root(area, Style::column().pad(12.0).gap(8.0), |ui| {
    for (label, value) in fields.iter_mut() {
        ui.row(Style::default().justify(Justify::SpaceBetween).align(Align::Center), |ui| {
            ui.add(Text::new(label));
            ui.add_styled(TextField::new(&format!("##{label}"), value),
                Style::default().w_px(180.0));
        });
    }
});
```

## Cards

There is no `Grid` widget yet. Use fixed-height rows:

```rust,ignore
ui.layout_root(area, Style::column().gap(8.0), |ui| {
    for row in cards.chunks_mut(2) {
        ui.row(Style::default().gap(8.0).fill_w(), |ui| {
            for card in row {
                ui.container(
                    Style::column().grow(1.0).h_px(120.0).pad(8.0).gap(4.0),
                    Paint::surface(theme),
                    |ui| {
                        ui.add(Text::new(&card.title).role(TypeRole::Body));
                        ui.add(Text::new(&card.sub).role(TypeRole::Caption));
                    },
                );
            }
        });
    }
});
```

## Shapes and circles

Use `Shape` for any geometric leaf: a dot, a ring, a progress arc, a pill.

```rust,ignore
use std::f32::consts::{FRAC_PI_2, TAU};

ui.layout_root(area, Style::column().gap(8.0), |ui| {
    ui.row(Style::row().gap(8.0).align(Align::Center), |ui| {
        ui.add(Shape::circle(24.0).fill(status_green));   // status dot
        ui.add(Text::new("Online"));
    });
    ui.add(Shape::ring(32.0, 3.0));                        // accent ring (theme accent)
    ui.add(Shape::arc(32.0, 3.0, -FRAC_PI_2, progress * TAU).fill(theme.accent));
    ui.add(Shape::pill(64.0, 20.0).fill(chip_bg));
    ui.add(Shape::circle(20.0).fill(bg).stroke(2.0, border)); // bordered disc
});
```

- `Shape` has a fixed size, so a row or column never stretches it. Circle, ring and arc paint the largest centred square of their box, so they stay round under any style override.
- Without a fill or a stroke, a shape is filled with the theme accent. For `ring` and `arc`, `fill` sets the band colour and `stroke(w, c)` sets its thickness and colour.
- Arc angles are radians with `0` on `+x`, clockwise on screen, so `-FRAC_PI_2` is the top.

**Use `Shape::circle` or `DrawList::fill_circle`, never a polygon.** Circles are exact signed-distance shapes with one physical pixel of anti-aliasing, so they need no render-scale tricks and stay sharp at any size.

**The stretch pitfall.** A row or column stretches each child on its cross axis (`Align::Stretch` is the default) unless the child has an explicit cross size. A box with a magic corner radius in a 60 px row therefore turns into a tall pill. Give the box an explicit size with `w_px` / `h_px`, or align it with `align_self(Align::Center)`. For a circle, use `Shape`, which is never stretched.

**Pills and avatars with containers.** `Paint::radius_full()` is the CSS `border-radius: 9999px` idiom. Each corner is clamped to half the short side when painted, so a square container becomes a circle and a wide one a pill. Children clip to the same curve.

```rust,ignore
ui.container(
    Style::row().size_px(48.0, 48.0).align(Align::Center).justify(Justify::Center),
    Paint::surface(theme).radius_full(),
    |ui| { ui.add(Text::new("JD")); },
);
```

**Immediate-mode painting** inside a `slot`, for custom shapes that are not leaf widgets:

```rust,ignore
ui.slot(Style::default().size_px(32.0, 32.0), move |ui, rect| {
    let c = (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5);
    ui.paint_circle(c, 14.0, bg);                              // filled disc
    ui.paint_circle_stroke(c, 14.0, 2.0, border);              // outline, inward band
    ui.draw.ring_arc(rect, 14.0, 3.0, -FRAC_PI_2, progress * TAU, accent);
});
```

`ui.draw.fill_circle` and `ui.draw.stroke_circle` take the same arguments as `paint_circle` and `paint_circle_stroke`. `ring_arc` takes the bounding rect instead of a centre, with the radius measured from its centre.

## Scrollable page

Reference: `src/app/settings_page.rs`.

```rust,ignore
ui.layout_root(page, Style::column().fill(), |ui| {
    ui.container(Style::column().fill().scroll_y(), Paint::none(), |ui| {
        ui.column(Style::column().pad(Space::S6).gap(Space::S5).max_w(440.0), |ui| {
            ui.add(Text::new("Settings").role(TypeRole::Title));
        });
    });
});
```

## Scroll feel

Every scroll view shares one physics model. Tune it globally on `ScrollConfig`:

```rust,ignore
ui.scroll_config = ScrollConfig {
    line_height: 40.0,     // pixels per wheel line
    smoothness: 0.06,      // glide time constant; smaller is snappier
    page_fraction: 0.85,   // PageUp/PageDown and track clicks
    keyboard: true,        // arrows and page keys scroll the hovered view
    chain: true,           // wheel overflow passes to the parent view
    ..ScrollConfig::default()
};

ui.scroll_config = ScrollConfig::no_momentum(); // notches jump, nothing coasts
```

Override per view by saving and restoring around the build:

```rust,ignore
let saved = ui.scroll_config;
ui.scroll_config = ScrollConfig::no_momentum();
let body = ui.scroll_begin("inspector", area, content_h);
// …rows at `body`…
ui.scroll_end();
ui.scroll_config = saved;
```

Programmatic scrolling. Calls may come before or after the view builds:

```rust,ignore
ui.scroll_to("log", ScrollToOptions::smooth(f32::INFINITY)); // pin to end
ui.scroll_by("log", -48.0, ScrollBehavior::Auto);
ui.scroll_into_view("files", Rect::new(0.0, row_y, width, row_h),
    ScrollIntoViewOptions { block: ScrollAlign::Nearest, behavior: ScrollBehavior::Smooth });
```

Read state with `scroll_state(label)` (`offset`, `content`) or `scroll_metrics(label)` (adds `viewport`, `max_offset`, `velocity`, `animating`, `progress()`).

## HUD overlay

`stack` places children by offset, so overlays do not fight the layout:

```rust,ignore
ui.layout_root(screen, Style::column().fill(), |ui| {
    ui.stack(Style::default().fill(), |ui| {
        ui.add_styled(Text::new("60 fps").role(TypeRole::Caption),
            Style::default().place(AlignX::Start, AlignY::Start));
        ui.add_styled(IconView::new("##aim", Icon::Plus),
            Style::default().place(AlignX::Center, AlignY::Center));
    });
});
```

## State and context

`use_state` returns a shared handle. Its key follows `push_id` scopes, and the initializer runs once even when layout replays:

```rust,ignore
let count = ui.use_state("counter", || 0usize);
ui.with_context(&count, |ui| {
    ui.layout_root(area, Style::column().gap(8.0), |ui| {
        let count = ui.use_context::<usize>().expect("counter provider");
        ui.add(Text::new(&format!("Count: {}", count.get())));
        if ui.add(Button::new("Increment")).clicked {
            count.update(|n| *n += 1);
        }
    });
});
```

- The nearest provider of a type wins. Use newtypes to separate contexts that share a primitive type.
- `State::new(value)` creates a host-owned provider.
- State changed outside a UI frame must schedule a frame.

## Presence

Keep a widget mounted while it animates out:

```rust,ignore
let p = ui.presence("notice", visible, 0.18);
if p.is_present() {
    let color = theme.fg.with_alpha((255.0 * p.opacity).round() as u8);
    ui.label(notice_rect, "Saved", theme.label(font).color(color));
}
```

- Call `presence` every frame for the same key.
- During exit, `is_interactive()` is false. Disable the widget's controls then.
- Reduced motion settles immediately.

## Keeping RAM low

A UI that stays still keeps roughly 100 MB of cold driver and loader pages resident. `dream::memory::trim_working_set` returns those pages to the OS. It lowers the working set (the figure Task Manager shows), not committed bytes. The pages are soft-faulted back in cheaply on the next frame.

`dream::memory::IdleTrim` decides when to trim: once after warm-up, and once after each quiet period.

```rust,ignore
use dream::memory::IdleTrim;
use std::time::Instant;
use winit::event_loop::ControlFlow;

// In the app state:
let mut trim = IdleTrim::new();

// Whenever a frame is built or input arrives:
trim.activity(Instant::now());

// In `about_to_wait`:
let now = Instant::now();
trim.trim_if_due(now);
match trim.next_deadline() {
    Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
    None => event_loop.set_control_flow(ControlFlow::Wait),
}
```

## Low-resource defaults

`dream::ui::app::run` applies all of these. Apply them yourself when you own the event loop:

- **Present mode.** Frames present with FIFO (vsync). `Renderer::set_present_mode(PresentMode::Mailbox)` lowers latency for games. Mailbox does not pace the app, so such a game must pace its own frames to the monitor refresh.
- **Direct composition.** Frames are composed straight into the swapchain, with no intermediate image. A crosshair or frame capture needs the composed image, so the renderer allocates one. Call `renderer.set_frame_capture(true)` before rendering the frames you read back with `capture_rgba8`, `save_png` or `frame_metrics`.
- **UI blend.** The UI blends in sRGB space, as browsers do. `Renderer::new_with_ui_blend(.., UiBlend::Linear)` opts into linear light.
- **Pacing.** Pace animation in software, to the monitor refresh, and do not rely on blocking in the swapchain. Some drivers (NVIDIA on Windows) busy-wait inside `vkAcquireNextImageKHR` and `vkWaitForFences`, which burns a whole CPU core at high refresh rates. Measured: an animating example went from about 109% to about 8% of one core after pacing in software.
- **Hidden windows.** When the window is hidden, call `renderer.release_scene_resources()` and `renderer.trim_memory()`.
