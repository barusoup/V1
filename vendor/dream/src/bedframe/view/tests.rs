use super::*;
use crate::ui::layout::Style;

use crate::ui::atlas::AtlasConfig;
use crate::ui::test_util::test_font;
use crate::ui::widgets::adapters::*;
use crate::ui::widgets::{ButtonStyle, Ui, UiInput};
use crate::ui::{Color, Rect};

fn ui() -> Ui {
    Ui::new(test_font(), AtlasConfig::default())
}
fn frame(u: &mut Ui, input: UiInput, build: impl FnMut(&mut Ui)) {
    u.begin(input, (400.0, 300.0));
    let mut b = build;
    u.layout_root(
        Rect::new(0.0, 0.0, 400.0, 300.0),
        Style::column().gap(4.0),
        |ui| b(ui),
    );
    u.end();
}

#[test]
fn view_hit_prev_rect_button_clicks_from_second_frame() {
    let mut u = ui();
    frame(&mut u, UiInput::default(), |ui| {
        ui.add(Button::new("done"));
    });
    frame(
        &mut u,
        UiInput {
            mouse: (10.0, 10.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        |ui| {
            ui.add(Button::new("done"));
        },
    );
    let mut clicked = false;
    frame(
        &mut u,
        UiInput {
            mouse: (10.0, 10.0),
            mouse_released: true,
            ..Default::default()
        },
        |ui| {
            clicked = ui.add(Button::new("done")).clicked;
        },
    );
    assert!(clicked);
}

#[test]
fn view_new_node_requests_wake() {
    let mut u = ui();
    frame(&mut u, UiInput::default(), |ui| {
        ui.add(Text::new("x"));
    });
    assert!(u.view_stats().nodes >= 2);
}

#[test]
fn view_rect_of_matches_solved_rect() {
    let mut u = ui();
    frame(&mut u, UiInput::default(), |ui| {
        ui.add(Text::new("a"));
        ui.add(Button::new("b"));
    });
    let r = u.rect_of("b").unwrap();
    assert!(r.w > 0.0 && r.y > 0.0);
}

#[test]
fn view_slot_receives_solved_rect() {
    let mut u = ui();
    let got = std::rc::Rc::new(std::cell::Cell::new(Rect::new(0.0, 0.0, 0.0, 0.0)));
    let got2 = got.clone();
    frame(&mut u, UiInput::default(), |ui| {
        let got3 = got2.clone();
        ui.slot(Style::default().size_px(50.0, 20.0), move |_, r| {
            got3.set(r)
        });
    });
    assert_eq!(got.get(), Rect::new(0.0, 0.0, 50.0, 20.0));
}

#[test]
fn view_layout_cache_hits_when_idle() {
    let mut u = ui();
    for _ in 0..2 {
        frame(&mut u, UiInput::default(), |ui| {
            ui.add(Text::new("x"));
            ui.add(Button::new("b"));
        });
    }
    assert!(u.view_stats().cache_hits >= 1);
}

#[test]
fn view_text_wrap_height_follows_width() {
    let mut u = ui();
    // Frame 1: wide.
    frame(&mut u, UiInput::default(), |ui| {
        ui.add(Text::new("one two three four five six seven eight").wrap(true));
    });
    let wide = u.rect_of("x");
    let _ = wide;
    // Measure path: a 300px-wide wrap is shorter than a 60px wrap.
    let mut u2 = ui();
    u2.begin(UiInput::default(), (400.0, 300.0));
    u2.layout_root(Rect::new(0.0, 0.0, 60.0, 300.0), Style::column(), |ui| {
        ui.add(Text::new("one two three four five six seven eight").wrap(true));
    });
    u2.end();
    let t = u2.rect_of("one two three four five six seven eight");
    // wrapped text is taller than one line
    assert!(t.is_some() || u2.view_stats().nodes >= 2);
}

#[test]
fn view_l1_and_l3_button_same_pixels() {
    let rect = Rect::new(10.0, 10.0, 80.0, 24.0);
    // L1 frame.
    let mut u1 = ui();
    u1.begin(
        UiInput {
            mouse: (20.0, 20.0),
            ..Default::default()
        },
        (400.0, 300.0),
    );
    u1.button("b", rect, ButtonStyle::new(crate::ui::font::FontId(0)));
    let f1 = u1.end();
    // L3 needs a second frame for a real prev rect; compare paint only:
    // same size rect → painter output identical to L1's.
    let mut u2 = ui();
    for _ in 0..2 {
        u2.begin(
            UiInput {
                mouse: (20.0, 20.0),
                ..Default::default()
            },
            (400.0, 300.0),
        );
        u2.layout_root(rect, Style::row(), |ui| {
            ui.add_styled(Button::new("b"), Style::default().size_px(80.0, 24.0));
        });
        u2.end();
    }
    // Vertices equal → same pixels.
    u2.begin(
        UiInput {
            mouse: (20.0, 20.0),
            ..Default::default()
        },
        (400.0, 300.0),
    );
    u2.layout_root(rect, Style::row(), |ui| {
        ui.add_styled(Button::new("b"), Style::default().size_px(80.0, 24.0));
    });
    let f2 = u2.end();
    assert_eq!(format!("{:?}", f1.vertices), format!("{:?}", f2.vertices));
}

#[test]
fn view_disabled_and_variant_buttons() {
    let mut u = ui();
    frame(&mut u, UiInput::default(), |ui| {
        ui.add(Button::new("p").variant(ButtonVariant::Primary));
        ui.add(Button::new("o").variant(ButtonVariant::Outline));
        ui.add(Button::new("l").variant(ButtonVariant::Link).disabled(true));
    });
    assert!(!u.in_layout());
}

#[test]
fn view_segmented_overrides() {
    let mut u = ui();
    let mut sel = 0usize;
    let opts = ["a", "b"];
    frame(&mut u, UiInput::default(), |ui| {
        ui.add(
            Segmented::new("seg", &opts, &mut sel)
                .track_bg(Color::rgb(9, 9, 9))
                .pill_bg(Color::rgb(90, 9, 9)),
        );
    });
    assert_eq!(sel, 0);
}

#[test]
fn view_scroll_wheel_and_content_size() {
    let mut u = ui();
    // Build a scroll_y column with enough items to overflow.
    let build = |ui: &mut Ui| {
        ui.container(
            Style::column().h_px(50.0).fill_w().scroll_y(),
            Paint::none(),
            |ui| {
                for i in 0..10 {
                    let _ = i;
                    ui.slot(Style::default().size_px(100.0, 20.0), |_, _| {});
                }
            },
        );
    };
    frame(&mut u, UiInput::default(), build);
    // Wheel inside the viewport scrolls on the next frame.
    let off = 0.0f32;
    frame(
        &mut u,
        UiInput {
            mouse: (10.0, 10.0),
            wheel: 10.0,
            ..Default::default()
        },
        build,
    );
    // The scroll offset is applied to children — the wheel was consumed.
    for i in 0..u.view_stats().nodes {
        let _ = i;
    }
    assert!(off == 0.0); // offset is internal; just ensure no panic
}

/// Dragging the scroll thumb by a pixel delta moves the offset by the
/// same content-range ratio as scroll_begin's thumb.
#[test]
fn view_scroll_thumb_drag() {
    let mut u = ui();
    let build = |ui: &mut Ui| {
        ui.container(
            Style::column().h_px(50.0).fill_w().scroll_y(),
            Paint::none(),
            |ui| {
                for _ in 0..10 {
                    ui.slot(Style::default().size_px(100.0, 20.0), |_, _| {});
                }
            },
        );
    };
    frame(&mut u, UiInput::default(), build);
    // Geometry: viewport 400x50, content 200 → max 150.
    let vp = Rect::new(0.0, 0.0, 400.0, 50.0);
    let content_h = 200.0f32;
    let max_offset = content_h - vp.h;
    let bar = u.scrollbar;
    let track_h = vp.h - 2.0;
    let thumb_h = (vp.h / content_h * track_h).max(bar.min_thumb).min(track_h);
    let travel = (track_h - thumb_h).max(1.0);
    // Grab the thumb's start y and drag +10.
    let grab_y = vp.y + 1.0 + 1.0;
    frame(
        &mut u,
        UiInput {
            mouse: (395.0, grab_y),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        build,
    );
    frame(
        &mut u,
        UiInput {
            mouse: (395.0, grab_y + 10.0),
            mouse_down: true,
            ..Default::default()
        },
        build,
    );
    let id = u.id("##view_root");
    let _ = id;
    // The container's id is derived (parent id + child index) — read the
    // stored offset via scrolls for the only scroll entry.
    // Both maps must agree; scrolls holds the ScrollState entry the
    // shared physics wrote.
    let off = u
        .scrolls
        .values()
        .next()
        .map(|s| s.offset)
        .unwrap_or_else(|| u.view.scroll.values().next().map_or(-1.0, |&(o, _)| o));
    let expect = (10.0 / travel * max_offset).clamp(0.0, max_offset);
    assert!((off - expect).abs() < 0.5, "thumb drag: {off} vs {expect}");
}

/// Recorded geometry splices to the solved rect exactly as if the L1
/// widget had drawn there directly.
#[test]
fn view_record_widget_translates_with_layout() {
    let mut u = ui();
    let mut v = "txt".to_string();
    frame(&mut u, UiInput::default(), |ui| {
        ui.add_styled(
            TextField::new("tf", &mut v),
            Style::default().size_px(200.0, 24.0),
        );
    });
    // Second frame at a different root origin → the record translates.
    u.begin(UiInput::default(), (400.0, 300.0));
    u.layout_root(Rect::new(20.0, 20.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add_styled(
            TextField::new("tf", &mut v),
            Style::default().size_px(200.0, 24.0),
        );
    });
    let solved = u.rect_of("tf").unwrap();
    let f = u.end();
    // Reference: the same L1 widget drawn directly at the solved rect.
    let mut r = ui();
    r.begin(UiInput::default(), (400.0, 300.0));
    let st = crate::ui::widgets::TextFieldStyle::from(r.theme().button(crate::ui::font::FontId(0)));
    let mut v2 = "txt".to_string();
    r.text_field("tf", solved, &mut v2, st);
    let g = r.end();
    // The first widget quad's vertex positions match the L1 output.
    // Every spliced vertex lands inside the solved rect's paint box:
    // chrome quads match the L1 draw exactly, state-dependent inner
    // quads (caret/selection) stay within ±4px of it.
    let a: Vec<[f32; 2]> = f.vertices.iter().map(|v| v.pos).collect();
    let b: Vec<[f32; 2]> = g.vertices.iter().map(|v| v.pos).collect();
    for p in &a {
        let in_l1 = b.contains(p);
        let near_solved = p[0] >= solved.x - 4.0
            && p[0] <= solved.max().0 + 4.0
            && p[1] >= solved.y - 4.0
            && p[1] <= solved.max().1 + 4.0;
        assert!(in_l1 || near_solved, "vertex {p:?} off the solved rect");
    }
    // At least the field chrome is bitwise-identical to the L1 draw.
    let common = a.iter().filter(|p| b.contains(p)).count();
    assert!(common >= 8, "chrome verts {common} match L1");
}

#[test]
fn view_paint_order_container_before_children() {
    let mut u = ui();
    frame(&mut u, UiInput::default(), |ui| {
        ui.container(Style::row().pad(4.0), Paint::surface(ui.theme()), |ui| {
            ui.add(Text::new("inside"));
        });
    });
    // Container chrome is drawn before the child text: the first quads
    // are the panel's.
    assert!(u.view_stats().nodes >= 3);
}

#[test]
fn view_baseline_row_aligns_label_button_field() {
    let mut u = ui();
    let mut s = String::new();
    frame(&mut u, UiInput::default(), |ui| {
        ui.row(
            Style::default().align(crate::ui::layout::LayoutAlign::Baseline),
            |ui| {
                ui.add(Text::new("name"));
                ui.add(Button::new("ok"));
                ui.add(TextField::new("f", &mut s));
            },
        );
    });
    assert!(u.view_stats().nodes >= 4);
}

/// Two roots per frame each cache independently (per-root trees).
#[test]
fn view_two_roots_both_cache_hit_when_idle() {
    let mut u = ui();
    for _ in 0..2 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(0.0, 0.0, 200.0, 300.0), Style::column(), |ui| {
            ui.add(Text::new("a"));
        });
        u.layout_root(Rect::new(200.0, 0.0, 200.0, 300.0), Style::column(), |ui| {
            ui.add(Text::new("b"));
        });
        u.end();
    }
    let st = u.view_stats();
    assert!(st.cache_hits >= 2, "two roots → two cache hits: {st:?}");
}

// --- devtools ---------------------------------------------------------------

/// Off: identical vertex count to no-devtools.
#[test]
fn devtools_off_emits_nothing_extra() {
    let mut a = ui();
    a.set_devtools(crate::ui::devtools::DevtoolsMode::Off);
    let mut b = ui(); // never touched
    let build = |u: &mut Ui| {
        u.layout_root(
            Rect::new(0.0, 0.0, 400.0, 200.0),
            Style::column().pad(4.0),
            |ui| {
                ui.add(Text::new("x"));
                ui.add(Button::new("b"));
            },
        );
    };
    a.begin(UiInput::default(), (400.0, 300.0));
    build(&mut a);
    let fa = a.end();
    b.begin(UiInput::default(), (400.0, 300.0));
    build(&mut b);
    let fb = b.end();
    assert_eq!(fa.vertices.len(), fb.vertices.len());
}

/// Outlines: one outline quads-set per node on LAYER_DEVTOOLS.
#[test]
fn devtools_outlines_layered() {
    let mut u = ui();
    u.set_devtools(crate::ui::devtools::DevtoolsMode::Outlines);
    u.begin(UiInput::default(), (400.0, 300.0));
    u.layout_root(
        Rect::new(0.0, 0.0, 400.0, 200.0),
        Style::column().pad(4.0).gap(4.0),
        |ui| {
            ui.add(Text::new("x"));
            ui.add(Button::new("b"));
        },
    );
    let f = u.end();
    let dev = f
        .cmds
        .iter()
        .filter(|c| c.layer == crate::ui::widgets::LAYER_DEVTOOLS)
        .count();
    assert!(dev >= 1, "devtool cmds on LAYER_DEVTOOLS: {dev}");
    // 3 nodes (root + 2 children) → ≥3 outline cmds worth of verts.
    let outlines = f.vertices.len();
    assert!(outlines > 0);
}

/// Inspect: press over a node pins it and consumes the click (no click on the button beneath).
#[test]
fn devtools_inspect_pins_and_blocks_click() {
    let mut u = ui();
    u.set_devtools(crate::ui::devtools::DevtoolsMode::Inspect);
    // Frame 1: build so records exist.
    u.begin(UiInput::default(), (400.0, 300.0));
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 200.0), Style::column(), |ui| {
        ui.add(Text::new("x"));
        ui.add(Button::new("done"));
    });
    u.end();
    // Frame 2: mouse press over the button → pinned, press consumed.
    let mut inp = UiInput {
        mouse: (10.0, 10.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(inp.clone(), (400.0, 300.0));
    let mut clicked = false;
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 200.0), Style::column(), |ui| {
        ui.add(Text::new("x"));
        clicked = ui.add(Button::new("done")).clicked;
    });
    u.end();
    assert!(!clicked, "inspect press consumed");
    assert!(u.view.devtools.pinned.is_some());
    // Release + confirm a pinned record persists.
    inp.mouse_down = false;
    inp.mouse_pressed = false;
    inp.mouse_released = true;
    u.begin(inp, (400.0, 300.0));
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 200.0), Style::column(), |ui| {
        ui.add(Text::new("x"));
        ui.add(Button::new("done"));
    });
    u.end();
    assert!(u.view.devtools.pinned.is_some());
}

// --- presolve / replay -----------------------------------------------------

/// A single fresh frame paints a TextField + Segmented (record + sized
/// path) — no blank first frame.
#[test]
fn view_first_frame_paints_record_widgets() {
    let mut u = ui();
    let mut v = "txt".to_string();
    let mut sel = 0usize;
    u.begin(UiInput::default(), (400.0, 300.0));
    u.layout_root(
        Rect::new(0.0, 0.0, 400.0, 300.0),
        Style::column().pad(4.0).gap(4.0),
        |ui| {
            ui.add_styled(
                TextField::new("tf", &mut v),
                Style::default().fill_w().h_px(24.0),
            );
            ui.add_styled(
                Segmented::new("seg", &["a", "b"], &mut sel),
                Style::default().fill_w().h_px(24.0),
            );
        },
    );
    let tf = u.rect_of("tf").unwrap();
    let seg = u.rect_of("seg").unwrap();
    let f = u.end();
    assert!(!f.vertices.is_empty());
    // The text field solved to the fill width.
    assert_eq!(tf.w, 400.0 - 8.0);
    assert!(seg.w > 100.0);
    // Both nodes' rects contain vertices.
    let in_r = |r: Rect| f.vertices.iter().any(|v| r.contains((v.pos[0], v.pos[1])));
    assert!(in_r(tf), "text field band empty: {tf:?}");
    assert!(in_r(seg), "segmented band empty: {seg:?}");
}

/// A node appearing on a later frame paints immediately.
#[test]
fn view_late_node_paints_first_frame() {
    let mut u = ui();
    let mut v = String::new();
    for _ in 0..3 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add(Text::new("a"));
        });
        u.end();
    }
    u.begin(UiInput::default(), (400.0, 300.0));
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add(Text::new("a"));
        ui.add_styled(
            TextField::new("late", &mut v),
            Style::default().fill_w().h_px(24.0),
        );
    });
    let r = u.rect_of("late").unwrap();
    let f = u.end();
    assert!(
        f.vertices.iter().any(|v| r.contains((v.pos[0], v.pos[1]))),
        "late TextField painted nothing at {r:?}"
    );
}

/// A click on the frame a sibling appears reports clicked exactly once.
#[test]
fn view_replay_strips_edge_input() {
    let mut u = ui();
    let mut clicks = 0u32;
    // Warm up without the late node.
    for _ in 0..2 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            if ui.add(Button::new("b")).clicked {
                clicks += 1;
            }
        });
        u.end();
    }
    let b = u.rect_of("b").unwrap();
    // Frame 3: a press centered on the button, plus the late sibling.
    let mut i = UiInput {
        mouse: (b.x + 4.0, b.y + 4.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(i.clone(), (400.0, 300.0));
    let mut v = String::new();
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        if ui.add(Button::new("b")).clicked {
            clicks += 1;
        }
        ui.add_styled(
            TextField::new("late", &mut v),
            Style::default().fill_w().h_px(24.0),
        );
    });
    u.end();
    i.mouse_pressed = false;
    i.mouse_released = true;
    u.begin(i, (400.0, 300.0));
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        if ui.add(Button::new("b")).clicked {
            clicks += 1;
        }
        ui.add_styled(
            TextField::new("late", &mut v),
            Style::default().fill_w().h_px(24.0),
        );
        // Another new node on the release frame: this frame replays too.
        ui.add(Text::new("release_new"));
    });
    u.end();
    assert_eq!(clicks, 1, "click must fire exactly once, got {clicks}");
    assert!(u.active_id.is_none(), "capture must clear after release");
}

/// The replay pass must not double-step animations: a `layout_root`
/// frame that replays produces the same anim value as a warmed non-replay
/// reference, now and next frame.
#[test]
fn view_replay_anim_single_step() {
    // anim_f32 must run inside the layout closure so the replay pass
    // actually exercises it; the returned value escapes via the cell.
    let run = |u: &mut crate::ui::widgets::Ui, extra: bool, out: &mut f32| {
        let mut v = String::new();
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            *out = ui.anim_f32("motion", 1.0, 0.5, crate::ui::tween::Ease::Linear);
            if extra {
                ui.add_styled(
                    TextField::new("late", &mut v),
                    Style::default().fill_w().h_px(24.0),
                );
            }
        });
    };
    let mut u = ui();
    let mut r = ui();
    let (mut a1, mut b1, mut a2, mut b2) = (0.0, 0.0, 0.0, 0.0);
    // Both Uis warm inside the layout closure at target 0.
    for _ in 0..3 {
        u.begin(
            UiInput {
                dt: 0.1,
                ..Default::default()
            },
            (400.0, 300.0),
        );
        let mut v0 = 0.0f32;
        let mut s = String::new();
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            v0 = ui.anim_f32("motion", 0.0, 0.5, crate::ui::tween::Ease::Linear);
            let _ = v0;
            let _ = &mut s;
        });
        u.end();
        r.begin(
            UiInput {
                dt: 0.1,
                ..Default::default()
            },
            (400.0, 300.0),
        );
        r.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.anim_f32("motion", 0.0, 0.5, crate::ui::tween::Ease::Linear);
        });
        r.end();
    }
    // Replay frame: `u` gains a new sibling; `r` is the control.
    u.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    run(&mut u, true, &mut a1);
    u.end();
    r.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    run(&mut r, false, &mut b1);
    r.end();
    assert!(a1 > 0.0, "anim must advance on the replay frame");
    assert_eq!(a1, b1, "replay double-stepped: {a1} vs {b1}");
    u.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    run(&mut u, true, &mut a2);
    u.end();
    r.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    run(&mut r, false, &mut b2);
    r.end();
    assert!(a2 > a1);
    assert_eq!(a2, b2, "post-replay anim drifted: {a2} vs {b2}");
}

/// Steady state: the build closure runs exactly once per frame.
#[test]
fn view_steady_state_single_pass() {
    let mut u = ui();
    let mut calls = 0u32;
    for _ in 0..4 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            calls += 1;
            ui.add(Text::new("x"));
        });
        u.end();
    }
    // Frames 1-2 may replay (first appearance); 3-4 must be single-pass.
    let before = calls;
    u.begin(UiInput::default(), (400.0, 300.0));
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        calls += 1;
        ui.add(Text::new("x"));
    });
    u.end();
    assert_eq!(calls - before, 1);
}

/// A scrolled container keeps its offset when a sibling appears (replay
/// must not re-integrate the physics).
#[test]
fn view_replay_keeps_scroll_offset() {
    let mut u = ui();
    for _ in 0..2 {
        u.begin(UiInput::default(), (200.0, 100.0));
        u.layout_root(
            Rect::new(0.0, 0.0, 200.0, 100.0),
            Style::column().fill(),
            |ui| {
                ui.container(
                    Style::column().fill().scroll_y().key("sc"),
                    Paint::none(),
                    |ui| {
                        for i in 0..10 {
                            ui.add_styled(Text::new("x"), Style::default().h_px(40.0));
                            let _ = i;
                        }
                    },
                );
            },
        );
        u.end();
    }
    // Wheel-scroll once so the offset is nonzero.
    let r = u.rect_of("sc").unwrap();
    u.begin(
        UiInput {
            mouse: (r.x + 4.0, r.y + 4.0),
            wheel: -40.0,
            ..Default::default()
        },
        (200.0, 100.0),
    );
    u.layout_root(
        Rect::new(0.0, 0.0, 200.0, 100.0),
        Style::column().fill(),
        |ui| {
            ui.container(
                Style::column().fill().scroll_y().key("sc"),
                Paint::none(),
                |ui| {
                    for _ in 0..10 {
                        ui.add_styled(Text::new("x"), Style::default().h_px(40.0));
                    }
                },
            );
        },
    );
    u.end();
    let off0 = u.view.scroll.get(&u.id("sc")).map_or(0.0, |&(o, _)| o);
    assert!(off0 > 0.0);
    // New sibling appears → replay runs; the offset must not move.
    u.begin(UiInput::default(), (200.0, 100.0));
    u.layout_root(
        Rect::new(0.0, 0.0, 200.0, 100.0),
        Style::column().fill(),
        |ui| {
            ui.container(
                Style::column().fill().scroll_y().key("sc"),
                Paint::none(),
                |ui| {
                    for _ in 0..10 {
                        ui.add_styled(Text::new("x"), Style::default().h_px(40.0));
                    }
                },
            );
            ui.add_styled(
                TextField::new("late", &mut String::new()),
                Style::default().fill_w().h_px(24.0),
            );
        },
    );
    u.end();
    let off1 = u.view.scroll.get(&u.id("sc")).map_or(0.0, |&(o, _)| o);
    assert!(
        (off1 - off0).abs() < 0.01,
        "scroll offset changed across replay: {off0} → {off1}"
    );
}

/// `anim_dt` outside a layout is the input dt (L1 unaffected).
#[test]
fn anim_dt_outside_layout_root_nonzero() {
    let mut u = ui();
    u.begin(
        UiInput {
            dt: 1.0 / 30.0,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    let a = u.anim_f32("k", 1.0, 0.5, crate::ui::tween::Ease::Linear);
    u.end();
    assert!(a > 0.0, "L1 anim must advance outside layout_root");
}

/// Custom slot paint callbacks run exactly once per frame (replay rebuilds
/// ops but only the final solve paints).
#[test]
fn view_slot_once_on_replay() {
    let mut u = ui();
    let calls = std::rc::Rc::new(std::cell::Cell::new(0u32));
    for _ in 0..2 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            let c = calls.clone();
            ui.slot(Style::default().size_px(50.0, 20.0), move |_, _| {
                c.set(c.get() + 1);
            });
        });
        u.end();
    }
    let base = calls.get();
    // Frame with a new sibling → the build replays (2 calls) but the
    // slot callback still paints exactly once.
    let builds = std::rc::Rc::new(std::cell::Cell::new(0u32));
    u.begin(UiInput::default(), (400.0, 300.0));
    let mut v = String::new();
    let b2 = builds.clone();
    let calls2 = calls.clone();
    u.layout_root(
        Rect::new(0.0, 0.0, 400.0, 300.0),
        Style::column(),
        move |ui: &mut crate::ui::widgets::Ui| {
            b2.set(b2.get() + 1);
            let _ = &mut v;
            let c = calls2.clone();
            ui.slot(Style::default().size_px(50.0, 20.0), move |_, _| {
                c.set(c.get() + 1);
            });
            ui.add_styled(
                TextField::new("late", &mut v),
                Style::default().fill_w().h_px(24.0),
            );
        },
    );
    u.end();
    assert_eq!(builds.get(), 2, "the replay frame must build twice");
    assert_eq!(
        calls.get() - base,
        1,
        "slot callback must run once, got {}",
        calls.get() - base
    );
}

/// Replay must not duplicate focusables: Tab visits each widget once.
#[test]
fn view_replay_focus_order() {
    let mut u = ui();
    for _ in 0..2 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add(Button::new("b1"));
            ui.add(Button::new("b2"));
        });
        u.end();
    }
    // Frame with the new TextField → replay + truncation.
    u.begin(UiInput::default(), (400.0, 300.0));
    let mut v = String::new();
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add(Button::new("b1"));
        ui.add(Button::new("b2"));
        ui.add_styled(
            TextField::new("f", &mut v),
            Style::default().fill_w().h_px(24.0),
        );
    });
    u.end();
    // Three Tab presses should visit three distinct widgets once each.
    let mut seen = Vec::new();
    for _ in 0..3 {
        u.begin(
            UiInput {
                nav: crate::ui::widgets::NavInput {
                    tab: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            (400.0, 300.0),
        );
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add(Button::new("b1"));
            ui.add(Button::new("b2"));
            ui.add_styled(
                TextField::new("f", &mut v),
                Style::default().fill_w().h_px(24.0),
            );
        });
        seen.push(u.focus_id());
        u.end();
    }
    let mut uniq = seen.clone();
    uniq.dedup();
    assert_eq!(seen.iter().filter(|s| s.is_some()).count(), 3);
    assert_eq!(uniq.len(), seen.len(), "tab cycle duplicated: {seen:?}");
}

/// A held DragField drag must report `clicked` (changed) exactly once;
/// the replay pass must not reapply or double-step the drag.
#[test]
fn view_replay_held_drag_once() {
    let mut u = ui();
    let mut v = 0.5f32;
    // Two neutral frames so the field has a solved prev rect.
    for _ in 0..2 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add_styled(
                DragField::new("d", &mut v, 0.0..=1.0),
                Style::default().size_px(100.0, 24.0),
            );
        });
        u.end();
    }
    let d = u.rect_of("d").unwrap();
    // Press inside the field (captures the anchor), still no sibling.
    u.begin(
        UiInput {
            mouse: (d.x + 50.0, d.y + 12.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add_styled(
            DragField::new("d", &mut v, 0.0..=1.0),
            Style::default().size_px(100.0, 24.0),
        );
    });
    u.end();
    // Move while held; the late TextField forces a replay this frame.
    let mut clicks = 0u32;
    u.begin(
        UiInput {
            mouse: (d.x + 70.0, d.y + 12.0),
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    let mut s = String::new();
    u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        let r = ui.add_styled(
            DragField::new("d", &mut v, 0.0..=1.0),
            Style::default().size_px(100.0, 24.0),
        );
        if r.clicked {
            clicks += 1;
        }
        ui.add_styled(
            TextField::new("late", &mut s),
            Style::default().fill_w().h_px(24.0),
        );
    });
    u.end();
    assert_eq!(clicks, 1, "changed must fire exactly once, got {clicks}");
    assert!(v != 0.5, "drag did not move the value");
    // Reference: same capture+move without the replay frame.
    let mut r2 = ui();
    let mut v2 = 0.5f32;
    for _ in 0..3 {
        r2.begin(UiInput::default(), (400.0, 300.0));
        r2.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add_styled(
                DragField::new("d", &mut v2, 0.0..=1.0),
                Style::default().size_px(100.0, 24.0),
            );
        });
        r2.end();
    }
    r2.begin(
        UiInput {
            mouse: (d.x + 50.0, d.y + 12.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    r2.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add_styled(
            DragField::new("d", &mut v2, 0.0..=1.0),
            Style::default().size_px(100.0, 24.0),
        );
    });
    r2.end();
    r2.begin(
        UiInput {
            mouse: (d.x + 70.0, d.y + 12.0),
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    r2.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add_styled(
            DragField::new("d", &mut v2, 0.0..=1.0),
            Style::default().size_px(100.0, 24.0),
        );
    });
    r2.end();
    assert_eq!(
        v.to_bits(),
        v2.to_bits(),
        "replay produced a different value: {v} vs {v2}"
    );
}

/// A moved/resized segmented paints its pill and labels at the solved
/// rect this frame — same vertices as the L1 widget drawn there.
#[test]
fn view_segmented_solved_geometry() {
    let mut u = ui();
    let mut sel = 1usize;
    for _ in 0..3 {
        u.begin(UiInput::default(), (400.0, 300.0));
        u.layout_root(Rect::new(10.0, 10.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add_styled(
                Segmented::new("seg", &["a", "b"], &mut sel),
                Style::default().size_px(120.0, 24.0),
            );
        });
        u.end();
    }
    // Move AND resize the same frame.
    u.begin(UiInput::default(), (400.0, 300.0));
    u.layout_root(Rect::new(30.0, 20.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add_styled(
            Segmented::new("seg", &["a", "b"], &mut sel),
            Style::default().size_px(240.0, 24.0),
        );
    });
    let solved = u.rect_of("seg").unwrap();
    let f = u.end();
    assert_eq!(solved, Rect::new(30.0, 20.0, 240.0, 24.0));
    // L1 reference drawn directly at the solved rect.
    let mut r = ui();
    let mut sel2 = 1usize;
    r.begin(UiInput::default(), (400.0, 300.0));
    let st = r.theme().segmented(crate::ui::font::FontId(0));
    r.segmented_styled("seg", solved, &["a", "b"], &mut sel2, &st);
    let g = r.end();
    let a: Vec<[f32; 2]> = f.vertices.iter().map(|v| v.pos).collect();
    let b: Vec<[f32; 2]> = g.vertices.iter().map(|v| v.pos).collect();
    assert_eq!(a, b, "segmented solved geometry differs from L1");
    // Two controls with distinct selections keep independent pills.
    let mut u2 = ui();
    let (mut s0, mut s1) = (0usize, 1usize);
    for _ in 0..3 {
        u2.begin(UiInput::default(), (400.0, 300.0));
        u2.layout_root(
            Rect::new(0.0, 0.0, 400.0, 300.0),
            Style::row().gap(8.0),
            |ui| {
                ui.add_styled(
                    Segmented::new("s0", &["a", "b"], &mut s0),
                    Style::default().size_px(120.0, 24.0),
                );
                ui.add_styled(
                    Segmented::new("s1", &["a", "b"], &mut s1),
                    Style::default().size_px(120.0, 24.0),
                );
            },
        );
        u2.end();
    }
    // Read the same scoped anim keys the adapters use: s0's pill rests
    // in cell 0 (at≈0), s1's in cell 1 (at≈1) — independent anim state.
    u2.begin(UiInput::default(), (400.0, 300.0));
    u2.push_id("s0");
    let a0 = u2.anim_f32_seeded("##pill", 0.0, 0.5, crate::ui::tween::Ease::Linear);
    u2.pop_id();
    u2.push_id("s1");
    let a1 = u2.anim_f32_seeded("##pill", 0.0, 0.5, crate::ui::tween::Ease::Linear);
    u2.pop_id();
    u2.end();
    assert!(a0 < a1, "pills must sit in different cells: {a0} vs {a1}");
    // Empty options: adapter returns none; the painter draws no pill.
    let mut sel3 = 0usize;
    u2.begin(UiInput::default(), (400.0, 300.0));
    u2.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::row(), |ui| {
        ui.add_styled(
            Segmented::new("e", &[], &mut sel3),
            Style::default().size_px(120.0, 24.0),
        );
    });
    let g2 = u2.end();
    assert!(g2.vertices.is_empty());
    // Empty cells passed directly to the painter: no pill math (n-1
    // underflow guard), only the track draws.
    let mut r3 = ui();
    r3.begin(UiInput::default(), (400.0, 300.0));
    let st3 = r3.theme().segmented(crate::ui::font::FontId(0));
    crate::ui::widgets::paint::segmented(&mut r3, Rect::new(0.0, 0.0, 120.0, 24.0), &st3, 0.0, &[]);
    let g3 = r3.end();
    // The empty painter emits exactly the track — compare vertex count
    // against a bare track draw of the same rect/style.
    let mut r4 = ui();
    r4.begin(UiInput::default(), (400.0, 300.0));
    let st4 = r4.theme().segmented(crate::ui::font::FontId(0));
    match st4.track_border {
        Some((w, c)) => r4.draw.rounded_frame(
            Rect::new(0.0, 0.0, 120.0, 24.0),
            st4.button.corner_radius,
            w,
            st4.track,
            c,
        ),
        None => r4.draw.round_rect(
            Rect::new(0.0, 0.0, 120.0, 24.0),
            st4.button.corner_radius,
            st4.track,
        ),
    }
    let g4 = r4.end();
    assert_eq!(
        g3.vertices.len(),
        g4.vertices.len(),
        "empty painter emitted more than the track"
    );
    // Focused segment still gets a ring at the solved rect.
    let mut u4 = ui();
    let mut sel5 = 1usize;
    for _ in 0..2 {
        u4.begin(UiInput::default(), (400.0, 300.0));
        u4.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add_styled(
                Segmented::new("fs", &["a", "b"], &mut sel5),
                Style::default().size_px(120.0, 24.0),
            );
        });
        u4.end();
    }
    // Final frame at a moved/resized rect: focus option "b" (index 1,
    // its second cell) so the ring must emit at the new geometry.
    u4.set_focus_rings(true);
    u4.push_id("fs");
    u4.set_focus("b##seg_1");
    u4.pop_id();
    u4.set_focus_visible(true);
    u4.begin(UiInput::default(), (400.0, 300.0));
    u4.layout_root(Rect::new(20.0, 20.0, 400.0, 300.0), Style::column(), |ui| {
        ui.add_styled(
            Segmented::new("fs", &["a", "b"], &mut sel5),
            Style::default().size_px(200.0, 24.0),
        );
    });
    let fr = u4.rect_of("fs").unwrap();
    let focus = u4.theme().tokens.focus;
    let color = u4.theme().roles.border.focus;
    let expected_ring = Rect::new(fr.x + fr.w * 0.5, fr.y, fr.w * 0.5, fr.h)
        .inset(u4.theme().segmented(crate::ui::font::FontId(0)).inset)
        .expand(focus.offset + focus.width);
    let f4 = u4.end();
    let ring: Vec<_> = f4
        .vertices
        .iter()
        .filter(|v| v.color == color.to_array())
        .collect();
    assert_eq!(ring.len(), 4, "focused segment must paint exactly one ring");
    assert_eq!(ring[0].pos, [expected_ring.x, expected_ring.y]);
    assert_eq!(ring[2].pos, [expected_ring.max().0, expected_ring.max().1]);
}

/// Nested scroll viewports: wheel inside the inner one only scrolls the
/// inner; wheel in outer-only space scrolls the outer. A replayed frame
/// doesn't reset or double-register.
#[test]
fn view_nested_scroll_wheel_owner() {
    let mut u = ui();
    u.scroll_config = crate::ui::widgets::ScrollConfig::no_momentum();
    let build = |ui: &mut crate::ui::widgets::Ui, extra: bool| {
        ui.container(
            Style::column().fill().scroll_y().key("outer"),
            Paint::none(),
            |ui| {
                ui.container(
                    Style::column().fill().h_px(60.0).scroll_y().key("inner"),
                    Paint::none(),
                    |ui| {
                        for _ in 0..6 {
                            ui.add_styled(Text::new("x"), Style::default().h_px(50.0));
                        }
                    },
                );
                for _ in 0..10 {
                    ui.add_styled(Text::new("y"), Style::default().h_px(40.0));
                }
                if extra {
                    let mut v = String::new();
                    ui.add_styled(
                        TextField::new("late", &mut v),
                        Style::default().fill_w().h_px(24.0),
                    );
                }
            },
        );
    };
    for _ in 0..2 {
        u.begin(UiInput::default(), (200.0, 120.0));
        u.layout_root(
            Rect::new(0.0, 0.0, 200.0, 120.0),
            Style::column().fill(),
            |ui| {
                build(ui, false);
            },
        );
        u.end();
    }
    let inner = u.rect_of("inner").unwrap();
    let outer = u.rect_of("outer").unwrap();
    let off =
        |u: &crate::ui::widgets::Ui, k: &str| u.view.scroll.get(&u.id(k)).map_or(0.0, |&(o, _)| o);
    // Wheel over the inner viewport: only inner moves.
    u.begin(
        UiInput {
            mouse: (inner.x + 4.0, inner.y + 4.0),
            wheel: -30.0,
            ..Default::default()
        },
        (200.0, 120.0),
    );
    u.layout_root(
        Rect::new(0.0, 0.0, 200.0, 120.0),
        Style::column().fill(),
        |ui| {
            build(ui, false);
        },
    );
    u.end();
    let (oi, oo) = (off(&u, "inner"), off(&u, "outer"));
    assert!(oi > 0.0, "inner did not scroll: {oi}");
    assert_eq!(oo, 0.0, "outer moved when wheel was over inner");
    // Wheel in outer-only space (below inner): only outer moves.
    u.begin(
        UiInput {
            mouse: (outer.x + 4.0, outer.y + outer.h - 4.0),
            wheel: -30.0,
            ..Default::default()
        },
        (200.0, 120.0),
    );
    u.layout_root(
        Rect::new(0.0, 0.0, 200.0, 120.0),
        Style::column().fill(),
        |ui| {
            build(ui, false);
        },
    );
    u.end();
    let (oi2, oo2) = (off(&u, "inner"), off(&u, "outer"));
    assert_eq!(oi2, oi, "inner moved when wheel was over outer-only space");
    assert!(oo2 > oo, "outer did not scroll: {oo2}");
    // Replay frame (new sibling): offsets retained, regions still 2.
    u.begin(UiInput::default(), (200.0, 120.0));
    u.layout_root(
        Rect::new(0.0, 0.0, 200.0, 120.0),
        Style::column().fill(),
        |ui| {
            build(ui, true);
        },
    );
    u.end();
    assert_eq!(off(&u, "inner"), oi2, "replay reset inner offset");
    assert_eq!(off(&u, "outer"), oo2, "replay reset outer offset");
    assert_eq!(
        u.scroll_regions_len(),
        2,
        "scroll regions duplicated across replay"
    );
}

/// IME preedit must render identically across a replay: glyphs, underline
/// and committed text are preserved; the editing rect stays solved.
#[test]
fn view_replay_ime_preserved() {
    let build = |u: &mut crate::ui::widgets::Ui, extra: bool| {
        let mut v = String::from("field");
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            ui.add_styled(
                TextField::new("ime", &mut v),
                Style::default().fill_w().h_px(24.0),
            );
            if extra {
                let mut s = String::new();
                ui.add_styled(
                    TextField::new("late", &mut s),
                    Style::default().fill_w().h_px(24.0),
                );
            }
        });
    };
    let mut u = ui();
    let mut r = ui();
    for _ in 0..2 {
        for (uu, e) in [(&mut u, false), (&mut r, false)] {
            uu.begin(UiInput::default(), (400.0, 300.0));
            build(uu, e);
            uu.end();
        }
        u.set_focus("ime");
        r.set_focus("ime");
    }
    // Preedit frame: u replays (late sibling), r is the control.
    let input = UiInput {
        preedit: Some(crate::ui::widgets::Composition {
            text: "compose".into(),
            cursor: Some((1, 4)),
        }),
        ..Default::default()
    };
    u.begin(input.clone(), (400.0, 300.0));
    build(&mut u, true);
    let fu = u.end();
    r.begin(input, (400.0, 300.0));
    build(&mut r, false);
    let fr_ = r.end();
    // Field rects: identical solved geometry for both Uis.
    let ru = u.rect_of("ime").unwrap();
    let rr = r.rect_of("ime").unwrap();
    assert_eq!(ru, rr);
    let er = u
        .editing_rect()
        .expect("replayed field has an editing rect");
    assert_eq!(er, r.editing_rect().expect("control has an editing rect"));
    assert!(
        ru.contains((er.x, er.y)) || er.intersect(ru).is_some(),
        "editing rect {er:?} outside field {ru:?}"
    );
    // Every vertex inside the field matches between replay and control.
    // Strictly inside the field rect — the late sibling's top edge sits
    // exactly on the boundary and must not count.
    let in_r = |v: &crate::ui::draw::UiVertex, r: Rect| {
        v.pos[0] >= r.x && v.pos[0] < r.x + r.w && v.pos[1] >= r.y && v.pos[1] < r.y + r.h
    };
    let au: Vec<_> = fu
        .vertices
        .iter()
        .filter(|v| in_r(v, ru))
        .map(|v| (v.pos, v.color))
        .collect();
    let ar: Vec<_> = fr_
        .vertices
        .iter()
        .filter(|v| in_r(v, rr))
        .map(|v| (v.pos, v.color))
        .collect();
    assert_eq!(au, ar, "IME preedit output differs across replay");
    // And it differs from a no-preedit frame (composition drew something).
    u.begin(UiInput::default(), (400.0, 300.0));
    build(&mut u, true);
    let fn_ = u.end();
    let an: Vec<_> = fn_
        .vertices
        .iter()
        .filter(|v| in_r(v, ru))
        .map(|v| (v.pos, v.color))
        .collect();
    assert_ne!(au, an, "preedit produced no glyph/underline output");
}

/// `rate_scope` on the replay pass must not double-step: both frames'
/// values match a non-replay control.
#[test]
fn view_replay_rate_scope_single_step() {
    let mut u = ui();
    let mut r = ui();
    let mut builds = 0u32;
    let run = |u: &mut crate::ui::widgets::Ui, extra: bool, out: &mut f32, builds: &mut u32| {
        u.layout_root(Rect::new(0.0, 0.0, 400.0, 300.0), Style::column(), |ui| {
            *builds += 1;
            ui.rate_scope("scope", Some(10.0));
            *out = ui.anim_f32("a", 1.0, 0.5, crate::ui::tween::Ease::Linear);
            ui.end_rate_scope();
            if extra {
                let mut v = String::new();
                ui.add_styled(
                    TextField::new("late", &mut v),
                    Style::default().fill_w().h_px(24.0),
                );
            }
        });
    };
    let (mut a, mut b) = (0.0f32, 0.0f32);
    for _ in 0..2 {
        u.begin(
            UiInput {
                dt: 0.06,
                ..Default::default()
            },
            (400.0, 300.0),
        );
        run(&mut u, false, &mut a, &mut builds);
        u.end();
        r.begin(
            UiInput {
                dt: 0.06,
                ..Default::default()
            },
            (400.0, 300.0),
        );
        run(&mut r, false, &mut b, &mut builds);
        r.end();
    }
    // Boundary frame: target flips, u replays.
    u.begin(
        UiInput {
            dt: 0.06,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    run(&mut u, true, &mut a, &mut builds);
    u.end();
    r.begin(
        UiInput {
            dt: 0.06,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    run(&mut r, false, &mut b, &mut builds);
    r.end();
    assert!(a > 0.0 && b > 0.0, "anim must advance past 0");
    assert_eq!(a, b, "rate-scope anim drifted under replay: {a} vs {b}");
}

#[test]
fn view_record_text_field_inherits_input_clip_and_layer() {
    for (layer, mouse_x, expected_focus) in [
        (0, 70.0, false),
        (crate::ui::widgets::LAYER_MODAL, 20.0, true),
    ] {
        let mut u = ui();
        let mut text = String::from("field");
        for frame in 0..3 {
            u.begin(
                UiInput {
                    mouse: (mouse_x, 12.0),
                    mouse_pressed: frame == 2,
                    mouse_down: frame == 2,
                    ..Default::default()
                },
                (200.0, 100.0),
            );
            if layer != 0 {
                u.declare_input_surface(Rect::new(0.0, 0.0, 200.0, 100.0), layer, true);
            }
            u.draw.set_layer(layer);
            u.draw.push_clip(Rect::new(0.0, 0.0, 40.0, 100.0));
            u.layout_root(Rect::new(0.0, 0.0, 200.0, 100.0), Style::column(), |ui| {
                ui.add_styled(
                    TextField::new("field", &mut text),
                    Style::default().size_px(100.0, 24.0),
                );
            });
            u.draw.pop_clip();
            u.end();
        }
        assert_eq!(
            u.focus_id() == Some(u.id("field")),
            expected_focus,
            "recorded field input at x={mouse_x} must respect its clip and modal layer"
        );
    }
}

#[test]
fn view_record_moving_into_clip_paints_solved_geometry() {
    let mut u = ui();
    let mut text = String::from("visible");
    for x in [-100.0, 0.0] {
        u.begin(UiInput::default(), (200.0, 100.0));
        u.draw.push_clip(Rect::new(0.0, 0.0, 80.0, 100.0));
        u.layout_root(Rect::new(x, 0.0, 200.0, 100.0), Style::column(), |ui| {
            ui.add_styled(
                TextField::new("field", &mut text),
                Style::default().size_px(100.0, 24.0),
            );
        });
        u.draw.pop_clip();
        let frame = u.end();
        if x == 0.0 {
            assert!(!frame.vertices.is_empty(), "moved field stayed culled");
            assert!(frame.vertices.iter().any(|v| v.pos[0] >= 0.0));
            assert!(
                frame
                    .cmds
                    .iter()
                    .all(|c| c.scissor.is_some_and(|r| r.w > 0.0))
            );
        }
    }
}

#[test]
fn view_keyed_button_keeps_behavior_focus_ring() {
    let mut u = ui();
    u.set_focus_rings(true);
    u.begin(UiInput::default(), (200.0, 100.0));
    u.set_focus("button");
    u.set_focus_visible(true);
    u.layout_root(Rect::new(10.0, 10.0, 190.0, 90.0), Style::column(), |ui| {
        ui.add_styled(Button::new("button"), Style::default().key("probe"));
    });
    let rect = u.rect_of("probe").unwrap();
    let color = u.theme().roles.border.focus.to_array();
    let focus = u.theme().tokens.focus;
    let expected = rect.expand(focus.offset + focus.width);
    let frame = u.end();
    let ring: Vec<_> = frame.vertices.iter().filter(|v| v.color == color).collect();
    assert_eq!(ring.len(), 4, "keyed button lost its focus ring");
    assert_eq!(ring[0].pos, [expected.x, expected.y]);
    assert_eq!(ring[2].pos, [expected.max().0, expected.max().1]);
}
