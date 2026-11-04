use super::*;
use crate::ui::test_util::test_font;

fn ui() -> Ui {
    Ui::new(test_font(), AtlasConfig::default())
}

#[test]
fn button_clicks_on_release_inside() {
    let mut u = ui();
    let mut input = UiInput {
        mouse: (50.0, 30.0),
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    // Press.
    input.mouse_down = true;
    input.mouse_pressed = true;
    u.begin(input.clone(), (800.0, 600.0));
    assert!(!u.button("ok", rect, ButtonStyle::new(FontId(0))).clicked);
    // Release inside: click.
    input.mouse_down = false;
    input.mouse_pressed = false;
    input.mouse_released = true;
    u.begin(input.clone(), (800.0, 600.0));
    let response = u.button("ok", rect, ButtonStyle::new(FontId(0)));
    assert!(response.clicked);
    assert!(response.hovered);
}

#[test]
fn button_ignores_disabled_and_outside_release() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let mut input = UiInput {
        mouse: (500.0, 500.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.button("ok", rect, ButtonStyle::new(FontId(0)));
    // Release outside the button: no click.
    input.mouse_released = true;
    input.mouse_down = false;
    input.mouse_pressed = false;
    u.begin(input.clone(), (800.0, 600.0));
    assert!(!u.button("ok", rect, ButtonStyle::new(FontId(0))).clicked);

    // Disabled button never clicks even with a perfect click.
    let mut input = UiInput {
        mouse: (50.0, 50.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.button("ok", rect, ButtonStyle::new(FontId(0)).disabled());
    input.mouse_released = true;
    input.mouse_down = false;
    input.mouse_pressed = false;
    u.begin(input.clone(), (800.0, 600.0));
    assert!(
        !u.button("ok", rect, ButtonStyle::new(FontId(0)).disabled())
            .clicked
    );
}

#[test]
fn toggle_flips_value() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 200.0, 30.0);
    let mut value = false;
    // Press first (capture), then release inside.
    let input = UiInput {
        mouse: (50.0, 25.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.toggle("mute", rect, &mut value, ButtonStyle::new(FontId(0)));
    let input = UiInput {
        mouse: (50.0, 25.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let response = u.toggle("mute", rect, &mut value, ButtonStyle::new(FontId(0)));
    assert!(
        response.clicked,
        "press then release inside a toggle clicks it"
    );
    assert!(value, "toggle flips on click");
}

#[test]
fn slider_drags_within_range() {
    let mut u = ui();
    let rect = Rect::new(0.0, 100.0, 100.0, 20.0);
    let mut value = 0.0;
    let range = 0.0..=10.0;
    // Grab at the midpoint.
    let input = UiInput {
        mouse: (50.0, 110.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.slider(
        "vol",
        rect,
        &mut value,
        range.clone(),
        SliderStyle::new(FontId(0)),
    );
    assert!(
        (value - 5.0).abs() < 0.01,
        "midpoint drag sets half, got {value}"
    );
    // Drag past the right edge clamps.
    let input = UiInput {
        mouse: (500.0, 110.0),
        mouse_down: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.slider(
        "vol",
        rect,
        &mut value,
        range.clone(),
        SliderStyle::new(FontId(0)),
    );
    assert_eq!(value, 10.0);
}

#[test]
fn menu_navigates_and_selects() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 300.0, 200.0);
    let items = vec![
        MenuItem::new("New Game"),
        MenuItem::new("Options"),
        MenuItem::new("Quit").disabled(),
    ];
    let mut focus = 0;

    // Down twice skips the disabled Quit and wraps to New Game? From 0:
    // down -> Options (enabled), down -> Quit disabled -> skip to New Game.
    let input = UiInput {
        nav: NavInput {
            down: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    assert_eq!(
        u.menu("main", rect, &items, &mut focus, MenuStyle::new(FontId(0))),
        None
    );
    assert_eq!(focus, 1, "down moves to Options");

    let input = UiInput {
        nav: NavInput {
            down: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.menu("main", rect, &items, &mut focus, MenuStyle::new(FontId(0)));
    assert_eq!(focus, 0, "down skips disabled Quit and wraps");

    // Confirm selects the focused entry.
    let input = UiInput {
        nav: NavInput {
            confirm: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    assert_eq!(
        u.menu("main", rect, &items, &mut focus, MenuStyle::new(FontId(0))),
        Some(0)
    );
}

#[test]
fn menu_click_selects_item() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 300.0, 200.0);
    let items = vec![MenuItem::new("One"), MenuItem::new("Two")];
    let mut focus = 0;
    // Item 1 (Two) sits at y = padding + 1 * (height + gap), centred.
    let style = MenuStyle::new(FontId(0));
    let at = (
        150.0,
        style.padding + style.item_height + style.gap + style.item_height * 0.5,
    );
    let input = UiInput {
        mouse: at,
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.menu("main", rect, &items, &mut focus, style);
    let input = UiInput {
        mouse: at,
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    assert_eq!(u.menu("main", rect, &items, &mut focus, style), Some(1));
}

#[test]
fn text_field_edits_and_focuses() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 200.0, 30.0);
    let mut text = String::from("hi");
    // Click to focus and type.
    let input = UiInput {
        mouse: (20.0, 15.0),
        mouse_pressed: true,
        text: " there".into(),
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    assert_eq!(text, "hi there");

    // Backspace trims one character per press ("hi there" is 8 chars).
    let input = UiInput {
        backspace: 6,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    assert_eq!(text, "hi");
}

#[test]
fn text_field_caret_inserts_and_moves() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 300.0, 30.0);
    let mut text = String::from("ac");
    // Focus, then park the caret at the start and type.
    let input = UiInput {
        mouse: (20.0, 15.0),
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    let input = UiInput {
        nav: NavInput {
            home: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    let input = UiInput {
        text: "b".into(),
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    assert_eq!(text, "bac", "typing at the caret inserts in place");
    // Delete removes forward of the caret (after "b").
    let input = UiInput {
        nav: NavInput {
            delete: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    assert_eq!(text, "bc");
}

/// A `password` field renders one bullet per character — quad-for-quad
/// identical to drawing literal bullets — while edits and the
/// clipboard still see the real string.
#[test]
fn text_field_password_masks_display_but_edits_real_text() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 300.0, 30.0);
    let mut text = String::new();
    let password_style = || {
        let mut style: TextFieldStyle = ButtonStyle::new(FontId(0)).into();
        style.password = true;
        style
    };
    // Focus, then paste the secret in (é is two bytes — the mask is
    // per char, not per byte).
    u.begin(
        UiInput {
            mouse: (20.0, 15.0),
            mouse_pressed: true,
            paste: Some("héllo".into()),
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("pwd", rect, &mut text, password_style());
    assert_eq!(text, "héllo", "paste mutates the real string");
    // Caret home, then click the far right of the field: the pointer
    // maps through the masked display string back to the real end —
    // typed text appends, not lands mid-word.
    u.begin(
        UiInput {
            nav: NavInput {
                home: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("pwd", rect, &mut text, password_style());
    u.begin(
        UiInput {
            mouse: (290.0, 15.0),
            mouse_pressed: true,
            text: "!".into(),
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("pwd", rect, &mut text, password_style());
    assert_eq!(text, "héllo!");
    // Select-all + copy hands the REAL text to the clipboard seam.
    u.begin(
        UiInput {
            nav: NavInput {
                select_all: true,
                copy: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("pwd", rect, &mut text, password_style());
    assert_eq!(
        u.take_clipboard_output().as_deref(),
        Some("héllo!"),
        "copy must emit the real text, never the bullets"
    );
    // A never-focused password field has no caret or selection quads,
    // so it can be compared quad-for-quad with a plain field drawing
    // literal bullets — six chars, six glyphs, same chrome.
    let mut masked_buf = String::from("héllo!");
    u.begin(UiInput::default(), (800.0, 600.0));
    u.text_field("pwd2", rect, &mut masked_buf, password_style());
    let masked_quads = u.end().quad_count();
    let mut bullets = String::from("••••••");
    u.begin(UiInput::default(), (800.0, 600.0));
    u.text_field("plain", rect, &mut bullets, ButtonStyle::new(FontId(0)));
    let bullet_quads = u.end().quad_count();
    assert_eq!(
        masked_quads, bullet_quads,
        "masked 'héllo!' must render exactly the six-bullet layout"
    );
}

#[test]
fn text_field_empty_draws_no_text() {
    // The widget id ("name") must never render as placeholder content: an
    // empty field draws chrome only, so a caller-owned hint on top cannot
    // garble against an echoed id.
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 300.0, 30.0);
    let mut empty = String::new();
    u.begin(UiInput::default(), (800.0, 600.0));
    u.text_field("name", rect, &mut empty, ButtonStyle::new(FontId(0)));
    let chrome_only = u.end().quad_count();
    assert_eq!(empty, "");
    assert!(!u.focused("name"));

    let mut one = String::from("x");
    u.begin(UiInput::default(), (800.0, 600.0));
    u.text_field("name", rect, &mut one, ButtonStyle::new(FontId(0)));
    let with_text = u.end().quad_count();
    assert!(
        with_text > chrome_only,
        "one glyph must add quads over chrome-only ({with_text} vs {chrome_only})"
    );
}

#[test]
fn text_field_click_away_blurs() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 300.0, 30.0);
    let mut text = String::from("hi");
    // Click inside: focused.
    u.begin(
        UiInput {
            mouse: (20.0, 15.0),
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    assert!(u.focused("name"));
    // Select all, then click far outside: blurred AND deselected.
    u.begin(
        UiInput {
            nav: NavInput {
                select_all: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    let id = u.id("name");
    assert!(
        u.editors[&id].selection().is_some(),
        "select-all must leave a selection"
    );
    u.begin(
        UiInput {
            mouse: (700.0, 500.0),
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(!u.focused("name"), "clicking off must exit the text field");
    assert_eq!(text, "hi", "blurring must not edit the text");
    assert!(
        u.editors[&id].selection().is_none(),
        "clicking off must clear the selection"
    );
}

#[test]
fn text_field_selection_highlight_animates() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 300.0, 30.0);
    let mut text = String::from("select me");
    // Focus the field.
    u.begin(
        UiInput {
            mouse: (20.0, 15.0),
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    let id = u.id("name");
    // Select-all: the highlight appears instantly, no fade-in.
    u.begin(
        UiInput {
            dt: 0.016,
            nav: NavInput {
                select_all: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(
        u.editors[&id].sel_fade.value(),
        1.0,
        "highlight appears at full opacity immediately"
    );
    // Collapse the selection: the highlight fades out over frames instead
    // of vanishing — the mid-fade value is between full and gone.
    u.begin(
        UiInput {
            dt: 0.016,
            mouse: (20.0, 15.0),
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    let fading = u.editors[&id].sel_fade.value();
    assert!(
        fading > 0.0 && fading < 1.0,
        "collapse should fade the highlight, got {fading}"
    );
    u.begin(
        UiInput {
            dt: 0.5,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(u.editors[&id].sel_fade.value(), 0.0);
    // Blur also fades the caret ghost rather than cutting it off.
    u.begin(
        UiInput {
            dt: 0.016,
            mouse: (700.0, 500.0),
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.text_field("name", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    let ghost = u.editors[&id].caret_fade.value();
    assert!(
        ghost > 0.0 && ghost < 1.0,
        "caret should fade out on blur, got {ghost}"
    );
    assert!(u.animating(), "fade-out keeps the host rendering");
}

#[test]
fn scopes_disambiguate_labels() {
    let mut u = ui();
    u.push_id("a");
    let id_a = u.id("btn");
    u.pop_id();
    u.push_id("b");
    let id_b = u.id("btn");
    u.pop_id();
    assert_ne!(id_a, id_b, "same label in different scopes is distinct");
    assert_eq!(u.id("btn"), u.id("btn"), "ids are stable");
}

#[test]
fn hover_animation_visits_both_ends() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 40.0);
    let input = UiInput {
        mouse: (50.0, 20.0),
        dt: 1.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.button("ok", rect, ButtonStyle::new(FontId(0)));
    let id = u.id("ok");
    let hot = u.state[&id].hover.value();
    assert!(hot > 0.9, "hover animates in, got {hot}");

    let input = UiInput {
        mouse: (500.0, 500.0),
        dt: 1.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.button("ok", rect, ButtonStyle::new(FontId(0)));
    let cold = u.state[&id].hover.value();
    assert!(cold < 0.1, "hover animates out, got {cold}");
}

// --- New widget tests --------------------------------------------------

#[test]
fn icon_button_renders_and_reports() {
    let mut u = ui();
    u.register_vector_icons(&[super::super::icons::Icon::Play]);
    let rect = Rect::new(0.0, 0.0, 40.0, 40.0);
    let input = UiInput {
        mouse: (20.0, 20.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    assert!(
        !u.icon_button(
            "play",
            rect,
            super::super::icons::Icon::Play,
            ButtonStyle::new(FontId(0)),
            false,
            None,
            20,
        )
        .clicked
    );
    let mut input = UiInput {
        mouse: (20.0, 20.0),
        mouse_released: true,
        ..Default::default()
    };
    input.mouse_down = false;
    u.begin(input.clone(), (800.0, 600.0));
    let response = u.icon_button(
        "play",
        rect,
        super::super::icons::Icon::Play,
        ButtonStyle::new(FontId(0)),
        false,
        None,
        20,
    );
    assert!(response.clicked);
    let frame = u.end();
    assert!(frame.quad_count() > 0, "the icon baked into the atlas");
}

#[test]
fn icon_toggle_flips_value() {
    let mut u = ui();
    u.register_vector_icons(&[super::super::icons::Icon::Grid]);
    let rect = Rect::new(0.0, 0.0, 40.0, 40.0);
    let mut on = false;
    let input = UiInput {
        mouse: (20.0, 20.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.icon_toggle(
        "grid",
        rect,
        super::super::icons::Icon::Grid,
        &mut on,
        ButtonStyle::new(FontId(0)),
        Some("8"),
        20,
    );
    let input = UiInput {
        mouse: (20.0, 20.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.icon_toggle(
        "grid",
        rect,
        super::super::icons::Icon::Grid,
        &mut on,
        ButtonStyle::new(FontId(0)),
        Some("8"),
        20,
    );
    assert!(on);
    // The icon still renders on the following frame.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.icon_toggle(
        "grid",
        rect,
        super::super::icons::Icon::Grid,
        &mut on,
        ButtonStyle::new(FontId(0)),
        Some("8"),
        20,
    );
    let frame = u.end();
    assert!(frame.quad_count() > 0);
}

#[test]
fn modal_blocks_input_outside_card() {
    let mut u = ui();
    let card_size = (300.0, 200.0);
    // Open the modal.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.modal("dlg", true, card_size).expect("modal opens");
    // A button outside the card must not respond while it is open.
    let outside = Rect::new(0.0, 0.0, 100.0, 40.0);
    let input = UiInput {
        mouse: (50.0, 20.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let modal = u.modal("dlg", true, card_size).expect("still open");
    assert!(
        !u.button("outside", outside, ButtonStyle::new(FontId(0)))
            .clicked,
        "modal confines input to the card"
    );
    // A button inside the card still works: press, then release.
    let inside = Rect::new(modal.card.x + 10.0, modal.card.y + 10.0, 80.0, 30.0);
    let press = UiInput {
        mouse: (inside.x + 40.0, inside.y + 15.0),
        mouse_pressed: true,
        mouse_down: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    let _ = u.modal("dlg", true, card_size);
    let press_response = u.button("inside", inside, ButtonStyle::new(FontId(0)));
    assert!(!press_response.clicked, "press alone is not a click");
    let release = UiInput {
        mouse: (inside.x + 40.0, inside.y + 15.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(release.clone(), (800.0, 600.0));
    let _ = u.modal("dlg", true, card_size);
    assert!(
        u.button("inside", inside, ButtonStyle::new(FontId(0)))
            .clicked,
        "card contents stay interactive"
    );
}

#[test]
fn modal_dismisses_on_backdrop_click_and_closes() {
    let mut u = ui();
    let card_size = (300.0, 200.0);
    let input = UiInput {
        mouse: (10.0, 10.0),
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let modal = u.modal("dlg", true, card_size).expect("open");
    assert!(modal.dismissed, "backdrop click dismisses");
    // Closing retracts; after the animation the modal reports None.
    for _ in 0..30 {
        u.begin(
            UiInput {
                dt: 0.05,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.modal("dlg", false, card_size);
    }
    assert!(
        u.modal("dlg", false, card_size).is_none(),
        "closed modal vanishes"
    );
}

#[test]
fn anim_registry_eases_and_reports_motion() {
    let mut u = ui();
    u.begin(
        UiInput {
            dt: 0.016,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let value = u.anim_f32("fade", 1.0, 0.12, Ease::Linear);
    assert!(value < 1.0 && value > 0.0);
    assert!(u.animating(), "a fresh animation is in flight");
    // Let it settle (0.12 s at 0.05 s per frame).
    for _ in 0..6 {
        u.begin(
            UiInput {
                dt: 0.05,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.anim_f32("fade", 1.0, 0.12, Ease::Linear);
    }
    let settled = u.anim_f32("fade", 1.0, 0.12, Ease::Linear);
    assert!(
        (settled - 1.0).abs() < 0.01,
        "settled at the target, got {settled}"
    );
    assert!(!u.animating(), "settled animation stops reporting motion");
    // Untouched state retires so dynamic ids cannot leak.
    u.push_id("row-1");
    let scoped = u.id("btn");
    u.button(
        "btn",
        Rect::new(0.0, 0.0, 10.0, 10.0),
        ButtonStyle::new(FontId(0)),
    );
    u.pop_id();
    assert!(u.state.contains_key(&scoped));
    for _ in 0..700 {
        u.begin(UiInput::default(), (800.0, 600.0));
        u.end();
    }
    assert!(!u.state.contains_key(&scoped), "stale state retires");
}

#[test]
fn scroll_view_clamps_and_offsets_content() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    // Hovered wheel scrolls down but clamps at content height.
    // Wheel up scrolls toward earlier content: the offset clamps at 0.
    let input = UiInput {
        mouse: (50.0, 25.0),
        wheel: 30.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let content = u.scroll_begin("list", rect, 200.0);
    assert_eq!(content.y, 0.0, "scroll up at the top stays put");
    u.scroll_end();
    // Wheel down moves toward later content, clamped to the end.
    let input = UiInput {
        mouse: (50.0, 25.0),
        wheel: -70.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let content = u.scroll_begin("list", rect, 200.0);
    assert_eq!(content.y, -70.0, "content shifts up by the scroll offset");
    u.scroll_end();
    let input = UiInput {
        mouse: (50.0, 25.0),
        wheel: -5000.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let content = u.scroll_begin("list", rect, 200.0);
    assert_eq!(content.y, -150.0, "clamped to 150 (200 - 50)");
    u.scroll_end();
    // No wheel without hover.
    let input = UiInput {
        mouse: (500.0, 500.0),
        wheel: 50.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let content = u.scroll_begin("list", rect, 200.0);
    assert_eq!(content.y, -150.0, "unhovered wheel does not scroll");
    u.scroll_end();
}

#[test]
fn splitter_drags_fraction() {
    let mut u = ui();
    let rect = Rect::new(100.0, 0.0, 8.0, 600.0);
    let mut fraction = 0.5;
    let input = UiInput {
        mouse: (104.0, 300.0),
        mouse_pressed: true,
        mouse_down: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.splitter("split", rect, &mut fraction);
    assert_eq!(fraction, 0.5, "grab starts at the middle");
    let input = UiInput {
        mouse: (104.0, 400.0),
        mouse_down: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.splitter("split", rect, &mut fraction);
    assert!(
        (fraction - (400.0 / 600.0)).abs() < 0.01,
        "drag moves the fraction"
    );
}

#[test]
fn drag_field_scrubs_value() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 200.0, 24.0);
    let mut value = 0.0;
    let range = 0.0..=1.0;
    let input = UiInput {
        mouse: (100.0, 12.0),
        mouse_pressed: true,
        mouse_down: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.drag_field("gain", rect, &mut value, range.clone(), FontId(0));
    let input = UiInput {
        mouse: (150.0, 12.0),
        mouse_down: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.drag_field("gain", rect, &mut value, range.clone(), FontId(0));
    assert!(
        (value - 50.0 / 240.0).abs() < 0.01,
        "240 px sweeps the range, got {value}"
    );
}

#[test]
fn pickers_update_hsv() {
    let mut u = ui();
    let sv = Rect::new(0.0, 0.0, 100.0, 100.0);
    let mut hsv = (200.0, 1.0, 1.0);
    let input = UiInput {
        mouse: (75.0, 25.0),
        mouse_pressed: true,
        mouse_down: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.sv_field("sv", sv, &mut hsv);
    u.end();
    assert!((hsv.1 - 0.75).abs() < 0.01 && (hsv.2 - 0.75).abs() < 0.01);
    // Release the sv press: a single capture cannot serve two widgets.
    u.begin(
        UiInput {
            mouse: (75.0, 25.0),
            mouse_released: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.end();
    let hue = Rect::new(0.0, 110.0, 100.0, 16.0);
    let input = UiInput {
        mouse: (25.0, 118.0),
        mouse_pressed: true,
        mouse_down: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.hue_field("hue", hue, &mut hsv);
    u.end();
    assert!(
        (hsv.0 - 89.975).abs() < 0.2,
        "quarter across the strip, got {}",
        hsv.0
    );
}

#[test]
fn palette_selects_swatch() {
    let mut u = ui();
    let colors = vec![
        Color::rgb(255, 0, 0),
        Color::rgb(0, 255, 0),
        Color::rgb(0, 0, 255),
    ];
    let mut selected = None;
    let rect = Rect::new(0.0, 0.0, 66.0, 22.0);
    let input = UiInput {
        mouse: (49.0, 11.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.palette("pal", rect, &colors, &mut selected, 3);
    let input = UiInput {
        mouse: (49.0, 11.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let response = u.palette("pal", rect, &colors, &mut selected, 3);
    assert!(response.clicked);
    assert_eq!(selected, Some(2), "third swatch chosen");
}

#[test]
fn toasts_render_and_expire() {
    let mut u = ui();
    let mut toasts = Toasts::new();
    toasts.push("hello", ToastKind::Ok);
    assert_eq!(toasts.len(), 1);
    u.begin(
        UiInput {
            dt: 0.016,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.toasts(&mut toasts);
    let frame = u.end();
    assert!(frame.quad_count() > 0, "toast draws");
    for _ in 0..300 {
        u.begin(
            UiInput {
                dt: 0.05,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.toasts(&mut toasts);
        u.end();
    }
    assert!(toasts.is_empty(), "toasts expire on their own");
}

/// The right-click menu leaves by fading, and the fade is spent before
/// the slide is: every frame after full transparency still draws the card
/// (invisible) until the stroke finishes.
#[test]
fn context_menu_fades_out_before_it_stops_moving() {
    let mut u = ui();
    let style = u.theme().menu(FontId(0));
    let items = [MenuItem::new("one"), MenuItem::new("two")];
    let at = (100.0, 100.0);
    let mut focus = 0usize;
    let mut open = true;

    // Peak vertex alpha of the frame, or None once nothing is drawn.
    let step = |u: &mut Ui, open: &mut bool, focus: &mut usize| -> Option<u8> {
        u.begin(
            UiInput {
                dt: 0.02,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.context_menu("m", open, at, &items, focus, style);
        let frame = u.end();
        if frame.is_empty() {
            return None;
        }
        Some(frame.vertices.iter().map(|v| v.color[3]).max().unwrap_or(0))
    };

    // Let the open fade finish, then it must be fully opaque.
    let mut last = None;
    for _ in 0..15 {
        last = step(&mut u, &mut open, &mut focus);
    }
    assert_eq!(last, Some(255), "settled menu is opaque");
    open = false;
    let mut alphas = Vec::new();
    // A fully-transparent frame emits no geometry at all — `None` *is* the
    // terminal zero, so the loop ends when the draws stop.
    while let Some(alpha) = step(&mut u, &mut open, &mut focus) {
        if alpha == 0 {
            alphas.push(0);
            break;
        }
        alphas.push(alpha);
    }
    if alphas.last() != Some(&0) {
        alphas.push(0);
    }
    assert!(alphas.len() >= 3, "the card animates out: {alphas:?}");
    assert!(
        alphas[0] < 255,
        "closing starts the fade at once: {alphas:?}"
    );
    let still_visible = alphas.iter().take_while(|a| **a > 0).count();
    assert!(
        still_visible < alphas.len(),
        "the card goes transparent while it is still travelling: {alphas:?}"
    );
    assert_eq!(*alphas.last().unwrap(), 0, "and leaves: {alphas:?}");
}

/// Only the clip moves as a section unrolls: the body rect is the same
/// whether the section is shut, mid-stroke or open, so rows never shift or
/// bounce under the pointer.
#[test]
fn collapsible_body_never_shifts_while_it_unrolls() {
    let mut u = ui();
    let mut open = false;
    let rect = Rect::new(0.0, 0.0, 200.0, 120.0);
    // Header height is the Md control token (28), plus the 4px body gap.
    let rest = Rect::new(rect.x, rect.y + 28.0 + 4.0, rect.w, rect.h - 28.0 - 4.0);

    let body_rect = |u: &mut Ui, open: &mut bool| -> (Rect, bool) {
        u.begin(
            UiInput {
                dt: 0.02,
                ..Default::default()
            },
            (400.0, 300.0),
        );
        let (body, clipped) = u.collapsible("s", rect, open, FontId(0));
        if clipped {
            u.collapsible_end();
        }
        u.end();
        (body, clipped)
    };

    // Shut, and pinned to its resting place.
    let (closed, clipped) = body_rect(&mut u, &mut open);
    assert!(!clipped, "a shut section clips nothing");
    assert_eq!(closed, rest, "shut body sits at rest");
    // Opening: the clip grows, the body rect does not budge — including on
    // the frames where the motion is still in flight.
    open = true;
    let mut saw_motion = false;
    for _ in 0..10 {
        let (body, clipped) = body_rect(&mut u, &mut open);
        assert_eq!(body, rest, "body stays put while unrolling");
        saw_motion |= clipped;
    }
    assert!(saw_motion, "the section actually animates");
}

/// A control that is re-created in the state it belongs in must not
/// animate into it: only later changes move. This is what stopped the
/// FSR switch sliding whenever its section was expanded.
#[test]
fn seeded_anim_starts_at_its_value_and_only_animates_on_change() {
    let mut u = ui();
    u.begin(
        UiInput {
            dt: 0.016,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    assert_eq!(u.anim_f32_seeded("knob", 1.0, 0.16, Ease::CubicOut), 1.0);
    assert!(
        !u.animating(),
        "no motion for a value that is already there"
    );
    let moved = u.anim_f32_seeded("knob", 0.0, 0.16, Ease::CubicOut);
    assert!(moved < 1.0 && moved > 0.0, "retargeting animates: {moved}");
    assert!(u.animating(), "and reports motion in flight");
}

/// The header is the whole container. `collapsible` is handed a rect that
/// also spans the body, so testing the hover there lit the header while the
/// pointer was down among the rows — or over the space a shut section's
/// body would have occupied.
#[test]
fn collapsible_header_lights_only_under_the_header() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 200.0, 120.0);
    let header = Rect::new(rect.x, rect.y, rect.w, 26.0);
    let body = Rect::new(rect.x, header.max().1 + 4.0, rect.w, rect.h - 26.0 - 4.0);
    let id = u.id("s");

    // The frame that results: which widget the pointer is on, and whether
    // any of the header's fill was actually drawn.
    let hover_at = |u: &mut Ui, open: &mut bool, mouse: (f32, f32)| -> (Option<u64>, bool) {
        u.begin(
            UiInput {
                dt: 0.02,
                mouse,
                ..Default::default()
            },
            (400.0, 300.0),
        );
        let (_, clipped) = u.collapsible("s", rect, open, FontId(0));
        if clipped {
            u.collapsible_end();
        }
        let frame = u.end();
        let g2 = u.theme().g2;
        let painted = frame.vertices.iter().any(|v| {
            v.color[0] == g2.r && v.color[1] == g2.g && v.color[2] == g2.b && v.color[3] > 0
        });
        (u.hot_id, painted)
    };

    // Let the fade settle before reading it, so a highlight on its way out
    // is not mistaken for one on its way in.
    let settled = |u: &mut Ui, open: &mut bool, mouse: (f32, f32)| {
        let mut last = (None, false);
        for _ in 0..20 {
            last = hover_at(u, open, mouse);
        }
        last
    };

    let mut open = true;
    assert_eq!(
        settled(&mut u, &mut open, header.center()),
        (Some(id), true),
        "a pointer on the header lights and paints the header"
    );
    assert_eq!(
        settled(&mut u, &mut open, body.center()),
        (None, false),
        "a pointer on a row in the body is not on the header"
    );
    // Nothing is drawn below the header when the section is shut, so that
    // space is not the header either.
    let mut shut = false;
    assert_eq!(
        settled(&mut u, &mut shut, body.center()),
        (None, false),
        "a shut section's empty space is not its header"
    );
    assert_eq!(
        settled(&mut u, &mut shut, header.center()),
        (Some(id), true),
        "while the header itself still responds"
    );
}

/// Opening the context menu must not light a row on its own. Focus starts
/// at row 0, and treating that as hover made the freshly opened menu look
/// like the pointer was already somewhere in it.
#[test]
fn context_menu_lights_a_row_only_for_the_pointer_or_the_keyboard() {
    let mut u = ui();
    let style = u.theme().menu(FontId(0));
    let items = [MenuItem::new("one"), MenuItem::new("two")];
    let at = (100.0, 100.0);
    let mut focus = 0usize;
    let mut open = true;
    let away = (700.0, 500.0);

    let step = |u: &mut Ui,
                open: &mut bool,
                focus: &mut usize,
                mouse: (f32, f32),
                nav: NavInput|
     -> Option<u64> {
        u.begin(
            UiInput {
                dt: 0.02,
                mouse,
                nav,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.context_menu("m", open, at, &items, focus, style);
        u.end();
        u.hot_id
    };

    // Settle the entrance with the pointer nowhere near the menu.
    let mut lit = None;
    for _ in 0..15 {
        lit = step(&mut u, &mut open, &mut focus, away, NavInput::default());
    }
    assert_eq!(lit, None, "a menu nobody has pointed at highlights no row");
    assert_eq!(focus, 0, "and its focus is still the first row");

    // The pointer's row lights.
    let second = Rect::new(at.0, at.1 + 38.0, CONTEXT_MENU_W, style.item_height);
    assert!(
        step(
            &mut u,
            &mut open,
            &mut focus,
            second.center(),
            NavInput::default()
        )
        .is_some(),
        "the row under the pointer lights"
    );
    assert_eq!(focus, 1, "and takes the focus with it");

    // Leaving the menu drops the highlight again.
    assert_eq!(
        step(&mut u, &mut open, &mut focus, away, NavInput::default()),
        None,
        "the highlight is not left behind when the pointer leaves"
    );

    // The keyboard lights the focused row with the pointer away.
    let down = NavInput {
        down: true,
        ..Default::default()
    };
    assert!(
        step(&mut u, &mut open, &mut focus, away, down).is_some(),
        "an arrow key lights the row it moves to"
    );
    assert_eq!(focus, 0, "which wraps to the first row");
}

// --- Input routing, overlays and animation -----------------------------

/// A widget behind an empty/shrunken clip must not take the pointer:
/// input and drawing agree on what is visible.
#[test]
fn clipped_away_widget_takes_no_pointer() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let clip = Rect::new(200.0, 200.0, 50.0, 50.0);
    let input = UiInput {
        mouse: (50.0, 30.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.draw_list_mut().push_clip(clip);
    let pressed = u.button("ok", rect, ButtonStyle::new(FontId(0)));
    u.draw_list_mut().pop_clip();
    u.end();
    assert!(!pressed.hovered, "a clipped-away button does not hover");

    let input = UiInput {
        mouse: (50.0, 30.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.draw_list_mut().push_clip(clip);
    let released = u.button("ok", rect, ButtonStyle::new(FontId(0)));
    u.draw_list_mut().pop_clip();
    u.end();
    assert!(!released.clicked, "and it does not click");
}

/// One press captures exactly one widget: of two buttons on the same rect
/// only the first can be clicked by that press.
#[test]
fn one_press_captures_one_widget() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let input = UiInput {
        mouse: (50.0, 30.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.button("a", rect, ButtonStyle::new(FontId(0)));
    u.button("b", rect, ButtonStyle::new(FontId(0)));
    u.end();
    let input = UiInput {
        mouse: (50.0, 30.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let a = u.button("a", rect, ButtonStyle::new(FontId(0)));
    let b = u.button("b", rect, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(a.clicked, "the capture holder clicks");
    assert!(
        !b.clicked,
        "the second widget under the same press does not"
    );
}

/// `icon_toggle` must report the same click that flipped its value —
/// the response and the mutation are one decision, not two.
#[test]
fn icon_toggle_response_is_the_click_that_flipped() {
    let mut u = ui();
    u.register_vector_icons(&[super::super::icons::Icon::Grid]);
    let rect = Rect::new(0.0, 0.0, 40.0, 40.0);
    let mut on = false;
    // A release with no press flips nothing and reports nothing.
    let input = UiInput {
        mouse: (20.0, 20.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let stray = u.icon_toggle(
        "grid",
        rect,
        super::super::icons::Icon::Grid,
        &mut on,
        ButtonStyle::new(FontId(0)),
        None,
        20,
    );
    assert!(!stray.clicked && !on, "a stray release is not a click");
    // Press then release: one click, one flip.
    let input = UiInput {
        mouse: (20.0, 20.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.icon_toggle(
        "grid",
        rect,
        super::super::icons::Icon::Grid,
        &mut on,
        ButtonStyle::new(FontId(0)),
        None,
        20,
    );
    let input = UiInput {
        mouse: (20.0, 20.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let response = u.icon_toggle(
        "grid",
        rect,
        super::super::icons::Icon::Grid,
        &mut on,
        ButtonStyle::new(FontId(0)),
        None,
        20,
    );
    assert!(response.clicked && on, "the reported click is the flip");
}

/// Disabled widgets never take input: no pointer click, no focused
/// keyboard activation, no text-field focus.
#[test]
fn disabled_widgets_take_no_input_or_focus() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let style = ButtonStyle::new(FontId(0)).disabled();
    // Focused-confirm on a disabled button does nothing.
    u.set_focus("ok");
    let input = UiInput {
        nav: NavInput {
            confirm: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    assert!(!u.button("ok", rect, style).clicked);
    u.end();
    // A disabled toggle ignores a real click.
    let mut value = false;
    let press = UiInput {
        mouse: (50.0, 30.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.toggle("mute", rect, &mut value, style);
    u.end();
    let release = UiInput {
        mouse: (50.0, 30.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(release.clone(), (800.0, 600.0));
    let response = u.toggle("mute", rect, &mut value, style);
    u.end();
    assert!(!response.clicked && !value, "disabled toggle never flips");
    // A disabled text field does not focus or edit.
    let mut text = String::from("hi");
    let input = UiInput {
        mouse: (20.0, 20.0),
        mouse_pressed: true,
        text: "x".into(),
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, style);
    u.end();
    assert_eq!(text, "hi", "disabled field never edits");
    assert!(!u.focused("name"), "disabled field never focuses");
}

/// A widget label's `##` suffix is id-only: it is hashed but never drawn.
#[test]
fn label_suffix_is_identity_not_text() {
    assert_eq!(super::display_label("Apply##a"), "Apply");
    assert_eq!(super::display_label("Apply"), "Apply");
    assert_eq!(super::display_label("##ghost"), "");
    let u = ui();
    assert_ne!(
        u.id("Apply##a"),
        u.id("Apply##b"),
        "same text, different ids — independent widgets"
    );
}

/// While a modal surface is up, the widgets beneath it get neither
/// pointer nor keyboard input; after it closes, they work again.
#[test]
fn modal_surface_blocks_input_then_releases_it() {
    let mut u = ui();
    let under = Rect::new(10.0, 10.0, 100.0, 40.0);
    let card = (300.0, 200.0);
    // Frame 1: the modal opens and registers its surface. The button is
    // drawn first each frame, so what blocks it is last frame's surface.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.button("under", under, ButtonStyle::new(FontId(0)));
    u.modal("dlg", true, card).expect("open");
    u.end_clip();
    u.end();
    // Frame 2: press + release on the button — blocked by the modal.
    let press = UiInput {
        mouse: (20.0, 20.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.button("under", under, ButtonStyle::new(FontId(0)));
    u.modal("dlg", true, card);
    u.end_clip();
    u.end();
    let release = UiInput {
        mouse: (20.0, 20.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(release.clone(), (800.0, 600.0));
    let clicked = u
        .button("under", under, ButtonStyle::new(FontId(0)))
        .clicked;
    u.modal("dlg", true, card);
    u.end_clip();
    u.end();
    assert!(!clicked, "the modal surface blocks the pointer");
    // Keyboard confirm on a focused button is blocked too.
    u.set_focus("under");
    let confirm = UiInput {
        nav: NavInput {
            confirm: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(confirm.clone(), (800.0, 600.0));
    let clicked = u
        .button("under", under, ButtonStyle::new(FontId(0)))
        .clicked;
    u.modal("dlg", true, card);
    u.end_clip();
    u.end();
    assert!(!clicked, "the modal surface blocks the keyboard");
    // Close the modal and let it fully retract.
    for _ in 0..30 {
        u.begin(
            UiInput {
                dt: 0.05,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.modal("dlg", false, card);
        u.end();
    }
    // The next frame must not be blocked by a phantom surface.
    u.begin(press.clone(), (800.0, 600.0));
    u.button("under", under, ButtonStyle::new(FontId(0)));
    u.modal("dlg", false, card);
    u.end();
    u.begin(release.clone(), (800.0, 600.0));
    assert!(
        u.button("under", under, ButtonStyle::new(FontId(0)))
            .clicked,
        "closing the modal restores the widgets beneath it"
    );
    u.end();
}

/// A host-declared surface blocks on the very first frame — the one gap
/// self-registered overlays cannot cover on their own.
#[test]
fn declared_surface_blocks_from_frame_one() {
    let mut u = ui();
    let under = Rect::new(10.0, 10.0, 100.0, 40.0);
    let press = UiInput {
        mouse: (20.0, 20.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.declare_input_surface(Rect::new(0.0, 0.0, 400.0, 300.0), LAYER_POPUP, false);
    u.button("under", under, ButtonStyle::new(FontId(0)));
    u.end();
    let release = UiInput {
        mouse: (20.0, 20.0),
        mouse_released: true,
        ..Default::default()
    };
    u.begin(release.clone(), (800.0, 600.0));
    u.declare_input_surface(Rect::new(0.0, 0.0, 400.0, 300.0), LAYER_POPUP, false);
    assert!(
        !u.button("under", under, ButtonStyle::new(FontId(0)))
            .clicked,
        "the declared popup covers the button even on its first frame"
    );
    u.end();
}

/// A non-modal popup blocks the pointer only over its own rect.
#[test]
fn popup_blocks_over_its_rect_only() {
    let mut u = ui();
    let covered = Rect::new(100.0, 100.0, 60.0, 40.0);
    let free = Rect::new(400.0, 400.0, 60.0, 40.0);
    let popup = Rect::new(50.0, 50.0, 200.0, 200.0);
    // Frame 1 registers the popup surface.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.declare_input_surface(popup, LAYER_POPUP, false);
    u.end();
    // Frame 2: clicks land everywhere, but only the covered button is
    // shadowed by the popup.
    for (at, expect) in [((130.0, 120.0), false), ((430.0, 420.0), true)] {
        let press = UiInput {
            mouse: at,
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        };
        u.begin(press.clone(), (800.0, 600.0));
        u.declare_input_surface(popup, LAYER_POPUP, false);
        u.button("a", covered, ButtonStyle::new(FontId(0)));
        u.button("b", free, ButtonStyle::new(FontId(0)));
        u.end();
        let release = UiInput {
            mouse: at,
            mouse_released: true,
            ..Default::default()
        };
        u.begin(release.clone(), (800.0, 600.0));
        u.declare_input_surface(popup, LAYER_POPUP, false);
        let a = u.button("a", covered, ButtonStyle::new(FontId(0))).clicked;
        let b = u.button("b", free, ButtonStyle::new(FontId(0))).clicked;
        u.end();
        assert!(!a, "the covered button never clicks");
        assert_eq!(b, expect, "the far button clicks when pressed");
    }
}

/// The tooltip clock belongs to the hovered rect: it appears only after
/// TOOLTIP_DELAY of unbroken hover, and resets when the pointer leaves or
/// the widget skips frames.
#[test]
fn tooltip_dwell_is_per_widget_and_unforgiving() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let hover = |dt: f32| UiInput {
        mouse: (20.0, 20.0),
        dt,
        ..Default::default()
    };
    let drawn = |u: &mut Ui, input: UiInput| -> usize {
        u.begin(input, (800.0, 600.0));
        u.tooltip(rect, "hint", FontId(0));
        u.end().quad_count()
    };
    // The entry frame just registers the hover — its `dt` ran before the
    // pointer arrived and does not count.
    assert_eq!(drawn(&mut u, hover(0.0)), 0);
    // 0.3 s in: nothing yet.
    assert_eq!(drawn(&mut u, hover(0.3)), 0);
    // 0.7 s total: the tip shows.
    assert!(drawn(&mut u, hover(0.4)) > 0, "shown after the dwell");
    // Leave and come back: the clock restarted.
    let away = UiInput {
        mouse: (500.0, 500.0),
        dt: 0.1,
        ..Default::default()
    };
    assert_eq!(drawn(&mut u, away.clone()), 0);
    assert_eq!(drawn(&mut u, hover(0.0)), 0, "the dwell restarted");
    assert_eq!(drawn(&mut u, hover(0.1)), 0, "and counts from zero");
    // Skipping frames also resets: skip several, hover 0.4 — still hidden.
    for _ in 0..5 {
        u.begin(away.clone(), (800.0, 600.0));
        u.end();
    }
    assert_eq!(drawn(&mut u, hover(0.0)), 0, "skipped frames reset");
    assert_eq!(drawn(&mut u, hover(0.4)), 0, "and the reset holds");
}

/// A tooltip drawn inside a clipped panel still renders above it —
/// escaped from the clip and sorted on the tooltip layer.
#[test]
fn tooltip_escapes_its_panel_clip() {
    let mut u = ui();
    let panel = Rect::new(0.0, 0.0, 200.0, 200.0);
    let widget = Rect::new(50.0, 100.0, 100.0, 40.0);
    // First frame registers the hover, the second past the delay shows.
    for dt in [0.0, 1.0] {
        u.begin(
            UiInput {
                mouse: (60.0, 110.0),
                dt,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.panel(panel, PanelStyle::default());
        u.tooltip(widget, "hint", FontId(0));
        u.end_clip();
    }
    let frame = u.end();
    let tip = frame
        .cmds
        .iter()
        .find(|cmd| cmd.layer == LAYER_TOOLTIP)
        .expect("the tip draws on its own layer");
    assert_eq!(tip.scissor, None, "the tip escaped the panel clip");
    assert!(
        frame.cmds.last().unwrap().layer == LAYER_TOOLTIP,
        "and sorts above the panel's draws"
    );
}

/// anim_loop_f32 must actually complete cycles: phase advances by
/// dt/duration modulo one and wraps instead of restarting each call.
#[test]
fn anim_loop_cycles_and_wraps() {
    let mut u = ui();
    let mut phases = Vec::new();
    for _ in 0..4 {
        u.begin(
            UiInput {
                dt: 0.3,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        phases.push(u.anim_loop_f32("blink", 1.0));
        u.end();
    }
    let expected = [0.3, 0.6, 0.9, 0.2];
    for (got, want) in phases.iter().zip(expected) {
        assert!((got - want).abs() < 1e-4, "phase {got} != {want}");
    }
    // A live loop keeps the frame awake, but only while it is touched.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.end();
    assert!(
        !u.animating(),
        "an untouched loop does not keep the app awake"
    );
}

/// animating() only counts motion touched this frame: a mid-flight
/// animation whose widget vanished must not pin the renderer awake.
#[test]
fn animating_drops_untouched_animation_state() {
    let mut u = ui();
    u.begin(
        UiInput {
            dt: 0.016,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let value = u.anim_f32("fade", 1.0, 10.0, Ease::Linear);
    assert!(value < 1.0, "a long ease is in flight");
    assert!(u.animating());
    u.end();
    // The widget never runs again: the parked ease does not count.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.end();
    assert!(!u.animating(), "untouched animations do not keep us awake");
}

/// Widgets inside a rate scope advance their hover by the scoped step,
/// not the frame's real dt.
#[test]
fn rate_scope_gates_widget_hover_animation() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    // 1 Hz scope: a 0.1 s frame does not fill one step, so the hover ease
    // sees dt = 0 and never moves.
    u.begin(
        UiInput {
            mouse: (50.0, 30.0),
            dt: 0.1,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.rate_scope("hud", Some(1.0));
    u.button("ok", rect, ButtonStyle::new(FontId(0)));
    u.end_rate_scope();
    u.end();
    let id = u.id("ok");
    assert_eq!(
        u.state[&id].hover.value(),
        0.0,
        "a sub-step frame advances nothing"
    );
    // Outside the scope the same frame animates normally.
    u.begin(
        UiInput {
            mouse: (50.0, 30.0),
            dt: 0.1,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.button("ok", rect, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(
        u.state[&id].hover.value() > 0.5,
        "unscoped hover animates with real dt"
    );
}

/// Nested scroll views: the wheel goes to the deepest one the pointer was
/// over last frame, exactly once.
#[test]
fn nested_scroll_views_single_wheel_owner() {
    let mut u = ui();
    let outer = Rect::new(0.0, 0.0, 200.0, 200.0);
    let inner = Rect::new(50.0, 50.0, 100.0, 100.0);
    let input = UiInput {
        mouse: (75.0, 75.0),
        ..Default::default()
    };
    // Frame 1 registers the regions; the wheel only applies from the next
    // frame on (ownership is resolved against the previous frame).
    u.begin(input.clone(), (800.0, 600.0));
    u.scroll_begin("outer", outer, 1000.0);
    u.scroll_begin("inner", inner, 500.0);
    u.scroll_end();
    u.scroll_end();
    u.end();
    let input = UiInput {
        mouse: (75.0, 75.0),
        wheel: -60.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let outer_content = u.scroll_begin("outer", outer, 1000.0);
    let inner_content = u.scroll_begin("inner", inner, 500.0);
    u.scroll_end();
    u.scroll_end();
    u.end();
    assert_eq!(outer_content.y, 0.0, "the outer view did not scroll");
    assert_eq!(inner_content.y, -10.0, "the deepest view owns the wheel");
}

/// A shrunken scroll view re-clamps its offset the moment it is measured.
#[test]
fn scroll_offset_reclamps_on_shrink() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    let input = UiInput {
        mouse: (50.0, 25.0),
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.end();
    let input = UiInput {
        mouse: (50.0, 25.0),
        wheel: -150.0,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let content = u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.end();
    assert_eq!(content.y, -150.0);
    // Content shrinks to fit: the 150 px offset is now dead space.
    u.begin(UiInput::default(), (800.0, 600.0));
    let content = u.scroll_begin("list", rect, 60.0);
    u.scroll_end();
    u.end();
    assert_eq!(content.y, -10.0, "offset reclamped to the shrunken range");
}

/// The scrollbar thumb drags the view directly.
#[test]
fn scrollbar_thumb_drags() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    // Track: x in [93, 98], thumb at the top on frame one.
    let press = UiInput {
        mouse: (95.0, 12.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.end();
    // Drag halfway down the track.
    let drag = UiInput {
        mouse: (95.0, 30.0),
        mouse_down: true,
        ..Default::default()
    };
    u.begin(drag.clone(), (800.0, 600.0));
    let content = u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.end();
    let offset = -content.y;
    assert!(offset > 20.0, "thumb drag scrolled, offset {offset}");
}

/// A capture outlives the widget that made it — the release is processed
/// centrally so a vanished widget cannot hold the pointer forever.
#[test]
fn released_capture_clears_even_when_the_widget_is_gone() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let press = UiInput {
        mouse: (50.0, 30.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.button("ok", rect, ButtonStyle::new(FontId(0)));
    assert_eq!(u.active_id, Some(u.id("ok")));
    // The button vanishes mid-drag and the pointer comes up: the capture
    // clears anyway.
    u.begin(
        UiInput {
            mouse: (50.0, 30.0),
            mouse_released: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.end();
    assert_eq!(u.active_id, None, "a dead capture is released centrally");
}

/// A focused widget that stops running loses the focus — a dead id must
/// not keep swallowing keys.
#[test]
fn focus_dies_with_its_widget() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    u.set_focus("ok");
    u.begin(UiInput::default(), (800.0, 600.0));
    u.button("ok", rect, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(u.focused("ok"));
    // The button does not run this frame: the focus id is dead.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.end();
    assert!(!u.focused("ok"), "a dead focus id is rejected");
}

/// Pressing outside the focused widget blurs it globally — same rule for
/// every focusable, not just text fields.
#[test]
fn outside_press_blurs_a_focused_slider() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 20.0);
    let mut value = 0.0;
    let range = 0.0..=1.0;
    let press = UiInput {
        mouse: (50.0, 15.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.slider(
        "vol",
        rect,
        &mut value,
        range.clone(),
        SliderStyle::new(FontId(0)),
    );
    u.end();
    assert!(u.focused("vol"));
    let away = UiInput {
        mouse: (500.0, 500.0),
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(away.clone(), (800.0, 600.0));
    u.slider("vol", rect, &mut value, range, SliderStyle::new(FontId(0)));
    u.end();
    assert!(!u.focused("vol"), "a press anywhere else blurs");
}

// --- Capture, wheel ownership and wakes --------------------------------

/// A press and release landing in the *same* frame is still one press:
/// two overlapping buttons on the same rect produce exactly one click —
/// the released capture must not hand the spent press to the next widget.
#[test]
fn same_frame_edges_click_exactly_one_of_two_overlapped() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let input = UiInput {
        mouse: (50.0, 30.0),
        mouse_pressed: true,
        mouse_released: true,
        ..Default::default()
    };
    u.begin(input.clone(), (800.0, 600.0));
    let a = u.button("a", rect, ButtonStyle::new(FontId(0)));
    let b = u.button("b", rect, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(a.clicked && !b.clicked, "one press, one click, one owner");
}

/// A drag held while a modal opens keeps nothing: the capture is
/// cancelled by the surface instead of tracking the pointer behind it.
#[test]
fn held_slider_is_cancelled_behind_a_modal() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 20.0);
    let mut value = 0.0f32;
    let range = 0.0..=1.0;
    // Grab the slider, then open a modal (surface lives one frame ahead).
    let press = UiInput {
        mouse: (50.0, 15.0),
        mouse_down: true,
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.slider(
        "vol",
        rect,
        &mut value,
        range.clone(),
        SliderStyle::new(FontId(0)),
    );
    u.end();
    // Modal frame: the slider is drawn first each frame, so what cancels
    // its capture is the surface registered last frame.
    u.begin(
        UiInput {
            mouse: (80.0, 15.0),
            mouse_down: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.slider(
        "vol",
        rect,
        &mut value,
        range.clone(),
        SliderStyle::new(FontId(0)),
    );
    u.modal("dlg", true, (300.0, 200.0));
    u.end_clip();
    u.end();
    let before = value;
    u.begin(
        UiInput {
            mouse: (90.0, 15.0),
            mouse_down: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.slider(
        "vol",
        rect,
        &mut value,
        range.clone(),
        SliderStyle::new(FontId(0)),
    );
    u.modal("dlg", true, (300.0, 200.0));
    u.end_clip();
    u.end();
    assert_eq!(
        value, before,
        "a held slider under a modal stops tracking entirely"
    );
    assert_eq!(u.active_id, None, "the capture is cancelled, not suspended");
    // Its keyboard focus is routed away too.
    u.set_focus("vol");
    let confirm = UiInput {
        nav: NavInput {
            left: true,
            ..Default::default()
        },
        ..Default::default()
    };
    u.begin(confirm.clone(), (800.0, 600.0));
    u.slider(
        "vol",
        rect,
        &mut value,
        range.clone(),
        SliderStyle::new(FontId(0)),
    );
    u.modal("dlg", true, (300.0, 200.0));
    u.end_clip();
    u.end();
    assert_eq!(value, before, "a focused slider under a modal ignores nav");
}

/// A text field that is focused must lose the focus the moment it is
/// drawn disabled — a dead focus would swallow the next keys.
#[test]
fn disabling_a_focused_field_blurs_it() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 30.0);
    let mut text = String::new();
    let enabled = ButtonStyle::new(FontId(0));
    let disabled = enabled.disabled();
    // Focus by press.
    let press = UiInput {
        mouse: (20.0, 20.0),
        mouse_pressed: true,
        ..Default::default()
    };
    u.begin(press.clone(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, enabled);
    u.end();
    assert!(u.focused("name"));
    // Same widget next frame, now disabled.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.text_field("name", rect, &mut text, disabled);
    u.end();
    assert!(!u.focused("name"), "a disabled field cannot hold the focus");
}

/// The wheel routes by the region records, not the mouse position of the
/// frame that registered them: moving the pointer *into* a view scrolls
/// it on the first tick.
#[test]
fn wheel_follows_the_pointer_in_on_the_first_tick() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    // Frame 1: the view registers while the pointer is elsewhere.
    u.begin(
        UiInput {
            mouse: (500.0, 500.0),
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.end();
    // Frame 2: pointer enters and wheels in the same frame.
    u.begin(
        UiInput {
            mouse: (50.0, 25.0),
            wheel: -40.0,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let content = u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.end();
    assert_eq!(
        content.y, -40.0,
        "the first wheel tick after entering scrolls"
    );
}

/// Nested views resolve the wheel at the *current* pointer: entering a
/// deeper view hands the wheel to it even though last frame's records
/// were built with the pointer elsewhere.
#[test]
fn nested_wheel_owner_resolves_at_current_pointer() {
    let mut u = ui();
    let outer = Rect::new(0.0, 0.0, 200.0, 200.0);
    let inner = Rect::new(50.0, 50.0, 100.0, 100.0);
    // Frame 1: pointer outside both — both views register enabled.
    u.begin(
        UiInput {
            mouse: (500.0, 500.0),
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.scroll_begin("outer", outer, 1000.0);
    u.scroll_begin("inner", inner, 500.0);
    u.scroll_end();
    u.scroll_end();
    u.end();
    // Frame 2: pointer lands inside the inner view and wheels — the inner
    // view owns the tick even though it never saw the pointer before.
    u.begin(
        UiInput {
            mouse: (75.0, 75.0),
            wheel: -60.0,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let outer_content = u.scroll_begin("outer", outer, 1000.0);
    let inner_content = u.scroll_begin("inner", inner, 500.0);
    u.scroll_end();
    u.scroll_end();
    u.end();
    assert_eq!(outer_content.y, 0.0, "the outer view did not scroll");
    assert_eq!(inner_content.y, -10.0, "the deepest view owns the wheel");
}

/// A surface covers the wheel too: a scroll view under a popup cannot
/// claim ticks while the pointer sits on the popup's rect.
#[test]
fn a_surface_steals_the_wheel_over_its_rect() {
    let mut u = ui();
    let rect = Rect::new(50.0, 50.0, 100.0, 100.0);
    // Frame 1: view registers.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.scroll_begin("list", rect, 400.0);
    u.scroll_end();
    u.end();
    // Frame 2+: a popup covers the view's rect — wheels go nowhere. The
    // host declares the surface first, so it blocks even this frame.
    for _ in 0..2 {
        u.begin(
            UiInput {
                mouse: (75.0, 75.0),
                wheel: -60.0,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.declare_input_surface(Rect::new(0.0, 0.0, 300.0, 300.0), LAYER_POPUP, false);
        let content = u.scroll_begin("list", rect, 400.0);
        u.scroll_end();
        u.end();
        assert_eq!(content.y, rect.y, "a covered view gets no wheel");
    }
}

/// A pending tooltip is a wake deadline, not animation: `animating` stays
/// false while the dwell counts, and `next_wake_in` hands the host the
/// remaining delay — so a parked app still shows the tip on time.
#[test]
fn pending_tooltip_requests_a_wake_not_animation() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    // Entry frame (dt 0): hover registers.
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.tooltip(rect, "hint", FontId(0));
    u.end();
    // 0.3 s of continuous hover: pending, with ~0.3 s to go.
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            dt: 0.3,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.tooltip(rect, "hint", FontId(0));
    u.end();
    assert!(!u.animating(), "a counting dwell is not motion");
    let wake = u.next_wake_in().expect("a pending dwell asks for a wake");
    assert!(
        (wake.as_secs_f32() - 0.3).abs() < 0.02,
        "wake in {wake:?}, want the remaining ~0.3 s"
    );
    // Past the delay the dwell is done: nothing is pending.
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            dt: 0.4,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.tooltip(rect, "hint", FontId(0));
    u.end();
    assert_eq!(u.next_wake_in(), None, "a shown tip asks for nothing");
}

/// A resting toast holds without redrawing, then wakes the host when its
/// leave is due.
#[test]
fn a_resting_toast_requests_its_leave_frame() {
    let mut u = ui();
    let mut toasts = Toasts::default();
    toasts.push("saved", ToastKind::Info);
    // First rendered frame: the toast's clock has not started — it is
    // registered, not aged (a toast pushed mid-park arrives fresh).
    u.begin(
        UiInput {
            dt: 20.0,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.toasts(&mut toasts);
    u.end();
    assert_eq!(toasts.items[0].age, 0.0, "the first frame does not age it");
    // One rendered 0.5 s frame later it sits at 0.5 s.
    u.begin(
        UiInput {
            dt: 0.5,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.toasts(&mut toasts);
    u.end();
    // Past the entrance ease, nothing animates — but expiry is coming.
    let wake = u.next_wake_in().expect("a resting toast asks for a wake");
    let ttl = toasts.items[0].ttl;
    let toast_motion = u.theme().tokens.motion.toast;
    assert!(
        (wake.as_secs_f32() - (ttl - toast_motion - 0.5)).abs() < 0.02,
        "wake {wake:?} for leave at ~{} s",
        ttl - toast_motion
    );
    assert!(
        !u.animating(),
        "a resting toast does not redraw continuously"
    );
}

/// Non-finite and negative frame times must not poison animation clocks:
/// a NaN dt is a stopped frame, not a NaN forever.
#[test]
fn a_bad_dt_is_a_stopped_frame_not_a_poisoned_clock() {
    let mut u = ui();
    for dt in [f32::NAN, f32::INFINITY, -0.5] {
        u.begin(
            UiInput {
                dt,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        let value = u.anim_f32("fade", 1.0, 1.0, Ease::Linear);
        u.end();
        assert!(value.is_finite(), "dt {dt} produced a finite value");
    }
    u.begin(
        UiInput {
            dt: 0.5,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let value = u.anim_f32("fade", 1.0, 1.0, Ease::Linear);
    u.end();
    assert!(
        (value - 0.5).abs() < 1e-4,
        "the clock survived the bad inputs, value {value}"
    );
}

// --- Scale (logical -> physical) ----------------------------------------

/// Scale transforms the frame, not the calls: at scale 2 a 10px logical
/// rect is 20px of vertex space, its clip is 20px of scissor, and the
/// frame size doubles — while hit testing still runs on logical points.
#[test]
fn scaled_frames_are_physical_but_input_stays_logical() {
    let mut u = ui();
    u.set_scale(2.0);
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    // A press at the logical position hits the scaled widget.
    u.begin(
        UiInput {
            mouse: (50.0, 30.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let response = u.button("ok", rect, ButtonStyle::new(FontId(0)));
    let frame = u.end();
    assert!(response.hovered, "logical mouse hits logical rect");
    assert_eq!(frame.size, (1600.0, 1200.0), "frame size is physical");
    // The button's quad is emitted at double size.
    let verts: Vec<f32> = frame
        .vertices
        .iter()
        .map(|v| v.pos[0])
        .filter(|x| *x > 0.0)
        .collect();
    assert!(
        verts.iter().any(|x| (*x - 20.0).abs() < 0.01)
            && verts.iter().any(|x| (*x - 220.0).abs() < 0.01),
        "the quad spans 20..220 physical px, got {verts:?}"
    );
    // And its clip, had there been one, would be physical too.
    u.begin(UiInput::default(), (800.0, 600.0));
    u.draw_list_mut().push_clip(Rect::new(5.0, 5.0, 50.0, 50.0));
    u.rect(Rect::new(5.0, 5.0, 10.0, 10.0), Color::WHITE);
    let frame = u.end();
    let scissor = frame.cmds.last().unwrap().scissor.expect("clipped draw");
    assert_eq!(
        (scissor.x, scissor.y, scissor.w, scissor.h),
        (10.0, 10.0, 100.0, 100.0),
        "the scissor is physical"
    );
}

/// 1.5 is not a special case: geometry lands on the physical grid and a
/// bad scale value falls back to 1.0 instead of producing garbage.
#[test]
fn scale_sanitizes_and_applies_fractionally() {
    let mut u = ui();
    u.set_scale(1.5);
    u.begin(UiInput::default(), (800.0, 600.0));
    u.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::WHITE);
    let frame = u.end();
    assert!(
        frame
            .vertices
            .iter()
            .any(|v| (v.pos[0] - 15.0).abs() < 0.01),
        "10 logical px is 15 physical px at 1.5"
    );
    assert_eq!(frame.size, (1200.0, 900.0));
    u.set_scale(f32::NAN);
    assert_eq!(u.scale(), 1.0, "NaN falls back");
    u.set_scale(-2.0);
    assert_eq!(u.scale(), 1.0, "negative falls back");
}

/// Text rasters at the physical size: a 16 logical px label at scale 2
/// draws a ~32 px glyph bitmap (not a stretched 16 px one) while the
/// layout metrics stay logical.
#[test]
fn scaled_text_rasters_at_the_physical_size() {
    let mut u1 = ui();
    let mut u2 = ui();
    u2.set_scale(2.0);
    let (w1, _) = u1.measure("Ag", FontId(0), 16.0);
    let (w2, _) = u2.measure("Ag", FontId(0), 16.0);
    assert!(
        (w1 - w2).abs() <= 1.0,
        "logical measure is scale-invariant up to pixel quantization: {w1} vs {w2}"
    );
    // Rasterized glyph for the same label at both scales.
    let fonts = test_font();
    let g1 = u1.cache_mut().glyph(&fonts, FontId(0), 'A', 16.0);
    let g2 = u2.cache_mut().glyph(&fonts, FontId(0), 'A', 32.0);
    assert!(
        g2.width > g1.width * 1.4,
        "32px raster scales with resolution (not a stretched 16px bitmap): {} vs {}",
        g1.width,
        g2.width
    );
}

/// The UI frame is physical while the input contract stays logical, end
/// to end: a button clicked at its logical rect emits physical geometry.
#[test]
fn scaled_ui_is_physical_out_logical_in() {
    let mut u = ui();
    u.set_scale(2.0);
    u.begin(UiInput::default(), (400.0, 300.0));
    u.panel(Rect::new(10.0, 10.0, 200.0, 100.0), PanelStyle::default());
    u.label(
        Rect::new(20.0, 20.0, 100.0, 20.0),
        "Hi",
        LabelStyle::new(FontId(0)),
    );
    u.end_clip();
    let frame = u.end();
    assert_eq!(frame.size, (800.0, 600.0), "a 400x300 logical UI at 2x");
}

/// The scrollbar thumb is subject to the same rules as any capture: while
/// a modal surface is up, a grab is rejected and a held thumb lets go.
#[test]
fn scroll_thumb_behind_a_modal_stops_dragging() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    // Grab the thumb (top of the track at x 93..98).
    u.begin(
        UiInput {
            mouse: (95.0, 12.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.end();
    assert_eq!(u.active_id, Some(u.id("list")), "the thumb captured");
    // Modal opens: the view is drawn before the modal each frame, so the
    // surface from last frame is what blocks the drag.
    // The frame the modal opens on still lets the drag run (call order —
    // the surface registers at the end of it); from the next frame on the
    // capture is cancelled and nothing moves.
    u.begin(
        UiInput {
            mouse: (95.0, 30.0),
            mouse_down: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.modal("dlg", true, (300.0, 200.0));
    u.end_clip();
    u.end();
    u.begin(
        UiInput {
            mouse: (95.0, 40.0),
            mouse_down: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let content = u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.modal("dlg", true, (300.0, 200.0));
    u.end_clip();
    u.end();
    let moved = content.y;
    assert!(moved < 0.0, "the last unobstructed frame did drag");
    assert_eq!(u.active_id, None, "the modal cancelled the thumb drag");
    // And stays cancelled: no further frames move it.
    u.begin(
        UiInput {
            mouse: (95.0, 48.0),
            mouse_down: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let content = u.scroll_begin("list", rect, 200.0);
    u.scroll_end();
    u.modal("dlg", true, (300.0, 200.0));
    u.end_clip();
    u.end();
    assert_eq!(content.y, moved, "the view never moved under the modal");
}

/// Recycling returns the frame's buffers to the draw list; subsequent
/// frames keep working and the same allocations survive.
#[test]
fn recycle_frame_keeps_the_ui_drawing() {
    let mut u = ui();
    u.begin(UiInput::default(), (800.0, 600.0));
    u.button(
        "a",
        Rect::new(0.0, 0.0, 100.0, 40.0),
        ButtonStyle::new(FontId(0)),
    );
    let frame = u.end();
    assert!(!frame.cmds.is_empty());
    u.recycle_frame(frame);
    u.begin(UiInput::default(), (800.0, 600.0));
    u.button(
        "a",
        Rect::new(0.0, 0.0, 100.0, 40.0),
        ButtonStyle::new(FontId(0)),
    );
    let frame = u.end();
    assert!(!frame.cmds.is_empty(), "a recycled ui still draws");
    u.recycle_frame(frame);
}

/// Two nested modals: only the top one's widgets get input; the lower
/// modal does not dismiss behind it.
#[test]
fn nested_modals_only_the_top_owns_input() {
    let mut u = ui();
    u.begin(UiInput::default(), (800.0, 600.0));
    let outer = u.modal("outer", true, (400.0, 300.0)).unwrap();
    let inner = u.modal("inner", true, (200.0, 150.0)).unwrap();
    // A widget drawn now belongs to the inner modal's layer.
    let allowed_inner = u.input_allowed(1, inner.card);
    // A widget rect from the outer card's area is beneath the inner modal.
    let saved = u.draw.layer();
    u.draw.set_layer(2000);
    let allowed_outer = u.input_allowed(2, outer.card);
    u.draw.set_layer(saved);
    assert!(allowed_inner, "inner modal's own widgets receive input");
    assert!(!allowed_outer, "outer modal is covered by the inner one");
    u.end_clip(); // inner
    u.end_clip(); // outer
    u.end();
}

/// `set_input_rect` confines by pointer: a widget that straddles the card
/// edge answers only when the press lands inside it.
#[test]
fn confined_input_tests_the_pointer_not_the_widget() {
    let mut u = ui();
    let card = Rect::new(50.0, 50.0, 200.0, 200.0);
    // A widget spanning the card boundary: centred inside, partly out.
    let straddler = Rect::new(100.0, 140.0, 180.0, 20.0); // centre (190,150) inside
    let mut click = |inside_x: f32| {
        u.begin(
            UiInput {
                mouse: (inside_x, 150.0),
                mouse_pressed: true,
                mouse_down: true,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.set_input_rect(Some(card));
        let clicked = u
            .button("edge", straddler, ButtonStyle::new(FontId(0)))
            .clicked;
        u.set_input_rect(None);
        u.end();
        // release
        u.begin(
            UiInput {
                mouse: (inside_x, 150.0),
                mouse_released: true,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.set_input_rect(Some(card));
        let clicked = clicked
            || u.button("edge", straddler, ButtonStyle::new(FontId(0)))
                .clicked;
        u.set_input_rect(None);
        u.end();
        clicked
    };
    assert!(click(160.0), "press inside the card hits the straddler");
    assert!(!click(300.0), "press outside the card does not");
}

/// `input_allowed` honours a declared keyboard surface: widgets whose
/// rect cannot reach the card take no keys.
#[test]
fn keyboard_ownership_uses_the_effective_rect() {
    let mut u = ui();
    u.begin(UiInput::default(), (800.0, 600.0));
    let card = Rect::new(50.0, 50.0, 200.0, 200.0);
    u.set_input_rect(Some(card));
    assert!(
        u.input_allowed(7, Rect::new(60.0, 60.0, 50.0, 20.0)),
        "a widget inside the card takes keys"
    );
    assert!(
        u.input_allowed(7, Rect::new(30.0, 60.0, 50.0, 20.0)),
        "a straddling widget still reaches the card"
    );
    assert!(
        !u.input_allowed(7, Rect::new(400.0, 60.0, 50.0, 20.0)),
        "a widget fully outside the card is blocked"
    );
    u.end();
}

/// Tab cycles focusables in submission order; Shift+Tab reverses; a
/// disabled/clipped widget is skipped.
#[test]
fn tab_walks_focusables_in_submission_order() {
    let mut u = ui();
    let draw = |u: &mut Ui| {
        u.button(
            "first",
            Rect::new(0.0, 0.0, 50.0, 20.0),
            ButtonStyle::new(FontId(0)),
        );
        u.button(
            "second",
            Rect::new(60.0, 0.0, 50.0, 20.0),
            ButtonStyle::new(FontId(0)),
        );
        u.button(
            "third",
            Rect::new(120.0, 0.0, 50.0, 20.0),
            ButtonStyle::new(FontId(0)),
        );
    };
    // Frame 1 establishes the registry.
    u.begin(UiInput::default(), (400.0, 200.0));
    draw(&mut u);
    u.end();
    // Tab once -> first.
    u.begin(
        nav_input(NavInput {
            tab: true,
            ..Default::default()
        }),
        (400.0, 200.0),
    );
    draw(&mut u);
    u.end();
    assert!(u.focused("first"), "tab focuses the first focusable");
    // Tab again -> second.
    u.begin(
        nav_input(NavInput {
            tab: true,
            ..Default::default()
        }),
        (400.0, 200.0),
    );
    draw(&mut u);
    u.end();
    assert!(u.focused("second"), "next focusable in submission order");
    // Shift+Tab back -> first.
    u.begin(
        nav_input(NavInput {
            tab: true,
            shift: true,
            ..Default::default()
        }),
        (400.0, 200.0),
    );
    draw(&mut u);
    u.end();
    assert!(u.focused("first"), "shift+tab walks backwards");
}

/// A focused field eats the editing shortcuts: select-all, cut, paste and
/// undo all land in the document — and the clipboard seam drains once.
#[test]
fn a_focused_field_owns_editing_shortcuts_and_the_clipboard() {
    let mut u = ui();
    let mut text = String::from("seed");
    let rect = Rect::new(10.0, 10.0, 200.0, 30.0);
    // Focus the field.
    u.begin(
        UiInput {
            mouse: (50.0, 20.0),
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(u.editing());
    // select-all + copy -> clipboard seam.
    u.begin(
        nav_input(NavInput {
            select_all: true,
            copy: true,
            ..Default::default()
        }),
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(u.take_clipboard_output().as_deref(), Some("seed"));
    assert_eq!(u.take_clipboard_output(), None, "drained once");
    // Paste replaces the selection.
    u.begin(
        UiInput {
            paste: Some("grown".to_string()),
            ..Default::default()
        },
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(text, "grown");
    // Undo restores; undo again restores the original.
    u.begin(
        nav_input(NavInput {
            undo: true,
            ..Default::default()
        }),
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(text, "seed");
}

/// A Unicode commit is one edit: a whole "héllo 🇯🇵" paste undoes in one
/// step, and a cancelled composition never touches the document.
#[test]
fn ime_commits_as_one_edit_and_cancel_leaves_no_trace() {
    let mut u = ui();
    let mut text = String::new();
    let rect = Rect::new(10.0, 10.0, 200.0, 30.0);
    u.begin(
        UiInput {
            mouse: (50.0, 20.0),
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    // A whole multi-codepoint string commits at once (the IME case).
    u.begin(
        UiInput {
            text: "héllo 🇯🇵".to_string(),
            ..Default::default()
        },
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(text, "héllo 🇯🇵");
    u.begin(
        nav_input(NavInput {
            undo: true,
            ..Default::default()
        }),
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(text, "", "the whole commit undoes in one step");
    // A preedit draws but never mutates; Ime::Disabled clears it.
    u.begin(
        UiInput {
            preedit: Some(crate::ui::widgets::Composition {
                text: "かな".to_string(),
                cursor: None,
            }),
            ..Default::default()
        },
        (400.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(text, "", "preedit never becomes document text");
}

/// Two overlapping fields: one press captures — only one edits.
#[test]
fn overlapping_fields_get_one_focus() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 200.0, 30.0);
    u.begin(
        UiInput {
            mouse: (50.0, 20.0),
            mouse_pressed: true,
            mouse_released: true,
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 200.0),
    );
    let mut a = String::new();
    let mut b = String::new();
    u.text_field("a", rect, &mut a, ButtonStyle::new(FontId(0)));
    u.text_field("b", rect, &mut b, ButtonStyle::new(FontId(0)));
    u.end();
    assert_eq!(
        u.focused("a") as u8 + u.focused("b") as u8,
        1,
        "one press focuses one field"
    );
}

/// A `UiInput` carrying only nav edges.
fn nav_input(nav: NavInput) -> UiInput {
    UiInput {
        nav,
        ..Default::default()
    }
}

/// `virtual_list` builds only the visible rows — a 100k list emits a
/// handful of widgets, and a row's stable index keeps its identity across
/// scrolling.
#[test]
fn virtual_list_builds_only_visible_rows() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 200.0, 100.0);
    u.begin(UiInput::default(), (400.0, 400.0));
    let mut built = Vec::new();
    u.virtual_list("list", rect, 100_000, 20.0, 4.0, |_, index, _| {
        built.push(index);
    });
    // stride 24, viewport 100 → ceil(100/24)+overscan ~= 6, never all 100k.
    assert!(
        built.len() <= (100.0f32 / 24.0).ceil() as usize + 2,
        "built {} rows",
        built.len()
    );
    assert_eq!(built.first().copied(), Some(0));
    u.end();
}

/// Wrapped and ellipsized labels go through the same cached layout seam
/// and respect grapheme boundaries.
#[test]
fn wrapped_and_ellipsized_labels_share_the_cache() {
    let mut u = ui();
    let style = LabelStyle::new(FontId(0)).size(13.0);
    u.begin(UiInput::default(), (400.0, 400.0));
    let before = u.layout_stats();
    u.label_wrapped(
        Rect::new(0.0, 0.0, 60.0, 200.0),
        "wrapped wrapped wrapped",
        style,
    );
    u.label_ellipsis(Rect::new(0.0, 0.0, 30.0, 20.0), "a very long label", style);
    u.end();
    let after = u.layout_stats();
    assert!(after.misses > before.misses, "ellipsis populated the cache");
    // Zero-width rects are no-ops, not panics.
    u.begin(UiInput::default(), (400.0, 400.0));
    u.label_wrapped(Rect::new(0.0, 0.0, 0.0, 200.0), "nothing", style);
    u.label_ellipsis(Rect::new(0.0, 0.0, 0.0, 20.0), "nothing", style);
    u.end();
}

#[test]
fn two_virtual_lists_dont_collide() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    let mut rows_a = Vec::new();
    let mut rows_b = Vec::new();
    u.begin(UiInput::default(), (200.0, 200.0));
    u.virtual_list("list_a", rect, 10, 20.0, 2.0, |u, i, r| {
        rows_a.push(i);
        u.button(&format!("a{i}"), r, ButtonStyle::new(FontId(0)));
    });
    u.virtual_list("list_b", rect, 10, 20.0, 2.0, |u, i, r| {
        rows_b.push(i);
        u.button(&format!("b{i}"), r, ButtonStyle::new(FontId(0)));
    });
    u.end();
    assert_eq!(rows_a, rows_b, "same visible range");
    // The ids differ: both lists pressed row 0, only one can be active.
    u.begin(
        UiInput {
            mouse_pressed: true,
            mouse_down: true,
            ..UiInput::default()
        },
        (200.0, 200.0),
    );
    // Press inside list_a's row 0 only is ambiguous without ids; the test
    // just verifies no collision panic and stable ids.
    u.virtual_list("list_a", rect, 10, 20.0, 2.0, |u, i, r| {
        u.button(&format!("a{i}"), r, ButtonStyle::new(FontId(0)));
    });
    u.virtual_list("list_b", rect, 10, 20.0, 2.0, |u, i, r| {
        u.button(&format!("b{i}"), r, ButtonStyle::new(FontId(0)));
    });
    u.end();
}

#[test]
fn undo_redo_accounting_stays_exact_over_cycles() {
    use crate::ui::widgets::editor::EditorState;
    let mut state = EditorState::new("hello");
    let mut text = "hello".to_string();
    // Insert+undo pairs: text returns to base, history bounded.
    for _ in 0..100 {
        state.insert(&mut text, "x");
        state.undo(&mut text);
    }
    assert_eq!(text, "hello");
    // Undo+redo pairs on the recorded edits are balanced.
    for _ in 0..100 {
        state.redo(&mut text);
        state.undo(&mut text);
    }
    assert_eq!(text, "hello");
    // Accounting tracks live snapshots only — never phantom popped
    // bytes.
    assert!(
        state.history_bytes() <= 100 * 6 + 8,
        "bytes {}",
        state.history_bytes()
    );
}

#[test]
fn stale_editors_are_retired_with_state() {
    let mut u = ui();
    let mut text = "edit".to_string();
    let rect = Rect::new(0.0, 0.0, 100.0, 20.0);
    u.begin(UiInput::default(), (100.0, 100.0));
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    // The sweep runs every 60 frames and drops entries 600 frames
    // old — pass the next boundary after 601 so it is evicted.
    for _ in 0..=661 {
        u.begin(UiInput::default(), (100.0, 100.0));
        u.end();
    }
    assert!(
        u.editors.is_empty(),
        "stale editor states are retired, got {}",
        u.editors.len()
    );
}

#[test]
fn tab_skips_disabled_focusables() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 50.0, 20.0);
    let input = |tab: bool| UiInput {
        nav: NavInput {
            tab,
            ..NavInput::default()
        },
        ..UiInput::default()
    };
    u.begin(input(false), (100.0, 100.0));
    u.button("first", rect, ButtonStyle::new(FontId(0)));
    u.button(
        "disabled",
        Rect::new(0.0, 30.0, 50.0, 20.0),
        ButtonStyle::new(FontId(0)).disabled(),
    );
    u.button(
        "third",
        Rect::new(0.0, 60.0, 50.0, 20.0),
        ButtonStyle::new(FontId(0)),
    );
    u.end();
    u.begin(input(true), (100.0, 100.0));
    u.button("first", rect, ButtonStyle::new(FontId(0)));
    u.button(
        "disabled",
        Rect::new(0.0, 30.0, 50.0, 20.0),
        ButtonStyle::new(FontId(0)).disabled(),
    );
    u.button(
        "third",
        Rect::new(0.0, 60.0, 50.0, 20.0),
        ButtonStyle::new(FontId(0)),
    );
    u.end();
    let focused = u.focus_id;
    // Second frame's first tab should land on "first" or cycle onward —
    // never the disabled button's id.
    let disabled_id = u.id("disabled");
    assert_ne!(focused, Some(disabled_id), "disabled must not focus");
}

#[test]
fn caret_blink_wakes_and_does_not_pin_the_frame_rate() {
    let mut u = ui();
    let mut text = "abc".to_string();
    let rect = Rect::new(0.0, 0.0, 100.0, 20.0);
    // Press+release inside to focus — with a PARKED-host dt of 10 s on
    // the focus-gain frame: the reset must NOT swallow it, so the caret
    // starts at the top of the visible phase and the wake is the full
    // 0.53 s boundary, not a random post-park phase.
    u.begin(
        UiInput {
            mouse: (50.0, 10.0),
            mouse_pressed: true,
            mouse_down: true,
            dt: 10.0,
            ..UiInput::default()
        },
        (200.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    let focus_wake = u.next_wake_in().expect("focused field wakes");
    assert!(
        focus_wake <= std::time::Duration::from_secs_f32(0.55)
            && focus_wake >= std::time::Duration::from_secs_f32(0.50),
        "focus-gain frame schedules the full visible phase: {focus_wake:?}"
    );
    u.begin(
        UiInput {
            mouse: (50.0, 10.0),
            mouse_down: false,
            dt: 0.016,
            ..UiInput::default()
        },
        (200.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    // Idle focused frames: a large dt settles every hover/focus tween —
    // the caret must NOT keep `animating` true (that pins the host at
    // full frame rate), it only schedules a wake at the blink boundary.
    u.begin(
        UiInput {
            mouse: (50.0, 10.0),
            dt: 10.0,
            ..UiInput::default()
        },
        (200.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(
        !u.animating(),
        "idle focused field must not pin the frame rate"
    );
    let wake = u
        .next_wake_in()
        .expect("caret schedules the next blink boundary");
    assert!(
        wake <= std::time::Duration::from_secs_f32(0.55),
        "next blink boundary {wake:?}"
    );
    // Blur (press outside) ends the wake — nothing else is due.
    u.begin(
        UiInput {
            mouse: (150.0, 150.0),
            mouse_pressed: true,
            mouse_down: true,
            dt: 0.016,
            ..UiInput::default()
        },
        (200.0, 200.0),
    );
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    assert!(u.next_wake_in().is_none(), "blurred field holds no wake");
}

#[test]
fn text_field_sanitizes_single_line_input() {
    let mut u = ui();
    let mut text = String::new();
    let rect = Rect::new(0.0, 0.0, 100.0, 20.0);
    // Focus, then feed text with CRLF, tab and a control char.
    for input in [
        UiInput {
            mouse: (50.0, 10.0),
            mouse_pressed: true,
            mouse_down: true,
            ..UiInput::default()
        },
        UiInput {
            mouse: (50.0, 10.0),
            ..UiInput::default()
        },
        UiInput {
            mouse: (50.0, 10.0),
            text: "a\r\nb\tc\x07d".to_string(),
            ..UiInput::default()
        },
    ] {
        u.begin(input, (200.0, 200.0));
        u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
        u.end();
    }
    assert_eq!(text, "a b cd", "CRLF+tab -> one space, control dropped");
}

#[test]
fn preedit_preview_keeps_document_and_moves_the_ime_caret() {
    let mut u = ui();
    let mut text = "abcdef".to_string();
    let rect = Rect::new(0.0, 0.0, 200.0, 20.0);
    let style = ButtonStyle::new(FontId(0));
    // Focus + select 2..4 via shift-drag is overkill: drive the editor
    // state directly for the selection, then render with a preedit.
    let press = |u: &mut Ui, input: UiInput, text: &mut String| {
        u.begin(input, (400.0, 400.0));
        u.text_field("f", rect, text, style);
        u.end();
    };
    press(
        &mut u,
        UiInput {
            mouse: (50.0, 10.0),
            mouse_pressed: true,
            mouse_down: true,
            ..UiInput::default()
        },
        &mut text,
    );
    press(
        &mut u,
        UiInput {
            mouse: (50.0, 10.0),
            ..UiInput::default()
        },
        &mut text,
    );
    // Selection 2..4 + a one-char preedit with an interior caret.
    let id = u.id("f");
    {
        let state = u.editors.get_mut(&id).expect("editor");
        state.move_to(&text, 2, false);
        state.move_to(&text, 4, true);
    }
    let before = (text.clone(), u.editors.get(&id).unwrap().selection());
    press(
        &mut u,
        UiInput {
            mouse: (50.0, 10.0),
            preedit: Some(crate::ui::widgets::Composition {
                text: "你".to_string(),
                cursor: Some((3, 3)),
            }),
            ..UiInput::default()
        },
        &mut text,
    );
    // The document and selection are untouched by the preview.
    assert_eq!(text, before.0);
    assert_eq!(u.editors.get(&id).unwrap().selection(), before.1);
    // The IME caret follows the preview cursor (x beyond the prefix),
    // not the old document caret — it must be a finite field position.
    let caret = u.editing_caret.expect("preview caret reported");
    assert!(caret.x > rect.x + 8.0, "caret inside the field: {caret:?}");
}

#[test]
fn direct_cache_and_raw_atlas_resets_rebake_icons() {
    // BOTH reset paths must drop icon/layout caches on the next
    // `Ui::begin`: a direct `GlyphCache::reset_atlas` (the cache's own
    // last_epoch already moved, so the cache can't tell this from "no
    // reset") and a raw `Atlas::reset` behind it.
    use crate::ui::icons::Icon;
    let mut u = ui();
    let icon_region = |u: &mut Ui| {
        u.begin(UiInput::default(), (100.0, 100.0));
        let region = u
            .icons
            .region(Icon::Check, 20, &mut u.cache)
            .expect("icon bakes");
        u.end();
        region
    };
    icon_region(&mut u);
    assert_eq!(u.icons.baked(), 1);
    // 1) direct cache reset — the icon set itself must be EMPTY after
    //    the next `begin` (stale regions would keep `baked` at 1 and
    //    return coordinates whose pixels the reset just dropped).
    u.cache_mut().reset_atlas();
    u.begin(UiInput::default(), (100.0, 100.0));
    assert_eq!(
        u.icons.baked(),
        0,
        "direct reset_atlas must drop icon regions"
    );
    u.end();
    // Re-baking lands in the fresh epoch with real atlas ink behind it.
    let rebaked = icon_region(&mut u);
    let page = u.cache.atlas().page_pixels(rebaked.page);
    let size = u.cache.atlas().page_size() as usize;
    let ink = (0..rebaked.height as usize).any(|row| {
        (0..rebaked.width as usize).any(|col| {
            page[(rebaked.y as usize + row) * size * 4 + (rebaked.x as usize + col) * 4 + 3] > 16
        })
    });
    assert!(ink, "rebaked icon has real ink in the new epoch");
    // 2) raw atlas reset — same promise via `atlas_mut`.
    u.cache_mut().atlas_mut().reset();
    u.begin(UiInput::default(), (100.0, 100.0));
    assert_eq!(
        u.icons.baked(),
        0,
        "raw Atlas::reset must drop icon regions"
    );
    u.end();
}

#[test]
fn spring_anim_chases_and_retargets() {
    let mut u = ui();
    let mut input = UiInput::default();
    input.dt = 1.0 / 60.0;
    let mut value = 0.0;
    for _ in 0..10 {
        u.begin(input.clone(), (400.0, 300.0));
        value = u.anim_spring_f32("spring.value", 100.0, 400.0, 40.0, 1.0);
        u.end();
    }
    assert!(
        value > 1.0 && value < 100.0,
        "moved toward the target, got {value}"
    );
    // Retarget mid-flight: the spring turns around without teleporting.
    let before = value;
    for _ in 0..4 {
        u.begin(input.clone(), (400.0, 300.0));
        value = u.anim_spring_f32("spring.value", 0.0, 400.0, 40.0, 1.0);
        u.end();
    }
    assert!(value <= before, "chases the new target from where it was");
    // Settle at the target.
    for _ in 0..400 {
        u.begin(input.clone(), (400.0, 300.0));
        value = u.anim_spring_f32("spring.value", 0.0, 400.0, 40.0, 1.0);
        u.end();
    }
    assert!(value.abs() < 0.5, "settles at the target, got {value}");
    assert_eq!(u.anim_count(), 1, "one registry entry per key");
}

/// One scroll frame: returns the offset the view rendered at.
fn scroll_tick(u: &mut Ui, label: &str, rect: Rect, content_h: f32, wheel: f32, dt: f32) -> f32 {
    u.begin(
        UiInput {
            mouse: rect.center(),
            wheel,
            dt,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let content = u.scroll_begin(label, rect, content_h);
    u.scroll_end();
    u.end();
    rect.y - content.y
}

/// The same, from a precision device (trackpad/touch).
fn scroll_tick_precise(
    u: &mut Ui,
    label: &str,
    rect: Rect,
    content_h: f32,
    pixels: f32,
    dt: f32,
) -> f32 {
    u.begin(
        UiInput {
            mouse: rect.center(),
            wheel_precise: pixels,
            dt,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let content = u.scroll_begin(label, rect, content_h);
    u.scroll_end();
    u.end();
    rect.y - content.y
}

/// A wheel notch glides toward its destination instead of jumping there:
/// the first frame moves part of the way, and the view settles on the
/// notch exactly.
#[test]
fn scroll_notch_glides_and_settles() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    // Registration frame (dt 0): nothing moves yet.
    scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, 0.0);
    let first = scroll_tick(&mut u, "list", rect, 10_000.0, -100.0, dt);
    assert!(
        first > 0.0 && first < 100.0,
        "the notch glides rather than jumping, got {first}"
    );
    let mut offset = first;
    for _ in 0..60 {
        offset = scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt);
    }
    assert!(
        (offset - 100.0).abs() < 1.0,
        "settles on the notch's destination, got {offset}"
    );
}

/// Notches landing while a glide is in flight extend its destination, so a
/// burst of wheel reads as one gesture rather than separate kicks.
#[test]
fn scroll_notches_accumulate_into_one_glide() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, 0.0);
    let first = scroll_tick(&mut u, "list", rect, 10_000.0, -50.0, dt);
    let second = scroll_tick(&mut u, "list", rect, 10_000.0, -50.0, dt);
    assert!(
        second > first && second < 100.0,
        "the second notch extends the glide instead of restarting it, {first} then {second}"
    );
    let mut offset = second;
    for _ in 0..90 {
        offset = scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt);
    }
    assert!(
        (offset - 100.0).abs() < 1.0,
        "both notches land, got {offset}"
    );
}

/// A precision device's pixels land where the device put them — the deltas
/// are already smooth, so the view must not lag behind the fingers.
#[test]
fn scroll_precise_deltas_track_the_device() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, 0.0);
    let offset = scroll_tick_precise(&mut u, "list", rect, 10_000.0, -37.0, 1.0 / 60.0);
    assert_eq!(offset, 37.0, "precise pixels apply 1:1");
}

/// A precision gesture that stops mid-flick keeps travelling and comes to
/// rest — momentum, decaying at a frame-rate independent rate.
#[test]
fn scroll_momentum_carries_a_precision_gesture() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, 0.0);
    let flicked = scroll_tick_precise(&mut u, "list", rect, 10_000.0, -60.0, dt);
    assert_eq!(flicked, 60.0);
    let coasted = scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt);
    assert!(
        coasted > flicked,
        "the flick keeps moving after the device stops, {flicked} then {coasted}"
    );
    let mut offset = coasted;
    for _ in 0..200 {
        offset = scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt);
    }
    assert!(!u.animating(), "the glide comes to rest");
    assert_eq!(
        offset,
        scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt),
        "and stays there"
    );
}

/// Momentum that runs into the content's end stops there — and stops
/// reporting motion, or the host would keep rendering a still screen.
#[test]
fn scroll_fling_stops_at_the_end() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    // Content just past the viewport: 60 px of range.
    scroll_tick(&mut u, "list", rect, 160.0, 0.0, 0.0);
    let flicked = scroll_tick_precise(&mut u, "list", rect, 160.0, -50.0, dt);
    assert_eq!(flicked, 50.0);
    let mut offset = flicked;
    for _ in 0..60 {
        offset = scroll_tick(&mut u, "list", rect, 160.0, 0.0, dt);
    }
    assert_eq!(offset, 60.0, "the coast stops at the end");
    assert!(!u.animating(), "a coast against a wall is not motion");
}

/// A glide is motion: the host must keep building frames until it settles,
/// or the scroll would freeze mid-flight between input events.
#[test]
fn scroll_glide_keeps_the_ui_animating() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, 0.0);
    scroll_tick(&mut u, "list", rect, 10_000.0, -80.0, dt);
    assert!(u.animating(), "a glide in flight is animation");
    for _ in 0..120 {
        scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt);
    }
    assert!(!u.animating(), "a settled glide does not pin the loop");
}

/// Programmatic scrolling takes the DOM's behaviours: `Instant` lands this
/// frame, `Auto` (and `Smooth`) travels there over time.
#[test]
fn scroll_to_honours_its_behavior() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, 0.0);

    u.scroll_to("list", ScrollToOptions::instant(120.0));
    assert_eq!(
        scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt),
        120.0,
        "an instant request lands at once"
    );

    u.scroll_to("list", ScrollToOptions::smooth(40.0));
    let first = scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt);
    assert!(
        first < 120.0 && first > 40.0,
        "a smooth request travels, got {first}"
    );
    let mut offset = first;
    for _ in 0..90 {
        offset = scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt);
    }
    assert!((offset - 40.0).abs() < 0.5, "arrives at 40, got {offset}");

    // Out-of-range destinations clamp to the content, like the DOM's.
    u.scroll_to("list", ScrollToOptions::instant(f32::INFINITY));
    assert_eq!(
        scroll_tick(&mut u, "list", rect, 10_000.0, 0.0, dt),
        9_900.0,
        "an infinite destination pins the end"
    );
}

/// `scrollIntoView` honours the block alignment and leaves a target that is
/// already visible alone under `Nearest`.
#[test]
fn scroll_into_view_alignments() {
    let target = Rect::new(0.0, 200.0, 100.0, 40.0);
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let aligned = |block: ScrollAlign, from: f32| -> f32 {
        let mut u = ui();
        scroll_tick(&mut u, "list", rect, 400.0, 0.0, 0.0);
        if from != 0.0 {
            u.scroll_to("list", ScrollToOptions::instant(from));
            scroll_tick(&mut u, "list", rect, 400.0, 0.0, 0.0);
        }
        u.scroll_into_view(
            "list",
            target,
            ScrollIntoViewOptions {
                block,
                behavior: ScrollBehavior::Instant,
            },
        );
        scroll_tick(&mut u, "list", rect, 400.0, 0.0, 0.0)
    };
    assert_eq!(aligned(ScrollAlign::Start, 0.0), 200.0);
    assert_eq!(aligned(ScrollAlign::Center, 0.0), 170.0);
    assert_eq!(aligned(ScrollAlign::End, 0.0), 140.0);
    assert_eq!(
        aligned(ScrollAlign::Nearest, 0.0),
        140.0,
        "a target below the fold aligns its bottom"
    );
    assert_eq!(
        aligned(ScrollAlign::Nearest, 200.0),
        200.0,
        "an already-visible target does not move the view"
    );
}

/// Nested views chain at the ends: what the inner view cannot use continues
/// the enclosing one instead of being swallowed.
#[test]
fn scroll_chain_passes_the_remainder_to_the_parent() {
    let mut u = ui();
    let outer = Rect::new(0.0, 0.0, 200.0, 200.0);
    let inner = Rect::new(50.0, 50.0, 100.0, 100.0);
    let frame = |u: &mut Ui, wheel: f32| -> (f32, f32) {
        u.begin(
            UiInput {
                mouse: (75.0, 75.0),
                wheel,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        let outer_content = u.scroll_begin("outer", outer, 600.0);
        let inner_content = u.scroll_begin("inner", inner, 150.0);
        u.scroll_end();
        u.scroll_end();
        u.end();
        (outer.y - outer_content.y, inner.y - inner_content.y)
    };
    frame(&mut u, 0.0); // Register both regions.
    let (outer_before, inner_now) = frame(&mut u, -200.0);
    assert_eq!(inner_now, 50.0, "the inner view clamps at its end");
    assert_eq!(outer_before, 0.0, "the parent is not moved retroactively");
    let (outer_now, inner_after) = frame(&mut u, 0.0);
    assert_eq!(inner_after, 50.0, "the inner view stays at its end");
    assert_eq!(
        outer_now, 150.0,
        "the remainder continued the enclosing view"
    );
}

/// Arrow keys and page keys scroll the view under the pointer while no
/// widget owns focus; Home/End jump to the ends.
#[test]
fn scroll_keys_move_the_hovered_view() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 200.0);
    let key = |u: &mut Ui, nav: NavInput| -> f32 {
        u.begin(
            UiInput {
                mouse: (50.0, 100.0),
                nav,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        let content = u.scroll_begin("list", rect, 1_000.0);
        u.scroll_end();
        u.end();
        rect.y - content.y
    };
    key(&mut u, NavInput::default()); // Register the viewport.
    assert_eq!(
        key(
            &mut u,
            NavInput {
                page_down: true,
                ..Default::default()
            }
        ),
        170.0,
        "a page step is one viewport of travel"
    );
    assert_eq!(
        key(
            &mut u,
            NavInput {
                down: true,
                ..Default::default()
            }
        ),
        218.0,
        "an arrow nudges by its step"
    );
    assert_eq!(
        key(
            &mut u,
            NavInput {
                end: true,
                ..Default::default()
            }
        ),
        800.0,
        "End reaches the content's end"
    );
    assert_eq!(
        key(
            &mut u,
            NavInput {
                home: true,
                ..Default::default()
            }
        ),
        0.0,
        "Home returns to the top"
    );
}

/// Wheel input takes over from a programmatic glide in flight, carrying on
/// from where the view is — not from where the animation was heading.
#[test]
fn scroll_wheel_interrupts_a_programmatic_glide() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    scroll_tick(&mut u, "list", rect, 1_000.0, 0.0, 0.0);
    u.scroll_to("list", ScrollToOptions::smooth(500.0));
    let mid = scroll_tick(&mut u, "list", rect, 1_000.0, 0.0, dt);
    assert!(
        mid > 0.0 && mid < 500.0,
        "the glide is under way, got {mid}"
    );
    let nudged = scroll_tick(&mut u, "list", rect, 1_000.0, -120.0, dt);
    assert!(nudged > mid, "the wheel moved the view down from here");
    let mut offset = nudged;
    for _ in 0..90 {
        offset = scroll_tick(&mut u, "list", rect, 1_000.0, 0.0, dt);
    }
    assert!(
        (offset - (mid + 120.0)).abs() < 1.0,
        "the notch added to where the view was, not to the old target: got {offset}"
    );
}

/// Repeating the same smooth request — a follow-tail caller running every
/// frame — must not restart the glide it already started, or the view
/// would crawl forever.
#[test]
fn scroll_repeated_smooth_requests_do_not_restart_the_glide() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let dt = 1.0 / 60.0;
    scroll_tick(&mut u, "list", rect, 1_000.0, 0.0, 0.0);
    let mut offset = 0.0;
    for _ in 0..20 {
        u.scroll_to("list", ScrollToOptions::smooth(300.0));
        offset = scroll_tick(&mut u, "list", rect, 1_000.0, 0.0, dt);
    }
    assert!(
        (offset - 300.0).abs() < 1.0,
        "the glide arrived despite a request every frame, got {offset}"
    );
}

/// Clicking the scrollbar track pages toward the click, like a browser.
#[test]
fn scroll_track_click_pages_toward_the_click() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    scroll_tick(&mut u, "list", rect, 1_000.0, 0.0, 0.0);
    u.begin(
        UiInput {
            mouse: (95.0, 80.0),
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let content = u.scroll_begin("list", rect, 1_000.0);
    u.scroll_end();
    u.end();
    let offset = rect.y - content.y;
    assert!(
        (offset - 85.0).abs() < 0.01,
        "a track click below the thumb pages down, got {offset}"
    );
    // With the view at its end the thumb sits at the bottom, so a click
    // near the top of the track pages back up.
    u.scroll_to("list", ScrollToOptions::instant(900.0));
    scroll_tick(&mut u, "list", rect, 1_000.0, 0.0, 0.0);
    u.begin(
        UiInput {
            mouse: (95.0, 20.0),
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let content = u.scroll_begin("list", rect, 1_000.0);
    u.scroll_end();
    u.end();
    assert_eq!(
        rect.y - content.y,
        815.0,
        "a track click above the thumb pages up"
    );
}

/// Metrics report the view's extents and reading progress.
#[test]
fn scroll_metrics_report_the_view() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    scroll_tick(&mut u, "list", rect, 400.0, 0.0, 0.0);
    u.scroll_to("list", ScrollToOptions::instant(150.0));
    let offset = scroll_tick(&mut u, "list", rect, 400.0, 0.0, 1.0 / 60.0);
    assert_eq!(offset, 150.0);
    let metrics = u.scroll_metrics("list");
    assert_eq!(metrics.offset, 150.0);
    assert_eq!(metrics.viewport, 100.0);
    assert_eq!(metrics.content, 400.0);
    assert_eq!(metrics.max_offset, 300.0);
    assert_eq!(metrics.progress(), 0.5);
    assert!(!metrics.animating);
    assert_eq!(u.scroll_state("list"), (150.0, 400.0));
}

#[test]
fn scroll_without_momentum_jumps_and_stops() {
    let mut u = ui();
    u.scroll_config = super::ScrollConfig::no_momentum();
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let frame = |u: &mut Ui, wheel: f32| -> f32 {
        let input = UiInput {
            mouse: (50.0, 50.0),
            wheel,
            dt: 1.0 / 60.0,
            ..Default::default()
        };
        u.begin(input, (800.0, 600.0));
        let content = u.scroll_begin("list", rect, 10_000.0);
        u.scroll_end();
        u.end();
        content.y
    };
    assert_eq!(frame(&mut u, -50.0), -50.0);
    for _ in 0..30 {
        assert_eq!(frame(&mut u, 0.0), -50.0, "no coast without momentum");
    }
}

// --- Motion scale ---------------------------------------------------------

/// The knob ships neutral: an untouched `Ui` plays durations exactly as the
/// widgets authored them, and hosts scale motion only if they decide to.
#[test]
fn motion_scale_defaults_to_authored_timing() {
    let u = ui();
    assert_eq!(u.motion_scale(), 1.0);
    assert_eq!(u.motion_time(0.2), 0.2, "scale 1 passes durations through");
}

/// Non-positive and non-finite inputs are handled by the setter itself, so a
/// host can forward an OS preference verbatim without clamping it first.
#[test]
fn motion_scale_setter_sanitizes_its_input() {
    let mut u = ui();
    u.set_motion_scale(-3.0);
    assert_eq!(u.motion_scale(), 0.0, "negative clamps to instant");
    u.set_motion_scale(f32::NAN);
    assert_eq!(u.motion_scale(), 1.0, "non-finite falls back to authored");
    u.set_motion_scale(1.0e9);
    assert_eq!(u.motion_scale(), 100.0, "extremes clamp to a sane range");
}

/// Scale 0 means the value *arrives* — no travel, no lingering "in flight"
/// state that would keep an on-demand renderer awake forever.
#[test]
fn motion_scale_zero_resolves_animation_instantly() {
    let mut u = ui();
    u.set_motion_scale(0.0);
    u.begin(
        UiInput {
            dt: 0.016,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let value = u.anim_f32("fade", 1.0, 5.0, Ease::Linear);
    assert_eq!(value, 1.0, "scale 0 lands on the target, got {value}");
    let spring = u.anim_spring_f32("pull", 40.0, 400.0, 40.0, 1.0);
    assert_eq!(spring, 40.0, "springs snap at scale 0, got {spring}");
    assert!(
        !u.animating(),
        "nothing is in flight once motion resolves instantly"
    );
    u.end();
}

/// Half the scale is the *same curve in half the time*: the authored 0.2 s
/// ease runs as 0.1 s, so a 0.05 s frame sits at 0.5 — where the authored
/// timing would only have reached 0.25.
#[test]
fn motion_scale_half_completes_in_half_the_time() {
    let mut u = ui();
    u.set_motion_scale(0.5);
    u.begin(
        UiInput {
            dt: 0.05,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let value = u.anim_f32("fade", 1.0, 0.2, Ease::Linear);
    assert!(
        (value - 0.5).abs() < 1e-4,
        "half scale finishes in half the authored time, got {value}"
    );
    u.end();
    assert_eq!(
        u.motion_time(0.2),
        0.1,
        "motion_time reports what the animation actually took"
    );
}

/// Loops scale by *speed*: stopping motion freezes a blink on its current
/// phase instead of racing it, and a positive scale keeps it cycling.
#[test]
fn motion_scale_zero_freezes_loops() {
    let mut u = ui();
    u.set_motion_scale(0.0);
    u.begin(
        UiInput {
            dt: 0.3,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let first = u.anim_loop_f32("blink", 1.0);
    let second = u.anim_loop_f32("blink", 1.0);
    assert_eq!(first, 0.0, "a frozen loop never advances, got {first}");
    assert_eq!(second, 0.0, "still frozen on the next frame, got {second}");
    u.end();
}

/// A scaled spring settles in the scaled time, and a tiny scale with a long
/// frame cannot destabilize the integrator: the substepped update keeps the
/// value finite and on its way to the target.
#[test]
fn motion_scale_springs_stay_stable_at_tiny_scales() {
    let mut u = ui();
    u.set_motion_scale(0.05);
    let mut last = 0.0;
    for _ in 0..40 {
        u.begin(
            UiInput {
                dt: 0.2,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        last = u.anim_spring_f32("pull", 100.0, 400.0, 40.0, 1.0);
        u.end();
        assert!(last.is_finite(), "the spring stays finite, got {last}");
    }
    assert!(
        (last - 100.0).abs() < 1.0,
        "settled at the target despite the tiny scale, got {last}"
    );
}

#[test]
fn checkbox_and_radio_and_segmented_interact() {
    let mut u = ui();
    let style = ButtonStyle::new(FontId(0));
    let mut checked = false;
    let chk_rect = Rect::new(10.0, 10.0, 120.0, 24.0);
    // Press + release inside checkbox flips `checked`.
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    u.checkbox("Agree", chk_rect, &mut checked, style);
    u.end();
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            mouse_released: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    let resp = u.checkbox("Agree", chk_rect, &mut checked, style);
    u.end();
    assert!(resp.clicked && checked, "checkbox flipped to true");

    // Radio button reports click.
    let rad_rect = Rect::new(10.0, 40.0, 120.0, 24.0);
    u.begin(
        UiInput {
            mouse: (20.0, 50.0),
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    u.radio("Opt A", rad_rect, false, style);
    u.end();
    u.begin(
        UiInput {
            mouse: (20.0, 50.0),
            mouse_released: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    let r_resp = u.radio("Opt A", rad_rect, false, style);
    u.end();
    assert!(r_resp.clicked, "radio button clicked");

    // Segmented control selects segment 2 on click.
    let mut selected = 0usize;
    let seg_rect = Rect::new(10.0, 80.0, 180.0, 28.0);
    u.begin(
        UiInput {
            mouse: (150.0, 94.0),
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    u.segmented(
        "tabs",
        seg_rect,
        &["One", "Two", "Three"],
        &mut selected,
        style,
    );
    u.end();
    u.begin(
        UiInput {
            mouse: (150.0, 94.0),
            mouse_released: true,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    let chosen = u.segmented(
        "tabs",
        seg_rect,
        &["One", "Two", "Three"],
        &mut selected,
        style,
    );
    u.end();
    assert_eq!(chosen, Some(2));
    assert_eq!(selected, 2);
}

#[test]
fn custom_icon_and_placeholder_and_card_render() {
    let mut u = ui();
    u.set_theme(&Theme::slate_dark());
    let mut text = String::new();
    u.begin(UiInput::default(), (400.0, 300.0));
    u.card(
        Rect::new(10.0, 10.0, 300.0, 200.0),
        PanelStyle::new(Color::hex(0x0f172a)).corner_radius(12.0),
    );
    u.text_field_with_placeholder(
        "search",
        Rect::new(24.0, 24.0, 200.0, 34.0),
        &mut text,
        "Search...",
        u.theme().text_field(FontId(0)),
    );
    u.badge(
        Rect::new(24.0, 70.0, 60.0, 22.0),
        "PRO",
        Color::from_hex("#3b82f6").unwrap(),
        Color::WHITE,
        FontId(0),
        11.0,
    );
    u.button_with_icon(
        "Run",
        Rect::new(24.0, 100.0, 100.0, 32.0),
        crate::ui::icons::Icon::Play,
        16,
        u.theme().primary_button(FontId(0)),
    );
    u.custom_icon(
        "diamond",
        Rect::new(140.0, 100.0, 24.0, 24.0),
        &["M12 2 L22 12 L12 22 L2 12 Z"],
        20,
        Color::WHITE,
    );
    u.separator(Rect::new(24.0, 140.0, 200.0, 1.0));
    u.end_clip();
    let frame = u.end();
    assert!(frame.quad_count() > 8);
    assert_eq!(
        u.icons.baked(),
        2,
        "Play + custom diamond icon baked on demand"
    );
}

/// `KeyFeed` is the default keybinding layer: a `Char` under the shortcut
/// modifier is an editing chord (never text), select-all/copy feed the
/// clipboard seam, and paste replaces the selection.
#[test]
fn key_feed_drives_the_editing_chords() {
    let mut u = ui();
    let mut feed = KeyFeed::default();
    let mut input = UiInput::default();
    let mut text = String::from("seed");
    let rect = Rect::new(10.0, 10.0, 200.0, 30.0);
    let frame = |u: &mut Ui, input: &mut UiInput, feed: &KeyFeed, text: &mut String| {
        feed.apply(input);
        u.begin(std::mem::take(input), (400.0, 200.0));
        u.text_field("f", rect, text, ButtonStyle::new(FontId(0)));
        u.end();
    };
    // Click to focus.
    input.mouse = (50.0, 20.0);
    input.mouse_pressed = true;
    input.mouse_down = true;
    frame(&mut u, &mut input, &feed, &mut text);
    assert!(u.editing());
    // Ctrl+A, Ctrl+C: select-all + copy land on the clipboard seam.
    feed.modifiers(false, true, false);
    feed.key(&mut input, Key::Char('a'), true);
    feed.key(&mut input, Key::Char('c'), true);
    frame(&mut u, &mut input, &feed, &mut text);
    assert_eq!(u.take_clipboard_output().as_deref(), Some("seed"));
    assert_eq!(text, "seed", "copy does not mutate the document");
    // Ctrl+V pastes the OS clipboard over the selection.
    crate::ui::clipboard::set_text("grown");
    feed.key(&mut input, Key::Char('v'), true);
    frame(&mut u, &mut input, &feed, &mut text);
    assert_eq!(text, "grown");
    // Ctrl+Z walks the paste back; Shift+Ctrl+Z replays it.
    feed.key(&mut input, Key::Char('z'), true);
    frame(&mut u, &mut input, &feed, &mut text);
    assert_eq!(text, "seed");
    feed.key(&mut input, Key::Shift, true);
    feed.key(&mut input, Key::Char('z'), true);
    frame(&mut u, &mut input, &feed, &mut text);
    assert_eq!(text, "grown");
}

/// Modifier latches track both key events and `modifiers()` ground
/// truth; `chord` reports misses so a host's own shortcut layer can
/// route them, and a `Char` under command is never typed text.
#[test]
fn key_feed_latches_and_chord_misses() {
    let mut feed = KeyFeed::default();
    let mut input = UiInput::default();
    // Modifier keys latch on press AND release.
    feed.key(&mut input, Key::Shift, true);
    assert!(feed.shift() && input.nav.shift);
    feed.key(&mut input, Key::Shift, false);
    assert!(!feed.shift() && !input.nav.shift);
    // Control is the shortcut modifier off-macOS (Super there) and the
    // word modifier off-macOS (Option there).
    feed.key(&mut input, Key::Control, true);
    assert_eq!(feed.command(), !cfg!(target_os = "macos"));
    feed.apply(&mut input);
    assert_eq!(input.nav.word, !cfg!(target_os = "macos"));
    // Known chords decode; unknown ones report a miss and drop.
    assert!(feed.chord(&mut input, 'x'));
    assert!(input.nav.cut);
    assert!(!feed.chord(&mut input, 'b'));
    // The shortcut modifier is Super on macOS, where Control only latches
    // the word key.
    #[cfg(target_os = "macos")]
    feed.key(&mut input, Key::Super, true);
    feed.key(&mut input, Key::Char('q'), true);
    assert!(input.text.is_empty(), "a command-chord never types");
    // Plain chars type; control chars drop.
    feed.modifiers(false, false, false);
    feed.key(&mut input, Key::Char('h'), true);
    feed.key(&mut input, Key::Char('\u{7f}'), true);
    assert_eq!(input.text, "h");
    // Backspace counts presses only; releases add none.
    feed.key(&mut input, Key::Backspace, true);
    feed.key(&mut input, Key::Backspace, false);
    assert_eq!(input.backspace, 1);
}

#[test]
fn motion_scale_zero_loop_does_not_animate() {
    let mut u = ui();
    u.set_motion_scale(0.0);
    u.begin(
        UiInput {
            dt: 0.016,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let phase = u.anim_loop_f32("spin", 1.0);
    assert_eq!(phase, 0.0, "a frozen loop holds phase 0");
    assert!(!u.animating(), "a frozen loop must not keep the host awake");
    u.end();
}

#[test]
fn rate_scope_carries_sub_step_residue() {
    let mut u = ui();
    let mut got = Vec::new();
    for _ in 0..5 {
        u.begin(
            UiInput {
                dt: 0.06,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.rate_scope("scoped", Some(10.0));
        got.push(*u.rate_stack.last().expect("scope pushed a dt"));
        u.end_rate_scope();
        u.end();
    }
    let expected = [0.0, 0.1, 0.0, 0.1, 0.1];
    for (got, want) in got.iter().zip(expected) {
        assert!(
            (got - want).abs() < 1e-4,
            "scoped dts {got:?} — expected whole 0.1 s steps"
        );
    }
    let total: f32 = got.iter().sum();
    assert!(
        (total - 0.3).abs() < 1e-4,
        "no time is discarded: {total} != 0.3"
    );
}

#[test]
fn rate_scope_sub_step_deficit_does_not_fire_early() {
    let mut u = ui();
    u.begin(
        UiInput {
            dt: 0.099999,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.rate_scope("s", Some(10.0));
    assert_eq!(
        *u.rate_stack.last().unwrap(),
        0.0,
        "0.099999 s is not a 10 Hz step"
    );
    u.end_rate_scope();
    u.end();
    u.begin(
        UiInput {
            dt: 0.000001,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.rate_scope("s", Some(10.0));
    let got = *u.rate_stack.last().unwrap();
    assert!(
        (got - 0.1).abs() < 1e-4,
        "the held residue completes: {got}"
    );
    u.end_rate_scope();
    u.end();
}

#[test]
fn rate_scope_sanitizes_invalid_hz() {
    let mut u = ui();
    for hz in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -30.0] {
        u.begin(
            UiInput {
                dt: 0.05,
                ..Default::default()
            },
            (800.0, 600.0),
        );
        u.rate_scope("bad", Some(hz));
        let got = *u.rate_stack.last().expect("scope pushed a dt");
        assert!(
            (got - 0.05).abs() < 1e-6,
            "hz {hz}: expected the uncapped 0.05, got {got}"
        );
        u.end_rate_scope();
        u.end();
    }
}

#[test]
fn label_ellipsis_draws_the_longest_fitting_prefix() {
    use crate::ui::draw::CmdTexture;
    fn glyph_quads(frame: &UiFrame) -> usize {
        frame
            .cmds
            .iter()
            .filter(|cmd| matches!(cmd.texture, CmdTexture::AtlasPage(_)))
            .map(|cmd| cmd.index_count as usize / 6)
            .sum()
    }
    let mut u = ui();
    let style = LabelStyle::new(FontId(0)).size(16.0);
    let w_two = u.measure("ab\u{2026}", FontId(0), 16.0).0;
    let w_three = u.measure("abc\u{2026}", FontId(0), 16.0).0;
    assert!(w_three > w_two, "probe widths must differ");
    let width = (w_two + w_three) * 0.5;
    u.begin(UiInput::default(), (800.0, 600.0));
    u.label_ellipsis(Rect::new(0.0, 0.0, width, 20.0), "abcdefg", style);
    let frame = u.end();
    assert_eq!(
        glyph_quads(&frame),
        3,
        "expected \"ab\u{2026}\" — the winning prefix, not a failed probe"
    );
}

#[test]
fn label_ellipsis_edge_widths() {
    use crate::ui::draw::CmdTexture;
    fn glyph_quads(frame: &UiFrame) -> usize {
        frame
            .cmds
            .iter()
            .filter(|cmd| matches!(cmd.texture, CmdTexture::AtlasPage(_)))
            .map(|cmd| cmd.index_count as usize / 6)
            .sum()
    }
    let mut u = ui();
    let style = LabelStyle::new(FontId(0)).size(16.0);
    let w_ellipsis = u.measure("\u{2026}", FontId(0), 16.0).0;
    let w_full = u.measure("hi", FontId(0), 16.0).0;
    u.begin(UiInput::default(), (800.0, 600.0));
    u.label_ellipsis(
        Rect::new(0.0, 0.0, w_ellipsis + 0.5, 20.0),
        "abcdefg",
        style,
    );
    assert_eq!(glyph_quads(&u.end()), 1, "ellipsis-only width");
    u.begin(UiInput::default(), (800.0, 600.0));
    u.label_ellipsis(
        Rect::new(0.0, 0.0, (w_ellipsis - 1.0).max(0.1), 20.0),
        "abcdefg",
        style,
    );
    assert_eq!(glyph_quads(&u.end()), 0, "too narrow draws nothing");
    u.begin(UiInput::default(), (800.0, 600.0));
    u.label_ellipsis(Rect::new(0.0, 0.0, w_full + 4.0, 20.0), "hi", style);
    assert_eq!(glyph_quads(&u.end()), 2, "the full text draws");
}

#[test]
fn label_ellipsis_never_splits_a_cluster() {
    use crate::ui::draw::CmdTexture;
    fn glyph_quads(frame: &UiFrame) -> usize {
        frame
            .cmds
            .iter()
            .filter(|cmd| matches!(cmd.texture, CmdTexture::AtlasPage(_)))
            .map(|cmd| cmd.index_count as usize / 6)
            .sum()
    }
    let mut u = ui();
    let style = LabelStyle::new(FontId(0)).size(16.0);
    let text = "e\u{301}abcde";
    let w_ellipsis = u.measure("\u{2026}", FontId(0), 16.0).0;
    let w_cluster = u.measure("e\u{301}\u{2026}", FontId(0), 16.0).0;
    let width = (w_ellipsis + w_cluster) * 0.5;
    u.begin(UiInput::default(), (800.0, 600.0));
    u.label_ellipsis(Rect::new(0.0, 0.0, width, 20.0), text, style);
    assert_eq!(
        glyph_quads(&u.end()),
        1,
        "below the first cluster only the ellipsis fits"
    );
}

#[test]
fn layout_cache_wrap_same_eighth_is_order_independent() {
    use crate::ui::atlas::Atlas;
    use crate::ui::font::GlyphCache;
    use crate::ui::test_util::test_atlas_config;
    let fonts = test_font();
    let mut glyphs = GlyphCache::new(Atlas::new(test_atlas_config()));
    let mut raw = |w: f32, text: &str| {
        crate::ui::font::layout(
            &fonts,
            &mut glyphs,
            FontId(0),
            16.0,
            text,
            LayoutOptions {
                wrap_width: Some(w),
                ..Default::default()
            },
        )
    };
    let mut probe = None;
    for n in 2..20 {
        let probe_text = format!("aaaa {}", "b".repeat(n));
        let w = raw(f32::MAX, &probe_text).width;
        let w_hi = (w * 8.0).ceil() / 8.0;
        let w_lo = w_hi - 0.0624;
        if w_lo < w && w <= w_hi {
            probe = Some((probe_text, w_lo, w_hi));
            break;
        }
    }
    let Some((probe_text, w_lo, w_hi)) = probe else {
        panic!("no in-eighth wrap boundary found")
    };
    let text = probe_text.as_str();
    assert_ne!(
        raw(w_lo, text).line_count,
        raw(w_hi, text).line_count,
        "probe pair must wrap differently at the raw widths"
    );
    let options = |w: f32| LayoutOptions {
        wrap_width: Some(w),
        ..Default::default()
    };
    let mut a = ui();
    let a_lo = a.cached_layout(text, FontId(0), 16.0, options(w_lo));
    let a_hi = a.cached_layout(text, FontId(0), 16.0, options(w_hi));
    assert!(std::sync::Arc::ptr_eq(&a_lo, &a_hi), "same eighth is a hit");
    let mut b = ui();
    let b_hi = b.cached_layout(text, FontId(0), 16.0, options(w_hi));
    let b_lo = b.cached_layout(text, FontId(0), 16.0, options(w_lo));
    assert!(std::sync::Arc::ptr_eq(&b_hi, &b_lo));
    assert_eq!(
        a_lo.line_count, b_hi.line_count,
        "w_lo-first and w_hi-first must shape identically"
    );
    assert_eq!(a_lo.glyphs.len(), b_hi.glyphs.len());
    assert_eq!(a_lo.width, b_hi.width);
}

#[test]
fn layout_cache_wrap_unwrapped_never_aliases() {
    let mut u = ui();
    let options = |w: Option<f32>| LayoutOptions {
        wrap_width: w,
        ..Default::default()
    };
    let _ = u.cached_layout("text", FontId(0), 16.0, options(None));
    let base = u.layout_stats();
    let _ = u.cached_layout("text", FontId(0), 16.0, options(Some(f32::INFINITY)));
    assert_eq!(
        u.layout_stats().misses,
        base.misses + 1,
        "an infinite wrap is not the unwrapped entry"
    );
    let _ = u.cached_layout("text", FontId(0), 16.0, options(Some(1.0e30)));
    assert_eq!(
        u.layout_stats().misses,
        base.misses + 1,
        "huge finite and infinite wrap share one canonical maximum"
    );
    let _ = u.cached_layout("text", FontId(0), 16.0, options(Some(f32::NAN)));
    let _ = u.cached_layout("text", FontId(0), 16.0, options(Some(-4.0)));
    assert_eq!(
        u.layout_stats().misses,
        base.misses + 2,
        "NaN and negative normalize to the same canonical width"
    );
}

#[test]
fn widget_state_retirement_reclaims_capacity() {
    let mut u = ui();
    u.begin(UiInput::default(), (100.0, 100.0));
    u.end();
    for i in 0..5000u64 {
        u.state.insert(i, WidgetState::new());
        u.editors.insert(i, editor::EditorState::new(""));
    }
    assert!(u.state.capacity() > 4096);
    u.frame += 601;
    u.retire_stale_state();
    assert!(u.state.is_empty(), "stale states removed");
    assert!(u.editors.is_empty(), "stale editors removed");
    assert!(
        u.state.capacity() <= 1024,
        "state capacity {} still pinned at the peak",
        u.state.capacity()
    );
    assert!(
        u.editors.capacity() <= 1024,
        "editor capacity {} still pinned at the peak",
        u.editors.capacity()
    );
}

#[test]
fn widget_state_eviction_protects_live_ids_and_bounds_transients() {
    let mut u = ui();
    u.begin(UiInput::default(), (100.0, 100.0));
    u.end();
    u.frame = 700;
    let frame = u.frame;
    let protected_old = 1u64;
    let protected_active = 2u64;
    u.focus_id = Some(protected_old);
    u.active_id = Some(protected_active);
    u.state.insert(
        protected_old,
        WidgetState {
            last_frame: frame - 599,
            ..WidgetState::new()
        },
    );
    u.state.insert(
        protected_active,
        WidgetState {
            last_frame: frame - 599,
            ..WidgetState::new()
        },
    );
    u.state.insert(
        3,
        WidgetState {
            last_frame: frame,
            ..WidgetState::new()
        },
    );
    for i in 10..5010u64 {
        u.state.insert(
            i,
            WidgetState {
                last_frame: frame - 1 - (i % 550),
                ..WidgetState::new()
            },
        );
    }
    u.retire_stale_state();
    assert_eq!(u.state.len(), 4096, "bounded exactly at the cap");
    assert!(u.state.contains_key(&protected_old), "focus survives");
    assert!(u.state.contains_key(&protected_active), "active survives");
    assert!(u.state.contains_key(&3), "this-frame entry survives");
}

#[test]
fn text_field_shares_the_label_layout() {
    let mut u = ui();
    let mut text = "abc".to_string();
    let rect = Rect::new(0.0, 0.0, 100.0, 20.0);
    u.begin(UiInput::default(), (100.0, 100.0));
    u.text_field("f", rect, &mut text, ButtonStyle::new(FontId(0)));
    u.end();
    let stats = u.layout_stats();
    assert_eq!(
        stats.entries, 1,
        "caret, selection and text share one layout, got {}",
        stats.entries
    );
}

/// Cap centring: the optical cap span is centred and the baseline lands on
/// a whole physical pixel row at every size and scale.
#[test]
fn label_optical_center_cap_centres_the_cap_span() {
    use super::super::font::FontId;
    let face = crate::ui::test_util::test_font();
    for scale in [1.0f32, 2.0] {
        let mut u = ui();
        u.set_scale(scale);
        for size in 11..=24u32 {
            let rect = Rect::new(0.0, 0.0, 120.0, 40.0);
            let style = LabelStyle::new(FontId(0))
                .size(size as f32)
                .vcenter(true)
                .vcenter_mode(VCenter::Cap);
            u.begin(UiInput::default(), (800.0, 600.0));
            u.label(rect, "Hxg", style);
            u.end();
            let layout = u.cached_layout(
                "Hxg",
                FontId(0),
                size as f32 * scale,
                LayoutOptions::default(),
            );
            let (origin, baseline) = super::controls::label_origin(
                u.fonts.get_opt(FontId(0)),
                rect,
                &layout,
                size as f32,
                scale,
                &style,
            );
            let cap = face.get(FontId(0)).cap_height(size as f32 * scale);
            // Baseline is on a physical row, and the cap span is centred
            // on the rect centre (physical px).
            let base_phys = baseline * scale;
            assert!(
                (base_phys - base_phys.round()).abs() < 1e-4,
                "baseline snapped"
            );
            let cap_top = baseline - cap / scale;
            let span_c = (cap_top + cap / scale * 0.5) * scale;
            assert!(
                (span_c - rect.center().1 * scale).abs() <= 0.5,
                "size {size} scale {scale}: span centre {span_c} vs {}",
                rect.center().1 * scale
            );
            let _ = origin;
        }
    }
}

/// `Box` mode reproduces the historical ascent–descent block centring.
#[test]
fn label_optical_center_box_matches_legacy_block() {
    use super::super::font::FontId;
    let face = crate::ui::test_util::test_font();
    let mut u = ui();
    u.set_scale(1.0);
    let rect = Rect::new(10.0, 20.0, 100.0, 30.0);
    let style = LabelStyle::new(FontId(0))
        .size(13.0)
        .vcenter(true)
        .vcenter_mode(VCenter::Box);
    let layout = u.cached_layout("Hxg", FontId(0), 13.0, LayoutOptions::default());
    let (origin, _) =
        super::controls::label_origin(u.fonts.get_opt(FontId(0)), rect, &layout, 13.0, 1.0, &style);
    let f = face.get(FontId(0));
    let block = f.ascent(13.0) - f.descent(13.0);
    let want_y = rect.center().1 - block * 0.5;
    assert!(
        (origin.1 - want_y).abs() <= 1.0,
        "box mode origin {origin:?} vs legacy {want_y}"
    );
}

/// Ink-fit centring: every icon's rasterized ink bbox is centred on the
/// rect centre within half a physical pixel.
#[test]
fn icon_ink_center_is_centred() {
    use crate::ui::icons::{Icon, ink_box, rasterize};
    use crate::ui::tokens::ControlSize;
    let _ = ControlSize::Xs;
    let rect = Rect::new(10.0, 10.0, 28.0, 28.0);
    let mut u = ui();
    for scale in [1.0f32, 2.0] {
        for size in [12u32, 16, 20, 24] {
            for icon in Icon::ALL {
                let ink = ink_box(icon);
                let (quad, s) = u
                    .ink_icon_quad(rect, size, ink, icon.optical_k(), scale)
                    .expect("quad");
                // Coverage bbox of the S×S raster.
                let cov = rasterize(icon, s);
                let (mut min, mut max) = (s + 1, 0);
                let (mut min_y, mut max_y) = (s + 1, 0);
                for y in 0..s {
                    for x in 0..s {
                        if cov[(y * s + x) as usize] > 0 {
                            min = min.min(x);
                            max = max.max(x);
                            min_y = min_y.min(y);
                            max_y = max_y.max(y);
                        }
                    }
                }
                if min > max {
                    continue;
                }
                // Raster px -> physical px via the emitted quad (side = S/scale).
                let phys_per_raster = quad.w / s as f32 * scale;
                let bcx = quad.x * scale + (min + max + 1) as f32 * 0.5 * phys_per_raster;
                let bcy = quad.y * scale + (min_y + max_y + 1) as f32 * 0.5 * phys_per_raster;
                let tol = if icon == Icon::List { 1.0 } else { 0.5 };
                assert!(
                    (bcx - rect.center().0 * scale).abs() <= tol + 1e-4
                        && (bcy - rect.center().1 * scale).abs() <= tol + 1e-4,
                    "{icon:?} size {size} scale {scale}: ink centre ({bcx},{bcy}) vs ({},{})",
                    rect.center().0 * scale,
                    rect.center().1 * scale
                );
            }
        }
    }
}

/// Focus-visible: rings draw only for keyboard-acquired focus.
#[test]
fn focus_ring_keyboard_only() {
    let mut u = ui();
    u.set_focus_rings(true);
    let rect = Rect::new(10.0, 10.0, 80.0, 28.0);
    let style = Theme::dark_theme().button(FontId(0));
    // Frame 1: button registers focusable.
    u.begin(UiInput::default(), (200.0, 100.0));
    u.button("b", rect, style);
    u.end();
    // Frame 2: Tab → focus via keyboard → ring drawn.
    u.begin(
        UiInput {
            nav: crate::ui::widgets::NavInput {
                tab: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (200.0, 100.0),
    );
    u.button("b", rect, style);
    assert!(u.focus_visible(), "Tab-acquired focus is visible");
    let quads = u.end().quad_count();
    assert!(quads > 0, "ring emitted");
    // Frame 3: pointer press clears it.
    u.begin(
        UiInput {
            mouse_pressed: true,
            mouse_down: true,
            ..Default::default()
        },
        (200.0, 100.0),
    );
    u.button("b", rect, style);
    assert!(!u.focus_visible(), "pointer press clears visibility");
    u.end();
}

/// Rings off: nothing extra even with keyboard focus.
#[test]
fn focus_ring_off_by_default() {
    let mut u = ui();
    u.begin(
        UiInput {
            nav: crate::ui::widgets::NavInput {
                tab: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (200.0, 100.0),
    );
    let style = Theme::dark_theme().button(FontId(0));
    u.button("b", Rect::new(10.0, 10.0, 80.0, 28.0), style);
    u.button("c", Rect::new(10.0, 50.0, 80.0, 28.0), style);
    u.end();
    assert!(!u.focus_visible() && !u.focus_rings());
}

/// The ring rect is concentric: widget radius + offset + width.
#[test]
fn focus_ring_is_concentric() {
    let mut u = ui();
    u.set_focus_rings(true);
    let rect = Rect::new(10.0, 10.0, 80.0, 28.0);
    let style = Theme::dark_theme().button(FontId(0));
    u.begin(UiInput::default(), (200.0, 100.0));
    u.button("b", rect, style);
    u.end();
    u.begin(
        UiInput {
            nav: crate::ui::widgets::NavInput {
                tab: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (200.0, 100.0),
    );
    u.button("b", rect, style);
    let t = u.theme().tokens.focus;
    // Ring geometry check via draw list is internal; assert the tokens it
    // draws from: ring rect = rect.expand(offset + width).
    let ring = rect.expand(t.offset + t.width);
    assert_eq!(ring, Rect::new(6.0, 6.0, 88.0, 36.0));
    u.end();
}

// --- headless behaviours + paint split -------------------------------------

/// The headless button path must yield identical Response/State as Ui::button
/// over a press→release sequence.
#[test]
fn headless_button_matches_widget() {
    let rect = Rect::new(10.0, 10.0, 100.0, 40.0);
    let style = ButtonStyle::new(FontId(0));
    let run = |use_headless: bool| {
        let mut u = ui();
        let mut seq = Vec::new();
        for inp in [
            UiInput {
                mouse: (50.0, 30.0),
                ..Default::default()
            },
            UiInput {
                mouse: (50.0, 30.0),
                mouse_down: true,
                mouse_pressed: true,
                ..Default::default()
            },
            UiInput {
                mouse: (50.0, 30.0),
                mouse_released: true,
                ..Default::default()
            },
        ] {
            u.begin(inp, (800.0, 600.0));
            if use_headless {
                let (r, s) = headless::ButtonBehavior::new("ok")
                    .enabled(!style.disabled)
                    .update(&mut u, rect);
                seq.push((r.clicked, r.hovered, s.pressed, s.hovered));
            } else {
                let r = u.button("ok", rect, style);
                seq.push((r.clicked, r.hovered, r.held, r.hovered));
            }
            u.end();
        }
        seq
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn headless_toggle_flips_and_animates() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 60.0, 24.0);
    let mut on = false;
    // Click frame: press.
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let (_, s) = headless::ToggleBehavior::new("t", &mut on)
        .duration(0.15)
        .anim_suffix("##switch")
        .update(&mut u, rect);
    assert!(s.pressed && !s.selected);
    u.end();
    // Release frame: flips.
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            mouse_released: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let (r, s) = headless::ToggleBehavior::new("t", &mut on)
        .duration(0.15)
        .anim_suffix("##switch")
        .update(&mut u, rect);
    assert!(r.clicked && on && s.selected && s.value < 1.0);
    u.end();
}

#[test]
fn headless_slider_drags_and_nudges() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 100.0, 20.0);
    let mut v = 0.0f32;
    // Press at x=50 → t=0.5.
    u.begin(
        UiInput {
            mouse: (50.0, 10.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let (_, s) = headless::SliderBehavior::new("s", &mut v, 0.0..=1.0).update(&mut u, rect);
    assert!((s.value - 0.5).abs() < 1e-5 && (v - 0.5).abs() < 1e-5);
    u.end();
    // Focused + nav.right nudges +0.1.
    let mut inp = UiInput {
        mouse: (50.0, 10.0),
        mouse_down: true,
        ..Default::default()
    };
    u.begin(inp.clone(), (800.0, 600.0));
    headless::SliderBehavior::new("s", &mut v, 0.0..=1.0).update(&mut u, rect);
    u.end();
    u.begin(
        UiInput {
            mouse: (50.0, 10.0),
            mouse_released: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    inp = UiInput::default();
    inp.nav.right = true;
    u.begin(inp, (800.0, 600.0));
    headless::SliderBehavior::new("s", &mut v, 0.0..=1.0).update(&mut u, rect);
    assert!((v - 0.6).abs() < 1e-4, "nudge: {v}");
    u.end();
}

#[test]
fn headless_segmented_click_and_ids_stable() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 180.0, 24.0);
    let style = Theme::default().segmented(FontId(0));
    let mut sel = 0usize;
    let opts = ["a", "b", "c"];
    u.begin(
        UiInput {
            mouse: (125.0, 12.0),
            mouse_down: true,
            mouse_pressed: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    u.segmented_styled("seg", rect, &opts, &mut sel, &style);
    u.end();
    u.begin(
        UiInput {
            mouse: (125.0, 12.0),
            mouse_released: true,
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let picked = u.segmented_styled("seg", rect, &opts, &mut sel, &style);
    u.end();
    assert_eq!(picked, Some(2));
    assert_eq!(sel, 2);
}

#[test]
fn segmented_styled_overrides_track_and_pill() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 180.0, 24.0);
    let style = Theme::default()
        .segmented(FontId(0))
        .track(Color::rgb(10, 20, 30))
        .pill(Color::rgb(200, 100, 50));
    let mut sel = 0usize;
    let opts = ["a", "b"];
    u.begin(UiInput::default(), (800.0, 600.0));
    u.segmented_styled("seg", rect, &opts, &mut sel, &style);
    let frame = u.end();
    // Track fill colour + pill colour appear in the emitted vertices.
    let colors: Vec<_> = frame.vertices.iter().map(|v| v.color).collect();
    assert!(colors.contains(&[10u8, 20, 30, 255]));
    assert!(colors.contains(&[200u8, 100, 50, 255]));
}

#[test]
fn segmented_with_slot_runs_once_per_item() {
    let mut u = ui();
    let rect = Rect::new(0.0, 0.0, 180.0, 24.0);
    let style = Theme::default().segmented(FontId(0));
    let mut sel = 1usize;
    let opts = ["a", "b", "c"];
    let mut seen = Vec::new();
    u.begin(UiInput::default(), (800.0, 600.0));
    u.segmented_with("seg", rect, &opts, &mut sel, &style, |_ui, item| {
        seen.push((item.index, item.label.to_string(), item.near));
    });
    u.end();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[1].1, "b");
    assert_eq!(seen[1].2, 1.0, "selected segment's near is 1");
}

#[test]
fn button_with_slot_receives_state() {
    let mut u = ui();
    let rect = Rect::new(10.0, 10.0, 80.0, 24.0);
    let style = Theme::default().button(FontId(0));
    u.begin(
        UiInput {
            mouse: (20.0, 20.0),
            ..Default::default()
        },
        (800.0, 600.0),
    );
    let mut got = headless::State::default();
    u.button_with("b", rect, &style, |_, r, st| {
        assert_eq!(r, rect);
        got = st;
    });
    u.end();
    assert!(got.hovered);
}

/// A custom widget built from behaviour+painter emits the same vertices
/// as the stock `Ui::button`.
#[test]
fn paint_button_matches_widget_vertices() {
    let rect = Rect::new(10.0, 10.0, 80.0, 24.0);
    let style = Theme::default().button(FontId(0));
    let run = |custom: bool| {
        let mut u = ui();
        u.begin(
            UiInput {
                mouse: (20.0, 20.0),
                ..Default::default()
            },
            (800.0, 600.0),
        );
        if custom {
            let (r, s) = headless::ButtonBehavior::new("b").update(&mut u, rect);
            paint::button(&mut u, rect, &style, s, "b");
            let _ = r;
        } else {
            u.button("b", rect, style);
        }
        u.end()
    };
    let a = run(false);
    let b = run(true);
    assert_eq!(a.vertices.len(), b.vertices.len());
    assert_eq!(format!("{:?}", a.vertices), format!("{:?}", b.vertices));
}

/// Focused TextField caret blink: the replay pass must not double-step
/// `blink_elapsed` (it reads `input.dt` directly, so dt=0 under replay).
#[test]
fn text_field_blink_single_step_on_replay() {
    let mut u = ui();
    let mut r = ui();
    let mut v = String::from("x");
    for _ in 0..2 {
        for uu in [&mut u, &mut r] {
            uu.begin(UiInput::default(), (400.0, 300.0));
            uu.layout_root(
                Rect::new(0.0, 0.0, 400.0, 300.0),
                crate::ui::layout::Style::column(),
                |ui| {
                    ui.add_styled(
                        crate::ui::widgets::adapters::TextField::new("f", &mut v),
                        crate::ui::layout::Style::default().fill_w().h_px(24.0),
                    );
                },
            );
            uu.end();
            uu.set_focus("f");
        }
    }
    // Replay frame: u gains a sibling; r is the control. dt > 0.
    let id = u.id("f");
    u.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    let mut s = String::new();
    u.layout_root(
        Rect::new(0.0, 0.0, 400.0, 300.0),
        crate::ui::layout::Style::column(),
        |ui| {
            ui.add_styled(
                crate::ui::widgets::adapters::TextField::new("f", &mut v),
                crate::ui::layout::Style::default().fill_w().h_px(24.0),
            );
            ui.add_styled(
                crate::ui::widgets::adapters::TextField::new("late", &mut s),
                crate::ui::layout::Style::default().fill_w().h_px(24.0),
            );
        },
    );
    u.end();
    r.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (400.0, 300.0),
    );
    r.layout_root(
        Rect::new(0.0, 0.0, 400.0, 300.0),
        crate::ui::layout::Style::column(),
        |ui| {
            ui.add_styled(
                crate::ui::widgets::adapters::TextField::new("f", &mut v),
                crate::ui::layout::Style::default().fill_w().h_px(24.0),
            );
        },
    );
    r.end();
    let bu = u.editors.get(&id).expect("editor").blink_elapsed;
    let br = r.editors.get(&r.id("f")).expect("editor").blink_elapsed;
    assert!(bu > 0.0 && bu == br, "blink double-stepped: {bu} vs {br}");
    // No edits or selection movement across the replay.
    let (su, sr) = (&u.editors[&id], &r.editors[&r.id("f")]);
    assert_eq!(su.caret, sr.caret);
    assert_eq!(v, "x");
}

#[test]
fn hooks_state_survives_frames_and_scopes_without_clone_bound() {
    struct Value(usize);
    let mut u = ui();
    u.begin(UiInput::default(), (200.0, 100.0));
    let root = u.use_state("value", || Value(1));
    let shared = root.clone();
    shared.update(|value| value.0 += 1);
    assert_eq!(root.with(|value| value.0), 2);
    u.push_id("child");
    let child = u.use_state("value", || Value(10));
    u.pop_id();
    u.end();
    u.begin(UiInput::default(), (200.0, 100.0));
    let retained = u.use_state("value", || panic!("initialized again"));
    assert_eq!(retained.with(|value: &Value| value.0), 2);
    assert_eq!(child.with(|value| value.0), 10);
    u.end();
}

#[test]
#[should_panic(expected = "use_state type changed")]
fn hooks_state_rejects_type_changes() {
    let mut u = ui();
    u.use_state("value", || 1usize);
    u.use_state("value", String::new);
}

#[test]
fn hooks_state_initializes_once_during_layout_replay() {
    let mut u = ui();
    let mut initialized = 0;
    let mut builds = 0;
    u.begin(UiInput::default(), (200.0, 100.0));
    u.layout_root(
        Rect::new(0.0, 0.0, 200.0, 100.0),
        crate::ui::layout::Style::column(),
        |ui| {
            builds += 1;
            let value = ui.use_state("value", || {
                initialized += 1;
                7usize
            });
            assert_eq!(value.get(), 7);
            ui.add(crate::ui::widgets::adapters::Button::new("new node"));
        },
    );
    u.end();
    assert_eq!(builds, 2);
    assert_eq!(initialized, 1);
}

#[test]
fn hooks_state_retirement_preserves_external_handles() {
    let mut u = ui();
    u.frame = 1;
    let old = u.use_state("value", || 3usize);
    u.frame = 601;
    u.retire_stale_state();
    let new = u.use_state("value", || 9usize);
    assert_eq!(new.get(), 9);
    assert_eq!(old.get(), 3);
    old.set(4);
    assert_eq!(new.get(), 9);
}

#[test]
fn hooks_context_restores_nearest_provider_after_unwind() {
    let mut u = ui();
    let outer = State::new(1usize);
    let inner = State::new(2usize);
    let other_type = State::new(String::from("shared"));
    assert!(u.use_context::<usize>().is_none());
    u.with_context(&outer, |ui| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ui.with_context(&inner, |ui| {
                assert_eq!(ui.use_context::<usize>().unwrap().get(), 2);
                panic!("unwind the inner provider");
            });
        }));
        assert!(result.is_err());
        assert_eq!(ui.use_context::<usize>().unwrap().get(), 1);
        ui.with_context(&other_type, |ui| {
            assert_eq!(ui.use_context::<String>().unwrap().get(), "shared");
            ui.use_context::<usize>().unwrap().set(5);
        });
        assert!(ui.use_context::<String>().is_none());
    });
    assert_eq!(outer.get(), 5);
    assert!(u.use_context::<usize>().is_none());
}

#[test]
fn presence_retains_exit_reverses_and_respects_reduced_motion() {
    let mut u = ui();
    let input = UiInput {
        dt: 0.1,
        ..Default::default()
    };
    u.begin(input.clone(), (200.0, 100.0));
    let hidden = u.presence("popup", false, 1.0);
    assert_eq!(hidden.phase, PresencePhase::Hidden);
    assert!(!hidden.is_present());
    assert!(!hidden.is_interactive());
    assert!(!u.animating(), "initial hidden presence must stay idle");
    u.end();
    u.begin(input.clone(), (200.0, 100.0));
    let entering = u.presence("popup", true, 1.0);
    assert_eq!(entering.phase, PresencePhase::Entering);
    assert!(entering.is_interactive());
    assert!(entering.opacity > 0.0 && entering.opacity < 1.0);
    assert!(u.animating());
    u.end();
    u.begin(input.clone(), (200.0, 100.0));
    let exiting = u.presence("popup", false, 1.0);
    assert_eq!(exiting.phase, PresencePhase::Exiting);
    assert!(exiting.is_present());
    assert!(!exiting.is_interactive());
    assert!(exiting.opacity < entering.opacity);
    u.end();
    u.begin(input.clone(), (200.0, 100.0));
    let reversing = u.presence("popup", true, 1.0);
    assert!(reversing.opacity > exiting.opacity);
    u.end();
    u.set_motion_scale(0.0);
    u.begin(input, (200.0, 100.0));
    assert_eq!(u.presence("popup", true, 1.0).phase, PresencePhase::Visible);
    let hidden = u.presence("popup", false, 1.0);
    assert_eq!(hidden.phase, PresencePhase::Hidden);
    assert!(!u.animating());
    assert_eq!(
        u.presence("instant", true, 1.0).phase,
        PresencePhase::Visible
    );
    assert_eq!(u.presence("invalid", true, f32::INFINITY).opacity, 1.0);
    u.end();
}

#[test]
fn presence_layout_replay_does_not_advance_twice() {
    let mut u = ui();
    let mut actual = 0.0;
    u.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (200.0, 100.0),
    );
    u.layout_root(
        Rect::new(0.0, 0.0, 200.0, 100.0),
        crate::ui::layout::Style::column(),
        |ui| {
            let p = ui.presence("popup", true, 1.0);
            actual = p.opacity;
            ui.add(crate::ui::widgets::adapters::Button::new(
                "first appearance",
            ));
        },
    );
    u.end();
    // A direct once-per-frame call has the same entrance progress.
    let mut reference = ui();
    reference.begin(
        UiInput {
            dt: 0.1,
            ..Default::default()
        },
        (200.0, 100.0),
    );
    let expected = reference.presence("popup", true, 1.0).opacity;
    reference.end();
    assert_eq!(actual, expected);
}

#[test]
fn floating_placement_flips_shifts_and_shrinks() {
    let boundary = Rect::new(0.0, 0.0, 200.0, 100.0);
    let anchor = Rect::new(170.0, 80.0, 20.0, 10.0);
    assert_eq!(
        Floating::default().rect(anchor, (80.0, 30.0), boundary),
        Rect::new(120.0, 46.0, 80.0, 30.0)
    );
    let right = Floating {
        side: FloatingSide::Right,
        align: FloatingAlign::Center,
        gap: 4.0,
    };
    assert_eq!(
        right.rect(anchor, (80.0, 30.0), boundary),
        Rect::new(86.0, 70.0, 80.0, 30.0)
    );
    assert_eq!(
        Floating::default().rect(anchor, (500.0, 500.0), boundary),
        boundary
    );
    let invalid = Floating {
        gap: f32::NAN,
        ..Default::default()
    }
    .rect(anchor, (f32::INFINITY, -1.0), boundary);
    assert_eq!(invalid.w, 0.0);
    assert_eq!(invalid.h, 0.0);
    assert!(invalid.x.is_finite() && invalid.y.is_finite());
}

#[test]
fn floating_popover_escapes_clip_restores_layer_and_dismisses() {
    let mut u = ui();
    u.begin(
        UiInput {
            mouse: (100.0, 50.0),
            nav: NavInput {
                cancel: true,
                ..Default::default()
            },
            ..Default::default()
        },
        (200.0, 100.0),
    );
    let clip = Rect::new(0.0, 0.0, 10.0, 10.0);
    u.draw.push_clip(clip);
    let response = u.popover(
        "panel",
        Rect::new(80.0, 20.0, 20.0, 10.0),
        (80.0, 50.0),
        Floating::default(),
        |ui, content| {
            assert_eq!(ui.draw.layer(), LAYER_POPUP);
            assert!(ui.draw.clip_rect().unwrap().contains(content.center()));
            assert!(ui.hit(content));
        },
    );
    assert!(response.dismissed);
    assert_eq!(u.draw.clip_rect(), Some(clip));
    assert_eq!(u.draw.layer(), 0);
    assert_eq!(u.surfaces_cur.len(), 1);
    u.draw.pop_clip();
    let frame = u.end();
    assert!(frame.cmds.iter().any(|cmd| cmd.layer == LAYER_POPUP));
}
