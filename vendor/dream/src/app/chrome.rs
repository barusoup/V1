//! The app shell: custom window bar, toolbar, tool rail and viewport menu.
//!
//! Surfaces are flat fills with hairline borders and text is sentence case
//! throughout. Icon state is conveyed by tint colour only â€” never by weight or
//! opacity â€” and controls sit on whole pixels so edges stay crisp.

use super::{App, ResizeEdge, WindowCommand, WindowCursor, main_pane};
use crate::ui::Rect;
use crate::ui::icons::Icon;
use crate::ui::widgets::{ButtonStyle, MenuItem};

/// Icon size inside a chrome button, in pixels (a pre-baked size).
const ICON: u32 = 20;
/// Caption glyph raster sizes, one per glyph.
///
/// Lucide's shapes have different extents inside its 24-unit box and the
/// stroker adds one stroke width, so one shared raster size gives three
/// different ink boxes. Measured on a rendered frame at 16px the rule lands
/// 12x2, the square 14x14 and the cross 10x10 â€” the square read a step larger
/// than the glyphs either side of it, so it is rasterized smaller to land the
/// same 10x10 box as the cross. (`ink = (units + 2) * size / 24`: the square
/// spans 18 units, the cross 12.)
// Ink-fit icon drawing normalises the caption glyphs — one shared size
// for all three (the old per-glyph sizes are gone).
/// Shared caption glyph size (ink-fit normalises the three glyphs).
const CAPTION_ICON: u32 = 16;
/// Width of a caption button in the custom window bar.
pub const CAPTION_W: f32 = 46.0;

/// Centre a square button of `size` vertically in `bar` at horizontal `x`.
fn slot(bar: Rect, x: f32, size: f32) -> Rect {
    Rect::new(x, bar.center().1 - size * 0.5, size, size)
}

/// The toolbar settings button's rect (the shortcut layer and the scripted
/// selfcheck key off it).
pub fn settings_button_rect(app: &App) -> Rect {
    let bar = app.layout.toolbar;
    let m = &app.theme.metrics;
    let x = bar.max().0 - m.pad - m.button;
    slot(bar, x, m.button)
}

/// The settings button's previous name: the theme button it used to name sat
/// in exactly this slot.
pub fn theme_button_rect(app: &App) -> Rect {
    settings_button_rect(app)
}

/// The debug-scene boot button rect on the scene management page.
pub fn debug_button_rect(app: &App) -> Rect {
    main_pane::debug_button_rect(app)
}

/// Fill a bar surface with a single hairline border on the given edges.
fn bar_surface(app: &mut App, rect: Rect, bottom: bool, left: bool, right: bool) {
    let theme = app.theme;
    app.ui.draw_list_mut().rect(rect, theme.g1);
    let b = theme.metrics.border;
    if bottom {
        app.ui
            .draw_list_mut()
            .rect(Rect::new(rect.x, rect.max().1 - b, rect.w, b), theme.border);
    }
    if left {
        app.ui
            .draw_list_mut()
            .rect(Rect::new(rect.x, rect.y, b, rect.h), theme.border);
    }
    if right {
        app.ui
            .draw_list_mut()
            .rect(Rect::new(rect.max().0 - b, rect.y, b, rect.h), theme.border);
    }
}

pub fn toolbar(app: &mut App) {
    let rect = app.layout.toolbar;
    bar_surface(app, rect, true, false, false);
    let theme = app.theme;
    let font = app.font();
    let m = theme.metrics;
    let style = theme.icon_button(font);

    // Right cluster: settings. The button latches while its page is showing.
    let rx = rect.max().0 - m.pad - m.button;
    let settings_on = app.main_pane_page == main_pane::MainPanePage::Settings;
    if app
        .ui
        .icon_button(
            "settings",
            slot(rect, rx, m.button),
            Icon::Settings,
            style,
            settings_on,
            None,
            ICON,
        )
        .clicked
    {
        // The gear latches while Settings is open: clicking again leaves.
        if settings_on {
            app.close_settings();
        } else {
            app.show_page(main_pane::MainPanePage::Settings);
        }
    }
}

pub fn rail(app: &mut App) {
    let rect = app.layout.rail;
    bar_surface(app, rect, false, false, true);
}

// --- Viewport context menu -------------------------------------------------

/// Rows of the viewport context menu, top to bottom. The menu only opens on
/// the viewport page, so every row is an action on the view itself; scene
/// management and page navigation live on the scene manager page.
fn menu_items() -> Vec<MenuItem> {
    vec![
        MenuItem::new("Reset camera"),
        MenuItem::new("Capture frame"),
    ]
}

/// The popup's rect for this frame, or `None` while it is closed. The app
/// confines pointer input to this rect before drawing anything else, so
/// nothing behind the menu can be clicked through it.
pub fn context_menu_rect(app: &App) -> Option<Rect> {
    if !app.menu_open || app.main_pane_page != main_pane::MainPanePage::Viewport {
        return None;
    }
    let items = menu_items();
    let style = app.theme.menu(app.font());
    // A little slack for the popup's slide-in offset.
    Some(
        app.ui
            .context_menu_rect(app.menu_at, &items, &style)
            .expand(12.0),
    )
}

/// Draw the viewport context menu and run the chosen row.
///
/// Called every frame: while the menu is animating shut it is still drawn (the
/// widget owns that motion), and it is a no-op once fully closed.
pub fn context_menu(app: &mut App) {
    // The menu belongs to the viewport page; if the page changed while it was
    // open (key shortcuts still reach the app with the menu up), the latch is
    // stale and dropped rather than drawn over the wrong page.
    if app.main_pane_page != main_pane::MainPanePage::Viewport {
        app.menu_open = false;
        return;
    }
    let items = menu_items();
    let style = app.theme.menu(app.font());
    let at = app.menu_at;
    let mut open = app.menu_open;
    let mut focus = app.menu_focus;
    let chosen = app
        .ui
        .context_menu("viewport-menu", &mut open, at, &items, &mut focus, style);
    app.menu_focus = focus;
    // The widget clears the latch when the pointer clicks away or Escape is
    // pressed; it then keeps drawing the exit for a few frames.
    app.menu_open = open;
    if let Some(index) = chosen {
        match index {
            0 => app.reset_camera(),
            1 => app.capture_png(),
            _ => {}
        }
    }
}

// --- Window bar ------------------------------------------------------------

/// The custom window bar: drawn after the panels so it floats above them.
/// Dragging it moves the window (borderless: there is no OS chrome), the right
/// cluster holds the caption buttons, and a thin frame around the window edges
/// resizes it.
///
/// It carries no title or app mark: the bar exists to be dragged and to hold
/// the caption buttons.
pub fn window_bar(app: &mut App) {
    let rect = app.layout.titlebar;
    bar_surface(app, rect, true, false, false);
    let theme = app.theme;
    let font = app.font();

    // Caption buttons (right cluster): minimize, maximize/restore, close.
    let close_rect = Rect::new(rect.max().0 - CAPTION_W, rect.y, CAPTION_W, rect.h);
    let max_rect = Rect::new(close_rect.x - CAPTION_W, rect.y, CAPTION_W, rect.h);
    let min_rect = Rect::new(max_rect.x - CAPTION_W, rect.y, CAPTION_W, rect.h);
    let base = theme.icon_button(font);
    let caption = ButtonStyle {
        hover: theme.g3,
        pressed: theme.pressed,
        text: theme.muted,
        text_hover: theme.fg,
        corner_radius: 0.0,
        ..base
    };
    let close_style = ButtonStyle {
        hover: theme.danger,
        text_hover: crate::ui::Color::rgb(0xff, 0xff, 0xff),
        ..caption
    };

    if app
        .ui
        .icon_button(
            "win-min",
            min_rect,
            Icon::Minus,
            caption,
            false,
            None,
            CAPTION_ICON,
        )
        .clicked
    {
        app.window_commands.push(WindowCommand::Minimize);
    }
    let max_icon = if app.window_maximized {
        Icon::Restore
    } else {
        Icon::Maximize
    };
    if app
        .ui
        .icon_button(
            "win-max",
            max_rect,
            max_icon,
            caption,
            false,
            None,
            CAPTION_ICON,
        )
        .clicked
    {
        app.window_commands.push(WindowCommand::ToggleMaximize);
    }
    if app
        .ui
        .icon_button(
            "win-close",
            close_rect,
            Icon::Close,
            close_style,
            false,
            None,
            CAPTION_ICON,
        )
        .clicked
    {
        app.window_commands.push(WindowCommand::Close);
    }

    // Drag the window by the bar's empty area.
    let on_caption = [min_rect, max_rect, close_rect]
        .iter()
        .any(|r| r.contains(app.mouse));
    let input = app.ui.input();
    let pressed = input.mouse_pressed;
    let held = input.mouse_down;
    if !on_caption && rect.contains(app.mouse) && pressed {
        app.titlebar_drag = true;
        app.window_commands.push(WindowCommand::Move);
    }
    if !held {
        app.titlebar_drag = false;
    }
    app.window_commands
        .push(WindowCommand::SetCursor(cursor_for(app)));
}

/// Cursor affordance for the resize frame / title bar.
fn cursor_for(app: &App) -> WindowCursor {
    if app.layout.titlebar.contains(app.mouse) {
        return WindowCursor::Default;
    }
    let (w, h) = app.screen;
    let frame = 6.0;
    let (x, y) = app.mouse;
    let left = x < frame;
    let right = x > w - frame;
    let top = y < frame && app.mouse.1 > app.layout.titlebar.max().1;
    let bottom = y > h - frame;
    match (left || right, top || bottom) {
        (true, true) => {
            if (left && top) || (right && bottom) {
                WindowCursor::ResizeDiagonal
            } else {
                WindowCursor::ResizeDiagonalAnti
            }
        }
        (true, false) => WindowCursor::ResizeHorizontal,
        (false, true) => WindowCursor::ResizeVertical,
        (false, false) => WindowCursor::Default,
    }
}

/// The resize edge under the pointer, if any (also hit-tests corners).
pub fn resize_edge_under(app: &App) -> Option<ResizeEdge> {
    let (w, h) = app.screen;
    let frame = 6.0;
    let corner = 14.0;
    let (x, y) = app.mouse;
    let left = x < frame;
    let right = x > w - frame;
    let top = y < frame && y > app.layout.titlebar.max().1;
    let bottom = y > h - frame;
    let near_corner = (x < corner || x > w - corner) && (y < corner || y > h - corner);
    if near_corner {
        let leftish = x < w * 0.5;
        let topish = y < h * 0.5;
        return Some(match (leftish, topish) {
            (true, true) => ResizeEdge::TopLeft,
            (false, true) => ResizeEdge::TopRight,
            (true, false) => ResizeEdge::BottomLeft,
            (false, false) => ResizeEdge::BottomRight,
        });
    }
    if left {
        Some(ResizeEdge::Left)
    } else if right {
        Some(ResizeEdge::Right)
    } else if top {
        Some(ResizeEdge::Top)
    } else if bottom {
        Some(ResizeEdge::Bottom)
    } else {
        None
    }
}
