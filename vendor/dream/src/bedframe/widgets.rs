//! Immediate-mode widgets with persistent, animated state.
//!
//! [`Ui`] owns the fonts, glyph atlas and draw list for one frame. Widgets
//! take a stable id (hashed from a scope stack plus a label), keep their
//! interaction state in [`Ui`] across frames, and animate it (hover fades,
//! knob travel) with [`crate::ui::tween`] — so menus and HUDs move without
//! per-widget code.
//!
//! Everything a widget draws, you can draw directly through
//! [`Ui::draw_list_mut`] — widgets are convenience, not a cage.

use super::atlas::AtlasConfig;
use super::draw::{DrawList, UiFrame};
use super::font::GlyphCache;
use super::font::{FontId, Fonts, LayoutOptions};
use super::theme::Theme;
use super::tween::{Anim, Ease};
use super::{Color, Rect};
use std::collections::{HashMap, HashSet};

pub mod adapters;
mod animation;
mod containers;
mod controls;
mod editor;
mod floating;
pub mod headless;
mod hooks;
mod keys;
mod layout_cache;
mod menus;

pub use editor::EditorState;
pub use floating::{Floating, FloatingAlign, FloatingSide, PopoverResponse};
pub use hooks::{Presence, PresencePhase, State};
pub use keys::{Key, KeyFeed};
pub use layout_cache::LayoutStats;
mod overlays;
pub mod paint;
mod pickers;
mod scroll;
mod styles;
mod text;

pub use scroll::{
    ScrollAlign, ScrollBehavior, ScrollConfig, ScrollIntoViewOptions, ScrollMetrics,
    ScrollToOptions,
};
pub(crate) use scroll::{ScrollRegion, ScrollState};
pub use styles::{
    ButtonStyle, IconFit, LabelStyle, MenuItem, MenuStyle, PanelStyle, SegmentedStyle, SliderStyle,
    TextFieldStyle, VCenter,
};

#[cfg(test)]
mod tests;

/// Pointer and keyboard state for one frame. Button states are *edge* states
/// (`pressed`/`released` true for exactly one frame).
#[derive(Clone, Debug, Default)]
pub struct UiInput {
    pub dt: f32,
    /// Pointer position in the same pixel space as the draw list.
    pub mouse: (f32, f32),
    pub mouse_down: bool,
    pub mouse_pressed: bool,
    pub mouse_released: bool,
    /// Characters typed this frame (text fields).
    pub text: String,
    /// Backspace presses this frame.
    pub backspace: u32,
    /// Menu/focus navigation, edge-triggered.
    pub nav: NavInput,
    /// Wheel scroll this frame in pixels (positive = up, toward earlier
    /// content; negative scrolls down), from a line-based wheel. Hosts turn
    /// an OS line delta into pixels with [`ScrollConfig::line_height`]; the
    /// view glides toward the accumulated destination.
    pub wheel: f32,
    /// Precision-wheel scroll this frame in pixels (positive = up), from a
    /// trackpad, touch surface or other pixel-accurate device. Applied 1:1
    /// — the device already produces smooth, inertial deltas — and only
    /// coasts when the gesture stops mid-flight.
    pub wheel_precise: f32,
    /// Right button: `_pressed` is the click edge, `_down` the held state.
    pub mouse_right_pressed: bool,
    pub mouse_right_down: bool,
    /// Paste text offered to the focused field this frame (host clipboard).
    pub paste: Option<String>,
    /// IME composition shown (but not committed) this frame.
    pub preedit: Option<Composition>,
}

/// An in-progress IME composition: the preview text and the composition
/// cursor as byte offsets into it (`None` = the system supplies none).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Composition {
    pub text: String,
    pub cursor: Option<(usize, usize)>,
}

/// Navigation keys, edge-triggered per frame — [`KeyFeed`] fills these
/// (and [`UiInput::paste`]) from raw OS key events.
#[derive(Clone, Copy, Debug, Default)]
pub struct NavInput {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub confirm: bool,
    pub cancel: bool,
    /// Forward delete, `home`/`end` jump the text caret.
    pub delete: bool,
    pub home: bool,
    pub end: bool,
    /// Page keys scroll the hovered view while no widget owns focus.
    pub page_up: bool,
    pub page_down: bool,
    /// Focus traversal (Tab), Shift held for reverse and selection extend.
    pub tab: bool,
    pub shift: bool,
    /// Word-movement modifier (Option on macOS, Ctrl elsewhere).
    pub word: bool,
    /// Clipboard/selection shortcuts routed to the focused editor.
    pub copy: bool,
    pub cut: bool,
    pub select_all: bool,
    /// Undo/redo routed to the focused editor (host sees them only when
    /// no field is editing).
    pub undo: bool,
    pub redo: bool,
}

/// What a widget interaction did this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Response {
    pub rect: Rect,
    /// Clicked (press and release inside) or confirmed via `nav.confirm`.
    pub clicked: bool,
    pub hovered: bool,
    /// Captured by this widget and the pointer is still down.
    pub held: bool,
}

impl Response {
    /// Plain response construction — no drawing. Themed focus rings are
    /// drawn separately by [`Ui::focus_ring`] so a headless behaviour can
    /// return a Response without emitting pixels.
    pub(crate) fn plain(rect: Rect, ui: &Ui, id: u64, clicked: bool) -> Self {
        Self {
            rect,
            clicked,
            hovered: ui.hot_id == Some(id),
            held: ui.active_id == Some(id) && ui.input.mouse_down,
        }
    }

    fn build(rect: Rect, ui: &mut Ui, id: u64, clicked: bool, radius: f32) -> Self {
        ui.focus_ring(rect, radius, id);
        Self::plain(rect, ui, id, clicked)
    }
}

/// Look of a scroll view's thumb. Public field on [`Ui::scrollbar`]; the
/// default matches the stock 5px thumb.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollbarStyle {
    /// Thumb width in px.
    pub width: f32,
    /// Gap between the thumb and the view's right edge, and its top and bottom edges, in px.
    pub inset: f32,
    /// Shortest the thumb may get, in px.
    pub min_thumb: f32,
    /// Thumb colour while the pointer is outside the view.
    pub idle: Color,
    /// Thumb colour while hovering the view or dragging the thumb.
    pub active: Color,
}

impl Default for ScrollbarStyle {
    /// The stock scrollbar: [`Theme::default`]'s themed scrollbar.
    fn default() -> Self {
        Theme::default().scrollbar()
    }
}

/// One registry spring plus its retirement stamp (same retention rules
/// as the keyframe registries).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpringAnim {
    pub spring: crate::ui::tween::Spring,
    pub last_frame: u64,
}

/// A running one-shot pulse (icon engage pop), keyed by widget id.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pulse {
    pub age: f32,
    pub ttl: f32,
    pub last_frame: u64,
}

impl Pulse {
    fn new(ttl: f32, frame: u64) -> Self {
        Self {
            age: 0.0,
            ttl,
            last_frame: frame,
        }
    }
}

/// An interactive surface registered for input routing (a modal card, an
/// open popup). [`Ui`] collects them per frame; [`Ui::hit`] and
/// [`Ui::input_allowed`] consult this and the previous frame's list, so a
/// surface keeps blocking widgets drawn beneath it even on the frames
/// before it is itself drawn.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InputSurface {
    pub id: u64,
    pub rect: Rect,
    pub layer: i32,
    /// A modal blocks input anywhere; other surfaces only cover their rect.
    pub modal: bool,
}

/// What an open clip scope did to the draw state, so the matching `*_end`
/// call restores exactly that (nested panels only ever undo themselves).
#[derive(Debug)]
pub(crate) enum ClipScope {
    /// An ordinary clip push (`panel`, `pixel_panel`).
    Panel,
    /// A collapsible section's reveal clip.
    Section,
    /// A scroll view's clip.
    Scroll,
    /// Clip plus a draw-layer change plus the suspended enclosing clips
    /// (`modal`): ending restores everything the scope borrowed.
    Overlay {
        saved_layer: i32,
        saved_clips: Vec<Rect>,
    },
}

/// Which `*_end` calls may close a [`ClipScope`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClipKind {
    Panel,
    Section,
    Scroll,
    Overlay,
}

impl ClipScope {
    fn kind(&self) -> ClipKind {
        match self {
            ClipScope::Panel => ClipKind::Panel,
            ClipScope::Section => ClipKind::Section,
            ClipScope::Scroll => ClipKind::Scroll,
            ClipScope::Overlay { .. } => ClipKind::Overlay,
        }
    }
}

/// Tooltip dwell accumulator: the rect under the pointer and how long it
/// has been hovered without interruption.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TooltipDwell {
    pub rect: Rect,
    pub elapsed: f32,
    pub last_frame: u64,
}

/// A looping animation's phase and retirement bookkeeping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoopState {
    /// Position in the cycle, 0..1.
    pub phase: f32,
    pub last_frame: u64,
}

/// A focusable widget this frame: id, where it was, its layer, whether it
/// may take focus, and whether it is a text editor (IME target).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Focusable {
    pub id: u64,
    pub rect: Rect,
    #[allow(dead_code)] // diagnostics: the surface layer a focusable sat at
    pub layer: i32,
    pub enabled: bool,
    #[allow(dead_code)] // diagnostics: whether it is an IME-eligible editor
    pub text_field: bool,
}

/// A drag anchor (drag fields, scrollbar thumbs): pointer coordinate and
/// widget value at grab time, plus retirement bookkeeping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DragAnchor {
    pub pos: f32,
    pub value: f32,
    pub last_frame: u64,
}

/// A [`Ui::rate_scope`] accumulator plus retirement bookkeeping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RateBucket {
    pub pending: f32,
    pub last_frame: u64,
}

#[derive(Debug)]
struct WidgetState {
    hover: Anim<f32>,
    /// Frame index this state was last touched (retirement diagnostics).
    last_frame: u64,
}

impl WidgetState {
    fn new() -> Self {
        Self {
            hover: Anim::new(0.0),
            last_frame: 0,
        }
    }
}

/// Immediate-mode UI for one frame.
///
/// ```ignore
/// ui.begin(input, screen_size);
/// let panel = Anchor::Center.place(Rect::new(0.0, 0.0, sw, sh), (320.0, 240.0), 0.0);
/// ui.panel(panel, PanelStyle::default());
/// if ui.button("play", play_rect, "Play", ButtonStyle::new(font)).clicked {
///     // ...
/// }
/// let frame = ui.end();
/// ```
/// Left/right padding for a menu row's label, in pixels, so the text never
/// sits flush against the hover card's edge.
const MENU_TEXT_INSET: f32 = 10.0;
/// Default popup width used by [`Ui::context_menu_rect`].
const CONTEXT_MENU_W: f32 = 184.0;

/// Draw/input layer for popup surfaces (context menus).
pub const LAYER_POPUP: i32 = 1000;
/// Draw/input layer for modal dialogs.
pub const LAYER_MODAL: i32 = 2000;
/// Draw layer for the toast stack (noninteractive).
pub const LAYER_TOAST: i32 = 2500;
/// Draw layer for tooltips (noninteractive).
pub const LAYER_TOOLTIP: i32 = 3000;
/// Devtools overlay layer (outlines/inspector draw above everything).
pub const LAYER_DEVTOOLS: i32 = 4000;

pub struct Ui {
    fonts: Fonts,
    cache: GlyphCache,
    pub(crate) draw: DrawList,
    pub(crate) input: UiInput,
    state: HashMap<u64, WidgetState>,
    /// Id scopes as a running seed: `scopes` is the stack of cumulative
    /// FNV-1a seeds (one per open scope), `scope_seed` the current one.
    /// Hashing an id is O(label) — never O(depth).
    scopes: Vec<u64>,
    scope_seed: u64,
    pub(crate) hot_id: Option<u64>,
    pub(crate) active_id: Option<u64>,
    focus_id: Option<u64>,
    pub(crate) screen: (f32, f32),
    /// Frame counter (retirement of stale per-id state).
    pub(crate) frame: u64,
    /// Vector icons baked into the atlas.
    icons: super::icons::IconSet,
    /// Bounded physical-layout cache (see `layout_cache`).
    layouts: layout_cache::LayoutCache,
    /// Atlas epoch the UI last observed — `GlyphCache::last_epoch` is
    /// updated by `reset_atlas` itself, so the cache's own bookkeeping
    /// can't tell a direct `reset_atlas` apart from "no reset". This
    /// independent observation catches BOTH reset paths.
    atlas_epoch: u64,
    /// Bumped whenever `fonts_mut` hands out write access: the layout cache
    /// keys on it so new faces invalidate old results.
    fonts_rev: u64,
    /// Persistent keyframe animations, keyed like widget ids.
    anims_f32: HashMap<u64, Anim<f32>>,
    anims_color: HashMap<u64, Anim<Color>>,
    /// Persistent spring animations, keyed like widget ids.
    anims_spring: HashMap<u64, SpringAnim>,
    /// Scroll-view physics for every view built this frame (public field:
    /// set it to whatever feel the app wants; nothing reads it but the
    /// scroll views).
    pub scroll_config: ScrollConfig,
    /// Scrollbar thumb look for every scroll view built this frame.
    pub scrollbar: ScrollbarStyle,
    /// When set, widget input is confined to this rect (an open modal card).
    blocked: Option<Rect>,
    /// Wheel-scroll offsets for scroll views, keyed by id.
    pub(crate) scrolls: HashMap<u64, ScrollState>,
    /// Content height each scroll view reported on its last frame, keyed
    /// like `scrolls` — the extent [`Ui::scroll_state`] reports so a host
    /// can pin a transcript to its newest row.
    pub(crate) scroll_contents: HashMap<u64, f32>,
    /// Drag anchors for `drag_field` and scrollbar thumbs, keyed by id.
    pub(crate) drags: HashMap<u64, DragAnchor>,
    /// Scrollbars waiting to be drawn by `scroll_end`: a scroll view paints
    /// its thumb over its content, not under it.
    pub(crate) scroll_thumbs: Vec<(u64, Rect, f32, f32)>,
    /// Wheel remainder passed from a nested scroll view to its parent
    /// (scroll chaining), with the frame it was written on: the parent picks
    /// it up on the next build, and an entry nobody claims expires.
    pub(crate) scroll_chain: HashMap<u64, (f32, u64)>,
    /// One-shot pulses (icon engage pop), keyed by id.
    pulses: HashMap<u64, Pulse>,
    /// Menus whose highlighted row is the keyboard's, not the pointer's
    /// (id -> last frame touched). A menu is absent until an arrow key moves
    /// its focus and absent again once the pointer enters it or it starts
    /// closing, so an untouched menu shows no highlight at all.
    menu_keyboard: HashMap<u64, u64>,
    /// Looping animations (caret blink, spinners), keyed like widget ids.
    loops: HashMap<u64, LoopState>,
    /// Interactive overlay surfaces registered this frame and last frame
    /// (input gating: [`Self::hit`], [`Self::input_allowed`]).
    surfaces_cur: Vec<InputSurface>,
    surfaces_prev: Vec<InputSurface>,
    /// Clip scopes opened this frame, innermost last (see [`ClipScope`]).
    clip_scopes: Vec<ClipScope>,
    /// Where the focused widget sat when it last ran (outside-press blur).
    focus_rect: Option<Rect>,
    /// Whether the focused widget ran this frame (dead-id rejection).
    focus_seen: bool,
    /// The focused id when it is a text editor (vs. a focused button):
    /// IME composition is offered only while this is the focused widget.
    editing_id: Option<u64>,
    /// The focused editor's caret rect this frame (IME placement).
    editing_caret: Option<Rect>,
    /// Between `begin` and `end`: recycling a finished frame while the
    /// next is mid-build would corrupt it (debug-guarded).
    building: bool,
    /// Tooltip dwell state (see [`Ui::tooltip`]).
    tooltip_dwell: Option<TooltipDwell>,
    /// Scroll views registered this frame and last frame, in call order —
    /// later entries are nested inside earlier ones, so the deepest region
    /// under the pointer owns the wheel.
    scroll_cur: Vec<ScrollRegion>,
    /// Toast ids rendered last frame / this frame: a toast only ages once
    /// it has spent a frame on screen, so a notification pushed during a
    /// long idle gap does not arrive pre-expired.
    toasts_prev: HashSet<u64>,
    toasts_cur: HashSet<u64>,
    scroll_prev: Vec<ScrollRegion>,
    /// Whether this frame's wheel was already consumed (a wheel tick is
    /// spent exactly once, by the deepest eligible view).
    wheel_taken: bool,
    /// Whether this frame's press already captured a widget: once a press
    /// is claimed, no other widget may acquire it — even a press and
    /// release landing in the same frame.
    pub(crate) press_consumed: bool,
    /// Earliest moment a pending clock (tooltip dwell, toast expiry) needs
    /// the host to build a frame; `None` when nothing is waiting.
    pub(crate) wake_in: Option<std::time::Duration>,
    /// UI scale: logical units per physical pixel ratio — `begin`'s input
    /// and screen are logical, `UiFrame` is physical.
    scale: f32,
    /// Ambient theme: the colours for widgets that take no explicit style
    /// (sections, toasts, tooltips, scrollbars, splitters, drag fields). Set
    /// once per frame with [`Self::set_theme`].
    theme: Theme,
    /// Animation time for the scopes currently open, innermost last. Empty
    /// means "advance by the real frame time" (the display's own rate).
    rate_stack: Vec<f32>,
    /// Per-scope accumulators for [`Self::rate_scope`], keyed by scope id.
    rates: HashMap<u64, RateBucket>,
    /// Shared text metrics the unstyled widgets read from.
    text_metrics: UiTextMetrics,
    /// Per-field editor state (caret/selection/undo), keyed by widget id.
    editors: HashMap<u64, editor::EditorState>,
    /// Retained undo bytes across all fields — the global 2 MiB bound.
    edit_history_bytes: usize,
    /// Text a field asked the host to put on the clipboard (copy/cut).
    clipboard_out: Option<String>,
    /// Segment cell scratch (reused each frame — no per-frame Vec).
    pub(crate) seg_cells: Vec<(f32, f32)>,
    hooks: hooks::Hooks,
    /// Segment hit results (reused each frame — no per-frame Vec).
    pub(crate) seg_scratch: Vec<(Rect, Response, crate::ui::widgets::headless::State)>,
    /// Devtools overlay mode ([`crate::ui::devtools`]).
    pub(crate) devtools: crate::ui::devtools::DevtoolsMode,
    /// L3 declarative view layer state (see [`crate::ui::view`]).
    pub(crate) view: crate::ui::view::ViewState,
    /// Focusables this frame in submission order (for Tab traversal).
    focusables_cur: Vec<Focusable>,
    focusables_prev: Vec<Focusable>,
    /// Whether focused widgets draw the accent focus ring — opt-in via
    /// [`Self::set_focus_rings`], off by default.
    focus_rings: bool,
    /// Focus-visible: set when the focused widget was reached by keyboard,
    /// cleared by a pointer press.
    focus_visible: bool,
    /// Multiplier applied to every *authored* animation duration:
    /// `1.0` keeps the timings as written, `0` makes motion instant, and
    /// values in between slow it down without changing any curve. Set
    /// through [`Self::set_motion_scale`]; the default is `1.0`, so the
    /// framework asserts no policy of its own.
    motion_scale: f32,
}

/// Shared text metrics for widgets that don't take an explicit style —
/// every hardcoded `FontId(0)`/size/pad in overlays and containers reads
/// here, so a host reskins the defaults in one call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiTextMetrics {
    /// Face for unstyled widget text.
    pub font: FontId,
    /// Body/section/toast text size (logical px).
    pub body: f32,
    /// Tooltip text size.
    pub tooltip: f32,
    /// Small badge/hint text size.
    pub badge: f32,
    /// Toast horizontal/vertical padding.
    pub toast_pad_x: f32,
    pub toast_pad_y: f32,
    /// Menu row padding.
    pub menu_pad: f32,
}

impl UiTextMetrics {
    /// Derive the shared metrics from the type scale: body/tooltip/badge
    /// sizes come from the Body/Small/Caption roles (badge stays 9 —
    /// Caption − 2) and the font is the family's regular face.
    pub fn from_tokens(tokens: &crate::ui::tokens::Tokens, fonts: &Fonts) -> Self {
        Self {
            font: fonts.resolve(crate::ui::tokens::FontWeight::Regular, false),
            body: tokens.type_style(crate::ui::tokens::TypeRole::Body).size,
            tooltip: tokens.type_style(crate::ui::tokens::TypeRole::Small).size,
            badge: tokens.type_style(crate::ui::tokens::TypeRole::Caption).size - 2.0,
            ..Default::default()
        }
    }
}

impl Default for UiTextMetrics {
    fn default() -> Self {
        Self {
            font: FontId(0),
            body: 13.0,
            tooltip: 12.0,
            badge: 9.0,
            toast_pad_x: 12.0,
            toast_pad_y: 9.0,
            menu_pad: 8.0,
        }
    }
}

impl Ui {
    /// `fonts` supplies the faces; `atlas_config` sizes the glyph atlas pages.
    pub fn new(fonts: Fonts, atlas_config: AtlasConfig) -> Self {
        let theme = Theme::dark_theme();
        let text_metrics = UiTextMetrics::from_tokens(&theme.tokens, &fonts);
        let mut ui = Self {
            fonts,
            cache: GlyphCache::new(super::atlas::Atlas::new(atlas_config)),
            draw: DrawList::new(),
            input: UiInput::default(),
            state: HashMap::new(),
            scopes: Vec::new(),
            scope_seed: FNV_OFFSET,
            hot_id: None,
            active_id: None,
            focus_id: None,
            screen: (0.0, 0.0),
            frame: 0,
            icons: super::icons::IconSet::new(),
            layouts: layout_cache::LayoutCache::default(),
            fonts_rev: 0,
            anims_f32: HashMap::new(),
            anims_color: HashMap::new(),
            anims_spring: HashMap::new(),
            scroll_config: ScrollConfig::default(),
            scrollbar: ScrollbarStyle::default(),
            blocked: None,
            scrolls: HashMap::new(),
            scroll_contents: HashMap::new(),
            drags: HashMap::new(),
            scroll_thumbs: Vec::new(),
            scroll_chain: HashMap::new(),
            pulses: HashMap::new(),
            menu_keyboard: HashMap::new(),
            loops: HashMap::new(),
            surfaces_cur: Vec::new(),
            surfaces_prev: Vec::new(),
            clip_scopes: Vec::new(),
            focus_rect: None,
            focus_seen: false,
            editing_id: None,
            editing_caret: None,
            building: false,
            tooltip_dwell: None,
            scroll_cur: Vec::new(),
            toasts_prev: HashSet::new(),
            toasts_cur: HashSet::new(),
            scroll_prev: Vec::new(),
            wheel_taken: false,
            press_consumed: false,
            wake_in: None,
            scale: 1.0,
            motion_scale: 1.0,
            theme,
            rate_stack: Vec::new(),
            rates: HashMap::new(),
            text_metrics,
            editors: HashMap::new(),
            edit_history_bytes: 0,
            clipboard_out: None,
            seg_cells: Vec::new(),
            hooks: hooks::Hooks::default(),
            seg_scratch: Vec::new(),
            devtools: crate::ui::devtools::DevtoolsMode::Off,
            view: crate::ui::view::ViewState::default(),
            focusables_cur: Vec::new(),
            focusables_prev: Vec::new(),
            focus_rings: false,
            focus_visible: false,
            atlas_epoch: 0,
        };
        ui.atlas_epoch = ui.cache.epoch();
        ui
    }

    /// Shared text metrics for unstyled widgets (see [`UiTextMetrics`]).
    pub fn text_metrics(&self) -> UiTextMetrics {
        self.text_metrics
    }

    /// Override the shared text metrics.
    pub fn set_text_metrics(&mut self, metrics: UiTextMetrics) {
        self.text_metrics = metrics;
    }

    /// Adopt `theme` for every widget that does not take an explicit style.
    ///
    /// Widgets with a style parameter are skinned by their caller; the rest
    /// read these colours, so a theme swap re-skins the whole UI — including
    /// section headers, toasts, menus' disabled rows and tooltips — with no
    /// widget changes.
    pub fn set_theme(&mut self, theme: &Theme) {
        // A scrollbar the user never customised follows the new theme; a
        // customised one is left alone.
        if self.scrollbar == self.theme.scrollbar() {
            self.scrollbar = theme.scrollbar();
        }
        // Same trick for the derived text metrics: they follow the theme's
        // tokens unless the host set its own.
        if self.text_metrics == UiTextMetrics::from_tokens(&self.theme.tokens, &self.fonts) {
            self.text_metrics = UiTextMetrics::from_tokens(&theme.tokens, &self.fonts);
        }
        self.theme = *theme;
    }

    /// The face a type role resolves to (its weight/mono against the
    /// registered family, `FontId(0)` without one).
    pub fn role_font(&self, role: crate::ui::tokens::TypeRole) -> FontId {
        let style = self.theme.tokens.type_style(role);
        self.fonts.resolve(style.weight, style.mono)
    }

    /// A [`LabelStyle`] for a type role: the family's weight/mono face, the
    /// role's size and line height, theme text colour (muted for Caption),
    /// optically vcentred.
    pub fn type_style(&self, role: crate::ui::tokens::TypeRole) -> LabelStyle {
        let ts = self.theme.tokens.type_style(role);
        LabelStyle::new(self.role_font(role))
            .size(ts.size)
            .line_height(ts.line_height / ts.size)
            .color(if role == crate::ui::tokens::TypeRole::Caption {
                self.theme.muted
            } else {
                self.theme.fg
            })
            .vcenter(true)
    }

    /// Draw `elevation`'s shadow stack under `rect` — the token's one or
    /// two shadows, each passed to [`DrawList::shadow`].
    pub fn elevation_shadow(
        &mut self,
        rect: Rect,
        radius: f32,
        elevation: crate::ui::tokens::Elevation,
    ) {
        for s in self
            .theme
            .tokens
            .elevation(elevation)
            .shadows
            .iter()
            .flatten()
        {
            self.draw
                .shadow(rect, radius, s.offset, s.spread, s.blur, s.color);
        }
    }

    /// [`Self::elevation_shadow`] with the token colours' alpha scaled by
    /// `fade` (open/close animation).
    pub fn elevation_shadow_faded(
        &mut self,
        rect: Rect,
        radius: f32,
        elevation: crate::ui::tokens::Elevation,
        fade: f32,
    ) {
        for s in self
            .theme
            .tokens
            .elevation(elevation)
            .shadows
            .iter()
            .flatten()
        {
            self.draw
                .shadow(rect, radius, s.offset, s.spread, s.blur, s.color.fade(fade));
        }
    }

    /// The theme in force for this frame.
    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    /// Whether the focused widget shows its ring — true when focus was
    /// last acquired by keyboard, false after a pointer press.
    pub fn focus_visible(&self) -> bool {
        self.focus_visible
    }

    /// Turn the keyboard-navigation focus ring on or off for every widget:
    /// when on, the focused widget gets a `theme.accent` outline. Off by
    /// default — hosts that want it opt in once at setup.
    pub fn set_focus_rings(&mut self, enabled: bool) {
        self.focus_rings = enabled;
    }

    /// Whether focused widgets draw the accent focus ring (default off).
    pub fn focus_rings(&self) -> bool {
        self.focus_rings
    }

    /// Set how much motion this UI performs: `1.0` plays everything exactly
    /// as the widgets authored it, smaller values do less of it, and `0`
    /// removes motion entirely while still changing state — eases land on
    /// their targets with no travel, springs snap, loops hold their phase.
    ///
    /// Between the endpoints the scale compresses each authored *duration*
    /// (`0.5` completes the same curve in half the time, so it is less
    /// motion rather than slower motion) and stretches loops (`0.5` runs
    /// half the cycles — repetitive motion slows instead of strobing);
    /// springs settle in `scale`× their natural time on the same rule,
    /// with stiffness, damping and mass left exactly as the app tuned them.
    ///
    /// This is the hook a host uses for an OS "reduce motion" preference:
    /// the framework ships the knob and the plumbing, not the policy —
    /// nothing here decides for an app when to turn it down. Durations
    /// passed to the `anim_*` helpers (widgets, popups, toasts, sections)
    /// are scaled inside those helpers, so callers keep passing authored
    /// values and never scale them by hand.
    pub fn set_motion_scale(&mut self, scale: f32) {
        self.motion_scale = if scale.is_finite() {
            scale.clamp(0.0, 100.0)
        } else {
            1.0
        };
    }

    /// The motion scale in force for this frame (default `1.0`).
    pub fn motion_scale(&self) -> f32 {
        self.motion_scale
    }

    /// The UI scale: logical units are multiplied by this to get physical
    /// pixels. Input, widget rects and `begin`'s screen size are always
    /// logical; the [`UiFrame`] `end` produces is physical. Values must be
    /// finite and positive — anything else keeps (or resets to) 1.0, so a
    /// degenerate scale factor can never poison a frame.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
    }

    /// The current UI scale (see [`Self::set_scale`]). Default 1.0.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// How long until a pending UI clock (a tooltip mid-dwell, a toast
    /// approaching expiry) needs the host to build a frame. `None` means
    /// nothing is waiting — the host may park until input arrives. A
    /// *pending* clock does not count as animation: [`Self::animating`]
    /// stays false while the UI waits on wall time, and this deadline is
    /// what wakes it. Fed by real frame `dt`, so a host must pass the true
    /// elapsed time on the woken frame or the deadline just reschedules.
    pub fn next_wake_in(&self) -> Option<std::time::Duration> {
        self.wake_in
    }

    /// Ask for a frame no later than `after` seconds from now (the minimum
    /// of all requests this frame wins).
    pub(crate) fn request_wake(&mut self, after: f32) {
        if !after.is_finite() || after < 0.0 {
            return;
        }
        let after = std::time::Duration::from_secs_f32(after);
        self.wake_in = Some(self.wake_in.map_or(after, |cur| cur.min(after)));
    }

    /// Cancel everything a lost focus event should end: the pointer
    /// capture, drag anchors and keyboard focus — a window that loses
    /// focus must not keep dragging a slider it can no longer see.
    pub fn cancel_input(&mut self) {
        self.active_id = None;
        self.drags.clear();
        self.focus_id = None;
        self.focus_rect = None;
    }

    /// Where the focused widget sat last frame (IME caret placement).
    pub fn focused_rect(&self) -> Option<Rect> {
        self.focus_rect
    }

    /// Whether the focused widget is a text editor: the host offers IME
    /// composition only then — a focused button is not an editing target.
    pub fn editing(&self) -> bool {
        self.editing_id.is_some() && self.editing_id == self.focus_id
    }

    /// The focused editor's caret rect (IME candidate placement): the
    /// field reports its caret subrect while it draws, so the host's IME
    /// window lands at the insertion point, not the whole field.
    pub fn editing_rect(&self) -> Option<Rect> {
        if !self.editing() {
            return None;
        }
        self.editing_caret.or(self.focus_rect)
    }

    /// The focused editor reports its caret rect this frame.
    pub(crate) fn note_editing_caret(&mut self, rect: Rect) {
        self.editing_caret = Some(rect);
    }

    /// Text a focused editor asked the host to clipboard (copy/cut); the
    /// host drains it once — the field never sees its own output twice.
    pub fn take_clipboard_output(&mut self) -> Option<String> {
        self.clipboard_out.take()
    }

    /// The default drain for the copy/cut seam: write any pending output
    /// to the OS clipboard. Hosts that intercept the seam keep
    /// [`Self::take_clipboard_output`].
    pub fn flush_clipboard(&mut self) {
        if let Some(text) = self.take_clipboard_output() {
            crate::ui::clipboard::set_text(&text);
        }
    }

    /// Start a frame. Call once before any widget. `input` and
    /// `screen_size` are in logical units — the frame [`Ui::end`] returns
    /// is physical (`screen * scale`, see [`Self::set_scale`]).
    pub fn begin(&mut self, input: UiInput, screen_size: (f32, f32)) {
        self.hooks.begin();
        // L3: this frame's solved rects become next frame's hit rects.
        std::mem::swap(&mut self.view.prev_rects, &mut self.view.cur_rects);
        self.view.cur_rects.clear();
        std::mem::swap(&mut self.view.prev_content, &mut self.view.cur_content);
        self.view.cur_content.clear();
        self.view.tree_cursor = 0;
        self.input = input;
        // Animation and rate clocks must never see a bogus dt: NaN would
        // poison every animation it touched forever, and a negative step
        // runs time backwards.
        if !self.input.dt.is_finite() || self.input.dt < 0.0 {
            self.input.dt = 0.0;
        }
        // An outside press blurs the focused widget *before* any editing
        // or navigation runs: the press that ends the focus is not also a
        // confirmation fed to the widget being left.
        if self.input.mouse_pressed
            && self.focus_id.is_some()
            && self
                .focus_rect
                .is_some_and(|rect| !rect.contains(self.input.mouse))
        {
            // A blurred editor drops its selection too — a stale highlight
            // on an unfocused field reads as still-editing.
            if let Some(state) = self.focus_id.and_then(|id| self.editors.get_mut(&id)) {
                state.collapse(state.caret);
            }
            self.focus_id = None;
            self.focus_rect = None;
        }
        // A mid-frame atlas overflow recovers here: new epoch, glyph and
        // icon caches rebake on demand, layout keys carry the epoch.
        if self.cache.overflow_pending() {
            self.cache.reset_atlas();
            self.icons.clear();
            self.layouts = layout_cache::LayoutCache::default();
        }
        self.screen = screen_size;
        self.draw.begin(screen_size);
        self.draw.set_scale(self.scale);
        self.scopes.clear();
        self.scope_seed = FNV_OFFSET;
        self.hot_id = None;
        self.blocked = None;
        self.focus_seen = false;
        self.rate_stack.clear();
        self.clip_scopes.clear();
        self.press_consumed = false;
        crate::ui::devtools::devtools_begin(self);
        self.wheel_taken = false;
        // A view whose `scroll_end` never ran (an early return in caller
        // code) leaves its thumb here; the next frame starts clean.
        self.scroll_thumbs.clear();
        self.wake_in = None;
        // Swap-and-clear: the previous frame's registries swap into
        // `*_prev` (capacity preserved in both halves — no realloc).
        std::mem::swap(&mut self.surfaces_prev, &mut self.surfaces_cur);
        self.surfaces_cur.clear();
        std::mem::swap(&mut self.scroll_prev, &mut self.scroll_cur);
        self.scroll_cur.clear();
        std::mem::swap(&mut self.toasts_prev, &mut self.toasts_cur);
        self.toasts_cur.clear();
        std::mem::swap(&mut self.focusables_prev, &mut self.focusables_cur);
        self.focusables_cur.clear();
        self.editing_id = None;
        self.editing_caret = None;
        self.frame += 1;
        // A chained wheel remainder lives exactly one build: written on
        // frame N, claimed on N+1. Anything older has no view left to hear
        // it.
        let frame = self.frame;
        self.scroll_chain
            .retain(|_, &mut (_, written)| written + 1 >= frame);
        self.building = true;
        // Any atlas reset — overflow-driven `reset_atlas` OR a raw
        // `Atlas::reset` — invalidates glyph maps, icon regions and the
        // layout cache. The cache's own epoch bookkeeping handles the
        // glyph maps; icons/layouts compare against the UI's own
        // observation (a direct `reset_atlas` already moved the cache's
        // marker, so it can't tell the two apart).
        self.cache.sync_epoch();
        if self.cache.epoch() != self.atlas_epoch {
            self.atlas_epoch = self.cache.epoch();
            self.icons.clear();
            self.layouts = layout_cache::LayoutCache::default();
        }
        if self.input.mouse_pressed || self.input.mouse_right_pressed {
            self.focus_visible = false;
        }
        // Tab traversal: pick the next focusable in submission order
        // (Shift reverses), skipping disabled, clipped or blocked entries
        // — a widget that did not run cannot be tabbed to.
        if self.input.nav.tab {
            self.tab_focus();
        }
    }

    /// Move focus to the next (or previous, with Shift) focusable in the
    /// order widgets submitted last frame. A first Tab focuses the first.
    fn tab_focus(&mut self) {
        let backward = self.input.nav.shift;
        let list = &self.focusables_prev;
        if list.is_empty() {
            return;
        }
        let current = self
            .focus_id
            .and_then(|id| list.iter().position(|f| f.id == id));
        let n = list.len();
        let eligible = |i: usize, this: &Ui| {
            let focusable = list[i];
            focusable.enabled
                && this.input_allowed_at(focusable.id, focusable.rect, focusable.layer)
        };
        let start = match current {
            Some(pos) => pos,
            None => usize::MAX,
        };
        for step in 1..=n {
            let i = match start {
                usize::MAX => {
                    if backward {
                        n - step
                    } else {
                        step - 1
                    }
                }
                pos => {
                    if backward {
                        (pos + n - step) % n
                    } else {
                        (pos + step) % n
                    }
                }
            };
            if eligible(i, self) {
                self.focus_id = Some(list[i].id);
                self.focus_rect = Some(list[i].rect);
                self.focus_seen = true;
                self.focus_visible = true;
                return;
            }
        }
    }

    /// Finish the frame; hand the result to `Renderer::render_ui`. Collects
    /// any atlas pages that gained glyphs so the renderer can upload them.
    ///
    /// Takes `&mut self` so one [`Ui`] (and its glyph cache) lives across the
    /// whole application: `begin` / widgets / `end` per frame.
    pub fn end(&mut self) -> UiFrame {
        let screen = self.screen;
        // The renderer gets the immutable snapshot, not a destructive
        // dirty list: a frame retained (or rendered out of order) still
        // carries the pixels it was built against.
        let snapshot = self.cache.atlas_mut().snapshot();
        // Central release: the pointer is up, so whatever captured it lets
        // go — even when the widget vanished mid-drag.
        if !self.input.mouse_down {
            self.active_id = None;
        }
        // A focused widget that did not run this frame is dead — it must
        // not keep swallowing keys forever.
        if self.focus_id.is_some() && !self.focus_seen {
            self.focus_id = None;
            self.focus_rect = None;
        }
        if self.frame.is_multiple_of(60) {
            self.retire_stale_state();
        }
        self.enforce_edit_history_bound();
        let mut frame = std::mem::take(&mut self.draw).finish(screen);
        frame.atlas_snapshot = Some(snapshot);
        self.building = false;
        frame
    }

    /// Hand a finished frame back so its geometry buffers can be reused
    /// for the next [`Self::begin`]: the draw list keeps whichever
    /// allocation of the two is larger. Recycling while a frame is being
    /// built would lose live geometry, so that is a logic error caught in
    /// debug builds.
    pub fn recycle_frame(&mut self, frame: UiFrame) {
        debug_assert!(
            !self.building,
            "recycling a frame while the next one is being built"
        );
        self.draw.recycle(frame);
    }

    /// Retire per-id state, carets and animations that no widget has touched
    /// for a few seconds. Ids come from labels, so an editor that creates
    /// and destroys dynamic rows (asset lists, history stacks) would grow
    /// these maps forever without this sweep.
    fn retire_stale_state(&mut self) {
        self.hooks.retire(self.frame);
        const STALE_FRAMES: u64 = 600; // ~10 s at 60 fps, more when idle
        let frame = self.frame;
        self.state
            .retain(|_, state| frame.saturating_sub(state.last_frame) < STALE_FRAMES);
        self.anims_f32
            .retain(|_, anim| frame.saturating_sub(anim.last_frame()) < STALE_FRAMES);
        self.anims_color
            .retain(|_, anim| frame.saturating_sub(anim.last_frame()) < STALE_FRAMES);
        self.anims_spring
            .retain(|_, anim| frame.saturating_sub(anim.last_frame) < STALE_FRAMES);
        self.scrolls
            .retain(|_, scroll| frame.saturating_sub(scroll.last_frame) < STALE_FRAMES);
        // Content extents only make sense next to a live scroll state.
        self.scroll_contents
            .retain(|id, _| self.scrolls.contains_key(id));
        self.loops
            .retain(|_, state| frame.saturating_sub(state.last_frame) < STALE_FRAMES);
        self.pulses
            .retain(|_, pulse| frame.saturating_sub(pulse.last_frame) < STALE_FRAMES);
        self.drags
            .retain(|_, drag| frame.saturating_sub(drag.last_frame) < STALE_FRAMES);
        self.view
            .scroll
            .retain(|_, s| frame.saturating_sub(s.1) < STALE_FRAMES);
        self.rates
            .retain(|_, rate| frame.saturating_sub(rate.last_frame) < STALE_FRAMES);
        self.menu_keyboard
            .retain(|_, touched| frame.saturating_sub(*touched) < STALE_FRAMES);
        self.editors
            .retain(|_, state| frame.saturating_sub(state.last_frame) < STALE_FRAMES);
        if self.editors.capacity() > 1024 && self.editors.len() * 4 < self.editors.capacity() {
            self.editors.shrink_to_fit();
        }
        // Hard bound: a pathological id space (unstable labels) cannot grow
        // these maps without limit. Stale entries retire first; entries
        // touched *this* frame and the live active/focus ids are never
        // evicted — if a UI really has more current ids than the bound,
        // keeping them is the honest answer (an active working set is not
        // leak state).
        const MAX_ENTRIES: usize = 4096;
        let frame_now = self.frame;
        let active = self.active_id;
        let focus = self.focus_id;
        let protected = |id: u64| Some(id) == active || Some(id) == focus;
        fn evict_eldest<V>(
            map: &mut HashMap<u64, V>,
            last: impl Fn(&V) -> u64,
            protected: &dyn Fn(u64) -> bool,
            frame_now: u64,
        ) {
            if map.len() > MAX_ENTRIES {
                let removals = map.len() - MAX_ENTRIES;
                let mut ages: Vec<(u64, u64)> = map
                    .iter()
                    .filter(|&(&id, value)| last(value) != frame_now && !protected(id))
                    .map(|(&id, value)| (id, last(value)))
                    .collect();
                if removals < ages.len() {
                    ages.select_nth_unstable_by_key(removals - 1, |&(_, last_frame)| last_frame);
                    ages.truncate(removals);
                }
                for (id, _) in ages {
                    map.remove(&id);
                }
            }
            // A historical peak should not pin capacity forever.
            if map.capacity() > 1024 && map.len() * 4 < map.capacity() {
                map.shrink_to_fit();
            }
        }
        let p = &protected;
        evict_eldest(&mut self.state, |s| s.last_frame, p, frame_now);
        evict_eldest(&mut self.anims_f32, |a| a.last_frame(), p, frame_now);
        evict_eldest(&mut self.anims_color, |a| a.last_frame(), p, frame_now);
        evict_eldest(&mut self.anims_spring, |a| a.last_frame, p, frame_now);
        evict_eldest(&mut self.scrolls, |s| s.last_frame, p, frame_now);
        evict_eldest(&mut self.loops, |s| s.last_frame, p, frame_now);
        evict_eldest(&mut self.pulses, |p_| p_.last_frame, p, frame_now);
        evict_eldest(&mut self.drags, |d| d.last_frame, p, frame_now);
        evict_eldest(&mut self.rates, |r| r.last_frame, p, frame_now);
        evict_eldest(&mut self.menu_keyboard, |t| *t, p, frame_now);
    }

    /// The fonts registry (load more faces any time).
    pub fn fonts(&self) -> &Fonts {
        &self.fonts
    }

    pub fn fonts_mut(&mut self) -> &mut Fonts {
        // A mutation could load a face — old layouts keyed on the previous
        // registry are no longer trustworthy.
        self.fonts_rev += 1;
        &mut self.fonts
    }

    /// Layout-cache diagnostics: hit/miss counts and retained byte
    /// estimate — raw numbers for the bench harness, not a memory claim.
    pub fn layout_stats(&self) -> LayoutStats {
        self.layouts.stats()
    }

    /// The glyph cache / atlas pair (icon packing, direct glyph access).
    pub fn cache(&self) -> &GlyphCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut GlyphCache {
        &mut self.cache
    }

    /// Raw draw access — the escape hatch for custom geometry.
    pub fn draw_list_mut(&mut self) -> &mut DrawList {
        &mut self.draw
    }

    // The shape primitives hosts reach for most, so an app does not have to
    // go through `draw_list_mut()` for a background or a border. These are
    // paint-only: clipping here does not register an input surface, so UI
    // chrome should still use `panel`/`scroll_begin` where input matters.

    /// Draw a rounded rectangle.
    pub fn paint_round_rect(&mut self, rect: Rect, radius: f32, color: Color) {
        self.draw.round_rect(rect, radius, color);
    }

    /// Draw a rounded rectangle outline `thickness` pixels wide.
    pub fn paint_round_ring(&mut self, rect: Rect, radius: f32, thickness: f32, color: Color) {
        self.draw.round_ring(rect, radius, thickness, color);
    }

    /// Draw an arc of a ring: `sweep` radians from `start`.
    pub fn paint_ring_arc(
        &mut self,
        rect: Rect,
        radius: f32,
        thickness: f32,
        start: f32,
        sweep: f32,
        color: Color,
    ) {
        self.draw
            .ring_arc(rect, radius, thickness, start, sweep, color);
    }

    /// Clip subsequent paint to `rect` (paint-only; pair with
    /// [`Self::paint_pop_clip`]).
    pub fn paint_push_clip(&mut self, rect: Rect) {
        self.draw.push_clip(rect);
    }

    /// Pop the clip pushed by [`Self::paint_push_clip`].
    pub fn paint_pop_clip(&mut self) {
        self.draw.pop_clip();
    }

    pub fn input(&self) -> &UiInput {
        &self.input
    }

    pub fn screen_size(&self) -> (f32, f32) {
        self.screen
    }

    // --- Ids ---------------------------------------------------------------

    /// All ids until [`Self::pop_id`] nest under `label`. Use around repeated
    /// widget sets (inventory slots, menu pages) so identical labels stay
    /// distinct.
    pub fn push_id(&mut self, label: &str) {
        self.scope_seed = fnv_push(self.scope_seed, label);
        self.scopes.push(self.scope_seed);
    }

    pub fn pop_id(&mut self) {
        self.scopes.pop();
        self.scope_seed = self.scopes.last().copied().unwrap_or(FNV_OFFSET);
    }

    /// Numeric scope key — the cheap [`Self::push_id`] for ids that are
    /// already numbers (toast serials, animation channels), no formatting.
    pub(crate) fn push_id_num(&mut self, key: u64) {
        let mut hash = self.scope_seed;
        for byte in key.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        self.scope_seed = hash;
        self.scopes.push(self.scope_seed);
    }

    pub(crate) fn id(&self, label: &str) -> u64 {
        fnv_push(self.scope_seed, label)
    }

    fn hover_level(&mut self, id: u64, hovered: bool) -> f32 {
        if hovered {
            self.hot_id = Some(id);
        }
        let frame = self.frame;
        let dt = self.anim_dt();
        let state = self.state.entry(id).or_insert_with(WidgetState::new);
        state.last_frame = frame;
        // Retarget only on change: a settled hover must stay *finished* so
        // `Ui::animating` can go false and the app can idle.
        let target = if hovered { 1.0 } else { 0.0 };
        if state.hover.target() != target {
            let motion = self.theme.tokens.motion;
            let (duration, ease) = if hovered {
                (motion.hover_in, Ease::CubicOut)
            } else {
                (motion.hover_out, Ease::CubicOut)
            };
            state.hover.animate_to(target, duration, ease);
        }
        state.hover.update(dt)
    }

    // --- Input routing ------------------------------------------------------

    /// Whether the pointer over `rect` reaches a widget there: inside the
    /// rect, inside the clip currently in force, not confined away by a host
    /// [`Self::set_input_rect`], and not covered by a higher-layer input
    /// surface — a popup under the pointer, or a modal anywhere. Surfaces
    /// register as they are drawn; the previous frame's list is consulted
    /// too so an overlay blocks the widgets beneath it on every frame it is
    /// up, not just the ones drawn after it.
    pub(crate) fn hit(&self, rect: Rect) -> bool {
        // Pointer hit: the *pointer's* position decides — never the
        // widget rect's centre (a half-covered widget still answers the
        // half that's reachable).
        if !rect.contains(self.input.mouse) {
            return false;
        }
        if let Some(card) = self.blocked
            && !card.contains(self.input.mouse)
        {
            // Confined input tests the pointer, not the widget's centre:
            // a rect that straddles the card edge answers only inside it.
            return false;
        }
        if let Some(clip) = self.draw.clip_rect()
            && !clip.contains(self.input.mouse)
        {
            return false;
        }
        !self.surface_covers(self.draw.layer(), self.input.mouse)
    }

    /// Whether a higher-layer interactive surface covers `point` from a
    /// widget at `layer`: a modal at a higher layer covers everywhere,
    /// anything else covers only its own rect. Both the surfaces drawn so
    /// far this frame and the previous frame's list count.
    pub(crate) fn surface_covers(&self, layer: i32, point: (f32, f32)) -> bool {
        let covered = |surface: &InputSurface| {
            surface.layer > layer && (surface.modal || surface.rect.contains(point))
        };
        self.surfaces_cur.iter().any(&covered) || self.surfaces_prev.iter().any(&covered)
    }

    /// Whether keyboard/nav input reaches widget `id` at `rect`: blocked
    /// while a higher-layer interactive surface owns the frame (a modal
    /// card or an open popup above this widget's layer), when the widget's
    /// rect is clipped away entirely, or while the host's confined input
    /// rect excludes it. Widgets drawn inside the surface's own layer
    /// scope are not covered by it.
    pub(crate) fn input_allowed(&self, id: u64, rect: Rect) -> bool {
        if self
            .draw
            .clip_rect()
            .is_some_and(|clip| clip.intersect(rect).is_none())
        {
            return false;
        }
        self.input_allowed_at(id, rect, self.draw.layer())
    }

    /// `input_allowed` against an explicit layer — focus traversal tests
    /// the layer the widget *registered* at, not the caller's current
    /// draw layer (which is always 0 between frames).
    pub(crate) fn input_allowed_at(&self, id: u64, rect: Rect, layer: i32) -> bool {
        // Confined input: the widget's *effective* rect must reach the
        // card — a widget clipped fully outside it takes no keys.
        if self
            .blocked
            .is_some_and(|card| card.intersect(rect).is_none())
        {
            return false;
        }
        // The surface's own id and its same-layer children are not covered.
        let covered = |surface: &InputSurface| surface.layer > layer && surface.id != id;
        !self.surfaces_cur.iter().any(&covered) && !self.surfaces_prev.iter().any(&covered)
    }

    /// Whether a widget may keep driving its capture this frame: the
    /// capture holder runs, so this is the input_allowed check a drag does
    /// before applying a delta. A blocked holder's capture is cancelled
    /// rather than suspended — when it reappears it can claim the next
    /// press cleanly.
    pub(crate) fn capture_allowed(&mut self, id: u64, rect: Rect) -> bool {
        if self.input_allowed(id, rect) {
            return true;
        }
        self.cancel_capture(id);
        false
    }

    /// Drop `id`'s pointer capture and drag anchor, if it holds them.
    pub(crate) fn cancel_capture(&mut self, id: u64) {
        if self.active_id == Some(id) {
            self.active_id = None;
        }
        self.drags.remove(&id);
    }

    /// Acquire the pointer for `id` on a press. Only the first widget to
    /// claim a press owns it: nothing else may capture for that press —
    /// not even after the claimer released it within the same frame.
    pub(crate) fn try_capture(&mut self, id: u64) -> bool {
        if self.active_id.is_some() || self.press_consumed {
            return false;
        }
        self.press_consumed = true;
        self.active_id = Some(id);
        true
    }

    /// A disabled widget must not hold the focus: clearing it here keeps a
    /// widget disabled *while* focused from swallowing the next key.
    pub(crate) fn blur_if_disabled(&mut self, id: u64, enabled: bool) {
        if !enabled {
            if self.focus_id == Some(id) {
                self.focus_id = None;
                self.focus_rect = None;
            }
            // A disabled widget cannot hold the pointer either: a button
            // captured on an earlier frame lets go the frame it is drawn
            // disabled, not on the next press.
            self.cancel_capture(id);
        }
    }

    /// Lengths of the per-frame registration lists, captured before a
    /// `layout_root` build so the paint-only replay can truncate what the
    /// live pass registered and re-register once at final geometry.
    pub(crate) fn regs_checkpoint(&self) -> (usize, usize, usize) {
        (
            self.focusables_cur.len(),
            self.surfaces_cur.len(),
            self.scroll_cur.len(),
        )
    }

    /// View-layer test hook: count of scroll regions registered this frame.
    #[cfg(test)]
    pub(crate) fn scroll_regions_len(&self) -> usize {
        self.scroll_cur.len()
    }

    /// Roll registration lists back to `cp` (replay rebuild registers
    /// final-geometry entries once).
    pub(crate) fn regs_restore(&mut self, cp: (usize, usize, usize)) {
        self.focusables_cur.truncate(cp.0);
        self.surfaces_cur.truncate(cp.1);
        self.scroll_cur.truncate(cp.2);
    }

    /// Shared press/release capture, including both edges in one frame.
    pub(crate) fn click(&mut self, id: u64, hovered: bool) -> bool {
        if self.view.replay {
            return false;
        }
        if self.input.mouse_pressed && hovered {
            self.try_capture(id);
        }
        let clicked = self.input.mouse_released && hovered && self.active_id == Some(id);
        if self.input.mouse_released && self.active_id == Some(id) {
            self.active_id = None;
        }
        clicked
    }

    /// The focused widget reports itself: keeps `focus_id` alive for the
    /// frame and remembers where it sits so an outside press can blur it.
    /// Register `id` as a live focusable this frame (dead-focus cleanup
    /// and the Tab traversal list both read it).
    /// Full focusable registration: `enabled` is the widget's own
    /// disabled flag; `text_field` marks an IME-eligible editor.
    pub(crate) fn note_focusable(&mut self, id: u64, rect: Rect, enabled: bool, text_field: bool) {
        self.focusables_cur.push(Focusable {
            id,
            rect,
            layer: self.draw.layer(),
            enabled,
            text_field,
        });
        self.note_focus_inner(id, rect);
    }

    fn note_focus_inner(&mut self, id: u64, rect: Rect) {
        if self.focus_id == Some(id) {
            self.focus_seen = true;
            self.focus_rect = Some(rect);
        }
    }

    /// Register an interactive overlay surface for input gating.
    pub(crate) fn register_surface(&mut self, id: u64, rect: Rect, layer: i32, modal: bool) {
        self.surfaces_cur.push(InputSurface {
            id,
            rect,
            layer,
            modal,
        });
    }

    /// Declare an input surface *before* the widgets it covers are drawn.
    ///
    /// Overlays built by [`Ui`] itself ([`Ui::modal`], [`Ui::context_menu`])
    /// register every frame and block the widgets beneath them from their
    /// second frame on — responses already returned on the opening frame
    /// cannot be un-returned. A host that opens its own overlay calls this
    /// first so even the opening frame blocks what sits behind it. `modal`
    /// surfaces block input anywhere; others only over `rect`. The
    /// declaration lasts one frame.
    pub fn declare_input_surface(&mut self, rect: Rect, layer: i32, modal: bool) {
        self.register_surface(0, rect, layer, modal);
    }

    /// Open an overlay draw scope at `layer`: overlays opened from inside
    /// another overlay always rise at least one layer above it, so a
    /// nested modal actually covers (and out-orders) its parent — plain
    /// `max` would draw them at the same layer and z-fight. Subsequent
    /// draws escape every enclosing clip until [`Self::overlay_end`].
    pub(crate) fn overlay_begin(&mut self, layer: i32) -> (i32, Vec<Rect>) {
        let current = self.draw.layer();
        let target = if current >= LAYER_POPUP {
            layer.max(current + 1)
        } else {
            layer.max(current)
        };
        let saved_layer = self.draw.set_layer(target);
        (saved_layer, self.draw.suspend_clip())
    }

    /// Close a scope opened by [`Self::overlay_begin`], restoring the clip
    /// stack and layer it borrowed.
    pub(crate) fn overlay_end(&mut self, save: (i32, Vec<Rect>)) {
        self.draw.restore_clip(save.1);
        self.draw.set_layer(save.0);
    }

    /// Open a plain clip scope (panels, sections, scroll views). The
    /// matching `*_end` calls restore only their own kind of scope, so a
    /// stray `end_clip` cannot pop a parent's clip.
    pub(crate) fn push_clip_scope(&mut self, scope: ClipScope) {
        self.clip_scopes.push(scope);
    }

    /// Close the innermost scope of one of `kinds`; a top of another kind
    /// is left alone rather than popped by mistake.
    /// The innermost open clip scope, if any — a balanced `*_end` call
    /// checks it before acting on what the scope owns.
    pub(crate) fn open_clip_kind(&self) -> Option<ClipKind> {
        self.clip_scopes.last().map(ClipScope::kind)
    }

    pub(crate) fn end_clip_scope(&mut self, kinds: &[ClipKind]) {
        let Some(scope) = self.clip_scopes.last() else {
            // A clip pushed directly onto the draw list escapes the scope
            // tracking: pop it verbatim, the historical behaviour.
            self.draw.pop_clip();
            return;
        };
        if !kinds.contains(&scope.kind()) {
            return;
        }
        match self.clip_scopes.pop().expect("top checked above") {
            ClipScope::Overlay {
                saved_layer,
                saved_clips,
            } => {
                self.draw.restore_clip(saved_clips);
                self.draw.set_layer(saved_layer);
            }
            _ => self.draw.pop_clip(),
        }
    }

    // --- Primitives (delegating to the draw list) ---------------------------

    /// Solid rectangle.
    pub fn rect(&mut self, rect: Rect, color: Color) {
        self.draw.rect(rect, color);
    }

    /// Textured rectangle.
    pub fn image(
        &mut self,
        rect: Rect,
        texture: &std::sync::Arc<crate::texture::GpuTexture>,
        tint: Color,
    ) {
        self.draw.image(rect, texture, tint);
    }

    /// Text extent without drawing — size a widget around its label.
    /// Shares the layout cache with [`Self::label`]: a measured-then-drawn
    /// string costs one layout. `size` is logical; the layout runs at the
    /// physical size and the answer is divided back.
    pub fn measure(&mut self, text: &str, font: FontId, size: f32) -> (f32, f32) {
        let layout = self.cached_layout(text, font, size * self.scale, LayoutOptions::default());
        (layout.width / self.scale, layout.height / self.scale)
    }

    /// The ink bbox of `text` laid out at `size` (logical pixels) — the
    /// union of the glyph bitmap rects, in logical units relative to the
    /// layout origin. Shares the layout cache with [`Self::measure`].
    pub fn ink_bounds(&mut self, text: &str, font: FontId, size: f32) -> Option<Rect> {
        let layout = self.cached_layout(text, font, size * self.scale, LayoutOptions::default());
        layout.ink_bounds().map(|r| {
            Rect::new(
                r.x / self.scale,
                r.y / self.scale,
                r.w / self.scale,
                r.h / self.scale,
            )
        })
    }

    /// Physical layout from the bounded cache, or a fresh `font::layout`
    /// admitted to it. `physical_size` is canonicalized to 1/8 px.
    pub(crate) fn cached_layout(
        &mut self,
        text: &str,
        font: FontId,
        physical_size: f32,
        options: LayoutOptions,
    ) -> std::sync::Arc<super::font::TextLayout> {
        let px_eighth = layout_cache::LayoutCache::canonical_eighth(physical_size);
        let canonical_px = px_eighth as f32 / 8.0;
        let wrap_eighth = options
            .wrap_width
            .map(|w| layout_cache::LayoutCache::canonical_eighth(w).min(u32::MAX - 1))
            .unwrap_or(u32::MAX);
        let mut options = options;
        options.wrap_width = match wrap_eighth {
            u32::MAX => None,
            eighth => Some(eighth as f32 / 8.0),
        };
        let request = layout_cache::LayoutRequest {
            text,
            font,
            px_eighth,
            wrap_eighth,
            align: options.align,
            line_height_bits: options.line_height.to_bits(),
            letter_spacing_bits: options.letter_spacing.to_bits(),
            fonts_rev: self.fonts_rev,
            atlas_epoch: self.cache.epoch(),
        };
        if let Some(layout) = self.layouts.get(&request) {
            return layout;
        }
        let layout = std::sync::Arc::new(super::font::layout(
            &self.fonts,
            &mut self.cache,
            font,
            canonical_px,
            text,
            options,
        ));
        self.layouts
            .insert(&request, std::sync::Arc::clone(&layout));
        layout
    }

    /// Focus control for keyboard navigation between arbitrary widgets.
    pub fn set_focus(&mut self, label: &str) {
        self.focus_id = Some(self.id(label));
    }

    /// Devtools overlay mode (Off by default).
    pub fn set_devtools(&mut self, mode: crate::ui::devtools::DevtoolsMode) {
        self.devtools = mode;
    }

    /// The active devtools mode.
    pub fn devtools(&self) -> crate::ui::devtools::DevtoolsMode {
        self.devtools
    }

    /// Draw the themed focus ring for `id` if rings are enabled, `id` is
    /// focused, and the focus is keyboard-acquired (focus-visible). The
    /// ring rect is `rect` grown by the token offset+width, concentric
    /// with `radius`. Custom widgets call this after painting.
    pub fn focus_ring(&mut self, rect: Rect, radius: f32, id: u64) {
        // Themed focus ring, opt-in and keyboard-focus only: a
        // pointer-focused widget does not get the ring (focus-visible).
        if self.focus_rings && self.focus_id == Some(id) && self.focus_visible {
            let focus = self.theme.tokens.focus;
            let color = self.theme.roles.border.focus;
            let ring = rect.expand(focus.offset + focus.width);
            self.draw.round_ring(
                ring,
                radius + focus.offset + focus.width,
                focus.width,
                color,
            );
        }
    }

    /// Mark the current focus as keyboard-acquired (the ring shows) or
    /// pointer-acquired (hidden). Hosts driving synthetic input use this
    /// to mirror the modality the real input paths maintain.
    pub fn set_focus_visible(&mut self, visible: bool) {
        self.focus_visible = visible;
    }

    pub fn focused(&self, label: &str) -> bool {
        self.focus_id == Some(self.id(label))
    }

    /// The focused widget id, if any (apps use it to keep shortcuts from
    /// stealing keys while a text field is being edited).
    pub fn focus_id(&self) -> Option<u64> {
        self.focus_id
    }

    /// Restore focus as the live pass left it (id may be `None`) so a
    /// replay can't blur or double-move it.
    pub(crate) fn restore_view_focus(&mut self, id: Option<u64>, visible: bool) {
        self.focus_id = id;
        self.focus_visible = visible;
    }

    pub fn clear_focus(&mut self) {
        self.focus_id = None;
    }

    /// Confine widget input to `rect` (an open modal card). Widgets whose
    /// centre is outside the rect stop responding to the pointer.
    pub fn set_input_rect(&mut self, rect: Option<Rect>) {
        self.blocked = rect;
    }

    /// Whether a widget at `rect` currently receives pointer input
    /// (false while a modal confines input elsewhere).
    pub fn interactive(&self, rect: Rect) -> bool {
        match self.blocked {
            Some(card) => card.contains(rect.center()),
            None => true,
        }
    }
}

/// One modal dialog's layout for a frame. Draw content into `card`, then
/// call `ui.end_clip()`.
#[derive(Clone, Copy, Debug)]
pub struct ModalResponse {
    pub card: Rect,
    /// Backdrop clicked or Escape pressed: the caller should close.
    pub dismissed: bool,
    /// Entrance animation alpha (0..1).
    pub alpha: f32,
}

/// A transient notification: slides in bottom-right, holds, slides out.
#[derive(Clone, Debug)]
pub struct Toast {
    /// Monotonic id: animation state keys off this, so a shifting stack does
    /// not hand one toast's motion to another.
    pub id: u64,
    pub text: String,
    /// Seconds this toast has been alive.
    pub age: f32,
    pub ttl: f32,
    pub kind: ToastKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Ok,
    Warn,
    Error,
}

/// Toast queue owned by the app; rendered by [`Ui::toasts`].
#[derive(Default)]
pub struct Toasts {
    items: Vec<Toast>,
    next_id: u64,
}

pub(crate) fn mix_color(a: Color, b: Color, t: f32) -> Color {
    super::tween::LerpValue::lerp_value(a, b, t.clamp(0.0, 1.0))
}

/// FNV-1a offset basis for the id scope seed.
pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hash `a`+`b` as one label (identical to `fnv_push(seed, &a+b)`),
/// without allocating.
pub(crate) fn fnv_push2(seed: u64, a: &str, b: &str) -> u64 {
    let mut hash = seed;
    for byte in (a.len() + b.len()).to_le_bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    for byte in a.bytes().chain(b.bytes()) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Numeric sibling of [`fnv_push`]: mix a u64 under an id scope without
/// formatting a string (view-layer ids use index salts).
pub(crate) fn fnv_push_num(seed: u64, n: u64) -> u64 {
    let mut hash = seed;
    for byte in n.to_le_bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Fold one label into a running scope seed, length-delimited so a scope
/// boundary can never be smuggled through the bytes: `("a","bc")` and
/// `("ab","c")` hash differently.
pub(crate) fn fnv_push(seed: u64, label: &str) -> u64 {
    let mut hash = seed;
    for byte in label.len().to_le_bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    for byte in label.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The visible text of a widget label: everything before the `##` id
/// discriminator (`"OK##a"` displays `"OK"`). Ids still hash the full
/// label, so `OK##a` and `OK##b` are two widgets that read the same. Only
/// widget *labels* go through this — [`Ui::label`] text is drawn verbatim.
pub(crate) fn display_label(label: &str) -> &str {
    label.split_once("##").map_or(label, |(text, _)| text)
}
