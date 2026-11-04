#[cfg(doc)]
use super::Ui;

use super::mix_color;
use crate::ui::Color;
use crate::ui::font::{Align, FontId};
use crate::ui::theme::Theme;

/// How a `vcenter` label positions its text vertically inside `rect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VCenter {
    /// Centre the font's ascent–descent line box — the historical block
    /// centring, which reads slightly low on caps-only text.
    Box,
    /// Centre the optical span: from the first line's cap top down to the
    /// last line's baseline. Capitals and digits sit optically centred.
    Cap,
    /// Same as [`VCenter::Cap`] but with the x-height — for lowercase
    /// text where the x body, not capitals, is the visual weight.
    XHeight,
    /// Put the first baseline exactly `rect.y + y` — explicit control.
    Baseline(f32),
}

/// How an icon fills its rect in [`Ui::icon_fit`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IconFit {
    /// Centre the whole 24x24 raster in the rect — every Lucide icon is
    /// authored on the full grid, so grid centring is exact for them.
    Grid,
    /// Centre the icon's ink bounds and normalise its size: the ink's
    /// major extent becomes `size · (20/24) · optical_k` logical pixels —
    /// Lucide's live area, so typical icons keep roughly today's size
    /// while outliers (wide or tall glyphs) stop reading oversized.
    #[default]
    Ink,
}

/// Text label style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelStyle {
    pub font: FontId,
    pub size: f32,
    pub color: Color,
    pub align: Align,
    pub line_height: f32,
    /// Centre the text vertically inside `rect` instead of starting at the
    /// rect's top edge. Which span is centred is [`Self::vcenter_mode`].
    pub vcenter: bool,
    /// The centring rule for `vcenter` — [`VCenter::Cap`] by default:
    /// labels in controls centre their capitals, not the whole line box.
    pub vcenter_mode: VCenter,
}

impl LabelStyle {
    /// Style for the given font; white, left-aligned, 16 px.
    pub fn new(font: FontId) -> Self {
        Self {
            font,
            size: 16.0,
            color: Color::WHITE,
            align: Align::Left,
            line_height: 1.15,
            vcenter: false,
            vcenter_mode: VCenter::Cap,
        }
    }

    /// Centre the text block vertically in its rect.
    pub fn vcenter(mut self, on: bool) -> Self {
        self.vcenter = on;
        self
    }

    /// The centring rule used while `vcenter` is on.
    pub fn vcenter_mode(mut self, mode: VCenter) -> Self {
        self.vcenter_mode = mode;
        self
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn align(mut self, align: Align) -> Self {
        self.align = align;
        self
    }

    pub fn line_height(mut self, line_height: f32) -> Self {
        self.line_height = line_height;
        self
    }

    pub fn font(mut self, font: FontId) -> Self {
        self.font = font;
        self
    }
}

/// Button appearance. Colours interpolate toward `hover`/`pressed` by the
/// widget's hover animation; `disabled` blocks clicks and swaps the label to
/// `text_disabled` (a solid colour — never opacity). `on_icon`/`accent` carry
/// the latched-icon and switch colours.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButtonStyle {
    pub font: FontId,
    pub size: f32,
    pub text: Color,
    /// Brighter text/icon colour on hover (a colour change, not opacity).
    pub text_hover: Color,
    /// Solid colour for disabled labels (not a faded variant of `text`).
    pub text_disabled: Color,
    pub normal: Color,
    pub hover: Color,
    pub pressed: Color,
    /// Solid accent fill for "on" states (e.g. a switch track).
    pub accent: Color,
    /// Optional explicit border colour (hairline). `None` keeps the
    /// hover-blended border the painter derives on its own.
    pub border: Option<Color>,
    pub corner_radius: f32,
    pub disabled: bool,
    pub on_icon: Color,
}

impl ButtonStyle {
    /// The stock button style: [`Theme::default`]'s themed button.
    pub fn new(font: FontId) -> Self {
        Theme::default().button(font)
    }

    pub fn disabled(mut self) -> Self {
        self.disabled = true;
        self
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;
        self
    }

    pub fn fill(mut self, normal: Color, hover: Color, pressed: Color) -> Self {
        self.normal = normal;
        self.hover = hover;
        self.pressed = pressed;
        self
    }

    pub fn normal(mut self, color: Color) -> Self {
        self.normal = color;
        self
    }

    pub fn hover(mut self, color: Color) -> Self {
        self.hover = color;
        self
    }

    pub fn pressed(mut self, color: Color) -> Self {
        self.pressed = color;
        self
    }

    pub fn text_color(mut self, color: Color) -> Self {
        self.text = color;
        self.text_hover = color;
        self
    }

    pub fn text_hover(mut self, color: Color) -> Self {
        self.text_hover = color;
        self
    }

    pub fn text_disabled(mut self, color: Color) -> Self {
        self.text_disabled = color;
        self
    }

    pub fn accent(mut self, color: Color) -> Self {
        self.accent = color;
        self
    }

    pub fn on_icon(mut self, color: Color) -> Self {
        self.on_icon = color;
        self
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }
}

/// `text_field` appearance. The colour roles mirror [`ButtonStyle`] —
/// `normal`/`hover` fill, `hover` unfocused border, `accent` focused
/// border and selection tint — plus dedicated knobs for everything the
/// widget used to hardcode: border width, caret geometry/colour/blink,
/// selection alpha/inset, text inset and the fade-out on disappearance.
///
/// [`Ui::text_field`] takes `impl Into<TextFieldStyle>`, so a plain
/// `ButtonStyle` still works: [`From`] maps its roles over and fills the
/// new knobs with the built-in defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextFieldStyle {
    pub font: FontId,
    pub size: f32,
    /// Text, caret-ghost base and IME-preedit underline colour.
    pub text: Color,
    /// Solid colour for disabled text (never an opacity fade).
    pub text_disabled: Color,
    pub normal: Color,
    /// Unfocused border; the fill also lifts toward this on hover/focus.
    pub hover: Color,
    /// Focused border and selection-highlight tint.
    pub accent: Color,
    pub corner_radius: f32,
    /// Border thickness in px; `0.0` draws no border.
    pub border_width: f32,
    /// Caret bar: colour, width and top/bottom inset inside the field.
    pub caret: Color,
    pub caret_width: f32,
    pub caret_inset: f32,
    /// Full blink period in seconds (half on, half off); `<= 0.0` keeps
    /// the caret solid.
    pub caret_blink: f32,
    /// Selection-highlight alpha at full opacity, 0..=255 (the RGB is
    /// `accent`); `0` draws no highlight at all.
    pub selection_alpha: u8,
    /// Selection-highlight top/bottom inset inside the field.
    pub selection_inset: f32,
    /// Left text inset in px — where text, caret x=0 and the pointer map.
    pub text_inset: f32,
    /// Seconds the selection highlight and blurred-caret ghost fade out
    /// over on disappearance; `0.0` removes them instantly.
    pub fade_out: f32,
    /// Renders one bullet per character while editing the real string —
    /// the field measures and draws the masked display text, so caret,
    /// selection and pointer geometry all stay consistent. Copy/cut
    /// still emit the real text.
    pub password: bool,
    pub disabled: bool,
}

impl TextFieldStyle {
    /// The stock field style: [`Theme::default`]'s themed text field.
    /// [`From<ButtonStyle>`] is unchanged — it maps a button's roles over
    /// and fills the field-only knobs with the built-in defaults.
    pub fn new(font: FontId) -> Self {
        Theme::default().text_field(font)
    }

    pub fn disabled(mut self) -> Self {
        self.disabled = true;
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;
        self
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn text_color(mut self, color: Color) -> Self {
        self.text = color;
        self
    }

    pub fn text_disabled(mut self, color: Color) -> Self {
        self.text_disabled = color;
        self
    }

    pub fn fill(mut self, normal: Color, hover: Color) -> Self {
        self.normal = normal;
        self.hover = hover;
        self
    }

    pub fn normal(mut self, color: Color) -> Self {
        self.normal = color;
        self
    }

    pub fn hover(mut self, color: Color) -> Self {
        self.hover = color;
        self
    }

    pub fn accent(mut self, color: Color) -> Self {
        self.accent = color;
        self
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }

    pub fn border_width(mut self, width: f32) -> Self {
        self.border_width = width;
        self
    }

    pub fn caret(mut self, color: Color, width: f32, inset: f32) -> Self {
        self.caret = color;
        self.caret_width = width;
        self.caret_inset = inset;
        self
    }

    pub fn caret_blink(mut self, period: f32) -> Self {
        self.caret_blink = period;
        self
    }

    pub fn selection(mut self, alpha: u8, inset: f32) -> Self {
        self.selection_alpha = alpha;
        self.selection_inset = inset;
        self
    }

    pub fn text_inset(mut self, inset: f32) -> Self {
        self.text_inset = inset;
        self
    }

    pub fn fade_out(mut self, seconds: f32) -> Self {
        self.fade_out = seconds;
        self
    }

    /// Mask the drawn text with one bullet per character (password
    /// entry). Editing, selection and the clipboard still see the real
    /// string — only what the field draws and measures is masked.
    pub fn password(mut self, masked: bool) -> Self {
        self.password = masked;
        self
    }
}

impl From<ButtonStyle> for TextFieldStyle {
    fn from(style: ButtonStyle) -> Self {
        Self {
            font: style.font,
            size: style.size,
            text: style.text,
            text_disabled: style.text_disabled,
            normal: style.normal,
            hover: style.hover,
            accent: style.accent,
            corner_radius: style.corner_radius,
            border_width: 1.0,
            caret: mix_color(style.text, style.normal, 0.4),
            caret_width: 1.0,
            caret_inset: 6.0,
            caret_blink: 1.06,
            selection_alpha: 90,
            selection_inset: 4.0,
            text_inset: 8.0,
            fade_out: 0.09,
            password: false,
            disabled: style.disabled,
        }
    }
}

/// Panel background.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelStyle {
    pub background: Color,
    pub border: Option<(f32, Color)>,
    pub corner_radius: f32,
}

impl PanelStyle {
    pub fn new(background: Color) -> Self {
        Self {
            background,
            ..Self::default()
        }
    }

    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    pub fn border(mut self, width: f32, color: Color) -> Self {
        self.border = Some((width, color));
        self
    }

    pub fn no_border(mut self) -> Self {
        self.border = None;
        self
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }
}

impl Default for PanelStyle {
    /// The stock panel: [`Theme::default`]'s themed panel.
    fn default() -> Self {
        Theme::default().panel()
    }
}

/// Slider appearance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliderStyle {
    pub font: FontId,
    pub track: Color,
    pub fill: Color,
    pub knob: Color,
    pub show_value: bool,
}

impl SliderStyle {
    /// The stock slider style: [`Theme::default`]'s themed slider.
    pub fn new(font: FontId) -> Self {
        Theme::default().slider(font)
    }

    pub fn track(mut self, color: Color) -> Self {
        self.track = color;
        self
    }

    pub fn fill(mut self, color: Color) -> Self {
        self.fill = color;
        self
    }

    pub fn knob(mut self, color: Color) -> Self {
        self.knob = color;
        self
    }

    pub fn show_value(mut self, show: bool) -> Self {
        self.show_value = show;
        self
    }
}

/// One menu entry.
#[derive(Clone, Debug)]
pub struct MenuItem {
    pub label: String,
    pub enabled: bool,
}

impl MenuItem {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            enabled: true,
        }
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// Menu appearance for [`Ui::menu`] and [`Ui::context_menu`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MenuStyle {
    pub font: FontId,
    pub size: f32,
    pub text: Color,
    pub item: Color,
    pub item_hover: Color,
    pub item_height: f32,
    pub gap: f32,
    pub padding: f32,
    pub align: Align,
    /// Popup card fill and hairline border (used by [`Ui::context_menu`]).
    pub background: Color,
    pub border: Color,
    pub radius: f32,
}

impl MenuStyle {
    /// The stock menu style: [`Theme::default`]'s themed menu.
    pub fn new(font: FontId) -> Self {
        Theme::default().menu(font)
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn align(mut self, align: Align) -> Self {
        self.align = align;
        self
    }

    pub fn text_color(mut self, color: Color) -> Self {
        self.text = color;
        self
    }

    pub fn item(mut self, color: Color) -> Self {
        self.item = color;
        self
    }

    pub fn item_hover(mut self, color: Color) -> Self {
        self.item_hover = color;
        self
    }

    pub fn item_height(mut self, height: f32) -> Self {
        self.item_height = height;
        self
    }

    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    pub fn border(mut self, color: Color) -> Self {
        self.border = color;
        self
    }

    pub fn radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }
}

/// Full style for [`Ui::segmented_styled`](crate::ui::widgets::Ui::segmented_styled):
/// the item button style plus track, pill and motion overrides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentedStyle {
    /// Base style cloned per segment (label font/size/colours).
    pub button: ButtonStyle,
    /// Track fill colour.
    pub track: Color,
    /// Optional track border `(width, colour)`; `None` draws fill only.
    pub track_border: Option<(f32, Color)>,
    /// The sliding pill's fill.
    pub pill: Color,
    /// Pill/track inset padding.
    pub inset: f32,
    /// Unselected segment text colour.
    pub text: Color,
    /// Selected segment text colour.
    pub text_selected: Color,
    /// Pill glide animation duration.
    pub glide: f32,
    /// Mid-travel pill narrowing factor.
    pub squash: f32,
}

impl SegmentedStyle {
    pub fn button(mut self, s: ButtonStyle) -> Self {
        self.button = s;
        self
    }
    pub fn track(mut self, c: Color) -> Self {
        self.track = c;
        self
    }
    pub fn track_border(mut self, w: f32, c: Color) -> Self {
        self.track_border = Some((w, c));
        self
    }
    pub fn pill(mut self, c: Color) -> Self {
        self.pill = c;
        self
    }
    pub fn inset(mut self, v: f32) -> Self {
        self.inset = v;
        self
    }
    pub fn text(mut self, c: Color) -> Self {
        self.text = c;
        self
    }
    pub fn text_selected(mut self, c: Color) -> Self {
        self.text_selected = c;
        self
    }
    pub fn glide(mut self, v: f32) -> Self {
        self.glide = v;
        self
    }
    pub fn squash(mut self, v: f32) -> Self {
        self.squash = v;
        self
    }
}
