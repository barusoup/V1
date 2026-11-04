use super::{Response, Ui, mix_color};
use crate::ui::{Color, Rect};

impl Ui {
    // --- Colour pickers (napxl side panel) ---------------------------------

    /// Saturation/value square for `hsv` (h drives the hue). Drag moves
    /// the marker; drawn as a stepped quad grid.
    pub fn sv_field(&mut self, label: &str, rect: Rect, hsv: &mut (f32, f32, f32)) -> Response {
        let id = self.id(label);
        let hovered = self.hit(rect);
        if self.input.mouse_pressed && hovered {
            self.try_capture(id);
        }
        let mut changed = false;
        if self.active_id == Some(id) && self.capture_allowed(id, rect) {
            if self.input.mouse_down {
                let s = ((self.input.mouse.0 - rect.x) / rect.w.max(1.0)).clamp(0.0, 1.0);
                let v = 1.0 - ((self.input.mouse.1 - rect.y) / rect.h.max(1.0)).clamp(0.0, 1.0);
                hsv.1 = s;
                hsv.2 = v;
                changed = true;
            } else {
                self.active_id = None;
            }
        }
        let hover = self.hover_level(id, hovered);
        let step = 3.0;
        let mut y = rect.y;
        while y < rect.max().1 {
            let h = (rect.max().1 - y).min(step);
            let mut x = rect.x;
            while x < rect.max().0 {
                let w = (rect.max().0 - x).min(step);
                let s = (x - rect.x) / rect.w.max(1.0);
                let v = 1.0 - (y - rect.y) / rect.h.max(1.0);
                let (r, g, b) = crate::ui::theme::hsv_to_rgb(hsv.0, s, v);
                self.draw.rect(Rect::new(x, y, w, h), Color::rgb(r, g, b));
                x += w;
            }
            y += h;
        }
        // Marker: 12px, 4px white ring + 4px black outline (HTML `.mk`).
        let mx = rect.x + hsv.1 * rect.w;
        let my = rect.y + (1.0 - hsv.2) * rect.h;
        let marker = Rect::new(mx - 6.0, my - 6.0, 12.0, 12.0);
        self.draw.rect(marker.expand(4.0), Color::BLACK);
        self.draw.rect(marker, Color::WHITE);
        let hover_tint = mix_color(Color::WHITE, Color::rgb(255, 240, 160), hover);
        self.draw.rect(marker.inset(2.0), hover_tint);
        Response::build(rect, self, id, changed, rect.h * 0.5)
    }

    /// Hue strip (0..360) with a 6x20 marker (HTML `#hue`).
    pub fn hue_field(&mut self, label: &str, rect: Rect, hsv: &mut (f32, f32, f32)) -> Response {
        let id = self.id(label);
        let hovered = self.hit(rect);
        if self.input.mouse_pressed && hovered {
            self.try_capture(id);
        }
        let mut changed = false;
        if self.active_id == Some(id) && self.capture_allowed(id, rect) {
            if self.input.mouse_down {
                hsv.0 = (((self.input.mouse.0 - rect.x) / rect.w.max(1.0)).clamp(0.0, 1.0)) * 359.9;
                changed = true;
            } else {
                self.active_id = None;
            }
        }
        let step = 2.0;
        let mut x = rect.x;
        while x < rect.max().0 {
            let w = (rect.max().0 - x).min(step);
            let hue = (x - rect.x) / rect.w.max(1.0) * 360.0;
            let (r, g, b) = crate::ui::theme::hsv_to_rgb(hue, 1.0, 1.0);
            self.draw
                .rect(Rect::new(x, rect.y, w, rect.h), Color::rgb(r, g, b));
            x += w;
        }
        let mx = rect.x + hsv.0 / 360.0 * rect.w;
        let marker = Rect::new(mx - 3.0, rect.y, 6.0, rect.h);
        self.draw.rect(marker, Color::WHITE);
        self.draw
            .rect(Rect::new(marker.x, marker.y, marker.w, 2.0), Color::BLACK);
        self.draw.rect(
            Rect::new(marker.x, marker.max().1 - 2.0, marker.w, 2.0),
            Color::BLACK,
        );
        Response::build(rect, self, id, changed, rect.h * 0.5)
    }

    /// Palette grid (napxl `#pal`): up to 8 columns, square swatches, the
    /// selected one ringed in the foreground colour.
    pub fn palette(
        &mut self,
        label: &str,
        rect: Rect,
        colors: &[Color],
        selected: &mut Option<usize>,
        columns: usize,
    ) -> Response {
        self.push_id(label);
        let columns = columns.max(1);
        let gap = 2.0;
        let cell = ((rect.w - gap * (columns - 1) as f32) / columns as f32)
            .floor()
            .max(4.0);
        let mut response = Response::default();
        for (index, color) in colors.iter().enumerate() {
            let col = index % columns;
            let row = index / columns;
            let cell_rect = Rect::new(
                rect.x + col as f32 * (cell + gap),
                rect.y + row as f32 * (cell + gap),
                cell,
                cell,
            );
            let item_id = self.id(&format!("##sw{index}"));
            let hovered = self.hit(cell_rect);
            if hovered {
                self.hot_id = Some(item_id);
            }
            let clicked = self.click(item_id, hovered);
            if clicked {
                *selected = Some(index);
                response.clicked = true;
            }
            self.draw.rect(cell_rect, *color);
            if *selected == Some(index) {
                // 4px ring just outside the swatch (HTML box-shadow).
                let ring = cell_rect.expand(2.0);
                let border = 2.0;
                self.draw
                    .rect(Rect::new(ring.x, ring.y, ring.w, border), Color::WHITE);
                self.draw.rect(
                    Rect::new(ring.x, ring.max().1 - border, ring.w, border),
                    Color::WHITE,
                );
                self.draw
                    .rect(Rect::new(ring.x, ring.y, border, ring.h), Color::WHITE);
                self.draw.rect(
                    Rect::new(ring.max().0 - border, ring.y, border, ring.h),
                    Color::WHITE,
                );
            }
            response.rect = response.rect.union(cell_rect);
        }
        self.pop_id();
        response
    }
}
