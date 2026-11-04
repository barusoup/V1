/// Halton(2,3) jitter over the sequence length FSR expects for the render ratio.
#[derive(Clone, Copy)]
pub struct JitterSequence {
    frame: i32,
    phase_count: i32,
}

impl Default for JitterSequence {
    fn default() -> Self {
        Self {
            frame: 0,
            phase_count: 8,
        }
    }
}

impl JitterSequence {
    /// Jitter in pixels of the render resolution.
    pub fn next(&mut self, render_width: u32, display_width: u32) -> [f32; 2] {
        let ratio = display_width as f32 / render_width.max(1) as f32;
        let target = (8.0 * ratio * ratio) as i32;
        self.phase_count = target.max(1);
        let index = self.frame % self.phase_count;
        self.frame = self.frame.wrapping_add(1);
        [halton(index + 1, 2) - 0.5, halton(index + 1, 3) - 0.5]
    }

    pub fn reset(&mut self) {
        self.frame = 0;
    }
}

pub(crate) fn halton(index: i32, base: i32) -> f32 {
    let mut f = 1.0f32;
    let mut result = 0.0f32;
    let mut i = index;
    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i = (i as f32 / base as f32).floor() as i32;
    }
    result
}
