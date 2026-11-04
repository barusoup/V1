use super::FramePacing;
use std::time::{Duration, Instant};

/// A wake this far past the grid is a stall — a window move, a resize, a
/// debugger pause, the machine going to sleep — not timer jitter. Re-anchor
/// instead of firing a burst of catch-up frames.
const PACE_RESYNC_AFTER: f32 = 0.25;

impl FramePacing {
    /// The rate frames are paced to: the display, the cap, or the lower one.
    pub fn target_hz(&self) -> Option<f32> {
        let valid = |hz: Option<f32>, min: f32| hz.filter(|hz| hz.is_finite() && *hz > min);
        match (valid(self.display_hz, 1.0), valid(self.cap_hz, 0.0)) {
            (Some(display), Some(cap)) => Some(display.min(cap)),
            (Some(display), None) => Some(display),
            (None, Some(cap)) => Some(cap),
            (None, None) => None,
        }
    }

    /// Time between paced frames. Clamped to a sane band so a bogus monitor
    /// report can never stall the loop or make it spin.
    pub fn period(&self) -> Option<Duration> {
        self.target_hz()
            .map(|hz| Duration::from_secs_f32(1.0 / hz.clamp(10.0, 1000.0)))
    }

    /// Report the display's refresh rate.
    ///
    /// A change re-anchors the grid, so moving the window to another monitor,
    /// changing that monitor's rate, or plugging one in takes effect on the
    /// next frame instead of after the old grid runs out.
    pub fn set_display_hz(&mut self, hz: Option<f32>) {
        let hz = hz.filter(|hz| hz.is_finite() && *hz > 1.0);
        if self.display_hz != hz {
            self.display_hz = hz;
            self.due = None;
        }
    }

    /// Anchor the grid to `now` — used when a frame answered immediately
    /// (input, a drag), so the schedule restarts from a frame the display
    /// actually received. The synced frame is spent: the next paced slot
    /// is one period out, not a second immediate frame.
    pub fn sync(&mut self, now: Instant) {
        self.due = Some(match self.period() {
            Some(period) => now + period,
            None => now,
        });
    }

    /// Whether a paced frame is due, advancing the grid past any slots
    /// missed since the last wake (each wake still answers at most once).
    ///
    /// The grid is what keeps the rate independent of the wake-up timer. If
    /// each frame instead measured `period` from the previous frame's
    /// *arrival*, every wake's lateness would be charged to that frame, and
    /// the intervals would quantise up to the timer's resolution: on a 360 Hz
    /// display a 1 ms timer turns a 2.78 ms period into 3 ms, a 7% shortfall,
    /// and a coarse 15.6 ms timer (Windows' default, before anything raises
    /// the resolution) turns it into a fraction of the rate.
    pub fn due_now(&mut self, now: Instant) -> bool {
        let Some(period) = self.period() else {
            // No display rate and no cap: answer every wake.
            return true;
        };
        let Some(due) = self.due else {
            self.due = Some(now + period);
            return true;
        };
        if now < due {
            return false;
        }
        if now.duration_since(due).as_secs_f32() > PACE_RESYNC_AFTER {
            self.due = Some(now + period);
            return true;
        }
        let missed = (now.duration_since(due).as_nanos() / period.as_nanos()) as u32 + 1;
        self.due = Some(due + period * missed);
        true
    }

    /// When the next paced frame is due, if there is one waiting.
    pub fn next_due(&self, now: Instant) -> Option<Instant> {
        match self.period() {
            None => Some(now),
            Some(_) => Some(self.due.unwrap_or(now).max(now)),
        }
    }

    /// One line for diagnostics and the selfcheck log.
    pub fn summary(&self) -> String {
        let (display, cap, target) = (self.display_hz, self.cap_hz, self.target_hz());
        format!(
            "display {} cap {} -> paced {} ({:.1} fps)",
            display.map_or("unknown".into(), |hz| format!("{hz:.2} Hz")),
            cap.map_or("none".into(), |hz| format!("{hz:.2} Hz")),
            target.map_or("every wake".into(), |hz| format!("{hz:.2} Hz")),
            target.map_or(0.0, |hz| hz),
        )
    }
}
