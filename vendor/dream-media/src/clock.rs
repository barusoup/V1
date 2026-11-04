//! The playback position as a pure function of time. Every method takes the
//! `Instant` the caller considers "now", so one poll uses one consistent time
//! and tests can drive the clock without sleeping.

use std::time::Instant;

/// A media clock that can run, hold, or jump to a position.
#[derive(Clone, Debug)]
pub struct Clock {
    base_ms: i64,
    started: Option<Instant>,
}

impl Clock {
    /// A stopped clock holding `at_ms`.
    pub fn new(at_ms: i64) -> Self {
        Self {
            base_ms: at_ms,
            started: None,
        }
    }

    /// Whether time is currently advancing the position.
    pub fn running(&self) -> bool {
        self.started.is_some()
    }

    /// Start advancing. Calling this on a running clock does nothing, so a
    /// repeated `play` cannot jump the position forward.
    pub fn start(&mut self, now: Instant) {
        if self.started.is_none() {
            self.started = Some(now);
        }
    }

    /// Freeze at the position `position(now)` reports.
    pub fn stop(&mut self, now: Instant) {
        if self.started.is_some() {
            self.base_ms = self.position(now);
            self.started = None;
        }
    }

    /// The media position in milliseconds at `now`.
    pub fn position(&self, now: Instant) -> i64 {
        match self.started {
            None => self.base_ms,
            Some(started) => {
                let elapsed = now.saturating_duration_since(started);
                self.base_ms + elapsed.as_micros() as i64 / 1000
            }
        }
    }

    /// Re-anchor to `ms`, keeping whether the clock is running.
    pub fn set(&mut self, ms: i64, now: Instant) {
        self.base_ms = ms;
        if self.started.is_some() {
            self.started = Some(now);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn a_new_clock_holds_its_position() {
        let base = Instant::now();
        let clock = Clock::new(1500);
        assert!(!clock.running());
        assert_eq!(clock.position(base), 1500);
        assert_eq!(clock.position(at(base, 9000)), 1500);
    }

    #[test]
    fn a_started_clock_advances_with_time() {
        let base = Instant::now();
        let mut clock = Clock::new(1000);
        clock.start(base);
        assert_eq!(clock.position(at(base, 250)), 1250);
    }

    #[test]
    fn stop_freezes_at_the_current_position() {
        let base = Instant::now();
        let mut clock = Clock::new(0);
        clock.start(base);
        clock.stop(at(base, 400));
        assert_eq!(clock.position(at(base, 5000)), 400);
    }

    #[test]
    fn set_while_running_reanchors_and_keeps_running() {
        let base = Instant::now();
        let mut clock = Clock::new(0);
        clock.start(base);
        clock.set(3000, at(base, 100));
        assert_eq!(clock.position(at(base, 150)), 3050);
    }

    #[test]
    fn starting_twice_does_not_reanchor() {
        let base = Instant::now();
        let mut clock = Clock::new(0);
        clock.start(base);
        clock.start(at(base, 300));
        assert_eq!(clock.position(at(base, 300)), 300);
    }
}
