//! Working-set trimming for GUI apps that sit idle.
//!
//! The resident memory of a GUI app is dominated by cold driver and loader
//! pages that were touched once during start-up and are rarely used again.
//! [`trim_working_set`] asks the OS to return those pages, which lowers the
//! process working set (the figure Task Manager shows). It does not change
//! committed or private bytes. Trimmed pages stay on the OS standby list, so
//! the next frame re-touches them with cheap soft faults rather than disk reads.
//!
//! Hosts should call [`IdleTrim`] from their event loop: call
//! [`IdleTrim::activity`] when a frame is built or input arrives, and
//! [`IdleTrim::trim_if_due`] when the loop is about to park.

use std::time::{Duration, Instant};

/// Ask the OS to take back every resident page the process is not using right now.
///
/// Returns whether the platform call ran. On targets without a supported
/// call (anything other than Windows, glibc Linux and macOS) this returns
/// `false` and does nothing.
pub fn trim_working_set() -> bool {
    trim_os()
}

#[cfg(windows)]
fn trim_os() -> bool {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn K32EmptyWorkingSet(process: *mut core::ffi::c_void) -> i32;
    }
    unsafe { K32EmptyWorkingSet(GetCurrentProcess()) != 0 }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim_os() -> bool {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    unsafe {
        malloc_trim(0);
    }
    true
}

#[cfg(target_os = "macos")]
fn trim_os() -> bool {
    unsafe extern "C" {
        fn malloc_zone_pressure_relief(zone: *mut core::ffi::c_void, goal: usize) -> usize;
    }
    unsafe {
        malloc_zone_pressure_relief(core::ptr::null_mut(), 0);
    }
    true
}

#[cfg(not(any(
    windows,
    all(target_os = "linux", target_env = "gnu"),
    target_os = "macos"
)))]
fn trim_os() -> bool {
    false
}

/// Decides when a host loop should call [`trim_working_set`].
///
/// A trim is due once after the start-up warm-up period, and once after each
/// quiet period of at least `idle_after` following activity. Each trim is
/// reported by [`IdleTrim::due`] exactly once.
#[derive(Clone, Copy, Debug)]
pub struct IdleTrim {
    idle_after: Duration,
    warmup_after: Duration,
    started: Option<Instant>,
    last_activity: Option<Instant>,
    warmup_done: bool,
    trimmed_since_activity: bool,
}

impl IdleTrim {
    /// Default quiet period before an idle trim.
    pub const DEFAULT_IDLE: Duration = Duration::from_millis(1500);
    /// Default delay after the first activity before the warm-up trim.
    pub const DEFAULT_WARMUP: Duration = Duration::from_millis(2000);

    /// Creates a trigger with the default timings.
    pub fn new() -> Self {
        Self::with_timing(Self::DEFAULT_IDLE, Self::DEFAULT_WARMUP)
    }

    /// Creates a trigger with a custom quiet period and warm-up delay.
    pub fn with_timing(idle_after: Duration, warmup_after: Duration) -> Self {
        Self {
            idle_after,
            warmup_after,
            started: None,
            last_activity: None,
            warmup_done: false,
            trimmed_since_activity: false,
        }
    }

    /// Records activity at `now`, such as a built frame or an input event.
    ///
    /// This restarts the quiet period, so the next idle trim is armed again.
    pub fn activity(&mut self, now: Instant) {
        if self.started.is_none() {
            self.started = Some(now);
        }
        self.last_activity = Some(now);
        self.trimmed_since_activity = false;
    }

    /// Returns `true` if a trim is due at `now`, and records that it was reported.
    ///
    /// A quiet period that has elapsed since the last activity takes priority.
    /// Otherwise the warm-up trim fires once, after `warmup_after` from the
    /// first activity.
    pub fn due(&mut self, now: Instant) -> bool {
        if let Some(last) = self.last_activity
            && !self.trimmed_since_activity
            && now >= last + self.idle_after
        {
            self.trimmed_since_activity = true;
            self.warmup_done = true;
            return true;
        }
        if !self.warmup_done
            && let Some(started) = self.started
            && now >= started + self.warmup_after
        {
            self.warmup_done = true;
            return true;
        }
        false
    }

    /// Returns the next instant at which [`IdleTrim::due`] may become true, if any.
    ///
    /// Hosts that park their loop can wait until this deadline. Returns `None`
    /// when no trim is pending.
    pub fn next_deadline(&self) -> Option<Instant> {
        let warmup = self
            .started
            .filter(|_| !self.warmup_done)
            .map(|s| s + self.warmup_after);
        let idle = self
            .last_activity
            .filter(|_| !self.trimmed_since_activity)
            .map(|l| l + self.idle_after);
        match (warmup, idle) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Calls [`trim_working_set`] if a trim is due at `now`.
    ///
    /// Returns `true` only when a trim was due and the platform call ran.
    pub fn trim_if_due(&mut self, now: Instant) -> bool {
        if self.due(now) {
            trim_working_set()
        } else {
            false
        }
    }
}

impl Default for IdleTrim {
    fn default() -> Self {
        Self::new()
    }
}

/// Resident set / working set size of this process in bytes; 0 where the platform offers no cheap query.
#[allow(clippy::needless_return)] // cfg'd tail blocks need explicit returns
pub fn resident_bytes() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                process: *mut core::ffi::c_void,
                counters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        let mut counters = unsafe { core::mem::zeroed::<ProcessMemoryCounters>() };
        counters.cb = core::mem::size_of::<ProcessMemoryCounters>() as u32;
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if ok != 0 {
            return counters.working_set_size as u64;
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        // `ps` reports resident bytes directly (KiB); /proc does not exist
        // on macOS so the Linux path would silently read 0.
        let pid = std::process::id();
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok();
        return output
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|text| text.trim().parse::<u64>().ok())
            .map(|kib| kib * 1024)
            .unwrap_or(0);
    }
    #[cfg(target_os = "linux")]
    {
        // statm field 2 is RESIDENT pages; field 1 is total virtual size.
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|text| {
                text.split_whitespace()
                    .nth(1)
                    .and_then(|pages| pages.parse::<u64>().ok())
            })
            .map(|pages| pages * 4096)
            .unwrap_or(0)
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trigger() -> IdleTrim {
        IdleTrim::with_timing(Duration::from_millis(100), Duration::from_millis(300))
    }

    #[test]
    fn memory_nothing_due_before_any_activity() {
        let t0 = Instant::now();
        let mut trim = trigger();
        assert!(!trim.due(t0 + Duration::from_secs(3600)));
        assert_eq!(trim.next_deadline(), None);
    }

    #[test]
    fn memory_idle_trim_fires_once_per_quiet_period() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut trim = trigger();
        trim.activity(t0);
        assert!(!trim.due(t0 + ms(99)));
        assert!(trim.due(t0 + ms(100)));
        assert!(!trim.due(t0 + ms(500)));
        trim.activity(t0 + ms(600));
        assert!(trim.due(t0 + ms(700)));
    }

    #[test]
    fn memory_warmup_trim_fires_once_under_continuous_activity() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut trim = trigger();
        for step in 0..=5 {
            trim.activity(t0 + ms(step * 50));
        }
        assert!(!trim.due(t0 + ms(250)));
        trim.activity(t0 + ms(300));
        assert!(trim.due(t0 + ms(300)));
        trim.activity(t0 + ms(350));
        assert!(!trim.due(t0 + ms(350)));
    }

    #[test]
    fn memory_idle_trim_also_satisfies_warmup() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut trim = trigger();
        trim.activity(t0);
        assert!(trim.due(t0 + ms(100)));
        trim.activity(t0 + ms(110));
        assert!(trim.due(t0 + ms(210)));
        assert!(!trim.due(t0 + ms(500)));
    }

    #[test]
    fn memory_next_deadline_tracks_pending_work() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut trim = trigger();
        trim.activity(t0);
        assert_eq!(trim.next_deadline(), Some(t0 + ms(100)));
        assert!(trim.due(t0 + ms(100)));
        assert_eq!(trim.next_deadline(), None);
    }

    #[test]
    fn memory_trim_working_set_does_not_panic() {
        let ran = trim_working_set();
        #[cfg(windows)]
        assert!(ran);
        #[cfg(not(windows))]
        let _ = ran;
    }

    #[test]
    fn memory_resident_bytes_reports_something_on_supported_platforms() {
        #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
        assert!(resident_bytes() > 0);
    }
}
