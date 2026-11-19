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

#[cfg(target_os = "linux")]
fn linux_page_size() -> u64 {
    static PAGE_SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        const SC_PAGESIZE: i32 = 30;
        unsafe extern "C" {
            fn sysconf(name: i32) -> isize;
        }
        let size = unsafe { sysconf(SC_PAGESIZE) };
        if size > 0 { size as u64 } else { 4096 }
    })
}

/// Parse a `/proc/self/smaps` VMA header into `Some((start, end))` when it is
/// a strictly read-only private regular file mapping eligible for clean
/// page-cache eviction (`r--p` or `r-xp`, absolute path, excluding `/dev/` and
/// unlinked `(deleted)` mappings).
#[cfg(target_os = "linux")]
fn parse_clean_ro_smaps_header(line: &str) -> Option<(usize, usize)> {
    let mut rest = line;
    let mut next_tok = || {
        rest = rest.trim_start_matches([' ', '\t']);
        if rest.is_empty() {
            return None;
        }
        match rest.find([' ', '\t']) {
            Some(pos) => {
                let tok = &rest[..pos];
                rest = &rest[pos..];
                Some(tok)
            }
            None => {
                let tok = rest;
                rest = "";
                Some(tok)
            }
        }
    };
    let range = next_tok()?;
    let perms = next_tok()?;
    if perms != "r--p" && perms != "r-xp" {
        return None;
    }
    let _offset = next_tok()?;
    let _dev = next_tok()?;
    let _inode = next_tok()?;
    let pathname = rest.trim();
    if !pathname.starts_with('/') || pathname.starts_with("/dev/") || pathname.contains("(deleted)")
    {
        return None;
    }
    let (start_hex, end_hex) = range.split_once('-')?;
    let start = usize::from_str_radix(start_hex, 16).ok()?;
    let end = usize::from_str_radix(end_hex, 16).ok()?;
    (end > start).then_some((start, end))
}

#[cfg(target_os = "linux")]
fn parse_smaps_kb(value: &str) -> Option<u64> {
    value.split_whitespace().next()?.parse::<u64>().ok()
}

/// Evict resident PTEs from clean read-only file-backed mappings (such as cold
/// driver `.so` code/rodata pages and unused embedded asset pages). Relies
/// strictly on `Private_Dirty == 0 && Shared_Dirty == 0 && Anonymous == 0 &&
/// Locked == 0`, keeping RELRO/GOT/anonymous data untouched and incurring zero
/// swap writes — pages reload on demand via soft faults from kernel page cache.
#[cfg(target_os = "linux")]
fn evict_clean_ro_file_mappings() {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return;
    };
    let page_size = linux_page_size() as usize;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    let mut clean = false;
    let mut saw_private_dirty = false;
    let mut saw_shared_dirty = false;
    let mut saw_anonymous = false;

    let flush = |ranges: &mut Vec<(usize, usize)>,
                 cur: Option<(usize, usize)>,
                 is_clean: bool,
                 p_dirty: bool,
                 s_dirty: bool,
                 anon: bool| {
        if let Some((start, end)) = cur
            && is_clean
            && p_dirty
            && s_dirty
            && anon
            && end > start
            && start.is_multiple_of(page_size)
            && end.is_multiple_of(page_size)
        {
            ranges.push((start, end));
        }
    };

    for line in smaps.lines() {
        let first_word = line.split_whitespace().next().unwrap_or("");
        if let Some(key) = first_word.strip_suffix(':') {
            if current.is_some() && clean {
                let val_str = line.split_once(':').map_or("", |(_, v)| v);
                match key {
                    "Private_Dirty" => {
                        saw_private_dirty = true;
                        if parse_smaps_kb(val_str).unwrap_or(1) != 0 {
                            clean = false;
                        }
                    }
                    "Shared_Dirty" => {
                        saw_shared_dirty = true;
                        if parse_smaps_kb(val_str).unwrap_or(1) != 0 {
                            clean = false;
                        }
                    }
                    "Anonymous" => {
                        saw_anonymous = true;
                        if parse_smaps_kb(val_str).unwrap_or(1) != 0 {
                            clean = false;
                        }
                    }
                    "Locked" if parse_smaps_kb(val_str).unwrap_or(1) != 0 => {
                        clean = false;
                    }
                    _ => {}
                }
            }
            continue;
        }
        flush(
            &mut ranges,
            current,
            clean,
            saw_private_dirty,
            saw_shared_dirty,
            saw_anonymous,
        );
        current = parse_clean_ro_smaps_header(line);
        clean = current.is_some();
        saw_private_dirty = false;
        saw_shared_dirty = false;
        saw_anonymous = false;
    }
    flush(
        &mut ranges,
        current,
        clean,
        saw_private_dirty,
        saw_shared_dirty,
        saw_anonymous,
    );

    const MADV_DONTNEED: i32 = 4;
    unsafe extern "C" {
        fn madvise(addr: *mut core::ffi::c_void, length: usize, advice: i32) -> i32;
    }
    for (start, end) in ranges {
        unsafe {
            madvise(start as *mut core::ffi::c_void, end - start, MADV_DONTNEED);
        }
    }
}

#[cfg(target_os = "linux")]
fn trim_os() -> bool {
    #[cfg(target_env = "gnu")]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe {
            malloc_trim(0);
        }
    }
    evict_clean_ro_file_mappings();
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

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
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
        // `proc_pid_rusage` is one libproc call: no subprocess, no `ps` fork,
        // and it reports the same resident size as `ps`'s RSS column. (Forking
        // `ps` from inside a frame is both slow and unavailable to sandboxed
        // apps, which is why that path could return 0.)
        #[repr(C)]
        struct RusageInfoV0 {
            uuid: [u8; 16],
            user_time: u64,
            system_time: u64,
            pkg_idle_wkups: u64,
            interrupt_wkups: u64,
            pageins: u64,
            wired_size: u64,
            resident_size: u64,
            phys_footprint: u64,
            proc_start_abstime: u64,
            proc_exit_abstime: u64,
        }
        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut core::ffi::c_void) -> i32;
        }
        const RUSAGE_INFO_V0: i32 = 0;
        let mut info = unsafe { core::mem::zeroed::<RusageInfoV0>() };
        let rc = unsafe {
            proc_pid_rusage(
                std::process::id() as i32,
                RUSAGE_INFO_V0,
                &mut info as *mut RusageInfoV0 as *mut core::ffi::c_void,
            )
        };
        if rc == 0 { info.resident_size } else { 0 }
    }
    #[cfg(target_os = "linux")]
    {
        // statm field 2 is RESIDENT pages; field 1 is total virtual size.
        let page_size = linux_page_size();
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|text| {
                text.split_whitespace()
                    .nth(1)
                    .and_then(|pages| pages.parse::<u64>().ok())
            })
            .map(|pages| pages * page_size)
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
