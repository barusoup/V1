// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! One running copy of V1 per Windows session.
//!
//! The guard is a named mutex: a second launch exits silently instead of opening another window,
//! and the kernel releases it if the process crashes. The updater needs a second layer for the
//! gap between this process exiting and the installer starting the updated build; that handoff
//! marker lives in [`crate::update`].

/// What a launch found when it asked to be the one running copy.
pub enum Acquire {
    /// This process owns the app; keep the guard until exit.
    First(Guard),
    /// Another copy already has the guard: exit silently.
    AlreadyRunning,
    /// The guard could not be created; the app starts anyway rather than fail shut.
    Unavailable,
}

#[cfg(windows)]
mod imp {
    use super::*;
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    const ERROR_ALREADY_EXISTS: u32 = 183;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateMutexW(attrs: *const c_void, initial_owner: i32, name: *const u16) -> *mut c_void;
        fn GetLastError() -> u32;
        fn SetLastError(code: u32);
        fn CloseHandle(handle: *mut c_void) -> i32;
    }

    /// Keeps the named mutex alive for as long as this process runs.
    pub struct Guard(*mut c_void);

    impl Drop for Guard {
        fn drop(&mut self) {
            // SAFETY: `self.0` is the handle returned by CreateMutexW and is closed once here.
            unsafe { CloseHandle(self.0) };
        }
    }

    pub fn acquire() -> Acquire {
        acquire_named("Local\\V1.App.SingleInstance")
    }

    fn acquire_named(name: &str) -> Acquire {
        let wide: Vec<u16> = std::ffi::OsStr::new(name)
            .encode_wide()
            .chain(Some(0))
            .collect();
        // CreateMutexW reports an existing object through GetLastError, so clear any stale value.
        // SAFETY: SetLastError takes a plain integer and touches no shared state.
        unsafe { SetLastError(0) };
        // SAFETY: `wide` is a NUL-terminated UTF-16 name that lives for the call.
        let handle = unsafe { CreateMutexW(std::ptr::null(), 0, wide.as_ptr()) };
        if handle.is_null() {
            return Acquire::Unavailable;
        }
        // SAFETY: no other thread has changed this thread's last-error since CreateMutexW.
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            // SAFETY: the handle is ours and the existing mutex has other owners.
            unsafe { CloseHandle(handle) };
            Acquire::AlreadyRunning
        } else {
            Acquire::First(Guard(handle))
        }
    }

    #[cfg(test)]
    pub fn test_acquire(name: &str) -> Acquire {
        acquire_named(name)
    }
}

#[cfg(unix)]
mod imp {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::os::fd::AsRawFd;

    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;

    extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }

    /// Keeps the advisory `flock` alive for as long as this process runs.
    pub struct Guard(#[allow(dead_code)] File);

    pub fn acquire() -> Acquire {
        acquire_named("v1-app-single-instance")
    }

    fn acquire_named(name: &str) -> Acquire {
        let path = std::env::temp_dir().join(format!("{name}.lock"));
        let Ok(file) = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
        else {
            return Acquire::Unavailable;
        };
        // SAFETY: `file.as_raw_fd()` is a valid open descriptor owned by `file`.
        let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
        if rc == 0 {
            Acquire::First(Guard(file))
        } else {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                Acquire::AlreadyRunning
            } else {
                Acquire::Unavailable
            }
        }
    }

    #[cfg(test)]
    pub fn test_acquire(name: &str) -> Acquire {
        acquire_named(name)
    }
}

#[cfg(not(any(windows, unix)))]
mod imp {
    use super::*;

    pub struct Guard;

    pub fn acquire() -> Acquire {
        Acquire::First(Guard)
    }
}

pub use imp::Guard;

/// Ask to be the one running copy of V1.
pub fn acquire() -> Acquire {
    imp::acquire()
}

#[cfg(all(test, any(windows, unix)))]
mod tests {
    use super::*;

    #[test]
    fn a_second_copy_is_refused_until_the_first_guard_drops() {
        let name = format!("Local\\V1.Test.{}", std::process::id());
        let first = imp::test_acquire(&name);
        assert!(matches!(first, Acquire::First(_)));
        let second = imp::test_acquire(&name);
        assert!(matches!(second, Acquire::AlreadyRunning));
        drop(first);
        let third = imp::test_acquire(&name);
        assert!(matches!(third, Acquire::First(_)));
    }
}
