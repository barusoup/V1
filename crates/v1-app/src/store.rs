// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Small file helpers the collection shares: a background writer, stable names for kept covers, and
//! the clock.
//!
//! Everything the collection puts on disk goes through one writer thread, so the UI thread never
//! waits on the disk, and two writes to the same file can never race. Files are written to a
//! temporary name and renamed into place, so a crash leaves the old file or the new one, never half
//! of either.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

enum Job {
    Put(PathBuf, Vec<u8>),
    Remove(PathBuf),
}

impl Job {
    fn path(&self) -> &Path {
        match self {
            Job::Put(path, _) | Job::Remove(path) => path,
        }
    }

    fn apply(self) {
        match self {
            Job::Put(path, bytes) => {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut temporary = path.clone().into_os_string();
                temporary.push(".tmp");
                let temporary = PathBuf::from(temporary);
                if std::fs::write(&temporary, bytes).is_ok()
                    && std::fs::rename(&temporary, &path).is_err()
                {
                    let _ = std::fs::remove_file(&temporary);
                }
            }
            Job::Remove(path) => {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

/// Writes and removals, done in order on a thread of their own. Dropping it finishes the queue
/// first, so nothing asked for is lost when the app closes.
pub struct Writer {
    tx: Option<Sender<Job>>,
    thread: Option<JoinHandle<()>>,
}

impl Writer {
    pub fn start() -> Writer {
        let (tx, rx) = mpsc::channel::<Job>();
        let thread = std::thread::Builder::new()
            .name("v1-store".to_owned())
            .spawn(move || {
                while let Ok(first) = rx.recv() {
                    // Whatever queued up meanwhile is folded together: only the last word on a
                    // file matters, so a burst of saves writes the index once.
                    let mut batch = vec![first];
                    while let Ok(more) = rx.try_recv() {
                        batch.push(more);
                    }
                    let mut latest: Vec<Job> = Vec::new();
                    for job in batch {
                        match latest.iter_mut().find(|kept| kept.path() == job.path()) {
                            Some(kept) => *kept = job,
                            None => latest.push(job),
                        }
                    }
                    for job in latest {
                        job.apply();
                    }
                }
            })
            .ok();
        Writer {
            tx: Some(tx),
            thread,
        }
    }

    pub fn put(&self, path: PathBuf, bytes: Vec<u8>) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Put(path, bytes));
        }
    }

    pub fn remove(&self, path: PathBuf) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Remove(path));
        }
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Seconds since the Unix epoch.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The file name a kept cover is stored under: a hash of its URL that does not change between runs
/// (FNV-1a, since the standard library's hasher makes no such promise).
pub fn cover_key(url: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Whether `id` is safe to use as a file name: the ids YouTube hands out are letters, digits, `-`
/// and `_`.
pub fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
