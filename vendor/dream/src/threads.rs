//! Managed worker threads.
//!
//! A [`Worker`] is one named OS thread with a FIFO job queue: post closures,
//! they run to completion in posting order, and [`Worker::shutdown`] stops
//! acceptance, drains what is queued, and joins the thread. A job that panics
//! is counted in [`Stats::failed`]; the worker keeps taking work (the panic
//! still prints through the standard hook).
//!
//! The engine owns its background work on these -- the renderer's swapchain
//! acquire-ahead runs on one. Games should prefer them over bare
//! `std::thread::spawn` when they want the same lifecycle guarantees: named
//! threads, deterministic shutdown, and counters to look at.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// A queued unit of work. Runs to completion on the worker thread.
pub type Job = Box<dyn FnOnce() + Send + 'static>;

/// Counters a [`Worker`] keeps about its queue and its thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Jobs accepted since the worker started.
    pub posted: u64,
    /// Jobs that ran to completion without panicking.
    pub executed: u64,
    /// Jobs that panicked. The worker keeps taking work afterwards.
    pub failed: u64,
    /// Largest queue length ever observed.
    pub peak_queued: usize,
}

/// Cheap handle for code that loops inside a job and must notice a pending
/// [`Worker::shutdown`] between iterations (a blocking driver call, a poll
/// loop). A shutdown cancels the job; whatever the loop had not committed
/// yet is its own responsibility to leave untouched.
#[derive(Clone)]
pub struct ShutdownToken(Arc<AtomicBool>);

impl ShutdownToken {
    /// True once [`Worker::shutdown`] has started; a looping job should
    /// return promptly.
    pub fn is_shutting_down(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Default)]
struct State {
    queue: VecDeque<Job>,
    closed: bool,
    stats: Stats,
}

struct Shared {
    state: Mutex<State>,
    ready: Condvar,
    shutdown: Arc<AtomicBool>,
}

/// One managed worker thread. Dropping a worker shuts it down (drains and
/// joins), so background work cannot outlive the objects it touches.
pub struct Worker {
    shared: Arc<Shared>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl Worker {
    /// Start a named worker. The name shows up in debuggers and profilers.
    pub fn new(name: &str) -> Worker {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            ready: Condvar::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
        });
        let thread_shared = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || Self::run(thread_shared))
            .expect("spawn worker thread");
        Worker {
            shared,
            handle: Mutex::new(Some(handle)),
        }
    }

    fn run(shared: Arc<Shared>) {
        loop {
            let job = {
                let mut state = shared.state.lock().unwrap();
                loop {
                    if let Some(job) = state.queue.pop_front() {
                        break Some(job);
                    }
                    // Closed and drained: exit. `shutdown` sets `closed`
                    // before waking, so nothing queued is lost.
                    if state.closed {
                        break None;
                    }
                    state = shared.ready.wait(state).unwrap();
                }
            };
            let Some(job) = job else { return };
            let panicked = catch_unwind(AssertUnwindSafe(job)).is_err();
            let mut state = shared.state.lock().unwrap();
            if panicked {
                state.stats.failed += 1;
            } else {
                state.stats.executed += 1;
            }
        }
    }

    /// Handle for jobs that need to observe a pending [`Self::shutdown`].
    pub fn shutdown_token(&self) -> ShutdownToken {
        ShutdownToken(Arc::clone(&self.shared.shutdown))
    }

    /// Queue a job. An [`Err`] hands the job back when the worker is
    /// shutting down, so the caller can still run or drop it deliberately.
    pub fn post(&self, job: impl FnOnce() + Send + 'static) -> Result<(), Job> {
        let mut state = self.shared.state.lock().unwrap();
        if state.closed {
            return Err(Box::new(job));
        }
        state.queue.push_back(Box::new(job));
        state.stats.posted += 1;
        state.stats.peak_queued = state.stats.peak_queued.max(state.queue.len());
        drop(state);
        self.shared.ready.notify_one();
        Ok(())
    }

    /// Jobs queued but not started yet.
    pub fn queued(&self) -> usize {
        self.shared.state.lock().unwrap().queue.len()
    }

    /// Counters snapshot.
    pub fn stats(&self) -> Stats {
        self.shared.state.lock().unwrap().stats
    }

    /// Stop accepting jobs, run what is queued, and join the thread.
    /// Idempotent; returns the final [`Stats`]. A job calling this on its
    /// own worker skips the join -- joining the worker from itself would
    /// deadlock -- and the thread exits once the queue drains.
    pub fn shutdown(&self) -> Stats {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.state.lock().unwrap().closed = true;
        self.shared.ready.notify_all();
        let handle = self.handle.lock().unwrap().take();
        if let Some(handle) = handle
            && handle.thread().id() != std::thread::current().id()
        {
            let _ = handle.join();
        }
        self.stats()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn jobs_run_in_posting_order() {
        let worker = Worker::new("test-order");
        let seen = Arc::new(Mutex::new(Vec::new()));
        for i in 0..64 {
            let seen = Arc::clone(&seen);
            worker
                .post(move || seen.lock().unwrap().push(i))
                .ok()
                .expect("post before shutdown");
        }
        let stats = worker.shutdown();
        assert_eq!(*seen.lock().unwrap(), (0..64).collect::<Vec<_>>());
        assert_eq!(stats.posted, 64);
        assert_eq!(stats.executed, 64);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn shutdown_drains_queued_jobs() {
        let worker = Worker::new("test-drain");
        let done = Arc::new(AtomicUsize::new(0));
        for _ in 0..32 {
            let done = Arc::clone(&done);
            worker
                .post(move || {
                    done.fetch_add(1, Ordering::Relaxed);
                })
                .ok()
                .expect("post before shutdown");
        }
        let stats = worker.shutdown();
        assert_eq!(done.load(Ordering::Relaxed), 32, "queued jobs must run");
        assert_eq!(stats.executed, 32);
    }

    #[test]
    fn post_after_shutdown_returns_the_job() {
        let worker = Worker::new("test-reject");
        worker.shutdown();
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        let rejected = worker
            .post(move || flag.store(true, Ordering::Relaxed))
            .expect_err("closed worker must reject");
        rejected();
        assert!(ran.load(Ordering::Relaxed), "caller gets the job back");
    }

    #[test]
    fn panicking_job_does_not_kill_the_worker() {
        let worker = Worker::new("test-panic");
        assert!(worker.post(|| panic!("job panic is contained")).is_ok());
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        worker
            .post(move || flag.store(true, Ordering::Relaxed))
            .ok()
            .expect("post");
        worker.shutdown();
        assert!(
            ran.load(Ordering::Relaxed),
            "worker survives a panicking job"
        );
        let stats = worker.stats();
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.executed, 1);
        assert_eq!(stats.posted, 2);
    }

    #[test]
    fn drop_joins_the_thread() {
        let worker = Worker::new("test-drop");
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        worker
            .post(move || flag.store(true, Ordering::Relaxed))
            .ok()
            .expect("post");
        drop(worker);
        assert!(done.load(Ordering::Relaxed), "drop must drain and join");
    }

    #[test]
    fn shutdown_token_flips_and_stats_track_the_queue() {
        let worker = Worker::new("test-token");
        let token = worker.shutdown_token();
        assert!(!token.is_shutting_down());
        let gate = Arc::new(AtomicBool::new(false));
        let started = Arc::clone(&gate);
        worker
            .post(move || {
                started.store(true, Ordering::Relaxed);
                while !token.is_shutting_down() {
                    std::thread::yield_now();
                }
            })
            .ok()
            .expect("post");
        while !gate.load(Ordering::Relaxed) {
            std::thread::yield_now();
        }
        let stats = worker.shutdown();
        assert!(worker.shutdown_token().is_shutting_down());
        assert_eq!(stats.posted, 1);
        assert_eq!(stats.executed, 1, "the looping job exits on shutdown");
        assert_eq!(stats.peak_queued, 1);
    }
}
