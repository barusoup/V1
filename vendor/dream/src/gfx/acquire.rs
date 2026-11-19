//! Swapchain acquire-ahead on a dedicated reusable thread.
//!
//! `vkAcquireNextImageKHR` is the one frame-start call that can block on the
//! presentation engine, and a frame-start block freezes the whole app. So
//! `render` keeps at most one request outstanding: the worker gates it on the
//! frame slot's fence and then acquires, while the app runs its next update.
//! The frame loop picks the token up within its wait budget and skips the
//! frame when it is not there yet.
//!
//! The fence gate is what makes the acquire semaphore reusable: the semaphore
//! is signalled per acquisition and waited by the slot's submission, so it may
//! only be signalled again after that wait completed. Waiting the slot's fence
//! on the worker first proves the slot's previous submission -- and with it
//! the wait -- finished.

use super::SurfaceError;
use super::pipeline::map_surface_error;
use super::vulkan::GpuContext;
use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long each blocking step inside the worker waits before re-checking
/// shutdown, so quiescing the worker stays bounded even mid-stall.
const POLL: Duration = Duration::from_millis(100);

/// How long a client with no frame wait budget blocks at most. `None` must
/// not mean "freeze the app loop behind a dead presentation engine": after
/// this the wait returns [`AcquireOutcome::Pending`] and the request stays
/// in flight for a later frame.
pub(super) const UNBUDGETED_WAIT_CAP: Duration = Duration::from_millis(250);

/// A swapchain image reserved for one frame slot.
#[derive(Clone, Copy, Debug)]
pub(super) struct Token {
    /// Frame slot the request was made for; its resources are free.
    pub slot: usize,
    pub image_index: u32,
    pub suboptimal: bool,
    /// Time the worker spent waiting on the slot's fence.
    pub fence_wait_us: f32,
    /// Time the worker spent inside `vkAcquireNextImageKHR`.
    pub acquire_us: f32,
}

pub(super) enum AcquireOutcome {
    Ready(Token),
    /// Still in flight; the frame should skip.
    Pending,
    /// The acquire reported an error (outdated swapchain, lost surface...).
    Failed(SurfaceError),
    /// The worker exited without answering (shutdown).
    Lost,
}

struct AcquireJob {
    ctx: Arc<GpuContext>,
    swapchain: vk::SwapchainKHR,
    slot: usize,
    fence: vk::Fence,
    semaphore: vk::Semaphore,
}

struct AcquireState {
    job: Option<AcquireJob>,
    reply: Option<Result<Token, SurfaceError>>,
    shutdown: bool,
    exited: bool,
}

struct Shared {
    state: Mutex<AcquireState>,
    cvar: Condvar,
    shutdown_flag: AtomicBool,
}

/// Owns the worker thread and the one outstanding request.
/// The thread starts lazily on the first [`Self::request`] and reuses one
/// `Condvar` slot across frames with zero per-frame channel or closure
/// allocations.
pub(super) struct AcquireWorker {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
    in_flight: bool,
}

impl AcquireWorker {
    pub(super) fn new() -> AcquireWorker {
        AcquireWorker {
            shared: Arc::new(Shared {
                state: Mutex::new(AcquireState {
                    job: None,
                    reply: None,
                    shutdown: false,
                    exited: false,
                }),
                cvar: Condvar::new(),
                shutdown_flag: AtomicBool::new(false),
            }),
            handle: None,
            in_flight: false,
        }
    }

    pub(super) fn has_request(&self) -> bool {
        self.in_flight
    }

    /// Ask for the next swapchain image into `slot`. The worker waits `fence`
    /// before signalling `semaphore`, which is the slot's acquire semaphore;
    /// at most one request is outstanding, so each slot's semaphore is
    /// signalled again only after the previous wait on it completed.
    pub(super) fn request(
        &mut self,
        ctx: Arc<GpuContext>,
        swapchain: vk::SwapchainKHR,
        slot: usize,
        fence: vk::Fence,
        semaphore: vk::Semaphore,
    ) {
        debug_assert!(!self.has_request(), "one acquire request at a time");
        if self.handle.is_none() {
            {
                let mut state = self.shared.state.lock().unwrap();
                state.job = None;
                state.reply = None;
                state.shutdown = false;
                state.exited = false;
            }
            self.shared.shutdown_flag.store(false, Ordering::Release);
            let shared = Arc::clone(&self.shared);
            let handle = std::thread::Builder::new()
                .name("dream-acquire".to_string())
                .spawn(move || Self::worker_loop(shared))
                .expect("spawn dream-acquire thread");
            self.handle = Some(handle);
        }
        {
            let mut state = self.shared.state.lock().unwrap();
            state.reply = None;
            state.job = Some(AcquireJob {
                ctx,
                swapchain,
                slot,
                fence,
                semaphore,
            });
            self.shared.cvar.notify_one();
        }
        self.in_flight = true;
    }

    fn worker_loop(shared: Arc<Shared>) {
        loop {
            let job = {
                let mut state = shared.state.lock().unwrap();
                while state.job.is_none() && !state.shutdown {
                    state = shared.cvar.wait(state).unwrap();
                }
                if state.shutdown {
                    state.exited = true;
                    shared.cvar.notify_all();
                    return;
                }
                state.job.take()
            };
            let Some(job) = job else {
                continue;
            };
            let outcome = Self::acquire(job, &shared.shutdown_flag);
            let mut state = shared.state.lock().unwrap();
            if let Some(res) = outcome {
                state.reply = Some(res);
            }
            if state.shutdown {
                state.exited = true;
                shared.cvar.notify_all();
                return;
            }
            shared.cvar.notify_all();
        }
    }

    fn acquire(job: AcquireJob, shutdown: &AtomicBool) -> Option<Result<Token, SurfaceError>> {
        let AcquireJob {
            ctx,
            swapchain,
            slot,
            fence,
            semaphore,
        } = job;
        let device = ctx.device().clone();
        let fence_start = Instant::now();
        loop {
            if shutdown.load(Ordering::Acquire) {
                // Nothing committed: the semaphore is untouched, and the
                // request simply expires.
                return None;
            }
            match unsafe { device.wait_for_fences(&[fence], true, POLL.as_nanos() as u64) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                Err(_) => return Some(Err(SurfaceError::Other)),
            }
        }
        let fence_wait_us = fence_start.elapsed().as_secs_f32() * 1e6;
        let acquire_start = Instant::now();
        loop {
            if shutdown.load(Ordering::Acquire) {
                return None;
            }
            let got = unsafe {
                ctx.swapchain_loader.acquire_next_image(
                    swapchain,
                    POLL.as_nanos() as u64,
                    semaphore,
                    vk::Fence::null(),
                )
            };
            match got {
                Ok((image_index, suboptimal)) => {
                    return Some(Ok(Token {
                        slot,
                        image_index,
                        suboptimal,
                        fence_wait_us,
                        acquire_us: acquire_start.elapsed().as_secs_f32() * 1e6,
                    }));
                }
                // Timed out without acquiring: the semaphore is not
                // signalled, so trying again is legal.
                Err(vk::Result::NOT_READY | vk::Result::TIMEOUT) => continue,
                Err(err) => return Some(Err(map_surface_error(err))),
            }
        }
    }

    /// Wait up to `budget` for the outstanding request; `None` waits at most
    /// [`UNBUDGETED_WAIT_CAP`], so no client ever blocks its loop behind a
    /// stalled presentation engine. A [`AcquireOutcome::Pending`] leaves the
    /// request in flight; every other outcome consumes it.
    pub(super) fn wait(&mut self, budget: Option<Duration>) -> AcquireOutcome {
        if !self.in_flight {
            return AcquireOutcome::Lost;
        }
        let bound = budget.unwrap_or(UNBUDGETED_WAIT_CAP);
        let deadline = Instant::now() + bound;
        let mut state = self.shared.state.lock().unwrap();
        loop {
            if let Some(res) = state.reply.take() {
                self.in_flight = false;
                return match res {
                    Ok(token) => AcquireOutcome::Ready(token),
                    Err(err) => AcquireOutcome::Failed(err),
                };
            }
            if state.exited {
                self.in_flight = false;
                return AcquireOutcome::Lost;
            }
            let now = Instant::now();
            if now >= deadline {
                return AcquireOutcome::Pending;
            }
            let (guard, wait_res) = self
                .shared
                .cvar
                .wait_timeout(state, deadline - now)
                .unwrap();
            state = guard;
            if wait_res.timed_out() && state.reply.is_none() && !state.exited {
                return AcquireOutcome::Pending;
            }
        }
    }

    /// Join the worker and forget any outstanding request. Returns the reply
    /// that landed before shutdown: `Ok(token)` means the token's slot
    /// semaphore is signalled and must be waited out before that slot can be
    /// acquired again; `None` means the semaphore is untouched.
    pub(super) fn shutdown(&mut self) -> Option<Result<Token, SurfaceError>> {
        self.shared.shutdown_flag.store(true, Ordering::Release);
        {
            let mut state = self.shared.state.lock().unwrap();
            state.shutdown = true;
            self.shared.cvar.notify_all();
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.in_flight = false;
        let mut state = self.shared.state.lock().unwrap();
        state.job = None;
        state.reply.take()
    }

    /// Start a fresh thread after a [`Self::shutdown`] (swapchain rebuild).
    /// The worker spawns lazily on the next [`Self::request`].
    pub(super) fn respawn(&mut self) {}
}

impl Drop for AcquireWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
