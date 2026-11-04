//! Swapchain acquire-ahead, run on a managed [`crate::threads::Worker`].
//!
//! `vkAcquireNextImageKHR` is the one frame-start call that can block on the
//! presentation engine, and a frame-start block freezes the whole app. So
//! `render` keeps exactly one request outstanding: the worker gates it on the
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
use crate::threads::{ShutdownToken, Worker};
use ash::vk;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
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

/// Owns the worker thread and the one outstanding request.
pub(super) struct AcquireWorker {
    worker: Worker,
    reply: Option<Receiver<Result<Token, SurfaceError>>>,
}

impl AcquireWorker {
    pub(super) fn new() -> AcquireWorker {
        AcquireWorker {
            worker: Worker::new("dream-acquire"),
            reply: None,
        }
    }

    pub(super) fn has_request(&self) -> bool {
        self.reply.is_some()
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
        let (tx, rx) = mpsc::channel();
        let shutdown = self.worker.shutdown_token();
        let posted = self.worker.post(move || {
            Self::acquire(ctx, swapchain, slot, fence, semaphore, shutdown, tx);
        });
        // A shut-down worker takes no jobs; the caller sees `Lost` and the
        // renderer is about to be quiesced or dropped anyway.
        self.reply = posted.is_ok().then_some(rx);
    }

    fn acquire(
        ctx: Arc<GpuContext>,
        swapchain: vk::SwapchainKHR,
        slot: usize,
        fence: vk::Fence,
        semaphore: vk::Semaphore,
        shutdown: ShutdownToken,
        tx: mpsc::Sender<Result<Token, SurfaceError>>,
    ) {
        let device = ctx.device().clone();
        let fence_start = Instant::now();
        loop {
            if shutdown.is_shutting_down() {
                // Nothing committed: the semaphore is untouched, and the
                // request simply expires.
                return;
            }
            match unsafe { device.wait_for_fences(&[fence], true, POLL.as_nanos() as u64) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                Err(_) => {
                    let _ = tx.send(Err(SurfaceError::Other));
                    return;
                }
            }
        }
        let fence_wait_us = fence_start.elapsed().as_secs_f32() * 1e6;
        let acquire_start = Instant::now();
        loop {
            if shutdown.is_shutting_down() {
                return;
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
                    let _ = tx.send(Ok(Token {
                        slot,
                        image_index,
                        suboptimal,
                        fence_wait_us,
                        acquire_us: acquire_start.elapsed().as_secs_f32() * 1e6,
                    }));
                    return;
                }
                // Timed out without acquiring: the semaphore is not
                // signalled, so trying again is legal.
                Err(vk::Result::NOT_READY | vk::Result::TIMEOUT) => continue,
                Err(err) => {
                    let _ = tx.send(Err(map_surface_error(err)));
                    return;
                }
            }
        }
    }

    /// Wait up to `budget` for the outstanding request; `None` waits at most
    /// [`UNBUDGETED_WAIT_CAP`], so no client ever blocks its loop behind a
    /// stalled presentation engine. A [`AcquireOutcome::Pending`] leaves the
    /// request in flight; every other outcome consumes it.
    pub(super) fn wait(&mut self, budget: Option<Duration>) -> AcquireOutcome {
        let Some(rx) = self.reply.as_ref() else {
            return AcquireOutcome::Lost;
        };
        let bound = budget.unwrap_or(UNBUDGETED_WAIT_CAP);
        let got = rx.recv_timeout(bound).map_err(|err| match err {
            RecvTimeoutError::Timeout => Received::Pending,
            RecvTimeoutError::Disconnected => Received::Lost,
        });
        match got {
            Ok(Ok(token)) => {
                self.reply = None;
                AcquireOutcome::Ready(token)
            }
            Ok(Err(err)) => {
                self.reply = None;
                AcquireOutcome::Failed(err)
            }
            Err(Received::Pending) => AcquireOutcome::Pending,
            Err(Received::Lost) => {
                self.reply = None;
                AcquireOutcome::Lost
            }
        }
    }

    /// Join the worker and forget any outstanding request. Returns the reply
    /// that landed before shutdown: `Ok(token)` means the token's slot
    /// semaphore is signalled and must be waited out before that slot can be
    /// acquired again; `None` means the semaphore is untouched.
    pub(super) fn shutdown(&mut self) -> Option<Result<Token, SurfaceError>> {
        self.worker.shutdown();
        let rx = self.reply.take()?;
        rx.try_recv().ok()
    }

    /// Start a fresh thread after a [`Self::shutdown`] (swapchain rebuild).
    pub(super) fn respawn(&mut self) {
        self.worker = Worker::new("dream-acquire");
    }
}

/// Whether a channel wait gave up or lost the sender.
enum Received {
    Pending,
    Lost,
}
