// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The queue and its behaviour. Engine-agnostic: it works with any [`Engine`].
//!
//! Behaviour, and why it is shaped this way:
//!
//! - **Lookahead.** The moment a track starts, the next one is resolved in the background and
//!   handed to the engine as a gapless append. When the current track ends, the next starts with
//!   no round trip. A resolve that is still in flight does not hold up the current track.
//! - **Stale work is dropped.** Every user action that changes what plays bumps a generation.
//!   A resolve that finishes for an older generation is ignored, so a skip cannot be undone by a
//!   lookahead that completed a moment too late.
//! - **Failures recover in order.** A failed lookahead is not an error: when the track ends, the
//!   player resolves the next one explicitly. A track that fails while playing is retried once,
//!   then skipped. A track that cannot be resolved is skipped at load time.
//! - **Position is throttled.** At most one position event every 250 ms during steady playback.
//!   A jump (seek, or a track change) is always sent at once so the UI never lags a seek.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::engine::{Engine, EngineEvent, Source};

/// Turns a video id into something playable. Implemented by the app, over the stream resolver.
pub trait StreamSource: Send + Sync + 'static {
    fn resolve(&self, video_id: &str) -> impl Future<Output = Result<Source, String>> + Send;
}

/// What the player tells the UI.
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerEvent {
    NowPlaying {
        index: usize,
        video_id: String,
    },
    Position(f64),
    Duration(f64),
    Paused(bool),
    /// A track could not be played. The player moves on.
    TrackFailed {
        video_id: String,
        message: String,
    },
    /// Nothing is left to play.
    QueueEnded,
}

/// The state of the gapless lookahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Next {
    /// Nothing is queued after the current track.
    Idle,
    /// A resolve is in flight for `index`.
    Resolving { index: usize, generation: u64 },
    /// `index` is appended to the engine and will start gaplessly.
    Primed { index: usize },
}

struct Resolution {
    generation: u64,
    index: usize,
    result: Result<Source, String>,
}

pub struct Player<E, S> {
    engine: E,
    source: Arc<S>,
    queue: Vec<String>,
    current: Option<usize>,
    next: Next,
    generation: u64,
    /// The index that already had its one retry after a mid-track failure.
    retried: Option<usize>,
    done_tx: mpsc::UnboundedSender<Resolution>,
    done_rx: mpsc::UnboundedReceiver<Resolution>,
    out: mpsc::UnboundedSender<PlayerEvent>,
    throttle: PositionThrottle,
}

impl<E: Engine, S: StreamSource> Player<E, S> {
    pub fn new(engine: E, source: S, out: mpsc::UnboundedSender<PlayerEvent>) -> Self {
        let (done_tx, done_rx) = mpsc::unbounded_channel();
        Player {
            engine,
            source: Arc::new(source),
            queue: Vec::new(),
            current: None,
            next: Next::Idle,
            generation: 0,
            retried: None,
            done_tx,
            done_rx,
            out,
            throttle: PositionThrottle::default(),
        }
    }

    pub fn queue(&self) -> &[String] {
        &self.queue
    }

    pub fn current_index(&self) -> Option<usize> {
        self.current
    }

    /// Replace the queue. Does not interrupt the track that is playing; it only changes what comes
    /// after it, so any lookahead is dropped.
    pub fn set_queue(&mut self, video_ids: Vec<String>) {
        self.queue = video_ids;
        self.generation += 1;
        self.engine.drop_next();
        self.next = Next::Idle;
        self.current = self.current.filter(|&c| c < self.queue.len());
    }

    /// Start the track at `index`. A track that cannot be resolved is reported and skipped.
    pub async fn play(&mut self, index: usize) {
        self.retried = None;
        self.load_from(index).await;
    }

    pub async fn next(&mut self) {
        if let Some(c) = self.current {
            self.play(c + 1).await;
        }
    }

    pub async fn previous(&mut self) {
        if let Some(c) = self.current {
            if c > 0 {
                self.play(c - 1).await;
            }
        }
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.engine.set_paused(paused);
    }

    pub fn seek(&mut self, secs: f64) {
        self.engine.seek(secs);
    }

    /// Read-only access to the engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Mutable access to the engine. For drivers and tests that need to move the clock.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Process what the engine has reported and any lookahead that has finished. Call this on a
    /// regular schedule (every 20 to 50 ms is plenty).
    pub async fn tick(&mut self) {
        for event in self.engine.poll() {
            match event {
                EngineEvent::Position(p) => {
                    if self.throttle.should_emit(p, Instant::now()) {
                        self.emit(PlayerEvent::Position(p));
                    }
                }
                EngineEvent::Duration(d) => self.emit(PlayerEvent::Duration(d)),
                EngineEvent::Paused(p) => self.emit(PlayerEvent::Paused(p)),
                EngineEvent::Advanced => self.on_advanced(),
                EngineEvent::Finished => self.on_finished().await,
                EngineEvent::Failed(message) => self.on_failed(message).await,
            }
        }
        self.drain_resolutions();
    }

    async fn load_from(&mut self, mut index: usize) {
        self.generation += 1;
        self.engine.drop_next();
        self.next = Next::Idle;
        loop {
            let Some(id) = self.queue.get(index).cloned() else {
                self.current = None;
                self.emit(PlayerEvent::QueueEnded);
                return;
            };
            let outcome = match self.source.resolve(&id).await {
                Ok(src) => self.engine.load(src).map_err(|e| e.to_string()),
                Err(message) => Err(message),
            };
            match outcome {
                Ok(()) => {
                    self.current = Some(index);
                    self.throttle = PositionThrottle::default();
                    self.emit(PlayerEvent::NowPlaying {
                        index,
                        video_id: id,
                    });
                    self.prime_next();
                    return;
                }
                Err(message) => {
                    tracing::info!(video_id = %id, %message, "track skipped");
                    self.emit(PlayerEvent::TrackFailed {
                        video_id: id,
                        message,
                    });
                    index += 1;
                }
            }
        }
    }

    /// Resolve the track after the current one in the background.
    fn prime_next(&mut self) {
        let Some(current) = self.current else { return };
        let index = current + 1;
        let Some(id) = self.queue.get(index).cloned() else {
            self.next = Next::Idle;
            return;
        };
        let generation = self.generation;
        self.next = Next::Resolving { index, generation };
        let source = self.source.clone();
        let tx = self.done_tx.clone();
        tokio::spawn(async move {
            let result = source.resolve(&id).await;
            let _ = tx.send(Resolution {
                generation,
                index,
                result,
            });
        });
    }

    fn drain_resolutions(&mut self) {
        while let Ok(res) = self.done_rx.try_recv() {
            let wanted = matches!(
                self.next,
                Next::Resolving { index, generation }
                    if index == res.index && generation == res.generation
            );
            if !wanted || res.generation != self.generation {
                continue; // stale: a user action has overtaken it
            }
            match res.result {
                Ok(src) => match self.engine.append(src) {
                    Ok(()) => self.next = Next::Primed { index: res.index },
                    Err(e) => {
                        tracing::warn!(error = %e, "could not queue the next track gaplessly");
                        self.next = Next::Idle;
                    }
                },
                Err(message) => {
                    // Not reported here. When the current track ends, the player resolves this one
                    // again and reports the failure if it still fails.
                    tracing::debug!(%message, "lookahead failed");
                    self.next = Next::Idle;
                }
            }
        }
    }

    fn on_advanced(&mut self) {
        match self.next {
            Next::Primed { index } => {
                self.current = Some(index);
                self.next = Next::Idle;
                self.retried = None;
                self.throttle = PositionThrottle::default();
                let video_id = self.queue[index].clone();
                self.emit(PlayerEvent::NowPlaying { index, video_id });
                self.prime_next();
            }
            _ => tracing::warn!("engine advanced with no primed track"),
        }
    }

    async fn on_finished(&mut self) {
        match self.current.map(|c| c + 1) {
            Some(index) if index < self.queue.len() => {
                // No gapless next was ready. Load it explicitly: a short wait, not a stall.
                self.load_from(index).await;
            }
            _ => {
                self.current = None;
                self.next = Next::Idle;
                self.emit(PlayerEvent::QueueEnded);
            }
        }
    }

    async fn on_failed(&mut self, message: String) {
        let Some(current) = self.current else { return };
        let video_id = self.queue[current].clone();
        self.emit(PlayerEvent::TrackFailed { video_id, message });
        if self.retried != Some(current) {
            // One more try for the same track: the URL may have been refreshed in the meantime.
            self.retried = Some(current);
            self.load_from(current).await;
        } else {
            self.load_from(current + 1).await;
        }
    }

    fn emit(&self, event: PlayerEvent) {
        let _ = self.out.send(event);
    }
}

/// Limits position updates to about four a second, except for jumps.
#[derive(Debug, Default, Clone, Copy)]
pub struct PositionThrottle {
    last: Option<(Instant, f64)>,
}

/// The longest gap between updates during steady playback.
pub const POSITION_INTERVAL: Duration = Duration::from_millis(250);

impl PositionThrottle {
    /// True when this position should be sent now.
    pub fn should_emit(&mut self, position: f64, now: Instant) -> bool {
        let emit = match self.last {
            None => true,
            Some((at, last_pos)) => {
                let elapsed = now.duration_since(at);
                // Playback can only move about as far as the clock did. Anything more is a seek.
                let jumped = (position - last_pos).abs() > elapsed.as_secs_f64() + 0.75;
                jumped || elapsed >= POSITION_INTERVAL
            }
        };
        if emit {
            self.last = Some((now, position));
        }
        emit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_playback_is_throttled_to_four_a_second() {
        let mut t = PositionThrottle::default();
        let start = Instant::now();
        let emitted = (0..100)
            .filter(|i| {
                let at = start + Duration::from_millis(*i * 10);
                t.should_emit(*i as f64 * 0.01, at)
            })
            .count();
        // 1 second of ticks every 10 ms: the first one, then one per 250 ms.
        assert!((4..=5).contains(&emitted), "emitted {emitted}");
    }

    #[test]
    fn a_seek_is_sent_at_once() {
        let mut t = PositionThrottle::default();
        let start = Instant::now();
        assert!(t.should_emit(10.0, start));
        assert!(!t.should_emit(10.05, start + Duration::from_millis(50)));
        assert!(
            t.should_emit(120.0, start + Duration::from_millis(60)),
            "a jump must not wait"
        );
    }
}
