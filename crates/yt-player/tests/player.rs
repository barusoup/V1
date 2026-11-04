// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Behaviour of the queue against a simulated engine and a scripted source.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;
use yt_player::{Player, PlayerEvent, SimEngine, Source, StreamSource};

/// A source whose behaviour the test scripts: per-id delay, and per-id failures to serve first.
#[derive(Clone, Default)]
struct Scripted {
    delay: Arc<Mutex<HashMap<String, Duration>>>,
    failures_left: Arc<Mutex<HashMap<String, u32>>>,
    calls: Arc<Mutex<Vec<String>>>,
}

impl Scripted {
    fn delay(&self, id: &str, d: Duration) {
        self.delay.lock().unwrap().insert(id.to_owned(), d);
    }
    fn fail(&self, id: &str, times: u32) {
        self.failures_left
            .lock()
            .unwrap()
            .insert(id.to_owned(), times);
    }
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl StreamSource for Scripted {
    fn resolve(&self, video_id: &str) -> impl Future<Output = Result<Source, String>> + Send {
        let id = video_id.to_owned();
        let this = self.clone();
        async move {
            this.calls.lock().unwrap().push(id.clone());
            let delay = this.delay.lock().unwrap().get(&id).copied();
            if let Some(d) = delay {
                tokio::time::sleep(d).await;
            }
            let fail = {
                let mut left = this.failures_left.lock().unwrap();
                match left.get_mut(&id) {
                    Some(n) if *n > 0 => {
                        *n -= 1;
                        true
                    }
                    _ => false,
                }
            };
            if fail {
                return Err(format!("{id} refused"));
            }
            Ok(Source {
                video_id: id.clone(),
                url: format!("http://example.test/{id}"),
                headers: vec![],
                duration_secs: Some(10.0),
                loudness_db: None,
            })
        }
    }
}

struct Harness {
    player: Player<SimEngine, Scripted>,
    events: mpsc::UnboundedReceiver<PlayerEvent>,
    source: Scripted,
}

fn harness(ids: &[&str]) -> Harness {
    let source = Scripted::default();
    let (tx, events) = mpsc::unbounded_channel();
    let mut player = Player::new(SimEngine::new(), source.clone(), tx);
    player.set_queue(ids.iter().map(|s| s.to_string()).collect());
    Harness {
        player,
        events,
        source,
    }
}

impl Harness {
    fn drain(&mut self) -> Vec<PlayerEvent> {
        let mut out = Vec::new();
        while let Ok(e) = self.events.try_recv() {
            out.push(e);
        }
        out
    }

    /// Let background resolves finish and the player process them.
    async fn settle(&mut self) {
        for _ in 0..20 {
            self.player.tick().await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        self.player.tick().await;
    }

    fn history(&self) -> Vec<String> {
        self.player.engine().history.clone()
    }
}

#[tokio::test]
async fn the_next_track_is_queued_before_the_current_one_ends() {
    let mut h = harness(&["a", "b", "c"]);
    h.player.play(0).await;
    h.settle().await;

    assert_eq!(h.player.current_index(), Some(0));
    assert!(
        h.history().contains(&"append b".to_owned()),
        "b should be primed: {:?}",
        h.history()
    );

    // The current track ends. b starts gaplessly, with no resolve after the end.
    let before = h.source.calls().len();
    h.player.engine_mut().advance(10.0);
    h.player.tick().await;
    assert_eq!(h.player.current_index(), Some(1));
    assert_eq!(
        h.source.calls().len(),
        before,
        "no network work at the moment of advance"
    );
    h.settle().await;
    assert!(h
        .drain()
        .iter()
        .any(|e| matches!(e, PlayerEvent::NowPlaying { index: 1, .. })));
}

#[tokio::test]
async fn a_lookahead_that_is_still_resolving_does_not_block_the_current_track() {
    let mut h = harness(&["a", "b"]);
    h.source.delay("b", Duration::from_millis(200));
    h.player.play(0).await;
    // The current track is playing even though b is far from ready.
    assert_eq!(h.player.current_index(), Some(0));
    h.player.tick().await;
    assert_eq!(h.player.current_index(), Some(0));
}

#[tokio::test]
async fn a_jump_discards_the_lookahead_that_was_in_flight() {
    let mut h = harness(&["a", "b", "c"]);
    h.source.delay("b", Duration::from_millis(80));
    h.player.play(0).await;
    // Jump before b's slow resolve has landed.
    h.player.play(2).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    h.settle().await;

    let history = h.history();
    let load_c = history
        .iter()
        .position(|s| s == "load c")
        .expect("c loaded");
    assert!(
        !history[load_c..].iter().any(|s| s == "append b"),
        "a stale lookahead for b was queued after the jump: {history:?}"
    );
}

#[tokio::test]
async fn a_track_that_cannot_be_resolved_is_skipped_when_loaded() {
    let mut h = harness(&["a", "b", "c"]);
    h.source.fail("b", 5);
    h.player.play(1).await;
    assert_eq!(h.player.current_index(), Some(2));
    let events = h.drain();
    assert!(events
        .iter()
        .any(|e| matches!(e, PlayerEvent::TrackFailed { video_id, .. } if video_id == "b")));
    assert!(events
        .iter()
        .any(|e| matches!(e, PlayerEvent::NowPlaying { index: 2, .. })));
}

#[tokio::test]
async fn a_failed_lookahead_is_resolved_again_when_the_track_ends() {
    let mut h = harness(&["a", "b"]);
    // Only the first attempt for b fails: the lookahead. The explicit load succeeds.
    h.source.fail("b", 1);
    h.player.play(0).await;
    h.settle().await;
    assert!(!h.history().contains(&"append b".to_owned()));

    h.player.engine_mut().advance(10.0);
    h.settle().await;
    assert_eq!(h.player.current_index(), Some(1), "b must still play");
    assert!(h.history().contains(&"load b".to_owned()));
}

#[tokio::test]
async fn a_mid_track_failure_is_retried_once_then_skipped() {
    let mut h = harness(&["a", "b", "c"]);
    h.player.play(0).await;
    h.settle().await;

    // The URL dies while a plays. The first recovery retries a, and the refetch succeeds.
    h.player.engine_mut().fail_current("403");
    h.settle().await;
    assert_eq!(
        h.player.current_index(),
        Some(0),
        "one retry of the same track"
    );
    assert!(h
        .drain()
        .iter()
        .any(|e| matches!(e, PlayerEvent::TrackFailed { video_id, .. } if video_id == "a")));

    // It dies again, and this time the refetch is refused too: now the player moves on.
    h.source.fail("a", 1);
    h.player.engine_mut().fail_current("403");
    h.settle().await;
    assert_eq!(
        h.player.current_index(),
        Some(1),
        "second failure skips to the next track"
    );
}

#[tokio::test]
async fn the_end_of_the_queue_is_announced() {
    let mut h = harness(&["a"]);
    h.player.play(0).await;
    h.settle().await;
    h.drain();
    h.player.engine_mut().advance(10.0);
    h.settle().await;
    assert!(h
        .drain()
        .iter()
        .any(|e| matches!(e, PlayerEvent::QueueEnded)));
    assert_eq!(h.player.current_index(), None);
}

#[tokio::test]
async fn positions_are_throttled_and_a_seek_is_not() {
    let mut h = harness(&["a"]);
    h.player.play(0).await;
    h.drain();

    // Ten updates in quick succession: only the first gets through.
    for i in 0..10 {
        h.player.engine_mut().advance(0.01 * i as f64 + 0.001);
    }
    h.player.tick().await;
    let steady = h
        .drain()
        .into_iter()
        .filter(|e| matches!(e, PlayerEvent::Position(_)))
        .count();
    assert!(
        steady <= 2,
        "steady position updates should be throttled, got {steady}"
    );

    h.player.seek(5.0);
    h.player.tick().await;
    assert!(h
        .drain()
        .iter()
        .any(|e| matches!(e, PlayerEvent::Position(p) if (*p - 5.0).abs() < 1e-9)));
}
