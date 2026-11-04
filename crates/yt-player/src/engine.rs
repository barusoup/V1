// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The playback engine boundary.
//!
//! Everything the player needs from an audio backend is in [`Engine`]. The backend is swapped by
//! implementing this trait. The rest of the app depends only on the trait.
//!
//! The model is pull-based: the player calls [`Engine::poll`] on its own schedule and gets the
//! events since the last call. No threads are shared, and the backend's timing is never observed
//! from inside the player.

use std::collections::VecDeque;

/// One playable item, as the playback layer needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub video_id: String,
    /// The URL to read audio from.
    pub url: String,
    /// Headers the URL needs.
    pub headers: Vec<(String, String)>,
    /// Length of the track, when known.
    pub duration_secs: Option<f64>,
    /// Loudness adjustment in dB, when known. The backend applies it.
    pub loudness_db: Option<f64>,
}

/// What the backend reports.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// Playback position in seconds.
    Position(f64),
    /// Length of the loaded track in seconds.
    Duration(f64),
    /// Pause state changed.
    Paused(bool),
    /// The track that was appended with [`Engine::append`] has started, gaplessly.
    Advanced,
    /// The current track ended and nothing was appended. The backend is idle.
    Finished,
    /// The current track failed to play.
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    #[error("the audio backend refused the source: {0}")]
    Refused(String),
}

pub trait Engine: Send {
    /// Replace everything and start `src` now.
    fn load(&mut self, src: Source) -> Result<(), EngineError>;

    /// Queue `src` to start the moment the current track ends, with no gap.
    fn append(&mut self, src: Source) -> Result<(), EngineError>;

    /// Forget any queued next track.
    fn drop_next(&mut self);

    fn set_paused(&mut self, paused: bool);

    /// Jump to `secs` in the current track.
    fn seek(&mut self, secs: f64);

    /// Stop and unload.
    fn stop(&mut self);

    /// Events since the last call.
    fn poll(&mut self) -> Vec<EngineEvent>;
}

/// A backend that runs on a clock the test (or a driver) controls.
///
/// It does not decode anything. It is exact about the things the player depends on: when a track
/// ends, when the gapless next track takes over, and what the position is. Used to test the player
/// without audio hardware.
#[derive(Default)]
pub struct SimEngine {
    current: Option<Source>,
    next: Option<Source>,
    position: f64,
    paused: bool,
    events: VecDeque<EngineEvent>,
    /// Every source that was ever loaded or appended, in order. For inspection in tests.
    pub history: Vec<String>,
}

impl SimEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move the clock forward by `dt` seconds. Handles the end of a track, including the gapless
    /// hand-off to the appended next track.
    pub fn advance(&mut self, dt: f64) {
        if self.current.is_none() || self.paused {
            return;
        }
        self.position += dt;
        let duration = self
            .current
            .as_ref()
            .and_then(|s| s.duration_secs)
            .unwrap_or(f64::MAX);
        if self.position >= duration {
            let overshoot = self.position - duration;
            match self.next.take() {
                Some(next) => {
                    self.current = Some(next);
                    self.position = overshoot;
                    self.events.push_back(EngineEvent::Advanced);
                    self.announce_duration();
                }
                None => {
                    self.current = None;
                    self.position = 0.0;
                    self.events.push_back(EngineEvent::Finished);
                    return;
                }
            }
        }
        self.events.push_back(EngineEvent::Position(self.position));
    }

    /// Make the current track fail, as a dead URL would.
    pub fn fail_current(&mut self, message: &str) {
        if self.current.take().is_some() {
            self.events
                .push_back(EngineEvent::Failed(message.to_owned()));
        }
    }

    pub fn current_video(&self) -> Option<&str> {
        self.current.as_ref().map(|s| s.video_id.as_str())
    }

    pub fn has_next(&self) -> bool {
        self.next.is_some()
    }

    fn announce_duration(&mut self) {
        if let Some(d) = self.current.as_ref().and_then(|s| s.duration_secs) {
            self.events.push_back(EngineEvent::Duration(d));
        }
    }
}

impl Engine for SimEngine {
    fn load(&mut self, src: Source) -> Result<(), EngineError> {
        self.history.push(format!("load {}", src.video_id));
        self.position = 0.0;
        self.next = None;
        self.paused = false;
        self.current = Some(src);
        self.announce_duration();
        Ok(())
    }

    fn append(&mut self, src: Source) -> Result<(), EngineError> {
        self.history.push(format!("append {}", src.video_id));
        self.next = Some(src);
        Ok(())
    }

    fn drop_next(&mut self) {
        self.next = None;
    }

    fn set_paused(&mut self, paused: bool) {
        if self.paused != paused {
            self.paused = paused;
            self.events.push_back(EngineEvent::Paused(paused));
        }
    }

    fn seek(&mut self, secs: f64) {
        self.position = secs.max(0.0);
        self.events.push_back(EngineEvent::Position(self.position));
    }

    fn stop(&mut self) {
        self.current = None;
        self.next = None;
        self.position = 0.0;
    }

    fn poll(&mut self) -> Vec<EngineEvent> {
        self.events.drain(..).collect()
    }
}
