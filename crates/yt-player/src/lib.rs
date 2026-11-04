// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! `yt-player`: the queue, lookahead and recovery behaviour, behind a swappable audio engine.
//!
//! The rest of the app depends on [`Player`] and the [`Engine`] trait. The audio backend is
//! replaceable, and [`SimEngine`] lets the whole behaviour be tested without audio hardware.

pub mod engine;
pub mod player;

pub use engine::{Engine, EngineError, EngineEvent, SimEngine, Source};
pub use player::{Player, PlayerEvent, PositionThrottle, StreamSource, POSITION_INTERVAL};
