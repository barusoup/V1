// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! `yt-js`: runs YouTube's own player script to decipher stream URLs. No browser, no WebView.
//!
//! Flow:
//! 1. The homepage names the player script for this build (`/s/player/<hash>/.../base.js`).
//! 2. The script is fetched once per hash and cached on disk. A new build is a new hash.
//! 3. [`discover`] finds this build's signature pipeline and URL class by the shape of its code.
//! 4. The script runs in an embedded QuickJS engine, inside a small set of browser stubs. The
//!    engine's exports call the pipeline, and the player's own URL class applies the `n`
//!    transform.
//!
//! Nothing here is looked up by name, so a rotated build breaks loudly at discovery, not silently
//! at playback.

pub mod discover;
pub mod engine;

use std::path::{Path, PathBuf};

pub use discover::{discover, DiscoverError, Entries};
pub use engine::{Engine, EngineError};

const USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0";
const YOUTUBE: &str = "https://www.youtube.com";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("network: {0}")]
    Http(#[from] reqwest::Error),
    #[error("YouTube answered {0} for the player script")]
    Status(u16),
    #[error("could not read or write the player cache: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Discover(#[from] DiscoverError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error("the homepage names no player script")]
    NoPlayerUrl,
}

/// A loaded player: one build, one engine thread.
#[derive(Clone)]
pub struct Player {
    engine: Engine,
    /// The build's hash, from its path. Changes with every player rotation.
    pub hash: String,
    /// The signature timestamp for `/player` requests, from this build's config.
    pub sts: Option<u32>,
}

impl Player {
    /// Load the player script at `js_path` (as the homepage names it), using `cache_dir`.
    pub async fn load(
        http: &reqwest::Client,
        cache_dir: &Path,
        js_path: &str,
    ) -> Result<Player, Error> {
        let hash = player_hash(js_path).ok_or(Error::NoPlayerUrl)?;
        let cached = cache_file(cache_dir, &hash);
        let src = match tokio::fs::read_to_string(&cached).await {
            Ok(src) => src,
            Err(_) => {
                let src = fetch(http, js_path).await?;
                write_atomic(&cached, &src).await?;
                src
            }
        };
        // Parsing and starting the engine takes a moment, so keep it off the async workers.
        let dir = cache_dir.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let player = Player::from_source(&hash, src)?;
            prune_cache(&dir, &hash);
            Ok(player)
        })
        .await
        .expect("player load task does not panic")
    }

    /// A player from script text already in memory.
    pub fn from_source(hash: &str, src: String) -> Result<Player, Error> {
        let entries = discover(&src)?;
        let sts = entries.sts;
        let engine = Engine::spawn(src, entries)?;
        Ok(Player {
            engine,
            hash: hash.to_owned(),
            sts,
        })
    }

    /// The URL with its signature applied (when `cipher` is `Some((s, sp))`) and its `n` transform
    /// applied. Use this for any URL from a web or TV client.
    pub async fn build_url(
        &self,
        url: &str,
        cipher: Option<(&str, &str)>,
    ) -> Result<String, Error> {
        Ok(self.engine.build(url, cipher).await?)
    }

    /// The signature for a ciphered `s` value.
    pub async fn sig(&self, s: &str) -> Result<String, Error> {
        Ok(self.engine.sig(s).await?)
    }
}

/// The build hash inside a player path: `/s/player/<hash>/...` gives `<hash>`.
pub fn player_hash(js_path: &str) -> Option<String> {
    let rest = js_path.strip_prefix("/s/player/")?;
    let hash = rest.split('/').next()?;
    (!hash.is_empty()).then(|| hash.to_owned())
}

fn cache_file(dir: &Path, hash: &str) -> PathBuf {
    dir.join(format!("{hash}.base.js"))
}

/// Delete the cached scripts of other builds. Every rotation of the player leaves a script of a
/// couple of megabytes behind, and only the current build's is ever read again.
fn prune_cache(dir: &Path, keep_hash: &str) {
    let keep = cache_file(dir, keep_hash);
    let Ok(files) = std::fs::read_dir(dir) else { return };
    for file in files.flatten() {
        let path = file.path();
        let name = file.file_name();
        let name = name.to_string_lossy();
        // Only this module's own files: `<hash>.base.js`, and a half-written `<hash>.base.tmp`.
        let ours = name.ends_with(".base.js") || name.ends_with(".base.tmp");
        if ours && path != keep {
            let _ = std::fs::remove_file(path);
        }
    }
}

async fn fetch(http: &reqwest::Client, js_path: &str) -> Result<String, Error> {
    let resp = http
        .get(format!("{YOUTUBE}{js_path}"))
        .header("user-agent", USER_AGENT)
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(Error::Status(resp.status().as_u16()));
    }
    Ok(resp.text().await?)
}

async fn write_atomic(path: &Path, text: &str) -> Result<(), Error> {
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, text).await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_is_the_path_segment_after_player() {
        assert_eq!(
            player_hash("/s/player/11a45ffb/player_es6.vflset/en_US/base.js").as_deref(),
            Some("11a45ffb")
        );
        assert_eq!(player_hash("/elsewhere/base.js"), None);
    }

    #[test]
    fn old_builds_are_pruned_and_the_current_one_and_strangers_are_kept() {
        let dir = std::env::temp_dir().join(format!("yt-js-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["old1.base.js", "old2.base.js", "half.base.tmp", "now.base.js", "notes.txt"] {
            std::fs::write(dir.join(name), "x").unwrap();
        }
        prune_cache(&dir, "now");
        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|f| f.file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, ["notes.txt", "now.base.js"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
