// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The lookup against a stub server, so every provider answer can be scripted: a 401 for Boidu's
//! uncached form, a 429 with `Retry-After`, an LRCLIB 500. Each test checks both what came back and
//! what was asked for and remembered.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::json;
use yt_lyrics::{Body, Cache, Endpoints, Lookup, Query, Service, Source};

struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: String,
}

impl Reply {
    fn json(status: u16, body: serde_json::Value) -> Reply {
        Reply {
            status,
            headers: vec![("Content-Type", "application/json".into())],
            body: body.to_string(),
        }
    }

    fn with_header(mut self, name: &'static str, value: &str) -> Reply {
        self.headers.push((name, value.to_owned()));
        self
    }
}

struct Stub {
    base: String,
    /// Every request path and query, in order.
    seen: Arc<Mutex<Vec<String>>>,
}

impl Stub {
    fn count(&self, prefix: &str) -> usize {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|p| p.starts_with(prefix))
            .count()
    }
}

/// Serve `route` on a loopback port until the test process exits.
fn serve(route: impl Fn(&str) -> Reply + Send + 'static) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_thread = seen.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                continue;
            }
            // Drain the headers.
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
            }
            let target = request_line
                .split_whitespace()
                .nth(1)
                .unwrap_or("/")
                .to_owned();
            seen_for_thread.lock().unwrap().push(target.clone());
            let reply = route(&target);
            let mut head = format!(
                "HTTP/1.1 {} Stub\r\nContent-Length: {}\r\nConnection: close\r\n",
                reply.status,
                reply.body.len()
            );
            for (name, value) in &reply.headers {
                head.push_str(&format!("{name}: {value}\r\n"));
            }
            head.push_str("\r\n");
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(reply.body.as_bytes());
        }
    });
    Stub { base, seen }
}

fn cache_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn scratch_cache(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("yt-lyrics-svc-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn service(boidu: &Stub, lrclib: &Stub, cache: Option<PathBuf>) -> Service {
    Service::new(reqwest::Client::new(), cache).with_endpoints(Endpoints {
        boidu: format!("{}/getLyrics", boidu.base),
        lrclib: format!("{}/api", lrclib.base),
    })
}

fn query(video_id: &str, duration: Option<f64>) -> Query {
    Query {
        video_id: video_id.into(),
        title: "Song".into(),
        artist: "Artist".into(),
        album: Some("Album".into()),
        duration_secs: duration,
    }
}

const TTML: &str = r#"<tt xmlns="http://www.w3.org/ns/ttml"><body><div><p begin="1.0">first</p><p begin="4.5">second</p></div></body></tt>"#;
const LRC: &str = "[00:01.00]lrc first\n[00:04.00]lrc second";

fn no_lrclib() -> Stub {
    serve(|_| Reply::json(404, json!({ "error": "not found" })))
}

#[tokio::test]
async fn boidu_is_asked_with_album_then_without_after_a_401() {
    let boidu = serve(|target| {
        if target.contains("al=") {
            Reply::json(401, json!({ "error": "API key required" }))
        } else {
            Reply::json(200, json!({ "ttml": TTML }))
        }
    });
    let lrclib = no_lrclib();
    let lookup = service(&boidu, &lrclib, None)
        .lookup(&query("vid_one-AB", Some(200.4)))
        .await;

    let Lookup::Found(lyrics) = lookup else {
        panic!("expected Boidu lyrics, got {lookup:?}")
    };
    assert_eq!(lyrics.source, Source::Boidu);
    assert!(lyrics.is_synced());
    assert_eq!(boidu.count("/getLyrics"), 2, "album form, then bare form");
    assert!(
        boidu.seen.lock().unwrap()[0].contains("d=200"),
        "duration is sent in whole seconds"
    );
}

#[tokio::test]
async fn a_rate_limit_pauses_boidu_and_lrclib_answers() {
    let boidu = serve(|_| {
        Reply::json(429, json!({ "error": "slow down" })).with_header("Retry-After", "120")
    });
    let lrclib = serve(|target| {
        if target.starts_with("/api/get") {
            Reply::json(200, json!({ "syncedLyrics": LRC, "duration": 200.0 }))
        } else {
            Reply::json(404, json!({}))
        }
    });
    let service = service(&boidu, &lrclib, None);

    let first = service.lookup(&query("vid_aaaaaaa", Some(200.0))).await;
    let Lookup::Found(lyrics) = first else {
        panic!("expected LRCLIB, got {first:?}")
    };
    assert_eq!(lyrics.source, Source::Lrclib);
    assert_eq!(boidu.count("/getLyrics"), 1);

    // Still inside the 120 s pause: Boidu is not asked again.
    let second = service.lookup(&query("vid_bbbbbbb", Some(200.0))).await;
    assert!(matches!(second, Lookup::Found(_)), "{second:?}");
    assert_eq!(boidu.count("/getLyrics"), 1, "paused, so no second request");
}

#[tokio::test]
async fn a_definitive_miss_is_remembered_and_not_asked_again() {
    let boidu = serve(|_| Reply::json(401, json!({ "error": "API key required" })));
    let lrclib = no_lrclib();
    let dir = scratch_cache("miss");
    let service = service(&boidu, &lrclib, Some(dir.clone()));

    assert_eq!(
        service.lookup(&query("vid_miss0001", Some(90.0))).await,
        Lookup::NotFound
    );
    let asked = (boidu.count("/"), lrclib.count("/"));

    assert_eq!(
        service.lookup(&query("vid_miss0001", Some(90.0))).await,
        Lookup::NotFound
    );
    assert_eq!(
        (boidu.count("/"), lrclib.count("/")),
        asked,
        "served from the cache"
    );
    assert_eq!(
        Cache::new(&dir).get("vid_miss0001", cache_now()),
        Some(None)
    );
}

#[tokio::test]
async fn a_failure_is_unavailable_and_never_remembered() {
    let boidu = serve(|_| Reply::json(401, json!({ "error": "API key required" })));
    let lrclib = serve(|_| Reply::json(500, json!({ "error": "boom" })));
    let dir = scratch_cache("failure");
    let service = service(&boidu, &lrclib, Some(dir.clone()));

    let lookup = service.lookup(&query("vid_fail0001", Some(90.0))).await;
    assert!(matches!(lookup, Lookup::Unavailable(_)), "{lookup:?}");
    assert_eq!(Cache::new(&dir).get("vid_fail0001", cache_now()), None);
}

#[tokio::test]
async fn a_miss_while_boidu_was_throttled_is_not_remembered() {
    let boidu = serve(|_| Reply::json(429, json!({})).with_header("Retry-After", "1"));
    let lrclib = no_lrclib();
    let dir = scratch_cache("throttled-miss");
    let service = service(&boidu, &lrclib, Some(dir.clone()));

    assert_eq!(
        service.lookup(&query("vid_thr00001", Some(90.0))).await,
        Lookup::NotFound
    );
    assert_eq!(
        Cache::new(&dir).get("vid_thr00001", cache_now()),
        None,
        "Boidu may have it once it is reachable again"
    );
}

#[tokio::test]
async fn without_a_length_boidu_is_skipped_and_nothing_is_remembered() {
    let boidu = serve(|_| Reply::json(200, json!({ "ttml": TTML })));
    let lrclib = serve(|_| Reply::json(200, json!({ "syncedLyrics": LRC })));
    let dir = scratch_cache("no-length");
    let service = service(&boidu, &lrclib, Some(dir.clone()));

    let lookup = service.lookup(&query("vid_nolen001", None)).await;
    let Lookup::Found(lyrics) = lookup else {
        panic!("{lookup:?}")
    };
    assert_eq!(lyrics.source, Source::Lrclib);
    assert_eq!(boidu.count("/"), 0, "Boidu cannot answer without a length");
    assert_eq!(Cache::new(&dir).get("vid_nolen001", cache_now()), None);
}

#[tokio::test]
async fn lrclib_search_takes_the_cut_nearest_our_length() {
    let boidu = serve(|_| Reply::json(401, json!({})));
    let lrclib = serve(|target| {
        if target.starts_with("/api/search") {
            Reply::json(
                200,
                json!([
                    { "duration": 300.0, "syncedLyrics": "[00:01.00]the wrong cut\n[00:03.00]more" },
                    { "duration": 356.0, "syncedLyrics": "[00:01.00]the right cut\n[00:03.00]more" },
                    { "duration": 230.0, "plainLyrics": "a plain far away" }
                ]),
            )
        } else {
            Reply::json(404, json!({}))
        }
    });
    let lookup = service(&boidu, &lrclib, None)
        .lookup(&query("vid_cut0001", Some(355.0)))
        .await;
    let Lookup::Found(lyrics) = lookup else {
        panic!("{lookup:?}")
    };
    let Body::Synced(lines) = lyrics.body else {
        panic!("expected synced")
    };
    assert_eq!(lines[0].text, "the right cut");
}

#[tokio::test]
async fn a_flat_boidu_timeline_is_shown_as_plain() {
    let flat = r#"<tt><body><div><p begin="0">a</p><p begin="0">b</p></div></body></tt>"#;
    let boidu = serve(move |_| Reply::json(200, json!({ "ttml": flat })));
    let lrclib = no_lrclib();
    let lookup = service(&boidu, &lrclib, None)
        .lookup(&query("vid_flat0001", Some(60.0)))
        .await;
    let Lookup::Found(lyrics) = lookup else {
        panic!("{lookup:?}")
    };
    assert_eq!(lyrics.body, Body::Plain(vec!["a".into(), "b".into()]));
}
