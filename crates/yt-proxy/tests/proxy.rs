// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end: a local upstream that honours ranges, a proxy in front of it, and a client reading
//! through the proxy. Checks the bytes, the chunking the upstream sees, and the headers.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use yt_proxy::{Config, Proxy};

const SIZE: usize = 1024 * 1024 + 777; // not a multiple of the chunk, on purpose

fn pattern() -> Vec<u8> {
    (0..SIZE).map(|i| (i % 251) as u8).collect()
}

#[derive(Default, Clone)]
struct Seen {
    ranges: Arc<Mutex<Vec<(u64, u64)>>>,
    missing_header: Arc<Mutex<u32>>,
}

/// A range-honouring upstream that records each range and checks one required header.
async fn upstream(seen: Seen) -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let data = Arc::new(pattern());
    tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            let data = data.clone();
            let seen = seen.clone();
            tokio::spawn(async move {
                let svc = service_fn(move |req: Request<Incoming>| {
                    let data = data.clone();
                    let seen = seen.clone();
                    async move {
                        if req.headers().get("x-test").map(|v| v.as_bytes()) != Some(b"yes") {
                            *seen.missing_header.lock().unwrap() += 1;
                        }
                        let range = req
                            .headers()
                            .get("range")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.strip_prefix("bytes="))
                            .and_then(|v| v.split_once('-'))
                            .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()));
                        let Some((start, end)) = range else {
                            // A bare GET is recorded as an unbounded range, so the chunk-size
                            // assertion below fails if the proxy ever forwards one.
                            seen.ranges.lock().unwrap().push((0, u64::MAX));
                            // Answer 200 with the whole file, as an ordinary server would.
                            return Ok::<_, Infallible>(Response::new(Full::new(Bytes::from(
                                data.as_ref().clone(),
                            ))));
                        };
                        seen.ranges.lock().unwrap().push((start, end));
                        let end = end.min(data.len() as u64 - 1);
                        let body = data[start as usize..=end as usize].to_vec();
                        let resp = Response::builder()
                            .status(StatusCode::PARTIAL_CONTENT)
                            .header(
                                "content-range",
                                format!("bytes {start}-{end}/{}", data.len()),
                            )
                            .header("content-length", body.len())
                            .header("content-type", "audio/webm")
                            .body(Full::new(Bytes::from(body)))
                            .unwrap();
                        Ok(resp)
                    }
                });
                let _ = http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), svc)
                    .await;
            });
        }
    });
    addr
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test]
async fn players_read_exact_bytes_and_upstream_only_sees_bounded_chunks() {
    let seen = Seen::default();
    let up = upstream(seen.clone()).await;
    let proxy = Proxy::start(
        client(),
        Config {
            chunk: 64 * 1024,
            stall: Duration::from_secs(5),
            max_registrations: 8,
        },
    )
    .await
    .unwrap();

    let local = proxy.register(
        &format!("http://{up}/media"),
        vec![("x-test".to_owned(), "yes".to_owned())],
    );

    // 1. An open-ended read, the shape a streaming player sends. It must come back complete.
    let body = client().get(&local).send().await.unwrap();
    assert_eq!(body.status(), 200);
    assert_eq!(body.headers()["content-length"], SIZE.to_string().as_str());
    assert_eq!(body.headers()["content-type"], "audio/webm");
    assert_eq!(body.bytes().await.unwrap().as_ref(), pattern().as_slice());

    // Every upstream request was bounded, and none was larger than one chunk.
    let ranges = seen.ranges.lock().unwrap().clone();
    assert!(
        ranges.len() >= 2,
        "a 1 MiB file at 64 KiB chunks needs several requests"
    );
    for (start, end) in &ranges {
        // Saturating, because the bare-GET sentinel is (0, u64::MAX).
        assert!(
            end.saturating_sub(*start) < 64 * 1024,
            "chunk too large: {start}-{end}"
        );
    }

    // 2. A seek: a bounded range in the middle. Exact bytes, 206, correct Content-Range.
    let resp = client()
        .get(&local)
        .header("range", "bytes=5000-9999")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 206);
    assert_eq!(
        resp.headers()["content-range"],
        format!("bytes 5000-9999/{SIZE}").as_str()
    );
    assert_eq!(
        resp.bytes().await.unwrap().as_ref(),
        &pattern()[5000..=9999]
    );

    // 3. An open-ended seek to the tail: the last partial chunk must be exact too.
    let resp = client()
        .get(&local)
        .header("range", format!("bytes={}-", SIZE - 300))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 206);
    assert_eq!(
        resp.bytes().await.unwrap().as_ref(),
        &pattern()[SIZE - 300..]
    );

    // Upstream headers were attached on every request, including the metadata probe.
    assert_eq!(*seen.missing_header.lock().unwrap(), 0);
}

#[tokio::test]
async fn head_reports_length_without_a_body_and_bad_paths_are_refused() {
    let up = upstream(Seen::default()).await;
    let proxy = Proxy::start(client(), Config::default()).await.unwrap();
    let local = proxy.register(&format!("http://{up}/media"), vec![]);

    let head = client().head(&local).send().await.unwrap();
    assert_eq!(head.status(), 200);
    assert_eq!(head.headers()["content-length"], SIZE.to_string().as_str());
    assert!(head.bytes().await.unwrap().is_empty());

    // Starting past the end is the one 416 case.
    let past = client()
        .get(&local)
        .header("range", format!("bytes={SIZE}-"))
        .send()
        .await
        .unwrap();
    assert_eq!(past.status(), 416);

    // A path with the wrong token must not reach any registration.
    let (prefix, id) = local.rsplit_once("/a/").unwrap(); // prefix = http://127.0.0.1:<port>/<token>
    let (origin, _token) = prefix.rsplit_once('/').unwrap();
    let forged = format!("{origin}/forged/a/{id}");
    let refused = client().get(&forged).send().await.unwrap();
    assert_eq!(refused.status(), 404);
}

#[tokio::test]
async fn the_registry_forgets_the_oldest_registration_first() {
    let up = upstream(Seen::default()).await;
    let proxy = Proxy::start(
        client(),
        Config {
            chunk: 64 * 1024,
            stall: Duration::from_secs(5),
            max_registrations: 2,
        },
    )
    .await
    .unwrap();
    let first = proxy.register(&format!("http://{up}/media"), vec![]);
    let _second = proxy.register(&format!("http://{up}/media"), vec![]);
    let _third = proxy.register(&format!("http://{up}/media"), vec![]);
    assert_eq!(client().head(&first).send().await.unwrap().status(), 404);
}

#[tokio::test]
async fn the_first_read_of_a_file_needs_no_separate_size_probe() {
    let seen = Seen::default();
    let up = upstream(seen.clone()).await;
    let proxy = Proxy::start(
        client(),
        Config {
            chunk: 64 * 1024,
            stall: Duration::from_secs(5),
            max_registrations: 8,
        },
    )
    .await
    .unwrap();
    let local = proxy.register(&format!("http://{up}/media"), vec![]);

    let body = client().get(&local).send().await.unwrap();
    assert_eq!(body.status(), 200);
    assert_eq!(body.bytes().await.unwrap().as_ref(), pattern().as_slice());

    let ranges = seen.ranges.lock().unwrap().clone();
    assert_eq!(
        ranges[0],
        (0, 64 * 1024 - 1),
        "the first request is the first chunk itself"
    );
    assert!(
        !ranges.contains(&(0, 0)),
        "no one-byte size probe was needed: {ranges:?}"
    );
    // The chunks are consecutive: the opening one was used, not fetched twice.
    for pair in ranges.windows(2) {
        assert_eq!(pair[1].0, pair[0].1 + 1, "{ranges:?}");
    }
}

#[tokio::test]
async fn a_short_bounded_first_read_gets_only_its_own_bytes() {
    let up = upstream(Seen::default()).await;
    let proxy = Proxy::start(
        client(),
        Config {
            chunk: 64 * 1024,
            stall: Duration::from_secs(5),
            max_registrations: 8,
        },
    )
    .await
    .unwrap();
    let local = proxy.register(&format!("http://{up}/media"), vec![]);

    // The first chunk fetched is 64 KiB; the client asked for 10 bytes of it.
    let resp = client()
        .get(&local)
        .header("range", "bytes=100-109")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 206);
    assert_eq!(resp.headers()["content-length"], "10");
    assert_eq!(resp.bytes().await.unwrap().as_ref(), &pattern()[100..=109]);

    // The registration now knows its size, so a later read works the ordinary way.
    let resp = client()
        .get(&local)
        .header("range", "bytes=0-4")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.bytes().await.unwrap().as_ref(), &pattern()[0..=4]);
}
