// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! `yt-proxy`: a loopback range proxy that turns one streamed download into bounded requests.
//!
//! Why: some CDNs throttle an open-ended `Range: bytes=N-` request far below the bitrate, but serve
//! bounded `bytes=N-M` requests at full speed. A player that streams with an open-ended request
//! therefore stalls on large files. This proxy answers the player's open-ended request by
//! fetching bounded chunks upstream and streaming them back as they arrive.
//!
//! Shape:
//! - Listens on `127.0.0.1` only, on an ephemeral port.
//! - Each registered URL is addressed as `/<token>/a/<id>`. The token is random per process, so
//!   no other local program can use the proxy by guessing paths.
//! - Upstream headers (user agent, cookie) are attached here, so the player never sees them.
//! - One chunk at a time is held in memory. A bounded channel applies backpressure, so a paused
//!   player does not make the proxy read ahead without limit.
//! - Every upstream read has a stall timeout. A connection that goes quiet without closing fails
//!   the request rather than hanging the player forever.

pub mod range;

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::io;
use std::net::Ipv4Addr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{combinators::BoxBody, BodyExt, Empty, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{self, HeaderValue};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::range::{
    parse_content_range, parse_range, resolve_window, total_from_content_range, Want,
};

pub type Body = BoxBody<Bytes, io::Error>;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Bytes requested upstream in one bounded range.
    pub chunk: u64,
    /// How long an upstream read may make no progress before the chunk is declared dead.
    pub stall: Duration,
    /// Registrations kept at once. The oldest is dropped first.
    pub max_registrations: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            chunk: 4 * 1024 * 1024,
            stall: Duration::from_secs(20),
            max_registrations: 64,
        }
    }
}

/// What the upstream file is, learned once per registration.
#[derive(Debug, Clone)]
struct Meta {
    total: u64,
    content_type: Option<String>,
}

struct Registration {
    url: String,
    headers: Vec<(String, String)>,
    meta: Mutex<Option<Meta>>,
}

#[derive(Default)]
struct Registry {
    map: HashMap<String, Arc<Registration>>,
    order: VecDeque<String>,
}

struct Shared {
    http: reqwest::Client,
    token: String,
    cfg: Config,
    regs: Mutex<Registry>,
}

/// A running proxy. Clone freely; clones share the server and its registrations.
#[derive(Clone)]
pub struct Proxy {
    shared: Arc<Shared>,
    port: u16,
}

impl Proxy {
    /// Bind the loopback listener and start serving.
    ///
    /// `http` should be a client with no default `Accept-Encoding` and no total timeout. The proxy
    /// sets its own per-read stall timeout, which is what it actually needs.
    pub async fn start(http: reqwest::Client, cfg: Config) -> io::Result<Proxy> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
        let port = listener.local_addr()?.port();
        let token = format!(
            "{:016x}{:016x}",
            rand::random::<u64>(),
            rand::random::<u64>()
        );
        let shared = Arc::new(Shared {
            http,
            token,
            cfg,
            regs: Mutex::new(Registry::default()),
        });

        let accept_shared = shared.clone();
        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(v) => v,
                    Err(e) => {
                        // EMFILE and friends fail instantly, so back off instead of spinning.
                        tracing::warn!(error = %e, "proxy accept failed");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        continue;
                    }
                };
                let shared = accept_shared.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| {
                        let shared = shared.clone();
                        async move { Ok::<_, Infallible>(handle(shared, req).await) }
                    });
                    let _ = http1::Builder::new()
                        .timer(TokioTimer::new())
                        .header_read_timeout(Duration::from_secs(15))
                        .serve_connection(TokioIo::new(stream), svc)
                        .await;
                });
            }
        });

        Ok(Proxy { shared, port })
    }

    /// Register an upstream URL. Returns the loopback URL the player should open.
    pub fn register(&self, url: &str, headers: Vec<(String, String)>) -> String {
        let id = format!("{:016x}", rand::random::<u64>());
        let reg = Arc::new(Registration {
            url: url.to_owned(),
            headers,
            meta: Mutex::new(None),
        });
        {
            let mut regs = self.shared.regs.lock().unwrap();
            while regs.order.len() >= self.shared.cfg.max_registrations.max(1) {
                if let Some(old) = regs.order.pop_front() {
                    regs.map.remove(&old);
                }
            }
            regs.map.insert(id.clone(), reg);
            regs.order.push_back(id.clone());
        }
        format!(
            "http://127.0.0.1:{}/{}/a/{}",
            self.port, self.shared.token, id
        )
    }
}

/// `/<token>/a/<id>` → `<id>`, when the token matches this process.
fn id_from<'a>(path: &'a str, token: &str) -> Option<&'a str> {
    let rest = path.strip_prefix('/')?;
    let (t, rest) = rest.split_once('/')?;
    let (kind, id) = rest.split_once('/')?;
    (t == token && kind == "a" && !id.is_empty()).then_some(id)
}

async fn handle(shared: Arc<Shared>, req: Request<Incoming>) -> Response<Body> {
    if !matches!(*req.method(), Method::GET | Method::HEAD) {
        return status_only(StatusCode::METHOD_NOT_ALLOWED);
    }
    let Some(id) = id_from(req.uri().path(), &shared.token) else {
        return status_only(StatusCode::NOT_FOUND);
    };
    let reg = match shared.regs.lock().unwrap().map.get(id).cloned() {
        Some(r) => r,
        None => return status_only(StatusCode::NOT_FOUND),
    };
    let want = parse_range(
        req.headers()
            .get(header::RANGE)
            .and_then(|v| v.to_str().ok()),
    );
    let is_head = req.method() == Method::HEAD;

    // The first request for a file also has to learn its size. Asking for that on its own costs a
    // whole round trip before any audio moves, so a GET goes straight for the first chunk and reads
    // the size from that answer. A HEAD, or any surprise, falls back to the one-byte probe.
    let cached = reg.meta.lock().unwrap().clone();
    let (meta, first) = match cached {
        Some(meta) => (meta, None),
        None if !is_head => match open_first_chunk(&shared, &reg, want).await {
            Ok((meta, first)) => (meta, Some(first)),
            Err(_) => match meta_for(&shared, &reg).await {
                Ok(meta) => (meta, None),
                Err(_) => return status_only(StatusCode::BAD_GATEWAY),
            },
        },
        None => match meta_for(&shared, &reg).await {
            Ok(meta) => (meta, None),
            Err(_) => return status_only(StatusCode::BAD_GATEWAY),
        },
    };

    let Some((start, end)) = resolve_window(want, meta.total) else {
        let mut r = status_only(StatusCode::RANGE_NOT_SATISFIABLE);
        if let Ok(v) = HeaderValue::from_str(&format!("bytes */{}", meta.total)) {
            r.headers_mut().insert(header::CONTENT_RANGE, v);
        }
        return r;
    };
    let ranged = !matches!(want, Want::Full);

    let body: Body = if is_head {
        Empty::<Bytes>::new().map_err(|e| match e {}).boxed()
    } else {
        let (tx, rx) = mpsc::channel::<io::Result<Bytes>>(16);
        tokio::spawn(pump(shared.clone(), reg.clone(), start, end, first, tx));
        stream_body(rx)
    };

    let mut builder = Response::builder()
        .status(if ranged {
            StatusCode::PARTIAL_CONTENT
        } else {
            StatusCode::OK
        })
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, end - start + 1);
    if let Some(ct) = &meta.content_type {
        builder = builder.header(header::CONTENT_TYPE, ct.as_str());
    }
    if ranged {
        builder = builder.header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{}", meta.total),
        );
    }
    builder
        .body(body)
        .unwrap_or_else(|_| status_only(StatusCode::INTERNAL_SERVER_ERROR))
}

/// The first chunk of a file, already requested, and the byte it starts at.
struct FirstChunk {
    resp: reqwest::Response,
    start: u64,
}

/// Request the chunk a client's first read starts in, and learn the file's size and type from the
/// answer. Anything unexpected is an error, and the caller falls back to [`meta_for`].
async fn open_first_chunk(
    shared: &Shared,
    reg: &Registration,
    want: Want,
) -> io::Result<(Meta, FirstChunk)> {
    let start = match want {
        Want::Full => 0,
        Want::Range { start, .. } => start,
    };
    let last = start.saturating_add(shared.cfg.chunk.max(1) - 1);
    let resp = upstream_range(shared, reg, start, last).await?;
    let (first, _, total) = resp
        .headers()
        .get(header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_content_range)
        .ok_or_else(|| io::Error::other("upstream gave no usable Content-Range"))?;
    if first != start {
        return Err(io::Error::other("upstream started the range elsewhere"));
    }
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let meta = Meta {
        total,
        content_type,
    };
    *reg.meta.lock().unwrap() = Some(meta.clone());
    Ok((meta, FirstChunk { resp, start }))
}

/// The file's size and type, asked of upstream once per registration.
async fn meta_for(shared: &Shared, reg: &Registration) -> io::Result<Meta> {
    if let Some(m) = reg.meta.lock().unwrap().clone() {
        return Ok(m);
    }
    // A one-byte range is enough to read the total from `Content-Range`.
    let resp = upstream_range(shared, reg, 0, 0).await?;
    let total = resp
        .headers()
        .get(header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(total_from_content_range)
        .ok_or_else(|| io::Error::other("upstream gave no total length"))?;
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let meta = Meta {
        total,
        content_type,
    };
    *reg.meta.lock().unwrap() = Some(meta.clone());
    Ok(meta)
}

/// One bounded upstream range, with the registration's headers. Anything but 206 is an error:
/// a 200 would mean upstream sent the whole file, which is the shape this proxy exists to avoid.
async fn upstream_range(
    shared: &Shared,
    reg: &Registration,
    start: u64,
    end: u64,
) -> io::Result<reqwest::Response> {
    let mut req = shared
        .http
        .get(&reg.url)
        .header(header::RANGE, format!("bytes={start}-{end}"))
        // Ranges and content encoding do not mix. Without this, upstream may compress and answer
        // with an encoded whole file.
        .header(header::ACCEPT_ENCODING, "identity");
    for (name, value) in &reg.headers {
        req = req.header(name.as_str(), value.as_str());
    }
    let resp = tokio::time::timeout(shared.cfg.stall, req.send())
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "upstream stalled"))?
        .map_err(io::Error::other)?;
    if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(io::Error::other(format!(
            "upstream answered {}",
            resp.status()
        )));
    }
    Ok(resp)
}

/// Fetch `start..=end` one bounded chunk at a time, and send the bytes down `tx`.
///
/// `first` is the opening chunk when the caller already requested it (see [`open_first_chunk`]); it
/// is used instead of asking again. Stops when the client goes away (the send fails), so an
/// abandoned seek stops downloading.
async fn pump(
    shared: Arc<Shared>,
    reg: Arc<Registration>,
    start: u64,
    end: u64,
    mut first: Option<FirstChunk>,
    tx: mpsc::Sender<io::Result<Bytes>>,
) {
    let mut pos = start;
    while pos <= end {
        let chunk_end = (pos + shared.cfg.chunk - 1).min(end);
        let expected = chunk_end - pos + 1;

        let resp = match first.take().filter(|f| f.start == pos) {
            Some(f) => f.resp,
            None => match upstream_range(&shared, &reg, pos, chunk_end).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            },
        };

        let mut received = 0u64;
        let mut stream = resp.bytes_stream();
        loop {
            match tokio::time::timeout(shared.cfg.stall, stream.next()).await {
                Err(_) => {
                    let _ = tx
                        .send(Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "upstream stalled",
                        )))
                        .await;
                    return;
                }
                Ok(None) => break,
                Ok(Some(Err(e))) => {
                    let _ = tx.send(Err(io::Error::other(e))).await;
                    return;
                }
                Ok(Some(Ok(mut bytes))) => {
                    // An opening chunk was asked for before the client's window was known, so it
                    // can run past a short bounded read. The surplus is not the client's.
                    let room = expected - received;
                    let cut = bytes.len() as u64 > room;
                    if cut {
                        bytes.truncate(room as usize);
                    }
                    received += bytes.len() as u64;
                    if !bytes.is_empty() && tx.send(Ok(bytes)).await.is_err() {
                        return;
                    }
                    if cut {
                        break;
                    }
                }
            }
        }
        if received < expected {
            // A short chunk would silently skip bytes. Fail the stream instead.
            let _ = tx
                .send(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short upstream chunk",
                )))
                .await;
            return;
        }
        pos = chunk_end + 1;
    }
}

fn stream_body(rx: mpsc::Receiver<io::Result<Bytes>>) -> Body {
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item.map(Frame::data), rx))
    });
    BodyExt::boxed(StreamBody::new(stream))
}

fn status_only(status: StatusCode) -> Response<Body> {
    let mut r = Response::new(Empty::<Bytes>::new().map_err(|e| match e {}).boxed());
    *r.status_mut() = status;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_our_token_and_kind_reach_an_id() {
        assert_eq!(id_from("/tok/a/abc", "tok"), Some("abc"));
        assert_eq!(id_from("/bad/a/abc", "tok"), None);
        assert_eq!(id_from("/tok/v/abc", "tok"), None);
        assert_eq!(id_from("/tok/a/", "tok"), None);
    }
}
