// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The JavaScript engine that runs YouTube's player script.
//!
//! QuickJS is single-threaded, so the engine lives on one thread of its own. Callers send a
//! request and wait for the answer on a oneshot channel, which keeps the engine off every async
//! worker thread.

use std::sync::mpsc;

use rquickjs::{Context, Runtime};
use tokio::sync::oneshot;

use crate::discover::Entries;

const SHIM: &str = include_str!("shim.js");

/// The closing line of the player's module, as it appears in `base.js`. The exports are placed
/// just before it, so they run inside the closure and can see the player's own functions.
const CLOSURE_TAIL: &str = "})(_yt_player);";

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum EngineError {
    #[error("the player script did not load: {0}")]
    Load(String),
    #[error("deciphering failed: {0}")]
    Call(String),
    #[error("the engine stopped")]
    Stopped,
}

enum Request {
    Build {
        url: String,
        cipher: Option<(String, String)>,
        reply: oneshot::Sender<Result<String, EngineError>>,
    },
    Sig {
        s: String,
        reply: oneshot::Sender<Result<String, EngineError>>,
    },
}

/// Handle to the engine thread.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::Sender<Request>,
}

impl Engine {
    /// Start the engine thread with `player_src` loaded. Returns once the player has loaded, or
    /// with the reason it did not.
    pub fn spawn(player_src: String, entries: Entries) -> Result<Engine, EngineError> {
        let (tx, rx) = mpsc::channel::<Request>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), EngineError>>();
        std::thread::Builder::new()
            .name("yt-js".to_owned())
            .spawn(move || match Worker::new(&player_src, &entries) {
                Ok(worker) => {
                    let _ = ready_tx.send(Ok(()));
                    for request in rx {
                        match request {
                            Request::Build { url, cipher, reply } => {
                                let _ = reply.send(worker.build(&url, cipher));
                            }
                            Request::Sig { s, reply } => {
                                let _ = reply.send(worker.sig(&s));
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| EngineError::Load(e.to_string()))?;
        ready_rx.recv().map_err(|_| EngineError::Stopped)??;
        Ok(Engine { tx })
    }

    /// The URL with its signature set and its `n` transform applied.
    pub async fn build(
        &self,
        url: &str,
        cipher: Option<(&str, &str)>,
    ) -> Result<String, EngineError> {
        let (reply, answer) = oneshot::channel();
        self.tx
            .send(Request::Build {
                url: url.to_owned(),
                cipher: cipher.map(|(s, sp)| (s.to_owned(), sp.to_owned())),
                reply,
            })
            .map_err(|_| EngineError::Stopped)?;
        answer.await.map_err(|_| EngineError::Stopped)?
    }

    /// The signature for a ciphered `s` value, on its own.
    pub async fn sig(&self, s: &str) -> Result<String, EngineError> {
        let (reply, answer) = oneshot::channel();
        self.tx
            .send(Request::Sig {
                s: s.to_owned(),
                reply,
            })
            .map_err(|_| EngineError::Stopped)?;
        answer.await.map_err(|_| EngineError::Stopped)?
    }
}

struct Worker {
    // The runtime must outlive the context. Keep both for the life of the thread.
    _runtime: Runtime,
    context: Context,
}

impl Worker {
    fn new(player_src: &str, entries: &Entries) -> Result<Worker, EngineError> {
        let idx = player_src
            .rfind(CLOSURE_TAIL)
            .ok_or_else(|| EngineError::Load("the player's closure tail is missing".into()))?;
        let exports = exports_for(entries);
        // Call the player's closure with the global as `this`, as a browser does. Without it the
        // player's `var window = this` reads undefined.
        let body = format!(
            "{}{}}}).call(globalThis, _yt_player);{}",
            &player_src[..idx],
            exports,
            &player_src[idx + CLOSURE_TAIL.len()..]
        );

        let runtime = Runtime::new().map_err(|e| EngineError::Load(e.to_string()))?;
        runtime.set_memory_limit(512 * 1024 * 1024);
        let context = Context::full(&runtime).map_err(|e| EngineError::Load(e.to_string()))?;

        context.with(|ctx| {
            ctx.eval::<(), _>("var _yt_player = {};")
                .map_err(|e| EngineError::Load(js_error(&ctx, e)))?;
            ctx.eval::<(), _>(SHIM)
                .map_err(|e| EngineError::Load(js_error(&ctx, e)))?;
            ctx.eval::<(), _>(body.as_str())
                .map_err(|e| EngineError::Load(js_error(&ctx, e)))?;
            Ok::<_, EngineError>(())
        })?;

        Ok(Worker {
            _runtime: runtime,
            context,
        })
    }

    fn build(&self, url: &str, cipher: Option<(String, String)>) -> Result<String, EngineError> {
        let (s, sp) = match cipher {
            Some((s, sp)) => (js_string(&s), js_string(&sp)),
            None => ("null".to_owned(), "null".to_owned()),
        };
        let expr = format!("globalThis.__ytjs.build({}, {s}, {sp})", js_string(url));
        self.eval_string(&expr)
    }

    fn sig(&self, s: &str) -> Result<String, EngineError> {
        self.eval_string(&format!("globalThis.__ytjs.sig({})", js_string(s)))
    }

    fn eval_string(&self, expr: &str) -> Result<String, EngineError> {
        self.context.with(|ctx| {
            ctx.eval::<String, _>(expr)
                .map_err(|e| EngineError::Call(js_error(&ctx, e)))
        })
    }
}

/// A JavaScript string literal for `s`. JSON strings are valid JavaScript string literals.
fn js_string(s: &str) -> String {
    serde_json::to_string(s).expect("a string always serialises")
}

/// The message of the exception the engine last threw, or the engine's own error text.
fn js_error(ctx: &rquickjs::Ctx<'_>, err: rquickjs::Error) -> String {
    match ctx.catch().as_exception() {
        Some(ex) => ex.message().unwrap_or_else(|| err.to_string()),
        None => err.to_string(),
    }
}

/// Exports injected inside the player's closure: the signature pipeline and the URL builder.
///
/// The builder is picked at runtime from the URL class's own methods, by asking each one for a
/// URL on a throwaway instance. That keeps the dispatch arguments out of the code entirely.
fn exports_for(e: &Entries) -> String {
    let p = &e.sig;
    format!(
        r#";
globalThis.__ytjs = (function () {{
  var builder = null;
  function pickBuilder() {{
    if (builder) return builder;
    var probe = "https://x.googlevideo.com/videoplayback?itag=140&n=probe";
    var names = Object.getOwnPropertyNames(g.{class}.prototype);
    for (var i = 0; i < names.length; i++) {{
      var name = names[i];
      if (name === "constructor") continue;
      try {{
        var out = new g.{class}(probe, true)[name]();
        if (typeof out === "string" && out.indexOf("videoplayback") >= 0) {{
          builder = name;
          return name;
        }}
      }} catch (err) {{}}
    }}
    throw new Error("no URL builder found on the player's URL class");
  }}
  function sig(s) {{
    return {second}(
      {second_op},
      {second_key},
      {first}({first_op}, {first_key}, {helper}({helper_op}, {helper_key}, s))
    );
  }}
  return {{
    sig: function (s) {{
      return sig(s);
    }},
    build: function (url, s, sp) {{
      var c = new g.{class}(url, true);
      if (s !== null) c.set(sp, sig(s));
      return c[pickBuilder()]();
    }}
  }};
}})();
"#,
        class = e.url_class,
        first = p.first.name,
        first_op = p.first.op,
        first_key = p.first.key,
        second = p.second.name,
        second_op = p.second.op,
        second_key = p.second.key,
        helper = p.helper.name,
        helper_op = p.helper.op,
        helper_key = p.helper.key,
    )
}
