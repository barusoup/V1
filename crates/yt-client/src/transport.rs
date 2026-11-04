// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The request path: one HTTP client, identity headers, account auth, retries and cookie rotation.

use std::fmt::Write as _;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::{HeaderMap, HeaderValue, SET_COOKIE};
use serde_json::{json, Value};

use crate::clients::{ClientKind, ClientSpec, LoginUse, VersionTable};
use crate::error::{Error, Result};
use crate::session::SharedSession;

pub const DEFAULT_BASE: &str = "https://music.youtube.com/youtubei/v1/";
pub const ORIGIN: &str = "https://music.youtube.com";

/// Whether a request may carry the account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Auth {
    /// Never sends the cookie, and never `onBehalfOfUser`. Use for anything the user should not be
    /// attributed for, such as typeahead.
    Anonymous,
    /// Sends the cookie when the session has one and the identity supports login.
    Account,
}

#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub base_url: String,
    /// Public web API key from the homepage. Optional; sent as `key=` when set.
    pub api_key: Option<String>,
    /// Outbound proxy, e.g. `http://127.0.0.1:8080`.
    pub proxy: Option<String>,
    /// Total time allowed for one request.
    pub timeout: Duration,
    /// Attempts for a request that failed at the network level.
    pub attempts: u32,
}

impl Default for TransportConfig {
    fn default() -> Self {
        TransportConfig {
            base_url: DEFAULT_BASE.to_owned(),
            api_key: None,
            proxy: None,
            timeout: Duration::from_secs(30),
            attempts: 3,
        }
    }
}

/// The InnerTube client. Clone freely; clones share the connection pool, session and versions.
#[derive(Clone)]
pub struct InnerTube {
    http: reqwest::Client,
    config: TransportConfig,
    pub session: SharedSession,
    pub versions: VersionTable,
}

impl InnerTube {
    pub fn new(config: TransportConfig, session: SharedSession) -> Result<Self> {
        // HTTP/1.1 on purpose: YouTube's pages answer an HTTP/2 request from this client with a
        // stream reset (PROTOCOL_ERROR), and the bootstrap reads the homepage. The pool below still
        // keeps connections warm, which is where the time was going.
        let mut builder = reqwest::Client::builder()
            .http1_only()
            .connect_timeout(Duration::from_secs(10))
            .timeout(config.timeout)
            .pool_idle_timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(8);
        if let Some(p) = &config.proxy {
            builder = builder.proxy(reqwest::Proxy::all(p)?);
        }
        Ok(InnerTube {
            http: builder.build()?,
            config,
            session,
            versions: VersionTable::default(),
        })
    }

    /// Use `key` as the public API key from now on. The bootstrap learns it after the client exists.
    pub fn set_api_key(&mut self, key: String) {
        self.config.api_key = Some(key);
    }

    /// The shared HTTP client, for the bootstrap and for anything else that needs the same
    /// connection pool and proxy.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// POST one InnerTube endpoint as `kind` and return the JSON body.
    ///
    /// `body` holds the endpoint's own fields. The `context` is added here.
    pub async fn post(
        &self,
        endpoint: &str,
        kind: ClientKind,
        auth: Auth,
        body: Value,
    ) -> Result<Value> {
        let spec = ClientSpec::of(kind);
        let session = self.session.snapshot();
        let use_account = auth == Auth::Account && spec.login == LoginUse::Supported;

        let mut full = body;
        let context = self.context(&spec, &session, use_account);
        if let Value::Object(map) = &mut full {
            map.insert("context".to_owned(), context);
        }

        // The music host and the video host answer different things for the same endpoint, and
        // some identities are only accepted on one of them. A caller that overrides `base_url`
        // (tests, a mirror) wins over the per-identity host.
        let base = if self.config.base_url == DEFAULT_BASE {
            format!("https://{}/youtubei/v1/", spec.host)
        } else {
            self.config.base_url.clone()
        };
        let mut url = format!("{base}{endpoint}?prettyPrint=false");
        // The API key is per-host. The music key is not accepted on the video host, so it is only
        // sent where it came from.
        if let Some(key) = &self.config.api_key {
            if spec.host == crate::clients::MUSIC_HOST_CONST {
                url.push_str("&key=");
                url.push_str(key);
            }
        }
        let payload = serde_json::to_vec(&full)?;

        let mut delay = Duration::from_millis(500);
        let mut attempt = 0;
        loop {
            attempt += 1;
            let headers = self.headers(&spec, &session, use_account, kind);
            let sent = self
                .http
                .post(&url)
                .headers(headers)
                .body(payload.clone())
                .send()
                .await;
            match sent {
                Ok(resp) => {
                    let status = resp.status();
                    self.absorb_cookies(resp.headers(), use_account);
                    let refused = status == reqwest::StatusCode::UNAUTHORIZED
                        || status == reqwest::StatusCode::FORBIDDEN;
                    if use_account && refused {
                        return Err(Error::SessionRejected);
                    }
                    if !status.is_success() {
                        return Err(Error::Status {
                            endpoint: endpoint.to_owned(),
                            status: status.as_u16(),
                        });
                    }
                    return Ok(resp.json::<Value>().await?);
                }
                Err(e) => {
                    let err = Error::Http(e);
                    // Retry only transient network failures, never an answer from YouTube.
                    if err.is_transient() && attempt < self.config.attempts {
                        tracing::debug!(attempt, endpoint, error = %err, "retrying");
                        tokio::time::sleep(delay).await;
                        delay *= 2;
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    /// `context` for a request. `onBehalfOfUser` is sent only with an account, because without a
    /// credential YouTube turns a request that would have worked into a 401.
    fn context(
        &self,
        spec: &ClientSpec,
        session: &crate::session::Session,
        use_account: bool,
    ) -> Value {
        let mut client = json!({
            "clientName": spec.name,
            "clientVersion": self.versions.get(spec.kind),
            "hl": session.hl,
            "gl": session.gl,
            "userAgent": spec.user_agent,
        });
        if let Some(vd) = &session.visitor_data {
            client["visitorData"] = json!(vd);
        }
        if let Some(d) = spec.device {
            client["osName"] = json!(d.os_name);
            client["osVersion"] = json!(d.os_version);
            client["deviceMake"] = json!(d.device_make);
            client["deviceModel"] = json!(d.device_model);
        }
        let mut ctx = json!({
            "client": client,
            "request": { "useSsl": true },
        });
        if spec.embedded {
            // An embedded player must say which third-party page it is on. Any non-YouTube URL
            // is accepted; without it the response is a bare UNPLAYABLE.
            ctx["thirdParty"] = json!({ "embedUrl": "https://www.reddit.com/" });
        }
        if use_account {
            if let Some(dsid) = &session.data_sync_id {
                ctx["user"] = json!({ "onBehalfOfUser": dsid });
            }
        }
        ctx
    }

    fn headers(
        &self,
        spec: &ClientSpec,
        session: &crate::session::Session,
        use_account: bool,
        kind: ClientKind,
    ) -> HeaderMap {
        let mut h = HeaderMap::new();
        let mut put = |name: &'static str, value: &str| {
            if let Ok(v) = HeaderValue::from_str(value) {
                h.insert(name, v);
            }
        };
        put("content-type", "application/json");
        put("accept", "application/json");
        put("x-goog-api-format-version", "1");
        // The numeric id. The name here is rejected or ignored by YouTube.
        put("x-youtube-client-name", &spec.id.to_string());
        put("x-youtube-client-version", &self.versions.get(kind));
        let origin = if spec.host == crate::clients::MUSIC_HOST_CONST {
            "https://music.youtube.com"
        } else {
            "https://www.youtube.com"
        };
        put("x-origin", origin);
        put("referer", &format!("{origin}/"));
        put("user-agent", spec.user_agent);
        put(
            "accept-language",
            &format!("{}-{},{};q=0.9", session.hl, session.gl, session.hl),
        );
        if let Some(vd) = &session.visitor_data {
            put("x-goog-visitor-id", vd);
        }
        if use_account {
            if let Some(jar) = &session.cookies {
                put("cookie", &jar.header());
                if let Some(sapisid) = jar.sapisid() {
                    put("authorization", &sapisid_hash(sapisid, ORIGIN));
                }
            }
        }
        h
    }

    /// Merge rotated cookies from a response into the session.
    fn absorb_cookies(&self, headers: &HeaderMap, use_account: bool) {
        if !use_account || headers.get(SET_COOKIE).is_none() {
            return;
        }
        let lines: Vec<&str> = headers
            .get_all(SET_COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .collect();
        self.session.update(|s| {
            if let Some(jar) = s.cookies.as_mut() {
                let changed = jar.merge_set_cookie(lines.iter().copied());
                if changed {
                    // Names only. Values are credentials, and this line ends up in logs.
                    let names: Vec<&str> = lines
                        .iter()
                        .filter_map(|l| l.split(';').next()?.split_once('=').map(|(n, _)| n.trim()))
                        .collect();
                    tracing::debug!(cookies = ?names, "kept rotated cookies");
                }
            }
        });
    }
}

/// `Authorization: SAPISIDHASH <ts>_<sha1("<ts> <SAPISID> <origin>")>`.
pub fn sapisid_hash(sapisid: &str, origin: &str) -> String {
    use sha1::{Digest, Sha1};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let digest = Sha1::digest(format!("{ts} {sapisid} {origin}").as_bytes());
    let mut hex = String::with_capacity(40);
    for b in digest {
        let _ = write!(hex, "{b:02x}");
    }
    format!("SAPISIDHASH {ts}_{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sapisid_hash_has_the_documented_shape() {
        let h = sapisid_hash("secret", ORIGIN);
        let rest = h.strip_prefix("SAPISIDHASH ").expect("prefix");
        let (ts, hex) = rest.split_once('_').expect("separator");
        assert!(ts.parse::<u64>().is_ok());
        assert_eq!(hex.len(), 40);
        assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()));
    }
}
