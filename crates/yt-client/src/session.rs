// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Session state: visitor identity, the account cookie jar, and locale.
//!
//! Shared behind an `Arc<RwLock<_>>`. Locks are never held across an `.await`.

use std::sync::{Arc, RwLock};

/// Cookies from the account, as a name → value list in the order they were first seen.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CookieJar(Vec<(String, String)>);

impl CookieJar {
    /// Parse a `Cookie:` header value (`a=1; b=2`).
    pub fn parse(header: &str) -> CookieJar {
        CookieJar(
            header
                .split(';')
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
                .filter(|(k, v)| !k.is_empty() && !v.is_empty())
                .collect(),
        )
    }

    /// The value of a cookie by name.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// The `Cookie:` header value.
    pub fn header(&self) -> String {
        self.0
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Apply `Set-Cookie` lines from a response. Returns true if anything changed.
    ///
    /// Google rotates `__Secure-*PSIDTS` on responses. Dropping the rotated value
    /// invalidates the login a few hours later, so every response's cookies are merged here.
    /// Deletions are ignored: a cookie Google wants gone is dead anyway, and honouring a stray
    /// deletion could sign the user out.
    pub fn merge_set_cookie<'a>(&mut self, lines: impl IntoIterator<Item = &'a str>) -> bool {
        let mut changed = false;
        for line in lines {
            let pair = line.split(';').next().unwrap_or_default();
            let Some((name, value)) = pair.split_once('=') else {
                continue;
            };
            let (name, value) = (name.trim(), value.trim());
            if name.is_empty() || value.is_empty() {
                continue;
            }
            match self.0.iter_mut().find(|(k, _)| k == name) {
                Some(entry) if entry.1 == value => {}
                Some(entry) => {
                    entry.1 = value.to_owned();
                    changed = true;
                }
                None => {
                    self.0.push((name.to_owned(), value.to_owned()));
                    changed = true;
                }
            }
        }
        changed
    }

    /// The SAPISID used to sign requests. Modern accounts carry `__Secure-3PAPISID` instead.
    pub fn sapisid(&self) -> Option<&str> {
        self.get("SAPISID")
            .or_else(|| self.get("__Secure-3PAPISID"))
    }

    /// Read a Netscape `cookies.txt` (what browser extensions export), keeping only the rows
    /// that belong to a Google/YouTube domain. Lines starting with `#` are comments; the
    /// `#HttpOnly_` prefix is part of the format, not a comment.
    pub fn from_netscape(text: &str) -> CookieJar {
        let mut jar = CookieJar::default();
        for line in text.lines() {
            let line = line.trim();
            let line = line.strip_prefix("#HttpOnly_").unwrap_or(line);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // domain, includeSubdomains, path, secure, expiry, name, value
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 7 {
                continue;
            }
            let domain = fields[0].trim_start_matches('.');
            if !(domain.ends_with("youtube.com") || domain.ends_with("google.com")) {
                continue;
            }
            let (name, value) = (fields[5].trim(), fields[6].trim());
            if name.is_empty() || value.is_empty() {
                continue;
            }
            match jar.0.iter_mut().find(|(k, _)| k == name) {
                Some(entry) => entry.1 = value.to_owned(),
                None => jar.0.push((name.to_owned(), value.to_owned())),
            }
        }
        jar
    }

    /// Whether the jar carries the cookies a signed-in InnerTube request needs.
    pub fn is_signed_in(&self) -> bool {
        self.sapisid().is_some() && self.get("SID").or_else(|| self.get("__Secure-1PSID")).is_some()
    }
}

/// Everything that identifies this client to YouTube.
#[derive(Debug, Clone)]
pub struct Session {
    /// Anonymous visitor id, from `sw.js_data`. Sent as `X-Goog-Visitor-Id` and in the context.
    pub visitor_data: Option<String>,
    /// Account cookies. `None` means anonymous.
    pub cookies: Option<CookieJar>,
    /// Which account on a multi-account login. Sent as `onBehalfOfUser`, only with cookies.
    pub data_sync_id: Option<String>,
    /// UI language (`hl`), e.g. `en`.
    pub hl: String,
    /// Region (`gl`), e.g. `US`.
    pub gl: String,
}

impl Default for Session {
    fn default() -> Self {
        Session {
            visitor_data: None,
            cookies: None,
            data_sync_id: None,
            hl: "en".to_owned(),
            gl: "US".to_owned(),
        }
    }
}

/// A shared handle to the session. Cheap to clone; every clone sees the same state.
#[derive(Debug, Clone, Default)]
pub struct SharedSession(pub(crate) Arc<RwLock<Session>>);

impl SharedSession {
    pub fn new(session: Session) -> Self {
        SharedSession(Arc::new(RwLock::new(session)))
    }

    pub fn snapshot(&self) -> Session {
        self.0.read().unwrap().clone()
    }

    pub fn update<R>(&self, f: impl FnOnce(&mut Session) -> R) -> R {
        f(&mut self.0.write().unwrap())
    }

    pub fn is_logged_in(&self) -> bool {
        self.0
            .read()
            .unwrap()
            .cookies
            .as_ref()
            .is_some_and(|jar| jar.is_signed_in())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_cookie_header() {
        let jar = CookieJar::parse("SAPISID=abc; __Secure-1PSIDTS=x1; empty=");
        assert_eq!(jar.get("SAPISID"), Some("abc"));
        assert_eq!(jar.get("empty"), None);
        assert_eq!(jar.sapisid(), Some("abc"));
    }

    #[test]
    fn modern_accounts_sign_with_3papisid() {
        let jar = CookieJar::parse("__Secure-3PAPISID=mod; x=1");
        assert_eq!(jar.sapisid(), Some("mod"));
    }

    #[test]
    fn rotated_cookies_replace_in_place_and_new_ones_append() {
        let mut jar = CookieJar::parse("SAPISID=abc; __Secure-1PSIDTS=old");
        let changed = jar.merge_set_cookie([
            "__Secure-1PSIDTS=new; Domain=.youtube.com; Path=/",
            "NID=1; Secure",
        ]);
        assert!(changed);
        assert_eq!(jar.header(), "SAPISID=abc; __Secure-1PSIDTS=new; NID=1");
    }

    #[test]
    fn identical_or_deleting_set_cookie_changes_nothing() {
        let mut jar = CookieJar::parse("SAPISID=abc");
        assert!(!jar.merge_set_cookie(["SAPISID=abc; Path=/"]));
        assert!(!jar.merge_set_cookie(["SAPISID=; Expires=Thu, 01 Jan 1970 00:00:00 GMT"]));
        assert_eq!(jar.get("SAPISID"), Some("abc"));
    }

    #[test]
    fn a_netscape_export_keeps_youtube_rows_and_the_httponly_prefix() {
        let text = "\
# Netscape HTTP Cookie File
.youtube.com\tTRUE\t/\tTRUE\t0\tSAPISID\tabc
#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\t__Secure-3PSID\tdef
.example.com\tTRUE\t/\tFALSE\t0\tOTHER\tzzz
.google.com\tFALSE\t/\tFALSE\t0\tNID\tghi
malformed line
";
        let jar = CookieJar::from_netscape(text);
        assert_eq!(jar.get("SAPISID"), Some("abc"));
        assert_eq!(jar.get("__Secure-3PSID"), Some("def"));
        assert_eq!(jar.get("NID"), Some("ghi"));
        assert_eq!(jar.get("OTHER"), None);
    }

    #[test]
    fn a_jar_without_a_signing_cookie_is_not_a_login() {
        assert!(!CookieJar::parse("SAPISID=abc").is_signed_in());
        assert!(CookieJar::parse("SAPISID=abc; __Secure-1PSID=def").is_signed_in());
    }
}
