// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Discovering what YouTube expects from us right now: the visitor id, the web client version and
//! the public API key. None of these are stable, so they are read from YouTube each run instead of
//! being hardcoded.

use crate::clients::{ClientKind, VersionTable};
use crate::error::{Error, Result};
use crate::transport::ORIGIN;

const SW_JS_DATA: &str = "https://music.youtube.com/sw.js_data";
const HOMEPAGE: &str = "https://music.youtube.com/";
/// The homepage only embeds its config for a browser-like user agent.
const BROWSER_UA: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0";

/// What the homepage tells us.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebConfig {
    pub client_version: String,
    pub api_key: String,
    /// The player script for this build, e.g. `/s/player/<hash>/.../base.js`.
    pub player_js_path: Option<String>,
}

/// Fetch an anonymous visitor id.
///
/// `sw.js_data` is a JSON-ish blob whose visitor id is a quoted string starting with `Cg`.
pub async fn fetch_visitor_data(http: &reqwest::Client) -> Result<String> {
    let text = http
        .get(SW_JS_DATA)
        .header("user-agent", BROWSER_UA)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    parse_visitor_data(&text).ok_or(Error::NoVisitorData)
}

/// Find the visitor id in a `sw.js_data` body.
pub fn parse_visitor_data(body: &str) -> Option<String> {
    let mut rest = body;
    while let Some(start) = rest.find("\"Cg") {
        let after = &rest[start + 1..];
        let end = after.find('"')?;
        let candidate = &after[..end];
        let valid = candidate.len() >= 20
            && candidate
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'%' | b'_' | b'=' | b'-'));
        if valid {
            return Some(candidate.to_owned());
        }
        rest = &after[end..];
    }
    None
}

/// Read the web client version and API key from the homepage's embedded config.
pub async fn fetch_web_config(http: &reqwest::Client) -> Result<WebConfig> {
    let page = http
        .get(HOMEPAGE)
        .header("user-agent", BROWSER_UA)
        .header("origin", ORIGIN)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    parse_web_config(&page).ok_or(Error::NoClientConfig)
}

/// Pull `"KEY":"value"` pairs out of the homepage without a full HTML parser.
pub fn parse_web_config(page: &str) -> Option<WebConfig> {
    let client_version = quoted_value(page, "INNERTUBE_CLIENT_VERSION")?;
    let api_key = quoted_value(page, "INNERTUBE_API_KEY")?;
    let player_js_path = quoted_value(page, "jsUrl").filter(|p| p.starts_with("/s/player/"));
    Some(WebConfig {
        client_version,
        api_key,
        player_js_path,
    })
}

fn quoted_value(haystack: &str, key: &str) -> Option<String> {
    let marker = format!("\"{key}\":\"");
    let start = haystack.find(&marker)? + marker.len();
    let end = haystack[start..].find('"')?;
    let value = &haystack[start..start + end];
    (!value.is_empty()).then(|| value.to_owned())
}

/// Apply a discovered config: the web client version goes into the version table.
pub fn apply_web_config(versions: &VersionTable, config: &WebConfig) {
    versions.set(ClientKind::WebRemix, config.client_version.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_visitor_id_among_other_fields() {
        let body = r#")]}'
[["yt.sw.adr",null,[["en","US",null,"1.2.3.4",null,"Cgshort"]]]]"#;
        assert_eq!(
            parse_visitor_data(body),
            None,
            "too short to be a visitor id"
        );

        let long = format!(r#"x["CgtXeklxaTVscWpDWSiK{}"]"#, "abcdefghij".repeat(4));
        let got = parse_visitor_data(&long).expect("visitor id");
        assert!(got.starts_with("CgtXeklxaTVscWpDWSiK"));
    }

    #[test]
    fn reads_version_and_key_from_the_homepage() {
        let page = r#"..."INNERTUBE_CLIENT_VERSION":"1.20261027.10.00","x":1,"INNERTUBE_API_KEY":"AIzaKEY""#;
        let cfg = parse_web_config(page).expect("config");
        assert_eq!(cfg.client_version, "1.20261027.10.00");
        assert_eq!(cfg.api_key, "AIzaKEY");
    }

    #[test]
    fn reads_the_player_script_path() {
        let page = r#""INNERTUBE_CLIENT_VERSION":"1.2","INNERTUBE_API_KEY":"K","jsUrl":"/s/player/11a45ffb/player_es6.vflset/en_US/base.js""#;
        let cfg = parse_web_config(page).expect("config");
        assert_eq!(
            cfg.player_js_path.as_deref(),
            Some("/s/player/11a45ffb/player_es6.vflset/en_US/base.js")
        );
    }

    #[test]
    fn a_page_without_config_is_an_error_not_a_default() {
        assert_eq!(parse_web_config("<html></html>"), None);
    }
}
