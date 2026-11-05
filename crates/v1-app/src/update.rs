// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Startup self-update. Before the window exists, the app asks GitHub for the latest full release.
//! If it is newer than this build and its installer matches the published checksum, a helper script
//! waits for this process to exit, installs silently, and starts the updated app in its place. Any
//! failure (no network, a failed request, a bad download) is dropped and the app starts as usual.
//!
//! Only installed copies update. The installer leaves `uninstall.exe` beside the app, and a build
//! run from a target folder has none. Set `V1_NO_UPDATE` to skip the check.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

const CURRENT: &str = env!("CARGO_PKG_VERSION");
const LATEST_URL: &str = "https://api.github.com/repos/barusoup/V1/releases/latest";
const DOWNLOAD_PREFIX: &str = "https://github.com/barusoup/V1/releases/download/";
const CHECKSUMS: &str = "SHA256SUMS";
const ATTEMPTS: u32 = 3;
const RETRY_PAUSE: Duration = Duration::from_millis(300);
/// The release lookup, retries included, may take this long before the app gives up on it.
const CHECK_BUDGET: Duration = Duration::from_secs(6);
/// The installer download may take this long. Past it, the update waits for the next launch.
const DOWNLOAD_BUDGET: Duration = Duration::from_secs(90);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(u64, u64, u64);

impl Version {
    /// Accepts `1.2.3` and `v1.2.3`, the form release tags use.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.strip_prefix('v').unwrap_or(text);
        let mut parts = text.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        parts.next().is_none().then_some(Self(major, minor, patch))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.0, self.1, self.2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Asset {
    name: String,
    url: String,
}

/// The newest full release, with the installer and checksum file it published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub version: Version,
    installer: Asset,
    checksums: Asset,
}

/// Reads the body of `releases/latest`. The installer must be named `V1-Setup-<version>.exe`, and
/// both assets must come from this repository's release downloads.
pub fn parse_latest(json: &[u8]) -> Option<Release> {
    let value: serde_json::Value = serde_json::from_slice(json).ok()?;
    let version = Version::parse(value.get("tag_name")?.as_str()?)?;
    let assets = value.get("assets")?.as_array()?;
    let installer = find_asset(assets, &format!("V1-Setup-{version}.exe"))?;
    let checksums = find_asset(assets, CHECKSUMS)?;
    Some(Release { version, installer, checksums })
}

fn find_asset(assets: &[serde_json::Value], name: &str) -> Option<Asset> {
    let asset = assets.iter().find(|asset| asset.get("name").and_then(|n| n.as_str()) == Some(name))?;
    let url = asset.get("browser_download_url")?.as_str()?;
    url.starts_with(DOWNLOAD_PREFIX)
        .then(|| Asset { name: name.to_owned(), url: url.to_owned() })
}

/// The SHA-256 the checksum file lists for `file`, in lower-case hex.
pub fn expected_hash(checksums: &str, file: &str) -> Option<String> {
    checksums.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        let hash = parts.next()?;
        let name = parts.next()?.trim_start_matches('*');
        let well_formed = hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit());
        (name == file && well_formed).then(|| hash.to_ascii_lowercase())
    })
}

/// Runs the update check and, when a newer release is ready, installs it and restarts the app.
/// Returns only when there is nothing to do or the update could not be staged.
pub fn run_at_startup() {
    if std::env::var_os("V1_NO_UPDATE").is_some() {
        return;
    }
    let Some(install_dir) = installed_dir() else { return };
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return };
    let Ok(client) = reqwest::Client::builder()
        .user_agent(format!("V1/{CURRENT}"))
        .connect_timeout(Duration::from_secs(3))
        .build()
    else {
        return;
    };

    let Some(release) = runtime.block_on(async {
        tokio::time::timeout(CHECK_BUDGET, newer_release(&client)).await.ok().flatten()
    }) else {
        return;
    };
    let Some(installer) = runtime.block_on(async {
        tokio::time::timeout(DOWNLOAD_BUDGET, stage_installer(&client, &release)).await.ok().flatten()
    }) else {
        return;
    };

    if restart_into_update(&installer, &install_dir).is_ok() {
        std::process::exit(0);
    }
    let _ = std::fs::remove_file(&installer);
}

/// The folder the app is installed in, or `None` for a build run from anywhere else.
fn installed_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.to_path_buf();
    dir.join("uninstall.exe").is_file().then_some(dir)
}

async fn newer_release(client: &reqwest::Client) -> Option<Release> {
    let body = get(client, LATEST_URL, Duration::from_secs(4)).await?;
    let release = parse_latest(&body)?;
    (release.version > Version::parse(CURRENT)?).then_some(release)
}

/// Downloads the installer for `release` and checks it against the published checksum. Returns the
/// path of the verified file.
async fn stage_installer(client: &reqwest::Client, release: &Release) -> Option<PathBuf> {
    let checksums = get(client, &release.checksums.url, Duration::from_secs(10)).await?;
    let expected = expected_hash(std::str::from_utf8(&checksums).ok()?, &release.installer.name)?;
    let installer = get(client, &release.installer.url, DOWNLOAD_BUDGET).await?;
    let actual = format!("{:x}", Sha256::digest(&installer));
    if actual != expected {
        return None;
    }
    let dir = std::env::temp_dir().join("v1-update");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(&release.installer.name);
    std::fs::write(&path, &installer).ok()?;
    Some(path)
}

/// GET with a few attempts. Each attempt has its own `timeout`.
async fn get(client: &reqwest::Client, url: &str, timeout: Duration) -> Option<Vec<u8>> {
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(RETRY_PAUSE).await;
        }
        let result: reqwest::Result<Vec<u8>> = async {
            let response = client
                .get(url)
                .header(reqwest::header::ACCEPT, "application/vnd.github+json")
                .timeout(timeout)
                .send()
                .await?
                .error_for_status()?;
            Ok(response.bytes().await?.to_vec())
        }
        .await;
        if let Ok(body) = result {
            return Some(body);
        }
    }
    None
}

/// Writes a script that waits for this process to exit, runs the installer silently into
/// `install_dir`, and starts the app again. The script runs detached, so it outlives this process.
#[cfg(windows)]
fn restart_into_update(installer: &Path, install_dir: &Path) -> std::io::Result<()> {
    use std::os::windows::process::CommandExt;
    // CREATE_NO_WINDOW gives the script and the tools it runs a hidden console. Do not add
    // DETACHED_PROCESS: Windows then ignores CREATE_NO_WINDOW, every tool gets a visible window,
    // and the tasklist check stops seeing its output, so the installer could run over a live app.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    /// About two minutes of waiting, then the update is dropped instead of hanging.
    const WAIT_TRIES: u32 = 120;

    let pid = std::process::id();
    let script = installer.with_file_name("v1-update.cmd");
    let body = format!(
        "@echo off\r\n\
         set /a tries=0\r\n\
         :wait\r\n\
         tasklist /FI \"PID eq {pid}\" /NH | find \"{pid}\" >nul\r\n\
         if errorlevel 1 goto install\r\n\
         set /a tries+=1\r\n\
         if %tries% GEQ {WAIT_TRIES} goto abort\r\n\
         ping -n 2 127.0.0.1 >nul\r\n\
         goto wait\r\n\
         :install\r\n\
         \"{installer}\" /S /D={dir}\r\n\
         start \"\" \"{dir}\\v1.exe\"\r\n\
         :abort\r\n\
         del \"{installer}\" >nul 2>&1\r\n\
         (goto) 2>nul & del \"%~f0\"\r\n",
        installer = installer.display(),
        dir = install_dir.display(),
    );
    std::fs::write(&script, body)?;
    std::process::Command::new("cmd")
        .arg("/C")
        .arg(&script)
        .creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
        .spawn()?;
    Ok(())
}

#[cfg(not(windows))]
fn restart_into_update(_installer: &Path, _install_dir: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other("self-update is only built for Windows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_parse_with_and_without_the_v_prefix() {
        assert_eq!(Version::parse("v1.2.3"), Some(Version(1, 2, 3)));
        assert_eq!(Version::parse("0.10.0"), Some(Version(0, 10, 0)));
        assert_eq!(Version::parse("1.2"), None);
        assert_eq!(Version::parse("1.2.3.4"), None);
        assert_eq!(Version::parse("nightly"), None);
    }

    #[test]
    fn versions_order_numerically() {
        assert!(Version(0, 10, 0) > Version(0, 9, 9));
        assert!(Version(1, 0, 0) > Version(0, 99, 99));
        assert!(Version(0, 1, 1) > Version(0, 1, 0));
        assert!(Version(0, 1, 0) <= Version(0, 1, 0));
    }

    fn release_json(tag: &str, installer: &str) -> String {
        format!(
            r#"{{"tag_name":"{tag}","assets":[
                {{"name":"{installer}","browser_download_url":"https://github.com/barusoup/V1/releases/download/{tag}/{installer}"}},
                {{"name":"SHA256SUMS","browser_download_url":"https://github.com/barusoup/V1/releases/download/{tag}/SHA256SUMS"}}
            ]}}"#
        )
    }

    #[test]
    fn a_release_needs_its_installer_and_checksums() {
        let release = parse_latest(release_json("v0.2.0", "V1-Setup-0.2.0.exe").as_bytes()).unwrap();
        assert_eq!(release.version, Version(0, 2, 0));
        assert_eq!(release.installer.name, "V1-Setup-0.2.0.exe");
        assert!(parse_latest(release_json("v0.2.0", "V1-Setup-0.1.0.exe").as_bytes()).is_none());
        assert!(parse_latest(release_json("v0.2.0", "Other.exe").as_bytes()).is_none());
    }

    #[test]
    fn assets_from_elsewhere_are_refused() {
        let json = r#"{"tag_name":"v0.2.0","assets":[
            {"name":"V1-Setup-0.2.0.exe","browser_download_url":"https://example.com/V1-Setup-0.2.0.exe"},
            {"name":"SHA256SUMS","browser_download_url":"https://github.com/barusoup/V1/releases/download/v0.2.0/SHA256SUMS"}
        ]}"#;
        assert!(parse_latest(json.as_bytes()).is_none());
    }

    #[test]
    fn checksums_are_matched_by_file_name() {
        let hash = "a".repeat(64);
        let sums = format!("{hash}  V1-Setup-0.2.0.exe\r\n{}  other.exe\r\n", "b".repeat(64));
        assert_eq!(expected_hash(&sums, "V1-Setup-0.2.0.exe"), Some(hash));
        assert_eq!(expected_hash(&sums, "missing.exe"), None);
        assert_eq!(expected_hash("short  V1-Setup-0.2.0.exe", "V1-Setup-0.2.0.exe"), None);
    }
}
