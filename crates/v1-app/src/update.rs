// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! In-app update checks. A worker thread asks GitHub for the latest full release every few hours,
//! so an open app can tell the listener when a new version exists instead of waiting for the next
//! launch. Installing is still a deliberate action: the popup's Update button stages the verified
//! Windows setup, closes the app, installs it, and starts the new build again.
//!
//! Only installed copies update: the Windows installer leaves `uninstall.exe` beside the app, so a
//! build run from a target folder sits this out. Set `V1_NO_UPDATE` to skip checks and installs.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

const CURRENT: &str = env!("CARGO_PKG_VERSION");
const LATEST_URL: &str = "https://api.github.com/repos/barusoup/V1/releases/latest";
const DOWNLOAD_PREFIX: &str = "https://github.com/barusoup/V1/releases/download/";
const CHECKSUMS: &str = "SHA256SUMS";
const ATTEMPTS: u32 = 3;
const RETRY_PAUSE: Duration = Duration::from_millis(300);
/// The release lookup, retries included, may take this long before the app gives up on it.
const CHECK_BUDGET: Duration = Duration::from_secs(6);
/// The package download may take this long. A Windows setup is ~90 MB, so a slow line needs more
/// than a minute; the download runs beside the app, so a long budget only delays the update itself.
const DOWNLOAD_BUDGET: Duration = Duration::from_secs(90);
/// Set by the installer, which starts the updated copy after replacing the old one.
pub(crate) const UPDATED_FLAG: &str = "--updated";
/// Marker covering the gap between the old process exiting and the installer starting the new one.
const HANDOFF_FILE: &str = "update-in-progress";
/// A marker older than this is treated as a crashed update, so a broken install cannot lock the app
/// out for long. The helper script waits up to two minutes before it gives up.
const HANDOFF_TTL: Duration = Duration::from_secs(300);
/// Let the first frame settle before the worker builds a runtime and opens a TLS connection.
const START_DELAY: Duration = Duration::from_secs(2);
/// How often a quiet session looks again. The check is a single small request; six hours keeps it
/// out of the way while still catching a release during a long listening session.
const CHECK_EVERY: Duration = Duration::from_secs(6 * 60 * 60);
/// A check that could not reach GitHub waits this long before trying again.
const RETRY_EVERY: Duration = Duration::from_secs(20 * 60);

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

/// A newer release that the UI can offer to install.
#[derive(Debug, Clone)]
pub struct Available {
    pub version: Version,
    release: Release,
}

/// What the background worker reports back to the UI thread.
#[derive(Debug)]
pub enum UpdateEvent {
    /// A newer release is ready. Sent once per version unless a manual check asks again.
    Available(Available),
    /// A manual check found nothing newer.
    UpToDate,
    /// A manual check or install could not finish.
    Failed(String),
}

/// The release asset this build installs from: the Windows setup program.
fn package_name(version: Version) -> String {
    format!("V1-Setup-{version}.exe")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Asset {
    name: String,
    url: String,
}

/// The newest full release, with the package and checksum file it published.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Release {
    version: Version,
    package: Asset,
    checksums: Asset,
}

/// Reads the body of `releases/latest`. The package must carry this platform's name (see
/// [`package_name`]), and both assets must come from this repository's release downloads.
fn parse_latest(json: &[u8]) -> Option<Release> {
    let value: serde_json::Value = serde_json::from_slice(json).ok()?;
    let version = Version::parse(value.get("tag_name")?.as_str()?)?;
    let assets = value.get("assets")?.as_array()?;
    let package = find_asset(assets, &package_name(version))?;
    let checksums = find_asset(assets, CHECKSUMS)?;
    Some(Release {
        version,
        package,
        checksums,
    })
}

fn find_asset(assets: &[serde_json::Value], name: &str) -> Option<Asset> {
    let asset = assets
        .iter()
        .find(|asset| asset.get("name").and_then(|n| n.as_str()) == Some(name))?;
    let url = asset.get("browser_download_url")?.as_str()?;
    url.starts_with(DOWNLOAD_PREFIX).then(|| Asset {
        name: name.to_owned(),
        url: url.to_owned(),
    })
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

/// Start the update worker. The returned sender asks for an immediate manual check; dropping it
/// stops the worker at its next wake-up. `None` means this build cannot update itself (a development
/// build, `--shot`, or `V1_NO_UPDATE`).
pub fn start_worker<F>(on_event: F) -> Option<Sender<bool>>
where
    F: Fn(UpdateEvent) + Send + 'static,
{
    if std::env::var_os("V1_NO_UPDATE").is_some() || install_target().is_none() {
        return None;
    }
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("v1-update".to_owned())
        .spawn(move || {
            let mut announced: Option<Version> = None;
            let mut wait = START_DELAY;
            loop {
                let manual = match rx.recv_timeout(wait) {
                    Ok(manual) => manual,
                    Err(RecvTimeoutError::Timeout) => false,
                    Err(RecvTimeoutError::Disconnected) => return,
                };
                match check_once() {
                    Ok(Some(update)) => {
                        let version = update.version;
                        if announced != Some(version) || manual {
                            announced = Some(version);
                            on_event(UpdateEvent::Available(update));
                        }
                        wait = CHECK_EVERY;
                    }
                    Ok(None) => {
                        if manual {
                            on_event(UpdateEvent::UpToDate);
                        }
                        wait = CHECK_EVERY;
                    }
                    Err(error) => {
                        if manual {
                            on_event(UpdateEvent::Failed(error));
                        }
                        wait = RETRY_EVERY;
                    }
                }
            }
        });
    worker.ok().map(|_| tx)
}

/// Check once, on the worker thread. `Ok(None)` means this build is current.
fn check_once() -> Result<Option<Available>, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_| "The update checker could not start.".to_owned())?;
    let client = reqwest::Client::builder()
        .user_agent(format!("V1/{CURRENT}"))
        .connect_timeout(Duration::from_secs(3))
        .build()
        .map_err(|_| "The update checker could not start.".to_owned())?;
    runtime.block_on(async {
        let release = tokio::time::timeout(CHECK_BUDGET, newer_release(&client))
            .await
            .map_err(|_| "The update check timed out.".to_owned())??;
        Ok(release.map(|release| Available {
            version: release.version,
            release,
        }))
    })
}

/// Download and verify the release, then hand off to the installer. This exits the process on
/// success; a failure comes back as a message for the popup.
pub fn install(update: &Available) -> Result<(), String> {
    if std::env::var_os("V1_NO_UPDATE").is_some() {
        return Err("Updates are turned off for this launch.".to_owned());
    }
    let target = install_target()
        .ok_or_else(|| "Updates are only available in an installed copy of V1.".to_owned())?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_| "The updater could not start.".to_owned())?;
    let client = reqwest::Client::builder()
        .user_agent(format!("V1/{CURRENT}"))
        .connect_timeout(Duration::from_secs(3))
        .build()
        .map_err(|_| "The updater could not start.".to_owned())?;
    let package = runtime
        .block_on(async {
            tokio::time::timeout(DOWNLOAD_BUDGET, stage_package(&client, &update.release))
                .await
                .ok()
                .flatten()
        })
        .ok_or_else(|| "The update could not be downloaded. Check your connection.".to_owned())?;

    if begin_handoff(update.version).is_err() {
        // Without the marker another launch could race the installer into the old build, so leave
        // the update for the next attempt rather than take that chance.
        let _ = std::fs::remove_file(&package);
        return Err("The update could not be prepared.".to_owned());
    }
    if let Err(error) = restart_into_update(&package, &target) {
        clear_handoff();
        let _ = std::fs::remove_file(&package);
        return Err(format!("The installer could not start: {error}"));
    }
    std::process::exit(0);
}

/// What an update replaces: the folder the Windows app is installed in. `None` for a build run
/// from anywhere else.
pub(crate) fn install_target() -> Option<PathBuf> {
    installed_target_for(&std::env::current_exe().ok()?)
}

fn installed_target_for(exe: &Path) -> Option<PathBuf> {
    let dir = exe.parent()?.to_path_buf();
    dir.join("uninstall.exe").is_file().then_some(dir)
}

fn handoff_path() -> PathBuf {
    crate::data_dir().join(HANDOFF_FILE)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// Record that this process is about to exit for an installer. The marker covers the gap until the
/// updated copy starts with [`UPDATED_FLAG`] and clears it.
fn begin_handoff(version: Version) -> std::io::Result<()> {
    let dir = crate::data_dir();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(handoff_path(), format!("{version}\n{}\n", unix_now()))
}

fn clear_handoff() {
    let _ = std::fs::remove_file(handoff_path());
}

fn parse_handoff(text: &str) -> Option<(Version, u64)> {
    let mut lines = text.lines();
    let version = Version::parse(lines.next()?.trim())?;
    let written = lines.next()?.trim().parse().ok()?;
    Some((version, written))
}

fn handoff_pending(text: &str, now: u64) -> bool {
    let Some((_, written)) = parse_handoff(text) else {
        return false;
    };
    // A clock that moved backwards still counts as fresh; a crashed update ages out.
    written > now || now.saturating_sub(written) <= HANDOFF_TTL.as_secs()
}

/// Whether this launch must exit because an update is being installed. The installer starts the
/// updated copy with [`UPDATED_FLAG`]; that launch clears the marker and continues.
pub fn blocked_by_update(from_installer: bool) -> bool {
    if from_installer {
        clear_handoff();
        return false;
    }
    let Ok(text) = std::fs::read_to_string(handoff_path()) else {
        return false;
    };
    let pending = handoff_pending(&text, unix_now());
    if !pending {
        clear_handoff();
    }
    pending
}

async fn newer_release(client: &reqwest::Client) -> Result<Option<Release>, String> {
    let body = get(client, LATEST_URL, Duration::from_secs(4)).await?;
    let release = parse_latest(&body)
        .ok_or_else(|| "The release information could not be read.".to_owned())?;
    let current = Version::parse(CURRENT).ok_or_else(|| "This build has no version.".to_owned())?;
    Ok((release.version > current).then_some(release))
}

/// Downloads the package for `release` and checks it against the published checksum. Returns the
/// path of the verified file.
async fn stage_package(client: &reqwest::Client, release: &Release) -> Option<PathBuf> {
    let checksums = get(client, &release.checksums.url, Duration::from_secs(10))
        .await
        .ok()?;
    let expected = expected_hash(std::str::from_utf8(&checksums).ok()?, &release.package.name)?;
    let package = get(client, &release.package.url, DOWNLOAD_BUDGET)
        .await
        .ok()?;
    let actual = format!("{:x}", Sha256::digest(&package));
    if actual != expected {
        return None;
    }
    let dir = std::env::temp_dir().join("v1-update");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(&release.package.name);
    std::fs::write(&path, &package).ok()?;
    Some(path)
}

/// GET with a few attempts. Each attempt has its own `timeout`.
async fn get(client: &reqwest::Client, url: &str, timeout: Duration) -> Result<Vec<u8>, String> {
    let mut last = "GitHub could not be reached.".to_owned();
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(RETRY_PAUSE).await;
        }
        let result: Result<Vec<u8>, String> = async {
            let response = client
                .get(url)
                .header(reqwest::header::ACCEPT, "application/vnd.github+json")
                .timeout(timeout)
                .send()
                .await
                .map_err(|error| {
                    if error.is_timeout() {
                        "The update check timed out.".to_owned()
                    } else {
                        "GitHub could not be reached.".to_owned()
                    }
                })?;
            let response = response
                .error_for_status()
                .map_err(|error| match error.status() {
                    Some(status) => format!("GitHub returned {status}."),
                    None => "GitHub could not be reached.".to_owned(),
                })?;
            Ok(response
                .bytes()
                .await
                .map(|body| body.to_vec())
                .map_err(|_| "The update download was interrupted.".to_owned())?)
        }
        .await;
        match result {
            Ok(body) => return Ok(body),
            Err(error) => last = error,
        }
    }
    Err(last)
}

/// About two minutes of waiting, then the update is dropped instead of hanging.
const WAIT_TRIES: u32 = 120;

/// The detached script that waits for this process to exit, installs silently, and starts the
/// updated copy with [`UPDATED_FLAG`]. On any abort it also removes the handoff marker so a failed
/// install cannot lock the app out.
#[cfg(windows)]
fn update_script(pid: u32, installer: &Path, install_dir: &Path, handoff: &Path) -> String {
    format!(
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
         if errorlevel 1 goto abort\r\n\
         start \"\" \"{dir}\\v1.exe\" {updated}\r\n\
         goto cleanup\r\n\
         :abort\r\n\
         del \"{handoff}\" >nul 2>&1\r\n\
         :cleanup\r\n\
         del \"{installer}\" >nul 2>&1\r\n\
         (goto) 2>nul & del \"%~f0\"\r\n",
        installer = installer.display(),
        dir = install_dir.display(),
        handoff = handoff.display(),
        updated = UPDATED_FLAG,
    )
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

    let pid = std::process::id();
    let script = installer.with_file_name("v1-update.cmd");
    let body = update_script(pid, installer, install_dir, &handoff_path());
    std::fs::write(&script, body)?;
    std::process::Command::new("cmd")
        .arg("/C")
        .arg(&script)
        .creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
        .spawn()?;
    Ok(())
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

    fn release_json(tag: &str, package: &str) -> String {
        format!(
            r#"{{"tag_name":"{tag}","assets":[
                {{"name":"{package}","browser_download_url":"https://github.com/barusoup/V1/releases/download/{tag}/{package}"}},
                {{"name":"SHA256SUMS","browser_download_url":"https://github.com/barusoup/V1/releases/download/{tag}/SHA256SUMS"}}
            ]}}"#
        )
    }

    #[test]
    fn a_release_needs_its_package_and_checksums() {
        let name = package_name(Version(0, 2, 0));
        let release = parse_latest(release_json("v0.2.0", &name).as_bytes()).unwrap();
        assert_eq!(release.version, Version(0, 2, 0));
        assert_eq!(release.package.name, name);
        // A release whose package is named for another version, or not at all, is not ours.
        let wrong_version = package_name(Version(0, 1, 0));
        assert!(parse_latest(release_json("v0.2.0", &wrong_version).as_bytes()).is_none());
        assert!(parse_latest(release_json("v0.2.0", "Other.bin").as_bytes()).is_none());
    }

    /// The old macOS and Linux assets must never be installed by a Windows build.
    #[test]
    fn packaging_from_retired_platforms_is_refused() {
        for other in ["V1-0.2.0-arm64.dmg", "V1-0.2.0-x86_64.AppImage"] {
            assert!(parse_latest(release_json("v0.2.0", other).as_bytes()).is_none());
        }
    }

    #[test]
    fn assets_from_elsewhere_are_refused() {
        let json = format!(
            r#"{{"tag_name":"v0.2.0","assets":[
            {{"name":"{}","browser_download_url":"https://example.com/package"}},
            {{"name":"SHA256SUMS","browser_download_url":"https://github.com/barusoup/V1/releases/download/v0.2.0/SHA256SUMS"}}
        ]}}"#,
            package_name(Version(0, 2, 0))
        );
        assert!(parse_latest(json.as_bytes()).is_none());
    }

    #[test]
    fn checksums_are_matched_by_file_name() {
        let hash = "a".repeat(64);
        let sums = format!(
            "{hash}  V1-Setup-0.2.0.exe\r\n{}  other.exe\r\n",
            "b".repeat(64)
        );
        assert_eq!(expected_hash(&sums, "V1-Setup-0.2.0.exe"), Some(hash));
        assert_eq!(expected_hash(&sums, "missing.exe"), None);
        assert_eq!(
            expected_hash("short  V1-Setup-0.2.0.exe", "V1-Setup-0.2.0.exe"),
            None
        );
    }

    #[test]
    fn update_handoff_expires_after_its_ttl() {
        let now = 1_700_000_000;
        let text = format!("0.1.7\n{now}\n");
        assert!(handoff_pending(&text, now));
        assert!(handoff_pending(&text, now + HANDOFF_TTL.as_secs()));
        assert!(!handoff_pending(&text, now + HANDOFF_TTL.as_secs() + 1));
        // A clock that moved backwards must not make a live handoff look stale.
        assert!(handoff_pending(&text, now - 10));
    }

    #[test]
    fn malformed_update_handoffs_do_not_block_startup() {
        let now = 1_700_000_000;
        assert!(!handoff_pending("", now));
        assert!(!handoff_pending("0.1.7", now));
        assert!(!handoff_pending("nightly\n1700000000\n", now));
        assert!(!handoff_pending("0.1.7\nnot-a-time\n", now));
    }

    #[cfg(windows)]
    #[test]
    fn the_update_script_claims_the_handoff_and_clears_it_on_abort() {
        let script = update_script(
            42,
            Path::new(r"C:\Temp\V1-Setup-0.1.7.exe"),
            Path::new(r"C:\Apps\V1"),
            Path::new(r"C:\Data\update-in-progress"),
        );
        assert!(script.contains(r#"tasklist /FI "PID eq 42""#));
        assert!(script.contains(r#""C:\Temp\V1-Setup-0.1.7.exe" /S /D=C:\Apps\V1"#));
        assert!(script.contains(r#"if errorlevel 1 goto abort"#));
        assert!(script.contains(r#"start "" "C:\Apps\V1\v1.exe" --updated"#));
        assert!(script.contains(r#"del "C:\Data\update-in-progress" >nul 2>&1"#));
    }
}
