// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Background updates. A worker thread checks GitHub shortly after launch and every few minutes
//! after that. When a release is newer, it downloads and verifies the Windows setup immediately, so
//! the sidebar can offer an Update entry that only has to hand off to the installer when clicked.
//!
//! Only installed copies update: the Windows installer leaves `uninstall.exe` beside the app, so a
//! build run from a target folder sits this out. Set `V1_NO_UPDATE` to skip checks and installs.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

const CURRENT: &str = env!("CARGO_PKG_VERSION");
const LATEST_URL: &str = "https://api.github.com/repos/barusoup/V1/releases/latest";
const DOWNLOAD_PREFIX: &str = "https://github.com/barusoup/V1/releases/download/";
const CHECKSUMS: &str = "SHA256SUMS";
const ATTEMPTS: u32 = 3;
const RETRY_PAUSE: Duration = Duration::from_millis(300);
/// The release lookup, retries included, may take this long before the worker tries again later.
const CHECK_BUDGET: Duration = Duration::from_secs(6);
/// A setup is about 90 MB. The download runs beside the app and is finished before the Update entry
/// appears, so it can afford a slow connection; the click itself never waits for the network.
const DOWNLOAD_BUDGET: Duration = Duration::from_secs(300);
/// Let the first frame settle before the worker starts a runtime and opens a TLS connection.
const START_DELAY: Duration = Duration::from_millis(500);
/// How often a quiet session looks again. The lookup is one small request.
const CHECK_EVERY: Duration = Duration::from_secs(15 * 60);
/// A failed lookup or download waits this long before trying again.
const RETRY_EVERY: Duration = Duration::from_secs(5 * 60);
/// Set by the installer, which starts the updated copy after replacing the old one.
pub(crate) const UPDATED_FLAG: &str = "--updated";
/// Marker covering the gap between the old process exiting and the installer starting the new one.
const HANDOFF_FILE: &str = "update-in-progress";
/// A marker older than this is treated as a crashed update, so a broken install cannot lock the app
/// out for long. The helper script waits up to two minutes before it gives up.
const HANDOFF_TTL: Duration = Duration::from_secs(300);

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

/// A verified setup that is already on disk, ready to install.
#[derive(Debug, Clone)]
pub struct Available {
    pub version: Version,
    package: PathBuf,
}

/// What the background worker reports back to the UI thread.
#[derive(Debug)]
pub enum UpdateEvent {
    /// A newer release has been downloaded and checked, and the Update entry can appear.
    Ready(Available),
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

/// Start the update worker. It checks and prefetches on its own; the UI only hears about a setup
/// that is already downloaded and verified.
pub fn start_worker<F>(on_event: F)
where
    F: Fn(UpdateEvent) + Send + 'static,
{
    if std::env::var_os("V1_NO_UPDATE").is_some() || install_target().is_none() {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("v1-update".to_owned())
        .spawn(move || {
            std::thread::sleep(START_DELAY);
            let mut announced: Option<Available> = None;
            loop {
                let mut wait = CHECK_EVERY;
                match prepare_update(announced.as_ref()) {
                    Ok(Some(update)) => {
                        let changed = announced
                            .as_ref()
                            .is_none_or(|last| last.version != update.version);
                        if changed {
                            on_event(UpdateEvent::Ready(update.clone()));
                        }
                        announced = Some(update);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        eprintln!("V1 update check: {error}");
                        wait = RETRY_EVERY;
                    }
                }
                std::thread::sleep(wait);
            }
        });
}

/// Look for a newer release and leave a verified setup ready on disk. If `known` is already the
/// latest and its file is still present, the download is not touched again.
fn prepare_update(known: Option<&Available>) -> Result<Option<Available>, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_| "The update checker could not start.".to_owned())?;
    let client = reqwest::Client::builder()
        .user_agent(format!("V1/{CURRENT}"))
        .connect_timeout(Duration::from_secs(3))
        .tcp_nodelay(true)
        .build()
        .map_err(|_| "The update checker could not start.".to_owned())?;
    runtime.block_on(async {
        let release = tokio::time::timeout(CHECK_BUDGET, newer_release(&client))
            .await
            .map_err(|_| "The update check timed out.".to_owned())??;
        let Some(release) = release else {
            return Ok(None);
        };
        if let Some(last) = known {
            if last.version == release.version && last.package.is_file() {
                return Ok(None);
            }
        }
        let package = stage_package(&client, &release).await?;
        Ok(Some(Available {
            version: release.version,
            package,
        }))
    })
}

/// Hand the verified setup to the installer and exit. This is the whole click path: no network, no
/// download, just the marker, the detached script and a process exit.
pub fn install(update: &Available) -> Result<(), String> {
    if std::env::var_os("V1_NO_UPDATE").is_some() {
        return Err("Updates are turned off for this launch.".to_owned());
    }
    let target = install_target()
        .ok_or_else(|| "Updates are only available in an installed copy of V1.".to_owned())?;
    if !update.package.is_file() {
        return Err("The downloaded update is missing. It will be fetched again.".to_owned());
    }
    if begin_handoff(update.version).is_err() {
        return Err("The update could not be prepared.".to_owned());
    }
    if let Err(error) = restart_into_update(&update.package, &target) {
        clear_handoff();
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

/// Download the package for `release`, verify it against the published checksum, and leave it in
/// the app data folder. A package left by an earlier attempt is reused after the same check.
async fn stage_package(client: &reqwest::Client, release: &Release) -> Result<PathBuf, String> {
    let checksums = get(client, &release.checksums.url, Duration::from_secs(10)).await?;
    let checksums = std::str::from_utf8(&checksums)
        .map_err(|_| "The checksum file could not be read.".to_owned())?;
    let expected = expected_hash(checksums, &release.package.name)
        .ok_or_else(|| "The release has no checksum for its package.".to_owned())?;

    let dir = crate::data_dir().join("updates");
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|_| "The update folder could not be created.".to_owned())?;
    let path = dir.join(&release.package.name);
    if cached_matches(&path, &expected).await {
        prune_old_packages(&dir, &release.package.name);
        return Ok(path);
    }

    let part = dir.join(format!("{}.part", release.package.name));
    let mut response = client
        .get(&release.package.url)
        .header(reqwest::header::ACCEPT, "application/octet-stream")
        .timeout(DOWNLOAD_BUDGET)
        .send()
        .await
        .map_err(download_error)?;
    if !response.status().is_success() {
        return Err(format!(
            "The update download returned {}.",
            response.status()
        ));
    }
    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|_| "The update could not be saved.".to_owned())?;
    let mut hasher = Sha256::new();
    while let Some(chunk) = response.chunk().await.map_err(download_error)? {
        hasher.update(&chunk);
        file.write_all(&chunk)
            .await
            .map_err(|_| "The update could not be saved.".to_owned())?;
    }
    file.flush()
        .await
        .map_err(|_| "The update could not be saved.".to_owned())?;
    drop(file);

    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        let _ = tokio::fs::remove_file(&part).await;
        return Err("The update download did not match its checksum.".to_owned());
    }
    // Windows rename does not replace an existing file; a stale, failed-verification copy may be
    // sitting at the final name.
    let _ = tokio::fs::remove_file(&path).await;
    if tokio::fs::rename(&part, &path).await.is_err() {
        let _ = tokio::fs::remove_file(&part).await;
        return Err("The update could not be finished.".to_owned());
    }
    prune_old_packages(&dir, &release.package.name);
    Ok(path)
}

async fn cached_matches(path: &Path, expected: &str) -> bool {
    let path = path.to_path_buf();
    let expected = expected.to_owned();
    tokio::task::spawn_blocking(move || hash_file(&path).is_some_and(|actual| actual == expected))
        .await
        .unwrap_or(false)
}

fn hash_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

/// Keep only the setup that is ready now; `.part` files from interrupted downloads go too.
fn prune_old_packages(dir: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name != keep && (name.starts_with("V1-Setup-") || name.ends_with(".part")) {
            let _ = std::fs::remove_file(path);
        }
    }
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
                .map_err(download_error)?;
            let response = response
                .error_for_status()
                .map_err(|error| match error.status() {
                    Some(status) => format!("GitHub returned {status}."),
                    None => "GitHub could not be reached.".to_owned(),
                })?;
            response
                .bytes()
                .await
                .map(|body| body.to_vec())
                .map_err(|_| "The update download was interrupted.".to_owned())
        }
        .await;
        match result {
            Ok(body) => return Ok(body),
            Err(error) => last = error,
        }
    }
    Err(last)
}

fn download_error(error: reqwest::Error) -> String {
    if error.is_timeout() {
        "The update download timed out.".to_owned()
    } else {
        "The update could not be downloaded.".to_owned()
    }
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
