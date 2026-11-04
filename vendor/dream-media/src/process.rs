//! Bounded child processes and live decoder streams.

use crate::error::{Error, Result};
use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Diagnostics kept from a decoder. The pipe is always drained; only this
/// much is retained for an error message.
pub(crate) const STDERR_CAP: usize = 16 * 1024;

/// A command that never flashes a console window on Windows.
pub(crate) fn command(program: &Path) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// What a bounded run produced.
pub(crate) struct Run {
    pub status: std::process::ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub timed_out: bool,
}

/// Run `cmd` with a deadline, draining both pipes so the child cannot block.
pub(crate) fn run(mut cmd: Command, deadline: Duration, stdout_cap: usize) -> std::io::Result<Run> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().map(|pipe| drain(pipe, stdout_cap));
    let stderr = child.stderr.take().map(|pipe| drain(pipe, STDERR_CAP));
    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= deadline {
            timed_out = true;
            let _ = child.kill();
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    Ok(Run {
        status,
        stdout: join(stdout),
        stderr: join(stderr),
        timed_out,
    })
}

fn join(handle: Option<std::thread::JoinHandle<Vec<u8>>>) -> Vec<u8> {
    handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default()
}

fn drain(mut pipe: impl Read + Send + 'static, cap: usize) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        // A first guess at the payload size: a decoded still is megabytes, and
        // growing from zero costs half a dozen copies to get there.
        let mut kept = Vec::with_capacity(cap.min(1024 * 1024));
        let mut buffer = [0u8; 16 * 1024];
        loop {
            match pipe.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    let room = cap.saturating_sub(kept.len());
                    kept.extend_from_slice(&buffer[..read.min(room)]);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        kept
    })
}

/// First non-empty stderr line, shortened for display.
pub(crate) fn stderr_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("the decoder reported no reason")
        .chars()
        .take(300)
        .collect()
}

/// Turn a failed process into the error the caller sees.
pub(crate) fn failure(what: &str, source: &Path, run: &Run) -> Error {
    if run.timed_out {
        return Error::msg(format!("{what} timed out for {}", source.display()));
    }
    Error::msg(format!(
        "{what} failed for {}: {}",
        source.display(),
        stderr_line(&run.stderr)
    ))
}

/// A long-lived decoder process with a live stdout. Dropping it kills and
/// reaps the child, so a seek or a closed player never leaves a decoder
/// running behind it.
pub(crate) struct Stream {
    child: Child,
    stdout: Option<ChildStdout>,
    stderr: Arc<Mutex<Vec<u8>>>,
}

impl Stream {
    pub(crate) fn spawn(mut cmd: Command) -> Result<Self> {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn()?;
        let stdout = child.stdout.take();
        let stderr = Arc::new(Mutex::new(Vec::new()));
        if let Some(pipe) = child.stderr.take() {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || {
                let mut pipe = pipe;
                let mut buffer = [0u8; 4096];
                loop {
                    match pipe.read(&mut buffer) {
                        Ok(0) => break,
                        Ok(read) => {
                            let mut kept = sink.lock().unwrap_or_else(|poison| poison.into_inner());
                            let room = STDERR_CAP.saturating_sub(kept.len());
                            kept.extend_from_slice(&buffer[..read.min(room)]);
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
            });
        }
        Ok(Self {
            child,
            stdout,
            stderr,
        })
    }

    /// The decoder's output pipe. Taken rather than borrowed so a reader
    /// thread can own it.
    pub(crate) fn stdout(&mut self) -> Option<ChildStdout> {
        self.stdout.take()
    }

    pub(crate) fn stderr_text(&self) -> String {
        let kept = self
            .stderr
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        String::from_utf8_lossy(&kept).trim().to_string()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
