# Why this copy exists

This is [`barusoup/dream-media`](https://github.com/barusoup/dream-media) at `main` (2026-11-03),
vendored so v1 can play a stream instead of a file.

Upstream deliberately locks FFmpeg down: every probe and decode runs with

```
-protocol_whitelist file
-format_whitelist <a long allowlist of containers>
```

which is the right default for a library that opens local media — and it makes an
`http://127.0.0.1:<port>/…` source fail with `Protocol 'http' not on whitelist 'file'!`.

## The change

In `src/tools.rs`:

- `PROTOCOLS = "file,http,https,tcp,tls"`, used by both `ffprobe_flat` and `input_options`.
- `input_options` also passes FFmpeg's reconnect flags (`-reconnect 1 -reconnect_streamed 1
  -reconnect_on_network_error 1 -reconnect_delay_max 2`), so one dropped connection does not end
  playback at a chunk boundary.

The format allowlist is untouched: `webm`, `matroska`, `ogg`, `opus` and `mov` already cover every
audio format YouTube serves.

## Also added: `Player::scope`

`src/audio.rs` keeps the last 4096 output samples, mixed to mono, in a fixed ring (`Scope`) that the
device callback fills with `try_lock`, so it never waits on the UI. `Player::scope(&mut [f32])`
copies them out with the sample rate. v1's lyric-page spectrum runs its FFT on these.

## What upstream could do instead

A `Player::open_url` (or a `source: Source::{Path, Url}` argument) that widens the protocol
allowlist only for URLs would keep the local-file default strict. When that lands, delete this
directory and point `crates/v1-app/Cargo.toml` back at the git dependency.

## Also added: `Player::open_audio_unprobed`

`Player::open` runs `ffprobe` on the source and then opens it again to decode. For a network source
that is a process launch and a second trip to the server before any sound. A caller that has just
resolved the stream itself already knows it is a single audio stream, so
`Player::open_audio_unprobed(source, options, duration_ms)` skips the probe and takes the little
that `MediaInfo` would have said from the caller. v1 uses it for streams from its loopback proxy
(about 115 ms less to first sound); saved files are still opened with `Player::open`.

Without a probe a dead source no longer fails at open, so `check_end` now reports an audio-only
session that drained without one frame ever played as a failure ("The audio stream gave no sound",
with the decoder's own message when it has one) instead of as a track that ended at once.
