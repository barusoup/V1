# Dream Media Handler

Media player for [Dream](https://github.com/barusoup/Dream) apps, built on FFmpeg.

## Usage

Still image:

```rust
use dream_media::Image;

let image = Image::open("poster.jpg", 1920)?; // longest side capped at 1920; 0 = source size
let texture = image.upload(&ctx);
ui.image(rect, &texture, Color::WHITE);
```

Video:

```rust
use dream_media::{PlayerOptions, VideoPlayer};

let mut video = VideoPlayer::open(&ctx, "clip.webm", PlayerOptions {
    max_size: (1280, 720),
    audio: true,
    looping: true,
    ..Default::default()
})?;

// once per rendered frame:
if let Some(texture) = video.poll(Instant::now()) {
    ui.image(rect, &texture, Color::WHITE);
}
```

Audio only:

```rust
use dream_media::{Player, PlayerOptions};

let mut player = Player::open("track.flac", PlayerOptions {
    max_size: (0, 0),
    audio: true,
    ..Default::default()
})?;
player.set_volume(0.8);
let _ = player.poll(Instant::now()); // keep polling to drive the clock and end-of-file
```

`poll` is non-blocking. It returns a texture only when a new frame was uploaded.

## Installation

```toml
[dependencies]
dream-media = { git = "https://github.com/barusoup/dream-media.git", tag = "v0.1.0" }
```

The `gpu` feature is on by default and pulls in Dream. For a headless build with
no Dream or Vulkan dependency, use `default-features = false`. `Image`, `Player`,
`probe` and `Tools` still work in that configuration.

## Requirements

- Rust 1.85 or newer (edition 2024).
- `ffmpeg` and `ffprobe`, found in this order:
  1. `DREAM_FFMPEG` and `DREAM_FFPROBE` (file paths)
  2. `DREAM_FFMPEG_DIR` (a folder containing both)
  3. an `ffmpeg/` folder next to the executable, then the executable's folder
  4. `PATH`

  To skip discovery, pass paths to `Tools::from_paths`.
- The `gpu` feature needs a Vulkan 1.3 context.

## Tests

```sh
cargo test                       # with gpu
cargo test --no-default-features # headless
```

Tests generate their media with the system `ffmpeg` and skip if it is not installed.

## License

AGPL-3.0 with additional attribution terms, matching Dream. See [LICENSE](LICENSE).

Open-source projects can use this freely. Credit "Dream Media Handler",
"Dream Engine" and Hunter Lehr in your documentation or about screen, with links
to both repositories. Closed-source use requires a commercial license from the
copyright holder: 1hunterlehr@gmail.com.
