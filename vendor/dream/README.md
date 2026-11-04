<p align="center">
  <img src="assets/branding/dream_icon.png" width="160" alt="Dream Engine logo">
  <h1>Dream Engine</h1>
</p>

A Rust game engine that doubles as an app framework. It includes BedFrame, a UI framework for apps and HUDs, and aims to be a working alternative to React-based app frameworks and to engines like Godot and Unity.

**Pre-alpha.** Work in progress. Many features are missing, and the code changes rapidly.

## Requirements

- Rust (edition 2024)
- GPU with Vulkan 1.3

## Run

```
cargo test --all-targets
```

Runtime media (FFmpeg stills, video and audio) lives in the separate [`dream-media`](https://github.com/barusoup/dream-media) crate; Dream does not depend on it.

## License

AGPL-3.0 with additional attribution and commercial-license terms. See [`LICENSE`](LICENSE). Third-party assets: [`NOTICE`](NOTICE).
