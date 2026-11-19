# Why this copy exists

This is [`barusoup/dream`](https://github.com/barusoup/dream) at `main`
(`37ab170`, 2026-11-19), vendored so V1 builds against a known engine and keeps a few
app-specific changes that upstream does not carry.

`37ab170` — "Cut RAM, VRAM and CPU cost and fix Linux/Wayland behaviour" — is the release
this copy tracks. Relative to the previous vendored snapshot (`d8b450a`, 2026-11-17) it:

- Cuts resident memory (Linux page-size aware idle trim that actually returns pages,
  lazy atlas page 0, reused draw lists, small-block GPU allocator), VRAM and idle CPU.
- Fixes Linux/Wayland presentation: the acquire-ahead worker no longer shuts down and
  discards an acquired swapchain image, which starved the 4-image Wayland FIFO
  swapchain and left the window mapped but never presented.
- Adds HiDPI scissor scaling, a 60 Hz display-rate fallback, throttled window-state
  polling, and single-resize-per-scale-change, plus `wl-copy`/`xclip`/`xsel` clipboard
  support on Linux and a native Metal backend on macOS.

## Local changes on top of upstream

Re-applied after the merge because V1 depends on them:

- `GpuContext::supports_bc()` plus the `texture_compression_bc` device feature, and
  `GpuTexture::from_bc7` / `blank_bc7` in `src/texture/gpu.rs` — the custom background
  is uploaded as BC7 (4:1 VRAM) when the GPU supports it. The Metal backend has the
  matching `from_bc7` / `blank_bc7` and reports BC support.
- `DrawList::backdrop_rect` and `DrawList::concave_corner` in `src/bedframe/draw.rs`,
  with the concave mode kept in the UA `finish()` scale pass, for the frosted-glass
  chrome and the sidebar's inward fillets.
- `Ui::label_wrapped_scaled` in `src/bedframe/widgets/controls.rs` — scales an already
  laid-out text run without re-rasterizing, used for the animated lyric transitions.
- The macOS clipboard implementation in `src/bedframe/clipboard.rs` uses the `objc`
  crate (hence the macOS-only `objc` dependency in `Cargo.toml`); V1's search and
  paste path is built on it. It falls back to Dream's in-process buffer.
- `GpuContext::new_headless()` and the offscreen Metal tests in `src/gfx/metal.rs`,
  so the Metal UI/scene path can be checked without a window or event loop.

## 0.2.5 merge notes

`37ab170` gates `gfx::vulkan` out on macOS. This copy previously compiled the Vulkan
texture module on macOS as a compatibility shim (`texture.rs`'s `gpu_vk` module and the
unconditional `create_sampler` export). That shim is not used by V1 and no longer
builds against the gated module, so it was dropped: `src/texture.rs` and
`src/texture/sampler.rs` now match upstream's `cfg` gating. The Metal renderer also
gained `Renderer::new_with_options` and reports `TextureObjectStats` through
`super::TextureObjectStats`, both as in upstream, so `cargo check --target
aarch64-apple-darwin -p dream --all-targets` is clean again.

When upstream ships these (or an equivalent), drop the local copy and point
`crates/v1-app/Cargo.toml` back at the git dependency.

## Keeping this copy current

The base is a plain upstream commit, so a future update is a three-way merge:

```
git clone https://github.com/barusoup/dream upstream
cd upstream
git checkout -B v1 <old-base>
rsync -a --delete --exclude=.git --exclude=target /path/to/V1/vendor/dream/ ./
git add -A && git commit -m "V1 patches on <old-base>"
git merge <new-base>
```

Resolve conflicts toward upstream except for the local changes listed above.
