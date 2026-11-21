# Why this copy exists

This is [`barusoup/dream`](https://github.com/barusoup/dream) at `main`
(`799f744`, 2026-11-21), vendored so V1 builds against a known engine and keeps a few
app-specific changes that upstream does not carry.

`799f744` is the Windows-only release this copy tracks. Relative to the previous vendored
snapshot (`37ab170`, 2026-11-19) it:

- Removes Linux, macOS and Metal support from `main`. Windows is the current target OS;
  macOS + Metal support is preserved on upstream's `macos-metal` branch, and Linux is not
  supported.
- Drops the macOS CoreText font backend, the Metal renderer, the Metal BC7 texture upload
  path and the Linux clipboard / memory-trim / application-id code, along with their
  platform dependencies.
- Keeps V1's Windows surface unchanged.

## Local changes on top of upstream

Re-applied after the merge because V1 depends on them:

- `GpuContext::supports_bc()` plus the `texture_compression_bc` device feature, and
  `GpuTexture::from_bc7` / `blank_bc7` in `src/texture/gpu.rs` — the custom background
  is uploaded as BC7 (4:1 VRAM) when the GPU supports it.
- `DrawList::backdrop_rect` and `DrawList::concave_corner` in `src/bedframe/draw.rs`,
  with the concave mode kept in the UA `finish()` scale pass, for the frosted-glass
  chrome and the sidebar's inward fillets.
- `Ui::label_wrapped_scaled` in `src/bedframe/widgets/controls.rs` — scales an already
  laid-out text run without re-rasterizing, used for the animated lyric transitions.

The macOS-only clipboard and headless Metal changes from the previous snapshot were
dropped with the platform itself.

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
