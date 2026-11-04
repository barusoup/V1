# Changelog

## 0.1.0

First release.

- `Image`: still images to RGBA, with in-process PNG and JPEG decode and an
  FFmpeg fallback for every other format; `Image::upload` for the GPU.
- `Player`: video with optional audio, or audio alone, on an audio-clocked
  timeline with a preallocated, recycled frame pool and zero steady-state
  allocations.
- `VideoPlayer` / `VideoSurface`: asynchronous two-texture upload into Dream's
  graphics pipeline through `dream::texture::TextureUploader`.
- `probe` / `MediaInfo`: flat ffprobe metadata, including rotation, sample
  aspect ratio and duration.
- `Tools`: FFmpeg discovery via environment, app directory or `PATH`.
- Seeking with keyframe coalescing and an automatic exact-seek fallback for
  long-GOP files; looping; pause/resume; volume and mute.
- Feature `gpu` (default) for the Dream bridge; `--no-default-features` removes
  the Dream and Vulkan dependency for headless decoding.
