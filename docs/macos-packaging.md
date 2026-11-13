# macOS packaging

`installer/build-macos.sh` creates an Apple Silicon (`aarch64-apple-darwin`)
release app and compressed DMG. It builds FFmpeg 9.0.2 from the official
`https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz` source archive and verifies
SHA-256 `8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e`
before extracting or building it. The script uses Apple's clang with a macOS
13.0 deployment target and FFmpeg's static internal libraries. It disables
external library autodetection and explicitly enables VideoToolbox encoding,
AudioToolbox codecs, and Secure Transport networking. This supplies the native
hardware H.264 encoder used by V1's video background conversion without adding
Homebrew or other external dylib dependencies. Native FFmpeg codecs, demuxers,
muxers, filters, and protocols stay enabled; V1 restricts accepted formats and
protocols when it invokes the tools. FFmpeg is configured for LGPL v3-or-later
with `--enable-version3`.

The app places `ffmpeg` and `ffprobe` beside `v1` in
`V1.app/Contents/MacOS`. This matches Dream Media Handler's executable-folder
discovery. `Contents/Resources` includes the FFmpeg LGPL license, the exact
source archive, and `FFmpeg-Build-Configuration.txt`, which records the source
hash and build configuration. The output DMG is
`target/aarch64-apple-darwin/release/V1-<Cargo version>-arm64.dmg`; the release
workflow uploads this artifact and publishes it with the Windows installer.

Run the script on an Apple Silicon Mac with Xcode command-line tools, Rust's
`aarch64-apple-darwin` target, `make`, `curl`, `tar`, `shasum`, and `hdiutil`.
Packaging fails on missing tools, a source checksum mismatch, missing build
outputs, failed bundled-tool version checks, or failed DMG creation. The app
and bundled FFmpeg link only to macOS system frameworks and libraries; both carry
a macOS 13.0 minimum deployment target.

Validate on the target Mac with `cargo check -p v1-app` and the tests listed in
README or the project workflow. After packaging, run `hdiutil verify` on the DMG,
launch the bundled executable, and inspect its dependencies with `otool -L`.
Check animated backgrounds and transitions on the built-in ProMotion display and
any external display at its native refresh rate. Display-rate reporting alone
does not measure sustained frame delivery.

Release bundles are not notarized. macOS automatic updates are not implemented;
install a newer DMG manually.
