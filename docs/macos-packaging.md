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
hash and build configuration.

## Signing

The script signs the finished bundle with `codesign` after every file is in
place, because macOS only opens a bundle whose signature covers its contents.
It signs `ffmpeg` and `ffprobe` first and then the bundle, so the seal covers
`Info.plist`, the icon, both helper tools, and the FFmpeg license and source
archive. It clears stray extended attributes before signing and finishes with
`codesign --verify --deep --strict`, failing the build if the seal does not
validate.

A bundle that is never signed carries only the signature the linker put on the
executable, which does not describe a bundle at all. Gatekeeper reads that as a
broken bundle and reports "V1 is damaged and can't be opened", with no option to
open it anyway. That is why the v0.2.1 DMG could verify while the app inside it
could not launch; `hdiutil verify` checks the image, not the app.

The default is an ad-hoc signature, which every Mac accepts once the user allows
the app. Two environment variables upgrade the result:

- `V1_CODESIGN_IDENTITY`: a `Developer ID Application: ...` identity from the
  login keychain. The script then also passes `--options runtime --timestamp`,
  which notarization requires.
- `V1_NOTARY_PROFILE`: a keychain profile created with `xcrun notarytool
  store-credentials`. The script submits the DMG, waits for the result, and
  staples the ticket, so a download opens with no prompt at all. Notarization
  needs a Developer ID identity as well.

Without those variables the release is signed but not notarized, so the first
launch asks for approval once: System Settings > Privacy & Security > "Open
Anyway", or right-click > Open. The DMG includes a note that says so.

The output DMG is `target/aarch64-apple-darwin/release/V1-<Cargo
version>-arm64.dmg`. Its volume holds `V1.app`, an `Applications` link, and that
note; the release workflow uploads the image and publishes it with the Windows
installer.

Run the script on an Apple Silicon Mac with Xcode command-line tools, Rust's
`aarch64-apple-darwin` target, `make`, `curl`, `tar`, `shasum`, `codesign`, and
`hdiutil`. Packaging fails on missing tools, a source checksum mismatch, missing
build outputs, failed bundled-tool version checks, a signature that does not
verify, failed DMG creation, or a copy of the app inside the DMG that does not
verify. The app and bundled FFmpeg link only to macOS system frameworks and
libraries; both carry a macOS 13.0 minimum deployment target.

## Validation

Validate on the target Mac with `cargo check -p v1-app` and the tests listed in
README or the project workflow. After packaging, the script itself runs
`codesign --verify --deep --strict` on the bundle, `hdiutil verify` on the
image, and the same signature check on the mounted copy inside the image.
`spctl -a -vvv` still rejects an ad-hoc or unnotarized build; that is what the
first-launch prompt is for, and it is not a packaging failure.

Launch the bundled executable from Applications, remove the download quarantine
with `xattr -dr com.apple.quarantine /Applications/V1.app` if the copy keeps it,
and inspect its dependencies with `otool -L`. Check animated backgrounds and
transitions on the built-in ProMotion display and any external display at its
native refresh rate. Display-rate reporting alone does not measure sustained
frame delivery.

Release bundles are signed ad-hoc unless a Developer ID is supplied, and are not
notarized, so the first launch needs one approval.

Automatic updates work as they do on Windows: a launch asks GitHub for the newest
full release and, when it is newer, downloads `V1-<version>-arm64.dmg`, checks it
against `SHA256SUMS`, and hands it to a detached `sh` script. The script waits for
the app to exit, mounts the image, verifies the bundled `.app` with
`codesign --verify --deep --strict`, copies it beside the installed bundle with
`ditto`, swaps the two, unmounts, and reopens the app with `--updated`. Nothing
touches the installed bundle until the new one is verified and copied, so a failed
download or a broken signature leaves the old app runnable.

The updater only acts on an installed copy: the executable must sit inside a `.app`
bundle outside `/Volumes`, so an app opened from the disk image or built into a
target folder never writes to itself. The bundle's folder must be writable by the
user (`/Applications` is, for an admin account; `/Applications` installed with
sudo is not, and those installs update by hand).
