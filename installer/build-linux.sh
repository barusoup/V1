#!/usr/bin/env bash
# Copyright (c) 2026 Hunter Lehr (Barusoup)
# SPDX-License-Identifier: AGPL-3.0-only
#
# Builds V1 as a self-contained x86-64 AppImage, with the pinned static FFmpeg tools inside it.
# The result is target/release/V1-<version>-x86_64.AppImage.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

FFMPEG_VERSION="9.0.2"
FFMPEG_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
FFMPEG_SHA256="8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e"
VERSION="$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/' | tr -d '\r')"
APP_NAME="V1"
DIST_DIR="$ROOT_DIR/target/release"
APPDIR="$ROOT_DIR/target/appimage/V1.AppDir"
OUT="$DIST_DIR/$APP_NAME-$VERSION-x86_64.AppImage"
TOOL_DIR="$ROOT_DIR/target/appimage"
TOOL="$TOOL_DIR/appimagetool-x86_64.AppImage"
TOOL_VERSION="1.9.1"
TOOL_URL="https://github.com/AppImage/appimagetool/releases/download/${TOOL_VERSION}/appimagetool-x86_64.AppImage"
TOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
FFMPEG_WORK="$ROOT_DIR/target/ffmpeg-linux-$FFMPEG_VERSION"
FFMPEG_SOURCE="$FFMPEG_WORK/ffmpeg-$FFMPEG_VERSION"
FFMPEG_ARCHIVE="$FFMPEG_WORK/ffmpeg-$FFMPEG_VERSION.tar.xz"

for tool in cargo curl sha256sum tar make gcc; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Error: required Linux build tool '$tool' was not found" >&2; exit 1; }
done

# --- FFmpeg ----------------------------------------------------------------------------------
# A static-internal build, like the macOS one: no external codec or TLS libraries, so the tools
# do not depend on what the host happens to have installed. V1's proxy serves streams over
# loopback HTTP, and dream-media pins the protocols it may open.
mkdir -p "$FFMPEG_WORK"
if [[ ! -f "$FFMPEG_ARCHIVE" ]]; then
    echo "Downloading FFmpeg $FFMPEG_VERSION source from $FFMPEG_URL..."
    curl --fail --location --retry 3 --output "$FFMPEG_ARCHIVE" "$FFMPEG_URL"
fi
actual_sha256="$(sha256sum "$FFMPEG_ARCHIVE" | awk '{print $1}')"
if [[ "$actual_sha256" != "$FFMPEG_SHA256" ]]; then
    echo "Error: FFmpeg source checksum mismatch (expected $FFMPEG_SHA256, got $actual_sha256)" >&2
    rm -f "$FFMPEG_ARCHIVE"
    exit 1
fi
if [[ ! -f "$FFMPEG_SOURCE/configure" ]]; then
    tar -xf "$FFMPEG_ARCHIVE" -C "$FFMPEG_WORK"
fi

if [[ ! -x "$FFMPEG_WORK/install/bin/ffmpeg" ]]; then
    echo "Building static-internal FFmpeg tools for Linux..."
    (
        cd "$FFMPEG_SOURCE"
        ./configure \
            --prefix="$FFMPEG_WORK/install" \
            --enable-static --disable-shared --disable-autodetect --enable-version3 \
            --disable-debug --disable-doc --disable-ffplay --disable-indevs --disable-outdevs \
            --disable-x86asm
        make -j"$(nproc)"
        make install
    )
fi

# --- AppDir ----------------------------------------------------------------------------------
echo "Building $APP_NAME $VERSION for x86_64 Linux..."
cargo build -p v1-app --release

BIN_PATH="$DIST_DIR/v1"
[[ -f "$BIN_PATH" ]] || { echo "Error: expected binary at $BIN_PATH" >&2; exit 1; }

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" \
    "$APPDIR/usr/share/applications" \
    "$APPDIR/usr/share/icons/hicolor/48x48/apps" \
    "$APPDIR/usr/share/icons/hicolor/256x256/apps" \
    "$APPDIR/usr/share/licenses/V1"

install -m 755 "$BIN_PATH" "$APPDIR/usr/bin/v1"
install -m 755 "$FFMPEG_WORK/install/bin/ffmpeg" "$APPDIR/usr/bin/ffmpeg"
install -m 755 "$FFMPEG_WORK/install/bin/ffprobe" "$APPDIR/usr/bin/ffprobe"
"$APPDIR/usr/bin/ffmpeg" -version | sed -n '1p' | grep -F "ffmpeg version $FFMPEG_VERSION" >/dev/null || {
    echo "Error: bundled FFmpeg failed its version check" >&2; exit 1;
}
"$APPDIR/usr/bin/ffprobe" -version | sed -n '1p' | grep -F "ffprobe version $FFMPEG_VERSION" >/dev/null || {
    echo "Error: bundled ffprobe failed its version check" >&2; exit 1;
}

# The desktop entry and icon sit both at the AppImage root (which is what the AppImage spec and
# launchers read) and in the hicolor theme inside the image.
install -m 644 installer/v1.desktop "$APPDIR/v1.desktop"
install -m 644 installer/v1.desktop "$APPDIR/usr/share/applications/v1.desktop"
install -m 644 crates/v1-app/assets/icon/app-48.png "$APPDIR/usr/share/icons/hicolor/48x48/apps/v1.png"
install -m 644 crates/v1-app/assets/icon/app-256.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/v1.png"
cp crates/v1-app/assets/icon/app-256.png "$APPDIR/v1.png"
ln -sf v1.png "$APPDIR/.DirIcon"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
# AppImage entry point. The app and its FFmpeg tools live in this image, so point the decoder at
# the bundled pair instead of whatever the host happens to have on PATH.
here="${APPDIR:-$(dirname "$(readlink -f "$0")")}"
export DREAM_FFMPEG="$here/usr/bin/ffmpeg"
export DREAM_FFPROBE="$here/usr/bin/ffprobe"
exec "$here/usr/bin/v1" "$@"
EOF
chmod 755 "$APPDIR/AppRun"

cp LICENSE NOTICE THIRD-PARTY-LICENSES.md "$APPDIR/usr/share/licenses/V1/"
cp "$FFMPEG_SOURCE/COPYING.LGPLv3" "$APPDIR/usr/share/licenses/V1/LICENSE-FFmpeg.txt"
cp "$FFMPEG_ARCHIVE" "$APPDIR/usr/share/licenses/V1/FFmpeg-9.0.2-Source.tar.xz"
cat > "$APPDIR/usr/share/licenses/V1/FFmpeg-Build-Configuration.txt" <<EOF
FFmpeg $FFMPEG_VERSION (official source archive: https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz)
SHA-256: $FFMPEG_SHA256
Configuration: --enable-static --disable-shared --disable-autodetect --enable-version3 --disable-debug --disable-doc --disable-ffplay --disable-indevs --disable-outdevs --disable-x86asm
Compiler: system GCC; static build for x86-64 Linux
External libraries: none (FFmpeg internal libraries are static; OS libraries are dynamically linked)
EOF

# --- AppImage --------------------------------------------------------------------------------
tool_ok() {
    [[ -x "$TOOL" ]] && [[ "$(sha256sum "$TOOL" | awk '{print $1}')" == "$TOOL_SHA256" ]]
}
if ! tool_ok; then
    echo "Downloading appimagetool $TOOL_VERSION..."
    mkdir -p "$TOOL_DIR"
    rm -f "$TOOL"
    curl --fail --location --retry 3 --output "$TOOL" "$TOOL_URL"
    actual_sha256="$(sha256sum "$TOOL" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$TOOL_SHA256" ]]; then
        echo "Error: appimagetool checksum mismatch (expected $TOOL_SHA256, got $actual_sha256)" >&2
        rm -f "$TOOL"
        exit 1
    fi
    chmod +x "$TOOL"
fi

echo "Packing $OUT..."
rm -f "$OUT"
# --appimage-extract-and-run keeps the tool working on machines (and CI runners) without FUSE.
ARCH=x86_64 "$TOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "$OUT"

echo "Created $OUT"
