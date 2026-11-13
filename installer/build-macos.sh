#!/usr/bin/env bash
# Copyright (c) 2026 Hunter Lehr (Barusoup)
# SPDX-License-Identifier: AGPL-3.0-only
#
# Builds V1.app and a compressed .dmg for macOS Apple Silicon (arm64).
# Intel Macs are not supported.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="aarch64-apple-darwin"
FFMPEG_VERSION="9.0.2"
FFMPEG_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
FFMPEG_SHA256="8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e"
VERSION="$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/' | tr -d '\r')"
APP_NAME="V1"
BUNDLE_ID="com.barusoup.v1"
DIST_DIR="$ROOT_DIR/target/$TARGET/release"
APP_DIR="$DIST_DIR/$APP_NAME.app"
DMG_PATH="$DIST_DIR/$APP_NAME-$VERSION-arm64.dmg"
FFMPEG_WORK="$ROOT_DIR/target/ffmpeg-macos-$FFMPEG_VERSION"
FFMPEG_SOURCE="$FFMPEG_WORK/ffmpeg-$FFMPEG_VERSION"
FFMPEG_ARCHIVE="$FFMPEG_WORK/ffmpeg-$FFMPEG_VERSION.tar.xz"

for tool in cargo curl shasum tar make clang; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Error: required macOS build tool '$tool' was not found" >&2; exit 1; }
done

mkdir -p "$FFMPEG_WORK"
if [[ ! -f "$FFMPEG_ARCHIVE" ]]; then
    echo "Downloading FFmpeg $FFMPEG_VERSION source from $FFMPEG_URL..."
    curl --fail --location --retry 3 --output "$FFMPEG_ARCHIVE" "$FFMPEG_URL"
fi
actual_sha256="$(shasum -a 256 "$FFMPEG_ARCHIVE" | awk '{print $1}')"
if [[ "$actual_sha256" != "$FFMPEG_SHA256" ]]; then
    echo "Error: FFmpeg source checksum mismatch (expected $FFMPEG_SHA256, got $actual_sha256)" >&2
    rm -f "$FFMPEG_ARCHIVE"
    exit 1
fi
if [[ ! -f "$FFMPEG_SOURCE/configure" ]]; then
    tar -xf "$FFMPEG_ARCHIVE" -C "$FFMPEG_WORK"
fi

echo "Building static-internal FFmpeg tools for macOS 13+..."
cd "$FFMPEG_SOURCE"
MACOS_MIN="-mmacosx-version-min=13.0"
CC="clang" CFLAGS="$MACOS_MIN" LDFLAGS="$MACOS_MIN" ./configure \
    --prefix="$FFMPEG_WORK/install" \
    --arch=arm64 --target-os=darwin \
    --enable-static --disable-shared --disable-autodetect --enable-version3 --enable-videotoolbox --enable-audiotoolbox --enable-securetransport \
    --disable-debug --disable-doc --disable-ffplay --disable-indevs --disable-outdevs
make -j"$(sysctl -n hw.ncpu)"
make install
cd "$ROOT_DIR"

echo "Building $APP_NAME $VERSION for $TARGET..."
MACOSX_DEPLOYMENT_TARGET=13.0 cargo build -p v1-app --release --target "$TARGET"

BIN_PATH="$DIST_DIR/v1"
if [[ ! -f "$BIN_PATH" ]]; then
    echo "Error: expected binary at $BIN_PATH" >&2
    exit 1
fi

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

cp LICENSE NOTICE THIRD-PARTY-LICENSES.md "$APP_DIR/Contents/Resources/"
cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/v1"
chmod +x "$APP_DIR/Contents/MacOS/v1"

cp "$FFMPEG_WORK/install/bin/ffmpeg" "$APP_DIR/Contents/MacOS/ffmpeg"
cp "$FFMPEG_WORK/install/bin/ffprobe" "$APP_DIR/Contents/MacOS/ffprobe"
chmod +x "$APP_DIR/Contents/MacOS/ffmpeg" "$APP_DIR/Contents/MacOS/ffprobe"
"$APP_DIR/Contents/MacOS/ffmpeg" -version | sed -n '1p' | grep -F "ffmpeg version $FFMPEG_VERSION" >/dev/null || {
    echo "Error: bundled FFmpeg failed its version check" >&2; exit 1;
}
"$APP_DIR/Contents/MacOS/ffprobe" -version | sed -n '1p' | grep -F "ffprobe version $FFMPEG_VERSION" >/dev/null || {
    echo "Error: bundled ffprobe failed its version check" >&2; exit 1;
}

# Include the exact source archive and license required for the LGPL build.
cp "$FFMPEG_SOURCE/COPYING.LGPLv3" "$APP_DIR/Contents/Resources/LICENSE-FFmpeg.txt"
cp "$FFMPEG_ARCHIVE" "$APP_DIR/Contents/Resources/FFmpeg-9.0.2-Source.tar.xz"
cat > "$APP_DIR/Contents/Resources/FFmpeg-Build-Configuration.txt" <<EOF
FFmpeg $FFMPEG_VERSION (official source archive: https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz)
SHA-256: $FFMPEG_SHA256
Configuration: --arch=arm64 --target-os=darwin --enable-static --disable-shared --disable-autodetect --enable-version3 --enable-videotoolbox --enable-audiotoolbox --enable-securetransport --disable-debug --disable-doc --disable-ffplay --disable-indevs --disable-outdevs
Compiler: Apple clang; deployment target macOS 13.0
External libraries: none (FFmpeg internal libraries are static; OS libraries/frameworks may be dynamically linked)
EOF

# Generate AppIcon.icns from the bundled 256x256 PNG if iconutil/sips are available.
ICON_SRC="$ROOT_DIR/crates/v1-app/assets/icon/app-256.png"
if [[ -f "$ICON_SRC" ]] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
    ICONSET_DIR="$(mktemp -d)/AppIcon.iconset"
    mkdir -p "$ICONSET_DIR"
    sips -z 16 16     "$ICON_SRC" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null
    sips -z 32 32     "$ICON_SRC" --out "$ICONSET_DIR/icon_16x16@2x.png" >/dev/null
    sips -z 32 32     "$ICON_SRC" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null
    sips -z 64 64     "$ICON_SRC" --out "$ICONSET_DIR/icon_32x32@2x.png" >/dev/null
    sips -z 128 128   "$ICON_SRC" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null
    sips -z 256 256   "$ICON_SRC" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
    sips -z 256 256   "$ICON_SRC" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null
    iconutil -c icns "$ICONSET_DIR" -o "$APP_DIR/Contents/Resources/AppIcon.icns"
    rm -rf "$(dirname "$ICONSET_DIR")"
fi

cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>
    <string>v1</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSArchitecturePriority</key>
    <array>
        <string>arm64</string>
    </array>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "Created $APP_DIR"

command -v hdiutil >/dev/null 2>&1 || { echo "Error: hdiutil is required to create the release DMG" >&2; exit 1; }
rm -f "$DMG_PATH"
hdiutil create -volname "$APP_NAME" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG_PATH" >/dev/null
[[ -s "$DMG_PATH" ]] || { echo "Error: DMG creation did not produce $DMG_PATH" >&2; exit 1; }
echo "Created $DMG_PATH"
