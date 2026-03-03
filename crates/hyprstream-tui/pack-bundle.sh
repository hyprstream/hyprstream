#!/bin/bash
# Pack the Wanix bundle for hyprstream-tui wizard
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p dist

# Copy the WASI binary into the bundle
if [ -f "../../target/wasm32-wasip1/release/hyprstream-tui.wasm" ]; then
  cp ../../target/wasm32-wasip1/release/hyprstream-tui.wasm bundle/
  echo "Copied release build"
elif [ -f "../../target/wasm32-wasip1/debug/hyprstream-tui.wasm" ]; then
  cp ../../target/wasm32-wasip1/debug/hyprstream-tui.wasm bundle/
  echo "Copied debug build"
else
  echo "ERROR: No hyprstream-tui.wasm found. Run cargo build -p hyprstream-tui --target wasm32-wasip1 first."
  exit 1
fi

# Create the tarball
tar czf dist/hyprstream-tui.tgz -C bundle .
echo "Created dist/hyprstream-tui.tgz"

# Copy to web-minerva-game public/wasm/ if it exists
WEB_DIR="$SCRIPT_DIR/../../../web-minerva-game/public/wasm"
if [ -d "$WEB_DIR" ]; then
  cp dist/hyprstream-tui.tgz "$WEB_DIR/"
  cp bundle/hyprstream-tui.wasm "$WEB_DIR/"
  echo "Copied to $WEB_DIR/ (tgz + wasm)"
else
  echo "Note: $WEB_DIR not found, skipping copy"
fi

echo "Bundle ready."
