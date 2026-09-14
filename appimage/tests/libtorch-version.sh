#!/usr/bin/env bash
# Tiny causal upgrade fixtures: no downloads, Cargo, or real libtorch required.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/appimage/build-appimage.sh"
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT
LIBTORCH_CACHE_DIR="$fixture/cache"
BUILD_DIR="$fixture/build"
OUTPUT_DIR="$fixture/output"
unset HYPRSTREAM_LIBTORCH_DIR
count=0
ok() { count=$((count + 1)); }
reject() { if "$@"; then echo "unexpected success: $*" >&2; exit 1; fi; ok; }
make_lib() {
    local dir="$1" version="$2"
    mkdir -p "$dir/include/torch/csrc/api/include/torch" "$dir/lib"
    printf '#define TORCH_VERSION "%s"\n' "$version" > "$dir/include/torch/csrc/api/include/torch/version.h"
    printf 'fixture\n' > "$dir/lib/libtorch.so"
    printf 'fixture\n' > "$dir/lib/libtorch_cpu.so"
}
make_lib "$fixture/archive/libtorch" 2.11.0
mkdir -p "$fixture/archive/libtorch/include/torch/headeronly"
printf '#define TORCH_VERSION \\\n  "2.11.0"\n' > "$fixture/archive/libtorch/include/torch/headeronly/version.h"
printf '#include <torch/headeronly/version.h>\n' > "$fixture/archive/libtorch/include/torch/csrc/api/include/torch/version.h"
printf '2.11.0+cpu\n' > "$fixture/archive/libtorch/build-version"
make_lib "$LIBTORCH_CACHE_DIR/cpu/libtorch" 2.10.0
mkdir -p "$LIBTORCH_CACHE_DIR"
(cd "$fixture/archive" && zip -qr "$LIBTORCH_CACHE_DIR/libtorch-2.11.0-cpu.zip" libtorch)
curl() { echo 'unexpected download' >&2; return 1; }
download_libtorch cpu
[[ -f "$LIBTORCH_CACHE_DIR/2.11.0-cpu/.complete" ]]
require_libtorch_version "$(libtorch_variant_dir cpu)" 2.11.0
[[ $(cat "$LIBTORCH_CACHE_DIR/cpu/libtorch/include/torch/csrc/api/include/torch/version.h") == *2.10.0* ]]
ok
# A complete matching extraction works without the archive or network.
rm "$LIBTORCH_CACHE_DIR/libtorch-2.11.0-cpu.zip"
download_libtorch cpu
ok
# Header-only wheel version discovery works, including local wheel suffix.
make_lib "$fixture/wheel" 2.11.0+cpu
require_libtorch_version "$fixture/wheel" 2.11.0
ok
HYPRSTREAM_LIBTORCH_DIR="$LIBTORCH_CACHE_DIR/cpu/libtorch"
reject download_libtorch cpu
# The real build and packaging entry points reject before Cargo/copying output.
cargo() { echo called > "$fixture/cargo-called"; return 99; }
reject build_binary cpu
reject create_appimage cpu
[[ ! -e "$fixture/cargo-called" && ! -e "$OUTPUT_DIR" ]]
HYPRSTREAM_LIBTORCH_DIR="$fixture/wheel"
download_libtorch cpu
ok
printf '2.10.0\n' > "$fixture/wheel/build-version"
reject require_libtorch_version "$fixture/wheel" 2.11.0
printf '2.11.0+cpu\n2.10.0\n' > "$fixture/wheel/build-version"
reject require_libtorch_version "$fixture/wheel" 2.11.0
rm "$fixture/wheel/build-version" "$fixture/wheel/lib/libtorch_cpu.so"
reject require_libtorch_version "$fixture/wheel" 2.11.0
unset HYPRSTREAM_LIBTORCH_DIR
rm "$LIBTORCH_CACHE_DIR/2.11.0-cpu/.complete"
reject download_libtorch cpu
touch "$LIBTORCH_CACHE_DIR/2.11.0-cpu/.complete"
# Stage checks the version and carries the real header for universal packaging.
mkdir -p "$BUILD_DIR/bin"
printf binary > "$BUILD_DIR/bin/hyprstream-cpu"
cmd_stage cpu
require_libtorch_version "$BUILD_DIR/universal-staging/2.11.0/lib/cpu/libtorch" 2.11.0
ok
# A corrupted staged library cannot be packaged successfully.
printf '#define TORCH_VERSION "2.10.0"\n' > "$BUILD_DIR/universal-staging/2.11.0/lib/cpu/libtorch/include/torch/headeronly/version.h"
ALL_VARIANTS=(cpu)
reject create_universal_appimage 1
cmd_clean cpu
[[ ! -e "$LIBTORCH_CACHE_DIR/2.11.0-cpu" && -d "$LIBTORCH_CACHE_DIR/cpu/libtorch" ]]
ok
printf 'libtorch packaging fixtures: %s PASS\n' "$count"
