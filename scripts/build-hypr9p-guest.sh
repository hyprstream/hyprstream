#!/usr/bin/env bash
# Build the native in-guest 9P-over-vsock client (hyprstream epic #729, V3 #732).
#
# This is a FOREIGN-toolchain (Go) guest artifact. It is intentionally OUT of
# the Cargo workspace and is NOT built by `cargo build`; build it with this
# script (or the standalone CI job). Produces a static (CGO-disabled) binary the
# operator stages into the kata guest rootfs (or delivers via the tenant VFS).
#
# Usage:
#   scripts/build-hypr9p-guest.sh [output-path]
#
# Env:
#   GOOS, GOARCH  cross-compile targets (default: host; guest is linux/amd64)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg_dir="${repo_root}/workers/hypr9p-guest"
out="${1:-${repo_root}/target/hypr9p-guest}"

if ! command -v go >/dev/null 2>&1; then
  echo "error: go toolchain not found on PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$out")"

echo "building hypr9p-guest -> ${out}"
( cd "${pkg_dir}" && CGO_ENABLED=0 go build -o "${out}" . )

echo "built: ${out}"
file "${out}" 2>/dev/null || true
