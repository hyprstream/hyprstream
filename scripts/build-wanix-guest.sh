#!/usr/bin/env bash
# Build the native Wanix guest (hyprstream #506, deliverable 2).
#
# This is a FOREIGN-toolchain (Go) guest artifact. It is intentionally OUT of
# the Cargo workspace and is NOT built by `cargo build`; build it with this
# script (or the standalone CI job). Produces a static (CGO-disabled) binary.
#
# Usage:
#   scripts/build-wanix-guest.sh [output-path]
#
# Env:
#   GOOS, GOARCH  cross-compile targets (default: host)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg_dir="${repo_root}/workers/wanix-guest"
out="${1:-${repo_root}/target/wanix-guest}"

if ! command -v go >/dev/null 2>&1; then
  echo "error: go toolchain not found on PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$out")"

echo "building wanix-guest -> ${out}"
( cd "${pkg_dir}" && CGO_ENABLED=0 go build -o "${out}" . )

echo "built: ${out}"
file "${out}" 2>/dev/null || true
