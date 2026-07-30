#!/usr/bin/env bash
# Install a headless-capable Chromium plus its matching chromedriver from the
# builder image's own package repo — a per-architecture-correct build (unlike
# Google's own Chrome-for-Testing distribution, which only ships linux-x64 and
# cannot be used on the arm64/Graviton self-hosted fleet these jobs run on).
#
# Root-only (apt-get/dnf). Shared by:
#   - the WASM (browser client) job (via browser-wasm-test-ci.sh), which runs
#     its whole container step as root; and
#   - the required merge-group `build` job (#1425 r4), which must run this
#     BEFORE it creates and switches to the non-root `ci` user (graviton-
#     build-test.sh itself runs as `ci` and cannot apt-get/dnf install).
set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq chromium chromium-driver
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y -q chromium chromedriver
else
  echo "install-chromium.sh: no supported package manager (apt-get/dnf) found" >&2
  exit 1
fi
