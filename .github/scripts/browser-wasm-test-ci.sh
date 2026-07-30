#!/usr/bin/env bash
# CI wrapper for browser-wasm-test.sh.
#
# Installs a headless-capable Chromium plus its matching chromedriver via
# whichever system package manager the rust-builder image provides (a distro
# package repo resolves the correct binary for the runner's own architecture —
# unlike Google's own Chrome-for-Testing distribution, which only ships
# linux-x64 and cannot be used on the arm64/Graviton self-hosted runner this
# job runs on), points the wasm-bindgen-test-runner at that exact binary via a
# generated `webdriver.json`, then delegates to browser-wasm-test.sh for the
# actual isolated build + real headless-browser execution.
set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq chromium chromium-driver
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y -q chromium chromedriver
else
  echo "browser-wasm-test-ci.sh: no supported package manager (apt-get/dnf) found for chromium install" >&2
  exit 1
fi

CHROME_BIN="$(command -v chromium || command -v chromium-browser || true)"
if [[ -z "$CHROME_BIN" ]]; then
  echo "browser-wasm-test-ci.sh: chromium/chromium-browser not found on PATH after install" >&2
  exit 1
fi

WEBDRIVER_JSON="$(mktemp)"
cat > "$WEBDRIVER_JSON" <<EOF
{"goog:chromeOptions": {"binary": "$CHROME_BIN"}}
EOF

export WASM_BINDGEN_TEST_WEBDRIVER_JSON="$WEBDRIVER_JSON"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/browser-wasm-test.sh"
