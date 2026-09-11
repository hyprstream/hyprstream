#!/usr/bin/env bash
# CI wrapper for browser-wasm-test.sh.
#
# Resolves a headless-capable Chromium (installing it via install-chromium.sh
# if it is not already on PATH — root-only; skipped when a caller already
# installed it, e.g. the required merge-group `build` job installs it as root
# before dropping to a non-root user and running this script from there),
# points the wasm-bindgen-test-runner at that exact binary via a generated
# `webdriver.json`, then delegates to browser-wasm-test.sh for the actual
# isolated build + real headless-browser execution.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHROME_BIN="$(command -v chromium || command -v chromium-browser || true)"
if [[ -z "$CHROME_BIN" ]]; then
  bash "$SCRIPT_DIR/install-chromium.sh"
  CHROME_BIN="$(command -v chromium || command -v chromium-browser || true)"
fi
if [[ -z "$CHROME_BIN" ]]; then
  echo "browser-wasm-test-ci.sh: chromium/chromium-browser not found on PATH after install" >&2
  exit 1
fi

WEBDRIVER_JSON="$(mktemp)"
cat > "$WEBDRIVER_JSON" <<EOF
{"goog:chromeOptions": {"binary": "$CHROME_BIN"}}
EOF

export WASM_BINDGEN_TEST_WEBDRIVER_JSON="$WEBDRIVER_JSON"
exec bash "$SCRIPT_DIR/browser-wasm-test.sh"
