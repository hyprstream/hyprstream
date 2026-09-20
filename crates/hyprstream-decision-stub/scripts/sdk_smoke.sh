#!/usr/bin/env bash
# P0.7 SDK portability smoke test: install the STOCK TypeSafe SDKs (MIT) into a
# throwaway venv / node_modules, boot the stub facade, and point each SDK at it via
# TYPESAFE_BASE_URL. Nothing is vendored; the SDKs come from PyPI/npm unmodified
# (S6a §4: stock-SDK repointing is the portability proof — the adapter never speaks
# the /v1/systemone wire).
#
# Usage: scripts/sdk_smoke.sh [path-to-stub-binary]
# Requires: python3 + pip, node + npm, network access to PyPI/npm.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUB_BIN="${1:-hyprstream-decision-stub}"
WORK="$(mktemp -d /tmp/jev-stub-smoke.XXXXXX)"
trap 'kill "${STUB_PID:-}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

# --- boot the stub facade on an ephemeral port -------------------------------
"$STUB_BIN" --listen 127.0.0.1:0 --flight-listen 127.0.0.1:0 > "$WORK/stub.log" 2>&1 &
STUB_PID=$!
for _ in $(seq 1 50); do
  grep -q "facade http" "$WORK/stub.log" && break
  sleep 0.2
done
FACADE_ADDR="$(sed -n 's/^facade http:\/\///p' "$WORK/stub.log")"
[ -n "$FACADE_ADDR" ] || { echo "stub did not start"; cat "$WORK/stub.log"; exit 1; }

export TYPESAFE_BASE_URL="http://$FACADE_ADDR"
export TYPESAFE_API_KEY="p07-smoke"   # any non-empty token is accepted

# --- Python SDK (PyPI typesafe-sdk) ------------------------------------------
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet typesafe-sdk
"$WORK/venv/bin/python" "$HERE/sdk_smoke.py"

# --- JS SDK (npm @typesafe-ai/sdk) -------------------------------------------
mkdir -p "$WORK/js"
cd "$WORK/js"
npm init -y >/dev/null
npm install --silent @typesafe-ai/sdk
cp "$HERE/sdk_smoke.mjs" .
node sdk_smoke.mjs

echo "SDK SMOKE: both stock SDKs passed against the stub at $TYPESAFE_BASE_URL"
