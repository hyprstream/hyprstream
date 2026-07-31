#!/usr/bin/env bash
# Execute the #1425 browser-fetch conformance test
# (crates/hyprstream-rpc/tests/wasm_browser_fetch.rs) in a real headless
# browser, as an isolated named wasm test artifact.
#
# Deliberately does NOT use `wasm-pack test`: wasm-pack unconditionally passes
# `--tests` to its underlying `cargo build`, which additionally compiles every
# other test target in the crate — including the lib's own unit-test target,
# which has hundreds of native-only `#[tokio::test]`s scattered through
# hyprstream-rpc's source files (transport, service, moq_stream, federation
# key sources, event crypto, ...). None of that is wasm32-buildable, so
# `wasm-pack test ... --test wasm_browser_fetch` fails during compilation
# before a browser is ever launched.
#
# Building the NAMED integration test directly via `cargo test --test <name>`
# (never `--tests`) links the lib crate normally (no `cfg(test)`), so none of
# the lib's own internal unit tests are pulled in — only this one artifact
# plus its own dependencies.
#
# The compiled `.wasm` test binary is then executed by a version-matched
# `wasm-bindgen-test-runner`: the runner's ABI must exactly match the
# `wasm-bindgen` crate version this workspace resolves (Cargo.lock), or the
# compiled test's `__wbindgen_describe` schema will not agree with what the
# runner expects. `cargo install wasm-bindgen-cli --version <exact>` builds
# that exact match; `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER` points cargo
# at it so `cargo test --target wasm32-unknown-unknown` invokes it as the
# runner for the produced `.wasm` artifact, which is what actually launches
# the (headless) browser and reports pass/fail.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CRATE_MANIFEST="$REPO_ROOT/crates/hyprstream-rpc/Cargo.toml"
TEST_NAME="wasm_browser_fetch"

append_rustflag() {
  local flag="$1"
  if [[ " ${RUSTFLAGS:-} " != *" ${flag} "* ]]; then
    RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }${flag}"
  fi
}

# Match browser-wasm-check.sh / www/scripts/build-wasm.sh.
append_rustflag '--cfg=web_sys_unstable_apis'
append_rustflag '--cfg=getrandom_backend="wasm_js"'
export RUSTFLAGS

# Resolve the exact wasm-bindgen version this workspace locks, from the
# workspace Cargo.lock (never guessed/hardcoded — it must track whatever the
# dependency graph actually resolves to, or the runner ABI check fails).
WASM_BINDGEN_VERSION="$(
  awk '
    /^name = "wasm-bindgen"$/ { found=1; next }
    found && /^version = / {
      gsub(/^version = "|"$/, "");
      print;
      exit
    }
  ' "$REPO_ROOT/Cargo.lock"
)"
[[ -n "$WASM_BINDGEN_VERSION" ]]

RUNNER_HOME="${BROWSER_WASM_TEST_RUNNER_HOME:-$REPO_ROOT/target/wasm-bindgen-test-runner-${WASM_BINDGEN_VERSION}}"
RUNNER_BIN="$RUNNER_HOME/bin/wasm-bindgen-test-runner"

if [[ ! -x "$RUNNER_BIN" ]]; then
  echo "installing wasm-bindgen-cli ${WASM_BINDGEN_VERSION} (version-matched test runner) into $RUNNER_HOME" >&2
  cargo install --locked --root "$RUNNER_HOME" --version "$WASM_BINDGEN_VERSION" wasm-bindgen-cli
fi

export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER="$RUNNER_BIN"

# `--test` (singular, named) — NOT `--tests` — is the entire point: it builds
# and runs only this one integration test binary.
cargo test --locked \
  --manifest-path "$CRATE_MANIFEST" \
  --target wasm32-unknown-unknown \
  --test "$TEST_NAME"

# Runtime execution receipt. When the caller supplies BROWSER_ATTEST_NONCE,
# echo it into a receipt file — reached only after the browser run above
# succeeded (set -e). The caller asserts the receipt out-of-band, so a change
# anywhere in the invocation chain that stops this script from running (or
# from reaching this point) is detected by the receipt being absent or stale,
# with no static analysis of the chain required. Local runs without the nonce
# write nothing.
if [[ -n "${BROWSER_ATTEST_NONCE:-}" ]]; then
  printf '%s\n' "$BROWSER_ATTEST_NONCE" > "$REPO_ROOT/browser-wasm-receipt.txt"
fi
