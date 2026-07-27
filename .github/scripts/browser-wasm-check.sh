#!/usr/bin/env bash
# Compile the complete Rust surface consumed by www's browser WASM build.
#
# Keep this as the single source of truth for both the fast PR WASM job and the
# required merge-group build. scripts/build-wasm.sh in www uses wasm-pack for
# hyprstream-rpc and hyprstream-rpc-std; rpc-std currently pulls in
# hyprstream-vfs transitively. All three packages remain explicit here so a
# future dependency refactor cannot silently remove VFS from the required gate.
set -euo pipefail

append_rustflag() {
  local flag="$1"
  if [[ " ${RUSTFLAGS:-} " != *" ${flag} "* ]]; then
    RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }${flag}"
  fi
}

# Match www/scripts/build-wasm.sh. WebTransport needs the unstable web-sys API
# cfg, while moq-net's rand/getrandom 0.3 graph needs the browser WebCrypto
# backend selected explicitly.
append_rustflag '--cfg=web_sys_unstable_apis'
append_rustflag '--cfg=getrandom_backend="wasm_js"'
export RUSTFLAGS

readonly -a BROWSER_WASM_PACKAGES=(
  hyprstream-rpc
  hyprstream-vfs
  hyprstream-rpc-std
)

package_args=()
for package in "${BROWSER_WASM_PACKAGES[@]}"; do
  package_args+=( -p "$package" )
done

cargo check --locked \
  --target wasm32-unknown-unknown \
  "${package_args[@]}"
