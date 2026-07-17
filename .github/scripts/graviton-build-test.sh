#!/usr/bin/env bash
# Merge-gate build + test for the self-hosted Graviton (arm64) fleet, run INSIDE
# the rust-builder container as a NON-ROOT user (#1011). rust.yml invokes it as:
#   runuser -u ci -- bash -euo pipefail /build/.github/scripts/graviton-build-test.sh
# from the bind-mounted workspace (cwd = /build). Root-only setup (cargo-nextest,
# git-lfs, wasm rustup targets, the ci user + perms) happens in the workflow before
# this runs; kept in a committed file so it is readable/reviewable rather than an
# escaped inline heredoc.
set -euo pipefail

# The rust toolchain lives under root's home in the image; the workflow made it
# a+rwX and created this ci user. CARGO_HOME/RUSTUP_HOME point back at it.
export PATH="/root/.cargo/bin:/usr/local/bin:${PATH}"
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export SCCACHE_DIR="${PWD}/.sccache"
mkdir -p "${SCCACHE_DIR}"

# Same-filesystem TMPDIR: the overlay_fsmount tests rename across layers, which
# fails EXDEV (cross-device link) when temp dirs land on /tmp (tmpfs) while the
# workspace is on another filesystem. Keep temp on the workspace fs.
export TMPDIR="${PWD}/.citmp"
mkdir -p "${TMPDIR}"

# Default features (parity with the former x86 gate); libtorch is the image's
# aarch64 wheel at /opt/libtorch, so NO download-libtorch feature here.
cargo build --release

# wasm guest artifacts for the sandbox/mount tests (deny-on-missing-guest guard).
# cd INTO each guest crate so cargo reads its .cargo/config.toml (the python guest
# needs getrandom_backend="custom" for wasm32-unknown-unknown; see #1013).
( cd crates/hyprstream-workers-python-guest && cargo build --release --target wasm32-unknown-unknown )
( cd crates/hyprstream-workers-wasmtime-fsguest && cargo build --release --target wasm32-wasip1 )
export HYPRSTREAM_PYGUEST_WASM="${PWD}/crates/hyprstream-workers-python-guest/target/wasm32-unknown-unknown/release/hyprstream_workers_python_guest.wasm"
export HYPRSTREAM_FSGUEST_WASM="${PWD}/crates/hyprstream-workers-wasmtime-fsguest/target/wasm32-wasip1/release/hyprstream-workers-wasmtime-fsguest.wasm"

# nextest ci profile enforces the per-test slow-timeout from .config/nextest.toml;
# fail-fast (the default) is what the merge gate wants.
cargo nextest run --cargo-profile ci-test --profile ci
# nextest does not run doctests; keep them in the merge gate.
cargo test --release --doc

sccache --show-stats
