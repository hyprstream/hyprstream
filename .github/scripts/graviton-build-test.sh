#!/usr/bin/env bash
# Merge-gate browser-WASM check + native build/test for the self-hosted Graviton
# (arm64) fleet, run INSIDE the rust-builder container as a NON-ROOT user
# (#1011). rust.yml invokes it as:
#   runuser -u ci -- bash -euo pipefail /build/.github/scripts/graviton-build-test.sh
# from the bind-mounted workspace (cwd = /build). Root-only setup (cargo-nextest,
# git-lfs, wasm rustup targets, the ci user + perms) happens in the workflow before
# this runs; kept in a committed file so it is readable/reviewable rather than an
# escaped inline heredoc.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/verify-libtorch.sh"

# The rust toolchain lives under root's home in the image; the workflow made it
# a+rwX and created this ci user. CARGO_HOME/RUSTUP_HOME point back at it.
export PATH="/root/.cargo/bin:/usr/local/bin:${PATH}"
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export SCCACHE_DIR="${PWD}/.sccache"
mkdir -p "${SCCACHE_DIR}"

# These are standalone workspaces, so Cargo's default target directory is
# relative to each guest crate.  When the CI workflow supplies a shared target
# directory, Cargo resolves a relative value from the crate's working
# directory and an absolute value verbatim.  Keep that resolution in one
# place so the artifacts we hand to the guest tests are the artifacts the
# preceding Cargo commands actually produced.
guest_target_dir() {
  local crate_dir="$1"
  if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    if [[ "${CARGO_TARGET_DIR}" = /* ]]; then
      printf '%s\n' "${CARGO_TARGET_DIR}"
    else
      printf '%s/%s\n' "${crate_dir}" "${CARGO_TARGET_DIR}"
    fi
  else
    printf '%s/target\n' "${crate_dir}"
  fi
}

require_guest_artifact() {
  local label="$1"
  local artifact="$2"
  if [[ ! -r "${artifact}" || ! -s "${artifact}" ]]; then
    echo "${label} guest artifact is missing, unreadable, or empty: ${artifact}" >&2
    return 1
  fi
}

# Same-filesystem TMPDIR: the overlay_fsmount tests rename across layers, which
# fails EXDEV (cross-device link) when temp dirs land on /tmp (tmpfs) while the
# workspace is on another filesystem. Keep temp on the workspace fs.
export TMPDIR="${PWD}/.citmp"
mkdir -p "${TMPDIR}"

# Final sccache stats on ANY exit path (bash runs EXIT traps on SIGTERM as
# well; only SIGKILL loses them — the per-phase stats in run_phase are the
# backstop for hard kills).
trap 'sccache --show-stats || true' EXIT

run_phase() {
  local label="$1"
  shift
  local started="${SECONDS}"
  local status

  echo "::group::${label}"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  local elapsed=$((SECONDS - started))
  # Cumulative sccache stats after every phase. When a run is slow or gets
  # killed, these are the only surviving evidence of cache misses / write
  # errors: the 2026-09-14 140-min timeout (run 34867497318) destroyed the
  # end-of-script stats that would have explained the run, costing hours of
  # forensics (resolved only via the re-run's surviving stats — 3,288 misses,
  # 2,261 write errors).
  sccache --show-stats || true
  echo "ci-phase name=${label@Q} elapsed_seconds=${elapsed} status=${status}"
  echo "::endgroup::"
  return "${status}"
}

# Fail fast on browser-only composition regressions in the required merge-gate
# job. The child script scopes both browser cfgs to its own process so they
# cannot leak into native release/test phases below. It checks the exact
# browser-facing rpc + vfs + rpc-std package set used by www.
run_phase "browser WASM check" bash .github/scripts/browser-wasm-check.sh

# #1425 r4: the compile-only check above cannot catch a regression in the
# actual browser-fetch runtime behavior (JS callback, Request/Response, nonce
# retry, response rejection) — only real execution can. Chromium + a matching
# chromedriver were installed as root by the workflow before this script
# dropped to the non-root `ci` user (see rust.yml's `build` job), so
# browser-wasm-test-ci.sh finds them already on PATH and skips straight to
# resolving + running crates/hyprstream-rpc/tests/wasm_browser_fetch.rs in a
# real headless Chromium — the same required-merge-gate invariant the fast PR
# `WASM (browser client)` job checks, but the fast job is explicitly skipped
# on `merge_group` (rust.yml:167), so this is the only required-gate path that
# actually launches a browser for the synthetic merge candidate.
run_phase "browser WASM real execution" bash .github/scripts/browser-wasm-test-ci.sh

# TEST LANE FIRST (2026-09-15 merge-queue spike): run everything the test
# suite consumes before any release/feature lane. nextest and doctests need
# ONLY the guest .wasm artifacts (verified: no test reads the release
# binaries — only HYPRSTREAM_PYGUEST_WASM / HYPRSTREAM_FSGUEST_WASM cross
# this boundary), and test failures were previously discovered after ~50 min
# of release-lane work they never consumed. Ordering the test lane second
# surfaces the most common failure class (2026-09-13/14 samples: 2 of 4 red
# candidates were nextest failures found at 63-68 min) at ~25 min instead.
# Green-run wall time is unchanged — the same phases run serially either way.

# wasm guest artifacts for the sandbox/mount tests (deny-on-missing-guest guard).
# cd INTO each guest crate so cargo reads its .cargo/config.toml (the python guest
# needs getrandom_backend="custom" for wasm32-unknown-unknown; see #1013).
run_phase "Python guest WASM build" bash -c \
  'cd crates/hyprstream-workers-python-guest && cargo build --release --target wasm32-unknown-unknown'
run_phase "Wasmtime guest WASM build" bash -c \
  'cd crates/hyprstream-workers-wasmtime-fsguest && cargo build --release --target wasm32-wasip1'
PYTHON_GUEST_DIR="${PWD}/crates/hyprstream-workers-python-guest"
FSGUEST_DIR="${PWD}/crates/hyprstream-workers-wasmtime-fsguest"
export HYPRSTREAM_PYGUEST_WASM="$(guest_target_dir "${PYTHON_GUEST_DIR}")/wasm32-unknown-unknown/release/hyprstream_workers_python_guest.wasm"
export HYPRSTREAM_FSGUEST_WASM="$(guest_target_dir "${FSGUEST_DIR}")/wasm32-wasip1/release/hyprstream-workers-wasmtime-fsguest.wasm"
require_guest_artifact "Python" "${HYPRSTREAM_PYGUEST_WASM}"
require_guest_artifact "Wasmtime" "${HYPRSTREAM_FSGUEST_WASM}"

# nextest ci profile enforces the per-test slow-timeout from .config/nextest.toml;
# fail-fast (the default) is what the merge gate wants.
run_phase "nextest" cargo nextest run --cargo-profile ci-test --profile ci
# nextest does not run doctests; keep them in the merge gate. Use the ci-test
# cargo profile (not --release) so doctests skip release ThinLTO — #1010's
# acceptance criterion: a measured 318s doctest phase was paying release LTO
# for almost no runtime win on doc examples. Parity with the nextest profile.
run_phase "doctests" cargo test --profile ci-test --doc

# RELEASE / FEATURE LANES (last): none of these feed the test suite above.

# Default features (parity with the former x86 gate); libtorch is the image's
# aarch64 wheel at /opt/libtorch, so NO download-libtorch feature here.
run_phase "native release build" cargo build --release

# The `metrics` standalone profile (DuckDB-backed; mutually exclusive with the
# default PGlite build at link time) was removed from the required gate on
# 2026-09-14: nothing ships or deploys that profile today and its two serial
# phases cost ~19 min of every merge candidate. Compile + typed-handler-test
# coverage continues in .github/workflows/metrics-profile-nightly.yml; restore
# both phases here if the metrics service becomes production.

# The RDS-backed PDS record store (#1257) is feature-gated and absent from
# the default-feature build above; check its full target set and run its
# contract/unit tests. Live DB tests skip themselves green unless
# HYPRSTREAM_POSTGRES_TEST_URL_FILE points at a scratch database.
run_phase "pds-postgres feature check" cargo check -p hyprstream --locked --all-targets --features pds-postgres
run_phase "pds-postgres contract tests" cargo test -p hyprstream --locked --lib --features pds-postgres -- services::pds_record_pg:: services::discovery::pg_tests:: config::tests::rds

# Prove the production credential profile is causal: omitting it must fail the
# build, never silently skip the deployable target. Compile-only and negative
# (its pass condition is a failure), so it runs last.
run_phase "credential-pds negative build gate" \
  bash .github/scripts/credential-pds-build-gate.sh

echo "::group::sccache statistics"
sccache --show-stats
echo "::endgroup::"
