#!/usr/bin/env bash
# Merge-gate RELEASE / FEATURE LANES for the self-hosted Graviton (arm64)
# fleet, run INSIDE the rust-builder container as a NON-ROOT user (#1011),
# in parallel with the test job (graviton-build-test.sh). rust.yml invokes
# it as:
#   runuser -u ci -- bash -euo pipefail /build/.github/scripts/graviton-release-lanes.sh
# from the bind-mounted workspace (cwd = /build).
#
# Contains the phases the test suite does NOT consume (verified 2026-09-14:
# no test reads the release binaries; only the guest .wasm artifacts cross
# the job boundary, and those are built by the test job): the native
# release build, the pds-postgres feature lanes, and the credential-pds
# negative build gate. Running them as a parallel job cuts the merge-gate
# critical path from the serial sum (~52 min post metrics-lane removal) to
# max(test lane, this lane) + provisioning. No Chromium is needed here —
# the browser conformance phases live in the test job.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/verify-libtorch.sh"

# The rust toolchain lives under root's home in the image; the workflow made
# it a+rwX and created this ci user. CARGO_HOME/RUSTUP_HOME point back at it.
export PATH="/root/.cargo/bin:/usr/local/bin:${PATH}"
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export SCCACHE_DIR="${PWD}/.sccache-release"
mkdir -p "${SCCACHE_DIR}"

# Same-filesystem TMPDIR (parity with graviton-build-test.sh): temp stays on
# the workspace filesystem.
export TMPDIR="${PWD}/.citmp-release"
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
  # Cumulative sccache stats after every phase (parity with the test job:
  # surviving evidence of cache misses / write errors on slow or killed runs).
  sccache --show-stats || true
  echo "ci-phase name=${label@Q} elapsed_seconds=${elapsed} status=${status}"
  echo "::endgroup::"
  return "${status}"
}

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
# The KV shell and RDS contract tests live in hyprstream-pds (pgsql_kv:: and
# rds::tests::); the resolver-side Postgres accepted-state authority tests
# live in hyprstream-discovery (checkpointed_pds::pg_tests::). The app crate
# keeps the store-level PG tests (services::discovery::pg_tests::).
run_phase "pds-postgres contract tests" cargo test -p hyprstream --locked --lib --features pds-postgres -- services::discovery::pg_tests::
run_phase "hyprstream-pds postgres contract tests" cargo test -p hyprstream-pds --locked --lib --features postgres -- pgsql_kv:: rds::tests::
run_phase "discovery postgres authority contract tests" cargo test -p hyprstream-discovery --locked --lib --features postgres,rocksdb -- checkpointed_pds::

# Prove the production credential profile is causal: omitting it must fail the
# build, never silently skip the deployable target. Compile-only and negative
# (its pass condition is a build failure), so it runs last.
run_phase "credential-pds negative build gate" \
  bash .github/scripts/credential-pds-build-gate.sh

echo "::group::sccache statistics"
sccache --show-stats
echo "::endgroup::"
