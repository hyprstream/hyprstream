#!/usr/bin/env bash
# Honest offline smoke: no daemon, unit, container, credential, DNS, or network.
set -Eeuo pipefail

umask 077

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$harness_dir/../.." && pwd -P)"
buildq=/home/birdetta/projects/cyberdione/.fleet-coord/buildq
git_common_dir="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)"
main_checkout="$(cd -- "$git_common_dir/.." && pwd -P)"
default_libtorch="$main_checkout/appimage/libtorch-cache/cuda130/libtorch"
libtorch_dir="${LIBTORCH:-$default_libtorch}"

[[ -f "$repo/Cargo.lock" && ! -L "$repo/Cargo.lock" ]] || {
  printf 'offline-smoke: Cargo.lock must be a regular non-symlink file\n' >&2
  exit 1
}
[[ -x "$buildq" && ! -L "$buildq" ]] || {
  printf 'offline-smoke: required fleet BuildQ wrapper is unavailable\n' >&2
  exit 1
}
[[ -d "$libtorch_dir/lib" && ! -L "$libtorch_dir" ]] || {
  printf 'offline-smoke: established libtorch cache is unavailable: %s\n' \
    "$libtorch_dir" >&2
  exit 1
}

"$harness_dir/tests/run.sh"

cd "$repo"
export LIBTORCH="$libtorch_dir"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$libtorch_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BUILDQ_SLOTS=1 BUILDQ_JOBS=4 "$buildq" -- \
  "$harness_dir/run-focused-cargo-smoke.sh"

printf '%s\n' \
  'offline-smoke: PASS (in-process only; ingest final contract and runtime E2E remain gated)'
