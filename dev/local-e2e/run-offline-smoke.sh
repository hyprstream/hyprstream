#!/usr/bin/env bash
# Honest offline smoke: no daemon, unit, container, credential, DNS, or network.
set -Eeuo pipefail

umask 077

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$harness_dir/../.." && pwd -P)"
default_helper=/home/birdetta/projects/cyberdione/build-skills/.claude/skills/hyprstream-build/scripts/hyprstream-build.sh
build_helper="${HYPRSTREAM_BUILD_HELPER:-$default_helper}"

[[ -f "$repo/Cargo.lock" && ! -L "$repo/Cargo.lock" ]] || {
  printf 'offline-smoke: Cargo.lock must be a regular non-symlink file\n' >&2
  exit 1
}
[[ -x "$build_helper" && ! -L "$build_helper" ]] || {
  printf 'offline-smoke: set HYPRSTREAM_BUILD_HELPER to the shared build helper\n' >&2
  exit 1
}

"$harness_dir/tests/run.sh"

cd "$repo"
"$build_helper" run -- "$harness_dir/run-focused-cargo-smoke.sh"

printf '%s\n' \
  'offline-smoke: PASS (in-process only; ingest final contract and runtime E2E remain gated)'
