#!/usr/bin/env bash
# Honest offline smoke: no daemon, unit, container, credential, DNS, or network.
set -Eeuo pipefail

umask 077

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$harness_dir/../.." && pwd -P)"
git_common_dir="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)"
main_checkout="$(cd -- "$git_common_dir/.." && pwd -P)"
default_libtorch="$main_checkout/appimage/libtorch-cache/cuda130/libtorch"
libtorch_dir="${LIBTORCH:-$default_libtorch}"

# Committed fleet BuildQ wrapper (build-skills, feat/buildq-skill, reviewed
# and merged). Host-neutral resolution: an explicit override, then the
# $HOME-relative canonical layout, then PATH — the absolute
# /home/birdetta/... form alone does not hold across hosts with different
# layouts (e.g. hypr1, where neither ~/.local/bin/buildq nor a PATH entry
# exists).
buildq_sha256=5645f738b12c99638c9dd1c8b5f24d900d0593ffb39f75f0e02f228e3418619f
buildq_home_relative="$HOME/projects/cyberdione/build-skills/.claude/skills/buildq/scripts/buildq"
if [[ -n "${HYPRSTREAM_BUILDQ:-}" ]]; then
  buildq_candidate="$HYPRSTREAM_BUILDQ"
elif [[ -e "$buildq_home_relative" ]]; then
  buildq_candidate="$buildq_home_relative"
else
  buildq_candidate="$(command -v buildq 2>/dev/null || true)"
fi
[[ -n "$buildq_candidate" ]] || {
  printf 'offline-smoke: no BuildQ wrapper found (set HYPRSTREAM_BUILDQ, install at %s, or put buildq on PATH)\n' \
    "$buildq_home_relative" >&2
  exit 1
}
# A symlink is the portable, host-neutral entry point and is allowed; what
# must not be substituted is the content it names, so resolve through it and
# verify the resolved file's hash rather than rejecting symlinks outright.
buildq="$(realpath -e -- "$buildq_candidate" 2>/dev/null)" || {
  printf 'offline-smoke: BuildQ wrapper does not resolve to a real file: %s\n' \
    "$buildq_candidate" >&2
  exit 1
}
[[ -f "$buildq" && -x "$buildq" ]] || {
  printf 'offline-smoke: resolved BuildQ wrapper is not an executable regular file: %s\n' \
    "$buildq" >&2
  exit 1
}
buildq_actual_sha256="$(sha256sum -- "$buildq" | cut -d' ' -f1)"
[[ "$buildq_actual_sha256" == "$buildq_sha256" ]] || {
  printf 'offline-smoke: BuildQ wrapper content hash mismatch at %s (expected %s, got %s)\n' \
    "$buildq" "$buildq_sha256" "$buildq_actual_sha256" >&2
  exit 1
}

[[ -f "$repo/Cargo.lock" && ! -L "$repo/Cargo.lock" ]] || {
  printf 'offline-smoke: Cargo.lock must be a regular non-symlink file\n' >&2
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
