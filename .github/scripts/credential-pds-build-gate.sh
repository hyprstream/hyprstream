#!/usr/bin/env bash
# Prove that the former production release command fails closed when the
# credential-pds feature is absent. At PR #1414's reviewed head (670715389),
# this command exited zero after silently skipping the hyprstream binary.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${HYPRSTREAM_REPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"
buildq="${HYPRSTREAM_BUILDQ:-}"
buildq_slots="${HYPRSTREAM_BUILDQ_SLOTS:-2}"

if [[ ! -f "${repo_root}/Cargo.toml" ]]; then
  echo "credential-pds gate: repository root has no Cargo.toml: ${repo_root}" >&2
  exit 2
fi

run_cargo() {
  if [[ -n "${buildq}" ]]; then
    "${buildq}" --slots "${buildq_slots}" -- cargo "$@"
  else
    cargo "$@"
  fi
}

gate_log="$(mktemp "${TMPDIR:-/tmp}/credential-pds-build-gate.XXXXXX.log")"
trap 'rm -f "${gate_log}"' EXIT

set +e
(
  cd -- "${repo_root}"
  run_cargo build --locked --release --no-default-features \
    --features otel,gittorrent,xet
) 2>&1 | tee "${gate_log}"
build_status=${PIPESTATUS[0]}
set -e

if [[ "${build_status}" -eq 0 ]]; then
  echo "credential-pds gate: feature-less release build unexpectedly succeeded" >&2
  exit 1
fi

if ! grep -Fq 'every Hyprstream build requires the `credential-pds` feature' "${gate_log}"; then
  echo "credential-pds gate: build failed without the provenance diagnostic" >&2
  exit 1
fi

echo "credential-pds gate: expected feature-less build failure (status ${build_status})"
