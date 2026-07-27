#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

[[ ${1:-} == "--execute" && $# -eq 1 ]] ||
  local_e2e_die "usage: $0 --execute"

systemctl --user stop \
  hyprstream-local-e2e.target \
  hyprstream-local-e2e-pds.service \
  hyprstream-local-e2e-registry.service \
  hyprstream-local-e2e-inference@0.service \
  hyprstream-local-e2e-inference@1.service \
  hyprstream-local-e2e-discovery.service \
  hyprstream-local-e2e-policy.service || true

# systemd normally removes RuntimeDirectory after the last unit exits. If it
# remains, remove only a tree carrying this harness's exact ownership marker.
if [[ -d "$LOCAL_E2E_RUNTIME_DIR" && ! -L "$LOCAL_E2E_RUNTIME_DIR" ]]; then
  owner="$LOCAL_E2E_RUNTIME_DIR/local-e2e.owner"
  local_e2e_assert_regular_file "$owner"
  [[ "$(<"$owner")" == "$LOCAL_E2E_DIR" ]] ||
    local_e2e_die "runtime ownership marker does not match; refusing cleanup"
  rm -r -- "$LOCAL_E2E_RUNTIME_DIR"
fi

local_e2e_log \
  "services stopped; durable local state and private age identities were preserved"
