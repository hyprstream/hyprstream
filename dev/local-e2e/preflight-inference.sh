#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

replica=${1:?usage: preflight-inference.sh REPLICA}
[[ "$replica" == "0" || "$replica" == "1" ]] ||
  local_e2e_die "replica must be 0 or 1"

local_e2e_assert_socket "$LOCAL_E2E_RUNTIME_DIR/policy.sock"
local_e2e_assert_socket "$LOCAL_E2E_RUNTIME_DIR/discovery.sock"
local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica"
local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR/credentials/inference-cpu-$replica"
for path in signing-key service-jwt; do
  local_e2e_assert_regular_file \
    "$LOCAL_E2E_RUNTIME_DIR/credentials/inference-cpu-$replica/$path"
done
for path in quic-chain.pem quic-key.pem; do
  local_e2e_assert_regular_file "$LOCAL_E2E_RUNTIME_DIR/credentials/tls/$path"
  [[ -r "$LOCAL_E2E_RUNTIME_DIR/credentials/tls/$path" ]] ||
    local_e2e_die "QUIC material is unreadable: $path"
done
[[ "$(readlink -- "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica/policy.sock")" == "../policy.sock" ]] ||
  local_e2e_die "policy socket emulation link is not exact"
[[ "$(readlink -- "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica/discovery.sock")" == "../discovery.sock" ]] ||
  local_e2e_die "discovery socket emulation link is not exact"
