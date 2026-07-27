#!/usr/bin/env bash
# Explicit local-only activation controller. Staging/verification never calls it.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

execute=0
model_path=
model_oid=
advertise_ip=
while (($#)); do
  case "$1" in
    --execute) execute=1 ;;
    --model-path) shift; model_path=${1:?--model-path requires a value} ;;
    --model-oid) shift; model_oid=${1:?--model-oid requires a value} ;;
    --advertise-ip) shift; advertise_ip=${1:?--advertise-ip requires a value} ;;
    *)
      local_e2e_die \
        "usage: $0 --execute --model-path PATH --model-oid OID --advertise-ip NON_LOOPBACK_IPV4"
      ;;
  esac
  shift
done
((execute == 1)) ||
  local_e2e_die "refusing activation without explicit --execute"
[[ -n "$model_path" && -n "$model_oid" && -n "$advertise_ip" ]] ||
  local_e2e_die "model path, model OID, and advertise IP are required"

for command in cargo curl git install systemctl; do
  local_e2e_require_command "$command"
done

local_e2e_log "building the native release binary (no service is running)"
(
  cd "$HYPRSTREAM_REPO"
  cargo build --release --bin hyprstream
)
local_e2e_assert_regular_file "$LOCAL_E2E_BINARY"

LOCAL_E2E_MODEL_PATH="$model_path" \
LOCAL_E2E_MODEL_OID="$model_oid" \
LOCAL_E2E_ADVERTISE_IP="$advertise_ip" \
  "$LOCAL_E2E_DIR/fixtures/generate-local-tls.sh"
LOCAL_E2E_MODEL_PATH="$model_path" \
LOCAL_E2E_MODEL_OID="$model_oid" \
LOCAL_E2E_ADVERTISE_IP="$advertise_ip" \
  "$LOCAL_E2E_DIR/prepare-local-state.sh"
"$LOCAL_E2E_DIR/mint-local-trust.sh"
"$LOCAL_E2E_DIR/project-local-runtime.sh" credentials
"$LOCAL_E2E_DIR/install-user-units.sh"

# This is deliberately before daemon-reload/start. On #1371 it always stops
# here because the verifier loader has only root-owned fixed paths.
"$LOCAL_E2E_DIR/require-user-trust-loader.sh"

systemctl --user daemon-reload
systemctl --user start hyprstream-local-e2e-policy.service
local_e2e_wait_socket "$LOCAL_E2E_RUNTIME_DIR/policy.sock"
systemctl --user start hyprstream-local-e2e-discovery.service
local_e2e_wait_socket "$LOCAL_E2E_RUNTIME_DIR/discovery.sock"

"$LOCAL_E2E_DIR/project-local-runtime.sh" links

systemctl --user start \
  hyprstream-local-e2e-registry.service \
  hyprstream-local-e2e-pds.service \
  hyprstream-local-e2e-inference@0.service \
  hyprstream-local-e2e-inference@1.service

local_e2e_wait_socket "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-0/inference.sock" 120
local_e2e_wait_socket "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-1/inference.sock" 120

for _ in $(seq 1 60); do
  if curl --silent --show-error --fail \
    --cacert "$LOCAL_E2E_STATE_DIR/tls/local-e2e-ca.pem" \
    --resolve pds.accounts.localhost:6791:127.0.0.1 \
    https://pds.accounts.localhost:6791/.well-known/oauth-authorization-server \
    >/dev/null; then
    break
  fi
  sleep 1
done
curl --silent --show-error --fail \
  --cacert "$LOCAL_E2E_STATE_DIR/tls/local-e2e-ca.pem" \
  --resolve pds.accounts.localhost:6791:127.0.0.1 \
  https://pds.accounts.localhost:6791/.well-known/oauth-authorization-server \
  >/dev/null

"$LOCAL_E2E_DIR/project-local-runtime.sh" check
systemctl --user start hyprstream-local-e2e.target
local_e2e_log "local E2E fabric is socket/HTTP ready; run run-e2e.sh --execute"
