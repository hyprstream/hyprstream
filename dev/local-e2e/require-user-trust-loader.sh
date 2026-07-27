#!/usr/bin/env bash
# Validate PR #1373's exact rootless deployment-trust selector contract.
# Current main fails this pre-start guard until #1371 + stacked #1373 are
# merged/composed into the binary source.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

for path in \
  "$LOCAL_E2E_RUNTIME_DIR/credentials/trust/deployment-ca.hybrid" \
  "$LOCAL_E2E_RUNTIME_DIR/credentials/trust/deployment-authority.log.json" \
  "$LOCAL_E2E_RUNTIME_DIR/credentials/trust/deployment-authority.head.json" \
  "$LOCAL_E2E_RUNTIME_DIR/credentials/registry-service.jwt"; do
  local_e2e_assert_regular_file "$path"
done

loader_source="$HYPRSTREAM_REPO/crates/hyprstream-discovery/src/service.rs"
if ! rg -Fq \
  'const DEPLOYMENT_TRUST_DIR_ENV: &str = "HYPRSTREAM_DEPLOYMENT_TRUST_DIR";' \
  "$loader_source" ||
  ! rg -Fq \
    'const SYSTEMD_CREDENTIALS_DIRECTORY_ENV: &str = "CREDENTIALS_DIRECTORY";' \
    "$loader_source" ||
  ! rg -Fq 'resolve_deployment_trust_paths()' "$loader_source"; then
  local_e2e_die \
    "current source has not composed open #1373 at a3d38414 on #1371; refusing pre-start"
fi

env_file="$LOCAL_E2E_CONFIG_DIR/trust-loader.env"
local_e2e_assert_regular_file "$env_file"
[[ "$(wc -l <"$env_file")" == "2" ]] ||
  local_e2e_die "trust-loader.env must contain exactly the two reviewed selectors"
rg -Fxq \
  "HYPRSTREAM_DEPLOYMENT_TRUST_DIR=$LOCAL_E2E_RUNTIME_DIR/credentials/trust" \
  "$env_file" ||
  local_e2e_die "trust-loader.env has an unexpected deployment trust directory"
rg -Fxq \
  "CREDENTIALS_DIRECTORY=$LOCAL_E2E_RUNTIME_DIR/credentials" \
  "$env_file" ||
  local_e2e_die "trust-loader.env has an unexpected credentials directory"

local_e2e_log "rootless trust loader contract matches reviewed #1373"
