#!/usr/bin/env bash
# Offline/static verification only: no build, mint, daemon, systemctl, or net.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

for command in bash rg sed systemd-analyze mktemp; do
  local_e2e_require_command "$command"
done

while IFS= read -r script; do
  bash -n "$script"
done < <(find "$LOCAL_E2E_DIR" -type f -name '*.sh' -print)

tmp_units="$(mktemp -d)"
trap 'rm -r -- "$tmp_units"' EXIT
for template in "$LOCAL_E2E_DIR"/systemd/*.in; do
  name="$(basename -- "$template" .in)"
  sed \
    -e 's|@@HYPRSTREAM_BIN@@|/bin/true|g' \
    -e "s|@@CONFIG_DIR@@|$LOCAL_E2E_CONFIG_DIR|g" \
    -e "s|@@RUNTIME_DIR@@|$LOCAL_E2E_RUNTIME_DIR|g" \
    -e "s|@@HARNESS_DIR@@|$LOCAL_E2E_DIR|g" \
    "$template" >"$tmp_units/$name"
done
systemd-analyze --user verify "$tmp_units"/* >/dev/null

rg -q 'service start oauth --foreground --ipc' \
  "$LOCAL_E2E_DIR/systemd/hyprstream-local-e2e-pds.service.in"
rg -q 'service start inference --foreground --ipc' \
  "$LOCAL_E2E_DIR/systemd/hyprstream-local-e2e-inference@.service.in"
rg -q 'HYPRSTREAM_INSTANCE=inference-cpu-%i' \
  "$LOCAL_E2E_DIR/systemd/hyprstream-local-e2e-inference@.service.in"
rg -q 'quic_port = %s' "$LOCAL_E2E_DIR/prepare-local-state.sh"
rg -Fq '$((7440 + replica))' "$LOCAL_E2E_DIR/prepare-local-state.sh"

rg -q 'MintDeploymentCa|DelegateRegistrySigner|MintRegistryJwt|VerifyDeployment' \
  < <(git -C "$HYPRSTREAM_REPO" show \
    feat/trust-mint-v2:crates/hyprstream/src/cli/commands/trust.rs)
rg -Fq 'const PUBLIC_CA_BYTES: usize = 32 + 1_952' \
  < <(git -C "$HYPRSTREAM_REPO" show \
    feat/trust-mint-v2:crates/hyprstream/src/cli/trust.rs)

loader_head=a3d38414e0dbeb0fd2d1495c3799dc119bf11c39
loader_source=crates/hyprstream-discovery/src/service.rs
rg -Fq \
  'const DEPLOYMENT_TRUST_DIR_ENV: &str = "HYPRSTREAM_DEPLOYMENT_TRUST_DIR";' \
  < <(git -C "$HYPRSTREAM_REPO" show "$loader_head:$loader_source")
rg -Fq \
  'const SYSTEMD_CREDENTIALS_DIRECTORY_ENV: &str = "CREDENTIALS_DIRECTORY";' \
  < <(git -C "$HYPRSTREAM_REPO" show "$loader_head:$loader_source")
rg -Fq 'directory.join(DEPLOYMENT_CA_ROOT_FILE)' \
  < <(git -C "$HYPRSTREAM_REPO" show "$loader_head:$loader_source")
rg -Fq 'directory.join(REGISTRY_DEPLOYMENT_CREDENTIAL_FILE)' \
  < <(git -C "$HYPRSTREAM_REPO" show "$loader_head:$loader_source")
rg -Fq \
  'HYPRSTREAM_DEPLOYMENT_TRUST_DIR=$LOCAL_E2E_RUNTIME_DIR/credentials/trust' \
  "$LOCAL_E2E_DIR/prepare-local-state.sh"
rg -Fq \
  'CREDENTIALS_DIRECTORY=$LOCAL_E2E_RUNTIME_DIR/credentials' \
  "$LOCAL_E2E_DIR/prepare-local-state.sh"
if ! git -C "$HYPRSTREAM_REPO" merge-base --is-ancestor "$loader_head" HEAD; then
  local_e2e_log \
    "EXPECTED PRE-START BLOCKER: current checkout has not composed open #1373"
fi

local_e2e_log "staged harness passes static shell/unit/source-contract checks"
