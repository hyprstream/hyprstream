#!/usr/bin/env bash
# Render config and bootstrap per-service service credentials. This performs
# local file writes only and does not start a daemon.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

model_path=${LOCAL_E2E_MODEL_PATH:?set LOCAL_E2E_MODEL_PATH to an immutable model checkout}
model_oid=${LOCAL_E2E_MODEL_OID:?set LOCAL_E2E_MODEL_OID to the exact model checkout Git OID}
advertise_ip=${LOCAL_E2E_ADVERTISE_IP:?set LOCAL_E2E_ADVERTISE_IP to a non-loopback local IP}

[[ "$model_path" == /* ]] || local_e2e_die "model path must be absolute"
[[ -d "$model_path" ]] || local_e2e_die "model path is not a directory: $model_path"
[[ "$model_oid" =~ ^([0-9a-fA-F]{40}|[0-9a-fA-F]{64})$ ]] ||
  local_e2e_die "model OID must be 40 or 64 hexadecimal characters"
[[ "$(git -C "$model_path" rev-parse HEAD)" == "$model_oid" ]] ||
  local_e2e_die "model checkout HEAD does not match LOCAL_E2E_MODEL_OID"
[[ -z "$(git -C "$model_path" status --porcelain=v1 --untracked-files=all --ignored=matching -- .)" ]] ||
  local_e2e_die "model subtree contains tracked, untracked, or ignored changes"
[[ "$advertise_ip" != "127.0.0.1" && "$advertise_ip" != "::1" && \
   "$advertise_ip" != "0.0.0.0" && "$advertise_ip" != "::" ]] ||
  local_e2e_die "inference code rejects loopback/unspecified advertise addresses"
[[ "$advertise_ip" != *:* ]] ||
  local_e2e_die "this staged renderer currently accepts an IPv4 advertise address only"

tls_dir="$LOCAL_E2E_STATE_DIR/tls"
bootstrap_dir="$LOCAL_E2E_STATE_DIR/bootstrap-credentials"
install -d -m 0700 "$LOCAL_E2E_CONFIG_DIR" "$LOCAL_E2E_STATE_DIR" "$bootstrap_dir"
install -d -m 0700 "$LOCAL_E2E_STATE_DIR/xdg-state" "$LOCAL_E2E_STATE_DIR/xdg-data"

write_common_config() {
  local destination=$1
  {
    printf '[services]\n'
    printf 'startup = ["policy", "discovery", "registry", "oauth", "inference"]\n\n'
    printf '[secrets]\npath = "%s"\n\n' "$bootstrap_dir"
    printf '[tls]\nenabled = true\nmode = "files"\n'
    printf 'server_name = "pds.accounts.localhost"\n'
    printf 'cert_path = "%s/quic-chain.pem"\n' "$tls_dir"
    printf 'key_path = "%s/quic-key.pem"\n\n' "$tls_dir"
    printf '[oauth]\nhost = "127.0.0.1"\nport = 6791\n'
    printf 'external_url = "https://pds.accounts.localhost:6791"\n'
    printf 'tls_cert = "%s/quic-chain.pem"\n' "$tls_dir"
    printf 'tls_key = "%s/quic-key.pem"\n' "$tls_dir"
    printf 'require_pushed_authorization_requests = true\n'
    printf 'xrpc_read_slice = true\n\n'
    printf '[oauth.cors]\nenabled = true\n'
    printf 'allowed_origins = ["http://localhost:3000", "https://www.accounts.localhost"]\n'
    printf 'allow_credentials = true\nmax_age = 3600\npermissive_headers = false\n\n'
    printf '[account]\nzone = "accounts.localhost"\n'
    printf 'wildcard_ipv4 = "127.0.0.1"\nwildcard_ipv6 = "::1"\n\n'
  } >"$destination"
  chmod 0600 "$destination"
}

write_common_config "$LOCAL_E2E_CONFIG_DIR/pds.toml"

for replica in 0 1; do
  config="$LOCAL_E2E_CONFIG_DIR/inference-$replica.toml"
  write_common_config "$config"
  {
    printf '[inference]\n'
    printf 'model_path = "%s"\n' "$model_path"
    printf 'model_ref = "model://local-e2e/demo/replica-%s"\n' "$replica"
    printf 'model_oid = "%s"\n' "$model_oid"
    printf 'tenant = "pds.accounts.localhost"\n'
    printf 'replica = %s\nstage_start = 0\nquic_port = %s\n' "$replica" "$((7440 + replica))"
    printf 'advertise_addr = "%s:%s"\n\n' "$advertise_ip" "$((7440 + replica))"
    printf '[quic]\nenabled = true\nbind_addr = "0.0.0.0:%s"\n' "$((7440 + replica))"
    printf 'server_name = "pds.accounts.localhost"\n'
    printf 'cert_path = "%s/quic-chain.pem"\n' "$tls_dir"
    printf 'key_path = "%s/quic-key.pem"\n' "$tls_dir"
    printf 'iroh = false\nrelay = ""\n'
  } >>"$config"
done

cat >"$LOCAL_E2E_CONFIG_DIR/runtime.env" <<EOF
HYPRSTREAM_BIN=$LOCAL_E2E_BINARY
HYPRSTREAM_LOCAL_E2E_CONFIG_DIR=$LOCAL_E2E_CONFIG_DIR
HYPRSTREAM_LOCAL_E2E_STATE_DIR=$LOCAL_E2E_STATE_DIR
HYPRSTREAM_LOCAL_E2E_RUNTIME_DIR=$LOCAL_E2E_RUNTIME_DIR
XDG_CONFIG_HOME=$LOCAL_E2E_CONFIG_HOME
XDG_STATE_HOME=$LOCAL_E2E_STATE_DIR/xdg-state
XDG_DATA_HOME=$LOCAL_E2E_STATE_DIR/xdg-data
EOF
chmod 0600 "$LOCAL_E2E_CONFIG_DIR/runtime.env"

cat >"$LOCAL_E2E_CONFIG_DIR/trust-loader.env" <<EOF
HYPRSTREAM_DEPLOYMENT_TRUST_DIR=$LOCAL_E2E_RUNTIME_DIR/credentials/trust
CREDENTIALS_DIRECTORY=$LOCAL_E2E_RUNTIME_DIR/credentials
EOF
chmod 0600 "$LOCAL_E2E_CONFIG_DIR/trust-loader.env"

# The wizard uses every registered factory named by services.startup. It writes
# a shared bootstrap tree; project-local-runtime.sh later scopes it per process.
XDG_STATE_HOME="$LOCAL_E2E_STATE_DIR/xdg-state" \
XDG_DATA_HOME="$LOCAL_E2E_STATE_DIR/xdg-data" \
  "$LOCAL_E2E_BINARY" --config "$LOCAL_E2E_CONFIG_DIR/pds.toml" \
    wizard --non-interactive --bootstrap-only

local_e2e_log "rendered exact configs and bootstrapped service credentials"
