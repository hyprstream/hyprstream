#!/usr/bin/env bash
# Materialize the suffix-identical rootless adapter for Metal #37.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

mode=${1:-credentials}
[[ "$mode" == "credentials" || "$mode" == "links" || "$mode" == "check" ]] ||
  local_e2e_die "usage: $0 {credentials|links|check}"

assert_tree_file() {
  local path=$1
  local_e2e_assert_regular_file "$path"
  [[ -r "$path" ]] || local_e2e_die "projected file is not readable: $path"
}

project_service_credentials() {
  local destination=$1
  local service=$2
  local bootstrap="$LOCAL_E2E_STATE_DIR/bootstrap-credentials"
  local source_dir="$bootstrap/$service"
  local node_name

  local_e2e_assert_real_dir "$bootstrap"
  local_e2e_assert_real_dir "$source_dir"
  install -d -m 0700 "$destination"
  if [[ "$service" == "policy" ]]; then
    assert_tree_file "$bootstrap/signing-key"
    install -m 0600 "$bootstrap/signing-key" "$destination/signing-key"
  else
    assert_tree_file "$source_dir/signing-key"
    install -m 0600 "$source_dir/signing-key" "$destination/signing-key"
  fi
  assert_tree_file "$source_dir/service-jwt"
  install -m 0600 "$source_dir/service-jwt" "$destination/service-jwt"

  for node_name in ca-pubkey bootstrap-pubkeys rsa-key tls-key tls-cert quic-key quic-cert; do
    if [[ -f "$bootstrap/$node_name" && ! -L "$bootstrap/$node_name" ]]; then
      install -m 0600 "$bootstrap/$node_name" "$destination/$node_name"
    fi
  done
  if [[ "$service" == "policy" ]]; then
    assert_tree_file "$bootstrap/ca-key"
    install -m 0600 "$bootstrap/ca-key" "$destination/ca-key"
  fi
}

project_credentials() {
  local bootstrap="$LOCAL_E2E_STATE_DIR/bootstrap-credentials"
  local trust="$LOCAL_E2E_STATE_DIR/trust"
  local tls="$LOCAL_E2E_STATE_DIR/tls"
  local staging
  local service

  if [[ -e "$LOCAL_E2E_RUNTIME_DIR" ]]; then
    local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR"
    if [[ -n "$(find "$LOCAL_E2E_RUNTIME_DIR" -mindepth 1 -print -quit)" ]]; then
      local_e2e_die \
        "shared runtime is not empty; refusing to collide with an existing Hyprstream fabric"
    fi
  fi
  install -d -m 0700 "$LOCAL_E2E_RUNTIME_DIR"
  local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR"

  for path in \
    "$trust/deployment-ca.hybrid" \
    "$trust/deployment-authority.log.json" \
    "$trust/deployment-authority.head.json" \
    "$trust/registry-service.jwt" \
    "$tls/quic-chain.pem" \
    "$tls/quic-key.pem"; do
    assert_tree_file "$path"
  done
  local_e2e_assert_real_dir "$bootstrap"
  local_e2e_assert_real_dir "$trust"
  local_e2e_assert_real_dir "$tls"
  [[ "$(stat -c %s -- "$trust/deployment-ca.hybrid")" == "1984" ]] ||
    local_e2e_die "deployment public CA is not 1984 bytes"

  staging="$(mktemp -d "$LOCAL_E2E_RUNTIME_DIR/.credentials.stage.XXXXXX")"
  chmod 0700 "$staging"
  trap '[[ -n ${staging:-} && -d ${staging:-} ]] && rm -r -- "$staging"' EXIT

  install -d -m 0700 "$staging/tls" "$staging/trust"
  install -m 0600 "$tls/quic-chain.pem" "$staging/tls/quic-chain.pem"
  install -m 0600 "$tls/quic-key.pem" "$staging/tls/quic-key.pem"
  install -m 0600 "$trust/deployment-ca.hybrid" "$staging/trust/deployment-ca.hybrid"
  install -m 0600 "$trust/deployment-authority.log.json" \
    "$staging/trust/deployment-authority.log.json"
  install -m 0600 "$trust/deployment-authority.head.json" \
    "$staging/trust/deployment-authority.head.json"
  install -m 0600 "$trust/registry-service.jwt" "$staging/registry-service.jwt"

  for service in policy discovery registry oauth; do
    project_service_credentials "$staging/$service" "$service"
  done

  # Hyprstream bootstrap currently mints one `service:inference` identity.
  # #37 assigns that identity to two isolated processes. Copying into two
  # distinct directories preserves the exact identity while preventing any
  # ambient or cross-replica directory fallback.
  project_service_credentials "$staging/inference-cpu-0" inference
  project_service_credentials "$staging/inference-cpu-1" inference

  find "$staging" -type d -exec chmod 0700 {} +
  find "$staging" -type f -exec chmod 0600 {} +
  if [[ -n "$(find "$staging" -type l -print -quit)" ]]; then
    local_e2e_die "credential staging tree contains a symlink"
  fi

  mv -- "$staging" "$LOCAL_E2E_RUNTIME_DIR/credentials"
  printf '%s\n' "$LOCAL_E2E_DIR" >"$LOCAL_E2E_RUNTIME_DIR/local-e2e.owner"
  chmod 0600 "$LOCAL_E2E_RUNTIME_DIR/local-e2e.owner"
  staging=
  trap - EXIT
  local_e2e_log "atomic runtime credential projection installed"
}

project_links() {
  local replica
  local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR"
  local_e2e_assert_socket "$LOCAL_E2E_RUNTIME_DIR/policy.sock"
  local_e2e_assert_socket "$LOCAL_E2E_RUNTIME_DIR/discovery.sock"

  for replica in 0 1; do
    instance_dir="$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica"
    install -d -m 0700 "$instance_dir"
    local_e2e_assert_real_dir "$instance_dir"
    local_e2e_safe_link "$instance_dir/policy.sock" ../policy.sock
    local_e2e_safe_link "$instance_dir/discovery.sock" ../discovery.sock
    local_e2e_safe_link \
      "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica.sock" \
      "inference-cpu-$replica/inference.sock"
  done
  local_e2e_log "rootless socket projection matches Metal #37 suffix contract"
}

check_projection() {
  local replica path
  local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR"
  local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR/credentials"
  local_e2e_assert_socket "$LOCAL_E2E_RUNTIME_DIR/policy.sock"
  local_e2e_assert_socket "$LOCAL_E2E_RUNTIME_DIR/discovery.sock"
  for replica in 0 1; do
    local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica"
    local_e2e_assert_real_dir "$LOCAL_E2E_RUNTIME_DIR/credentials/inference-cpu-$replica"
    for path in signing-key service-jwt; do
      assert_tree_file \
        "$LOCAL_E2E_RUNTIME_DIR/credentials/inference-cpu-$replica/$path"
    done
    [[ "$(readlink -- "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica/policy.sock")" == "../policy.sock" ]] ||
      local_e2e_die "unexpected policy socket link for replica $replica"
    [[ "$(readlink -- "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica/discovery.sock")" == "../discovery.sock" ]] ||
      local_e2e_die "unexpected discovery socket link for replica $replica"
    [[ "$(readlink -- "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica.sock")" == \
      "inference-cpu-$replica/inference.sock" ]] ||
      local_e2e_die "unexpected health alias for replica $replica"
    local_e2e_assert_socket \
      "$LOCAL_E2E_RUNTIME_DIR/inference-cpu-$replica/inference.sock"
  done
}

case "$mode" in
  credentials) project_credentials ;;
  links) project_links ;;
  check) check_projection ;;
esac
