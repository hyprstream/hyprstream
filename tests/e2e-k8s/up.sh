#!/usr/bin/env bash
# Bring up the hyprstream KIND-in-Podman e2e cluster.
#
# Idempotent-ish: if a cluster with the configured name already exists it is
# *reused* (not recreated) so re-running after a reboot is cheap. Pass
# `--recreate` to force a fresh cluster. Requires rootless podman + kind on PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

CLUSTER_NAME="${CLUSTER_NAME:-hyprstream}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/kind-config.yaml}"
RECREATE="${RECREATE:-0}"
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

log "KIND-in-Podman e2e harness — bringing up cluster '${CLUSTER_NAME}'"
require_cmd podman kind kubectl

# Rootless podman is the hard requirement; make it explicit for this process
# and any child (kind, kubectl via wrapper) regardless of shell config.
export KIND_EXPERIMENTAL_PROVIDER=podman
export PODMAN_USERNS=keep-id

preflight

# Reuse an existing, healthy cluster if present (cheap re-run path).
clusters="$(kind_get_clusters)" || die "failed to query kind clusters via the podman provider"
if grep -qx "${CLUSTER_NAME}" <<<"$clusters"; then
  if [[ "${RECREATE}" == "1" ]]; then
    log "found existing cluster '${CLUSTER_NAME}', recreating (--recreate)"
    kind_delete_cluster "${CLUSTER_NAME}"
  elif kube_ready "${CLUSTER_NAME}" 2>/dev/null; then
    log "cluster '${CLUSTER_NAME}' already up and healthy — reusing"
    write_kubeconfig "${CLUSTER_NAME}"
    log "done. kubeconfig written to ${KUBECONFIG}"
    exit 0
  else
    warn "cluster '${CLUSTER_NAME}' registered with kind but not reachable; recreating"
    kind_delete_cluster "${CLUSTER_NAME}"
  fi
fi

log "creating cluster from ${CONFIG}"
kind create cluster --name "${CLUSTER_NAME}" --config "${CONFIG}"

write_kubeconfig "${CLUSTER_NAME}"

log "waiting for control-plane + nodes Ready"
kubectl --kubeconfig "${KUBECONFIG}" wait \
  --for=condition=Ready nodes \
  --all --timeout=300s

# metallb gives us UDP-capable LoadBalancers for the QUIC/WebTransport spike
# (#799). It is optional; `lib/metallb.sh` falls back to NodePort messaging if
# metallb can't be installed.
log "wiring LoadBalancer support (metallb)"
# shellcheck source=lib/metallb.sh
source "${SCRIPT_DIR}/lib/metallb.sh"
install_metallb || warn "metallb install failed — NodePort fallback only (#799 still runnable via NodePort)"

log "cluster '${CLUSTER_NAME}' is up."
log "  KUBECONFIG=${KUBECONFIG}"
log "  kubectl --kubeconfig \"${KUBECONFIG}\" get nodes"
log "tear down with: ${SCRIPT_DIR}/down.sh"
