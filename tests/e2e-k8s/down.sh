#!/usr/bin/env bash
# Tear down the hyprstream KIND-in-Podman e2e cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

CLUSTER_NAME="${CLUSTER_NAME:-hyprstream}"
export KIND_EXPERIMENTAL_PROVIDER=podman

if kind_get_clusters | grep -qx "${CLUSTER_NAME}"; then
  log "deleting cluster '${CLUSTER_NAME}'"
  kind delete cluster --name "${CLUSTER_NAME}"
  log "done"
else
  log "no cluster '${CLUSTER_NAME}' registered with kind; nothing to delete"
fi
