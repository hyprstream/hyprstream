#!/usr/bin/env bash
# Address helpers for spike scripts. Sourced.

# The first node's address as reachable from the host. Under KIND-in-Podman the
# node container sits on the podman "kind" bridge; the host can route to its IP.
# We prefer the address metallb/kube-proxy would advertise; fall back to the
# control-plane container IP.
node_external_ip() {
  local ip
  ip="$(kubectl --kubeconfig "$KUBECONFIG" get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)"
  [[ -n "$ip" ]] && { printf '%s\n' "$ip"; return 0; }
  ip="$(podman inspect "${CLUSTER_NAME:-hyprstream}-control-plane" 2>/dev/null \
        | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["NetworkSettings"]["Networks"]["kind"]["IPAddress"])' 2>/dev/null)" || true
  [[ -n "$ip" ]] && { printf '%s\n' "$ip"; return 0; }
  echo "127.0.0.1"   # NodePort host-mapping fallback (kind-config maps hostPort)
}
