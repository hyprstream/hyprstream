#!/usr/bin/env bash
# metallb wiring for the KIND-in-Podman harness.
#
# KIND's stock Docker provider ships a metallb helper (`kind load` + the
# metalLB manifests), but under the podman provider the documented address-pool
# discovery (the docker bridge IP) does not apply. Here we:
#
#   1. Discover the podman "kind" bridge subnet (the network the node containers
#      are attached to) and carve the top of the second octet range for LB VIPs.
#   2. Apply metallb native manifests (CRDs + controller + the IPAddressPool +
#      L2Advertisement) pointing at that pool.
#
# metallb is OPTIONAL for the spikes: #799 is fully runnable over a stable
# NodePort (up.sh maps 30443 host<->node), and #458 only needs a ClusterIP
# service + endpoint flapping, no LB at all. So a failure here is downgraded to
# a warning by up.sh.
#
# Sourced by up.sh. Expects: KUBECONFIG set, lib/common.sh sourced.

install_metallb() {
  local v="v0.14.9"   # metallb native; works on kind's Kubernetes (1.30-ish)
  local manifests=(
    "https://raw.githubusercontent.com/metallb/metallb/${v}/config/manifests/metallb-native.yaml"
  )
  log "applying metallb ${v}"
  local m
  for m in "${manifests[@]}"; do
    kubectl --kubeconfig "$KUBECONFIG" apply -f "$m" >/dev/null 2>&1 || {
      warn "failed to apply $m"
      return 1
    }
  done

  # metallb memberlist uses a secret for speaker auth; the native manifest sets a
  # default. Wait for the controller to roll.
  kubectl --kubeconfig "$KUBECONFIG" -n metallb-system rollout status \
    deployment/controller --timeout=120s >/dev/null 2>&1 || {
    warn "metallb controller did not roll out in 120s"
    return 1
  }

  local pool
  pool="$(discover_pool)" || { warn "could not discover kind podman subnet"; return 1; }
  log "metallb address pool: ${pool}"

  # First address = pool start, derive end by bumping the last octet to .250
  # (sufficient for a handful of LoadBalancers; the pool is /16-ish so this is
  # conservative). The sed keeps this robust to a /16 or /24 podman bridge.
  local start end
  start="${pool%%-*}"
  end="$(echo "${start}" | awk -F. '{printf "%s.%s.%s.250", $1,$2,$3}')"

  kubectl --kubeconfig "$KUBECONFIG" apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: kind-podman
  namespace: metallb-system
spec:
  addresses:
  - ${start}-${end}
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: kind-podman
  namespace: metallb-system
spec:
  ipAddressPools:
  - kind-podman
EOF
  log "metallb ready; LoadBalancer Services will be assigned from ${start}-${end}"
}

# Discover the podman "kind" bridge subnet. Prefer the bridge interface IP the
# node containers share; fall back to scanning the podman network JSON.
discover_pool() {
  # The podman "kind" network gateway is the LB anchor subnet.
  podman network inspect kind 2>/dev/null \
    | python3 -c '
import sys, json
data = json.load(sys.stdin)
sub = data[0]["subnets"][0]
cidr = sub["subnet"]
# carve addresses out of the subnet, leaving the gateway alone.
print(cidr.split("/")[0] + "-" + cidr.rsplit(".",1)[0] + ".250")
' 2>/dev/null && return 0

  # Fallback: a node container's IP on the kind network.
  local nodeip
  nodeip="$(podman inspect hyprstream-control-plane 2>/dev/null \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["NetworkSettings"]["Networks"]["kind"]["IPAddress"])' 2>/dev/null)" || true
  if [[ -n "$nodeip" ]]; then
    echo "${nodeip}-${nodeip%.*}.250"
    return 0
  fi
  return 1
}
