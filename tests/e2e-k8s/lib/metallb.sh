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
# metallb is OPTIONAL: LoadBalancer cases can fall back to a stable
# NodePort (up.sh maps 30443 host<->node), and #458 only needs a ClusterIP
# service + endpoint flapping, no LB at all. So a failure here is downgraded to
# a warning by up.sh.
#
# Sourced by up.sh. Expects: KUBECONFIG set, lib/common.sh sourced.

install_metallb() {
  command -v python3 >/dev/null 2>&1 || { warn "python3 is required to discover a safe metallb address pool"; return 1; }
  local v="5765ee504d21a3a237d9dab223ca707661269ecd"   # v0.14.9 commit; tag is intentionally not used at runtime.
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

  local start end
  start="${pool%%-*}"
  end="${pool##*-}"

  if ! kubectl --kubeconfig "$KUBECONFIG" apply -f - <<EOF
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
  then
    warn "failed to configure the metallb address pool"
    return 1
  fi
  log "metallb ready; LoadBalancer Services will be assigned from ${start}-${end}"
}

# Discover the podman "kind" bridge subnet. Prefer the bridge interface IP the
# node containers share; fall back to scanning the podman network JSON.
discover_pool() {
  # The podman "kind" network gateway is the LB anchor subnet.
  podman network inspect kind 2>/dev/null \
    | python3 -c '
import ipaddress
import json
import sys

def walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk(v)
    elif isinstance(value, str):
        yield value

data = json.load(sys.stdin)[0]
sub = data["subnets"][0]
net = ipaddress.ip_network(sub["subnet"], strict=False)
reserved = {net.network_address, net.broadcast_address}
if sub.get("gateway"):
    reserved.add(ipaddress.ip_address(sub["gateway"]))
for value in walk(data.get("containers", {})):
    try:
        ip = ipaddress.ip_interface(value).ip
    except ValueError:
        continue
    if ip in net:
        reserved.add(ip)

free = [ip for ip in net.hosts() if ip not in reserved]
needed = min(32, len(free))
for idx in range(len(free) - needed, -1, -1):
    run = free[idx:idx + needed]
    if int(run[-1]) - int(run[0]) == len(run) - 1:
        print(f"{run[0]}-{run[-1]}")
        raise SystemExit(0)
raise SystemExit(1)
' 2>/dev/null && return 0

  # Fallback: a node container's IP on the kind network.
  local nodeip
  nodeip="$(podman inspect hyprstream-control-plane 2>/dev/null \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["NetworkSettings"]["Networks"]["kind"]["IPAddress"])' 2>/dev/null)" || true
  if [[ -n "$nodeip" ]]; then
    python3 - "$nodeip" <<'PY'
import ipaddress
import sys

node = ipaddress.ip_address(sys.argv[1])
net = ipaddress.ip_network(f"{node}/24", strict=False)
reserved = {net.network_address, net.broadcast_address, node}
reserved.add(ipaddress.ip_address(f"{node.exploded.rsplit('.', 1)[0]}.1"))
free = [ip for ip in net.hosts() if ip not in reserved]
needed = min(32, len(free))
for idx in range(len(free) - needed, -1, -1):
    run = free[idx:idx + needed]
    if int(run[-1]) - int(run[0]) == len(run) - 1:
        print(f"{run[0]}-{run[-1]}")
        raise SystemExit(0)
raise SystemExit(1)
PY
    return $?
  fi
  return 1
}
