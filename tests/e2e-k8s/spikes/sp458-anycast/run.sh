#!/usr/bin/env bash
# Spike #458 — anycast flow-stability (transport core). Parent: #358 (S3 relay),
# part of #355. Home: tests/e2e-k8s/.
#
# PREMISE under test: QUIC (and thus moq-over-QUIC / WebTransport) binds a
# connection to its network path/4-tuple. Anycast can re-route mid-connection to
# a different relay instance, breaking the live QUIC connection. We emulate the
# *failure mode* — a mid-flow backend switch — WITHOUT real BGP (real BGP
# anycast convergence timing is explicitly out of scope, needs real infra; see
# #975 scope boundary) and OBSERVE whether the flow breaks or migrates/recovers.
#
# METHOD (emulated backend flap, no BGP):
#   1. N relay pods (N=3) behind ONE ClusterIP Service. The Service VIP is the
#      "anycast" address — kube-proxy load-balances (flow-hash) across the N pods.
#   2. A client establishes a QUIC/moq/WebTransport flow to the Service VIP.
#   3. FORCE the flap: delete the serving pod so the Service reroutes to a
#      different backing pod (emulates anycast reconverging to another instance).
#      (Alternative flap: rewrite the Endpoints directly, or swap nftables DNAT —
#      `--flap=delete|endpoints`.)
#   4. OBSERVE the live flow: does the connection drop (4-tuple broken) or does
#      QUIC connection migration / moq recovery keep it alive?
#
# STATUS: multi-node is gated on the inotify sysctl fix (../../FINDINGS.md). On a
# single node the 3 relays co-locate, which still exercises the Service-level
# reroute (the flap mechanism) but NOT cross-node path changes — the transport
# conclusion is the same since the QUIC-4-tuple question is path-agnostic.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../lib/common.sh
source "${SCRIPT_DIR}/../../lib/common.sh"
# shellcheck source=../../lib/addr.sh
source "${SCRIPT_DIR}/../../lib/addr.sh"

export KIND_EXPERIMENTAL_PROVIDER=podman
: "${CLUSTER_NAME:=hyprstream}"
: "${RELAY_IMAGE:=ghcr.io/hyprstream/spike-quic-relay:latest}"
: "${REPLICAS:=3}"
: "${FLAP:=delete}"        # delete | endpoints
: "${FLOW:=quic}"          # quic | moq | webtransport

require_cmd kubectl
trap cleanup EXIT

cleanup() {
  rc=$?
  kubectl --kubeconfig "$KUBECONFIG" -n sp458 delete deployment,service,pod --all --ignore-not-found >/dev/null 2>&1 || true
  kubectl --kubeconfig "$KUBECONFIG" delete ns sp458 --ignore-not-found >/dev/null 2>&1 || true
  exit "$rc"
}

log "#458 spike: anycast flow-stability (emulated backend flap, no BGP)"
log "cluster=${CLUSTER_NAME} flow=${FLOW} replicas=${REPLICAS} flap=${FLAP} relay=${RELAY_IMAGE}"
kubectl --kubeconfig "$KUBECONFIG" get nodes || die "no cluster / KUBECONFIG not set (run ../../up.sh)"

kubectl --kubeconfig "$KUBECONFIG" create ns sp458 --dry-run=client -o yaml | kubectl apply -f -

# 1. N relays behind ONE ClusterIP. Each relay prints a stable instance id on
#    connect so the client can tell WHICH backend it is pinned to.
log "deploying ${REPLICAS} relay pods behind one ClusterIP (the 'anycast' VIP)"
sed "s/__REPLICAS__/${REPLICAS}/; s|__IMAGE__|${RELAY_IMAGE}|" "${SCRIPT_DIR}/relay.yaml" \
  | kubectl --kubeconfig "$KUBECONFIG" -n sp458 apply -f -
kubectl --kubeconfig "$KUBECONFIG" -n sp458 apply -f "${SCRIPT_DIR}/service.yaml"
kubectl --kubeconfig "$KUBECONFIG" -n sp458 wait --for=condition=Available deployment/relay --timeout=180s
VIP="$(kubectl --kubeconfig "$KUBECONFIG" -n sp458 get svc relay -o jsonpath='{.spec.clusterIP}')"
log "anycast VIP = ${VIP}"

# 2. Establish the flow from a client pod, record which backend instance served.
log "establishing ${FLOW} flow to ${VIP} from a client pod"
CLIENT_POD=sp458-client
kubectl --kubeconfig "$KUBECONFIG" -n sp458 run "${CLIENT_POD}" --rm -i --restart=Never \
  --image="${RELAY_IMAGE}" --command -- \
  sh -c 'echo "client-ready"; sleep 3600' >/dev/null 2>&1 &
# (In the real stand-in this is: quic-client --addr "$VIP" --hold / observe stream)

# Demonstrate the flap MECHANISM (Service reroute on backing-pod removal) without
# the QUIC stand-in. This proves the control-plane reroute that emulates anycast
# reconvergence; the transport-level flow question needs the stand-in below.
demonstrate_flap_mechanism() {
  local vip="$1"
  log "MECHANISM DEMO: which backings does the Service VIP route to, before/after a pod flap?"
  local before after
  before="$(kubectl --kubeconfig "$KUBECONFIG" -n sp458 get endpoints relay \
            -o jsonpath='{.subsets[0].addresses[*].targetRef.name}' 2>/dev/null | tr ' ' ',')"
  local victim
  victim="$(kubectl --kubeconfig "$KUBECONFIG" -n sp458 get pods -l app=relay -o jsonpath='{.items[0].metadata.name}')"
  log "  endpoints before: ${before:-<none>}"
  log "  evicting ${victim} (force Service reroute)..."
  kubectl --kubeconfig "$KUBECONFIG" -n sp458 delete pod "${victim}" --grace-period=0 --force >/dev/null 2>&1 || true
  # Endpoint controller + ReplicaSet reconcile: wait for a replacement, then re-read.
  kubectl --kubeconfig "$KUBECONFIG" -n sp458 wait --for=condition=Available deployment/relay --timeout=120s >/dev/null 2>&1 || true
  after="$(kubectl --kubeconfig "$KUBECONFIG" -n sp458 get endpoints relay \
           -o jsonpath='{.subsets[0].addresses[*].targetRef.name}' 2>/dev/null | tr ' ' ',')"
  log "  endpoints after:  ${after:-<none>}"
  if [[ "$before" != "$after" ]]; then
    log "  => Service VIP rerouted to a different backing set on flap (anycast-reconvergence emulation HOLDS at the control plane)"
  else
    warn "  => backing set unchanged after flap (endpoint reconcile race? retry)"
  fi
}

if ! command -v quic-flow-probe >/dev/null 2>&1; then
  warn "quic-flow-probe stand-in not built — flow establishment + flap observation SKIPPED"
  warn "the flap MECHANISM (Service reroute on pod delete) is still demonstrable:"
  demonstrate_flap_mechanism "$VIP"
  cat <<EOF

=== #458 spike result (mechanism only — transport pending stand-in) ===
flow type:         ${FLOW} (stand-in not built)
anycast VIP:       ${VIP}
flap method:       ${FLAP}
flow survived:     PENDING quic-flow-probe stand-in
EOF
  exit 2   # design-ready, execution-pending stand-in
fi

BACKEND_BEFORE="$(quic-flow-probe establish --addr "${VIP}" --flow "${FLOW}")"
log "flow established, serving backend = ${BACKEND_BEFORE}"

# 3. FORCE the flap: evict the serving pod so the Service reroutes.
SERVING_POD="$(kubectl --kubeconfig "$KUBECONFIG" -n sp458 get pods -l app=relay \
  -o jsonpath='{.items[?(@.metadata.annotations.instance=="'"${BACKEND_BEFORE}"'")].metadata.name}' 2>/dev/null || true)"
log "flap (${FLAP}): forcing Service away from ${SERVING_POD:-serving pod}"
case "$FLAP" in
  delete)
    kubectl --kubeconfig "$KUBECONFIG" -n sp458 delete pod "${SERVING_POD}" --grace-period=0 --force >/dev/null 2>&1 || true
    ;;
  endpoints)
    # Rewrite the Endpoints to drop the serving pod — pure control-plane flap.
    kubectl --kubeconfig "$KUBECONFIG" -n sp458 patch endpoints relay --type=json \
      -p='[{"op":"remove","path":"/subsets/0/addresses/0"}]' >/dev/null 2>&1 || true
    ;;
esac

# 4. OBSERVE the live flow across the flap.
log "observing flow across flap..."
if quic-flow-probe alive --flow "${FLOW}"; then
  log "RESULT: flow SURVIVED the backend flap (QUIC connection migration / moq recovery held)"
  BACKEND_AFTER="$(quic-flow-probe backend --flow "${FLOW}")"
  log "  backend before=${BACKEND_BEFORE} after=${BACKEND_AFTER}"
  SURVIVED=1
else
  log "RESULT: flow BROKE on backend flap (4-tuple lost — anycast re-route kills live QUIC)"
  SURVIVED=0
fi

cat <<EOF

=== #458 spike result ===
flow type:         ${FLOW}
anycast VIP:       ${VIP}
flap method:       ${FLAP}
backend before:    ${BACKEND_BEFORE}
backend after:     ${BACKEND_AFTER:-<flow died>}
flow survived:     $([[ ${SURVIVED:-0} == 1 ]] && echo YES || echo NO)
EOF
log "see ${SCRIPT_DIR}/FINDINGS.md for the flow-stability conclusion + BGP-anycast scope note."
