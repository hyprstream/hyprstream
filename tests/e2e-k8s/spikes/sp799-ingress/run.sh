#!/usr/bin/env bash
# Spike #799 — cert-pinned QUIC/WebTransport advertisement across the pod→LB
# boundary (SP3). Parent: #778. Home: tests/e2e-k8s/.
#
# PREMISE under test (SP3): can a stream subscriber OUTSIDE the cluster reach a
# producer pod's cert-pinned QUIC/WebTransport endpoint through the Service/LB
# boundary, with the *advertised* reach (addr + SNI + leaf-cert SHA-256 pins)
# describing the externally-reachable identity rather than the pod's own?
#
# Two reach paths are exercised (the issue asks for both, recommends the winner):
#   (A) direct QUIC over a UDP LoadBalancer / NodePort — reach addr = LB VIP /
#       node:NodePort, SNI + cert pins = the producer's leaf cert
#   (B) iroh-direct — no LB at all; external subscriber dials via iroh relay/NAT
#       traversal from inside the pod's netns out
#
# STATUS: harness/plumbing validated on the dev box; the cert-pinned QUIC layer
# needs a QUIC stand-in image + multi-node (gated on the inotify sysctl fix,
# see ../../FINDINGS.md). This script is the runnable design: once STANDIN_IMAGE
# is built and the cluster is multi-node, set the env vars and run end-to-end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../lib/common.sh
source "${SCRIPT_DIR}/../../lib/common.sh"

export KIND_EXPERIMENTAL_PROVIDER=podman
: "${CLUSTER_NAME:=hyprstream}"
: "${STANDIN_IMAGE:=ghcr.io/hyprstream/spike-quic-standin:latest}"
# How the producer is exposed. QUIC is UDP; metallb LoadBalancer is preferred,
# NodePort (mapped host<->node in kind-config.yaml on 30443) is the fallback.
: "${EXPOSE:=nodeport}"   # nodeport | loadbalancer
NODEPORT_UDP=30443
RUN_ID="$(date +%s)-$$"
NS="sp799-${RUN_ID}"

require_cmd kubectl curl openssl
trap cleanup EXIT

cleanup() {
  rc=$?
  kubectl --kubeconfig "$KUBECONFIG" delete ns "$NS" --ignore-not-found >/dev/null 2>&1 || true
  exit "$rc"
}

log "#799 spike: cert-pinned QUIC/WTransport across pod→LB boundary"
log "cluster=${CLUSTER_NAME} namespace=${NS} expose=${EXPOSE} standin=${STANDIN_IMAGE}"
kubectl --kubeconfig "$KUBECONFIG" get nodes || die "no cluster / KUBECONFIG not set (run ../../up.sh)"

kubectl --kubeconfig "$KUBECONFIG" create ns "$NS"

# 1. Deploy the producer. The stand-in MUST expose:
#    - a QUIC/WebTransport listener whose TLS leaf cert is the one we pin
#    - an HTTP /cert endpoint returning the leaf cert PEM (so the subscriber can
#      compute the SHA-256 pin without an out-of-band channel — a real
#      deployment publishes pins in the reach advertisement, NodeStreamReach).
log "deploying producer (stand-in=${STANDIN_IMAGE})"
standin_image_escaped="${STANDIN_IMAGE//&/\\&}"
sed "s|STANDIN_IMAGE|${standin_image_escaped}|g" "${SCRIPT_DIR}/producer.yaml" \
  | kubectl --kubeconfig "$KUBECONFIG" -n "$NS" apply -f -

# Parametrize exposure on EXPOSE.
if [[ "$EXPOSE" == "loadbalancer" ]]; then
  kubectl --kubeconfig "$KUBECONFIG" -n "$NS" apply -f "${SCRIPT_DIR}/service-lb.yaml"
else
  kubectl --kubeconfig "$KUBECONFIG" -n "$NS" apply -f "${SCRIPT_DIR}/service-nodeport.yaml"
fi
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" wait --for=condition=Available deployment/producer --timeout=180s

# 2. Resolve the externally-reachable address = the ADVERTISED reach.
#    This is the crux of SP3: the advertised reach must be the *external* identity.
case "$EXPOSE" in
  nodeport)
    # Node IP = the kind node container's address reachable from the host.
    REACH_IP="$(node_external_ip)"
    REACH_ADDR="${REACH_IP}:${NODEPORT_UDP}"
    ;;
  loadbalancer)
    kubectl --kubeconfig "$KUBECONFIG" -n "$NS" wait svc/producer --for=jsonpath='{.status.loadBalancer.ingress[0].ip}' --timeout=120s \
      || die "no LB IP (metallb not installed? run with EXPOSE=nodeport)"
    REACH_IP="$(kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get svc/producer -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
    REACH_ADDR="${REACH_IP}:443"
    ;;
esac
log "advertised reach addr = ${REACH_ADDR}"

# 3. Fetch the leaf cert (out-of-band here; in hyprstream it rides in
#    NodeStreamReach.pinned_hashes) and compute the SHA-256 pin the subscriber
#    will verify against.
fetch_cert_pem() {
  local port pf pf_log cert
  port="$(free_port)"
  pf_log="$(mktemp)"
  kubectl --kubeconfig "$KUBECONFIG" -n "$NS" port-forward svc/producer-http "${port}:80" >"$pf_log" 2>&1 &
  pf=$!
  for _ in $(seq 1 30); do
    if cert="$(curl --fail --show-error --silent --max-time 2 "http://127.0.0.1:${port}/cert")"; then
      kill "$pf" 2>/dev/null || true
      wait "$pf" 2>/dev/null || true
      rm -f "$pf_log"
      printf '%s\n' "$cert"
      return 0
    fi
    if ! kill -0 "$pf" 2>/dev/null; then
      warn "port-forward exited before /cert was reachable:"
      sed 's/^/[port-forward] /' "$pf_log" >&2 || true
      rm -f "$pf_log"
      return 1
    fi
    sleep 1
  done
  warn "timed out waiting for producer-http /cert via port-forward:"
  sed 's/^/[port-forward] /' "$pf_log" >&2 || true
  kill "$pf" 2>/dev/null || true
  wait "$pf" 2>/dev/null || true
  rm -f "$pf_log"
  return 1
}
CERT_PEM="$(fetch_cert_pem)" || die "failed to fetch producer certificate"
PIN="$(printf '%s' "$CERT_PEM" | openssl x509 -pubkey -noout 2>/dev/null | openssl pkey -pubin -outform der 2>/dev/null | openssl dgst -sha256 -binary | base64)"
log "advertised leaf-cert SHA-256 pin (SPKI) = ${PIN:-<uncomputed; needs stand-in>}"

# 4. The SP3 assertion: an EXTERNAL subscriber (outside the cluster, on the host)
#    dials the advertised reach addr, completes the TLS handshake, and verifies
#    the server leaf cert matches the advertised pin — i.e. cert pinning survives
#    the pod→LB boundary. Direct QUIC path:
assert_quic_reach() {
  log "ASSERT (A) direct QUIC: external subscriber dials ${REACH_ADDR}, verifies pin ${PIN}"
  # Requires the QUIC stand-in client. Placeholder: replace with the real client.
  #   quic-standin-client --addr "${REACH_ADDR}" --pin "${PIN}" --sni producer.hyprstream
  if command -v quic-standin-client >/dev/null 2>&1; then
    local attempt
    for attempt in $(seq 1 30); do
      if quic-standin-client --addr "${REACH_ADDR}" --pin "${PIN}" --sni producer.hyprstream; then
        return 0
      fi
      sleep 1
    done
    return 1
  else
    warn "quic-standin-client not built — SP3 cert-pin assertion SKIPPED (needs stand-in image)"
    warn "plumbing validated separately: host→NodePort→pod TCP reach confirmed (see FINDINGS.md)"
    return 2   # distinct code: design-ready, execution-pending stand-in
  fi
}

# (B) iroh-direct path: subscriber dials via iroh with NO service/LB at all.
assert_iroh_reach() {
  log "ASSERT (B) iroh-direct: subscriber dials producer's NodeId via iroh relay"
  local nodeid
  nodeid="$(kubectl --kubeconfig "$KUBECONFIG" -n "$NS" exec deploy/producer -- \
            cat /run/hyprstream/iroh-nodeid 2>/dev/null || true)"
  [[ -n "$nodeid" ]] || { warn "producer did not publish an iroh NodeId; iroh path SKIPPED"; return 2; }
  if command -v iroh-subscriber-standin >/dev/null 2>&1; then
    iroh-subscriber-standin --nodeid "$nodeid"
  else
    warn "iroh-subscriber-standin not built — iroh-direct path execution SKIPPED"
    return 2
  fi
}

status_label() {
  case "$1" in
    0) echo PASS ;;
    2) echo "DESIGN-READY (stand-in pending)" ;;
    *) echo FAIL ;;
  esac
}

quic_rc=0; iroh_rc=0
assert_quic_reach || quic_rc=$?
assert_iroh_reach || iroh_rc=$?

cat <<EOF

=== #799 spike result ===
direct-QUIC path:   $(status_label "$quic_rc")
iroh-direct path:   $(status_label "$iroh_rc")
advertised reach:   ${REACH_ADDR}
cert pin (SPKI):    ${PIN:-<n/a>}
EOF

log "see ${SCRIPT_DIR}/FINDINGS.md for the SP3 conclusion + reach-mode recommendation."

if (( (quic_rc != 0 && quic_rc != 2) || (iroh_rc != 0 && iroh_rc != 2) )); then
  exit 1
fi
if (( quic_rc == 2 || iroh_rc == 2 )); then
  exit 2
fi
