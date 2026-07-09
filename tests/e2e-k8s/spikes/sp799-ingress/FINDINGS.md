# Spike #799 — cert-pinned QUIC/WebTransport across the pod→LB boundary (SP3)

Parent epic #778. Gates the real value of #792 (K6a). Run via `./run.sh`
(`EXPOSE=nodeport|loadbalancer`, `STANDIN_IMAGE=...`).

## Premise under test (SP3)

The moq stream plane advertises a producer's reach as `StreamInfo.reach` — a
bound socket address + TLS SNI + pinned leaf-cert SHA-256 hashes
(`NodeStreamReach`, `moq_stream.rs`). Inside Kubernetes the pod's bound address
is cluster-internal; for an external subscriber to dial it, the advertised reach
must describe the **externally reachable identity** or the cert-pin check fails.
SP3 asks: does that advertisement survive the pod → Service/LoadBalancer →
external-subscriber path?

## What was validated on the dev box (this branch)

Cluster bring-up is **partly blocked** by the host (see `../../FINDINGS.md`):
single-node KIND-in-Podman works; multi-node (workers) fails because the host's
`fs.inotify.max_user_instances=128` exhausts the per-user inotify budget once the
control-plane kubelet/containerd/cAdvisor consume theirs, so worker kubelets
crash-loop with `inotify_init: too many open files`. One `sudo sysctl` fixes it.

What we **could** validate on the single-node control-plane (real signal):

- ✅ **NodePort cross-boundary reachability (TCP).** A pod behind a `NodePort`
  service is reachable from the host through the kind-config-mapped host port
  (`30080` in the validation run; `30443` in the committed config). This is the
  SP3 *plumbing premise*: the pod→Service→host boundary is traversable for an
  external subscriber. The echo pod's body returned correctly through the mapped
  port. `kubectl port-forward` (the TCP cert-fetch path) likewise works.
- ⚠️ **UDP / QUIC through NodePort: not yet exercised.** The plumbing for UDP
  NodePort is configured (`30443/udp` mapped host↔node), but exercising an actual
  QUIC handshake requires the stand-in image + the external subscriber client
  (both `DESIGN-READY`, execution pending the stand-in build). TCP success +
  kind's documented UDP NodePort support make UDP reach highly likely but
  **unproven on this host today**.

## What remains (DESIGN-READY, runnable once host + stand-in are ready)

`run.sh` is the complete deploy→assert→teardown flow. The pieces it needs:

1. **QUIC stand-in image** (`STANDIN_IMAGE`): a small server exposing a QUIC /
   WebTransport listener on `:443/udp` with a known leaf cert, plus an HTTP
   `/cert` endpoint returning that cert PEM. The natural build target is the
   hyprstream streaming binary itself with a minimal config (preferred — tests
   the real `ProducerReachConfig` path), or a `quinn`/`web_transport_quinn` echo.
2. **External subscriber client** that dials the advertised reach and verifies
   the leaf cert against the advertised pin (`quic-standin-client`,
   `iroh-subscriber-standin` placeholders in `run.sh`).
3. **Multi-node** (or accept single-node for the LB path). For the LoadBalancer
   assertion metallb is layered on; for cross-node rerouting see #458.

Once those land, `run.sh` exercises both reach paths the issue names:

- **(A) direct QUIC** over a UDP LoadBalancer (or NodePort fallback), with
  `ProducerReachConfig.quic_reach` populated from the **Service's** external
  address/cert, not the pod's.
- **(B) iroh-direct** with no LB at all — does the subscriber dial via iroh
  relay/NAT-traversal from inside the pod's netns out?

## Code-vs-wiring question (the issue's open scope item)

"If direct QUIC requires new code: scope it precisely (e.g. a
`ProducerReachConfig` field for an externally-configured addr/cert distinct from
the bind address)."

**Likely answer: wiring, not new code — but a config seam to confirm.** The
moq plane already separates `QuicConfig.bind_addr` (the socket the producer
listens on = the pod IP) from `ProducerReachConfig`'s advertised reach. The spike
needs the chart/operator (#792/K6a) to populate the *advertised* reach from the
Service's external address + the mounted/SPIFFE-issued leaf cert, leaving
`bind_addr` as the pod-local listener. If `ProducerReachConfig` does not already
accept an externally-overridden addr/cert pair **distinct** from bind, that is
the one small code site to add (a reach-override field) — `run.sh`'s
`REACH_ADDR` derivation is the place that would consume it. Confirm against
`moq_stream.rs` before #792's chart defaults to either path.

## Provisional SP3 conclusion (to confirm once the stand-in runs)

Given (1) the plumbing is confirmed traversable, (2) Gateway API/HTTPRoute has no
UDP/QUIC story, and (3) the issue's own note that iroh-direct "validates the
identity-over-location bet in the exact scenario it was designed for" — the
working hypothesis is:

- **Default #792/K6a reach mode to iroh-direct** for in-cluster producers
  reaching external subscribers (zero Service/Ingress surface; pods egress
  freely; cert/identity rides in the iroh NodeId).
- **Offer direct-QUIC-over-UDP-LB as the opt-in** for deployments that want
  deterministic addressing / can't rely on iroh relays, wired via the
  reach-override seam above.

This matches the issue's framing and avoids new UDP-gateway infra. The spike's
job once runnable is to **prove the iroh path works pod→out** and quantify the
direct-QUIC path's reach-override cost. Both are gated on the host sysctl fix +
the stand-in build, tracked here.
