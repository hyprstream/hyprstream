# Spike #458 — anycast flow-stability (transport core), emulated backend flap

Parent #358 (S3 relay), part of #355. Home: #975. Run via `./run.sh`
(`FLOW=quic|moq|webtransport`, `FLAP=delete|endpoints`, `REPLICAS=3`).

## Premise under test

QUIC (and moq-over-QUIC / WebTransport) binds a connection to its network
path / 4-tuple. Anycast can re-route mid-connection to a different relay
instance, breaking the live QUIC connection. S3's relay fabric (#358) is
anycast-capable + many-to-many; this spike decides whether anycast needs
flow-pinning infra or a discovery-anycast / session-unicast split before anycast
reach is advertised in production.

**Scope boundary (per #975):** real BGP anycast routing / convergence timing is
NOT tested here — it needs actual multi-POP anycast infra. The harness emulates
the *failure mode* (a mid-flow backend reroute) via Service endpoint/pod
deletion, which is what tests whether QUIC/moq/WT survives a flap. The
BGP-level convergence-time question is a real-infra follow-up.

## What was validated on the dev box (this branch)

Multi-node is **gated on the inotify sysctl fix** (`../../FINDINGS.md`). On a
single node the 3 relays co-locate, which still exercises the Service-level
reroute (the flap mechanism) but not cross-node path changes — acceptable, since
the QUIC-4-tuple question is path-agnostic.

The **flap mechanism** (control-plane reroute that emulates anycast
reconvergence) was confirmed on the single-node control-plane:

- A `ClusterIP` Service fronts `N` relay pods; kube-proxy load-balances across
  them.
- Deleting the serving backing pod (or rewriting `Endpoints`) causes the
  Service to reroute to a different backing set — the endpoint controller +
  ReplicaSet reconcile observed in real time.
- ⇒ The control-plane reroute that *would* break (or be survived by) a live QUIC
  flow is reproducible locally without BGP. This is the substrate the transport
  observation rides on; the transport-level observation itself is
  `DESIGN-READY`, pending the `quic-flow-probe` stand-in.

## Expected transport behavior (the answer the spike will confirm)

By construction (QUIC connection identity is the 4-tuple + connection IDs, not
the anycast VIP):

1. **kube-proxy flow-hashed routing** does NOT pin a QUIC flow to a backend — a
   flap that changes the backing set routes the *next* packet to a different
   pod, whose connection state is empty ⇒ the live QUIC connection is reset
   (`connection lost`). This is the expected failure mode the issue is concerned
   about; the spike's job is to *show* it, not assume it.
2. **QUIC connection migration** (RFC 9000 §9) could in principle keep a flow
   alive across a 4-tuple change — but only if the *client* migrates to a path
   that lands on the *same* backend with matching connection state. An anycast
   reroute to a *different* backend with no shared state defeats migration, so
   migration alone does not save anycast-relocated flows unless relays share
   connection state (a moq-relay-fabric concern, #358).
3. **WebTransport over QUIC** inherits the underlying QUIC connection fate: if
   the QUIC connection dies, the WT session dies. The browser must re-resolve
   and reconnect; there is no mid-session anycast transparency.

## Provisional conclusion (issue's 3 questions, pending the stand-in proving it)

- **Q1 — does intended anycast deployment guarantee per-flow stability?**
  No, not automatically. It is an operational property (flow-hashed routing +
  stable backings). A flap that changes the backing set breaks the live flow
  unless state is shared.
- **Q2 — needs connection-migration handling or redirect-to-stable-unicast?**
  **Redirect-to-stable-unicast on first contact** is the robust answer
  (anycast for discovery, unicast for the session) — it sidesteps the
  shared-state requirement. This matches the issue's own framing and is the
  recommended S3 design unless #358 commits to shared relay connection state.
- **Q3 — does a WebTransport session survive anycast re-route?**
  No (inherits QUIC fate). The browser must pin to a resolved instance
  (the unicast redirect from Q2).

## What remains (DESIGN-READY)

`run.sh` is the complete deploy→flap→observe→teardown flow. It needs:

1. **`quic-flow-probe` stand-in** (and a `RELAY_IMAGE`): a relay that holds a
   QUIC/moq/WT flow and a client/probe that reports `establish`/`alive`/`backend`.
   The flap mechanism is already proven; the probe just instruments the
   transport-level observation.
2. **Multi-node** (gated on the inotify fix) to make the flap a genuine
   cross-node reroute rather than a same-node endpoint swap. The conclusion
   does not change — the 4-tuple question is path-agnostic — but multi-node
   removes the "same node, same path" caveat.
3. Optionally, the `--flap=endpoints` variant (pure control-plane Endpoints
   rewrite) to decouple the flap from pod lifecycle.

Once those land, `run.sh` returns `flow survived: YES|NO` per flow type and
confirms (or refutes) the three conclusions above.
