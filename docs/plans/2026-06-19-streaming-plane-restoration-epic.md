# Streaming Plane Restoration — Epic plan

**Date:** 2026-06-19 · **Branch:** `feature/streaming-plane-restoration` (off `main` @ 3176362a4) · all
epic PRs target this branch; merging it closes the epic. **Status:** planning (audit done; no impl yet).

## Why
The ZMQ→moq migration (#131/#138) removed the `StreamService` — a PUSH/PULL→XPUB **blind-forwarding
rendezvous** (neither end dialed the other) — and replaced it with an **in-process moq origin served over
UDS** (`GLOBAL_MOQ_UDS_PATH`). The networked + rendezvous path was **never rebuilt**. Streaming inference
works today only by **direct producer-dial** (`producer_reach()` only ever emits `Role::Direct`;
`Role::Relay` is defined but never constructed). Two consequences are **live bugs on main**.

## Current flow (verified)
`POST /oai/v1/chat/completions` → ModelService (pass-through) → InferenceService derives a **DH-blind
topic** (`derive_stream_keys`) → publishes `StreamBlock`s on the process-global `MoqStreamOrigin`
(`{prefix}/{topic}`) served over UDS + QUIC/WebTransport `/moq`; the signed `StreamInfo`
(dhPublic/broadcastPath/`announcedAt:List(Destination)`) is returned; the **consumer dials the producer
directly** (`MoqStreamHandle::networked`→`dial_stream`). **Works:** co-located (UDS), directly-reachable
QUIC. **Breaks:** NAT'd producer, cross-instance (broadcast on another PDS process), mutually-unreachable,
anonymization.

## Decisions (user, 2026-06-19)
- **Transport split: iroh+pkarr DIRECT for native peers; moq relay only for browsers + fan-out +
  anonymization.** The relay was partly a pre-pkarr crutch; iroh (NAT hole-punch) + pkarr (discovery)
  gives native peers direct reach + cross-instance without a content relay. iroh is `cfg(not(wasm32))` →
  browsers still need a WebTransport-reachable relay; fan-out + anonymization also need a relay.
- **Relay restoration is its OWN epic** (spans inference + TUI + notification + 3 schemas + AEAD + iroh;
  on main). Multi-gpu #320 is a CONSUMER/subset (its mesh reach = iroh-direct, S2).
- **Live cleanup bugs deferred INTO this epic** (S1).

## Sub-issues
- **S1 (PR-C) — unify StreamInfo reach model + fix the two live bugs.** Migrate `tui.capnp` (divergent
  StreamInfo: `subEndpoint`/`moqBroadcastPath`/`moqUdsPath`) and `notification.capnp` (dead
  `streamEndpoint @2` + `moqUdsPath @3`/`moqBroadcastPath @4`) onto the networked `announcedAt:List(Destination)`
  reach + a single `producer_reach()` builder; delete dead UDS/endpoint fields. **Fixes #142**
  (cross-instance model-load notifications silently fail — git_handlers UDS-only) **and the TUI
  cross-process UDS timeout** (#275-class). Pure-Rust + schema (capnp). HUMAN-GATE: the schema spin.
- **S2 (PR-E-direct) — iroh+pkarr direct reach for native peers.** Add `node_id` to
  `TransportConfig.iroh` (today `:Void` → can't advertise iroh reach); advertise the iroh reach in
  `announcedAt`; wire the `dial_stream` iroh arm to consume it; pkarr discovery for dial-by-node_id.
  Fixes NAT + cross-instance + discovery for native peers with NO content relay. **This is the reach #320
  depends on.** HUMAN-GATE: the `TransportConfig.iroh` schema field.
- **S3 (PR-D) — moq relay (rendezvous) for browsers + fan-out + anonymization.** Port the EXISTING
  event-plane rendezvous (`moq_event.rs:194-260` `with_origin` = ingest-and-re-serve) to the inference
  plane: a relaying node `dial_stream`s the upstream producer, subscribes to `broadcastPath`, re-announces
  into its local origin; advertise `Destination{Role::Relay, Quic(...)}`; consumer relay-selection.
  WebTransport-served for browsers. Relay stays content-blind (routes on opaque topic; AEAD via S5).
  Mostly net-new but templated. HUMAN-GATE: relay topology / authz (relay subscribe rights).
- **S4 (cleanup) — delete teardown debt.** Orphaned ZMQ XPUB/XSUB endpoint constants
  (`hyprstream-workers/src/events/endpoints.rs`, no callers); stale `StreamService` comments/warns
  (`notification.rs`, `factories.rs`); verify-and-remove unused `StreamRegister`/`StreamResume` structs
  (moq Track cache replaced them). Pure-Rust; verify-unused before delete.
- **S5 (PR-B) — blind-relay AEAD + provenance — DONE as #321/#354 on `ewindisch/310-multi-gpu`.** Tracked
  as a DEPENDENCY of S3 (relay blindness). Reconcile/port onto this branch/main (it currently lives on the
  multi-gpu branch). enc_key in derive_stream_keys + encrypt-at-source + COSE provenance.
- **S6 (PR-E follow-ons, optional/defer) — pkarr full discovery + oblivious-relay.** Off the critical path.

## Completeness (audit) ≈ 40%
PR-A codegen ~done; PR-B (#354) done-on-310; PR-C ~50% (inference done, tui/notification not = S1);
PR-D ~20% (the gap, templated on moq_event.rs = S3); PR-E ~55% (iroh dial+pkarr wired; blocked only by
`TransportConfig.iroh :Void` = S2). Reuse: generated codec, `Role::Relay` enum, moq-net
publish+consume+origin, the event-plane rendezvous template, `dial_stream` (QUIC+iroh), COSE+AEAD.

## Sequencing
S1 (fix live bugs + unify reach) → S2 (iroh-direct for native, unblocks #320) ∥ S4 (cleanup) →
S3 (relay for browser/fanout, needs S5 AEAD for blindness) → S6 follow-ons. S5 is done (reconcile from 310).

## Cross-epic
#310 #320 (multi-gpu mesh reach) DEPENDS ON S2 (iroh-direct). #353 (Worker GA authz) and worker sandbox
streams also ride this plane. Keep the schema spins (S1/S2/S3 touch streaming/tui/notification capnp)
human-gated.
