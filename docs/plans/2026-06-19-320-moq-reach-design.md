# #320 — InferenceService network endpoint over the moq DH-blind-topic plane (design)

Read-only spike, grounded @ a31e90b65. For discussion before building.

## Reality check
- **No moq-relay PROCESS today.** moq is a shared-origin plane: inference publishes on a DH-derived
  blind broadcast topic (`derive_stream_keys`, key_exchange.rs:136-168 → 64-hex track name + mac_key);
  the consumer dials the PRODUCER directly (point-to-point QUIC `/moq`). `Role::Relay` (streaming.capnp:184)
  is schema-reserved but `producer_reach()` (moq_stream.rs:132-145) only emits `Role::Direct`. moq-net Server
  can `with_publish`+`with_consume` one origin (relay-CAPABLE) — relay is a deployment/wiring choice, unwired.
- **No-duplication already holds:** ModelService `handle_generate_stream` (model.rs:935-944) passes the
  worker's StreamInfo through UNCHANGED; it never subscribes/republishes. One broadcast, one HMAC chain,
  one DH topic (bound to the END CLIENT's ephemeral key, flows MS→worker untouched).

## #320 = the REQUEST leg only
- **Request leg (RPC):** ModelService→worker InferenceClient. Today inproc `inference-{safe}`
  (model.rs:230,367). #320: `select_inference_server` (model.rs:939 seam, net-new) resolves
  service:inference:host-<label> (#328) → reach, dials worker RPC via dial.rs Quic arm + expected verifying
  key. Reach roster = a QuicReach-shaped field on `MeshPeerConfig` (config/mod.rs:743, keys-only today).
  **Pure-Rust, NO capnp.**
- **Stream leg (moq):** unchanged pass-through; client subscribes once to the worker's DH-blind broadcast.

## Shapes
- **Shape D [rec v1]:** request-leg routing now; stream stays direct (client dials worker's advertised
  reach, = what `networked()` does, moq_stream.rs:576-639). Smallest delta; needs client→worker L3 reach.
- **Shape R (the "moq-relay blind topic"):** a hyprstream node ingests the worker's blind broadcast into
  its origin and re-serves; ModelService appends `Role::Relay` Destination to StreamInfo `announcedAt@4`.
  Pure-Rust, NO schema change (Role::Relay + QuicReach exist). Needs a relay node deployed; relay stays
  content-blind by routing on the opaque topic — TRUE blindness gated on #321 AEAD (enc_key).

## Frame-duplication traps (avoided by design)
- A: ModelService subscribe+republish → 2nd broadcast. Avoid: keep pass-through.
- B: a separate MS↔worker DH handshake → different topic. Avoid: reuse the single client-bound topic.
- C: parallel transport — the RPC dial is fine for the REQUEST leg; the STREAM must stay on the worker's
  existing moq broadcast (no 2nd stream transport).

## Roster split
host RPC reach = static config (MeshPeerConfig); DH topic = dynamic per-stream (NEVER static — blindness);
relay reach (Shape R) = the relay node's self-config, appended at StreamInfo-relay time.

## Pure-Rust vs capnp
All of #320 (D and R) is **pure-Rust, no capnp** — Role::Relay/QuicReach/Destination already in
streaming.capnp; the only capnp gate in this epic is #321 provenance.

## Open questions
1. Shape D (direct, v1) vs wire Shape R (relay) now? (No relay node runs today → D is the honest v1.)
2. "moq-relay" = a hyprstream node fanning out its origin (Shape R), NOT an external relay daemon (none
   in tree; moq-net ships none). Confirm — pulling an external relay = a new-dep decision.
3. Shape R: who appends the relay reach — ModelService (natural, it touches StreamInfo) needs to know the
   relay's reach (its own /moq config).
4. Relay ingest authz: the relay subscribing to the worker's blind topic is a moq subscribe (moq_authz,
   per-tenant) — needs broad subscribe rights; stays content-blind via #321 AEAD.
5. Shape R blindness is convention-only until #321 AEAD lands (key_exchange.rs has no enc_key yet) →
   #320-relay depends on #321.
