# Spike #324 — `forward_layers` as a network RPC: capnp schema + RPC surface

**Date:** 2026-06-18 · **Status:** research spike (no code) · **Gate:** capnp schema + RPC
surface are human-gated — this is the decision menu to bring to the human before Phase-4
(#324/#325) code lands. Companion to `2026-06-18-multi-gpu-multi-host-inference-spike.md`.

## Ground truth (from code, `ewindisch/310-multi-gpu` @ 0660b3215)
- **Remote unit = `forward_layers`** (`architectures/mod.rs:375`; llama impl `llama.rs:2137`).
  Trait doc *is* the wire contract: only `hidden` + `start_pos` cross a boundary;
  `position_ids` recomputed receiver-side; KV + SSM conv/rec state stage-local, never sent;
  `range` is **global** layer indices (per-host `layer_offset` remap).
- **Cuts:** stage-0 `embed_tokens`→`forward_layers(0..b)`; middle `forward_layers(a..b)`;
  last `forward_layers(a..N)`→`apply_final_norm`→`lm_head` — all five methods already exist.
- **Control cap** `MAX_FRAME_BYTES = 4 MiB` (`rpc_session.rs:49`, control plane only; bulk →
  streaming). **Streaming plane** no app cap (`dial_stream`, `moq_stream.rs:472` writes one
  Frame=`StreamBlock||mac[16]` per Group + `finish()` — the "one frame per blob" #324 must
  chunk; `sequenceNumber == moq Group id`).
- **Serializer** `serialize_state_dict_to_bytes` (`tenant_delta.rs:1076`) forces
  `to_device(Cpu).contiguous()` per tensor — the M-WIRE GPU→CPU→bytes bounce, in code; device
  not preserved.
- **Zero-copy facts:** capnp `unaligned` on + tested (`streaming.rs:1075`);
  `read_message_from_flat_slice` parses from wire `Bytes`; moq yields `bytes::Bytes`. → keep
  the activation payload OUTSIDE the capnp message as a length-delimited flat region → decode =
  unaligned flat-slice + one host→GPU upload.
- **Arrow** present but scoped to `hyprstream-flight` only; inference/RPC crates don't depend on
  it; zero plan-doc precedent. Using it for activations = new hot-path dep.
- **No existing pipeline RPC** — greenfield. Existing surface = nested-union
  `ModelRequest{infer:…}` / `*Stream → StreamInfo` (`model.capnp:171`, `inference.capnp:79`).

## THE HUMAN DECISION MENU (capnp + RPC surface — needs sign-off before #324)
1. **Activation wire format** — (A) reuse safetensors; (B) raw bytes + 3-field capnp
   `TensorMeta{dtype,shape}` header; (C) Arrow IPC. **Recommend B** (A as v1 shim; reject C —
   hot-path arrow dep, no zero-copy win over B). CPU bounce itself is physics (tensors `!Send`,
   only bytes cross); RDMA/GPUDirect is a future *bytes-hop* swap, schema-orthogonal.
2. **Invocation vs payload** — (2a) control RPC pins chain + activations on moq; (2b) invocation
   header rides in-band in the moq StreamBlock. **Recommend 2b** — the per-block chained MAC +
   `(epoch,sequenceNumber)` authenticate header↔payload binding for free — plus a tiny control
   RPC for chain setup only.
3. **Schema home** — (1) new `pipeline.capnp` reusing `StreamBlock`; (2) extend
   `inference.capnp` with a `forwardLayers` arm returning `StreamInfo`; (3) fold into
   `streaming.capnp`. **Recommend new `pipeline.capnp` (per-hop framing) + a
   `StreamInfo`-returning `openPipeline` control RPC (setup); reject (3)** (pollutes the
   generic streaming vocab → forces all stream consumers to recompile). Net change to
   streaming.capnp = one additive union arm (wire-safe, no break).
4. **Chunking (M-CHUNK)** — multiple Frames per Group with
   `ChunkInfo{index,total,byteOffset,totalBytes}`, replacing single `write_frame`+`finish`.
   **Decide chunk size** (suggest 1–4 MiB; 67 MB prefill → ~17–67 frames/Group).
5. **Flow control (M-FLOW)** — (A) `StreamOpt.overflowPolicy=block` (lossless, already in
   schema) + bounded queue; (B) per-microbatch credit RPC. **Recommend A** (dropped frame
   already fatal via MAC + sequence gap).
6. **Provenance (C-PROV)** — (A) in-band `Provenance{signerKid,sig}` in the header (binds to
   exact bytes); (B) separate control attestation. **Recommend A**; per-host SVID signer
   (COSE Ed25519+ML-DSA hybrid, as ResponseEnvelopes), receiver verifies signer == assigned
   predecessor.
7. **AEAD (C-AEAD)** — make `TaggedPayload` AEAD **mandatory** on pipeline broadcasts (flip moq
   default integrity-only). **Recommend mandatory**, no opt-out for mesh activations/deltas;
   `StreamInfo.dhPublic` already keys it.
8. **Flow shape** — (A) push (publish→subscribe, matches existing moq code); (B) pull/credit.
   **Recommend A.**

## Candidate `pipeline.capnp` (Shape 1, recommended)
```capnp
struct StageActivation {
  jobId        @0 :Data $fixedSize(16);   # tenant-scoped routing key
  microbatchId @1 :UInt32;                # M-BUBBLE
  layerRange   @2 :LayerRange;            # GLOBAL [start,end)
  startPos     @3 :UInt64;                # position_ids recomputed receiver-side
  tensorMeta   @4 :TensorMeta;            # dtype + shape → receiver placement
  chunk        @5 :ChunkInfo;             # M-CHUNK
  provenance   @6 :Provenance;            # C-PROV per-host signature
  deltaRef     @7 :Data;                  # tenant delta content-hash (empty = base)
}
struct LayerRange { start @0 :UInt32; end @1 :UInt32; }
struct TensorMeta { dtype @0 :DType; shape @1 :List(UInt32); }
enum DType { bf16 @0; f16 @1; f32 @2; }
struct ChunkInfo { index @0 :UInt32; total @1 :UInt32; byteOffset @2 :UInt64; totalBytes @3 :UInt64; }
struct Provenance { signerKid @0 :Data; sig @1 :Data; }
```
Bulk hidden bytes ride the existing `StreamBlock`→`StreamPayload` (reuse `tagged` AEAD arm or a
new `activation @5 :Data`), kept a flat slice for unaligned zero-copy.

## RPC surface
- **Setup (control):** `openPipeline(jobId, plan:List(StageAssignment{host:TransportConfig,
  layerRange}), tenant, deltaRef) -> PipelineInfo`. Router computes capacity-aware stage→host
  plan (#325/§H); each stage gets its successor reach (`announcedAt` already models this).
- **Flow (streaming):** per hop subscribe predecessor broadcast → `forward_layers` → publish to
  successor on `{jobId, layerRange, kind=activation}` (push; backpressure via OverflowPolicy.block).
- **Failure (#325):** MAC + `(epoch,sequenceNumber)` make a dropped/reordered activation fatal;
  stage death → track ends → fail jobId → router re-pins. Per-stage KV lost on death → re-run
  from last-intact-KV stage (a #325 design deliverable, not schema-solvable).

## Non-decisions (settled by code/plan)
Remote unit = `forward_layers`; bulk → streaming, control < 4 MiB; per-token hop pays the M-WIRE
CPU bounce (physics); position_ids never sent; global indices + `layer_offset`; capacity-aware
planner deferred (#325/§H), swaps planner without touching the wire.
