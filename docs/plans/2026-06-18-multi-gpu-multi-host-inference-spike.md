# Multi-GPU & Multi-Host Inference + Training — Architecture Spike

**Date:** 2026-06-18
**Status:** Research spike (evaluation + sequencing); no code yet. **Revised after a
5-perspective review — see [§ Review v1](#review-v1--five-perspective-findings--revisions) at
the end, which supersedes the original phasing.**
**Context:** Evaluate the current model/inference-service design and the federation/auth
model *in the context of* planning multi-GPU and multi-host inference & training.

> **Code-path note:** all file:line citations are against the **moq-epic codebase** (worktree
> `.worktrees/ewindisch/readme`, = old main + the 153-commit epic). The local `main` /
> `origin/main` refs in this checkout are **stale at the pre-epic merge-base `2a463263d`**
> (no `dial.rs`, no `moq_stream.rs` — it predates the ZMQ→moq migration). Verify against the
> epic worktree, not the stale local `main`. (See R-DATA in §Review v1.)

## Target requirements

1. **One inference service may own multiple GPUs.**
2. **The model service talks to multiple inference services.**
3. **Inference services operate peer-to-peer (within a PDS, for now) for networked
   inference *and* training tasks.**

## Decision #0 — ANSWERED (hardware & scale target)

- **North star: multi-terabyte models.** This **keeps inter-host pipeline (2c) firmly in
  scope** — multi-TB exceeds any single host, so networked pipeline parallelism is the
  *essential endgame*, not the "probably-deletable" phase the review guessed. It is sequenced
  later, not dropped.
- **v1: ~35 GB model on 2× AMD MI210, single host, homogeneous.** MI210 = 64 GB HBM2e each, so
  a 35 GB model **fits on one GPU** → this hardware is the **2a/2b proving ground**, and (key
  testing property) it lets us validate pipeline-split output against **single-GPU ground
  truth** before trusting the split on models that don't fit. Build 2b here as a *deliberate
  de-risking step*, even though the model fits one card.
- **Then: cross-vendor, cross-machine — MI210 + RTX 5090 on separate boxes.** Adds the axis the
  plan was missing: **heterogeneity** (see new finding H below). RTX 5090 = 32 GB GDDR7,
  Blackwell/CUDA; MI210 = 64 GB, CDNA2/ROCm. This is simultaneously cross-vendor (ROCm+CUDA)
  and cross-host (→ exercises 2c).

**Hardware → phase mapping:** homogeneous 2×MI210 single host = Phase 1 (2a/2b) proving ground;
cross-vendor 2-machine = Phase 4 (2c) + heterogeneity proving ground; multi-TB = the scale 2c
ultimately serves.

---

## TL;DR

- The **substrate is ~70% there for cross-host *inference*.** Identity-bound RPC and
  cross-host moq streaming both work today over QUIC and iroh; the schema already supports
  streaming. The missing inference pieces are *routing/pooling* and *remote spawn/discovery*,
  not transport.
- **Multi-GPU is single-device-assumed today, but the model is unusually well-shaped for
  layer (pipeline) parallelism.** `TorchEngine` pins one immutable `Device` and the loader
  places the whole model on it — yet the forward pass is a textbook-clean sequential loop
  (`for layer in self.layers`, hidden-state-only hand-off), the KV cache is already
  per-layer (`Vec<LayerKVCache>`), embed and norm+lm_head sit cleanly outside the loop, and
  the layer count and shard locality come straight from the model's own `config.json` +
  `model.safetensors.index.json` (no tensor-name regex needed), so a layer shard is cleanly
  selectable. **Splitting a single model's layers across local GPUs (intra-process) and across
  inference services (inter-host pipeline stages) is a clean partition, not a rework.** The real cost is
  per-token *hop latency*, not code entanglement. (True **tensor** parallelism — splitting
  *within* a layer, all-reduce per layer — is the separate hard one; out of scope here.)
- **The cross-org federation gate is the wrong tool for the inference mesh — but the PDS is
  NOT a trust zone.** The admission gate is built for cross-*org* peering (origin Casbin +
  DID-key binding). The inference nodes need a *lighter* path than that — **but the PDS is
  multi-tenant and runs untrusted workloads**, so co-location is **not** a trust signal. The
  mesh is a **named SPIFFE-style trust *group*** within the multi-tenant PDS: each node is a
  workload identity (SVID), mesh rights are granted **only** to the specific inference-node
  identities (deny-by-default for every other workload on the PDS), and AEAD + provenance must
  hold against **same-PDS adversaries**, not just external ones. "Co-trusted fleet" was the
  wrong framing — it's least-privilege per workload identity. SPIFFE-forward, with an interop
  bridge to the did:web/did:key + COSE model.
- **Distributed *training* is the genuinely net-new build.** There is **no** tensor/gradient
  transfer primitive and **no** collective-comms (all-reduce/gather). Given this codebase
  trains **TTT/LoRA deltas** (not full-model DDP), a **parameter-server-style delta
  aggregation** (peers ship deltas → aggregator merges → git checkpoint) is a far better fit
  than classical NCCL all-reduce.

**Recommended mental model — a clean two-level hierarchy that maps onto existing code:**

```
ModelService (cluster router)            ← requirement 2 + 3 (inference)
  └─ InferenceService (per host, owns 1..N local GPUs)   ← requirement 1
       └─ TorchEngine (per GPU)          ← exists today, one Device each
```

Keep the two scaling axes separate: **InferenceService = the unit that owns local GPUs**
(host-local scheduler); **ModelService = the cross-host router** over a pool of
InferenceServices. Don't conflate them. **Pipeline (layer) parallelism then composes along
*both* axes from the same clean partition:** within an InferenceService (layer groups on
different local GPUs, microsecond NVLink/PCIe hops) and across InferenceServices (pipeline
stages on different hosts, network-RTT hops — best for models too big for one host and for
throughput via microbatching).

---

## Current architecture (evaluated)

### A. Model ↔ inference topology  (`services/model.rs`, `services/inference.rs`, `services/factories.rs`)

- **`ModelService`** (`model.rs:139`, inner `:104`) is a singleton manager: LRU cache of
  loaded models (default 5, `factories.rs:164`), routes inference requests.
- **`InferenceService`** (`inference.rs:176`, config `:2441`) wraps `TorchEngine`; one is
  spawned **per loaded model** via `LocalServiceBridge::spawn_with` (`model.rs:365`,
  `inference.rs:2487`) on a **dedicated thread** because tch-rs tensors are `!Send`
  (`inference.rs:191`).
- Relationship is effectively **1:1 model→inference** today: `get_inference_client`
  (`model.rs:583`) returns a single cached `InferenceClient` (`LoadedModel.client`,
  `model.rs:72`). No pool, no load-balancing.
- Endpoint naming is deterministic: model ref → `inference-{safe}` → registry endpoint
  (`model.rs:230`). **All in-process today.**
- Request path (5 hops): `POST /oai/v1/chat/completions` (`openai.rs:300`) → `ModelClient`
  (`openai.rs:412`) → `ModelService` handler (`model.rs:946`) → `InferenceClient` →
  `TorchEngine`. Streaming returns `StreamInfo` immediately (`model.rs:935`).
- **Cap'n Proto schema already supports the streaming/continuation pattern** — no RPC-shape
  schema change needed to add remote inference.
- **Naming caveat:** comments referencing "ZMQ / inproc / `SocketKind::Rep`" are **residual**
  from the pre-moq era. The live transport is the `dial()` abstraction (moq + Cap'n Proto).
  Because InferenceService already sits behind that RPC boundary, making it remote means
  **giving it a network endpoint + discovery**, not rewriting the service.

**Seam for 1→N:** `LoadedModel.client: InferenceClient` → `Vec<InferenceServerInfo{host,
endpoint, client, gpu_id, load}>`; `get_inference_client` → `select_inference_server`
(router); spawn N with affinity / accept remote self-registration.

### B. Multi-GPU / device handling  (`runtime/torch_engine.rs`, `config/mod.rs`, `training/ttt.rs`, `runtime/kv_cache.rs`)

- **Two-phase device selection.** (1) *Backend* (libtorch variant cuda130/rocm71/cpu) chosen
  **once per process** at launch (`cli/gpu_detect.rs:14`, `HYPRSTREAM_BACKEND`). (2) *Device
  index* chosen **once per `TorchEngine`** from `RuntimeConfig.gpu_device_id: Option<usize>`
  (`config/mod.rs:1593`, `HYPRSTREAM_GPU_DEVICE` `:1614`) → `Device::Cuda(id)`
  (`torch_engine.rs:321`).
- `TorchEngine.device` is **immutable** (`torch_engine.rs:59`); all forward/backward use
  `self.device`. Weights load directly to the device; CPU only for output serialization.
- **"Per-process GPU selection" = one env var per spawned process, NOT multi-device in one
  process.**
- **No multi-device anywhere:** no `world_size`/`rank`/`nccl`/`all_reduce`/`tensor_parallel`/
  `pipeline_parallel`/`shard`. ("parallel LoRA" in `batched_lora.rs` is batched matmul, not
  multi-GPU.) Training (`ttt.rs:548`) takes a single `Device`; per-tenant delta pool is
  single-device.

**Chokepoints for "one service owns N GPUs":** `TorchEngine.device` (single, immutable),
`RuntimeConfig.gpu_device_id` (`Option<usize>`, not `Vec`), device fixed at
`TorchEngine::new()`. Need a `DevicePool`/device-strategy layer + `RuntimeConfig.devices:
Vec<usize>`. Two flavors: **replication** (N TorchEngines, one per GPU — easy, throughput
only) and **pipeline / layer split** (one model's layers across GPUs — see B2; the
higher-value target for big models).

### B2. Model / pipeline-parallelism feasibility  (`runtime/architectures/*.rs`, `model_factory.rs`, `model_config.rs`)

Dedicated spike into the forward pass + weight loading. **Verdict: the architecture is
unusually well-suited to layer (pipeline) parallelism — clean partition, ~8.5/10.**

- **Forward pass is a textbook sequential loop** (`qwen3_5.rs:1514`, `llama.rs:~1730`):
  `for (idx, layer) in self.layers.iter()` with the *only* cross-layer dependency being the
  `hidden` tensor (`[batch, seq, hidden]`). No lookahead, no side-channels.
- **A pipeline-stage boundary carries:** the hidden-state tensor **plus** `position_ids` +
  `start_pos` — both cheap and **recomputable per stage** (generated once before the loop,
  `qwen3_5.rs:1505`), not large state. **KV cache and SSM conv/rec state are per-layer
  (`Vec<LayerKVCache>`, `kv_cache.rs:892`) and stay *stage-local* — never transferred.**
- **Embed (before loop, `qwen3_5.rs:1487`) and final norm + lm_head (after loop, `:1717`) are
  cleanly separable** → three independent regions: stage-0 embed, decoder stages `[a..b)`,
  last-stage norm+lm_head.
- **The split is planned from structured JSON, not name-parsing.** `num_hidden_layers` (+
  hidden size, arch) come straight from the model's **`config.json`** (`model_config.rs:174,215`;
  arch detection `:343`) — a lightweight read that touches **no weights**. Weight locality comes
  from the authoritative HF shard manifest **`model.safetensors.index.json`** and its
  `weight_map` (tensor → shard file, `model_factory.rs:127,162`). So a pipeline planner computes
  stage boundaries from `config.json` and resolves each stage's tensors via the index manifest —
  both documents the model already ships. The regex scan of tensor names (`count_layers`,
  `model_config.rs:332`) is **only the fallback** when `config.json` is absent (`:150`). Net:
  **each service loads only its layer shard** (a `StreamingWeightProvider` skeleton already
  exists, unused).
- **What's single-device-baked today:** `from_weights(.., device: &Device, ..)` places *all*
  layers on one device (`qwen3_5.rs:1218`); the loader moves every tensor to the global
  device (`model_factory.rs:436`). Needs a per-layer `device_map` + per-layer `.to()` at
  construction, and (for cross-device) a `hidden.to_device(next)` at stage boundaries in the
  forward loop.
- **Cost model (correcting an inflated subagent estimate):** the hidden state is
  `seq × hidden × 2B`. **Decode = one token ≈ a few KB per hop** (bandwidth negligible);
  **prefill ≈ tens of MB once** (e.g. 8K×4096×2B ≈ 67 MB, *not* 1.2 GB). So the binding
  constraint for inter-host pipeline is **per-token hop latency** (network RTT × stage
  boundaries per token), not bandwidth — intra-host (NVLink/PCIe) hops are microseconds and
  effectively free.

**Change surface (per the spike):** layer-subset loading ≈ small (filter by layer range);
per-layer device placement ≈ moderate (signature threading through `from_weights`/`build_layer`
+ boundary `.to_device`); a stage-aware `forward_split_pipeline` entry on `TorchEngine`
(`torch_engine.rs:902`). No fundamental entanglement to undo.

### C. RPC + streaming substrate  (`hyprstream-rpc/dial.rs`, `spawner/service.rs`, `moq_stream.rs`, `rpc_session.rs`)

- **`dial()` supports Inproc / Quic / Iroh / Ipc / SystemdFd — all active** (`dial.rs:147-216`).
- **iroh IS bound** (not stubbed), gated on `qc.iroh_enabled` (`spawner/service.rs:196`);
  shared endpoint installed for outbound dials (`:235`). Two ALPNs run in parallel:
  `hyprstream-rpc/1` (Cap'n Proto RPC) + `moql` (streaming).
- **Cross-host RPC works** by Ed25519 identity; **cross-host moq streaming works** over QUIC
  and iroh (`moq_stream.rs:593`), in-process producers auto-visible to remote subscribers,
  late-join via moq track cache.
- **Payload limits:** RPC frame cap **4 MiB** (`rpc_session.rs:49`) — control plane only. The
  **moq streaming plane has no hard payload cap** → suitable for large blobs
  (activations / KV-cache / weights).
- **Missing:** no tensor/gradient transfer primitive; no collective comms; weights move only
  via git checkout (`training/checkpoint.rs` is local-FS + git). `dial_stream` iroh arm does
  **not** yet fall back to pkarr-only discovery (#282) — must publish explicit reach (direct
  addrs or relay) today.

### D. Federation / auth model  (`auth/federation_admission.rs`, `hyprstream-rpc/admission.rs`, `node_identity.rs`, `auth/policy_templates.rs`, `auth/key_source.rs`)

- **"PDS" today = one node = one origin = one DID identity** (`did_document.rs:257`). **No
  multi-host-under-one-PDS grouping exists.**
- **Admission gate is built for cross-org federation:** stage 1 origin Casbin
  (`federation:register`, deny-by-default, `federation_admission.rs:41`), stage 2 peer-key↔
  DID-VM binding (`admission.rs:195`). Stage-2 live peer-key extraction on QUIC/WT is still a
  `TODO(#200/#185/#282)` (`federation_admission.rs:100`).
- **Each daemon has its own identity:** HKDF-derived purpose keys (`node_identity.rs:33`), own
  iroh `node_id`, own DID. **`ClusterKeySource` already assumes cluster services trust one
  PolicyService** and issue JWTs with `service:*` subjects (`key_source.rs`).
- **Policy vocabulary is user→model and service→policy** (`policy_templates.rs:74`); there is
  **no resource for inference↔inference RPC**.

**Fit verdict:** the federation gate is **over-heavy and wrong-shaped** for a co-trusted
intra-PDS fleet (it would admit/deny all same-origin hosts together; #200 seam not wired for
QUIC/WT). The right fit is the **cluster-CA / service-identity (JWT) path** over iroh's
transport-layer identity binding — the channel already proves Ed25519 identity, so the gate is
redundant for same-trust-domain hosts.

---

## Gap analysis per requirement

| Requirement | Have today | Gap | Difficulty |
|---|---|---|---|
| **1. Inference service owns N GPUs** | Single immutable `Device` per `TorchEngine`; thread-per-engine; clean sequential forward loop; per-layer KV cache; HF-named weights | Replication: DevicePool + `RuntimeConfig.devices`. **Pipeline/layer split: per-layer `device_map` + boundary `.to_device` + layer-subset load** | **Med** (replication) / **Med** (intra-host pipeline) |
| **1b. Single model split across services** | Same clean partition; cross-host moq/iroh transport already moves bytes | Stage-aware `forward_split_pipeline`; ship hidden state between stages; per-service shard load; stage orchestration | **Med–Hard** (latency-bound, not entanglement-bound) |
| **2. Model service → N inference services** | Singleton ModelService; 1 cached client/model; deterministic naming; clean RPC boundary; schema ready | `LoadedModel`→pool; `select_inference_server` router; remote spawn + dynamic discovery; network endpoint for InferenceService | **Med** |
| **3. Inference P2P within a PDS (inference)** | iroh+QUIC RPC + moq streaming cross-host; per-daemon identity; cluster-CA JWT path | Intra-PDS trust domain (use cluster-CA, not fed gate); `mesh:rpc`/`inference:peer-call` policy vocab; publish reach (pre-#282) | **Med** |
| **3. …for *training* tasks** | Single-node TTT/LoRA; local git checkpoint; moq plane can carry large blobs | **Net-new:** tensor/delta transfer primitive; collective/aggregation pattern; distributed optimizer/merge | **Hard (net-new)** |

---

## Recommended sequencing

### Phase 0 — Foundations & decisions (small, unblocks everything)
- **Trust model:** introduce an explicit **intra-PDS / cluster trust domain**. Co-hosted
  inference daemons authenticate by **service-identity JWT (cluster-CA)** over iroh's
  identity-bound channel, **bypassing the cross-org admission gate**. (Decision needed:
  shared cluster root key vs. per-host keys enrolled into a cluster roster.)
- **Policy vocab:** add `mesh:rpc` / `inference:peer-call` resource+action; template rule
  `service:model, *, mesh:rpc, invoke, allow`.
- **Give InferenceService a network endpoint** (Quic/iroh) + register host/GPU metadata in a
  discoverable registry (replace static inproc naming with a dynamic registry).

### Phase 1 — ModelService → N InferenceServices (remote-capable pool)  *(req. 2; pragmatic 1+3-inference)*
- `LoadedModel.client` → `Vec<InferenceServerInfo>`; add `select_inference_server` (round-robin
  / least-loaded / GPU-affinity) at `model.rs:939` seam.
- Model service dials inference services via `dial()` (inproc fast-path for co-located, Quic/
  iroh for remote). Support remote spawn or self-registration.
- **Delivers:** model talks to many inference services; running one inference process per GPU
  per host already yields a working multi-GPU + multi-host *inference* story.

### Phase 2 — Multi-GPU within a host  *(req. 1)*
- **2a (replication / data-parallel — quickest throughput win):** `DevicePool` in one process,
  N `TorchEngine`s (one per local GPU), route requests across local devices.
  `RuntimeConfig.devices: Vec<usize>`. Each replica still needs the whole model to fit on one GPU.
- **2b (intra-host pipeline / layer split — the big-model unlock, now first-class):** one model's
  layers split across local GPUs. Per the B2 spike this is a *clean partition*, not a rework:
  - `RuntimeConfig.devices: Vec<usize>` + a layer→device `device_map` (round-robin or
    balanced-by-params).
  - Thread the device through `from_weights`/`build_layer`; `hidden.to_device(next)` at group
    boundaries in the forward loop (microsecond NVLink/PCIe hops).
  - Layer-subset loading so a process need not materialize the whole model on one device first.
  - Lets a model **larger than a single GPU** run on one host. (Tensor parallelism — intra-layer
    all-reduce — remains out of scope; pipeline gets us the big-model win without per-layer
    collectives.)

### Phase 2c — Single model split across InferenceServices (inter-host pipeline)  *(req. 1b)*
- Same partition as 2b, but stages live on **different hosts** and the hidden state crosses the
  network at each boundary over the existing moq/iroh substrate.
- Add a **stage-aware `forward_split_pipeline`** entry on `TorchEngine` (`torch_engine.rs:902`);
  each service loads only its layer shard; a planner assigns stages from `config.json`
  `num_hidden_layers` before any weights load.
- **Latency-bound, not entanglement-bound:** single-stream decode pays network RTT × stage
  boundaries per token, so target (a) models too big for one host and (b) **throughput via
  microbatching / continuous batching** (keep all stages busy across many concurrent requests),
  not single-request latency.
- Reuses Phase-0 cluster trust + Phase-1 routing/discovery; the stage chain is a pinned route
  through the InferenceService pool.

### Phase 3 — Networked training  *(req. 3, training — the hard, net-new part)*
- **Tensor/delta transfer primitive** over the **moq streaming plane** (no 4 MiB cap):
  SafeTensors/delta blobs, topic = `{job, layer, kind}`.
- **Aggregation pattern:** prefer **parameter-server-style delta aggregation** over NCCL
  all-reduce — peers ship **TTT/LoRA deltas**, an aggregator **merges** (reuse existing
  delta-pool/merge) and **git-checkpoints** the result. This maps onto the existing
  RSI *propose → train → evaluate → promote* loop far better than full-model DDP.
- **Discovery:** publish explicit reach until pkarr lands in `dial_stream` (#282).

---

## Key risks & open decisions
- **Replication vs pipeline (now both in scope).** Replication + routing (Phases 1–2a) is the
  quickest throughput win and stays cheap. **Pipeline/layer split (2b intra-host, 2c
  inter-host) is the path for single models bigger than one GPU/host** and — per the B2 spike —
  is a clean partition rather than the "hard epic" originally assumed. Open question is
  *ordering*: ship 2b (intra-host, microsecond hops, high value, lower risk) before 2c
  (inter-host, latency-bound)? Recommended yes.
- **Tensor parallelism deliberately deferred.** Pipeline split avoids per-layer collectives;
  intra-layer tensor parallelism (heads/MLP columns + all-reduce each layer) is a separate,
  latency-sensitive effort — only revisit if a single *layer* won't fit one GPU.
- **Pipeline decode latency.** Inter-host pipeline (2c) adds RTT-per-boundary per token; needs
  microbatching/continuous batching to be worthwhile. Confirm the target workload (big-model
  serving / throughput vs. low-latency single stream).
- **Cluster trust provisioning.** Shared cluster root key (simplest) vs. per-host keys with a
  cluster roster (better isolation, revocation). Affects Phase 0.
- **Training topology.** Parameter-server/delta-merge (recommended, fits the code) vs. true
  collective all-reduce (more general, much more to build). Confirm the training shape
  (TTT/LoRA deltas only, or full-weight gradients?).
- **#282 pkarr seam** gates frictionless discovery; until then reach must be published.
- **Don't reuse the federation admission gate intra-PDS** — explicitly scope it to cross-org
  peering so the cluster path stays light.

## Source map (representative)
- Topology: `services/model.rs:104,139,230,365,583,935,946`; `services/inference.rs:176,191,2441,2487`; `services/factories.rs:164,587`
- Multi-GPU: `runtime/torch_engine.rs:59,321,902`; `config/mod.rs:1593,1614`; `training/ttt.rs:548`; `runtime/kv_cache.rs:216`
- Pipeline parallelism: forward loop `runtime/architectures/qwen3_5.rs:1487,1505,1514,1717`, `llama.rs:~1730`; per-layer KV `runtime/kv_cache.rs:892`; device-baked load `qwen3_5.rs:1218`, `model_factory.rs:225,366,436`; layer-count/config from `config.json` `model_config.rs:174,215,343` (regex fallback `:332`); shard manifest `model.safetensors.index.json` weight_map `model_factory.rs:127,162` + unused `weight_provider.rs`
- Substrate: `hyprstream-rpc/dial.rs:147-216`; `hyprstream-service/.../spawner/service.rs:196-247`; `hyprstream-rpc/moq_stream.rs:383,593`; `hyprstream-rpc/transport/rpc_session.rs:41-96`; `training/checkpoint.rs`
- Federation/auth: `auth/federation_admission.rs:41,100`; `hyprstream-rpc/admission.rs:195`; `node_identity.rs:33`; `auth/policy_templates.rs:74`; `auth/key_source.rs`; `services/oauth/did_document.rs:257`

---

## Review v1 — five-perspective findings & revisions

Reviewed 2026-06-18 by five independent subagents: distributed/ML-infra architect, security/auth,
principal Rust engineer, scope/sequencing, performance/cost. Findings deduped and ranked. This
section **supersedes the original phasing** where they conflict.

### R-DATA (process finding — read first)
The five reviewers were pointed at the **stale local `main` checkout** (pre-moq-epic,
`2a463263d`); the round-1 research agents read the **epic worktree**. So two reviewer "code
facts" are **artifacts, not real defects**, now verified against the epic code:
- ❌ "`dial.rs`/`moq_stream.rs`/`rpc_session.rs` don't exist / `dial_stream` unbuilt" — **false.**
  All exist in the epic; `dial()` at `dial.rs:115`, `dial_stream()` at `:264` (built, has
  fail-fast tests). The "missing" reading is the stale checkout.
- ❌ "the 4 MiB RPC cap is really 64 MiB" — **conflated two layers.** Both exist: `MAX_FRAME_BYTES
  = 4 MiB` (RPC request/reply app cap, `rpc_session.rs:49`) **and** `MAX_FRAME_SIZE = 64 MiB`
  (ZMTP transport framing, `zmtp_framing.rs:27`). The doc's "4 MiB RPC cap" was correct.
- ✅ Real doc bug they caught: architecture dir is **`runtime/architectures/`**, not
  `runtime/models/` (now fixed throughout).
- ⚠️ **Open infra question this surfaced:** local `main`/`origin/main` sit at the pre-epic
  merge-base with **0 commits past it**, which contradicts the "epic merged to `origin/main`"
  assumption. Either the local clone is unfetched or the epic isn't actually on `origin/main`.
  **Confirm before building** (a `git fetch` / GitHub check) — the whole substrate story
  depends on the epic being the real base.

Everything below is **design/logic** (valid regardless of which checkout was read).

### Critical (resolve as design deliverables before building the affected phase)

- **C-SEQ — reorder: ship intra-host multi-GPU FIRST.** The highest-value, lowest-risk win
  (2a replication + 2b intra-host pipeline → "a model bigger than one GPU runs") needs **zero
  networking** and is gated behind P0 (cluster trust) + P1 (router) for no real reason. Promote
  it to Phase 1. *(scope C1)*
- **C-GATE0 — ANSWERED (see [Decision #0](#decision-0--answered-hardware--scale-target)).**
  North star is **multi-TB**, so all three regimes are in scope: v1 fits one MI210 (replication
  proving ground), 2b on the 2×MI210 host (de-risk against single-GPU ground truth), 2c for the
  multi-TB endgame. The reviewers' guess that "2c may have no workload" is **overturned** — 2c is
  the essential endgame, just sequenced later. *(scope C3, perf C1)*
- **C-SEND — the `!Send` multi-device threading model is the real risk inside "easy" 2b.**
  tch tensors are `!Send`, engines are one-per-thread, and RoPE is a thread-local cache keyed by
  `(layer_idx, rope_theta)` (`qwen3_5.rs:990`). One process owning multiple CUDA devices is a
  *threading* problem, not just tensor placement. **Spike single-process multi-device tch BEFORE
  committing to 2b.** *(architect M2, rust C2)*
- **C-AEAD — mandate encryption of activations/deltas in motion.** The moq plane defaults to
  integrity-only (chained HMAC); AEAD (`TaggedPayload`) is opt-in. Relays and intermediate
  pipeline stages **terminate TLS → see plaintext**. Hidden states leak prompt content
  (activation-inversion); deltas are model IP. Make end-to-end AEAD a **requirement**, not an
  option, for any mesh activation/delta payload. *(security C2)*
- **C-IDENT — per-host identity + roster, not shared root.** Today `ClusterKeySource` is a single
  shared CA key (kid ignored, empty-issuer trusted) and `NodeIdentityProvider::resolve` collapses
  every key to one `"system"` subject → no per-host identity, no revocation, fleet-wide blast
  radius on one key leak. Phase 0 must issue **per-host service identities enrolled in a roster**,
  reject empty-issuer on the mesh, and fail-closed on authz errors. *(security C1, M3)*
- **C-PROV — provenance + delta-validation (anti-poisoning).** The chained HMAC proves "held the
  DH key + in order," not "this host computed this correctly." A rogue stage can return wrong
  hidden states; a rogue training peer can poison the aggregated delta that gets git-promoted.
  Sign stage outputs/deltas with **host identity**, and gate aggregation with norm-bounding +
  anomaly checks + a held-out eval the contributors don't control. *(security C3)*
- **C-AGG — the param-server "aggregator" is net-new, not reuse.** `DO-Merge` is **non-associative**
  (`merge(merge(a,b),c) ≠ …`) and asserts **shape equality** (`merge.rs:99`), which the per-layer
  **rank oracle violates** when workers emit differently-ranked LoRA deltas. A real reducer
  (order-independent, rank-reconciling) must be designed. *(architect C4)*

### Major

- **M-TRAIN-COUPLING — do NOT cleanly split "training" out; TTT is coupled to the partition.**
  (Refines the reviewers' "spin training into its own epic.") TTT is *inference-time* training —
  it adapts the **same model partition** that inference uses, and `ttt_trainer` shares the engine's
  device. So "training" is **not one thing**; it splits into two with very different coupling:
  1. **TTT-on-the-partition (tightly coupled — keep IN the inference phases).** Whatever partition
     you build for inference, TTT's backward pass must traverse it. On **2b (intra-host)** this comes
     nearly for free: PyTorch autograd runs the reverse of the forward loop across the local
     `device_map`, materializing grads on each param's device (handle the device-correct grad +
     KV-device-affinity traps, rust m2). **Build TTT-on-split *with* 2b** — it's the RSI story and
     it's cheap once the device map exists. Validate against single-GPU TTT ground truth on the
     MI210 pair (Decision #0's testing gift applies to training too).
  2. **Cross-host distributed training (net-new — stageable workstream, design-aware now).** Two
     sub-modes, increasing difficulty: (a) **delta aggregation across replicas** (param-server:
     independent workers each TTT, ship LoRA/TTT deltas, aggregator merges → git checkpoint) —
     coupled to **2a + cluster trust**, the cleaner first distributed-training target; (b)
     **inter-host pipeline-parallel training** (backward gradient comms across 2c stage boundaries,
     1F1B-style) — coupled to **2c**, the hard one, multi-TB endgame. Both are net-new
     (aggregator/comms + provenance + the C-AGG reducer problem) and depend on C-IDENT trust.
  - **Net:** don't retitle to "Inference-only" and don't fully wall training off. **Fold
    TTT-on-partition into Phases 1/4; keep cross-host *training coordination* as a stageable
    workstream** — but design the inference partition so it does **not preclude backward** (don't
    build forward-only hand-offs that can't carry gradients). *(scope C2, architect M4; user steer:
    TTT coupling)*
- **M-2C-SCALE — inter-host pipeline (2c) is the multi-TB endgame, not conditional.** (Decision #0
  overturns the "conditional" framing.) It is sequenced *later* but is required. Still bind it to
  the break-even envelope so you don't pay its latency before you need it: single-stream ceiling ≈
  `1/(K·RTT)`; added latency ≈ `(K−1)·RTT`; ~330 tok/s/stream at K=4, 1 ms RTT; **LAN-only,
  minimize boundaries**. Multi-TB serving is throughput-oriented, so the microbatching story
  (M-BUBBLE) is mandatory for 2c, not optional. *(perf C1/C2, architect M3)*

- **H — Cross-vendor heterogeneity is a first-class axis (NEW, from Decision #0).** Target hardware
  mixes ROCm (MI210) and CUDA (RTX 5090) on separate hosts with **asymmetric VRAM (64 GB vs 32
  GB)**. Implications:
  - **One process = one libtorch (one vendor).** cuda130 vs rocm71 are chosen per-process at
    launch (`cli/gpu_detect.rs`). Mixed-vendor is therefore *only* viable at the **inter-service /
    inter-host boundary** — which the pipeline design already is. Each host's InferenceService runs
    its own backend; **the hidden-state bytes crossing the wire are vendor-neutral** (BF16 numbers
    via safetensors, not device memory). This *reinforces* the inter-service architecture; do **not**
    attempt a single process spanning both vendors.
  - **ROCm presents as `Device::Cuda(i)` via HIP**, so the existing device indexing is already
    vendor-agnostic *within* a process — intra-host 2b on the MI210 pair needs no vendor-special
    code, only the device-map work.
  - **Capacity-aware stage assignment, not param-balanced.** The 64 GB/32 GB asymmetry means the
    pipeline planner must size stages by **available VRAM per host**, giving the MI210 more layers
    than the RTX 5090. Add this to the planner spec.
  - **Cross-vendor numerical consistency** must be validated: ROCm vs CUDA matmul/softmax/BF16
    handling differ slightly. Tolerable for sequential pipeline inference (no split-merge), but
    matters for TTT/delta reproducibility — add a cross-vendor numerics acceptance test.
  - **Interconnect caveat (supersedes "NVLink-class"):** intra-host MI210↔MI210 is **Infinity
    Fabric or PCIe**, not NVLink. Verify the 2b "microsecond hop" assumption on the actual MI210
    P2P path before relying on it.
- **M-BUBBLE — quantify pipeline utilization + confirm continuous batching exists.** Bubble
  fraction `(S−1)/(m+S−1)`; ~28 microbatches in flight for 90% util at S=4. Single-stream at S=4
  = 75% idle. The "throughput via microbatching" claim **requires continuous batching that may be
  net-new** — it's not in the gap table; add it as a deliverable or confirm it exists. *(perf C3)*
- **M-FLOW — backpressure is a correctness dependency, not tuning.** moq exposes no app-level
  flow control; under overflow/retention a dropped frame = **silent corruption of the
  hidden-state stream** (catastrophic, not graceful). Mesh activation transfer needs explicit
  bounded queues + a size/rate cap per job. *(architect C2, perf m-series)*
- **M-FAIL — multi-hop failure/recovery is unspecified.** Stage/host death mid-request, partial
  per-stage KV state, drain, head-of-line blocking on a pinned chain. Must be a 2c design
  deliverable. *(architect C1)*
- **M-LOAD — layer-subset loading is Medium, not "small."** `from_weights` hard-requires
  `embed_tokens`/`norm`/`lm_head` (middle stages have none → must become stage-aware with
  `is_first`/`is_last`); `validate_with_weights` checks vocab against `embed_tokens`; the layer
  `Vec` + per-layer `conv/rec/KV` vecs are sized to global `num_hidden_layers` → need a
  global↔local layer-index remap (`forward` indexes `conv_guard[idx]` by loop position). *(rust M1)*
- **M-WIRE — specify the hidden-state wire format + its real cost.** Reuse
  `serialize_state_dict_to_bytes` (safetensors); **device is NOT preserved** (receiver chooses) and
  every boundary is a **GPU→CPU→bytes→CPU→GPU bounce per token** — a fixed per-hop latency the
  "bandwidth negligible" line omits. Bytes cross threads/hosts; the `!Send` Tensor never does.
  *(rust C2)*
- **M-MESH-AUTHZ — `mesh:rpc`/`inference:peer-call` must be the narrowest grant, not a wildcard.**
  Scope to specific `service:inference:host-X` subjects with a non-wildcard cluster domain; split
  read (`query.status`) from authority (`infer.stage`, `delta.submit`) actions; add a deny-by-default
  test that `*`/`anonymous` never match. **Resolve the policy wildcard regression first.** *(security M1)*
- **M-ROUTER — the P1 router needs health/readiness + KV-cache session stickiness.** Round-robin
  breaks multi-turn KV reuse (sticky session→replica required); define the load metric; health,
  drain/failover, and basic metrics are **not deferrable**. *(scope M2, architect m5)*
- **M-DISCOVERY — be explicit that v1 discovery is a static reach roster.** pkarr dial-by-node_id
  is not wired into `dial_stream` (#282); P0's "dynamic registry" silently depends on operators
  publishing explicit reach. A static config roster is the acceptable v1 for a co-trusted fleet —
  say so. *(scope M3)*
- **M-CHUNK — large prefill activations need explicit chunking.** A 67 MB (8K×4096×2B) prefill
  activation exceeds the 64 MiB moq frame; bigger models/contexts reach GBs. `moq_stream.rs` emits
  **one frame per blob** today — chunking is net-new for prefill on the streaming plane. *(perf M1/M2)*

### Minor / endorsements
- **No-fragile-fallbacks sweep (rust M2):** mandate `config.json` + `index.json` **required**,
  fail-fast; validate `num_hidden_layers` against the weight_map. Kill the silent defaults
  `num_hidden_layers.unwrap_or(32)` / `hidden_size.unwrap_or(4096)` (`model_config.rs:214,217`) —
  more dangerous than a missing file. Make `Device::cuda_if_available()` **fail-fast** under a
  `strict_device` flag (a process told GPU 3 that silently lands on CPU tanks the pipeline).
  Flag mmap-optionality and index→glob fallback as silent-degrade paths.
- **KV interaction (perf M3):** per-stage KV is footprint-*neutral*, not a win (S× more concurrent
  requests to fill the pipe). `cached_token_ids` prefix-match is **model-wide** (stage-0 concern),
  contradicting the blanket "all cache state is per-layer." KV blocks are on a fixed device →
  thread device through `BlockPool`/`initialize_kv_registry` or a moved layer device-mismatches.
- **Interconnect (perf m1):** "microsecond hops free" holds for **NVLink**; PCIe-only intra-host
  boundaries for GB prefill activations are not free — qualify as "NVLink-class."
- **Stage-count balancing:** minimize boundaries (fewest stages that fit), don't maximize parallelism.
- **Endorsed:** deferring tensor parallelism (correct gate: only if a single *layer* won't fit one
  GPU); cluster-CA over the federation gate intra-PDS (correct shape — *with* C-IDENT guardrails);
  the clean-partition feasibility read for 2b (genuinely supported by the forward loop).

### Net revised sequencing
Decision #0 ANSWERED: north-star multi-TB; v1 = 35 GB on 2×MI210 homogeneous single host;
then cross-vendor (MI210+RTX5090) cross-machine. So 2c is the endgame (not conditional), and
heterogeneity (finding H) is a first-class axis from Phase 4 on.
```
Phase 1  Intra-host multi-GPU on the 2×MI210 host, ONE process, NO networking:
         2a replication (sticky-session aware)  → throughput
         2b intra-host pipeline (layer split)   → big-model muscle
            + TTT-on-split rides 2b (autograd reverse over the device_map)
         ── gate 2b on the C-SEND !Send multi-device spike ──
         ── validate 2b + TTT vs SINGLE-GPU ground truth (35 GB fits one MI210) ──
         ── verify MI210 P2P hop cost (Infinity Fabric/PCIe, not NVLink) ──
Phase 2  Cluster trust (per-host keys + roster, fail-closed) + mesh policy (narrow)
         + InferenceService network endpoint + static reach registry
Phase 3  ModelService → N InferenceServices pool + router (health + KV stickiness)
         → multi-host inference (1 proc/GPU/host); first cross-vendor cross-machine test
            (MI210 ROCm + RTX5090 CUDA; vendor-neutral hidden-state bytes; numerics test)
Phase 4  Inter-host pipeline (2c) — multi-TB endgame; capacity-aware stage assignment
         (64 GB vs 32 GB); requires M-FLOW + M-FAIL + M-CHUNK + AEAD + microbatching
Training (coupled, staged — NOT a walled-off epic):
   • TTT-on-partition: folded into Phase 1 (2b) and Phase 4 (2c backward)
   • Distributed training coordination (stageable, design-aware now):
       (a) delta aggregation across replicas — after Phase 2 trust; param-server +
           C-PROV provenance + C-AGG real (order-independent, rank-reconciling) reducer
       (b) inter-host pipeline-parallel training — backward comms across 2c stages (hardest)
   • Constraint: inference partitions must NOT preclude backward (no forward-only hand-offs).
```

### Cluster trust & enrollment — Phase 2 sub-spec (user-directed)

**Trust model (SPIFFE-forward; corrected — the PDS is multi-tenant with untrusted workloads).**
Co-location on a PDS is **not** a trust signal. Each node is a **workload identity (SVID-style)**,
`spiffe://<trust-domain>/inference/host-<id>`-shaped, issued by the PDS (the trust-domain authority
/ SPIRE-server analog = PolicyService). The PDS *also* issues identities to **untrusted workloads**
(tenant sandboxed apps, agent tool-calls, other tenants' models — the Kata/Wanix sandboxes). So:
- The **inference mesh is a named trust *group*** within the PDS, **not** the whole PDS. `mesh:rpc`/
  `inference:peer-call` is granted **only** to the specific inference-node SVID paths, **deny-by-default
  for every other workload identity** — least-privilege per workload, never "any SVID from this PDS."
- **AEAD + provenance (C-AEAD/C-PROV) must hold against same-PDS adversaries**, not just external
  peers — a co-tenant untrusted workload on the same host/PDS is in the threat model.
- **Tenant isolation:** a node serving tenant A must not be reachable via `mesh:rpc` by tenant B's
  workload; the router/ModelService enforces tenant→node scoping.
- **SPIFFE interop:** SVID (JWT-SVID/X.509-SVID) ↔ the did:web/did:key + COSE identity; SPIFFE
  trust-domain federation (bundle endpoint) ↔ atproto/did:web federation. SPIFFE-native, bridged.

**Enrollment reuses existing machinery:** the PDS is already the OAuth AS *and* the issuing
authority (PolicyService). Enrolling a node into the inference trust group is a **privileged,
admin-gated** operation (distinct from a tenant workload getting an ordinary SVID). No new trust
primitive. Resolves C-IDENT + the security review's "enrollment ceremony undefined" (M3).

**Two layers, two interfaces (there is NO "cluster" object to join — user correction).** A "cluster
join" command was the wrong literal interface; it conflated identity onboarding with an authz grant.
Split them:

- **Layer 1 — host attaches to a PDS (identity).** The real onboarding: the host registers its
  self-generated identity (did:key derived from its iroh `node_id`) with a **home PDS** and obtains a
  PDS-scoped, **PoP-bound** credential (DPoP/`cnf` or the node_id) so it can't be replayed. This is
  *"join a PDS,"* and **every** workload that participates does it — including untrusted tenant
  workloads. Literal interface: **part of `hyprstream wizard`** (choose PDS, authenticate, enroll);
  a thin `hyprstream pds attach <url>` (+ headless one-time token) covers non-wizard / re-enroll /
  GPU-box cases. The PDS records the identity in its **workload registry** (identity layer — *all*
  known workloads, not a mesh roster). One **home** PDS per host (its trust-domain authority);
  reaching *other* PDSes is **federation**, not multi-home.
- **Layer 2 — mesh membership is a POLICY grant on top (authz).** Whether an attached identity is in
  the **inference trust group** is a Casbin/PolicyService decision: grant `mesh:rpc` /
  `inference:peer-call` to that specific host SVID, deny-by-default for every other workload. **No
  bespoke ceremony** — it's `hyprstream policy …` / a policy template / an admin grant. The "cluster"
  is **emergent from policy**, not a first-class joinable thing. Revocation = revoke the grant (and/or
  the credential). This is where multi-tenant least-privilege + tenant→node isolation live.

**Net:** drop `cluster join`. Onboarding = *attach to a PDS* (identity, via wizard/`pds attach`);
mesh membership = *a policy grant* (authz). Both reuse existing machinery — the PDS is already the
OAuth AS + DID authority + PolicyService; neither layer is a new trust primitive.

**Authn/authz on the mesh:** iroh channel binds the `node_id`; re-verify the app-layer envelope
signer key per call (admission seam, #200 for QUIC/WT — so **pin cluster mesh to iroh** meanwhile).
`mesh:rpc`/`inference:peer-call` granted to the **specific host subject** in the cluster domain,
least-privilege, deny-by-default, fail-closed.

**Decisions — RESOLVED (2026-06-18):**
- **Onboarding (D2): device-code OAuth first, pre-auth token for unattended.** Interactive OAuth via
  **RFC 8628 device-code** (the Tailscale `tailscale up` model) — CLI prints URL+code, admin approves
  elsewhere; works headless when *attended*. **Pre-auth token** (Tailscale "auth key") for *unattended*
  provisioning: admin mints a short-lived, single-use, scoped token; host redeems at
  `pds attach --token <key>`; PDS validates (unused+unexpired) → issues PoP-bound credential → marks
  consumed. Ship device-code first; add pre-auth tokens for fleet automation.
- **Interfaces (D3): CLI `hyprstream pds join/attach <url>` AND a wizard flow**, both first-class.
- **Authz staging (D3): minimal / semi-open for now — assume the `federation-open` policy.** Get the
  attach + mesh mechanism working under open policy in the homelab phase; defer strict per-host
  least-privilege. **Caveat:** `federation-open` + the unfixed `ClusterKeySource` (C-IDENT) is
  fail-unsafe — strict per-host mesh authz **and** the ClusterKeySource fix MUST land before any
  internet-exposed or genuinely-untrusted-tenant deployment. Fine for 2–3-box homelab dev only.
- **Trust-domain shape (D4): one trust domain per PDS, path-based isolation.** This is what *enables*
  federation, not a limitation: SPIFFE federation is **between** trust domains. PDS↔PDS = trust-domain
  federation via bundle endpoints — and the **did:web / DID document already plays the bundle-endpoint
  role**, with the **#137 admission gate as the cross-domain trust decision**. Bridge the names:
  **SPIFFE trust-domain ↔ PDS DID / `at://` handle**. Inner ring (path isolation within a domain) and
  outer ring (bundle/DID federation across domains) compose.

### Overall verdict (consensus)
Feasibility analysis is credible and the "clean partition" insight is real. Phases 1–2a and the
intra-host pipeline are safe to start (2b pending the `!Send` spike). The plan's weak points are
**packaging** (front-loads trust/routing before the GPU win; bundles a net-new training epic) and
**treating the two genuinely hard distributed-systems problems — multi-hop failure recovery and the
absence of backpressure — plus the security trust/provenance gaps, as out-of-scope rather than as
the gating questions they are.** Resolve C-GATE0 first; it may delete 2c entirely.
