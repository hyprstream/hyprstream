# Spike #329 — continuous / in-flight batching (design)

**Date:** 2026-06-18 · read-only design spike · branch `ewindisch/310-multi-gpu` @ `582b73562`.
Net: continuous batching is ABSENT (decode hardwired batch=1); Phase-1 throughput win + 2c
prerequisite. Entire core is **pure-Rust, no capnp/auth gate** (unless scheduler knobs go on the wire).

## Chokepoints (verified file:line, under crates/hyprstream/src/)
- batch=1 forward: `.unsqueeze(0)` `runtime/torch_engine.rs:923` (+ :971,:1016,:1051,:1820); KV budget
  `let batch_size=1` `:575`; per-stream decode `TextStream::poll_next:2660`, `sample_next_token:2403`
  → `forward_cached(&[last],pos)`; per-request `TextStream` `:1488,:1509`.
- **Hardest blocker — inline causal mask** (both archs, manual matmul/softmax; SDPA panics on this
  ROCm build per `llama.rs:586`): llama `compute_attention:797`, mask `:831` (`tril`), chunk `:855`;
  qwen3_5 attn `:881`, mask `:999`, chunk `:1007`. `position_ids = arange(start_pos..)` is one scalar
  for the whole batch (`llama.rs:2008`, qwen3 `apply_rope:957`). KV `update/get` take scalar
  `start_pos` (`kv_cache.rs:1111,1320`).
- paged `BlockPool` `kv_cache.rs:436` (allocate:495/free:500, blocks `[1,256,kv_heads,dim]`);
  `new_paged:964/1789`; `paged:false:107`; **dead — `init_block_pool:185` + `with_paged(true)` have
  zero live callers**. Gaps: **no Drop → block leak on registry evict (`:336` no clear)**; hard
  "pool exhausted" `:1295` (no preempt); `paged+quantized` unsupported `:977`; **`get()` reassembles
  dense via `cat` `:1408` — no paged-attention gather kernel**; single pool Mutex per-layer-per-step.
- engine thread model: `InferenceServiceInner.engine: RwLock<TorchEngine>` `services/inference.rs:129`;
  pinned to one thread via `LocalServiceBridge` actor (`iroh_rpc.rs:394`, current-thread rt + LocalSet,
  requests as `BridgeMessage` over mpsc, `spawn_local` per req). Today: read-lock held across awaits →
  N independent batch=1 forwards, no queue/admission/batching. Global `active_cache_owner`
  (`torch_engine.rs:81`) is single-sequence. Each request carries `subject`(tenant)+`delta`
  (`inference.rs:75`).
- **Two missed constraints:** (1) per-seq LoRA delta injected inside attention (llama `:479`, qwen3
  `:902`) → multi-LoRA batched matmul is hard → **v1 batches same-delta (incl None) only**; (2) qwen3_5
  hybrid SSM per-seq `conv_state`/`rec_state` (`:586`) → **Llama-only first PR**.

## Design
- **Scheduler = actor on the bridge thread** (`Rc<RefCell<Scheduler>>` in the LocalSet). Per-request
  spawn_local enqueues a `SequenceHandle` (Send: tokens, sampling params, subject, delta Arc, mpsc
  chunk sender) and awaits chunks; one driver task owns the engine, runs step loop: pick ready batch →
  one batched forward → scatter sampled tokens per channel. Tensors never cross thread (!Send safe).
  Retires `active_cache_owner` global.
- **Batched ragged forward:** thread an explicit additive `[B,1,q,kv]` mask into attention (replace
  inline tril) + per-row `[B,seq]` position_ids (apply_rope already takes a tensor). v1 KV: read each
  row's dense (k,v), pad to batch-max, stack `[B,kv,heads,dim]`, mask pad; per-row update after step.
  New batched entry on `forward_cached/forward_varstore` skipping `.unsqueeze(0)`; keep batch=1 paths
  as reference for equivalence tests.
- **Prefill/decode interleave:** chunked prefill (CHUNK=1024 exists) mixed into decode batch (needs the
  ragged mask); v1 simpler "prefill-then-join". Token budget bounds per-step work.
- **forward_layers (#314) compat:** same mask+positions params thread through unchanged — keep
  signatures batch/mask-ready so a later microbatch scheduler feeds pipeline stages (M-BUBBLE). Don't
  implement now.

## Phased plan
- **PR-0** (pure-Rust prereq): paged-pool Drop/release + fail-soft alloc + per-owner ownership assert.
- **PR-1** (the win, Llama-only, config-gated OFF): scheduler-on-bridge-thread + batched decode +
  explicit mask + per-row positions + padded-dense KV; same-delta batching; batch=1 kept as reference.
- **PR-2**: chunked prefill / mixed-q_len join.
- **PR-3**: qwen3_5 SSM-state stacking + multi-LoRA grouped gather + true block-table paged kernel.
- **PR-4**: wire batched mask/positions through forward_layers for pipeline microbatching (2c).

## Gate: CPU equivalence test — batched(N seqs) logits == serially-run(each seq) logits, ragged lengths
(attention is plain matmul/softmax, runs on Device::Cpu; model after the existing
`forward_layers_full_range_matches_whole_model_forward` tests `llama.rs:2831`, qwen3 `:2598`).

## HUMAN DECISION MENU
1. Scheduler policy: **FCFS v1** [rec] vs priority vs fair-share (Subject weighting available later).
2. Max batch / token budget: config default (e.g. 16 / ~2048), tunable.
3. Paged-pool default: **off until PR-0 lands** [rec] (padded-dense works meanwhile).
4. Exhaustion: **admission-defer v1** [rec] vs preempt-recompute.
5. **Config-flag gate the whole scheduler, batch=1 fallback** [strongly rec].
6. First-PR arch: **Llama-only** [rec] vs Llama+qwen3_5 (SSM harder).
7. Multi-tenant batching: **same-delta v1** [rec] vs grouped multi-LoRA later.

Keep v1 entirely internal → no capnp/RPC/auth/policy changes. Human-gated ONLY if priority/fair-share
fields are added to the capnp GenerationRequest (`inference.rs:1873`) + policy enforcement.
