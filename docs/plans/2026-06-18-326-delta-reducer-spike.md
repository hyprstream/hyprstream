# Spike #326 — param-server delta aggregation + a real reducer (replace DO-Merge)

**Date:** 2026-06-18 · **Status:** research spike (no code) · grounded in repo code. Answers the
human's questions: viability, layer-resize cost, mixed/variable (DeepSeek-V4) quant interaction.

## Code ground truth
- **DO-Merge** (`training/merge.rs`): `do_merge()` (:127) = Frobenius-mag + unit-direction interp;
  `additive_merge` (:99) + `do_merge` (:128) assert shape-equality → **hard-fail on rank-divergent
  layers**. Operates on raw LoRA A/B factors element-wise per key. **Non-associative** (path-dependent
  blend weights + direction renorm). Caller `inference.rs:1657 handle_save_adaptation` = **1-vs-1
  load-modify-store at save time**; NO param-server / all-reduce / N-way reducer anywhere.
- **Delta representation** (`training/tenant_delta.rs`): separate low-rank factors `lora_a[r,in]`,
  `lora_b[out,r]`, keyed `"layer.module"` (:183); **always FP32** (:432, for Muon/TTT precision);
  pure **delta-space**, applied additively `base.apply(x)+delta.forward_2d(x)` (llama.rs:428,
  qwen3_5.rs:732); **never folded into base**.
- **Rank does NOT resize tensors:** A allocated once at `max_rank` (:312); rank change = narrow a
  **view** (`a.narrow(0,0,eff)`, :457); `set_effective_rank` never reallocs (:628). Cross-tenant/
  worker `max_rank` *can* differ (per-profile `layer_overrides`) — that's what breaks a naive merge.
- **Rank-mismatch reconcile ALREADY EXISTS:** `compose()` (:646) sums factor-space; on rank mismatch
  falls back to dense `B·A` reconstruction (:707) — the seed of the right N-way reducer.
- **Rank oracle** (`ttt.rs`, `runtime/ttn_profile.rs`): B-spectral utilization (Gram trick), per-tenant;
  `auto_adapt` **off by default**, evaluated every 10 adaptations → **not round-churny**. Offline
  `entropy_to_rank` → {32,16,8,4} per-layer overrides via `from_profile()` (tenant_delta.rs:113).
- **Variable quant:** block-wise FP8 (DeepSeek 128×128) is REAL — `weight_scale_inv`
  (`model_factory.rs:540`), lazy 4D-broadcast dequant at matmul (`llama.rs:62–77`). Precision varies
  per layer **as a checkpoint property** — there is **no importance-driven quant policy** in-repo. Base
  is **never requantized after a delta** (only KV-cache requant at kv_cache.rs:1071, unrelated).

## Answers
- **Viable:** yes. Reusable: delta-space repr, serialization, `compose()` reconcile kernel,
  `delta_norm_ratio` (:473) as norm-bound gate. Net-new: N-way reducer, param-server transport +
  provenance/signing (C-IDENT/C-PROV), delta-validation gate (norm+anomaly+held-out eval), promote-to-git.
- **Resize cost:** NOT per-merge — per-tenant resize already avoided (narrow views). Pure reducer
  design choice. **Recommended reducer = stack contributors' (A,B) in delta-space (concat, O(N·r·d),
  no resize) → ONE truncated/randomized SVD per round to a FIXED rank R.** Cost = 1 SVD/round/layer on
  tall-skinny `[Σr,d]`, order-independent, rank-heterogeneity-tolerant. (vs dense `ΣBᵢAᵢ` = O(N·r·d²)
  mem; vs element-wise mean = needs uniform rank, the DO-Merge failure.)
- **Quant interaction:** orthogonal during aggregation (deltas FP32 delta-space; base FP8 only a
  transient dequant view). Couples only at (1) apply-time noise floor → cap delta rank on coarse-quant
  layers; (2) promote-time IF folding into quantized base (not done today; defer fold+requant to
  checkpoint, per 128×128 block, NEVER per-merge). DeepSeek-V4 block quant forces nothing during merge.
- **Right questions?** Yes. Reframe: resize = reducer-design choice, not per-merge cost. Missing Qs:
  aggregation-locus (delta-space every round, fold/promote rarely); fixed R as operational contract;
  provenance/norm weighting folded INTO stack weights; held-out-eval host = the real long pole; drop
  Muon momentum at aggregation (averaging optimizer state across independent TTT runs is dubious).

## HUMAN DECISION MENU (#326 — training coordination; resolve when #326 starts)
1. **Reducer algo:** (a) **stack + truncated-SVD in delta-space [REC]**; (b) dense ΣBᵢAᵢ then re-SVD
   (O(d²) mem); (c) element-wise mean (rejected — needs uniform rank).
2. **Reducer rank:** (a) **fixed R [REC — stable checkpoint shapes]**; (b) adaptive (churn).
3. **Aggregation locus:** (a) **aggregate delta-space, fold only at promote [REC]**; (b) fold per-round
   into base (rejected — compounding requant, expensive).
4. **Rank↔quant coupling:** (a) **static cap: clamp oracle rank by stored dtype in from_profile() [REC]**;
   (b) live in auto_adapt (rejected); (c) none (wastes rank on FP8 layers).
5. **Provenance/norm (C-PROV/C-AGG):** (a) **gate+weight contributors via signed host id +
   delta_norm_ratio + held-out eval, folded into stack weights [REC]**; (b) merge-then-validate
   (rejected). Held-out-eval host decided separately (long pole).
6. **Optimizer state:** (a) **drop Muon momentum, fresh optimizer post-checkpoint [REC]**; (b) average
   (dubious).

Depends on Phase-2 cluster trust (#328 identity) for provenance. Sequenced after Phase 2 (delta
aggregation across replicas) per the plan; #327 (inter-host pipeline-parallel *training*) is the harder
follow-on coupled to 2c.
