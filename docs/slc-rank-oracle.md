# SLC-Guided Runtime Rank Oracle

## Overview

This document describes the runtime rank adaptation system added to hyprstream's Test-Time Training (TTT) subsystem. The system uses Selective Linearization Characterization (SLC) principles to dynamically adjust LoRA rank allocation per layer during inference, starting from the static per-layer ranks produced by a TTN layer profile.

Three components work together:

1. **Muon optimizer** — replaces AdamW for LoRA parameter updates
2. **Narrow-based effective rank** — runtime rank control without tensor reallocation
3. **Per-layer gradient gating** — freezes low-signal layers during multi-step TTT

The static ranks these operate on come from a **TTN layer profile** (see [TTN Layer Profiles](#ttn-layer-profiles)) — a per-layer `recommended_rank` derived either from ablation ground truth (embedded models) or from a conservative, capped spectral-entropy estimate (everything else). A runtime `RankUtilizationTracker` then narrows the *effective* rank below that ceiling in response to observed utilization.

> The W-SLC streaming rank allocator (a Python research port with a dimensionally-inconsistent `|rank − entropy|` cost term and no production call-sites) was removed in the #202 cleanup; the active W-SLC research lives in the separate Python research repo and is unaffected. Runtime rank redistribution is now solely the utilization-driven oracle below.

## Muon Optimizer

**File:** `crates/hyprstream/src/training/muon.rs`

TTT operates under a 1–5 gradient step budget per request. AdamW needs many steps for its moment buffers (m, v) to warm up. Muon orthogonalizes each gradient update independently via Newton-Schulz iteration — no warmup needed.

**Algorithm per step:**
1. Nesterov momentum accumulation
2. Newton-Schulz orthogonalization (Polar Express variant, 5 iterations)
3. Decoupled weight decay
4. Rectangularity-scaled update

Newton-Schulz maps the gradient to its orthogonal factor UV^T, equalizing all singular values to 1. For LoRA matrices (typically [8, 2048]), this operates on an [8, 8] inner product — microsecond cost.

**Key parameters:**
| Parameter | Default | Notes |
|-----------|---------|-------|
| Learning rate | 0.02 | ~67x larger than AdamW's 3e-4 |
| Momentum | 0.95 | Nesterov momentum coefficient |
| NS iterations | 5 | Polar Express precomputed coefficients |
| Weight decay | 0.0 | Rarely needed for Muon-optimized params |

**Why Muon fits LoRA:** All TenantDelta parameters are 2D matrices (A: `[rank, in_features]`, B: `[out_features, rank]`). Muon is designed exclusively for 2D weight matrices. No AdamW fallback is needed within the delta.

## Narrow-Based Effective Rank

**File:** `crates/hyprstream/src/training/tenant_delta.rs`

### Problem

Physically resizing LoRA tensors requires rebuilding the VarStore and optimizer, destroying momentum state for all layers. With only 1–5 gradient steps per TTT request, losing momentum is catastrophic.

### Solution

Allocate all LoRA pairs at `max_rank` from the TTN profile. Store a per-key `effective_rank` that controls `.narrow()` during forward/backward:

```
A: [max_rank, in_features]  →  A.narrow(0, 0, eff_rank): [eff_rank, in_features]
B: [out_features, max_rank]  →  B.narrow(1, 0, eff_rank): [out_features, eff_rank]
```

**Properties:**
- No VarStore surgery — same tensors, narrowed views
- Optimizer momentum preserved — unused dimensions get zero gradients, momentum decays naturally
- Reversible — widening the narrow recovers dormant dimensions
- Saves compute — narrower matmuls when effective rank < max rank
- Scaling auto-adjusts — `set_effective_rank()` recalculates `alpha / eff_rank`

**Memory overhead:** ~16MB for a ~0.8B hybrid model (72 modules × 28 extra rank × 2048 dims × 4 bytes). Negligible.

## Per-Layer Gradient Gating

**File:** `crates/hyprstream/src/training/ttt.rs`

After the first gradient step, layers with gradient L2 norm below a threshold are frozen via `requires_grad_(false)` for subsequent steps. This:

1. Prevents gradient computation entirely (saves backward-pass FLOPs)
2. Prevents momentum buffer updates (no gradient → no momentum change)
3. Prevents weight decay from drifting frozen parameters

All parameters are restored to `requires_grad_(true)` after the TTT call completes.

**Configuration:**
| Parameter | Default | Notes |
|-----------|---------|-------|
| `enabled` | true | Toggle gradient gating |
| `min_grad_norm` | 1e-5 | L2 norm threshold for freezing |
| `warmup_steps` | 1 | Steps before gating activates |

## TTN Layer Profiles

**File:** `crates/hyprstream/src/runtime/ttn_profile.rs`

A TTN layer profile gives every layer a `recommended_rank` — the `max_rank` the narrow-based effective-rank mechanism allocates and the oracle narrows from. Profiles resolve through a three-tier lookup in `get_layer_profile()`, in order:

| Tier | Source | Cost | Trust |
|------|--------|------|-------|
| **1** | Embedded profile for a known model family/geometry (`find_embedded_profile`) | const lookup | **Gold standard** — ranks from ablation perplexity delta |
| **2** | Cached profile on disk (`load_cached_profile`) | ~1ms | Treated as a persisted Tier-3 result |
| **3** | Computed from weight spectra (`compute_weight_entropy_profile`) | ~2–5s SVD, then cached | Unvalidated — capped (see below) |

### The unvalidated-rank cap

`UNVALIDATED_RANK_CAP = 8` is the ceiling on any rank not backed by ablation ground truth — Tier-3 computed ranks, the uniform fallback, and Tier-2 cache loads alike. Pre-cap or hand-edited caches that previously bypassed it are clamped on load. The cap is the same value as the uniform fallback on purpose: without validation, a spectral-entropy rank is not trusted to claim more capacity than the uninformed default. (A model with an embedded Tier-1 profile returns before the cache path, so embedded ranks are never clamped.)

### Why spectral entropy is a weak proxy (and treated as such)

`entropy_to_rank` maps normalized weight-bond entropy to rank on the *assumed* premise that lower entropy (more structured spectrum) means a layer needs more adaptation capacity. That premise is **unvalidated**:

- The linearization-resistance theory in the underlying research is about **attention-pattern** entropy, not weight-bond entropy from SVD — a different quantity the research never connected to adaptation rank.
- On the research's own n=6 attention layers, the sign of the entropy↔importance correlation is projection-dependent: it holds for `v_proj` (r ≈ −0.73), is absent for `k_proj` (r ≈ −0.13), and **inverts** for `q_proj` (r ≈ +0.31).
- The cutoff constants (`ENTROPY_CUTOFF_RANK_{32,16,8}`) were hand-fitted to a single ablated reference model and are not expected to generalize.

This is why the cap exists and why `entropy_to_rank`'s doc comment flags the sign as assumed. A continuous, data-calibrated mapping is deferred to the rank-proxy validation spike (#842 → #844), which must test the sign per-projection on more than one model.

### GatedDeltaNet layers

GDN layers receive a fixed `GDN_LORA_RANK = 4` rather than an entropy-derived rank. GDN adaptation capacity lives in the recurrent/conv/gating parameters, which the weight-spectrum pipeline does not model; deriving a rank from `out_proj`'s spectrum alone presented a guess as analysis. (This matches the embedded GDN rank and sits under the cap.) No SVD is computed for GDN layers — they are the majority of a hybrid stack, so skipping them is also the dominant profiling-speed win.

### Embedded profile honesty

The embedded profile carries `perplexity_delta` per attention layer (the consumed, ablation-derived input to `recommended_rank`) but its `bond_entropy` maps are intentionally empty. Weight-bond entropies were never validated against rank and the originally-transcribed values did not match the research source; an empty map is more honest than unconsumed, unvalidated numbers in a "gold standard" profile.

### Rank Utilization Tracker

A rolling-window tracker (`RankUtilizationTracker`) records per-key utilization values and produces adaptation signals. It narrows the *effective* rank below the profile ceiling at runtime:

| Signal | Condition | Action |
|--------|-----------|--------|
| Increase | mean utilization > 0.85 | Double effective rank (clamped to max) |
| Decrease | mean utilization < 0.25 | Halve effective rank (clamped to 1) |
| Hold | between thresholds | No change |

## Rank Oracle Integration

**File:** `crates/hyprstream/src/training/ttt.rs`

The `RankOracle` is per-tenant (stored in `TenantDelta`, not in `TestTimeTrainer`) because:
- Different tenants have independent utilization histories
- `TestTimeTrainer` passes `&self` to `adapt_tenant` — can't mutate shared state
- Mixing signals across tenants would produce meaningless recommendations

**Feedback loop:**
1. After each TTT adaptation, collect per-key utilization via `delta_rank_utilization()`
2. Feed to oracle: `oracle.observe(&utilizations)`
3. Every `adaptation_interval` observations, evaluate signals
4. If `auto_adapt` is enabled, adjust effective ranks on the delta
5. Log rank changes via tracing

**Configuration (`TTTTrainingConfig`):**
```toml
[ttt.rank_oracle]
adaptation_interval = 10
auto_adapt = false          # conservative: log-only by default
rank_levels = [1, 2, 4, 8]
low_utilization_threshold = 0.25
high_utilization_threshold = 0.85
```

## Architecture Diagram

```
Request
  │
  ▼
TestTimeTrainer::adapt_tenant(&self, delta)
  │
  ├─ Step 1: Forward + backward (all layers)
  │    └─ Uses narrow-based effective rank in forward()
  │
  ├─ Gradient gating: freeze low-signal layers
  │
  ├─ Steps 2..N: Forward + backward (unfrozen layers only)
  │
  ├─ Muon optimizer step (per trainable param)
  │    └─ Newton-Schulz orthogonalize → scale → update
  │
  ├─ Collect rank utilization
  │    └─ delta.rank_oracle.observe(utilizations)
  │
  └─ If oracle.should_evaluate() && auto_adapt:
       └─ Adjust effective_ranks per signal
```

## Files Changed

| File | Change |
|------|--------|
| `training/muon.rs` | New — Muon optimizer with Newton-Schulz |
| `training/tenant_delta.rs` | Muon integration, effective rank, oracle field |
| `training/ttt.rs` | Gradient gating, rank oracle, optimizer wiring |
| `training/mod.rs` | Re-exports |
| `training/delta_pool.rs` | Oracle config propagation |
| `runtime/ttn_profile.rs` | Layer profiles, utilization tracker (W-SLC allocator port removed in #202 cleanup) |
| `config/mod.rs` | TTTTrainingConfig fields |
| `services/inference.rs` | Service wiring |
