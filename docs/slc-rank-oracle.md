# SLC-Guided Runtime Rank Oracle

## Overview

This document describes the runtime rank adaptation system added to hyprstream's Test-Time Training (TTT) subsystem. The system uses Selective Linearization Characterization (SLC) principles to dynamically adjust LoRA rank allocation per layer during inference, starting from the static per-layer ranks produced by a TTN layer profile.

Three components work together:

1. **Muon optimizer** — replaces AdamW for LoRA parameter updates
2. **Narrow-based effective rank** — runtime rank control without tensor reallocation
3. **Per-layer gradient gating** — freezes low-signal layers during multi-step TTT

The static ranks these operate on come from a **TTN layer profile** (see [TTN Layer Profiles](#ttn-layer-profiles)) — a per-layer `recommended_rank` derived either from ablation ground truth (embedded models) or a flat unvalidated cap (everything else). A runtime `RankUtilizationTracker` then narrows the *effective* rank below that ceiling in response to observed utilization.

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
| **3** | Computed from weight-key geometry (`compute_layer_profile`) | negligible, then cached | Unvalidated — flat prior (see below) |

### Flat prior — no per-layer spectral allocation

`UNVALIDATED_RANK_CAP = 8` is the rank assigned to every attention/standard layer in a Tier-3 (computed) or Tier-2 (cached) profile without ablation ground truth; GatedDeltaNet layers get `GDN_LORA_RANK = 4`. The runtime utilization oracle then narrows the effective rank below these ceilings from observed utilization. Pre-cap or hand-edited caches that previously held larger ranks are clamped to the cap on load. (A model with an embedded Tier-1 profile returns before the Tier-2/Tier-3 path, so embedded ranks are untouched.)

There is intentionally **no per-layer spectral allocation**. A weight-bond spectral-entropy → rank mapping (`entropy_to_rank` + hand-fitted cutoff constants) previously lived here and was removed as unvalidated:

- The linearization-resistance theory in the underlying research is about *attention-pattern* entropy, not weight-bond entropy from SVD — a different quantity the research never connected to adaptation rank.
- On the research's own n=6 attention layers, the sign of the entropy↔importance correlation is projection-dependent: it holds for `v_proj` (r ≈ −0.73), is absent for `k_proj` (r ≈ −0.13), and *inverts* for `q_proj` (r ≈ +0.31).
- The cutoff constants were hand-fitted to a single ablated reference model.

After the cap, the mapping only ever distinguished rank 8 from 4 with an unreliable sign — a flat prior is more honest, the oracle adjusts at runtime, and first-inference SVD cost (~2–5s) is eliminated. A data-calibrated allocator is gated on the rank-proxy validation spike (#842 → #844), which must test any candidate per-projection on more than one model.

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
