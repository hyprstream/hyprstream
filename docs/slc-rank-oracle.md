# SLC-Guided Runtime Rank Oracle

## Overview

This document describes the runtime rank adaptation system added to hyprstream's Test-Time Training (TTT) subsystem. The system uses Selective Linearization Characterization (SLC) principles to dynamically adjust LoRA rank allocation per layer during inference, replacing the static rank assignments from TTN profiles.

Four components work together:

1. **Muon optimizer** — replaces AdamW for LoRA parameter updates
2. **Narrow-based effective rank** — runtime rank control without tensor reallocation
3. **Per-layer gradient gating** — freezes low-signal layers during multi-step TTT
4. **W-SLC online rank allocator** — budget-constrained rank redistribution algorithms

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

**Memory overhead:** ~16MB for Qwen3.5-0.8B (72 modules × 28 extra rank × 2048 dims × 4 bytes). Negligible.

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

## W-SLC Online Rank Allocator

**File:** `crates/hyprstream/src/runtime/ttn_profile.rs`

Ported from `qonk/experiments/ttn-transformer/src/wasserstein_slc.py`. Three algorithms operate under a per-step budget constraint: `sum of ranks ≤ median(rank_levels) × num_layers`.

### Algorithms

**Greedy:** Assigns closest available rank to each layer's entropy, then iteratively reduces the highest-rank layer until budget is met. Fast, online-capable, good for streaming.

**Balanced:** Averages entropy across time, assigns uniform allocation. Best when entropy is stable across requests.

**Offline Optimal (DP):** Dynamic programming over the full budget space. Minimizes total |entropy - rank| cost. Requires integer budget. O(T × L × B × R) where B = budget, R = number of rank levels.

### Rank Utilization Tracker

A rolling-window tracker (`RankUtilizationTracker`) records per-key utilization values and produces adaptation signals:

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
| `runtime/ttn_profile.rs` | Allocator port, utilization tracker |
| `config/mod.rs` | TTTTrainingConfig fields |
| `services/inference.rs` | Service wiring |
