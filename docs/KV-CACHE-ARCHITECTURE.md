# KV Cache Architecture

How hyprstream caches attention key/value states across autoregressive
generation, concurrent sessions, and TTT adaptation. The implementation lives
in `crates/hyprstream/src/runtime/kv_cache.rs`, with the prefix-caching prefill
path in `crates/hyprstream/src/runtime/torch_engine.rs`.

## Overview

KV caching avoids recomputing past key/value states during inference (10-50x
speedup for long sequences). The design supports:

- Concurrent inference sessions, each with an isolated cache
- Session-persistent caches with cross-turn prefix reuse
- Memory budgets with two-tier eviction (GPU → CPU offload → removal)
- Optional blockwise quantization (bitsandbytes) and paged block storage
  (PagedAttention-style)
- TTT integration: caches computed under a tenant's LoRA delta are invalidated
  when that delta changes

## Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                       KVCacheRegistry                        │
│  caches: DashMap<CacheOwner, Arc<Mutex<KVCacheManager>>>     │
│  delta_dependencies: DashMap<CacheOwner, Option<String>>     │
│  block_pool: Option<Arc<Mutex<BlockPool>>>                   │
│  memory_budget_bytes: Option<usize>                          │
└───────────────┬──────────────────────────────────────────────┘
                │ get_or_create(owner)
    ┌───────────┼────────────────┐
    ▼           ▼                ▼
 Session("abc") Stateless(42)  Training{adapter, run_id}
 KVCacheManager KVCacheManager KVCacheManager   (each behind its own Mutex)
    │
    └── layer_caches: DashMap<usize, LayerKVCache>   (one per transformer layer)
            └── KVStorage::{FullPrecision | Quantized | Paged}
```

### CacheOwner (`kv_cache.rs:46`)

Every cache is keyed by its owner, giving session isolation by construction:

```rust
pub enum CacheOwner {
    /// Conversational sessions — cache persists across requests for context reuse.
    Session(String),
    /// One-off completions — cache is discarded on release.
    Stateless(u64),
    /// Training validation inference during a TTT run.
    Training { adapter: String, run_id: u64 },
}
```

### KVCacheRegistry

The registry maps owners to caches via `DashMap` (lock-free across *different*
owners), while each individual `KVCacheManager` sits behind a
`parking_lot::Mutex` — required because tch-rs `Tensor` holds raw pointers and
is not `Sync`. Two requests on different sessions never contend; two requests
on the *same* session serialize on that session's mutex.

`registry.get_or_create(owner)` is the single entry point:

- **Fast path** — an existing cache is touched (LRU timestamp) and, if it was
  offloaded to CPU, transparently restored to GPU before being returned.
- **Slow path** — a new `KVCacheManager` is created, paged (from the shared
  `BlockPool`) or contiguous depending on the default `CacheConfig`.

`release(owner)` removes `Stateless` caches immediately; `Session` and
`Training` caches are kept for reuse and left to budget-driven eviction.

### KVCacheManager

Per-owner manager holding one `LayerKVCache` per transformer layer
(`DashMap<usize, LayerKVCache>`), plus:

- `cached_token_ids: Vec<i64>` — the token sequence this cache was computed
  for, recorded after each generation so the next turn can prefix-match
- `last_access_ms` / `access_count` — LRU bookkeeping (atomics)
- `location: CacheLocation::{Gpu, Cpu}` — where the tensors currently reside

`truncate_to(pos)` discards KV entries past a token position (used when a
prefix matches but the suffix changed); `offload_to_cpu()` /
`restore_to_gpu(device)` move all layer tensors between devices while
preserving contents.

## Storage Modes (`KVStorage`)

Each `LayerKVCache` owns one of three storage variants. In all modes, capacity
grows in chunks (`DEFAULT_CHUNK_SIZE = 1024` tokens) rather than pre-allocating
`max_seq_len`, saving multiple GB of VRAM for large context windows.

### FullPrecision (default)

Plain FP16/BF16 `keys`/`values` tensors, grown chunk-wise.

### Quantized (`bnb` feature)

Hybrid blockwise quantization via bitsandbytes (`QUANT_BLOCKSIZE = 64`),
supporting `Int8`, `Nf4`, and `Fp4`:

- **Historical tokens** are stored quantized (memory savings)
- **Recent tokens** accumulate in a full-precision buffer and are flushed to
  quantized storage in batches (`BUFFER_FLUSH_THRESHOLD = 64` tokens)
- A **dequantized view is cached** per forward pass so each layer does not
  re-dequantize the same history
- On CUDA/ROCm with BF16/FP16 + Int8, quantization runs on GPU-native
  bitsandbytes kernels — no CPU round-trip

Without the `bnb` feature, quantization requests log a warning and fall back to
full precision.

### Paged (`KVStorage::Paged`)

PagedAttention-style block storage: fixed-size blocks (`BLOCK_SIZE = 256`
tokens, shape `[1, BLOCK_SIZE, num_kv_heads, head_dim]`) allocated from a
shared, pre-allocated `BlockPool` on the registry
(`registry.init_block_pool(...)`). This eliminates GPU memory fragmentation —
total KV capacity is fixed at pool creation, and per-sequence growth allocates
whole blocks tagged with their `CacheOwner`. Reads assemble blocks into a
contiguous view (cached until the next update). Pool exhaustion surfaces as a
typed `BlockPoolExhausted` error rather than a CUDA OOM.

Paged mode is enabled per-registry via `CacheConfig::with_paged(true)`; if the
pool was not initialized, cache creation falls back to contiguous storage with
a warning.

## Eviction: Two-Tier, Budget-Driven (`kv_cache.rs:276-344`)

When `memory_budget_bytes` is set, `evict_to_budget()` runs over LRU-sorted
candidates. `Training` owners are always skipped, and `CacheConfig` supports an
`eviction_exempt` flag.

1. **Tier 1 — offload**: least-recently-used GPU caches are moved to CPU RAM
   (`offload_to_cpu`), freeing GPU memory while preserving the data.
2. **Tier 2 — remove**: if still over budget, already-offloaded CPU caches are
   removed entirely.

The restore direction is transparent: `get_or_create` on an offloaded session
moves it back to GPU before returning it, so callers never observe the offload
state.

## Prefix Caching and Partial Prefill (`torch_engine.rs:2373-2691`)

For session-owned caches, the engine reuses computed KV states across
conversation turns:

1. On a new request, the session's cache is swapped into the model
   (`swap_session_cache`).
2. `prefix_match_len(&prompt_tokens)` compares the new prompt against
   `cached_token_ids` from the previous turn.
3. On a hit, the cache is `truncate_to(prefix_len)` (discarding the stale
   suffix, e.g. the previous turn's generation) and **prefill starts at
   `prefix_len`** instead of 0 — only the new suffix is computed.
4. On a miss, the cache is cleared and prefill starts fresh.
5. After generation, `save_cached_tokens` records the full token sequence for
   the next turn's match.

A typical multi-turn chat reuses the entire shared history, skipping most of
the prefill cost.

## TTT Delta-Dependency Invalidation

KV states are a function of the model weights — including any active per-tenant
LoRA delta. The registry tracks which delta each cache was computed under:

- `register_delta_dependency(owner, tenant_id)` records the association at
  generation time.
- `invalidate_for_tenant(tenant_id)` removes every dependent cache (clearing
  cached token IDs first, so prefix matching can never resurrect stale KV
  values) when that tenant's delta is evicted, zeroed, or written back.

This keeps TTT adaptation and prefix caching correct together: a weight change
invalidates exactly the caches that observed it.

## Concurrency Model Summary

| Level | Mechanism | Contention |
|-------|-----------|------------|
| Registry (across owners) | `DashMap` | None between different sessions |
| One cache (within an owner) | `parking_lot::Mutex<KVCacheManager>` | Serializes requests on the same session (tch `Tensor` is not `Sync`) |
| Layers (within a manager) | `DashMap<usize, LayerKVCache>` | Per-layer entries; each sequence has its own manager, so layer borrows never alias |
| Paged block pool | `Arc<Mutex<BlockPool>>` | Brief lock on block alloc/free/assemble |
