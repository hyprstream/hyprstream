# KV Cache Architecture

**Design Document** - December 2025

## Overview

This document describes the KV (Key-Value) cache architecture for hyprstream, designed to support:
- Concurrent inference requests
- Concurrent LoRA adapter training
- Mixed inference + training workloads
- Future pipeline parallelism
- Optional quantization for memory efficiency

## Problem Statement

### Current Limitations

The existing KV cache design has a single `seq_pos` per layer, making it single-sequence:

```rust
// Current: Single-sequence design
pub struct LayerKVCache {
    pub keys: Option<Tensor>,
    pub values: Option<Tensor>,
    pub seq_pos: usize,  // ← Only one position!
}
```

This prevents:
1. **Concurrent inference**: Two requests can't share a cache manager
2. **Concurrent training**: Can't train adapter A while training adapter B
3. **Mixed workloads**: Can't serve inference while training in background

### Locking Analysis

| Approach | Overhead | Concurrency | Training Support |
|----------|----------|-------------|------------------|
| HashMap + Mutex | Low | None | Single-sequence only |
| DashMap (flat) | Medium | Layer-level | Breaks on shared seq_pos |
| **Registry + Per-Sequence** | Low-Medium | **Full isolation** | **Multi-adapter** |

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     KVCacheRegistry                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  caches: DashMap<CacheOwner, Arc<SequenceKVCache>>  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│         ┌──────────────────┼──────────────────┐            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │ Inference:1 │   │ Inference:2 │   │Training:lora│      │
│  │SequenceKV   │   │SequenceKV   │   │SequenceKV   │      │
│  │  Cache      │   │  Cache      │   │  Cache      │      │
│  └─────────────┘   └─────────────┘   └─────────────┘      │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────────────────────────────────────────┐      │
│  │  Vec<RwLock<LayerKVCache>>  (per sequence)      │      │
│  │  [Layer0] [Layer1] [Layer2] ... [LayerN]        │      │
│  └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Type Definitions

```rust
/// Identifies the owner of a cache instance
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub enum CacheOwner {
    /// Inference sequence (request_id, conversation_id, etc.)
    Inference(u64),

    /// Training run for a specific adapter
    Training {
        adapter: String,
        run_id: u64
    },
}

/// Configuration for cache instances
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Number of transformer layers
    pub num_layers: usize,

    /// Maximum sequence length (context window)
    pub max_seq_len: usize,

    /// Quantization type for memory efficiency
    pub quant_type: KVQuantType,

    /// For training: layers to gradient checkpoint (recompute vs cache)
    pub checkpoint_layers: Option<Vec<usize>>,
}

/// Top-level registry managing all cache instances
pub struct KVCacheRegistry {
    /// Active caches indexed by owner
    caches: DashMap<CacheOwner, Arc<SequenceKVCache>>,

    /// Default configuration for new caches
    default_config: CacheConfig,

    /// Memory budget for automatic eviction
    memory_budget_bytes: usize,
}

/// Per-sequence cache with isolated state
pub struct SequenceKVCache {
    /// Per-layer caches with fine-grained locking
    layers: Vec<parking_lot::RwLock<LayerKVCache>>,

    /// Current sequence position (atomic for lock-free reads)
    seq_pos: AtomicUsize,

    /// Configuration for this cache instance
    config: CacheConfig,

    /// Metrics for eviction decisions
    created_at: Instant,
    last_access: AtomicU64,  // Unix timestamp
    access_count: AtomicU64,
}

/// Single layer's KV cache (unchanged from current)
pub struct LayerKVCache {
    storage: KVStorage,  // Full precision or quantized
    seq_pos: usize,
    max_seq_len: usize,
    allocated_capacity: usize,
    quant_type: KVQuantType,
}
```

## Usage Patterns

### Single Inference Request

```rust
// Get or create cache for this request
let cache = registry.get_or_create(CacheOwner::Inference(request_id));

// Forward pass - sequential layer access
for (idx, layer) in model.layers.iter().enumerate() {
    let mut kv = cache.layer(idx).write();
    let output = layer.forward_with_cache(&input, &mut kv, start_pos)?;
}

// Release when request completes
registry.release(&CacheOwner::Inference(request_id));
```

### Concurrent Inference Requests

```rust
// Request A and B get completely isolated caches
let cache_a = registry.get_or_create(CacheOwner::Inference(request_a_id));
let cache_b = registry.get_or_create(CacheOwner::Inference(request_b_id));

// Can process concurrently - no lock contention
tokio::join!(
    process_request(model.clone(), cache_a, prompt_a),
    process_request(model.clone(), cache_b, prompt_b),
);
```

### LoRA Adapter Training

```rust
// Each adapter training run gets isolated cache
let cache = registry.get_or_create(CacheOwner::Training {
    adapter: "coding-assistant".into(),
    run_id: training_run_id,
});

// Training loop
for batch in data_loader {
    // Forward pass with KV caching
    let logits = model.forward_with_cache(&batch.input, &cache)?;

    // Backward pass
    let loss = compute_loss(&logits, &batch.target);
    loss.backward();
    optimizer.step();

    // Clear cache between sequences (or keep for continued generation)
    cache.clear();
}

registry.release(&CacheOwner::Training {
    adapter: "coding-assistant".into(),
    run_id: training_run_id,
});
```

### Concurrent Adapter Training

```rust
// Train multiple adapters simultaneously
let handles: Vec<_> = adapters.iter().map(|adapter_name| {
    let registry = registry.clone();
    let model = model.clone();
    let data = training_data.get(adapter_name).clone();

    tokio::spawn(async move {
        let cache = registry.get_or_create(CacheOwner::Training {
            adapter: adapter_name.clone(),
            run_id: generate_run_id(),
        });

        train_adapter(&model, &cache, data).await
    })
}).collect();

// All train concurrently with full isolation
futures::future::join_all(handles).await;
```

### Mixed Inference + Training

```rust
// Production serving continues
let inference_task = tokio::spawn(async move {
    loop {
        let request = receive_request().await;
        let cache = registry.get_or_create(CacheOwner::Inference(request.id));
        serve_request(&model, &cache, request).await;
        registry.release(&CacheOwner::Inference(request.id));
    }
});

// Background training doesn't block inference
let training_task = tokio::spawn(async move {
    let cache = registry.get_or_create(CacheOwner::Training {
        adapter: "new-adapter".into(),
        run_id: 1,
    });
    train_adapter(&model, &cache, training_data).await;
});

tokio::join!(inference_task, training_task);
```

## Memory Management

### Automatic Eviction

```rust
impl KVCacheRegistry {
    /// Evict least-recently-used caches to stay within budget
    pub fn evict_to_budget(&self) {
        let current_usage = self.total_memory_usage();
        if current_usage <= self.memory_budget_bytes {
            return;
        }

        // Collect candidates (skip active training runs)
        let mut candidates: Vec<_> = self.caches.iter()
            .filter(|entry| matches!(entry.key(), CacheOwner::Inference(_)))
            .map(|entry| {
                let last_access = entry.value().last_access.load(Ordering::Relaxed);
                (entry.key().clone(), last_access, entry.value().memory_usage())
            })
            .collect();

        // Sort by last access (oldest first)
        candidates.sort_by_key(|(_, last_access, _)| *last_access);

        // Evict until under budget
        let mut freed = 0;
        for (owner, _, size) in candidates {
            if current_usage - freed <= self.memory_budget_bytes {
                break;
            }
            self.caches.remove(&owner);
            freed += size;
            tracing::info!("Evicted cache {:?}, freed {} bytes", owner, size);
        }
    }
}
```

### Quantization Integration

```rust
impl SequenceKVCache {
    /// Create with quantization for memory efficiency
    pub fn new_quantized(config: CacheConfig) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| {
                RwLock::new(LayerKVCache::new(
                    config.max_seq_len,
                    config.quant_type,  // Int8, Nf4, Fp4
                ))
            })
            .collect();

        Self {
            layers,
            seq_pos: AtomicUsize::new(0),
            config,
            created_at: Instant::now(),
            last_access: AtomicU64::new(timestamp_now()),
            access_count: AtomicU64::new(0),
        }
    }
}
```

Memory savings with quantization:

| Quant Type | Bits | Memory vs FP16 | Quality Impact |
|------------|------|----------------|----------------|
| None (FP16) | 16 | 100% | Baseline |
| Int8 | 8 | 50% | Minimal |
| Nf4 | 4 | 25% | Low |
| Fp4 | 4 | 25% | Low-Medium |

## Future: Pipeline Parallelism

The per-layer `RwLock` design enables pipeline parallelism:

```
Time →
        ┌─────┬─────┬─────┬─────┐
Batch 0 │ L0  │ L1  │ L2  │ L3  │
        └─────┴─────┴─────┴─────┘
              ┌─────┬─────┬─────┬─────┐
Batch 1       │ L0  │ L1  │ L2  │ L3  │
              └─────┴─────┴─────┴─────┘
                    ┌─────┬─────┬─────┬─────┐
Batch 2             │ L0  │ L1  │ L2  │ L3  │
                    └─────┴─────┴─────┴─────┘
```

```rust
// Pipeline parallel forward (future implementation)
async fn pipeline_forward(
    batches: Vec<Tensor>,
    cache: &SequenceKVCache,
) -> Vec<Tensor> {
    let (tx, rx) = mpsc::channel(num_layers);

    // Each layer processes batches as they arrive
    for layer_idx in 0..num_layers {
        let layer_cache = cache.layer(layer_idx);
        tokio::spawn(async move {
            while let Some((batch_idx, input)) = rx.recv().await {
                let mut kv = layer_cache.write();
                let output = process_layer(input, &mut kv);
                tx.send((batch_idx, output)).await;
            }
        });
    }

    // Feed batches into pipeline
    for (batch_idx, batch) in batches.iter().enumerate() {
        tx.send((batch_idx, batch.clone())).await;
    }

    // Collect outputs...
}
```

## Migration Path

### Phase 1: Add Registry (Non-Breaking)

1. Introduce `KVCacheRegistry` alongside existing code
2. Existing `KVCacheManager` becomes internal implementation detail
3. Callers gradually migrate to registry API

### Phase 2: Update Architectures

1. Update `llama.rs`, `qwen.rs`, etc. to use registry
2. Remove outer `Mutex` wrapping (registry handles concurrency)
3. Add `CacheOwner` parameter to forward methods

### Phase 3: Enable Concurrent Workloads

1. Update server to assign unique `CacheOwner` per request
2. Enable concurrent inference
3. Add training API with cache isolation

## Dependencies

```toml
[dependencies]
dashmap = "5"           # Concurrent hashmap for registry
parking_lot = "0.12"    # Fast RwLock for per-layer access
```

Optional for quantization:
```toml
[dependencies]
bitsandbytes-sys = { path = "../bitsandbytes-sys", optional = true }

[features]
bnb = ["bitsandbytes-sys"]
```

## References

- [Gradient Checkpointing](https://arxiv.org/abs/1604.06174) - Trading compute for memory
- [Pipeline Parallelism](https://arxiv.org/abs/1811.06965) - GPipe paper
- [KV Cache Quantization](https://arxiv.org/abs/2401.18079) - Memory-efficient inference
