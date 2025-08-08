# Hyprstream Adaptive Learning Implementation Roadmap
## Step-by-Step Guide to Transform Hyprstream

### Quick Start Commands

```bash
# Clone and setup
cd /mnt/hyprstream/hyprstream
git checkout -b feature/adaptive-learning

# Add new dependencies
cargo add candle-core candle-transformers tokenizers safetensors
cargo add nanovdb memmap2 async-stream
cargo add rdkafka --features "cmake-build"

# Create new module structure
mkdir -p src/{models,adapters,streaming,inference}
mkdir -p src/storage/vdb
```

---

## Week 1: Foundation Setup

### TODO-001: Setup NanoVDB Integration
**File**: `src/storage/vdb/mod.rs`

```rust
// Create new VDB storage module
use nanovdb::{Grid, GridBuilder};
use memmap2::{Mmap, MmapMut};
use std::path::Path;

pub struct VDBStorage {
    grids: HashMap<String, Grid<f32>>,
    mmap_files: HashMap<String, MmapMut>,
}

impl VDBStorage {
    pub fn new(base_path: &Path) -> Result<Self> {
        // TODO: Implement VDB initialization
        // - Create storage directory
        // - Initialize grid collection
        // - Setup memory mapping
    }
}
```

**Actions**:
1. Remove `src/storage/duckdb.rs` and `src/storage/adbc.rs`
2. Update `src/storage/mod.rs` to export VDB module
3. Create VDB grid serialization/deserialization
4. Add compression support (LZ4/Zstd)

### TODO-002: Implement Base Model Loader
**File**: `src/models/qwen3.rs`

```rust
use candle_core::{Device, Tensor};
use candle_transformers::models::qwen2::{Config, Model};
use tokenizers::Tokenizer;

pub struct Qwen3Model {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
}

impl Qwen3Model {
    pub async fn from_pretrained(model_id: &str) -> Result<Self> {
        // TODO: Load Qwen3-1.7B from HuggingFace
        // - Download model weights
        // - Load tokenizer
        // - Initialize on GPU if available
    }
}
```

**Actions**:
1. Create model downloading logic
2. Implement weight loading from safetensors
3. Setup tokenizer integration
4. Add device selection (CUDA/CPU)

### TODO-003: Create Sparse Adapter Framework
**File**: `src/adapters/sparse_lora.rs`

```rust
pub struct SparseLoRAAdapter {
    lora_a: Tensor,  // [in_features, rank]
    lora_b: Tensor,  // [rank, out_features]
    mask: Tensor,    // Sparsity mask
    sparsity: f32,   // Target sparsity (0.99)
}

impl SparseLoRAAdapter {
    pub fn new(in_features: usize, out_features: usize, rank: usize) -> Self {
        // TODO: Initialize sparse adapter
        // - Create low-rank matrices
        // - Initialize sparsity mask
        // - Setup gradient tracking
    }
    
    pub fn to_vdb(&self) -> VDBGrid {
        // TODO: Convert to VDB sparse grid
    }
}
```

**Actions**:
1. Implement LoRA weight initialization
2. Add sparsity enforcement
3. Create VDB serialization
4. Add gradient accumulation

---

## Week 2: Storage Layer Completion

### TODO-004: VDB Adapter Persistence
**File**: `src/storage/vdb/adapter_store.rs`

```rust
pub struct AdapterStore {
    vdb_storage: Arc<VDBStorage>,
    metadata: HashMap<String, AdapterMetadata>,
}

impl AdapterStore {
    pub async fn save_adapter(&self, domain: &str, adapter: &SparseLoRAAdapter) -> Result<()> {
        // TODO: Save adapter to VDB
        // - Convert adapter to VDB grid
        // - Compress with 99% sparsity
        // - Update metadata
    }
    
    pub async fn load_adapter(&self, domain: &str) -> Result<SparseLoRAAdapter> {
        // TODO: Load adapter from VDB
    }
}
```

### TODO-005: Checkpoint Management
**File**: `src/storage/checkpoint.rs`

```rust
pub struct CheckpointManager {
    base_path: PathBuf,
    max_checkpoints: usize,
}

impl CheckpointManager {
    pub async fn save_checkpoint(&self, model_state: &ModelState) -> Result<()> {
        // TODO: Implement checkpoint saving
        // - Snapshot current adapters
        // - Save training state
        // - Rotate old checkpoints
    }
}
```

---

## Week 3-4: Streaming Pipeline

### TODO-006: Gradient Computation
**File**: `src/streaming/gradient.rs`

```rust
pub struct GradientComputer {
    model: Arc<Qwen3Model>,
    batch_size: usize,
}

impl GradientComputer {
    pub async fn compute_gradient(&self, batch: Vec<Event>) -> SparseGradient {
        // TODO: Compute sparse gradients
        // - Forward pass
        // - Compute loss
        // - Backward pass (sparse)
        // - Return top 1% gradients
    }
}
```

### TODO-007: Weight Update System
**File**: `src/streaming/updates.rs`

```rust
pub struct WeightUpdater {
    adapters: Arc<RwLock<HashMap<String, SparseLoRAAdapter>>>,
    learning_rate: f32,
}

impl WeightUpdater {
    pub async fn apply_update(&self, domain: &str, gradient: SparseGradient) {
        // TODO: Apply sparse weight update
        // - Lock-free update mechanism
        // - In-place VDB modification
        // - Maintain 99% sparsity
    }
}
```

### TODO-008: Event Ingestion
**File**: `src/streaming/ingestion.rs`

```rust
pub struct EventIngestion {
    kafka_consumer: StreamConsumer,
    http_receiver: HttpReceiver,
    ws_handler: WebSocketHandler,
}

impl EventIngestion {
    pub async fn start(&self) -> impl Stream<Item = Event> {
        // TODO: Multi-source event streaming
        // - Kafka consumption
        // - HTTP webhook handling
        // - WebSocket connections
    }
}
```

---

## Week 5-6: Inference Server

### TODO-009: Transform Service Layer
**File**: `src/service.rs`

```rust
// Replace FlightSqlServer with InferenceServer
pub struct InferenceServer {
    model_pool: Arc<ModelPool>,
    adapter_manager: Arc<AdapterManager>,
}

impl InferenceServer {
    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        // TODO: Setup inference endpoints
        // - /generate endpoint
        // - /stream endpoint
        // - /adapters endpoint
    }
}
```

### TODO-010: Model Pool Management
**File**: `src/inference/pool.rs`

```rust
pub struct ModelPool {
    instances: Vec<Arc<Qwen3Model>>,
    available: Arc<Semaphore>,
}

impl ModelPool {
    pub async fn acquire(&self) -> ModelGuard {
        // TODO: Model instance management
        // - Acquire available instance
        // - Return guard for auto-release
        // - Handle scaling
    }
}
```

### TODO-011: Adapter Routing
**File**: `src/inference/routing.rs`

```rust
pub struct AdapterRouter {
    domain_mapping: HashMap<String, String>,
    default_adapter: Option<String>,
}

impl AdapterRouter {
    pub async fn route_request(&self, request: &Request) -> String {
        // TODO: Route to appropriate adapter
        // - Extract domain from request
        // - Map to adapter
        // - Handle fallback
    }
}
```

---

## Week 7: Client Updates

### TODO-012: Python Client Updates
**File**: `examples/client/python/hyprstream_client/ml_client.py`

```python
class AdaptiveClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def generate(self, prompt: str, domain: str = None) -> str:
        """Generate text with optional domain adapter"""
        # TODO: Implement generation endpoint
        
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Stream token generation"""
        # TODO: Implement streaming
        
    async def train_adapter(self, domain: str, data: List[str]):
        """Train domain-specific adapter"""
        # TODO: Implement training submission
```

### TODO-013: CLI Updates
**File**: `src/cli/commands/ml.rs`

```rust
pub fn ml_commands() -> Command {
    Command::new("ml")
        .subcommand(Command::new("generate")
            .arg(Arg::new("prompt").required(true))
            .arg(Arg::new("domain").long("domain")))
        .subcommand(Command::new("train")
            .arg(Arg::new("domain").required(true))
            .arg(Arg::new("data").required(true)))
}
```

---

## Week 8: Testing & Optimization

### TODO-014: Integration Tests
**File**: `tests/integration/adaptive_test.rs`

```rust
#[tokio::test]
async fn test_end_to_end_inference() {
    // TODO: Complete inference test
    let server = setup_test_server().await;
    let response = server.generate("Test prompt", "test_domain").await;
    assert!(!response.is_empty());
}

#[tokio::test]
async fn test_streaming_updates() {
    // TODO: Test weight streaming
    let pipeline = setup_streaming_pipeline().await;
    pipeline.submit_event(test_event()).await;
    // Verify weight update
}
```

### TODO-015: Benchmarks
**File**: `benches/inference_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_inference(c: &mut Criterion) {
    c.bench_function("generate_128_tokens", |b| {
        b.iter(|| {
            // TODO: Benchmark token generation
        })
    });
    
    c.bench_function("weight_update", |b| {
        b.iter(|| {
            // TODO: Benchmark sparse update
        })
    });
}
```

---

## Migration Checklist

### Phase 1: Parallel Development ✅
- [ ] Create feature branch
- [ ] Add ML dependencies
- [ ] Keep metrics code intact
- [ ] Add feature flags

### Phase 2: Implementation ⚡
- [ ] Week 1: Foundation (TODO 1-3)
- [ ] Week 2: Storage (TODO 4-5)
- [ ] Week 3-4: Streaming (TODO 6-8)
- [ ] Week 5-6: Inference (TODO 9-11)
- [ ] Week 7: Clients (TODO 12-13)
- [ ] Week 8: Testing (TODO 14-15)

### Phase 3: Deployment 🚀
- [ ] Build Docker image
- [ ] Deploy to staging
- [ ] Performance testing
- [ ] Progressive rollout
- [ ] Monitor metrics

### Phase 4: Cleanup 🧹
- [ ] Remove old metrics code
- [ ] Update documentation
- [ ] Archive unused modules
- [ ] Update CI/CD

---

## Configuration Files to Update

### `Cargo.toml`
```toml
[dependencies]
# Add
candle-core = "0.4"
candle-transformers = "0.4"
tokenizers = "0.15"
safetensors = "0.4"
nanovdb = "0.1"
memmap2 = "0.9"

# Remove
# duckdb = ...
# polars = ...
```

### `config/default.toml`
```toml
[model]
base_model = "Qwen/Qwen3-1.7B"
device = "cuda"

[adapter]
sparsity = 0.99
rank = 16

[streaming]
batch_size = 32
update_interval_ms = 100
```

---

## Success Metrics

| Week | Milestone | Success Criteria |
|------|-----------|-----------------|
| 1 | Foundation | Model loads, VDB works |
| 2 | Storage | Adapters save/load |
| 3-4 | Streaming | Updates < 2ms |
| 5-6 | Inference | Latency < 15ms |
| 7 | Clients | API functional |
| 8 | Testing | All tests pass |

---

## Commands for Quick Testing

```bash
# Test model loading
cargo test --test model_loading

# Test VDB storage
cargo test --test vdb_storage

# Benchmark inference
cargo bench --bench inference_bench

# Run server
cargo run --release -- serve --config config/ml.toml

# Test with client
python examples/client/python/test_ml.py
```

---

## Next Steps

1. **Immediate**: Create feature branch and add dependencies
2. **Today**: Start TODO-001 (VDB integration)
3. **This Week**: Complete foundation (TODO 1-3)
4. **Next Week**: Implement storage layer

This roadmap provides clear, actionable steps to transform Hyprstream into a real-time adaptive learning server with Qwen3-1.7B and NanoVDB-backed sparse adapters.