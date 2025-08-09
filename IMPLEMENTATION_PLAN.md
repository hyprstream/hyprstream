# Hyprstream Implementation Plan - CRITICAL ARCHITECTURE FIX

## ⚠️ URGENT: Architecture Issue Identified

**PROBLEM:** The `src/inference/` module contains FlightSQL transport code instead of LLaMA.cpp inference logic. This architectural confusion prevents proper model inference despite having functional LLaMA.cpp integration in `src/runtime/`.

**ROOT CAUSE:** Mixed concerns between transport protocols and ML inference logic.

**IMPACT:** Cannot perform actual model inference with LLaMA.cpp.

## IMMEDIATE PRIORITY: Fix Architecture (Next 48 Hours)

### Critical Path to Working LLaMA.cpp Inference

## Phase 1: Foundation & Core Dependencies (Weeks 1-4)

### **Milestone 1.1: Dependency Integration (Week 1)**

#### Add Missing Crates to Cargo.toml
```toml
[dependencies]
# Neural Network Operations
candle-core = "0.3"
candle-nn = "0.3"  
candle-transformers = "0.3"
safetensors = "0.4"
memmap2 = "0.9"

# HTTP & API
reqwest = { version = "0.11", features = ["json", "stream"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
urlencoding = "2.1"

# Data Formats
serde_yaml = "0.9"
hf-hub = "0.3"  # HuggingFace Hub API

# Performance
rayon = "1.7"  # Parallel processing
indicatif = "0.17"  # Progress bars

# Optional GPU Support
cudarc = { version = "0.9", optional = true }
```

#### Build System Setup
- [ ] Configure NanoVDB C++ library integration in `build.rs`
- [ ] Add CUDA detection and compilation flags
- [ ] Set up cross-platform build scripts
- [ ] Create feature flags for GPU/CPU-only builds

### **Milestone 1.2: Basic Model Loading (Weeks 2-3)**

#### SafeTensors Integration
```rust
// src/inference/model_loader.rs
pub struct ModelLoader {
    base_path: PathBuf,
    tensors: SafeTensors<'static>,
    mmap: Mmap,
    metadata: ModelMetadata,
}

impl ModelLoader {
    pub async fn load_safetensors(path: &Path) -> Result<Self> {
        // Memory-map SafeTensors file
        // Parse tensor headers and metadata
        // Validate model architecture
    }
    
    pub fn get_tensor(&self, name: &str) -> Result<TensorView> {
        // Zero-copy tensor access via mmap
    }
}
```

#### GGUF Support (Basic)
```rust
// src/inference/gguf_loader.rs
pub struct GGUFLoader {
    // Basic GGUF parsing for model weights
    // Focus on Qwen3-compatible formats initially
}
```

### **Milestone 1.3: Minimal Inference Engine (Week 4)**

#### Core Inference Structure
```rust
// src/inference/engine.rs
pub struct InferenceEngine {
    device: Device,
    model: Box<dyn ModelTrait>,
    tokenizer: Tokenizer,
}

impl InferenceEngine {
    pub async fn forward_pass(
        &self, 
        input_ids: &[u32], 
        adapters: &[LoRAAdapter]
    ) -> Result<Tensor> {
        // Basic transformer forward pass
        // LoRA adaptation during forward pass
        // Return logits for next token prediction
    }
}
```

## Phase 2: Neural Network Integration (Weeks 5-8)

### **Milestone 2.1: Transformer Implementation (Weeks 5-6)**

#### Qwen3 Model Architecture
```rust
// src/models/qwen3_candle.rs
pub struct Qwen3Model {
    embeddings: Embedding,
    layers: Vec<TransformerLayer>,
    norm: LayerNorm,
    lm_head: Linear,
}

impl Qwen3Model {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Implement transformer forward pass
        // Support for attention, MLP, residual connections
        // Layer normalization and output projection
    }
    
    pub fn forward_with_adapters(
        &self, 
        input_ids: &Tensor, 
        adapters: &HashMap<String, LoRAAdapter>
    ) -> Result<Tensor> {
        // Forward pass with LoRA adapter fusion
        // Apply adapters to query, key, value projections
        // Maintain 99% sparsity during computation
    }
}
```

#### LoRA Adapter Integration
```rust
// src/adapters/lora_fusion.rs
pub struct LoRAFusionLayer {
    base_weight: Tensor,
    lora_a: Option<Tensor>,  // Low-rank matrices
    lora_b: Option<Tensor>,
    alpha: f32,
    sparse_mask: SparseMask,  // 99% sparse pattern
}

impl LoRAFusionLayer {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Base computation: input @ base_weight
        // LoRA computation: input @ (lora_b @ lora_a) * (alpha / rank)
        // Sparse combination with 99% sparsity maintained
    }
}
```

### **Milestone 2.2: Tokenization & Text Processing (Week 7)**

#### Tokenizer Integration
```rust
// src/inference/tokenizer.rs
pub struct Qwen3Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    vocab_size: usize,
    special_tokens: HashMap<String, u32>,
}

impl Qwen3Tokenizer {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Text -> token IDs
        // Handle special tokens, padding
    }
    
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        // Token IDs -> text
        // Handle streaming decode for generation
    }
}
```

### **Milestone 2.3: Text Generation (Week 8)**

#### Sampling Strategies
```rust
// src/inference/generation.rs
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
}

pub struct TextGenerator {
    engine: InferenceEngine,
    config: GenerationConfig,
}

impl TextGenerator {
    pub async fn generate(
        &self,
        prompt: &str,
        lora_adapters: &[String],
    ) -> Result<GenerationResult> {
        // Auto-regressive text generation
        // Apply sampling strategies (temperature, top-p, top-k)
        // Support streaming generation
    }
}
```

## Phase 3: Training & Adaptation System (Weeks 9-12)

### **Milestone 3.1: Gradient Computation (Weeks 9-10)**

#### Backpropagation Implementation
```rust
// src/training/backprop.rs
pub struct GradientComputer {
    device: Device,
    loss_fn: CrossEntropyLoss,
}

impl GradientComputer {
    pub async fn compute_gradients(
        &self,
        model: &Qwen3Model,
        input_ids: &Tensor,
        target_ids: &Tensor,
        adapters: &HashMap<String, LoRAAdapter>,
    ) -> Result<HashMap<String, AdapterGradients>> {
        // Forward pass with loss computation
        // Backward pass through LoRA layers only
        // Maintain sparsity during gradient computation
        
        let loss = self.compute_loss(&logits, target_ids)?;
        let gradients = self.backward_pass(&loss, adapters)?;
        
        // Apply sparsity constraints to gradients
        self.apply_sparsity_mask(&mut gradients, 0.99)?;
        
        Ok(gradients)
    }
}
```

#### Loss Functions
```rust
// src/training/loss.rs
pub enum LossFunction {
    CrossEntropy,
    KLDivergence,
    MSE,
}

impl LossFunction {
    pub fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        match self {
            Self::CrossEntropy => {
                // Implement cross-entropy loss for language modeling
                // Support for label smoothing
            }
            // Other loss functions...
        }
    }
}
```

### **Milestone 3.2: Optimizer Implementation (Week 11)**

#### Sparse-Aware Optimizers
```rust
// src/training/optimizer.rs
pub struct SparseAdamW {
    learning_rate: f32,
    betas: (f32, f32),
    weight_decay: f32,
    sparsity_target: f32,
    momentum: HashMap<String, Tensor>,
    variance: HashMap<String, Tensor>,
}

impl SparseAdamW {
    pub fn step(
        &mut self,
        params: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        for (name, param) in params.iter_mut() {
            if let Some(grad) = gradients.get(name) {
                // AdamW update with sparsity constraints
                // Magnitude-based pruning to maintain 99% sparsity
                self.update_with_sparsity(name, param, grad)?;
            }
        }
        Ok(())
    }
    
    fn update_with_sparsity(
        &mut self,
        name: &str,
        param: &mut Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        // Standard AdamW update
        // Apply magnitude-based pruning
        // Ensure sparsity ratio is maintained
    }
}
```

### **Milestone 3.3: Auto-Regressive Training Loop (Week 12)**

#### Training Pipeline
```rust
// src/training/trainer.rs
pub struct AutoRegressiveTrainer {
    model: Arc<Qwen3Model>,
    optimizer: SparseAdamW,
    gradient_computer: GradientComputer,
    sample_queue: mpsc::Receiver<TrainingSample>,
    vdb_storage: Arc<VDBSparseStorage>,
}

impl AutoRegressiveTrainer {
    pub async fn training_loop(&mut self) -> Result<()> {
        let mut batch = Vec::new();
        
        while let Some(sample) = self.sample_queue.recv().await {
            batch.push(sample);
            
            if batch.len() >= self.config.batch_size {
                // Process batch
                let gradients = self.compute_batch_gradients(&batch).await?;
                
                // Update LoRA adapters
                self.apply_gradients(&gradients).await?;
                
                // Update VDB storage with new sparse weights
                self.update_vdb_storage(&gradients).await?;
                
                batch.clear();
            }
        }
        
        Ok(())
    }
}
```

## Phase 4: Production Features & Optimization (Weeks 13-16)

### **Milestone 4.1: Neural Compression (Week 13)**

#### NeuralVDB Implementation
```rust
// src/compression/neural_vdb.rs
pub struct NeuralCompressionCodec {
    topology_classifier: TopologyNet,
    value_regressor: ValueNet,
    device: Device,
}

impl NeuralCompressionCodec {
    pub async fn compress(&self, adapter: &LoRAAdapter) -> Result<CompressedRepresentation> {
        // Extract sparse topology pattern
        let topology = self.extract_topology(&adapter.weights)?;
        
        // Classify topology using small neural network
        let topology_code = self.topology_classifier.encode(&topology)?;
        
        // Compress non-zero values using regression network
        let value_code = self.value_regressor.encode(&adapter.values)?;
        
        Ok(CompressedRepresentation {
            topology_code,
            value_code,
            metadata: adapter.metadata.clone(),
        })
    }
}
```

### **Milestone 4.2: Complete API Integration (Week 14)**

#### HuggingFace Hub Client
```rust
// src/registry/huggingface.rs
impl HuggingFaceClient {
    pub async fn download_with_progress(
        &self,
        model_uri: &ModelUri,
        files: &[String],
        progress_callback: impl Fn(u64, u64),
    ) -> Result<DownloadResult> {
        for file in files {
            let url = self.get_download_url(model_uri, file).await?;
            
            let response = self.client.get(&url).send().await?;
            let total_size = response.content_length().unwrap_or(0);
            
            // Stream download with progress updates
            let mut stream = response.bytes_stream();
            let mut downloaded = 0u64;
            
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                // Write chunk and update progress
                downloaded += chunk.len() as u64;
                progress_callback(downloaded, total_size);
            }
        }
        
        Ok(download_result)
    }
}
```

#### Ollama Registry Implementation
```rust
// src/registry/ollama.rs
pub struct OllamaClient {
    base_url: String,
    client: reqwest::Client,
}

#[async_trait]
impl ModelRegistry for OllamaClient {
    async fn download_model(&self, model_uri: &ModelUri, ...) -> Result<DownloadResult> {
        // Implement Ollama API integration
        // Support for GGUF models from Ollama
    }
}
```

### **Milestone 4.3: CLI Completion (Week 15)**

#### Export/Import Functionality
```rust
// src/cli/export.rs
pub async fn export_lora_adapter(
    lora_id: &str,
    output_path: &Path,
    format: ExportFormat,
    include_base: bool,
) -> Result<()> {
    match format {
        ExportFormat::SafeTensors => {
            // Export as SafeTensors format
            // Include metadata and configuration
        }
        ExportFormat::GGUF => {
            // Export as GGUF format for Ollama compatibility
        }
        ExportFormat::PyTorch => {
            // Export as PyTorch state dict
        }
    }
}

pub async fn import_lora_adapter(
    input_path: &Path,
    name: Option<String>,
) -> Result<String> {
    // Auto-detect format and import
    // Register in VDB storage
    // Return new LoRA ID
}
```

#### Model Verification
```rust
// src/cli/verification.rs
pub async fn verify_model_integrity(
    model_uri: &ModelUri,
    storage: &ModelStorage,
) -> Result<ValidationResult> {
    let metadata = storage.get_metadata(model_uri).await?;
    let model_path = model_uri.local_path(storage.base_dir());
    
    // Check file existence
    // Verify file sizes match metadata
    // Compute and verify checksums if available
    // Test model loading capability
    
    Ok(ValidationResult {
        is_valid: true,
        issues: vec![],
        recommendations: vec![],
    })
}
```

### **Milestone 4.4: Performance Optimization (Week 16)**

#### GPU Acceleration
```rust
// src/acceleration/gpu.rs
#[cfg(feature = "cuda")]
pub struct CudaAccelerator {
    context: cudarc::driver::CudaContext,
    streams: Vec<cudarc::driver::CudaStream>,
    memory_pool: CudaMemoryPool,
}

impl CudaAccelerator {
    pub async fn accelerated_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        sparse_mask: Option<&SparseMask>,
    ) -> Result<Tensor> {
        // CUDA kernel for sparse matrix multiplication
        // Optimized for 99% sparse LoRA adapters
        // Memory-efficient attention computation
    }
}
```

#### Memory Optimization
```rust
// src/optimization/memory.rs
pub struct MemoryManager {
    tensor_pool: TensorPool,
    gc_threshold: usize,
}

impl MemoryManager {
    pub fn manage_tensor_lifecycle(&mut self, model: &Qwen3Model) -> Result<()> {
        // Implement tensor reuse pools
        // Gradient accumulation buffers
        // Automatic garbage collection
    }
}
```

## Implementation Dependencies & Prerequisites

### **Development Environment**
- Rust 1.75+ (for latest async/await features)
- CUDA Toolkit 12.0+ (optional, for GPU acceleration)
- Git LFS (for large model files)
- Docker (for consistent build environments)

### **External Libraries**
- **NanoVDB**: Latest release with Rust bindings
- **cuBLAS/cuDNN**: For optimized CUDA operations
- **Intel MKL**: For optimized CPU operations (alternative to OpenBLAS)

### **Model Assets**
- Qwen3-1.7B model files (SafeTensors format preferred)
- Tokenizer vocabulary and configuration files
- Test datasets for validation

## Risk Mitigation Strategies

### **Technical Risks**
1. **NanoVDB Integration Complexity**
   - Mitigation: Start with simplified VDB operations, gradually add complexity
   - Fallback: Use standard sparse matrices if VDB integration fails

2. **Memory Management with Large Models**
   - Mitigation: Implement model sharding and streaming loading
   - Monitoring: Add memory usage tracking and alerts

3. **CUDA Compatibility Issues**
   - Mitigation: Maintain CPU-only fallback paths
   - Testing: Automated testing on different GPU configurations

### **Performance Risks**
1. **Inference Latency**
   - Mitigation: Implement model quantization and caching
   - Optimization: Profile and optimize hot code paths

2. **Training Convergence**
   - Mitigation: Implement learning rate scheduling and early stopping
   - Validation: Regular checkpoint evaluation

## Testing Strategy

### **Unit Tests (Continuous)**
- Individual component testing (tokenizer, model layers, etc.)
- Mock implementations for external dependencies
- Property-based testing for numerical operations

### **Integration Tests (Weekly)**
- End-to-end model loading and inference
- LoRA adapter creation and training
- API endpoint functionality

### **Performance Tests (Bi-weekly)**
- Inference latency benchmarks
- Memory usage profiling
- Throughput measurements under load

### **Model Quality Tests (Monthly)**
- Text generation quality assessment
- LoRA adaptation effectiveness
- Training convergence validation

## Success Metrics

### **Phase 1 Success Criteria**
- [ ] Successful SafeTensors model loading
- [ ] Basic inference pipeline working
- [ ] Clean build with all dependencies

### **Phase 2 Success Criteria**
- [ ] Qwen3 model generates coherent text
- [ ] LoRA adapters can be applied during inference
- [ ] Tokenization working correctly

### **Phase 3 Success Criteria**
- [ ] Auto-regressive training loop functional
- [ ] LoRA adapters improve with training samples
- [ ] 99% sparsity maintained during training

### **Phase 4 Success Criteria**
- [ ] Neural compression achieving >10x ratios
- [ ] Complete CLI functionality working
- [ ] Production-ready performance metrics

## Resource Requirements

### **Development Team**
- 1 Senior Rust/ML Engineer (Full-time)
- 1 ML Research Engineer (Part-time, for neural compression)
- 1 DevOps Engineer (Part-time, for build/deployment)

### **Infrastructure**
- GPU-enabled development machines (RTX 4090 or better)
- CI/CD pipeline with GPU runners
- Model storage (500GB+ for model assets)
- Testing infrastructure for different hardware configurations

### **Timeline Summary**
- **Phase 1**: 4 weeks - Foundation & Dependencies
- **Phase 2**: 4 weeks - Neural Network Integration  
- **Phase 3**: 4 weeks - Training & Adaptation
- **Phase 4**: 4 weeks - Production Features
- **Total**: 16 weeks for complete implementation

This plan transforms Hyprstream from its current architectural state into a fully functional VDB-first adaptive ML inference server with auto-regressive training capabilities.