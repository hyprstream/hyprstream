# Hyprstream Runtime Integration Plan

## Executive Summary

This plan integrates proven runtime technologies from llama.cpp and vLLM while maintaining Hyprstream's unique VDB-first architecture for sparse LoRA adaptation. We'll build on battle-tested inference engines rather than implementing everything from scratch.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Hyprstream Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  VDB-First Sparse LoRA Layer (Unique to Hyprstream)        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   NanoVDB       │ │ Neural Codec    │ │ Sparse Training ││
│  │   Storage       │ │ (10-100x comp)  │ │ Auto-regressive ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│           Proven Runtime Layer (Adapt Existing)            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   llama.cpp     │ │     vLLM        │ │   Candle/       ││
│  │   GGUF/Model    │ │   PagedAttention│ │   SafeTensors   ││
│  │   Loading       │ │   + Batching    │ │   Inference     ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Runtime Foundation (Weeks 1-3)

### **Milestone 1.1: Choose Primary Runtime (Week 1)**

#### Option A: llama.cpp Integration (Recommended)
**Advantages:**
- GGUF format support (most models available)
- Excellent CPU/GPU hybrid inference
- Battle-tested quantization
- C API for easy FFI integration
- Smaller memory footprint

**Implementation:**
```rust
// Cargo.toml
[dependencies]
llama-cpp-2 = "0.1.67"  # Rust bindings for llama.cpp
candle-core = "0.3"     # For LoRA operations

[build-dependencies]
cmake = "0.1"           # For building llama.cpp

// src/runtime/llamacpp.rs
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{LlamaModel, AddBos},
};

pub struct LlamaCppEngine {
    backend: LlamaBackend,
    model: LlamaModel,
    context_params: LlamaContextParams,
}

impl LlamaCppEngine {
    pub fn load_model(path: &Path) -> Result<Self> {
        // Load GGUF model using llama.cpp
        // Configure context parameters for inference
        let backend = LlamaBackend::init()?;
        let model = LlamaModel::load_from_file(&backend, path, &params)?;
        
        Ok(Self { backend, model, context_params })
    }
    
    pub fn inference_with_lora(&self, 
        prompt: &str, 
        lora_adapters: &[LoRAAdapter]
    ) -> Result<String> {
        // Use llama.cpp for base inference
        // Apply LoRA modifications through our VDB layer
    }
}
```

#### Option B: vLLM Integration (Advanced)
**Use Case:** High-throughput serving scenarios
```rust
// Python-Rust bridge for vLLM integration
use pyo3::prelude::*;

#[pyclass]
pub struct VLLMEngine {
    engine: PyObject,  // vLLM AsyncLLMEngine
}

impl VLLMEngine {
    pub async fn generate_with_lora(&self, 
        requests: Vec<GenerationRequest>,
        lora_adapters: &HashMap<String, LoRAAdapter>
    ) -> Result<Vec<GenerationResult>> {
        // Use vLLM's continuous batching
        // Apply LoRA through our sparse layer
    }
}
```

### **Milestone 1.2: Qwen3 Model Support (Week 2)**

#### Qwen3 Architecture Implementation
Based on the transformer architecture, implement Qwen3-specific components:

```rust
// src/models/qwen3/mod.rs
pub struct Qwen3Config {
    pub vocab_size: usize,      // ~151936 for Qwen3
    pub hidden_size: usize,     // 1536 for 1.7B model
    pub intermediate_size: usize, // ~8960 (5.77 * hidden_size)
    pub num_hidden_layers: usize, // 28 for 1.7B
    pub num_attention_heads: usize, // 24
    pub max_position_embeddings: usize, // 8192 or 32768
    pub rope_theta: f64,        // 1000000.0 for extended context
    pub attention_dropout: f64,
    pub hidden_dropout: f64,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            max_position_embeddings: 32768,
            rope_theta: 1000000.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_sliding_window: false,
            sliding_window: None,
        }
    }
}

pub struct Qwen3Model {
    config: Qwen3Config,
    runtime_engine: Box<dyn RuntimeEngine>, // llama.cpp or vLLM
    lora_layer: VDBLoRALayer,
}
```

### **Milestone 1.3: LoRA Integration Layer (Week 3)**

#### VDB-LoRA Bridge
```rust
// src/adapters/vdb_lora_bridge.rs
pub struct VDBLoRALayer {
    vdb_storage: Arc<VDBSparseStorage>,
    active_adapters: HashMap<String, SparseLoRAAdapter>,
    fusion_cache: LRUCache<String, Tensor>,
}

impl VDBLoRALayer {
    pub fn apply_adapters_to_inference(
        &self,
        base_output: &Tensor,
        layer_name: &str,
        active_lora_ids: &[String],
    ) -> Result<Tensor> {
        let mut modified_output = base_output.clone();
        
        for lora_id in active_lora_ids {
            if let Some(adapter) = self.active_adapters.get(lora_id) {
                // Apply sparse LoRA: output = base + (input @ lora_b @ lora_a) * alpha
                let lora_delta = self.compute_lora_delta(base_output, adapter)?;
                modified_output = modified_output + lora_delta;
            }
        }
        
        Ok(modified_output)
    }
    
    fn compute_lora_delta(
        &self,
        input: &Tensor,
        adapter: &SparseLoRAAdapter,
    ) -> Result<Tensor> {
        // Use NanoVDB for sparse matrix operations
        // Maintain 99% sparsity during computation
        let sparse_result = self.vdb_storage.sparse_matmul(
            input,
            &adapter.lora_a,
            &adapter.lora_b,
            adapter.alpha,
        )?;
        
        Ok(sparse_result)
    }
}
```

## Phase 2: Production Integration (Weeks 4-8)

### **Milestone 2.1: GGUF LoRA Fusion (Weeks 4-5)**

#### Extend GGUF Format for LoRA
```rust
// src/formats/gguf_lora.rs
pub struct GGUFLoRAExtension {
    base_gguf: GGUFFile,
    lora_metadata: HashMap<String, LoRAMetadata>,
}

impl GGUFLoRAExtension {
    pub fn load_with_lora_adapters(
        gguf_path: &Path,
        lora_adapters: &[String],
        vdb_storage: &VDBSparseStorage,
    ) -> Result<Self> {
        // Load base GGUF model
        // Load specified LoRA adapters from VDB
        // Create fusion plan for inference
        
        let base_gguf = GGUFFile::load(gguf_path)?;
        let mut lora_metadata = HashMap::new();
        
        for lora_id in lora_adapters {
            let adapter = vdb_storage.load_adapter_neural_compressed(lora_id, Default::default()).await?;
            let metadata = self.analyze_adapter_compatibility(&adapter, &base_gguf)?;
            lora_metadata.insert(lora_id.clone(), metadata);
        }
        
        Ok(Self { base_gguf, lora_metadata })
    }
}
```

### **Milestone 2.2: PagedAttention Integration (Week 6)**

#### Adapt vLLM's PagedAttention for LoRA
```rust
// src/attention/paged_lora.rs
pub struct PagedLoRAAttention {
    base_attention: PagedAttention,  // From vLLM
    lora_adaptations: HashMap<String, LoRAAdaptation>,
    block_manager: BlockSpaceManager,
}

impl PagedLoRAAttention {
    pub fn compute_attention_with_lora(
        &self,
        query: &Tensor,
        key_cache: &KVCache,
        value_cache: &KVCache,
        active_loras: &[String],
    ) -> Result<Tensor> {
        // Compute base attention using vLLM's optimized kernels
        let base_attention = self.base_attention.forward(query, key_cache, value_cache)?;
        
        // Apply LoRA adaptations to query, key, value projections
        let adapted_attention = self.apply_lora_to_attention(
            &base_attention,
            query,
            active_loras,
        )?;
        
        Ok(adapted_attention)
    }
}
```

### **Milestone 2.3: Continuous Batching with Multi-LoRA (Week 7)**

#### Dynamic LoRA Serving
```rust
// src/serving/multi_lora_batch.rs
pub struct MultiLoRABatchManager {
    batch_manager: BatchManager,  // From vLLM
    lora_registry: HashMap<String, LoRAAdapter>,
    active_requests: HashMap<RequestId, Vec<String>>,  // Request -> LoRA IDs
}

impl MultiLoRABatchManager {
    pub async fn process_batch_with_lora(&mut self) -> Result<Vec<GenerationResult>> {
        let batch = self.batch_manager.get_next_batch().await?;
        let mut results = Vec::new();
        
        // Group requests by LoRA combinations for efficient batching
        let lora_groups = self.group_requests_by_lora(&batch);
        
        for (lora_combo, requests) in lora_groups {
            // Load LoRA combination into GPU memory
            self.prepare_lora_adapters(&lora_combo).await?;
            
            // Process batch with specific LoRA combination
            let batch_results = self.run_inference_batch(requests, &lora_combo).await?;
            results.extend(batch_results);
        }
        
        Ok(results)
    }
}
```

### **Milestone 2.4: OpenAI API Compatibility (Week 8)**

#### Extend OpenAI Endpoints for LoRA Selection
```rust
// src/api/openai_lora.rs
#[derive(Deserialize)]
pub struct LoRACompletionRequest {
    #[serde(flatten)]
    pub base_request: ChatCompletionRequest,
    
    // Extended fields for LoRA selection
    pub lora_adapters: Option<Vec<String>>,
    pub lora_weights: Option<HashMap<String, f32>>,
    pub lora_combination_strategy: Option<String>, // "weighted_sum", "sequential", etc.
}

pub async fn chat_completions_with_lora(
    State(state): State<ApiState>,
    Json(request): Json<LoRACompletionRequest>,
) -> Result<JsonResponse<ChatCompletionResponse>, StatusCode> {
    // Determine LoRA adapters to use
    let lora_adapters = request.lora_adapters.unwrap_or_default();
    
    // Create inference session with multi-LoRA support
    let session_id = state.multi_lora_engine
        .create_session_with_loras(lora_adapters, request.lora_weights)
        .await?;
    
    // Run inference
    let result = state.multi_lora_engine
        .generate(session_id, &request.base_request)
        .await?;
    
    Ok(JsonResponse(convert_to_openai_response(result)))
}
```

## Phase 3: Advanced Features (Weeks 9-12)

### **Milestone 3.1: Quantization-Aware LoRA (Week 9)**

#### INT4/INT8 LoRA Adapters
```rust
// src/quantization/quantized_lora.rs
pub struct QuantizedLoRAAdapter {
    lora_a_int4: QuantizedTensor,    // 4-bit weights
    lora_b_int4: QuantizedTensor,
    scales: Tensor,                   // FP16 scales
    zeros: Tensor,                    // Zero points
    alpha: f32,
}

impl QuantizedLoRAAdapter {
    pub fn quantize_from_fp16(adapter: &SparseLoRAAdapter) -> Result<Self> {
        // Apply INT4 quantization to LoRA weights
        // Use same techniques as llama.cpp (Q4_0, Q4_1 formats)
        let (lora_a_int4, scales_a, zeros_a) = quantize_tensor_int4(&adapter.lora_a)?;
        let (lora_b_int4, scales_b, zeros_b) = quantize_tensor_int4(&adapter.lora_b)?;
        
        Ok(Self {
            lora_a_int4,
            lora_b_int4,
            scales: concat_tensors(&[scales_a, scales_b])?,
            zeros: concat_tensors(&[zeros_a, zeros_b])?,
            alpha: adapter.alpha,
        })
    }
}
```

### **Milestone 3.2: Speculative Decoding with LoRA (Week 10)**

#### Draft Model + LoRA Verification
```rust
// src/speculative/lora_speculative.rs
pub struct LoRASpeculativeDecoder {
    draft_model: LlamaCppEngine,      // Small, fast model
    target_model: VLLMEngine,         // Large model with LoRA
    lora_adapters: Vec<String>,
}

impl LoRASpeculativeDecoder {
    pub async fn speculative_generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let mut generated = String::new();
        let mut current_prompt = prompt.to_string();
        
        loop {
            // Draft model generates multiple tokens quickly
            let draft_tokens = self.draft_model.generate(
                &current_prompt,
                4, // Draft 4 tokens ahead
            )?;
            
            // Target model with LoRA verifies draft tokens
            let verified_tokens = self.target_model.verify_with_lora(
                &current_prompt,
                &draft_tokens,
                &self.lora_adapters,
            ).await?;
            
            generated.push_str(&verified_tokens);
            if generated.len() >= max_tokens { break; }
            
            current_prompt.push_str(&verified_tokens);
        }
        
        Ok(generated)
    }
}
```

### **Milestone 3.3: Memory-Efficient Training (Week 11)**

#### Gradient Checkpointing + LoRA
```rust
// src/training/efficient_training.rs
pub struct MemoryEfficientTrainer {
    model: Qwen3Model,
    lora_adapters: HashMap<String, SparseLoRAAdapter>,
    gradient_checkpointing: bool,
    mixed_precision: bool,
}

impl MemoryEfficientTrainer {
    pub async fn train_step_with_checkpointing(
        &mut self,
        batch: &TrainingBatch,
    ) -> Result<TrainingMetrics> {
        // Only checkpoint every N layers to save memory
        let checkpointed_forward = |layer_idx: usize, input: &Tensor| {
            if layer_idx % 4 == 0 {
                // Checkpoint: save activations, recompute gradients
                self.checkpoint_layer(layer_idx, input)
            } else {
                // Normal forward pass
                self.forward_layer(layer_idx, input)
            }
        };
        
        // Forward pass with selective checkpointing
        let loss = self.forward_with_checkpointing(batch, checkpointed_forward)?;
        
        // Backward pass only through LoRA parameters (much smaller)
        let lora_gradients = self.backward_lora_only(&loss)?;
        
        // Apply gradients with sparsity constraints
        self.apply_sparse_gradients(&lora_gradients)?;
        
        Ok(TrainingMetrics {
            loss: loss.to_scalar()?,
            memory_used: self.get_memory_usage(),
            tokens_per_second: batch.num_tokens as f32 / batch.duration.as_secs_f32(),
        })
    }
}
```

### **Milestone 3.4: Production Monitoring (Week 12)**

#### Runtime Performance Metrics
```rust
// src/monitoring/performance.rs
pub struct PerformanceMonitor {
    metrics_collector: PrometheusCollector,
    latency_histogram: Histogram,
    throughput_gauge: Gauge,
    lora_usage_counter: CounterVec,
}

impl PerformanceMonitor {
    pub fn record_inference(&self, 
        lora_ids: &[String],
        latency: Duration,
        tokens_generated: usize,
    ) {
        // Record latency by LoRA combination
        self.latency_histogram
            .with_label_values(&[&format!("{:?}", lora_ids)])
            .observe(latency.as_secs_f64());
        
        // Record throughput
        let tokens_per_second = tokens_generated as f64 / latency.as_secs_f64();
        self.throughput_gauge.set(tokens_per_second);
        
        // Track LoRA adapter usage
        for lora_id in lora_ids {
            self.lora_usage_counter
                .with_label_values(&[lora_id])
                .inc();
        }
    }
}
```

## Integration with Existing Ecosystem

### **Model Format Support**

1. **GGUF (Primary)** - Use llama.cpp's proven quantization and loading
2. **SafeTensors** - For training and fine-tuning workflows  
3. **HuggingFace Hub** - Direct model downloads with authentication
4. **Ollama** - Community model distribution

### **Runtime Performance Comparison**

| Feature | llama.cpp | vLLM | Hyprstream |
|---------|-----------|------|------------|
| Model Loading | GGUF, fast | HF format | Both + LoRA |
| Memory Usage | Low | Medium | Low (sparse) |
| Throughput | Medium | High | High + adaptive |
| LoRA Support | Basic | Multi-LoRA | VDB-optimized |
| Quantization | INT4/INT8 | INT4/INT8/FP8 | + sparse quant |

### **API Compatibility**

```bash
# Standard OpenAI API
curl -X POST /v1/chat/completions \
  -d '{"model": "qwen3-1.7b", "messages": [...]}'

# Extended for LoRA selection
curl -X POST /v1/chat/completions \
  -d '{"model": "qwen3-1.7b", "messages": [...], 
       "lora_adapters": ["customer-support", "technical-writing"],
       "lora_weights": {"customer-support": 0.7, "technical-writing": 0.3}}'
```

## Implementation Timeline

### **Quick Start (Week 1-2): Proof of Concept**
```bash
# Immediate goals
1. Load Qwen3 GGUF model using llama.cpp bindings
2. Apply single LoRA adapter through our VDB layer
3. Generate text with basic LoRA modification
4. Demonstrate 10x+ compression with sparse storage
```

### **Production Ready (Week 8): Full System**
```bash
# Complete capabilities
1. Multi-LoRA inference with continuous batching
2. Auto-regressive training from user interactions
3. OpenAI-compatible API with LoRA extensions
4. Neural compression achieving 50x+ ratios
5. Production monitoring and metrics
```

This plan leverages proven runtimes while adding Hyprstream's unique VDB-first sparse adaptation capabilities, creating a production-ready system that builds on existing ecosystem strengths rather than reinventing the wheel.

## Risk Mitigation

### **Technical Risks**
- **Runtime Integration**: Start with llama.cpp (simpler FFI), add vLLM later
- **Memory Management**: Use proven quantization from existing runtimes
- **Performance**: Benchmark against llama.cpp and vLLM baselines

### **Ecosystem Risks**  
- **Model Compatibility**: Focus on GGUF format (most widely supported)
- **API Changes**: Maintain OpenAI compatibility, extend don't replace
- **Community Adoption**: Provide clear migration paths from existing tools

This approach transforms Hyprstream into a production-ready system by Q2 2024, leveraging battle-tested inference engines while adding unique sparse adaptation capabilities.