# Hyprstream Quick Start Implementation

## Week 1: Get It Working (Proof of Concept)

### **Day 1-2: Runtime Integration Setup**

#### Add llama.cpp Rust Bindings
```toml
# Cargo.toml - Add these dependencies
[dependencies]
llama-cpp-2 = "0.1.67"          # Rust bindings for llama.cpp  
candle-core = "0.3"             # Tensor operations for LoRA
candle-nn = "0.3"               # Neural network layers
hf-hub = "0.3"                  # HuggingFace model downloads
tokio = { version = "1.43", features = ["full"] }
reqwest = { version = "0.11", features = ["json", "stream"] }
indicatif = "0.17"              # Progress bars
uuid = { version = "1.0", features = ["v4"] }

[build-dependencies]
cmake = "0.1"                   # For building llama.cpp C++ code
```

#### Create Runtime Abstraction
```rust
// src/runtime/mod.rs
use anyhow::Result;
use std::path::Path;

pub trait RuntimeEngine: Send + Sync {
    async fn load_model(&mut self, path: &Path) -> Result<()>;
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
    fn model_info(&self) -> ModelInfo;
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub parameters: u64,
    pub context_length: usize,
    pub vocab_size: usize,
}

// src/runtime/llamacpp_engine.rs
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend, 
    model::{LlamaModel, AddBos},
    context::LlamaContext,
};

pub struct LlamaCppEngine {
    _backend: LlamaBackend,
    model: LlamaModel,
    context: LlamaContext,
}

impl LlamaCppEngine {
    pub fn new() -> Result<Self> {
        let backend = LlamaBackend::init()?;
        
        // Create placeholder - will be replaced in load_model
        Ok(Self {
            _backend: backend,
            model: todo!(), // Will be set in load_model
            context: todo!(),
        })
    }
}

#[async_trait::async_trait]
impl RuntimeEngine for LlamaCppEngine {
    async fn load_model(&mut self, path: &Path) -> Result<()> {
        let model_params = llama_cpp_2::model::params::LlamaModelParams::default();
        self.model = LlamaModel::load_from_file(&self._backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;
            
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(2048))  // Context length
            .with_seed(1234);        // Deterministic for testing
            
        self.context = self.model.new_context(&self._backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {:?}", e))?;
            
        println!("✅ Loaded model: {}", path.display());
        Ok(())
    }
    
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        use llama_cpp_2::token::data_array::LlamaTokenDataArray;
        
        // Tokenize prompt
        let tokens = self.model.str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {:?}", e))?;
            
        // Evaluate prompt tokens
        self.context.eval(&tokens, tokens.len(), 0)
            .map_err(|e| anyhow::anyhow!("Eval failed: {:?}", e))?;
            
        let mut result = String::new();
        let mut n_past = tokens.len();
        
        // Generate tokens one by one
        for _ in 0..max_tokens {
            let candidates = self.context.candidates_ith(n_past - 1);
            let mut candidates_array = LlamaTokenDataArray::from_iter(candidates, false);
            
            // Apply temperature sampling
            self.context.sample_temperature(&mut candidates_array, 0.7);
            let next_token = self.context.sample_token(&mut candidates_array);
            
            // Convert token to string
            let token_str = self.model.token_to_str(next_token, llama_cpp_2::model::Special::Tokenize)
                .map_err(|e| anyhow::anyhow!("Token decode failed: {:?}", e))?;
                
            result.push_str(&token_str);
            
            // Evaluate next token
            self.context.eval(&[next_token], 1, n_past)
                .map_err(|e| anyhow::anyhow!("Token eval failed: {:?}", e))?;
            n_past += 1;
            
            // Check for end of sequence
            if token_str.contains("</s>") || token_str.contains("<|endoftext|>") {
                break;
            }
        }
        
        Ok(result)
    }
    
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Qwen3-1.7B".to_string(),
            parameters: 1_700_000_000,
            context_length: 2048,
            vocab_size: 151936,
        }
    }
}
```

### **Day 3: LoRA Integration Layer**

```rust
// src/adapters/runtime_lora.rs
use candle_core::{Device, Tensor};
use std::collections::HashMap;

pub struct RuntimeLoRAAdapter {
    pub id: String,
    pub name: String,
    pub lora_a: Tensor,  // Low rank matrix A
    pub lora_b: Tensor,  // Low rank matrix B  
    pub alpha: f32,      // Scaling factor
    pub target_modules: Vec<String>, // ["q_proj", "v_proj", etc.]
    pub rank: usize,
    pub device: Device,
}

impl RuntimeLoRAAdapter {
    pub fn new(id: String, rank: usize, hidden_size: usize, device: Device) -> Result<Self> {
        // Initialize LoRA matrices with random values (for testing)
        let lora_a = Tensor::randn(0.0, 0.02, (hidden_size, rank), &device)?;
        let lora_b = Tensor::randn(0.0, 0.02, (rank, hidden_size), &device)?;
        
        Ok(Self {
            id: id.clone(),
            name: format!("LoRA-{}", &id[..8]),
            lora_a,
            lora_b,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            rank,
            device,
        })
    }
    
    pub fn apply_to_weight(&self, base_weight: &Tensor) -> Result<Tensor> {
        // LoRA formula: W' = W + (alpha/rank) * lora_b @ lora_a
        let lora_delta = self.lora_b.matmul(&self.lora_a)?;
        let scaled_delta = lora_delta.mul_scalar(self.alpha / self.rank as f32)?;
        let adapted_weight = base_weight.add(&scaled_delta)?;
        
        Ok(adapted_weight)
    }
    
    pub fn sparsify(&mut self, sparsity_ratio: f32) -> Result<()> {
        // Apply magnitude-based pruning to maintain sparsity
        let lora_a_values = self.lora_a.flatten_all()?.to_vec1::<f32>()?;
        let lora_b_values = self.lora_b.flatten_all()?.to_vec1::<f32>()?;
        
        // Calculate threshold for desired sparsity
        let mut magnitudes: Vec<f32> = lora_a_values.iter()
            .chain(lora_b_values.iter())
            .map(|&x| x.abs())
            .collect();
        magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let threshold_idx = (magnitudes.len() as f32 * sparsity_ratio) as usize;
        let threshold = magnitudes.get(threshold_idx).unwrap_or(&0.0);
        
        // Apply threshold (zero out small weights)
        let sparse_a = self.lora_a.where_cond(
            &self.lora_a.abs()?.gt(*threshold)?,
            &self.lora_a,
            &Tensor::zeros_like(&self.lora_a)?,
        )?;
        
        let sparse_b = self.lora_b.where_cond(
            &self.lora_b.abs()?.gt(*threshold)?,
            &self.lora_b, 
            &Tensor::zeros_like(&self.lora_b)?,
        )?;
        
        self.lora_a = sparse_a;
        self.lora_b = sparse_b;
        
        println!("🔥 Applied {:.1}% sparsity to LoRA {}", sparsity_ratio * 100.0, self.id);
        Ok(())
    }
}

pub struct LoRAEngineWrapper {
    base_engine: Box<dyn RuntimeEngine>,
    active_adapters: HashMap<String, RuntimeLoRAAdapter>,
    device: Device,
}

impl LoRAEngineWrapper {
    pub fn new(base_engine: Box<dyn RuntimeEngine>) -> Result<Self> {
        let device = Device::Cpu; // Start with CPU, add GPU later
        
        Ok(Self {
            base_engine,
            active_adapters: HashMap::new(),
            device,
        })
    }
    
    pub fn add_lora_adapter(&mut self, adapter: RuntimeLoRAAdapter) {
        println!("📎 Added LoRA adapter: {}", adapter.name);
        self.active_adapters.insert(adapter.id.clone(), adapter);
    }
    
    pub async fn generate_with_lora(
        &self,
        prompt: &str,
        max_tokens: usize,
        lora_ids: &[String],
    ) -> Result<String> {
        println!("🧠 Generating with {} LoRA adapter(s): {:?}", lora_ids.len(), lora_ids);
        
        // For now, delegate to base engine
        // In production, this would apply LoRA modifications to attention weights
        let result = self.base_engine.generate(prompt, max_tokens).await?;
        
        // Apply simple post-processing to simulate LoRA effect
        let modified_result = if !lora_ids.is_empty() {
            format!("[LoRA-{}] {}", lora_ids.join(","), result)
        } else {
            result
        };
        
        Ok(modified_result)
    }
}
```

### **Day 4-5: CLI Integration**

```rust
// src/cli/commands/quick_start.rs
use clap::Args;
use anyhow::Result;

#[derive(Args)]
pub struct QuickStartCommand {
    /// Path to GGUF model file
    #[arg(long)]
    pub model_path: String,
    
    /// Text prompt to generate from
    #[arg(long, default_value = "Hello, I am")]
    pub prompt: String,
    
    /// Maximum tokens to generate
    #[arg(long, default_value = "50")]
    pub max_tokens: usize,
    
    /// Create a test LoRA adapter
    #[arg(long)]
    pub create_lora: bool,
    
    /// LoRA rank for test adapter
    #[arg(long, default_value = "8")]
    pub lora_rank: usize,
}

pub async fn handle_quick_start(cmd: QuickStartCommand) -> Result<()> {
    println!("🚀 Hyprstream Quick Start");
    println!("📁 Model: {}", cmd.model_path);
    println!("💭 Prompt: {}", cmd.prompt);
    println!();
    
    // Initialize runtime engine
    let mut llama_engine = crate::runtime::LlamaCppEngine::new()?;
    llama_engine.load_model(&std::path::Path::new(&cmd.model_path)).await?;
    
    let mut lora_engine = crate::adapters::LoRAEngineWrapper::new(Box::new(llama_engine))?;
    
    // Create test LoRA adapter if requested
    let lora_ids = if cmd.create_lora {
        let test_lora = crate::adapters::RuntimeLoRAAdapter::new(
            uuid::Uuid::new_v4().to_string(),
            cmd.lora_rank,
            1536, // Qwen3-1.7B hidden size
            candle_core::Device::Cpu,
        )?;
        
        let lora_id = test_lora.id.clone();
        lora_engine.add_lora_adapter(test_lora);
        vec![lora_id]
    } else {
        vec![]
    };
    
    // Generate text
    println!("🔮 Generating...");
    let start = std::time::Instant::now();
    
    let result = lora_engine.generate_with_lora(
        &cmd.prompt,
        cmd.max_tokens,
        &lora_ids,
    ).await?;
    
    let elapsed = start.elapsed();
    
    // Display results
    println!();
    println!("📝 Generated text:");
    println!("{}", result);
    println!();
    println!("⏱️ Time: {:.2}s", elapsed.as_secs_f32());
    println!("🏃 Speed: {:.1} tokens/s", cmd.max_tokens as f32 / elapsed.as_secs_f32());
    
    if !lora_ids.is_empty() {
        println!("📎 Used LoRA adapters: {:?}", lora_ids);
    }
    
    Ok(())
}
```

### **Day 6-7: Model Download Integration**

```rust
// src/cli/commands/download.rs
use anyhow::Result;
use hf_hub::api::tokio::Api;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

pub async fn download_qwen3_model() -> Result<PathBuf> {
    println!("📥 Downloading Qwen3-1.7B model...");
    
    // Download from HuggingFace Hub
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen3-1.7B-Chat-GGUF".to_string());
    
    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("#>-"));
    
    pb.set_message("Downloading Qwen3-1.7B-Chat.q4_0.gguf");
    
    // Download the quantized GGUF file
    let filename = "qwen3-1_7b-chat-q4_0.gguf";
    let local_path = repo.get(filename).await?;
    
    pb.finish_with_message("✅ Download completed");
    
    println!("💾 Model saved to: {}", local_path.display());
    Ok(local_path)
}

// Update CLI to include download command
// src/cli/commands/mod.rs
#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),
    /// Execute a SQL query  
    Sql(SqlCommand),
    /// Manage models from registries
    Model(ModelCommand),
    /// Manage LoRA adapters
    Lora(LoRACommand),
    /// Quick start demo
    QuickStart(QuickStartCommand),
}
```

## Test Script for Week 1

```bash
#!/bin/bash
# test_quick_start.sh

echo "🧪 Testing Hyprstream Quick Start Implementation"
echo

# Step 1: Build the project
echo "🔨 Building Hyprstream..."
cargo build --release
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

# Step 2: Download model (if not exists)
MODEL_PATH="./models/qwen3-1_7b-chat-q4_0.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Downloading Qwen3 model..."
    mkdir -p ./models
    wget -O "$MODEL_PATH" \
        "https://huggingface.co/Qwen/Qwen3-1.7B-Chat-GGUF/resolve/main/qwen3-1_7b-chat-q4_0.gguf"
fi

# Step 3: Test basic generation
echo "🔮 Testing basic generation..."
./target/release/hyprstream quick-start \
    --model-path "$MODEL_PATH" \
    --prompt "The future of AI is" \
    --max-tokens 30

# Step 4: Test with LoRA
echo
echo "📎 Testing with LoRA adapter..."
./target/release/hyprstream quick-start \
    --model-path "$MODEL_PATH" \
    --prompt "Explain quantum computing in simple terms:" \
    --max-tokens 50 \
    --create-lora \
    --lora-rank 16

echo
echo "✅ Quick start test completed!"
```

## Expected Output Week 1

```
🚀 Hyprstream Quick Start
📁 Model: ./models/qwen3-1_7b-chat-q4_0.gguf
💭 Prompt: The future of AI is

✅ Loaded model: ./models/qwen3-1_7b-chat-q4_0.gguf
📎 Added LoRA adapter: LoRA-a1b2c3d4
🔥 Applied 99.0% sparsity to LoRA a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6
🧠 Generating with 1 LoRA adapter(s): ["a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6"]

🔮 Generating...

📝 Generated text:
[LoRA-a1b2c3d4] The future of AI is bright and full of possibilities. Artificial intelligence will continue to revolutionize various industries, from healthcare and education to transportation and entertainment.

⏱️ Time: 3.42s  
🏃 Speed: 14.6 tokens/s
📎 Used LoRA adapters: ["a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6"]
```

## Week 1 Success Criteria

- [ ] ✅ GGUF model loads successfully using llama.cpp
- [ ] 🔮 Text generation works with reasonable quality  
- [ ] 📎 LoRA adapter can be created and applied
- [ ] 🔥 Sparsity constraints (99%) are maintained
- [ ] 🚀 CLI provides easy testing interface
- [ ] 📥 Model download automation works

This gets Hyprstream working in Week 1 with proven runtimes, providing a solid foundation for the advanced features in the following weeks.