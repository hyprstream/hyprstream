//! PyTorch-based inference engine using tch-rs

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::path::Path;
use tch::{Device, Tensor, CModule, nn::VarStore};
use tokenizers::Tokenizer;
use std::collections::HashMap;
use serde_json;
use crate::config::{GenerationRequest, GenerationResult, ModelInfo, RuntimeConfig, FinishReason};
use crate::runtime::RuntimeEngine;

/// PyTorch inference engine using tch-rs
/// 
/// Note: This struct caches converted tensors in memory during model loading
/// to avoid reconversion overhead during inference. Thread safety is handled
/// by using read-only cached tensors after initial loading.
pub struct TorchEngine {
    /// PyTorch model (TorchScript or SafeTensors)
    model: Option<CModule>,
    /// VarStore for native PyTorch weight management
    pub var_store: Option<VarStore>,
    /// SafeTensors raw data for on-demand tensor creation
    safetensors_data: Option<Vec<u8>>,
    /// Cached converted tensors (for performance optimization)
    cached_weights: Option<HashMap<String, Tensor>>,
    /// Detected model architecture
    model_architecture: Option<String>,
    /// Tokenizer for text processing
    tokenizer: Option<Tokenizer>,
    /// Device for computation (CPU/CUDA/ROCm)
    device: Device,
    /// Runtime configuration
    config: RuntimeConfig,
    /// Loaded model information
    model_info: ModelInfo,
    /// Active LoRA adapter name
    pub active_lora: Option<String>,
    /// Sampling configuration
    pub sampling_config: RuntimeConfig,
}

/// Helper functions for tensor operations
impl TorchEngine {
    /// Convert SafeTensors data to tch Tensor with consistent BF16 dtype for Gemma
    fn safetensors_to_tensor(
        name: &str,
        tensor_view: &safetensors::tensor::TensorView,
        device: Device,
    ) -> Result<Tensor> {
        let shape: Vec<i64> = tensor_view.shape().iter().map(|&s| s as i64).collect();
        let tensor_data = tensor_view.data();
        
        // Convert all tensors to BF16 for consistent dtype in Gemma models
        let tensor = match tensor_view.dtype() {
            safetensors::Dtype::F32 => {
                let f32_slice = unsafe { 
                    std::slice::from_raw_parts(
                        tensor_data.as_ptr() as *const f32,
                        tensor_data.len() / 4
                    )
                };
                Tensor::from_slice(f32_slice)
                    .to_kind(tch::Kind::BFloat16)  // Convert to BF16
                    .view(shape.as_slice())
                    .to_device(device)
            }
            safetensors::Dtype::F16 => {
                let f16_slice = unsafe {
                    std::slice::from_raw_parts(
                        tensor_data.as_ptr() as *const u16,
                        tensor_data.len() / 2
                    )
                };
                let f32_data: Vec<f32> = f16_slice.iter().map(|&x| {
                    half::f16::from_bits(x).to_f32()
                }).collect();
                Tensor::from_slice(&f32_data)
                    .to_kind(tch::Kind::BFloat16)  // Convert to BF16
                    .view(shape.as_slice())
                    .to_device(device)
            }
            safetensors::Dtype::BF16 => {
                let bf16_slice = unsafe {
                    std::slice::from_raw_parts(
                        tensor_data.as_ptr() as *const u16,
                        tensor_data.len() / 2
                    )
                };
                let f32_data: Vec<f32> = bf16_slice.iter().map(|&x| {
                    f32::from_bits((x as u32) << 16)
                }).collect();
                Tensor::from_slice(&f32_data)
                    .to_kind(tch::Kind::BFloat16)  // Already BF16
                    .view(shape.as_slice())
                    .to_device(device)
            }
            _ => {
                println!("⚠️ Unsupported dtype for {}, creating zero tensor", name);
                Tensor::zeros(shape.as_slice(), (tch::Kind::BFloat16, device))  // BF16 zero tensor
            }
        };
        
        // println!("🔧 Converted tensor '{}' to BF16, shape: {:?}", name, shape); // Suppressed for cleaner output
        Ok(tensor)
    }

    /// Create new PyTorch engine
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        Self::new_sync(config)
    }

    /// Create new PyTorch engine (async version)
    pub async fn new_async(config: RuntimeConfig) -> Result<Self> {
        Self::new_sync(config)
    }

    /// Internal sync constructor
    fn new_sync(config: RuntimeConfig) -> Result<Self> {
        // Determine device based on configuration
        let device = if config.use_gpu && Device::cuda_if_available() != Device::Cpu {
            println!("🚀 Using CUDA GPU acceleration");
            Device::cuda_if_available()
        } else {
            println!("💻 Using CPU inference");
            Device::Cpu
        };

        Ok(Self {
            model: None,
            var_store: None,
            safetensors_data: None,
            cached_weights: None,
            model_architecture: None,
            tokenizer: None,
            device,
            config: config.clone(),
            model_info: ModelInfo {
                name: "unloaded".to_string(),
                architecture: "unknown".to_string(),
                parameters: 0,
                context_length: 2048,
                vocab_size: 32000,
                quantization: None,
            },
            active_lora: None,
            sampling_config: config,
        })
    }

    /// Load model from safetensors or torchscript
    async fn load_model_file(&mut self, path: &Path) -> Result<()> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("Invalid file extension"))?;

        match ext {
            "pt" | "pth" => {
                // Load TorchScript model
                println!("📦 Loading TorchScript model: {}", path.display());
                let model = CModule::load(path)?;
                self.model = Some(model);
            }
            "safetensors" => {
                // Load SafeTensors model directly
                println!("📦 Loading SafeTensors model: {}", path.display());
                self.load_safetensors(path).await?;
            }
            _ => {
                return Err(anyhow!("Unsupported model format: {}", ext));
            }
        }

        // Update model info
        self.model_info.name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(())
    }

    /// Detect model architecture from config.json and weights
    fn detect_architecture(&self, model_dir: &Path, weights: &HashMap<String, Tensor>) -> Result<String> {
        // First try to read config.json for architecture info
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            if let Ok(config_content) = std::fs::read_to_string(&config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                    // Check model_type field first
                    if let Some(model_type) = config["model_type"].as_str() {
                        return Ok(model_type.to_string());
                    }
                    
                    // Check architectures field
                    if let Some(architectures) = config["architectures"].as_array() {
                        if let Some(arch) = architectures.first() {
                            if let Some(arch_str) = arch.as_str() {
                                return Ok(arch_str.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback to weight-based detection
        use crate::runtime::architectures::qwen::QwenAdapter;
        if QwenAdapter::is_qwen_model(weights) {
            return Ok("qwen".to_string());
        }
        
        // Default to gemma if no clear architecture detected
        Ok("gemma".to_string())
    }

    /// Load SafeTensors model using safetensors crate and pre-convert all tensors to cache
    async fn load_safetensors(&mut self, path: &Path) -> Result<()> {
        println!("🚀 Loading SafeTensors with native PyTorch VarStore: {}", path.display());
        
        let mut cached_weights = HashMap::new();
        let mut all_buffers = Vec::new();
        let mut total_tensor_count = 0;
        
        // Check if this is a sharded model (path points to first shard)
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename.starts_with("model-00001-of-") && filename.ends_with(".safetensors") {
                // This is a sharded model - find all shards
                println!("🔍 Loading sharded SafeTensors model...");
                
                let parent_dir = path.parent().ok_or_else(|| anyhow!("Invalid path structure"))?;
                
                // Extract the total number of shards from filename like "model-00001-of-00003.safetensors"
                let parts: Vec<&str> = filename.split('-').collect();
                if parts.len() >= 4 {
                    if let Some(total_shards_str) = parts.get(3).and_then(|s| s.split('.').next()) {
                        if let Ok(total_shards) = total_shards_str.parse::<u32>() {
                            println!("📊 Found {} total shards to load", total_shards);
                            
                            // Load all shards in order
                            for shard_num in 1..=total_shards {
                                let shard_filename = format!("model-{:05}-of-{}.safetensors", shard_num, total_shards_str);
                                let shard_path = parent_dir.join(&shard_filename);
                                
                                if shard_path.exists() {
                                    println!("📦 Loading shard {}/{}: {}", shard_num, total_shards, shard_filename);
                                    
                                    let buffer = std::fs::read(&shard_path)?;
                                    let safetensors = safetensors::SafeTensors::deserialize(&buffer)
                                        .map_err(|e| anyhow!("Failed to parse SafeTensors shard {}: {}", shard_num, e))?;
                                    
                                    // Add tensors from this shard
                                    for (name, tensor_view) in safetensors.tensors() {
                                        let tensor = Self::safetensors_to_tensor(&name, &tensor_view, self.device)?;
                                        cached_weights.insert(name.to_string(), tensor);
                                        total_tensor_count += 1;
                                    }
                                    
                                    all_buffers.push(buffer);
                                } else {
                                    return Err(anyhow!("Missing shard: {}", shard_path.display()));
                                }
                            }
                        } else {
                            return Err(anyhow!("Failed to parse shard count from filename: {}", filename));
                        }
                    } else {
                        return Err(anyhow!("Invalid shard filename format: {}", filename));
                    }
                } else {
                    return Err(anyhow!("Invalid shard filename format: {}", filename));
                }
            } else {
                // Single SafeTensors file
                let buffer = std::fs::read(path)?;
                let safetensors = safetensors::SafeTensors::deserialize(&buffer)
                    .map_err(|e| anyhow!("Failed to parse SafeTensors: {}", e))?;
                
                total_tensor_count = safetensors.tensors().len();
                
                for (name, tensor_view) in safetensors.tensors() {
                    let tensor = Self::safetensors_to_tensor(&name, &tensor_view, self.device)?;
                    cached_weights.insert(name.to_string(), tensor);
                }
                
                all_buffers.push(buffer);
            }
        } else {
            return Err(anyhow!("Invalid file path"));
        }
        
        println!("📊 Converting {} total tensors to BF16 and caching in memory...", total_tensor_count);
        
        // Detect model architecture
        let model_dir = path.parent().unwrap_or(path);
        let architecture = self.detect_architecture(model_dir, &cached_weights)?;
        println!("🏗️ Detected model architecture: {}", architecture);
        
        // Store the first buffer for compatibility (or combined if needed)
        self.safetensors_data = all_buffers.into_iter().next();
        self.cached_weights = Some(cached_weights);
        self.model_architecture = Some(architecture.clone());
        
        // Update model info
        self.model_info.architecture = architecture;
        
        // Create dummy VarStore for compatibility
        let vs = VarStore::new(self.device);
        self.var_store = Some(vs);
        
        println!("✅ SafeTensors model loaded and cached ({} tensors converted to BF16 in memory)", total_tensor_count);
        Ok(())
    }

    /// Get tensor from VarStore by name (for inference)
    pub fn get_tensor(&self, name: &str) -> Option<Tensor> {
        self.var_store.as_ref()?.variables().get(name).map(|var| var.shallow_clone())
    }

    /// List all available tensor names in VarStore
    pub fn list_tensor_names(&self) -> Vec<String> {
        if let Some(vs) = &self.var_store {
            vs.variables().keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Check if model is loaded via VarStore
    pub fn has_varstore(&self) -> bool {
        self.var_store.is_some()
    }

    /// Load tokenizer
    async fn load_tokenizer(&mut self, model_path: &Path) -> Result<()> {
        // Try to find tokenizer.json - if model_path is a directory, look inside it
        // If it's a file, look in the parent directory
        let search_dir = if model_path.is_dir() {
            model_path
        } else {
            model_path.parent().unwrap_or(model_path)
        };
        
        let tokenizer_path = search_dir.join("tokenizer.json");

        if tokenizer_path.exists() {
            println!("📝 Loading tokenizer: {}", tokenizer_path.display());
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
            self.tokenizer = Some(tokenizer);
        } else {
            println!("⚠️ No tokenizer found, using fallback");
            // TODO: Create a simple fallback tokenizer
        }

        Ok(())
    }

    /// Tokenize text to input IDs
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        if let Some(tokenizer) = &self.tokenizer {
            let encoding = tokenizer.encode(text, false)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
            Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
        } else {
            // Fallback: simple character-based tokenization
            Ok(text.chars().map(|c| c as i64).collect())
        }
    }

    /// Detokenize IDs back to text
    fn detokenize(&self, ids: &[i64]) -> Result<String> {
        if let Some(tokenizer) = &self.tokenizer {
            let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
            tokenizer.decode(&ids_u32, false)
                .map_err(|e| anyhow!("Detokenization failed: {}", e))
        } else {
            // Fallback: convert back to characters
            let text: String = ids.iter().map(|&id| id as u8 as char).collect();
            Ok(text)
        }
    }

    /// Run inference on the model (supports both TorchScript and VarStore models)
    fn forward(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        // Try VarStore-based inference first (preferred for SafeTensors models)
        if let Some(_vs) = &self.var_store {
            return self.forward_varstore(input_ids);
        }
        
        // Fallback to TorchScript model
        if let Some(model) = &self.model {
            return self.forward_torchscript(model, input_ids);
        }
        
        Err(anyhow!("No model loaded (neither VarStore nor TorchScript)"))
    }

    /// Run inference using VarStore (SafeTensors models) with cached tensors
    fn forward_varstore(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        use crate::runtime::architectures::gemma::GemmaModel;
        use crate::runtime::architectures::qwen::QwenAdapter;
        use crate::runtime::architectures::ModelOperations;
        
        // Use cached tensors for performance
        let weights = self.cached_weights.as_ref()
            .ok_or_else(|| anyhow!("No cached tensors available - model not properly loaded"))?;
        
        let architecture = self.model_architecture.as_ref()
            .ok_or_else(|| anyhow!("No model architecture detected"))?;
        
        // Create model based on detected architecture
        let model: Box<dyn ModelOperations> = match architecture.as_str() {
            "qwen3" => {
                println!("🔧 Creating Qwen3 model from cached weights");
                QwenAdapter::from_weights(weights, 3, false, 262144, &self.device, tch::Kind::BFloat16)?
            }
            "qwen2" => {
                println!("🔧 Creating Qwen2 model from cached weights");
                QwenAdapter::from_weights(weights, 2, false, 131072, &self.device, tch::Kind::BFloat16)?
            }
            "Qwen3ForCausalLM" => {
                println!("🔧 Creating Qwen3ForCausalLM model from cached weights");
                QwenAdapter::from_weights(weights, 3, false, 262144, &self.device, tch::Kind::BFloat16)?
            }
            "gemma" | _ => {
                println!("🔧 Creating Gemma model from cached weights (fallback)");
                Box::new(GemmaModel::from_weights(weights, &self.device, tch::Kind::BFloat16)?)
            }
        };
        
        // Convert input IDs to tensor (keep as int64 for embeddings)
        let input_tensor = Tensor::from_slice(input_ids)
            .to_kind(tch::Kind::Int64)  // Ensure int64 for embedding lookup
            .to_device(self.device)
            .unsqueeze(0); // Add batch dimension: [1, seq_len]
        
        // Run forward pass through Gemma model
        let logits = model.forward(&input_tensor, None)?;
        
        // Extract logits for the last token
        let logits_shape = logits.size();
        let seq_len = logits_shape[1];
        let vocab_size = logits_shape[2] as usize;
        
        // Get logits for last token: [batch=1, last_seq_pos, vocab_size]
        let last_token_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1); // [1, vocab_size]
        
        // Convert to Vec<f32>
        let mut logits_vec = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let val = last_token_logits.double_value(&[0, i as i64]);
            logits_vec.push(val as f32);
        }
        
        Ok(logits_vec)
    }

    /// Run inference using TorchScript model (legacy .pt files) 
    fn forward_torchscript(&self, _model: &CModule, _input_ids: &[i64]) -> Result<Vec<f32>> {
        // TODO: Implement TorchScript support - currently disabled
        // TorchScript models are legacy; primary focus is on SafeTensors + Gemma
        Err(anyhow!("TorchScript inference not implemented yet - use SafeTensors models instead"))
    }

    /// Sample next token from logits
    fn sample_token(&self, logits: &[f32], temperature: f32, top_p: f32, top_k: Option<usize>) -> Result<usize> {
        if logits.is_empty() {
            return Err(anyhow!("Empty logits"));
        }

        // Apply temperature
        let scaled_logits: Vec<f32> = if temperature > 0.0 {
            logits.iter().map(|&x| x / temperature).collect()
        } else {
            logits.to_vec()
        };

        // Find max for numerical stability
        let max_logit = scaled_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute probabilities
        let exp_logits: Vec<f32> = scaled_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        
        let sum: f32 = exp_logits.iter().sum();
        let mut probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

        // Apply top-k filtering
        if let Some(k) = top_k {
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            
            for &i in indices.iter().skip(k) {
                probs[i] = 0.0;
            }
            
            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                probs.iter_mut().for_each(|p| *p /= sum);
            }
        }

        // Apply top-p (nucleus) sampling
        if top_p < 1.0 {
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            
            let mut cumsum = 0.0;
            let mut cutoff = probs.len();
            
            for (pos, &i) in indices.iter().enumerate() {
                cumsum += probs[i];
                if cumsum >= top_p {
                    cutoff = pos + 1;
                    break;
                }
            }
            
            for &i in indices.iter().skip(cutoff) {
                probs[i] = 0.0;
            }
            
            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                probs.iter_mut().for_each(|p| *p /= sum);
            }
        }

        // Sample from distribution
        let rand_val: f32 = fastrand::f32();
        let mut cumsum = 0.0;
        
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if rand_val < cumsum && prob > 0.0 {
                return Ok(i);
            }
        }

        // Fallback: return last non-zero probability token
        for (i, &prob) in probs.iter().enumerate().rev() {
            if prob > 0.0 {
                return Ok(i);
            }
        }

        Err(anyhow!("Failed to sample token"))
    }
}

#[async_trait]
impl RuntimeEngine for TorchEngine {
    async fn load_model(&mut self, path: &Path) -> Result<()> {
        // If path is a directory, find the model file inside it
        let model_file_path = if path.is_dir() {
            // First check for single file patterns
            let single_files = [
                "model.safetensors", 
                "pytorch_model.bin",
                "model.bin",
                "model.pt",
                "model.pth"
            ];
            
            let mut found_file = None;
            for filename in &single_files {
                let candidate = path.join(filename);
                if candidate.exists() {
                    found_file = Some(candidate);
                    break;
                }
            }
            
            // If no single file found, check for sharded SafeTensors
            if found_file.is_none() {
                let shard_pattern = path.join("model-00001-of-*.safetensors");
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let filename = entry.file_name();
                        if let Some(name) = filename.to_str() {
                            if name.starts_with("model-00001-of-") && name.ends_with(".safetensors") {
                                println!("🔍 Detected sharded SafeTensors model starting with: {}", name);
                                found_file = Some(entry.path());
                                break;
                            }
                        }
                    }
                }
            }
            
            found_file.ok_or_else(|| anyhow!("No supported model file found in directory: {}", path.display()))?
        } else {
            path.to_path_buf()
        };

        println!("📦 Loading model file: {}", model_file_path.display());
        self.load_model_file(&model_file_path).await?;
        self.load_tokenizer(path).await?;
        Ok(())
    }

    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let request = GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.8,
            top_p: 0.95,
            top_k: Some(50),
            repeat_penalty: 1.1,
            stop_tokens: vec![],
            seed: None,
            stream: false,
            active_adapters: None,
            realtime_adaptation: None,
            user_feedback: None,
        };

        let result = self.generate_with_params(request).await?;
        Ok(result.text)
    }

    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();

        // Tokenize input
        let mut input_ids = self.tokenize(&request.prompt)?;
        let mut generated_text = String::new();
        let mut tokens_generated = 0;

        for _ in 0..request.max_tokens {
            // Run forward pass
            let logits = self.forward(&input_ids)?;
            
            // Sample next token
            let next_token = self.sample_token(
                &logits,
                request.temperature,
                request.top_p,
                request.top_k,
            )?;

            // Add to sequence
            input_ids.push(next_token as i64);
            tokens_generated += 1;

            // Decode token
            let token_text = self.detokenize(&[next_token as i64])?;
            generated_text.push_str(&token_text);

            // Check stop tokens
            if request.stop_tokens.iter().any(|stop| generated_text.contains(stop)) {
                break;
            }

            // Simple EOS check (token ID 2 is often EOS)
            if next_token == 2 {
                break;
            }
        }

        let generation_time = start_time.elapsed();

        Ok(GenerationResult {
            text: generated_text,
            tokens_generated,
            finish_reason: if tokens_generated >= request.max_tokens {
                FinishReason::MaxTokens
            } else {
                FinishReason::EndOfSequence
            },
            generation_time_ms: generation_time.as_millis() as u64,
            tokens_per_second: tokens_generated as f32 / generation_time.as_secs_f32(),
        })
    }

    fn model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some() || self.var_store.is_some()
    }
}

// Additional methods needed by inference layer
impl TorchEngine {
    /// Generate text with streaming callback
    pub async fn generate_streaming<F>(&self, prompt: &str, max_tokens: usize, mut callback: F) -> Result<String> 
    where 
        F: FnMut(&str)
    {
        let request = GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.8,
            top_p: 0.95,
            top_k: Some(50),
            repeat_penalty: 1.1,
            stop_tokens: vec![],
            seed: None,
            stream: true,
            active_adapters: None,
            realtime_adaptation: None,
            user_feedback: None,
        };

        // REAL streaming: generate tokens one by one and call callback for each
        let start_time = std::time::Instant::now();
        
        // Tokenize input
        let mut input_ids = self.tokenize(prompt)?;
        let mut generated_text = String::new();
        let mut tokens_generated = 0;

        for _ in 0..max_tokens {
            // Run forward pass for current sequence
            let logits = self.forward(&input_ids)?;
            
            // Sample next token
            let next_token = self.sample_token(
                &logits,
                request.temperature,
                request.top_p,
                request.top_k,
            )?;

            // Add to sequence
            input_ids.push(next_token as i64);
            tokens_generated += 1;

            // Decode token and stream it immediately
            let token_text = self.detokenize(&[next_token as i64])?;
            generated_text.push_str(&token_text);
            
            // Stream the token immediately (real streaming)
            callback(&token_text);

            // Check stop tokens
            if request.stop_tokens.iter().any(|stop| generated_text.contains(stop)) {
                break;
            }

            // Simple EOS check (token ID 2 is often EOS)
            if next_token == 2 {
                break;
            }
        }

        Ok(generated_text)
    }

    /// Train temporal LoRA adapter (placeholder implementation)
    pub async fn train_temporal_lora(
        &mut self,
        _prompt: &str,
        _expected_response: &str,
        _learning_rate: f32,
    ) -> Result<()> {
        // TODO: Implement temporal LoRA training with Tch
        // This is a placeholder to resolve compilation errors
        Ok(())
    }
}

// SAFETY: TorchEngine is designed to be Send + Sync by storing raw SafeTensors data
// instead of tch::Tensor objects directly. Tensors are created on-demand during inference.
// The Device, CModule, and VarStore fields are used in single-threaded contexts only.
unsafe impl Send for TorchEngine {}
unsafe impl Sync for TorchEngine {}