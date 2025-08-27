//! PyTorch-based inference engine using tch-rs

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::path::Path;
use tch::{Device, Tensor, nn::VarStore};
use tokenizers::Tokenizer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use serde_json;
use crate::config::{GenerationRequest, GenerationResult, ModelInfo, RuntimeConfig, FinishReason};
use crate::runtime::RuntimeEngine;

/// Qwen2 special token IDs (based on official HuggingFace implementation)
#[derive(Debug, Clone)]
pub struct QwenSpecialTokens {
    /// "<|endoftext|>" - Used as BOS, EOS, and PAD token
    pub endoftext: u32,
    /// "<|im_start|>" - Instruction/message start
    pub im_start: u32,
    /// "<|im_end|>" - Instruction/message end (primary EOS for conversations)
    pub im_end: u32,
    /// Additional special tokens for multimodal support
    pub object_ref_start: u32,
    pub object_ref_end: u32,
    pub box_start: u32,
    pub box_end: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    pub vision_pad: u32,
    pub image_pad: u32,
    pub video_pad: u32,
}

impl Default for QwenSpecialTokens {
    fn default() -> Self {
        Self {
            endoftext: 151643,      // "<|endoftext|>"
            im_start: 151644,       // "<|im_start|>" 
            im_end: 151645,         // "<|im_end|>"
            object_ref_start: 151646,
            object_ref_end: 151647,
            box_start: 151648,
            box_end: 151649,
            vision_start: 151652,
            vision_end: 151653,
            vision_pad: 151654,
            image_pad: 151655,
            video_pad: 151656,
        }
    }
}
use crate::runtime::architectures::ModelOperations;

/// Basic context state for tracking generation state
#[derive(Debug, Clone)]
pub struct ContextState {
    /// Current sequence length
    pub sequence_length: usize,
    /// Context window size
    pub context_window: usize,
    /// Whether model is initialized
    pub initialized: bool,
}

/// PyTorch inference engine using tch-rs
/// 
/// Thread-safe implementation using proper synchronization primitives.
/// All mutable state is protected by mutexes with poisoning recovery.
pub struct TorchEngine {
    /// VarStore for native PyTorch weight management - not thread safe, requires external sync
    var_store: Arc<Mutex<Option<VarStore>>>,
    /// SafeTensors raw data for on-demand tensor creation - thread safe after initialization
    safetensors_data: Arc<Mutex<Option<Vec<u8>>>>,
    /// Cached converted tensors (for performance optimization) - thread safe after initialization
    cached_weights: Arc<Mutex<Option<HashMap<String, Tensor>>>>,
    /// Detected model architecture - thread safe after initialization
    model_architecture: Arc<Mutex<Option<String>>>,
    /// Persistent model instance to avoid recreation on every forward pass
    /// Using Arc<Mutex<>> for interior mutability since ModelOperations has mutable methods
    persistent_model: Option<Arc<Mutex<Box<dyn ModelOperations>>>>,
    /// Basic KV cache storage for context tracking - thread safe with mutex
    context_state: Arc<Mutex<Option<ContextState>>>,
    /// Tokenizer for text processing - thread safe after initialization
    tokenizer: Arc<Mutex<Option<Tokenizer>>>,
    /// Device for computation (CPU/CUDA/ROCm) - immutable after construction
    device: Device,
    /// Runtime configuration - immutable after construction
    config: RuntimeConfig,
    /// Loaded model information - protected by mutex
    model_info: Arc<Mutex<ModelInfo>>,
    /// Active LoRA adapter name - thread safe with mutex
    pub active_lora: Arc<Mutex<Option<String>>>,
    /// Sampling configuration - immutable after construction
    sampling_config: RuntimeConfig,
    /// Qwen special tokens for proper conversation handling - immutable after construction
    special_tokens: QwenSpecialTokens,
}

/// Helper functions for tensor operations
impl TorchEngine {
    /// Handle mutex poisoning with recovery
    fn handle_poison<T>(&self, result: Result<T, PoisonError<T>>) -> Result<T, anyhow::Error> {
        match result {
            Ok(guard) => Ok(guard),
            Err(poisoned) => {
                tracing::warn!("Mutex poisoned, recovering by taking inner value");
                Ok(poisoned.into_inner())
            }
        }
    }
    
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
            var_store: Arc::new(Mutex::new(None)),
            safetensors_data: Arc::new(Mutex::new(None)),
            cached_weights: Arc::new(Mutex::new(None)),
            model_architecture: Arc::new(Mutex::new(None)),
            persistent_model: None,
            context_state: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            device,
            config: config.clone(),
            model_info: Arc::new(Mutex::new(ModelInfo {
                name: "unloaded".to_string(),
                architecture: "unknown".to_string(),
                parameters: 0,
                context_length: 2048,
                vocab_size: 32000,
                quantization: None,
            })),
            active_lora: Arc::new(Mutex::new(None)),
            sampling_config: config,
            special_tokens: QwenSpecialTokens::default(),
        })
    }

    /// Load model from safetensors or torchscript
    async fn load_model_file(&mut self, path: &Path) -> Result<()> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("Invalid file extension"))?;

        match ext {
            "pt" | "pth" => {
                return Err(anyhow!("TorchScript models (.pt/.pth) are no longer supported. Please use SafeTensors format (.safetensors) instead."));
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

        // Update model info with thread safety
        let mut model_info_guard = self.handle_poison(self.model_info.lock())?;
        model_info_guard.name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(())
    }

    /// Detect model architecture from config.json and weights - thread safe
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
        
        // Try weight-based detection as last resort
        use crate::runtime::architectures::qwen::QwenAdapter;
        if QwenAdapter::is_qwen_model(weights) {
            return Ok("qwen".to_string());
        }
        
        // No architecture could be detected - fail with clear error
        Err(anyhow!(
            "Could not detect model architecture. Ensure config.json exists with valid 'model_type' or 'architectures' field. \
            Supported architectures: qwen3, qwen2, Qwen3ForCausalLM, gemma"
        ))
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
        
        // Store the first buffer for compatibility (or combined if needed) - thread safe
        {
            let mut safetensors_guard = self.handle_poison(self.safetensors_data.lock())?;
            *safetensors_guard = all_buffers.into_iter().next();
        }
        {
            let mut cached_weights_guard = self.handle_poison(self.cached_weights.lock())?;
            *cached_weights_guard = Some(cached_weights);
        }
        {
            let mut arch_guard = self.handle_poison(self.model_architecture.lock())?;
            *arch_guard = Some(architecture.clone());
        }
        
        // Determine context window first before moving architecture
        let context_window = match architecture.as_str() {
            "qwen3" => 262144,
            "qwen2" => 131072,
            _ => 32768,
        };
        
        // Update model info with thread safety
        {
            let mut model_info_guard = self.handle_poison(self.model_info.lock())?;
            model_info_guard.architecture = architecture;
        }
        
        // Create dummy VarStore for compatibility - thread safe
        {
            let vs = VarStore::new(self.device);
            let mut var_store_guard = self.handle_poison(self.var_store.lock())?;
            *var_store_guard = Some(vs);
        }
        
        // Initialize persistent model once during loading
        self.initialize_persistent_model()?;
        
        // Initialize context state - thread safe
        {
            let mut context_guard = self.handle_poison(self.context_state.lock())?;
            *context_guard = Some(ContextState {
                sequence_length: 0,
                context_window,
                initialized: true,
            });
        }
        
        println!("✅ SafeTensors model loaded and cached ({} tensors converted to BF16 in memory)", total_tensor_count);
        println!("🚀 Persistent model initialized - ready for efficient inference");
        Ok(())
    }

    /// Get tensor from VarStore by name (for inference) - thread safe
    pub fn get_tensor(&self, name: &str) -> Option<Tensor> {
        let var_store_guard = self.handle_poison(self.var_store.lock()).ok()?;
        let vs = var_store_guard.as_ref()?;
        vs.variables().get(name).map(|var| var.shallow_clone())
    }

    /// List all available tensor names in VarStore - thread safe
    pub fn list_tensor_names(&self) -> Vec<String> {
        if let Ok(var_store_guard) = self.handle_poison(self.var_store.lock()) {
            if let Some(vs) = var_store_guard.as_ref() {
                vs.variables().keys().cloned().collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Check if model is loaded via VarStore - thread safe
    pub fn has_varstore(&self) -> bool {
        self.handle_poison(self.var_store.lock())
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Initialize the persistent model instance from cached weights - thread safe
    fn initialize_persistent_model(&mut self) -> Result<()> {
        use crate::runtime::architectures::gemma::GemmaModel;
        use crate::runtime::architectures::qwen::QwenAdapter;
        
        // Get cached weights with thread safety - clone to avoid lifetime issues
        let cached_weights_guard = self.handle_poison(self.cached_weights.lock())?;
        let weights = match cached_weights_guard.as_ref() {
            Some(w) => w.clone(),
            None => {
                return Err(anyhow!("No cached tensors available for persistent model initialization"));
            }
        };
        // cached_weights_guard automatically dropped here
        
        // Get architecture with thread safety - clone to avoid lifetime issues  
        let arch_guard = self.handle_poison(self.model_architecture.lock())?;
        let architecture = match arch_guard.as_ref() {
            Some(a) => a.clone(),
            None => {
                return Err(anyhow!("No model architecture detected for persistent model initialization"));
            }
        };
        // arch_guard automatically dropped here
        
        // Create model based on detected architecture (only once!)
        let model: Box<dyn ModelOperations> = match architecture.as_str() {
            "qwen3" => {
                println!("🏗️ Initializing persistent Qwen3 model from cached weights");
                QwenAdapter::from_weights(&weights, 3, false, 262144, &self.device, tch::Kind::BFloat16)?
            }
            "qwen2" => {
                println!("🏗️ Initializing persistent Qwen2 model from cached weights");
                QwenAdapter::from_weights(&weights, 2, false, 131072, &self.device, tch::Kind::BFloat16)?
            }
            "Qwen3ForCausalLM" => {
                println!("🏗️ Initializing persistent Qwen3ForCausalLM model from cached weights");
                QwenAdapter::from_weights(&weights, 3, false, 262144, &self.device, tch::Kind::BFloat16)?
            }
            "gemma" => {
                println!("🏗️ Initializing persistent Gemma model from cached weights");
                Box::new(GemmaModel::from_weights(&weights, &self.device, tch::Kind::BFloat16)?)
            }
            unknown => {
                return Err(anyhow!(
                    "Unsupported model architecture: '{}'. Supported architectures: qwen3, qwen2, Qwen3ForCausalLM, gemma",
                    unknown
                ));
            }
        };
        
        self.persistent_model = Some(Arc::new(Mutex::new(model)));
        println!("✅ Persistent model initialized successfully");
        Ok(())
    }

    /// Load tokenizer - thread safe
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
            
            // Thread safe assignment
            let mut tokenizer_guard = self.handle_poison(self.tokenizer.lock())?;
            *tokenizer_guard = Some(tokenizer);
        } else {
            return Err(anyhow!(
                "Tokenizer not found at {}. A proper tokenizer.json file is required for inference.",
                tokenizer_path.display()
            ));
        }

        Ok(())
    }

    /// Tokenize text to input IDs - thread safe
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        let tokenizer_guard = self.handle_poison(self.tokenizer.lock())?;
        let tokenizer = tokenizer_guard.as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))?;
        
        let encoding = tokenizer.encode(text, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        
        if token_ids.is_empty() {
            return Err(anyhow!("Tokenization produced empty token sequence for text: '{}'", text));
        }
        
        tracing::debug!("Tokenized '{}' -> {:?}", text, token_ids);
        Ok(token_ids)
    }

    /// Format text with Qwen2 chat template (basic implementation)
    fn format_chat_message(&self, system: Option<&str>, user: &str) -> String {
        let mut formatted = String::new();
        
        // Add system message if provided
        if let Some(system_msg) = system {
            formatted.push_str("<|im_start|>system\n");
            formatted.push_str(system_msg);
            formatted.push_str("<|im_end|>\n");
        }
        
        // Add user message
        formatted.push_str("<|im_start|>user\n");
        formatted.push_str(user);
        formatted.push_str("<|im_end|>\n");
        
        // Add assistant prompt
        formatted.push_str("<|im_start|>assistant\n");
        
        formatted
    }
    
    /// Check if a token ID is a special EOS token for Qwen
    fn is_eos_token(&self, token_id: usize) -> bool {
        let token_id_u32 = token_id as u32;
        // Qwen can end on either im_end (for conversations) or endoftext (for completion)
        token_id_u32 == self.special_tokens.im_end || 
        token_id_u32 == self.special_tokens.endoftext
    }

    /// Detokenize IDs back to text - thread safe
    fn detokenize(&self, ids: &[i64]) -> Result<String> {
        let tokenizer_guard = self.handle_poison(self.tokenizer.lock())?;
        let tokenizer = tokenizer_guard.as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))?;
        
        if ids.is_empty() {
            return Ok(String::new());
        }
        
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        let decoded_text = tokenizer.decode(&ids_u32, false)
            .map_err(|e| anyhow!("Detokenization failed for token IDs {:?}: {}", ids, e))?;
        
        tracing::debug!("Detokenized {:?} -> '{}'", ids, decoded_text);
        Ok(decoded_text)
    }

    /// Run inference on the model (supports both TorchScript and VarStore models) - thread safe
    fn forward(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        // Try VarStore-based inference first (preferred for SafeTensors models)
        if self.has_varstore() {
            return self.forward_varstore(input_ids);
        }
        
        Err(anyhow!("No model loaded - call load_model() first"))
    }

    /// Run inference using VarStore (SafeTensors models) with persistent model - thread safe
    fn forward_varstore(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        // Use the persistent model instance - NO recreation!
        let model_arc = self.persistent_model.as_ref()
            .ok_or_else(|| anyhow!("Persistent model not initialized - call load_model() first"))?;
        
        // Verify context state with thread safety
        {
            let context_guard = self.handle_poison(self.context_state.lock())?;
            let _context_state = context_guard.as_ref()
                .ok_or_else(|| anyhow!("Context state not initialized"))?;
        }
        
        // Context initialization check moved to is_persistent_model_ready()
        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized"));
        }
        
        // Convert input IDs to tensor (keep as int64 for embeddings)
        let input_tensor = Tensor::from_slice(input_ids)
            .to_kind(tch::Kind::Int64)  // Ensure int64 for embedding lookup
            .to_device(self.device)
            .unsqueeze(0); // Add batch dimension: [1, seq_len]
        
        // Lock the model and run forward pass (efficient!) with poison recovery
        let model = self.handle_poison(model_arc.lock())?;
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
    
    /// Run optimized inference with KV caching - only process new tokens
    fn forward_cached(&self, input_ids: &[i64], start_pos: usize, use_cache: bool) -> Result<Vec<f32>> {
        // Use the persistent model instance
        let model_arc = self.persistent_model.as_ref()
            .ok_or_else(|| anyhow!("Persistent model not initialized"))?;
        
        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized"));
        }
        
        // For KV cached generation, only process new tokens after initial prompt
        let tokens_to_process = if use_cache && start_pos > 0 {
            // Only process the last token (the newly generated one)
            &input_ids[input_ids.len()-1..]
        } else {
            // Process all tokens (initial prompt or no caching)
            input_ids
        };
        
        // Convert to tensor
        let input_tensor = Tensor::from_slice(tokens_to_process)
            .to_kind(tch::Kind::Int64)
            .to_device(self.device)
            .unsqueeze(0); // [1, seq_len]
        
        // Run forward pass with position info for proper KV cache usage
        let model = self.handle_poison(model_arc.lock())?;
        
        // TODO: Modify ModelOperations trait to accept start_pos parameter
        // For now, use regular forward which will still benefit from internal caching
        let logits = model.forward(&input_tensor, None)?;
        
        // Extract logits for the last token
        let logits_shape = logits.size();
        let seq_len = logits_shape[1];
        let vocab_size = logits_shape[2] as usize;
        
        // Get logits for last token
        let last_token_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1);
        
        // Convert to Vec<f32>
        let mut logits_vec = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let val = last_token_logits.double_value(&[0, i as i64]);
            logits_vec.push(val as f32);
        }
        
        Ok(logits_vec)
    }


    /// Sample next token from logits
    fn sample_token(&self, logits: &[f32], temperature: f32, top_p: f32, top_k: Option<usize>, repeat_penalty: f32, previous_tokens: &[i64]) -> Result<usize> {
        if logits.is_empty() {
            return Err(anyhow!("Empty logits"));
        }

        // Apply repetition penalty first
        let mut penalized_logits = logits.to_vec();
        if repeat_penalty != 1.0 {
            for &prev_token in previous_tokens.iter().take(100) {  // Only consider last 100 tokens
                let token_id = prev_token as usize;
                if token_id < penalized_logits.len() {
                    if penalized_logits[token_id] > 0.0 {
                        penalized_logits[token_id] /= repeat_penalty;
                    } else {
                        penalized_logits[token_id] *= repeat_penalty;
                    }
                }
            }
        }

        // Apply temperature
        let scaled_logits: Vec<f32> = if temperature > 0.0 {
            penalized_logits.iter().map(|&x| x / temperature).collect()
        } else {
            penalized_logits
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
            indices.sort_by(|&a, &b| {
                probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            
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
            indices.sort_by(|&a, &b| {
                probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            
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
        // Format prompt with Qwen2 chat template
        let formatted_prompt = self.format_chat_message(
            Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."), 
            prompt
        );
        
        let request = GenerationRequest {
            prompt: formatted_prompt,
            max_tokens,
            temperature: 0.7,  // Official recommendation
            top_p: 0.8,        // Official recommendation  
            top_k: Some(20),   // Official recommendation
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
        // Ensure persistent model is ready
        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized - persistent model not ready"));
        }
        
        let start_time = std::time::Instant::now();

        // Tokenize input
        let mut input_ids = self.tokenize(&request.prompt)?;
        let mut generated_text = String::new();
        let mut tokens_generated = 0;
        let prompt_len = input_ids.len();

        for i in 0..request.max_tokens {
            // Use KV cached forward pass after first iteration
            let logits = if i == 0 {
                // First pass: process entire prompt
                self.forward(&input_ids)?
            } else {
                // Subsequent passes: only process new token with KV cache
                self.forward_cached(&input_ids, prompt_len + i - 1, true)?
            };
            
            // Sample next token with proper parameters
            let next_token = self.sample_token(
                &logits,
                request.temperature,
                request.top_p,
                request.top_k,
                request.repeat_penalty,
                &input_ids,
            )?;
            
            tracing::info!("Sampled token ID: {}", next_token);

            // Add to sequence
            input_ids.push(next_token as i64);
            tokens_generated += 1;

            // Decode token
            let token_text = self.detokenize(&[next_token as i64])?;
            tracing::info!("Token text: '{}'", token_text);
            generated_text.push_str(&token_text);

            // Check stop tokens
            if request.stop_tokens.iter().any(|stop| generated_text.contains(stop)) {
                break;
            }

            // Proper EOS check for Qwen2 (im_end or endoftext tokens)
            if self.is_eos_token(next_token) {
                tracing::info!("EOS token detected: {}", next_token);
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
        self.handle_poison(self.model_info.lock())
            .map(|guard| guard.clone())
            .unwrap_or_else(|_| ModelInfo {
                name: "error".to_string(),
                architecture: "unknown".to_string(),
                parameters: 0,
                context_length: 2048,
                vocab_size: 32000,
                quantization: None,
            })
    }

    fn is_loaded(&self) -> bool {
        let varstore_loaded = self.has_varstore();
        let persistent_loaded = self.persistent_model.is_some();
        
        varstore_loaded || persistent_loaded
    }
}

// Additional methods needed by inference layer
impl TorchEngine {
    /// Generate text with streaming callback
    pub async fn generate_streaming<F>(&self, prompt: &str, max_tokens: usize, mut callback: F) -> Result<String> 
    where 
        F: FnMut(&str)
    {
        // Ensure persistent model is ready
        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized - persistent model not ready"));
        }
        
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
        let prompt_len = input_ids.len();

        for i in 0..max_tokens {
            // Use KV cached forward pass after first iteration
            let logits = if i == 0 {
                // First pass: process entire prompt
                self.forward(&input_ids)?
            } else {
                // Subsequent passes: only process new token with KV cache
                self.forward_cached(&input_ids, prompt_len + i - 1, true)?
            };
            
            // Sample next token
            let next_token = self.sample_token(
                &logits,
                request.temperature,
                request.top_p,
                request.top_k,
                request.repeat_penalty,
                &input_ids,
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

            // Proper EOS check for Qwen2 (im_end or endoftext tokens)
            if self.is_eos_token(next_token) {
                tracing::info!("EOS token detected: {}", next_token);
                break;
            }
        }

        Ok(generated_text)
    }

    /// Update context state for tracking generation progress
    /// Note: This is currently a no-op since generation methods take &self
    fn update_context_state(&self, _sequence_length: usize) {
        // TODO: Implement with interior mutability if needed
        // Currently disabled to avoid borrowing conflicts
    }

    /// Check if persistent model is initialized - thread safe
    pub fn is_persistent_model_ready(&self) -> bool {
        let persistent_ready = self.persistent_model.is_some();
        let context_ready = self.handle_poison(self.context_state.lock())
            .map(|guard| guard.as_ref().map_or(false, |c| c.initialized))
            .unwrap_or(false);
        
        persistent_ready && context_ready
    }

    /// Generate text with raw prompt (no chat formatting)
    pub async fn generate_raw(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let request = GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.7,
            top_p: 0.8,
            top_k: Some(20),
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

    /// Generate text with custom chat formatting
    pub async fn generate_chat(&self, system: Option<&str>, user: &str, max_tokens: usize) -> Result<String> {
        let formatted_prompt = self.format_chat_message(system, user);
        self.generate_raw(&formatted_prompt, max_tokens).await
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

// TorchEngine is now thread-safe through proper use of Arc<Mutex<T>> for all mutable state
// No unsafe implementations needed - Send + Sync are automatically derived
// All critical sections are protected by mutexes with poisoning recovery