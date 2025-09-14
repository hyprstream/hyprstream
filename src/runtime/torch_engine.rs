//! PyTorch-based inference engine using tch-rs

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::path::Path;
use tch::{Device, Tensor, nn::VarStore};
use tokenizers::Tokenizer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use serde_json;
use json_threat_protection as jtp;
use crate::config::{GenerationRequest, GenerationResult, ModelInfo, RuntimeConfig, FinishReason};
use crate::runtime::RuntimeEngine;
use crate::runtime::template_engine::{TemplateEngine, TemplateConfig, ChatMessage};

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
#[derive(Clone)]
pub struct TorchEngine {
    /// VarStore for native PyTorch weight management - not thread safe, requires external sync
    var_store: Arc<Mutex<Option<VarStore>>>,
    /// SafeTensors raw data for on-demand tensor creation - thread safe after initialization
    /// Detected model architecture - thread safe after initialization
    model_architecture: Arc<Mutex<Option<String>>>,
    /// Persistent model instance to avoid recreation on every forward pass
    /// Using Arc<Mutex<>> for interior mutability since ModelOperations has mutable methods
    persistent_model: Option<Arc<Mutex<Box<dyn ModelOperations>>>>,
    /// Basic KV cache storage for context tracking - thread safe with mutex
    context_state: Arc<Mutex<Option<ContextState>>>,
    /// Tokenizer for text processing - thread safe after initialization
    tokenizer: Arc<Mutex<Option<Tokenizer>>>,
    /// Template engine for chat formatting
    template_engine: Arc<Mutex<Option<TemplateEngine>>>,
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
            model_architecture: Arc::new(Mutex::new(None)),
            persistent_model: None,
            context_state: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            template_engine: Arc::new(Mutex::new(None)),
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

        Ok(())
    }


    /// Load SafeTensors model using ModelFactory for unified weight loading
    async fn load_safetensors(&mut self, path: &Path) -> Result<()> {
        println!("🚀 Loading SafeTensors model: {}", path.display());
        
        // Get the model directory (parent of the safetensors file)
        let model_dir = path.parent().unwrap_or(path);
        
        // Use ModelFactory to load the model (handles all weight loading internally)
        self.initialize_persistent_model(model_dir)?;
        
        // Extract model info from the loaded model
        if let Some(model) = &self.persistent_model {
            let model_guard = self.handle_poison(model.lock())?;
            
            // Get architecture info from the model
            // For now, use a default since we don't have a method to query this
            let architecture = "auto".to_string();
            
            // Update model info
            {
                let mut model_info_guard = self.handle_poison(self.model_info.lock())?;
                model_info_guard.architecture = architecture.clone();
            }
            
            // Set architecture
            {
                let mut arch_guard = self.handle_poison(self.model_architecture.lock())?;
                *arch_guard = Some(architecture.clone());
            }
        }
        
        // Determine context window based on architecture
        // This should eventually come from the model config
        let context_window = 32768; // Default, will be overridden by model config
        
        // Initialize context state
        {
            let mut context_guard = self.handle_poison(self.context_state.lock())?;
            *context_guard = Some(ContextState {
                sequence_length: 0,
                context_window,
                initialized: true,
            });
        }
        
        // Create dummy VarStore for backward compatibility
        {
            let vs = VarStore::new(self.device);
            let mut var_store_guard = self.handle_poison(self.var_store.lock())?;
            *var_store_guard = Some(vs);
        }
        
        println!("✅ SafeTensors model loaded via ModelFactory");
        println!("🚀 Model initialized and ready for inference");
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

    /// Initialize the persistent model instance using ModelFactory
    fn initialize_persistent_model(&mut self, model_path: &Path) -> Result<()> {
        use crate::runtime::model_factory::ModelFactory;
        
        println!("🏗️ Initializing persistent model using ModelFactory");
        
        // Use the factory to create the model with proper configuration management
        let model = ModelFactory::create(model_path, &self.device, tch::Kind::BFloat16)?;
        
        self.persistent_model = Some(Arc::new(Mutex::new(model)));
        println!("✅ Persistent model initialized successfully");
        Ok(())
    }

    /// Load tokenizer and template configuration - thread safe
    async fn load_tokenizer(&mut self, model_path: &Path) -> Result<()> {
        // Try to find tokenizer.json - if model_path is a directory, look inside it
        // If it's a file, look in the parent directory
        let search_dir = if model_path.is_dir() {
            model_path
        } else {
            model_path.parent().unwrap_or(model_path)
        };
        
        let tokenizer_path = search_dir.join("tokenizer.json");
        let tokenizer_config_path = search_dir.join("tokenizer_config.json");

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
        
        // Load template configuration if available
        if tokenizer_config_path.exists() {
            println!("📋 Loading tokenizer config: {}", tokenizer_config_path.display());
            let config_content = tokio::fs::read_to_string(&tokenizer_config_path).await?;
            
            // Validate before parsing
            jtp::from_str(&config_content)
                .with_max_depth(10)
                .with_max_string_length(50000)
                .validate()
                .map_err(|e| anyhow!("Invalid tokenizer config: {:?}", e))?;
            
            let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
            
            // Parse template configuration
            let template_config = TemplateEngine::from_tokenizer_config(&config_json)?;
            
            // Create template engine
            let template_engine = TemplateEngine::new(template_config)?;
            
            // Store template engine
            let mut template_guard = self.handle_poison(self.template_engine.lock())?;
            *template_guard = Some(template_engine);
            
            println!("✅ Template engine initialized");
        } else {
            println!("⚠️ No tokenizer_config.json found, using fallback templates");
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

    /// Format text with dynamic chat template
    fn format_chat_message(&self, system: Option<&str>, user: &str) -> String {
        // Try to use template engine if available
        if let Ok(template_guard) = self.template_engine.lock() {
            if let Some(ref engine) = *template_guard {
                let mut messages = Vec::new();
                
                // Add system message if provided
                if let Some(system_msg) = system {
                    messages.push(ChatMessage {
                        role: "system".to_string(),
                        content: system_msg.to_string(),
                    });
                }
                
                // Add user message
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: user.to_string(),
                });
                
                // Apply template
                if let Ok(formatted) = engine.apply_chat_template(&messages, Some(true)) {
                    return formatted;
                }
            }
        }
        
        // Fallback to hardcoded Qwen2 template if template engine not available
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
        
        // Use the new forward_with_cache method that properly tracks position
        let logits = model.forward_with_cache(&input_tensor, start_pos)?;
        
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


    /// Sample next token from logits with proper ordering
    fn sample_token(&self, logits: &[f32], temperature: f32, top_p: f32, top_k: Option<usize>, repeat_penalty: f32, previous_tokens: &[i64]) -> Result<usize> {
        tracing::debug!("Sampling: logits_len={}, temp={}, top_p={}, top_k={:?}, repeat_penalty={}", 
                      logits.len(), temperature, top_p, top_k, repeat_penalty);
        if logits.is_empty() {
            return Err(anyhow!("Empty logits"));
        }

        // STEP 1: Apply repetition penalty to LOGITS (before softmax)
        // This is much more effective than applying to probabilities
        let mut adjusted_logits = logits.to_vec();
        if repeat_penalty != 1.0 && !previous_tokens.is_empty() {
            // Create a frequency map for recent tokens
            let mut token_counts = std::collections::HashMap::new();
            for &token_id in previous_tokens.iter().rev().take(64) {  // Last 64 tokens
                *token_counts.entry(token_id as usize).or_insert(0) += 1;
            }
            
            // Apply penalty based on frequency
            for (token_id, count) in token_counts {
                if token_id < adjusted_logits.len() {
                    // Stronger penalty for more frequent tokens
                    let penalty = repeat_penalty.powf(count as f32);
                    if adjusted_logits[token_id] > 0.0 {
                        adjusted_logits[token_id] /= penalty;
                    } else {
                        adjusted_logits[token_id] *= penalty;
                    }
                }
            }
        }

        // STEP 2: Apply temperature scaling  
        let scaled_logits: Vec<f32> = if temperature > 0.0 && temperature != 1.0 {
            adjusted_logits.iter().map(|&x| x / temperature).collect()
        } else {
            adjusted_logits
        };

        // Find max for numerical stability in softmax
        let max_logit = scaled_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute softmax probabilities
        let exp_logits: Vec<f32> = scaled_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        
        let sum: f32 = exp_logits.iter().sum();
        let mut probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

        // STEP 3: Apply top-k filtering (select top k most likely tokens)
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

        // STEP 4: Apply top-p (nucleus) sampling (cumulative probability threshold)
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

        // STEP 5: Sample from the final distribution
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
        // Store the original path for model naming
        let original_path = path.to_path_buf();
        
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
        
        // Set the model name based on the canonical path
        // Use directory name if loading from directory, otherwise use filename
        let model_name = if original_path.is_dir() {
            // Get the last component of the directory path as the model name
            original_path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            // Use the file stem for single files
            original_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        };
        
        // Update model info with the correct name
        {
            let mut model_info_guard = self.handle_poison(self.model_info.lock())?;
            model_info_guard.name = model_name;
        }
        
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
        // Add timeout protection (default 120 seconds, configurable via HYPRSTREAM_GENERATION_TIMEOUT env var)
        let timeout_secs = std::env::var("HYPRSTREAM_GENERATION_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(120);
        let timeout_duration = std::time::Duration::from_secs(timeout_secs);
        
        // Run the CPU-intensive generation in a blocking thread to avoid blocking the async runtime
        let self_clone = self.clone();
        let generation_future = tokio::task::spawn_blocking(move || {
            // Ensure persistent model is ready
            if !self_clone.is_persistent_model_ready() {
                return Err(anyhow!("Model not properly initialized - persistent model not ready"));
            }
            
            let start_time = std::time::Instant::now();

            // Tokenize input
            let mut input_ids = self_clone.tokenize(&request.prompt)?;
            let mut generated_text = String::new();
            let mut tokens_generated = 0;
            let prompt_len = input_ids.len();

            for i in 0..request.max_tokens {
                // Use KV cached forward pass after first iteration
                let logits = if i == 0 {
                    // First pass: process entire prompt
                    self_clone.forward(&input_ids)?
                } else {
                    // Subsequent passes: only process new token with KV cache
                    self_clone.forward_cached(&input_ids, prompt_len + i - 1, true)?
                };
                
                // Sample next token with proper parameters
                let next_token = self_clone.sample_token(
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
                let token_text = self_clone.detokenize(&[next_token as i64])?;
                tracing::info!("Token text: '{}'", token_text);
                generated_text.push_str(&token_text);

                // Check stop tokens
                if request.stop_tokens.iter().any(|stop| generated_text.contains(stop)) {
                    break;
                }

                // Proper EOS check for Qwen2 (im_end or endoftext tokens)
                if self_clone.is_eos_token(next_token) {
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
        });
        
        // Apply timeout to prevent runaway generation
        match tokio::time::timeout(timeout_duration, generation_future).await {
            Ok(Ok(Ok(result))) => Ok(result),  // Unwrap nested Results: timeout -> JoinHandle -> generation
            Ok(Ok(Err(e))) => Err(e),          // Generation error
            Ok(Err(e)) => Err(anyhow!("Blocking task panicked: {}", e)), // Task panic
            Err(_) => {
                Err(anyhow!("Generation timed out after {:?}", timeout_duration))
            }
        }
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
    
    fn apply_chat_template(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        // Use our template engine if available
        let template_guard = self.handle_poison(self.template_engine.lock())?;
        
        if let Some(ref engine) = *template_guard {
            // Use the template engine
            engine.apply_chat_template(messages, Some(add_generation_prompt))
        } else {
            // Fallback to simple formatting
            let mut formatted = String::new();
            for msg in messages {
                formatted.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
            if add_generation_prompt {
                formatted.push_str("assistant: ");
            }
            Ok(formatted)
        }
    }
}

// Additional methods needed by inference layer
impl TorchEngine {
    /// Generate text with streaming callback and custom parameters
    pub async fn generate_streaming_with_params<F>(
        &self, 
        prompt: &str, 
        max_tokens: usize, 
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
        mut callback: F
    ) -> Result<String>
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
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            stop_tokens: vec![],
            seed: None,
            stream: true,
            active_adapters: None,
            realtime_adaptation: None,
            user_feedback: None,
        };
        
        // Use the internal streaming implementation with these parameters
        self.generate_streaming_internal(request, callback).await
    }
    
    /// Generate text with streaming callback (using default parameters)
    pub async fn generate_streaming<F>(&self, prompt: &str, max_tokens: usize, mut callback: F) -> Result<String> 
    where 
        F: FnMut(&str)
    {
        // Use model-specific defaults or fallback to generic defaults
        self.generate_streaming_with_params(
            prompt,
            max_tokens,
            0.8,       // Default temperature
            0.95,      // Default top_p
            Some(50),  // Default top_k
            1.1,       // Default repeat_penalty
            callback
        ).await
    }
    
    /// Internal streaming implementation
    async fn generate_streaming_internal<F>(&self, request: GenerationRequest, mut callback: F) -> Result<String>
    where
        F: FnMut(&str)
    {
        let prompt = &request.prompt;
        let max_tokens = request.max_tokens;
        let temperature = request.temperature;
        let top_p = request.top_p;
        let top_k = request.top_k;
        let repeat_penalty = request.repeat_penalty;

        // REAL streaming: generate tokens one by one and call callback for each
        let start_time = std::time::Instant::now();
        
        // Tokenize input
        let mut input_ids = self.tokenize(prompt)?;
        let mut generated_text = String::new();
        let mut tokens_generated = 0;
        let prompt_len = input_ids.len();
        
        tracing::info!("Starting generation with prompt of {} tokens", prompt_len);
        tracing::debug!("Initial token IDs: {:?}", &input_ids[..prompt_len.min(20)]);

        for i in 0..max_tokens {
            // Use KV cached forward pass after first iteration
            let logits = if i == 0 {
                // First pass: process entire prompt
                self.forward(&input_ids)?
            } else {
                // Subsequent passes: only process new token with KV cache
                // Position is prompt_len + (i-1) since we're processing the token generated in the previous iteration
                self.forward_cached(&input_ids, prompt_len + i - 1, true)?
            };
            
            // Sample next token
            let next_token = self.sample_token(
                &logits,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &input_ids,
            )?;

            // Add to sequence
            input_ids.push(next_token as i64);
            tokens_generated += 1;

            // Decode token and stream it immediately
            let token_text = self.detokenize(&[next_token as i64])?;
            generated_text.push_str(&token_text);
            
            tracing::debug!("Iteration {}: generated token {} -> '{}'", 
                         i, next_token, token_text);
            
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
    fn update_context_state(&self, _sequence_length: usize) {
    }

    /// Check if persistent model is initialized - thread safe
    pub fn is_persistent_model_ready(&self) -> bool {
        let persistent_ready = self.persistent_model.is_some();
        let context_ready = self.handle_poison(self.context_state.lock())
            .map(|guard| guard.as_ref().map_or(false, |c| c.initialized))
            .unwrap_or(false);
        
        persistent_ready && context_ready
    }

    /// Generate text with async streaming callback
    pub async fn generate_streaming_async(
        &self,
        request: GenerationRequest,
        mut callback: Box<dyn crate::runtime::streaming::StreamingCallback>,
        context: crate::runtime::streaming::GenerationContext,
    ) -> Result<GenerationResult> {
        use crate::runtime::streaming::ContinueGeneration;
        use tokio::time::timeout;
        
        let start_time = std::time::Instant::now();
        
        // Notify start
        callback.on_start().await;
        
        // Tokenize input
        let mut input_ids = self.tokenize(&request.prompt)?;
        let mut generated_text = String::new();
        let mut tokens_generated = 0;
        let prompt_len = input_ids.len();
        
        tracing::info!("Starting async generation with prompt of {} tokens", prompt_len);
        
        // Create timeout future
        let generation_future = async {
            for i in 0..request.max_tokens {
                // Check cancellation
                if context.cancel_token.is_cancelled() {
                    tracing::info!("Generation cancelled by client");
                    break;
                }
                
                // Forward pass
                let logits = if i == 0 {
                    self.forward(&input_ids)?
                } else {
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
                
                // Decode token
                let token_text = self.detokenize(&[next_token as i64])?;
                generated_text.push_str(&token_text);
                
                // Stream token via callback
                match callback.on_token(&token_text).await {
                    Ok(ContinueGeneration::Continue) => {},
                    Ok(ContinueGeneration::Stop) => {
                        tracing::info!("Generation stopped by callback");
                        break;
                    },
                    Ok(ContinueGeneration::Pause(duration)) => {
                        tokio::time::sleep(duration).await;
                    },
                    Err(e) => {
                        callback.on_error(anyhow::anyhow!("{}", e)).await;
                        return Err(e);
                    }
                }
                
                // Check stop conditions
                if request.stop_tokens.iter().any(|stop| generated_text.contains(stop)) {
                    break;
                }
                
                if self.is_eos_token(next_token) {
                    tracing::info!("EOS token detected: {}", next_token);
                    break;
                }
            }
            
            Ok::<_, anyhow::Error>(())
        };
        
        // Apply timeout
        match timeout(context.timeout, generation_future).await {
            Ok(Ok(())) => {
                // Success
                let finish_reason = if tokens_generated >= request.max_tokens {
                    FinishReason::MaxTokens
                } else if self.is_eos_token(*input_ids.last().unwrap() as u32 as usize) {
                    FinishReason::Stop
                } else {
                    FinishReason::Stop
                };
                
                callback.on_complete(finish_reason.clone()).await;
                
                Ok(GenerationResult {
                    text: generated_text,
                    tokens_generated,
                    finish_reason,
                    generation_time_ms: start_time.elapsed().as_millis() as u64,
                    tokens_per_second: tokens_generated as f32 / start_time.elapsed().as_secs_f32(),
                })
            }
            Ok(Err(e)) => {
                // Generation error
                callback.on_error(anyhow::anyhow!("{}", e)).await;
                Err(e)
            }
            Err(_) => {
                // Timeout
                let err = anyhow::anyhow!("Generation timeout after {:?}", context.timeout);
                callback.on_error(anyhow::anyhow!("{}", err)).await;
                Err(err)
            }
        }
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
        Ok(())
    }
}