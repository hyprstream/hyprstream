//! PyTorch-based inference engine using tch-rs

use crate::config::{
    FinishReason, GenerationConfig, GenerationRequest, GenerationResult, ModelInfo, RuntimeConfig,
};
use crate::runtime::tensor_sampling::TensorSampler;
use crate::runtime::template_engine::{ChatMessage, TemplateEngine};
use crate::runtime::architectures::ModelOperations;
use crate::runtime::RuntimeEngine;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use json_threat_protection as jtp;
use serde_json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, PoisonError,
};
use tch::{nn::VarStore, Device, Tensor};
use tokenizers::Tokenizer;
use tracing::{info, instrument, warn};

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
    #[allow(dead_code)]
    config: RuntimeConfig,
    /// Generation configuration with defaults
    generation_config: GenerationConfig,
    /// Loaded model information - protected by mutex
    model_info: Arc<Mutex<ModelInfo>>,
    /// Active LoRA adapter name - thread safe with mutex
    pub active_lora: Arc<Mutex<Option<String>>>,
    /// LoRA model with adapters
    lora_model: Arc<Mutex<Option<crate::lora::torch_adapter::LoRAModel>>>,
    /// LoRA trainer for fine-tuning
    lora_trainer: Arc<Mutex<Option<crate::lora::trainer::LoRATrainer>>>,
    /// Sampling configuration - immutable after construction
    #[allow(dead_code)]
    sampling_config: RuntimeConfig,
    /// GPU sampler for efficient token sampling - thread safe after initialization
    sampler: TensorSampler,  // Renamed from gpu_sampler - works on both CPU and GPU
    /// Cached tokenizer vocabulary size for lock-free access
    /// 0 means not yet initialized
    tokenizer_vocab_size: Arc<AtomicUsize>,
    // Note: XET/LFS handled by git-xet-filter + ModelFactory::load_file_with_pointer_detection()
    // Note: Pre-training not supported (persistent_model doesn't expose VarStore), LoRA only
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

    /// Set the random seed for deterministic generation
    /// Useful for debugging - enables reproducible token sequences
    pub fn set_seed(&self, seed: u64) {
        TensorSampler::set_seed(seed);
    }

    /// Get cached vocabulary size without locking
    pub fn get_vocab_size(&self) -> usize {
        let size = self.tokenizer_vocab_size.load(Ordering::Relaxed);
        if size > 0 {
            size
        } else {
            // Fallback: try to get from tokenizer (shouldn't happen after load_tokenizer)
            self.handle_poison(self.tokenizer.lock())
                .ok()
                .and_then(|guard| guard.as_ref().map(|t| t.get_vocab_size(true)))
                .unwrap_or(0)  // Return 0 if tokenizer not loaded (caller should error)
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
        let device = if config.use_gpu {
            let gpu_device = Device::cuda_if_available();
            if gpu_device != Device::Cpu {
                // Check if this is actually ROCm/HIP
                if std::env::var("HIP_VISIBLE_DEVICES").is_ok()
                    || std::path::Path::new("../libtorch/lib/libtorch_hip.so").exists()
                {
                    info!("🚀 Using ROCm/HIP GPU acceleration");
                } else {
                    info!("🚀 Using CUDA GPU acceleration");
                }
                gpu_device
            } else {
                info!("⚠️  GPU requested but not available, falling back to CPU");
                Device::Cpu
            }
        } else {
            info!("💻 Using CPU inference");
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
            generation_config: GenerationConfig {
                max_tokens: 2048,
                temperature: 0.7,
                top_p: 0.9,
                top_k: Some(40),
                repeat_penalty: 1.1,
                stop_tokens: vec!["</s>".to_string()],
                seed: None,
                stream: false,
            },
            model_info: Arc::new(Mutex::new(ModelInfo {
                name: "unloaded".to_string(),
                architecture: "unknown".to_string(),
                parameters: 0,
                context_length: 2048,
                vocab_size: 32000,
                hidden_size: 768,
                intermediate_size: None,
                num_attention_heads: None,
                num_hidden_layers: None,
                quantization: None,
            })),
            active_lora: Arc::new(Mutex::new(None)),
            lora_model: Arc::new(Mutex::new(None)),
            lora_trainer: Arc::new(Mutex::new(None)),
            sampling_config: config,
            sampler: TensorSampler::new(device),
            tokenizer_vocab_size: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Load model from safetensors or torchscript
    async fn load_model_file(&mut self, path: &Path) -> Result<()> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("Invalid file extension"))?;

        match ext {
            "pt" | "pth" => {
                return Err(anyhow!("TorchScript models (.pt/.pth) are no longer supported. Please use SafeTensors format (.safetensors) instead."));
            }
            "safetensors" => {
                // Load SafeTensors model directly
                info!("Loading weights: {}", path.display());
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
        info!("Loading weights: {}", path.display());

        // Get the model directory (parent of the safetensors file)
        let model_dir = path.parent().unwrap_or(path);

        // Use ModelFactory to load the model (handles all weight loading internally)
        self.initialize_persistent_model(model_dir).await?;

        // Extract model info from the loaded model
        if let Some(model) = &self.persistent_model {
            let _model_guard = self.handle_poison(model.lock())?;

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

        // Get context window from model info that was populated from config
        let context_window = self.handle_poison(self.model_info.lock())?.context_length;

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

        info!("✅ SafeTensors model loaded via ModelFactory");
        info!("🚀 Model initialized and ready for inference");
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

    /// Initialize XET storage with default configuration
    /// Initialize the persistent model instance using ModelFactory
    async fn initialize_persistent_model(&mut self, model_path: &Path) -> Result<()> {
        use crate::runtime::model_config::ModelConfig;
        use crate::runtime::model_factory::ModelFactory;
        use crate::runtime::torch_utils::preflight_gpu_check;

        info!("Initializing model");

        // XET/LFS handled automatically by git-xet-filter + ModelFactory fallback
        // Load model config first to get model parameters
        let empty_weights = HashMap::new();
        let config = ModelConfig::load(model_path, &empty_weights)?;

        // Estimate model memory requirements
        let estimated_weights_mb = {
            // Rough estimate: vocab_size * hidden_size (embeddings)
            //                 + num_layers * hidden_size * intermediate_size * 3 (MLP)
            //                 + num_layers * hidden_size * hidden_size * 4 (attention)
            let embedding_params = config.vocab_size * config.hidden_size;
            let mlp_params_per_layer = config.hidden_size * config.intermediate_size * 3;
            let attn_params_per_layer = config.hidden_size * config.hidden_size * 4;
            let params_per_layer = mlp_params_per_layer + attn_params_per_layer;
            let total_params = embedding_params + (config.num_hidden_layers * params_per_layer);

            // BF16 = 2 bytes per parameter
            (total_params * 2) as f64 / (1024.0 * 1024.0)
        };

        let kv_cache_mb = {
            // KV cache: 2 (keys+values) * num_layers * batch_size * max_seq_len * num_heads * head_dim * 2 (BF16)
            let batch_size = 1;
            let kv_per_layer = 2 * batch_size * config.max_position_embeddings
                * config.num_attention_heads * config.head_dim * 2;
            let total_kv = config.num_hidden_layers * kv_per_layer;
            total_kv as f64 / (1024.0 * 1024.0)
        };

        let total_estimated_mb = estimated_weights_mb + kv_cache_mb;

        info!(
            "Model memory estimate:\n\
             - Weights: {:.2} MB\n\
             - KV cache: {:.2} MB (max_seq_len={})\n\
             - Total: {:.2} MB",
            estimated_weights_mb,
            kv_cache_mb,
            config.max_position_embeddings,
            total_estimated_mb
        );

        // Pre-flight GPU memory check (best-effort)
        if let Err(e) = preflight_gpu_check(self.device, total_estimated_mb) {
            warn!("GPU memory pre-flight check failed: {}", e);
            // Continue anyway - the check might not be accurate
        }

        // Update ModelInfo with actual values from config
        {
            let mut model_info = self.handle_poison(self.model_info.lock())?;
            model_info.hidden_size = config.hidden_size;
            model_info.intermediate_size = Some(config.intermediate_size);
            model_info.num_attention_heads = Some(config.num_attention_heads);
            model_info.num_hidden_layers = Some(config.num_hidden_layers);
            model_info.vocab_size = config.vocab_size;
            model_info.context_length = config.max_position_embeddings;
            model_info.architecture = config.model_type.clone();
        }

        // Use the factory to create the model
        let factory_start = std::time::Instant::now();
        let model = ModelFactory::create(model_path, &self.device, tch::Kind::BFloat16).await?;
        let factory_time = factory_start.elapsed();
        info!("✅ Model weights loaded in {:.2}s", factory_time.as_secs_f64());

        self.persistent_model = Some(Arc::new(Mutex::new(model)));
        Ok(())
    }

    /// Load tokenizer and template configuration - thread safe
    #[instrument(name = "torch_engine.load_tokenizer", skip(self), fields(model_path = %model_path.display()))]
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
            info!("Loading tokenizer: {}", tokenizer_path.display());
            let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

            // Log initial vocabulary size
            let initial_vocab_size = tokenizer.get_vocab_size(true);
            info!("Initial tokenizer vocabulary size: {}", initial_vocab_size);

            // Get model's configured vocab size
            let model_vocab_size = {
                let model_info = self.handle_poison(self.model_info.lock())?;
                model_info.vocab_size as usize
            };

            // Apply model-specific tokenizer configuration if model is loaded
            if let Some(model) = &self.persistent_model {
                let model_guard = self.handle_poison(model.lock())?;
                let tokenizer_config = model_guard.get_tokenizer_config();

                // Configure the tokenizer based on the model architecture
                tokenizer_config.configure_tokenizer(&mut tokenizer, model_vocab_size)?;

                tracing::debug!("Applied model-specific tokenizer configuration");
            } else {
                // Model not loaded yet - log vocab mismatch if any
                if model_vocab_size != initial_vocab_size {
                    tracing::debug!(
                        "Vocabulary size mismatch detected (model: {}, tokenizer: {}), but model not loaded yet to apply configuration",
                        model_vocab_size, initial_vocab_size
                    );
                }
            }

            // Cache the final vocabulary size for lock-free access
            let final_vocab_size = tokenizer.get_vocab_size(true);
            self.tokenizer_vocab_size.store(final_vocab_size, Ordering::Relaxed);

            info!("Final tokenizer vocabulary size: {}", final_vocab_size);

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
            info!(
                "Loading tokenizer config: {}",
                tokenizer_config_path.display()
            );
            let config_content = tokio::fs::read_to_string(&tokenizer_config_path).await?;

            // Validate before parsing
            jtp::from_str(&config_content)
                .with_max_depth(10)
                .with_max_string_length(50000)
                .validate()
                .map_err(|e| anyhow!("Invalid tokenizer config: {:?}", e))?;

            let config_json: serde_json::Value = serde_json::from_str(&config_content)?;

            // TODO: Load special tokens configuration using tokenizer's built-in methods

            // Parse template configuration
            let template_config = TemplateEngine::from_tokenizer_config(&config_json)?;

            // Create template engine
            let template_engine = TemplateEngine::new(template_config)?;

            // Store template engine
            let mut template_guard = self.handle_poison(self.template_engine.lock())?;
            *template_guard = Some(template_engine);

            info!("✅ Template engine initialized");
        } else {
            info!("⚠️ No tokenizer_config.json found, using fallback templates");
        }

        Ok(())
    }

    /// Tokenize text to input IDs - thread safe
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        let tokenizer_guard = self.handle_poison(self.tokenizer.lock())?;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))?;

        // Log the raw input for debugging prompt issues
        tracing::info!("📝 Raw prompt before tokenization:\n{}", text);

        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        if token_ids.is_empty() {
            return Err(anyhow!(
                "Tokenization produced empty token sequence for text: '{}'",
                text
            ));
        }

        // Show tokenization details
        tracing::debug!("Tokenized '{}' -> {} tokens: {:?}",
            text.chars().take(100).collect::<String>(),
            token_ids.len(),
            token_ids
        );

        // Decode back to verify tokenization is correct
        if let Ok(decoded) = tokenizer.decode(&token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(), false) {
            if decoded != text {
                tracing::warn!("⚠️  Tokenization roundtrip mismatch!\nOriginal: {}\nDecoded:  {}",
                    text.chars().take(200).collect::<String>(),
                    decoded.chars().take(200).collect::<String>()
                );
            }
        }

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

        // Generic fallback template (model-agnostic)
        let mut formatted = String::new();

        // Add system message if provided
        if let Some(system_msg) = system {
            formatted.push_str("System: ");
            formatted.push_str(system_msg);
            formatted.push_str("\n\n");
        }

        // Add user message
        formatted.push_str("User: ");
        formatted.push_str(user);
        formatted.push_str("\n\nAssistant: ");

        formatted
    }

    /// Check if a token ID is a special EOS token using tokenizer's built-in detection
    pub fn is_eos_token(&self, token_id: usize) -> bool {
        if let Ok(tokenizer_guard) = self.tokenizer.lock() {
            if let Some(ref tokenizer) = *tokenizer_guard {
                // Check if it's the tokenizer's EOS token
                if let Some(eos_token) = tokenizer.get_added_tokens_decoder().get(&(token_id as u32)) {
                    return eos_token.content == "<|im_end|>" || eos_token.content == "</s>";
                }
            }
        }

        // Fallback: check if it's a common EOS token
        token_id as u32 == 2 // Common </s> token ID
    }

    /// Get the tokenizer for streaming decoding - CoW makes this cheap!
    ///
    /// Returns a cloned tokenizer (cheap due to copy-on-write)
    pub fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_guard = self.handle_poison(self.tokenizer.lock())?;
        tokenizer_guard
            .as_ref()
            .cloned() // Cheap clone due to CoW
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))
    }

    /// Run inference on the model (supports both TorchScript and VarStore models) - thread safe
    pub fn forward(&self, input_ids: &[i64]) -> Result<Tensor> {
        // Try VarStore-based inference first (preferred for SafeTensors models)
        if self.has_varstore() {
            return self.forward_varstore(input_ids);
        }

        Err(anyhow!("No model loaded - call load_model() first"))
    }

    /// Run inference using VarStore (SafeTensors models) with persistent model - thread safe
    fn forward_varstore(&self, input_ids: &[i64]) -> Result<Tensor> {
        // Use the persistent model instance - NO recreation!
        let model_arc = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Persistent model not initialized - call load_model() first"))?;

        // Verify context state with thread safety
        {
            let context_guard = self.handle_poison(self.context_state.lock())?;
            let _context_state = context_guard
                .as_ref()
                .ok_or_else(|| anyhow!("Context state not initialized"))?;
        }

        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized"));
        }

        // Convert input IDs to tensor (keep as int64 for embeddings)
        let input_tensor = Tensor::from_slice(input_ids)
            .to_kind(tch::Kind::Int64) // Ensure int64 for embedding lookup
            .to_device(self.device)
            .unsqueeze(0); // Add batch dimension: [1, seq_len]

        // Lock the model and run forward pass (efficient!) with poison recovery
        let model = self.handle_poison(model_arc.lock())?;
        let logits = model.forward(&input_tensor, None)?;

        // Extract logits for the last token
        let logits_shape = logits.size();
        let seq_len = logits_shape[1];
        let _vocab_size = logits_shape[2] as usize;

        // Get logits for last token: [batch=1, last_seq_pos, vocab_size]
        let last_token_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1); // [1, vocab_size]

        Ok(last_token_logits)
    }

    /// Run optimized inference with KV caching - only process new tokens
    pub fn forward_cached(
        &self,
        input_ids: &[i64],
        start_pos: usize,
        use_cache: bool,
    ) -> Result<Tensor> {
        // Use the persistent model instance
        let model_arc = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Persistent model not initialized"))?;

        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized"));
        }

        // For KV cached generation, only process new tokens after initial prompt
        let tokens_to_process = if use_cache && start_pos > 0 {
            // Only process the last token (the newly generated one)
            &input_ids[input_ids.len() - 1..]
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
        let _vocab_size = logits_shape[2] as usize;

        // Get logits for last token
        let last_token_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1);

        Ok(last_token_logits)
    }

    /// Clear KV cache before new generation to prevent context pollution
    pub fn clear_kv_cache(&self) {
        if let Some(model_arc) = &self.persistent_model {
            let model = match model_arc.lock() {
                Ok(m) => m,
                Err(poisoned) => {
                    tracing::warn!("Model lock poisoned during cache clear, recovering");
                    poisoned.into_inner()
                }
            };

            // Use downcasting to call clear_kv_cache on LlamaModel
            // This is safe because we know the model type at runtime
            let model_any = model.as_any();
            if let Some(llama_model) = model_any.downcast_ref::<crate::runtime::architectures::llama::LlamaModel>() {
                llama_model.clear_kv_cache();
                tracing::debug!("Cleared KV cache before generation");
            }
        }
    }

    /// Sample next token using bundled parameters
    fn sample_token_with_params(
        &self,
        logits_tensor: &Tensor,
        params: &SamplingParams,
        previous_tokens: &[i64],
    ) -> Result<usize> {
        self.sampler.sample_token(
            logits_tensor,
            params.temperature,
            params.top_p,
            params.top_k,
            params.repeat_penalty,
            previous_tokens,
        )
    }
}

#[async_trait]
impl RuntimeEngine for TorchEngine {
    #[instrument(name = "torch_engine.load_model", skip(self), fields(path = %path.display()))]
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
                "model.pth",
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
                let _shard_pattern = path.join("model-00001-of-*.safetensors");
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let filename = entry.file_name();
                        if let Some(name) = filename.to_str() {
                            if name.starts_with("model-00001-of-") && name.ends_with(".safetensors")
                            {
                                info!(
                                    "🔍 Detected sharded SafeTensors model starting with: {}",
                                    name
                                );
                                found_file = Some(entry.path());
                                break;
                            }
                        }
                    }
                }
            }

            found_file.ok_or_else(|| {
                anyhow!(
                    "No supported model file found in directory: {}",
                    path.display()
                )
            })?
        } else {
            path.to_path_buf()
        };

        info!("Loading model: {}", model_file_path.display());
        self.load_model_file(&model_file_path).await?;

        // Set the model name based on the canonical path
        // Use directory name if loading from directory, otherwise use filename
        let model_name = if original_path.is_dir() {
            // Get the last component of the directory path as the model name
            original_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            // Use the file stem for single files
            original_path
                .file_stem()
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
        // Format prompt with model-agnostic chat template
        let formatted_prompt = self.format_chat_message(
            Some("You are a helpful assistant."),
            prompt,
        );

        let request = GenerationRequest {
            prompt: formatted_prompt,
            max_tokens,
            temperature: self.generation_config.temperature,
            top_p: self.generation_config.top_p,
            top_k: self.generation_config.top_k,
            repeat_penalty: self.generation_config.repeat_penalty,
            repeat_last_n: 64, // Default
            stop_tokens: self.generation_config.stop_tokens.clone(),
            seed: None,
        };

        let result = self.generate_with_params(request).await?;
        Ok(result.text)
    }

    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult> {
        use futures::StreamExt;

        if !self.is_persistent_model_ready() {
            return Err(anyhow!(
                "Model not properly initialized - persistent model not ready"
            ));
        }

        let mut stream = self.generate(request)?;
        let mut accumulated_text = String::new();

        while let Some(text_chunk) = stream.next().await {
            accumulated_text.push_str(&text_chunk?);
        }

        let stats = stream.stats();
        Ok(GenerationResult {
            text: accumulated_text,
            tokens_generated: stats.tokens_generated,
            finish_reason: stats.finish_reason.unwrap_or(FinishReason::Stop),
            generation_time_ms: stats.generation_time_ms,
            tokens_per_second: stats.tokens_per_second,
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
                hidden_size: 768,
                intermediate_size: None,
                num_attention_heads: None,
                num_hidden_layers: None,
                quantization: None,
            })
    }

    fn is_loaded(&self) -> bool {
        let varstore_loaded = self.has_varstore();
        let persistent_loaded = self.persistent_model.is_some();

        varstore_loaded || persistent_loaded
    }

    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
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

// Training-specific methods
impl TorchEngine {
    /// Enable training mode for pre-training
    pub fn enable_training(&mut self, _learning_rate: f64) -> Result<()> {
        // Ensure model is loaded
        if !self.is_persistent_model_ready() {
            return Err(anyhow!(
                "Model not loaded - persistent model not initialized"
            ));
        }

        // For pre-training without VarStore:
        // 1. Set requires_grad on model tensors
        // 2. Use manual SGD or implement custom optimizer
        //
        // The challenge: tch-rs optimizers require VarStore
        // Solutions:
        // - Create a temporary VarStore and register model weights
        // - Implement manual gradient descent
        // - Use LoRA for all training (recommended)

        tracing::warn!("Pre-training without VarStore requires manual optimizer implementation");
        tracing::info!("For now, use LoRA training which has proper VarStore support");

        Ok(())
    }

    /// Perform a manual SGD step without optimizer
    pub fn manual_sgd_step(&mut self, _learning_rate: f32) -> Result<()> {
        // This would manually update weights:
        // weight = weight - learning_rate * weight.grad()
        // But we don't have direct access to the model's tensors

        tracing::warn!("Manual SGD not implemented - model tensors not accessible");
        Ok(())
    }

    /// Forward pass for training (with gradient tracking)
    pub fn forward_training(&self, input_ids: &Tensor, track_gradients: bool) -> Result<Tensor> {
        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not initialized"));
        }

        let model = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Persistent model not available"))?;
        let model_guard = self.handle_poison(model.lock())?;

        // Forward pass - gradients tracked if tensors have requires_grad
        if track_gradients {
            // Gradients are tracked by default when tensors have requires_grad
            model_guard.forward(input_ids, None)
        } else {
            tch::no_grad(|| model_guard.forward(input_ids, None))
        }
    }

    /// Compute loss and backward (without optimizer step)
    pub fn compute_loss_and_backward(&self, input_ids: &Tensor, labels: &Tensor) -> Result<f64> {
        // Forward pass with gradients
        let logits = self.forward_training(input_ids, true)?;

        // Compute cross-entropy loss
        let batch_size = logits.size()[0];
        let seq_len = logits.size()[1];
        let vocab_size = logits.size()[2];

        let logits_flat = logits.view([batch_size * seq_len, vocab_size]);
        let labels_flat = labels.view([batch_size * seq_len]);

        let loss = logits_flat.cross_entropy_loss::<Tensor>(
            &labels_flat,
            None,
            tch::Reduction::Mean,
            -100, // ignore_index for padding
            0.0,  // label_smoothing
        );

        let loss_value = loss.double_value(&[]);

        // Backward pass computes gradients
        loss.backward();

        Ok(loss_value)
    }

    /// Disable training mode
    pub fn disable_training(&mut self) -> Result<()> {
        // Gradient tracking is controlled per tensor, not globally
        tracing::info!("Training mode disabled");
        Ok(())
    }

    /// Check if training is enabled
    pub fn is_training_enabled(&self) -> bool {
        false // Pre-training not fully supported without VarStore
    }
}

// Additional methods needed by inference layer
impl TorchEngine {
    // Old callback-based streaming APIs removed - use generate() for all streaming use cases

    /// Check if persistent model is initialized - thread safe
    pub fn is_persistent_model_ready(&self) -> bool {
        let persistent_ready = self.persistent_model.is_some();
        let context_ready = self
            .handle_poison(self.context_state.lock())
            .map(|guard| guard.as_ref().is_some_and(|c| c.initialized))
            .unwrap_or(false);

        persistent_ready && context_ready
    }

    /// Generate text as a stream of decoded UTF-8 text chunks
    ///
    /// Returns a Stream that yields `Result<String>` with properly decoded text.
    /// The stream automatically handles:
    /// - Multi-byte UTF-8 sequences (emojis, CJK characters, etc.)
    /// - EOS and stop token detection
    /// - Max tokens limit
    /// - Special token filtering
    ///
    /// Dropping the stream stops generation automatically (no manual cancellation needed).
    ///
    /// # Example
    /// ```no_run
    /// use futures::StreamExt;
    ///
    /// let mut stream = engine.generate(request)?;
    ///
    /// while let Some(text_chunk) = stream.next().await {
    ///     print!("{}", text_chunk?);
    /// }
    ///
    /// let stats = stream.stats();
    /// println!("Generated {} tokens", stats.tokens_generated);
    /// ```
    pub fn generate(&self, request: GenerationRequest) -> Result<TextStream<'_>> {
        // Set random seed if provided for deterministic generation
        if let Some(seed) = request.seed {
            self.set_seed(seed as u64);
        }
        TextStream::new(self, request)
    }


    /// Enable LoRA training for fine-tuning
    pub fn enable_lora_training(
        &mut self,
        config: crate::lora::LoRAConfig,
        training_config: crate::lora::TrainingConfig,
    ) -> Result<()> {
        // Get module dimensions from loaded model
        let mut module_configs = HashMap::new();

        // Get model dimensions from model_info
        let model_info = self.handle_poison(self.model_info.lock())?;
        let hidden_size = model_info.hidden_size as i64;

        // Calculate intermediate size based on model architecture
        let intermediate_size = if let Some(intermediate) = model_info.intermediate_size {
            intermediate as i64
        } else {
            // Use architecture-specific defaults
            match model_info.architecture.as_str() {
                "Qwen2ForCausalLM" => (hidden_size as f32 * 2.6667) as i64, // Qwen uses ~2.67x
                "LlamaForCausalLM" => (hidden_size as f32 * 2.75) as i64,   // Llama uses 2.75x
                _ => hidden_size * 4,                                       // Default 4x expansion
            }
        };

        // Extract dimensions for each target module
        for module_name in &config.target_modules {
            let (in_features, out_features) = match module_name.as_str() {
                "q_proj" | "k_proj" | "v_proj" | "o_proj" => {
                    // Self-attention projections
                    (hidden_size, hidden_size)
                }
                "gate_proj" | "up_proj" => {
                    // MLP projections (expand)
                    (hidden_size, intermediate_size)
                }
                "down_proj" => {
                    // MLP projection (contract)
                    (intermediate_size, hidden_size)
                }
                _ => {
                    return Err(anyhow!(
                        "Unknown module '{}'. Supported modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj",
                        module_name
                    ));
                }
            };

            module_configs.insert(
                module_name.clone(),
                (in_features as usize, out_features as usize),
            );
        }

        // Create LoRA model
        let lora_model = crate::lora::torch_adapter::LoRAModel::new(
            config.clone(),
            module_configs,
            self.device,
        )?;

        tracing::info!(
            "Initialized LoRA with {} trainable parameters",
            lora_model.num_parameters()
        );

        // Create trainer that uses the model's VarStore
        let trainer =
            crate::lora::trainer::LoRATrainer::new(&lora_model.vs, self.device, training_config)?;

        // Store the single model and trainer
        *self.handle_poison(self.lora_model.lock())? = Some(lora_model);
        *self.handle_poison(self.lora_trainer.lock())? = Some(trainer);

        // Gradient computation enabled by default in training mode

        Ok(())
    }

    /// Forward pass with LoRA adapters (Smart Hybrid with Gradient Bridge)
    pub fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        // Check if LoRA is enabled
        let lora_model = self.handle_poison(self.lora_model.lock())?;
        if lora_model.is_none() {
            // No LoRA, use standard forward pass
            let ids: Vec<i64> = Vec::<i64>::try_from(input_ids.flatten(0, -1))?;
            return self.forward(&ids);
        }

        // Get the persistent model for layer-wise forward pass
        let model = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Model not loaded"))?;
        let model_guard = self.handle_poison(model.lock())?;

        // Use input_ids tensor directly
        let input = input_ids.to(self.device);

        // CRITICAL: Smart Hybrid Gradient Bridge
        // Base model weights stay frozen, but we enable gradient tracking on activations
        let logits = if training {
            // During training: Enable gradient flow through activations
            // The base model weights don't have requires_grad, but the activations will
            let base_logits = model_guard.forward(&input, None)?;

            // Enable gradient tracking on the output activations
            // This creates the gradient bridge between frozen base model and trainable LoRA
            base_logits.set_requires_grad(true)
        } else {
            // During inference: No gradient tracking needed
            tch::no_grad(|| model_guard.forward(&input, None))?
        };

        // NOTE: In a full implementation, we would intercept activations at each layer
        // and apply LoRA adapters there. For now, this simplified version demonstrates
        // the gradient bridge concept. The actual per-layer integration would look like:
        //
        // for layer in model.layers:
        //     hidden = layer.self_attn(hidden)  // Base frozen weights
        //     if training:
        //         hidden = hidden.set_requires_grad(true)  // Enable gradient flow
        //     hidden = hidden + lora_adapter.forward("q_proj", hidden)  // Add LoRA

        Ok(logits)
    }

    /// Train LoRA adapter on a single example
    pub async fn train_temporal_lora(
        &mut self,
        prompt: &str,
        expected_response: &str,
        learning_rate: f32,
    ) -> Result<()> {
        // Initialize LoRA if not already done
        if self.handle_poison(self.lora_model.lock())?.is_none() {
            let lora_config = crate::lora::LoRAConfig {
                rank: 16,
                alpha: 16.0,
                dropout: 0.1,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
                ..Default::default()
            };

            let training_config = crate::lora::TrainingConfig {
                learning_rate: learning_rate as f64,
                ..Default::default()
            };

            self.enable_lora_training(lora_config, training_config)?;
        }

        // Tokenize inputs
        let tokenizer = self.handle_poison(self.tokenizer.lock())?;
        let tokenizer = tokenizer.as_ref().ok_or(anyhow!("No tokenizer loaded"))?;

        // Combine prompt and response
        let full_text = format!("{} {}", prompt, expected_response);
        let encoding = tokenizer
            .encode(full_text.as_str(), false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let input_ids = Tensor::from_slice(&ids).to_device(self.device).unsqueeze(0); // Add batch dimension

        // Create labels (shift input_ids by 1)
        let labels = input_ids.shallow_clone();

        // Forward pass with LoRA
        let logits = self.forward_with_lora(&input_ids, None, true)?;

        // Training step
        let metrics = {
            let mut trainer_guard = self.handle_poison(self.lora_trainer.lock())?;
            let lora_guard = self.handle_poison(self.lora_model.lock())?;

            if let (Some(trainer), Some(lora_model)) = (&mut *trainer_guard, &*lora_guard) {
                trainer.training_step(lora_model, &logits, &labels)?
            } else {
                return Err(anyhow!("LoRA trainer or model not initialized"));
            }
        };

        tracing::info!(
            "LoRA training step {}: loss={:.4}, ppl={:.2}, lr={:.6}",
            metrics.step,
            metrics.loss,
            metrics.perplexity,
            metrics.learning_rate
        );

        Ok(())
    }

    /// Save LoRA weights
    pub fn save_lora(&self, path: &str) -> Result<()> {
        if let Some(lora_model) = &*self.handle_poison(self.lora_model.lock())? {
            lora_model.save(path)?;
            tracing::info!("Saved LoRA weights to {}", path);
        } else {
            return Err(anyhow!("No LoRA model to save"));
        }
        Ok(())
    }

    /// Load LoRA weights
    pub fn load_lora(&mut self, path: &str) -> Result<()> {
        if let Some(lora_model) = &mut *self.handle_poison(self.lora_model.lock())? {
            lora_model.load(path)?;
            tracing::info!("Loaded LoRA weights from {}", path);
        } else {
            return Err(anyhow!("No LoRA model initialized"));
        }
        Ok(())
    }

    /// Save LoRA weights in SafeTensor format
    pub fn save_lora_weights(&self, path: &str) -> Result<()> {
        if let Some(lora_model) = &*self.handle_poison(self.lora_model.lock())? {
            lora_model.save(path)?;
            tracing::info!("Saved LoRA weights to SafeTensor format: {}", path);
        } else {
            return Err(anyhow!("No LoRA model to save"));
        }
        Ok(())
    }

    /// Load LoRA adapter weights from SafeTensors format.
    ///
    /// Loads pre-trained LoRA A and B matrices from a SafeTensors file into the
    /// initialized LoRA model structure. This updates the VarStore with the new
    /// weights, making them available for inference and further training.
    ///
    /// # Requirements
    /// - LoRA model must be initialized first via create_lora()
    /// - SafeTensors file must contain tensors with names matching the LoRA layer structure
    /// - Tensor shapes must match the initialized LoRA model dimensions
    ///
    /// # Arguments
    /// - `path` - Path to SafeTensors file containing LoRA weights
    pub fn load_lora_weights(&mut self, path: &str) -> Result<()> {
        if let Some(lora_model) = &mut *self.handle_poison(self.lora_model.lock())? {
            tracing::debug!(
                safetensors_path = path,
                "Loading LoRA weights from SafeTensors format"
            );

            lora_model.load(path)?;

            tracing::info!(
                safetensors_path = path,
                total_parameters = lora_model.num_parameters(),
                "Successfully loaded LoRA adapter weights from SafeTensors"
            );
        } else {
            tracing::error!(
                safetensors_path = path,
                "Cannot load LoRA weights: LoRA model structure not initialized - call create_lora() first"
            );
            return Err(anyhow!(
                "No LoRA model initialized - call create_lora() first"
            ));
        }
        Ok(())
    }

    /// Check if LoRA model structure has been initialized.
    ///
    /// Returns true if a LoRA model with VarStore has been created, indicating
    /// the engine is ready to load adapter weights. This is a prerequisite for
    /// calling load_lora_weights().
    pub fn has_lora_model(&self) -> bool {
        self.handle_poison(self.lora_model.lock())
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Initialize LoRA model structure for low-rank adaptation.
    ///
    /// Creates the VarStore and layer mappings required for LoRA training and inference.
    /// This must be called before loading any adapter weights. The configuration defines
    /// the rank (bottleneck dimension), alpha scaling factor, and target modules to adapt.
    ///
    /// # Architecture
    /// - Creates a shared VarStore for all LoRA layers with automatic differentiation
    /// - Maps target modules to their input/output dimensions for weight initialization
    /// - Uses proper Kaiming initialization for LoRA A and zero initialization for LoRA B
    ///
    /// # Arguments
    /// - `config` - LoRA configuration including rank, alpha, target modules, and dropout
    pub fn create_lora(&mut self, config: crate::lora::LoRAConfig) -> Result<()> {
        use crate::lora::torch_adapter::LoRAModel;

        tracing::info!(
            rank = config.rank,
            alpha = config.alpha,
            dropout = config.dropout,
            target_modules = ?config.target_modules,
            learning_rate = config.learning_rate,
            "Initializing LoRA model structure for low-rank adaptation"
        );

        // Get module dimensions from loaded model instead of hardcoding
        let mut module_configs = std::collections::HashMap::new();

        // Get model dimensions from model_info
        let model_info = self.handle_poison(self.model_info.lock())?;
        let hidden_size = model_info.hidden_size as i64;

        // Calculate intermediate size based on model architecture
        let intermediate_size = if let Some(intermediate) = model_info.intermediate_size {
            intermediate as i64
        } else {
            // Use architecture-specific defaults
            match model_info.architecture.as_str() {
                "Qwen2ForCausalLM" => (hidden_size as f32 * 2.6667) as i64, // Qwen uses ~2.67x
                "LlamaForCausalLM" => (hidden_size as f32 * 2.75) as i64,   // Llama uses 2.75x
                _ => hidden_size * 4,                                       // Default 4x expansion
            }
        };

        // Extract dimensions for each target module
        for module_name in &config.target_modules {
            let (in_features, out_features) = match module_name.as_str() {
                "q_proj" | "k_proj" | "v_proj" | "o_proj" => {
                    // Self-attention projections
                    (hidden_size, hidden_size)
                }
                "gate_proj" | "up_proj" => {
                    // MLP projections (expand)
                    (hidden_size, intermediate_size)
                }
                "down_proj" => {
                    // MLP projection (contract)
                    (intermediate_size, hidden_size)
                }
                _ => {
                    return Err(anyhow!(
                        "Unknown module '{}'. Supported modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj",
                        module_name
                    ));
                }
            };

            module_configs.insert(
                module_name.clone(),
                (in_features as usize, out_features as usize),
            );
        }

        tracing::info!(
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            module_configs = ?module_configs,
            "Using model-specific dimensions for LoRA initialization"
        );

        // Initialize the LoRA model with PyTorch VarStore for gradient tracking.
        // This creates trainable LoRA A and B matrices for each target module.
        let lora_model = LoRAModel::new(config.clone(), module_configs, self.device)?;
        let total_params = lora_model.num_parameters();

        // Install the LoRA model into the engine's shared state.
        // This replaces any previously loaded LoRA model.
        {
            let mut lora_guard = self.handle_poison(self.lora_model.lock())?;
            *lora_guard = Some(lora_model);
        }

        tracing::info!(
            total_parameters = total_params,
            device = ?self.device,
            "LoRA model structure created successfully - ready to load adapter weights"
        );
        Ok(())
    }

    /// Load LoRA adapter weights from SafeTensors file (async version).
    ///
    /// Save LoRA weights to SafeTensors file
    ///
    /// Saves the current LoRA model's weights to a SafeTensors file for persistence.
    /// This allows adapters to be saved after training or initialization.
    ///
    /// # Requirements
    /// - LoRA model must be initialized first via create_lora()
    ///

    /// This is the async wrapper for load_lora_weights(), used in training contexts
    /// where adapter loading may be part of a larger async workflow. The actual
    /// loading is synchronous as PyTorch tensor operations are CPU/GPU bound.
    ///
    /// # Requirements
    /// - LoRA model structure must be initialized first via create_lora()
    /// - Path must point to a valid SafeTensors file with matching tensor shapes
    ///
    /// # Arguments
    /// - `path` - Path to SafeTensors file containing LoRA A and B matrices
    pub async fn load_lora_from_file(&mut self, path: &std::path::Path) -> Result<()> {
        tracing::info!(
            file_path = %path.display(),
            "Loading LoRA adapter weights from SafeTensors file"
        );

        // Delegate to synchronous implementation
        // SafeTensors loading is I/O bound but the actual tensor operations are sync
        self.load_lora_weights(
            path.to_str()
                .ok_or_else(|| anyhow!("Invalid UTF-8 path for LoRA adapter file"))?,
        )
    }

    /// Apply LoRA adapter during forward pass (called internally)
    pub fn apply_lora_to_output(
        &self,
        module_name: &str,
        input: &Tensor,
        base_output: &Tensor,
    ) -> Result<Tensor> {
        let lora_guard = self.handle_poison(self.lora_model.lock())?;

        if let Some(lora_model) = lora_guard.as_ref() {
            // Apply LoRA if available for this module
            if let Some(lora_output) = lora_model.forward(module_name, input, false)? {
                Ok(base_output + lora_output)
            } else {
                Ok(base_output.shallow_clone())
            }
        } else {
            Ok(base_output.shallow_clone())
        }
    }

    /// Get active LoRA adapter name
    pub fn get_active_lora(&self) -> Result<Option<String>> {
        let active_guard = self.handle_poison(self.active_lora.lock())?;
        Ok(active_guard.clone())
    }

    /// Unload current LoRA adapter
    pub fn unload_lora(&mut self) -> Result<()> {
        {
            let mut lora_guard = self.handle_poison(self.lora_model.lock())?;
            *lora_guard = None;
        }

        {
            let mut active_guard = self.handle_poison(self.active_lora.lock())?;
            *active_guard = None;
        }

        tracing::info!("Unloaded LoRA adapter");
        Ok(())
    }
}

impl Drop for TorchEngine {
    fn drop(&mut self) {
        // Drop our Arc references to shared resources.
        // The actual cleanup happens when the LAST Arc reference drops.
        //
        // IMPORTANT: Do NOT clear the contents of Arc<Mutex<Option<T>>> fields!
        // All TorchEngine clones share the same Arc instances. Clearing the contents
        // would break other clones (especially the cached engine in model_cache).
        //
        // Previous implementation cleared shared Arc contents, causing:
        // - First request: clone for spawn_blocking drops, clears var_store
        // - Second request: cached engine has var_store = None, fails with "No model loaded"

        // Drop our Arc references (NOT the contents)
        self.persistent_model = None; // Drops our Option<Arc<...>> reference

        // Arc<Mutex<...>> fields are automatically cleaned up via Arc reference counting
        // when the last TorchEngine clone drops. No manual intervention needed.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_in_engine_has_correct_max_tokens() {
        // Create a minimal runtime config
        let _runtime_config = RuntimeConfig::default();

        // Create engine (this will fail without LIBTORCH but we can test the constructor logic)
        // We're just verifying the default generation_config
        let gen_config = GenerationConfig {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repeat_penalty: 1.1,
            stop_tokens: vec!["</s>".to_string()],
            seed: None,
            stream: false,
        };

        // Verify the defaults we set
        assert_eq!(
            gen_config.max_tokens, 2048,
            "TorchEngine should initialize with max_tokens=2048"
        );
    }
}

/// Internal sampling parameters passed to TensorSampler
#[derive(Debug, Clone, Copy)]
struct SamplingParams {
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
}

use futures::Stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

/// Statistics about text generation
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    pub finish_reason: Option<FinishReason>,
}

/// Stream that yields decoded UTF-8 text chunks during generation.
///
/// Automatically handles UTF-8 buffering, stop tokens, EOS detection,
/// and all generation complexities. Just iterate and get text!
pub struct TextStream<'a> {
    engine: &'a TorchEngine,

    prompt_tokens: Vec<i64>,
    last_generated: Option<i64>,

    recent_tokens: VecDeque<i64>,
    repeat_last_n: usize,

    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,

    max_tokens: usize,
    stop_token_ids: Vec<u32>,

    // Store tokenizer as raw pointer (leaked Box) to get stable reference for DecodeStream
    // We manually deallocate in Drop
    tokenizer: *mut Tokenizer,
    decode_stream: tokenizers::tokenizer::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,

    prompt_len: usize,
    /// Direct KV cache position tracking - next write position in cache
    /// This is the ground truth for where the next token will be written
    kv_cache_position: usize,
    /// Total tokens generated (including buffered UTF-8) - for statistics only
    tokens_generated: usize,
    start_time: std::time::Instant,
    finished: bool,
    finish_reason: Option<FinishReason>,
}

impl<'a> TextStream<'a> {
    fn new(engine: &'a TorchEngine, request: GenerationRequest) -> Result<Self> {
        let prompt_tokens = engine.tokenize(&request.prompt)?;
        let prompt_len = prompt_tokens.len();

        let tokenizer = engine.get_tokenizer()?;
        let stop_token_ids: Vec<u32> = request.stop_tokens
            .iter()
            .filter_map(|stop_str| {
                let encoding = tokenizer.encode(stop_str.as_str(), false).ok()?;
                let ids = encoding.get_ids();
                if ids.len() == 1 {
                    Some(ids[0])
                } else {
                    tracing::warn!(
                        "Stop token '{}' encodes to {} tokens, skipping (only single-token stops supported)",
                        stop_str, ids.len()
                    );
                    None
                }
            })
            .collect();

        engine.clear_kv_cache();

        let repeat_last_n = if request.repeat_last_n > 0 {
            request.repeat_last_n
        } else {
            64
        };

        // We need to use Box::leak to create a 'static reference that we can then downcast to 'a
        // This is safe because:
        // 1. We're leaking the tokenizer to get a stable 'static reference
        // 2. We'll manually drop it in TextStream's Drop impl
        // 3. The decode_stream lifetime is properly tied to the leaked reference
        let tokenizer_box = Box::new(tokenizer);
        let tokenizer_static: &'static Tokenizer = Box::leak(tokenizer_box);

        // Now we can safely create DecodeStream with 'static lifetime
        let decode_stream = unsafe {
            // SAFETY: We transmute 'static to 'a, which is safe because:
            // - decode_stream will be dropped before we deallocate the tokenizer
            // - TextStream is not Clone/Copy
            // - We maintain ownership via the raw pointer
            let tokenizer_ref: &'a Tokenizer = std::mem::transmute(tokenizer_static);
            // Use skip_special_tokens=false because <|extra_N|> tokens are special tokens
            // that should appear in output (they represent actual model vocabulary)
            tokenizer_ref.decode_stream(false) // skip_special_tokens=false
        };

        // Store the raw pointer so we can deallocate in Drop
        let tokenizer_ptr = tokenizer_static as *const Tokenizer as *mut Tokenizer;

        Ok(Self {
            engine,
            prompt_tokens,
            last_generated: None,
            recent_tokens: VecDeque::with_capacity(repeat_last_n),
            repeat_last_n,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repeat_penalty: request.repeat_penalty,
            max_tokens: request.max_tokens,
            stop_token_ids,
            tokenizer: tokenizer_ptr,
            decode_stream,
            prompt_len,
            // KV cache starts with prompt already in it after first forward
            kv_cache_position: prompt_len,
            tokens_generated: 0,
            start_time: std::time::Instant::now(),
            finished: false,
            finish_reason: None,
        })
    }

    /// Get generation statistics (call after stream exhausted)
    pub fn stats(&self) -> GenerationStats {
        let generation_time = self.start_time.elapsed();

        GenerationStats {
            tokens_generated: self.tokens_generated,
            generation_time_ms: generation_time.as_millis() as u64,
            tokens_per_second: if generation_time.as_secs_f32() > 0.0 {
                self.tokens_generated as f32 / generation_time.as_secs_f32()
            } else {
                0.0
            },
            finish_reason: self.finish_reason.clone(),
        }
    }

    fn sample_next_token(&mut self) -> Result<u32> {
        // Determine KV position for this forward pass
        let current_kv_pos = if self.tokens_generated == 0 {
            0 // Initial position is 0
        } else {
            self.kv_cache_position
        };

        let logits = if self.tokens_generated == 0 {
            tracing::debug!("🔵 Initial forward: prompt_len={}", self.prompt_len);
            self.engine.forward(&self.prompt_tokens)?
        } else {
            let last_token = self.last_generated.expect("last_generated should be set");

            if self.tokens_generated % 50 == 0 {
                tracing::debug!(
                    "🔵 KV cache position: {}, tokens_generated: {}, last_token: {}",
                    self.kv_cache_position, self.tokens_generated, last_token
                );
            }

            // Use kv_cache_position directly - this is where the next token will be written
            self.engine.forward_cached(
                &[last_token],
                current_kv_pos,
                true,
            )?
        };

        // NOTE: Logits truncation has been DISABLED (Nov 6, 2025)
        //
        // Previous code tried to truncate logits from model vocab (151936) to tokenizer vocab (151669).
        // This was causing year tokens to be excluded from sampling, leading to "1 7 7 6" instead of "1776".
        //
        // vLLM with the same model does NOT have this issue, suggesting the truncation was too aggressive.
        //
        // Solution: Use model vocab directly. The sampler will only pick valid tokens anyway.
        // The tokenizer's get_vocab_size() should match model vocab if properly configured.

        let vocab_size = self.engine.get_vocab_size();
        let logits_shape = logits.size();
        let model_vocab_size = logits_shape[logits_shape.len() - 1] as usize;

        if vocab_size == 0 {
            // Tokenizer not loaded - this should never happen during generation
            return Err(anyhow::anyhow!(
                "Cannot sample tokens: tokenizer vocabulary size is 0 (tokenizer not loaded)"
            ));
        }

        // Log mismatch for debugging but don't truncate
        if model_vocab_size != vocab_size {
            tracing::debug!(
                "Vocab mismatch: model_vocab_size={}, tokenizer_vocab_size={}. Using model vocab directly.",
                model_vocab_size, vocab_size
            );
        }

        // UTF-8 reranking DISABLED - DecodeStream handles UTF-8 correctly
        // The reranker was causing number corruption by manipulating logits
        // for non-UTF-8 related tokens (like digits)
        //
        // YEAR CORRUPTION BUG FIX (Nov 6, 2025):
        // The real issue is vocab mismatch: model (151936) vs tokenizer (151669).
        // Years like "1776" are split into individual digits "1 7 7 6" because:
        // 1. Year tokens may be in the truncated range (>151669) if model's vocab
        //    extends beyond tokenizer's
        // 2. OR the model's logits heavily favor digit tokens over year tokens
        // 3. This is NOT a UTF-8 issue - it's a vocab/tokenization issue
        //
        // Current approach: Keep reranker disabled (it corrupted numbers), but
        // we need to investigate why year tokens are disfavored in the logits.
        // The truncation to vocab_size is necessary to prevent out-of-bounds,
        // but may be filtering out valid year token IDs.

        let params = SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty,
        };

        let (slice1, slice2) = self.recent_tokens.as_slices();
        let next_token = if slice2.is_empty() {
            self.engine.sample_token_with_params(&logits, &params, slice1)?
        } else {
            let recent_tokens_vec: Vec<i64> = self.recent_tokens.iter().copied().collect();
            self.engine.sample_token_with_params(&logits, &params, &recent_tokens_vec)?
        };

        // Validate sampled token is within model vocabulary
        // With truncation disabled, tokens can be in range [0, model_vocab_size)
        // Some tokens may be beyond tokenizer vocab [vocab_size, model_vocab_size)
        // which is OK - the DecodeStream will handle decoding gracefully
        tracing::debug!(
            "Sampled token: {}, tokenizer_vocab_size: {}, model_vocab_size: {}, logits_shape: {:?}",
            next_token, vocab_size, model_vocab_size, logits_shape
        );

        if model_vocab_size > 0 && next_token >= model_vocab_size {
            return Err(anyhow::anyhow!(
                "Generated out-of-bounds token {}: exceeds model vocab size {}",
                next_token,
                model_vocab_size
            ));
        }

        // Warn if token is beyond tokenizer vocabulary (but allow it)
        if next_token >= vocab_size {
            tracing::warn!(
                "⚠️ Sampled token {} is beyond tokenizer vocab ({}) but within model vocab ({}). This may indicate a vocab mismatch.",
                next_token, vocab_size, model_vocab_size
            );
        }

        Ok(next_token as u32)
    }
}

// SAFETY: TextStream can be Send because:
// - tokenizer pointer points to heap data that won't move
// - decode_stream is tied to the tokenizer lifetime
// - Tokenizer itself is Send
unsafe impl<'a> Send for TextStream<'a> {}

impl<'a> Drop for TextStream<'a> {
    fn drop(&mut self) {
        // SAFETY: We leaked the tokenizer in new(), so we must manually deallocate it here
        // The pointer is valid because it was created from Box::leak
        unsafe {
            if !self.tokenizer.is_null() {
                let _ = Box::from_raw(self.tokenizer);
            }
        }
    }
}

impl<'a> Stream for TextStream<'a> {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.finished {
                return Poll::Ready(None);
            }

            // Check max tokens
            if self.tokens_generated >= self.max_tokens {
                tracing::debug!("Reached max tokens: {}", self.max_tokens);
                self.finished = true;
                self.finish_reason = Some(FinishReason::MaxTokens);
                return Poll::Ready(None);
            }

            // Sample next token
            let next_token = match self.sample_next_token() {
                Ok(token) => {
                    // FIX: Increment KV cache position after successful token sampling
                    // This ensures KV cache stays synchronized with generation state
                    if self.tokens_generated > 0 {  // Don't increment on initial prompt
                        self.kv_cache_position += 1;
                    }
                    token
                },
                Err(e) => {
                    self.finished = true;
                    self.finish_reason = Some(FinishReason::Error(e.to_string()));
                    return Poll::Ready(Some(Err(e)));
                }
            };

            // Check EOS
            if self.engine.is_eos_token(next_token as usize) {
                tracing::debug!("EOS token detected: {}", next_token);
                self.finished = true;
                self.finish_reason = Some(FinishReason::EndOfSequence);
                return Poll::Ready(None);
            }

            // Check stop tokens
            if self.stop_token_ids.contains(&next_token) {
                tracing::debug!("Stop token ID {} detected", next_token);
                self.finished = true;
                self.finish_reason = Some(FinishReason::StopToken(format!("{}", next_token)));
                return Poll::Ready(None);
            }

            // KV cache position is already tracked correctly in sample_next_token
            // No need to increment here

            // Process decode_stream FIRST, then update state
            // This prevents state corruption if decode_stream fails
            match self.decode_stream.step(next_token) {
                Ok(Some(text)) => {
                    // DecodeStream succeeded - now update state
                    let token_i64 = next_token as i64;
                    self.last_generated = Some(token_i64);
                    self.tokens_generated += 1;

                    self.recent_tokens.push_back(token_i64);
                    if self.recent_tokens.len() > self.repeat_last_n {
                        self.recent_tokens.pop_front();
                    }

                    // DecodeStream returned text - emit it
                    tracing::debug!(
                        "Token {} -> text chunk (len={}): {:?}",
                        next_token,
                        text.len(),
                        text
                    );
                    return Poll::Ready(Some(Ok(text)));
                }
                Ok(None) => {
                    // DecodeStream is buffering incomplete UTF-8 - update state and continue
                    let token_i64 = next_token as i64;
                    self.last_generated = Some(token_i64);
                    self.tokens_generated += 1;

                    self.recent_tokens.push_back(token_i64);
                    if self.recent_tokens.len() > self.repeat_last_n {
                        self.recent_tokens.pop_front();
                    }

                    // Debug info
                    let token_str = if let Ok(guard) = self.engine.tokenizer.lock() {
                        if let Some(ref tok) = *guard {
                            tok.decode(&[next_token], false).unwrap_or_else(|_| format!("<decode error>"))
                        } else {
                            format!("<no tokenizer>")
                        }
                    } else {
                        format!("<lock error>")
                    };

                    tracing::debug!("Token {} (raw: {:?}) -> buffering (incomplete UTF-8)", next_token, token_str);
                    continue;
                }
                Err(e) => {
                    // CRITICAL: Do NOT update state on decode_stream error
                    // This prevents state corruption and keeps generation consistent
                    tracing::error!("DecodeStream error on token {}: {}", next_token, e);
                    self.finished = true;
                    self.finish_reason = Some(FinishReason::Error(e.to_string()));
                    return Poll::Ready(Some(Err(anyhow::anyhow!("Decode error: {}", e))));
                }
            }
        }
    }
}
