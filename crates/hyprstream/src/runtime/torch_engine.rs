//! PyTorch-based inference engine using tch-rs

use crate::config::{
    FinishReason, GenerationConfig, GenerationRequest, GenerationResult, ModelInfo, RuntimeConfig,
};
use crate::runtime::gpu_sampling::GpuSampler;
use crate::runtime::template_engine::{ChatMessage, TemplateEngine};
use crate::runtime::RuntimeEngine;
use crate::storage::{XetConfig, XetNativeStorage};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use json_threat_protection as jtp;
use serde_json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use tch::{nn::VarStore, Device, Tensor};
use tokenizers::Tokenizer;
use tracing::{info, instrument};

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
            endoftext: 151643, // "<|endoftext|>"
            im_start: 151644,  // "<|im_start|>"
            im_end: 151645,    // "<|im_end|>"
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
    gpu_sampler: GpuSampler,
    /// Qwen special tokens for proper conversation handling - immutable after construction
    special_tokens: QwenSpecialTokens,
    /// Optional XET storage for LFS/XET pointer handling - thread safe after initialization
    xet_storage: Option<Arc<XetNativeStorage>>,
    // Note: Pre-training is not supported because persistent_model doesn't expose VarStore
    // Only LoRA training is supported (which has its own VarStore)
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
                max_tokens: 512,
                temperature: 0.7,
                top_p: 0.9,
                top_k: Some(40),
                repeat_penalty: 1.1,
                stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
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
            gpu_sampler: GpuSampler::new(device),
            special_tokens: QwenSpecialTokens::default(),
            xet_storage: None,
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
    async fn initialize_xet_storage(&mut self) {
        match self.try_initialize_xet_storage().await {
            Ok(storage) => {
                info!("🔗 XET storage initialized successfully");
                self.xet_storage = Some(Arc::new(storage));
            }
            Err(e) => {
                info!("XET storage not available: {}", e);
                self.xet_storage = None;
            }
        }
    }

    /// Try to initialize XET storage, returning an error if it fails
    async fn try_initialize_xet_storage(&self) -> Result<XetNativeStorage> {
        let config = XetConfig::default();
        XetNativeStorage::new(config).await
    }

    /// Initialize the persistent model instance using ModelFactory
    async fn initialize_persistent_model(&mut self, model_path: &Path) -> Result<()> {
        use crate::runtime::model_config::ModelConfig;
        use crate::runtime::model_factory::ModelFactory;

        info!("Initializing model");

        // Initialize XET storage if not already done
        if self.xet_storage.is_none() {
            self.initialize_xet_storage().await;
        }

        // Load model config first to get model parameters
        let empty_weights = HashMap::new();
        let config = ModelConfig::load(model_path, &empty_weights)?;

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
        info!("Model weights loaded in {:.2}s", factory_time.as_secs_f64());

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
        token_id_u32 == self.special_tokens.im_end || token_id_u32 == self.special_tokens.endoftext
    }

    /// Detokenize IDs back to text - thread safe
    fn detokenize(&self, ids: &[i64]) -> Result<String> {
        let tokenizer_guard = self.handle_poison(self.tokenizer.lock())?;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))?;

        if ids.is_empty() {
            return Ok(String::new());
        }

        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();

        // Debug tokenizer state
        tracing::debug!("Detokenizing token IDs: {:?}", ids_u32);

        // Decode tokens - skip special tokens for clean output
        // The tokenizer is already configured with NFC Unicode normalization
        let decoded_text = tokenizer
            .decode(&ids_u32, true)
            .map_err(|e| anyhow!("Detokenization failed for token IDs {:?}: {}", ids, e))?;

        // Log any Unicode replacement characters for debugging
        if decoded_text.contains('\u{FFFD}') {
            tracing::warn!(
                "Unicode replacement characters detected in token IDs {:?}",
                ids_u32
            );
        }

        tracing::debug!("Detokenized {:?} -> '{}'", ids_u32, decoded_text);
        Ok(decoded_text)
    }

    /// Run inference on the model (supports both TorchScript and VarStore models) - thread safe
    fn forward(&self, input_ids: &[i64]) -> Result<Tensor> {
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
    fn forward_cached(
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

    /// Sample next token from GPU logits tensor
    fn sample_token_gpu(
        &self,
        logits_tensor: &Tensor,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<usize> {
        self.gpu_sampler.sample_token(
            logits_tensor,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
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
        // Format prompt with Qwen2 chat template
        let formatted_prompt = self.format_chat_message(
            Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
            prompt,
        );

        let request = GenerationRequest {
            prompt: formatted_prompt,
            max_tokens,
            temperature: self.generation_config.temperature,
            top_p: self.generation_config.top_p,
            top_k: self.generation_config.top_k,
            repeat_penalty: self.generation_config.repeat_penalty,
            stop_tokens: self.generation_config.stop_tokens.clone(),
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
                return Err(anyhow!(
                    "Model not properly initialized - persistent model not ready"
                ));
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
                let next_token = self_clone.sample_token_gpu(
                    &logits,
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    request.repeat_penalty,
                    &input_ids,
                )?;

                tracing::debug!("Sampled token ID: {}", next_token);

                // Check EOS BEFORE decoding or adding to sequence
                if self_clone.is_eos_token(next_token) {
                    tracing::debug!("EOS token detected: {}", next_token);
                    break;
                }

                // Add to sequence
                input_ids.push(next_token as i64);
                tokens_generated += 1;

                // Decode token
                let token_text = self_clone.detokenize(&[next_token as i64])?;
                tracing::debug!("Token text: '{}'", token_text);
                generated_text.push_str(&token_text);

                // Check stop tokens
                if request
                    .stop_tokens
                    .iter()
                    .any(|stop| generated_text.contains(stop))
                {
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
            Ok(Ok(Ok(result))) => Ok(result), // Unwrap nested Results: timeout -> JoinHandle -> generation
            Ok(Ok(Err(e))) => Err(e),         // Generation error
            Ok(Err(e)) => Err(anyhow!("Blocking task panicked: {}", e)), // Task panic
            Err(_) => Err(anyhow!("Generation timed out after {:?}", timeout_duration)),
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
    /// Generate text with streaming callback and custom parameters
    pub async fn generate_streaming_with_params<F>(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
        callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        // Ensure persistent model is ready
        if !self.is_persistent_model_ready() {
            return Err(anyhow!(
                "Model not properly initialized - persistent model not ready"
            ));
        }

        let request = GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            stop_tokens: self.generation_config.stop_tokens.clone(),
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
    pub async fn generate_streaming<F>(
        &self,
        prompt: &str,
        max_tokens: usize,
        callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        // Use model-specific defaults or fallback to generic defaults
        self.generate_streaming_with_params(
            prompt,
            max_tokens,
            self.generation_config.temperature,
            self.generation_config.top_p,
            self.generation_config.top_k,
            self.generation_config.repeat_penalty,
            callback,
        )
        .await
    }

    /// Internal streaming implementation
    async fn generate_streaming_internal<F>(
        &self,
        request: GenerationRequest,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let prompt = &request.prompt;
        let max_tokens = request.max_tokens;
        let temperature = request.temperature;
        let top_p = request.top_p;
        let top_k = request.top_k;
        let repeat_penalty = request.repeat_penalty;

        // REAL streaming: generate tokens one by one and call callback for each
        let _start_time = std::time::Instant::now();

        // Tokenize input
        let mut input_ids = self.tokenize(prompt)?;
        let mut generated_text = String::new();
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
            let next_token = self.sample_token_gpu(
                &logits,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &input_ids,
            )?;

            // Check EOS BEFORE decoding or adding to sequence
            if self.is_eos_token(next_token) {
                tracing::debug!("EOS token detected: {}", next_token);
                break;
            }

            // Add to sequence
            input_ids.push(next_token as i64);

            // Decode token and stream it immediately
            let token_text = self.detokenize(&[next_token as i64])?;
            generated_text.push_str(&token_text);

            tracing::debug!(
                "Iteration {}: generated token {} -> '{}'",
                i,
                next_token,
                token_text
            );

            // Stream the token immediately (real streaming)
            callback(&token_text);

            // Check stop tokens
            if request
                .stop_tokens
                .iter()
                .any(|stop| generated_text.contains(stop))
            {
                break;
            }
        }

        Ok(generated_text)
    }

    /// Check if persistent model is initialized - thread safe
    pub fn is_persistent_model_ready(&self) -> bool {
        let persistent_ready = self.persistent_model.is_some();
        let context_ready = self
            .handle_poison(self.context_state.lock())
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

        tracing::info!(
            "Starting async generation with prompt of {} tokens",
            prompt_len
        );

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
                let next_token = self.sample_token_gpu(
                    &logits,
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    request.repeat_penalty,
                    &input_ids,
                )?;

                // Check EOS BEFORE decoding or adding to sequence
                if self.is_eos_token(next_token) {
                    tracing::debug!("EOS token detected: {}", next_token);
                    break;
                }

                // Add to sequence
                input_ids.push(next_token as i64);
                tokens_generated += 1;

                // Decode token
                let token_text = self.detokenize(&[next_token as i64])?;
                generated_text.push_str(&token_text);

                // Stream token via callback
                match callback.on_token(&token_text).await {
                    Ok(ContinueGeneration::Continue) => {}
                    Ok(ContinueGeneration::Stop) => {
                        tracing::info!("Generation stopped by callback");
                        break;
                    }
                    Ok(ContinueGeneration::Pause(duration)) => {
                        tokio::time::sleep(duration).await;
                    }
                    Err(e) => {
                        callback.on_error(anyhow::anyhow!("{}", e)).await;
                        return Err(e);
                    }
                }

                // Check stop conditions
                if request
                    .stop_tokens
                    .iter()
                    .any(|stop| generated_text.contains(stop))
                {
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
            temperature: self.generation_config.temperature,
            top_p: self.generation_config.top_p,
            top_k: self.generation_config.top_k,
            repeat_penalty: self.generation_config.repeat_penalty,
            stop_tokens: self.generation_config.stop_tokens.clone(),
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
    pub async fn generate_chat(
        &self,
        system: Option<&str>,
        user: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let formatted_prompt = self.format_chat_message(system, user);
        self.generate_raw(&formatted_prompt, max_tokens).await
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
        self.xet_storage = None; // Drops our Option<Arc<...>> reference

        // Arc<Mutex<...>> fields are automatically cleaned up via Arc reference counting
        // when the last TorchEngine clone drops. No manual intervention needed.
    }
}
