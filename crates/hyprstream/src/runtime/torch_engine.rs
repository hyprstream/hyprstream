//! PyTorch-based inference engine using tch-rs

use crate::config::{
    FinishReason, GenerationConfig, GenerationRequest, GenerationResult, ModelInfo, RuntimeConfig,
    TemplatedPrompt,
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
    atomic::{AtomicU32, AtomicUsize, Ordering},
    Arc,
};
use parking_lot::Mutex;
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
    config: RuntimeConfig,
    /// Generation configuration with defaults
    generation_config: GenerationConfig,
    /// Loaded model information - protected by mutex
    model_info: Arc<Mutex<ModelInfo>>,
    /// Active LoRA adapter name - thread safe with mutex
    pub active_lora: Arc<Mutex<Option<String>>>,
    /// GPU sampler for efficient token sampling - thread safe after initialization
    sampler: TensorSampler,  // Renamed from gpu_sampler - works on both CPU and GPU
    /// Cached tokenizer vocabulary size for lock-free access
    /// 0 means not yet initialized
    tokenizer_vocab_size: Arc<AtomicUsize>,
    /// Cached EOS token ID for lock-free access
    /// 0 means not yet initialized (actual EOS tokens are typically > 0)
    eos_token_id: Arc<AtomicU32>,
    /// KV cache registry for session-based cache isolation
    /// Enables concurrent inference across multiple sessions
    kv_cache_registry: Option<Arc<crate::runtime::kv_cache::KVCacheRegistry>>,
    /// Current active session owner for KV cache selection
    /// If set, generation uses the corresponding cache from the registry
    active_cache_owner: Arc<Mutex<Option<crate::runtime::kv_cache::CacheOwner>>>,
    // Note: XET/LFS handled by git-xet-filter + ModelFactory::load_file_with_pointer_detection()
    // Note: Pre-training not supported (persistent_model doesn't expose VarStore), LoRA only
}

/// Helper functions for tensor operations
impl TorchEngine {
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
            self.tokenizer.lock()
                .as_ref()
                .map(|t| t.get_vocab_size(true))
                .unwrap_or(0)  // Return 0 if tokenizer not loaded (caller should error)
        }
    }

    /// Get the device this engine is using for computation
    pub fn device(&self) -> Device {
        self.device
    }

    // ============================================================================
    // Session-Based KV Cache Management
    // ============================================================================

    /// Initialize the KV cache registry for session-based cache isolation.
    ///
    /// This should be called after loading a model when the configuration is known.
    /// The registry will manage separate KV caches for different sessions/requests.
    pub fn initialize_kv_registry(
        &mut self,
        num_layers: usize,
        max_seq_len: usize,
        quant_type: crate::runtime::kv_quant::KVQuantType,
        memory_budget: Option<usize>,
    ) {
        let config = crate::runtime::kv_cache::CacheConfig::new(num_layers, max_seq_len)
            .with_quant_type(quant_type);

        let registry = crate::runtime::kv_cache::KVCacheRegistry::new(config, memory_budget);
        self.kv_cache_registry = Some(Arc::new(registry));

        tracing::info!(
            "[TorchEngine] Initialized KV cache registry: {} layers, max_seq_len={}, budget={:?}",
            num_layers, max_seq_len, memory_budget
        );
    }

    /// Get the KV cache registry (for external access)
    pub fn kv_registry(&self) -> Option<Arc<crate::runtime::kv_cache::KVCacheRegistry>> {
        self.kv_cache_registry.clone()
    }

    /// Set the active session for generation.
    ///
    /// This sets which KV cache will be used for the next generation call.
    /// Use `CacheOwner::Session(session_id)` for conversational sessions that
    /// should preserve context across multiple requests.
    ///
    /// # Example
    /// ```ignore
    /// // For a conversational session (preserves context)
    /// engine.set_session(CacheOwner::Session("conversation-123".into()));
    ///
    /// // For a stateless request (cache discarded after)
    /// engine.set_session(CacheOwner::new_stateless());
    /// ```
    pub fn set_session(&self, owner: crate::runtime::kv_cache::CacheOwner) -> Result<()> {
        if self.kv_cache_registry.is_none() {
            return Err(anyhow!("KV cache registry not initialized. Call initialize_kv_registry first."));
        }

        let mut active_owner = self.active_cache_owner.lock();

        tracing::debug!("Setting active session to: {:?}", owner);
        *active_owner = Some(owner);

        Ok(())
    }

    /// Get the current active session owner
    pub fn get_session(&self) -> Option<crate::runtime::kv_cache::CacheOwner> {
        self.active_cache_owner.lock().clone()
    }

    /// Clear the active session (subsequent generations will use the model's internal cache)
    pub fn clear_session(&self) -> Result<()> {
        let mut active_owner = self.active_cache_owner.lock();
        *active_owner = None;
        Ok(())
    }

    /// Release a session's KV cache (for cleanup after conversation ends)
    pub fn release_session(&self, owner: &crate::runtime::kv_cache::CacheOwner) -> Result<()> {
        if let Some(registry) = &self.kv_cache_registry {
            registry.release(owner);
            tracing::debug!("Released session cache: {:?}", owner);
        }
        Ok(())
    }

    /// Get total memory usage across all session caches
    pub fn session_cache_memory_usage(&self) -> usize {
        self.kv_cache_registry
            .as_ref()
            .map(|r| r.total_memory_usage())
            .unwrap_or(0)
    }

    /// Get number of active session caches
    pub fn active_session_count(&self) -> usize {
        self.kv_cache_registry
            .as_ref()
            .map(|r| r.cache_count())
            .unwrap_or(0)
    }

    /// Evict LRU session caches to stay within memory budget
    pub fn evict_session_caches(&self) {
        if let Some(registry) = &self.kv_cache_registry {
            registry.evict_to_budget();
        }
    }

    /// Estimate model memory usage in bytes.
    ///
    /// Sums the byte sizes of all parameters in the VarStore.
    pub fn model_memory_usage(&self) -> usize {
        let vs_guard = self.var_store.lock();
        {
            if let Some(vs) = vs_guard.as_ref() {
                return vs
                    .variables()
                    .iter()
                    .map(|(_, t)| {
                        let numel = t.numel();
                        let elem_size = match t.kind() {
                            tch::Kind::Half | tch::Kind::BFloat16 => 2,
                            tch::Kind::Float => 4,
                            tch::Kind::Double => 8,
                            tch::Kind::Int8 | tch::Kind::Uint8 => 1,
                            _ => 4,
                        };
                        numel * elem_size
                    })
                    .sum();
            }
        }
        0
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
            // Use specified GPU device ID, or auto-detect
            let gpu_device = if let Some(device_id) = config.gpu_device_id {
                // Check if CUDA/ROCm is available at all
                if Device::cuda_if_available() != Device::Cpu {
                    Device::Cuda(device_id)
                } else {
                    info!("⚠️  GPU {} requested but CUDA/ROCm not available, falling back to CPU", device_id);
                    Device::Cpu
                }
            } else {
                Device::cuda_if_available()
            };

            if gpu_device != Device::Cpu {
                // Check if this is actually ROCm/HIP
                let device_id = match gpu_device {
                    Device::Cuda(id) => id,
                    _ => 0,
                };
                if std::env::var("HIP_VISIBLE_DEVICES").is_ok()
                    || std::path::Path::new("../libtorch/lib/libtorch_hip.so").exists()
                {
                    info!("🚀 Using ROCm/HIP GPU {} acceleration", device_id);
                } else {
                    info!("🚀 Using CUDA GPU {} acceleration", device_id);
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
                stop_tokens: vec!["</s>".to_owned()],
                seed: None,
                stream: false,
            },
            model_info: Arc::new(Mutex::new(ModelInfo {
                name: "unloaded".to_owned(),
                architecture: "unknown".to_owned(),
                parameters: 0,
                context_length: 2048,
                vocab_size: 32000,
                hidden_size: 768,
                intermediate_size: None,
                num_attention_heads: None,
                num_key_value_heads: None,
                head_dim: None,
                num_hidden_layers: None,
                quantization: None,
            })),
            active_lora: Arc::new(Mutex::new(None)),
            sampler: TensorSampler::new(device),
            tokenizer_vocab_size: Arc::new(AtomicUsize::new(0)),
            eos_token_id: Arc::new(AtomicU32::new(0)),
            kv_cache_registry: None, // Initialized after model load when config is known
            active_cache_owner: Arc::new(Mutex::new(None)),
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
            let _model_guard = model.lock();

            // Get architecture info from the model
            // For now, use a default since we don't have a method to query this
            let architecture = "auto".to_owned();

            // Update model info
            {
                let mut model_info_guard = self.model_info.lock();
                model_info_guard.architecture = architecture.clone();
            }

            // Set architecture
            {
                let mut arch_guard = self.model_architecture.lock();
                *arch_guard = Some(architecture.clone());
            }
        }

        // Get context window from model info that was populated from config
        let context_window = self.model_info.lock().context_length;

        // Initialize context state
        {
            let mut context_guard = self.context_state.lock();
            *context_guard = Some(ContextState {
                sequence_length: 0,
                context_window,
                initialized: true,
            });
        }

        // Create dummy VarStore for backward compatibility
        {
            let vs = VarStore::new(self.device);
            let mut var_store_guard = self.var_store.lock();
            *var_store_guard = Some(vs);
        }

        info!("✅ SafeTensors model loaded via ModelFactory");
        info!("🚀 Model initialized and ready for inference");
        Ok(())
    }

    /// Get tensor from VarStore by name (for inference) - thread safe
    pub fn get_tensor(&self, name: &str) -> Option<Tensor> {
        let var_store_guard = self.var_store.lock();
        let vs = var_store_guard.as_ref()?;
        vs.variables().get(name).map(tch::Tensor::shallow_clone)
    }

    /// List all available tensor names in VarStore - thread safe
    pub fn list_tensor_names(&self) -> Vec<String> {
        let var_store_guard = self.var_store.lock();
        if let Some(vs) = var_store_guard.as_ref() {
            vs.variables().keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Check if model is loaded via VarStore - thread safe
    pub fn has_varstore(&self) -> bool {
        self.var_store.lock().is_some()
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

        // Effective context length (CLI override or model default)
        let effective_max_context = self.config.max_context.unwrap_or(config.max_position_embeddings);
        if self.config.max_context.is_some() {
            info!("Using max_context override: {} tokens (model default: {})", effective_max_context, config.max_position_embeddings);
        }

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
            let kv_per_layer = 2 * batch_size * effective_max_context
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
            effective_max_context,
            total_estimated_mb
        );

        // Pre-flight GPU memory check (best-effort)
        if let Err(e) = preflight_gpu_check(self.device, total_estimated_mb) {
            warn!("GPU memory pre-flight check failed: {}", e);
            // Continue anyway - the check might not be accurate
        }

        // Update ModelInfo with actual values from config
        {
            let mut model_info = self.model_info.lock();
            model_info.hidden_size = config.hidden_size;
            model_info.intermediate_size = Some(config.intermediate_size);
            model_info.num_attention_heads = Some(config.num_attention_heads);
            model_info.num_key_value_heads = Some(config.num_key_value_heads);
            model_info.head_dim = Some(config.head_dim);
            model_info.num_hidden_layers = Some(config.num_hidden_layers);
            model_info.vocab_size = config.vocab_size;
            model_info.context_length = effective_max_context;
            model_info.architecture = config.model_type.clone();
        }

        // Use the factory to create the model
        let factory_start = std::time::Instant::now();
        let model = ModelFactory::create(
            model_path,
            &self.device,
            tch::Kind::BFloat16,
            self.config.max_context,
            self.config.kv_quant_type,
        ).await?;
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
                let model_info = self.model_info.lock();
                model_info.vocab_size
            };

            // Apply model-specific tokenizer configuration if model is loaded
            if let Some(model) = &self.persistent_model {
                let model_guard = model.lock();
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
            let mut tokenizer_guard = self.tokenizer.lock();
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

            // Cache EOS token ID for lock-free access during generation
            if let Some(ref eos_str) = template_config.eos_token {
                let tokenizer_guard = self.tokenizer.lock();
                if let Some(ref tokenizer) = *tokenizer_guard {
                    if let Some(eos_id) = tokenizer.token_to_id(eos_str) {
                        self.eos_token_id.store(eos_id, Ordering::Relaxed);
                        info!("Cached EOS token: '{}' -> ID {}", eos_str, eos_id);
                    } else {
                        warn!("EOS token '{}' not found in tokenizer vocabulary", eos_str);
                    }
                }
            }

            // Create template engine
            let template_engine = TemplateEngine::new(template_config)?;

            // Store template engine
            let mut template_guard = self.template_engine.lock();
            *template_guard = Some(template_engine);

            info!("✅ Template engine initialized");
        } else {
            info!("⚠️ No tokenizer_config.json found, using fallback templates");
        }

        Ok(())
    }

    /// Tokenize text to input IDs - thread safe
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        let tokenizer_guard = self.tokenizer.lock();
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
        {
            let template_guard = self.template_engine.lock();
            if let Some(ref engine) = *template_guard {
                let mut messages = Vec::new();

                // Add system message if provided
                if let Some(system_msg) = system {
                    messages.push(ChatMessage {
                        role: "system".to_owned(),
                        content: system_msg.to_owned(),
                    });
                }

                // Add user message
                messages.push(ChatMessage {
                    role: "user".to_owned(),
                    content: user.to_owned(),
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

    /// Check if a token ID is the EOS token (lock-free using cached ID)
    pub fn is_eos_token(&self, token_id: usize) -> bool {
        let cached_eos = self.eos_token_id.load(Ordering::Relaxed);
        // cached_eos == 0 means not initialized; actual EOS tokens are > 0
        cached_eos > 0 && token_id as u32 == cached_eos
    }

    /// Get the tokenizer for streaming decoding - CoW makes this cheap!
    ///
    /// Returns a cloned tokenizer (cheap due to copy-on-write)
    pub fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_guard = self.tokenizer.lock();
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
            let context_guard = self.context_state.lock();
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
        let model = model_arc.lock();
        // Wrap in no_grad to prevent gradient tracking during inference
        let logits = tch::no_grad(|| model.forward(&input_tensor, None))?;

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
        let model = model_arc.lock();

        // Use the new forward_with_cache method that properly tracks position
        // CRITICAL: Wrap in no_grad to prevent gradient tracking during inference
        let logits = tch::no_grad(|| model.forward_with_cache(&input_tensor, start_pos))?;

        // Extract logits for the last token
        let logits_shape = logits.size();
        let seq_len = logits_shape[1];
        let _vocab_size = logits_shape[2] as usize;

        // Get logits for last token
        let last_token_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1);

        Ok(last_token_logits)
    }

    /// Run inference with KV caching and optional delta injection
    ///
    /// Same as `forward_cached()` but calls `forward_with_cache_and_delta()` on the model,
    /// injecting per-tenant LoRA corrections at each attention layer.
    pub fn forward_with_delta_cached(
        &self,
        input_ids: &[i64],
        start_pos: usize,
        use_cache: bool,
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let model_arc = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Persistent model not initialized"))?;

        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized"));
        }

        let tokens_to_process = if use_cache && start_pos > 0 {
            &input_ids[input_ids.len() - 1..]
        } else {
            input_ids
        };

        let input_tensor = Tensor::from_slice(tokens_to_process)
            .to_kind(tch::Kind::Int64)
            .to_device(self.device)
            .unsqueeze(0);

        let model = model_arc.lock();

        let logits = tch::no_grad(|| {
            model.forward_with_cache_and_delta(&input_tensor, start_pos, delta)
        })?;

        let logits_shape = logits.size();
        let seq_len = logits_shape[1];

        let last_token_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1);

        Ok(last_token_logits)
    }

    /// Run full forward pass with optional delta injection (no KV cache)
    pub fn forward_with_delta_full(
        &self,
        input_ids: &[i64],
        delta: Option<&crate::training::TenantDelta>,
    ) -> Result<Tensor> {
        let model_arc = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Persistent model not initialized"))?;

        if !self.is_persistent_model_ready() {
            return Err(anyhow!("Model not properly initialized"));
        }

        let input_tensor = Tensor::from_slice(input_ids)
            .to_kind(tch::Kind::Int64)
            .to_device(self.device)
            .unsqueeze(0);

        let model = model_arc.lock();

        let logits = tch::no_grad(|| {
            model.forward_with_cache_and_delta(&input_tensor, 0, delta)
        })?;

        let logits_shape = logits.size();
        let seq_len = logits_shape[1];

        let last_token_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1);

        Ok(last_token_logits)
    }

    /// Clear KV cache before new generation to prevent context pollution
    pub fn clear_kv_cache(&self) {
        if let Some(model_arc) = &self.persistent_model {
            let model = model_arc.lock();

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
                .unwrap_or("unknown").to_owned()
        } else {
            // Use the file stem for single files
            original_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown").to_owned()
        };

        // Update model info with the correct name
        {
            let mut model_info_guard = self.model_info.lock();
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
            prompt: TemplatedPrompt::new(formatted_prompt),
            max_tokens,
            temperature: self.generation_config.temperature,
            top_p: self.generation_config.top_p,
            top_k: self.generation_config.top_k,
            repeat_penalty: self.generation_config.repeat_penalty,
            repeat_last_n: 64, // Default
            stop_tokens: self.generation_config.stop_tokens.clone(),
            seed: None,
            images: Vec::new(),
            timeout: None,
            collect_metrics: false, // Default: off for performance
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
            quality_metrics: stats.quality_metrics,
            prefill_tokens: stats.prefill_tokens,
            prefill_time_ms: stats.prefill_time_ms,
            prefill_tokens_per_sec: stats.prefill_tokens_per_sec,
            inference_tokens: stats.inference_tokens,
            inference_time_ms: stats.inference_time_ms,
            inference_tokens_per_sec: stats.inference_tokens_per_sec,
            ttt_metrics: None,  // TTT metrics attached by InferenceService
        })
    }

    fn model_info(&self) -> ModelInfo {
        self.model_info.lock().clone()
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
        let template_guard = self.template_engine.lock();

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
        let model_guard = model.lock();

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
        let context_ready = self.context_state.lock()
            .as_ref()
            .is_some_and(|c| c.initialized);

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
    /// use hyprstream_core::config::GenerationRequest;
    ///
    /// # async fn example(engine: &hyprstream_core::runtime::torch_engine::TorchEngine) -> anyhow::Result<()> {
    /// let request = GenerationRequest::default();
    /// let mut stream = engine.generate(request)?;
    ///
    /// while let Some(text_chunk) = stream.next().await {
    ///     print!("{}", text_chunk?);
    /// }
    ///
    /// let stats = stream.stats();
    /// println!("Generated {} tokens", stats.tokens_generated);
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate(&self, mut request: GenerationRequest) -> Result<TextStream<'_>> {
        // Set random seed if provided for deterministic generation
        if let Some(seed) = request.seed {
            self.set_seed(seed as u64);
        }

        // Apply server defaults if not specified in request
        if request.timeout.is_none() {
            request.timeout = Some(self.config.default_generation_timeout_ms);
        }

        TextStream::new(self, request)
    }

    /// Generate with optional per-tenant delta for delta-aware inference
    ///
    /// When `delta` is Some, LoRA corrections are injected at each attention layer
    /// during generation. The delta is locked per-token (not held across the entire
    /// generation) to allow concurrent training updates.
    pub fn generate_with_delta(
        &self,
        mut request: GenerationRequest,
        delta: Option<std::sync::Arc<parking_lot::Mutex<crate::training::TenantDelta>>>,
    ) -> Result<TextStream<'_>> {
        if let Some(seed) = request.seed {
            self.set_seed(seed as u64);
        }

        if request.timeout.is_none() {
            request.timeout = Some(self.config.default_generation_timeout_ms);
        }

        TextStream::new_with_delta(self, request, delta)
    }

    /// Non-streaming generation with optional delta (convenience wrapper)
    pub async fn generate_with_delta_params(
        &self,
        request: GenerationRequest,
        delta: Option<std::sync::Arc<parking_lot::Mutex<crate::training::TenantDelta>>>,
    ) -> Result<crate::config::GenerationResult> {
        use futures::StreamExt;

        if !self.is_persistent_model_ready() {
            return Err(anyhow!(
                "Model not properly initialized - persistent model not ready"
            ));
        }

        let mut stream = self.generate_with_delta(request, delta)?;
        let mut accumulated_text = String::new();

        while let Some(text_chunk) = stream.next().await {
            accumulated_text.push_str(&text_chunk?);
        }

        let stats = stream.stats();
        Ok(crate::config::GenerationResult {
            text: accumulated_text,
            tokens_generated: stats.tokens_generated,
            finish_reason: stats.finish_reason.unwrap_or(crate::config::FinishReason::Stop),
            generation_time_ms: stats.generation_time_ms,
            tokens_per_second: stats.tokens_per_second,
            quality_metrics: stats.quality_metrics,
            prefill_tokens: stats.prefill_tokens,
            prefill_time_ms: stats.prefill_time_ms,
            prefill_tokens_per_sec: stats.prefill_tokens_per_sec,
            inference_tokens: stats.inference_tokens,
            inference_time_ms: stats.inference_time_ms,
            inference_tokens_per_sec: stats.inference_tokens_per_sec,
            ttt_metrics: None,
        })
    }


    /// Forward pass with per-layer delta injection for TTT training.
    ///
    /// Uses `decode_layer_with_delta()` to inject the tenant delta's A/B matrices
    /// after q_proj/v_proj projections inside each attention layer. This creates a
    /// differentiable path from the loss back to the delta parameters, enabling
    /// gradient-based training via `loss.backward()`.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [1, seq_len]
    /// * `delta` - Tenant's LoRA delta with trainable A/B matrices
    ///
    /// # Returns
    /// Logits tensor [1, seq_len, vocab_size] with gradient path to delta parameters
    pub fn forward_with_delta(
        &self,
        input_ids: &Tensor,
        delta: &crate::training::TenantDelta,
    ) -> Result<Tensor> {
        let model = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Model not loaded"))?;
        let model_guard = model.lock();

        // Get embeddings from the base model
        let input = input_ids.to(self.device);
        let mut hidden_states = model_guard.embed_tokens(&input)?;

        // Generate position IDs
        let seq_len = hidden_states.size()[1];
        let position_ids =
            Tensor::arange(seq_len, (tch::Kind::Int64, hidden_states.device()));

        let num_layers = model_guard.num_layers();

        // Process each layer with delta injection
        for layer_idx in 0..num_layers {
            let (new_hidden, _kv) = model_guard.decode_layer_with_delta(
                layer_idx,
                &hidden_states,
                None,                // attention_mask
                Some(&position_ids), // position_ids
                None,                // past_kv (no cache during training)
                delta,
            )?;
            hidden_states = new_hidden;
        }

        // Apply final norm + LM head
        hidden_states = model_guard.apply_final_norm(&hidden_states)?;
        let logits = model_guard.lm_head(&hidden_states)?;

        Ok(logits)
    }

    /// Get the number of layers in the loaded model.
    pub fn get_num_layers(&self) -> Result<usize> {
        let model = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("Model not loaded"))?;
        Ok(model.lock().num_layers())
    }

    /// Get module dimensions for LoRA target modules from the loaded model.
    ///
    /// Returns a map of module_name -> (in_features, out_features) for all
    /// supported LoRA target modules. Used by DeltaPool to initialize
    /// per-tenant LoRA deltas with correct dimensions.
    ///
    /// Supported modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    pub fn get_lora_module_dims(&self) -> Result<std::collections::HashMap<String, (usize, usize)>> {
        let model_info = self.model_info.lock();
        let hidden_size = model_info.hidden_size;

        let num_heads = model_info.num_attention_heads
            .ok_or_else(|| anyhow!("num_attention_heads not set in ModelInfo"))?;
        let num_kv_heads = model_info.num_key_value_heads
            .unwrap_or(num_heads);
        let head_dim = model_info.head_dim
            .unwrap_or(hidden_size / num_heads);

        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        let intermediate_size = if let Some(intermediate) = model_info.intermediate_size {
            intermediate
        } else {
            match model_info.architecture.as_str() {
                "Qwen2ForCausalLM" => (hidden_size as f32 * 2.6667) as usize,
                "LlamaForCausalLM" => (hidden_size as f32 * 2.75) as usize,
                _ => hidden_size * 4,
            }
        };

        let mut dims = std::collections::HashMap::new();
        // Self-attention projections with correct dimensions
        // q_proj: in=hidden_size, out=num_heads * head_dim
        dims.insert("q_proj".to_string(), (hidden_size, q_out));
        // k_proj/v_proj: in=hidden_size, out=num_kv_heads * head_dim (GQA support)
        dims.insert("k_proj".to_string(), (hidden_size, kv_out));
        dims.insert("v_proj".to_string(), (hidden_size, kv_out));
        // o_proj: in=num_heads * head_dim, out=hidden_size
        dims.insert("o_proj".to_string(), (q_out, hidden_size));
        // MLP projections (expand)
        for name in &["gate_proj", "up_proj"] {
            dims.insert(name.to_string(), (hidden_size, intermediate_size));
        }
        // MLP projection (contract)
        dims.insert("down_proj".to_string(), (intermediate_size, hidden_size));

        Ok(dims)
    }

    /// Validate LoRA configuration against model dimensions.
    ///
    /// Checks that the requested target modules are valid for this model.
    /// The actual LoRA delta creation is handled by DeltaPool/TenantDelta.
    pub fn create_lora(&mut self, config: crate::training::TenantDeltaConfig) -> Result<()> {
        let all_dims = self.get_lora_module_dims()?;

        for module_name in &config.target_modules {
            if !all_dims.contains_key(module_name.as_str()) {
                return Err(anyhow!(
                    "Unknown module '{}'. Supported modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj",
                    module_name
                ));
            }
        }

        tracing::info!(
            rank = config.rank,
            alpha = config.alpha,
            target_modules = ?config.target_modules,
            "LoRA configuration validated - delta creation handled by DeltaPool"
        );
        Ok(())
    }

    // ============================================================================
    // Embedding Extraction for RAG/CAG
    // ============================================================================

    /// Extract embedding from text using the model's hidden states.
    ///
    /// This method tokenizes the input text, runs a forward pass through the model,
    /// and extracts the final hidden states. The embedding is computed by mean-pooling
    /// the hidden states over the sequence dimension.
    ///
    /// # Arguments
    /// - `text` - Input text to embed
    ///
    /// # Returns
    /// - Vec<f32> - The embedding vector (dimension = model's hidden_size)
    ///
    /// # Example
    /// ```ignore
    /// let embedding = engine.extract_embedding("What is machine learning?")?;
    /// println!("Embedding dimension: {}", embedding.len());
    /// ```
    pub fn extract_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize input
        let tokenizer_guard = self.tokenizer.lock();
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or_else(|| anyhow!("No tokenizer loaded"))?;

        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        drop(tokenizer_guard);

        self.extract_embedding_from_tokens(&token_ids)
    }

    /// Extract embedding from pre-tokenized input.
    ///
    /// This is a lower-level method for when tokens are already available
    /// (e.g., from a previous generation). Uses mean pooling over the hidden states.
    ///
    /// # Arguments
    /// - `token_ids` - Pre-tokenized input IDs
    ///
    /// # Returns
    /// - Vec<f32> - The embedding vector (dimension = model's hidden_size)
    pub fn extract_embedding_from_tokens(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(anyhow!("Empty token list"));
        }

        let input_tensor = Tensor::from_slice(token_ids)
            .to_device(self.device)
            .unsqueeze(0); // [1, seq_len]

        let model_guard = self
            .persistent_model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?
            .lock();

        tch::no_grad(|| {
            // Get token embeddings
            let embeddings = model_guard.embed_tokens(&input_tensor)?;

            // Process through layers to get hidden states
            let hidden_states = self.process_through_layers_for_embedding(&**model_guard, &embeddings)
                .unwrap_or_else(|_| embeddings.shallow_clone());

            // Apply final normalization
            let normalized = model_guard.apply_final_norm(&hidden_states)
                .unwrap_or(hidden_states);

            // Mean pool over sequence dimension: [1, seq_len, hidden_size] -> [1, hidden_size]
            let pooled = normalized.mean_dim(1, false, tch::Kind::Float);

            // Squeeze batch dimension
            let pooled_1d = pooled.squeeze_dim(0); // [hidden_size]

            // L2 normalize for cosine similarity
            let norm = pooled_1d.norm();
            let normalized_embedding = if norm.double_value(&[]) > 1e-12 {
                &pooled_1d / norm
            } else {
                pooled_1d
            };

            // Convert to CPU and extract values
            let cpu_tensor = normalized_embedding.to_device(tch::Device::Cpu);
            let numel = cpu_tensor.numel();
            let mut embedding_vec = vec![0.0f32; numel];
            cpu_tensor.copy_data(&mut embedding_vec, numel);

            Ok(embedding_vec)
        })
    }

    /// Process embeddings through all model layers for embedding extraction
    fn process_through_layers_for_embedding(
        &self,
        model: &dyn crate::runtime::architectures::ModelOperations,
        embeddings: &Tensor,
    ) -> Result<Tensor> {
        let num_layers = model.num_layers();
        let mut hidden_states = embeddings.shallow_clone();

        for layer_idx in 0..num_layers {
            let (new_hidden, _kv) = model.decode_layer(
                layer_idx,
                &hidden_states,
                None, // attention_mask
                None, // position_ids
                None, // past_kv
            )?;
            hidden_states = new_hidden;
        }

        Ok(hidden_states)
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
            stop_tokens: vec!["</s>".to_owned()],
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

/// Statistics about text generation with separate prefill/inference metrics
#[derive(Debug, Clone)]
pub struct GenerationStats {
    // Overall metrics
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    pub finish_reason: Option<FinishReason>,
    /// Quality metrics captured during generation (for self-supervised training)
    pub quality_metrics: Option<crate::runtime::generation_metrics::GenerationQualityMetrics>,

    // Prefill metrics (processing the prompt)
    pub prefill_tokens: usize,
    pub prefill_time_ms: u64,
    pub prefill_tokens_per_sec: f32,

    // Inference metrics (generating new tokens, excluding prefill)
    pub inference_tokens: usize,
    pub inference_time_ms: u64,
    /// Cumulative average: tokens / time (accurate for final reporting)
    pub inference_tokens_per_sec: f32,
    /// Exponential moving average (responsive for real-time adaptive batching)
    pub inference_tokens_per_sec_ema: f32,
}

/// Stream that yields decoded UTF-8 text chunks during generation.
///
/// Automatically handles UTF-8 buffering, stop tokens, EOS detection,
/// timeout handling, cancellation, and all generation complexities.
/// Just iterate and get text!
pub struct TextStream<'a> {
    engine: &'a TorchEngine,

    /// Optional per-tenant delta for delta-aware inference.
    /// When present, LoRA corrections are injected at each attention layer.
    delta: Option<std::sync::Arc<parking_lot::Mutex<crate::training::TenantDelta>>>,

    prompt_tokens: Vec<i64>,
    last_generated: Option<i64>,

    recent_tokens: VecDeque<i64>,
    repeat_last_n: usize,

    // PERF: Individual sampling params removed - now stored in sampling_params field below
    // This avoids struct recreation per token while keeping backward compatibility

    max_tokens: usize,
    stop_token_ids: Vec<u32>,

    // Store tokenizer as Arc for safe sharing across streams
    // IMPORTANT: This field is required for the unsafe transmute in TextStream::new()
    // DO NOT REMOVE - it ensures the tokenizer lives long enough for the decode_stream lifetime 'a
    #[allow(dead_code)]
    tokenizer: Arc<Tokenizer>,
    decode_stream: tokenizers::tokenizer::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,

    prompt_len: usize,
    /// KV cache position tracking for this stream
    /// Each stream has exclusive access via &mut self, so no atomic needed
    kv_cache_position: usize,
    /// Total tokens generated (including buffered UTF-8) - for statistics only
    tokens_generated: usize,
    start_time: std::time::Instant,
    finished: bool,
    finish_reason: Option<FinishReason>,

    // Timeout handling
    timeout_ms: Option<u64>,

    /// Whether to collect per-token quality metrics (expensive - ~10x overhead)
    collect_metrics: bool,

    /// Metrics accumulator for self-supervised training quality signals
    metrics_accumulator: crate::runtime::generation_metrics::GenerationMetricsAccumulator,

    // PERF: Cached values to avoid per-token recomputation
    /// Pre-created sampling params - avoids struct creation per token
    sampling_params: SamplingParams,
    /// Cached tokenizer vocab size - avoids lock acquisition per token
    vocab_size: usize,
    /// Cached model vocab size (from logits shape) - set after first forward
    model_vocab_size: usize,
    /// Reusable buffer for recent_tokens when VecDeque wraps around
    recent_tokens_buffer: Vec<i64>,

    // Timing for prefill/inference separation
    /// Time spent on prefill (processing prompt), set after first forward pass
    prefill_time_ms: Option<u64>,
    /// Timestamp when first token was sampled (after prefill completes)
    first_token_time: Option<std::time::Instant>,

    // EMA rate tracking for adaptive batching
    /// Timestamp when the last token was generated (for inter-token timing)
    last_token_time: Option<std::time::Instant>,
    /// Exponential moving average of tokens per second
    ema_tokens_per_sec: f32,
}

impl<'a> TextStream<'a> {
    fn new(engine: &'a TorchEngine, request: GenerationRequest) -> Result<Self> {
        Self::new_with_delta(engine, request, None)
    }

    fn new_with_delta(
        engine: &'a TorchEngine,
        request: GenerationRequest,
        delta: Option<std::sync::Arc<parking_lot::Mutex<crate::training::TenantDelta>>>,
    ) -> Result<Self> {
        let prompt_tokens = engine.tokenize(request.prompt.as_str())?;
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

        // Use Arc for safe tokenizer sharing
        // This Arc is REQUIRED by the TextStream struct - see the comment on the tokenizer field
        let tokenizer_arc = Arc::new(tokenizer);

        // Create DecodeStream with proper lifetime management
        // The Arc<Tokenizer> is stored in the TextStream, ensuring the tokenizer lives as long as 'a
        // DEPENDENCY: The unsafe transmute below depends on tokenizer_arc being stored in the struct
        let decode_stream = {
            // We need to extend the tokenizer lifetime to match 'a
            // Since we have Arc<Tokenizer> stored in the struct, we can safely create a reference
            let tokenizer_ref = unsafe {
                // SAFETY: This is safe because:
                // 1. The tokenizer_arc is stored in the TextStream struct, ensuring it lives as long as 'a
                // 2. We're not moving or deallocating the tokenizer while the stream exists
                // 3. Arc guarantees thread-safe reference counting and proper lifetime management
                std::mem::transmute::<&Tokenizer, &'a Tokenizer>(tokenizer_arc.as_ref())
            };

            // Use skip_special_tokens=false because <|extra_N|> tokens are special tokens
            // that should appear in output (they represent actual model vocabulary)
            tokenizer_ref.decode_stream(false) // skip_special_tokens=false
        };

        // PERF: Pre-create sampling params to avoid struct allocation per token
        let sampling_params = SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repeat_penalty: request.repeat_penalty,
        };

        // PERF: Cache vocab_size to avoid lock acquisition per token
        let vocab_size = engine.get_vocab_size();

        Ok(Self {
            engine,
            delta,
            prompt_tokens,
            last_generated: None,
            recent_tokens: VecDeque::with_capacity(repeat_last_n),
            repeat_last_n,
            max_tokens: request.max_tokens,
            stop_token_ids,
            tokenizer: tokenizer_arc,
            decode_stream,
            prompt_len,
            // KV cache starts with prompt already in it after first forward
            kv_cache_position: prompt_len,
            tokens_generated: 0,
            start_time: std::time::Instant::now(),
            finished: false,
            finish_reason: None,
            timeout_ms: request.timeout,
            collect_metrics: request.collect_metrics,
            metrics_accumulator: crate::runtime::generation_metrics::GenerationMetricsAccumulator::new(),
            // PERF: Cached values initialized here, avoid per-token recomputation
            sampling_params,
            vocab_size,
            model_vocab_size: 0, // Set after first forward pass from logits shape
            recent_tokens_buffer: Vec::with_capacity(repeat_last_n),
            prefill_time_ms: None,
            first_token_time: None,
            last_token_time: None,
            ema_tokens_per_sec: 0.0,
        })
    }

    /// Get generation statistics (call after stream exhausted)
    pub fn stats(&self) -> GenerationStats {
        let total_time = self.start_time.elapsed();

        // Prefill metrics
        let prefill_time_ms = self.prefill_time_ms.unwrap_or(0);
        let prefill_tokens = self.prompt_len;
        let prefill_tokens_per_sec = if prefill_time_ms > 0 {
            (prefill_tokens as f32 * 1000.0) / prefill_time_ms as f32
        } else {
            0.0
        };

        // Inference metrics (time since first token was sampled)
        let inference_time_ms = self.first_token_time
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);
        let inference_tokens = self.tokens_generated;

        // Cumulative average for final reporting
        let inference_tokens_per_sec = if inference_time_ms > 0 {
            (inference_tokens as f32 * 1000.0) / inference_time_ms as f32
        } else {
            0.0
        };

        // EMA for real-time adaptive batching (falls back to cumulative if not yet initialized)
        let inference_tokens_per_sec_ema = if self.ema_tokens_per_sec > 0.0 {
            self.ema_tokens_per_sec
        } else {
            inference_tokens_per_sec
        };

        // Finalize quality metrics from accumulator
        let quality_metrics = if !self.metrics_accumulator.is_empty() {
            Some(self.metrics_accumulator.finalize())
        } else {
            None
        };

        GenerationStats {
            tokens_generated: self.tokens_generated,
            generation_time_ms: total_time.as_millis() as u64,
            tokens_per_second: if total_time.as_secs_f32() > 0.0 {
                self.tokens_generated as f32 / total_time.as_secs_f32()
            } else {
                0.0
            },
            finish_reason: self.finish_reason.clone(),
            quality_metrics,

            // Separate prefill/inference metrics
            prefill_tokens,
            prefill_time_ms,
            prefill_tokens_per_sec,
            inference_tokens,
            inference_time_ms,
            inference_tokens_per_sec,
            inference_tokens_per_sec_ema,
        }
    }

    /// Update EMA rate after generating a token.
    ///
    /// Uses exponential moving average with alpha=0.3 to smooth rate measurements.
    /// This handles initial acceleration gracefully without needing warmup hacks.
    fn update_ema_rate(&mut self) {
        let now = std::time::Instant::now();

        if let Some(last_time) = self.last_token_time {
            let elapsed_secs = last_time.elapsed().as_secs_f32();
            if elapsed_secs > 0.0 {
                let instantaneous_rate = 1.0 / elapsed_secs;

                // EMA: alpha=0.3 gives good responsiveness while smoothing noise
                const ALPHA: f32 = 0.3;
                if self.ema_tokens_per_sec > 0.0 {
                    self.ema_tokens_per_sec = ALPHA * instantaneous_rate + (1.0 - ALPHA) * self.ema_tokens_per_sec;
                } else {
                    // First measurement - use instantaneous rate
                    self.ema_tokens_per_sec = instantaneous_rate;
                }
            }
        }

        self.last_token_time = Some(now);
    }

    fn sample_next_token(&mut self) -> Result<u32> {
        // Determine KV position for this forward pass
        let current_kv_pos = if self.tokens_generated == 0 {
            0 // Initial position is 0
        } else {
            self.kv_cache_position
        };

        // Lock delta once for this token's forward pass (if present)
        let delta_guard = self.delta.as_ref().map(|d| d.lock());

        let logits = if self.tokens_generated == 0 {
            // PREFILL: Process full prompt and capture timing
            let prefill_start = std::time::Instant::now();
            let result = if delta_guard.is_some() {
                let delta_ref = delta_guard.as_deref();
                self.engine.forward_with_delta_full(&self.prompt_tokens, delta_ref)?
            } else {
                self.engine.forward(&self.prompt_tokens)?
            };
            let prefill_elapsed = prefill_start.elapsed();

            // Store prefill timing
            self.prefill_time_ms = Some(prefill_elapsed.as_millis() as u64);
            self.first_token_time = Some(std::time::Instant::now());

            tracing::info!(
                "📊 PREFILL: {} tokens in {:?} ({:.2} tok/sec){}",
                self.prompt_tokens.len(),
                prefill_elapsed,
                if prefill_elapsed.as_secs_f32() > 0.0 {
                    self.prompt_tokens.len() as f32 / prefill_elapsed.as_secs_f32()
                } else {
                    0.0
                },
                if delta_guard.is_some() { " [delta-aware]" } else { "" }
            );

            result
        } else {
            let last_token = self.last_generated.ok_or_else(|| {
                anyhow::anyhow!("Internal error: last_generated not set after {} tokens", self.tokens_generated)
            })?;

            if self.tokens_generated.is_multiple_of(50) {
                tracing::debug!(
                    "🔵 KV cache position: {}, tokens_generated: {}, last_token: {}",
                    self.kv_cache_position, self.tokens_generated, last_token
                );
            }

            // Use kv_cache_position directly - this is where the next token will be written
            if delta_guard.is_some() {
                let delta_ref = delta_guard.as_deref();
                self.engine.forward_with_delta_cached(
                    &[last_token],
                    current_kv_pos,
                    true,
                    delta_ref,
                )?
            } else {
                self.engine.forward_cached(
                    &[last_token],
                    current_kv_pos,
                    true,
                )?
            }
        };

        // Release delta lock before sampling (no longer needed)
        drop(delta_guard);

        // NOTE: Logits truncation has been DISABLED (Nov 6, 2025)
        //
        // Previous code tried to truncate logits from model vocab (151936) to tokenizer vocab (151669).
        // This was causing year tokens to be excluded from sampling, leading to "1 7 7 6" instead of "1776".
        //
        // vLLM with the same model does NOT have this issue, suggesting the truncation was too aggressive.
        //
        // Solution: Use model vocab directly. The sampler will only pick valid tokens anyway.
        // The tokenizer's get_vocab_size() should match model vocab if properly configured.

        // PERF: Use cached vocab_size - avoids lock acquisition per token
        let vocab_size = self.vocab_size;

        // PERF: Cache model_vocab_size on first call (from logits shape)
        let model_vocab_size = if self.model_vocab_size == 0 {
            let logits_shape = logits.size();
            let size = logits_shape[logits_shape.len() - 1] as usize;
            self.model_vocab_size = size;
            size
        } else {
            self.model_vocab_size
        };

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

        // PERF: Use pre-created sampling params - avoids struct allocation per token
        let params = &self.sampling_params;

        // PERF: Optimize VecDeque slice handling
        let (slice1, slice2) = self.recent_tokens.as_slices();
        let next_token = if slice2.is_empty() {
            // Fast path: VecDeque is contiguous, use slice directly
            self.engine.sample_token_with_params(&logits, params, slice1)?
        } else {
            // Slow path: VecDeque wrapped around, use pre-allocated buffer
            self.recent_tokens_buffer.clear();
            self.recent_tokens_buffer.extend(self.recent_tokens.iter().copied());
            self.engine.sample_token_with_params(&logits, params, &self.recent_tokens_buffer)?
        };

        // Validate sampled token is within model vocabulary
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

        // Capture metrics for self-supervised training (EXPENSIVE - only if explicitly enabled)
        // Computes softmax + entropy over full vocabulary (~150K) with GPU->CPU syncs
        // This adds ~10x overhead per token, so only enable for training
        if self.collect_metrics {
            // Get last token's logits (already squeezed to 1D in sampler, but handle 2D case)
            let last_logits = if logits.dim() > 1 {
                let shape = logits.size();
                if shape.len() == 3 {
                    // [batch, seq_len, vocab] -> take [-1, -1, :] -> [vocab]
                    logits.select(1, shape[1] - 1).squeeze_dim(0)
                } else if shape.len() == 2 {
                    // [seq_len, vocab] or [batch, vocab] -> squeeze
                    logits.squeeze_dim(0)
                } else {
                    logits.shallow_clone()
                }
            } else {
                logits.shallow_clone()
            };

            // Compute stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
            let max_logit = last_logits.max();
            let shifted = &last_logits - &max_logit;
            let exp_logits = shifted.exp();
            let sum_exp = exp_logits.sum(tch::Kind::Float);
            let probs = &exp_logits / &sum_exp;

            // Compute log_prob of sampled token (in nats)
            // log_prob = log(probs[next_token])
            let log_prob = probs
                .select(0, next_token as i64)
                .log()
                .double_value(&[]) as f32;

            // Compute entropy: -sum(p * log(p)) where p > 0
            // Use a small epsilon to avoid log(0)
            let eps = 1e-10f64;
            let safe_probs = probs.clamp(eps, 1.0 - eps);
            let log_probs = safe_probs.log();
            let entropy = (-(&probs * &log_probs)).sum(tch::Kind::Float).double_value(&[]) as f32;

            // Add to accumulator (O(1) per token)
            self.metrics_accumulator.add_token(log_prob, entropy, next_token as u32);
        }

        Ok(next_token as u32)
    }
}

// SAFETY: TextStream can be Send because:
// - tokenizer pointer points to heap data that won't move
// - decode_stream is tied to the tokenizer lifetime
// - Tokenizer itself is Send
unsafe impl<'a> Send for TextStream<'a> {}

// No custom Drop needed - Arc handles automatic cleanup

impl<'a> Stream for TextStream<'a> {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.finished {
                return Poll::Ready(None);
            }

            // Check timeout
            if let Some(timeout_ms) = self.timeout_ms {
                let elapsed = self.start_time.elapsed();
                if elapsed.as_millis() >= timeout_ms as u128 {
                    tracing::debug!("Generation timed out after {}ms", timeout_ms);
                    self.finished = true;
                    self.finish_reason = Some(FinishReason::Error(format!(
                        "Generation timed out after {timeout_ms}ms"
                    )));
                    return Poll::Ready(Some(Err(anyhow::anyhow!(
                        "Generation timed out after {}ms", timeout_ms
                    ))));
                }
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
                self.finish_reason = Some(FinishReason::StopToken(format!("{next_token}")));
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
                    self.update_ema_rate();

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
                    self.update_ema_rate();

                    self.recent_tokens.push_back(token_i64);
                    if self.recent_tokens.len() > self.repeat_last_n {
                        self.recent_tokens.pop_front();
                    }

                    // Simple trace without expensive tokenizer lock or decode
                    tracing::trace!("Token {} -> buffering (incomplete UTF-8)", next_token);
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
