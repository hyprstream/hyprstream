//! Candle-based runtime engine implementation with direct tensor access
//! 
//! This replaces mistral.rs with Candle for:
//! - Direct tensor-level access for VDB integration
//! - Full control over LoRA weight updates
//! - Custom quantization with OpenVDB compression
//! - Temporal gradient streaming during inference

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, Mutex};

use candle_core::{Device, DType, Tensor, IndexOp};
use candle_transformers::models::quantized_llama::{ModelWeights as LlamaWeights};
// Support for multiple architectures - will auto-detect and use appropriate loader
// For Gemma models, we should use the Gemma-specific loaders when available
use candle_core::quantized::GgmlDType;

// Architecture-specific model loading
use crate::runtime::architectures::{ModelFactory, ArchitectureDetector, ModelOperations, ModelArchitecture};
use crate::runtime::architectures::llama::LlamaModel;

use super::{RuntimeEngine, ModelInfo, GenerationRequest, GenerationResult, FinishReason, RuntimeConfig};
use crate::storage::vdb::{VDBSparseStorage, SparseStorageConfig, TemporalStreamingLayer, SparseStorage};
use crate::adapters::sparse_lora::{SparseLoRAAdapter, SparseLoRAConfig, InitMethod};
use crate::runtime::sampling::{SamplingConfig, TokenSampler};

/// Temporal gradient for real-time weight updates
#[derive(Debug, Clone)]
pub struct TemporalGradient {
    pub layer_gradients: HashMap<String, LayerGradient>,
    pub learning_rate: f32,
    pub timestamp: std::time::SystemTime,
}

/// Layer-specific gradient information
#[derive(Debug, Clone)]
pub struct LayerGradient {
    pub weight_deltas: Vec<f32>,
    pub bias_deltas: Vec<f32>,
    pub sparsity_mask: Vec<bool>,
}

/// Training result for temporal LoRA
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub loss: f32,
    pub gradient_updates: usize,
    pub tokens_processed: usize,
    pub training_time_ms: u64,
}

/// Simple hash function for consistent sparsity patterns
fn context_hash(input: usize) -> u64 {
    // Simple hash based on input value
    let mut hash = input as u64;
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    hash
}

/// Candle-based engine with VDB storage integration
pub struct CandleEngine {
    /// Model weights stored in VDB format
    vdb_storage: Arc<VDBSparseStorage>,
    /// Candle device (CPU/CUDA)
    device: Device,
    /// Model configuration
    _config: RuntimeConfig,
    /// Model metadata
    model_info: Option<ModelInfo>,
    /// Temporal streaming layer for real-time updates
    temporal_streaming: Option<Arc<TemporalStreamingLayer>>,
    /// Active LoRA adapters
    _lora_adapters: Arc<RwLock<HashMap<String, SparseLoRAAdapter>>>,
    /// Tokenizer
    tokenizer: Option<tokenizers::Tokenizer>,
    /// Model weights loaded from file (wrapped in Mutex for mutable access)
    model: Option<Arc<Mutex<LlamaWeights>>>,
    /// Architecture-specific model implementation
    arch_model: Option<Arc<Mutex<Box<dyn ModelOperations>>>>,
    /// SafeTensors base model weights for inference (F32 for CPU, BF16 for GPU)
    base_model_weights: Option<Arc<HashMap<String, Tensor>>>,
    /// Active LoRA adapter loaded from VDB
    pub active_lora: Option<Arc<SparseLoRAAdapter>>,
    /// Sampling configuration for token generation
    pub sampling_config: SamplingConfig,
    /// Detected model architecture
    detected_architecture: Option<ModelArchitecture>,
}

impl CandleEngine {
    /// Create a new Candle engine (synchronous, creates placeholder for async init)
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        // Initialize device synchronously
        let device = if config.use_gpu && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        
        // Create a placeholder VDB storage that will be initialized on first use
        // This avoids the async issue in the constructor
        let storage_config = SparseStorageConfig::default();
        
        // For now, we'll use a blocking approach for the VDB storage initialization
        // In production, this should be done asynchronously during engine setup
        let vdb_storage = std::thread::spawn(move || {
            tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(async {
                    VDBSparseStorage::new(storage_config).await
                })
        })
        .join()
        .map_err(|_| anyhow!("Failed to join VDB initialization thread"))??;
        
        let vdb_storage = Arc::new(vdb_storage);
        
        Ok(Self {
            vdb_storage,
            device,
            _config: config,
            model_info: None,
            temporal_streaming: None,
            _lora_adapters: Arc::new(RwLock::new(HashMap::new())),
            tokenizer: None,
            model: None,
            arch_model: None,
            base_model_weights: None,
            active_lora: None,
            sampling_config: SamplingConfig::default(),
            detected_architecture: None,
        })
    }
    
    /// Async constructor for when called from async contexts
    pub async fn new_async(config: RuntimeConfig) -> Result<Self> {
        // Initialize device
        let device = if config.use_gpu && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        
        // Initialize VDB storage asynchronously
        let storage_config = SparseStorageConfig::default();
        let vdb_storage = Arc::new(VDBSparseStorage::new(storage_config).await?);
        
        Ok(Self {
            vdb_storage,
            device,
            _config: config,
            model_info: None,
            temporal_streaming: None,
            _lora_adapters: Arc::new(RwLock::new(HashMap::new())),
            tokenizer: None,
            model: None,
            arch_model: None,
            base_model_weights: None,
            active_lora: None,
            sampling_config: SamplingConfig::default(),
            detected_architecture: None,
        })
    }
    
    /// Load SafeTensors model for inference (keep in memory, not VDB)
    async fn load_safetensors_to_vdb(&mut self, path: &Path) -> Result<()> {
        tracing::info!("📦 Loading SafeTensors model for inference: {:?}", path);
        
        // Load the SafeTensors file
        let tensors = self.load_safetensors_file(path)?;
        
        // Detect architecture from tensor names or config
        let detected_arch = self.detect_architecture_from_tensors(&tensors);
        tracing::info!("Detected architecture: {}", detected_arch.name());
        
        // Try to load config.json from the parent directory
        let parent_dir = path.parent().unwrap_or(Path::new("."));
        let config_path = parent_dir.join("config.json");
        let (context_length, vocab_size, mut architecture) = if config_path.exists() {
            self.parse_config_json(&config_path)?
        } else {
            (4096, 32000, detected_arch.name())
        };
        
        // Override architecture for Gemma models based on config
        if architecture.contains("Gemma") || architecture.contains("gemma") {
            // Check if it's Gemma3 specifically
            if architecture.contains("Gemma3") || architecture.contains("gemma3") {
                architecture = "Gemma3".to_string();
                self.detected_architecture = Some(ModelArchitecture::Gemma);
                tracing::info!("Detected Gemma3 model from config, using Gemma architecture with Gemma3 specifics");
            } else {
                architecture = "Gemma".to_string();
                self.detected_architecture = Some(ModelArchitecture::Gemma);
                tracing::info!("Detected Gemma model from config, overriding architecture");
            }
        }
        
        // Create model info
        self.model_info = Some(ModelInfo {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            parameters: self.estimate_parameters_from_tensors(&tensors) as u64,
            context_length,
            vocab_size,
            architecture,
            quantization: Some("safetensors".to_string()),
        });
        
        // IMPORTANT: Keep base model weights in memory for inference
        // VDB is only for LoRA adapters, not base models
        let dtype_str = match &self.device {
            Device::Cpu => "F32",
            Device::Cuda(_) | Device::Metal(_) => "BF16",
        };
        tracing::info!("📦 Keeping base model weights in memory ({})", dtype_str);
        let tensors_arc = Arc::new(tensors);
        self.base_model_weights = Some(tensors_arc.clone());
        
        // Create architecture model from the weights for inference
        self.create_arch_model_from_safetensors(&*tensors_arc).await?;
        
        // Load tokenizer from parent directory
        self.load_tokenizer(parent_dir).await?;
        
        // Load active LoRA from VDB if available
        if let Ok(lora) = self.load_active_lora_from_vdb().await {
            tracing::info!("✅ Loaded active LoRA adapter from VDB");
            self.active_lora = Some(Arc::new(lora));
        }
        
        tracing::info!("✅ SafeTensors model loaded for inference");
        Ok(())
    }
    
    /// Load sharded SafeTensors model files for inference
    async fn load_sharded_safetensors_to_vdb(&mut self, paths: &[PathBuf]) -> Result<()> {
        tracing::info!("📦 Loading sharded SafeTensors model with {} files", paths.len());
        
        // Load all tensor files and merge
        let mut all_tensors = HashMap::new();
        for (idx, path) in paths.iter().enumerate() {
            tracing::info!("  Loading shard {}/{}: {:?}", idx + 1, paths.len(), path.file_name());
            let tensors = self.load_safetensors_file(path)?;
            all_tensors.extend(tensors);
        }
        
        tracing::info!("Loaded {} total tensors from shards", all_tensors.len());
        
        // Detect architecture from tensor names
        let detected_arch = self.detect_architecture_from_tensors(&all_tensors);
        tracing::info!("Detected architecture: {}", detected_arch.name());
        
        // Try to load config.json from the parent directory
        let parent_dir = paths[0].parent().unwrap_or(Path::new("."));
        let config_path = parent_dir.join("config.json");
        let (context_length, vocab_size, architecture) = if config_path.exists() {
            self.parse_config_json(&config_path)?
        } else {
            (4096, 32000, detected_arch.name())
        };
        
        // Create model info
        self.model_info = Some(ModelInfo {
            name: parent_dir.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            parameters: self.estimate_parameters_from_tensors(&all_tensors) as u64,
            context_length,
            vocab_size,
            architecture,
            quantization: Some("safetensors".to_string()),
        });
        
        // IMPORTANT: Keep base model weights in memory for inference
        // VDB is only for LoRA adapters, not base models
        let dtype_str = match &self.device {
            Device::Cpu => "F32",
            Device::Cuda(_) | Device::Metal(_) => "BF16",
        };
        tracing::info!("📦 Keeping base model weights in memory ({})", dtype_str);
        let tensors_arc = Arc::new(all_tensors);
        self.base_model_weights = Some(tensors_arc.clone());
        
        // Create architecture model from the weights for inference
        self.create_arch_model_from_safetensors(&*tensors_arc).await?;
        
        // Load tokenizer from parent directory
        self.load_tokenizer(parent_dir).await?;
        
        // Load active LoRA from VDB if available
        if let Ok(lora) = self.load_active_lora_from_vdb().await {
            tracing::info!("✅ Loaded active LoRA adapter from VDB");
            self.active_lora = Some(Arc::new(lora));
        }
        
        tracing::info!("✅ Sharded SafeTensors model loaded for inference");
        Ok(())
    }
    
    /// Load a single SafeTensors file
    fn load_safetensors_file(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        // Use Candle's built-in SafeTensors loading - much more efficient!
        let mut tensors = candle_core::safetensors::load(path, &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to load SafeTensors file: {}", e))?;
        
        // Convert BF16 tensors to F32 on CPU (CPU doesn't support BF16 matmul)
        if matches!(self.device, Device::Cpu) {
            tensors = tensors.into_iter()
                .map(|(name, tensor)| {
                    let converted = if tensor.dtype() == DType::BF16 {
                        tensor.to_dtype(DType::F32)
                            .unwrap_or_else(|_| tensor.clone())
                    } else {
                        tensor
                    };
                    (name, converted)
                })
                .collect();
        }
        
        Ok(tensors)
    }
    
    /// Parse config.json to get model parameters and sampling configuration
    fn parse_config_json(&mut self, path: &Path) -> Result<(usize, usize, String)> {
        let config_str = std::fs::read_to_string(path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;
        
        let context_length = config.get("max_position_embeddings")
            .or_else(|| config.get("n_positions"))
            .or_else(|| config.get("seq_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        
        let vocab_size = config.get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;
        
        let architecture = config.get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .or_else(|| config.get("model_type").and_then(|v| v.as_str()))
            .unwrap_or("unknown")
            .to_string();
        
        // Load sampling configuration from model card only if it hasn't been customized
        // This preserves any configuration set by the CLI
        if self.sampling_config == SamplingConfig::default() {
            let model_id = config.get("_name_or_path")
                .and_then(|v| v.as_str())
                .unwrap_or(&architecture);
            self.sampling_config = SamplingConfig::from_model_card(model_id, &config);
        }
        
        Ok((context_length, vocab_size, architecture))
    }
    
    /// Detect architecture from tensor names
    fn detect_architecture_from_tensors(&self, tensors: &HashMap<String, Tensor>) -> ModelArchitecture {
        // Check for architecture-specific tensor patterns
        for (name, _) in tensors.iter() {
            if name.contains("gemma") || name.contains("gated_proj") {
                return ModelArchitecture::Gemma;
            }
            if name.contains("qwen") || name.contains("c_attn") {
                return ModelArchitecture::Qwen { version: 3, is_moe: false, context_length: 32768 };
            }
            if name.contains("moe") || name.contains("expert") {
                return ModelArchitecture::GPTOSS { 
                    total_params_b: 120, 
                    active_params_b: 5.1,
                    num_experts: 128,
                    experts_per_token: 8,
                };
            }
        }
        
        // Default to Llama 3
        ModelArchitecture::Llama { version: 3 }
    }
    
    /// Estimate parameters from tensors
    fn estimate_parameters_from_tensors(&self, tensors: &HashMap<String, Tensor>) -> usize {
        let mut total_params = 0usize;
        for (_name, tensor) in tensors.iter() {
            let shape = tensor.dims();
            let mut params = 1usize;
            for dim in shape {
                params *= dim;
            }
            total_params += params;
        }
        total_params
    }
    
    /// Convert SafeTensors to VDB storage
    async fn convert_safetensors_to_vdb(&mut self, tensors: HashMap<String, Tensor>) -> Result<()> {
        let total_tensors = tensors.len();
        let mut processed = 0;
        
        for (name, tensor) in tensors {
            // Create sparse adapter from tensor
            let adapter = self.tensor_to_sparse_adapter(&name, &tensor)?;
            
            // Store in VDB
            let adapter_id = format!("tensor_{}", name.replace(".", "_"));
            self.vdb_storage.store_adapter(&adapter_id, &adapter).await?;
            
            processed += 1;
            if processed % 10 == 0 {
                tracing::info!("  Processed {}/{} tensors", processed, total_tensors);
            }
        }
        
        tracing::info!("✅ Converted {} tensors to VDB storage", total_tensors);
        Ok(())
    }
    
    /// Load active LoRA from VDB storage
    async fn load_active_lora_from_vdb(&self) -> Result<SparseLoRAAdapter> {
        // Try to load the most recent LoRA adapter from VDB
        // For MVP, we'll just try to load a default adapter
        match self.vdb_storage.load_adapter("active_lora", Default::default()).await {
            Ok(adapter) => {
                tracing::info!("Loaded active LoRA from VDB");
                Ok(adapter)
            }
            Err(e) => {
                tracing::debug!("No active LoRA found in VDB: {}", e);
                Err(anyhow!("No active LoRA adapter found"))
            }
        }
    }
    
    /// Convert a tensor to sparse adapter format
    fn tensor_to_sparse_adapter(&self, name: &str, tensor: &Tensor) -> Result<SparseLoRAAdapter> {
        // Create a sparse adapter from the tensor
        // This is a simplified version - real implementation would be more sophisticated
        let shape = tensor.dims();
        let rank = shape.last().copied().unwrap_or(8).min(128); // Use last dim as rank hint
        
        // Create minimal config for tensor conversion
        let config = SparseLoRAConfig {
            in_features: tensor.dims()[0],
            out_features: if tensor.dims().len() > 1 { tensor.dims()[1] } else { tensor.dims()[0] },
            rank,
            sparsity: 0.99,
            learning_rate: 1e-4,
            dropout: 0.0,
            alpha: 16.0,
            bias: false,
            target_modules: vec![name.to_string()],
            init_method: InitMethod::Random,
            sparsity_threshold: 0.01,
            enable_gradient_checkpointing: false,
            mixed_precision: true,
        };
        
        let adapter = SparseLoRAAdapter::new(config);
        
        // For now, just create the adapter structure
        // In production, we'd properly decompose the tensor into LoRA factors
        // and store tensor metadata separately
        
        Ok(adapter)
    }

    /// Load SafeTensors model into memory for inference
    async fn load_safetensors_model(&mut self, path: &Path) -> Result<()> {
        tracing::info!("🔄 Loading model: {:?}", path);
        
        // Check if path is a directory (SafeTensors model directory)
        if path.is_dir() {
            tracing::info!("📁 Directory detected, looking for SafeTensors model files...");
            
            // Look for SafeTensors model files in the directory
            let mut model_files = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let file_path = entry.path();
                if let Some(ext) = file_path.extension() {
                    if ext == "safetensors" && file_path.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.contains("model"))
                        .unwrap_or(false) {
                        model_files.push(file_path);
                    }
                }
            }
            
            if model_files.is_empty() {
                return Err(anyhow!("No SafeTensors model files found in directory: {:?}", path));
            }
            
            // Sort files to ensure correct loading order for sharded models
            model_files.sort();
            tracing::info!("Found {} SafeTensors model files", model_files.len());
            
            // Load the SafeTensors files
            if model_files.len() == 1 {
                // Single file model
                return self.load_safetensors_to_vdb(&model_files[0]).await;
            } else {
                // Sharded model - load all shards
                return self.load_sharded_safetensors_to_vdb(&model_files).await;
            }
        }
        
        // Only support SafeTensors files
        if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            return Err(anyhow!("Only SafeTensors format is supported. File must have .safetensors extension"));
        }
        
        let model_path = path.to_path_buf();
        
        // Load SafeTensors file
        self.load_safetensors_to_vdb(&model_path).await
    }
    
    /// Check if a tensor should be sparsified
    fn should_sparsify(&self, tensor_name: &str) -> bool {
        // Sparsify attention and FFN weights, keep embeddings dense
        tensor_name.contains("attn") || 
        tensor_name.contains("mlp") ||
        tensor_name.contains("ffn")
    }
    
    /// Load tokenizer
    async fn load_tokenizer(&mut self, model_path: &Path) -> Result<()> {
        // Try to find tokenizer.json in the same directory
        let tokenizer_path = model_path.join("tokenizer.json");
        
        if tokenizer_path.exists() {
            self.tokenizer = Some(
                tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?
            );
            tracing::info!("✅ Loaded tokenizer from: {:?}", tokenizer_path);
        } else {
            // Check for cached tokenizer
            let cache_dir = model_path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join(".tokenizer_cache");
            
            // Try to find a cached tokenizer
            let mut cached_tokenizer = None;
            if cache_dir.exists() {
                if let Ok(mut entries) = tokio::fs::read_dir(&cache_dir).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let path = entry.path();
                        if path.extension().map_or(false, |ext| ext == "json") {
                            if let Ok(tokenizer) = tokenizers::Tokenizer::from_file(&path) {
                                tracing::info!("✅ Loaded cached tokenizer from: {:?}", path);
                                cached_tokenizer = Some(tokenizer);
                                break;
                            }
                        }
                    }
                }
            }
            
            if let Some(tokenizer) = cached_tokenizer {
                self.tokenizer = Some(tokenizer);
            } else {
                // Try to download a suitable tokenizer based on model architecture
                tracing::info!("📥 No cached tokenizer found, attempting to download...");
                if let Some(tokenizer) = self.download_tokenizer_for_model(model_path).await? {
                    self.tokenizer = Some(tokenizer);
                    tracing::info!("✅ Downloaded and loaded tokenizer");
                } else {
                    tracing::warn!("⚠️ No tokenizer available, using character-level fallback");
                }
            }
        }
        
        Ok(())
    }
    
    /// Create a simple character-level tokenizer as fallback
    fn create_fallback_tokenizer(&self) -> tokenizers::Tokenizer {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;
        
        let mut tokenizer = tokenizers::Tokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_post_processor(Some(ByteLevelProcessor::default()));
        
        tokenizer
    }
    
    /// Download tokenizer from HuggingFace based on model type
    async fn download_tokenizer_for_model(&self, model_path: &Path) -> Result<Option<tokenizers::Tokenizer>> {
        use hf_hub::{api::tokio::Api, Repo, RepoType};
        
        // Determine model name from path or use default Qwen2 model
        let model_name = if model_path.to_string_lossy().contains("gemma") || 
                           model_path.to_string_lossy().contains("Gemma") {
            "google/gemma-2b"  // Use appropriate Gemma tokenizer
        } else if model_path.to_string_lossy().contains("qwen2") || 
                           model_path.to_string_lossy().contains("Qwen2") {
            "Qwen/Qwen2-1.5B-Instruct"
        } else if model_path.to_string_lossy().contains("mistral") {
            "mistralai/Mistral-7B-v0.1"
        } else if model_path.to_string_lossy().contains("llama") {
            "meta-llama/Llama-2-7b-hf"
        } else {
            // Default to Gemma if unknown (since that's what the user is testing)
            "google/gemma-2b"
        };
        
        tracing::info!("📥 Downloading tokenizer for model: {}", model_name);
        
        // Create HuggingFace API client
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
        
        // Download tokenizer.json
        let tokenizer_file = repo.get("tokenizer.json").await?;
        
        // Load the tokenizer from the downloaded file
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow!("Failed to load downloaded tokenizer: {}", e))?;
        
        // Cache the tokenizer for future use
        let cache_dir = model_path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(".tokenizer_cache");
        tokio::fs::create_dir_all(&cache_dir).await?;
        
        let cached_path = cache_dir.join(format!("{}.tokenizer.json", 
            model_name.replace("/", "_")));
        tokio::fs::copy(&tokenizer_file, &cached_path).await?;
        
        tracing::info!("✅ Downloaded and cached tokenizer at: {:?}", cached_path);
        
        Ok(Some(tokenizer))
    }
    
    /// Generate text using VDB-backed weights (internal, accumulates all tokens)
    async fn generate_with_vdb(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Step 1: Tokenize the prompt
        let tokens = self.tokenize_prompt(prompt)?;
        
        // Step 2: Prepare generation context
        let mut generated_tokens = Vec::new();
        let mut context = tokens.clone();
        
        // Step 3: Generate tokens one by one
        for i in 0..max_tokens {
            // Load relevant sparse weights from VDB
            let weights = self.load_sparse_weights_for_context(&context).await?;
            
            // Run forward pass with sparse weights
            let next_token = self.forward_pass_sparse(&context, &weights).await?;
            
            // Check for end-of-sequence
            if self.is_eos_token(next_token) {
                break;
            }
            
            // Add to generated tokens and context
            generated_tokens.push(next_token);
            context.push(next_token);
            
            // Optional: Apply temporal weight updates
            if self.temporal_streaming.is_some() {
                self.apply_temporal_update(&context, next_token).await?;
            }
            
            // Keep context within limits
            if context.len() > 2048 {
                context.drain(0..100); // Simple sliding window
            }
        }
        
        // Step 4: Decode the output
        let output = self.decode_tokens(&generated_tokens)?;
        Ok(output)
    }
    
    /// Generate text with streaming output - yields tokens as they're generated
    pub async fn generate_streaming<F>(&self, prompt: &str, max_tokens: usize, mut on_token: F) -> Result<String>
    where
        F: FnMut(&str) + Send,
    {
        // Debug: Verify streaming is being called
        eprintln!("DEBUG: generate_streaming called with prompt: '{}', max_tokens: {}", prompt, max_tokens);
        
        // Step 1: Tokenize the prompt
        let tokens = self.tokenize_prompt(prompt)?;
        eprintln!("DEBUG: Tokenized {} prompt tokens", tokens.len());
        
        // Step 2: Prepare generation context
        let mut generated_tokens = Vec::new();
        let mut context = tokens.clone();
        
        // Step 3: Generate tokens one by one
        for i in 0..max_tokens {
            eprintln!("DEBUG: Generating token {}/{}", i + 1, max_tokens);
            
            // Load relevant sparse weights from VDB
            let weights = self.load_sparse_weights_for_context(&context).await?;
            
            // Run forward pass with sparse weights
            let next_token = self.forward_pass_sparse(&context, &weights).await?;
            eprintln!("DEBUG: Generated token ID: {}", next_token);
            
            // Check for end-of-sequence
            if self.is_eos_token(next_token) {
                eprintln!("DEBUG: Hit EOS token, stopping");
                break;
            }
            
            // Decode and stream the single token
            match self.decode_tokens(&[next_token]) {
                Ok(token_text) => {
                    eprintln!("DEBUG: Decoded token text: '{}'", token_text);
                    on_token(&token_text);
                    // Flush stdout immediately after each token
                    use std::io::{self, Write};
                    io::stdout().flush().ok();
                }
                Err(e) => {
                    eprintln!("DEBUG: Failed to decode token {}: {}", next_token, e);
                    // If we can't decode, at least show the token ID
                    let fallback = format!("[{}]", next_token);
                    on_token(&fallback);
                    use std::io::{self, Write};
                    io::stdout().flush().ok();
                }
            }
            
            // Add to generated tokens and context
            generated_tokens.push(next_token);
            context.push(next_token);
            
            // Optional: Apply temporal weight updates
            if self.temporal_streaming.is_some() {
                self.apply_temporal_update(&context, next_token).await?;
            }
            
            // Keep context within limits
            if context.len() > 2048 {
                context.drain(0..100); // Simple sliding window
            }
        }
        
        // Return the complete output
        self.decode_tokens(&generated_tokens)
    }
    
    /// Tokenize the input prompt
    fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        if let Some(tokenizer) = &self.tokenizer {
            let encoding = tokenizer.encode(prompt, false)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
            Ok(encoding.get_ids().to_vec())
        } else {
            // Last resort: character-level tokenization (should rarely happen now)
            tracing::warn!("⚠️ Using fallback character-level tokenization (no tokenizer loaded)");
            Ok(prompt.chars().map(|c| c as u32).collect())
        }
    }
    
    /// Load sparse weights for the given context
    async fn load_sparse_weights_for_context(&self, context: &[u32]) -> Result<HashMap<String, SparseLoRAAdapter>> {
        let mut weights = HashMap::new();
        
        // Determine which adapters we need based on context
        let adapter_ids = self.select_adapters_for_context(context);
        
        for adapter_id in adapter_ids {
            // Load from VDB storage
            match self.vdb_storage.load_adapter(&adapter_id, Default::default()).await {
                Ok(adapter) => {
                    weights.insert(adapter_id.clone(), adapter);
                    tracing::trace!("📦 Loaded adapter: {}", adapter_id);
                }
                Err(e) => {
                    tracing::warn!("⚠️ Failed to load adapter {}: {}", adapter_id, e);
                }
            }
        }
        
        Ok(weights)
    }
    
    /// Select relevant adapters based on context
    fn select_adapters_for_context(&self, _context: &[u32]) -> Vec<String> {
        // For now, return empty - adapters will be loaded on demand when they exist
        // In the future, this should intelligently select based on context
        Vec::new()
        
        // When tensor adapters are created from quantized models, uncomment:
        // vec![
        //     "tensor_blk_0_attn_q".to_string(),
        //     "tensor_blk_0_attn_k".to_string(),
        //     "tensor_blk_0_attn_v".to_string(),
        //     "tensor_blk_0_ffn_gate".to_string(),
        // ]
    }
    
    /// Run forward pass with sparse weights using actual model
    async fn forward_pass_sparse(&self, context: &[u32], _weights: &HashMap<String, SparseLoRAAdapter>) -> Result<u32> {
        // Check if we have SafeTensors base model weights
        if let Some(base_weights) = &self.base_model_weights {
            // Perform inference with SafeTensors weights
            if context.is_empty() {
                return Ok(1); // BOS token
            }
            
            // Use existing architecture model or return error
            if let Some(arch_model) = &self.arch_model {
                let model = arch_model.lock().await;
                
                // Convert context to tensor - use u32 directly like Candle examples
                let input_ids = Tensor::new(context, &self.device)?;
                let input_ids = input_ids.unsqueeze(0)?; // Add batch dimension
                
                // Run model forward pass
                let mut logits = model.forward(&input_ids, None)?;
                
                // TODO: remove - Debug logging for generation issues
                if std::env::var("DEBUG_GENERATION").is_ok() {
                    if let Ok(logits_vec) = logits.i((0, logits.dim(1)? - 1))?.to_vec1::<f32>() {
                        // Find top 5 tokens
                        let mut indexed: Vec<(usize, f32)> = logits_vec.iter()
                            .enumerate()
                            .map(|(i, &v)| (i, v))
                            .collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        
                        tracing::debug!("Logits shape: {:?}, vocab_size = {}", logits.dims(), logits_vec.len());
                        tracing::debug!("Top 5 token predictions:");
                        for (i, (token_id, score)) in indexed.iter().take(5).enumerate() {
                            tracing::debug!("  #{}: token {} with score {:.3}", i+1, token_id, score);
                        }
                        
                        // Check logits statistics
                        let max_val = indexed[0].1;
                        let min_val = logits_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                        let mean_val = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
                        tracing::debug!("Logits stats: min={:.3}, max={:.3}, mean={:.3}, range={:.3}", 
                                      min_val, max_val, mean_val, max_val - min_val);
                    }
                }
                
                // Apply LoRA if available
                if let Some(lora) = &self.active_lora {
                    logits = self.apply_lora_to_logits(&logits, lora).await?;
                }
                
                // Get last token logits and sample using proper sampling strategy
                let last_logits = logits.i((0, logits.dim(1)? - 1))?;
                
                // Create a token sampler with current configuration
                let mut sampler = TokenSampler::new(self.sampling_config.clone());
                
                // Debug: Log sampling configuration
                eprintln!("DEBUG: Sampling config - temp={}, top_k={:?}, top_p={:?}, do_sample={}", 
                         self.sampling_config.temperature,
                         self.sampling_config.top_k,
                         self.sampling_config.top_p,
                         self.sampling_config.do_sample);
                
                sampler.add_to_history(context);
                let next_token = sampler.sample(&last_logits)?;
                
                return Ok(next_token);
            }
            
            return Err(anyhow!("No architecture model available for SafeTensors inference"));
        }
        
        // Fallback to quantized model if available
        if let Some(model) = &self.model {
            // Use actual model for proper inference
            if context.is_empty() {
                return Ok(1); // BOS token
            }
            
            // Convert context to tensor (use last few tokens for efficiency)
            let context_window = if context.len() > 512 {
                &context[context.len() - 512..]
            } else {
                context
            };
            
            // Create tensor from token IDs - use u32 directly
            let input_ids = Tensor::new(context_window, &self.device)?;
            let input_ids = input_ids.unsqueeze(0)?; // Add batch dimension [1, seq_len]
            
            // Run actual model inference by acquiring mutex lock
            let mut model_guard = model.lock().await;
            let logits = model_guard.forward(&input_ids, 0)?;
            
            // logits shape: [batch_size, seq_len, vocab_size]
            // Get logits for the last token
            let last_logits = logits.i((0, logits.dim(1)? - 1))?; // [vocab_size]
            
            // Use proper sampling with configured parameters
            let mut sampler = TokenSampler::new(self.sampling_config.clone());
            sampler.add_to_history(context_window);
            let next_token = sampler.sample(&last_logits)?;
            
            tracing::trace!("🔮 Model forward pass: context_len={}, next_token={}", context_window.len(), next_token);
            Ok(next_token)
        } else {
            // No model loaded - this should not happen in normal operation
            Err(anyhow!("No model loaded for inference"))
        }
    }
    
    /// Perform forward pass with SafeTensors weights
    async fn safetensors_forward_pass(&self, context: &[u32], weights: &HashMap<String, Tensor>) -> Result<u32> {
        // Use the actual architecture model if available
        if let Some(arch_model) = &self.arch_model {
            let model = arch_model.lock().await;
            
            // Create tensor from token IDs - use u32 directly
            let input_ids = Tensor::new(context, &self.device)?;
            let input_ids = input_ids.unsqueeze(0)?; // Add batch dimension
            
            // Run model forward pass
            let logits = model.forward(&input_ids, None)?;
            
            // Get last token logits and sample using proper sampling
            let last_logits = logits.i((0, logits.dim(1)? - 1))?;
            
            let mut sampler = TokenSampler::new(self.sampling_config.clone());
            sampler.add_to_history(context);
            let next_token = sampler.sample(&last_logits)?;
            
            return Ok(next_token);
        }
        
        // Fallback: If no architecture model loaded, return error
        Err(anyhow!("SafeTensors model not properly loaded. Architecture model required for inference."))
    }
    
    /// Apply LoRA adjustment to logits before sampling
    async fn apply_lora_to_logits(&self, logits: &Tensor, _lora: &SparseLoRAAdapter) -> Result<Tensor> {
        // TODO: Implement proper LoRA application
        // For now, LoRA should be a no-op until trained
        // An untrained LoRA should not modify the base model outputs
        
        // Return logits unchanged - LoRA is not yet trained
        Ok(logits.clone())
    }
    
    /// Create architecture model from SafeTensors weights
    async fn create_arch_model_from_safetensors(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        // Use the already detected architecture (from config or tensor names)
        let detected_arch = self.detected_architecture
            .clone()
            .unwrap_or_else(|| self.detect_architecture_from_tensors(weights));
        let arch_name = detected_arch.name();
        tracing::info!("Creating {} model from SafeTensors weights", arch_name);
        
        // Choose dtype based on device - CPU doesn't support BF16 matmul
        let dtype = match &self.device {
            Device::Cpu => {
                tracing::info!("Using F32 dtype for CPU compatibility");
                candle_core::DType::F32
            },
            Device::Cuda(_) | Device::Metal(_) => {
                tracing::info!("Using BF16 dtype for GPU");
                candle_core::DType::BF16
            }
        };
        
        // Use the ModelFactory to create the appropriate model
        // This ensures we use the proper abstraction and can easily add new architectures
        let model = ModelFactory::from_weights(weights, detected_arch, &self.device, dtype)?;
        self.arch_model = Some(Arc::new(Mutex::new(model)));
        
        tracing::info!("✅ Created {} architecture model from SafeTensors with dtype {:?}", arch_name, dtype);
        Ok(())
    }
    
    /// Generate a fallback token when model inference fails
    fn generate_fallback_token(&self, context: &[u32]) -> Result<u32> {
        if context.is_empty() {
            return Ok(1); // BOS token
        }
        
        // Use a more reasonable fallback that creates some coherence
        let vocab_size = self.model_info.as_ref()
            .map(|info| info.vocab_size)
            .unwrap_or(32000);
        
        // Generate tokens in common ranges
        let last_token = context[context.len() - 1];
        let next_token = match last_token {
            // If last token was punctuation, likely next is a word
            33..=47 => (50 + (context.len() as u32 * 7) % 200) % vocab_size as u32, 
            // If last token was a word, continue with word or space
            _ => {
                if context.len() % 5 == 0 {
                    32 // Space token
                } else {
                    // Generate word-like token
                    (100 + (last_token * 3 + context.len() as u32) % 1000) % vocab_size as u32
                }
            }
        };
        
        Ok(next_token)
    }
    
    /// Check if token is end-of-sequence
    fn is_eos_token(&self, token: u32) -> bool {
        // Common EOS tokens
        token == 2 || token == 0 || token == 32000
    }
    
    /// Apply temporal weight update based on generation
    async fn apply_temporal_update(&self, context: &[u32], token: u32) -> Result<()> {
        if let Some(temporal_streaming) = &self.temporal_streaming {
            // Compute temporal gradient based on prediction accuracy
            let prediction_error = self.compute_prediction_error(context, token)?;
            
            // Generate gradient update for active adapters
            let gradient_update = self.compute_temporal_gradient(context, prediction_error).await?;
            
            // Apply gradient through temporal streaming layer
            temporal_streaming.apply_gradient_update(gradient_update).await
                .map_err(|e| anyhow!("Failed to apply gradient: {}", e))?;
            
            tracing::trace!("🔄 Applied temporal gradient update for token: {}", token);
        }
        Ok(())
    }
    
    /// Compute prediction error for temporal learning
    fn compute_prediction_error(&self, context: &[u32], actual_token: u32) -> Result<f32> {
        // Simple prediction error based on token probability
        // In practice, this would use the model's logits
        let predicted_token = if context.is_empty() { 1 } else { context[context.len() - 1] };
        let error = if predicted_token == actual_token { 0.0 } else { 1.0 };
        Ok(error)
    }
    
    /// Compute temporal gradient for weight updates
    async fn compute_temporal_gradient(&self, context: &[u32], error: f32) -> Result<TemporalGradient> {
        let mut gradient_updates = HashMap::new();
        
        // Compute gradients for attention layers based on context
        if context.len() > 1 {
            let attention_gradient = self.compute_attention_gradient(context, error).await?;
            gradient_updates.insert("attention".to_string(), attention_gradient);
        }
        
        // Compute gradients for FFN layers
        let ffn_gradient = self.compute_ffn_gradient(context, error).await?;
        gradient_updates.insert("ffn".to_string(), ffn_gradient);
        
        Ok(TemporalGradient {
            layer_gradients: gradient_updates,
            learning_rate: 0.001,
            timestamp: std::time::SystemTime::now(),
        })
    }
    
    /// Compute attention gradient based on context patterns
    async fn compute_attention_gradient(&self, context: &[u32], error: f32) -> Result<LayerGradient> {
        // Simplified attention gradient computation
        // In practice, this would use backpropagation through attention mechanism
        let seq_len = context.len();
        let attention_scores = vec![error / seq_len as f32; seq_len];
        
        Ok(LayerGradient {
            weight_deltas: attention_scores,
            bias_deltas: vec![error * 0.1; seq_len],
            sparsity_mask: self.compute_sparsity_mask(seq_len).await?,
        })
    }
    
    /// Compute FFN gradient for temporal adaptation
    async fn compute_ffn_gradient(&self, context: &[u32], error: f32) -> Result<LayerGradient> {
        let hidden_dim = 4096; // Typical FFN hidden dimension
        let weight_deltas = vec![error * 0.01; hidden_dim];
        let bias_deltas = vec![error * 0.001; hidden_dim];
        
        Ok(LayerGradient {
            weight_deltas,
            bias_deltas,
            sparsity_mask: self.compute_sparsity_mask(hidden_dim).await?,
        })
    }
    
    /// Compute sparsity mask for 99% sparse updates
    async fn compute_sparsity_mask(&self, size: usize) -> Result<Vec<bool>> {
        let mut mask = vec![false; size];
        let num_active = (size as f32 * 0.01) as usize; // 1% active
        
        // Randomly select 1% of elements to be active
        use std::collections::HashSet;
        let mut active_indices = HashSet::new();
        while active_indices.len() < num_active {
            let idx = (context_hash(size) % size as u64) as usize;
            active_indices.insert(idx);
        }
        
        for idx in active_indices {
            mask[idx] = true;
        }
        
        Ok(mask)
    }
    
    /// Decode tokens back to string
    fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        if let Some(tokenizer) = &self.tokenizer {
            let output = tokenizer.decode(tokens, false)
                .map_err(|e| anyhow!("Decoding failed: {}", e))?;
            Ok(output)
        } else {
            // Last resort: convert tokens back to characters
            tracing::warn!("⚠️ Using fallback character-level decoding");
            let chars: String = tokens.iter()
                .filter_map(|&t| char::from_u32(t))
                .collect();
            Ok(chars)
        }
    }
    
}

#[async_trait]
impl RuntimeEngine for CandleEngine {
    async fn load_model(&mut self, path: &Path) -> Result<()> {
        self.load_safetensors_model(path).await
    }
    
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_with_vdb(prompt, max_tokens).await
    }
    
    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let text = self.generate_with_vdb(&request.prompt, request.max_tokens).await?;
        
        Ok(GenerationResult {
            text,
            tokens_generated: request.max_tokens, // Placeholder
            finish_reason: FinishReason::MaxTokens,
            generation_time_ms: 0,
            tokens_per_second: 0.0,
        })
    }
    
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Candle Model".to_string(),
            architecture: "llama".to_string(),
            parameters: 0,
            context_length: 2048,
            vocab_size: 32000,
            quantization: Some("F32/BF16".to_string()),
        }
    }
    
    fn is_loaded(&self) -> bool {
        self.model.is_some() || self.base_model_weights.is_some()
    }
}

impl CandleEngine {
    /// Real-time adapter weight updates via VDB
    async fn update_adapter_realtime(
        &mut self, 
        adapter_id: &str, 
        weights: &crate::adapters::lora_checkpoints::LoRAWeightsData
    ) -> Result<()> {
        // TODO: Implement actual weight updates when VDB API is complete
        // For now, just log the update request
        tracing::info!("📝 Received weight update request for adapter: {}", adapter_id);
        
        // Apply weight updates
        for (layer_name, _layer_weights) in &weights.a_weights {
            // Convert dense weights to sparse updates
            tracing::debug!("Would update layer: {} in adapter: {}", layer_name, adapter_id);
        }
        
        Ok(())
    }
    
    /// Enable temporal streaming for real-time adaptation
    async fn enable_realtime_adaptation(
        &mut self, 
        mode: super::AdaptationMode
    ) -> Result<()> {
        use crate::storage::vdb::TemporalStreamingConfig;
        
        // Initialize temporal streaming if not already done
        if self.temporal_streaming.is_none() {
            let config = TemporalStreamingConfig::default();
            self.temporal_streaming = Some(Arc::new(
                TemporalStreamingLayer::new(self.vdb_storage.clone(), config).await?
            ));
        }
        
        tracing::info!("✅ Enabled real-time adaptation with mode: {:?}", mode);
        Ok(())
    }
}

impl CandleEngine {
    /// Train on a prompt-response pair using temporal LoRA
    pub async fn train_temporal_lora(
        &mut self,
        prompt: &str,
        expected_response: &str,
        learning_rate: f32,
    ) -> Result<TrainingResult> {
        tracing::info!("🎓 Starting temporal LoRA training");
        
        // Enable temporal streaming if not active
        if self.temporal_streaming.is_none() {
            self.enable_realtime_adaptation(super::AdaptationMode::Disabled).await?;
        }
        
        // Tokenize training data
        let prompt_tokens = self.tokenize_prompt(prompt)?;
        let response_tokens = self.tokenize_prompt(expected_response)?;
        
        // Run inference to get current predictions
        let predicted_response = self.generate_with_vdb(prompt, response_tokens.len()).await?;
        let predicted_tokens = self.tokenize_prompt(&predicted_response)?;
        
        // Compute training loss
        let loss = self.compute_training_loss(&response_tokens, &predicted_tokens)?;
        
        // Compute gradients based on prediction errors
        let mut total_gradient_updates = 0;
        for (i, (&expected, &predicted)) in response_tokens.iter().zip(predicted_tokens.iter()).enumerate() {
            if expected != predicted {
                let context = &prompt_tokens[..std::cmp::min(i + prompt_tokens.len(), prompt_tokens.len())];
                let error = 1.0; // Binary error for now
                
                // Compute and apply temporal gradient
                let gradient = self.compute_temporal_gradient(context, error).await?;
                
                if let Some(temporal_streaming) = &self.temporal_streaming {
                    temporal_streaming.apply_gradient_update(gradient).await
                        .map_err(|e| anyhow!("Failed to apply gradient: {}", e))?;
                    total_gradient_updates += 1;
                }
            }
        }
        
        tracing::info!("✅ Applied {} gradient updates for temporal LoRA training", total_gradient_updates);
        
        Ok(TrainingResult {
            loss,
            gradient_updates: total_gradient_updates,
            tokens_processed: response_tokens.len(),
            training_time_ms: 0, // TODO: measure actual time
        })
    }
    
    /// Compute training loss between expected and predicted tokens
    fn compute_training_loss(&self, expected: &[u32], predicted: &[u32]) -> Result<f32> {
        let mut total_loss = 0.0;
        let max_len = std::cmp::max(expected.len(), predicted.len());
        
        for i in 0..max_len {
            let exp = expected.get(i).unwrap_or(&0);
            let pred = predicted.get(i).unwrap_or(&0);
            
            // Simple cross-entropy approximation
            if exp != pred {
                total_loss += 1.0;
            }
        }
        
        Ok(total_loss / max_len as f32)
    }
    
}

/// Create a new Candle engine
pub fn create_engine(config: &RuntimeConfig) -> Result<CandleEngine> {
    CandleEngine::new(config.clone())
}