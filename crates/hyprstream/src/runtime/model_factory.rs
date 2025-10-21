//! Model factory for creating models with unified configuration management
//!
//! This replaces the chaotic multiple paths for model creation with a single,
//! clean factory pattern.

use anyhow::{Result, anyhow};
use tracing::{info, instrument};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Tensor, Kind as DType};

use super::model_config::{ModelConfig, ModelArchitecture};
use super::architectures::{ModelOperations, llama::LlamaModel, gemma::GemmaModel};

/// Factory for creating models with proper configuration management
pub struct ModelFactory;

impl ModelFactory {
    /// Create a model from a directory containing weights and optionally config.json
    /// This is the ONLY way models should be created to ensure consistency
    #[instrument(name = "model_factory.create", skip(device, dtype), fields(model_path = %model_path.display()))]
    pub async fn create(
        model_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        info!("Loading model: {}", model_path.display());

        // Check if we have sharded files that need incremental loading
        let shard_files = Self::find_shard_files(model_path)?;

        if !shard_files.is_empty() && shard_files.len() > 1 {
            // Use incremental loading for large sharded models
            info!("📦 Using incremental loading for {} shards", shard_files.len());
            Self::create_incremental(model_path, device, dtype, shard_files).await
        } else {
            // Standard loading for single files or small models
            let weights = Self::load_weights(model_path, device, dtype).await?;
            let config = ModelConfig::load(model_path, &weights)?;
            let model = Self::create_model_from_config(config, weights, device, dtype)?;
            info!("✅ ModelFactory: Model created successfully");
            Ok(model)
        }
    }

    /// Find all shard files in a model directory
    fn find_shard_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut shard_files = Vec::new();

        // Check for single file first
        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            return Ok(vec![single_file]);
        }

        // Look for sharded files
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let filename = entry.file_name();
            if let Some(name) = filename.to_str() {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    shard_files.push(entry.path());
                }
            }
        }

        shard_files.sort();
        Ok(shard_files)
    }

    /// Create model using incremental loading for large sharded models
    #[instrument(name = "model_factory.create_incremental", skip(device, dtype, shard_files), fields(shard_count = shard_files.len()))]
    async fn create_incremental(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        shard_files: Vec<std::path::PathBuf>,
    ) -> Result<Box<dyn ModelOperations>> {
        // For now, we still need to load all weights, but we do it more efficiently
        // by processing shards sequentially and immediately transferring to GPU
        info!("Loading {} weight shards", shard_files.len());

        let mut all_weights = HashMap::new();

        for (idx, shard_file) in shard_files.iter().enumerate() {
            info!("Loading shard {}/{}", idx + 1, shard_files.len());

            // Load shard weights directly to GPU to minimize CPU memory usage
            Self::load_safetensors_file(shard_file, &mut all_weights, device, dtype).await?;

            // Note: In a true streaming implementation, we would:
            // 1. Load layer weights
            // 2. Create that layer on GPU
            // 3. Free CPU memory before loading next layer
            // But this requires refactoring model architectures
        }

        // Load config and create model
        let config = ModelConfig::load(model_path, &all_weights)?;
        let model = Self::create_model_from_config(config, all_weights, device, dtype)?;

        info!("Model loaded");
        Ok(model)
    }
    
    /// Load weights from safetensors files
    async fn load_weights(
        model_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let mut all_weights = HashMap::new();
        
        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            info!("Loading model.safetensors");
            Self::load_safetensors_file(&single_file, &mut all_weights, device, dtype).await?;
            return Ok(all_weights);
        }
        
        // Look for sharded safetensors files (model-00001-of-00002.safetensors pattern)
        let model_path_buf = model_path.to_path_buf();
        let mut shard_files = tokio::task::spawn_blocking(move || -> Result<Vec<std::path::PathBuf>> {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(&model_path_buf)? {
                let entry = entry?;
                let filename = entry.file_name();
                if let Some(name) = filename.to_str() {
                    if name.starts_with("model-") && name.ends_with(".safetensors") {
                        files.push(entry.path());
                    }
                }
            }
            Ok(files)
        }).await??;
        
        if !shard_files.is_empty() {
            shard_files.sort();
            info!("Loading {} weight shards", shard_files.len());
            for shard_file in shard_files {
                Self::load_safetensors_file(&shard_file, &mut all_weights, device, dtype).await?;
            }
            return Ok(all_weights);
        }
        
        Err(anyhow!("No safetensors files found in {}", model_path.display()))
    }
    
    /// Load a single safetensors file
    #[instrument(name = "model_factory.load_safetensor_file", skip(weights, device, dtype), fields(file = %path.display()))]
    async fn load_safetensors_file(
        path: &Path,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        // Check if mmap is enabled via environment variable
        let use_mmap = std::env::var("HYPRSTREAM_USE_MMAP")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        // Load file data in a blocking task to avoid blocking the async runtime
        let path_buf = path.to_path_buf();

        // Use mmap for large files to reduce memory pressure
        if use_mmap {
            // Memory-mapped approach - OS manages paging
            use std::fs::File;
            use memmap2::Mmap;

            let file = File::open(&path_buf)?;
            let mmap = unsafe { Mmap::map(&file)? };

            // Note: We must deserialize and create tensors while mmap is alive
            // The tensors will copy the data they need during creation
            let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
            Self::create_tensors_from_safetensors(tensors, weights, device, dtype)?;

            // mmap drops here - tensors have already copied what they need
            return Ok(());
        }

        // Standard approach - load entire file into RAM
        let tensor_data = tokio::fs::read(&path_buf).await?;
        let tensors = safetensors::SafeTensors::deserialize(&tensor_data)?;
        Self::create_tensors_from_safetensors(tensors, weights, device, dtype)
    }

    /// Create tensors from deserialized safetensors
    fn create_tensors_from_safetensors(
        tensors: safetensors::SafeTensors,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        for (name, tensor_view) in tensors.tensors() {
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
            let data = tensor_view.data();

            // Create tensor optimized for GPU memory usage
            let tensor = match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                    };
                    // Create directly on target device to avoid CPU copy
                    Tensor::from_slice(slice)
                        .reshape(&shape)
                        .to_device(*device)
                        .to_kind(dtype)
                }
                safetensors::Dtype::F16 => {
                    // F16 requires conversion to F32 first
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
                    };
                    let f32_vec: Vec<f32> = slice.iter().map(|x| x.to_f32()).collect();
                    // Create and immediately transfer to GPU
                    Tensor::from_slice(&f32_vec)
                        .reshape(&shape)
                        .to_device(*device)
                        .to_kind(dtype)
                }
                safetensors::Dtype::BF16 => {
                    // BF16 handling - must copy data to avoid use-after-free
                    // The source buffer will be dropped, so we CANNOT use from_blob

                    if matches!(device, Device::Cuda(_)) && dtype == tch::Kind::BFloat16 {
                        // GPU path: Create BF16 tensor directly from owned data (no F32 conversion)

                        // Copy BF16 data into owned vector (avoids use-after-free)
                        let owned_bytes: Vec<u8> = data.to_vec();

                        // Create tensor directly from bytes on CPU with BF16 dtype
                        // This uses tch's internal C++ API to wrap the data
                        let cpu_tensor = unsafe {
                            // Reinterpret Vec<u8> as *const i16 for BF16 (represented as i16 in tch)
                            let ptr = owned_bytes.as_ptr() as *const i16;
                            let mut t = Tensor::from_blob(
                                ptr as *const _,
                                &shape,
                                &[],  // strides (empty = contiguous)
                                tch::Kind::BFloat16,
                                Device::Cpu,
                            );

                            // Make the tensor own the data by cloning it
                            // This ensures the data persists after owned_bytes is dropped
                            t = t.copy();

                            t
                        };

                        // Transfer directly to GPU (no conversion needed)
                        cpu_tensor.to_device(*device)
                    } else {
                        // CPU path or non-BF16 target: Convert to F32 since CPU doesn't support BF16
                        info!("Converting BF16 to F32 for CPU inference (BF16 not supported on CPU)");

                        let bf16_slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2)
                        };

                        // Convert BF16 to f32 with pre-allocated capacity
                        let mut f32_vec = Vec::with_capacity(bf16_slice.len());
                        f32_vec.extend(bf16_slice.iter().map(|&bf16_bits| {
                            f32::from_bits((bf16_bits as u32) << 16)
                        }));

                        // Create tensor from copied data
                        Tensor::from_slice(&f32_vec)
                            .reshape(&shape)
                            .to_device(*device)
                            .to_kind(dtype)
                    }
                }
                _ => {
                    info!("⚠️ Skipping tensor {} with unsupported dtype", name);
                    continue;
                }
            };

            weights.insert(name.to_string(), tensor);
        }

        Ok(())
    }

    /// Create model instance from configuration
    fn create_model_from_config(
        config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        match config.architecture {
            ModelArchitecture::Llama => {
                info!("Creating Llama model");
                Self::create_llama_model(config, weights, device, dtype)
            }
            ModelArchitecture::Qwen => {
                info!("Creating Qwen model");
                Self::create_qwen_model(config, weights, device, dtype)
            }
            ModelArchitecture::Gemma => {
                info!("Creating Gemma model");
                Self::create_gemma_model(config, weights, device, dtype)
            }
            ModelArchitecture::Mistral => {
                info!("Creating Mistral model");
                // For now, Mistral uses Llama architecture
                Self::create_llama_model(config, weights, device, dtype)
            }
            ModelArchitecture::Unknown(arch) => {
                Err(anyhow!("Unknown architecture: {}", arch))
            }
        }
    }
    
    fn create_llama_model(
        config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        use super::architectures::llama::LlamaConfig;
        
        // Convert unified config to LlamaConfig
        let llama_config = LlamaConfig {
            version: config.version as u8,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            vocab_size: config.vocab_size,
            num_hidden_layers: config.num_hidden_layers,
            rope_theta: config.rope_theta,
            rope_scaling: None,
            hidden_activation: config.hidden_activation,
            query_pre_attn_scalar: config.query_pre_attn_scalar,
            use_qk_norm: config.use_qk_norm,
            scale_embeddings: config.scale_embeddings,
            layer_types: vec![],
            rope_local_base_freq: None,
        };
        
        Ok(Box::new(LlamaModel::from_weights_with_config(
            &weights,
            llama_config,
            device,
            dtype,
        )?))
    }
    
    fn create_qwen_model(
        config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Qwen uses Llama architecture with specific configuration
        // The key difference is in the config values, not the architecture
        info!("   Using Llama architecture with Qwen configuration");
        info!("   rope_theta: {} (from config)", config.rope_theta);
        Self::create_llama_model(config, weights, device, dtype)
    }
    
    fn create_gemma_model(
        _config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Gemma has its own implementation
        Ok(Box::new(GemmaModel::from_weights(&weights, device, dtype)?))
    }
}