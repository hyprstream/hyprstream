//! Model factory for creating models with unified configuration management
//!
//! This replaces the chaotic multiple paths for model creation with a single,
//! clean factory pattern.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Kind as DType, Tensor};
use tracing::{info, instrument};

use super::architectures::{gemma::GemmaModel, llama::LlamaModel, ModelOperations};
use super::model_config::{ModelArchitecture, ModelConfig};
#[cfg(feature = "xet")]
use crate::storage::{LfsXetBridge, XetConfig};

/// Factory for creating models with proper configuration management
pub struct ModelFactory;

impl ModelFactory {
    /// Detect the dtype of a model by examining its tensors
    pub async fn detect_model_dtype(model_path: &Path) -> Result<DType> {
        // Check for single file first
        let single_file = model_path.join("model.safetensors");
        let file_to_check = if single_file.exists() {
            single_file
        } else {
            // Look for first shard file
            let shard_files = Self::find_shard_files(model_path)?;
            if shard_files.is_empty() {
                return Err(anyhow!("No model weights found in {}", model_path.display()));
            }
            shard_files[0].clone()
        };

        // Load just the metadata to check dtype
        let file_content = std::fs::read(&file_to_check)?;
        let tensors = safetensors::SafeTensors::deserialize(&file_content)?;

        // Check the first few tensors to determine predominant dtype
        let mut f16_count = 0;
        let mut bf16_count = 0;
        let mut f32_count = 0;
        let mut other_count = 0;

        for (_, tensor) in tensors.tensors().into_iter().take(10) {
            match tensor.dtype() {
                safetensors::Dtype::F16 => f16_count += 1,
                safetensors::Dtype::BF16 => bf16_count += 1,
                safetensors::Dtype::F32 => f32_count += 1,
                _ => other_count += 1,
            }
        }

        // Return the most common dtype
        if f16_count > bf16_count && f16_count > f32_count {
            info!("Detected F16 model");
            Ok(tch::Kind::Half)
        } else if bf16_count >= f16_count && bf16_count >= f32_count {
            info!("Detected BF16 model");
            Ok(tch::Kind::BFloat16)
        } else if f32_count > 0 {
            info!("Detected F32 model");
            Ok(tch::Kind::Float)
        } else {
            info!("Could not detect model dtype, defaulting to BF16");
            Ok(tch::Kind::BFloat16)
        }
    }

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
            info!(
                "📦 Using incremental loading for {} shards",
                shard_files.len()
            );
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
        let mut shard_files =
            tokio::task::spawn_blocking(move || -> Result<Vec<std::path::PathBuf>> {
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
            })
            .await??;

        if !shard_files.is_empty() {
            shard_files.sort();
            info!("Loading {} weight shards", shard_files.len());
            for shard_file in shard_files {
                Self::load_safetensors_file(&shard_file, &mut all_weights, device, dtype).await?;
            }
            return Ok(all_weights);
        }

        Err(anyhow!(
            "No safetensors files found in {}",
            model_path.display()
        ))
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
            use memmap2::Mmap;
            use std::fs::File;

            let file = File::open(&path_buf)?;
            let mmap = unsafe { Mmap::map(&file)? };

            // Note: We must deserialize and create tensors while mmap is alive
            // The tensors will copy the data they need during creation
            let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
            Self::create_tensors_from_safetensors(tensors, weights, device, dtype)?;

            // mmap drops here - tensors have already copied what they need
            return Ok(());
        }

        // Standard approach - load file with LFS/XET pointer detection
        // This handles both:
        // 1. Already-smudged files (fast path via git-xet-filter)
        // 2. Un-smudged pointers (fallback via explicit load_file)
        let tensor_data = Self::load_file_with_pointer_detection(&path_buf).await?;
        let tensors = safetensors::SafeTensors::deserialize(&tensor_data)?;
        Self::create_tensors_from_safetensors(tensors, weights, device, dtype)
    }

    /// Load file with automatic LFS/XET pointer detection
    ///
    /// Fast path for already-smudged files, fallback for un-smudged pointers.
    async fn load_file_with_pointer_detection(path: &Path) -> Result<Vec<u8>> {
        let metadata = tokio::fs::metadata(path).await?;

        // Large files cannot be LFS pointers (which are < 1KB)
        if metadata.len() >= 1024 {
            return tokio::fs::read(path).await.map_err(Into::into);
        }

        let data = tokio::fs::read(path).await?;

        // Check for LFS pointer header
        if data.len() < 1024 {
            if let Ok(text) = String::from_utf8(data.clone()) {
                if text.starts_with("version https://git-lfs") {
                    #[cfg(feature = "xet")]
                    {
                        debug!("Un-smudged LFS pointer, using XET fallback: {}", path.display());
                        let bridge = LfsXetBridge::new(XetConfig::default()).await?;
                        return bridge.load_file(path).await;
                    }

                    #[cfg(not(feature = "xet"))]
                    {
                        anyhow::bail!(
                            "Un-smudged LFS pointer at {} but XET feature disabled. \
                             Enable with --features xet or ensure files are smudged during checkout.",
                            path.display()
                        );
                    }
                }
            }
        }

        Ok(data)
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

            // Support both F16 and BF16 models
            let tensor = match tensor_view.dtype() {
                safetensors::Dtype::BF16 => {
                    // Verify target dtype matches
                    if dtype != tch::Kind::BFloat16 && dtype != tch::Kind::Half {
                        return Err(anyhow!(
                            "Model dtype BF16 but target dtype is {:?}",
                            dtype
                        ));
                    }

                    // Create tensor from borrowed data, then make an owned copy
                    // IMPORTANT: from_blob only borrows the data pointer, so we must
                    // copy the tensor to own the data before the source buffer is freed
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::BFloat16,
                            Device::Cpu,
                        )
                    };

                    // Make an owned copy to prevent use-after-free
                    let cpu_tensor = borrowed_tensor.copy();

                    // Convert dtype if needed
                    let cpu_tensor = if dtype == tch::Kind::Half {
                        cpu_tensor.to_kind(tch::Kind::Half)
                    } else {
                        cpu_tensor
                    };

                    // Transfer to target device if needed
                    if *device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(*device);
                        drop(cpu_tensor); // Explicitly free CPU memory
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                safetensors::Dtype::F16 => {
                    // Verify target dtype matches
                    if dtype != tch::Kind::Half && dtype != tch::Kind::BFloat16 {
                        return Err(anyhow!(
                            "Model dtype F16 but target dtype is {:?}",
                            dtype
                        ));
                    }

                    // Create tensor from borrowed data
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Half,
                            Device::Cpu,
                        )
                    };

                    // Make an owned copy to prevent use-after-free
                    let cpu_tensor = borrowed_tensor.copy();

                    // Convert dtype if needed
                    let cpu_tensor = if dtype == tch::Kind::BFloat16 {
                        cpu_tensor.to_kind(tch::Kind::BFloat16)
                    } else {
                        cpu_tensor
                    };

                    // Transfer to target device if needed
                    if *device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(*device);
                        drop(cpu_tensor); // Explicitly free CPU memory
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                safetensors::Dtype::F32 => {
                    // Support F32 as well for completeness
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Float,
                            Device::Cpu,
                        )
                    };

                    // Make an owned copy
                    let cpu_tensor = borrowed_tensor.copy();

                    // Convert to target dtype
                    let cpu_tensor = match dtype {
                        tch::Kind::Half => cpu_tensor.to_kind(tch::Kind::Half),
                        tch::Kind::BFloat16 => cpu_tensor.to_kind(tch::Kind::BFloat16),
                        tch::Kind::Float => cpu_tensor,
                        _ => {
                            return Err(anyhow!(
                                "Cannot convert F32 to target dtype {:?}",
                                dtype
                            ))
                        }
                    };

                    // Transfer to target device if needed
                    if *device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(*device);
                        drop(cpu_tensor); // Explicitly free CPU memory
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                dtype => {
                    return Err(anyhow!(
                        "Tensor '{}' has unsupported dtype {:?}. Supported: F16, BF16, F32",
                        name, dtype
                    ));
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
            ModelArchitecture::Unknown(arch) => Err(anyhow!("Unknown architecture: {}", arch)),
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
            original_vocab_size: config.vocab_size,  // Will be updated if padding is applied
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

        let model = LlamaModel::from_weights_with_config(
            &weights,
            llama_config,
            device,
            dtype,
        )?;

        Ok(Box::new(model))
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
