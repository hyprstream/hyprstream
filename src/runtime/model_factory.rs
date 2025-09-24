//! Model factory for creating models with unified configuration management
//!
//! This replaces the chaotic multiple paths for model creation with a single,
//! clean factory pattern.

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Tensor, Kind as DType};

use super::model_config::{ModelConfig, ModelArchitecture};
use super::architectures::{ModelOperations, llama::LlamaModel, gemma::GemmaModel};
use crate::storage::XetNativeStorage;

/// Factory for creating models with proper configuration management
pub struct ModelFactory;

impl ModelFactory {
    /// Create a model from a directory containing weights and optionally config.json
    /// This is the ONLY way models should be created to ensure consistency
    pub async fn create(
        model_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        Self::create_with_xet(model_path, device, dtype, None).await
    }

    /// Create a model with optional XET storage for pointer file handling
    pub async fn create_with_xet(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        xet_storage: Option<&XetNativeStorage>,
    ) -> Result<Box<dyn ModelOperations>> {
        println!("🏭 ModelFactory: Loading model from {}", model_path.display());

        let weights = Self::load_weights_with_xet(model_path, device, dtype, xet_storage).await?;
        let config = ModelConfig::load(model_path, &weights)?;
        let model = Self::create_model_from_config(config, weights, device, dtype)?;

        println!("✅ ModelFactory: Model created successfully");
        Ok(model)
    }
    
    /// Load weights from safetensors files
    async fn load_weights(
        model_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        Self::load_weights_with_xet(model_path, device, dtype, None).await
    }

    /// Load weights from safetensors files with optional XET storage
    async fn load_weights_with_xet(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        xet_storage: Option<&XetNativeStorage>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut all_weights = HashMap::new();
        
        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            println!("📦 Loading single safetensors file");
            Self::load_safetensors_file_with_xet(&single_file, &mut all_weights, device, dtype, xet_storage).await?;
            return Ok(all_weights);
        }
        
        // Look for sharded safetensors files (model-00001-of-00002.safetensors pattern)
        let entries = std::fs::read_dir(model_path)?;
        let mut shard_files = Vec::new();
        
        for entry in entries {
            let entry = entry?;
            let filename = entry.file_name();
            if let Some(name) = filename.to_str() {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    shard_files.push(entry.path());
                }
            }
        }
        
        if !shard_files.is_empty() {
            shard_files.sort();
            println!("📦 Loading {} safetensors shards", shard_files.len());
            for shard_file in shard_files {
                Self::load_safetensors_file_with_xet(&shard_file, &mut all_weights, device, dtype, xet_storage).await?;
            }
            return Ok(all_weights);
        }
        
        Err(anyhow!("No safetensors files found in {}", model_path.display()))
    }
    
    /// Load a single safetensors file
    /// Can optionally use memory mapping for large models on systems with
    /// persistent memory (e.g., Optane drives)
    fn load_safetensors_file(
        path: &Path,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        // Check if mmap should be used (via environment variable for now)
        let use_mmap = std::env::var("HYPRSTREAM_USE_MMAP").unwrap_or_default() == "true";

        let buffer: Vec<u8>;
        let mmap_holder: Option<memmap2::Mmap>;

        let tensor_data = if use_mmap {
            // Memory-mapped loading for Optane/persistent memory systems
            let file = std::fs::File::open(path)?;

            // Memory map the file for efficient loading

            // Create memory map
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .populate() // Pre-populate pages for better performance
                    .map(&file)?
            };

            mmap_holder = Some(mmap);
            // Get a reference to the mmap data
            mmap_holder.as_ref().unwrap().as_ref()
        } else {
            // Standard loading into memory (default for most systems)
            buffer = std::fs::read(path)?;
            &buffer
        };

        // Deserialize from either mmap or buffer
        let tensors = safetensors::SafeTensors::deserialize(tensor_data)?;

        for (name, tensor_view) in tensors.tensors() {
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
            let data = tensor_view.data();

            // Create tensor with native dtype support when possible
            let tensor = match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                    };
                    Tensor::from_slice(slice).reshape(&shape)
                }
                safetensors::Dtype::F16 => {
                    // F16 - convert via F32 for now
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
                    };
                    let f32_vec: Vec<f32> = slice.iter().map(|x| x.to_f32()).collect();
                    Tensor::from_slice(&f32_vec).reshape(&shape)
                }
                safetensors::Dtype::BF16 => {
                    // BF16 - only support on BF16-capable devices
                    if dtype != tch::Kind::BFloat16 {
                        return Err(anyhow!("Model requires BF16 but target dtype is {:?}. BF16 models only work with BF16 dtype.", dtype));
                    }

                    // BF16 tensor creation - must create on CPU first due to from_blob GPU issues
                    // Creating BF16 directly on GPU causes cuda:-2 errors
                    unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],  // Use default strides
                            tch::Kind::BFloat16,
                            Device::Cpu,  // Must create on CPU first
                        )
                    }
                }
                _ => {
                    println!("⚠️ Skipping tensor {} with unsupported dtype", name);
                    continue;
                }
            };

            // Only convert if needed - handle CPU-created tensors properly
            let tensor = if tensor.device() == Device::Cpu && *device == Device::Cpu && tensor.kind() == dtype {
                tensor  // Already on CPU with correct dtype, no transfer needed
            } else if tensor.kind() == dtype {
                tensor.to_device(*device)  // Only device transfer needed
            } else if tensor.device() == *device {
                tensor.to_kind(dtype)  // Only dtype conversion needed
            } else {
                tensor.to_device(*device).to_kind(dtype)  // Both needed
            };
            weights.insert(name.to_string(), tensor);
        }

        Ok(())
    }

    /// Load a single safetensors file with optional XET storage support
    /// This version can handle XET pointers and LFS pointers automatically
    async fn load_safetensors_file_with_xet(
        path: &Path,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
        xet_storage: Option<&XetNativeStorage>,
    ) -> Result<()> {
        // If XET storage is available, use it for universal file loading
        let tensor_data = if let Some(xet) = xet_storage {
            // Use XET's universal file loading (handles regular files, XET pointers, and LFS pointers)
            xet.load_file(path).await?
        } else {
            // Fallback to standard file reading
            std::fs::read(path)?
        };

        // Deserialize SafeTensors format

        // Deserialize safetensors
        let tensors = safetensors::SafeTensors::deserialize(&tensor_data)?;

        for (name, tensor_view) in tensors.tensors() {
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
            let data = tensor_view.data();

            // Create tensor with native dtype support when possible
            let tensor = match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                    };
                    Tensor::from_slice(slice).reshape(&shape)
                }
                safetensors::Dtype::F16 => {
                    // F16 - convert via F32 for now
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
                    };
                    let f32_vec: Vec<f32> = slice.iter().map(|x| x.to_f32()).collect();
                    Tensor::from_slice(&f32_vec).reshape(&shape)
                }
                safetensors::Dtype::BF16 => {
                    // BF16 - only support on BF16-capable devices
                    if dtype != tch::Kind::BFloat16 {
                        return Err(anyhow!("Model requires BF16 but target dtype is {:?}. BF16 models only work with BF16 dtype.", dtype));
                    }

                    // BF16 tensor creation - must create on CPU first due to from_blob GPU issues
                    // Creating BF16 directly on GPU causes cuda:-2 errors
                    unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],  // Use default strides
                            tch::Kind::BFloat16,
                            Device::Cpu,  // Must create on CPU first
                        )
                    }
                }
                _ => {
                    println!("⚠️ Skipping tensor {} with unsupported dtype", name);
                    continue;
                }
            };

            // Only convert if needed - handle CPU-created tensors properly
            let tensor = if tensor.device() == Device::Cpu && *device == Device::Cpu && tensor.kind() == dtype {
                tensor  // Already on CPU with correct dtype, no transfer needed
            } else if tensor.kind() == dtype {
                tensor.to_device(*device)  // Only device transfer needed
            } else if tensor.device() == *device {
                tensor.to_kind(dtype)  // Only dtype conversion needed
            } else {
                tensor.to_device(*device).to_kind(dtype)  // Both needed
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
                println!("🦙 Creating Llama model");
                Self::create_llama_model(config, weights, device, dtype)
            }
            ModelArchitecture::Qwen => {
                println!("🐉 Creating Qwen model");
                Self::create_qwen_model(config, weights, device, dtype)
            }
            ModelArchitecture::Gemma => {
                println!("💎 Creating Gemma model");
                Self::create_gemma_model(config, weights, device, dtype)
            }
            ModelArchitecture::Mistral => {
                println!("🌪️ Creating Mistral model");
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
        println!("   Using Llama architecture with Qwen configuration");
        println!("   rope_theta: {} (from config)", config.rope_theta);
        Self::create_llama_model(config, weights, device, dtype)
    }
    
    fn create_gemma_model(
        config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Gemma has its own implementation
        Ok(Box::new(GemmaModel::from_weights(&weights, device, dtype)?))
    }
}