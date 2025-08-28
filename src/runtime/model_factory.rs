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

/// Factory for creating models with proper configuration management
pub struct ModelFactory;

impl ModelFactory {
    /// Create a model from a directory containing weights and optionally config.json
    /// This is the ONLY way models should be created to ensure consistency
    pub fn create(
        model_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        println!("🏭 ModelFactory: Loading model from {}", model_path.display());
        
        // Step 1: Load weights
        let weights = Self::load_weights(model_path, device, dtype)?;
        
        // Step 2: Load unified configuration (config.json + weight detection)
        let config = ModelConfig::load(model_path, &weights)?;
        
        // Step 3: Create model based on architecture
        let model = Self::create_model_from_config(config, weights, device, dtype)?;
        
        println!("✅ ModelFactory: Model created successfully");
        Ok(model)
    }
    
    /// Load weights from safetensors files
    fn load_weights(
        model_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let mut all_weights = HashMap::new();
        
        // Check for single safetensors file
        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            println!("📦 Loading single safetensors file");
            Self::load_safetensors_file(&single_file, &mut all_weights, device, dtype)?;
            return Ok(all_weights);
        }
        
        // Check for sharded safetensors
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
                Self::load_safetensors_file(&shard_file, &mut all_weights, device, dtype)?;
            }
            return Ok(all_weights);
        }
        
        Err(anyhow!("No safetensors files found in {}", model_path.display()))
    }
    
    /// Load a single safetensors file
    fn load_safetensors_file(
        path: &Path,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let buffer = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
        
        for (name, tensor_view) in tensors.tensors() {
            // Convert to tch tensor
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
            let data = tensor_view.data();
            
            // Create tensor based on original dtype
            let tensor = match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                    };
                    Tensor::from_slice(slice).reshape(&shape)
                }
                safetensors::Dtype::F16 => {
                    // Convert F16 to F32 first
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
                    };
                    let f32_vec: Vec<f32> = slice.iter().map(|x| x.to_f32()).collect();
                    Tensor::from_slice(&f32_vec).reshape(&shape)
                }
                safetensors::Dtype::BF16 => {
                    // Convert BF16 to F32 first
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len() / 2)
                    };
                    let f32_vec: Vec<f32> = slice.iter().map(|x| x.to_f32()).collect();
                    Tensor::from_slice(&f32_vec).reshape(&shape)
                }
                _ => {
                    println!("⚠️ Skipping tensor {} with unsupported dtype", name);
                    continue;
                }
            };
            
            // Move to device and convert to target dtype
            let tensor = tensor.to_device(*device).to_kind(dtype);
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
            rope_scaling: None, // TODO: Convert rope_scaling
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