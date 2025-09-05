//! Model loader for base models with memory mapping support

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use memmap2::Mmap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::constants::limits::*;

/// Handle to a loaded base model
#[derive(Debug, Clone)]
pub struct BaseModelHandle {
    /// Path to model files
    pub model_path: PathBuf,
    
    /// Model configuration
    pub config: ModelConfig,
    
    /// Memory-mapped weight files
    pub weight_maps: HashMap<String, WeightMap>,
    
    /// Tokenizer handle
    pub tokenizer: Option<TokenizerHandle>,
    
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Model configuration loaded from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    
    // Optional fields that might be present
    pub num_key_value_heads: Option<usize>,
    pub rope_theta: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: "unknown".to_string(),
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-6,
            num_key_value_heads: None,
            rope_theta: Some(10000.0),
            tie_word_embeddings: Some(false),
        }
    }
}

/// Memory-mapped weight file
#[derive(Debug)]
pub struct WeightMap {
    /// Memory map of the file
    pub mmap: Mmap,
    
    /// Weight tensors in this file
    pub tensors: HashMap<String, TensorInfo>,
    
    /// File path for debugging
    pub file_path: PathBuf,
}

impl Clone for WeightMap {
    fn clone(&self) -> Self {
        use std::fs::File;
        use memmap2::MmapOptions;
        
        // Re-open the file and create a new memory map
        let file = File::open(&self.file_path)
            .expect("Failed to reopen file for cloning WeightMap");
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .expect("Failed to create memory map for cloning WeightMap");
        
        WeightMap {
            mmap,
            tensors: self.tensors.clone(),
            file_path: self.file_path.clone(),
        }
    }
}

/// Information about a tensor in a weight file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte offset where tensor data starts in the file
    pub offset: usize,
    pub size_bytes: usize,
}

/// Tokenizer handle
#[derive(Debug, Clone)]
pub struct TokenizerHandle {
    pub vocab_size: usize,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub total_size_bytes: u64,
    pub files: Vec<String>,
    pub loaded_at: i64,
}

/// Model loader implementation
pub struct ModelLoader {
    /// Currently loaded models
    loaded_models: HashMap<String, BaseModelHandle>,
}

impl ModelLoader {
    /// Create new model loader
    pub async fn new(_model_path: &Path) -> Result<Self> {
        Ok(Self {
            loaded_models: HashMap::new(),
        })
    }
    
    /// Load a model from the given path
    pub async fn load_model(&mut self, model_path: &Path) -> Result<BaseModelHandle> {
        let model_name = model_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        println!("🔄 Loading model from: {}", model_path.display());
        
        // Check if already loaded
        if let Some(handle) = self.loaded_models.get(&model_name) {
            return Ok(handle.clone());
        }
        
        // Load configuration
        let config = self.load_config(model_path).await?;
        
        // Load weight files
        let weight_maps = self.load_weights(model_path).await?;
        
        // Load tokenizer
        let tokenizer = self.load_tokenizer(model_path).await.ok();
        
        // Calculate metadata
        let total_size: u64 = weight_maps.values()
            .map(|w| w.mmap.len() as u64)
            .sum();
        
        let files: Vec<String> = weight_maps.keys()
            .map(|k| k.clone())
            .collect();
        
        let parameter_count = self.calculate_parameter_count(&weight_maps);
        
        let metadata = ModelMetadata {
            name: model_name.clone(),
            architecture: config.model_type.clone(),
            parameter_count,
            total_size_bytes: total_size,
            files,
            loaded_at: chrono::Utc::now().timestamp(),
        };
        
        let handle = BaseModelHandle {
            model_path: model_path.to_path_buf(),
            config,
            weight_maps,
            tokenizer,
            metadata,
        };
        
        // Cache the loaded model
        self.loaded_models.insert(model_name, handle.clone());
        
        println!("✅ Model loaded: {} ({} parameters, {} MB)", 
                handle.metadata.name,
                handle.metadata.parameter_count,
                handle.metadata.total_size_bytes / (1024 * 1024));
        
        Ok(handle)
    }
    
    /// Load model configuration from config.json
    async fn load_config(&self, model_path: &Path) -> Result<ModelConfig> {
        let config_path = model_path.join("config.json");
        
        if !config_path.exists() {
            println!("⚠️ No config.json found, using default configuration");
            return Ok(ModelConfig::default());
        }
        
        let content = tokio::fs::read_to_string(&config_path).await?;
        let config: ModelConfig = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;
        
        Ok(config)
    }
    
    /// Load all weight files in the model directory
    async fn load_weights(&self, model_path: &Path) -> Result<HashMap<String, WeightMap>> {
        let mut weight_maps = HashMap::new();
        
        let mut entries = tokio::fs::read_dir(model_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            // Look for SafeTensors files (preferred)
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    let weight_map = self.load_safetensors_file(&path).await?;
                    let file_name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    weight_maps.insert(file_name, weight_map);
                }
            }
        }
        
        if weight_maps.is_empty() {
            return Err(anyhow::anyhow!("No weight files found in {}", model_path.display()));
        }
        
        Ok(weight_maps)
    }
    
    /// Load a SafeTensors file with memory mapping
    async fn load_safetensors_file(&self, file_path: &Path) -> Result<WeightMap> {
        use std::fs::File;
        
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Parse SafeTensors header
        let tensors = self.parse_safetensors_header(&mmap)?;
        
        Ok(WeightMap {
            mmap,
            tensors,
            file_path: file_path.to_path_buf(),
        })
    }
    
    /// Parse SafeTensors file header to extract tensor information
    fn parse_safetensors_header(&self, mmap: &Mmap) -> Result<HashMap<String, TensorInfo>> {
        if mmap.len() < 8 {
            return Err(anyhow::anyhow!("File too small to be a valid SafeTensors file"));
        }
        
        // Validate file size first
        if mmap.len() < MIN_SAFETENSORS_SIZE {
            return Err(anyhow::anyhow!("File too small to be a valid SafeTensors file"));
        }
        
        // Read header length (first 8 bytes)
        let header_len = u64::from_le_bytes(
            mmap[0..8].try_into()
                .map_err(|_| anyhow::anyhow!("Failed to read header length"))?
        ) as usize;
        
        // Validate header length
        if header_len as u64 > MAX_HEADER_SIZE {
            return Err(anyhow::anyhow!("Header size {} exceeds maximum allowed {}", header_len, MAX_HEADER_SIZE));
        }
        
        if mmap.len() < 8 + header_len {
            return Err(anyhow::anyhow!("File too small for declared header size"));
        }
        
        // Read header JSON
        let header_bytes = &mmap[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|_| anyhow::anyhow!("Header is not valid UTF-8"))?;
        
        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse header JSON: {}", e))?;
        
        let mut tensors = HashMap::new();
        let data_offset = 8 + header_len;
        
        for (name, info) in header.iter() {
            if name == "__metadata__" {
                continue; // Skip metadata entry
            }
            
            let tensor_info = self.parse_tensor_info(name, info, data_offset)?;
            
            // Validate that tensor bounds are within file
            if tensor_info.offset + tensor_info.size_bytes > mmap.len() {
                return Err(anyhow::anyhow!(
                    "Tensor '{}' extends beyond file bounds: offset {} + size {} > file size {}",
                    name, tensor_info.offset, tensor_info.size_bytes, mmap.len()
                ));
            }
            
            tensors.insert(name.clone(), tensor_info);
        }
        
        Ok(tensors)
    }
    
    /// Parse individual tensor information from SafeTensors header
    fn parse_tensor_info(
        &self,
        name: &str,
        info: &serde_json::Value,
        base_offset: usize,
    ) -> Result<TensorInfo> {
        let obj = info.as_object()
            .ok_or_else(|| anyhow::anyhow!("Tensor info must be an object"))?;
        
        let dtype = obj.get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing dtype"))?
            .to_string();
        
        let shape: Vec<usize> = obj.get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing shape"))?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("Invalid shape format"))?;
        
        let data_offsets = obj.get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing data_offsets"))?;
        
        let start_offset = data_offsets[0].as_u64()
            .ok_or_else(|| anyhow::anyhow!("Invalid start offset"))? as usize;
        
        let end_offset = data_offsets[1].as_u64()
            .ok_or_else(|| anyhow::anyhow!("Invalid end offset"))? as usize;
        
        // Validate offsets
        if end_offset < start_offset {
            return Err(anyhow::anyhow!("Invalid offsets: end ({}) < start ({})", end_offset, start_offset));
        }
        
        let size_bytes = end_offset - start_offset;
        let final_offset = base_offset + start_offset;
        
        // Sanity check: Individual tensors shouldn't exceed 100GB
        // (even the largest models have individual tensors under this)
        if size_bytes as u64 > MAX_TENSOR_SIZE {
            return Err(anyhow::anyhow!("Tensor size {} exceeds maximum allowed size of 100GB", size_bytes));
        }
        
        Ok(TensorInfo {
            name: name.to_string(),
            dtype,
            shape,
            offset: final_offset,
            size_bytes,
        })
    }
    
    /// Load tokenizer configuration
    async fn load_tokenizer(&self, model_path: &Path) -> Result<TokenizerHandle> {
        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        
        if !tokenizer_config_path.exists() {
            return Err(anyhow::anyhow!("No tokenizer_config.json found"));
        }
        
        let content = tokio::fs::read_to_string(&tokenizer_config_path).await?;
        let config: serde_json::Value = serde_json::from_str(&content)?;
        
        let vocab_size = config.get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;
        
        let pad_token_id = config.get("pad_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        
        let bos_token_id = config.get("bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        
        let eos_token_id = config.get("eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        
        let unk_token_id = config.get("unk_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        
        Ok(TokenizerHandle {
            vocab_size,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            unk_token_id,
        })
    }
    
    /// Calculate total parameter count from weight maps
    fn calculate_parameter_count(&self, weight_maps: &HashMap<String, WeightMap>) -> u64 {
        let mut total_params = 0u64;
        
        for weight_map in weight_maps.values() {
            for tensor in weight_map.tensors.values() {
                let params: u64 = tensor.shape.iter()
                    .map(|&dim| dim as u64)
                    .product();
                total_params += params;
            }
        }
        
        total_params
    }
    
    /// Get a loaded model by name
    pub fn get_model(&self, model_name: &str) -> Option<&BaseModelHandle> {
        self.loaded_models.get(model_name)
    }
    
    /// List all loaded models
    pub fn list_models(&self) -> Vec<String> {
        self.loaded_models.keys().cloned().collect()
    }
    
    /// Unload a model from memory
    pub fn unload_model(&mut self, model_name: &str) -> Result<()> {
        self.loaded_models.remove(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model not loaded: {}", model_name))?;
        
        println!("🗑️ Unloaded model: {}", model_name);
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let mut total_size = 0u64;
        let mut models = Vec::new();
        
        for (name, handle) in &self.loaded_models {
            total_size += handle.metadata.total_size_bytes;
            models.push(ModelMemoryInfo {
                name: name.clone(),
                size_bytes: handle.metadata.total_size_bytes,
                parameter_count: handle.metadata.parameter_count,
            });
        }
        
        MemoryStats {
            total_models: self.loaded_models.len(),
            total_size_bytes: total_size,
            models,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_models: usize,
    pub total_size_bytes: u64,
    pub models: Vec<ModelMemoryInfo>,
}

/// Memory info for individual model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMemoryInfo {
    pub name: String,
    pub size_bytes: u64,
    pub parameter_count: u64,
}

use chrono;