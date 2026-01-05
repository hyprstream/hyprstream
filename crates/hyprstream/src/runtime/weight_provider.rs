//! Weight provider for streaming large models
//!
//! This module provides an abstraction for loading model weights on-demand,
//! allowing models larger than system memory to be loaded.

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tch::{Device, Kind as DType, Tensor};
use tokio::sync::RwLock;
use tracing::info;

/// Trait for providing weights to models
pub trait WeightProvider: Send + Sync {
    /// Get a tensor by name, loading it if necessary
    fn get_tensor(&self, name: &str) -> Result<Tensor>;

    /// Check if a tensor exists
    fn has_tensor(&self, name: &str) -> bool;

    /// Get all tensor names (for config detection)
    fn tensor_names(&self) -> Vec<String>;

    /// Preload a set of tensors (for optimization)
    fn preload(&mut self, names: &[&str]) -> Result<()>;
}

/// Simple in-memory weight provider (current behavior)
/// Wraps tensors in Arc<Mutex<>> for thread safety
pub struct MemoryWeightProvider {
    weights: Arc<Mutex<HashMap<String, Tensor>>>,
}

impl MemoryWeightProvider {
    pub fn new(weights: HashMap<String, Tensor>) -> Self {
        Self {
            weights: Arc::new(Mutex::new(weights)),
        }
    }
}

impl WeightProvider for MemoryWeightProvider {
    fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let weights = self.weights.lock();
        weights
            .get(name)
            .map(|t| t.shallow_clone())
            .ok_or_else(|| anyhow!("Tensor {} not found", name))
    }

    fn has_tensor(&self, name: &str) -> bool {
        let weights = self.weights.lock();
        weights.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        let weights = self.weights.lock();
        weights.keys().cloned().collect()
    }

    fn preload(&mut self, _names: &[&str]) -> Result<()> {
        // Already in memory, nothing to do
        Ok(())
    }
}

/// Streaming weight provider that loads shards on demand
///
/// Note: We wrap non-thread-safe types to ensure Send + Sync
pub struct StreamingWeightProvider {
    shard_files: Vec<PathBuf>,
    device: Device,
    dtype: DType,
    /// Currently loaded shard weights
    current_weights: Arc<RwLock<HashMap<String, Tensor>>>,
    /// Map from tensor name to shard index
    tensor_shard_map: HashMap<String, usize>,
    /// Currently loaded shard index
    current_shard: Arc<RwLock<Option<usize>>>,
}

// Manually implement Send + Sync since we ensure thread-safe access
unsafe impl Send for StreamingWeightProvider {}
unsafe impl Sync for StreamingWeightProvider {}

impl StreamingWeightProvider {
    /// Create a new streaming weight provider
    pub async fn new(shard_files: Vec<PathBuf>, device: Device, dtype: DType) -> Result<Self> {
        // Build tensor -> shard mapping by reading headers
        let mut tensor_shard_map = HashMap::new();

        for (shard_idx, shard_path) in shard_files.iter().enumerate() {
            // Read just the safetensors header to get tensor names
            let header = Self::read_safetensors_header(shard_path).await?;
            for tensor_name in header.tensor_names {
                tensor_shard_map.insert(tensor_name, shard_idx);
            }
        }

        Ok(Self {
            shard_files,
            device,
            dtype,
            current_weights: Arc::new(RwLock::new(HashMap::new())),
            tensor_shard_map,
            current_shard: Arc::new(RwLock::new(None)),
        })
    }

    /// Read safetensors header without loading tensor data
    async fn read_safetensors_header(path: &Path) -> Result<SafeTensorsHeader> {
        use tokio::fs::File;
        use tokio::io::AsyncReadExt;

        let mut file = File::open(path).await?;

        // Read header size (first 8 bytes)
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes).await?;
        let header_size = u64::from_le_bytes(header_size_bytes);

        // Read header JSON
        let mut header_bytes = vec![0u8; header_size as usize];
        file.read_exact(&mut header_bytes).await?;

        let header_json = String::from_utf8(header_bytes)?;
        let header: serde_json::Value = serde_json::from_str(&header_json)?;

        // Extract tensor names
        let tensor_names: Vec<String> = header
            .as_object()
            .ok_or_else(|| anyhow!("Invalid safetensors header"))?
            .keys()
            .filter(|k| *k != "__metadata__")
            .cloned()
            .collect();

        Ok(SafeTensorsHeader { tensor_names })
    }

    /// Load a specific shard
    async fn load_shard(&self, shard_idx: usize) -> Result<HashMap<String, Tensor>> {
        info!("Loading shard {}/{}", shard_idx + 1, self.shard_files.len());

        let shard_path = &self.shard_files[shard_idx];
        let mut weights = HashMap::new();

        // Use existing load_safetensors_file logic
        // This would call into model_factory::load_safetensors_file
        // For now, simplified version:
        let tensor_data = tokio::fs::read(shard_path).await?;
        let tensors = safetensors::SafeTensors::deserialize(&tensor_data)?;

        for (name, tensor_view) in tensors.tensors() {
            // Create tensor (simplified - would use full logic from model_factory)
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
            let data = tensor_view.data();

            // Support F16, BF16, and F32 models
            let tensor = match tensor_view.dtype() {
                safetensors::Dtype::BF16 => {
                    let cpu_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::BFloat16,
                            Device::Cpu,
                        )
                    };

                    // Convert dtype if needed
                    let cpu_tensor = if self.dtype == tch::Kind::Half {
                        cpu_tensor.to_kind(tch::Kind::Half)
                    } else if self.dtype != tch::Kind::BFloat16 {
                        cpu_tensor.to_kind(self.dtype)
                    } else {
                        cpu_tensor
                    };

                    if self.device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(self.device);
                        drop(cpu_tensor);
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                safetensors::Dtype::F16 => {
                    let cpu_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Half,
                            Device::Cpu,
                        )
                    };

                    // Convert dtype if needed
                    let cpu_tensor = if self.dtype == tch::Kind::BFloat16 {
                        cpu_tensor.to_kind(tch::Kind::BFloat16)
                    } else if self.dtype != tch::Kind::Half {
                        cpu_tensor.to_kind(self.dtype)
                    } else {
                        cpu_tensor
                    };

                    if self.device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(self.device);
                        drop(cpu_tensor);
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                safetensors::Dtype::F32 => {
                    let cpu_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Float,
                            Device::Cpu,
                        )
                    };

                    // Convert to target dtype
                    let cpu_tensor = match self.dtype {
                        tch::Kind::Half => cpu_tensor.to_kind(tch::Kind::Half),
                        tch::Kind::BFloat16 => cpu_tensor.to_kind(tch::Kind::BFloat16),
                        tch::Kind::Float => cpu_tensor,
                        _ => cpu_tensor.to_kind(self.dtype),
                    };

                    if self.device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(self.device);
                        drop(cpu_tensor);
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                dtype => {
                    return Err(anyhow::anyhow!(
                        "Tensor '{}' has unsupported dtype {:?}. Supported: F16, BF16, F32",
                        name, dtype
                    ));
                }
            };

            weights.insert(name.to_string(), tensor);
        }

        Ok(weights)
    }
}

impl WeightProvider for StreamingWeightProvider {
    fn get_tensor(&self, name: &str) -> Result<Tensor> {
        // Check if tensor is in current shard
        let current_weights = self.current_weights.blocking_read();
        if let Some(tensor) = current_weights.get(name) {
            return Ok(tensor.shallow_clone());
        }
        drop(current_weights);

        // Find which shard contains this tensor
        let shard_idx = self
            .tensor_shard_map
            .get(name)
            .ok_or_else(|| anyhow!("Tensor {} not found in any shard", name))?;

        // Load the shard if it's not current
        let mut current_shard = self.current_shard.blocking_write();
        if current_shard.as_ref() != Some(shard_idx) {
            // Load new shard (blocking for simplicity - could be async)
            let new_weights =
                tokio::runtime::Handle::current().block_on(self.load_shard(*shard_idx))?;

            let mut current_weights = self.current_weights.blocking_write();
            *current_weights = new_weights;
            *current_shard = Some(*shard_idx);
        }
        drop(current_shard);

        // Now get the tensor from loaded shard
        let current_weights = self.current_weights.blocking_read();
        current_weights
            .get(name)
            .map(|t| t.shallow_clone())
            .ok_or_else(|| anyhow!("Tensor {} not found after loading shard", name))
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_shard_map.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensor_shard_map.keys().cloned().collect()
    }

    fn preload(&mut self, names: &[&str]) -> Result<()> {
        // Find all shards needed for these tensors
        let mut needed_shards: Vec<usize> = names
            .iter()
            .filter_map(|name| self.tensor_shard_map.get(*name))
            .copied()
            .collect();
        needed_shards.sort_unstable();
        needed_shards.dedup();

        // For now, just load the first needed shard
        if let Some(shard_idx) = needed_shards.first() {
            let new_weights =
                tokio::runtime::Handle::current().block_on(self.load_shard(*shard_idx))?;

            let mut current_weights = self.current_weights.blocking_write();
            *current_weights = new_weights;
            let mut current_shard = self.current_shard.blocking_write();
            *current_shard = Some(*shard_idx);
        }

        Ok(())
    }
}

struct SafeTensorsHeader {
    tensor_names: Vec<String>,
}
