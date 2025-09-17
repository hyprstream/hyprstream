//! LoRA (Low-Rank Adaptation) implementation with multiple backends
//! 
//! This module provides a unified interface for LoRA with support for:
//! - PyTorch backend with full autograd support for training
//! - Sparse backend for efficient inference
//! - OpenVDB backend for caching and persistence

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::collections::HashMap;
use tch::Device;

pub mod torch_adapter;
pub mod trainer;
pub mod config;
pub mod checkpoint;
pub mod merge;
pub mod openvdb;
pub mod utils;
pub mod error;

// Re-export implementations
pub use torch_adapter::{TorchLoRALayer, LoRALayerConfig, LoRAModel, LoRAModel as PyTorchLoRA};
pub use trainer::{LoRATrainer, TrainingMetrics};
pub use config::{TrainingConfig, QuantizationConfig, QuantizationType};
pub use checkpoint::{
    CheckpointManager, CheckpointInfo, find_best_checkpoint,
};
pub use merge::{LoRAMerger, MergeStrategy};
pub use error::{LoRAError, LoRAResult};
pub use utils::{
    lora_forward, validate_lora_config,
    InitStrategy, initialize_lora_weights,
};

// ============================================================================
// Common Types - Single source of truth for all LoRA implementations
// ============================================================================

/// Unified LoRA configuration used by all backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Low-rank dimension (r in the paper)
    pub rank: usize,
    
    /// Scaling factor (alpha in the paper, typically = rank)
    pub alpha: f32,
    
    /// Dropout probability for training
    pub dropout: f32,
    
    /// Module names to apply LoRA to (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
    
    /// Learning rate (if training)
    pub learning_rate: f32,
    
    /// Optional backend-specific settings
    #[serde(default)]
    pub backend: LoRABackend,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
            ],
            learning_rate: 1e-4,
            backend: LoRABackend::default(),
        }
    }
}

/// Backend implementation choice
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LoRABackend {
    /// PyTorch with autograd (default, best for training)
    #[default]
    PyTorch,
    
    /// OpenVDB (for caching/persistence only)
    OpenVDB,
}


// ============================================================================
// Core Trait - All LoRA implementations must satisfy this interface
// ============================================================================

/// Common interface for all LoRA adapter implementations
#[async_trait]
pub trait LoRAAdapter: Send + Sync {
    /// Get adapter configuration
    fn config(&self) -> &LoRAConfig;
    
    /// Save weights to disk
    async fn save(&self, path: &Path) -> Result<()>;
    
    /// Load weights from disk
    async fn load(&mut self, path: &Path) -> Result<()>;
    
    /// Export weights to PyTorch tensors for GPU transfer
    /// Returns (lora_a, lora_b) tensors for each target module
    /// Note: Implementations that don't use PyTorch internally will create new tensors
    fn to_tensors(&self, device: tch::Device) -> Result<HashMap<String, (tch::Tensor, tch::Tensor)>>;
    
    /// Import weights from PyTorch tensors (e.g., from GPU)
    /// Takes (lora_a, lora_b) tensors for each target module
    fn from_tensors(&mut self, tensors: HashMap<String, (tch::Tensor, tch::Tensor)>) -> Result<()>;
}

// ============================================================================
// Factory for creating adapters with appropriate backend
// ============================================================================

/// Create a LoRA adapter with the specified configuration
pub fn create_adapter(
    config: LoRAConfig,
    module_configs: HashMap<String, (usize, usize)>,
) -> Result<Box<dyn LoRAAdapter>> {
    match config.backend {
        LoRABackend::PyTorch => {
            let device = Device::cuda_if_available();
            let adapter = LoRAModel::new(
                config,
                module_configs,
                device,
            )?;
            Ok(Box::new(adapter))
        }
        LoRABackend::OpenVDB => {
            let vdb_config = openvdb::OpenVDBConfig::default();
            let adapter = openvdb::OpenVDBLoRAAdapter::new(
                "default".to_string(),
                config,
                vdb_config,
            )?;
            Ok(Box::new(adapter))
        }
    }
}