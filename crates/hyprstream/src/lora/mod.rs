//! Clean LoRA implementation for transformer models
//!
//! Single source of truth for LoRA adapters with git-based storage.

pub mod torch_adapter;
pub mod trainer;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tch::Tensor;

/// Single LoRA configuration - only source of truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Low-rank dimension (r in the paper)
    pub rank: usize,
    /// Scaling factor (alpha in the paper)
    pub alpha: f32,
    /// Dropout probability for training
    pub dropout: f32,
    /// Module names to apply LoRA to
    pub target_modules: Vec<String>,
    /// Learning rate for training
    pub learning_rate: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_owned(), "v_proj".to_owned()],
            learning_rate: 1e-4,
        }
    }
}

/// LoRA adapter trait - single interface
#[async_trait]
pub trait LoRAAdapter: Send {
    /// Get adapter configuration
    fn config(&self) -> &LoRAConfig;

    /// Save weights to SafeTensors
    async fn save(&self, path: &Path) -> Result<()>;

    /// Load weights from SafeTensors
    async fn load(&mut self, path: &Path) -> Result<()>;

    /// Forward pass for a module
    fn forward(&self, module_name: &str, input: &Tensor) -> Result<Option<Tensor>>;

    /// Get number of parameters
    fn num_parameters(&self) -> i64;
}

// Re-export the torch implementation
pub use torch_adapter::{LoRALayerConfig, LoRAModel, TorchLoRALayer};
pub use trainer::{CheckpointMetrics, LoRATrainer, TrainingConfig};
