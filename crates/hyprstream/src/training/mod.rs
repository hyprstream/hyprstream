//! Training utilities and checkpoint management
//!
//! This module provides:
//! - `ttt`: Test-Time Training for research-valid model adaptation
//! - `quality_filter`: Heuristic-based output filtering (NOT for training)
//! - `checkpoint`: Checkpoint management with git integration
//! - `data_loader`: Training data loading utilities
//! - `lora_trainer`: LoRA fine-tuning

pub mod checkpoint;
pub mod data_loader;
pub mod lora_trainer;
pub mod quality_filter;
pub mod ttt;

pub use checkpoint::{
    CheckpointConfig, CheckpointInfo, CheckpointManager, CheckpointRequest, TrainingMetrics,
    WeightFormat, WeightSnapshot,
};

pub use data_loader::{ChatTemplateDataLoader, TrainingDataset, TrainingSample};

pub use lora_trainer::{LoRATrainer, LoRATrainingConfig};

pub use quality_filter::{FilterReason, FilterResult, QualityFilter, QualityFilterConfig};

pub use ttt::{TTTConfig, TTTContext, TTTResult, TTTVerifier, TestTimeTrainer};
