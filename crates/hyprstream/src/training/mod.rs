//! Training utilities and checkpoint management
//!
//! This module provides:
//! - `ttt`: Test-Time Training for research-valid model adaptation
//! - `tenant_delta`: Per-tenant LoRA delta for isolated TTT
//! - `delta_pool`: Per-tenant delta registry with LRU eviction
//! - `merge`: LoRA adapter merging strategies (DO-Merge, additive, replace)
//! - `quality_filter`: Heuristic-based output filtering (NOT for training)
//! - `checkpoint`: Checkpoint management with git integration
//! - `data_loader`: Training data loading utilities

pub mod checkpoint;
pub mod data_loader;
pub mod delta_pool;
pub mod merge;
pub mod quality_filter;
pub mod tenant_delta;
pub mod ttt;

pub use checkpoint::{
    CheckpointConfig, CheckpointInfo, CheckpointManager, CheckpointRequest, TrainingMetrics,
    WeightFormat, WeightSnapshot,
};

pub use data_loader::{ChatTemplateDataLoader, TrainingDataset, TrainingSample};

pub use delta_pool::DeltaPool;

pub use quality_filter::{FilterReason, FilterResult, QualityFilter, QualityFilterConfig};

pub use tenant_delta::{TenantDelta, TenantDeltaConfig, SharedTenantDelta, serialize_state_dict_to_bytes, load_state_dict_from_bytes};

pub use merge::{MergeStrategy, merge_state_dicts};

pub use ttt::{TTTConfig, TTTContext, TTTOverrides, TTTResult, TTTVerifier, TestTimeTrainer};
