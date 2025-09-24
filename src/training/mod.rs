//! Training utilities and checkpoint management

pub mod checkpoint;

pub use checkpoint::{
    CheckpointManager,
    CheckpointRequest,
    WeightSnapshot,
    WeightFormat,
    TrainingMetrics,
    CheckpointConfig,
    CheckpointInfo,
};