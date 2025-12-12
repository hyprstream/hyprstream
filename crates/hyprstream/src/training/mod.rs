//! Training utilities and checkpoint management

pub mod checkpoint;
pub mod data_loader;
pub mod lora_trainer;
pub mod self_supervised;

pub use checkpoint::{
    CheckpointConfig, CheckpointInfo, CheckpointManager, CheckpointRequest, TrainingMetrics,
    WeightFormat, WeightSnapshot,
};

pub use data_loader::{ChatTemplateDataLoader, TrainingDataset, TrainingSample};

pub use lora_trainer::{LoRATrainer, LoRATrainingConfig};

pub use self_supervised::{
    ReplayBuffer, ReplayBufferConfig, ReplayBufferStats, SelfSupervisedConfig,
    SelfSupervisedTrainer, TrainerStats, TrainingCycleResult, TrainingExample, TrainingStepResult,
};
