//! REST API for creating and managing sparse auto-regressive LoRA training layers

pub mod openai_compat;
pub mod training_service;
pub mod tools;

// Re-export training types
pub use training_service::{TrainingSample, TrainingStatus};

