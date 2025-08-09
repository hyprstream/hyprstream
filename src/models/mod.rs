//! Model abstractions that use the unified runtime system

// Re-export the unified config system
pub use crate::config::{
    HyprConfig, ModelConfig, RuntimeConfig, GenerationConfig, LoRAConfig,
    ModelInfo, GenerationRequest, GenerationResult, FinishReason
};

// Re-export the runtime engine - models are just wrappers around this
pub use crate::runtime::{RuntimeEngine, LlamaCppEngine, create_engine};

pub mod model_registry;
pub mod qwen3;

pub use model_registry::ModelRegistry;
pub use qwen3::Qwen3Wrapper;