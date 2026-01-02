//! Storage for model weights and adapters
//!
//! This module provides storage for:
//! - Neural network weights
//! - LoRA adapters
//! - Model checkpoints
//! - Memory-mapped disk persistence
//! - XET-based content-addressable storage (via git2db, requires `xet` feature)
//! - Git LFS to XET translation for Hugging Face models (via git2db::lfs, requires `xet` feature)
//! - Git-native model registry
pub mod adapter_manager;
pub mod errors;
pub mod model_ref;
pub mod model_storage;
pub mod paths;

// Re-export types for backward compatibility
pub use adapter_manager::{AdapterConfig, AdapterInfo, AdapterManager};
pub use errors::{ModelRefError, ModelRefResult};
pub use model_ref::{validate_model_name, GitRef, ModelRef};
pub use model_storage::{ClonedModel, ModelId, ModelMetadata, ModelStorage};
pub use paths::StoragePaths;

// Compatibility types (moved from model_registry)
/// Options for checkout operations
#[derive(Debug, Default)]
pub struct CheckoutOptions {
    pub create_branch: bool,
    pub force: bool,
}

/// Result of a checkout operation
#[derive(Debug, Clone)]
pub struct CheckoutResult {
    pub previous_oid: git2db::Oid,
    pub new_oid: git2db::Oid,
    pub previous_ref_name: Option<String>,
    pub new_ref_name: Option<String>,
    pub was_forced: bool,
    pub files_changed: usize,
    pub has_submodule: bool,
}

