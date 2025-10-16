//! Storage for model weights and adapters
//!
//! This module provides storage for:
//! - Neural network weights
//! - LoRA adapters
//! - Model checkpoints
//! - Memory-mapped disk persistence
//! - Xet-based content-addressable storage
//! - Git-native model registry
pub mod adapter_manager;
pub mod errors;
pub mod model_ref;
pub mod model_storage;
pub mod operations;
pub mod paths;
pub mod xet_native;

// Re-export types for backward compatibility
pub use adapter_manager::{AdapterConfig, AdapterInfo, AdapterManager};
pub use errors::{ModelRefError, ModelRefResult};
pub use model_ref::{validate_model_name, GitRef, ModelRef};
pub use model_storage::{ModelId, ModelMetadata, ModelStorage};
pub use paths::StoragePaths;
pub use xet_native::{XetConfig, XetNativeStorage};

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

