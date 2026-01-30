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
pub mod paths;

// Re-export types for backward compatibility
pub use adapter_manager::{AdapterConfig, AdapterInfo, AdapterManager};
pub use errors::{ModelRefError, ModelRefResult};
pub use model_ref::{validate_model_name, GitRef, ModelRef};
pub use paths::StoragePaths;

// ============================================================================
// Compatibility types (formerly in model_storage.rs)
// ============================================================================

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a model
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(Uuid);

impl ModelId {
    /// Create a new random model ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ModelId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata for a cloned/tracked model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Display name (may include version/branch info)
    pub display_name: Option<String>,
    /// Model type (e.g., "language_model", "embedding")
    pub model_type: String,
    /// When the model was first tracked
    pub created_at: i64,
    /// When the model was last updated
    pub updated_at: i64,
    /// Size in bytes (if known)
    pub size_bytes: Option<u64>,
    /// Tags for metadata enrichment
    pub tags: Vec<String>,
    /// Whether the model has uncommitted changes
    pub is_dirty: bool,
}

/// Result of a clone operation
#[derive(Debug, Clone)]
pub struct ClonedModel {
    /// Unique identifier for this model
    pub id: ModelId,
    /// Model name
    pub name: String,
    /// Path to the cloned model
    pub path: std::path::PathBuf,
}

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

