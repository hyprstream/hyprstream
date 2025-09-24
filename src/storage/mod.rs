//! Storage for model weights and adapters
//!
//! This module provides storage for:
//! - Neural network weights
//! - LoRA adapters
//! - Model checkpoints
//! - Memory-mapped disk persistence
//! - Xet-based content-addressable storage
//! - Git-native model registry
pub mod paths;
pub mod xet_native;
pub mod model_ref;
pub mod model_registry;
pub mod model_storage;
pub mod git_source;
pub mod sharing;
pub mod operations;

// XDG-compliant path management (internal use only)
pub use paths::StoragePaths;
pub use xet_native::{XetNativeStorage, XetConfig};
pub use model_ref::{ModelRef, validate_model_name};
pub use model_registry::{ModelRegistry, SharedModelRegistry};
pub use model_storage::{ModelStorage, ModelId, ModelMetadata, ModelMetadataFile};
pub use git_source::GitModelSource;
pub use sharing::{ModelSharing, ShareableModelRef, ModelType};
