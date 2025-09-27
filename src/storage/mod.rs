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
pub mod adapter_manager;
pub mod errors;
pub mod domain_types;
pub mod repository_cache;
pub mod repository_patterns;
pub mod unified_repository;
pub mod registry_repair;

// XDG-compliant path management (internal use only)
pub use paths::StoragePaths;
pub use xet_native::{XetNativeStorage, XetConfig};
pub use model_ref::{ModelRef, GitRef, validate_model_name};
pub use model_registry::{ModelRegistry, SharedModelRegistry, ModelStatus, CheckoutOptions, CheckoutResult, ModelInfo};
pub use model_storage::{ModelStorage, ModelId, ModelMetadata, ModelMetadataFile};
pub use git_source::GitModelSource;
pub use sharing::{ModelSharing, ShareableModelRef, ModelType};
pub use adapter_manager::{AdapterManager, AdapterInfo, AdapterConfig};
pub use errors::{StorageError, ModelRefError, GitOperationError, StorageResult, ModelRefResult, GitOperationResult};
pub use domain_types::{ModelName, BranchName, TagName, RevSpec, AdapterName, RemoteName};
pub use repository_cache::{RepositoryCache, RepositoryCacheConfig, CacheStats, local_repository_cache, get_cached_repository};
pub use repository_patterns::{
    RepositoryHandle, SubmoduleInfo, ReferenceInfo, CommitInfo,
    RepositoryOperation, RepositoryOperationBuilder, CachedRepositoryFactory,
    Unmodified, Modified, Committed
};
pub use unified_repository::{
    ModelRepository, UnifiedModelRepository, UnifiedModelInfo, ModelSource, SyncReport
};
pub use registry_repair::{
    RegistryRepair, RepairReport, repair_registry
};
