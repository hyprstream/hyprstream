//! Storage for model weights and adapters
//!
//! This module provides storage for:
//! - Neural network weights
//! - LoRA adapters
//! - Model checkpoints
//! - Memory-mapped disk persistence
pub mod paths;

// XDG-compliant path management (internal use only)
pub use paths::StoragePaths;
