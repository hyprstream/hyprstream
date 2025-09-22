//! Storage for model weights and adapters
//!
//! This module provides storage for:
//! - Neural network weights
//! - LoRA adapters
//! - Model checkpoints
//! - Memory-mapped disk persistence
//! - Xet-based content-addressable storage
pub mod paths;
pub mod xet_native;

// XDG-compliant path management (internal use only)
pub use paths::StoragePaths;
pub use xet_native::{XetNativeStorage, XetConfig};
