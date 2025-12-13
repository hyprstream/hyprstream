//! Checkpoint system for versioned metrics storage.
//!
//! This module provides git2db-based checkpointing with:
//! - DuckDB table export to Parquet format
//! - Git-based versioning and recovery
//! - Background checkpoint scheduling
//! - Startup recovery from latest checkpoint

pub mod manager;
pub mod recovery;
pub mod state;

pub use manager::{CheckpointConfig, CheckpointManager};
pub use recovery::{RecoveryManager, RecoveryStatus};
pub use state::{Checkpoint, CheckpointMetadata};
