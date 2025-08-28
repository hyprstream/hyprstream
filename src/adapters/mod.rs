//! Sparse adaptive layer implementations
//!
//! This module provides various types of sparse adapters for fine-tuning
//! large language models with minimal memory overhead and real-time updates.

pub mod sparse_lora;
pub mod openvdb_lora;
pub mod lora_checkpoints;

pub use sparse_lora::*;
pub use openvdb_lora::*;
pub use lora_checkpoints::*;