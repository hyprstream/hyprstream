//! Adapter implementations for storage and caching
//!
//! This module provides checkpoint management for LoRA adapters
//! The main LoRA implementations have been moved to src/lora/

pub mod lora_checkpoints;

pub use lora_checkpoints::*;