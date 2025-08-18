//! Command-line interface module.
//!
//! This module provides the CLI functionality for:
//! - Server management
//! - Configuration handling
//! - SQL query execution
//! - Model management (HuggingFace, Ollama, etc.)
//! - LoRA adapter creation and training

pub mod commands;
pub mod handlers;

pub use handlers::{handle_config, handle_server, handle_model_command, handle_lora_command, handle_chat_command};
