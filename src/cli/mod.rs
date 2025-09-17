//! Command-line interface module.
//!
//! This module provides the CLI functionality for:
//! - Server management
//! - Configuration handling
//! - Model management (HuggingFace, etc.)
//! - LoRA adapter creation and training

pub mod commands;
pub mod handlers;
pub mod pytorch_lora_handler;

pub use handlers::{handle_config, handle_server, handle_model_command, handle_chat_command};
pub use pytorch_lora_handler::handle_pytorch_lora_command;
