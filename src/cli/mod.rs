//! Command-line interface module.
//!
//! This module provides the CLI functionality for:
//! - Server management
//! - Configuration handling
//! - SQL query execution

pub mod commands;
pub mod handlers;

pub use handlers::{handle_config, handle_server};
