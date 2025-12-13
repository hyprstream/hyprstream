//! Command-line interface module.
//!
//! This module provides the CLI functionality for:
//! - Server management
//! - Configuration handling

pub mod handlers;

pub use handlers::{handle_config, handle_server};
