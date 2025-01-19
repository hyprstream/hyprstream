pub mod config;
pub mod server;

pub use server::{CacheConfig, EngineConfig, ServerCommand, ServerConfig};

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),
}
