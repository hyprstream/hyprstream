pub mod config;
pub mod server;

pub use server::{ServerCommand, ServerConfig, EngineConfig, CacheConfig};

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),
}