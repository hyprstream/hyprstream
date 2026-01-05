pub mod config;
pub mod server;
pub mod sql;

pub use server::{CacheConfig, EngineConfig, ServerCommand, ServerConfig};
pub use sql::SqlCommand;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),
    /// Execute a SQL query
    Sql(SqlCommand),
}
