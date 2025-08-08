pub mod config;
pub mod server;
pub mod sql;
pub mod model;
pub mod lora;
pub mod quick_start;
pub mod download;
pub mod auth;

pub use server::{CacheConfig, EngineConfig, ServerCommand, ServerConfig};
pub use sql::SqlCommand;
pub use model::ModelCommand;
pub use lora::LoRACommand;
pub use quick_start::QuickStartCommand;
pub use auth::AuthCommand;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),
    /// Execute a SQL query
    Sql(SqlCommand),
    /// Manage models from registries (HuggingFace, Ollama, etc.)
    Model(ModelCommand),
    /// Manage LoRA adapters and training
    Lora(LoRACommand),
    /// Quick start demo with llama.cpp and LoRA
    QuickStart(QuickStartCommand),
    /// Manage authentication for providers
    Auth(AuthCommand),
}
