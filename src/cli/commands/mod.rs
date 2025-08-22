pub mod config;
pub mod server;
pub mod model;
pub mod lora;
pub mod quick_start;
pub mod download;
pub mod auth;
pub mod chat;

pub use server::{CacheConfig, EngineConfig, ServerCommand, ServerConfig};
pub use model::ModelCommand;
pub use lora::LoRACommand;
pub use quick_start::QuickStartCommand;
pub use auth::AuthCommand;
pub use chat::ChatCommand;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),
    /// Manage models from registries (HuggingFace, etc.)
    Model(ModelCommand),
    /// Manage LoRA adapters and training
    Lora(LoRACommand),
    /// Quick start demo with llama.cpp and LoRA
    QuickStart(QuickStartCommand),
    /// Manage authentication for providers
    Auth(AuthCommand),
    /// Chat with a model or composed model
    Chat(ChatCommand),
}
