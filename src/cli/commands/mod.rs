pub mod config;
pub mod server;
pub mod model;
pub mod lora;
pub mod chat;

pub use server::{CacheConfig, EngineConfig, ServerCommand, ServerConfig};
pub use model::ModelCommand;
pub use lora::LoRACommand;
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
    /// Chat with a model or composed model
    Chat(ChatCommand),
}
