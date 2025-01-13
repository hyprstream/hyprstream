mod server;
mod models;
mod chat;
mod sql;

pub use server::ServerCommand;
pub use models::ModelCommands;
pub use chat::ChatCommand;
pub use sql::SqlCommand;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),

    /// Model management commands
    Models {
        #[command(subcommand)]
        command: ModelCommands,
    },

    /// Interactive chat with a model
    Chat(ChatCommand),

    /// Execute SQL queries
    Sql(SqlCommand),
}
