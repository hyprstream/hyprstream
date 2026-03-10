//! User management CLI commands

use clap::Subcommand;

/// User management subcommands
#[derive(Debug, Subcommand)]
pub enum UserCommand {
    /// Register a user with their Ed25519 public key
    Register {
        /// Username
        username: String,
        /// Ed25519 public key in base64 encoding
        pubkey_base64: String,
    },
    /// List all registered users
    List,
    /// Remove a registered user
    Remove {
        /// Username
        username: String,
        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },
    /// Show a user's public key
    Show {
        /// Username
        username: String,
    },
}
