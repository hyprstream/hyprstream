//! Authentication management commands
//!
//! This module provides commands for managing authentication tokens
//! for various providers like HuggingFace Hub.

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct AuthCommand {
    #[command(subcommand)]
    pub action: AuthAction,
}

#[derive(Subcommand)]
pub enum AuthAction {
    /// Login to a provider (HuggingFace Hub)
    Login {
        /// Provider to login to
        #[arg(short, long, default_value = "huggingface")]
        provider: String,
        
        /// Authentication token (if not provided, will prompt)
        #[arg(short, long)]
        token: Option<String>,
        
        /// Read token from stdin
        #[arg(long)]
        stdin: bool,
    },
    
    /// Show authentication status
    Status {
        /// Provider to check status for
        #[arg(short, long, default_value = "huggingface")]
        provider: String,
    },
    
    /// Logout from a provider
    Logout {
        /// Provider to logout from
        #[arg(short, long, default_value = "huggingface")]
        provider: String,
    },
    
    /// List supported providers
    Providers,
}