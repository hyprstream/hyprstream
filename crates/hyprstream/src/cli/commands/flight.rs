//! Flight SQL client command
//!
//! Provides `hyprstream flight` subcommand for querying datasets via Flight SQL.

use clap::Args;

/// Flight SQL client arguments
#[derive(Args, Debug)]
pub struct FlightArgs {
    /// Server address to connect to
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub addr: String,

    /// SQL query to execute (interactive mode if not provided)
    #[arg(short, long)]
    pub query: Option<String>,

    /// Output format (table, csv, json)
    #[arg(long, default_value = "table")]
    pub format: String,
}
