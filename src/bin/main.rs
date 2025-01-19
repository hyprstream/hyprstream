//! Hyprstream binary.
//!
//! This binary provides the main entry point for the Hyprstream service, a next-generation application
//! for real-time data ingestion, windowed aggregation, caching, and serving.

use clap::Parser;
use hyprstream_core::{
    cli::{run_server, Cli, Commands},
    config::Settings,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Handle commands
    match cli.command {
        Commands::Server(server_cmd) => {
            // Initialize settings
            let settings = Settings::new(
                server_cmd.server,
                server_cmd.engine,
                server_cmd.cache,
                server_cmd.config,
            )?;

            run_server(server_cmd.detach, settings).await?;
        }
    }

    Ok(())
}
