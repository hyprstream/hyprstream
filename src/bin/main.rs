//! Hyprstream binary.
//!
//! This binary provides the main entry point for the Hyprstream service, a next-generation application
//! for real-time data ingestion, windowed aggregation, caching, and serving.

use clap::Parser;
use hyprstream_core::{
    cli::{Cli, Commands, run_server},
    config::{Settings, CliArgs},
};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Handle commands
    match cli.command {
        Commands::Server(server_cmd) => {
            // Convert server command args to CliArgs
            let cli_args = CliArgs::from(&server_cmd);
            
            // Initialize settings
            let settings = Settings::new(cli_args)?;

            // Initialize tracing
            let subscriber = fmt()
                .with_env_filter(EnvFilter::new(&settings.server.log_level))
                .finish();
            let _guard = tracing::subscriber::set_default(subscriber);

            run_server(server_cmd.detach, settings).await?;
        }
    }

    Ok(())
}
