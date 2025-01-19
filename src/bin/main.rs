//! Hyprstream server binary.
//!
//! This binary provides the main entry point for the Hyprstream service, a next-generation application
//! for real-time data ingestion, windowed aggregation, caching, and serving.

use clap::Parser;
use hyprstream_core::{
    cli::{Cli, run_server},
    config::Settings,
};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize settings
    let settings = Settings::new(cli.args)?;

    // Initialize tracing
    let subscriber = fmt()
        .with_env_filter(EnvFilter::new(&settings.server.log_level))
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    // Run the server
    run_server(cli.detach, settings).await?;

    Ok(())
}
