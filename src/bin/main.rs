//! Hyprstream binary.
//!
//! This binary provides the main entry point for the Hyprstream service, a next-generation application
//! for real-time data ingestion, windowed aggregation, caching, and serving.

use clap::Parser;
use config::Config;
use hyprstream_core::{
    cli::commands::Commands,
    cli::handlers::{handle_server, handle_embedding_query, handle_model_command, handle_lora_command, handle_auth_command},
};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Get logging config from command
    let (level, filter) = match &cli.command {
        Commands::Server(cmd) => (&cmd.logging.get_effective_level(), &cmd.logging.log_filter),
        Commands::Sql(cmd) => (&cmd.logging.get_effective_level(), &cmd.logging.log_filter),
        Commands::Model(_) => (&"info", &None),
        Commands::Lora(_) => (&"info", &None),
        Commands::QuickStart(_) => (&"info", &None),
        Commands::Auth(_) => (&"info", &None),
    };

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(level.parse().unwrap_or(Level::INFO).into())
                .parse_lossy(filter.as_deref().unwrap_or("hyprstream_core=debug"))
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("Hyprstream starting up");

    let cli = Cli::parse();

    match cli.command {
        Commands::Server(cmd) => {
            let mut config = Config::builder()
                .set_default("host", "127.0.0.1")?
                .set_default("port", "50051")?
                .set_default("storage.path", "./vdb_storage")?
                .set_default("storage.neural_compression", true)?
                .set_default("storage.hardware_acceleration", true)?
                .set_default("storage.cache_size_mb", 2048)?;

            // Set TLS configuration from command line args
            if cmd.server.tls_cert.is_some() {
                config = config
                    .set_default("tls.enabled", "true")?
                    .set_default("tls.cert_path", cmd.server.tls_cert.unwrap().to_str().unwrap())?;
            }
            if cmd.server.tls_key.is_some() {
                config = config
                    .set_default("tls.key_path", cmd.server.tls_key.unwrap().to_str().unwrap())?;
            }
            if cmd.server.tls_client_ca.is_some() {
                config = config
                    .set_default("tls.ca_path", cmd.server.tls_client_ca.unwrap().to_str().unwrap())?;
            }

            let config = config.build()?;
            handle_server(config).await?
        }
        Commands::Sql(cmd) => {
            handle_embedding_query(
                cmd.host,
                &cmd.query,
                cmd.tls_cert.as_deref(),
                cmd.tls_key.as_deref(),
                cmd.tls_ca.as_deref(),
                cmd.tls_skip_verify,
                cmd.logging.verbose > 0,
            ).await?
        }
        Commands::Model(cmd) => {
            let server_url = "http://localhost:50051".to_string(); // Default server URL
            handle_model_command(cmd, server_url).await?
        }
        Commands::Lora(cmd) => {
            let server_url = "http://localhost:50051".to_string(); // Default server URL
            handle_lora_command(cmd, server_url).await?
        }
        Commands::QuickStart(cmd) => {
            hyprstream_core::cli::commands::quick_start::handle_quick_start(cmd).await?
        }
        Commands::Auth(cmd) => {
            handle_auth_command(cmd).await?
        }
    }

    Ok(())
}
