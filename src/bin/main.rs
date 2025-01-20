//! Hyprstream binary.
//!
//! This binary provides the main entry point for the Hyprstream service, a next-generation application
//! for real-time data ingestion, windowed aggregation, caching, and serving.

use clap::Parser;
use hyprstream_core::{
    cli::{execute_sql, run_server, Cli, Commands},
    config::Settings,
};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize logging with debug level for hyprstream_core
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
                .parse_lossy("hyprstream_core=debug"),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("Hyprstream starting up");

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
        Commands::Sql(sql_cmd) => {
            let addr = if let Some(host) = sql_cmd.host {
                Some(host.parse().map_err(|e| {
                    format!("Invalid host:port address '{}': {}", host, e)
                })?)
            } else {
                None
            };
            
            execute_sql(
                addr,
                sql_cmd.query,
                sql_cmd.tls_cert.as_deref(),
                sql_cmd.tls_key.as_deref(),
                sql_cmd.tls_ca.as_deref(),
                sql_cmd.tls_skip_verify,
                sql_cmd.verbose,
            )
            .await?;
        }
    }

    Ok(())
}
