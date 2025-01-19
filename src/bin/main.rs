//! Hyprstream binary.
//!
//! This binary provides the main entry point for the Hyprstream service, a next-generation application
//! for real-time data ingestion, windowed aggregation, caching, and serving.

use clap::Parser;
use hyprstream_core::{
    cli::{execute_sql, run_server, Cli, Commands},
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
        Commands::Sql(sql_cmd) => {
            execute_sql(
                sql_cmd.host,
                sql_cmd.query,
                sql_cmd.tls_cert.as_deref(),
                sql_cmd.tls_key.as_deref(),
                sql_cmd.tls_ca.as_deref(),
                sql_cmd.tls_skip_verify,
            )
            .await?;
        }
    }

    Ok(())
}
