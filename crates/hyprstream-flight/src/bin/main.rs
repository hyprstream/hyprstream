//! Flight SQL server binary for hyprstream-metrics.

use clap::{Parser, Subcommand};
use config::Config;
use hyprstream_flight::cli::handlers::{handle_server, handle_config};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(author, version, about = "Flight SQL server for hyprstream-metrics")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (can be specified multiple times)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Flight SQL server
    Serve {
        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value = "50051")]
        port: u16,

        /// DuckDB connection string (use :memory: for in-memory)
        #[arg(long, default_value = ":memory:")]
        connection: String,

        /// Path to TLS certificate file
        #[arg(long)]
        tls_cert: Option<PathBuf>,

        /// Path to TLS private key file
        #[arg(long)]
        tls_key: Option<PathBuf>,

        /// Path to TLS CA certificate file for client authentication
        #[arg(long)]
        tls_ca: Option<PathBuf>,

        /// Path to configuration file
        #[arg(long, short)]
        config: Option<PathBuf>,
    },

    /// Check configuration
    Config {
        /// Path to configuration file
        #[arg(long, short)]
        config: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize logging
    let level = match cli.verbose {
        0 => Level::INFO,
        1 => Level::DEBUG,
        _ => Level::TRACE,
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(level.into())
                .parse_lossy("hyprstream_flight=debug,hyprstream_metrics=debug")
        )
        .with_target(true)
        .init();

    info!("hyprstream-flight starting up");

    match cli.command {
        Commands::Serve {
            host,
            port,
            connection,
            tls_cert,
            tls_key,
            tls_ca,
            config: config_path,
        } => {
            let mut builder = Config::builder()
                .set_default("host", host)?
                .set_default("port", port.to_string())?
                .set_default("storage.type", "duckdb")?
                .set_default("storage.connection", connection)?;

            // Load config file if provided
            if let Some(path) = config_path {
                builder = builder.add_source(config::File::from(path));
            }

            // Set TLS configuration from command line args
            if let Some(ref cert) = tls_cert {
                builder = builder
                    .set_default("tls.enabled", "true")?
                    .set_default("tls.cert_path", cert.to_str().unwrap())?;
            }
            if let Some(ref key) = tls_key {
                builder = builder
                    .set_default("tls.key_path", key.to_str().unwrap())?;
            }
            if let Some(ref ca) = tls_ca {
                builder = builder
                    .set_default("tls.ca_path", ca.to_str().unwrap())?;
            }

            let config = builder.build()?;
            handle_server(config).await?
        }

        Commands::Config { config } => {
            handle_config(config)?
        }
    }

    Ok(())
}
