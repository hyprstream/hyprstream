//! Server CLI commands and arguments
//!
//! This module defines the CLI interface for the server command.
//! Configuration logic is in crate::config module.

use super::config::LoggingConfig;
use super::KVQuantArg;
use crate::config::ServerConfigBuilder;
use clap::Args;
use serde::Deserialize;
use std::path::PathBuf;

/// CLI arguments for server configuration (for clap integration)
#[derive(Debug, Default, Deserialize, Args)]
pub struct ServerCliArgs {
    /// Server host address
    #[arg(long, env = "HYPRSTREAM_SERVER_HOST")]
    pub host: Option<String>,

    /// Server port
    #[arg(long, env = "HYPRSTREAM_SERVER_PORT")]
    pub port: Option<u16>,

    /// Working directory for the server when running in detached mode
    #[arg(long, env = "HYPRSTREAM_WORKING_DIR")]
    pub working_dir: Option<String>,

    /// PID file location when running in detached mode
    #[arg(long, env = "HYPRSTREAM_PID_FILE")]
    pub pid_file: Option<String>,

    /// Path to TLS certificate file
    #[arg(long = "tls-cert", env = "HYPRSTREAM_TLS_CERT")]
    pub tls_cert: Option<PathBuf>,

    /// Path to TLS private key file
    #[arg(long = "tls-key", env = "HYPRSTREAM_TLS_KEY")]
    pub tls_key: Option<PathBuf>,

    /// Path to CA certificate for client authentication (enables mTLS)
    #[arg(long = "tls-client-ca", env = "HYPRSTREAM_TLS_CLIENT_CA")]
    pub tls_client_ca: Option<PathBuf>,

    /// Minimum TLS version (1.2|1.3)
    #[arg(long, env = "HYPRSTREAM_TLS_MIN_VERSION")]
    pub tls_min_version: Option<String>,

    /// Allowed TLS cipher suites
    #[arg(long, env = "HYPRSTREAM_TLS_CIPHER_LIST")]
    pub tls_cipher_list: Option<String>,

    /// Prefer server cipher order
    #[arg(long, env = "HYPRSTREAM_TLS_PREFER_SERVER_CIPHERS")]
    pub tls_prefer_server_ciphers: Option<bool>,

    /// Allow all headers in CORS (permissive mode for development - NOT recommended for production)
    #[arg(long, env = "HYPRSTREAM_CORS_PERMISSIVE_HEADERS")]
    pub cors_permissive_headers: Option<bool>,

    /// Maximum context length for KV cache allocation.
    /// Overrides model's max_position_embeddings to reduce GPU memory usage.
    /// Lower values = less memory (e.g., 2048 instead of 40960 can save ~18GB on Qwen3-4B).
    #[arg(long, env = "HYPRSTREAM_MAX_CONTEXT")]
    pub max_context: Option<usize>,

    /// KV cache quantization type for reduced GPU memory usage.
    /// Reduces GPU memory by 50-75% at slight quality cost.
    #[arg(long, value_enum, default_value = "none", env = "HYPRSTREAM_KV_QUANT")]
    #[serde(default)]
    pub kv_quant: KVQuantArg,
}

impl ServerCliArgs {
    /// Merge these CLI args into a ServerConfigBuilder
    pub fn apply_to_builder(&self, builder: ServerConfigBuilder) -> ServerConfigBuilder {
        let mut builder = builder;

        if let Some(ref host) = self.host {
            builder = builder.host(host.clone());
        }
        if let Some(port) = self.port {
            builder = builder.port(port);
        }
        if let Some(ref cert) = self.tls_cert {
            builder = builder.tls_cert(cert.clone());
        }
        if let Some(ref key) = self.tls_key {
            builder = builder.tls_key(key.clone());
        }
        if let Some(ref ca) = self.tls_client_ca {
            builder = builder.tls_client_ca(ca.clone());
        }
        if let Some(ref version) = self.tls_min_version {
            builder = builder.tls_min_version(version.clone());
        }
        if let Some(ref ciphers) = self.tls_cipher_list {
            builder = builder.tls_cipher_list(ciphers.clone());
        }
        if let Some(prefer) = self.tls_prefer_server_ciphers {
            builder = builder.tls_prefer_server_ciphers(prefer);
        }
        if let Some(permissive) = self.cors_permissive_headers {
            builder = builder.cors_permissive_headers(permissive);
        }
        if let Some(ref dir) = self.working_dir {
            builder = builder.working_dir(PathBuf::from(dir));
        }
        if let Some(ref file) = self.pid_file {
            builder = builder.pid_file(PathBuf::from(file));
        }
        if let Some(max_context) = self.max_context {
            builder = builder.max_context(Some(max_context));
        }

        // Always apply kv_quant (has default value)
        builder = builder.kv_quant(self.kv_quant);

        builder
    }
}

/// Main server command
#[derive(Debug, Args)]
pub struct ServerCommand {
    /// Run server in detached mode
    #[arg(short = 'd', long = "detach")]
    pub detach: bool,

    /// Path to the configuration file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Dataset name to serve via Flight SQL (enables Flight SQL server)
    #[arg(long, env = "HYPRSTREAM_DATASET")]
    pub dataset: Option<String>,

    /// Flight SQL server port (default: 50051)
    #[arg(long, env = "HYPRSTREAM_FLIGHT_PORT", default_value = "50051")]
    pub flight_port: u16,

    /// Flight SQL server host (default: same as HTTP server host)
    #[arg(long, env = "HYPRSTREAM_FLIGHT_HOST")]
    pub flight_host: Option<String>,

    #[command(flatten)]
    pub server: ServerCliArgs,

    #[command(flatten)]
    pub logging: LoggingConfig,
}
