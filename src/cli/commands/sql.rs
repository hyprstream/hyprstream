use clap::Args;

#[derive(Args)]
#[command(disable_help_flag = true)]
pub struct SqlCommand {
    /// Server host:port address (e.g. localhost:8080)
    #[arg(short = 'h', long = "host", value_name = "HOST:PORT")]
    pub host: Option<String>,

    /// SQL query to execute
    #[arg(short = 'c', required = true, value_name = "QUERY")]
    pub query: String,

    /// Path to client certificate for mTLS
    #[arg(long = "tls-cert", value_name = "FILE")]
    pub tls_cert: Option<std::path::PathBuf>,

    /// Path to client private key for mTLS
    #[arg(long = "tls-key", value_name = "FILE")]
    pub tls_key: Option<std::path::PathBuf>,

    /// Path to CA certificate for server verification
    #[arg(long = "tls-ca", value_name = "FILE")]
    pub tls_ca: Option<std::path::PathBuf>,

    /// Skip TLS certificate verification
    #[arg(long = "tls-skip-verify")]
    pub tls_skip_verify: bool,

    /// Print help information
    #[arg(long = "help", action = clap::ArgAction::Help)]
    pub help: Option<bool>,

    /// Enable verbose output
    #[arg(short = 'v', long = "verbose")]
    pub verbose: bool,
}
