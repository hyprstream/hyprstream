//! GitTorrent daemon binary

use clap::Parser;
use gittorrent::{daemon::{run_daemon, DaemonConfig}, init_tracing};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "gittorrentd")]
#[command(about = "GitTorrent daemon for sharing git repositories over P2P")]
struct Cli {
    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,

    /// Repositories directory
    #[arg(short, long)]
    repositories: Option<PathBuf>,

    /// Git protocol port
    #[arg(short, long, default_value = "9418")]
    git_port: u16,

    /// P2P/DHT listen port (0 = random port)
    #[arg(short = 'p', long, default_value = "0")]
    p2p_port: u16,

    /// Bind address
    #[arg(short, long, default_value = "127.0.0.1")]
    bind_address: String,

    /// Disable Git server
    #[arg(long)]
    no_git_server: bool,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Bootstrap nodes for P2P network (comma-separated)
    #[arg(long)]
    bootstrap: Option<String>,

    /// Run in standalone mode (don't attempt to connect to existing network)
    #[arg(long)]
    standalone: bool,
}

#[tokio::main]
async fn main() -> gittorrent::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    init_tracing()?;

    info!("Starting GitTorrent daemon v{}", gittorrent::VERSION);

    // Create daemon configuration
    let mut config = DaemonConfig::default();

    if let Some(repos_dir) = cli.repositories {
        config.repositories_dir = repos_dir;
    }

    config.git_port = cli.git_port;
    config.service.bind_address = cli.bind_address;
    config.service.p2p_port = cli.p2p_port;
    config.enable_git_server = !cli.no_git_server;

    // Parse bootstrap nodes if provided (but not in standalone mode)
    if cli.standalone {
        config.service.bootstrap_nodes.clear();
        info!("Standalone mode enabled - no bootstrap nodes will be used");
    } else if let Some(bootstrap_str) = cli.bootstrap {
        config.service.bootstrap_nodes = bootstrap_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
    }

    info!("Daemon configuration:");
    info!("  Repositories: {}", config.repositories_dir.display());
    info!("  Git server: {} (port {})",
          if config.enable_git_server { "enabled" } else { "disabled" },
          config.git_port);
    info!("  Bind address: {}", config.service.bind_address);
    info!("  P2P port: {} ({})",
          config.service.p2p_port,
          if config.service.p2p_port == 0 { "random" } else { "fixed" });
    info!("  Bootstrap nodes: {:?}", config.service.bootstrap_nodes);

    if cli.standalone {
        info!("  Mode: Standalone (first node in network)");
        info!("  Other nodes can connect using this node's listen addresses as bootstrap");
    } else if config.service.bootstrap_nodes.is_empty() {
        info!("  Note: No bootstrap nodes provided, running in standalone mode");
        info!("  Use --bootstrap to specify P2P bootstrap nodes or --standalone for first node");
    } else {
        info!("  P2P networking enabled with {} bootstrap nodes", config.service.bootstrap_nodes.len());
    }

    // Run the daemon
    run_daemon(config).await?;

    Ok(())
}