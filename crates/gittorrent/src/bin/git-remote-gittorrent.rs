//! Git remote helper for gittorrent:// URLs
//!
//! This binary implements the Git remote helper protocol for gittorrent:// URLs,
//! allowing standard Git commands to work with GitTorrent repositories.
//!
//! Git invokes this automatically when encountering gittorrent:// URLs.
//! Configuration is read from environment variables and config files.

use gittorrent::{Result, git::remote_helper::run_git_remote_helper, service::GitTorrentConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Git passes the remote name and URL as command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: git-remote-gittorrent <remote-name> <url>");
        std::process::exit(1);
    }

    let _remote_name = &args[1];
    let url = &args[2];

    // Load configuration from file and environment variables
    // Uses GitTorrentConfig::load() which handles both automatically
    let mut config = GitTorrentConfig::load()?;

    // For remote helpers, enable auto-discovery by default if no bootstrap nodes
    if config.bootstrap_nodes.is_empty() {
        config.auto_discovery = true;
    }

    // Run the Git remote helper with configuration and URL
    match run_git_remote_helper(Some(config), Some(url)).await {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}