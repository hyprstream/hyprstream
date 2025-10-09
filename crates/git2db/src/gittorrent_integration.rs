//! GitTorrent transport integration for git2db
//!
//! This module provides integration between git2db's transport system
//! and the gittorrent-rs P2P git transport.
//!
//! Configuration is loaded automatically using `GitTorrentConfig::load()` with priority:
//! 1. `GITTORRENT_*` environment variables
//! 2. `~/.config/gittorrent/git-remote-gittorrent.*` config file
//! 3. Default values

use crate::transport::TransportFactory;
use git2::transport::SmartSubtransport;
use std::sync::Arc;

/// Bridge between gittorrent's TransportFactory and git2db's TransportFactory
pub struct GittorrentTransportBridge {
    factory: gittorrent::GittorrentTransportFactory,
}

impl GittorrentTransportBridge {
    /// Create a new bridge with the given gittorrent service
    pub fn new(service: Arc<gittorrent::service::GitTorrentService>) -> Self {
        Self {
            factory: gittorrent::GittorrentTransportFactory::new(service),
        }
    }
}

impl TransportFactory for GittorrentTransportBridge {
    fn create_transport(&self, url: &str) -> anyhow::Result<Box<dyn SmartSubtransport>> {
        // Delegate to gittorrent's factory
        gittorrent::git::transport::TransportFactory::create_transport(&self.factory, url)
    }

    fn supports_url(&self, url: &str) -> bool {
        gittorrent::git::transport::TransportFactory::supports_url(&self.factory, url)
    }
}

/// Register gittorrent transport with GitManager
///
/// Accepts a `GitTorrentConfig` which can come from:
/// - `Git2DBConfig::gittorrent` (unified configuration)
/// - Custom configuration programmatically created
/// - Loaded directly via `GitTorrentConfig::load()`
///
/// Returns the service Arc for advanced usage if needed.
///
/// # Example
///
/// ```rust,ignore
/// use git2db::{Git2DBConfig, gittorrent_integration};
///
/// // Option 1: Use git2db's unified config
/// let config = Git2DBConfig::builder()
///     .load_env()
///     .gittorrent_p2p_port(4001)
///     .build()?;
///
/// let service = gittorrent_integration::register_gittorrent(
///     manager,
///     config.gittorrent
/// ).await?;
///
/// // Option 2: Load gittorrent config independently
/// let gittorrent_config = gittorrent::service::GitTorrentConfig::load()?;
/// let service = gittorrent_integration::register_gittorrent(
///     manager,
///     gittorrent_config
/// ).await?;
/// ```
pub async fn register_gittorrent(
    manager: &crate::manager::GitManager,
    config: gittorrent::service::GitTorrentConfig,
) -> anyhow::Result<Arc<gittorrent::service::GitTorrentService>> {
    let service = Arc::new(gittorrent::service::GitTorrentService::new(config).await?);
    let bridge = Arc::new(GittorrentTransportBridge::new(service.clone()));
    manager.register_transport("gittorrent", bridge)?;

    Ok(service)
}
