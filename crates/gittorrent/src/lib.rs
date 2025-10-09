//! GitTorrent - Decentralized Git hosting using libp2p
//!
//! This library implements a peer-to-peer network for sharing Git repositories
//! using libp2p and a Kademlia distributed hash table (DHT).

pub mod crypto;
pub mod dht;
pub mod error;
pub mod git;
// Using libp2p for P2P networking directly in DHT module
pub mod types;
pub mod xet_integration;
pub mod service;
pub mod lfs_xet;
pub mod daemon;

// Re-export commonly used types
pub use error::{Error, Result};
pub use types::*;

// Re-export transport functionality for consuming applications
pub use git::transport::{GittorrentTransportFactory, TransportFactory, register_gittorrent_transport};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize tracing for the library
pub fn init_tracing() -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gittorrent=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .try_init()
        .map_err(|_| Error::other("Tracing already initialized"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}