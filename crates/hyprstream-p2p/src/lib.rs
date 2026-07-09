//! GitTorrent - Decentralized Git hosting over iroh-blobs + a mainline locator.
//!
//! The object plane (`put_object`/`get_object`) is backed by iroh-blobs with a
//! mainline (BEP5) DHT rendezvous locator (F1–F3, epic #880 Track F). The old
//! libp2p Kademlia stack was retired in F3 (#901).

pub mod blobs;
pub mod crypto;
pub mod error;
pub mod git;
pub mod types;
pub mod service;
pub mod daemon;

/// at9p mainline (BEP5) locator — see #889 / epic #880 Track C.
pub mod locator;

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
                .unwrap_or_else(|_| "hyprstream-p2p=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .try_init()
        .map_err(|_| Error::other("Tracing already initialized"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_version() {
        // VERSION comes from CARGO_PKG_VERSION and is always non-empty
        assert!(!VERSION.is_empty());
    }
}