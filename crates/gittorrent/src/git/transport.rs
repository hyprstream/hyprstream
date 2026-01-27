//! Git2 transport implementation for gittorrent:// URLs
//!
//! This module provides git2::SmartSubtransport trait implementations that allow
//! applications to use GitTorrent programmatically through the git2 library,
//! rather than just through the git-remote-gittorrent binary.

use crate::service::GitTorrentService;
use crate::{Result, Error, GitTorrentUrl};
use git2::transport::{SmartSubtransport, SmartSubtransportStream, Service};
use std::sync::Arc;
use std::io::{Read, Write};

/// Factory for creating GitTorrent transports
///
/// Implements git2db::TransportFactory trait for integration with git2db
#[derive(Clone)]
pub struct GittorrentTransportFactory {
    service: Arc<GitTorrentService>,
}

impl GittorrentTransportFactory {
    /// Create a new GitTorrent transport factory
    pub fn new(service: Arc<GitTorrentService>) -> Self {
        Self { service }
    }
}

/// Git2 transport factory trait implementation
pub trait TransportFactory {
    /// Create a new transport for the given URL
    fn create_transport(&self, url: &str) -> anyhow::Result<Box<dyn SmartSubtransport>>;

    /// Check if this factory supports the given URL
    fn supports_url(&self, url: &str) -> bool;
}

impl TransportFactory for GittorrentTransportFactory {
    fn create_transport(&self, url: &str) -> anyhow::Result<Box<dyn SmartSubtransport>> {
        Ok(Box::new(GittorrentTransport::new(url, Arc::clone(&self.service))?))
    }

    fn supports_url(&self, url: &str) -> bool {
        url.starts_with("gittorrent://")
    }
}

/// GitTorrent smart subtransport implementation
pub struct GittorrentTransport {
    url: GitTorrentUrl,
    service: Arc<GitTorrentService>,
}

impl GittorrentTransport {
    /// Create a new GitTorrent transport
    pub fn new(url: &str, service: Arc<GitTorrentService>) -> Result<Self> {
        let parsed_url = GitTorrentUrl::parse(url)?;
        Ok(Self {
            url: parsed_url,
            service,
        })
    }
}

impl SmartSubtransport for GittorrentTransport {
    fn action(
        &self,
        url: &str,
        service: Service,
    ) -> std::result::Result<Box<dyn SmartSubtransportStream>, git2::Error> {
        tracing::debug!("GitTorrent transport action: {} (service: {})", url, match service {
            Service::UploadPack => "upload-pack",
            Service::ReceivePack => "receive-pack",
            Service::UploadPackLs => "upload-pack-ls",
            Service::ReceivePackLs => "receive-pack-ls",
        });

        match service {
            Service::UploadPack => {
                // Handle fetch operations (upload-pack)
                GittorrentStream::new_upload_pack(
                    self.url.clone(),
                    Arc::clone(&self.service),
                ).map(|s| Box::new(s) as Box<dyn SmartSubtransportStream>)
                .map_err(|e| git2::Error::from_str(&e.to_string()))
            }
            Service::ReceivePack => {
                // Handle push operations (receive-pack)
                GittorrentStream::new_receive_pack(
                    self.url.clone(),
                    Arc::clone(&self.service),
                ).map(|s| Box::new(s) as Box<dyn SmartSubtransportStream>)
                .map_err(|e| git2::Error::from_str(&e.to_string()))
            }
            Service::UploadPackLs => {
                // Handle reference discovery for fetch
                GittorrentStream::new_upload_pack(
                    self.url.clone(),
                    Arc::clone(&self.service),
                ).map(|s| Box::new(s) as Box<dyn SmartSubtransportStream>)
                .map_err(|e| git2::Error::from_str(&e.to_string()))
            }
            Service::ReceivePackLs => {
                // Handle reference discovery for push
                GittorrentStream::new_receive_pack(
                    self.url.clone(),
                    Arc::clone(&self.service),
                ).map(|s| Box::new(s) as Box<dyn SmartSubtransportStream>)
                .map_err(|e| git2::Error::from_str(&e.to_string()))
            }
        }
    }

    fn close(&self) -> std::result::Result<(), git2::Error> {
        tracing::debug!("GitTorrent transport closing");
        Ok(())
    }
}

/// GitTorrent stream for handling Git protocol communications
///
/// Note: This is a placeholder implementation. The full Git protocol
/// implementation would require the URL and service for:
/// - URL: Identifying which repository to interact with
/// - Service: Accessing the DHT and P2P network for git objects
pub struct GittorrentStream {
    buffer: Vec<u8>,
    position: usize,
    finished: bool,
}

impl GittorrentStream {
    /// Create a new stream for upload-pack (fetch) operations
    fn new_upload_pack(_url: GitTorrentUrl, _service: Arc<GitTorrentService>) -> Result<Self> {
        // TODO: Use url and service for actual Git protocol implementation
        Ok(Self {
            buffer: Vec::new(),
            position: 0,
            finished: false,
        })
    }

    /// Create a new stream for receive-pack (push) operations
    fn new_receive_pack(_url: GitTorrentUrl, _service: Arc<GitTorrentService>) -> Result<Self> {
        // TODO: Use url and service for actual Git protocol implementation
        Ok(Self {
            buffer: Vec::new(),
            position: 0,
            finished: false,
        })
    }
}

// SmartSubtransportStream is automatically implemented for types that implement Read + Write + Send + 'static

impl Read for GittorrentStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.finished {
            return Ok(0);
        }

        // If we don't have data buffered, try to get some
        if self.position >= self.buffer.len() {
            // For now, just return EOF since protocol handling isn't complete
            self.finished = true;
            return Ok(0);
        }

        let available = self.buffer.len() - self.position;
        let to_read = std::cmp::min(buf.len(), available);

        buf[..to_read].copy_from_slice(&self.buffer[self.position..self.position + to_read]);
        self.position += to_read;

        Ok(to_read)
    }
}

impl Write for GittorrentStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Handle writes from Git (commands, pack data, etc.)
        tracing::debug!("GitTorrent stream received {} bytes", buf.len());

        // In a full implementation, this would parse Git protocol commands
        // and handle them appropriately based on the service type

        // For now, just acknowledge the write
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Nothing to flush in this basic implementation
        Ok(())
    }
}

/// Helper function to register GitTorrent transport with git2
///
/// This registers the gittorrent:// URL scheme handler globally with git2.
/// Once registered, any git2 operations (clone, fetch, push) with gittorrent:// URLs
/// will automatically use the P2P transport.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use gittorrent::service::GitTorrentService;
/// use gittorrent::git::transport::register_gittorrent_transport;
///
/// # async fn example() -> anyhow::Result<()> {
/// let service = Arc::new(GitTorrentService::new(Default::default()).await?);
/// register_gittorrent_transport(service)?;
///
/// // Now gittorrent:// URLs work with standard git2 operations
/// let repo = git2::Repository::clone("gittorrent://hash/repo", "/tmp/repo")?;
/// # Ok(())
/// # }
/// ```
pub fn register_gittorrent_transport(service: Arc<GitTorrentService>) -> Result<()> {
    let factory = GittorrentTransportFactory::new(service);

    unsafe {
        git2::transport::register("gittorrent", move |remote| {
            let url = remote.url().ok_or_else(|| {
                git2::Error::from_str("Remote URL is required")
            })?;

            tracing::debug!("Creating GitTorrent transport for URL: {}", url);

            match GittorrentTransport::new(url, Arc::clone(&factory.service)) {
                Ok(subtransport) => {
                    git2::transport::Transport::smart(remote, true, subtransport)
                },
                Err(e) => {
                    tracing::error!("Failed to create GitTorrent transport: {}", e);
                    Err(git2::Error::from_str(&format!("Transport creation failed: {e}")))
                }
            }
        }).map_err(|e| Error::other(format!("Failed to register gittorrent transport: {e}")))?;
    }

    tracing::info!("GitTorrent transport registered for gittorrent:// URLs");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::GitTorrentConfig;

    #[tokio::test]
    async fn test_transport_factory_creation() {
        let config = GitTorrentConfig::default();
        let service = Arc::new(GitTorrentService::new(config).await.unwrap());
        let factory = GittorrentTransportFactory::new(service);

        assert!(factory.supports_url("gittorrent://abc123def456"));
        assert!(!factory.supports_url("https://github.com/user/repo"));
    }

    #[tokio::test]
    async fn test_transport_creation() {
        let config = GitTorrentConfig::default();
        let service = Arc::new(GitTorrentService::new(config).await.unwrap());

        // Test with a short string (not a valid hash) - should parse as Username
        let transport = GittorrentTransport::new(
            "gittorrent://abc123def456",
            service.clone()
        ).unwrap();
        assert!(matches!(transport.url, GitTorrentUrl::Username { .. }));

        // Test with a valid 40-char SHA1 - should parse as Commit
        // Note: GitTorrentUrl currently only accepts 64-char (SHA256) for Commit variant
        // When we update GitTorrentUrl to support SHA1, this can be enabled:
        // let transport = GittorrentTransport::new(
        //     "gittorrent://0123456789abcdef0123456789abcdef01234567",
        //     service.clone()
        // ).unwrap();
        // assert!(matches!(transport.url, GitTorrentUrl::Commit { .. }));

        // For now, test with a 64-char SHA256
        let transport = GittorrentTransport::new(
            "gittorrent://0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            service
        ).unwrap();
        assert!(matches!(transport.url, GitTorrentUrl::Commit { .. }));
    }
}