//! Custom git transport support for git2db
//!
//! This module provides trait-based extension points for registering custom
//! git transports. This enables integration with P2P systems (GitTorrent),
//! content-addressable storage (IPFS), cloud storage (S3), and other backends.

use git2::transport::{SmartSubtransport, SmartSubtransportStream};
use std::sync::Arc;

/// Wrapper to make Box<dyn SmartSubtransport> implement SmartSubtransport
///
/// This is needed because trait objects don't automatically implement their trait.
/// We use this wrapper to pass boxed transports to git2::Transport::smart().
pub struct BoxedSubtransport(pub Box<dyn SmartSubtransport>);

impl SmartSubtransport for BoxedSubtransport {
    fn action(
        &self,
        url: &str,
        action: git2::transport::Service,
    ) -> Result<Box<dyn SmartSubtransportStream>, git2::Error> {
        self.0.action(url, action)
    }

    fn close(&self) -> Result<(), git2::Error> {
        self.0.close()
    }
}

/// Factory trait for creating custom git transports
///
/// Implementations of this trait can be registered with GitManager to handle
/// custom URL schemes like `gittorrent://`, `ipfs://`, `s3://`, etc.
///
/// # Example
///
/// ```ignore
/// use git2db::transport::TransportFactory;
/// use git2::transport::SmartSubtransport;
///
/// struct MyTransportFactory {
///     config: MyConfig,
/// }
///
/// impl TransportFactory for MyTransportFactory {
///     fn create_transport(&self, url: &str) -> anyhow::Result<Box<dyn SmartSubtransport>> {
///         Ok(Box::new(MyTransport::new(url, &self.config)?))
///     }
///
///     fn supports_url(&self, url: &str) -> bool {
///         url.starts_with("mytransport://")
///     }
/// }
/// ```
pub trait TransportFactory: Send + Sync {
    /// Create a new transport instance for the given URL
    ///
    /// This method is called by git2 when a remote operation (clone, fetch, push)
    /// needs to communicate with a repository at the given URL.
    ///
    /// # Arguments
    ///
    /// * `url` - The repository URL (e.g., "gittorrent://abc123/repo")
    ///
    /// # Returns
    ///
    /// A boxed `SmartSubtransport` implementation that handles git protocol operations
    fn create_transport(&self, url: &str) -> anyhow::Result<Box<dyn SmartSubtransport>>;

    /// Check if this factory can handle the given URL
    ///
    /// This is checked before calling `create_transport` to ensure the factory
    /// is appropriate for the URL scheme.
    ///
    /// # Arguments
    ///
    /// * `url` - The repository URL to check
    ///
    /// # Returns
    ///
    /// `true` if this factory can create transports for this URL, `false` otherwise
    fn supports_url(&self, url: &str) -> bool;
}

/// Helper trait for cloning TransportFactory trait objects
///
/// This is a workaround for the fact that `Clone` cannot be made into a trait object.
/// Implementations should return a boxed clone of themselves.
pub trait TransportFactoryClone {
    fn clone_box(&self) -> Box<dyn TransportFactory>;
}

impl<T> TransportFactoryClone for T
where
    T: TransportFactory + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn TransportFactory> {
        Box::new(self.clone())
    }
}

/// Extension trait to add Clone support to Arc<dyn TransportFactory>
///
/// This allows us to clone transport factories wrapped in Arc when needed
/// for passing to multiple callbacks or storing in collections.
pub trait TransportFactoryExt {
    fn clone_arc(&self) -> Arc<dyn TransportFactory>;
}

impl TransportFactoryExt for Arc<dyn TransportFactory> {
    fn clone_arc(&self) -> Arc<dyn TransportFactory> {
        Arc::clone(self)
    }
}

/// Mock transport for testing
///
/// This transport always fails with a predefined error, useful for testing
/// transport registration and error handling without real network operations.
pub struct MockTransport;

impl MockTransport {
    pub fn new(_url: &str) -> Self {
        Self
    }
}

impl SmartSubtransport for MockTransport {
    fn action(
        &self,
        _url: &str,
        _action: git2::transport::Service,
    ) -> Result<Box<dyn SmartSubtransportStream>, git2::Error> {
        Err(git2::Error::from_str("Mock transport: not implemented"))
    }

    fn close(&self) -> Result<(), git2::Error> {
        Ok(())
    }
}

/// Mock transport factory for testing
#[cfg(test)]
#[derive(Clone)]
pub struct MockTransportFactory;

#[cfg(test)]
impl TransportFactory for MockTransportFactory {
    fn create_transport(&self, url: &str) -> anyhow::Result<Box<dyn SmartSubtransport>> {
        Ok(Box::new(MockTransport::new(url)))
    }

    fn supports_url(&self, url: &str) -> bool {
        url.starts_with("mock://")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_transport_factory() {
        let factory = MockTransportFactory;
        assert!(factory.supports_url("mock://test/repo"));
        assert!(!factory.supports_url("https://example.com/repo"));

        let transport = factory.create_transport("mock://test/repo").unwrap();
        assert!(transport
            .action("mock://test/repo", git2::transport::Service::UploadPackLs)
            .is_err());
    }

    #[test]
    fn test_transport_factory_clone() {
        let factory = MockTransportFactory;
        let boxed: Box<dyn TransportFactory> = factory.clone_box();
        assert!(boxed.supports_url("mock://test/repo"));
    }
}
