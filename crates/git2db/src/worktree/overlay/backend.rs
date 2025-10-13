//! Overlay backend trait and implementations

use crate::errors::Git2DBResult;
use async_trait::async_trait;
use std::path::Path;

/// Backend for mounting overlayfs
#[async_trait]
pub trait OverlayBackend: Send + Sync {
    /// Mount overlayfs at the specified location
    async fn mount(
        &self,
        lower: &Path,
        upper: &Path,
        work: &Path,
        target: &Path,
    ) -> Git2DBResult<()>;

    /// Unmount overlayfs
    async fn unmount(&self, target: &Path) -> Git2DBResult<()>;

    /// Check if this backend is available
    fn is_available(&self) -> bool;

    /// Get backend name for logging
    fn name(&self) -> &'static str;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Requires elevated privileges
    pub requires_privileges: bool,

    /// Requires external binary
    pub requires_binary: Option<&'static str>,

    /// Performance relative to kernel overlayfs (1.0 = same)
    pub relative_performance: f32,

    /// Can work in user namespace
    pub user_namespace_compatible: bool,

    /// Expected space savings percentage (0-100)
    pub space_savings_percent: u8,
}
