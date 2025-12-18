//! Registry client trait and error types.

use crate::{Git2DBError, RepoId, TrackedRepository};
use async_trait::async_trait;
use std::path::Path;
use thiserror::Error;

/// Transport-agnostic registry client trait.
///
/// Implementations can be in-process (channels) or remote (gRPC, etc.).
/// All methods return owned data to avoid lifetime issues across transports.
#[async_trait]
pub trait RegistryClient: Send + Sync {
    // === Discovery (read-heavy, consider using cached_list for performance) ===

    /// List all tracked repositories.
    ///
    /// Goes through the service channel to get fresh data.
    /// For read-heavy workloads, prefer `cached_list()` when available.
    async fn list(&self) -> Result<Vec<TrackedRepository>, ServiceError>;

    /// Get repository by ID.
    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, ServiceError>;

    /// Get repository by name.
    async fn get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>, ServiceError>;

    /// Fast path: get cached list (bypasses channel if available).
    ///
    /// Returns `None` if caching is not supported by this client.
    /// Default implementation returns `None`.
    fn cached_list(&self) -> Option<Vec<TrackedRepository>> {
        None
    }

    // === Mutation (always through channel) ===

    /// Clone a repository from URL.
    ///
    /// # Arguments
    /// * `url` - Repository URL to clone
    /// * `name` - Optional name for the repository (defaults to URL-derived name)
    async fn clone_repo(&self, url: &str, name: Option<&str>) -> Result<RepoId, ServiceError>;

    /// Register an existing repository.
    ///
    /// # Arguments
    /// * `id` - Repository ID to assign
    /// * `name` - Optional name for the repository
    /// * `path` - Path to the existing repository
    async fn register(
        &self,
        id: &RepoId,
        name: Option<&str>,
        path: &Path,
    ) -> Result<(), ServiceError>;

    /// Upsert: update if exists, create if not.
    ///
    /// This is useful for ensuring a repository exists without checking first.
    async fn upsert(&self, name: &str, url: &str) -> Result<RepoId, ServiceError>;

    /// Remove a repository from the registry.
    async fn remove(&self, id: &RepoId) -> Result<(), ServiceError>;

    // === Health ===

    /// Check service health (for testing/monitoring).
    ///
    /// Returns `Ok(())` if the service is healthy, error otherwise.
    async fn health_check(&self) -> Result<(), ServiceError>;
}

/// Service error type wrapping Git2DBError with service-layer concerns.
#[derive(Debug, Error)]
pub enum ServiceError {
    /// Registry operation failed (wraps underlying Git2DBError).
    #[error("Registry operation failed: {0}")]
    Registry(#[from] Git2DBError),

    /// Service communication failed (channel closed, send failed, etc.).
    #[error("Service communication failed: {0}")]
    Channel(String),

    /// Service is unavailable (not started, shutdown, etc.).
    #[error("Service unavailable")]
    Unavailable,
}

impl ServiceError {
    /// Check if this error suggests retrying the operation.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Registry(e) => e.should_retry(),
            Self::Channel(_) => true,
            Self::Unavailable => true,
        }
    }

    /// Create a channel error.
    pub fn channel<S: Into<String>>(msg: S) -> Self {
        Self::Channel(msg.into())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_error_is_retryable() {
        // Channel errors should be retryable
        let err = ServiceError::channel("test");
        assert!(err.is_retryable());

        // Unavailable should be retryable
        let err = ServiceError::Unavailable;
        assert!(err.is_retryable());
    }
}
