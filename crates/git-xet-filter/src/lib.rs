//! XET large file storage filter for git2db
//!
//! Provides transparent integration with XET repositories via libgit2 filters.
//!
//! This crate can be used standalone or as part of git2db.

pub mod config;
pub mod error;

#[cfg(feature = "xet-storage")]
pub mod ffi;

#[cfg(feature = "xet-storage")]
pub mod filter;

#[cfg(feature = "xet-storage")]
pub mod storage;

#[cfg(feature = "xet-storage")]
pub mod runtime;

#[cfg(feature = "xet-storage")]
pub mod callbacks;

#[cfg(feature = "xet-storage")]
use tokio::sync::OnceCell;

#[cfg(feature = "xet-storage")]
use std::sync::Arc;

pub use config::XetConfig;
pub use error::{Result, XetError, XetErrorKind};

/// Global filter instance (initialized once)
#[cfg(feature = "xet-storage")]
static FILTER_INSTANCE: OnceCell<Arc<filter::XetFilter<filter::Registered>>> =
    OnceCell::const_new();

/// Initialize XET filter support
///
/// This function is idempotent - calling it multiple times with the same
/// config is safe and will return success without re-registering.
///
/// # Errors
///
/// Returns error if:
/// - XET storage initialization fails
/// - Filter registration with libgit2 fails
/// - Called with different config after already initialized
#[cfg(feature = "xet-storage")]
pub async fn initialize(config: XetConfig) -> Result<()> {
    FILTER_INSTANCE
        .get_or_try_init(|| async {
            tracing::info!("Initializing XET filter...");

            // Create unregistered filter
            let filter = filter::XetFilter::new(config).await?;

            // Register with libgit2 (priority 100, same as git-lfs)
            let registered = filter.register(100)?;

            tracing::info!("XET filter registered successfully");
            Ok::<_, XetError>(Arc::new(registered))
        })
        .await?;

    Ok(())
}

/// Check if XET filter is initialized
#[cfg(feature = "xet-storage")]
pub fn is_initialized() -> bool {
    FILTER_INSTANCE.get().is_some()
}

/// Get the last XET filter error from this thread
///
/// When a git operation fails and XET filter was involved, this function
/// can provide detailed error information that wouldn't be available through
/// git2's error handling.
///
/// # Example
/// ```ignore
/// use git2::Repository;
/// use git2db::xet_filter;
///
/// let repo = Repository::open("path/to/repo")?;
/// if let Err(e) = repo.checkout_head(None) {
///     // Git operation failed - check if XET filter has details
///     if let Some(xet_error) = xet_filter::last_error() {
///         eprintln!("XET filter error: {:?}", xet_error);
///         eprintln!("Error kind: {:?}", xet_error.kind);
///         eprintln!("Message: {}", xet_error.message);
///     }
/// }
/// ```
///
/// # Thread-Local Storage
/// This function uses thread-local storage, so it only returns errors from
/// git operations performed on the current thread.
pub fn last_error() -> Option<XetError> {
    error::get_last_error()
}

/// Clear the last XET filter error for this thread
///
/// This can be useful to prevent stale errors from previous operations
/// from being mistaken for new errors.
pub fn clear_last_error() {
    error::clear_last_error()
}

#[cfg(all(test, feature = "xet-storage"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_initialize_idempotent() {
        let config = XetConfig {
            endpoint: "https://cas.xet.dev".to_string(),
            token: Some("test".to_string()),
            compression: None,
        };

        // First call
        initialize(config.clone()).await.unwrap();
        assert!(is_initialized());

        // Second call (should succeed)
        initialize(config).await.unwrap();
    }
}
