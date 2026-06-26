//! XET large file storage filter for git2db
//!
//! Provides transparent integration with XET repositories via libgit2 filters.
//!
//! This crate can be used standalone or as part of git2db.
//!
//! # Feature Flags
//!
//! - **`xet-storage`** (disabled by default): Enables actual XET CAS integration.
//!   Without this feature, only configuration types and error handling are available.
//!
//! ## Available without `xet-storage`:
//! - [`XetConfig`] - Configuration for XET endpoints
//! - [`XetError`], [`XetErrorKind`], [`Result`] - Error types for handling XET failures
//!
//! ## Available with `xet-storage`:
//! - [`initialize()`] - Register the XET filter with libgit2
//! - [`is_initialized()`] - Check if filter is registered
//! - Filter storage and runtime modules

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

#[cfg(feature = "ssh-transport")]
pub mod ssh_client;

#[cfg(feature = "gittorrent-transport")]
pub mod gittorrent_storage;

#[cfg(feature = "xet-storage")]
use tokio::sync::OnceCell;

use std::sync::Arc;
use std::cell::{Cell, RefCell};

/// Callback invoked on each smudge (download) operation during checkout.
///
/// Arguments: `(files_completed, file_path)`
/// Called from libgit2 filter callbacks (sync context, inside `spawn_blocking`).
pub type SmudgeProgressCallback = dyn Fn(usize, &str) + Send + Sync;

thread_local! {
    /// Per-thread smudge progress hook — set before worktree checkout, cleared after.
    /// Thread-local ensures concurrent checkouts on different `spawn_blocking` threads
    /// don't interfere with each other.
    static SMUDGE_HOOK: RefCell<Option<Arc<SmudgeProgressCallback>>> = const { RefCell::new(None) };

    /// Per-thread counter of smudge operations completed (reset on set_smudge_progress).
    static SMUDGE_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Set a callback to receive per-file smudge progress during checkout.
/// Resets the internal counter to 0.
///
/// Uses thread-local storage, so each `spawn_blocking` thread gets its own
/// independent hook. Safe for concurrent checkouts.
pub fn set_smudge_progress(callback: Arc<SmudgeProgressCallback>) {
    SMUDGE_COUNT.with(|c| c.set(0));
    SMUDGE_HOOK.with(|h| *h.borrow_mut() = Some(callback));
}

/// Clear the smudge progress callback for the current thread.
pub fn clear_smudge_progress() {
    SMUDGE_HOOK.with(|h| *h.borrow_mut() = None);
}

/// Called internally after each successful smudge operation.
pub(crate) fn notify_smudge_progress(path: &str) {
    SMUDGE_COUNT.with(|c| {
        let count = c.get() + 1;
        c.set(count);
        SMUDGE_HOOK.with(|h| {
            if let Some(ref cb) = *h.borrow() {
                cb(count, path);
            }
        });
    });
}

pub use config::{XetConfig, HUGGINGFACE_XET_ENDPOINT};
pub use error::{Result, XetError, XetErrorKind};

/// Global filter instance (initialized once)
#[cfg(feature = "xet-storage")]
static FILTER_INSTANCE: OnceCell<Arc<filter::XetFilter<filter::Registered>>> =
    OnceCell::const_new();

/// Config used during initialization (stored for later retrieval)
#[cfg(feature = "xet-storage")]
static FILTER_CONFIG: OnceCell<XetConfig> = OnceCell::const_new();

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
    // Check if already initialized with different config
    if let Some(existing) = FILTER_CONFIG.get() {
        if existing.endpoint != config.endpoint {
            tracing::warn!(
                "XET already initialized with endpoint '{}', ignoring new endpoint '{}'",
                existing.endpoint,
                config.endpoint
            );
        }
    } else {
        // Store config for later retrieval (first call wins)
        let _ = FILTER_CONFIG.set(config.clone());
    }

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

/// Get the XET config used during initialization
///
/// Returns `None` if the filter hasn't been initialized yet.
#[cfg(feature = "xet-storage")]
pub fn get_config() -> Option<XetConfig> {
    FILTER_CONFIG.get().cloned()
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
///         eprintln!("Error kind: {:?}", xet_error.kind());
///         eprintln!("Message: {}", xet_error.message());
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
    error::clear_last_error();
}

#[cfg(all(test, feature = "xet-storage"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_initialize_idempotent() -> std::result::Result<(), XetError> {
        let config = XetConfig {
            endpoint: "https://cas.xet.dev".to_owned(),
            token: Some("test".to_owned()),
            compression: None,
        };

        // First call
        initialize(config.clone()).await?;
        assert!(is_initialized());

        // Second call (should succeed)
        initialize(config).await?;
        Ok(())
    }
}
