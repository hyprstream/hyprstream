//! Dedicated runtime for XET filter async operations
//!
//! Provides a safe way to run async operations from synchronous C callbacks

use crate::error::{Result, XetError, XetErrorKind};
use std::future::Future;
use std::mem::ManuallyDrop;
use std::time::Duration;
use tokio::runtime::{Builder, Runtime};

/// Dedicated runtime for XET filter operations
///
/// This runtime is created once per filter initialization and used
/// for all async operations. It's single-threaded to avoid complexity
/// and is separate from any user-created runtimes.
pub struct XetRuntime {
    // Use ManuallyDrop to control when the runtime is dropped
    // This prevents tokio from panicking when dropped in async context
    runtime: ManuallyDrop<Runtime>,
    timeout: Duration,
}

impl XetRuntime {
    /// Create a new dedicated runtime
    ///
    /// NOTE: This should NOT be called from within an existing Tokio runtime context.
    /// Instead, use the handle-based approach or spawn_blocking.
    pub fn new() -> Result<Self> {
        // Check if we're already in a runtime context
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(XetError::new(
                XetErrorKind::RuntimeError,
                "Cannot create XetRuntime from within existing async context. \
                 XetRuntime creates its own runtime and cannot be nested.".to_string(),
            ));
        }

        let runtime = Builder::new_current_thread()
            .thread_name("xet-filter")
            .enable_all()
            .build()
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::RuntimeError,
                    format!("Failed to create runtime: {}", e),
                )
            })?;

        Ok(Self {
            runtime: ManuallyDrop::new(runtime),
            timeout: Duration::from_secs(300), // 5 minute timeout
        })
    }

    /// Block on a future with timeout
    ///
    /// This is safe to call from C callbacks because we own the runtime
    /// and don't rely on Handle::current()
    pub fn block_on<F>(&self, future: F) -> Result<F::Output>
    where
        F: Future,
    {
        // Use our dedicated runtime, not current
        self.runtime.block_on(async {
            tokio::time::timeout(self.timeout, future)
                .await
                .map_err(|_| XetError::new(XetErrorKind::RuntimeError, "Operation timed out"))
        })
    }

    /// Block on a future without timeout (use sparingly)
    pub fn block_on_unchecked<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.runtime.block_on(future)
    }
}

impl Drop for XetRuntime {
    fn drop(&mut self) {
        // Shutdown the runtime in a blocking context to avoid panics
        // SAFETY: We're in Drop, so this is the last use of the runtime
        unsafe {
            // If we're in an async context, spawn a thread to drop the runtime
            // Otherwise, drop it directly
            if tokio::runtime::Handle::try_current().is_ok() {
                // We're in an async context - need to drop in a separate thread
                let runtime = ManuallyDrop::take(&mut self.runtime);
                std::thread::spawn(move || {
                    drop(runtime);
                });
            } else {
                // Safe to drop directly
                ManuallyDrop::drop(&mut self.runtime);
            }
        }
    }
}

// Safety: Runtime is Send + Sync
unsafe impl Send for XetRuntime {}
unsafe impl Sync for XetRuntime {}
