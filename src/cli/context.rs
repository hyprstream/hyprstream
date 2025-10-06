//! Application context for CLI commands
//!
//! This module provides the Context pattern for dependency injection,
//! allowing configuration and storage to be passed to command handlers
//! in an idiomatic Rust way.

use std::sync::{Arc, Mutex};
use crate::config::HyprConfig;
use crate::storage::ModelStorage;
use anyhow::{Result, Context as _};

/// Application context passed to all command handlers
///
/// This follows the "Context Pattern" common in Rust CLI applications.
/// Configuration is immutable after creation, and expensive resources
/// like storage are lazily initialized on first use.
///
/// # Example
///
/// ```rust,ignore
/// let config = HyprConfig::load()?;
/// let context = AppContext::new(config);
///
/// // Pass to handlers
/// handle_server(context.clone()).await?;
/// ```
#[derive(Clone)]
pub struct AppContext {
    /// Shared configuration (immutable after creation)
    config: Arc<HyprConfig>,

    /// Lazy-initialized storage (created on demand with thread safety)
    storage: Arc<tokio::sync::OnceCell<ModelStorage>>,

    /// Initialization lock to prevent race conditions
    init_lock: Arc<Mutex<()>>,
}

impl AppContext {
    /// Create new context from config
    pub fn new(config: HyprConfig) -> Self {
        Self {
            config: Arc::new(config),
            storage: Arc::new(tokio::sync::OnceCell::new()),
            init_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Get configuration reference
    pub fn config(&self) -> &HyprConfig {
        &self.config
    }

    /// Get or create storage instance (lazy initialization)
    ///
    /// Storage is created on first access and reused for subsequent calls.
    /// This avoids creating storage for commands that don't need it.
    /// Uses a mutex to prevent race conditions during initialization.
    pub async fn storage(&self) -> Result<&ModelStorage> {
        // Fast path: if already initialized, return immediately
        if let Some(storage) = self.storage.get() {
            return Ok(storage);
        }

        // Slow path: initialize with lock to prevent race conditions
        let _guard = self.init_lock.lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire initialization lock: {}", e))?;

        // Double-check after acquiring lock (double-checked locking pattern)
        if let Some(storage) = self.storage.get() {
            return Ok(storage);
        }

        // Initialize storage
        let storage = ModelStorage::create(self.config.models_dir().clone())
            .await
            .context("Failed to create model storage")?;

        // Store in OnceCell (this will only succeed once)
        self.storage.set(storage)
            .map_err(|_| anyhow::anyhow!("Storage was already initialized by another thread"))?;

        // Return reference to the newly initialized storage
        Ok(self.storage.get().unwrap())
    }

    /// Create storage immediately (for testing or pre-initialization)
    pub async fn init_storage(&self) -> Result<()> {
        self.storage().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let config = HyprConfig::default();
        let ctx = AppContext::new(config);

        assert_eq!(ctx.config().models_dir(), HyprConfig::default().models_dir());
    }
}
