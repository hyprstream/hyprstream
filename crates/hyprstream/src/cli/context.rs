//! Application context for CLI commands
//!
//! This module provides the Context pattern for dependency injection,
//! allowing configuration and storage to be passed to command handlers
//! in an idiomatic Rust way.

use crate::config::HyprConfig;
use crate::services::RegistryClient;
use crate::storage::ModelStorage;
use anyhow::{Context as _, Result};
use std::sync::{Arc, Mutex};

/// Application context passed to all command handlers
///
/// This follows the "Context Pattern" common in Rust CLI applications.
/// Configuration is immutable after creation, and expensive resources
/// like storage are lazily initialized on first use.
///
/// # Example
///
/// ```rust,ignore
/// use crate::services::{RegistryService, RegistryZmqClient};
///
/// // Start registry service ONCE at CLI level
/// let _handle = RegistryService::start(&models_dir).await?;
/// let client: Arc<dyn RegistryClient> = Arc::new(RegistryZmqClient::new());
/// let context = AppContext::with_client(config, client);
///
/// // Pass to handlers - all share the same registry
/// handle_server(context.clone()).await?;
/// ```
#[derive(Clone)]
pub struct AppContext {
    /// Shared configuration (immutable after creation)
    config: Arc<HyprConfig>,

    /// Shared registry client (if provided)
    client: Option<Arc<dyn RegistryClient>>,

    /// Lazy-initialized storage (created on demand with thread safety)
    storage: Arc<tokio::sync::OnceCell<ModelStorage>>,

    /// Initialization lock to prevent race conditions
    init_lock: Arc<Mutex<()>>,
}

impl AppContext {
    /// Create new context from config (legacy - starts internal service)
    ///
    /// **Deprecated**: Use `with_client()` instead to share a single
    /// registry service across all components.
    pub fn new(config: HyprConfig) -> Self {
        Self {
            config: Arc::new(config),
            client: None,
            storage: Arc::new(tokio::sync::OnceCell::new()),
            init_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Create new context with a shared registry client
    ///
    /// This is the preferred method - the registry client should be started
    /// once in main() and shared across all components.
    pub fn with_client(config: HyprConfig, client: Arc<dyn RegistryClient>) -> Self {
        Self {
            config: Arc::new(config),
            client: Some(client),
            storage: Arc::new(tokio::sync::OnceCell::new()),
            init_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Get configuration reference
    pub fn config(&self) -> &HyprConfig {
        &self.config
    }

    /// Get the registry client if available
    pub fn registry_client(&self) -> Option<&Arc<dyn RegistryClient>> {
        self.client.as_ref()
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
        let _guard = self
            .init_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire initialization lock: {}", e))?;

        // Double-check after acquiring lock (double-checked locking pattern)
        if let Some(storage) = self.storage.get() {
            return Ok(storage);
        }

        // Initialize storage - use shared client if available
        let storage = if let Some(client) = &self.client {
            ModelStorage::new(client.clone(), self.config.models_dir().clone())
        } else {
            // Fallback: start internal service (legacy behavior)
            ModelStorage::create(self.config.models_dir().clone())
                .await
                .context("Failed to create model storage")?
        };

        // Store in OnceCell (this will only succeed once)
        self.storage
            .set(storage)
            .map_err(|_| anyhow::anyhow!("Storage was already initialized by another thread"))?;

        // Return reference to the newly initialized storage
        Ok(self.storage.get().expect("storage was just initialized"))
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

        assert_eq!(
            ctx.config().models_dir(),
            HyprConfig::default().models_dir()
        );
    }
}
