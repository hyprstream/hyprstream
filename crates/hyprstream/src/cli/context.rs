//! Application context for CLI commands
//!
//! This module provides the Context pattern for dependency injection,
//! allowing configuration and the registry client to be passed to command handlers
//! in an idiomatic Rust way.

use crate::config::HyprConfig;
use crate::services::RegistryClient;
use crate::storage::ModelStorage;
use anyhow::Result;
use std::sync::Arc;

/// Application context passed to all command handlers
///
/// This follows the "Context Pattern" common in Rust CLI applications.
/// Configuration is immutable after creation, and the registry client
/// is shared across all handlers.
///
/// # Example
///
/// ```rust,ignore
/// use crate::services::{RegistryService, RegistryZmqClient};
///
/// // Start registry service ONCE at CLI level
/// let _handle = RegistryService::start(&models_dir).await?;
/// let client: Arc<dyn RegistryClient> = Arc::new(RegistryZmqClient::new());
/// let context = AppContext::new(config, client);
///
/// // Pass to handlers - all share the same registry
/// handle_server(context.clone()).await?;
/// ```
#[derive(Clone)]
pub struct AppContext {
    /// Shared configuration (immutable after creation)
    config: Arc<HyprConfig>,

    /// Shared registry client for all git operations
    registry: Arc<dyn RegistryClient>,

    /// Cached ModelStorage instance (lazily initialized)
    storage: Arc<tokio::sync::OnceCell<ModelStorage>>,
}

impl AppContext {
    /// Create new context with configuration and registry client.
    ///
    /// The registry client should be started once in main() and shared
    /// across all components.
    pub fn new(config: HyprConfig, registry: Arc<dyn RegistryClient>) -> Self {
        Self {
            config: Arc::new(config),
            registry,
            storage: Arc::new(tokio::sync::OnceCell::new()),
        }
    }

    /// Alias for `new()` - maintained for backward compatibility.
    pub fn with_client(config: HyprConfig, client: Arc<dyn RegistryClient>) -> Self {
        Self::new(config, client)
    }

    /// Get configuration reference
    pub fn config(&self) -> &HyprConfig {
        &self.config
    }

    /// Get the registry client
    pub fn registry(&self) -> &Arc<dyn RegistryClient> {
        &self.registry
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> &std::path::Path {
        self.config.models_dir()
    }

    /// Get ModelStorage instance (uses shared registry, never starts a new service).
    ///
    /// This provides backward compatibility for handlers that still use ModelStorage.
    /// The storage uses the shared registry client, ensuring no duplicate services.
    pub async fn storage(&self) -> Result<&ModelStorage> {
        self.storage
            .get_or_try_init(|| async {
                Ok(ModelStorage::new(
                    self.registry.clone(),
                    self.config.models_dir().to_path_buf(),
                ))
            })
            .await
    }
}
