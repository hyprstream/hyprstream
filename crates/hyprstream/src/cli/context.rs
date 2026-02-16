//! Application context for CLI commands
//!
//! This module provides the Context pattern for dependency injection,
//! allowing configuration and the registry client to be passed to command handlers
//! in an idiomatic Rust way.

use crate::config::HyprConfig;
use crate::services::GenRegistryClient;
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
/// use crate::services::GenRegistryClient;
///
/// // Start registry service ONCE at CLI level
/// let client: GenRegistryClient = crate::services::core::create_service_client(
///     &endpoint, signing_key, identity,
/// );
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
    registry: GenRegistryClient,
}

impl AppContext {
    /// Create new context with configuration and registry client.
    ///
    /// The registry client should be started once in main() and shared
    /// across all components.
    pub fn new(config: HyprConfig, registry: GenRegistryClient) -> Self {
        Self {
            config: Arc::new(config),
            registry,
        }
    }

    /// Alias for `new()` - maintained for backward compatibility.
    pub fn with_client(config: HyprConfig, client: GenRegistryClient) -> Self {
        Self::new(config, client)
    }

    /// Get configuration reference
    pub fn config(&self) -> &HyprConfig {
        &self.config
    }

    /// Get the registry client
    pub fn registry(&self) -> &GenRegistryClient {
        &self.registry
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> &std::path::Path {
        self.config.models_dir()
    }
}
