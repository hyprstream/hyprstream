//! Dynamic sink registry for event bus subscribers
//!
//! The SinkRegistry manages event sinks that receive events from the EventBus.
//! Sinks are spawned as background tasks and can be registered/unregistered at runtime.

use super::bus::{EventBus, EventSubscriber};
use super::config::{SinkConfig, SinkType};
use super::sinks;
use super::EventEnvelope;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Information about a registered sink
#[derive(Debug, Clone)]
pub struct SinkInfo {
    /// Sink name
    pub name: String,
    /// Topic filter
    pub topic_filter: String,
    /// Sink type
    pub sink_type: String,
    /// Whether the sink is running
    pub is_running: bool,
}

/// Handle for a registered sink
struct SinkHandle {
    name: String,
    config: SinkConfig,
    task: JoinHandle<()>,
}

impl SinkHandle {
    fn is_running(&self) -> bool {
        !self.task.is_finished()
    }
}

/// Registry for managing event sinks
///
/// The registry spawns background tasks for each sink that consume events
/// from the EventBus and forward them to their destinations.
pub struct SinkRegistry {
    event_bus: Arc<EventBus>,
    sinks: RwLock<HashMap<String, SinkHandle>>,
}

impl SinkRegistry {
    /// Create a new sink registry
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        Self {
            event_bus,
            sinks: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new sink
    ///
    /// Creates a subscriber for the configured topic filter and spawns
    /// a background task to handle events.
    pub async fn register(&self, config: SinkConfig) -> Result<()> {
        let name = config.name.clone();

        // Check if already registered
        {
            let sinks = self.sinks.read().await;
            if sinks.contains_key(&name) {
                return Err(anyhow!("sink '{}' already registered", name));
            }
        }

        // Create subscriber
        let subscriber = self
            .event_bus
            .subscriber(&config.subscribe)
            .await
            .map_err(|e| anyhow!("failed to create subscriber for '{}': {}", name, e))?;

        // Spawn sink task
        let task = match &config.sink_type {
            SinkType::InProcess { handler } => {
                let handler = handler.clone();
                tokio::spawn(async move {
                    sinks::in_process_loop(subscriber, &handler).await;
                })
            }
            SinkType::Webhook { url, headers } => {
                let url = url.clone();
                let headers = headers.clone();
                tokio::spawn(async move {
                    sinks::webhook_loop(subscriber, &url, headers).await;
                })
            }
            SinkType::Nats {
                url,
                subject_prefix,
            } => {
                let url = url.clone();
                let subject_prefix = subject_prefix.clone();
                tokio::spawn(async move {
                    sinks::nats_loop(subscriber, &url, &subject_prefix).await;
                })
            }
            SinkType::Mcp { tool, endpoint } => {
                let tool = tool.clone();
                let endpoint = endpoint.clone();
                tokio::spawn(async move {
                    sinks::mcp_loop(subscriber, &tool, &endpoint).await;
                })
            }
            SinkType::Container { image, runtime } => {
                let image = image.clone();
                let runtime = runtime.clone();
                tokio::spawn(async move {
                    sinks::container_loop(subscriber, &image, runtime.as_deref()).await;
                })
            }
        };

        // Store handle
        let handle = SinkHandle {
            name: name.clone(),
            config,
            task,
        };

        self.sinks.write().await.insert(name.clone(), handle);
        info!("registered sink '{}'", name);

        Ok(())
    }

    /// Register a sink with a custom handler function
    ///
    /// This is useful for in-process sinks that need direct access to
    /// application state.
    pub async fn register_handler<F, Fut>(
        &self,
        name: &str,
        topic_filter: &str,
        handler: F,
    ) -> Result<()>
    where
        F: Fn(EventEnvelope) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        // Check if already registered
        {
            let sinks = self.sinks.read().await;
            if sinks.contains_key(name) {
                return Err(anyhow!("sink '{}' already registered", name));
            }
        }

        // Create subscriber
        let mut subscriber = self
            .event_bus
            .subscriber(topic_filter)
            .await
            .map_err(|e| anyhow!("failed to create subscriber for '{}': {}", name, e))?;

        let name_clone = name.to_string();

        // Spawn handler task
        let task = tokio::spawn(async move {
            loop {
                match subscriber.recv().await {
                    Ok(event) => {
                        handler(event).await;
                    }
                    Err(e) => {
                        warn!("sink '{}' recv error: {}", name_clone, e);
                        // Brief backoff on error
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }
            }
        });

        // Create synthetic config for tracking
        let config = SinkConfig {
            name: name.to_string(),
            sink_type: SinkType::InProcess {
                handler: "<custom>".to_string(),
            },
            subscribe: topic_filter.to_string(),
        };

        let handle = SinkHandle {
            name: name.to_string(),
            config,
            task,
        };

        self.sinks.write().await.insert(name.to_string(), handle);
        info!("registered handler sink '{}'", name);

        Ok(())
    }

    /// Unregister and stop a sink
    pub async fn unregister(&self, name: &str) -> Result<()> {
        let handle = self
            .sinks
            .write()
            .await
            .remove(name)
            .ok_or_else(|| anyhow!("sink '{}' not found", name))?;

        // Abort the task
        handle.task.abort();

        // Wait for it to finish
        let _ = handle.task.await;

        info!("unregistered sink '{}'", name);
        Ok(())
    }

    /// List all registered sinks
    pub async fn list(&self) -> Vec<SinkInfo> {
        let sinks = self.sinks.read().await;
        sinks
            .values()
            .map(|h| SinkInfo {
                name: h.name.clone(),
                topic_filter: h.config.subscribe.clone(),
                sink_type: format!("{:?}", h.config.sink_type),
                is_running: h.is_running(),
            })
            .collect()
    }

    /// Get information about a specific sink
    pub async fn get(&self, name: &str) -> Option<SinkInfo> {
        let sinks = self.sinks.read().await;
        sinks.get(name).map(|h| SinkInfo {
            name: h.name.clone(),
            topic_filter: h.config.subscribe.clone(),
            sink_type: format!("{:?}", h.config.sink_type),
            is_running: h.is_running(),
        })
    }

    /// Check if a sink is registered
    pub async fn contains(&self, name: &str) -> bool {
        self.sinks.read().await.contains_key(name)
    }

    /// Stop all sinks
    pub async fn shutdown(&self) {
        let mut sinks = self.sinks.write().await;

        for (name, handle) in sinks.drain() {
            debug!("stopping sink '{}'", name);
            handle.task.abort();
            let _ = handle.task.await;
        }

        info!("all sinks stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let bus = Arc::new(EventBus::new().await.unwrap());
        let registry = SinkRegistry::new(bus);

        let sinks = registry.list().await;
        assert!(sinks.is_empty());
    }

    #[tokio::test]
    async fn test_register_unregister() {
        let bus = Arc::new(EventBus::new().await.unwrap());
        let registry = SinkRegistry::new(bus);

        let config = SinkConfig {
            name: "test-sink".to_string(),
            sink_type: SinkType::InProcess {
                handler: "test".to_string(),
            },
            subscribe: "test".to_string(),
        };

        // Register
        registry.register(config).await.unwrap();
        assert!(registry.contains("test-sink").await);

        // List
        let sinks = registry.list().await;
        assert_eq!(sinks.len(), 1);
        assert_eq!(sinks[0].name, "test-sink");

        // Unregister
        registry.unregister("test-sink").await.unwrap();
        assert!(!registry.contains("test-sink").await);
    }

    #[tokio::test]
    async fn test_duplicate_registration() {
        let bus = Arc::new(EventBus::new().await.unwrap());
        let registry = SinkRegistry::new(bus);

        let config = SinkConfig {
            name: "test-sink".to_string(),
            sink_type: SinkType::InProcess {
                handler: "test".to_string(),
            },
            subscribe: "test".to_string(),
        };

        // First registration succeeds
        registry.register(config.clone()).await.unwrap();

        // Second registration fails
        let result = registry.register(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_custom_handler() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let bus = Arc::new(EventBus::new().await.unwrap());
        let registry = SinkRegistry::new(bus.clone());

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        registry
            .register_handler("test-handler", "test", move |_event| {
                let counter = counter_clone.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            })
            .await
            .unwrap();

        assert!(registry.contains("test-handler").await);

        // Cleanup
        registry.shutdown().await;
    }
}
