//! Dynamic sink registry for event bus subscribers
//!
//! The SinkRegistry manages event sinks that receive events from the EventBus.
//! Sinks are spawned as background tasks and can be registered/unregistered at runtime.
//!
//! Note: Sinks run on blocking threads because ZMQ sockets are not thread-safe.
//! The subscriber is created on the same thread that will use it.

use super::bus::{EventBus, EventSubscriber, INPROC_ENDPOINT};
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

        // Get context for creating subscriber on blocking thread
        let context = self.event_bus.context().clone();
        let topic_filter = config.subscribe.clone();
        let endpoint = INPROC_ENDPOINT.to_string();

        // Spawn sink task on blocking thread
        // The subscriber is created ON the blocking thread to ensure thread locality
        let task = match &config.sink_type {
            SinkType::InProcess { handler } => {
                let handler = handler.clone();
                tokio::task::spawn_blocking(move || {
                    // Create subscriber on this thread
                    match sinks::create_subscriber(&context, &topic_filter, &endpoint) {
                        Ok(subscriber) => {
                            sinks::in_process_loop(subscriber, &handler);
                        }
                        Err(e) => {
                            error!("failed to create subscriber: {}", e);
                        }
                    }
                })
            }
            SinkType::Webhook { url, headers } => {
                let url = url.clone();
                let headers = headers.clone();
                tokio::task::spawn_blocking(move || {
                    match sinks::create_subscriber(&context, &topic_filter, &endpoint) {
                        Ok(subscriber) => {
                            sinks::webhook_loop(subscriber, &url, headers);
                        }
                        Err(e) => {
                            error!("failed to create subscriber: {}", e);
                        }
                    }
                })
            }
            SinkType::Nats {
                url,
                subject_prefix,
            } => {
                let url = url.clone();
                let subject_prefix = subject_prefix.clone();
                tokio::task::spawn_blocking(move || {
                    match sinks::create_subscriber(&context, &topic_filter, &endpoint) {
                        Ok(subscriber) => {
                            sinks::nats_loop(subscriber, &url, &subject_prefix);
                        }
                        Err(e) => {
                            error!("failed to create subscriber: {}", e);
                        }
                    }
                })
            }
            SinkType::Mcp { tool, endpoint: mcp_endpoint } => {
                let tool = tool.clone();
                let mcp_endpoint = mcp_endpoint.clone();
                tokio::task::spawn_blocking(move || {
                    match sinks::create_subscriber(&context, &topic_filter, &endpoint) {
                        Ok(subscriber) => {
                            sinks::mcp_loop(subscriber, &tool, &mcp_endpoint);
                        }
                        Err(e) => {
                            error!("failed to create subscriber: {}", e);
                        }
                    }
                })
            }
            SinkType::Container { image, runtime } => {
                let image = image.clone();
                let runtime = runtime.clone();
                tokio::task::spawn_blocking(move || {
                    match sinks::create_subscriber(&context, &topic_filter, &endpoint) {
                        Ok(subscriber) => {
                            sinks::container_loop(subscriber, &image, runtime.as_deref());
                        }
                        Err(e) => {
                            error!("failed to create subscriber: {}", e);
                        }
                    }
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
    ///
    /// Note: The handler runs on a blocking thread. For async work,
    /// spawn tasks using `tokio::runtime::Handle::try_current()`.
    pub async fn register_handler<F>(
        &self,
        name: &str,
        topic_filter: &str,
        handler: F,
    ) -> Result<()>
    where
        F: Fn(EventEnvelope) + Send + 'static,
    {
        // Check if already registered
        {
            let sinks = self.sinks.read().await;
            if sinks.contains_key(name) {
                return Err(anyhow!("sink '{}' already registered", name));
            }
        }

        let context = self.event_bus.context().clone();
        let topic_filter_owned = topic_filter.to_string();
        let endpoint = INPROC_ENDPOINT.to_string();
        let name_clone = name.to_string();

        // Spawn handler task on blocking thread
        let task = tokio::task::spawn_blocking(move || {
            match sinks::create_subscriber(&context, &topic_filter_owned, &endpoint) {
                Ok(subscriber) => {
                    loop {
                        match subscriber.recv() {
                            Ok(event) => {
                                handler(event);
                            }
                            Err(e) => {
                                warn!("sink '{}' recv error: {}", name_clone, e);
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("failed to create subscriber for '{}': {}", name_clone, e);
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
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .await
            .unwrap();

        assert!(registry.contains("test-handler").await);

        // Cleanup
        registry.shutdown().await;
    }
}
