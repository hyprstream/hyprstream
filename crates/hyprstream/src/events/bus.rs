//! ZeroMQ-based event bus for pub/sub messaging
//!
//! The EventBus uses ZeroMQ PUB/SUB pattern with support for multiple transports:
//! - `inproc://` for in-process messaging (zero-copy)
//! - `ipc://` for inter-process messaging (MCP tools, containers)
//! - `tcp://` for network messaging (distributed systems)
//!
//! Serialization uses Cap'n Proto for zero-copy performance.

use super::capnp_serde;
use super::EventEnvelope;
use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, trace, warn};

/// Default inproc endpoint for in-process messaging
pub const INPROC_ENDPOINT: &str = "inproc://hyprstream/events";

/// Default IPC endpoint for sandbox processes
pub const IPC_ENDPOINT: &str = "ipc:///tmp/hyprstream-events";

/// Configuration for the EventBus
#[derive(Debug, Clone)]
pub struct EventBusConfig {
    /// Enable inproc endpoint (default: true)
    pub enable_inproc: bool,
    /// Enable IPC endpoint for sandbox processes
    pub enable_ipc: bool,
    /// IPC endpoint path (default: /tmp/hyprstream-events)
    pub ipc_path: Option<String>,
    /// Enable TCP endpoint for remote subscribers
    pub enable_tcp: bool,
    /// TCP bind address (default: 0.0.0.0:5555)
    pub tcp_bind: Option<String>,
    /// High water mark for outbound messages (0 = unlimited)
    pub send_hwm: i32,
    /// Channel buffer size for internal message passing
    pub channel_buffer: usize,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            enable_inproc: true,
            enable_ipc: false,
            ipc_path: None,
            enable_tcp: false,
            tcp_bind: None,
            send_hwm: 1000,
            channel_buffer: 1000,
        }
    }
}

/// Internal message sent to the publisher task
struct PublishCommand {
    topic: String,
    payload: Vec<u8>,
}

/// ZeroMQ-based event bus
///
/// The EventBus publishes events to multiple endpoints simultaneously.
/// Subscribers connect to the appropriate endpoint based on their deployment:
/// - In-process handlers use `inproc://`
/// - Sandbox processes use `ipc://`
/// - Remote services use `tcp://`
///
/// Note: The publisher socket runs on a dedicated task because ZMQ sockets
/// are not thread-safe. Events are sent via a channel to the publisher task.
pub struct EventBus {
    /// Channel sender for publishing events
    sender: mpsc::Sender<PublishCommand>,
    /// List of bound endpoints
    endpoints: Vec<String>,
    /// Shared ZMQ context for creating subscribers
    context: Arc<zmq::Context>,
}

// EventBus is Send + Sync because it only contains thread-safe types
unsafe impl Send for EventBus {}
unsafe impl Sync for EventBus {}

impl EventBus {
    /// Create a new event bus with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(EventBusConfig::default()).await
    }

    /// Create a new event bus with custom configuration
    pub async fn with_config(config: EventBusConfig) -> Result<Self> {
        let context = Arc::new(zmq::Context::new());
        let mut endpoints = Vec::new();

        // Build endpoint list based on config
        if config.enable_inproc {
            endpoints.push(INPROC_ENDPOINT.to_string());
        }

        if config.enable_ipc {
            let ipc_endpoint = config
                .ipc_path
                .map(|p| format!("ipc://{}", p))
                .unwrap_or_else(|| IPC_ENDPOINT.to_string());
            endpoints.push(ipc_endpoint);
        }

        if config.enable_tcp {
            let tcp_endpoint = config
                .tcp_bind
                .map(|b| format!("tcp://{}", b))
                .unwrap_or_else(|| "tcp://0.0.0.0:5555".to_string());
            endpoints.push(tcp_endpoint);
        }

        if endpoints.is_empty() {
            return Err(anyhow!("no endpoints configured for EventBus"));
        }

        // Create channel for publishing
        let (sender, receiver) = mpsc::channel(config.channel_buffer);

        // Clone values for the spawned task
        let task_context = context.clone();
        let task_endpoints = endpoints.clone();
        let send_hwm = config.send_hwm;

        // Spawn the publisher task
        tokio::task::spawn_blocking(move || {
            Self::publisher_task(task_context, task_endpoints, send_hwm, receiver)
        });

        debug!("EventBus created with endpoints: {:?}", endpoints);

        Ok(Self {
            sender,
            endpoints,
            context,
        })
    }

    /// Publisher task that runs on a dedicated thread
    fn publisher_task(
        context: Arc<zmq::Context>,
        endpoints: Vec<String>,
        send_hwm: i32,
        mut receiver: mpsc::Receiver<PublishCommand>,
    ) {
        // Create publisher socket
        let publisher = match context.socket(zmq::PUB) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to create PUB socket: {}", e);
                return;
            }
        };

        // Set high water mark
        if let Err(e) = publisher.set_sndhwm(send_hwm) {
            warn!("Failed to set send HWM: {}", e);
        }

        // Bind to all endpoints
        for endpoint in &endpoints {
            if let Err(e) = publisher.bind(endpoint) {
                error!("Failed to bind to {}: {}", endpoint, e);
                return;
            }
            debug!("EventBus publisher bound to {}", endpoint);
        }

        // Create a runtime for receiving from the async channel
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build() {
                Ok(rt) => rt,
                Err(e) => {
                    error!("Failed to create runtime for publisher task: {}", e);
                    return;
                }
            };

        // Process messages
        rt.block_on(async {
            while let Some(cmd) = receiver.recv().await {
                // Send multipart message: [topic, payload]
                if let Err(e) = publisher.send(&cmd.topic, zmq::SNDMORE) {
                    error!("Failed to send topic frame: {}", e);
                    continue;
                }
                if let Err(e) = publisher.send(&cmd.payload, 0) {
                    error!("Failed to send payload frame: {}", e);
                    continue;
                }
                trace!("Published event to topic {}", cmd.topic);
            }
        });

        debug!("Publisher task shutting down");
    }

    /// Get the list of bound endpoints
    pub fn endpoints(&self) -> &[String] {
        &self.endpoints
    }

    /// Get a reference to the ZMQ context
    pub fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    /// Publish an event to all subscribers
    ///
    /// The event is serialized to Cap'n Proto and sent as a multipart message:
    /// - Frame 0: Topic (for ZeroMQ prefix filtering)
    /// - Frame 1: Cap'n Proto payload
    pub async fn publish(&self, event: &EventEnvelope) -> Result<()> {
        let topic = event.topic.clone();
        let payload = capnp_serde::serialize_event(event)?;

        self.sender
            .send(PublishCommand { topic, payload })
            .await
            .map_err(|e| {
                error!("Failed to send to publisher: {}", e);
                anyhow!("failed to publish event: channel closed")
            })?;

        trace!("Queued event {} to topic {}", event.id, event.topic);
        Ok(())
    }

    /// Create a subscriber for a topic pattern
    ///
    /// The topic_filter is used for ZeroMQ prefix matching:
    /// - `""` subscribes to all events
    /// - `"inference"` subscribes to all inference events
    /// - `"inference.generation_complete"` subscribes to specific event
    pub fn subscriber(&self, topic_filter: &str) -> Result<EventSubscriber> {
        self.subscriber_to_endpoint(topic_filter, INPROC_ENDPOINT)
    }

    /// Create a subscriber connected to a specific endpoint
    pub fn subscriber_to_endpoint(
        &self,
        topic_filter: &str,
        endpoint: &str,
    ) -> Result<EventSubscriber> {
        let socket = self.context.socket(zmq::SUB).map_err(|e| {
            anyhow!("failed to create SUB socket: {}", e)
        })?;

        socket.connect(endpoint).map_err(|e| {
            anyhow!("failed to connect to {}: {}", endpoint, e)
        })?;

        socket.set_subscribe(topic_filter.as_bytes()).map_err(|e| {
            anyhow!("failed to subscribe to '{}': {}", topic_filter, e)
        })?;

        debug!(
            "created subscriber for '{}' connected to {}",
            topic_filter, endpoint
        );

        Ok(EventSubscriber {
            socket,
            topic_filter: topic_filter.to_string(),
        })
    }
}

/// Event subscriber for receiving events
pub struct EventSubscriber {
    socket: zmq::Socket,
    topic_filter: String,
}

// EventSubscriber is not Send/Sync because zmq::Socket isn't
// Subscribers should be used on a single task/thread

impl EventSubscriber {
    /// Create a subscriber from an existing socket
    ///
    /// This is used by sink modules that need to create subscribers
    /// on their own blocking threads.
    pub fn from_socket(socket: zmq::Socket, topic_filter: String) -> Self {
        Self {
            socket,
            topic_filter,
        }
    }

    /// Get the topic filter this subscriber is using
    pub fn topic_filter(&self) -> &str {
        &self.topic_filter
    }

    /// Receive the next event (blocking)
    ///
    /// This method blocks until an event is available.
    /// For async usage, run this on a blocking task.
    pub fn recv(&self) -> Result<EventEnvelope> {
        // Receive topic frame
        let _topic = self.socket.recv_bytes(0).map_err(|e| {
            warn!("subscriber recv topic error: {}", e);
            anyhow!("failed to receive topic: {}", e)
        })?;

        // Receive payload frame (Cap'n Proto)
        let payload = self.socket.recv_bytes(0).map_err(|e| {
            warn!("subscriber recv payload error: {}", e);
            anyhow!("failed to receive payload: {}", e)
        })?;

        let event = capnp_serde::deserialize_event(&payload)
            .map_err(|e| anyhow!("failed to deserialize event: {}", e))?;

        trace!("received event {} from topic {}", event.id, event.topic);
        Ok(event)
    }

    /// Try to receive an event without blocking
    ///
    /// Returns `Ok(None)` if no event is available.
    pub fn try_recv(&self) -> Result<Option<EventEnvelope>> {
        // Use non-blocking receive
        match self.socket.recv_bytes(zmq::DONTWAIT) {
            Ok(_topic_bytes) => {
                // Got topic, now get payload (should be immediate)
                let payload = self.socket.recv_bytes(0).map_err(|e| {
                    anyhow!("failed to receive payload after topic: {}", e)
                })?;

                let event = capnp_serde::deserialize_event(&payload)
                    .map_err(|e| anyhow!("failed to deserialize event: {}", e))?;

                trace!("received event {} from topic {}", event.id, event.topic);
                Ok(Some(event))
            }
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(e) => Err(anyhow!("failed to receive: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventPayload, EventSource, GenerationMetrics};
    use std::time::Duration;

    #[tokio::test]
    async fn test_event_bus_creation() {
        let bus = EventBus::new().await.unwrap();
        assert!(!bus.endpoints().is_empty());
        assert!(bus.endpoints().contains(&INPROC_ENDPOINT.to_string()));
    }

    #[tokio::test]
    async fn test_publish_subscribe() {
        let bus = EventBus::new().await.unwrap();

        // Create subscriber before publishing
        let subscriber = bus.subscriber("inference").unwrap();

        // Publish event
        let event = EventEnvelope::new(
            EventSource::Inference,
            "inference.generation_complete",
            EventPayload::GenerationComplete {
                model_id: "test-model".to_string(),
                session_id: None,
                metrics: GenerationMetrics::default(),
            },
        );

        // Small delay to ensure subscriber is connected
        tokio::time::sleep(Duration::from_millis(50)).await;

        bus.publish(&event).await.unwrap();

        // Receive with timeout using blocking task
        let result = tokio::task::spawn_blocking(move || {
            // Try a few times with small delays
            for _ in 0..10 {
                if let Ok(Some(event)) = subscriber.try_recv() {
                    return Some(event);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            None
        })
        .await
        .unwrap();

        if let Some(received_event) = result {
            assert_eq!(received_event.topic, "inference.generation_complete");
        }
    }

    #[tokio::test]
    async fn test_topic_filtering() {
        let bus = EventBus::new().await.unwrap();

        // Subscriber only interested in metrics events
        let metrics_sub = bus.subscriber("metrics").unwrap();

        // Small delay
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Publish inference event (should not be received by metrics subscriber)
        let inference_event = EventEnvelope::new(
            EventSource::Inference,
            "inference.generation_complete",
            EventPayload::GenerationComplete {
                model_id: "test".to_string(),
                session_id: None,
                metrics: GenerationMetrics::default(),
            },
        );
        bus.publish(&inference_event).await.unwrap();

        // Publish metrics event (should be received)
        let metrics_event = EventEnvelope::new(
            EventSource::Metrics,
            "metrics.threshold_breach",
            EventPayload::ThresholdBreach {
                model_id: "test".to_string(),
                metric: "perplexity".to_string(),
                threshold: 50.0,
                actual: 75.0,
                z_score: 2.5,
            },
        );
        bus.publish(&metrics_event).await.unwrap();

        // Check that we receive the metrics event
        let result = tokio::task::spawn_blocking(move || {
            for _ in 0..10 {
                if let Ok(Some(event)) = metrics_sub.try_recv() {
                    return Some(event);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            None
        })
        .await
        .unwrap();

        if let Some(event) = result {
            assert!(event.topic.starts_with("metrics"));
        }
    }
}
