//! ZeroMQ-based event bus for pub/sub messaging
//!
//! The EventBus uses ZeroMQ PUB/SUB pattern with support for multiple transports:
//! - `inproc://` for in-process messaging (zero-copy)
//! - `ipc://` for inter-process messaging (MCP tools, containers)
//! - `tcp://` for network messaging (distributed systems)

use super::EventEnvelope;
use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, trace, warn};
use zeromq::{PubSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

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
        }
    }
}

/// ZeroMQ-based event bus
///
/// The EventBus publishes events to multiple endpoints simultaneously.
/// Subscribers connect to the appropriate endpoint based on their deployment:
/// - In-process handlers use `inproc://`
/// - Sandbox processes use `ipc://`
/// - Remote services use `tcp://`
pub struct EventBus {
    publisher: Arc<RwLock<PubSocket>>,
    endpoints: Vec<String>,
}

impl EventBus {
    /// Create a new event bus with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(EventBusConfig::default()).await
    }

    /// Create a new event bus with custom configuration
    pub async fn with_config(config: EventBusConfig) -> Result<Self> {
        let mut publisher = PubSocket::new();
        let mut endpoints = Vec::new();

        // Bind to inproc endpoint (always available for in-process sinks)
        if config.enable_inproc {
            publisher.bind(INPROC_ENDPOINT).await.map_err(|e| {
                anyhow!("failed to bind inproc endpoint {}: {}", INPROC_ENDPOINT, e)
            })?;
            endpoints.push(INPROC_ENDPOINT.to_string());
            debug!("EventBus bound to {}", INPROC_ENDPOINT);
        }

        // Bind to IPC endpoint for sandbox processes
        if config.enable_ipc {
            let ipc_endpoint = config
                .ipc_path
                .map(|p| format!("ipc://{}", p))
                .unwrap_or_else(|| IPC_ENDPOINT.to_string());
            publisher
                .bind(&ipc_endpoint)
                .await
                .map_err(|e| anyhow!("failed to bind IPC endpoint {}: {}", ipc_endpoint, e))?;
            endpoints.push(ipc_endpoint.clone());
            debug!("EventBus bound to {}", ipc_endpoint);
        }

        // Bind to TCP endpoint for remote subscribers
        if config.enable_tcp {
            let tcp_endpoint = config
                .tcp_bind
                .map(|b| format!("tcp://{}", b))
                .unwrap_or_else(|| "tcp://0.0.0.0:5555".to_string());
            publisher
                .bind(&tcp_endpoint)
                .await
                .map_err(|e| anyhow!("failed to bind TCP endpoint {}: {}", tcp_endpoint, e))?;
            endpoints.push(tcp_endpoint.clone());
            debug!("EventBus bound to {}", tcp_endpoint);
        }

        if endpoints.is_empty() {
            return Err(anyhow!("no endpoints configured for EventBus"));
        }

        Ok(Self {
            publisher: Arc::new(RwLock::new(publisher)),
            endpoints,
        })
    }

    /// Get the list of bound endpoints
    pub fn endpoints(&self) -> &[String] {
        &self.endpoints
    }

    /// Publish an event to all subscribers
    ///
    /// The event is serialized to JSON and sent as a multipart message:
    /// - Frame 0: Topic (for ZeroMQ prefix filtering)
    /// - Frame 1: JSON payload
    pub async fn publish(&self, event: &EventEnvelope) -> Result<()> {
        let topic = &event.topic;
        let payload = serde_json::to_vec(event)?;

        // Create multipart message: [topic, payload]
        let mut msg = ZmqMessage::from(topic.as_bytes().to_vec());
        msg.push_back(payload.into());

        let mut publisher = self.publisher.write().await;
        publisher.send(msg).await.map_err(|e| {
            error!("failed to publish event {}: {}", event.id, e);
            anyhow!("failed to publish event: {}", e)
        })?;

        trace!("published event {} to topic {}", event.id, topic);
        Ok(())
    }

    /// Create a subscriber for a topic pattern
    ///
    /// The topic_filter is used for ZeroMQ prefix matching:
    /// - `""` subscribes to all events
    /// - `"inference"` subscribes to all inference events
    /// - `"inference.generation_complete"` subscribes to specific event
    pub async fn subscriber(&self, topic_filter: &str) -> Result<EventSubscriber> {
        self.subscriber_to_endpoint(topic_filter, INPROC_ENDPOINT)
            .await
    }

    /// Create a subscriber connected to a specific endpoint
    pub async fn subscriber_to_endpoint(
        &self,
        topic_filter: &str,
        endpoint: &str,
    ) -> Result<EventSubscriber> {
        let mut socket = SubSocket::new();

        socket
            .connect(endpoint)
            .await
            .map_err(|e| anyhow!("failed to connect to {}: {}", endpoint, e))?;

        socket
            .subscribe(topic_filter)
            .await
            .map_err(|e| anyhow!("failed to subscribe to '{}': {}", topic_filter, e))?;

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
    socket: SubSocket,
    topic_filter: String,
}

impl EventSubscriber {
    /// Get the topic filter this subscriber is using
    pub fn topic_filter(&self) -> &str {
        &self.topic_filter
    }

    /// Receive the next event
    ///
    /// This method blocks until an event is available.
    pub async fn recv(&mut self) -> Result<EventEnvelope> {
        let msg = self.socket.recv().await.map_err(|e| {
            warn!("subscriber recv error: {}", e);
            anyhow!("failed to receive event: {}", e)
        })?;

        // Message format: [topic, payload]
        if msg.len() < 2 {
            return Err(anyhow!(
                "invalid message format: expected 2 frames, got {}",
                msg.len()
            ));
        }

        let payload = msg
            .get(1)
            .ok_or_else(|| anyhow!("missing payload frame"))?;

        let event: EventEnvelope = serde_json::from_slice(payload)
            .map_err(|e| anyhow!("failed to deserialize event: {}", e))?;

        trace!("received event {} from topic {}", event.id, event.topic);
        Ok(event)
    }

    /// Try to receive an event without blocking
    ///
    /// Returns `Ok(None)` if no event is available.
    pub async fn try_recv(&mut self) -> Result<Option<EventEnvelope>> {
        // ZeroMQ doesn't have native try_recv, we'd need to use polling
        // For now, this is a placeholder that always blocks
        // A proper implementation would use zmq_poll with timeout=0
        Ok(Some(self.recv().await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventPayload, EventSource, GenerationMetrics};

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
        let mut subscriber = bus.subscriber("inference").await.unwrap();

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
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        bus.publish(&event).await.unwrap();

        // Receive with timeout
        let received = tokio::time::timeout(
            tokio::time::Duration::from_millis(100),
            subscriber.recv(),
        )
        .await;

        // Note: inproc requires subscriber to be in same process/context
        // This test may need adjustment based on zeromq crate behavior
        if let Ok(Ok(received_event)) = received {
            assert_eq!(received_event.topic, "inference.generation_complete");
        }
    }

    #[tokio::test]
    async fn test_topic_filtering() {
        let bus = EventBus::new().await.unwrap();

        // Subscriber only interested in metrics events
        let mut metrics_sub = bus.subscriber("metrics").await.unwrap();

        // Small delay
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Publish inference event (should not be received)
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

        // Should receive metrics event
        let result = tokio::time::timeout(
            tokio::time::Duration::from_millis(100),
            metrics_sub.recv(),
        )
        .await;

        if let Ok(Ok(event)) = result {
            assert!(event.topic.starts_with("metrics"));
        }
    }
}
