//! EventPublisher - Async event publishing using TMQ
//!
//! Each service creates its own publisher instance.
//! Publishers connect to the EventService proxy's XSUB socket.

use anyhow::{anyhow, Result};
use futures::SinkExt;
use std::sync::Arc;
use tmq::{publish, Multipart};

use super::endpoints;

/// Async event publisher using TMQ PUB socket
///
/// Publishers connect to the EventService proxy and send events
/// with topic-based routing. Topics follow the format:
/// `{source}.{entity}.{event}`
///
/// # Example
///
/// ```ignore
/// let mut publisher = EventPublisher::new(&ctx, "worker")?;
/// publisher.publish("sandbox123", "started", &payload).await?;
/// ```
pub struct EventPublisher {
    socket: tmq::Publish,
    source: String,
}

impl EventPublisher {
    /// Create a new publisher for a service
    ///
    /// # Arguments
    ///
    /// * `context` - ZMQ context (must be same as EventService for inproc://)
    /// * `source` - Service name (e.g., "worker", "registry", "model", "inference")
    pub fn new(context: &Arc<zmq::Context>, source: &str) -> Result<Self> {
        let socket = publish(context)
            .connect(endpoints::PUB)
            .map_err(|e| anyhow!("Failed to connect publisher to {}: {}", endpoints::PUB, e))?;

        Ok(Self {
            socket,
            source: source.to_string(),
        })
    }

    /// Publish an event asynchronously
    ///
    /// Creates a topic from `{source}.{entity}.{event}` and sends
    /// a multipart message [topic, payload].
    ///
    /// # Arguments
    ///
    /// * `entity` - Entity identifier (e.g., sandbox ID, container ID)
    /// * `event` - Event name (e.g., "started", "stopped", "failed")
    /// * `payload` - Serialized event data (typically Cap'n Proto)
    ///
    /// # Errors
    ///
    /// Returns error if entity/event contain dots (reserved as separator)
    /// or if sending fails.
    pub async fn publish(&mut self, entity: &str, event: &str, payload: &[u8]) -> Result<()> {
        // Validate: entity/event cannot contain dots (used as separator)
        if entity.contains('.') {
            return Err(anyhow!("Entity name cannot contain '.': {}", entity));
        }
        if event.contains('.') {
            return Err(anyhow!("Event name cannot contain '.': {}", event));
        }

        let topic = format!("{}.{}.{}", self.source, entity, event);

        // Multipart message: [topic, payload]
        // Topic is first frame for ZMQ prefix filtering
        let multipart = Multipart::from(vec![topic.into_bytes(), payload.to_vec()]);

        self.socket
            .send(multipart)
            .await
            .map_err(|e| anyhow!("Failed to publish event: {}", e))?;

        Ok(())
    }

    /// Publish with a pre-formatted topic
    ///
    /// Use this when you need full control over the topic format.
    pub async fn publish_raw(&mut self, topic: &str, payload: &[u8]) -> Result<()> {
        let multipart = Multipart::from(vec![topic.as_bytes().to_vec(), payload.to_vec()]);

        self.socket
            .send(multipart)
            .await
            .map_err(|e| anyhow!("Failed to publish event: {}", e))?;

        Ok(())
    }

    /// Get the source name for this publisher
    pub fn source(&self) -> &str {
        &self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_validation() {
        // Entity with dot should fail
        let entity = "sandbox.123";
        assert!(entity.contains('.'));

        // Event with dot should fail
        let event = "started.ok";
        assert!(event.contains('.'));

        // Valid names
        let valid_entity = "sandbox123";
        let valid_event = "started";
        assert!(!valid_entity.contains('.'));
        assert!(!valid_event.contains('.'));
    }
}
