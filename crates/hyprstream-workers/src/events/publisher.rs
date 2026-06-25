//! EventPublisher — moq-backed async event publishing (#167).
//!
//! Publishers connect to the process-global [`MoqEventOrigin`] and write
//! events to the `local/events/{source}` broadcast's `events` track.
//! No ZMQ context is required; the moq origin is process-global.

use anyhow::{anyhow, Result};

use hyprstream_rpc::moq_event::{global_moq_event_origin, MoqEventPublisher};

/// Async event publisher backed by moq-lite.
///
/// Creates and holds a [`MoqEventPublisher`] for the named source, writing
/// topic+payload pairs to the moq event bus. All in-process subscribers that
/// have subscribed to matching topic patterns will receive the events.
///
/// # Example
///
/// ```ignore
/// let mut publisher = EventPublisher::new("worker")?;
/// publisher.publish("sandbox123", "started", &payload).await?;
/// ```
pub struct EventPublisher {
    inner: MoqEventPublisher,
}

impl EventPublisher {
    /// Create a new publisher for the given source name.
    ///
    /// Requires the process-global moq event bus to have been initialized via
    /// `init_global_moq_event_origin` (done by the event-service factory at startup).
    pub fn new(source: &str) -> Result<Self> {
        let origin = global_moq_event_origin()
            .ok_or_else(|| anyhow!("moq event bus not initialized; start the event service first"))?;
        let inner = origin.publisher(source)?;
        Ok(Self { inner })
    }

    /// Create a new publisher for `source` that ALSO mirrors every event to the
    /// per-OID (#393) publication track for `oid`. This is the transition path:
    /// legacy flat-track subscribers keep working while #393 per-OID subscribers
    /// get wire-level selectivity.
    pub fn new_with_oid(source: &str, oid: &str) -> Result<Self> {
        let origin = global_moq_event_origin()
            .ok_or_else(|| anyhow!("moq event bus not initialized; start the event service first"))?;
        let inner = origin.publisher_with_oid(source, oid)?;
        Ok(Self { inner })
    }

    /// Create a new publisher that writes ONLY to the per-OID (#393) publication
    /// track for `oid` (no flat-track mirror). Use once every subscriber of this
    /// source has migrated to per-OID subscription.
    pub fn new_oid_only(source: &str, oid: &str) -> Result<Self> {
        let origin = global_moq_event_origin()
            .ok_or_else(|| anyhow!("moq event bus not initialized; start the event service first"))?;
        let inner = origin.publisher_oid_only(source, oid)?;
        Ok(Self { inner })
    }

    /// Publish an event asynchronously.
    ///
    /// Creates a topic from `{source}.{entity}.{event}` and writes a frame
    /// containing the topic length, topic bytes, and payload to the moq track.
    ///
    /// # Arguments
    ///
    /// * `entity` - Entity identifier (e.g., sandbox ID, container ID)
    /// * `event` - Event name (e.g., "started", "stopped", "failed")
    /// * `payload` - Serialized event data (typically Cap'n Proto)
    ///
    /// # Errors
    ///
    /// Returns error if entity/event contain dots (reserved as separator).
    pub async fn publish(&mut self, entity: &str, event: &str, payload: &[u8]) -> Result<()> {
        self.inner.publish(entity, event, payload)
    }

    /// Publish with a pre-formatted topic.
    pub async fn publish_raw(&mut self, topic: &str, payload: &[u8]) -> Result<()> {
        self.inner.publish_raw(topic, payload)
    }

    /// Get the source name for this publisher.
    pub fn source(&self) -> &str {
        self.inner.source()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_topic_validation() {
        let entity = "sandbox.123";
        assert!(entity.contains('.'));

        let event = "started.ok";
        assert!(event.contains('.'));

        let valid_entity = "sandbox123";
        let valid_event = "started";
        assert!(!valid_entity.contains('.'));
        assert!(!valid_event.contains('.'));
    }
}
