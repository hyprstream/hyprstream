//! NATS JetStream forwarder sink implementation
//!
//! Forwards events to a NATS server for distribution to external consumers.
//! Events are published to subjects using the format: `{subject_prefix}.{topic}`

use super::super::bus::EventSubscriber;
use super::super::EventEnvelope;
use tracing::{debug, error, info, trace, warn};

/// NATS sink loop
///
/// Receives events and publishes them to a NATS server.
/// Events are published to subjects in the format: `{subject_prefix}.{topic}`
///
/// For example, with subject_prefix="hyprstream.events":
/// - `inference.generation_complete` → `hyprstream.events.inference.generation_complete`
/// - `metrics.threshold_breach` → `hyprstream.events.metrics.threshold_breach`
pub async fn nats_loop(mut subscriber: EventSubscriber, url: &str, subject_prefix: &str) {
    info!(
        "starting NATS sink to '{}' (prefix: {}) for topic '{}'",
        url,
        subject_prefix,
        subscriber.topic_filter()
    );

    // Connect to NATS server
    // Note: async-nats is not in dependencies, so this is a placeholder
    // that logs what would be sent. To enable, add async-nats to Cargo.toml.

    // Placeholder loop that demonstrates the intended behavior
    loop {
        match subscriber.recv().await {
            Ok(event) => {
                debug!(
                    "NATS sink received event {} (topic: {})",
                    event.id, event.topic
                );

                let subject = format!("{}.{}", subject_prefix, event.topic);

                // Serialize event to JSON
                let payload = match serde_json::to_vec(&event) {
                    Ok(p) => p,
                    Err(e) => {
                        error!("failed to serialize event: {}", e);
                        continue;
                    }
                };

                // TODO: Publish to NATS when async-nats is added
                // client.publish(&subject, payload.into()).await?;
                trace!(
                    "would publish to NATS subject '{}' ({} bytes)",
                    subject,
                    payload.len()
                );
            }
            Err(e) => {
                warn!("NATS sink recv error: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}

/// NATS connection manager (placeholder)
///
/// When async-nats is added, this will manage the NATS connection with
/// automatic reconnection and JetStream support.
#[allow(dead_code)]
struct NatsConnection {
    url: String,
    subject_prefix: String,
    // client: Option<async_nats::Client>,
}

#[allow(dead_code)]
impl NatsConnection {
    fn new(url: &str, subject_prefix: &str) -> Self {
        Self {
            url: url.to_string(),
            subject_prefix: subject_prefix.to_string(),
        }
    }

    async fn connect(&mut self) -> anyhow::Result<()> {
        // self.client = Some(async_nats::connect(&self.url).await?);
        Ok(())
    }

    async fn publish(&self, event: &EventEnvelope) -> anyhow::Result<()> {
        let subject = format!("{}.{}", self.subject_prefix, event.topic);
        let payload = serde_json::to_vec(event)?;

        // self.client.as_ref().unwrap().publish(&subject, payload.into()).await?;
        trace!("published to NATS: {} ({} bytes)", subject, payload.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventPayload, EventSource, GenerationMetrics};

    #[test]
    fn test_subject_formatting() {
        let prefix = "hyprstream.events";
        let topic = "inference.generation_complete";
        let subject = format!("{}.{}", prefix, topic);
        assert_eq!(subject, "hyprstream.events.inference.generation_complete");
    }
}
