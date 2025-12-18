//! Sink implementations for the event bus
//!
//! Each sink type has a corresponding loop function that receives events
//! from an EventSubscriber and forwards them to the appropriate destination.

mod mcp;
mod nats;
mod webhook;

pub use mcp::mcp_loop;
pub use nats::nats_loop;
pub use webhook::webhook_loop;

use super::bus::EventSubscriber;
use tracing::{debug, error, info, trace, warn};

/// In-process handler loop
///
/// For in-process handlers, we log the events. The actual handler logic
/// is typically implemented via `SinkRegistry::register_handler()` which
/// allows custom async closures.
pub async fn in_process_loop(mut subscriber: EventSubscriber, handler: &str) {
    info!(
        "starting in-process sink '{}' for topic '{}'",
        handler,
        subscriber.topic_filter()
    );

    loop {
        match subscriber.recv().await {
            Ok(event) => {
                debug!(
                    "in-process handler '{}' received event {} (topic: {})",
                    handler, event.id, event.topic
                );
                // For string-based handlers, we just log.
                // Actual handling is done via register_handler() with closures.
                trace!("event payload: {:?}", event.payload);
            }
            Err(e) => {
                warn!("in-process handler '{}' recv error: {}", handler, e);
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}

/// Container sink loop
///
/// Executes events in a container sandbox. The container receives the event
/// as JSON on stdin and can emit response events.
pub async fn container_loop(
    mut subscriber: EventSubscriber,
    image: &str,
    runtime: Option<&str>,
) {
    let runtime = runtime.unwrap_or("podman");
    info!(
        "starting container sink '{}' (runtime: {}) for topic '{}'",
        image,
        runtime,
        subscriber.topic_filter()
    );

    loop {
        match subscriber.recv().await {
            Ok(event) => {
                debug!(
                    "container sink '{}' received event {} (topic: {})",
                    image, event.id, event.topic
                );

                // Serialize event to JSON
                let json = match serde_json::to_string(&event) {
                    Ok(j) => j,
                    Err(e) => {
                        error!("failed to serialize event: {}", e);
                        continue;
                    }
                };

                // Execute container
                // TODO: Implement actual container execution via hyprstream-sandbox
                // For now, we log what would be executed
                trace!(
                    "would execute: {} run --rm -i {} <<< '{}'",
                    runtime,
                    image,
                    json
                );

                // Placeholder for actual implementation:
                // let output = tokio::process::Command::new(runtime)
                //     .args(["run", "--rm", "-i", image])
                //     .stdin(std::process::Stdio::piped())
                //     .stdout(std::process::Stdio::piped())
                //     .spawn();
            }
            Err(e) => {
                warn!("container sink '{}' recv error: {}", image, e);
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::bus::EventBus;
    use crate::events::{EventEnvelope, EventPayload, EventSource, GenerationMetrics};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_subscriber_creation() {
        let bus = Arc::new(EventBus::new().await.unwrap());
        let subscriber = bus.subscriber("test").await.unwrap();
        assert_eq!(subscriber.topic_filter(), "test");
    }
}
