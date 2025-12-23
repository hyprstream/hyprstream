//! Sink implementations for the event bus
//!
//! Each sink type has a corresponding loop function that receives events
//! from an EventSubscriber and forwards them to the appropriate destination.
//!
//! Note: All sink loops run on blocking threads because ZMQ sockets are not
//! thread-safe. The subscriber is created on the same thread that will use it.

mod mcp;
mod nats;
mod webhook;

pub use mcp::mcp_loop;
pub use nats::nats_loop;
pub use webhook::webhook_loop;

use super::bus::EventSubscriber;
use super::EventEnvelope;
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

/// In-process handler loop
///
/// For in-process handlers, we log the events. The actual handler logic
/// is typically implemented via `SinkRegistry::register_handler()` which
/// allows custom async closures.
///
/// This runs on a blocking thread.
pub fn in_process_loop(subscriber: EventSubscriber, handler: &str) {
    info!(
        "starting in-process sink '{}' for topic '{}'",
        handler,
        subscriber.topic_filter()
    );

    loop {
        match subscriber.recv() {
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
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

/// Container sink loop
///
/// Executes events in a container sandbox. The container receives the event
/// as JSON on stdin and can emit response events.
///
/// This runs on a blocking thread.
pub fn container_loop(subscriber: EventSubscriber, image: &str, runtime: Option<&str>) {
    let runtime = runtime.unwrap_or("podman");
    info!(
        "starting container sink '{}' (runtime: {}) for topic '{}'",
        image,
        runtime,
        subscriber.topic_filter()
    );

    loop {
        match subscriber.recv() {
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
                // let output = std::process::Command::new(runtime)
                //     .args(["run", "--rm", "-i", image])
                //     .stdin(std::process::Stdio::piped())
                //     .stdout(std::process::Stdio::piped())
                //     .spawn();
            }
            Err(e) => {
                warn!("container sink '{}' recv error: {}", image, e);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

/// Create a subscriber on the current thread
///
/// This helper creates a subscriber using the provided ZMQ context.
/// It should be called from within a blocking thread context.
pub fn create_subscriber(
    context: &Arc<zmq::Context>,
    topic_filter: &str,
    endpoint: &str,
) -> anyhow::Result<EventSubscriber> {
    let socket = context
        .socket(zmq::SUB)
        .map_err(|e| anyhow::anyhow!("failed to create SUB socket: {}", e))?;

    socket
        .connect(endpoint)
        .map_err(|e| anyhow::anyhow!("failed to connect to {}: {}", endpoint, e))?;

    socket
        .set_subscribe(topic_filter.as_bytes())
        .map_err(|e| anyhow::anyhow!("failed to subscribe to '{}': {}", topic_filter, e))?;

    debug!(
        "created subscriber for '{}' connected to {}",
        topic_filter, endpoint
    );

    Ok(EventSubscriber::from_socket(socket, topic_filter.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::bus::EventBus;

    #[tokio::test]
    async fn test_subscriber_creation() {
        let bus = Arc::new(EventBus::new().await.unwrap());
        let subscriber = bus.subscriber("test").unwrap();
        assert_eq!(subscriber.topic_filter(), "test");
    }
}
