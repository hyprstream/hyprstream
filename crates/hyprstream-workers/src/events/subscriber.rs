//! EventSubscriber - Async event subscription using TMQ
//!
//! Subscribers connect to the EventService proxy's XPUB socket
//! and receive events matching their subscription patterns.

use anyhow::{anyhow, Result};
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tmq::subscribe;

use super::endpoints;

/// Async event subscriber using TMQ SUB socket
///
/// Subscribers connect to the EventService proxy and receive events
/// matching their subscription patterns. Patterns use prefix matching.
///
/// **Note**: You must call `subscribe()` at least once before calling `recv()`.
/// ZMQ SUB sockets don't receive any messages until subscribed to a topic.
///
/// # Example
///
/// ```ignore
/// let mut subscriber = EventSubscriber::new(&ctx)?;
/// subscriber.subscribe("worker.")?;  // All worker events
///
/// while let Ok((topic, payload)) = subscriber.recv().await {
///     println!("Received: {}", topic);
/// }
/// ```
pub struct EventSubscriber {
    /// Before first subscribe: Some(SubscribeWithoutTopic)
    /// After first subscribe: None (moved to `subscribed`)
    unsubscribed: Option<tmq::SubscribeWithoutTopic>,
    /// After first subscribe: Some(Subscribe)
    subscribed: Option<tmq::Subscribe>,
}

impl EventSubscriber {
    /// Create a new subscriber
    ///
    /// # Arguments
    ///
    /// * `context` - ZMQ context (must be same as EventService for inproc://)
    ///
    /// # Note
    ///
    /// The subscriber won't receive any messages until `subscribe()` is called.
    pub fn new(context: &Arc<zmq::Context>) -> Result<Self> {
        let socket = subscribe(context)
            .connect(endpoints::SUB)
            .map_err(|e| anyhow!("Failed to connect subscriber to {}: {}", endpoints::SUB, e))?;

        Ok(Self {
            unsubscribed: Some(socket),
            subscribed: None,
        })
    }

    /// Subscribe to a topic pattern (prefix match)
    ///
    /// # Examples
    ///
    /// - `"worker."` - All worker events
    /// - `"worker.sandbox123."` - Events for specific sandbox
    /// - `"worker.sandbox123.started"` - Exact topic match
    /// - `""` - All events (empty string = subscribe all)
    pub fn subscribe(&mut self, pattern: &str) -> Result<()> {
        if let Some(socket) = self.unsubscribed.take() {
            // First subscription - converts SubscribeWithoutTopic to Subscribe
            let subscribed = socket
                .subscribe(pattern.as_bytes())
                .map_err(|e| anyhow!("Failed to subscribe to '{}': {}", pattern, e))?;
            self.subscribed = Some(subscribed);
        } else if let Some(ref mut socket) = self.subscribed {
            // Already subscribed - add another topic
            socket
                .subscribe(pattern.as_bytes())
                .map_err(|e| anyhow!("Failed to subscribe to '{}': {}", pattern, e))?;
        } else {
            return Err(anyhow!("Subscriber in invalid state"));
        }

        Ok(())
    }

    /// Subscribe to all events
    pub fn subscribe_all(&mut self) -> Result<()> {
        self.subscribe("")
    }

    /// Unsubscribe from a topic pattern
    pub fn unsubscribe(&mut self, pattern: &str) -> Result<()> {
        match &mut self.subscribed {
            None => Err(anyhow!("Cannot unsubscribe: no active subscriptions")),
            Some(socket) => {
                socket
                    .unsubscribe(pattern.as_bytes())
                    .map_err(|e| anyhow!("Failed to unsubscribe from '{}': {}", pattern, e))?;
                Ok(())
            }
        }
    }

    /// Receive the next event asynchronously
    ///
    /// Returns (topic, payload) tuple.
    /// Blocks until an event is received or the stream ends.
    ///
    /// # Errors
    ///
    /// Returns error if `subscribe()` hasn't been called yet.
    pub async fn recv(&mut self) -> Result<(String, Vec<u8>)> {
        let socket = self.subscribed.as_mut().ok_or_else(|| {
            anyhow!("No subscriptions active. Call subscribe() before recv()")
        })?;

        match socket.next().await {
            Some(Ok(multipart)) => {
                let parts: Vec<_> = multipart.into_iter().collect();
                if parts.len() < 2 {
                    return Err(anyhow!(
                        "Invalid message format: expected 2 frames, got {}",
                        parts.len()
                    ));
                }
                let topic = String::from_utf8(parts[0].to_vec())
                    .map_err(|e| anyhow!("Invalid UTF-8 in topic: {}", e))?;
                let payload = parts[1].to_vec();
                Ok((topic, payload))
            }
            Some(Err(e)) => Err(anyhow!("Receive error: {}", e)),
            None => Err(anyhow!("Subscriber stream ended")),
        }
    }

    /// Receive with timeout
    ///
    /// Returns `Ok(Some((topic, payload)))` if event received,
    /// `Ok(None)` if timeout elapsed, or `Err` on error.
    pub async fn recv_timeout(&mut self, timeout: Duration) -> Result<Option<(String, Vec<u8>)>> {
        match tokio::time::timeout(timeout, self.recv()).await {
            Ok(result) => result.map(Some),
            Err(_) => Ok(None),
        }
    }

    /// Try to receive without blocking
    ///
    /// Returns `Ok(Some((topic, payload)))` if event available,
    /// `Ok(None)` if no event ready.
    pub async fn try_recv(&mut self) -> Result<Option<(String, Vec<u8>)>> {
        self.recv_timeout(Duration::from_millis(0)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_patterns() {
        // These are valid subscription patterns
        let patterns = vec![
            "",                           // All events
            "worker.",                    // All worker events
            "worker.sandbox123.",         // Specific sandbox
            "worker.sandbox123.started",  // Exact match
        ];

        for pattern in patterns {
            assert!(pattern.is_ascii(), "Pattern should be ASCII: {}", pattern);
        }
    }
}
