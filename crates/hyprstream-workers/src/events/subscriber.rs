//! EventSubscriber — moq-backed async event subscription (#167).
//!
//! Subscribers connect to the process-global [`MoqEventOrigin`] via a background
//! task that watches for announced source broadcasts and reads their `events` tracks.
//! No external context is required.

use anyhow::Result;
use std::time::Duration;

use hyprstream_rpc::moq_event::MoqEventSubscriber as Inner;

/// Async event subscriber backed by moq-lite.
///
/// Wraps [`MoqEventSubscriber`] from `hyprstream-rpc`. Patterns use dot-separated
/// prefix matching using dot-separated topic prefixes.
///
/// **Note**: Call `subscribe()` at least once before calling `recv()`.
///
/// # Example
///
/// ```ignore
/// let mut subscriber = EventSubscriber::new()?;
/// subscriber.subscribe("worker.")?;  // All worker events
///
/// while let Ok((topic, payload)) = subscriber.recv().await {
///     println!("Received: {}", topic);
/// }
/// ```
pub struct EventSubscriber {
    inner: Inner,
}

impl EventSubscriber {
    /// Create a new subscriber.
    ///
    /// No background task is started until the first `recv()` call.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: Inner::new(),
        })
    }

    /// Subscribe to a topic pattern (prefix match).
    ///
    /// # Examples
    ///
    /// - `"worker."` - All worker events
    /// - `"worker.sandbox123."` - Events for specific sandbox
    /// - `"worker.sandbox123.started"` - Exact topic match
    /// - `""` - All events (subscribe-all)
    pub fn subscribe(&mut self, pattern: &str) -> Result<()> {
        self.inner.subscribe(pattern)
    }

    /// Subscribe to all events.
    pub fn subscribe_all(&mut self) -> Result<()> {
        self.inner.subscribe_all()
    }

    /// Unsubscribe from a topic pattern. Must be called before `recv()`.
    pub fn unsubscribe(&mut self, pattern: &str) -> Result<()> {
        self.inner.unsubscribe(pattern)
    }

    /// Subscribe to a single model OID's per-OID publication track (#393).
    ///
    /// The subscriber scopes to `local/events/publications/{oid_hash}` and
    /// receives only events published for `oid` — not the whole firehose. This
    /// is mutually exclusive with [`Self::subscribe`] / [`Self::subscribe_all`]
    /// and must be called before the first `recv()`.
    pub fn subscribe_oid(&mut self, oid: &str) -> Result<()> {
        self.inner.subscribe_oid(oid)
    }

    /// Set the late-join retention mode (#393 decision A: firehose-backfill).
    ///
    /// On first `recv()`, a [`BackfillMode::FirehoseBackfill`] subscriber asks
    /// its [`BackfillSource`] for `oid` history and replays it before going live.
    /// If the source is unavailable the subscriber silently degrades to
    /// live-only (never fails). Must be called before the first `recv()`.
    pub fn with_backfill(&mut self, mode: hyprstream_rpc::moq_event::BackfillMode) -> Result<()> {
        self.inner.with_backfill(mode)
    }

    /// Select the delivery QoS (#606). Pass
    /// `hyprstream_rpc::stream_info::EventReliable::stream_opt()` for
    /// at-least-once delivery (events that must not be silently dropped).
    /// Defaults to `EventLive` (at-most-once, drop-oldest) if never called.
    /// Must be called before the first `recv()`.
    pub fn with_qos(&mut self, qos: hyprstream_rpc::stream_info::StreamOpt) -> Result<()> {
        self.inner.with_qos(qos)
    }

    /// Skip live events already delivered in a prior session (offset-resume,
    /// #606). See [`hyprstream_rpc::moq_event::MoqEventSubscriber::with_resume_from`]
    /// for the multi-source caveat. Must be called before the first `recv()`.
    pub fn with_resume_from(&mut self, sequence: u64) -> Result<()> {
        self.inner.with_resume_from(sequence)
    }

    /// Highest live-group sequence delivered so far (resume hint, #606).
    pub fn last_sequence(&self) -> u64 {
        self.inner.last_sequence()
    }

    /// Count of items evicted under drop-oldest backpressure (#606).
    pub fn dropped_count(&self) -> u64 {
        self.inner.dropped_count()
    }

    /// Receive the next event asynchronously.
    ///
    /// Returns `(topic, payload)`. Blocks until an event arrives.
    pub async fn recv(&mut self) -> Result<(String, Vec<u8>)> {
        self.inner.recv().await
    }

    /// Receive with timeout.
    ///
    /// Returns `Ok(Some((topic, payload)))` if event received within timeout,
    /// `Ok(None)` if timeout elapsed, or `Err` on error.
    pub async fn recv_timeout(&mut self, timeout: Duration) -> Result<Option<(String, Vec<u8>)>> {
        self.inner.recv_timeout(timeout).await
    }

    /// Try to receive without blocking.
    pub fn try_recv(&mut self) -> Result<Option<(String, Vec<u8>)>> {
        self.inner.try_recv()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_subscription_patterns() {
        let patterns = vec![
            "",                          // All events
            "worker.",                   // All worker events
            "worker.sandbox123.",        // Specific sandbox
            "worker.sandbox123.started", // Exact match
        ];

        for pattern in patterns {
            assert!(pattern.is_ascii(), "Pattern should be ASCII: {pattern}");
        }
    }
}
