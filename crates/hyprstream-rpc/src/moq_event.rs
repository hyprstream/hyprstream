//! moq-lite event bus plane — replaces the ZMQ XPUB/XSUB ProxyService (#167).
//!
//! # Architecture
//!
//! The event bus is a process-global [`MoqEventOrigin`] rooted at `local/events`.
//! Each service registers its own broadcast under `local/events/{source}` with a
//! single `events` track. Each published frame carries the full dot-separated topic
//! (`{source}.{entity}.{event}`) and the raw payload, packed as:
//!
//! ```text
//!   [4 bytes topic_len BE][topic UTF-8][payload bytes]
//! ```
//!
//! Subscribers connect to the origin, discover source broadcasts via
//! `OriginConsumer::announced()`, subscribe to their `events` track, and do
//! in-memory topic-prefix filtering (matching ZMQ's prefix-filter semantics).
//!
//! This is the "Live" preset for the event bus: fan-out, unbounded, at-most-once.
//! No chained HMAC or policy axes — events are best-effort lifecycle signals, not
//! auditable streams. `SecureEventPublisher` / `SecureEventSubscriber` (Phase 7)
//! layer group-key encryption on top and are unaffected by this transport change.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use moq_net::{BroadcastProducer, Group, Origin, OriginConsumer, OriginProducer, Path, Track, TrackProducer};
use parking_lot::Mutex;
use tokio::sync::mpsc;

// ============================================================================
// Process-global moq event bus origin — set once at startup by the factory.
// ============================================================================

static GLOBAL_MOQ_EVENT_ORIGIN: OnceLock<MoqEventOrigin> = OnceLock::new();

/// Register the process-global moq event bus origin.
///
/// Must be called once at startup (from the event-service factory) before any
/// `EventPublisher::new()` or `EventSubscriber::new()` call.
/// Returns `true` on the first call, `false` if already initialized (idempotent).
pub fn init_global_moq_event_origin(origin: MoqEventOrigin) -> bool {
    GLOBAL_MOQ_EVENT_ORIGIN.set(origin).is_ok()
}

/// Borrow the process-global moq event bus origin, if initialized.
///
/// Returns `None` when the event bus has not yet been wired (unit tests, or
/// before `create_event_service` factory has run).
pub fn global_moq_event_origin() -> Option<&'static MoqEventOrigin> {
    GLOBAL_MOQ_EVENT_ORIGIN.get()
}

/// Track name for the event fanout track within each source broadcast.
pub const EVENT_TRACK: &str = "events";

/// Broadcast path prefix (the `local/events` root under which per-source
/// broadcasts are registered).
pub const EVENT_PREFIX: &str = "local/events";

// ============================================================================
// MoqEventOrigin
// ============================================================================

/// Shared moq origin for the event bus plane.
///
/// Replaces the ZMQ XPUB/XSUB `ProxyService`. Publishers register broadcasts
/// under `local/events/{source}`; subscribers watch the origin consumer for
/// announcements and subscribe to the `events` track of each source broadcast.
#[derive(Clone)]
pub struct MoqEventOrigin {
    inner: Arc<EventOriginInner>,
}

struct EventOriginInner {
    /// Root-scoped producer: publishes under `local/events/{source}`.
    producer: OriginProducer,
    /// Consumer over the same origin tree (for subscribers).
    consumer: OriginConsumer,
    /// Keep `BroadcastProducer`s alive so their broadcasts stay announced.
    /// Keyed by source name so re-registration replaces the stale producer.
    broadcasts: Mutex<HashMap<String, BroadcastProducer>>,
}

impl MoqEventOrigin {
    /// Create a standalone event origin (used at startup / in unit tests).
    pub fn new() -> Self {
        let producer = Origin::random().produce();
        let consumer = producer.consume();
        Self {
            inner: Arc::new(EventOriginInner {
                producer,
                consumer,
                broadcasts: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Create from an existing producer/consumer pair (e.g. shared with the
    /// iroh moq substrate so external relay subscribers see the same tree).
    pub fn from_pair(producer: OriginProducer, consumer: OriginConsumer) -> Self {
        Self {
            inner: Arc::new(EventOriginInner {
                producer,
                consumer,
                broadcasts: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Create an event publisher for `source` (e.g. `"worker"`, `"system"`).
    ///
    /// Registers the broadcast `local/events/{source}` and opens its `events` track.
    pub fn publisher(&self, source: &str) -> Result<MoqEventPublisher> {
        let path = format!("{}/{}", EVENT_PREFIX, source);
        let mut broadcast = self
            .inner
            .producer
            .create_broadcast(path.as_str())
            .ok_or_else(|| anyhow!("create_broadcast denied for {path}"))?;

        let track = broadcast.create_track(Track::new(EVENT_TRACK))?;

        // Retain the broadcast producer so it stays announced for the publisher's lifetime.
        // HashMap keyed by source: re-registration replaces the stale producer atomically.
        self.inner.broadcasts.lock().insert(source.to_owned(), broadcast);

        Ok(MoqEventPublisher {
            track,
            source: source.to_owned(),
            next_group: 0,
        })
    }

    /// Clone the origin consumer (for subscriber background tasks).
    pub fn consumer(&self) -> OriginConsumer {
        self.inner.consumer.clone()
    }
}

impl Default for MoqEventOrigin {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MoqEventPublisher
// ============================================================================

/// In-process moq event publisher.
///
/// Publishes topic+payload pairs to the `local/events/{source}` broadcast.
/// Each `publish_raw` call writes one moq Group (one Frame) on the `events` track.
pub struct MoqEventPublisher {
    track: TrackProducer,
    source: String,
    next_group: u64,
}

impl MoqEventPublisher {
    /// Publish a raw topic + payload.
    ///
    /// Frame format: `[4 bytes topic_len BE][topic bytes][payload bytes]`.
    pub fn publish_raw(&mut self, topic: &str, payload: &[u8]) -> Result<()> {
        let topic_bytes = topic.as_bytes();
        let topic_len = topic_bytes.len() as u32;

        let mut frame = Vec::with_capacity(4 + topic_bytes.len() + payload.len());
        frame.extend_from_slice(&topic_len.to_be_bytes());
        frame.extend_from_slice(topic_bytes);
        frame.extend_from_slice(payload);

        let group_id = self.next_group;
        self.next_group += 1;

        let mut group = self.track.create_group(Group::from(group_id))?;
        group.write_frame(Bytes::from(frame))?;
        group.finish()?;
        Ok(())
    }

    /// Publish `{source}.{entity}.{event}` with payload.
    pub fn publish(&mut self, entity: &str, event: &str, payload: &[u8]) -> Result<()> {
        if entity.contains('.') {
            return Err(anyhow!("Entity name cannot contain '.': {}", entity));
        }
        if event.contains('.') {
            return Err(anyhow!("Event name cannot contain '.': {}", event));
        }
        let topic = format!("{}.{}.{}", self.source, entity, event);
        self.publish_raw(&topic, payload)
    }

    /// Source name for this publisher.
    pub fn source(&self) -> &str {
        &self.source
    }
}

// ============================================================================
// MoqEventSubscriber
// ============================================================================

/// In-process moq event subscriber.
///
/// Subscribes to one or more topic-prefix patterns. Patterns use the same
/// dot-separated prefix semantics as ZMQ (`"worker."` matches all worker events,
/// `""` matches everything). Backed by a background Tokio task that watches the
/// origin consumer for new source broadcasts and reads their `events` tracks.
pub struct MoqEventSubscriber {
    /// Patterns added via `subscribe()`. Finalized before `recv()` is called.
    patterns: Vec<String>,
    /// Receiving end of the background task's channel. `None` until first `recv()`.
    rx: Option<mpsc::Receiver<(String, Vec<u8>)>>,
    /// Background task handle (kept alive for the subscriber's lifetime).
    _task: Option<tokio::task::JoinHandle<()>>,
}

impl MoqEventSubscriber {
    /// Create a new subscriber. No background task is started yet.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            rx: None,
            _task: None,
        }
    }

    /// Add a topic-prefix filter (prefix match, ZMQ semantics).
    ///
    /// - `"worker."` — all events from source `worker`
    /// - `"worker.sandbox123.started"` — exact topic
    /// - `""` — all events (subscribe-all)
    ///
    /// Must be called before the first `recv()`.
    pub fn subscribe(&mut self, pattern: &str) -> Result<()> {
        if self.rx.is_some() {
            return Err(anyhow!("subscribe() must be called before recv()"));
        }
        self.patterns.push(pattern.to_owned());
        Ok(())
    }

    /// Subscribe to all events (`pattern = ""`).
    pub fn subscribe_all(&mut self) -> Result<()> {
        self.subscribe("")
    }

    /// Unsubscribe from a pattern. Must be called before `recv()`.
    pub fn unsubscribe(&mut self, pattern: &str) -> Result<()> {
        if self.rx.is_some() {
            return Err(anyhow!("unsubscribe() must be called before recv()"));
        }
        self.patterns.retain(|p| p != pattern);
        Ok(())
    }

    /// Lazily start the background task and return a mutable reference to the channel receiver.
    fn ensure_started(&mut self) -> Result<&mut mpsc::Receiver<(String, Vec<u8>)>> {
        if self.rx.is_none() {
            let origin = global_moq_event_origin()
                .ok_or_else(|| anyhow!("moq event bus not initialized; call init_global_moq_event_origin first"))?;

            let (tx, rx) = mpsc::channel(256);
            let consumer = origin.consumer();
            let patterns = self.patterns.clone();

            let task = tokio::spawn(run_subscriber_task(consumer, patterns, tx));
            self.rx = Some(rx);
            self._task = Some(task);
        }
        self.rx.as_mut().ok_or_else(|| anyhow!("subscriber channel not initialized"))
    }

    /// Receive the next event. Blocks until an event arrives or the bus is closed.
    pub async fn recv(&mut self) -> Result<(String, Vec<u8>)> {
        let rx = self.ensure_started()?;
        rx.recv().await.ok_or_else(|| anyhow!("event subscriber closed"))
    }

    /// Receive with timeout.
    pub async fn recv_timeout(&mut self, timeout: Duration) -> Result<Option<(String, Vec<u8>)>> {
        match tokio::time::timeout(timeout, self.recv()).await {
            Ok(result) => result.map(Some),
            Err(_) => Ok(None),
        }
    }

    /// Try to receive without blocking.
    pub fn try_recv(&mut self) -> Result<Option<(String, Vec<u8>)>> {
        let rx = self.ensure_started()?;
        match rx.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                Err(anyhow!("event subscriber closed"))
            }
        }
    }
}

impl Default for MoqEventSubscriber {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Background subscriber task
// ============================================================================

/// Background task: watches the origin consumer for announced source broadcasts,
/// subscribes to each `events` track, and relays matching topic+payload pairs to `tx`.
async fn run_subscriber_task(
    mut consumer: OriginConsumer,
    patterns: Vec<String>,
    tx: mpsc::Sender<(String, Vec<u8>)>,
) {
    use tokio::task::JoinSet;

    let patterns = Arc::new(patterns);

    // Scope the consumer to the `local/events` subtree, then further narrow by
    // source names extracted from the patterns.
    match scope_consumer_for_patterns(&consumer, &patterns) {
        Some(c) => consumer = c,
        None => {
            // Origin has no `local/events` tree yet — using the root consumer would
            // expose all RPC traffic to the event channel. Abort instead.
            tracing::warn!("event subscriber: origin has no event scope; subscriber task exiting");
            return;
        }
    }

    // Track per-broadcast read tasks so they are cancelled when this loop exits.
    let mut tasks: JoinSet<()> = JoinSet::new();

    loop {
        let (path, broadcast) = match consumer.announced().await {
            None => break,               // origin closed
            Some((_, None)) => continue, // unannounce
            Some((path, Some(b))) => (path, b),
        };

        let tx2 = tx.clone();
        let patterns2 = patterns.clone();
        let path_str = path.as_str().to_owned();

        tasks.spawn(async move {
            read_event_broadcast(broadcast, path_str, patterns2, tx2).await;
        });
    }
    // Dropping `tasks` here cancels all per-broadcast read tasks.
}

/// Scope the origin consumer to `local/events/{sources}` based on patterns.
///
/// If any pattern is `""` (subscribe-all), returns `None` and the caller uses
/// an unscoped consumer rooted at `local/events`.
fn scope_consumer_for_patterns(
    consumer: &OriginConsumer,
    patterns: &[String],
) -> Option<OriginConsumer> {
    // Scope to local/events first
    let events_consumer = consumer.scope(&[Path::new(EVENT_PREFIX)])?;

    // If subscribe-all pattern present, return events-scoped consumer
    if patterns.iter().any(String::is_empty) {
        return Some(events_consumer);
    }

    // Extract unique source names from patterns (e.g., "worker." → "worker")
    let sources: Vec<String> = patterns
        .iter()
        .filter_map(|p| {
            let src = p.split('.').next()?;
            if src.is_empty() {
                None
            } else {
                Some(src.to_owned())
            }
        })
        .collect::<HashSet<String>>()
        .into_iter()
        .collect();

    if sources.is_empty() {
        return Some(events_consumer);
    }

    let path_refs: Vec<Path<'_>> = sources.iter().map(|s| Path::new(s.as_str())).collect();
    events_consumer.scope(&path_refs)
}

/// Read all groups from a single source broadcast's `events` track and relay
/// matching events to `tx`.
async fn read_event_broadcast(
    broadcast: moq_net::BroadcastConsumer,
    path: String,
    patterns: Arc<Vec<String>>,
    tx: mpsc::Sender<(String, Vec<u8>)>,
) {
    let mut track = match broadcast.subscribe_track(&Track::new(EVENT_TRACK)) {
        Ok(t) => t,
        Err(_) => return,
    };

    loop {
        let mut group = match tokio::time::timeout(
            crate::moq_stream::GROUP_IDLE_TIMEOUT,
            track.next_group(),
        ).await {
            Ok(Ok(Some(g))) => g,
            Ok(Ok(None)) | Ok(Err(_)) => break,
            Err(_elapsed) => {
                tracing::debug!("event subscriber idle timeout — source broadcast may be gone");
                break;
            }
        };

        let frame = match group.read_frame().await {
            Ok(Some(f)) => f,
            Ok(None) => break,   // group ended without a frame
            Err(e) => {
                tracing::warn!(error = %e, "event broadcast: frame read error");
                break;
            }
        };

        // Decode: [4 bytes topic_len BE][topic][payload]
        if frame.len() < 4 {
            continue;
        }
        let topic_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        if frame.len() < 4 + topic_len {
            continue;
        }
        let topic = match std::str::from_utf8(&frame[4..4 + topic_len]) {
            Ok(t) => t.to_owned(),
            Err(_) => continue,
        };
        let payload = frame[4 + topic_len..].to_vec();

        // Apply pattern filter (prefix match, ZMQ semantics)
        if topic_matches_patterns(&topic, &patterns)
            && tx.send((topic, payload)).await.is_err()
        {
            break; // receiver dropped
        }
    }

    let _ = path; // kept for debugging context; unused in the hot path
}

/// True if `topic` matches any pattern in `patterns` (prefix match).
fn topic_matches_patterns(topic: &str, patterns: &[String]) -> bool {
    for pat in patterns {
        if pat.is_empty() {
            return true; // subscribe-all
        }
        if pat.ends_with('.') {
            if topic.starts_with(pat.as_str()) {
                return true;
            }
        } else if topic == pat.as_str() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_matches_patterns() {
        let patterns = vec!["worker.".to_owned()];
        assert!(topic_matches_patterns("worker.sandbox123.started", &patterns));
        assert!(!topic_matches_patterns("system.event.ready", &patterns));
        assert!(!topic_matches_patterns("worker", &patterns)); // no trailing dot match

        let patterns_all = vec!["".to_owned()];
        assert!(topic_matches_patterns("any.thing", &patterns_all));

        let patterns_exact = vec!["worker.abc.started".to_owned()];
        assert!(topic_matches_patterns("worker.abc.started", &patterns_exact));
        assert!(!topic_matches_patterns("worker.abc.stopped", &patterns_exact));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moq_event_pub_sub_roundtrip() -> Result<()> {
        let origin = MoqEventOrigin::new();
        init_global_moq_event_origin(origin.clone());

        // Publisher
        let mut pub_ = origin.publisher("worker")?;

        // Subscriber
        let mut sub = MoqEventSubscriber::new();
        sub.subscribe("worker.")?;

        // Start subscriber background task (lazy; triggered by recv)
        // Publish before recv so the broadcast is announced when the task starts.
        pub_.publish("sandbox123", "started", b"payload")?;

        // Give the origin a moment to propagate the announcement.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Manually start the task by calling ensure_started and check recv.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            sub.recv(),
        ).await;

        match result {
            Ok(Ok((topic, payload))) => {
                assert_eq!(topic, "worker.sandbox123.started");
                assert_eq!(payload, b"payload");
            }
            Ok(Err(e)) => panic!("recv error: {e}"),
            Err(_) => panic!("recv timeout"),
        }

        Ok(())
    }
}
