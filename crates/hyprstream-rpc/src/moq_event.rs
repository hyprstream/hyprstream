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

    /// Clone the origin producer.
    ///
    /// Needed to wire this origin into a `moq_net::Server`/`Client` with
    /// `with_origin` / `with_consume` so externally-published broadcasts ingest
    /// into the same tree subscribers read from.
    pub fn producer(&self) -> OriginProducer {
        self.inner.producer.clone()
    }
}

impl Default for MoqEventOrigin {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Cross-process event plane over UDS (#275)
//
// In the same-process (`InprocManager`) deployment every service shares one
// process-global `MoqEventOrigin`, so an in-process publisher and subscriber
// see the same tree. In the systemd / `--ipc` deployment each service runs in
// its OWN process; only the `event` service's process initializes the global
// origin via the factory. Other processes (notably `worker`) had no origin at
// all — `EventPublisher::new` failed with "moq event bus not initialized".
//
// The fix mirrors the streaming plane (`moq_stream::serve_moq_uds_background`):
//
//   * The `event` service SERVES its origin over a well-known UDS path
//     ([`serve_event_moq_uds_background`]) using `moq_net::Server::with_origin`
//     so the session is BIDIRECTIONAL — it both serves announced broadcasts to
//     subscribers AND ingests broadcasts a connected client publishes.
//
//   * Every NON-event process creates its own local `MoqEventOrigin`, registers
//     it as the process global, then spawns a background task
//     ([`connect_event_moq_uds_background`]) that connects a `moq_net::Client`
//     to the event service's UDS with `with_origin(local_producer)`. That makes
//     the session bidirectional too: locally-published event broadcasts are
//     announced UP to the event service (which fans them out to all
//     subscribers), and broadcasts from other processes flow DOWN into the
//     local origin so local subscribers see them.
//
// The UDS path is the stable, cross-process [`crate::paths::event_socket`] —
// not a per-PID path — so any process can find it without an RPC round-trip.
// ============================================================================

/// Reconnect backoff for the client-side event plane link.
const EVENT_RECONNECT_DELAY: Duration = Duration::from_millis(500);

/// Serve the event-bus origin over a UDS socket so cross-process publishers and
/// subscribers reach the same tree (#275).
///
/// Mirrors [`crate::moq_stream::serve_moq_uds_background`], but the moq server is
/// built with `with_origin` (publish + consume) rather than `with_publish`
/// alone, because the event plane is bidirectional: a connected client both
/// announces its own event broadcasts (ingested into this origin) and subscribes
/// to broadcasts from every other process.
///
/// Called once by the event-service factory. Idempotent: a second call is a
/// no-op (the first bind wins). Connection errors are logged, not fatal.
pub fn serve_event_moq_uds_background(origin: MoqEventOrigin, path: std::path::PathBuf) {
    use crate::transport::uds_session::{accept_uds, PLANE_MOQ};
    use moq_net::Server as MoqServer;

    // Ensure the parent directory exists (runtime_dir may not be created yet).
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Remove a stale socket from a previous run (best-effort).
    let _ = std::fs::remove_file(&path);

    let listener = match std::os::unix::net::UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(path = %path.display(), "event moq UDS bind failed: {e}");
            return;
        }
    };

    // 0o600: owner read/write only; SO_PEERCRED enforces uid match (uds_server.rs).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }

    // tokio requires the listener be non-blocking before adoption.
    if let Err(e) = listener.set_nonblocking(true) {
        tracing::error!("event moq UDS set_nonblocking failed: {e}");
        return;
    }

    let listener = match tokio::net::UnixListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("event moq UDS listener conversion failed: {e}");
            return;
        }
    };

    tracing::info!(path = %path.display(), "event moq UDS listener ready");

    tokio::spawn(async move {
        loop {
            let stream = match listener.accept().await {
                Ok((s, _)) => s,
                Err(e) => {
                    tracing::warn!("event moq UDS accept error: {e}");
                    continue;
                }
            };
            // Each session gets its own producer/consumer view of the SAME origin
            // tree, so a client's published broadcasts are ingested and re-served
            // to every other connected session.
            let producer = origin.producer();
            tokio::spawn(async move {
                let (plane, session) = match accept_uds(stream).await {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::debug!("event moq UDS handshake error: {e}");
                        return;
                    }
                };
                if plane != PLANE_MOQ {
                    tracing::debug!("event moq UDS: unexpected plane 0x{plane:02x} — dropping");
                    return;
                }
                // `accept` returns once the handshake completes; the protocol then
                // runs in the background. The returned `Session` MUST be held —
                // dropping it closes the connection. Hold it until it closes so
                // the bidirectional bridge stays live (publish + consume).
                match MoqServer::new().with_origin(producer).accept(session).await {
                    Ok(moq_session) => {
                        let reason = moq_session.closed().await;
                        tracing::debug!("event moq UDS session closed: {reason:?}");
                    }
                    Err(e) => tracing::debug!("event moq UDS accept error: {e}"),
                }
            });
        }
    });
}

/// Ensure this process can publish/subscribe events, wiring a CLIENT-mode
/// event origin connected to the event service's UDS plane if needed (#275).
///
/// Idempotent and safe to call from any process at startup:
///
///   * If the process-global origin is ALREADY initialized (this process hosts
///     the `event` service, which calls [`init_global_moq_event_origin`] +
///     [`serve_event_moq_uds_background`] in its factory), this is a no-op —
///     the in-process origin is authoritative.
///
///   * Otherwise it creates a fresh local origin, registers it as the global,
///     and spawns the background link to the event service's UDS
///     ([`connect_event_moq_uds_background`]). After this returns,
///     `EventPublisher::new` / `EventSubscriber` succeed in this process and
///     events flow cross-process to the shared bus.
///
/// Must be called from within a tokio runtime (it spawns a background task).
/// `path` is the event service's UDS socket (use [`crate::paths::event_socket`]).
pub fn ensure_event_client_origin(path: std::path::PathBuf) {
    if global_moq_event_origin().is_some() {
        // The event service is co-located in this process; nothing to dial.
        return;
    }
    let origin = MoqEventOrigin::new();
    // Register first so any `EventPublisher::new` after this point resolves the
    // global immediately, even before the background link is established (moq
    // buffers/announces the broadcast and it flows up once the link connects).
    if !init_global_moq_event_origin(origin.clone()) {
        // Lost a race with the event factory or a concurrent caller; the global
        // that won is authoritative, so drop ours.
        return;
    }
    connect_event_moq_uds_background(origin, path);
}

/// Connect a local event origin to the event service's UDS plane (#275).
///
/// Spawns a background task that connects a `moq_net::Client` (built with
/// `with_origin(local_producer)`) to `path`, keeping the link alive and
/// reconnecting on failure. The session is bidirectional: locally-published
/// event broadcasts are announced UP to the event service, and broadcasts from
/// other processes flow DOWN into `origin` so local subscribers see them.
///
/// Called once, in non-event processes, AFTER registering `origin` as the
/// process global. The task runs for the process lifetime.
pub fn connect_event_moq_uds_background(origin: MoqEventOrigin, path: std::path::PathBuf) {
    tokio::spawn(async move {
        loop {
            if let Err(e) = run_event_client_link(&origin, &path).await {
                tracing::debug!(
                    path = %path.display(),
                    "event moq client link ended: {e}; reconnecting in {:?}",
                    EVENT_RECONNECT_DELAY
                );
            }
            tokio::time::sleep(EVENT_RECONNECT_DELAY).await;
        }
    });
}

/// One connect-and-pump cycle of the client-side event link. Returns when the
/// session closes (so the caller can reconnect).
async fn run_event_client_link(origin: &MoqEventOrigin, path: &std::path::Path) -> Result<()> {
    use crate::transport::uds_session::{connect_uds, PLANE_MOQ};
    use moq_net::Client as MoqClient;

    let session = connect_uds(path, PLANE_MOQ)
        .await
        .map_err(|e| anyhow!("event moq UDS connect {}: {e}", path.display()))?;

    // `with_origin` makes this a bidirectional link: publish local broadcasts UP
    // and consume remote broadcasts DOWN into the SAME origin tree subscribers read.
    let moq_client = MoqClient::new().with_origin(origin.producer());
    let moq_session = moq_client
        .connect(session)
        .await
        .map_err(|e| anyhow!("event moq handshake: {e}"))?;

    tracing::info!(path = %path.display(), "event moq client link established");

    // Hold the session open until it closes; broadcasts flow in both directions
    // for the duration via the origin producer/consumer wired above.
    let reason = moq_session.closed().await;
    Err(anyhow!("event moq session closed: {reason:?}"))
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
#[allow(clippy::expect_used, clippy::unwrap_used)]
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

    /// #275 cross-process bug: a non-event process publishes events through a
    /// CLIENT-mode origin linked over UDS to a separate SERVER-mode origin
    /// (the event service). A subscriber on the SERVER origin must see the
    /// event — proving the bus is no longer a process-local island.
    ///
    /// This mirrors the systemd / --ipc deployment: the "event service" origin
    /// is served via `serve_event_moq_uds_background`; the "worker" origin is a
    /// distinct origin (as in a separate process) wired via
    /// `connect_event_moq_uds_background`. The publisher resolves against the
    /// client origin (no in-process event factory ran for it), and the event
    /// reaches a subscriber reading the server origin's consumer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cross_process_event_over_uds_client_link() -> Result<()> {
        use std::time::Duration;

        // Unique, stable socket path for this test (cross-"process" rendezvous).
        let sock = std::env::temp_dir()
            .join(format!("hyprstream-event-test-{}.sock", std::process::id()));

        // ── "event service" process: serve a SERVER-mode origin over UDS ──────
        let server_origin = MoqEventOrigin::new();
        serve_event_moq_uds_background(server_origin.clone(), sock.clone());
        // Let the listener bind.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // ── "worker" process: a DISTINCT origin linked to the event UDS ───────
        let client_origin = MoqEventOrigin::new();
        connect_event_moq_uds_background(client_origin.clone(), sock.clone());

        // A publisher built on the client origin (the failing path before #275).
        let mut publisher = client_origin.publisher("worker")?;

        // Drive the link + announcement by publishing repeatedly; read the
        // subscriber on the SERVER origin's consumer (a "different process").
        // Read the server consumer directly (the same data a server-side
        // EventSubscriber would receive) to avoid depending on the global.
        let consumer = server_origin.consumer();
        let scoped = consumer
            .scope(&[Path::new(EVENT_PREFIX)])
            .ok_or_else(|| anyhow!("server origin has no event scope yet"))?;

        let recv = tokio::spawn(async move {
            let mut scoped = scoped;
            loop {
                let (_path, bc) = match scoped.announced().await {
                    None => return None,
                    Some((_, None)) => continue,
                    Some((path, Some(b))) => (path, b),
                };
                let mut track = match bc.subscribe_track(&Track::new(EVENT_TRACK)) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let mut group = match track.next_group().await {
                    Ok(Some(g)) => g,
                    _ => continue,
                };
                if let Ok(Some(frame)) = group.read_frame().await {
                    return Some(frame.to_vec());
                }
            }
        });

        // Re-publish periodically (the moq broadcast is retained by the origin,
        // but re-asserting it ensures the announcement is live once the
        // client→server link establishes). Concurrently await the receiver.
        let publish_loop = async {
            for _ in 0..50 {
                let _ = publisher.publish("sandbox123", "started", b"payload");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            // Keep the publisher (and its broadcast) alive until the receiver
            // finishes by parking here.
            std::future::pending::<()>().await;
        };

        let got: Option<Vec<u8>> = tokio::select! {
            r = recv => r.ok().flatten(),
            _ = publish_loop => None,
            _ = tokio::time::sleep(Duration::from_secs(10)) => None,
        };

        let _ = std::fs::remove_file(&sock);
        let frame = got.expect("event did not cross the UDS link to the server origin");

        // Decode [4 bytes topic_len BE][topic][payload].
        assert!(frame.len() >= 4, "frame too short");
        let topic_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let topic = std::str::from_utf8(&frame[4..4 + topic_len]).unwrap();
        let payload = &frame[4 + topic_len..];
        assert_eq!(topic, "worker.sandbox123.started");
        assert_eq!(payload, b"payload");

        Ok(())
    }
}
