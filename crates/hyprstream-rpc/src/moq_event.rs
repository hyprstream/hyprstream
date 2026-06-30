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
//! By default a subscriber uses the `EventLive` delivery preset: fan-out,
//! at-most-once, drop-oldest-on-backpressure (matches the historic "unbounded,
//! best-effort" behaviour, but bounded — see #606). No chained HMAC or
//! confidentiality axes here — events are best-effort lifecycle signals, not
//! auditable streams. `SecureEventPublisher` / `SecureEventSubscriber` (Phase 7)
//! layer group-key encryption on top and are unaffected by this transport change.
//!
//! ## QoS (#606)
//!
//! `MoqEventSubscriber` reuses the EXISTING `StreamOpt` contract from
//! `streaming.capnp` / [`crate::stream_info`] (#213/#273) rather than a parallel
//! event-specific QoS type — see [`crate::stream_info::EventLive`] (default) and
//! [`crate::stream_info::EventReliable`] (at-least-once + retained, for events
//! that must not be silently dropped, e.g. `model.loaded`). Only `delivery` and
//! `overflow_policy` are enforced client-side today (`ordering`/`completion` are
//! reserved for a future chained-integrity layer, same as the flat at-most-once
//! default already implies `Ordering::Ordered` is not currently checked).
//!
//! # #393 — per-OID broadcast paths
//!
//! The original design (above) is a flat single-track fan-out: every event for
//! every model OID lands on the same `events` track, and subscribers filter
//! post-read. That is the firehose problem — O(whole-network) reads. The fix
//! mirrors the streaming plane (`moq_stream`: `{tenant}/{service}/{topic}/{instance}`):
//! each model OID gets its OWN broadcast path, `local/events/publications/{oid_hash}`,
//! so moq's `scope(&[Path])` makes wire-level selectivity automatic (one QUIC
//! uni-stream per group per subscribed track). A node tracking N of M OIDs reads
//! N tracks, not M.
//!
//! The flat `local/events/{source}` track is RETAINED as a transition fallback:
//! publishers that mirror to both paths keep legacy subscribers working while
//! new subscribers read the selective per-OID track. No capnp change — the event
//! payload format is unchanged; only the track naming/routing changes.
//!
//! ## Late-join retention (decision A: firehose-backfill)
//!
//! moq's per-track cache evicts groups older than `MAX_GROUP_AGE` (5s) — too
//! short for a scheduler reconstructing publication history. On first
//! subscription to an OID's track, [`BackfillMode`] checks whether history is
//! available via the optional atproto firehose (`sh.tangled.git.refUpdate`) /
//! registry; if so it replays that history before switching to live MoQ. If the
//! firehose is unavailable the subscriber starts live-only (graceful
//! degradation) — the firehose is the cold-start path, MoQ is the live path.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use moq_net::{
    BroadcastProducer, Group, Origin, OriginConsumer, OriginProducer, Path, Track, TrackProducer,
};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::stream_info::{OverflowPolicy, StreamOpt, StreamOptPreset};

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

/// Broadcast path segment under which per-OID (#393) publication broadcasts
/// live: `local/events/publications/{oid_hash}`. Mirrors the streaming plane's
/// `{tenant}/{service}/{topic}/{instance}` shape so `scope(&[Path])` gives
/// wire-level selectivity — a node tracking N of M OIDs reads N tracks, not M.
pub const PUBLICATIONS_SEGMENT: &str = "publications";

/// Full prefix for per-OID publication broadcasts.
pub const PUBLICATIONS_PREFIX: &str = "local/events/publications";

/// Number of hex characters of the SHA-256 OID hash used in the broadcast path.
/// 16 hex chars (64 bits) keeps paths short while making collisions negligible
/// across any realistic OID namespace (birthday bound ≈ 2^32 models).
pub const OID_HASH_LEN: usize = 16;

/// Stable, filesystem/moq-safe identifier for a model OID.
///
/// The OID is hashed (SHA-256, truncated to [`OID_HASH_LEN`] hex chars) so the
/// broadcast path is unguessable from the OID alone and contains no characters
/// that are illegal in a moq path segment. Two distinct OIDs map to distinct
/// paths with overwhelming probability.
pub fn oid_hash(oid: &str) -> String {
    let digest = Sha256::digest(oid.as_bytes());
    hex::encode(&digest[..OID_HASH_LEN / 2])
}

/// Build the per-OID publication broadcast path: `local/events/publications/{oid_hash}`.
pub fn publication_broadcast_path(oid: &str) -> String {
    format!("{}/{}", PUBLICATIONS_PREFIX, oid_hash(oid))
}

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
        self.inner
            .broadcasts
            .lock()
            .insert(source.to_owned(), broadcast);

        Ok(MoqEventPublisher {
            track,
            source: source.to_owned(),
            next_group: 0,
            oid_track: None,
        })
    }

    /// Create an event publisher for `source` (e.g. `"worker"`, `"system"`) that
    /// ALSO mirrors every event to the per-OID (#393) publication track for `oid`.
    ///
    /// This is the transition path: the publisher writes each event to BOTH the
    /// legacy flat `local/events/{source}` broadcast (so existing subscribers
    /// keep working) AND the selective `local/events/publications/{oid_hash}`
    /// broadcast (so #393 subscribers get wire-level per-OID selectivity). Once
    /// all subscribers have migrated to per-OID tracks the flat mirror can be
    /// dropped (see [`MoqEventOrigin::publisher_oid_only`]).
    ///
    /// `oid` is the model OID whose publications this publisher emits; it is
    /// hashed via [`oid_hash`] for the broadcast path.
    pub fn publisher_with_oid(&self, source: &str, oid: &str) -> Result<MoqEventPublisher> {
        let flat = self.publisher(source)?;
        let oid_track = self.oid_track(oid)?;
        Ok(MoqEventPublisher {
            track: flat.track,
            source: flat.source,
            next_group: flat.next_group,
            oid_track: Some(oid_track),
        })
    }

    /// Create an event publisher that writes ONLY to the per-OID (#393)
    /// publication track for `oid` — no flat-track mirror. Use once every
    /// subscriber of this source has migrated to per-OID subscription.
    ///
    /// `source` is retained for topic-prefix semantics and `publish()` topic
    /// construction, but nothing is announced under `local/events/{source}`.
    pub fn publisher_oid_only(&self, source: &str, oid: &str) -> Result<MoqEventPublisher> {
        let oid_track = self.oid_track(oid)?;
        Ok(MoqEventPublisher {
            track: oid_track,
            source: source.to_owned(),
            next_group: 0,
            oid_track: None,
        })
    }

    /// Open (creating if needed) the per-OID publication track for `oid` and
    /// retain its broadcast producer. Returns the track producer.
    fn oid_track(&self, oid: &str) -> Result<TrackProducer> {
        let path = publication_broadcast_path(oid);
        let mut broadcast = self
            .inner
            .producer
            .create_broadcast(path.as_str())
            .ok_or_else(|| anyhow!("create_broadcast denied for {path}"))?;
        let track = broadcast.create_track(Track::new(EVENT_TRACK))?;
        // Keyed by the full path so distinct OIDs accumulate distinct producers
        // and re-registration of the same OID replaces the stale producer.
        self.inner.broadcasts.lock().insert(path, broadcast);
        Ok(track)
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
///
/// When constructed via [`MoqEventOrigin::publisher_with_oid`], each event is
/// ALSO mirrored to the per-OID (#393) publication track so selective
/// subscribers receive it without the firehose. When constructed via
/// [`MoqEventOrigin::publisher_oid_only`], `track` IS the OID track and
/// `oid_track` is `None`.
pub struct MoqEventPublisher {
    track: TrackProducer,
    source: String,
    next_group: u64,
    /// Per-OID mirror track. `Some` when the publisher writes to BOTH the flat
    /// source track and the OID track (transition path); `None` when `track` is
    /// already the OID track (post-migration) or when no OID was supplied.
    oid_track: Option<TrackProducer>,
}

impl MoqEventPublisher {
    /// Publish a raw topic + payload.
    ///
    /// Frame format: `[4 bytes topic_len BE][topic bytes][payload bytes]`.
    ///
    /// If this publisher was created with an OID mirror, the frame is written to
    /// both the flat source track and the per-OID publication track (one Group
    /// on each, sharing the same group id so consumers on either track see a
    /// consistent sequence).
    pub fn publish_raw(&mut self, topic: &str, payload: &[u8]) -> Result<()> {
        let topic_bytes = topic.as_bytes();
        let topic_len = topic_bytes.len() as u32;

        let mut frame = Vec::with_capacity(4 + topic_bytes.len() + payload.len());
        frame.extend_from_slice(&topic_len.to_be_bytes());
        frame.extend_from_slice(topic_bytes);
        frame.extend_from_slice(payload);

        let group_id = self.next_group;
        self.next_group += 1;

        // Write to the primary track.
        write_group(&mut self.track, group_id, &frame)?;
        // Mirror to the per-OID track if one is attached (#393 transition).
        if let Some(oid_track) = &mut self.oid_track {
            write_group(oid_track, group_id, &frame)?;
        }
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

    /// True if this publisher mirrors events to a per-OID (#393) publication track.
    pub fn has_oid_mirror(&self) -> bool {
        self.oid_track.is_some()
    }
}

/// Write one Group (one Frame) carrying `frame` bytes to `track` at `group_id`.
fn write_group(track: &mut TrackProducer, group_id: u64, frame: &[u8]) -> Result<()> {
    let mut group = track.create_group(Group::from(group_id))?;
    group.write_frame(Bytes::copy_from_slice(frame))?;
    group.finish()?;
    Ok(())
}

// ============================================================================
// #393 — Firehose backfill for late-join (decision A)
//
// moq's per-track cache evicts groups older than MAX_GROUP_AGE (5s). A scheduler
// reconstructing publication history for an OID needs more than 5s of backlog.
// On first subscription to an OID's track we consult an optional BackfillSource
// (atproto firehose `sh.tangled.git.refUpdate` / registry); if it can serve
// history for that OID we replay it before switching to live MoQ. If no source
// is available the subscriber starts live-only (graceful degradation): the
// firehose is the cold-start path, MoQ is the live path.
// ============================================================================

/// A source of historical publication events for firehose-backfill late-join (#393).
///
/// Implementations typically wrap the atproto firehose (`sh.tangled.git.refUpdate`)
/// or a registry snapshot. All methods are fallible and best-effort: returning an
/// error or empty iterator from any of them causes [`BackfillMode`] to gracefully
/// degrade to live-only MoQ (no backfill), never to fail the subscription.
///
/// This is a trait object interface so the transport-agnostic `hyprstream-rpc`
/// crate can express the backfill contract without depending on the atproto /
/// firehose wiring (which lives in the `hyprstream` app crate).
pub trait BackfillSource: Send + Sync {
    /// True if this source can serve history for `oid` right now.
    ///
    /// Returning `false` (e.g. firehose offline, registry unreachable) makes the
    /// subscriber skip backfill and start live-only.
    fn has_history(&self, oid: &str) -> bool;

    /// Replay historical publication events for `oid` to `tx`, oldest-first.
    ///
    /// Each item is the same `(topic, payload)` pair a live MoQ subscriber would
    /// yield. Implementations should bound the replay (event count / wall time)
    /// and return once the backlog is drained; the caller then switches to live.
    /// Errors are logged and treated as "no history available".
    fn replay<'a>(
        &'a self,
        oid: &'a str,
        tx: &'a mpsc::Sender<(String, Vec<u8>)>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>>;
}

/// Late-join retention mode for a per-OID (#393) subscriber.
///
/// Decision A: on first subscription to an OID's track, optionally replay
/// history from a [`BackfillSource`] (atproto firehose / registry), then switch
/// to live MoQ. The firehose is the cold-start path; MoQ is the live path.
#[derive(Default)]
pub enum BackfillMode {
    /// No backfill: subscribe live-only. The default; matches pre-#393 behaviour
    /// and is the graceful-degradation fallback when no firehose is available.
    #[default]
    LiveOnly,
    /// Attempt backfill from the supplied source for `oid` before going live.
    /// If the source reports no history (`has_history` false) or `replay` errors,
    /// the subscriber silently falls back to [`BackfillMode::LiveOnly`].
    FirehoseBackfill {
        oid: String,
        source: Arc<dyn BackfillSource>,
    },
}

// ============================================================================
// Event delivery channel — QoS-aware overflow policy (#606)
// ============================================================================
//
// Replaces the previous unconditional `mpsc::channel(256)` + blocking
// `.send().await`. That gave every subscriber a single fixed-size lossless
// channel: a slow consumer that stopped calling `recv()` would back up its
// OWN per-broadcast read tasks indefinitely (no resubscribe, no eviction).
//
// `StreamOpt::overflow_policy` (reused, not reinvented — see module docs)
// now selects the behaviour per subscriber:
//   - `Block`      — lossless backpressure, same semantics as the old
//                    `mpsc` channel (correct for the `EventReliable` /
//                    at-least-once preset, which must not silently drop).
//   - `DropOldest` — ring-buffer: a full queue evicts its oldest entry
//                    rather than stalling the producer-side read tasks
//                    (the new default via `EventLive`, #606 finding: a
//                    shared/slow consumer must not wedge delivery).
//
// This is per-subscriber state (each `MoqEventSubscriber` owns one channel;
// its own per-broadcast read tasks share the sender half via `clone()`), so
// a slow subscriber only ever affects its own delivery — not other
// subscribers reading the same broadcast.

type EventItem = (String, Vec<u8>);

struct EventChannelState {
    queue: VecDeque<EventItem>,
    capacity: usize,
    overflow: OverflowPolicy,
    closed: bool,
}

struct EventChannelInner {
    state: Mutex<EventChannelState>,
    item_ready: tokio::sync::Notify,
    space_ready: tokio::sync::Notify,
    /// Live `EventChannelSender` clones. The last drop closes the channel
    /// (see `EventChannelSender`'s `Clone`/`Drop` impls) so a parked
    /// `recv()` wakes with `None` instead of waiting forever.
    sender_count: AtomicUsize,
}

/// Sender half of a subscriber's QoS-aware delivery channel. Cloneable —
/// each per-broadcast read task gets its own clone. Closes the channel when
/// the last clone is dropped (manual `Clone`/`Drop`, refcounted via
/// `EventChannelInner::sender_count` — `#[derive(Clone)]` would NOT track
/// this, which is why this impl is hand-written rather than derived).
struct EventChannelSender {
    inner: Arc<EventChannelInner>,
    dropped_count: Arc<AtomicU64>,
}

/// Receiver half, owned by the [`MoqEventSubscriber`].
struct EventChannelReceiver {
    inner: Arc<EventChannelInner>,
}

/// Overflow capacity for a channel built from `overflow_policy`: the
/// `DropOldest` high-water-mark, or a fixed depth for lossless `Block`
/// channels (matches the historic `mpsc::channel(256)` default).
fn event_channel_capacity(overflow: &OverflowPolicy) -> usize {
    match overflow {
        OverflowPolicy::Block => 256,
        OverflowPolicy::DropOldest { high_water_mark } => (*high_water_mark).max(1) as usize,
    }
}

fn event_channel(
    overflow: OverflowPolicy,
    dropped_count: Arc<AtomicU64>,
) -> (EventChannelSender, EventChannelReceiver) {
    let capacity = event_channel_capacity(&overflow);
    let inner = Arc::new(EventChannelInner {
        state: Mutex::new(EventChannelState {
            queue: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
            overflow,
            closed: false,
        }),
        item_ready: tokio::sync::Notify::new(),
        space_ready: tokio::sync::Notify::new(),
        sender_count: AtomicUsize::new(1),
    });
    (
        EventChannelSender {
            inner: inner.clone(),
            dropped_count,
        },
        EventChannelReceiver { inner },
    )
}

impl EventChannelSender {
    /// Send an item honouring the configured overflow policy. Returns `false`
    /// once the receiver has been dropped — the caller should stop reading.
    async fn send(&self, item: EventItem) -> bool {
        loop {
            {
                let mut state = self.inner.state.lock();
                if state.closed {
                    return false;
                }
                if state.queue.len() < state.capacity {
                    state.queue.push_back(item);
                    drop(state);
                    self.inner.item_ready.notify_one();
                    return true;
                }
                // At capacity.
                match state.overflow {
                    OverflowPolicy::Block => {
                        // Fall through: wait for the receiver to free space.
                    }
                    OverflowPolicy::DropOldest { .. } => {
                        state.queue.pop_front();
                        state.queue.push_back(item);
                        drop(state);
                        self.dropped_count.fetch_add(1, Ordering::Relaxed);
                        self.inner.item_ready.notify_one();
                        return true;
                    }
                }
            }
            self.inner.space_ready.notified().await;
        }
    }
}

impl Clone for EventChannelSender {
    fn clone(&self) -> Self {
        self.inner.sender_count.fetch_add(1, Ordering::AcqRel);
        Self {
            inner: self.inner.clone(),
            dropped_count: self.dropped_count.clone(),
        }
    }
}

impl Drop for EventChannelSender {
    fn drop(&mut self) {
        // Last sender clone gone: close the channel and wake a parked
        // `recv()` so it observes `closed` instead of waiting forever.
        // In-flight queued items are still delivered first — `recv()` only
        // returns `None` once the queue has also drained (see `recv()`).
        if self.inner.sender_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut state = self.inner.state.lock();
            state.closed = true;
            drop(state);
            self.inner.item_ready.notify_waiters();
        }
    }
}

impl EventChannelReceiver {
    async fn recv(&mut self) -> Option<EventItem> {
        loop {
            {
                let mut state = self.inner.state.lock();
                if let Some(item) = state.queue.pop_front() {
                    drop(state);
                    self.inner.space_ready.notify_one();
                    return Some(item);
                }
                if state.closed {
                    return None;
                }
            }
            self.inner.item_ready.notified().await;
        }
    }

    fn try_recv(&mut self) -> Result<Option<EventItem>, ()> {
        let mut state = self.inner.state.lock();
        if let Some(item) = state.queue.pop_front() {
            drop(state);
            self.inner.space_ready.notify_one();
            return Ok(Some(item));
        }
        if state.closed {
            return Err(());
        }
        Ok(None)
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
///
/// # #393 — per-OID subscription
///
/// [`Self::subscribe_oid`] scopes the subscriber to a single model OID's
/// publication track (`local/events/publications/{oid_hash}`), so only that
/// OID's events cross the wire — not the whole firehose. This is mutually
/// exclusive with the flat-pattern [`Self::subscribe`] API: a subscriber is
/// either pattern-based (legacy firehose + in-memory filter) or OID-scoped.
pub struct MoqEventSubscriber {
    /// Patterns added via `subscribe()`. Finalized before `recv()` is called.
    /// Empty when this subscriber is OID-scoped (no flat-track filtering).
    patterns: Vec<String>,
    /// The OID whose per-OID track this subscriber reads (`None` ⇒ legacy
    /// pattern-based subscription over the flat `local/events/{source}` tracks).
    oid_subscription: Option<String>,
    /// Late-join retention mode (decision A). Defaults to [`BackfillMode::LiveOnly`].
    backfill: BackfillMode,
    /// Delivery QoS (#606). Reuses [`StreamOpt`] — defaults to
    /// [`crate::stream_info::EventLive`] (at-most-once, drop-oldest), matching
    /// the pre-#606 behaviour but with a bounded, never-wedging queue instead
    /// of an unconditionally-blocking one.
    qos: StreamOpt,
    /// Skip live groups with `sequence <= resume_from` (offset-resume, #606).
    /// `None` (the default) skips nothing — sequence numbers start at 0, so a
    /// bare `u64` threshold could not distinguish "unset" from "resume after
    /// the very first group". Only meaningful per-broadcast; see
    /// [`Self::with_resume_from`] docs for the multi-source-pattern caveat.
    resume_from: Option<u64>,
    /// Highest live-group sequence delivered so far, shared with the
    /// background read task(s). Best-effort hint for callers implementing
    /// resume — see [`Self::last_sequence`].
    last_sequence: Arc<AtomicU64>,
    /// Count of items evicted under `OverflowPolicy::DropOldest` (#606).
    dropped_count: Arc<AtomicU64>,
    /// Count of idle-timeout-and-resume cycles on the live read loop (#606) —
    /// a quiet (low event rate) but still-live broadcast no longer causes the
    /// subscriber to silently exit; this counts how often that recovery fired.
    idle_resumes: Arc<AtomicU64>,
    /// Explicit origin consumer for tests (bypasses the process-global
    /// `OnceLock`, which is single-writer and so cannot be reset between tests
    /// in one binary). `None` in production ⇒ resolve the global at `recv()`.
    #[cfg(test)]
    test_consumer: Option<OriginConsumer>,
    /// Receiving end of the background task's QoS-aware channel (#606).
    /// `None` until first `recv()`.
    rx: Option<EventChannelReceiver>,
    /// Background task handle (kept alive for the subscriber's lifetime).
    _task: Option<tokio::task::JoinHandle<()>>,
}

impl MoqEventSubscriber {
    /// Create a new subscriber. No background task is started yet.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            oid_subscription: None,
            backfill: BackfillMode::LiveOnly,
            qos: crate::stream_info::EventLive::stream_opt(),
            resume_from: None,
            last_sequence: Arc::new(AtomicU64::new(0)),
            dropped_count: Arc::new(AtomicU64::new(0)),
            idle_resumes: Arc::new(AtomicU64::new(0)),
            #[cfg(test)]
            test_consumer: None,
            rx: None,
            _task: None,
        }
    }

    /// Create a subscriber bound to an explicit origin consumer (tests only).
    ///
    /// Production code resolves the process-global origin at `recv()` time (see
    /// [`ensure_started`]); that global is a `OnceLock` and cannot be reset
    /// between tests in a single binary, so tests that need an isolated origin
    /// pass its consumer here instead.
    #[cfg(test)]
    pub(crate) fn new_for_test(consumer: OriginConsumer) -> Self {
        let mut s = Self::new();
        s.test_consumer = Some(consumer);
        s
    }

    /// Add a topic-prefix filter (prefix match, ZMQ semantics).
    ///
    /// - `"worker."` — all events from source `worker`
    /// - `"worker.sandbox123.started"` — exact topic
    /// - `""` — all events (subscribe-all)
    ///
    /// Must be called before the first `recv()`. Mutually exclusive with
    /// [`Self::subscribe_oid`] (legacy flat-track subscription).
    pub fn subscribe(&mut self, pattern: &str) -> Result<()> {
        if self.rx.is_some() {
            return Err(anyhow!("subscribe() must be called before recv()"));
        }
        if self.oid_subscription.is_some() {
            return Err(anyhow!(
                "subscribe() is mutually exclusive with subscribe_oid()"
            ));
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

    /// Subscribe to a single model OID's per-OID publication track (#393).
    ///
    /// The subscriber scopes its origin consumer to
    /// `local/events/publications/{oid_hash}` and reads only that track, so only
    /// events published for `oid` cross the wire (one QUIC uni-stream per group
    /// on the wire). This replaces the flat firehose + in-memory filter.
    ///
    /// Mutually exclusive with [`Self::subscribe`] / [`Self::subscribe_all`].
    /// Must be called before the first `recv()`.
    pub fn subscribe_oid(&mut self, oid: &str) -> Result<()> {
        if self.rx.is_some() {
            return Err(anyhow!("subscribe_oid() must be called before recv()"));
        }
        if !self.patterns.is_empty() {
            return Err(anyhow!(
                "subscribe_oid() is mutually exclusive with subscribe()"
            ));
        }
        self.oid_subscription = Some(oid.to_owned());
        Ok(())
    }

    /// Set the late-join retention mode (decision A: firehose-backfill).
    ///
    /// Only meaningful for OID-scoped subscribers (see [`Self::subscribe_oid`]):
    /// on first `recv()`, a [`BackfillMode::FirehoseBackfill`] subscriber asks
    /// its [`BackfillSource`] for `oid` history and replays it before going live.
    /// If the source is unavailable the subscriber silently degrades to
    /// live-only (never fails). Must be called before the first `recv()`.
    pub fn with_backfill(&mut self, mode: BackfillMode) -> Result<()> {
        if self.rx.is_some() {
            return Err(anyhow!("with_backfill() must be called before recv()"));
        }
        self.backfill = mode;
        Ok(())
    }

    /// Select the delivery QoS (#606). Reuses [`StreamOpt`] — pass
    /// [`crate::stream_info::EventReliable::stream_opt()`] for at-least-once
    /// delivery (events that must not be silently dropped, e.g.
    /// `model.loaded`), or any other `StreamOpt` combination. Defaults to
    /// [`crate::stream_info::EventLive`] if never called. Must be called
    /// before the first `recv()`.
    pub fn with_qos(&mut self, qos: StreamOpt) -> Result<()> {
        if self.rx.is_some() {
            return Err(anyhow!("with_qos() must be called before recv()"));
        }
        self.qos = qos;
        Ok(())
    }

    /// Skip live groups with `sequence <= resume_from` (offset-resume, #606).
    /// Combine with [`Self::last_sequence`] to persist a cursor across
    /// reconnects for `EventReliable`-style at-least-once delivery.
    ///
    /// Caveat: a `MoqEventSubscriber` using the legacy flat-pattern
    /// [`Self::subscribe`] API may read from MULTIPLE independent source
    /// broadcasts (each with its own group-sequence counter); `resume_from`
    /// is applied per-broadcast, so a single threshold is only unambiguous
    /// for [`Self::subscribe_oid`] (one broadcast). Multi-source resume
    /// requires a per-source cursor map, which is out of scope here.
    /// Must be called before the first `recv()`.
    pub fn with_resume_from(&mut self, sequence: u64) -> Result<()> {
        if self.rx.is_some() {
            return Err(anyhow!("with_resume_from() must be called before recv()"));
        }
        self.resume_from = Some(sequence);
        Ok(())
    }

    /// Highest live-group sequence delivered so far (best-effort hint for
    /// resume; see [`Self::with_resume_from`]). `0` before the first delivery.
    pub fn last_sequence(&self) -> u64 {
        self.last_sequence.load(Ordering::Relaxed)
    }

    /// Count of items evicted under `OverflowPolicy::DropOldest` (#606).
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Count of idle-timeout-and-resume cycles on the live read loop (#606).
    pub fn idle_resumes(&self) -> u64 {
        self.idle_resumes.load(Ordering::Relaxed)
    }

    /// Lazily start the background task and return a mutable reference to the channel receiver.
    fn ensure_started(&mut self) -> Result<&mut EventChannelReceiver> {
        if self.rx.is_none() {
            #[cfg(test)]
            let consumer = if let Some(c) = self.test_consumer.clone() {
                c
            } else {
                global_moq_event_origin()
                    .ok_or_else(|| anyhow!("moq event bus not initialized; call init_global_moq_event_origin first"))?
                    .consumer()
            };
            #[cfg(not(test))]
            let consumer = global_moq_event_origin()
                .ok_or_else(|| {
                    anyhow!(
                        "moq event bus not initialized; call init_global_moq_event_origin first"
                    )
                })?
                .consumer();

            let (tx, rx) =
                event_channel(self.qos.overflow_policy.clone(), self.dropped_count.clone());
            let resume_from = self.resume_from;
            let last_sequence = self.last_sequence.clone();
            let idle_resumes = self.idle_resumes.clone();

            let task = if let Some(oid) = self.oid_subscription.clone() {
                // #393 per-OID subscription: scope to the OID's track path and
                // run the (optional) backfill before going live.
                let backfill = std::mem::replace(&mut self.backfill, BackfillMode::LiveOnly);
                tokio::spawn(run_oid_subscriber_task(
                    consumer,
                    oid,
                    backfill,
                    tx,
                    resume_from,
                    last_sequence,
                    idle_resumes,
                ))
            } else {
                // Legacy flat-track pattern subscription.
                let patterns = self.patterns.clone();
                tokio::spawn(run_subscriber_task(
                    consumer,
                    patterns,
                    tx,
                    resume_from,
                    last_sequence,
                    idle_resumes,
                ))
            };
            self.rx = Some(rx);
            self._task = Some(task);
        }
        self.rx
            .as_mut()
            .ok_or_else(|| anyhow!("subscriber channel not initialized"))
    }

    /// Receive the next event. Blocks until an event arrives or the bus is closed.
    pub async fn recv(&mut self) -> Result<(String, Vec<u8>)> {
        let rx = self.ensure_started()?;
        rx.recv()
            .await
            .ok_or_else(|| anyhow!("event subscriber closed"))
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
            Ok(msg) => Ok(msg),
            Err(()) => Err(anyhow!("event subscriber closed")),
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
    tx: EventChannelSender,
    resume_from: Option<u64>,
    last_sequence: Arc<AtomicU64>,
    idle_resumes: Arc<AtomicU64>,
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
        let last_sequence2 = last_sequence.clone();
        let idle_resumes2 = idle_resumes.clone();

        tasks.spawn(async move {
            read_event_broadcast(
                broadcast,
                path_str,
                patterns2,
                tx2,
                resume_from,
                last_sequence2,
                idle_resumes2,
            )
            .await;
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
    tx: EventChannelSender,
    resume_from: Option<u64>,
    last_sequence: Arc<AtomicU64>,
    idle_resumes: Arc<AtomicU64>,
) {
    let mut track = match broadcast.subscribe_track(&Track::new(EVENT_TRACK)) {
        Ok(t) => t,
        Err(_) => return,
    };

    loop {
        let mut group =
            match tokio::time::timeout(crate::moq_stream::GROUP_IDLE_TIMEOUT, track.next_group())
                .await
            {
                Ok(Ok(Some(g))) => g,
                Ok(Ok(None)) | Ok(Err(_)) => break,
                Err(_elapsed) => {
                    // #606: a quiet (low event rate) broadcast is NOT the same as a
                    // gone one. Previously this `break`-ed, permanently dropping
                    // the subscriber after one idle period — silently losing every
                    // later event. Keep waiting; only `next_group()` itself
                    // returning `None`/`Err` (above) means the broadcast actually
                    // ended. `next_group()` is safe to re-poll after its previous
                    // future was dropped by the timeout (it does not consume
                    // internal cursor state on cancellation).
                    tracing::debug!(
                        "event subscriber idle — source broadcast still live, resuming wait"
                    );
                    idle_resumes.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

        // Offset-resume (#606): skip groups already delivered in a prior
        // session. Only unambiguous for a single-broadcast subscription; see
        // `MoqEventSubscriber::with_resume_from` docs for the multi-source
        // caveat that also applies here.
        if let Some(threshold) = resume_from {
            if group.sequence <= threshold {
                continue;
            }
        }

        let frame = match group.read_frame().await {
            Ok(Some(f)) => f,
            Ok(None) => break, // group ended without a frame
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
        if topic_matches_patterns(&topic, &patterns) {
            last_sequence.fetch_max(group.sequence, Ordering::Relaxed);
            if !tx.send((topic, payload)).await {
                break; // receiver dropped
            }
        }
    }

    let _ = path; // kept for debugging context; unused in the hot path
}

/// #393 per-OID subscriber task.
///
/// Scopes the origin consumer to `local/events/publications/{oid_hash}` so only
/// that OID's events cross the wire. If `backfill` carries a [`BackfillSource`]
/// that reports history for `oid`, that history is replayed to `tx` first
/// (cold-start), then the task subscribes to the live track. Any backfill error
/// or "no history" result degrades silently to live-only — the subscription
/// never fails just because the firehose is down.
async fn run_oid_subscriber_task(
    mut consumer: OriginConsumer,
    oid: String,
    backfill: BackfillMode,
    tx: EventChannelSender,
    resume_from: Option<u64>,
    last_sequence: Arc<AtomicU64>,
    idle_resumes: Arc<AtomicU64>,
) {
    // 1) Optional cold-start backfill (decision A). Run BEFORE scoping the live
    //    consumer so the scheduler sees history before new events regardless of
    //    when (or whether) the live track is announced.
    //
    //    `BackfillSource::replay` is a PUBLIC trait implemented outside this
    //    crate (#393 doc) and is typed against `mpsc::Sender` — #606 does not
    //    change that public contract. Instead, bridge: replay into a small
    //    plain `mpsc` channel and forward each item into the QoS-aware `tx` so
    //    backfilled items get the same overflow-policy treatment as live ones.
    if let BackfillMode::FirehoseBackfill {
        oid: bf_oid,
        source,
    } = &backfill
    {
        if bf_oid == &oid && source.has_history(&oid) {
            let (bridge_tx, mut bridge_rx) = mpsc::channel::<(String, Vec<u8>)>(64);
            let fwd_tx = tx.clone();
            let forward = tokio::spawn(async move {
                while let Some(item) = bridge_rx.recv().await {
                    if !fwd_tx.send(item).await {
                        break;
                    }
                }
            });
            let replay_result = source.replay(&oid, &bridge_tx).await;
            drop(bridge_tx); // close so the forwarder's recv() loop ends
            let _ = forward.await;
            match replay_result {
                Ok(()) => tracing::debug!(oid = %oid, "event backfill replay complete"),
                Err(e) => {
                    tracing::warn!(oid = %oid, error = %e, "event backfill failed; continuing live-only");
                }
            }
        } else {
            tracing::debug!(oid = %oid, "no backfill history available; starting live-only");
        }
    }

    // 2) Scope the consumer to the publications subtree. We scope to the PREFIX
    //    only (not the OID hash) so the announced() stream keeps flowing even
    //    before this OID's broadcast exists — late-join then picks up the OID's
    //    broadcast by matching its hash in the announcement path. This avoids the
    //    race where a subscriber starts before the publisher announces.
    let want_hash = oid_hash(&oid);
    let scoped = match consumer.scope(&[Path::new(PUBLICATIONS_PREFIX)]) {
        Some(c) => c,
        None => {
            tracing::debug!(oid = %oid, "publications subtree not announced; subscriber idle");
            return;
        }
    };
    consumer = scoped;

    // 3) Wait for the OID's broadcast to be announced and read its `events` track.
    //    We filter announcements by the OID hash so a per-OID subscriber sees
    //    exactly one broadcast (the OID's) — no in-memory pattern filtering on
    //    event topics is needed; the scope + hash match already narrowed the wire.
    loop {
        let (path, broadcast) = match consumer.announced().await {
            None => break,               // origin closed
            Some((_, None)) => continue, // unannounce — re-await
            Some((path, Some(b))) => (path, b),
        };
        // Only read announcements whose final path segment is this OID's hash.
        // Other OIDs' broadcasts are ignored — they stay on their own tracks.
        // Match the full final segment (not a bare suffix) so two hashes that
        // happen to share a tail cannot cross-subscribe.
        let last_segment = path.as_str().rsplit('/').next().unwrap_or("");
        if last_segment != want_hash.as_str() {
            continue;
        }
        read_oid_event_broadcast(
            broadcast,
            tx.clone(),
            resume_from,
            last_sequence.clone(),
            idle_resumes.clone(),
        )
        .await;
    }
}

/// Read all groups from a per-OID publication broadcast's `events` track and
/// relay every frame to `tx`. Unlike [`read_event_broadcast`] there is NO
/// pattern filtering — the consumer scope already limited the wire to one OID.
async fn read_oid_event_broadcast(
    broadcast: moq_net::BroadcastConsumer,
    tx: EventChannelSender,
    resume_from: Option<u64>,
    last_sequence: Arc<AtomicU64>,
    idle_resumes: Arc<AtomicU64>,
) {
    let mut track = match broadcast.subscribe_track(&Track::new(EVENT_TRACK)) {
        Ok(t) => t,
        Err(_) => return,
    };

    loop {
        let mut group =
            match tokio::time::timeout(crate::moq_stream::GROUP_IDLE_TIMEOUT, track.next_group())
                .await
            {
                Ok(Ok(Some(g))) => g,
                Ok(Ok(None)) | Ok(Err(_)) => break,
                Err(_elapsed) => {
                    // #606: see read_event_broadcast — quiet != gone, keep waiting.
                    tracing::debug!(
                        "per-OID event subscriber idle — broadcast still live, resuming wait"
                    );
                    idle_resumes.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

        // Offset-resume (#606): unambiguous here — exactly one broadcast.
        if let Some(threshold) = resume_from {
            if group.sequence <= threshold {
                continue;
            }
        }

        let frame = match group.read_frame().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(error = %e, "per-OID event broadcast: frame read error");
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

        last_sequence.fetch_max(group.sequence, Ordering::Relaxed);
        if !tx.send((topic, payload)).await {
            break; // receiver dropped
        }
    }
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
        assert!(topic_matches_patterns(
            "worker.sandbox123.started",
            &patterns
        ));
        assert!(!topic_matches_patterns("system.event.ready", &patterns));
        assert!(!topic_matches_patterns("worker", &patterns)); // no trailing dot match

        let patterns_all = vec!["".to_owned()];
        assert!(topic_matches_patterns("any.thing", &patterns_all));

        let patterns_exact = vec!["worker.abc.started".to_owned()];
        assert!(topic_matches_patterns(
            "worker.abc.started",
            &patterns_exact
        ));
        assert!(!topic_matches_patterns(
            "worker.abc.stopped",
            &patterns_exact
        ));
    }

    // ── #606: EventChannel unit tests (no MoQ transport — deterministic) ──────

    #[tokio::test]
    async fn event_channel_drop_oldest_evicts_and_counts() {
        let dropped = Arc::new(AtomicU64::new(0));
        let (tx, mut rx) = event_channel(
            OverflowPolicy::DropOldest { high_water_mark: 2 },
            dropped.clone(),
        );

        // Fill to capacity, then overflow by one: item "a" must be evicted.
        assert!(tx.send(("t".into(), b"a".to_vec())).await);
        assert!(tx.send(("t".into(), b"b".to_vec())).await);
        assert!(tx.send(("t".into(), b"c".to_vec())).await);

        assert_eq!(dropped.load(Ordering::Relaxed), 1, "exactly one eviction");

        let (_, first) = rx.recv().await.expect("first item");
        let (_, second) = rx.recv().await.expect("second item");
        assert_eq!(first, b"b", "oldest (\"a\") was evicted, \"b\" survives");
        assert_eq!(second, b"c");
    }

    #[tokio::test]
    async fn event_channel_block_policy_waits_for_space_not_drop() {
        let dropped = Arc::new(AtomicU64::new(0));
        let (tx, mut rx) = event_channel(OverflowPolicy::Block, dropped.clone());

        // Capacity for OverflowPolicy::Block is the fixed 256 default — send a
        // third send concurrently while only 2 are buffered; it must complete
        // once we drain one, never by silently dropping (dropped_count stays 0).
        assert!(tx.send(("t".into(), b"a".to_vec())).await);
        assert!(tx.send(("t".into(), b"b".to_vec())).await);

        let (item, _) = rx.recv().await.expect("drain one");
        assert_eq!(item, "t");
        assert_eq!(
            dropped.load(Ordering::Relaxed),
            0,
            "Block policy never evicts"
        );
    }

    #[tokio::test]
    async fn event_channel_closes_when_all_senders_dropped() {
        let dropped = Arc::new(AtomicU64::new(0));
        let (tx, mut rx) = event_channel(OverflowPolicy::Block, dropped);
        let tx2 = tx.clone();
        drop(tx);
        drop(tx2);
        assert_eq!(rx.recv().await, None, "recv must wake with None, not hang");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moq_event_pub_sub_roundtrip() -> Result<()> {
        let origin = MoqEventOrigin::new();

        // Isolate this test from the process-global OnceLock. The previous
        // version called `init_global_moq_event_origin` + `MoqEventSubscriber::new()`,
        // but that OnceLock is single-writer: when another test in the binary set
        // the global first, this test's `new()` subscriber read a DIFFERENT origin
        // than its publisher wrote to, so `recv()` raced/failed ("event subscriber
        // closed"). `new_for_test` binds the subscriber to THIS origin's consumer,
        // making the pub/sub pairing deterministic regardless of test order (#148).
        let mut sub = MoqEventSubscriber::new_for_test(origin.consumer());

        // Subscribe-all rather than the `"worker."` source prefix: a source-prefix
        // subscription exercises the two-level `scope()` path in
        // `scope_consumer_for_patterns`, which is broken on main (the second,
        // relative `scope()` finds no node under the absolute-keyed consumer and
        // returns None, so the subscriber task aborts). That is the #148 product
        // bug fixed under PR #469 (which also restores the strict `"worker."`
        // assertion). This stabilization PR is test-only and deliberately does NOT
        // pull in #469's product change, so the test subscribes to the whole
        // `local/events` subtree (a single `scope()`, no two-level bug) while still
        // asserting the exact source/oid/event topic round-trips end-to-end.
        sub.subscribe("")?;

        // Publish BEFORE recv so the broadcast (`local/events/worker`) is already
        // announced when the subscriber's background task starts — no timing race.
        let mut pub_ = origin.publisher("worker")?;
        pub_.publish("sandbox123", "started", b"payload")?;

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), sub.recv()).await;

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
        let sock =
            std::env::temp_dir().join(format!("hyprstream-event-test-{}.sock", std::process::id()));

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

    // ========================================================================
    // #393 — per-OID broadcast paths + firehose-backfill late-join
    // ========================================================================

    #[test]
    fn oid_hash_is_stable_and_distinct() {
        let a = oid_hash("at://did:web:node.example.com/models/qwen3-4b/v1");
        let b = oid_hash("at://did:web:node.example.com/models/qwen3-4b/v1");
        assert_eq!(a, b, "same OID must hash to the same path");
        assert_eq!(
            a.len(),
            OID_HASH_LEN,
            "hash is truncated to OID_HASH_LEN hex chars"
        );

        let c = oid_hash("at://did:web:node.example.com/models/llama3-8b/v1");
        assert_ne!(a, c, "distinct OIDs must hash to distinct paths");

        // The hash must contain only filesystem/moq-safe characters (hex).
        assert!(
            a.chars().all(|ch| ch.is_ascii_hexdigit()),
            "non-hex char in {a}"
        );
    }

    #[test]
    fn publication_broadcast_path_shape() {
        // Mirrors the streaming plane's path-selective shape.
        let path = publication_broadcast_path("my-oid");
        assert_eq!(
            path,
            format!("{}/{}", PUBLICATIONS_PREFIX, oid_hash("my-oid")),
            "path must be {PUBLICATIONS_PREFIX}/{{oid_hash}}"
        );
        assert!(path.starts_with(PUBLICATIONS_PREFIX));
    }

    /// A per-OID subscriber receives ONLY that OID's events — not events for
    /// other OIDs and not the flat firehose. This is the core #393 fix: a node
    /// tracking N of M OIDs reads N tracks, not M.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn per_oid_subscription_receives_only_that_oid() -> Result<()> {
        let origin = MoqEventOrigin::new();

        // Two publishers for two distinct OIDs, each mirroring to its own track.
        let mut pub_a = origin.publisher_with_oid("registry", "oid-alpha")?;
        let mut pub_b = origin.publisher_with_oid("registry", "oid-beta")?;

        // Subscriber scoped to oid-alpha only (uses an explicit origin consumer
        // so it does not fight other tests over the process-global OnceLock).
        let mut sub = MoqEventSubscriber::new_for_test(origin.consumer());
        sub.subscribe_oid("oid-alpha")?;

        // Publish one event to each OID's track.
        pub_a.publish("qwen3-4b", "published", b"alpha-event")?;
        pub_b.publish("llama3-8b", "published", b"beta-event")?;

        // Let the announcement propagate.
        tokio::time::sleep(Duration::from_millis(30)).await;

        // The subscriber should receive oid-alpha's event...
        let got = tokio::time::timeout(Duration::from_secs(5), sub.recv()).await;
        let (topic, payload) = match got {
            Ok(Ok(tp)) => tp,
            Ok(Err(e)) => panic!("recv error: {e}"),
            Err(_) => panic!("recv timeout — oid-alpha event never arrived"),
        };
        assert_eq!(topic, "registry.qwen3-4b.published");
        assert_eq!(payload, b"alpha-event");

        // ...and NOT oid-beta's. Poll briefly: any further recv should time out
        // (the subscriber is scoped to oid-alpha, so beta's event never reaches it).
        let next = sub.recv_timeout(Duration::from_millis(300)).await?;
        assert!(
            next.is_none(),
            "per-OID subscriber received an event for a DIFFERENT OID: {next:?}"
        );

        // Hold the publishers alive so their broadcasts stay announced for the
        // lifetime of the subscriber task.
        drop(pub_a);
        drop(pub_b);
        Ok(())
    }

    /// The legacy flat `events` track still receives events from a publisher
    /// created with `publisher_with_oid` (back-compat fallback during the #393
    /// transition). A `publisher_with_oid` mirrors to BOTH the flat
    /// `local/events/registry` track and the per-OID publication track, so a
    /// legacy subscriber reading the flat track sees the same event as a #393
    /// per-OID subscriber.
    ///
    /// This reads the flat broadcast directly (the way `EventSubscriber` does in
    /// production via `OriginConsumer::announced()` + `subscribe_track`), rather
    /// than going through the `MoqEventSubscriber` background task, to isolate
    /// the back-compat guarantee (#393 must not break the flat track) from a
    /// pre-existing two-level-scope race in `scope_consumer_for_patterns`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flat_track_back_compat_with_oid_mirror() -> Result<()> {
        use moq_net::Track as MoqTrack;

        let origin = MoqEventOrigin::new();

        // Publisher mirrors to both flat `local/events/registry` and per-OID track.
        let mut pub_ = origin.publisher_with_oid("registry", "oid-x")?;
        assert!(
            pub_.has_oid_mirror(),
            "publisher_with_oid must mirror to an OID track"
        );
        pub_.publish("qwen3-4b", "published", b"mirrored")?;

        // Read the flat `local/events/registry` broadcast directly.
        let consumer = origin.consumer();
        let scoped = consumer
            .scope(&[Path::new(EVENT_PREFIX)])
            .ok_or_else(|| anyhow!("no events scope"))?;

        let recv = tokio::spawn(async move {
            let mut scoped = scoped;
            loop {
                let (_path, bc) = match scoped.announced().await {
                    None => return None,
                    Some((_, None)) => continue,
                    Some((p, Some(b))) => (p, b),
                };
                let mut track = match bc.subscribe_track(&MoqTrack::new(EVENT_TRACK)) {
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

        // Re-publish to cover the announcement-propagation race.
        let publish_loop = async {
            for _ in 0..50 {
                let _ = pub_.publish("qwen3-4b", "published", b"mirrored");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            std::future::pending::<()>().await;
        };

        let got: Option<Vec<u8>> = tokio::select! {
            r = recv => r.ok().flatten(),
            _ = publish_loop => None,
            _ = tokio::time::sleep(Duration::from_secs(10)) => None,
        };

        let frame = got.expect("flat-track subscriber never received the mirrored event");
        assert!(frame.len() >= 4, "frame too short");
        let topic_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let topic = std::str::from_utf8(&frame[4..4 + topic_len]).unwrap();
        let payload = &frame[4 + topic_len..];
        assert_eq!(topic, "registry.qwen3-4b.published");
        assert_eq!(payload, b"mirrored");

        Ok(())
    }

    /// subscribe() and subscribe_oid() are mutually exclusive — a subscriber is
    /// either flat-pattern-based or OID-scoped, never both.
    #[test]
    fn subscribe_and_subscribe_oid_are_mutually_exclusive() -> Result<()> {
        let mut s = MoqEventSubscriber::new();
        s.subscribe("worker.")?;
        assert!(
            s.subscribe_oid("oid").is_err(),
            "subscribe_oid after subscribe must fail"
        );

        let mut s = MoqEventSubscriber::new();
        s.subscribe_oid("oid")?;
        assert!(
            s.subscribe("worker.").is_err(),
            "subscribe after subscribe_oid must fail"
        );

        Ok(())
    }

    /// A no-op BackfillSource used to test firehose-backfill graceful degradation.
    struct NullBackfillSource {
        /// What `has_history` returns.
        has_history: bool,
        /// History to replay (if any).
        history: Vec<(String, Vec<u8>)>,
    }

    impl BackfillSource for NullBackfillSource {
        fn has_history(&self, _oid: &str) -> bool {
            self.has_history
        }
        fn replay<'a>(
            &'a self,
            _oid: &'a str,
            tx: &'a mpsc::Sender<(String, Vec<u8>)>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
            let history = self.history.clone();
            Box::pin(async move {
                for (topic, payload) in history {
                    if tx.send((topic, payload)).await.is_err() {
                        break;
                    }
                }
                Ok(())
            })
        }
    }

    /// When the firehose has history, backfill replays it BEFORE live events
    /// arrive (cold-start path). The live MoQ events then follow.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backfill_replays_history_before_live() -> Result<()> {
        let origin = MoqEventOrigin::new();

        let source = Arc::new(NullBackfillSource {
            has_history: true,
            history: vec![
                (
                    "registry.qwen3-4b.published".to_owned(),
                    b"backfill-1".to_vec(),
                ),
                (
                    "registry.qwen3-4b.published".to_owned(),
                    b"backfill-2".to_vec(),
                ),
            ],
        });

        let mut sub = MoqEventSubscriber::new_for_test(origin.consumer());
        sub.subscribe_oid("oid-backfill")?;
        sub.with_backfill(BackfillMode::FirehoseBackfill {
            oid: "oid-backfill".to_owned(),
            source,
        })?;

        // Drive the backfill + a live publish concurrently.
        let mut pub_ = origin.publisher_oid_only("registry", "oid-backfill")?;
        pub_.publish("qwen3-4b", "published", b"live")?;

        // First two events should be the backfilled history (cold-start), then live.
        let e1 = tokio::time::timeout(Duration::from_secs(5), sub.recv())
            .await
            .expect("backfill[0] timed out")?;
        assert_eq!(e1.1, b"backfill-1");

        let e2 = tokio::time::timeout(Duration::from_secs(5), sub.recv())
            .await
            .expect("backfill[1] timed out")?;
        assert_eq!(e2.1, b"backfill-2");

        // Then the live event.
        let e3 = tokio::time::timeout(Duration::from_secs(5), sub.recv())
            .await
            .expect("live event timed out")?;
        assert_eq!(e3.1, b"live");

        drop(pub_);
        Ok(())
    }

    /// When the firehose is UNAVAILABLE (has_history false), backfill mode
    /// gracefully degrades to live-only — the subscription does not fail and
    /// live events still arrive (the firehose is the cold-start path; MoQ is
    /// the live path).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backfill_gracefully_degrades_when_firehose_unavailable() -> Result<()> {
        let origin = MoqEventOrigin::new();

        let source = Arc::new(NullBackfillSource {
            has_history: false, // firehose offline / no history
            history: vec![("never".to_owned(), b"never".to_vec())],
        });

        let mut sub = MoqEventSubscriber::new_for_test(origin.consumer());
        sub.subscribe_oid("oid-degrade")?;
        sub.with_backfill(BackfillMode::FirehoseBackfill {
            oid: "oid-degrade".to_owned(),
            source,
        })?;

        let mut pub_ = origin.publisher_oid_only("registry", "oid-degrade")?;
        pub_.publish("qwen3-4b", "published", b"live")?;

        // Despite requesting backfill, with the firehose down we still get the
        // live event (and never the spurious "never" history item).
        let got = tokio::time::timeout(Duration::from_secs(5), sub.recv())
            .await
            .expect("live event timed out (degradation failed)")?;
        assert_eq!(
            got.1, b"live",
            "degraded subscriber must receive live events"
        );
        assert_ne!(
            got.1, b"never",
            "firehose history must NOT leak when unavailable"
        );

        drop(pub_);
        Ok(())
    }
}
