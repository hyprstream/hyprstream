//! moq-lite streaming plane (M2a of epic #134).
//!
//! Point-to-point / per-recipient streaming only (request/response, token
//! streams, and per-recipient unique topics such as `NotificationService`'s
//! blind broadcast — each recipient gets its own topic, not a shared fan-out
//! track). Broadcast fan-out (N subscribers learning about one event from one
//! source) is consolidated on the sibling `crate::moq_event` origin instead —
//! see its module doc for the two-origins split.
//!
//! This is the moq-native replacement for the ZMQ PULL→XPUB queuing proxy in
//! [`crate::service::streaming`]. Instead of a forwarding proxy with a custom
//! per-topic queue + late-join rejoin buffer, in-process publishers append
//! directly to a shared `moq_net::Origin`, and external subscribers consume the
//! *same* origin over the `moql` ALPN via
//! [`crate::transport::iroh_moq::IrohMoqProtocolHandler`].
//!
//! # Topic → Track mapping
//!
//! The existing call-site contract is a topic string (64-hex DH-derived, or a
//! `notify-*` / control topic) plus opaque payload bytes. We map:
//!
//! ```text
//!   topic-string  ->  moq Broadcast path  {tenant}/{service}/{topic}/{instance}
//!   StreamBlock   ->  one moq Group (sequence = block index)
//!                       containing one Frame = capnp_bytes || mac[16]
//! ```
//!
//! For the in-process / direct path (M2a), the broadcast path is built from a
//! caller-supplied `{tenant}/{service}/.../{instance}` prefix joined with the
//! opaque topic. The topic itself stays opaque (and, for the relay'd path in
//! M2b, DH-derived/unguessable). A single track name (`STREAM_TRACK`) carries
//! the block sequence.
//!
//! # §7.5 chained-HMAC tokenstream
//!
//! The chained-HMAC envelope is preserved 1:1: each Group's Frame payload is
//! the same `[capnp StreamBlock || 16-byte truncated MAC]` that the ZMQ wire
//! format carried in frames 1+2 (frame 0, the topic, is now the Track/Broadcast
//! path and is therefore dropped from the payload). [`StreamHmacState`] /
//! [`StreamVerifier`] are reused unchanged, so the MAC chain — which binds each
//! block to its predecessor — is byte-identical to the ZMQ path.
//!
//! Late-join is now moq's job: a subscriber that joins late is served the
//! latest Group natively by the moq Track cache (respecting upstream Group
//! cache consts), so the custom `StreamResume` / rejoin-buffer code is gone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use moq_net::{BroadcastProducer, Group, OriginConsumer, OriginProducer, Track, TrackProducer};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use crate::crypto::StreamHmacState;
use crate::streaming::{StreamContext, StreamPayloadData, StreamVerifier};

// ============================================================================
// Process-global moq streaming origin — set once at startup, read everywhere.
// Allows StreamChannel (publisher side) to use the same origin as StreamService
// (server side) without threading it through every service factory.
// ============================================================================

static GLOBAL_MOQ_ORIGIN: OnceLock<MoqStreamOrigin> = OnceLock::new();
/// The UDS socket path serving the moq plane (set by `serve_moq_uds_background`).
static GLOBAL_MOQ_UDS_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Register the process-global moq streaming origin.
///
/// Must be called once at startup (from the streams service factory) before any
/// `StreamChannel::publisher()` call. Returns `true` on first call, `false` if
/// already set (idempotent — a second set is silently ignored).
pub fn init_global_moq_origin(origin: MoqStreamOrigin) -> bool {
    GLOBAL_MOQ_ORIGIN.set(origin).is_ok()
}

/// Borrow the process-global moq streaming origin, if initialized.
///
/// `None` when moq is not yet wired (unit tests, ZMQ-only deployments).
pub fn global_moq_origin() -> Option<&'static MoqStreamOrigin> {
    GLOBAL_MOQ_ORIGIN.get()
}

/// Path of the UDS socket that serves the moq streaming plane to local clients.
///
/// Set by [`serve_moq_uds_background`] once the listener is ready.
/// `None` in ZMQ-only or unit-test deployments.
pub fn global_moq_uds_path() -> Option<&'static Path> {
    GLOBAL_MOQ_UDS_PATH.get().map(PathBuf::as_path)
}

// ============================================================================
// Process-global producer reach (#274) — the network-routable way external
// subscribers reach this node's moq plane. Set once when the QUIC /
// web_transport_quinn server binds; read by every StreamInfo producer site so
// the `reach` field is built from ONE source (no per-site assembly, no drift),
// the same Rust value `root_did_document()` uses for its QuicTransport entry.
// ============================================================================

/// The node's network-routable moq reach: the bound QUIC address, its TLS
/// server name, and the leaf-cert SHA-256 pins. `None` until the daemon binds
/// its `web_transport_quinn` server (UDS-only / unit-test deployments).
static GLOBAL_PRODUCER_REACH: OnceLock<NodeStreamReach> = OnceLock::new();

/// The node's own moq reach parameters — the single source for the `reach`
/// list every StreamInfo producer publishes.
#[derive(Clone, Debug)]
pub struct NodeStreamReach {
    /// Bound socket address external subscribers dial (`/moq` over WebTransport).
    pub addr: std::net::SocketAddr,
    /// TLS SNI / WebPKI validation name advertised for the endpoint.
    pub server_name: String,
    /// Acceptable leaf-cert SHA-256 pins (self-signed mesh; rotation = multiple).
    pub cert_hashes: Vec<[u8; 32]>,
}

/// Register the node's network-routable moq reach (idempotent — first wins).
///
/// Called once when the daemon binds its `web_transport_quinn` server, with the
/// same `(addr, server_name, cert_hash)` the RPC endpoint registers and the
/// DID-doc `#quic` entry advertises.
pub fn init_global_producer_reach(reach: NodeStreamReach) -> bool {
    GLOBAL_PRODUCER_REACH.set(reach).is_ok()
}

/// Borrow the node's registered network reach, if the QUIC server is bound.
pub fn global_producer_reach() -> Option<&'static NodeStreamReach> {
    GLOBAL_PRODUCER_REACH.get()
}

/// The node's own iroh `EndpointId` (Ed25519 public key) once the iroh substrate
/// is bound (#357). Registered separately from [`GLOBAL_PRODUCER_REACH`] because
/// the QUIC reach is set when the quinn server binds, whereas iroh binds slightly
/// later in the same bootstrap; folding it back into the first-wins
/// [`NodeStreamReach`] would require reordering the bind sequence.
static GLOBAL_IROH_NODE_ID: OnceLock<[u8; 32]> = OnceLock::new();

/// Register the node's iroh `EndpointId` so [`producer_reach`] can advertise an
/// iroh-direct [`Destination`] for native peers (#357). Idempotent (first wins).
///
/// Called once when the daemon binds its iroh substrate (the `node_id` ==
/// `signing_key.verifying_key()`, already covered by the node's DID).
pub fn init_global_iroh_node_id(node_id: [u8; 32]) -> bool {
    GLOBAL_IROH_NODE_ID.set(node_id).is_ok()
}

/// Borrow the node's registered iroh `EndpointId`, if the iroh substrate is bound.
pub fn global_iroh_node_id() -> Option<&'static [u8; 32]> {
    GLOBAL_IROH_NODE_ID.get()
}

// ============================================================================
// Process-global producer-chosen moq RELAY (#358) — the rendezvous endpoint a
// producing service advertises so that NEITHER the publisher NOR the subscriber
// must be directly reachable by the other: both rendezvous through this relay.
//
// The relay is *producer-chosen* (default = the service's PDS / federation
// anchor) and set once when the producing node learns its relay. It is the
// network-routable `TransportConfig` of the relay's moq plane — the SAME codec
// the DID document's transport `service` entries use ([`crate::service_entry`]),
// so the advertised stream relay and the DID transport address never drift.
//
// The relay carries AEAD-sealed ciphertext it cannot read (the `enc_key` /
// `TaggedPayload` path seals at source) and never holds the `mac_key` /
// `enc_key`: it is blind by construction, not by trust. Per-PDS, shared, and
// oblivious relays are therefore all safe for content confidentiality.
// ============================================================================

/// The producer-chosen moq relay this node rendezvouses through (#358).
///
/// Held as the wire-form [`crate::stream_info::TransportConfig`] (the reach
/// codec), so the SAME value is both advertised verbatim in `StreamInfo.reach`
/// and dialed (via the shared [`reach_to_transport_config`] resolver) by the
/// relay client — no per-site assembly, no drift between the advertised stream
/// relay and the dialed one.
///
/// `None` until the node is configured with a relay (no relay = direct-only
/// advertisement, the S1/S2 behaviour). The default deployment co-locates this
/// with the node's `#atproto_pds` DID service entry where the node is the PDS.
static GLOBAL_RELAY_REACH: OnceLock<crate::stream_info::TransportConfig> = OnceLock::new();

/// Register the producer-chosen moq relay endpoint (idempotent — first wins).
///
/// `relay` is the relay's network-routable transport in wire-reach form
/// ([`crate::stream_info::TransportConfig`] — Quic / WebTransport or iroh). A
/// node sources this from its resolved DID transport entry decoded by the SAME
/// [`crate::service_entry`] codec the DID document uses (default: the PDS /
/// federation anchor), never hand-assembled — see [`relay_reach_from_decoded`].
///
/// After this is set, [`producer_reach`] advertises a `Role::Relay`
/// [`Destination`] and [`serve_origin_to_relay_background`] should be spawned to
/// announce this node's broadcasts UP to the relay.
pub fn init_global_relay_reach(relay: crate::stream_info::TransportConfig) -> bool {
    GLOBAL_RELAY_REACH.set(relay).is_ok()
}

/// Borrow the producer-chosen relay endpoint (wire-reach form), if configured.
pub fn global_relay_reach() -> Option<&'static crate::stream_info::TransportConfig> {
    GLOBAL_RELAY_REACH.get()
}

// ============================================================================
// Per-server reach context (#384) — de-singletonize GLOBAL_RELAY_REACH.
//
// Relay/reach choice is logically PER-STREAM, but was historically wired
// PER-PROCESS through three `OnceLock` singletons (`GLOBAL_IROH_NODE_ID`,
// `GLOBAL_PRODUCER_REACH`, `GLOBAL_RELAY_REACH`). That granularity prevented a
// relay-only-anonymized stream X from coexisting with a direct stream Y in the
// same process, blocked per-tenant relay isolation, and the first-write-wins
// `OnceLock` semantics silently clobbered later writes.
//
// [`ProducerReachConfig`] carries the node/server's reach inputs as plain,
// `Clone` config data (trivially `Send + Sync`). The reach list is built by its
// [`ProducerReachConfig::reach`] METHOD reading its OWN fields — never process
// globals. A per-stream relay override
// ([`ProducerReachConfig::reach_with_relay`]) lets one stream pick a different
// relay (or go relay-only / anonymized) while another in the same process stays
// direct — the heterogeneous-anonymization property #384 requires.
//
// The process globals remain ONLY as a populated-once compatibility source for
// the deprecated free-function [`producer_reach`]; new code threads a
// `ProducerReachConfig` (see [`global_reach_config`]). (Tier 3 — scheduled
// relay selection — is deliberately not built but not structurally precluded:
// a scheduler can simply hand a per-stream override to `reach_with_relay`.)
// ============================================================================

/// A node/server's moq reach inputs (#384) — the per-server context that builds
/// a producer's `StreamInfo.reach`, replacing the three `OnceLock` singletons.
///
/// Plain config data (`Clone`, `Send + Sync`). The common-case relay is a
/// server-global value (often an anycast PDS / federation-anchor address)
/// carried in [`relay`]; a per-stream override is supplied at build time via
/// [`ProducerReachConfig::reach_with_relay`].
#[derive(Clone, Debug, Default)]
pub struct ProducerReachConfig {
    /// The node's own iroh `EndpointId` (Ed25519 public key, 32 bytes) when the
    /// iroh substrate is bound (#357). `None` → no iroh-direct reach advertised.
    pub iroh_node_id: Option<[u8; 32]>,
    /// The node's network-routable Quic / WebTransport reach (the bound QUIC
    /// address + TLS name + leaf-cert pins). `None` → no Quic-direct reach
    /// (UDS-only / unit-test deployments).
    pub quic_reach: Option<NodeStreamReach>,
    /// The server-global producer-chosen relay (#358), in wire-reach form.
    /// `None` → direct-only advertisement (S1/S2 behaviour). A per-stream
    /// override may supersede this for an individual stream.
    pub relay: Option<crate::stream_info::TransportConfig>,
}

impl ProducerReachConfig {
    /// Build this server's `StreamInfo.reach` using its server-global relay.
    ///
    /// Equivalent to `reach_with_relay(RelayChoice::ServerDefault)`. See
    /// [`ProducerReachConfig::reach_with_relay`] for the per-stream override and
    /// the reach-ordering contract.
    pub fn reach(&self) -> Vec<crate::stream_info::Destination> {
        self.reach_with_relay(RelayChoice::ServerDefault)
    }

    /// Build a stream's `StreamInfo.reach`, applying a per-stream relay choice (#384).
    ///
    /// Reach ORDERING contract (unchanged — `select_reach` does the QoS reorder
    /// downstream): **iroh-direct first, then Quic, then relay** ("direct-first"
    /// wire order). Server authority is preserved: the client may only route
    /// among the reaches advertised here, so a relay-only stream
    /// ([`RelayChoice::Only`]) omits the direct reaches and stays anonymized.
    pub fn reach_with_relay(
        &self,
        relay_choice: RelayChoice,
    ) -> Vec<crate::stream_info::Destination> {
        use crate::stream_info::{Destination, IrohReach, QuicReach, Role, TransportConfig};
        let mut reach = Vec::new();

        // A per-stream `Only` (relay-only / anonymized) override omits ALL direct
        // reaches so the client cannot route around the relay (server authority).
        let relay_only = matches!(relay_choice, RelayChoice::Only(_));

        if !relay_only {
            // iroh-direct first (#357): native peers prefer the NAT-traversing,
            // pkarr-discoverable direct path. Listed ahead of Quic so the shared
            // resolver (`connect_moq_reach`) tries it before the Quic/UDS
            // fallbacks. `None` when iroh is disabled/unbound (Quic-only).
            if let Some(node_id) = self.iroh_node_id {
                reach.push(Destination {
                    role: Role::Direct,
                    // moq streaming reach → moql ALPN; relayUrl empty = direct/pkarr (#282).
                    transport: TransportConfig::Iroh(IrohReach {
                        node_id,
                        alpn: String::from_utf8_lossy(
                            crate::transport::iroh_substrate::ALPN_MOQ_LITE,
                        )
                        .into_owned(),
                        relay_url: String::new(),
                    }),
                });
            }

            // Quic / WebTransport reach: directly-reachable peers + browsers.
            // Kept as a fallback alongside iroh (S1's fallbacks preserved).
            if let Some(r) = &self.quic_reach {
                reach.push(Destination {
                    role: Role::Direct,
                    transport: TransportConfig::Quic(QuicReach {
                        addr: r.addr.to_string(),
                        server_name: r.server_name.clone(),
                        cert_hashes: r.cert_hashes.iter().map(|h| h.to_vec()).collect(),
                    }),
                });
            }
        }

        // Relay reach (#358): the rendezvous endpoint. A subscriber that cannot
        // (or, per QoS, prefers not to) reach the producer directly rendezvouses
        // through it instead. Listed AFTER the direct reaches ("direct-first"
        // wire order); the QoS-aware `select_reach` reorders relay-first for
        // resumable/retained/fan-out streams (Job/Log).
        //
        // The per-stream choice picks WHICH relay (or none):
        //   - `ServerDefault` → this server's `self.relay` (the common case),
        //   - `Override(r)`   → a per-stream relay (per-tenant isolation),
        //   - `Only(r)`       → a per-stream relay, direct reaches omitted,
        //   - `NoRelay`        → no relay reach (direct-only for this stream).
        let relay = match relay_choice {
            RelayChoice::ServerDefault => self.relay.clone(),
            RelayChoice::Override(r) | RelayChoice::Only(r) => Some(r),
            RelayChoice::NoRelay => None,
        };
        if let Some(relay) = relay {
            reach.push(Destination { role: Role::Relay, transport: relay });
        }

        reach
    }
}

/// Per-stream relay selection for [`ProducerReachConfig::reach_with_relay`] (#384).
///
/// Lets one stream pick a relay (or go relay-only / direct-only) independently
/// of another in the SAME process — the heterogeneous-anonymization property.
#[derive(Clone, Debug, Default)]
pub enum RelayChoice {
    /// Use the server-global relay ([`ProducerReachConfig::relay`]) — the common
    /// case. Falls back to direct-only when the server has no relay configured.
    #[default]
    ServerDefault,
    /// Override the server-global relay with a per-stream relay (e.g. per-tenant
    /// isolation). Direct reaches are still advertised alongside it.
    Override(crate::stream_info::TransportConfig),
    /// Relay-ONLY (anonymized): use this per-stream relay and OMIT all direct
    /// reaches, so the client can only route through the relay (server authority).
    Only(crate::stream_info::TransportConfig),
    /// No relay reach for this stream — advertise the direct reaches only.
    ///
    /// Named `NoRelay` (not `None`) to avoid shadowing [`Option::None`] under a
    /// glob import of this enum's variants.
    NoRelay,
}

/// Snapshot the process globals into a [`ProducerReachConfig`] (#384 compat).
///
/// Bridges the deprecated [`producer_reach`] free function (and call sites not
/// yet threaded with a `ProducerReachConfig`) to the per-server context. New
/// code should thread an explicit [`ProducerReachConfig`] instead — this reads
/// the `OnceLock` singletons whose first-write-wins clobbering #384 removes.
pub fn global_reach_config() -> ProducerReachConfig {
    ProducerReachConfig {
        iroh_node_id: global_iroh_node_id().copied(),
        quic_reach: global_producer_reach().cloned(),
        relay: global_relay_reach().cloned(),
    }
}

/// Build the `StreamInfo.reach` list for a producer on this node (#274).
///
/// **Deprecated (#384):** reads the process-global `OnceLock` singletons, which
/// cannot express per-stream / per-tenant relay choice and clobber on a second
/// write. New code should construct a [`ProducerReachConfig`] (threaded from the
/// daemon bind site) and call [`ProducerReachConfig::reach`] /
/// [`ProducerReachConfig::reach_with_relay`]. Retained as a thin compatibility
/// shim over [`global_reach_config`] for call sites not yet threaded.
pub fn producer_reach() -> Vec<crate::stream_info::Destination> {
    global_reach_config().reach()
}

/// Build the producer-chosen relay's wire-reach [`crate::stream_info::TransportConfig`]
/// from a DID-document transport entry decoded by [`crate::service_entry`] (#358).
///
/// This is the bridge that keeps the stream relay address and the DID transport
/// address from drifting: a node resolves its relay's DID service entry (default
/// `#atproto_pds`), decodes it with the shared [`crate::service_entry::decode_service_entry`]
/// codec, and passes the resulting [`crate::transport::TransportConfig`] here to
/// obtain the wire-reach form to register via [`init_global_relay_reach`].
///
/// Returns `None` for same-host transports (`ipc`/`inproc`/`systemd-fd`), which
/// are never network-routable relay endpoints.
pub fn relay_reach_from_decoded(
    config: &crate::transport::TransportConfig,
) -> Option<crate::stream_info::TransportConfig> {
    use crate::stream_info::{IrohReach, QuicReach, TransportConfig as ReachTransport};
    use crate::transport::EndpointType;
    match &config.endpoint {
        EndpointType::Quic { addr, server_name, auth } => Some(ReachTransport::Quic(QuicReach {
            addr: addr.to_string(),
            server_name: server_name.clone(),
            cert_hashes: auth.accept_cert_hashes().iter().map(|h| h.to_vec()).collect(),
        })),
        EndpointType::Iroh { node_id, relay_url, .. } => {
            // A relay carries the moq stream → moql ALPN; relay_url passes through.
            Some(ReachTransport::Iroh(IrohReach {
                node_id: *node_id,
                alpn: String::from_utf8_lossy(
                    crate::transport::iroh_substrate::ALPN_MOQ_LITE,
                )
                .into_owned(),
                relay_url: relay_url.clone().unwrap_or_default(),
            }))
        }
        EndpointType::Ipc { .. }
        | EndpointType::SystemdFd { .. }
        | EndpointType::Inproc { .. } => None,
    }
}

/// Start a UDS moq server in the background, serving the moq origin's consumer
/// to local cross-process subscribers (e.g. `hyprstream tui attach`).
///
/// Each accepted connection that presents [`crate::transport::uds_session::PLANE_MOQ`]
/// gets a dedicated moq server session via `moq_net::Server::with_publish`.
/// The origin consumer is cloned per session so every subscriber sees the
/// same live broadcast tree.
///
/// Idempotent: a second call is a no-op (the first path wins).
pub fn serve_moq_uds_background(origin: MoqStreamOrigin, path: PathBuf) {
    use crate::transport::uds_session::{accept_uds, PLANE_MOQ};
    use moq_net::Server as MoqServer;

    // Remove stale socket from a previous run (best-effort).
    let _ = std::fs::remove_file(&path);

    // Bind synchronously so the socket exists before we advertise the path.
    // Any caller reading global_moq_uds_path() is guaranteed the socket is ready.
    let listener = match std::os::unix::net::UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(path = %path.display(), "moq UDS bind failed: {e}");
            return;
        }
    };

    // 0o600: owner read/write only; SO_PEERCRED enforces uid match on Linux (see uds_server.rs).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }

    // tokio requires the listener be non-blocking before adoption; otherwise
    // `from_std` panics ("Registering a blocking socket with the tokio runtime
    // is unsupported"). Mirrors the event-bus plane (moq_event.rs).
    if let Err(e) = listener.set_nonblocking(true) {
        tracing::error!("moq UDS set_nonblocking failed: {e}");
        return;
    }

    // Convert to async and publish the path only after the socket is bound.
    let listener = match tokio::net::UnixListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("moq UDS listener conversion failed: {e}");
            return;
        }
    };

    if GLOBAL_MOQ_UDS_PATH.set(path.clone()).is_err() {
        return; // already started (concurrent call)
    }

    tracing::info!(path = %path.display(), "moq UDS listener ready");

    tokio::spawn(async move {
        loop {
            let stream = match listener.accept().await {
                Ok((s, _)) => s,
                Err(e) => {
                    tracing::warn!("moq UDS accept error: {e}");
                    continue;
                }
            };
            let consumer = origin.consumer().clone();
            tokio::spawn(async move {
                let (plane, session) = match accept_uds(stream).await {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::debug!("moq UDS handshake error: {e}");
                        return;
                    }
                };
                if plane != PLANE_MOQ {
                    tracing::debug!("moq UDS: unexpected plane 0x{plane:02x} — dropping");
                    return;
                }
                if let Err(e) = MoqServer::new().with_publish(consumer).accept(session).await {
                    tracing::debug!("moq UDS session ended: {e}");
                }
            });
        }
    });
}

/// Timeout waiting for the moq origin to announce a broadcast at startup.
pub const BROADCAST_ANNOUNCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Timeout between consecutive moq Groups on a subscribed track.
/// A timeout here signals the publisher is gone; the subscriber breaks out.
pub const GROUP_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Timeout reading a single Frame from an already-opened Group.
pub const FRAME_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// The single track name that carries StreamBlock groups for a broadcast.
pub const STREAM_TRACK: &str = "stream";

/// Default `{tenant}/{service}` broadcast-path prefix for the in-process plane.
///
/// The instance segment is appended per-publisher (the opaque topic). Callers
/// that want tenant isolation pass their own prefix to
/// [`MoqStreamOrigin::with_prefix`].
pub const DEFAULT_PREFIX: &str = "local/streams";

/// Shared moq origin for the streaming plane, plus the in-process publish gate.
///
/// Holds the `OriginProducer` (what in-process publishers append into) and the
/// `OriginConsumer` (handed to `moq_net::Server` to serve external subscribers).
/// This is the moq replacement for [`crate::service::StreamService`]'s ZMQ
/// proxy: there is no forwarding loop — producers and consumers share one tree.
#[derive(Clone)]
pub struct MoqStreamOrigin {
    inner: Arc<OriginInner>,
}

/// Builder for [`MoqStreamOrigin`] — set the prefix and publish gate before the
/// shared `Arc` is constructed (avoids clone-on-write of the origin tree).
pub struct MoqStreamOriginBuilder {
    producer: OriginProducer,
    consumer: OriginConsumer,
    prefix: String,
    authorize_signer: Option<Arc<dyn Fn(&[u8; 32]) -> bool + Send + Sync>>,
}

impl MoqStreamOriginBuilder {
    /// Set the `{tenant}/{service}` broadcast-path prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Install the in-process publish gate (mirrors
    /// `StreamService::with_authorize_signer`).
    pub fn with_authorize_signer(
        mut self,
        f: impl Fn(&[u8; 32]) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.authorize_signer = Some(Arc::new(f));
        self
    }

    /// Finish building.
    pub fn build(self) -> MoqStreamOrigin {
        MoqStreamOrigin {
            inner: Arc::new(OriginInner {
                producer: self.producer,
                consumer: self.consumer,
                prefix: self.prefix,
                authorize_signer: self.authorize_signer,
                broadcasts: Mutex::new(HashMap::new()),
            }),
        }
    }
}

struct OriginInner {
    producer: OriginProducer,
    consumer: OriginConsumer,
    /// `{tenant}/{service}` prefix for broadcast paths.
    prefix: String,
    /// Optional in-process publish gate (mirrors `StreamService::authorize_signer`).
    ///
    /// When set, [`MoqStreamOrigin::authorize_signer`] must return `true` before
    /// a publisher is created for a topic. `None` accepts any caller
    /// (testing/bootstrap), matching the ZMQ default.
    authorize_signer: Option<Arc<dyn Fn(&[u8; 32]) -> bool + Send + Sync>>,
    /// Keep broadcast producers alive while their publishers are active.
    ///
    /// Keyed by broadcast path (replace semantics) so a re-announced topic
    /// drops the old `BroadcastProducer` (unannouncing it) rather than
    /// accumulating unboundedly (#164).
    broadcasts: Mutex<HashMap<String, BroadcastProducer>>,
}

impl MoqStreamOrigin {
    /// Begin building from an existing producer/consumer pair (e.g. the one
    /// held by [`crate::transport::iroh_moq::IrohMoqProtocolHandler`]).
    pub fn builder(producer: OriginProducer, consumer: OriginConsumer) -> MoqStreamOriginBuilder {
        MoqStreamOriginBuilder {
            producer,
            consumer,
            prefix: DEFAULT_PREFIX.to_owned(),
            authorize_signer: None,
        }
    }

    /// Build directly from a producer/consumer pair with defaults.
    pub fn from_pair(producer: OriginProducer, consumer: OriginConsumer) -> Self {
        Self::builder(producer, consumer).build()
    }

    /// Begin building a standalone origin with a fresh random id.
    ///
    /// Used when no shared substrate origin is available yet (M2a bootstrap).
    /// Callers that have the substrate's `IrohMoqProtocolHandler` should use
    /// [`Self::builder`] with its `origin_producer()` / `origin_consumer()`
    /// instead, so external subscribers see the same tree.
    pub fn standalone() -> MoqStreamOriginBuilder {
        let producer = moq_net::Origin::random().produce();
        let consumer = producer.consume();
        Self::builder(producer, consumer)
    }

    /// Borrow the consumer (hand this to `moq_net::Server::with_publish`).
    pub fn consumer(&self) -> &OriginConsumer {
        &self.inner.consumer
    }

    /// Borrow the producer.
    pub fn producer(&self) -> &OriginProducer {
        &self.inner.producer
    }

    /// Check the in-process publish gate for a signer pubkey.
    ///
    /// Returns `true` when no gate is installed (bootstrap/testing).
    pub fn authorize_signer(&self, signer: &[u8; 32]) -> bool {
        match &self.inner.authorize_signer {
            Some(f) => f(signer),
            None => true,
        }
    }

    /// Build the broadcast path for an opaque topic: `{prefix}/{topic}`.
    pub fn broadcast_path(&self, topic: &str) -> String {
        format!("{}/{}", self.inner.prefix, topic)
    }

    /// Create an in-process publisher for `ctx`'s topic.
    ///
    /// Creates (or replaces) the broadcast at `{prefix}/{topic}` with a single
    /// [`STREAM_TRACK`] track, and returns a [`MoqStreamPublisher`] whose
    /// chained-HMAC state is seeded from `ctx.mac_key()` / `ctx.topic()` — i.e.
    /// byte-identical to the ZMQ `StreamBuilder`.
    pub fn publisher(&self, ctx: &StreamContext) -> Result<MoqStreamPublisher> {
        self.publisher_with_provenance(ctx, None)
    }

    /// Create an in-process publisher that additionally signs each StreamBlock
    /// with the node's per-host hybrid COSE identity (#321 provenance / C-PROV).
    ///
    /// `provenance = Some(signer)` attaches a `StreamBlock.provenance` signature
    /// so consumers can attribute each block to the producing host (threat T3);
    /// `None` keeps the legacy chained-HMAC-only block.
    pub fn publisher_with_provenance(
        &self,
        ctx: &StreamContext,
        provenance: Option<crate::stream_provenance::ProvenanceSigner>,
    ) -> Result<MoqStreamPublisher> {
        let path = self.broadcast_path(ctx.topic());
        let mut broadcast = self
            .inner
            .producer
            .create_broadcast(path.as_str())
            .ok_or_else(|| anyhow!("create_broadcast denied for {path}"))?;
        let track = broadcast.create_track(Track::new(STREAM_TRACK))?;

        // Retain the broadcast producer so it stays announced for the
        // publisher's lifetime (dropping it would unannounce the broadcast).
        // Replace-semantics: inserting the same path twice drops the old
        // BroadcastProducer rather than accumulating indefinitely (#164).
        self.inner.broadcasts.lock().insert(path, broadcast);

        Ok(MoqStreamPublisher {
            hmac_state: StreamHmacState::new(*ctx.mac_key(), ctx.topic().to_owned()),
            // #321: AEAD enc_key — `Some` only on the DH (mesh) path. `None` on the
            // keyless `StreamContext::new` path (NotificationService topics, whose
            // payloads are already E2E-encrypted), where transport AEAD is skipped.
            enc_key: ctx.enc_key().copied(),
            provenance,
            track,
            next_group: 0,
            cancel_token: ctx.cancel_token().clone(),
            terminated: false,
            topic: ctx.topic().to_owned(),
        })
    }
}

/// In-process moq publisher with the §7.5 chained-HMAC tokenstream.
///
/// API mirrors [`crate::streaming::StreamPublisher`] (`publish_data`,
/// `publish_error`, `complete`, ...) but appends to a moq Track instead of a
/// ZMQ PUSH socket. Each call produces one StreamBlock = one moq Group whose
/// single Frame payload is `capnp_bytes || mac[16]`.
/// NOTE (M2a vs M2b): the StreamBlock *encoding* + HMAC chain are byte-identical
/// to the ZMQ path (shared `encode_stream_block`), but the *batching policy*
/// differs — this emits one payload per block/Group, whereas the ZMQ
/// `StreamBuilder` adaptively batches multiple payloads per block. The MAC
/// chain is valid either way (the verifier is batch-agnostic). Port
/// `BatchingConfig` in M2b for granularity parity.
pub struct MoqStreamPublisher {
    hmac_state: StreamHmacState,
    /// Transport-level AEAD key (#321). `Some` ⇒ each Data/Complete payload is
    /// sealed with AES-256-GCM into a `Tagged` payload before the HMAC chain runs;
    /// `None` ⇒ cleartext (keyless notification path).
    enc_key: Option<[u8; 32]>,
    /// Per-host provenance signer (#321). `Some` ⇒ each StreamBlock carries a
    /// hybrid COSE signature over its canonical signed region.
    provenance: Option<crate::stream_provenance::ProvenanceSigner>,
    track: TrackProducer,
    next_group: u64,
    cancel_token: CancellationToken,
    terminated: bool,
    topic: String,
}

impl MoqStreamPublisher {
    /// Publish one binary payload as a StreamBlock group.
    pub async fn publish_data(&mut self, data: &[u8]) -> Result<()> {
        if self.cancel_token.is_cancelled() {
            anyhow::bail!("stream cancelled");
        }
        self.write_block(&[StreamPayloadData::Data(data.to_vec())])
    }

    /// Publish an error payload (terminal).
    pub async fn publish_error(&mut self, message: &str) -> Result<()> {
        self.terminated = true;
        self.write_block(&[StreamPayloadData::Error(message.to_owned())])
    }

    /// Complete the stream with metadata (terminal).
    pub async fn complete(mut self, metadata: &[u8]) -> Result<()> {
        self.complete_ref(metadata).await
    }

    /// Complete the stream without consuming `self`.
    pub async fn complete_ref(&mut self, metadata: &[u8]) -> Result<()> {
        self.terminated = true;
        self.write_block(&[StreamPayloadData::Complete(metadata.to_vec())])
    }

    /// The opaque topic this publisher serves.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Whether a terminal frame (Error/Complete) was sent.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Whether the stream has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Serialize payloads into a StreamBlock, chain the MAC, and append as a
    /// moq Group (one Frame = `capnp || mac`).
    fn write_block(&mut self, payloads: &[StreamPayloadData]) -> Result<()> {
        // The StreamBlock sequenceNumber (#219) IS the moq Group id — unified so the
        // in-block sequenceNumber the consumer authenticates matches the transport Group.
        // epoch is 0 until the #223 key-epoch lifecycle lands.
        let sequence_number = self.next_group;
        self.next_group += 1;
        let epoch = 0u64;

        // #321: on the mesh/DH path (`enc_key = Some`), seal each Data/Complete
        // payload with AES-256-GCM into a `Tagged` payload BEFORE the HMAC chain
        // runs (so the chain authenticates the ciphertext — no double-encryption,
        // ordering/anti-replay unchanged). Error frames stay cleartext (operational
        // status, terminal). The AEAD AAD/key-commitment are bound to the block's
        // `epoch` (and topic), so a rekey can't replay a block across epochs.
        let sealed: Vec<StreamPayloadData>;
        let payloads: &[StreamPayloadData] = match self.enc_key {
            Some(ref enc_key) => {
                sealed = payloads
                    .iter()
                    .map(|p| seal_payload(enc_key, &self.topic, epoch, p))
                    .collect::<Result<Vec<_>>>()?;
                &sealed
            }
            None => payloads,
        };

        // Canonical signed region (#321): the StreamBlock with provenance EMPTY.
        // This is what the provenance signature covers and what the consumer
        // reconstructs; it also equals the legacy block when provenance is off.
        let signed_region = crate::streaming::encode_stream_block(
            self.hmac_state.prev_mac_bytes(),
            sequence_number,
            epoch,
            payloads,
        )?;

        // #321 provenance: sign the signed region with the host's hybrid identity,
        // then emit the block WITH the provenance field. The HMAC below covers the
        // full wire bytes (incl. provenance); the sig covers the signed region, so
        // verification is a layer on top of HMAC. No provenance ⇒ wire == signed_region.
        let capnp_bytes = match self.provenance {
            Some(ref signer) => {
                let (signer_kid, sig) = signer.sign(&signed_region)?;
                crate::streaming::encode_stream_block_with_provenance(
                    self.hmac_state.prev_mac_bytes(),
                    sequence_number,
                    epoch,
                    payloads,
                    Some((&signer_kid, &sig)),
                )?
            }
            None => signed_region,
        };
        let mac = self.hmac_state.compute_next(&capnp_bytes);

        let mut frame = Vec::with_capacity(capnp_bytes.len() + 16);
        frame.extend_from_slice(&capnp_bytes);
        frame.extend_from_slice(&mac);

        let mut group = self.track.create_group(Group::from(sequence_number))?;
        group.write_frame(Bytes::from(frame))?;
        group.finish()?;
        Ok(())
    }
}

// ============================================================================
// AnyStreamPublisher — moq-lite publish API
// ============================================================================

/// Publisher for the moq-lite streaming plane.
///
/// Type alias for [`MoqStreamPublisher`]; the ZMQ variant was removed in the
/// N4 hard cutover (#138/#213). Retained as an alias so existing call sites
/// compile unchanged.
pub type AnyStreamPublisher = MoqStreamPublisher;

impl MoqStreamPublisher {
    /// Publish binary data with a rate hint (ignored — each call maps 1:1 to one moq Group).
    pub async fn publish_data_with_rate(&mut self, data: &[u8], rate: f32) -> Result<()> {
        let _ = rate;
        self.publish_data(data).await
    }

    /// Publish a progress update (`stage:current:total`).
    pub async fn publish_progress(&mut self, stage: &str, current: usize, total: usize) -> Result<()> {
        let data = format!("{}:{}:{}", stage, current, total);
        self.publish_data(data.as_bytes()).await
    }

    /// Flush — no-op (each block is published immediately).
    pub async fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Non-blocking publish — always succeeds immediately (no HWM concept).
    /// The `rate` hint is ignored.
    pub async fn try_publish_data(&mut self, data: &[u8], rate: f32) -> Result<bool> {
        let _ = rate;
        self.publish_data(data).await.map(|_| true)
    }
}

/// moq stream consumer handle for MCP tool calls and other async consumers.
///
/// Connects to the moq UDS plane, subscribes to the broadcast, verifies the
/// chained-HMAC frames, and delivers [`crate::streaming::StreamPayload`]s via
/// an internal channel. Mirrors the `StreamHandle` interface (same `recv_next()`,
/// `stream_id()`, `cancel_token()`, and `futures::Stream` impl) so callers need
/// no code change beyond the constructor.
pub struct MoqStreamHandle {
    rx: tokio::sync::mpsc::Receiver<anyhow::Result<crate::streaming::StreamPayload>>,
    broadcast_path: String,
    cancel: tokio_util::sync::CancellationToken,
}

impl MoqStreamHandle {
    /// Construct a handle and immediately spawn the background receive task.
    ///
    /// The task connects to the UDS moq server, subscribes to `broadcast_path`,
    /// verifies frames with the chained HMAC (derived from `mac_key` + `topic`),
    /// and forwards payloads to the channel. Errors and end-of-stream close the
    /// channel so `recv_next()` returns `Err` or `Ok(None)` respectively.
    pub fn new(
        uds_path: String,
        broadcast_path: String,
        mac_key: [u8; 32],
        enc_key: [u8; 32],
        topic: String,
    ) -> Self {
        let cancel = tokio_util::sync::CancellationToken::new();
        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<crate::streaming::StreamPayload>>(64);
        tokio::spawn(moq_stream_handle_task(uds_path, broadcast_path.clone(), mac_key, enc_key, topic, tx, cancel.clone()));
        Self { rx, broadcast_path, cancel }
    }

    /// Construct a handle that subscribes over the **network** (#274).
    ///
    /// Resolves the signed `StreamInfo`'s `reach` list and dials the first
    /// dialable network reach via [`crate::dial::dial_stream`] over
    /// `web_transport_quinn` — exactly as the working CLI `quick infer` consumer
    /// does. After connecting, it subscribes to `broadcast_path` and verifies the
    /// chained-HMAC frames exactly as [`Self::new`] does. Errors/end-of-stream
    /// close the channel.
    ///
    /// ## Transport selection (#275 TUI streaming fix)
    ///
    /// The producer's `reach` is the source of truth for where the stream lives.
    /// A stream is a networked address; the subscriber dials the **producer's**
    /// reach. Loopback works same-host (the bound QUIC `/moq` endpoint), proven by
    /// the CLI, so the networked reach is used uniformly whenever the StreamInfo
    /// carries one.
    ///
    /// The local moq UDS plane ([`global_moq_uds_path`]) is used **only as a
    /// fallback** when the StreamInfo carries no dialable reach (UDS-only /
    /// unit-test deployments). It must NOT be preferred when a reach is present:
    /// the local UDS is *this process's own* moq plane, which only carries the
    /// producer's broadcast if the producer is co-located in this very process.
    /// A consumer process that runs its own moq plane for unrelated streams (e.g.
    /// the TUI daemon serving its PTY/shell stdout stream) would otherwise connect
    /// to its own empty plane and time out waiting for the model service's
    /// broadcast that was published on a *different* process's plane (#275).
    ///
    /// ## QoS-driven direct-vs-relay routing (#358)
    ///
    /// `qos` is the service-signed [`crate::stream_info::StreamOpt`]; it selects the
    /// preferred *topology* (direct vs relay) via [`select_reach`] — relay-first for
    /// retained/resumable/fan-out streams (Job/Log), direct-first for live pipes
    /// (`Retention::Live`). This is a stable reorder of the SERVICE-advertised reach
    /// only: the client never invents a reach, so a relay-only (anonymized) stream
    /// stays relay-only. Topology is orthogonal to the delivery/integrity contract.
    pub fn networked(
        reach: Vec<crate::stream_info::Destination>,
        qos: &crate::stream_info::StreamOpt,
        broadcast_path: String,
        mac_key: [u8; 32],
        enc_key: [u8; 32],
        topic: String,
    ) -> Self {
        // QoS-aware topology selection over the SERVICE-advertised reach (server
        // authority preserved: stable reorder, never an invented/forced reach).
        let reach = select_reach(&reach, qos);
        let cancel = tokio_util::sync::CancellationToken::new();
        let (tx, rx) =
            tokio::sync::mpsc::channel::<anyhow::Result<crate::streaming::StreamPayload>>(64);
        // Prefer the producer's networked reach (the StreamInfo source of truth):
        // dial the producer directly, mirroring the CLI. Only fall back to the
        // local moq UDS plane when the StreamInfo carries no dialable reach —
        // never the other way around (see method docs; #275).
        let has_dialable_reach = reach.iter().any(|d| reach_to_transport_config(d).is_some());
        if has_dialable_reach {
            tokio::spawn(moq_stream_handle_task_networked(
                reach,
                broadcast_path.clone(),
                mac_key,
                enc_key,
                topic,
                tx,
                cancel.clone(),
            ));
        } else if let Some(uds) = global_moq_uds_path() {
            let uds_path = uds.to_string_lossy().into_owned();
            tokio::spawn(moq_stream_handle_task(
                uds_path,
                broadcast_path.clone(),
                mac_key,
                enc_key,
                topic,
                tx,
                cancel.clone(),
            ));
        } else {
            // No dialable reach and no local UDS plane: surface a clear error
            // rather than spawning a task that cannot connect.
            tokio::spawn(async move {
                let _ = tx
                    .send(Err(anyhow!(
                        "no dialable reach in StreamInfo and no local moq UDS plane — \
                         cannot subscribe to broadcast"
                    )))
                    .await;
            });
        }
        Self { rx, broadcast_path, cancel }
    }

    /// Receive the next stream payload.
    ///
    /// Returns `Ok(None)` when the stream ends cleanly, `Err` on a stream error.
    pub async fn recv_next(&mut self) -> anyhow::Result<Option<crate::streaming::StreamPayload>> {
        match self.rx.recv().await {
            Some(Ok(p)) => Ok(Some(p)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    /// Returns the moq broadcast path used as stream identifier.
    pub fn stream_id(&self) -> &str {
        &self.broadcast_path
    }

    /// Returns a cancellation token that aborts the background receive task.
    pub fn cancel_token(&self) -> &tokio_util::sync::CancellationToken {
        &self.cancel
    }

    /// Cancel the background receive task immediately.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl futures::Stream for MoqStreamHandle {
    type Item = anyhow::Result<crate::streaming::StreamPayload>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

// TODO: add reconnect backoff on UDS disconnect (ZMQ auto-reconnect parity)
#[allow(clippy::too_many_arguments)]
async fn moq_stream_handle_task(
    uds_path: String,
    broadcast_path: String,
    mac_key: [u8; 32],
    enc_key: [u8; 32],
    topic: String,
    tx: tokio::sync::mpsc::Sender<anyhow::Result<crate::streaming::StreamPayload>>,
    cancel: tokio_util::sync::CancellationToken,
) {
    use crate::streaming::StreamVerifier;
    use crate::transport::uds_session::{connect_uds, PLANE_MOQ};
    use moq_net::{Client as MoqClient, Origin, Track};

    let session = match connect_uds(&uds_path, PLANE_MOQ).await {
        Ok(s) => s,
        Err(e) => { let _ = tx.send(Err(anyhow!("moq UDS connect {uds_path}: {e}"))).await; return; }
    };
    let client_origin = Origin::random().produce();
    let client_consumer = client_origin.consume();
    let moq_client = MoqClient::new().with_consume(client_origin);
    let _session = match moq_client.connect(session).await {
        Ok(s) => s,
        Err(e) => { let _ = tx.send(Err(anyhow!("moq handshake: {e}"))).await; return; }
    };
    let bc = match tokio::time::timeout(
        BROADCAST_ANNOUNCE_TIMEOUT,
        client_consumer.announced_broadcast(&broadcast_path),
    ).await {
        Ok(Some(bc)) => bc,
        Ok(None) => {
            let _ = tx.send(Err(anyhow!("broadcast {broadcast_path} not announced"))).await;
            return;
        }
        Err(_) => {
            let _ = tx.send(Err(anyhow!("timeout waiting for broadcast {broadcast_path}"))).await;
            return;
        }
    };
    let track = match bc.subscribe_track(&Track::new(STREAM_TRACK)) {
        Ok(t) => t,
        Err(e) => { let _ = tx.send(Err(anyhow!("subscribe_track: {e}"))).await; return; }
    };
    // #321: AEAD ON for this DH-keyed mesh stream — open sealed Tagged blocks.
    let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
    // #145: read groups by EXACT sequence (get_group), not arrival-order next_group.
    // Each Group is served on its own QUIC uni-stream, so Groups can arrive out of
    // order; next_group's monotonic cursor returns the first Group with sequence >=
    // its cursor in *arrival* order, so a lower-seq Group that arrives after a higher
    // one (e.g. the small terminal / always-retained max_sequence Group) is skipped —
    // fatally breaking the ordered, gap-fatal chained HMAC. get_group(seq) waits for
    // each exact Group (even out-of-order), guaranteeing in-order, gap-free delivery.
    let mut expected_seq = 0u64;
    loop {
        if cancel.is_cancelled() {
            break;
        }
        let mut group = match tokio::time::timeout(GROUP_IDLE_TIMEOUT, track.get_group(expected_seq)).await {
            Ok(Ok(Some(g))) => g,
            Ok(Ok(None)) => break, // track ended cleanly
            Err(_elapsed) => {
                let _ = tx.send(Err(anyhow!(
                    "stream idle: no group for {}s",
                    GROUP_IDLE_TIMEOUT.as_secs()
                ))).await;
                break;
            }
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("moq next_group: {e}"))).await;
                break;
            }
        };
        expected_seq += 1;
        let frame: bytes::Bytes = match tokio::time::timeout(FRAME_READ_TIMEOUT, group.read_frame()).await {
            Ok(Ok(Some(f))) => f,
            Ok(Ok(None)) => break, // group ended without a frame
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("frame read error: {e}"))).await;
                break;
            }
            Err(_elapsed) => {
                let _ = tx.send(Err(anyhow!(
                    "frame read timeout after {}s",
                    FRAME_READ_TIMEOUT.as_secs()
                ))).await;
                break;
            }
        };
        match verify_moq_frame(&mut verifier, &topic, &frame) {
            Ok(payloads) => {
                for p in payloads {
                    let is_terminal = matches!(
                        p,
                        crate::streaming::StreamPayload::Complete(_)
                            | crate::streaming::StreamPayload::Error(_)
                    );
                    if tx.send(Ok(p)).await.is_err() {
                        return;
                    }
                    if is_terminal {
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        }
    }
}

/// Convert a wire-published [`crate::stream_info::Destination`] into the local
/// [`crate::transport::TransportConfig`] [`crate::dial::dial_stream`] dials (#274).
///
/// Only the network-routable transports map; same-host endpoints are never
/// carried in `reach` (co-located clients use the UDS fast path).
pub fn reach_to_transport_config(
    reach: &crate::stream_info::Destination,
) -> Option<crate::transport::TransportConfig> {
    wire_transport_to_dial(&reach.transport)
}

/// The single wire→dial reach codec (#274/#320): map one wire
/// [`crate::stream_info::TransportConfig`] union arm to the local dialable
/// [`crate::transport::TransportConfig`]. Returns `None` for an un-routable arm
/// (e.g. an iroh reach with no relay). Shared by the streaming plane
/// ([`reach_to_transport_config`]) and the RPC inference router (#320), so there
/// is exactly ONE iroh/quic reach decoding.
pub fn wire_transport_to_dial(
    wire: &crate::stream_info::TransportConfig,
) -> Option<crate::transport::TransportConfig> {
    use crate::stream_info::TransportConfig as ReachTransport;
    use crate::transport::{QuicServerAuth, TransportConfig};
    match wire {
        ReachTransport::Quic(q) => {
            let addr: std::net::SocketAddr = q.addr.parse().ok()?;
            // Fixed-size cert pins (SHA-256 = 32 bytes); skip any malformed entry.
            let hashes: Vec<[u8; 32]> = q
                .cert_hashes
                .iter()
                .filter_map(|h| <[u8; 32]>::try_from(h.as_slice()).ok())
                .collect();
            let auth = if hashes.is_empty() {
                QuicServerAuth::web_pki()
            } else {
                // Pinned self-signed mesh (matches the DID-doc #quic entry).
                QuicServerAuth::pinned(hashes).ok()?
            };
            Some(TransportConfig::quic_with_auth(addr, q.server_name.clone(), auth))
        }
        // #320/#357: dial the wire-advertised iroh reach by node_id. iroh binds
        // the dialed `moql` connection to this `EndpointId` (its Ed25519 pubkey);
        // `dial_stream`'s iroh arm (#282/S2) consumes it and opens the `moql`
        // session. When `relayUrl` is empty the shared client endpoint's pkarr /
        // n0 DNS discovery (`presets::N0`) resolves the routable addresses, so a
        // native peer can dial by node_id alone (no direct addrs are carried on
        // the wire — see `IrohReach` in streaming.capnp).
        ReachTransport::Iroh(i) => {
            let relay_url = if i.relay_url.is_empty() { None } else { Some(i.relay_url.clone()) };
            Some(TransportConfig::iroh(i.node_id, Vec::new(), relay_url))
        }
    }
}

/// The single dial→wire reach codec (#320): map a local dialable
/// [`crate::transport::TransportConfig`] to a wire-publishable
/// [`crate::stream_info::TransportConfig`] union arm.
///
/// Returns `None` for same-host endpoints (`Inproc`/`Ipc`/`SystemdFd`) — these
/// are NEVER advertised on the wire (a remote caller could not dial them; a
/// co-located caller resolves them from local config). So a co-located-only
/// service yields an empty published reach list. The inverse of
/// [`wire_transport_to_dial`]; both live here as the one reach codec.
pub fn dial_transport_to_wire(
    dial: &crate::transport::TransportConfig,
) -> Option<crate::stream_info::TransportConfig> {
    use crate::stream_info::{IrohReach, QuicReach, TransportConfig as ReachTransport};
    use crate::transport::EndpointType;
    match &dial.endpoint {
        EndpointType::Quic { addr, server_name, auth } => Some(ReachTransport::Quic(QuicReach {
            addr: addr.to_string(),
            server_name: server_name.clone(),
            cert_hashes: auth.accept_cert_hashes().iter().map(|h| h.to_vec()).collect(),
        })),
        // The wire iroh reach carries the carrier address (nodeId) + relay only; direct
        // addrs are not published (privacy + iroh discovery), matching the
        // DID-doc `IrohTransport` entry shape (#280/#282).
        EndpointType::Iroh { node_id, relay_url, .. } => Some(ReachTransport::Iroh(IrohReach {
            node_id: *node_id,
            alpn: String::from_utf8_lossy(
                crate::transport::iroh_substrate::ALPN_HYPRSTREAM_RPC,
            )
            .into_owned(),
            relay_url: relay_url.clone().unwrap_or_default(),
        })),
        // Same-host endpoints are never wire-advertised (#320).
        EndpointType::Inproc { .. }
        | EndpointType::Ipc { .. }
        | EndpointType::SystemdFd { .. } => None,
    }
}

/// A live moq subscriber connection resolved from a producer's reach (#356).
///
/// Holds the `OriginConsumer` callers `announced_broadcast` + `subscribe_track`
/// against, plus the underlying `moq_net::Session` which MUST be kept alive for
/// the duration of the subscription (dropping it tears the transport down).
pub struct MoqReachConnection {
    /// The consumer side of the client origin the producer's broadcast is
    /// announced into. Subscribe to the producer's `broadcast_path` on this.
    pub consumer: OriginConsumer,
    /// The live moq session. Held only to keep the transport open; not used
    /// directly by callers, but must not be dropped before the consumer.
    _session: moq_net::Session,
}

/// Resolve a producer's reach into a live moq subscriber connection (#356).
///
/// This is the **single** reach→connection resolver shared by every networked
/// subscriber (the inference [`MoqStreamHandle::networked`] task and the CLI
/// model-load / `notify subscribe` consumers). It enforces one transport policy
/// in one place:
///
///   1. **Networked first** — dial the first dialable `Destination` in `reach`
///      (the producer's wire-advertised QUIC/`/moq` endpoint) via
///      [`crate::dial::dial_stream`]. This is the source of truth: the stream
///      lives wherever the *producer* advertised, so cross-process / cross-
///      instance subscribers reach it (fixes #142 + the TUI cross-process bug).
///   2. **Local UDS fallback** — only when `reach` carries NO dialable network
///      reach (UDS-only / unit-test deployments) do we connect to *this
///      process's* local moq UDS plane ([`global_moq_uds_path`]). The UDS path is
///      resolved from LOCAL knowledge, never from the wire — a same-host fast
///      path, never advertised.
///   3. **Fail closed** — if neither is available, return an error rather than
///      connecting to an empty/wrong plane and timing out.
///
/// The UDS fast path is NEVER preferred over a present networked reach: the local
/// UDS plane only carries the producer's broadcast when the producer is co-located
/// in this very process, so preferring it silently breaks cross-process delivery.
pub async fn connect_moq_reach(
    reach: &[crate::stream_info::Destination],
) -> Result<MoqReachConnection> {
    use crate::transport::uds_session::{connect_uds, PLANE_MOQ};
    use moq_net::{Client as MoqClient, Origin};

    let client_origin = Origin::random().produce();
    let consumer = client_origin.consume();
    let moq_client = MoqClient::new().with_consume(client_origin);

    // 1. Networked reach (source of truth): dial the first reach we can resolve.
    let mut last_err: Option<String> = None;
    for dest in reach {
        let Some(cfg) = reach_to_transport_config(dest) else {
            continue;
        };
        match crate::dial::dial_stream(&cfg).await {
            Ok(stream_session) => match stream_session.connect_moq(&moq_client).await {
                Ok(session) => return Ok(MoqReachConnection { consumer, _session: session }),
                Err(e) => last_err = Some(format!("moq handshake: {e}")),
            },
            Err(e) => last_err = Some(e.to_string()),
        }
    }

    // 2. Local UDS fallback (same-host fast path, resolved from LOCAL config).
    if let Some(uds) = global_moq_uds_path() {
        let session = connect_uds(uds, PLANE_MOQ)
            .await
            .with_context(|| format!("moq UDS connect {}", uds.display()))?;
        let session = moq_client
            .connect(session)
            .await
            .map_err(|e| anyhow!("moq UDS handshake: {e}"))?;
        return Ok(MoqReachConnection { consumer, _session: session });
    }

    // 3. Fail closed.
    Err(anyhow!(
        "no dialable reach in StreamInfo and no local moq UDS plane — cannot subscribe to broadcast{}",
        last_err.map(|e| format!(" (last dial error: {e})")).unwrap_or_default()
    ))
}

// ============================================================================
// QoS-driven reach selection (#358) — topology, orthogonal to delivery/integrity
//
// Topology (direct vs relay) is NOT a new QoS integrity axis: the delivery /
// ordering / completion / retention contract (`StreamOpt`) stays orthogonal to
// where the bytes flow. The selector only REORDERS the reaches the SERVICE
// advertised so the preferred topology is tried first; `connect_moq_reach`'s
// in-order try-then-fallback loop then provides automatic reachability fallback
// (direct → relay, or relay → direct) within the advertised set.
//
// **Server-authority invariant (enforced here):** selection is a stable reorder
// of `advertised` — it never invents a reach, never fabricates a `direct` reach
// the service didn't publish, and never drops a reach. So a service that
// published a stream relay-ONLY (omitting any direct reach to keep the producer
// anonymized) stays relay-only: there is no direct reach to promote, and the
// client cannot force one. The client READS qos/reach; it never writes them.
// ============================================================================

/// Reorder the service-advertised `reach` list by the QoS-preferred topology
/// (#358), returning a new list the caller hands to [`connect_moq_reach`].
///
/// - `Retention::Live` (Pipe / live console, lowest-latency) → **direct-first**:
///   a live pipe wants the shortest path; the relay is the fallback only if the
///   producer is not directly reachable.
/// - retained / resumable / fan-out (Job / Log, `Retention::{Blocks,Seconds}`)
///   → **relay-first**: the relay is the natural late-join / retained / fan-out
///   surface, and lets a subscriber rendezvous without dialing the producer.
///
/// The reorder is **stable** (it preserves the service's relative ordering within
/// each role), so direct reaches keep their advertised priority (e.g. iroh before
/// Quic) and the only change is whether the relay group floats to the front.
pub fn select_reach(
    advertised: &[crate::stream_info::Destination],
    qos: &crate::stream_info::StreamOpt,
) -> Vec<crate::stream_info::Destination> {
    use crate::stream_info::{Retention, Role};
    let relay_first = !matches!(qos.retention, Retention::Live);
    let mut ordered: Vec<crate::stream_info::Destination> = Vec::with_capacity(advertised.len());
    // Stable partition: pull the preferred role to the front, keep within-role order.
    let prefer = if relay_first { Role::Relay } else { Role::Direct };
    ordered.extend(advertised.iter().filter(|d| d.role == prefer).cloned());
    ordered.extend(advertised.iter().filter(|d| d.role != prefer).cloned());
    ordered
}

/// Resolve a service-advertised reach into a live moq connection, honouring the
/// QoS-preferred topology (#358).
///
/// Thin composition of [`select_reach`] (QoS reorder, server-authority-safe) and
/// [`connect_moq_reach`] (in-order dial with automatic direct↔relay fallback).
/// This is the single entry point networked subscribers should use when they
/// have the signed `StreamOpt` in hand, so direct-vs-relay routing lives in ONE
/// place and the reach the service published is the only thing ever dialed.
pub async fn connect_moq_reach_for_qos(
    reach: &[crate::stream_info::Destination],
    qos: &crate::stream_info::StreamOpt,
) -> Result<MoqReachConnection> {
    let ordered = select_reach(reach, qos);
    connect_moq_reach(&ordered).await
}

// ============================================================================
// moq relay client (#358) — producer-side announce UP to the relay.
//
// Restores the rendezvous property the ZMQ→moq migration removed: a producing
// node that has a configured relay opens a moq link to it and announces its
// local origin's broadcasts UP to the relay (`with_origin`), so a subscriber
// that dials the relay sees the SAME broadcast by the SAME `broadcastPath`
// without ever dialing the producer. Mirrors the event-plane UDS link
// ([`connect_event_moq_uds_background`] / [`run_event_client_link`]) but points
// at a REMOTE relay via [`crate::dial::dial_stream`] instead of `connect_uds`.
//
// Relay-blind: the link carries the AEAD-sealed, chained-HMAC frames opaquely
// (`Bytes`); the relay holds no `enc_key` / `mac_key` and never decrypts.
// ============================================================================

/// Base reconnect delay for the producer→relay link; the backoff grows
/// exponentially from this up to [`RELAY_RECONNECT_MAX`], with jitter.
const RELAY_RECONNECT_BASE: std::time::Duration = std::time::Duration::from_millis(500);
/// Cap on the exponential reconnect backoff.
const RELAY_RECONNECT_MAX: std::time::Duration = std::time::Duration::from_secs(30);
/// After this many consecutive failures, escalate the reconnect log from
/// `debug!` to `warn!` so a persistently-unreachable relay is observable.
const RELAY_RECONNECT_WARN_AFTER: u32 = 5;

/// Fail-closed relay path-integrity gate (#504 item 3) — assert the dial is
/// pinned to the explicitly configured carrier endpoint before this node
/// announces its origin UP to it (`with_origin`).
///
/// ## The gap this closes
/// [`run_relay_announce_link`] dials the producer-chosen relay and announces this
/// node's broadcasts (track names / `broadcastPath`, traffic patterns) UP to it.
/// Frames are AEAD-sealed so **content** confidentiality holds regardless, but the
/// announce-handshake exposes broadcast **metadata** to whatever endpoint answers.
/// moq-net's session handshake exposes **no relay-role / announce-capability**
/// negotiation (the setup parameters are opaque `Bytes` and `Client` offers no
/// role hook). Relay-role authorization is an upstream configuration/policy
/// decision. This function only prevents path substitution by checking the
/// configured cert pin or iroh `EndpointId`; it grants no role or identity.
///
/// ## The check (fail-closed)
/// - **iroh**: the connection is bound to the configured carrier `EndpointId`,
///   so target/path substitution is rejected. No application identity follows.
/// - **QUIC, leaf-pinned** (`accept_cert_hashes` non-empty, with or without
///   WebPKI): the dialed peer must present a pinned leaf — accepted.
/// - **QUIC, WebPKI-only** (`require_web_pki` with an EMPTY pin set): **rejected.**
///   WebPKI proves only "some cert valid for this SNI", so any holder of a
///   CA-valid cert for that name could be substituted as the relay and silently
///   receive our broadcast announcements. We will not announce our origin UP to
///   an endpoint that is not pinned to the configured relay target.
/// - **same-host (`Ipc`/`Inproc`/`SystemdFd`)**: not reachable here
///   ([`reach_to_transport_config`] yields `None` for them), but rejected
///   defensively — they carry no network target pin at this seam.
///
/// Returns `Err` (and the caller skips `with_origin`) when target/path integrity
/// cannot be confirmed, rather than leaking announcements to a substituted
/// endpoint. Endpoint equality never authorizes application identity.
fn assert_relay_path_pinned(cfg: &crate::transport::TransportConfig) -> Result<()> {
    use crate::transport::EndpointType;
    match &cfg.endpoint {
        // Iroh pins the dial path to the configured EndpointId.
        EndpointType::Iroh { .. } => Ok(()),
        // QUIC target/path integrity requires a leaf-cert pin.
        EndpointType::Quic { auth, .. } => {
            if auth.accept_cert_hashes().is_empty() {
                Err(anyhow!(
                    "relay endpoint is WebPKI-only (no leaf-cert pin): cannot confirm it is the \
                     configured relay target — refusing to announce origin UP (fail-closed, #504)"
                ))
            } else {
                Ok(())
            }
        }
        other => Err(anyhow!(
            "relay endpoint {other:?} carries no network target pin — \
             refusing to announce origin UP (fail-closed, #504)"
        )),
    }
}

/// Announce this node's streaming origin UP to the producer-chosen relay (#358).
///
/// Spawns a background task that dials the relay via [`crate::dial::dial_stream`],
/// runs the moq handshake with `with_origin(producer)` (so the producer's
/// broadcasts are announced UP to the relay and re-served to its subscribers),
/// holds the session open, and reconnects on failure. The task runs for the
/// process lifetime.
///
/// Called once by the producing service's factory after [`init_global_relay_reach`]
/// — typically with `global_moq_origin().producer()`. No-op-safe to omit when the
/// node has no relay (direct-only deployments).
pub fn serve_origin_to_relay_background(
    producer: OriginProducer,
    relay: crate::stream_info::TransportConfig,
) {
    tokio::spawn(async move {
        let mut failures: u32 = 0;
        loop {
            match run_relay_announce_link(&producer, &relay).await {
                // A link was established and the session closed; reset the backoff.
                Ok(()) => failures = 0,
                Err(e) => {
                    failures = failures.saturating_add(1);
                    if failures >= RELAY_RECONNECT_WARN_AFTER {
                        tracing::warn!(
                            failures,
                            "moq relay announce link failing repeatedly: {e}; relay may be unreachable"
                        );
                    } else {
                        tracing::debug!("moq relay announce link ended: {e}; reconnecting");
                    }
                }
            }
            // Exponential backoff with equal jitter, capped — de-correlates many
            // producers reconnecting to a shared relay (avoids a thundering-herd /
            // self-DoS against the federation anchor). `failures == 0` still waits
            // ~base, so an instantly-closing session can't spin a tight loop.
            let shift = failures.min(6); // 2^6 * 500ms = 32s, clamped by MAX below
            let backoff = RELAY_RECONNECT_BASE
                .checked_mul(1u32 << shift)
                .unwrap_or(RELAY_RECONNECT_MAX)
                .min(RELAY_RECONNECT_MAX);
            let half = backoff / 2;
            let jitter = if half.is_zero() {
                std::time::Duration::ZERO
            } else {
                std::time::Duration::from_nanos(rand::random::<u64>() % half.as_nanos() as u64)
            };
            tokio::time::sleep(half + jitter).await;
        }
    });
}

/// One connect-and-hold cycle of the producer→relay announce link (#358).
///
/// Dials the relay, runs the moq handshake bidirectionally (`with_origin`), and
/// returns when the session closes so the caller can reconnect. The relay
/// ingests the announced broadcasts and re-serves them to its subscribers by
/// track name — the producer never learns who subscribes, the subscriber never
/// dials the producer.
///
/// Exposed (rather than only via [`serve_origin_to_relay_background`]) so a
/// caller that owns its own reconnect/lifecycle — and the rendezvous test — can
/// drive a single connect cycle directly.
pub async fn run_relay_announce_link(
    producer: &OriginProducer,
    relay: &crate::stream_info::TransportConfig,
) -> Result<()> {
    use moq_net::Client as MoqClient;

    // Reuse the ONE wire-reach → dial-config resolver so the relay is dialed by
    // the same code path (Quic / WebTransport / iroh) every direct reach uses.
    let dest = crate::stream_info::Destination { role: crate::stream_info::Role::Relay, transport: relay.clone() };
    let cfg = reach_to_transport_config(&dest)
        .ok_or_else(|| anyhow!("relay reach is not a dialable network transport"))?;

    // #504 item 3 — relay-capability gate (fail-closed): do NOT announce this
    // node's origin (broadcast track names / `broadcastPath`, traffic patterns)
    // UP to an endpoint whose path does not match the configured relay target.
    // moq exposes no relay-role handshake; role authorization is upstream, while
    // this seam checks only the cert-hash pin / iroh EndpointId. A WebPKI-only
    // (unpinned) or identity-less endpoint is refused here — surfaced loudly and
    // the announce is skipped — rather than leaking announcements to a
    // misconfigured / substituted relay. See [`assert_relay_path_pinned`].
    if let Err(e) = assert_relay_path_pinned(&cfg) {
        tracing::warn!("moq relay announce gate: {e}");
        return Err(e);
    }

    let stream_session = crate::dial::dial_stream(&cfg).await?;
    // `with_origin` makes the link bidirectional: this node's broadcasts are
    // announced UP to the relay; the relay re-serves them to its subscribers.
    let moq_client = MoqClient::new().with_origin(producer.clone());
    let moq_session = stream_session
        .connect_moq(&moq_client)
        .await
        .map_err(|e| anyhow!("relay moq handshake: {e}"))?;

    tracing::info!("moq relay announce link established");
    let reason = moq_session.closed().await;
    Err(anyhow!("moq relay session closed: {reason:?}"))
}

/// Networked variant of [`moq_stream_handle_task`]: dial a `/moq`
/// `web_transport_quinn` session from the `reach` list instead of the UDS
/// plane, then run the identical subscribe + chained-HMAC verify loop (#274).
#[allow(clippy::too_many_arguments)]
async fn moq_stream_handle_task_networked(
    reach: Vec<crate::stream_info::Destination>,
    broadcast_path: String,
    mac_key: [u8; 32],
    enc_key: [u8; 32],
    topic: String,
    tx: tokio::sync::mpsc::Sender<anyhow::Result<crate::streaming::StreamPayload>>,
    cancel: tokio_util::sync::CancellationToken,
) {
    use crate::streaming::StreamVerifier;
    use moq_net::Track;

    // Resolve the producer's reach into a live moq connection via the single
    // shared resolver (networked-first, local-UDS fallback, fail-closed; #356).
    let conn = match connect_moq_reach(&reach).await {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(Err(anyhow!("moq networked dial failed: {e}"))).await;
            return;
        }
    };
    // Borrow the consumer from `conn`; `conn` (and its session) stays alive for
    // the whole subscribe loop.
    let client_consumer = &conn.consumer;
    let bc = match tokio::time::timeout(
        BROADCAST_ANNOUNCE_TIMEOUT,
        client_consumer.announced_broadcast(&broadcast_path),
    )
    .await
    {
        Ok(Some(bc)) => bc,
        Ok(None) => {
            let _ = tx.send(Err(anyhow!("broadcast {broadcast_path} not announced"))).await;
            return;
        }
        Err(_) => {
            let _ = tx
                .send(Err(anyhow!("timeout waiting for broadcast {broadcast_path}")))
                .await;
            return;
        }
    };
    let track = match bc.subscribe_track(&Track::new(STREAM_TRACK)) {
        Ok(t) => t,
        Err(e) => {
            let _ = tx.send(Err(anyhow!("subscribe_track: {e}"))).await;
            return;
        }
    };
    // #321: AEAD ON for this DH-keyed mesh stream — open sealed Tagged blocks.
    let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
    // #145: read groups by EXACT sequence (get_group), not arrival-order next_group.
    // Each Group is served on its own QUIC uni-stream, so Groups can arrive out of
    // order; next_group's monotonic cursor returns the first Group with sequence >=
    // its cursor in *arrival* order, so a lower-seq Group that arrives after a higher
    // one (e.g. the small terminal / always-retained max_sequence Group) is skipped —
    // fatally breaking the ordered, gap-fatal chained HMAC. get_group(seq) waits for
    // each exact Group (even out-of-order), guaranteeing in-order, gap-free delivery.
    let mut expected_seq = 0u64;
    loop {
        if cancel.is_cancelled() {
            break;
        }
        let mut group = match tokio::time::timeout(GROUP_IDLE_TIMEOUT, track.get_group(expected_seq)).await {
            Ok(Ok(Some(g))) => g,
            Ok(Ok(None)) => break,
            Err(_elapsed) => {
                let _ = tx
                    .send(Err(anyhow!("stream idle: no group for {}s", GROUP_IDLE_TIMEOUT.as_secs())))
                    .await;
                break;
            }
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("moq next_group: {e}"))).await;
                break;
            }
        };
        expected_seq += 1;
        let frame: bytes::Bytes = match tokio::time::timeout(FRAME_READ_TIMEOUT, group.read_frame()).await {
            Ok(Ok(Some(f))) => f,
            Ok(Ok(None)) => break,
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("frame read error: {e}"))).await;
                break;
            }
            Err(_elapsed) => {
                let _ = tx
                    .send(Err(anyhow!("frame read timeout after {}s", FRAME_READ_TIMEOUT.as_secs())))
                    .await;
                break;
            }
        };
        match verify_moq_frame(&mut verifier, &topic, &frame) {
            Ok(payloads) => {
                for p in payloads {
                    let is_terminal = matches!(
                        p,
                        crate::streaming::StreamPayload::Complete(_)
                            | crate::streaming::StreamPayload::Error(_)
                    );
                    if tx.send(Ok(p)).await.is_err() {
                        return;
                    }
                    if is_terminal {
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        }
    }
}

/// Consumer-side helper: split a moq Frame payload back into the ZMQ-style
/// `[topic, capnp, mac]` frames expected by [`StreamVerifier::verify`], then
/// verify and parse it.
///
/// `topic` is the opaque topic (not the broadcast path) and must match the
/// verifier's topic, exactly as the ZMQ frame-0 contract required.
pub fn verify_moq_frame(
    verifier: &mut StreamVerifier,
    topic: &str,
    frame: &[u8],
) -> Result<Vec<crate::streaming::StreamPayload>> {
    if frame.len() < 16 {
        anyhow::bail!("moq frame too short: {} bytes", frame.len());
    }
    let split = frame.len() - 16;
    // Zero-copy: pass slices into the received frame (`Bytes` from moq-net) straight
    // to the verifier — no per-frame `Vec` allocation of the (potentially large) payload.
    verifier.verify_parts(topic.as_bytes(), &frame[..split], &frame[split..])
}

/// Mesh consumer entry point (#321): verify a moq frame's chained HMAC + AEAD
/// (via the `verifier`) AND its per-host provenance signature against a roster.
///
/// `roster` resolves a signer's anchored ML-DSA-65 key; `is_enrolled` confirms the
/// signer is a known mesh peer. Provenance is REQUIRED (fail-closed): a block with
/// no/invalid/unknown signer is rejected. Use [`verify_moq_frame`] for the
/// non-mesh client path that has no roster.
pub fn verify_moq_frame_with_provenance(
    verifier: &mut StreamVerifier,
    topic: &str,
    frame: &[u8],
    roster: &dyn crate::envelope::PqTrustStore,
    is_enrolled: &dyn Fn(&[u8; 32]) -> bool,
) -> Result<Vec<crate::streaming::StreamPayload>> {
    if frame.len() < 16 {
        anyhow::bail!("moq frame too short: {} bytes", frame.len());
    }
    let split = frame.len() - 16;
    let capnp_data = &frame[..split];

    // 1) Chained HMAC + AEAD open (also rejects topic/order/MAC failures).
    let payloads = verifier.verify_parts(topic.as_bytes(), capnp_data, &frame[split..])?;

    // 2) Per-host provenance, layered on top: parse the block, reconstruct the
    //    provenance-cleared signed region, and verify the signature + roster.
    let mut slice: &[u8] = capnp_data;
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut slice,
        capnp::message::ReaderOptions::default(),
    )?;
    let block = reader.get_root::<crate::streaming_capnp::stream_block::Reader>()?;
    let prov = block.get_provenance()?;
    let signer_kid = prov.get_signer_kid()?;
    let sig = prov.get_sig()?;
    let signed_region = crate::stream_provenance::signed_region_from_block(&block)?;
    crate::stream_provenance::verify_provenance(
        signer_kid,
        sig,
        &signed_region,
        roster,
        is_enrolled,
    )?;

    Ok(payloads)
}

/// Seal a single Data/Complete payload into an AES-256-GCM `Tagged` payload
/// (#321). The 1-byte kind tag is prepended to the plaintext so the consumer can
/// restore the original variant. Error/Tagged/other variants pass through
/// unchanged (Error is operational, already-Tagged is the E2E notification path).
///
/// Shares its AAD + kind-tag framing with the cross-target open path
/// ([`crate::stream_consumer::open_sealed_payload`]).
fn seal_payload(
    enc_key: &[u8; 32],
    topic: &str,
    epoch: u64,
    payload: &StreamPayloadData,
) -> Result<StreamPayloadData> {
    use crate::crypto::event_crypto::{encrypt_event, EventPrivacy};
    use crate::stream_consumer::{stream_aead_aad, SEALED_KIND_COMPLETE, SEALED_KIND_DATA};

    let (kind, body): (u8, &[u8]) = match payload {
        StreamPayloadData::Data(d) => (SEALED_KIND_DATA, d),
        StreamPayloadData::Complete(d) => (SEALED_KIND_COMPLETE, d),
        // Leave non-sealed variants untouched.
        other => return Ok(other.clone()),
    };

    let mut plaintext = Vec::with_capacity(1 + body.len());
    plaintext.push(kind);
    plaintext.extend_from_slice(body);

    let aad = stream_aead_aad(topic, epoch);
    let (tag, ciphertext, nonce, key_commitment) =
        encrypt_event(enc_key, &aad, &plaintext, EventPrivacy::ZeroKnowledge)
            .map_err(|e| anyhow!("stream AEAD seal failed: {e}"))?;

    Ok(StreamPayloadData::Tagged {
        tag,
        payload: ciphertext,
        nonce: nonce.to_vec(),
        key_commitment: key_commitment.to_vec(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::streaming::StreamPayload;
    use moq_net::Origin;

    fn origin() -> MoqStreamOrigin {
        let producer = Origin::random().produce();
        let consumer = producer.consume();
        MoqStreamOrigin::from_pair(producer, consumer)
    }

    /// In-process publish → in-process consume over the *same* origin (the data
    /// an external moq subscriber sees on the wire), verifying the chained-HMAC.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moq_stream_round_trip() -> Result<()> {
        let origin = origin();
        let (_client_secret, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
        let topic = ctx.topic().to_owned();
        let mac_key = *ctx.mac_key();
        // #321: DH path is AEAD-on; the verifier shares the same enc_key.
        let enc_key = *ctx.enc_key().expect("DH ctx has enc_key");

        let mut pub_ = origin.publisher(&ctx)?;
        pub_.publish_data(b"hello").await?;
        pub_.publish_data(b"world").await?;
        pub_.complete_ref(b"{}").await?;

        // Consume from the shared origin (same bytes a wire subscriber reads).
        let path = origin.broadcast_path(&topic);
        let bc = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin.consumer().announced_broadcast(path.as_str()),
        )
        .await?
        .ok_or_else(|| anyhow!("broadcast not announced"))?;
        let mut track = bc.subscribe_track(&Track::new(STREAM_TRACK))?;

        let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        let mut got: Vec<StreamPayload> = Vec::new();
        for _ in 0..3 {
            let mut group = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                track.next_group(),
            )
            .await??
            .ok_or_else(|| anyhow!("next_group None"))?;
            let frame = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                group.read_frame(),
            )
            .await??
            .ok_or_else(|| anyhow!("read_frame None"))?;
            got.extend(verify_moq_frame(&mut verifier, &topic, &frame)?);
        }

        assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"hello"));
        assert!(matches!(&got[1], StreamPayload::Data(d) if d == b"world"));
        assert!(matches!(&got[2], StreamPayload::Complete(_)));
        Ok(())
    }

    /// `AnyStreamPublisher` round-trip: publish via the type alias and verify
    /// the same bytes arrive on the moq consumer side.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn any_stream_publisher_moq_round_trip() -> Result<()> {
        let origin = origin();
        let (_client_secret, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
        let topic = ctx.topic().to_owned();
        let mac_key = *ctx.mac_key();
        // #321: DH path is AEAD-on; the verifier shares the same enc_key.
        let enc_key = *ctx.enc_key().expect("DH ctx has enc_key");

        let mut any_pub: AnyStreamPublisher = origin.publisher(&ctx)?;
        any_pub.publish_data(b"ping").await?;
        any_pub.complete_ref(b"{}").await?;

        // Consume and verify
        let path = origin.broadcast_path(&topic);
        let bc = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin.consumer().announced_broadcast(path.as_str()),
        )
        .await?
        .ok_or_else(|| anyhow!("broadcast not announced"))?;
        let mut track = bc.subscribe_track(&Track::new(STREAM_TRACK))?;
        let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        let mut got: Vec<StreamPayload> = Vec::new();
        for _ in 0..2 {
            let mut group = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                track.next_group(),
            )
            .await??
            .ok_or_else(|| anyhow!("next_group None"))?;
            let frame = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                group.read_frame(),
            )
            .await??
            .ok_or_else(|| anyhow!("read_frame None"))?;
            got.extend(verify_moq_frame(&mut verifier, &topic, &frame)?);
        }
        assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"ping"));
        assert!(matches!(&got[1], StreamPayload::Complete(_)));
        Ok(())
    }

    /// #320: the iroh reach codec round-trips carrier address (nodeId) + relay;
    /// an empty-relay reach is decoded but is not dialable
    /// (dial/dial_stream fail-fast on it — pkarr-only is deferred to #282).
    #[test]
    fn iroh_reach_dial_wire_roundtrip() {
        use crate::transport::{EndpointType, TransportConfig};
        let node_id = [0x42u8; 32];
        let dial = TransportConfig::iroh(node_id, Vec::new(), Some("https://relay.example".to_owned()));
        let wire = dial_transport_to_wire(&dial).expect("iroh dial → wire");
        match &wire {
            crate::stream_info::TransportConfig::Iroh(i) => {
                assert_eq!(i.node_id, node_id);
                assert_eq!(i.relay_url, "https://relay.example");
                assert_eq!(i.alpn, "hyprstream-rpc/1");
            }
            other => panic!("expected wire Iroh, got {other:?}"),
        }
        let back = wire_transport_to_dial(&wire).expect("iroh wire → dial");
        match back.endpoint {
            EndpointType::Iroh { node_id: n, relay_url, direct_addrs } => {
                assert_eq!(n, node_id);
                assert_eq!(relay_url.as_deref(), Some("https://relay.example"));
                assert!(direct_addrs.is_empty(), "wire iroh reach never carries direct addrs");
            }
            other => panic!("expected dial Iroh, got {other:?}"),
        }
    }

    /// #320: same-host endpoints (Inproc/Ipc) are NEVER advertised on the wire —
    /// `dial_transport_to_wire` returns None (a co-located-only service yields an
    /// empty published reach list).
    #[test]
    fn same_host_endpoints_never_wire_advertised() {
        use crate::transport::TransportConfig;
        assert!(dial_transport_to_wire(&TransportConfig::inproc("hyprstream/inference-x")).is_none());
        assert!(dial_transport_to_wire(&TransportConfig::ipc("/tmp/x.sock")).is_none());
    }

    /// #320: a wire iroh reach with an empty relayUrl preserves its carrier address
    /// but is not dialable — `dial` fails fast rather than hang in discovery.
    #[test]
    fn iroh_reach_empty_relay_decodes_but_not_dialable() {
        use crate::stream_info::{IrohReach, TransportConfig as ReachTransport};
        let wire = ReachTransport::Iroh(IrohReach {
            node_id: [9u8; 32],
            alpn: "moql".to_owned(),
            relay_url: String::new(),
        });
        let dial = wire_transport_to_dial(&wire).expect("decodes");
        // No relay + no direct addrs ⇒ dial() fail-fast.
        let signer = crate::signer::LocalSigner::new(crate::crypto::SigningKey::generate(&mut rand::rngs::OsRng));
        assert!(crate::dial::dial(&dial, signer, None, None).is_err(), "unreachable iroh reach must not dial");
    }

    /// #275: a StreamInfo carrying a dialable Quic reach must be classified as
    /// dialable, so `networked()` routes to the producer's networked reach
    /// (`dial_stream`) rather than this process's local moq UDS plane.
    #[test]
    fn quic_reach_is_dialable() {
        use crate::stream_info::{Destination, QuicReach, Role, TransportConfig as ReachTransport};
        let reach = Destination {
            role: Role::Direct,
            transport: ReachTransport::Quic(QuicReach {
                addr: "127.0.0.1:4433".to_owned(),
                server_name: "localhost".to_owned(),
                cert_hashes: vec![vec![0u8; 32]],
            }),
        };
        assert!(
            reach_to_transport_config(&reach).is_some(),
            "a Quic reach must resolve to a dialable TransportConfig (selects the \
             networked dial_stream path, not the local UDS)"
        );
    }

    /// #357: an advertised iroh reach (carrying only the producer's node_id) must
    /// resolve to a dialable `EndpointType::Iroh { node_id, .. }` — the dial-by-
    /// node_id-alone path `dial_stream` opens over the `moql` ALPN (pkarr / n0 DNS
    /// discovery resolves the addresses on the shared client endpoint). This is
    /// the encode→resolve round-trip for the iroh-direct reach.
    #[test]
    fn iroh_reach_resolves_to_node_id() {
        use crate::stream_info::{Destination, IrohReach, Role, TransportConfig as ReachTransport};
        use crate::transport::EndpointType;

        let node_id = [0x7u8; 32];
        // This is exactly the Destination `producer_reach()` emits for the iroh arm.
        let reach = Destination {
            role: Role::Direct,
            transport: ReachTransport::Iroh(IrohReach {
                node_id,
                alpn: "moql".to_owned(),
                relay_url: String::new(),
            }),
        };
        assert_eq!(reach.role, Role::Direct, "iroh reach is a direct producer reach");

        let cfg = reach_to_transport_config(&reach)
            .expect("an iroh reach with a node_id must resolve to a dialable TransportConfig");
        match cfg.endpoint {
            EndpointType::Iroh { node_id: got, direct_addrs, relay_url } => {
                assert_eq!(got, node_id, "resolved node_id must round-trip the advertised one");
                // S2 advertises node_id alone; discovery supplies reachability.
                assert!(direct_addrs.is_empty(), "S2 iroh reach carries no direct addrs (pkarr)");
                assert!(relay_url.is_none(), "S2 iroh reach carries no relay URL (pkarr)");
            }
            other => panic!("iroh reach must resolve to EndpointType::Iroh, got {other:?}"),
        }
    }

    /// #357: an iroh-only reach list must be classified as dialable, so a
    /// native peer routes to the producer's iroh reach rather than falling
    /// through to the local UDS plane (the source-of-truth ordering S1 established).
    #[test]
    fn iroh_reach_is_dialable() {
        use crate::stream_info::{Destination, IrohReach, Role, TransportConfig as ReachTransport};
        let reach = [Destination {
            role: Role::Direct,
            transport: ReachTransport::Iroh(IrohReach {
                node_id: [0xABu8; 32],
                alpn: "moql".to_owned(),
                relay_url: String::new(),
            }),
        }];
        assert!(
            reach.iter().any(|d| reach_to_transport_config(d).is_some()),
            "an iroh reach must resolve to a dialable TransportConfig"
        );
    }

    /// #275: the consumer dials the producer's networked reach even when this
    /// process also serves its own local moq UDS plane. Regression for the TUI
    /// timeout: the local UDS must NOT shadow a present reach (it is this
    /// process's plane, not the producer's).
    ///
    /// We assert the branch selection deterministically. With a dialable reach
    /// present we set the local UDS path to a socket that does NOT exist: the
    /// **UDS branch** would fail in milliseconds with a connect error naming that
    /// path, whereas the **networked branch** dials QUIC (which, to a closed
    /// loopback port, keeps retrying past a short deadline). So: a UDS-path error
    /// before the deadline ⇒ the pre-fix bug (UDS preferred — FAIL); reaching the
    /// deadline still trying ⇒ the networked dial was taken (PASS). A networked
    /// dial error is also an acceptable PASS (reach was dialed).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn networked_prefers_reach_over_local_uds() -> Result<()> {
        use crate::stream_info::{Destination, QuicReach, Role, TransportConfig as ReachTransport};

        // Pretend this process has a local moq UDS plane (the TUI daemon does, to
        // serve its PTY stdout). The pre-fix code short-circuited onto this path
        // and ignored the reach; the fix must dial the reach instead. The path is
        // distinctive so a UDS connect error is unambiguously identifiable.
        let uds_marker = "/nonexistent/hyprstream-275-uds-marker.sock";
        let _ = GLOBAL_MOQ_UDS_PATH.set(PathBuf::from(uds_marker));

        // A dialable Quic reach to a closed loopback port.
        let reach = vec![Destination {
            role: Role::Direct,
            transport: ReachTransport::Quic(QuicReach {
                addr: "127.0.0.1:1".to_owned(),
                server_name: "localhost".to_owned(),
                cert_hashes: vec![[0xABu8; 32].to_vec()],
            }),
        }];

        let qos = crate::stream_info::StreamOpt::default();
        let mut handle = MoqStreamHandle::networked(
            reach,
            &qos,
            "local/streams/test/deadbeef".to_owned(),
            [0u8; 32],
            [0u8; 32],
            "deadbeef".repeat(8),
        );

        match tokio::time::timeout(std::time::Duration::from_secs(3), handle.recv_next()).await {
            // Still dialing QUIC at the deadline — the networked branch was taken.
            Err(_elapsed) => Ok(()),
            // An early error: it must be a *networked* dial failure, NOT a UDS
            // connect to our marker path (which would prove the bug).
            Ok(res) => {
                let err = res.expect_err("dial to a closed port must not yield a payload");
                let msg = err.to_string();
                assert!(
                    !msg.contains(uds_marker),
                    "consumer connected to the local UDS plane instead of dialing the \
                     producer's reach (#275 regression): {msg}"
                );
                assert!(
                    msg.contains("networked dial"),
                    "expected a networked dial failure (reach was dialed), got: {msg}"
                );
                Ok(())
            }
        }
    }

    #[test]
    fn authorize_signer_gate() {
        let producer = Origin::random().produce();
        let consumer = producer.consume();
        let gated = MoqStreamOrigin::builder(producer, consumer)
            .with_authorize_signer(|pk| pk[0] == 1)
            .build();
        assert!(gated.authorize_signer(&[1u8; 32]));
        assert!(!gated.authorize_signer(&[2u8; 32]));
        // No gate -> accept all.
        assert!(origin().authorize_signer(&[9u8; 32]));
    }

    // ── #358 relay reach / selection unit tests ─────────────────────────────

    use crate::stream_info::{
        Destination, IrohReach, Job, Log, Pipe, QuicReach, Role, StreamOpt, StreamOptPreset,
        TransportConfig as ReachTransport,
    };

    fn direct_iroh(b: u8) -> Destination {
        Destination {
            role: Role::Direct,
            transport: ReachTransport::Iroh(IrohReach {
                node_id: [b; 32],
                alpn: String::new(),
                relay_url: String::new(),
            }),
        }
    }
    fn relay_quic(addr: &str) -> Destination {
        Destination {
            role: Role::Relay,
            transport: ReachTransport::Quic(QuicReach {
                addr: addr.to_owned(),
                server_name: "relay".to_owned(),
                cert_hashes: vec![vec![0u8; 32]],
            }),
        }
    }

    // ── #504 item 3: configured relay target/path pin gate ──────────────────

    /// An iroh relay dial is pinned to the configured EndpointId, so path
    /// substitution is rejected without treating equality as identity authority.
    #[test]
    fn relay_gate_accepts_iroh_target_pinned_without_granting_identity() {
        // Equality or inequality with any application key cannot change this
        // path-only result: both values are opaque carrier targets here.
        for node_id in [[7u8; 32], [19u8; 32]] {
            let cfg = crate::transport::TransportConfig::iroh(node_id, Vec::new(), None);
            assert!(assert_relay_path_pinned(&cfg).is_ok());
        }
    }

    /// A leaf-cert-pinned QUIC relay has a confirmed target path — accepted.
    #[test]
    fn relay_gate_accepts_quic_leaf_pinned() {
        let addr = "127.0.0.1:7777".parse().expect("addr");
        let cfg = crate::transport::TransportConfig::quic_pinned(addr, "relay", [3u8; 32]);
        assert!(assert_relay_path_pinned(&cfg).is_ok());
    }

    /// A WebPKI-only QUIC relay (no leaf pin) proves only "some cert valid for
    /// this SNI" — any CA-valid cert holder could be substituted, so the gate
    /// REJECTS it fail-closed and the origin is never announced UP.
    #[test]
    fn relay_gate_rejects_quic_webpki_only() {
        let addr = "127.0.0.1:7777".parse().expect("addr");
        let cfg = crate::transport::TransportConfig::quic(addr, "relay");
        let err = assert_relay_path_pinned(&cfg).expect_err("WebPKI-only relay must be rejected");
        assert!(
            err.to_string().contains("WebPKI-only"),
            "expected a WebPKI-only rejection, got: {err}"
        );
    }

    /// A WebPKI+leaf-pin QUIC relay (defence in depth) is still leaf-pinned —
    /// accepted.
    #[test]
    fn relay_gate_accepts_quic_webpki_plus_pin() {
        let addr = "127.0.0.1:7777".parse().expect("addr");
        let auth = crate::transport::QuicServerAuth::web_pki_pinned(vec![[5u8; 32]]).expect("auth");
        let cfg = crate::transport::TransportConfig::quic_with_auth(addr, "relay", auth);
        assert!(assert_relay_path_pinned(&cfg).is_ok());
    }

    /// A same-host endpoint carries no network target pin — rejected (defensive;
    /// `reach_to_transport_config` already yields None for these).
    #[test]
    fn relay_gate_rejects_unpinned_same_host() {
        let cfg = crate::transport::TransportConfig::inproc("hyprstream/x");
        assert!(assert_relay_path_pinned(&cfg).is_err());
    }

    /// A [`ProducerReachConfig`] with a server-global relay advertises a
    /// `Role::Relay` Destination — populating the existing schema, no new field.
    ///
    /// No `OnceLock` dance (#384): the config is per-instance, so this is
    /// deterministic regardless of process-global state or test ordering.
    #[test]
    fn producer_reach_advertises_relay_when_configured() {
        let relay = ReachTransport::Quic(QuicReach {
            addr: "10.0.0.9:4433".to_owned(),
            server_name: "pds".to_owned(),
            cert_hashes: vec![vec![1u8; 32]],
        });
        let cfg = ProducerReachConfig { relay: Some(relay.clone()), ..Default::default() };
        let reach = cfg.reach();
        assert!(
            reach.iter().any(|d| d.role == Role::Relay && d.transport == relay),
            "ProducerReachConfig::reach must advertise the configured Role::Relay reach"
        );
    }

    /// #384 regression: two DIFFERENT reach contexts in ONE process produce
    /// DIFFERENT reach — a relay-only-anonymized stream X coexists with a
    /// direct stream Y. This is the heterogeneous-anonymization property the
    /// `OnceLock` singletons structurally prevented (first-write-wins clobber).
    #[test]
    fn heterogeneous_reach_in_one_process() {
        let relay_x = ReachTransport::Quic(QuicReach {
            addr: "10.0.0.9:4433".to_owned(),
            server_name: "relay-x".to_owned(),
            cert_hashes: vec![vec![7u8; 32]],
        });
        // A server with BOTH a direct (iroh) reach and a server-global relay.
        let cfg = ProducerReachConfig {
            iroh_node_id: Some([3u8; 32]),
            quic_reach: None,
            relay: Some(relay_x.clone()),
        };

        // Stream Y: server default → direct (iroh) THEN relay, direct-first order.
        let reach_y = cfg.reach();
        assert_eq!(reach_y[0].role, Role::Direct, "Y must advertise its direct reach first");
        assert!(
            reach_y.iter().any(|d| d.role == Role::Relay),
            "Y must also advertise the server-global relay"
        );

        // Stream X: relay-ONLY (anonymized) override → NO direct reach, only the
        // per-stream relay — same process, same config instance, different result.
        let relay_only = ReachTransport::Quic(QuicReach {
            addr: "10.0.0.50:4433".to_owned(),
            server_name: "relay-anon".to_owned(),
            cert_hashes: vec![vec![9u8; 32]],
        });
        let reach_x = cfg.reach_with_relay(RelayChoice::Only(relay_only.clone()));
        assert_eq!(reach_x.len(), 1, "relay-only stream X must omit all direct reaches");
        assert_eq!(reach_x[0].role, Role::Relay, "X is relay-only (anonymized)");
        assert_eq!(reach_x[0].transport, relay_only, "X uses its per-stream relay override");
        assert!(
            !reach_x.iter().any(|d| d.role == Role::Direct),
            "X must NOT leak a direct reach (server authority / anonymization)"
        );

        // The two streams genuinely differ in the same process — the property the
        // singletons precluded.
        assert_ne!(reach_x, reach_y, "X (relay-only) and Y (direct+relay) must differ");
    }

    /// Per-stream relay `Override` swaps the relay but KEEPS the direct reaches
    /// (per-tenant isolation without anonymization), preserving direct-first order.
    #[test]
    fn per_stream_relay_override_keeps_direct() {
        let server_relay = ReachTransport::Quic(QuicReach {
            addr: "10.0.0.1:4433".to_owned(),
            server_name: "server".to_owned(),
            cert_hashes: vec![vec![1u8; 32]],
        });
        let tenant_relay = ReachTransport::Quic(QuicReach {
            addr: "10.0.0.2:4433".to_owned(),
            server_name: "tenant".to_owned(),
            cert_hashes: vec![vec![2u8; 32]],
        });
        let cfg = ProducerReachConfig {
            iroh_node_id: Some([4u8; 32]),
            quic_reach: None,
            relay: Some(server_relay.clone()),
        };
        let reach = cfg.reach_with_relay(RelayChoice::Override(tenant_relay.clone()));
        assert_eq!(reach[0].role, Role::Direct, "override keeps the direct reach, listed first");
        let relay_d = reach.iter().find(|d| d.role == Role::Relay).expect("relay advertised");
        assert_eq!(relay_d.transport, tenant_relay, "per-stream override replaces the server relay");
    }

    /// QoS-driven selection: live pipes prefer DIRECT, retained/fan-out prefer RELAY.
    #[test]
    fn select_reach_orders_by_qos_topology() {
        let advertised = vec![direct_iroh(1), relay_quic("10.0.0.1:4433"), direct_iroh(2)];

        // Pipe (Retention::Live) → direct-first; within-role order preserved.
        let pipe = Pipe::stream_opt();
        let ordered = select_reach(&advertised, &pipe);
        assert_eq!(ordered[0].role, Role::Direct, "live pipe must try direct first");
        assert_eq!(ordered[2].role, Role::Relay, "relay falls to the back for a live pipe");
        assert_eq!(ordered[0], direct_iroh(1), "within-role advertised order preserved");
        assert_eq!(ordered[1], direct_iroh(2));

        // Job (retained/resumable) → relay-first.
        let job = Job::stream_opt();
        let ordered = select_reach(&advertised, &job);
        assert_eq!(ordered[0].role, Role::Relay, "retained job must try relay first");

        // Log (retained) → relay-first.
        let log = Log::stream_opt();
        assert_eq!(select_reach(&advertised, &log)[0].role, Role::Relay);
    }

    /// Server authority: selection NEVER invents, fabricates `direct`, or drops a
    /// reach. A relay-ONLY advertisement (anonymized stream) stays relay-only even
    /// for a live-pipe QoS that would *prefer* direct — there is nothing to promote.
    #[test]
    fn select_reach_preserves_server_authority() {
        let relay_only = vec![relay_quic("10.0.0.1:4433")];

        // Even with default (Live → direct-preferred) qos, no direct reach appears.
        let ordered = select_reach(&relay_only, &StreamOpt::default());
        assert_eq!(ordered.len(), 1, "selection must not add or drop reaches");
        assert_eq!(ordered[0].role, Role::Relay, "a relay-only stream stays relay-only");
        assert!(
            !ordered.iter().any(|d| d.role == Role::Direct),
            "the client must NOT fabricate a direct reach the service didn't advertise"
        );

        // It is also a pure permutation: same multiset in, same multiset out.
        let advertised = vec![direct_iroh(1), relay_quic("a:1"), direct_iroh(2)];
        let mut a = advertised.clone();
        let mut b = select_reach(&advertised, &Job::stream_opt());
        a.sort_by_key(|d| (d.role == Role::Relay, format!("{d:?}")));
        b.sort_by_key(|d| (d.role == Role::Relay, format!("{d:?}")));
        assert_eq!(a, b, "selection must be a permutation of the advertised reach");
    }

    /// `relay_reach_from_decoded` round-trips a DID-decoded transport into the
    /// wire-reach form advertised as the relay — the shared codec, no drift.
    #[test]
    fn relay_reach_from_decoded_round_trips() {
        // iroh decoded config → iroh wire reach carrying the same node_id.
        let cfg = crate::transport::TransportConfig::iroh([0x5u8; 32], Vec::new(), None);
        match relay_reach_from_decoded(&cfg) {
            Some(ReachTransport::Iroh(i)) => assert_eq!(i.node_id, [0x5u8; 32]),
            other => panic!("expected iroh wire reach, got {other:?}"),
        }
        // Same-host transports are never relay endpoints.
        assert!(
            relay_reach_from_decoded(&crate::transport::TransportConfig::ipc("/tmp/x.sock")).is_none(),
            "a same-host ipc transport is not a network-routable relay"
        );
    }

    // ── #321 AEAD seal/open ────────────────────────────────────────────────

    #[test]
    fn aead_seal_open_roundtrip() {
        let enc_key = [0x42u8; 32];
        let topic = "deadbeef";
        let epoch = 7u64;
        for payload in [
            StreamPayloadData::Data(b"hello tokens".to_vec()),
            StreamPayloadData::Complete(b"{\"done\":true}".to_vec()),
        ] {
            let sealed = seal_payload(&enc_key, topic, epoch, &payload).unwrap();
            let StreamPayloadData::Tagged { tag, payload: ct, nonce, key_commitment } = &sealed
            else {
                panic!("seal must produce a Tagged payload");
            };
            let opened = crate::stream_consumer::open_sealed_payload(
                &enc_key, topic, epoch, tag, ct, nonce, key_commitment,
            )
            .unwrap();
            match (&payload, &opened) {
                (StreamPayloadData::Data(a), StreamPayload::Data(b))
                | (StreamPayloadData::Complete(a), StreamPayload::Complete(b)) => {
                    assert_eq!(a, b);
                }
                _ => panic!("opened variant must match sealed variant"),
            }
        }
    }

    #[test]
    fn aead_open_rejects_wrong_key_and_tamper() {
        let enc_key = [0x42u8; 32];
        let topic = "t";
        let epoch = 1u64;
        let sealed =
            seal_payload(&enc_key, topic, epoch, &StreamPayloadData::Data(b"secret".to_vec()))
                .unwrap();
        let StreamPayloadData::Tagged { tag, payload: ct, nonce, key_commitment } = &sealed else {
            panic!("expected Tagged");
        };

        // Wrong key.
        let wrong = [0x99u8; 32];
        assert!(crate::stream_consumer::open_sealed_payload(
            &wrong, topic, epoch, tag, ct, nonce, key_commitment
        )
        .is_err());

        // Wrong epoch (AAD mismatch) — anti-replay across epochs.
        assert!(crate::stream_consumer::open_sealed_payload(
            &enc_key, topic, 2, tag, ct, nonce, key_commitment
        )
        .is_err());

        // Tampered ciphertext.
        let mut bad_ct = ct.clone();
        if let Some(b) = bad_ct.first_mut() {
            *b ^= 0xFF;
        }
        assert!(crate::stream_consumer::open_sealed_payload(
            &enc_key, topic, epoch, tag, &bad_ct, nonce, key_commitment
        )
        .is_err());
    }

    // ── #321 provenance over a published block ─────────────────────────────

    /// Publish a block with provenance ON, then verify HMAC + AEAD + provenance
    /// against a roster; assert wrong/unknown signers are rejected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provenance_publish_verify_and_reject() -> Result<()> {
        use crate::envelope::KeyedPqTrustStore;
        use crate::stream_provenance::ProvenanceSigner;
        use ed25519_dalek::SigningKey;

        let origin = origin();
        let (_cs, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
        let topic = ctx.topic().to_owned();
        let mac_key = *ctx.mac_key();
        let enc_key = *ctx.enc_key().expect("DH ctx has enc_key");

        let host_ed = SigningKey::from_bytes(&[5u8; 32]);
        let signer = ProvenanceSigner::from_ed25519(host_ed.clone());
        let kid = signer.signer_kid();

        let mut pub_ = origin.publisher_with_provenance(&ctx, Some(signer))?;
        pub_.publish_data(b"hi").await?;
        pub_.complete_ref(b"{}").await?;

        // Drain the two frames from the shared origin.
        let path = origin.broadcast_path(&topic);
        let bc = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin.consumer().announced_broadcast(path.as_str()),
        )
        .await?
        .ok_or_else(|| anyhow!("broadcast not announced"))?;
        let mut track = bc.subscribe_track(&Track::new(STREAM_TRACK))?;
        let mut frames: Vec<bytes::Bytes> = Vec::new();
        for _ in 0..2 {
            let mut group = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                track.next_group(),
            )
            .await??
            .ok_or_else(|| anyhow!("next_group None"))?;
            let frame = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                group.read_frame(),
            )
            .await??
            .ok_or_else(|| anyhow!("read_frame None"))?;
            frames.push(frame);
        }

        // Roster anchoring the host's mesh ML-DSA key + enrolled-set closure.
        use ml_dsa::Keypair;
        let mut roster = KeyedPqTrustStore::new();
        roster.bind(kid, &crate::node_identity::derive_mesh_mldsa_key(&host_ed).verifying_key());
        let enrolled = |k: &[u8; 32]| *k == kid;

        // Valid: verify HMAC + AEAD + provenance.
        let mut v = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        let got = verify_moq_frame_with_provenance(&mut v, &topic, &frames[0], &roster, &enrolled)?;
        assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"hi"));

        // Unknown signer: empty roster / not enrolled → reject (re-verify frame 0
        // with a fresh verifier so the HMAC chain restarts at the same block).
        let empty = KeyedPqTrustStore::new();
        let none_enrolled = |_: &[u8; 32]| false;
        let mut v2 = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        assert!(verify_moq_frame_with_provenance(
            &mut v2, &topic, &frames[0], &empty, &none_enrolled
        )
        .is_err());
        Ok(())
    }
}
