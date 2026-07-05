//! The `dial()` factory: the single place transport choice is made.
//!
//! Today transport selection leaks into generated client code, which hardcodes
//! `ZmqConnection::new(...)`. Per the A1 addressing spike, that decision belongs
//! in exactly one place: [`dial`] takes a resolved [`TransportConfig`] and
//! returns a ready [`Arc<dyn RpcClient>`], erasing the concrete transport behind
//! the object-safe client trait (`Transport` itself is not object-safe — it has
//! `Sub`/`Pub` associated types — so erasure happens at the `RpcClient` layer).
//!
//! # Inproc dial table
//!
//! `inproc://` names resolve through a process-local registry mapping a name to
//! a co-located service's [`IrohRequestProcessor`]. The registry — not the
//! [`TransportConfig`] — holds the live handle: a `TransportConfig` is
//! `Clone + Eq` and wire-publishable (DiscoveryService), so an `Arc<dyn
//! IrohRequestProcessor>` cannot live inside it. Naming (resolver →
//! `TransportConfig`) and handles (registry → processor) stay separate; `dial()`
//! is where they meet.
//!
//! # Construction is synchronous; transports connect lazily
//!
//! `dial()` is sync and does no I/O — the inproc arm only looks up an existing
//! processor. Networked transports (quinn/iroh/moq) connect lazily on first
//! `send()` (cached like `ZmqConnection`), so dialing never blocks and the
//! `inventory`-registered sync factory pattern is preserved. Those arms land in
//! a follow-up increment of #151(a); ZMQ ipc/systemd endpoints stay on the
//! existing codegen path during the transition.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, Weak};

use anyhow::{anyhow, bail, Result};
use parking_lot::RwLock;

use crate::crypto::VerifyingKey;
use crate::rpc_client::{RpcClient, RpcClientImpl};
use crate::transport::in_memory::InMemoryTransport;
use crate::transport::rpc_session::IrohRequestProcessor;
use crate::transport::{EndpointType, TransportConfig};
use crate::transport_traits::Signer;

/// Process-local map of inproc endpoint name → co-located request processor.
///
/// Entries are `Weak`: the registry is a *lookup index*, not the owner. The
/// service spawn site retains the strong `Arc` for the service's lifetime, so
/// dropping the service (e.g. on shutdown) automatically tears down its bridge
/// thread AND leaves a dead `Weak` here that self-evicts on the next lookup —
/// no leak-by-forgotten-unregister, and no strong ref pinning a bridge thread
/// past shutdown.
type InprocRegistry = RwLock<HashMap<String, Weak<dyn IrohRequestProcessor>>>;

static INPROC_REGISTRY: OnceLock<InprocRegistry> = OnceLock::new();

fn registry() -> &'static InprocRegistry {
    INPROC_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a co-located service's request processor under an `inproc://` name.
///
/// Called at service spawn. The `name` is the endpoint without the scheme
/// (e.g. `hyprstream/registry` for `inproc://hyprstream/registry`). The caller
/// MUST retain `processor` (the strong `Arc`) for as long as the service should
/// be dialable — the registry only holds a `Weak`.
///
/// Overwriting a name whose service is still live is almost always a bug
/// (name-squatting / double-spawn); it is logged loudly. Existing dialed
/// clients keep the processor they captured; only future dials see the new one.
pub fn register_inproc(name: impl Into<String>, processor: &Arc<dyn IrohRequestProcessor>) {
    let name = name.into();
    let mut map = registry().write();
    if map.get(&name).is_some_and(|w| w.strong_count() > 0) {
        tracing::warn!(
            endpoint = %name,
            "register_inproc: overwriting a still-live in-process service registration"
        );
    }
    map.insert(name, Arc::downgrade(processor));
}

/// Explicitly drop a name's registration (best-effort; dead entries also
/// self-evict on lookup once the service's strong `Arc` is gone).
pub fn unregister_inproc(name: &str) {
    registry().write().remove(name);
}

/// Look up a co-located service's processor by inproc name, upgrading the
/// `Weak`. A stale (dead-service) entry is pruned in passing.
pub fn lookup_inproc(name: &str) -> Option<Arc<dyn IrohRequestProcessor>> {
    let mut map = registry().write();
    match map.get(name).and_then(Weak::upgrade) {
        Some(arc) => Some(arc),
        None => {
            // Present-but-dead → evict; absent → no-op.
            map.remove(name);
            None
        }
    }
}

/// Dial a resolved [`TransportConfig`], returning a ready RPC client.
///
/// `server_verifying_key` is the destination's response-verification key.
/// `None` does NOT disable signature verification — the response is still
/// cryptographically verified against the key embedded in its envelope; `None`
/// only declines to pin *which* identity that key must be. Passing `None` is
/// only sound when the transport itself authenticates the peer (e.g. pinned
/// TLS / QUIC cert).
///
/// **For `inproc://` there is no transport-level peer authentication** (it is a
/// function call into a registry-resolved processor), so `None` is *discouraged*
/// on the inproc path: without it, a name-squatting registration could be dialed
/// without detection. Callers SHOULD pass the resolved service verifying key for
/// inproc. (The codegen wire-up will thread the resolver-supplied key through;
/// see #151(a) follow-up.)
pub fn dial<S>(
    target: &TransportConfig,
    signer: S,
    server_verifying_key: Option<VerifyingKey>,
    token: Option<String>,
) -> Result<Arc<dyn RpcClient>>
where
    S: Signer + 'static,
{
    /// Wrap a built transport as an `RpcClient`, applying the optional default
    /// JWT (CA-signed trust cert included in request envelopes) if present.
    fn build_client<S2, T2>(
        signer: S2,
        transport: T2,
        vk: Option<VerifyingKey>,
        token: Option<String>,
    ) -> Arc<dyn RpcClient>
    where
        S2: Signer + 'static,
        T2: crate::transport_traits::Transport + 'static,
    {
        let rpc = RpcClientImpl::new(signer, transport, vk);
        let rpc = match token {
            Some(t) => rpc.with_default_jwt(t),
            None => rpc,
        };
        Arc::new(rpc) as Arc<dyn RpcClient>
    }

    // Matched exhaustively on purpose: this is the one place transport choice is
    // made, so a newly-added EndpointType variant MUST be a compile error here
    // rather than silently falling through to a runtime bail.
    match &target.endpoint {
        EndpointType::Inproc { endpoint } => {
            let processor = lookup_inproc(endpoint).ok_or_else(|| {
                anyhow!("no in-process service registered for inproc endpoint '{endpoint}'")
            })?;
            let transport = InMemoryTransport::new(processor);
            Ok(build_client(signer, transport, server_verifying_key, token))
        }
        EndpointType::Quic { addr, server_name, auth } => {
            // SECURITY (#185): QUIC channel auth (WebPKI / cert-hash pin) binds the
            // *channel*, not the peer's DID identity. Identity is established at the
            // application layer — the response signature is verified against the
            // envelope-embedded key even when `server_verifying_key` is `None`,
            // just not *pinned* to an expected identity. For a networked peer,
            // prefer passing the resolver-known key; note (debug) when we can't.
            // (iroh's arm needs no such note — it is identity-bound at the
            // transport, RFC 7250.)
            if server_verifying_key.is_none() {
                tracing::debug!(
                    %addr,
                    "dial: networked QUIC peer with no expected verifying key — \
                     response identity unpinned (channel-level auth only, #185)"
                );
            }
            let transport = crate::transport::lazy_quinn::LazyQuinnTransport::new(
                *addr,
                server_name.clone(),
                auth.clone(),
            );
            Ok(build_client(signer, transport, server_verifying_key, token))
        }
        EndpointType::Iroh { direct_addrs, relay_url, .. }
            if direct_addrs.is_empty() && relay_url.is_none() =>
        {
            // Fail fast rather than hand iroh an EndpointId with no reachability,
            // which would fall through to discovery and time out (~10-30s). The
            // resolver is expected to supply at least one direct addr or a relay.
            bail!(
                "dial(): iroh endpoint has neither direct addrs nor a relay URL — \
                 not dialable; the resolver must supply reachability"
            )
        }
        EndpointType::Iroh { node_id, direct_addrs, relay_url } => {
            // iroh binds the connection to the peer's EndpointId (its pubkey),
            // so the transport authenticates the peer *identity* — a `None`
            // server_verifying_key is sound (response sig still verified, and
            // the channel is identity-bound, unlike QUIC's cert pin).
            let transport = crate::transport::lazy_iroh::LazyIrohTransport::new(
                *node_id,
                direct_addrs.clone(),
                relay_url.clone(),
            );
            Ok(build_client(signer, transport, server_verifying_key, token))
        }
        // Same-host `ipc` plane: connect a UdsSession (RPC plane) at the socket
        // path. systemd socket-activation is the same client-side dial — the fd
        // is the *server's* pre-bound listener; clients connect by `client_path`.
        // UDS has no transport-level peer identity; the app-layer SignedEnvelope
        // is the authentication (a `None` server_verifying_key leaves the
        // response identity unpinned but still signature-verified). Socket perms
        // + SO_PEERCRED are daemon-owned defense-in-depth (#207).
        EndpointType::Ipc { path } => {
            let transport = crate::transport::lazy_uds::LazyUdsTransport::new(path.clone());
            Ok(build_client(signer, transport, server_verifying_key, token))
        }
        EndpointType::SystemdFd { client_path, .. } => {
            let transport = crate::transport::lazy_uds::LazyUdsTransport::new(client_path.clone());
            Ok(build_client(signer, transport, server_verifying_key, token))
        }
    }
}

/// The URL path the daemon's `web_transport_quinn` server dispatches to the moq
/// streaming plane (default path → RPC). Both server and client use this.
pub const MOQ_PATH: &str = "/moq";

/// The URL path the daemon's `web_transport_quinn` server dispatches to the 9P
/// export plane (H1b / #765): the QUIC path-mux sibling of H1a's `/9p` axum
/// WebSocket endpoint, for the cert-pinned self-signed mesh. A WebTransport
/// session whose CONNECT URL path is this is handed to the injected 9P handler
/// (see [`crate::transport::quinn_transport`]); the mount ticket rides
/// `Tattach.uname`, not the URL, since the session is already cert-pinned.
pub const NINEP_PATH: &str = "/9p";

/// A dialed moq streaming session, erased over the concrete transport.
///
/// [`dial_stream`] returns this so a caller can dial the streaming plane over
/// either quinn/WebTransport (`/moq` path-dispatch) **or** iroh (the `moql`
/// ALPN) and hand the result straight to `moq_net::Client::connect`, which is
/// generic over `web_transport_trait::Session`. The two underlying session
/// types (`web_transport_quinn::Session` / `web_transport_iroh::Session`) do not
/// share a concrete type, so this enum dispatches at the `connect` call.
pub enum MoqStreamSession {
    /// quinn / WebTransport `/moq` session (#274).
    Quinn(web_transport_quinn::Session),
    /// iroh `moql`-ALPN session (#282).
    Iroh(web_transport_iroh::Session),
}

impl MoqStreamSession {
    /// Run the moq-lite handshake over this session, returning the live
    /// `moq_net::Session`. Dispatches to the concrete transport's `connect`.
    pub async fn connect_moq(
        self,
        client: &moq_net::Client,
    ) -> std::result::Result<moq_net::Session, moq_net::Error> {
        match self {
            MoqStreamSession::Quinn(s) => client.connect(s).await,
            MoqStreamSession::Iroh(s) => client.connect(s).await,
        }
    }
}

/// Dial the moq streaming plane for a network-routable reach, returning a live
/// [`web_transport_quinn::Session`] (#274).
///
/// The sibling of [`dial`] for the *streaming* plane: it dials the `/moq` path
/// over `web_transport_quinn` (the server path-dispatches `/moq` → moq, default
/// → RPC), reusing [`crate::transport::quinn_transport`]'s connect + cert-pin
/// helpers. The returned session is handed straight to `moq_net::Client::connect`.
///
/// Only the **network-routable** transports are dialable here. `Ipc` / `Inproc`
/// / `SystemdFd` are same-host endpoints resolved from LOCAL config — never from
/// a wire-published reach — so they are rejected: a co-located client must use
/// the same-host UDS fast path instead of dialing.
pub async fn dial_stream(
    target: &TransportConfig,
) -> Result<MoqStreamSession> {
    use crate::transport::quinn_transport::{
        connect_pinned_hashes_path, connect_webpki_path, verify_peer_cert_pinned,
    };
    match &target.endpoint {
        EndpointType::Quic { addr, server_name, auth } => {
            let pins = auth.accept_cert_hashes();
            let session = if auth.require_web_pki() {
                // WebPKI (optionally + pin): validate via system roots + SNI.
                let session = connect_webpki_path(server_name, addr.port(), MOQ_PATH).await?;
                if !pins.is_empty() {
                    // Defence in depth (#185): also require the leaf to be pinned.
                    verify_peer_cert_pinned(&session, pins)?;
                }
                session
            } else {
                // Self-signed mesh: pin the leaf by its SHA-256 (dial by IP).
                if pins.is_empty() {
                    bail!("dial_stream(): pinned QUIC reach has no cert hashes — not dialable");
                }
                connect_pinned_hashes_path(*addr, pins, MOQ_PATH).await?
            };
            Ok(MoqStreamSession::Quinn(session))
        }
        EndpointType::Iroh { node_id, direct_addrs, relay_url } => {
            // #357: dial-by-node_id-alone is now supported — when no direct addrs
            // or relay are supplied, the shared client endpoint's pkarr / n0 DNS
            // discovery (`presets::N0`) resolves the routable addresses from the
            // `EndpointId`. This is the S2 native-peer direct path (the wire
            // `IrohReach` carries only the node_id). When the resolver *does*
            // supply direct addrs / a relay, they are used as hints to skip /
            // accelerate discovery (faster than waiting on pkarr).
            //
            // #282: dial the iroh `moql` ALPN from the shared process-wide client
            // endpoint (the SAME endpoint the daemon's inbound iroh substrate
            // listens on, installed once at startup), then wrap the authenticated
            // iroh Connection as a `web_transport_iroh::Session` for moq-net.
            //
            // iroh binds the connection to the peer's EndpointId (its Ed25519
            // pubkey), so the channel is identity-bound (stronger than quinn's
            // cert-hash pin); the per-Frame chained-HMAC envelope (§7.5) remains
            // the application-layer integrity check the subscriber verifies.
            let session = dial_iroh_moq(node_id, direct_addrs, relay_url).await?;
            Ok(MoqStreamSession::Iroh(session))
        }
        EndpointType::Ipc { .. }
        | EndpointType::SystemdFd { .. }
        | EndpointType::Inproc { .. } => {
            bail!(
                "dial_stream(): same-host endpoint ({:?}) is resolved from local \
                 config, not dialed from the wire — use the UDS fast path",
                target.endpoint
            )
        }
    }
}

/// Dial the iroh `moql` (moq-lite) ALPN to `node_id` and wrap the connection as
/// a `web_transport_iroh::Session` ready for `moq_net::Client::connect`.
///
/// Uses the process-wide client iroh endpoint installed at startup
/// ([`crate::transport::lazy_iroh::install_iroh_client_endpoint`]) — the same
/// install-once dialer the RPC plane's `LazyIrohTransport` uses, so the streaming
/// dial reuses the node identity and bound sockets. Mirrors
/// [`crate::transport::iroh_substrate::IrohSubstrate::connect`].
async fn dial_iroh_moq(
    node_id: &[u8; 32],
    direct_addrs: &[std::net::SocketAddr],
    relay_url: &Option<String>,
) -> Result<web_transport_iroh::Session> {
    use crate::transport::iroh_substrate::ALPN_MOQ_LITE;
    use iroh::{EndpointAddr, EndpointId, TransportAddr};

    let endpoint = crate::transport::lazy_iroh::iroh_client_endpoint().ok_or_else(|| {
        anyhow!(
            "dial_stream(): no iroh client endpoint installed — call \
             install_iroh_client_endpoint() at startup before dialing iroh streams"
        )
    })?;

    let id = EndpointId::from_bytes(node_id).map_err(|e| anyhow!("invalid iroh node_id: {e}"))?;
    let mut transport_addrs: Vec<TransportAddr> =
        direct_addrs.iter().copied().map(TransportAddr::Ip).collect();
    if let Some(url) = relay_url {
        let relay = url.parse().map_err(|e| anyhow!("invalid iroh relay_url '{url}': {e}"))?;
        transport_addrs.push(TransportAddr::Relay(relay));
    }
    let addr = EndpointAddr::from_parts(id, transport_addrs);

    let conn = endpoint
        .connect(addr, ALPN_MOQ_LITE)
        .await
        .map_err(|e| anyhow!("iroh moql connect: {e}"))?;
    Ok(web_transport_iroh::Session::raw(conn))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::SigningKey;
    use crate::transport::QuicServerAuth;
    use crate::signer::LocalSigner;
    use crate::transport::rpc_session::from_fn;
    use bytes::Bytes;
    use rand::rngs::OsRng;

    fn test_signer() -> LocalSigner {
        LocalSigner::new(SigningKey::generate(&mut OsRng))
    }

    fn echo() -> Arc<dyn IrohRequestProcessor> {
        Arc::new(from_fn(|r: Bytes| async move { Ok(r) }))
    }

    #[test]
    fn register_lookup_unregister() {
        let name = "test/dial/register_lookup_unregister";
        assert!(lookup_inproc(name).is_none());
        let proc = echo();
        register_inproc(name, &proc);
        assert!(lookup_inproc(name).is_some());
        unregister_inproc(name);
        assert!(lookup_inproc(name).is_none());
    }

    #[test]
    fn lookup_self_evicts_when_service_dropped() {
        let name = "test/dial/self_evict";
        let proc = echo();
        register_inproc(name, &proc);
        assert!(lookup_inproc(name).is_some());
        // Service shuts down: drop the only strong ref. The Weak in the
        // registry is now dead and must not resolve (and is pruned).
        drop(proc);
        assert!(lookup_inproc(name).is_none(), "dead-service entry must not resolve");
    }

    #[test]
    fn dial_inproc_resolves_registered_processor() {
        let name = "test/dial/dial_inproc_resolves";
        let proc = echo();
        register_inproc(name, &proc);

        let cfg = TransportConfig::inproc(name);
        let client = dial(&cfg, test_signer(), None, None);
        assert!(client.is_ok(), "dialing a registered inproc endpoint must succeed");

        unregister_inproc(name);
    }

    #[test]
    fn dial_inproc_unregistered_errors() {
        let cfg = TransportConfig::inproc("test/dial/never_registered");
        let err = dial(&cfg, test_signer(), None, None);
        assert!(err.is_err(), "dialing an unregistered inproc endpoint must error");
    }

    #[test]
    fn dial_ipc_builds_lazy_client() {
        // ipc dial builds a lazy UdsSession client — sync, no I/O, connects on
        // first send (the socket need not exist yet at dial() time).
        let cfg = TransportConfig::ipc("/tmp/hyprstream-test-dial.sock");
        let client = dial(&cfg, test_signer(), None, None);
        assert!(client.is_ok(), "ipc dial must build a lazy client without connecting");
    }

    #[test]
    fn dial_systemd_fd_builds_lazy_client() {
        // systemd socket-activation: client dials the client_path via UDS.
        let cfg = TransportConfig::systemd_fd(7, "/tmp/hyprstream-test-systemd.sock");
        let client = dial(&cfg, test_signer(), None, None);
        assert!(client.is_ok(), "systemd-fd dial must build a lazy client via client_path");
    }

    #[test]
    fn dial_quic_webpki_builds_client() {
        // Plain `quic()` = WebPKI/CA validation — a valid auth policy, so dial()
        // builds a (lazy, not-yet-connected) client.
        let cfg = TransportConfig::quic("127.0.0.1:9999".parse().unwrap(), "localhost");
        let client = dial(&cfg, test_signer(), None, None);
        assert!(client.is_ok(), "WebPKI QUIC dial must build a client without connecting");
    }

    #[test]
    fn dial_quic_pinned_builds_client() {
        // Cert-hash-pinned QUIC: builds a lazy client — sync, no I/O.
        let cfg = TransportConfig::quic_pinned("127.0.0.1:9999".parse().unwrap(), "localhost", [7u8; 32]);
        let client = dial(&cfg, test_signer(), None, None);
        assert!(client.is_ok(), "pinned QUIC dial must build a client without connecting");
    }

    #[test]
    fn dial_quic_web_pki_pinned_builds_client() {
        // WebPKI + pin (defence in depth): builds a lazy client.
        let cfg = TransportConfig {
            endpoint: EndpointType::Quic {
                addr: "127.0.0.1:9999".parse().unwrap(),
                server_name: "localhost".to_owned(),
                auth: QuicServerAuth::web_pki_pinned(vec![[9u8; 32]]).unwrap(),
            },
            bind_mode: crate::transport::BindMode::Connect,
        };
        let client = dial(&cfg, test_signer(), None, None);
        assert!(client.is_ok(), "WebPKI+pin QUIC dial must build a client");
    }

    #[test]
    fn quic_server_auth_rejects_no_requirement() {
        assert!(QuicServerAuth::pinned(vec![]).is_err(), "empty pin set is no auth");
        assert!(QuicServerAuth::web_pki_pinned(vec![]).is_err(), "empty pin set is no auth");
        assert!(QuicServerAuth::web_pki().require_web_pki());
    }

    #[test]
    fn dial_stream_iroh_node_id_alone_requires_installed_endpoint() {
        // #357: dial-by-node_id-alone is supported (pkarr / n0 DNS discovery
        // resolves addresses from the EndpointId on the shared client endpoint),
        // so an iroh reach with no direct addrs / relay is no longer rejected
        // up-front. With no client endpoint installed, the dial still fails — but
        // because the shared dialer is missing, not because reachability is absent.
        let cfg = TransportConfig::iroh([1u8; 32], Vec::new(), None);
        let err = match futures::executor::block_on(dial_stream(&cfg)) {
            Ok(_) => panic!("dial must fail with no installed iroh client endpoint"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("no iroh client endpoint installed"),
            "expected an install-endpoint error (node_id-alone is dialable once an \
             endpoint is installed); got: {err}"
        );
    }

    /// #282: `dial_stream`'s iroh arm dials the `moql` ALPN and returns a live
    /// moq session. A loopback: a server `IrohSubstrate` serves the moq handler
    /// with one published broadcast; the client installs its endpoint as the
    /// process-global dialer, then `dial_stream(EndpointType::Iroh{..})` →
    /// `connect_moq` → subscribe → read the same frame back.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dial_stream_iroh_moql_loopback_round_trip() -> Result<()> {
        use crate::transport::iroh_moq::{IrohMoqProtocolHandler, OriginShared};
        use crate::transport::iroh_substrate::{IrohSubstrate, NoopHandler};
        use crate::transport::lazy_iroh::install_iroh_client_endpoint;
        use moq_net::{Client, Group, Origin, OriginConsumer, OriginProducer, Track};
        use rand::RngCore;

        fn fresh_key() -> [u8; 32] {
            let mut k = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut k);
            k
        }

        // ── Server: moq handler on the `moql` ALPN with one broadcast ──────────
        let shared = OriginShared::new();
        let producer = shared.producer().clone();
        let moq_handler = IrohMoqProtocolHandler::with_origin(shared);
        let server =
            IrohSubstrate::new(fresh_key(), moq_handler, NoopHandler::new("rpc")).await?;
        let server_id: [u8; 32] = *server.endpoint_id().as_bytes();
        let direct: Vec<std::net::SocketAddr> =
            server.endpoint().bound_sockets().into_iter().collect();

        let mut broadcast = producer
            .create_broadcast("alice/run-1")
            .ok_or_else(|| anyhow!("create_broadcast denied"))?;
        let mut track = broadcast.create_track(Track::new("tokens"))?;
        let mut group = track.create_group(Group::from(0u64))?;
        group.write_frame(bytes::Bytes::from_static(b"hello-dial-stream"))?;
        drop(group);

        // ── Client: install its endpoint as the process-global dialer ──────────
        let client = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let _ = install_iroh_client_endpoint(client.endpoint().clone());

        // ── dial_stream over iroh → MoqStreamSession::Iroh ─────────────────────
        let cfg = TransportConfig::iroh(server_id, direct, None);
        let session = dial_stream(&cfg).await?;
        assert!(matches!(session, MoqStreamSession::Iroh(_)), "must be an iroh session");

        // Run the moq handshake via the enum dispatcher and subscribe.
        let client_origin: OriginProducer = Origin::random().produce();
        let client_consumer: OriginConsumer = client_origin.consume();
        let moq_client = Client::new().with_consume(client_origin);
        let _moq_session = session
            .connect_moq(&moq_client)
            .await
            .map_err(|e| anyhow!("moq handshake: {e}"))?;

        let bc = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client_consumer.announced_broadcast("alice/run-1"),
        )
        .await?
        .ok_or_else(|| anyhow!("broadcast not announced"))?;
        let mut tc = bc.subscribe_track(&Track::new("tokens"))?;
        let mut gc = tokio::time::timeout(std::time::Duration::from_secs(5), tc.next_group())
            .await??
            .ok_or_else(|| anyhow!("next_group None"))?;
        let frame = tokio::time::timeout(std::time::Duration::from_secs(5), gc.read_frame())
            .await??
            .ok_or_else(|| anyhow!("read_frame None"))?;
        assert_eq!(&frame[..], b"hello-dial-stream");

        server.shutdown().await?;
        // Do NOT shut down `client`: its endpoint is the install-once global.
        drop(client);
        Ok(())
    }
}
