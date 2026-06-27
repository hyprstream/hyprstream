//! E4 (#421) — Iroh transport end-to-end (single host, two endpoints).
//!
//! Validates that the Iroh substrate (#410, PRIMARY production transport, default
//! ON per #411) carries BOTH production planes over a single shared QUIC endpoint:
//!
//! 1. **Daemon binds Iroh** (one [`IrohSubstrate`]) → serves BOTH ALPNs
//!    (`moql` + `hyprstream-rpc/1`) concurrently.
//! 2. **Client dials Iroh** via the [`dial`] factory with
//!    `EndpointType::Iroh { node_id, direct_addrs, .. }` → `Arc<dyn RpcClient>`
//!    (the object-safe client trait, erasing the concrete `LazyIrohTransport`).
//! 3. **RPC round-trip over Iroh**: `model.status()` and `registry.list()`-shaped
//!    operations traverse the signed-envelope pipeline both directions.
//! 4. **moq stream over Iroh**: `dial_stream()` → `MoqStreamSession::Iroh` → a
//!    real `moq_net::Client` pub/sub round-trip.
//!
//! The model/registry *business logic* lives in the libtorch-bound `hyprstream`
//! crate; this test models those services as test-local [`RequestService`] impls
//! that dispatch on the same operation-name payloads the real services do. The
//! transport, envelope, and moq-session wiring under test here is the *same*
//! production wiring — `IrohSubstrate` + `LocalServiceBridge` +
//! `IrohRpcProtocolHandler` + `LazyIrohTransport` + `IrohMoqProtocolHandler` —
//! so a green run means a real `model.status()` call would traverse this path
//! identically.
//!
//! Libtorch is NOT required (this is transport, not inference).
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytes::Bytes;
use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::RngCore;

use hyprstream_rpc::dial::{dial, dial_stream, MoqStreamSession};
use hyprstream_rpc::envelope::InMemoryNonceCache;
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_moq::{IrohMoqProtocolHandler, OriginShared};
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{
    ALPN_HYPRSTREAM_RPC, IrohSubstrate, NoopHandler,
};
use hyprstream_rpc::transport::lazy_iroh::install_iroh_client_endpoint;
use hyprstream_rpc::transport::iroh_transport::IrohTransport;
use hyprstream_rpc::transport::TransportConfig;
use iroh::{EndpointAddr, EndpointId, TransportAddr};
use moq_net::{Client, Group, Origin, OriginConsumer, OriginProducer, Track};

// ─────────────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────────────

fn fresh_signing_key() -> SigningKey {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    SigningKey::from_bytes(&bytes)
}

fn fresh_node_key() -> [u8; 32] {
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    k
}

/// Build an `EndpointAddr` for a server directly from its bound sockets +
/// endpoint id (skips n0 relay/pkarr resolution — no DNS, no network egress).
/// Same hermetic-dial construction used by the Phase 1 smoke test and
/// `iroh_rpc_e2e`.
fn direct_addr(substrate: &IrohSubstrate) -> EndpointAddr {
    EndpointAddr::from_parts(
        substrate.endpoint_id(),
        substrate
            .endpoint()
            .bound_sockets()
            .into_iter()
            .map(TransportAddr::Ip),
    )
}

/// Install the Classical verify policy. Integration tests compile this crate in
/// non-test mode, where the (uninstalled) global verify policy fail-closes to
/// `Hybrid`; this test exercises the classical (EdDSA-only) wire+envelope path,
/// so opt in to what it actually validates. Idempotent-by-first-write.
fn install_classical_verify_policy() {
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// shared process-global iroh client endpoint
// ─────────────────────────────────────────────────────────────────────────────
//
// `dial(EndpointType::Iroh{..})` (the production path under test) connects via
// the process-wide client endpoint installed by `install_iroh_client_endpoint`
// — an install-once, first-write-wins global. In production the daemon
// provisions this once at bootstrap; in a test binary the same constraint
// applies: every test in this binary MUST share ONE client endpoint, or a later
// test's install silently no-ops and its dials ride a stale/already-dropped
// endpoint (`iroh connect: Internal consistency error`).
//
// `shared_client_endpoint()` installs exactly one substrate as the global dialer
// the first time it is called, and returns a cheap `Endpoint` clone (Arc-backed)
// every test reuses. Each test still builds its own server substrate and dials
// through the real `dial()` factory, so the production path is exercised
// faithfully — only the client-side endpoint is shared, exactly as in prod.

// The shared client endpoint is built ONCE on a DEDICATED, never-terminated
// background runtime so its iroh state-actor + Router tasks outlive every
// per-test `#[tokio::test]` runtime.
//
// The previous `tokio::sync::OnceCell` version ran `IrohSubstrate::new()` (which
// `bind()`s the endpoint and `spawn()`s the Router) on whichever test called it
// FIRST. iroh spawns those tasks on the runtime active at `bind()`, so once that
// first test's `#[tokio::test]` runtime was torn down the endpoint's state actor
// stopped and every later test's `connect()` flaked with iroh's "Internal
// consistency error" (`RemoteStateActorStopped`) under CI load. Pinning the
// substrate `Arc` did NOT help — it's the runtime/tasks, not the Arc, that must
// survive. Hosting the substrate on a process-lifetime runtime removes the
// cross-runtime hazard; calling `connect()` from each test's own runtime is fine
// (iroh drives the long-lived actor over channels — the same shape as prod,
// where one endpoint serves the whole process).
static SHARED_CLIENT: std::sync::OnceLock<iroh::Endpoint> = std::sync::OnceLock::new();

/// The shared process-global client endpoint, building + installing it on first
/// call on a dedicated runtime that lives for the whole test binary. Every test
/// reuses this one endpoint (matching the daemon's one-endpoint bootstrap), so
/// `dial()` / `dial_stream()` always see an installed, *live* global no matter
/// which test runs (or finishes) first.
fn shared_client_endpoint_blocking() -> iroh::Endpoint {
    SHARED_CLIENT
        .get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("shared-iroh-client".into())
                .spawn(move || {
                    let rt = tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect("build shared iroh client runtime");
                    // Build the substrate on `rt` and keep BOTH on this thread's
                    // stack (never dropped) so the Router + endpoint state actor
                    // run for the whole process. Dropping the substrate would shut
                    // the Router down; we publish the endpoint clone, then park.
                    let _substrate = rt.block_on(async {
                        let substrate = IrohSubstrate::new(
                            fresh_node_key(),
                            NoopHandler::new("shared-client-moq"),
                            NoopHandler::new("shared-client-rpc"),
                        )
                        .await
                        .expect("bind shared client endpoint");
                        let endpoint = substrate.endpoint().clone();
                        let _ = install_iroh_client_endpoint(endpoint.clone());
                        tx.send(endpoint).expect("publish shared client endpoint");
                        substrate
                    });
                    // Hold the runtime + substrate (and thus the endpoint's tasks)
                    // alive forever.
                    loop {
                        std::thread::park();
                    }
                })
                .expect("spawn shared iroh client thread");
            rx.recv().expect("receive shared client endpoint")
        })
        .clone()
}

/// Async wrapper so existing `shared_client_endpoint().await` call sites are
/// unchanged. The endpoint lives on its own runtime (see [`SHARED_CLIENT`]).
async fn shared_client_endpoint() -> iroh::Endpoint {
    shared_client_endpoint_blocking()
}

// ─────────────────────────────────────────────────────────────────────────────
// test-local services: model.status() + registry.list() dispatch
// ─────────────────────────────────────────────────────────────────────────────
//
// These mirror the dispatch shape of the real `hyprstream::services::model` and
// registry services: a `RequestService` whose `handle_request` decodes an
// operation name from the payload and returns an op-specific response. The
// transport, envelope, and bridge wiring exercised here is the SAME production
// wiring the real services ride, so a green round-trip means the Iroh transport
// carries a real `model.status()` / `registry.list()` call identically.
//
// Payload convention (test-local, length-prefixed op name + opaque arg):
//   request  = [u8; 4] op_len BE | op_name bytes | arg bytes
//   response = op-specific (see handlers)

/// Decode the operation name + remainder from a `handle_request` payload.
fn decode_op(payload: &[u8]) -> Result<(&str, &[u8])> {
    if payload.len() < 4 {
        return Err(anyhow!("payload too short for op-name prefix"));
    }
    let op_len = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if payload.len() < 4 + op_len {
        return Err(anyhow!("payload shorter than declared op_len"));
    }
    let op = std::str::from_utf8(&payload[4..4 + op_len])?;
    Ok((op, &payload[4 + op_len..]))
}

fn encode_op(op: &str, arg: &[u8]) -> Vec<u8> {
    let op_len = op.len() as u32;
    let mut out = Vec::with_capacity(4 + op.len() + arg.len());
    out.extend_from_slice(&op_len.to_be_bytes());
    out.extend_from_slice(op.as_bytes());
    out.extend_from_slice(arg);
    out
}

/// Test-local stand-in for the production `model` service. Handles `status`:
/// - `status` with no arg        → JSON-ish list of all known models
/// - `status:<model_ref>`        → status of one model
///
/// Returns an op-tagged response so the client can assert the round-trip
/// preserved both the operation and its argument.
struct ModelService {
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
}

impl ModelService {
    fn new(signing_key: SigningKey) -> Self {
        Self {
            name: "model".to_owned(),
            transport: TransportConfig::inproc("model-iroh-e2e-unused"),
            signing_key,
        }
    }
}

#[async_trait(?Send)]
impl RequestService for ModelService {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        let (op, arg) = decode_op(payload)?;
        match op {
            "status" => {
                // Real `model_status_all` / `model_status_single` shape: a list
                // of {model_ref, status} entries. We emit a deterministic
                // response that encodes the arg so the test can verify the
                // request payload survived the round-trip.
                let model_ref = if arg.is_empty() {
                    "all".to_owned()
                } else {
                    String::from_utf8_lossy(arg).into_owned()
                };
                let body = format!("model.status:ok:{model_ref}=loaded");
                Ok((body.into_bytes(), None))
            }
            other => Err(anyhow!("model: unknown op '{other}'")),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    // Force Classical responses: this test exercises the EdDSA-only envelope
    // path (no PQ store provisioned), matching the verify policy installed below.
    fn pq_signing_key(&self) -> Option<hyprstream_rpc::crypto::pq::MlDsaSigningKey> {
        None
    }
}

/// Test-local stand-in for the production `registry` service. Handles `list`:
/// returns the set of registered service names.
struct RegistryService {
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
    /// Service names this fake registry advertises.
    services: Vec<String>,
}

impl RegistryService {
    fn new(signing_key: SigningKey, services: Vec<String>) -> Self {
        Self {
            name: "registry".to_owned(),
            transport: TransportConfig::inproc("registry-iroh-e2e-unused"),
            signing_key,
            services,
        }
    }
}

#[async_trait(?Send)]
impl RequestService for RegistryService {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        let (op, _arg) = decode_op(payload)?;
        match op {
            "list" => {
                // Real `registry_list` shape: a newline-delimited list of
                // registered service names.
                let body = self.services.join(",");
                Ok((format!("registry.list:ok:{body}").into_bytes(), None))
            }
            other => Err(anyhow!("registry: unknown op '{other}'")),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn pq_signing_key(&self) -> Option<hyprstream_rpc::crypto::pq::MlDsaSigningKey> {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E4.1 + E4.2 + E4.3: RPC round-trip over Iroh (dial → Arc<dyn RpcClient>)
// ─────────────────────────────────────────────────────────────────────────────
//
// One server substrate binds the `hyprstream-rpc/1` ALPN with a `LocalServiceBridge`
// over a multi-op service that dispatches both `model.status` and `registry.list`.
// The client installs its endpoint as the process-global iroh dialer, then dials
// via the production `dial()` factory with `EndpointType::Iroh { .. }`, which
// returns a ready `Arc<dyn RpcClient>`. Two ops round-trip over the same
// connection to exercise multi-stream multiplexing.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rpc_round_trip_model_status_and_registry_list_over_iroh() -> Result<()> {
    install_classical_verify_policy();

    // ─── Server: bind Iroh substrate serving `hyprstream-rpc/1` ──────────────
    //
    // The bridge wraps a single `RequestService` that dispatches BOTH
    // `model.status` and `registry.list`, mirroring how a real daemon's RPC
    // plane fronts multiple ops behind one signed-envelope processor. The moq
    // ALPN is a NoopHandler here — the moq round-trip has its own test below.
    let server_signing = fresh_signing_key();
    let server_vk: VerifyingKey = server_signing.verifying_key();
    let nonce_cache = Arc::new(InMemoryNonceCache::new());

    // A composite service that dispatches like a real multi-op service: it
    // carries both the model and registry service bodies and routes by op name.
    let composite = CompositeRpcService::new(
        server_signing.clone(),
        ModelService::new(fresh_signing_key()),
        RegistryService::new(
            fresh_signing_key(),
            vec!["model".to_owned(), "registry".to_owned(), "policy".to_owned()],
        ),
    );

    let bridge = LocalServiceBridge::spawn(composite, Arc::clone(&nonce_cache), 0)?;
    let server = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("moq-not-wired-in-rpc-test"),
        IrohRpcProtocolHandler::new(bridge, server_signing.clone()),
    )
    .await?;

    let server_node_id: [u8; 32] = *server.endpoint_id().as_bytes();
    let server_direct: Vec<std::net::SocketAddr> =
        server.endpoint().bound_sockets().into_iter().collect();
    let server_addr = direct_addr(&server);

    // ─── Client: dial via the production `dial()` factory ───────────────────
    //
    // `dial(EndpointType::Iroh{..})` is the production path: it returns a
    // `LazyIrohTransport` wrapped in `RpcClientImpl`, erased behind
    // `Arc<dyn RpcClient>`. The lazy transport connects on first `send()` and
    // reuses the shared process-global client endpoint (one per process, exactly
    // as in the daemon bootstrap).
    let client_endpoint = shared_client_endpoint().await;

    let target = TransportConfig::iroh(server_node_id, server_direct.clone(), None);
    let rpc: Arc<dyn hyprstream_rpc::rpc_client::RpcClient> = dial(
        &target,
        LocalSigner::new(fresh_signing_key()),
        Some(server_vk),
        None,
    )?;

    // ─── RPC round-trip #1: model.status() ──────────────────────────────────
    //
    // First `send()` lazily dials iroh (ALPN `hyprstream-rpc/1`), opens a bidi
    // stream, writes the signed envelope, and reads back the signed response,
    // which `RpcClientImpl` verifies against `server_vk`.
    let status_req = encode_op("status", b"qwen3-4b");
    let resp = rpc.call(status_req).await?;
    let resp_str = std::str::from_utf8(&resp)?;
    assert_eq!(resp_str, "model.status:ok:qwen3-4b=loaded");

    // A second op on the same dialed connection exercises stream multiplexing.
    let status_all_req = encode_op("status", b"");
    let resp_all = rpc.call(status_all_req).await?;
    let resp_all_str = std::str::from_utf8(&resp_all)?;
    assert_eq!(resp_all_str, "model.status:ok:all=loaded");

    // ─── RPC round-trip #2: registry.list() ─────────────────────────────────
    let list_req = encode_op("list", b"");
    let resp_list = rpc.call(list_req).await?;
    let resp_list_str = std::str::from_utf8(&resp_list)?;
    assert_eq!(resp_list_str, "registry.list:ok:model,registry,policy");

    // ─── Direct-connection parity: the same ops round-trip over an explicitly
    // dialed `IrohTransport` (no `dial()` factory), proving the lazy path and
    // the explicit path terminate the same server handler. This guards against
    // a regression where `dial()`'s erasure drops a setup step. Uses the SAME
    // shared client endpoint the `dial()` factory rides.
    {
        let conn = client_endpoint.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;
        let explicit_rpc = RpcClientImpl::new(
            LocalSigner::new(fresh_signing_key()),
            IrohTransport::new(conn),
            Some(server_vk),
        );
        let r = explicit_rpc.call(encode_op("status", b"explicit")).await?;
        assert_eq!(std::str::from_utf8(&r)?, "model.status:ok:explicit=loaded");
    }

    // Do NOT shut down the shared client endpoint: it is the install-once
    // process-global dialer shared with every other test in this binary (and
    // matches the never-reset prod lifecycle).
    server.shutdown().await?;
    Ok(())
}

/// A `RequestService` that routes by op-name prefix to one of two inner services
/// (`model` / `registry`), mirroring how a real daemon fronts multiple services
/// behind one RPC plane. The inner services own their dispatch; this composite
/// only picks which one to delegate to.
struct CompositeRpcService {
    signing_key: SigningKey,
    model: ModelService,
    registry: RegistryService,
}

impl CompositeRpcService {
    fn new(signing_key: SigningKey, model: ModelService, registry: RegistryService) -> Self {
        Self {
            signing_key,
            model,
            registry,
        }
    }
}

#[async_trait(?Send)]
impl RequestService for CompositeRpcService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        // Peek the op name without consuming the payload so the inner service
        // can re-decode it.
        let (op, _) = decode_op(payload)?;
        match op {
            "status" => self.model.handle_request(ctx, payload).await,
            "list" => self.registry.handle_request(ctx, payload).await,
            other => Err(anyhow!("composite: unknown op '{other}'")),
        }
    }

    fn name(&self) -> &str {
        "composite-rpc"
    }

    fn transport(&self) -> &TransportConfig {
        self.model.transport()
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn pq_signing_key(&self) -> Option<hyprstream_rpc::crypto::pq::MlDsaSigningKey> {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E4.4: moq stream over Iroh (MoqStreamSession::Iroh pub/sub round-trip)
// ─────────────────────────────────────────────────────────────────────────────
//
// The server substrate binds the `moql` ALPN with a real `IrohMoqProtocolHandler`
// holding a shared origin. The client dials via the production `dial_stream()`
// factory, which returns `MoqStreamSession::Iroh`; the moq handshake runs through
// `connect_moq()`, and a published Frame is read back by the subscriber.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn moq_stream_pub_sub_round_trip_over_iroh() -> Result<()> {
    // ─── Server: moq handler on the `moql` ALPN with one published broadcast
    let shared = OriginShared::new();
    let producer = shared.producer().clone();
    let moq_handler = IrohMoqProtocolHandler::with_origin(shared);
    let server = IrohSubstrate::new(
        fresh_node_key(),
        moq_handler,
        NoopHandler::new("rpc-not-wired-in-moq-test"),
    )
    .await?;

    let server_node_id: [u8; 32] = *server.endpoint_id().as_bytes();
    let server_direct: Vec<std::net::SocketAddr> =
        server.endpoint().bound_sockets().into_iter().collect();

    // Publish a broadcast with one track + one group + one frame BEFORE the
    // subscriber connects (late-join: the frame must still be readable).
    const BROADCAST: &str = "alice/run-1";
    const TRACK: &str = "tokens";
    const FRAME: &[u8] = b"hello-moq-over-iroh";
    // Keep `broadcast` + `track` alive for the rest of the test: dropping the
    // `BroadcastProducer`/`TrackProducer` un-announces the broadcast from the
    // origin, so a late-joining subscriber would never see it. Only the group
    // is dropped (it just finalizes that group's frames).
    let mut broadcast = producer
        .create_broadcast(BROADCAST)
        .ok_or_else(|| anyhow!("create_broadcast denied"))?;
    let mut track = broadcast.create_track(Track::new(TRACK))?;
    {
        let mut group = track.create_group(Group::from(0u64))?;
        group.write_frame(Bytes::from_static(FRAME))?;
        drop(group);
    }

    // ─── Client: dial_stream via the shared process-global iroh endpoint ─────
    // `dial_stream`'s iroh arm reuses the same install-once client endpoint as
    // the RPC plane's `dial()`, so it shares the endpoint installed by
    // `shared_client_endpoint()`.
    let _client_endpoint = shared_client_endpoint().await;

    // ─── dial_stream over iroh → MoqStreamSession::Iroh ──────────────────────
    let cfg = TransportConfig::iroh(server_node_id, server_direct, None);
    let session = dial_stream(&cfg).await?;
    assert!(
        matches!(session, MoqStreamSession::Iroh(_)),
        "dial_stream must return MoqStreamSession::Iroh for an iroh reach"
    );

    // Run the moq handshake via the enum dispatcher, then subscribe + read.
    let client_origin: OriginProducer = Origin::random().produce();
    let client_consumer: OriginConsumer = client_origin.consume();
    let moq_client = Client::new().with_consume(client_origin);
    let _moq_session = session
        .connect_moq(&moq_client)
        .await
        .map_err(|e| anyhow!("moq handshake over iroh: {e}"))?;

    let bc = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client_consumer.announced_broadcast(BROADCAST),
    )
    .await?
    .ok_or_else(|| anyhow!("broadcast '{BROADCAST}' not announced"))?;
    let mut tc = bc.subscribe_track(&Track::new(TRACK))?;
    let mut gc = tokio::time::timeout(std::time::Duration::from_secs(5), tc.next_group())
        .await??
        .ok_or_else(|| anyhow!("next_group returned None"))?;
    let frame: Bytes = tokio::time::timeout(std::time::Duration::from_secs(5), gc.read_frame())
        .await??
        .ok_or_else(|| anyhow!("read_frame returned None"))?;
    assert_eq!(&frame[..], FRAME, "moq frame must round-trip over iroh");

    // Do NOT shut down the shared client endpoint (install-once global dialer).
    server.shutdown().await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// E4.1 (both ALPNs): one substrate serves RPC + moq concurrently
// ─────────────────────────────────────────────────────────────────────────────
//
// The #410/#411 production posture: a SINGLE `IrohSubstrate` serves BOTH ALPNs
// (`moql` + `hyprstream-rpc/1`) with the same node identity. This test binds one
// substrate with a real moq handler AND a real RPC handler, then exercises BOTH
// planes against it — proving the router dispatches by ALPN and that the two
// planes do not interfere. This is the full E4 acceptance check in one process.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_substrate_serves_both_alpns_rpc_and_moq() -> Result<()> {
    install_classical_verify_policy();

    // ─── Server: one substrate, two real handlers ───────────────────────────
    let shared = OriginShared::new();
    let producer = shared.producer().clone();
    let moq_handler = IrohMoqProtocolHandler::with_origin(shared);

    let server_signing = fresh_signing_key();
    let server_vk: VerifyingKey = server_signing.verifying_key();
    let nonce_cache = Arc::new(InMemoryNonceCache::new());
    let bridge = LocalServiceBridge::spawn(
        ModelService::new(server_signing.clone()),
        Arc::clone(&nonce_cache),
        0,
    )?;
    let rpc_handler = IrohRpcProtocolHandler::new(bridge, server_signing.clone());

    let server = IrohSubstrate::new(fresh_node_key(), moq_handler, rpc_handler).await?;
    let server_node_id: [u8; 32] = *server.endpoint_id().as_bytes();
    let server_direct: Vec<std::net::SocketAddr> =
        server.endpoint().bound_sockets().into_iter().collect();

    // Publish a frame so the moq plane has something to serve.
    const BROADCAST: &str = "bob/run-2";
    const TRACK: &str = "tokens";
    const FRAME: &[u8] = b"both-alpns-frame";
    // Keep `broadcast` + `track` alive past the subscriber's connect: dropping
    // the producers un-announces the broadcast (see the moq-only test above).
    let mut broadcast = producer
        .create_broadcast(BROADCAST)
        .ok_or_else(|| anyhow!("create_broadcast denied"))?;
    let mut track = broadcast.create_track(Track::new(TRACK))?;
    {
        let mut group = track.create_group(Group::from(0u64))?;
        group.write_frame(Bytes::from_static(FRAME))?;
        drop(group);
    }

    // ─── Client: the shared process-global endpoint dials BOTH ALPNs ────────
    // One shared client endpoint serves both the RPC `dial()` and the moq
    // `dial_stream()` — exactly the production posture (one node identity,
    // outbound, two ALPNs).
    let _client_endpoint = shared_client_endpoint().await;

    // ── RPC plane over `hyprstream-rpc/1` (via the `dial()` factory) ─────────
    let target = TransportConfig::iroh(server_node_id, server_direct.clone(), None);
    let rpc: Arc<dyn hyprstream_rpc::rpc_client::RpcClient> = dial(
        &target,
        LocalSigner::new(fresh_signing_key()),
        Some(server_vk),
        None,
    )?;
    let resp = rpc.call(encode_op("status", b"both-alpns-model")).await?;
    assert_eq!(
        std::str::from_utf8(&resp)?,
        "model.status:ok:both-alpns-model=loaded"
    );

    // ── Streaming plane over `moql` (via `dial_stream()`) ────────────────────
    let stream_cfg = TransportConfig::iroh(server_node_id, server_direct, None);
    let session = dial_stream(&stream_cfg).await?;
    assert!(matches!(session, MoqStreamSession::Iroh(_)));

    let client_origin: OriginProducer = Origin::random().produce();
    let client_consumer: OriginConsumer = client_origin.consume();
    let moq_client = Client::new().with_consume(client_origin);
    let _moq_session = session
        .connect_moq(&moq_client)
        .await
        .map_err(|e| anyhow!("moq handshake over iroh: {e}"))?;

    let bc = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client_consumer.announced_broadcast(BROADCAST),
    )
    .await?
    .ok_or_else(|| anyhow!("broadcast '{BROADCAST}' not announced"))?;
    let mut tc = bc.subscribe_track(&Track::new(TRACK))?;
    let mut gc = tokio::time::timeout(std::time::Duration::from_secs(5), tc.next_group())
        .await??
        .ok_or_else(|| anyhow!("next_group returned None"))?;
    let frame: Bytes = tokio::time::timeout(std::time::Duration::from_secs(5), gc.read_frame())
        .await??
        .ok_or_else(|| anyhow!("read_frame returned None"))?;
    assert_eq!(&frame[..], FRAME);

    // Do NOT shut down the shared client endpoint (install-once global dialer).
    server.shutdown().await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Sanity: a bare iroh reach (node_id only, no direct addrs / relay) to a node
// that is NOT discoverable must FAIL — bounded, not hang.
//
// #357 changed `dial_stream`'s iroh arm: node_id-alone is now dialable via the
// shared endpoint's pkarr / n0 DNS discovery, so `dial_stream` no longer
// fast-fails up-front on `direct_addrs.is_empty() && relay_url.is_none()` (that
// reachability precondition now lives only in the RPC-plane `dial()` factory).
// This previously asserted that removed "not dialable / reachability" message
// and so failed deterministically. The meaningful post-#357 invariant — and the
// regression guard kept here — is that a reach to an *undiscoverable* random
// EndpointId resolves to an error in bounded time rather than hanging. (The
// matching unit test is `dial::tests::dial_stream_iroh_node_id_alone_requires_installed_endpoint`.)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dial_stream_iroh_without_reach_fails_fast() {
    // Ensure the shared client endpoint is installed + live (otherwise the dial
    // would short-circuit on "no iroh client endpoint installed", which is
    // order-dependent across this binary's tests). With it installed we exercise
    // the real #357 discovery path against a key that has no published record.
    let _client_endpoint = shared_client_endpoint().await;

    // EndpointId must be a valid Ed25519 pubkey; use a well-formed nonzero one
    // that is not published anywhere, so discovery cannot resolve any address.
    let cfg = TransportConfig::iroh(
        EndpointId::from_bytes(&[7u8; 32])
            .map(|id| *id.as_bytes())
            .unwrap_or([7u8; 32]),
        Vec::new(),
        None,
    );

    // Bound the dial: it must COMPLETE with an error, not hang. A `Err(_elapsed)`
    // here would be a real regression (a node_id-only reach hanging in discovery).
    let res = tokio::time::timeout(std::time::Duration::from_secs(30), dial_stream(&cfg)).await;
    match res {
        Ok(inner) => assert!(
            inner.is_err(),
            "iroh reach to an undiscoverable node_id must fail, not succeed"
        ),
        Err(_elapsed) => panic!("dial_stream hung on an undiscoverable node_id-only reach"),
    }
}
