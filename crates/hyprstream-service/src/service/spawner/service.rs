//! Unified service spawner for RequestService hosting.
//!
//! Provides a single API for spawning RequestService implementations with different
//! execution modes (Tokio task, dedicated thread, or subprocess).

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use tokio::sync::Notify;

use super::{ProcessConfig, ProcessSpawner, SpawnedProcess};
use hyprstream_rpc::error::Result;
use hyprstream_rpc::registry::{ServiceRegistration, SocketKind};
use hyprstream_rpc::service::RequestService;
use hyprstream_rpc::transport::TransportConfig;

// Import anyhow! macro for error creation in ServiceManager impl
use anyhow::anyhow;

// Re-export Spawnable trait from hyprstream-rpc (where it's defined so
// types in that crate can implement it without circular deps).
pub use hyprstream_rpc::service::Spawnable;

// ============================================================================
// QuicServiceLoop — Spawnable wrapper for QUIC-only service loop
// ============================================================================

/// Spawnable wrapper that runs a RequestService with explicit QUIC configuration.
///
/// This is used when a service factory wants to pass a `QuicLoopConfig` to
/// `RequestLoop::with_quic()` without relying on the blanket `Spawnable` impl.
///
/// The blanket `impl Spawnable for S: RequestService` creates a plain `RequestLoop`;
/// this wrapper additionally enables QUIC when `quic_config` is `Some`.
pub struct UnifiedServiceConfig<S: RequestService + Send + 'static> {
    service: S,
    quic_config: Option<hyprstream_rpc::service::QuicLoopConfig>,
}

impl<S: RequestService + Send + 'static> UnifiedServiceConfig<S> {
    /// Create a unified service config with optional QUIC.
    pub fn new(service: S, quic_config: Option<hyprstream_rpc::service::QuicLoopConfig>) -> Self {
        Self { service, quic_config }
    }
}

impl<S: RequestService + Send + Sync + 'static> Spawnable for UnifiedServiceConfig<S> {
    fn name(&self) -> &str {
        RequestService::name(&self.service)
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![(SocketKind::Rep, RequestService::transport(&self.service).clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<()> {
        use hyprstream_rpc::transport::rpc_session::IrohRequestProcessor;

        let UnifiedServiceConfig { service, quic_config } = *self;
        let transport = RequestService::transport(&service).clone();
        let signing_key = RequestService::signing_key(&service);
        let server_pubkey = signing_key.verifying_key();
        let service_name = RequestService::name(&service).to_owned();
        let reach_config_handle = service.producer_reach_config_handle();
        let moq_origin_handle = service.moq_origin_handle();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        // A LocalSet is retained for any `!Send` per-service internals reachable
        // through the bridge; the quinn accept loop itself is fully `Send`.
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
            let _ = server_pubkey; // identity is verified app-layer via the bridge
            let nonce_cache = Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new());
            let bridge = hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge::spawn(
                service, Arc::clone(&nonce_cache), 0,
            ).map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("bridge: {e}")))?;
            let processor: Arc<dyn IrohRequestProcessor> = Arc::new(bridge);

            if let Some(mut qc) = quic_config {
                // web-transport-quinn has no per-builder provider hook and
                // resolves rustls's process default. Install and validate it at
                // the actual bind seam so task/thread/subprocess startup cannot
                // silently inherit a non-PQ first-installed provider (#557).
                hyprstream_rpc::transport::install_pq_crypto_provider().map_err(|error| {
                    hyprstream_rpc::error::RpcError::SpawnFailed(format!(
                        "QUIC crypto provider: {error}"
                    ))
                })?;
                // #274: serve the RPC plane over `web_transport_quinn` (replacing
                // the bespoke h3 WebTransportServer) and multiplex the moq
                // streaming plane on the same endpoint via `/moq` path-dispatch.
                let _ = &qc.protected_resource_json; // RFC 9728 metadata served by axum (:6790)
                let chain: Vec<rustls::pki_types::CertificateDer<'static>> = qc
                    .cert_chain
                    .iter()
                    .map(|d| rustls::pki_types::CertificateDer::from(d.clone()))
                    .collect();
                let key = rustls::pki_types::PrivateKeyDer::try_from((*qc.key_der).clone())
                    .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("QUIC key: {e}")))?;
                let wt_server = web_transport_quinn::ServerBuilder::new()
                    .with_addr(qc.bind_addr)
                    .with_certificate(chain, key)
                    .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("QUIC bind: {e}")))?;
                let actual_addr = wt_server.local_addr()
                    .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("QUIC local_addr: {e}")))?;
                // The bound addr is unspecified (0.0.0.0 / ::) when binding all
                // interfaces — that is NOT a dialable destination, so advertising
                // it makes dial_stream's QUIC connect time out. Advertise loopback
                // for same-host (the single-node default); cross-host routable-IP
                // advertisement is the wire-reach follow-up (#131/#282).
                let advertise_addr = if actual_addr.ip().is_unspecified() {
                    match actual_addr {
                        std::net::SocketAddr::V4(_) => {
                            std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, actual_addr.port()))
                        }
                        std::net::SocketAddr::V6(_) => {
                            std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, actual_addr.port()))
                        }
                    }
                } else {
                    actual_addr
                };
                let pin = hyprstream_rpc::transport::quinn_transport::cert_sha256(&qc.cert_chain[0]);
                let relay_origin = qc.moq_relay.as_ref().map(|_| {
                    hyprstream_rpc::moq_stream::MoqStreamOrigin::standalone()
                        .with_prefix(hyprstream_rpc::moq_stream::DEFAULT_PREFIX)
                        .build()
                });
                if let Some(handle) = &moq_origin_handle {
                    *handle.write() = relay_origin.clone();
                }
                let moq_origin = relay_origin
                    .clone()
                    .or_else(|| hyprstream_rpc::moq_stream::global_moq_origin().cloned());

                let mut rpc_server = hyprstream_rpc::transport::quinn_transport::QuinnRpcServer::with_capacity(
                    wt_server,
                    Arc::clone(&processor),
                    signing_key.clone(),
                    hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
                );
                // Relay-configured services get a distinct origin; other services
                // retain the process-global origin and existing behavior.
                if let Some(origin) = &moq_origin {
                    rpc_server = rpc_server.with_moq_consumer(origin.consumer().clone());
                    // #276: subscribe-authz + per-tenant announce scoping. The
                    // default config is permissive (no resolver, no authorizer)
                    // so the working open same-tenant subscribe model is
                    // preserved. A deployment opts into per-tenant scoping /
                    // private-stream gating by building a
                    // `hyprstream_rpc::transport::iroh_moq::MoqAuthzConfig` with
                    // a `tenant_resolver` and a `DefaultAuthorizer` whose policy
                    // gate calls into the Casbin `PolicyManager`
                    // (`hyprstream::auth::policy_manager::global_policy_manager()`
                    // -> `check_with_domain(subject, tenant, broadcast, "subscribe")`).
                    // That gate is constructed in the `hyprstream` crate (which
                    // owns PolicyManager) and threaded down via the service
                    // factory; wiring it here would create a `hyprstream-service`
                    // -> `hyprstream` dependency cycle, so it is left as a seam.
                    // TODO(#276): thread a `MoqAuthzConfig` through the service
                    // factory (built in the `hyprstream` crate) and call
                    // `.with_moq_authz(cfg)` here. NOTE: on this quinn `/moq`
                    // path the peer is anonymous (no mutual TLS), so the
                    // resolver/gate sees `PeerIdentity::anonymous()` — full
                    // per-peer enforcement requires the iroh `moql` path, where
                    // `with_authz` is already honoured.
                    let _moq_authz_default =
                        hyprstream_rpc::transport::iroh_moq::MoqAuthzConfig::default();

                    // #1153: authenticate the `/moq` WebTransport CONNECT and
                    // bind the session to a tenant on a *shared* multi-tenant
                    // endpoint. The boundary is opt-in: the `hyprstream` crate
                    // builds a `MoqConnectAuthz` over the authoritative
                    // subject→tenant provisioning map (PolicyManager) and
                    // registers it via
                    // `QuinnRpcServer::set_global_moq_connect_authz` at startup;
                    // `QuinnRpcServer::with_capacity` picks it up here
                    // automatically (same global-seam pattern as the 9P
                    // handler). When registered, an unauthenticated or
                    // tenant-less CONNECT is refused before the handshake
                    // completes (fail-closed) and each authenticated peer sees
                    // only its own tenant's broadcasts. When NOT registered the
                    // `/moq` plane stays single-tenant/open — a deployment
                    // choice, not a hidden wildcard. NOTE: this does not
                    // enumerate the registered resolver here; it is consumed
                    // implicitly via the global inside `QuinnRpcServer`. The
                    // explicit pickup below documents that we expect it.
                    if hyprstream_rpc::transport::quinn_transport::global_moq_connect_authz()
                        .is_some()
                    {
                        tracing::info!(
                            "/moq CONNECT authentication is enabled (shared-endpoint tenant boundary #1153)"
                        );
                    }
                }

                // Register this endpoint for RPC resolution and, once, as the
                // node's network reach for StreamInfo producers (#274) — the
                // SAME (addr, server_name, cert-hash) the DID-doc #quic entry
                // advertises, built from one source so no per-site drift.
                if let Some(reg) = hyprstream_rpc::registry::try_global() {
                    reg.register(
                        &service_name,
                        hyprstream_rpc::registry::SocketKind::Quic,
                        hyprstream_rpc::transport::TransportConfig::quic_pinned(
                            advertise_addr,
                            &qc.server_name,
                            pin,
                        ),
                        None,
                    );
                }
                if let Some(handle) = &reach_config_handle {
                    *handle.write() = hyprstream_rpc::moq_stream::ProducerReachConfig {
                        iroh_node_id: None,
                        quic_reach: Some(hyprstream_rpc::moq_stream::NodeStreamReach {
                        addr: advertise_addr,
                        server_name: qc.server_name.clone(),
                        cert_hashes: vec![pin],
                        }),
                        relay: qc.moq_relay.clone(),
                    };
                }
                if let Some(cb) = qc.on_quic_bound.take() {
                    cb(service_name.clone(), advertise_addr, qc.server_name.clone());
                }

                // Link a relay only to this service's scoped origin. The shared
                // process origin would leak other services' broadcasts into it.
                if let Some(relay) = qc.moq_relay.take() {
                    if let Some(origin) = relay_origin {
                        hyprstream_rpc::moq_stream::serve_origin_to_relay_background(
                            origin.producer().clone(),
                            relay,
                        );
                        tracing::info!(
                            service = %service_name,
                            "moq relay rendezvous enabled (announcing origin UP to relay)"
                        );
                    }
                }

                // #410/#282: bind an iroh substrate as the PRIMARY production
                // endpoint (on by default; `[quic] iroh = false` opts out), serving
                // BOTH ALPNs (`hyprstream-rpc/1` + `moql`) with the SAME request
                // processor + moq origin, in parallel to the quinn endpoint (kept
                // for back-compat). The iroh endpoint uses a domain-separated
                // transport key, so its NodeId is not even byte-equal to the
                // service signer. It remains only a carrier address, never a DID
                // or admission input (#1031).
                // (`presets::N0`), so this node is dial-by-node_id-discoverable.
                //
                // Kept alive in this scope via `_iroh_substrate`; on shutdown the
                // spawned task calls `IrohSubstrate::shutdown` to drain handlers.
                let _iroh_substrate_guard = if qc.iroh_enabled {
                    let iroh_transport_key = hyprstream_rpc::node_identity::derive_purpose_key(
                        &signing_key,
                        "hyprstream-iroh-transport-v1",
                    );
                    let iroh_secret = iroh_transport_key.to_bytes();
                    let node_id: [u8; 32] = iroh_transport_key.verifying_key().to_bytes();
                    // Use the same service-specific origin as the quinn `/moq`
                    // path, so relay-scoped broadcasts remain isolated on iroh.
                    let moq_handler = match moq_origin.as_ref() {
                        Some(origin) => {
                            let shared = hyprstream_rpc::transport::iroh_moq::OriginShared::from_pair(
                                origin.producer().clone(),
                                origin.consumer().clone(),
                            );
                            hyprstream_rpc::transport::iroh_moq::IrohMoqProtocolHandler::with_origin(shared)
                        }
                        None => hyprstream_rpc::transport::iroh_moq::IrohMoqProtocolHandler::new(),
                    };
                    // RPC plane: same processor + signing key as the quinn path.
                    let rpc_handler =
                        hyprstream_rpc::transport::iroh_rpc::IrohRpcProtocolHandler::with_stream_limit(
                            Arc::clone(&processor),
                            signing_key.clone(),
                            hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
                        );
                    match hyprstream_rpc::transport::iroh_substrate::IrohSubstrate::new(
                        iroh_secret, moq_handler, rpc_handler,
                    )
                    .await
                    {
                        Ok(substrate) => {
                            // Install the shared client endpoint (install-once) so
                            // outbound iroh RPC/stream dials reuse this carrier.
                            let _ = hyprstream_rpc::transport::lazy_iroh::install_iroh_client_endpoint(
                                substrate.owned_client_endpoint(),
                            );
                            if let Some(handle) = &reach_config_handle {
                                handle.write().iroh_node_id = Some(node_id);
                            }
                            // Advertise iroh reachability only now that the carrier
                            // is bound; EndpointId is never application authority.
                            if let Some(cb) = qc.on_iroh_bound.take() {
                                cb(service_name.clone(), node_id);
                            }
                            tracing::info!(
                                service = %service_name,
                                node_id = %hex_short(&node_id),
                                "iroh substrate bound (ALPNs hyprstream-rpc/1 + moql)"
                            );
                            Some(substrate)
                        }
                        Err(e) => {
                            // Fail soft: an iroh bind failure must not take down the
                            // working quinn plane. Log and continue quinn-only.
                            tracing::warn!(
                                service = %service_name,
                                "iroh substrate bind failed; continuing quinn-only: {e}"
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                // Drain the iroh substrate on shutdown (parallel to quinn drain).
                if let Some(substrate) = _iroh_substrate_guard {
                    let iroh_shutdown = Arc::clone(&shutdown);
                    tokio::spawn(async move {
                        iroh_shutdown.notified().await;
                        if let Err(e) = substrate.shutdown().await {
                            tracing::warn!("iroh substrate shutdown error: {e}");
                        }
                    });
                }

                // Bridge the `Notify` shutdown to the server's graceful drain.
                let drain_limit = rpc_server.stream_limit();
                let drain_capacity = rpc_server.capacity();
                let drain_token = rpc_server.shutdown_token();
                let drain_shutdown = Arc::clone(&shutdown);
                tokio::spawn(async move {
                    drain_shutdown.notified().await;
                    hyprstream_rpc::transport::quinn_transport::QuinnRpcServer::shutdown(
                        &drain_limit, drain_capacity, &drain_token,
                    ).await;
                });

                let rep_fut = hyprstream_rpc::service::serve::serve_bridged(
                    &transport, Arc::clone(&processor), signing_key.clone(),
                    Arc::clone(&shutdown), on_ready,
                );
                let quic_fut = rpc_server.run();
                let (rep_result, quic_result) = tokio::join!(rep_fut, quic_fut);
                if let Err(e) = quic_result {
                    tracing::warn!("QUIC server loop ended with error: {e}");
                }
                rep_result.map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(e.to_string()))
            } else {
                hyprstream_rpc::service::serve::serve_bridged(
                    &transport, processor, signing_key, shutdown, on_ready,
                ).await.map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(e.to_string()))
            }
        })
    }
}

/// Short hex fingerprint of a 32-byte node_id for logs (never the full key).
fn hex_short(id: &[u8; 32]) -> String {
    format!("{:02x}{:02x}{:02x}{:02x}…", id[0], id[1], id[2], id[3])
}

/// Mode for spawning services.
#[derive(Debug, Clone)]
pub enum ServiceMode {
    /// Spawn as a tokio task in the current runtime.
    Tokio,

    /// Spawn on a dedicated thread with its own tokio runtime.
    /// Useful for services with !Send types (like tch-rs tensors).
    Thread,

    /// Spawn as a subprocess.
    Subprocess {
        /// Path to the binary.
        binary: PathBuf,
    },
}

/// Unified service spawner.
///
/// Spawns `Spawnable` services with consistent lifecycle management,
/// regardless of execution mode.
///
/// # Example
///
/// ```ignore
/// use hyprstream_rpc::service::spawner::{ServiceSpawner, ProxyService};
/// use hyprstream_rpc::transport::TransportConfig;
///
/// // Spawn a REQ/REP service (RequestService implementations are directly Spawnable)
/// let service = MyRequestService::new(ctx, transport, verifying_key);
/// let spawner = ServiceSpawner::tokio();
/// let spawned = spawner.spawn(service).await?;
///
/// // Spawn an XSUB/XPUB proxy on dedicated thread
/// let proxy = ProxyService::new("events", ctx, pub_transport, sub_transport);
/// let spawner = ServiceSpawner::threaded();
/// let spawned = spawner.spawn(proxy).await?;
///
/// // Stop the service
/// spawned.stop().await?;
/// ```
pub struct ServiceSpawner {
    mode: ServiceMode,
    process_spawner: Option<ProcessSpawner>,
}

impl ServiceSpawner {
    /// Create a spawner that runs services as tokio tasks.
    pub fn tokio() -> Self {
        Self {
            mode: ServiceMode::Tokio,
            process_spawner: None,
        }
    }

    /// Create a spawner that runs services on dedicated threads.
    ///
    /// Each service gets its own thread with a single-threaded tokio runtime.
    /// Use this for services with !Send types (like tch-rs tensors).
    pub fn threaded() -> Self {
        Self {
            mode: ServiceMode::Thread,
            process_spawner: None,
        }
    }

    /// Create a spawner that runs services as subprocesses.
    ///
    /// The binary should be a `hyprstream service <name>` command.
    pub fn subprocess(binary: PathBuf) -> Self {
        Self {
            mode: ServiceMode::Subprocess { binary },
            process_spawner: Some(ProcessSpawner::new()),
        }
    }

    /// Create a subprocess spawner with a custom process spawner.
    pub fn subprocess_with(binary: PathBuf, process_spawner: ProcessSpawner) -> Self {
        Self {
            mode: ServiceMode::Subprocess { binary },
            process_spawner: Some(process_spawner),
        }
    }

    /// Get the spawning mode.
    pub fn mode(&self) -> &ServiceMode {
        &self.mode
    }

    /// Spawn any Spawnable service with registry integration.
    ///
    /// This is the unified spawning API that works with both:
    /// - Any `S: RequestService` - REQ/REP services (directly Spawnable via blanket impl)
    /// - `ProxyService` - XSUB/XPUB proxies
    ///
    /// The service is automatically registered with the EndpointRegistry and
    /// unregistered when the returned handle is dropped.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Spawn a REQ/REP service (RequestService implementations are directly Spawnable)
    /// let service = MyRequestService::new(ctx, transport, verifying_key);
    /// let spawned = spawner.spawn(service).await?;
    ///
    /// // Spawn a proxy
    /// let proxy = ProxyService::new("events", ctx, pub_transport, sub_transport);
    /// let spawned = spawner.spawn(proxy).await?;
    /// ```
    pub async fn spawn<S: Spawnable>(&self, service: S) -> Result<SpawnedService> {
        // 1. Register with EndpointRegistry (if initialized)
        let registrations = service.registrations();
        let registration = if !registrations.is_empty() {
            ServiceRegistration::multi(service.name(), registrations, None).ok()
        } else {
            None
        };

        // 2. Spawn based on mode
        match &self.mode {
            ServiceMode::Tokio => self.spawn_tokio(service, registration).await,
            ServiceMode::Thread => self.spawn_thread(service, registration).await,
            ServiceMode::Subprocess { binary } => {
                self.spawn_subprocess(service, binary.clone(), registration).await
            }
        }
    }

    /// Spawn a Spawnable as a tokio task.
    async fn spawn_tokio<S: Spawnable>(
        &self,
        service: S,
        registration: Option<ServiceRegistration>,
    ) -> Result<SpawnedService> {
        let name = service.name().to_owned();
        let name_for_spawn = name.clone();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();

        // Spawn on the blocking thread pool.
        // Spawnable::run() creates a new_current_thread runtime and calls block_on(),
        // which would block a worker thread (or panic) if called directly in tokio::spawn.
        let blocking_handle = tokio::task::spawn_blocking(move || {
            if let Err(e) = Box::new(service).run(shutdown_clone, Some(ready_tx)) {
                tracing::error!("Service {} failed: {}", name_for_spawn, e);
            }
        });

        // Wrap in tokio::spawn so ServiceHandle gets a JoinHandle<()> (not JoinHandle<Result<(), JoinError>>)
        let join_handle = tokio::spawn(async move {
            if let Err(e) = blocking_handle.await {
                tracing::error!("Service blocking task panicked: {}", e);
            }
        });

        // Wait for socket to bind before returning
        if ready_rx.await.is_err() {
            return Err(hyprstream_rpc::error::RpcError::SpawnFailed(
                "service task exited before ready".to_owned(),
            ));
        }

        // Create a handle wrapper
        let handle = hyprstream_rpc::service::ServiceHandle::from_task(join_handle, shutdown);

        Ok(SpawnedService {
            id: format!("{name}-tokio"),
            kind: ServiceKind::TokioTask {
                handle: Some(handle),
            },
            _registration: registration,
        })
    }

    /// Spawn a Spawnable on a dedicated thread.
    async fn spawn_thread<S: Spawnable>(
        &self,
        service: S,
        registration: Option<ServiceRegistration>,
    ) -> Result<SpawnedService> {
        let name = service.name().to_owned();
        let name_for_thread = name.clone();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();

        // Spawn thread
        let thread_handle = thread::Builder::new()
            .name(format!("{}-service", name))
            .spawn(move || {
                // Run the service - it will signal ready after socket binds
                if let Err(e) = Box::new(service).run(shutdown_clone, Some(ready_tx)) {
                    tracing::error!("Service {} failed: {}", name_for_thread, e);
                }
            })
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("thread spawn: {e}")))?;

        // Wait for ready signal (sent by service after socket binds)
        if ready_rx.await.is_err() {
            return Err(hyprstream_rpc::error::RpcError::SpawnFailed(
                "service thread exited before ready".to_owned(),
            ));
        }

        Ok(SpawnedService {
            id: format!("{name}-thread"),
            kind: ServiceKind::Thread {
                handle: Some(thread_handle),
                shutdown,
            },
            _registration: registration,
        })
    }

    /// Spawn a Spawnable as a subprocess with PID file tracking.
    async fn spawn_subprocess<S: Spawnable>(
        &self,
        service: S,
        binary: PathBuf,
        registration: Option<ServiceRegistration>,
    ) -> Result<SpawnedService> {
        let name = service.name().to_owned();
        let spawner = match self.process_spawner.as_ref() {
            Some(s) => s,
            None => {
                return Err(hyprstream_rpc::error::RpcError::SpawnFailed(
                    "subprocess mode requires process_spawner".to_owned(),
                ));
            }
        };

        let process_config =
            ProcessConfig::new(&name, binary).args(["service", &name]);

        let process = spawner.spawn(process_config).await?;

        // Write PID file for lifecycle management
        let pid_file = hyprstream_rpc::paths::service_pid_file(&name);
        if let Some(pid) = process.pid() {
            // Ensure runtime directory exists
            if let Some(parent) = pid_file.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(&pid_file, pid.to_string()) {
                tracing::warn!("Failed to write PID file {:?}: {}", pid_file, e);
            } else {
                tracing::debug!("Wrote PID {} to {:?}", pid, pid_file);
            }
        }

        Ok(SpawnedService {
            id: process.id.clone(),
            kind: ServiceKind::Subprocess {
                process,
                pid_file,
            },
            _registration: registration,
        })
    }

}

impl Default for ServiceSpawner {
    fn default() -> Self {
        Self::tokio()
    }
}

/// Kind of spawned service (determines cleanup behavior).
pub enum ServiceKind {
    /// Running as a tokio task.
    TokioTask {
        handle: Option<hyprstream_rpc::service::ServiceHandle>,
    },

    /// Running on a dedicated thread.
    Thread {
        handle: Option<JoinHandle<()>>,
        shutdown: Arc<Notify>,
    },

    /// Running as a subprocess with PID file tracking.
    Subprocess {
        process: SpawnedProcess,
        /// PID file path (XDG-compliant) for lifecycle management.
        pid_file: PathBuf,
    },
}

/// Handle for a spawned service.
pub struct SpawnedService {
    /// Unique identifier.
    id: String,

    /// Service kind (determines cleanup behavior).
    kind: ServiceKind,

    /// Registry registration (RAII cleanup on drop).
    /// Stored as Option to allow Drop to consume it.
    _registration: Option<ServiceRegistration>,
}

impl SpawnedService {
    /// Create a dummy handle for services that manage their own lifecycle
    /// (e.g., systemd-managed services)
    pub fn dummy() -> Self {
        Self {
            id: "dummy".to_owned(),
            kind: ServiceKind::TokioTask { handle: None },
            _registration: None,
        }
    }

    /// Create a subprocess handle
    pub fn subprocess(id: String, process: SpawnedProcess, pid_file: PathBuf) -> Self {
        Self {
            id,
            kind: ServiceKind::Subprocess {
                process,
                pid_file,
            },
            _registration: None,
        }
    }

    /// Create a thread handle
    pub fn thread(
        id: String,
        handle: Option<JoinHandle<()>>,
        shutdown: Arc<Notify>,
        registration: Option<ServiceRegistration>,
    ) -> Self {
        Self {
            id,
            kind: ServiceKind::Thread {
                handle,
                shutdown,
            },
            _registration: registration,
        }
    }

    /// Get the service ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if the service is running.
    pub fn is_running(&self) -> bool {
        match &self.kind {
            ServiceKind::TokioTask { handle } => {
                handle.as_ref().map(hyprstream_rpc::service::ServiceHandle::is_running).unwrap_or(false)
            }
            ServiceKind::Thread { handle, .. } => {
                handle.as_ref().map(|h| !h.is_finished()).unwrap_or(false)
            }
            ServiceKind::Subprocess { pid_file, .. } => {
                // Check if PID file exists and process is alive (signal 0)
                if let Ok(pid_str) = std::fs::read_to_string(pid_file) {
                    if let Ok(pid) = pid_str.trim().parse::<i32>() {
                        // Signal 0 checks if process exists without sending a signal
                        return nix::sys::signal::kill(
                            nix::unistd::Pid::from_raw(pid),
                            None,
                        )
                        .is_ok();
                    }
                }
                false
            }
        }
    }

    /// Stop the service.
    ///
    /// Idempotent: subsequent calls are no-ops if already stopped.
    pub async fn stop(&mut self) -> Result<()> {
        match &mut self.kind {
            ServiceKind::TokioTask { handle } => {
                if let Some(mut h) = handle.take() {
                    h.stop().await;
                }
            }
            ServiceKind::Thread { handle, shutdown } => {
                // Signal shutdown
                shutdown.notify_one();

                // Wait for thread to finish
                if let Some(h) = handle.take() {
                    let _ = h.join();
                }
            }
            ServiceKind::Subprocess { process, pid_file } => {
                // Read PID from file and send SIGTERM
                if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = pid_str.trim().parse::<i32>() {
                        tracing::info!("Sending SIGTERM to subprocess {} (PID {})", process.id, pid);
                        if let Err(e) = nix::sys::signal::kill(
                            nix::unistd::Pid::from_raw(pid),
                            nix::sys::signal::Signal::SIGTERM,
                        ) {
                            tracing::warn!("Failed to send SIGTERM to PID {}: {}", pid, e);
                        }
                    }
                }
                // Clean up PID file
                if let Err(e) = std::fs::remove_file(&pid_file) {
                    tracing::debug!("Failed to remove PID file {:?}: {}", pid_file, e);
                }
            }
        }

        tracing::info!("Service {} stopped", self.id);
        Ok(())
    }
}

// ============================================================================
// InprocManager - ServiceManager for in-process spawning
// ============================================================================

use crate::service::manager::ServiceManager;
use async_trait::async_trait;

/// In-process service manager
///
/// Spawns services in the current process using ServiceSpawner.
pub struct InprocManager {
    #[allow(dead_code)]
    spawner: ServiceSpawner,
}

impl InprocManager {
    /// Create a new InprocManager with threaded spawner
    pub fn new() -> Self {
        Self {
            spawner: ServiceSpawner::threaded(),
        }
    }

    /// Create with a custom spawner mode
    pub fn with_spawner(spawner: ServiceSpawner) -> Self {
        Self { spawner }
    }
}

impl Default for InprocManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ServiceManager for InprocManager {
    async fn install(&self, _service: &str) -> anyhow::Result<()> {
        // No-op for inproc
        Ok(())
    }

    async fn uninstall(&self, _service: &str) -> anyhow::Result<()> {
        Ok(())
    }

    async fn start(&self, _service: &str) -> anyhow::Result<()> {
        // Services spawned via spawn(), not start()
        Ok(())
    }

    async fn stop(&self, _service: &str) -> anyhow::Result<()> {
        // Services managed via SpawnedService handles
        Ok(())
    }

    async fn is_active(&self, _service: &str) -> anyhow::Result<bool> {
        // Always true for inproc (services managed by handles)
        Ok(true)
    }

    async fn reload(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn spawn(&self, service: Box<dyn Spawnable>) -> anyhow::Result<SpawnedService> {
        // Can't call spawn(*service) because trait objects aren't Sized
        // Need to spawn inline instead
        let name = service.name().to_owned();
        let registrations = service.registrations();

        // Register with EndpointRegistry
        let _registration = if !registrations.is_empty() {
            ServiceRegistration::multi(&name, registrations, None).ok()
        } else {
            None
        };

        // Spawn on dedicated thread
        // The ready signal is sent by the service after the socket binds
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();
        let name_clone = name.clone(); // Clone for closure use

        let thread_handle = thread::Builder::new()
            .name(format!("{}-service", name))
            .spawn(move || {
                if let Err(e) = service.run(shutdown_clone, Some(ready_tx)) {
                    tracing::error!("Service {} failed: {}", name_clone, e);
                }
            })
            .map_err(|e| anyhow!("thread spawn: {}", e))?;

        // Wait for service to be ready (socket bound)
        // Note: systemd notification is handled inside run() after socket binds
        if ready_rx.await.is_err() {
            return Err(anyhow!("service thread exited before ready"));
        }

        Ok(SpawnedService::thread(
            format!("{name}-thread"),
            Some(thread_handle),
            shutdown,
            _registration,
        ))
    }
}

// ============================================================================
// DualSpawnable - Run two Spawnables concurrently
// ============================================================================

/// Wrapper that runs two Spawnables: primary on calling thread, secondary on a sub-thread.
///
/// Used when a service needs to listen on two transports simultaneously
/// (e.g., ZMQ and QUIC). Since `Spawnable::run()` is blocking, we can't run
/// both on the same thread.
///
/// # Example
///
/// ```ignore
/// let zmq_loop = create_zmq_loop(&ctx)?;
/// let quic_loop = QuicServiceLoop::new(quic_rep, service);
///
/// let dual = DualSpawnable::new(zmq_loop, quic_loop);
/// Ok(Box::new(dual))
/// ```
pub struct DualSpawnable {
    primary: Box<dyn Spawnable>,
    secondary: Box<dyn Spawnable>,
}

impl DualSpawnable {
    /// Create a new DualSpawnable.
    ///
    /// The primary spawnable runs on the calling thread.
    /// The secondary spawnable runs on a dedicated sub-thread.
    pub fn new(primary: Box<dyn Spawnable>, secondary: Box<dyn Spawnable>) -> Self {
        Self { primary, secondary }
    }
}

impl Spawnable for DualSpawnable {
    fn name(&self) -> &str {
        // Use primary's name as the service name
        self.primary.name()
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        // Merge registrations from both
        let mut regs = self.primary.registrations();
        regs.extend(self.secondary.registrations());
        regs
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<()> {
        let shutdown2 = shutdown.clone();
        let secondary = self.secondary;
        let secondary_name = secondary.name().to_owned();

        // Spawn secondary on a sub-thread
        let handle = thread::Builder::new()
            .name(format!("{}-quic", secondary_name))
            .spawn(move || {
                if let Err(e) = secondary.run(shutdown2, None) {
                    tracing::error!("Secondary service {} failed: {}", secondary_name, e);
                }
            })
            .map_err(|e| hyprstream_rpc::error::RpcError::Other(format!("thread spawn: {}", e)))?;

        // Run primary on current thread (blocks until shutdown)
        let result = self.primary.run(shutdown, on_ready);

        // Wait for secondary thread to finish
        let _ = handle.join();

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use hyprstream_rpc::prelude::SigningKey;
    use hyprstream_rpc::service::RequestService;
    use anyhow::Result as AnyhowResult;

    /// Test service that includes infrastructure (new pattern)
    struct EchoService {
        transport: TransportConfig,
        signing_key: SigningKey,
    }

    impl EchoService {
        fn new(transport: TransportConfig, signing_key: SigningKey) -> Self {
            Self { transport, signing_key }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl RequestService for EchoService {
        async fn handle_request(
            &self,
            _ctx: &hyprstream_rpc::service::EnvelopeContext,
            payload: &[u8],
        ) -> AnyhowResult<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
            Ok((payload.to_vec(), None))
        }

        fn name(&self) -> &str {
            "echo"
        }

        fn transport(&self) -> &TransportConfig {
            &self.transport
        }

        fn signing_key(&self) -> SigningKey {
            self.signing_key.clone()
        }
    }

    #[tokio::test]
    async fn test_tokio_spawner() -> hyprstream_rpc::Result<()> {
        let (signing_key, _verifying_key) = generate_signing_keypair();
        let transport = TransportConfig::inproc("test-spawner-tokio");

        // Service is directly Spawnable - no wrapping needed
        let service = EchoService::new(transport, signing_key);

        let spawner = ServiceSpawner::tokio();
        let mut spawned = spawner.spawn(service).await?;

        assert!(spawned.is_running());

        spawned.stop().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_thread_spawner() -> hyprstream_rpc::Result<()> {
        let (signing_key, _verifying_key) = generate_signing_keypair();
        let transport = TransportConfig::inproc("test-spawner-thread");

        // Service is directly Spawnable - no wrapping needed
        let service = EchoService::new(transport, signing_key);

        let spawner = ServiceSpawner::threaded();
        let mut spawned = spawner.spawn(service).await?;

        assert!(spawned.is_running());

        spawned.stop().await?;
        Ok(())
    }
}
