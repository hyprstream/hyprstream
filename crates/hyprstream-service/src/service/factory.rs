//! Service factory infrastructure for inventory-based service registration.
//!
//! This module provides the `ServiceFactory` type and `ServiceContext` for
//! implementing the same inventory pattern used for `ScopeDefinition` and
//! `DriverFactory`.
//!
//! # Example
//!
//! ```ignore
//! use hyprstream_service::service::factory::{ServiceContext, ServiceFactory};
//! use hyprstream_rpc_derive::service_factory;
//!
//! #[service_factory("policy")]
//! fn create_policy_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
//!     // Services include infrastructure and are directly Spawnable
//!     let policy = PolicyService::new(
//!         ...,
//!         ctx.transport("policy", SocketKind::Rep),
//!         ctx.verifying_key(),
//!     );
//!     Ok(Box::new(policy))
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use ed25519_dalek::{SigningKey, VerifyingKey};
use zeroize::Zeroizing;

use hyprstream_rpc::registry::{global as global_registry, SocketKind};
use crate::service::metadata::SchemaMetadataFn;
use crate::service::spawner::Spawnable;
use hyprstream_rpc::transport::TransportConfig;

/// Shared QUIC/WebTransport configuration for all services.
///
/// Contains TLS materials and base settings shared across services.
/// Each service gets its own port via `for_service()`.
#[derive(Clone)]
pub struct QuicSharedConfig {
    /// DER-encoded TLS certificate chain (leaf first, then intermediates/CA)
    pub cert_chain: Vec<Vec<u8>>,
    /// DER-encoded TLS private key — zeroed on drop.
    pub key_der: Zeroizing<Vec<u8>>,
    /// Base IP address for binding (e.g., 0.0.0.0)
    pub base_ip: std::net::IpAddr,
    /// TLS server name (for certificate validation and discovery)
    pub server_name: String,
    /// OAuth issuer URL for RFC 9728 protected resource metadata
    pub oauth_issuer_url: Option<String>,
    /// JWT verifying key (derived from root via HKDF "hyprstream-jwt-v1").
    /// Published as `x_root_pubkey` in RFC 9728 metadata for client-side trust pinning.
    pub jwt_verifying_key: Option<ed25519_dalek::VerifyingKey>,
}

impl QuicSharedConfig {
    /// Build a per-service `QuicLoopConfig` with the given port.
    ///
    /// Port 0 = ephemeral (OS-assigned).
    pub fn for_service(&self, service_name: &str, port: u16) -> hyprstream_rpc::service::QuicLoopConfig {
        let bind_addr = std::net::SocketAddr::new(self.base_ip, port);
        let metadata = self.oauth_issuer_url.as_ref().map(|issuer| {
            let mut meta = serde_json::json!({
                "resource": format!("https://{}/{}", self.server_name, service_name),
                "authorization_servers": [issuer],
                "bearer_methods_supported": ["header"],
            });
            // Publish root pubkey for client-side trust pinning (TOFU)
            if let Some(ref vk) = self.jwt_verifying_key {
                use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
                #[allow(clippy::unwrap_used)] // meta is always a JSON object
                meta.as_object_mut().unwrap().insert(
                    "x_root_pubkey".to_owned(),
                    serde_json::Value::String(URL_SAFE_NO_PAD.encode(vk.to_bytes())),
                );
            }
            meta.to_string().into_bytes()
        });
        hyprstream_rpc::service::QuicLoopConfig {
            cert_chain: self.cert_chain.clone(),
            key_der: Zeroizing::new((*self.key_der).clone()),
            bind_addr,
            server_name: self.server_name.clone(),
            protected_resource_json: metadata,
            on_quic_bound: None,
        }
    }

    /// Build a per-service `QuicLoopConfig` with an announce callback.
    ///
    /// After binding, the callback announces the QUIC endpoint to the DiscoveryService.
    ///
    /// If the service JWT is close to expiry, the callback requests a renewed JWT
    /// from PolicyService via the `issueToken` RPC (no local CA key needed).
    pub fn for_service_with_announce(
        &self,
        service_name: &str,
        port: u16,
        signing_key: hyprstream_rpc::prelude::SigningKey,
        service_jwt: Option<String>,
        policy_verifying_key: VerifyingKey,
        discovery_verifying_key: VerifyingKey,
    ) -> hyprstream_rpc::service::QuicLoopConfig {
        let mut config = self.for_service(service_name, port);
        config.on_quic_bound = Some(Box::new(move |svc_name, addr, sn| {
            let endpoint = format!("quic://{}:{}:{}", sn, addr.ip(), addr.port());
            let sk = signing_key.clone();
            let jwt = service_jwt.clone();

            // Check if JWT needs renewal (within 2 days of expiry, or missing)
            let needs_renewal = jwt.as_ref().is_none_or(|j| {
                let parts: Vec<&str> = j.split('.').collect();
                if parts.len() != 3 {
                    return true;
                }
                use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
                if let Ok(payload) = URL_SAFE_NO_PAD.decode(parts[1]) {
                    if let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&payload) {
                        if let Some(exp) = claims["exp"].as_i64() {
                            let now = chrono::Utc::now().timestamp();
                            return now > exp - 2 * 86_400;
                        }
                    }
                }
                true
            });

            if needs_renewal {
                tracing::info!("Service JWT for '{svc_name}' needs renewal — requesting from PolicyService");
            }

            // Spawn async announce in a new thread since we may not be in a tokio context
            let discovery_vk = discovery_verifying_key;
            let policy_vk = policy_verifying_key;
            std::thread::spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::warn!("Failed to create announce runtime: {}", e);
                        return;
                    }
                };
                rt.block_on(async {
                    // Request JWT renewal from PolicyService if needed
                    if needs_renewal {
                        // TODO: Call PolicyService issueToken RPC for renewal.
                        // Requires adding existingToken field to policy.capnp and
                        // regenerating codegen. For now, log a warning.
                        tracing::warn!(
                            "Service JWT for '{svc_name}' expired or near-expiry. \
                             Renewal RPC pending capnp schema update. \
                             Re-run wizard to refresh JWTs."
                        );
                        let _ = policy_vk; // suppress unused warning
                    }

                    let client = match hyprstream_discovery::DiscoveryClient::for_service(
                        sk,
                        discovery_vk,
                        None,
                    ) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!("Failed to build DiscoveryClient: {}", e);
                            return;
                        }
                    };
                    match client.announce(&hyprstream_discovery::ServiceAnnouncement {
                        service_name: svc_name,
                        socket_kind: "quic".to_owned(),
                        endpoint,
                        service_jwt: jwt,
                    }).await {
                        Ok(_) => tracing::info!("Announced QUIC endpoint to DiscoveryService"),
                        Err(e) => tracing::warn!("Failed to announce QUIC endpoint: {}", e),
                    }
                });
            });
        }));
        config
    }
}

/// Context for service creation.
///
/// Contains all shared resources needed by services during initialization.
/// Passed to factory functions registered via `#[service_factory]`.
pub struct ServiceContext {
    /// Server's signing key (for JWT generation)
    signing_key: SigningKey,

    /// Server's verifying key (for envelope/JWT verification)
    verifying_key: VerifyingKey,

    /// Identity provider for purpose-keyed signing
    identity_provider: Arc<hyprstream_rpc::node_identity::NodeIdentityProvider>,

    /// Whether running in IPC mode (vs inproc)
    ipc: bool,

    /// Models directory path
    models_dir: std::path::PathBuf,

    /// Shared QUIC/WebTransport config (TLS materials + base settings).
    /// Per-service ports are resolved via `into_spawnable_quic()`.
    quic_shared: Option<QuicSharedConfig>,

    /// OAuth issuer URL for protected resource metadata (RFC 9728).
    /// When set, QUIC services serve `.well-known/oauth-protected-resource`.
    oauth_issuer_url: Option<String>,

    /// Shared federation key resolver (None when no trusted_issuers are configured).
    federation_key_source: Option<Arc<dyn hyprstream_rpc::auth::FederationKeySource>>,

    /// Per-service signing keys (independent Ed25519 keypairs).
    ///
    /// Populated from:
    /// - Single-process: generated in memory at startup
    /// - Multi-process: loaded from per-service credential files
    service_keys: HashMap<String, SigningKey>,

    /// CA verifying key (trust anchor for verifying service JWTs).
    ///
    /// In single-process mode, this is derived from the root key.
    /// In multi-process mode, this is loaded from the ca-pubkey credential.
    ca_verifying_key: Option<VerifyingKey>,

    /// Optional JWKS fetcher for JWKS-backed key resolution.
    /// When set, `cluster_key_source()` returns `JwksKeySource(Mode::Isolated)`
    /// instead of `ClusterKeySource`.
    jwks_fetcher: Option<hyprstream_rpc::auth::JwksFetcher>,

    /// Shared ML-DSA-65 verifying keys for PQ-hybrid JWT verification.
    /// Updated by the rotation task; shared across all key sources.
    ///
    /// Uses `std::sync::RwLock` to match the cross-crate `Arc<RwLock<..>>`
    /// contract with `hyprstream::auth::key_rotation` and `JwtKeySource`.
    #[allow(clippy::disallowed_types)]
    ml_dsa_verifying_keys: std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>>,
}

impl ServiceContext {
    /// Create a new service context.
    pub fn new(
        signing_key: SigningKey,
        verifying_key: VerifyingKey,
        ipc: bool,
        models_dir: std::path::PathBuf,
    ) -> Self {
        let identity_provider = Arc::new(
            hyprstream_rpc::node_identity::NodeIdentityProvider::new(&signing_key)
        );
        Self {
            signing_key,
            verifying_key,
            identity_provider,
            ipc,
            models_dir,
            quic_shared: None,
            oauth_issuer_url: None,
            federation_key_source: None,
            service_keys: HashMap::new(),
            ca_verifying_key: None,
            jwks_fetcher: None,
            ml_dsa_verifying_keys: {
                #[allow(clippy::disallowed_types)]
                std::sync::Arc::new(std::sync::RwLock::new(Vec::new()))
            },
        }
    }

    /// Set the shared ML-DSA-65 verifying keys for PQ-hybrid JWT verification.
    #[allow(clippy::disallowed_types)]
    pub fn set_ml_dsa_verifying_keys(&mut self, keys: std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>>) {
        self.ml_dsa_verifying_keys = keys;
    }

    /// Get a clone of the shared ML-DSA verifying keys Arc.
    #[allow(clippy::disallowed_types)]
    pub fn ml_dsa_verifying_keys_arc(&self) -> std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>> {
        self.ml_dsa_verifying_keys.clone()
    }

    /// Add a per-service independent signing key.
    ///
    /// In single-process mode, these are generated in memory at startup.
    /// In multi-process mode, these are loaded from per-service credential files.
    pub fn with_service_key(mut self, service_name: &str, signing_key: SigningKey) -> Self {
        self.service_keys.insert(service_name.to_owned(), signing_key);
        self
    }

    /// Bulk-register service keys from an iterator.
    pub fn with_service_keys<I>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = (String, SigningKey)>,
    {
        for (name, sk) in keys {
            self.service_keys.insert(name, sk);
        }
        self
    }

    /// Set the CA verifying key (trust anchor for verifying service JWTs).
    pub fn with_ca_verifying_key(mut self, key: VerifyingKey) -> Self {
        self.ca_verifying_key = Some(key);
        self
    }

    /// Swap the signing key to an independent per-service key.
    ///
    /// Used in IPC mode for non-policy services: replaces the root/CA key
    /// with the service's own independent Ed25519 key. The CA key is no
    /// longer accessible via `signing_key()` after this call.
    pub fn swap_signing_key(mut self, new_key: SigningKey) -> Self {
        self.verifying_key = new_key.verifying_key();
        self.identity_provider = Arc::new(
            hyprstream_rpc::node_identity::NodeIdentityProvider::new(&new_key)
        );
        self.signing_key = new_key;
        self
    }

    /// Generate independent Ed25519 keypairs for all listed services and
    /// issue CA-signed service JWTs in memory.
    ///
    /// Used by single-process mode where all services run in one process.
    /// The root signing key serves as the CA key — PolicyService will use it
    /// directly (it IS the CA), while all other services get independent keys.
    ///
    /// Populates `service_keys`, `ca_verifying_key`, and the global trust store
    /// from the generated materials.
    pub fn generate_independent_service_keys(self, service_names: &[String]) -> Self {

        let ca_signing_key = hyprstream_rpc::node_identity::derive_purpose_key(
            &self.signing_key, "hyprstream-jwt-v1",
        );
        let ca_verifying_key = ca_signing_key.verifying_key();

        let mut ctx = self
            .with_ca_verifying_key(ca_verifying_key);

        let now = chrono::Utc::now().timestamp();
        let expiry = now + 7 * 86_400; // 7 days

        // Capture root signing/verifying keys before ctx is moved in the loop.
        let root_signing_key = ctx.signing_key.clone();
        let root_verifying_key = ctx.verifying_key();

        // Populate the global trust store with all service keys.
        // The trust store is the source of truth for key-centric identity:
        // keys ARE identity, service names are authorization scopes.
        let trust = crate::service::trust_store::global_trust_store();

        // Remove any stale entries for services we're about to regenerate.
        // This happens when CLI mode loaded bootstrap pubkeys from disk (no JWTs)
        // before the service start handler generates fresh in-memory keys with JWTs.
        for name in service_names {
            if let Some(stale_vk) = trust.resolve_one(name) {
                let stale = trust.get(&stale_vk);
                if stale.as_ref().is_some_and(|a| a.jwt.is_none()) {
                    tracing::debug!(service = name, "Removing stale trust store entry (no JWT)");
                    trust.remove(&stale_vk);
                }
            }
        }

        for name in service_names {
            if name == "policy" {
                // PolicyService uses the root key directly (it IS the CA).
                // Register its signing key in the registry
                // so service_signing_key("policy") works.
                ctx = ctx
                    .with_service_key(name, root_signing_key.clone());

                // PolicyService key never expires — it IS the trust anchor.
                trust.insert(root_verifying_key, crate::service::trust_store::Attestation {
                    scopes: std::iter::once("policy".to_owned()).collect(),
                    subject: None,
                    jwt: None,
                    expires_at: 0,
                    attested_by: None,
                });
                continue;
            }

            // Generate independent Ed25519 keypair
            let service_key = SigningKey::generate(&mut rand::rngs::OsRng);
            let service_vk = service_key.verifying_key();

            // Issue service JWT (CA-signed certificate binding name → pubkey).
            // Set iss and aud to match PolicyService's local_issuer_url and
            // default_audience so it recognizes these CA-signed tokens as local.
            let oauth_issuer = ctx.oauth_issuer_url().map(str::to_owned);
            let mut claims = hyprstream_rpc::auth::Claims::new(
                format!("service:{name}"),
                now,
                expiry,
            )
            .with_cnf_jwk(service_vk.as_bytes());
            if let Some(ref iss) = oauth_issuer {
                claims = claims
                    .with_issuer(iss.clone())
                    .with_audience(Some(iss.clone()));
            }

            let jwt = hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, &ca_signing_key);

            ctx = ctx
                .with_service_key(name, service_key);

            // Register this service's key in the trust store.
            trust.insert(service_vk, crate::service::trust_store::Attestation {
                scopes: std::iter::once(name.clone()).collect(),
                subject: None,
                jwt: Some(jwt),
                expires_at: expiry,
                attested_by: Some(root_verifying_key.to_bytes()),
            });
        }

        ctx
    }

    /// Set the shared QUIC/WebTransport configuration.
    ///
    /// Per-service ports are resolved via `into_spawnable_quic()`.
    pub fn with_quic(mut self, config: QuicSharedConfig) -> Self {
        self.quic_shared = Some(config);
        self
    }

    /// Get the shared QUIC config (if enabled).
    pub fn quic_shared(&self) -> Option<&QuicSharedConfig> {
        self.quic_shared.as_ref()
    }

    /// Check if QUIC/WebTransport is enabled.
    pub fn has_quic(&self) -> bool {
        self.quic_shared.is_some()
    }

    /// Set the OAuth issuer URL for RFC 9728 metadata.
    pub fn with_oauth_issuer(mut self, url: String) -> Self {
        self.oauth_issuer_url = Some(url);
        self
    }

    /// Get the OAuth issuer URL (if configured).
    pub fn oauth_issuer_url(&self) -> Option<&str> {
        self.oauth_issuer_url.as_deref()
    }

    /// Get the federation key source (if configured).
    pub fn federation_key_source(
        &self,
    ) -> Option<Arc<dyn hyprstream_rpc::auth::FederationKeySource>> {
        self.federation_key_source.clone()
    }

    /// Set the shared federation key source for multi-issuer ZMQ token acceptance.
    pub fn with_federation_key_source(
        mut self,
        src: Arc<dyn hyprstream_rpc::auth::FederationKeySource>,
    ) -> Self {
        self.federation_key_source = Some(src);
        self
    }

    /// Get the root signing key.
    ///
    /// **Only PolicyService (the CA) should call this.** All other services must
    /// use `service_signing_key("name")` for per-service key derivation. The root
    /// key is needed by PolicyService to issue service JWTs and by node-level
    /// operations (main.rs) for QuicSharedConfig construction.
    ///
    /// If you're writing a factory function and the service name is not "policy",
    /// use `service_signing_key(service_name)` instead.
    #[inline]
    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    /// Get the signing key for a specific service.
    ///
    /// Lookup order:
    /// 1. `service_keys` registry (independent keypair per service)
    /// 2. PolicyService special case (returns root key — it IS the CA)
    /// 3. "multi" fallback (multi-service IPC mode shares one key for all services)
    ///
    /// Panics if no key is registered. Ensure `generate_independent_service_keys()`
    /// was called (inproc) or the service's own key was loaded from credentials (IPC).
    pub fn service_signing_key(&self, service_name: &str) -> SigningKey {
        if let Some(sk) = self.service_keys.get(service_name) {
            return sk.clone();
        }
        if service_name == "policy" {
            return self.signing_key.clone();
        }
        // Multi-service IPC mode registers a single "multi" key; fall back to it.
        if let Some(sk) = self.service_keys.get("multi") {
            return sk.clone();
        }
        panic!(
            "service_signing_key({service_name}): no independent key registered. \
             Ensure generate_independent_service_keys() was called (inproc) or \
             the service's signing-key credential was loaded (IPC mode)."
        );
    }

    /// Get the CA verifying key (trust anchor).
    pub fn ca_verifying_key(&self) -> Option<VerifyingKey> {
        self.ca_verifying_key
    }

    /// Get the verifying key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.verifying_key
    }

    /// Get the JWT verifying key (CA verifying key — trust anchor).
    ///
    /// This is the key that verifies all service JWTs. Published as `x_root_pubkey`
    /// in RFC 9728 metadata. Services use this to verify service JWTs from peers.
    pub fn jwt_verifying_key(&self) -> VerifyingKey {
        match self.ca_verifying_key {
            Some(k) => k,
            None => panic!(
                "jwt_verifying_key: ca_verifying_key not set. \
                 Ensure ca-pubkey credential was loaded (IPC mode) or \
                 generate_independent_service_keys() was called (inproc mode)."
            ),
        }
    }

    /// Create a key source for regular services.
    ///
    /// When a JWKS fetcher is configured, returns `JwksKeySource(Mode::Isolated)`
    /// with kid-based resolution. Otherwise falls back to `ClusterKeySource`
    /// (single hardcoded CA key).
    ///
    /// # Panics
    ///
    /// Panics if `ca_verifying_key` is not set.
    pub fn cluster_key_source(&self) -> std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource> {
        let issuer_url = self.oauth_issuer_url().unwrap_or_default().to_owned();

        if let Some(ref fetcher) = self.jwks_fetcher {
            let jwks_url = format!("{}/oauth/jwks", issuer_url.trim_end_matches('/'));
            let source = hyprstream_rpc::auth::JwksKeySource::new(
                hyprstream_rpc::auth::JwksMode::Isolated { jwks_url },
                issuer_url,
                fetcher.clone(),
            );
            let source = source.with_ml_dsa_verifying_keys(self.ml_dsa_verifying_keys.clone());
            std::sync::Arc::new(source)
        } else {
            let source = hyprstream_rpc::auth::ClusterKeySource::new(
                self.jwt_verifying_key(),
                issuer_url,
            );
            let source = source.with_ml_dsa_verifying_keys(self.ml_dsa_verifying_keys.clone());
            std::sync::Arc::new(source)
        }
    }

    /// Set the JWKS fetcher for JWKS-backed key resolution.
    ///
    /// When set, `cluster_key_source()` returns `JwksKeySource(Mode::Isolated)`
    /// instead of `ClusterKeySource`, enabling kid-based key selection from
    /// the local `/oauth/jwks` endpoint.
    pub fn set_jwks_fetcher(&mut self, fetcher: hyprstream_rpc::auth::JwksFetcher) {
        self.jwks_fetcher = Some(fetcher);
    }

    /// Get the identity provider for purpose-keyed signing.
    pub fn identity_provider(&self) -> &Arc<hyprstream_rpc::node_identity::NodeIdentityProvider> {
        &self.identity_provider
    }

    /// Check if running in IPC mode.
    pub fn is_ipc(&self) -> bool {
        self.ipc
    }

    /// Get models directory path.
    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    /// Get transport config for a service endpoint from the registry.
    ///
    /// This looks up the endpoint from the global EndpointRegistry.
    pub fn endpoint(&self, service: &str, kind: SocketKind) -> TransportConfig {
        global_registry().endpoint(service, kind)
    }

    /// Get unified transport config for a service.
    ///
    /// In IPC mode, returns a Unix socket path in the runtime directory.
    /// In inproc mode, returns the endpoint from the global registry.
    ///
    /// This unifies the transport resolution logic that was previously
    /// duplicated across factory functions.
    pub fn transport(&self, service: &str, kind: SocketKind) -> TransportConfig {
        if self.ipc {
            let runtime_dir = hyprstream_rpc::paths::runtime_dir();
            TransportConfig::ipc(runtime_dir.join(format!("{service}.sock")))
        } else {
            global_registry().endpoint(service, kind)
        }
    }

    /// Wrap a RequestService for spawning with a per-service QUIC port.
    ///
    /// - `quic_port: None` → use ephemeral port (0) when QUIC is globally enabled
    /// - `quic_port: Some(0)` → ephemeral (OS-assigned) port
    /// - `quic_port: Some(N)` → explicit port N
    ///
    /// When `[quic] enabled = true` in config, all services get QUIC on
    /// auto-assigned ephemeral ports by default. Set an explicit port to
    /// control which port a service uses.
    pub fn into_spawnable_quic<S: hyprstream_rpc::service::RequestService + Send + Sync + 'static>(
        &self,
        service: S,
        quic_port: Option<u16>,
    ) -> Box<dyn Spawnable> {
        let quic = match &self.quic_shared {
            Some(shared) => {
                let port = quic_port.unwrap_or(0);
                // Use announce callback for all services except discovery itself
                // (discovery can't announce to itself)
                if service.name() == "discovery" {
                    Some(shared.for_service(service.name(), port))
                } else {
                    // Read JWT from trust store attestation
                    let trust = crate::service::trust_store::global_trust_store();
                    let service_jwt = trust.resolve_one(service.name())
                        .and_then(|vk| trust.get(&vk).and_then(|att| att.jwt.clone()));
                    let policy_vk = trust.resolve_one("policy")
                        .unwrap_or_else(|| panic!("trust store has no policy key"));
                    let discovery_vk = crate::service::trust_store::global_trust_store()
                        .resolve_one("discovery")
                        .unwrap_or_else(|| panic!("trust store has no discovery key"));
                    Some(shared.for_service_with_announce(
                        service.name(),
                        port,
                        self.service_signing_key(service.name()),
                        service_jwt,
                        policy_vk,
                        discovery_vk,
                    ))
                }
            }
            None => None,
        };
        if quic.is_some() {
            Box::new(crate::service::spawner::UnifiedServiceConfig::new(
                service, quic,
            ))
        } else {
            Box::new(service)
        }
    }

    /// Wrap a RequestService for spawning, enabling QUIC when globally configured.
    ///
    /// Uses ephemeral port (0) for QUIC when `[quic] enabled = true`.
    pub fn into_spawnable<S: hyprstream_rpc::service::RequestService + Send + Sync + 'static>(
        &self,
        service: S,
    ) -> Box<dyn Spawnable> {
        self.into_spawnable_quic(service, None)
    }
}

// ServiceClient trait removed — generated clients use Arc<dyn RpcClient> directly.

/// Factory function signature for creating services.
///
/// Takes a `ServiceContext` and returns a boxed `Spawnable` service.
pub type ServiceFactoryFn = fn(&ServiceContext) -> anyhow::Result<Box<dyn Spawnable>>;

/// Service factory for inventory-based registration.
///
/// Services register their factory function using `#[service_factory("name")]`,
/// which generates an `inventory::submit!` for this type.
///
/// # Pattern
///
/// Same pattern as:
/// - `ScopeDefinition` with `#[register_scopes]` for authorization scopes
/// - `DriverFactory` in git2db for storage drivers
pub struct ServiceFactory {
    /// Service name (matches config.services.startup entries)
    pub name: &'static str,

    /// Factory function that creates the service
    pub factory: ServiceFactoryFn,

    /// Raw `.capnp` schema bytes (compile-time embedded via `include_bytes!`)
    pub schema: Option<&'static [u8]>,

    /// Schema metadata function for compile-time scope discovery.
    ///
    /// When set, returns `(service_name, &[MethodMeta])` derived from Cap'n Proto
    /// schema annotations. Used by PolicyService to discover supported scopes.
    pub metadata: Option<SchemaMetadataFn>,

    /// Names of services that must be started before this one.
    pub depends_on: &'static [&'static str],
}

impl ServiceFactory {
    /// Create a new service factory (without schema).
    ///
    /// Called by the `#[service_factory]` macro-generated code.
    pub const fn new(name: &'static str, factory: ServiceFactoryFn) -> Self {
        Self { name, factory, schema: None, metadata: None, depends_on: &[] }
    }

    /// Create a new service factory with schema bytes.
    ///
    /// Called by the `#[service_factory("name", schema = "...")]` macro-generated code.
    pub const fn with_schema(name: &'static str, factory: ServiceFactoryFn, schema: &'static [u8]) -> Self {
        Self { name, factory, schema: Some(schema), metadata: None, depends_on: &[] }
    }

    /// Create a new service factory with schema bytes and metadata.
    ///
    /// Called by the `#[service_factory("name", schema = "...", metadata = ...)]` macro-generated code.
    pub const fn with_metadata(
        name: &'static str,
        factory: ServiceFactoryFn,
        schema: &'static [u8],
        metadata: SchemaMetadataFn,
    ) -> Self {
        Self { name, factory, schema: Some(schema), metadata: Some(metadata), depends_on: &[] }
    }

    /// Set service dependencies (chained builder).
    ///
    /// Services listed in `depends_on` must be started before this one.
    /// Used by the `#[service_factory("name", depends_on = ["policy"])]` macro.
    pub const fn with_depends_on(mut self, deps: &'static [&'static str]) -> Self {
        self.depends_on = deps;
        self
    }
}

// Collect all registered factories
inventory::collect!(ServiceFactory);

/// Get a service factory by name.
///
/// Looks up the factory from compile-time registered factories.
///
/// # Example
///
/// ```ignore
/// let factory = get_factory("policy").ok_or_else(|| anyhow!("Unknown service: policy"))?;
/// let spawnable = (factory.factory)(&ctx)?;
/// manager.spawn(spawnable).await?;
/// ```
pub fn get_factory(name: &str) -> Option<&'static ServiceFactory> {
    inventory::iter::<ServiceFactory>().find(|f| f.name == name)
}

/// List all registered service factories.
///
/// Useful for introspection and help text.
pub fn list_factories() -> impl Iterator<Item = &'static ServiceFactory> {
    inventory::iter::<ServiceFactory>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_factory_creation() {
        fn dummy_factory(_ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
            Err(anyhow::anyhow!("dummy"))
        }

        let factory = ServiceFactory::new("test", dummy_factory);
        assert_eq!(factory.name, "test");
    }
}
