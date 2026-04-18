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
//!         ctx.zmq_context(),
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

use hyprstream_rpc::envelope::RequestIdentity;
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
    /// If the service JWT is close to expiry (within 2x heartbeat interval),
    /// the callback mints a fresh JWT using the CA JWT signing key before announcing.
    pub fn for_service_with_announce(
        &self,
        service_name: &str,
        port: u16,
        signing_key: hyprstream_rpc::prelude::SigningKey,
        service_jwt: Option<String>,
        root_signing_key: hyprstream_rpc::prelude::SigningKey,
    ) -> hyprstream_rpc::service::QuicLoopConfig {
        let mut config = self.for_service(service_name, port);
        let _server_name = self.server_name.clone();
        let ca_jwt_key = hyprstream_rpc::node_identity::derive_purpose_key(
            &root_signing_key, "hyprstream-jwt-v1",
        );
        config.on_quic_bound = Some(Box::new(move |svc_name, addr, sn| {
            let endpoint = format!("quic://{}:{}:{}", sn, addr.ip(), addr.port());
            let sk = signing_key.clone();
            let ca_key = ca_jwt_key.clone();
            let mut jwt = service_jwt.clone();

            // Check if JWT needs renewal (within 2 days of expiry, or missing)
            let needs_renewal = jwt.as_ref().map_or(true, |j| {
                // Decode without verification to check expiry
                let parts: Vec<&str> = j.split('.').collect();
                if parts.len() != 3 {
                    return true;
                }
                use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
                if let Ok(payload) = URL_SAFE_NO_PAD.decode(parts[1]) {
                    if let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&payload) {
                        if let Some(exp) = claims["exp"].as_i64() {
                            let now = chrono::Utc::now().timestamp();
                            return now > exp - 2 * 86_400; // Renew within 2 days of expiry
                        }
                    }
                }
                true
            });

            if needs_renewal {
                use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
                let now = chrono::Utc::now().timestamp();
                let expiry = now + 7 * 86_400; // 7 days
                let claims = hyprstream_rpc::auth::Claims::new(
                    format!("service:{svc_name}"),
                    now,
                    expiry,
                )
                .with_pub_key(URL_SAFE_NO_PAD.encode(sk.verifying_key().as_bytes()));
                jwt = Some(hyprstream_rpc::auth::jwt::encode(&claims, &ca_key));
                tracing::info!("Renewed service JWT for '{svc_name}' (expiry: {expiry})");
            }

            // Spawn async announce in a new thread since we may not be in a tokio context
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
                    // Use root key for HKDF derivation of discovery verifying key.
                    // In single-process mode, ca_key is derived from the root key so we
                    // can re-derive discovery's key. In multi-process mode with independent
                    // keys, the announce callback should receive the root key directly.
                    // For now, derive from the CA key's root (which is the same root).
                    let discovery_vk = hyprstream_rpc::node_identity::service_verifying_key(
                        &sk, "discovery",
                    );
                    let client = hyprstream_discovery::DiscoveryClient::for_service(
                        sk,
                        hyprstream_rpc::RequestIdentity::anonymous(),
                        discovery_vk,
                    );
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
    /// ZMQ context (shared across all services)
    zmq_context: Arc<zmq::Context>,

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

    /// Resolved service pubkeys for RPC client construction.
    ///
    /// Maps service name → verifying key. Populated from:
    /// - Single-process: generated alongside service_keys
    /// - Multi-process: bootstrap-pubkeys credential + discovery lookup
    pubkey_registry: HashMap<String, VerifyingKey>,

    /// CA verifying key (trust anchor for verifying service JWTs).
    ///
    /// In single-process mode, this is derived from the root key.
    /// In multi-process mode, this is loaded from the ca-pubkey credential.
    ca_verifying_key: Option<VerifyingKey>,

    /// Per-service JWTs (CA-signed certificates).
    ///
    /// In single-process mode, these are minted in memory at startup.
    /// In multi-process mode, these are loaded from per-service credential files.
    service_jwts: HashMap<String, String>,
}

impl ServiceContext {
    /// Create a new service context.
    pub fn new(
        zmq_context: Arc<zmq::Context>,
        signing_key: SigningKey,
        verifying_key: VerifyingKey,
        ipc: bool,
        models_dir: std::path::PathBuf,
    ) -> Self {
        let identity_provider = Arc::new(
            hyprstream_rpc::node_identity::NodeIdentityProvider::new(&signing_key)
        );
        Self {
            zmq_context,
            signing_key,
            verifying_key,
            identity_provider,
            ipc,
            models_dir,
            quic_shared: None,
            oauth_issuer_url: None,
            federation_key_source: None,
            service_keys: HashMap::new(),
            pubkey_registry: HashMap::new(),
            ca_verifying_key: None,
            service_jwts: HashMap::new(),
        }
    }

    /// Add a per-service independent signing key.
    ///
    /// In single-process mode, these are generated in memory at startup.
    /// In multi-process mode, these are loaded from per-service credential files.
    pub fn with_service_key(mut self, service_name: &str, signing_key: SigningKey) -> Self {
        let vk = signing_key.verifying_key();
        self.service_keys.insert(service_name.to_owned(), signing_key);
        self.pubkey_registry.insert(service_name.to_owned(), vk);
        self
    }

    /// Bulk-register service keys from an iterator.
    pub fn with_service_keys<I>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = (String, SigningKey)>,
    {
        for (name, sk) in keys {
            let vk = sk.verifying_key();
            self.service_keys.insert(name.clone(), sk);
            self.pubkey_registry.insert(name, vk);
        }
        self
    }

    /// Set the CA verifying key (trust anchor for verifying service JWTs).
    pub fn with_ca_verifying_key(mut self, key: VerifyingKey) -> Self {
        self.ca_verifying_key = Some(key);
        self
    }

    /// Bulk-register service JWTs.
    pub fn with_service_jwts(mut self, jwts: HashMap<String, String>) -> Self {
        self.service_jwts = jwts;
        self
    }

    /// Add a single service JWT.
    pub fn with_service_jwt(mut self, service_name: &str, jwt: String) -> Self {
        self.service_jwts.insert(service_name.to_owned(), jwt);
        self
    }

    /// Register a known pubkey for a service (without the signing key).
    ///
    /// Used for bootstrap services where we only have the pubkey (e.g.,
    /// loaded from bootstrap-pubkeys credential).
    pub fn with_known_pubkey(mut self, service_name: &str, verifying_key: VerifyingKey) -> Self {
        self.pubkey_registry.insert(service_name.to_owned(), verifying_key);
        self
    }

    /// Generate independent Ed25519 keypairs for all listed services and
    /// issue CA-signed service JWTs in memory.
    ///
    /// Used by single-process mode where all services run in one process.
    /// The root signing key serves as the CA key — PolicyService will use it
    /// directly (it IS the CA), while all other services get independent keys.
    ///
    /// Populates `service_keys`, `pubkey_registry`, `service_jwts`, and
    /// `ca_verifying_key` from the generated materials.
    pub fn generate_independent_service_keys(self, service_names: &[String]) -> Self {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

        let ca_signing_key = hyprstream_rpc::node_identity::derive_purpose_key(
            &self.signing_key, "hyprstream-jwt-v1",
        );
        let ca_verifying_key = ca_signing_key.verifying_key();

        let mut ctx = self
            .with_ca_verifying_key(ca_verifying_key);

        let now = chrono::Utc::now().timestamp();
        let expiry = now + 7 * 86_400; // 7 days

        for name in service_names {
            if name == "policy" {
                // PolicyService uses the root key directly (it IS the CA)
                // No need to generate an independent key or issue a JWT
                continue;
            }

            // Generate independent Ed25519 keypair
            let service_key = SigningKey::generate(&mut rand::rngs::OsRng);
            let service_vk = service_key.verifying_key();

            // Issue service JWT (CA-signed certificate binding name → pubkey)
            let claims = hyprstream_rpc::auth::Claims::new(
                format!("service:{name}"),
                now,
                expiry,
            )
            .with_pub_key(URL_SAFE_NO_PAD.encode(service_vk.as_bytes()));

            let jwt = hyprstream_rpc::auth::jwt::encode(&claims, &ca_signing_key);

            ctx = ctx
                .with_service_key(name, service_key)
                .with_service_jwt(name, jwt);
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

    /// Get the shared ZMQ context.
    pub fn zmq_context(&self) -> Arc<zmq::Context> {
        self.zmq_context.clone()
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
    /// 3. HKDF fallback (for services not yet migrated to independent keys)
    pub fn service_signing_key(&self, service_name: &str) -> SigningKey {
        if let Some(sk) = self.service_keys.get(service_name) {
            return sk.clone();
        }
        if service_name == "policy" {
            return self.signing_key.clone();
        }
        // Fallback: HKDF-derived key for single-process mode without explicit keys
        hyprstream_rpc::node_identity::derive_purpose_key(
            &self.signing_key,
            &format!("service:{service_name}"),
        )
    }

    /// Get the verifying key for a target service.
    ///
    /// Lookup order:
    /// 1. `pubkey_registry` (populated from independent keys or bootstrap pubkeys)
    /// 2. HKDF fallback
    pub fn service_verifying_key(&self, service_name: &str) -> VerifyingKey {
        if let Some(vk) = self.pubkey_registry.get(service_name) {
            return *vk;
        }
        // Fallback: derive from signing key (HKDF or root for policy)
        self.service_signing_key(service_name).verifying_key()
    }

    /// Get the service JWT for a specific service (if available).
    pub fn service_jwt(&self, service_name: &str) -> Option<&str> {
        self.service_jwts.get(service_name).map(|s| s.as_str())
    }

    /// Get the CA verifying key (trust anchor).
    pub fn ca_verifying_key(&self) -> Option<VerifyingKey> {
        self.ca_verifying_key
    }

    /// Get the verifying key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.verifying_key
    }

    /// Get the JWT verifying key (derived from root via HKDF "hyprstream-jwt-v1").
    ///
    /// This is the key that verifies all service JWTs. Published as `x_root_pubkey`
    /// in RFC 9728 metadata. Services use this to verify service JWTs from peers.
    pub fn jwt_verifying_key(&self) -> VerifyingKey {
        hyprstream_rpc::node_identity::derive_purpose_key(
            &self.signing_key,
            "hyprstream-jwt-v1",
        ).verifying_key()
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

    /// Create a typed client for a compile-time-known service.
    ///
    /// Uses `RpcClient<LocalSigner, ZmqConnection>` via the generated `connect_to()` constructor.
    pub fn rpc_client(&self, service: &str) -> std::sync::Arc<dyn hyprstream_rpc::RpcClient> {
        let endpoint = self.endpoint(service, SocketKind::Rep).to_zmq_string();
        let signer = hyprstream_rpc::signer::LocalSigner::new(
            self.signing_key.clone(),
            RequestIdentity::anonymous(),
        );
        let transport = hyprstream_rpc::zmq_connection::ZmqConnection::new(
            &endpoint,
            self.zmq_context.clone(),
        );
        let rpc = hyprstream_rpc::rpc_client::RpcClientImpl::new(
            signer, transport, self.verifying_key,
        );
        std::sync::Arc::new(rpc)
    }

    /// Wrap a ZmqService for spawning with a per-service QUIC port.
    ///
    /// - `quic_port: None` → use ephemeral port (0) when QUIC is globally enabled
    /// - `quic_port: Some(0)` → ephemeral (OS-assigned) port
    /// - `quic_port: Some(N)` → explicit port N
    ///
    /// When `[quic] enabled = true` in config, all services get QUIC on
    /// auto-assigned ephemeral ports by default. Set an explicit port to
    /// control which port a service uses.
    pub fn into_spawnable_quic<S: hyprstream_rpc::service::ZmqService + Send + Sync + 'static>(
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
                    let service_jwt = self.service_jwt(service.name()).map(|s| s.to_owned());
                    Some(shared.for_service_with_announce(
                        service.name(),
                        port,
                        self.service_signing_key(service.name()),
                        service_jwt,
                        self.signing_key.clone(),
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

    /// Wrap a ZmqService for spawning, enabling QUIC when globally configured.
    ///
    /// Uses ephemeral port (0) for QUIC when `[quic] enabled = true`.
    pub fn into_spawnable<S: hyprstream_rpc::service::ZmqService + Send + Sync + 'static>(
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
