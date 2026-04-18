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
    pub fn for_service_with_announce(
        &self,
        service_name: &str,
        port: u16,
        signing_key: hyprstream_rpc::prelude::SigningKey,
    ) -> hyprstream_rpc::service::QuicLoopConfig {
        let mut config = self.for_service(service_name, port);
        let _server_name = self.server_name.clone();
        config.on_quic_bound = Some(Box::new(move |svc_name, addr, sn| {
            let endpoint = format!("quic://{}:{}:{}", sn, addr.ip(), addr.port());
            let sk = signing_key;
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
                    let client = hyprstream_discovery::DiscoveryClient::for_service(
                        sk.clone(),
                        hyprstream_rpc::RequestIdentity::anonymous(),
                        hyprstream_rpc::node_identity::service_verifying_key(&sk, "discovery"),
                    );
                    match client.announce(&hyprstream_discovery::ServiceAnnouncement {
                        service_name: svc_name,
                        socket_kind: "quic".to_owned(),
                        endpoint,
                        service_jwt: None, // TODO(phase-2c): include service JWT when heartbeat is implemented
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
        }
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

    /// Derive a per-service signing key via HKDF.
    ///
    /// Each service gets its own Ed25519 key derived from the root key
    /// using `HKDF(root, "service:{name}")`. This provides key separation
    /// between services while maintaining a single root of trust.
    ///
    /// PolicyService keeps the root key directly (it IS the CA).
    pub fn service_signing_key(&self, service_name: &str) -> SigningKey {
        if service_name == "policy" {
            self.signing_key.clone()
        } else {
            hyprstream_rpc::node_identity::derive_purpose_key(
                &self.signing_key,
                &format!("service:{service_name}"),
            )
        }
    }

    /// Get the verifying key for a target service.
    ///
    /// PolicyService ("policy") uses the root key directly (it IS the CA).
    /// All other services use `HKDF(root, "service:{name}")` derived keys.
    pub fn service_verifying_key(&self, service_name: &str) -> VerifyingKey {
        self.service_signing_key(service_name).verifying_key()
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
                    Some(shared.for_service_with_announce(
                        service.name(),
                        port,
                        self.service_signing_key(service.name()),
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
