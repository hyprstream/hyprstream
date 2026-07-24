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

use crate::service::metadata::SchemaMetadataFn;
use crate::service::spawner::Spawnable;
use hyprstream_rpc::registry::{global as global_registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;

/// Complete, already-validated native announcement ready for publication.
pub struct NativeAnnouncementRequest {
    pub service_name: String,
    pub endpoint: String,
    pub signing_key: SigningKey,
    pub service_jwt: Option<String>,
    pub discovery_verifying_key: VerifyingKey,
    pub service_did: hyprstream_rpc::identity::Did,
    pub capabilities: Vec<String>,
    pub accepted_state_digest: Vec<u8>,
    pub accepted_state_epoch: u64,
    pub response_key_id: String,
    pub request_kem_key_id: String,
    pub request_kem_recipient: Vec<u8>,
    pub expires_at_unix_ms: i64,
}

pub type NativeAnnouncementPublisher =
    Arc<dyn Fn(NativeAnnouncementRequest) + Send + Sync + 'static>;

/// Complete native announcement material verified against one accepted state.
#[derive(Clone)]
pub struct NativeServiceAnnouncement {
    service_did: hyprstream_rpc::identity::Did,
    capabilities: Vec<String>,
    accepted_state_digest: [u8; 64],
    accepted_state_epoch: u64,
    accepted_state_expires_at_unix_ms: i64,
    response_key_id: String,
    response_verifying_key: [u8; 32],
    request_kem_key_id: String,
    request_kem_recipient: hyprstream_rpc::crypto::hybrid_kem::RecipientPublic,
}

impl NativeServiceAnnouncement {
    /// Project a complete native announcement from the opaque #1004 accepted
    /// state. The local service key must be the accepted current key and the
    /// named service must be present in that exact state.
    pub fn from_accepted_state(
        service_name: &str,
        signer: &SigningKey,
        state: &hyprstream_pds::at9p_duplicity::AcceptedAt9pState,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            state
                .current
                .services
                .iter()
                .any(|entry| entry.id == format!("#{service_name}")),
            "accepted state does not authorize service {service_name}"
        );
        // `subjectKeys` is a published set (#1188 / #1183): the local service
        // signer must be *one of* the accepted current response keys, not
        // positionally `first()`. An overlap-rotating identity publishes several
        // usable keys at once; a service holding any of them is authorized.
        anyhow::ensure!(
            state
                .current
                .subject_keys
                .iter()
                .any(|key| key.ed25519_pub.as_slice() == signer.verifying_key().as_bytes()),
            "service signer is not one of the accepted current response keys"
        );
        let expires_at = state.expires_at.as_deref().ok_or_else(|| {
            anyhow::anyhow!("genesis-only accepted state has no bounded production expiry")
        })?;
        let expires_at = chrono::DateTime::parse_from_rfc3339(expires_at)?.timestamp_millis();
        let did = hyprstream_rpc::identity::Did::from(state.did.clone());
        let recipient = hyprstream_rpc::node_identity::derive_mesh_kem_recipient(signer)?.public();
        let announcement = Self {
            service_did: did.clone(),
            capabilities: vec!["hyprstream-rpc/1".to_owned(), "hyprstream-moq/1".to_owned()],
            accepted_state_digest: state.head_digest,
            accepted_state_epoch: state.epoch,
            accepted_state_expires_at_unix_ms: expires_at,
            response_key_id: format!("{did}#response"),
            response_verifying_key: signer.verifying_key().to_bytes(),
            request_kem_key_id: format!("{did}#mesh-kem"),
            request_kem_recipient: recipient,
        };
        announcement.validate(service_name, &signer.verifying_key())?;
        Ok(announcement)
    }

    fn validate(&self, service_name: &str, signer: &VerifyingKey) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.service_did.is_did_at9p(),
            "native announcement requires did:at9p identity"
        );
        anyhow::ensure!(
            !service_name.is_empty() && self.capabilities.iter().any(|c| c == "hyprstream-rpc/1"),
            "native announcement lacks canonical service capability"
        );
        anyhow::ensure!(
            self.response_verifying_key == signer.to_bytes(),
            "accepted response key does not match service signer"
        );
        anyhow::ensure!(
            self.response_key_id
                .starts_with(&format!("{}#", self.service_did)),
            "response key id crosses service authority"
        );
        anyhow::ensure!(
            self.request_kem_key_id
                .starts_with(&format!("{}#", self.service_did)),
            "KEM key id crosses service authority"
        );
        anyhow::ensure!(
            self.request_kem_recipient.suite_id
                == hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
            "native announcement requires hybrid KEM suite"
        );
        self.request_kem_recipient.validate()?;
        anyhow::ensure!(
            self.accepted_state_epoch > 0
                && self.accepted_state_expires_at_unix_ms > chrono::Utc::now().timestamp_millis(),
            "accepted state is unbounded or expired"
        );
        Ok(())
    }
}

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
    /// #410/#282: bind an iroh substrate (ALPNs `hyprstream-rpc/1` + `moql`)
    /// as the PRIMARY production transport, in parallel to the quinn endpoint
    /// (kept for back-compat), for every QUIC-enabled service. On by default;
    /// an operator opts out via `[quic] iroh = false` to run quinn-only (legacy).
    pub iroh_enabled: bool,
    /// #358: the producer-chosen moq RELAY every QUIC-enabled service on this node
    /// rendezvouses through, in wire-reach form. `None` = direct-only. Sourced
    /// from the relay DID transport entry (default: the PDS / federation anchor)
    /// decoded by [`hyprstream_rpc::service_entry`]; see
    /// [`hyprstream_rpc::moq_stream::relay_reach_from_decoded`]. Threaded into each
    /// service's [`QuicLoopConfig`] so the spawner advertises a `Role::Relay` reach
    /// and links the origin UP to the relay.
    pub moq_relay: Option<hyprstream_rpc::stream_info::TransportConfig>,
    /// Application-owned publisher. Keeping this callback here avoids making
    /// orchestration depend on the Discovery implementation crate.
    pub native_announcement_publisher: Option<NativeAnnouncementPublisher>,
}

impl QuicSharedConfig {
    /// Build a per-service `QuicLoopConfig` with the given port.
    ///
    /// Port 0 = ephemeral (OS-assigned).
    pub fn for_service(
        &self,
        service_name: &str,
        port: u16,
    ) -> hyprstream_rpc::service::QuicLoopConfig {
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
            // #282: bind iroh in parallel when the deployment opted in.
            iroh_enabled: self.iroh_enabled,
            on_iroh_bound: None,
            // #358: thread the producer-chosen relay through so the spawner
            // advertises a Role::Relay reach + links the origin up to the relay.
            moq_relay: self.moq_relay.clone(),
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
        accepted: Option<NativeServiceAnnouncement>,
    ) -> hyprstream_rpc::service::QuicLoopConfig {
        let mut config = self.for_service(service_name, port);
        let publisher = self.native_announcement_publisher.clone();
        config.on_quic_bound = Some(Box::new(move |svc_name, addr, sn| {
            let endpoint = format!("quic://{sn}:{addr}");
            let sk = signing_key.clone();
            let jwt = service_jwt.clone();
            let accepted = accepted.clone();

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
                tracing::info!(
                    "Service JWT for '{svc_name}' needs renewal — requesting from PolicyService"
                );
            }

            let discovery_vk = discovery_verifying_key;
            let policy_vk = policy_verifying_key;
            let Some(accepted) = accepted else {
                tracing::warn!("Refusing production network announcement for '{svc_name}': accepted native identity/KEM bundle is unavailable");
                return;
            };
            if let Err(error) = accepted.validate(&svc_name, &sk.verifying_key()) {
                tracing::warn!(
                    "Refusing production network announcement for '{svc_name}': {error}"
                );
                return;
            }
            if needs_renewal {
                tracing::warn!(
                    "Service JWT for '{svc_name}' expired or near-expiry. \
                     Renewal RPC pending capnp schema update. Re-run wizard to refresh JWTs."
                );
                let _ = policy_vk;
            }
            let Some(publish) = publisher.clone() else {
                tracing::warn!("Refusing production network announcement for '{svc_name}': publisher is unavailable");
                return;
            };
            publish(NativeAnnouncementRequest {
                service_name: svc_name,
                endpoint,
                signing_key: sk,
                service_jwt: jwt,
                discovery_verifying_key: discovery_vk,
                service_did: accepted.service_did,
                capabilities: accepted.capabilities,
                accepted_state_digest: accepted.accepted_state_digest.to_vec(),
                accepted_state_epoch: accepted.accepted_state_epoch,
                response_key_id: accepted.response_key_id,
                request_kem_key_id: accepted.request_kem_key_id,
                request_kem_recipient: accepted.request_kem_recipient.encode(),
                expires_at_unix_ms: accepted.accepted_state_expires_at_unix_ms,
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

    native_announcements: HashMap<String, NativeServiceAnnouncement>,

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
    ml_dsa_verifying_keys:
        std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>>,
}

impl ServiceContext {
    /// Create a new service context.
    pub fn new(
        signing_key: SigningKey,
        verifying_key: VerifyingKey,
        ipc: bool,
        models_dir: std::path::PathBuf,
    ) -> Self {
        let identity_provider = Arc::new(hyprstream_rpc::node_identity::NodeIdentityProvider::new(
            &signing_key,
        ));
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
            native_announcements: HashMap::new(),
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
    pub fn set_ml_dsa_verifying_keys(
        &mut self,
        keys: std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>>,
    ) {
        self.ml_dsa_verifying_keys = keys;
    }

    /// Get a clone of the shared ML-DSA verifying keys Arc.
    #[allow(clippy::disallowed_types)]
    pub fn ml_dsa_verifying_keys_arc(
        &self,
    ) -> std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>> {
        self.ml_dsa_verifying_keys.clone()
    }

    /// Add a per-service independent signing key.
    ///
    /// In single-process mode, these are generated in memory at startup.
    /// In multi-process mode, these are loaded from per-service credential files.
    pub fn with_service_key(mut self, service_name: &str, signing_key: SigningKey) -> Self {
        self.service_keys
            .insert(service_name.to_owned(), signing_key);
        self
    }

    pub fn with_native_announcement(
        mut self,
        service_name: impl Into<String>,
        announcement: NativeServiceAnnouncement,
    ) -> Self {
        self.native_announcements
            .insert(service_name.into(), announcement);
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
            hyprstream_rpc::node_identity::NodeIdentityProvider::new(&new_key),
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
            &self.signing_key,
            "hyprstream-jwt-v1",
        );
        let ca_verifying_key = ca_signing_key.verifying_key();

        let mut ctx = self.with_ca_verifying_key(ca_verifying_key);

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
                ctx = ctx.with_service_key(name, root_signing_key.clone());

                // PolicyService key never expires — it IS the trust anchor.
                trust.insert(
                    root_verifying_key,
                    crate::service::trust_store::Attestation {
                        scopes: std::iter::once("policy".to_owned()).collect(),
                        subject: None,
                        jwt: None,
                        expires_at: 0,
                        attested_by: None,
                    },
                );
                continue;
            }

            // Generate independent Ed25519 keypair
            let service_key = SigningKey::generate(&mut rand::rngs::OsRng);
            let service_vk = service_key.verifying_key();

            // Issue service JWT (CA-signed certificate binding name → pubkey).
            // Set iss and aud to match PolicyService's local_issuer_url and
            // default_audience so it recognizes these CA-signed tokens as local.
            let oauth_issuer = ctx.oauth_issuer_url().map(str::to_owned);
            let mut claims =
                hyprstream_rpc::auth::Claims::new(format!("service:{name}"), now, expiry)
                    .with_cnf_jwk(service_vk.as_bytes());
            if let Some(ref iss) = oauth_issuer {
                claims = claims
                    .with_issuer(iss.clone())
                    .with_audience(Some(iss.clone()));
            }

            let jwt = hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, &ca_signing_key);

            ctx = ctx.with_service_key(name, service_key);

            // Register this service's key in the trust store.
            trust.insert(
                service_vk,
                crate::service::trust_store::Attestation {
                    scopes: std::iter::once(name.clone()).collect(),
                    subject: None,
                    jwt: Some(jwt),
                    expires_at: expiry,
                    attested_by: Some(root_verifying_key.to_bytes()),
                },
            );
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
            // Authoritative local CA key for offline service-JWT resolution
            // (no dependency on the HTTP /oauth/jwks endpoint at startup).
            let source = source.with_local_ca_key(self.jwt_verifying_key());
            std::sync::Arc::new(source)
        } else {
            let source =
                hyprstream_rpc::auth::ClusterKeySource::new(self.jwt_verifying_key(), issuer_url);
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

    /// Canonical service-owned deployment directory.
    ///
    /// This deliberately ignores the caller-provided `models_dir`: public
    /// `ServiceContext::new` creates local/factory context only and cannot
    /// select the production Discovery authority root.
    pub fn deployment_data_dir(&self) -> anyhow::Result<std::path::PathBuf> {
        deployment_data_dir()
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
    pub fn into_spawnable_quic<
        S: hyprstream_rpc::service::RequestService + Send + Sync + 'static,
    >(
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
                    // Bind the announcement JWT to the exact signer. A
                    // singleton lookup can select a sibling during overlap.
                    let trust = crate::service::trust_store::global_trust_store();
                    let signing_key = self.service_signing_key(service.name());
                    let service_jwt = trust
                        .get(&signing_key.verifying_key())
                        .and_then(|att| att.jwt);
                    let policy_vk = trust
                        .resolve_one("policy")
                        .unwrap_or_else(|| panic!("trust store has no policy key"));
                    let discovery_vk = self.service_signing_key("discovery").verifying_key();
                    if !trust.is_authorized(&discovery_vk, "discovery") {
                        panic!("trust store has no discovery key");
                    }
                    Some(shared.for_service_with_announce(
                        service.name(),
                        port,
                        signing_key,
                        service_jwt,
                        policy_vk,
                        discovery_vk,
                        self.native_announcements.get(service.name()).cloned(),
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

/// Deployment-owned registry directory shared by the checkpoint writer and
/// authenticated process bootstrap. No caller-provided context or path can
/// influence this value.
pub fn deployment_data_dir() -> anyhow::Result<std::path::PathBuf> {
    let data_dir = hyprstream_rpc::paths::try_data_dir()?
        .ok_or_else(|| anyhow::anyhow!("deployment data directory is unavailable"))?;
    anyhow::ensure!(
        data_dir.is_absolute(),
        "deployment data directory must be absolute"
    );
    Ok(data_dir.join("models/.registry"))
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
        Self {
            name,
            factory,
            schema: None,
            metadata: None,
            depends_on: &[],
        }
    }

    /// Create a new service factory with schema bytes.
    ///
    /// Called by the `#[service_factory("name", schema = "...")]` macro-generated code.
    pub const fn with_schema(
        name: &'static str,
        factory: ServiceFactoryFn,
        schema: &'static [u8],
    ) -> Self {
        Self {
            name,
            factory,
            schema: Some(schema),
            metadata: None,
            depends_on: &[],
        }
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
        Self {
            name,
            factory,
            schema: Some(schema),
            metadata: Some(metadata),
            depends_on: &[],
        }
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
#[allow(clippy::expect_used, clippy::unwrap_used)]
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

    #[test]
    fn native_announcement_rejects_incomplete_hybrid_recipient() {
        let signer = SigningKey::from_bytes(&[0x51; 32]);
        let announcement = NativeServiceAnnouncement {
            service_did: hyprstream_rpc::identity::Did::from("did:at9p:test"),
            capabilities: vec!["hyprstream-rpc/1".to_owned(), "hyprstream-moq/1".to_owned()],
            accepted_state_digest: [0x31; 64],
            accepted_state_epoch: 1,
            accepted_state_expires_at_unix_ms: i64::MAX,
            response_key_id: "did:at9p:test#response".to_owned(),
            response_verifying_key: signer.verifying_key().to_bytes(),
            request_kem_key_id: "did:at9p:test#kem".to_owned(),
            request_kem_recipient: hyprstream_rpc::crypto::hybrid_kem::RecipientPublic {
                suite_id: hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
                eks: Vec::new(),
            },
        };
        assert!(announcement
            .validate("model", &signer.verifying_key())
            .is_err());
    }

    /// #1188 / #1183: a native service announcement projects from an accepted
    /// state whose capsule publishes a SET of subject keys. The local service
    /// signer must be admitted when it is ANY published subject key, not only
    /// position 0 — an overlap-rotating identity publishes several usable keys at
    /// once. The pre-fix code compared only `subject_keys.first()`, so a service
    /// holding the second published key failed startup.
    #[test]
    fn native_announcement_admits_non_first_published_service_key() {
        use ed25519_dalek::SigningKey as EdSigningKey;
        use hyprstream_pds::at9p::{
            CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
        };
        use hyprstream_pds::at9p_duplicity::AcceptedAt9pState;
        use hyprstream_pds::at9p_gate::verify_genesis_capsule;
        use hyprstream_pds::at9p_sign::sign_capsule;
        use hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes;

        // Two service signers; k1 self-certifies, k2 is also published.
        let k1 = EdSigningKey::from_bytes(&[0x61; 32]);
        let k2 = EdSigningKey::from_bytes(&[0x62; 32]);
        let pq1 = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&k1);
        let pq2 = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&k2);
        let kp1 = HybridKeyPair::new(
            k1.verifying_key().to_bytes().to_vec(),
            ml_dsa_sk_to_vk_bytes(&pq1),
        )
        .unwrap();
        let kp2 = HybridKeyPair::new(
            k2.verifying_key().to_bytes().to_vec(),
            ml_dsa_sk_to_vk_bytes(&pq2),
        )
        .unwrap();

        let endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://reach").unwrap();
        let service = ServiceEntry::new("#model", ServiceType::NinePExport, endpoint).unwrap();
        // k1 FIRST, k2 second; self-certify with k1.
        let body = CapsuleBody::new(vec![kp1, kp2], vec![service]).unwrap();
        let genesis = sign_capsule(body, &k1, &pq1).unwrap();
        let bytes = genesis.to_dag_cbor().unwrap();
        let cid = genesis.cid512().unwrap();
        let verified = verify_genesis_capsule(&cid, &bytes).unwrap();
        let state = AcceptedAt9pState::from_verified_genesis(&verified).unwrap();

        // Genesis has no bounded successor expiry; from_accepted_state requires
        // one, so this projection is expected to reject on expiry — but it must
        // get PAST the signer-membership check for k2 first. We assert the error
        // is the expiry gate, NOT a "signer is not a current response key"
        // rejection (which is what the pre-fix positional check produced for k2).
        let signer2 = SigningKey::from_bytes(&[0x62; 32]);
        let err = match NativeServiceAnnouncement::from_accepted_state("model", &signer2, &state) {
            Ok(_) => panic!("genesis-only state has no bounded expiry"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("bounded production expiry") || msg.contains("expiry"),
            "second published key must pass the membership check and fail only on \
             the genesis expiry gate; got: {msg}"
        );
        assert!(
            !msg.contains("not one of the accepted current response keys"),
            "the second published subject key must be accepted as a member; got: {msg}"
        );

        // And a signer that is NOT published is rejected on membership.
        let outsider = SigningKey::from_bytes(&[0x63; 32]);
        let err = match NativeServiceAnnouncement::from_accepted_state("model", &outsider, &state) {
            Ok(_) => panic!("a non-member signer must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.to_string()
                .contains("not one of the accepted current response keys"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn public_context_cannot_select_production_authority() {
        const CHILD: &str = "HYPRSTREAM_TEST_AUTHORITY_CHILD";
        const EXPECTED: &str = "HYPRSTREAM_TEST_AUTHORITY_EXPECTED";
        if std::env::var_os(CHILD).is_some() {
            let expected = std::path::PathBuf::from(
                std::env::var_os(EXPECTED).expect("deployment-owned expected path"),
            );
            let first = SigningKey::from_bytes(&[0x71; 32]);
            let second = SigningKey::from_bytes(&[0x73; 32]);
            let caller_one = tempfile::tempdir().expect("caller models one");
            let caller_two = tempfile::tempdir().expect("caller models two");
            let local_one = ServiceContext::new(
                first.clone(),
                first.verifying_key(),
                false,
                caller_one.path().to_owned(),
            );
            let local_two = ServiceContext::new(
                second.clone(),
                second.verifying_key(),
                true,
                caller_two.path().to_owned(),
            );
            assert_eq!(local_one.deployment_data_dir().unwrap(), expected);
            assert_eq!(local_two.deployment_data_dir().unwrap(), expected);
            assert_ne!(local_one.models_dir(), local_two.models_dir());
            return;
        }

        let deployment = tempfile::tempdir().expect("deployment data root");
        let caller_store = tempfile::tempdir().expect("caller store");
        let caller = SigningKey::from_bytes(&[0x72; 32]);
        let expected = deployment.path().join("hyprstream/models/.registry");
        let status = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .arg("--exact")
            .arg("service::factory::tests::public_context_cannot_select_production_authority")
            .arg("--nocapture")
            .env(CHILD, "1")
            .env(EXPECTED, &expected)
            .env("XDG_DATA_HOME", deployment.path())
            .env("HYPRSTREAM__PDS__STORE_PATH", caller_store.path())
            .env(
                "HYPRSTREAM__PDS__ACCEPTANCE_IDENTITY",
                hex::encode(caller.verifying_key().to_bytes()),
            )
            .status()
            .expect("authority mutation subprocess");
        assert!(status.success(), "authority mutation subprocess failed");
    }
}
