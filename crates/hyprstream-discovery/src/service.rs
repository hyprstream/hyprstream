//! Discovery service — exposes EndpointRegistry over RPC.
//!
//! Allows remote clients to discover registered services, their endpoints,
//! socket kinds, and schemas via the standard REQ/REP transport.

use async_trait::async_trait;
use hyprstream_rpc::service::{EnvelopeContext, RequestService};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::registry::{self, EndpointRegistry, SocketKind};
use hyprstream_rpc::resolver::Resolver;
use hyprstream_rpc::SigningKey;

use crate::generated::discovery_client::{
    DiscoveryHandler, DiscoveryResponseVariant,
    ErrorInfo, ServiceList, ServiceSummary, ServiceEndpoints, EndpointInfo,
    PingInfo, AuthMetadata, AuthMetadataList, ServiceAnnouncement,
    RegisterEntityStatementRequest, RegisterEnvelopeKeysetRequest,
    EntityStatement, EnvelopeKeyset, IssuerList,
    GetRecordRequest, RecordCar,
    QueryCandidatesRequest,
    dispatch_discovery, serialize_response,
};

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use parking_lot::RwLock;
use tracing::{trace, info};

// ============================================================================
// AuthorizationProvider trait (D5)
// ============================================================================

/// Trait for pluggable authorization checks.
///
/// Decouples DiscoveryService from PolicyClient. The main `hyprstream` crate
/// provides a `PolicyAuthProvider` that wraps `PolicyClient`.
#[async_trait(?Send)]
pub trait AuthorizationProvider: Send + Sync {
    /// Check if a subject is authorized for the given operation on a resource.
    async fn check(
        &self,
        subject: &str,
        domain: &str,
        resource: &str,
        operation: &str,
    ) -> Result<bool>;
}

// ============================================================================
// RecordResolver trait (#431)
// ============================================================================

/// A record returned as a CARv1 proof (the in-memory form of `RecordCar`).
///
/// `uri` is the at:// URI this CAR answers; `car` is the CARv1 bytes
/// (roots = [commit CID]; blocks = commit + MST path + record). The caller
/// validates it offline via `hyprstream_pds::car::verify_record_proof` — the
/// DiscoveryService is an untrusted relay, so integrity comes from the signed
/// proof, never from trusting the responder.
#[derive(Clone, Debug)]
pub struct RecordCarData {
    pub uri: String,
    pub car: Vec<u8>,
}

/// Trait for resolving atproto records/repos from the node's local PDS.
///
/// Decouples DiscoveryService from `hyprstream-pds` (a pure crypto/metadata
/// crate with no networking) and from the per-account record stores held by the
/// main `hyprstream` crate. The main crate provides a `PdsRecordResolver` that
/// builds CAR proofs via `hyprstream_pds::car::build_record_proof_car`.
///
/// IMPORTANT: a resolver MUST NOT perform access control — confidentiality is
/// enforced by the DiscoveryService handler's Casbin check on the target
/// DID/collection *before* the resolver is consulted. The resolver only answers
/// "does this record exist, and what is its signed CAR proof".
#[async_trait(?Send)]
pub trait RecordResolver: Send + Sync {
    /// Resolve a single record to a CAR proof. `collection` is the NSID
    /// (e.g. `ai.hyprstream.model`), `rkey` the TID record key. Returns
    /// `Ok(None)` when the DID/collection/rkey does not name a stored record.
    async fn resolve_record(
        &self,
        did: &str,
        collection: &str,
        rkey: &str,
    ) -> Result<Option<RecordCarData>>;

    /// Resolve a full repo CAR for a DID (commit + MST + all records).
    /// Returns `Ok(None)` when no repo is stored for the DID.
    async fn resolve_repo(&self, did: &str) -> Result<Option<RecordCarData>>;
}

// ============================================================================
// DiscoveryService
// ============================================================================

/// Endpoint data stored per announced entry.
struct AnnouncedEndpoint {
    /// Socket kind (e.g. "quic", "rep")
    socket_kind: String,
    /// Endpoint string (e.g. "quic://localhost:0.0.0.0:4433")
    endpoint: String,
    /// Service JWT attesting to the service's identity and pubkey
    service_jwt: String,
    /// Last heartbeat timestamp (Instant)
    last_heartbeat: Instant,
}

/// Phase 0.5 Stage D — cached signed OIDF entity statement.
struct CachedEntityStatement {
    /// Signed OpenID Federation 1.0 entity statement (compact JWS).
    jwt: String,
    /// Unix seconds when this was registered (set on push from issuer).
    fetched_at: i64,
}

/// Phase 0.5 Stage D — cached envelope COSE_KeySet.
struct CachedEnvelopeKeyset {
    /// CBOR-encoded COSE_KeySet (RFC 9052 §7).
    cose_keyset_cbor: Vec<u8>,
    /// Unix seconds when this was registered.
    fetched_at: i64,
}

/// Parse an `at://<did>/<collection>/<rkey>` URI into its three components.
///
/// The DID itself may contain `:` (e.g. `did:web:alice.example.com`,
/// `did:plc:abc123`) but no `/`, so splitting the post-`at://` remainder on `/`
/// yields exactly `[did, collection, rkey]`. Returns an error for any other shape.
fn parse_at_uri(uri: &str) -> Result<(String, String, String)> {
    let rest = uri
        .strip_prefix("at://")
        .ok_or_else(|| anyhow::anyhow!("at-uri must start with \"at://\": {uri:?}"))?;
    let mut parts = rest.splitn(3, '/');
    let did = parts.next().filter(|s| !s.is_empty());
    let collection = parts.next().filter(|s| !s.is_empty());
    let rkey = parts.next().filter(|s| !s.is_empty());
    match (did, collection, rkey) {
        (Some(d), Some(c), Some(r)) => Ok((d.to_owned(), c.to_owned(), r.to_owned())),
        _ => anyhow::bail!("at-uri must be at://<did>/<collection>/<rkey>: {uri:?}"),
    }
}

fn unix_seconds_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Discovery service that exposes EndpointRegistry over RPC.
pub struct DiscoveryService {
    /// Timestamp when the service was created
    started_at: Instant,
    /// Ed25519 signing key for envelope signing
    signing_key: Arc<SigningKey>,
    /// JWT verifying key for service JWT verification (derived from root via HKDF)
    jwt_verifying_key: hyprstream_rpc::prelude::VerifyingKey,
    /// JWT key source for client JWT verification (RequestService trait)
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
    /// OAuth issuer URL for RFC 9728 metadata (None = not configured)
    oauth_issuer_url: Option<String>,
    /// Expected audience for JWT validation (resource URL)
    expected_audience: Option<String>,
    /// Authorization provider (None = no authorization)
    auth_provider: Option<Box<dyn AuthorizationProvider>>,
    /// Record resolver backing getRecord/getRepo (#431). None = no local PDS,
    /// so getRecord/getRepo report NOT_FOUND for everything.
    record_resolver: Option<Box<dyn RecordResolver>>,
    /// Endpoints announced by other services (cross-process).
    /// Maps service_name → Vec<AnnouncedEndpoint>.
    announced_endpoints: RwLock<HashMap<String, Vec<AnnouncedEndpoint>>>,
    /// Phase 0.5 Stage D — cached signed OIDF entity statements per issuer URL.
    /// Pushed by IdPService/OAuth at startup + on every signing-key rotation.
    /// Consumed by FederationKeyResolver before falling back to HTTPS.
    entity_statements: RwLock<HashMap<String, CachedEntityStatement>>,
    /// Phase 0.5 Stage D — cached envelope COSE_KeySets per service did:web.
    /// Pushed by each service at startup + rotation. Consumed by RequestService
    /// receivers verifying COSE_Sign1 envelope signatures.
    envelope_keysets: RwLock<HashMap<String, CachedEnvelopeKeyset>>,
    /// Pre-computed TLS endorsement: Sign(tls_key, ed25519_pubkey || domain).
    /// Empty when TLS endorsement is not available (e.g. self-signed certs).
    tls_endorsement: Vec<u8>,
    /// Domain the TLS endorsement covers (empty when no endorsement).
    tls_domain: String,
    // Infrastructure (for Spawnable)
    transport: TransportConfig,
}

impl DiscoveryService {
    /// Create a new discovery service with infrastructure.
    ///
    /// `signing_key` is used for envelope signing (should be the per-service key
    /// derived via HKDF). `jwt_verifying_key` is the root-derived JWT verifying
    /// key for verifying service JWTs (derived from root via HKDF "hyprstream-jwt-v1").
    pub fn new(
        signing_key: Arc<SigningKey>,
        jwt_verifying_key: hyprstream_rpc::prelude::VerifyingKey,
        transport: TransportConfig,
    ) -> Self {
        Self {
            started_at: Instant::now(),
            signing_key,
            jwt_verifying_key,
            jwt_key_source: None,
            oauth_issuer_url: None,
            expected_audience: None,
            auth_provider: None,
            record_resolver: None,
            announced_endpoints: RwLock::new(HashMap::new()),
            entity_statements: RwLock::new(HashMap::new()),
            envelope_keysets: RwLock::new(HashMap::new()),
            tls_endorsement: Vec::new(),
            tls_domain: String::new(),
            transport,
        }
    }

    /// Set the pre-computed TLS endorsement and domain.
    ///
    /// The endorsement is `Sign(tls_private_key, "TLS_ENDORSEMENT_V1" || ed25519_pubkey || domain)`,
    /// computed once at startup by the service factory where TLS key materials are available.
    /// Discovery never touches the TLS private key directly.
    pub fn with_tls_endorsement(mut self, endorsement: Vec<u8>, domain: String) -> Self {
        self.tls_endorsement = endorsement;
        self.tls_domain = domain;
        self
    }

    /// Set the OAuth issuer URL for RFC 9728 auth metadata responses.
    pub fn with_oauth_issuer(mut self, url: String) -> Self {
        self.oauth_issuer_url = Some(url);
        self
    }

    /// Set the expected audience for JWT validation.
    pub fn with_expected_audience(mut self, audience: String) -> Self {
        self.expected_audience = Some(audience);
        self
    }

    /// Set the JWT key source for client JWT verification.
    pub fn with_jwt_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    ) -> Self {
        self.jwt_key_source = Some(src);
        self
    }

    /// Set the authorization provider for access control.
    pub fn with_auth_provider(mut self, provider: Box<dyn AuthorizationProvider>) -> Self {
        self.auth_provider = Some(provider);
        self
    }

    /// Set the record resolver backing getRecord/getRepo (#431).
    pub fn with_record_resolver(mut self, resolver: Box<dyn RecordResolver>) -> Self {
        self.record_resolver = Some(resolver);
        self
    }

    /// Get the global EndpointRegistry (D9: graceful error, not panic).
    fn reg(&self) -> Result<impl std::ops::Deref<Target = EndpointRegistry> + '_> {
        registry::try_global()
            .ok_or_else(|| anyhow::anyhow!("EndpointRegistry not initialized"))
    }
}

// ============================================================================
// Resolver implementation — DiscoveryService IS the authoritative resolver
// ============================================================================

#[async_trait]
impl Resolver for DiscoveryService {
    async fn resolve(&self, name: &str, kind: SocketKind) -> anyhow::Result<TransportConfig> {
        // Delegate to the global EndpointRegistry (D9: non-panicking).
        // In the future this will query internal state directly
        // (once EndpointRegistry is owned by DiscoveryService).
        Ok(self.reg()?.endpoint(name, kind))
    }
}

/// Convert a `SocketKind` to a lowercase string for serialization.
fn socket_kind_to_string(kind: SocketKind) -> &'static str {
    match kind {
        SocketKind::Req => "req",
        SocketKind::Rep => "rep",
        SocketKind::Quic => "quic",
    }
}

// ============================================================================
// DiscoveryHandler implementation (generated trait)
// ============================================================================

#[async_trait(?Send)]
impl DiscoveryHandler for DiscoveryService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        // Delegate to authorization provider if available
        if let Some(ref auth) = self.auth_provider {
            let subject = ctx.subject().to_string();
            let allowed = auth
                .check(&subject, "*", resource, operation)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("Discovery auth check failed for {}: {}", subject, e);
                    false
                });
            if allowed {
                Ok(())
            } else {
                anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
            }
        } else {
            // No auth provider — allow (backward compat for local-only deployments)
            Ok(())
        }
    }

    async fn handle_list_services(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        trace!("Discovery: listing services");

        // Process-local registry
        let reg = self.reg()?;
        let service_names = reg.list_services();
        let mut summaries: Vec<ServiceSummary> = service_names
            .iter()
            .filter_map(|name| {
                reg.service_entry(name).map(|entry| {
                    let socket_kinds: Vec<String> = entry
                        .endpoints
                        .keys()
                        .map(|k| socket_kind_to_string(*k).to_owned())
                        .collect();
                    ServiceSummary {
                        name: entry.name,
                        description: entry.description.unwrap_or_default(),
                        socket_kinds,
                        has_schema: entry.schema.is_some(),
                    }
                })
            })
            .collect();
        drop(reg);

        // Merge announced endpoints from other processes
        let announced = self.announced_endpoints.read();
        let local_names: Vec<String> = summaries.iter().map(|s| s.name.clone()).collect();
        for (name, endpoints) in announced.iter() {
            if local_names.iter().any(|n| n == name) {
                // Service exists locally — add announced socket kinds
                if let Some(summary) = summaries.iter_mut().find(|s| s.name == *name) {
                    for ep in endpoints {
                        if !summary.socket_kinds.contains(&ep.socket_kind) {
                            summary.socket_kinds.push(ep.socket_kind.clone());
                        }
                    }
                }
            } else {
                // Service only known from announcements
                summaries.push(ServiceSummary {
                    name: name.clone(),
                    description: String::new(),
                    socket_kinds: endpoints.iter().map(|e| e.socket_kind.clone()).collect(),
                    has_schema: false,
                });
            }
        }

        Ok(DiscoveryResponseVariant::ListServicesResult(ServiceList {
            services: summaries,
        }))
    }

    async fn handle_get_endpoints(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let service_name = data;
        trace!("Discovery: getting endpoints for '{}'", service_name);

        // Process-local registry
        let reg = self.reg()?;
        let endpoints_map = reg.service_endpoints(service_name);
        drop(reg);

        // Build from local registry (no self-proof — discovery's own JWT not stored locally)
        let mut endpoints: Vec<EndpointInfo> = match endpoints_map {
            Some(map) => map
                .iter()
                .map(|(kind, transport)| EndpointInfo {
                    socket_kind: socket_kind_to_string(*kind).to_owned(),
                    endpoint: transport.endpoint_string(),
                    service_jwt: String::new(),
                    tls_endorsement: self.tls_endorsement.clone(),
                    tls_domain: self.tls_domain.clone(),
                })
                .collect(),
            None => Vec::new(),
        };

        // Merge announced endpoints from other processes (carry service JWT)
        let announced = self.announced_endpoints.read();
        if let Some(announced_eps) = announced.get(service_name) {
            for ep in announced_eps {
                // Don't duplicate if already present from local registry
                if !endpoints.iter().any(|e| e.socket_kind == ep.socket_kind) {
                    endpoints.push(EndpointInfo {
                        socket_kind: ep.socket_kind.clone(),
                        endpoint: ep.endpoint.clone(),
                        service_jwt: ep.service_jwt.clone(),
                        tls_endorsement: self.tls_endorsement.clone(),
                        tls_domain: self.tls_domain.clone(),
                    });
                }
            }
        }

        if endpoints.is_empty() {
            Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("Service '{}' not found", service_name),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            }))
        } else {
            Ok(DiscoveryResponseVariant::GetEndpointsResult(
                ServiceEndpoints { endpoints },
            ))
        }
    }

    async fn handle_get_schema(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let service_name = data;
        trace!("Discovery: getting schema for '{}'", service_name);

        let reg = self.reg()?;
        let schema = reg.service_schema(service_name);
        drop(reg);

        match schema {
            Some(bytes) => Ok(DiscoveryResponseVariant::GetSchemaResult(bytes.to_vec())),
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("No schema registered for service '{}'", service_name),
                code: "NO_SCHEMA".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_ping(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        trace!("Discovery: ping");

        let reg = self.reg()?;
        let service_count = reg.list_services().len() as u32;
        drop(reg);

        let uptime = self.started_at.elapsed().as_secs();

        Ok(DiscoveryResponseVariant::PingResult(PingInfo {
            status: "ok".to_owned(),
            service_count,
            uptime,
        }))
    }

    async fn handle_get_auth_metadata(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let filter = data;
        trace!("Discovery: get auth metadata (filter='{}')", filter);

        let issuer = match &self.oauth_issuer_url {
            Some(url) => url.clone(),
            None => {
                return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                    message: "OAuth issuer URL not configured".to_owned(),
                    code: "NOT_CONFIGURED".to_owned(),
                    details: String::new(),
                }));
            }
        };

        let reg = self.reg()?;
        let service_names = reg.list_services();

        let services: Vec<AuthMetadata> = service_names
            .iter()
            .filter(|name| filter.is_empty() || *name == filter)
            .filter_map(|name| {
                let quic_transport = reg.endpoint(name, SocketKind::Quic);
                let resource = quic_transport.quic_resource_url(name)?;
                Some(AuthMetadata {
                    service_name: name.clone(),
                    resource,
                    authorization_servers: vec![issuer.clone()],
                    scopes_supported: Vec::new(),
                    resource_name: format!("HyprStream {} Service", name),
                })
            })
            .collect();
        drop(reg);

        Ok(DiscoveryResponseVariant::GetAuthMetadataResult(
            AuthMetadataList { services },
        ))
    }

    async fn handle_prepare_stream(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        Ok(DiscoveryResponseVariant::Error(ErrorInfo {
            message: "prepareStream removed — use StreamChannel::prepare_stream for authenticated streaming".to_owned(),
            code: "REMOVED".to_owned(),
            details: String::new(),
        }))
    }

    async fn handle_get_stream(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        Ok(DiscoveryResponseVariant::Error(ErrorInfo {
            message: "getStream removed — use StreamChannel::prepare_stream for authenticated streaming".to_owned(),
            code: "REMOVED".to_owned(),
            details: String::new(),
        }))
    }

    async fn handle_list_streams(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        Ok(DiscoveryResponseVariant::Error(ErrorInfo {
            message: "listStreams removed — use StreamChannel::prepare_stream for authenticated streaming".to_owned(),
            code: "REMOVED".to_owned(),
            details: String::new(),
        }))
    }

    async fn handle_announce(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ServiceAnnouncement,
    ) -> Result<DiscoveryResponseVariant> {
        info!(
            "Discovery: service '{}' announced {} endpoint: {} (from {})",
            data.service_name, data.socket_kind, data.endpoint, ctx.subject()
        );

        let svc_name = data.service_name.clone();
        let sock_kind = data.socket_kind.clone();
        let endpoint = data.endpoint.clone();
        let service_jwt = data.service_jwt.clone().unwrap_or_default();

        // R3: Verify service JWT signature + subject matches serviceName.
        // Full JWT verification (not decode_unverified) to prevent forged identities.
        if !service_jwt.is_empty() {
            let verified = hyprstream_rpc::auth::jwt::decode_with_key(
                &service_jwt,
                &self.jwt_verifying_key,
                self.expected_audience.as_deref(),
            ).map_err(|e| {
                tracing::warn!("Service JWT verification failed in announce: {}", e);
                anyhow::anyhow!("Invalid service JWT in announce: {}", e)
            })?;
            // Check that sub matches "service:{serviceName}"
            let expected_sub = format!("service:{}", svc_name);
            if verified.sub != expected_sub {
                anyhow::bail!(
                    "Service JWT subject mismatch: expected '{}', got '{}'",
                    expected_sub,
                    verified.sub
                );
            }
        }

        let mut endpoints = self.announced_endpoints.write();
        let entry = endpoints.entry(svc_name).or_default();
        // Replace existing endpoint for the same socket kind, or add new
        if let Some(existing) = entry.iter_mut().find(|e| e.socket_kind == sock_kind) {
            existing.endpoint = endpoint;
            existing.service_jwt = service_jwt;
            existing.last_heartbeat = Instant::now();
        } else {
            entry.push(AnnouncedEndpoint {
                socket_kind: sock_kind,
                endpoint,
                service_jwt,
                last_heartbeat: Instant::now(),
            });
        }

        Ok(DiscoveryResponseVariant::AnnounceResult)
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase 0.5 Stage D — federation directory
    // ────────────────────────────────────────────────────────────────────

    async fn handle_register_entity_statement(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterEntityStatementRequest,
    ) -> Result<DiscoveryResponseVariant> {
        if data.issuer.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "issuer is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        if data.jwt.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "jwt is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // NOTE: we intentionally do NOT gate cache writes by the
        // `federation:register` policy here. The load-bearing trust
        // check happens at use time in FederationKeyResolver::get_key,
        // which queries policy before accepting any cached entity
        // statement for verification. Gating writes too would force
        // discovery to call PolicyService on every register, which is
        // a perf hit on a hot path that already requires a passing
        // `authorize()` check on the caller side. Cache may contain
        // statements for issuers no operator currently trusts —
        // they're inert until the resolver's policy check admits them.
        let cached = CachedEntityStatement {
            jwt: data.jwt.clone(),
            fetched_at: unix_seconds_now(),
        };
        let mut map = self.entity_statements.write();
        map.insert(data.issuer.clone(), cached);
        let total = map.len();
        drop(map);

        info!(
            issuer = %data.issuer,
            caller = %ctx.subject(),
            total_cached = total,
            "Discovery: entity statement registered"
        );
        Ok(DiscoveryResponseVariant::RegisterEntityStatementResult)
    }

    async fn handle_get_entity_statement(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let issuer = data;
        let map = self.entity_statements.read();
        match map.get(issuer) {
            Some(cached) => {
                trace!(issuer = %issuer, "Discovery: entity statement cache hit");
                Ok(DiscoveryResponseVariant::GetEntityStatementResult(EntityStatement {
                    issuer: issuer.to_owned(),
                    jwt: cached.jwt.clone(),
                    fetched_at: cached.fetched_at,
                }))
            }
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no entity statement cached for issuer: {}", issuer),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_register_envelope_keyset(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterEnvelopeKeysetRequest,
    ) -> Result<DiscoveryResponseVariant> {
        if data.service_did.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "serviceDid is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        if data.cose_keyset_cbor.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "coseKeysetCbor is required".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        let cached = CachedEnvelopeKeyset {
            cose_keyset_cbor: data.cose_keyset_cbor.clone(),
            fetched_at: unix_seconds_now(),
        };
        let mut map = self.envelope_keysets.write();
        map.insert(data.service_did.clone(), cached);
        let total = map.len();
        drop(map);

        info!(
            service_did = %data.service_did,
            caller = %ctx.subject(),
            total_cached = total,
            "Discovery: envelope keyset registered"
        );
        Ok(DiscoveryResponseVariant::RegisterEnvelopeKeysetResult)
    }

    async fn handle_get_envelope_keyset(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let service_did = data;
        let map = self.envelope_keysets.read();
        match map.get(service_did) {
            Some(cached) => {
                trace!(service_did = %service_did, "Discovery: envelope keyset cache hit");
                Ok(DiscoveryResponseVariant::GetEnvelopeKeysetResult(EnvelopeKeyset {
                    service_did: service_did.to_owned(),
                    cose_keyset_cbor: cached.cose_keyset_cbor.clone(),
                    fetched_at: cached.fetched_at,
                }))
            }
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no envelope keyset cached for service: {}", service_did),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_list_known_issuers(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        // Federation topology leak — require authentication. Per
        // Phase 0.5 plan Q10: getEntityStatement/getEnvelopeKeyset are
        // anonymous-readable (public artifacts), but listKnownIssuers
        // enumerates which partners we trust, which is operator-sensitive.
        if let Err(e) = self.authorize(ctx, "discovery:federation", "list-known-issuers").await {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("unauthorized: {}", e),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }
        let map = self.entity_statements.read();
        let issuers: Vec<String> = map.keys().cloned().collect();
        Ok(DiscoveryResponseVariant::ListKnownIssuersResult(IssuerList { issuers }))
    }

    // ────────────────────────────────────────────────────────────────────
    // #431 — federated record lookup as a verifiable CAR proof
    // ────────────────────────────────────────────────────────────────────

    async fn handle_get_record(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetRecordRequest,
    ) -> Result<DiscoveryResponseVariant> {
        // Resolve the three components from either the at:// URI or the structured
        // fields. When `uri` is set it wins (the fields are ignored, per schema).
        let (did, collection, rkey) = if !data.uri.is_empty() {
            match parse_at_uri(&data.uri) {
                Ok(parts) => parts,
                Err(e) => {
                    return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                        message: format!("invalid at-uri: {e}"),
                        code: "INVALID_ARGUMENT".to_owned(),
                        details: String::new(),
                    }));
                }
            }
        } else if !data.did.is_empty() && !data.collection.is_empty() && !data.rkey.is_empty() {
            (data.did.clone(), data.collection.clone(), data.rkey.clone())
        } else {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "getRecord requires either `uri` or all of (did, collection, rkey)".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        };

        // ── Access control (CONFIDENTIALITY — the load-bearing check) ──
        // The schema's auto-generated discovery:query gate already ran in dispatch,
        // but that only authorizes "may call getRecord at all". An atproto record's
        // CID is content-derived and predictable, so presenting a valid at:// / CID
        // must NOT by itself grant a read. Independently require the caller to be
        // permitted to read THIS target DID's collection. Integrity (the signed CAR
        // proof, verified offline) does NOT substitute for this check.
        let resource = format!("discovery:record:{did}/{collection}");
        if let Err(e) = self.authorize(ctx, &resource, "query").await {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("unauthorized to read {resource}: {e}"),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }

        let resolver = match &self.record_resolver {
            Some(r) => r,
            None => {
                return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                    message: "no record resolver configured on this node".to_owned(),
                    code: "NOT_FOUND".to_owned(),
                    details: String::new(),
                }));
            }
        };

        match resolver.resolve_record(&did, &collection, &rkey).await {
            Ok(Some(rec)) => Ok(DiscoveryResponseVariant::GetRecordResult(RecordCar {
                uri: rec.uri,
                car: rec.car,
            })),
            Ok(None) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no record at at://{did}/{collection}/{rkey}"),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
            Err(e) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("record resolution failed: {e}"),
                code: "INTERNAL".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_get_repo(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        value: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let did = value;
        if did.is_empty() {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: "getRepo requires a non-empty DID".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Access control: a full repo CAR exposes every record in the DID's
        // collections, so gate on the DID itself (broadest read of that repo).
        let resource = format!("discovery:record:{did}");
        if let Err(e) = self.authorize(ctx, &resource, "query").await {
            return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("unauthorized to read repo {did}: {e}"),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }

        let resolver = match &self.record_resolver {
            Some(r) => r,
            None => {
                return Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                    message: "no record resolver configured on this node".to_owned(),
                    code: "NOT_FOUND".to_owned(),
                    details: String::new(),
                }));
            }
        };

        match resolver.resolve_repo(did).await {
            Ok(Some(rec)) => Ok(DiscoveryResponseVariant::GetRepoResult(RecordCar {
                uri: rec.uri,
                car: rec.car,
            })),
            Ok(None) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("no repo stored for DID {did}"),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
            Err(e) => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("repo resolution failed: {e}"),
                code: "INTERNAL".to_owned(),
                details: String::new(),
            })),
        }
    }

    /// #523 P0 / #524 — placement candidate query. The vocabulary + filter/rank/
    /// select helpers landed in `crate::scheduling` (#628); the live
    /// directory + authz-prefiltered handler is P1's job (#524). Stubbed here
    /// so the RPC surface compiles; returns UNIMPLEMENTED until P1 wires it.
    async fn handle_query_candidates(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        _data: &QueryCandidatesRequest,
    ) -> Result<DiscoveryResponseVariant> {
        Ok(DiscoveryResponseVariant::Error(ErrorInfo {
            message: "queryCandidates not yet wired (P1 placement directory; see #524)".to_owned(),
            code: "UNIMPLEMENTED".to_owned(),
            details: String::new(),
        }))
    }
}

// ============================================================================
// RequestService implementation
// ============================================================================

#[async_trait(?Send)]
impl RequestService for DiscoveryService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<hyprstream_rpc::Continuation>)> {
        trace!(
            "Discovery request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_discovery(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "discovery"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> hyprstream_rpc::crypto::SigningKey {
        (*self.signing_key).clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        self.jwt_key_source.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = DiscoveryResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// #431 — getRecord / getRepo handler tests (access control + CAR proof)
// ============================================================================

#[cfg(test)]
mod get_record_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;

    use std::collections::BTreeMap;

    use hyprstream_pds::car::{build_record_proof_car, parse_car_v1, verify_record_proof};
    use hyprstream_pds::cid::Cid as PdsCid;
    use hyprstream_pds::commit::{Commit, UnsignedCommit};
    use hyprstream_pds::mst::{Node, Proof};
    use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
    use hyprstream_pds::tid::Tid;
    use p256::ecdsa::{SigningKey as P256SigningKey, VerifyingKey as P256VerifyingKey};

    const TEST_DID: &str = "did:web:alice.example.com";

    /// Mock authorization provider: allows exactly the (resource, operation)
    /// pairs in its allow-list, denies everything else. Mirrors the Casbin gate
    /// the production PolicyAuthProvider wraps.
    struct MockAuth {
        allow: Vec<(String, String)>,
    }
    #[async_trait(?Send)]
    impl AuthorizationProvider for MockAuth {
        async fn check(
            &self,
            _subject: &str,
            _domain: &str,
            resource: &str,
            operation: &str,
        ) -> Result<bool> {
            Ok(self
                .allow
                .iter()
                .any(|(r, o)| r == resource && o == operation))
        }
    }

    /// Mock record resolver backed by a real PDS-built CAR proof, so a returned
    /// RecordCar verifies under `verify_record_proof`. `None` records → NOT_FOUND.
    struct MockResolver {
        /// (collection, rkey) → (car bytes, uri). Built once in `build`.
        records: BTreeMap<(String, String), (Vec<u8>, String)>,
    }

    /// Build a small repo with `n` records and return the resolver plus the
    /// published `#atproto` verifying key, target rkey, target record, and proof
    /// so tests can independently verify the CAR the handler returns.
    fn build_repo(
        n: u64,
    ) -> (
        MockResolver,
        P256VerifyingKey,
        String,             // target rkey
        ModelRecord,        // target record
        PdsCid,             // target record CID
        Proof,              // target inclusion proof
    ) {
        let signing_key = P256SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = P256VerifyingKey::from(&signing_key);

        let mut records: BTreeMap<Tid, ModelRecord> = BTreeMap::new();
        let mut record_cids: BTreeMap<Tid, PdsCid> = BTreeMap::new();
        for i in 0..n {
            let tid = Tid::from_micros(1_700_000_000_000_000 + i * 1000, i as u16);
            let rec = ModelRecord::new(
                format!("at://{TEST_DID}"),
                format!("bafyreiexampleoid{i:020}"),
                "2026-06-24T00:00:00.000Z",
            )
            .unwrap();
            record_cids.insert(tid, rec.cid());
            records.insert(tid, rec);
        }
        let tree = Node::from_records(COLLECTION_NSID, &record_cids);
        let root = tree.root_cid();
        let unsigned = UnsignedCommit::new(TEST_DID.to_owned(), root, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing_key);
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();

        // Pick a middle record as the target.
        let target_tid = *records.keys().nth((n / 2) as usize).unwrap();
        let target_record = records.get(&target_tid).cloned().unwrap();
        let target_cid = record_cids[&target_tid];
        let target_rkey = target_tid.encode();

        // Build a CAR proof for EVERY record so the resolver can answer any rkey.
        let mut out = BTreeMap::new();
        for (tid, rec) in &records {
            let proof = tree.proof(COLLECTION_NSID, tid).unwrap();
            let car = build_record_proof_car(&commit, &proof, &node_blocks, rec);
            let rkey = tid.encode();
            let uri = format!("at://{TEST_DID}/{COLLECTION_NSID}/{rkey}");
            out.insert((COLLECTION_NSID.to_owned(), rkey), (car, uri));
        }
        let target_proof = tree.proof(COLLECTION_NSID, &target_tid).unwrap();

        (
            MockResolver { records: out },
            verifying_key,
            target_rkey,
            target_record,
            target_cid,
            target_proof,
        )
    }

    #[async_trait(?Send)]
    impl RecordResolver for MockResolver {
        async fn resolve_record(
            &self,
            _did: &str,
            collection: &str,
            rkey: &str,
        ) -> Result<Option<RecordCarData>> {
            Ok(self
                .records
                .get(&(collection.to_owned(), rkey.to_owned()))
                .map(|(car, uri)| RecordCarData {
                    uri: uri.clone(),
                    car: car.clone(),
                }))
        }
        async fn resolve_repo(&self, _did: &str) -> Result<Option<RecordCarData>> {
            Ok(self
                .records
                .values()
                .next()
                .map(|(car, uri)| RecordCarData {
                    uri: uri.clone(),
                    car: car.clone(),
                }))
        }
    }

    fn service_with(
        allow: Vec<(&str, &str)>,
        resolver: MockResolver,
    ) -> DiscoveryService {
        let (sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();
        let allow = allow
            .into_iter()
            .map(|(r, o)| (r.to_owned(), o.to_owned()))
            .collect();
        DiscoveryService::new(
            Arc::new(sk),
            vk,
            hyprstream_rpc::transport::TransportConfig::inproc("discovery-test"),
        )
        .with_auth_provider(Box::new(MockAuth { allow }))
        .with_record_resolver(Box::new(resolver))
    }

    fn test_ctx() -> EnvelopeContext {
        // A genuine service-identity caller; subject() → "service:test-caller".
        EnvelopeContext::from_callback_service(1, "test-caller")
    }

    /// (a) An accessible record returns a valid CAR proof that verifies offline.
    #[tokio::test]
    async fn get_record_returns_verifiable_car_when_authorized() {
        let (resolver, vk, rkey, target_record, target_cid, proof) = build_repo(6);
        let resource = format!("discovery:record:{TEST_DID}/{COLLECTION_NSID}");
        let svc = service_with(vec![(&resource, "query")], resolver);

        let req = GetRecordRequest {
            uri: format!("at://{TEST_DID}/{COLLECTION_NSID}/{rkey}"),
            did: String::new(),
            collection: String::new(),
            rkey: String::new(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        let car = match resp {
            DiscoveryResponseVariant::GetRecordResult(rc) => rc,
            other => panic!("expected GetRecordResult, got {other:?}"),
        };

        // The CAR parses and contains the target record block.
        let (roots, blocks) = parse_car_v1(&car.car).unwrap();
        assert!(!roots.is_empty(), "CAR must have a root commit CID");
        let block_cids: std::collections::BTreeSet<PdsCid> =
            blocks.iter().map(|(c, _)| *c).collect();
        assert!(
            block_cids.contains(&target_cid),
            "CAR must carry the target record block"
        );

        // The returned proof verifies offline against the published #atproto key.
        let commit = blocks
            .iter()
            .find(|(c, _)| *c == roots[0])
            .map(|(_, b)| Commit::from_dag_cbor(b).unwrap())
            .expect("CAR must contain its commit block");
        verify_record_proof(&commit, &vk, &proof, &target_record)
            .expect("the returned CAR proof must verify offline (D5 untrusted-relay)");
    }

    /// (b) THE LOAD-BEARING TEST: denied when the caller lacks read permission
    /// for the target DID/collection — even though the at:// / CID is valid.
    #[tokio::test]
    async fn get_record_denied_without_read_permission() {
        let (resolver, _vk, rkey, _rec, _cid, _proof) = build_repo(6);
        // Allow-list does NOT include discovery:record:<did>/<collection>.
        let svc = service_with(vec![("discovery:something-else", "query")], resolver);

        let req = GetRecordRequest {
            uri: format!("at://{TEST_DID}/{COLLECTION_NSID}/{rkey}"),
            did: String::new(),
            collection: String::new(),
            rkey: String::new(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        match resp {
            DiscoveryResponseVariant::Error(e) => {
                assert_eq!(e.code, "UNAUTHORIZED", "deny must surface as UNAUTHORIZED");
                // CRITICAL: the deny must NOT leak the record bytes.
                assert!(
                    !e.message.is_empty(),
                    "unauthorized error should explain the denied resource"
                );
            }
            DiscoveryResponseVariant::GetRecordResult(_) => {
                panic!("ACCESS-CONTROL FAILURE: a valid at:// returned a record without read permission");
            }
            other => panic!("expected UNAUTHORIZED error, got {other:?}"),
        }
    }

    /// (c) A missing record errors cleanly with NOT_FOUND (authorized caller).
    #[tokio::test]
    async fn get_record_missing_returns_not_found() {
        let (resolver, _vk, _rkey, _rec, _cid, _proof) = build_repo(6);
        let resource = format!("discovery:record:{TEST_DID}/{COLLECTION_NSID}");
        let svc = service_with(vec![(&resource, "query")], resolver);

        let req = GetRecordRequest {
            uri: format!("at://{TEST_DID}/{COLLECTION_NSID}/3zzzzzzzzzzzz"),
            did: String::new(),
            collection: String::new(),
            rkey: String::new(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        match resp {
            DiscoveryResponseVariant::Error(e) => {
                assert_eq!(e.code, "NOT_FOUND", "absent record must be NOT_FOUND");
            }
            other => panic!("expected NOT_FOUND error, got {other:?}"),
        }
    }

    /// at:// parser: rejects malformed URIs, accepts the canonical 3-part form.
    #[test]
    fn parse_at_uri_round_trip() {
        let (did, coll, rkey) =
            parse_at_uri("at://did:web:alice.example.com/ai.hyprstream.model/3zztslq4be52u")
                .unwrap();
        assert_eq!(did, "did:web:alice.example.com");
        assert_eq!(coll, "ai.hyprstream.model");
        assert_eq!(rkey, "3zztslq4be52u");

        assert!(parse_at_uri("https://x").is_err());
        assert!(parse_at_uri("at://did:web:x").is_err()); // no collection/rkey
        assert!(parse_at_uri("at://did:web:x/coll").is_err()); // no rkey
    }

    /// getRecord with structured (did, collection, rkey) fields (uri empty).
    #[tokio::test]
    async fn get_record_by_structured_fields() {
        let (resolver, _vk, rkey, _rec, target_cid, _proof) = build_repo(4);
        let resource = format!("discovery:record:{TEST_DID}/{COLLECTION_NSID}");
        let svc = service_with(vec![(&resource, "query")], resolver);

        let req = GetRecordRequest {
            uri: String::new(),
            did: TEST_DID.to_owned(),
            collection: COLLECTION_NSID.to_owned(),
            rkey: rkey.clone(),
        };
        let resp = svc.handle_get_record(&test_ctx(), 1, &req).await.unwrap();
        let car = match resp {
            DiscoveryResponseVariant::GetRecordResult(rc) => rc,
            other => panic!("expected GetRecordResult, got {other:?}"),
        };
        let (_roots, blocks) = parse_car_v1(&car.car).unwrap();
        let block_cids: std::collections::BTreeSet<PdsCid> =
            blocks.iter().map(|(c, _)| *c).collect();
        assert!(block_cids.contains(&target_cid));
    }
}
