//! Discovery service — exposes EndpointRegistry over ZMQ RPC.
//!
//! Allows remote clients to discover registered services, their endpoints,
//! socket kinds, and schemas via the standard ZMQ REQ/REP transport.

use async_trait::async_trait;
use hyprstream_rpc::service::{EnvelopeContext, ZmqService};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::registry::{self, EndpointRegistry, SocketKind};
use hyprstream_rpc::resolver::Resolver;
use hyprstream_rpc::SigningKey;

use crate::generated::discovery_client::{
    DiscoveryHandler, DiscoveryResponseVariant,
    ErrorInfo, ServiceList, ServiceSummary, ServiceEndpoints, EndpointInfo,
    PingInfo, AuthMetadata, AuthMetadataList, ServiceAnnouncement,
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
// DiscoveryService
// ============================================================================

/// Discovery service that exposes EndpointRegistry over ZMQ RPC.
pub struct DiscoveryService {
    /// Timestamp when the service was created
    started_at: Instant,
    /// Ed25519 signing key for envelope signing
    signing_key: Arc<SigningKey>,
    /// Purpose-derived key for self-proof signing (not the root key)
    proof_signing_key: SigningKey,
    /// OAuth issuer URL for RFC 9728 metadata (None = not configured)
    oauth_issuer_url: Option<String>,
    /// Expected audience for JWT validation (resource URL)
    expected_audience: Option<String>,
    /// Authorization provider (None = no authorization)
    auth_provider: Option<Box<dyn AuthorizationProvider>>,
    /// Endpoints announced by other services (cross-process).
    /// Maps service_name → Vec<(socket_kind, endpoint)>.
    announced_endpoints: RwLock<HashMap<String, Vec<(String, String)>>>,
    /// Pre-computed TLS endorsement: Sign(tls_key, ed25519_pubkey || domain).
    /// Empty when TLS endorsement is not available (e.g. self-signed certs).
    tls_endorsement: Vec<u8>,
    /// Domain the TLS endorsement covers (empty when no endorsement).
    tls_domain: String,
    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
}

impl DiscoveryService {
    /// Create a new discovery service with infrastructure.
    pub fn new(
        signing_key: Arc<SigningKey>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        let proof_signing_key = hyprstream_rpc::node_identity::derive_purpose_key(
            &signing_key,
            "discovery-self-proof-v1",
        );
        Self {
            started_at: Instant::now(),
            signing_key,
            proof_signing_key,
            oauth_issuer_url: None,
            expected_audience: None,
            auth_provider: None,
            announced_endpoints: RwLock::new(HashMap::new()),
            tls_endorsement: Vec::new(),
            tls_domain: String::new(),
            context,
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

    /// Set the authorization provider for access control.
    pub fn with_auth_provider(mut self, provider: Box<dyn AuthorizationProvider>) -> Self {
        self.auth_provider = Some(provider);
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
        SocketKind::Dealer => "dealer",
        SocketKind::Router => "router",
        SocketKind::Pub => "pub",
        SocketKind::Sub => "sub",
        SocketKind::XPub => "xpub",
        SocketKind::XSub => "xsub",
        SocketKind::Push => "push",
        SocketKind::Pull => "pull",
        SocketKind::Pair => "pair",
        SocketKind::Stream => "stream",
        SocketKind::Quic => "quic",
    }
}

// ============================================================================
// DiscoveryHandler implementation (generated trait)
// ============================================================================

#[async_trait(?Send)]
impl DiscoveryHandler for DiscoveryService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        // System (node key) bypass for read operations only.
        // Write operations (manage scope) always go through Casbin.
        if ctx.subject() == hyprstream_rpc::envelope::Subject::new("system")
            && operation == "query"
        {
            return Ok(());
        }
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
                    for (kind, _) in endpoints {
                        if !summary.socket_kinds.contains(kind) {
                            summary.socket_kinds.push(kind.clone());
                        }
                    }
                }
            } else {
                // Service only known from announcements
                summaries.push(ServiceSummary {
                    name: name.clone(),
                    description: String::new(),
                    socket_kinds: endpoints.iter().map(|(k, _)| k.clone()).collect(),
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

        let pubkey = self.proof_signing_key.verifying_key().to_bytes().to_vec();

        // Self-signed proof: Sign(purpose_derived_key, pubkey || timestamp || expiry)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let expiry = now + 86400; // 24h
        let mut proof_data = Vec::with_capacity(32 + 8 + 8);
        proof_data.extend_from_slice(&pubkey);
        proof_data.extend_from_slice(&now.to_le_bytes());
        proof_data.extend_from_slice(&expiry.to_le_bytes());
        use ed25519_dalek::Signer as _;
        let self_proof = self.proof_signing_key.sign(&proof_data).to_bytes().to_vec();

        let mut endpoints: Vec<EndpointInfo> = match endpoints_map {
            Some(map) => map
                .iter()
                .map(|(kind, transport)| EndpointInfo {
                    socket_kind: socket_kind_to_string(*kind).to_owned(),
                    endpoint: transport.to_zmq_string(),
                    pubkey: pubkey.clone(),
                    self_proof: self_proof.clone(),
                    proof_timestamp: now,
                    proof_expiry: expiry,
                    tls_endorsement: self.tls_endorsement.clone(),
                    tls_domain: self.tls_domain.clone(),
                })
                .collect(),
            None => Vec::new(),
        };

        // Merge announced endpoints from other processes
        let announced = self.announced_endpoints.read();
        if let Some(announced_eps) = announced.get(service_name) {
            for (kind, ep) in announced_eps {
                // Don't duplicate if already present from local registry
                if !endpoints.iter().any(|e| e.socket_kind == *kind) {
                    endpoints.push(EndpointInfo {
                        socket_kind: kind.clone(),
                        endpoint: ep.clone(),
                        pubkey: pubkey.clone(),
                        self_proof: self_proof.clone(),
                        proof_timestamp: now,
                        proof_expiry: expiry,
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

        let mut endpoints = self.announced_endpoints.write();
        let entry = endpoints.entry(svc_name).or_default();
        // Replace existing endpoint for the same socket kind, or add new
        if let Some(existing) = entry.iter_mut().find(|(k, _)| *k == sock_kind) {
            existing.1 = endpoint;
        } else {
            entry.push((sock_kind, endpoint));
        }

        Ok(DiscoveryResponseVariant::AnnounceResult)
    }
}

// ============================================================================
// ZmqService implementation
// ============================================================================

#[async_trait(?Send)]
impl ZmqService for DiscoveryService {
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

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
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

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = DiscoveryResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}
