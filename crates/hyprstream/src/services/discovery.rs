//! Discovery service — exposes EndpointRegistry over ZMQ RPC
//!
//! Allows remote clients to discover registered services, their endpoints,
//! socket kinds, and schemas via the standard ZMQ REQ/REP transport.

use async_trait::async_trait;
use crate::services::{EnvelopeContext, PolicyClient, ZmqService};
use crate::services::generated::discovery_client::{
    DiscoveryClient, DiscoveryHandler, DiscoveryResponseVariant,
    ErrorInfo, ServiceList, ServiceSummary, ServiceEndpoints, EndpointInfo,
    PingInfo, AuthMetadata, AuthMetadataList,
    StreamPrepareRequest, StreamEndpoints, StreamEndpointsList,
    dispatch_discovery, serialize_response,
};
use anyhow::Result;
use dashmap::DashMap;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, trace};

/// Service name for endpoint registry
const SERVICE_NAME: &str = "discovery";

// ============================================================================
// DiscoveryService (server-side)
// ============================================================================

/// Discovery service that exposes EndpointRegistry over ZMQ RPC.
pub struct DiscoveryService {
    /// Timestamp when the service was created
    started_at: Instant,
    /// Ed25519 signing key for envelope signing
    signing_key: Arc<SigningKey>,
    /// OAuth issuer URL for RFC 9728 metadata (None = not configured)
    oauth_issuer_url: Option<String>,
    /// Expected audience for JWT validation (resource URL)
    expected_audience: Option<String>,
    /// Policy client for authorization checks (None = no authorization)
    policy_client: Option<PolicyClient>,
    /// Named stream registry (name -> StreamEndpoints)
    stream_registry: Arc<DashMap<String, StreamEndpoints>>,
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
        Self {
            started_at: Instant::now(),
            signing_key,
            oauth_issuer_url: None,
            expected_audience: None,
            policy_client: None,
            stream_registry: Arc::new(DashMap::new()),
            context,
            transport,
        }
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

    /// Set the policy client for authorization checks.
    pub fn with_policy_client(mut self, client: PolicyClient) -> Self {
        self.policy_client = Some(client);
        self
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

#[async_trait::async_trait(?Send)]
impl DiscoveryHandler for DiscoveryService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        // Local callers are always authorized
        if ctx.identity.is_local() {
            return Ok(());
        }
        // Delegate to policy service if available
        if let Some(ref policy_client) = self.policy_client {
            let subject = ctx.subject().to_string();
            let allowed = policy_client.check(&subject, "*", resource, operation)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("Discovery policy check failed for {}: {}", subject, e);
                    false
                });
            if allowed {
                Ok(())
            } else {
                anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
            }
        } else {
            // No policy client — allow (backward compat for local-only deployments)
            Ok(())
        }
    }

    async fn handle_list_services(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        trace!("Discovery: listing services");

        // Clone data out before any async work — don't hold the registry guard
        let reg = registry();
        let service_names = reg.list_services();
        let summaries: Vec<ServiceSummary> = service_names
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

        let reg = registry();
        let endpoints_map = reg.service_endpoints(service_name);
        drop(reg);

        match endpoints_map {
            Some(map) => {
                let endpoints: Vec<EndpointInfo> = map
                    .iter()
                    .map(|(kind, transport)| EndpointInfo {
                        socket_kind: socket_kind_to_string(*kind).to_owned(),
                        endpoint: transport.to_zmq_string(),
                    })
                    .collect();
                Ok(DiscoveryResponseVariant::GetEndpointsResult(
                    ServiceEndpoints { endpoints },
                ))
            }
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("Service '{}' not found", service_name),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
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

        let reg = registry();
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

        let reg = registry();
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

        let reg = registry();
        let service_names = reg.list_services();

        let services: Vec<AuthMetadata> = service_names
            .iter()
            .filter(|name| filter.is_empty() || *name == filter)
            .filter_map(|name| {
                // Look up QUIC endpoint to derive the resource URL
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
        data: &StreamPrepareRequest,
    ) -> Result<DiscoveryResponseVariant> {
        let name = &data.name;
        info!("Discovery: prepare_stream('{}')", name);

        // Idempotent: return existing stream if already registered
        if let Some(existing) = self.stream_registry.get(name) {
            return Ok(DiscoveryResponseVariant::PrepareStreamResult(
                existing.value().clone(),
            ));
        }

        // Generate random topic (64 hex chars) + mac_key (32 bytes)
        let mut rng = rand::thread_rng();
        let mut topic_bytes = [0u8; 32];
        let mut mac_key = [0u8; 32];
        rng.fill(&mut topic_bytes);
        rng.fill(&mut mac_key);
        let topic = hex::encode(topic_bytes);

        // Look up StreamService endpoints from registry
        let reg = registry();
        let push_ep = reg.endpoint("streams", SocketKind::Push).to_zmq_string();
        let sub_ep = reg.endpoint("streams", SocketKind::Sub).to_zmq_string();
        drop(reg);

        let endpoints = StreamEndpoints {
            name: name.clone(),
            stream_id: format!("stream-{}", uuid::Uuid::new_v4()),
            topic,
            push_endpoint: push_ep,
            sub_endpoint: sub_ep,
            mac_key: mac_key.to_vec(),
        };

        self.stream_registry
            .insert(name.clone(), endpoints.clone());

        Ok(DiscoveryResponseVariant::PrepareStreamResult(endpoints))
    }

    async fn handle_get_stream(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &str,
    ) -> Result<DiscoveryResponseVariant> {
        let name = data;
        trace!("Discovery: get_stream('{}')", name);

        match self.stream_registry.get(name) {
            Some(entry) => Ok(DiscoveryResponseVariant::GetStreamResult(
                entry.value().clone(),
            )),
            None => Ok(DiscoveryResponseVariant::Error(ErrorInfo {
                message: format!("Stream '{}' not found", name),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        }
    }

    async fn handle_list_streams(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<DiscoveryResponseVariant> {
        trace!("Discovery: list_streams");

        let streams: Vec<StreamEndpoints> = self
            .stream_registry
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        Ok(DiscoveryResponseVariant::ListStreamsResult(
            StreamEndpointsList { streams },
        ))
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
    ) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
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

    fn signing_key(&self) -> SigningKey {
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

// ============================================================================
// DiscoveryClient construction (uses create_service_client pattern)
// ============================================================================

impl DiscoveryClient {
    /// Create a new discovery client (endpoint from registry).
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        crate::services::core::create_service_client(&endpoint, signing_key, identity)
    }
}
