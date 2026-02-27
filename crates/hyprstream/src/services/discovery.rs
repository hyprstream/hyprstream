//! Discovery service — exposes EndpointRegistry over ZMQ RPC
//!
//! Allows remote clients to discover registered services, their endpoints,
//! socket kinds, and schemas via the standard ZMQ REQ/REP transport.

use async_trait::async_trait;
use crate::services::{EnvelopeContext, ZmqService};
use crate::services::generated::discovery_client::{
    DiscoveryClient, DiscoveryHandler, DiscoveryResponseVariant,
    ErrorInfo, ServiceList, ServiceSummary, ServiceEndpoints, EndpointInfo,
    PingInfo, dispatch_discovery, serialize_response,
};
use anyhow::Result;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
use std::sync::Arc;
use std::time::Instant;
use tracing::trace;

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
            context,
            transport,
        }
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
    }
}

// ============================================================================
// DiscoveryHandler implementation (generated trait)
// ============================================================================

#[async_trait::async_trait(?Send)]
impl DiscoveryHandler for DiscoveryService {
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
