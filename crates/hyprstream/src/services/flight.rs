//! FlightService - Arrow Flight SQL with RPC control channel
//!
//! Dual-protocol service:
//! - gRPC server for Flight SQL queries (data plane)
//! - RPC (iroh/UDS) for health, metrics, shutdown (control plane)
//!
//! # Architecture
//!
//! ```text
//! Flight SQL Clients ──► gRPC Server ──► FlightService
//!                             │
//!                             └──► DuckDB / Dataset Backend
//!
//! Control ──► RPC endpoint ──► FlightService (health, metrics)
//! ```
//!
//! # Usage
//!
//! FlightService is typically started via the factory system:
//!
//! ```ignore
//! // In config.toml
//! [services]
//! startup = ["event", "registry", "policy", "model", "flight"]
//!
//! [flight]
//! host = "0.0.0.0"
//! port = 50051
//! default_dataset = "my-metrics"
//! ```

use crate::config::FlightConfig as HyprFlightConfig;
use anyhow::Result;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_service::Spawnable;
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{error, info};

/// Service name for registry and logging
pub const SERVICE_NAME: &str = "flight";

/// Flight's gRPC authentication and tenant-policy adapter.
pub struct TenantFlightAuthorizer {
    auth: crate::server::state::ResourceAuthState,
    policy_client: crate::services::PolicyClient,
}

impl TenantFlightAuthorizer {
    pub fn new(
        auth: crate::server::state::ResourceAuthState,
        policy_client: crate::services::PolicyClient,
    ) -> Self {
        Self {
            auth,
            policy_client,
        }
    }
}

#[tonic::async_trait]
impl hyprstream_flight::FlightAuthorizer for TenantFlightAuthorizer {
    async fn authorize(
        &self,
        authorization: Option<&str>,
        resource: &str,
        operation: &str,
    ) -> Result<(), hyprstream_flight::FlightAuthError> {
        let authorization = authorization.ok_or_else(|| {
            hyprstream_flight::FlightAuthError::Unauthenticated("Bearer token required".to_owned())
        })?;
        let token = authorization.strip_prefix("Bearer ").ok_or_else(|| {
            hyprstream_flight::FlightAuthError::Unauthenticated("Bearer token required".to_owned())
        })?;
        let claims = crate::server::middleware::verify_resource_token_claims(&self.auth, token)
            .await
            .map_err(|_| {
                hyprstream_flight::FlightAuthError::Unauthenticated(
                    "Invalid access token".to_owned(),
                )
            })?;
        if claims.cnf_jkt().is_some() {
            return Err(hyprstream_flight::FlightAuthError::Unauthenticated(
                "DPoP-bound tokens are not accepted on Flight without a proof transport".to_owned(),
            ));
        }
        let atproto_issuer =
            crate::services::oauth::state::canonical_issuer_origin(
                &self.auth.oauth_issuer_url,
            )
            .unwrap_or_else(|| self.auth.oauth_issuer_url.clone());
        let local_issuers =
            [self.auth.oauth_issuer_url.as_str(), atproto_issuer.as_str()];
        let subject = claims.subject(&local_issuers);
        subject.validate().map_err(|_| {
            hyprstream_flight::FlightAuthError::Unauthenticated(
                "Invalid access-token subject".to_owned(),
            )
        })?;
        let identity = crate::server::middleware::AuthenticatedUser {
            user: subject.name().ok_or_else(|| {
                hyprstream_flight::FlightAuthError::Unauthenticated(
                    "Invalid access-token subject".to_owned(),
                )
            })?.to_owned(),
            verified_tenant: claims.tenant,
            token: Some(token.to_owned()),
            exp: Some(claims.exp),
        };
        let domain = identity.authorization_domain().map_err(|_| {
            hyprstream_flight::FlightAuthError::Forbidden(
                "Verified hosted-account tenant binding required".to_owned(),
            )
        })?;
        let upstream_subject =
            hyprstream_rpc::envelope::Subject::new(identity.user.clone());
        let request = crate::services::generated::policy_client::PolicyCheck {
            subject: identity.user,
            domain,
            resource: resource.to_owned(),
            operation: operation.to_owned(),
        };
        let allowed = crate::services::policy::check_with_verified_bearer(
            &self.policy_client,
            &request,
            Some(token),
            &upstream_subject,
        )
        .await
        .unwrap_or(false);
        if allowed {
            Ok(())
        } else {
            Err(hyprstream_flight::FlightAuthError::Forbidden(
                "Flight request denied by tenant policy".to_owned(),
            ))
        }
    }
}

/// FlightService - Arrow Flight SQL with RPC control channel
///
/// This service provides:
/// - gRPC server with Flight SQL protocol for dataset queries
/// - RPC control channel for health checks
///
/// The Flight server uses hyprstream-flight crate for the actual
/// Flight SQL implementation with DuckDB backend.
pub struct FlightService {
    /// Flight configuration (host, port, dataset)
    config: HyprFlightConfig,

    /// Optional registry client for dataset lookup
    registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>>,

    /// Transport configuration for RPC control channel
    control_transport: TransportConfig,

    /// Verifying key for envelope verification
    #[allow(dead_code)]
    verifying_key: VerifyingKey,

    /// Mandatory JWT + hosted-tenant + Casbin boundary for the gRPC data plane.
    authorizer: Arc<dyn hyprstream_flight::FlightAuthorizer>,
}

impl FlightService {
    /// Create a new FlightService
    ///
    /// # Arguments
    ///
    /// * `config` - Flight configuration (host, port, dataset)
    /// * `registry_client` - Optional registry client for dataset lookup
    /// * `control_transport` - Transport for RPC control channel
    /// * `verifying_key` - Key for verifying signed envelopes
    pub fn new(
        config: HyprFlightConfig,
        registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>>,
        control_transport: TransportConfig,
        verifying_key: VerifyingKey,
        authorizer: Arc<dyn hyprstream_flight::FlightAuthorizer>,
    ) -> Self {
        Self {
            config,
            registry_client,
            control_transport,
            verifying_key,
            authorizer,
        }
    }
}

impl Spawnable for FlightService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        // Register control channel endpoint
        vec![(SocketKind::Rep, self.control_transport.clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        // Create multi-threaded runtime for gRPC server
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        rt.block_on(async move {
            // Build hyprstream_flight config from our config
            let flight_config = hyprstream_flight::FlightConfig::default()
                .with_host(&self.config.host)
                .with_port(self.config.port);

            // Get dataset name (empty string for in-memory mode)
            let dataset_name = self.config.default_dataset.as_deref().unwrap_or("");

            info!(
                "FlightService starting on {}:{} (dataset: {})",
                self.config.host,
                self.config.port,
                if dataset_name.is_empty() {
                    "<in-memory>"
                } else {
                    dataset_name
                }
            );

            // Signal ready before starting server
            // Note: The Flight server doesn't expose a "bound" callback,
            // so we signal ready optimistically
            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            // Notify systemd that service is ready
            let _ = hyprstream_rpc::notify::ready();

            // Run Flight server with shutdown handling
            // start_flight_server blocks, so we need to select! with shutdown
            tokio::select! {
                biased;

                _ = shutdown.notified() => {
                    info!("FlightService received shutdown signal");
                    // Note: Tonic server doesn't have graceful shutdown in this version
                    // It will be terminated when the runtime drops
                }

                result = hyprstream_flight::start_flight_server_with_authorizer(
                    self.registry_client.clone(),
                    dataset_name,
                    flight_config,
                    Arc::clone(&self.authorizer),
                ) => {
                    match result {
                        Ok(()) => info!("FlightService stopped normally"),
                        Err(e) => error!("FlightService error: {}", e),
                    }
                }
            }

            info!("FlightService stopped");
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_name() {
        assert_eq!(SERVICE_NAME, "flight");
    }
}
