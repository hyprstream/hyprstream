//! OAIService - OpenAI-compatible HTTP API
//!
//! The HTTP server handles OpenAI API requests and uses authenticated RPC
//! clients to reach Model, Policy, and Registry services.
//!
//! # Architecture
//!
//! ```text
//! HTTP Clients ──► HTTP Server (Axum) ──► OAIService
//!                        │
//!                        ├──► ModelClient ──► ModelService
//!                        └──► PolicyClient ──► PolicyService
//! ```
//!
//! # Usage
//!
//! OAIService is typically started via the factory system:
//!
//! ```ignore
//! // In config.toml
//! [services]
//! startup = ["event", "registry", "policy", "model", "oai"]
//!
//! [oai]
//! host = "0.0.0.0"
//! port = 8080
//! ```

use crate::config::{OAIConfig, TlsConfig};
use crate::server::tls::{bind_listener, resolve_rustls_config, serve_bound};
use crate::server::{create_app, state::ServerState};
use anyhow::Result;
use hyprstream_service::Spawnable;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tracing::info;

/// Service name for registry and logging
pub const SERVICE_NAME: &str = "oai";

const MODEL_MISSING_TENANT_DENIAL: &str = "authorization denied: no verified tenant domain";

/// Prove that the tenantless OAI service can reach the authenticated Model
/// dispatcher while preserving Model's tenant boundary.
///
/// The generated-module helper preserves the canonical method binding and
/// returns a typed response only after the response envelope has been
/// decrypted and verified against the resolved Model identity. Classify the
/// expected denial only after that verification.
pub(crate) async fn prove_model_reachability_denial(
    client: &crate::services::generated::model_client::ModelClient,
) -> Result<(), hyprstream_rpc::error::RpcError> {
    use crate::services::generated::model_client::{
        verified_health_check_response, ModelResponseVariant,
    };

    let response = verified_health_check_response(client)
        .await
        .map_err(|error| {
            hyprstream_rpc::error::RpcError::SpawnFailed(format!(
                "OAI Model reachability transport/authentication/response failed: {error}"
            ))
        })?;
    match response {
        ModelResponseVariant::Error(error)
            if error.code == "INTERNAL"
                && error.message == MODEL_MISSING_TENANT_DENIAL
                && error.details.is_empty() =>
        {
            Ok(())
        }
        ModelResponseVariant::HealthCheckResult(_) => {
            Err(hyprstream_rpc::error::RpcError::SpawnFailed(
                "OAI tenantless Model reachability probe was unexpectedly authorized".to_owned(),
            ))
        }
        response => Err(hyprstream_rpc::error::RpcError::SpawnFailed(format!(
            "OAI Model reachability returned an unexpected authenticated response: {response:?}"
        ))),
    }
}

pub(crate) async fn healthy_registry(
    client: &crate::services::RegistryClient,
) -> Result<(), hyprstream_rpc::error::RpcError> {
    let health = client.health_check().await.map_err(|error| {
        hyprstream_rpc::error::RpcError::SpawnFailed(format!(
            "OAI native Registry readiness: {error}"
        ))
    })?;
    if health.status != "healthy" {
        return Err(hyprstream_rpc::error::RpcError::SpawnFailed(format!(
            "OAI native Registry readiness returned status {:?}",
            health.status
        )));
    }
    Ok(())
}

pub(crate) async fn await_required_native_dependencies(
    model: &crate::services::generated::model_client::ModelClient,
    registry: &crate::services::RegistryClient,
    shutdown: Arc<Notify>,
    timeout: Duration,
) -> Result<(), hyprstream_rpc::error::RpcError> {
    let checks = async {
        prove_model_reachability_denial(model).await?;
        healthy_registry(registry).await?;
        Ok(())
    };

    tokio::select! {
        _ = shutdown.notified() => Err(hyprstream_rpc::error::RpcError::SpawnFailed(
            "OAI native dependency readiness cancelled".to_owned(),
        )),
        result = tokio::time::timeout(timeout, checks) => {
            result.map_err(|_| hyprstream_rpc::error::RpcError::SpawnFailed(format!(
                "OAI native dependency readiness timed out after {} milliseconds",
                timeout.as_millis()
            )))?
        }
    }
}

/// OAIService - OpenAI-compatible HTTP API
///
/// The HTTP server handles requests through authenticated Model, Policy, and
/// Registry clients. OAI has no inbound RPC control schema or handler.
pub struct OAIService {
    /// OAI-specific configuration (host, port, TLS)
    config: OAIConfig,

    /// Global TLS configuration (passed from factory, avoids re-loading config)
    tls_config: TlsConfig,

    /// Account-zone configuration (epic #1158, A3) — used for DNS-01 wildcard
    /// TLS provisioning. Passed from the factory to avoid re-loading config.
    account_config: crate::account::AccountZoneConfig,

    /// Shared server state containing clients and metrics
    server_state: ServerState,

    /// Required deployments must prove their outbound native dependencies
    /// before advertising HTTP readiness.
    native_dependencies_required: bool,
}

impl OAIService {
    /// Create a new OAIService
    ///
    /// # Arguments
    ///
    /// * `config` - OAI configuration (host, port, TLS settings)
    /// * `tls_config` - Global TLS configuration
    /// * `server_state` - Shared state with RPC clients and metrics
    /// * `native_dependencies_required` - gate readiness on authenticated
    ///   Model reachability and Registry health responses
    pub fn new(
        config: OAIConfig,
        tls_config: TlsConfig,
        account_config: crate::account::AccountZoneConfig,
        server_state: ServerState,
        native_dependencies_required: bool,
    ) -> Self {
        Self {
            config,
            tls_config,
            account_config,
            server_state,
            native_dependencies_required,
        }
    }

    /// Get the HTTP bind address
    pub fn http_addr(&self) -> Result<SocketAddr> {
        let addr_str = format!("{}:{}", self.config.host, self.config.port);
        addr_str
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid address: {}", e))
    }

    async fn await_native_dependencies(
        &self,
        shutdown: Arc<Notify>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        if !self.native_dependencies_required {
            return Ok(());
        }

        await_required_native_dependencies(
            &self.server_state.model_client,
            &self.server_state.registry,
            shutdown,
            Duration::from_secs(30),
        )
        .await
    }

    pub(crate) async fn run_inner(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        // Parse HTTP bind address
        let addr = self.http_addr().map_err(|e| {
            hyprstream_rpc::error::RpcError::SpawnFailed(format!("Invalid HTTP address: {e}"))
        })?;

        // Resolve TLS configuration (tls_config passed from factory, not re-loaded)
        let rustls_config = resolve_rustls_config(
            &self.tls_config,
            &self.account_config,
            self.config.tls_cert.as_ref(),
            self.config.tls_key.as_ref(),
        )
        .await
        .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("TLS config: {e}")))?;

        let scheme = if rustls_config.is_some() {
            "https"
        } else {
            "http"
        };

        // Create HTTP server
        let app = create_app(self.server_state.clone());

        info!("OpenAI-compatible API available at {scheme}://{addr}/oai/v1");

        // PREBIND the HTTP(S) listener BEFORE any readiness signal: an
        // occupied port fails here, before the supervisor is told the
        // service is ready (#1585 YuI7 companion correction).
        let bound = bind_listener(addr, rustls_config, "OAIService")?;

        // Required deployments use the process-owned bootstrap Iroh
        // endpoint for outbound RPC. Prove the actual Model and Registry
        // request paths before exposing the HTTP face as ready. Policy was
        // already probed by process authority-store initialization before
        // this factory ran.
        self.await_native_dependencies(Arc::clone(&shutdown))
            .await?;

        // Signal ready only after the listener and required native
        // dependencies are ready.
        if let Some(tx) = on_ready {
            let _ = tx.send(());
        }

        // Notify systemd that service is ready
        let _ = hyprstream_rpc::notify::ready();

        // Run HTTP(S) server with graceful shutdown; the serving result
        // propagates (post-READY runtime failure — recorded lifecycle gap).
        serve_bound(bound, app, shutdown, "OAIService").await
    }
}

impl Spawnable for OAIService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn registrations(
        &self,
    ) -> Vec<(
        hyprstream_rpc::SocketKind,
        hyprstream_rpc::transport::TransportConfig,
    )> {
        Vec::new()
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        // Create multi-threaded runtime for HTTP server
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        rt.block_on(self.run_inner(shutdown, on_ready))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_name() {
        assert_eq!(SERVICE_NAME, "oai");
    }
}
