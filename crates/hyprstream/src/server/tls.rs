//! Shared TLS materials and HTTPS serving helpers for HTTP services.
//!
//! Provides:
//! - `TlsMaterials`: DER-encoded cert + key pair
//! - `get_or_init_tls_materials()`: OnceLock-based lazy generation (self-signed or file-loaded)
//! - `resolve_rustls_config()`: Resolves per-service or shared TLS config
//! - `serve_app()`: Serves an Axum router over HTTPS or HTTP with graceful shutdown

use crate::config::TlsConfig;
use hyprstream_rpc::error::RpcError;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::Notify;
use tracing::info;
use zeroize::Zeroizing;

/// DER-encoded TLS certificate and private key.
///
/// Key material is wrapped in `Zeroizing` to ensure it is zeroed on drop,
/// preventing private key bytes from lingering in freed heap memory.
#[derive(Clone)]
pub struct TlsMaterials {
    pub cert_der: Vec<u8>,
    pub key_der: Zeroizing<Vec<u8>>,
}

/// Shared self-signed TLS materials (generated once, reused across all services).
///
/// Stored behind an `Arc` so callers share the same allocation rather than
/// cloning key bytes on every call.
static SHARED_TLS: OnceLock<Arc<TlsMaterials>> = OnceLock::new();

/// Get or initialize the shared TLS materials.
///
/// - If already initialized, returns a clone of the `Arc` (no key bytes copied).
/// - If both `cert_path`/`key_path` are set in config, loads from PEM files.
/// - Otherwise, loads persisted materials from `secrets.path` (generating on first run).
///   The key is stable across restarts; certs are renewed when approaching expiry.
pub fn get_or_init_tls_materials(config: &TlsConfig) -> anyhow::Result<Arc<TlsMaterials>> {
    if let Some(existing) = SHARED_TLS.get() {
        return Ok(Arc::clone(existing));
    }

    let materials = if config.use_self_signed() {
        let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir();
        let server_name = &config.server_name;
        info!(
            "Loading/generating TLS materials (365-day validity) from '{}'",
            secrets_dir.display()
        );
        crate::auth::credentials::load_or_generate_tls_materials(&secrets_dir, server_name, 365)?
    } else {
        // Both paths guaranteed Some when use_self_signed() returns false
        let cert_path = config.cert_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("tls.cert_path must be set when tls.key_path is set"))?;
        let key_path = config.key_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("tls.key_path must be set when tls.cert_path is set"))?;

        let cert_pem = std::fs::read(cert_path)
            .map_err(|e| anyhow::anyhow!("failed to read tls.cert_path '{}': {}", cert_path.display(), e))?;
        let key_pem = std::fs::read(key_path)
            .map_err(|e| anyhow::anyhow!("failed to read tls.key_path '{}': {}", key_path.display(), e))?;

        let cert_der = rustls_pemfile::certs(&mut &cert_pem[..])
            .next()
            .ok_or_else(|| anyhow::anyhow!("no certificate found in {}", cert_path.display()))?
            .map_err(|e| anyhow::anyhow!("invalid certificate PEM: {}", e))?
            .to_vec();

        let key_der = Zeroizing::new(
            rustls_pemfile::private_key(&mut &key_pem[..])
                .map_err(|e| anyhow::anyhow!("invalid key PEM: {}", e))?
                .ok_or_else(|| anyhow::anyhow!("no private key found in {}", key_path.display()))?
                .secret_der()
                .to_vec(),
        );

        TlsMaterials { cert_der, key_der }
    };

    // If another thread beat us, use theirs (Arc::clone, no key bytes copied)
    Ok(Arc::clone(SHARED_TLS.get_or_init(|| Arc::new(materials))))
}

/// Resolve the rustls configuration for a service.
///
/// Resolution order:
/// 1. `tls_config.enabled == false` → `None` (plain HTTP)
/// 2. Both `service_cert` + `service_key` set → per-service PEM files
/// 3. Otherwise → shared self-signed/configured materials
pub async fn resolve_rustls_config(
    tls_config: &TlsConfig,
    service_cert: Option<&PathBuf>,
    service_key: Option<&PathBuf>,
) -> anyhow::Result<Option<axum_server::tls_rustls::RustlsConfig>> {
    if !tls_config.enabled {
        return Ok(None);
    }

    // Ensure the ring crypto provider is installed for rustls.
    // Services that don't use QUIC (which installs it via quinn) need this
    // before any RustlsConfig can be created. Idempotent — ignores if already set.
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Per-service override: both cert and key must be set
    if let (Some(cert), Some(key)) = (service_cert, service_key) {
        let rustls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key)
            .await
            .map_err(|e| anyhow::anyhow!("failed to load per-service TLS from {:?}/{:?}: {}", cert, key, e))?;
        return Ok(Some(rustls_config));
    }

    // Warn if only one of cert/key is set at the per-service level
    if service_cert.is_some() != service_key.is_some() {
        tracing::warn!(
            "Per-service TLS requires both tls_cert and tls_key — falling back to shared TLS config"
        );
    }

    // Shared materials (self-signed or from global [tls] config)
    let materials = get_or_init_tls_materials(tls_config)?;
    let rustls_config = axum_server::tls_rustls::RustlsConfig::from_der(
        vec![materials.cert_der.clone()],
        (*materials.key_der).clone(),
    )
    .await
    .map_err(|e| anyhow::anyhow!("failed to build RustlsConfig from DER: {}", e))?;

    Ok(Some(rustls_config))
}

/// Serve an Axum router over HTTPS (if rustls_config is Some) or plain HTTP.
///
/// Uses `axum_server` for HTTPS with `Handle`-based graceful shutdown,
/// or standard `axum::serve` for HTTP.
pub async fn serve_app(
    addr: SocketAddr,
    app: axum::Router,
    rustls_config: Option<axum_server::tls_rustls::RustlsConfig>,
    shutdown: Arc<Notify>,
    service_name: &str,
) -> Result<(), RpcError> {
    let scheme = if rustls_config.is_some() { "https" } else { "http" };
    info!("{service_name} listening on {scheme}://{addr}");

    match rustls_config {
        Some(tls) => {
            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            let name = service_name.to_owned();

            // Spawn shutdown listener
            tokio::spawn(async move {
                shutdown.notified().await;
                info!("{name} received shutdown signal");
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(30)));
            });

            axum_server::bind_rustls(addr, tls)
                .handle(handle)
                .serve(app.into_make_service())
                .await
                .map_err(|e| RpcError::SpawnFailed(format!("{service_name} HTTPS server error: {e}")))?;
        }
        None => {
            let name = service_name.to_owned();
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| RpcError::SpawnFailed(format!("{name} HTTP bind failed: {e}")))?;

            let shutdown_clone = shutdown.clone();
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    shutdown_clone.notified().await;
                    info!("{name} received shutdown signal");
                })
                .await
                .map_err(|e| RpcError::SpawnFailed(format!("{service_name} HTTP server error: {e}")))?;
        }
    }

    info!("{service_name} stopped");
    Ok(())
}
