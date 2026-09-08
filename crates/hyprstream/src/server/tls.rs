//! Shared TLS materials and HTTPS serving helpers for HTTP services.
//!
//! Provides:
//! - `TlsMaterials`: DER-encoded cert + key pair
//! - `get_or_init_tls_materials()`: OnceLock-based lazy generation (self-signed or file-loaded)
//! - `resolve_rustls_config()`: Resolves per-service or shared TLS config
//! - `serve_app()`: Serves an Axum router over HTTPS or HTTP with graceful shutdown
//!
//! TLS modes (set via `[tls] mode`):
//!  - `self-signed` (default) — rcgen-generated, dev/air-gapped only
//!  - `acme` — automatic via RFC 8555 (Let's Encrypt or self-hosted step-ca / Pebble)
//!  - `files` — operator-supplied PEM paths

use crate::account::AccountZoneConfig;
use crate::config::{TlsConfig, TlsMode};
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
        let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
        let server_name = &config.server_name;
        info!(
            "Loading/generating TLS materials (365-day validity) from '{}'",
            secrets_dir.display()
        );
        crate::auth::identity_store::load_or_generate_tls_materials(&secrets_dir, server_name, 365)?
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
/// 2. Both `service_cert` + `service_key` set → per-service PEM files (overrides mode)
/// 3. `mode = "acme"` → obtain/renew via ACME; spawns background renewal task
/// 4. Otherwise → shared self-signed / `files` mode materials
pub async fn resolve_rustls_config(
    tls_config: &TlsConfig,
    account_cfg: &AccountZoneConfig,
    service_cert: Option<&PathBuf>,
    service_key: Option<&PathBuf>,
) -> anyhow::Result<Option<axum_server::tls_rustls::RustlsConfig>> {
    if !tls_config.enabled {
        return Ok(None);
    }

    // Ensure the PQ-hybrid (aws-lc-rs, X25519MLKEM768) crypto provider is
    // installed for rustls (#557 / S6). Services that don't use QUIC (which
    // installs it via quinn) need this before any RustlsConfig can be created.
    // Idempotent, but fail closed if another provider was installed first.
    hyprstream_rpc::transport::install_pq_crypto_provider()?;

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

    match tls_config.effective_mode() {
        TlsMode::Acme => {
            let config = init_acme_rustls_config(tls_config).await?;
            Ok(Some(config))
        }
        TlsMode::AcmeDns01 => {
            // Account-zone DNS-01 wildcard issuance (epic #1158, A3 / #1162).
            // The concrete ACME-DNS-01 client and DNS backend are product-layer
            // seams (`hyprstream::account::{WildcardCertIssuer, DnsProvider}`);
            // here we only sanity-check provisioning. When the deployment
            // configured no account zone or DNS-01 credential, this mode
            // **degrades sanely** to the existing shared self-signed materials
            // — no panic, no synthesized zone name (one-way door #1).
            match crate::account::tls::require_provisioned(account_cfg) {
                Ok(zone) => {
                    // NOTE (#1182 review): this branch validates that an account
                    // zone + zone-scoped DNS-01 credential are *configured*. It
                    // does NOT yet invoke `WildcardCertIssuer` or bind a
                    // `CertHandle` to this listener — those are product-layer /
                    // B3 (#1165) seams. Until that integration lands, the
                    // account zone's wildcard cert is NOT what this service
                    // serves; we fall through to shared self-signed materials so
                    // the deployment keeps booting. Do not read the log below as
                    // "the wildcard cert is live"; it means "config is
                    // well-formed and provisioning is detectable".
                    tracing::info!(
                        apex = %zone.apex(),
                        wildcard = %zone.wildcard_domain(),
                        "TLS mode acme-dns01: account-zone config is provisioned (zone + \
                         zone-scoped DNS-01 credential present). Wildcard issuance + rustls \
                         binding are product-layer/B3 (#1165) seams not yet wired here — \
                         serving shared self-signed TLS until then.",
                    );
                    let materials = get_or_init_tls_materials(tls_config)?;
                    let rustls_config = axum_server::tls_rustls::RustlsConfig::from_der(
                        vec![materials.cert_der.clone()],
                        (*materials.key_der).clone(),
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!("failed to build RustlsConfig from DER: {}", e))?;
                    Ok(Some(rustls_config))
                }
                Err(e) => {
                    // Sane-degrade: clear error in the log, fall back to shared
                    // self-signed materials. We do NOT bail: account minting is
                    // disabled, but the rest of the deployment (OAI/OAuth/MCP
                    // faces) must still boot.
                    tracing::warn!(
                        "TLS mode acme-dns01 requested but account zone is not provisioned \
                         — did:web minting disabled, serving with shared self-signed TLS: {e}"
                    );
                    let materials = get_or_init_tls_materials(tls_config)?;
                    let rustls_config = axum_server::tls_rustls::RustlsConfig::from_der(
                        vec![materials.cert_der.clone()],
                        (*materials.key_der).clone(),
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!("failed to build RustlsConfig from DER: {}", e))?;
                    Ok(Some(rustls_config))
                }
            }
        }
        _ => {
            // Shared materials (self-signed or Files mode)
            let materials = get_or_init_tls_materials(tls_config)?;
            let rustls_config = axum_server::tls_rustls::RustlsConfig::from_der(
                vec![materials.cert_der.clone()],
                (*materials.key_der).clone(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("failed to build RustlsConfig from DER: {}", e))?;
            Ok(Some(rustls_config))
        }
    }
}

/// Initialize an ACME-managed TLS config and spawn the renewal event-loop task.
///
/// Uses `rustls-acme` which handles ACME-01 TLS-ALPN challenges internally and
/// updates the `Arc<rustls::ServerConfig>` in-place on each renewal — no explicit
/// hot-swap needed.
///
/// For self-hosted ACME, point `acme_directory` at a step-ca or Pebble URL.
/// See `docs/tls.md` for operator guidance.
async fn init_acme_rustls_config(
    tls_config: &TlsConfig,
) -> anyhow::Result<axum_server::tls_rustls::RustlsConfig> {
    use rustls_acme::AcmeConfig;

    let domain = tls_config.acme_domain.as_deref().ok_or_else(|| {
        anyhow::anyhow!("[tls] mode = \"acme\" requires acme_domain to be set")
    })?;
    let contact = tls_config.acme_contact.as_deref().unwrap_or("");
    let cache_dir = tls_config.acme_cache_dir.clone().unwrap_or_else(|| {
        dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/var/cache"))
            .join("hyprstream")
            .join("acme")
    });

    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| anyhow::anyhow!("cannot create ACME cache dir {:?}: {e}", cache_dir))?;

    let mut acme = AcmeConfig::new([domain])
        .directory_lets_encrypt(true)
        .contact_push(contact)
        .cache_option(Some(rustls_acme::caches::DirCache::new(cache_dir)));

    if let Some(dir_url) = tls_config.acme_directory.as_deref() {
        acme = acme.directory(dir_url);
    }

    info!("ACME TLS: obtaining certificate for domain '{}' from {}", domain,
        tls_config.acme_directory.as_deref().unwrap_or("Let's Encrypt"));

    let mut state = acme.state();
    let rustls_server_config = acme_https_config(state.resolver());

    // Spawn the ACME event loop — handles challenge responses and renewals.
    tokio::spawn(async move {
        use tokio_stream::StreamExt;
        loop {
            match state.next().await {
                Some(Ok(event)) => {
                    info!("ACME event: {event:?}");
                }
                Some(Err(e)) => {
                    tracing::warn!("ACME error (will retry): {e}");
                }
                None => break,
            }
        }
    });

    // Wrap the rustls ServerConfig in an axum-server RustlsConfig.
    // rustls-acme updates the shared certificate resolver on renewal.
    let rustls_config = axum_server::tls_rustls::RustlsConfig::from_config(rustls_server_config);
    Ok(rustls_config)
}

/// A bound HTTP(S) listener retained between the bind and serve phases.
///
/// Bind early (before readiness signals); serve later, consuming the handle.
/// No second bind occurs: the TLS arm keeps the already-bound std listener
/// and hands it to the pinned `axum-server` prebound constructor.
pub enum BoundHttpListener {
    /// Prebound nonblocking listener adopted by tokio.
    Http(tokio::net::TcpListener),
    /// Prebound std listener + TLS config, served via
    /// `axum_server::from_tcp_rustls`.
    Https(std::net::TcpListener, axum_server::tls_rustls::RustlsConfig),
}

/// Bind phase: a real `std::net::TcpListener::bind` plus `set_nonblocking`.
///
/// Call this BEFORE any readiness signal so a bind failure (e.g. an occupied
/// port) surfaces before the launcher is told the service is ready.
pub fn bind_listener(
    addr: SocketAddr,
    rustls_config: Option<axum_server::tls_rustls::RustlsConfig>,
    service_name: &str,
) -> Result<BoundHttpListener, RpcError> {
    let std_listener = std::net::TcpListener::bind(addr).map_err(|e| {
        RpcError::SpawnFailed(format!("{service_name} HTTP(S) bind failed: {e}"))
    })?;
    std_listener.set_nonblocking(true).map_err(|e| {
        RpcError::SpawnFailed(format!(
            "{service_name} HTTP(S) set_nonblocking failed: {e}"
        ))
    })?;
    match rustls_config {
        Some(tls) => Ok(BoundHttpListener::Https(std_listener, tls)),
        None => Ok(BoundHttpListener::Http(
            tokio::net::TcpListener::from_std(std_listener).map_err(|e| {
                RpcError::SpawnFailed(format!(
                    "{service_name} HTTP listener adoption failed: {e}"
                ))
            })?,
        )),
    }
}

/// Serve phase: consume a [`BoundHttpListener`] with graceful shutdown.
///
/// The TLS arm uses the pinned free function
/// `axum_server::from_tcp_rustls(listener, tls)` on the already-bound std
/// listener — no second bind.
pub async fn serve_bound(
    bound: BoundHttpListener,
    app: axum::Router,
    shutdown: Arc<Notify>,
    service_name: &str,
) -> Result<(), RpcError> {
    match bound {
        BoundHttpListener::Https(std_listener, tls) => {
            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            let name = service_name.to_owned();

            // Spawn shutdown listener
            tokio::spawn(async move {
                shutdown.notified().await;
                info!("{name} received shutdown signal");
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(30)));
            });

            axum_server::from_tcp_rustls(std_listener, tls)
                .handle(handle)
                .serve(app.into_make_service())
                .await
                .map_err(|e| RpcError::SpawnFailed(format!("{service_name} HTTPS server error: {e}")))?;
        }
        BoundHttpListener::Http(listener) => {
            let shutdown_clone = shutdown.clone();
            let name = service_name.to_owned();
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

/// Serve an Axum router over HTTPS (if rustls_config is Some) or plain HTTP.
///
/// Uses `axum_server` for HTTPS with `Handle`-based graceful shutdown,
/// or standard `axum::serve` for HTTP. Bind-then-serve wrapper over
/// [`bind_listener`] + [`serve_bound`]; existing callers are unchanged.
pub async fn serve_app(
    addr: SocketAddr,
    app: axum::Router,
    rustls_config: Option<axum_server::tls_rustls::RustlsConfig>,
    shutdown: Arc<Notify>,
    service_name: &str,
) -> Result<(), RpcError> {
    let scheme = if rustls_config.is_some() { "https" } else { "http" };
    info!("{service_name} listening on {scheme}://{addr}");

    let bound = bind_listener(addr, rustls_config, service_name)?;
    serve_bound(bound, app, shutdown, service_name).await
}

/// The shared listener must negotiate both HTTP and TLS-ALPN-01. The ACME
/// resolver selects the challenge certificate only for an ACME ClientHello;
/// ordinary clients receive the currently issued certificate.
fn acme_https_config(
    resolver: Arc<dyn rustls::server::ResolvesServerCert>,
) -> Arc<rustls::ServerConfig> {
    let mut config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_cert_resolver(resolver);
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec(), b"acme-tls/1".to_vec()];
    Arc::new(config)
}

#[cfg(test)]
mod acme_tests {
    use super::*;
    use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
    use std::io::Cursor;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct FixtureResolver {
        key: Arc<rustls::sign::CertifiedKey>,
        challenge_seen: AtomicBool,
    }

    impl rustls::server::ResolvesServerCert for FixtureResolver {
        fn resolve(
            &self,
            hello: rustls::server::ClientHello<'_>,
        ) -> Option<Arc<rustls::sign::CertifiedKey>> {
            self.challenge_seen
                .store(rustls_acme::is_tls_alpn_challenge(&hello), Ordering::SeqCst);
            Some(self.key.clone())
        }
    }

    /// Exercise the actual production config with real ClientHello negotiation.
    /// Replacing its ALPN list with the old challenge-only list fails HTTP.
    #[test]
    fn acme_https_negotiates_http_and_challenge() -> anyhow::Result<()> {
        hyprstream_rpc::transport::install_pq_crypto_provider()?;
        let generated = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])?;
        let cert = generated.cert.der().clone();
        let private =
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(generated.key_pair.serialize_der()));
        let provider = rustls::crypto::CryptoProvider::get_default()
            .ok_or_else(|| anyhow::anyhow!("crypto provider missing"))?;
        let key = Arc::new(rustls::sign::CertifiedKey::from_der(
            vec![cert.clone()],
            private,
            provider,
        )?);
        for protocol in [b"h2".as_slice(), b"http/1.1", b"acme-tls/1"] {
            let resolver = Arc::new(FixtureResolver {
                key: key.clone(),
                challenge_seen: AtomicBool::new(false),
            });
            let mut roots = rustls::RootCertStore::empty();
            roots.add(cert.clone())?;
            let mut client_config = rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth();
            client_config.alpn_protocols = vec![protocol.to_vec()];
            let mut client = rustls::ClientConnection::new(
                Arc::new(client_config),
                ServerName::try_from("localhost")?,
            )?;
            let mut server = rustls::ServerConnection::new(acme_https_config(resolver.clone()))?;
            for _ in 0..16 {
                let mut bytes = Vec::new();
                client.write_tls(&mut bytes)?;
                server.read_tls(&mut Cursor::new(bytes))?;
                server.process_new_packets()?;
                let mut bytes = Vec::new();
                server.write_tls(&mut bytes)?;
                client.read_tls(&mut Cursor::new(bytes))?;
                client.process_new_packets()?;
                if !client.is_handshaking() && !server.is_handshaking() {
                    break;
                }
            }
            assert!(!client.is_handshaking() && !server.is_handshaking());
            assert_eq!(client.alpn_protocol(), Some(protocol));
            assert_eq!(
                resolver.challenge_seen.load(Ordering::SeqCst),
                protocol == b"acme-tls/1"
            );
        }
        Ok(())
    }
}


#[cfg(test)]
mod bound_serve_tests {
    use super::*;
    use axum::routing::get;
    use rustls::pki_types::{CertificateDer, ServerName};

    /// Positive HTTPS regression through the production bind/serve seam:
    /// `bind_listener` retains the real socket, `serve_bound` hands it to the
    /// pinned `axum_server::from_tcp_rustls` constructor, a client that
    /// TRUSTS the served certificate and verifies the hostname normally (no
    /// verification bypass) completes a real request, and shutdown is
    /// bounded. Same helper OAuth and OAI call before their readiness
    /// signals.
    #[tokio::test]
    async fn bound_https_serves_verified_client_and_shuts_down_bounded()
    -> anyhow::Result<()> {
        hyprstream_rpc::transport::install_pq_crypto_provider()?;
        let generated = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])?;
        let rustls_config = axum_server::tls_rustls::RustlsConfig::from_pem(
            generated.cert.pem().into_bytes(),
            generated.key_pair.serialize_pem().into_bytes(),
        )
        .await?;

        // Reserve, then bind through the production seam (no second bind).
        let probe = std::net::TcpListener::bind("127.0.0.1:0")?;
        let addr = probe.local_addr()?;
        drop(probe);
        let bound = bind_listener(addr, Some(rustls_config), "BoundServeTest")?;

        let app = axum::Router::new().route(
            "/bound-serve-probe",
            get(|| async { "bound-serve-ok" }),
        );
        let shutdown = Arc::new(Notify::new());
        let shutdown_signal = Arc::clone(&shutdown);
        let server = tokio::spawn(serve_bound(bound, app, Arc::clone(&shutdown), "BoundServeTest"));

        // Client trusts the served certificate and verifies "localhost"
        // normally; no dangerous verification-disabling configuration.
        let mut roots = rustls::RootCertStore::empty();
        roots.add(CertificateDer::from(generated.cert.der().clone()))?;
        let client_config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        // The client runs on a blocking thread over a real socket. `StreamOwned`
        // implements blocking Read/Write that drives rustls IO internally (the
        // raw ClientConnection reader only drains buffered plaintext); hostname
        // and certificate verification stay fully enabled.
        let client = std::thread::spawn(move || -> anyhow::Result<String> {
            use std::io::{Read, Write};
            let mut tcp = std::net::TcpStream::connect(addr)?;
            tcp.set_read_timeout(Some(Duration::from_secs(10)))?;
            let tls = rustls::ClientConnection::new(
                Arc::new(client_config),
                ServerName::try_from("localhost")?,
            )?;
            let mut stream = rustls::StreamOwned::new(tls, tcp);
            stream.write_all(
                b"GET /bound-serve-probe HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )?;
            let mut response = Vec::new();
            stream.read_to_end(&mut response)?;
            Ok(String::from_utf8_lossy(&response).into_owned())
        });

        let response = tokio::task::spawn_blocking(move || client.join())
            .await
            .expect("client joiner must not fail")
            .expect("client thread must not panic")?;
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "verified-TLS client must receive HTTP 200: got {response:?}"
        );
        assert!(
            response.contains("bound-serve-ok"),
            "exact probe body must be served through the TLS path: got {response:?}"
        );

        // Bounded shutdown: the retained handle's server ends cleanly.
        shutdown_signal.notify_one();
        tokio::time::timeout(Duration::from_secs(30), server)
            .await
            .expect("serve_bound must terminate within the shutdown budget")?
            .expect("graceful shutdown must complete Ok");
        Ok(())
    }

}
