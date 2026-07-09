//! Unified configuration system for Hyprstream
//!
//! This module provides a layered configuration architecture:
//! - `HyprConfig`: Root configuration combining all subsystems
//! - `ServerConfig`: HTTP server configuration (network, CORS, TLS)
//! - Model and runtime configs for ML inference

pub mod server;

// Re-export main configuration types
pub use server::{CorsConfig, SamplingParamDefaults, ServerConfig, ServerConfigBuilder};

// Export root configuration and builder (defined below in this module)
// Note: HyprConfig and HyprConfigBuilder are exported automatically as pub structs

use crate::runtime::generation_metrics::GenerationQualityMetrics;
use crate::storage::paths::StoragePaths;
use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use zeroize::{Zeroize, Zeroizing};

/// Unified configuration for the Hyprstream system
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct HyprConfig {
    /// HTTP server configuration
    #[serde(default)]
    pub server: ServerConfig,

    /// Model configuration
    #[serde(default)]
    pub model: ModelConfig,

    /// Runtime execution settings
    #[serde(default)]
    pub runtime: RuntimeConfig,

    /// Text generation parameters
    #[serde(default)]
    pub generation: GenerationConfig,

    /// LoRA adapter settings
    #[serde(default)]
    pub lora: LoraAppConfig,

    /// Storage paths configuration
    #[serde(default)]
    pub storage: StorageConfig,

    /// Git storage and P2P transport configuration
    #[serde(default)]
    pub git2db: git2db::config::Git2DBConfig,

    /// Worker service configuration (optional)
    ///
    /// When present, the WorkerService will be started for container/sandbox management.
    /// This enables Kata-based isolated workload execution.
    #[serde(default)]
    pub worker: Option<hyprstream_workers::config::WorkerConfig>,

    /// Service management configuration
    ///
    /// Controls which services are started at startup in ipc-systemd mode.
    #[serde(default)]
    pub services: ServicesConfig,

    /// JWT token configuration
    #[serde(default)]
    pub token: TokenConfig,

    /// OpenAI-compatible HTTP API configuration
    #[serde(default)]
    pub oai: OAIConfig,

    /// HuggingFace-XET CAS HTTP face configuration (epic #654)
    #[serde(default)]
    pub xet: XetConfig,

    /// Arrow Flight SQL server configuration
    #[serde(default)]
    pub flight: FlightConfig,

    /// MCP service configuration (HTTP/SSE for Model Context Protocol)
    #[serde(default)]
    pub mcp: MCPConfig,

    /// OAuth 2.1 authorization server configuration
    #[serde(default)]
    pub oauth: OAuthConfig,

    /// Credentials storage backend (user profiles, pubkeys, refresh tokens).
    #[serde(default)]
    pub credentials: CredentialsConfig,

    /// StreamService configuration (buffer sizes, TTL, etc.)
    #[serde(default)]
    pub streaming: StreamingConfig,

    /// RPC transport server tunables (stream cap, connection cap, timeouts).
    ///
    /// All values default to the process-wide constants in `hyprstream-rpc`
    /// (`DEFAULT_STREAM_LIMIT=64`, `DEFAULT_CONNECTION_LIMIT=256`, etc.).
    /// Override via `[rpc]` in the config file or `HYPRSTREAM__RPC__*` env vars.
    #[serde(default)]
    pub rpc: RpcServerConfig,

    /// TLS configuration for HTTP services (OAI, OAuth, MCP)
    ///
    /// Enabled by default. Auto-generates self-signed cert when paths are unset.
    /// Per-service `tls_cert`/`tls_key` overrides take precedence.
    #[serde(default)]
    pub tls: TlsConfig,

    /// QUIC/WebTransport configuration
    ///
    /// Enabled by default. Services expose a WebTransport endpoint alongside ZMQ,
    /// allowing browsers to connect directly via HTTP/3 + QUIC.
    /// Set `enabled = false` to disable.
    #[serde(default)]
    pub quic: QuicConfig,

    /// Event proxy service configuration
    #[serde(default)]
    pub event: EventServiceConfig,

    /// Registry service configuration
    #[serde(default)]
    pub registry: RegistryServiceConfig,

    /// Policy service configuration
    #[serde(default)]
    pub policy: PolicyServiceConfig,

    /// Discovery service configuration
    #[serde(default)]
    pub discovery: DiscoveryServiceConfig,

    /// TUI display server configuration
    #[serde(default)]
    pub tui: TuiServiceConfig,

    /// Metrics service configuration (DuckDB/DataFusion ingest + query)
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Hex-encoded Ed25519 node signing key bytes (bypasses OS keyring lookup).
    ///
    /// **TEST USE ONLY.** Set via `HYPRSTREAM__SIGNING_KEY` env var to inject a
    /// pre-generated key in isolated test environments where a keyring daemon is
    /// unavailable. Never set this in production config files.
    #[serde(default, skip_serializing)]
    pub signing_key: Option<String>,

    /// Persistent secrets storage configuration.
    ///
    /// Controls where signing keys and TLS materials are read from and written to.
    /// On systemd, overridden at runtime by
    /// `HYPRSTREAM__SECRETS__PATH=%d` in the service unit (pointing to the
    /// systemd credentials directory).
    #[serde(default)]
    pub secrets: SecretsConfig,
}

/// Persistent secrets storage configuration.
///
/// Determines the directory used for reading and writing persistent secret key
/// material: signing keys and TLS certificates/keys.
///
/// On systemd, the generated service unit sets
/// `Environment=HYPRSTREAM__SECRETS__PATH=%d` so that at runtime `path` resolves
/// to the systemd credentials directory (non-swappable ramfs, access-restricted).
///
/// On non-systemd systems the default (`<config_dir>/credentials`) is used.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecretsConfig {
    /// Override the directory from which secret files are read (and written on
    /// first run).  `None` → resolved at runtime to `<config_dir>/credentials`.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

impl SecretsConfig {
    /// Resolve the secrets directory.
    ///
    /// Returns `path` if set, otherwise `<config_dir>/credentials`.
    pub fn resolve_dir(&self, config_dir: &Path) -> PathBuf {
        self.path.clone().unwrap_or_else(|| config_dir.join("credentials"))
    }

    /// Return the default secrets directory when no config is available.
    ///
    /// Routes through `StoragePaths` so `XDG_CONFIG_HOME` is respected
    /// consistently with every other directory in the application.
    /// Falls back to `/etc/hyprstream/credentials` if XDG resolution fails.
    pub fn default_dir() -> PathBuf {
        StoragePaths::new()
            .and_then(|p| p.config_dir())
            .map(|d| d.join("credentials"))
            .unwrap_or_else(|_| PathBuf::from("/etc/hyprstream/credentials"))
    }
}

/// TLS configuration for HTTP services (OAI, OAuth, MCP).
///
/// When `cert_path`/`key_path` are unset, a self-signed ECDSA P-256 certificate
/// is auto-generated at startup with 365-day validity. Per-service overrides
/// (`tls_cert`/`tls_key` on OAI/OAuth/MCP configs) take precedence.
///
/// # Example TOML
///
/// ```toml
/// [tls]
/// mode = "self-signed"   # or "acme" or "files"
/// server_name = "localhost"
/// # ACME mode:
/// # acme_domain = "node.example.com"
/// # acme_contact = "mailto:ops@example.com"
/// # acme_cache_dir = "/var/lib/hyprstream/acme"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TlsMode {
    /// Auto-generate a self-signed certificate at startup (dev/air-gapped only).
    #[default]
    SelfSigned,
    /// Obtain a certificate automatically via ACME (RFC 8555) — Let's Encrypt or step-ca.
    Acme,
    /// Load certificate and key from `cert_path`/`key_path` (operator-managed).
    Files,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Whether TLS is enabled for HTTP services (defaults to true)
    #[serde(default = "default_tls_enabled")]
    pub enabled: bool,

    /// TLS provisioning mode. Defaults to `self-signed` when cert_path/key_path are unset.
    #[serde(default)]
    pub mode: TlsMode,

    /// Path to TLS certificate (PEM). Used when mode = "files".
    #[serde(default)]
    pub cert_path: Option<PathBuf>,

    /// Path to TLS private key (PEM). Used when mode = "files".
    #[serde(default)]
    pub key_path: Option<PathBuf>,

    /// Server name for TLS certificate (self-signed CN or ACME domain).
    #[serde(default = "default_tls_server_name")]
    pub server_name: String,

    /// ACME: domain to obtain a certificate for. Required when mode = "acme".
    #[serde(default)]
    pub acme_domain: Option<String>,

    /// ACME: contact email URI, e.g. "mailto:ops@example.com".
    #[serde(default)]
    pub acme_contact: Option<String>,

    /// ACME: directory URL. Defaults to Let's Encrypt production.
    /// Set to a step-ca or Pebble URL for self-hosted ACME.
    #[serde(default)]
    pub acme_directory: Option<String>,

    /// ACME: directory for certificate cache and account keys.
    #[serde(default)]
    pub acme_cache_dir: Option<PathBuf>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: TlsMode::SelfSigned,
            cert_path: None,
            key_path: None,
            server_name: default_tls_server_name(),
            acme_domain: None,
            acme_contact: None,
            acme_directory: None,
            acme_cache_dir: None,
        }
    }
}

impl TlsConfig {
    /// Effective TLS mode, resolving legacy cert_path/key_path into Files mode.
    pub fn effective_mode(&self) -> TlsMode {
        // Explicit mode overrides; legacy cert_path/key_path implies Files.
        if self.mode != TlsMode::SelfSigned {
            return self.mode.clone();
        }
        if self.cert_path.is_some() && self.key_path.is_some() {
            return TlsMode::Files;
        }
        TlsMode::SelfSigned
    }

    /// Check if self-signed certificate should be generated.
    pub fn use_self_signed(&self) -> bool {
        self.effective_mode() == TlsMode::SelfSigned
    }
}

fn default_tls_enabled() -> bool { true }
fn default_tls_server_name() -> String { "localhost".to_owned() }

/// QUIC/WebTransport transport configuration.
///
/// Enables WebTransport alongside ZMQ for browser-direct RPC.
/// When `cert_path` is empty, a self-signed certificate is generated at startup.
///
/// # Example TOML
///
/// ```toml
/// [quic]
/// enabled = true
/// bind_addr = "0.0.0.0:4433"
/// server_name = "localhost"
/// cert_path = ""
/// key_path = ""
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicConfig {
    /// Whether QUIC/WebTransport is enabled (defaults to true)
    #[serde(default = "default_quic_enabled")]
    pub enabled: bool,

    /// Address to bind the WebTransport server
    #[serde(default = "default_quic_bind_addr")]
    pub bind_addr: String,

    /// Server name for TLS certificate (used in self-signed cert generation)
    #[serde(default = "default_quic_server_name")]
    pub server_name: String,

    /// Path to TLS certificate (PEM). Empty = generate self-signed.
    #[serde(default)]
    pub cert_path: String,

    /// Path to TLS private key (PEM). Empty = generate self-signed.
    #[serde(default)]
    pub key_path: String,

    /// #410/#282: bind an iroh substrate (ALPNs `hyprstream-rpc/1` + `moql`)
    /// as the PRIMARY production transport, in parallel with the quinn/WebTransport
    /// endpoint (kept for back-compat). Iroh is ON by default — it provides
    /// node_id-addressed (pkarr/N0-DNS-discoverable) federation reach, NAT
    /// traversal, self-certifying Ed25519 identity, and PQ-hybrid key exchange.
    /// Opt out with `[quic] iroh = false` to run quinn-only (legacy). Native-only.
    #[serde(default = "default_iroh_enabled")]
    pub iroh: bool,

    /// #358: the producer-chosen moq RELAY this node rendezvouses through, as a
    /// dialable URI (`https://host:port` for the relay's WebTransport `/moq`
    /// endpoint, or an iroh node URI). Empty = direct-only (the baseline). When
    /// set, every QUIC-enabled service advertises a `Role::Relay` reach and links
    /// its streaming origin UP to the relay, so neither publisher nor subscriber
    /// need be directly reachable by the other. Default deployments point this at
    /// the node's PDS / federation anchor (the `#atproto_pds` DID service entry).
    #[serde(default)]
    pub relay: String,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind_addr: default_quic_bind_addr(),
            server_name: default_quic_server_name(),
            cert_path: String::new(),
            key_path: String::new(),
            iroh: default_iroh_enabled(),
            relay: String::new(),
        }
    }
}

impl QuicConfig {
    /// Parse bind_addr into a SocketAddr.
    pub fn socket_addr(&self) -> anyhow::Result<std::net::SocketAddr> {
        self.bind_addr.parse().map_err(|e| anyhow::anyhow!("invalid quic.bind_addr '{}': {}", self.bind_addr, e))
    }

    /// Check if self-signed certificate should be generated.
    ///
    /// Warns when only one of cert_path/key_path is set (misconfiguration).
    pub fn use_self_signed(&self) -> bool {
        match (self.cert_path.is_empty(), self.key_path.is_empty()) {
            (true, true) => true,
            (false, false) => false,
            (false, true) => {
                tracing::warn!(
                    "quic.cert_path is set but quic.key_path is missing — generating self-signed cert"
                );
                true
            }
            (true, false) => {
                tracing::warn!(
                    "quic.key_path is set but quic.cert_path is missing — generating self-signed cert"
                );
                true
            }
        }
    }

    /// Generate or load TLS materials, returning `(cert_der, key_der)`.
    ///
    /// `key_der` is wrapped in `Zeroizing` to ensure it is zeroed on drop.
    ///
    /// For self-signed certs, loads persisted ECDSA P-256 materials from the secrets
    /// directory (generating on first run) with ≤14 day validity, as required by
    /// WebTransport `serverCertificateHashes` (W3C spec).
    ///
    /// QUIC uses a separate cert (`quic-cert`) from the HTTP cert (`tls-cert`) because
    /// WebTransport requires ≤14-day validity while HTTP allows 365 days.
    /// Load TLS materials for QUIC/WebTransport.
    ///
    /// Returns a certificate **chain** (leaf first, then intermediates/CA) and the
    /// private key. When loading from PEM files, all certificates in the file are
    /// included — this allows CA-signed certs (e.g. mkcert) to work by bundling
    /// the leaf + CA cert in a single PEM file.
    pub fn load_tls_materials(&self) -> anyhow::Result<(Vec<Vec<u8>>, Zeroizing<Vec<u8>>)> {
        if self.use_self_signed() {
            let secrets_dir = HyprConfig::resolve_secrets_dir();
            // Use quic-specific secret names so QUIC and HTTP certs have different
            // validity windows without stomping each other's files.
            let materials = crate::auth::identity_store::load_or_generate_tls_materials_named(
                &secrets_dir,
                &self.server_name,
                14,
                "quic-key",
                "quic-cert",
            )?;
            // Self-signed: chain is just the leaf cert
            Ok((vec![materials.cert_der], materials.key_der))
        } else {
            // Load from files
            let cert_pem = std::fs::read(&self.cert_path)
                .map_err(|e| anyhow::anyhow!("failed to read cert_path '{}': {}", self.cert_path, e))?;
            let key_pem = std::fs::read(&self.key_path)
                .map_err(|e| anyhow::anyhow!("failed to read key_path '{}': {}", self.key_path, e))?;

            // Parse ALL certs from PEM (leaf + intermediates + CA)
            let cert_chain: Vec<Vec<u8>> = rustls_pemfile::certs(&mut &cert_pem[..])
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("invalid certificate PEM: {}", e))?
                .into_iter()
                .map(|c| c.to_vec())
                .collect();
            if cert_chain.is_empty() {
                return Err(anyhow::anyhow!("no certificate found in {}", self.cert_path));
            }

            let key_der = Zeroizing::new(
                rustls_pemfile::private_key(&mut &key_pem[..])
                    .map_err(|e| anyhow::anyhow!("invalid key PEM: {}", e))?
                    .ok_or_else(|| anyhow::anyhow!("no private key found in {}", self.key_path))?
                    .secret_der()
                    .to_vec(),
            );

            Ok((cert_chain, key_der))
        }
    }

    /// Build a `QuicLoopConfig` for use with `RequestLoop`.
    ///
    /// Optionally embeds RFC 9728 Protected Resource Metadata so HTTP/3 clients
    /// can discover the OAuth authorization server for this QUIC endpoint.
    pub fn to_loop_config(
        &self,
        service_name: &str,
        oauth_issuer_url: Option<&str>,
    ) -> anyhow::Result<hyprstream_rpc::service::QuicLoopConfig> {
        let addr = self.socket_addr()?;
        let (cert_chain, key_der) = self.load_tls_materials()?;
        let meta_json = oauth_issuer_url.map(|issuer| {
            let meta = crate::services::oauth::protected_resource_metadata(
                &format!("https://{}/{}", self.server_name, service_name),
                issuer,
            );
            serde_json::to_vec(&meta).unwrap_or_default()
        });
        Ok(hyprstream_rpc::service::QuicLoopConfig {
            cert_chain,
            key_der,
            bind_addr: addr,
            server_name: self.server_name.clone(),
            protected_resource_json: meta_json,
            on_quic_bound: None,
            // #410: iroh is the primary production transport (on by default).
            // This minimal builder mirrors the daemon bootstrap default; the
            // full `QuicSharedConfig` path in `main.rs` honours `[quic] iroh`.
            iroh_enabled: default_iroh_enabled(),
            iroh_admission: None,
            on_iroh_bound: None,
            // #358: relay rendezvous is provisioned by the daemon bootstrap
            // (`QuicSharedConfig`), not this minimal builder. Direct-only here.
            moq_relay: None,
        })
    }
}

fn default_quic_enabled() -> bool { true }
/// #410: iroh substrate is the PRIMARY production transport — on by default.
/// The quinn-only baseline is the legacy path; opt out with `[quic] iroh = false`.
fn default_iroh_enabled() -> bool { true }
fn default_quic_bind_addr() -> String { "0.0.0.0:4433".to_owned() }
fn default_quic_server_name() -> String { "localhost".to_owned() }

/// JWT token issuance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    #[serde(default = "default_token_ttl")]
    pub default_ttl_seconds: u32,

    #[serde(default = "default_max_token_ttl")]
    pub max_ttl_seconds: u32,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            default_ttl_seconds: 172_800, // 48 hours
            max_ttl_seconds: 172_800,    // 48 hours
        }
    }
}

fn default_token_ttl() -> u32 { 172_800 }    // 48 hours
fn default_max_token_ttl() -> u32 { 172_800 } // 48 hours

/// OpenAI-compatible HTTP API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAIConfig {
    /// Host address for HTTP server
    #[serde(default = "default_oai_host")]
    pub host: String,

    /// Port for HTTP server
    #[serde(default = "default_oai_port")]
    pub port: u16,

    /// External URL for this server (used in OAuth metadata and WWW-Authenticate headers).
    /// Auto-derived from host:port if not set.
    #[serde(default)]
    pub external_url: Option<String>,

    /// TLS certificate path (optional)
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path (optional)
    #[serde(default)]
    pub tls_key: Option<PathBuf>,

    /// Request timeout in seconds
    #[serde(default = "default_oai_timeout")]
    pub request_timeout_secs: u64,

    /// CORS configuration
    #[serde(default)]
    pub cors: server::CorsConfig,

    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

impl Default for OAIConfig {
    fn default() -> Self {
        Self {
            host: default_oai_host(),
            port: default_oai_port(),
            external_url: None,
            tls_cert: None,
            tls_key: None,
            request_timeout_secs: default_oai_timeout(),
            cors: server::CorsConfig::default(),
            quic_port: None,
        }
    }
}

impl OAIConfig {
    /// Get the resource URL, using external_url if set, otherwise deriving from host:port.
    /// Auto-derives `https://` when global TLS is enabled.
    pub fn resource_url(&self) -> String {
        if let Some(ref url) = self.external_url {
            url.clone()
        } else {
            let scheme = if HyprConfig::load().map(|c| c.tls.enabled).unwrap_or(false) {
                "https"
            } else {
                "http"
            };
            let host = if self.host == "0.0.0.0" { "localhost" } else { &self.host };
            format!("{scheme}://{host}:{}", self.port)
        }
    }
}

fn default_oai_host() -> String { "0.0.0.0".to_owned() }
fn default_oai_port() -> u16 { 6789 }
fn default_oai_timeout() -> u64 { 300 }

/// XetService configuration — dual-stack HuggingFace-XET CAS HTTP face.
///
/// When enabled, exposes the HF-XET CAS wire routes (`/get_xorb/{hash}/`,
/// `/v1/reconstructions/{hash}`, `/v1/chunks/{key}`, `/v1/xorbs/{key}`,
/// `/v1/shards`) so a standard xet-enabled git repo can point its CAS endpoint
/// at hyprstream as an alternative XET backend to HuggingFace. See epic #654.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XetConfig {
    /// Whether the XetService is served. Disabled by default: this is a
    /// foundation surface (most routes 501 pending interop verification).
    #[serde(default)]
    pub enabled: bool,

    /// Host address for the HTTP server.
    #[serde(default = "default_xet_host")]
    pub host: String,

    /// Port for the HTTP server. Default 6792 (oai=6789, mcp=6790, oauth=6791).
    #[serde(default = "default_xet_port")]
    pub port: u16,

    /// External URL for this server (JWT audience / metadata). Auto-derived from
    /// host:port when unset, mirroring `OAIConfig::resource_url`.
    #[serde(default)]
    pub external_url: Option<String>,

    /// TLS certificate path (optional; falls back to the global `[tls]` config).
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path (optional; falls back to the global `[tls]` config).
    #[serde(default)]
    pub tls_key: Option<PathBuf>,
}

impl Default for XetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: default_xet_host(),
            port: default_xet_port(),
            external_url: None,
            tls_cert: None,
            tls_key: None,
        }
    }
}

impl XetConfig {
    /// Resource URL used as the JWT audience for this server. Uses `external_url`
    /// when set, else derives `scheme://host:port` (https when global TLS is on),
    /// mirroring `OAIConfig::resource_url`.
    pub fn resource_url(&self) -> String {
        if let Some(ref url) = self.external_url {
            url.clone()
        } else {
            let scheme = if HyprConfig::load().map(|c| c.tls.enabled).unwrap_or(false) {
                "https"
            } else {
                "http"
            };
            let host = if self.host == "0.0.0.0" { "localhost" } else { &self.host };
            format!("{scheme}://{host}:{}", self.port)
        }
    }
}

fn default_xet_host() -> String { "0.0.0.0".to_owned() }
fn default_xet_port() -> u16 { 6792 }

/// Arrow Flight SQL server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightConfig {
    /// Host address for Flight SQL server
    #[serde(default = "default_flight_host")]
    pub host: String,

    /// Port for Flight SQL server
    #[serde(default = "default_flight_port")]
    pub port: u16,

    /// Default dataset to serve (optional)
    #[serde(default)]
    pub default_dataset: Option<String>,

    /// TLS certificate path (optional)
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path (optional)
    #[serde(default)]
    pub tls_key: Option<PathBuf>,

    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

impl Default for FlightConfig {
    fn default() -> Self {
        Self {
            host: default_flight_host(),
            port: default_flight_port(),
            default_dataset: None,
            tls_cert: None,
            tls_key: None,
            quic_port: None,
        }
    }
}

fn default_flight_host() -> String { "0.0.0.0".to_owned() }
fn default_flight_port() -> u16 { 50051 }

/// MCP service configuration (Model Context Protocol)
///
/// This service provides an MCP-compliant interface for AI coding assistants
/// (Claude Code, Cursor, etc.) to interact with hyprstream via HTTP/SSE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    /// Host address for HTTP/SSE server
    #[serde(default = "default_mcp_host")]
    pub host: String,

    /// Port for HTTP/SSE server
    #[serde(default = "default_mcp_port")]
    pub http_port: u16,

    /// External URL for this server (used in OAuth metadata).
    /// Auto-derived from host:http_port if not set.
    #[serde(default)]
    pub external_url: Option<String>,

    /// TLS certificate path (optional, overrides global [tls])
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path (optional, overrides global [tls])
    #[serde(default)]
    pub tls_key: Option<PathBuf>,

    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,

    /// CORS configuration for the MCP HTTP/SSE server
    #[serde(default)]
    pub cors: server::CorsConfig,
}

impl Default for MCPConfig {
    fn default() -> Self {
        Self {
            host: default_mcp_host(),
            http_port: default_mcp_port(),
            external_url: None,
            tls_cert: None,
            tls_key: None,
            quic_port: None,
            cors: server::CorsConfig::default(),
        }
    }
}

impl MCPConfig {
    /// Get the resource URL, using external_url if set, otherwise deriving from host:http_port.
    /// Auto-derives `https://` when global TLS is enabled.
    pub fn resource_url(&self) -> String {
        if let Some(ref url) = self.external_url {
            url.clone()
        } else {
            let scheme = if HyprConfig::load().map(|c| c.tls.enabled).unwrap_or(false) {
                "https"
            } else {
                "http"
            };
            let host = if self.host == "0.0.0.0" { "localhost" } else { &self.host };
            format!("{scheme}://{host}:{}", self.http_port)
        }
    }
}

fn default_mcp_host() -> String { "0.0.0.0".to_owned() }
fn default_mcp_port() -> u16 { 6790 }

/// Configuration for a trusted external OIDC issuer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrustedIssuerConfig {
    /// Override the JWKS URI directly (skips AS metadata discovery).
    /// If absent, JWKS URI is auto-discovered from `{issuer}/.well-known/oauth-authorization-server`.
    #[serde(default)]
    pub jwks_uri: Option<String>,
    /// How long to cache the JWKS before re-fetching (default: 300 seconds).
    #[serde(default = "default_jwks_cache_ttl")]
    pub jwks_cache_ttl_secs: u64,
    /// Allow plain HTTP for JWKS fetches (default: false).
    ///
    /// **SECURITY WARNING:** Enabling this allows MITM attacks on the JWKS endpoint.
    /// Only use for internal networks or local development. Never enable in production.
    #[serde(default)]
    pub allow_http: bool,
}

fn default_jwks_cache_ttl() -> u64 { 300 }

/// Configuration for a trusted mesh peer's post-quantum signing identity (#157).
///
/// Admin-anchored entry binding a peer's Ed25519 mesh **signer** identity (the
/// envelope/COSE signer key, used as the kid anchor) to its trusted ML-DSA-65
/// mesh verifying key. These entries populate the process-global
/// `KeyedPqTrustStore` eagerly at startup; the store is immutable thereafter.
///
/// Both keys are supplied **inline**, out-of-band, as `Multikey`
/// `publicKeyMultibase` strings (base58btc, multicodec-prefixed) — the same
/// encoding the node publishes in its DID document (`#mesh` ed25519-pub `0xed01`
/// and `#mesh-pq` ml-dsa-65-pub `0x1211`). An operator copies a peer's
/// `#mesh` and `#mesh-pq` `publicKeyMultibase` values from that peer's DID doc.
///
/// This matches the `KeyedPqTrustStore` contract ("Entries MUST be established
/// out-of-band") and the Tiles interop admission model: only keys an operator
/// configured are trusted. If `mesh_peers` is empty, the store is empty and
/// behavior is unchanged (Hybrid fails closed for unknown peers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshPeerConfig {
    /// Peer's Ed25519 mesh signer public key as a `Multikey` `publicKeyMultibase`
    /// string (base58btc `z…`, multicodec `ed25519-pub` `0xed01`). This is the
    /// kid anchor the COSE composite is verified against.
    pub ed25519_multibase: String,
    /// Peer's ML-DSA-65 mesh verifying key as a `Multikey` `publicKeyMultibase`
    /// string (base58btc `z…`, multicodec `ml-dsa-65-pub` `0x1211`). The trusted
    /// post-quantum key bound to `ed25519_multibase`.
    pub mldsa65_multibase: String,
}

/// Protocol kind for an external OAuth/OIDC provider.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// Full OpenID Connect — discovery document + id_token JWT verification. Default.
    #[default]
    Oidc,
    /// Generic OAuth 2.0 with a userinfo endpoint. No discovery, no id_token.
    /// Requires `authorization_endpoint`, `token_endpoint_url`, and `userinfo_endpoint`.
    OAuth2,
}

/// Claim field name overrides for mapping a userinfo JSON response to standard claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimMapping {
    /// Field name for the stable subject identifier. Default: `"sub"`.
    /// GitHub uses `"id"` (numeric integer, coerced to string).
    /// Discord uses `"id"` (string snowflake).
    #[serde(default = "default_claim_sub")]
    pub sub: String,
    /// Field name for display name. `None` omits the name from the synthetic claims.
    #[serde(default = "default_claim_name")]
    pub name: Option<String>,
    /// Field name for email address. `None` omits email.
    #[serde(default = "default_claim_email")]
    pub email: Option<String>,
    /// Field name for the email-verified boolean.
    /// `None`, or a field that is absent/null in the response, is treated as `false`.
    #[serde(default = "default_claim_email_verified")]
    pub email_verified: Option<String>,
}

impl Default for ClaimMapping {
    fn default() -> Self {
        Self {
            sub: default_claim_sub(),
            name: default_claim_name(),
            email: default_claim_email(),
            email_verified: default_claim_email_verified(),
        }
    }
}

fn default_claim_sub() -> String { "sub".into() }
fn default_claim_name() -> Option<String> { Some("name".into()) }
fn default_claim_email() -> Option<String> { Some("email".into()) }
fn default_claim_email_verified() -> Option<String> { Some("email_verified".into()) }

/// Configuration for an external OIDC provider (login delegation).
///
/// Hyprstream acts as an OIDC Relying Party to this provider. Users authenticate
/// with the provider; hyprstream validates the external id_token and issues its
/// own JWT with scopes from the policy engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcProviderConfig {
    /// Provider protocol kind. Default: `"oidc"`.
    #[serde(default)]
    pub kind: ProviderKind,
    /// OIDC issuer URL. Required for `kind = "oidc"`.
    /// Must support `/.well-known/openid-configuration`.
    #[serde(default)]
    pub issuer_url: Option<String>,
    /// Authorization endpoint URL. Required for `kind = "oauth2"`.
    #[serde(default)]
    pub authorization_endpoint: Option<String>,
    /// Token endpoint URL. Required for `kind = "oauth2"`. Named `token_endpoint_url`
    /// to avoid collision with method names on OIDC metadata types.
    #[serde(default)]
    pub token_endpoint_url: Option<String>,
    /// Userinfo endpoint URL. Required for `kind = "oauth2"`.
    #[serde(default)]
    pub userinfo_endpoint: Option<String>,
    /// Whether the provider supports PKCE (`code_challenge`). Default: `true`.
    /// Set to `false` for providers that reject the parameter (e.g. GitHub).
    #[serde(default = "default_pkce_supported")]
    pub pkce_supported: bool,
    /// Claim field name overrides for mapping userinfo JSON to synthetic claims.
    #[serde(default)]
    pub claim_mapping: ClaimMapping,
    /// Client ID registered with the external provider.
    pub client_id: String,
    /// Client secret (optional — omit for public client with PKCE).
    #[serde(default)]
    pub client_secret: Option<String>,
    /// Scopes to request from the provider.
    #[serde(default)]
    pub scopes: Vec<String>,
    /// Display name for the login UI (e.g., "Sign in with GitHub").
    #[serde(default)]
    pub display_name: Option<String>,
    /// Allow plain HTTP for discovery (dev only).
    #[serde(default)]
    pub allow_http: bool,
    /// User identity mapping strategy.
    #[serde(default)]
    pub user_mapping: UserMappingStrategy,
    /// User provisioning mode.
    #[serde(default)]
    pub provisioning: ProvisioningMode,
    /// Allowed email domains (when provisioning = allowlist).
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    /// Default hyprstream scopes for auto-provisioned users.
    #[serde(default)]
    pub default_scopes: Vec<String>,
    /// Clock skew tolerance for JWT validation (seconds). OIDC only.
    #[serde(default = "default_clock_skew")]
    pub clock_skew_seconds: u64,
}

fn default_pkce_supported() -> bool { true }

impl OidcProviderConfig {
    pub fn effective_authorization_endpoint(&self) -> Option<&str> {
        self.authorization_endpoint.as_deref()
    }

    pub fn effective_token_endpoint_url(&self) -> Option<&str> {
        self.token_endpoint_url.as_deref()
    }

    pub fn effective_userinfo_endpoint(&self) -> Option<&str> {
        self.userinfo_endpoint.as_deref()
    }

    pub fn effective_pkce_supported(&self) -> bool {
        self.pkce_supported
    }

    pub fn effective_claim_mapping(&self) -> ClaimMapping {
        self.claim_mapping.clone()
    }

    pub fn effective_scopes(&self) -> Vec<String> {
        if !self.scopes.is_empty() {
            return self.scopes.clone();
        }
        match self.kind {
            ProviderKind::Oidc => default_oidc_scopes(),
            ProviderKind::OAuth2 => vec![],
        }
    }
}

fn default_oidc_scopes() -> Vec<String> {
    vec!["openid".into(), "profile".into(), "email".into()]
}
fn default_clock_skew() -> u64 { 60 }

/// How to map an external OIDC identity to a local hyprstream subject.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum UserMappingStrategy {
    /// `provider_slug:external_sub` (e.g., `keycloak:12345`). Default.
    #[default]
    Namespaced,
    /// Use the `email` claim (requires `email_verified=true`).
    Email,
    /// Use a specific claim value.
    Claim {
        /// The claim name to use as the local subject.
        name: String,
    },
    /// `did:web:{issuer_authority}:users:{external_sub}` per Phase 0.5
    /// architecture-doc Subject Identity Format.
    ///
    /// Uses the OAuth issuer URL's authority (host[:port]) as the DID
    /// method-specific identifier and the external `sub` claim as the
    /// user path component. Example: `did:web:hyprstream.example.com:users:12345`.
    ///
    /// **Opt-in**: existing deployments continue to use [`Namespaced`] by
    /// default so Casbin policies don't break. Operators migrating to
    /// did:web should land Phase 0e (Casbin policy migration) before
    /// switching this strategy.
    DidWeb,
}

/// Whether to auto-create users on first external login.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProvisioningMode {
    /// Reject unknown users (admin must pre-create).
    #[default]
    Deny,
    /// Auto-provision on first login.
    Auto,
    /// Auto-provision only for matching `allowed_domains`.
    Allowlist,
}

/// OAuth 2.1 authorization server configuration
///
/// Provides OAuth 2.1 (draft-ietf-oauth-v2-1-13) authorization for MCP and OAI services.
/// Supports RFC 7591 (Dynamic Client Registration), RFC 8414 (AS Metadata),
/// RFC 8707 (Resource Indicators), and Client ID Metadata Documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConfig {
    /// Host address for OAuth server
    #[serde(default = "default_oauth_host")]
    pub host: String,

    /// Port for OAuth server
    #[serde(default = "default_oauth_port")]
    pub port: u16,

    /// External URL for this server (used in metadata responses).
    /// Auto-derived from host:port if not set.
    #[serde(default)]
    pub external_url: Option<String>,

    /// Default scopes granted to new clients
    #[serde(default = "default_oauth_scopes")]
    pub default_scopes: Vec<String>,

    /// Access token TTL in seconds
    #[serde(default = "default_oauth_token_ttl")]
    pub token_ttl_seconds: u32,

    /// Refresh token TTL in seconds (default: 72 hours)
    #[serde(default = "default_refresh_token_ttl")]
    pub refresh_token_ttl_seconds: u32,

    /// TLS certificate path (optional, overrides global [tls])
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path (optional, overrides global [tls])
    #[serde(default)]
    pub tls_key: Option<PathBuf>,

    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,

    /// CORS configuration for the OAuth HTTP server
    #[serde(default = "default_oauth_cors")]
    pub cors: server::CorsConfig,

    /// Trusted external OIDC issuers for federation.
    /// Key = issuer URL (must match JWT `iss` claim exactly).
    /// Value = configuration for fetching/caching that issuer's JWKS.
    #[serde(default)]
    pub trusted_issuers: std::collections::HashMap<String, TrustedIssuerConfig>,

    /// Trusted mesh peers' post-quantum signing identities (#157).
    /// Key = an operator-chosen peer label (informational only).
    /// Value = the peer's Ed25519 mesh signer key + ML-DSA-65 mesh verifying
    /// key, supplied inline as out-of-band `Multikey` strings. Populates the
    /// process-global `KeyedPqTrustStore` eagerly at startup (admin-anchored,
    /// immutable). Empty = empty store = unchanged behavior (Hybrid fails
    /// closed for unknown peers). Distinct from `trusted_issuers`.
    #[serde(default)]
    pub mesh_peers: std::collections::HashMap<String, MeshPeerConfig>,

    /// OpenID Federation 1.0 Trust Anchor URLs (optional).
    /// When set, included as `authority_hints` in the entity configuration JWT,
    /// making this node discoverable within the named federations.
    /// Example: `["https://federation.example.org"]`
    #[serde(default)]
    pub authority_hints: Vec<String>,

    /// External OIDC providers for login delegation (IdP-agnostic).
    ///
    /// Users authenticate with the external provider; hyprstream validates the
    /// external id_token and issues its own JWT with scopes from the policy engine.
    /// Separate from `trusted_issuers` which is for hyprstream federation (direct
    /// JWT acceptance between nodes).
    #[serde(default)]
    pub oidc_providers: std::collections::HashMap<String, OidcProviderConfig>,

    /// Hex-encoded Ed25519 private key bytes for the local user identity (bypasses OS keyring).
    ///
    /// **TEST USE ONLY.** Set via `HYPRSTREAM__OAUTH__USER_SIGNING_KEY` env var.
    /// Allows the `sign-challenge` CLI and wizard to use a pre-generated key in
    /// isolated test environments without a keyring daemon.
    /// Never set this in production config files.
    #[serde(default, skip_serializing)]
    pub user_signing_key: Option<String>,

    /// How long to cache a third-party OAuth client's JWKS fetched
    /// via `jwks_uri` (RFC 7591 §2.1). Applies to `private_key_jwt`
    /// client authentication at the token endpoint.
    ///
    /// Clients that rotate signing keys faster than this should keep
    /// the old key in their JWKS for at least one cache window so
    /// verification spans the rotation (standard JWKS rotation
    /// hygiene). Default: 3600 seconds (1 hour).
    ///
    /// Distinct from `trusted_issuers[*].jwks_cache_ttl_secs` (which
    /// configures hyprstream federation peers, not OAuth clients).
    #[serde(default = "default_client_jwks_uri_cache_ttl")]
    pub client_jwks_uri_cache_ttl_secs: u64,

    /// How many days a JWT signing key remains in the active (issuance) slot.
    #[serde(default = "default_jwt_key_active_days")]
    pub jwt_key_active_days: u32,

    /// How many days before active-key expiry a lead key is pre-generated.
    #[serde(default = "default_jwt_key_lead_days")]
    pub jwt_key_lead_days: u32,

    /// How many extra days after active-key expiry the drain key remains in JWKS for verification.
    #[serde(default = "default_jwt_key_drain_days")]
    pub jwt_key_drain_days: u32,

    /// Override active key lifetime in seconds (takes precedence over `jwt_key_active_days`).
    /// Intended for integration tests with short rotation cycles.
    #[serde(default)]
    pub jwt_key_active_secs: Option<u32>,

    /// Override lead key pre-generation window in seconds (takes precedence over `jwt_key_lead_days`).
    #[serde(default)]
    pub jwt_key_lead_secs: Option<u32>,

    /// Override drain key retention window in seconds (takes precedence over `jwt_key_drain_days`).
    #[serde(default)]
    pub jwt_key_drain_secs: Option<u32>,

    /// Override rotation check interval in seconds (default: 21600 = 6 hours).
    #[serde(default)]
    pub jwt_key_rotation_check_secs: Option<u64>,

    /// Enforce RFC 9126 Pushed Authorization Requests at `/oauth/authorize`.
    ///
    /// When `true`, the authorization endpoint rejects any request that does
    /// not arrive via a `request_uri` referencing a prior `/oauth/par` call.
    /// Advertised in server metadata as `require_pushed_authorization_requests`.
    /// Defaults to `false` for compatibility.
    #[serde(default)]
    pub require_pushed_authorization_requests: bool,
}

fn default_oauth_cors() -> server::CorsConfig {
    server::CorsConfig::public()
}

impl Default for OAuthConfig {
    fn default() -> Self {
        Self {
            host: default_oauth_host(),
            port: default_oauth_port(),
            external_url: None,
            default_scopes: default_oauth_scopes(),
            token_ttl_seconds: default_oauth_token_ttl(),
            refresh_token_ttl_seconds: default_refresh_token_ttl(),
            client_jwks_uri_cache_ttl_secs: default_client_jwks_uri_cache_ttl(),
            tls_cert: None,
            tls_key: None,
            quic_port: None,
            cors: default_oauth_cors(),
            trusted_issuers: std::collections::HashMap::new(),
            mesh_peers: std::collections::HashMap::new(),
            authority_hints: Vec::new(),
            oidc_providers: std::collections::HashMap::new(),
            user_signing_key: None,
            jwt_key_active_days: default_jwt_key_active_days(),
            jwt_key_lead_days: default_jwt_key_lead_days(),
            jwt_key_drain_days: default_jwt_key_drain_days(),
            jwt_key_active_secs: None,
            jwt_key_lead_secs: None,
            jwt_key_drain_secs: None,
            jwt_key_rotation_check_secs: None,
            require_pushed_authorization_requests: false,
        }
    }
}

impl OAuthConfig {
    /// Active key lifetime in seconds (`_secs` override wins over `_days * 86400`).
    pub fn active_secs(&self) -> i64 {
        self.jwt_key_active_secs.map_or_else(
            || i64::from(self.jwt_key_active_days) * 86400,
            i64::from,
        )
    }

    /// Lead pre-generation window in seconds.
    pub fn lead_secs(&self) -> i64 {
        self.jwt_key_lead_secs.map_or_else(
            || i64::from(self.jwt_key_lead_days) * 86400,
            i64::from,
        )
    }

    /// Drain retention window in seconds.
    pub fn drain_secs(&self) -> i64 {
        self.jwt_key_drain_secs.map_or_else(
            || i64::from(self.jwt_key_drain_days) * 86400,
            i64::from,
        )
    }

    /// Rotation check interval (default 6 hours).
    pub fn rotation_check_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.jwt_key_rotation_check_secs.unwrap_or(6 * 3600))
    }

    /// Get the issuer URL, using external_url if set, otherwise deriving from host:port.
    /// Auto-derives `https://` when global TLS is enabled.
    pub fn issuer_url(&self) -> String {
        if let Some(ref url) = self.external_url {
            url.clone()
        } else {
            let scheme = if HyprConfig::load().map(|c| c.tls.enabled).unwrap_or(false) {
                "https"
            } else {
                "http"
            };
            let host = if self.host == "0.0.0.0" { "localhost" } else { &self.host };
            format!("{scheme}://{host}:{}", self.port)
        }
    }
}

fn default_oauth_host() -> String { "0.0.0.0".to_owned() }
fn default_oauth_port() -> u16 { 6791 }

/// Which backend stores user credentials and refresh tokens.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CredentialsBackend {
    #[default]
    Rocksdb,
    Valkey,
}

/// Valkey connection settings (used when `backend = "valkey"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValkeyCredentialsConfig {
    #[serde(default = "default_valkey_url")]
    pub url: String,
}

fn default_valkey_url() -> String { "redis://127.0.0.1:6379".to_owned() }

impl Default for ValkeyCredentialsConfig {
    fn default() -> Self { Self { url: default_valkey_url() } }
}

/// Credentials storage configuration.
///
/// Selects the backend for user profiles, pubkeys, and refresh tokens.
///
/// # Example TOML
/// ```toml
/// [credentials]
/// backend = "valkey"
/// [credentials.valkey]
/// url = "redis://127.0.0.1:6379"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CredentialsConfig {
    #[serde(default)]
    pub backend: CredentialsBackend,
    #[serde(default)]
    pub valkey: ValkeyCredentialsConfig,
}
fn default_oauth_scopes() -> Vec<String> {
    vec![
        "read:*:*".to_owned(),
        "infer:model:*".to_owned(),
        "write:*:*".to_owned(),
    ]
}
fn default_oauth_token_ttl() -> u32 { 3600 }
fn default_refresh_token_ttl() -> u32 { 2_628_000 } // 730 hours (~30 days)
fn default_client_jwks_uri_cache_ttl() -> u64 { 3600 } // 1 hour
fn default_jwt_key_active_days() -> u32 { 14 }
fn default_jwt_key_lead_days() -> u32 { 7 }
fn default_jwt_key_drain_days() -> u32 { 30 }

/// StreamService configuration
///
/// RPC transport server tunables. Mirrors `hyprstream_rpc::transport::rpc_session::RpcConfig`
/// so operators can tune these via the config file or `HYPRSTREAM__RPC__*` env vars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcServerConfig {
    /// Max concurrent in-flight bidi streams (server-wide semaphore).
    #[serde(default = "default_rpc_stream_limit")]
    pub stream_limit: usize,
    /// Max concurrent accepted connections per server.
    #[serde(default = "default_rpc_connection_limit")]
    pub connection_limit: usize,
    /// Max wall-clock seconds to read a single request frame.
    #[serde(default = "default_rpc_request_read_timeout_secs")]
    pub request_read_timeout_secs: u64,
    /// Max seconds for a peer's QUIC/WebTransport handshake to complete.
    #[serde(default = "default_rpc_handshake_timeout_secs")]
    pub handshake_timeout_secs: u64,
    /// Grace period (seconds) after writing a response for the peer to ack FIN.
    #[serde(default = "default_rpc_stopped_grace_secs")]
    pub stopped_grace_secs: u64,
    /// Max seconds for graceful drain on shutdown.
    #[serde(default = "default_rpc_drain_timeout_secs")]
    pub drain_timeout_secs: u64,
}

impl Default for RpcServerConfig {
    fn default() -> Self {
        use hyprstream_rpc::transport::rpc_session as rpc;
        Self {
            stream_limit: rpc::DEFAULT_STREAM_LIMIT,
            connection_limit: rpc::DEFAULT_CONNECTION_LIMIT,
            request_read_timeout_secs: rpc::REQUEST_READ_TIMEOUT.as_secs(),
            handshake_timeout_secs: rpc::HANDSHAKE_TIMEOUT.as_secs(),
            stopped_grace_secs: rpc::STOPPED_GRACE.as_secs(),
            drain_timeout_secs: rpc::DRAIN_TIMEOUT.as_secs(),
        }
    }
}

fn default_rpc_stream_limit() -> usize { hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT }
fn default_rpc_connection_limit() -> usize { hyprstream_rpc::transport::rpc_session::DEFAULT_CONNECTION_LIMIT }
fn default_rpc_request_read_timeout_secs() -> u64 { hyprstream_rpc::transport::rpc_session::REQUEST_READ_TIMEOUT.as_secs() }
fn default_rpc_handshake_timeout_secs() -> u64 { hyprstream_rpc::transport::rpc_session::HANDSHAKE_TIMEOUT.as_secs() }
fn default_rpc_stopped_grace_secs() -> u64 { hyprstream_rpc::transport::rpc_session::STOPPED_GRACE.as_secs() }
fn default_rpc_drain_timeout_secs() -> u64 { hyprstream_rpc::transport::rpc_session::DRAIN_TIMEOUT.as_secs() }

impl RpcServerConfig {
    /// Convert to the `hyprstream_rpc` wire type consumed by server builders.
    // TODO: wire into serve_bridged / QuinnRpcServer init paths (tracking issue #197)
    pub fn to_rpc_config(&self) -> hyprstream_rpc::transport::rpc_session::RpcConfig {
        use std::time::Duration;
        hyprstream_rpc::transport::rpc_session::RpcConfig {
            stream_limit: self.stream_limit,
            connection_limit: self.connection_limit,
            request_read_timeout: Duration::from_secs(self.request_read_timeout_secs),
            handshake_timeout: Duration::from_secs(self.handshake_timeout_secs),
            stopped_grace: Duration::from_secs(self.stopped_grace_secs),
            drain_timeout: Duration::from_secs(self.drain_timeout_secs),
        }
    }
}

/// moq streaming plane configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Deprecated ZMQ-era field — no longer used. Retained for config compat.
    #[serde(default = "default_max_pending_per_topic", skip_serializing)]
    pub max_pending_per_topic: usize,

    /// Deprecated ZMQ-era field — no longer used. Retained for config compat.
    #[serde(default = "default_message_ttl_secs", skip_serializing)]
    pub message_ttl_secs: u64,

    /// Deprecated ZMQ-era field — no longer used. Retained for config compat.
    #[serde(default = "default_compact_interval_secs", skip_serializing)]
    pub compact_interval_secs: u64,

    /// StreamBlock batching configuration (rate control)
    ///
    /// Controls adaptive batching based on throughput rate.
    /// Higher rates → larger batches (reduced overhead).
    /// Lower rates → smaller batches (reduced latency).
    #[serde(flatten, default)]
    pub batching: hyprstream_rpc::streaming::BatchingConfig,

    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,

    /// Timeout (seconds) waiting for the moq origin to announce a broadcast.
    /// Default: 10
    // TODO: wire into moq_stream.rs BROADCAST_ANNOUNCE_TIMEOUT constant (tracking issue #NNN)
    #[serde(default = "default_broadcast_announce_timeout_secs")]
    pub broadcast_announce_timeout_secs: u64,

    /// Timeout (seconds) between consecutive moq Groups on a subscribed track.
    /// A subscriber that sees no new Group for this long treats the publisher as gone.
    /// Default: 30
    // TODO: wire into moq_stream.rs GROUP_IDLE_TIMEOUT constant (tracking issue #NNN)
    #[serde(default = "default_group_idle_timeout_secs")]
    pub group_idle_timeout_secs: u64,

    /// Timeout (seconds) reading a single Frame from an already-opened moq Group.
    /// Default: 5
    // TODO: wire into moq_stream.rs FRAME_READ_TIMEOUT constant (tracking issue #NNN)
    #[serde(default = "default_frame_read_timeout_secs")]
    pub frame_read_timeout_secs: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_pending_per_topic: default_max_pending_per_topic(),
            message_ttl_secs: default_message_ttl_secs(),
            compact_interval_secs: default_compact_interval_secs(),
            batching: hyprstream_rpc::streaming::BatchingConfig::default(),
            quic_port: None,
            broadcast_announce_timeout_secs: default_broadcast_announce_timeout_secs(),
            group_idle_timeout_secs: default_group_idle_timeout_secs(),
            frame_read_timeout_secs: default_frame_read_timeout_secs(),
        }
    }
}

fn default_max_pending_per_topic() -> usize { 1000 }
fn default_message_ttl_secs() -> u64 { 30 }
fn default_compact_interval_secs() -> u64 { 5 }
fn default_broadcast_announce_timeout_secs() -> u64 {
    hyprstream_rpc::moq_stream::BROADCAST_ANNOUNCE_TIMEOUT.as_secs()
}
fn default_group_idle_timeout_secs() -> u64 {
    hyprstream_rpc::moq_stream::GROUP_IDLE_TIMEOUT.as_secs()
}
fn default_frame_read_timeout_secs() -> u64 {
    hyprstream_rpc::moq_stream::FRAME_READ_TIMEOUT.as_secs()
}

/// Storage paths and directories configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Models directory path
    pub models_dir: PathBuf,
    /// LoRAs directory path
    pub loras_dir: PathBuf,
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Config directory path
    pub config_dir: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        // Try to get XDG-compliant paths, fall back to current directory
        let (models_dir, loras_dir, cache_dir, config_dir) = match StoragePaths::new() {
            Ok(storage_paths) => (
                storage_paths.models_dir().unwrap_or_else(|_| PathBuf::from("./models")),
                storage_paths.loras_dir().unwrap_or_else(|_| PathBuf::from("./loras")),
                storage_paths.cache_dir().unwrap_or_else(|_| PathBuf::from("./cache")),
                storage_paths.config_dir().unwrap_or_else(|_| PathBuf::from("./config")),
            ),
            Err(e) => {
                tracing::warn!("XDG paths unavailable: {}, using local directories", e);
                (
                    PathBuf::from("./models"),
                    PathBuf::from("./loras"),
                    PathBuf::from("./cache"),
                    PathBuf::from("./config"),
                )
            }
        };

        Self {
            models_dir,
            loras_dir,
            cache_dir,
            config_dir,
        }
    }
}

/// Event proxy service configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventServiceConfig {
    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

/// Registry service configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryServiceConfig {
    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

/// Policy service configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyServiceConfig {
    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

/// Discovery service configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiscoveryServiceConfig {
    /// QUIC/WebTransport port. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

/// TUI display server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiServiceConfig {
    /// QUIC/WebTransport port for TUI viewers. None = no QUIC.
    #[serde(default)]
    pub quic_port: Option<u16>,
    /// Maximum concurrent sessions.
    #[serde(default = "default_tui_max_sessions")]
    pub max_sessions: u32,
    /// Scrollback lines per pane.
    #[serde(default = "default_tui_scrollback")]
    pub scrollback_lines: usize,
    /// WebTransport certificate validity in days (max 14).
    #[serde(default = "default_tui_wt_cert_days")]
    pub wt_cert_validity_days: u32,
}

fn default_tui_max_sessions() -> u32 { 16 }
fn default_tui_scrollback() -> usize { 2000 }
fn default_tui_wt_cert_days() -> u32 { 14 }

impl Default for TuiServiceConfig {
    fn default() -> Self {
        Self {
            quic_port: None,
            max_sessions: default_tui_max_sessions(),
            scrollback_lines: default_tui_scrollback(),
            wt_cert_validity_days: default_tui_wt_cert_days(),
        }
    }
}

/// Metrics service configuration (DuckDB-backed time-series ingest + DataFusion query).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// DuckDB connection string. ":memory:" for in-process, or a file path.
    /// Use the same path as flight config to share data between services.
    #[serde(default = "default_metrics_db")]
    pub db_path: String,

    /// Background checkpoint interval in seconds. 0 = disabled.
    #[serde(default = "default_checkpoint_interval_secs")]
    pub checkpoint_interval_secs: u64,

    /// QUIC/WebTransport port. None = no QUIC.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

fn default_metrics_db() -> String {
    ":memory:".to_owned()
}

fn default_checkpoint_interval_secs() -> u64 {
    300
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            db_path: default_metrics_db(),
            checkpoint_interval_secs: default_checkpoint_interval_secs(),
            quic_port: None,
        }
    }
}

/// Service management configuration
///
/// Controls which services are started at startup in ipc-systemd mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicesConfig {
    /// Services to start automatically at startup (ipc-systemd mode)
    ///
    /// Default: ["registry", "policy", "worker", "event"]
    #[serde(default = "default_startup_services")]
    pub startup: Vec<String>,
}

impl Default for ServicesConfig {
    fn default() -> Self {
        Self {
            startup: default_startup_services(),
        }
    }
}

/// Default list of services to start at startup
fn default_startup_services() -> Vec<String> {
    vec![
        "event".to_owned(),     // Must start first (message bus)
        "registry".to_owned(),  // Model registry
        "policy".to_owned(),    // Authorization
        "streams".to_owned(),       // Streaming proxy with JWT validation
        "notification".to_owned(),  // Encrypted notification relay (uses streams)
        "worker".to_owned(),        // Container workloads
        "model".to_owned(),         // Model management (publishes to notification)
        "oauth".to_owned(),     // OAuth 2.1 authorization server
        "oai".to_owned(),       // OpenAI-compatible HTTP API
        "flight".to_owned(),    // Arrow Flight SQL server
        "discovery".to_owned(), // Endpoint discovery (RFC 9728 metadata)
        "mcp".to_owned(),       // Model Context Protocol service
        "tui".to_owned(),       // Terminal multiplexer display server
        "metrics".to_owned(),   // Metrics ingest and query (DuckDB/DataFusion)
    ]
}

/// Model loading and identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to model file
    pub path: PathBuf,
    /// Model identifier (e.g., "qwen2-1.5b")
    pub name: String,
    /// Architecture type ("llama", "qwen", etc.)
    pub architecture: String,
    /// Expected parameter count
    pub parameters: Option<u64>,
    /// QUIC/WebTransport port for model service. None = no QUIC, Some(0) = ephemeral, Some(N) = explicit.
    #[serde(default)]
    pub quic_port: Option<u16>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            name: String::new(),
            architecture: String::new(),
            parameters: None,
            quic_port: None,
        }
    }
}

/// Runtime execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Context window size
    pub context_length: usize,
    /// Maximum context length override for KV cache allocation.
    /// None = use model's max_position_embeddings (can be very large, e.g., 40K tokens)
    /// Some(n) = cap KV cache at n tokens (significantly reduces GPU memory)
    pub max_context: Option<u32>,
    /// KV cache quantization type (None, INT8, NF4, FP4).
    /// Reduces GPU memory by 50-75% at slight quality cost.
    #[serde(default)]
    pub kv_quant_type: crate::runtime::KVQuantType,
    /// Batch processing size
    pub batch_size: usize,
    /// CPU threads (None = auto-detect)
    pub cpu_threads: Option<usize>,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// GPU device ID (None = auto-detect, typically device 0).
    ///
    /// Legacy single-GPU selector. For multi-GPU, prefer [`Self::devices`];
    /// this field remains the back-compat fallback when `devices` is empty.
    pub gpu_device_id: Option<usize>,
    /// Explicit set of GPU device indices for multi-GPU (#313, epic #310).
    ///
    /// Empty = unset → fall back to the single [`Self::gpu_device_id`] /
    /// `HYPRSTREAM_GPU_DEVICE` (existing single-GPU behavior is unchanged).
    /// Parsed from `HYPRSTREAM_GPU_DEVICES` (comma-separated, e.g. `0,1`).
    /// Resolution + validation lives in [`Self::resolve_device_indices`] and is
    /// consumed by `runtime::DevicePool`.
    #[serde(default)]
    pub devices: Vec<usize>,
    /// Fail fast when a *requested* GPU is unavailable instead of silently
    /// downgrading to CPU (#315, epic #310).
    ///
    /// A process told to run on GPU 3 that silently lands on CPU tanks a pipeline
    /// split, so strictness is the safe default for the multi-GPU path. This only
    /// affects the case where a GPU was *explicitly* requested (`use_gpu` with an
    /// explicit `gpu_device_id`/`devices`); pure auto-detect (`use_gpu` with no
    /// device requested) still falls back to CPU so the legacy single-GPU
    /// "use a GPU if there is one" behavior is unchanged.
    /// Defaults to `true`; override with `HYPRSTREAM_STRICT_DEVICE=0`.
    #[serde(default = "default_strict_device")]
    pub strict_device: bool,
    /// GPU layers to offload (None = auto)
    pub gpu_layers: Option<usize>,
    /// Use memory mapping for model files
    pub mmap: bool,
    /// KV cache size in MB
    pub kv_cache_size_mb: usize,
    /// Precision mode (BF16/FP16/FP32/FP8)
    pub precision_mode: Option<String>,
    // NEW: Concurrency and timeout settings
    pub max_concurrent_loads: usize,
    pub max_concurrent_generations: usize,
    pub default_generation_timeout_ms: u64,
    pub default_model_load_timeout_ms: u64,

    /// Continuous / in-flight batching (#329, epic #310). **Default: off.**
    ///
    /// When enabled, the inference scheduler groups concurrent decode steps of
    /// same-tenant-delta sequences into a single batched forward (Llama only).
    /// When off, each stream runs the unchanged batch=1 decode path. Override
    /// with `HYPRSTREAM_CONTINUOUS_BATCH` (truthy = on). Off by default while the
    /// scheduler wiring lands incrementally — the batched kernel is correctness-
    /// gated by `batched_ragged_decode_matches_serial`.
    #[serde(default = "default_continuous_batching")]
    pub continuous_batching: bool,
    /// Max sequences fused into one batched decode step when
    /// [`Self::continuous_batching`] is on (spike default 16). Tunable via
    /// `HYPRSTREAM_CONTINUOUS_BATCH_MAX`.
    #[serde(default = "default_continuous_batch_max")]
    pub continuous_batch_max: usize,
}

/// Default for [`RuntimeConfig::continuous_batching`]: off unless
/// `HYPRSTREAM_CONTINUOUS_BATCH` is set truthy (#329). Off is the safe default —
/// the batch=1 path is the verified reference.
fn default_continuous_batching() -> bool {
    std::env::var("HYPRSTREAM_CONTINUOUS_BATCH")
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Default for [`RuntimeConfig::continuous_batch_max`] (#329): 16 (spike rec),
/// overridable via `HYPRSTREAM_CONTINUOUS_BATCH_MAX`.
fn default_continuous_batch_max() -> usize {
    std::env::var("HYPRSTREAM_CONTINUOUS_BATCH_MAX")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(16)
}

/// Default for [`RuntimeConfig::strict_device`]: strict (fail-fast) unless
/// `HYPRSTREAM_STRICT_DEVICE` is set to a falsy value. Strictness is the safe
/// default for the multi-GPU path (#315).
fn default_strict_device() -> bool {
    std::env::var("HYPRSTREAM_STRICT_DEVICE")
        .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        // Check environment variables for runtime configuration
        // Precedence: CLI args > env vars (read here) > hardcoded defaults.
        // Environment variables set initial defaults; CLI args may override them later.
        let gpu_device_id = std::env::var("HYPRSTREAM_GPU_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        // `devices` is populated leniently here (Default cannot return errors);
        // the authoritative strict parse + validation lives in
        // `RuntimeConfig::resolve_device_indices`, which re-reads the env var and
        // turns parse errors into hard errors.
        let devices = std::env::var("HYPRSTREAM_GPU_DEVICES")
            .ok()
            .and_then(|s| Self::parse_device_list(&s).ok())
            .unwrap_or_default();

        let max_context = std::env::var("HYPRSTREAM_MAX_CONTEXT")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());

        let kv_quant_type = std::env::var("HYPRSTREAM_KV_QUANT")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "int8" => Some(crate::runtime::KVQuantType::Int8),
                "nf4" => Some(crate::runtime::KVQuantType::Nf4),
                "fp4" => Some(crate::runtime::KVQuantType::Fp4),
                "none" | "" => Some(crate::runtime::KVQuantType::None),
                _ => None,
            })
            .unwrap_or(crate::runtime::KVQuantType::None);

        Self {
            context_length: 4096,
            max_context,
            kv_quant_type,
            batch_size: 512,
            cpu_threads: None,
            use_gpu: true,
            gpu_device_id, // From env or None (auto-detect device 0)
            devices,       // From HYPRSTREAM_GPU_DEVICES or empty (→ fall back to gpu_device_id)
            strict_device: default_strict_device(),
            gpu_layers: None,
            mmap: true,
            kv_cache_size_mb: 2048,
            precision_mode: Some("auto".to_owned()),
            max_concurrent_loads: 2,
            max_concurrent_generations: 10,
            default_generation_timeout_ms: 120000, // 2 minutes
            default_model_load_timeout_ms: 300000, // 5 minutes
            continuous_batching: default_continuous_batching(),
            continuous_batch_max: default_continuous_batch_max(),
        }
    }
}

impl RuntimeConfig {
    /// Parse a comma-separated GPU device list (e.g. `"0,1"`), strictly.
    ///
    /// Whitespace around entries is trimmed. Any non-numeric entry, or a
    /// trailing/empty field (e.g. `"0,"` or `"0,,1"`), is a hard error — there
    /// is no silent default. Returns the parsed indices (possibly with
    /// duplicates; dedup/validation is the caller's job).
    fn parse_device_list(raw: &str) -> anyhow::Result<Vec<usize>> {
        raw.split(',')
            .map(|part| {
                let trimmed = part.trim();
                trimmed.parse::<usize>().map_err(|e| {
                    anyhow::anyhow!(
                        "invalid GPU device index {trimmed:?} in HYPRSTREAM_GPU_DEVICES={raw:?}: {e}"
                    )
                })
            })
            .collect()
    }

    /// Resolve the explicitly-requested *multi-GPU* device set, fail-fast.
    ///
    /// This is the seam the multi-GPU foundation uses to decide whether to engage
    /// the new `DevicePool` path. It considers **only** the explicit multi-GPU
    /// inputs and deliberately excludes the legacy single [`Self::gpu_device_id`]
    /// so that existing single-GPU behavior is left entirely on its old code
    /// path (#313 introduces the pool without changing single-GPU runtime
    /// behavior). Precedence:
    ///
    /// 1. `HYPRSTREAM_GPU_DEVICES` env var (re-parsed here strictly, so a
    ///    malformed value is a hard error rather than a silent fallback).
    /// 2. The [`Self::devices`] field (e.g. from a config file / CLI).
    ///
    /// Returns `Ok(None)` when no explicit multi-GPU set was requested,
    /// `Ok(Some(indices))` (non-empty, duplicate-free) otherwise, and `Err` on
    /// parse errors or duplicate indices.
    pub fn resolve_explicit_multi_device_indices(&self) -> anyhow::Result<Option<Vec<usize>>> {
        // (1) env var wins and is parsed strictly here. An absent or
        //     empty/whitespace-only value is treated as "unset" so resolution
        //     falls through to (2) the struct field (config file / CLI).
        let env_raw = std::env::var("HYPRSTREAM_GPU_DEVICES").ok();
        let env_trimmed = env_raw.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let indices = match env_trimmed {
            Some(raw) => Self::parse_device_list(raw)?,
            None => self.devices.clone(),
        };

        Self::validate_index_set(indices)
    }

    /// Resolve the GPU device indices to use, fail-fast, including the legacy
    /// single-GPU fallback.
    ///
    /// This is the full-precedence resolver consumed by
    /// `runtime::DevicePool::from_config`. It is
    /// [`Self::resolve_explicit_multi_device_indices`] plus a final fallback to
    /// the legacy single [`Self::gpu_device_id`] (mapped to a one-element set),
    /// preserving back-compat for callers that build a pool directly from config.
    ///
    /// `Ok(None)` means nothing was requested (caller should auto-detect).
    pub fn resolve_device_indices(&self) -> anyhow::Result<Option<Vec<usize>>> {
        if let Some(indices) = self.resolve_explicit_multi_device_indices()? {
            return Ok(Some(indices));
        }
        // Legacy single-GPU selector as the final fallback.
        match self.gpu_device_id {
            Some(id) => Self::validate_index_set(vec![id]),
            None => Ok(None),
        }
    }

    /// Shared post-processing for a resolved index list: empty → `None`, reject
    /// duplicates, otherwise `Some(indices)`.
    fn validate_index_set(indices: Vec<usize>) -> anyhow::Result<Option<Vec<usize>>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let mut seen = std::collections::HashSet::with_capacity(indices.len());
        for &idx in &indices {
            if !seen.insert(idx) {
                return Err(anyhow::anyhow!(
                    "duplicate GPU device index {idx} in requested set {indices:?}"
                ));
            }
        }
        Ok(Some(indices))
    }
}

/// Text generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0-2.0)
    pub temperature: f32,
    /// Nucleus sampling threshold
    pub top_p: f32,
    /// Top-k sampling limit
    pub top_k: Option<usize>,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Stop sequences
    pub stop_tokens: Vec<String>,
    /// Random seed for reproducible generation
    pub seed: Option<u32>,
    /// Enable streaming output
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repeat_penalty: 1.1,
            stop_tokens: vec!["</s>".to_owned(), "<|endoftext|>".to_owned()],
            seed: None,
            stream: false,
        }
    }
}

/// Application-level LoRA settings (TOML config: enabled, max_adapters, etc.)
///
/// This is distinct from `TenantDeltaConfig` which configures LoRA weight parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAppConfig {
    /// Enable LoRA adapters
    pub enabled: bool,
    /// Maximum number of active adapters
    pub max_adapters: usize,
    /// LoRA scaling factor (alpha)
    pub alpha: f32,
    /// Target sparsity ratio (0.0-1.0)
    pub sparsity: f32,
}

impl Default for LoraAppConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_adapters: 4,
            alpha: 32.0,
            sparsity: 0.99,
        }
    }
}


/// Builder for Hyprstream configuration
pub struct HyprConfigBuilder {
    server_builder: ServerConfigBuilder,
    model: ModelConfig,
    runtime: RuntimeConfig,
    generation: GenerationConfig,
    lora: LoraAppConfig,
    storage: StorageConfig,
    git2db: git2db::config::Git2DBConfig,
    worker: Option<hyprstream_workers::config::WorkerConfig>,
    services: ServicesConfig,
    token: TokenConfig,
    oai: OAIConfig,
    flight: FlightConfig,
    mcp: MCPConfig,
    oauth: OAuthConfig,
    streaming: StreamingConfig,
    tls: TlsConfig,
    quic: QuicConfig,
    event: EventServiceConfig,
    registry: RegistryServiceConfig,
    policy: PolicyServiceConfig,
    discovery: DiscoveryServiceConfig,
    tui: TuiServiceConfig,
    metrics: MetricsConfig,
}

impl HyprConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            server_builder: ServerConfigBuilder::new(),
            model: ModelConfig::default(),
            runtime: RuntimeConfig::default(),
            generation: GenerationConfig::default(),
            lora: LoraAppConfig::default(),
            storage: StorageConfig::default(),
            git2db: git2db::config::Git2DBConfig::default(),
            worker: None,
            services: ServicesConfig::default(),
            token: TokenConfig::default(),
            oai: OAIConfig::default(),
            flight: FlightConfig::default(),
            mcp: MCPConfig::default(),
            oauth: OAuthConfig::default(),
            streaming: StreamingConfig::default(),
            tls: TlsConfig::default(),
            quic: QuicConfig::default(),
            event: EventServiceConfig::default(),
            registry: RegistryServiceConfig::default(),
            policy: PolicyServiceConfig::default(),
            discovery: DiscoveryServiceConfig::default(),
            tui: TuiServiceConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }

    /// Start from an existing config
    pub fn from_config(config: HyprConfig) -> Self {
        Self {
            server_builder: config.server.to_builder(),
            model: config.model,
            runtime: config.runtime,
            generation: config.generation,
            lora: config.lora,
            storage: config.storage,
            git2db: config.git2db,
            worker: config.worker,
            services: config.services,
            token: config.token,
            oai: config.oai,
            flight: config.flight,
            mcp: config.mcp,
            oauth: config.oauth,
            streaming: config.streaming,
            tls: config.tls,
            quic: config.quic,
            event: config.event,
            registry: config.registry,
            policy: config.policy,
            discovery: config.discovery,
            tui: config.tui,
            metrics: config.metrics,
        }
    }

    /// Access server builder for chaining
    pub fn server(mut self, f: impl FnOnce(ServerConfigBuilder) -> ServerConfigBuilder) -> Self {
        self.server_builder = f(self.server_builder);
        self
    }

    /// Load all configurations from environment variables
    pub fn from_env(mut self) -> Self {
        self.server_builder = self.server_builder.from_env();
        self
    }

    /// Build the final configuration
    pub fn build(self) -> HyprConfig {
        HyprConfig {
            server: self.server_builder.build(),
            model: self.model,
            runtime: self.runtime,
            generation: self.generation,
            lora: self.lora,
            storage: self.storage,
            git2db: self.git2db,
            worker: self.worker,
            services: self.services,
            token: self.token,
            oai: self.oai,
            xet: Default::default(),
            flight: self.flight,
            mcp: self.mcp,
            oauth: self.oauth,
            credentials: Default::default(),
            streaming: self.streaming,
            rpc: Default::default(),
            tls: self.tls,
            quic: self.quic,
            event: self.event,
            registry: self.registry,
            policy: self.policy,
            discovery: self.discovery,
            tui: self.tui,
            metrics: self.metrics,
            signing_key: None,
            secrets: Default::default(),
        }
    }

    /// Set the worker service configuration
    pub fn worker(mut self, config: hyprstream_workers::config::WorkerConfig) -> Self {
        self.worker = Some(config);
        self
    }
}

impl Default for HyprConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HyprConfig {
    /// Create a builder for the application configuration
    pub fn builder() -> HyprConfigBuilder {
        HyprConfigBuilder::new()
    }

    /// Load configuration using the config crate with XDG directories and environment variables
    pub fn load() -> Result<Self, ConfigError> {
        let storage = StoragePaths::new().map_err(|e| {
            ConfigError::Message(format!("Failed to initialize storage paths: {e}"))
        })?;

        let config_dir = storage
            .config_dir()
            .map_err(|e| ConfigError::Message(format!("Failed to get config directory: {e}")))?;

        let settings = Config::builder()
            // Load from default configuration structure
            .add_source(Config::try_from(&HyprConfig::default())?)
            // Load from config file if it exists
            .add_source(File::from(config_dir.join("config")).required(false))
            .add_source(File::from(config_dir.join("config.toml")).required(false))
            .add_source(File::from(config_dir.join("config.json")).required(false))
            .add_source(File::from(config_dir.join("config.yaml")).required(false))
            // Load from environment variables with HYPRSTREAM__ prefix (double underscore for nesting)
            .add_source(Environment::with_prefix("HYPRSTREAM").separator("__").try_parsing(true));

        // Build and deserialize configuration
        let mut hypr_config: HyprConfig = settings.build()?.try_deserialize()?;

        // Load git2db config from environment/file (it has its own env handling)
        // This ensures GIT2DB__* environment variables are properly loaded
        match git2db::config::Git2DBConfig::load() {
            Ok(git2db_config) => {
                tracing::info!(
                    "Loaded git2db config, token present: {}",
                    git2db_config.network.access_token.is_some()
                );
                hypr_config.git2db = git2db_config;
            }
            Err(e) => {
                tracing::warn!("Failed to load git2db config: {}, using default", e);
                hypr_config.git2db = git2db::config::Git2DBConfig::default();
            }
        }

        Ok(hypr_config)
    }

    /// Load configuration from file
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("toml");

        let config = match extension {
            "json" => serde_json::from_str(&contents)?,
            "yaml" | "yml" => serde_yaml::from_str(&contents)?,
            _ => toml::from_str(&contents)?,
        };

        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file(&self, path: &Path) -> anyhow::Result<()> {
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("toml");

        let contents = match extension {
            "json" => serde_json::to_string_pretty(self)?,
            "yaml" | "yml" => serde_yaml::to_string(self)?,
            _ => toml::to_string_pretty(self)?,
        };

        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validate the entire configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate model config
        if !self.model.path.as_os_str().is_empty() && !self.model.path.exists() {
            anyhow::bail!(
                "Configured model path does not exist: {}",
                self.model.path.display()
            );
        }

        Ok(())
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> &PathBuf {
        &self.storage.models_dir
    }

    /// Get the LoRAs directory path
    pub fn loras_dir(&self) -> &PathBuf {
        &self.storage.loras_dir
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &PathBuf {
        &self.storage.cache_dir
    }

    /// Get the config directory path
    pub fn config_dir(&self) -> &PathBuf {
        &self.storage.config_dir
    }

    /// Ensure all configured directories exist
    pub fn ensure_directories(&self) -> Result<(), std::io::Error> {
        std::fs::create_dir_all(&self.storage.models_dir)?;
        std::fs::create_dir_all(&self.storage.loras_dir)?;
        std::fs::create_dir_all(&self.storage.cache_dir)?;
        std::fs::create_dir_all(&self.storage.config_dir)?;
        Ok(())
    }

    /// Update model configuration after downloading
    pub fn set_model(&mut self, model_path: PathBuf, model_name: String, architecture: String) {
        self.model.path = model_path;
        self.model.name = model_name;
        self.model.architecture = architecture;
    }

    /// Create generation request from config + prompt
    /// Save configuration to default location
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let storage = StoragePaths::new()?;
        let config_dir = storage.config_dir()?;
        let config_path = config_dir.join("config.toml");

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;

        tracing::info!("✅ Configuration saved to: {}", config_path.display());
        Ok(())
    }

    // ── Secrets / key bypass helpers ────────────────────────────────────────

    /// Resolve the secrets directory from config, or the platform XDG default.
    ///
    /// Prefer calling this over inlining the fallback logic everywhere.
    pub fn resolve_secrets_dir() -> PathBuf {
        match Self::load() {
            Ok(cfg) => cfg.secrets.resolve_dir(cfg.config_dir()),
            Err(_) => SecretsConfig::default_dir(),
        }
    }

    /// Check for the `HYPRSTREAM__SIGNING_KEY` test bypass.
    ///
    /// Returns `Ok(Some(key))` when the bypass is set and valid,
    /// `Ok(None)` when not configured, `Err` when malformed.
    pub fn node_signing_key_bypass() -> anyhow::Result<Option<ed25519_dalek::SigningKey>> {
        if let Ok(cfg) = Self::load() {
            if let Some(ref hex_key) = cfg.signing_key {
                let mut bytes = hex::decode(hex_key)
                    .map_err(|e| anyhow::anyhow!("HYPRSTREAM__SIGNING_KEY: invalid hex: {e}"))?;
                let mut arr: [u8; 32] = bytes.as_slice()
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("HYPRSTREAM__SIGNING_KEY: expected 32 bytes"))?;
                let sk = ed25519_dalek::SigningKey::from_bytes(&arr);
                bytes.zeroize();
                arr.zeroize();
                tracing::info!("Using node signing key from config (test bypass)");
                return Ok(Some(sk));
            }
        }
        Ok(None)
    }

    /// Check for the `HYPRSTREAM__OAUTH__USER_SIGNING_KEY` test bypass.
    ///
    /// Returns `Ok(Some((sk, vk)))` when the bypass is set and valid,
    /// `Ok(None)` when not configured, `Err` when malformed.
    pub fn user_signing_key_bypass(
    ) -> anyhow::Result<Option<(ed25519_dalek::SigningKey, ed25519_dalek::VerifyingKey)>> {
        if let Ok(cfg) = Self::load() {
            if let Some(ref hex_key) = cfg.oauth.user_signing_key {
                let mut bytes = hex::decode(hex_key)
                    .map_err(|e| anyhow::anyhow!("HYPRSTREAM__OAUTH__USER_SIGNING_KEY: invalid hex: {e}"))?;
                let mut arr: [u8; 32] = bytes.as_slice()
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("HYPRSTREAM__OAUTH__USER_SIGNING_KEY: expected 32 bytes"))?;
                let sk = ed25519_dalek::SigningKey::from_bytes(&arr);
                bytes.zeroize();
                arr.zeroize();
                let vk = sk.verifying_key();
                tracing::info!("Using user signing key from config (test bypass)");
                return Ok(Some((sk, vk)));
            }
        }
        Ok(None)
    }

    /// Create a default configuration for a specific model path
    pub fn default_for_model(model_path: &Path) -> anyhow::Result<Self> {
        let storage_paths = StoragePaths::new()?;
        let mut config = Self::default();

        config.model.path = model_path.to_path_buf();
        config.model.name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown").to_owned();
        config.model.architecture = "auto".to_owned(); // Auto-detect from model

        // Update storage paths to use XDG directories
        config.storage = StorageConfig {
            models_dir: storage_paths.models_dir()?,
            loras_dir: storage_paths.loras_dir()?,
            cache_dir: storage_paths.cache_dir()?,
            config_dir: storage_paths.config_dir()?,
        };

        Ok(config)
    }
}

/// A prompt string that has been processed through the chat template engine.
/// This newtype prevents accidentally passing untemplated strings to generation.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(transparent)]
pub struct TemplatedPrompt(String);

impl TemplatedPrompt {
    /// Create from a templated string. Only call after template application.
    pub fn new(s: String) -> Self {
        Self(s)
    }

    /// Get the prompt as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the inner string.
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Get the length of the prompt in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the prompt is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for TemplatedPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unified sampling parameters with Option fields for clean precedence merging.
///
/// All fields are Option<T> to represent "not specified", enabling clear
/// precedence: Server defaults → Model defaults → User overrides
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub stop_tokens: Option<Vec<String>>,
    pub seed: Option<u64>,

    // Advanced parameters (HuggingFace transformers compatibility)
    #[serde(default)]
    pub length_penalty: Option<f32>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub epsilon_cutoff: Option<f32>,
    #[serde(default)]
    pub eta_cutoff: Option<f32>,
    #[serde(default)]
    pub do_sample: Option<bool>,

    // NEW: Async parameters
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

impl SamplingParams {
    /// Load model-specific config from a model directory
    pub async fn from_model_path(model_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let gen_config_path = model_path.join("generation_config.json");
        if gen_config_path.exists() {
            let content = tokio::fs::read_to_string(&gen_config_path).await?;
            let config: serde_json::Value = serde_json::from_str(&content)?;
            return Ok(Self::from_generation_config(&config));
        }

        let config_path = model_path.join("config.json");
        if config_path.exists() {
            let content = tokio::fs::read_to_string(&config_path).await?;
            let config: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(gen_config) = config.get("generation_config") {
                return Ok(Self::from_generation_config(gen_config));
            }
        }

        Ok(Self::default())
    }

    /// Parse HuggingFace generation_config.json format
    fn from_generation_config(config: &serde_json::Value) -> Self {
        Self {
            temperature: config.get("temperature").and_then(serde_json::Value::as_f64).map(|v| v as f32),
            top_k: config.get("top_k").and_then(serde_json::Value::as_u64).map(|v| v as usize),
            top_p: config.get("top_p").and_then(serde_json::Value::as_f64).map(|v| v as f32),
            repeat_penalty: config.get("repetition_penalty").and_then(serde_json::Value::as_f64).map(|v| v as f32),
            max_tokens: config.get("max_new_tokens").and_then(serde_json::Value::as_u64).map(|v| v as usize)
                .or_else(|| config.get("max_length").and_then(serde_json::Value::as_u64).map(|v| v as usize)),
            length_penalty: config.get("length_penalty").and_then(serde_json::Value::as_f64).map(|v| v as f32),
            typical_p: config.get("typical_p").and_then(serde_json::Value::as_f64).map(|v| v as f32),
            epsilon_cutoff: config.get("epsilon_cutoff").and_then(serde_json::Value::as_f64).map(|v| v as f32),
            eta_cutoff: config.get("eta_cutoff").and_then(serde_json::Value::as_f64).map(|v| v as f32),
            do_sample: config.get("do_sample").and_then(serde_json::Value::as_bool),
            stop_tokens: config.get("eos_token_id").and_then(|v| {
                if let Some(arr) = v.as_array() {
                    let tokens: Vec<String> = arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    if tokens.is_empty() { None } else { Some(tokens) }
                } else {
                    None
                }
            }),
            seed: config.get("seed").and_then(serde_json::Value::as_u64),
            repeat_last_n: None,
            timeout_ms: None,
        }
    }

    /// Merge with another config. The other config takes precedence for any Some values.
    /// This enables clear precedence: `base.merge(override)` where override wins.
    pub fn merge(self, other: Self) -> Self {
        Self {
            max_tokens: other.max_tokens.or(self.max_tokens),
            temperature: other.temperature.or(self.temperature),
            top_p: other.top_p.or(self.top_p),
            top_k: other.top_k.or(self.top_k),
            repeat_penalty: other.repeat_penalty.or(self.repeat_penalty),
            repeat_last_n: other.repeat_last_n.or(self.repeat_last_n),
            stop_tokens: other.stop_tokens.or(self.stop_tokens),
            seed: other.seed.or(self.seed),
            length_penalty: other.length_penalty.or(self.length_penalty),
            typical_p: other.typical_p.or(self.typical_p),
            epsilon_cutoff: other.epsilon_cutoff.or(self.epsilon_cutoff),
            eta_cutoff: other.eta_cutoff.or(self.eta_cutoff),
            do_sample: other.do_sample.or(self.do_sample),
            timeout_ms: other.timeout_ms.or(self.timeout_ms),
        }
    }

    /// Build a `GenerationRequest` from these sampling params and a prompt.
    ///
    /// Applies `SamplingParams` fields as `Option<T>` — `None` means "not specified",
    /// letting the engine use its defaults.
    pub fn into_generation_request(self, prompt: String) -> crate::services::generated::inference_client::GenerationRequest {
        crate::services::generated::inference_client::GenerationRequest {
            prompt,
            max_tokens: self.max_tokens.map(|v| v as u32),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k.map(|v| v as u32),
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n.map(|v| v as u32),
            stop_tokens: self.stop_tokens,
            seed: self.seed.map(|v| v as u32),
            timeout_ms: self.timeout_ms,
            ..Default::default()
        }
    }

    /// Resolve to concrete values with defaults
    pub fn resolve(self) -> ResolvedSamplingParams {
        ResolvedSamplingParams {
            max_tokens: self.max_tokens.unwrap_or(2048),
            temperature: self.temperature.unwrap_or(0.7),
            top_p: self.top_p.unwrap_or(0.95),
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty.unwrap_or(1.0),
            repeat_last_n: self.repeat_last_n.unwrap_or(64),
            stop_tokens: self.stop_tokens.unwrap_or_default(),
            seed: self.seed,
            length_penalty: self.length_penalty.unwrap_or(1.0),
            typical_p: self.typical_p,
            epsilon_cutoff: self.epsilon_cutoff,
            eta_cutoff: self.eta_cutoff,
            do_sample: self.do_sample.unwrap_or(true),
            timeout_ms: self.timeout_ms.unwrap_or(120000), // Use RuntimeConfig default (2 minutes)
        }
    }
}

/// Resolved sampling parameters with concrete values (no Options for required fields)
#[derive(Debug, Clone)]
pub struct ResolvedSamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub stop_tokens: Vec<String>,
    pub seed: Option<u64>,
    pub length_penalty: f32,
    pub typical_p: Option<f32>,
    pub epsilon_cutoff: Option<f32>,
    pub eta_cutoff: Option<f32>,
    pub do_sample: bool,
    // NEW: Async parameters
    pub timeout_ms: u64,
}

impl From<&crate::config::server::SamplingParamDefaults> for SamplingParams {
    fn from(defaults: &crate::config::server::SamplingParamDefaults) -> Self {
        Self {
            max_tokens: Some(defaults.max_tokens),
            temperature: Some(defaults.temperature),
            top_p: Some(defaults.top_p),
            repeat_penalty: Some(defaults.repeat_penalty),
            top_k: None,
            repeat_last_n: None,
            stop_tokens: None,
            seed: None,
            length_penalty: None,
            typical_p: None,
            epsilon_cutoff: None,
            eta_cutoff: None,
            do_sample: None,
            timeout_ms: None, // Don't set timeout here - let engine handle it
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::expect_used)]
    fn test_config_with_worker_section_at_defaults_serializes_to_toml() {
        // F3 (#761) regression: `AdmissionConfig`'s default per-Subject/per-group
        // quotas used to be `usize::MAX`, which is not a valid TOML i64, so
        // `Config::save()` (which does `toml::to_string_pretty(self)`) failed
        // for any config carrying a `[worker]` section. This reproduces that
        // exact save path and round-trips it.
        let mut config = HyprConfig::default();
        config.worker = Some(hyprstream_workers::config::WorkerConfig::default());

        let toml_str = toml::to_string_pretty(&config)
            .expect("a default worker section must serialize to TOML (F3)");
        let parsed: HyprConfig =
            toml::from_str(&toml_str).expect("round-trip parse must succeed");
        let admission = parsed
            .worker
            .expect("worker section round-trips")
            .pool
            .admission;
        assert_eq!(admission.max_per_subject, None);
        assert_eq!(admission.max_per_group, None);
    }

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();

        // Verify max_tokens is set to 2048 (not 100)
        assert_eq!(config.max_tokens, 2048, "Default max_tokens should be 2048 for thinking mode support");

        // Verify other reasonable defaults
        assert!(config.temperature > 0.0, "Temperature should be non-zero");
        assert!(config.top_p > 0.0 && config.top_p <= 1.0, "top_p should be in valid range");
    }

    #[test]
    fn test_sampling_params_into_generation_request() {
        let params = SamplingParams {
            temperature: Some(0.8),
            top_k: Some(30),
            max_tokens: Some(1000),
            ..Default::default()
        };

        let request = params.into_generation_request("test prompt".to_owned());

        assert_eq!(request.prompt, "test prompt");
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.top_k, Some(30));
        assert_eq!(request.max_tokens, Some(1000));
    }

    #[test]
    fn test_clean_config_precedence() {
        let server_defaults = SamplingParams {
            max_tokens: Some(1024),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(50),
            repeat_penalty: Some(1.1),
            ..Default::default()
        };

        let model_defaults = SamplingParams {
            temperature: Some(0.7),
            top_k: Some(40),
            typical_p: Some(0.95),
            ..Default::default()
        };

        let user_overrides = SamplingParams {
            temperature: Some(0.9),
            max_tokens: Some(512),
            ..Default::default()
        };

        let final_config = server_defaults
            .merge(model_defaults)
            .merge(user_overrides);

        assert_eq!(final_config.temperature, Some(0.9));
        assert_eq!(final_config.max_tokens, Some(512));
        assert_eq!(final_config.top_k, Some(40));
        assert_eq!(final_config.top_p, Some(0.9));
        assert_eq!(final_config.repeat_penalty, Some(1.1));
        assert_eq!(final_config.typical_p, Some(0.95));
    }

    #[test]
    fn test_builder_flow() {
        let server_params = SamplingParams {
            max_tokens: Some(2048),
            temperature: Some(0.7),
            top_p: Some(0.95),
            ..Default::default()
        };

        let model_params = SamplingParams {
            temperature: Some(0.6),
            repeat_penalty: Some(1.2),
            ..Default::default()
        };

        let user_overrides = SamplingParams {
            temperature: Some(0.8),
            max_tokens: Some(512),
            ..Default::default()
        };

        let params = server_params.merge(model_params).merge(user_overrides);
        let request = params.into_generation_request("test prompt".to_owned());

        assert_eq!(request.prompt, "test prompt");
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.max_tokens, Some(512));
        assert_eq!(request.top_p, Some(0.95));
        assert_eq!(request.repeat_penalty, Some(1.2));
    }

    #[test]
    fn test_resolved_params() {
        let params = SamplingParams {
            temperature: Some(0.5),
            max_tokens: None,
            ..Default::default()
        };

        let resolved = params.resolve();

        assert_eq!(resolved.temperature, 0.5);
        assert_eq!(resolved.max_tokens, 2048);
        assert_eq!(resolved.top_p, 0.95);
        assert_eq!(resolved.repeat_penalty, 1.0);
        assert!(resolved.do_sample);
    }

    #[test]
    fn test_oauth_secs_overrides() {
        let mut config = OAuthConfig::default();
        assert_eq!(config.active_secs(), i64::from(config.jwt_key_active_days) * 86400);
        config.jwt_key_active_secs = Some(30);
        config.jwt_key_lead_secs = Some(25);
        config.jwt_key_drain_secs = Some(20);
        config.jwt_key_rotation_check_secs = Some(3);
        assert_eq!(config.active_secs(), 30);
        assert_eq!(config.lead_secs(), 25);
        assert_eq!(config.drain_secs(), 20);
        assert_eq!(config.rotation_check_interval(), std::time::Duration::from_secs(3));
    }

    #[test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    fn test_oauth_secs_from_env() {
        // Simulate env vars the same way HyprConfig::load() does
        std::env::set_var("HYPRSTREAM__OAUTH__JWT_KEY_ACTIVE_SECS", "30");
        let result = config::Config::builder()
            .add_source(config::Config::try_from(&HyprConfig::default()).unwrap())
            .add_source(config::Environment::with_prefix("HYPRSTREAM").separator("__").try_parsing(true))
            .build()
            .and_then(config::Config::try_deserialize::<HyprConfig>);
        std::env::remove_var("HYPRSTREAM__OAUTH__JWT_KEY_ACTIVE_SECS");
        let cfg = result.expect("config should parse with env var");
        assert_eq!(cfg.oauth.jwt_key_active_secs, Some(30), "jwt_key_active_secs should be 30 from env");
        assert_eq!(cfg.oauth.active_secs(), 30);
    }

    // ---- Multi-GPU device resolution (#313) ----

    #[test]
    #[allow(clippy::unwrap_used)]
    fn parse_device_list_basic() {
        assert_eq!(RuntimeConfig::parse_device_list("0,1").unwrap(), vec![0, 1]);
        // Whitespace around entries is tolerated.
        assert_eq!(RuntimeConfig::parse_device_list(" 0 , 2 ").unwrap(), vec![0, 2]);
        assert_eq!(RuntimeConfig::parse_device_list("3").unwrap(), vec![3]);
    }

    #[test]
    fn parse_device_list_rejects_garbage() {
        // Non-numeric, empty fields, and negatives are hard errors (no silent default).
        assert!(RuntimeConfig::parse_device_list("0,foo").is_err());
        assert!(RuntimeConfig::parse_device_list("0,").is_err());
        assert!(RuntimeConfig::parse_device_list("0,,1").is_err());
        assert!(RuntimeConfig::parse_device_list("-1").is_err());
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn validate_index_set_dedup_and_empty() {
        // Empty → None (auto-detect).
        assert_eq!(RuntimeConfig::validate_index_set(vec![]).unwrap(), None);
        // Duplicates → error.
        assert!(RuntimeConfig::validate_index_set(vec![0, 0]).is_err());
        // Distinct → Some, order preserved.
        assert_eq!(
            RuntimeConfig::validate_index_set(vec![2, 0, 1]).unwrap(),
            Some(vec![2, 0, 1])
        );
    }

    /// Serializes the two tests that mutate the shared `HYPRSTREAM_GPU_DEVICES`
    /// process env var so they don't race under the parallel test runner.
    static GPU_DEVICES_ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    #[test]
    #[allow(clippy::unwrap_used)]
    fn resolve_uses_devices_field_and_legacy_fallback() {
        let _serial = GPU_DEVICES_ENV_LOCK.lock();
        // Guard against the env var leaking from the ambient environment so the
        // struct-field/legacy precedence is exercised deterministically.
        let _guard = EnvVarGuard::unset("HYPRSTREAM_GPU_DEVICES");

        // Explicit multi-device field wins.
        let mut cfg = RuntimeConfig::default();
        cfg.devices = vec![0, 1];
        cfg.gpu_device_id = Some(7);
        assert_eq!(
            cfg.resolve_explicit_multi_device_indices().unwrap(),
            Some(vec![0, 1])
        );
        assert_eq!(cfg.resolve_device_indices().unwrap(), Some(vec![0, 1]));

        // No explicit multi-device set → explicit resolver is None, but the full
        // resolver falls back to the legacy single gpu_device_id.
        let mut legacy = RuntimeConfig::default();
        legacy.devices = vec![];
        legacy.gpu_device_id = Some(3);
        assert_eq!(legacy.resolve_explicit_multi_device_indices().unwrap(), None);
        assert_eq!(legacy.resolve_device_indices().unwrap(), Some(vec![3]));

        // Nothing requested anywhere → None (auto-detect path preserved).
        let mut none = RuntimeConfig::default();
        none.devices = vec![];
        none.gpu_device_id = None;
        assert_eq!(none.resolve_device_indices().unwrap(), None);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn resolve_env_var_overrides_and_is_strict() {
        let _serial = GPU_DEVICES_ENV_LOCK.lock();
        let mut cfg = RuntimeConfig::default();
        cfg.devices = vec![5];
        cfg.gpu_device_id = Some(9);

        {
            let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "0,1,2");
            assert_eq!(
                cfg.resolve_explicit_multi_device_indices().unwrap(),
                Some(vec![0, 1, 2]),
                "env var must override the devices field"
            );
        }
        {
            // Malformed env var is a hard error (not a silent fallback to field).
            let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "0,nope");
            assert!(cfg.resolve_explicit_multi_device_indices().is_err());
        }
        {
            // Duplicate in env var → error.
            let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "1,1");
            assert!(cfg.resolve_explicit_multi_device_indices().is_err());
        }
        {
            // Explicitly-empty env var is treated as unset (falls back to field).
            let _g = EnvVarGuard::set("HYPRSTREAM_GPU_DEVICES", "  ");
            assert_eq!(
                cfg.resolve_explicit_multi_device_indices().unwrap(),
                Some(vec![5])
            );
        }
    }

    /// Serializes tests mutating the shared `HYPRSTREAM_STRICT_DEVICE` env var.
    static STRICT_DEVICE_ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// #315: strict_device defaults to fail-fast, and can be opted out via env.
    #[test]
    fn strict_device_defaults_to_true_and_respects_env() {
        let _serial = STRICT_DEVICE_ENV_LOCK.lock();

        {
            let _g = EnvVarGuard::unset("HYPRSTREAM_STRICT_DEVICE");
            assert!(
                RuntimeConfig::default().strict_device,
                "strict_device must default to true (safe default for multi-GPU)"
            );
        }
        for falsy in ["0", "false", "no", "off"] {
            let _g = EnvVarGuard::set("HYPRSTREAM_STRICT_DEVICE", falsy);
            assert!(
                !RuntimeConfig::default().strict_device,
                "HYPRSTREAM_STRICT_DEVICE={falsy} must disable strict_device"
            );
        }
        {
            let _g = EnvVarGuard::set("HYPRSTREAM_STRICT_DEVICE", "1");
            assert!(RuntimeConfig::default().strict_device);
        }
    }

    /// RAII guard to set/unset a process env var for the duration of a test and
    /// restore the previous value, keeping env-mutating tests from leaking state.
    struct EnvVarGuard {
        key: String,
        prev: Option<String>,
    }
    impl EnvVarGuard {
        fn set(key: &str, val: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, val);
            Self { key: key.to_owned(), prev }
        }
        fn unset(key: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key: key.to_owned(), prev }
        }
    }
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}

/// Generation result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: FinishReason,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    /// Quality metrics for self-supervised training
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_metrics: Option<GenerationQualityMetrics>,

    // Prefill metrics (processing the prompt)
    #[serde(default)]
    pub prefill_tokens: usize,
    #[serde(default)]
    pub prefill_time_ms: u64,
    #[serde(default)]
    pub prefill_tokens_per_sec: f32,

    // Inference metrics (generating new tokens, excluding prefill)
    #[serde(default)]
    pub inference_tokens: usize,
    #[serde(default)]
    pub inference_time_ms: u64,
    #[serde(default)]
    pub inference_tokens_per_sec: f32,

    /// Online training (TTT) adaptation metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttt_metrics: Option<TTTMetrics>,
}

/// TTT adaptation metrics (mirrors training::ttt::TTTResult)
///
/// Exposed as "Online Training" metrics in user-facing APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTTMetrics {
    pub avg_loss: f32,
    pub loss_improvement: f32,
    pub steps_performed: usize,
    pub adaptation_time_ms: u64,
    pub skipped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,

    // Advanced metrics (expert recommendation)
    pub avg_grad_norm: f32,
    pub max_grad_norm: f32,
    pub gradient_clipped: bool,
    pub tokens_used: usize,
    pub tokens_provided: usize,
    pub was_truncated: bool,

    // Tenant-aware TTT metrics
    /// Initial perplexity before adaptation
    #[serde(default)]
    pub initial_perplexity: f32,
    /// Final perplexity after adaptation
    #[serde(default)]
    pub final_perplexity: f32,
    /// Server's recommendation: true = commit, false = rollback
    #[serde(default)]
    pub recommendation: bool,
    /// Number of steps determined by perplexity gating
    #[serde(default)]
    pub gated_steps: usize,
    /// Whether adaptation is pending client commit/rollback
    #[serde(default)]
    pub pending: bool,
}

impl From<crate::training::ttt::TTTResult> for TTTMetrics {
    fn from(r: crate::training::ttt::TTTResult) -> Self {
        Self {
            avg_loss: r.avg_loss,
            loss_improvement: r.loss_improvement,
            steps_performed: r.steps_performed,
            adaptation_time_ms: r.adaptation_time_ms,
            skipped: r.skipped,
            skip_reason: r.skip_reason,
            avg_grad_norm: r.avg_grad_norm,
            max_grad_norm: r.max_grad_norm,
            gradient_clipped: r.gradient_clipped,
            tokens_used: r.tokens_used,
            tokens_provided: r.tokens_provided,
            was_truncated: r.was_truncated,
            initial_perplexity: r.initial_perplexity,
            final_perplexity: r.final_perplexity,
            recommendation: r.recommendation,
            gated_steps: r.gated_steps,
            pending: r.pending,
        }
    }
}

/// Why generation stopped
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    MaxTokens,
    StopToken(String),
    EndOfSequence,
    Error(String),
    Stop,
}

// =============================================================================
// Training Mode Configuration (Phase D)
// =============================================================================

/// Model-level training mode configuration (embedded in config.json under "hyprstream_training")
///
/// This allows inference to automatically adapt models when enabled.
/// The training mode is set via `hyprstream training set test_time_training`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyprstreamTrainingConfig {
    /// Training mode: disabled, test_time_training, supervised
    #[serde(default)]
    pub mode: TrainingMode,

    /// Target adapter to train (e.g., "01_coding")
    pub target_adapter: Option<String>,

    /// Learning rate for training
    #[serde(default = "default_training_learning_rate")]
    pub learning_rate: f64,

    /// Batch size for training (used by supervised mode)
    #[serde(default = "default_training_batch_size")]
    pub batch_size: usize,

    /// Training steps per cycle (used by supervised mode)
    #[serde(default = "default_training_steps_per_cycle")]
    pub steps_per_cycle: usize,

    /// Minimum quality score to keep examples (0.0-1.0)
    #[serde(default = "default_training_min_quality")]
    pub min_quality_threshold: f32,

    /// Enable training on base model weights (vs LoRA only)
    #[serde(default)]
    pub train_base_model: bool,

    /// TTT-specific configuration (for TestTimeTraining mode)
    #[serde(default)]
    pub ttt: TTTTrainingConfig,

    /// LoRA rank for TTT delta (default: 8)
    #[serde(default = "default_lora_rank")]
    pub lora_rank: usize,

    /// LoRA alpha scaling factor (default: None, which means alpha = rank)
    #[serde(default)]
    pub lora_alpha: Option<f32>,

    /// Target modules for LoRA adaptation (default: ["q_proj", "v_proj"])
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
}

/// TTT-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTTTrainingConfig {
    /// Learning rate for TTT adaptation (higher than fine-tuning)
    #[serde(default = "default_ttt_learning_rate")]
    pub learning_rate: f64,

    /// Number of gradient steps per input
    #[serde(default = "default_ttt_gradient_steps")]
    pub gradient_steps: u32,

    /// Maximum gradient norm for clipping
    #[serde(default = "default_ttt_max_grad_norm")]
    pub max_grad_norm: f64,

    /// Minimum input length (tokens) to trigger TTT
    #[serde(default = "default_ttt_min_input_length")]
    pub min_input_length: u32,

    /// Maximum input length to process for TTT
    #[serde(default = "default_ttt_max_context")]
    pub max_ttt_context: u32,

    /// Rank oracle configuration (optional — omit to disable runtime rank adaptation)
    #[serde(default)]
    pub rank_oracle: Option<crate::training::RankOracleConfig>,

    /// Per-layer gradient gating (optional — enabled by default)
    #[serde(default)]
    pub gradient_gating: Option<crate::training::GradientGatingConfig>,
}

fn default_ttt_learning_rate() -> f64 {
    3e-4
}
fn default_ttt_gradient_steps() -> u32 {
    3
}
fn default_ttt_max_grad_norm() -> f64 {
    1.0
}
fn default_ttt_min_input_length() -> u32 {
    32
}
fn default_ttt_max_context() -> u32 {
    512
}

impl Default for TTTTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_ttt_learning_rate(),
            gradient_steps: default_ttt_gradient_steps(),
            max_grad_norm: default_ttt_max_grad_norm(),
            min_input_length: default_ttt_min_input_length(),
            max_ttt_context: default_ttt_max_context(),
            rank_oracle: None,
            gradient_gating: None,
        }
    }
}

impl HyprstreamTrainingConfig {
    /// Check if training is enabled (mode != Disabled)
    pub fn is_enabled(&self) -> bool {
        self.mode != TrainingMode::Disabled
    }
}

impl Default for HyprstreamTrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::default(),
            target_adapter: None,
            learning_rate: default_training_learning_rate(),
            batch_size: default_training_batch_size(),
            steps_per_cycle: default_training_steps_per_cycle(),
            min_quality_threshold: default_training_min_quality(),
            train_base_model: false,
            ttt: TTTTrainingConfig::default(),
            lora_rank: default_lora_rank(),
            lora_alpha: None,
            target_modules: default_target_modules(),
        }
    }
}

/// Training mode configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    /// Training disabled (default)
    #[default]
    Disabled,
    /// Test-Time Training: adapts to input context before generation
    /// Research-valid approach based on TTT-E2E
    TestTimeTraining,
    /// Supervised training with explicit training data
    Supervised,
}

// Default functions for HyprstreamTrainingConfig
pub fn default_lora_rank() -> usize {
    8
}
pub fn default_target_modules() -> Vec<String> {
    vec!["q_proj".to_owned(), "v_proj".to_owned()]
}
fn default_training_learning_rate() -> f64 {
    1e-5
}
fn default_training_batch_size() -> usize {
    4
}
fn default_training_steps_per_cycle() -> usize {
    10
}
fn default_training_min_quality() -> f32 {
    0.3
}
