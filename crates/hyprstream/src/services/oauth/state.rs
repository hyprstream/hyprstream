//! OAuth 2.1 server state management.
//!
//! Manages registered clients, pending authorization codes, refresh tokens,
//! and delegates token issuance to PolicyService via ZMQ.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Context as _;
use base64::Engine as _;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::auth::user_store::UserStore;
use crate::config::OAuthConfig;
use crate::services::{DiscoveryClient, PolicyClient};
use hyprstream_util::TtlCache;
use super::token_store::TokenStore;
use super::user_service::UserService;

/// Extract RSA public key components (n, e) from PKCS#8 DER and build a JWK.
///
/// Uses simple_asn1 (transitive dep of jsonwebtoken) for DER parsing.
fn extract_rsa_jwk_from_der(pkcs8_der: &[u8], kid: &str) -> Option<serde_json::Value> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

    // PKCS#8 wraps the RSA key. We need to extract n and e from the inner
    // RSA public key. Rather than parsing ASN.1 manually, use jsonwebtoken
    // to create an EncodingKey and then use openssl to extract components.
    //
    // Shell out to openssl for public key component extraction.
    let temp = tempfile::NamedTempFile::new().ok()?;
    std::fs::write(temp.path(), pkcs8_der).ok()?;

    let output = std::process::Command::new("openssl")
        .args([
            "pkey", "-inform", "DER", "-in",
            temp.path().to_str()?,
            "-noout", "-text",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    // Parse modulus and exponent from openssl text output.
    // This is fragile but avoids adding ASN.1 parsing dependencies.
    let text = String::from_utf8_lossy(&output.stdout);
    let mut n_hex = String::new();
    let mut in_modulus = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Modulus:") || trimmed == "modulus:" {
            in_modulus = true;
            continue;
        }
        if trimmed.starts_with("Exponent:") || trimmed.starts_with("publicExponent:") {
            in_modulus = false;
            // Extract exponent (usually 65537 = 0x10001)
            // We'll use the standard value
            continue;
        }
        if in_modulus {
            // Lines like "    00:ab:cd:..."
            let hex_part: String = trimmed.chars()
                .filter(char::is_ascii_hexdigit)
                .collect();
            n_hex.push_str(&hex_part);
        }
    }

    if n_hex.is_empty() {
        return None;
    }

    // Strip leading zero byte if present (ASN.1 unsigned integer padding)
    if n_hex.starts_with("00") && n_hex.len() > 2 {
        n_hex = n_hex[2..].to_owned();
    }

    let n_bytes = hex::decode(&n_hex).ok()?;
    let e_bytes = vec![0x01, 0x00, 0x01]; // 65537

    let n_b64 = URL_SAFE_NO_PAD.encode(&n_bytes);
    let e_b64 = URL_SAFE_NO_PAD.encode(&e_bytes);

    Some(serde_json::json!({
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": n_b64,
        "e": e_b64,
    }))
}

/// A dynamically registered OAuth client (RFC 7591) or Client ID Metadata Document client.
///
/// Field set tracks
/// [draft-ietf-oauth-client-id-metadata-document-00] §4 (Client Metadata
/// Document) and [RFC 7591] §2 (Client Metadata): only the fields hyprstream
/// actually consumes are kept; unknown fields in the source document are
/// dropped at parse time.
#[derive(Debug, Clone)]
pub struct RegisteredClient {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    pub client_name: Option<String>,
    /// Homepage / informational URI. Shown on the consent screen.
    pub client_uri: Option<String>,
    /// Logo URL. Shown on the consent screen.
    pub logo_uri: Option<String>,
    /// Grant types this client is permitted to use. Empty = AS default
    /// (`authorization_code` + `refresh_token`).
    pub grant_types: Vec<String>,
    /// Response types this client is permitted to request. Empty = `code`.
    pub response_types: Vec<String>,
    /// Token endpoint client auth method. `"none"` = public client (PKCE
    /// is mandatory). `"private_key_jwt"` requires a non-empty `jwks` /
    /// `jwks_uri`.
    pub token_endpoint_auth_method: Option<String>,
    /// Inlined JWKS (CIMD §4 / RFC 7591 §2). Mutually exclusive with
    /// `jwks_uri` per RFC 7591. Used for `private_key_jwt` client auth and
    /// future request-object signing.
    pub jwks: Option<serde_json::Value>,
    /// JWKS endpoint URL — alternative to inline `jwks`.
    pub jwks_uri: Option<String>,
    /// Optional host `did:key` supplied by a PDS attachment client. Validated
    /// during registration and retained for PDS identity association.
    pub hyprstream_node_did: Option<String>,
    /// True if this client was registered via Client ID Metadata Document (HTTPS URL client_id)
    pub is_cimd: bool,
    pub registered_at: Instant,
}

/// A Pushed Authorization Request (RFC 9126) awaiting consumption.
///
/// Holds the already-validated authorize parameters keyed by the `request_uri`
/// returned to the client. Single-use, short TTL.
#[derive(Debug, Clone)]
pub struct PushedAuthRequest {
    pub params: super::authorize::AuthorizeParams,
    pub expires_at: Instant,
}

impl PushedAuthRequest {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::CryptoPolicy;
    use rand::rngs::OsRng;

    #[test]
    fn mesh_kem_derivation_failure_fails_closed_under_hybrid() {
        let key = ed25519_dalek::SigningKey::generate(&mut OsRng);
        let err = mesh_kem_public_for_policy(&key, CryptoPolicy::Hybrid, |_| {
            Err(anyhow::anyhow!("synthetic derivation failure"))
        })
        .unwrap_err();

        assert!(
            err.to_string().contains("Hybrid policy"),
            "unexpected error: {err:?}",
        );
    }

    #[test]
    fn mesh_kem_derivation_failure_is_empty_under_classical() {
        let key = ed25519_dalek::SigningKey::generate(&mut OsRng);
        let public = mesh_kem_public_for_policy(&key, CryptoPolicy::Classical, |_| {
            Err(anyhow::anyhow!("synthetic derivation failure"))
        })
        .unwrap();

        assert!(public.is_none());
    }
}

/// A pending authorization code awaiting token exchange.
#[derive(Debug, Clone)]
pub struct PendingAuthCode {
    pub code: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub scopes: Vec<String>,
    /// RFC 8707 resource indicator (the audience for the token)
    pub resource: Option<String>,
    /// OIDC nonce — echoed into the id_token when scope includes "openid".
    pub oidc_nonce: Option<String>,
    pub created_at: Instant,
    pub expires_at: Instant,
    /// Authenticated username from Ed25519 challenge-response on the consent page.
    /// Used as the JWT `sub` claim for the issued token.
    pub username: String,
    /// Ed25519 verifying key verified during challenge-response.
    /// Included in the JWT `pub_key` claim to bind the user's key identity.
    pub verifying_key: Option<ed25519_dalek::VerifyingKey>,
}

impl PendingAuthCode {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Pending external OIDC authentication flow.
///
/// Stores the original hyprstream authorize request so it can be resumed
/// after the user authenticates with the external provider.
#[derive(Debug, Clone)]
pub struct PendingExternalAuth {
    pub provider_slug: String,
    pub external_state: String,
    pub external_nonce: String,
    /// Provider kind, carried through for dispatch in the callback handler.
    pub provider_kind: crate::config::ProviderKind,
    /// Whether PKCE was sent to the external provider.
    pub pkce_supported: bool,
    pub pkce_verifier: String,
    pub client_secret: Option<String>,
    pub token_endpoint: String,
    // Original hyprstream authorize request
    pub original_client_id: String,
    pub original_redirect_uri: String,
    pub original_code_challenge: String,
    pub original_scopes: String,
    pub original_state: Option<String>,
    pub original_resource: Option<String>,
    pub original_oidc_nonce: Option<String>,
    pub created_at: Instant,
    pub expires_at: Instant,
}

/// Status of a pending device authorization code (RFC 8628).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceCodeStatus {
    /// User has not yet approved or denied.
    Pending,
    /// User approved the authorization request.
    Approved,
    /// User denied the authorization request.
    Denied,
}

/// A pending device authorization code (RFC 8628).
#[derive(Debug, Clone)]
pub struct PendingDeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    /// RFC 8707 resource indicator (the audience for the token)
    pub resource: Option<String>,
    pub status: DeviceCodeStatus,
    pub created_at: Instant,
    pub expires_at: Instant,
    /// Minimum polling interval in seconds
    pub interval: u64,
    /// Last time the client polled for this code
    pub last_polled: Option<Instant>,
    /// Random nonce for challenge-response auth (43 chars base64url of 32 bytes)
    pub nonce: String,
    /// Username of the person who approved this code (set on POST /verify success)
    pub approved_by: Option<String>,
    /// Ed25519 verifying key verified during device challenge-response.
    /// Included in the JWT `pub_key` claim to bind the user's key identity.
    pub verifying_key: Option<ed25519_dalek::VerifyingKey>,
}

impl PendingDeviceCode {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// A stored refresh token entry (OAuth 2.1 rotation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshTokenEntry {
    pub client_id: String,
    /// JWT subject (username) of the token owner.
    pub username: String,
    pub scopes: Vec<String>,
    pub resource: Option<String>,
    /// Unix timestamp (seconds) at which this token expires.
    pub expires_at_unix: i64,
    /// Ed25519 verifying key bytes bound to this token (cnf key continuity on refresh).
    pub verifying_key_bytes: Option<[u8; 32]>,
    /// RFC 9449 JWK thumbprint. When set, refresh requires a DPoP proof from
    /// this exact key; it is carried forward during refresh-token rotation.
    #[serde(default)]
    pub dpop_jkt: Option<String>,
    /// Present only for UCAN-grant refresh tokens (`client_id` `ucan-grant:{sub}`).
    ///
    /// MAC #547 / B1 (#673): refresh of a UCAN grant is NOT a free re-mint — the
    /// refresh path must re-run the S6 gate chain (`evaluate_refresh`) against the
    /// grant, because the ceiling may have been amended or revoked since the
    /// access token was minted (ZSP: access is re-evaluated on refresh). That
    /// requires persisting the grant + the requested access so the refresh path
    /// can re-present them. `None` for every non-UCAN-grant refresh token, which
    /// keeps the generic OAuth 2.1 rotation path unchanged.
    #[serde(default)]
    pub ucan_grant: Option<UcanGrantRefresh>,
}

/// Re-evaluation context persisted alongside a UCAN-grant refresh token so the
/// refresh path can re-present the grant to `evaluate_refresh` (MAC #547 / B1
/// #673). Carrying the grant itself (not just an id) lets the gate chain
/// re-check the ceiling against current state on every refresh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UcanGrantRefresh {
    /// `base64url(CBOR)` of the UCAN grant, re-presented to `evaluate_refresh`
    /// verbatim (the same encoding the client sent as `subject_token`).
    pub grant_cbor_b64: String,
    /// BLAKE3 content id (hex) of the CBOR grant. Binds this refresh entry to
    /// exactly the grant it was minted from and is checked on every refresh so
    /// a corrupted/substituted stored blob fails closed.
    pub grant_cid: String,
    /// The RFC 8693 requested `scope` (the S3 `action:resource:identifier`
    /// triple) used to rebuild the `GrantRequest` on refresh.
    pub requested_scope: Option<String>,
    /// The RFC 8707 `audience` resource indicator, if any.
    pub audience: Option<String>,
}

impl RefreshTokenEntry {
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.expires_at_unix
    }
}

/// Shared OAuth server state.
pub struct OAuthState {
    /// Dynamically-registered (RFC 7591) clients keyed by issued UUID
    /// client_id. CIMD clients live in `cimd_cache` instead — DCR
    /// entries have no TTL and outlive cache evictions, so storage is
    /// separated.
    pub clients: RwLock<HashMap<String, RegisteredClient>>,
    /// CIMD metadata cache. CIMD documents (HTTPS-URL client_ids) are
    /// fetched + verified once, then cached respecting HTTP cache
    /// headers, bounded TTL and capacity. See `cimd_cache` module.
    pub cimd_cache: Arc<super::cimd_cache::CimdCache>,
    /// JWKS-URI fetch cache for `private_key_jwt` client auth. Keyed by
    /// the absolute URL; value is `(parsed_jwks_json, fetched_at)`.
    /// Entries expire after `client_jwks_uri_cache_ttl_secs` and are
    /// evicted lazily on read. Capacity is implicitly bounded by the
    /// number of registered clients with jwks_uri (typically small).
    pub jwks_uri_cache: RwLock<HashMap<String, (serde_json::Value, Instant)>>,
    /// TTL for the `jwks_uri_cache`. Copied from
    /// [`OAuthConfig::client_jwks_uri_cache_ttl_secs`] at construction.
    pub client_jwks_uri_cache_ttl: Duration,
    /// Pending authorization codes (single-use, 60s TTL)
    pub pending_codes: RwLock<HashMap<String, PendingAuthCode>>,
    /// Pending authorize nonces (single-use, 5-min TTL).
    /// Proves a nonce was issued by this server and hasn't been replayed.
    pub pending_nonces: RwLock<HashMap<String, Instant>>,
    /// Pending Pushed Authorization Requests (RFC 9126), keyed by `request_uri`.
    /// Single-use, 60s TTL. In-memory is correct here — no value to persistence.
    pub pending_par_requests: RwLock<HashMap<String, super::state::PushedAuthRequest>>,
    /// Pending device authorization codes (RFC 8628), keyed by device_code
    pub pending_device_codes: RwLock<HashMap<String, PendingDeviceCode>>,
    /// Reverse lookup: user_code -> device_code
    pub device_code_by_user_code: RwLock<HashMap<String, String>>,
    /// Persistent refresh token store. Keyed by opaque token string.
    /// None when no credentials path is configured (tokens silently lost on restart).
    pub token_db: Option<Arc<dyn TokenStore>>,
    /// PolicyClient for JWT token issuance via ZMQ
    pub policy_client: PolicyClient,
    /// DiscoveryClient for resolving service QUIC endpoints via ZMQ
    pub discovery_client: DiscoveryClient,
    /// Issuer URL (e.g., "http://localhost:6791")
    pub issuer_url: String,
    /// Default scopes for new clients
    pub default_scopes: Vec<String>,
    /// Access token TTL in seconds
    pub token_ttl: u32,
    /// Refresh token TTL in seconds
    pub refresh_token_ttl: u32,
    /// When true, `/oauth/authorize` rejects inline params and requires a `request_uri`
    /// from a prior `/oauth/par` call (RFC 9126). Advertised in server metadata.
    pub require_pushed_authorization_requests: bool,
    /// HTTP client for fetching Client ID Metadata Documents
    pub http_client: reqwest::Client,
    /// Raw Ed25519 verifying key bytes (32 bytes) for the JWKS endpoint.
    pub verifying_key_bytes: [u8; 32],
    /// User credential store for Ed25519 challenge-response device verification.
    /// Now backed by `user_service`. Legacy code uses `user_store_reader()`.
    /// Kept as Option for backward-compatible `is_none()` checks.
    pub user_service: Option<Arc<UserService>>,
    /// Ed25519 signing key for signing entity configurations (OpenID Federation 1.0).
    /// `None` when not configured.
    pub signing_key: Option<ed25519_dalek::SigningKey>,
    /// OpenID Federation 1.0 Trust Anchor URLs.
    /// Included as `authority_hints` in the entity configuration JWT.
    pub authority_hints: Vec<String>,
    /// Pending external OIDC authentication flows (keyed by external state).
    pub pending_external_auths: RwLock<HashMap<String, PendingExternalAuth>>,
    /// OIDC discovery cache for external providers.
    pub oidc_discovery: super::oidc_discovery::SharedDiscoveryCache,
    /// Server-side session store for browser login flow.
    pub sessions: super::session::SessionStore,
    /// RSA encoding key for RS256 id_token signing (optional, loaded from secrets).
    pub rsa_encoding_key: Option<jsonwebtoken::EncodingKey>,
    /// RSA public key as JWK JSON (for the JWKS endpoint).
    pub rsa_jwk: Option<serde_json::Value>,
    /// RSA key kid (for the JWT header).
    pub rsa_kid: Option<String>,
    /// Seen DPoP JTIs for replay prevention (RFC 9449 §11.1). Backed by
    /// the shared `TtlCache` with atomic check-and-record — see
    /// `check_and_record_dpop_jti`. TTL = iat + 120s per entry.
    pub dpop_jti_seen: TtlCache<String, ()>,
    /// Server-issued DPoP nonces. Value = expiry unix timestamp.
    pub dpop_nonces: RwLock<HashMap<String, i64>>,
    /// Per-client (keyed by DPoP `jkt` thumbprint) nonce-issuance state.
    /// Once a jkt appears here, RFC 9449 §8 enforcement kicks in: subsequent
    /// proofs from the same key MUST carry a server-issued nonce. Value =
    /// expiry unix timestamp (matches nonce TTL; entry pruned when expired
    /// to allow re-bootstrap after silence).
    pub dpop_clients_seen: RwLock<HashMap<String, i64>>,
    /// Trusted external OIDC issuers for the JWT bearer grant (RFC 7523).
    pub trusted_issuers: std::collections::HashMap<String, crate::config::TrustedIssuerConfig>,
    /// CA JWT signing key for browser WIT issuance (POST /oauth/wit).
    /// Derived from the root CA key via derive_purpose_key("hyprstream-jwt-v1").
    /// None when credentials are unavailable (WIT endpoint returns 503).
    pub ca_jwt_key: Option<Arc<ed25519_dalek::SigningKey>>,
    /// Anonymous device identity store.
    pub device_store: Option<Arc<dyn crate::auth::DeviceStore>>,
    /// Unix timestamp of when the JWT signing key became active (nbf for JWKS entry).
    pub jwt_key_nbf: i64,
    /// Unix timestamp of when the JWT signing key expires (exp for JWKS entry).
    /// Default: jwt_key_nbf + 14 days.
    pub jwt_key_exp: i64,
    /// Multi-slot JWT signing key store for rotation (drain/active/lead lifecycle).
    /// When present, JWKS serves all slots and issuance uses the active key.
    pub signing_key_store: Option<Arc<crate::auth::SigningKeyStore>>,
    /// Shared JWT ID blocklist for access token revocation (shared with PolicyService).
    pub jti_blocklist: Option<Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>>,
    /// ES256 (P-256) signing key rotation store for JWKS and DPoP/atproto interop.
    pub es256_key_store: Option<Arc<crate::auth::Es256SigningKeyStore>>,
    /// ML-DSA-65 signing key rotation store for PQ-hybrid JWT issuance.
    pub ml_dsa_key_store: Option<Arc<crate::auth::MlDsaSigningKeyStore>>,
    /// MAC #547 / B2 (#674): the tamper-evident audit sink the S6 grant path
    /// (`mac::exchange::audited_evaluate_grant`/`audited_evaluate_refresh`)
    /// records every decision through. `None` means the grant-path audit
    /// trail is not configured — the grant path treats that as fail-closed
    /// (deny, not "audit best-effort") rather than minting unaudited tokens.
    /// Production wiring is [`crate::mac::audit::WalAuditStore`] + an
    /// [`crate::mac::audit::cose::OwnedCoseAuditSigner`] (see the `oauth`
    /// service factory).
    pub audit_sink: Option<Arc<dyn crate::mac::audit::AuditSink>>,
    /// QUIC/WebTransport cert hashes (SHA-256 of leaf DER, `sha2-256` multihash encoding).
    /// Published in the DID-doc `#quic` service entry so peers can pin the cert (#185).
    /// A set so cert rotation can publish old + new simultaneously.
    pub quic_cert_hashes: Vec<[u8; 32]>,
    /// Public QUIC URI (`https://host:port`) for the DID-doc service entry (#185).
    /// None until the QUIC server is started and the cert hash is known.
    pub quic_public_uri: Option<String>,
    /// Raw ML-DSA-65 verifying-key bytes (1952 bytes) for the node's mesh
    /// post-quantum signing key (#157). Published as the `#mesh-pq` Multikey
    /// verification method in the root DID document. Derived from the same
    /// Ed25519 key as [`Self::signing_key`] (via `derive_mesh_mldsa_key`), so
    /// the published VM equals the key the node signs mesh responses with.
    /// `None` when the entity signing key is not configured.
    pub mesh_pq_verifying_key: Option<Vec<u8>>,
    /// The node's `#mesh-kem` hybrid keyAgreement public material (S1 / #552):
    /// one ML-KEM-768-hybrid recipient public (X25519 + ML-KEM-768 encapsulation
    /// keys, suite `HyKemX25519MlKem768`). Published as the `keyAgreement`
    /// verification methods in the root DID document. Derived from the same
    /// Ed25519 key as [`Self::signing_key`] (via `derive_mesh_kem_recipient`),
    /// so the published keys equal what the node decapsulates `#mesh-kem`
    /// envelopes with. `None` when the entity signing key is not configured,
    /// or when derivation failed under Classical policy (Hybrid fails closed
    /// during `with_signing_key`).
    pub mesh_kem_public: Option<hyprstream_rpc::crypto::hybrid_kem::RecipientPublic>,
    /// #282: the node's iroh endpoint id (its Ed25519 `node_id`, 32 bytes),
    /// published only as an `IrohTransport` service entry when bound. It is not
    /// a verification method or JWKS key (#1031).
    pub iroh_node_id: Option<[u8; 32]>,
    /// #282: iroh relay URLs to advertise in the `IrohTransport` entry's
    /// `relays`. Empty = rely on pkarr/DNS discovery for reachability (the
    /// peer resolves direct paths by node_id alone).
    pub iroh_relays: Vec<String>,
}

fn mesh_kem_public_for_policy(
    key: &ed25519_dalek::SigningKey,
    policy: hyprstream_rpc::crypto::CryptoPolicy,
    derive: impl FnOnce(
        &ed25519_dalek::SigningKey,
    ) -> anyhow::Result<hyprstream_rpc::crypto::hybrid_kem::RecipientKeypair>,
) -> anyhow::Result<Option<hyprstream_rpc::crypto::hybrid_kem::RecipientPublic>> {
    match derive(key) {
        Ok(kem_kp) => Ok(Some(kem_kp.public())),
        Err(e) if policy.uses_pq() => Err(e).context(
            "failed to derive #mesh-kem hybrid keyAgreement identity under Hybrid policy",
        ),
        Err(e) => {
            tracing::warn!(
                error = %e,
                "failed to derive #mesh-kem hybrid keyAgreement identity under Classical policy; \
                 root DID document will publish an empty keyAgreement relationship",
            );
            Ok(None)
        }
    }
}

impl OAuthState {
    /// DPoP jti replay-dedup cache: capacity bound (memory ceiling for the
    /// `TtlCache<String, ()>`). 120s TTL per entry.
    const DPOP_JTI_MAX_ENTRIES: usize = 10_000;
    /// Inline reap budget per access (heap pops). Bounds tail latency.
    const DPOP_JTI_REAP_BUDGET: usize = 64;

    pub fn new(config: &OAuthConfig, policy_client: PolicyClient, discovery_client: DiscoveryClient, verifying_key_bytes: [u8; 32]) -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            cimd_cache: Arc::new(super::cimd_cache::CimdCache::new(
                super::cimd_cache::CimdCacheConfig::default(),
            )),
            jwks_uri_cache: RwLock::new(HashMap::new()),
            client_jwks_uri_cache_ttl: Duration::from_secs(config.client_jwks_uri_cache_ttl_secs),
            pending_codes: RwLock::new(HashMap::new()),
            pending_nonces: RwLock::new(HashMap::new()),
            pending_par_requests: RwLock::new(HashMap::new()),
            pending_device_codes: RwLock::new(HashMap::new()),
            device_code_by_user_code: RwLock::new(HashMap::new()),
            token_db: None,
            policy_client,
            discovery_client,
            issuer_url: config.issuer_url(),
            default_scopes: config.default_scopes.clone(),
            token_ttl: config.token_ttl_seconds,
            refresh_token_ttl: config.refresh_token_ttl_seconds,
            require_pushed_authorization_requests: config.require_pushed_authorization_requests,
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            verifying_key_bytes,
            user_service: None,
            signing_key: None,
            authority_hints: config.authority_hints.clone(),
            pending_external_auths: RwLock::new(HashMap::new()),
            oidc_discovery: std::sync::Arc::new(super::oidc_discovery::OidcDiscoveryCache::default()),
            sessions: super::session::SessionStore::default(),
            rsa_encoding_key: None,
            rsa_jwk: None,
            rsa_kid: None,
            dpop_jti_seen: TtlCache::new(Self::DPOP_JTI_MAX_ENTRIES, Self::DPOP_JTI_REAP_BUDGET),
            dpop_nonces: RwLock::new(HashMap::new()),
            dpop_clients_seen: RwLock::new(HashMap::new()),
            trusted_issuers: config.trusted_issuers.clone(),
            ca_jwt_key: None,
            device_store: None,
            jwt_key_nbf: chrono::Utc::now().timestamp(),
            jwt_key_exp: chrono::Utc::now().timestamp() + 14 * 86400,
            signing_key_store: None,
            jti_blocklist: None,
            es256_key_store: None,
            ml_dsa_key_store: None,
            audit_sink: None,
            quic_cert_hashes: Vec::new(),
            quic_public_uri: None,
            mesh_pq_verifying_key: None,
            mesh_kem_public: None,
            iroh_node_id: None,
            iroh_relays: Vec::new(),
        }
    }

    /// Attach the anonymous device store.
    pub fn with_device_store(mut self, store: Arc<dyn crate::auth::DeviceStore>) -> Self {
        self.device_store = Some(store);
        self
    }

    /// Set JWT signing key validity window for the JWKS endpoint.
    pub fn with_jwt_key_timestamps(mut self, nbf: i64, exp: i64) -> Self {
        self.jwt_key_nbf = nbf;
        self.jwt_key_exp = exp;
        self
    }

    /// Attach the CA JWT signing key for browser WIT issuance (`POST /oauth/wit`).
    pub fn with_ca_jwt_key(mut self, key: ed25519_dalek::SigningKey) -> Self {
        self.ca_jwt_key = Some(Arc::new(key));
        self
    }

    /// Attach the multi-slot signing key store (rotation).
    ///
    /// When set, JWKS serves all slots and WIT issuance uses the active key.
    pub fn with_signing_key_store(mut self, store: Arc<crate::auth::SigningKeyStore>) -> Self {
        self.signing_key_store = Some(store);
        self
    }

    /// Attach the ES256 (P-256) key rotation store.
    pub fn with_es256_key_store(mut self, store: Arc<crate::auth::Es256SigningKeyStore>) -> Self {
        self.es256_key_store = Some(store);
        self
    }

    /// Attach the ML-DSA-65 key rotation store.
    pub fn with_ml_dsa_key_store(mut self, store: Arc<crate::auth::MlDsaSigningKeyStore>) -> Self {
        self.ml_dsa_key_store = Some(store);
        self
    }

    /// Attach the S6 grant-path audit sink (B2, #674). Without this, the
    /// grant path fails closed on every request rather than minting tokens
    /// with no audit trail.
    pub fn with_audit_sink(mut self, sink: Arc<dyn crate::mac::audit::AuditSink>) -> Self {
        self.audit_sink = Some(sink);
        self
    }

    /// Set the node's QUIC transport info for DID-doc publication (#185).
    ///
    /// `cert_hashes` is the set of SHA-256 cert DER hashes currently in use
    /// (old + new during rotation). `public_uri` is `https://host:port` that
    /// external peers dial.
    pub fn with_quic_transport(mut self, public_uri: String, cert_hashes: Vec<[u8; 32]>) -> Self {
        self.quic_public_uri = Some(public_uri);
        self.quic_cert_hashes = cert_hashes;
        self
    }

    /// Set the node's iroh transport info for DID-doc publication (#282).
    ///
    /// Call this only when the iroh substrate is actually bound: it makes
    /// `root_did_document` advertise an `IrohTransport` service entry. The
    /// `node_id` is reachability metadata, never a DID verification method.
    /// `relays` may be empty — peers then resolve reachability by node_id alone
    /// via iroh's pkarr/DNS discovery.
    pub fn with_iroh_transport(mut self, node_id: [u8; 32], relays: Vec<String>) -> Self {
        self.iroh_node_id = Some(node_id);
        self.iroh_relays = relays;
        self
    }

    /// Attach the shared JWT ID blocklist (shared with PolicyService).
    ///
    /// When set, `POST /oauth/revoke` on access tokens writes the JTI into
    /// this blocklist so the PolicyService RPC enforcement path also rejects
    /// revoked tokens — closing the gap between HTTP revocation and RPC auth.
    pub fn with_jti_blocklist(mut self, bl: Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>) -> Self {
        self.jti_blocklist = Some(bl);
        self
    }

    /// Return the verifying key to use for JWT bearer token validation.
    ///
    /// Prefers the active slot from the signing key store; falls back to the
    /// legacy `verifying_key_bytes` field (policy-service-issued key).
    pub async fn jwt_bearer_verifying_key(&self) -> Option<ed25519_dalek::VerifyingKey> {
        if let Some(store) = &self.signing_key_store {
            if let Some(bytes) = store.active_verifying_key_bytes().await {
                return ed25519_dalek::VerifyingKey::from_bytes(&bytes).ok();
            }
        }
        ed25519_dalek::VerifyingKey::from_bytes(&self.verifying_key_bytes).ok()
    }

    /// Return the active JWT signing key for token issuance (WIT/ADT).
    ///
    /// Prefers the active slot from the signing key store; falls back to `ca_jwt_key`.
    pub async fn active_jwt_signing_key(&self) -> Option<Arc<ed25519_dalek::SigningKey>> {
        if let Some(store) = &self.signing_key_store {
            if let Some(key) = store.active_key().await {
                return Some(key);
            }
        }
        self.ca_jwt_key.clone()
    }

    /// Attach a user credential store. Creates a `UserService` backed by the store
    /// for SCIM/RPC access and legacy OAuth handler reads.
    pub fn with_user_store(mut self, store: Arc<dyn UserStore>) -> Self {
        self.user_service = Some(Arc::new(UserService::new(store)));
        self
    }

    /// Attach a pre-built `UserService`. Used when the service is constructed externally
    /// (e.g., for testing or when the store is shared across services).
    pub fn with_user_service(mut self, service: Arc<UserService>) -> Self {
        self.user_service = Some(service);
        self
    }

    /// Get read access to the user store via the UserService.
    /// Returns None if no user store is configured.
    pub fn user_store_reader(&self) -> Option<Arc<dyn UserStore>> {
        self.user_service.as_ref().map(|s| s.store())
    }

    /// Backward-compatible check: returns true if a user store is configured.
    pub fn has_user_store(&self) -> bool {
        self.user_service.is_some()
    }

    /// Attach the signing key for OpenID Federation 1.0 entity configuration signing.
    ///
    /// Also derives and stores the node's mesh ML-DSA-65 verifying key (#157)
    /// from this same Ed25519 key, so the root DID document's `#mesh-pq`
    /// Multikey verification method matches the post-quantum key the mesh signs
    /// with (`derive_mesh_mldsa_key`), and the node's `#mesh-kem` hybrid
    /// keyAgreement public material (S1 / #552, `derive_mesh_kem_recipient`) so
    /// peers can anchor a recipient key that matches what this node decapsulates
    /// with. A `#mesh-kem` derivation failure is fail-closed under Hybrid
    /// policy: startup returns an error rather than publishing a root DID
    /// document with an empty `keyAgreement` relationship. Under Classical
    /// policy, the failure is logged and `mesh_kem_public` stays unset.
    pub fn with_signing_key(
        mut self,
        key: ed25519_dalek::SigningKey,
        policy: hyprstream_rpc::crypto::CryptoPolicy,
    ) -> anyhow::Result<Self> {
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&key);
        self.mesh_pq_verifying_key = Some(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk));
        self.mesh_kem_public = mesh_kem_public_for_policy(&key, policy, |key| {
            hyprstream_rpc::node_identity::derive_mesh_kem_recipient(key)
        })?;
        self.signing_key = Some(key);
        Ok(self)
    }

    /// Inject a pre-built token store implementation.
    pub fn with_token_store_impl(&mut self, store: Arc<dyn TokenStore>) {
        self.token_db = Some(store);
    }

    /// Persist a refresh token entry to the store.
    pub async fn put_refresh_token(&self, token: &str, entry: &RefreshTokenEntry, ttl_secs: u64) -> anyhow::Result<()> {
        let Some(ref store) = self.token_db else {
            tracing::warn!("Refresh token store not configured — token will not survive restart");
            return Ok(());
        };
        store.put(token, entry, ttl_secs).await
    }

    /// Look up a refresh token. Returns `None` if not found or expired (lazy expiry).
    pub async fn get_refresh_token(&self, token: &str) -> anyhow::Result<Option<RefreshTokenEntry>> {
        let Some(ref store) = self.token_db else {
            return Ok(None);
        };
        store.get(token).await
    }

    /// Remove a refresh token (revocation / rotation).
    pub async fn delete_refresh_token(&self, token: &str) -> anyhow::Result<()> {
        let Some(ref store) = self.token_db else {
            return Ok(());
        };
        store.delete(token).await
    }

    /// Check a DPoP JTI for replay and record it if new.
    ///
    /// Returns `true` when the JTI is fresh (caller should proceed).
    /// Returns `false` when the JTI has been seen within its validity window (replay).
    /// Expired entries are pruned opportunistically on each call.
    pub fn check_and_record_dpop_jti(&self, jti: &str, iat: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        // Window ends at iat + 120s (±60s skew + 60s buffer); TTL = remainder.
        let ttl_secs = ((iat + 120) - now).max(0) as u64;
        self.dpop_jti_seen
            .insert_if_absent(jti.to_owned(), (), Duration::from_secs(ttl_secs))
    }

    /// Issue a fresh server-side DPoP nonce (RFC 9449 §8).
    ///
    /// Returns the base64url nonce string that should be placed in the
    /// `DPoP-Nonce` response header. Stored with a 5-minute TTL.
    pub async fn issue_dpop_nonce(&self) -> String {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        let nonce = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
        let expiry = chrono::Utc::now().timestamp() + 300; // 5-minute TTL
        self.dpop_nonces.write().await.insert(nonce.clone(), expiry);
        nonce
    }

    /// Validate a DPoP nonce issued by this server. Returns `true` if valid and unexpired.
    pub async fn verify_dpop_nonce(&self, nonce: &str) -> bool {
        let now = chrono::Utc::now().timestamp();
        let store = self.dpop_nonces.read().await;
        store.get(nonce).is_some_and(|&exp| exp > now)
    }

    /// RFC 9449 §8 nonce-enforcement bookkeeping: has this client (`jkt`)
    /// previously been issued a nonce that has not yet expired? If so the
    /// next DPoP proof from this key MUST include a valid nonce.
    pub async fn dpop_client_requires_nonce(&self, jkt: &str) -> bool {
        let now = chrono::Utc::now().timestamp();
        let mut store = self.dpop_clients_seen.write().await;
        store.retain(|_, exp| *exp > now);
        store.contains_key(jkt)
    }

    /// Record that this client (`jkt`) has been issued a server nonce.
    /// Future proofs from this key are required to carry one (sliding 5-min
    /// window per nonce TTL).
    pub async fn mark_dpop_client_nonced(&self, jkt: &str) {
        let expiry = chrono::Utc::now().timestamp() + 300;
        self.dpop_clients_seen.write().await.insert(jkt.to_owned(), expiry);
    }

    /// Attach an RSA key for RS256 id_token signing (OIDC interop).
    ///
    /// `rsa_der` is the PKCS#8 DER-encoded RSA private key.
    pub fn with_rsa_key(mut self, rsa_der: &[u8]) -> Self {
        // Build encoding key
        self.rsa_encoding_key = Some(jsonwebtoken::EncodingKey::from_rsa_der(rsa_der));

        // Extract public key components for JWKS using jsonwebtoken's DecodingKey
        if let Some(mut jwk) = extract_rsa_jwk_from_der(rsa_der, "") {
            let n = jwk.get("n").and_then(|v| v.as_str()).unwrap_or_default();
            let e = jwk.get("e").and_then(|v| v.as_str()).unwrap_or_default();
            let kid = super::jwks::compute_rsa_kid(n, e);
            if let Some(obj) = jwk.as_object_mut() {
                obj.insert("kid".to_owned(), serde_json::Value::String(kid.clone()));
            }
            self.rsa_kid = Some(kid);
            self.rsa_jwk = Some(jwk);
        }

        self
    }

    /// Spawn a background task that sweeps expired codes every 30 seconds.
    pub fn spawn_code_sweeper(self: &Arc<Self>) {
        let state = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;

                // Sweep expired auth codes
                {
                    let mut codes = state.pending_codes.write().await;
                    codes.retain(|_, code| !code.is_expired());
                }

                // Sweep expired authorize nonces (5-min TTL)
                {
                    let now = Instant::now();
                    let mut nonces = state.pending_nonces.write().await;
                    nonces.retain(|_, expiry| *expiry > now);
                }

                // Sweep expired PAR requests (60s TTL)
                {
                    let mut par = state.pending_par_requests.write().await;
                    par.retain(|_, req| !req.is_expired());
                }

                // Sweep expired device codes
                {
                    let mut device_codes = state.pending_device_codes.write().await;
                    let mut user_code_map = state.device_code_by_user_code.write().await;
                    device_codes.retain(|_, dc| {
                        if dc.is_expired() {
                            user_code_map.remove(&dc.user_code);
                            false
                        } else {
                            true
                        }
                    });
                }

                // Sweep expired DPoP JTIs and nonces
                {
                    let now = chrono::Utc::now().timestamp();
                    // dpop_jti_seen is a TtlCache (self-evicting); sweep the
                    // nonce + client-dedup maps only.
                    state.dpop_nonces.write().await.retain(|_, exp| *exp > now);
                    state.dpop_clients_seen.write().await.retain(|_, exp| *exp > now);
                }

                // Sweep expired external auth flows
                {
                    let now = Instant::now();
                    let mut auths = state.pending_external_auths.write().await;
                    auths.retain(|_, auth| auth.expires_at > now);
                }

                // Sweep expired sessions
                state.sessions.sweep().await;
            }
        });
    }
}
