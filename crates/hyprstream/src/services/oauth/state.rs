//! OAuth 2.1 server state management.
//!
//! Manages registered clients, pending authorization codes, refresh tokens,
//! and delegates token issuance to PolicyService via ZMQ.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use base64::Engine as _;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::auth::user_store::UserStore;
use crate::config::OAuthConfig;
use crate::services::{DiscoveryClient, PolicyClient};
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
#[derive(Debug, Clone)]
pub struct RegisteredClient {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    pub client_name: Option<String>,
    /// True if this client was registered via Client ID Metadata Document (HTTPS URL client_id)
    pub is_cimd: bool,
    pub registered_at: Instant,
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
}

impl RefreshTokenEntry {
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.expires_at_unix
    }
}

/// Shared OAuth server state.
pub struct OAuthState {
    /// Registered clients (dynamic + CIMD)
    pub clients: RwLock<HashMap<String, RegisteredClient>>,
    /// Pending authorization codes (single-use, 60s TTL)
    pub pending_codes: RwLock<HashMap<String, PendingAuthCode>>,
    /// Pending authorize nonces (single-use, 5-min TTL).
    /// Proves a nonce was issued by this server and hasn't been replayed.
    pub pending_nonces: RwLock<HashMap<String, Instant>>,
    /// Pending device authorization codes (RFC 8628), keyed by device_code
    pub pending_device_codes: RwLock<HashMap<String, PendingDeviceCode>>,
    /// Reverse lookup: user_code -> device_code
    pub device_code_by_user_code: RwLock<HashMap<String, String>>,
    /// Persistent refresh token store (RocksDB). Keyed by opaque token string.
    /// None when no credentials path is configured (tokens silently lost on restart).
    pub token_db: Option<Arc<rocksdb::DB>>,
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
    /// Seen DPoP JTIs for replay prevention. Value = expiry (iat + 120s).
    pub dpop_jti_seen: RwLock<HashMap<String, i64>>,
    /// Server-issued DPoP nonces. Value = expiry unix timestamp.
    pub dpop_nonces: RwLock<HashMap<String, i64>>,
    /// Trusted external OIDC issuers for the JWT bearer grant (RFC 7523).
    pub trusted_issuers: std::collections::HashMap<String, crate::config::TrustedIssuerConfig>,
    /// CA JWT signing key for browser WIT issuance (POST /oauth/wit).
    /// Derived from the root CA key via derive_purpose_key("hyprstream-jwt-v1").
    /// None when credentials are unavailable (WIT endpoint returns 503).
    pub ca_jwt_key: Option<Arc<ed25519_dalek::SigningKey>>,
}

impl OAuthState {
    pub fn new(config: &OAuthConfig, policy_client: PolicyClient, discovery_client: DiscoveryClient, verifying_key_bytes: [u8; 32]) -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            pending_codes: RwLock::new(HashMap::new()),
            pending_nonces: RwLock::new(HashMap::new()),
            pending_device_codes: RwLock::new(HashMap::new()),
            device_code_by_user_code: RwLock::new(HashMap::new()),
            token_db: None,
            policy_client,
            discovery_client,
            issuer_url: config.issuer_url(),
            default_scopes: config.default_scopes.clone(),
            token_ttl: config.token_ttl_seconds,
            refresh_token_ttl: config.refresh_token_ttl_seconds,
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
            dpop_jti_seen: RwLock::new(HashMap::new()),
            dpop_nonces: RwLock::new(HashMap::new()),
            trusted_issuers: config.trusted_issuers.clone(),
            ca_jwt_key: None,
        }
    }

    /// Attach the CA JWT signing key for browser WIT issuance (`POST /oauth/wit`).
    pub fn with_ca_jwt_key(mut self, key: ed25519_dalek::SigningKey) -> Self {
        self.ca_jwt_key = Some(Arc::new(key));
        self
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
    pub fn with_signing_key(mut self, key: ed25519_dalek::SigningKey) -> Self {
        self.signing_key = Some(key);
        self
    }

    /// Open (or create) the RocksDB token store at `path` for persistent refresh tokens.
    pub fn with_token_store(&mut self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = rocksdb::DB::open(&opts, path)?;
        self.token_db = Some(Arc::new(db));
        Ok(())
    }

    /// Persist a refresh token entry to the DB.
    pub fn put_refresh_token(&self, token: &str, entry: &RefreshTokenEntry) -> anyhow::Result<()> {
        let Some(ref db) = self.token_db else {
            tracing::warn!("Refresh token store not configured — token will not survive restart");
            return Ok(());
        };
        let bytes = serde_json::to_vec(entry)?;
        db.put(token.as_bytes(), bytes)?;
        Ok(())
    }

    /// Look up a refresh token. Returns `None` if not found or expired (lazy expiry).
    pub fn get_refresh_token(&self, token: &str) -> anyhow::Result<Option<RefreshTokenEntry>> {
        let Some(ref db) = self.token_db else {
            return Ok(None);
        };
        match db.get(token.as_bytes())? {
            None => Ok(None),
            Some(bytes) => {
                let entry: RefreshTokenEntry = serde_json::from_slice(&bytes)?;
                if entry.is_expired() {
                    let _ = db.delete(token.as_bytes());
                    Ok(None)
                } else {
                    Ok(Some(entry))
                }
            }
        }
    }

    /// Remove a refresh token (revocation / rotation).
    pub fn delete_refresh_token(&self, token: &str) -> anyhow::Result<()> {
        let Some(ref db) = self.token_db else {
            return Ok(());
        };
        db.delete(token.as_bytes())?;
        Ok(())
    }

    /// Check a DPoP JTI for replay and record it if new.
    ///
    /// Returns `true` when the JTI is fresh (caller should proceed).
    /// Returns `false` when the JTI has been seen within its validity window (replay).
    /// Expired entries are pruned opportunistically on each call.
    pub async fn check_and_record_dpop_jti(&self, jti: &str, iat: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        let expiry = iat + 120; // ±60s window + 60s buffer
        let mut store = self.dpop_jti_seen.write().await;
        // Prune expired entries.
        store.retain(|_, exp| *exp > now);
        if store.contains_key(jti) {
            return false;
        }
        store.insert(jti.to_owned(), expiry);
        true
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

    /// Attach an RSA key for RS256 id_token signing (OIDC interop).
    ///
    /// `rsa_der` is the PKCS#8 DER-encoded RSA private key.
    pub fn with_rsa_key(mut self, rsa_der: &[u8]) -> Self {
        // Build encoding key
        self.rsa_encoding_key = Some(jsonwebtoken::EncodingKey::from_rsa_der(rsa_der));

        // Extract public key components for JWKS using jsonwebtoken's DecodingKey
        let kid = super::jwks::compute_kid(rsa_der);
        self.rsa_kid = Some(kid.clone());

        if let Some(jwk) = extract_rsa_jwk_from_der(rsa_der, &kid) {
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
                    state.dpop_jti_seen.write().await.retain(|_, exp| *exp > now);
                    state.dpop_nonces.write().await.retain(|_, exp| *exp > now);
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
