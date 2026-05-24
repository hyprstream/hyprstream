//! JWKS fetch and cache for multi-issuer JWT verification.
//!
//! `FederationKeyResolver` resolves external issuer URLs to Ed25519 verifying
//! keys by fetching and caching JWKS (RFC 7517) documents.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::VerifyingKey;
use hyprstream_rpc::auth::FederationKeySource;
use hyprstream_discovery::DiscoveryClient;

struct CachedKey {
    key: VerifyingKey,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedKey {
    fn is_expired(&self) -> bool {
        self.fetched_at.elapsed() > self.ttl
    }
}

/// Resolves external issuer URLs to Ed25519 verifying keys via JWKS.
///
/// Thread-safe (uses `Arc<RwLock<...>>`). Caches keys to avoid fetching
/// on every request. Falls back to OIDC metadata discovery if no explicit
/// JWKS URI is configured.
///
/// By default all JWKS URIs must use HTTPS with TLS certificate verification.
/// Set `allow_http: true` in `TrustedIssuerConfig` on a per-issuer basis to
/// permit plain HTTP (development / internal use only).
pub struct FederationKeyResolver {
    /// issuer_url → cached key.
    ///
    /// A single `Mutex` (not `RwLock`) serialises both reads and writes so
    /// that concurrent cache-miss events for the same issuer do not race:
    /// the first waiter fetches, writes the result, and the second waiter
    /// then finds a valid entry on re-check.
    cache: Mutex<HashMap<String, CachedKey>>,
    /// issuer_url → (optional jwks_uri_override, ttl, allow_http)
    config: HashMap<String, (Option<String>, Duration, bool)>,
    http: reqwest::Client,
    /// Phase 0.5 Stage D — optional DiscoveryService client.
    ///
    /// When configured, `get_key` consults DiscoveryService for a cached
    /// signed entity statement before falling back to HTTPS JWKS fetch.
    /// This avoids the HTTPS round-trip for local-issuer verification and
    /// the 300s TTL becomes irrelevant for cluster-internal traffic.
    /// Falls through to HTTPS on miss/error (conservative: never downgrade
    /// trust on missing data).
    discovery_client: Option<Arc<DiscoveryClient>>,
}

impl FederationKeyResolver {
    pub fn new(
        trusted_issuers: &HashMap<String, crate::config::TrustedIssuerConfig>,
    ) -> Self {
        let config = trusted_issuers
            .iter()
            .map(|(iss, cfg)| {
                (
                    iss.clone(),
                    (
                        cfg.jwks_uri.clone(),
                        Duration::from_secs(cfg.jwks_cache_ttl_secs),
                        cfg.allow_http,
                    ),
                )
            })
            .collect();
        Self {
            cache: Mutex::new(HashMap::new()),
            config,
            // TLS certificate verification is enabled by default (reqwest default).
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            discovery_client: None,
        }
    }

    /// Phase 0.5 Stage D — wire a DiscoveryService client for federation
    /// directory consultation. When set, [`get_key`] will try fetching the
    /// issuer's signed entity statement via DiscoveryService before falling
    /// back to HTTPS JWKS fetch.
    ///
    /// Conservative on errors: any failure of the DiscoveryService path
    /// (network, parse, missing entity, no Ed25519 key) falls through to
    /// the existing HTTPS path. The resolver never downgrades trust based
    /// on missing data — if both Discovery AND HTTPS fail, key resolution
    /// fails and the caller rejects the JWT.
    pub fn with_discovery_client(mut self, client: Arc<DiscoveryClient>) -> Self {
        self.discovery_client = Some(client);
        self
    }

    /// Returns true if this issuer is in the trusted list.
    pub fn is_trusted(&self, issuer: &str) -> bool {
        self.config.contains_key(issuer)
    }

    /// Get the verifying key for an issuer, fetching/refreshing JWKS as needed.
    ///
    /// Uses a single `Mutex` for both cache reads and writes so that concurrent
    /// requests for the same issuer serialise: the second waiter re-checks the
    /// cache after the first finishes, finding a fresh entry rather than issuing
    /// a redundant JWKS fetch.
    pub async fn get_key(&self, issuer: &str) -> Result<VerifyingKey> {
        // Look up config before acquiring cache lock (config is immutable after new())
        let (jwks_uri_override, ttl, allow_http) = self
            .config
            .get(issuer)
            .map(|(u, t, h)| (u.clone(), *t, *h))
            .ok_or_else(|| anyhow!("Issuer not in trusted list: {}", issuer))?;

        // Acquire the cache lock once. Check first; only fetch if expired/absent.
        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(issuer) {
            if !entry.is_expired() {
                return Ok(entry.key);
            }
        }

        // Phase 0.5 Stage D — try DiscoveryService first if configured.
        // This is an optimization: federation peers can cache signed
        // entity statements that contain JWKS, letting us skip the HTTPS
        // round-trip. Conservative on errors: any failure falls through
        // to the existing HTTPS path. Never downgrades trust.
        if let Some(ref dc) = self.discovery_client {
            match self.try_get_key_from_discovery(dc, issuer).await {
                Ok(key) => {
                    cache.insert(
                        issuer.to_owned(),
                        CachedKey {
                            key,
                            fetched_at: Instant::now(),
                            ttl,
                        },
                    );
                    return Ok(key);
                }
                Err(e) => {
                    tracing::debug!(
                        issuer = %issuer,
                        error = %e,
                        "DiscoveryService federation lookup miss; falling through to HTTPS"
                    );
                }
            }
        }

        // Cache miss or expired — fetch while holding the lock so concurrent
        // callers for the same issuer block here and reuse the result.
        let jwks_uri = if let Some(uri) = jwks_uri_override {
            self.check_scheme(&uri, allow_http)?;
            uri
        } else {
            self.discover_jwks_uri(issuer, allow_http).await?
        };

        let key = self.fetch_ed25519_key(&jwks_uri).await?;

        cache.insert(
            issuer.to_owned(),
            CachedKey {
                key,
                fetched_at: Instant::now(),
                ttl,
            },
        );

        Ok(key)
    }

    /// Try fetching the Ed25519 verifying key for `issuer` via DiscoveryService.
    ///
    /// Calls `DiscoveryClient::get_entity_statement(issuer)` to retrieve a
    /// cached signed OIDF entity statement. Decodes the JWT payload
    /// (without verifying its signature — that's a Phase 0.5b follow-up
    /// tied to trust-anchor verification) and extracts the first Ed25519
    /// key from the embedded `jwks` claim.
    ///
    /// Returns `Err` on any of:
    ///   - DiscoveryService RPC failure
    ///   - No entity statement cached for the issuer
    ///   - JWT structurally invalid
    ///   - No Ed25519 key in the embedded JWKS
    ///
    /// The caller treats `Err` as "fall through to HTTPS" — never as
    /// "trust failure" — so a Discovery miss is operationally benign.
    async fn try_get_key_from_discovery(
        &self,
        dc: &DiscoveryClient,
        issuer: &str,
    ) -> Result<VerifyingKey> {
        let stmt = dc
            .get_entity_statement(issuer)
            .await
            .map_err(|e| anyhow!("DiscoveryService.getEntityStatement RPC failed: {}", e))?;

        if stmt.jwt.is_empty() {
            anyhow::bail!("DiscoveryService returned empty entity statement JWT");
        }

        // Decode JWT payload (header.payload.signature) — we extract `jwks`
        // and find an Ed25519 key. Signature verification of this entity
        // statement against trust anchors is deferred to a follow-up commit
        // (Phase 0.5 Decision 4); for now we treat DiscoveryService as a
        // trusted intra-bus delivery channel.
        let parts: Vec<&str> = stmt.jwt.split('.').collect();
        if parts.len() != 3 {
            anyhow::bail!("entity statement JWT not in 3-part compact form");
        }
        let payload_bytes = URL_SAFE_NO_PAD
            .decode(parts[1])
            .map_err(|e| anyhow!("entity statement JWT payload not base64url: {}", e))?;
        let claims: serde_json::Value = serde_json::from_slice(&payload_bytes)
            .map_err(|e| anyhow!("entity statement JWT payload not JSON: {}", e))?;

        // OIDF places the JWKS at `jwks.keys`. Some emitters use the
        // `metadata.openid_provider.jwks_uri` path instead; the simple
        // first-cut here looks for the embedded keyset.
        let keys = claims
            .get("jwks")
            .and_then(|j| j.get("keys"))
            .and_then(|k| k.as_array())
            .ok_or_else(|| anyhow!("entity statement missing jwks.keys"))?;

        for key in keys {
            if key.get("kty").and_then(|v| v.as_str()) == Some("OKP")
                && key.get("crv").and_then(|v| v.as_str()) == Some("Ed25519")
            {
                let x = key
                    .get("x")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("OKP key missing 'x'"))?;
                let raw = URL_SAFE_NO_PAD
                    .decode(x)
                    .map_err(|e| anyhow!("Ed25519 'x' not base64url: {}", e))?;
                let bytes: [u8; 32] = raw
                    .try_into()
                    .map_err(|_| anyhow!("Ed25519 key not 32 bytes"))?;
                let vk = VerifyingKey::from_bytes(&bytes)
                    .map_err(|e| anyhow!("invalid Ed25519 key bytes: {}", e))?;
                tracing::debug!(
                    issuer = %issuer,
                    "Resolved Ed25519 key from DiscoveryService entity statement"
                );
                return Ok(vk);
            }
        }

        anyhow::bail!("entity statement JWKS contains no Ed25519 key")
    }

    /// Validate that `url` uses HTTPS (or HTTP when explicitly permitted).
    ///
    /// Rejects all non-HTTP(S) schemes (e.g. `ftp://`, `file://`) regardless
    /// of the `allow_http` flag.
    fn check_scheme(&self, url: &str, allow_http: bool) -> Result<()> {
        if url.starts_with("https://") || (allow_http && url.starts_with("http://")) {
            Ok(())
        } else if url.starts_with("http://") {
            // HTTP but not explicitly allowed
            anyhow::bail!(
                "JWKS URI must use HTTPS for security (got '{}'). \
                 Set allow_http: true in trusted_issuers config to permit HTTP \
                 (development / internal networks only).",
                url
            )
        } else {
            // Non-HTTP(S) scheme (ftp://, file://, javascript:, etc.)
            anyhow::bail!(
                "JWKS URI must use HTTPS or HTTP scheme (got '{}').",
                url
            )
        }
    }

    async fn discover_jwks_uri(&self, issuer: &str, allow_http: bool) -> Result<String> {
        // Validate the issuer URL scheme before fetching AS metadata
        self.check_scheme(issuer, allow_http)?;
        let meta_url = format!("{}/.well-known/oauth-authorization-server", issuer);
        let meta: serde_json::Value = self
            .http
            .get(&meta_url)
            .send()
            .await?
            .json()
            .await?;
        let jwks_uri = meta["jwks_uri"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("No jwks_uri in AS metadata for {}", issuer))?;
        // Validate the discovered jwks_uri scheme as well
        self.check_scheme(&jwks_uri, allow_http)?;
        Ok(jwks_uri)
    }

    async fn fetch_ed25519_key(&self, jwks_uri: &str) -> Result<VerifyingKey> {
        let jwks: serde_json::Value = self
            .http
            .get(jwks_uri)
            .send()
            .await?
            .json()
            .await?;
        let keys = jwks["keys"]
            .as_array()
            .ok_or_else(|| anyhow!("JWKS missing 'keys' array at {}", jwks_uri))?;
        for key in keys {
            if key["kty"].as_str() == Some("OKP") && key["crv"].as_str() == Some("Ed25519") {
                let x = key["x"]
                    .as_str()
                    .ok_or_else(|| anyhow!("OKP key missing 'x' field"))?;
                let raw = URL_SAFE_NO_PAD.decode(x)?;
                let bytes: [u8; 32] = raw
                    .try_into()
                    .map_err(|_| anyhow!("Ed25519 key must be 32 bytes"))?;
                return VerifyingKey::from_bytes(&bytes).map_err(|e| anyhow!("{}", e));
            }
        }
        Err(anyhow!("No Ed25519 key found in JWKS at {}", jwks_uri))
    }
}

// Adapter: delegates to the inherent methods. Using fully-qualified paths
// prevents latent infinite recursion if the inherent methods are later
// removed or made private during API cleanup.
#[async_trait::async_trait]
impl FederationKeySource for FederationKeyResolver {
    fn is_trusted(&self, issuer: &str) -> bool {
        FederationKeyResolver::is_trusted(self, issuer)
    }

    async fn get_key(&self, issuer: &str) -> anyhow::Result<ed25519_dalek::VerifyingKey> {
        FederationKeyResolver::get_key(self, issuer).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_untrusted_issuer_rejected() {
        let resolver = FederationKeyResolver::new(&Default::default());
        assert!(!resolver.is_trusted("https://evil.example.com"));
        assert!(!resolver.is_trusted(""));
    }

    #[test]
    fn test_trusted_issuer_recognized() {
        let mut issuers = HashMap::new();
        issuers.insert(
            "https://trusted.example.com".to_owned(),
            crate::config::TrustedIssuerConfig {
                jwks_uri: Some("https://trusted.example.com/jwks".to_owned()),
                jwks_cache_ttl_secs: 300,
                allow_http: false,
            },
        );
        let resolver = FederationKeyResolver::new(&issuers);
        assert!(resolver.is_trusted("https://trusted.example.com"));
        assert!(!resolver.is_trusted("https://untrusted.example.com"));
    }
}
