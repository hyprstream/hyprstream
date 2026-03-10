//! JWKS fetch and cache for multi-issuer JWT verification.
//!
//! `FederationKeyResolver` resolves external issuer URLs to Ed25519 verifying
//! keys by fetching and caching JWKS (RFC 7517) documents.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::VerifyingKey;
use hyprstream_rpc::auth::FederationKeySource;

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
    /// issuer_url → cached key
    cache: RwLock<HashMap<String, CachedKey>>,
    /// issuer_url → (optional jwks_uri_override, ttl, allow_http)
    config: HashMap<String, (Option<String>, Duration, bool)>,
    http: reqwest::Client,
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
            cache: RwLock::new(HashMap::new()),
            config,
            // TLS certificate verification is enabled by default (reqwest default).
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Returns true if this issuer is in the trusted list.
    pub fn is_trusted(&self, issuer: &str) -> bool {
        self.config.contains_key(issuer)
    }

    /// Get the verifying key for an issuer, fetching/refreshing JWKS as needed.
    pub async fn get_key(&self, issuer: &str) -> Result<VerifyingKey> {
        // Check cache first (read lock)
        {
            let cache = self.cache.read().await;
            if let Some(entry) = cache.get(issuer) {
                if !entry.is_expired() {
                    return Ok(entry.key);
                }
            }
        }

        // Fetch JWKS (outside read lock)
        let (jwks_uri_override, ttl, allow_http) = self
            .config
            .get(issuer)
            .map(|(u, t, h)| (u.clone(), *t, *h))
            .ok_or_else(|| anyhow!("Issuer not in trusted list: {}", issuer))?;

        let jwks_uri = if let Some(uri) = jwks_uri_override {
            self.check_scheme(&uri, allow_http)?;
            uri
        } else {
            self.discover_jwks_uri(issuer, allow_http).await?
        };

        let key = self.fetch_ed25519_key(&jwks_uri).await?;

        // Update cache (write lock)
        {
            let mut cache = self.cache.write().await;
            cache.insert(
                issuer.to_owned(),
                CachedKey {
                    key,
                    fetched_at: Instant::now(),
                    ttl,
                },
            );
        }

        Ok(key)
    }

    /// Validate that `url` uses HTTPS unless `allow_http` is explicitly set.
    fn check_scheme(&self, url: &str, allow_http: bool) -> Result<()> {
        if !allow_http && !url.starts_with("https://") {
            anyhow::bail!(
                "JWKS URI must use HTTPS for security (got '{}'). \
                 Set allow_http: true in trusted_issuers config to permit HTTP \
                 (development / internal networks only).",
                url
            );
        }
        Ok(())
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
