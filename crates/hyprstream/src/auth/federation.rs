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
pub struct FederationKeyResolver {
    /// issuer_url → cached key
    cache: RwLock<HashMap<String, CachedKey>>,
    /// issuer_url → (optional jwks_uri_override, ttl)
    config: HashMap<String, (Option<String>, Duration)>,
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
                    ),
                )
            })
            .collect();
        Self {
            cache: RwLock::new(HashMap::new()),
            config,
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
        let (jwks_uri_override, ttl) = self
            .config
            .get(issuer)
            .map(|(u, t)| (u.clone(), *t))
            .ok_or_else(|| anyhow!("Issuer not in trusted list: {}", issuer))?;

        let jwks_uri = if let Some(uri) = jwks_uri_override {
            uri
        } else {
            self.discover_jwks_uri(issuer).await?
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

    async fn discover_jwks_uri(&self, issuer: &str) -> Result<String> {
        let meta_url = format!("{}/.well-known/oauth-authorization-server", issuer);
        let meta: serde_json::Value = self
            .http
            .get(&meta_url)
            .send()
            .await?
            .json()
            .await?;
        meta["jwks_uri"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("No jwks_uri in AS metadata for {}", issuer))
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
            },
        );
        let resolver = FederationKeyResolver::new(&issuers);
        assert!(resolver.is_trusted("https://trusted.example.com"));
        assert!(!resolver.is_trusted("https://untrusted.example.com"));
    }
}
