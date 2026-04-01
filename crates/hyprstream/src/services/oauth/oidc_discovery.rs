//! OIDC provider discovery and metadata caching.
//!
//! Fetches and caches `/.well-known/openid-configuration` documents
//! from external OIDC providers configured in `[oauth.oidc_providers]`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tokio::sync::RwLock;

/// Cached OIDC provider metadata (from `/.well-known/openid-configuration`).
#[derive(Debug, Clone, Deserialize)]
pub struct OidcProviderMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(default)]
    pub userinfo_endpoint: Option<String>,
    pub jwks_uri: String,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
    #[serde(default)]
    pub id_token_signing_alg_values_supported: Vec<String>,
}

struct CachedMetadata {
    metadata: OidcProviderMetadata,
    fetched_at: Instant,
}

/// Fetches and caches OIDC discovery documents for configured providers.
pub struct OidcDiscoveryCache {
    cache: RwLock<HashMap<String, CachedMetadata>>,
    http: reqwest::Client,
    ttl: Duration,
}

impl OidcDiscoveryCache {
    /// Create a new cache with the given TTL and HTTP client.
    pub fn new(http: reqwest::Client, ttl: Duration) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            http,
            ttl,
        }
    }

    /// Get cached metadata or fetch from the provider's discovery endpoint.
    ///
    /// # SSRF Protection
    ///
    /// The `issuer_url` must be from config (not user input). Discovery
    /// documents are only fetched from HTTPS URLs (unless `allow_http`).
    pub async fn get_metadata(
        &self,
        issuer_url: &str,
        allow_http: bool,
    ) -> Result<OidcProviderMetadata> {
        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(issuer_url) {
                if cached.fetched_at.elapsed() < self.ttl {
                    return Ok(cached.metadata.clone());
                }
            }
        }

        // Fetch
        let discovery_url = format!("{}/.well-known/openid-configuration", issuer_url.trim_end_matches('/'));

        if !allow_http && !discovery_url.starts_with("https://") {
            return Err(anyhow!("OIDC discovery requires HTTPS (set allow_http=true for dev): {}", discovery_url));
        }

        let response = self.http
            .get(&discovery_url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .with_context(|| format!("Failed to fetch OIDC discovery from {}", discovery_url))?;

        if !response.status().is_success() {
            return Err(anyhow!("OIDC discovery returned {}: {}", response.status(), discovery_url));
        }

        // Size limit: 256KB
        let bytes = response.bytes().await?;
        if bytes.len() > 256 * 1024 {
            return Err(anyhow!("OIDC discovery response too large: {} bytes", bytes.len()));
        }

        let metadata: OidcProviderMetadata = serde_json::from_slice(&bytes)
            .with_context(|| format!("Invalid OIDC discovery JSON from {}", discovery_url))?;

        // Validate issuer matches
        if metadata.issuer != issuer_url {
            return Err(anyhow!(
                "OIDC discovery issuer mismatch: expected '{}', got '{}'",
                issuer_url, metadata.issuer
            ));
        }

        // Enforce HTTPS on derived URLs (SSRF protection)
        if !allow_http {
            for (name, url) in [
                ("token_endpoint", &metadata.token_endpoint),
                ("jwks_uri", &metadata.jwks_uri),
            ] {
                if !url.starts_with("https://") {
                    return Err(anyhow!(
                        "OIDC discovery {} must be HTTPS: {} (set allow_http=true for dev)",
                        name, url
                    ));
                }
            }
        }

        // Cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(issuer_url.to_owned(), CachedMetadata {
                metadata: metadata.clone(),
                fetched_at: Instant::now(),
            });
        }

        Ok(metadata)
    }
}

impl Default for OidcDiscoveryCache {
    fn default() -> Self {
        Self::new(
            reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            Duration::from_secs(300),
        )
    }
}

/// Shared reference for use in OAuthState.
pub type SharedDiscoveryCache = Arc<OidcDiscoveryCache>;
