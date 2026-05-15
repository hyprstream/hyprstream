//! JWT key source abstraction for unified key resolution.
//!
//! `JwtKeySource` provides a unified interface for resolving JWT verification keys,
//! abstracting the difference between:
//! - Regular services (trust only cluster CA)
//! - PolicyService (handles federation via ID-JAG, SPIFFE, OIDC)
//!
//! # Trust Model
//!
//! Services within a cluster all trust the same PolicyService. The CA verifying key
//! is derived from the cluster's root signing key via HKDF:
//! `derive_purpose_key(&root_key, "hyprstream-jwt-v1").verifying_key()`
//!
//! # Usage
//!
//! Services receive a `JwtKeySource` at construction time, binding them to their
//! trust anchors. Most services use `ClusterKeySource`; PolicyService uses
//! `FederatedKeySource` for cross-cluster token exchange.
//!
//! ```ignore
//! // In service factory:
//! let key_source = ctx.cluster_key_source();
//! let service = MyService::new(...).with_jwt_key_source(key_source);
//! ```

use anyhow::Result;
use ed25519_dalek::VerifyingKey;

/// Source of JWT verification keys.
///
/// Abstracts key resolution for both local (cluster CA) and federated
/// (external IdP) tokens. Services receive this at construction time,
/// binding them to their trust anchors.
#[async_trait::async_trait]
pub trait JwtKeySource: Send + Sync + 'static {
    /// Get the verifying key for a given issuer and optional kid.
    ///
    /// For local issuers (empty or matching `local_issuers()`), returns the
    /// cluster CA key. For federated issuers, fetches from JWKS or SPIFFE bundle.
    /// The `kid` hint enables key selection when multiple keys exist (e.g.
    /// rotation slots). Implementations may ignore `kid` if they only have one key.
    ///
    /// # Errors
    ///
    /// Returns error if the issuer is untrusted or key resolution fails.
    async fn get_key(&self, issuer: &str, kid: Option<&str>) -> Result<VerifyingKey>;

    /// Check if an issuer is trusted (before attempting key fetch).
    ///
    /// Returns `true` for local issuers and configured federated issuers.
    fn is_trusted(&self, issuer: &str) -> bool;

    /// List of issuer URLs considered "local" (for subject formatting).
    ///
    /// Used by `Claims::subject()` to determine whether to namespace the
    /// subject with the issuer URL (federated) or use it bare (local).
    fn local_issuers(&self) -> &[String];
}

/// Key source for regular services — trusts only the cluster CA.
///
/// This is the common case: all services in a cluster trust one PolicyService,
/// identified by its OAuth issuer URL. JWTs with empty `iss` or matching the
/// local issuer URL are verified against the CA key.
#[derive(Clone)]
pub struct ClusterKeySource {
    ca_verifying_key: VerifyingKey,
    local_issuer_url: String,
    local_issuers_vec: Vec<String>,
}

impl ClusterKeySource {
    /// Create a new cluster key source.
    ///
    /// # Arguments
    ///
    /// * `ca_verifying_key` - The cluster's CA verifying key (from PolicyService)
    /// * `local_issuer_url` - The OAuth issuer URL (e.g., "http://127.0.0.1:9080")
    pub fn new(ca_verifying_key: VerifyingKey, local_issuer_url: String) -> Self {
        let local_issuers_vec = if local_issuer_url.is_empty() {
            vec![]
        } else {
            vec![local_issuer_url.clone()]
        };
        Self {
            ca_verifying_key,
            local_issuer_url,
            local_issuers_vec,
        }
    }

    /// Get the CA verifying key.
    pub fn ca_verifying_key(&self) -> VerifyingKey {
        self.ca_verifying_key
    }
}

#[async_trait::async_trait]
impl JwtKeySource for ClusterKeySource {
    async fn get_key(&self, issuer: &str, _kid: Option<&str>) -> Result<VerifyingKey> {
        if self.is_trusted(issuer) {
            Ok(self.ca_verifying_key)
        } else {
            anyhow::bail!("Untrusted issuer: {}", issuer)
        }
    }

    fn is_trusted(&self, issuer: &str) -> bool {
        // Empty issuer is always local
        if issuer.is_empty() {
            return true;
        }
        // Match against local issuer URL
        issuer == self.local_issuer_url
    }

    fn local_issuers(&self) -> &[String] {
        &self.local_issuers_vec
    }
}

/// Key source that combines local cluster trust with federation.
///
/// Used by PolicyService to accept tokens from:
/// - Local cluster (via `ClusterKeySource`)
/// - Federated IdPs (via `FederationKeySource`)
///
/// This enables ID-JAG token exchange for cross-cluster requests.
pub struct FederatedKeySource {
    local: ClusterKeySource,
    federation: std::sync::Arc<dyn super::FederationKeySource>,
}

impl FederatedKeySource {
    /// Create a federated key source.
    ///
    /// # Arguments
    ///
    /// * `local` - The cluster's local key source
    /// * `federation` - Source for resolving federated issuer keys
    pub fn new(
        local: ClusterKeySource,
        federation: std::sync::Arc<dyn super::FederationKeySource>,
    ) -> Self {
        Self { local, federation }
    }
}

#[async_trait::async_trait]
impl JwtKeySource for FederatedKeySource {
    async fn get_key(&self, issuer: &str, kid: Option<&str>) -> Result<VerifyingKey> {
        // Try local first
        if self.local.is_trusted(issuer) {
            return self.local.get_key(issuer, kid).await;
        }
        // Fall back to federation
        if self.federation.is_trusted(issuer) {
            return self.federation.get_key(issuer).await;
        }
        anyhow::bail!("Untrusted issuer: {}", issuer)
    }

    fn is_trusted(&self, issuer: &str) -> bool {
        self.local.is_trusted(issuer) || self.federation.is_trusted(issuer)
    }

    fn local_issuers(&self) -> &[String] {
        self.local.local_issuers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn test_ca_key() -> VerifyingKey {
        let signing_key = SigningKey::from_bytes(&[1u8; 32]);
        signing_key.verifying_key()
    }

    #[tokio::test]
    async fn cluster_key_source_trusts_local_issuer() -> anyhow::Result<()> {
        let ks = ClusterKeySource::new(test_ca_key(), "http://localhost:9080".to_owned());

        assert!(ks.is_trusted("http://localhost:9080"));
        assert!(ks.is_trusted("")); // Empty issuer is always local
        assert!(!ks.is_trusted("http://other.example.com"));

        let key = ks.get_key("http://localhost:9080", None).await?;
        assert_eq!(key, test_ca_key());
        Ok(())
    }

    #[tokio::test]
    async fn cluster_key_source_rejects_untrusted() {
        let ks = ClusterKeySource::new(test_ca_key(), "http://localhost:9080".to_owned());

        let result = ks.get_key("http://evil.example.com", None).await;
        assert!(result.is_err());
        let err_msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err_msg.contains("Untrusted issuer"));
    }

    #[test]
    fn cluster_key_source_local_issuers() {
        let ks = ClusterKeySource::new(test_ca_key(), "http://localhost:9080".to_owned());
        assert_eq!(ks.local_issuers(), &["http://localhost:9080"]);

        let ks_empty = ClusterKeySource::new(test_ca_key(), String::new());
        assert!(ks_empty.local_issuers().is_empty());
    }

    #[test]
    fn trait_object_compiles() {
        let ks = ClusterKeySource::new(test_ca_key(), "http://localhost:9080".to_owned());
        let _: std::sync::Arc<dyn JwtKeySource> = std::sync::Arc::new(ks);
    }
}
