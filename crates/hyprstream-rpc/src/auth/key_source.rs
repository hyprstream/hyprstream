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

// The ML-DSA-65 verifying-key list is shared across crates (the rotation task
// in `hyprstream` and the service factory in `hyprstream-service`) via an
// `Arc<std::sync::RwLock<..>>` contract, so this module intentionally uses
// `std::sync::RwLock` for that field rather than `parking_lot::RwLock`.
#![allow(clippy::disallowed_types)]

use anyhow::Result;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::VerifyingKey;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

/// Source of JWT verification keys.
///
/// Abstracts key resolution for both local (cluster CA) and federated
/// (external IdP) tokens. Services receive this at construction time,
/// binding them to their trust anchors.
#[async_trait::async_trait]
pub trait JwtKeySource: Send + Sync + 'static {
    /// Get the verifying key for a given issuer and optional kid.
    ///
    /// A supplied `kid` is a strict selector: implementations MUST return only
    /// the key published under that exact name or fail. It is never permission
    /// to substitute another co-published key. Kid-less multi-key verification
    /// uses [`JwtKeySource::get_keys`] instead.
    ///
    /// # Errors
    ///
    /// Returns error if the issuer is untrusted or key resolution fails.
    async fn get_key(&self, issuer: &str, kid: Option<&str>) -> Result<VerifyingKey>;

    /// Resolve **all** algorithm-compatible candidates for an issuer/kid pair.
    ///
    /// This is the safe default for verifying a kid-less JWT: when no `kid`
    /// selector is present the verifier cannot know which published key signed
    /// the token, so it MUST retain every compatible candidate and try each
    /// until one verifies — never collapse the published set to a positional
    /// singleton (`first`/`next`/`values().next()`). See #1183 / #1184.
    ///
    /// Single-key sources return a one-element vec. JWKS-backed sources
    /// override this to return every Ed25519 entry when `kid` is `None`.
    ///
    /// Callers pair this with [`crate::auth::jwt::decode_with_any_key`] to
    /// try each candidate against the JWT signature.
    async fn get_keys(&self, issuer: &str, kid: Option<&str>) -> Result<Vec<VerifyingKey>> {
        let key = self.get_key(issuer, kid).await?;
        Ok(vec![key])
    }

    /// Check if an issuer is trusted (before attempting key fetch).
    ///
    /// Returns `true` for local issuers and configured federated issuers.
    fn is_trusted(&self, issuer: &str) -> bool;

    /// List of issuer URLs considered "local" (for subject formatting).
    ///
    /// Used by `Claims::subject()` to determine whether to namespace the
    /// subject with the issuer URL (federated) or use it bare (local).
    fn local_issuers(&self) -> &[String];

    /// Get all current ML-DSA-65 verifying keys for PQ-hybrid JWT verification.
    ///
    /// Returns keys for all rotation slots (drain/active/lead) so that tokens
    /// signed by any current slot can be verified. Empty vec disables PQ verification.
    fn ml_dsa_verifying_keys(&self) -> Vec<crate::crypto::pq::MlDsaVerifyingKey> {
        vec![]
    }

    /// Return the authoritative exact-pair composite key ledger.
    fn composite_key_set(&self) -> Arc<super::CompositeKeySet> {
        super::global_composite_key_set()
    }

    /// Return the list of `alg` values bound to a given `kid` in the JWKS.
    ///
    /// This is used as a **stripping defense**: when a JWKS entry carries a
    /// composite-signature `alg` (e.g. `ML-DSA-65-Ed25519`) under a kid, the
    /// verifier MUST require that any JWT presenting that kid uses the exact
    /// same alg — preventing an attacker who controls one component key from
    /// presenting a single-algorithm JWT under the composite kid.
    ///
    /// Default implementation returns an empty vec, which means "no policy
    /// constraint" (legacy behavior). Implementations backed by JWKS should
    /// override to surface the algs they observed.
    fn kid_algs(&self, _kid: &str) -> Vec<String> {
        Vec::new()
    }
}

/// Key source for regular services — trusts only the cluster CA.
///
/// This is the common case: all services in a cluster trust one PolicyService,
/// identified by its OAuth issuer URL. JWTs with empty `iss` or matching the
/// local issuer URL are verified against the CA key.
///
/// # NOT a mesh authority (#328)
///
/// `ClusterKeySource` holds a SINGLE shared CA key and treats an empty `iss` as
/// always-trusted/local. That is correct for the in-cluster bare-`sub` token
/// plane, but it MUST NOT be the verification authority for networked **mesh
/// peers**: a single shared key cannot distinguish per-host peers, and the
/// empty-`iss` shortcut would let a networked peer inherit local trust. Mesh
/// peer identity is established by the per-host key roster (Ed25519 signer →
/// `service:inference:host-<label>`, resolved fail-closed via
/// `RequestService::resolve_key_subject`), and the empty-`iss` shortcut is
/// confined to genuine in-process callers in `verify_claims`
/// (`EnvelopeContext::is_local_caller`). For kid-routed multi-key verification
/// prefer [`JwksKeySource`], which honors the `kid` hint.
#[derive(Clone)]
pub struct ClusterKeySource {
    ca_verifying_key: VerifyingKey,
    local_issuer_url: String,
    local_issuers_vec: Vec<String>,
    ml_dsa_vks: Arc<std::sync::RwLock<Vec<crate::crypto::pq::MlDsaVerifyingKey>>>,
    composite_keys: Arc<super::CompositeKeySet>,
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
            ml_dsa_vks: Arc::new(std::sync::RwLock::new(Vec::new())),
            composite_keys: super::global_composite_key_set(),
        }
    }

    /// Set the shared ML-DSA-65 verifying key list for PQ-hybrid JWT verification.
    ///
    /// The Arc is shared with the rotation task so keys stay current.
    pub fn with_ml_dsa_verifying_keys(
        mut self,
        vks: Arc<std::sync::RwLock<Vec<crate::crypto::pq::MlDsaVerifyingKey>>>,
    ) -> Self {
        self.ml_dsa_vks = vks;
        self
    }

    /// Override the composite key ledger, primarily for isolated service instances.
    pub fn with_composite_key_set(mut self, keys: Arc<super::CompositeKeySet>) -> Self {
        self.composite_keys = keys;
        self
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
        // Empty issuer is treated as local here. This is SOUND only because the
        // empty-`iss` shortcut is gated by transport upstream in
        // `verify_claims` (#328): a networked / mesh caller presenting an
        // empty-`iss` token is rejected before this is consulted, so empty `iss`
        // reaches here only for genuine in-process / IPC callers.
        if issuer.is_empty() {
            return true;
        }
        // Match against local issuer URL
        issuer == self.local_issuer_url
    }

    fn local_issuers(&self) -> &[String] {
        &self.local_issuers_vec
    }

    fn ml_dsa_verifying_keys(&self) -> Vec<crate::crypto::pq::MlDsaVerifyingKey> {
        self.ml_dsa_vks
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn composite_key_set(&self) -> Arc<super::CompositeKeySet> {
        self.composite_keys.clone()
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
        // Fall back to federation. A named token may use only the exact named
        // key; a kid-less legacy caller retains the single-key fallback this
        // trait requires.
        if self.federation.is_trusted(issuer) {
            let candidates = self.federation.get_keys(issuer, kid).await?;
            return candidates
                .iter()
                .find(|candidate| match kid {
                    Some(named_kid) => candidate.kid.as_deref() == Some(named_kid),
                    None => true,
                })
                .map(|candidate| candidate.verifying_key)
                .ok_or_else(|| anyhow::anyhow!("No Ed25519 key for issuer {}", issuer));
        }
        anyhow::bail!("Untrusted issuer: {}", issuer)
    }

    async fn get_keys(&self, issuer: &str, kid: Option<&str>) -> Result<Vec<VerifyingKey>> {
        if self.local.is_trusted(issuer) {
            return self.local.get_keys(issuer, kid).await;
        }
        if !self.federation.is_trusted(issuer) {
            anyhow::bail!("Untrusted issuer: {}", issuer);
        }

        let candidates = self.federation.get_keys(issuer, kid).await?;
        let keys: Vec<VerifyingKey> = candidates
            .iter()
            .filter(|candidate| match kid {
                Some(named_kid) => candidate.kid.as_deref() == Some(named_kid),
                None => true,
            })
            .map(|candidate| candidate.verifying_key)
            .collect();
        if keys.is_empty() {
            anyhow::bail!("No Ed25519 key for issuer {}", issuer);
        }
        Ok(keys)
    }

    fn is_trusted(&self, issuer: &str) -> bool {
        self.local.is_trusted(issuer) || self.federation.is_trusted(issuer)
    }

    fn local_issuers(&self) -> &[String] {
        self.local.local_issuers()
    }

    fn ml_dsa_verifying_keys(&self) -> Vec<crate::crypto::pq::MlDsaVerifyingKey> {
        self.local.ml_dsa_verifying_keys()
    }

    fn composite_key_set(&self) -> Arc<super::CompositeKeySet> {
        self.local.composite_key_set()
    }

    fn kid_algs(&self, kid: &str) -> Vec<String> {
        self.local.kid_algs(kid)
    }
}

/// Async function that fetches raw JWKS JSON from a URL.
pub type JwksFetcher = Arc<
    dyn Fn(
            String,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

/// Deployment mode for JWKS-backed key resolution.
#[derive(Clone)]
pub enum JwksMode {
    /// Single-node: fetches JWKS from local `/oauth/jwks` endpoint.
    Isolated { jwks_url: String },
    /// Multi-node / cross-org: resolves issuer → JWKS URL via `IssuerResolver`.
    Federated {
        local_jwks_url: String,
        resolver: Arc<dyn IssuerResolver>,
    },
}

impl std::fmt::Debug for JwksMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Isolated { jwks_url } => f
                .debug_struct("Isolated")
                .field("jwks_url", jwks_url)
                .finish(),
            Self::Federated { local_jwks_url, .. } => f
                .debug_struct("Federated")
                .field("local_jwks_url", local_jwks_url)
                .finish(),
        }
    }
}

/// Resolves an issuer string to its JWKS endpoint URL.
#[async_trait::async_trait]
pub trait IssuerResolver: Send + Sync + 'static {
    async fn resolve(&self, issuer: &str) -> Result<String>;
}

/// Cached JWKS key entry.
struct CachedKey {
    verifying_key: VerifyingKey,
    #[allow(dead_code)]
    fetched_at: std::time::Instant,
}

/// JWKS-backed key source with kid-based resolution.
///
/// Replaces `ClusterKeySource` with standards-aligned JWKS key lookup:
/// - Caches keys by kid after fetching from the JWKS endpoint
/// - On cache miss, refetches JWKS (primary invalidation mechanism)
/// - Negative cache prevents DoS from random kid spam (5s TTL)
/// - Semaphore bounds concurrent JWKS fetches
pub struct JwksKeySource {
    mode: JwksMode,
    local_issuer_url: String,
    local_issuers_vec: Vec<String>,
    fetcher: JwksFetcher,
    cache: RwLock<HashMap<String, CachedKey>>,
    negative_cache: RwLock<HashMap<String, std::time::Instant>>,
    /// Multi-alg map: kid → all `alg` values present in the JWKS for that kid.
    /// Populated alongside `cache` during `fetch_and_cache`. Used by `kid_algs`
    /// to implement the stripping-defense policy (verifier must require all
    /// listed algs for a composite-signed kid).
    kid_alg_map: parking_lot::RwLock<HashMap<String, Vec<String>>>,
    fetch_semaphore: Semaphore,
    /// Soft TTL for cache refresh (keys older than this trigger background refresh)
    #[allow(dead_code)]
    soft_ttl: std::time::Duration,
    /// Negative cache TTL (unknown kids cached as missing for this duration)
    negative_ttl: std::time::Duration,
    ml_dsa_vks: Arc<std::sync::RwLock<Vec<crate::crypto::pq::MlDsaVerifyingKey>>>,
    /// Authoritative local CA verifying key (on-disk, from PolicyService).
    ///
    /// Lets local-issuer JWTs — notably service-to-service WIT JWTs signed by
    /// the CA over the local IPC plane — verify WITHOUT a network round-trip to
    /// `/oauth/jwks`. The HTTP JWKS endpoint is not guaranteed to be up during
    /// service startup (and #441 makes registration a hard precondition), so
    /// depending on it for local service auth is a startup-ordering fail.
    /// Consulted only for local issuers (`is_local`), and only when the JOSE
    /// `kid` matches this key's JWK thumbprint — so it never overrides a
    /// rotated/JWKS-published key, and is irrelevant to federated issuers.
    local_ca_key: Option<VerifyingKey>,
    local_ca_kid: Option<String>,
}

impl JwksKeySource {
    pub fn new(mode: JwksMode, local_issuer_url: String, fetcher: JwksFetcher) -> Self {
        let local_issuers_vec = if local_issuer_url.is_empty() {
            vec![]
        } else {
            vec![local_issuer_url.clone()]
        };
        Self {
            mode,
            local_issuer_url,
            local_issuers_vec,
            fetcher,
            cache: RwLock::new(HashMap::new()),
            negative_cache: RwLock::new(HashMap::new()),
            kid_alg_map: parking_lot::RwLock::new(HashMap::new()),
            fetch_semaphore: Semaphore::new(4),
            soft_ttl: std::time::Duration::from_secs(300),
            negative_ttl: std::time::Duration::from_secs(5),
            ml_dsa_vks: Arc::new(std::sync::RwLock::new(Vec::new())),
            local_ca_key: None,
            local_ca_kid: None,
        }
    }

    /// Set the shared ML-DSA-65 verifying key list for PQ-hybrid JWT verification.
    pub fn with_ml_dsa_verifying_keys(
        mut self,
        vks: Arc<std::sync::RwLock<Vec<crate::crypto::pq::MlDsaVerifyingKey>>>,
    ) -> Self {
        self.ml_dsa_vks = vks;
        self
    }

    /// Provide the authoritative local CA verifying key for offline resolution
    /// of local-issuer (service) JWTs. See the `local_ca_key` field docs.
    pub fn with_local_ca_key(mut self, ca_vk: VerifyingKey) -> Self {
        self.local_ca_kid = Some(crate::auth::jwt::jwk_thumbprint(
            &crate::auth::jwt::JwkThumbprintInput::Ed25519 {
                x: ca_vk.as_bytes(),
            },
        ));
        self.local_ca_key = Some(ca_vk);
        self
    }

    fn jwks_url_for_issuer(&self, issuer: &str) -> Option<String> {
        match &self.mode {
            JwksMode::Isolated { jwks_url } => {
                if self.is_local(issuer) {
                    Some(jwks_url.clone())
                } else {
                    None
                }
            }
            JwksMode::Federated { local_jwks_url, .. } => {
                if self.is_local(issuer) {
                    Some(local_jwks_url.clone())
                } else {
                    None // federated resolution is async, handled in get_key
                }
            }
        }
    }

    fn is_local(&self, issuer: &str) -> bool {
        issuer.is_empty() || issuer == self.local_issuer_url
    }

    async fn resolve_jwks_url(&self, issuer: &str) -> Result<String> {
        if let Some(url) = self.jwks_url_for_issuer(issuer) {
            return Ok(url);
        }
        match &self.mode {
            JwksMode::Federated { resolver, .. } => resolver.resolve(issuer).await,
            JwksMode::Isolated { .. } => {
                anyhow::bail!("Untrusted issuer in isolated mode: {}", issuer)
            }
        }
    }

    async fn fetch_and_cache(&self, issuer: &str) -> Result<()> {
        let _permit = self
            .fetch_semaphore
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("JWKS fetch semaphore closed"))?;

        let url = self.resolve_jwks_url(issuer).await?;
        let jwks = (self.fetcher)(url).await?;

        let keys = jwks
            .get("keys")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("JWKS response missing 'keys' array"))?;

        let now = std::time::Instant::now();
        let mut cache = self.cache.write().await;

        // Rebuild kid → algs map from this JWKS snapshot.
        let mut new_kid_algs: HashMap<String, Vec<String>> = HashMap::new();

        for key in keys {
            let kty = key.get("kty").and_then(|v| v.as_str()).unwrap_or_default();
            let crv = key.get("crv").and_then(|v| v.as_str()).unwrap_or_default();
            let kid = key.get("kid").and_then(|v| v.as_str()).map(str::to_owned);
            let alg = key.get("alg").and_then(|v| v.as_str()).map(str::to_owned);

            // Track ALL JWKS entries (Ed25519, P-256, AKP) under their kid.
            // The stripping defense fires when a kid has a composite alg, so we
            // need to record those algs even though we can't verify-decode them
            // here.
            if let (Some(kid_str), Some(alg)) = (kid.as_ref(), alg) {
                let entry = new_kid_algs.entry(kid_str.clone()).or_default();
                if !entry.contains(&alg) {
                    entry.push(alg);
                }
            }

            if kty == "OKP" && crv == "Ed25519" {
                if let Some(kid) = kid {
                    if let Some(x) = key.get("x").and_then(|v| v.as_str()) {
                        if let Ok(x_bytes) = URL_SAFE_NO_PAD.decode(x) {
                            if x_bytes.len() == 32 {
                                let mut arr = [0u8; 32];
                                arr.copy_from_slice(&x_bytes);
                                if let Ok(vk) = VerifyingKey::from_bytes(&arr) {
                                    cache.insert(
                                        kid,
                                        CachedKey {
                                            verifying_key: vk,
                                            fetched_at: now,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Publish the new kid→algs view atomically.
        *self.kid_alg_map.write() = new_kid_algs;

        // Clear negative cache entries for keys we now have
        let mut neg = self.negative_cache.write().await;
        neg.retain(|kid, _| !cache.contains_key(kid));

        Ok(())
    }

    /// Resolve a single key by `kid`, consulting the offline local-CA path,
    /// the JWKS cache, the negative cache, and (on miss) a refetch. The
    /// `kid` selector makes the choice unambiguous — contrast with the
    /// kid-less path in [`JwtKeySource::get_keys`] which must return every
    /// candidate.
    async fn resolve_by_kid(&self, issuer: &str, kid_str: &str) -> Result<VerifyingKey> {
        // Authoritative local CA key (offline): for a LOCAL issuer whose
        // `kid` matches the on-disk CA key thumbprint, resolve without any
        // network fetch. This is the service-to-service WIT JWT path; the
        // HTTP /oauth/jwks endpoint may not be up yet during startup, and
        // #441 makes service-key registration a hard precondition, so this
        // resolution must not depend on it. Never overrides a JWKS-published
        // (rotated) key because it's keyed on the exact CA thumbprint.
        if self.is_local(issuer) {
            if let (Some(ca_kid), Some(ca_key)) = (self.local_ca_kid.as_deref(), self.local_ca_key) {
                if ca_kid == kid_str {
                    return Ok(ca_key);
                }
            }
        }
        {
            let cache = self.cache.read().await;
            if let Some(entry) = cache.get(kid_str) {
                return Ok(entry.verifying_key);
            }
        }

        // Check negative cache
        {
            let neg = self.negative_cache.read().await;
            if let Some(&ts) = neg.get(kid_str) {
                if ts.elapsed() < self.negative_ttl {
                    anyhow::bail!("Key not found for kid={} (negative cached)", kid_str);
                }
            }
        }

        // On-miss refetch
        if let Err(e) = self.fetch_and_cache(issuer).await {
            tracing::warn!("JWKS fetch failed for issuer={}: {}", issuer, e);
        }

        // Retry from cache
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(kid_str) {
            return Ok(entry.verifying_key);
        }

        // Add to negative cache
        {
            let mut neg = self.negative_cache.write().await;
            neg.insert(kid_str.to_owned(), std::time::Instant::now());
        }

        anyhow::bail!("Key not found in JWKS for kid={}", kid_str)
    }

    /// Ensure the JWKS cache has been populated for `issuer` (refetch on
    /// empty). Used by the kid-less resolution paths before they read every
    /// candidate out of the cache.
    async fn ensure_cached(&self, issuer: &str) {
        if self.cache.read().await.is_empty() {
            if let Err(e) = self.fetch_and_cache(issuer).await {
                tracing::warn!("JWKS fetch failed for issuer={}: {}", issuer, e);
            }
        }
    }
}

#[async_trait::async_trait]
impl JwtKeySource for JwksKeySource {
    async fn get_key(&self, issuer: &str, kid: Option<&str>) -> Result<VerifyingKey> {
        if !self.is_trusted(issuer) {
            anyhow::bail!("Untrusted issuer: {}", issuer);
        }

        // A presented `kid` is unambiguous: resolve that exact slot.
        if let Some(kid_str) = kid {
            return self.resolve_by_kid(issuer, kid_str).await;
        }

        // No kid: refuse to collapse a multi-key published set to a positional
        // singleton. `get_keys` is the safe API for kid-less tokens — it
        // returns every Ed25519 candidate so the caller can try each. Here we
        // only succeed when the JWKS publishes exactly one key, which is the
        // unambiguous case. Two or more published keys is overlap rotation /
        // PQ-hybrid rollout territory and MUST NOT be resolved to "the first
        // one" (#1183 / #1184).
        self.ensure_cached(issuer).await;
        let cache = self.cache.read().await;
        let mut iter = cache.values();
        match (iter.next(), iter.next()) {
            (Some(first), None) => Ok(first.verifying_key),
            (None, _) => anyhow::bail!("No Ed25519 keys in JWKS for issuer {}", issuer),
            (Some(_), Some(_)) => anyhow::bail!(
                "Issuer {} publishes multiple Ed25519 keys and the JWT carries no kid; \
                 use get_keys() and try each candidate (overlap rotation / hybrid rollout)",
                issuer
            ),
        }
    }

    async fn get_keys(&self, issuer: &str, kid: Option<&str>) -> Result<Vec<VerifyingKey>> {
        if !self.is_trusted(issuer) {
            anyhow::bail!("Untrusted issuer: {}", issuer);
        }

        // A presented `kid` selects one slot.
        if let Some(kid_str) = kid {
            let vk = self.resolve_by_kid(issuer, kid_str).await?;
            return Ok(vec![vk]);
        }

        // No kid: return EVERY Ed25519 candidate so the caller can try each
        // until one verifies. Never positional selection (#1183 / #1184).
        self.ensure_cached(issuer).await;
        let cache = self.cache.read().await;
        let v: Vec<VerifyingKey> = cache.values().map(|e| e.verifying_key).collect();
        if v.is_empty() {
            anyhow::bail!("No Ed25519 keys in JWKS for issuer {}", issuer);
        }
        Ok(v)
    }

    fn is_trusted(&self, issuer: &str) -> bool {
        if self.is_local(issuer) {
            return true;
        }
        matches!(&self.mode, JwksMode::Federated { resolver, .. } if {
            // Synchronous trust check — resolver.resolve() is async,
            // so we can only check local issuers synchronously.
            // For federated mode, we optimistically trust and let get_key fail.
            true
        })
    }

    fn local_issuers(&self) -> &[String] {
        &self.local_issuers_vec
    }

    fn ml_dsa_verifying_keys(&self) -> Vec<crate::crypto::pq::MlDsaVerifyingKey> {
        self.ml_dsa_vks
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn kid_algs(&self, kid: &str) -> Vec<String> {
        self.kid_alg_map
            .read()
            .get(kid)
            .cloned()
            .unwrap_or_default()
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

    /// #441/AAA: a JwksKeySource with a local CA key resolves a local-issuer
    /// service JWT OFFLINE (no JWKS fetch) when the JOSE `kid` matches the CA
    /// thumbprint — the service-to-service auth path must not depend on the HTTP
    /// /oauth/jwks endpoint being up during startup.
    #[tokio::test]
    async fn jwks_source_resolves_local_ca_key_offline() -> anyhow::Result<()> {
        // A fetcher that always fails — proves resolution does NOT touch it.
        let fetcher: JwksFetcher = std::sync::Arc::new(|_url: String| {
            Box::pin(
                async move { anyhow::bail!("network must not be used for local CA resolution") },
            )
        });
        let ca_sk = SigningKey::from_bytes(&[7u8; 32]);
        let ca_vk = ca_sk.verifying_key();
        let ca_kid = crate::auth::jwt::kid_for_key(&ca_sk);

        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://127.0.0.1:1/oauth/jwks".to_owned(),
            },
            "http://localhost:9080".to_owned(),
            fetcher,
        )
        .with_local_ca_key(ca_vk);

        // Local issuer + matching kid -> resolved offline to the CA key.
        let key = ks.get_key("http://localhost:9080", Some(&ca_kid)).await?;
        assert_eq!(key, ca_vk);

        // Empty issuer is also local (in-process plane) -> same offline resolution.
        let key = ks.get_key("", Some(&ca_kid)).await?;
        assert_eq!(key, ca_vk);

        Ok(())
    }

    /// The offline CA fallback must NOT fire for a non-matching kid (forces a
    /// JWKS fetch, which fails here) nor for a non-local issuer — it is scoped
    /// strictly to local-issuer tokens signed by the exact CA key.
    #[tokio::test]
    async fn jwks_source_local_ca_key_scoped_to_matching_kid_and_local_issuer() {
        let fetcher: JwksFetcher = std::sync::Arc::new(|_url: String| {
            Box::pin(async move { anyhow::bail!("no network") })
        });
        let ca_vk = SigningKey::from_bytes(&[7u8; 32]).verifying_key();
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://127.0.0.1:1/oauth/jwks".to_owned(),
            },
            "http://localhost:9080".to_owned(),
            fetcher,
        )
        .with_local_ca_key(ca_vk);

        // Local issuer but a different kid -> CA fallback does NOT fire; the
        // JWKS fetch is attempted and fails -> error (not the CA key).
        assert!(
            ks.get_key("http://localhost:9080", Some("some-other-kid"))
                .await
                .is_err()
        );

        // Non-local issuer is untrusted in isolated mode regardless of kid.
        assert!(
            ks.get_key("https://evil.example.com", Some("some-other-kid"))
                .await
                .is_err()
        );
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

    fn mock_jwks_json(keys: &[&SigningKey]) -> serde_json::Value {
        let jwk_entries: Vec<serde_json::Value> = keys
            .iter()
            .map(|sk| {
                let vk = sk.verifying_key();
                let x = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(vk.as_bytes());
                let kid = crate::auth::jwt::kid_for_key(sk);
                serde_json::json!({
                    "kty": "OKP",
                    "crv": "Ed25519",
                    "use": "sig",
                    "alg": "EdDSA",
                    "kid": kid,
                    "x": x,
                })
            })
            .collect();
        serde_json::json!({ "keys": jwk_entries })
    }

    fn mock_fetcher(jwks: serde_json::Value) -> JwksFetcher {
        Arc::new(move |_url| {
            let jwks = jwks.clone();
            Box::pin(async move { Ok(jwks) })
        })
    }

    #[tokio::test]
    async fn jwks_key_source_resolves_by_kid() -> anyhow::Result<()> {
        let sk_a = SigningKey::from_bytes(&[0xAA; 32]);
        let sk_b = SigningKey::from_bytes(&[0xBB; 32]);
        let kid_a = crate::auth::jwt::kid_for_key(&sk_a);
        let kid_b = crate::auth::jwt::kid_for_key(&sk_b);

        let jwks = mock_jwks_json(&[&sk_a, &sk_b]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        let key_a = ks.get_key("http://localhost", Some(&kid_a)).await?;
        assert_eq!(key_a, sk_a.verifying_key());

        let key_b = ks.get_key("http://localhost", Some(&kid_b)).await?;
        assert_eq!(key_b, sk_b.verifying_key());
        Ok(())
    }

    #[tokio::test]
    async fn jwks_key_source_unknown_kid_fails() {
        let sk = SigningKey::from_bytes(&[0xCC; 32]);
        let jwks = mock_jwks_json(&[&sk]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        let result = ks
            .get_key("http://localhost", Some("nonexistent-kid"))
            .await;
        assert!(result.is_err());
    }

    /// #1184: with exactly one published key, `get_key(None)` is unambiguous
    /// and resolves to that key. (Two or more published keys is the overlap
    /// case, covered by `jwks_key_source_no_kid_multi_key_get_keys_tries_all`.)
    #[tokio::test]
    async fn jwks_key_source_no_kid_single_key_resolves() -> anyhow::Result<()> {
        let sk = SigningKey::from_bytes(&[0xDD; 32]);
        let jwks = mock_jwks_json(&[&sk]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        let key = ks.get_key("http://localhost", None).await?;
        assert_eq!(key, sk.verifying_key());
        Ok(())
    }

    /// I2: even if another published key actually signed the token, a declared
    /// kid selects exactly its own key. Resolution plus verification rejects
    /// kid-new/signed-by-kid-old substitution.
    #[tokio::test]
    async fn jwks_named_kid_cannot_substitute_another_published_key() -> anyhow::Result<()> {
        use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
        use ed25519_dalek::Signer as _;

        let old = SigningKey::from_bytes(&[0x31; 32]);
        let new = SigningKey::from_bytes(&[0x32; 32]);
        let new_kid = crate::auth::jwt::kid_for_key(&new);
        let jwks = mock_jwks_json(&[&old, &new]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated { jwks_url: "http://localhost/oauth/jwks".to_owned() },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        let now = chrono::Utc::now().timestamp();
        let claims = crate::auth::Claims::new("user-1".to_owned(), now, now + 300);
        let header = serde_json::json!({"alg": "EdDSA", "typ": "at+jwt", "kid": new_kid});
        let signing_input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header)?),
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims)?),
        );
        let token = format!(
            "{}.{}",
            signing_input,
            URL_SAFE_NO_PAD.encode(old.sign(signing_input.as_bytes()).to_bytes())
        );

        let selected = ks
            .get_key("http://localhost", header["kid"].as_str())
            .await?;
        assert_eq!(selected, new.verifying_key());
        assert!(crate::auth::jwt::decode_with_key(&token, &selected, None).is_err());
        Ok(())
    }

    /// #1184: `get_key(None)` MUST refuse to collapse a multi-key published
    /// set to a positional singleton — that forecloses overlap rotation and
    /// PQ-hybrid rollout. The previous implementation returned
    /// `HashMap::values().next()` (nondeterministic); reverting this hunk to
    /// "return the first candidate" makes the assertion below fail.
    #[tokio::test]
    async fn jwks_key_source_no_kid_multi_key_get_key_refuses_arbitrary() {
        let sk_a = SigningKey::from_bytes(&[0x11; 32]);
        let sk_b = SigningKey::from_bytes(&[0x22; 32]);
        let jwks = mock_jwks_json(&[&sk_a, &sk_b]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated { jwks_url: "http://localhost/oauth/jwks".to_owned() },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        let result = ks.get_key("http://localhost", None).await;
        assert!(result.is_err(), "multi-key no-kid get_key must refuse arbitrary selection");
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("multiple Ed25519 keys") && msg.contains("no kid"),
            "expected a refusal to pick arbitrarily, got: {msg}"
        );
    }

    /// #1184 overlap: two Ed25519 keys published simultaneously. `get_keys`
    /// MUST return BOTH so a kid-less verifier can try each — overlap
    /// rotation and PQ-hybrid rollout depend on this. A token signed by the
    /// non-first key still verifies because both candidates are tried.
    #[tokio::test]
    async fn jwks_key_source_no_kid_multi_key_get_keys_tries_all() -> anyhow::Result<()> {
        let sk_a = SigningKey::from_bytes(&[0x11; 32]); // "first" in JWKS order
        let sk_b = SigningKey::from_bytes(&[0x22; 32]); // "second"
        let vk_a = sk_a.verifying_key();
        let vk_b = sk_b.verifying_key();

        let jwks = mock_jwks_json(&[&sk_a, &sk_b]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated { jwks_url: "http://localhost/oauth/jwks".to_owned() },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        let candidates = ks.get_keys("http://localhost", None).await?;
        assert_eq!(candidates.len(), 2, "both published keys must be returned");
        assert!(candidates.contains(&vk_a));
        assert!(candidates.contains(&vk_b));
        Ok(())
    }

    /// #1184 overlap: a token signed by the non-first published key verifies
    /// when the caller uses `get_keys` + `decode_with_any_key`. Reverting
    /// `JwksKeySource` to return a single arbitrary candidate makes this fail
    /// (the wrong candidate is returned roughly half the time).
    #[tokio::test]
    async fn jwks_key_source_overlap_non_first_signer_verifies() -> anyhow::Result<()> {
        use crate::auth::jwt::{decode_with_any_key_lenient, encode};

        let sk_a = SigningKey::from_bytes(&[0x11; 32]);
        let sk_b = SigningKey::from_bytes(&[0x22; 32]);

        let jwks = mock_jwks_json(&[&sk_a, &sk_b]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated { jwks_url: "http://localhost/oauth/jwks".to_owned() },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        // Sign with the SECOND key — no kid in the header, so the verifier
        // must try both candidates.
        let now = chrono::Utc::now().timestamp();
        let claims = crate::auth::Claims::new("user-1".to_owned(), now, now + 300);
        let token = encode(&claims, &sk_b);
        let candidates = ks.get_keys("http://localhost", None).await?;
        let verified = decode_with_any_key_lenient(&token, &candidates, None)?;
        assert_eq!(verified.sub, "user-1");
        Ok(())
    }

    #[tokio::test]
    async fn jwks_key_source_negative_cache() {
        let sk = SigningKey::from_bytes(&[0xEE; 32]);
        let jwks = mock_jwks_json(&[&sk]);
        let fetch_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = fetch_count.clone();
        let fetcher: JwksFetcher = Arc::new(move |_url| {
            let jwks = jwks.clone();
            let count = count.clone();
            Box::pin(async move {
                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(jwks)
            })
        });

        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            fetcher,
        );

        // First miss triggers fetch
        let _ = ks.get_key("http://localhost", Some("bad-kid-1")).await;
        assert_eq!(fetch_count.load(std::sync::atomic::Ordering::Relaxed), 1);

        // Same kid within negative TTL does NOT refetch
        let _ = ks.get_key("http://localhost", Some("bad-kid-1")).await;
        assert_eq!(fetch_count.load(std::sync::atomic::Ordering::Relaxed), 1);

        // Different kid triggers a new fetch
        let _ = ks.get_key("http://localhost", Some("bad-kid-2")).await;
        assert_eq!(fetch_count.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn jwks_key_source_rejects_untrusted_issuer() {
        let sk = SigningKey::from_bytes(&[0xFF; 32]);
        let jwks = mock_jwks_json(&[&sk]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );

        let result = ks.get_key("http://evil.example.com", None).await;
        assert!(result.is_err());
    }

    /// dpop_stripping_jwks_lists_composite_alg:
    /// When the JWKS lists a kid with a composite alg (e.g.
    /// `ML-DSA-65-Ed25519`), `kid_algs` must surface that alg so the
    /// verifier's stripping-defense policy can require all components.
    #[tokio::test]
    async fn dpop_stripping_jwks_kid_algs_surface_composite() -> anyhow::Result<()> {
        let kid = "composite-test-kid";
        let jwks = serde_json::json!({
            "keys": [
                {
                    "kty": "AKP",
                    "alg": "ML-DSA-65-Ed25519",
                    "use": "sig",
                    "kid": kid,
                    "pub": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0u8; 16]),
                }
            ]
        });
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );
        // Trigger cache fill (looking up an unknown Ed25519 kid still pulls JWKS).
        let _ = ks.get_key("http://localhost", Some("unknown")).await;
        let algs = ks.kid_algs(kid);
        assert_eq!(algs, vec!["ML-DSA-65-Ed25519".to_owned()]);
        Ok(())
    }

    /// dpop_stripping_jwks_multi_alg_per_kid:
    /// When the JWKS publishes two entries sharing a kid with different algs
    /// (e.g. an Ed25519 entry and an ML-DSA-65 entry under the same kid for
    /// composite key publication), `kid_algs` must return both, so the
    /// verifier's "all-listed-required" policy can fire.
    #[tokio::test]
    async fn dpop_stripping_jwks_multi_alg_per_kid() -> anyhow::Result<()> {
        let sk = SigningKey::from_bytes(&[0x77; 32]);
        let kid = crate::auth::jwt::kid_for_key(&sk);
        let x =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sk.verifying_key().as_bytes());
        let jwks = serde_json::json!({
            "keys": [
                {"kty": "OKP", "crv": "Ed25519", "use": "sig", "alg": "EdDSA", "kid": kid.clone(), "x": x},
                {"kty": "AKP", "alg": "ML-DSA-65", "use": "sig", "kid": kid.clone(),
                 "pub": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0u8; 16])},
            ]
        });
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );
        let _ = ks.get_key("http://localhost", Some(&kid)).await?;
        let mut algs = ks.kid_algs(&kid);
        algs.sort();
        assert_eq!(algs, vec!["EdDSA".to_owned(), "ML-DSA-65".to_owned()]);
        Ok(())
    }

    /// dpop_stripping_default_impl_empty:
    /// `ClusterKeySource` doesn't track JWKS metadata, so `kid_algs` returns
    /// empty (legacy behavior, no policy constraint).
    #[test]
    fn dpop_stripping_default_impl_empty() {
        let ks = ClusterKeySource::new(test_ca_key(), "http://localhost:9080".to_owned());
        assert!(ks.kid_algs("anything").is_empty());
    }

    #[test]
    fn jwks_key_source_trait_object_compiles() {
        let sk = SigningKey::from_bytes(&[0x11; 32]);
        let jwks = mock_jwks_json(&[&sk]);
        let ks = JwksKeySource::new(
            JwksMode::Isolated {
                jwks_url: "http://localhost/oauth/jwks".to_owned(),
            },
            "http://localhost".to_owned(),
            mock_fetcher(jwks),
        );
        let _: Arc<dyn JwtKeySource> = Arc::new(ks);
    }
}
