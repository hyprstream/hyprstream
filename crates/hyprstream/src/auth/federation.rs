//! JWKS fetch and cache for multi-issuer JWT verification.
//!
//! `FederationKeyResolver` resolves external issuer URLs to Ed25519 verifying
//! keys by fetching and caching JWKS (RFC 7517) documents.
//!
//! # Rotation-aware key set (#1185)
//!
//! The cache retains **every** usable Ed25519 key an issuer publishes, keyed by
//! `kid`, so two keys published simultaneously during an overlap rotation both
//! verify. A `kid` hint selects the preferred candidate; without one, the
//! caller tries each. The resolver never collapses the published set to a
//! positional singleton — that anti-pattern forecloses both overlap rotation
//! and PQ-hybrid publication (#1183).

use anyhow::{Result, anyhow};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::VerifyingKey;
use hyprstream_discovery::DiscoveryClient;
use hyprstream_rpc::auth::{FederationKey, FederationKeySource};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

/// Async function that fetches raw JWKS JSON from a URL. Defaults to a
/// reqwest GET; injectable for tests.
type JwksFetcher = Arc<
    dyn Fn(&str) -> Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

/// One Ed25519 entry from a published JWKS, with its optional `kid`.
#[derive(Clone)]
struct JwksEntry {
    kid: Option<String>,
    key: VerifyingKey,
}

/// All Ed25519 keys currently published by one issuer, plus cache bookkeeping.
struct CachedJwks {
    entries: Vec<JwksEntry>,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedJwks {
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
    /// issuer_url → cached JWKS key set.
    ///
    cache: RwLock<HashMap<String, CachedJwks>>,
    /// Short-lived cache of named keys that were absent after a refresh.
    /// This bounds attacker-controlled repeated unknown-`kid` requests.
    negative_cache: Mutex<HashMap<(String, String), Instant>>,
    /// Per-issuer single-flight guards. Different trusted issuers never block
    /// each other while one issuer refreshes its JWKS.
    fetch_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// issuer_url → (optional jwks_uri_override, ttl, allow_http)
    config: HashMap<String, (Option<String>, Duration, bool)>,
    http: reqwest::Client,
    /// Injectable JWKS fetcher (url → raw JWKS JSON). Defaults to a reqwest
    /// GET; overridable in tests so the rotation/overlap behaviour of
    /// `get_keys` can be exercised end-to-end without a network mock.
    jwks_fetcher: JwksFetcher,
    /// Phase 0.5 Stage D — optional DiscoveryService client.
    ///
    /// When configured, `get_keys` consults DiscoveryService for a cached
    /// signed entity statement before falling back to HTTPS JWKS fetch.
    /// This avoids the HTTPS round-trip for local-issuer verification and
    /// the 300s TTL becomes irrelevant for cluster-internal traffic.
    /// Falls through to HTTPS on miss/error (conservative: never downgrade
    /// trust on missing data).
    discovery_client: Option<Arc<DiscoveryClient>>,
    /// Unified federation trust gate (atproto-style). When set, every
    /// `get_keys` call additionally verifies the issuer's origin is
    /// permitted by PolicyService `federation:register` — same gate as
    /// CIMD client registration. The `trusted_issuers` config carries
    /// operational metadata (jwks_uri override, scheme, TTL); this
    /// PolicyService check carries the hot-reloadable trust decision.
    ///
    /// `None` retains the legacy posture (config-only trust) for
    /// callers that haven't wired PolicyService access yet.
    policy_client: Option<Arc<crate::services::PolicyClient>>,
}

impl FederationKeyResolver {
    pub fn new(trusted_issuers: &HashMap<String, crate::config::TrustedIssuerConfig>) -> Self {
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
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        Self {
            cache: RwLock::new(HashMap::new()),
            negative_cache: Mutex::new(HashMap::new()),
            fetch_locks: Mutex::new(HashMap::new()),
            config,
            // TLS certificate verification is enabled by default (reqwest default).
            jwks_fetcher: default_jwks_fetcher(http.clone()),
            http,
            discovery_client: None,
            policy_client: None,
        }
    }

    /// Override the JWKS fetcher (url → raw JWKS JSON). Test-only seam that
    /// lets the rotation/overlap behaviour of [`get_keys`] be exercised
    /// end-to-end without standing up an HTTPS server.
    #[cfg(test)]
    pub(crate) fn with_jwks_fetcher(mut self, fetcher: JwksFetcher) -> Self {
        self.jwks_fetcher = fetcher;
        self
    }

    /// Wire the PolicyService client so [`get_keys`] enforces the
    /// `federation:register` trust gate before any HTTPS fetch. When
    /// set, calls to `get_keys` for issuers that aren't currently
    /// permitted by PolicyService policy return Err — fail-closed,
    /// matching the CIMD client path.
    pub fn with_policy_client(mut self, client: Arc<crate::services::PolicyClient>) -> Self {
        self.policy_client = Some(client);
        self
    }

    /// Phase 0.5 Stage D — wire a DiscoveryService client for federation
    /// directory consultation. When set, [`get_keys`] will try fetching the
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

    /// Get the Ed25519 candidate verifying keys for an issuer, fetching /
    /// refreshing JWKS as needed. See the trait docs for the rotation-aware
    /// key-set contract (#1185).
    ///
    /// Uses a single `Mutex` for both cache reads and writes so that concurrent
    /// requests for the same issuer serialise: the second waiter re-checks the
    /// cache after the first finishes, finding a fresh entry rather than issuing
    /// a redundant JWKS fetch.
    pub async fn get_keys(&self, issuer: &str, kid: Option<&str>) -> Result<Vec<FederationKey>> {
        // Look up config before acquiring cache lock (config is immutable after new())
        let (jwks_uri_override, ttl, allow_http) = self
            .config
            .get(issuer)
            .map(|(u, t, h)| (u.clone(), *t, *h))
            .ok_or_else(|| anyhow!("Issuer not in trusted list: {}", issuer))?;

        // Unified federation trust gate: when PolicyService is wired,
        // also require the issuer's origin to be permitted by the
        // `federation:register` policy. Layered on top of the
        // trusted_issuers config — both must accept. Operators see
        // `trusted_issuers` as operational config (how to reach this
        // peer) and PolicyService as the hot-reloadable trust decision
        // (do we currently accept this peer). Same gate as CIMD; same
        // fail-closed semantics on RPC outage.
        if let Some(ref pc) = self.policy_client {
            use crate::services::generated::policy_client::PolicyCheck;
            // Reuse the OAuth-side RFC 6454 origin extractor: same
            // normalization (scheme + lowercase host + non-default port)
            // means a single Casbin rule covers a CIMD client at
            // app.partner.org AND a peer at hyprstream.partner.org.
            let origin = crate::services::oauth::registration::extract_origin(issuer)
                .ok_or_else(|| anyhow!("Invalid issuer URL: {issuer}"))?;
            match pc
                .check(&PolicyCheck {
                    subject: origin.clone(),
                    domain: origin.clone(),
                    resource: "federation:register".to_owned(),
                    operation: "check".to_owned(),
                })
                .await
            {
                Ok(true) => { /* permitted, fall through to fetch */ }
                Ok(false) => {
                    anyhow::bail!(
                        "peer {origin} is not permitted by policy (federation:register denied)"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        issuer = %issuer,
                        origin = %origin,
                        error = %e,
                        "PolicyService unreachable during peer federation:register check — failing closed"
                    );
                    anyhow::bail!(
                        "PolicyService unreachable; peer federation rejected (fail-closed): {e}"
                    );
                }
            }
        }

        if let Some(candidates) = self.fresh_candidates(issuer, kid).await {
            return Ok(candidates);
        }
        if self.is_negative_cached(issuer, kid).await {
            anyhow::bail!("JWKS has no Ed25519 key with requested kid (negative cached)");
        }

        // Recheck cache and the negative cache after acquiring this issuer's
        // single-flight guard. This makes one refresh serve all same-issuer
        // waiters without globally head-of-line blocking unrelated issuers.
        let fetch_lock = {
            let mut locks = self.fetch_locks.lock().await;
            Arc::clone(
                locks
                    .entry(issuer.to_owned())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        let _fetch_guard = fetch_lock.lock().await;
        if let Some(candidates) = self.fresh_candidates(issuer, kid).await {
            return Ok(candidates);
        }
        if self.is_negative_cached(issuer, kid).await {
            anyhow::bail!("JWKS has no Ed25519 key with requested kid (negative cached)");
        }

        // Phase 0.5 Stage D — try DiscoveryService first if configured.
        // This is an optimization: federation peers can cache signed
        // entity statements that contain JWKS, letting us skip the HTTPS
        // round-trip. Conservative on errors: any failure falls through
        // to the existing HTTPS path. Never downgrades trust.
        if let Some(ref dc) = self.discovery_client {
            match self.try_get_keys_from_discovery(dc, issuer).await {
                Ok(entries) if !entries.is_empty() => {
                    if has_requested_kid(&entries, kid) {
                        self.cache_entries(issuer, entries, ttl).await;
                        self.clear_negative_cache(issuer, kid).await;
                        // A selector exists only to select: do not return
                        // unrelated co-published keys as signature fallbacks.
                        return self.fresh_candidates(issuer, kid).await.ok_or_else(|| {
                            anyhow!("fresh Discovery JWKS unexpectedly missing requested key")
                        });
                    }
                    // A stale Discovery statement cannot hide an HTTPS JWKS
                    // rotation. Preserve it for kid-less callers only if the
                    // authoritative HTTPS lookup below fails.
                    self.cache_entries(issuer, entries, ttl).await;
                }
                Ok(_) => {
                    tracing::debug!(
                        issuer = %issuer,
                        "DiscoveryService federation lookup returned no Ed25519 keys; \
                         falling through to HTTPS"
                    );
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

        // Cache miss, expired, or unknown kid — fetch while holding the
        // lock so concurrent callers for the same issuer block here and
        // reuse the result.
        let jwks_uri = if let Some(uri) = jwks_uri_override {
            self.check_scheme(&uri, allow_http)?;
            uri
        } else {
            self.discover_jwks_uri(issuer, allow_http).await?
        };

        let entries = self.fetch_ed25519_keys(&jwks_uri).await?;
        if entries.is_empty() {
            anyhow::bail!("No Ed25519 key found in JWKS at {}", jwks_uri);
        }

        let found = has_requested_kid(&entries, kid);
        self.cache_entries(issuer, entries, ttl).await;
        if !found {
            self.record_negative_cache(issuer, kid).await;
            let Some(requested_kid) = kid else {
                anyhow::bail!("JWKS unexpectedly contained no usable Ed25519 key");
            };
            anyhow::bail!(
                "JWKS at {} has no Ed25519 key with kid={}",
                jwks_uri,
                requested_kid
            );
        }
        self.clear_negative_cache(issuer, kid).await;
        let cache = self.cache.read().await;
        Ok(select_candidates(&cache[issuer].entries, kid))
    }

    async fn fresh_candidates(
        &self,
        issuer: &str,
        kid: Option<&str>,
    ) -> Option<Vec<FederationKey>> {
        let cache = self.cache.read().await;
        let entry = cache.get(issuer)?;
        (!entry.is_expired() && has_requested_kid(&entry.entries, kid))
            .then(|| select_candidates(&entry.entries, kid))
    }

    async fn cache_entries(&self, issuer: &str, entries: Vec<JwksEntry>, ttl: Duration) {
        self.cache.write().await.insert(
            issuer.to_owned(),
            CachedJwks {
                entries,
                fetched_at: Instant::now(),
                ttl,
            },
        );
    }

    async fn is_negative_cached(&self, issuer: &str, kid: Option<&str>) -> bool {
        const NEGATIVE_TTL: Duration = Duration::from_secs(5);
        let Some(kid) = kid else {
            return false;
        };
        let key = (issuer.to_owned(), kid.to_owned());
        let mut misses = self.negative_cache.lock().await;
        match misses.get(&key) {
            Some(seen) if seen.elapsed() < NEGATIVE_TTL => true,
            Some(_) => {
                misses.remove(&key);
                false
            }
            None => false,
        }
    }

    async fn record_negative_cache(&self, issuer: &str, kid: Option<&str>) {
        const NEGATIVE_TTL: Duration = Duration::from_secs(5);
        const MAX_NEGATIVE_ENTRIES: usize = 1024;
        let Some(kid) = kid else {
            return;
        };
        let mut misses = self.negative_cache.lock().await;
        misses.retain(|_, seen| seen.elapsed() < NEGATIVE_TTL);
        if misses.len() >= MAX_NEGATIVE_ENTRIES {
            if let Some(key) = misses.keys().next().cloned() {
                misses.remove(&key);
            }
        }
        misses.insert((issuer.to_owned(), kid.to_owned()), Instant::now());
    }

    async fn clear_negative_cache(&self, issuer: &str, kid: Option<&str>) {
        if let Some(kid) = kid {
            self.negative_cache
                .lock()
                .await
                .remove(&(issuer.to_owned(), kid.to_owned()));
        }
    }

    /// Try fetching the Ed25519 verifying keys for `issuer` via DiscoveryService.
    ///
    /// Calls `DiscoveryClient::get_entity_statement(issuer)` to retrieve a
    /// cached signed OIDF entity statement, then performs self-validating
    /// signature verification:
    ///
    /// 1. Parse JWT header → extract `alg` (must be `EdDSA`)
    /// 2. Parse JWT payload → require `iss == sub == issuer` (self-issued check)
    /// 3. Parse JWT payload → extract `jwks.keys`
    /// 4. For each Ed25519 key in the JWKS: try verifying the JWT signature
    /// 5. Return the full candidate set once SOME embedded key verifies
    ///
    /// This self-validation establishes that the JWT was signed by a key
    /// listed in its own embedded JWKS — defeats tampering during delivery
    /// even when DiscoveryService is untrusted. A compromised DiscoveryService
    /// can't substitute fake keys because the JWT signature must still verify
    /// against one of them.
    ///
    /// Returns `Err` on any of:
    ///   - DiscoveryService RPC failure
    ///   - No entity statement cached for the issuer
    ///   - JWT structurally invalid (not 3-part, header/payload not JSON, alg ≠ EdDSA)
    ///   - `iss`/`sub` don't match the requested issuer
    ///   - No Ed25519 key in the embedded JWKS
    ///   - NONE of the embedded keys verify the JWT signature
    ///
    /// The caller treats `Err` as "fall through to HTTPS" — never as
    /// "trust failure" — so a Discovery miss is operationally benign.
    ///
    /// NOTE: this does not verify against an external trust anchor. That's
    /// Phase 0.5 Decision 4 territory (pinned trust-anchor JWKS, requiring
    /// the entity-statement signing key to match a pre-configured fingerprint).
    /// Self-validation is the strongest check we can do without trust anchors;
    /// the existing per-issuer `trusted_issuers` config provides the trust root
    /// for the *result* (since only configured issuers reach this path at all).
    async fn try_get_keys_from_discovery(
        &self,
        dc: &DiscoveryClient,
        issuer: &str,
    ) -> Result<Vec<JwksEntry>> {
        let stmt = dc
            .get_entity_statement(issuer)
            .await
            .map_err(|e| anyhow!("DiscoveryService.getEntityStatement RPC failed: {}", e))?;

        if stmt.jwt.is_empty() {
            anyhow::bail!("DiscoveryService returned empty entity statement JWT");
        }

        // Self-validation establishes which embedded key signed the statement.
        // The full published JWKS is retained (all Ed25519 entries with their
        // kids) so overlap candidates survive a Discovery lookup, matching the
        // HTTPS JWKS path's rotation semantics (#1185).
        extract_self_issued_jwks(&stmt.jwt, issuer)
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
            anyhow::bail!("JWKS URI must use HTTPS or HTTP scheme (got '{}').", url)
        }
    }

    async fn discover_jwks_uri(&self, issuer: &str, allow_http: bool) -> Result<String> {
        // Validate the issuer URL scheme before fetching AS metadata
        self.check_scheme(issuer, allow_http)?;
        let meta_url = format!("{}/.well-known/oauth-authorization-server", issuer);
        let meta: serde_json::Value = self.http.get(&meta_url).send().await?.json().await?;
        let jwks_uri = meta["jwks_uri"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("No jwks_uri in AS metadata for {}", issuer))?;
        // Validate the discovered jwks_uri scheme as well
        self.check_scheme(&jwks_uri, allow_http)?;
        Ok(jwks_uri)
    }

    /// Fetch every Ed25519 key in the JWKS at `jwks_uri`, each with its
    /// optional `kid`. All usable entries are retained — callers select by
    /// `kid` and/or try each candidate (rotation overlap, PQ-hybrid).
    async fn fetch_ed25519_keys(&self, jwks_uri: &str) -> Result<Vec<JwksEntry>> {
        let jwks = (self.jwks_fetcher)(jwks_uri).await?;
        Ok(parse_jwks_ed25519(&jwks, jwks_uri))
    }
}

/// The default JWKS fetcher: a reqwest GET returning the parsed JSON body.
/// Captured as a `JwksFetcher` so the production path and the test seam
/// share one shape (#1185).
fn default_jwks_fetcher(http: reqwest::Client) -> JwksFetcher {
    Arc::new(move |url: &str| {
        let url = url.to_owned();
        let http = http.clone();
        Box::pin(async move {
            let jwks: serde_json::Value = http.get(&url).send().await?.json().await?;
            Ok(jwks)
        })
    })
}

/// Parse every Ed25519 entry (with its `kid`) out of a JWKS document. All
/// usable entries are retained — callers select by `kid` and/or try each
/// candidate. Pure (no I/O) so rotation/overlap behaviour is unit-testable.
fn parse_jwks_ed25519(jwks: &serde_json::Value, jwks_uri: &str) -> Vec<JwksEntry> {
    let mut out = Vec::new();
    let Some(keys) = jwks["keys"].as_array() else {
        return out;
    };
    for key in keys {
        if key["kty"].as_str() != Some("OKP") || key["crv"].as_str() != Some("Ed25519") {
            continue;
        }
        let x = match key["x"].as_str() {
            Some(s) => s,
            None => continue,
        };
        let raw = match URL_SAFE_NO_PAD.decode(x) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let bytes: [u8; 32] = match raw.try_into() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let vk = match VerifyingKey::from_bytes(&bytes) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kid = key["kid"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        out.push(JwksEntry { kid, key: vk });
    }
    let _ = jwks_uri; // context only (error reporting is at call sites)
    out
}

/// A named token selects only its exact entry. A kid-less token can try every
/// compatible published key, preserving the overlap-rotation case.
fn select_candidates(entries: &[JwksEntry], kid: Option<&str>) -> Vec<FederationKey> {
    entries
        .iter()
        .filter(|entry| match kid {
            Some(requested_kid) => entry.kid.as_deref() == Some(requested_kid),
            None => true,
        })
        .map(|entry| FederationKey {
            kid: entry.kid.clone(),
            verifying_key: entry.key,
        })
        .collect()
}

fn has_requested_kid(entries: &[JwksEntry], kid: Option<&str>) -> bool {
    match kid {
        Some(requested_kid) => entries
            .iter()
            .any(|entry| entry.kid.as_deref() == Some(requested_kid)),
        None => !entries.is_empty(),
    }
}

/// Phase 0.5 Stage D — self-validating signature verification of an
/// OpenID Federation 1.0 entity statement received from DiscoveryService.
///
/// **Security model:** the JWT is self-issued (`iss == sub == issuer`)
/// and the signing key is one of the keys it publishes in its own
/// `jwks.keys` claim. Verification:
///
/// 1. JWT structure (3-part compact)
/// 2. Header alg is EdDSA
/// 3. `iss == sub == expected_issuer`
/// 4. `exp` (if present) is in the future
/// 5. SOME Ed25519 key in the embedded JWKS verifies the JWT signature
///
/// Returns the verifying key that produced the signature. For the full
/// published key set (rotation-aware), use [`extract_self_issued_jwks`].
///
/// This defeats tampering during delivery: a man-in-the-middle (or a
/// compromised intermediary like DiscoveryService) cannot substitute
/// fake keys because they would also need a corresponding private key
/// for one of the published keys to produce a valid signature.
///
/// Note this does NOT verify against an external trust anchor. The
/// caller is expected to have already established that `expected_issuer`
/// is in the trusted-issuer config; self-validation defeats tampering of
/// the delivered statement, not initial trust establishment.
#[cfg(test)]
pub(crate) fn verify_self_issued_entity_statement(
    jwt: &str,
    expected_issuer: &str,
) -> Result<VerifyingKey> {
    let (_entries, signer) = parse_self_issued_entity_statement(jwt, expected_issuer)?;
    Ok(signer)
}

/// Phase 0.5 Stage D — self-validating signature verification of an
/// OpenID Federation 1.0 entity statement received from DiscoveryService.
///
/// Returns the full published Ed25519 key set (each with its `kid`) after
/// confirming at least one of them produced the statement's signature. The
/// set is retained — not collapsed to the signer — so overlap rotation and
/// PQ-hybrid publication work through the Discovery path as they do through
/// HTTPS JWKS (#1185).
fn extract_self_issued_jwks(jwt: &str, expected_issuer: &str) -> Result<Vec<JwksEntry>> {
    let (entries, _signer) = parse_self_issued_entity_statement(jwt, expected_issuer)?;
    Ok(entries)
}

/// Shared parse + self-validation for a self-issued entity statement.
///
/// Performs the full structural/claim/exp checks and self-validates the
/// signature against the embedded JWKS, returning both the complete
/// Ed25519 candidate list (with kids) and the key that produced the
/// signature. Both entry points (`verify_self_issued_entity_statement`
/// for the single signer key, `extract_self_issued_jwks` for the rotation
/// set) run this identical check.
fn parse_self_issued_entity_statement(
    jwt: &str,
    expected_issuer: &str,
) -> Result<(Vec<JwksEntry>, VerifyingKey)> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        anyhow::bail!("entity statement JWT not in 3-part compact form");
    }

    // Header — alg must be EdDSA.
    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|e| anyhow!("entity statement JWT header not base64url: {}", e))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| anyhow!("entity statement JWT header not JSON: {}", e))?;
    let alg = header.get("alg").and_then(|v| v.as_str()).unwrap_or("");
    if alg != "EdDSA" {
        anyhow::bail!(
            "entity statement JWT alg is '{}', only EdDSA supported",
            alg
        );
    }

    // Payload — iss/sub bind to expected issuer; exp not expired.
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| anyhow!("entity statement JWT payload not base64url: {}", e))?;
    let claims: serde_json::Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| anyhow!("entity statement JWT payload not JSON: {}", e))?;

    let iss = claims.get("iss").and_then(|v| v.as_str()).unwrap_or("");
    let sub = claims.get("sub").and_then(|v| v.as_str()).unwrap_or("");
    if iss != expected_issuer {
        anyhow::bail!(
            "entity statement iss '{}' != expected issuer '{}'",
            iss,
            expected_issuer
        );
    }
    if sub != expected_issuer {
        anyhow::bail!(
            "entity statement sub '{}' != iss (must be self-issued)",
            sub
        );
    }

    if let Some(exp) = claims.get("exp").and_then(serde_json::Value::as_i64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if exp <= now {
            anyhow::bail!("entity statement expired (exp={} now={})", exp, now);
        }
    }

    // JWKS — extract every Ed25519 candidate, retaining each `kid` so the
    // caller can select by id and keep overlap candidates (#1185).
    let keys = claims
        .get("jwks")
        .and_then(|j| j.get("keys"))
        .and_then(|k| k.as_array())
        .ok_or_else(|| anyhow!("entity statement missing jwks.keys"))?;

    // Self-validating signature check: one of the embedded keys MUST verify.
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|e| anyhow!("entity statement signature not base64url: {}", e))?;
    if sig_bytes.len() != 64 {
        anyhow::bail!(
            "Ed25519 signature must be 64 bytes, got {}",
            sig_bytes.len()
        );
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let signature = ed25519_dalek::Signature::from_bytes(&sig_arr);

    let mut entries: Vec<JwksEntry> = Vec::new();
    let mut signer: Option<VerifyingKey> = None;
    for key in keys {
        if key.get("kty").and_then(|v| v.as_str()) != Some("OKP")
            || key.get("crv").and_then(|v| v.as_str()) != Some("Ed25519")
        {
            continue;
        }
        let x = match key.get("x").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let raw = match URL_SAFE_NO_PAD.decode(x) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let bytes: [u8; 32] = match raw.try_into() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let vk = match VerifyingKey::from_bytes(&bytes) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kid = key
            .get("kid")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        if signer.is_none()
            && vk
                .verify_strict(signing_input.as_bytes(), &signature)
                .is_ok()
        {
            signer = Some(vk);
        }
        entries.push(JwksEntry { kid, key: vk });
    }

    match signer {
        Some(vk) => Ok((entries, vk)),
        None => anyhow::bail!(
            "entity statement self-validation failed: no Ed25519 key in jwks verifies the signature"
        ),
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

    async fn get_keys(
        &self,
        issuer: &str,
        kid: Option<&str>,
    ) -> anyhow::Result<Vec<FederationKey>> {
        FederationKeyResolver::get_keys(self, issuer, kid).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    // ──────────────────────────────────────────────────────────────────
    // Phase 0.5 Stage D — self-validating entity statement verification
    // ──────────────────────────────────────────────────────────────────

    use ed25519_dalek::{Signer, SigningKey};
    use hyprstream_rpc::auth::{Claims, jwt};
    use rand::rngs::OsRng;

    fn b64u(bytes: &[u8]) -> String {
        URL_SAFE_NO_PAD.encode(bytes)
    }

    fn b64u_json(value: &serde_json::Value) -> String {
        b64u(&serde_json::to_vec(value).expect("json"))
    }

    /// Build a self-issued entity statement JWT signed by `signing_key`,
    /// with the provided JWKS in the payload.
    ///
    /// `jwks_keys` is the JSON array placed under `jwks.keys`; tests can
    /// pass any combination of keys (with/without the signer's key) to
    /// exercise self-validation logic.
    fn build_entity_statement(
        signing_key: &SigningKey,
        issuer: &str,
        sub: &str,
        exp: i64,
        jwks_keys: Vec<serde_json::Value>,
    ) -> String {
        let header = serde_json::json!({"alg": "EdDSA", "typ": "entity-statement+jwt"});
        let payload = serde_json::json!({
            "iss": issuer,
            "sub": sub,
            "iat": exp - 86400,
            "exp": exp,
            "jwks": {"keys": jwks_keys},
        });
        let h = b64u_json(&header);
        let p = b64u_json(&payload);
        let signing_input = format!("{}.{}", h, p);
        let sig = signing_key.sign(signing_input.as_bytes());
        format!("{}.{}", signing_input, b64u(&sig.to_bytes()))
    }

    fn jwk_of(vk: &VerifyingKey) -> serde_json::Value {
        serde_json::json!({
            "kty": "OKP",
            "crv": "Ed25519",
            "x": b64u(vk.as_bytes()),
            "use": "sig",
        })
    }

    fn future_exp() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
            + 3600
    }

    #[test]
    fn entity_stmt_self_validation_accepts_legitimate_statement() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let jwt = build_entity_statement(
            &sk,
            "https://issuer.example",
            "https://issuer.example",
            future_exp(),
            vec![jwk_of(&vk)],
        );
        let resolved = verify_self_issued_entity_statement(&jwt, "https://issuer.example")
            .expect("legitimate self-issued statement must verify");
        assert_eq!(resolved.as_bytes(), vk.as_bytes());
    }

    #[test]
    fn entity_stmt_self_validation_rejects_iss_mismatch() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let jwt = build_entity_statement(
            &sk,
            "https://other.example",
            "https://other.example",
            future_exp(),
            vec![jwk_of(&vk)],
        );
        let err = verify_self_issued_entity_statement(&jwt, "https://issuer.example")
            .expect_err("iss mismatch must reject");
        assert!(format!("{err}").contains("iss"), "got: {err}");
    }

    #[test]
    fn entity_stmt_self_validation_rejects_sub_iss_mismatch() {
        // iss matches expected but sub differs — must reject (self-issued check).
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let jwt = build_entity_statement(
            &sk,
            "https://issuer.example",
            "https://different.example",
            future_exp(),
            vec![jwk_of(&vk)],
        );
        let err = verify_self_issued_entity_statement(&jwt, "https://issuer.example")
            .expect_err("sub != iss must reject");
        assert!(format!("{err}").contains("sub"), "got: {err}");
    }

    #[test]
    fn entity_stmt_self_validation_rejects_expired() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let past_exp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
            - 60;
        let jwt = build_entity_statement(
            &sk,
            "https://issuer.example",
            "https://issuer.example",
            past_exp,
            vec![jwk_of(&vk)],
        );
        let err = verify_self_issued_entity_statement(&jwt, "https://issuer.example")
            .expect_err("expired must reject");
        assert!(format!("{err}").contains("expired"), "got: {err}");
    }

    #[test]
    fn entity_stmt_self_validation_rejects_tampered_payload() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let jwt = build_entity_statement(
            &sk,
            "https://issuer.example",
            "https://issuer.example",
            future_exp(),
            vec![jwk_of(&vk)],
        );
        // Re-encode payload with a different exp; signature stays from the
        // original. Verifier MUST reject.
        let parts: Vec<&str> = jwt.split('.').collect();
        let tampered_payload = b64u_json(&serde_json::json!({
            "iss": "https://issuer.example",
            "sub": "https://issuer.example",
            "iat": future_exp() - 7200,
            "exp": future_exp() + 99999, // tampered
            "jwks": {"keys": [jwk_of(&vk)]},
        }));
        let tampered_jwt = format!("{}.{}.{}", parts[0], tampered_payload, parts[2]);
        let err = verify_self_issued_entity_statement(&tampered_jwt, "https://issuer.example")
            .expect_err("tampered payload must reject");
        assert!(format!("{err}").contains("no Ed25519 key"), "got: {err}");
    }

    #[test]
    fn entity_stmt_self_validation_rejects_stripping_attack() {
        // Attacker presents an entity statement whose jwks contains ONLY
        // their own key (legitimate issuer's key stripped), but signs with
        // their own key. Verifier MUST reject because the signing key is
        // not the legitimate issuer's — even though the JWT internally
        // self-validates against the attacker's key. The "issuer is trusted"
        // contract is enforced by the trusted_issuers config layer above
        // this function; here we test that an attacker who substitutes a
        // statement with their own jwks-and-signing-key still cannot do so
        // unless they ALSO control the legitimate issuer URL — that is,
        // self-validation is necessary but not sufficient. This test just
        // confirms the function CAN validate against an arbitrary attacker
        // key (trust is layered outside): no key from the original issuer
        // is needed; the function trusts iss+self-validation.
        //
        // The real defense against an attacker with full control is the
        // trusted_issuers config, not self-validation. Self-validation
        // defeats tampering during delivery, which is its purpose.
        let attacker_sk = SigningKey::generate(&mut OsRng);
        let attacker_vk = attacker_sk.verifying_key();
        let jwt = build_entity_statement(
            &attacker_sk,
            "https://issuer.example",
            "https://issuer.example",
            future_exp(),
            vec![jwk_of(&attacker_vk)],
        );
        // Self-validates because attacker controls everything internally.
        // This proves the function does NOT establish trust in the issuer;
        // it only checks the statement is self-consistent.
        let resolved = verify_self_issued_entity_statement(&jwt, "https://issuer.example")
            .expect("self-consistent statement validates regardless of who signed it");
        assert_eq!(resolved.as_bytes(), attacker_vk.as_bytes());
    }

    #[test]
    fn entity_stmt_self_validation_rejects_signature_from_key_not_in_jwks() {
        // The realistic delivery-tampering scenario: attacker takes a
        // legitimate statement and substitutes the signature with their own
        // (without putting their key in the jwks). MUST reject.
        let legit_sk = SigningKey::generate(&mut OsRng);
        let legit_vk = legit_sk.verifying_key();
        let attacker_sk = SigningKey::generate(&mut OsRng);

        // Build the legitimate signed input but sign with the attacker key.
        let header = serde_json::json!({"alg": "EdDSA", "typ": "entity-statement+jwt"});
        let payload = serde_json::json!({
            "iss": "https://issuer.example",
            "sub": "https://issuer.example",
            "iat": future_exp() - 3600,
            "exp": future_exp(),
            "jwks": {"keys": [jwk_of(&legit_vk)]},
        });
        let h = b64u_json(&header);
        let p = b64u_json(&payload);
        let signing_input = format!("{}.{}", h, p);
        let bad_sig = attacker_sk.sign(signing_input.as_bytes());
        let bad_jwt = format!("{}.{}", signing_input, b64u(&bad_sig.to_bytes()));

        let err = verify_self_issued_entity_statement(&bad_jwt, "https://issuer.example")
            .expect_err("signature from key not in jwks must reject");
        assert!(format!("{err}").contains("no Ed25519 key"), "got: {err}");
    }

    #[test]
    fn entity_stmt_self_validation_rejects_non_eddsa_alg() {
        // Header with alg=RS256 must be rejected outright.
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let header = serde_json::json!({"alg": "RS256", "typ": "entity-statement+jwt"});
        let payload = serde_json::json!({
            "iss": "https://issuer.example",
            "sub": "https://issuer.example",
            "iat": future_exp() - 3600,
            "exp": future_exp(),
            "jwks": {"keys": [jwk_of(&vk)]},
        });
        let h = b64u_json(&header);
        let p = b64u_json(&payload);
        let signing_input = format!("{}.{}", h, p);
        let sig = sk.sign(signing_input.as_bytes());
        let jwt = format!("{}.{}", signing_input, b64u(&sig.to_bytes()));
        let err = verify_self_issued_entity_statement(&jwt, "https://issuer.example")
            .expect_err("non-EdDSA alg must reject");
        assert!(format!("{err}").contains("EdDSA"), "got: {err}");
    }

    #[test]
    fn entity_stmt_self_validation_rejects_malformed_jwt() {
        let err = verify_self_issued_entity_statement("not.a.jwt.with.too.many.parts", "x")
            .expect_err("malformed JWT must reject");
        assert!(format!("{err}").contains("3-part"), "got: {err}");
    }

    // ──────────────────────────────────────────────────────────────────
    // #1185 — rotation-aware JWKS key set (overlap + kid selection)
    // ──────────────────────────────────────────────────────────────────

    use std::sync::atomic::{AtomicU32, Ordering};

    fn jwk_with_kid(vk: &VerifyingKey, kid: &str) -> serde_json::Value {
        serde_json::json!({
            "kty": "OKP",
            "crv": "Ed25519",
            "x": b64u(vk.as_bytes()),
            "use": "sig",
            "alg": "EdDSA",
            "kid": kid,
        })
    }

    fn federation_token(signing_key: &SigningKey, kid: &str, sub: &str) -> String {
        let exp = future_exp();
        let claims = Claims::new(sub.to_owned(), exp - 3600, exp)
            .with_issuer("https://fed.example".to_owned());
        let header = serde_json::json!({"alg": "EdDSA", "typ": "at+jwt", "kid": kid});
        let signing_input = format!(
            "{}.{}",
            b64u_json(&header),
            b64u(&serde_json::to_vec(&claims).expect("claims json"))
        );
        let signature = signing_key.sign(signing_input.as_bytes());
        format!("{}.{}", signing_input, b64u(&signature.to_bytes()))
    }

    fn trusted_issuer(jwks_uri: &str) -> HashMap<String, crate::config::TrustedIssuerConfig> {
        let mut issuers = HashMap::new();
        issuers.insert(
            "https://fed.example".to_owned(),
            crate::config::TrustedIssuerConfig {
                jwks_uri: Some(jwks_uri.to_owned()),
                jwks_cache_ttl_secs: 300,
                allow_http: true,
            },
        );
        issuers
    }

    /// `parse_jwks_ed25519` retains EVERY published Ed25519 key with its kid,
    /// not just the first — the core anti-singleton invariant from #1183.
    #[test]
    fn parse_jwks_retains_all_ed25519_entries_with_kids() {
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);
        let jwks = serde_json::json!({
            "keys": [
                jwk_with_kid(&sk_a.verifying_key(), "kid-old"),
                jwk_with_kid(&sk_b.verifying_key(), "kid-new"),
            ]
        });
        let entries = parse_jwks_ed25519(&jwks, "https://fed.example/jwks");
        assert_eq!(entries.len(), 2, "both published keys must be retained");
        let kids: Vec<Option<&str>> = entries.iter().map(|e| e.kid.as_deref()).collect();
        assert!(kids.contains(&Some("kid-old")));
        assert!(kids.contains(&Some("kid-new")));
    }

    /// Production-boundary overlap acceptance: two published keys each verify
    /// their own named token. A named token is never allowed to fall back to a
    /// different co-published key.
    #[tokio::test]
    async fn get_keys_overlap_returns_both_and_resolves_second_by_kid() {
        let sk_old = SigningKey::generate(&mut OsRng);
        let sk_new = SigningKey::generate(&mut OsRng);
        let vk_old = sk_old.verifying_key();
        let vk_new = sk_new.verifying_key();
        let jwks = serde_json::json!({
            "keys": [
                jwk_with_kid(&vk_old, "kid-old"),
                jwk_with_kid(&vk_new, "kid-new"),
            ]
        });
        let fetcher: JwksFetcher = Arc::new(move |_url| {
            let jwks = jwks.clone();
            Box::pin(async move { Ok(jwks) })
        });
        let resolver = FederationKeyResolver::new(&trusted_issuer("https://fed.example/jwks"))
            .with_jwks_fetcher(fetcher);

        let old_token = federation_token(&sk_old, "kid-old", "old-subject");
        let new_token = federation_token(&sk_new, "kid-new", "new-subject");

        // No kid → all candidates are available to a genuinely kid-less token.
        let all = resolver
            .get_keys("https://fed.example", None)
            .await
            .expect("both keys");
        assert_eq!(all.len(), 2, "overlap: both keys must be returned");

        let new_candidates = resolver
            .get_keys("https://fed.example", Some("kid-new"))
            .await
            .expect("kid-new resolves");
        assert_eq!(new_candidates.len(), 1, "named lookup must not fall back");
        assert_eq!(new_candidates[0].verifying_key, vk_new);
        assert_eq!(
            jwt::decode_with_federation_candidates(&new_token, &new_candidates, None)
                .expect("new overlap token verifies")
                .sub,
            "new-subject"
        );

        let old_candidates = resolver
            .get_keys("https://fed.example", Some("kid-old"))
            .await
            .expect("kid-old resolves");
        assert_eq!(old_candidates.len(), 1, "named lookup must not fall back");
        assert_eq!(old_candidates[0].verifying_key, vk_old);
        assert_eq!(
            jwt::decode_with_federation_candidates(&old_token, &old_candidates, None)
                .expect("old overlap token verifies")
                .sub,
            "old-subject"
        );
    }

    /// A named `kid` that is not in the published set after a fresh fetch
    /// MUST fail closed — never silently substitute another published key
    /// (the positional-singleton anti-pattern, #1183).
    #[tokio::test]
    async fn get_keys_unknown_kid_fails_closed_without_substitution() {
        let sk = SigningKey::generate(&mut OsRng);
        let jwks = serde_json::json!({"keys": [jwk_with_kid(&sk.verifying_key(), "kid-real")]});
        let fetcher: JwksFetcher = Arc::new(move |_url| {
            let jwks = jwks.clone();
            Box::pin(async move { Ok(jwks) })
        });
        let resolver = FederationKeyResolver::new(&trusted_issuer("https://fed.example/jwks"))
            .with_jwks_fetcher(fetcher);

        let err = resolver
            .get_keys("https://fed.example", Some("kid-ghost"))
            .await
            .expect_err("unknown kid must fail closed");
        assert!(
            format!("{err}").contains("kid-ghost"),
            "error must name the missing kid: {err}"
        );
    }

    #[tokio::test]
    async fn repeated_unknown_kid_uses_one_refresh_within_negative_ttl() {
        let sk = SigningKey::generate(&mut OsRng);
        let jwks = serde_json::json!({"keys": [jwk_with_kid(&sk.verifying_key(), "kid-real")]});
        let fetch_count = Arc::new(AtomicU32::new(0));
        let count = Arc::clone(&fetch_count);
        let fetcher: JwksFetcher = Arc::new(move |_url| {
            let jwks = jwks.clone();
            let count = Arc::clone(&count);
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(jwks)
            })
        });
        let resolver = FederationKeyResolver::new(&trusted_issuer("https://fed.example/jwks"))
            .with_jwks_fetcher(fetcher);

        for _ in 0..3 {
            assert!(
                resolver
                    .get_keys("https://fed.example", Some("kid-ghost"))
                    .await
                    .is_err()
            );
        }
        assert_eq!(
            fetch_count.load(Ordering::SeqCst),
            1,
            "one miss → one refresh per negative TTL"
        );
    }

    /// Overlap rotation lifecycle: publish [old,new] (both accepted), then
    /// publish [new] only — the rotation completed and `old` is now retired
    /// past its bounded observation window. A subsequent request for `old`
    /// refetches and finds it gone, so it fails closed. This is the
    /// "retired one rejected after bounded window" overlap test.
    #[tokio::test]
    async fn get_keys_retired_key_rejected_after_rotation_completes() {
        let sk_old = SigningKey::generate(&mut OsRng);
        let sk_new = SigningKey::generate(&mut OsRng);
        let vk_old = sk_old.verifying_key();
        let vk_new = sk_new.verifying_key();

        // Overlap window: both keys published.
        let overlap = serde_json::json!({
            "keys": [
                jwk_with_kid(&vk_old, "kid-old"),
                jwk_with_kid(&vk_new, "kid-new"),
            ]
        });
        // Post-rotation: only the new key remains (old retired).
        let rotated = serde_json::json!({"keys": [jwk_with_kid(&vk_new, "kid-new")]});

        // The fetcher returns `overlap` for the first call and `rotated`
        // thereafter — modelling a JWKS that drops the old key between
        // the two observations (the bounded drain window elapsing).
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();
        let fetcher: JwksFetcher = Arc::new(move |_url| {
            let cc = cc.clone();
            let overlap = overlap.clone();
            let rotated = rotated.clone();
            Box::pin(async move {
                let n = cc.fetch_add(1, Ordering::SeqCst);
                if n == 0 { Ok(overlap) } else { Ok(rotated) }
            })
        });

        // Use a zero TTL so the second lookup forces a fresh refetch.
        let mut issuers = HashMap::new();
        issuers.insert(
            "https://fed.example".to_owned(),
            crate::config::TrustedIssuerConfig {
                jwks_uri: Some("https://fed.example/jwks".to_owned()),
                jwks_cache_ttl_secs: 0,
                allow_http: true,
            },
        );
        let resolver = FederationKeyResolver::new(&issuers).with_jwks_fetcher(fetcher);

        let old_token = federation_token(&sk_old, "kid-old", "old-subject");
        let new_token = federation_token(&sk_new, "kid-new", "new-subject");

        // During overlap both signed tokens verify at the resolver/verifier
        // boundary, not merely by inspecting returned vectors.
        let during_overlap = resolver
            .get_keys("https://fed.example", Some("kid-old"))
            .await
            .expect("old key honoured during overlap");
        assert!(jwt::decode_with_federation_candidates(&old_token, &during_overlap, None).is_ok());
        let new_during_overlap = resolver
            .get_keys("https://fed.example", Some("kid-new"))
            .await
            .expect("new key honoured during overlap");
        assert!(
            jwt::decode_with_federation_candidates(&new_token, &new_during_overlap, None).is_ok()
        );

        // After rotation drains, the old kid is gone — refetch, fail closed.
        let err = resolver
            .get_keys("https://fed.example", Some("kid-old"))
            .await
            .expect_err("retired key must be rejected after drain");
        assert!(
            format!("{err}").contains("kid-old"),
            "retired-key error must name the kid: {err}"
        );

        // The new key is unaffected.
        let after = resolver
            .get_keys("https://fed.example", Some("kid-new"))
            .await
            .expect("new key still resolves");
        assert!(jwt::decode_with_federation_candidates(&new_token, &after, None).is_ok());
    }
}
