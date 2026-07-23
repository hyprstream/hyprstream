//! `did:plc` → DID document resolution — federation intake (#1161, epic #1158 E1).
//!
//! The epic #1158 decision: hyprstream **mints** host-form `did:web`, but
//! **accepts federation from `did:plc` identities** — the rule governs what we
//! mint, not what we trust. This module is the intake path that makes an
//! external (e.g. Bluesky) user's DID resolvable here: it fetches the subject's
//! DID document from a PLC directory and validates it before any consumer sees
//! it.
//!
//! # Guarantees (#1161)
//!
//! - **Configurable base URL.** The PLC directory's base URL is required from
//!   [`PlcResolverConfig`] at construction; this crate deliberately provides no
//!   default directory. (did:plc was rejected as a *minting* method for its
//!   single-operator dependency; hardcoding its host here would reintroduce
//!   that dependency on the read side.)
//! - **`doc.id` validation.** The returned document must claim the DID that was
//!   asked for ([`validate_plc_document`]). A resolver that accepts a document
//!   for a different subject is a substitution oracle.
//! - **TTL cache.** The native resolver constructs [`HttpPlcFetcher`] from its
//!   one [`PlcResolverConfig`], so validated documents are cached for exactly
//!   the configured TTL and the cache is bounded to a fixed number of entries.
//! - **Fail-closed.** Resolution failure, a malformed document, an ambiguous or
//!   invalid DID, or a failed validation all return `Err`. The cache never
//!   serves an expired entry and failures are never cached — there is no
//!   cached-but-expired or partially-parsed fallback.
//! - **Egress allowlist.** The fetcher issues requests only to the configured
//!   base URL's origin; any URL whose origin is not on the allowlist is refused
//!   before a request is dispatched.
//! - **Read-only forever.** We never write to a PLC directory. This is
//!   structural: the types in this module expose no write methods — the HTTP
//!   client is private and used for GET only. Adding a write path is a
//!   protocol decision, not a refactor, and belongs in its own epic.
//!
//! # Out of scope
//!
//! Historical verification of the plc audit log is **E2 (#1174)** and must go
//! through the C1 (#1167) verifier trait. This module deliberately contains no
//! audit-log verifier: two verifiers over the same log format would diverge,
//! and one would be the lenient one.
//!
//! # Layering
//!
//! - **DID → URL** ([`did_plc_url`]) — pure derivation against a validated
//!   base URL; strict method-specific-identifier validation.
//! - **Fetch + cache** — the injected [`DidDocFetcher`] trait (shared with
//!   `did:web`) keeps the parse/validate path testable without a live network;
//!   [`HttpPlcFetcher`] is the native allowlisted implementation.
//! - **Validate** ([`validate_plc_document`]) — `doc.id` subject binding and
//!   recognized-member validation, applied before a native document enters the
//!   cache or any document is returned to its consumer.

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use serde_json::Value;
use url::Url;

use crate::did_web::{parse_did_document_no_duplicates, DidDocFetcher};

/// Maximum DID-document body we will buffer (1 MiB). A DID document is small;
/// this ceiling bounds memory against a hostile or buggy directory streaming an
/// unbounded body.
const MAX_DID_DOC_BYTES: usize = 1024 * 1024;

/// Maximum cached PLC documents. The cache key is derived from a validated DID,
/// but it remains bounded to keep memory use independent of request volume.
const MAX_CACHE_ENTRIES: usize = 256;

struct CachedDoc {
    doc: Value,
    fetched_at: std::time::Instant,
}

/// Read a response body with a hard byte ceiling before parsing it as JSON.
async fn read_capped(resp: reqwest::Response, max: usize) -> Result<Vec<u8>> {
    use futures::StreamExt;

    if let Some(len) = resp.content_length() {
        if len > max as u64 {
            bail!("DID document exceeds {max}-byte cap (Content-Length {len})");
        }
    }

    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len() + chunk.len() > max {
            bail!("DID document exceeds {max}-byte cap");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Bound the cache before insertion by removing expired entries, then the
/// oldest remaining entry when it is at capacity.
fn evict_for_insert(
    cache: &mut std::collections::HashMap<String, CachedDoc>,
    ttl: std::time::Duration,
) {
    cache.retain(|_, entry| entry.fetched_at.elapsed() < ttl);
    while cache.len() >= MAX_CACHE_ENTRIES {
        let Some(oldest) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.fetched_at)
            .map(|(url, _)| url.clone())
        else {
            break;
        };
        cache.remove(&oldest);
    }
}

// ── did:plc identifier validation ─────────────────────────────────────────────

/// The `did:plc:` method prefix.
pub const DID_PLC_PREFIX: &str = "did:plc:";

/// Length of a `did:plc` method-specific identifier: 24 lowercase base32
/// characters, per the did:plc method specification.
const PLC_MSI_LEN: usize = 24;

/// Whether `did` is a `did:plc` identifier (the federation-intake arm, #1161).
pub fn is_did_plc(did: &str) -> bool {
    did.starts_with(DID_PLC_PREFIX)
}

/// Validate a `did:plc` identifier and return its method-specific identifier.
///
/// Strict by construction (fail-closed on ambiguity): the msi must be exactly
/// 24 characters of lowercase base32 (`a-z2-7`) — nothing else can appear in a
/// conformant did:plc, and nothing else may reach URL derivation, where a
/// looser charset could smuggle path segments (`/`, `%`, `..`) into the fetch
/// URL. A DID URL fragment/query (`did:plc:…#atproto`) is stripped: resolution
/// always fetches the base document.
fn plc_msi(did: &str) -> Result<&str> {
    let did = did.split(['#', '?']).next().unwrap_or(did);
    let msi = did
        .strip_prefix(DID_PLC_PREFIX)
        .ok_or_else(|| anyhow!("not a did:plc identifier: {did}"))?;
    if msi.len() != PLC_MSI_LEN {
        bail!(
            "did:plc method-specific identifier must be {PLC_MSI_LEN} characters (got {}): {did}",
            msi.len()
        );
    }
    if !msi.bytes().all(|b| matches!(b, b'a'..=b'z' | b'2'..=b'7')) {
        bail!("did:plc method-specific identifier is not lowercase base32 (`a-z2-7`): {did}");
    }
    Ok(msi)
}

// ── resolver configuration ────────────────────────────────────────────────────

/// Configuration for a `did:plc` resolver: the PLC directory base URL and the
/// document-cache TTL.
///
/// The base URL **is** the egress boundary: every fetch is pinned beneath it
/// and the fetcher refuses any other origin. Validated at construction —
/// https-only (a directory fetched over plaintext HTTP would let any on-path
/// party substitute identity documents), no credentials (they would leak to
/// the directory operator's logs and imply an authority relationship the
/// protocol does not have), no query/fragment (they have no meaning for a
/// document root).
#[derive(Clone, Debug)]
pub struct PlcResolverConfig {
    base_url: Url,
    ttl: std::time::Duration,
}

impl PlcResolverConfig {
    /// Construct a config, validating the base URL (see the type docs).
    pub fn new(base_url: Url, ttl: std::time::Duration) -> Result<Self> {
        validate_plc_base_url(&base_url)?;
        Ok(Self { base_url, ttl })
    }

    /// The PLC directory base URL every fetch is pinned beneath.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// How long a resolved document may be served from cache.
    pub fn ttl(&self) -> std::time::Duration {
        self.ttl
    }
}

/// Validate the operator-supplied PLC directory base URL.
fn validate_plc_base_url(base_url: &Url) -> Result<()> {
    if base_url.scheme() != "https" {
        bail!("PLC directory base URL must use https:// (got {base_url})");
    }
    if base_url.host().is_none() {
        bail!("PLC directory base URL must include a host: {base_url}");
    }
    if !base_url.username().is_empty() || base_url.password().is_some() {
        bail!("PLC directory base URL must not carry credentials: {base_url}");
    }
    if base_url.query().is_some() || base_url.fragment().is_some() {
        bail!("PLC directory base URL must not carry a query or fragment: {base_url}");
    }
    Ok(())
}

// ── did:plc → URL derivation ──────────────────────────────────────────────────

/// Derive the DID-document URL for a `did:plc` identifier beneath `base_url`:
/// `{base_url}/{full-did}`.
///
/// The base URL's path is treated as a directory (a trailing `/` is added if
/// absent), so a base of `https://plc.example/mirror` resolves
/// `did:plc:{msi}` to `https://plc.example/mirror/did:plc:{msi}`.
///
/// Fail-closed: returns `Err` for a non-`did:plc` or non-conformant identifier
/// ([`plc_msi`]), and refuses — as defense in depth — any derived URL whose
/// origin is not exactly the configured base URL's origin.
pub fn did_plc_url(did: &str, base_url: &Url) -> Result<String> {
    validate_plc_base_url(base_url)?;
    plc_msi(did)?;
    let did = did.split(['#', '?']).next().unwrap_or(did);

    // Join beneath the base path: normalize it to a directory first, or
    // `Url::join` would replace the last path segment.
    let mut base = base_url.clone();
    if !base.path().ends_with('/') {
        let dir = format!("{}/", base.path());
        base.set_path(&dir);
    }
    let configured_base = base.to_string();
    base.path_segments_mut()
        .map_err(|()| {
            anyhow!("configured PLC base URL cannot accept path segments: {configured_base}")
        })?
        .push(did);
    let url = base;

    // Defense in depth: the msi charset check above already makes an
    // origin-shift impossible, but a resolution URL must provably stay on the
    // configured origin — check it rather than assume it.
    if url.origin() != base_url.origin() {
        bail!("derived did:plc URL {url} escaped the configured origin {base_url} (fail-closed)");
    }
    Ok(url.into())
}

// ── doc.id validation ─────────────────────────────────────────────────────────

/// Validate that `doc` is a DID document **for `did`** — the subject binding
/// that prevents a substitution oracle (#1161).
///
/// Checks, in order (each fail-closed):
///
/// 1. The document is a JSON object.
/// 2. It carries a string `id`.
/// 3. `id` is exactly the DID that was asked for (a fragment/query on the
///    asked DID is ignored — resolution fetches the base document).
/// 4. Recognized collection members, when present, have valid member shapes
///    and no duplicate semantic identifiers.
///
/// Exact string equality is deliberate: a did:plc has no equivalent forms
/// (lowercase base32, no path, no port), so any difference — case, padding,
/// a different subject — is a mismatch.
pub fn validate_plc_document(did: &str, doc: &Value) -> Result<()> {
    let asked = did.split(['#', '?']).next().unwrap_or(did);
    let object = doc.as_object().ok_or_else(|| {
        anyhow!("did:plc document for {asked} is not a JSON object (fail-closed)")
    })?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("did:plc document for {asked} has no string `id` (fail-closed)"))?;
    if id != asked {
        bail!(
            "did:plc document id mismatch: asked {asked}, document claims {id} — substitution oracle (fail-closed)"
        );
    }

    if let Some(also_known_as) = object.get("alsoKnownAs") {
        validate_string_collection(asked, "alsoKnownAs", also_known_as)?;
    }
    if let Some(verification_methods) = object.get("verificationMethod") {
        validate_verification_methods(asked, verification_methods)?;
    }
    if let Some(services) = object.get("service") {
        validate_services(asked, services)?;
    }
    Ok(())
}

fn validate_string_collection(did: &str, member: &str, value: &Value) -> Result<()> {
    let values = value.as_array().ok_or_else(|| {
        anyhow!("did:plc document for {did} has non-array `{member}` (fail-closed)")
    })?;
    let mut seen = std::collections::HashSet::new();
    for value in values {
        let value = value
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow!("did:plc document for {did} has invalid `{member}` member (fail-closed)")
            })?;
        if !seen.insert(value) {
            bail!("did:plc document for {did} has duplicate `{member}` member `{value}` (fail-closed)");
        }
    }
    Ok(())
}

fn required_nonempty_string<'a>(
    did: &str,
    member: &str,
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "did:plc document for {did} has `{member}` member without non-empty string `{field}` (fail-closed)"
            )
        })
}

fn verification_method_has_key_material(object: &serde_json::Map<String, Value>) -> bool {
    object
        .get("publicKeyJwk")
        .and_then(Value::as_object)
        .is_some_and(|jwk| !jwk.is_empty())
        || [
            "publicKeyMultibase",
            "publicKeyBase58",
            "publicKeyHex",
            "publicKeyPem",
            "publicKeyPgp",
        ]
        .iter()
        .any(|field| {
            object
                .get(*field)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
        })
}

fn validate_verification_methods(did: &str, value: &Value) -> Result<()> {
    let methods = value.as_array().ok_or_else(|| {
        anyhow!("did:plc document for {did} has non-array `verificationMethod` (fail-closed)")
    })?;
    let mut ids = std::collections::HashSet::new();
    for value in methods {
        let method = value.as_object().ok_or_else(|| {
            anyhow!("did:plc document for {did} has non-object `verificationMethod` member (fail-closed)")
        })?;
        let id = required_nonempty_string(did, "verificationMethod", method, "id")?;
        required_nonempty_string(did, "verificationMethod", method, "type")?;
        required_nonempty_string(did, "verificationMethod", method, "controller")?;
        if !verification_method_has_key_material(method) {
            bail!("did:plc document for {did} has `verificationMethod` member without key material (fail-closed)");
        }
        if !ids.insert(id) {
            bail!("did:plc document for {did} has duplicate `verificationMethod` id `{id}` (fail-closed)");
        }
    }
    Ok(())
}

fn is_valid_service_endpoint(value: &Value) -> bool {
    match value {
        Value::String(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty() && values.iter().all(is_valid_service_endpoint),
        _ => false,
    }
}

fn validate_services(did: &str, value: &Value) -> Result<()> {
    let services = value.as_array().ok_or_else(|| {
        anyhow!("did:plc document for {did} has non-array `service` (fail-closed)")
    })?;
    let mut ids = std::collections::HashSet::new();
    for value in services {
        let service = value.as_object().ok_or_else(|| {
            anyhow!("did:plc document for {did} has non-object `service` member (fail-closed)")
        })?;
        let id = required_nonempty_string(did, "service", service, "id")?;
        required_nonempty_string(did, "service", service, "type")?;
        if !service
            .get("serviceEndpoint")
            .is_some_and(is_valid_service_endpoint)
        {
            bail!("did:plc document for {did} has `service` member without valid `serviceEndpoint` (fail-closed)");
        }
        if !ids.insert(id) {
            bail!("did:plc document for {did} has duplicate `service` id `{id}` (fail-closed)");
        }
    }
    Ok(())
}

// ── resolver ──────────────────────────────────────────────────────────────────

/// Resolves a `did:plc` identifier to its validated DID document.
///
/// Generic over a [`DidDocFetcher`] so the parse/validate path is testable
/// with an injected fixture (no live network); the native implementation is
/// [`HttpPlcFetcher`]. This is the federation-intake artifact E2 (#1174,
/// foreign handle resolution) and E3 (#1175, exchange audience) build on.
///
/// Read-only forever: there is no write path. See the module docs.
pub struct DidPlcResolver<F: DidDocFetcher> {
    fetcher: F,
    config: PlcResolverConfig,
}

impl<F: DidDocFetcher> DidPlcResolver<F> {
    /// Construct a resolver over an injected document fetcher. This is for
    /// fixtures and alternate transports; native callers should use
    /// [`DidPlcResolver::<HttpPlcFetcher>::new`], which couples its cache TTL
    /// to this resolver's one configuration object.
    pub fn with_fetcher(fetcher: F, config: PlcResolverConfig) -> Self {
        Self { fetcher, config }
    }

    /// The resolver's directory configuration (base URL, TTL).
    pub fn config(&self) -> &PlcResolverConfig {
        &self.config
    }

    /// Resolve `did` to its DID document, validated.
    ///
    /// Derives the pinned URL beneath the configured base URL, fetches it, and
    /// enforces the subject binding ([`validate_plc_document`]). Every failure
    /// — malformed DID, fetch error, malformed document, `doc.id` mismatch —
    /// is `Err` (fail-closed).
    pub async fn resolve_document(&self, did: &str) -> Result<Value> {
        let url = did_plc_url(did, self.config.base_url())?;
        let doc = self.fetcher.fetch(&url).await?;
        validate_plc_document(did, &doc)?;
        Ok(doc)
    }
}

impl DidPlcResolver<HttpPlcFetcher> {
    /// Construct the native HTTPS resolver from one configuration object.
    ///
    /// The fetcher is created here rather than accepted from the caller, so the
    /// cache enforcing the TTL and the resolver advertising it cannot diverge.
    pub fn new(config: PlcResolverConfig) -> Result<Self> {
        let fetcher = HttpPlcFetcher::new(&config)?;
        Ok(Self { fetcher, config })
    }
}

// ── native HTTPS fetcher (egress-allowlisted, TTL-cached, read-only) ─────────

/// Native HTTPS `did:plc` document fetcher.
///
/// Same fetch+cache posture as the did:web fetcher
/// ([`crate::did_web::HttpDidDocFetcher`]) — no redirects, 10s connect/total
/// timeouts, a 1 MiB body cap ([`read_capped`]), a bounded TTL cache
/// ([`evict_for_insert`], failures never cached, expired entries never
/// served) — plus the #1161 **egress allowlist**: a request is dispatched only
/// if its URL's origin is on the allowlist, which is exactly the configured
/// PLC directory's origin. A mis-derived or attacker-influenced URL pointing
/// anywhere else (link-local metadata endpoints, loopback, a different
/// directory) is refused before any network I/O.
///
/// **Read-only forever:** the HTTP client is private and used for GET only;
/// this type exposes no write methods and no way to reach the client. That is
/// structural, not a convention — a PLC write path does not exist to be
/// misconfigured.
pub struct HttpPlcFetcher {
    http: reqwest::Client,
    cache: parking_lot::Mutex<std::collections::HashMap<String, CachedDoc>>,
    ttl: std::time::Duration,
    allowed_origins: Vec<url::Origin>,
}

impl HttpPlcFetcher {
    /// Construct a fetcher for `config`'s PLC directory.
    ///
    /// Returns `Err` if the reqwest client fails to build — propagated rather
    /// than swallowed (a `unwrap_or_default()` would yield a client WITHOUT
    /// the configured timeout/redirect policy, defeating the hardening).
    fn new(config: &PlcResolverConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            // SSRF: do NOT follow redirects. did:plc resolution is a direct
            // HTTPS GET; a redirect to http://127.0.0.1 / link-local would
            // bypass both the https-only check and the egress allowlist.
            .redirect(reqwest::redirect::Policy::none())
            // Bound how long a connect/request can hang.
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow!("failed to build did:plc HTTPS client: {e}"))?;
        Ok(Self {
            http,
            cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            ttl: config.ttl(),
            // The egress allowlist is exactly the configured directory's
            // origin: the deployment's single deliberate PLC egress point.
            allowed_origins: vec![config.base_url().origin()],
        })
    }

    /// The egress allowlist: the only origins this fetcher will issue
    /// requests to.
    pub fn allowed_origins(&self) -> &[url::Origin] {
        &self.allowed_origins
    }
}

#[async_trait]
impl DidDocFetcher for HttpPlcFetcher {
    async fn fetch(&self, url: &str) -> Result<Value> {
        // Egress allowlist + https-only, checked before any I/O.
        let parsed = Url::parse(url)
            .map_err(|e| anyhow!("did:plc document URL does not parse ({url}): {e}"))?;
        if parsed.scheme() != "https" {
            bail!("did:plc document URL must use https:// (got {url})");
        }
        if !self.allowed_origins.contains(&parsed.origin()) {
            bail!(
                "did:plc fetch refused: origin of {url} is not in the egress allowlist (fail-closed)"
            );
        }
        let asked = parsed
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .ok_or_else(|| anyhow!("did:plc document URL has no DID path segment: {url}"))?;
        plc_msi(asked)?;

        // Cache hit (not expired)? An expired entry is NEVER served — no
        // cached-but-stale fallback (#1161 fail-closed).
        {
            let cache = self.cache.lock();
            if let Some(entry) = cache.get(url) {
                if entry.fetched_at.elapsed() < self.ttl {
                    validate_plc_document(asked, &entry.doc)?;
                    return Ok(entry.doc.clone());
                }
            }
        }

        // Miss / expired — fetch over HTTPS (lock not held across await).
        let resp = self.http.get(url).send().await?;
        if !resp.status().is_success() {
            bail!(
                "did:plc directory returned non-success status {} for {url}",
                resp.status()
            );
        }
        let body = read_capped(resp, MAX_DID_DOC_BYTES).await?;
        let doc = parse_did_document_no_duplicates(&body, url)?;
        validate_plc_document(asked, &doc)?;

        {
            let mut cache = self.cache.lock();
            evict_for_insert(&mut cache, self.ttl);
            cache.insert(
                url.to_owned(),
                CachedDoc {
                    doc: doc.clone(),
                    fetched_at: std::time::Instant::now(),
                },
            );
        }
        Ok(doc)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A spec-shaped did:plc for tests: 24 lowercase base32 chars.
    const DID: &str = "did:plc:ewvi7nxzyoun6zhxrhs64oiz";

    fn configured_base() -> Url {
        Url::parse("https://plc.example").unwrap()
    }

    // ── identifier validation ────────────────────────────────────────────

    #[test]
    fn is_did_plc_prefix_check() {
        assert!(is_did_plc(DID));
        assert!(!is_did_plc("did:web:example.com"));
        assert!(!is_did_plc("did:plc")); // no trailing colon
    }

    #[test]
    fn msi_accepts_conformant_did() {
        assert_eq!(plc_msi(DID).unwrap(), "ewvi7nxzyoun6zhxrhs64oiz");
        // Fragment/query strip: resolution fetches the base document.
        assert_eq!(
            plc_msi(&format!("{DID}#atproto")).unwrap(),
            "ewvi7nxzyoun6zhxrhs64oiz"
        );
    }

    #[test]
    fn msi_rejects_non_plc_and_malformed() {
        assert!(plc_msi("did:web:example.com").is_err());
        assert!(plc_msi("did:plc:").is_err());
        // Wrong length (short / long).
        assert!(plc_msi("did:plc:abc").is_err());
        assert!(plc_msi("did:plc:ewvi7nxzyoun6zhxrhs64oizextra").is_err());
        // Uppercase is not base32-lowercase.
        assert!(plc_msi("did:plc:EWVI7NXZYOUN6ZHXRHS64OIZ").is_err());
        // `0`, `1`, `8`, `9` are not in the base32 alphabet (`a-z2-7`).
        assert!(plc_msi("did:plc:0wvi7nxzyoun6zhxrhs64oiz").is_err());
        assert!(plc_msi("did:plc:1wvi7nxzyoun6zhxrhs64oiz").is_err());
        assert!(plc_msi("did:plc:8wvi7nxzyoun6zhxrhs64oiz").is_err());
        assert!(plc_msi("did:plc:9wvi7nxzyoun6zhxrhs64oiz").is_err());
        // Path-injection characters can never reach URL derivation.
        assert!(plc_msi("did:plc:ewvi7nxzyoun6zhxrhs64oi/").is_err());
        assert!(plc_msi("did:plc:ewvi7nxzyoun6zhxrhs64oi%").is_err());
    }

    // ── config validation ────────────────────────────────────────────────

    #[test]
    fn config_accepts_https_base() {
        let cfg =
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap();
        assert_eq!(cfg.base_url().as_str(), "https://plc.example/");
        assert_eq!(cfg.ttl(), std::time::Duration::from_secs(60));
    }

    #[test]
    fn config_rejects_http_credentials_query_fragment() {
        assert!(PlcResolverConfig::new(
            Url::parse("http://plc.invalid").unwrap(),
            std::time::Duration::from_secs(60)
        )
        .is_err());
        assert!(PlcResolverConfig::new(
            Url::parse("https://user:pass@plc.invalid").unwrap(),
            std::time::Duration::from_secs(60)
        )
        .is_err());
        assert!(PlcResolverConfig::new(
            Url::parse("https://plc.invalid/?token=x").unwrap(),
            std::time::Duration::from_secs(60)
        )
        .is_err());
        assert!(PlcResolverConfig::new(
            Url::parse("https://plc.invalid/#frag").unwrap(),
            std::time::Duration::from_secs(60)
        )
        .is_err());
    }

    // ── URL derivation ───────────────────────────────────────────────────
    #[test]
    fn url_derivation_configured_base() {
        assert_eq!(
            did_plc_url(DID, &configured_base()).unwrap(),
            format!("https://plc.example/{DID}")
        );
    }

    #[test]
    fn url_derivation_custom_base_with_path() {
        // Base path is treated as a directory: the full DID is appended, not
        // substituted for the last segment.
        let base = Url::parse("https://plc.example/mirror").unwrap();
        assert_eq!(
            did_plc_url(DID, &base).unwrap(),
            "https://plc.example/mirror/did:plc:ewvi7nxzyoun6zhxrhs64oiz"
        );
        // Idempotent with an existing trailing slash.
        let base = Url::parse("https://plc.example/mirror/").unwrap();
        assert_eq!(
            did_plc_url(DID, &base).unwrap(),
            "https://plc.example/mirror/did:plc:ewvi7nxzyoun6zhxrhs64oiz"
        );
        // A port survives on the origin.
        let base = Url::parse("https://localhost:8443").unwrap();
        assert_eq!(
            did_plc_url(DID, &base).unwrap(),
            "https://localhost:8443/did:plc:ewvi7nxzyoun6zhxrhs64oiz"
        );
    }

    #[test]
    fn url_derivation_rejects_malformed_did() {
        assert!(did_plc_url("did:web:example.com", &configured_base()).is_err());
        assert!(did_plc_url("did:plc:tooshort", &configured_base()).is_err());
    }

    // ── doc.id validation ────────────────────────────────────────────────

    #[test]
    fn validate_accepts_matching_subject() {
        let doc = json!({ "id": DID, "verificationMethod": [] });
        assert!(validate_plc_document(DID, &doc).is_ok());
        // Fragment on the asked DID is ignored (base document).
        assert!(validate_plc_document(&format!("{DID}#atproto"), &doc).is_ok());
    }

    #[test]
    fn validate_rejects_substituted_subject() {
        // A document for a DIFFERENT subject returned for our DID: the
        // substitution-oracle case (#1161).
        let doc = json!({ "id": "did:plc:zxcvbnmasdfghjklqwertyu", "verificationMethod": [] });
        let err = validate_plc_document(DID, &doc).unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
    }

    #[test]
    fn validate_rejects_missing_or_nonstring_id() {
        assert!(validate_plc_document(DID, &json!({ "verificationMethod": [] })).is_err());
        assert!(validate_plc_document(DID, &json!({ "id": 42 })).is_err());
        // Not an object at all.
        assert!(validate_plc_document(DID, &json!(["not", "an", "object"])).is_err());
    }

    #[test]
    fn validate_rejects_malformed_collection_members() {
        assert!(validate_plc_document(DID, &json!({ "id": DID, "service": {} })).is_err());
        assert!(
            validate_plc_document(DID, &json!({ "id": DID, "verificationMethod": {} })).is_err()
        );
    }

    #[test]
    fn validate_rejects_malformed_or_ambiguous_collection_members() {
        assert!(validate_plc_document(DID, &json!({ "id": DID, "alsoKnownAs": [null] })).is_err());
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{ "id": "#key", "type": "Multikey" }]
            })
        )
        .is_err());
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [
                    {
                        "id": "#key",
                        "type": "Multikey",
                        "controller": DID,
                        "publicKeyMultibase": "zKey"
                    },
                    {
                        "id": "#key",
                        "type": "Multikey",
                        "controller": DID,
                        "publicKeyMultibase": "zOtherKey"
                    }
                ]
            })
        )
        .is_err());
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "service": [{ "id": "#pds", "type": "AtprotoPersonalDataServer" }]
            })
        )
        .is_err());
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "service": [
                    { "id": "#pds", "type": "AtprotoPersonalDataServer", "serviceEndpoint": "https://pds.example" },
                    { "id": "#pds", "type": "AtprotoPersonalDataServer", "serviceEndpoint": "https://other.example" }
                ]
            })
        )
        .is_err());
    }

    #[test]
    fn parser_rejects_ambiguous_duplicate_keys() {
        let duplicate_id = format!(r#"{{"id":"{DID}","id":"did:plc:zxcvbnmasdfghjklqwertyu"}}"#);
        assert!(parse_did_document_no_duplicates(
            duplicate_id.as_bytes(),
            "https://plc.example/doc"
        )
        .is_err());
        assert!(parse_did_document_no_duplicates(
            br#"{"id":"did:plc:ewvi7nxzyoun6zhxrhs64oiz","service":[{"type":"a","type":"b"}] }"#,
            "https://plc.example/doc"
        )
        .is_err());
    }

    // ── resolver over an injected fixture fetcher (no network) ───────────

    struct FixtureFetcher {
        doc: Value,
    }

    #[async_trait]
    impl DidDocFetcher for FixtureFetcher {
        async fn fetch(&self, url: &str) -> Result<Value> {
            assert_eq!(url, "https://plc.example/did:plc:ewvi7nxzyoun6zhxrhs64oiz");
            Ok(self.doc.clone())
        }
    }

    /// A fetcher that MUST NOT be called: a malformed DID must fail before
    /// any fetch.
    struct NeverFetcher;

    #[async_trait]
    impl DidDocFetcher for NeverFetcher {
        async fn fetch(&self, url: &str) -> Result<Value> {
            panic!("resolver must not fetch for a malformed DID (called with {url})");
        }
    }

    fn fixture_resolver(doc: Value) -> DidPlcResolver<FixtureFetcher> {
        DidPlcResolver::with_fetcher(
            FixtureFetcher { doc },
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap(),
        )
    }

    #[tokio::test]
    async fn resolve_document_returns_validated_doc() {
        let doc = json!({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": DID,
            "verificationMethod": [],
            "service": [],
        });
        let resolver = fixture_resolver(doc.clone());
        let got = resolver.resolve_document(DID).await.unwrap();
        assert_eq!(got, doc);
    }

    #[tokio::test]
    async fn resolve_document_accepts_captured_reference_directory_document() {
        // Captured unchanged from https://plc.directory/did:plc:ewvi7nxzyoun6zhxrhs64oiz
        // on 2026-07-22. This guards the actual reference-directory shape and
        // the full-DID request target, rather than a hand-written approximation.
        let doc = parse_did_document_no_duplicates(
            include_bytes!("fixtures/plc_directory_ewvi7nxzyoun6zhxrhs64oiz.json"),
            "https://plc.directory/did:plc:ewvi7nxzyoun6zhxrhs64oiz",
        )
        .unwrap();
        let resolver = fixture_resolver(doc.clone());
        assert_eq!(resolver.resolve_document(DID).await.unwrap(), doc);
    }

    #[tokio::test]
    async fn resolve_document_rejects_id_mismatch() {
        let doc = json!({ "id": "did:plc:zxcvbnmasdfghjklqwertyu" });
        let resolver = fixture_resolver(doc);
        let err = resolver.resolve_document(DID).await.unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
    }

    #[tokio::test]
    async fn resolve_document_rejects_missing_id() {
        let resolver = fixture_resolver(json!({ "verificationMethod": [] }));
        assert!(resolver.resolve_document(DID).await.is_err());
    }

    #[tokio::test]
    async fn resolve_document_fetch_failure_fails_closed() {
        struct FailingFetcher;
        #[async_trait]
        impl DidDocFetcher for FailingFetcher {
            async fn fetch(&self, url: &str) -> Result<Value> {
                bail!("simulated directory failure for {url}");
            }
        }
        let resolver = DidPlcResolver::with_fetcher(
            FailingFetcher,
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap(),
        );
        assert!(resolver.resolve_document(DID).await.is_err());
    }

    #[tokio::test]
    async fn resolve_document_malformed_did_never_fetches() {
        let resolver = DidPlcResolver::with_fetcher(
            NeverFetcher,
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap(),
        );
        assert!(resolver.resolve_document("did:plc:tooshort").await.is_err());
        assert!(resolver
            .resolve_document("did:web:example.com")
            .await
            .is_err());
    }

    // ── fetcher hardening (egress allowlist / https-only) ────────────────

    #[test]
    fn fetcher_allowlist_is_configured_origin() {
        let cfg = PlcResolverConfig::new(
            Url::parse("https://plc.example/mirror").unwrap(),
            std::time::Duration::from_secs(60),
        )
        .unwrap();
        let f = HttpPlcFetcher::new(&cfg).unwrap();
        assert_eq!(
            f.allowed_origins(),
            &[Url::parse("https://plc.example").unwrap().origin()]
        );
    }

    #[test]
    fn native_resolver_uses_its_single_cache_configuration() {
        let config =
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(37)).unwrap();
        let resolver = DidPlcResolver::new(config).unwrap();
        assert_eq!(resolver.fetcher.ttl, resolver.config.ttl());
        assert_eq!(
            resolver.fetcher.allowed_origins,
            vec![resolver.config.base_url().origin()]
        );
    }

    #[tokio::test]
    async fn fetcher_rejects_non_https() {
        let config =
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap();
        let f = HttpPlcFetcher::new(&config).unwrap();
        let err = f
            .fetch("http://plc.example/ewvi7nxzyoun6zhxrhs64oiz")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("https"), "{err}");
    }

    #[tokio::test]
    async fn fetcher_rejects_origin_outside_allowlist() {
        let config =
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap();
        let f = HttpPlcFetcher::new(&config).unwrap();
        // Same-shaped URL on a different origin: refused BEFORE any network
        // I/O (the egress allowlist, #1161).
        let err = f
            .fetch("https://evil.example/ewvi7nxzyoun6zhxrhs64oiz")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("egress allowlist"), "{err}");
        // A port-shifted URL is a different origin and equally refused.
        let err = f
            .fetch("https://plc.example:4443/ewvi7nxzyoun6zhxrhs64oiz")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("egress allowlist"), "{err}");
    }

    #[tokio::test]
    async fn fetcher_rejects_unparseable_url() {
        let config =
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap();
        let f = HttpPlcFetcher::new(&config).unwrap();
        assert!(f.fetch("not a url").await.is_err());
    }
}
