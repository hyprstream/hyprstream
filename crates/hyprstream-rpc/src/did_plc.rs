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
//! - **Self-authenticating resolution.** The returned document must claim the
//!   DID that was asked for ([`validate_plc_document`]), and its identity-bearing
//!   fields must equal the state derived from a verified PLC audit log. Genesis
//!   binds to the DID; every operation CID, `prev` link, and ECDSA signature is
//!   verified before the directory's current document is accepted.
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
//! Full historical-time queries and independent recovery-window/fork-policy
//! auditing remain E2 (#1174) through the C1 (#1167) verifier trait. This
//! resolver nevertheless verifies the complete returned signature chain now;
//! it does not trust the directory's current document as an unsigned assertion.
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

use anyhow::{anyhow, bail, ensure, Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};
use url::Url;

use crate::did_web::{parse_did_document_no_duplicates, DidDocFetcher};

/// Maximum DID-document body we will buffer (1 MiB). A DID document is small;
/// this ceiling bounds memory against a hostile or buggy directory streaming an
/// unbounded body.
const MAX_DID_DOC_BYTES: usize = 1024 * 1024;

/// Maximum PLC per-DID audit history buffered for verification (8 MiB).
const MAX_PLC_AUDIT_BYTES: usize = 8 * 1024 * 1024;

/// Protocol maximum for one signed operation's canonical DAG-CBOR encoding.
const MAX_PLC_OPERATION_BYTES: usize = 7_500;

/// Maximum cached PLC document/audit responses. Cache keys are derived from
/// validated DID URLs, but the cache remains bounded independently of volume.
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

    // Join the DID beneath the base path with exactly one separator. The base
    // path is normalized by trimming any trailing `/` and appending a single
    // `/` + the DID: using `Url::path_segments_mut().push()` here would *keep*
    // the trailing empty segment and derive a double-slash target
    // (`https://plc.example/mirror//did:plc:`), which a PLC mirror mounted
    // below an origin path will commonly 404 on — exactly the configurable-base
    // case this resolver exists to support. (`set_path` does not percent-encode
    // the DID's `:` — verified by `url_derivation_configured_base` — and the
    // `plc_msi` charset check above makes an embedded `/` impossible.)
    let mut url = base_url.clone();
    let trimmed = url.path().trim_end_matches('/');
    url.set_path(&format!("{trimmed}/{did}"));

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
        // Each member MUST be an absolute URI per DID Core (e.g. an `at://`
        // handle or an https URL) — non-emptiness alone admits arbitrary
        // strings, which is not the contract here.
        if !is_absolute_uri(value) {
            bail!("did:plc document for {did} has `{member}` member that is not a valid URI: `{value}` (fail-closed)");
        }
        if !seen.insert(value) {
            bail!("did:plc document for {did} has duplicate `{member}` member `{value}` (fail-closed)");
        }
    }
    Ok(())
}

// ── DID / URI reference handling ──────────────────────────────────────────────
//
// DID Core verification-method and service `id`s may be either absolute DID
// URLs (`did:plc:…#key`) or relative references resolved against the subject
// DID (`#key`, `?query`). The two forms name the **same** resource, so
// deduplicating them as raw strings admits a semantically-duplicate pair — the
// validator resolves every reference against the subject before the uniqueness
// check. This is an interop surface, so classical DID Core shapes are accepted
// and unknown algorithms ignored rather than rejected (see the material
// validator).

/// Whether `s` is a syntactically valid DID per DID Core
/// (`did:method-name:method-specific-id`). `method-name` is lowercase ASCII
/// alpha/digit; `method-specific-id` allows `a-zA-Z0-9._:%` (idchar plus the
/// `:` separator and pct-encoded triples).
fn is_did(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("did:") else {
        return false;
    };
    let Some((method, msi)) = rest.split_once(':') else {
        return false;
    };
    !method.is_empty()
        && method
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && !msi.is_empty()
        && msi
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b':' | b'%'))
}

/// Whether `s` is a DID URL — a DID followed by optional path, query, and/or
/// fragment per DID Core (`did:m:m[/path][?query][#fragment]`). The authority
/// (method + method-specific-id) must be a valid DID; the suffix is unchecked
/// beyond its separator-led structure.
fn is_did_url(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("did:") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let Some((method, msi)) = authority.split_once(':') else {
        return false;
    };
    !method.is_empty()
        && method
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && !msi.is_empty()
        && msi
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b':' | b'%'))
}

/// Whether `s` is an absolute URI per RFC 3986 — a scheme
/// (`ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`) precedes the first path,
/// query, or fragment separator. Used to tell absolute DID URLs / URIs from
/// relative references (`#key`, `?query`, `./path`).
fn is_absolute_uri(s: &str) -> bool {
    let Some(colon) = s.find(':') else {
        return false;
    };
    let first_separator = s.find(['/', '?', '#']).unwrap_or(usize::MAX);
    if colon >= first_separator {
        return false;
    }
    let scheme = &s[..colon];
    !scheme.is_empty()
        && scheme
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic())
        && scheme
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
}

/// Resolve a (possibly relative) DID-URL reference against the document subject
/// DID, returning the canonical absolute DID URL used for uniqueness checks.
fn resolve_did_url(subject: &str, reference: &str) -> Result<String> {
    if is_absolute_uri(reference) {
        return Ok(reference.to_owned());
    }
    if reference.is_empty() {
        bail!("empty DID-URL reference");
    }
    let base = subject.split(['#', '?']).next().unwrap_or(subject);
    if reference.starts_with('#') || reference.starts_with('?') || reference.starts_with('/') {
        Ok(format!("{base}{reference}"))
    } else {
        // A bare-segment relative reference; append after a path separator.
        Ok(format!("{base}/{reference}"))
    }
}

/// Canonicalize a verification-method or service `id`: absolute DID URLs are
/// validated as such and used as-is; relative references are resolved against
/// the subject DID.
fn canonical_member_id(did: &str, id: &str, member: &str) -> Result<String> {
    if is_absolute_uri(id) {
        if !is_did_url(id) {
            bail!("did:plc document for {did} has `{member}` with non-DID-URL id `{id}` (fail-closed)");
        }
        Ok(id.to_owned())
    } else {
        resolve_did_url(did, id)
    }
}

// ── verification material ─────────────────────────────────────────────────────

/// Known `publicKey*` verification-material property names recognized by DID
/// Core / the Linked Data Security vocabularies. Exactly one must be present
/// per verification method (DID Core §5.1); multiple known material properties
/// is ambiguous and fail-closed.
const KNOWN_MATERIAL_FIELDS: &[&str] = &[
    "publicKeyJwk",
    "publicKeyMultibase",
    "publicKeyBase58",
    "publicKeyHex",
    "publicKeyPem",
    "publicKeyPgp",
];

/// Decode the leading unsigned-varint multicodec, returning the codec value and
/// the remaining key payload.
fn read_multicodec(bytes: &[u8]) -> Option<(u64, &[u8])> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for (i, &b) in bytes.iter().enumerate() {
        value |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((value, &bytes[i + 1..]));
        }
        shift = shift.checked_add(7)?;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// Multicodec unsigned-varint values for the public-key codecs we recognize.
mod multicodec {
    /// `ed25519-pub`.
    pub const ED25519_PUB: u64 = 0xed;
    /// `secp256k1-pub`.
    pub const SECP256K1_PUB: u64 = 0xe7;
    /// `p256-pub`.
    pub const P256_PUB: u64 = 0x1200;
    /// `ml-dsa-65-pub`.
    pub const ML_DSA_65_PUB: u64 = 0x1211;
}

/// ML-DSA-65 public-key length in bytes.
const ML_DSA_65_PUB_KEY_LEN: usize = 1952;

/// Floor for a Multikey payload whose multicodec we do not recognize. Every
/// real public key is far larger; this only rejects obvious non-keys (`zKey`)
/// while **accepting unknown algorithms** per the interop rule (this is a
/// classical, read-only resolver — an unrecognized multicodec is not an attack,
/// it is an algorithm we do not happen to admit yet).
const MIN_UNKNOWN_MULTIKEY_PAYLOAD: usize = 16;

/// Whether `s` is a structurally valid Multikey `publicKeyMultibase`: base58btc
/// multibase (`z` prefix), a leading multicodec varint, and a key payload of
/// the right length for the codec (exact for known codecs, floored for unknown
/// codecs so `zKey` is rejected but an unknown algorithm with a real key is
/// preserved).
fn is_valid_multikey(s: &str) -> bool {
    let Some(body) = s.strip_prefix('z') else {
        return false;
    };
    let Ok(decoded) = bs58::decode(body).into_vec() else {
        return false;
    };
    let Some((codec, payload)) = read_multicodec(&decoded) else {
        return false;
    };
    match codec {
        multicodec::ED25519_PUB => payload.len() == 32,
        multicodec::SECP256K1_PUB | multicodec::P256_PUB => payload.len() == 33,
        multicodec::ML_DSA_65_PUB => payload.len() == ML_DSA_65_PUB_KEY_LEN,
        _ => payload.len() >= MIN_UNKNOWN_MULTIKEY_PAYLOAD,
    }
}

/// Whether `jwk` is a syntactically valid JSON Web Key (RFC 7517/7518): a known
/// `kty` plus its required coordinate members. `{ "garbage": true }` (no
/// `kty`) is rejected.
fn is_valid_jwk(jwk: &serde_json::Map<String, Value>) -> bool {
    let Some(kty) = jwk.get("kty").and_then(Value::as_str) else {
        return false;
    };
    let has_str = |k: &str| {
        jwk.get(k)
            .and_then(Value::as_str)
            .is_some_and(|v| !v.is_empty())
    };
    match kty {
        "EC" | "OKP" => has_str("crv") && has_str("x"),
        "RSA" => has_str("n") && has_str("e"),
        "oct" => has_str("k"),
        _ => false,
    }
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

/// Validate the single declared verification-material property of a
/// verification method by its own syntax (the property name fixes the encoding,
/// independent of the method `type`).
fn validate_verification_material(
    did: &str,
    id: &str,
    method: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<()> {
    match field {
        "publicKeyMultibase" => {
            let v = method
                .get(field)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty());
            if !v.is_some_and(is_valid_multikey) {
                bail!("did:plc document for {did} has malformed Multikey `publicKeyMultibase` in `{id}` (fail-closed)");
            }
        }
        "publicKeyJwk" => {
            let Some(v) = method.get(field).and_then(Value::as_object) else {
                bail!("did:plc document for {did} has non-object `publicKeyJwk` in `{id}` (fail-closed)");
            };
            if !is_valid_jwk(v) {
                bail!("did:plc document for {did} has malformed `publicKeyJwk` in `{id}` (fail-closed)");
            }
        }
        "publicKeyBase58" | "publicKeyHex" | "publicKeyPem" | "publicKeyPgp" => {
            if method
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|v| !v.is_empty())
            {
                return Ok(());
            }
            bail!("did:plc document for {did} has empty `{field}` in `{id}` (fail-closed)");
        }
        _ => {}
    }
    Ok(())
}

fn validate_verification_methods(did: &str, value: &Value) -> Result<()> {
    let methods = value.as_array().ok_or_else(|| {
        anyhow!("did:plc document for {did} has non-array `verificationMethod` (fail-closed)")
    })?;
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for value in methods {
        let method = value.as_object().ok_or_else(|| {
            anyhow!("did:plc document for {did} has non-object `verificationMethod` member (fail-closed)")
        })?;
        let id = required_nonempty_string(did, "verificationMethod", method, "id")?;
        required_nonempty_string(did, "verificationMethod", method, "type")?;
        let controller = required_nonempty_string(did, "verificationMethod", method, "controller")?;
        if !is_did(controller) {
            bail!("did:plc document for {did} has `verificationMethod` `{id}` with non-DID controller `{controller}` (fail-closed)");
        }

        // Exactly one known verification-material property (DID Core §5.1).
        // Multiple known properties is ambiguous → fail closed. Zero known
        // properties is a material-less method → fail closed (a conforming
        // extension still carries a standard material property).
        let present: Vec<&str> = KNOWN_MATERIAL_FIELDS
            .iter()
            .copied()
            .filter(|f| method.contains_key(*f))
            .collect();
        if present.len() > 1 {
            bail!("did:plc document for {did} has `verificationMethod` `{id}` with multiple verification-material properties (fail-closed)");
        }
        let Some(field) = present.into_iter().next() else {
            bail!("did:plc document for {did} has `verificationMethod` `{id}` without a recognized verification-material property (fail-closed)");
        };
        validate_verification_material(did, id, method, field)?;

        // Canonicalize the id (relative refs resolve against the subject DID)
        // so a relative `#key` and the absolute `did:…#key` cannot both
        // appear — they name the same resource.
        let canonical = canonical_member_id(did, id, "verificationMethod")?;
        if !ids.insert(canonical) {
            bail!("did:plc document for {did} has duplicate `verificationMethod` id `{id}` (fail-closed)");
        }
    }
    Ok(())
}

fn is_valid_service_endpoint(value: &Value) -> bool {
    match value {
        // A string endpoint MUST be an absolute URI (DID Core).
        Value::String(value) => !value.is_empty() && is_absolute_uri(value),
        Value::Object(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty() && values.iter().all(is_valid_service_endpoint),
        _ => false,
    }
}

fn validate_services(did: &str, value: &Value) -> Result<()> {
    let services = value.as_array().ok_or_else(|| {
        anyhow!("did:plc document for {did} has non-array `service` (fail-closed)")
    })?;
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
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
        let canonical = canonical_member_id(did, id, "service")?;
        if !ids.insert(canonical) {
            bail!("did:plc document for {did} has duplicate `service` id `{id}` (fail-closed)");
        }
    }
    Ok(())
}

// ── PLC operation-log verification ──────────────────────────────────────────

/// Derive the audit-log URL for a validated DID beneath the configured base.
fn did_plc_audit_url(did: &str, base_url: &Url) -> Result<String> {
    Ok(format!("{}/log/audit", did_plc_url(did, base_url)?))
}

/// Canonical DAG-CBOR encoder for the JSON value domain used by PLC operations,
/// including RFC 8949 deterministic map-key ordering.
fn plc_dag_cbor(value: &Value) -> Result<Vec<u8>> {
    fn write_uint(out: &mut Vec<u8>, major: u8, value: u64) {
        let prefix = major << 5;
        if value < 24 {
            out.push(prefix | value as u8);
        } else if value <= u8::MAX as u64 {
            out.extend_from_slice(&[prefix | 24, value as u8]);
        } else if value <= u16::MAX as u64 {
            out.push(prefix | 25);
            out.extend_from_slice(&(value as u16).to_be_bytes());
        } else if value <= u32::MAX as u64 {
            out.push(prefix | 26);
            out.extend_from_slice(&(value as u32).to_be_bytes());
        } else {
            out.push(prefix | 27);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }

    fn encode(value: &Value, out: &mut Vec<u8>, depth: usize) -> Result<()> {
        ensure!(depth <= 128, "PLC operation exceeds DAG-CBOR nesting limit");
        match value {
            Value::Null => out.push(0xf6),
            Value::Bool(false) => out.push(0xf4),
            Value::Bool(true) => out.push(0xf5),
            Value::String(text) => {
                write_uint(out, 3, text.len() as u64);
                out.extend_from_slice(text.as_bytes());
            }
            Value::Array(values) => {
                write_uint(out, 4, values.len() as u64);
                for value in values {
                    encode(value, out, depth + 1)?;
                }
            }
            Value::Object(object) => {
                let mut entries: Vec<_> = object.iter().collect();
                // RFC 8949 deterministic ordering: shorter encoded text keys
                // first, then bytewise lexical order for equal lengths.
                entries.sort_unstable_by(|(left, _), (right, _)| {
                    left.len()
                        .cmp(&right.len())
                        .then_with(|| left.as_bytes().cmp(right.as_bytes()))
                });
                write_uint(out, 5, entries.len() as u64);
                for (key, value) in entries {
                    write_uint(out, 3, key.len() as u64);
                    out.extend_from_slice(key.as_bytes());
                    encode(value, out, depth + 1)?;
                }
            }
            Value::Number(_) => bail!("PLC operations cannot contain JSON numbers"),
        }
        Ok(())
    }

    let mut encoded = Vec::new();
    encode(value, &mut encoded, 0)?;
    ensure!(
        encoded.len() <= MAX_PLC_OPERATION_BYTES,
        "PLC operation exceeds {MAX_PLC_OPERATION_BYTES}-byte DAG-CBOR limit"
    );
    Ok(encoded)
}

fn operation_cid(operation: &Value) -> Result<String> {
    let digest = Sha256::digest(plc_dag_cbor(operation)?);
    let mut bytes = Vec::with_capacity(36);
    // CIDv1 || dag-cbor || sha2-256 multihash.
    bytes.extend_from_slice(&[0x01, 0x71, 0x12, 0x20]);
    bytes.extend_from_slice(&digest);
    Ok(format!(
        "b{}",
        data_encoding::BASE32_NOPAD
            .encode(&bytes)
            .to_ascii_lowercase()
    ))
}

fn did_from_genesis(operation: &Value) -> Result<String> {
    let digest = Sha256::digest(plc_dag_cbor(operation)?);
    let encoded = data_encoding::BASE32_NOPAD
        .encode(&digest)
        .to_ascii_lowercase();
    Ok(format!("did:plc:{}", &encoded[..PLC_MSI_LEN]))
}

fn operation_prev(operation: &Value) -> Result<Option<&str>> {
    let object = operation
        .as_object()
        .ok_or_else(|| anyhow!("PLC audit operation is not an object"))?;
    match object.get("prev") {
        Some(Value::Null) => Ok(None),
        Some(Value::String(prev)) if !prev.is_empty() => Ok(Some(prev)),
        Some(_) => bail!("PLC operation has invalid `prev`"),
        None => bail!("PLC operation omits required `prev`"),
    }
}

fn operation_rotation_keys(operation: &Value) -> Result<Vec<&str>> {
    let object = operation
        .as_object()
        .ok_or_else(|| anyhow!("PLC audit operation is not an object"))?;
    match object.get("type").and_then(Value::as_str) {
        Some("plc_operation") => {
            let keys = object
                .get("rotationKeys")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("PLC operation omits `rotationKeys`"))?;
            ensure!(
                (1..=5).contains(&keys.len()),
                "PLC operation must carry 1..=5 rotation keys"
            );
            let mut seen = std::collections::HashSet::new();
            keys.iter()
                .map(|key| {
                    let key = key
                        .as_str()
                        .filter(|key| !key.is_empty())
                        .ok_or_else(|| anyhow!("PLC operation has invalid rotation key"))?;
                    ensure!(seen.insert(key), "PLC operation has duplicate rotation key");
                    Ok(key)
                })
                .collect()
        }
        Some("create") => {
            let key = object
                .get("recoveryKey")
                .and_then(Value::as_str)
                .filter(|key| !key.is_empty())
                .ok_or_else(|| anyhow!("legacy PLC create operation omits `recoveryKey`"))?;
            Ok(vec![key])
        }
        Some("plc_tombstone") => bail!("a PLC tombstone carries no successor rotation keys"),
        Some(other) => bail!("unsupported PLC operation type `{other}`"),
        None => bail!("PLC operation omits string `type`"),
    }
}

fn verify_operation_signature(operation: &Value, rotation_keys: &[&str]) -> Result<()> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use k256::ecdsa::signature::Verifier as _;

    let object = operation
        .as_object()
        .ok_or_else(|| anyhow!("PLC audit operation is not an object"))?;
    let signature_text = object
        .get("sig")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("PLC operation omits string `sig`"))?;
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(signature_text)
        .context("PLC operation signature is not canonical base64url")?;
    ensure!(
        signature_bytes.len() == 64 && URL_SAFE_NO_PAD.encode(&signature_bytes) == signature_text,
        "PLC operation signature is not canonical 64-byte base64url"
    );

    let mut unsigned = operation.clone();
    let unsigned_object = unsigned
        .as_object_mut()
        .ok_or_else(|| anyhow!("PLC audit operation is not an object"))?;
    unsigned_object.remove("sig");
    let message = plc_dag_cbor(&unsigned)?;

    for did_key in rotation_keys {
        let Some(multibase) = did_key.strip_prefix("did:key:z") else {
            continue;
        };
        let Ok(decoded) = bs58::decode(multibase).into_vec() else {
            continue;
        };
        let Some((codec, key_bytes)) = read_multicodec(&decoded) else {
            continue;
        };
        let verified = match codec {
            multicodec::SECP256K1_PUB => {
                if !decoded.starts_with(&[0xe7, 0x01]) {
                    continue;
                }
                let Ok(key) = k256::ecdsa::VerifyingKey::from_sec1_bytes(key_bytes) else {
                    continue;
                };
                let Ok(signature) = k256::ecdsa::Signature::from_slice(&signature_bytes) else {
                    continue;
                };
                if signature.normalize_s().is_some() {
                    continue;
                }
                key.verify(&message, &signature).is_ok()
            }
            multicodec::P256_PUB => {
                if !decoded.starts_with(&[0x80, 0x24]) {
                    continue;
                }
                let Ok(key) = p256::ecdsa::VerifyingKey::from_sec1_bytes(key_bytes) else {
                    continue;
                };
                let Ok(signature) = p256::ecdsa::Signature::from_slice(&signature_bytes) else {
                    continue;
                };
                if signature.normalize_s().is_some() {
                    continue;
                }
                key.verify(&message, &signature).is_ok()
            }
            _ => false,
        };
        if verified {
            return Ok(());
        }
    }
    bail!("PLC operation signature did not verify under any authorized rotation key")
}

fn document_from_operation(did: &str, operation: &Value) -> Result<Value> {
    let object = operation
        .as_object()
        .ok_or_else(|| anyhow!("PLC audit head operation is not an object"))?;
    let op_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("PLC audit head omits string `type`"))?;
    if op_type == "plc_tombstone" {
        bail!("did:plc identity {did} is deactivated");
    }

    let (verification_methods, also_known_as, services) = if op_type == "create" {
        let signing_key = object
            .get("signingKey")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("legacy PLC create omits `signingKey`"))?;
        let handle = object
            .get("handle")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("legacy PLC create omits `handle`"))?;
        let service = object
            .get("service")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("legacy PLC create omits `service`"))?;
        (
            serde_json::json!({ "atproto": signing_key }),
            serde_json::json!([format!("at://{handle}")]),
            serde_json::json!({
                "atproto_pds": {
                    "type": "AtprotoPersonalDataServer",
                    "endpoint": service
                }
            }),
        )
    } else if op_type == "plc_operation" {
        (
            object
                .get("verificationMethods")
                .cloned()
                .ok_or_else(|| anyhow!("PLC operation omits `verificationMethods`"))?,
            object
                .get("alsoKnownAs")
                .cloned()
                .ok_or_else(|| anyhow!("PLC operation omits `alsoKnownAs`"))?,
            object
                .get("services")
                .cloned()
                .ok_or_else(|| anyhow!("PLC operation omits `services`"))?,
        )
    } else {
        bail!("unsupported PLC audit head operation type `{op_type}`");
    };

    let methods = verification_methods
        .as_object()
        .ok_or_else(|| anyhow!("PLC `verificationMethods` is not an object"))?
        .iter()
        .map(|(name, key)| {
            ensure!(
                !name.is_empty() && !name.starts_with('#'),
                "invalid PLC verification-method name"
            );
            let multibase = key
                .as_str()
                .and_then(|key| key.strip_prefix("did:key:"))
                .ok_or_else(|| anyhow!("PLC verification method `{name}` is not a did:key"))?;
            Ok(serde_json::json!({
                "id": format!("{did}#{name}"),
                "type": "Multikey",
                "controller": did,
                "publicKeyMultibase": multibase,
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    let rendered_services = services
        .as_object()
        .ok_or_else(|| anyhow!("PLC `services` is not an object"))?
        .iter()
        .map(|(name, service)| {
            ensure!(
                !name.is_empty() && !name.starts_with('#'),
                "invalid PLC service name"
            );
            let service = service
                .as_object()
                .ok_or_else(|| anyhow!("PLC service `{name}` is not an object"))?;
            let service_type = service
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("PLC service `{name}` omits string `type`"))?;
            let endpoint = service
                .get("endpoint")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("PLC service `{name}` omits string `endpoint`"))?;
            Ok(serde_json::json!({
                "id": format!("#{name}"),
                "type": service_type,
                "serviceEndpoint": endpoint,
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(serde_json::json!({
        "id": did,
        "alsoKnownAs": also_known_as,
        "verificationMethod": methods,
        "service": rendered_services,
    }))
}

fn normalized_document_members(
    did: &str,
    member: &str,
    value: Option<&Value>,
) -> Result<std::collections::BTreeMap<String, Value>> {
    let values = value
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("did:plc document omits array `{member}`"))?;
    values
        .iter()
        .map(|value| {
            let mut object = value
                .as_object()
                .cloned()
                .ok_or_else(|| anyhow!("did:plc document has non-object `{member}` entry"))?;
            let id = object
                .remove("id")
                .and_then(|id| id.as_str().map(str::to_owned))
                .ok_or_else(|| anyhow!("did:plc document `{member}` entry omits string `id`"))?;
            Ok((
                canonical_member_id(did, &id, member)?,
                Value::Object(object),
            ))
        })
        .collect()
}

/// Verify the complete returned operation signature chain and bind the current
/// DID document's identity-bearing fields to the verified active head.
pub fn verify_plc_audit(did: &str, audit: &Value, current_document: &Value) -> Result<()> {
    plc_msi(did)?;
    validate_plc_document(did, current_document)?;
    let entries = audit
        .as_array()
        .filter(|entries| !entries.is_empty())
        .ok_or_else(|| anyhow!("PLC audit log for {did} is empty or not an array"))?;
    let mut cid_to_index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut active = Vec::new();
    let mut genesis_count = 0usize;

    for (index, entry) in entries.iter().enumerate() {
        let entry = entry
            .as_object()
            .ok_or_else(|| anyhow!("PLC audit entry {index} is not an object"))?;
        ensure!(
            entry.get("did").and_then(Value::as_str) == Some(did),
            "PLC audit entry {index} is for a different DID"
        );
        let operation = entry
            .get("operation")
            .ok_or_else(|| anyhow!("PLC audit entry {index} omits `operation`"))?;
        if operation.get("type").and_then(Value::as_str) != Some("plc_tombstone") {
            // Validate every operation's published successor authority,
            // including the active head (which has no child to consume it).
            operation_rotation_keys(operation)?;
        }
        let cid = entry
            .get("cid")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("PLC audit entry {index} omits string `cid`"))?;
        ensure!(
            operation_cid(operation)? == cid,
            "PLC audit entry {index} CID does not match its signed operation"
        );

        let prev = operation_prev(operation)?;
        let authorized_keys = if let Some(prev) = prev {
            let parent_index = *cid_to_index.get(prev).ok_or_else(|| {
                anyhow!("PLC operation `{cid}` references unknown or future prev `{prev}`")
            })?;
            let parent = entries[parent_index]
                .get("operation")
                .ok_or_else(|| anyhow!("PLC parent entry omits operation"))?;
            operation_rotation_keys(parent)?
        } else {
            genesis_count += 1;
            ensure!(
                did_from_genesis(operation)? == did,
                "PLC genesis operation does not bind to {did}"
            );
            operation_rotation_keys(operation)?
        };
        verify_operation_signature(operation, &authorized_keys).with_context(|| {
            format!("PLC audit entry {index} (`{cid}`) signature verification failed")
        })?;
        ensure!(
            cid_to_index.insert(cid.to_owned(), index).is_none(),
            "PLC audit contains duplicate operation CID `{cid}`"
        );

        let nullified = entry
            .get("nullified")
            .and_then(Value::as_bool)
            .ok_or_else(|| anyhow!("PLC audit entry {index} omits boolean `nullified`"))?;
        if !nullified {
            active.push((cid, operation));
        }
    }

    ensure!(
        genesis_count == 1,
        "PLC audit must contain exactly one genesis operation"
    );
    ensure!(
        !active.is_empty(),
        "PLC audit has no active operation chain"
    );
    ensure!(
        operation_prev(active[0].1)?.is_none(),
        "PLC active chain does not start at genesis"
    );
    for pair in active.windows(2) {
        ensure!(
            operation_prev(pair[1].1)? == Some(pair[0].0),
            "PLC active operation chain has a broken `prev` link"
        );
    }

    let active_head = active
        .last()
        .ok_or_else(|| anyhow!("PLC audit has no active operation chain"))?;
    let verified_document = document_from_operation(did, active_head.1)?;
    ensure!(
        current_document.get("id") == verified_document.get("id"),
        "PLC current document `id` does not match the verified audit-log head"
    );
    ensure!(
        current_document.get("alsoKnownAs") == verified_document.get("alsoKnownAs"),
        "PLC current document `alsoKnownAs` does not match the verified audit-log head"
    );
    for member in ["verificationMethod", "service"] {
        ensure!(
            normalized_document_members(did, member, current_document.get(member))?
                == normalized_document_members(did, member, verified_document.get(member))?,
            "PLC current document `{member}` does not match the verified audit-log head"
        );
    }
    Ok(())
}

// ── resolver ──────────────────────────────────────────────────────────────────

/// A PLC fetcher must provide both the current DID document and its audit log.
///
/// Keeping the audit fetch in the resolver's required interface makes it
/// impossible to instantiate a resolver that silently skips chain verification.
#[async_trait]
pub trait PlcAuditFetcher: DidDocFetcher {
    async fn fetch_audit(&self, url: &str) -> Result<Value>;
}

/// Resolves a `did:plc` identifier to its validated DID document.
///
/// Generic over a [`PlcAuditFetcher`] so the parse/validate/verify path is
/// testable with an injected fixture (no live network); the native
/// implementation is [`HttpPlcFetcher`]. This is the federation-intake
/// artifact E2 (#1174, foreign handle resolution) and E3 (#1175, exchange
/// audience) build on.
///
/// Read-only forever: there is no write path. See the module docs.
pub struct DidPlcResolver<F: PlcAuditFetcher> {
    fetcher: F,
    config: PlcResolverConfig,
}

impl<F: PlcAuditFetcher> DidPlcResolver<F> {
    /// Construct a resolver over an injected document+audit fetcher. This is for
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
        let audit_url = did_plc_audit_url(did, self.config.base_url())?;
        let audit = self.fetcher.fetch_audit(&audit_url).await?;
        verify_plc_audit(did, &audit, &doc)?;
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
            // HTTPS GET; a redirect to a loopback / link-local address would
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

    fn validate_fetch_url(&self, url: &str) -> Result<Url> {
        let parsed = Url::parse(url)
            .map_err(|e| anyhow!("did:plc directory URL does not parse ({url}): {e}"))?;
        if parsed.scheme() != "https" {
            bail!("did:plc directory URL must use https:// (got {url})");
        }
        if !self.allowed_origins.contains(&parsed.origin()) {
            bail!(
                "did:plc fetch refused: origin of {url} is not in the egress allowlist (fail-closed)"
            );
        }
        Ok(parsed)
    }

    async fn fetch_json(&self, url: &str, max_bytes: usize) -> Result<Value> {
        {
            let cache = self.cache.lock();
            if let Some(entry) = cache.get(url) {
                if entry.fetched_at.elapsed() < self.ttl {
                    return Ok(entry.doc.clone());
                }
            }
        }

        let resp = self.http.get(url).send().await?;
        if !resp.status().is_success() {
            bail!(
                "did:plc directory returned non-success status {} for {url}",
                resp.status()
            );
        }
        let body = read_capped(resp, max_bytes).await?;
        let value = parse_did_document_no_duplicates(&body, url)?;

        {
            let mut cache = self.cache.lock();
            evict_for_insert(&mut cache, self.ttl);
            cache.insert(
                url.to_owned(),
                CachedDoc {
                    doc: value.clone(),
                    fetched_at: std::time::Instant::now(),
                },
            );
        }
        Ok(value)
    }
}

#[async_trait]
impl DidDocFetcher for HttpPlcFetcher {
    async fn fetch(&self, url: &str) -> Result<Value> {
        let parsed = self.validate_fetch_url(url)?;
        let asked = parsed
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .ok_or_else(|| anyhow!("did:plc document URL has no DID path segment: {url}"))?;
        plc_msi(asked)?;
        let doc = self.fetch_json(url, MAX_DID_DOC_BYTES).await?;
        validate_plc_document(asked, &doc)?;
        Ok(doc)
    }
}

#[async_trait]
impl PlcAuditFetcher for HttpPlcFetcher {
    async fn fetch_audit(&self, url: &str) -> Result<Value> {
        let parsed = self.validate_fetch_url(url)?;
        let segments: Vec<_> = parsed
            .path_segments()
            .ok_or_else(|| anyhow!("did:plc audit URL has no path segments: {url}"))?
            .collect();
        ensure!(
            segments.len() >= 3
                && segments[segments.len() - 2] == "log"
                && segments[segments.len() - 1] == "audit",
            "did:plc audit URL must end in /log/audit"
        );
        plc_msi(segments[segments.len() - 3])?;
        self.fetch_json(url, MAX_PLC_AUDIT_BYTES).await
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

    #[test]
    fn captured_audit_verifies_cids_genesis_and_signature_chain() {
        verify_plc_audit(DID, &captured_audit(), &captured_document()).unwrap();
    }

    #[test]
    fn audit_rejects_signature_tamper_even_with_recomputed_cid() {
        let mut audit = captured_audit();
        let entries = audit.as_array_mut().unwrap();
        let last = entries.last_mut().unwrap();
        let operation = last.get_mut("operation").unwrap();
        operation["sig"] =
            Value::String("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".into());
        last["cid"] = Value::String(operation_cid(operation).unwrap());

        let err = verify_plc_audit(DID, &audit, &captured_document()).unwrap_err();
        assert!(err.to_string().contains("signature"), "{err:#}");
    }

    #[test]
    fn audit_rejects_unsigned_current_document_substitution() {
        let mut substituted = captured_document();
        substituted["verificationMethod"][0]["publicKeyMultibase"] =
            Value::String("zQ3shXjHeiBuRCKmM36cuYnm7YEMzhGnCmCyW92sRJ9pribSF".into());
        let err = verify_plc_audit(DID, &captured_audit(), &substituted).unwrap_err();
        assert!(err.to_string().contains("verified audit-log head"), "{err}");
    }

    #[test]
    fn audit_head_comparison_canonicalizes_relative_member_ids() {
        let mut document = captured_document();
        document["verificationMethod"][0]["id"] = Value::String("#atproto".into());
        verify_plc_audit(DID, &captured_audit(), &document).unwrap();
    }

    // ── Finding 2: typed verification-material / member validation ───────
    //
    // Each case below was wrongly ACCEPTED by the prior ad-hoc presence check.
    // The validator must fail closed on malformed material, ambiguous members,
    // and semantically-duplicate identifiers, while preserving conforming
    // extensions (an unknown algorithm with a real key still resolves).

    /// A `publicKeyMultibase` that is not a valid Multikey public key
    /// (`zKey` — too short to carry any multicodec + key) and a `publicKeyJwk`
    /// that is not a JWK (`{ "garbage": true }`, no `kty`) must be rejected.
    #[test]
    fn validate_rejects_malformed_verification_material() {
        // Multikey with a structurally-invalid `publicKeyMultibase`.
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "#bad-multikey", "type": "Multikey", "controller": DID,
                    "publicKeyMultibase": "zKey"
                }]
            })
        )
        .is_err());
        // A `publicKeyMultibase` that is not even multibase base58btc.
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "#bad-multikey2", "type": "Multikey", "controller": DID,
                    "publicKeyMultibase": "not-multibase"
                }]
            })
        )
        .is_err());
        // `publicKeyJwk` missing `kty` is not a JWK.
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "#bad-jwk", "type": "JsonWebKey", "controller": DID,
                    "publicKeyJwk": { "garbage": true }
                }]
            })
        )
        .is_err());
        // A JWK with an unknown `kty` is not a recognized JWK either.
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "#bad-jwk2", "type": "JsonWebKey", "controller": DID,
                    "publicKeyJwk": { "kty": "NotARealKty" }
                }]
            })
        )
        .is_err());
        // A verification method with NO recognized material property is
        // non-conforming (DID Core §5.1 requires material).
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "#no-material", "type": "Multikey", "controller": DID
                }]
            })
        )
        .is_err());
    }

    /// A conforming extension — an unknown algorithm that still carries a
    /// valid Multikey payload — must be preserved (interop: ignore unknown
    /// algs rather than failing).
    #[test]
    fn validate_preserves_unknown_algorithm_with_valid_material() {
        // A synthetic unknown multicodec (0x0bad) with a 32-byte payload is
        // structurally a real Multikey; the resolver must not reject it.
        let unknown_key = {
            let payload = [0x0bu8, 0xad, 0x01]; // varint codec 0x0bad01-ish + ...
            let mut bytes = vec![0xfd, 0x01]; // some unknown 2-byte codec
            bytes.extend_from_slice(&payload);
            bytes.extend_from_slice(&[0u8; 30]); // bring payload ≥ floor
            format!("z{}", bs58::encode(&bytes).into_string())
        };
        let doc = json!({
            "id": DID,
            "verificationMethod": [{
                "id": "#unknown", "type": "UnknownVerificationMethod2026",
                "controller": DID, "publicKeyMultibase": unknown_key
            }]
        });
        assert!(validate_plc_document(DID, &doc).is_ok());
    }

    /// Multiple verification-material properties on one method is ambiguous
    /// and must fail closed (DID Core requires exactly one).
    #[test]
    fn validate_rejects_multiple_material_properties() {
        // A valid Ed25519 Multikey present alongside a JWK: ambiguous.
        let mk = "zQ3shunBKsXixLxKtC5qeSG9E4J5RkGN57im31pcTzbNQnm5w";
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "#both", "type": "Multikey", "controller": DID,
                    "publicKeyMultibase": mk,
                    "publicKeyJwk": { "kty": "OKP", "crv": "Ed25519", "x": "11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlnw_vyHashME" }
                }]
            })
        )
        .is_err());
    }

    /// A relative reference `#key` and the absolute DID URL
    /// `did:plc:…#key` name the SAME resource (DID Core relative-URL
    /// resolution); both must not be accepted together.
    #[test]
    fn validate_rejects_semantically_duplicate_ids() {
        let mk = "zQ3shunBKsXixLxKtC5qeSG9E4J5RkGN57im31pcTzbNQnm5w";
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [
                    { "id": "#key", "type": "Multikey", "controller": DID, "publicKeyMultibase": mk },
                    { "id": format!("{DID}#key"), "type": "Multikey", "controller": DID, "publicKeyMultibase": mk }
                ]
            })
        )
        .is_err());
    }

    /// DID/URI fields are checked for their required syntax, not merely
    /// non-emptiness: `alsoKnownAs` entries, string `serviceEndpoint`s, VM
    /// `id`s, and `controller`s must be URIs / DID URLs / DIDs respectively.
    #[test]
    fn validate_rejects_non_did_or_non_uri_fields() {
        // alsoKnownAs must be URIs.
        assert!(
            validate_plc_document(DID, &json!({ "id": DID, "alsoKnownAs": ["not-a-uri"] }))
                .is_err()
        );
        // controller must be a DID.
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "#k", "type": "Multikey", "controller": "not-a-did",
                    "publicKeyMultibase": "zQ3shunBKsXixLxKtC5qeSG9E4J5RkGN57im31pcTzbNQnm5w"
                }]
            })
        )
        .is_err());
        // An absolute VM id that is not a DID URL (https URL) is rejected.
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "verificationMethod": [{
                    "id": "https://evil.example/key", "type": "Multikey", "controller": DID,
                    "publicKeyMultibase": "zQ3shunBKsXixLxKtC5qeSG9E4J5RkGN57im31pcTzbNQnm5w"
                }]
            })
        )
        .is_err());
        // A string serviceEndpoint must be a URI.
        assert!(validate_plc_document(
            DID,
            &json!({
                "id": DID,
                "service": [{ "id": "#pds", "type": "T", "serviceEndpoint": "no-scheme-here" }]
            })
        )
        .is_err());
    }

    // ── resolver over an injected fixture fetcher (no network) ───────────

    struct FixtureFetcher {
        doc: Value,
        audit: Value,
    }

    #[async_trait]
    impl DidDocFetcher for FixtureFetcher {
        async fn fetch(&self, url: &str) -> Result<Value> {
            assert_eq!(url, "https://plc.example/did:plc:ewvi7nxzyoun6zhxrhs64oiz");
            Ok(self.doc.clone())
        }
    }

    #[async_trait]
    impl PlcAuditFetcher for FixtureFetcher {
        async fn fetch_audit(&self, url: &str) -> Result<Value> {
            assert_eq!(
                url,
                "https://plc.example/did:plc:ewvi7nxzyoun6zhxrhs64oiz/log/audit"
            );
            Ok(self.audit.clone())
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

    #[async_trait]
    impl PlcAuditFetcher for NeverFetcher {
        async fn fetch_audit(&self, url: &str) -> Result<Value> {
            panic!("resolver must not fetch audit for a malformed DID (called with {url})");
        }
    }

    fn captured_document() -> Value {
        parse_did_document_no_duplicates(
            include_bytes!("fixtures/plc_directory_ewvi7nxzyoun6zhxrhs64oiz.json"),
            "captured PLC document",
        )
        .unwrap()
    }

    fn captured_audit() -> Value {
        parse_did_document_no_duplicates(
            include_bytes!("fixtures/plc_directory_ewvi7nxzyoun6zhxrhs64oiz_audit.json"),
            "captured PLC audit",
        )
        .unwrap()
    }

    fn fixture_resolver(doc: Value) -> DidPlcResolver<FixtureFetcher> {
        DidPlcResolver::with_fetcher(
            FixtureFetcher {
                doc,
                audit: captured_audit(),
            },
            PlcResolverConfig::new(configured_base(), std::time::Duration::from_secs(60)).unwrap(),
        )
    }

    #[tokio::test]
    async fn resolve_document_returns_signature_chain_verified_doc() {
        let doc = captured_document();
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
        #[async_trait]
        impl PlcAuditFetcher for FailingFetcher {
            async fn fetch_audit(&self, url: &str) -> Result<Value> {
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
