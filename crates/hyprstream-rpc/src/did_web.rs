//! `did:web` → dialable [`TransportConfig`] resolution (#279).
//!
//! The mesh's only live [`crate::resolver::Resolver`] is the
//! [`crate::registry::EndpointRegistry`] (service-name → local endpoint). This
//! module adds the *federated* reach path: a [`DidWebResolver`] that turns a
//! `did:web` identifier into one-or-more dialable [`TransportConfig`]s by
//! fetching the subject's DID document and decoding its typed transport
//! `service` entries.
//!
//! # Layering
//!
//! - **DID → URL** ([`did_web_to_url`]) — pure, per the did:web spec.
//! - **DID-doc → reach** ([`transport_entries`] / [`preferred_transport`]) —
//!   reuses [`crate::service_entry::decode_service_entry`] (the canonical
//!   transport codec; this module never re-implements transport decoding) and
//!   applies a deterministic preference order.
//! - **Fetch + cache** ([`DidDocFetcher`]) — abstracted behind a trait so the
//!   parse/decode path is testable with an injected fixture (no live network).
//!   The native HTTPS implementation lives in [`HttpDidDocFetcher`] (TTL-cached,
//!   mirroring the JWKS fetch+cache posture in `FederationKeyResolver`).
//!
//! # Scope (#279)
//!
//! Transport **reach** only. Peer *identity* anchoring (`#mesh` ML-DSA, the
//! PQ trust store) is #157/#154/#137 (already landed) and is NOT done here.
//! `did:key` resolution is #281 (a later ticket); only `did:web` here.

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::registry::SocketKind;
use crate::resolver::Resolver;
use crate::service_entry::{decode_service_entry, DecodedEntry};
use crate::transport::{EndpointType, TransportConfig};

// ── did:web → URL derivation ──────────────────────────────────────────────────

/// Derive the HTTPS DID-document URL for a `did:web` identifier, per the did:web
/// spec (<https://w3c-ccg.github.io/did-method-web/#read-resolve>).
///
/// Two forms:
///
/// - **Bare host** — `did:web:{host}` → `https://{host}/.well-known/did.json`.
///   `{host}` may carry a percent-encoded port (`%3A` → `:`), e.g.
///   `did:web:localhost%3A8443` → `https://localhost:8443/.well-known/did.json`.
/// - **Path** — `did:web:{host}:{p1}:{p2}…` → `https://{host}/{p1}/{p2}…/did.json`.
///   The method-specific identifier's colons separate path segments; within each
///   segment, percent-encoding is decoded (so a literal colon in a segment is
///   `%3A`). Only the *first* segment is the host (it alone may carry a port).
///
/// Returns `Err` if the input is not a `did:web` identifier or has an empty host.
pub fn did_web_to_url(did: &str) -> Result<String> {
    // Strip any DID URL fragment / query (we resolve the base document).
    let did = did.split(['#', '?']).next().unwrap_or(did);
    let msi = did
        .strip_prefix("did:web:")
        .ok_or_else(|| anyhow!("not a did:web identifier: {did}"))?;
    if msi.is_empty() {
        bail!("did:web has empty method-specific identifier: {did}");
    }

    // Colons delimit the host then path segments; percent-decode each.
    let mut segments = msi.split(':');
    let host_raw = segments.next().unwrap_or_default();
    let host = percent_decode(host_raw);
    if host.is_empty() {
        bail!("did:web has empty host: {did}");
    }
    // SECURITY: a percent-decoded host may legitimately carry a port
    // (`host:port`, the only place `:` survives decoding here), but must never
    // contain path-y / control content that the `url` crate could normalize into
    // a different authority or inject a path. Reject `/`, `\`, and NUL.
    validate_host_segment(&host, did)?;

    // SECURITY: percent-decoding a path segment can yield `/` (`%2F`), `..`
    // (`%2E%2E`), `\`, or NUL — all of which the `url` crate would normalize into
    // a DIFFERENT document path (path traversal). Reject any such segment, plus
    // `.`/`..`/empty, before joining. (`split` already drops the empty segments
    // produced by consecutive colons; this also rejects a segment that decodes to
    // empty.)
    let mut path_segments: Vec<String> = Vec::new();
    for raw in segments.filter(|s| !s.is_empty()) {
        let seg = percent_decode(raw);
        validate_path_segment(&seg, did)?;
        path_segments.push(seg);
    }

    if path_segments.is_empty() {
        // Bare host form → well-known document.
        Ok(format!("https://{host}/.well-known/did.json"))
    } else {
        // Path form → document at the joined path.
        Ok(format!("https://{host}/{}/did.json", path_segments.join("/")))
    }
}

/// Minimal percent-decoder for did:web identifier segments.
///
/// did:web only mandates that `:` (port / segment separators) be percent-encoded
/// (`%3A`); a general decoder handles any `%XX`. Invalid escapes are left
/// verbatim (defensive: never panic on malformed input).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Reject a decoded host that contains path-injecting or control characters.
///
/// A did:web host may carry an encoded port (`%3A` → `:`), so `:` is allowed
/// here, but `/`, `\`, and NUL would let an attacker inject a path or smuggle a
/// different authority once interpolated into the URL.
fn validate_host_segment(host: &str, did: &str) -> Result<()> {
    if host.contains(['/', '\\', '\0']) {
        bail!("did:web host contains an illegal character (`/`, `\\`, or NUL): {did}");
    }
    Ok(())
}

/// Reject a decoded path segment that would cause path traversal / injection.
///
/// After percent-decoding, a segment must be a single non-empty path component:
/// not `.` or `..`, and free of `/`, `\`, and NUL. This blocks `%2F`/`%2E%2E`
/// style segments that the `url` crate would otherwise normalize into a
/// different document path.
fn validate_path_segment(seg: &str, did: &str) -> Result<()> {
    if seg.is_empty() || seg == "." || seg == ".." {
        bail!("did:web path segment is empty, `.`, or `..` (traversal): {did}");
    }
    if seg.contains(['/', '\\', '\0']) {
        bail!("did:web path segment contains an illegal character (`/`, `\\`, or NUL): {did}");
    }
    Ok(())
}

// ── DID-doc → reach (preference-ordered) ──────────────────────────────────────

/// Decode all typed transport `service` entries from a DID document into
/// dialable [`DecodedEntry`]s, in **preference order**.
///
/// Preference (deterministic):
///   1. By transport class: `IrohTransport` before `QuicTransport` — iroh is
///      identity-bound at the transport (the `nodeId` *is* the Ed25519 identity,
///      so the channel authenticates the peer; see [`crate::service_entry`]),
///      whereas QUIC's cert pin is channel-only.
///   2. Within a class, document order is preserved (stable sort) — operators
///      can express secondary preference by ordering entries in the doc.
///
/// Entries that aren't recognized transport services (`HyprstreamService`,
/// `AtprotoPersonalDataServer`, etc.) or that fail to decode are skipped
/// silently — a DID document with no transport reach resolves to an **empty**
/// vec (graceful), not an error.
pub fn transport_entries(doc: &Value) -> Vec<DecodedEntry> {
    let Some(services) = doc.get("service").and_then(Value::as_array) else {
        return Vec::new();
    };

    // (preference_rank, doc_index, decoded)
    let mut ranked: Vec<(u8, usize, DecodedEntry)> = Vec::new();
    for (idx, entry) in services.iter().enumerate() {
        let ty = entry.get("type").and_then(Value::as_str).unwrap_or_default();
        let rank = match ty {
            "IrohTransport" => 0,
            "QuicTransport" => 1,
            "OnionTransport" => {
                // The DID-doc builder may emit OnionTransport, but neither
                // `service_entry::decode_service_entry` nor the `dial`/`dial_stream`
                // layer can represent or dial an onion endpoint yet (there is no
                // `EndpointType::Onion`). Don't silently drop it — note that a
                // reachable onion entry was skipped so it's diagnosable.
                // TODO(onion): #279 — add an onion EndpointType + dial support and
                // fold this into the preference order (after quic).
                tracing::debug!(
                    service_type = %ty,
                    "skipping OnionTransport DID-doc reach: onion is not yet dialable (no EndpointType::Onion)"
                );
                continue;
            }
            // Not a transport service entry — skip without attempting to decode.
            _ => continue,
        };
        // `decode_service_entry` is the canonical codec; tolerate per-entry
        // failures (e.g. a malformed pin) by skipping rather than failing the
        // whole resolve.
        match decode_service_entry(entry) {
            Ok(decoded) => ranked.push((rank, idx, decoded)),
            Err(e) => {
                tracing::debug!(error = %e, service_type = %ty, "skipping undecodable transport service entry");
            }
        }
    }

    // Stable sort by (rank, doc index): preserves document order within a class.
    ranked.sort_by_key(|(rank, idx, _)| (*rank, *idx));
    ranked.into_iter().map(|(_, _, d)| d).collect()
}

/// The single most-preferred transport [`TransportConfig`] from a DID document,
/// optionally constrained to a [`SocketKind`].
///
/// When `kind` is `Some(SocketKind::Quic)`, only QUIC entries are considered
/// (the [`Resolver`] contract asks for a specific socket kind). `None` returns
/// the overall top preference from [`transport_entries`]. Returns `None` when no
/// matching reach exists.
pub fn preferred_transport(doc: &Value, kind: Option<SocketKind>) -> Option<TransportConfig> {
    let entries = transport_entries(doc);
    entries
        .into_iter()
        .find(|d| match kind {
            Some(SocketKind::Quic) => matches!(d.config.endpoint, EndpointType::Quic { .. }),
            // The DID-doc reach is always network transport (Iroh/Quic); any
            // other requested kind has no federated reach to offer.
            Some(_) => false,
            None => true,
        })
        .map(|d| d.config)
}

// ── DID-doc → verification-method keys (#137 admission stage 2) ────────────────

/// Multicodec `ed25519-pub` unsigned-varint prefix (`0xed01` → bytes `0xed 0x01`).
///
/// Mirrors `hyprstream::auth::mesh_trust::MULTICODEC_ED25519_PUB`; duplicated here
/// (a 2-byte constant) because `hyprstream-rpc` is the lower crate and cannot
/// depend on `hyprstream` for the `verificationMethod` decode the admission gate
/// (#137) needs at this layer.
const MULTICODEC_ED25519_PUB: [u8; 2] = [0xed, 0x01];

/// Decode a `Multikey` `publicKeyMultibase` string into raw Ed25519 key bytes.
///
/// Verifies the base58btc multibase prefix (`z`) and the `ed25519-pub` multicodec
/// header, returning the 32-byte payload. This is the `did_web`/`hyprstream-rpc`
/// sibling of `hyprstream::auth::mesh_trust::decode_multikey` (kept local because
/// `hyprstream-rpc` cannot depend on `hyprstream`); both are exercised against the
/// same `encode_multikey` output in tests.
///
/// Returns `Err` for a wrong multibase, an undecodable base58, a non-Ed25519
/// codec, or a payload that is not exactly 32 bytes.
pub fn decode_ed25519_multikey(multibase: &str) -> Result<[u8; 32]> {
    let body = multibase
        .strip_prefix('z')
        .ok_or_else(|| anyhow!("Multikey must use base58btc multibase ('z') prefix"))?;
    let decoded = bs58::decode(body)
        .into_vec()
        .map_err(|e| anyhow!("invalid base58btc Multikey: {e}"))?;
    if decoded.len() < 2 || decoded[..2] != MULTICODEC_ED25519_PUB {
        bail!(
            "unexpected multicodec prefix (expected ed25519-pub {MULTICODEC_ED25519_PUB:02x?}, got {:02x?})",
            decoded.get(..2).unwrap_or(&decoded)
        );
    }
    let raw: [u8; 32] = decoded[2..]
        .try_into()
        .map_err(|_| anyhow!("ed25519 Multikey payload is {} bytes (expected 32)", decoded.len() - 2))?;
    Ok(raw)
}

// ── did:key (Ed25519) interop (#281) ──────────────────────────────────────────

/// Decode a `did:key` (Ed25519) identifier into its raw 32-byte Ed25519 key.
///
/// Tiles (and the broader did:key ecosystem) uses `did:key` for self-certifying
/// device/account identity. For Ed25519 a `did:key` is *exactly*
/// `"did:key:" + multibase-base58btc(0xed01 ‖ pubkey)` — i.e. the method-specific
/// identifier is the same `Multikey` `publicKeyMultibase` value we already decode
/// for `verificationMethod` (see [`decode_ed25519_multikey`]). A `did:key` is
/// therefore **self-contained**: the key *is* the identity, so this is a pure
/// decode with **no network fetch** (the inverse of `did:web`, which resolves a
/// document). Reach for a `did:key` peer comes from iroh discovery (#282), not the
/// DID string.
///
/// Returns `Err` for a non-`did:key` identifier, a non-Ed25519 multicodec, a bad
/// multibase, or a wrong-length payload. A DID URL fragment / query is stripped
/// (we decode the base identifier).
pub fn did_key_to_ed25519(did: &str) -> Result<[u8; 32]> {
    // Strip any DID URL fragment / query (`did:key:z6Mk…#z6Mk…` self-references
    // the same key; we decode the base method-specific identifier).
    let did = did.split(['#', '?']).next().unwrap_or(did);
    let msi = did
        .strip_prefix("did:key:")
        .ok_or_else(|| anyhow!("not a did:key identifier: {did}"))?;
    if msi.is_empty() {
        bail!("did:key has empty method-specific identifier: {did}");
    }
    // The method-specific id IS a Multikey `publicKeyMultibase`; reuse the one
    // source of truth for the ed25519-pub multicodec (no duplicated 0xed01).
    decode_ed25519_multikey(msi)
        .map_err(|e| anyhow!("did:key {did} is not a valid Ed25519 Multikey: {e}"))
}

/// Encode a raw 32-byte Ed25519 public key as a `did:key` (Ed25519) identifier
/// (`did:key:z6Mk…`).
///
/// Reverse interop for #281: lets our Ed25519 keys (the `#mesh` / `#iroh` VMs)
/// render as the `did:key` form Tiles and other did:key consumers expect. The
/// produced string is `"did:key:" + ed25519_to_multibase(key)` and round-trips
/// with [`did_key_to_ed25519`]; the Multikey body is byte-identical to the
/// `publicKeyMultibase` our DID documents publish (`#280`'s `ed25519_to_multibase`
/// over the same `0xed01 ‖ key` payload — one multicodec source of truth).
pub fn ed25519_to_did_key(key: &[u8; 32]) -> String {
    let mut payload = Vec::with_capacity(2 + 32);
    payload.extend_from_slice(&MULTICODEC_ED25519_PUB);
    payload.extend_from_slice(key);
    format!("did:key:z{}", bs58::encode(payload).into_string())
}

/// Whether `did` is a `did:key` identifier (the self-certifying interop arm).
///
/// The admission gate (#137) uses this to route a `did:key` peer down the
/// self-certifying path (the key *is* the identity — no DID-doc resolution /
/// fetch) instead of the `did:web` resolver path.
pub fn is_did_key(did: &str) -> bool {
    did.starts_with("did:key:")
}

/// Extract every Ed25519 public key published in a DID document's
/// `verificationMethod` array, decoded to raw 32-byte keys (#137 stage 2).
///
/// Only `Multikey` / `Ed25519VerificationKey2020`-shaped entries with a
/// `publicKeyMultibase` carrying the `ed25519-pub` multicodec are returned (the
/// `#mesh` / `#iroh` Ed25519 VMs the mesh publishes). Entries with a different
/// codec, a non-multibase encoding (e.g. `publicKeyJwk`-only), or that fail to
/// decode are skipped — a malformed VM must not be admitted, and a doc with no
/// Ed25519 VM yields an **empty** vec (the caller treats empty as "no match →
/// reject", preserving §4.4 fail-closed posture).
pub fn verification_method_ed25519_keys(doc: &Value) -> Vec<[u8; 32]> {
    let Some(vms) = doc.get("verificationMethod").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut keys = Vec::new();
    for vm in vms {
        let Some(mb) = vm.get("publicKeyMultibase").and_then(Value::as_str) else {
            continue;
        };
        match decode_ed25519_multikey(mb) {
            Ok(k) => keys.push(k),
            Err(e) => {
                tracing::debug!(error = %e, "skipping undecodable verificationMethod Multikey");
            }
        }
    }
    keys
}

/// Extract every Ed25519 public key from a JWKS document (RFC 7517), decoded to
/// raw 32-byte keys (#137 stage 2, JWKS fallback path).
///
/// Mirrors `FederationKeyResolver::fetch_ed25519_key`'s parse: an OKP key with
/// `crv == "Ed25519"` and a base64url `x` of exactly 32 bytes. Keys that don't
/// match that shape (wrong `kty`/`crv`, malformed `x`) are skipped. A JWKS with
/// no usable Ed25519 key yields an **empty** vec (→ no match → reject).
///
/// NOTE: this does not itself enforce a "federation"-tagged `use`; the caller
/// supplies the federation JWKS document (e.g. the per-issuer `jwks_uri` already
/// resolved/cached by `FederationKeyResolver`), and tag filtering, if any, is the
/// caller's policy. This keeps the decode pure and reuses the existing JWKS cache
/// rather than adding a new fetch pipeline.
pub fn jwks_ed25519_keys(jwks: &Value) -> Vec<[u8; 32]> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let Some(keys) = jwks.get("keys").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for key in keys {
        if key.get("kty").and_then(Value::as_str) != Some("OKP") {
            continue;
        }
        if key.get("crv").and_then(Value::as_str) != Some("Ed25519") {
            continue;
        }
        let Some(x) = key.get("x").and_then(Value::as_str) else {
            continue;
        };
        match URL_SAFE_NO_PAD.decode(x).ok().and_then(|raw| <[u8; 32]>::try_from(raw).ok()) {
            Some(k) => out.push(k),
            None => {
                tracing::debug!("skipping JWKS OKP/Ed25519 key with malformed/wrong-length 'x'");
            }
        }
    }
    out
}

// ── fetcher abstraction ───────────────────────────────────────────────────────

/// Fetches a DID document (parsed JSON) for a resolved HTTPS URL.
///
/// Abstracted so the resolver's parse/decode path is testable without a live
/// network (inject a fixture). The native HTTPS+cache implementation is
/// [`HttpDidDocFetcher`].
#[async_trait]
pub trait DidDocFetcher: Send + Sync {
    /// Fetch and parse the DID document at `url` (already derived from the DID).
    async fn fetch(&self, url: &str) -> Result<Value>;
}

// ── resolver ──────────────────────────────────────────────────────────────────

/// A [`Resolver`] that turns a `did:web` identifier into a dialable
/// [`TransportConfig`].
///
/// The resolver's `name` argument is the `did:web` identifier; the `kind`
/// constrains the desired socket (today only [`SocketKind::Quic`] has a
/// federated reach). Non-`did:web` names are rejected — chain this *behind* the
/// registry resolver (see [`ChainedResolver`]) so local service-name lookups
/// still work.
pub struct DidWebResolver<F: DidDocFetcher> {
    fetcher: F,
}

impl<F: DidDocFetcher> DidWebResolver<F> {
    /// Construct a resolver over a document fetcher.
    pub fn new(fetcher: F) -> Self {
        Self { fetcher }
    }

    /// Resolve a `did:web` identifier to its preference-ordered reach list.
    ///
    /// Empty vec when the document has no transport service entries.
    pub async fn resolve_all(&self, did: &str) -> Result<Vec<DecodedEntry>> {
        let url = did_web_to_url(did)?;
        let doc = self.fetcher.fetch(&url).await?;
        Ok(transport_entries(&doc))
    }

    /// Resolve a `did:web` identifier to its **raw** DID document JSON.
    ///
    /// `resolve`/`resolve_all` discard everything but transport reach; the #137
    /// admission gate needs the full document to read `verificationMethod` (the
    /// peer's published Ed25519 keys). Reuses the same derive-URL + cached fetch
    /// path — no new fetch/cache infra.
    pub async fn resolve_document(&self, did: &str) -> Result<Value> {
        let url = did_web_to_url(did)?;
        self.fetcher.fetch(&url).await
    }
}

#[async_trait]
impl<F: DidDocFetcher> Resolver for DidWebResolver<F> {
    async fn resolve(&self, name: &str, kind: SocketKind) -> Result<TransportConfig> {
        let url = did_web_to_url(name)?;
        let doc = self.fetcher.fetch(&url).await?;
        preferred_transport(&doc, Some(kind)).ok_or_else(|| {
            anyhow!("did:web {name}: no dialable {kind:?} transport reach in DID document")
        })
    }
}

/// Composes a primary resolver with a fallback: tries `primary` first, and on
/// error falls back to `fallback`.
///
/// Use this to install [`DidWebResolver`] *without* clobbering the registry's
/// local service-name resolution: `ChainedResolver::new(did_web, registry)`
/// resolves `did:web:*` via the DID resolver and everything else via the
/// registry (the DID resolver rejects non-`did:web` names, so it falls
/// through). Install via [`crate::resolver::set_global`].
pub struct ChainedResolver {
    primary: std::sync::Arc<dyn Resolver>,
    fallback: std::sync::Arc<dyn Resolver>,
}

impl ChainedResolver {
    /// Try `primary`, fall back to `fallback` on error.
    pub fn new(
        primary: std::sync::Arc<dyn Resolver>,
        fallback: std::sync::Arc<dyn Resolver>,
    ) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl Resolver for ChainedResolver {
    async fn resolve(&self, name: &str, kind: SocketKind) -> Result<TransportConfig> {
        match self.primary.resolve(name, kind).await {
            Ok(cfg) => Ok(cfg),
            Err(primary_err) => self.fallback.resolve(name, kind).await.map_err(|fb_err| {
                anyhow!("resolve {name}: primary failed ({primary_err}); fallback failed ({fb_err})")
            }),
        }
    }
}

/// Install a [`DidWebResolver`] as the global resolver, chained in front of the
/// existing global resolver (typically the registry) so local service-name
/// lookups still resolve.
///
/// No-op replacement of the global resolver if none is currently set: in that
/// case the DID resolver is installed alone (callers should normally
/// `registry::init()` first).
pub fn install_chained<F: DidDocFetcher + 'static>(did_web: DidWebResolver<F>) {
    let did_web: std::sync::Arc<dyn Resolver> = std::sync::Arc::new(did_web);
    match crate::resolver::try_global() {
        Some(existing) => {
            crate::resolver::set_global(std::sync::Arc::new(ChainedResolver::new(did_web, existing)));
        }
        None => crate::resolver::set_global(did_web),
    }
}

// ── native HTTPS fetcher (TTL-cached) ─────────────────────────────────────────

/// Maximum DID-document body we will buffer (1 MiB). A DID document is small;
/// this ceiling bounds memory against a hostile or buggy peer streaming an
/// unbounded body. Bodies larger than this are rejected.
const MAX_DID_DOC_BYTES: usize = 1024 * 1024;

/// Maximum number of cached DID documents. The cache is keyed by an
/// attacker-influenceable URL, so it is bounded: on insert we sweep expired
/// entries and, if still at the cap, evict the oldest (least-recently-fetched)
/// entry. Keeps cache memory O(MAX_CACHE_ENTRIES) regardless of distinct DIDs.
const MAX_CACHE_ENTRIES: usize = 256;

/// Native HTTPS DID-document fetcher with a TTL cache.
///
/// Mirrors the JWKS fetch+cache posture in `FederationKeyResolver`
/// (`crates/hyprstream/src/auth/federation.rs`): HTTPS-only by default, a
/// `reqwest` client with a request timeout, and a single `Mutex`-guarded cache
/// so concurrent misses for the same URL serialize on one fetch. The cache key
/// is the resolved document URL.
///
/// HTTPS is required (did:web mandates it). A `Mutex` (not `RwLock`) serializes
/// reads and writes so a second waiter re-checks after the first finishes.
///
/// # SSRF / DoS hardening
///
/// - **No redirects.** The client is built with
///   [`reqwest::redirect::Policy::none()`]: the did:web spec retrieves the
///   document by a direct HTTPS GET, so an upstream `302` to
///   `http://169.254.169.254/…` or `http://127.0.0.1:…` cannot bypass the
///   https-only check (SSRF). The https-only check on the resolved URL is kept.
/// - **Bounded body.** Responses are read with a [`MAX_DID_DOC_BYTES`] ceiling
///   (read N+1 bytes; error if exceeded) before JSON parsing — bounds memory.
/// - **Bounded cache.** The TTL cache is capped at [`MAX_CACHE_ENTRIES`]
///   (sweep-expired + evict-oldest on insert) so an attacker-driven flood of
///   distinct DIDs cannot grow it without bound. Failures are never cached.
pub struct HttpDidDocFetcher {
    http: reqwest::Client,
    cache: parking_lot::Mutex<std::collections::HashMap<String, CachedDoc>>,
    ttl: std::time::Duration,
}

struct CachedDoc {
    doc: Value,
    fetched_at: std::time::Instant,
}

impl HttpDidDocFetcher {
    /// Construct a fetcher with a cache TTL (mirrors the JWKS cache TTL default
    /// of one hour; callers honoring per-issuer TTL should pass it through).
    ///
    /// Returns `Err` if the reqwest client fails to build — propagated rather
    /// than swallowed (a `unwrap_or_default()` would yield a client WITHOUT the
    /// configured timeout/redirect policy, defeating the SSRF/DoS hardening).
    pub fn new(ttl: std::time::Duration) -> Result<Self> {
        let http = reqwest::Client::builder()
            // SSRF: do NOT follow redirects. did:web is a direct HTTPS GET; a
            // redirect to http://127.0.0.1 / link-local would bypass https-only.
            .redirect(reqwest::redirect::Policy::none())
            // Bound how long a connect/request can hang.
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow!("failed to build did:web HTTPS client: {e}"))?;
        Ok(Self {
            http,
            cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            ttl,
        })
    }
}

impl Default for HttpDidDocFetcher {
    // The reqwest client builder is infallible by contract for this fixed
    // config; surface a clear panic message if that ever changes rather than a
    // silently-misconfigured client (the `unwrap_or_default()` anti-pattern).
    #[allow(clippy::expect_used)]
    fn default() -> Self {
        // One hour, matching `default_client_jwks_uri_cache_ttl()`.
        Self::new(std::time::Duration::from_secs(3600))
            .expect("did:web HTTPS client must build with the default configuration")
    }
}

#[async_trait]
impl DidDocFetcher for HttpDidDocFetcher {
    async fn fetch(&self, url: &str) -> Result<Value> {
        // did:web is HTTPS-only.
        if !url.starts_with("https://") {
            bail!("did:web document URL must use https:// (got {url})");
        }

        // Cache hit (not expired)?
        {
            let cache = self.cache.lock();
            if let Some(entry) = cache.get(url) {
                if entry.fetched_at.elapsed() < self.ttl {
                    return Ok(entry.doc.clone());
                }
            }
        }

        // Miss / expired — fetch over HTTPS (lock not held across await).
        let resp = self.http.get(url).send().await?.error_for_status()?;
        let body = read_capped(resp, MAX_DID_DOC_BYTES).await?;
        let doc: Value = serde_json::from_slice(&body)
            .map_err(|e| anyhow!("did:web document at {url} is not valid JSON: {e}"))?;

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

/// Read a response body with a hard ceiling: stream chunks and error if the
/// accumulated size would exceed `max` bytes (memory-DoS guard). Returns the
/// buffered body on success.
async fn read_capped(resp: reqwest::Response, max: usize) -> Result<Vec<u8>> {
    use futures::StreamExt;

    // Fast reject when the server advertises an oversized body up front.
    if let Some(len) = resp.content_length() {
        if len > max as u64 {
            bail!("did:web document exceeds {max}-byte cap (Content-Length {len})");
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        push_capped(&mut buf, &chunk, max)?;
    }
    Ok(buf)
}

/// Append `chunk` to `buf`, erroring if the result would exceed `max` bytes.
/// The cap-enforcement core of [`read_capped`], factored out for direct
/// testing (constructing a real `reqwest::Response` cross-version is brittle).
fn push_capped(buf: &mut Vec<u8>, chunk: &[u8], max: usize) -> Result<()> {
    // Read up to max+1 bytes' worth: as soon as we cross the ceiling, error.
    if buf.len() + chunk.len() > max {
        bail!("did:web document exceeds {max}-byte cap");
    }
    buf.extend_from_slice(chunk);
    Ok(())
}

/// Bound the cache before an insert: drop expired entries, then — if still at or
/// over the cap — evict the oldest (least-recently-fetched) entry. Keeps size
/// ≤ [`MAX_CACHE_ENTRIES`] after the caller's subsequent insert.
fn evict_for_insert(
    cache: &mut std::collections::HashMap<String, CachedDoc>,
    ttl: std::time::Duration,
) {
    // Sweep expired entries first (cheap, also reclaims stale memory).
    cache.retain(|_, entry| entry.fetched_at.elapsed() < ttl);

    // If still at/over the cap, evict the oldest until there's room for one more.
    while cache.len() >= MAX_CACHE_ENTRIES {
        let Some(oldest) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.fetched_at)
            .map(|(k, _)| k.clone())
        else {
            break;
        };
        cache.remove(&oldest);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::service_entry::{encode_iroh, encode_quic};
    use crate::transport::QuicServerAuth;
    use serde_json::json;

    // ── did:web → URL derivation ──────────────────────────────────────────

    #[test]
    fn url_bare_host() {
        assert_eq!(
            did_web_to_url("did:web:example.com").unwrap(),
            "https://example.com/.well-known/did.json"
        );
    }

    #[test]
    fn url_bare_host_with_encoded_port() {
        // did:web:localhost%3A8443 → https://localhost:8443/.well-known/did.json
        assert_eq!(
            did_web_to_url("did:web:localhost%3A8443").unwrap(),
            "https://localhost:8443/.well-known/did.json"
        );
    }

    #[test]
    fn url_path_form() {
        assert_eq!(
            did_web_to_url("did:web:example.com:users:alice").unwrap(),
            "https://example.com/users/alice/did.json"
        );
    }

    #[test]
    fn url_path_form_with_encoded_port() {
        assert_eq!(
            did_web_to_url("did:web:127.0.0.1%3A6791:users:12345").unwrap(),
            "https://127.0.0.1:6791/users/12345/did.json"
        );
    }

    #[test]
    fn url_strips_fragment_and_query() {
        assert_eq!(
            did_web_to_url("did:web:example.com#mesh").unwrap(),
            "https://example.com/.well-known/did.json"
        );
        assert_eq!(
            did_web_to_url("did:web:example.com:users:alice?versionId=1").unwrap(),
            "https://example.com/users/alice/did.json"
        );
    }

    #[test]
    fn url_rejects_non_did_web() {
        assert!(did_web_to_url("did:key:z6Mk...").is_err());
        assert!(did_web_to_url("https://example.com").is_err());
        assert!(did_web_to_url("did:web:").is_err());
    }

    #[test]
    fn url_rejects_path_traversal() {
        // Encoded `/` in the host smuggles a path.
        assert!(did_web_to_url("did:web:victim.com:..%2F..%2Fsecret").is_err());
        // A path segment that decodes to `..` (traversal).
        assert!(did_web_to_url("did:web:victim.com:%2E%2E").is_err());
        assert!(did_web_to_url("did:web:victim.com:..").is_err());
        // A path segment that decodes to contain `/`.
        assert!(did_web_to_url("did:web:victim.com:a%2Fb").is_err());
        // A path segment that decodes to `.`.
        assert!(did_web_to_url("did:web:victim.com:%2E").is_err());
        // Backslash / NUL in a segment.
        assert!(did_web_to_url("did:web:victim.com:a%5Cb").is_err());
        assert!(did_web_to_url("did:web:victim.com:a%00b").is_err());
        // Host with an injected path separator.
        assert!(did_web_to_url("did:web:victim.com%2Fevil").is_err());
        // Sanity: an encoded-port host with a legit path still works (regression).
        assert_eq!(
            did_web_to_url("did:web:127.0.0.1%3A6791:users:12345").unwrap(),
            "https://127.0.0.1:6791/users/12345/did.json"
        );
    }

    // ── DID-doc → reach extraction + preference order ──────────────────────

    /// Build a DID document with the given (already-encoded) transport service
    /// entries, matching the `did_document.rs` builder's shape.
    fn doc_with_services(services: Vec<Value>) -> Value {
        json!({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": "did:web:example.com",
            "verificationMethod": [],
            "service": services,
        })
    }

    fn iroh_service(node_id: [u8; 32]) -> Value {
        json!({
            "id": "did:web:example.com#iroh",
            "type": "IrohTransport",
            "serviceEndpoint": encode_iroh(&node_id, &["https://relay.example".to_owned()], &["hyprstream-rpc/1", "moql"]),
        })
    }

    fn quic_service(uri: &str, hash: [u8; 32]) -> Value {
        let auth = QuicServerAuth::pinned(vec![hash]).unwrap();
        json!({
            "id": "did:web:example.com#quic",
            "type": "QuicTransport",
            "serviceEndpoint": encode_quic(uri, &auth, &["hyprstream-rpc/1", "moql"]),
        })
    }

    #[test]
    fn extract_orders_iroh_before_quic() {
        // Doc order: quic first, then iroh — preference must reorder iroh first.
        let doc = doc_with_services(vec![
            quic_service("https://10.0.0.1:4433", [1u8; 32]),
            iroh_service([7u8; 32]),
        ]);
        let entries = transport_entries(&doc);
        assert_eq!(entries.len(), 2);
        assert!(
            matches!(entries[0].config.endpoint, EndpointType::Iroh { .. }),
            "iroh must be preferred first"
        );
        assert!(matches!(entries[1].config.endpoint, EndpointType::Quic { .. }));
        // iroh entry carries the identity key; quic does not.
        assert_eq!(entries[0].identity_key, Some([7u8; 32]));
        assert_eq!(entries[1].identity_key, None);
    }

    #[test]
    fn extract_preserves_doc_order_within_class() {
        let doc = doc_with_services(vec![
            quic_service("https://10.0.0.1:4433", [1u8; 32]),
            quic_service("https://10.0.0.2:4433", [2u8; 32]),
        ]);
        let entries = transport_entries(&doc);
        assert_eq!(entries.len(), 2);
        match (&entries[0].config.endpoint, &entries[1].config.endpoint) {
            (EndpointType::Quic { addr: a, .. }, EndpointType::Quic { addr: b, .. }) => {
                assert_eq!(a.to_string(), "10.0.0.1:4433");
                assert_eq!(b.to_string(), "10.0.0.2:4433");
            }
            other => panic!("expected two Quic entries, got {other:?}"),
        }
    }

    #[test]
    fn extract_skips_non_transport_services() {
        let doc = doc_with_services(vec![
            json!({
                "id": "did:web:example.com#hyprstream",
                "type": "HyprstreamService",
                "serviceEndpoint": "https://example.com",
            }),
            json!({
                "id": "did:web:example.com#atproto_pds",
                "type": "AtprotoPersonalDataServer",
                "serviceEndpoint": "https://example.com",
            }),
            quic_service("https://10.0.0.1:4433", [1u8; 32]),
        ]);
        let entries = transport_entries(&doc);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].config.endpoint, EndpointType::Quic { .. }));
    }

    #[test]
    fn extract_no_transport_entries_is_empty() {
        // A doc with only non-transport services → empty (graceful, not panic/err).
        let doc = doc_with_services(vec![json!({
            "id": "did:web:example.com#hyprstream",
            "type": "HyprstreamService",
            "serviceEndpoint": "https://example.com",
        })]);
        assert!(transport_entries(&doc).is_empty());

        // A doc with no `service` array at all → empty.
        let doc = json!({ "id": "did:web:example.com", "verificationMethod": [] });
        assert!(transport_entries(&doc).is_empty());
    }

    #[test]
    fn extract_skips_undecodable_entry_gracefully() {
        // A QuicTransport with webpki=false AND no certHashes is undecodable
        // (no auth). It must be skipped, not panic, and the valid iroh remains.
        let doc = doc_with_services(vec![
            json!({
                "id": "did:web:example.com#quic",
                "type": "QuicTransport",
                "serviceEndpoint": { "uri": "https://10.0.0.1:4433", "webpki": false },
            }),
            iroh_service([7u8; 32]),
        ]);
        let entries = transport_entries(&doc);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].config.endpoint, EndpointType::Iroh { .. }));
    }

    #[test]
    fn extract_skips_onion_transport() {
        // OnionTransport is recognized but not yet dialable — it must be skipped
        // (logged) and the valid quic entry must remain.
        let doc = doc_with_services(vec![
            json!({
                "id": "did:web:example.com#onion",
                "type": "OnionTransport",
                "serviceEndpoint": { "uri": "abcdef.onion:443" },
            }),
            quic_service("https://10.0.0.1:4433", [1u8; 32]),
        ]);
        let entries = transport_entries(&doc);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].config.endpoint, EndpointType::Quic { .. }));
    }

    #[test]
    fn preferred_transport_filters_by_kind() {
        let doc = doc_with_services(vec![
            iroh_service([7u8; 32]),
            quic_service("https://10.0.0.1:4433", [1u8; 32]),
        ]);
        // None → overall top preference = iroh.
        assert!(matches!(
            preferred_transport(&doc, None).unwrap().endpoint,
            EndpointType::Iroh { .. }
        ));
        // Quic kind → the quic entry, skipping the (preferred) iroh.
        assert!(matches!(
            preferred_transport(&doc, Some(SocketKind::Quic)).unwrap().endpoint,
            EndpointType::Quic { .. }
        ));
        // A kind with no federated reach → None.
        assert!(preferred_transport(&doc, Some(SocketKind::Rep)).is_none());
    }

    // ── resolver over an injected fixture fetcher (no network) ─────────────

    struct FixtureFetcher {
        doc: Value,
    }

    #[async_trait]
    impl DidDocFetcher for FixtureFetcher {
        async fn fetch(&self, url: &str) -> Result<Value> {
            assert_eq!(url, "https://example.com/.well-known/did.json");
            Ok(self.doc.clone())
        }
    }

    #[tokio::test]
    async fn resolver_resolves_quic_from_fixture_doc() {
        let doc = doc_with_services(vec![
            iroh_service([7u8; 32]),
            quic_service("https://10.0.0.1:4433", [1u8; 32]),
        ]);
        let resolver = DidWebResolver::new(FixtureFetcher { doc });
        let cfg = resolver
            .resolve("did:web:example.com", SocketKind::Quic)
            .await
            .unwrap();
        match cfg.endpoint {
            EndpointType::Quic { addr, .. } => assert_eq!(addr.to_string(), "10.0.0.1:4433"),
            other => panic!("expected Quic, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolver_resolve_all_returns_preference_order() {
        let doc = doc_with_services(vec![
            quic_service("https://10.0.0.1:4433", [1u8; 32]),
            iroh_service([7u8; 32]),
        ]);
        let resolver = DidWebResolver::new(FixtureFetcher { doc });
        let all = resolver.resolve_all("did:web:example.com").await.unwrap();
        assert_eq!(all.len(), 2);
        assert!(matches!(all[0].config.endpoint, EndpointType::Iroh { .. }));
        assert!(matches!(all[1].config.endpoint, EndpointType::Quic { .. }));
    }

    #[tokio::test]
    async fn resolver_empty_doc_errors_for_kind() {
        let doc = doc_with_services(vec![]);
        let resolver = DidWebResolver::new(FixtureFetcher { doc });
        let res = resolver.resolve("did:web:example.com", SocketKind::Quic).await;
        assert!(res.is_err(), "no reach → resolve() errors (resolve_all would be empty)");
    }

    // ── chaining ───────────────────────────────────────────────────────────

    struct AlwaysOk(TransportConfig);
    #[async_trait]
    impl Resolver for AlwaysOk {
        async fn resolve(&self, _name: &str, _kind: SocketKind) -> Result<TransportConfig> {
            Ok(self.0.clone())
        }
    }
    struct AlwaysErr;
    #[async_trait]
    impl Resolver for AlwaysErr {
        async fn resolve(&self, _name: &str, _kind: SocketKind) -> Result<TransportConfig> {
            bail!("nope")
        }
    }

    #[tokio::test]
    async fn chained_falls_back_on_primary_error() {
        let fallback = TransportConfig::inproc("hyprstream/local");
        let chained = ChainedResolver::new(
            std::sync::Arc::new(AlwaysErr),
            std::sync::Arc::new(AlwaysOk(fallback.clone())),
        );
        let cfg = chained.resolve("anything", SocketKind::Rep).await.unwrap();
        assert_eq!(cfg.endpoint_string(), fallback.endpoint_string());
    }

    #[tokio::test]
    async fn chained_prefers_primary() {
        let primary = TransportConfig::inproc("hyprstream/primary");
        let fallback = TransportConfig::inproc("hyprstream/fallback");
        let chained = ChainedResolver::new(
            std::sync::Arc::new(AlwaysOk(primary.clone())),
            std::sync::Arc::new(AlwaysOk(fallback)),
        );
        let cfg = chained.resolve("anything", SocketKind::Rep).await.unwrap();
        assert_eq!(cfg.endpoint_string(), primary.endpoint_string());
    }

    // ── fetcher hardening (SSRF / DoS) ─────────────────────────────────────

    #[test]
    fn fetcher_constructs_with_hardened_client() {
        // The constructor returns Ok (client built with the redirect/timeout
        // config). A build failure would surface here rather than a silent
        // unwrap_or_default() yielding an unhardened client.
        let f = HttpDidDocFetcher::new(std::time::Duration::from_secs(1));
        assert!(f.is_ok(), "hardened client must build");
        // Default is also infallible by contract.
        let _ = HttpDidDocFetcher::default();
    }

    #[tokio::test]
    async fn fetcher_rejects_non_https() {
        let f = HttpDidDocFetcher::new(std::time::Duration::from_secs(1)).unwrap();
        let err = f.fetch("http://example.com/.well-known/did.json").await.unwrap_err();
        assert!(err.to_string().contains("https"), "must reject non-https: {err}");
    }

    #[test]
    fn push_capped_rejects_oversized_body() {
        // Accumulating past the cap (across chunks) must error at the boundary,
        // not buffer the whole oversized body (memory-DoS guard).
        let mut buf = Vec::new();
        // Fill to the cap exactly — OK.
        assert!(push_capped(&mut buf, &vec![b'x'; MAX_DID_DOC_BYTES], MAX_DID_DOC_BYTES).is_ok());
        assert_eq!(buf.len(), MAX_DID_DOC_BYTES);
        // One more byte crosses the ceiling — Err.
        let err = push_capped(&mut buf, b"!", MAX_DID_DOC_BYTES).unwrap_err();
        assert!(err.to_string().contains("cap"), "{err}");
        // Buffer was NOT grown past the cap.
        assert_eq!(buf.len(), MAX_DID_DOC_BYTES);
    }

    #[test]
    fn push_capped_accepts_under_cap() {
        let mut buf = Vec::new();
        push_capped(&mut buf, b"{\"ok\":", 16).unwrap();
        push_capped(&mut buf, b"true}", 16).unwrap();
        assert_eq!(buf, b"{\"ok\":true}");
    }

    #[test]
    fn cache_is_bounded() {
        let mut cache: std::collections::HashMap<String, CachedDoc> = std::collections::HashMap::new();
        let ttl = std::time::Duration::from_secs(3600);
        // Insert far more than the cap; eviction must keep it bounded.
        for i in 0..(MAX_CACHE_ENTRIES * 3) {
            evict_for_insert(&mut cache, ttl);
            cache.insert(
                format!("https://h{i}.example/.well-known/did.json"),
                CachedDoc {
                    doc: json!({ "i": i }),
                    fetched_at: std::time::Instant::now(),
                },
            );
            assert!(
                cache.len() <= MAX_CACHE_ENTRIES,
                "cache exceeded cap at i={i}: len={}",
                cache.len()
            );
        }
        assert!(cache.len() <= MAX_CACHE_ENTRIES);
    }

    // ── did:key (Ed25519) interop (#281) ───────────────────────────────────

    /// Encode raw Ed25519 bytes as an `ed25519-pub` Multikey `publicKeyMultibase`
    /// (`z` + base58btc(0xed01 ‖ key)). Mirrors `#280`'s `ed25519_to_multibase`
    /// and `mesh_trust::encode_multikey`; kept local to the test.
    fn ed25519_multikey(raw: &[u8; 32]) -> String {
        let mut payload = Vec::with_capacity(2 + 32);
        payload.extend_from_slice(&MULTICODEC_ED25519_PUB);
        payload.extend_from_slice(raw);
        format!("z{}", bs58::encode(payload).into_string())
    }

    fn rand_ed25519() -> [u8; 32] {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;
        SigningKey::generate(&mut OsRng).verifying_key().to_bytes()
    }

    #[test]
    fn did_key_decodes_known_fixture() {
        // The canonical did:key test vector for an all-zero Ed25519 public key
        // (multibase-base58btc of 0xed01 followed by 32 zero bytes). This is a
        // stable, spec-shaped fixture: the multicodec header + payload, not a
        // random key, so it pins the exact wire format we interoperate on.
        let did = ed25519_to_did_key(&[0u8; 32]);
        assert!(did.starts_with("did:key:z6Mk"), "ed25519 did:key must start with z6Mk: {did}");
        assert_eq!(did_key_to_ed25519(&did).unwrap(), [0u8; 32]);
    }

    #[test]
    fn did_key_roundtrips_random_key() {
        for _ in 0..16 {
            let raw = rand_ed25519();
            assert_eq!(did_key_to_ed25519(&ed25519_to_did_key(&raw)).unwrap(), raw);
        }
    }

    #[test]
    fn did_key_strips_fragment() {
        // A did:key with a fragment (self-referencing VM id) decodes the base key.
        let raw = [3u8; 32];
        let base = ed25519_to_did_key(&raw);
        let mb = base.strip_prefix("did:key:").unwrap();
        let with_fragment = format!("{base}#{mb}");
        assert_eq!(did_key_to_ed25519(&with_fragment).unwrap(), raw);
    }

    #[test]
    fn did_key_rejects_non_did_key() {
        assert!(did_key_to_ed25519("did:web:example.com").is_err());
        assert!(did_key_to_ed25519("https://example.com").is_err());
        assert!(did_key_to_ed25519("did:key:").is_err());
        assert!(!is_did_key("did:web:example.com"));
        assert!(is_did_key("did:key:z6MkfooBar"));
    }

    #[test]
    fn did_key_rejects_wrong_multicodec() {
        // A Multikey carrying a p256-pub multicodec (0x1200 → 0x80 0x24), not
        // ed25519-pub, must be rejected — only Ed25519 did:key is in scope.
        let mut payload = vec![0x80u8, 0x24];
        payload.extend_from_slice(&[0u8; 33]); // compressed p256-shaped body
        let did = format!("did:key:z{}", bs58::encode(payload).into_string());
        assert!(did_key_to_ed25519(&did).is_err());
    }

    #[test]
    fn did_key_rejects_bad_length() {
        // ed25519-pub multicodec but a 31-byte payload (wrong length).
        let mut payload = vec![0xedu8, 0x01];
        payload.extend_from_slice(&[0u8; 31]);
        let did = format!("did:key:z{}", bs58::encode(payload).into_string());
        assert!(did_key_to_ed25519(&did).is_err());
    }

    #[test]
    fn did_key_rejects_non_base58btc_multibase() {
        // 'b' is base32 multibase, not the base58btc 'z' Multikey uses.
        assert!(did_key_to_ed25519("did:key:bMkfoo").is_err());
    }

    #[test]
    fn did_key_equivalent_to_vm_multikey() {
        // Equivalence: our Ed25519 VM `publicKeyMultibase` (#280 form) is exactly
        // the method-specific id of the did:key for the same key — one multicodec
        // source of truth, no divergence between did:web VMs and did:key.
        let raw = rand_ed25519();
        let vm_multibase = ed25519_multikey(&raw); // == #280 ed25519_to_multibase
        let did_key = ed25519_to_did_key(&raw);
        assert_eq!(did_key, format!("did:key:{vm_multibase}"));
        // And both decode to the same key.
        assert_eq!(decode_ed25519_multikey(&vm_multibase).unwrap(), raw);
        assert_eq!(did_key_to_ed25519(&did_key).unwrap(), raw);
    }
}
