//! Federation admission gate over DID/key binding (#137, milestone M3).
//!
//! A transport-agnostic, two-stage **fail-closed** gate that decides whether an
//! inbound federated peer may be admitted over *any*
//! [`crate::transport_traits::Transport`]-backed session (QUIC/WebTransport,
//! iroh, …). It is deliberately independent of the concrete transport: the
//! accept path supplies the peer's already-authenticated material; this module
//! makes the trust decision.
//!
//! # The two stages (both must pass)
//!
//! 1. **Origin admission** — is the peer's *origin* (RFC 6454: scheme + host +
//!    non-default port) permitted by policy? This reuses the existing unified
//!    federation trust gate (Casbin `federation:register`, the CIMD /
//!    `FederationKeyResolver` decision point) via the [`OriginAdmission`] trait,
//!    implemented in the `hyprstream` crate over `PolicyService` (the lower
//!    `hyprstream-rpc` crate has no `PolicyService` client, so the live decision
//!    is injected). The gate calls it unchanged.
//!
//! 2. **Key binding** — does the peer's authenticated channel key (the raw
//!    Ed25519 public key it proved possession of when establishing the session /
//!    signing its envelopes) bind to the peer's DID? Two DID-method paths:
//!
//!    - **`did:web`** — match the key against a `verificationMethod` in the
//!      peer's *resolved* DID document (the `#mesh` / `#iroh` Ed25519 VM), OR an
//!      Ed25519 key in a federation JWKS the caller supplies. The DID document is
//!      resolved via the [`crate::did_web::DidWebResolver`] landed in #279 (no new
//!      fetch/cache infra); the VM extraction is
//!      [`crate::did_web::verification_method_ed25519_keys`] (#280 Multikey decode).
//!    - **`did:key` (Tiles interop, #281)** — a `did:key` is **self-certifying**:
//!      the Ed25519 key *is* the identity (`did:key:z6Mk…` =
//!      `multibase(0xed01 ‖ pubkey)`). There is **no document to resolve and no
//!      network fetch**; the gate decodes the key from the DID
//!      ([`crate::did_web::did_key_to_ed25519`]) and admits iff the peer's channel
//!      key equals it. Reach for such a peer comes from iroh discovery (#282), not
//!      the DID string.
//!
//! Either stage failing — or any resolution / I/O error — rejects (§4.4
//! fail-closed). A successful run yields an [`AdmittedIdentity`] binding the
//! origin to the matched key.
//!
//! # What is wired vs. what is a documented seam
//!
//! Stage 2's *logic* (resolve DID doc → extract VM keys → constant-time-ish
//! membership match, with JWKS fallback) is fully implemented and unit-tested
//! here against fixtures. The *live peer-key extraction at the transport accept
//! path* is the integration seam: see the module-level note in
//! [`PeerChannelKey`]. On the WebTransport/QUIC server accept path the client's
//! raw Ed25519 is **not** available today (no mTLS client-cert verifier; RFC 7250
//! raw-public-key binding is #200, not wired; iroh's live `node_id` is #282,
//! pending). The accept path can wire stage 1 (origin) immediately and feed
//! stage 2 the app-layer signed-envelope signer key (the architectural identity
//! root per `transport::QuicServerAuth` docs) once that key is surfaced to the
//! gate. Until then the live wiring carries a `TODO(#200/#185/#282)`.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::did_web::{
    did_key_to_ed25519, is_did_key, jwks_ed25519_keys, verification_method_ed25519_keys, DidDocFetcher,
    DidWebResolver,
};

/// The peer's authenticated channel key: the raw 32-byte Ed25519 public key the
/// connecting peer proved possession of for this session.
///
/// # Where this comes from (integration seam)
///
/// In this architecture peer *identity* is established at the **application
/// layer** — every response is a signed COSE envelope verified against the
/// peer's published keys (see [`crate::transport::QuicServerAuth`] docs). The
/// authenticated key is therefore the envelope **signer key** (`cnf`, 32 bytes),
/// which is the value to feed here. On transports where the channel itself binds
/// the identity (iroh `node_id`; RFC 7250 raw-public-key QUIC, #200) that key is
/// the same value and may be sourced from the channel instead.
///
/// This newtype keeps the gate honest about *which* bytes it is matching and
/// makes the seam explicit at every call site.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PeerChannelKey(pub [u8; 32]);

impl PeerChannelKey {
    /// The raw 32-byte Ed25519 public key.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Debug for PeerChannelKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Render a short fingerprint, not the full key, to keep logs tidy.
        write!(f, "PeerChannelKey({:02x}{:02x}…)", self.0[0], self.0[1])
    }
}

/// Stage 1: origin admission decision (RFC 6454 origin → permitted?).
///
/// Implemented in the `hyprstream` crate over `PolicyService`
/// (`federation:register`), the same unified federation trust gate used by CIMD
/// client registration and `FederationKeyResolver`. Abstracted as a trait so the
/// lower `hyprstream-rpc` crate (which has no `PolicyService` client) can run the
/// gate, and so the decision is independently testable.
///
/// **Fail-closed contract:** implementations MUST return `Ok(())` only when the
/// origin is affirmatively permitted, and `Err(_)` on denial **or** on any
/// inability to reach the policy decision (RPC outage) — never default-allow.
#[async_trait]
pub trait OriginAdmission: Send + Sync {
    /// Permit the given RFC 6454 origin, or return why it is rejected.
    async fn admit_origin(&self, origin: &str) -> Result<()>;
}

/// Resolves a peer DID identifier to its DID document JSON.
///
/// Implemented for [`DidWebResolver`] below (reusing #279's fetch+cache); a
/// fixture impl makes the gate testable without a network.
#[async_trait]
pub trait DidDocResolve: Send + Sync {
    /// Fetch and parse the DID document for `did`.
    async fn resolve_doc(&self, did: &str) -> Result<Value>;
}

#[async_trait]
impl<F: DidDocFetcher> DidDocResolve for DidWebResolver<F> {
    async fn resolve_doc(&self, did: &str) -> Result<Value> {
        // Reuse the resolver's derive-URL + cached fetch path (#279); we want the
        // raw document so stage 2 can read `verificationMethod`, which the
        // transport-only `resolve`/`resolve_all` discard.
        self.resolve_document(did).await
    }
}

/// The result of a successful two-stage admission: the peer's normalized origin
/// bound to the authenticated key that matched its published DID-doc VM / JWKS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedIdentity {
    /// The RFC 6454 origin that passed stage 1.
    pub origin: String,
    /// The peer DID identifier whose document carried the matched VM (when the
    /// match came from a DID document; `None` if matched via JWKS fallback).
    pub did: Option<String>,
    /// The authenticated Ed25519 key that matched (32 bytes).
    pub key: [u8; 32],
}

/// The federation admission gate: stage 1 (origin) then stage 2 (key binding),
/// fail-closed if **either** fails.
///
/// Transport-agnostic: construct it once with the origin-admission handle and a
/// DID-doc resolver, then call [`admit`](FederationAdmissionGate::admit) from
/// any transport accept path with the per-connection (origin, peer key, did,
/// optional federation JWKS).
pub struct FederationAdmissionGate<O: OriginAdmission, R: DidDocResolve> {
    origin_admission: O,
    resolver: R,
}

impl<O: OriginAdmission, R: DidDocResolve> FederationAdmissionGate<O, R> {
    /// Construct a gate over an origin-admission handle and a DID-doc resolver.
    pub fn new(origin_admission: O, resolver: R) -> Self {
        Self { origin_admission, resolver }
    }

    /// Run the two-stage gate for an inbound peer.
    ///
    /// - `origin` — the peer's RFC 6454 origin (caller extracts it from the peer
    ///   DID / advertised issuer with `extract_origin`).
    /// - `peer_key` — the peer's authenticated channel/envelope key (see
    ///   [`PeerChannelKey`]).
    /// - `did` — the peer's DID identifier for stage 2 (`did:web:…` resolved via
    ///   the DID-doc resolver, or `did:key:…` self-certifying / no fetch — #281).
    /// - `federation_jwks` — an optional already-resolved federation JWKS
    ///   document (e.g. from the existing `jwks_uri` cache) used as a **fallback**
    ///   key source when the DID document carries no matching Ed25519 VM. Reuses
    ///   the caller's JWKS cache; this module never fetches JWKS itself.
    ///
    /// Returns the [`AdmittedIdentity`] on success; `Err` (reject) if stage 1
    /// denies, if DID resolution fails, or if no VM/JWKS key matches the peer key.
    pub async fn admit(
        &self,
        origin: &str,
        peer_key: PeerChannelKey,
        did: &str,
        federation_jwks: Option<&Value>,
    ) -> Result<AdmittedIdentity> {
        // ── Stage 1: origin admission (fail-closed) ───────────────────────────
        // Run FIRST so a disallowed origin is rejected before we ever fetch its
        // DID document (no SSRF/work amplification for un-permitted peers).
        self.origin_admission
            .admit_origin(origin)
            .await
            .map_err(|e| anyhow!("admission stage 1 (origin {origin}) denied: {e}"))?;

        // ── Stage 2: key binding (fail-closed) ────────────────────────────────
        admit_key_against_did(&self.resolver, origin, peer_key, did, federation_jwks).await
    }
}

/// Stage 2 core, factored out so it is independently unit-testable: bind the
/// peer's authenticated channel key to its DID's published key material, and
/// admit iff it matches.
///
/// Two DID-method paths (chosen by the DID string):
///
/// - **`did:key` (self-certifying, #281)** — Tiles-style Ed25519 device/account
///   identity. The key *is* the identity: `did:key:z6Mk…` is exactly
///   `multibase(0xed01 ‖ pubkey)`, so there is **no document to fetch and no
///   resolver call**. We decode the key directly ([`did_key_to_ed25519`]) and
///   admit iff `peer_key` equals it. (Reach for such a peer comes from iroh
///   discovery #282, not the DID string — out of scope here.)
/// - **`did:web` (resolver fetch, #279/#137)** — resolve the peer DID document
///   and admit iff `peer_key` matches one of its Ed25519 `verificationMethod`
///   keys, falling back to a supplied federation JWKS.
///
/// Fail-closed: a parse error, a resolution error, an empty/absent VM set with no
/// JWKS match, or a key mismatch all return `Err`.
pub async fn admit_key_against_did<R: DidDocResolve>(
    resolver: &R,
    origin: &str,
    peer_key: PeerChannelKey,
    did: &str,
    federation_jwks: Option<&Value>,
) -> Result<AdmittedIdentity> {
    // ── did:key self-certifying arm (#281) ────────────────────────────────────
    // A did:key carries its own Ed25519 key — no DID-doc resolution / network
    // fetch. The peer's authenticated channel key MUST equal the key the DID
    // encodes; any mismatch or parse error fails closed.
    if is_did_key(did) {
        let did_key = did_key_to_ed25519(did)
            .map_err(|e| admission_reject(origin, peer_key, &format!("did:key {did} is invalid: {e}")))?;
        if key_eq(&did_key, peer_key.as_bytes()) {
            return Ok(AdmittedIdentity {
                origin: origin.to_owned(),
                did: Some(did.to_owned()),
                key: *peer_key.as_bytes(),
            });
        }
        return Err(admission_reject(
            origin,
            peer_key,
            &format!("peer key does not match the self-certifying did:key identity {did}"),
        ));
    }

    // ── did:web resolver arm (#279/#137) ──────────────────────────────────────
    // Resolve the peer's DID document. A resolution failure is fail-closed:
    // without the published key material we cannot bind the channel key.
    let doc = resolver
        .resolve_doc(did)
        .await
        .map_err(|e| admission_reject(origin, peer_key, &format!("DID {did} did not resolve: {e}")))?;

    let vm_keys = verification_method_ed25519_keys(&doc);
    if vm_keys.iter().any(|k| key_eq(k, peer_key.as_bytes())) {
        return Ok(AdmittedIdentity {
            origin: origin.to_owned(),
            did: Some(did.to_owned()),
            key: *peer_key.as_bytes(),
        });
    }

    // Fallback: a federation-tagged JWKS the caller already resolved/cached.
    if let Some(jwks) = federation_jwks {
        if jwks_ed25519_keys(jwks).iter().any(|k| key_eq(k, peer_key.as_bytes())) {
            return Ok(AdmittedIdentity {
                origin: origin.to_owned(),
                // Matched via JWKS, not a specific DID-doc VM.
                did: None,
                key: *peer_key.as_bytes(),
            });
        }
    }

    Err(admission_reject(
        origin,
        peer_key,
        &format!(
            "peer key does not match any Ed25519 verificationMethod in DID doc for {did}{}",
            if federation_jwks.is_some() { " (nor the federation JWKS)" } else { "" }
        ),
    ))
}

/// Constant-time-ish equality for two 32-byte keys.
///
/// The keys here are *public* (no secret to leak), so a data-dependent compare is
/// not a vulnerability; we still fold over all bytes to avoid an early-exit habit
/// leaking into copies of this matcher elsewhere.
fn key_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Build a uniform rejection error (also a single tracing point so denials are
/// diagnosable without leaking the full key).
fn admission_reject(origin: &str, key: PeerChannelKey, why: &str) -> anyhow::Error {
    tracing::warn!(origin = %origin, peer_key = ?key, "federation admission stage 2 rejected: {why}");
    anyhow!("admission stage 2 (origin {origin}) rejected: {why}")
}

/// Convenience: the rejection used when stage 2 is reached without a live peer
/// key (the documented #200/#185/#282 seam). Callers on a transport that cannot
/// yet surface the peer's authenticated Ed25519 MUST reject rather than admit.
pub fn reject_no_peer_key(origin: &str) -> anyhow::Error {
    anyhow!(
        "admission stage 2 (origin {origin}): peer channel key not available at this accept path \
         (RFC 7250 raw-public-key #200 / iroh node_id #282 not wired) — failing closed"
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::did_web::decode_ed25519_multikey;
    use anyhow::bail;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use serde_json::json;

    /// Encode raw Ed25519 bytes as an `ed25519-pub` Multikey `publicKeyMultibase`
    /// (`z` + base58btc(multicodec || key)). Mirrors `mesh_trust::encode_multikey`
    /// for the Ed25519 codec; kept local to the test (no `hyprstream` dep).
    fn ed25519_multikey(raw: &[u8; 32]) -> String {
        let mut payload = Vec::with_capacity(2 + 32);
        payload.extend_from_slice(&[0xed, 0x01]);
        payload.extend_from_slice(raw);
        format!("z{}", bs58::encode(payload).into_string())
    }

    fn random_ed25519() -> [u8; 32] {
        SigningKey::generate(&mut OsRng).verifying_key().to_bytes()
    }

    /// Decode-roundtrip sanity: our local Multikey encode is the inverse of the
    /// did_web decoder (same wire format as `mesh_trust`).
    #[test]
    fn multikey_roundtrip() {
        let raw = random_ed25519();
        let mb = ed25519_multikey(&raw);
        assert_eq!(decode_ed25519_multikey(&mb).unwrap(), raw);
    }

    fn did_doc_with_vm(keys: &[[u8; 32]]) -> Value {
        let vms: Vec<Value> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| {
                json!({
                    "id": format!("did:web:peer.example#key-{i}"),
                    "type": "Multikey",
                    "controller": "did:web:peer.example",
                    "publicKeyMultibase": ed25519_multikey(k),
                })
            })
            .collect();
        json!({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": "did:web:peer.example",
            "verificationMethod": vms,
            "service": [],
        })
    }

    fn jwks_with_keys(keys: &[[u8; 32]]) -> Value {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
        let ks: Vec<Value> = keys
            .iter()
            .map(|k| json!({ "kty": "OKP", "crv": "Ed25519", "x": URL_SAFE_NO_PAD.encode(k) }))
            .collect();
        json!({ "keys": ks })
    }

    // ── Test doubles ───────────────────────────────────────────────────────

    struct AllowOrigin;
    #[async_trait]
    impl OriginAdmission for AllowOrigin {
        async fn admit_origin(&self, _origin: &str) -> Result<()> {
            Ok(())
        }
    }

    struct DenyOrigin;
    #[async_trait]
    impl OriginAdmission for DenyOrigin {
        async fn admit_origin(&self, origin: &str) -> Result<()> {
            bail!("origin {origin} not permitted by policy (federation:register denied)")
        }
    }

    struct FixtureDoc(Value);
    #[async_trait]
    impl DidDocResolve for FixtureDoc {
        async fn resolve_doc(&self, _did: &str) -> Result<Value> {
            Ok(self.0.clone())
        }
    }

    struct FailingResolve;
    #[async_trait]
    impl DidDocResolve for FailingResolve {
        async fn resolve_doc(&self, did: &str) -> Result<Value> {
            bail!("simulated resolution failure for {did}")
        }
    }

    const DID: &str = "did:web:peer.example";
    const ORIGIN: &str = "https://peer.example";

    // ── Stage 2: extractor sanity ────────────────────────────────────────────

    #[test]
    fn vm_extraction_returns_published_ed25519_keys() {
        let k1 = random_ed25519();
        let k2 = random_ed25519();
        let doc = did_doc_with_vm(&[k1, k2]);
        let got = verification_method_ed25519_keys(&doc);
        assert_eq!(got, vec![k1, k2]);
    }

    #[test]
    fn vm_extraction_skips_undecodable_and_non_ed25519() {
        let good = random_ed25519();
        let doc = json!({
            "verificationMethod": [
                // wrong multibase prefix
                { "id": "#a", "publicKeyMultibase": "Qbogus" },
                // a publicKeyJwk-only VM (no multibase) → skipped
                { "id": "#b", "publicKeyJwk": { "kty": "OKP", "crv": "Ed25519", "x": "AAAA" } },
                // valid ed25519 Multikey
                { "id": "#c", "publicKeyMultibase": ed25519_multikey(&good) },
            ],
        });
        assert_eq!(verification_method_ed25519_keys(&doc), vec![good]);
    }

    #[test]
    fn vm_extraction_empty_when_no_vm() {
        assert!(verification_method_ed25519_keys(&json!({ "id": DID })).is_empty());
        assert!(verification_method_ed25519_keys(&json!({ "verificationMethod": [] })).is_empty());
    }

    // ── Two-stage gate behavior ──────────────────────────────────────────────

    #[tokio::test]
    async fn admits_when_peer_key_matches_did_vm() {
        let key = random_ed25519();
        let gate = FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[key])));
        let admitted = gate
            .admit(ORIGIN, PeerChannelKey(key), DID, None)
            .await
            .expect("matching VM must admit");
        assert_eq!(admitted.origin, ORIGIN);
        assert_eq!(admitted.did.as_deref(), Some(DID));
        assert_eq!(admitted.key, key);
    }

    #[tokio::test]
    async fn rejects_when_peer_key_mismatches_did_vm() {
        // Doc publishes a different key than the peer presents.
        let published = random_ed25519();
        let peer = random_ed25519();
        let gate = FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[published])));
        let err = gate
            .admit(ORIGIN, PeerChannelKey(peer), DID, None)
            .await
            .expect_err("mismatched key must reject");
        assert!(err.to_string().contains("stage 2"), "{err}");
    }

    #[tokio::test]
    async fn rejects_when_did_doc_has_no_matching_vm() {
        // Doc has no verificationMethod at all → no key to match → reject.
        let peer = random_ed25519();
        let doc = json!({ "id": DID, "verificationMethod": [], "service": [] });
        let gate = FederationAdmissionGate::new(AllowOrigin, FixtureDoc(doc));
        let err = gate
            .admit(ORIGIN, PeerChannelKey(peer), DID, None)
            .await
            .expect_err("no VM must reject");
        assert!(err.to_string().contains("stage 2"), "{err}");
    }

    #[tokio::test]
    async fn rejects_when_origin_denied_before_stage2() {
        // DenyOrigin rejects at stage 1; the (otherwise matching) DID doc is
        // never even consulted. Use a FailingResolve to prove stage 2 isn't run:
        // if it were, the error would be a resolution failure, not a stage-1 one.
        let peer = random_ed25519();
        let gate = FederationAdmissionGate::new(DenyOrigin, FailingResolve);
        let err = gate
            .admit(ORIGIN, PeerChannelKey(peer), DID, None)
            .await
            .expect_err("denied origin must reject");
        assert!(err.to_string().contains("stage 1"), "expected stage-1 rejection, got: {err}");
        assert!(!err.to_string().contains("did not resolve"), "stage 2 must not run: {err}");
    }

    #[tokio::test]
    async fn rejects_when_did_resolution_fails() {
        let peer = random_ed25519();
        let gate = FederationAdmissionGate::new(AllowOrigin, FailingResolve);
        let err = gate
            .admit(ORIGIN, PeerChannelKey(peer), DID, None)
            .await
            .expect_err("resolution failure must reject (fail-closed)");
        assert!(err.to_string().contains("did not resolve"), "{err}");
    }

    // ── JWKS fallback path ───────────────────────────────────────────────────

    #[tokio::test]
    async fn admits_via_jwks_fallback_when_no_did_vm_match() {
        // DID doc has a non-matching VM; the federation JWKS carries the peer key.
        let published = random_ed25519();
        let peer = random_ed25519();
        let jwks = jwks_with_keys(&[peer]);
        let gate = FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[published])));
        let admitted = gate
            .admit(ORIGIN, PeerChannelKey(peer), DID, Some(&jwks))
            .await
            .expect("JWKS fallback must admit");
        // Matched via JWKS, not a specific DID-doc VM.
        assert_eq!(admitted.did, None);
        assert_eq!(admitted.key, peer);
    }

    #[tokio::test]
    async fn jwks_fallback_still_rejects_unknown_key() {
        let published = random_ed25519();
        let jwks_key = random_ed25519();
        let peer = random_ed25519(); // in neither the doc nor the JWKS
        let jwks = jwks_with_keys(&[jwks_key]);
        let gate = FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[published])));
        let err = gate
            .admit(ORIGIN, PeerChannelKey(peer), DID, Some(&jwks))
            .await
            .expect_err("key in neither source must reject");
        assert!(err.to_string().contains("nor the federation JWKS"), "{err}");
    }

    #[test]
    fn no_peer_key_helper_is_fail_closed() {
        let err = reject_no_peer_key(ORIGIN);
        assert!(err.to_string().contains("failing closed"), "{err}");
        assert!(err.to_string().contains("#200"), "{err}");
    }

    // ── did:key self-certifying arm (#281) ───────────────────────────────────

    use crate::did_web::ed25519_to_did_key;

    /// A resolver that MUST NOT be called: the did:key arm is self-certifying and
    /// must never fetch/resolve. If the gate calls it, the test fails loudly.
    struct NeverResolve;
    #[async_trait]
    impl DidDocResolve for NeverResolve {
        async fn resolve_doc(&self, did: &str) -> Result<Value> {
            panic!("did:key admission must not resolve a DID document (called for {did})");
        }
    }

    #[tokio::test]
    async fn admits_did_key_peer_whose_channel_key_matches() {
        // Self-certifying: the peer's channel key equals the key the did:key
        // encodes. No DID-doc resolution — NeverResolve proves the gate never
        // calls the resolver for a did:key.
        let key = random_ed25519();
        let did = ed25519_to_did_key(&key);
        let gate = FederationAdmissionGate::new(AllowOrigin, NeverResolve);
        let admitted = gate
            .admit(ORIGIN, PeerChannelKey(key), &did, None)
            .await
            .expect("matching did:key must admit (self-certifying)");
        assert_eq!(admitted.origin, ORIGIN);
        assert_eq!(admitted.did.as_deref(), Some(did.as_str()));
        assert_eq!(admitted.key, key);
    }

    #[tokio::test]
    async fn rejects_did_key_peer_on_key_mismatch() {
        // The peer presents a different key than the did:key identity encodes.
        let identity = random_ed25519();
        let peer = random_ed25519();
        let did = ed25519_to_did_key(&identity);
        let gate = FederationAdmissionGate::new(AllowOrigin, NeverResolve);
        let err = gate
            .admit(ORIGIN, PeerChannelKey(peer), &did, None)
            .await
            .expect_err("mismatched did:key must reject");
        assert!(err.to_string().contains("stage 2"), "{err}");
        assert!(err.to_string().contains("self-certifying"), "{err}");
    }

    #[tokio::test]
    async fn did_key_still_enforces_stage1_origin() {
        // Even a self-certifying did:key whose key would match must be rejected
        // when stage 1 (origin) denies — stage 1 runs first and the did:key arm
        // is never reached (NeverResolve also guards against any resolve attempt).
        let key = random_ed25519();
        let did = ed25519_to_did_key(&key);
        let gate = FederationAdmissionGate::new(DenyOrigin, NeverResolve);
        let err = gate
            .admit(ORIGIN, PeerChannelKey(key), &did, None)
            .await
            .expect_err("denied origin must reject a did:key peer");
        assert!(err.to_string().contains("stage 1"), "expected stage-1 rejection, got: {err}");
    }

    #[tokio::test]
    async fn rejects_invalid_did_key_fail_closed() {
        // A malformed did:key (ed25519-pub multicodec absent / wrong) must fail
        // closed without resolving.
        let peer = random_ed25519();
        let bad_did = "did:key:zNotAValidEd25519Multikey";
        let gate = FederationAdmissionGate::new(AllowOrigin, NeverResolve);
        let err = gate
            .admit(ORIGIN, PeerChannelKey(peer), bad_did, None)
            .await
            .expect_err("invalid did:key must reject (fail-closed)");
        assert!(err.to_string().contains("stage 2"), "{err}");
    }
}
