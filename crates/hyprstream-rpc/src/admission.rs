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
//! 2. **Key binding** — does the independently verified application-envelope
//!    signer key bind to the peer's DID? Carrier keys are never accepted here.
//!    Two DID-method paths:
//!
//!    - **`did:web`** — match the key against a `verificationMethod` in the
//!      peer's *resolved* DID document (for example `#mesh`), OR an
//!      Ed25519 key in a federation JWKS the caller supplies. The DID document is
//!      resolved via the [`crate::did_web::DidWebResolver`] landed in #279 (no new
//!      fetch/cache infra); the VM extraction is
//!      [`crate::did_web::verification_method_ed25519_keys`] (#280 Multikey decode).
//!    - **`did:key` (Tiles interop, #281)** — a `did:key` is **self-certifying**:
//!      the Ed25519 key *is* the identity (`did:key:z6Mk…` =
//!      `multibase(0xed01 ‖ pubkey)`). There is **no document to resolve and no
//!      network fetch**; the gate decodes the key from the DID
//!      ([`crate::did_web::did_key_to_ed25519`]) and admits iff the application
//!      signer key equals it. Reach is resolved separately, never from this DID.
//!
//! Either stage failing — or any resolution / I/O error — rejects (§4.4
//! fail-closed). A successful run yields an [`AdmittedIdentity`] binding the
//! origin to the matched key.
//!
//! # What is wired vs. what is a documented seam
//!
//! Stage 2's *logic* (resolve DID doc → extract VM keys → constant-time-ish
//! membership match, with JWKS fallback) is fully implemented and unit-tested
//! here against fixtures. Live callers must feed stage 2 only the signer key from
//! a verified application envelope. A NodeId, EndpointId, TLS certificate key,
//! endpoint field, or caller assertion is not a [`VerifiedApplicationSigner`]. If a
//! network path has no verified application proof, it must not invoke this gate
//! as though carrier establishment supplied one.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::auth::mac::VerifiedKeyMaterial;
use crate::did_web::{
    did_key_to_ed25519, is_did_key, jwks_ed25519_keys, verification_method_ed25519_keys,
    DidDocFetcher, DidWebResolver,
};
use crate::envelope::KeyedPqTrustStore;
use crate::identity::DID_AT9P_PREFIX;
use crate::identity_resolver::At9pCapsuleResolver;

/// Whether `did` is a `did:at9p` identifier (the self-certifying hybrid-PQC arm).
///
/// Delegates to the shared [`DID_AT9P_PREFIX`] literal (the single source of
/// truth shared with [`crate::identity::Did::is_did_at9p`] and `hyprstream-pds`'s
/// GATE code) so the prefix cannot drift across arms (#964).
fn is_did_at9p(did: &str) -> bool {
    did.starts_with(DID_AT9P_PREFIX)
}

/// Inputs for the `did:at9p` admission arm (D2/#894).
///
/// Parallel to the `is_did_key` arm, but a `did:at9p` proof upgrades a verified
/// application signer to a hybrid-PQC anchor, so the arm additionally needs:
/// - the **capsule bytes** carried with the application proof,
/// - a **GATE verifier** ([`At9pCapsuleResolver`]) that runs the A4
///   canon→hash→sig pipeline (#884) and returns the content-verified subject keys,
/// - the **[`KeyedPqTrustStore`]** to bind those verified hybrid keys into — the
///   atomic `ed25519 → ml_dsa_65` binding that replaces the out-of-band
///   `mesh_peers` config path for `did:at9p` peers.
///
/// The arm is exercised by passing `Some(At9pAdmission { .. })` to
/// [`admit_key_against_did`]; the live mesh admission flow is not wired beyond
/// this arm, so [`FederationAdmissionGate::admit`] passes `None` (a `did:at9p`
/// peer reaching that path fails closed until the accept path feeds capsule bytes).
pub struct At9pAdmission<'a> {
    /// The raw genesis capsule bytes the peer presented for `did:at9p:<cid512>`.
    pub capsule_bytes: &'a [u8],
    /// The GATE verifier (implemented in `hyprstream-pds` over `verify_did_at9p`).
    pub gate: &'a dyn At9pCapsuleResolver,
    /// The native PQ trust store to bind the verified hybrid keys into.
    pub pq_store: &'a mut KeyedPqTrustStore,
}

/// The raw Ed25519 signer key from independently verified application proof.
///
/// # Where this comes from (integration seam)
///
/// Peer identity is established at the application layer. The key here is the
/// verified COSE envelope signer (`cnf`, 32 bytes) after the surrounding proof
/// policy succeeds. It must never be sourced from iroh NodeId/EndpointId or any
/// other carrier key, even when the bytes happen to be equal.
///
/// This newtype keeps the gate honest about *which* bytes it is matching and
/// makes the seam explicit at every call site.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ApplicationSignerKey(pub [u8; 32]);

impl ApplicationSignerKey {
    /// The raw 32-byte Ed25519 public key.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Debug for ApplicationSignerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Render a short fingerprint, not the full key, to keep logs tidy.
        write!(
            f,
            "ApplicationSignerKey({:02x}{:02x}…)",
            self.0[0], self.0[1]
        )
    }
}

/// The trust boundary on which admission is being attempted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionTrustSurface {
    /// A native/SPINE trust surface. Classical-only proof is a downgrade and
    /// must fail closed.
    Native,
    /// A named third-party interoperability perimeter where a foreign identity
    /// may only provide classical signing material.
    ThirdPartyInterop,
}

/// Application signer plus the assurance actually established by envelope
/// verification. This is crypto-derived evidence, never a DID-document claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedApplicationSigner {
    key: ApplicationSignerKey,
    key_material: VerifiedKeyMaterial,
}

impl VerifiedApplicationSigner {
    /// A signer whose Ed25519 + bound ML-DSA-65 composite verified.
    pub fn pq_hybrid(key: [u8; 32]) -> Self {
        Self {
            key: ApplicationSignerKey(key),
            key_material: VerifiedKeyMaterial::PqHybrid,
        }
    }

    /// A classical signer verified specifically at a third-party interop edge.
    ///
    /// The deliberately long name prevents a generic/native caller from
    /// selecting classical verification as an accidental default.
    pub fn third_party_interop_classical(key: [u8; 32]) -> Self {
        Self {
            key: ApplicationSignerKey(key),
            key_material: VerifiedKeyMaterial::Classical,
        }
    }

    pub fn key(&self) -> ApplicationSignerKey {
        self.key
    }

    pub fn key_material(&self) -> VerifiedKeyMaterial {
        self.key_material
    }
}

fn require_key_assurance(
    origin: &str,
    signer: VerifiedApplicationSigner,
    surface: AdmissionTrustSurface,
) -> Result<ApplicationSignerKey> {
    match (surface, signer.key_material()) {
        (AdmissionTrustSurface::Native, VerifiedKeyMaterial::PqHybrid)
        | (
            AdmissionTrustSurface::ThirdPartyInterop,
            VerifiedKeyMaterial::Classical | VerifiedKeyMaterial::PqHybrid,
        ) => Ok(signer.key()),
        (AdmissionTrustSurface::Native, _) => Err(admission_reject(
            origin,
            signer.key(),
            "native/SPINE admission requires verified PQ-hybrid application proof; classical Ed25519 alone is a downgrade",
        )),
        (_, VerifiedKeyMaterial::Unverified) => Err(admission_reject(
            origin,
            signer.key(),
            "application signer proof is unverified",
        )),
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

/// A configured `did:plc` resolver can serve the existing federation admission
/// boundary. Its URL derivation, egress allowlist, cache, and `doc.id` binding
/// are all enforced before this gate inspects verification methods.
#[async_trait]
impl<F: crate::did_plc::PlcAuditFetcher> DidDocResolve for crate::did_plc::DidPlcResolver<F> {
    async fn resolve_doc(&self, did: &str) -> Result<Value> {
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
/// any transport accept path with the per-connection (origin, application signer key, did,
/// optional federation JWKS).
pub struct FederationAdmissionGate<O: OriginAdmission, R: DidDocResolve> {
    origin_admission: O,
    resolver: R,
}

impl<O: OriginAdmission, R: DidDocResolve> FederationAdmissionGate<O, R> {
    /// Construct a gate over an origin-admission handle and a DID-doc resolver.
    pub fn new(origin_admission: O, resolver: R) -> Self {
        Self {
            origin_admission,
            resolver,
        }
    }

    /// Run the two-stage gate for an inbound peer.
    ///
    /// - `origin` — the peer's RFC 6454 origin (caller extracts it from the peer
    ///   DID / advertised issuer with `extract_origin`).
    /// - `application_signer` — the verified application-envelope signer key and assurance (see
    ///   [`VerifiedApplicationSigner`]).
    /// - `trust_surface` — explicit native/SPINE vs third-party interop
    ///   classification. Native admission requires PQ-hybrid proof.
    /// - `did` — the peer's DID identifier for stage 2 (`did:web:…` resolved via
    ///   the DID-doc resolver, or `did:key:…` self-certifying / no fetch — #281).
    /// - `federation_jwks` — an optional already-resolved federation JWKS
    ///   document (e.g. from the existing `jwks_uri` cache) used as a **fallback**
    ///   key source when the DID document carries no matching Ed25519 VM. Reuses
    ///   the caller's JWKS cache; this module never fetches JWKS itself.
    ///
    /// Returns the [`AdmittedIdentity`] on success; `Err` (reject) if stage 1
    /// denies, if DID resolution fails, or if no VM/JWKS key matches the application signer key.
    pub async fn admit(
        &self,
        origin: &str,
        application_signer: VerifiedApplicationSigner,
        trust_surface: AdmissionTrustSurface,
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
        admit_key_against_did(
            &self.resolver,
            origin,
            application_signer,
            trust_surface,
            did,
            federation_jwks,
            None,
        )
        .await
    }
}

/// Stage 2 core, factored out so it is independently unit-testable: bind the
/// verified application signer key to its DID's published key material, and
/// admit iff it matches.
///
/// The assurance floor is enforced before DID dispatch: native/SPINE surfaces
/// require verified PQ-hybrid proof; classical proof is accepted only when the
/// caller explicitly names [`AdmissionTrustSurface::ThirdPartyInterop`].
///
/// Three DID-method paths (chosen by the DID string):
///
/// - **`did:key` (self-certifying, #281)** — Tiles-style Ed25519 device/account
///   identity. The key *is* the identity: `did:key:z6Mk…` is exactly
///   `multibase(0xed01 ‖ pubkey)`, so there is **no document to fetch and no
///   resolver call**. We decode the key directly ([`did_key_to_ed25519`]) and
///   admit iff `application_signer_key` equals it. Reach is an independent resolver output.
/// - **`did:at9p` (self-certifying hybrid-PQC, #894/D2)** — the capsule *is* the
///   hybrid key commitment: GATE-verify the peer-presented bytes
///   (canon→hash→sig, #884) via the supplied [`At9pCapsuleResolver`], admit iff
///   the verified application signer equals the capsule's Ed25519 subject key,
///   and bind
///   the verified `ed25519 → ml_dsa_65` pair into the [`KeyedPqTrustStore`].
///   This is the classical→hybrid trust upgrade, populated from content-verified
///   material (no config trust). No resolver/fetch: the capsule is
///   self-certifying. Requires `at9p = Some(_)`; otherwise fail-closed.
/// - **`did:web` (resolver fetch, #279/#137)** — resolve the peer DID document
///   and admit iff `application_signer_key` matches one of its Ed25519 `verificationMethod`
///   keys, falling back to a supplied federation JWKS.
///
/// Fail-closed: a parse error, a resolution error, an empty/absent VM set with no
/// JWKS match, or a key mismatch all return `Err`.
pub async fn admit_key_against_did<R: DidDocResolve>(
    resolver: &R,
    origin: &str,
    application_signer: VerifiedApplicationSigner,
    trust_surface: AdmissionTrustSurface,
    did: &str,
    federation_jwks: Option<&Value>,
    at9p: Option<At9pAdmission<'_>>,
) -> Result<AdmittedIdentity> {
    let application_signer_key = require_key_assurance(origin, application_signer, trust_surface)?;
    // ── did:key self-certifying arm (#281) ────────────────────────────────────
    // A did:key carries its own Ed25519 key — no DID-doc resolution / network
    // fetch. The verified application signer MUST equal the key the DID
    // encodes; any mismatch or parse error fails closed.
    if is_did_key(did) {
        let did_key = did_key_to_ed25519(did).map_err(|e| {
            admission_reject(
                origin,
                application_signer_key,
                &format!("did:key {did} is invalid: {e}"),
            )
        })?;
        if key_eq(&did_key, application_signer_key.as_bytes()) {
            return Ok(AdmittedIdentity {
                origin: origin.to_owned(),
                did: Some(did.to_owned()),
                key: *application_signer_key.as_bytes(),
            });
        }
        return Err(admission_reject(
            origin,
            application_signer_key,
            &format!(
                "application signer key does not match the self-certifying did:key identity {did}"
            ),
        ));
    }

    // ── did:at9p self-certifying hybrid arm (#894/D2) ──────────────────────────
    // The capsule proves the ed25519→ml_dsa_65 binding (GATE pipeline #884), so no
    // DID-doc fetch and no out-of-band config trust: the binding is installed
    // straight from content-verified material. A GATE failure rejects and binds
    // nothing; a missing admission context rejects too (the live accept path does
    // not yet feed capsule bytes — `FederationAdmissionGate::admit` passes None).
    if is_did_at9p(did) {
        let ctx = at9p.ok_or_else(|| {
            admission_reject(
                origin,
                application_signer_key,
                &format!(
                    "did:at9p {did} presented without a capsule/admission context — fail-closed"
                ),
            )
        })?;
        let verified = ctx.gate.verify_bytes(did, ctx.capsule_bytes).map_err(|e| {
            admission_reject(
                origin,
                application_signer_key,
                &format!("did:at9p {did} capsule GATE rejected: {e}"),
            )
        })?;
        // The capsule's Ed25519 subject key is genesis identity material; the
        // separately verified application signer must equal it. This comparison
        // says nothing about carrier reach or liveness.
        if !key_eq(verified.ed25519(), application_signer_key.as_bytes()) {
            return Err(admission_reject(
                origin,
                application_signer_key,
                &format!(
                    "application signer key does not match the did:at9p capsule Ed25519 subject key {did}"
                ),
            ));
        }
        // Atomic hybrid binding from the verified capsule — replaces the
        // out-of-band `mesh_peers` config path for did:at9p peers (#894).
        ctx.pq_store.bind(*verified.ed25519(), verified.ml_dsa_65());
        return Ok(AdmittedIdentity {
            origin: origin.to_owned(),
            did: Some(did.to_owned()),
            key: *application_signer_key.as_bytes(),
        });
    }

    // ── did:web resolver arm (#279/#137) ──────────────────────────────────────
    // Resolve the peer's DID document. A resolution failure is fail-closed:
    // without the published key material we cannot bind the application signer.
    let doc = resolver.resolve_doc(did).await.map_err(|e| {
        admission_reject(
            origin,
            application_signer_key,
            &format!("DID {did} did not resolve: {e}"),
        )
    })?;

    let vm_keys = verification_method_ed25519_keys(&doc);
    if vm_keys
        .iter()
        .any(|k| key_eq(k, application_signer_key.as_bytes()))
    {
        return Ok(AdmittedIdentity {
            origin: origin.to_owned(),
            did: Some(did.to_owned()),
            key: *application_signer_key.as_bytes(),
        });
    }

    // Fallback: a federation-tagged JWKS the caller already resolved/cached.
    if let Some(jwks) = federation_jwks {
        if jwks_ed25519_keys(jwks)
            .iter()
            .any(|k| key_eq(k, application_signer_key.as_bytes()))
        {
            return Ok(AdmittedIdentity {
                origin: origin.to_owned(),
                // Matched via JWKS, not a specific DID-doc VM.
                did: None,
                key: *application_signer_key.as_bytes(),
            });
        }
    }

    Err(admission_reject(
        origin,
        application_signer_key,
        &format!(
            "application signer key does not match any Ed25519 verificationMethod in DID doc for {did}{}",
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
fn admission_reject(origin: &str, key: ApplicationSignerKey, why: &str) -> anyhow::Error {
    tracing::warn!(origin = %origin, application_signer_key = ?key, "federation admission stage 2 rejected: {why}");
    anyhow!("admission stage 2 (origin {origin}) rejected: {why}")
}

/// Convenience rejection for a network path lacking verified application proof.
pub fn reject_no_application_signer_proof(origin: &str) -> anyhow::Error {
    anyhow!(
        "admission stage 2 (origin {origin}): verified application signer proof is unavailable; \
         carrier NodeId/EndpointId is not an admission key — failing closed"
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

    fn interop_signer(key: [u8; 32]) -> VerifiedApplicationSigner {
        VerifiedApplicationSigner::third_party_interop_classical(key)
    }

    fn hybrid_signer(key: [u8; 32]) -> VerifiedApplicationSigner {
        VerifiedApplicationSigner::pq_hybrid(key)
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
    async fn admits_when_application_signer_key_matches_did_vm() {
        let key = random_ed25519();
        let gate = FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[key])));
        let admitted = gate
            .admit(
                ORIGIN,
                interop_signer(key),
                AdmissionTrustSurface::ThirdPartyInterop,
                DID,
                None,
            )
            .await
            .expect("matching VM must admit");
        assert_eq!(admitted.origin, ORIGIN);
        assert_eq!(admitted.did.as_deref(), Some(DID));
        assert_eq!(admitted.key, key);
    }

    #[tokio::test]
    async fn rejects_when_application_signer_key_mismatches_did_vm() {
        // Doc publishes a different key than the peer presents.
        let published = random_ed25519();
        let peer = random_ed25519();
        let gate =
            FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[published])));
        let err = gate
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                DID,
                None,
            )
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
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                DID,
                None,
            )
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
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                DID,
                None,
            )
            .await
            .expect_err("denied origin must reject");
        assert!(
            err.to_string().contains("stage 1"),
            "expected stage-1 rejection, got: {err}"
        );
        assert!(
            !err.to_string().contains("did not resolve"),
            "stage 2 must not run: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_when_did_resolution_fails() {
        let peer = random_ed25519();
        let gate = FederationAdmissionGate::new(AllowOrigin, FailingResolve);
        let err = gate
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                DID,
                None,
            )
            .await
            .expect_err("resolution failure must reject (fail-closed)");
        assert!(err.to_string().contains("did not resolve"), "{err}");
    }

    #[tokio::test]
    async fn native_surface_rejects_classical_ed25519_before_resolution() {
        let peer = random_ed25519();
        let err = admit_key_against_did(
            &NeverResolve,
            ORIGIN,
            interop_signer(peer),
            AdmissionTrustSurface::Native,
            &ed25519_to_did_key(&peer),
            None,
            None,
        )
        .await
        .expect_err("native admission must require PQ-hybrid assurance");
        assert!(err.to_string().contains("PQ-hybrid"), "{err}");
        assert!(err.to_string().contains("downgrade"), "{err}");
    }

    #[tokio::test]
    async fn native_surface_accepts_verified_hybrid_proof() {
        let peer = random_ed25519();
        let did = ed25519_to_did_key(&peer);
        let admitted = admit_key_against_did(
            &NeverResolve,
            ORIGIN,
            hybrid_signer(peer),
            AdmissionTrustSurface::Native,
            &did,
            None,
            None,
        )
        .await
        .expect("hybrid proof satisfies the native assurance floor");
        assert_eq!(admitted.key, peer);
    }

    // ── JWKS fallback path ───────────────────────────────────────────────────

    #[tokio::test]
    async fn admits_via_jwks_fallback_when_no_did_vm_match() {
        // DID doc has a non-matching VM; the federation JWKS carries the application signer key.
        let published = random_ed25519();
        let peer = random_ed25519();
        let jwks = jwks_with_keys(&[peer]);
        let gate =
            FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[published])));
        let admitted = gate
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                DID,
                Some(&jwks),
            )
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
        let gate =
            FederationAdmissionGate::new(AllowOrigin, FixtureDoc(did_doc_with_vm(&[published])));
        let err = gate
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                DID,
                Some(&jwks),
            )
            .await
            .expect_err("key in neither source must reject");
        assert!(err.to_string().contains("nor the federation JWKS"), "{err}");
    }

    #[test]
    fn no_application_signer_proof_helper_is_fail_closed() {
        let err = reject_no_application_signer_proof(ORIGIN);
        assert!(err.to_string().contains("failing closed"), "{err}");
        assert!(
            err.to_string().contains("carrier NodeId/EndpointId"),
            "{err}"
        );
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
    async fn admits_did_key_peer_whose_application_signer_matches() {
        // Self-certifying: the verified application signer equals the did:key
        // encodes. No DID-doc resolution — NeverResolve proves the gate never
        // calls the resolver for a did:key.
        let key = random_ed25519();
        let did = ed25519_to_did_key(&key);
        let gate = FederationAdmissionGate::new(AllowOrigin, NeverResolve);
        let admitted = gate
            .admit(
                ORIGIN,
                interop_signer(key),
                AdmissionTrustSurface::ThirdPartyInterop,
                &did,
                None,
            )
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
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                &did,
                None,
            )
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
            .admit(
                ORIGIN,
                interop_signer(key),
                AdmissionTrustSurface::ThirdPartyInterop,
                &did,
                None,
            )
            .await
            .expect_err("denied origin must reject a did:key peer");
        assert!(
            err.to_string().contains("stage 1"),
            "expected stage-1 rejection, got: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_invalid_did_key_fail_closed() {
        // A malformed did:key (ed25519-pub multicodec absent / wrong) must fail
        // closed without resolving.
        let peer = random_ed25519();
        let bad_did = "did:key:zNotAValidEd25519Multikey";
        let gate = FederationAdmissionGate::new(AllowOrigin, NeverResolve);
        let err = gate
            .admit(
                ORIGIN,
                interop_signer(peer),
                AdmissionTrustSurface::ThirdPartyInterop,
                bad_did,
                None,
            )
            .await
            .expect_err("invalid did:key must reject (fail-closed)");
        assert!(err.to_string().contains("stage 2"), "{err}");
    }

    // ── did:at9p self-certifying hybrid arm (D2/#894) ──────────────────────────

    use crate::crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use crate::envelope::PqTrustStore;
    use crate::identity_resolver::{At9pCapsuleResolver, VerifiedAt9pKeys};

    const AT9P_DID: &str = "did:at9p:fakecid512";

    /// A capsule gate test double: returns a fixed [`VerifiedAt9pKeys`] on
    /// `verify_bytes`, or `Err` when `fail` is set (simulating a GATE rejection —
    /// bad sig / wrong cid). The real GATE lives in `hyprstream-pds`
    /// (`at9p_resolver::At9pGateResolver`); this fixture keeps the arm testable in
    /// the lower rpc crate without a pds dependency.
    struct FixtureGate {
        keys: VerifiedAt9pKeys,
        fail: bool,
    }

    impl At9pCapsuleResolver for FixtureGate {
        fn verify_bytes(&self, _did: &str, _capsule_bytes: &[u8]) -> Result<VerifiedAt9pKeys> {
            if self.fail {
                bail!("simulated GATE failure (bad sig / wrong cid)");
            }
            Ok(self.keys.clone())
        }
    }

    fn verified_subject(ed: [u8; 32]) -> VerifiedAt9pKeys {
        let (_sk, vk) = ml_dsa_generate_keypair();
        VerifiedAt9pKeys::new_gate_verified(ed, vk)
    }

    #[tokio::test]
    async fn admits_did_at9p_peer_and_binds_hybrid_keys_from_capsule() {
        // Valid capsule → GATE passes → the verified ed25519→ml_dsa_65 pair is
        // bound into KeyedPqTrustStore, atomic hybrid binding from content-verified
        // material (no config trust).
        let ed = random_ed25519();
        let keys = verified_subject(ed);
        let gate = FixtureGate {
            keys: keys.clone(),
            fail: false,
        };
        let mut store = KeyedPqTrustStore::new();
        assert!(store.is_empty(), "store starts empty (no config trust)");

        let admitted = admit_key_against_did(
            &NeverResolve,
            ORIGIN,
            hybrid_signer(ed),
            AdmissionTrustSurface::Native,
            AT9P_DID,
            None,
            Some(At9pAdmission {
                capsule_bytes: b"<capsule>",
                gate: &gate,
                pq_store: &mut store,
            }),
        )
        .await
        .expect("verified capsule must admit");

        assert_eq!(admitted.did.as_deref(), Some(AT9P_DID));
        assert_eq!(admitted.key, ed);
        // The binding is exactly the capsule's verified keys — comes from the
        // capsule, not config.
        assert_eq!(store.len(), 1, "exactly one hybrid binding installed");
        let bound = store.ml_dsa_key_for(&ed).expect("ed25519→ml_dsa_65 bound");
        assert_eq!(ml_dsa_vk_bytes(&bound), ml_dsa_vk_bytes(keys.ml_dsa_65()));
    }

    #[tokio::test]
    async fn rejects_did_at9p_on_gate_failure_and_binds_nothing() {
        // A capsule that fails GATE (bad sig / wrong cid) → reject, and crucially
        // NOTHING is bound into the trust store (no downgrade, no partial trust).
        let ed = random_ed25519();
        let gate = FixtureGate {
            keys: verified_subject(ed),
            fail: true,
        };
        let mut store = KeyedPqTrustStore::new();

        let err = admit_key_against_did(
            &NeverResolve,
            ORIGIN,
            hybrid_signer(ed),
            AdmissionTrustSurface::Native,
            AT9P_DID,
            None,
            Some(At9pAdmission {
                capsule_bytes: b"<capsule>",
                gate: &gate,
                pq_store: &mut store,
            }),
        )
        .await
        .expect_err("GATE failure must reject (fail-closed)");
        assert!(err.to_string().contains("stage 2"), "{err}");
        assert!(err.to_string().contains("GATE"), "{err}");
        assert!(store.is_empty(), "a rejected capsule must bind nothing");
    }

    #[tokio::test]
    async fn rejects_did_at9p_on_application_signer_mismatch_and_binds_nothing() {
        // The capsule verifies, but the application signer is not the capsule's
        // Ed25519 subject key, so reject and bind nothing.
        let identity = random_ed25519();
        let peer = random_ed25519();
        assert_ne!(identity, peer);
        let gate = FixtureGate {
            keys: verified_subject(identity),
            fail: false,
        };
        let mut store = KeyedPqTrustStore::new();

        let err = admit_key_against_did(
            &NeverResolve,
            ORIGIN,
            hybrid_signer(peer),
            AdmissionTrustSurface::Native,
            AT9P_DID,
            None,
            Some(At9pAdmission {
                capsule_bytes: b"<capsule>",
                gate: &gate,
                pq_store: &mut store,
            }),
        )
        .await
        .expect_err("application-signer mismatch must reject");
        assert!(err.to_string().contains("stage 2"), "{err}");
        assert!(store.is_empty(), "a mismatched peer must bind nothing");
    }

    #[tokio::test]
    async fn rejects_did_at9p_without_an_admission_context() {
        // did:at9p presented but no capsule/admission context supplied (the live
        // accept path does not feed bytes yet) → fail closed.
        let peer = random_ed25519();
        let err = admit_key_against_did(
            &NeverResolve,
            ORIGIN,
            hybrid_signer(peer),
            AdmissionTrustSurface::Native,
            AT9P_DID,
            None,
            None,
        )
        .await
        .expect_err("missing capsule context must reject");
        assert!(err.to_string().contains("stage 2"), "{err}");
        assert!(err.to_string().contains("fail-closed"), "{err}");
    }

    #[tokio::test]
    async fn did_at9p_still_enforces_stage1_origin() {
        // Stage 1 (origin) runs first; a denied origin rejects before the at9p arm.
        let ed = random_ed25519();
        let gate = FederationAdmissionGate::new(DenyOrigin, NeverResolve);
        let err = gate
            .admit(
                ORIGIN,
                hybrid_signer(ed),
                AdmissionTrustSurface::Native,
                AT9P_DID,
                None,
            )
            .await
            .expect_err("denied origin must reject a did:at9p peer");
        // gate.admit passes None for the at9p context, so stage 1's denial is what
        // surfaces (stage 2 never runs).
        assert!(
            err.to_string().contains("stage 1"),
            "expected stage-1 rejection, got: {err}"
        );
    }
}
