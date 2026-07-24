//! Method-dispatched [`IdentityResolver`] — the real `Did → IdentityKeys` binding (#579).
//!
//! This is the single, non-mirror implementation of the canonical
//! [`crate::identity::IdentityResolver`] contract. It replaces the fixture /
//! test-double resolvers that stood in for it while the #549 perimeter gateway
//! and #137 admission gate were built against the interface.
//!
//! # Division of labor
//!
//! The resolver owns **verification + assurance derivation**; it does *not* own
//! object labelling, the `KeyedPqTrustStore.bind` mesh anchor (#894), or any
//! perimeter policy — those consumers are generic over the trait and unchanged.
//!
//! # Method dispatch
//!
//! [`resolve_identity_keys`](IdentityResolver::resolve_identity_keys) branches on
//! the DID method:
//!
//! - **`did:web`** — resolve the DID document and extract its verification
//!   methods. An Ed25519 identity VM (for example `#mesh`) is the classical anchor;
//!   its explicitly named ML-DSA-65 companion (`#mesh-pq`, multicodec `0x1211`)
//!   makes that candidate [`Assurance::PqHybrid`]. Other candidates remain
//!   [`Assurance::Classical`]. No Ed25519 VM ⇒ nothing to bind ⇒ `Err` (fail-closed).
//! - **`did:key`** — a single Ed25519 key decoded from the DID string itself; no
//!   fetch. A single classical key can never be hybrid, so the honest ceiling is
//!   [`Assurance::Classical`].
//! - **`did:at9p`** — the self-certifying hybrid-PQC capsule identity (#879/#880
//!   D2, #894). The resolver delegates to an injected [`At9pCapsuleResolver`]
//!   that runs the A4 GATE pipeline (canon→hash→sig, #884) over the capsule and
//!   returns its content-verified subject keys; a verified capsule yields
//!   [`Assurance::PqHybrid`]. With no capsule resolver configured the arm fails
//!   closed (production wiring of the mainline locator is Track C, #890).
//! - **any other method** — `Err` (fail-closed).
//!
//! # Fail-closed
//!
//! Per the [`IdentityResolver`] contract, every inability to establish key
//! material — an unknown method, a fetch failure, a malformed document, a
//! did:web with no Ed25519 VM — returns `Err`. A resolver **never** returns a
//! default-`Unverified` `Ok`.
//!
//! # Sync trait over an async fetch
//!
//! The contract is synchronous and enrollment happens **off the auth path**
//! (once, at session establishment), so did:web document retrieval is abstracted
//! behind a synchronous [`DidDocumentProvider`]. This keeps the decode +
//! assurance-derivation core pure and testable with an injected fixture (no live
//! network), mirroring the injected-fetcher pattern in [`crate::did_web`]. The
//! async→sync bridge to the live [`crate::did_web::HttpDidDocFetcher`] is a
//! wiring concern for the accept-path integration (there is no production
//! construction site of the perimeter gateway today).

use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde_json::Value;

use crate::crypto::pq::{MlDsaVerifyingKey, ml_dsa_vk_from_bytes};
use crate::did_web::{
    MULTICODEC_ML_DSA_65_PUB, decode_ed25519_multikey, decode_multikey, did_key_to_ed25519,
};
use crate::identity::{Did, IdentityKeyCandidate, IdentityKeys, IdentityResolver};

/// A synchronous DID-document provider for the did:web arm.
///
/// Abstracted so the resolver's decode/assurance core is testable with an
/// injected fixture (no network). Enrollment is off the auth path, so a blocking
/// implementation is acceptable; the trait keeps that policy out of the resolver.
pub trait DidDocumentProvider: Send + Sync {
    /// Fetch and parse the DID document for `did` (already known to be `did:web`).
    fn document(&self, did: &str) -> Result<Value>;
}

/// One content-verified atomic hybrid subject key: an Ed25519 identity key and
/// the ML-DSA-65 key **bound to it in the same capsule entry** (#1188 / #1183).
///
/// The pairing is atomic: the ML-DSA-65 half is never recombined with a
/// different Ed25519 half. Selecting a subject key by its Ed25519 identity
/// therefore yields exactly the PQ key the capsule author bound to it, which is
/// what overlap rotation and PQ-hybrid rollout require — publish several keys,
/// let each verifier take the whole pair it recognizes.
#[derive(Clone)]
pub struct VerifiedAt9pKey {
    ed25519: [u8; 32],
    ml_dsa_65: MlDsaVerifyingKey,
}

impl VerifiedAt9pKey {
    /// The content-verified Ed25519 identity key.
    pub fn ed25519(&self) -> &[u8; 32] {
        &self.ed25519
    }

    /// The ML-DSA-65 key atomically bound to [`Self::ed25519`] in the capsule.
    pub fn ml_dsa_65(&self) -> &MlDsaVerifyingKey {
        &self.ml_dsa_65
    }
}

/// The content-verified subject key **set** of a `did:at9p` capsule, re-expressed
/// in the rpc-local key types so this crate need not depend on `hyprstream-pds`
/// (pds depends on rpc; the dependency does not reverse).
///
/// `subjectKeys` is a *set* of currently-usable identity keys, never an ordered
/// list with a positional "primary" (#1188 / #1183): a producer publishes the
/// new key alongside the old to rotate without a flag day, and a classical key
/// beside a PQ/composite key for hybrid rollout. A consumer selects the member
/// it recognizes — here [`Self::for_ed25519`], keyed on the Ed25519 application
/// signer identity — and skips the rest, never letting position confer
/// authority. The set is non-empty and its Ed25519 halves are unique (enforced
/// by the capsule schema), so selection is unambiguous.
///
/// A [`VerifiedAt9pKeys`] is only constructable via the GATE mints
/// ([`VerifiedAt9pKeys::new_gate_verified`] / [`VerifiedAt9pKeys::new_gate_verified_set`]),
/// which only an [`At9pCapsuleResolver`] that ran the A4 GATE pipeline
/// (canon→hash→sig, #884) to completion reaches. The private field makes the
/// "only constructable after the GATE ran" provenance a type-level invariant
/// rather than a bare rustdoc claim: arbitrary code cannot build one with an
/// unverified `ed25519`↔`ml_dsa_65` binding, so holding one is proof the binding
/// came from **content-verified capsule material**, not a config file or a
/// caller assertion (the D2/#894 provenance boundary, hardened in #964).
#[derive(Clone)]
pub struct VerifiedAt9pKeys {
    /// The content-verified subject key set — non-empty, unique Ed25519 halves.
    keys: Vec<VerifiedAt9pKey>,
}

impl VerifiedAt9pKeys {
    /// The GATE mint for a **single** verified subject key — a convenience over
    /// [`Self::new_gate_verified_set`] for the (common) single-key capsule.
    ///
    /// This is the constructor-witness chokepoint: the field is private, so the
    /// *only* way to produce a `VerifiedAt9pKeys` is one of these mints, which
    /// encode the GATE-ran provenance in their names. Trusted callers are the A4
    /// GATE resolver implementations (the real [`At9pCapsuleResolver`] impl in
    /// `hyprstream-pds::at9p_resolver`, plus in-crate test fixtures). Code that
    /// has not run the GATE has no business calling this — review a new call site
    /// as a provenance decision, not a mechanical field set.
    ///
    /// (Rust cannot seal this across crates — `hyprstream-pds` is a *dependent* of
    /// `hyprstream-rpc`, so rpc has no way to grant pds the mint while denying it
    /// to arbitrary code. The private field + named mints convert the provenance
    /// from "any code can write the struct literal" to "any code must go through
    /// one review-visible chokepoint", which is the proportionate hardening; the
    /// trust root remains "whoever injects the gate is trusted".)
    pub fn new_gate_verified(ed25519: [u8; 32], ml_dsa_65: MlDsaVerifyingKey) -> Self {
        Self {
            keys: vec![VerifiedAt9pKey { ed25519, ml_dsa_65 }],
        }
    }

    /// The GATE mint for the full verified subject key **set** (#1188).
    ///
    /// The projected set MUST be non-empty (the schema guarantees it) and MUST
    /// have unique Ed25519 halves so [`Self::for_ed25519`] selection is
    /// unambiguous — the same discipline the capsule schema enforces on
    /// `subjectKeys`. A violation is a projection bug, not attacker input, so it
    /// is a fail-closed `Err` rather than a silent first-wins.
    pub fn new_gate_verified_set(keys: Vec<VerifiedAt9pKey>) -> Result<Self> {
        if keys.is_empty() {
            return Err(anyhow!(
                "VerifiedAt9pKeys: capsule projected an empty subject key set (fail-closed)"
            ));
        }
        for (i, a) in keys.iter().enumerate() {
            if keys[i + 1..].iter().any(|b| b.ed25519 == a.ed25519) {
                return Err(anyhow!(
                    "VerifiedAt9pKeys: duplicate Ed25519 subject key in projected set — \
                     ambiguous selection (fail-closed)"
                ));
            }
        }
        Ok(Self { keys })
    }

    /// Build a [`VerifiedAt9pKey`] for the set mint — pds-side projection helper.
    pub fn key(ed25519: [u8; 32], ml_dsa_65: MlDsaVerifyingKey) -> VerifiedAt9pKey {
        VerifiedAt9pKey { ed25519, ml_dsa_65 }
    }

    /// Select the verified subject key whose Ed25519 half equals `ed25519`, if
    /// the capsule published one — the set-semantics accessor (#1188).
    ///
    /// A consumer holding an Ed25519 application-signer identity finds *its*
    /// hybrid pair by identity and takes the atomically-bound ML-DSA-65 half from
    /// the same entry. Returns `None` when no published key matches; the caller
    /// then rejects fail-closed rather than falling back to a positional key.
    pub fn for_ed25519(&self, ed25519: &[u8; 32]) -> Option<&VerifiedAt9pKey> {
        self.keys.iter().find(|k| &k.ed25519 == ed25519)
    }

    /// Whether `ed25519` is one of the published, content-verified subject keys.
    pub fn publishes_ed25519(&self, ed25519: &[u8; 32]) -> bool {
        self.for_ed25519(ed25519).is_some()
    }

    /// All content-verified subject keys in the set (non-empty).
    pub fn keys(&self) -> &[VerifiedAt9pKey] {
        &self.keys
    }
}

/// Verifies a `did:at9p` capsule and yields its content-verified subject keys.
///
/// The A4 GATE pipeline (`canon-gate → hash-gate(H512 == cid512) → sig-gate`,
/// #884) lives in `hyprstream-pds` beside the capsule schema. The lower
/// `hyprstream-rpc` crate — which both the admission gate (`crate::admission`) and
/// this resolver belong to — cannot depend on pds, so the GATE is abstracted
/// behind this object-safe trait and implemented in pds over
/// `hyprstream_pds::at9p_gate::verify_did_at9p`.
///
/// Two entry points, each fail-closed on any GATE failure (never a default-`Ok`):
///
/// - [`verify_bytes`](Self::verify_bytes) — the **admission path**: the peer
///   presented its capsule bytes over the channel; the GATE decides whether to
///   accept them. Used by `admit_key_against_did`'s `did:at9p` arm to bind the
///   verified hybrid keys into the [`crate::envelope::KeyedPqTrustStore`] (#894).
/// - [`resolve`](Self::resolve) — the **resolver path**: fetch the capsule bytes
///   for `did` from the (untrusted) mainline locator (#890, Track C) and GATE-verify
///   them. Used by this module's `did:at9p` arm to derive `PqHybrid` assurance.
///
/// The default impls return `Err` so a configuration that supplies only one path
/// (admission today; the locator is not wired) leaves the other fail-closed rather
/// than silently degrading.
pub trait At9pCapsuleResolver: Send + Sync {
    /// GATE-verify `capsule_bytes` the peer presented for `did:at9p:<cid512>`
    /// (admission path). Pure over the bytes — no fetch, no I/O.
    fn verify_bytes(&self, did: &str, capsule_bytes: &[u8]) -> Result<VerifiedAt9pKeys> {
        let _ = (did, capsule_bytes);
        Err(anyhow!(
            "at9p capsule-byte verification is not configured (fail-closed)"
        ))
    }

    /// Fetch the capsule for `did` from the untrusted locator and GATE-verify it
    /// (resolver path). The locator itself is Track C (#890); until it is wired an
    /// honest implementation returns `Err`.
    fn resolve(&self, did: &str) -> Result<VerifiedAt9pKeys> {
        let _ = did;
        Err(anyhow!(
            "at9p capsule resolution (locator fetch) is not configured (fail-closed)"
        ))
    }
}

/// The real, method-dispatched [`IdentityResolver`] (#579).
///
/// Generic over a [`DidDocumentProvider`] so the did:web fetch is injectable
/// (fixture in tests, blocking HTTPS fetcher in production wiring). The optional
/// [`At9pCapsuleResolver`] supplies the `did:at9p` arm (D2/#894); when `None` a
/// `did:at9p` identity fails closed exactly as it did before the arm existed.
pub struct MethodDispatchResolver<P: DidDocumentProvider> {
    docs: P,
    at9p: Option<Arc<dyn At9pCapsuleResolver>>,
    now: Arc<dyn Fn() -> i64 + Send + Sync>,
}

impl<P: DidDocumentProvider> MethodDispatchResolver<P> {
    /// Construct a resolver over a DID-document provider.
    pub fn new(docs: P) -> Self {
        Self {
            docs,
            at9p: None,
            now: Arc::new(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |duration| duration.as_secs() as i64)
            }),
        }
    }

    /// Override the wall clock used to enforce bounded DID verification methods.
    ///
    /// Production uses Unix time. Tests and embedders can inject a deterministic
    /// clock to prove overlap and retirement behavior without sleeping.
    pub fn with_clock(mut self, now: impl Fn() -> i64 + Send + Sync + 'static) -> Self {
        self.now = Arc::new(now);
        self
    }

    /// Attach the `did:at9p` capsule resolver (D2/#894). Without it a `did:at9p`
    /// identity fails closed; with it, a GATE-verified capsule yields `PqHybrid`.
    pub fn with_at9p(mut self, at9p: Arc<dyn At9pCapsuleResolver>) -> Self {
        self.at9p = Some(at9p);
        self
    }

    /// The did:web arm: resolve the document and derive key material + assurance.
    ///
    /// Each usable Ed25519 VM becomes a named candidate. A same-name `-pq` VM
    /// explicitly upgrades only that candidate to `PqHybrid`; unrelated PQ VMs
    /// are never paired by document order.
    /// - No Ed25519 VM ⇒ `Err` (fail-closed: nothing to anchor).
    ///
    /// # Assurance quality — the at9p asymmetry
    ///
    /// Unlike the `did:at9p` arm, where the `ed25519`↔`ml_dsa_65` pair is
    /// **content-verified** by the GATE over a self-certifying capsule, the
    /// `did:web` arm's `PqHybrid` rests on **co-presence in a TLS-fetched
    /// document**, not on a content-binding between the two keys. Two caveats
    /// follow, consistent with the ratified #579/D1 "did:web we operate" model:
    ///
    /// - **Named pairing:** an Ed25519 VM `#mesh-old` may be paired only with
    ///   the explicitly named ML-DSA-65 VM `#mesh-old-pq`. This gives each
    ///   overlap slot an independent hybrid binding; a PQ VM with no matching
    ///   Ed25519 VM is ignored rather than attached to an arbitrary key.
    /// - **No content binding:** the document attests both keys, but nothing in the
    ///   TLS fetch proves the Ed25519 and ML-DSA-65 keys belong to the same
    ///   principal beyond the document's own say-so. The at9p capsule proves this
    ///   cryptographically; did:web trusts the (TLS-transported) document.
    fn resolve_did_web(&self, did: &Did) -> Result<IdentityKeys> {
        let doc = self
            .docs
            .document(did.as_str())
            .map_err(|e| anyhow!("did:web {did} document did not resolve: {e}"))?;

        let candidates = did_web_key_candidates(&doc, (self.now)());
        if candidates.is_empty() {
            anyhow::bail!("did:web {did}: no Ed25519 verificationMethod to anchor — fail-closed");
        }
        Ok(IdentityKeys::new(candidates))
    }

    /// The did:key arm: a single self-certifying Ed25519 key, no fetch.
    ///
    /// A `did:key` encodes exactly one key; a single classical key can never be
    /// hybrid, so assurance is `Classical` (the honest ceiling).
    fn resolve_did_key(&self, did: &Did) -> Result<IdentityKeys> {
        let ed25519 = did_key_to_ed25519(did.as_str())
            .map_err(|e| anyhow!("did:key {did} is not a valid Ed25519 identity: {e}"))?;
        Ok(IdentityKeys::single_ed25519(did.as_str(), ed25519))
    }

    /// The did:at9p arm (#894, D2): GATE-verify the capsule and derive `PqHybrid`.
    ///
    /// Delegates to the configured [`At9pCapsuleResolver`], which runs the A4
    /// canon→hash→sig pipeline (#884). Only a verified capsule reaches here: both
    /// the classical Ed25519 and the bound ML-DSA-65 keys come from
    /// content-verified material, so the honest assurance ceiling is `PqHybrid`
    /// (the classical→hybrid trust upgrade). Any GATE failure, and a missing
    /// capsule resolver, fail closed.
    ///
    /// # Set semantics (#1188 / #1183)
    ///
    /// The capsule publishes a *set* of subject keys so overlap rotation and
    /// PQ-hybrid rollout work. Every content-verified pair becomes a candidate;
    /// callers select by the admitted Ed25519 signer, never by capsule position.
    fn resolve_did_at9p(&self, did: &Did) -> Result<IdentityKeys> {
        let at9p = self.at9p.as_ref().ok_or_else(|| {
            anyhow!("did:at9p {did}: no capsule resolver configured — fail-closed")
        })?;
        let verified = at9p
            .resolve(did.as_str())
            .map_err(|e| anyhow!("did:at9p {did}: capsule GATE failed: {e}"))?;
        Ok(IdentityKeys::new(
            verified
                .keys()
                .iter()
                .map(|key| IdentityKeyCandidate {
                    id: format!(
                        "{}#subject-{}",
                        did.as_str(),
                        bs58::encode(key.ed25519()).into_string()
                    ),
                    ed25519: *key.ed25519(),
                    ml_dsa_65: Some(key.ml_dsa_65().clone()),
                })
                .collect(),
        ))
    }
}

/// Decode the named DID verification-method set into compatible candidates.
///
/// DID VM ids are the selector. An Ed25519 VM `id` is upgraded only by an
/// ML-DSA-65 VM whose id is `"{id}-pq"`; this is the mesh's stable, explicit
/// hybrid binding convention (`#mesh` ↔ `#mesh-pq`). The function deliberately
/// does not expose a positional selection API: callers receive every usable
/// candidate and choose using a verified signer or another protocol selector.
fn did_web_key_candidates(doc: &Value, now: i64) -> Vec<IdentityKeyCandidate> {
    let Some(vms) = doc.get("verificationMethod").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut ed25519 = Vec::new();
    let mut ml_dsa_65 = std::collections::HashMap::new();
    for vm in vms {
        let (Some(id), Some(multibase)) = (
            vm.get("id").and_then(Value::as_str),
            vm.get("publicKeyMultibase").and_then(Value::as_str),
        ) else {
            continue;
        };
        if !verification_method_is_live(vm, now) {
            continue;
        }
        if let Ok(key) = decode_ed25519_multikey(multibase) {
            ed25519.push((id.to_owned(), key));
            continue;
        }
        let Ok(raw) = decode_multikey(multibase, &MULTICODEC_ML_DSA_65_PUB) else {
            continue;
        };
        match ml_dsa_vk_from_bytes(&raw) {
            Ok(key) => {
                ml_dsa_65.insert(id.to_owned(), key);
            }
            Err(error) => {
                tracing::debug!(%error, verification_method = %id, "skipping invalid ML-DSA-65 DID verification method");
            }
        }
    }

    let legacy_pq_id = doc
        .get("id")
        .and_then(Value::as_str)
        .map(|did| format!("{did}#mesh-pq"));
    let legacy_singleton = ed25519.len() == 1 && ed25519[0].0.ends_with("#key-1");

    ed25519
        .into_iter()
        .map(|(id, ed25519)| {
            let pq_id = format!("{id}-pq");
            let explicit = ml_dsa_65.remove(&pq_id);
            // Rolling-upgrade bridge for the previously published singleton
            // `#key-1` + `#mesh-pq` document. It is deliberately restricted to
            // that exact one-Ed25519 legacy shape so an unrelated PQ method can
            // never be attached in a multi-candidate document.
            let legacy = legacy_singleton
                .then_some(legacy_pq_id.as_ref())
                .flatten()
                .and_then(|legacy_id| ml_dsa_65.remove(legacy_id));
            IdentityKeyCandidate {
                id,
                ed25519,
                ml_dsa_65: explicit.or(legacy),
            }
        })
        .collect()
}

fn verification_method_is_live(vm: &Value, now: i64) -> bool {
    let not_before = match vm.get("nbf") {
        None => true,
        Some(value) => value.as_i64().is_some_and(|nbf| now >= nbf),
    };
    let not_expired = match vm.get("exp") {
        None => true,
        Some(value) => value.as_i64().is_some_and(|exp| now < exp),
    };
    not_before && not_expired
}

impl<P: DidDocumentProvider> IdentityResolver for MethodDispatchResolver<P> {
    fn resolve_identity_keys(&self, did: &Did) -> Result<IdentityKeys> {
        if did.is_did_web() {
            self.resolve_did_web(did)
        } else if did.is_did_key() {
            self.resolve_did_key(did)
        } else if did.is_did_at9p() {
            self.resolve_did_at9p(did)
        } else {
            // Unknown DID method — fail closed (never a default-Unverified Ok).
            Err(anyhow!(
                "unsupported DID method for {did}: no resolver arm — fail-closed"
            ))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::auth::mac::Assurance;
    use crate::crypto::pq::{ml_dsa_generate_keypair, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_bytes};
    use crate::did_key::{MULTICODEC_ED25519_PUB, MULTICODEC_ML_DSA_65_PUB, ed25519_to_did_key};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use serde_json::json;

    fn rand_ed25519() -> [u8; 32] {
        SigningKey::generate(&mut OsRng).verifying_key().to_bytes()
    }

    fn encode_multikey(raw: &[u8], codec: [u8; 2]) -> String {
        let mut payload = Vec::with_capacity(2 + raw.len());
        payload.extend_from_slice(&codec);
        payload.extend_from_slice(raw);
        format!("z{}", bs58::encode(payload).into_string())
    }

    /// A DID document with an Ed25519 `#mesh` VM and, optionally, an ML-DSA-65
    /// `#mesh-pq` VM.
    fn did_doc(ed: &[u8; 32], ml_dsa_vk_bytes: Option<&[u8]>) -> Value {
        did_doc_with_named_vms(&[("mesh", ed)], ml_dsa_vk_bytes.map(|pq| ("mesh-pq", pq)))
    }

    fn did_doc_with_named_vms(
        ed25519: &[(&str, &[u8; 32])],
        ml_dsa_65: Option<(&str, &[u8])>,
    ) -> Value {
        let mut vms = Vec::with_capacity(ed25519.len() + usize::from(ml_dsa_65.is_some()));
        for (name, key) in ed25519 {
            vms.push(json!({
                "id": format!("{DID_WEB}#{name}"),
                "type": "Multikey",
                "controller": DID_WEB,
                "publicKeyMultibase": encode_multikey(*key, MULTICODEC_ED25519_PUB),
            }));
        }
        if let Some((name, key)) = ml_dsa_65 {
            vms.push(json!({
                "id": format!("{DID_WEB}#{name}"),
                "type": "Multikey",
                "controller": DID_WEB,
                "publicKeyMultibase": encode_multikey(key, MULTICODEC_ML_DSA_65_PUB),
            }));
        }
        json!({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": DID_WEB,
            "verificationMethod": vms,
            "service": [],
        })
    }

    fn did_doc_with_two_hybrid_pairs(
        old_ed: &[u8; 32],
        old_pq: &[u8],
        new_ed: &[u8; 32],
        new_pq: &[u8],
    ) -> Value {
        let vms = vec![
            json!({
                "id": "did:web:peer.example#mesh-old",
                "type": "Multikey",
                "controller": "did:web:peer.example",
                "publicKeyMultibase": encode_multikey(old_ed, MULTICODEC_ED25519_PUB),
            }),
            json!({
                "id": "did:web:peer.example#mesh-old-pq",
                "type": "Multikey",
                "controller": "did:web:peer.example",
                "publicKeyMultibase": encode_multikey(old_pq, MULTICODEC_ML_DSA_65_PUB),
            }),
            json!({
                "id": "did:web:peer.example#mesh-new",
                "type": "Multikey",
                "controller": "did:web:peer.example",
                "publicKeyMultibase": encode_multikey(new_ed, MULTICODEC_ED25519_PUB),
            }),
            json!({
                "id": "did:web:peer.example#mesh-new-pq",
                    "type": "Multikey",
                    "controller": "did:web:peer.example",
                "publicKeyMultibase": encode_multikey(new_pq, MULTICODEC_ML_DSA_65_PUB),
            }),
        ];
        json!({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": "did:web:peer.example",
            "verificationMethod": vms,
            "service": [],
        })
    }

    /// Fixture provider returning a fixed document.
    struct FixtureDocs(Value);
    impl DidDocumentProvider for FixtureDocs {
        fn document(&self, _did: &str) -> Result<Value> {
            Ok(self.0.clone())
        }
    }

    #[derive(Clone)]
    struct RotatingDocs(std::sync::Arc<parking_lot::Mutex<Value>>);
    impl RotatingDocs {
        fn new(document: Value) -> Self {
            Self(std::sync::Arc::new(parking_lot::Mutex::new(document)))
        }

        fn publish(&self, document: Value) {
            *self.0.lock() = document;
        }
    }
    impl DidDocumentProvider for RotatingDocs {
        fn document(&self, _did: &str) -> Result<Value> {
            Ok(self.0.lock().clone())
        }
    }

    /// Provider that always fails — exercises the fail-closed fetch path.
    struct FailingDocs;
    impl DidDocumentProvider for FailingDocs {
        fn document(&self, did: &str) -> Result<Value> {
            Err(anyhow!("simulated fetch failure for {did}"))
        }
    }

    /// A provider that MUST NOT be called (did:key / did:at9p never fetch).
    struct NeverDocs;
    impl DidDocumentProvider for NeverDocs {
        fn document(&self, did: &str) -> Result<Value> {
            panic!("resolver must not fetch a document for {did}");
        }
    }

    const DID_WEB: &str = "did:web:peer.example";

    #[test]
    fn did_web_ed25519_only_is_classical() {
        let ed = rand_ed25519();
        let resolver = MethodDispatchResolver::new(FixtureDocs(did_doc(&ed, None)));
        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("ed25519-only did:web resolves");
        let candidate = keys
            .candidate_for_ed25519(&ed)
            .expect("published candidate");
        assert_eq!(candidate.assurance(), Assurance::Classical);
        assert!(candidate.ml_dsa_65.is_none());
    }

    #[test]
    fn did_web_with_ml_dsa_is_pqhybrid() {
        let ed = rand_ed25519();
        let (_sk, vk) = ml_dsa_generate_keypair();
        let pq_bytes = ml_dsa_vk_bytes(&vk);
        let resolver = MethodDispatchResolver::new(FixtureDocs(did_doc(&ed, Some(&pq_bytes))));
        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("hybrid did:web resolves");
        let candidate = keys
            .candidate_for_ed25519(&ed)
            .expect("published candidate");
        assert_eq!(candidate.assurance(), Assurance::PqHybrid);
        // The resolved PQ key is exactly the published one.
        let resolved = candidate.ml_dsa_65.as_ref().expect("PQ anchor present");
        assert_eq!(ml_dsa_vk_bytes(resolved), pq_bytes);
    }

    #[test]
    fn legacy_key_1_mesh_pq_document_remains_hybrid_during_upgrade() {
        let ed = rand_ed25519();
        let (_, pq) = ml_dsa_generate_keypair();
        let pq = ml_dsa_vk_bytes(&pq);
        let doc = did_doc_with_named_vms(&[("key-1", &ed)], Some(("mesh-pq", &pq)));
        let keys = MethodDispatchResolver::new(FixtureDocs(doc))
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("legacy singleton resolves");
        let candidate = keys
            .candidate_for_ed25519(&ed)
            .expect("legacy signer candidate");
        assert_eq!(candidate.assurance(), Assurance::PqHybrid);
        assert_eq!(ml_dsa_vk_bytes(candidate.ml_dsa_65.as_ref().unwrap()), pq);
    }

    #[test]
    fn authentication_relationship_is_not_a_second_inconsistent_authority_parser() {
        let ed = rand_ed25519();
        let mut doc = did_doc(&ed, None);
        // Admission and identity resolution both consume verificationMethod.
        // A malformed, relative, or embedded relationship cannot disable or
        // tighten a separate enrollment-only filter because no such filter is
        // applied here.
        for authentication in [
            json!("bad"),
            json!({}),
            json!(["#mesh"]),
            json!([{
                "id": format!("{DID_WEB}#mesh"),
                "type": "Multikey",
                "controller": DID_WEB,
                "publicKeyMultibase": encode_multikey(&ed, MULTICODEC_ED25519_PUB),
            }]),
        ] {
            doc["authentication"] = authentication;
            let keys = MethodDispatchResolver::new(FixtureDocs(doc.clone()))
                .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
                .expect("verificationMethod remains authoritative");
            assert!(keys.candidate_for_ed25519(&ed).is_some());
        }
    }

    #[test]
    fn bounded_verification_methods_follow_the_injected_clock() {
        let old = rand_ed25519();
        let new = rand_ed25519();
        let mut doc = did_doc_with_named_vms(&[("mesh-old", &old), ("mesh-new", &new)], None);
        let vms = doc["verificationMethod"].as_array_mut().unwrap();
        vms[0]["nbf"] = json!(100);
        vms[0]["exp"] = json!(200);
        vms[1]["nbf"] = json!(150);
        vms[1]["exp"] = json!(300);

        let now = Arc::new(std::sync::atomic::AtomicI64::new(175));
        let clock = Arc::clone(&now);
        let resolver = MethodDispatchResolver::new(FixtureDocs(doc))
            .with_clock(move || clock.load(std::sync::atomic::Ordering::SeqCst));
        let did = Did::new(DID_WEB.to_owned());
        let overlap = resolver.resolve_identity_keys(&did).unwrap();
        assert!(overlap.candidate_for_ed25519(&old).is_some());
        assert!(overlap.candidate_for_ed25519(&new).is_some());

        now.store(200, std::sync::atomic::Ordering::SeqCst);
        let retired = resolver.resolve_identity_keys(&did).unwrap();
        assert!(retired.candidate_for_ed25519(&old).is_none());
        assert!(retired.candidate_for_ed25519(&new).is_some());
    }

    #[test]
    fn did_web_derives_pq_key_from_signing_key_roundtrip() {
        // A mesh-derived ML-DSA key (as node_identity produces) round-trips
        // through the VM extractor into a usable verifying key.
        let ed = rand_ed25519();
        let peer_ed = SigningKey::generate(&mut OsRng);
        let peer_pq_sk = crate::node_identity::derive_mesh_mldsa_key(&peer_ed);
        let pq_bytes = ml_dsa_sk_to_vk_bytes(&peer_pq_sk);
        let resolver = MethodDispatchResolver::new(FixtureDocs(did_doc(&ed, Some(&pq_bytes))));
        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("hybrid did:web resolves");
        let candidate = keys
            .candidate_for_ed25519(&ed)
            .expect("published candidate");
        assert_eq!(candidate.assurance(), Assurance::PqHybrid);
        assert_eq!(
            ml_dsa_vk_bytes(candidate.ml_dsa_65.as_ref().unwrap()),
            pq_bytes
        );
    }

    #[test]
    fn did_web_classical_overlap_accepts_both_then_rejects_retired_key_after_drain_window() {
        let old = rand_ed25519();
        let new = rand_ed25519();
        let docs = RotatingDocs::new(did_doc_with_named_vms(
            &[("mesh-old", &old), ("mesh-new", &new)],
            None,
        ));
        let resolver = MethodDispatchResolver::new(docs.clone());

        let overlap = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("overlap document resolves");
        assert_eq!(overlap.len(), 2);
        assert_eq!(
            overlap.candidate_for_ed25519(&old).unwrap().id,
            format!("{DID_WEB}#mesh-old")
        );
        assert_eq!(
            overlap.candidate_for_ed25519(&new).unwrap().id,
            format!("{DID_WEB}#mesh-new")
        );

        // The old VM remains usable for the bounded drain window, then the
        // publisher removes it. A fresh resolution must no longer accept it.
        docs.publish(did_doc_with_named_vms(&[("mesh-new", &new)], None));
        let after_drain = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("post-drain document resolves");
        assert!(after_drain.candidate_for_ed25519(&old).is_none());
        assert!(after_drain.candidate_for_ed25519(&new).is_some());
    }

    #[test]
    fn did_web_two_hybrid_pairs_remain_independent_overlap_candidates() {
        let old_ed = rand_ed25519();
        let new_ed = rand_ed25519();
        let (_, old_pq) = ml_dsa_generate_keypair();
        let (_, new_pq) = ml_dsa_generate_keypair();
        let old_pq = ml_dsa_vk_bytes(&old_pq);
        let new_pq = ml_dsa_vk_bytes(&new_pq);
        let resolver = MethodDispatchResolver::new(FixtureDocs(did_doc_with_two_hybrid_pairs(
            &old_ed, &old_pq, &new_ed, &new_pq,
        )));

        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("hybrid overlap document resolves");
        let old = keys
            .candidate_for_ed25519(&old_ed)
            .expect("old hybrid candidate");
        let new = keys
            .candidate_for_ed25519(&new_ed)
            .expect("new hybrid candidate");
        assert_eq!(old.assurance(), Assurance::PqHybrid);
        assert_eq!(new.assurance(), Assurance::PqHybrid);
        assert_eq!(ml_dsa_vk_bytes(old.ml_dsa_65.as_ref().unwrap()), old_pq);
        assert_eq!(ml_dsa_vk_bytes(new.ml_dsa_65.as_ref().unwrap()), new_pq);
    }

    #[test]
    fn did_web_no_ed25519_vm_fails_closed() {
        // A doc with only an ML-DSA VM (no Ed25519 anchor) cannot be bound.
        let (_sk, vk) = ml_dsa_generate_keypair();
        let doc = json!({
            "id": DID_WEB,
            "verificationMethod": [{
                "id": "did:web:peer.example#mesh-pq",
                "publicKeyMultibase": encode_multikey(&ml_dsa_vk_bytes(&vk), MULTICODEC_ML_DSA_65_PUB),
            }],
        });
        let resolver = MethodDispatchResolver::new(FixtureDocs(doc));
        assert!(
            resolver
                .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
                .is_err()
        );
    }

    #[test]
    fn did_web_empty_doc_fails_closed() {
        let doc = json!({ "id": DID_WEB, "verificationMethod": [] });
        let resolver = MethodDispatchResolver::new(FixtureDocs(doc));
        assert!(
            resolver
                .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
                .is_err()
        );
    }

    #[test]
    fn did_web_fetch_failure_fails_closed() {
        let resolver = MethodDispatchResolver::new(FailingDocs);
        assert!(
            resolver
                .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
                .is_err()
        );
    }

    #[test]
    fn did_key_is_classical_without_fetch() {
        let ed = rand_ed25519();
        let did = ed25519_to_did_key(&ed);
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let keys = resolver
            .resolve_identity_keys(&Did::new(did))
            .expect("did:key resolves without fetch");
        let candidate = keys.candidate_for_ed25519(&ed).expect("did:key candidate");
        assert_eq!(candidate.assurance(), Assurance::Classical);
        assert!(candidate.ml_dsa_65.is_none());
    }

    #[test]
    fn did_key_malformed_fails_closed() {
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let did = Did::new("did:key:zNotAValidEd25519Multikey".to_owned());
        assert!(resolver.resolve_identity_keys(&did).is_err());
    }

    #[test]
    fn did_at9p_arm_fails_closed_without_a_capsule_resolver() {
        // No At9pCapsuleResolver configured ⇒ did:at9p fails closed (the arm
        // exists but has nothing to verify against).
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let did = Did::new("did:at9p:cid512abcdef".to_owned());
        // `IdentityKeys` is not `Debug` (the ML-DSA key type isn't), so match on
        // the result rather than `unwrap_err`.
        match resolver.resolve_identity_keys(&did) {
            Ok(_) => panic!("did:at9p must fail closed without a capsule resolver"),
            Err(err) => {
                assert!(
                    err.to_string().contains("fail-closed"),
                    "missing-resolver arm should fail closed: {err}"
                );
            }
        }
    }

    // ── did:at9p arm (D2/#894) ───────────────────────────────────────────────

    use super::{At9pCapsuleResolver, VerifiedAt9pKeys};
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// A capsule resolver that returns a fixed [`VerifiedAt9pKeys`] (the GATE is
    /// abstracted in rpc; pds owns the real implementation). Records the calls it
    /// received so tests can assert the arm routed through it.
    struct FixtureCapsule {
        keys: VerifiedAt9pKeys,
        calls: Mutex<Vec<String>>,
        fail: bool,
    }

    impl At9pCapsuleResolver for FixtureCapsule {
        fn resolve(&self, did: &str) -> Result<VerifiedAt9pKeys> {
            self.calls.lock().push(did.to_owned());
            if self.fail {
                Err(anyhow!("simulated GATE failure"))
            } else {
                Ok(self.keys.clone())
            }
        }
    }

    fn verified_keys() -> VerifiedAt9pKeys {
        let (_sk, vk) = ml_dsa_generate_keypair();
        VerifiedAt9pKeys::new_gate_verified([9u8; 32], vk)
    }

    #[test]
    fn did_at9p_with_verified_capsule_is_pqhybrid() {
        // A GATE-verified capsule yields PqHybrid assurance, carrying the
        // capsule's own Ed25519 + ML-DSA-65 keys (crypto-derived, not asserted).
        let keys = verified_keys();
        let fixture = Arc::new(FixtureCapsule {
            keys: keys.clone(),
            calls: Mutex::new(vec![]),
            fail: false,
        });
        let resolver = MethodDispatchResolver::new(NeverDocs).with_at9p(fixture.clone());
        let did = Did::new("did:at9p:cid512abcdef".to_owned());
        let resolved = resolver
            .resolve_identity_keys(&did)
            .expect("verified capsule resolves at PqHybrid");
        let only = &keys.keys()[0];
        let candidate = resolved
            .candidate_for_ed25519(only.ed25519())
            .expect("capsule candidate");
        assert_eq!(candidate.assurance(), Assurance::PqHybrid);
        assert_eq!(
            ml_dsa_vk_bytes(candidate.ml_dsa_65.as_ref().unwrap()),
            ml_dsa_vk_bytes(only.ml_dsa_65())
        );
        // The arm routed through the capsule resolver exactly once.
        assert_eq!(fixture.calls.lock().as_slice(), &["did:at9p:cid512abcdef"]);
    }

    #[test]
    fn did_at9p_multi_key_capsule_preserves_the_full_candidate_set() {
        let (_, old_pq) = ml_dsa_generate_keypair();
        let (_, new_pq) = ml_dsa_generate_keypair();
        let old = [10u8; 32];
        let new = [11u8; 32];
        let verified = VerifiedAt9pKeys::new_gate_verified_set(vec![
            VerifiedAt9pKeys::key(old, old_pq),
            VerifiedAt9pKeys::key(new, new_pq),
        ])
        .unwrap();
        let fixture = Arc::new(FixtureCapsule {
            keys: verified,
            calls: Mutex::new(vec![]),
            fail: false,
        });
        let resolved = MethodDispatchResolver::new(NeverDocs)
            .with_at9p(fixture)
            .resolve_identity_keys(&Did::new("did:at9p:cid512overlap".to_owned()))
            .expect("verified overlap set resolves");
        assert_eq!(resolved.len(), 2);
        assert_eq!(
            resolved.candidate_for_ed25519(&old).unwrap().assurance(),
            Assurance::PqHybrid
        );
        assert_eq!(
            resolved.candidate_for_ed25519(&new).unwrap().assurance(),
            Assurance::PqHybrid
        );
    }

    #[test]
    fn did_at9p_gate_failure_fails_closed() {
        // A capsule that fails the GATE (bad sig / wrong cid) is rejected — no
        // default assurance, no Ok.
        let fixture = Arc::new(FixtureCapsule {
            keys: verified_keys(),
            calls: Mutex::new(vec![]),
            fail: true,
        });
        let resolver = MethodDispatchResolver::new(NeverDocs).with_at9p(fixture);
        let did = Did::new("did:at9p:cid512abcdef".to_owned());
        assert!(resolver.resolve_identity_keys(&did).is_err());
    }

    #[test]
    fn did_at9p_assurance_is_not_asserted_by_other_arms() {
        // The did:web and did:key arms must never produce PqHybrid from a path
        // text or hint — only the verified-capsule arm does. (Provenance: a
        // did:web with no PQ VM is Classical, regardless of any at9p resolver.)
        let fixture = Arc::new(FixtureCapsule {
            keys: verified_keys(),
            calls: Mutex::new(vec![]),
            fail: false,
        });
        let ed = rand_ed25519();
        let resolver =
            MethodDispatchResolver::new(FixtureDocs(did_doc(&ed, None))).with_at9p(fixture);
        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("did:web resolves");
        assert_eq!(
            keys.candidate_for_ed25519(&ed).unwrap().assurance(),
            Assurance::Classical
        );
    }

    #[test]
    fn unknown_method_fails_closed() {
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let did = Did::new("did:plc:unsupported".to_owned());
        assert!(resolver.resolve_identity_keys(&did).is_err());
    }
}
