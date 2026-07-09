//! Authoritative direction for `did:web` / `did:key` ↔ `did:at9p` aliasing
//! (#896 / D4; design #879 Q4, ratified #905 §2/§6).
//!
//! This is the **pure, I/O-free core** of the D4 aliasing bridge — the rule
//! that decides, given a classical DID's `alsoKnownAs` claim and a
//! GATE-verified [`VerifiedCapsule`], whether the two identities are *mutually*
//! attested and — when they are — which one is authoritative. The I/O half
//! (fetch the capsule bytes from an untrusted locator, run GATE) composes this
//! rule and lives in `hyprstream-discovery::at9p_alias`, sibling to the D1
//! capsule resolver.
//!
//! # The authoritative-direction rule
//!
//! A `did:web` / `did:key` identity can declare (via `alsoKnownAs`) that it is
//! *also* a `did:at9p`. The reverse vouch lives inside the at9p capsule body
//! (the [`CapsuleBody::also_known_as`] field, #896 schema). Two legs therefore
//! exist, of **different strength**:
//!
//! | leg | carrier | signature strength |
//! |-----|---------|--------------------|
//! | at9p → classical | the GATE-verified capsule body's `alsoKnownAs` | **Hybrid** (EdDSA + ML-DSA-65, pinned) |
//! | classical → at9p | the classical DID document's `alsoKnownAs` | **Classical** (the doc's best) |
//!
//! The ratified semantics (#905 §2/§6) are:
//!
//! 1. **Bidirectional to be believed.** Both documents must name each other. A
//!    one-way claim is not trusted → [`Err`].
//! 2. **at9p is authoritative.** The content-verified capsule (BLAKE3-512
//!    self-certifying, hybrid-signed) is the stronger identity. When the
//!    attestation is mutual, the at9p identity is authoritative at `PqHybrid`
//!    assurance — the assurance of the capsule leg actually walked.
//! 3. **Assurance = the leg walked, never the max.** A verifier reaching the
//!    identity *through* the classical leg lands `Classical`; reaching it
//!    *through* the GATE-verified capsule lands `PqHybrid`. This rule is only
//!    ever evaluated with a [`VerifiedCapsule`] in hand (the capsule leg was
//!    walked), so its result is `PqHybrid`.
//!
//! at9p is the PQ **upgrade path**, additive to — not a replacement for — a
//! principal's classical atproto identity (the method-allowlist bridge: public
//! atproto infra accepts only `did:plc`/`did:web` as repo authorities, so
//! public records publish under the classical DID with `alsoKnownAs` to the
//! at9p identity; our own PDS accepts `did:at9p` natively, #908/G3).

use anyhow::{ensure, Result};

use hyprstream_rpc::auth::mac::Assurance;
use hyprstream_rpc::identity::Did;

use crate::at9p_gate::VerifiedCapsule;

/// The authoritative identity resolved through the at9p aliasing bridge.
///
/// Produced by [`resolve_authoritative_alias`] only when a classical DID and a
/// GATE-verified `did:at9p` capsule mutually attest each other. The at9p
/// identity is authoritative; `assurance` is `PqHybrid` (the capsule leg).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthoritativeIdentity {
    /// The content-verified `did:at9p:<cid512>` — the stronger identity.
    pub at9p_did: Did,
    /// The classical DID that bidirectionally attests to the at9p identity.
    pub classical_did: Did,
    /// The assurance established: `PqHybrid` (reached via the GATE-verified
    /// capsule leg — never the classical leg's best).
    pub assurance: Assurance,
}

/// Resolve the authoritative identity for `classical_did` via mutual
/// `alsoKnownAs` attestation with a GATE-verified `did:at9p` capsule.
///
/// - `classical_did` — the classical (`did:web` / `did:key` / `did:plc`)
///   identity the caller reached.
/// - `classical_aka_at9p` — the **classical leg's** claim: the `did:at9p` the
///   classical DID document names in its own `alsoKnownAs`. Classical-signed,
///   best-effort.
/// - `verified` — the **capsule leg**: the [`VerifiedCapsule`] fetched for
///   `classical_aka_at9p` and passed by the GATE pipeline. Its own
///   `alsoKnownAs` is the hybrid-signed reciprocal vouch.
///
/// Returns the authoritative at9p identity at `PqHybrid` assurance when both
/// legs name each other; **fail-closed** (`Err`) on any one-way or
/// unattested claim. See the module docs for the full rule.
pub fn resolve_authoritative_alias(
    classical_did: &Did,
    classical_aka_at9p: &Did,
    verified: &VerifiedCapsule,
) -> Result<AuthoritativeIdentity> {
    // The capsule is the authoritative leg — its did is the stronger identity.
    let at9p_did = Did::new(verified.did());

    // Leg 1 (classical): the classical DID document names THIS capsule as its
    // at9p alias. A classical leg pointing at a different capsule is a one-way
    // (or cross-identity) claim — fail-closed.
    ensure!(
        classical_aka_at9p == &at9p_did,
        "aliasing bridge: classical leg {} does not name the verified capsule {} \
         (one-way claim — fail-closed)",
        classical_aka_at9p,
        at9p_did,
    );

    // Leg 2 (hybrid): the GATE-verified capsule reciprocally names the
    // classical DID in its alsoKnownAs. Exact string match is the
    // conservative, fail-closed choice (no normalization that could widen trust).
    let reciprocated = capsule_names(verified, classical_did);
    ensure!(
        reciprocated,
        "aliasing bridge: verified capsule {} does not name {} in its alsoKnownAs \
         (one-way claim — fail-closed)",
        at9p_did,
        classical_did,
    );

    Ok(AuthoritativeIdentity {
        at9p_did,
        classical_did: classical_did.clone(),
        // The capsule leg was walked ⇒ PqHybrid. Never the classical leg's max.
        assurance: Assurance::PqHybrid,
    })
}

/// Whether a verified capsule's `alsoKnownAs` names `did`.
fn capsule_names(verified: &VerifiedCapsule, did: &Did) -> bool {
    verified
        .capsule()
        .body
        .also_known_as
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .any(|alias| Did::new(alias.clone()) == *did)
}

/// The classical aliases a verified `did:at9p` capsule attests to — the
/// **method-allowlist bridge** (#905 §6).
///
/// Public atproto infra (PLC directory, Bluesky, Tangled) allowlists only
/// `did:plc` / `did:web` as repo authorities, so public `at://` records
/// publish under the classical DID with `alsoKnownAs` to the `did:at9p`
/// identity. This accessor projects a verified capsule back to those classical
/// aliases — the at9p → classical direction — so a holder can publish under a
/// classical authority while the at9p identity remains the PQ upgrade path.
/// Returned in the capsule's declared order.
pub fn classical_aliases(verified: &VerifiedCapsule) -> Vec<Did> {
    verified
        .capsule()
        .body
        .also_known_as
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|alias| Did::new(alias.clone()))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
    };
    use crate::at9p_gate::verify_did_at9p;
    use crate::at9p_sign::sign_capsule;

    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};

    struct Signer {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer(tag: u8) -> Signer {
        let mut seed = [0u8; 32];
        seed[0] = tag;
        seed[31] = tag.wrapping_add(7);
        let ed_sk = SigningKey::from_bytes(&seed);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let keypair = HybridKeyPair::new(
            ed_sk.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_vk),
        )
        .unwrap();
        Signer {
            ed_sk,
            pq_sk,
            keypair,
        }
    }

    /// Build a GATE-verified capsule that attests `aliases` as its classical
    /// `alsoKnownAs`, returning (verified, did).
    fn verified_with_aliases(aliases: Vec<String>) -> (VerifiedCapsule, String) {
        let s = signer(1);
        let endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://node1").unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![s.keypair.clone()], vec![service]).unwrap();
        if !aliases.is_empty() {
            body.also_known_as = Some(aliases);
        }
        let capsule = sign_capsule(body, &s.ed_sk, &s.pq_sk).unwrap();
        let bytes = capsule.to_dag_cbor();
        let did = format!("did:at9p:{}", capsule.cid512().unwrap());
        let verified = verify_did_at9p(&did, &bytes).unwrap();
        (verified, did)
    }

    #[test]
    fn mutual_attestation_resolves_authoritative_at9p_at_pqhybrid() {
        let classical = Did::new("did:web:node.example".to_owned());
        // Capsule attests the classical alias (hybrid leg).
        let (verified, did) = verified_with_aliases(vec![classical.as_str().to_owned()]);
        let at9p = Did::new(did.clone());
        // Classical leg names the capsule (classical leg).
        let res = resolve_authoritative_alias(&classical, &at9p, &verified).unwrap();
        assert_eq!(res.at9p_did, at9p);
        assert_eq!(res.classical_did, classical);
        assert_eq!(res.assurance, Assurance::PqHybrid);
    }

    #[test]
    fn one_way_classical_claim_fails_closed() {
        let classical = Did::new("did:web:node.example".to_owned());
        let (verified, _did) = verified_with_aliases(vec![classical.as_str().to_owned()]);
        // Classical leg names a DIFFERENT capsule than the one verified.
        let other = Did::new("did:at9p:bafyDIFFERENT".to_owned());
        let err = resolve_authoritative_alias(&classical, &other, &verified).unwrap_err();
        assert!(
            err.to_string()
                .contains("does not name the verified capsule"),
            "{err}"
        );
    }

    #[test]
    fn one_way_capsule_claim_fails_closed() {
        // Capsule attests alice; the classical DID reaching us is bob — the
        // capsule does not reciprocate bob.
        let (verified, did) = verified_with_aliases(vec!["did:web:alice.example".to_owned()]);
        let at9p = Did::new(did);
        let bob = Did::new("did:web:bob.example".to_owned());
        let err = resolve_authoritative_alias(&bob, &at9p, &verified).unwrap_err();
        assert!(err.to_string().contains("does not name"), "{err}");
    }

    #[test]
    fn capsule_with_no_aliases_fails_closed() {
        let (verified, did) = verified_with_aliases(Vec::new()); // no aliases
        let at9p = Did::new(did);
        let classical = Did::new("did:web:node.example".to_owned());
        let err = resolve_authoritative_alias(&classical, &at9p, &verified).unwrap_err();
        assert!(err.to_string().contains("does not name"), "{err}");
    }

    #[test]
    fn classical_aliases_project_method_allowlist_bridge() {
        let a = "did:web:node.example".to_owned();
        let b = "did:key:z6Mkalias".to_owned();
        let (verified, _did) = verified_with_aliases(vec![a.clone(), b.clone()]);
        let aliases = classical_aliases(&verified);
        assert_eq!(
            aliases,
            vec![Did::new(a), Did::new(b)],
            "projected in declared order"
        );
    }
}
