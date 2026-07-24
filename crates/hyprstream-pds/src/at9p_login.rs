//! Public verification of a `did:at9p` **login assertion** — the pure,
//! I/O-free core behind the credential-free HTTPS endpoint (#1114).
//!
//! The consumer is an external web app that cannot speak Cap'n Proto and holds
//! no mesh credentials. It has fetched a genesis capsule over the static
//! well-known capsule transport (#1143), issued a fresh challenge to the holder,
//! received the holder's signature back, and now wants one answer: **is this
//! `did:at9p` login assertion authentic and live?** This module is that
//! answer; the axum face in `hyprstream` is a thin JSON translator over it.
//!
//! # Trust boundary — where each fact comes from
//!
//! Every key-material fact is derived from the **GATE-verified capsule**, never
//! from a DID document. The operator decision on #1157 (2026-07-22) is binding
//! here: the `did:web` document is **advisory** — reciprocal vouch and rotation
//! hint only. This module has no type for a DID-document key, so document-sourced
//! trust cannot be reintroduced at this layer even if a caller wanted to.
//!
//! Composition (each step fails closed; none is skippable):
//!
//! 1. **GATE** ([`crate::at9p_gate::verify_did_at9p`]) — `canon → hash → sig`.
//!    This is the self-certification binding: the identity *is* the BLAKE3-512
//!    hash of the capsule, so verification means recomputing the DID from the
//!    fetched bytes and comparing — not parsing the identifier string. The
//!    primary subject Ed25519 key used below is read from this
//!    [`VerifiedCapsule`], i.e. it is content-bound to the identity.
//! 2. **Liveness** — the holder's Ed25519 signature over a canonical,
//!    domain-separated binding of `(did, challenge, issued_at)`, verified
//!    against the GATE-verified subject key. The GATE explicitly "is not live
//!    possession"; this gate is. A missing, malformed, stale, or mismatched
//!    signature fails closed.
//! 3. **Aliasing (optional)** — only when the holder also claims a classical
//!    DID. The ratified #905 §2/§6 rule is *bidirectional or not believed*:
//!    [`hyprstream_pds::at9p_alias::resolve_authoritative_alias`] requires both
//!    legs to name each other. The classical leg's reciprocal vouch
//!    (`classical_aka_at9p`) is the advisory `alsoKnownAs` string the web app
//!    took from the `did:web` document; **no key material from that document is
//!    accepted** — the only authoritative leg is the GATE-verified capsule.
//!
//! # What a successful [`VerifiedLogin`] does and does not assert
//!
//! **Does assert:**
//! - The `did:at9p:<cid>` identifier is the BLAKE3-512 self-certifying hash of
//!   the supplied capsule bytes (GATE hash-gate).
//! - The capsule is authentically self-signed under pinned Hybrid
//!   (Ed25519 **and** ML-DSA-65) — GATE sig-gate.
//! - The holder possessed the capsule's primary subject Ed25519 private key at
//!   `issued_at` (liveness signature verified against the content-bound key).
//! - `issued_at` is within the freshness window of the verifier's clock.
//! - When `classical` is present and verification succeeded, the GATE-verified
//!   capsule and the classical DID mutually attest each other (`PqHybrid`).
//!
//! **Does NOT assert:**
//! - Current reachability, availability, or any transport endpoint for the
//!   identity (GATE proves the genesis claim, not liveness of any service —
//!   the discovery resolver still fails closed pending an independent
//!   EndpointId, #1031).
//! - That the classical DID document is itself authentic — the classical leg
//!   is advisory; a web app that needs the classical identity rooted must
//!   verify that document itself (this floor is `Assurance::Classical` when no
//!   alias is claimed, matching the #1114 contract).
//! - Anything about the holder's authorization, account standing, or tenant
//!   membership — this is a *verification* endpoint, not an authorization or
//!   session-minting endpoint. It produces no credential.
//!
//! # Fail-closed posture
//!
//! Every failure mode named in #1114 — capsule unfetchable, hash mismatch,
//! one-sided alias, expired or absent liveness proof — returns `Err`. The
//! function never returns a partial or downgraded `Ok` to "be helpful"; the
//! only success is a fully-verified assertion.

use anyhow::{ensure, Result};
use ed25519_dalek::{Signature, VerifyingKey};
use hyprstream_rpc::auth::mac::Assurance;
use hyprstream_rpc::identity::Did;

use crate::at9p::ED25519_PUBLIC_KEY_LEN;
use crate::at9p_alias::{resolve_authoritative_alias, AuthoritativeIdentity};
use crate::at9p_gate::{verify_did_at9p, VerifiedCapsule};

/// Domain separator bound into every liveness signature. Changing it
/// invalidates all outstanding assertions (intentional — it is a version tag).
pub const LOGIN_BINDING_DOMAIN: &str = "at9p-login-v1";

/// The Ed25519 signature length for the liveness proof.
pub const ED25519_LIVENESS_SIG_LEN: usize = 64;

/// A classical (`did:web` / `did:key` / `did:plc`) alias claim the holder
/// additionally makes when logging in.
///
/// Both fields are **advisory strings gathered by the web app from the
/// classical DID document** — never key material. `classical_aka_at9p` is what
/// the classical document names as its at9p alias (the classical→at9p leg);
/// the at9p→classical reciprocal leg is read from the GATE-verified capsule
/// inside [`verify_login_assertion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassicalClaim<'a> {
    /// The classical DID the holder claims to also be.
    pub classical_did: &'a str,
    /// The `alsoKnownAs` entry the classical DID document names for its at9p
    /// alias — the classical→at9p leg. Advisory; sourced from the classical
    /// document, not trusted for key material.
    pub classical_aka_at9p: &'a str,
}

/// A `did:at9p` login assertion presented by a holder to a web app.
///
/// All fields are borrowed views over caller-owned data so the pure core does
/// no allocation of its own and cannot outlive the request. The web app
/// supplies every input; the verifier trusts only the GATE-verified capsule
/// for key material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoginAssertion<'a> {
    /// The claimed `did:at9p:<cid512>` identifier. Compared against the DID
    /// recomputed from `capsule_bytes` at GATE hash-gate.
    pub did: &'a str,
    /// Raw canonical DAG-CBOR bytes of the genesis capsule the web app fetched
    /// for `did`. Attacker-controlled until GATE accepts them.
    pub capsule_bytes: &'a [u8],
    /// The fresh challenge string the web app issued to the holder for this
    /// login attempt. Bound into the liveness signature.
    pub challenge: &'a str,
    /// The holder's claimed signing instant, Unix seconds. Must be within the
    /// verifier's freshness window (`|now - issued_at| <= max_skew_seconds`).
    pub issued_at: u64,
    /// The holder's Ed25519 signature over
    /// [`login_binding`](login_binding) (`did`, `challenge`, `issued_at`),
    /// made with the capsule's primary subject key.
    pub liveness_signature: &'a [u8],
    /// Optional classical alias claim. When `None`, a successful verification
    /// floors at [`Assurance::Classical`] (#1114). When `Some`, both aliasing
    /// legs must verify or the assertion fails closed.
    pub classical: Option<ClassicalClaim<'a>>,
}

/// The result of a fully-verified login assertion — a witness that every gate
/// (GATE, liveness, and — when claimed — bidirectional aliasing) passed.
///
/// Constructible only by [`verify_login_assertion`]. Holding this value is
/// proof of the facts listed in the module docs; it authorizes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLogin {
    /// The content-verified `did:at9p:<cid512>` (recomputed from the capsule,
    /// not the claimed string).
    pub did: Did,
    /// The assurance floor reached. `Classical` when no alias was claimed;
    /// `PqHybrid` when a classical alias was mutually attested.
    pub assurance: Assurance,
    /// The GATE-verified primary subject Ed25519 key — the key the liveness
    /// signature verified against. Content-bound to `did`.
    pub subject_ed25519: [u8; ED25519_PUBLIC_KEY_LEN],
    /// The authoritative classical alias, present only when the holder claimed
    /// one and the bidirectional #905 §2/§6 rule held.
    pub classical: Option<AuthoritativeIdentity>,
}

/// Injects "now" so the pure core has no hidden clock dependency and tests are
/// deterministic. Implementors must return Unix seconds.
pub trait Clock {
    /// The verifier's current time, Unix seconds.
    fn now_unix_seconds(&self) -> u64;
}

/// Canonical, domain-separated bytes the holder signs for the liveness proof.
///
/// Fixed layout: `"{domain}\n{did}\n{challenge}\n{issued_at}"` where
/// `issued_at` is decimal Unix seconds. Every field is unambiguous (the domain
/// tags the whole string, `did`/`challenge` cannot collide with the decimal
/// timestamp field), and the string is stable across signer and verifier
/// regardless of locale or JSON encoding.
pub fn login_binding(did: &str, challenge: &str, issued_at: u64) -> Vec<u8> {
    format!("{LOGIN_BINDING_DOMAIN}\n{did}\n{challenge}\n{issued_at}").into_bytes()
}

/// Verify a `did:at9p` login assertion end-to-end.
///
/// See the [module docs](self) for the full trust boundary, the composition
/// order, and what a successful [`VerifiedLogin`] does and does not assert.
/// `clock` supplies "now"; `max_skew_seconds` bounds the freshness window
/// symmetrically (the holder may be slightly ahead or behind the verifier).
///
/// Returns `Err` for every failure mode — there is no partial success.
pub fn verify_login_assertion<C: Clock>(
    assertion: &LoginAssertion<'_>,
    clock: &C,
    max_skew_seconds: u64,
) -> Result<VerifiedLogin> {
    // ── Gate 1: GATE (canon → hash → sig). The identity IS the hash of the
    // bytes; this is where every key-material fact below is rooted. A capsule
    // the web app "could not fetch" arrives as empty/garbage bytes and fails
    // here (canon- or hash-gate).
    let verified: VerifiedCapsule = verify_did_at9p(assertion.did, assertion.capsule_bytes)?;
    let recomputed_did = verified.did();
    let capsule = verified.capsule();

    let primary = capsule
        .body
        .subject_keys
        .first()
        .ok_or_else(|| anyhow::anyhow!("at9p capsule carries no subject key"))?;
    let ed_bytes: [u8; ED25519_PUBLIC_KEY_LEN] = primary
        .ed25519_pub
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("ed25519 subject key is not 32 bytes"))?;
    let ed_vk = VerifyingKey::from_bytes(&ed_bytes)
        .map_err(|_| anyhow::anyhow!("content-bound ed25519 key is not a valid point"))?;

    // ── Gate 2: liveness. The GATE "is not live possession"; this gate is.
    // Absent or wrong-length signature, signature mismatch, or a stale
    // timestamp all fail closed. The signature MUST verify against the
    // GATE-verified key — never a key supplied by the caller.
    let sig: [u8; ED25519_LIVENESS_SIG_LEN] = assertion
        .liveness_signature
        .try_into()
        .map_err(|_| anyhow::anyhow!("liveness signature must be {ED25519_LIVENESS_SIG_LEN} bytes"))?;
    let sig = Signature::from_bytes(&sig);
    let binding = login_binding(&recomputed_did, assertion.challenge, assertion.issued_at);
    ed_vk
        .verify_strict(&binding, &sig)
        .map_err(|_| anyhow::anyhow!("liveness signature does not verify against the GATE-verified subject key"))?;

    // Freshness window. `abs_diff` is symmetric so clock skew in either
    // direction is tolerated up to the bound; outside it, expired.
    let now = clock.now_unix_seconds();
    let issued_at = assertion.issued_at;
    let drift = now.abs_diff(issued_at);
    ensure!(
        drift <= max_skew_seconds,
        "liveness proof is stale: issued_at {issued_at} is {drift}s from now {now} (max skew {max_skew_seconds}s)"
    );

    // ── Gate 3 (optional): bidirectional aliasing. The classical leg is
    // advisory strings only — no key material crosses. Both legs must name
    // each other or the assertion fails closed as a one-sided claim.
    let (assurance, classical) = match assertion.classical {
        None => (Assurance::Classical, None),
        Some(claim) => {
            let classical_did = Did::new(claim.classical_did.to_owned());
            let classical_aka_at9p = Did::new(claim.classical_aka_at9p.to_owned());
            let authoritative = resolve_authoritative_alias(
                &classical_did,
                &classical_aka_at9p,
                &verified.clone(),
            )?;
            (Assurance::PqHybrid, Some(authoritative))
        }
    };

    Ok(VerifiedLogin {
        did: Did::new(recomputed_did),
        assurance,
        subject_ed25519: ed_bytes,
        classical,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::at9p::{
        Capsule, CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
    };
    use crate::at9p_gate::DID_AT9P_PREFIX;
    use crate::at9p_sign::sign_capsule;

    use ed25519_dalek::{Signer, SigningKey};
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};

    /// A fixed, mutable verifier clock so freshness tests are deterministic.
    #[derive(Default)]
    struct FixedClock {
        now: u64,
    }
    impl Clock for FixedClock {
        fn now_unix_seconds(&self) -> u64 {
            self.now
        }
    }

    struct Signer_ {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer(tag: u8) -> Signer_ {
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
        Signer_ {
            ed_sk,
            pq_sk,
            keypair,
        }
    }

    fn body_for(s: &Signer_, tag: u8, aliases: Vec<String>) -> CapsuleBody {
        let endpoint = ServiceEndpoint::new(Transport::Iroh, format!("iroh://node{tag}")).unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![s.keypair.clone()], vec![service]).unwrap();
        if !aliases.is_empty() {
            body.also_known_as = Some(aliases);
        }
        body
    }

    /// A self-signed capsule, its canonical bytes, its DID, and the signer.
    fn signed(tag: u8, aliases: Vec<String>) -> (Capsule, Vec<u8>, String, Signer_) {
        let s = signer(tag);
        let body = body_for(&s, tag, aliases);
        let capsule = sign_capsule(body, &s.ed_sk, &s.pq_sk).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("{DID_AT9P_PREFIX}{}", capsule.cid512().unwrap());
        (capsule, bytes, did, s)
    }

    fn sign_liveness(sk: &SigningKey, did: &str, challenge: &str, issued_at: u64) -> [u8; 64] {
        let binding = login_binding(did, challenge, issued_at);
        sk.sign(&binding).to_bytes()
    }

    const NOW: u64 = 1_700_000_000;
    const SKEW: u64 = 300;

    /// Baseline: a fully valid assertion verifies at Classical assurance.
    #[test]
    fn valid_assertion_verifies_classical() {
        let (_capsule, bytes, did, s) = signed(1, Vec::new());
        let sig = sign_liveness(&s.ed_sk, &did, "challenge-abc", NOW);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "challenge-abc",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: None,
        };
        let v = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW).unwrap();
        assert_eq!(v.did.as_str(), did);
        assert_eq!(v.assurance, Assurance::Classical);
        assert!(v.classical.is_none());
        assert_eq!(v.subject_ed25519, s.ed_sk.verifying_key().to_bytes());
    }

    /// Fail-closed: capsule bytes do not hash to the claimed DID (hash
    /// mismatch). REVERT the GATE-call hunk and this test passes — proving the
    /// hash check is load-bearing here, not incidental.
    #[test]
    fn hash_mismatch_fails_closed() {
        let (_a, bytes_a, _did_a, _s_a) = signed(2, Vec::new());
        let (_b, _bytes_b, did_b, s_b) = signed(3, Vec::new());
        // A signature valid under B's key, but we present A's bytes under B's
        // identity — GATE hash-gate must reject before liveness is even read.
        let sig = sign_liveness(&s_b.ed_sk, &did_b, "c", NOW);
        let assertion = LoginAssertion {
            did: &did_b,
            capsule_bytes: &bytes_a,
            challenge: "c",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: None,
        };
        let err = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("hash-gate") || err.contains("GATE rejected"),
            "expected a GATE/hash rejection, got: {err}"
        );
    }

    /// Fail-closed: "capsule unfetchable" — malformed/empty bytes fail GATE.
    #[test]
    fn unfetchable_capsule_fails_closed() {
        let (_c, _bytes, did, s) = signed(4, Vec::new());
        let sig = sign_liveness(&s.ed_sk, &did, "c", NOW);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: b"\xff\xff not a capsule",
            challenge: "c",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: None,
        };
        assert!(verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW).is_err());
    }

    /// Fail-closed: absent liveness proof (empty signature).
    #[test]
    fn absent_liveness_fails_closed() {
        let (_c, bytes, did, _s) = signed(5, Vec::new());
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "c",
            issued_at: NOW,
            liveness_signature: &[],
            classical: None,
        };
        let err = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW)
            .unwrap_err()
            .to_string();
        assert!(err.contains("liveness signature must be"), "{err}");
    }

    /// Fail-closed: wrong-key signature (a different identity's key signs).
    /// The liveness gate MUST verify against the GATE-verified key, not accept
    /// any plausible signature.
    #[test]
    fn wrong_key_liveness_fails_closed() {
        let (_c, bytes, did, _s) = signed(6, Vec::new());
        let other = signer(99);
        let sig = sign_liveness(&other.ed_sk, &did, "c", NOW);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "c",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: None,
        };
        let err = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not verify against the GATE-verified subject key"),
            "{err}"
        );
    }

    /// Fail-closed: expired (issued_at outside the freshness window).
    #[test]
    fn expired_liveness_fails_closed() {
        let (_c, bytes, did, s) = signed(7, Vec::new());
        let stale = NOW - (SKEW + 60);
        let sig = sign_liveness(&s.ed_sk, &did, "c", stale);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "c",
            issued_at: stale,
            liveness_signature: &sig,
            classical: None,
        };
        let err = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW)
            .unwrap_err()
            .to_string();
        assert!(err.contains("liveness proof is stale"), "{err}");
    }

    /// Boundary: issued_at exactly max_skew_seconds away is still accepted.
    #[test]
    fn freshness_window_boundary_accepted() {
        let (_c, bytes, did, s) = signed(8, Vec::new());
        let at = NOW - SKEW;
        let sig = sign_liveness(&s.ed_sk, &did, "c", at);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "c",
            issued_at: at,
            liveness_signature: &sig,
            classical: None,
        };
        assert!(verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW).is_ok());
    }

    /// Fail-closed: one-sided alias — the capsule does NOT name the claimed
    /// classical DID back. REVERT the alias branch and this test passes.
    #[test]
    fn one_sided_alias_fails_closed() {
        // Capsule attests alice; holder claims to also be bob.
        let (_c, bytes, did, s) = signed(9, vec!["did:web:alice.example".to_owned()]);
        let sig = sign_liveness(&s.ed_sk, &did, "c", NOW);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "c",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: Some(ClassicalClaim {
                classical_did: "did:web:bob.example",
                // Bob's doc (advisory) names this capsule — classical leg ok,
                // but the at9p→classical leg names alice, not bob.
                classical_aka_at9p: &did,
            }),
        };
        let err = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW)
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not name"), "{err}");
    }

    /// Fail-closed: the classical leg names a *different* capsule than the one
    /// verified — the reverse one-sided failure.
    #[test]
    fn classical_leg_names_wrong_capsule_fails_closed() {
        let (_c, bytes, did, s) =
            signed(10, vec!["did:web:node.example".to_owned()]);
        let sig = sign_liveness(&s.ed_sk, &did, "c", NOW);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "c",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: Some(ClassicalClaim {
                classical_did: "did:web:node.example",
                // Classical doc (advisory) names a different at9p than the one
                // verified — classical leg fails.
                classical_aka_at9p: "did:at9p:bafyDIFFERENT",
            }),
        };
        let err = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not name the verified capsule"),
            "{err}"
        );
    }

    /// Mutual alias upgrades to PqHybrid.
    #[test]
    fn mutual_alias_upgrades_to_pqhybrid() {
        let classical = "did:web:node.example";
        let (_c, bytes, did, s) = signed(11, vec![classical.to_owned()]);
        let sig = sign_liveness(&s.ed_sk, &did, "c", NOW);
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "c",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: Some(ClassicalClaim {
                classical_did: classical,
                classical_aka_at9p: &did,
            }),
        };
        let v = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW).unwrap();
        assert_eq!(v.assurance, Assurance::PqHybrid);
        let auth = v.classical.as_ref().expect("alias resolved");
        assert_eq!(auth.classical_did.as_str(), classical);
        assert_eq!(auth.at9p_did.as_str(), did);
    }

    /// The liveness binding is domain-separated: a signature over the bare
    /// challenge (no domain/did/timestamp) must NOT verify, even with the right
    /// key — defends against cross-protocol signature reuse.
    #[test]
    fn bare_challenge_signature_rejected() {
        let (_c, bytes, did, s) = signed(12, Vec::new());
        // Sign just the challenge string, not the binding.
        let sig = s.ed_sk.sign(b"challenge-abc").to_bytes();
        let assertion = LoginAssertion {
            did: &did,
            capsule_bytes: &bytes,
            challenge: "challenge-abc",
            issued_at: NOW,
            liveness_signature: &sig,
            classical: None,
        };
        let err = verify_login_assertion(&assertion, &FixedClock { now: NOW }, SKEW)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not verify against the GATE-verified subject key"),
            "{err}"
        );
    }
}
