//! **Delegation-chain + ceiling/attenuation validation** — the core correctness
//! component of S5 (#571).
//!
//! Per the epic (#547, design "ceiling/attenuation"): *every* delegation in a
//! UCAN chain must be a SUBSET (attenuation) of its delegator's authority — a
//! child can never widen. This module walks the proof chain of a UCAN and
//! enforces, at each link, the two structural invariants that make the chain a
//! genuine ceiling:
//!
//! 1. **Linkage:** the delegate UCAN's *issuer* must equal the delegator
//!    (proof) UCAN's *audience*. Authority flows `proof.audience → ucan.issuer`;
//!    a chain where the issuer was not the party the proof delegated to is
//!    forged.
//! 2. **Attenuation (⊆):** every capability the delegate UCAN claims must be
//!    authorized by the delegator's capability set
//!    ([`super::capability::set_attenuates`]). A delegate that widens on
//!    resource, ability, or caveats is rejected.
//!
//! Plus the obvious gates: temporal validity (a child cannot outlive its parent
//! — its validity window must be within the parent's), and recursion to the
//! root (a root has no proofs and grants whatever it self-asserts).
//!
//! **Fail-closed everywhere.** Any malformation, missing link, widening, or
//! out-of-window condition rejects the entire chain. This is intentionally
//! conservative TCB code: a false reject is a denied request; a false accept is
//! a privilege escalation, so the asymmetry is always resolved toward rejection.
//!
//! ## Scope (milestone 1)
//!
//! Signature verification is assumed already done by
//! [`super::token::Ucan::verify_signatures`] — `validate_chain` is *structural*
//! (linkage + attenuation + time). Callers run signatures first, then this. The
//! combined entry point [`validate`] does both in the correct order. Revocation
//! lists and a richer multi-proof "any-path" model are milestone-2 seams; M1
//! validates the single-delegator-per-link chain (each UCAN's `proofs[i]` is *a*
//! delegator that must independently cover the claims).

use super::capability::set_attenuates;
use super::token::{Ucan, UcanError, UcanVerifier};
use std::fmt;

/// Why a delegation chain was rejected. Every variant is a fail-closed denial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// A signature failed to verify (wraps [`UcanError`]).
    Signature(UcanError),
    /// Structural malformation (wraps [`UcanError`]).
    Structure(UcanError),
    /// The delegate's issuer is not the delegator's audience — broken linkage.
    BrokenLinkage {
        /// The delegate UCAN's issuer (who claims to wield the authority).
        delegate_issuer: String,
        /// The delegator (proof) UCAN's audience (who the authority was given to).
        delegator_audience: String,
    },
    /// The delegate claims a capability its delegator does not authorize — an
    /// attempt to WIDEN authority across a delegation. The cardinal violation.
    Widening {
        /// 0-based index of the offending link from the leaf (0 = leaf→its proof).
        link: usize,
    },
    /// The child's validity window is not contained within its delegator's
    /// (a child cannot be valid when its parent is not).
    WindowEscape {
        /// 0-based link index.
        link: usize,
    },
    /// A non-root UCAN carried no proofs (cannot be authorized) — only a root
    /// may have an empty proof set, and a root is the *top* of the chain.
    MissingProof,
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainError::Signature(e) => write!(f, "chain signature invalid: {e}"),
            ChainError::Structure(e) => write!(f, "chain structurally invalid: {e}"),
            ChainError::BrokenLinkage { delegate_issuer, delegator_audience } => write!(
                f,
                "broken delegation linkage: delegate issuer {delegate_issuer} != delegator audience {delegator_audience}"
            ),
            ChainError::Widening { link } => {
                write!(f, "attenuation violation (widening) at chain link {link}")
            }
            ChainError::WindowEscape { link } => {
                write!(f, "validity window escapes delegator at chain link {link}")
            }
            ChainError::MissingProof => {
                write!(f, "non-root UCAN has no delegating proof (cannot authorize)")
            }
        }
    }
}

impl std::error::Error for ChainError {}

/// Is the child's validity window contained within the parent's? A child may not
/// start before the parent or end after it. `None` bounds are treated as the
/// widest possible on that side, so:
/// - child unbounded-below + parent bounded-below ⇒ escape.
/// - child unbounded-above + parent bounded-above ⇒ escape.
fn window_within(child: &Ucan, parent: &Ucan) -> bool {
    // not_before: child must not begin earlier than parent.
    let nbf_ok = match (child.payload.not_before, parent.payload.not_before) {
        (_, None) => true,        // parent unbounded below
        (None, Some(_)) => false, // child unbounded below, parent not
        (Some(c), Some(p)) => c >= p,
    };
    // expiration: child must not end later than parent.
    let exp_ok = match (child.payload.expiration, parent.payload.expiration) {
        (_, None) => true,        // parent unbounded above
        (None, Some(_)) => false, // child unbounded above, parent not
        (Some(c), Some(p)) => c <= p,
    };
    nbf_ok && exp_ok
}

/// Validate ONE delegation edge: `delegate` is authorized by `delegator`
/// (`delegate`'s proof). Enforces linkage, attenuation, and window containment.
/// `link` is the index used in error reporting.
fn validate_edge(delegate: &Ucan, delegator: &Ucan, link: usize) -> Result<(), ChainError> {
    // 1. Linkage: authority flows delegator.audience → delegate.issuer.
    if delegate.issuer() != delegator.audience() {
        return Err(ChainError::BrokenLinkage {
            delegate_issuer: delegate.issuer().0.clone(),
            delegator_audience: delegator.audience().0.clone(),
        });
    }
    // 2. Attenuation: the delegator's capabilities must cover the delegate's.
    if !set_attenuates(delegator.capabilities(), delegate.capabilities()) {
        return Err(ChainError::Widening { link });
    }
    // 3. Window containment: child cannot outlive its parent.
    if !window_within(delegate, delegator) {
        return Err(ChainError::WindowEscape { link });
    }
    Ok(())
}

/// Recursively validate the delegation chain rooted at `ucan` (structural only —
/// run signatures separately, or use [`validate`]).
///
/// Walks every proof: for each `proof` in `ucan.proofs`, validates the edge
/// `ucan ⟶ proof` (linkage + attenuation + window), then recurses into the
/// proof's own chain. A UCAN with no proofs is treated as a **root** and
/// authorizes whatever it self-asserts (its capabilities become the ceiling).
///
/// `link` is the depth from the original leaf, threaded for diagnostics.
pub fn validate_chain(ucan: &Ucan) -> Result<(), ChainError> {
    validate_chain_at(ucan, 0)
}

fn validate_chain_at(ucan: &Ucan, link: usize) -> Result<(), ChainError> {
    if ucan.proofs.is_empty() {
        // Root: self-asserted authority is the top of the ceiling. Nothing above
        // to attenuate against. (Whether a given root DID is *trusted* to assert
        // is the verifier/trust-anchor concern, handled at signature time and by
        // the approval binding — not structural attenuation.)
        return Ok(());
    }
    for proof in &ucan.proofs {
        validate_edge(ucan, proof, link)?;
        validate_chain_at(proof, link + 1)?;
    }
    Ok(())
}

/// The full fail-closed validation a compiler front-end runs before lowering a
/// UCAN into a ceiling: **(1) structure, (2) signatures, (3) delegation chain +
/// attenuation**, in that order. Returns `Ok(())` only if every link is
/// structurally sound, correctly signed under the hybrid policy, properly
/// linked, and strictly attenuating.
///
/// This is the single entry point S5's compiler calls; it guarantees that only a
/// genuine ceiling reaches the (deferred) bundle-emission stage.
pub fn validate<V: UcanVerifier>(ucan: &Ucan, verifier: &V) -> Result<(), ChainError> {
    ucan.validate_structure().map_err(ChainError::Structure)?;
    ucan.verify_signatures(verifier)
        .map_err(ChainError::Signature)?;
    validate_chain(ucan)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::token::test_support::*;
    use super::*;
    use crate::auth::ucan::capability::{Ability, Capability, CaveatValue, Caveats, Resource};
    use crate::auth::ucan::token::{Did, Ucan, UcanPayload};
    use crate::crypto::cose_sign::sign_composite;
    use std::collections::BTreeMap;

    /// Re-sign a (possibly mutated) payload so we can test *structural* chain
    /// rejections without the signature failing first.
    fn resign(issuer: &TestIdentity, payload: UcanPayload, proofs: Vec<Ucan>) -> Ucan {
        let bytes = payload.signing_bytes().unwrap();
        let signature =
            sign_composite(&issuer.ed_sk, Some(&issuer.pq_sk), &bytes, UCAN_AAD).unwrap();
        Ucan {
            payload,
            proofs,
            signature,
        }
    }

    fn store(ids: &[&TestIdentity]) -> TestTrustStore {
        let mut s = TestTrustStore::new();
        for id in ids {
            s.anchor(id);
        }
        s
    }

    // ---- Happy path: a properly attenuating chain ------------------------

    #[test]
    fn valid_attenuating_chain_accepted() {
        // root: mac://* / *   →  alice: mac://model/* / model/*  →  bob: mac://model/qwen / model/read
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();

        let root_ucan = signed_ucan(&root, &alice.did(), vec![cap("mac://*", "*")], vec![]);
        let alice_ucan = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://model/*", "model/*")],
            vec![root_ucan],
        );
        let bob_ucan = signed_ucan(
            &bob,
            &bob.did(),
            vec![cap("mac://model/qwen", "model/read")],
            vec![alice_ucan],
        );

        let s = store(&[&root, &alice, &bob]);
        assert!(
            validate(&bob_ucan, &s).is_ok(),
            "a strictly-attenuating chain must validate"
        );
    }

    #[test]
    fn root_with_no_proofs_is_self_authorizing() {
        let root = TestIdentity::generate();
        let u = signed_ucan(&root, &root.did(), vec![cap("mac://anything", "*")], vec![]);
        let s = store(&[&root]);
        assert!(validate(&u, &s).is_ok());
    }

    // ---- Widening: the cardinal rejection --------------------------------

    #[test]
    fn widening_resource_rejected() {
        // alice was delegated mac://model/* but bob claims mac://other/x.
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let root_ucan = signed_ucan(&root, &alice.did(), vec![cap("mac://model/*", "*")], vec![]);
        let alice_ucan = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://other/x", "model/read")],
            vec![root_ucan],
        );
        let s = store(&[&root, &alice]);
        match validate(&alice_ucan, &s) {
            Err(ChainError::Widening { link: 0 }) => {}
            other => panic!("expected Widening at link 0, got {other:?}"),
        }
    }

    #[test]
    fn widening_ability_rejected() {
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let root_ucan = signed_ucan(
            &root,
            &alice.did(),
            vec![cap("mac://*", "model/read")],
            vec![],
        );
        // bob tries to gain model/* from a model/read delegation.
        let alice_ucan = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://model/qwen", "model/*")],
            vec![root_ucan],
        );
        let s = store(&[&root, &alice]);
        assert!(matches!(
            validate(&alice_ucan, &s),
            Err(ChainError::Widening { .. })
        ));
    }

    #[test]
    fn widening_deep_in_chain_rejected() {
        // root→alice valid, alice→bob widens. Reject at link 1 (the alice→its-proof edge
        // is fine; the bob→alice edge — link 0 here — is the widening one).
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let root_ucan = signed_ucan(
            &root,
            &alice.did(),
            vec![cap("mac://model/*", "model/*")],
            vec![],
        );
        let alice_ucan = signed_ucan(
            &alice,
            &bob.did(),
            vec![cap("mac://model/qwen", "model/read")],
            vec![root_ucan],
        );
        // bob legitimately attenuates from alice, but we then tamper alice's claim to widen
        // beyond root and re-sign so signatures still pass.
        let mut alice_payload = alice_ucan.payload.clone();
        alice_payload.capabilities = vec![cap("mac://admin/*", "admin/*")]; // widens past root
        let widened_alice = resign(&alice, alice_payload, alice_ucan.proofs.clone());
        let bob_ucan = signed_ucan(
            &bob,
            &bob.did(),
            vec![cap("mac://admin/x", "admin/*")],
            vec![widened_alice],
        );
        let s = store(&[&root, &alice, &bob]);
        // The bob→alice edge passes (bob ⊆ widened alice), but alice→root widens at link 1.
        assert!(matches!(
            validate(&bob_ucan, &s),
            Err(ChainError::Widening { link: 1 })
        ));
    }

    #[test]
    fn caveat_drop_is_widening() {
        // root delegates with a tenant caveat; alice drops it.
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let mut cv = BTreeMap::new();
        cv.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        let restricted = Capability::with_caveats(
            Resource::new("mac://model/*"),
            Ability::new("*"),
            Caveats(cv),
        );
        let root_ucan = signed_ucan(&root, &alice.did(), vec![restricted], vec![]);
        // alice claims the same resource/ability but WITHOUT the tenant caveat → widening.
        let alice_ucan = signed_ucan(
            &alice,
            &alice.did(),
            vec![cap("mac://model/qwen", "infer")],
            vec![root_ucan],
        );
        let s = store(&[&root, &alice]);
        assert!(matches!(
            validate(&alice_ucan, &s),
            Err(ChainError::Widening { .. })
        ));
    }

    // ---- Linkage ---------------------------------------------------------

    #[test]
    fn broken_linkage_rejected() {
        // root delegates to alice, but bob (not alice) presents root as its proof.
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let bob = TestIdentity::generate();
        let root_to_alice = signed_ucan(&root, &alice.did(), vec![cap("mac://*", "*")], vec![]);
        // bob issues a UCAN citing root_to_alice as proof — but bob != alice.
        let bob_ucan = signed_ucan(
            &bob,
            &bob.did(),
            vec![cap("mac://model/x", "infer")],
            vec![root_to_alice],
        );
        let s = store(&[&root, &bob]);
        assert!(matches!(
            validate(&bob_ucan, &s),
            Err(ChainError::BrokenLinkage { .. })
        ));
    }

    // ---- Temporal window -------------------------------------------------

    #[test]
    fn child_outliving_parent_rejected() {
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        // root expires at 1000.
        let mut root_payload = UcanPayload {
            issuer: root.did(),
            audience: alice.did(),
            capabilities: vec![cap("mac://*", "*")],
            not_before: None,
            expiration: Some(1000),
            nonce: vec![],
        };
        let root_ucan = resign(&root, root_payload.clone(), vec![]);
        // alice expires at 2000 — outlives root → window escape.
        let alice_payload = UcanPayload {
            issuer: alice.did(),
            audience: alice.did(),
            capabilities: vec![cap("mac://model/x", "infer")],
            not_before: None,
            expiration: Some(2000),
            nonce: vec![],
        };
        let alice_ucan = resign(&alice, alice_payload, vec![root_ucan]);
        let s = store(&[&root, &alice]);
        assert!(matches!(
            validate(&alice_ucan, &s),
            Err(ChainError::WindowEscape { .. })
        ));

        // Same window (alice within root) is fine.
        root_payload.expiration = Some(2000);
        let root_ok = resign(&root, root_payload, vec![]);
        let alice_ok_payload = UcanPayload {
            issuer: alice.did(),
            audience: alice.did(),
            capabilities: vec![cap("mac://model/x", "infer")],
            not_before: None,
            expiration: Some(2000),
            nonce: vec![],
        };
        let alice_ok = resign(&alice, alice_ok_payload, vec![root_ok]);
        let _ = Did::new("did:key:placeholder"); // keep Did import used
        assert!(validate(&alice_ok, &store(&[&root, &alice])).is_ok());
    }

    // ---- Signature ordering ----------------------------------------------

    #[test]
    fn validate_runs_signatures_before_chain() {
        // A widening chain whose signatures are ALSO invalid should report a
        // signature error (signatures gate first).
        let root = TestIdentity::generate();
        let alice = TestIdentity::generate();
        let root_ucan = signed_ucan(&root, &alice.did(), vec![cap("mac://model/*", "*")], vec![]);
        let mut alice_ucan = signed_ucan(
            &alice,
            &alice.did(),
            vec![cap("mac://other/x", "*")],
            vec![root_ucan],
        );
        alice_ucan.signature[0] ^= 0xFF; // break alice's signature
        let s = store(&[&root, &alice]);
        assert!(matches!(
            validate(&alice_ucan, &s),
            Err(ChainError::Signature(_))
        ));
    }
}
