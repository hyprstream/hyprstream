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
//! Plus the temporal and structural gates, which are TWO independent checks:
//!
//! - **Relative window containment** ([`window_within`]): a child cannot outlive
//!   its parent — its `[not_before, expiration]` window must be within the
//!   delegator's. This bounds the chain *shape* but says nothing about the
//!   wall-clock.
//! - **Absolute validity at `now`** ([`super::token::Ucan::is_valid_at`]): EVERY
//!   link must be live at the validation instant — not expired and not
//!   future-dated (`not_before <= now <= expiration`). Relative containment alone
//!   is NOT sufficient: a fully-attenuating, correctly-linked chain in which
//!   every link has already EXPIRED still satisfies window containment, so
//!   without the absolute check it would validate and could be replayed/compiled
//!   into a ceiling. Both gates are therefore enforced, per link.
//!
//! And recursion to the root (a root has no proofs and grants whatever it
//! self-asserts), with a bounded proof depth ([`MAX_PROOF_DEPTH`]) so a
//! maliciously deep nested-proof chain cannot exhaust the stack (a DoS guard, not
//! an authority concern).
//!
//! **Fail-closed everywhere.** Any malformation, missing link, widening,
//! out-of-window, expired/not-yet-valid, or over-depth condition rejects the
//! entire chain. This is intentionally conservative TCB code: a false reject is a
//! denied request; a false accept is a privilege escalation, so the asymmetry is
//! always resolved toward rejection.
//!
//! ## Scope (milestone 1)
//!
//! Signature verification is assumed already done by
//! [`super::token::Ucan::verify_signatures`] — `validate_chain` covers linkage,
//! attenuation, relative window containment, absolute validity at `now`, and the
//! depth bound. Callers run signatures first, then this; the combined entry point
//! [`validate`] does both in the correct order. Revocation lists and a richer
//! multi-proof "any-path" model are milestone-2 seams; M1 validates the
//! single-delegator-per-link chain (each UCAN's `proofs[i]` is *a* delegator that
//! must independently cover the claims).

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
    /// A link is not live at the validation instant `now`: it has expired
    /// (`now > expiration`) or is not yet valid (`now < not_before`). Absolute
    /// time check — independent of relative window containment. Fail-closed: an
    /// expired chain MUST deny.
    Expired {
        /// 0-based link index (0 = leaf).
        link: usize,
    },
    /// The proof chain is nested deeper than [`MAX_PROOF_DEPTH`]. A DoS guard
    /// against stack exhaustion from a maliciously deep chain — not an
    /// authority-widening concern, but rejected fail-closed all the same.
    DepthExceeded {
        /// The depth at which the bound was exceeded.
        depth: usize,
    },
    /// A non-root UCAN carried no proofs (cannot be authorized) — only a root
    /// may have an empty proof set, and a root is the *top* of the chain.
    MissingProof,
}

/// Maximum delegation-chain (nested-proof) depth this validator will walk. A
/// chain deeper than this is rejected fail-closed ([`ChainError::DepthExceeded`])
/// before recursion can exhaust the stack. UCAN delegation chains are short by
/// construction (a handful of hops); this bound is generous for legitimate use
/// while capping a hostile `proofs`-within-`proofs` nesting (a DoS vector at both
/// parse time and walk time). The same bound is enforced at CBOR decode in
/// [`super::token::Ucan::from_cbor`] so an over-deep token is refused before it
/// is ever walked.
pub const MAX_PROOF_DEPTH: usize = 32;

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
            ChainError::Expired { link } => {
                write!(
                    f,
                    "chain link {link} is not live at the validation instant (expired or not yet valid)"
                )
            }
            ChainError::DepthExceeded { depth } => {
                write!(
                    f,
                    "delegation chain exceeds maximum proof depth {MAX_PROOF_DEPTH} (at depth {depth})"
                )
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

/// Recursively validate the delegation chain rooted at `ucan` (structural +
/// temporal — run signatures separately, or use [`validate`]).
///
/// Checks, on EVERY node (the leaf, every intermediate, and the root):
/// - **absolute validity at `now`** via [`Ucan::is_valid_at`] — the link must be
///   live (not expired, not future-dated) at the single caller-supplied instant;
/// - for each `proof` in `ucan.proofs`, the edge `ucan ⟶ proof` (linkage +
///   attenuation + relative window containment), then recurses into the proof.
///
/// A UCAN with no proofs is a **root** and authorizes whatever it self-asserts
/// (its capabilities become the ceiling) — but it is STILL clock-checked.
///
/// `now` is a **single trusted clock value supplied by the caller**, threaded
/// down UNCHANGED to every link. It is never taken from a token's own fields, so
/// a forged `not_before`/`expiration` cannot self-certify validity — it can only
/// narrow (the relative-window check) or fail the absolute check against the one
/// authoritative `now`.
///
/// `link` is the depth from the original leaf, used both for diagnostics and as
/// the [`MAX_PROOF_DEPTH`] DoS bound.
pub fn validate_chain(ucan: &Ucan, now: u64) -> Result<(), ChainError> {
    validate_chain_at(ucan, now, 0)
}

fn validate_chain_at(ucan: &Ucan, now: u64, link: usize) -> Result<(), ChainError> {
    // Depth bound FIRST — cap recursion before doing any work at this level so a
    // hostile deep chain cannot exhaust the stack (DoS guard, fail-closed).
    if link >= MAX_PROOF_DEPTH {
        return Err(ChainError::DepthExceeded { depth: link });
    }
    // Absolute time on THIS node (covers leaf, intermediates, and the root). The
    // same `now` is used at every level — never derived from the token.
    if !ucan.is_valid_at(now) {
        return Err(ChainError::Expired { link });
    }
    if ucan.proofs.is_empty() {
        // Root: self-asserted authority is the top of the ceiling. Nothing above
        // to attenuate against. (Whether a given root DID is *trusted* to assert
        // is the verifier/trust-anchor concern, handled at signature time and by
        // the approval binding — not structural attenuation.) Still clock-checked
        // above.
        return Ok(());
    }
    for proof in &ucan.proofs {
        validate_edge(ucan, proof, link)?;
        validate_chain_at(proof, now, link + 1)?;
    }
    Ok(())
}

/// The full fail-closed validation a compiler front-end runs before lowering a
/// UCAN into a ceiling: **(1) structure, (2) signatures, (3) delegation chain +
/// attenuation + temporal validity at `now`**, in that order. Returns `Ok(())`
/// only if every link is structurally sound, correctly signed under the hybrid
/// policy, properly linked, strictly attenuating, within its delegator's window,
/// AND live at `now` (not expired, not future-dated), with the chain no deeper
/// than [`MAX_PROOF_DEPTH`].
///
/// `now` is the single trusted current time (unix seconds) the caller supplies;
/// it is threaded unchanged to every link. The caller is responsible for sourcing
/// a trustworthy clock — the validator never reads "now" from the token.
///
/// This is the single entry point S5's compiler calls; it guarantees that only a
/// genuine, currently-live ceiling reaches the (deferred) bundle-emission stage.
pub fn validate<V: UcanVerifier>(ucan: &Ucan, verifier: &V, now: u64) -> Result<(), ChainError> {
    ucan.validate_structure().map_err(ChainError::Structure)?;
    ucan.verify_signatures(verifier)
        .map_err(ChainError::Signature)?;
    validate_chain(ucan, now)
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

    /// A wall-clock instant that is within the validity window of every UCAN the
    /// `signed_ucan` helper mints (`not_before: None`, `expiration:
    /// 9_999_999_999`) and within the explicit windows used by the temporal
    /// tests. Passed as the trusted `now` to `validate`/`validate_chain`.
    const NOW: u64 = 1_000;

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
            validate(&bob_ucan, &s, NOW).is_ok(),
            "a strictly-attenuating chain must validate"
        );
    }

    #[test]
    fn root_with_no_proofs_is_self_authorizing() {
        let root = TestIdentity::generate();
        let u = signed_ucan(&root, &root.did(), vec![cap("mac://anything", "*")], vec![]);
        let s = store(&[&root]);
        assert!(validate(&u, &s, NOW).is_ok());
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
        match validate(&alice_ucan, &s, NOW) {
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
            validate(&alice_ucan, &s, NOW),
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
            validate(&bob_ucan, &s, NOW),
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
            validate(&alice_ucan, &s, NOW),
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
            validate(&bob_ucan, &s, NOW),
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
            validate(&alice_ucan, &s, NOW),
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
        assert!(validate(&alice_ok, &store(&[&root, &alice]), NOW).is_ok());
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
            validate(&alice_ucan, &s, NOW),
            Err(ChainError::Signature(_))
        ));
    }

    // ---- Absolute time at `now` (the fail-open fix, #590) ----------------

    /// Mint a self-issued (root) UCAN with an explicit validity window, properly
    /// hybrid-signed, plus its trust store.
    fn windowed_root(nbf: Option<u64>, exp: Option<u64>) -> (Ucan, TestTrustStore) {
        let id = TestIdentity::generate();
        let payload = UcanPayload {
            issuer: id.did(),
            audience: id.did(),
            capabilities: vec![cap("mac://model/*", "infer")],
            not_before: nbf,
            expiration: exp,
            nonce: vec![],
        };
        let u = resign(&id, payload, vec![]);
        (u, store(&[&id]))
    }

    #[test]
    fn expired_chain_denied() {
        // A perfectly signed, linked, self-asserting chain that EXPIRED at 500.
        // Relative window containment is satisfied (root has no parent), so before
        // the absolute check this validated — the fail-open. Now: deny at now=1000.
        let (u, s) = windowed_root(None, Some(500));
        assert!(matches!(
            validate(&u, &s, 1000),
            Err(ChainError::Expired { link: 0 })
        ));
    }

    #[test]
    fn future_nbf_denied() {
        // Not-yet-valid: not_before is in the future relative to `now`.
        let (u, s) = windowed_root(Some(5000), Some(9000));
        assert!(matches!(
            validate(&u, &s, 1000),
            Err(ChainError::Expired { link: 0 })
        ));
    }

    #[test]
    fn valid_at_now_accepted() {
        // Live window straddling `now` validates.
        let (u, s) = windowed_root(Some(500), Some(2000));
        assert!(validate(&u, &s, 1000).is_ok());
        // Boundary instants are inclusive (now == nbf and now == exp).
        assert!(validate(&u, &s, 500).is_ok());
        assert!(validate(&u, &s, 2000).is_ok());
    }

    #[test]
    fn expiry_checked_across_a_multi_link_chain() {
        // A 2-link root → leaf chain that is fully within-window (containment OK).
        // It validates while live and is denied once expired — exercising the
        // per-link absolute check through the recursion (not just a 1-node chain).
        //
        // Note on why "leaf live but a deeper proof expired" is NOT testable here:
        // window containment forces leaf.window ⊆ proof.window, so the leaf can
        // never outlive its proof. The per-link check on deeper nodes is therefore
        // exercised by (a) `expired_chain_denied` — a root (the *bottom* link) is
        // clock-checked — and (b) this multi-link chain denying when expired.
        let root = TestIdentity::generate();
        let leaf = TestIdentity::generate();
        let root_payload = UcanPayload {
            issuer: root.did(),
            audience: leaf.did(),
            capabilities: vec![cap("mac://*", "*")],
            not_before: Some(100),
            expiration: Some(2000),
            nonce: vec![],
        };
        let root_ucan = resign(&root, root_payload, vec![]);
        let leaf_payload = UcanPayload {
            issuer: leaf.did(),
            audience: leaf.did(),
            capabilities: vec![cap("mac://model/x", "infer")],
            not_before: Some(100),
            expiration: Some(2000),
            nonce: vec![],
        };
        let leaf_ucan = resign(&leaf, leaf_payload, vec![root_ucan]);
        let s = store(&[&root, &leaf]);
        // Live: validates.
        assert!(validate(&leaf_ucan, &s, 1000).is_ok());
        // Expired: denied on the clock.
        assert!(matches!(
            validate(&leaf_ucan, &s, 3000),
            Err(ChainError::Expired { .. })
        ));
        // Not yet valid: denied.
        assert!(matches!(
            validate(&leaf_ucan, &s, 50),
            Err(ChainError::Expired { .. })
        ));
    }

    // ---- Proof-depth DoS bound ------------------------------------------

    /// Build a linear self-delegating chain of `n` links (leaf has `n-1` nested
    /// proofs), all hybrid-signed and live at `NOW`. Returns (leaf, store).
    fn linear_chain(n: usize) -> (Ucan, TestTrustStore) {
        let id = TestIdentity::generate();
        let mk = |proofs: Vec<Ucan>| {
            let payload = UcanPayload {
                issuer: id.did(),
                audience: id.did(),
                capabilities: vec![cap("mac://model/*", "infer")],
                not_before: None,
                expiration: Some(9_999_999_999),
                nonce: vec![],
            };
            resign(&id, payload, proofs)
        };
        let mut u = mk(vec![]);
        for _ in 1..n {
            u = mk(vec![u]);
        }
        (u, store(&[&id]))
    }

    #[test]
    fn max_depth_chain_accepted_one_over_rejected() {
        // A chain exactly at the bound walks fine.
        let (ok, s) = linear_chain(MAX_PROOF_DEPTH);
        assert!(validate(&ok, &s, NOW).is_ok());
        // One link deeper than the bound is rejected fail-closed.
        let (deep, s2) = linear_chain(MAX_PROOF_DEPTH + 1);
        assert!(matches!(
            validate(&deep, &s2, NOW),
            Err(ChainError::DepthExceeded { .. })
        ));
    }

    #[test]
    fn over_deep_chain_rejected_at_cbor_decode() {
        // The DoS bound is also enforced at parse time: an over-deep token is
        // refused by `from_cbor` before it can ever be walked.
        let (deep, _) = linear_chain(MAX_PROOF_DEPTH + 1);
        let bytes = deep.to_cbor().unwrap();
        assert!(matches!(
            Ucan::from_cbor(&bytes),
            Err(crate::auth::ucan::token::UcanError::Malformed(_))
        ));
        // A chain within the bound round-trips through from_cbor fine.
        let (ok, _) = linear_chain(MAX_PROOF_DEPTH);
        let ok_bytes = ok.to_cbor().unwrap();
        assert!(Ucan::from_cbor(&ok_bytes).is_ok());
    }
}
