//! The **subject security context** — a subject's verified clearance, the
//! `subject.ctx` side of the per-op dominance check `subject.ctx ⊒ object.label`.
//!
//! A subject context is the [`SecurityLabel`] a verified principal is cleared
//! up to. Where the *object* label is content-bound (in a manifest), the
//! *subject* context rides the verified claims/envelope (design §3, §11 data
//! structures: `subject ctx in the verified claims/envelope`).
//!
//! ## Where it lives on the wire
//!
//! The subject context is **derived**, not self-asserted. Two inputs feed it,
//! both already verified by the time the monitor runs:
//! 1. The principal's **clearance** (level + compartments) — assigned by policy
//!    / enrollment and carried in the verified [`crate::auth::Claims`] (the JWT
//!    is signed by the issuing node, so the clearance is authority-asserted,
//!    not subject-asserted). The wire field is added by the
//!    [`SubjectContextClaims`] seam below; the actual `Claims`/`common.capnp`
//!    field add is gated on S8/#574 (hybrid envelope) — see BLOCKERS.
//! 2. The principal's **assurance** (#548) — derived from the *verified key
//!    material*, never claimed. See [`SecurityContext::derive_assurance`].
//!
//! ## Fail-closed
//!
//! A subject with no derivable clearance has **no context** (`None`), and the
//! monitor denies — there is no `Default` clearance. Assurance defaults to
//! [`Assurance::Unverified`] (the floor) precisely so an un-attested key
//! dominates nothing above the floor.

use super::label::{Assurance, CompartmentSet, Level, SecurityLabel};
use serde::{Deserialize, Serialize};

/// How a principal's key material was verified, used to *derive* (never trust) an
/// [`Assurance`]. Produced by the envelope/claims verification layer — this type
/// is the seam between "what crypto verified" and "what the lattice sees".
///
/// This intentionally mirrors the existing crypto reality (`crypto/pq.rs`,
/// `register_pq_trust`, [`crate::crypto::CryptoPolicy`]): an Ed25519 identity is
/// `Classical` until a bound ML-DSA-65 anchor lifts it to `PqHybrid`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifiedKeyMaterial {
    /// Identity / signature could not be verified at all → floor.
    Unverified,
    /// A classical signature verified (Ed25519 / p256 / secp256k1), and no
    /// bound post-quantum anchor is present. The federation edge.
    Classical,
    /// A hybrid composite verified: an Ed25519 identity with a bound ML-DSA-65
    /// verifying key (the `register_pq_trust` binding) under
    /// [`crate::crypto::CryptoPolicy::Hybrid`].
    PqHybrid,
}

impl VerifiedKeyMaterial {
    /// Map verified key material → the assurance lattice axis (#548). This is
    /// the *only* place assurance is assigned to a subject; it is a function of
    /// verified crypto, so it cannot be spoofed by a claim.
    pub fn assurance(self) -> Assurance {
        match self {
            VerifiedKeyMaterial::Unverified => Assurance::Unverified,
            VerifiedKeyMaterial::Classical => Assurance::Classical,
            VerifiedKeyMaterial::PqHybrid => Assurance::PqHybrid,
        }
    }
}

/// A verified subject's security context = the clearance label the monitor
/// compares against object labels.
///
/// Construction is deliberately gated: you build it from a *clearance* (level +
/// compartments, authority-asserted) and *verified key material* (assurance,
/// crypto-derived). The assurance axis of the resulting label is ALWAYS the
/// derived one — a caller cannot pass in a higher assurance than the crypto
/// supports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityContext {
    /// The full clearance label (level + assurance + compartments). The
    /// assurance component is crypto-derived; see [`SecurityContext::new`].
    clearance: SecurityLabel,
}

impl SecurityContext {
    /// Build a subject context from an authority-asserted clearance
    /// (`level` + `compartments`) and crypto-derived `key_material`.
    ///
    /// The resulting clearance's assurance axis is **clamped to** the derived
    /// assurance — a classical key can never carry a `PqHybrid` clearance even
    /// if policy mistakenly assigned one. This is the load-bearing invariant of
    /// #548: assurance is a property of the verified identity, not a grant.
    pub fn new(
        level: Level,
        compartments: CompartmentSet,
        key_material: VerifiedKeyMaterial,
    ) -> Self {
        SecurityContext {
            clearance: SecurityLabel::new(level, key_material.assurance(), compartments),
        }
    }

    /// Build from a pre-assembled clearance label and verified key material,
    /// clamping the assurance axis down to what the crypto supports.
    ///
    /// `min(clearance.assurance, derived)` — clamp DOWN only, never up. If
    /// policy assigned `PqHybrid` but the key is `Classical`, the effective
    /// assurance is `Classical` (fail-closed/least-privilege).
    pub fn from_clearance(clearance: SecurityLabel, key_material: VerifiedKeyMaterial) -> Self {
        let derived = key_material.assurance();
        SecurityContext {
            clearance: SecurityLabel {
                assurance: clearance.assurance.min(derived),
                ..clearance
            },
        }
    }

    /// Derive assurance directly from verified key material (#548).
    /// Thin forwarder kept as a named entry-point so the enforcement/PDP layers
    /// (S2/S4) have one obvious call site.
    pub fn derive_assurance(key_material: VerifiedKeyMaterial) -> Assurance {
        key_material.assurance()
    }

    /// The clearance label. This is the `subject.ctx` the monitor feeds into
    /// `can_access(object_label)`.
    pub fn clearance(&self) -> &SecurityLabel {
        &self.clearance
    }

    /// `self ⊒ object` — does this subject context dominate the object's
    /// content-bound label? The per-op MAC floor (design §10). Pure forward to
    /// the intrinsic lattice order.
    #[must_use]
    pub fn can_access(&self, object_label: &SecurityLabel) -> bool {
        self.clearance.can_access(object_label)
    }

    pub fn level(&self) -> Level {
        self.clearance.level
    }
    pub fn assurance(&self) -> Assurance {
        self.clearance.assurance
    }
    pub fn compartments(&self) -> CompartmentSet {
        self.clearance.compartments
    }
}

/// **Seam** for carrying a subject clearance on the verified claims/envelope.
///
/// S1 defines the *shape* of the clearance the claims must carry; the concrete
/// wire field on [`crate::auth::Claims`] / `common.capnp` is added under S8
/// (#574, hybrid envelope) so the new authz field ships on the hybrid-signed
/// envelope rather than the classical one. Until then this trait is the typed
/// contract S2/S4 program against, and a node assembles a [`SecurityContext`]
/// by:
///   1. reading the (future) `clearance` field off verified `Claims`, and
///   2. deriving assurance from the envelope's verified key material.
///
/// Implementing this on `Claims` is a one-liner once the field lands; defining
/// it now lets S2/S4 compile and test against a stable interface.
pub trait SubjectContextClaims {
    /// The authority-asserted clearance carried in the verified claims, if any.
    /// `None` ⇒ unlabeled subject ⇒ the monitor MUST deny (no default clearance).
    fn clearance_label(&self) -> Option<SecurityLabel>;

    /// Assemble the full subject context, clamping assurance to the verified
    /// key material. Returns `None` (→ deny) if the subject carries no
    /// clearance. Default impl composes the two inputs; implementors normally
    /// only need to provide [`clearance_label`].
    ///
    /// [`clearance_label`]: SubjectContextClaims::clearance_label
    fn security_context(&self, key_material: VerifiedKeyMaterial) -> Option<SecurityContext> {
        self.clearance_label()
            .map(|c| SecurityContext::from_clearance(c, key_material))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A compartment bitset from bit indices (the policy interns names → bits;
    /// these tests work directly in bits).
    fn comps(bits: &[u32]) -> CompartmentSet {
        bits.iter().copied().collect()
    }

    #[test]
    fn assurance_derived_from_key_material() {
        assert_eq!(
            VerifiedKeyMaterial::Unverified.assurance(),
            Assurance::Unverified
        );
        assert_eq!(
            VerifiedKeyMaterial::Classical.assurance(),
            Assurance::Classical
        );
        assert_eq!(
            VerifiedKeyMaterial::PqHybrid.assurance(),
            Assurance::PqHybrid
        );
    }

    #[test]
    fn new_clamps_assurance_to_key_material() {
        // Even at Secret level, a classical key yields a classical-assurance ctx.
        let ctx = SecurityContext::new(Level::Secret, comps(&[0]), VerifiedKeyMaterial::Classical);
        assert_eq!(ctx.assurance(), Assurance::Classical);
        assert_eq!(ctx.level(), Level::Secret);
    }

    #[test]
    fn from_clearance_clamps_down_never_up() {
        // policy mistakenly assigned PqHybrid, but key is only Classical.
        let claimed = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, comps(&[0]));
        let ctx = SecurityContext::from_clearance(claimed, VerifiedKeyMaterial::Classical);
        assert_eq!(ctx.assurance(), Assurance::Classical); // clamped down
    }

    #[test]
    fn from_clearance_does_not_raise_assurance() {
        // policy assigned Classical, key is PqHybrid → stays Classical (no raise).
        let claimed = SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let ctx = SecurityContext::from_clearance(claimed, VerifiedKeyMaterial::PqHybrid);
        assert_eq!(ctx.assurance(), Assurance::Classical);
    }

    #[test]
    fn classical_context_cannot_dominate_pqc_object() {
        let ctx = SecurityContext::new(Level::Secret, comps(&[0]), VerifiedKeyMaterial::Classical);
        let pqc_object = SecurityLabel::new(Level::Public, Assurance::PqHybrid, CompartmentSet::EMPTY);
        assert!(!ctx.can_access(&pqc_object));
    }

    struct FakeClaims(Option<SecurityLabel>);
    impl SubjectContextClaims for FakeClaims {
        fn clearance_label(&self) -> Option<SecurityLabel> {
            self.0
        }
    }

    #[test]
    fn unlabeled_subject_has_no_context() {
        let claims = FakeClaims(None);
        assert!(claims
            .security_context(VerifiedKeyMaterial::PqHybrid)
            .is_none());
    }

    #[test]
    fn labeled_subject_assembles_context_with_clamped_assurance() {
        let claims = FakeClaims(Some(SecurityLabel::new(
            Level::Confidential,
            Assurance::PqHybrid,
            comps(&[0]),
        )));
        let ctx = claims
            .security_context(VerifiedKeyMaterial::Classical)
            .unwrap();
        assert_eq!(ctx.level(), Level::Confidential);
        assert_eq!(ctx.assurance(), Assurance::Classical); // clamped to verified key
    }
}
