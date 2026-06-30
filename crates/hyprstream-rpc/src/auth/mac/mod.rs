//! Native MAC — security labels, the lattice, subject contexts, and genesis
//! labeling. **S1 (#567), the foundation of epic #547.**
//!
//! This module is the **TCB seam** the rest of the security epic consumes. It
//! defines *what a label is*, *how labels compare* (dominance ⊒ and IFC join ⊔),
//! *where a subject's clearance comes from*, *where an object's label comes
//! from*, and *how everything gets an initial label* (genesis). It deliberately
//! contains no enforcement, no policy distribution, and no token logic — those
//! are S2/S4/S6.
//!
//! ## The lattice (small + verifiable, by design)
//!
//! A label is a point in the product lattice
//! ```text
//!     Level  ×  Assurance  ×  P(Compartment)
//!     (MLS)     (#548)         (need-to-know)
//! ```
//! - **Level** — total order `public < internal < confidential < secret`.
//! - **Assurance** — total order `unverified < classical < pq-hybrid`, *derived
//!   from verified key material* (#548), never claimed.
//! - **Compartments** — powerset lattice under ⊆ (a closed vocabulary, policy).
//!
//! Dominance is the product order (≥ on each total axis, ⊇ on the compartment
//! set); join is component-wise (max / max / ∪). Both are pure functions on
//! [`SecurityLabel`] with **no policy argument** — content truth, uncircumventable.
//!
//! ## Interface contract for S2 (enforcement) and S4 (PDP)
//!
//! S1 hands the downstream tickets exactly these stable entry-points:
//!
//! - **S2 — reference-monitor per-op check (#568).** For each op the monitor:
//!   1. obtains the **object label** via [`manifest::LabeledObject::security_label`]
//!      — `None` ⇒ DENY (unlabeled object);
//!   2. obtains the **subject context** via
//!      [`context::SubjectContextClaims::security_context`] (clearance from
//!      verified claims, assurance derived from verified key material) — `None`
//!      ⇒ DENY (unlabeled subject);
//!   3. evaluates the MAC floor [`context::SecurityContext::can_access`]
//!      (≡ [`SecurityLabel::can_access`]) — `false` ⇒ DENY.
//!   The monitor takes object labels ONLY from `LabeledObject`, never from a
//!   token/UCAN/caveat (design §3, §14). It needs no `Lattice` on the hot path:
//!   dominance is intrinsic.
//!
//! - **S2 — IFC at seal/rollup (#568).** A derived object's label =
//!   [`SecurityLabel::join_all`] over its provenance inputs' labels. Aggregation
//!   cannot launder classification.
//!
//! - **S4 — PDP / lattice engine (#570).** The PDP owns the [`lattice::Lattice`]
//!   policy object: it [`validate`](lattice::Lattice::validate)s labels at
//!   enrollment/seal, exposes the pinned [`lattice::LatticeVersion`] for the
//!   `(UCAN, bundle_hash)` approval binding (S5), and computes the
//!   **label-ceiling clamp** (§5a) as `min`/dominance over the lattice. The PDP
//!   never re-implements the order — it reuses the intrinsic [`SecurityLabel`]
//!   algebra, so the PDP's clamp and the monitor's floor are provably the same
//!   relation.
//!
//! - **Both — derived value:** `effective = approved-ceiling ∩ MAC clearance ∩
//!   self-attenuation` (design §5). The "∩ MAC clearance" factor is exactly the
//!   dominance/clamp defined here; S5/S6 compose it, they do not redefine it.
//!
//! ## BLOCKERS (what S1 cannot finish, and why)
//!
//! S1 ships everything that is genuinely unblocked (the label + lattice + ops +
//! subject context + genesis mechanism + static-node labels, all tested). The
//! following are **defined as seams** here but gated on other work:
//!
//! 1. **Content-bound manifest labels** ← the data-plane CAS manifest store.
//!    The content-addressed manifest with a `security_label` field *does not
//!    exist yet* (today's "manifests" are OCI image / release / context
//!    manifests, none data-plane CAS). S1 defines the
//!    [`manifest::ContentBoundLabel`] contract + [`manifest::LabeledObject`]
//!    seam; the implementation (label-in-CID binding + `verify_binding`) lands
//!    with the CAS manifest layer. **Static-node labels are NOT blocked** and
//!    are implemented here.
//!
//! 2. **Subject clearance wire field on `Claims`/envelope** ← S8 (#574, hybrid
//!    envelope). S1 defines the [`context::SubjectContextClaims`] trait that the
//!    verified claims will implement; the concrete `clearance` field is added to
//!    `Claims` / `common.capnp` under S8 so the new authz field ships on the
//!    hybrid-signed (EdDSA + ML-DSA-65) envelope, not the classical one. Until
//!    then S2/S4 program against the trait and test with a stub.
//!
//! 3. **Assurance derivation source** ← #441 / #280 (authoritative key binding)
//!    + `register_pq_trust`. [`context::VerifiedKeyMaterial`] is the seam; the
//!    *production* of it (was the ML-DSA-65 anchor actually bound + verified for
//!    this identity?) is the envelope-verification + key-binding layer's job.
//!    Until a binding is authoritative, an identity is `Classical` at best and
//!    `Unverified` if even the classical signature can't be checked — both
//!    fail-closed.
//!
//! 4. **Compartment vocabulary / genesis content** ← S3 (#569, `$scope`/TE
//!    annotations). S1 defines [`lattice::Lattice`] (the closed vocabulary
//!    holder) and [`genesis::GenesisMap`] (the path→label mechanism + the
//!    completeness gate); the *actual* compartment names and the per-node label
//!    table come from the schema annotations + site policy that S3 produces.
//!
//! 5. **PQ-confidentiality labels are unsatisfiable today** ← #153 (streaming
//!    DH → ML-KEM-768 hybrid KEM). The assurance axis covers *signature*
//!    assurance, which is shippable. A future *confidentiality* requirement
//!    (data that must travel a PQ-KEM path) has no satisfying transport until
//!    #153 wires `pq_kem_ciphertext`; such a label is simply undominated →
//!    fail-closed, which is correct. No type change needed when #153 lands —
//!    it's a new compartment/level value, not a new axis.

pub mod context;
pub mod genesis;
pub mod label;
pub mod lattice;
pub mod manifest;

pub use context::{SecurityContext, SubjectContextClaims, VerifiedKeyMaterial};
pub use genesis::{GenesisMap, GenesisReport};
pub use label::{
    Assurance, Compartment, CompartmentSet, Level, SecurityLabel, MAX_COMPARTMENTS,
};
pub use lattice::{LabelError, Lattice, LatticeCodecError, LatticeDecodeError, LatticeVersion};
pub use manifest::{ContentBoundLabel, LabeledObject, StaticNodeLabel};

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod integration_tests {
    //! End-to-end shape of the per-op check S2 will perform, exercised against
    //! the S1 seams only (object via `LabeledObject`, subject via
    //! `SubjectContextClaims`, floor via `can_access`).
    use super::*;

    struct StubClaims(Option<SecurityLabel>);
    impl SubjectContextClaims for StubClaims {
        fn clearance_label(&self) -> Option<SecurityLabel> {
            self.0
        }
    }

    /// The exact ALLOW predicate the monitor (S2) computes for the MAC floor,
    /// using only S1 seams. Returns true ⇒ floor permits; both unlabeled cases
    /// return false (deny).
    fn mac_floor(
        object: &dyn LabeledObject,
        claims: &dyn SubjectContextClaims,
        key_material: VerifiedKeyMaterial,
    ) -> bool {
        let Some(object_label) = object.security_label() else {
            return false; // unlabeled object → deny
        };
        let Some(ctx) = claims.security_context(key_material) else {
            return false; // unlabeled subject → deny
        };
        ctx.can_access(&object_label)
    }

    /// A compartment bitset from bit indices (bit 0 = "pii", 1 = "finance", … in
    /// these tests). The policy interns names→bits; here we work in bits.
    fn comps(bits: &[u32]) -> CompartmentSet {
        bits.iter().copied().collect()
    }

    #[test]
    fn allow_when_cleared() {
        let object = StaticNodeLabel::labeled(SecurityLabel::new(
            Level::Confidential,
            Assurance::Classical,
            comps(&[0]), // pii
        ));
        let claims = StubClaims(Some(SecurityLabel::new(
            Level::Secret,
            Assurance::PqHybrid,
            comps(&[0, 1]), // pii, finance
        )));
        assert!(mac_floor(&object, &claims, VerifiedKeyMaterial::PqHybrid));
    }

    #[test]
    fn deny_unlabeled_object() {
        let object = StaticNodeLabel::unlabeled();
        let claims = StubClaims(Some(Lattice::new(LatticeVersion(1), []).top()));
        assert!(!mac_floor(&object, &claims, VerifiedKeyMaterial::PqHybrid));
    }

    #[test]
    fn deny_unlabeled_subject() {
        let object = StaticNodeLabel::labeled(SecurityLabel::new(
            Level::Public,
            Assurance::Classical,
            CompartmentSet::EMPTY,
        ));
        let claims = StubClaims(None);
        assert!(!mac_floor(&object, &claims, VerifiedKeyMaterial::PqHybrid));
    }

    #[test]
    fn deny_classical_subject_on_pqc_object_no_special_path() {
        // #548 end-to-end: the assurance axis denies via the SAME dominance
        // check — no separate code path.
        let object = StaticNodeLabel::labeled(SecurityLabel::new(
            Level::Public,
            Assurance::PqHybrid,
            CompartmentSet::EMPTY,
        ));
        let claims = StubClaims(Some(SecurityLabel::new(
            Level::Secret,
            Assurance::PqHybrid, // policy assigned high, but...
            CompartmentSet::EMPTY,
        )));
        // ...the verified key material is only Classical → clamped → denied.
        assert!(!mac_floor(&object, &claims, VerifiedKeyMaterial::Classical));
        // a PQC-bound key passes.
        assert!(mac_floor(&object, &claims, VerifiedKeyMaterial::PqHybrid));
    }

    #[test]
    fn ifc_join_propagates_to_derived_object() {
        // A derived object sealed from a secret + a classical input is
        // secret/pq-hybrid — and a merely-classical subject can't read it.
        let secret_in = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, comps(&[0]));
        let public_in = SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY);
        let derived = SecurityLabel::join_all([&secret_in, &public_in]);
        let object = StaticNodeLabel::labeled(derived);

        let low = StubClaims(Some(SecurityLabel::new(
            Level::Internal,
            Assurance::Classical,
            CompartmentSet::EMPTY,
        )));
        assert!(!mac_floor(&object, &low, VerifiedKeyMaterial::Classical));
    }
}
