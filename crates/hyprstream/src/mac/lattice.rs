//! Lattice surface for the PDP — **re-export of S1's canonical types** (#567).
//!
//! S4 (#570) originally developed the TE evaluator + AVC against a *stub* lattice
//! (`StubLinearLattice` + a local `SecurityLabel(u32)` + a `Lattice` dominance
//! trait) so the two streams could proceed in parallel. **S1 has landed**
//! (`feature/security-mac-ucan` @ `406ff9fdc`) with the real, canonical types in
//! [`hyprstream_rpc::auth::mac`], and this file is now a thin re-export so the rest
//! of the PDP (`te.rs`, `avc.rs`, `compiled.rs`) consumes S1 directly. The stub is
//! deleted.
//!
//! ## What changed at reconciliation (the S1 contract, as landed)
//!
//! - [`SecurityLabel`] is no longer an opaque `u32` interned id. It is S1's
//!   **structured value** — a point in the product lattice
//!   `Level × Assurance × CompartmentSet` — and is `Copy + Ord + Hash`
//!   (compartments are a packed `u64` bitset), so it still keys the AVC directly
//!   with no interning indirection. Crucially it has **no `Default`**: an
//!   unlabeled subject/object is `Option::None` ⇒ deny, never a permissive
//!   default.
//! - Dominance (`⊒`) and join (`⊔`) are **intrinsic** to the value
//!   ([`SecurityLabel::can_access`], [`SecurityLabel::join`] /
//!   [`SecurityLabel::join_all`]), and re-exposed on
//!   [`SecurityContext::can_access`] for a verified subject. They are NOT methods
//!   on a `Lattice` trait — content truth, no policy argument. The old
//!   `Lattice::can_access`/`join` trait is gone; the PDP floor now calls the
//!   intrinsic order (see `te.rs`).
//! - [`Lattice`] is now S1's **policy object** (the closed, versioned compartment
//!   name↔bit vocabulary). The PDP holds it to (a) align its TE matrix columns to
//!   the exact bits a label carries via [`Lattice::compartment_names`], (b)
//!   [`validate`](Lattice::validate) labels at enrollment/seal, and (c) embed it,
//!   via [`Lattice::to_bytes`], inside the signed `CompiledPolicy` so every
//!   process reconstructs the **identical** bit map ([`compiled`](crate::mac::compiled)).
//!   [`LatticeVersion`] is bound to `CompiledPolicy.generation`.
//!
//! The MAC floor (per-op `subject.ctx ⊒ object.label` + IFC join) is computed
//! against the intrinsic order and is **independent of any grant/token/UCAN**
//! (design §3, §10).

// Re-export S1's canonical MAC types from the RPC crate (the TCB seam). Everything
// the PDP needs flows through here so a future S1 shape change is a one-file edit.
pub use hyprstream_rpc::auth::mac::{
    Assurance, Compartment, CompartmentSet, LabelError, Lattice, LatticeCodecError,
    LatticeDecodeError, LatticeVersion, Level, SecurityContext, SecurityLabel,
    SubjectContextClaims, VerifiedKeyMaterial, MAX_COMPARTMENTS,
};

/// IFC convenience: join a slice of input labels into the derived label (design
/// §3 — sealed at rollup). Thin forwarder over S1's intrinsic
/// [`SecurityLabel::join_all`]. Non-empty joins use the per-axis algebraic identity
/// `(Public, PqHybrid, {})`; empty input instead yields the fail-closed
/// `(Public, Unverified, {})`, which the seal/rollup policy (S2) must treat as
/// suspicious. This is the function S6/seal-time labeling calls; it lives here so
/// PDP callers have a single entry-point without reaching across crates.
#[inline]
#[must_use]
pub fn ifc_join(inputs: &[SecurityLabel]) -> SecurityLabel {
    SecurityLabel::join_all(inputs.iter())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn lattice() -> Lattice {
        Lattice::new(
            LatticeVersion(1),
            [
                Compartment::new("pii"),
                Compartment::new("finance"),
                Compartment::new("tenant:acme"),
            ],
        )
    }

    #[test]
    fn intrinsic_dominance_is_product_order() {
        let l = lattice();
        let cleared = l
            .label(
                Level::Secret,
                Assurance::PqHybrid,
                [Compartment::new("pii"), Compartment::new("finance")],
            )
            .unwrap();
        let needs_pii = l
            .label(
                Level::Confidential,
                Assurance::Classical,
                [Compartment::new("pii")],
            )
            .unwrap();
        // dominates on every axis (level ≥, assurance ≥, compartments ⊇).
        assert!(cleared.can_access(&needs_pii));
        assert!(!needs_pii.can_access(&cleared));
    }

    #[test]
    fn ifc_join_combines_blp_confidentiality_and_biba_integrity() {
        let l = lattice();
        let secret = l
            .label(
                Level::Secret,
                Assurance::PqHybrid,
                [Compartment::new("pii")],
            )
            .unwrap();
        let public = l.label(Level::Public, Assurance::Classical, []).unwrap();
        let derived = ifc_join(&[secret, public]);

        // Confidentiality and compartments take their upper bounds.
        assert!(derived.level >= secret.level && derived.level >= public.level);
        assert!(secret.compartments.is_subset(derived.compartments));
        assert!(public.compartments.is_subset(derived.compartments));
        // Biba provenance integrity takes the floor (weakest input).
        assert!(derived.assurance <= secret.assurance);
        assert!(derived.assurance <= public.assurance);
        assert_eq!(derived.assurance, Assurance::Classical);

        // The algebraic identity is (Public, PqHybrid, {}), but an empty input
        // is deliberately fail-closed rather than returning that identity.
        let algebraic_identity =
            SecurityLabel::new(Level::Public, Assurance::PqHybrid, CompartmentSet::EMPTY);
        assert_eq!(ifc_join(&[algebraic_identity, secret]), secret);
        let empty = ifc_join(&[]);
        assert_eq!(empty.level, Level::Public);
        assert_eq!(empty.assurance, Assurance::Unverified);
        assert!(empty.compartments.is_empty());
        assert_ne!(empty, algebraic_identity);
    }

    #[test]
    fn lattice_validate_and_bytes_roundtrip() {
        // The desync-proof property compiled.rs relies on: same bytes → identical
        // bit assignment in every process.
        let l = lattice();
        let restored = Lattice::from_bytes(&l.to_bytes()).unwrap();
        assert_eq!(restored.compartment_names(), l.compartment_names());
        let label = l
            .label(
                Level::Internal,
                Assurance::Classical,
                [Compartment::new("tenant:acme")],
            )
            .unwrap();
        assert!(restored.validate(&label).is_ok());
        assert_eq!(restored.bit_of(&Compartment::new("tenant:acme")), Some(2));
    }
}
