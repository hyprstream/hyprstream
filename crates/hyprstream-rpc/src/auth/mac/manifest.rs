//! The **object-label seam** — where an object's content-bound label comes from.
//!
//! Design §3: "Data layers: `security_label` in the content-addressed manifest —
//! cryptographically bound; can't be relabeled without changing the hash
//! (stronger than SELinux xattrs)." The per-op monitor (S2) reads the object's
//! label from here and feeds it into `subject.ctx ⊒ object.label`.
//!
//! ## Why this is a trait, not a struct
//!
//! **The content-addressed CAS manifest layer does not exist yet** in this
//! codebase (the only "manifest" types today are OCI image manifests in
//! `hyprstream-workers` and release/context manifests — none is the data-plane
//! CAS manifest with a `security_label`). So S1 cannot fabricate it. Instead S1
//! defines:
//!
//! - [`LabeledObject`] — the seam the monitor (S2) and PDP (S4) program against:
//!   "give me this object's label." Implemented later by the real manifest type.
//! - [`ContentBoundLabel`] — the *binding contract* a real manifest must honor:
//!   the label is covered by the content hash, so relabeling changes the CID.
//! - [`StaticNodeLabel`] — the satisfiable-today case: static schema nodes
//!   (status/ctl/health) whose label is declared in the `$scope` annotation, not
//!   stored in a manifest. These can be labeled and enforced immediately.
//!
//! This keeps the interface S2/S4 consume stable while honestly flagging the
//! data-plane part as blocked (see module-level BLOCKERS in `mod.rs`).

use super::label::SecurityLabel;

/// Anything the reference monitor can ask for a security label.
///
/// The monitor never accepts a label from a token/UCAN/caveat (design §3, §14);
/// it asks the object. For data layers the impl reads the manifest's
/// content-bound `security_label`; for static nodes it returns the schema-
/// declared label.
pub trait LabeledObject {
    /// This object's security label. `None` ⇒ **unlabeled** ⇒ the monitor MUST
    /// deny (no unlabeled-default-allow; design §1 invariant 2). Implementors
    /// must NOT manufacture a permissive default here.
    fn security_label(&self) -> Option<SecurityLabel>;
}

/// The binding contract a real CAS manifest must satisfy: the label is bound to
/// the content hash, so an attacker cannot relabel without producing a new CID
/// (and thus a new object). This is the property that makes the object label
/// *content-truth* rather than a mutable xattr.
///
/// S1 specifies the contract; the implementation lands with the manifest layer.
/// A conforming manifest serializes `security_label` *inside* the bytes that the
/// CID covers — verifying this is the manifest layer's test, asserted here as a
/// documented invariant.
pub trait ContentBoundLabel: LabeledObject {
    /// The content identifier (hash) of the manifest. The label returned by
    /// [`LabeledObject::security_label`] MUST be part of the preimage of this
    /// CID.
    fn content_id(&self) -> &[u8];

    /// Recompute the CID over the manifest bytes (including the label) and check
    /// it equals [`content_id`]. A real impl uses this to *verify* the binding
    /// when a manifest is loaded from the store — a mismatch ⇒ tampered ⇒ deny.
    ///
    /// Default impl is `false` (fail-closed) so a stub that forgets to implement
    /// verification denies rather than trusts.
    ///
    /// [`content_id`]: ContentBoundLabel::content_id
    fn verify_binding(&self) -> bool {
        false
    }
}

/// A static schema node's label (status/ctl/health and other declared nodes).
///
/// This is the immediately-satisfiable half of object labeling: the label comes
/// from the `$scope`/TE annotation in the schema (S3, #569), not from a CAS
/// manifest, so it needs no data-plane store. Genesis labeling (see
/// [`super::genesis`]) assigns one of these to every static node before
/// enforcement turns on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticNodeLabel {
    /// Schema-declared label for this node. Always `Some` once genesis is
    /// complete — a static node with no declared label is a genesis gap and the
    /// `Option` surfaces it as "unlabeled → deny".
    label: Option<SecurityLabel>,
}

impl StaticNodeLabel {
    /// A labeled static node.
    pub fn labeled(label: SecurityLabel) -> Self {
        StaticNodeLabel { label: Some(label) }
    }

    /// An explicitly-unlabeled node (genesis gap). Provided so callers can
    /// represent "no label declared" without reaching for a permissive default.
    pub fn unlabeled() -> Self {
        StaticNodeLabel { label: None }
    }
}

impl LabeledObject for StaticNodeLabel {
    fn security_label(&self) -> Option<SecurityLabel> {
        self.label
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::label::{Assurance, CompartmentSet, Level};
    use super::*;

    #[test]
    fn static_node_returns_declared_label() {
        let l = SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let node = StaticNodeLabel::labeled(l);
        assert_eq!(node.security_label(), Some(l));
    }

    #[test]
    fn unlabeled_static_node_returns_none() {
        // → monitor denies. No permissive default.
        assert!(StaticNodeLabel::unlabeled().security_label().is_none());
    }

    // A stub manifest demonstrating the seam compiles and fails closed.
    struct StubManifest {
        cid: Vec<u8>,
        label: SecurityLabel,
    }
    impl LabeledObject for StubManifest {
        fn security_label(&self) -> Option<SecurityLabel> {
            Some(self.label)
        }
    }
    impl ContentBoundLabel for StubManifest {
        fn content_id(&self) -> &[u8] {
            &self.cid
        }
        // intentionally does NOT override verify_binding → default false.
    }

    #[test]
    fn content_bound_default_verify_fails_closed() {
        let m = StubManifest {
            cid: vec![1, 2, 3],
            label: SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY),
        };
        assert!(!m.verify_binding(), "unimplemented binding check must fail closed");
        assert_eq!(m.content_id(), &[1, 2, 3]);
    }
}
