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

// ────────────────────────────────────────────────────────────────────────────
// Object-label RESOLUTION seam (#699, epic #547 activation prerequisite)
// ────────────────────────────────────────────────────────────────────────────
//
// `LabeledObject` (above) answers "what is THIS object's label" once you
// already hold an object. The reference monitor doesn't hold objects — it
// holds a *walked reference* (a path, or a CID) and needs to resolve THAT to
// a label before it can even construct the `ObjectCtx` the PDP evaluates.
// `ObjectLabelResolver` is that keyed lookup, ratified as ticket #699's scope
// item 2 ("An ObjectLabelResolver trait keyed on the walked path / CID that
// the PEP calls before decide").
//
// This module intentionally does NOT wire a concrete resolver against the
// 9P/VFS `Mount`/`Namespace` types (`hyprstream-vfs`) — that wiring is S2/
// #568's job (the generated-mount PEP), and this crate is platform-
// independent by design (no `hyprstream-vfs` dependency, so it keeps
// compiling for wasm). What lands here is the seam plus the **clamp
// invariants** ratified alongside #699's carriers, which are pure functions
// over `SecurityLabel` with no I/O or VFS dependency at all.

/// A reference to the object being labeled, as available to whatever calls
/// [`ObjectLabelResolver::resolve`] — before any object bytes are read. Mirrors
/// the two carriers #699 actually has data for at this layer: a walked path
/// (matches `Mount::walk`'s `&[&str]` component slice in `hyprstream-vfs`,
/// without this crate depending on that crate) or a content identifier (a
/// manifest CID / CAS hash).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectRef<'a> {
    /// Path components of a walked object (schema/synthetic mount nodes).
    Path(&'a [&'a str]),
    /// Content identifier of a content-addressed object (CAS manifest).
    Cid(&'a [u8]),
}

/// The keyed seam the reference monitor (S2/#568) calls **before** constructing
/// an `ObjectCtx` for the PDP: "resolve this walked reference to its security
/// label." Distinct from [`LabeledObject`] (which answers for an object you
/// already hold) — this answers for a reference you are about to walk.
///
/// `None` ⇒ **unresolvable** ⇒ the monitor MUST deny (design §1 invariant 2,
/// same "no unlabeled-default-allow" rule as [`LabeledObject::security_label`]).
/// Implementors must NOT manufacture a permissive default.
pub trait ObjectLabelResolver {
    /// Resolve `object` to its security label, or `None` if unresolvable.
    fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel>;
}

/// Bind-time label clamp (#699 carrier c, D2 amendment ratified 2026-07-03):
/// a client-controlled mount (the plan9 model — users compose their own
/// namespaces) can label what it serves *up* but never *down below what it
/// can read*, or a subject could read data at a high clearance through its own
/// access and re-export it labeled low through its own mount — declassifying
/// by re-serving.
///
/// `effective = join(binder_floor, declared)`. `join` (BLP least-upper-bound:
/// max level, max assurance requirement, union of compartments) can only make
/// the effective label *more* restrictive than either input, never less — so
/// this is a floor, not a negotiation: `effective ⊒ binder_floor` and
/// `effective ⊒ declared` always hold.
///
/// `binder_floor` is the binding subject's own [`super::context::SecurityContext`]
/// clearance (what `Namespace::bind_mount`'s threaded `Subject` resolves to);
/// `declared` is the label the mount asserts for what it serves.
#[must_use]
pub fn bind_time_label(binder_floor: SecurityLabel, declared: SecurityLabel) -> SecurityLabel {
    binder_floor.join(&declared)
}

/// Foreign-label import floor (#699, D1/D2 amendment ratified 2026-07-03): a
/// label asserted by a federated peer — whether a manifest's `security_label`
/// or (per the D1 ruling) a remote node's own capnp-schema-derived label — is
/// a **hint, not a guarantee**, when we act as a client of that peer. It is
/// trusted only as far as the peer's own signer, never face-value across a
/// domain boundary.
///
/// `effective = join(import_floor_for_that_peer, translated)`. Same clamp
/// shape as [`bind_time_label`] (restrict-only, never escalates); the
/// distinction is only which floor is being enforced — a local subject's own
/// clearance there, a per-peer import ceiling here. `import_floor_for_that_peer`
/// is the enrollment/perimeter-assigned ceiling for imports from that specific
/// peer (see the atproto perimeter gateway, #549); `translated` is the peer's
/// asserted label after perimeter translation into this node's lattice
/// vocabulary.
#[must_use]
pub fn import_label(import_floor_for_that_peer: SecurityLabel, translated: SecurityLabel) -> SecurityLabel {
    import_floor_for_that_peer.join(&translated)
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

    // ── #699: ObjectLabelResolver seam + clamp invariants ────────────────────

    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::PqHybrid, CompartmentSet::EMPTY)
    }

    /// A stub resolver over both `ObjectRef` variants, demonstrating the seam
    /// compiles and fails closed for an unknown reference.
    struct StubResolver;
    impl ObjectLabelResolver for StubResolver {
        fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
            match object {
                ObjectRef::Path(["srv", "model", "qwen"]) => Some(label(Level::Confidential)),
                ObjectRef::Cid(_) | ObjectRef::Path(_) => None,
            }
        }
    }

    #[test]
    fn resolver_resolves_known_path_denies_unknown() {
        let r = StubResolver;
        assert_eq!(
            r.resolve(ObjectRef::Path(&["srv", "model", "qwen"])),
            Some(label(Level::Confidential))
        );
        assert!(
            r.resolve(ObjectRef::Path(&["srv", "model", "unknown"])).is_none(),
            "an unresolvable reference must deny, not fall back to a default label"
        );
        assert!(r.resolve(ObjectRef::Cid(&[9, 9, 9])).is_none());
    }

    /// Bind-time clamp: a mount can label what it serves MORE restrictive than
    /// its binder's own floor, but never less — the declassification/re-export
    /// laundering case the D2 amendment closes.
    #[test]
    fn bind_time_label_cannot_declassify_below_the_binder_floor() {
        let floor = label(Level::Secret);

        // Attempt to re-export Secret-read data as Public: the clamp raises it
        // back to (at least) Secret.
        let declared_low = label(Level::Public);
        let effective = bind_time_label(floor, declared_low);
        assert_eq!(
            effective.level,
            Level::Secret,
            "declaring a lower label must not lower the effective label below the binder's floor"
        );

        // Declaring MORE restrictive than the floor is honored as declared.
        let declared_high = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY);
        let effective_high = bind_time_label(floor, declared_high);
        assert_eq!(effective_high, declared_high.join(&floor));
    }

    /// Import clamp: a peer's asserted label is honored only up to the
    /// per-peer import ceiling — a permissive peer cannot hand us a label more
    /// permissive than our own floor for that peer.
    #[test]
    fn import_label_cannot_exceed_peer_floor_downward() {
        let peer_floor = label(Level::Internal);
        let peer_asserts = label(Level::Public); // a peer claiming its data is public
        let effective = import_label(peer_floor, peer_asserts);
        assert_eq!(
            effective.level,
            Level::Internal,
            "a peer's permissive self-assertion cannot go below our import floor for that peer"
        );
    }

    /// Both clamps are restrict-only: the effective label's level is always
    /// the MAXIMUM of the two inputs' levels — never a mix that is LESS
    /// restrictive than either (the property that makes this a floor, not a
    /// negotiation).
    #[test]
    fn clamps_never_produce_a_label_less_restrictive_than_either_input() {
        let a = label(Level::Confidential);
        let b = label(Level::Internal);
        let joined = bind_time_label(a, b);
        assert_eq!(joined.level, Level::Confidential.max(Level::Internal));
        assert!(joined.level >= a.level);
        assert!(joined.level >= b.level);
    }
}
