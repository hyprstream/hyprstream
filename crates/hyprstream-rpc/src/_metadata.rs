//! RPC schema introspection metadata types.
//!
//! These types represent metadata extracted from Cap'n Proto schema annotations
//! (`$scope`/`$capability`, `$mcpDescription`, `$cliHidden`, `$paramDescription`).
//! They are the shared source of truth used by both the `generate_rpc_service!`
//! codegen macro and the runtime `ServiceFactory` inventory.

/// Schema metadata for a single RPC method parameter.
///
/// Extracted from Cap'n Proto struct field definitions and `$paramDescription` annotations.
#[derive(Debug, Clone)]
pub struct ParamMeta {
    pub name: &'static str,
    pub type_name: &'static str,
    pub required: bool,
    pub description: &'static str,
    /// Default value hint (empty = no default). For Bool fields this is "false".
    pub default_value: &'static str,
}

/// Schema metadata for a single RPC method.
///
/// Extracted from Cap'n Proto union variants and schema annotations
/// (`$scope`/`$capability`, `$mcpDescription`, `$cliHidden`).
#[derive(Debug, Clone)]
pub struct MethodMeta {
    pub name: &'static str,
    pub params: &'static [ParamMeta],
    pub description: &'static str,
    /// Required authorization scope action from `$scope`/`$capability` — one of the
    /// `ScopeAction` enumerants (S3, #547). Mandatory: empty only for a method that
    /// declared `$scopeExempt`. Maps 1:1 onto `hyprstream::auth::Operation`.
    pub scope: &'static str,
    pub is_streaming: bool,
    pub is_scoped: bool,
    pub scope_field: &'static str,
    /// Whether to hide from auto-generated interfaces (from `$cliHidden`).
    pub hidden: bool,
    /// VFS usage example from `$docExample` annotation. Empty = no example.
    pub doc_example: &'static str,
}

/// The 9P/VFS node kind a generated method projects to (epic #539).
///
/// Mirrors the `VfsNodeKind` capnp enum. `File`/`Dir`/`Query` are read-only
/// (side-effect-free 9P reads); `Ctl` is a write-a-verb control file; `Stream`
/// is a `/stream`-style pipe.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VfsNodeKind {
    /// Readable (maybe writable) leaf; `read` renders the method's response.
    File,
    /// Directory; `readdir` lists child entries (e.g. a `list` method).
    Dir,
    /// Write-a-verb control file; a `write` invokes the method.
    Ctl,
    /// `/stream`-style pipe (data/info/ctl) for a streaming method.
    Stream,
    /// Synthetic query file: write a spec, read the result.
    Query,
}

impl VfsNodeKind {
    /// Whether a `read` of this node must be side-effect-free (9P read invariant).
    /// `File`/`Dir`/`Query` are read surfaces; `Ctl`/`Stream` mutate/produce.
    pub fn is_read_only(self) -> bool {
        matches!(self, Self::File | Self::Dir | Self::Query)
    }
}

/// A single node in a service's generated 9P/VFS projection (epic #539, T2).
///
/// Built by codegen from each method's `MethodMeta` + `$vfs*` annotations. The
/// runtime mount engine (T3) consumes these to serve `/srv/{service}` — `path`
/// is mount-relative (fid-0 rooted, never absolute); `{braces}` in `path` are
/// segments that bind the method's scalar argument.
#[derive(Copy, Clone, Debug)]
pub struct VfsNode {
    /// The method this node projects (snake_case), used to dispatch on access.
    pub method: &'static str,
    /// Mount-relative path; `{name}` segments bind a scalar arg from the walk.
    pub path: &'static str,
    /// The projected node kind (inferred from scope+shape, or `$vfsKind`).
    pub kind: VfsNodeKind,
    /// Authorization scope action (from `MethodMeta::scope`).
    pub scope: &'static str,
    /// `$vfsBulk`: `read` returns raw mmap bytes, bypassing capnp (DMA path).
    pub bulk: bool,
    /// `$vfsMac` annotation text declaring this node's MAC security label
    /// (#699 carrier (a)): the label is a property of the node's type, derived
    /// at codegen. Decode with [`VfsNode::mac_label`]. Empty ⇒ the node is
    /// unlabeled ⇒ deny ⇒ a genesis-coverage finding (no permissive default).
    pub mac: &'static str,
}

impl VfsNode {
    /// Decode this node's `$vfsMac` annotation into a [`SecurityLabel`].
    ///
    /// Returns `None` when the annotation is empty, malformed, or carries a
    /// non-`pq-hybrid` assurance — the node is then **unlabeled**, which the MAC
    /// model treats as **deny** (there is no permissive default; absence is
    /// denial). Such nodes surface as gaps in the genesis coverage report rather
    /// than being silently defaulted.
    ///
    /// Format: `"<level>:<assurance>[:<compBit>[,<compBit>...]]"` where `level`
    /// is one of `public`/`internal`/`confidential`/`secret`, `assurance` is
    /// **required** and must be exactly `pq-hybrid` (the internal non-interop
    /// carrier rejects `unverified`/`classical`/omitted — #556), and each
    /// `compBit` is a `u32` compartment bit index (the lattice vocabulary is S3's
    /// job, #569).
    pub fn mac_label(&self) -> Option<crate::auth::mac::SecurityLabel> {
        parse_mac_annotation(self.mac)
    }
}

/// Parse a `$vfsMac` annotation string into a [`SecurityLabel`].
///
/// `""` / unparseable / non-`pq-hybrid` ⇒ `None` (unlabeled ⇒ deny). See
/// [`VfsNode::mac_label`].
///
/// **Assurance is mandatory and MUST be `pq-hybrid`** (#556, #1228): the
/// generated 9P/VFS surface is an *internal, non-interop* carrier, and the
/// operator ruling is PQ-hybrid required on all deployments. A level-only
/// annotation, `unverified`, or `classical` would produce a label a classical
/// subject dominates — the non-PQ path #556 was enacted to remove, reappearing
/// as a silent default. Those forms are therefore **rejected** (⇒ unlabeled ⇒
/// deny ⇒ a genesis-coverage finding), never silently downgraded.
fn parse_mac_annotation(s: &str) -> Option<crate::auth::mac::SecurityLabel> {
    use crate::auth::mac::{Assurance, CompartmentSet, Level, SecurityLabel};

    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut parts = s.split(':');
    let level = match parts.next()?.trim() {
        "public" => Level::Public,
        "internal" => Level::Internal,
        "confidential" => Level::Confidential,
        "secret" => Level::Secret,
        _ => return None,
    };
    // Assurance is MANDATORY and must be exactly `pq-hybrid`. Omitted,
    // `unverified`, and `classical` are all rejected ⇒ None ⇒ unlabeled ⇒ deny
    // (this is the internal non-interop surface; #556 removes the classical
    // floor, and this carrier must not silently reintroduce it).
    let assurance = match parts.next().map(str::trim) {
        Some("pq-hybrid") => Assurance::PqHybrid,
        // Omitted, empty, unverified, classical, or unknown ⇒ reject.
        None | Some("") | Some("unverified") | Some("classical") | Some(_) => return None,
    };
    let compartments = match parts.next().map(str::trim) {
        None | Some("") => CompartmentSet::EMPTY,
        Some(bits) => {
            let mut set = CompartmentSet::EMPTY;
            for b in bits.split(',') {
                let b = b.trim();
                if b.is_empty() {
                    continue;
                }
                let idx: u32 = b.parse().ok()?;
                set = set.union(CompartmentSet::single(idx));
            }
            set
        }
    };
    // A fourth field is not part of the grammar → malformed → deny.
    if parts.next().is_some() {
        return None;
    }
    Some(SecurityLabel::new(level, assurance, compartments))
}

/// Function type for a service's VFS node table.
///
/// Returns `(service_name, nodes)`.
pub type VfsNodesFn = fn() -> (&'static str, &'static [VfsNode]);

/// Inventory-registered generated VFS node table (#699 carrier (a) inventory).
///
/// Submitted by `generate_rpc_service!` for every schema that has request
/// variants (i.e. every service that projects a `/srv/{name}` mount with
/// generated nodes). The genesis coverage gate collects the union of these
/// tables via [`inventory::iter::<VfsNodeTable>()`] so the startup gate covers
/// every reachable generated object class — not just the ones a hand-built
/// fixture happened to include.
///
/// The `nodes_fn` indirection keeps the table lazy (zero-cost until the gate
/// walks it) and lets this crate stay free of a `hyprstream-service`
/// dependency: the macro emits the `inventory::submit!` in whatever crate owns
/// the generated client module, and the gate (in the daemon crate) iterates it.
pub struct VfsNodeTable {
    /// The `/srv/{name}` service this table belongs to.
    pub name: &'static str,
    /// `fn() -> (service_name, &[VfsNode])` — the generated `vfs_nodes()`.
    pub nodes_fn: VfsNodesFn,
}

inventory::collect!(VfsNodeTable);

/// Function type for service schema metadata.
///
/// Returns `(service_name, methods)`.
pub type SchemaMetadataFn = fn() -> (&'static str, &'static [MethodMeta]);

/// Function type for scoped service schema metadata.
///
/// Returns `(service_name, scope_name, methods)`.
pub type ScopedSchemaMetadataFn = fn() -> (&'static str, &'static str, &'static [MethodMeta]);

/// Tree node describing a scoped client and its nested children.
/// Generated by the proc macro, consumed by CLI/MCP for dynamic dispatch.
#[derive(Copy, Clone, Debug)]
pub struct ScopedClientTreeNode {
    pub scope_name: &'static str,
    pub scope_field: &'static str,
    pub metadata_fn: ScopedSchemaMetadataFn,
    pub nested: &'static [ScopedClientTreeNode],
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::mac::{Assurance, Level};

    fn node(mac: &'static str) -> VfsNode {
        VfsNode {
            method: "m",
            path: "p",
            kind: VfsNodeKind::File,
            scope: "query",
            bulk: false,
            mac,
        }
    }

    #[test]
    fn mac_label_decodes_level_and_pq_hybrid_assurance() {
        // The only accepted form: <level>:pq-hybrid. Assurance is mandatory.
        let l = node("confidential:pq-hybrid").mac_label().unwrap();
        assert_eq!(l.level, Level::Confidential);
        assert_eq!(l.assurance, Assurance::PqHybrid);
        assert!(l.compartments.is_empty());
    }

    #[test]
    fn mac_label_decodes_level_and_assurance() {
        let l = node("secret:pq-hybrid").mac_label().unwrap();
        assert_eq!(l.level, Level::Secret);
        assert_eq!(l.assurance, Assurance::PqHybrid);
    }

    #[test]
    fn mac_label_decodes_compartments() {
        // Compartments are the optional third field, after the mandatory
        // `pq-hybrid` assurance.
        let l = node("internal:pq-hybrid:0,3").mac_label().unwrap();
        assert_eq!(l.level, Level::Internal);
        assert_eq!(l.assurance, Assurance::PqHybrid);
        assert!(l.compartments.contains(0));
        assert!(l.compartments.contains(3));
        assert!(!l.compartments.contains(1));
    }

    #[test]
    fn mac_label_empty_is_unlabeled_deny() {
        // No $vfsMac ⇒ unlabeled ⇒ None ⇒ deny. No permissive default.
        assert!(node("").mac_label().is_none());
        assert!(node("   ").mac_label().is_none());
    }

    #[test]
    fn mac_label_level_only_is_denied_assurance_is_mandatory() {
        // A level-only annotation (no assurance) is rejected ⇒ None ⇒ deny.
        // Assurance is mandatory on this internal non-interop carrier; a
        // level-only form would default to Unverified, the non-PQ path #556
        // removed. Each level is rejected without an assurance.
        for level in ["public", "internal", "confidential", "secret"] {
            assert!(
                node(level).mac_label().is_none(),
                "level-only `{level}` must deny: assurance is mandatory"
            );
        }
    }

    #[test]
    fn mac_label_unverified_assurance_is_denied() {
        // `unverified` is the non-PQ floor ⇒ rejected on this internal carrier.
        // Each form an operator might reach for that silently drops PQ is denied.
        assert!(node("internal:unverified").mac_label().is_none());
        assert!(node("public:unverified").mac_label().is_none());
    }

    #[test]
    fn mac_label_classical_assurance_is_denied() {
        // `classical` is the explicit non-PQ path #556 removes from internal VFS
        // — a classical subject would dominate it. Rejected ⇒ None ⇒ deny.
        assert!(node("internal:classical").mac_label().is_none());
        assert!(node("secret:classical:0").mac_label().is_none());
    }

    #[test]
    fn mac_label_malformed_is_unlabeled_deny() {
        // An unknown level / assurance / a trailing field ⇒ None ⇒ deny
        // (fail-closed, never a guessed label).
        assert!(node("topsecret").mac_label().is_none());
        assert!(node("secret:quantum").mac_label().is_none());
        assert!(node("secret:pq-hybrid:0:extra").mac_label().is_none());
    }

    #[test]
    fn mac_label_non_numeric_compartment_is_deny() {
        assert!(node("secret:pq-hybrid:pii").mac_label().is_none());
    }
}
