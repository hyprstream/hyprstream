//! **Production genesis labeling** — content, enumeration, resolution, and the
//! boot-time coverage gate for the native-MAC data plane (S1 activation, #567,
//! epic #547).
//!
//! S1's *mechanism* already lives in [`hyprstream_rpc::auth::mac::genesis`]:
//! [`GenesisMap`] (an explicit path→label map with **no** catch-all default) and
//! [`GenesisReport`] (the completeness gate). This module supplies the pieces the
//! mechanism can't: the actual *content* and the wiring that turns "a map exists"
//! into "every node in the running namespace is covered, and here is the
//! evidence."
//!
//! Four net-new pieces, in the order the boot path uses them:
//!
//! 1. [`NamespaceEnumerator`] — walks the production export surface (the
//!    `/srv/{name}` service inventory + the export-root skeleton +, optionally, a
//!    live [`Namespace`]'s mount prefixes) to produce the **node set** the
//!    completeness check runs against. Without a concrete node set the report is
//!    vacuous; this is what makes "every existing object" a finite, checkable
//!    list.
//! 2. [`SitePolicy`] — the site-policy label table. A small set of **explicit**
//!    per-node rules plus a conservative **floor**, [materialized](SitePolicy::materialize)
//!    into concrete per-node [`GenesisMap`] assignments. This is the sanctioned
//!    "explicit `*`-style rule that materializes to concrete assignments" the
//!    genesis mechanism permits — never an implicit catch-all inside the map.
//! 3. [`CompositeObjectLabelResolver`] — the production [`ObjectLabelResolver`]:
//!    genesis static nodes → bind-time D2 descendant inheritance → CAS
//!    [`BlobManifest`](crate::storage::cas::manifest::BlobManifest)-carried
//!    labels; a manifest present but unlabeled is denied; anything
//!    unresolvable returns `None` ⇒ deny.
//! 4. [`GenesisGate`] — constructed at boot: it runs the enumerator, materialize,
//!    and report, and holds the resolver, so the startup path can log/emit the
//!    [`GenesisReport`] and a status surface can render coverage evidence.
//!
//! The production resolver is consumed by the authoritative 9P translator PEP.
//! Missing and unlabeled objects deny; there is no permissive fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hyprstream_rpc::auth::mac::{
    clamp_descendant, import_label, Assurance, BindLabelMap, GenesisMap, GenesisReport, Lattice,
    LatticeVersion, Level, ObjectLabelResolver, ObjectRef, SecurityLabel,
};
use hyprstream_rpc::metadata::VfsNodeTable;

/// The export-root skeleton nodes the host VFS composes under (mirrors
/// `server::routes::ninep::build_export_mount` + the `#730` host VFS layout).
/// These are structural container directories, always present regardless of which
/// services are running.
pub const EXPORT_ROOTS: &[&str] = &["/", "/srv", "/worktree", "/stream"];

/// Non-service mount points the standard namespace binds (see
/// `services::namespace_builder::build_standard_namespace` and
/// [`hyprstream_vfs::STANDARD_NAMESPACE_PATHS`]). `/srv/model` and `/srv/registry`
/// are *also* services, so they're covered by the service inventory; these are the
/// ones that are not services.
pub const STANDARD_NON_SERVICE_MOUNTS: &[&str] = &["/bin", "/env", "/lang/tcl"];

// ────────────────────────────────────────────────────────────────────────────
// 1. Namespace enumerator
// ────────────────────────────────────────────────────────────────────────────

/// Accumulates the set of node paths a genesis must cover, deduplicated and
/// deterministically ordered.
///
/// The enumerator is intentionally *source-agnostic*: it collects candidate node
/// paths from whichever surfaces are available (the compile-time service factory
/// inventory, the export-root skeleton, a live [`Namespace`]'s mount prefixes) and
/// normalizes them into one sorted, deduplicated set. The completeness check
/// ([`GenesisReport`]) then runs against exactly this set — so the report's node
/// list is only as complete as what the enumerator was told to walk. Adding a new
/// export surface means adding a source here, not changing the mechanism.
#[derive(Debug, Clone, Default)]
pub struct NamespaceEnumerator {
    nodes: BTreeSet<String>,
}

impl NamespaceEnumerator {
    /// An empty enumerator.
    pub fn new() -> Self {
        NamespaceEnumerator::default()
    }

    /// The default production node set: export-root skeleton + every registered
    /// `/srv/{name}` service mount + the standard non-service mounts. This is the
    /// set a daemon exports today; it's derived from the compile-time inventory so
    /// it needs no running namespace.
    pub fn production() -> Self {
        NamespaceEnumerator::new()
            .with_export_roots()
            .with_service_inventory()
            .with_standard_mounts()
    }

    /// Add the export-root skeleton nodes ([`EXPORT_ROOTS`]).
    pub fn with_export_roots(mut self) -> Self {
        for root in EXPORT_ROOTS {
            self.push(root);
        }
        self
    }

    /// Add `/srv/{name}` for every service registered via `#[service_factory]`
    /// (the inventory the daemon actually mounts under `/srv`).
    pub fn with_service_inventory(mut self) -> Self {
        for factory in hyprstream_service::list_factories() {
            self.push(&format!("/srv/{}", factory.name));
        }
        self
    }

    /// Add the standard non-service mount points ([`STANDARD_NON_SERVICE_MOUNTS`]).
    pub fn with_standard_mounts(mut self) -> Self {
        for mount in STANDARD_NON_SERVICE_MOUNTS {
            self.push(mount);
        }
        self
    }

    /// Add every mount prefix of a live [`Namespace`] (e.g. a composed export
    /// namespace). Complements the static inventory with whatever a particular
    /// process actually bound at runtime.
    pub fn with_namespace(mut self, ns: &hyprstream_vfs::Namespace) -> Self {
        for prefix in ns.mount_prefixes() {
            self.push(prefix);
        }
        self
    }

    /// Add a single explicit node path (site-specific extension point).
    pub fn with_node(mut self, path: impl AsRef<str>) -> Self {
        self.push(path.as_ref());
        self
    }

    fn push(&mut self, path: &str) {
        let normalized = normalize_node_path(path);
        if !normalized.is_empty() {
            self.nodes.insert(normalized);
        }
    }

    /// The number of distinct nodes collected.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether no nodes have been collected.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The collected node set, sorted and deduplicated.
    pub fn into_nodes(self) -> Vec<String> {
        self.nodes.into_iter().collect()
    }

    /// Borrow the collected node set (sorted, deduplicated).
    pub fn nodes(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(String::as_str)
    }
}

/// Normalize a node path to a canonical absolute form: leading slash, no trailing
/// slash, no empty components. `""`/`"/"` normalize to the addressable root.
fn normalize_node_path(path: &str) -> String {
    let mut out = String::new();
    for component in path.split('/') {
        if component.is_empty() {
            continue;
        }
        out.push('/');
        out.push_str(component);
    }
    if out.is_empty() {
        "/".to_owned()
    } else {
        out
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Genesis content source (site-policy table)
// ────────────────────────────────────────────────────────────────────────────

/// The floor label for the existing world: `(Public, Unverified, {})` — the
/// lattice bottom. Used as the D2 clamp target for descendants / unlabeled
/// carriers and as the value for explicitly-public structural nodes.
pub fn floor_label() -> SecurityLabel {
    SecurityLabel::bottom()
}

/// The genesis lattice the floor labels are validated against.
///
/// The floor content carries **no compartments** (compartment vocabulary is S3's
/// job, #569), so an empty-vocabulary lattice validates every floor label. When S3
/// lands a real `$scope`-derived vocabulary, this is where it's swapped in — the
/// rest of this module is unchanged.
pub fn genesis_lattice() -> Lattice {
    Lattice::new(LatticeVersion(1), [])
}

/// A site-policy label table: a set of **explicit** per-node rules plus a
/// conservative floor for every other enumerated node.
///
/// This is the genesis *content*. It is deliberately not a wildcard matcher: rules
/// are keyed on the exact normalized node path, and [`materialize`](Self::materialize)
/// emits one concrete [`GenesisMap`] assignment per enumerated node (matched rule,
/// else the floor). That materialization is what the genesis mechanism sanctions —
/// the completeness check then sees real per-node coverage, never an implicit
/// catch-all inside the map.
///
/// The default ([`SitePolicy::conservative`]) reflects the current world's actual
/// sensitivity — control-plane services above data services above public
/// structure — so the genesis is *not* the "label-everything-public" no-op the
/// mechanism warns against. Real per-node labels ultimately come from S3
/// `$scope`/TE annotations; this table is the site-policy seam until then.
#[derive(Debug, Clone)]
pub struct SitePolicy {
    /// Exact node path → label. Longest wins is unnecessary because materialize
    /// runs over a finite enumerated node set and each rule is an exact match.
    rules: std::collections::BTreeMap<String, SecurityLabel>,
    /// Label materialized for any enumerated node without an explicit rule.
    floor: SecurityLabel,
}

impl SitePolicy {
    /// A site policy with only a floor and no explicit elevations. Every
    /// enumerated node materializes to `floor`.
    pub fn floor_only(floor: SecurityLabel) -> Self {
        SitePolicy {
            rules: std::collections::BTreeMap::new(),
            floor,
        }
    }

    /// The default conservative site policy for the current world.
    ///
    /// - **Control plane** (`policy`, `oauth`, `ledger`) → `Confidential`: the
    ///   authorization/identity/audit services are the most sensitive surface.
    /// - **Data / internal services** and the non-service mounts → `Internal`
    ///   (the floor): model/registry/worker/etc. content is org-internal, not
    ///   world-readable.
    /// - **Structural export roots** (`/srv`, `/worktree`, `/stream`) → `Public`:
    ///   these are container directories whose *listing* is not itself sensitive.
    ///
    /// Assurance is left at the `Unverified` floor across the board: elevating the
    /// crypto-assurance axis (#548) is a per-object decision, not a blanket
    /// genesis floor — a genesis that demanded `PqHybrid` everywhere would make
    /// most objects unreachable before the assurance-binding layer is wired.
    pub fn conservative() -> Self {
        let internal =
            SecurityLabel::new(Level::Internal, Assurance::Unverified, Default::default());
        let confidential = SecurityLabel::new(
            Level::Confidential,
            Assurance::Unverified,
            Default::default(),
        );
        let public = SecurityLabel::new(Level::Public, Assurance::Unverified, Default::default());

        let mut rules = std::collections::BTreeMap::new();
        // Control-plane services.
        for name in ["policy", "oauth", "ledger"] {
            rules.insert(format!("/srv/{name}"), confidential);
        }
        // Structural export roots (directory listings, not sensitive content).
        for root in EXPORT_ROOTS {
            rules.insert(normalize_node_path(root), public);
        }
        SitePolicy {
            rules,
            floor: internal,
        }
    }

    /// Add or override an explicit per-node rule (builder).
    pub fn with_rule(mut self, path: impl AsRef<str>, label: SecurityLabel) -> Self {
        self.rules.insert(normalize_node_path(path.as_ref()), label);
        self
    }

    /// The floor label materialized for unmatched nodes.
    pub fn floor(&self) -> SecurityLabel {
        self.floor
    }

    /// The label this policy assigns to `node`: its explicit rule if any, else the
    /// floor. Total — every node gets a concrete label (that's the point).
    pub fn label_for(&self, node: &str) -> SecurityLabel {
        let normalized = normalize_node_path(node);
        self.rules.get(&normalized).copied().unwrap_or(self.floor)
    }

    /// Materialize this policy into a [`GenesisMap`] over `nodes`: one explicit
    /// assignment per node (matched rule, else floor). No implicit catch-all is
    /// left in the map — every assignment is concrete, so the completeness check
    /// sees real coverage.
    pub fn materialize<'a>(&self, nodes: impl IntoIterator<Item = &'a str>) -> GenesisMap {
        let mut map = GenesisMap::new();
        for node in nodes {
            map = map.assign(node, self.label_for(node));
        }
        map
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 3. Composite production ObjectLabelResolver
// ────────────────────────────────────────────────────────────────────────────

/// Resolves a content-addressed object (a CAS [`BlobManifest`]) to its carried
/// label. The seam between the composite resolver and the CAS manifest store.
///
/// The outer `Option` is "does a manifest exist for this CID?" and the inner is
/// the manifest's `security_label` carrier field:
/// - `Some(Some(label))` — manifest present and labeled ⇒ use the label.
/// - `Some(None)` — manifest present but unlabeled ⇒ deny.
/// - `None` — no manifest for this CID ⇒ unresolvable ⇒ deny.
pub trait ManifestLabelSource: Send + Sync {
    /// The label carried by the manifest for `cid`, per the semantics above.
    fn label_for(&self, cid: &[u8]) -> Option<Option<SecurityLabel>>;
}

/// The Stage-0 default manifest source: no CAS manifest store is wired to the
/// resolver yet, so every CID is unresolvable (`None` ⇒ deny). Honest and
/// fail-closed — it never invents a label for content it can't see.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoManifests;

impl ManifestLabelSource for NoManifests {
    fn label_for(&self, _cid: &[u8]) -> Option<Option<SecurityLabel>> {
        None
    }
}

/// The production [`ObjectLabelResolver`]: composes the label sources the
/// reference monitor needs, in priority order, and fails closed when none applies.
///
/// For a **path** reference:
/// 1. **Carrier (a) deny-set** — a *generated* node whose schema declared no
///    `$vfsMac` is unlabeled ⇒ **deny**. It does NOT inherit an ancestor's label
///    (that would be a permissive default on a known surface the operator
///    explicitly left unannotated). Only the *exact* path of a known unlabeled
///    generated node denies; a path the inventory never saw still walks its
///    ancestors.
/// 2. **Carrier (c) bind-time floor** — a path *under* a synthetic mount
///    ([`BindLabelMap`]) inherits that mount's bind-time floor, D2-joined with
///    any genesis/generated label found below (the mount can only raise, never
///    lower).
/// 3. **Genesis static node** (incl. materialized carrier-(a) labels) — an exact
///    assignment, or the nearest labeled non-structural-root ancestor.
/// 4. None ⇒ `None` ⇒ deny.
///
/// For a **CID** reference: the CAS [`ManifestLabelSource`] — a present-but-
/// unlabeled manifest denies, and an absent manifest denies.
///
/// Every arm honors the S1 invariant: an unresolvable reference returns `None`
/// (deny), never a manufactured permissive default.
pub struct CompositeObjectLabelResolver {
    genesis: GenesisMap,
    floor: SecurityLabel,
    manifests: Arc<dyn ManifestLabelSource>,
    /// Carrier (a): exact paths of generated nodes whose schema declared no
    /// `$vfsMac` (or a non-`pq-hybrid` one). These **deny** — they do not fall
    /// through to ancestor inheritance, because the operator left a *known*
    /// surface unannotated and a permissive default there is exactly the silent
    /// hole #1228 closes.
    generated_unlabeled: BTreeSet<String>,
    /// Carrier (c): synthetic-mount bind-time labels. Empty at boot (Stage 0 —
    /// no composed mounts yet); the bind operation (`BindLabelMap::bind`, which
    /// takes both labels as `SecurityLabel` — no label-less construction) feeds
    /// this. `Arc` so the future PEP bind path and this resolver share one map.
    binds: Arc<BindLabelMap>,
}

impl CompositeObjectLabelResolver {
    /// Construct from a materialized genesis map, the D2 clamp floor, a CAS
    /// manifest source, the carrier-(a) generated-node deny-set, and the
    /// carrier-(c) bind-time label map.
    pub fn new(
        genesis: GenesisMap,
        floor: SecurityLabel,
        manifests: Arc<dyn ManifestLabelSource>,
        generated_unlabeled: BTreeSet<String>,
        binds: Arc<BindLabelMap>,
    ) -> Self {
        CompositeObjectLabelResolver {
            genesis,
            floor,
            manifests,
            generated_unlabeled,
            binds,
        }
    }

    /// Resolve a walked path (already split into components) to a label.
    fn resolve_path(&self, components: &[&str]) -> Option<SecurityLabel> {
        if components.is_empty() {
            return self.genesis.resolve("/").copied();
        }
        let full = join_components(components);

        // Carrier (a): a generated node whose schema declared no $vfsMac (or a
        // non-pq-hybrid one) ⇒ unlabeled ⇒ deny. Do NOT walk ancestors — a known
        // unannotated surface must not inherit a permissive label from its
        // service mount (#1228).
        if self.generated_unlabeled.contains(&full) {
            return None;
        }

        // Carrier (c): nearest enclosing synthetic mount's bind-time floor (the
        // D2 clamp target for everything beneath it). `BindLabelMap::resolve`
        // already walks ancestors internally.
        let bind_floor = self.binds.resolve(components);

        // Genesis (incl. materialized carrier-a labels): nearest labeled
        // ancestor, with the structural-root inheritance carve-out.
        let node_label = self.resolve_genesis_label(components);

        // D2-join the bind floor with the node label (restrict-only). Either
        // alone is enough to label the object; neither ⇒ deny.
        match (bind_floor, node_label) {
            (Some(f), Some(l)) => Some(clamp_descendant(f, l)),
            (Some(f), None) => Some(f),
            (None, Some(l)) => Some(l),
            (None, None) => None,
        }
    }

    /// Walk ancestors for the nearest genesis assignment, honoring the
    /// structural-root inheritance carve-out (a `/srv`/`/worktree`/`/stream`
    /// Public label does NOT flow to descendants — otherwise the structural root
    /// acts as an implicit permissive wildcard).
    fn resolve_genesis_label(&self, components: &[&str]) -> Option<SecurityLabel> {
        let full = components.len();
        let mut depth = full;
        while depth > 0 {
            let path = join_components(&components[..depth]);
            if let Some(label) = self.genesis.resolve(&path) {
                let inherited = depth < full;
                let structural_root = EXPORT_ROOTS.contains(&path.as_str());
                if !inherited || !structural_root {
                    return Some(*label);
                }
            }
            depth -= 1;
        }
        None
    }

    /// Resolve a content identifier via the CAS manifest source.
    fn resolve_cid(&self, cid: &[u8]) -> Option<SecurityLabel> {
        match self.manifests.label_for(cid) {
            // Manifest present and labeled ⇒ D2 import clamp to the boundary
            // floor. `import_label` is the restrict-only join: a carried label
            // can only ever be raised toward the floor, never returned below it,
            // so a `Public` carrier under an `Internal` floor still clamps to
            // `Internal` (D2 — object labels clamp to the importing boundary's
            // floor, never returned unrestricted).
            Some(Some(label)) => Some(import_label(self.floor, label)),
            // Present-but-unlabeled and absent manifests both deny. Never
            // manufacture a label or join either case to a floor.
            Some(None) | None => None,
        }
    }
}

impl ObjectLabelResolver for CompositeObjectLabelResolver {
    fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
        match object {
            ObjectRef::Path(components) => self.resolve_path(components),
            ObjectRef::Cid(cid) => self.resolve_cid(cid),
        }
    }
}

/// Join path components into a normalized absolute path (`["srv","model"]` →
/// `"/srv/model"`), matching [`normalize_node_path`]'s output so genesis lookups
/// hit.
fn join_components(components: &[&str]) -> String {
    let mut out = String::new();
    for c in components {
        if c.is_empty() {
            continue;
        }
        out.push('/');
        out.push_str(c);
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Startup gate + report surface
// ────────────────────────────────────────────────────────────────────────────

/// The boot-time genesis coverage gate: enumerate → materialize → report, holding
/// the composite resolver ready for a PEP to consume once activation is decided.
///
/// Constructed once at startup. It does **not** enable enforcement — it produces
/// the [`GenesisReport`] evidence (`is_complete()`, plus the `unlabeled` /
/// `ill_formed` lists) the activation decision gates on, and it owns the
/// production [`CompositeObjectLabelResolver`] that a real decider will later be
/// wired to.
pub struct GenesisGate {
    nodes: Vec<String>,
    lattice: Lattice,
    report: GenesisReport,
    resolver: CompositeObjectLabelResolver,
    /// Carrier (a) coverage over the real generated-node inventory (#699). Kept
    /// on the gate so an operator / status surface can see exactly which
    /// generated nodes are annotated and which are findings.
    generated: GeneratedNodeCoverage,
}

impl GenesisGate {
    /// Build a gate from explicit parts, folding in the **real** generated-node
    /// inventory (carrier (a), collected from every registered `VfsNodeTable`).
    ///
    /// The generated inventory is part of the gate, not a manually-callable
    /// helper: a generated node whose schema declared no `$vfsMac` (or a
    /// non-`pq-hybrid` one) is an unlabeled **finding** that makes the report
    /// incomplete, so enforcement cannot be enabled until every reachable
    /// generated object class is annotated. Labeled generated nodes are
    /// materialized into the genesis map (so the resolver resolves them);
    /// unlabeled ones go into the resolver's deny-set (so they deny, never
    /// inherit an ancestor's label).
    pub fn build(
        nodes: Vec<String>,
        lattice: Lattice,
        policy: &SitePolicy,
        manifests: Arc<dyn ManifestLabelSource>,
    ) -> Self {
        // Site-policy genesis over the enumerated MOUNT nodes only — generated
        // node labels come exclusively from `$vfsMac` below, never from the
        // site-policy floor (an unannotated generated node must deny, not pick
        // up the floor).
        let mut genesis = policy.materialize(nodes.iter().map(String::as_str));

        // Carrier (a): the real generated-node inventory.
        let generated = GeneratedNodeCoverage::for_all_services();
        for (path, label) in &generated.labeled {
            genesis = genesis.assign(path, *label);
        }

        // The report node set = mount nodes ∪ generated-node paths, so the
        // completeness check covers every reachable object class.
        let mut all_nodes: Vec<String> = nodes;
        for (path, _) in &generated.labeled {
            all_nodes.push(path.clone());
        }
        for path in &generated.unlabeled {
            all_nodes.push(path.clone());
        }
        all_nodes.sort();
        all_nodes.dedup();

        let report = genesis.report(all_nodes.iter().map(String::as_str), &lattice);
        let generated_unlabeled: BTreeSet<String> = generated.unlabeled.iter().cloned().collect();
        // Carrier (c): Stage 0 — no synthetic mounts composed yet. The bind map
        // is empty; the future bind operation (`BindLabelMap::bind`, which takes
        // both labels as `SecurityLabel`) feeds the shared map the resolver
        // consults. No label-less construction exists.
        let binds = Arc::new(BindLabelMap::new());
        let resolver = CompositeObjectLabelResolver::new(
            genesis,
            policy.floor(),
            manifests,
            generated_unlabeled,
            binds,
        );
        GenesisGate {
            nodes: all_nodes,
            lattice,
            report,
            resolver,
            generated,
        }
    }

    /// Build the default production gate: the production node inventory, the
    /// conservative site policy, the genesis lattice, and no CAS manifest source
    /// wired yet ([`NoManifests`], Stage 0). The generated-node inventory is
    /// collected from the real `VfsNodeTable` registry.
    pub fn production() -> Self {
        GenesisGate::build(
            NamespaceEnumerator::production().into_nodes(),
            genesis_lattice(),
            &SitePolicy::conservative(),
            Arc::new(NoManifests),
        )
    }

    /// Carrier (a) coverage over the real generated-node inventory: the
    /// `(path, label)` pairs that resolved, and the paths that are findings
    /// (unlabeled ⇒ deny). This is the per-node evidence an operator annotates
    /// against before activation.
    pub fn generated_coverage(&self) -> &GeneratedNodeCoverage {
        &self.generated
    }

    /// The completeness report over the enumerated node set.
    pub fn report(&self) -> &GenesisReport {
        &self.report
    }

    /// The production object-label resolver (for wiring a real decider later).
    pub fn resolver(&self) -> &CompositeObjectLabelResolver {
        &self.resolver
    }

    /// Consume the gate and transfer its production resolver to the live PEP.
    pub fn into_resolver(self) -> CompositeObjectLabelResolver {
        self.resolver
    }

    /// The enumerated node set the report covers.
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// The genesis lattice.
    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Emit the coverage report to the tracing log at boot. `info` on a complete
    /// genesis; `warn` (with the gap lists) on an incomplete one, so an operator
    /// activating enforcement sees exactly which nodes would deny. **Log only —
    /// never changes enforcement state.**
    pub fn log_report(&self) {
        let r = &self.report;
        if r.is_complete() {
            tracing::info!(
                nodes = self.nodes.len(),
                labeled = r.labeled.len(),
                "MAC genesis coverage COMPLETE (dormant): every enumerated node is labeled and well-formed"
            );
        } else {
            tracing::warn!(
                nodes = self.nodes.len(),
                labeled = r.labeled.len(),
                unlabeled = r.unlabeled.len(),
                ill_formed = r.ill_formed.len(),
                "MAC genesis coverage INCOMPLETE (dormant): enforcement must NOT be enabled until gaps are closed"
            );
            for node in &r.unlabeled {
                tracing::warn!(node = %node, "MAC genesis gap: unlabeled node (would deny under enforcement)");
            }
            for (node, err) in &r.ill_formed {
                tracing::warn!(node = %node, error = %err, "MAC genesis gap: ill-formed label (fail-closed → treated as unlabeled)");
            }
        }
    }

    /// Render a human-readable coverage report for a CLI / status surface. This is
    /// the activation coverage-gate evidence: a summary line plus the full
    /// `unlabeled` / `ill_formed` gap lists and the per-node label table.
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let r = &self.report;
        let mut out = String::new();

        let status = if r.is_complete() {
            "COMPLETE"
        } else {
            "INCOMPLETE"
        };
        let _ = writeln!(
            out,
            "MAC genesis coverage: {status} (enforcement dormant — Stage 0)"
        );
        let _ = writeln!(
            out,
            "  nodes={}  labeled={}  unlabeled={}  ill_formed={}  lattice={}",
            self.nodes.len(),
            r.labeled.len(),
            r.unlabeled.len(),
            r.ill_formed.len(),
            self.lattice.version(),
        );

        if !r.unlabeled.is_empty() {
            let _ = writeln!(out, "\n  UNLABELED (would deny under enforcement):");
            for node in &r.unlabeled {
                let _ = writeln!(out, "    - {node}");
            }
        }
        if !r.ill_formed.is_empty() {
            let _ = writeln!(out, "\n  ILL-FORMED (fail-closed → treated as unlabeled):");
            for (node, err) in &r.ill_formed {
                let _ = writeln!(out, "    - {node}: {err}");
            }
        }

        let _ = writeln!(out, "\n  Labeled nodes:");
        for node in &r.labeled {
            // Show each labeled node with the label the site policy assigned it.
            if let Some(label) = self.resolver.genesis.resolve(node) {
                let _ = writeln!(out, "    {node} -> {label}");
            } else {
                let _ = writeln!(out, "    {node} -> <unresolved>");
            }
        }

        out
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 5. Carrier (a) — schema-derived labels on generated mount nodes (#699)
// ────────────────────────────────────────────────────────────────────────────

/// Per-node label coverage of a service's generated `vfs_nodes()` table
/// (#699 carrier (a)): each generated `/srv/{service}/*` node carries its MAC
/// label as a property of its type, derived from the `$vfsMac` schema annotation
/// at codegen (see [`hyprstream_rpc::metadata::VfsNode::mac_label`]).
///
/// A node whose schema declared no `$vfsMac` decodes to `None` ⇒ **unlabeled** ⇒
/// deny ⇒ a **finding** the operator must close (by annotating the schema) before
/// enforcement is enabled. There is no permissive default and no catch-all: this
/// type only partitions what the schemas declared.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeneratedNodeCoverage {
    /// `(mount-relative path, label)` for every node that resolves to a label.
    pub labeled: Vec<(String, SecurityLabel)>,
    /// Mount-relative paths of nodes whose schema declared no `$vfsMac`
    /// (unlabeled ⇒ deny ⇒ finding).
    pub unlabeled: Vec<String>,
}

impl GeneratedNodeCoverage {
    /// Coverage is complete iff every generated node resolves to a label.
    pub fn is_complete(&self) -> bool {
        self.unlabeled.is_empty()
    }

    /// Classify the **real** generated-node inventory — every `vfs_nodes()`
    /// table registered at codegen via `inventory::submit! { VfsNodeTable }` —
    /// into labeled / unlabeled. This is the source the startup gate walks: it
    /// covers every reachable generated object class, not whatever a hand-built
    /// fixture happened to include.
    ///
    /// Deterministic: tables are sorted by service name, and the output's
    /// `labeled`/`unlabeled` are sorted (stable reports + tests).
    pub fn for_all_services() -> Self {
        let mut by_service: BTreeMap<&'static str, &'static [hyprstream_rpc::metadata::VfsNode]> =
            BTreeMap::new();
        for t in inventory::iter::<VfsNodeTable> {
            let (name, nodes) = (t.nodes_fn)();
            by_service.insert(name, nodes);
        }
        let mut labeled = Vec::new();
        let mut unlabeled = Vec::new();
        for (service, nodes) in &by_service {
            let cov = GeneratedNodeCoverage::for_service(service, nodes);
            labeled.extend(cov.labeled);
            unlabeled.extend(cov.unlabeled);
        }
        labeled.sort_by(|a, b| a.0.cmp(&b.0));
        unlabeled.sort();
        GeneratedNodeCoverage { labeled, unlabeled }
    }

    /// Classify a service's generated node table by carrier (a).
    ///
    /// `service` is the `/srv/{service}` mount name (for reporting); `nodes` is
    /// the table the codegen macro emitted (from `vfs_nodes()`).
    pub fn for_service(service: &str, nodes: &[hyprstream_rpc::metadata::VfsNode]) -> Self {
        let mut labeled = Vec::new();
        let mut unlabeled = Vec::new();
        for n in nodes {
            let path = format!("/srv/{}{}", service, normalize_relative(n.path));
            match n.mac_label() {
                Some(label) => labeled.push((path, label)),
                None => unlabeled.push(path),
            }
        }
        // Deterministic order (tests + stable reports).
        labeled.sort_by(|a, b| a.0.cmp(&b.0));
        unlabeled.sort();
        GeneratedNodeCoverage { labeled, unlabeled }
    }
}

/// Normalize a mount-relative node path into an absolute `/srv/{service}/...`
/// segment suffix: prepend `/` if the node path is non-empty, else "" (the
/// service root itself). `{brace}` arg-segments are kept verbatim — they are
/// part of the node's path identity, not resolved here.
fn normalize_relative(path: &str) -> String {
    if path.is_empty() {
        String::new()
    } else if path.starts_with('/') {
        path.to_owned()
    } else {
        format!("/{path}")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Enumerator ───────────────────────────────────────────────────────────

    #[test]
    fn enumerator_production_covers_services_roots_and_mounts() {
        let nodes = NamespaceEnumerator::production().into_nodes();
        // Export roots present.
        for root in EXPORT_ROOTS {
            assert!(
                nodes.contains(&root.to_string()),
                "missing export root {root}"
            );
        }
        // Standard non-service mounts present.
        for m in STANDARD_NON_SERVICE_MOUNTS {
            assert!(nodes.contains(&m.to_string()), "missing standard mount {m}");
        }
        // Known services from the factory inventory appear as /srv/{name}.
        assert!(nodes.contains(&"/srv/policy".to_owned()));
        assert!(nodes.contains(&"/srv/model".to_owned()));
        assert!(nodes.contains(&"/srv/registry".to_owned()));
    }

    #[test]
    fn enumerator_is_sorted_and_deduplicated() {
        let nodes = NamespaceEnumerator::new()
            .with_node("/srv/model")
            .with_node("/srv/model") // dup
            .with_node("/a")
            .with_node("/srv/model/") // trailing slash normalizes to same
            .into_nodes();
        assert_eq!(nodes, vec!["/a".to_owned(), "/srv/model".to_owned()]);
    }

    #[test]
    fn enumerator_normalizes_paths() {
        assert_eq!(normalize_node_path("/srv//model/"), "/srv/model");
        assert_eq!(normalize_node_path("srv/model"), "/srv/model");
        assert_eq!(normalize_node_path("/"), "/");
        assert_eq!(normalize_node_path(""), "/");
    }

    #[test]
    fn enumerator_from_namespace_uses_mount_prefixes() {
        use hyprstream_vfs::Namespace;
        let mut ns = Namespace::new();
        // Mount a trivial synthetic tree at a couple of prefixes.
        let tree = Arc::new(crate::services::fs::SyntheticTree::new(
            crate::services::fs::SyntheticNode::Dir {
                children: Default::default(),
            },
        ));
        ns.mount("/srv/custom", tree.clone()).unwrap();
        ns.mount("/data", tree).unwrap();
        let nodes = NamespaceEnumerator::new().with_namespace(&ns).into_nodes();
        assert!(nodes.contains(&"/srv/custom".to_owned()));
        assert!(nodes.contains(&"/data".to_owned()));
    }

    // ── Site policy / content ────────────────────────────────────────────────

    #[test]
    fn conservative_policy_elevates_control_plane() {
        let p = SitePolicy::conservative();
        assert_eq!(p.label_for("/srv/policy").level, Level::Confidential);
        assert_eq!(p.label_for("/srv/oauth").level, Level::Confidential);
        assert_eq!(p.label_for("/srv/ledger").level, Level::Confidential);
        // Data services fall to the internal floor.
        assert_eq!(p.label_for("/srv/model").level, Level::Internal);
        assert_eq!(p.label_for("/bin").level, Level::Internal);
        // Structural roots are public.
        assert_eq!(p.label_for("/srv").level, Level::Public);
        assert_eq!(p.label_for("/worktree").level, Level::Public);
    }

    #[test]
    fn conservative_policy_is_not_the_public_everything_noop() {
        // The genesis.rs warning: a label-everything-public genesis makes MAC a
        // no-op. Assert the floor is genuinely restrictive (not Public).
        let p = SitePolicy::conservative();
        assert_eq!(p.floor().level, Level::Internal);
    }

    #[test]
    fn materialize_assigns_every_node_no_catch_all() {
        let nodes = [
            "/srv/policy".to_owned(),
            "/srv/model".to_owned(),
            "/srv".to_owned(),
        ];
        let p = SitePolicy::conservative();
        let map = p.materialize(nodes.iter().map(String::as_str));
        // Every node resolves to a concrete label (materialized, not a default).
        assert!(map.resolve("/srv/policy").is_some());
        assert!(map.resolve("/srv/model").is_some());
        assert!(map.resolve("/srv").is_some());
        // A node NOT in the enumerated set is genuinely absent (no catch-all).
        assert!(map.resolve("/srv/unenumerated").is_none());
    }

    #[test]
    fn materialize_report_is_complete_over_enumerated_nodes() {
        let nodes = NamespaceEnumerator::production().into_nodes();
        let lattice = genesis_lattice();
        let map = SitePolicy::conservative().materialize(nodes.iter().map(String::as_str));
        let report = map.report(nodes.iter().map(String::as_str), &lattice);
        assert!(
            report.is_complete(),
            "conservative genesis must fully cover the production node set; gaps: {:?}",
            report.unlabeled
        );
    }

    // ── Composite resolver ───────────────────────────────────────────────────

    fn resolver_over(
        nodes: &[&str],
        manifests: Arc<dyn ManifestLabelSource>,
    ) -> CompositeObjectLabelResolver {
        let p = SitePolicy::conservative();
        let map = p.materialize(nodes.iter().copied());
        CompositeObjectLabelResolver::new(
            map,
            p.floor(),
            manifests,
            BTreeSet::new(),
            Arc::new(BindLabelMap::new()),
        )
    }

    #[test]
    fn resolver_exact_genesis_node() {
        let r = resolver_over(&["/srv/policy"], Arc::new(NoManifests));
        let label = r.resolve(ObjectRef::Path(&["srv", "policy"])).unwrap();
        assert_eq!(label.level, Level::Confidential);
    }

    #[test]
    fn resolver_descendant_inherits_ancestor_label_d2() {
        // A read under /srv/policy inherits the mount's (Confidential) floor —
        // bind-time D2 descendant inheritance, never LESS restrictive.
        let r = resolver_over(&["/srv/policy"], Arc::new(NoManifests));
        let label = r
            .resolve(ObjectRef::Path(&["srv", "policy", "config", "current"]))
            .unwrap();
        assert_eq!(label.level, Level::Confidential);
    }

    #[test]
    fn resolver_unknown_path_denies() {
        let r = resolver_over(&["/srv/policy"], Arc::new(NoManifests));
        // No genesis node and no ancestor → None → deny.
        assert!(r
            .resolve(ObjectRef::Path(&["nowhere", "at", "all"]))
            .is_none());
    }

    struct FixedManifests(Option<Option<SecurityLabel>>);
    impl ManifestLabelSource for FixedManifests {
        fn label_for(&self, _cid: &[u8]) -> Option<Option<SecurityLabel>> {
            self.0
        }
    }

    #[test]
    fn resolver_cid_labeled_manifest() {
        let labeled = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, Default::default());
        let import_floor =
            SecurityLabel::new(Level::Internal, Assurance::Classical, Default::default());
        let genesis = SitePolicy::floor_only(import_floor).materialize(["/srv/policy"]);
        let r = CompositeObjectLabelResolver::new(
            genesis,
            import_floor,
            Arc::new(FixedManifests(Some(Some(labeled)))),
            BTreeSet::new(),
            Arc::new(BindLabelMap::new()),
        );
        let expected = SecurityLabel::new(Level::Secret, Assurance::Classical, Default::default());
        assert_eq!(
            r.resolve(ObjectRef::Cid(&[1, 2, 3])),
            Some(expected),
            "PqHybrid provenance imported through a Classical floor must degrade to Classical"
        );
    }

    #[test]
    fn resolver_cid_unlabeled_manifest_is_denied() {
        let r = resolver_over(&["/srv/policy"], Arc::new(FixedManifests(Some(None))));
        assert!(r.resolve(ObjectRef::Cid(&[1, 2, 3])).is_none());
    }

    #[test]
    fn resolver_cid_absent_manifest_denies() {
        // No manifest for this CID ⇒ unresolvable ⇒ deny.
        let r = resolver_over(&["/srv/policy"], Arc::new(FixedManifests(None)));
        assert!(r.resolve(ObjectRef::Cid(&[9, 9, 9])).is_none());
    }

    #[test]
    fn resolver_no_manifests_denies_every_cid() {
        let r = resolver_over(&["/srv/policy"], Arc::new(NoManifests));
        assert!(r.resolve(ObjectRef::Cid(&[0])).is_none());
    }

    // ── Gate ─────────────────────────────────────────────────────────────────

    #[test]
    fn production_gate_mount_coverage_is_complete() {
        // The site-policy half: every enumerated MOUNT node (/srv/{service},
        // structural roots, standard mounts) is labeled by the conservative
        // policy. This is the pre-existing coverage; carrier (a) adds the
        // generated-node dimension on top.
        let gate = GenesisGate::production();
        let mount_nodes = NamespaceEnumerator::production().into_nodes();
        for node in &mount_nodes {
            assert!(
                gate.report().labeled.contains(node),
                "mount node {node} must be labeled by the site-policy genesis"
            );
        }
    }

    #[test]
    fn production_gate_generated_coverage_is_collected() {
        // Carrier (a) wiring (#1228): the gate folds in the REAL generated-node
        // inventory — not a hand-built fixture. The inventory is non-empty (the
        // daemon's services project generated nodes), and the coverage is
        // surfaced on the gate for operator visibility.
        let gate = GenesisGate::production();
        let cov = gate.generated_coverage();
        let total = cov.labeled.len() + cov.unlabeled.len();
        assert!(
            total > 0,
            "the real generated-node inventory must be collected into the gate"
        );
    }

    #[test]
    fn production_gate_is_incomplete_until_all_generated_nodes_annotated() {
        // The startup activation gate (#1228): a generated node whose schema
        // declared no `$vfsMac` is a finding, so the production gate is
        // INCOMPLETE until every reachable generated object class is annotated.
        // This is the honest Stage 0 state — enforcement MUST NOT be enabled
        // while any generated node is unlabeled. (Schemas are not yet fully
        // `$vfsMac`-annotated, so the unannotated majority are findings here.)
        let gate = GenesisGate::production();
        assert!(
            !gate.generated_coverage().is_complete(),
            "the production gate must surface the unannotated generated nodes as \
             findings — a complete gate before schemas are annotated would mean \
             the inventory is not really gating"
        );
        // And the report itself is incomplete (the generated unlabeled paths are
        // in the report's unlabeled list, not silently dropped).
        assert!(
            !gate.report().is_complete(),
            "the report must reflect the generated-node findings"
        );
    }

    #[test]
    fn gate_render_report_contains_summary_and_labels() {
        // The rendered report carries the status line and each labeled node's
        // assignment, whether the gate is COMPLETE or INCOMPLETE.
        let gate = GenesisGate::production();
        let rendered = gate.render_report();
        assert!(
            rendered.contains("MAC genesis coverage:"),
            "rendered report has a status line: {rendered}"
        );
        assert!(
            rendered.contains("/srv/policy -> confidential"),
            "labeled mount nodes appear in the report: {rendered}"
        );
    }

    #[test]
    fn gate_with_gap_reports_incomplete() {
        // A node set that the floor-only policy still covers is complete; force an
        // incomplete report by using a floor-only policy but reporting over an
        // extra node that was never materialized.
        let nodes = ["/srv/model".to_owned()];
        let lattice = genesis_lattice();
        let policy = SitePolicy::conservative();
        let genesis = policy.materialize(nodes.iter().map(String::as_str));
        // Report over a superset that includes an unmaterialized node.
        let all = ["/srv/model", "/srv/ghost"];
        let report = genesis.report(all, &lattice);
        assert!(!report.is_complete());
        assert_eq!(report.unlabeled, vec!["/srv/ghost".to_owned()]);
    }

    #[test]
    fn gate_resolver_matches_report_coverage() {
        // Every labeled node in the report must resolve to the label materialized
        // into genesis: generated nodes use their schema carrier, while mount
        // nodes use the site-policy label.
        let gate = GenesisGate::production();
        let policy = SitePolicy::conservative();
        for node in &gate.report().labeled {
            let components: Vec<&str> = node
                .split('/')
                .filter(|component| !component.is_empty())
                .collect();
            let carrier_label = gate
                .generated_coverage()
                .labeled
                .iter()
                .find_map(|(path, label)| (path == node).then_some(*label));
            let expected = carrier_label.unwrap_or_else(|| policy.label_for(node));
            assert_eq!(
                gate.resolver().resolve(ObjectRef::Path(&components)),
                Some(expected),
                "labeled node {node} must resolve to its materialized genesis label"
            );
            if carrier_label.is_none() {
                assert_eq!(expected.assurance, Assurance::Unverified);
            }
        }
    }

    // ── Carrier wiring (#1228): the carriers reach the resolver, end-to-end ──

    #[test]
    fn resolver_resolves_real_annotated_generated_node() {
        // End-to-end over the REAL generated-node inventory (not a fixture):
        // the production gate folds in every registered `VfsNodeTable`, so a
        // `$vfsMac`-annotated real schema node resolves through the production
        // resolver to its schema-declared label. `registry`'s `healthCheck` is
        // annotated `internal:pq-hybrid` and projects to `/srv/registry/health`.
        let gate = GenesisGate::production();
        let cov = gate.generated_coverage();
        let health_path = "/srv/registry/health";
        assert!(
            cov.labeled.iter().any(|(p, _)| p == health_path),
            "the annotated registry health node must be in the generated coverage: {:?}",
            cov.labeled
        );
        let label = gate
            .resolver()
            .resolve(ObjectRef::Path(&["srv", "registry", "health"]))
            .unwrap(); // annotated node must resolve to its label
        assert_eq!(label.level, Level::Internal);
        assert_eq!(label.assurance, Assurance::PqHybrid);
    }

    #[test]
    fn resolver_denies_real_unannotated_generated_node() {
        // End-to-end over the REAL inventory: an unannotated generated node (one
        // the schema left without a `$vfsMac`) denies — it does NOT inherit its
        // service mount's label. `registry`'s `getByName` projects to
        // `/srv/registry/{name}` and carries no `$vfsMac`, so it is a finding
        // and resolves to None ⇒ deny (#1228: the silent hole is closed).
        let gate = GenesisGate::production();
        let cov = gate.generated_coverage();
        let name_path = "/srv/registry/{name}";
        assert!(
            cov.unlabeled.iter().any(|p| p == name_path),
            "the unannotated registry {{name}} node must be a finding: {:?}",
            cov.unlabeled
        );
        assert!(
            gate.resolver()
                .resolve(ObjectRef::Path(&["srv", "registry", "{name}"]))
                .is_none(),
            "an unannotated real node must deny, not inherit the mount label"
        );
    }

    #[test]
    fn resolver_bind_time_floor_clamps_descendant_carrier_c() {
        // Carrier (c) reaches the resolver: a synthetic mount bound with a
        // floor clamps every descendant to at least that floor (D2 restrict-
        // only). A descendant with its own label is joined; one without inherits
        // the floor. A path under no bound and no genesis label denies.
        let p = SitePolicy::conservative();
        let map = p.materialize(["/srv/model"]);
        let secret_floor =
            SecurityLabel::new(Level::Secret, Assurance::PqHybrid, Default::default());
        let mut binds = BindLabelMap::new();
        // A Secret-cleared subject binds a synthetic mount at /synth.
        binds.bind("/synth", secret_floor, secret_floor);
        let resolver = CompositeObjectLabelResolver::new(
            map,
            p.floor(),
            Arc::new(NoManifests),
            std::collections::BTreeSet::new(),
            Arc::new(binds),
        );
        // A descendant with no label of its own inherits the bind floor.
        let under = resolver
            .resolve(ObjectRef::Path(&["synth", "child"]))
            .unwrap(); // descendant of a bound mount inherits the floor
        assert_eq!(under.level, Level::Secret);
        // A path under no bound prefix and no genesis assignment denies.
        assert!(resolver
            .resolve(ObjectRef::Path(&["nowhere", "at", "all"]))
            .is_none());
    }

    #[test]
    fn resolver_bind_floor_joins_genesis_label_restrict_only() {
        // Carrier (c) × genesis: a descendant under a Secret bind floor whose
        // genesis label is Internal resolves to Secret (the join — floor wins,
        // D2 restrict-only, never less).
        let p = SitePolicy::conservative();
        let internal = SecurityLabel::new(Level::Internal, Assurance::PqHybrid, Default::default());
        let secret = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, Default::default());
        let map = p.materialize(["/srv"]).assign("/synth/low", internal);
        let mut binds = BindLabelMap::new();
        binds.bind("/synth", secret, secret);
        let resolver = CompositeObjectLabelResolver::new(
            map,
            p.floor(),
            Arc::new(NoManifests),
            std::collections::BTreeSet::new(),
            Arc::new(binds),
        );
        let label = resolver
            .resolve(ObjectRef::Path(&["synth", "low"]))
            .unwrap(); // bind floor joined with genesis label resolves
        assert_eq!(
            label.level,
            Level::Secret,
            "join(Secret floor, Internal label) = Secret (restrict-only)"
        );
    }

    // ── Carrier (a): generated-node label coverage (#699) ─────────────────────

    use hyprstream_rpc::metadata::{VfsNode, VfsNodeKind};

    fn vnode(path: &'static str, mac: &'static str) -> VfsNode {
        VfsNode {
            method: "m",
            path,
            kind: VfsNodeKind::File,
            scope: "query",
            bulk: false,
            mac,
        }
    }

    #[test]
    fn every_annotated_generated_node_resolves_to_a_label() {
        // Carrier (a): a fully-`$vfsMac`-annotated service's generated table has
        // NO unlabeled nodes — every node resolves to a label. Every annotation
        // carries the mandatory `pq-hybrid` assurance.
        let nodes = [
            vnode("status", "internal:pq-hybrid"),
            vnode("{name}", "confidential:pq-hybrid"),
            vnode("ctl", "secret:pq-hybrid:0"),
        ];
        let cov = GeneratedNodeCoverage::for_service("model", &nodes);
        assert!(cov.is_complete(), "unlabeled findings: {:?}", cov.unlabeled);
        assert_eq!(cov.labeled.len(), 3);
        assert_eq!(cov.unlabeled, Vec::<String>::new());
        // Paths are /srv/{service}/{node}.
        let paths: Vec<&str> = cov.labeled.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&"/srv/model/status"));
        assert!(paths.contains(&"/srv/model/{name}"));
        assert!(paths.contains(&"/srv/model/ctl"));
    }

    #[test]
    fn unannotated_generated_node_is_a_finding_not_a_default() {
        // A node whose schema declared no `$vfsMac` is unlabeled ⇒ deny ⇒ a
        // finding. It is NOT defaulted to a permissive label.
        let nodes = [vnode("status", "internal:pq-hybrid"), vnode("secret", "")];
        let cov = GeneratedNodeCoverage::for_service("model", &nodes);
        assert!(!cov.is_complete());
        assert_eq!(cov.unlabeled, vec!["/srv/model/secret"]);
        assert_eq!(cov.labeled.len(), 1);
    }

    #[test]
    fn non_pq_hybrid_assurance_is_a_finding() {
        // #556/#1228: the internal VFS carrier requires `pq-hybrid`. A
        // level-only annotation, `unverified`, or `classical` all decode to
        // None ⇒ unlabeled ⇒ finding (the non-PQ path must not silently land).
        let nodes = [
            vnode("a", "internal"), // level-only — assurance mandatory
            vnode("b", "internal:unverified"),
            vnode("c", "internal:classical"),
            vnode("ok", "internal:pq-hybrid"), // the one accepted form
        ];
        let cov = GeneratedNodeCoverage::for_service("model", &nodes);
        assert!(!cov.is_complete());
        assert!(cov.unlabeled.contains(&"/srv/model/a".to_owned()));
        assert!(cov.unlabeled.contains(&"/srv/model/b".to_owned()));
        assert!(cov.unlabeled.contains(&"/srv/model/c".to_owned()));
        assert_eq!(cov.labeled.len(), 1);
    }

    #[test]
    fn malformed_vfs_mac_is_a_finding() {
        // A garbled `$vfsMac` decodes to None ⇒ finding (fail-closed).
        let nodes = [
            vnode("status", "internal:pq-hybrid"),
            vnode("x", "not-a-level"),
        ];
        let cov = GeneratedNodeCoverage::for_service("model", &nodes);
        assert!(!cov.is_complete());
        assert!(cov.unlabeled.contains(&"/srv/model/x".to_owned()));
    }
}
