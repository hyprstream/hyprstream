//! **Carrier (c) — bind-time labels for synthetic VFS mounts** (#699 scope item 1,
//! epic #547).
//!
//! A synthetic mount (one a subject binds into its own namespace, the Plan 9
//! composition model) carries a security label computed at bind time from two
//! inputs the binder supplies:
//!
//! - **`binder_floor`** — the binding subject's own clearance floor. Not optional:
//!   a mount must carry a label, and that label can never be *less* restrictive
//!   than what the binder is itself cleared to read (otherwise a subject could
//!   re-export high-clearance data through its own mount labeled low —
//!   declassification by re-serving, the D2 amendment case).
//! - **`declared`** — the label the mount asserts for what it serves.
//!
//! The effective label is [`bind_time_label`]`binder_floor.join(declared)` — the
//! BLP least-upper-bound, a floor not a negotiation. Every node beneath the mount
//! is clamped to *at least* this label ([`BindLabelMap::resolve`] returns it for a
//! descendant), and a descendant's own carrier-(a) label is joined on top
//! ([`clamp_descendant`]) — restrict-only, never less.
//!
//! ## No escape hatch
//!
//! There is **no label-less bind**. [`BindLabelMap::bind`] takes both inputs as
//! `SecurityLabel` (not `Option`), so a mount cannot be constructed without a
//! label at the type level — and a path under a prefix that was never bound
//! resolves to `None` ⇒ **deny** (the MAC model has no default label; absence is
//! denial — the same invariant `hyprstream::mac::exchange` states for grants).
//! There is no `Default`, no
//! `NotApplicable`, no flag that turns the clamp off.

use std::collections::BTreeMap;

use super::label::SecurityLabel;
use super::manifest::bind_time_label;

/// The effective bind-time label a synthetic mount carries: the restrict-only
/// join of the binder's floor and the mount's declared label, via
/// [`bind_time_label`].
///
/// Constructed only by [`BindLabelMap::bind`], which guarantees the label is
/// `⊒ binder_floor` and `⊒ declared` (the join can only raise, never lower).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindLabel(SecurityLabel);

impl BindLabel {
    /// The effective label everything beneath this mount is clamped to.
    pub fn effective(&self) -> SecurityLabel {
        self.0
    }
}

/// The set of synthetic-mount bind-time labels keyed by mount prefix.
///
/// This is the carrier-(c) companion to [`super::genesis::GenesisMap`] (carrier
/// (a) static nodes): where genesis labels the *generated* `/srv/{name}` surface
/// from schema annotations, this labels the *synthetic* mounts a subject composes
/// into its own namespace, from the binder's floor × the mount's declared label.
///
/// Both carriers feed the same [`super::manifest::ObjectLabelResolver`] at
/// enforcement time; this type is the pure-data source for the bind-time half.
#[derive(Debug, Clone, Default)]
pub struct BindLabelMap {
    /// Normalized absolute prefix → effective bind-time label.
    bindings: BTreeMap<String, BindLabel>,
}

impl BindLabelMap {
    /// An empty bind-label map.
    pub fn new() -> Self {
        BindLabelMap::default()
    }

    /// Bind `binder_floor × declared` at `prefix`, storing the effective
    /// [`bind_time_label`] and returning the stored label by value.
    ///
    /// **Both labels are mandatory** (`SecurityLabel`, not `Option`): there is no
    /// label-less bind — a synthetic mount must carry a label, and this is the
    /// only way to add one. The effective label is the restrict-only join, so it
    /// is always `⊒ binder_floor` and `⊒ declared`.
    pub fn bind(
        &mut self,
        prefix: &str,
        binder_floor: SecurityLabel,
        declared: SecurityLabel,
    ) -> BindLabel {
        let normalized = normalize_prefix(prefix);
        let effective = bind_time_label(binder_floor, declared);
        let entry = BindLabel(effective);
        self.bindings.insert(normalized, entry);
        entry
    }

    /// The effective label a mount bound at `prefix` carries, if any. A prefix
    /// that was never bound ⇒ `None` ⇒ unlabeled ⇒ deny.
    pub fn label_for(&self, prefix: &str) -> Option<SecurityLabel> {
        self.bindings
            .get(&normalize_prefix(prefix))
            .map(BindLabel::effective)
    }

    /// Resolve a walked object (path components relative to the namespace root)
    /// to the effective bind-time label of the **nearest enclosing** synthetic
    /// mount — the floor every node beneath that mount is clamped to.
    ///
    /// Walks from the full path up its ancestors; the first bound prefix found is
    /// the enclosing mount, and its label is the descendant floor (the clamp).
    /// `None` ⇒ no enclosing labeled mount ⇒ unlabeled ⇒ deny.
    pub fn resolve(&self, components: &[&str]) -> Option<SecurityLabel> {
        let full = components.len();
        let mut depth = full;
        while depth > 0 {
            let path = join_components(&components[..depth]);
            if let Some(entry) = self.bindings.get(&path) {
                return Some(entry.effective());
            }
            depth -= 1;
        }
        None
    }

    /// Whether any synthetic mount has been bound.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// The number of bound synthetic mounts.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Iterate the bound `(prefix, effective-label)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, SecurityLabel)> {
        self.bindings
            .iter()
            .map(|(p, b)| (p.as_str(), b.effective()))
    }
}

/// Clamp a descendant node's own (carrier-a/manifest) label to at least its
/// enclosing mount's bind-time floor.
///
/// `effective = mount_floor.join(node_label)` — restrict-only: the descendant's
/// effective label is `⊒ mount_floor` and `⊒ node_label`. A node that *declares*
/// a less-restrictive label than its mount cannot lower the effective label below
/// the mount's floor (the declassification-by-declaration case the clamp closes).
/// A node may always declare itself *more* restrictive.
#[must_use]
pub fn clamp_descendant(mount_floor: SecurityLabel, node_label: SecurityLabel) -> SecurityLabel {
    mount_floor.join(&node_label)
}

/// Normalize a prefix to a canonical absolute form: leading slash, no trailing
/// slash, no empty components (`"/srv/model/"`, `"srv/model"` → `"/srv/model"`).
/// The empty/`"/"` prefix normalizes to `""` and is never stored (the namespace
/// root is not itself an addressable mount here).
fn normalize_prefix(prefix: &str) -> String {
    join_components(&prefix.split('/').collect::<Vec<_>>())
}

/// Join path components into a normalized absolute path (`["srv","model"]` →
/// `"/srv/model"`), skipping empty components.
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::label::{Assurance, CompartmentSet, Level};
    use super::*;

    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::PqHybrid, CompartmentSet::EMPTY)
    }

    // ── Test (2): a mount without a label fails to construct ──────────────────
    //
    // `bind` takes both labels as `SecurityLabel` (not Option), so a label-less
    // mount is unconstructible at the type level. Runtime mirror: a path under a
    // prefix that was never bound resolves to None ⇒ deny (no default label).

    #[test]
    fn unbound_prefix_resolves_to_none_deny() {
        // A namespace where /srv/model is bound but /srv/policy is not.
        let mut map = BindLabelMap::new();
        map.bind("/srv/model", label(Level::Internal), label(Level::Internal));
        // /srv/policy carries no label → its descendants are denied.
        assert!(
            map.resolve(&["srv", "policy", "config"]).is_none(),
            "a path under a mount that carries no label must deny, not fall back to a default"
        );
    }

    #[test]
    fn bind_requires_both_labels_type_level() {
        // The only `bind` constructor takes (binder_floor, declared) as
        // SecurityLabel. The effective label is the join of exactly those two —
        // there is no third "no-label" path the caller can reach.
        let mut map = BindLabelMap::new();
        let b = map.bind("/srv/model", label(Level::Internal), label(Level::Confidential));
        // effective = join(Internal, Confidential) = Confidential.
        assert_eq!(b.effective().level, Level::Confidential);
    }

    // ── Test (3): a node beneath a mount is clamped to at least the floor ─────

    #[test]
    fn descendant_inherits_mount_effective_label() {
        let mut map = BindLabelMap::new();
        // A Secret-cleared binder declares Internal; effective = Secret (floor wins).
        map.bind("/srv/policy", label(Level::Secret), label(Level::Internal));
        let under = map.resolve(&["srv", "policy", "config", "current"]).unwrap();
        assert_eq!(
            under.level,
            Level::Secret,
            "a descendant is clamped to at least the mount's effective floor"
        );
    }

    // ── Test (4): the clamp is restrict-only ──────────────────────────────────

    #[test]
    fn clamp_descendant_is_restrict_only() {
        let mount_floor = label(Level::Secret);
        // A node that declares itself LESS restrictive than its mount cannot
        // lower the effective label below the mount floor.
        let node_declared_low = label(Level::Public);
        assert_eq!(
            clamp_descendant(mount_floor, node_declared_low).level,
            Level::Secret,
            "a descendant declaring a lower label cannot lower below the mount floor"
        );
        // A node that declares itself MORE restrictive is honored as-is (still ⊒ mount).
        let node_declared_higher = SecurityLabel::new(
            Level::Secret,
            Assurance::PqHybrid,
            CompartmentSet::single(0),
        );
        let clamped = clamp_descendant(mount_floor, node_declared_higher);
        assert!(clamped.compartments.contains(0));
        assert!(clamped.level >= mount_floor.level);
    }

    #[test]
    fn effective_label_never_below_binder_floor() {
        // The D2 invariant: re-exporting Secret-read data as Public via a mount
        // the binder labeled low still clamps back to (at least) Secret.
        let mut map = BindLabelMap::new();
        map.bind("/reexport", label(Level::Secret), label(Level::Public));
        let effective = map.label_for("/reexport").unwrap();
        assert_eq!(
            effective.level,
            Level::Secret,
            "bind_time_label must not let a declared-low mount go below the binder floor"
        );
    }

    #[test]
    fn resolve_walks_to_nearest_enclosing_mount() {
        let mut map = BindLabelMap::new();
        map.bind("/srv", label(Level::Public), label(Level::Public));
        map.bind(
            "/srv/model",
            label(Level::Internal),
            label(Level::Internal),
        );
        // /srv/model/qwen/status → nearest is /srv/model (Internal), not /srv.
        assert_eq!(
            map.resolve(&["srv", "model", "qwen", "status"]).unwrap().level,
            Level::Internal
        );
        // /srv/other → nearest is /srv (Public).
        assert_eq!(
            map.resolve(&["srv", "other"]).unwrap().level,
            Level::Public
        );
    }

    #[test]
    fn empty_map_denies_everything() {
        let map = BindLabelMap::new();
        assert!(map.is_empty());
        assert!(map.resolve(&["anything"]).is_none());
    }
}
