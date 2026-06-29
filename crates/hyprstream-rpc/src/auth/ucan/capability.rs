//! The typed **capability** — a single `(resource, ability)` grant — and the
//! **attenuation** (⊆) relation that is the core ceiling invariant of S5 (#571).
//!
//! A UCAN delegates *authority* as a set of capabilities. The cardinal rule of
//! the whole epic (#547, design "ceiling/attenuation") is that **a delegation
//! can only ever narrow, never widen**: every capability a child UCAN claims
//! must be *covered by* (⊆) some capability its delegator held. If it is not,
//! the chain is rejected — fail-closed. The compiler then lowers the *root* of a
//! validated chain into a TE ceiling that is guaranteed to be no wider than what
//! was reviewed.
//!
//! ## What lives here vs. what is deferred
//!
//! This module is deliberately **vocabulary-independent** (S5 milestone 1). A
//! [`Capability`] is `(Resource, Ability, Caveats)` where:
//! - [`Resource`] is a hierarchical URI-like string (`mac://model/qwen-7b`); the
//!   subset relation on resources is *prefix/`*` containment* — structural, not
//!   tied to any concrete app vocabulary.
//! - [`Ability`] is the action verb (`infer`, `model/*`, `*`); subset is
//!   namespace containment (`a/b` ⊆ `a/*` ⊆ `*`).
//! - [`Caveats`] is an opaque CBOR map of restrictions; for milestone 1 the
//!   subset rule is the conservative "more caveats = more restricted, and a
//!   child may only *add* caveats or keep them equal" — the safe default. The
//!   value-aware caveat algebra (e.g. numeric ranges) is a milestone-2 seam.
//!
//! The mapping from an [`Ability`]/[`Resource`] to S3's concrete
//! `ScopeAction`/`Operation` + object types — i.e. *what the verbs mean to the
//! enforcement layer* — is the [`super::seam`] that lands after S3 (#582). This
//! module never needs it: attenuation is purely structural.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// A hierarchical resource identifier a capability applies to.
///
/// Treated as a `/`-delimited path with an optional trailing `*` wildcard
/// segment (UCAN-style "resource pointer"). The string is the source of truth;
/// the [`Resource::covers`] relation is the structural subset used by
/// attenuation. We do **not** interpret the scheme/authority semantically here —
/// that interpretation is the deferred vocabulary seam.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Resource(pub String);

impl Resource {
    pub fn new(s: impl Into<String>) -> Self {
        Resource(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Does `self` **cover** `other` — i.e. is `other` within the authority of
    /// `self`? This is the resource half of attenuation: a child resource is
    /// valid only if its delegator's resource covers it.
    ///
    /// Rules (fail-closed; when in doubt, do NOT cover):
    /// - `*` (or empty) covers everything.
    /// - An exact string match covers itself.
    /// - A trailing `/*` (or bare `*` suffix on a segment boundary) covers any
    ///   resource sharing the prefix up to that boundary.
    ///
    /// A child resource that is *broader* than the parent (e.g. parent
    /// `mac://m/a`, child `mac://m/*`) is NOT covered → rejected.
    #[must_use]
    pub fn covers(&self, other: &Resource) -> bool {
        let parent = self.0.as_str();
        let child = other.0.as_str();

        // Top wildcard covers all.
        if parent == "*" || parent.is_empty() {
            return true;
        }
        if parent == child {
            return true;
        }
        // Prefix wildcard: `prefix/*` covers `prefix/...` (at a segment
        // boundary, so `a/b/*` does not cover `a/bc`).
        if let Some(prefix) = parent.strip_suffix("/*") {
            // `prefix/*` covers `prefix` itself and anything under `prefix/`.
            return child == prefix || child.starts_with(&format!("{prefix}/"));
        }
        if let Some(prefix) = parent.strip_suffix('*') {
            // Bare-`*` suffix (no slash): textual prefix containment. Narrower
            // form, kept conservative — only used when an author writes `a*`.
            return child.starts_with(prefix);
        }
        false
    }
}

impl fmt::Display for Resource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// An action/verb a capability authorizes, e.g. `infer`, `model/read`, `*`.
///
/// Subset is namespace containment over `/`-delimited verb segments: `a/b` ⊆
/// `a/*` ⊆ `*`. This mirrors the UCAN ability convention and is intentionally
/// independent of S3's `ScopeAction` enum — the verb→`ScopeAction` mapping is the
/// deferred [`super::seam`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Ability(pub String);

impl Ability {
    pub fn new(s: impl Into<String>) -> Self {
        Ability(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Does `self` **cover** `other`? `*` covers all; `a/*` covers `a` and any
    /// `a/...`; otherwise exact match. Fail-closed: a broader child verb is not
    /// covered.
    #[must_use]
    pub fn covers(&self, other: &Ability) -> bool {
        let parent = self.0.as_str();
        let child = other.0.as_str();
        if parent == "*" {
            return true;
        }
        if parent == child {
            return true;
        }
        if let Some(prefix) = parent.strip_suffix("/*") {
            return child == prefix || child.starts_with(&format!("{prefix}/"));
        }
        false
    }
}

impl fmt::Display for Ability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque restriction map attached to a capability (UCAN "caveats").
///
/// Milestone-1 semantics are deliberately conservative and value-blind: a child
/// capability's caveats must be a **superset** (more restrictive) of — or equal
/// to — the parent's. I.e. a child may *add* restrictions or repeat the
/// parent's, but may never *drop* or *loosen* one. Concretely [`Caveats::covers`]
/// requires every parent key to be present in the child with a byte-identical
/// value; the child may carry extra keys (extra restrictions).
///
/// The value-aware algebra (a child `{"max": 5}` being covered by a parent
/// `{"max": 10}`) is a milestone-2 seam — see the module docs and
/// [`super::seam`]. Until then we never *widen* on caveats, which is the
/// fail-closed direction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Caveats(pub BTreeMap<String, CaveatValue>);

/// A single caveat value. Kept as a small, comparable, deterministically
/// serializable shape so caveat maps hash/compare stably. Rich/value-range
/// caveats are milestone 2.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaveatValue {
    /// A textual restriction (the common UCAN case).
    Text(String),
    /// An integer restriction.
    Int(i64),
    /// A boolean restriction.
    Bool(bool),
}

impl Caveats {
    /// The empty (unrestricted) caveat set.
    pub fn empty() -> Self {
        Caveats(BTreeMap::new())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Does `self` (the **parent's** caveats) permit a child carrying
    /// `child` caveats? True iff every restriction the parent imposes is also
    /// imposed, identically, by the child — the child may add more but never
    /// drop or change one. Fail-closed (value-blind in milestone 1).
    #[must_use]
    pub fn covers(&self, child: &Caveats) -> bool {
        self.0.iter().all(|(k, v)| child.0.get(k) == Some(v))
    }
}

/// A single capability: authority to perform [`Ability`] on [`Resource`] subject
/// to [`Caveats`]. The atom of UCAN delegation and the unit attenuation operates
/// on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    /// The resource (`with` in UCAN parlance).
    pub resource: Resource,
    /// The ability/action (`can` in UCAN parlance).
    pub ability: Ability,
    /// Restrictions (UCAN `nb`/caveats). Empty = unrestricted.
    #[serde(default)]
    pub caveats: Caveats,
}

impl Capability {
    /// Construct a capability with no caveats.
    pub fn new(resource: Resource, ability: Ability) -> Self {
        Self {
            resource,
            ability,
            caveats: Caveats::empty(),
        }
    }

    /// Construct with explicit caveats.
    pub fn with_caveats(resource: Resource, ability: Ability, caveats: Caveats) -> Self {
        Self {
            resource,
            ability,
            caveats,
        }
    }

    /// **Attenuation predicate**: does `self` (held by a delegator) *authorize*
    /// `other` (claimed by a delegate)? This is the load-bearing ⊆ relation. A
    /// delegate capability is valid iff its delegator covers it on **all three**
    /// axes:
    ///
    /// 1. resource: `self.resource.covers(other.resource)`
    /// 2. ability:  `self.ability.covers(other.ability)`
    /// 3. caveats:  `self.caveats.covers(other.caveats)` (child at least as
    ///    restricted)
    ///
    /// Conjunction of all three; any axis failing ⇒ NOT authorized (fail-closed).
    /// A delegate that widens on *any* axis is rejected.
    #[must_use]
    pub fn authorizes(&self, other: &Capability) -> bool {
        self.resource.covers(&other.resource)
            && self.ability.covers(&other.ability)
            && self.caveats.covers(&other.caveats)
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.ability, self.resource)?;
        if !self.caveats.is_empty() {
            write!(f, " {:?}", self.caveats.0)?;
        }
        Ok(())
    }
}

/// Is the capability set `claimed` **entirely covered** by the capability set
/// `held`? True iff *every* claimed capability is authorized by *at least one*
/// held capability. This is the set-level attenuation used between adjacent
/// links of a delegation chain.
///
/// Fail-closed: an empty `held` covers only an empty `claimed`; any claimed
/// capability with no covering held capability rejects the whole set.
#[must_use]
pub fn set_attenuates(held: &[Capability], claimed: &[Capability]) -> bool {
    claimed.iter().all(|c| held.iter().any(|h| h.authorizes(c)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn res(s: &str) -> Resource {
        Resource::new(s)
    }
    fn ab(s: &str) -> Ability {
        Ability::new(s)
    }
    fn cap(r: &str, a: &str) -> Capability {
        Capability::new(res(r), ab(a))
    }

    // ---- Resource::covers ------------------------------------------------

    #[test]
    fn resource_exact_match_covers() {
        assert!(res("mac://model/qwen").covers(&res("mac://model/qwen")));
        assert!(!res("mac://model/qwen").covers(&res("mac://model/llama")));
    }

    #[test]
    fn resource_top_wildcard_covers_all() {
        assert!(res("*").covers(&res("mac://anything/at/all")));
        assert!(res("").covers(&res("mac://anything")));
    }

    #[test]
    fn resource_prefix_wildcard_covers_subtree_at_boundary() {
        let parent = res("mac://model/*");
        assert!(parent.covers(&res("mac://model")));
        assert!(parent.covers(&res("mac://model/qwen")));
        assert!(parent.covers(&res("mac://model/qwen/v2")));
        // Must respect the segment boundary: `model/*` does NOT cover `modelX`.
        assert!(!parent.covers(&res("mac://modelX")));
    }

    #[test]
    fn resource_child_cannot_widen() {
        // parent narrow, child broad → NOT covered (the anti-widening invariant).
        assert!(!res("mac://model/qwen").covers(&res("mac://model/*")));
        assert!(!res("mac://model/qwen").covers(&res("*")));
    }

    // ---- Ability::covers -------------------------------------------------

    #[test]
    fn ability_wildcard_and_namespace() {
        assert!(ab("*").covers(&ab("infer")));
        assert!(ab("model/*").covers(&ab("model/read")));
        assert!(ab("model/*").covers(&ab("model")));
        assert!(!ab("model/read").covers(&ab("model/write")));
        // child cannot widen.
        assert!(!ab("model/read").covers(&ab("model/*")));
        assert!(!ab("infer").covers(&ab("*")));
    }

    // ---- Caveats::covers -------------------------------------------------

    #[test]
    fn caveats_empty_parent_covers_anything() {
        let parent = Caveats::empty();
        let mut child = BTreeMap::new();
        child.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        assert!(parent.covers(&Caveats(child)));
    }

    #[test]
    fn caveats_child_may_add_but_not_drop() {
        let mut p = BTreeMap::new();
        p.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        let parent = Caveats(p);

        // Child keeps the restriction AND adds one: covered (more restricted).
        let mut c = BTreeMap::new();
        c.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        c.insert("max".to_owned(), CaveatValue::Int(5));
        assert!(parent.covers(&Caveats(c)));

        // Child drops the parent's restriction: NOT covered (would widen).
        assert!(!parent.covers(&Caveats::empty()));

        // Child changes the restriction's value: NOT covered (value-blind, M1).
        let mut c2 = BTreeMap::new();
        c2.insert("tenant".to_owned(), CaveatValue::Text("other".to_owned()));
        assert!(!parent.covers(&Caveats(c2)));
    }

    // ---- Capability::authorizes (the 3-axis conjunction) -----------------

    #[test]
    fn capability_authorizes_requires_all_three_axes() {
        let held = cap("mac://model/*", "model/*");
        assert!(held.authorizes(&cap("mac://model/qwen", "model/read")));
        // widen resource → reject
        assert!(!held.authorizes(&cap("mac://other/qwen", "model/read")));
        // widen ability → reject
        assert!(!held.authorizes(&cap("mac://model/qwen", "admin/*")));
    }

    #[test]
    fn capability_authorizes_rejects_caveat_drop() {
        let mut p = BTreeMap::new();
        p.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        let held = Capability::with_caveats(res("mac://model/*"), ab("*"), Caveats(p));
        // identical resource/ability but child drops the caveat → reject.
        assert!(!held.authorizes(&cap("mac://model/qwen", "infer")));
    }

    // ---- set_attenuates --------------------------------------------------

    #[test]
    fn set_attenuates_every_claimed_must_be_covered() {
        let held = vec![
            cap("mac://model/*", "model/*"),
            cap("mac://stream/*", "subscribe"),
        ];
        let claimed = vec![
            cap("mac://model/qwen", "model/read"),
            cap("mac://stream/abc", "subscribe"),
        ];
        assert!(set_attenuates(&held, &claimed));

        // one claimed cap escapes the held set → whole set rejected.
        let escapes = vec![
            cap("mac://model/qwen", "model/read"),
            cap("mac://admin/x", "manage"),
        ];
        assert!(!set_attenuates(&held, &escapes));
    }

    #[test]
    fn set_attenuates_empty_held_covers_only_empty() {
        assert!(set_attenuates(&[], &[]));
        assert!(!set_attenuates(&[], &[cap("mac://x", "y")]));
    }
}
