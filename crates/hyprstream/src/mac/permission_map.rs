//! The production [`PermissionMap`] (#676): the S3 scope ↔ S5 TE-rule
//! vocabulary, **injective and exact**.
//!
//! ## Shape (and why)
//!
//! A [`TeRule`] is a concrete numeric triple `(subject_type, object_type,
//! action)` — a wildcard is unrepresentable in the compiled matrix, and
//! inventing a sentinel "any object" id would put wildcard matching on the PEP
//! decision path (a standing fail-open hazard). So wildcards exist **only on
//! the grant side** (`Capability::authorizes` does directional covering) and
//! this map expands them **at compile time** over a closed registry:
//!
//! - `mac://<class>/*` expands to one rule per **registered** object of that
//!   class — never beyond the registry.
//! - a `*` ability expands over exactly [`ScopeAction::ALL`] (a closed enum).
//! - anything unrecognized (class, name, verb) maps to **nothing**
//!   (fail-closed; never a guessed rule).
//!
//! Because the expansion is injective — each concrete `(class, name, verb)`
//! maps to its own distinct rule, and no two distinct accesses share a rule —
//! [`granted_access`](ScopePermissionMap) is the **exact inverse** of a rule,
//! which is trivially its most-permissive access: the tightest upper bound of
//! a singleton is itself. There is no join/LUB logic that could be "too
//! narrow" and mask an escalation (the #676 risk). The property tests below
//! pin exactly this: every request that projects onto a rule is authorized by
//! that rule's `granted_access`.
//!
//! ## Registry determinism
//!
//! Object ids are assigned positionally over the **sorted** `(class, name)`
//! registry, so two nodes constructing the map from the same object list agree
//! on every id — the compiled matrix and the PEP intern identically with no
//! side table. The registry is closed at construction: an object added later
//! has no rule and is **denied by default**; the remedy is a re-compile at a
//! new policy generation, not a silent wildcard.

use std::collections::BTreeMap;

use hyprstream_rpc::auth::ucan::capability::{Ability, Capability, Caveats, Resource};

use crate::mac::compiler::{AccessRequest, PermissionMap};
use crate::mac::te::{Action, ObjectType, ScopeAction, SubjectType, TeRule};
use std::collections::BTreeSet;

/// The production S3-scope permission map. See the module docs for the
/// injective/exact contract and the registry-determinism rule.
#[derive(Debug, Clone)]
pub struct ScopePermissionMap {
    /// The TE domain this map compiles rules for.
    subject_type: SubjectType,
    /// `(class, name)` → interned object id. Ids are positional over the
    /// sorted registry (see module docs).
    by_name: BTreeMap<(String, String), ObjectType>,
    /// Exact inverse of `by_name`.
    by_id: BTreeMap<ObjectType, (String, String)>,
}

impl ScopePermissionMap {
    /// Build the map for `subject_type` over a closed object registry of
    /// `(class, name)` pairs (e.g. `("model", "llama")`). Duplicates collapse;
    /// ids are assigned `0..n` over the sorted registry (deterministic across
    /// nodes given the same registry).
    pub fn new<C, N>(subject_type: SubjectType, objects: impl IntoIterator<Item = (C, N)>) -> Self
    where
        C: Into<String>,
        N: Into<String>,
    {
        let sorted: BTreeSet<(String, String)> = objects
            .into_iter()
            .map(|(c, n)| (c.into(), n.into()))
            .collect();
        let mut by_name = BTreeMap::new();
        let mut by_id = BTreeMap::new();
        for (i, key) in sorted.into_iter().enumerate() {
            let id = ObjectType(i as u32);
            by_name.insert(key.clone(), id);
            by_id.insert(id, key);
        }
        Self {
            subject_type,
            by_name,
            by_id,
        }
    }

    /// Number of registered objects (diagnostics / tests).
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Is the registry empty? (An empty registry grants nothing — fail-closed.)
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Look up a registered object's interned id (PEP-side interning: the PEP
    /// resolves the same `(class, name)` to the same id the compiler used).
    pub fn object_type(&self, class: &str, name: &str) -> Option<ObjectType> {
        self.by_name
            .get(&(class.to_owned(), name.to_owned()))
            .copied()
    }

    /// Parse a `mac://<class>/<identifier>` resource URI into its parts.
    /// `None` for any other shape (fail-closed). The identifier may be `*`.
    fn parse_resource(resource: &str) -> Option<(&str, &str)> {
        let rest = resource.strip_prefix("mac://")?;
        let (class, ident) = rest.split_once('/')?;
        if class.is_empty() || ident.is_empty() || class == "*" {
            // A wildcard *class* is not expandable: classes partition the
            // registry and a grant over "every class" must be written per
            // class. Fail closed rather than guess.
            return None;
        }
        Some((class, ident))
    }

    /// The registered objects of `class`, in id order.
    fn objects_of_class<'a>(
        &'a self,
        class: &'a str,
    ) -> impl Iterator<Item = ObjectType> + 'a {
        self.by_name
            .range((class.to_owned(), String::new())..)
            .take_while(move |((c, _), _)| c == class)
            .map(|(_, id)| *id)
    }
}

impl PermissionMap for ScopePermissionMap {
    fn permissions_for(&self, cap: &Capability) -> BTreeSet<TeRule> {
        let mut out = BTreeSet::new();

        // Verb(s): the closed ScopeAction vocabulary. `*` expands over ALL;
        // an unrecognized verb grants nothing.
        let ability = cap.ability.as_str();
        let actions: Vec<Action> = if ability == "*" {
            ScopeAction::ALL.iter().map(|a| Action::from(*a)).collect()
        } else {
            match ScopeAction::parse(ability) {
                Some(a) => vec![Action::from(a)],
                None => return out,
            }
        };

        // Object(s): the closed registry. `*` expands over the class's
        // registered objects; an unregistered name grants nothing.
        let Some((class, ident)) = Self::parse_resource(cap.resource.as_str()) else {
            return out;
        };
        let objects: Vec<ObjectType> = if ident == "*" {
            self.objects_of_class(class).collect()
        } else {
            match self.object_type(class, ident) {
                Some(id) => vec![id],
                None => return out,
            }
        };

        for object_type in objects {
            for &action in &actions {
                out.insert(TeRule {
                    subject_type: self.subject_type,
                    object_type,
                    action,
                });
            }
        }
        out
    }

    fn granted_access(&self, rule: &TeRule) -> Option<AccessRequest> {
        // Exact inverse: the rule was only ever emitted for one concrete
        // (class, name, verb), so that concrete access IS the most-permissive
        // access the rule grants. Unknown subject/object/action ⇒ None ⇒
        // check_no_escalation treats the rule as an escalation (fail-closed).
        if rule.subject_type != self.subject_type {
            return None;
        }
        let (class, name) = self.by_id.get(&rule.object_type)?;
        let verb = ScopeAction::from_action(rule.action)?.as_str();
        Some(AccessRequest {
            resource: Resource::new(format!("mac://{class}/{name}")),
            ability: Ability::new(verb),
            // Empty caveats = the broadest claim (Caveats::covers requires the
            // grant's caveats to be present in the claim). A rule enforces no
            // caveats at the PEP, so its most-permissive access is caveat-free
            // — and a caveated grant therefore cannot authorize it (correct:
            // the matrix would have dropped the caveat, i.e. widened).
            caveats: Caveats::empty(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! #676: the map is injective+exact and `granted_access` upper-bounds every
    //! request that projects onto its rule. The property tests are the tripwire
    //! that fails if the map is ever made non-injective or the inverse drifts.
    use super::*;
    use proptest::prelude::*;

    fn map() -> ScopePermissionMap {
        ScopePermissionMap::new(
            SubjectType(1),
            [
                ("model", "llama"),
                ("model", "qwen"),
                ("model", "mistral"),
                ("adapter", "lora-a"),
                ("adapter", "lora-b"),
            ],
        )
    }

    fn cap(res: &str, ab: &str) -> Capability {
        Capability::new(Resource::new(res), Ability::new(ab))
    }

    // ── Determinism / interning ─────────────────────────────────────────────

    #[test]
    fn ids_are_deterministic_across_construction_order() {
        let a = ScopePermissionMap::new(
            SubjectType(1),
            [("model", "llama"), ("adapter", "lora-a"), ("model", "qwen")],
        );
        let b = ScopePermissionMap::new(
            SubjectType(1),
            [("model", "qwen"), ("model", "llama"), ("adapter", "lora-a")],
        );
        for (class, name) in [("model", "llama"), ("model", "qwen"), ("adapter", "lora-a")] {
            assert_eq!(
                a.object_type(class, name),
                b.object_type(class, name),
                "id assignment must not depend on registry iteration order"
            );
        }
    }

    // ── Fail-closed: unknown anything grants nothing ────────────────────────

    #[test]
    fn unrecognized_inputs_grant_nothing() {
        let m = map();
        // Unknown verb, unknown object, unknown class, malformed URI, wildcard class.
        for c in [
            cap("mac://model/llama", "fly"),
            cap("mac://model/gpt-x", "query"),
            cap("mac://widget/llama", "query"),
            cap("https://model/llama", "query"),
            cap("mac://model", "query"),
            cap("mac://*/llama", "query"),
        ] {
            assert!(
                m.permissions_for(&c).is_empty(),
                "unrecognized capability must grant nothing: {c}"
            );
        }
        // Unknown rule components invert to None (⇒ escalation upstream).
        assert!(m
            .granted_access(&TeRule {
                subject_type: SubjectType(2), // wrong domain
                object_type: ObjectType(0),
                action: Action(0),
            })
            .is_none());
        assert!(m
            .granted_access(&TeRule {
                subject_type: SubjectType(1),
                object_type: ObjectType(999), // unregistered object
                action: Action(0),
            })
            .is_none());
        assert!(m
            .granted_access(&TeRule {
                subject_type: SubjectType(1),
                object_type: ObjectType(0),
                action: Action(99), // outside the schema enum
            })
            .is_none());
    }

    // ── Wildcard expansion is exactly the registry, nothing more ────────────

    #[test]
    fn wildcard_identifier_expands_to_exactly_the_class_registry() {
        let m = map();
        let rules = m.permissions_for(&cap("mac://model/*", "query"));
        assert_eq!(rules.len(), 3, "3 registered models, one rule each");
        // No cross-class leak: none of the adapter ids appear.
        for rule in &rules {
            let (class, _) = &m.by_id[&rule.object_type];
            assert_eq!(class, "model");
        }
    }

    #[test]
    fn wildcard_ability_expands_to_exactly_the_closed_verb_set() {
        let m = map();
        let rules = m.permissions_for(&cap("mac://model/llama", "*"));
        assert_eq!(rules.len(), ScopeAction::ALL.len());
    }

    // ── THE #676 obligation: granted_access upper-bounds the rule's grant ───

    /// Round-trip exactness: for every capability the grant could contain,
    /// every rule it lowers to inverts to an access that (a) exists and
    /// (b) the capability itself authorizes. If the map ever lowered a narrow
    /// capability to a broad rule, (b) is the assertion that fails.
    #[test]
    fn every_emitted_rule_inverts_to_an_access_the_capability_authorizes() {
        let m = map();
        let grants = [
            cap("mac://model/llama", "query"),
            cap("mac://model/*", "infer"),
            cap("mac://adapter/*", "*"),
            cap("mac://model/qwen", "train"),
        ];
        for g in &grants {
            let rules = m.permissions_for(g);
            assert!(!rules.is_empty(), "grant {g} must lower to at least one rule");
            for rule in rules {
                let access = m
                    .granted_access(&rule)
                    .expect("every emitted rule must have a recognized inverse");
                let claimed = Capability::with_caveats(
                    access.resource.clone(),
                    access.ability.clone(),
                    access.caveats.clone(),
                );
                assert!(
                    g.authorizes(&claimed),
                    "granted_access({rule:?}) = {claimed} must be within the \
                     capability {g} that emitted the rule"
                );
            }
        }
    }

    proptest! {
        /// The upper-bound property (#676): any concrete request that PROJECTS
        /// onto a rule this map can emit is authorized by that rule's
        /// `granted_access`. Because the map is injective+exact this is an
        /// equality — this test is the tripwire that fails if the map is ever
        /// made collapsing/non-injective without upgrading `granted_access`
        /// to a least-upper-bound (the change that could silently mask
        /// escalation).
        #[test]
        fn granted_access_upper_bounds_every_request_projecting_to_the_rule(
            class_i in 0usize..4,      // model, adapter + 2 unregistered classes
            name_i in 0usize..5,       // registered + unregistered names
            verb_i in 0usize..11,      // 9 canonical verbs + 2 junk
        ) {
            let m = map();
            let classes = ["model", "adapter", "widget", "tensor"];
            let names = ["llama", "qwen", "mistral", "lora-a", "ghost"];
            let mut verbs: Vec<&str> = ScopeAction::ALL.iter().map(|a| a.as_str()).collect();
            verbs.push("fly");
            verbs.push("q");

            let class = classes[class_i];
            let name = names[name_i];
            let verb = verbs[verb_i];

            // The PEP-side projection of the concrete request: same interning
            // the compiler used (object_type via the shared registry, action
            // via the canonical ScopeAction ids).
            let projected = m.object_type(class, name).zip(ScopeAction::parse(verb)).map(
                |(object_type, a)| TeRule {
                    subject_type: SubjectType(1),
                    object_type,
                    action: Action::from(a),
                },
            );

            match projected {
                None => {
                    // Unregistered/unrecognized: no rule exists to permit it —
                    // deny-by-default. Nothing to upper-bound.
                }
                Some(rule) => {
                    let access = m.granted_access(&rule)
                        .expect("a projectable rule must have a recognized inverse");
                    let granted = Capability::with_caveats(
                        access.resource.clone(),
                        access.ability.clone(),
                        access.caveats.clone(),
                    );
                    let request = Capability::new(
                        Resource::new(format!("mac://{class}/{name}")),
                        Ability::new(verb),
                    );
                    prop_assert!(
                        granted.authorizes(&request),
                        "granted_access({rule:?}) = {granted} must cover the \
                         request {request} that projects onto the rule"
                    );
                }
            }
        }

        /// Injectivity: two DISTINCT concrete accesses never project onto the
        /// same rule. This is the structural precondition that makes the exact
        /// inverse sound; if someone deliberately collapses the map, this
        /// fails first and points at the granted_access contract.
        #[test]
        fn distinct_accesses_project_to_distinct_rules(
            i in 0usize..5, j in 0usize..5,
            vi in 0usize..9, vj in 0usize..9,
        ) {
            let m = map();
            let objects = [
                ("model", "llama"), ("model", "qwen"), ("model", "mistral"),
                ("adapter", "lora-a"), ("adapter", "lora-b"),
            ];
            let (ci, ni) = objects[i];
            let (cj, nj) = objects[j];
            let va = ScopeAction::ALL[vi];
            let vb = ScopeAction::ALL[vj];
            let ra = TeRule {
                subject_type: SubjectType(1),
                object_type: m.object_type(ci, ni).unwrap(),
                action: Action::from(va),
            };
            let rb = TeRule {
                subject_type: SubjectType(1),
                object_type: m.object_type(cj, nj).unwrap(),
                action: Action::from(vb),
            };
            let same_access = (ci, ni, va) == (cj, nj, vb);
            prop_assert_eq!(
                ra == rb,
                same_access,
                "rules must be equal exactly when the concrete accesses are"
            );
        }
    }

    // ── End-to-end through the compiler gate ───────────────────────────────

    #[test]
    fn compile_and_gate_end_to_end_with_the_production_map() {
        use crate::mac::compiler::{check_no_escalation, compile};
        use crate::mac::lattice::{Compartment, Lattice, LatticeVersion};
        use hyprstream_rpc::auth::ucan::token::{Did, Ucan, UcanPayload};

        // compile/check_no_escalation read only the capability set; chain/
        // signature verification happens upstream (mirrors compiler.rs tests).
        let did = Did::from_ed25519(&[0u8; 32]);
        let ucan = Ucan {
            payload: UcanPayload {
                issuer: did.clone(),
                audience: did,
                capabilities: vec![cap("mac://model/*", "query")],
                not_before: None,
                expiration: Some(9_999_999_999),
                nonce: vec![],
            },
            proofs: vec![],
            signature: vec![],
        };

        let m = map();
        let lattice = Lattice::new(LatticeVersion(1), [Compartment::new("pii")]);
        let policy = compile(&ucan, &lattice, &m);
        // 3 registered models × 1 verb.
        assert_eq!(policy.matrix.sorted_rules().0.len(), 3);
        // The gate passes: nothing in the matrix exceeds the grant.
        check_no_escalation(&ucan, &policy, &m)
            .expect("an honestly-compiled wildcard grant must pass the gate");
    }
}
