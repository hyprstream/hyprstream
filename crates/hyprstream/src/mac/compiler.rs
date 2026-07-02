//! UCAN → Type Enforcement policy compiler (S5 / #571).
//!
//! Compiles a validated UCAN grant into a TE [`CompiledPolicy`], then verifies the
//! compiled policy grants no permission the grant does not authorize (no privilege
//! escalation) before it is signed. Deny-by-default, fail-closed.
//!
//! Three steps: [`compile`] (grant → TE allow-rules), [`check_no_escalation`]
//! (compiled policy ⊆ grant), and [`compile_policy`] (compile, check, sign — signs
//! only if the check passes).
//!
//! The capability → rule mapping is the [`PermissionMap`] seam; the concrete S3
//! action vocabulary (#582) plugs in here.
// TODO(#571): an SMT proof that the compiled policy ⊆ grant over all requests is
// owed; deferred — no SMT solver is vendored. The differential property tests plus
// `check_no_escalation` are the current guarantees.

use std::collections::{BTreeSet, HashSet};

use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::ucan::approval::{ApprovalBinding, ApprovalError, SignedApproval};
use hyprstream_rpc::auth::ucan::capability::{Ability, Capability, Caveats, Resource};
use hyprstream_rpc::auth::ucan::token::Ucan;
use hyprstream_rpc::crypto::pq::MlDsaSigningKey;

use crate::mac::compiled::{CompiledPolicy, PolicyDistError};
use crate::mac::lattice::Lattice;
use crate::mac::te::{Decision, TeMatrix, TeRule};

/// Maps a UCAN capability to the TE rules it grants. The injection point for the
/// concrete action vocabulary — the production impl is
/// [`crate::mac::permission_map::ScopePermissionMap`] (#676).
pub trait PermissionMap {
    /// The TE rules a single capability grants. Empty for an unrecognized
    /// capability — it grants nothing (fail-closed; never a wildcard rule).
    fn permissions_for(&self, cap: &Capability) -> BTreeSet<TeRule>;

    /// The most-permissive concrete access a rule grants at runtime, used to check
    /// the rule against the grant. `None` for an unrecognized rule (treated as an
    /// escalation). Must be deterministic and independent of `permissions_for`.
    fn granted_access(&self, rule: &TeRule) -> Option<AccessRequest>;
}

/// A concrete access request: the (resource, ability, caveats) a subject presents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessRequest {
    pub resource: Resource,
    pub ability: Ability,
    pub caveats: Caveats,
}

impl AccessRequest {
    fn as_capability(&self) -> Capability {
        Capability::with_caveats(
            self.resource.clone(),
            self.ability.clone(),
            self.caveats.clone(),
        )
    }
}

/// Compile a validated UCAN grant into a TE policy: the union of every capability's
/// rules over the given lattice generation. Deterministic (order-independent), and
/// emits no rule not granted by some capability.
pub fn compile(ucan: &Ucan, lattice: &Lattice, permissions: &impl PermissionMap) -> CompiledPolicy {
    let mut allow: BTreeSet<TeRule> = BTreeSet::new();
    for cap in ucan.capabilities() {
        allow.extend(permissions.permissions_for(cap));
    }
    let matrix = TeMatrix::from_allow(allow.into_iter().collect::<HashSet<_>>());
    CompiledPolicy::new(matrix, lattice)
}

/// A rule the compiled policy grants that the UCAN grant does not authorize — a
/// privilege escalation. A policy with one of these must never be signed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivilegeEscalation {
    /// The offending rule.
    pub rule: TeRule,
    /// The access the rule grants (its `granted_access`); `None` if the rule has no
    /// recognized meaning (also an escalation).
    pub access: Option<AccessRequest>,
}

/// Verify the compiled policy grants no permission beyond the UCAN grant.
///
/// For each rule the matrix permits, this projects the rule back to the access it
/// grants (`granted_access`) and requires some capability in the grant to authorize
/// that access. It checks the emitted matrix against the independent `authorizes`
/// test — it does NOT re-run `permissions_for`. That matters: re-running the forward
/// map would let a `PermissionMap` that lowers a narrow capability to a broad rule
/// pass itself; checking each rule's access against the grant instead catches it.
pub fn check_no_escalation(
    ucan: &Ucan,
    policy: &CompiledPolicy,
    permissions: &impl PermissionMap,
) -> Result<(), PrivilegeEscalation> {
    let grant = ucan.capabilities();
    let (allow, _escalate) = policy.matrix.sorted_rules();
    for rule in allow {
        if policy.matrix.te_decision(rule) != Decision::Permit {
            continue;
        }
        let access = permissions.granted_access(&rule);
        let authorized = match &access {
            Some(req) => {
                let claimed = req.as_capability();
                grant.iter().any(|c| c.authorizes(&claimed))
            }
            None => false,
        };
        if !authorized {
            return Err(PrivilegeEscalation { rule, access });
        }
    }
    Ok(())
}

/// The first rule the grant's capabilities map to that the compiled policy omits, if
/// any — a completeness gap. Non-security (a narrower policy never over-grants) and
/// NOT part of the signing gate; for diagnostics only.
pub fn missing_permission(
    ucan: &Ucan,
    policy: &CompiledPolicy,
    permissions: &impl PermissionMap,
) -> Option<TeRule> {
    let present: BTreeSet<TeRule> = policy.matrix.sorted_rules().0.into_iter().collect();
    for cap in ucan.capabilities() {
        for rule in permissions.permissions_for(cap) {
            if !present.contains(&rule) {
                return Some(rule);
            }
        }
    }
    None
}

/// Decide a request against a UCAN grant directly, independent of any compiled
/// policy: permit iff some capability in the grant authorizes it, else deny. This is
/// the reference used to check the compiled policy.
pub fn authorize(grant: &[Capability], request: &AccessRequest) -> Decision {
    let claimed = request.as_capability();
    if grant.iter().any(|c| c.authorizes(&claimed)) {
        Decision::Permit
    } else {
        Decision::Deny
    }
}

/// As [`authorize`], but also deny if `now` is outside the grant's validity window
/// `[not_before, expiration]` (inclusive) — matching the chain validator's temporal
/// gate.
pub fn authorize_at(
    grant: &[Capability],
    not_before: Option<u64>,
    expiration: Option<u64>,
    request: &AccessRequest,
    now: u64,
) -> Decision {
    if let Some(nbf) = not_before {
        if now < nbf {
            return Decision::Deny;
        }
    }
    if let Some(exp) = expiration {
        if now > exp {
            return Decision::Deny;
        }
    }
    authorize(grant, request)
}

/// Errors from [`compile_policy`].
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// The compiled policy grants more than the grant authorizes; not signed.
    #[error("compiled policy escalates privilege beyond the grant: {0:?}")]
    Escalation(PrivilegeEscalation),
    /// Hashing the policy failed.
    #[error("policy hash failed: {0}")]
    Hash(#[from] PolicyDistError),
    /// Signing the approval failed.
    #[error("approval signing failed: {0}")]
    Approval(#[from] ApprovalError),
}

/// Compile a policy from a validated UCAN grant, verify it does not escalate, and
/// sign the approval — signing ONLY if the check passes (fail-closed). The caller
/// must have already validated the UCAN chain. Returns the compiled policy (for
/// distribution) and the signed approval (for the loader).
pub fn compile_policy(
    ucan: &Ucan,
    lattice: &Lattice,
    permissions: &impl PermissionMap,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<(CompiledPolicy, SignedApproval), CompileError> {
    let policy = compile(ucan, lattice, permissions);
    if let Err(e) = check_no_escalation(ucan, &policy, permissions) {
        return Err(CompileError::Escalation(e));
    }
    let hash = policy.policy_hash()?;
    let binding = ApprovalBinding::new(ucan, policy.generation, hash)?;
    let signed = SignedApproval::sign(binding, ed_sk, pq_sk)?;
    Ok((policy, signed))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mac::compiled::{
        sign_policy, PolicyApproval, PolicyLoader, PolicySigner, PolicyVerifier,
    };
    use crate::mac::lattice::{Compartment, LatticeVersion};
    use crate::mac::te::{Action as TeAction, ObjectType, SubjectType};
    use hyprstream_rpc::auth::ucan::capability::{CaveatValue, Caveats};
    use hyprstream_rpc::auth::ucan::token::{Did, UcanPayload};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    // A concrete, deterministic test PermissionMap over a closed alphabet:
    // resources `mac://model/<name>` (+ wildcard `mac://model/*`) → ObjectType;
    // abilities `model/read` → Action(0), `model/write` → Action(1). A wildcard
    // resource fans out to every concrete object (so the compiler enumerates, never
    // emits a wildcard rule). Unknown verbs/resources → empty (fail-closed).

    const OBJECTS: &[&str] = &["qwen", "llama", "mistral"];

    fn object_id(name: &str) -> Option<ObjectType> {
        OBJECTS
            .iter()
            .position(|o| *o == name)
            .map(|i| ObjectType(i as u32))
    }

    fn action_id(verb: &str) -> Option<TeAction> {
        match verb {
            "model/read" => Some(TeAction(0)),
            "model/write" => Some(TeAction(1)),
            _ => None,
        }
    }

    // Inverses for granted_access (id → concrete name/verb), independent of
    // permissions_for.
    fn object_name(id: ObjectType) -> Option<&'static str> {
        OBJECTS.get(id.0 as usize).copied()
    }
    fn action_verb(a: TeAction) -> Option<&'static str> {
        match a.0 {
            0 => Some("model/read"),
            1 => Some("model/write"),
            _ => None,
        }
    }

    /// The allow rules of a compiled policy, as an order-stable set.
    fn allow_set(p: &CompiledPolicy) -> BTreeSet<TeRule> {
        p.matrix.sorted_rules().0.into_iter().collect()
    }

    struct TestPermissions {
        subject: SubjectType,
    }

    impl PermissionMap for TestPermissions {
        fn permissions_for(&self, cap: &Capability) -> BTreeSet<TeRule> {
            let mut out = BTreeSet::new();
            let Some(action) = action_id(cap.ability.as_str()) else {
                return out;
            };
            let res = cap.resource.as_str();
            let concrete: Vec<&str> = if let Some(rest) = res.strip_prefix("mac://model/") {
                if rest == "*" {
                    OBJECTS.to_vec()
                } else if OBJECTS.contains(&rest) {
                    vec![rest]
                } else {
                    vec![]
                }
            } else {
                vec![]
            };
            for name in concrete {
                if let Some(object_type) = object_id(name) {
                    out.insert(TeRule {
                        subject_type: self.subject,
                        object_type,
                        action,
                    });
                }
            }
            out
        }

        fn granted_access(&self, rule: &TeRule) -> Option<AccessRequest> {
            if rule.subject_type != self.subject {
                return None;
            }
            let name = object_name(rule.object_type)?;
            let verb = action_verb(rule.action)?;
            Some(AccessRequest {
                resource: Resource::new(format!("mac://model/{name}")),
                ability: Ability::new(verb),
                caveats: Caveats::empty(),
            })
        }
    }

    fn permissions() -> TestPermissions {
        TestPermissions {
            subject: SubjectType(1),
        }
    }

    fn lattice(gen: u32) -> Lattice {
        Lattice::new(
            LatticeVersion(gen),
            [Compartment::new("pii"), Compartment::new("finance")],
        )
    }

    fn cap(res: &str, ab: &str) -> Capability {
        Capability::new(Resource::new(res), Ability::new(ab))
    }

    // A real hybrid identity, so the signing-gate tests can sign.
    struct Identity {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        pq_vk: hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
        did: Did,
    }

    fn identity() -> Identity {
        use ed25519_dalek::SigningKey as Sk;
        use hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair;
        use rand::rngs::OsRng;
        let ed_sk = Sk::generate(&mut OsRng);
        let did = Did::from_ed25519(&ed_sk.verifying_key().to_bytes());
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        Identity {
            ed_sk,
            pq_sk,
            pq_vk,
            did,
        }
    }

    fn ucan_with(caps: Vec<Capability>) -> Ucan {
        let did = Did::from_ed25519(&[0u8; 32]);
        Ucan {
            payload: UcanPayload {
                issuer: did.clone(),
                audience: did,
                capabilities: caps,
                not_before: None,
                expiration: Some(9_999_999_999),
                nonce: vec![],
            },
            proofs: vec![],
            signature: vec![],
        }
    }

    fn ucan_signed_by(id: &Identity, caps: Vec<Capability>) -> Ucan {
        use hyprstream_rpc::crypto::cose_sign::sign_composite;
        let payload = UcanPayload {
            issuer: id.did.clone(),
            audience: id.did.clone(),
            capabilities: caps,
            not_before: None,
            expiration: Some(9_999_999_999),
            nonce: vec![],
        };
        let bytes = payload.signing_bytes().unwrap();
        let signature = sign_composite(
            &id.ed_sk,
            Some(&id.pq_sk),
            &bytes,
            hyprstream_rpc::auth::ucan::APPROVAL_AAD,
        )
        .unwrap();
        Ucan {
            payload,
            proofs: vec![],
            signature,
        }
    }

    // ---- compile: no escalation (policy ⊆ grant) ----------------------------

    #[test]
    fn compile_is_subset_of_grant() {
        let p = permissions();
        let l = lattice(3);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let policy = compile(&u, &l, &p);
        assert_eq!(check_no_escalation(&u, &policy, &p), Ok(()));
        assert_eq!(
            allow_set(&policy),
            BTreeSet::from([TeRule {
                subject_type: SubjectType(1),
                object_type: ObjectType(0),
                action: TeAction(0),
            }])
        );
    }

    #[test]
    fn check_detects_injected_escalation() {
        // Inject a rule the grant does NOT authorize, whose honest granted_access
        // (model/llama, model/write) the single-capability grant can't cover.
        let p = permissions();
        let l = lattice(3);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let mut policy = compile(&u, &l, &p);
        let rogue = TeRule {
            subject_type: SubjectType(1),
            object_type: ObjectType(1), // llama
            action: TeAction(1),        // write
        };
        let mut allow: HashSet<TeRule> = policy.matrix.sorted_rules().0.into_iter().collect();
        allow.insert(rogue);
        policy.matrix = TeMatrix::from_allow(allow);
        match check_no_escalation(&u, &policy, &p) {
            Err(PrivilegeEscalation { rule, access }) => {
                assert_eq!(rule, rogue);
                assert_eq!(
                    access,
                    Some(AccessRequest {
                        resource: Resource::new("mac://model/llama"),
                        ability: Ability::new("model/write"),
                        caveats: Caveats::empty(),
                    })
                );
            }
            other => panic!("expected escalation, got {other:?}"),
        }
    }

    #[test]
    fn missing_permission_reports_dropped_rule() {
        // Completeness is separate and non-security.
        let p = permissions();
        let l = lattice(3);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let mut policy = compile(&u, &l, &p);
        // Drop the only rule.
        policy.matrix = TeMatrix::from_allow(HashSet::new());
        // The security check (no escalation) is trivially OK on an empty matrix...
        assert_eq!(check_no_escalation(&u, &policy, &p), Ok(()));
        // ...but the completeness check reports the dropped rule.
        assert!(missing_permission(&u, &policy, &p).is_some());
    }

    #[test]
    fn wildcard_grant_fans_out_no_wildcard_rule() {
        // A `mac://model/*` grant enumerates every concrete object — the matrix has
        // interned ids, never a wildcard, so it cannot widen to a future object the
        // grant never reviewed.
        let p = permissions();
        let l = lattice(3);
        let u = ucan_with(vec![cap("mac://model/*", "model/read")]);
        let policy = compile(&u, &l, &p);
        assert_eq!(allow_set(&policy).len(), OBJECTS.len());
        assert_eq!(check_no_escalation(&u, &policy, &p), Ok(()));
    }

    // ---- default-deny on unmapped capabilities / requests -------------------

    #[test]
    fn unmapped_capability_grants_nothing() {
        let p = permissions();
        let l = lattice(3);
        let u = ucan_with(vec![
            cap("mac://model/qwen", "admin/superuser"),
            cap("mac://other/x", "model/read"),
        ]);
        let policy = compile(&u, &l, &p);
        assert!(allow_set(&policy).is_empty(), "unmapped grants emit no rules");
        assert_eq!(check_no_escalation(&u, &policy, &p), Ok(()));
    }

    #[test]
    fn default_deny_on_unmapped_request() {
        let p = permissions();
        let l = lattice(3);
        let u = ucan_with(vec![cap("mac://model/*", "model/read")]);
        let policy = compile(&u, &l, &p);
        // Unmapped verb → no rule → deny.
        let req = AccessRequest {
            resource: Resource::new("mac://model/qwen"),
            ability: Ability::new("admin/x"),
            caveats: Caveats::empty(),
        };
        assert_eq!(matrix_decides(&p, &policy.matrix, &req), Decision::Deny);
        // Unmapped resource → no rule → deny.
        let req2 = AccessRequest {
            resource: Resource::new("mac://other/z"),
            ability: Ability::new("model/read"),
            caveats: Caveats::empty(),
        };
        assert_eq!(matrix_decides(&p, &policy.matrix, &req2), Decision::Deny);
    }

    // ---- TE matrix well-formedness ------------------------------------------

    #[test]
    fn matrix_well_formed_no_contradiction() {
        // allow ∩ escalate = ∅ (a triple is never both Permit and Escalate), even
        // with a non-empty escalate band; te_decision is single-valued and
        // default-deny; and the compiler itself emits no escalate band.
        let allow = HashSet::from([
            TeRule {
                subject_type: SubjectType(1),
                object_type: ObjectType(0),
                action: TeAction(0),
            },
            TeRule {
                subject_type: SubjectType(1),
                object_type: ObjectType(1),
                action: TeAction(0),
            },
        ]);
        let escalate = HashSet::from([TeRule {
            subject_type: SubjectType(1),
            object_type: ObjectType(2),
            action: TeAction(1),
        }]);
        let matrix = TeMatrix::new(allow, escalate);
        let (a_vec, e_vec) = matrix.sorted_rules();
        let a_set: BTreeSet<_> = a_vec.into_iter().collect();
        let e_set: BTreeSet<_> = e_vec.into_iter().collect();
        assert!(!e_set.is_empty(), "must exercise a non-empty escalate band");
        assert!(a_set.is_disjoint(&e_set), "allow ∩ escalate must be ∅");
        for r in &a_set {
            assert_eq!(matrix.te_decision(*r), Decision::Permit);
        }
        for r in &e_set {
            assert_eq!(matrix.te_decision(*r), Decision::Escalate);
        }
        let absent = TeRule {
            subject_type: SubjectType(9),
            object_type: ObjectType(9),
            action: TeAction(9),
        };
        assert_eq!(matrix.te_decision(absent), Decision::Deny, "absent ⇒ deny");

        let p = permissions();
        let l = lattice(3);
        let policy = compile(&ucan_with(vec![cap("mac://model/qwen", "model/read")]), &l, &p);
        assert!(
            policy.matrix.sorted_rules().1.is_empty(),
            "the compiler must not produce an escalate band"
        );
    }

    // ---- compile: determinism -----------------------------------------------

    #[test]
    fn compile_is_deterministic_byte_identical_hash() {
        let p = permissions();
        let l = lattice(5);
        let u1 = ucan_with(vec![
            cap("mac://model/qwen", "model/read"),
            cap("mac://model/llama", "model/write"),
        ]);
        let u2 = ucan_with(vec![
            cap("mac://model/llama", "model/write"),
            cap("mac://model/qwen", "model/read"),
        ]);
        let h1 = compile(&u1, &l, &p).policy_hash().unwrap();
        let h2 = compile(&u2, &l, &p).policy_hash().unwrap();
        assert_eq!(h1, h2, "compilation must be order-independent");
    }

    #[test]
    fn compile_generation_bound_to_lattice() {
        let p = permissions();
        let l = lattice(42);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        assert_eq!(compile(&u, &l, &p).generation, 42);
    }

    // ---- authorize (the reference decision) ---------------------------------

    #[test]
    fn authorize_decides_via_attenuation() {
        let grant = vec![cap("mac://model/*", "model/read")];
        assert_eq!(
            authorize(
                &grant,
                &AccessRequest {
                    resource: Resource::new("mac://model/qwen"),
                    ability: Ability::new("model/read"),
                    caveats: Caveats::empty(),
                }
            ),
            Decision::Permit
        );
        assert_eq!(
            authorize(
                &grant,
                &AccessRequest {
                    resource: Resource::new("mac://model/qwen"),
                    ability: Ability::new("model/write"),
                    caveats: Caveats::empty(),
                }
            ),
            Decision::Deny
        );
    }

    #[test]
    fn authorize_respects_caveats() {
        let mut cv = BTreeMap::new();
        cv.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        let grant = vec![Capability::with_caveats(
            Resource::new("mac://model/*"),
            Ability::new("model/read"),
            Caveats(cv.clone()),
        )];
        // No caveat → not covered → deny.
        assert_eq!(
            authorize(
                &grant,
                &AccessRequest {
                    resource: Resource::new("mac://model/qwen"),
                    ability: Ability::new("model/read"),
                    caveats: Caveats::empty(),
                }
            ),
            Decision::Deny
        );
        // Exact caveat → permit.
        assert_eq!(
            authorize(
                &grant,
                &AccessRequest {
                    resource: Resource::new("mac://model/qwen"),
                    ability: Ability::new("model/read"),
                    caveats: Caveats(cv.clone()),
                }
            ),
            Decision::Permit
        );
        // Different value for the same key → not covered → deny.
        let mut wrong = BTreeMap::new();
        wrong.insert("tenant".to_owned(), CaveatValue::Text("evil".to_owned()));
        assert_eq!(
            authorize(
                &grant,
                &AccessRequest {
                    resource: Resource::new("mac://model/qwen"),
                    ability: Ability::new("model/read"),
                    caveats: Caveats(wrong),
                }
            ),
            Decision::Deny
        );
        // Required caveat plus extra restriction (more restricted) → permit.
        let mut more = cv;
        more.insert("max".to_owned(), CaveatValue::Int(5));
        assert_eq!(
            authorize(
                &grant,
                &AccessRequest {
                    resource: Resource::new("mac://model/qwen"),
                    ability: Ability::new("model/read"),
                    caveats: Caveats(more),
                }
            ),
            Decision::Permit
        );
    }

    #[test]
    fn authorize_at_temporal_gate() {
        // An expired / not-yet-valid grant denies even a covered request.
        let grant = vec![cap("mac://model/qwen", "model/read")];
        let nbf = Some(100);
        let exp = Some(200);
        let req = AccessRequest {
            resource: Resource::new("mac://model/qwen"),
            ability: Ability::new("model/read"),
            caveats: Caveats::empty(),
        };
        assert_eq!(authorize_at(&grant, nbf, exp, &req, 150), Decision::Permit);
        assert_eq!(authorize_at(&grant, nbf, exp, &req, 99), Decision::Deny);
        assert_eq!(authorize_at(&grant, nbf, exp, &req, 201), Decision::Deny);
        // Inclusive boundaries.
        assert_eq!(authorize_at(&grant, nbf, exp, &req, 100), Decision::Permit);
        assert_eq!(authorize_at(&grant, nbf, exp, &req, 200), Decision::Permit);
    }

    // ---- compile_policy: fail-closed signing --------------------------------

    #[test]
    fn compile_policy_signs_only_when_no_escalation() {
        let p = permissions();
        let l = lattice(11);
        let id = identity();
        let u = ucan_signed_by(&id, vec![cap("mac://model/qwen", "model/read")]);
        let (policy, signed) = compile_policy(&u, &l, &p, &id.ed_sk, &id.pq_sk).unwrap();
        signed
            .verify_binds(
                &id.ed_sk.verifying_key(),
                &id.pq_vk,
                &u,
                policy.generation,
                &policy.policy_hash().unwrap(),
            )
            .unwrap();
    }

    /// A PermissionMap that lowers ANY recognized capability to read AND write on
    /// ALL objects — granting more than the source capability authorizes. Its
    /// granted_access is the honest id→request inverse (it does not hide the
    /// over-grant), so check_no_escalation catches the escalation.
    struct EscalatingPermissions {
        subject: SubjectType,
    }
    impl PermissionMap for EscalatingPermissions {
        fn permissions_for(&self, cap: &Capability) -> BTreeSet<TeRule> {
            if action_id(cap.ability.as_str()).is_none() {
                return BTreeSet::new();
            }
            if !cap.resource.as_str().starts_with("mac://model/") {
                return BTreeSet::new();
            }
            let mut out = BTreeSet::new();
            for (oi, _name) in OBJECTS.iter().enumerate() {
                for a in [TeAction(0), TeAction(1)] {
                    out.insert(TeRule {
                        subject_type: self.subject,
                        object_type: ObjectType(oi as u32),
                        action: a,
                    });
                }
            }
            out
        }
        fn granted_access(&self, rule: &TeRule) -> Option<AccessRequest> {
            if rule.subject_type != self.subject {
                return None;
            }
            let name = object_name(rule.object_type)?;
            let verb = action_verb(rule.action)?;
            Some(AccessRequest {
                resource: Resource::new(format!("mac://model/{name}")),
                ability: Ability::new(verb),
                caveats: Caveats::empty(),
            })
        }
    }

    #[test]
    fn compile_policy_refuses_escalating_map() {
        // Drive compile_policy end-to-end with a genuinely escalating map and a
        // narrow grant: it must return CompileError::Escalation and produce no
        // SignedApproval.
        let wide = EscalatingPermissions {
            subject: SubjectType(1),
        };
        let l = lattice(3);
        let id = identity();
        let u = ucan_signed_by(&id, vec![cap("mac://model/qwen", "model/read")]);
        match compile_policy(&u, &l, &wide, &id.ed_sk, &id.pq_sk) {
            Err(CompileError::Escalation(PrivilegeEscalation { access, .. })) => {
                assert!(access.is_some(), "escalation must name a concrete access");
            }
            other => panic!("an escalating map MUST be rejected (no signature); got {other:?}"),
        }
    }

    #[test]
    fn check_catches_escalation_a_self_check_would_miss() {
        // A check that re-ran permissions_for would compare the map's output to a
        // re-fold of itself and pass. check_no_escalation checks each rule's access
        // against the grant instead, so it catches the over-grant.
        let wide = EscalatingPermissions {
            subject: SubjectType(1),
        };
        let l = lattice(3);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let policy = compile(&u, &l, &wide);

        // The emitted allow-set equals a re-fold of the same permissions_for (a
        // self-check would pass).
        let self_fold: BTreeSet<TeRule> = {
            let mut s = BTreeSet::new();
            for c in u.capabilities() {
                s.extend(wide.permissions_for(c));
            }
            s
        };
        assert_eq!(allow_set(&policy), self_fold, "a self-check would pass this");

        // The authority-anchored check catches it.
        match check_no_escalation(&u, &policy, &wide) {
            Err(PrivilegeEscalation {
                access: Some(a), ..
            }) => {
                let claimed = a.as_capability();
                assert!(
                    !u.capabilities().iter().any(|h| h.authorizes(&claimed)),
                    "the escalation's access must be unauthorized by the grant"
                );
            }
            other => panic!("must catch the escalation; got {other:?}"),
        }
    }

    // ---- from_verified keystone + tamper (S5 → S4 wiring) -------------------

    struct StubSigner {
        key: [u8; 32],
    }
    impl PolicySigner for StubSigner {
        fn sign(&self, input: &[u8]) -> Result<Vec<u8>, PolicyDistError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            Ok(h.finalize().as_bytes().to_vec())
        }
    }
    struct StubVerifier {
        key: [u8; 32],
    }
    impl PolicyVerifier for StubVerifier {
        fn verify(&self, input: &[u8], sig: &[u8]) -> Result<(), PolicyDistError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            if h.finalize().as_bytes().as_slice() == sig {
                Ok(())
            } else {
                Err(PolicyDistError::BadSignature("stub mismatch".into()))
            }
        }
    }

    #[test]
    fn from_verified_keystone_then_loader_accepts() {
        let p = permissions();
        let l = lattice(13);
        let id = identity();
        let u = ucan_signed_by(&id, vec![cap("mac://model/qwen", "model/read")]);
        let (policy, signed) = compile_policy(&u, &l, &p, &id.ed_sk, &id.pq_sk).unwrap();

        let binding = signed.verify(&id.ed_sk.verifying_key(), &id.pq_vk).unwrap();
        let approval = PolicyApproval::from_verified(binding);
        assert_eq!(approval.generation, policy.generation);
        assert_eq!(approval.approved_hash, policy.policy_hash().unwrap());

        let key = [5u8; 32];
        let signed_policy = sign_policy(&policy, &StubSigner { key }).unwrap();
        let loader = PolicyLoader::new(StubVerifier { key }).with_approval(approval);
        let loaded = loader.load(&signed_policy).unwrap();
        assert_eq!(loaded.generation, policy.generation);
    }

    #[test]
    fn tampered_hash_fails_verify_binds_before_loader() {
        let p = permissions();
        let l = lattice(13);
        let id = identity();
        let u = ucan_signed_by(&id, vec![cap("mac://model/qwen", "model/read")]);
        let (policy, signed) = compile_policy(&u, &l, &p, &id.ed_sk, &id.pq_sk).unwrap();

        let mut tampered = policy.policy_hash().unwrap();
        tampered[0] ^= 0xFF;
        let res = signed.verify_binds(
            &id.ed_sk.verifying_key(),
            &id.pq_vk,
            &u,
            policy.generation,
            &tampered,
        );
        assert!(
            matches!(
                res,
                Err(hyprstream_rpc::auth::ucan::approval::ApprovalError::BundleHashMismatch)
            ),
            "a tampered hash must fail verify_binds before the loader"
        );
    }

    // ---- differential property tests: matrix evaluation ≡ authorize ---------

    /// Adversarial capabilities: exact, wildcard, near-misses, and an unmapped
    /// namespace.
    fn arb_capability() -> impl Strategy<Value = Capability> {
        let resources = prop_oneof![
            Just("mac://model/qwen"),
            Just("mac://model/llama"),
            Just("mac://model/mistral"),
            Just("mac://model/*"),
            Just("mac://model/qwen2"), // superstring of "qwen"
            Just("mac://model/qwe"),   // non-boundary prefix of "qwen"
            Just("mac://model"),       // bare prefix
            Just("mac://other/x"),     // unmapped namespace
        ];
        let abilities = prop_oneof![
            Just("model/read"),
            Just("model/write"),
            Just("model/rea"), // near-miss prefix of "model/read"
            Just("model"),     // bare namespace
            Just("admin/x"),   // unmapped verb
        ];
        (resources, abilities).prop_map(|(r, a)| cap(r, a))
    }

    fn arb_grant() -> impl Strategy<Value = Vec<Capability>> {
        prop::collection::vec(arb_capability(), 0..5)
    }

    /// Adversarial requests: mapped objects/verbs plus near-misses the matrix must
    /// not permit unless a grant capability truly authorizes them.
    fn arb_request() -> impl Strategy<Value = AccessRequest> {
        let resources = prop_oneof![
            Just("mac://model/qwen"),
            Just("mac://model/llama"),
            Just("mac://model/mistral"),
            Just("mac://model/qwen2"),
            Just("mac://model/qwe"),
            Just("mac://other/z"),
        ];
        let abilities = prop_oneof![
            Just("model/read"),
            Just("model/write"),
            Just("model/rea"),
            Just("admin/y"),
        ];
        (resources, abilities).prop_map(|(r, a)| AccessRequest {
            resource: Resource::new(r),
            ability: Ability::new(a),
            caveats: Caveats::empty(),
        })
    }

    /// Is the request in the map's recognized domain (concrete object AND verb)?
    /// Uses the inverse id maps, not permissions_for.
    fn is_mapped(req: &AccessRequest) -> bool {
        let obj_ok = req
            .resource
            .as_str()
            .strip_prefix("mac://model/")
            .is_some_and(|name| OBJECTS.contains(&name));
        let verb_ok = action_id(req.ability.as_str()).is_some();
        obj_ok && verb_ok
    }

    /// Decide a request against the emitted matrix directly: permit iff some emitted
    /// rule's granted_access authorizes it. Evaluates the compiled matrix the way a
    /// PEP would, then maps each granted rule back through `authorizes` — no call to
    /// permissions_for.
    fn matrix_decides(p: &TestPermissions, matrix: &TeMatrix, req: &AccessRequest) -> Decision {
        let claimed = req.as_capability();
        let (allow, _escalate) = matrix.sorted_rules();
        for rule in &allow {
            if matrix.te_decision(*rule) != Decision::Permit {
                continue;
            }
            if let Some(a) = p.granted_access(rule) {
                if a.as_capability().authorizes(&claimed) {
                    return Decision::Permit;
                }
            }
        }
        Decision::Deny
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// The matrix evaluation agrees with `authorize` on every adversarial
        /// request, and the policy never escalates for any grant.
        #[test]
        fn matrix_agrees_with_authorize(
            grant in arb_grant(),
            req in arb_request(),
        ) {
            let p = permissions();
            let l = lattice(1);
            let u = ucan_with(grant.clone());
            let policy = compile(&u, &l, &p);

            prop_assert_eq!(check_no_escalation(&u, &policy, &p), Ok(()));

            let reference = authorize(u.capabilities(), &req);
            let matrix = matrix_decides(&p, &policy.matrix, &req);

            // Security direction over the full space: the matrix never permits what
            // the reference denies.
            if matrix == Decision::Permit {
                prop_assert_eq!(reference, Decision::Permit,
                    "matrix permitted what authorize denies (escalation) on {:?}", req);
            }

            // Full agreement holds on the mapped domain; outside it the matrix is
            // intentionally silent (default-deny) — a safe under-grant.
            if is_mapped(&req) {
                prop_assert_eq!(matrix, reference,
                    "on the mapped domain, matrix and authorize must agree on {:?} for grant {:?}",
                    req, grant);
            }
        }

        /// One-directional no-escalation: the matrix can never permit what
        /// `authorize` denies.
        #[test]
        fn matrix_permit_implies_authorize_permit(
            grant in arb_grant(),
            req in arb_request(),
        ) {
            let p = permissions();
            let l = lattice(1);
            let u = ucan_with(grant);
            let policy = compile(&u, &l, &p);
            if matrix_decides(&p, &policy.matrix, &req) == Decision::Permit {
                prop_assert_eq!(authorize(u.capabilities(), &req), Decision::Permit,
                    "matrix permitted what authorize denies — an escalation");
            }
        }

        /// Temporal: out of the grant's window, `authorize_at` denies even when the
        /// matrix would permit; in window, matrix permit implies authorize permit.
        #[test]
        fn temporal_gate_differential(
            grant in arb_grant(),
            req in arb_request(),
            now in 0u64..400u64,
        ) {
            let p = permissions();
            let l = lattice(1);
            let mut u = ucan_with(grant);
            u.payload.not_before = Some(100);
            u.payload.expiration = Some(200);
            let policy = compile(&u, &l, &p);

            let matrix_permits = matrix_decides(&p, &policy.matrix, &req) == Decision::Permit;
            let at = authorize_at(
                u.capabilities(),
                u.payload.not_before,
                u.payload.expiration,
                &req,
                now,
            );
            let in_window = (100..=200).contains(&now);

            if !in_window {
                prop_assert_eq!(at, Decision::Deny,
                    "out-of-window request must be denied (now={})", now);
            } else if matrix_permits {
                prop_assert_eq!(at, Decision::Permit,
                    "in-window matrix permit must match authorize permit");
            }
        }

        /// Reordering the grant's capabilities yields a byte-identical policy hash.
        #[test]
        fn compile_is_deterministic_under_reordering(
            grant in arb_grant(),
            seed in any::<u64>(),
        ) {
            let p = permissions();
            let l = lattice(1);
            let mut shuffled = grant.clone();
            if shuffled.len() > 1 {
                let n = shuffled.len();
                let k = (seed as usize) % n;
                shuffled.rotate_left(k);
                shuffled.reverse();
            }
            let h1 = compile(&ucan_with(grant), &l, &p).policy_hash().unwrap();
            let h2 = compile(&ucan_with(shuffled), &l, &p).policy_hash().unwrap();
            prop_assert_eq!(h1, h2, "reordering the grant must not change the hash");
        }
    }
}
