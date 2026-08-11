//! Generated per-method signature policy (§4.4, §5.2).
//!
//! Enrollment equivalence is not authorization. An enrolled signer proves who
//! signed; the **method's** generated policy decides what signing topology that
//! method requires: whether it may be called with no credential at all, which
//! cryptographic suite the primary logical signer must use, and whether
//! additional enrolled approvers are required.
//!
//! The policy is a property of the decoded leaf, not of the caller. It is
//! resolved after the single Cap'n Proto decode and evaluated before replay
//! admission, so a request that fails it never consumes store capacity.
//!
//! A leaf with no policy row is **unlisted** and denies. That is the same rule
//! the design gives for a leaf that exists only in a newer client schema: an
//! unlisted `(service, leaf path)` denies, never falls back to a coarser row.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{bail, Result};

use super::{verify::VerifiedProof, ProofDisposition, SUITE_CLASSICAL, SUITE_HYBRID};

/// The cryptographic suite a method requires of its primary logical signer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoSuite {
    /// Standalone Ed25519 (`hs-cose-sign-ed25519-v1`).
    Classical,
    /// Weakly-non-separable Ed25519 + ML-DSA-65
    /// (`hs-cose-sign-ed25519-mldsa65-wns-v1`).
    Hybrid,
}

impl CryptoSuite {
    /// The exact suite ID a signed plan must declare for this suite.
    pub fn suite_id(self) -> &'static str {
        match self {
            Self::Classical => SUITE_CLASSICAL,
            Self::Hybrid => SUITE_HYBRID,
        }
    }
}

/// One approver group the generated policy allows, pinned exactly.
///
/// A group is identified by the signed plan's logical signer group ID, and
/// must present the suite and hold the enrolled approver role this row names.
/// Nothing outside this set may appear in a proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedApproverGroup {
    /// The logical signer group ID this approver must occupy in the signed plan.
    pub group_id: u64,
    /// The exact suite this group must declare and verify under.
    pub suite: CryptoSuite,
    /// The enrolled approver role this group must hold.
    pub role: String,
}

/// How many, and which, additional enrolled logical signers must approve.
///
/// Every variant carries the complete allowed set: an approver group whose ID
/// is not listed, or whose verified suite or enrolled role does not match its
/// row, denies before any threshold is counted. A threshold can therefore
/// never be satisfied — or accompanied — by a group the method did not name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApproverRule {
    /// Every listed group must be present.
    All { groups: Vec<AllowedApproverGroup> },
    /// Any `k` of the listed groups must be present.
    KOfN {
        k: usize,
        groups: Vec<AllowedApproverGroup>,
    },
    /// `k` of the listed groups holding the named role must be present.
    Role {
        role: String,
        k: usize,
        groups: Vec<AllowedApproverGroup>,
    },
}

impl ApproverRule {
    /// The complete set of approver groups this rule allows.
    pub fn allowed_groups(&self) -> &[AllowedApproverGroup] {
        match self {
            Self::All { groups } | Self::KOfN { groups, .. } | Self::Role { groups, .. } => groups,
        }
    }
}

/// The signing topology a method requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignaturePolicy {
    /// Legal only for a `$dispatchPublic` method. With no credential the proof
    /// key set receives no identity, role, clearance, or assurance. With a
    /// credential presented, the proof MUST instead be credential-bound —
    /// presenting a token never leaves the proof in the unattributed branch.
    UnauthenticatedOrTokenBound { suite: CryptoSuite },
    /// The primary logical signer's verified suite keys must match the
    /// credential `cnf`-resolved signer-suite record.
    TokenBound { suite: CryptoSuite },
    /// As `TokenBound`, plus distinct enrolled approvers satisfying the rule.
    TokenBoundAndApproved {
        primary_suite: CryptoSuite,
        approver_rule: ApproverRule,
    },
}

impl SignaturePolicy {
    /// The suite required of the primary logical signer.
    pub fn primary_suite(&self) -> CryptoSuite {
        match self {
            Self::UnauthenticatedOrTokenBound { suite } | Self::TokenBound { suite } => *suite,
            Self::TokenBoundAndApproved { primary_suite, .. } => *primary_suite,
        }
    }

    /// Whether this method may be dispatched with no credential at all.
    pub fn allows_unattributed(&self) -> bool {
        matches!(self, Self::UnauthenticatedOrTokenBound { .. })
    }
}

/// Resolves the generated policy row for one decoded leaf.
///
/// Rows are produced by code generation from the service schema; this trait is
/// how the generated table reaches dispatch. There is no permissive default
/// and no wildcard fallback: a leaf the table does not list denies.
pub trait DispatchMethodPolicy: Send + Sync {
    fn policy_for(&self, service_domain: &str, leaf_path: &str) -> Option<SignaturePolicy>;
}

/// An explicit policy table keyed by `(service domain, leaf path)`.
#[derive(Default)]
pub struct InMemoryMethodPolicy {
    rows: HashMap<(String, String), SignaturePolicy>,
}

impl InMemoryMethodPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        service_domain: &str,
        leaf_path: &str,
        policy: SignaturePolicy,
    ) -> &mut Self {
        self.rows.insert(
            (service_domain.to_owned(), leaf_path.to_owned()),
            policy,
        );
        self
    }
}

impl DispatchMethodPolicy for InMemoryMethodPolicy {
    fn policy_for(&self, service_domain: &str, leaf_path: &str) -> Option<SignaturePolicy> {
        self.rows
            .get(&(service_domain.to_owned(), leaf_path.to_owned()))
            .cloned()
    }
}

static METHOD_POLICY: OnceLock<Box<dyn DispatchMethodPolicy>> = OnceLock::new();

pub fn set_global_method_policy(
    table: Box<dyn DispatchMethodPolicy>,
) -> std::result::Result<(), Box<dyn DispatchMethodPolicy>> {
    METHOD_POLICY.set(table)
}

pub fn global_method_policy() -> Option<&'static dyn DispatchMethodPolicy> {
    METHOD_POLICY.get().map(|t| &**t)
}

/// Evaluate a resolved policy against a verified proof.
///
/// Called after signature verification, so the disposition, the primary
/// suite, and the approver set are all facts established by cryptography and
/// enrollment rather than claims read off the wire.
pub fn evaluate(
    policy: &SignaturePolicy,
    disposition: ProofDisposition,
    verified: &VerifiedProof,
) -> Result<()> {
    // 1. Disposition. Only a public method may be dispatched unattributed.
    if disposition == ProofDisposition::Unattributed && !policy.allows_unattributed() {
        bail!("method requires a credential-bound proof; unattributed proof denied");
    }

    // 2. The primary logical signer's suite must be exactly the one the method
    //    declares — a standalone proof cannot satisfy a hybrid method, and a
    //    hybrid proof cannot be presented where a standalone suite is declared.
    let required = policy.primary_suite().suite_id();
    if verified.primary_suite != required {
        bail!(
            "method requires primary suite '{}', proof declares '{}'",
            required,
            verified.primary_suite
        );
    }

    // 3. Approvers.
    match policy {
        SignaturePolicy::UnauthenticatedOrTokenBound { .. } | SignaturePolicy::TokenBound { .. } => {
            if !verified.approvers.is_empty() {
                bail!(
                    "method declares no approver rule but the proof carries {} approver group(s)",
                    verified.approvers.len()
                );
            }
            Ok(())
        }
        SignaturePolicy::TokenBoundAndApproved { approver_rule, .. } => {
            evaluate_approvers(approver_rule, verified)
        }
    }
}

fn evaluate_approvers(rule: &ApproverRule, verified: &VerifiedProof) -> Result<()> {
    let allowed = rule.allowed_groups();

    // 1. Every group the proof carries must be one this method named, matched
    //    by its signed logical group ID, and must present exactly the suite
    //    and enrolled role that row pins. An unknown or mismatched group is a
    //    denial, not a group that merely fails to count.
    for approver in &verified.approvers {
        let row = allowed
            .iter()
            .find(|row| row.group_id == approver.group_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "signer group {} is not an approver group this method allows",
                    approver.group_id
                )
            })?;
        if approver.suite != row.suite.suite_id() {
            bail!(
                "approver group {} must use suite '{}', proof declares '{}'",
                approver.group_id,
                row.suite.suite_id(),
                approver.suite
            );
        }
        if approver.role.as_deref() != Some(row.role.as_str()) {
            bail!(
                "approver group {} must be enrolled with role '{}', enrollment holds {:?}",
                approver.group_id,
                row.role,
                approver.role
            );
        }
    }

    // 2. No group may be presented twice, and distinct groups must resolve to
    //    distinct principals — one holder can never satisfy two approvals.
    let mut group_ids: Vec<u64> = verified.approvers.iter().map(|a| a.group_id).collect();
    let presented = group_ids.len();
    group_ids.sort_unstable();
    group_ids.dedup();
    if group_ids.len() != presented {
        bail!("the same approver group is presented more than once");
    }
    let mut principals: Vec<&str> = verified
        .approvers
        .iter()
        .map(|a| a.principal.as_str())
        .collect();
    principals.sort_unstable();
    principals.dedup();
    if principals.len() != presented {
        bail!("approver groups do not resolve to distinct principals");
    }

    // 3. Threshold, counted only over groups that passed (1).
    match rule {
        ApproverRule::All { groups } => {
            for row in groups {
                if !verified
                    .approvers
                    .iter()
                    .any(|a| a.group_id == row.group_id)
                {
                    bail!(
                        "approver rule requires group {} (role '{}'), which did not sign",
                        row.group_id,
                        row.role
                    );
                }
            }
            Ok(())
        }
        ApproverRule::KOfN { k, .. } => {
            if presented < *k {
                bail!("approver rule requires {k} of the allowed groups, proof carries {presented}");
            }
            Ok(())
        }
        ApproverRule::Role { role, k, groups } => {
            let holding = verified
                .approvers
                .iter()
                .filter(|a| {
                    groups
                        .iter()
                        .any(|row| row.group_id == a.group_id && row.role == *role)
                })
                .count();
            if holding < *k {
                bail!(
                    "approver rule requires {k} allowed group(s) holding role '{role}', proof carries {holding}"
                );
            }
            Ok(())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::verify::VerifiedApprover;
    use super::*;

    /// Approvers as (group_id, principal, role), all on the classical suite
    /// unless a test overrides it.
    fn verified(suite: CryptoSuite, approvers: &[(u64, &str, Option<&str>)]) -> VerifiedProof {
        verified_with_suites(
            suite,
            &approvers
                .iter()
                .map(|(g, p, r)| (*g, CryptoSuite::Classical, *p, *r))
                .collect::<Vec<_>>(),
        )
    }

    fn verified_with_suites(
        suite: CryptoSuite,
        approvers: &[(u64, CryptoSuite, &str, Option<&str>)],
    ) -> VerifiedProof {
        VerifiedProof {
            replay_thumbprint: [0u8; 32],
            primary_principal: Some("client".into()),
            primary_suite: suite.suite_id().to_owned(),
            approvers: approvers
                .iter()
                .map(|(group_id, group_suite, p, r)| VerifiedApprover {
                    group_id: *group_id,
                    suite: group_suite.suite_id().to_owned(),
                    principal: (*p).to_owned(),
                    role: r.map(ToOwned::to_owned),
                })
                .collect(),
        }
    }

    fn group(group_id: u64, role: &str) -> AllowedApproverGroup {
        AllowedApproverGroup {
            group_id,
            suite: CryptoSuite::Classical,
            role: role.to_owned(),
        }
    }

    #[test]
    fn an_unlisted_leaf_has_no_policy() {
        let table = InMemoryMethodPolicy::new();
        assert!(table.policy_for("registry.svc", "create").is_none());
    }

    #[test]
    fn a_policy_row_is_exact_and_never_wildcarded() {
        let mut table = InMemoryMethodPolicy::new();
        table.insert(
            "registry.svc",
            "create",
            SignaturePolicy::TokenBound {
                suite: CryptoSuite::Hybrid,
            },
        );
        assert!(table.policy_for("registry.svc", "create").is_some());
        assert!(table.policy_for("registry.svc", "delete").is_none());
        assert!(table.policy_for("other.svc", "create").is_none());
        assert!(table.policy_for("registry.svc", "*").is_none());
    }

    #[test]
    fn only_a_public_method_accepts_an_unattributed_proof() {
        let v = verified(CryptoSuite::Classical, &[]);
        let public = SignaturePolicy::UnauthenticatedOrTokenBound {
            suite: CryptoSuite::Classical,
        };
        let bound = SignaturePolicy::TokenBound {
            suite: CryptoSuite::Classical,
        };
        assert!(evaluate(&public, ProofDisposition::Unattributed, &v).is_ok());
        assert!(evaluate(&bound, ProofDisposition::Unattributed, &v).is_err());
        assert!(evaluate(&bound, ProofDisposition::Authenticated, &v).is_ok());
    }

    #[test]
    fn the_primary_suite_must_match_exactly() {
        let classical = verified(CryptoSuite::Classical, &[]);
        let hybrid = verified(CryptoSuite::Hybrid, &[]);
        let needs_hybrid = SignaturePolicy::TokenBound {
            suite: CryptoSuite::Hybrid,
        };
        let needs_classical = SignaturePolicy::TokenBound {
            suite: CryptoSuite::Classical,
        };
        assert!(evaluate(&needs_hybrid, ProofDisposition::Authenticated, &classical).is_err());
        assert!(evaluate(&needs_classical, ProofDisposition::Authenticated, &hybrid).is_err());
        assert!(evaluate(&needs_hybrid, ProofDisposition::Authenticated, &hybrid).is_ok());
    }

    /// Extra approvals on a method that declares no approver rule are not
    /// harmless: an unrecognized signature group denies.
    #[test]
    fn unexpected_approvers_deny() {
        let v = verified(CryptoSuite::Classical, &[(2, "approver", Some("security"))]);
        let bound = SignaturePolicy::TokenBound {
            suite: CryptoSuite::Classical,
        };
        assert!(evaluate(&bound, ProofDisposition::Authenticated, &v).is_err());
    }

    fn approved(rule: ApproverRule) -> SignaturePolicy {
        SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: rule,
        }
    }

    /// A group the method never named cannot satisfy a threshold, and cannot
    /// ride along beside groups that do.
    #[test]
    fn an_unnamed_approver_group_denies() {
        let policy = approved(ApproverRule::KOfN {
            k: 1,
            groups: vec![group(2, "security")],
        });
        let named = verified(CryptoSuite::Classical, &[(2, "a", Some("security"))]);
        let unnamed = verified(CryptoSuite::Classical, &[(3, "b", Some("security"))]);
        let both = verified(
            CryptoSuite::Classical,
            &[(2, "a", Some("security")), (3, "b", Some("security"))],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &named).is_ok());
        assert!(
            evaluate(&policy, ProofDisposition::Authenticated, &unnamed).is_err(),
            "an unnamed group must not satisfy the threshold"
        );
        assert!(
            evaluate(&policy, ProofDisposition::Authenticated, &both).is_err(),
            "an unnamed group must not merely accompany a satisfied threshold"
        );
    }

    /// Each allowed group is pinned to its own suite: a group signing under a
    /// different suite denies even though its ID and role match.
    #[test]
    fn an_approver_group_under_the_wrong_suite_denies() {
        let policy = approved(ApproverRule::KOfN {
            k: 1,
            groups: vec![group(2, "security")],
        });
        let wrong_suite = verified_with_suites(
            CryptoSuite::Classical,
            &[(2, CryptoSuite::Hybrid, "a", Some("security"))],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &wrong_suite).is_err());
    }

    /// A group present under an enrolled role the method did not pin for that
    /// group denies — roles are per-group, not a global pool.
    #[test]
    fn an_approver_group_with_the_wrong_enrolled_role_denies() {
        let policy = approved(ApproverRule::KOfN {
            k: 1,
            groups: vec![group(2, "security")],
        });
        for role in [Some("finance"), None] {
            let v = verified(CryptoSuite::Classical, &[(2, "a", role)]);
            assert!(evaluate(&policy, ProofDisposition::Authenticated, &v).is_err());
        }
    }

    #[test]
    fn k_of_n_counts_only_allowed_groups() {
        let policy = approved(ApproverRule::KOfN {
            k: 2,
            groups: vec![group(2, "security"), group(3, "finance")],
        });
        let one = verified(CryptoSuite::Classical, &[(2, "a", Some("security"))]);
        let two = verified(
            CryptoSuite::Classical,
            &[(2, "a", Some("security")), (3, "b", Some("finance"))],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &one).is_err());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &two).is_ok());
    }

    /// One principal counted twice can never satisfy a threshold, whether it
    /// arrives as one group repeated or as two allowed groups.
    #[test]
    fn a_repeated_group_or_principal_cannot_satisfy_a_threshold() {
        let policy = approved(ApproverRule::KOfN {
            k: 2,
            groups: vec![group(2, "security"), group(3, "finance")],
        });
        let repeated_group = verified(
            CryptoSuite::Classical,
            &[(2, "a", Some("security")), (2, "b", Some("security"))],
        );
        let repeated_principal = verified(
            CryptoSuite::Classical,
            &[(2, "a", Some("security")), (3, "a", Some("finance"))],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &repeated_group).is_err());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &repeated_principal).is_err());
    }

    #[test]
    fn role_rules_count_only_allowed_groups_holding_that_role() {
        let policy = approved(ApproverRule::Role {
            role: "security".into(),
            k: 2,
            groups: vec![
                group(2, "security"),
                group(3, "security"),
                group(4, "finance"),
            ],
        });
        let one_security = verified(
            CryptoSuite::Classical,
            &[(2, "a", Some("security")), (4, "b", Some("finance"))],
        );
        let two_security = verified(
            CryptoSuite::Classical,
            &[(2, "a", Some("security")), (3, "b", Some("security"))],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &one_security).is_err());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &two_security).is_ok());
    }

    #[test]
    fn all_rules_require_every_named_group() {
        let policy = approved(ApproverRule::All {
            groups: vec![group(2, "security"), group(3, "finance")],
        });
        let missing = verified(CryptoSuite::Classical, &[(2, "a", Some("security"))]);
        let complete = verified(
            CryptoSuite::Classical,
            &[(2, "a", Some("security")), (3, "b", Some("finance"))],
        );
        let extra = verified(
            CryptoSuite::Classical,
            &[
                (2, "a", Some("security")),
                (3, "b", Some("finance")),
                (5, "c", Some("legal")),
            ],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &missing).is_err());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &complete).is_ok());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &extra).is_err());
    }

}
