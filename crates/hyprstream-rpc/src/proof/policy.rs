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

/// How many, and which, additional enrolled logical signers must approve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApproverRule {
    /// Every named approver role must be present, each satisfied by a distinct
    /// enrolled principal.
    All { roles: Vec<String> },
    /// Any `k` of `n` enrolled approvers, each a distinct principal.
    KOfN { k: usize, n: usize },
    /// `k` distinct enrolled principals holding the named approver role.
    Role { role: String, k: usize },
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
    // Distinctness of principals is already enforced during verification; this
    // re-checks it here so the threshold can never be met by one principal
    // counted twice, whatever produced the verified set.
    let mut principals: Vec<&str> = verified
        .approvers
        .iter()
        .map(|a| a.principal.as_str())
        .collect();
    principals.sort_unstable();
    let distinct = principals.len();
    principals.dedup();
    if principals.len() != distinct {
        bail!("approver groups do not resolve to distinct principals");
    }

    match rule {
        ApproverRule::All { roles } => {
            for role in roles {
                if !verified
                    .approvers
                    .iter()
                    .any(|a| a.role.as_deref() == Some(role.as_str()))
                {
                    bail!("approver rule requires role '{role}', which no enrolled approver holds");
                }
            }
            if verified.approvers.len() != roles.len() {
                bail!(
                    "approver rule requires exactly {} approver(s), proof carries {}",
                    roles.len(),
                    verified.approvers.len()
                );
            }
            Ok(())
        }
        ApproverRule::KOfN { k, n } => {
            if verified.approvers.len() > *n {
                bail!(
                    "approver rule permits at most {n} approver(s), proof carries {}",
                    verified.approvers.len()
                );
            }
            if verified.approvers.len() < *k {
                bail!(
                    "approver rule requires {k} distinct approver(s), proof carries {}",
                    verified.approvers.len()
                );
            }
            Ok(())
        }
        ApproverRule::Role { role, k } => {
            let holding = verified
                .approvers
                .iter()
                .filter(|a| a.role.as_deref() == Some(role.as_str()))
                .count();
            if holding < *k {
                bail!(
                    "approver rule requires {k} approver(s) holding role '{role}', proof carries {holding}"
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

    fn verified(suite: CryptoSuite, approvers: &[(&str, Option<&str>)]) -> VerifiedProof {
        VerifiedProof {
            replay_thumbprint: [0u8; 32],
            primary_principal: Some("client".into()),
            primary_suite: suite.suite_id().to_owned(),
            approvers: approvers
                .iter()
                .map(|(p, r)| VerifiedApprover {
                    principal: (*p).to_owned(),
                    role: r.map(ToOwned::to_owned),
                })
                .collect(),
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
        let v = verified(CryptoSuite::Classical, &[("approver", Some("security"))]);
        let bound = SignaturePolicy::TokenBound {
            suite: CryptoSuite::Classical,
        };
        assert!(evaluate(&bound, ProofDisposition::Authenticated, &v).is_err());
    }

    #[test]
    fn k_of_n_requires_k_distinct_approvers_and_permits_at_most_n() {
        let rule = ApproverRule::KOfN { k: 2, n: 3 };
        let policy = |rule: ApproverRule| SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: rule,
        };
        let one = verified(CryptoSuite::Classical, &[("a", None)]);
        let two = verified(CryptoSuite::Classical, &[("a", None), ("b", None)]);
        let four = verified(
            CryptoSuite::Classical,
            &[("a", None), ("b", None), ("c", None), ("d", None)],
        );
        assert!(evaluate(&policy(rule.clone()), ProofDisposition::Authenticated, &one).is_err());
        assert!(evaluate(&policy(rule.clone()), ProofDisposition::Authenticated, &two).is_ok());
        assert!(evaluate(&policy(rule), ProofDisposition::Authenticated, &four).is_err());
    }

    /// One principal counted twice can never satisfy a threshold.
    #[test]
    fn a_repeated_principal_cannot_satisfy_a_threshold() {
        let policy = SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: ApproverRule::KOfN { k: 2, n: 3 },
        };
        let duplicated = verified(CryptoSuite::Classical, &[("a", None), ("a", None)]);
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &duplicated).is_err());
    }

    #[test]
    fn role_rules_count_only_approvers_holding_that_role() {
        let policy = SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: ApproverRule::Role {
                role: "security".into(),
                k: 2,
            },
        };
        let wrong_role = verified(
            CryptoSuite::Classical,
            &[("a", Some("finance")), ("b", Some("finance"))],
        );
        let one_right = verified(
            CryptoSuite::Classical,
            &[("a", Some("security")), ("b", Some("finance"))],
        );
        let both = verified(
            CryptoSuite::Classical,
            &[("a", Some("security")), ("b", Some("security"))],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &wrong_role).is_err());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &one_right).is_err());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &both).is_ok());
    }

    #[test]
    fn all_rules_require_every_named_role_and_no_extras() {
        let policy = SignaturePolicy::TokenBoundAndApproved {
            primary_suite: CryptoSuite::Classical,
            approver_rule: ApproverRule::All {
                roles: vec!["security".into(), "finance".into()],
            },
        };
        let missing = verified(CryptoSuite::Classical, &[("a", Some("security"))]);
        let complete = verified(
            CryptoSuite::Classical,
            &[("a", Some("security")), ("b", Some("finance"))],
        );
        let extra = verified(
            CryptoSuite::Classical,
            &[
                ("a", Some("security")),
                ("b", Some("finance")),
                ("c", Some("legal")),
            ],
        );
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &missing).is_err());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &complete).is_ok());
        assert!(evaluate(&policy, ProofDisposition::Authenticated, &extra).is_err());
    }
}
