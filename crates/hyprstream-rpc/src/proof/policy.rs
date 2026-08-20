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

use crate::auth::mac::{Assurance, CompartmentSet, Level, SecurityLabel};

use super::{verify::VerifiedProof, ProofDisposition, SUITE_CLASSICAL, SUITE_HYBRID};

/// The system-low label — the exact expansion of `$dispatchPublic` and the
/// mandatory value of every row's `transitional_label` during migration
/// (v16 §7.3). Public ⇔ `target_label` is exactly this label.
pub const SYSTEM_LOW_LABEL: SecurityLabel = SecurityLabel {
    level: Level::Public,
    assurance: Assurance::Unverified,
    compartments: CompartmentSet::EMPTY,
};

/// Construct a dispatch label from its parsed axes (v16 §6). This is the ONE
/// constructor generated rows use — there is no permissive default and no
/// string parsing at runtime; the grammar was checked at code generation
/// against the checked-in `InitialLabelMap`.
pub const fn dispatch_label(level: Level, assurance: Assurance, bits: &[u32]) -> SecurityLabel {
    let mut set = CompartmentSet::EMPTY;
    let mut i = 0;
    while i < bits.len() {
        set = set.union(CompartmentSet::single(bits[i]));
        i += 1;
    }
    SecurityLabel {
        level,
        assurance,
        compartments: set,
    }
}

/// The side-effect-free scope actions (S3 `ScopeAction` Block A: read-class).
/// Every other action in the closed vocabulary is mutating.
pub const READ_CLASS_ACTIONS: &[&str] = &["query", "subscribe"];

/// The generated application policy a mutating method declares (v16 §4.8) —
/// distinct from request-proof replay admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationSemantics {
    /// Retrying the method with the same intent is safe without extra
    /// machinery.
    NaturallyIdempotent,
    /// The method's payload carries an application idempotency key; the
    /// idempotency/result ledger returns the recorded result on retry.
    IdempotencyKeyRequired,
    /// Exactly-once-visible behavior is claimed; the mutation commits with
    /// the ledger in one transaction or equivalent fencing protocol.
    TransactionLedgerRequired,
}

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

/// Whether a generated row admits the `Unauthenticated` disposition (v16 §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthenticationRequirement {
    /// The method requires a verified credential; `$scope`-annotated leaves.
    CredentialRequired,
    /// The method is publicly dispatchable (`$scopeExempt` leaves) — legal
    /// only together with an `UnauthenticatedOrTokenBound` signature policy
    /// and a recorded public reason.
    UnauthenticatedAllowed,
}

/// One generated method-policy row (v16 §6.1).
///
/// Rows are emitted by `generate_rpc_service!` from the same leaf-tree walk
/// that produces the service's signed-body decoder, so the decoder can never
/// derive a leaf the inventory does not list and vice versa. They are
/// aggregated across all linked service crates through [`inventory`] and
/// installed once at startup by [`install_generated_method_policy`].
#[derive(Debug, Clone)]
pub struct GeneratedMethodPolicyRow {
    /// Canonical service domain (the dispatcher's `RequestService::name`).
    pub service: &'static str,
    /// Full numeric union-discriminant chain from the service root union.
    pub leaf_path: &'static [u16],
    /// Dotted human-readable path — review metadata, never a lookup key.
    pub symbolic_path: &'static str,
    /// `$scope`/`$capability` action; empty for `$dispatchPublic` leaves and
    /// for `$dispatchMac` leaves whose control-plane scope is explicitly
    /// exempted (`$scopeExempt` — the gate is a different mechanism, e.g. a
    /// CA-signed JWT attestation, documented in the schema).
    pub scope_action: &'static str,
    /// Whether the leaf's control-plane scope is `$scopeExempt`-exempted. The
    /// DISPATCH floor is unaffected (a `$scopeExempt` + `$dispatchMac` leaf
    /// still requires a credential); this records why `scope_action` is empty
    /// on a credential-required row.
    pub scope_exempt: bool,
    /// Whether the leaf admits the `Unauthenticated` disposition.
    pub authentication: AuthenticationRequirement,
    /// The signing topology the leaf requires.
    pub signature_policy: SignaturePolicy,
    /// The application mutation policy a mutating leaf declares (v16 §4.8);
    /// `None` exactly for read-class (side-effect-free) scope actions.
    pub mutation_semantics: Option<MutationSemantics>,
    /// The migration label: always system low while the transitional column
    /// is selected (v16 §7.3); deleted after the target flip.
    pub transitional_label: SecurityLabel,
    /// The operator-reviewable target label parsed from `$dispatchMac`, or
    /// exactly system low for a `$dispatchPublic` leaf.
    pub target_label: SecurityLabel,
    /// The audited reason a publicly dispatchable leaf is public.
    pub public_reason: Option<&'static str>,
}

impl GeneratedMethodPolicyRow {
    /// The dotted numeric leaf key the policy table is resolved with.
    pub fn leaf_key(&self) -> String {
        self.leaf_path
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// Inventory registration submitted by every server-side generated module.
pub struct GeneratedMethodPolicyProvider {
    /// The schema/service name the rows belong to.
    pub service: &'static str,
    /// Builder for the service's complete generated row set.
    pub rows_fn: fn() -> Vec<GeneratedMethodPolicyRow>,
}

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(GeneratedMethodPolicyProvider);

/// Validate one complete generated row set (v16 §6.1 build gates).
///
/// Every violation here is a **build error** — this function runs both in a
/// permanent unit test over the full linked inventory (so CI fails when a
/// schema change produces an invalid inventory) and at startup installation
/// (so a production process refuses to serve under an invalid table rather
/// than serving a partial one).
pub fn validate_generated_rows(rows: &[GeneratedMethodPolicyRow]) -> Result<()> {
    use std::collections::HashSet;
    let mut leaf_keys: HashSet<(&str, String)> = HashSet::new();
    let mut symbolic: HashSet<(&str, &str)> = HashSet::new();

    for row in rows {
        if row.leaf_path.is_empty() {
            bail!(
                "generated policy row '{}':'{}' has an empty leaf path",
                row.service,
                row.symbolic_path
            );
        }
        if row.symbolic_path.is_empty() {
            bail!("generated policy row '{}' has an empty symbolic path", row.service);
        }
        if !leaf_keys.insert((row.service, row.leaf_key())) {
            bail!(
                "generated policy collision: duplicate leaf '{}':'{}' ({})",
                row.service,
                row.leaf_key(),
                row.symbolic_path
            );
        }
        if !symbolic.insert((row.service, row.symbolic_path)) {
            bail!(
                "generated policy symbolic-name drift: duplicate '{}':'{}'",
                row.service,
                row.symbolic_path
            );
        }

        // A publicly dispatchable row and an identity-only signer policy are
        // contradictory in both directions (v16 §6.1).
        match row.authentication {
            AuthenticationRequirement::UnauthenticatedAllowed => {
                if !row.signature_policy.allows_unattributed() {
                    bail!(
                        "public leaf '{}':'{}' carries an identity-only signature policy",
                        row.service,
                        row.symbolic_path
                    );
                }
                let reason_ok = row
                    .public_reason
                    .map(|r| !r.trim().is_empty())
                    .unwrap_or(false);
                if !reason_ok {
                    bail!(
                        "public leaf '{}':'{}' has no recorded public reason",
                        row.service,
                        row.symbolic_path
                    );
                }
                if !row.scope_action.is_empty() {
                    bail!(
                        "public leaf '{}':'{}' also declares scope action '{}'",
                        row.service,
                        row.symbolic_path,
                        row.scope_action
                    );
                }
                // `$dispatchPublic` expands to exactly system low (v16 §6) —
                // anything else on a public row is a generation defect.
                if row.target_label != SYSTEM_LOW_LABEL {
                    bail!(
                        "public leaf '{}':'{}' target label {} is not exactly system low",
                        row.service,
                        row.symbolic_path,
                        row.target_label
                    );
                }
            }
            AuthenticationRequirement::CredentialRequired => {
                if row.signature_policy.allows_unattributed() {
                    bail!(
                        "credential-required leaf '{}':'{}' carries an unauthenticated-capable signature policy",
                        row.service,
                        row.symbolic_path
                    );
                }
                // An empty scope action on a credential-required row is legal
                // only as a recorded control-plane exemption ($scopeExempt:
                // the leaf is gated by a different, documented mechanism such
                // as a CA-signed JWT attestation — the dispatch floor still
                // requires the credential).
                if row.scope_action.is_empty() && !row.scope_exempt {
                    bail!(
                        "credential-required leaf '{}':'{}' has no scope action and no recorded \
                         control-plane exemption",
                        row.service,
                        row.symbolic_path
                    );
                }
                if !row.scope_action.is_empty() && row.scope_exempt {
                    bail!(
                        "leaf '{}':'{}' is scope-exempt but carries scope action '{}'",
                        row.service,
                        row.symbolic_path,
                        row.scope_action
                    );
                }
                // A MAC'd leaf whose target is exactly system low would be
                // indistinguishable from a public row — the label grammar
                // rejects this at codegen; this is the generated-row floor.
                if row.target_label == SYSTEM_LOW_LABEL {
                    bail!(
                        "credential-required leaf '{}':'{}' has the system-low target label — \
                         system low through the MAC path is a build error (v16 §6)",
                        row.service,
                        row.symbolic_path
                    );
                }
                // A public reason belongs to the public branch only.
                if row.public_reason.is_some() {
                    bail!(
                        "credential-required leaf '{}':'{}' carries a public reason",
                        row.service,
                        row.symbolic_path
                    );
                }
            }
        }

        // The transitional column is system low for every row while migration
        // runs (v16 §7.3) — a generated row carrying anything else is a
        // generation defect, never an operator choice.
        if row.transitional_label != SYSTEM_LOW_LABEL {
            bail!(
                "row '{}':'{}' transitional label {} is not system low — the \
                 transitional column is fixed until the target flip (v16 §7.3)",
                row.service,
                row.symbolic_path,
                row.transitional_label
            );
        }

        // Mutation consistency (v16 §6.1): a mutating scope action requires an
        // explicit `MutationSemantics`; a read-class action must not claim one.
        // The gate keys off the checked `ScopeAction` block structure, never a
        // method name. A public row carries no scope action and is reviewed
        // through its public reason instead.
        if !row.scope_action.is_empty() {
            let is_read_class = READ_CLASS_ACTIONS.contains(&row.scope_action);
            match (is_read_class, row.mutation_semantics) {
                (true, Some(_)) => bail!(
                    "read-class leaf '{}':'{}' (scope '{}') declares mutation semantics — \
                     read-class actions are side-effect-free",
                    row.service,
                    row.symbolic_path,
                    row.scope_action
                ),
                (false, None) => bail!(
                    "mutating leaf '{}':'{}' (scope '{}') has no MutationSemantics — a \
                     mutating scope action requires an explicit one (v16 §6.1)",
                    row.service,
                    row.symbolic_path,
                    row.scope_action
                ),
                _ => {}
            }
        }

        // Approver rules must be satisfiable and exact.
        if let SignaturePolicy::TokenBoundAndApproved { approver_rule, .. } =
            &row.signature_policy
        {
            validate_approver_rule(row, approver_rule)?;
        }
    }
    Ok(())
}

fn validate_approver_rule(row: &GeneratedMethodPolicyRow, rule: &ApproverRule) -> Result<()> {
    let groups = rule.allowed_groups();
    if groups.is_empty() {
        bail!(
            "approved leaf '{}':'{}' names no allowed approver groups",
            row.service,
            row.symbolic_path
        );
    }
    let mut ids: Vec<u64> = groups.iter().map(|g| g.group_id).collect();
    let count = ids.len();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != count {
        bail!(
            "approved leaf '{}':'{}' names a duplicate approver group ID",
            row.service,
            row.symbolic_path
        );
    }
    for group in groups {
        if group.role.trim().is_empty() {
            bail!(
                "approved leaf '{}':'{}' names approver group {} with an empty role",
                row.service,
                row.symbolic_path,
                group.group_id
            );
        }
    }
    match rule {
        ApproverRule::All { .. } => Ok(()),
        ApproverRule::KOfN { k, groups } => {
            if *k == 0 || *k > groups.len() {
                bail!(
                    "approved leaf '{}':'{}' has an unsatisfiable threshold {k} of {}",
                    row.service,
                    row.symbolic_path,
                    groups.len()
                );
            }
            Ok(())
        }
        ApproverRule::Role { role, k, groups } => {
            if role.trim().is_empty() {
                bail!(
                    "approved leaf '{}':'{}' has a role rule with an empty role",
                    row.service,
                    row.symbolic_path
                );
            }
            let holding = groups.iter().filter(|g| g.role == *role).count();
            if *k == 0 || *k > holding {
                bail!(
                    "approved leaf '{}':'{}' requires {k} group(s) holding role '{role}' but names {holding}",
                    row.service,
                    row.symbolic_path
                );
            }
            Ok(())
        }
    }
}

/// Collect the complete generated inventory, deterministically sorted by
/// `(service, numeric leaf path)`.
#[cfg(not(target_arch = "wasm32"))]
pub fn collect_generated_rows() -> Result<Vec<GeneratedMethodPolicyRow>> {
    let mut rows: Vec<GeneratedMethodPolicyRow> = Vec::new();
    for provider in inventory::iter::<GeneratedMethodPolicyProvider> {
        let provided = (provider.rows_fn)();
        for row in &provided {
            if row.service != provider.service {
                bail!(
                    "generated policy provider '{}' emitted a row for service '{}'",
                    provider.service,
                    row.service
                );
            }
        }
        rows.extend(provided);
    }
    rows.sort_by(|a, b| {
        a.service
            .cmp(b.service)
            .then_with(|| a.leaf_path.cmp(b.leaf_path))
    });
    Ok(rows)
}

/// Build and validate the complete generated method-policy table.
#[cfg(not(target_arch = "wasm32"))]
pub fn build_generated_method_policy() -> Result<(InMemoryMethodPolicy, usize)> {
    let rows = collect_generated_rows()?;
    if rows.is_empty() {
        bail!("no generated method-policy rows are linked into this binary");
    }
    validate_generated_rows(&rows)?;
    let mut table = InMemoryMethodPolicy::new();
    let count = rows.len();
    for row in rows {
        table.insert(row.service, &row.leaf_key(), row.signature_policy);
    }
    Ok((table, count))
}

/// Install the complete generated method-policy inventory as the process
/// policy table (v16 §6.1). Returns the number of installed rows.
///
/// Fails — leaving proof-bearing dispatch fail-closed with **no** table —
/// when the inventory is empty, inconsistent, colliding, or contradictory,
/// or when a table was already installed.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_generated_method_policy() -> Result<usize> {
    let (table, count) = build_generated_method_policy()?;
    set_global_method_policy(Box::new(table))
        .map_err(|_| anyhow::anyhow!("a global method-policy table is already installed"))?;
    Ok(count)
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

    // ── Generated-inventory validation: seeded negative controls ─────────
    //
    // Each control seeds one invalid generated-row shape and proves the
    // validator fails the build/install for it (v16 §6.1). CI fails if any
    // seeded invalid case stops failing.

    fn valid_row(leaf: &'static [u16], symbolic: &'static str) -> GeneratedMethodPolicyRow {
        GeneratedMethodPolicyRow {
            service: "svc",
            leaf_path: leaf,
            symbolic_path: symbolic,
            scope_action: "query",
            scope_exempt: false,
            authentication: AuthenticationRequirement::CredentialRequired,
            signature_policy: SignaturePolicy::TokenBound {
                suite: CryptoSuite::Hybrid,
            },
            mutation_semantics: None,
            transitional_label: SYSTEM_LOW_LABEL,
            target_label: dispatch_label(Level::Internal, Assurance::PqHybrid, &[]),
            public_reason: None,
        }
    }

    fn public_row(leaf: &'static [u16], symbolic: &'static str) -> GeneratedMethodPolicyRow {
        GeneratedMethodPolicyRow {
            service: "svc",
            leaf_path: leaf,
            symbolic_path: symbolic,
            scope_action: "",
            scope_exempt: true,
            authentication: AuthenticationRequirement::UnauthenticatedAllowed,
            signature_policy: SignaturePolicy::UnauthenticatedOrTokenBound {
                suite: CryptoSuite::Hybrid,
            },
            mutation_semantics: None,
            transitional_label: SYSTEM_LOW_LABEL,
            target_label: SYSTEM_LOW_LABEL,
            public_reason: Some("declared $dispatchPublic in the service schema"),
        }
    }

    #[test]
    fn a_valid_generated_inventory_validates() {
        let rows = vec![valid_row(&[0], "a"), valid_row(&[1, 0], "b.c"), public_row(&[2], "p")];
        validate_generated_rows(&rows).expect("valid inventory must validate");
    }

    #[test]
    fn a_leaf_path_collision_fails_the_build() {
        let rows = vec![valid_row(&[0], "a"), valid_row(&[0], "b")];
        assert!(validate_generated_rows(&rows).is_err());
    }

    #[test]
    fn symbolic_name_drift_fails_the_build() {
        let rows = vec![valid_row(&[0], "a"), valid_row(&[1], "a")];
        assert!(validate_generated_rows(&rows).is_err());
    }

    #[test]
    fn an_empty_leaf_path_fails_the_build() {
        assert!(validate_generated_rows(&[valid_row(&[], "a")]).is_err());
    }

    #[test]
    fn a_public_row_without_a_reason_fails_the_build() {
        let mut row = public_row(&[0], "p");
        row.public_reason = None;
        assert!(validate_generated_rows(&[row.clone()]).is_err());
        row.public_reason = Some("  ");
        assert!(validate_generated_rows(&[row]).is_err());
    }

    /// The §6.1 contradiction in both directions: a public method with an
    /// identity-only signer policy, and a credential-required method with an
    /// unauthenticated-capable signer policy.
    #[test]
    fn public_and_identity_only_contradictions_fail_the_build() {
        let mut public_identity_only = public_row(&[0], "p");
        public_identity_only.signature_policy = SignaturePolicy::TokenBound {
            suite: CryptoSuite::Hybrid,
        };
        assert!(validate_generated_rows(&[public_identity_only]).is_err());

        let mut credential_unattributed = valid_row(&[0], "a");
        credential_unattributed.signature_policy = SignaturePolicy::UnauthenticatedOrTokenBound {
            suite: CryptoSuite::Hybrid,
        };
        assert!(validate_generated_rows(&[credential_unattributed]).is_err());
    }

    #[test]
    fn a_credential_required_row_without_a_scope_fails_the_build() {
        let mut row = valid_row(&[0], "a");
        row.scope_action = "";
        assert!(validate_generated_rows(&[row]).is_err());
    }

    /// The `$dispatchMac`/`$dispatchPublic` label gates (v16 §6/§7.3): public
    /// expands to exactly system low, a MAC'd row may never be system low, and
    /// the transitional column is fixed at system low until the target flip.
    #[test]
    fn label_column_contradictions_fail_the_build() {
        // A public row whose target is anything but system low.
        let mut public_labeled = public_row(&[0], "p");
        public_labeled.target_label = dispatch_label(Level::Internal, Assurance::PqHybrid, &[]);
        assert!(validate_generated_rows(&[public_labeled]).is_err());

        // A credential-required row whose target IS system low.
        let mut mac_system_low = valid_row(&[0], "a");
        mac_system_low.target_label = SYSTEM_LOW_LABEL;
        assert!(validate_generated_rows(&[mac_system_low]).is_err());

        // Any row whose transitional label left system low during migration.
        let mut bad_transitional = valid_row(&[0], "a");
        bad_transitional.transitional_label =
            dispatch_label(Level::Internal, Assurance::PqHybrid, &[]);
        assert!(validate_generated_rows(&[bad_transitional]).is_err());

        // Compartments ride along on the target label and validate.
        let mut compartmented = valid_row(&[0], "a");
        compartmented.target_label =
            dispatch_label(Level::Secret, Assurance::PqHybrid, &[0, 3]);
        assert!(validate_generated_rows(&[compartmented]).is_ok());
    }

    /// A `$scopeExempt` + `$dispatchMac` leaf (e.g. `policy.registerServiceKey`,
    /// gated by CA-signed JWT attestation at the control plane) carries an empty
    /// scope action LEGALLY — but only with the exemption recorded, and never
    /// together with a scope action.
    #[test]
    fn control_plane_exemptions_on_mac_rows_are_recorded_and_exact() {
        // Exempt + empty scope + credential required: legal.
        let mut exempt = valid_row(&[0], "a");
        exempt.scope_action = "";
        exempt.scope_exempt = true;
        assert!(validate_generated_rows(&[exempt]).is_ok());

        // Empty scope WITHOUT the recorded exemption: build error.
        let mut bare = valid_row(&[0], "a");
        bare.scope_action = "";
        bare.scope_exempt = false;
        let err = validate_generated_rows(&[bare]).unwrap_err();
        assert!(err.to_string().contains("no recorded control-plane exemption"), "{err}");

        // Exempt AND carrying a scope action: contradictory.
        let mut both = valid_row(&[0], "a");
        both.scope_exempt = true;
        let err = validate_generated_rows(&[both]).unwrap_err();
        assert!(err.to_string().contains("scope-exempt but carries scope action"), "{err}");
    }

    /// Mutation consistency (v16 §6.1): mutating scope actions require an
    /// explicit `MutationSemantics`; read-class actions must not claim one.
    #[test]
    fn mutation_semantics_mismatch_fails_the_build() {
        // read-class + Some = contradiction.
        let mut read_claiming = valid_row(&[0], "a");
        read_claiming.scope_action = "query";
        read_claiming.mutation_semantics = Some(MutationSemantics::NaturallyIdempotent);
        assert!(validate_generated_rows(&[read_claiming]).is_err());

        // mutating + None = contradiction. A public row (empty scope action)
        // is exempt from this gate — it is reviewed via its public reason.
        let mut mutating_bare = valid_row(&[0], "a");
        mutating_bare.scope_action = "write";
        mutating_bare.mutation_semantics = None;
        assert!(validate_generated_rows(&[mutating_bare]).is_err());
        assert!(validate_generated_rows(&[public_row(&[3], "p2")]).is_ok());

        // mutating + Some stays green, for each declared variant.
        for semantics in [
            MutationSemantics::NaturallyIdempotent,
            MutationSemantics::IdempotencyKeyRequired,
            MutationSemantics::TransactionLedgerRequired,
        ] {
            let mut row = valid_row(&[0], "a");
            row.scope_action = "write";
            row.mutation_semantics = Some(semantics);
            assert!(validate_generated_rows(&[row]).is_ok());
        }

        // Every scope action outside the read-class block is mutating.
        let mut subscribe = valid_row(&[0], "a");
        subscribe.scope_action = "subscribe";
        subscribe.mutation_semantics = Some(MutationSemantics::NaturallyIdempotent);
        assert!(validate_generated_rows(&[subscribe]).is_err());
    }

    #[test]
    fn unsatisfiable_or_malformed_approver_rules_fail_the_build() {
        let approved = |rule: ApproverRule| {
            let mut row = valid_row(&[0], "a");
            row.signature_policy = SignaturePolicy::TokenBoundAndApproved {
                primary_suite: CryptoSuite::Hybrid,
                approver_rule: rule,
            };
            row
        };
        // No allowed groups at all.
        assert!(validate_generated_rows(&[approved(ApproverRule::All { groups: vec![] })]).is_err());
        // Threshold of zero, and threshold above the allowed set.
        assert!(validate_generated_rows(&[approved(ApproverRule::KOfN {
            k: 0,
            groups: vec![group(2, "security")],
        })])
        .is_err());
        assert!(validate_generated_rows(&[approved(ApproverRule::KOfN {
            k: 2,
            groups: vec![group(2, "security")],
        })])
        .is_err());
        // Duplicate group IDs in the allowed set.
        assert!(validate_generated_rows(&[approved(ApproverRule::All {
            groups: vec![group(2, "security"), group(2, "finance")],
        })])
        .is_err());
        // A role rule naming a role no allowed group holds (unknown group/role).
        assert!(validate_generated_rows(&[approved(ApproverRule::Role {
            role: "legal".into(),
            k: 1,
            groups: vec![group(2, "security")],
        })])
        .is_err());
        // An empty role on an allowed group.
        assert!(validate_generated_rows(&[approved(ApproverRule::KOfN {
            k: 1,
            groups: vec![group(2, "")],
        })])
        .is_err());
        // The satisfiable control stays green.
        assert!(validate_generated_rows(&[approved(ApproverRule::KOfN {
            k: 1,
            groups: vec![group(2, "security")],
        })])
        .is_ok());
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
