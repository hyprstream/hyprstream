//! **Typed RPC dispatch labels** — the focused #1499 boot-labeling slice.
//!
//! The mandatory RPC dispatch PEP (`hyprstream_rpc::auth::mac::dispatch_pep`)
//! evaluates a **typed declared `(service, leaf/method)` object identity** plus
//! a **deliberate service subject clearance**. This module supplies both, keyed
//! on the dispatch plane's own coordinates:
//!
//! - [`DispatchMethodId`] — the typed object identity: the canonical service
//!   name plus the canonical request-union discriminant (the leaf/method),
//!   decoded once from the verified payload by the dispatch pipeline. It is
//!   **never** a VFS path, and bare service domains are never routed through
//!   the VFS object-label adapter (`CompositeObjectLabelResolver`'s
//!   `RpcObjectLabelResolver` impl now serves the VFS plane only).
//! - [`ServiceSubjectClearance`] — the deliberate clearance a declared
//!   bootstrap service holds as a *caller*, the dispatch-plane seed of the
//!   v16 `ServiceEnrollmentManifest` clearance. The assurance axis is clamped
//!   by the cryptographically verified key material at evaluation time, so the
//!   declaration can never outrun what the envelope signature proved.
//!
//! ## Deny-by-default
//!
//! There is no wildcard, catch-all, prefix rule, or permissive default. An
//! unknown service, an unknown leaf on a known service, a bare/VFS-shaped
//! alias (`"/srv/policy"`, `"srv/policy"`), and a call whose payload carries
//! no canonical method identity all resolve to `None` ⇒ `UnlabeledObject` ⇒
//! deny before handler entry. A caller with no declared service clearance —
//! including an anonymous caller, for whom the activation floor would
//! otherwise suffice against a floor-labeled row — denies `NoClearance`. The
//! permit of a declared bootstrap call is therefore justified by the
//! *declared pair* (typed object + deliberate clearance), never by a
//! fabricated anonymous/public clearance.
//!
//! ## Relationship to #1507 (v16)
//!
//! This is the smallest deny-by-default representation the production PEP can
//! evaluate today. #1507 replaces the hand-maintained tables with the
//! generated full-method inventory (transitional/target label columns, scope
//! and signature policy) without changing the PEP contract: typed `(service,
//! leaf)` resolution, declared subject clearance, activation-gated selection,
//! and intrinsic lattice dominance all stay. Object labels here are
//! deliberately the lattice floor (`SecurityLabel::bottom()`): the bootstrap
//! identity-lifecycle leaves precede identity standing and must remain
//! reachable under the permanent narrow-to-floor incident control
//! (`FloorOnly`); every raising above the floor is #1507's reviewed
//! transitional/target column work, not this slice.

use hyprstream_rpc::auth::mac::{
    Assurance, CompartmentSet, Level, MacDecision, MacDenyReason, MacDispatchPep,
    RpcObjectLabelResolver, SecurityContext, SecurityLabel,
};
use hyprstream_rpc::service::EnvelopeContext;

/// Subject-name prefix of a verified service identity (`service:{name}`).
///
/// Service JWTs minted by the CA carry `sub = "service:{name}"`
/// (`crates/hyprstream-rpc-std/schema/policy.capnp`), so the verified subject
/// produced by claims verification has this exact shape.
pub const SERVICE_SUBJECT_PREFIX: &str = "service:";

/// Canonical `PolicyRequest` union discriminants (the leaf/method axis of
/// [`DispatchMethodId`]).
///
/// The values are the Cap'n Proto union discriminant ordinals — 0-based
/// declaration order in `hyprstream-rpc-std/schema/policy.capnp`, which
/// assigns no explicit discriminants. The drift test in this module decodes
/// real serialized requests and pins these values to the schema; changing the
/// union order fails CI rather than silently relabeling the dispatch plane.
pub mod policy_methods {
    /// `registerServiceKey` — a keyed service installs its identity with the
    /// CA. Bootstrap-critical: it precedes identity standing.
    pub const REGISTER_SERVICE_KEY: u16 = 18;
    /// `refreshServiceToken` — a registered service renews the identity the
    /// registration created. Same bootstrap identity-lifecycle class; without
    /// a declared row the first hourly renewal would deny on any live node.
    pub const REFRESH_SERVICE_TOKEN: u16 = 19;
}

/// The deliberate subject clearance declared for the staging bootstrap
/// services: an internal system principal presenting a verified classical
/// key. The assurance axis is a *ceiling*, not a grant — evaluation clamps it
/// to the cryptographically verified key material (#548), so a caller whose
/// envelope proved less derives less.
pub const BOOTSTRAP_SERVICE_CLEARANCE: SecurityLabel = SecurityLabel {
    level: Level::Internal,
    assurance: Assurance::Classical,
    compartments: CompartmentSet::EMPTY,
};

/// Typed dispatch object identity: canonical service name + canonical
/// request-union discriminant. Never a VFS path; never interpreted by another
/// plane's resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DispatchMethodId {
    /// The canonical registered service name (`policy`, `discovery`, …).
    pub service: &'static str,
    /// The canonical request-union discriminant of the method leaf.
    pub method: u16,
}

/// One declared `(service, leaf/method)` policy row: the typed object label
/// for exactly one RPC method on exactly one service.
#[derive(Debug, Clone, Copy)]
pub struct DispatchMethodPolicy {
    /// The typed object identity this row labels.
    pub id: DispatchMethodId,
    /// The symbolic leaf name (`registerServiceKey`) — audit and drift-test
    /// evidence; resolution is by the numeric identity only.
    pub method_name: &'static str,
    /// The declared object label. Bootstrap rows are the lattice floor; see
    /// the module docs for why no row may rise above it in this slice.
    pub label: SecurityLabel,
    /// Why this row exists and why its label is correct (reviewed text).
    pub justification: &'static str,
}

/// The deliberate MAC clearance a declared service holds as a dispatch
/// *caller* — the dispatch-plane seed of the v16 `ServiceEnrollmentManifest`
/// service clearance.
#[derive(Debug, Clone, Copy)]
pub struct ServiceSubjectClearance {
    /// The canonical service name (the `service:{name}` subject suffix).
    pub service: &'static str,
    /// The declared clearance. Assurance is clamped to the verified key
    /// material at evaluation time.
    pub clearance: SecurityLabel,
    /// Why this clearance is correct for this service (reviewed text).
    pub justification: &'static str,
}

/// The declared dispatch table: a closed, deny-by-default set of typed
/// `(service, leaf/method)` object labels plus the deliberate per-service
/// subject clearances. Consumed by [`DeclaredDispatchPep`]; replaceable by
/// #1507's generated inventory without changing the PEP contract.
#[derive(Debug, Clone, Copy)]
pub struct DeclaredDispatchTable {
    methods: &'static [DispatchMethodPolicy],
    clearances: &'static [ServiceSubjectClearance],
}

impl DeclaredDispatchTable {
    /// The production staging-bootstrap declarations (#1499).
    ///
    /// Object rows cover exactly the dispatch calls the fresh-state
    /// `service start --services event,policy,discovery,registry,oauth` boot
    /// graph makes: every keyed non-policy service registers its signing key
    /// with the PolicyService CA at startup and renews that identity
    /// hourly. Subject rows declare the deliberate caller clearance for all
    /// five bootstrap services.
    #[must_use]
    pub fn production() -> &'static Self {
        &PRODUCTION_TABLE
    }

    /// Construct a table from explicit rows (test fixtures). Production uses
    /// [`Self::production`].
    #[must_use]
    pub const fn from_rows(
        methods: &'static [DispatchMethodPolicy],
        clearances: &'static [ServiceSubjectClearance],
    ) -> Self {
        Self {
            methods,
            clearances,
        }
    }

    /// Every declared method row (table-driven test and audit evidence).
    #[must_use]
    pub fn methods(&self) -> &'static [DispatchMethodPolicy] {
        self.methods
    }

    /// Every declared service subject clearance row.
    #[must_use]
    pub fn clearances(&self) -> &'static [ServiceSubjectClearance] {
        self.clearances
    }

    /// Resolve the declared policy row for a typed `(service, leaf)` call.
    ///
    /// Exact match only: `method` must be `Some` (the canonical discriminant
    /// committed by the verified payload) and the pair must be declared.
    /// Unknown services, unknown leaves, VFS-shaped aliases, and calls with
    /// no committed method identity all return `None` ⇒ deny.
    #[must_use]
    pub fn resolve_row(
        &self,
        service_domain: &str,
        method: Option<u16>,
    ) -> Option<&DispatchMethodPolicy> {
        let method = method?;
        self.methods
            .iter()
            .find(|row| row.id.service == service_domain && row.id.method == method)
    }

    /// The deliberate clearance declared for a service caller, if any.
    ///
    /// `service` is the canonical service name (the `service:` subject prefix
    /// is stripped by the caller). Exact match; no declaration ⇒ `None` ⇒ the
    /// caller has no dispatch authority (deny `NoClearance`).
    #[must_use]
    pub fn service_clearance(&self, service: &str) -> Option<SecurityLabel> {
        self.clearances
            .iter()
            .find(|row| row.service == service)
            .map(|row| row.clearance)
    }
}

impl RpcObjectLabelResolver for DeclaredDispatchTable {
    /// Typed dispatch-plane resolution. No path splitting, no prefix
    /// matching, no VFS adapter: an exact declared `(service, leaf)` row or
    /// `None` (deny).
    fn resolve(&self, service_domain: &str, method: Option<u16>) -> Option<SecurityLabel> {
        self.resolve_row(service_domain, method)
            .map(|row| row.label)
    }
}

/// The production dispatch PEP over the declared table.
///
/// Decision procedure (every missing input denies; no permissive mode):
///
/// 1. **Activation-selected context** — the coverage-gated operator control
///    (`FloorOnly` ⇒ anonymous floor; `IdentityAware` ⇒ the verified
///    Claims × VerifiedKeyMaterial derivation). `None` ⇒ `NoClearance`. The
///    activation gate and the narrow-to-floor kill-switch are unchanged: a
///    narrowed process still evaluates every dispatch, and only floor-labeled
///    (bootstrap-critical) rows remain reachable.
/// 2. **Deliberate service subject clearance** — the verified subject must be
///    a `service:{name}` identity with a real authenticated envelope signer
///    key AND a declared clearance in the table. The declared clearance is
///    composed with the crypto-derived assurance (`SecurityContext::
///    from_clearance` clamps down, never up). No declaration ⇒ `NoClearance`
///    — an anonymous or undeclared caller can never ride the activation floor
///    into a declared row.
/// 3. **Typed object identity** — an exact declared `(service, leaf)` row.
///    `None` ⇒ `UnlabeledObject`.
/// 4. **Dominance** — BOTH the activation-selected context and the deliberate
///    declared-service context must dominate the declared object label
///    (`can_access`). Otherwise `FloorDeny`.
///
/// Like the genesis PEP it replaces at the install seam, this PEP performs no
/// permit caching and holds no permissive state.
pub struct DeclaredDispatchPep {
    table: &'static DeclaredDispatchTable,
    activation_controlled: bool,
}

impl DeclaredDispatchPep {
    /// Construct over a declared table.
    ///
    /// Direct unit-test and embedding constructors keep the historical
    /// identity-aware behavior (the verified context is used as-is);
    /// production installs with [`Self::with_activation_control`].
    #[must_use]
    pub fn new(table: &'static DeclaredDispatchTable) -> Self {
        Self {
            table,
            activation_controlled: false,
        }
    }

    /// Select subject contexts through the process-global coverage gate.
    /// Production's install path opts in explicitly.
    #[must_use]
    pub fn with_activation_control(mut self) -> Self {
        self.activation_controlled = true;
        self
    }

    /// The declared table this PEP evaluates.
    #[must_use]
    pub fn table(&self) -> &'static DeclaredDispatchTable {
        self.table
    }
}

/// Extract the canonical service name from a verified service subject.
///
/// The declared clearance attaches only to a `service:{name}` identity backed
/// by a real authenticated envelope signer key — never to an anonymous
/// subject and never to a keyless internal callback context
/// (`EnvelopeContext::from_callback_service` zeroes `cnf`, so
/// `authenticated_signer_key` refuses it).
fn declared_service_subject(ctx: &EnvelopeContext) -> Option<String> {
    ctx.authenticated_signer_key()?;
    let subject = ctx.subject();
    let name = subject.name()?;
    name.strip_prefix(SERVICE_SUBJECT_PREFIX)
        .filter(|suffix| !suffix.is_empty())
        .map(str::to_owned)
}

impl MacDispatchPep for DeclaredDispatchPep {
    fn check(
        &self,
        ctx: &EnvelopeContext,
        service_domain: &str,
        method: Option<u16>,
    ) -> MacDecision {
        // Preserve the verified context for the direct VFS/CAS/MoQ PEPs, whose
        // low-level APIs carry Subject but not the full verified envelope.
        // (Parity with DefaultMacDispatchPep.)
        hyprstream_rpc::auth::mac::remember_verified_subject(ctx);

        // 1. Activation-selected subject context (floor-only or verified
        //    identity-aware). The operator gate is unchanged by this PEP.
        let selected = if self.activation_controlled {
            hyprstream_rpc::auth::mac::global_mac_activation_control()
                .select_context(ctx.security_context())
        } else {
            ctx.security_context()
        };
        let Some(selected) = selected else {
            return MacDecision::Deny(MacDenyReason::NoClearance);
        };

        // 2. Deliberate declared service subject clearance. The assurance axis
        //    is clamped to the verified key material; the declaration cannot
        //    outrun the crypto.
        let Some(service_name) = declared_service_subject(ctx) else {
            return MacDecision::Deny(MacDenyReason::NoClearance);
        };
        let Some(declared_clearance) = self.table.service_clearance(&service_name) else {
            return MacDecision::Deny(MacDenyReason::NoClearance);
        };
        let service_ctx =
            SecurityContext::from_clearance(declared_clearance, ctx.verified_key_material());

        // 3. Typed declared (service, leaf) object identity.
        let Some(row) = self.table.resolve_row(service_domain, method) else {
            return MacDecision::Deny(MacDenyReason::UnlabeledObject);
        };

        // 4. Intrinsic lattice floor: both the activation-selected context and
        //    the deliberate declared-service context must dominate the label.
        if selected.can_access(&row.label) && service_ctx.can_access(&row.label) {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::FloorDeny)
        }
    }
}

// ── Production declarations (the staging bootstrap set, #1499) ─────────────

static BOOTSTRAP_METHODS: &[DispatchMethodPolicy] = &[
    DispatchMethodPolicy {
        id: DispatchMethodId {
            service: "policy",
            method: policy_methods::REGISTER_SERVICE_KEY,
        },
        method_name: "registerServiceKey",
        label: SecurityLabel::bottom(),
        justification: "a keyed service installs its identity with the CA; \
             bootstrap-critical (it precedes identity standing) and reachable \
             only by declared service subjects; handler-level CA-JWT, subject, \
             and cnf binding checks are unchanged",
    },
    DispatchMethodPolicy {
        id: DispatchMethodId {
            service: "policy",
            method: policy_methods::REFRESH_SERVICE_TOKEN,
        },
        method_name: "refreshServiceToken",
        label: SecurityLabel::bottom(),
        justification: "a registered service renews the identity registration \
             created; same bootstrap identity-lifecycle class — the handler \
             re-binds cnf to the verified caller key and requires prior \
             registration",
    },
];

static BOOTSTRAP_SERVICE_CLEARANCES: &[ServiceSubjectClearance] = &[
    ServiceSubjectClearance {
        service: "discovery",
        clearance: BOOTSTRAP_SERVICE_CLEARANCE,
        justification: "announce/presence precedes identity standing; \
             discovery registers its key at boot",
    },
    ServiceSubjectClearance {
        service: "event",
        clearance: BOOTSTRAP_SERVICE_CLEARANCE,
        justification: "the event bus is startup wiring, not content; \
             declared so the bootstrap set holds one uniform deliberate \
             clearance",
    },
    ServiceSubjectClearance {
        service: "model",
        clearance: BOOTSTRAP_SERVICE_CLEARANCE,
        justification: "the staged synthetic model registers its key before \
             its inference RPC surface can start",
    },
    ServiceSubjectClearance {
        service: "oauth",
        clearance: BOOTSTRAP_SERVICE_CLEARANCE,
        justification: "login/session issuance is how identity standing is \
             acquired at all; oauth registers its key at boot",
    },
    ServiceSubjectClearance {
        service: "policy",
        clearance: BOOTSTRAP_SERVICE_CLEARANCE,
        justification: "the CA itself; it makes no boot RPC calls, but its \
             enrolled caller clearance is declared with the same deliberate \
             value as the services it certifies",
    },
    ServiceSubjectClearance {
        service: "registry",
        clearance: BOOTSTRAP_SERVICE_CLEARANCE,
        justification: "service/model registration is part of boot; registry \
             registers its key at boot",
    },
];

static PRODUCTION_TABLE: DeclaredDispatchTable =
    DeclaredDispatchTable::from_rows(BOOTSTRAP_METHODS, BOOTSTRAP_SERVICE_CLEARANCES);

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ── fixtures ────────────────────────────────────────────────────────

    fn service_subject_ctx(service: &str, key_byte: u8) -> EnvelopeContext {
        let signer = ed25519_dalek::SigningKey::from_bytes(&[key_byte; 32]);
        EnvelopeContext::for_test_authenticated_subject(
            hyprstream_rpc::Subject::new(format!("service:{service}")),
            signer.verifying_key(),
        )
    }

    fn user_subject_ctx() -> EnvelopeContext {
        let signer = ed25519_dalek::SigningKey::from_bytes(&[0x77; 32]);
        EnvelopeContext::for_test_authenticated_subject(
            hyprstream_rpc::Subject::new("did:web:alice"),
            signer.verifying_key(),
        )
    }

    /// The production PEP exactly as the daemon installs it (floor-only
    /// activation control, production table).
    fn production_pep() -> DeclaredDispatchPep {
        DeclaredDispatchPep::new(DeclaredDispatchTable::production()).with_activation_control()
    }

    // ── table coverage: every declared (service, leaf) + clearance ──────

    #[test]
    fn every_declared_call_resolves_to_the_intended_typed_label_and_clearance() {
        let table = DeclaredDispatchTable::production();

        // The fresh boot graph: discovery, registry, and oauth each call
        // policy.registerServiceKey on fresh state; renewal uses
        // policy.refreshServiceToken. Every declared row resolves to the
        // intended typed label — the lattice floor, deliberate and reviewed.
        let expected: &[(u16, &str)] = &[
            (policy_methods::REGISTER_SERVICE_KEY, "registerServiceKey"),
            (policy_methods::REFRESH_SERVICE_TOKEN, "refreshServiceToken"),
        ];
        assert_eq!(table.methods().len(), expected.len());
        for (method, name) in expected {
            let row = table
                .resolve_row("policy", Some(*method))
                .unwrap_or_else(|| panic!("declared row for policy.{name} must resolve"));
            assert_eq!(row.method_name, *name);
            assert_eq!(row.id.service, "policy");
            assert_eq!(
                row.label,
                SecurityLabel::bottom(),
                "bootstrap-critical row {name} stays at the floor (reachable under narrow-to-floor)"
            );
            assert!(!row.justification.is_empty());
        }

        // The six staging bootstrap services each hold the deliberate
        // service subject clearance, and each declared caller's clearance
        // dominates every declared object label (the boot calls evaluate).
        let mut services: Vec<&str> = table.clearances().iter().map(|row| row.service).collect();
        services.sort_unstable();
        assert_eq!(
            services,
            ["discovery", "event", "model", "oauth", "policy", "registry"]
        );
        for row in table.clearances() {
            assert_eq!(row.clearance, BOOTSTRAP_SERVICE_CLEARANCE);
            assert_eq!(
                table.service_clearance(row.service),
                Some(row.clearance),
                "clearance lookup by canonical name"
            );
            assert!(!row.justification.is_empty());
            for method in table.methods() {
                let ctx = SecurityContext::from_clearance(
                    row.clearance,
                    hyprstream_rpc::auth::mac::VerifiedKeyMaterial::Classical,
                );
                assert!(
                    ctx.can_access(&method.label),
                    "declared clearance for {} must dominate declared label {} ({})",
                    row.service,
                    method.label,
                    method.method_name
                );
            }
        }
    }

    #[test]
    fn table_has_no_duplicate_or_unsorted_rows() {
        let table = DeclaredDispatchTable::production();
        let ids: Vec<DispatchMethodId> = table.methods().iter().map(|row| row.id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids, sorted, "method rows must be sorted and unique");
        let services: Vec<&str> = table.clearances().iter().map(|row| row.service).collect();
        let mut sorted_services = services.clone();
        sorted_services.sort_unstable();
        sorted_services.dedup();
        assert_eq!(
            services, sorted_services,
            "clearance rows must be sorted and unique"
        );
    }

    // ── deny-by-default: unknown service / leaf / VFS alias ─────────────

    #[test]
    fn unknown_service_unknown_leaf_and_vfs_alias_resolve_to_none() {
        let table = DeclaredDispatchTable::production();

        // Unknown service, even with a declared method number.
        assert!(table
            .resolve_row("ghost", Some(policy_methods::REGISTER_SERVICE_KEY))
            .is_none());
        // Unknown leaf on a known service: resolveServiceKey (17) is a real
        // policy method but NOT declared — declaration, not schema, is the
        // authority.
        assert!(table.resolve_row("policy", Some(17)).is_none());
        // Out-of-schema leaf on a known service.
        assert!(table.resolve_row("policy", Some(u16::MAX)).is_none());
        // No committed method identity (a payload without a canonical
        // discriminant) matches no declared row, even on a known service.
        assert!(table.resolve_row("policy", None).is_none());
        // Bare/VFS-shaped aliases never enter the VFS adapter and never match.
        for alias in ["/srv/policy", "srv/policy", "/policy", "policy/", "/"] {
            assert!(
                table
                    .resolve_row(alias, Some(policy_methods::REGISTER_SERVICE_KEY))
                    .is_none(),
                "VFS-shaped alias {alias:?} must not resolve"
            );
        }
        // Case and whitespace variants are not canonical names.
        assert!(table
            .resolve_row("Policy", Some(policy_methods::REGISTER_SERVICE_KEY))
            .is_none());
        assert!(table
            .resolve_row(" policy", Some(policy_methods::REGISTER_SERVICE_KEY))
            .is_none());
        // Undeclared service clearance.
        assert!(table.service_clearance("ghost").is_none());
        assert!(table.service_clearance("metrics").is_none());

        // The RpcObjectLabelResolver view agrees (None ⇒ deny).
        let resolver: &dyn RpcObjectLabelResolver = table;
        assert!(resolver
            .resolve("policy", Some(policy_methods::REGISTER_SERVICE_KEY))
            .is_some());
        assert!(resolver.resolve("policy", Some(17)).is_none());
        assert!(resolver
            .resolve("/srv/policy", Some(policy_methods::REGISTER_SERVICE_KEY))
            .is_none());
    }

    // ── causal PEP pair over the production table ───────────────────────

    #[test]
    fn declared_call_permits_and_identical_undeclared_call_denies() {
        let pep = production_pep();
        let caller = service_subject_ctx("discovery", 0x61);

        // Positive: the exact fresh-boot call — discovery registering its key.
        let declared = pep.check(
            &caller,
            "policy",
            Some(policy_methods::REGISTER_SERVICE_KEY),
        );
        assert_eq!(
            declared,
            MacDecision::Permit,
            "declared (policy, registerServiceKey) from declared discovery must permit"
        );

        // Causal twin: identical caller, identical service, undeclared leaf.
        let undeclared = pep.check(&caller, "policy", Some(17));
        assert_eq!(
            undeclared,
            MacDecision::Deny(MacDenyReason::UnlabeledObject),
            "an otherwise identical undeclared call must deny UnlabeledObject"
        );

        // Unknown service denies the same way.
        assert_eq!(
            pep.check(&caller, "ghost", Some(policy_methods::REGISTER_SERVICE_KEY)),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );

        // VFS-shaped alias of a declared pair denies (never routed through the
        // VFS adapter).
        assert_eq!(
            pep.check(
                &caller,
                "/srv/policy",
                Some(policy_methods::REGISTER_SERVICE_KEY)
            ),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );

        // No committed method identity ⇒ no declared row ⇒ deny.
        assert_eq!(
            pep.check(&caller, "policy", None),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
    }

    #[test]
    fn permit_requires_the_deliberate_service_clearance_not_the_anonymous_floor() {
        let pep = production_pep();

        // An undeclared service subject denies even for a declared object row:
        // the activation floor alone must NOT carry a caller into the table —
        // that is the fabricated-anonymous-clearance hole this slice closes.
        let ghost = service_subject_ctx("ghost", 0x62);
        assert_eq!(
            pep.check(&ghost, "policy", Some(policy_methods::REGISTER_SERVICE_KEY)),
            MacDecision::Deny(MacDenyReason::NoClearance),
            "undeclared service subject must deny NoClearance"
        );

        // A non-service (user) identity has no service clearance.
        let user = user_subject_ctx();
        assert_eq!(
            pep.check(&user, "policy", Some(policy_methods::REGISTER_SERVICE_KEY)),
            MacDecision::Deny(MacDenyReason::NoClearance)
        );

        // A keyless internal callback context (zeroed cnf) asserting a service
        // subject has no authenticated signer key and must not inherit the
        // declared clearance.
        let callback = EnvelopeContext::from_callback_service(1, "discovery");
        assert_eq!(
            pep.check(
                &callback,
                "policy",
                Some(policy_methods::REGISTER_SERVICE_KEY)
            ),
            MacDecision::Deny(MacDenyReason::NoClearance),
            "keyless callback contexts must not claim a service clearance"
        );

        // Every declared bootstrap service caller permits the boot
        // registration call (deliberate clearance composed with verified key
        // material).
        for service in ["discovery", "event", "model", "oauth", "policy", "registry"] {
            let ctx = service_subject_ctx(service, 0x63);
            assert_eq!(
                pep.check(&ctx, "policy", Some(policy_methods::REGISTER_SERVICE_KEY)),
                MacDecision::Permit,
                "declared service:{service} must permit the declared bootstrap call"
            );
        }
    }

    #[test]
    fn floor_deny_when_declared_clearance_cannot_dominate_the_label() {
        // Fixture table: a declared row above the declared clearance. Both
        // dominance checks must hold; here the declared service context
        // (Internal) cannot dominate a Secret object label.
        static SECRET_ROW: &[DispatchMethodPolicy] = &[DispatchMethodPolicy {
            id: DispatchMethodId {
                service: "policy",
                method: 7,
            },
            method_name: "fixture",
            label: SecurityLabel {
                level: Level::Secret,
                assurance: Assurance::Unverified,
                compartments: CompartmentSet::EMPTY,
            },
            justification: "test fixture: label deliberately above the declared clearance",
        }];
        static TABLE: DeclaredDispatchTable =
            DeclaredDispatchTable::from_rows(SECRET_ROW, BOOTSTRAP_SERVICE_CLEARANCES);
        let pep = DeclaredDispatchPep::new(&TABLE).with_activation_control();
        let caller = service_subject_ctx("discovery", 0x64);
        assert_eq!(
            pep.check(&caller, "policy", Some(7)),
            MacDecision::Deny(MacDenyReason::FloorDeny),
            "a label above the declared clearance must FloorDeny"
        );
    }

    // ── installed-PEP causal pair ───────────────────────────────────────
    //
    // The process-global install + "the PEP stayed installed" assertion lives
    // in the integration binary `tests/mac_dispatch_boot_labeling.rs`, which
    // owns its own globals; lib tests share this binary's global PEP slot with
    // the `install_explicit_test_dispatch_pep` plumbing fixtures and cannot
    // serialize against them.

    // ── drift: declared discriminants == real schema discriminants ──────

    #[test]
    fn declared_policy_discriminants_match_the_serialized_schema() {
        use capnp::message::Builder;

        // registerServiceKey
        let mut message = Builder::new_default();
        {
            let mut req =
                message.init_root::<hyprstream_rpc_std::policy_capnp::policy_request::Builder>();
            req.set_id(1);
            let mut call = req.reborrow().init_register_service_key();
            call.set_service_name("discovery");
            call.set_verifying_key(&[0x42; 32]);
            call.set_service_jwt("header.payload.signature");
        }
        let bytes = capnp::serialize::write_message_to_words(&message);
        assert_eq!(
            hyprstream_rpc::browser_provisioning::canonical_method_discriminator(&bytes).unwrap(),
            policy_methods::REGISTER_SERVICE_KEY,
            "the declared registerServiceKey discriminant must match the schema union ordinal"
        );

        // refreshServiceToken
        let mut message = Builder::new_default();
        {
            let mut req =
                message.init_root::<hyprstream_rpc_std::policy_capnp::policy_request::Builder>();
            req.set_id(1);
            req.reborrow()
                .init_refresh_service_token()
                .set_ttl_seconds(2_592_000);
        }
        let bytes = capnp::serialize::write_message_to_words(&message);
        assert_eq!(
            hyprstream_rpc::browser_provisioning::canonical_method_discriminator(&bytes).unwrap(),
            policy_methods::REFRESH_SERVICE_TOKEN,
            "the declared refreshServiceToken discriminant must match the schema union ordinal"
        );
    }

    // ── drift: declared services exist in the factory inventory ─────────

    #[test]
    fn declared_bootstrap_services_are_registered_factories() {
        let registered: Vec<&str> = hyprstream_service::list_factories()
            .map(|factory| factory.name)
            .collect();
        for row in DeclaredDispatchTable::production().clearances() {
            assert!(
                registered.contains(&row.service),
                "declared bootstrap service {} must be a registered service factory",
                row.service
            );
        }
        for row in DeclaredDispatchTable::production().methods() {
            assert!(
                registered.contains(&row.id.service),
                "declared dispatch target {} must be a registered service factory",
                row.id.service
            );
        }
    }
}
