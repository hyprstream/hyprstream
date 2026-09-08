//! Full-leaf derivation and generated method-policy inventory over the REAL
//! production schemas linked into the hyprstream binary (v16 §5.1/§6.1,
//! issues #1504/#1505).
//!
//! Proves on real generated artifacts:
//!
//! 1. the generated single-decode body decoder derives the FULL nested
//!    numeric leaf path — root, scoped, and doubly-nested scoped methods each
//!    produce distinct multi-level paths;
//! 2. every derivable leaf resolves a row in the generated method-policy
//!    inventory (decoder/inventory agreement by construction); and
//! 3. the complete linked inventory — every service crate's generated rows —
//!    validates and installs deterministically. This test IS the build gate:
//!    a schema change that produces a colliding, contradictory, or
//!    unannotated row fails here.
//!
//! WS-D additions (v16 §6): every row derives from the strict
//! `$dispatchMac`/`$dispatchPublic` pair — the transitional label is exactly
//! system low everywhere, MAC rows never target system low. The public rows
//! are `policy.check` and `policy.checkBatch`, each with the same recorded
//! circular-authorization reason; `policy
//! registerServiceKey` is MAC'd (not public) with its control-plane exemption
//! recorded, and mutation semantics are consistent with the checked
//! scope-action blocks over the whole real inventory.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use hyprstream_core::registry_capnp;
use hyprstream_core::services::generated::registry_client::decode_registry_request_body;
use hyprstream_rpc::auth::mac::{Assurance, Level};
use hyprstream_rpc::proof::policy;

fn to_bytes(message: &capnp::message::Builder<capnp::message::HeapAllocator>) -> Vec<u8> {
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, message).unwrap();
    bytes
}

fn leaf_of(payload: &[u8]) -> Vec<u16> {
    decode_registry_request_body(payload)
        .expect("real request decodes")
        .leaf_path()
        .expect("real request derives a leaf")
        .to_vec()
}

/// Root, 2-level, and 3-level requests derive distinct full paths of the
/// expected depth — the nested unions extend the leaf instead of collapsing
/// onto their parent selector.
#[test]
fn nested_methods_derive_full_multi_level_paths() {
    // Root leaf: `list`.
    let mut root = capnp::message::Builder::new_default();
    {
        let mut req = root.init_root::<registry_capnp::registry_request::Builder>();
        req.set_id(1);
        req.set_list(());
    }
    let root_leaf = leaf_of(&to_bytes(&root));
    assert_eq!(root_leaf.len(), 1, "root method is a one-element path");

    // Scoped leaf: `repo.listWorktrees`.
    let mut scoped = capnp::message::Builder::new_default();
    {
        let mut req = scoped.init_root::<registry_capnp::registry_request::Builder>();
        req.set_id(2);
        let mut repo = req.init_repo();
        repo.set_repo_id("r1");
        repo.set_list_worktrees(());
    }
    let scoped_leaf = leaf_of(&to_bytes(&scoped));
    assert_eq!(
        scoped_leaf.len(),
        2,
        "a scoped method derives selector + method: {scoped_leaf:?}"
    );

    // Doubly-nested leaf: `repo.worktree.clunk`.
    let mut nested = capnp::message::Builder::new_default();
    {
        let mut req = nested.init_root::<registry_capnp::registry_request::Builder>();
        req.set_id(3);
        let mut repo = req.init_repo();
        repo.set_repo_id("r1");
        let mut worktree = repo.init_worktree();
        worktree.set_name("wt");
        let mut clunk = worktree.init_clunk();
        clunk.set_fid(4);
    }
    let nested_leaf = leaf_of(&to_bytes(&nested));
    assert_eq!(
        nested_leaf.len(),
        3,
        "a doubly-nested method derives the full three-level path: {nested_leaf:?}"
    );

    // The nested paths share their prefix with the scope selector and are
    // all distinct identities.
    assert_eq!(scoped_leaf[0], nested_leaf[0], "same repo selector");
    assert_ne!(root_leaf, scoped_leaf);
    assert_ne!(scoped_leaf, nested_leaf);
}

/// A malformed body denies at decode — never a truncated or coarser path.
#[test]
fn malformed_bodies_do_not_decode() {
    assert!(decode_registry_request_body(b"garbage").is_err());
    assert!(decode_registry_request_body(&[]).is_err());
}

/// Every derivable leaf resolves a generated inventory row, symbolically
/// named for review, and the complete linked inventory validates + installs.
#[test]
fn generated_inventory_is_complete_valid_and_lists_derived_leaves() {
    let rows = policy::collect_generated_rows().expect("inventory collects");
    policy::validate_generated_rows(&rows).expect("the complete linked inventory validates");

    // Every server-linked schema contributed rows.
    for service in [
        "registry", "model", "policy", "inference", "mcp", "tui", "metrics", "oauth",
        "discovery", "worker", "workflow",
    ] {
        assert!(
            rows.iter().any(|row| row.service == service),
            "no generated rows for service '{service}'"
        );
    }

    // Deterministic order: sorted by (service, numeric leaf path).
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| {
        a.service
            .cmp(b.service)
            .then_with(|| a.leaf_path.cmp(b.leaf_path))
    });
    assert!(
        rows.iter()
            .zip(&sorted)
            .all(|(a, b)| a.service == b.service && a.leaf_path == b.leaf_path),
        "collected inventory must already be deterministically sorted"
    );

    // Decoder/inventory agreement on real nested requests.
    let mut nested = capnp::message::Builder::new_default();
    {
        let mut req = nested.init_root::<registry_capnp::registry_request::Builder>();
        req.set_id(3);
        let mut repo = req.init_repo();
        repo.set_repo_id("r1");
        let mut worktree = repo.init_worktree();
        worktree.set_name("wt");
        worktree.init_clunk().set_fid(4);
    }
    let body = decode_registry_request_body(&to_bytes(&nested)).unwrap();
    let leaf_key = body.leaf_path_string().unwrap();
    let row = rows
        .iter()
        .find(|row| row.service == "registry" && row.leaf_key() == leaf_key)
        .expect("the derived nested leaf must be listed in the generated inventory");
    assert_eq!(row.symbolic_path, "repo.worktree.clunk");
    assert_eq!(row.scope_action, "query");
    assert_eq!(
        row.authentication,
        policy::AuthenticationRequirement::CredentialRequired
    );

    // The built table resolves exactly the listed leaves.
    use policy::DispatchMethodPolicy as _;
    let (table, count) = policy::build_generated_method_policy().unwrap();
    assert_eq!(count, rows.len());
    assert!(table.policy_for("registry", &leaf_key).is_some());
    assert!(table.policy_for("registry", "9999.9999").is_none());
}

/// WS-D (v16 §6): the strict dispatch pair over the complete REAL inventory.
///
/// - every row's transitional label is exactly system low (§7.3);
/// - every `$dispatchMac` row targets the parsed label — never system low —
///   and, for the uniform production default, exactly
///   `internal:pq-hybrid` (the target inventory stays operator-reviewable;
///   this asserts what the schemas declare today, not a ceiling);
/// - the public set is exactly `policy.check` and `policy.checkBatch`, each
///   with its circular-authorization reason recorded; and
/// - `policy.registerServiceKey` is `$dispatchMac` (credential required), its
///   control-plane exemption recorded, never public.
#[test]
fn the_complete_inventory_carries_the_strict_dispatch_pair() {
    let rows = policy::collect_generated_rows().expect("inventory collects");
    policy::validate_generated_rows(&rows).expect("inventory validates");

    let mut public_rows = Vec::new();
    for row in &rows {
        assert_eq!(
            row.transitional_label,
            policy::SYSTEM_LOW_LABEL,
            "{}:{} transitional column must be system low",
            row.service,
            row.symbolic_path
        );
        match row.authentication {
            policy::AuthenticationRequirement::CredentialRequired => {
                assert_ne!(
                    row.target_label,
                    policy::SYSTEM_LOW_LABEL,
                    "{}:{} MAC row must not target system low",
                    row.service,
                    row.symbolic_path
                );
                assert!(
                    row.public_reason.is_none(),
                    "{}:{} MAC row carries a public reason",
                    row.service,
                    row.symbolic_path
                );
                // The uniform production default today (operator-reviewable).
                assert_eq!(
                    row.target_label,
                    policy::dispatch_label(Level::Internal, Assurance::PqHybrid, &[]),
                    "{}:{} target label is not the declared uniform default",
                    row.service,
                    row.symbolic_path
                );
            }
            policy::AuthenticationRequirement::UnauthenticatedAllowed => {
                assert_eq!(
                    row.target_label,
                    policy::SYSTEM_LOW_LABEL,
                    "{}:{} public row must expand to exactly system low",
                    row.service,
                    row.symbolic_path
                );
                public_rows.push((row.service, row.symbolic_path));
            }
        }
    }

    assert_eq!(
        public_rows,
        vec![("policy", "check"), ("policy", "checkBatch")],
        "the public dispatch set is exactly the circular-authorization policy checks"
    );
    for path in ["check", "checkBatch"] {
        let row = rows
            .iter()
            .find(|row| row.service == "policy" && row.symbolic_path == path)
            .expect("public policy check row present");
        assert_eq!(
            row.public_reason,
            Some("the authz check itself cannot require authz without circularity; it is the one leaf dispatched unauthenticated"),
            "policy.{path} records the circular-authorization public reason"
        );
    }

    // registerServiceKey: MAC'd, credential-required, control-plane exemption
    // recorded (scope action empty + scope_exempt), never public.
    let rsk = rows
        .iter()
        .find(|r| r.service == "policy" && r.symbolic_path == "registerServiceKey")
        .expect("policy.registerServiceKey row present");
    assert_eq!(
        rsk.authentication,
        policy::AuthenticationRequirement::CredentialRequired
    );
    assert!(rsk.scope_action.is_empty());
    assert!(rsk.scope_exempt, "control-plane exemption recorded");
    assert!(rsk.public_reason.is_none());
}

/// Mutation semantics are consistent with the checked scope-action blocks
/// across the whole real inventory. Authorization scope and application-effect
/// policy are separate: a read-class authorization may explicitly classify a
/// bounded session/subscription mutation, while non-read scope requires one.
#[test]
fn mutation_semantics_follow_the_scope_action_blocks() {
    let rows = policy::collect_generated_rows().expect("inventory collects");
    for row in &rows {
        if row.scope_action.is_empty() {
            assert!(
                row.scope_exempt,
                "{}:{} has no scope action without its recorded exemption",
                row.service,
                row.symbolic_path
            );
            continue;
        }
        let read_class = policy::READ_CLASS_ACTIONS.contains(&row.scope_action);
        if !read_class {
            assert!(
                row.mutation_semantics.is_some(),
                "{}:{} (scope {}) must declare mutation semantics",
                row.service,
                row.symbolic_path,
                row.scope_action
            );
        }
    }
}

/// TUI declarations follow the actual handler effects: setting focus/size is
/// convergent, process input and allocation paths require a future
/// application idempotency key/result record before retry behavior may be
/// enabled, and `pollStdin` is a destructive queue drain — a lost reply
/// consumes bytes a retry cannot redeliver.
#[test]
fn tui_mutation_policies_are_explicit_and_handler_accurate() {
    use policy::MutationSemantics;

    let rows = policy::collect_generated_rows().expect("inventory collects");
    let semantics = |leaf: &str| {
        rows.iter()
            .find(|row| row.service == "tui" && row.symbolic_path == leaf)
            .unwrap_or_else(|| panic!("missing tui:{leaf}"))
            .mutation_semantics
    };
    assert_eq!(semantics("focusWindow"), Some(MutationSemantics::NaturallyIdempotent));
    assert_eq!(semantics("focusPane"), Some(MutationSemantics::NaturallyIdempotent));
    assert_eq!(semantics("resize"), Some(MutationSemantics::NaturallyIdempotent));
    assert_eq!(
        semantics("sendInput"),
        Some(MutationSemantics::IdempotencyKeyRequired),
        "replaying bytes to a process stdin requires an application idempotency key"
    );
    assert_eq!(
        semantics("createWindow"),
        Some(MutationSemantics::IdempotencyKeyRequired)
    );
    assert_eq!(
        semantics("pollStdin"),
        Some(MutationSemantics::TransactionLedgerRequired),
        "pollStdin drains the viewer stdin queue (pop_front): a lost reply \
         consumes bytes a retry cannot redeliver, so at-most-once delivery \
         requires an atomic result ledger or fencing — required semantics; \
         no ledger machinery is implemented"
    );
    // Pure state reads stay policy-free.
    assert_eq!(semantics("listWindows"), None);
    assert_eq!(semantics("snapshot"), None);
}

/// Metrics' read-scope leaves follow their real handler effects: `queryStream`
/// prepares a server-side third-party interop stream under the client's
/// ephemeral pubkey and schedules the query continuation before the reply is
/// observed, while `query` executes a structured read synchronously. Raw SQL
/// remains a wire-compatible field but is rejected before storage preparation.
#[test]
fn metrics_query_leaves_declare_only_real_effect_semantics() {
    use policy::MutationSemantics;

    let rows = policy::collect_generated_rows().expect("inventory collects");
    let semantics = |leaf: &str| {
        rows.iter()
            .find(|row| row.service == "metrics" && row.symbolic_path == leaf)
            .unwrap_or_else(|| panic!("missing metrics:{leaf}"))
            .mutation_semantics
    };
    assert_eq!(
        semantics("queryStream"),
        Some(MutationSemantics::TransactionLedgerRequired),
        "a replayed queryStream duplicates the interop-stream preparation and \
         scheduled continuation"
    );
    assert_eq!(semantics("query"), None, "structured query is a genuine read");
    // Genuine reads stay policy-free.
    assert_eq!(semantics("listViews"), None);
}

/// The at9p admission path returns the durably fenced accepted head on an
/// exact replay. Its atomic watermark/conditional-advance protocol therefore
/// satisfies the transaction-ledger contract, rather than merely relying on
/// natural state convergence.
#[test]
fn registry_at9p_admission_declares_its_existing_transaction_ledger() {
    use policy::MutationSemantics;

    let rows = policy::collect_generated_rows().expect("inventory collects");
    let ingest = rows
        .iter()
        .find(|row| row.service == "registry" && row.symbolic_path == "ingestAt9pCandidate")
        .expect("registry.ingestAt9pCandidate row present");
    assert_eq!(
        ingest.mutation_semantics,
        Some(MutationSemantics::TransactionLedgerRequired)
    );
}

/// Fid lifecycle calls preserve their existing query authorization while
/// declaring their real session effects. Auto-allocation makes `walk` unsafe
/// to replay without a caller key; releasing an already-gone fid converges.
#[test]
fn registry_stateful_query_leaves_declare_their_effects() {
    use policy::MutationSemantics;

    let rows = policy::collect_generated_rows().expect("inventory collects");
    let semantics = |leaf: &str| {
        rows.iter()
            .find(|row| row.service == "registry" && row.symbolic_path == leaf)
            .unwrap_or_else(|| panic!("missing registry:{leaf}"))
            .mutation_semantics
    };
    assert_eq!(
        semantics("repo.worktree.walk"),
        Some(MutationSemantics::IdempotencyKeyRequired)
    );
    assert_eq!(
        semantics("repo.worktree.clunk"),
        Some(MutationSemantics::NaturallyIdempotent)
    );
}

/// The synthetic model 9P surface has the same bounded fid lifecycle as the
/// registry worktree surface. `walk` can allocate a fresh handle on a retry;
/// `open` is a one-way state transition and `clunk` is a convergent release.
#[test]
fn model_stateful_query_leaves_declare_their_effects() {
    use policy::MutationSemantics;

    let rows = policy::collect_generated_rows().expect("inventory collects");
    let semantics = |leaf: &str| {
        rows.iter()
            .find(|row| row.service == "model" && row.symbolic_path == leaf)
            .unwrap_or_else(|| panic!("missing model:{leaf}"))
            .mutation_semantics
    };
    assert_eq!(
        semantics("fs.walk"),
        Some(MutationSemantics::IdempotencyKeyRequired)
    );
    assert_eq!(semantics("fs.open"), Some(MutationSemantics::NaturallyIdempotent));
    assert_eq!(semantics("fs.clunk"), Some(MutationSemantics::NaturallyIdempotent));
}

/// Scope exemption records an authorization exception, not an implicit
/// read-only classification. The CA-attested key registration leaf is the
/// causal control-plane case: it is convergent but still explicitly declares
/// its mutation behavior, while public authorization checks remain policy-free.
#[test]
fn scope_exempt_mutator_is_explicitly_classified() {
    use policy::MutationSemantics;

    let rows = policy::collect_generated_rows().expect("inventory collects");
    let lookup = |symbolic: &str| {
        rows.iter()
            .find(|row| row.service == "policy" && row.symbolic_path == symbolic)
            .unwrap_or_else(|| panic!("missing policy:{symbolic}"))
    };
    let register = lookup("registerServiceKey");
    assert!(register.scope_action.is_empty() && register.scope_exempt);
    assert_eq!(
        register.mutation_semantics,
        Some(MutationSemantics::NaturallyIdempotent)
    );
    for public in ["check", "checkBatch"] {
        assert!(lookup(public).scope_action.is_empty());
        assert_eq!(lookup(public).mutation_semantics, None, "{public}");
    }
}
