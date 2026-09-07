//! Full-leaf derivation and generated method-policy inventory over the REAL
//! production schemas linked into the hyprstream binary (v16 §5.1/§6.1,
//! issue #1504).
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

#![allow(clippy::unwrap_used, clippy::expect_used)]

use hyprstream_core::registry_capnp;
use hyprstream_core::services::generated::registry_client::decode_registry_request_body;
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
