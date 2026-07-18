//! T5 registry pilot (epic #539): the generated 9P/VFS projection (Mechanism A)
//! must match the paths the registry's old `$docExample` prose documented, with
//! no change to the hand-written git-worktree FS (`FsHandler`, Mechanism B —
//! not exercised here; it lives in `hyprstream::services::registry`).
//!
//! THE key signal: `generated paths == documented convention`.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use hyprstream_rpc::metadata::{VfsNode, VfsNodeKind};

fn nodes() -> &'static [VfsNode] {
    hyprstream_rpc_std::registry_client::vfs_nodes().1
}

fn find(method: &str) -> &'static VfsNode {
    nodes()
        .iter()
        .find(|n| n.method == method)
        .unwrap_or_else(|| panic!("no generated node for method `{method}`"))
}

/// The pilot's headline assertion: every path the pre-epic `$docExample` prose
/// described is now produced by the generated projection, byte-for-byte.
///
/// Old `$docExample` convention (registry.capnp, pre-epic):
///   list        → `ls /srv/registry`             (dir at `.`)
///   getByName   → `cat /srv/registry/my-model`   (file at `{name}`)
///   clone       → `ctl /srv/registry clone …`    (ctl at `ctl`)
///   healthCheck → `cat /srv/registry/health`     (file at `health`)
#[test]
fn generated_paths_match_docexample_convention() {
    let list = find("list");
    assert_eq!(list.path, ".");
    assert_eq!(list.kind, VfsNodeKind::Dir);

    let by_name = find("get_by_name");
    assert_eq!(by_name.path, "{name}");
    assert_eq!(by_name.kind, VfsNodeKind::File);

    let clone = find("clone");
    assert_eq!(clone.path, "ctl");
    assert_eq!(clone.kind, VfsNodeKind::Ctl);

    let health = find("health_check");
    assert_eq!(health.path, "health");
    assert_eq!(health.kind, VfsNodeKind::File);
}

/// The write/manage verbs share the single `ctl` node — a `ctl /srv/registry
/// <verb>` write dispatches by verb, exactly as the old prose implied.
#[test]
fn write_verbs_share_the_ctl_node() {
    for verb in ["clone", "register", "remove"] {
        let n = find(verb);
        assert_eq!(n.path, "ctl", "`{verb}` must project to the shared ctl node");
        assert_eq!(n.kind, VfsNodeKind::Ctl);
    }
    // …and the query reads must NOT collapse onto ctl.
    assert_ne!(find("list").path, "ctl");
    assert_ne!(find("get_by_name").path, "ctl");
}

/// `get` (by id) and `getByName` are distinct query files — before annotation
/// both inferred to the same `{arg}` path and collided; the pilot disambiguates
/// them (`by-id/{id}` vs `{name}`).
#[test]
fn get_and_get_by_name_do_not_collide() {
    let by_id = find("get");
    let by_name = find("get_by_name");
    assert_eq!(by_id.kind, VfsNodeKind::File);
    assert_eq!(by_name.kind, VfsNodeKind::File);
    assert_ne!(
        by_id.path, by_name.path,
        "the two read-by-key methods must occupy distinct paths"
    );
    assert_eq!(by_name.path, "{name}");
}

/// The `repo` scoped client projects as a `repo/{repoId}/` subdirectory with
/// its own recursively-generated node table exposing `status`, `head`, `ctl`
/// verbs, and the branch/tag/remote listings — not a flat leaf. The subdir is
/// NOT a top-level node; it's a separate table carrying its mount-relative
/// prefix (from the `repo` variant's `$vfsPath`).
#[test]
fn repo_projects_as_a_subdirectory_with_children() {
    // `repo` is a scoped client, so it is NOT in the flat top-level table.
    assert!(
        !nodes().iter().any(|n| n.method == "repo"),
        "scoped client `repo` must not appear as a flat leaf"
    );

    let (prefix, sub) = hyprstream_rpc_std::registry_client::vfs_nodes_repo();
    assert_eq!(prefix, "repo/{repoId}", "subdir carries the $vfsPath prefix");
    let has = |m: &str| sub.iter().any(|n| n.method == m);
    // Representative children the old prose implied under a repo dir.
    assert!(has("status"), "repo subdir must expose `status`");
    assert!(has("get_head"), "repo subdir must expose the head read");
    assert!(has("list_branches"), "repo subdir must expose branch listing");
    assert!(has("commit"), "repo subdir must expose the `commit` ctl verb");

    // The subdir table obeys the same scope→kind soundness as the top level.
    for n in sub {
        let is_read = matches!(n.scope, "query" | "subscribe" | "");
        if is_read {
            assert_ne!(n.kind, VfsNodeKind::Ctl, "repo `{}` read must not be a ctl", n.method);
        }
    }
}

/// Scope→kind soundness holds across the whole generated table: no query/read
/// method projects to a `ctl` (a read that invokes a verb), and no write/manage
/// method projects to a read-only node. This is the guard T2 enforces at
/// compile time — asserted here as a runtime property of the shipped table so a
/// future annotation change can't silently violate it.
#[test]
fn scope_and_kind_are_consistent() {
    for n in nodes() {
        let is_read = matches!(n.scope, "query" | "subscribe" | "");
        if is_read {
            assert_ne!(
                n.kind,
                VfsNodeKind::Ctl,
                "query method `{}` must not be a ctl (read must be side-effect-free)",
                n.method
            );
        } else {
            assert!(
                !matches!(n.kind, VfsNodeKind::File | VfsNodeKind::Dir | VfsNodeKind::Query),
                "write/manage method `{}` must not project to a read-only node",
                n.method
            );
        }
    }
}
