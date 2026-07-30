//! Integration tests for `git2db::merge::CandidateComposer` (issue #1428).
//!
//! Causal acceptance suite — each test names the behaviour it pins:
//!  1. clean three-way merge → conflict-free tree,
//!  2. conflicting merge → structured `ConflictReport` (path + side),
//!  3. sequential composition of 3 heads → deterministic, stable tree OID,
//!  4. detached candidate commit does not move HEAD.
//!
//! All fixtures are built directly with `git2` against a `tempfile::TempDir`,
//! mirroring the convention in `v2_core_operations.rs`. No network, no
//! registry — `CandidateComposer` is a pure library wrapping `merge_trees`.

use git2::{Oid, Repository, Signature};
use git2db::merge::{Candidate, CandidateComposer, ConflictSide};
use tempfile::TempDir;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Deterministic identity so commits are reproducible regardless of host git
/// config. Same author/committer/time-shape (via `now`) keeps tree OIDs
/// independent of commit metadata — what we assert on.
fn sig() -> Result<Signature<'static>> {
    Ok(Signature::now(
        "git2db-merge-tests",
        "merge-tests@git2db.local",
    )?)
}

/// Build a flat tree from `files` (path → contents) using blobs only — never
/// touches the working directory, so in-memory merges stay hermetic. All test
/// fixtures use single-segment filenames.
fn write_tree<'a>(repo: &'a Repository, files: &[(&'a str, &'a str)]) -> Result<git2::Tree<'a>> {
    let mut builder = repo.treebuilder(None)?;
    for &(path, contents) in files {
        let blob_oid = repo.blob(contents.as_bytes())?;
        builder.insert(path, blob_oid, 0o100644)?;
    }
    let tree_oid = builder.write()?;
    Ok(repo.find_tree(tree_oid)?)
}

/// Commit `tree` under `message` with the given parents, updating HEAD only
/// when `update_head` is true. Returns the commit OID.
fn commit(
    repo: &Repository,
    message: &str,
    tree: &git2::Tree<'_>,
    parents: &[&git2::Commit<'_>],
    update_head: bool,
) -> Result<Oid> {
    let sig = sig()?;
    Ok(repo.commit(
        if update_head { Some("HEAD") } else { None },
        &sig,
        &sig,
        message,
        tree,
        parents,
    )?)
}

/// Build a trivial base repo with one initial commit and return
/// `(repo, base_commit_oid)`.
fn base_repo() -> Result<(TempDir, Repository, Oid)> {
    let dir = TempDir::new()?;
    let repo = Repository::init(dir.path())?;
    let oid = {
        let tree = write_tree(&repo, &[("README.md", "# base\n")])?;
        commit(&repo, "base", &tree, &[], true)?
    };
    Ok((dir, repo, oid))
}

/// Resolve `refname` to a commit OID.
fn commit_oid(repo: &Repository, refname: &str) -> Result<Oid> {
    Ok(repo.revparse_single(refname)?.peel_to_commit()?.id())
}

/// Collect the entry names of a tree, sorted. Fails on entries without a name.
fn sorted_names(tree: &git2::Tree<'_>) -> Result<Vec<String>> {
    let mut names: Vec<String> = Vec::new();
    for entry in tree.iter() {
        let name = entry.name().ok_or("tree entry without a name")?;
        names.push(name.to_owned());
    }
    names.sort_unstable();
    Ok(names)
}

#[test]
fn clean_merge_yields_conflict_free_tree() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;

    // Two divergent, non-overlapping edits off `base`.
    let ours_tree = write_tree(&repo, &[("README.md", "# base\n"), ("ours.txt", "ours\n")])?;
    let theirs_tree = write_tree(
        &repo,
        &[("README.md", "# base\n"), ("theirs.txt", "theirs\n")],
    )?;
    let base_tree = repo.find_commit(base)?.tree()?;

    let composer = CandidateComposer::new(&repo);
    let res = composer.merge_trees(&base_tree, &ours_tree, &theirs_tree)?;
    assert!(
        !res.has_conflicts(),
        "unexpected conflicts: {:?}",
        res.conflicts
    );
    let tree_oid = res.tree.ok_or("clean merge should yield a tree")?;

    // Both sides' additions must appear in the merged tree.
    let merged = repo.find_tree(tree_oid)?;
    assert_eq!(
        sorted_names(&merged)?,
        vec![
            "README.md".to_owned(),
            "ours.txt".to_owned(),
            "theirs.txt".to_owned()
        ]
    );
    Ok(())
}

#[test]
fn conflicting_merge_returns_structured_report() -> Result<()> {
    let (_dir, repo, _base) = base_repo()?;

    // Both sides edit `conflict.txt` at the same line differently.
    let ours_tree = write_tree(&repo, &[("conflict.txt", "ours-version\n")])?;
    let theirs_tree = write_tree(&repo, &[("conflict.txt", "theirs-version\n")])?;
    let base_tree = write_tree(&repo, &[("conflict.txt", "base-version\n")])?;

    let composer = CandidateComposer::new(&repo);
    let res = composer.merge_trees(&base_tree, &ours_tree, &theirs_tree)?;
    // merge_trees does not error on conflict — it reports structurally.

    // A modify/modify conflict yields one entry per present stage: both Ours
    // and Theirs carry `conflict.txt`. No tree is written on conflict.
    assert!(res.has_conflicts(), "expected a conflict report");
    assert!(res.tree.is_none(), "no tree should be written on conflict");

    let ours_entry = res
        .conflicts
        .iter()
        .find(|e| e.path == "conflict.txt" && e.side == ConflictSide::Ours);
    let theirs_entry = res
        .conflicts
        .iter()
        .find(|e| e.path == "conflict.txt" && e.side == ConflictSide::Theirs);
    assert!(
        ours_entry.is_some(),
        "expected an Ours entry for conflict.txt, got {:?}",
        res.conflicts.entries
    );
    assert!(
        theirs_entry.is_some(),
        "expected a Theirs entry for conflict.txt, got {:?}",
        res.conflicts.entries
    );
    assert_eq!(res.conflicts.paths(), vec!["conflict.txt"]);
    Ok(())
}

#[test]
fn sequential_composition_is_deterministic_across_reruns() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;

    // Three heads, each adding a distinct file off `base`.
    let base_commit = repo.find_commit(base)?;
    let h1 = commit(
        &repo,
        "h1",
        &write_tree(&repo, &[("README.md", "# base\n"), ("a.txt", "1\n")])?,
        &[&base_commit],
        false,
    )?;
    let h2 = commit(
        &repo,
        "h2",
        &write_tree(&repo, &[("README.md", "# base\n"), ("b.txt", "2\n")])?,
        &[&base_commit],
        false,
    )?;
    let h3 = commit(
        &repo,
        "h3",
        &write_tree(&repo, &[("README.md", "# base\n"), ("c.txt", "3\n")])?,
        &[&base_commit],
        false,
    )?;

    let compose = || -> Result<Candidate> {
        let composer = CandidateComposer::new(&repo);
        Ok(composer.compose_candidate(base, &[h1, h2, h3], "compose 3 heads")?)
    };

    let first = compose()?;
    let second = compose()?;

    // Stable tree identity: same inputs ⇒ same OID across reruns.
    assert_eq!(
        first.tree, second.tree,
        "sequential compose must be deterministic"
    );

    // All three additions must be present in the composed tree.
    let tree = repo.find_tree(first.tree)?;
    assert_eq!(
        sorted_names(&tree)?,
        vec![
            "README.md".to_owned(),
            "a.txt".to_owned(),
            "b.txt".to_owned(),
            "c.txt".to_owned()
        ]
    );

    // The detached commit must be addressable and carry all heads as parents.
    let commit_obj = repo.find_commit(first.commit)?;
    assert_eq!(commit_obj.parent_count(), 3, "parents == heads");
    assert_eq!(commit_obj.tree()?.id(), first.tree);
    Ok(())
}

#[test]
fn detached_candidate_does_not_move_head() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;

    let head_before = commit_oid(&repo, "HEAD")?;

    let head = commit(
        &repo,
        "head",
        &write_tree(&repo, &[("README.md", "# base\n"), ("x.txt", "x\n")])?,
        &[&repo.find_commit(base)?],
        false,
    )?;

    let composer = CandidateComposer::new(&repo);
    let candidate = composer.compose_candidate(base, &[head], "compose single head")?;

    // The candidate commit exists in the object database…
    assert!(repo.find_commit(candidate.commit).is_ok());

    // …but HEAD is unchanged.
    let head_after = commit_oid(&repo, "HEAD")?;
    assert_eq!(
        head_before, head_after,
        "detached candidate must not move HEAD"
    );
    Ok(())
}

#[test]
fn compose_candidate_rejects_empty_heads() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;
    let composer = CandidateComposer::new(&repo);
    let err = match composer.compose_candidate(base, &[], "no heads") {
        Ok(c) => return Err(format!("expected an error, got candidate {c:?}").into()),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("compose_candidate requires at least one head"),
        "expected empty-heads error, got: {err}"
    );
    Ok(())
}
