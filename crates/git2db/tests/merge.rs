//! Integration tests for `git2db::merge::CandidateComposer` (issue #1428).
//!
//! Causal acceptance suite — each test names the behaviour it pins:
//!  1. clean three-way merge → conflict-free tree,
//!  2. conflicting merge → structured `ConflictReport` (path + side),
//!  3. sequential composition of 3 heads → deterministic, stable tree OID,
//!  4. detached candidate commit does not move HEAD,
//!  5. per-head merge base keeps a stale head from dropping base-only changes,
//!  6. candidate descends from base (fast-forwardable),
//!  7. unrelated histories yield an empty merge-base set,
//!  8. conflict count uses distinct paths,
//!  9. stacked/revert heads fold correctly against the accumulated candidate.
//!
//! All fixtures are built directly with `git2` against a `tempfile::TempDir`,
//! mirroring the convention in `v2_core_operations.rs`. No network, no
//! registry — `CandidateComposer` is a pure library wrapping `merge_trees`.

use git2::{Oid, Repository, Signature};
use git2db::merge::{Candidate, CandidateComposer, ConflictSide};
use tempfile::TempDir;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A fixed author/committer identity independent of host git config.
///
/// `Signature::now` embeds the **current wall clock**, so the resulting
/// commit OIDs are *not* stable across runs. Tree OIDs, however, are
/// content-addressed and fully deterministic — so every test asserts on
/// tree content/OID, never on a commit OID's value.
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

    // The detached commit must be addressable and carry base + heads as
    // parents (base preserved so the candidate descends from it).
    let commit_obj = repo.find_commit(first.commit)?;
    assert_eq!(commit_obj.parent_count(), 4, "parents == base + heads");
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

// ---- P1.1: per-head merge base keeps a stale head from dropping base-only
// changes. ----

/// Build a three-commit history so a head can be "stale" relative to base:
///   `root` ── `base` (adds base-only feature `base_only.txt`)
///      └──── `stale` (adds `head.txt`, branched off `root`, lacks the feature)
fn stale_head_fixture() -> Result<(TempDir, Repository, Oid, Oid)> {
    let dir = TempDir::new()?;
    let repo = Repository::init(dir.path())?;
    let root = {
        let t = write_tree(&repo, &[("shared.txt", "root\n")])?;
        commit(&repo, "root", &t, &[], true)?
    };
    let base = {
        let t = write_tree(
            &repo,
            &[("shared.txt", "root\n"), ("base_only.txt", "keep me\n")],
        )?;
        commit(&repo, "base", &t, &[&repo.find_commit(root)?], false)?
    };
    let stale = {
        // Branched off `root` — does NOT carry base_only.txt.
        let t = write_tree(&repo, &[("shared.txt", "root\n"), ("head.txt", "head\n")])?;
        commit(&repo, "stale", &t, &[&repo.find_commit(root)?], false)?
    };
    Ok((dir, repo, base, stale))
}

#[test]
fn stale_head_does_not_drop_base_only_changes() -> Result<()> {
    let (_dir, repo, base, stale) = stale_head_fixture()?;

    let composer = CandidateComposer::new(&repo);
    let candidate = composer.compose_candidate(base, &[stale], "fold stale head")?;

    // The base-only feature must survive: the stale head never touched it, so
    // merging it in must not delete `base_only.txt`. (With a fixed
    // ancestor=base_tree this path would read as "deleted by theirs".)
    let tree = repo.find_tree(candidate.tree)?;
    let names = sorted_names(&tree)?;
    assert!(
        names.contains(&"base_only.txt".to_owned()),
        "base-only feature was dropped by stale head; tree = {names:?}"
    );
    assert!(names.contains(&"head.txt".to_owned()), "head addition lost");
    assert!(names.contains(&"shared.txt".to_owned()), "shared file lost");
    Ok(())
}

// ---- P1.2: base preserved in ancestry → candidate is fast-forwardable. ----

#[test]
fn candidate_descends_from_base_for_fast_forward() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;
    let base_commit = repo.find_commit(base)?;
    let head = commit(
        &repo,
        "head",
        &write_tree(&repo, &[("README.md", "# base\n"), ("y.txt", "y\n")])?,
        &[&base_commit],
        false,
    )?;

    let composer = CandidateComposer::new(&repo);
    let candidate = composer.compose_candidate(base, &[head], "ff-able")?;

    // base must be an ancestor of the candidate (graph_descendant_of(base,
    // candidate) == false ⟺ base is NOT a descendant of candidate; we want the
    // inverse: candidate descends from base).
    assert!(
        repo.graph_descendant_of(candidate.commit, base)?,
        "candidate must descend from base so a branch at base can fast-forward"
    );
    // And base's merge-base with the candidate is base itself.
    let mb = repo.merge_base(base, candidate.commit)?;
    assert_eq!(
        mb, base,
        "base must be the merge base of itself and the candidate"
    );
    Ok(())
}

// ---- P2: merge_bases returns empty for unrelated histories. ----

#[test]
fn merge_bases_empty_for_unrelated_histories() -> Result<()> {
    let dir = TempDir::new()?;
    let repo = Repository::init(dir.path())?;

    // Two independent root commits in the same object database, with no
    // shared ancestry.
    let a = {
        let t = write_tree(&repo, &[("a.txt", "a\n")])?;
        commit(&repo, "a", &t, &[], false)?
    };
    let b = {
        let t = write_tree(&repo, &[("b.txt", "b\n")])?;
        commit(&repo, "b", &t, &[], false)?
    };

    let composer = CandidateComposer::new(&repo);
    let bases = composer.merge_bases(a, b)?;
    assert!(
        bases.is_empty(),
        "unrelated histories must yield an empty merge-base set, got {bases:?}"
    );
    // The singular form surfaces the same case as a typed InvalidOperation.
    let err = composer
        .merge_base(a, b)
        .err()
        .ok_or("expected an error for unrelated merge_base")?;
    assert!(
        err.to_string().contains("no merge base"),
        "expected a 'no merge base' error, got: {err}"
    );
    Ok(())
}

// ---- P3: conflict count counts distinct paths, not stage entries. ----

#[test]
fn conflict_count_uses_distinct_paths() -> Result<()> {
    let (_dir, repo, _base) = base_repo()?;

    // A single modify/modify conflict on `same.txt` produces two stage
    // entries (Ours + Theirs) but is one conflicted path.
    let ours_tree = write_tree(&repo, &[("same.txt", "ours\n")])?;
    let theirs_tree = write_tree(&repo, &[("same.txt", "theirs\n")])?;
    let base_tree = write_tree(&repo, &[("same.txt", "base\n")])?;

    let composer = CandidateComposer::new(&repo);
    let res = composer.merge_trees(&base_tree, &ours_tree, &theirs_tree)?;
    assert!(res.has_conflicts());

    let report = &res.conflicts;
    assert_eq!(
        report.len(),
        1,
        "len() must count distinct conflicted paths, not stage entries"
    );
    // A modify/modify conflict where the ancestor carried the file yields
    // three stage entries (Ancestor + Ours + Theirs); the point is that this
    // raw entry count exceeds the distinct-path count.
    assert!(
        report.entry_count() >= 2,
        "entry_count() must reflect multiple stage entries, got {}",
        report.entry_count()
    );
    assert_eq!(report.paths(), vec!["same.txt"]);

    // Display reports the distinct-path count, not the raw entry count.
    let display = report.to_string();
    assert!(
        display.starts_with("1 conflict(s)"),
        "Display must report 1 conflict, got: {display}"
    );
    Ok(())
}

// ---- Fable P1: fold against the accumulated candidate, not a fixed base.
// A stacked head (built on an earlier head) and a revert head must not
// produce false conflicts or silently lose changes. ----

#[test]
fn stacked_head_folds_without_false_conflict() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;
    let base_commit = repo.find_commit(base)?;

    // head1: adds `stacked.txt` off base.
    let head1 = commit(
        &repo,
        "head1",
        &write_tree(&repo, &[("README.md", "# base\n"), ("stacked.txt", "v1\n")])?,
        &[&base_commit],
        false,
    )?;
    // head2: STACKED on head1, modifies stacked.txt. Its true merge-base
    // with the accumulator (== head1's tree after head1 is folded) is head1,
    // NOT base — base never had stacked.txt.
    let head2 = commit(
        &repo,
        "head2",
        &write_tree(&repo, &[("README.md", "# base\n"), ("stacked.txt", "v2\n")])?,
        &[&repo.find_commit(head1)?],
        false,
    )?;

    let composer = CandidateComposer::new(&repo);
    let candidate = composer.compose_candidate(base, &[head1, head2], "stacked fold")?;

    // With the old fixed-base ancestor this was a false add/add conflict on
    // stacked.txt; the correct fold yields v2 (head2's modification) cleanly.
    let tree = repo.find_tree(candidate.tree)?;
    let entry = tree
        .get_name("stacked.txt")
        .ok_or("stacked.txt missing from result")?;
    let blob = repo.find_blob(entry.id())?;
    assert_eq!(
        blob.content(),
        b"v2\n",
        "stacked head's modification must land"
    );
    Ok(())
}

#[test]
fn revert_head_is_not_silently_lost() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;
    let base_commit = repo.find_commit(base)?;

    // head1: modifies x.txt v1 → v2 (off base; base's x.txt == v1).
    let head1 = commit(
        &repo,
        "head1",
        &write_tree(&repo, &[("README.md", "# base\n"), ("x.txt", "v2\n")])?,
        &[&base_commit],
        false,
    )?;
    // head2: STACKED on head1, reverts x.txt back to v1.
    let head2 = commit(
        &repo,
        "head2",
        &write_tree(&repo, &[("README.md", "# base\n"), ("x.txt", "v1\n")])?,
        &[&repo.find_commit(head1)?],
        false,
    )?;

    let composer = CandidateComposer::new(&repo);
    let candidate = composer.compose_candidate(base, &[head1, head2], "fold revert")?;

    // With the old fixed-base ancestor the revert was lost: merge_trees saw
    // theirs==ancestor(base=v1) and kept ours(=v2). Folding against the
    // accumulated candidate (ancestor=head1=v2) correctly applies the revert.
    let tree = repo.find_tree(candidate.tree)?;
    let entry = tree.get_name("x.txt").ok_or("x.txt missing from result")?;
    let blob = repo.find_blob(entry.id())?;
    assert_eq!(
        blob.content(),
        b"v1\n",
        "revert head must take effect, not be silently lost"
    );
    Ok(())
}

#[test]
fn compose_candidate_rejects_base_in_heads() -> Result<()> {
    let (_dir, repo, base) = base_repo()?;
    let composer = CandidateComposer::new(&repo);
    let err = match composer.compose_candidate(base, &[base], "base as head") {
        Ok(c) => return Err(format!("expected an error, got candidate {c:?}").into()),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("base must not appear in heads"),
        "expected base-in-heads error, got: {err}"
    );
    Ok(())
}
