//! In-memory candidate merge composition (epic #1427, issue #1428).
//!
//! [`CandidateComposer`] is a pure-library wrapper over
//! `git2::Repository::merge_trees` that produces **detached** merge
//! candidates — trees and commits materialised into the object database
//! without ever moving HEAD or touching the working tree. This is the M1
//! building block for deterministic sequential candidate composition
//! (M1, M1+2, …): each `merge_trees` step is a pure function of three trees,
//! so the same inputs always yield the same tree OID.
//!
//! # Scope
//!
//! This module deliberately stops at *candidate* materialisation. It does not:
//! - perform network clone/fetch (B4's job),
//! - talk to any forge API (B4),
//! - dispatch workflows (#989),
//! - decide merge authority, or
//! - land/merge a candidate onto a branch.
//!
//! # Design note — detached commits
//!
//! The issue spec names `commit_create_buffer`, but that primitive yields a
//! commit *content buffer* (for signing flows) rather than a `CommitOid`. The
//! return contract here is a real, addressable `CommitOid`, so we use
//! [`git2::Repository::commit`] with `update_ref = None`, which writes the
//! commit into the object database without updating any ref and without
//! moving HEAD — satisfying the "detached commit does not move HEAD"
//! acceptance criterion directly.

use crate::errors::{Git2DBError, Git2DBResult};
use git2::{Index, Oid, Repository, Tree};
use std::fmt;

/// Which side of a three-way merge a conflicted path belongs to.
///
/// A single [`Index`](git2::Index) conflict can carry entries on multiple
/// sides (e.g. a modify/delete conflict has `Ours` and `Theirs` but no
/// `Ancestor`), so one conflict typically yields several [`ConflictEntry`]
/// records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictSide {
    /// The merge-base (common-ancestor) stage.
    Ancestor,
    /// The "ours" stage — the accumulator/tree we are merging into.
    Ours,
    /// The "their" stage — the head currently being incorporated.
    Theirs,
}

impl fmt::Display for ConflictSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConflictSide::Ancestor => f.write_str("ancestor"),
            ConflictSide::Ours => f.write_str("ours"),
            ConflictSide::Theirs => f.write_str("theirs"),
        }
    }
}

/// A single conflicted path entry captured from [`git2::Index::conflicts`].
///
/// Git paths are raw bytes (conventionally UTF-8, but not guaranteed to be),
/// so the entry stores both the exact [`path_bytes`](Self::path_bytes) — used
/// for identity/dedup — and a lossy [`path`](Self::path) display string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictEntry {
    /// Raw repository-relative path bytes, as libgit2 reports them.
    pub path_bytes: Vec<u8>,
    /// Lossy UTF-8 display form of [`path_bytes`](Self::path_bytes). Two
    /// distinct non-UTF-8 paths may share this display string, so use
    /// [`path_bytes`](Self::path_bytes) for identity comparisons.
    pub path: String,
    /// Which stage of the three-way merge this entry represents.
    pub side: ConflictSide,
}

/// Structured capture of merge conflicts from an in-memory `merge_trees`.
///
/// Returned from [`CandidateComposer::merge_trees`] whenever the produced
/// [`git2::Index`] carries conflict stages. This is the structured
/// replacement for the string-only `"Merge conflicts detected"` error raised
/// by the legacy in-place [`RepositoryHandle::merge`][super::RepositoryHandle].
#[derive(Debug, Clone, Default)]
pub struct ConflictReport {
    /// One entry per (conflicted path, present side).
    pub entries: Vec<ConflictEntry>,
}

impl ConflictReport {
    /// Build a report by walking the conflict stages of an index.
    ///
    /// Pure read: does not mutate the index. Returns an empty report when the
    /// index has no conflicts.
    pub fn from_index(index: &Index) -> Git2DBResult<Self> {
        let mut entries = Vec::new();
        for conflict in index.conflicts().map_err(Git2DBError::from)? {
            let conflict = conflict.map_err(Git2DBError::from)?;
            if let Some(a) = conflict.ancestor {
                entries.push(ConflictEntry {
                    path_bytes: a.path.clone(),
                    path: String::from_utf8_lossy(&a.path).into_owned(),
                    side: ConflictSide::Ancestor,
                });
            }
            if let Some(o) = conflict.our {
                entries.push(ConflictEntry {
                    path_bytes: o.path.clone(),
                    path: String::from_utf8_lossy(&o.path).into_owned(),
                    side: ConflictSide::Ours,
                });
            }
            if let Some(t) = conflict.their {
                entries.push(ConflictEntry {
                    path_bytes: t.path.clone(),
                    path: String::from_utf8_lossy(&t.path).into_owned(),
                    side: ConflictSide::Theirs,
                });
            }
        }
        Ok(Self { entries })
    }

    /// `true` when no conflicts were captured.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of **distinct conflicted paths**.
    ///
    /// This is the meaningful "conflict count": a modify/modify conflict on
    /// one file is one conflict, even though it contributes two stage entries
    /// (`Ours` and `Theirs`). For the raw per-side entry count, use
    /// [`entry_count`](Self::entry_count).
    pub fn len(&self) -> usize {
        self.paths().len()
    }

    /// Raw number of `(path, side)` entries captured — one per present stage,
    /// so a single conflicted path typically counts as 2–3.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over the captured entries.
    pub fn iter(&self) -> std::slice::Iter<'_, ConflictEntry> {
        self.entries.iter()
    }

    /// Distinct conflicted paths' display strings, in first-seen order.
    ///
    /// Dedup is by raw [`ConflictEntry::path_bytes`], **not** the lossy
    /// `path` string: two distinct non-UTF-8 git paths that happen to share
    /// a lossy display string are counted separately (a `String`-keyed dedup
    /// would undercount them). The returned display strings may therefore
    /// contain visually-duplicate values; use
    /// [`path_keys`](Self::path_keys) for exact byte identity.
    pub fn paths(&self) -> Vec<&str> {
        let mut seen_keys: Vec<&[u8]> = Vec::new();
        let mut out: Vec<&str> = Vec::new();
        for e in &self.entries {
            if !seen_keys.contains(&e.path_bytes.as_slice()) {
                seen_keys.push(e.path_bytes.as_slice());
                out.push(e.path.as_str());
            }
        }
        out
    }

    /// Distinct conflicted paths' raw byte keys, in first-seen order.
    ///
    /// This is the identity-stable view: two paths are the same path iff
    /// their byte keys are equal, regardless of UTF-8 validity.
    pub fn path_keys(&self) -> Vec<&[u8]> {
        let mut seen: Vec<&[u8]> = Vec::new();
        for e in &self.entries {
            if !seen.contains(&e.path_bytes.as_slice()) {
                seen.push(e.path_bytes.as_slice());
            }
        }
        seen
    }
}

impl fmt::Display for ConflictReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Count distinct paths, not raw stage entries: one conflicted file is
        // one conflict regardless of how many sides are present.
        let paths = self.paths();
        write!(f, "{} conflict(s)", paths.len())?;
        for path in paths.iter().take(8) {
            write!(f, "; {path}")?;
        }
        if paths.len() > 8 {
            write!(f, "; …(+{} more)", paths.len() - 8)?;
        }
        Ok(())
    }
}

impl std::error::Error for ConflictReport {}

/// Result of an in-memory three-way tree merge.
///
/// `tree` is [`None`] when [`ConflictReport`] is non-empty — libgit2 refuses
/// to write a tree from an index that still carries conflict stages.
#[derive(Debug, Clone)]
pub struct MergeTreesResult {
    /// Materialised tree OID, present only for a clean (conflict-free) merge.
    pub tree: Option<Oid>,
    /// Structured conflict capture; empty for a clean merge.
    pub conflicts: ConflictReport,
}

impl MergeTreesResult {
    /// `true` if the merge produced any conflicts.
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }
}

/// A detached candidate commit + its tree, produced by
/// [`CandidateComposer::compose_candidate`].
///
/// The commit lives in the object database but is referenced by no ref; HEAD
/// is unchanged. Callers land the candidate explicitly (out of scope here).
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    /// OID of the detached commit object.
    pub commit: Oid,
    /// OID of the candidate's tree.
    pub tree: Oid,
}

/// In-memory, HEAD-preserving merge composition.
///
/// Wraps a borrowed [`git2::Repository`] and exposes the small surface needed
/// to compose sequential merge candidates without mutating HEAD or the working
/// tree. Obtain a `&Repository` via the existing escape hatch
/// ([`RepositoryHandle::open_repo`][super::RepositoryHandle::open_repo]) and
/// construct a composer over it:
///
/// ```rust,no_run
/// # use git2db::merge::CandidateComposer;
/// # fn example(
/// #     repo: &git2::Repository,
/// #     oid_a: git2::Oid,
/// #     oid_b: git2::Oid,
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let composer = CandidateComposer::new(repo);
/// let mb = composer.merge_base(oid_a, oid_b)?;
/// # Ok(())
/// # }
/// ```
pub struct CandidateComposer<'a> {
    repo: &'a Repository,
}

impl<'a> CandidateComposer<'a> {
    /// Wrap a borrowed repository.
    pub fn new(repo: &'a Repository) -> Self {
        Self { repo }
    }

    /// Compute the best common ancestor of two commits (`git merge-base`).
    ///
    /// Thin, pure-read wrapper over [`git2::Repository::merge_base`]. Returns
    /// [`Git2DBError::InvalidOperation`] when the two commits share no history
    /// (libgit2's `NotFound`), matching the "no common ancestor" semantics of
    /// [`merge_bases`](Self::merge_bases).
    pub fn merge_base(&self, one: Oid, two: Oid) -> Git2DBResult<Oid> {
        match self.repo.merge_base(one, two) {
            Ok(oid) => Ok(oid),
            Err(e) if e.code() == git2::ErrorCode::NotFound => Err(Git2DBError::invalid_operation(
                format!("no merge base found for {one} and {two} (unrelated histories)"),
            )),
            Err(e) => Err(Git2DBError::internal(format!("merge_base failed: {e}"))),
        }
    }

    /// Compute all merge bases of two commits (`git merge-base --all`).
    ///
    /// Thin, pure-read wrapper over [`git2::Repository::merge_bases`]. Returns
    /// an empty `Vec` when the commits share no history (libgit2's `NotFound`
    /// is mapped to the empty set, since "no common ancestor" is a well-defined
    /// answer rather than an error here).
    pub fn merge_bases(&self, one: Oid, two: Oid) -> Git2DBResult<Vec<Oid>> {
        match self.repo.merge_bases(one, two) {
            Ok(arr) => Ok(arr.iter().copied().collect()),
            Err(e) if e.code() == git2::ErrorCode::NotFound => Ok(Vec::new()),
            Err(e) => Err(Git2DBError::internal(format!("merge_bases failed: {e}"))),
        }
    }

    /// Three-way merge of trees, fully in memory.
    ///
    /// Wraps [`git2::Repository::merge_trees`] → [`git2::Index`] → tree OID.
    /// On conflict, `tree` is [`None`] and `conflicts` carries a structured
    /// [`ConflictReport`] (path + side), replacing the legacy string-only
    /// `"Merge conflicts detected"` error.
    ///
    /// Neither HEAD nor the working tree is touched.
    pub fn merge_trees(
        &self,
        ancestor: &Tree<'_>,
        ours: &Tree<'_>,
        theirs: &Tree<'_>,
    ) -> Git2DBResult<MergeTreesResult> {
        let mut index = self
            .repo
            .merge_trees(ancestor, ours, theirs, None)
            .map_err(|e| Git2DBError::internal(format!("merge_trees failed: {e}")))?;
        let conflicts = ConflictReport::from_index(&index)?;
        let tree = if conflicts.is_empty() {
            // `write_tree_to` materialises against the owning repo's object
            // database — the in-memory index from `merge_trees` is not the
            // repo's working index, so be explicit about the destination.
            Some(
                index
                    .write_tree_to(self.repo)
                    .map_err(|e| Git2DBError::internal(format!("write_tree failed: {e}")))?,
            )
        } else {
            None
        };
        Ok(MergeTreesResult { tree, conflicts })
    }

    /// Deterministically compose a sequential candidate from `base` + `heads`.
    ///
    /// Performs an octopus-style cumulative merge: starting from `base`'s tree
    /// as the accumulator, each head is folded in with a three-way
    /// `merge_trees` step whose **ancestor is the per-head merge base** —
    /// `merge_base(base, head)` — rather than `base` itself. Computing the
    /// ancestor per head is what keeps a *stale* head (one branched off an
    /// older base) from silently dropping changes that only exist in the
    /// supplied `base`: libgit2 only interprets a side as "deleted" when the
    /// ancestor had the content, so using each head's true divergence point
    /// preserves base-only additions.
    ///
    /// Because each `merge_trees` step is a pure function of its three trees,
    /// the resulting tree OID is stable across reruns for the same inputs.
    ///
    /// On success, writes a **detached** commit — referenced by no ref, HEAD
    /// unchanged — whose parents are `base` **plus** the `heads`. Including
    /// `base` as a parent makes the candidate a descendant of `base`, so a
    /// branch at `base` can fast-forward onto the candidate.
    ///
    /// # Errors
    ///
    /// Returns [`Git2DBError::MergeConflictReport`] if any step conflicts,
    /// carrying the structured [`ConflictReport`] for that step. Returns
    /// [`Git2DBError::InvalidOperation`] if `heads` is empty, or if a head
    /// shares no merge base with `base` (unrelated history).
    pub fn compose_candidate(
        &self,
        base: Oid,
        heads: &[Oid],
        message: &str,
    ) -> Git2DBResult<Candidate> {
        if heads.is_empty() {
            return Err(Git2DBError::invalid_operation(
                "compose_candidate requires at least one head",
            ));
        }

        // Resolve the base commit + tree up front; base is also preserved as
        // the first parent of the candidate so the result descends from it.
        let base_commit = self.commit_of(base)?;
        let base_tree = base_commit.tree()?;

        // Resolve each head's tree and its *per-head* merge-base tree with
        // `base`. Using the true divergence point as the ancestor (rather than
        // `base_tree`) is what prevents a stale head from dropping base-only
        // changes — see the method doc.
        let mut head_trees: Vec<Tree<'a>> = Vec::with_capacity(heads.len());
        let mut head_ancestors: Vec<Tree<'a>> = Vec::with_capacity(heads.len());
        let mut head_commits: Vec<git2::Commit<'a>> = Vec::with_capacity(heads.len());
        for &h in heads {
            let commit = self.commit_of(h)?;
            head_trees.push(commit.tree()?);
            // The per-head merge base is the ancestor for this step. A NotFound
            // (unrelated history) surfaces as InvalidOperation via merge_base.
            let ancestor_oid = self.merge_base(base, h)?;
            head_ancestors.push(self.tree_of(ancestor_oid)?);
            head_commits.push(commit);
        }

        // Sequential fold: `accumulator` starts at base's tree and absorbs
        // each head using that head's own merge-base tree as the ancestor.
        let mut accumulator = base_tree.clone();
        for (head_tree, head_ancestor) in head_trees.iter().zip(head_ancestors.iter()) {
            let step = self.merge_trees(head_ancestor, &accumulator, head_tree)?;
            match step.tree {
                Some(t) => accumulator = self.repo.find_tree(t).map_err(Git2DBError::from)?,
                None => return Err(Git2DBError::MergeConflictReport(step.conflicts)),
            }
        }

        // Detached commit: update_ref = None → no ref update, no HEAD move.
        // Parents = [base, ...heads]: base is preserved in ancestry so the
        // candidate is a descendant of base (fast-forwardable).
        // Fall back to a deterministic identity when the repo has no
        // configured signature (e.g. a fresh test repository).
        let sig = self
            .repo
            .signature()
            .or_else(|_| git2::Signature::now("git2db-merge", "git2db@local"))
            .map_err(|e| Git2DBError::internal(format!("signature resolution failed: {e}")))?;
        let mut parents: Vec<&git2::Commit<'_>> = Vec::with_capacity(head_commits.len() + 1);
        parents.push(&base_commit);
        for hc in &head_commits {
            parents.push(hc);
        }
        let commit_oid = self
            .repo
            .commit(None, &sig, &sig, message, &accumulator, &parents)
            .map_err(|e| Git2DBError::internal(format!("detached commit failed: {e}")))?;

        Ok(Candidate {
            commit: commit_oid,
            tree: accumulator.id(),
        })
    }

    fn tree_of(&self, oid: Oid) -> Git2DBResult<Tree<'a>> {
        self.repo
            .find_tree(oid)
            .or_else(|_| {
                // Treat `oid` as a commit-ish and peel to its tree.
                let commit = self.repo.find_commit(oid)?;
                commit.tree()
            })
            .map_err(|e| {
                Git2DBError::reference(oid.to_string(), format!("not a tree or commit OID: {e}"))
            })
    }

    fn commit_of(&self, oid: Oid) -> Git2DBResult<git2::Commit<'a>> {
        self.repo
            .find_commit(oid)
            .map_err(|e| Git2DBError::reference(oid.to_string(), format!("not a commit: {e}")))
    }
}

#[cfg(test)]
mod tests {
    //! Internal sanity tests; the causal acceptance suite lives in
    //! `tests/merge.rs`.

    use super::*;

    #[test]
    fn conflict_report_default_is_empty() {
        let r = ConflictReport::default();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.paths().is_empty());
        assert!(r.path_keys().is_empty());
    }

    /// Helper: build an entry from raw path bytes (lossy display derived).
    fn entry(bytes: &[u8], side: ConflictSide) -> ConflictEntry {
        ConflictEntry {
            path_bytes: bytes.to_vec(),
            path: String::from_utf8_lossy(bytes).into_owned(),
            side,
        }
    }

    #[test]
    fn conflict_report_paths_dedups() {
        let r = ConflictReport {
            entries: vec![
                entry(b"a", ConflictSide::Ours),
                entry(b"a", ConflictSide::Theirs),
                entry(b"b", ConflictSide::Ours),
            ],
        };
        // len() counts distinct conflicted paths (2), not stage entries (3).
        assert_eq!(r.len(), 2);
        assert_eq!(r.entry_count(), 3);
        assert_eq!(r.paths(), vec!["a", "b"]);
        assert_eq!(r.path_keys(), vec![b"a".as_slice(), b"b".as_slice()]);
    }

    #[test]
    fn conflict_report_dedups_by_raw_bytes_not_lossy_string() {
        // Two distinct non-UTF-8 byte paths that both lossy-convert to the
        // same display string (one U+FFFD). A String-keyed dedup would
        // collapse them and undercount; byte-keyed dedup keeps them apart.
        let lossy_a = String::from_utf8_lossy(b"\x80").into_owned();
        let lossy_b = String::from_utf8_lossy(b"\xfe").into_owned();
        assert_eq!(
            lossy_a, lossy_b,
            "test precondition: both lossy strings must be equal"
        );
        let r = ConflictReport {
            entries: vec![
                entry(b"\x80", ConflictSide::Ours),
                entry(b"\xfe", ConflictSide::Theirs),
            ],
        };
        assert_eq!(
            r.len(),
            2,
            "distinct raw-byte paths must not collide via the lossy String"
        );
        assert_eq!(r.entry_count(), 2);
        assert_eq!(r.path_keys().len(), 2);
        // paths() returns one display representative per distinct byte key,
        // so its length matches the byte-keyed count even though the two
        // display strings are visually identical.
        assert_eq!(r.paths().len(), 2);
    }

    #[test]
    fn conflict_side_display() {
        assert_eq!(ConflictSide::Ancestor.to_string(), "ancestor");
        assert_eq!(ConflictSide::Ours.to_string(), "ours");
        assert_eq!(ConflictSide::Theirs.to_string(), "theirs");
    }
}
