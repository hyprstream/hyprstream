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
/// Paths are decoded lossily from the raw bytes libgit2 reports; git paths
/// are conventionally UTF-8 but are not guaranteed to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictEntry {
    /// Repository-relative path of the conflicted entry (lossy UTF-8).
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
                    path: String::from_utf8_lossy(&a.path).into_owned(),
                    side: ConflictSide::Ancestor,
                });
            }
            if let Some(o) = conflict.our {
                entries.push(ConflictEntry {
                    path: String::from_utf8_lossy(&o.path).into_owned(),
                    side: ConflictSide::Ours,
                });
            }
            if let Some(t) = conflict.their {
                entries.push(ConflictEntry {
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

    /// Number of captured (path, side) entries.
    ///
    /// One conflicted path typically contributes 2–3 entries (one per
    /// present stage); use [`paths`](Self::paths) for a de-duplicated path
    /// count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over the captured entries.
    pub fn iter(&self) -> std::slice::Iter<'_, ConflictEntry> {
        self.entries.iter()
    }

    /// Distinct conflicted paths, in first-seen order.
    pub fn paths(&self) -> Vec<&str> {
        let mut seen = Vec::new();
        for e in &self.entries {
            if !seen.contains(&e.path.as_str()) {
                seen.push(e.path.as_str());
            }
        }
        seen
    }
}

impl fmt::Display for ConflictReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} conflict(s)", self.entries.len())?;
        for e in self.entries.iter().take(8) {
            write!(f, "; {}={}", e.side, e.path)?;
        }
        if self.entries.len() > 8 {
            write!(f, "; …(+{} more)", self.entries.len() - 8)?;
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
/// # fn example(repo: &git2::Repository) -> Result<(), Box<dyn std::error::Error>> {
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
    /// Thin, pure-read wrapper over [`git2::Repository::merge_base`].
    pub fn merge_base(&self, one: Oid, two: Oid) -> Git2DBResult<Oid> {
        self.repo
            .merge_base(one, two)
            .map_err(|e| Git2DBError::internal(format!("merge_base failed: {e}")))
    }

    /// Compute all merge bases of two commits (`git merge-base --all`).
    ///
    /// Thin, pure-read wrapper over [`git2::Repository::merge_bases`]. Returns
    /// an empty `Vec` when the commits share no history.
    pub fn merge_bases(&self, one: Oid, two: Oid) -> Git2DBResult<Vec<Oid>> {
        self.repo
            .merge_bases(one, two)
            .map_err(|e| Git2DBError::internal(format!("merge_bases failed: {e}")))
            .map(|arr| arr.iter().copied().collect())
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
    /// `merge_trees(ancestor = base_tree, ours = accumulator, theirs = head)`
    /// step. Because each step is a pure function of its three trees, the
    /// resulting tree OID is stable across reruns for the same inputs.
    ///
    /// On success, writes a **detached** commit — referenced by no ref, HEAD
    /// unchanged — whose parents are the `heads` (the common base is implied
    /// by the merge bases and is not duplicated as a parent).
    ///
    /// # Errors
    ///
    /// Returns [`Git2DBError::MergeConflictReport`] if any step conflicts,
    /// carrying the structured [`ConflictReport`] for that step. Returns
    /// [`Git2DBError::InvalidOperation`] if `heads` is empty.
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

        // Resolve trees and parent commits up front so a missing head fails
        // fast before any merge work is done.
        let base_tree = self.tree_of(base)?;
        let mut head_trees = Vec::with_capacity(heads.len());
        let mut parent_commits = Vec::with_capacity(heads.len());
        for &h in heads {
            let commit = self.commit_of(h)?;
            head_trees.push(commit.tree()?);
            parent_commits.push(commit);
        }

        // Sequential fold: base_tree is the fixed ancestor for every step
        // (the common merge base), and `accumulator` carries the running
        // composed tree.
        let mut accumulator = base_tree.clone();
        for head_tree in &head_trees {
            let step = self.merge_trees(&base_tree, &accumulator, head_tree)?;
            match step.tree {
                Some(t) => accumulator = self.repo.find_tree(t).map_err(Git2DBError::from)?,
                None => {
                    return Err(Git2DBError::MergeConflictReport(step.conflicts));
                }
            }
        }

        // Detached commit: update_ref = None → no ref update, no HEAD move.
        // Fall back to a deterministic identity when the repo has no
        // configured signature (e.g. a fresh test repository).
        let sig = self
            .repo
            .signature()
            .or_else(|_| git2::Signature::now("git2db-merge", "git2db@local"))
            .map_err(|e| Git2DBError::internal(format!("signature resolution failed: {e}")))?;
        let parent_refs: Vec<&git2::Commit<'_>> = parent_commits.iter().collect();
        let commit_oid = self
            .repo
            .commit(None, &sig, &sig, message, &accumulator, &parent_refs)
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
    }

    #[test]
    fn conflict_report_paths_dedups() {
        let r = ConflictReport {
            entries: vec![
                ConflictEntry {
                    path: "a".into(),
                    side: ConflictSide::Ours,
                },
                ConflictEntry {
                    path: "a".into(),
                    side: ConflictSide::Theirs,
                },
                ConflictEntry {
                    path: "b".into(),
                    side: ConflictSide::Ours,
                },
            ],
        };
        assert_eq!(r.len(), 3);
        assert_eq!(r.paths(), vec!["a", "b"]);
    }

    #[test]
    fn conflict_side_display() {
        assert_eq!(ConflictSide::Ancestor.to_string(), "ancestor");
        assert_eq!(ConflictSide::Ours.to_string(), "ours");
        assert_eq!(ConflictSide::Theirs.to_string(), "theirs");
    }
}
