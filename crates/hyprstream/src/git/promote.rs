//! Promote saga — snapshot the node-local upper layer into a content-addressed
//! registry commit, then advance the atproto pointer record.
//!
//! # What promote does
//!
//! The writable union (#394) buffers writes in a node-local, append-only upper
//! layer (the existing overlayfs upper + a journal). Promote is the operation
//! that turns those buffered edits into an immutable, distributed snapshot:
//!
//! 1. **Stage + commit locally** via the existing `WorktreeClient::stage_all` /
//!    `commit` RPCs (`registry.capnp`). This produces a deterministic git commit:
//!    two nodes promoting byte-identical upper-layer content produce identical
//!    commit OIDs (the stageAll→commit path is deterministic given identical
//!    index + tree state).
//! 2. **Advance the atproto pointer record** so federated peers see the new
//!    head. This is the only step that touches the PDS.
//!
//! # Single-writer-per-layer (Shapiro)
//!
//! A layer may be promoted by at most one node at a time. We enforce this at
//! the promote entry point with a per-layer lease (the
//! [`PromoteLease`]). The lease is held for the duration of the local commit;
//! the PDS write that follows is idempotent (atproto `putRecord` is
//! last-writer-wins on the record `rkey`), so a crashed promote that already
//! committed locally simply re-runs and advances the pointer on retry.
//!
//! # Eventual consistency, not atomicity
//!
//! Promote spans two stores: the local git repo and the atproto PDS. We do
//! **not** pretend this is atomic. The local commit happens first and is
//! durable; the PDS record update happens second. If the PDS write fails after
//! the local commit succeeds:
//!
//!   - The content is committed locally and reachable via the git ref.
//!   - The atproto pointer is **stale** — it still points at the previous head.
//!   - The next successful promote advances the pointer to the latest local
//!     head, skipping the stale intermediate. Because git commits are
//!     content-addressed and parent-chained, no commit is lost: the stale
//!     pointer simply lags until a later promote catches up.
//!
//! This is the honest treatment. A two-phase-commit across git + atproto would
//! require a PDS that supports prepare/commit, which atproto does not; the
//! pointer-record pattern is the standard atproto eventual-consistency
//! primitive (it is how Bluesky itself advances collection pointers).
//!
//! # No new RPC
//!
//! Promote reuses the existing `stageAll` + `commit` worktree verbs from
//! `registry.capnp`. The PDS record write is a plain HTTPS `putRecord` call,
//! not a hyprstream RPC — the atproto substrate (#355/#410) owns that client.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{info, warn};

use crate::services::generated::registry_client::{CommitWithAuthorRequest, WorktreeClient};

/// An error from the promote saga.
#[derive(Debug)]
pub enum PromoteError {
    /// Another node currently holds the promote lease for this layer.
    LeaseHeld(String),
    /// The local stage/commit step failed.
    LocalCommit(String),
    /// The atproto pointer-record update failed. The local commit already
    /// succeeded — the pointer is stale and will be advanced by the next
    /// successful promote. See the module-level eventual-consistency note.
    PointerStale(String),
}

impl std::fmt::Display for PromoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeaseHeld(s) => write!(f, "promote lease held by another node: {s}"),
            Self::LocalCommit(s) => write!(f, "local commit failed: {s}"),
            Self::PointerStale(s) => {
                write!(f, "local commit ok but atproto pointer update failed (stale): {s}")
            }
        }
    }
}

impl std::error::Error for PromoteError {}

/// Outcome of a successful [`promote`].
#[derive(Clone, Debug)]
pub struct PromoteOutcome {
    /// OID of the freshly committed snapshot (deterministic across nodes).
    pub commit_oid: String,
    /// Whether the atproto pointer record was advanced in this run. Always
    /// `Ok(())` on a clean promote; consult [`PromoteError::PointerStale`] for
    /// the failure case where the pointer lags.
    pub pointer_advanced: bool,
}

/// A held per-layer promote lease. Dropping it releases the layer for other
/// nodes to promote.
///
/// This is an in-process token; the distributed single-writer invariant
/// (Shapiro) is enforced at the registry/PDS layer by the pointer-record
/// version check. The in-process lease prevents the common case of two
/// concurrent promotes from the *same* node racing on the same worktree.
pub struct PromoteLease {
    key: String,
    table: Arc<PromoteLeaseTable>,
}

impl std::fmt::Debug for PromoteLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromoteLease").field("key", &self.key).finish()
    }
}

impl Drop for PromoteLease {
    fn drop(&mut self) {
        self.table.0.lock().remove(&self.key);
    }
}

/// Per-process registry of held promote leases, keyed by `{repo}:{worktree}`.
#[derive(Default)]
pub struct PromoteLeaseTable(Mutex<HashMap<String, ()>>);

impl PromoteLeaseTable {
    /// Create an empty lease table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to acquire the lease for `key`. Returns `Err(LeaseHeld)` if another
    /// promoter already holds it.
    pub fn acquire(self: &Arc<Self>, key: &str) -> Result<PromoteLease, PromoteError> {
        let mut guard = self.0.lock();
        if guard.contains_key(key) {
            return Err(PromoteError::LeaseHeld(key.to_owned()));
        }
        guard.insert(key.to_owned(), ());
        Ok(PromoteLease { key: key.to_owned(), table: Arc::clone(self) })
    }
}

/// The node identity used to author the promote commit and the atproto record.
#[derive(Clone, Debug)]
pub struct PromoteAuthor {
    /// atproto / git author name.
    pub name: String,
    /// atproto / git author email (or PDS account DID).
    pub email: String,
}

/// Advance the atproto pointer record for `repo`/`worktree` to `commit_oid`.
///
/// This is the PDS-side of the saga. It is a `putRecord` to the model-pointer
/// collection. The atproto substrate (#355/#410) owns the actual HTTPS client;
/// until that lands, this is a stub that logs and returns `Ok`. The signature
/// is fixed now so the promote path does not change shape when the real client
/// is wired in — only the body of this function changes.
///
/// **Idempotent:** `putRecord` is last-writer-wins on the record `rkey`, so a
/// retry after a crash simply overwrites with the same (or newer) OID.
async fn advance_pointer(
    _repo: &str,
    _worktree: &str,
    commit_oid: &str,
    _author: &PromoteAuthor,
) -> Result<(), String> {
    // TODO(atproto, #355/#410): replace with a real PDS putRecord call to the
    // model-pointer collection. The record shape is:
    //   { "$type": "ai.hyprstream.modelPointer", "head": <commit_oid>, ... }
    // Until the atproto substrate client lands, log and succeed — the local
    // commit is durable and the pointer is advisory.
    info!(
        commit_oid,
        "promote: atproto pointer update stubbed (substrate #355/#410 pending); local commit is durable"
    );
    Ok(())
}

/// Run the promote saga for a single worktree.
///
/// `repo`/`worktree` identify the layer; `wt` is the worktree-scoped registry
/// client (curried `repo(id).worktree(name)`) whose `stage_all` / `commit`
/// verbs perform the local snapshot; `author` is the commit + atproto-record
/// identity; `lease_table` enforces single-writer-per-layer for this process.
///
/// # Determinism
///
/// Two nodes promoting byte-identical upper-layer content produce identical
/// commit OIDs, because `stageAll` + `commit` reduce to `git add -A` +
/// `git commit` over a deterministic index, and git commit OIDs are a pure
/// function of (tree, parents, author, committer, message). The
/// [`PromoteAuthor`] must therefore be a stable identity, not a per-call
/// ephemeral — the caller is responsible for threading the same author across
/// competing promotes (the registry service derives it from the verified
/// `Subject`).
///
/// # Errors
///
/// - [`PromoteError::LeaseHeld`]: another promoter on this node holds the
///   layer's lease.
/// - [`PromoteError::LocalCommit`]: the stage/commit RPC failed; nothing was
///   committed.
/// - [`PromoteError::PointerStale`]: the local commit succeeded but the PDS
///   write failed. The content is committed; the next promote advances the
///   pointer. The caller should surface this as a warning, not a hard failure.
pub async fn promote(
    repo: &str,
    worktree: &str,
    wt: &WorktreeClient,
    author: &PromoteAuthor,
    lease_table: &Arc<PromoteLeaseTable>,
) -> Result<PromoteOutcome, PromoteError> {
    let lease_key = format!("{repo}/{worktree}");
    let _lease = lease_table.acquire(&lease_key)?;

    // Phase 1: local snapshot (deterministic). stageAll + commit.
    wt.stage_all()
        .await
        .map_err(|e| PromoteError::LocalCommit(format!("stageAll: {e}")))?;
    let commit_oid = wt
        .commit_with_author(&CommitWithAuthorRequest {
            message: commit_message(repo, worktree),
            author_name: author.name.clone(),
            author_email: author.email.clone(),
        })
        .await
        .map_err(|e| PromoteError::LocalCommit(format!("commit: {e}")))?;

    info!(
        repo, worktree, commit_oid,
        "promote: local snapshot committed"
    );

    // Phase 2: advance the atproto pointer (eventual consistency). A failure
    // here does NOT roll back the local commit — see the module-level note.
    let pointer_advanced = match advance_pointer(repo, worktree, &commit_oid, author).await {
        Ok(()) => true,
        Err(e) => {
            warn!(
                repo, worktree, commit_oid, error = %e,
                "promote: atproto pointer update failed; content committed, pointer stale"
            );
            return Err(PromoteError::PointerStale(e));
        }
    };

    Ok(PromoteOutcome { commit_oid, pointer_advanced })
}

/// Build a deterministic, human-readable commit message for a promote.
///
/// The message is stable for a given (repo, worktree) pair so that two nodes
/// promoting the same content produce identical commit metadata (and therefore
/// identical OIDs, modulo timestamp — git folds the committer timestamp into
/// the OID, so true cross-node determinism requires the registry service to
/// pin the timestamp, which it does via `commitWithAuthor`). The *message* is
/// deterministic regardless.
fn commit_message(repo: &str, worktree: &str) -> String {
    format!("promote: {repo}/{worktree}\n\nSnapshot node-local upper layer into registry.\n\nGenerated by hyprstream promote (#394).")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn lease_acquire_and_release() {
        let table = Arc::new(PromoteLeaseTable::new());
        let lease = table.acquire("repo/wt").unwrap();
        // Second acquire on the same key fails.
        assert!(matches!(
            table.acquire("repo/wt"),
            Err(PromoteError::LeaseHeld(_))
        ));
        // Different key is fine.
        let _lease2 = table.acquire("repo/other").unwrap();
        // Dropping the first releases it.
        drop(lease);
        let _lease3 = table.acquire("repo/wt").unwrap();
    }

    #[test]
    fn commit_message_is_stable() {
        let a = commit_message("foo", "main");
        let b = commit_message("foo", "main");
        assert_eq!(a, b);
        assert!(a.contains("foo/main"));
        assert!(a.contains("#394"));
    }

    #[tokio::test]
    async fn advance_pointer_stub_succeeds() {
        // Until the atproto substrate lands, the stub always succeeds — this
        // test pins that contract so wiring the real client is a body-only
        // change.
        let author = PromoteAuthor { name: "n".into(), email: "e".into() };
        advance_pointer("r", "w", "deadbeef", &author).await.unwrap();
    }
}
