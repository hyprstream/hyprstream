//! XET content provenance store (the #436 / #509 fix).
//!
//! # Why this exists
//!
//! The XET CAS is **global and content-addressed**: bytes are deduplicated by
//! their computed merkle root with no per-tenant namespacing. The pre-existing
//! `getBlob` authorization (condition (b)) decided entitlement from a git-xet /
//! git-lfs **pointer blob committed into the grant repo's HEAD tree**. That
//! pointer is **caller-committed and therefore forgeable**: Tenant B can commit
//! `oid sha256:<Tenant A's private merkle>` into their own repo and read A's
//! private bytes back over the shared CAS. The pointer-only check structurally
//! cannot tell a *planted* reference from a *genuine* one.
//!
//! # What this provides
//!
//! A persisted, **server-side-only** mapping `merkle_hex → {repo_id, …}` recording
//! which repositories actually *contributed* (uploaded) the bytes behind a merkle.
//! The bytes themselves remain globally deduplicated in the CAS — only this
//! provenance mapping is per-repo. The read gate consults this store and **fails
//! closed**: if no provenance record binds the merkle to the grant repo, the
//! request is denied. A forgeable pointer can never, on its own, grant access.
//!
//! # Trust boundary (load-bearing)
//!
//! `record()` is the *only* mutating entry point and MUST be called exclusively
//! from a **server-authenticated** code path that has already established the
//! pushing repo's identity (NOT from caller-supplied identity such as `$USER` or
//! a request field). See the module-level note in `registry.rs` for the wiring
//! status of the authenticated upload binding (issue #509, part 2).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// On-disk representation. Versioned so the format can evolve.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct ProvenanceFile {
    #[serde(default = "default_version")]
    version: String,
    /// merkle_hex → set of repo_id (UUID strings) that uploaded these bytes.
    #[serde(default)]
    bindings: HashMap<String, HashSet<String>>,
}

fn default_version() -> String {
    "1.0.0".to_owned()
}

/// Persisted `(merkle_hex → {repo_id})` provenance mapping for XET CAS content.
///
/// Lives alongside the registry metadata (`<base_dir>/.registry/`) so it shares
/// the registry's durability and 0600 permission posture.
pub struct XetProvenanceStore {
    path: PathBuf,
    inner: RwLock<HashMap<String, HashSet<String>>>,
}

impl XetProvenanceStore {
    /// File name within the `.registry` directory.
    const FILE_NAME: &'static str = "xet_provenance.json";

    /// Load the provenance store from `<base_dir>/.registry/xet_provenance.json`,
    /// creating an empty in-memory store if the file does not yet exist.
    ///
    /// A missing file is the expected first-run state and yields an **empty**
    /// store — which, combined with the fail-closed read gate, denies all
    /// XetMerkle `getBlob` requests until provenance is recorded. That is the
    /// intended secure default.
    pub async fn load(base_dir: &Path) -> Result<Self> {
        let path = base_dir.join(".registry").join(Self::FILE_NAME);
        let bindings = match tokio::fs::read(&path).await {
            Ok(bytes) => {
                let parsed: ProvenanceFile = serde_json::from_slice(&bytes).with_context(|| {
                    format!("failed to parse XET provenance store at {}", path.display())
                })?;
                debug!(
                    entries = parsed.bindings.len(),
                    path = %path.display(),
                    "loaded XET provenance store"
                );
                parsed.bindings
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                debug!(path = %path.display(), "no XET provenance store yet; starting empty (fail-closed)");
                HashMap::new()
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!("failed to read XET provenance store at {}", path.display())
                });
            }
        };
        Ok(Self {
            path,
            inner: RwLock::new(bindings),
        })
    }

    /// Record that `repo_id` legitimately contributed the bytes behind
    /// `merkle_hex`, persisting the mapping.
    ///
    /// SECURITY: callers MUST have authenticated the pushing repo's identity
    /// server-side before invoking this. Never pass caller-supplied identity.
    /// The mapping is additive (global dedup means the same bytes may be
    /// uploaded by multiple repos; each legitimately gets a binding).
    pub async fn record(&self, merkle_hex: &str, repo_id: &str) -> Result<()> {
        let merkle_hex = merkle_hex.trim().to_ascii_lowercase();
        let repo_id = repo_id.trim().to_owned();
        if merkle_hex.is_empty() || repo_id.is_empty() {
            anyhow::bail!("record provenance denied: empty merkle or repo_id");
        }
        {
            let mut guard = self.inner.write().await;
            let entry = guard.entry(merkle_hex.clone()).or_default();
            if !entry.insert(repo_id.clone()) {
                // Already recorded; nothing to persist.
                return Ok(());
            }
        }
        self.persist().await.with_context(|| {
            format!("failed to persist XET provenance for merkle {merkle_hex}")
        })?;
        debug!(merkle = %merkle_hex, repo = %repo_id, "recorded XET provenance");
        Ok(())
    }

    /// Fail-closed gate check: is `merkle_hex` bound to `repo_id` by a recorded
    /// provenance entry? Returns `false` (deny) when no record exists.
    pub async fn is_bound(&self, merkle_hex: &str, repo_id: &str) -> bool {
        let merkle_hex = merkle_hex.trim().to_ascii_lowercase();
        let repo_id = repo_id.trim();
        if merkle_hex.is_empty() || repo_id.is_empty() {
            return false;
        }
        let guard = self.inner.read().await;
        guard
            .get(&merkle_hex)
            .map(|repos| repos.contains(repo_id))
            .unwrap_or(false)
    }

    /// Atomically persist the in-memory mapping to disk (tmp file + rename),
    /// with 0600 permissions (matches the registry metadata posture).
    async fn persist(&self) -> Result<()> {
        let snapshot = {
            let guard = self.inner.read().await;
            ProvenanceFile {
                version: default_version(),
                bindings: guard.clone(),
            }
        };
        let json = serde_json::to_vec_pretty(&snapshot)
            .context("failed to serialize XET provenance store")?;

        if let Some(parent) = self.path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                warn!(error = %e, dir = %parent.display(), "failed to ensure provenance dir");
            }
        }

        let tmp = self.path.with_extension("json.tmp");
        tokio::fs::write(&tmp, &json)
            .await
            .with_context(|| format!("failed to write provenance tmp {}", tmp.display()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = tokio::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600)).await;
        }

        tokio::fs::rename(&tmp, &self.path)
            .await
            .with_context(|| format!("failed to rename provenance into place {}", self.path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // A 64-hex-char merkle stand-in for "Tenant A's private content".
    const MERKLE_A: &str = "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1";
    const REPO_A: &str = "11111111-1111-1111-1111-111111111111";
    const REPO_B: &str = "22222222-2222-2222-2222-222222222222";

    async fn store_in(dir: &Path) -> XetProvenanceStore {
        tokio::fs::create_dir_all(dir.join(".registry")).await.unwrap();
        XetProvenanceStore::load(dir).await.unwrap()
    }

    /// The core #436 / #509 invariant at the gate layer: a planted pointer with
    /// NO provenance record must be denied (fail-closed). This is the
    /// substantive content behind `oid_field_planted_pointer_must_be_denied`.
    #[tokio::test]
    async fn oid_field_planted_pointer_must_be_denied() {
        let dir = tempdir().unwrap();
        let store = store_in(dir.path()).await;

        // Tenant A genuinely uploaded the bytes → provenance bound to repo A.
        store.record(MERKLE_A, REPO_A).await.unwrap();

        // Tenant B planted a pointer referencing A's merkle but never uploaded
        // the bytes → NO provenance binding for repo B → MUST be denied.
        assert!(
            !store.is_bound(MERKLE_A, REPO_B).await,
            "planted-pointer repo B must NOT be entitled to A's merkle"
        );

        // The legitimate owner (repo A) still resolves.
        assert!(
            store.is_bound(MERKLE_A, REPO_A).await,
            "the repo that genuinely uploaded the merkle must still resolve it"
        );
    }

    /// Fail-closed default: an unknown merkle (no record at all) is denied for
    /// everyone — never falls back to the forgeable pointer check.
    #[tokio::test]
    async fn unknown_merkle_denied_fail_closed() {
        let dir = tempdir().unwrap();
        let store = store_in(dir.path()).await;
        assert!(!store.is_bound(MERKLE_A, REPO_A).await);
        assert!(!store.is_bound(MERKLE_A, REPO_B).await);
    }

    /// Records survive a reload (persistence), and case/whitespace are normalized.
    #[tokio::test]
    async fn provenance_persists_across_reload() {
        let dir = tempdir().unwrap();
        {
            let store = store_in(dir.path()).await;
            store.record(&MERKLE_A.to_ascii_uppercase(), REPO_A).await.unwrap();
        }
        let reloaded = XetProvenanceStore::load(dir.path()).await.unwrap();
        assert!(reloaded.is_bound(MERKLE_A, REPO_A).await);
        assert!(!reloaded.is_bound(MERKLE_A, REPO_B).await);
    }

    /// Global dedup: the same bytes uploaded by two repos bind to both, and each
    /// is entitled independently.
    #[tokio::test]
    async fn shared_bytes_bind_per_repo() {
        let dir = tempdir().unwrap();
        let store = store_in(dir.path()).await;
        store.record(MERKLE_A, REPO_A).await.unwrap();
        store.record(MERKLE_A, REPO_B).await.unwrap();
        assert!(store.is_bound(MERKLE_A, REPO_A).await);
        assert!(store.is_bound(MERKLE_A, REPO_B).await);
    }
}
