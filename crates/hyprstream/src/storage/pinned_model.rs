//! Local admission and owned artifact acquisition for an exact commit model.

use anyhow::{ensure, Context, Result};
use git2::{Oid, StatusOptions};
use git2db::pinned_disk::DiskPinnedTreeProjection;
use git2db::pinned_tree::{PinnedTree, SealedTreeProjection};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use tokio::sync::Mutex;

/// Retain the selected commit's verified inputs through every loader read.
/// Only the fixed staging profile needs the disk-backed, capacity-limited path.
#[derive(Debug)]
pub enum PinnedModelArtifact {
    StagingDisk(DiskPinnedTreeProjection),
    Sealed { projection: SealedTreeProjection, owner_pid: u32 },
}

/// Reuse one immutable staging projection across tenant-specific Inference
/// instances. The weak cache does not keep disk reservations alive by itself;
/// the serving instances own strong references for as long as they need the
/// verified file descriptors.
#[derive(Default)]
pub struct StagingPinnedArtifactCache {
    entries: Mutex<HashMap<(PathBuf, git2::Oid), Weak<PinnedModelArtifact>>>,
}

impl StagingPinnedArtifactCache {
    pub async fn acquire(
        &self,
        worktree_path: &Path,
        commit: git2::Oid,
    ) -> Result<Arc<PinnedModelArtifact>> {
        self.acquire_in(
            worktree_path,
            commit,
            Path::new("/var/cache/hyprstream-pinned"),
        )
        .await
    }

    async fn acquire_in(
        &self,
        worktree_path: &Path,
        commit: git2::Oid,
        private_disk_parent: &Path,
    ) -> Result<Arc<PinnedModelArtifact>> {
        self.acquire_in_with_observer(worktree_path, commit, private_disk_parent, || Ok(()))
            .await
    }

    async fn acquire_in_with_observer(
        &self,
        worktree_path: &Path,
        commit: git2::Oid,
        private_disk_parent: &Path,
        after_capture: impl FnOnce() -> Result<()>,
    ) -> Result<Arc<PinnedModelArtifact>> {
        // A cache hit may share only the exact commit from the exact local
        // checkout, and the checkout must still satisfy the current load's
        // clean/HEAD admission rule.
        verify_selected_checkout(worktree_path, commit)?;
        let key = (worktree_path.to_path_buf(), commit);
        let mut entries = self.entries.lock().await;
        entries.retain(|_, artifact| artifact.strong_count() > 0);
        if let Some(artifact) = entries.get(&key).and_then(Weak::upgrade) {
            return Ok(artifact);
        }

        // Hold the async cache lock while acquiring to make same-commit loads
        // coalesce. The bounded acquisition itself runs on the projection I/O
        // runtime, not the Model service reactor.
        let projection = DiskPinnedTreeProjection::acquire_bounded(
            worktree_path,
            commit,
            private_disk_parent,
        )
        .await
        .with_context(|| format!("capture model commit {commit}"))?;
        after_capture()?;
        // Capturing a large model can take long enough for a checkout update
        // to race the admission. Do not publish an artifact unless the exact
        // selected checkout is still clean and at the requested commit.
        verify_selected_checkout(worktree_path, commit)?;
        let artifact = Arc::new(PinnedModelArtifact::StagingDisk(projection));
        entries.insert(key, Arc::downgrade(&artifact));
        Ok(artifact)
    }
}

impl PinnedModelArtifact {
    pub fn commit(&self) -> Oid {
        match self {
            Self::StagingDisk(artifact) => artifact.commit(),
            Self::Sealed { projection, .. } => projection.commit(),
        }
    }

    pub fn root(&self) -> &Path {
        match self {
            Self::StagingDisk(artifact) => artifact.root(),
            Self::Sealed { projection, .. } => projection.root(),
        }
    }

    pub fn file_count(&self) -> usize {
        match self {
            Self::StagingDisk(artifact) => artifact.file_count(),
            Self::Sealed { projection, .. } => projection.file_count(),
        }
    }

    pub fn ensure_current_process(&self) -> Result<()> {
        match self {
            Self::StagingDisk(artifact) => artifact.ensure_current_process()?,
            Self::Sealed { owner_pid, .. } => {
                ensure!(*owner_pid == std::process::id(), "sealed pinned projection cannot cross a process boundary");
            }
        }
        Ok(())
    }
}

/// Admit only the selected, clean checkout of `commit`, then capture the
/// requested Git tree independently of worktree bytes. The second checkout
/// observation enforces the dirty policy during acquisition; the Git object
/// capture and private disk projection, not these observations, prove loader bytes.
pub async fn acquire_pinned_model(
    worktree_path: &Path,
    commit: Oid,
) -> Result<Arc<PinnedModelArtifact>> {
    verify_selected_checkout(worktree_path, commit)?;
    let projection = PinnedTree::acquire(worktree_path, commit)
        .await
        .with_context(|| format!("capture model commit {commit}"))?
        .into_sealed_projection()?;
    verify_selected_checkout(worktree_path, commit)?;
    ensure!(
        projection.commit() == commit,
        "pinned projection commit mismatch"
    );
    Ok(Arc::new(PinnedModelArtifact::Sealed {
        projection,
        owner_pid: std::process::id(),
    }))
}

/// Bounded single-slot projection lease for the staging admission profile.
pub async fn acquire_pinned_model_bounded(
    worktree_path: &Path,
    commit: Oid,
) -> Result<Arc<PinnedModelArtifact>> {
    StagingPinnedArtifactCache::default()
        .acquire(worktree_path, commit)
        .await
}

#[cfg(test)]
pub(crate) async fn acquire_pinned_model_bounded_in(
    cache: &StagingPinnedArtifactCache,
    worktree_path: &Path,
    commit: Oid,
    private_disk_parent: &Path,
) -> Result<Arc<PinnedModelArtifact>> {
    cache
        .acquire_in(worktree_path, commit, private_disk_parent)
        .await
}

#[cfg(test)]
pub(crate) async fn acquire_pinned_model_in(
    worktree_path: &Path,
    commit: Oid,
    private_disk_parent: &Path,
) -> Result<Arc<DiskPinnedTreeProjection>> {
    Ok(Arc::new(
        acquire_pinned_model_with_observer(
            worktree_path,
            commit,
            private_disk_parent,
            false,
            || Ok(()),
        )
        .await?,
    ))
}

#[cfg(test)]
async fn acquire_pinned_model_with_observer(
    worktree_path: &Path,
    commit: Oid,
    private_disk_parent: &Path,
    bounded: bool,
    after_capture: impl FnOnce() -> Result<()>,
) -> Result<DiskPinnedTreeProjection> {
    verify_selected_checkout(worktree_path, commit)?;
    let projection = if bounded {
        DiskPinnedTreeProjection::acquire_bounded(worktree_path, commit, private_disk_parent).await
    } else {
        DiskPinnedTreeProjection::acquire(worktree_path, commit, private_disk_parent).await
    }
    .with_context(|| format!("capture model commit {commit}"))?;
    after_capture()?;
    verify_selected_checkout(worktree_path, commit)?;
    ensure!(
        projection.commit() == commit,
        "pinned projection commit mismatch"
    );
    Ok(projection)
}

pub fn verify_selected_checkout(worktree_path: &Path, commit: Oid) -> Result<()> {
    let repo = git2db::GitManager::global()
        .get_repository(worktree_path)
        .context("open selected model worktree")?
        .open()
        .context("open selected model repository")?;
    ensure!(!repo.is_bare(), "selected model checkout is bare");
    let head = repo
        .head()
        .context("read selected model HEAD")?
        .peel_to_commit()
        .context("resolve selected model HEAD commit")?
        .id();
    ensure!(
        head == commit,
        "selected model checkout is at {head}, requested {commit}"
    );
    let mut options = StatusOptions::new();
    options
        .include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_ignored(true)
        .recurse_ignored_dirs(true);
    let statuses = repo
        .statuses(Some(&mut options))
        .context("read selected model checkout status")?;
    ensure!(statuses.is_empty(), "selected model checkout is dirty");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    // The bounded projection capacity is intentionally one process-global
    // slot. Serialize tests that acquire it so parallel libtest scheduling
    // cannot make the sharing assertion fail for unrelated test timing.
    static BOUNDED_PROJECTION_TEST_LOCK: tokio::sync::Mutex<()> =
        tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn generic_commit_uses_sealed_projection_without_private_disk_parent() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let repo = git2db::Repository::init(dir.path())?;
        std::fs::write(dir.path().join("config.json"), b"reviewed")?;
        let mut index = repo.index()?;
        index.add_path(Path::new("config.json"))?;
        index.write()?;
        let tree = repo.find_tree(index.write_tree()?)?;
        let sig = git2::Signature::now("test", "test@example.invalid")?;
        let commit = repo.commit(Some("HEAD"), &sig, &sig, "reviewed", &tree, &[])?;
        drop(tree);
        drop(repo);

        let artifact = acquire_pinned_model(dir.path(), commit).await?;
        assert!(matches!(&*artifact, PinnedModelArtifact::Sealed { .. }));
        std::fs::write(dir.path().join("config.json"), b"changed")?;
        assert_eq!(
            std::fs::read(artifact.root().join("config.json"))?,
            b"reviewed"
        );
        Ok(())
    }

    #[tokio::test]
    async fn mismatched_or_dirty_checkout_fails_before_artifact_success() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let parent = tempfile::tempdir()?;
        std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o700))?;
        let repo = git2db::Repository::init(dir.path())?;
        std::fs::write(dir.path().join("config.json"), b"reviewed")?;
        let mut index = repo.index()?;
        index.add_path(Path::new("config.json"))?;
        index.write()?;
        let tree = repo.find_tree(index.write_tree()?)?;
        let sig = git2::Signature::now("test", "test@example.invalid")?;
        let commit = repo.commit(Some("HEAD"), &sig, &sig, "reviewed", &tree, &[])?;
        drop(tree);
        drop(repo);
        assert!(acquire_pinned_model_in(dir.path(), Oid::zero(), parent.path()).await.is_err());
        std::fs::write(dir.path().join("config.json"), b"changed")?;
        assert!(acquire_pinned_model_in(dir.path(), commit, parent.path()).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn mutation_during_acquisition_fails_closed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let parent = tempfile::tempdir()?;
        std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o700))?;
        let repo = git2db::Repository::init(dir.path())?;
        std::fs::write(dir.path().join("config.json"), b"reviewed")?;
        let mut index = repo.index()?;
        index.add_path(Path::new("config.json"))?;
        index.write()?;
        let tree = repo.find_tree(index.write_tree()?)?;
        let sig = git2::Signature::now("test", "test@example.invalid")?;
        let commit = repo.commit(Some("HEAD"), &sig, &sig, "reviewed", &tree, &[])?;
        drop(tree);
        drop(repo);
        assert!(acquire_pinned_model_with_observer(dir.path(), commit, parent.path(), false, || {
            std::fs::write(dir.path().join("config.json"), b"changed")?;
            Ok(())
        }).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn bounded_cache_rejects_checkout_mutation_during_capture() -> Result<()> {
        let _guard = BOUNDED_PROJECTION_TEST_LOCK.lock().await;
        let dir = tempfile::tempdir()?;
        let parent = tempfile::tempdir()?;
        std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o700))?;
        let repo = git2db::Repository::init(dir.path())?;
        std::fs::write(dir.path().join("config.json"), b"reviewed")?;
        let mut index = repo.index()?;
        index.add_path(Path::new("config.json"))?;
        index.write()?;
        let tree = repo.find_tree(index.write_tree()?)?;
        let sig = git2::Signature::now("test", "test@example.invalid")?;
        let commit = repo.commit(Some("HEAD"), &sig, &sig, "reviewed", &tree, &[])?;
        drop(tree);
        drop(repo);

        let cache = StagingPinnedArtifactCache::default();
        assert!(cache
            .acquire_in_with_observer(dir.path(), commit, parent.path(), || {
                std::fs::write(dir.path().join("config.json"), b"changed")?;
                Ok(())
            })
            .await
            .is_err());
        assert!(cache.entries.lock().await.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn loader_inputs_remain_exact_after_checkout_replacement() -> Result<()> {
        use crate::runtime::model_config::ModelConfig;
        use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

        let dir = tempfile::tempdir()?;
        let parent = tempfile::tempdir()?;
        std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o700))?;
        let repo = git2db::Repository::init(dir.path())?;
        let config = br#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2}"#;
        std::fs::write(dir.path().join("config.json"), config)?;
        std::fs::write(dir.path().join("model.safetensors"), b"reviewed weights")?;
        std::fs::write(dir.path().join("generation_config.json"), br#"{"temperature":0.25}"#)?;
        Tokenizer::new(WordLevel::default())
            .save(dir.path().join("tokenizer.json"), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut index = repo.index()?;
        for name in ["config.json", "model.safetensors", "generation_config.json", "tokenizer.json"] {
            index.add_path(Path::new(name))?;
        }
        index.write()?;
        let tree = repo.find_tree(index.write_tree()?)?;
        let sig = git2::Signature::now("test", "test@example.invalid")?;
        let commit = repo.commit(Some("HEAD"), &sig, &sig, "reviewed", &tree, &[])?;
        drop(tree);
        drop(repo);

        let artifact = acquire_pinned_model_in(dir.path(), commit, parent.path()).await?;
        std::fs::write(dir.path().join("config.json"), b"invalid replacement")?;
        std::fs::write(dir.path().join("model.safetensors"), b"replacement weights")?;
        std::fs::write(dir.path().join("tokenizer.json"), b"invalid replacement")?;
        std::fs::write(dir.path().join("generation_config.json"), br#"{"temperature":0.9}"#)?;
        std::fs::write(dir.path().join("untracked.safetensors"), b"unreviewed")?;

        let loaded_config = ModelConfig::load(artifact.root(), &std::collections::HashMap::new())?;
        assert_eq!(loaded_config.hidden_size, 8);
        assert_eq!(std::fs::read(artifact.root().join("model.safetensors"))?, b"reviewed weights");
        assert!(Tokenizer::from_file(artifact.root().join("tokenizer.json")).is_ok());
        let sampling = crate::config::SamplingParams::from_model_path(artifact.root())
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        assert_eq!(sampling.temperature, Some(0.25));
        assert!(!artifact.root().join("untracked.safetensors").exists());
        Ok(())
    }

    #[tokio::test]
    async fn bounded_staging_loads_share_one_projection_for_the_same_commit() -> Result<()> {
        let _guard = BOUNDED_PROJECTION_TEST_LOCK.lock().await;
        let repo_dir = tempfile::tempdir()?;
        let private_parent = tempfile::tempdir_in(
            std::env::current_exe()?
                .parent()
                .ok_or_else(|| anyhow::anyhow!("test binary has no parent directory"))?,
        )?;
        std::fs::set_permissions(
            private_parent.path(),
            std::fs::Permissions::from_mode(0o700),
        )?;
        let repo = git2db::Repository::init(repo_dir.path())?;
        std::fs::write(repo_dir.path().join("config.json"), b"reviewed")?;
        let mut index = repo.index()?;
        index.add_path(Path::new("config.json"))?;
        index.write()?;
        let tree = repo.find_tree(index.write_tree()?)?;
        let signature = git2::Signature::now("test", "test@example.invalid")?;
        let commit = repo.commit(Some("HEAD"), &signature, &signature, "reviewed", &tree, &[])?;
        drop(tree);
        drop(repo);

        let cache = StagingPinnedArtifactCache::default();
        let first = acquire_pinned_model_bounded_in(
            &cache,
            repo_dir.path(),
            commit,
            private_parent.path(),
        )
        .await?;
        let second = acquire_pinned_model_bounded_in(
            &cache,
            repo_dir.path(),
            commit,
            private_parent.path(),
        )
        .await?;

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.commit(), commit);
        assert_eq!(
            std::fs::read(first.root().join("config.json"))?,
            b"reviewed"
        );
        Ok(())
    }
}
