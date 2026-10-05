//! Local admission and owned artifact acquisition for an exact commit model.

use anyhow::{ensure, Context, Result};
use git2::{Oid, StatusOptions};
use git2db::pinned_disk::DiskPinnedTreeProjection;
use std::path::Path;
use std::sync::Arc;

/// Admit only the selected, clean checkout of `commit`, then capture the
/// requested Git tree independently of worktree bytes. The second checkout
/// observation enforces the dirty policy during acquisition; the Git object
/// capture and private disk projection, not these observations, prove loader bytes.
pub async fn acquire_pinned_model(
    worktree_path: &Path,
    commit: Oid,
) -> Result<Arc<DiskPinnedTreeProjection>> {
    acquire_pinned_model_in(worktree_path, commit, Path::new("/var/cache/hyprstream-pinned")).await
}

/// Bounded single-slot projection lease for the staging admission profile.
pub async fn acquire_pinned_model_bounded(
    worktree_path: &Path,
    commit: Oid,
) -> Result<Arc<DiskPinnedTreeProjection>> {
    acquire_pinned_model_with_observer(
        worktree_path,
        commit,
        Path::new("/var/cache/hyprstream-pinned"),
        true,
        || Ok(()),
    ).await
}

pub(crate) async fn acquire_pinned_model_in(
    worktree_path: &Path,
    commit: Oid,
    private_disk_parent: &Path,
) -> Result<Arc<DiskPinnedTreeProjection>> {
    acquire_pinned_model_with_observer(worktree_path, commit, private_disk_parent, false, || Ok(())).await
}

async fn acquire_pinned_model_with_observer(
    worktree_path: &Path,
    commit: Oid,
    private_disk_parent: &Path,
    bounded: bool,
    after_capture: impl FnOnce() -> Result<()>,
) -> Result<Arc<DiskPinnedTreeProjection>> {
    verify_selected_checkout(worktree_path, commit)?;
    let projection = if bounded {
        DiskPinnedTreeProjection::acquire_bounded(worktree_path, commit, private_disk_parent).await
    } else {
        DiskPinnedTreeProjection::acquire(worktree_path, commit, private_disk_parent).await
    }.with_context(|| format!("capture model commit {commit}"))?;
    after_capture()?;
    verify_selected_checkout(worktree_path, commit)?;
    ensure!(
        projection.commit() == commit,
        "pinned projection commit mismatch"
    );
    Ok(Arc::new(projection))
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
}
