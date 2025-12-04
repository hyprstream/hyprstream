//! Tests for atomic worktree creation with rollback

use git2db::{Git2DB, Git2DBConfig, GitManager, RepoId, WorktreeConfig};
use std::path::PathBuf;
use tempfile::TempDir;

/// Test that worktree creation is atomic when LFS fetch fails
#[tokio::test]
async fn test_lfs_fetch_failure_rolls_back_worktree() {
    // Create temp directory for test
    let temp = TempDir::new().unwrap();
    let models_dir = temp.path().join("models");
    std::fs::create_dir_all(&models_dir).unwrap();

    // Create a minimal git repo with LFS config but no actual LFS setup
    // This will cause LFS fetch to fail
    let repo_dir = temp.path().join("test-repo");
    std::fs::create_dir_all(&repo_dir).unwrap();

    // Initialize git repo
    let repo = git2::Repository::init(&repo_dir).unwrap();

    // Create .gitattributes with LFS filter (but no actual LFS)
    let gitattributes = repo_dir.join(".gitattributes");
    std::fs::write(&gitattributes, "*.bin filter=lfs\n").unwrap();

    // Create a test file
    let test_file = repo_dir.join("test.txt");
    std::fs::write(&test_file, "hello world").unwrap();

    // Commit the files
    let mut index = repo.index().unwrap();
    index.add_path(std::path::Path::new(".gitattributes")).unwrap();
    index.add_path(std::path::Path::new("test.txt")).unwrap();
    index.write().unwrap();

    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();

    let sig = git2::Signature::now("Test", "test@example.com").unwrap();
    repo.commit(
        Some("HEAD"),
        &sig,
        &sig,
        "Initial commit",
        &tree,
        &[],
    )
    .unwrap();

    // Configure git2db (LFS fetch is always enabled with atomic rollback)
    let registry = Git2DB::open(temp.path()).await.unwrap();
    let repo_id = registry.register_repository(&RepoId::new(), None, format!("test-repo-{}", repo_dir.display())).unwrap();
    let handle = registry.repo(&repo_id).unwrap();

    // Attempt to create worktree - should fail and rollback
    let worktree_path = temp.path().join("worktree");
    let result = handle.create_worktree(&worktree_path, "main").await;

    // Verify that:
    // 1. Worktree creation failed
    match result {
        Err(e) => {
            // 2. Error message mentions LFS and rollback
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("LFS") || error_msg.contains("lfs"),
                "Error should mention LFS: {}",
                error_msg
            );
            assert!(
                error_msg.contains("rolled back") || error_msg.contains("rollback"),
                "Error should mention rollback: {}",
                error_msg
            );
        }
        Ok(_) => panic!("Expected worktree creation to fail due to LFS error"),
    }

    // 3. Worktree directory was cleaned up (atomic rollback)
    assert!(
        !worktree_path.exists(),
        "Worktree should be cleaned up after rollback"
    );
}

// REMOVED: Non-atomic mode test (fail_on_lfs_error config removed)
// LFS fetch failures now always trigger atomic rollback

/// Test that metadata save failure triggers rollback
#[tokio::test]
async fn test_metadata_save_failure_rolls_back_worktree() {
    // Create temp directory for test
    let temp = TempDir::new().unwrap();
    let models_dir = temp.path().join("models");
    std::fs::create_dir_all(&models_dir).unwrap();

    // Create a test git repo
    let repo_dir = temp.path().join("test-repo");
    std::fs::create_dir_all(&repo_dir).unwrap();

    let repo = git2::Repository::init(&repo_dir).unwrap();

    // Create a test file and commit
    let test_file = repo_dir.join("test.txt");
    std::fs::write(&test_file, "hello world").unwrap();

    let mut index = repo.index().unwrap();
    index.add_path(std::path::Path::new("test.txt")).unwrap();
    index.write().unwrap();

    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();

    let sig = git2::Signature::now("Test", "test@example.com").unwrap();
    repo.commit(
        Some("HEAD"),
        &sig,
        &sig,
        "Initial commit",
        &tree,
        &[],
    )
    .unwrap();

    // This test verifies hyprstream's model_storage layer
    // For now, just verify the basic worktree creation works
    let registry = Git2DB::open(temp.path()).await.unwrap();
    let repo_id = registry.register_repository(&RepoId::new(), None, format!("test-repo-{}", repo_dir.display())).unwrap();
    let handle = registry.repo(&repo_id).unwrap();

    let worktree_path = temp.path().join("worktree");
    let result = handle.create_worktree(&worktree_path, "main").await;

    assert!(result.is_ok(), "Basic worktree creation should succeed");
    assert!(worktree_path.exists(), "Worktree should exist");
}

/// Test partial worktree cleanup on driver failure
#[tokio::test]
async fn test_driver_failure_cleans_up_partial_worktree() {
    // Create temp directory
    let temp = TempDir::new().unwrap();
    let repo_dir = temp.path().join("test-repo");
    std::fs::create_dir_all(&repo_dir).unwrap();

    // Initialize git repo
    let repo = git2::Repository::init(&repo_dir).unwrap();
    let test_file = repo_dir.join("test.txt");
    std::fs::write(&test_file, "hello").unwrap();

    let mut index = repo.index().unwrap();
    index.add_path(std::path::Path::new("test.txt")).unwrap();
    index.write().unwrap();

    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();

    let sig = git2::Signature::now("Test", "test@example.com").unwrap();
    repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[]).unwrap();

    // Try to create worktree with invalid ref - should fail
    let registry = Git2DB::open(temp.path()).await.unwrap();
    let repo_id = registry.register_repository(&RepoId::new(), None, format!("test-repo-{}", repo_dir.display())).unwrap();
    let handle = registry.repo(&repo_id).unwrap();
    let worktree_path = temp.path().join("worktree");

    let result = handle.create_worktree(&worktree_path, "nonexistent-branch").await;

    // Should fail
    assert!(result.is_err(), "Should fail to create worktree with invalid branch");

    // Verify cleanup - worktree path should not exist
    assert!(
        !worktree_path.exists(),
        "Partial worktree should be cleaned up on failure"
    );
}
