//! Integration tests for transaction commit logic

use git2db::{Git2DB, IsolationMode};
use tempfile::TempDir;

#[tokio::test]
async fn test_transaction_commit_clone() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = Git2DB::open(temp_dir.path()).await.unwrap();

    // Verify empty registry
    assert_eq!(registry.list().count(), 0);

    // Start transaction with optimistic mode (fastest for testing)
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await
        .unwrap();

    // Queue a clone operation
    let id = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await
        .unwrap();

    println!("Queued clone operation with ID: {}", id);

    // Repository shouldn't exist yet
    assert_eq!(registry.list().count(), 0, "Repository shouldn't exist before commit");

    // Commit the transaction
    println!("Committing transaction...");
    tx.commit_to(&mut registry).await.unwrap();

    // Now repository should exist
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 1, "Repository should exist after commit");
    assert_eq!(repos[0].id, id, "ID should match");
    assert_eq!(repos[0].name.as_deref(), Some("git"), "Name should match");

    println!("✓ Transaction committed successfully");
}

#[tokio::test]
async fn test_transaction_rollback() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = Git2DB::open(temp_dir.path()).await.unwrap();

    // Start transaction
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await
        .unwrap();

    // Queue operation
    let id = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await
        .unwrap();

    println!("Queued clone with ID: {}, now rolling back...", id);

    // Rollback instead of commit
    tx.rollback().await.unwrap();

    // Repository should NOT exist
    assert_eq!(registry.list().count(), 0, "Repository should not exist after rollback");

    println!("✓ Transaction rolled back successfully");
}

#[tokio::test]
async fn test_transaction_multiple_operations() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = Git2DB::open(temp_dir.path()).await.unwrap();

    // Start transaction
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await
        .unwrap();

    // Queue multiple operations
    let id1 = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await
        .unwrap();

    let id2 = tx
        .clone_repo("git2-rs", "https://github.com/rust-lang/git2-rs.git")
        .await
        .unwrap();

    println!("Queued 2 clone operations");
    println!("Operations: {:?}", tx.operations().await);

    // Commit
    tx.commit_to(&mut registry).await.unwrap();

    // Both should exist
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 2, "Should have 2 repositories");

    assert!(
        repos.iter().any(|r| r.id == id1),
        "First repo should exist"
    );
    assert!(
        repos.iter().any(|r| r.id == id2),
        "Second repo should exist"
    );

    println!("✓ Multiple operations committed successfully");
}

#[tokio::test]
async fn test_transaction_ensure_idempotent() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = Git2DB::open(temp_dir.path()).await.unwrap();

    // First transaction: add repository
    let tx1 = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await
        .unwrap();

    let id1 = tx1
        .ensure("git", "https://github.com/git/git.git")
        .await
        .unwrap();

    tx1.commit_to(&mut registry).await.unwrap();

    // Second transaction: ensure same repository (should return existing)
    let tx2 = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await
        .unwrap();

    let id2 = tx2
        .ensure("git", "https://github.com/git/git.git")
        .await
        .unwrap();

    // Should return the existing ID without cloning again
    assert_eq!(id1, id2, "ensure() should return existing repository ID");

    // Commit should be no-op
    tx2.commit_to(&mut registry).await.unwrap();

    // Still only one repository
    assert_eq!(registry.list().count(), 1, "Should still have only 1 repository");

    println!("✓ ensure() is idempotent");
}

#[tokio::test]
#[ignore] // Requires network access
async fn test_transaction_worktree_mode() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = Git2DB::open(temp_dir.path()).await.unwrap();

    // Start transaction with full worktree isolation
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Worktree)
        .await
        .unwrap();

    let id = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await
        .unwrap();

    println!("Queued clone in worktree mode");

    // Commit (this will create and then prune a git worktree)
    tx.commit_to(&mut registry).await.unwrap();

    // Verify repository exists
    assert!(registry.get_by_id(&id).is_some(), "Repository should exist");

    // Verify worktree was cleaned up
    let worktrees_dir = temp_dir.path().join(".registry/.worktrees");
    if worktrees_dir.exists() {
        let entries = std::fs::read_dir(&worktrees_dir).unwrap();
        assert_eq!(
            entries.count(),
            0,
            "Worktrees directory should be empty after commit"
        );
    }

    println!("✓ Worktree mode works correctly");
}
