#![allow(clippy::print_stdout)]
//! Integration tests for transaction commit logic

use git2db::{Git2DB, IsolationMode};
use tempfile::TempDir;

#[tokio::test]
async fn test_transaction_commit_clone() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let mut registry = Git2DB::open(temp_dir.path()).await?;

    // Capture baseline count (.registry self-entry exists on fresh open)
    let baseline = registry.list().count();

    // Start transaction with optimistic mode (fastest for testing)
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await?;

    // Queue a clone operation
    let id = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await?;

    println!("Queued clone operation with ID: {id}");

    // Repository shouldn't exist yet
    assert_eq!(
        registry.list().count(),
        baseline,
        "Repository shouldn't exist before commit"
    );

    // Commit the transaction
    println!("Committing transaction...");
    tx.commit_to(&mut registry).await?;

    // Now repository should exist
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), baseline + 1, "Repository should exist after commit");
    assert!(repos.iter().any(|r| r.id == id), "ID should match");
    assert!(
        repos.iter().any(|r| r.name.as_deref() == Some("git")),
        "Name should match"
    );

    println!("✓ Transaction committed successfully");
    Ok(())
}

#[tokio::test]
async fn test_transaction_rollback() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let registry = Git2DB::open(temp_dir.path()).await?;

    let baseline = registry.list().count();

    // Start transaction
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await?;

    // Queue operation
    let id = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await?;

    println!("Queued clone with ID: {id}, now rolling back...");

    // Rollback instead of commit
    tx.rollback().await?;

    // Repository should NOT exist
    assert_eq!(
        registry.list().count(),
        baseline,
        "Repository should not exist after rollback"
    );

    println!("✓ Transaction rolled back successfully");
    Ok(())
}

#[tokio::test]
async fn test_transaction_multiple_operations() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let mut registry = Git2DB::open(temp_dir.path()).await?;

    let baseline = registry.list().count();

    // Start transaction
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await?;

    // Queue multiple operations
    let id1 = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await?;

    let id2 = tx
        .clone_repo("git2-rs", "https://github.com/rust-lang/git2-rs.git")
        .await?;

    println!("Queued 2 clone operations");
    println!("Operations: {:?}", tx.operations().await);

    // Commit
    tx.commit_to(&mut registry).await?;

    // Both should exist
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), baseline + 2, "Should have 2 new repositories");

    assert!(repos.iter().any(|r| r.id == id1), "First repo should exist");
    assert!(
        repos.iter().any(|r| r.id == id2),
        "Second repo should exist"
    );

    println!("✓ Multiple operations committed successfully");
    Ok(())
}

#[tokio::test]
async fn test_transaction_ensure_idempotent() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let mut registry = Git2DB::open(temp_dir.path()).await?;

    let baseline = registry.list().count();

    // First transaction: add repository
    let tx1 = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await?;

    let id1 = tx1
        .ensure("git", "https://github.com/git/git.git")
        .await?;

    tx1.commit_to(&mut registry).await?;

    // Second transaction: ensure same repository (should return existing)
    let tx2 = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await?;

    let id2 = tx2
        .ensure("git", "https://github.com/git/git.git")
        .await?;

    // Should return the existing ID without cloning again
    assert_eq!(id1, id2, "ensure() should return existing repository ID");

    // Commit should be no-op
    tx2.commit_to(&mut registry).await?;

    // Still only one user repository added
    assert_eq!(
        registry.list().count(),
        baseline + 1,
        "Should still have only 1 new repository"
    );

    println!("✓ ensure() is idempotent");
    Ok(())
}

#[tokio::test]
#[ignore] // Requires network access
async fn test_transaction_worktree_mode() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let mut registry = Git2DB::open(temp_dir.path()).await?;

    // Start transaction with full worktree isolation
    let tx = registry
        .start_transaction_with_mode(IsolationMode::Worktree)
        .await?;

    let id = tx
        .clone_repo("git", "https://github.com/git/git.git")
        .await?;

    println!("Queued clone in worktree mode");

    // Commit (this will create and then prune a git worktree)
    tx.commit_to(&mut registry).await?;

    // Verify repository exists
    assert!(registry.get_by_id(&id).is_some(), "Repository should exist");

    // Verify worktree was cleaned up
    let worktrees_dir = temp_dir.path().join(".registry/.worktrees");
    if worktrees_dir.exists() {
        let entries = std::fs::read_dir(&worktrees_dir)?;
        assert_eq!(
            entries.count(),
            0,
            "Worktrees directory should be empty after commit"
        );
    }

    println!("✓ Worktree mode works correctly");
    Ok(())
}

#[tokio::test]
async fn test_registry_version_accessor() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let registry = Git2DB::open(temp_dir.path()).await?;

    // Verify registry exposes a non-empty version from metadata
    let version = registry.version();
    assert!(!version.is_empty(), "Registry should have a version");

    // Version should be a valid semver-like string
    let first_char = version.chars().next().ok_or("Version string is empty")?;
    assert!(
        first_char.is_ascii_digit(),
        "Version should start with a digit"
    );

    // Verify version remains consistent across calls
    assert_eq!(registry.version(), version);

    // Start a transaction to ensure version() works in transaction contexts
    // (transaction snapshots use registry.version() internally)
    let _tx = registry
        .start_transaction_with_mode(IsolationMode::Optimistic)
        .await?;

    // Version should still be accessible and unchanged
    assert_eq!(registry.version(), version);

    println!("✓ Registry version accessor works correctly");
    Ok(())
}
