//! Integration tests for git2db v2.0 core operations
//!
//! Tests the complete workflow of repository management with the v2 API

use git2db::config::RepositoryConfig;
use git2db::{Git2DB, Git2DBConfig, GitManager};
use std::fs;
use tempfile::TempDir;

/// Helper to setup logging for tests
fn setup_logging() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
}

/// Helper to initialize GitManager with shallow clones disabled
/// (Required for file:// URLs which don't support shallow clones)
fn init_git_manager_no_shallow() {
    let mut config = Git2DBConfig::default();
    config.repository = RepositoryConfig {
        prefer_shallow: false, // Disable shallow clones for file:// URLs
        shallow_depth: None,
        auto_init: true,
        auto_init_submodules: false,
    };

    // Try to init - ignore if already initialized
    let _ = GitManager::init_with_config(config);
}

/// Helper to create a simple test repository
async fn create_test_repo(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use git2::Repository;

    // Initialize git repo
    let repo = Repository::init(path)?;

    // Create initial commit
    let sig = repo.signature()?;
    let tree_id = {
        let mut index = repo.index()?;

        // Create a test file
        let file_path = path.join("README.md");
        fs::write(&file_path, "# Test Repository\n")?;

        index.add_path(std::path::Path::new("README.md"))?;
        index.write()?;
        index.write_tree()?
    };

    let tree = repo.find_tree(tree_id)?;
    repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])?;

    Ok(())
}

#[tokio::test]
async fn test_registry_open_and_initialize() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();

    // Open registry (should create new)
    let _registry = Git2DB::open(registry_path).await.unwrap();

    // Verify internal structure
    let internal_registry = registry_path.join(".registry");
    assert!(
        internal_registry.exists(),
        ".registry directory should exist"
    );
    assert!(
        internal_registry.join(".git").exists(),
        ".registry should be a git repo"
    );
    assert!(
        internal_registry.join("registry.json").exists(),
        "registry.json should exist"
    );
    assert!(
        internal_registry.join("repos").exists(),
        "repos/ directory should exist"
    );

    // Verify it can be opened again
    let _registry2 = Git2DB::open(registry_path).await.unwrap();
}

#[tokio::test]
async fn test_upsert_repository_idempotent() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Create a test repository to clone
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir).unwrap();
    create_test_repo(&test_repo_dir).await.unwrap();

    let url = format!("file://{}", test_repo_dir.display());

    // First upsert - should clone
    let id1 = registry
        .upsert_repository("test-project", &url)
        .await
        .unwrap();

    // Verify repository exists
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 1);
    assert_eq!(repos[0].name.as_deref(), Some("test-project"));

    // Second upsert - should return existing
    let id2 = registry
        .upsert_repository("test-project", &url)
        .await
        .unwrap();

    // Should be the same ID
    assert_eq!(id1, id2);

    // Still only one repository
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 1);
}

#[tokio::test]
async fn test_repository_handle_operations() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Create test repository
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir).unwrap();
    create_test_repo(&test_repo_dir).await.unwrap();

    let url = format!("file://{}", test_repo_dir.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    // Get repository handle
    let repo = registry.repo(&repo_id).unwrap();

    // Test metadata access
    assert_eq!(repo.name().unwrap(), Some("test-repo"));
    assert_eq!(repo.url().unwrap(), url);

    // Test worktree access
    let worktree_path = repo.worktree().unwrap();
    assert!(worktree_path.exists());
    assert!(worktree_path.join("README.md").exists());

    // Test status - need to get worktree handle first
    let default_branch = repo.default_branch().unwrap();
    let mut worktree_handle = repo.get_worktree(&default_branch).await.unwrap().unwrap();
    let status = worktree_handle.status().await.unwrap();
    assert!(status.is_clean);
}

#[tokio::test]
async fn test_remove_repository_cleanup() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Create and add test repository
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir).unwrap();
    create_test_repo(&test_repo_dir).await.unwrap();

    let url = format!("file://{}", test_repo_dir.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    // Verify repository exists
    let repo = registry.repo(&repo_id).unwrap();
    let worktree_path = repo.worktree().unwrap().to_path_buf();
    assert!(worktree_path.exists());

    // Get internal paths before removal
    let internal_registry = registry_path.join(".registry");
    let submodule_path = internal_registry.join("repos").join(repo_id.to_string());
    let modules_path = internal_registry
        .join(".git/modules")
        .join(repo_id.to_string());

    // Remove repository
    registry.remove_repository(&repo_id).await.unwrap();

    // Verify complete cleanup
    assert!(!worktree_path.exists(), "Worktree should be removed");
    assert!(!submodule_path.exists(), "Submodule path should be removed");
    assert!(
        !modules_path.exists(),
        ".git/modules entry should be removed"
    );

    // Verify not in metadata
    assert!(registry.get_by_id(&repo_id).is_none());

    // Verify .gitmodules doesn't contain the submodule
    let gitmodules_path = internal_registry.join(".gitmodules");
    if gitmodules_path.exists() {
        let content = fs::read_to_string(&gitmodules_path).unwrap();
        assert!(
            !content.contains(&repo_id.to_string()),
            ".gitmodules should not contain removed submodule"
        );
    }
}

// NOTE: Transaction tests temporarily disabled due to complex lifetime constraints
// The transaction API works but requires more complex test setup
// TODO: Implement transaction tests with proper async lifetime handling

/*
#[tokio::test]
async fn test_transaction_commit() {
    // Transaction tests require special async lifetime handling
    // See documentation for transaction() API usage
}

#[tokio::test]
async fn test_transaction_rollback() {
    // Transaction tests require special async lifetime handling
    // See documentation for transaction() API usage
}
*/

#[tokio::test]
async fn test_update_repository() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Create test repository
    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo(&test_repo).await.unwrap();

    let url1 = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url1).await.unwrap();

    // Verify initial URL
    let repo_info = registry.get_by_id(&repo_id).unwrap();
    assert_eq!(repo_info.url, url1);

    // Update URL
    let new_url = "https://new-host.com/repo.git";
    registry
        .update_repository(&repo_id, Some(new_url.to_string()))
        .await
        .unwrap();

    // Verify URL was updated
    let repo_info = registry.get_by_id(&repo_id).unwrap();
    assert_eq!(repo_info.url, new_url);

    // Verify origin remote was also updated
    let origin_remote = repo_info.remotes.iter().find(|r| r.name == "origin");
    assert!(origin_remote.is_some());
    assert_eq!(origin_remote.unwrap().url, new_url);
}

#[tokio::test]
async fn test_repo_by_name_lookup() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Create test repository
    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo(&test_repo).await.unwrap();
    let url = format!("file://{}", test_repo.display());

    // Add repository with name
    let repo_id = registry.add_repository("my-project", &url).await.unwrap();

    // Lookup by name
    let repo = registry.repo_by_name("my-project").unwrap();
    assert_eq!(repo.id(), &repo_id);

    // Lookup by ID
    let repo2 = registry.repo(&repo_id).unwrap();
    assert_eq!(repo2.name().unwrap(), Some("my-project"));

    // Non-existent name should fail
    let result = registry.repo_by_name("non-existent");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_list_repositories() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Initially empty
    assert_eq!(registry.list().count(), 0);

    // Create test repositories
    for i in 1..=3 {
        let test_repo = temp_dir.path().join(format!("repo{}", i));
        fs::create_dir(&test_repo).unwrap();
        create_test_repo(&test_repo).await.unwrap();
        let url = format!("file://{}", test_repo.display());

        registry
            .add_repository(&format!("repo{}", i), &url)
            .await
            .unwrap();
    }

    // Should have 3 repositories
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 3);

    // Check display names
    let names: Vec<_> = repos.iter().map(|r| r.display_name()).collect();
    assert!(names.contains(&"repo1".to_string()));
    assert!(names.contains(&"repo2".to_string()));
    assert!(names.contains(&"repo3".to_string()));
}

#[tokio::test]
async fn test_worktree_status_ahead_behind() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Create a test repository with multiple commits
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir).unwrap();
    
    use git2::Repository;
    let repo = Repository::init(&test_repo_dir).unwrap();
    let sig = repo.signature().unwrap();

    // Create initial commit
    let mut index = repo.index().unwrap();
    let readme_path = test_repo_dir.join("README.md");
    fs::write(&readme_path, "# Test Repository\n").unwrap();
    index.add_path(std::path::Path::new("README.md")).unwrap();
    index.write().unwrap();
    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();
    repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[]).unwrap();

    // Create a second commit (local is ahead)
    let file2_path = test_repo_dir.join("file2.txt");
    fs::write(&file2_path, "Content\n").unwrap();
    let mut index = repo.index().unwrap();
    index.add_path(std::path::Path::new("file2.txt")).unwrap();
    index.write().unwrap();
    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();
    let head = repo.head().unwrap().peel_to_commit().unwrap();
    repo.commit(Some("HEAD"), &sig, &sig, "Second commit", &tree, &[&head]).unwrap();

    // Add repository to registry
    let url = format!("file://{}", test_repo_dir.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();
    let handle = registry.repo(&repo_id).unwrap();

    // Create a new branch "test-branch" for the worktree (can't use main since it's already checked out)
    let base_repo_path = handle.worktree().unwrap();
    {
        let base_repo = Repository::open(&base_repo_path).unwrap();
        let head_commit = base_repo.head().unwrap().peel_to_commit().unwrap();
        base_repo.branch("test-branch", &head_commit, false).unwrap();
    }

    // Create a worktree for "test-branch"
    let worktree_path = temp_dir.path().join("worktrees").join("test-branch");
    let mut worktree_handle = handle.create_worktree(&worktree_path, "test-branch").await.unwrap();

    // Get status - should show ahead: 0, behind: 0 since there's no upstream tracking
    let status = worktree_handle.status().await.unwrap();
    assert_eq!(status.ahead, 0, "Should be 0 ahead when no upstream is configured");
    assert_eq!(status.behind, 0, "Should be 0 behind when no upstream is configured");

    // Now set up upstream tracking by creating a "remote" branch in the same repo
    // and setting it as upstream
    let repo_path = &worktree_path;
    let repo = Repository::open(repo_path).unwrap();
    
    // Create a remote branch reference (simulating origin/test-branch)
    let head_commit = repo.head().unwrap().peel_to_commit().unwrap();
    let test_branch = "test-branch";
    let remote_ref_name = format!("refs/remotes/origin/{}", test_branch);
    repo.reference(&remote_ref_name, head_commit.id(), false, "Create remote ref").unwrap();

    // Set upstream for the current branch (test-branch)
    let mut branch = repo.find_branch(test_branch, git2::BranchType::Local).unwrap();
    branch.set_upstream(Some(&format!("origin/{}", test_branch))).unwrap();

    // Now make another local commit (we'll be ahead of the remote)
    let file3_path = repo_path.join("file3.txt");
    fs::write(&file3_path, "More content\n").unwrap();
    let mut index = repo.index().unwrap();
    index.add_path(std::path::Path::new("file3.txt")).unwrap();
    index.write().unwrap();
    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();
    let head = repo.head().unwrap().peel_to_commit().unwrap();
    repo.commit(Some("HEAD"), &sig, &sig, "Third commit", &tree, &[&head]).unwrap();

    // Get status again - should show ahead: 1, behind: 0
    let status = worktree_handle.status().await.unwrap();
    assert_eq!(status.ahead, 1, "Should be 1 commit ahead of upstream");
    assert_eq!(status.behind, 0, "Should be 0 commits behind upstream");

    // Now simulate a divergence scenario:
    // We need the remote to have a commit that local doesn't have.
    // To do this, we'll:
    // 1. Create a new commit on top of the original HEAD (before our local commit)
    // 2. Update the remote ref to point to that new commit
    //
    // Current state:
    //   remote: A -> B (original head_commit)
    //   local:  A -> B -> C (our "Third commit")
    //
    // We want:
    //   remote: A -> B -> D (new commit branching from B)
    //   local:  A -> B -> C (our "Third commit")
    //
    // This will make local 1 ahead (has C) and 1 behind (doesn't have D)

    // Get the parent of the current HEAD (this is commit B, before we added "Third commit")
    let current_head = repo.head().unwrap().peel_to_commit().unwrap();
    let parent_commit = current_head.parent(0).unwrap();

    // Create a "remote" commit D on top of B (the parent)
    let remote_file = repo_path.join("remote_file.txt");
    fs::write(&remote_file, "Remote-only content\n").unwrap();
    let mut index = repo.index().unwrap();
    index.add_path(std::path::Path::new("remote_file.txt")).unwrap();
    index.write().unwrap();
    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();
    // Create commit D with parent B (not C)
    let remote_commit = repo.commit(None, &sig, &sig, "Remote commit D", &tree, &[&parent_commit]).unwrap();

    // Update the remote ref to point to this new commit D
    repo.reference(&remote_ref_name, remote_commit, true, "Update remote to diverged commit").unwrap();

    // Clean up the working directory (remove the remote_file we added to index)
    fs::remove_file(&remote_file).unwrap();
    // Reset the index to match HEAD
    repo.reset(current_head.as_object(), git2::ResetType::Mixed, None).unwrap();

    // Get status - should show ahead: 1, behind: 1 (diverged branches)
    let status = worktree_handle.status().await.unwrap();
    assert_eq!(status.ahead, 1, "Should be 1 commit ahead of upstream");
    assert_eq!(status.behind, 1, "Should be 1 commit behind upstream");
}
