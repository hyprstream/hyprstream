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

#[tokio::test]
async fn test_repository_handle_commit() {
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

    // Add repository to registry
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();
    let handle = registry.repo(&repo_id).unwrap();

    // Make a change to the repository
    let worktree_path = handle.worktree().unwrap();
    let test_file = worktree_path.join("test.txt");
    fs::write(&test_file, "Hello, world!").unwrap();

    // Stage the change
    handle.staging().add("test.txt").await.unwrap();

    // Commit using the new commit API
    let commit_oid = handle.commit("Add test file").await.unwrap();

    // Verify the commit was created
    assert!(!commit_oid.is_zero(), "Commit OID should not be zero");

    // Verify the file is in the commit
    let repo = handle.open_repo().unwrap();
    let commit = repo.find_commit(commit_oid).unwrap();
    let tree = commit.tree().unwrap();
    assert!(tree.get_path(std::path::Path::new("test.txt")).is_ok(), "test.txt should be in the commit");

    // Verify commit message
    assert_eq!(commit.message(), Some("Add test file\n"));

    // Test commit_as with custom signature
    let test_file2 = worktree_path.join("test2.txt");
    fs::write(&test_file2, "Another file").unwrap();
    handle.staging().add("test2.txt").await.unwrap();

    use git2::Signature;
    let custom_sig = Signature::now("Test User", "test@example.com").unwrap();
    let commit_oid2 = handle.commit_as(&custom_sig, "Add second test file").await.unwrap();

    // Verify custom signature was used
    let commit2 = repo.find_commit(commit_oid2).unwrap();
    assert_eq!(commit2.author().name(), Some("Test User"));
    assert_eq!(commit2.author().email(), Some("test@example.com"));
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
