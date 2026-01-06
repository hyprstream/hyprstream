//! Integration tests for BranchManager and RemoteManager
//!
//! Tests git-native branch and remote operations

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

/// Helper to create a test repository with branches
async fn create_test_repo_with_branches(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use git2::{Repository, Signature};

    let repo = Repository::init(path)?;
    // Use explicit signature for CI environments without git config
    let sig = Signature::now("Test User", "test@example.com")?;

    // Create initial commit on main
    let tree_id = {
        let mut index = repo.index()?;
        let file_path = path.join("README.md");
        fs::write(&file_path, "# Test Repository\n")?;
        index.add_path(std::path::Path::new("README.md"))?;
        index.write()?;
        index.write_tree()?
    };

    let tree = repo.find_tree(tree_id)?;
    let commit = repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])?;
    let commit_obj = repo.find_commit(commit)?;

    // Create develop branch
    repo.branch("develop", &commit_obj, false)?;

    // Create feature branch
    repo.branch("feature-x", &commit_obj, false)?;

    Ok(())
}

#[tokio::test]
async fn test_branch_list() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    // Create test repo with branches
    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    // Get branch manager
    let repo = registry.repo(&repo_id).unwrap();
    let branches = repo.branch().list().await.unwrap();

    // Should have at least main branch
    let branch_names: Vec<_> = branches
        .iter()
        .filter(|b| b.is_local())
        .map(|b| b.name.as_str())
        .collect();

    assert!(branch_names.contains(&"main") || branch_names.contains(&"master"));
}

#[tokio::test]
async fn test_branch_current() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let current = repo.branch().current().await.unwrap();

    assert!(current.is_some());
    let current_branch = current.unwrap();
    assert!(current_branch.is_head);
    assert!(current_branch.name == "main" || current_branch.name == "master");
}

#[tokio::test]
async fn test_branch_create_and_checkout() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.branch();

    // Create new branch from current HEAD (None = HEAD)
    mgr.create::<&str>("new-feature", None).await.unwrap();

    // List branches - should include new branch
    let branches = mgr.list().await.unwrap();
    let has_new_branch = branches
        .iter()
        .any(|b| b.name == "new-feature" && b.is_local());
    assert!(has_new_branch);

    // Checkout the new branch
    mgr.checkout("new-feature").await.unwrap();

    // Current branch should be new-feature
    let current = mgr.current().await.unwrap().unwrap();
    assert_eq!(current.name, "new-feature");
    assert!(current.is_head);
}

#[tokio::test]
async fn test_branch_delete() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.branch();

    // Get the current branch name (could be main or master)
    let default_branch = mgr.current().await.unwrap().unwrap().name.clone();

    // Create and checkout a temporary branch from HEAD
    mgr.create::<&str>("temp-branch", None).await.unwrap();

    // Switch back to default branch
    mgr.checkout(&default_branch).await.unwrap();

    // Remove the temporary branch (force=true since no divergent commits)
    mgr.remove("temp-branch", true).await.unwrap();

    // Verify it's gone
    let branches = mgr.list().await.unwrap();
    let has_temp = branches.iter().any(|b| b.name == "temp-branch");
    assert!(!has_temp);
}

#[tokio::test]
async fn test_branch_delete_current_fails() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.branch();

    // Get current branch name
    let current = mgr.current().await.unwrap().unwrap();
    let current_name = current.name.clone();

    // Try to remove current branch - should fail
    let result = mgr.remove(&current_name, false).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_branch_rename() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.branch();

    // Create a test branch from HEAD
    mgr.create::<&str>("old-name", None).await.unwrap();

    // Rename it
    mgr.rename(Some("old-name"), "new-name").await.unwrap();

    // Verify old name is gone and new name exists
    let branches = mgr.list().await.unwrap();
    let has_old = branches.iter().any(|b| b.name == "old-name");
    let has_new = branches.iter().any(|b| b.name == "new-name");

    assert!(!has_old);
    assert!(has_new);
}

#[tokio::test]
async fn test_remote_add_and_list() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.remote();

    // Should have origin remote
    let remotes = mgr.list().await.unwrap();
    assert!(remotes.iter().any(|r| r.name == "origin"));

    // Add backup remote
    mgr.add("backup", "https://backup.com/repo.git")
        .await
        .unwrap();

    // Should now have both
    let remotes = mgr.list().await.unwrap();
    assert!(remotes.iter().any(|r| r.name == "origin"));
    assert!(remotes.iter().any(|r| r.name == "backup"));
}

#[tokio::test]
async fn test_remote_set_url() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.remote();

    // Change origin URL
    let new_url = "https://new-host.com/repo.git";
    mgr.set_url("origin", new_url).await.unwrap();

    // Verify URL changed
    let remotes = mgr.list().await.unwrap();
    let origin = remotes.iter().find(|r| r.name == "origin").unwrap();
    assert_eq!(origin.url, new_url);
}

#[tokio::test]
async fn test_remote_rename() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.remote();

    // Add a remote
    mgr.add("backup", "https://backup.com/repo.git")
        .await
        .unwrap();

    // Rename it
    mgr.rename("backup", "mirror").await.unwrap();

    // Verify rename
    let remotes = mgr.list().await.unwrap();
    assert!(!remotes.iter().any(|r| r.name == "backup"));
    assert!(remotes.iter().any(|r| r.name == "mirror"));
}

#[tokio::test]
async fn test_remote_remove() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.remote();

    // Add a remote
    mgr.add("temp", "https://temp.com/repo.git").await.unwrap();

    // Verify it exists
    let remotes = mgr.list().await.unwrap();
    assert!(remotes.iter().any(|r| r.name == "temp"));

    // Remove it
    mgr.remove("temp").await.unwrap();

    // Verify it's gone
    let remotes = mgr.list().await.unwrap();
    assert!(!remotes.iter().any(|r| r.name == "temp"));
}

#[tokio::test]
async fn test_remote_default() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.remote();

    // Default should be origin
    let default = mgr.default().await.unwrap();
    assert!(default.is_some());
    assert_eq!(default.unwrap().name, "origin");
}

#[tokio::test]
async fn test_multi_remote_setup() {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await.unwrap();

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo).unwrap();
    create_test_repo_with_branches(&test_repo).await.unwrap();

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await.unwrap();

    let repo = registry.repo(&repo_id).unwrap();
    let mgr = repo.remote();

    // Setup multiple remotes (simulating distributed setup)
    mgr.add("p2p", "gittorrent://peer/repo").await.unwrap();
    mgr.add("backup", "https://backup.com/repo.git")
        .await
        .unwrap();
    mgr.add("mirror", "https://mirror.org/repo.git")
        .await
        .unwrap();

    // List all remotes
    let remotes = mgr.list().await.unwrap();
    assert_eq!(remotes.len(), 4); // origin + 3 new ones

    let names: Vec<_> = remotes.iter().map(|r| r.name.as_str()).collect();
    assert!(names.contains(&"origin"));
    assert!(names.contains(&"p2p"));
    assert!(names.contains(&"backup"));
    assert!(names.contains(&"mirror"));
}
