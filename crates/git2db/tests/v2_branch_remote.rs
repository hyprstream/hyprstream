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
async fn test_branch_list() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create test repo with branches
    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    // Get branch manager
    let repo = registry.repo(&repo_id)?;
    let branches = repo.branch().list().await?;

    // Should have at least one local branch (the default branch, whatever it's named)
    let local_branches: Vec<_> = branches.iter().filter(|b| b.is_local()).collect();

    assert!(
        !local_branches.is_empty(),
        "Should have at least one local branch"
    );
    // The default branch should be marked as HEAD
    assert!(
        local_branches.iter().any(|b| b.is_head),
        "One branch should be HEAD"
    );
    Ok(())
}

#[tokio::test]
async fn test_branch_current() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let current = repo.branch().current().await?;

    assert!(current.is_some(), "Should have a current branch");
    if let Some(current_branch) = current {
        assert!(current_branch.is_head, "Current branch should be HEAD");
        assert!(
            !current_branch.name.is_empty(),
            "Current branch should have a name"
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_branch_create_and_checkout() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.branch();

    // Create new branch from current HEAD (None = HEAD)
    mgr.create::<&str>("new-feature", None).await?;

    // List branches - should include new branch
    let branches = mgr.list().await?;
    let has_new_branch = branches
        .iter()
        .any(|b| b.name == "new-feature" && b.is_local());
    assert!(has_new_branch);

    // Checkout the new branch
    mgr.checkout("new-feature").await?;

    // Current branch should be new-feature
    let current = mgr.current().await?.ok_or("No current branch")?;
    assert_eq!(current.name, "new-feature");
    assert!(current.is_head);
    Ok(())
}

#[tokio::test]
async fn test_branch_delete() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.branch();

    // Get the current branch name (could be main or master)
    let default_branch = mgr.current().await?.ok_or("No current branch")?.name.clone();

    // Create and checkout a temporary branch from HEAD
    mgr.create::<&str>("temp-branch", None).await?;

    // Switch back to default branch
    mgr.checkout(&default_branch).await?;

    // Remove the temporary branch (force=true since no divergent commits)
    mgr.remove("temp-branch", true).await?;

    // Verify it's gone
    let branches = mgr.list().await?;
    let has_temp = branches.iter().any(|b| b.name == "temp-branch");
    assert!(!has_temp);
    Ok(())
}

#[tokio::test]
async fn test_branch_delete_current_fails() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.branch();

    // Get current branch name
    let current = mgr.current().await?.ok_or("No current branch")?;
    let current_name = current.name.clone();

    // Try to remove current branch - should fail
    let result = mgr.remove(&current_name, false).await;
    assert!(result.is_err());
    Ok(())
}

#[tokio::test]
async fn test_branch_rename() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.branch();

    // Create a test branch from HEAD
    mgr.create::<&str>("old-name", None).await?;

    // Rename it
    mgr.rename(Some("old-name"), "new-name").await?;

    // Verify old name is gone and new name exists
    let branches = mgr.list().await?;
    let has_old = branches.iter().any(|b| b.name == "old-name");
    let has_new = branches.iter().any(|b| b.name == "new-name");

    assert!(!has_old);
    assert!(has_new);
    Ok(())
}

#[tokio::test]
async fn test_remote_add_and_list() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.remote();

    // Should have origin remote
    let remotes = mgr.list().await?;
    assert!(remotes.iter().any(|r| r.name == "origin"));

    // Add backup remote
    mgr.add("backup", "https://backup.com/repo.git")
        .await?;

    // Should now have both
    let remotes = mgr.list().await?;
    assert!(remotes.iter().any(|r| r.name == "origin"));
    assert!(remotes.iter().any(|r| r.name == "backup"));
    Ok(())
}

#[tokio::test]
async fn test_remote_set_url() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.remote();

    // Change origin URL
    let new_url = "https://new-host.com/repo.git";
    mgr.set_url("origin", new_url).await?;

    // Verify URL changed
    let remotes = mgr.list().await?;
    let origin = remotes.iter().find(|r| r.name == "origin").ok_or("origin remote not found")?;
    assert_eq!(origin.url, new_url);
    Ok(())
}

#[tokio::test]
async fn test_remote_rename() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.remote();

    // Add a remote
    mgr.add("backup", "https://backup.com/repo.git")
        .await?;

    // Rename it
    mgr.rename("backup", "mirror").await?;

    // Verify rename
    let remotes = mgr.list().await?;
    assert!(!remotes.iter().any(|r| r.name == "backup"));
    assert!(remotes.iter().any(|r| r.name == "mirror"));
    Ok(())
}

#[tokio::test]
async fn test_remote_remove() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.remote();

    // Add a remote
    mgr.add("temp", "https://temp.com/repo.git").await?;

    // Verify it exists
    let remotes = mgr.list().await?;
    assert!(remotes.iter().any(|r| r.name == "temp"));

    // Remove it
    mgr.remove("temp").await?;

    // Verify it's gone
    let remotes = mgr.list().await?;
    assert!(!remotes.iter().any(|r| r.name == "temp"));
    Ok(())
}

#[tokio::test]
async fn test_remote_default() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.remote();

    // Default should be origin
    let default = mgr.default().await?;
    assert!(default.is_some());
    if let Some(default_remote) = default {
        assert_eq!(default_remote.name, "origin");
    }
    Ok(())
}

#[tokio::test]
async fn test_multi_remote_setup() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo_with_branches(&test_repo).await?;

    let url = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    let repo = registry.repo(&repo_id)?;
    let mgr = repo.remote();

    // Setup multiple remotes (simulating distributed setup)
    mgr.add("p2p", "gittorrent://peer/repo").await?;
    mgr.add("backup", "https://backup.com/repo.git")
        .await?;
    mgr.add("mirror", "https://mirror.org/repo.git")
        .await?;

    // List all remotes
    let remotes = mgr.list().await?;
    assert_eq!(remotes.len(), 4); // origin + 3 new ones

    let names: Vec<_> = remotes.iter().map(|r| r.name.as_str()).collect();
    assert!(names.contains(&"origin"));
    assert!(names.contains(&"p2p"));
    assert!(names.contains(&"backup"));
    assert!(names.contains(&"mirror"));
    Ok(())
}
