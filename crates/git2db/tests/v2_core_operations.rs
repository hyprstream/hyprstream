//! Integration tests for git2db v2.0 core operations
//!
//! Tests the complete workflow of repository management with the v2 API

use git2db::config::RepositoryConfig;
use git2db::{CloneBuilder, Git2DB, Git2DBConfig, GitManager};
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

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
    use git2::{Repository, Signature};

    // Initialize git repo
    let repo = Repository::init(path)?;

    // Create initial commit
    // Use explicit signature for CI environments without git config
    let sig = Signature::now("Test User", "test@example.com")?;
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
async fn test_registry_open_and_initialize() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();

    // Open registry (should create new)
    let _registry = Git2DB::open(registry_path).await?;

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
    let _registry2 = Git2DB::open(registry_path).await?;
    Ok(())
}

#[tokio::test]
async fn test_upsert_repository_idempotent() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create a test repository to clone
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir)?;
    create_test_repo(&test_repo_dir).await?;

    let url = format!("file://{}", test_repo_dir.display());

    // First upsert - should clone
    let id1 = registry
        .upsert_repository("test-project", &url)
        .await?;

    // Verify repository exists (2 = .registry self-track + test-project)
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 2);
    assert!(repos.iter().any(|r| r.name.as_deref() == Some("test-project")));

    // Second upsert - should return existing
    let id2 = registry
        .upsert_repository("test-project", &url)
        .await?;

    // Should be the same ID
    assert_eq!(id1, id2);

    // Still only one user repository (+ .registry)
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 2);
    Ok(())
}

#[tokio::test]
async fn test_repository_handle_operations() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create test repository
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir)?;
    create_test_repo(&test_repo_dir).await?;

    let url = format!("file://{}", test_repo_dir.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    // Get repository handle
    let repo = registry.repo(&repo_id)?;

    // Test metadata access
    assert_eq!(repo.name()?, Some("test-repo"));
    assert_eq!(repo.url()?, url);

    // Test worktree access (this is the cloned repo, not a separate worktree)
    let worktree_path = repo.worktree()?;
    assert!(worktree_path.exists());
    assert!(worktree_path.join("README.md").exists());

    // Test default branch detection
    let default_branch = repo.default_branch()?;
    assert!(!default_branch.is_empty(), "Should detect default branch");

    // Note: get_worktree() is for finding separately created worktrees,
    // not the main cloned repository. The cloned repo is accessible via worktree().
    Ok(())
}

#[tokio::test]
async fn test_remove_repository_cleanup() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create and add test repository
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir)?;
    create_test_repo(&test_repo_dir).await?;

    let url = format!("file://{}", test_repo_dir.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;

    // Verify repository exists
    let repo = registry.repo(&repo_id)?;
    let worktree_path = repo.worktree()?.to_path_buf();
    assert!(worktree_path.exists());

    // Get internal paths before removal
    let internal_registry = registry_path.join(".registry");
    let submodule_path = internal_registry.join("repos").join(repo_id.to_string());
    let modules_path = internal_registry
        .join(".git/modules")
        .join(repo_id.to_string());

    // Remove repository
    registry.remove_repository(&repo_id).await?;

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
        let content = fs::read_to_string(&gitmodules_path)?;
        assert!(
            !content.contains(&repo_id.to_string()),
            ".gitmodules should not contain removed submodule"
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_repository_handle_commit() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create test repository
    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo(&test_repo).await?;
    let url = format!("file://{}", test_repo.display());

    // Add repository to registry
    let repo_id = registry.add_repository("test-repo", &url).await?;
    let handle = registry.repo(&repo_id)?;

    // Make a change to the repository
    let worktree_path = handle.worktree()?;
    let test_file = worktree_path.join("test.txt");
    fs::write(&test_file, "Hello, world!")?;

    // Stage the change
    handle.staging().add("test.txt").await?;

    // Commit using the new commit API
    let commit_oid = handle.commit("Add test file").await?;

    // Verify the commit was created
    assert!(!commit_oid.is_zero(), "Commit OID should not be zero");

    // Verify the file is in the commit
    let repo = handle.open_repo()?;
    let commit = repo.find_commit(commit_oid)?;
    let tree = commit.tree()?;
    assert!(tree.get_path(std::path::Path::new("test.txt")).is_ok(), "test.txt should be in the commit");

    // Verify commit message (git2 may or may not include trailing newline)
    let msg = commit.message().ok_or("Commit message not found")?;
    assert!(msg.starts_with("Add test file"), "Commit message should start with expected text");

    // Test commit_as with custom signature
    let test_file2 = worktree_path.join("test2.txt");
    fs::write(&test_file2, "Another file")?;
    handle.staging().add("test2.txt").await?;

    use git2::Signature;
    let custom_sig = Signature::now("Test User", "test@example.com")?;
    let commit_oid2 = handle.commit_as(&custom_sig, "Add second test file").await?;

    // Verify custom signature was used
    let commit2 = repo.find_commit(commit_oid2)?;
    assert_eq!(commit2.author().name(), Some("Test User"));
    assert_eq!(commit2.author().email(), Some("test@example.com"));
    Ok(())
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
async fn test_update_repository() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create test repository
    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo(&test_repo).await?;

    let url1 = format!("file://{}", test_repo.display());
    let repo_id = registry.add_repository("test-repo", &url1).await?;

    // Verify initial URL
    let repo_info = registry.get_by_id(&repo_id).ok_or("Repository not found")?;
    assert_eq!(repo_info.url, url1);

    // Update URL
    let new_url = "https://new-host.com/repo.git";
    registry
        .update_repository(&repo_id, Some(new_url.to_owned()))
        .await?;

    // Verify URL was updated in metadata
    let repo_info = registry.get_by_id(&repo_id).ok_or("Repository not found after update")?;
    assert_eq!(repo_info.url, new_url);

    // Note: repo_info.remotes tracks registry-level remote config, not git remotes.
    // The actual git "origin" remote is set during clone and can be verified via git2.
    Ok(())
}

#[tokio::test]
async fn test_repo_by_name_lookup() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create test repository
    let test_repo = temp_dir.path().join("repo");
    fs::create_dir(&test_repo)?;
    create_test_repo(&test_repo).await?;
    let url = format!("file://{}", test_repo.display());

    // Add repository with name
    let repo_id = registry.add_repository("my-project", &url).await?;

    // Lookup by name
    let repo = registry.repo_by_name("my-project")?;
    assert_eq!(repo.id(), &repo_id);

    // Lookup by ID
    let repo2 = registry.repo(&repo_id)?;
    assert_eq!(repo2.name()?, Some("my-project"));

    // Non-existent name should fail
    let result = registry.repo_by_name("non-existent");
    assert!(result.is_err());
    Ok(())
}

#[tokio::test]
async fn test_list_repositories() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Only the self-tracked .registry entry
    assert_eq!(registry.list().count(), 1);

    // Create test repositories
    for i in 1..=3 {
        let test_repo = temp_dir.path().join(format!("repo{i}"));
        fs::create_dir(&test_repo)?;
        create_test_repo(&test_repo).await?;
        let url = format!("file://{}", test_repo.display());

        registry
            .add_repository(&format!("repo{i}"), &url)
            .await?;
    }

    // Should have 3 repositories + .registry self-tracking entry
    let repos: Vec<_> = registry.list().collect();
    assert_eq!(repos.len(), 4);

    // Check display names
    let names: Vec<_> = repos.iter().map(|r| r.display_name()).collect();
    assert!(names.contains(&"repo1".to_owned()));
    assert!(names.contains(&"repo2".to_owned()));
    assert!(names.contains(&"repo3".to_owned()));
    Ok(())
}

#[tokio::test]
async fn test_registry_clone_with_auth_callbacks() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let registry = Arc::new(RwLock::new(Git2DB::open(registry_path).await?));

    // Create a test repository to clone
    let test_repo = temp_dir.path().join("source-repo");
    fs::create_dir(&test_repo)?;
    create_test_repo(&test_repo).await?;
    let url = format!("file://{}", test_repo.display());

    // Test that CloneBuilder works (this uses bare clone with auth callbacks)
    // Even though file:// URLs don't require auth, this verifies the code path
    // that sets up authentication callbacks is being exercised
    let repo_id = CloneBuilder::new(Arc::clone(&registry), &url)
        .name("cloned-repo")
        .exec()
        .await?;

    // Verify the repository was cloned successfully
    let registry_guard = registry.read().await;
    let handle = registry_guard
        .repo(&repo_id)?;
    assert_eq!(
        handle.name()?,
        Some("cloned-repo")
    );

    // Verify the bare repository exists
    // Note: handle.worktree() returns the bare repo path for cloned repos
    let bare_repo_path = handle
        .worktree()?;
    assert!(bare_repo_path.exists(), "Bare repository path should exist");

    // Verify it's actually a bare repository
    use git2::Repository;
    let bare_repo = Repository::open_bare(bare_repo_path)?;
    assert!(bare_repo.is_bare(), "Repository should be bare");

    // Verify the default worktree exists and has the expected files
    // CloneBuilder creates worktrees at {repo_dir}/worktrees/{branch}
    // The branch name depends on git's init.defaultBranch config (main or master)
    let repo_dir = registry_path.join("cloned-repo");
    let default_branch = bare_repo
        .head()
        .ok()
        .and_then(|h| h.shorthand().map(String::from))
        .unwrap_or_else(|| "main".to_owned());
    let default_worktree = repo_dir.join("worktrees").join(&default_branch);
    assert!(
        default_worktree.exists(),
        "Default worktree should exist at {default_worktree:?}"
    );
    assert!(
        default_worktree.join("README.md").exists(),
        "README.md should exist in the worktree"
    );
    Ok(())
}

#[tokio::test]
async fn test_worktree_status_ahead_behind() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    init_git_manager_no_shallow();

    let temp_dir = TempDir::new()?;
    let registry_path = temp_dir.path();
    let mut registry = Git2DB::open(registry_path).await?;

    // Create a test repository with multiple commits
    let test_repo_dir = temp_dir.path().join("test-repo");
    fs::create_dir(&test_repo_dir)?;

    use git2::{Repository, Signature};
    let repo = Repository::init(&test_repo_dir)?;
    // Use explicit signature for CI environments without git config
    let sig = Signature::now("Test User", "test@example.com")?;

    // Create initial commit
    let mut index = repo.index()?;
    let readme_path = test_repo_dir.join("README.md");
    fs::write(&readme_path, "# Test Repository\n")?;
    index.add_path(std::path::Path::new("README.md"))?;
    index.write()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;
    repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])?;

    // Create a second commit (local is ahead)
    let file2_path = test_repo_dir.join("file2.txt");
    fs::write(&file2_path, "Content\n")?;
    let mut index = repo.index()?;
    index.add_path(std::path::Path::new("file2.txt"))?;
    index.write()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;
    let head = repo.head()?.peel_to_commit()?;
    repo.commit(Some("HEAD"), &sig, &sig, "Second commit", &tree, &[&head])?;

    // Add repository to registry
    let url = format!("file://{}", test_repo_dir.display());
    let repo_id = registry.add_repository("test-repo", &url).await?;
    let handle = registry.repo(&repo_id)?;

    // Create a new branch "test-branch" for the worktree (can't use main since it's already checked out)
    let base_repo_path = handle.worktree()?;
    {
        let base_repo = Repository::open(base_repo_path)?;
        let head_commit = base_repo.head()?.peel_to_commit()?;
        base_repo.branch("test-branch", &head_commit, false)?;
    }

    // Create a worktree for "test-branch"
    let worktree_path = temp_dir.path().join("worktrees").join("test-branch");
    let mut worktree_handle = handle.create_worktree(&worktree_path, "test-branch").await?;

    // Get status - should show ahead: 0, behind: 0 since there's no upstream tracking
    let status = worktree_handle.status().await?;
    assert_eq!(status.ahead, 0, "Should be 0 ahead when no upstream is configured");
    assert_eq!(status.behind, 0, "Should be 0 behind when no upstream is configured");

    // Now set up upstream tracking by creating a "remote" branch in the same repo
    // and setting it as upstream
    let repo_path = &worktree_path;
    let repo = Repository::open(repo_path)?;

    // Create a remote branch reference (simulating origin/test-branch)
    let head_commit = repo.head()?.peel_to_commit()?;
    let test_branch = "test-branch";
    let remote_ref_name = format!("refs/remotes/origin/{test_branch}");
    repo.reference(&remote_ref_name, head_commit.id(), false, "Create remote ref")?;

    // Set upstream for the current branch (test-branch)
    let mut branch = repo.find_branch(test_branch, git2::BranchType::Local)?;
    branch.set_upstream(Some(&format!("origin/{test_branch}")))?;

    // Now make another local commit (we'll be ahead of the remote)
    let file3_path = repo_path.join("file3.txt");
    fs::write(&file3_path, "More content\n")?;
    let mut index = repo.index()?;
    index.add_path(std::path::Path::new("file3.txt"))?;
    index.write()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;
    let head = repo.head()?.peel_to_commit()?;
    repo.commit(Some("HEAD"), &sig, &sig, "Third commit", &tree, &[&head])?;

    // Get status again - should show ahead: 1, behind: 0
    let status = worktree_handle.status().await?;
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
    let current_head = repo.head()?.peel_to_commit()?;
    let parent_commit = current_head.parent(0)?;

    // Create a "remote" commit D on top of B (the parent)
    let remote_file = repo_path.join("remote_file.txt");
    fs::write(&remote_file, "Remote-only content\n")?;
    let mut index = repo.index()?;
    index.add_path(std::path::Path::new("remote_file.txt"))?;
    index.write()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;
    // Create commit D with parent B (not C)
    let remote_commit = repo.commit(None, &sig, &sig, "Remote commit D", &tree, &[&parent_commit])?;

    // Update the remote ref to point to this new commit D
    repo.reference(&remote_ref_name, remote_commit, true, "Update remote to diverged commit")?;

    // Clean up the working directory (remove the remote_file we added to index)
    fs::remove_file(&remote_file)?;
    // Reset the index to match HEAD
    repo.reset(current_head.as_object(), git2::ResetType::Mixed, None)?;

    // Get status - should show ahead: 1, behind: 1 (diverged branches)
    let status = worktree_handle.status().await?;
    assert_eq!(status.ahead, 1, "Should be 1 commit ahead of upstream");
    assert_eq!(status.behind, 1, "Should be 1 commit behind upstream");
    Ok(())
}
