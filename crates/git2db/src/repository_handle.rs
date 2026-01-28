//! Repository handle for git-native operations
//!
//! Provides a scoped view into a tracked repository with familiar git operations

use crate::branch::BranchManager;
use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::GitRef;
use crate::registry::{Git2DB, RepoId, TrackedRepository};
use crate::remote::RemoteManager;
use crate::storage::{DriverOpts, WorktreeHandle};
use crate::stage::StageManager;
use git2::{Oid, Repository, Signature};
use std::path::{Path, PathBuf};
use tracing::info;

/// Handle to a tracked repository
///
/// Provides git-native operations on a repository tracked in the registry.
/// This handle borrows from the Git2DB registry and provides a scoped interface.
pub struct RepositoryHandle<'a> {
    registry: &'a Git2DB,
    repo_id: RepoId,
}

impl<'a> RepositoryHandle<'a> {
    /// Create a new repository handle
    pub(crate) fn new(registry: &'a Git2DB, repo_id: RepoId) -> Self {
        Self { registry, repo_id }
    }

    /// Get the repository ID
    pub fn id(&self) -> &RepoId {
        &self.repo_id
    }

    /// Get the repository metadata
    pub fn metadata(&self) -> Git2DBResult<&TrackedRepository> {
        self.registry.get_by_id(&self.repo_id).ok_or_else(|| {
            Git2DBError::invalid_repository(
                self.repo_id.to_string(),
                "Repository not found in registry",
            )
        })
    }

    /// Get the repository name (if set)
    pub fn name(&self) -> Git2DBResult<Option<&str>> {
        Ok(self.metadata()?.name.as_deref())
    }

    /// Get the worktree path where files are checked out
    pub fn worktree(&self) -> Git2DBResult<&Path> {
        Ok(&self.metadata()?.worktree_path)
    }

    /// Get the primary URL
    pub fn url(&self) -> Git2DBResult<&str> {
        Ok(&self.metadata()?.url)
    }

    /// Get the tracking ref (branch, tag, or commit)
    pub fn tracking_ref(&self) -> Git2DBResult<&GitRef> {
        Ok(&self.metadata()?.tracking_ref)
    }

    /// Get the current OID (commit hash)
    pub fn current_oid(&self) -> Git2DBResult<Option<Oid>> {
        match &self.metadata()?.current_oid {
            Some(oid_str) => Ok(Some(Oid::from_str(oid_str).map_err(|e| {
                Git2DBError::internal(format!("Invalid OID in metadata: {e}"))
            })?)),
            None => Ok(None),
        }
    }

    /// Open the underlying git repository
    pub fn open_repo(&self) -> Git2DBResult<Repository> {
        let path = self.worktree()?;
        Repository::open(path)
            .map_err(|e| Git2DBError::repository(path, format!("Failed to open repository: {e}")))
    }

    /// Checkout a specific reference (branch, tag, or commit)
    ///
    /// Accepts strings, GitRef enums, or direct Oids for type safety.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // String reference
    /// repo.checkout("main").await?;
    ///
    /// // Explicit GitRef
    /// repo.checkout(git2db::GitRef::Branch("develop".into())).await?;
    ///
    /// // Direct Oid (type-safe)
    /// let oid = git2::Oid::from_str("abc123...")?;
    /// repo.checkout(oid).await?;
    /// # Ok(())
    /// # }
    /// ```
    /// Get remote manager for this repository
    ///
    /// Provides git-native remote operations like `git remote`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Add remotes
    /// repo.remote().add("origin", "https://github.com/user/repo.git").await?;
    /// repo.remote().add("p2p", "gittorrent://peer/repo").await?;
    ///
    /// // List remotes
    /// for remote in repo.remote().list().await? {
    ///     println!("{}: {}", remote.name, remote.url);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn remote(&self) -> RemoteManager<'a> {
        RemoteManager::new(self.registry, self.repo_id.clone())
    }

    /// Get branch manager for this repository
    ///
    /// Provides git-native branch operations like `git branch`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // List branches
    /// for branch in repo.branch().list().await? {
    ///     println!("{} {}", if branch.is_head { "*" } else { " " }, branch.name);
    /// }
    ///
    /// // Get current branch
    /// if let Some(current) = repo.branch().current().await? {
    ///     println!("On branch: {}", current.name);
    /// }
    ///
    /// // Create and checkout
    /// repo.branch().create("feature", Some("main")).await?;
    /// repo.branch().checkout("feature").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn branch(&self) -> BranchManager<'a> {
        BranchManager::new(self.registry, self.repo_id.clone())
    }

    /// Get staging area manager for this repository
    ///
    /// Provides git-native staging operations like `git add` and `git rm`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Add files to staging area
    /// repo.staging().add("src/main.rs").await?;
    /// repo.staging().add_all().await?;
    ///
    /// // Check staged files
    /// for file in repo.staging().staged_files().await? {
    ///     println!("Staged: {:?}", file.path);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn staging(&self) -> StageManager<'a> {
        StageManager::new(self.registry, self.repo_id.clone())
    }

    /// Commit staged changes with default signature
    ///
    /// Similar to `git commit -m "<message>"`
    ///
    /// This method commits all staged changes using the default signature from
    /// GitManager configuration. The signature is created from git config or
    /// the Git2DB configuration.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Stage changes
    /// repo.staging().add_all().await?;
    ///
    /// // Commit with default signature
    /// let oid = repo.commit("Update documentation").await?;
    /// println!("Committed: {}", oid);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - There are no staged changes
    /// - The repository cannot be opened
    /// - The signature cannot be created (check git config)
    /// - The commit cannot be created
    pub async fn commit(&self, message: &str) -> Git2DBResult<Oid> {
        // Get default signature from the global GitManager
        let git_manager = crate::manager::GitManager::global();
        let sig = git_manager.create_signature(None, None)?;

        self.commit_as(&sig, message).await
    }

    /// Commit staged changes with a specific author/committer
    ///
    /// Similar to `git commit -m "<message>" --author="<name> <email>"`
    ///
    /// This method commits all staged changes using the provided signature.
    /// Both author and committer will be set to the provided signature.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// use git2::Signature;
    ///
    /// // Stage changes
    /// repo.staging().add("README.md").await?;
    ///
    /// // Create custom signature
    /// let sig = Signature::now("Alice", "alice@example.com")?;
    ///
    /// // Commit as specific user
    /// let oid = repo.commit_as(&sig, "Update docs").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - There are no staged changes
    /// - The repository cannot be opened
    /// - The commit cannot be created
    pub async fn commit_as(&self, signature: &Signature<'_>, message: &str) -> Git2DBResult<Oid> {
        let repo_path = self.worktree()?.to_path_buf();
        let message = message.to_owned();
        // Clone signature data to move into the blocking task
        // Note: Signature has a lifetime, so we need to extract the data
        let sig_name = signature.name().unwrap_or("Unknown").to_owned();
        let sig_email = signature.email().unwrap_or("unknown@example.com").to_owned();
        // Preserve the original timestamp from the signature
        let sig_time = signature.when();
        let sig_time_seconds = sig_time.seconds();
        let sig_time_offset = sig_time.offset_minutes();

        tracing::info!(message = %message, author = %sig_name, "Creating commit");

        tokio::task::spawn_blocking(move || -> Git2DBResult<Oid> {
            let repo = Repository::open(&repo_path).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}"))
            })?;

            // Get the current index
            let mut index = repo
                .index()
                .map_err(|e| Git2DBError::internal(format!("Failed to get index: {e}")))?;

            // Write index to tree
            let tree_oid = index
                .write_tree()
                .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {e}")))?;

            let tree = repo
                .find_tree(tree_oid)
                .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {e}")))?;

            // Get parent commit (if any)
            let parent_commits = if let Ok(head) = repo.head() {
                vec![head.peel_to_commit().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD commit: {e}"))
                })?]
            } else {
                vec![]
            };

            let parent_refs: Vec<&git2::Commit> = parent_commits.iter().collect();

            // Check for empty commits (nothing staged that differs from parent)
            if let Some(parent) = parent_refs.first() {
                if parent.tree_id() == tree_oid {
                    return Err(Git2DBError::invalid_operation(
                        "Nothing to commit: no changes staged",
                    ));
                }
            }

            // Recreate signature in the blocking task, preserving the original timestamp
            let time = git2::Time::new(sig_time_seconds, sig_time_offset);
            let sig = Signature::new(&sig_name, &sig_email, &time)
                .map_err(|e| Git2DBError::internal(format!("Failed to create signature: {e}")))?;

            // Create the commit
            let commit_oid = repo
                .commit(
                    Some("HEAD"),
                    &sig,
                    &sig,
                    &message,
                    &tree,
                    &parent_refs,
                )
                .map_err(|e| Git2DBError::internal(format!("Failed to create commit: {e}")))?;

            Ok(commit_oid)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Get repository status (branch, dirty files, etc.)
    ///
    /// Similar to `git status` but returns structured data instead of printing.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// let status = repo.status().await?;
    /// println!("Branch: {:?}", status.branch);
    /// println!("Clean: {}", status.is_clean);
    /// if !status.is_clean {
    ///     for file in &status.modified_files {
    ///         println!("Modified: {}", file.display());
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The repository cannot be opened
    /// - The status cannot be retrieved
    pub async fn status(&self) -> Git2DBResult<RepositoryStatus> {
        let repo_path = self.worktree()?.to_path_buf();

        tokio::task::spawn_blocking(move || {
            let repo = crate::manager::GitManager::global()
                .get_repository(&repo_path)
                .map_err(|e| Git2DBError::repository(&repo_path, format!("Failed to get repository: {e}")))?
                .open()
                .map_err(|e| Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}")))?;

            // Get HEAD
            let head = repo.head().ok();
            let branch = head.as_ref().and_then(|h| h.shorthand().map(String::from));
            let head_oid = head.as_ref().and_then(git2::Reference::target);

            // Get statuses
            let statuses = repo
                .statuses(None)
                .map_err(|e| Git2DBError::internal(format!("Failed to get statuses: {e}")))?;

            let is_clean = statuses.is_empty();

            // Collect modified file paths
            let modified_files: Vec<PathBuf> = statuses
                .iter()
                .filter_map(|entry| entry.path().map(PathBuf::from))
                .collect();

            Ok(RepositoryStatus {
                branch,
                head: head_oid,
                ahead: 0,  // TODO: Implement ahead/behind detection
                behind: 0, // TODO: Implement ahead/behind detection
                is_clean,
                modified_files,
            })
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Merge a branch or reference into the current HEAD
    ///
    /// Similar to `git merge <source>`
    ///
    /// # Arguments
    ///
    /// * `source` - The branch or ref to merge (e.g., "feature-branch", "refs/tags/v1.0")
    /// * `message` - Optional commit message (defaults to "Merge branch 'source'")
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Merge a branch
    /// let merge_oid = repo.merge("feature-branch", None).await?;
    ///
    /// // Merge with custom message
    /// let merge_oid = repo.merge("feature-branch", Some("Merge feature-branch")).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The repository cannot be opened
    /// - The source ref cannot be found
    /// - Merge conflicts are detected
    /// - The merge commit cannot be created
    pub async fn merge(&self, source: &str, message: Option<&str>) -> Git2DBResult<Oid> {
        let repo_path = self.worktree()?.to_path_buf();
        let source = source.to_owned();
        let message = message.map(std::borrow::ToOwned::to_owned);

        tokio::task::spawn_blocking(move || {
            let repo = crate::manager::GitManager::global()
                .get_repository(&repo_path)
                .map_err(|e| Git2DBError::repository(&repo_path, format!("Failed to get repository: {e}")))?
                .open()
                .map_err(|e| Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}")))?;

            // Get default signature from the global GitManager
            let git_manager = crate::manager::GitManager::global();
            let sig = git_manager.create_signature(None, None)?;

            // Find the source ref to merge
            let (obj, _src_ref) = repo
                .revparse_ext(&source)
                .map_err(|e| Git2DBError::reference(&source, format!("Failed to find ref: {e}")))?;

            let source_commit = obj
                .peel_to_commit()
                .map_err(|e| Git2DBError::reference(&source, format!("Failed to peel to commit: {e}")))?;

            // Get annotated commit for merge
            let source_annotated = repo
                .find_annotated_commit(source_commit.id())
                .map_err(|e| Git2DBError::internal(format!("Failed to find annotated commit: {e}")))?;

            // Perform the merge analysis
            let (analysis, _preference) = repo.merge_analysis(&[&source_annotated])
                .map_err(|e| Git2DBError::internal(format!("Failed to analyze merge: {e}")))?;

            // Check if up-to-date
            if analysis.contains(git2::MergeAnalysis::ANALYSIS_UP_TO_DATE) {
                return Err(Git2DBError::invalid_operation("Already up to date"));
            }

            // Check if fast-forward
            if analysis.contains(git2::MergeAnalysis::ANALYSIS_FASTFORWARD) {
                // Do fast-forward merge
                let source_obj = source_commit.as_object();
                repo.checkout_tree(source_obj, None)
                    .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {e}")))?;

                repo.set_head(&format!("refs/heads/{source}"))
                    .map_err(|e| Git2DBError::internal(format!("Failed to set HEAD: {e}")))?;

                return Ok(source_commit.id());
            }

            // Normal merge required - commit merge
            repo.merge(&[&source_annotated], None, None)
                .map_err(|e| Git2DBError::internal(format!("Failed to perform merge: {e}")))?;

            // Check for conflicts again after merge
            let mut index = repo
                .index()
                .map_err(|e| Git2DBError::internal(format!("Failed to get index: {e}")))?;

            if index.has_conflicts() {
                repo.cleanup_state()
                    .map_err(|e| Git2DBError::internal(format!("Failed to cleanup merge state: {e}")))?;
                return Err(Git2DBError::invalid_operation("Merge conflicts detected"));
            }

            // Write tree and create merge commit
            let tree_id = index
                .write_tree()
                .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {e}")))?;

            let tree = repo
                .find_tree(tree_id)
                .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {e}")))?;

            // Get parent commits
            let head = repo
                .head()
                .map_err(|e| Git2DBError::internal(format!("Failed to get HEAD: {e}")))?
                .peel_to_commit()
                .map_err(|e| Git2DBError::internal(format!("Failed to resolve HEAD commit: {e}")))?;

            let merge_msg = message.unwrap_or_else(|| format!("Merge branch '{source}'"));

            // Create merge commit
            let merge_oid = repo
                .commit(
                    Some("HEAD"),
                    &sig,
                    &sig,
                    &merge_msg,
                    &tree,
                    &[&head, &source_commit],
                )
                .map_err(|e| Git2DBError::internal(format!("Failed to create merge commit: {e}")))?;

            // Cleanup merge state
            repo.cleanup_state()
                .map_err(|e| Git2DBError::internal(format!("Failed to cleanup merge state: {e}")))?;

            Ok(merge_oid)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// List all references (branches and tags) with their OIDs
    ///
    /// Similar to `git show-ref`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// for (name, oid) in repo.list_refs()? {
    ///     println!("{}: {}", name, oid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn list_refs(&self) -> Git2DBResult<Vec<(String, Oid)>> {
        let repo = self.open_repo()?;
        let mut refs = Vec::new();

        for reference in repo
            .references()
            .map_err(|e| Git2DBError::internal(format!("Failed to list references: {e}")))?
        {
            let r = reference
                .map_err(|e| Git2DBError::internal(format!("Failed to read reference: {e}")))?;

            if let Some(name) = r.shorthand() {
                if let Some(oid) = r.target() {
                    refs.push((name.to_owned(), oid));
                }
            }
        }

        Ok(refs)
    }

    /// Get the default branch name
    ///
    /// Similar to `git symbolic-ref --short HEAD` or detecting main/master
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// let default_branch = repo.default_branch()?;
    /// println!("Default branch: {}", default_branch);
    /// # Ok(())
    /// # }
    /// ```
    pub fn default_branch(&self) -> Git2DBResult<String> {
        let repo = self.open_repo()?;

        // Try to get the symbolic reference HEAD points to
        if let Ok(head_ref) = repo.head() {
            if let Some(name) = head_ref.symbolic_target() {
                // Remove refs/heads/ prefix to get just the branch name
                if let Some(branch_name) = name.strip_prefix("refs/heads/") {
                    return Ok(branch_name.to_owned());
                }
            }
        }

        // If HEAD doesn't point to a symbolic ref, try to find the default branch
        // Check for common default branch names
        for default_name in ["main", "master"] {
            if repo
                .find_branch(default_name, git2::BranchType::Local)
                .is_ok()
            {
                return Ok(default_name.to_owned());
            }
        }

        // Fallback: get the first branch
        let mut branches = repo
            .branches(Some(git2::BranchType::Local))
            .map_err(|e| Git2DBError::internal(format!("Failed to list branches: {e}")))?;

        if let Some(Ok((branch, _))) = branches.next() {
            if let Some(name) = branch
                .name()
                .map_err(|e| Git2DBError::internal(format!("Failed to get branch name: {e}")))?
            {
                return Ok(name.to_owned());
            }
        }

        // Final fallback
        Ok("main".to_owned())
    }

    /// Resolve a GitRef to an OID
    ///
    /// This is a general-purpose method that can resolve any type of git reference
    /// to its commit OID.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// use git2db::GitRef;
    ///
    /// // Resolve branch
    /// let oid = repo.resolve_git_ref(&GitRef::Branch("main".into())).await?;
    ///
    /// // Resolve tag
    /// let oid = repo.resolve_git_ref(&GitRef::Tag("v1.0.0".into())).await?;
    ///
    /// // Resolve commit (returns same OID)
    /// let oid = repo.resolve_git_ref(&GitRef::Commit(oid)).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn resolve_git_ref(&self, git_ref: &GitRef) -> Git2DBResult<Oid> {
        let repo = self.open_repo()?;

        match git_ref {
            GitRef::DefaultBranch => {
                let head = repo.head().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD: {e}"))
                })?;
                head.peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to peel to commit: {e}"))
                })
            }
            GitRef::Branch(branch_name) => {
                let branch = repo
                    .find_branch(branch_name, git2::BranchType::Local)
                    .or_else(|_| repo.find_branch(branch_name, git2::BranchType::Remote))
                    .map_err(|e| {
                        Git2DBError::reference(branch_name, format!("Branch not found: {e}"))
                    })?;
                branch.get().peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference(branch_name, format!("Failed to peel to commit: {e}"))
                })
            }
            GitRef::Tag(tag_name) => {
                let reference = repo
                    .find_reference(&format!("refs/tags/{tag_name}"))
                    .map_err(|e| {
                        Git2DBError::reference(tag_name, format!("Tag not found: {e}"))
                    })?;
                reference.peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference(tag_name, format!("Failed to peel to commit: {e}"))
                })
            }
            GitRef::Commit(oid) => Ok(*oid),
            GitRef::Revspec(spec) => {
                let obj = repo.revparse_single(spec).map_err(|e| {
                    Git2DBError::reference(spec, format!("Failed to resolve revspec: {e}"))
                })?;
                obj.peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference(spec, format!("Failed to peel to commit: {e}"))
                })
            }
        }
    }

    /// Get information about a specific reference
    ///
    /// Returns the reference name and target OID if it exists.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// if let Some((name, oid)) = repo.ref_info("main")? {
    ///     println!("Branch {}: {}", name, oid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn ref_info(&self, ref_name: &str) -> Git2DBResult<Option<(String, Oid)>> {
        let repo = self.open_repo()?;

        // Try as branch first
        let ref_path = if ref_name.starts_with("refs/") {
            ref_name.to_owned()
        } else if repo.find_branch(ref_name, git2::BranchType::Local).is_ok() {
            format!("refs/heads/{ref_name}")
        } else if repo
            .find_reference(&format!("refs/tags/{ref_name}"))
            .is_ok()
        {
            format!("refs/tags/{ref_name}")
        } else {
            ref_name.to_owned()
        };

        // Find and extract data before returning
        let result = match repo.find_reference(&ref_path) {
            Ok(reference) => {
                let name = reference.shorthand().unwrap_or(ref_name).to_owned();
                let oid_opt = reference.target();
                (Some(name), oid_opt)
            }
            Err(_) => (None, None),
        };

        match result {
            (Some(name), Some(oid)) => Ok(Some((name, oid))),
            _ => Ok(None),
        }
    }

    /// Resolve a revspec string to an OID
    ///
    /// Similar to `git rev-parse`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Resolve complex revspecs
    /// let oid = repo.resolve_revspec("HEAD~3").await?;
    /// let oid = repo.resolve_revspec("main@{yesterday}").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn resolve_revspec(&self, spec: &str) -> Git2DBResult<Oid> {
        let repo = self.open_repo()?;
        let obj = repo.revparse_single(spec).map_err(|e| {
            Git2DBError::reference(spec, format!("Failed to resolve revspec: {e}"))
        })?;
        Ok(obj.id())
    }

    /// Get all worktrees managed by the storage driver for this repository
    ///
    /// Returns all worktrees that were created using storage drivers for this repository.
    /// This is useful for discovering existing worktrees managed by overlay2, vfs, etc.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // List all worktrees for this repository
    /// for worktree in repo.get_worktrees().await? {
    ///     println!("Worktree: {} (driver: {})", worktree.path().display(), worktree.driver_name());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_worktrees(&self) -> Git2DBResult<Vec<WorktreeHandle>> {
        let tracked_repo = self.metadata()?;

        // Use pre-loaded storage driver from registry
        let driver = self.registry.storage_driver().clone();

        // Query storage driver directly for worktrees
        driver.get_worktrees(&tracked_repo.worktree_path).await
    }

    /// Get a specific worktree by branch name
    ///
    /// Returns the worktree if it exists and is managed by the storage driver.
    /// This is useful for checking if a specific branch worktree exists.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Check if a worktree for a specific branch exists
    /// if let Some(worktree) = repo.get_worktree("feature-branch").await? {
    ///     println!("Found worktree at: {}", worktree.path().display());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_worktree(&self, branch: &str) -> Git2DBResult<Option<WorktreeHandle>> {
        let tracked_repo = self.metadata()?;

        // Use pre-loaded storage driver from registry
        let driver = self.registry.storage_driver().clone();

        // Query storage driver directly for specific worktree
        driver.get_worktree(&tracked_repo.worktree_path, branch).await
    }

    /// Create a new worktree using the configured storage driver
    ///
    /// This creates a new worktree using the storage driver selected in the configuration.
    /// On Linux, this will typically use overlay2 for ~80% space savings, while on other
    /// platforms it will use vfs (plain git worktrees).
    ///
    /// The worktree will be created at the specified path and checked out to the given branch/ref.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a worktree for a feature branch with overlay2 optimization
    /// let worktree = repo.create_worktree("/tmp/feature-branch", "feature-branch").await?;
    /// println!("Created worktree at: {}", worktree.path().display());
    ///
    /// // Create worktree at custom path
    /// let worktree = repo.create_worktree("/tmp/custom-path", "main").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Fetch LFS files for the worktree
    ///
    /// This function tries XET first if initialized (faster for XET-enabled repos),
    /// then falls back to `git lfs pull` if XET fails or isn't available.
    pub async fn fetch_lfs_files(repo_path: &Path) -> Git2DBResult<()> {
        if !repo_path.exists() {
            return Err(Git2DBError::internal(format!(
                "Worktree path does not exist: {}",
                repo_path.display()
            )));
        }

        let gitattributes_path = repo_path.join(".gitattributes");

        let uses_lfs = if gitattributes_path.exists() {
            tokio::fs::read_to_string(&gitattributes_path)
                .await
                .map(|content| content.contains("filter=lfs"))
                .unwrap_or(false)
        } else {
            false
        };

        if !uses_lfs {
            tracing::debug!("Repository does not use Git LFS, skipping LFS fetch");
            return Ok(());
        }

        // Try XET first if initialized (faster for XET-enabled repos)
        #[cfg(feature = "xet-storage")]
        if crate::xet_filter::is_initialized() {
            tracing::info!("XET initialized, attempting XET fetch for LFS files...");
            match Self::try_xet_lfs_fetch(repo_path).await {
                Ok(stats) => {
                    if stats.processed > 0 || stats.skipped > 0 {
                        tracing::info!(
                            "XET fetch complete: {} processed, {} skipped, {} failed",
                            stats.processed, stats.skipped, stats.failed
                        );
                        if stats.failed == 0 {
                            return Ok(());
                        }
                        // Some files failed, fall through to git-lfs for remaining
                        tracing::warn!("Some files failed via XET, falling back to git-lfs for remaining files");
                    }
                }
                Err(e) => {
                    tracing::warn!("XET fetch failed: {}. Falling back to git-lfs...", e);
                }
            }
        }

        // Fallback to git lfs pull
        Self::run_git_lfs_pull(repo_path).await
    }

    /// Try to fetch LFS files via XET
    #[cfg(feature = "xet-storage")]
    async fn try_xet_lfs_fetch(repo_path: &Path) -> Git2DBResult<crate::lfs::ProcessingStats> {
        use crate::lfs::LfsStorage;

        // Get the current XET config
        let config = crate::xet_filter::get_config().ok_or_else(|| {
            Git2DBError::internal("BUG: XET filter initialized but FILTER_CONFIG was not set")
        })?;

        let storage = LfsStorage::new(&config).await.map_err(|e| {
            Git2DBError::internal(format!("Failed to create LFS storage: {e}"))
        })?;

        storage.process_worktree(repo_path).await
    }

    /// Run git lfs pull as fallback
    async fn run_git_lfs_pull(repo_path: &Path) -> Git2DBResult<()> {
        use tokio::process::Command;

        let lfs_available = Command::new("git")
            .args(["lfs", "version"])
            .output()
            .await
            .map(|output| output.status.success())
            .unwrap_or(false);

        if !lfs_available {
            tracing::warn!("git-lfs not installed or not in PATH");
            return Err(Git2DBError::internal(
                "git-lfs not available. Install with: sudo apt install git-lfs"
            ));
        }

        tracing::info!("Fetching LFS files via git-lfs (this may take a while for large files)...");
        println!("📥 Downloading LFS files... (this may take a while for large models)");

        let mut child = Command::new("git")
            .args(["lfs", "pull"])
            .current_dir(repo_path)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| {
                Git2DBError::internal(format!(
                    "Failed to spawn 'git lfs pull': {e}. Is git-lfs installed?"
                ))
            })?;

        let status = child.wait().await.map_err(|e| {
            Git2DBError::internal(format!("Failed to wait for 'git lfs pull': {e}"))
        })?;

        if !status.success() {
            let exit_code = status.code().unwrap_or(-1);
            tracing::error!("git lfs pull failed with exit code: {}", exit_code);

            let error_msg = match exit_code {
                128 => "git lfs pull failed (exit code 128): Authentication failed.\n\
                     Check your git credentials or LFS endpoint.".to_owned(),
                1 => "git lfs pull failed (exit code 1): Network error.\n\
                     Check network connectivity and LFS endpoint.".to_owned(),
                _ => format!(
                    "git lfs pull failed (exit code {exit_code}): Unknown error.\n\
                     Check logs above for details."
                ),
            };

            return Err(Git2DBError::internal(error_msg));
        }

        tracing::info!("Successfully fetched LFS files via git-lfs");
        Ok(())
    }

    pub async fn create_worktree(&self, worktree_path: &Path, branch: &str) -> Git2DBResult<WorktreeHandle> {
        let tracked_repo = self.metadata()?;

        // Use pre-loaded storage driver from registry
        let driver = self.registry.storage_driver().clone();

        info!(
            "Creating worktree with {} driver: base={}, path={}, ref={}",
            driver.name(),
            tracked_repo.worktree_path.display(),
            worktree_path.display(),
            branch
        );

        // Create driver options
        let opts = DriverOpts {
            base_repo: tracked_repo.worktree_path.clone(),
            worktree_path: worktree_path.to_path_buf(),
            ref_spec: branch.to_owned(),
        };

        // Create worktree using driver
        let result = driver.create_worktree(&opts).await;

        match &result {
            Ok(_handle) => {
                // Already logged above
            }
            Err(e) => {
                tracing::error!(
                    "Failed to create worktree at {}: {}",
                    worktree_path.display(),
                    e
                );

                // Provide helpful hints
                #[cfg(feature = "overlayfs")]
                {
                    if driver.name() == "overlay2" {
                        tracing::error!(
                            "Hint: Install fuse-overlayfs:\n  \
                             sudo apt install fuse-overlayfs\n  \
                             OR try a different driver like 'vfs'"
                        );
                    }
                }
            }
        }

        match result {
            Ok(mut handle) => {
                if let Err(e) = Self::fetch_lfs_files(worktree_path).await {
                    tracing::error!(
                        "Failed to fetch LFS files for worktree at {}: {}",
                        worktree_path.display(),
                        e
                    );
                    tracing::info!("Rolling back worktree creation due to LFS fetch failure");

                    if let Err(cleanup_err) = handle.cleanup().await {
                        tracing::error!(
                            "Failed to cleanup worktree during rollback: {}",
                            cleanup_err
                        );
                    }

                    return Err(Git2DBError::internal(format!(
                        "Worktree creation failed: LFS fetch error: {e}. Worktree has been rolled back."
                    )));
                }

                Ok(handle)
            }
            Err(e) => {
                tracing::error!(
                    "Failed to create worktree at {}: {}",
                    worktree_path.display(),
                    e
                );

                if worktree_path.exists() {
                    tracing::info!("Cleaning up partial worktree at {}", worktree_path.display());
                    tokio::fs::remove_dir_all(worktree_path)
                        .await
                        .unwrap_or_else(|cleanup_err| {
                            tracing::warn!("Failed to cleanup partial worktree: {}", cleanup_err);
                        });
                }

                Err(e)
            }
        }
    }
}

/// Repository status information
#[derive(Debug, Clone)]
pub struct RepositoryStatus {
    pub branch: Option<String>,
    pub head: Option<Oid>,
    pub ahead: usize,
    pub behind: usize,
    pub is_clean: bool,
    pub modified_files: Vec<PathBuf>,
}
