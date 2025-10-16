//! Repository handle for git-native operations
//!
//! Provides a scoped view into a tracked repository with familiar git operations

use crate::branch::BranchManager;
use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::{GitRef, IntoGitRef};
use crate::registry::{Git2DB, RepoId, TrackedRepository};
use crate::remote::RemoteManager;
use crate::stage::StageManager;
use git2::{Oid, Repository, Signature};
use std::path::{Path, PathBuf};

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
                &self.repo_id.to_string(),
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
                Git2DBError::internal(format!("Invalid OID in metadata: {}", e))
            })?)),
            None => Ok(None),
        }
    }

    /// Open the underlying git repository
    pub fn open_repo(&self) -> Git2DBResult<Repository> {
        let path = self.worktree()?;
        Repository::open(path)
            .map_err(|e| Git2DBError::repository(path, format!("Failed to open repository: {}", e)))
    }

    /// Get repository status
    pub async fn status(&self) -> Git2DBResult<RepositoryStatus> {
        let repo = self.open_repo()?;

        let head = repo.head().ok();
        let branch = head
            .as_ref()
            .and_then(|h| h.shorthand())
            .map(|s| s.to_string());
        let head_oid = head.as_ref().and_then(|h| h.target());

        let statuses = repo.statuses(None).map_err(|e| {
            Git2DBError::internal(format!("Failed to get repository status: {}", e))
        })?;

        let is_clean = statuses.is_empty();
        let modified_files: Vec<PathBuf> = statuses
            .iter()
            .filter_map(|entry| entry.path().map(PathBuf::from))
            .collect();

        Ok(RepositoryStatus {
            branch,
            head: head_oid,
            ahead: 0, // TODO: Calculate ahead/behind
            behind: 0,
            is_clean,
            modified_files,
        })
    }

    /// Checkout a specific reference (branch, tag, or commit)
    ///
    /// Accepts strings, GitRef enums, or direct Oids for type safety.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
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
    pub async fn checkout(&self, reference: impl IntoGitRef) -> Git2DBResult<()> {
        let repo = self.open_repo()?;
        let git_ref = reference.into_git_ref();

        // Resolve the reference to an OID
        let oid = match git_ref {
            GitRef::DefaultBranch => {
                let head = repo.head().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
                })?;
                head.target().ok_or_else(|| {
                    Git2DBError::reference("HEAD", "HEAD is not a direct reference")
                })?
            }
            GitRef::Branch(ref branch_name) => {
                let branch = repo
                    .find_branch(branch_name, git2::BranchType::Local)
                    .or_else(|_| repo.find_branch(branch_name, git2::BranchType::Remote))
                    .map_err(|e| {
                        Git2DBError::reference(branch_name, format!("Branch not found: {}", e))
                    })?;
                branch
                    .get()
                    .target()
                    .ok_or_else(|| Git2DBError::reference(branch_name, "Branch has no target"))?
            }
            GitRef::Tag(ref tag_name) => {
                let reference = repo
                    .find_reference(&format!("refs/tags/{}", tag_name))
                    .map_err(|e| {
                        Git2DBError::reference(tag_name, format!("Tag not found: {}", e))
                    })?;
                reference
                    .target()
                    .ok_or_else(|| Git2DBError::reference(tag_name, "Tag has no target"))?
            }
            GitRef::Commit(oid) => oid,
            GitRef::Revspec(ref spec) => {
                let obj = repo.revparse_single(spec).map_err(|e| {
                    Git2DBError::reference(spec, format!("Failed to resolve revspec: {}", e))
                })?;
                obj.id()
            }
        };

        // Checkout the commit
        let commit = repo
            .find_commit(oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to find commit: {}", e)))?;

        repo.checkout_tree(commit.as_object(), None)
            .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {}", e)))?;

        // Update HEAD
        repo.set_head_detached(oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to update HEAD: {}", e)))?;

        Ok(())
    }

    /// Fetch from a remote
    pub async fn fetch(&self, remote: Option<&str>) -> Git2DBResult<()> {
        let repo = self.open_repo()?;
        let remote_name = remote.unwrap_or("origin");

        let mut remote_obj = repo
            .find_remote(remote_name)
            .map_err(|e| Git2DBError::reference(remote_name, format!("Remote not found: {}", e)))?;

        remote_obj.fetch(&[] as &[&str], None, None).map_err(|e| {
            Git2DBError::network(format!("Failed to fetch from {}: {}", remote_name, e))
        })?;

        Ok(())
    }

    /// Update to latest from tracking remote
    pub async fn update(&self) -> Git2DBResult<()> {
        self.fetch(None).await?;
        // TODO: Merge or rebase based on tracking branch
        Ok(())
    }

    /// Push to a remote
    ///
    /// Similar to `git push <remote> <refspec>`
    ///
    /// Accepts strings, GitRef enums, or direct Oids for the refspec.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Push current branch to origin
    /// repo.push(Some("origin"), "main").await?;
    ///
    /// // Push using GitRef
    /// repo.push(Some("origin"), git2db::GitRef::Branch("develop".into())).await?;
    ///
    /// // Push specific commit (type-safe)
    /// let oid = git2::Oid::from_str("abc123...")?;
    /// repo.push(Some("origin"), oid).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn push(&self, remote: Option<&str>, refspec: impl IntoGitRef) -> Git2DBResult<()> {
        let repo = self.open_repo()?;
        let remote_name = remote.unwrap_or("origin");
        let git_ref = refspec.into_git_ref();

        let mut remote_obj = repo
            .find_remote(remote_name)
            .map_err(|e| Git2DBError::reference(remote_name, format!("Remote not found: {}", e)))?;

        // Build refspec for push
        let refspec_str = match &git_ref {
            GitRef::DefaultBranch => {
                // Push current HEAD branch
                let head = repo.head().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
                })?;
                head.name().unwrap_or("HEAD").to_string()
            }
            GitRef::Branch(name) => format!("refs/heads/{}:refs/heads/{}", name, name),
            GitRef::Tag(name) => format!("refs/tags/{}:refs/tags/{}", name, name),
            GitRef::Commit(oid) => format!("{}:refs/heads/main", oid), // Push commit to main by default
            GitRef::Revspec(spec) => format!("{}:refs/heads/{}", spec, spec),
        };

        remote_obj
            .push(&[refspec_str.as_str()], None)
            .map_err(|e| {
                Git2DBError::network(format!("Failed to push to {}: {}", remote_name, e))
            })?;

        Ok(())
    }

    /// Pull from a remote
    ///
    /// Similar to `git pull <remote> <refspec>`
    ///
    /// Accepts strings, GitRef enums, or direct Oids for the refspec.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Pull main branch from origin
    /// repo.pull(Some("origin"), "main").await?;
    ///
    /// // Pull using GitRef
    /// repo.pull(Some("origin"), git2db::GitRef::Branch("develop".into())).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn pull(&self, remote: Option<&str>, refspec: impl IntoGitRef) -> Git2DBResult<()> {
        let repo = self.open_repo()?;
        let remote_name = remote.unwrap_or("origin");
        let git_ref = refspec.into_git_ref();

        // First fetch
        let mut remote_obj = repo
            .find_remote(remote_name)
            .map_err(|e| Git2DBError::reference(remote_name, format!("Remote not found: {}", e)))?;

        remote_obj.fetch(&[] as &[&str], None, None).map_err(|e| {
            Git2DBError::network(format!("Failed to fetch from {}: {}", remote_name, e))
        })?;

        // Then merge/rebase
        let fetch_head = repo.find_reference("FETCH_HEAD").map_err(|e| {
            Git2DBError::reference("FETCH_HEAD", format!("Failed to find FETCH_HEAD: {}", e))
        })?;

        let commit = fetch_head.peel_to_commit().map_err(|e| {
            Git2DBError::reference("FETCH_HEAD", format!("Failed to get commit: {}", e))
        })?;

        // Checkout the commit
        repo.checkout_tree(commit.as_object(), None)
            .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {}", e)))?;

        // Update HEAD
        if let GitRef::Branch(branch_name) = git_ref {
            let ref_name = format!("refs/heads/{}", branch_name);
            repo.set_head(&ref_name)
                .map_err(|e| Git2DBError::internal(format!("Failed to update HEAD: {}", e)))?;
        } else {
            repo.set_head_detached(commit.id())
                .map_err(|e| Git2DBError::internal(format!("Failed to update HEAD: {}", e)))?;
        }

        Ok(())
    }

    /// Get remote manager for this repository
    ///
    /// Provides git-native remote operations like `git remote`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
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
    /// # fn example(repo: &git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Add files to staging area
    /// repo.staging().add("src/main.rs")?;
    /// repo.staging().add_all()?;
    ///
    /// // Check staged files
    /// for file in repo.staging().staged_files()? {
    ///     println!("Staged: {:?}", file.path);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn staging(&self) -> StageManager<'a> {
        StageManager::new(self.registry, self.repo_id.clone())
    }

    /// Commit staged changes
    ///
    /// Similar to `git commit -m "<message>"`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Stage changes
    /// repo.staging().add("src/main.rs")?;
    ///
    /// // Commit
    /// let oid = repo.commit("Add new feature").await?;
    /// println!("Created commit: {}", oid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn commit(&self, message: &str) -> Git2DBResult<Oid> {
        // Get default signature from GitManager
        let sig = self.registry.git_manager().create_signature(None, None)?;

        self.commit_as(&sig, message).await
    }

    /// Commit staged changes with a specific author/committer
    ///
    /// Similar to `git commit -m "<message>" --author="<name> <email>"`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// use git2::Signature;
    ///
    /// // Stage changes
    /// repo.staging().add("README.md")?;
    ///
    /// // Create custom signature
    /// let sig = Signature::now("Alice", "alice@example.com")?;
    ///
    /// // Commit as specific user
    /// let oid = repo.commit_as(&sig, "Update docs").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn commit_as(&self, signature: &Signature<'_>, message: &str) -> Git2DBResult<Oid> {
        let repo = self.open_repo()?;

        // Get the current index
        let mut index = repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?;

        // Write index to tree
        let tree_oid = index
            .write_tree()
            .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {}", e)))?;

        let tree = repo
            .find_tree(tree_oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {}", e)))?;

        // Get parent commit (if any)
        let parent_commits = if let Ok(head) = repo.head() {
            vec![head.peel_to_commit().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD commit: {}", e))
            })?]
        } else {
            vec![]
        };

        let parent_refs: Vec<&git2::Commit> = parent_commits.iter().collect();

        // Create the commit
        let commit_oid = repo
            .commit(
                Some("HEAD"),
                signature,
                signature,
                message,
                &tree,
                &parent_refs,
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to create commit: {}", e)))?;

        Ok(commit_oid)
    }

    /// Amend the last commit
    ///
    /// Similar to `git commit --amend`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Stage additional changes
    /// repo.staging().add("forgotten_file.rs")?;
    ///
    /// // Amend previous commit
    /// let oid = repo.amend(Some("Updated commit message")).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn amend(&self, message: Option<&str>) -> Git2DBResult<Oid> {
        let repo = self.open_repo()?;

        // Get HEAD commit
        let head = repo
            .head()
            .map_err(|e| Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e)))?;

        let head_commit = head.peel_to_commit().map_err(|e| {
            Git2DBError::reference("HEAD", format!("Failed to get HEAD commit: {}", e))
        })?;

        // Get current index
        let mut index = repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?;

        // Write index to tree
        let tree_oid = index
            .write_tree()
            .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {}", e)))?;

        let tree = repo
            .find_tree(tree_oid)
            .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {}", e)))?;

        // Use provided message or keep original
        let commit_message =
            message.unwrap_or_else(|| head_commit.message().unwrap_or("Amended commit"));

        // Amend the commit
        let amended_oid = head_commit
            .amend(
                Some("HEAD"),
                None, // Keep original author
                None, // Keep original committer
                None, // Keep original encoding
                Some(commit_message),
                Some(&tree),
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to amend commit: {}", e)))?;

        Ok(amended_oid)
    }

    /// Merge a branch into the current HEAD
    ///
    /// Similar to `git merge <branch>`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(repo: git2db::RepositoryHandle<'_>) -> Result<(), Box<dyn std::error::Error>> {
    /// // Fast-forward merge (default)
    /// repo.merge("feature-branch", false, false).await?;
    ///
    /// // Force merge commit (--no-ff)
    /// repo.merge("feature-branch", false, true).await?;
    ///
    /// // Fast-forward only (--ff-only)
    /// repo.merge("feature-branch", true, false).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn merge(
        &self,
        branch: &str,
        ff_only: bool,
        no_ff: bool,
    ) -> Git2DBResult<Oid> {
        let repo = self.open_repo()?;

        // Resolve branch reference
        let branch_ref = repo
            .find_reference(&format!("refs/heads/{}", branch))
            .or_else(|_| repo.find_reference(&format!("refs/remotes/origin/{}", branch)))
            .or_else(|_| repo.find_reference(branch))
            .map_err(|e| {
                Git2DBError::reference(branch, format!("Branch not found: {}", e))
            })?;

        let branch_commit = branch_ref.peel_to_commit().map_err(|e| {
            Git2DBError::reference(branch, format!("Failed to resolve commit: {}", e))
        })?;

        let annotated_commit = repo.find_annotated_commit(branch_commit.id()).map_err(|e| {
            Git2DBError::internal(format!("Failed to create annotated commit: {}", e))
        })?;

        // Perform merge analysis
        let (merge_analysis, _) = repo.merge_analysis(&[&annotated_commit]).map_err(|e| {
            Git2DBError::internal(format!("Merge analysis failed: {}", e))
        })?;

        // Already up-to-date
        if merge_analysis.is_up_to_date() {
            return Ok(branch_commit.id());
        }

        // Check fast-forward constraints
        if ff_only && !merge_analysis.is_fast_forward() {
            return Err(Git2DBError::merge_conflict(
                "Cannot fast-forward - branches have diverged",
            ));
        }

        // Fast-forward merge (if possible and not --no-ff)
        if merge_analysis.is_fast_forward() && !no_ff {
            let mut head_ref = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
            })?;

            head_ref
                .set_target(branch_commit.id(), "Fast-forward merge")
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to update HEAD: {}", e))
                })?;

            repo.checkout_head(Some(
                git2::build::CheckoutBuilder::default()
                    .force()
            ))
            .map_err(|e| {
                Git2DBError::internal(format!("Checkout failed: {}", e))
            })?;

            return Ok(branch_commit.id());
        }

        // Regular merge (create merge commit)
        repo.merge(&[&annotated_commit], None, None)
            .map_err(|e| Git2DBError::internal(format!("Merge failed: {}", e)))?;

        // Check for conflicts
        if repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?
            .has_conflicts()
        {
            return Err(Git2DBError::merge_conflict(
                "Merge conflicts detected - please resolve manually",
            ));
        }

        // Create merge commit
        let sig = self.registry.git_manager().create_signature(None, None)?;

        let tree_id = repo
            .index()
            .map_err(|e| Git2DBError::internal(format!("Failed to get index: {}", e)))?
            .write_tree()
            .map_err(|e| Git2DBError::internal(format!("Failed to write tree: {}", e)))?;

        let tree = repo
            .find_tree(tree_id)
            .map_err(|e| Git2DBError::internal(format!("Failed to find tree: {}", e)))?;

        let parent = repo
            .head()
            .map_err(|e| Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e)))?
            .peel_to_commit()
            .map_err(|e| {
                Git2DBError::internal(format!("Failed to resolve HEAD commit: {}", e))
            })?;

        let merge_oid = repo
            .commit(
                Some("HEAD"),
                &sig,
                &sig,
                &format!("Merge branch '{}'", branch),
                &tree,
                &[&parent, &branch_commit],
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to create merge commit: {}", e)))?;

        repo.cleanup_state()
            .map_err(|e| Git2DBError::internal(format!("Failed to cleanup merge state: {}", e)))?;

        Ok(merge_oid)
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
            .map_err(|e| Git2DBError::internal(format!("Failed to list references: {}", e)))?
        {
            let r = reference
                .map_err(|e| Git2DBError::internal(format!("Failed to read reference: {}", e)))?;

            if let Some(name) = r.shorthand() {
                if let Some(oid) = r.target() {
                    refs.push((name.to_string(), oid));
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
                    return Ok(branch_name.to_string());
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
                return Ok(default_name.to_string());
            }
        }

        // Fallback: get the first branch
        let mut branches = repo
            .branches(Some(git2::BranchType::Local))
            .map_err(|e| Git2DBError::internal(format!("Failed to list branches: {}", e)))?;

        if let Some(Ok((branch, _))) = branches.next() {
            if let Some(name) = branch
                .name()
                .map_err(|e| Git2DBError::internal(format!("Failed to get branch name: {}", e)))?
            {
                return Ok(name.to_string());
            }
        }

        // Final fallback
        Ok("main".to_string())
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
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD: {}", e))
                })?;
                head.peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to peel to commit: {}", e))
                })
            }
            GitRef::Branch(branch_name) => {
                let branch = repo
                    .find_branch(branch_name, git2::BranchType::Local)
                    .or_else(|_| repo.find_branch(branch_name, git2::BranchType::Remote))
                    .map_err(|e| {
                        Git2DBError::reference(branch_name, format!("Branch not found: {}", e))
                    })?;
                branch.get().peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference(branch_name, format!("Failed to peel to commit: {}", e))
                })
            }
            GitRef::Tag(tag_name) => {
                let reference = repo
                    .find_reference(&format!("refs/tags/{}", tag_name))
                    .map_err(|e| {
                        Git2DBError::reference(tag_name, format!("Tag not found: {}", e))
                    })?;
                reference.peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference(tag_name, format!("Failed to peel to commit: {}", e))
                })
            }
            GitRef::Commit(oid) => Ok(*oid),
            GitRef::Revspec(spec) => {
                let obj = repo.revparse_single(spec).map_err(|e| {
                    Git2DBError::reference(spec, format!("Failed to resolve revspec: {}", e))
                })?;
                obj.peel_to_commit().map(|c| c.id()).map_err(|e| {
                    Git2DBError::reference(spec, format!("Failed to peel to commit: {}", e))
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
            ref_name.to_string()
        } else if repo.find_branch(ref_name, git2::BranchType::Local).is_ok() {
            format!("refs/heads/{}", ref_name)
        } else if repo
            .find_reference(&format!("refs/tags/{}", ref_name))
            .is_ok()
        {
            format!("refs/tags/{}", ref_name)
        } else {
            ref_name.to_string()
        };

        // Find and extract data before returning
        let result = match repo.find_reference(&ref_path) {
            Ok(reference) => {
                let name = reference.shorthand().unwrap_or(ref_name).to_string();
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
            Git2DBError::reference(spec, format!("Failed to resolve revspec: {}", e))
        })?;
        Ok(obj.id())
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
