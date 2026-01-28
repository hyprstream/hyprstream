//! Branch management for tracked repositories
//!
//! Provides git-native branch operations similar to `git branch`

use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::{GitRef, IntoGitRef};
use crate::registry::{Git2DB, RepoId};
use crate::repo_accessor::RepositoryAccessor;
use git2::{BranchType, Oid, Repository};

/// Manager for repository branches
///
/// Provides operations similar to `git branch` commands.
///
/// # Examples
///
/// ```rust,ignore
/// let repo = registry.repo(repo_id)?;
/// let branch_mgr = repo.branch();
///
/// // List branches
/// for branch in branch_mgr.list().await? {
///     let marker = if branch.is_head { "*" } else { " " };
///     println!("{} {}", marker, branch.name);
/// }
///
/// // Get current branch
/// if let Some(current) = branch_mgr.current().await? {
///     println!("On branch: {}", current.name);
/// }
///
/// // Create new branch
/// branch_mgr.create("feature", Some("main")).await?;
///
/// // Checkout branch
/// branch_mgr.checkout("feature").await?;
///
/// // Remove branch
/// branch_mgr.remove("old-feature", false).await?;
/// ```
pub struct BranchManager<'a> {
    registry: &'a Git2DB,
    repo_id: RepoId,
}

impl<'a> RepositoryAccessor for BranchManager<'a> {
    fn registry(&self) -> &Git2DB {
        self.registry
    }

    fn repo_id(&self) -> &RepoId {
        &self.repo_id
    }
}

impl<'a> BranchManager<'a> {
    /// Create a new branch manager
    pub(crate) fn new(registry: &'a Git2DB, repo_id: RepoId) -> Self {
        Self { registry, repo_id }
    }

    /// List all branches (local and remote)
    ///
    /// Similar to `git branch -a`
    pub async fn list(&self) -> Git2DBResult<Vec<Branch>> {
        let path = self.repo_path()?;

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {e}"))
            })?;

            let mut branches = Vec::new();

            // Get HEAD for comparison
            let head_ref = repo.head().ok();
            let head_name = head_ref
                .as_ref()
                .and_then(|h| h.shorthand())
                .map(std::borrow::ToOwned::to_owned);

            // List local branches
            for (branch, _) in (repo.branches(Some(BranchType::Local)).map_err(|e| {
                Git2DBError::internal(format!("Failed to list local branches: {e}"))
            })?).flatten() {
            if let Some(name) = branch.name().ok().flatten() {
                let is_head = head_name.as_deref() == Some(name);
                let oid = branch.get().target();

                branches.push(Branch {
                    name: name.to_owned(),
                    branch_type: BranchKind::Local,
                    is_head,
                    oid,
                    tracking: branch
                        .upstream()
                        .ok()
                        .and_then(|u| u.name().ok().flatten().map(std::borrow::ToOwned::to_owned)),
                });
            }
            }

            // List remote branches
            for (branch, _) in (repo.branches(Some(BranchType::Remote)).map_err(|e| {
                Git2DBError::internal(format!("Failed to list remote branches: {e}"))
            })?).flatten() {
            if let Some(name) = branch.name().ok().flatten() {
                let oid = branch.get().target();

                branches.push(Branch {
                    name: name.to_owned(),
                    branch_type: BranchKind::Remote,
                    is_head: false,
                    oid,
                    tracking: None,
                });
            }
            }

            Ok(branches)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Get the current branch (HEAD)
    ///
    /// Returns None if HEAD is detached
    pub async fn current(&self) -> Git2DBResult<Option<Branch>> {
        let path = self.repo_path()?;

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {e}"))
            })?;

            let head = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {e}"))
            })?;

            // Check if HEAD is a branch
            if !head.is_branch() {
                return Ok(None);
            }

            let name = head
                .shorthand()
                .ok_or_else(|| Git2DBError::reference("HEAD", "HEAD has no name"))?.to_owned();

            let oid = head.target();

            // Try to find the branch to get tracking info
            let tracking = if let Ok(branch) = repo.find_branch(&name, BranchType::Local) {
                branch
                    .upstream()
                    .ok()
                    .and_then(|u| u.name().ok().flatten().map(std::borrow::ToOwned::to_owned))
            } else {
                None
            };

            Ok(Some(Branch {
                name,
                branch_type: BranchKind::Local,
                is_head: true,
                oid,
                tracking,
            }))
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Create a new branch
    ///
    /// Similar to `git branch <name> [from]`
    ///
    /// Accepts strings, GitRef enums, or direct Oids for the base reference.
    /// If `from` is None, creates from HEAD.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Create from HEAD
    /// repo.branch().create("feature", None).await?;
    ///
    /// // Create from string reference
    /// repo.branch().create("feature", Some("main")).await?;
    ///
    /// // Create from explicit GitRef
    /// repo.branch().create("hotfix", Some(git2db::GitRef::Tag("v1.0.0".into()))).await?;
    ///
    /// // Create from specific commit (type-safe)
    /// let oid = git2::Oid::from_str("abc123...")?;
    /// repo.branch().create("from-commit", Some(oid)).await?;
    /// ```
    pub async fn create<R: IntoGitRef>(&self, name: &str, from: Option<R>) -> Git2DBResult<()> {
        let path = self.repo_path()?;
        let name = name.to_owned();
        let from_ref = from.map(super::references::IntoGitRef::into_git_ref);

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {e}"))
            })?;

            // Resolve the starting point
            let commit = if let Some(git_ref) = from_ref {
                let ref_str = git_ref.display_name();

                // Resolve GitRef to commit
                let oid = match git_ref {
                    GitRef::DefaultBranch => {
                        let head = repo.head().map_err(|e| {
                            Git2DBError::reference("HEAD", format!("Failed to get HEAD: {e}"))
                        })?;
                        head.peel_to_commit()
                            .map_err(|e| {
                                Git2DBError::reference(
                                    "HEAD",
                                    format!("HEAD is not a commit: {e}"),
                                )
                            })?
                            .id()
                    }
                    GitRef::Commit(oid) => oid,
                    GitRef::Branch(branch_name)
                    | GitRef::Tag(branch_name)
                    | GitRef::Revspec(branch_name) => {
                        let obj = repo.revparse_single(&branch_name).map_err(|e| {
                            Git2DBError::reference(
                                &ref_str,
                                format!("Failed to resolve reference: {e}"),
                            )
                        })?;
                        obj.peel_to_commit()
                            .map_err(|e| {
                                Git2DBError::reference(&ref_str, format!("Not a commit: {e}"))
                            })?
                            .id()
                    }
                };

                repo.find_commit(oid).map_err(|e| {
                    Git2DBError::reference(&ref_str, format!("Failed to find commit: {e}"))
                })?
            } else {
                // Use HEAD
                let head = repo.head().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD: {e}"))
                })?;
                head.peel_to_commit().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("HEAD is not a commit: {e}"))
                })?
            };

            // Create the branch
            repo.branch(&name, &commit, false).map_err(|e| {
                Git2DBError::reference(&name, format!("Failed to create branch: {e}"))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Checkout a branch
    ///
    /// Similar to `git checkout <name>`
    pub async fn checkout(&self, name: &str) -> Git2DBResult<()> {
        let path = self.repo_path()?;
        let name = name.to_owned();

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {e}"))
            })?;

            // Find the branch
            let branch = repo
                .find_branch(&name, BranchType::Local)
                .map_err(|_| Git2DBError::reference(&name, "Branch not found"))?;

            let reference = branch.into_reference();
            let commit = reference.peel_to_commit().map_err(|e| {
                Git2DBError::reference(&name, format!("Failed to get commit: {e}"))
            })?;

            // Checkout the branch
            repo.checkout_tree(commit.as_object(), None)
                .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {e}")))?;

            // Set HEAD to the branch
            repo.set_head(&format!("refs/heads/{name}"))
                .map_err(|e| Git2DBError::internal(format!("Failed to update HEAD: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Remove a branch
    ///
    /// Similar to `git branch -d <name>` (or `-D` if force=true)
    pub async fn remove(&self, name: &str, force: bool) -> Git2DBResult<()> {
        // Check if this is the current branch
        if let Ok(Some(current)) = self.current().await {
            if current.name == name {
                return Err(Git2DBError::configuration(format!(
                    "Cannot delete current branch '{name}'"
                )));
            }
        }

        let path = self.repo_path()?;
        let name = name.to_owned();

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {e}"))
            })?;

            // Find and delete the branch
            let mut branch = repo
                .find_branch(&name, BranchType::Local)
                .map_err(|_| Git2DBError::reference(&name, "Branch not found"))?;

            if !force {
                // Check if branch is fully merged
                let branch_oid = branch
                    .get()
                    .target()
                    .ok_or_else(|| Git2DBError::reference(&name, "Branch has no target"))?;

                // Check if reachable from HEAD
                let head = repo.head().map_err(|e| {
                    Git2DBError::reference("HEAD", format!("Failed to get HEAD: {e}"))
                })?;
                let head_oid = head
                    .target()
                    .ok_or_else(|| Git2DBError::reference("HEAD", "HEAD has no target"))?;

                if !repo
                    .graph_descendant_of(head_oid, branch_oid)
                    .unwrap_or(false)
                {
                    return Err(Git2DBError::configuration(format!(
                        "Branch '{name}' is not fully merged. Use force=true to delete anyway."
                    )));
                }
            }

            branch.delete().map_err(|e| {
                Git2DBError::reference(&name, format!("Failed to delete branch: {e}"))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Rename current branch or a specific branch
    ///
    /// Similar to `git branch -m [old] <new>`
    pub async fn rename(&self, old_name: Option<&str>, new_name: &str) -> Git2DBResult<()> {
        let branch_name = if let Some(name) = old_name {
            name.to_owned()
        } else {
            // Get current branch name
            self.current()
                .await?
                .ok_or_else(|| Git2DBError::reference("HEAD", "Not on a branch (detached HEAD)"))?
                .name
        };

        let path = self.repo_path()?;
        let new_name = new_name.to_owned();

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {e}"))
            })?;

            // Find and rename the branch
            let mut branch = repo
                .find_branch(&branch_name, BranchType::Local)
                .map_err(|_| Git2DBError::reference(&branch_name, "Branch not found"))?;

            branch.rename(&new_name, false).map_err(|e| {
                Git2DBError::reference(&new_name, format!("Failed to rename branch: {e}"))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }
}

/// Branch information
#[derive(Debug, Clone)]
pub struct Branch {
    pub name: String,
    pub branch_type: BranchKind,
    pub is_head: bool,
    pub oid: Option<Oid>,
    pub tracking: Option<String>,
}

impl Branch {
    /// Check if this is a local branch
    pub fn is_local(&self) -> bool {
        matches!(self.branch_type, BranchKind::Local)
    }

    /// Check if this is a remote branch
    pub fn is_remote(&self) -> bool {
        matches!(self.branch_type, BranchKind::Remote)
    }

    /// Get short commit hash (first 7 chars)
    pub fn short_oid(&self) -> Option<String> {
        self.oid.map(|oid| format!("{oid:.7}"))
    }
}

/// Branch type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchKind {
    Local,
    Remote,
}
