//! Architectural patterns for safe git2 repository management
//!
//! This module provides idiomatic Rust patterns for handling git2 objects
//! with proper lifetime management and zero-cost abstractions.
//!
//! ## Design Patterns Implemented:
//!
//! 1. **Repository Handle Pattern**: Wraps git2::Repository with Arc for safe sharing
//! 2. **Visitor Pattern**: Execute operations on git2 objects within their lifetime scope
//! 3. **Factory Pattern**: Create git2 objects with proper ownership transfer
//! 4. **RAII Pattern**: Ensure repositories are properly managed through their lifecycle
//! 5. **Type State Pattern**: Compile-time guarantees for repository states

use anyhow::{Result, Context, anyhow};
use git2::{Repository, Submodule, Reference, Commit, Oid};
use std::path::{Path, PathBuf};
use std::marker::PhantomData;

/// Type-safe wrapper for git2::Repository that ensures proper lifetime management
///
/// This pattern addresses the fundamental issue with git2 object lifetimes by
/// ensuring that all operations on git2 objects happen within a controlled scope.
pub struct RepositoryHandle {
    /// The actual repository
    repo: Repository,
    /// Path to the repository for re-opening if needed
    path: PathBuf,
}

impl Clone for RepositoryHandle {
    fn clone(&self) -> Self {
        // Since git2::Repository doesn't implement Clone, we need to re-open
        // TODO: This could be optimized with proper repository sharing
        Self::open(&self.path).expect("Failed to clone repository handle")
    }
}

impl RepositoryHandle {
    /// Create a new repository handle
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().canonicalize()
            .with_context(|| format!("Failed to canonicalize path: {:?}", path.as_ref()))?;

        let repo = Repository::open(&path)
            .with_context(|| format!("Failed to open repository at {:?}", path))?;

        Ok(Self { repo, path })
    }

    /// Create from an existing repository
    pub fn from_repository(repo: Repository) -> Result<Self> {
        let path = repo.path().to_path_buf();
        Ok(Self { repo, path })
    }

    /// Get the repository path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Execute a closure with the repository
    ///
    /// This is the primary pattern for safely working with git2 objects.
    /// The closure receives a reference to the repository and must return
    /// owned data (not references).
    pub fn with_repository<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Repository) -> Result<R>,
        R: 'static,  // Ensures the result doesn't contain repository references
    {
        f(&self.repo)
    }

    /// Execute a closure with a submodule
    ///
    /// This pattern ensures submodule operations happen within the repository's lifetime
    pub fn with_submodule<F, R>(&self, path: &str, f: F) -> Result<Option<R>>
    where
        F: FnOnce(&Submodule) -> Result<R>,
        R: 'static,
    {
        match self.repo.find_submodule(path) {
            Ok(submodule) => Ok(Some(f(&submodule)?)),
            Err(e) if e.code() == git2::ErrorCode::NotFound => Ok(None),
            Err(e) => Err(anyhow!("Failed to access submodule {}: {}", path, e)),
        }
    }

    /// Open a submodule's repository as a new handle
    ///
    /// This creates a new, independent repository handle for the submodule
    pub fn open_submodule(&self, path: &str) -> Result<Option<Self>> {
        self.with_submodule(path, |submodule| {
            let submodule_path = self.path.join(submodule.path());
            Self::open(submodule_path)
        })
    }

    /// Get submodule information without lifetime issues
    pub fn get_submodule_info(&self, path: &str) -> Result<Option<SubmoduleInfo>> {
        self.with_submodule(path, |submodule| {
            // Try to get index_id, but provide fallback if not available
            let index_id = submodule.index_id().or_else(|| {
                // Fallback: try to get HEAD commit from submodule repository
                if let Ok(submodule_repo) = submodule.open() {
                    if let Ok(head) = submodule_repo.head() {
                        if let Ok(commit) = head.peel_to_commit() {
                            return Some(commit.id());
                        }
                    }
                }
                None
            });

            Ok(SubmoduleInfo {
                name: submodule.name().map(String::from),
                path: submodule.path().to_path_buf(),
                url: submodule.url().map(String::from),
                index_id,
                head_id: submodule.head_id(),
            })
        })
    }

    /// Find a reference and extract its information
    pub fn get_reference_info(&self, name: &str) -> Result<Option<ReferenceInfo>> {
        match self.repo.find_reference(name) {
            Ok(reference) => Ok(Some(ReferenceInfo::from_reference(&reference)?)),
            Err(e) if e.code() == git2::ErrorCode::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get commit information without lifetime issues
    pub fn get_commit_info(&self, oid: Oid) -> Result<CommitInfo> {
        let commit = self.repo.find_commit(oid)?;
        Ok(CommitInfo::from_commit(&commit))
    }
}

/// Owned submodule information that doesn't depend on repository lifetime
#[derive(Debug, Clone)]
pub struct SubmoduleInfo {
    pub name: Option<String>,
    pub path: PathBuf,
    pub url: Option<String>,
    pub index_id: Option<Oid>,
    pub head_id: Option<Oid>,
}

/// Owned reference information
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    pub name: String,
    pub shorthand: String,
    pub target: Option<Oid>,
    pub symbolic_target: Option<String>,
    pub is_branch: bool,
    pub is_remote: bool,
    pub is_tag: bool,
}

impl ReferenceInfo {
    fn from_reference(reference: &Reference) -> Result<Self> {
        Ok(Self {
            name: reference.name()
                .ok_or_else(|| anyhow!("Invalid reference name"))?
                .to_string(),
            shorthand: reference.shorthand()
                .ok_or_else(|| anyhow!("Invalid reference shorthand"))?
                .to_string(),
            target: reference.target(),
            symbolic_target: reference.symbolic_target().map(String::from),
            is_branch: reference.is_branch(),
            is_remote: reference.is_remote(),
            is_tag: reference.is_tag(),
        })
    }
}

/// Owned commit information
#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub id: Oid,
    pub summary: Option<String>,
    pub message: Option<String>,
    pub author_name: String,
    pub author_email: String,
    pub time: i64,
    pub parent_ids: Vec<Oid>,
}

impl CommitInfo {
    fn from_commit(commit: &Commit) -> Self {
        Self {
            id: commit.id(),
            summary: commit.summary().map(String::from),
            message: commit.message().map(String::from),
            author_name: commit.author().name().unwrap_or("Unknown").to_string(),
            author_email: commit.author().email().unwrap_or("unknown@example.com").to_string(),
            time: commit.time().seconds(),
            parent_ids: commit.parent_ids().collect(),
        }
    }
}

/// Type-state pattern for repository operations
///
/// This pattern provides compile-time guarantees about repository states
pub struct RepositoryOperation<State> {
    handle: RepositoryHandle,
    _state: PhantomData<State>,
}

/// Repository states
pub struct Unmodified;
pub struct Modified;
pub struct Committed;

impl RepositoryOperation<Unmodified> {
    pub fn new(handle: RepositoryHandle) -> Self {
        Self {
            handle,
            _state: PhantomData,
        }
    }

    /// Perform modifications, transitioning to Modified state
    pub fn modify<F>(self, f: F) -> Result<RepositoryOperation<Modified>>
    where
        F: FnOnce(&RepositoryHandle) -> Result<()>,
    {
        f(&self.handle)?;
        Ok(RepositoryOperation {
            handle: self.handle,
            _state: PhantomData,
        })
    }
}

impl RepositoryOperation<Modified> {
    /// Commit changes, transitioning to Committed state
    pub fn commit(self, message: &str) -> Result<RepositoryOperation<Committed>> {
        self.handle.with_repository(|repo| {
            let sig = git2::Signature::now("hyprstream", "hyprstream@local")?;
            let mut index = repo.index()?;
            let tree_id = index.write_tree()?;
            let tree = repo.find_tree(tree_id)?;

            let parent = repo.head()?.peel_to_commit()?;
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                message,
                &tree,
                &[&parent],
            )?;
            Ok(())
        })?;

        Ok(RepositoryOperation {
            handle: self.handle,
            _state: PhantomData,
        })
    }

    /// Discard changes and return to unmodified state
    pub fn discard(self) -> Result<RepositoryOperation<Unmodified>> {
        self.handle.with_repository(|repo| {
            repo.reset(
                &repo.head()?.peel_to_commit()?.into_object(),
                git2::ResetType::Hard,
                None,
            )?;
            Ok(())
        })?;

        Ok(RepositoryOperation {
            handle: self.handle,
            _state: PhantomData,
        })
    }
}

/// Builder pattern for complex repository operations
pub struct RepositoryOperationBuilder {
    handle: RepositoryHandle,
    operations: Vec<Box<dyn FnOnce(&Repository) -> Result<()>>>,
}

impl RepositoryOperationBuilder {
    pub fn new(handle: RepositoryHandle) -> Self {
        Self {
            handle,
            operations: Vec::new(),
        }
    }

    /// Add an operation to the builder
    pub fn add_operation<F>(mut self, operation: F) -> Self
    where
        F: FnOnce(&Repository) -> Result<()> + 'static,
    {
        self.operations.push(Box::new(operation));
        self
    }

    /// Execute all operations in sequence
    pub fn execute(self) -> Result<()> {
        self.handle.with_repository(|repo| {
            for operation in self.operations {
                operation(repo)?;
            }
            Ok(())
        })
    }

    /// Execute all operations and commit if successful
    pub fn execute_and_commit(self, message: &str) -> Result<Oid> {
        self.handle.with_repository(|repo| {
            // Execute all operations
            for operation in self.operations {
                operation(repo)?;
            }

            // Commit the changes
            let sig = git2::Signature::now("hyprstream", "hyprstream@local")?;
            let mut index = repo.index()?;
            let tree_id = index.write_tree()?;
            let tree = repo.find_tree(tree_id)?;

            let parent = repo.head()?.peel_to_commit()?;
            let oid = repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                message,
                &tree,
                &[&parent],
            )?;

            Ok(oid)
        })
    }
}

/// Cache-aware repository factory
///
/// This pattern integrates with GitManager's repository cache while maintaining proper lifetimes
pub struct CachedRepositoryFactory;

impl CachedRepositoryFactory {
    /// Get or create a repository handle using GitManager's cache
    pub fn get_or_create<P: AsRef<Path>>(path: P) -> Result<RepositoryHandle> {
        let cache = git2db::GitManager::global().get_repository(path)
            .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))?;
        let repo = cache.open()
            .map_err(|e| anyhow::anyhow!("Failed to open repository: {}", e))?;
        RepositoryHandle::from_repository(repo)
    }

    /// Create a repository handle for a submodule
    pub fn create_for_submodule(
        parent_handle: &RepositoryHandle,
        submodule_path: &str,
    ) -> Result<Option<RepositoryHandle>> {
        parent_handle.with_submodule(submodule_path, |submodule| {
            let full_path = parent_handle.path().join(submodule.path());
            Self::get_or_create(full_path)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_repository_handle() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path().join("test_repo");
        Repository::init(&repo_path).unwrap();

        let handle = RepositoryHandle::open(&repo_path).unwrap();

        // Test with_repository pattern
        let head_exists = handle.with_repository(|repo| {
            Ok(repo.head().is_ok())
        }).unwrap();

        assert!(!head_exists); // New repo has no HEAD
    }

    #[test]
    fn test_type_state_pattern() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path().join("test_repo");
        Repository::init(&repo_path).unwrap();

        // Create initial commit
        let repo = Repository::open(&repo_path).unwrap();
        let sig = git2::Signature::now("test", "test@example.com").unwrap();
        let tree_id = {
            let mut index = repo.index().unwrap();
            index.write_tree().unwrap()
        };
        let tree = repo.find_tree(tree_id).unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Initial", &tree, &[]).unwrap();

        let handle = RepositoryHandle::open(&repo_path).unwrap();
        let operation = RepositoryOperation::<Unmodified>::new(handle);

        // This demonstrates compile-time state transitions
        let _modified = operation.modify(|_handle| Ok(())).unwrap();
        // Can only commit from Modified state
        // let _committed = modified.commit("Test commit").unwrap();
    }
}