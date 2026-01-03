//! Reference resolution and management
//!
//! Consolidated reference handling patterns from the original codebase

use crate::errors::{Git2DBError, Git2DBResult};
use crate::manager::GitManager;
use git2::{Oid, Repository};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, trace};

/// Git reference types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GitRef {
    /// Default branch (HEAD)
    DefaultBranch,
    /// Specific branch
    Branch(String),
    /// Specific tag
    Tag(String),
    /// Specific commit
    #[serde(serialize_with = "serialize_oid", deserialize_with = "deserialize_oid")]
    Commit(Oid),
    /// Arbitrary revspec
    Revspec(String),
}

fn serialize_oid<S>(oid: &Oid, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&oid.to_string())
}

fn deserialize_oid<'de, D>(deserializer: D) -> Result<Oid, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Oid::from_str(&s).map_err(serde::de::Error::custom)
}

impl GitRef {
    /// Parse a reference string
    pub fn parse(ref_str: &str) -> Git2DBResult<Self> {
        if ref_str.is_empty() || ref_str == "HEAD" {
            return Ok(GitRef::DefaultBranch);
        }

        // Try to parse as commit hash
        if ref_str.len() >= 7 && ref_str.chars().all(|c| c.is_ascii_hexdigit()) {
            if ref_str.len() == 40 {
                // Full SHA
                let oid = Oid::from_str(ref_str)
                    .map_err(|_| Git2DBError::reference(ref_str, "Invalid commit hash"))?;
                return Ok(GitRef::Commit(oid));
            } else if ref_str.len() >= 7 {
                // Short SHA - treat as revspec for now
                return Ok(GitRef::Revspec(ref_str.to_string()));
            }
        }

        // Check for tag prefix
        if ref_str.starts_with("refs/tags/") {
            let tag_name = ref_str.strip_prefix("refs/tags/").unwrap();
            return Ok(GitRef::Tag(tag_name.to_string()));
        }

        if ref_str.starts_with("tags/") {
            let tag_name = ref_str.strip_prefix("tags/").unwrap();
            return Ok(GitRef::Tag(tag_name.to_string()));
        }

        // Check for branch prefix
        if ref_str.starts_with("refs/heads/") {
            let branch_name = ref_str.strip_prefix("refs/heads/").unwrap();
            return Ok(GitRef::Branch(branch_name.to_string()));
        }

        if ref_str.starts_with("origin/") {
            let branch_name = ref_str.strip_prefix("origin/").unwrap();
            return Ok(GitRef::Branch(branch_name.to_string()));
        }

        // Default to branch name or revspec
        if git2::Reference::is_valid_name(&format!("refs/heads/{}", ref_str)) {
            Ok(GitRef::Branch(ref_str.to_string()))
        } else {
            Ok(GitRef::Revspec(ref_str.to_string()))
        }
    }

    /// Convert to reference string
    pub fn to_ref_string(&self) -> Option<String> {
        match self {
            GitRef::DefaultBranch => None,
            GitRef::Branch(name) => Some(format!("refs/heads/{}", name)),
            GitRef::Tag(name) => Some(format!("refs/tags/{}", name)),
            GitRef::Commit(oid) => Some(oid.to_string()),
            GitRef::Revspec(spec) => Some(spec.clone()),
        }
    }

    /// Get display name for the reference
    pub fn display_name(&self) -> String {
        match self {
            GitRef::DefaultBranch => "HEAD".to_string(),
            GitRef::Branch(name) => name.clone(),
            GitRef::Tag(name) => format!("tags/{}", name),
            GitRef::Commit(oid) => format!("{:.8}", oid),
            GitRef::Revspec(spec) => spec.clone(),
        }
    }
}

impl std::fmt::Display for GitRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Trait for types that can be converted into a GitRef
///
/// This allows for flexible API design where methods can accept either:
/// - String references (parsed at runtime)
/// - Explicit GitRef enums (clear intent)
/// - Direct Oids (type-safe, no ambiguity)
///
/// # Examples
///
/// ```rust,ignore
/// # use git2db::{Git2DB, GitRef};
/// # use git2::Oid;
/// # async fn example(worktree: &mut git2db::WorktreeHandle) -> Result<(), Box<dyn std::error::Error>> {
/// // String reference (ergonomic)
/// worktree.checkout("main").await?;
///
/// // Explicit GitRef (clear intent)
/// worktree.checkout(GitRef::Branch("develop".into())).await?;
///
/// // Direct Oid (type-safe)
/// let oid = Oid::from_str("abc123...")?;
/// worktree.checkout(oid).await?;
/// # Ok(())
/// # }
/// ```
pub trait IntoGitRef {
    /// Convert self into a GitRef
    fn into_git_ref(self) -> GitRef;
}

impl IntoGitRef for &str {
    fn into_git_ref(self) -> GitRef {
        GitRef::parse(self).unwrap_or_else(|_| GitRef::Revspec(self.to_string()))
    }
}

impl IntoGitRef for String {
    fn into_git_ref(self) -> GitRef {
        GitRef::parse(&self).unwrap_or(GitRef::Revspec(self))
    }
}

impl IntoGitRef for GitRef {
    fn into_git_ref(self) -> GitRef {
        self
    }
}

impl IntoGitRef for Oid {
    fn into_git_ref(self) -> GitRef {
        GitRef::Commit(self)
    }
}

/// Reference information
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    /// Reference name
    pub name: String,
    /// Target OID
    pub target: Oid,
    /// Reference type
    pub ref_type: git2::ReferenceType,
    /// Whether this is the current HEAD
    pub is_head: bool,
}

/// Reference resolver with caching
pub struct ReferenceResolver {
    cache: parking_lot::RwLock<HashMap<String, CachedReference>>,
    cache_ttl: std::time::Duration,
}

/// Cached reference information
#[derive(Debug, Clone)]
struct CachedReference {
    oid: Oid,
    cached_at: std::time::Instant,
}

impl Default for ReferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceResolver {
    /// Create a new reference resolver
    ///
    /// Uses the global GitManager singleton for repository access.
    pub fn new() -> Self {
        Self {
            cache: parking_lot::RwLock::new(HashMap::new()),
            cache_ttl: std::time::Duration::from_secs(60), // 1 minute cache
        }
    }

    /// Create with custom cache TTL
    pub fn with_cache_ttl(cache_ttl: std::time::Duration) -> Self {
        Self {
            cache: parking_lot::RwLock::new(HashMap::new()),
            cache_ttl,
        }
    }

    /// Resolve a git reference to an OID
    pub async fn resolve<P: AsRef<Path>>(
        &self,
        repo_path: P,
        git_ref: &GitRef,
    ) -> Git2DBResult<Oid> {
        let repo_path = repo_path.as_ref().to_path_buf();
        let cache_key = format!("{}:{}", repo_path.display(), git_ref.display_name());

        // Check cache first (sync - just HashMap access)
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(&cache_key) {
                if cached.cached_at.elapsed() < self.cache_ttl {
                    trace!("Reference cache hit for {}", cache_key);
                    return Ok(cached.oid);
                }
            }
        }

        // Clone git_ref for move into closure
        let git_ref = git_ref.clone();

        // Resolve in blocking task
        let oid = tokio::task::spawn_blocking(move || -> Git2DBResult<Oid> {
            let repo_cache = GitManager::global().get_repository(&repo_path)?;
            let repo = repo_cache.open()?;
            resolve_in_repo(&repo, &git_ref)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))??;

        // Update cache (sync - just HashMap write)
        {
            let mut cache = self.cache.write();
            cache.insert(
                cache_key,
                CachedReference {
                    oid,
                    cached_at: std::time::Instant::now(),
                },
            );
        }

        Ok(oid)
    }

    /// Resolve reference information (name and target)
    pub async fn resolve_reference<P: AsRef<Path>>(
        &self,
        repo_path: P,
        git_ref: &GitRef,
    ) -> Git2DBResult<Option<ReferenceInfo>> {
        let repo_path = repo_path.as_ref().to_path_buf();
        let git_ref = git_ref.clone();

        tokio::task::spawn_blocking(move || -> Git2DBResult<Option<ReferenceInfo>> {
            let repo_cache = GitManager::global().get_repository(&repo_path)?;
            let repo = repo_cache.open()?;

            match &git_ref {
                GitRef::DefaultBranch => match repo.head() {
                    Ok(head_ref) => {
                        let name = head_ref.shorthand().unwrap_or("HEAD").to_string();
                        let target = head_ref.target().unwrap_or_else(Oid::zero);
                        let ref_type = head_ref.kind().unwrap_or(git2::ReferenceType::Direct);
                        Ok(Some(ReferenceInfo {
                            name,
                            target,
                            ref_type,
                            is_head: true,
                        }))
                    }
                    Err(_) => Ok(None),
                },

                GitRef::Branch(branch_name) => {
                    let ref_name = format!("refs/heads/{}", branch_name);
                    match repo.find_reference(&ref_name) {
                        Ok(reference) => {
                            let target = reference.target().unwrap_or_else(Oid::zero);
                            let ref_type = reference.kind().unwrap_or(git2::ReferenceType::Direct);

                            // Check if this is the current HEAD
                            let is_head = repo
                                .head()
                                .map(|head| head.name() == Some(&ref_name))
                                .unwrap_or(false);

                            Ok(Some(ReferenceInfo {
                                name: branch_name.clone(),
                                target,
                                ref_type,
                                is_head,
                            }))
                        }
                        Err(_) => Ok(None),
                    }
                }

                GitRef::Tag(tag_name) => {
                    let ref_name = format!("refs/tags/{}", tag_name);
                    match repo.find_reference(&ref_name) {
                        Ok(reference) => {
                            let target = reference.target().unwrap_or_else(Oid::zero);
                            let ref_type = reference.kind().unwrap_or(git2::ReferenceType::Direct);
                            Ok(Some(ReferenceInfo {
                                name: tag_name.clone(),
                                target,
                                ref_type,
                                is_head: false,
                            }))
                        }
                        Err(_) => Ok(None),
                    }
                }

                GitRef::Commit(_) | GitRef::Revspec(_) => {
                    // Commits and revspecs don't have associated references
                    Ok(None)
                }
            }
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))?
    }

    /// List all references in repository
    pub async fn list_references<P: AsRef<Path>>(
        &self,
        repo_path: P,
    ) -> Git2DBResult<Vec<ReferenceInfo>> {
        let repo_path = repo_path.as_ref().to_path_buf();

        tokio::task::spawn_blocking(move || -> Git2DBResult<Vec<ReferenceInfo>> {
            let repo_cache = GitManager::global().get_repository(&repo_path)?;
            let repo = repo_cache.open()?;

            let mut references = Vec::new();
            let current_head = repo.head().ok();

            // Iterate through all references
            let refs = repo.references().map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to list references: {}", e))
            })?;

            for reference in refs {
                let reference = reference.map_err(|e| {
                    Git2DBError::repository(&repo_path, format!("Failed to read reference: {}", e))
                })?;

                if let Some(name) = reference.shorthand() {
                    let target = reference.target().unwrap_or_else(Oid::zero);
                    let ref_type = reference.kind().unwrap_or(git2::ReferenceType::Direct);

                    // Check if this is the current HEAD
                    let is_head = current_head
                        .as_ref()
                        .and_then(|head| head.name())
                        .map(|head_name| reference.name() == Some(head_name))
                        .unwrap_or(false);

                    references.push(ReferenceInfo {
                        name: name.to_string(),
                        target,
                        ref_type,
                        is_head,
                    });
                }
            }

            Ok(references)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))?
    }

    /// Get the default branch name
    pub async fn get_default_branch<P: AsRef<Path>>(&self, repo_path: P) -> Git2DBResult<String> {
        let repo_path = repo_path.as_ref().to_path_buf();

        tokio::task::spawn_blocking(move || -> Git2DBResult<String> {
            let repo_cache = GitManager::global().get_repository(&repo_path)?;
            let repo = repo_cache.open()?;

            // Try to get the symbolic reference HEAD points to
            if let Ok(head_ref) = repo.head() {
                if let Some(name) = head_ref.symbolic_target() {
                    if let Some(branch_name) = name.strip_prefix("refs/heads/") {
                        return Ok(branch_name.to_string());
                    }
                }
            }

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
            let mut branches = repo.branches(Some(git2::BranchType::Local)).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to list branches: {}", e))
            })?;

            if let Some(Ok((branch, _))) = branches.next() {
                if let Some(name) = branch.name().map_err(|e| {
                    Git2DBError::repository(&repo_path, format!("Failed to get branch name: {}", e))
                })? {
                    return Ok(name.to_string());
                }
            }

            // Final fallback
            Ok("main".to_string())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))?
    }

    /// Clear the reference cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        debug!("Cleared reference cache");
    }
}

/// Resolve reference within a repository (standalone helper function)
fn resolve_in_repo(repo: &Repository, git_ref: &GitRef) -> Git2DBResult<Oid> {
    match git_ref {
        GitRef::DefaultBranch => repo
            .head()
            .and_then(|head| head.peel_to_commit())
            .map(|commit| commit.id())
            .map_err(|e| Git2DBError::reference("HEAD", format!("Failed to resolve HEAD: {}", e))),

        GitRef::Commit(oid) => {
            // Verify the commit exists
            repo.find_commit(*oid).map(|_| *oid).map_err(|e| {
                Git2DBError::reference(oid.to_string(), format!("Commit not found: {}", e))
            })
        }

        GitRef::Branch(branch_name) => {
            let ref_str = format!("refs/heads/{}", branch_name);
            // Try to find as a reference first
            if let Ok(reference) = repo.find_reference(&ref_str) {
                reference
                    .peel_to_commit()
                    .map(|commit| commit.id())
                    .map_err(|e| {
                        Git2DBError::reference(
                            &ref_str,
                            format!("Failed to resolve reference: {}", e),
                        )
                    })
            } else {
                // Fallback to revparse for complex expressions
                repo.revparse_single(&ref_str)
                    .and_then(|obj| obj.peel_to_commit())
                    .map(|commit| commit.id())
                    .map_err(|e| {
                        Git2DBError::reference(
                            &ref_str,
                            format!("Failed to resolve revspec: {}", e),
                        )
                    })
            }
        }

        GitRef::Tag(tag_name) => {
            let ref_str = format!("refs/tags/{}", tag_name);
            // Try to find as a reference first
            if let Ok(reference) = repo.find_reference(&ref_str) {
                reference
                    .peel_to_commit()
                    .map(|commit| commit.id())
                    .map_err(|e| {
                        Git2DBError::reference(
                            &ref_str,
                            format!("Failed to resolve reference: {}", e),
                        )
                    })
            } else {
                // Fallback to revparse for complex expressions
                repo.revparse_single(&ref_str)
                    .and_then(|obj| obj.peel_to_commit())
                    .map(|commit| commit.id())
                    .map_err(|e| {
                        Git2DBError::reference(
                            &ref_str,
                            format!("Failed to resolve revspec: {}", e),
                        )
                    })
            }
        }

        GitRef::Revspec(revspec) => {
            let ref_str = revspec.clone();

            // Try to find as a reference first
            if let Ok(reference) = repo.find_reference(&ref_str) {
                reference
                    .peel_to_commit()
                    .map(|commit| commit.id())
                    .map_err(|e| {
                        Git2DBError::reference(
                            &ref_str,
                            format!("Failed to resolve reference: {}", e),
                        )
                    })
            } else {
                // Fallback to revparse for complex expressions
                repo.revparse_single(&ref_str)
                    .and_then(|obj| obj.peel_to_commit())
                    .map(|commit| commit.id())
                    .map_err(|e| {
                        Git2DBError::reference(
                            &ref_str,
                            format!("Failed to resolve revspec: {}", e),
                        )
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_ref_parse() {
        assert_eq!(GitRef::parse("").unwrap(), GitRef::DefaultBranch);
        assert_eq!(GitRef::parse("HEAD").unwrap(), GitRef::DefaultBranch);
        assert_eq!(
            GitRef::parse("main").unwrap(),
            GitRef::Branch("main".to_string())
        );
        assert_eq!(
            GitRef::parse("refs/heads/main").unwrap(),
            GitRef::Branch("main".to_string())
        );
        assert_eq!(
            GitRef::parse("refs/tags/v1.0").unwrap(),
            GitRef::Tag("v1.0".to_string())
        );
        assert_eq!(
            GitRef::parse("tags/v1.0").unwrap(),
            GitRef::Tag("v1.0".to_string())
        );

        // Test commit hash
        let full_hash = "1234567890abcdef1234567890abcdef12345678";
        if let GitRef::Commit(oid) = GitRef::parse(full_hash).unwrap() {
            assert_eq!(oid.to_string(), full_hash);
        } else {
            panic!("Expected commit reference");
        }

        // Test short hash (should be revspec)
        assert_eq!(
            GitRef::parse("1234567").unwrap(),
            GitRef::Revspec("1234567".to_string())
        );
    }

    #[test]
    fn test_git_ref_display() {
        assert_eq!(GitRef::DefaultBranch.display_name(), "HEAD");
        assert_eq!(GitRef::Branch("main".to_string()).display_name(), "main");
        assert_eq!(GitRef::Tag("v1.0".to_string()).display_name(), "tags/v1.0");
        assert_eq!(
            GitRef::Revspec("feature".to_string()).display_name(),
            "feature"
        );
    }

    #[test]
    fn test_git_ref_to_ref_string() {
        assert_eq!(GitRef::DefaultBranch.to_ref_string(), None);
        assert_eq!(
            GitRef::Branch("main".to_string()).to_ref_string(),
            Some("refs/heads/main".to_string())
        );
        assert_eq!(
            GitRef::Tag("v1.0".to_string()).to_ref_string(),
            Some("refs/tags/v1.0".to_string())
        );
    }
}
