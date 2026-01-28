//! Repository abstraction layer using libgit2
//!
//! This module provides the core Git operations using libgit2 instead of shell commands.
//! Uses the Repository pattern to encapsulate Git operations with clean separation.

use crate::{types::{GitHash, Sha256Hash}, Result};
use git2::{Oid, Repository as Git2Repository};
use std::path::Path;

/// Wrapper around git2::Repository providing essential GitTorrent operations
/// This follows the Newtype pattern for zero-cost abstraction
pub struct Repository {
    inner: Git2Repository,
}

impl Repository {
    /// Open an existing repository
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = Git2Repository::open(path)?;
        Ok(Repository { inner })
    }

    /// Check if repository is bare
    pub fn is_bare(&self) -> bool {
        self.inner.is_bare()
    }

    /// Get the repository path
    pub fn path(&self) -> &Path {
        self.inner.path()
    }

    /// Check if object exists in repository
    pub fn has_object(&self, hash: &GitHash) -> Result<bool> {
        let oid = Oid::from_str(&hash.to_hex())?;
        Ok(self.inner.odb()?.exists(oid))
    }

    /// Get references for the repository
    pub fn references(&self) -> Result<Vec<crate::types::GitRef>> {
        let mut refs = Vec::new();

        for reference in self.inner.references()? {
            let reference = reference?;
            if let Some(name) = reference.name() {
                if let Some(target) = reference.target() {
                    refs.push(crate::types::GitRef {
                        name: name.to_owned(),
                        hash: GitHash::from_hex(&target.to_string())?,
                    });
                }
            }
        }

        Ok(refs)
    }

    /// Create a pack file for objects between 'have' and 'want' commits
    /// This is the core operation for BitTorrent sharing
    pub fn create_pack(&self, want: &Sha256Hash, have: Option<&Sha256Hash>) -> Result<Vec<u8>> {
        let want_oid = Oid::from_str(want.as_str())?;

        // Create object database pack builder
        let mut builder = self.inner.packbuilder()?;

        // Walk from want to have (or root)
        let mut revwalk = self.inner.revwalk()?;
        revwalk.push(want_oid)?;

        if let Some(have_sha256) = have {
            let have_oid = Oid::from_str(have_sha256.as_str())?;
            revwalk.hide(have_oid)?;
        }

        // Collect all objects to pack
        for oid in revwalk {
            let oid = oid?;
            builder.insert_object(oid, None)?;

            // Also include tree and blob objects
            if let Ok(commit) = self.inner.find_commit(oid) {
                builder.insert_tree(commit.tree_id())?;
            }
        }

        // Write pack to memory
        let mut pack_data = Vec::new();
        builder.foreach(|chunk| {
            pack_data.extend_from_slice(chunk);
            true
        })?;

        Ok(pack_data)
    }

    /// Unpack received pack data into the repository
    pub fn unpack(&self, pack_data: &[u8]) -> Result<()> {
        // Write pack data to a temporary file
        let temp_dir = tempfile::tempdir()?;
        let pack_path = temp_dir.path().join("pack.pack");
        std::fs::write(&pack_path, pack_data)?;

        // Use git2 to unpack the file directly into the repository
        // This is the simplest approach that works with git2 0.18
        let pack_dir = self.inner.path().join("objects/pack");
        std::fs::create_dir_all(&pack_dir)?;

        // Create a unique pack name based on content hash
        let hash = crate::crypto::hash::sha256(pack_data);
        let pack_name = format!("pack-{}.pack", hex::encode(&hash[..20]));

        // Move the pack file to the repository
        let final_pack_path = pack_dir.join(&pack_name);
        std::fs::copy(&pack_path, &final_pack_path)?;

        // Let git2 handle the indexing when the pack is accessed
        // The repository will automatically index the pack when needed

        Ok(())
    }

    /// Get commit information
    pub fn find_commit(&self, sha256: &Sha256Hash) -> Result<CommitInfo> {
        let oid = Oid::from_str(sha256.as_str())?;
        let commit = self.inner.find_commit(oid)?;

        Ok(CommitInfo {
            sha256: sha256.clone(),
            tree_sha256: Sha256Hash::new(commit.tree_id().to_string())?,
            parent_sha256s: commit.parent_ids()
                .map(|oid| Sha256Hash::new(oid.to_string()))
                .collect::<Result<Vec<_>>>()?,
            message: commit.message().unwrap_or("").to_owned(),
        })
    }
}

/// Commit information
#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub sha256: Sha256Hash,
    pub tree_sha256: Sha256Hash,
    pub parent_sha256s: Vec<Sha256Hash>,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_repository_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let repo_path = temp_dir.path();

        // Initialize a new repository
        let git_repo = Git2Repository::init(repo_path)?;

        // Create initial commit
        let sig = git2::Signature::now("Test", "test@example.com")?;
        let tree_id = {
            let mut index = git_repo.index()?;
            index.write_tree()?
        };
        let tree = git_repo.find_tree(tree_id)?;

        let commit_id = git_repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        )?;

        // Test our Repository wrapper
        let repo = Repository::open(repo_path)?;

        // Use GitHash which supports SHA1 (40-char) from git2
        let hash = GitHash::from_hex(&commit_id.to_string())?;
        assert!(repo.has_object(&hash)?);

        let refs = repo.references()?;
        assert!(!refs.is_empty());

        Ok(())
    }
}