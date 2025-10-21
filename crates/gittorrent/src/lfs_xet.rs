//! LFS + XET Integration for GitTorrent
//!
//! This module handles Git LFS pointer files and integrates them with XET
//! for distributed chunk storage. It provides transparent handling of large
//! files by converting between LFS pointers and XET chunks.

use crate::xet_integration::{MerkleHash, ContentAddress};
use crate::{Result, Error};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};

/// Git LFS pointer file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsPointer {
    /// LFS version (usually "https://git-lfs.github.com/spec/v1")
    pub version: String,
    /// Object ID (SHA-256 hash)
    pub oid: String,
    /// File size in bytes
    pub size: u64,
}

impl LfsPointer {
    /// Parse LFS pointer from text content
    pub fn parse(content: &str) -> Result<Self> {
        let mut version = None;
        let mut oid = None;
        let mut size = None;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(v) = line.strip_prefix("version ") {
                version = Some(v.to_string());
            } else if let Some(o) = line.strip_prefix("oid sha256:") {
                oid = Some(o.to_string());
            } else if let Some(s) = line.strip_prefix("size ") {
                size = Some(s.parse::<u64>()
                    .map_err(|_| Error::other("Invalid size in LFS pointer"))?);
            }
        }

        let version = version.ok_or_else(|| Error::other("Missing version in LFS pointer"))?;
        let oid = oid.ok_or_else(|| Error::other("Missing oid in LFS pointer"))?;
        let size = size.ok_or_else(|| Error::other("Missing size in LFS pointer"))?;

        // Validate OID format (should be 64 character hex)
        if oid.len() != 64 || !oid.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(Error::other("Invalid LFS OID format: expected 64 character hex string"));
        }

        Ok(LfsPointer { version, oid, size })
    }

    /// Convert to text format
    pub fn to_string(&self) -> String {
        format!(
            "version {}\noid sha256:{}\nsize {}\n",
            self.version, self.oid, self.size
        )
    }

    /// Check if content looks like an LFS pointer
    pub fn is_lfs_pointer(content: &str) -> bool {
        content.starts_with("version https://git-lfs.github.com/spec/v1")
    }
}

/// XET pointer file structure (JSON format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XetPointer {
    /// XET version
    pub version: String,
    /// Merkle hash of the content
    pub hash: String,
    /// File size in bytes
    pub file_size: u64,
    /// Chunk information
    pub chunks: Vec<XetChunk>,
}

/// Individual XET chunk information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XetChunk {
    /// Chunk hash
    pub hash: String,
    /// Chunk size
    pub size: u64,
    /// Chunk offset in original file
    pub offset: u64,
}

impl XetPointer {
    /// Parse XET pointer from JSON content
    pub fn parse(content: &str) -> Result<Self> {
        serde_json::from_str(content)
            .map_err(|e| Error::other(format!("Invalid XET pointer JSON: {}", e)))
    }

    /// Convert to JSON string
    pub fn to_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::other(format!("Failed to serialize XET pointer: {}", e)))
    }

    /// Check if content looks like an XET pointer
    pub fn is_xet_pointer(content: &str) -> bool {
        content.trim_start().starts_with('{') &&
        content.contains("\"hash\"") &&
        content.contains("\"file_size\"")
    }

    /// Convert to MerkleHash
    pub fn to_merkle_hash(&self) -> Result<MerkleHash> {
        MerkleHash::from_hex(&self.hash)
    }
}

/// LFS + XET integration manager
pub struct LfsXetManager {
    /// Local LFS objects directory
    lfs_dir: PathBuf,
}

impl LfsXetManager {
    /// Create a new LFS-XET manager
    pub fn new(repo_path: &Path) -> Self {
        Self {
            lfs_dir: repo_path.join(".git").join("lfs").join("objects"),
        }
    }

    /// Check if a file is an LFS pointer
    pub async fn is_lfs_pointer_file(&self, file_path: &Path) -> Result<bool> {
        if !file_path.exists() {
            return Ok(false);
        }

        // Read first few bytes to check for LFS marker
        let content = tokio::fs::read_to_string(file_path).await
            .map_err(Error::from)?;

        Ok(LfsPointer::is_lfs_pointer(&content))
    }

    /// Check if a file is an XET pointer
    pub async fn is_xet_pointer_file(&self, file_path: &Path) -> Result<bool> {
        if !file_path.exists() {
            return Ok(false);
        }

        let content = tokio::fs::read_to_string(file_path).await
            .map_err(Error::from)?;

        Ok(XetPointer::is_xet_pointer(&content))
    }

    /// Convert LFS pointer to XET chunks
    pub async fn lfs_to_xet(&self, lfs_pointer: &LfsPointer) -> Result<XetPointer> {
        // Get LFS object file path
        let obj_path = self.get_lfs_object_path(&lfs_pointer.oid);

        if !obj_path.exists() {
            return Err(Error::not_found(format!(
                "LFS object not found: {}", lfs_pointer.oid
            )));
        }

        // Read the actual file content
        let content = tokio::fs::read(&obj_path).await
            .map_err(Error::from)?;

        // For now, treat the entire file as one chunk
        // In a real implementation, this would use XET's chunking algorithm
        let content_addr = ContentAddress::new(&content);
        let merkle_hash = MerkleHash { hash: content_addr.root_hash };

        let xet_pointer = XetPointer {
            version: "1.0".to_string(),
            hash: merkle_hash.to_hex(),
            file_size: content.len() as u64,
            chunks: vec![XetChunk {
                hash: merkle_hash.to_hex(),
                size: content.len() as u64,
                offset: 0,
            }],
        };

        Ok(xet_pointer)
    }

    /// Convert XET pointer back to LFS format
    pub async fn xet_to_lfs(&self, xet_pointer: &XetPointer) -> Result<LfsPointer> {
        // Calculate SHA-256 of the content for LFS compatibility
        // This would normally reconstruct the file from XET chunks first
        let oid = format!("{:064x}", 0u64); // Placeholder - would be actual SHA-256

        Ok(LfsPointer {
            version: "https://git-lfs.github.com/spec/v1".to_string(),
            oid,
            size: xet_pointer.file_size,
        })
    }

    /// Process LFS pointer file and convert to XET
    pub async fn process_lfs_file(&self, file_path: &Path) -> Result<()> {
        // Read LFS pointer
        let content = tokio::fs::read_to_string(file_path).await
            .map_err(Error::from)?;

        let lfs_pointer = LfsPointer::parse(&content)?;

        // Convert to XET
        let xet_pointer = self.lfs_to_xet(&lfs_pointer).await?;

        // Write XET pointer back to file
        let xet_content = xet_pointer.to_string()?;
        tokio::fs::write(file_path, xet_content).await
            .map_err(Error::from)?;

        tracing::info!("Converted LFS pointer to XET: {}", file_path.display());
        Ok(())
    }

    /// Scan repository for LFS files and convert them
    pub async fn convert_lfs_files(&self, repo_path: &Path) -> Result<Vec<PathBuf>> {
        Box::pin(self.convert_lfs_files_impl(repo_path)).await
    }

    /// Implementation of LFS file conversion (to avoid recursion issues)
    async fn convert_lfs_files_impl(&self, repo_path: &Path) -> Result<Vec<PathBuf>> {
        let mut converted_files = Vec::new();

        // Recursively scan for LFS pointer files
        let mut entries = tokio::fs::read_dir(repo_path).await
            .map_err(Error::from)?;

        while let Some(entry) = entries.next_entry().await.map_err(Error::from)? {
            let path = entry.path();

            if path.is_file() && self.is_lfs_pointer_file(&path).await? {
                self.process_lfs_file(&path).await?;
                converted_files.push(path);
            } else if path.is_dir() && !path.file_name().unwrap_or_default().to_string_lossy().starts_with('.') {
                // Recursively process subdirectories (except .git)
                let mut sub_files = Box::pin(self.convert_lfs_files_impl(&path)).await?;
                converted_files.append(&mut sub_files);
            }
        }

        Ok(converted_files)
    }

    /// Get LFS object file path from OID
    fn get_lfs_object_path(&self, oid: &str) -> PathBuf {
        // LFS stores objects as .git/lfs/objects/XX/YYYYYY where XX is first 2 chars of OID
        let prefix = &oid[..2];
        let suffix = &oid[2..];
        self.lfs_dir.join(prefix).join(suffix)
    }

    /// Get XET chunks for a file
    pub async fn get_xet_chunks(&self, file_path: &Path) -> Result<Vec<MerkleHash>> {
        if !self.is_xet_pointer_file(file_path).await? {
            return Ok(vec![]);
        }

        let content = tokio::fs::read_to_string(file_path).await
            .map_err(Error::from)?;

        let xet_pointer = XetPointer::parse(&content)?;

        let mut chunks = Vec::new();
        for chunk in &xet_pointer.chunks {
            let merkle_hash = MerkleHash::from_hex(&chunk.hash)?;
            chunks.push(merkle_hash);
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_lfs_pointer_parsing() {
        let lfs_content = "version https://git-lfs.github.com/spec/v1\noid sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890\nsize 1024\n";

        let pointer = LfsPointer::parse(lfs_content).unwrap();
        assert_eq!(pointer.version, "https://git-lfs.github.com/spec/v1");
        assert_eq!(pointer.oid, "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890");
        assert_eq!(pointer.size, 1024);
    }

    #[test]
    fn test_xet_pointer_parsing() {
        let xet_content = r#"{
            "version": "1.0",
            "hash": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "file_size": 1024,
            "chunks": []
        }"#;

        let pointer = XetPointer::parse(xet_content).unwrap();
        assert_eq!(pointer.version, "1.0");
        assert_eq!(pointer.hash, "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890");
        assert_eq!(pointer.file_size, 1024);
    }

    #[test]
    fn test_pointer_detection() {
        let lfs_content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 100\n";
        assert!(LfsPointer::is_lfs_pointer(lfs_content));

        let xet_content = r#"{"hash": "abc123", "file_size": 100}"#;
        assert!(XetPointer::is_xet_pointer(xet_content));

        let regular_content = "This is just a regular file";
        assert!(!LfsPointer::is_lfs_pointer(regular_content));
        assert!(!XetPointer::is_xet_pointer(regular_content));
    }

    #[tokio::test]
    async fn test_lfs_xet_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = LfsXetManager::new(temp_dir.path());

        // Test file should not be detected as LFS or XET pointer
        let test_file = temp_dir.path().join("test.txt");
        tokio::fs::write(&test_file, "regular content").await.unwrap();

        assert!(!manager.is_lfs_pointer_file(&test_file).await.unwrap());
        assert!(!manager.is_xet_pointer_file(&test_file).await.unwrap());
    }
}