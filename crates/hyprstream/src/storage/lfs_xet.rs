//! LFS to XET bridge for Hugging Face model repositories
//!
//! This module provides ML-model-specific functionality for working with Git LFS
//! repositories (common in Hugging Face) and translating them to use XET storage.
//!
//! Core XET operations are delegated to git2db's XetStorage implementation.

use anyhow::{anyhow, bail, Context, Result};
use git2db::xet::{StorageBackend, XetStorage};
use merklehash::MerkleHash;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::fs;
use tokio::io::AsyncReadExt;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

// Re-export XetConfig from git2db for convenience
pub use git2db::XetConfig;

/// LFS pointer information parsed from pointer file
#[derive(Debug, Clone)]
pub struct LfsPointer {
    pub version: String,
    pub oid: String,
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
                size = Some(s.parse::<u64>().context("Invalid size in LFS pointer")?);
            }
        }

        let version = version.ok_or_else(|| anyhow!("Missing version in LFS pointer"))?;
        let oid = oid.ok_or_else(|| anyhow!("Missing oid in LFS pointer"))?;
        let size = size.ok_or_else(|| anyhow!("Missing size in LFS pointer"))?;

        // Validate OID format (should be 64 character hex)
        if oid.len() != 64 || !oid.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("Invalid LFS OID format: expected 64 character hex string");
        }

        Ok(LfsPointer { version, oid, size })
    }
}

/// Bridge between Git LFS and XET storage
///
/// This wraps git2db's XetStorage and adds LFS-specific functionality
/// for working with Hugging Face model repositories.
#[derive(Clone)]
pub struct LfsXetBridge {
    xet: Arc<XetStorage>,
    config: XetConfig,
}

impl LfsXetBridge {
    /// Create a new LFS-XET bridge
    pub async fn new(config: XetConfig) -> Result<Self> {
        let xet = XetStorage::new(&config).await?;
        Ok(Self {
            xet: Arc::new(xet),
            config,
        })
    }

    // ========================================================================
    // Core XET Operations (delegated to git2db)
    // ========================================================================

    /// Upload file and return XET pointer
    pub async fn clean_file(&self, file_path: &Path) -> Result<String> {
        self.xet.clean_file(file_path).await
            .map_err(|e| anyhow!("XET clean_file failed: {}", e))
    }

    /// Upload data from memory and return XET pointer
    pub async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        self.xet.clean_bytes(data).await
            .map_err(|e| anyhow!("XET clean_bytes failed: {}", e))
    }

    /// Download XET pointer to file
    pub async fn smudge_file(&self, pointer_content: &str, output_path: &Path) -> Result<()> {
        self.xet.smudge_file(pointer_content, output_path).await
            .map_err(|e| anyhow!("XET smudge_file failed: {}", e))
    }

    /// Download XET pointer to memory
    pub async fn smudge_bytes(&self, pointer_content: &str) -> Result<Vec<u8>> {
        self.xet.smudge_bytes(pointer_content).await
            .map_err(|e| anyhow!("XET smudge_bytes failed: {}", e))
    }

    /// Check if content is a XET pointer
    pub fn is_xet_pointer(&self, content: &str) -> bool {
        self.xet.is_pointer(content)
    }

    // ========================================================================
    // LFS-Specific Functionality (ML model domain)
    // ========================================================================

    /// Check if content is a Git LFS pointer file
    pub fn is_lfs_pointer(&self, content: &str) -> bool {
        content.starts_with("version https://git-lfs.github.com/spec/v1")
    }

    /// Check if content is any type of pointer file (XET or LFS)
    pub fn is_pointer(&self, content: &str) -> bool {
        self.is_xet_pointer(content) || self.is_lfs_pointer(content)
    }

    /// Parse LFS pointer from content
    pub fn parse_lfs_pointer(&self, content: &str) -> Result<LfsPointer> {
        LfsPointer::parse(content)
    }

    /// Convert LFS SHA256 to MerkleHash format expected by XET
    fn sha256_to_merkle_hash(&self, sha256_hex: &str) -> Result<MerkleHash> {
        MerkleHash::from_hex(sha256_hex).context("Failed to convert LFS SHA256 to MerkleHash")
    }

    /// Validate in-memory content against LFS pointer metadata
    ///
    /// This performs both size and SHA256 validation before writing to disk.
    fn validate_content_against_pointer(
        &self,
        content: &[u8],
        lfs_pointer: &LfsPointer,
    ) -> Result<()> {
        // Size validation
        if content.len() as u64 != lfs_pointer.size {
            bail!(
                "Size mismatch: expected {} bytes, got {} bytes (SHA256: {})",
                lfs_pointer.size,
                content.len(),
                &lfs_pointer.oid[..8]
            );
        }

        // SHA256 validation
        let mut hasher = Sha256::new();
        hasher.update(content);
        let computed_hash = format!("{:x}", hasher.finalize());

        if computed_hash != lfs_pointer.oid {
            bail!(
                "SHA256 mismatch: expected {}, got {} (size: {} bytes)",
                lfs_pointer.oid,
                computed_hash,
                content.len()
            );
        }

        debug!(
            "✅ Validated content: {} bytes, SHA256: {}...",
            lfs_pointer.size,
            &lfs_pointer.oid[..8]
        );

        Ok(())
    }

    /// Validate a file against LFS pointer metadata
    ///
    /// This performs streaming SHA256 validation to minimize memory usage.
    async fn validate_lfs_file(&self, file_path: &Path, lfs_pointer: &LfsPointer) -> Result<()> {
        // Size validation
        let metadata = fs::metadata(file_path)
            .await
            .context("Failed to get file metadata for validation")?;

        if metadata.len() != lfs_pointer.size {
            bail!(
                "Size mismatch for {}: expected {} bytes, got {} bytes",
                file_path.display(),
                lfs_pointer.size,
                metadata.len()
            );
        }

        // Streaming SHA256 validation
        let mut file = fs::File::open(file_path)
            .await
            .context("Failed to open file for SHA256 validation")?;

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192]; // 8KB buffer for streaming

        loop {
            let n = file.read(&mut buffer).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        let computed_hash = format!("{:x}", hasher.finalize());

        if computed_hash != lfs_pointer.oid {
            bail!(
                "SHA256 mismatch for {}: expected {}, got {}",
                file_path.display(),
                lfs_pointer.oid,
                computed_hash
            );
        }

        info!(
            "✅ Validated LFS file: {} ({} bytes, SHA256: {}...)",
            file_path.display(),
            lfs_pointer.size,
            &lfs_pointer.oid[..8]
        );

        Ok(())
    }

    /// Check if a file is already properly smudged (not a pointer)
    ///
    /// This helps avoid redundant processing of files that have already been smudged.
    async fn is_already_smudged(&self, file_path: &Path) -> Result<bool> {
        // Read first 200 bytes to check for pointer markers
        let mut file = match fs::File::open(file_path).await {
            Ok(f) => f,
            Err(_) => return Ok(false), // File doesn't exist = not smudged
        };

        let mut buffer = vec![0u8; 200];
        let n = file.read(&mut buffer).await?;

        if n == 0 {
            return Ok(false); // Empty file = not smudged
        }

        // Try to parse as text for pointer detection
        if let Ok(content) = String::from_utf8(buffer[..n].to_vec()) {
            if self.is_lfs_pointer(&content) || self.is_xet_pointer(&content) {
                return Ok(false); // Still a pointer, needs smudging
            }
        }

        // Not a pointer = already smudged
        debug!("File already smudged: {}", file_path.display());
        Ok(true)
    }

    /// Resolve LFS pointer to actual content via XET
    ///
    /// This method takes an LFS pointer and attempts to retrieve the actual file content
    /// from XET storage using the SHA256 OID.
    pub async fn smudge_lfs_pointer(&self, lfs_pointer: &LfsPointer) -> Result<Vec<u8>> {
        // Convert LFS SHA256 to MerkleHash format expected by XET
        let merkle_hash = self.sha256_to_merkle_hash(&lfs_pointer.oid)?;

        // Use XET's hash-based download
        self.xet.smudge_from_hash(&merkle_hash).await
            .map_err(|e| anyhow!("XET smudge_from_hash failed: {}", e))
    }

    /// Universal file loader - handles XET pointers, LFS pointers, and regular files
    ///
    /// This is the main interface for loading model files. It automatically detects
    /// if the file contains a XET pointer, LFS pointer, and retrieves the actual content,
    /// or returns the file content directly if it's not a pointer.
    /// Load file with automatic LFS/XET pointer detection and smudging
    ///
    /// This method implements LFS spec-compliant pointer detection:
    /// 1. Checks file size via metadata (pointers MUST be < 1024 bytes)
    /// 2. Reads first 100 bytes for pointer detection (per LFS spec)
    /// 3. If pointer detected, reads rest and smudges
    /// 4. If not pointer, reads full file efficiently
    ///
    /// Use this for files outside git operations (git-xet-filter handles files in git repos).
    ///
    /// # LFS Specification
    ///
    /// Per the Git LFS spec:
    /// - "Pointer files must be less than 1024 bytes in size"
    /// - "Read 100 bytes. If the content is ASCII and matches the pointer file format"
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file to load
    ///
    /// # Returns
    ///
    /// File contents, with pointers automatically smudged to actual content
    pub async fn load_file(&self, file_path: &Path) -> Result<Vec<u8>> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        // 1. Check metadata FIRST (instant, no file content I/O)
        // LFS spec: "Pointer files must be less than 1024 bytes in size"
        let metadata = fs::metadata(file_path)
            .await
            .context("Failed to get file metadata")?;
        let file_size = metadata.len();

        // 2. Large files (>= 1024 bytes) cannot be pointers - read directly
        if file_size >= 1024 {
            return fs::read(file_path)
                .await
                .context("Failed to read file");
        }

        // 3. Small file - might be a pointer
        // LFS spec: "Read 100 bytes. If the content is ASCII and matches the pointer file format"
        let mut file = fs::File::open(file_path)
            .await
            .context("Failed to open file")?;

        let mut header = vec![0u8; 100];
        let n = file.read(&mut header).await?;
        header.truncate(n);

        // 4. Check for pointer markers (LFS and XET pointers are UTF-8 text)
        if let Ok(text) = String::from_utf8(header.clone()) {
            if text.starts_with("version https://git-lfs") || text.starts_with("# xet version") {
                // It's a pointer - read the rest (file is < 1024 bytes total)
                let mut rest = Vec::new();
                file.read_to_end(&mut rest).await?;

                let full_content = [&header[..n], &rest[..]].concat();
                let full_text = String::from_utf8(full_content)
                    .context("Pointer file is not valid UTF-8")?;

                // Parse and smudge the appropriate pointer type
                if self.is_lfs_pointer(&full_text) {
                    let lfs_pointer = self.parse_lfs_pointer(&full_text)?;
                    return self.smudge_lfs_pointer(&lfs_pointer).await;
                } else if self.is_xet_pointer(&full_text) {
                    return self.smudge_bytes(&full_text).await;
                }
            }
        }

        // 5. Not a pointer - seek back to start and read full file
        file.seek(std::io::SeekFrom::Start(0)).await?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).await?;

        Ok(contents)
    }

    /// Scan directory for LFS files and return their paths, respecting .gitignore
    ///
    /// This method uses git2's native tree traversal and status APIs for efficiency
    pub async fn scan_lfs_files_in_directory(&self, directory: &Path) -> Result<Vec<PathBuf>> {
        let directory = directory.to_path_buf();
        let bridge = self.clone();

        tokio::task::spawn_blocking(move || bridge.scan_lfs_files_in_directory_sync(&directory))
            .await?
    }

    /// Synchronous implementation of LFS file scanning using git2's tree traversal
    fn scan_lfs_files_in_directory_sync(&self, directory: &Path) -> Result<Vec<PathBuf>> {
        use git2::{ObjectType, TreeWalkMode, TreeWalkResult};

        // Find the git repository for this directory
        let repo = self.find_git_repository(directory)?;
        let repo_workdir = repo
            .workdir()
            .ok_or_else(|| anyhow!("Repository has no working directory"))?;

        // Get the current HEAD commit and its tree
        let head = repo.head().context("Failed to get HEAD reference")?;
        let commit = head.peel_to_commit().context("Failed to get HEAD commit")?;
        let tree = commit.tree().context("Failed to get commit tree")?;

        // Convert directory to relative path if needed
        let scan_prefix = if directory.starts_with(repo_workdir) {
            directory
                .strip_prefix(repo_workdir)
                .context("Failed to get relative path")?
                .to_string_lossy()
                .into_owned()
        } else {
            String::new() // Scan entire repo
        };

        // Collect LFS candidates using git2's tree walk
        let mut lfs_candidates = Vec::new();

        tree.walk(TreeWalkMode::PreOrder, |root, entry| {
            // Build full path
            let entry_path = if root.is_empty() {
                entry.name().unwrap_or("").to_string()
            } else {
                format!(
                    "{}/{}",
                    root.trim_end_matches('/'),
                    entry.name().unwrap_or("")
                )
            };

            // Check if this path is within our scan prefix
            if !scan_prefix.is_empty() && !entry_path.starts_with(&scan_prefix) {
                return TreeWalkResult::Skip;
            }

            // Only process regular files (blobs)
            if let Some(ObjectType::Blob) = entry.kind() {
                // Try to read the blob content to check for LFS pointer
                if let Ok(blob) = repo.find_blob(entry.id()) {
                    if let Ok(content) = std::str::from_utf8(blob.content()) {
                        if self.is_lfs_pointer(content) {
                            let full_path = repo_workdir.join(&entry_path);
                            lfs_candidates.push(full_path);
                        }
                    }
                }
            }

            TreeWalkResult::Ok
        })?;

        // Filter candidates using git status to respect .gitignore
        let filtered_lfs_files = self.filter_files_by_git_status_sync(&repo, lfs_candidates)?;

        Ok(filtered_lfs_files)
    }

    /// Find the git repository for a given directory
    fn find_git_repository(&self, directory: &Path) -> Result<git2::Repository> {
        git2::Repository::discover(directory)
            .context("Directory is not part of a git repository")
    }

    /// Filter candidate files using git status to respect .gitignore (sync version)
    fn filter_files_by_git_status_sync(
        &self,
        repo: &git2::Repository,
        candidates: Vec<PathBuf>,
    ) -> Result<Vec<PathBuf>> {
        let repo_workdir = repo
            .workdir()
            .ok_or_else(|| anyhow!("Repository has no working directory"))?;

        let mut filtered_files = Vec::new();
        let candidate_count = candidates.len();

        for candidate in candidates {
            // Convert to relative path for git operations
            if let Ok(relative_path) = candidate.strip_prefix(repo_workdir) {
                // Check if file exists in worktree
                if candidate.exists() {
                    // Check if it's ignored using git's ignore rules
                    let is_ignored = repo
                        .status_file(relative_path)
                        .map(|status| status.contains(git2::Status::IGNORED))
                        .unwrap_or(false);

                    if !is_ignored {
                        debug!("Found LFS file: {}", candidate.display());
                        filtered_files.push(candidate);
                    }
                }
            }
        }

        debug!(
            "Filtered {} LFS candidates to {} files",
            candidate_count,
            filtered_files.len()
        );
        Ok(filtered_files)
    }

    /// Process all LFS files in a worktree by smudging them to actual content
    ///
    /// This scans the entire worktree directory for LFS pointer files and attempts
    /// to resolve them via XET storage.
    pub async fn process_worktree_lfs(&self, worktree_path: &Path) -> Result<Vec<PathBuf>> {
        if !worktree_path.is_dir() {
            bail!(
                "Worktree path does not exist or is not a directory: {}",
                worktree_path.display()
            );
        }

        info!(
            "Scanning worktree for LFS files: {}",
            worktree_path.display()
        );
        let lfs_files = self.scan_lfs_files_in_directory(worktree_path).await?;

        if lfs_files.is_empty() {
            debug!("No LFS files found in worktree");
            return Ok(Vec::new());
        }

        info!("Found {} LFS files to process", lfs_files.len());
        let mut processed_files = Vec::new();

        for lfs_file in &lfs_files {
            debug!("Processing LFS file: {}", lfs_file.display());

            match self.smudge_lfs_file_in_place(lfs_file).await {
                Ok(()) => {
                    processed_files.push(lfs_file.clone());
                    debug!("Successfully processed: {}", lfs_file.display());
                }
                Err(e) => {
                    warn!("Failed to process LFS file {}: {}", lfs_file.display(), e);
                    // Continue processing other files instead of failing completely
                }
            }
        }

        info!(
            "Processed {}/{} LFS files",
            processed_files.len(),
            lfs_files.len()
        );
        Ok(processed_files)
    }

    /// Smudge an LFS file in place (replace pointer with actual content)
    async fn smudge_lfs_file_in_place(&self, file_path: &Path) -> Result<()> {
        // Check if already smudged to avoid redundant work
        if self.is_already_smudged(file_path).await? {
            debug!("Skipping already smudged file: {}", file_path.display());
            return Ok(());
        }

        // Read the LFS pointer content
        let pointer_content = fs::read_to_string(file_path)
            .await
            .context("Failed to read LFS pointer file")?;

        if !self.is_lfs_pointer(&pointer_content) {
            bail!("File is not an LFS pointer: {}", file_path.display());
        }

        // Parse the LFS pointer
        let lfs_pointer = self.parse_lfs_pointer(&pointer_content)?;

        debug!(
            "Smudging LFS file: {} ({} bytes, SHA256: {}...)",
            file_path.display(),
            lfs_pointer.size,
            &lfs_pointer.oid[..8]
        );

        // Retrieve actual content via XET
        let actual_content = self.smudge_lfs_pointer(&lfs_pointer).await?;

        // Validate content BEFORE writing (pre-validation)
        self.validate_content_against_pointer(&actual_content, &lfs_pointer)
            .context("Pre-write validation failed")?;

        // Write actual content to a temporary file first
        let temp_file = NamedTempFile::new_in(file_path.parent().unwrap_or(Path::new(".")))
            .context("Failed to create temporary file for atomic replacement")?;

        fs::write(temp_file.path(), &actual_content)
            .await
            .context("Failed to write smudged content to temporary file")?;

        // Atomically replace the original file
        let temp_path = temp_file.into_temp_path();
        temp_path
            .persist(file_path)
            .context("Failed to atomically replace LFS pointer with actual content")?;

        // Validate file AFTER writing (post-validation, defense in depth)
        self.validate_lfs_file(file_path, &lfs_pointer)
            .await
            .context("Post-write validation failed")?;

        Ok(())
    }

    /// Process all LFS files efficiently in parallel
    pub async fn batch_process_lfs_files(&self, worktree_path: &Path) -> Result<Vec<PathBuf>> {
        let lfs_files = self.scan_lfs_files_in_directory(worktree_path).await?;

        if lfs_files.is_empty() {
            debug!("No LFS files found in worktree");
            return Ok(Vec::new());
        }

        info!("Found {} LFS files to process in parallel", lfs_files.len());

        // Process files concurrently
        let semaphore = Arc::new(Semaphore::new(8)); // Limit concurrency to 8
        let mut tasks = Vec::new();

        for lfs_file in lfs_files {
            let semaphore = Arc::clone(&semaphore);
            let bridge = self.clone();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                bridge
                    .smudge_lfs_file_in_place(&lfs_file)
                    .await
                    .map(|_| lfs_file)
            });
            tasks.push(task);
        }

        // Collect results
        let mut processed_files = Vec::new();
        let total_files = tasks.len();
        for task in tasks {
            match task.await? {
                Ok(file_path) => {
                    debug!("Successfully processed: {}", file_path.display());
                    processed_files.push(file_path);
                }
                Err(e) => {
                    warn!("Failed to process LFS file: {}", e);
                    // Continue processing other files
                }
            }
        }

        info!(
            "Processed {}/{} LFS files",
            processed_files.len(),
            total_files
        );
        Ok(processed_files)
    }

    /// Get configuration
    pub fn config(&self) -> &XetConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lfs_pointer_detection() {
        let config = XetConfig::default();
        let bridge = LfsXetBridge::new(config).await.unwrap();

        // Test valid LFS pointer
        let valid_lfs_pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        assert!(bridge.is_lfs_pointer(valid_lfs_pointer));
        assert!(bridge.is_pointer(valid_lfs_pointer));

        // Test invalid content
        let regular_content = "This is just regular file content";
        assert!(!bridge.is_lfs_pointer(regular_content));
        assert!(!bridge.is_pointer(regular_content));
    }

    #[tokio::test]
    async fn test_lfs_pointer_parsing() {
        let config = XetConfig::default();
        let bridge = LfsXetBridge::new(config).await.unwrap();

        // Test valid LFS pointer parsing
        let valid_lfs_content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        let parsed = bridge.parse_lfs_pointer(valid_lfs_content).unwrap();

        assert_eq!(parsed.version, "https://git-lfs.github.com/spec/v1");
        assert_eq!(
            parsed.oid,
            "abc123def456789012345678901234567890123456789012345678901234567890"
        );
        assert_eq!(parsed.size, 12345);
    }

    #[test]
    fn test_lfs_pointer_struct_validation() {
        // Test valid OID
        let valid_oid = "a".repeat(64);
        let result = LfsPointer::parse(&format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize 1000",
            valid_oid
        ));
        assert!(result.is_ok());

        // Test invalid OID length
        let short_oid = "a".repeat(32);
        let result = LfsPointer::parse(&format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize 1000",
            short_oid
        ));
        assert!(result.is_err());

        // Test invalid OID characters
        let invalid_oid = "g".repeat(64); // 'g' is not hex
        let result = LfsPointer::parse(&format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize 1000",
            invalid_oid
        ));
        assert!(result.is_err());
    }
}
