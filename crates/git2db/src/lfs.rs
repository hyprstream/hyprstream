//! Git LFS pointer support with XET storage backend
//!
//! This module provides complete LFS handling for git2db:
//! - Pointer parsing (spec-compliant, supports legacy hawser version)
//! - File loading with automatic pointer detection
//! - Worktree scanning and batch processing
//! - SHA256 validation (defense in depth against XET CAS bugs)
//!
//! # Architecture
//!
//! ```text
//! Application → git2db::lfs::LfsStorage → git2db::xet::XetStorage
//!                                       → git-xet-filter (XET only)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use git2db::lfs::{LfsStorage, is_lfs_pointer};
//!
//! // Initialize LFS storage
//! let config = git2db::XetConfig::default();
//! let storage = LfsStorage::new(&config).await?;
//!
//! // Load file with automatic pointer detection
//! let data = storage.load_file(&path).await?;
//!
//! // Or parse and smudge manually
//! if is_lfs_pointer(&content) {
//!     let pointer = storage.parse_lfs_pointer(&content)?;
//!     let data = storage.smudge_lfs_pointer(&pointer).await?;
//! }
//! ```

use crate::errors::{Git2DBError, Git2DBResult, LfsErrorKind};
use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

#[cfg(feature = "xet-storage")]
use crate::xet::{StorageBackend, XetConfig, XetStorage};

/// Parsed Git LFS pointer with spec-compliant fields
///
/// Per the Git LFS specification, pointer files contain:
/// - `version` - URL of the LFS spec
/// - `oid sha256:` - SHA256 hash of the content
/// - `size` - Size in bytes
/// - Additional extension fields (preserved for spec compliance)
///
/// Fields are private to allow future API evolution without breaking changes.
#[derive(Debug, Clone)]
pub struct LfsPointer {
    /// LFS version URL (git-lfs or legacy hawser)
    version: String,
    /// Hash method used (currently always "sha256", future-proof)
    hash_method: String,
    /// Object ID (SHA256 hash in hex)
    oid: String,
    /// File size in bytes
    size: u64,
    /// Extension fields (preserves unknown keys per LFS spec)
    extensions: HashMap<String, String>,
}

impl LfsPointer {
    /// Create a new LFS pointer with the given fields
    ///
    /// This constructor validates the OID format.
    pub fn new(version: String, oid: String, size: u64) -> Git2DBResult<Self> {
        // Default to sha256 hash method
        Self::new_with_hash_method(version, "sha256".to_string(), oid, size)
    }

    /// Create a new LFS pointer with explicit hash method
    pub fn new_with_hash_method(
        version: String,
        hash_method: String,
        oid: String,
        size: u64,
    ) -> Git2DBResult<Self> {
        // Validate OID format for SHA256 (should be 64 character hex)
        if hash_method == "sha256" {
            if oid.len() != 64 || !oid.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(Git2DBError::lfs(
                    LfsErrorKind::InvalidPointer,
                    format!("Invalid OID format: expected 64 hex characters, got {}", oid.len()),
                ));
            }
        }

        Ok(LfsPointer {
            version,
            hash_method,
            oid,
            size,
            extensions: HashMap::new(),
        })
    }

    /// Parse LFS pointer from text content
    ///
    /// Supports both standard git-lfs and legacy hawser version URLs.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let content = r#"version https://git-lfs.github.com/spec/v1
    /// oid sha256:abc123...
    /// size 1024
    /// "#;
    /// let pointer = LfsPointer::parse(content)?;
    /// ```
    pub fn parse(content: &str) -> Git2DBResult<Self> {
        let mut version = None;
        let mut hash_method = None;
        let mut oid = None;
        let mut size = None;
        let mut extensions = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(v) = line.strip_prefix("version ") {
                version = Some(v.to_string());
            } else if let Some(o) = line.strip_prefix("oid sha256:") {
                hash_method = Some("sha256".to_string());
                oid = Some(o.to_string());
            } else if let Some(s) = line.strip_prefix("size ") {
                size = Some(s.parse::<u64>().map_err(|_| {
                    Git2DBError::lfs(LfsErrorKind::ParseError, "Invalid size value")
                })?);
            } else if let Some((key, value)) = line.split_once(' ') {
                // Preserve extension fields per LFS spec
                extensions.insert(key.to_string(), value.to_string());
            }
        }

        let version = version.ok_or_else(|| {
            Git2DBError::lfs(LfsErrorKind::InvalidPointer, "Missing version field")
        })?;
        let hash_method = hash_method.ok_or_else(|| {
            Git2DBError::lfs(LfsErrorKind::InvalidPointer, "Missing or unsupported oid hash method")
        })?;
        let oid = oid.ok_or_else(|| {
            Git2DBError::lfs(LfsErrorKind::InvalidPointer, "Missing oid field")
        })?;
        let size = size.ok_or_else(|| {
            Git2DBError::lfs(LfsErrorKind::InvalidPointer, "Missing size field")
        })?;

        // Validate OID format (should be 64 character hex for SHA256)
        if oid.len() != 64 || !oid.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(Git2DBError::lfs(
                LfsErrorKind::InvalidPointer,
                format!("Invalid OID format: expected 64 hex characters, got {}", oid.len()),
            ));
        }

        Ok(LfsPointer {
            version,
            hash_method,
            oid,
            size,
            extensions,
        })
    }

    /// Get the LFS version URL
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get the hash method (e.g., "sha256")
    pub fn hash_method(&self) -> &str {
        &self.hash_method
    }

    /// Get the object ID (SHA256 hash in hex)
    pub fn oid(&self) -> &str {
        &self.oid
    }

    /// Get the file size in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get extension fields (spec compliance)
    pub fn extensions(&self) -> &HashMap<String, String> {
        &self.extensions
    }

    /// Convert LFS OID (SHA256) to MerkleHash format for XET
    #[cfg(feature = "xet-storage")]
    pub fn to_merkle_hash(&self) -> Git2DBResult<merklehash::MerkleHash> {
        merklehash::MerkleHash::from_hex(&self.oid).map_err(|e| {
            Git2DBError::lfs(
                LfsErrorKind::HashConversion,
                format!("Failed to convert SHA256 to MerkleHash: {}", e),
            )
        })
    }
}

impl std::fmt::Display for LfsPointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "version {}\noid {}:{}\nsize {}\n",
            self.version, self.hash_method, self.oid, self.size
        )
    }
}

/// Check if content is a Git LFS pointer file
///
/// Supports both standard git-lfs and legacy hawser version URLs.
///
/// # Example
///
/// ```rust,ignore
/// if git2db::lfs::is_lfs_pointer(&content) {
///     // Handle as LFS pointer
/// }
/// ```
pub fn is_lfs_pointer(content: &str) -> bool {
    content.starts_with("version https://git-lfs.github.com/spec/v1")
        || content.starts_with("version https://hawser.github.com/spec/v1")
}

/// Extension trait for XetStorage - adds LFS capabilities
///
/// This allows using `xet.smudge_lfs(&pointer)` syntax for LFS operations.
#[cfg(feature = "xet-storage")]
#[async_trait]
pub trait LfsSmudge {
    /// Smudge LFS pointer to memory
    async fn smudge_lfs(&self, pointer: &LfsPointer) -> Git2DBResult<Vec<u8>>;

    /// Smudge LFS pointer directly to file
    async fn smudge_lfs_to_file(&self, pointer: &LfsPointer, path: &Path) -> Git2DBResult<()>;

    /// Smudge LFS pointer with size and SHA256 validation
    ///
    /// This provides defense-in-depth validation against potential XET CAS bugs.
    async fn smudge_lfs_validated(&self, pointer: &LfsPointer) -> Git2DBResult<Vec<u8>>;
}

#[cfg(feature = "xet-storage")]
#[async_trait]
impl LfsSmudge for XetStorage {
    async fn smudge_lfs(&self, pointer: &LfsPointer) -> Git2DBResult<Vec<u8>> {
        let hash = pointer.to_merkle_hash()?;
        self.smudge_from_hash(&hash).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::SmudgeFailed, format!("XET smudge failed: {}", e))
        })
    }

    async fn smudge_lfs_to_file(&self, pointer: &LfsPointer, path: &Path) -> Git2DBResult<()> {
        let hash = pointer.to_merkle_hash()?;
        self.smudge_from_hash_to_file(&hash, path).await.map_err(|e| {
            Git2DBError::lfs(
                LfsErrorKind::SmudgeFailed,
                format!("XET smudge to file failed: {}", e),
            )
        })
    }

    async fn smudge_lfs_validated(&self, pointer: &LfsPointer) -> Git2DBResult<Vec<u8>> {
        let content = self.smudge_lfs(pointer).await?;

        // Validate size
        if content.len() as u64 != pointer.size() {
            return Err(Git2DBError::lfs(
                LfsErrorKind::SizeMismatch,
                format!(
                    "Size mismatch: expected {} bytes, got {} bytes",
                    pointer.size(),
                    content.len()
                ),
            ));
        }

        // Validate SHA256
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let computed_hash = format!("{:x}", hasher.finalize());

        if computed_hash != pointer.oid() {
            return Err(Git2DBError::lfs(
                LfsErrorKind::HashMismatch,
                format!(
                    "SHA256 mismatch: expected {}, got {}",
                    pointer.oid(),
                    computed_hash
                ),
            ));
        }

        debug!(
            "Validated LFS content: {} bytes, SHA256: {}...",
            pointer.size(),
            &pointer.oid()[..8]
        );

        Ok(content)
    }
}

/// Processing statistics for batch operations
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    /// Total files found
    pub total_files: usize,
    /// Files successfully processed
    pub processed: usize,
    /// Files skipped (already smudged)
    pub skipped: usize,
    /// Files that failed to process
    pub failed: usize,
    /// Total bytes processed
    pub bytes_processed: u64,
}

/// Complete LFS storage operations
///
/// This struct provides all LFS functionality in one place, delegating
/// to XetStorage for the actual smudge/clean operations.
#[cfg(feature = "xet-storage")]
#[derive(Clone)]
pub struct LfsStorage {
    xet: Arc<XetStorage>,
}

#[cfg(feature = "xet-storage")]
impl LfsStorage {
    /// Create new LFS storage with XET backend
    pub async fn new(config: &XetConfig) -> Git2DBResult<Self> {
        let xet = XetStorage::new(config).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to create XET storage: {}", e))
        })?;
        Ok(Self { xet: Arc::new(xet) })
    }

    // === Pointer Detection ===

    /// Check if content is an LFS pointer
    pub fn is_lfs_pointer(&self, content: &str) -> bool {
        is_lfs_pointer(content)
    }

    /// Check if content is an XET pointer
    pub fn is_xet_pointer(&self, content: &str) -> bool {
        self.xet.is_pointer(content)
    }

    /// Check if content is any type of pointer (LFS or XET)
    pub fn is_pointer(&self, content: &str) -> bool {
        self.is_lfs_pointer(content) || self.is_xet_pointer(content)
    }

    /// Parse LFS pointer from content
    pub fn parse_lfs_pointer(&self, content: &str) -> Git2DBResult<LfsPointer> {
        LfsPointer::parse(content)
    }

    // === Core Operations ===

    /// Smudge LFS pointer to memory
    pub async fn smudge_lfs_pointer(&self, pointer: &LfsPointer) -> Git2DBResult<Vec<u8>> {
        self.xet.smudge_lfs(pointer).await
    }

    /// Smudge LFS pointer with validation
    pub async fn smudge_lfs_pointer_validated(&self, pointer: &LfsPointer) -> Git2DBResult<Vec<u8>> {
        self.xet.smudge_lfs_validated(pointer).await
    }

    /// Upload file and return XET pointer
    pub async fn clean_file(&self, path: &Path) -> Git2DBResult<String> {
        self.xet.clean_file(path).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("XET clean_file failed: {}", e))
        })
    }

    /// Upload data from memory and return XET pointer
    pub async fn clean_bytes(&self, data: &[u8]) -> Git2DBResult<String> {
        self.xet.clean_bytes(data).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("XET clean_bytes failed: {}", e))
        })
    }

    /// Download XET pointer to file
    pub async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Git2DBResult<()> {
        self.xet.smudge_file(pointer, output_path).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::SmudgeFailed, format!("XET smudge_file failed: {}", e))
        })
    }

    /// Download XET pointer to memory
    pub async fn smudge_bytes(&self, pointer: &str) -> Git2DBResult<Vec<u8>> {
        self.xet.smudge_bytes(pointer).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::SmudgeFailed, format!("XET smudge_bytes failed: {}", e))
        })
    }

    // === Universal File Loading ===

    /// Load file with automatic LFS/XET pointer detection and smudging
    ///
    /// This implements LFS spec-compliant pointer detection:
    /// 1. Checks file size via metadata (pointers MUST be < 1024 bytes)
    /// 2. Reads first 100 bytes for pointer detection (per LFS spec)
    /// 3. If pointer detected, reads rest and smudges
    /// 4. If not pointer, reads full file efficiently
    ///
    /// Use this for files outside git operations (git-xet-filter handles files in git repos).
    pub async fn load_file(&self, file_path: &Path) -> Git2DBResult<Vec<u8>> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        // 1. Check metadata FIRST (instant, no file content I/O)
        // LFS spec: "Pointer files must be less than 1024 bytes in size"
        let metadata = tokio::fs::metadata(file_path).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to get file metadata: {}", e))
        })?;
        let file_size = metadata.len();

        // 2. Large files (>= 1024 bytes) cannot be pointers - read directly
        if file_size >= 1024 {
            return tokio::fs::read(file_path).await.map_err(|e| {
                Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to read file: {}", e))
            });
        }

        // 3. Small file - might be a pointer
        // LFS spec: "Read 100 bytes. If the content is ASCII and matches the pointer file format"
        let mut file = tokio::fs::File::open(file_path).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to open file: {}", e))
        })?;

        let mut header = vec![0u8; 100];
        let n = file.read(&mut header).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to read header: {}", e))
        })?;
        header.truncate(n);

        // 4. Check for pointer markers (LFS and XET pointers are UTF-8 text)
        if let Ok(text) = String::from_utf8(header.clone()) {
            if text.starts_with("version https://git-lfs")
                || text.starts_with("version https://hawser")
                || text.starts_with("# xet version")
            {
                // It's a pointer - read the rest (file is < 1024 bytes total)
                let mut rest = Vec::new();
                file.read_to_end(&mut rest).await.map_err(|e| {
                    Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to read rest: {}", e))
                })?;

                let full_content = [&header[..n], &rest[..]].concat();
                let full_text = String::from_utf8(full_content).map_err(|_| {
                    Git2DBError::lfs(LfsErrorKind::InvalidPointer, "Pointer file is not valid UTF-8")
                })?;

                // Parse and smudge the appropriate pointer type
                if self.is_lfs_pointer(&full_text) {
                    let lfs_pointer = self.parse_lfs_pointer(&full_text)?;
                    return self.smudge_lfs_pointer_validated(&lfs_pointer).await;
                } else if self.is_xet_pointer(&full_text) {
                    return self.smudge_bytes(&full_text).await;
                }
            }
        }

        // 5. Not a pointer - seek back to start and read full file
        file.seek(std::io::SeekFrom::Start(0)).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to seek: {}", e))
        })?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to read file: {}", e))
        })?;

        Ok(contents)
    }

    // === Worktree Operations ===

    /// Scan directory for LFS files using git2 tree traversal
    pub async fn scan_lfs_files(&self, directory: &Path) -> Git2DBResult<Vec<PathBuf>> {
        let directory = directory.to_path_buf();
        let this = self.clone();

        tokio::task::spawn_blocking(move || this.scan_lfs_files_sync(&directory))
            .await
            .map_err(|e| Git2DBError::lfs(LfsErrorKind::IoError, format!("Task join failed: {}", e)))?
    }

    fn scan_lfs_files_sync(&self, directory: &Path) -> Git2DBResult<Vec<PathBuf>> {
        use git2::{ObjectType, TreeWalkMode, TreeWalkResult};

        // Find the git repository for this directory
        let repo = git2::Repository::discover(directory).map_err(|e| {
            Git2DBError::lfs(
                LfsErrorKind::NotInRepository,
                format!("Directory is not part of a git repository: {}", e),
            )
        })?;

        let repo_workdir = repo.workdir().ok_or_else(|| {
            Git2DBError::lfs(LfsErrorKind::NotInRepository, "Repository has no working directory")
        })?;

        // Get the current HEAD commit and its tree
        let head = repo.head().map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to get HEAD: {}", e))
        })?;
        let commit = head.peel_to_commit().map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to get HEAD commit: {}", e))
        })?;
        let tree = commit.tree().map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to get commit tree: {}", e))
        })?;

        // Convert directory to relative path if needed
        let scan_prefix = if directory.starts_with(repo_workdir) {
            directory
                .strip_prefix(repo_workdir)
                .map_err(|_| {
                    Git2DBError::lfs(LfsErrorKind::IoError, "Failed to get relative path")
                })?
                .to_string_lossy()
                .into_owned()
        } else {
            String::new() // Scan entire repo
        };

        // Collect LFS candidates using git2's tree walk
        let mut lfs_candidates = Vec::new();
        let this = self.clone();

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
                        if this.is_lfs_pointer(content) {
                            let full_path = repo_workdir.join(&entry_path);
                            lfs_candidates.push(full_path);
                        }
                    }
                }
            }

            TreeWalkResult::Ok
        })
        .map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Tree walk failed: {}", e))
        })?;

        // Filter candidates using git status to respect .gitignore
        let filtered = self.filter_by_git_status(&repo, lfs_candidates)?;

        Ok(filtered)
    }

    fn filter_by_git_status(
        &self,
        repo: &git2::Repository,
        candidates: Vec<PathBuf>,
    ) -> Git2DBResult<Vec<PathBuf>> {
        let repo_workdir = repo.workdir().ok_or_else(|| {
            Git2DBError::lfs(LfsErrorKind::NotInRepository, "Repository has no working directory")
        })?;

        let mut filtered = Vec::new();
        let total = candidates.len();

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
                        filtered.push(candidate);
                    }
                }
            }
        }

        debug!("Filtered {} LFS candidates to {} files", total, filtered.len());
        Ok(filtered)
    }

    /// Process all LFS files in a worktree by smudging them in place
    pub async fn process_worktree(&self, worktree_path: &Path) -> Git2DBResult<ProcessingStats> {
        if !worktree_path.is_dir() {
            return Err(Git2DBError::lfs(
                LfsErrorKind::IoError,
                format!("Path is not a directory: {}", worktree_path.display()),
            ));
        }

        info!("Scanning worktree for LFS files: {}", worktree_path.display());
        let lfs_files = self.scan_lfs_files(worktree_path).await?;

        if lfs_files.is_empty() {
            debug!("No LFS files found in worktree");
            return Ok(ProcessingStats::default());
        }

        info!("Found {} LFS files to process", lfs_files.len());
        let mut stats = ProcessingStats {
            total_files: lfs_files.len(),
            ..Default::default()
        };

        for lfs_file in &lfs_files {
            match self.smudge_file_in_place(lfs_file).await {
                Ok(bytes) => {
                    stats.processed += 1;
                    stats.bytes_processed += bytes;
                    debug!("Processed: {}", lfs_file.display());
                }
                Err(e) => {
                    stats.failed += 1;
                    warn!("Failed to process {}: {}", lfs_file.display(), e);
                }
            }
        }

        info!(
            "Processed {}/{} LFS files ({} bytes)",
            stats.processed, stats.total_files, stats.bytes_processed
        );
        Ok(stats)
    }

    /// Batch process LFS files with concurrency control
    pub async fn batch_process(
        &self,
        worktree_path: &Path,
        concurrency: usize,
    ) -> Git2DBResult<ProcessingStats> {
        let lfs_files = self.scan_lfs_files(worktree_path).await?;

        if lfs_files.is_empty() {
            debug!("No LFS files found in worktree");
            return Ok(ProcessingStats::default());
        }

        info!(
            "Found {} LFS files to process with concurrency {}",
            lfs_files.len(),
            concurrency
        );

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let mut tasks = Vec::new();

        for lfs_file in lfs_files {
            let semaphore = Arc::clone(&semaphore);
            let storage = self.clone();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                storage
                    .smudge_file_in_place(&lfs_file)
                    .await
                    .map(|bytes| (lfs_file, bytes))
            });
            tasks.push(task);
        }

        let mut stats = ProcessingStats {
            total_files: tasks.len(),
            ..Default::default()
        };

        for task in tasks {
            match task.await {
                Ok(Ok((path, bytes))) => {
                    stats.processed += 1;
                    stats.bytes_processed += bytes;
                    debug!("Processed: {}", path.display());
                }
                Ok(Err(e)) => {
                    stats.failed += 1;
                    warn!("Processing failed: {}", e);
                }
                Err(e) => {
                    stats.failed += 1;
                    warn!("Task failed: {}", e);
                }
            }
        }

        info!(
            "Batch processed {}/{} LFS files ({} bytes)",
            stats.processed, stats.total_files, stats.bytes_processed
        );
        Ok(stats)
    }

    // === Internal Helpers ===

    async fn smudge_file_in_place(&self, file_path: &Path) -> Git2DBResult<u64> {
        // Check if already smudged
        if self.is_already_smudged(file_path).await? {
            debug!("Skipping already smudged file: {}", file_path.display());
            return Ok(0);
        }

        // Read the LFS pointer content
        let pointer_content = tokio::fs::read_to_string(file_path).await.map_err(|e| {
            Git2DBError::lfs(
                LfsErrorKind::IoError,
                format!("Failed to read LFS pointer: {}", e),
            )
        })?;

        if !self.is_lfs_pointer(&pointer_content) {
            return Err(Git2DBError::lfs(
                LfsErrorKind::InvalidPointer,
                format!("File is not an LFS pointer: {}", file_path.display()),
            ));
        }

        // Parse the LFS pointer
        let lfs_pointer = self.parse_lfs_pointer(&pointer_content)?;

        debug!(
            "Smudging LFS file: {} ({} bytes, SHA256: {}...)",
            file_path.display(),
            lfs_pointer.size(),
            &lfs_pointer.oid()[..8]
        );

        // Retrieve and validate content via XET
        let actual_content = self.smudge_lfs_pointer_validated(&lfs_pointer).await?;
        let bytes_written = actual_content.len() as u64;

        // Write to temporary file first
        let temp_file =
            tempfile::NamedTempFile::new_in(file_path.parent().unwrap_or(Path::new("."))).map_err(
                |e| {
                    Git2DBError::lfs(
                        LfsErrorKind::IoError,
                        format!("Failed to create temp file: {}", e),
                    )
                },
            )?;

        tokio::fs::write(temp_file.path(), &actual_content)
            .await
            .map_err(|e| {
                Git2DBError::lfs(
                    LfsErrorKind::IoError,
                    format!("Failed to write temp file: {}", e),
                )
            })?;

        // Atomically replace the original file
        let temp_path = temp_file.into_temp_path();
        temp_path.persist(file_path).map_err(|e| {
            Git2DBError::lfs(
                LfsErrorKind::IoError,
                format!("Failed to replace file atomically: {}", e),
            )
        })?;

        Ok(bytes_written)
    }

    async fn is_already_smudged(&self, file_path: &Path) -> Git2DBResult<bool> {
        use tokio::io::AsyncReadExt;

        // Read first 200 bytes to check for pointer markers
        let mut file = match tokio::fs::File::open(file_path).await {
            Ok(f) => f,
            Err(_) => return Ok(false), // File doesn't exist = not smudged
        };

        let mut buffer = vec![0u8; 200];
        let n = file.read(&mut buffer).await.map_err(|e| {
            Git2DBError::lfs(LfsErrorKind::IoError, format!("Failed to read header: {}", e))
        })?;

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
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfs_pointer_detection() {
        // Standard git-lfs pointer
        assert!(is_lfs_pointer(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n"
        ));

        // Legacy hawser pointer
        assert!(is_lfs_pointer(
            "version https://hawser.github.com/spec/v1\noid sha256:abc\nsize 100\n"
        ));

        // Not an LFS pointer
        assert!(!is_lfs_pointer("This is regular content"));
        assert!(!is_lfs_pointer("{\"hash\": \"abc123\"}")); // XET pointer
    }

    #[test]
    fn test_lfs_pointer_parsing() {
        let content = r#"version https://git-lfs.github.com/spec/v1
oid sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
size 1024
"#;

        let pointer = LfsPointer::parse(content).unwrap();

        assert_eq!(pointer.version(), "https://git-lfs.github.com/spec/v1");
        assert_eq!(pointer.hash_method(), "sha256");
        assert_eq!(
            pointer.oid(),
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        );
        assert_eq!(pointer.size(), 1024);
    }

    #[test]
    fn test_lfs_pointer_extensions() {
        let content = r#"version https://git-lfs.github.com/spec/v1
oid sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
size 1024
ext-0-custom value1
ext-1-another value2
"#;

        let pointer = LfsPointer::parse(content).unwrap();

        assert_eq!(pointer.extensions().get("ext-0-custom"), Some(&"value1".to_string()));
        assert_eq!(pointer.extensions().get("ext-1-another"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_lfs_pointer_invalid_oid() {
        // Too short
        let short_oid = r#"version https://git-lfs.github.com/spec/v1
oid sha256:abc123
size 1024
"#;
        assert!(LfsPointer::parse(short_oid).is_err());

        // Invalid characters
        let invalid_chars = r#"version https://git-lfs.github.com/spec/v1
oid sha256:gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
size 1024
"#;
        assert!(LfsPointer::parse(invalid_chars).is_err());
    }

    #[test]
    fn test_lfs_pointer_missing_fields() {
        // Missing version
        assert!(LfsPointer::parse("oid sha256:abc123\nsize 100\n").is_err());

        // Missing oid
        assert!(LfsPointer::parse("version https://git-lfs.github.com/spec/v1\nsize 100\n").is_err());

        // Missing size
        let valid_oid = "a".repeat(64);
        assert!(LfsPointer::parse(&format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{}\n",
            valid_oid
        ))
        .is_err());
    }

    #[test]
    fn test_lfs_pointer_display() {
        let content = r#"version https://git-lfs.github.com/spec/v1
oid sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
size 1024
"#;

        let pointer = LfsPointer::parse(content).unwrap();
        let display = pointer.to_string();

        assert!(display.contains("version https://git-lfs.github.com/spec/v1"));
        assert!(display.contains("oid sha256:"));
        assert!(display.contains("size 1024"));
    }
}
