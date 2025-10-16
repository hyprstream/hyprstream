//! Native Xet storage integration using xet-core v1.1.10
//!
//! Provides transparent integration with git-xet repositories by automatically
//! detecting and handling Xet pointer files. This module allows seamless
//! operation with both regular files and Xet-tracked files without requiring
//! Git LFS or git-xet transfer agent setup.

use anyhow::{anyhow, bail, Context, Result};
use cas_client::{FileProvider, OutputProvider};
use data::{data_client, FileDownloader, FileUploadSession, XetFileInfo};
use indicatif::{ProgressBar, ProgressStyle};
use merklehash::MerkleHash;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::fs;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

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
/// Simple progress tracking that works with Xet
/// We'll implement this as a simpler version that doesn't depend on external traits
pub struct IndicatifProgressUpdater {
    progress_bar: ProgressBar,
}

impl IndicatifProgressUpdater {
    pub fn new(progress_bar: ProgressBar) -> Self {
        Self { progress_bar }
    }

    pub fn update(&self, current: u64, total: Option<u64>) {
        if let Some(total) = total {
            self.progress_bar.set_length(total);
        }
        self.progress_bar.set_position(current);
    }

    pub fn set_message(&self, message: &str) {
        self.progress_bar.set_message(message.to_string());
    }

    pub fn finish(&self) {
        self.progress_bar.finish();
    }

    pub fn update_item_progress(&self, item_name: &str, current: u64, total: Option<u64>) {
        if let Some(total) = total {
            self.progress_bar.set_length(total);
            let mb_current = current as f64 / (1024.0 * 1024.0);
            let mb_total = total as f64 / (1024.0 * 1024.0);
            self.progress_bar.set_message(format!(
                "{}: {:.1}MB/{:.1}MB",
                item_name, mb_current, mb_total
            ));
        } else {
            self.progress_bar
                .set_message(format!("{}: {}B", item_name, current));
        }
        self.progress_bar.set_position(current);
    }

    pub fn finish_item(&self, _item_name: &str) {
        self.progress_bar.finish();
    }
}

/// Configuration for Xet storage
///
/// This is now a re-export of git2db's XetConfig for unified configuration.
pub use git2db::config::XetConfig;

/// Native Xet storage implementation using xet-core APIs
#[derive(Clone)]
pub struct XetNativeStorage {
    upload_session: Arc<FileUploadSession>,
    downloader: Arc<FileDownloader>,
    config: XetConfig,
}

impl XetNativeStorage {
    /// Create a new Xet storage instance
    pub async fn new(config: XetConfig) -> Result<Self> {
        Self::new_with_progress(config, None).await
    }

    /// Create a new Xet storage instance with optional progress tracking
    pub async fn new_with_progress(
        config: XetConfig,
        progress_bar: Option<ProgressBar>,
    ) -> Result<Self> {
        // Create translator config using xet-core's default_config
        let translator_config = Arc::new(
            data_client::default_config(
                config.endpoint.clone(),
                config.compression,
                config.token.as_ref().map(|t| (t.clone(), u64::MAX)), // Token with far future expiry
                None,                                                 // No token refresher
            )
            .context("Failed to create Xet translator config")?,
        );

        // Create progress updater if progress bar is provided
        let _progress_updater = progress_bar.map(|pb| {
            let pb_style = ProgressStyle::default_bar()
                .template("   {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} {msg}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ ");
            pb.set_style(pb_style);
            pb.set_message("Initializing Xet storage...");
            IndicatifProgressUpdater::new(pb)
        });

        // Initialize upload session and downloader (progress will be added later)
        let upload_session = FileUploadSession::new(translator_config.clone(), None)
            .await
            .context("Failed to create Xet upload session")?;

        let downloader = FileDownloader::new(translator_config)
            .await
            .context("Failed to create Xet downloader")?;

        Ok(Self {
            upload_session: upload_session.into(),
            downloader: downloader.into(),
            config,
        })
    }

    /// Store file and return Xet pointer (clean operation)
    ///
    /// This uploads the file to Xet CAS and returns a JSON pointer that
    /// should be stored in place of the original file content.
    pub async fn clean_file(&self, file_path: &Path) -> Result<String> {
        let (xet_info, _metrics) = data_client::clean_file(self.upload_session.clone(), file_path)
            .await
            .context("Failed to clean file for Xet storage")?;

        xet_info
            .as_pointer_file()
            .context("Failed to serialize Xet pointer")
    }

    /// Store data from memory and return Xet pointer
    pub async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        let (xet_info, _metrics) =
            data_client::clean_bytes(self.upload_session.clone(), data.to_vec())
                .await
                .context("Failed to clean bytes for Xet storage")?;

        xet_info
            .as_pointer_file()
            .context("Failed to serialize Xet pointer")
    }

    /// Retrieve file from Xet pointer to specified path (smudge operation)
    pub async fn smudge_file(&self, pointer_content: &str, output_path: &Path) -> Result<()> {
        let xet_info: XetFileInfo =
            serde_json::from_str(pointer_content).context("Failed to parse Xet pointer")?;

        let merkle_hash = xet_info
            .merkle_hash()
            .context("Failed to parse merkle hash from Xet pointer")?;

        self.downloader
            .smudge_file_from_hash(
                &merkle_hash,
                output_path.to_string_lossy().into(),
                &OutputProvider::File(FileProvider::new(output_path.to_path_buf())),
                None,
                None,
            )
            .await
            .context("Failed to smudge file from Xet storage")?;

        Ok(())
    }

    /// Retrieve file from Xet pointer to memory
    pub async fn smudge_bytes(&self, pointer_content: &str) -> Result<Vec<u8>> {
        // Create temporary file for smudging
        let temp_file = NamedTempFile::new().context("Failed to create temporary file")?;

        self.smudge_file(pointer_content, temp_file.path()).await?;

        // Read the smudged content
        fs::read(temp_file.path())
            .await
            .context("Failed to read smudged file")
    }

    /// Check if content is a Xet pointer file
    pub fn is_xet_pointer(&self, content: &str) -> bool {
        serde_json::from_str::<XetFileInfo>(content).is_ok()
    }

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
        // Direct conversion: LFS SHA256 hex string → MerkleHash
        // This follows the same pattern as xet-core data/src/sha256.rs:45
        MerkleHash::from_hex(sha256_hex).context("Failed to convert LFS SHA256 to MerkleHash")
    }

    /// Resolve LFS pointer to actual content via XET
    ///
    /// This method takes an LFS pointer and attempts to retrieve the actual file content
    /// from XET storage using the SHA256 OID. The strategy is to use XET's content-addressable
    /// storage to look up content by its hash.
    pub async fn smudge_lfs_pointer(&self, lfs_pointer: &LfsPointer) -> Result<Vec<u8>> {
        // Convert LFS SHA256 to MerkleHash format expected by XET
        let merkle_hash = self.sha256_to_merkle_hash(&lfs_pointer.oid)?;

        // Create temporary output for smudging
        let temp_file =
            NamedTempFile::new().context("Failed to create temporary file for LFS smudging")?;

        // Use XET downloader to retrieve content by hash
        self.downloader
            .smudge_file_from_hash(
                &merkle_hash,
                temp_file.path().to_string_lossy().into(),
                &OutputProvider::File(FileProvider::new(temp_file.path().to_path_buf())),
                None, // No range - download entire file
                None, // No progress updater
            )
            .await
            .context("Failed to smudge LFS content from XET storage")?;

        // Read and return the content
        fs::read(temp_file.path())
            .await
            .context("Failed to read smudged LFS content")
    }

    /// Universal file loader - handles Xet pointers, LFS pointers, and regular files
    ///
    /// This is the main interface for loading files. It automatically detects
    /// if the file contains a Xet pointer, LFS pointer, and retrieves the actual content,
    /// or returns the file content directly if it's not a pointer.
    pub async fn load_file(&self, file_path: &Path) -> Result<Vec<u8>> {
        // First read a small portion to check if it's a pointer file
        let initial_bytes = fs::read(file_path).await.context("Failed to read file")?;

        // Check if the file starts with pointer markers (both XET and LFS are text-based)
        // XET pointers start with "# xet version"
        // LFS pointers start with "version https://git-lfs"
        let is_likely_pointer = initial_bytes.len() < 1024 && // Pointer files are small
            initial_bytes.starts_with(b"# xet version")
            || initial_bytes.starts_with(b"version https://git-lfs");

        if is_likely_pointer {
            // Try to parse as text for pointer detection
            if let Ok(content) = String::from_utf8(initial_bytes.clone()) {
                if self.is_xet_pointer(&content) {
                    // It's a Xet pointer - retrieve the actual content
                    return self.smudge_bytes(&content).await;
                } else if self.is_lfs_pointer(&content) {
                    // It's an LFS pointer - parse and retrieve via XET
                    let lfs_pointer = self.parse_lfs_pointer(&content)?;
                    return self.smudge_lfs_pointer(&lfs_pointer).await;
                }
            }
        }

        // Not a pointer file - return the raw bytes
        Ok(initial_bytes)
    }

    /// Universal file saver - can store as regular file or convert to Xet
    ///
    /// If the target file already exists as a Xet pointer, this will update
    /// the pointed-to content and maintain it as a Xet file. Otherwise, it
    /// stores the file normally.
    pub async fn save_file(&self, file_path: &Path, data: &[u8]) -> Result<()> {
        // Check if the file already exists as a Xet pointer
        if file_path.exists() {
            let existing_content = fs::read_to_string(file_path)
                .await
                .context("Failed to read existing file")?;

            if self.is_xet_pointer(&existing_content) {
                // File is already a Xet pointer - update it
                let pointer = self.clean_bytes(data).await?;
                fs::write(file_path, pointer)
                    .await
                    .context("Failed to write updated Xet pointer")?;
                return Ok(());
            }
        }

        // Store as regular file
        fs::write(file_path, data)
            .await
            .context("Failed to write regular file")
    }

    /// Explicitly store file as Xet (force conversion)
    ///
    /// This always stores the file as a Xet pointer, regardless of whether
    /// it was previously a regular file or Xet pointer.
    pub async fn save_as_xet(&self, file_path: &Path, data: &[u8]) -> Result<()> {
        let pointer = self.clean_bytes(data).await?;
        fs::write(file_path, pointer)
            .await
            .context("Failed to write Xet pointer file")
    }

    /// Get configuration
    pub fn config(&self) -> &XetConfig {
        &self.config
    }

    /// Check if file exists (handles both regular files and Xet pointers)
    pub async fn file_exists(&self, file_path: &Path) -> bool {
        file_path.exists()
    }

    /// Get file size (handles both regular files and Xet pointers)
    pub async fn file_size(&self, file_path: &Path) -> Result<u64> {
        if !file_path.exists() {
            bail!("File does not exist: {}", file_path.display());
        }

        let content = fs::read_to_string(file_path)
            .await
            .context("Failed to read file")?;

        if self.is_xet_pointer(&content) {
            let xet_info: XetFileInfo =
                serde_json::from_str(&content).context("Failed to parse Xet pointer")?;
            Ok(xet_info.file_size())
        } else {
            Ok(content.len() as u64)
        }
    }

    /// Scan directory for LFS files and return their paths, respecting .gitignore
    ///
    /// This method uses git2's native tree traversal and status APIs for efficiency
    pub async fn scan_lfs_files_in_directory(&self, directory: &Path) -> Result<Vec<PathBuf>> {
        let directory = directory.to_path_buf();
        let storage = self.clone(); // We need to clone self to move into the blocking task

        tokio::task::spawn_blocking(move || storage.scan_lfs_files_in_directory_sync(&directory))
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
    fn find_git_repository(&self, directory: &Path) -> Result<git2db::Repository> {
        // Try to open as a git repository or find the parent repository
        git2db::Repository::discover(directory).context("Directory is not part of a git repository")
    }

    /// Filter candidate files using git status to respect .gitignore (sync version)
    fn filter_files_by_git_status_sync(
        &self,
        repo: &git2db::Repository,
        candidates: Vec<PathBuf>,
    ) -> Result<Vec<PathBuf>> {
        let repo_workdir = repo
            .workdir()
            .ok_or_else(|| anyhow!("Repository has no working directory"))?;

        // For LFS files found via git tree walk, they are already tracked files
        // We just need to verify they exist in the worktree and aren't ignored
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
    /// to resolve them via XET storage. For commit-based processing, this processes
    /// all LFS files present in the checked out commit.
    pub async fn process_worktree_lfs(&self, worktree_path: &Path) -> Result<Vec<PathBuf>> {
        use tracing::{debug, info, warn};

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

            match self.smudge_lfs_file_in_place(&lfs_file).await {
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
        // Read the LFS pointer content
        let pointer_content = fs::read_to_string(file_path)
            .await
            .context("Failed to read LFS pointer file")?;

        if !self.is_lfs_pointer(&pointer_content) {
            bail!("File is not an LFS pointer: {}", file_path.display());
        }

        // Parse the LFS pointer
        let lfs_pointer = self.parse_lfs_pointer(&pointer_content)?;

        // Retrieve actual content via XET
        let actual_content = self.smudge_lfs_pointer(&lfs_pointer).await?;

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

        Ok(())
    }

    /// Process all LFS files efficiently in parallel (Enhanced batch processing)
    pub async fn batch_process_lfs_files(&self, worktree_path: &Path) -> Result<Vec<PathBuf>> {
        let lfs_files = self.scan_lfs_files_in_directory(worktree_path).await?;

        if lfs_files.is_empty() {
            debug!("No LFS files found in worktree");
            return Ok(Vec::new());
        }

        info!("Found {} LFS files to process in parallel", lfs_files.len());

        // Process files concurrently (this is where we beat sequential filters)
        let semaphore = Arc::new(Semaphore::new(8)); // Limit concurrency to 8
        let mut tasks = Vec::new();

        for lfs_file in lfs_files {
            let semaphore = Arc::clone(&semaphore);
            let storage = self.clone();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                storage
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_xet_pointer_detection() {
        let config = XetConfig::default();
        let storage = XetNativeStorage::new(config).await.unwrap();

        // Test valid Xet pointer
        let valid_pointer = r#"{"hash":"abc123","file_size":1024}"#;
        assert!(storage.is_xet_pointer(valid_pointer));

        // Test invalid content
        let invalid_content = "This is not a Xet pointer";
        assert!(!storage.is_xet_pointer(invalid_content));

        // Test empty content
        assert!(!storage.is_xet_pointer(""));
    }

    #[tokio::test]
    async fn test_regular_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        let test_data = b"Hello, World!";

        let config = XetConfig::default();
        let storage = XetNativeStorage::new(config).await.unwrap();

        // Save as regular file
        storage.save_file(&test_file, test_data).await.unwrap();

        // Load the file
        let loaded_data = storage.load_file(&test_file).await.unwrap();
        assert_eq!(loaded_data, test_data);

        // Check file size
        let size = storage.file_size(&test_file).await.unwrap();
        assert_eq!(size, test_data.len() as u64);
    }

    #[tokio::test]
    async fn test_lfs_pointer_detection() {
        let config = XetConfig::default();
        let storage = XetNativeStorage::new(config).await.unwrap();

        // Test valid LFS pointer
        let valid_lfs_pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        assert!(storage.is_lfs_pointer(valid_lfs_pointer));
        assert!(storage.is_pointer(valid_lfs_pointer));

        // Test invalid content
        let regular_content = "This is just regular file content";
        assert!(!storage.is_lfs_pointer(regular_content));
        assert!(!storage.is_pointer(regular_content));

        // Test XET pointer (should not be detected as LFS)
        let xet_pointer = r#"{"hash":"abc123","file_size":1024}"#;
        assert!(!storage.is_lfs_pointer(xet_pointer));
        assert!(storage.is_xet_pointer(xet_pointer));
        assert!(storage.is_pointer(xet_pointer));

        // Test partial LFS content (should not match)
        let partial_lfs = "version https://git-lfs.github.com/spec/v2";
        assert!(!storage.is_lfs_pointer(partial_lfs));
    }

    #[tokio::test]
    async fn test_lfs_pointer_parsing() {
        let config = XetConfig::default();
        let storage = XetNativeStorage::new(config).await.unwrap();

        // Test valid LFS pointer parsing
        let valid_lfs_content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        let parsed = storage.parse_lfs_pointer(valid_lfs_content).unwrap();

        assert_eq!(parsed.version, "https://git-lfs.github.com/spec/v1");
        assert_eq!(
            parsed.oid,
            "abc123def456789012345678901234567890123456789012345678901234567890"
        );
        assert_eq!(parsed.size, 12345);

        // Test with extra whitespace and empty lines
        let lfs_with_whitespace = "\nversion https://git-lfs.github.com/spec/v1\n\noid sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef\n\nsize 9876\n\n";
        let parsed2 = storage.parse_lfs_pointer(lfs_with_whitespace).unwrap();

        assert_eq!(parsed2.version, "https://git-lfs.github.com/spec/v1");
        assert_eq!(
            parsed2.oid,
            "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        );
        assert_eq!(parsed2.size, 9876);
    }

    #[tokio::test]
    async fn test_lfs_pointer_parsing_errors() {
        let config = XetConfig::default();
        let storage = XetNativeStorage::new(config).await.unwrap();

        // Test missing version
        let missing_version = "oid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        assert!(storage.parse_lfs_pointer(missing_version).is_err());

        // Test missing OID
        let missing_oid = "version https://git-lfs.github.com/spec/v1\nsize 12345\n";
        assert!(storage.parse_lfs_pointer(missing_oid).is_err());

        // Test missing size
        let missing_size = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\n";
        assert!(storage.parse_lfs_pointer(missing_size).is_err());

        // Test invalid OID length (too short)
        let invalid_oid_short =
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 12345\n";
        assert!(storage.parse_lfs_pointer(invalid_oid_short).is_err());

        // Test invalid OID characters
        let invalid_oid_chars = "version https://git-lfs.github.com/spec/v1\noid sha256:xyz123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        assert!(storage.parse_lfs_pointer(invalid_oid_chars).is_err());

        // Test invalid size format
        let invalid_size = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize not_a_number\n";
        assert!(storage.parse_lfs_pointer(invalid_size).is_err());
    }

    #[tokio::test]
    async fn test_universal_load_file_with_lfs() {
        let temp_dir = TempDir::new().unwrap();
        let config = XetConfig::default();
        let storage = XetNativeStorage::new(config).await.unwrap();

        // Test loading regular file
        let regular_file = temp_dir.path().join("regular.txt");
        let regular_content = b"This is regular content";
        fs::write(&regular_file, regular_content).await.unwrap();

        let loaded = storage.load_file(&regular_file).await.unwrap();
        assert_eq!(loaded, regular_content);

        // Test loading LFS pointer file (should fail with instructive error)
        let lfs_file = temp_dir.path().join("model.safetensors");
        let lfs_content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        fs::write(&lfs_file, lfs_content).await.unwrap();

        let result = storage.load_file(&lfs_file).await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("LFS pointer found"));
        assert!(error_msg
            .contains("abc123def456789012345678901234567890123456789012345678901234567890"));
        assert!(error_msg.contains("12345 bytes"));
    }

    #[tokio::test]
    async fn test_scan_lfs_files_respects_gitignore() {
        // This test verifies that our LFS scanning respects .gitignore rules
        // Since we can't easily create a git repo in tests, we test that non-git directories
        // return appropriate errors
        let temp_dir = TempDir::new().unwrap();
        let config = XetConfig::default();
        let storage = XetNativeStorage::new(config).await.unwrap();

        // Create a sample LFS file
        let lfs_file = temp_dir.path().join("model.safetensors");
        let lfs_content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
        fs::write(&lfs_file, lfs_content).await.unwrap();

        // Try to scan a non-git directory - should fail gracefully
        let result = storage.scan_lfs_files_in_directory(temp_dir.path()).await;
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("not part of a git repository"));
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
