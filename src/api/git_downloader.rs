//! Git-based model downloader using git2 crate with integrated LFS/XET support via xet-core

use anyhow::{Result, bail};
use git2::{Repository, FetchOptions, build::RepoBuilder};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressStyle};
use crate::api::model_storage::ModelId;
use crate::storage::{XetNativeStorage, XetConfig};
use tokio::fs;
use sha2::{Sha256, Digest};

// Constants per LFS specification
const MAX_POINTER_FILE_SIZE: u64 = 1024;
const LFS_VERSION_SPEC_URL: &str = "https://git-lfs.github.com/spec/v1";
const LFS_VERSION_KEY: &str = "version";
const LFS_OID_KEY: &str = "oid";
const LFS_SIZE_KEY: &str = "size";
const LFS_SHA256_PREFIX: &str = "sha256:";
const COMMON_LFS_PATTERNS: &[&str] = &["*.safetensors", "*.bin", "*.pt", "*.pth", "*.onnx"];

/// LFS pointer file structure per Git LFS specification
#[derive(Debug, Clone)]
struct LfsPointer {
    version: String,
    oid: String,    // Full OID including hash method (e.g., "sha256:abc123...")
    size: u64,
}

/// Git-based model source using git2 with xet-core integration
pub struct GitModelSource {
    cache_dir: PathBuf,
    xet_storage: Option<Arc<XetNativeStorage>>,
}

impl GitModelSource {
    /// Create a new Git model source
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            xet_storage: None,
        }
    }

    /// Create a new Git model source with XET storage support
    pub fn with_xet_storage(cache_dir: PathBuf, xet_storage: Arc<XetNativeStorage>) -> Self {
        Self {
            cache_dir,
            xet_storage: Some(xet_storage),
        }
    }

    /// Initialize XET storage with default config
    pub async fn with_default_xet(cache_dir: PathBuf) -> Result<Self> {
        let xet_config = XetConfig::default();
        let xet_storage = Arc::new(XetNativeStorage::new(xet_config).await?);
        Ok(Self::with_xet_storage(cache_dir, xet_storage))
    }

    /// Create GitModelSource with XET support, falling back to basic git on failure
    pub async fn with_xet_fallback(cache_dir: PathBuf) -> Self {
        match Self::with_default_xet(cache_dir.clone()).await {
            Ok(source) => {
                tracing::debug!("Initialized GitModelSource with XET support");
                source
            }
            Err(e) => {
                tracing::warn!("Failed to initialize XET support, falling back to basic git: {}", e);
                Self::new(cache_dir)
            }
        }
    }
    
    /// Clone a model repository using git2 with progress tracking
    pub async fn clone_model(
        &self,
        repo_url: &str,
    ) -> Result<(ModelId, PathBuf)> {
        self.clone_model_with_progress(repo_url, true).await
    }

    /// Clone a model repository with optional progress tracking
    pub async fn clone_model_with_progress(
        &self,
        repo_url: &str,
        show_progress: bool,
    ) -> Result<(ModelId, PathBuf)> {
        let git_url = repo_url.to_string();

        // Always generate new model ID for clones
        let model_id = ModelId::new();
        let model_path = self.cache_dir.join(model_id.0.to_string());

        // Check if already cloned
        if model_path.join(".git").exists() {
            println!("📦 Model already cloned at: {}", model_path.display());
            return Ok((model_id, model_path));
        }

        println!("📥 Cloning model from: {}", git_url);

        // Create progress bar
        let progress_bar = if show_progress {
            let pb = ProgressBar::new(100);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("   {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                    .unwrap()
                    .progress_chars("█▉▊▋▌▍▎▏ ")
            );
            pb.set_message("Initializing...");
            Some(pb)
        } else {
            None
        };

        // Move git operations to blocking task to avoid Send issues
        let git_url_clone = git_url.clone();
        let model_path_clone = model_path.clone();
        let progress_bar_clone = progress_bar.clone();

        let result = tokio::task::spawn_blocking(move || {
            // Clone with git2
            let mut builder = RepoBuilder::new();

            // Configure fetch options to use system git config
            let mut fetch_opts = FetchOptions::new();

            // Set up progress and credential callbacks
            fetch_opts.remote_callbacks(Self::get_callbacks_with_progress(progress_bar_clone));
            builder.fetch_options(fetch_opts);

            // Clone the repository
            builder.clone(&git_url_clone, &model_path_clone)
        }).await?;

        // Finish progress bar
        if let Some(pb) = &progress_bar {
            match &result {
                Ok(_) => {
                    pb.set_message("Completed");
                    pb.finish_with_message("✅ Clone completed");
                },
                Err(_) => {
                    pb.abandon_with_message("❌ Clone failed");
                }
            }
        }

        match result {
            Ok(_) => {
                if !show_progress {
                    println!("✅ Model cloned successfully");
                }

                // Process LFS/XET files after successful clone
                if let Some(ref xet_storage) = self.xet_storage {
                    match self.process_pointer_files(&model_path, xet_storage, progress_bar.as_ref()).await {
                        Ok(processed_count) => {
                            if processed_count > 0 {
                                println!("📥 Downloaded {} LFS/XET files", processed_count);
                            }
                        }
                        Err(e) => {
                            eprintln!("⚠️  Warning: Failed to process some LFS/XET files: {}", e);
                        }
                    }
                }

                Ok((model_id, model_path))
            }
            Err(e) => {
                // Provide helpful error messages
                if e.message().contains("authentication") || e.message().contains("Authentication") {
                    bail!("Git authentication failed. Please configure authentication:\n\
                           \n\
                           For SSH URLs (git@...):\n\
                           • Use ssh-add to add your SSH key to the agent\n\
                           • Or ensure ~/.ssh/id_rsa or ~/.ssh/id_ed25519 exists\n\
                           \n\
                           For HTTPS URLs:\n\
                           • Configure git credential helper: git config --global credential.helper store\n\
                           • For HuggingFace: export HF_TOKEN=your_token\n\
                           • For GitHub: export GITHUB_TOKEN=your_token\n\
                           \n\
                           Error: {}", e);
                } else if e.message().contains("not found") || e.message().contains("repository") {
                    bail!("Repository not found or you don't have access: {}", git_url);
                } else {
                    bail!("Git clone failed: {}", e);
                }
            }
        }
    }
    
    /// Clone with specific branch or tag
    pub async fn clone_ref(
        &self,
        repo_url: &str,
        git_ref: &str,
    ) -> Result<(ModelId, PathBuf)> {
        let git_url = repo_url.to_string();

        // Always generate new model ID
        let model_id = ModelId::new();
        let model_path = self.cache_dir.join(model_id.0.to_string());

        if model_path.join(".git").exists() {
            println!("📦 Model already cloned, checking out ref: {}", git_ref);

            // Move git operations to blocking task
            let model_path_clone = model_path.clone();
            let git_ref_clone = git_ref.to_string();

            tokio::task::spawn_blocking(move || -> Result<()> {
                // Open existing repo and checkout ref
                let repo = Repository::open(&model_path_clone)?;

                // Fetch the ref first
                let mut remote = repo.find_remote("origin")?;
                remote.fetch(&[&git_ref_clone], Some(&mut FetchOptions::new()), None)?;

                // Checkout the ref
                let reference = repo.find_reference(&format!("refs/remotes/origin/{}", git_ref_clone))
                    .or_else(|_| repo.find_reference(&format!("refs/tags/{}", git_ref_clone)))
                    .or_else(|_| repo.find_reference(&git_ref_clone))?;

                let commit = reference.peel_to_commit()?;
                repo.checkout_tree(commit.as_object(), None)?;
                repo.set_head_detached(commit.id())?;

                Ok(())
            }).await??;

            // Process LFS/XET files after checkout
            if let Some(ref xet_storage) = self.xet_storage {
                match self.process_pointer_files(&model_path, xet_storage, None).await {
                    Ok(processed_count) => {
                        if processed_count > 0 {
                            println!("📥 Downloaded {} LFS/XET files", processed_count);
                        }
                    }
                    Err(e) => {
                        eprintln!("⚠️  Warning: Failed to process some LFS/XET files: {}", e);
                    }
                }
            }

            return Ok((model_id, model_path));
        }

        println!("🎯 Cloning specific ref: {}", git_ref);

        // Use blocking task for git operations
        let model_path_clone = model_path.clone();
        let git_ref_clone = git_ref.to_string();

        let result = tokio::task::spawn_blocking(move || -> Result<Repository> {
            let mut builder = RepoBuilder::new();
            builder.branch(&git_ref_clone);

            // Configure fetch options
            let mut fetch_opts = FetchOptions::new();
            fetch_opts.remote_callbacks(Self::get_callbacks());
            builder.fetch_options(fetch_opts);

            builder.clone(&git_url, &model_path_clone).map_err(|e| anyhow::anyhow!("Git clone failed: {}", e))
        }).await?;

        match result {
            Ok(_) => {
                println!("✅ Model cloned successfully at ref: {}", git_ref);

                // Process LFS/XET files after successful clone
                if let Some(ref xet_storage) = self.xet_storage {
                    match self.process_pointer_files(&model_path, xet_storage, None).await {
                        Ok(processed_count) => {
                            if processed_count > 0 {
                                println!("📥 Downloaded {} LFS/XET files", processed_count);
                            }
                        }
                        Err(e) => {
                            eprintln!("⚠️  Warning: Failed to process some LFS/XET files: {}", e);
                        }
                    }
                }

                Ok((model_id, model_path))
            }
            Err(e) => {
                bail!("Git clone of ref {} failed: {}", git_ref, e);
            }
        }
    }
    
    /// Update a cloned model repository
    pub fn update_model(&self, model_path: &Path) -> Result<()> {
        println!("🔄 Updating model repository...");
        
        let repo = Repository::open(model_path)?;
        
        // Fetch from origin
        let mut remote = repo.find_remote("origin")?;
        let mut fetch_opts = FetchOptions::new();
        fetch_opts.remote_callbacks(Self::get_callbacks());
        
        remote.fetch(&["refs/heads/*:refs/remotes/origin/*"], Some(&mut fetch_opts), None)?;
        
        // Fast-forward merge if possible
        let fetch_head = repo.find_reference("FETCH_HEAD")?;
        let fetch_commit = fetch_head.peel_to_commit()?;
        let annotated_commit = repo.find_annotated_commit(fetch_commit.id())?;
        let analysis = repo.merge_analysis(&[&annotated_commit])?;
        
        if analysis.0.is_fast_forward() {
            // Fast-forward
            let mut reference = repo.head()?;
            reference.set_target(fetch_commit.id(), "Fast-forward")?;
            repo.set_head(reference.name().unwrap())?;
            repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
            println!("✅ Model updated successfully");
        } else if analysis.0.is_up_to_date() {
            println!("✅ Model is already up to date");
        } else {
            bail!("Cannot fast-forward merge. Manual intervention required.");
        }
        
        Ok(())
    }
    
    /// Get git2 callbacks with progress tracking
    fn get_callbacks_with_progress(progress_bar: Option<ProgressBar>) -> git2::RemoteCallbacks<'static> {
        let mut callbacks = git2::RemoteCallbacks::new();

        // Set up progress callback using correct git2 API
        if let Some(pb) = progress_bar.clone() {
            callbacks.transfer_progress(move |progress: git2::Progress| {
                let received_objects = progress.received_objects();
                let total_objects = progress.total_objects();
                let received_bytes = progress.received_bytes();

                if total_objects > 0 {
                    pb.set_length(total_objects as u64);
                    pb.set_position(received_objects as u64);

                    let mb_received = received_bytes as f64 / (1024.0 * 1024.0);
                    pb.set_message(format!("Receiving objects... {:.1}MB ({}/{})", mb_received, received_objects, total_objects));
                } else {
                    // Fallback for indeterminate progress
                    pb.set_message("Receiving objects...");
                    pb.inc(1);
                }

                true // Continue transfer
            });
        }

        // Set up credential callbacks
        callbacks.credentials(|_url, username_from_url, allowed_types| {
            // First try SSH key from agent
            if allowed_types.contains(git2::CredentialType::SSH_KEY) {
                match git2::Cred::ssh_key_from_agent(username_from_url.unwrap_or("git")) {
                    Ok(cred) => return Ok(cred),
                    Err(_) => {} // Continue to next method
                }
            }
            // Try default SSH key locations
            if allowed_types.contains(git2::CredentialType::SSH_KEY) {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                let ssh_dir = PathBuf::from(home).join(".ssh");
                match git2::Cred::ssh_key(
                    username_from_url.unwrap_or("git"),
                    None,
                    &ssh_dir.join("id_rsa"),
                    None,
                ).or_else(|_| git2::Cred::ssh_key(
                    username_from_url.unwrap_or("git"),
                    None,
                    &ssh_dir.join("id_ed25519"),
                    None,
                )) {
                    Ok(cred) => return Ok(cred),
                    Err(_) => {} // Continue to next method
                }
            }

            // Try HF_TOKEN for HuggingFace repos
            if allowed_types.contains(git2::CredentialType::USER_PASS_PLAINTEXT) {
                if let Ok(token) = std::env::var("HF_TOKEN") {
                    match git2::Cred::userpass_plaintext("token", &token) {
                        Ok(cred) => return Ok(cred),
                        Err(_) => {} // Continue to next method
                    }
                }
                if let Ok(token) = std::env::var("GITHUB_TOKEN") {
                    match git2::Cred::userpass_plaintext("token", &token) {
                        Ok(cred) => return Ok(cred),
                        Err(_) => {} // Continue to next method
                    }
                }
            }

            // Fall back to git credential helper
            if allowed_types.contains(git2::CredentialType::DEFAULT) {
                match git2::Cred::default() {
                    Ok(cred) => return Ok(cred),
                    Err(_) => {} // Continue to final fallback
                }
            }

            // Final fallback - try default again or return error
            match git2::Cred::default() {
                Ok(cred) => Ok(cred),
                Err(e) => Err(e),
            }
        });

        callbacks
    }

    /// Get git2 callbacks that respect system configuration (without progress)
    fn get_callbacks() -> git2::RemoteCallbacks<'static> {
        Self::get_callbacks_with_progress(None)
    }

    /// Process LFS and XET pointer files after clone using xet-core
    async fn process_pointer_files(
        &self,
        repo_path: &Path,
        xet_storage: &XetNativeStorage,
        progress_bar: Option<&ProgressBar>,
    ) -> Result<usize> {
        let mut processed_count = 0;

        if let Some(pb) = progress_bar {
            pb.set_message("Scanning for pointer files...");
        }

        // Find all potential pointer files by scanning the repository
        let pointer_files = self.find_pointer_files(repo_path).await?;

        if pointer_files.is_empty() {
            return Ok(0);
        }

        if let Some(pb) = progress_bar {
            pb.set_length(pointer_files.len() as u64);
            pb.set_message("Processing pointer files...");
        }

        // Process each pointer file
        for (idx, file_path) in pointer_files.iter().enumerate() {
            if let Some(pb) = progress_bar {
                pb.set_position(idx as u64);
                pb.set_message(format!("Processing {}", file_path.file_name().unwrap_or_default().to_string_lossy()));
            }

            match self.process_single_pointer_file(file_path, xet_storage).await {
                Ok(true) => {
                    processed_count += 1;
                }
                Ok(false) => {
                    // File was not a pointer or already downloaded
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to process {}: {}", file_path.display(), e);
                }
            }
        }

        if let Some(pb) = progress_bar {
            pb.finish_with_message("Pointer file processing complete");
        }

        Ok(processed_count)
    }

    /// Find files that might be LFS or XET pointers
    async fn find_pointer_files(&self, repo_path: &Path) -> Result<Vec<PathBuf>> {
        let mut pointer_files = Vec::new();

        // Get patterns from .gitattributes
        let tracked_patterns = self.get_tracked_patterns(repo_path).await?;

        // Common large file extensions that might be pointers

        // Combine all patterns
        let all_patterns = tracked_patterns.iter().map(|s| s.as_str())
            .chain(COMMON_LFS_PATTERNS.iter().copied())
            .collect::<Vec<_>>();

        // Find files matching patterns
        for pattern in all_patterns {
            self.find_files_by_pattern(repo_path, pattern, &mut pointer_files).await?;
        }

        // Deduplicate
        pointer_files.sort();
        pointer_files.dedup();

        Ok(pointer_files)
    }

    /// Get tracked patterns from .gitattributes
    async fn get_tracked_patterns(&self, repo_path: &Path) -> Result<Vec<String>> {
        let gitattributes_path = repo_path.join(".gitattributes");
        if gitattributes_path.exists() {
            let content = fs::read_to_string(&gitattributes_path).await?;
            Ok(self.parse_lfs_patterns(&content))
        } else {
            Ok(Vec::new())
        }
    }

    /// Find files matching a pattern and add potential pointers to the list
    async fn find_files_by_pattern(&self, repo_path: &Path, pattern: &str, pointer_files: &mut Vec<PathBuf>) -> Result<()> {
        if let Ok(entries) = glob::glob(&repo_path.join(pattern).to_string_lossy()) {
            for entry in entries.flatten() {
                if entry.is_file() && self.is_likely_pointer_file(&entry).await? {
                    pointer_files.push(entry);
                }
            }
        }
        Ok(())
    }

    /// Parse .gitattributes for LFS-tracked patterns
    fn parse_lfs_patterns(&self, content: &str) -> Vec<String> {
        content
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.contains("filter=lfs") || line.contains("filter=xet") {
                    line.split_whitespace().next().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if a file is likely a pointer file (small size with specific content)
    async fn is_likely_pointer_file(&self, file_path: &Path) -> Result<bool> {
        let metadata = fs::metadata(file_path).await?;

        // Pointer files are typically small
        if metadata.len() > MAX_POINTER_FILE_SIZE {
            return Ok(false);
        }

        let content = fs::read_to_string(file_path).await?;
        Ok(self.is_lfs_pointer_content(&content) || self.is_xet_pointer_content(&content))
    }

    /// Check if content matches LFS pointer format per specification
    fn is_lfs_pointer_content(&self, content: &str) -> bool {
        self.parse_lfs_pointer(content).is_some()
    }

    /// Check if content matches XET pointer format
    fn is_xet_pointer_content(&self, content: &str) -> bool {
        let trimmed = content.trim();
        trimmed.starts_with('{') && trimmed.ends_with('}')
            && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    }

    /// Process a single pointer file using xet-core
    async fn process_single_pointer_file(
        &self,
        file_path: &Path,
        xet_storage: &XetNativeStorage,
    ) -> Result<bool> {
        let content = fs::read_to_string(file_path).await?;

        // Try XET pointer first
        if let Some(actual_content) = self.try_process_xet_pointer(&content, xet_storage).await? {
            fs::write(file_path, actual_content).await?;
            return Ok(true);
        }

        // Try LFS pointer with xet-core fallback
        if let Some(actual_content) = self.try_process_lfs_pointer(&content, file_path, xet_storage).await? {
            fs::write(file_path, actual_content).await?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Try to process as XET pointer
    async fn try_process_xet_pointer(&self, content: &str, xet_storage: &XetNativeStorage) -> Result<Option<Vec<u8>>> {
        if xet_storage.is_xet_pointer(content) {
            Ok(Some(xet_storage.smudge_bytes(content).await?))
        } else {
            Ok(None)
        }
    }

    /// Try to process as LFS pointer using xet-core
    async fn try_process_lfs_pointer(&self, content: &str, file_path: &Path, xet_storage: &XetNativeStorage) -> Result<Option<Vec<u8>>> {
        if let Some(lfs_pointer) = self.parse_lfs_pointer(content) {
            // Try xet-core's universal file loader for LFS files in git-xet repositories
            match xet_storage.load_file(file_path).await {
                Ok(actual_content) => {
                    // Only return if content is actually different from the pointer
                    if actual_content.len() as u64 != lfs_pointer.size || actual_content != content.as_bytes() {
                        // Validate content integrity before returning
                        self.validate_content_integrity(&actual_content, &lfs_pointer)?;
                        return Ok(Some(actual_content));
                    }
                }
                Err(_) => {
                    // Expected for pure LFS repos without XET integration
                }
            }
        }
        Ok(None)
    }

    /// Validate content integrity against LFS pointer metadata
    fn validate_content_integrity(&self, content: &[u8], lfs_pointer: &LfsPointer) -> Result<()> {
        // Validate file size
        if content.len() as u64 != lfs_pointer.size {
            bail!(
                "Content size mismatch: expected {}, got {}",
                lfs_pointer.size,
                content.len()
            );
        }

        // Validate SHA256 hash
        if lfs_pointer.oid.starts_with(LFS_SHA256_PREFIX) {
            let expected_hash = &lfs_pointer.oid[LFS_SHA256_PREFIX.len()..];
            let mut hasher = Sha256::new();
            hasher.update(content);
            let actual_hash = format!("{:x}", hasher.finalize());

            if actual_hash != expected_hash {
                bail!(
                    "SHA256 hash mismatch: expected {}, got {}",
                    expected_hash,
                    actual_hash
                );
            }
        } else {
            bail!("Unsupported hash algorithm in OID: {}", lfs_pointer.oid);
        }

        Ok(())
    }

    /// Parse LFS pointer file content per Git LFS specification
    fn parse_lfs_pointer(&self, content: &str) -> Option<LfsPointer> {
        // Handle empty file case per spec: "An empty file serves as its own pointer without modification"
        if content.is_empty() {
            return None; // Empty files are not LFS pointers, they represent themselves
        }

        // Validate file size constraint (< 1024 bytes)
        if content.len() > MAX_POINTER_FILE_SIZE as usize {
            return None;
        }

        // Parse key-value pairs - be more lenient per spec
        let mut pairs = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for line in lines {
            // Skip empty lines (not explicitly forbidden by spec)
            if line.is_empty() {
                continue;
            }

            // Each line must be "{key} {value}" format with exactly one space
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return None; // Invalid format
            }

            let (key, value) = (parts[0], parts[1]);

            // Validate key format: only [a-z] [0-9] . -
            if !self.is_valid_lfs_key(key) {
                return None;
            }

            // MUST NOT: Values cannot contain return or newline characters per spec
            if value.contains('\r') || value.contains('\n') {
                return None;
            }

            pairs.push((key, value));
        }

        // Must have at least the required fields
        if pairs.is_empty() {
            return None;
        }

        // Validate required fields and extract values
        self.validate_and_extract_lfs_fields(pairs)
    }


    /// Validate LFS key format per specification
    fn is_valid_lfs_key(&self, key: &str) -> bool {
        !key.is_empty() && key.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-')
    }

    /// Validate required LFS fields and extract pointer data
    fn validate_and_extract_lfs_fields(&self, pairs: Vec<(&str, &str)>) -> Option<LfsPointer> {
        let mut version = None;
        let mut oid = None;
        let mut size = None;

        // Validate first line is version
        if pairs.is_empty() || pairs[0].0 != LFS_VERSION_KEY {
            return None;
        }

        // Check that version is first, then remaining lines are sorted alphabetically
        if pairs.len() > 1 {
            let remaining_keys: Vec<&str> = pairs.iter().skip(1).map(|(k, _)| *k).collect();
            let mut sorted_remaining = remaining_keys.clone();
            sorted_remaining.sort();
            if remaining_keys != sorted_remaining {
                return None; // Lines after version must be sorted alphabetically
            }
        }

        // Extract required fields
        for (key, value) in pairs {
            match key {
                LFS_VERSION_KEY => {
                    if value != LFS_VERSION_SPEC_URL {
                        return None; // Only v1 spec supported
                    }
                    version = Some(value.to_string());
                }
                LFS_OID_KEY => {
                    // Must be "sha256:hash" format (currently only sha256 supported per spec)
                    if !value.starts_with(LFS_SHA256_PREFIX) {
                        return None;
                    }
                    let hash = &value[LFS_SHA256_PREFIX.len()..];
                    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                        return None; // Invalid SHA256 hash format
                    }
                    oid = Some(value.to_string());
                }
                LFS_SIZE_KEY => {
                    // Size must be a valid positive integer (0 is valid for empty files)
                    if let Ok(parsed_size) = value.parse::<u64>() {
                        size = Some(parsed_size);
                    } else {
                        return None; // Invalid size format
                    }
                }
                _ => {
                    // Other keys are allowed but ignored
                }
            }
        }

        // All required fields must be present
        match (version, oid, size) {
            (Some(version), Some(oid), Some(size)) => Some(LfsPointer { version, oid, size }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a GitModelSource for testing
    fn create_test_git_source() -> GitModelSource {
        GitModelSource::new(PathBuf::from("/tmp/test"))
    }

    mod lfs_pointer_parsing {
        use super::*;

        #[test]
        fn test_valid_lfs_pointer_basic() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should parse valid LFS pointer");

            let pointer = result.unwrap();
            assert_eq!(pointer.version, "https://git-lfs.github.com/spec/v1");
            assert_eq!(pointer.oid, "sha256:abc123def456789012345678901234567890123456789012345678901234567890");
            assert_eq!(pointer.size, 12345);
        }

        #[test]
        fn test_valid_lfs_pointer_real_example() {
            let git_source = create_test_git_source();
            // Real LFS pointer from the cloned model
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223\nsize 3957900840\n";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should parse real LFS pointer");

            let pointer = result.unwrap();
            assert_eq!(pointer.size, 3957900840);
        }

        #[test]
        fn test_valid_lfs_pointer_uppercase_hex() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:ABC123DEF456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should accept uppercase hex");
        }

        #[test]
        fn test_valid_lfs_pointer_mixed_case_hex() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:aBc123DeF456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should accept mixed case hex");
        }

        #[test]
        fn test_valid_lfs_pointer_with_empty_lines() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\n\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\n\nsize 12345\n";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should skip empty lines");
        }

        #[test]
        fn test_valid_lfs_pointer_no_trailing_newline() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should accept content without trailing newline");
        }

        #[test]
        fn test_valid_lfs_pointer_zero_size() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 0\n";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should accept zero size");
            assert_eq!(result.unwrap().size, 0);
        }

        #[test]
        fn test_valid_lfs_pointer_with_extra_fields() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\ncustom.field value\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            let result = git_source.parse_lfs_pointer(content);
            assert!(result.is_some(), "Should accept extra fields if properly sorted");
        }
    }

    mod lfs_pointer_validation_errors {
        use super::*;

        #[test]
        fn test_empty_content() {
            let git_source = create_test_git_source();
            assert!(git_source.parse_lfs_pointer("").is_none(), "Should reject empty content");
        }

        #[test]
        fn test_too_large_content() {
            let git_source = create_test_git_source();
            let large_content = "a".repeat(2000);
            assert!(git_source.parse_lfs_pointer(&large_content).is_none(), "Should reject content > 1024 bytes");
        }

        #[test]
        fn test_invalid_version_not_first() {
            let git_source = create_test_git_source();
            let content = "oid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nversion https://git-lfs.github.com/spec/v1\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject when version is not first");
        }

        #[test]
        fn test_invalid_version_url() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v2\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject invalid version URL");
        }

        #[test]
        fn test_missing_required_field_oid() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject missing oid");
        }

        #[test]
        fn test_missing_required_field_size() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject missing size");
        }

        #[test]
        fn test_invalid_oid_format_no_prefix() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject OID without sha256: prefix");
        }

        #[test]
        fn test_invalid_oid_hash_too_short() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject short hash");
        }

        #[test]
        fn test_invalid_oid_hash_too_long() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890ab\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject long hash");
        }

        #[test]
        fn test_invalid_oid_hash_non_hex() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:xyz123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject non-hex characters in hash");
        }

        #[test]
        fn test_invalid_size_format() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize abc\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject non-numeric size");
        }

        #[test]
        fn test_invalid_line_format_no_space() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noidsha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject line without space");
        }

        #[test]
        fn test_invalid_key_format_uppercase() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\nOID sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject uppercase in key");
        }

        #[test]
        fn test_invalid_key_format_invalid_chars() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid_test sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject underscore in key");
        }

        #[test]
        fn test_invalid_value_with_newline() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 123\n45\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject newline in value");
        }

        #[test]
        fn test_invalid_value_with_return() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 123\r45\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject carriage return in value");
        }

        #[test]
        fn test_invalid_unsorted_keys() {
            let git_source = create_test_git_source();
            let content = "version https://git-lfs.github.com/spec/v1\nsize 12345\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\n";

            assert!(git_source.parse_lfs_pointer(content).is_none(), "Should reject unsorted keys");
        }
    }

    mod lfs_key_validation {
        use super::*;

        #[test]
        fn test_valid_keys() {
            let git_source = create_test_git_source();

            assert!(git_source.is_valid_lfs_key("version"));
            assert!(git_source.is_valid_lfs_key("oid"));
            assert!(git_source.is_valid_lfs_key("size"));
            assert!(git_source.is_valid_lfs_key("custom.field"));
            assert!(git_source.is_valid_lfs_key("field-name"));
            assert!(git_source.is_valid_lfs_key("field123"));
        }

        #[test]
        fn test_invalid_keys() {
            let git_source = create_test_git_source();

            assert!(!git_source.is_valid_lfs_key(""), "Empty key should be invalid");
            assert!(!git_source.is_valid_lfs_key("Version"), "Uppercase should be invalid");
            assert!(!git_source.is_valid_lfs_key("field_name"), "Underscore should be invalid");
            assert!(!git_source.is_valid_lfs_key("field name"), "Space should be invalid");
            assert!(!git_source.is_valid_lfs_key("field@name"), "Special chars should be invalid");
        }
    }

    mod pointer_detection {
        use super::*;

        #[test]
        fn test_lfs_pointer_content_detection() {
            let git_source = create_test_git_source();

            let valid_lfs = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123def456789012345678901234567890123456789012345678901234567890\nsize 12345\n";
            assert!(git_source.is_lfs_pointer_content(valid_lfs), "Should detect valid LFS content");

            let invalid_content = "This is not an LFS pointer";
            assert!(!git_source.is_lfs_pointer_content(invalid_content), "Should not detect non-LFS content");

            let binary_content = &[0x8b, 0x00, 0x01, 0x02];
            let binary_string = String::from_utf8_lossy(binary_content);
            assert!(!git_source.is_lfs_pointer_content(&binary_string), "Should not detect binary content");
        }

        #[test]
        fn test_xet_pointer_content_detection() {
            let git_source = create_test_git_source();

            let valid_json = r#"{"hash":"abc123","file_size":1024}"#;
            assert!(git_source.is_xet_pointer_content(valid_json), "Should detect valid JSON");

            let invalid_json = "not json";
            assert!(!git_source.is_xet_pointer_content(invalid_json), "Should not detect non-JSON");

            let empty_content = "";
            assert!(!git_source.is_xet_pointer_content(empty_content), "Should not detect empty content");
        }
    }
}