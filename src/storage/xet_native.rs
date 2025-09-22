//! Native Xet storage integration using xet-core v1.1.10
//!
//! Provides transparent integration with git-xet repositories by automatically
//! detecting and handling Xet pointer files. This module allows seamless
//! operation with both regular files and Xet-tracked files without requiring
//! Git LFS or git-xet transfer agent setup.

use anyhow::{Result, Context, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::fs;
use indicatif::{ProgressBar, ProgressStyle};
use data::{
    FileUploadSession, FileDownloader, XetFileInfo,
    data_client,
};
use cas_client::{OutputProvider, FileProvider};
use cas_object::CompressionScheme;
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
            self.progress_bar.set_message(format!("{}: {:.1}MB/{:.1}MB", item_name, mb_current, mb_total));
        } else {
            self.progress_bar.set_message(format!("{}: {}B", item_name, current));
        }
        self.progress_bar.set_position(current);
    }

    pub fn finish_item(&self, _item_name: &str) {
        self.progress_bar.finish();
    }
}

/// Configuration for Xet storage
#[derive(Debug, Clone)]
pub struct XetConfig {
    /// CAS endpoint URL (e.g., "https://cas.xet.dev")
    pub endpoint: String,
    /// Authentication token
    pub token: Option<String>,
    /// Optional compression scheme
    pub compression: Option<CompressionScheme>,
}

impl Default for XetConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://cas.xet.dev".to_string(),
            token: None,
            compression: None,
        }
    }
}

/// Native Xet storage implementation using xet-core APIs
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
    pub async fn new_with_progress(config: XetConfig, progress_bar: Option<ProgressBar>) -> Result<Self> {
        // Create translator config using xet-core's default_config
        let translator_config = Arc::new(data_client::default_config(
            config.endpoint.clone(),
            config.compression,
            config.token.as_ref().map(|t| (t.clone(), u64::MAX)), // Token with far future expiry
            None, // No token refresher
        ).context("Failed to create Xet translator config")?);

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
        let (xet_info, _metrics) = data_client::clean_file(
            self.upload_session.clone(),
            file_path
        ).await.context("Failed to clean file for Xet storage")?;

        xet_info.as_pointer_file()
            .context("Failed to serialize Xet pointer")
    }

    /// Store data from memory and return Xet pointer
    pub async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        let (xet_info, _metrics) = data_client::clean_bytes(
            self.upload_session.clone(),
            data.to_vec()
        ).await.context("Failed to clean bytes for Xet storage")?;

        xet_info.as_pointer_file()
            .context("Failed to serialize Xet pointer")
    }

    /// Retrieve file from Xet pointer to specified path (smudge operation)
    pub async fn smudge_file(&self, pointer_content: &str, output_path: &Path) -> Result<()> {
        let xet_info: XetFileInfo = serde_json::from_str(pointer_content)
            .context("Failed to parse Xet pointer")?;

        let merkle_hash = xet_info.merkle_hash()
            .context("Failed to parse merkle hash from Xet pointer")?;

        self.downloader.smudge_file_from_hash(
            &merkle_hash,
            output_path.to_string_lossy().into(),
            &OutputProvider::File(FileProvider::new(output_path.to_path_buf())),
            None,
            None
        )
        .await
        .context("Failed to smudge file from Xet storage")?;

        Ok(())
    }

    /// Retrieve file from Xet pointer to memory
    pub async fn smudge_bytes(&self, pointer_content: &str) -> Result<Vec<u8>> {
        // Create temporary file for smudging
        let temp_file = NamedTempFile::new()
            .context("Failed to create temporary file")?;

        self.smudge_file(pointer_content, temp_file.path()).await?;

        // Read the smudged content
        fs::read(temp_file.path()).await
            .context("Failed to read smudged file")
    }

    /// Check if content is a Xet pointer file
    pub fn is_xet_pointer(&self, content: &str) -> bool {
        serde_json::from_str::<XetFileInfo>(content).is_ok()
    }

    /// Universal file loader - handles both Xet pointers and regular files
    ///
    /// This is the main interface for loading files. It automatically detects
    /// if the file contains a Xet pointer and retrieves the actual content,
    /// or returns the file content directly if it's not a pointer.
    pub async fn load_file(&self, file_path: &Path) -> Result<Vec<u8>> {
        let content = fs::read_to_string(file_path).await
            .context("Failed to read file")?;

        if self.is_xet_pointer(&content) {
            // It's a Xet pointer - retrieve the actual content
            self.smudge_bytes(&content).await
        } else {
            // Regular file - return content as bytes
            Ok(content.into_bytes())
        }
    }

    /// Universal file saver - can store as regular file or convert to Xet
    ///
    /// If the target file already exists as a Xet pointer, this will update
    /// the pointed-to content and maintain it as a Xet file. Otherwise, it
    /// stores the file normally.
    pub async fn save_file(&self, file_path: &Path, data: &[u8]) -> Result<()> {
        // Check if the file already exists as a Xet pointer
        if file_path.exists() {
            let existing_content = fs::read_to_string(file_path).await
                .context("Failed to read existing file")?;

            if self.is_xet_pointer(&existing_content) {
                // File is already a Xet pointer - update it
                let pointer = self.clean_bytes(data).await?;
                fs::write(file_path, pointer).await
                    .context("Failed to write updated Xet pointer")?;
                return Ok(());
            }
        }

        // Store as regular file
        fs::write(file_path, data).await
            .context("Failed to write regular file")
    }

    /// Explicitly store file as Xet (force conversion)
    ///
    /// This always stores the file as a Xet pointer, regardless of whether
    /// it was previously a regular file or Xet pointer.
    pub async fn save_as_xet(&self, file_path: &Path, data: &[u8]) -> Result<()> {
        let pointer = self.clean_bytes(data).await?;
        fs::write(file_path, pointer).await
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

        let content = fs::read_to_string(file_path).await
            .context("Failed to read file")?;

        if self.is_xet_pointer(&content) {
            let xet_info: XetFileInfo = serde_json::from_str(&content)
                .context("Failed to parse Xet pointer")?;
            Ok(xet_info.file_size())
        } else {
            Ok(content.len() as u64)
        }
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
}