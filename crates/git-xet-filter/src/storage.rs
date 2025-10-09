//! Storage backend trait and XET implementation

#[cfg(feature = "xet-storage")]
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "xet-storage")]
use data::{FileDownloader, FileUploadSession, XetFileInfo};

use crate::error::{Result, XetError, XetErrorKind};

/// Storage backend trait for filter operations
#[cfg(feature = "xet-storage")]
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Upload file and return pointer JSON
    async fn clean_file(&self, path: &Path) -> Result<String>;

    /// Download from pointer JSON to bytes
    async fn smudge_pointer(&self, pointer: &str) -> Result<Vec<u8>>;

    /// Check if content is a valid pointer
    fn is_pointer(&self, content: &str) -> bool;
}

/// XET storage backend implementation
#[cfg(feature = "xet-storage")]
pub struct XetStorage {
    upload_session: Arc<FileUploadSession>,
    downloader: Arc<FileDownloader>,
}

#[cfg(feature = "xet-storage")]
impl XetStorage {
    pub async fn new(config: &crate::config::XetConfig) -> Result<Self> {
        let translator_config = Arc::new(
            data::data_client::default_config(
                config.endpoint.clone(),
                config.compression,
                config.token.as_ref().map(|t| (t.clone(), u64::MAX)),
                None,
            )
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::RuntimeError,
                    format!("Failed to create XET config: {}", e),
                )
            })?,
        );

        let upload_session = FileUploadSession::new(translator_config.clone(), None)
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::UploadFailed,
                    format!("Failed to create upload session: {}", e),
                )
            })?;

        let downloader = FileDownloader::new(translator_config).await.map_err(|e| {
            XetError::new(
                XetErrorKind::DownloadFailed,
                format!("Failed to create downloader: {}", e),
            )
        })?;

        Ok(Self {
            upload_session,
            downloader: Arc::new(downloader),
        })
    }
}

#[cfg(feature = "xet-storage")]
#[async_trait]
impl StorageBackend for XetStorage {
    async fn clean_file(&self, path: &Path) -> Result<String> {
        let (xet_info, _metrics) = data::data_client::clean_file(self.upload_session.clone(), path)
            .await
            .map_err(|e| {
                XetError::new(XetErrorKind::UploadFailed, format!("Clean failed: {}", e))
            })?;

        xet_info.as_pointer_file().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Failed to serialize pointer: {}", e),
            )
        })
    }

    async fn smudge_pointer(&self, pointer: &str) -> Result<Vec<u8>> {
        let xet_info: XetFileInfo = serde_json::from_str(pointer).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid pointer JSON: {}", e),
            )
        })?;

        let merkle_hash = xet_info.merkle_hash().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid merkle hash: {}", e),
            )
        })?;

        // Download to temp file
        let temp_file = tempfile::NamedTempFile::new().map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to create temp file: {}", e),
            )
        })?;

        use cas_client::{FileProvider, OutputProvider};
        self.downloader
            .smudge_file_from_hash(
                &merkle_hash,
                temp_file.path().to_string_lossy().into(),
                &OutputProvider::File(FileProvider::new(temp_file.path().to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge failed: {}", e),
                )
            })?;

        // Read back
        std::fs::read(temp_file.path()).map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to read smudged file: {}", e),
            )
        })
    }

    fn is_pointer(&self, content: &str) -> bool {
        serde_json::from_str::<XetFileInfo>(content).is_ok()
    }
}
