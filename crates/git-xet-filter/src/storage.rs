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

    /// Upload data from memory and return pointer JSON
    async fn clean_bytes(&self, data: &[u8]) -> Result<String>;

    /// Download from pointer directly to a file path
    async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Result<()>;

    /// Download from pointer to bytes (better-named version of smudge_pointer)
    async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>>;

    /// Download from merkle hash to bytes (for LFS pointer resolution)
    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>>;

    /// Download from merkle hash directly to file (for LFS pointer resolution)
    async fn smudge_from_hash_to_file(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
    ) -> Result<()>;
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

    /// Upload data from memory and return XET pointer JSON
    ///
    /// This is more efficient than writing to a temp file first when the data
    /// is already in memory (e.g., from tensor serialization).
    pub async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        let (xet_info, _metrics) = data::data_client::clean_bytes(
            self.upload_session.clone(),
            data.to_vec(),
        )
        .await
        .map_err(|e| {
            XetError::new(
                XetErrorKind::UploadFailed,
                format!("Clean bytes failed: {}", e),
            )
        })?;

        xet_info.as_pointer_file().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Failed to serialize pointer: {}", e),
            )
        })
    }

    /// Download from XET pointer directly to a file path
    ///
    /// This is more efficient than downloading to memory first when the final
    /// destination is a file (e.g., model weights).
    pub async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Result<()> {
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

        use cas_client::{FileProvider, OutputProvider};
        self.downloader
            .smudge_file_from_hash(
                &merkle_hash,
                output_path.to_string_lossy().into(),
                &OutputProvider::File(FileProvider::new(output_path.to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge to file failed: {}", e),
                )
            })?;

        Ok(())
    }

    /// Download from XET pointer to memory
    ///
    /// This is a better-named version of `smudge_pointer()` and uses async I/O.
    pub async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>> {
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
                    format!("Smudge to bytes failed: {}", e),
                )
            })?;

        // Read back with async I/O
        tokio::fs::read(temp_file.path()).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to read smudged file: {}", e),
            )
        })
    }

    /// Download from XET by merkle hash to memory
    ///
    /// This is useful for LFS pointer resolution where you have a SHA256 hash
    /// and need to retrieve the content directly without a XET pointer.
    pub async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>> {
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
                hash,
                temp_file.path().to_string_lossy().into(),
                &OutputProvider::File(FileProvider::new(temp_file.path().to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge from hash failed: {}", e),
                )
            })?;

        // Read back with async I/O
        tokio::fs::read(temp_file.path()).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to read smudged file: {}", e),
            )
        })
    }

    /// Download from XET by merkle hash directly to a file path
    ///
    /// This is useful for LFS pointer resolution where you have a SHA256 hash
    /// and want to write directly to disk.
    pub async fn smudge_from_hash_to_file(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
    ) -> Result<()> {
        use cas_client::{FileProvider, OutputProvider};
        self.downloader
            .smudge_file_from_hash(
                hash,
                output_path.to_string_lossy().into(),
                &OutputProvider::File(FileProvider::new(output_path.to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge from hash to file failed: {}", e),
                )
            })?;

        Ok(())
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
        // Delegate to smudge_bytes for better implementation
        self.smudge_bytes(pointer).await
    }

    fn is_pointer(&self, content: &str) -> bool {
        serde_json::from_str::<XetFileInfo>(content).is_ok()
    }

    async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        // Delegate to the implementation method
        XetStorage::clean_bytes(self, data).await
    }

    async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Result<()> {
        // Delegate to the implementation method
        XetStorage::smudge_file(self, pointer, output_path).await
    }

    async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>> {
        // Delegate to the implementation method
        XetStorage::smudge_bytes(self, pointer).await
    }

    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>> {
        // Delegate to the implementation method
        XetStorage::smudge_from_hash(self, hash).await
    }

    async fn smudge_from_hash_to_file(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
    ) -> Result<()> {
        // Delegate to the implementation method
        XetStorage::smudge_from_hash_to_file(self, hash, output_path).await
    }
}
