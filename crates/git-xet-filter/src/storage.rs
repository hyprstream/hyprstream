//! Storage backend for XET content-addressed storage operations.
//!
//! This module provides the [`StorageBackend`] trait for abstracting XET CAS
//! (Content-Addressed Storage) operations, and the [`XetStorage`] implementation
//! that integrates with XET's data client.
//!
//! # Overview
//!
//! The storage backend handles two main operations:
//!
//! - **Clean**: Upload file content to XET CAS and return a JSON pointer
//! - **Smudge**: Download content from XET CAS using a JSON pointer or merkle hash
//!
//! # Example
//!
//! ```ignore
//! use git_xet_filter::storage::{StorageBackend, XetStorage};
//! use git_xet_filter::config::XetConfig;
//!
//! let config = XetConfig::huggingface();
//! let storage = XetStorage::new(&config).await?;
//!
//! // Upload a file
//! let pointer = storage.clean_file(Path::new("model.safetensors")).await?;
//!
//! // Download to bytes
//! let data = storage.smudge_bytes(&pointer).await?;
//! ```

#[cfg(feature = "xet-storage")]
use async_trait::async_trait;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};

#[cfg(feature = "xet-storage")]
use data::{FileDownloader, FileUploadSession, XetFileInfo};
#[cfg(feature = "xet-storage")]
use tokio::io::AsyncWrite;

use crate::error::{Result, XetError, XetErrorKind};

/// Storage backend trait for XET filter operations.
///
/// This trait abstracts the XET content-addressed storage operations,
/// allowing for clean (upload) and smudge (download) of file content.
///
/// All methods are async and designed for use with the XET data client.
/// Implementations must be `Send + Sync` for use across threads.
#[cfg(feature = "xet-storage")]
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Upload a file to XET CAS and return a JSON pointer string.
    ///
    /// The returned pointer contains metadata (size, merkle hash) needed
    /// to retrieve the content later via [`smudge_bytes`] or [`smudge_file`].
    ///
    /// [`smudge_bytes`]: StorageBackend::smudge_bytes
    /// [`smudge_file`]: StorageBackend::smudge_file
    async fn clean_file(&self, path: &Path) -> Result<String>;

    /// Check if a string is a valid XET pointer.
    ///
    /// Returns `true` if the content can be parsed as a valid XET pointer JSON.
    fn is_pointer(&self, content: &str) -> bool;

    /// Upload data from memory to XET CAS and return a JSON pointer string.
    ///
    /// This is more efficient than [`clean_file`] when data is already in memory.
    ///
    /// [`clean_file`]: StorageBackend::clean_file
    async fn clean_bytes(&self, data: &[u8]) -> Result<String>;

    /// Download content from a pointer directly to a file.
    ///
    /// This is more efficient than [`smudge_bytes`] when the destination is a file,
    /// as it avoids intermediate memory allocation.
    ///
    /// [`smudge_bytes`]: StorageBackend::smudge_bytes
    async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Result<()>;

    /// Download content from a pointer to an in-memory buffer.
    ///
    /// Use this for small files or when the content needs processing.
    /// For large files, prefer [`smudge_file`] to write directly to disk.
    ///
    /// [`smudge_file`]: StorageBackend::smudge_file
    async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>>;

    /// Download content by merkle hash to an in-memory buffer.
    ///
    /// This is useful for LFS pointer resolution where you have a SHA256 hash
    /// instead of a full XET pointer JSON string.
    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>>;

    /// Download content by merkle hash directly to a file.
    ///
    /// Combines the efficiency of direct file writes with hash-based lookups
    /// for LFS pointer resolution.
    async fn smudge_from_hash_to_file(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
    ) -> Result<()>;
}

/// XET storage backend implementation.
///
/// Wraps the XET data client to provide clean/smudge operations for
/// git filter integration. Uses async I/O for all network operations.
///
/// # Example
///
/// ```ignore
/// use git_xet_filter::storage::XetStorage;
/// use git_xet_filter::config::XetConfig;
///
/// let config = XetConfig::huggingface();
/// let storage = XetStorage::new(&config).await?;
///
/// // Clean (upload) a file
/// let pointer = storage.clean_file(Path::new("weights.safetensors")).await?;
///
/// // Smudge (download) to memory
/// let data = storage.smudge_bytes(&pointer).await?;
/// ```
#[cfg(feature = "xet-storage")]
pub struct XetStorage {
    upload_session: Arc<FileUploadSession>,
    downloader: Arc<FileDownloader>,
}

/// Sequential CAS output that refuses writes beyond a caller-provided byte
/// ceiling. Unlike a post-download metadata check, this bounds bytes reaching
/// the private projection even when a CAS object disagrees with its pointer.
#[cfg(feature = "xet-storage")]
struct BoundedFileWriter {
    file: std::fs::File,
    limit: u64,
    written: u64,
}

#[cfg(feature = "xet-storage")]
impl BoundedFileWriter {
    fn create(path: &Path, limit: u64) -> io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        Ok(Self {
            file,
            limit,
            written: 0,
        })
    }
}

#[cfg(feature = "xet-storage")]
impl AsyncWrite for BoundedFileWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let Some(end) = self.written.checked_add(buf.len() as u64) else {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "CAS output size overflow",
            )));
        };
        if end > self.limit {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CAS output exceeds {0}-byte limit", self.limit),
            )));
        }
        loop {
            match self.file.write(buf) {
                Ok(written) => {
                    self.written += written as u64;
                    return Poll::Ready(Ok(written));
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.file.flush())
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.file.flush())
    }
}

#[cfg(all(test, feature = "xet-storage"))]
mod bounded_output_tests {
    use super::BoundedFileWriter;
    use std::io;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn oversized_cas_chunk_is_rejected_before_file_growth() -> io::Result<()> {
        let temp = tempfile::tempdir()?;
        let output = temp.path().join("bounded-output");
        let mut writer = BoundedFileWriter::create(&output, 4)?;

        assert!(writer.write_all(b"oversized").await.is_err());
        writer.flush().await?;
        drop(writer);
        assert_eq!(std::fs::metadata(&output)?.len(), 0);
        assert_eq!(std::fs::read(&output)?, b"");
        Ok(())
    }

    #[tokio::test]
    async fn understated_pointer_cannot_write_past_declared_size() -> io::Result<()> {
        let temp = tempfile::tempdir()?;
        let output = temp.path().join("bounded-output");
        let mut writer = BoundedFileWriter::create(&output, 4)?;

        writer.write_all(b"1234").await?;
        assert!(writer.write_all(b"5").await.is_err());
        writer.flush().await?;
        drop(writer);
        assert_eq!(std::fs::read(&output)?, b"1234");
        assert_eq!(std::fs::metadata(&output)?.len(), 4);
        Ok(())
    }
}

#[cfg(feature = "xet-storage")]
impl XetStorage {
    /// Create a new XET storage backend with the given configuration.
    ///
    /// Initializes the upload session and downloader with the endpoint
    /// and authentication from the config.
    ///
    /// # Errors
    ///
    /// Returns an error if the XET data client fails to initialize,
    /// typically due to network issues or invalid credentials.
    pub async fn new(config: &crate::config::XetConfig) -> Result<Self> {
        let translator_config = Arc::new(
            data::data_client::default_config(
                config.endpoint.clone(),
                config.compression,
                config.token.as_ref().map(|t| (t.clone(), u64::MAX)),
                None,
                "hyprstream".to_owned(),
            )
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::RuntimeError,
                    format!("Failed to create XET config: {e}"),
                )
            })?,
        );

        let upload_session = FileUploadSession::new(translator_config.clone(), None)
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::UploadFailed,
                    format!("Failed to create upload session: {e}"),
                )
            })?;

        let downloader = FileDownloader::new(translator_config).await.map_err(|e| {
            XetError::new(
                XetErrorKind::DownloadFailed,
                format!("Failed to create downloader: {e}"),
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
        let (xet_info, _metrics) =
            data::data_client::clean_bytes(self.upload_session.clone(), data.to_vec())
                .await
                .map_err(|e| {
                    XetError::new(
                        XetErrorKind::UploadFailed,
                        format!("Clean bytes failed: {e}"),
                    )
                })?;

        xet_info.as_pointer_file().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Failed to serialize pointer: {e}"),
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
                format!("Invalid pointer JSON: {e}"),
            )
        })?;

        let merkle_hash = xet_info.merkle_hash().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid merkle hash: {e}"),
            )
        })?;

        use cas_client::{FileProvider, SeekingOutputProvider};
        self.downloader
            .smudge_file_from_hash(
                &merkle_hash,
                output_path.to_string_lossy().into(),
                SeekingOutputProvider::File(FileProvider::new(output_path.to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge to file failed: {e}"),
                )
            })?;

        Ok(())
    }

    /// Download a pointer into a new file while enforcing `max_output_bytes`
    /// at each sequential write. This is intended for callers that must bound
    /// untrusted or stale pointer metadata before the payload reaches disk.
    pub async fn smudge_file_bounded(
        &self,
        pointer: &str,
        output_path: &Path,
        max_output_bytes: u64,
    ) -> Result<u64> {
        let info: XetFileInfo = serde_json::from_str(pointer).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid pointer JSON: {e}"),
            )
        })?;
        if info.file_size() > max_output_bytes {
            return Err(XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Pointer size exceeds {max_output_bytes}-byte output limit"),
            ));
        }
        let hash = info.merkle_hash().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid merkle hash: {e}"),
            )
        })?;
        self.smudge_hash_to_file_bounded(&hash, output_path, max_output_bytes)
            .await
    }

    /// Download content by Merkle hash using a strict write-time byte ceiling.
    pub async fn smudge_hash_to_file_bounded(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
        max_output_bytes: u64,
    ) -> Result<u64> {
        let writer = BoundedFileWriter::create(output_path, max_output_bytes).map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Create bounded CAS output: {e}"),
            )
        })?;
        let output: cas_client::SequentialOutput = Box::new(writer);
        let written = self
            .downloader
            .smudge_file_from_hash_sequential(
                hash,
                output_path.to_string_lossy().into(),
                output,
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Bounded smudge to file failed: {e}"),
                )
            })?;
        if written > max_output_bytes {
            return Err(XetError::new(
                XetErrorKind::DownloadFailed,
                "CAS downloader reported bytes beyond its enforced output limit",
            ));
        }
        Ok(written)
    }

    /// Download from XET pointer to memory
    ///
    /// This is a better-named version of `smudge_pointer()` and uses async I/O.
    pub async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>> {
        let xet_info: XetFileInfo = serde_json::from_str(pointer).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid pointer JSON: {e}"),
            )
        })?;

        let merkle_hash = xet_info.merkle_hash().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid merkle hash: {e}"),
            )
        })?;

        // Download to temp file
        let temp_file = tempfile::NamedTempFile::new().map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to create temp file: {e}"),
            )
        })?;

        use cas_client::{FileProvider, SeekingOutputProvider};
        self.downloader
            .smudge_file_from_hash(
                &merkle_hash,
                temp_file.path().to_string_lossy().into(),
                SeekingOutputProvider::File(FileProvider::new(temp_file.path().to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge to bytes failed: {e}"),
                )
            })?;

        // Read back with async I/O
        tokio::fs::read(temp_file.path()).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to read smudged file: {e}"),
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
                format!("Failed to create temp file: {e}"),
            )
        })?;

        use cas_client::{FileProvider, SeekingOutputProvider};
        self.downloader
            .smudge_file_from_hash(
                hash,
                temp_file.path().to_string_lossy().into(),
                SeekingOutputProvider::File(FileProvider::new(temp_file.path().to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge from hash failed: {e}"),
                )
            })?;

        // Read back with async I/O
        tokio::fs::read(temp_file.path()).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to read smudged file: {e}"),
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
        use cas_client::{FileProvider, SeekingOutputProvider};
        self.downloader
            .smudge_file_from_hash(
                hash,
                output_path.to_string_lossy().into(),
                SeekingOutputProvider::File(FileProvider::new(output_path.to_path_buf())),
                None,
                None,
            )
            .await
            .map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("Smudge from hash to file failed: {e}"),
                )
            })?;

        Ok(())
    }
}

#[cfg(feature = "xet-storage")]
#[async_trait]
impl StorageBackend for XetStorage {
    async fn clean_file(&self, path: &Path) -> Result<String> {
        let (xet_info, _metrics) =
            data::data_client::clean_file(self.upload_session.clone(), path, "")
                .await
                .map_err(|e| {
                    XetError::new(XetErrorKind::UploadFailed, format!("Clean failed: {e}"))
                })?;

        xet_info.as_pointer_file().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Failed to serialize pointer: {e}"),
            )
        })
    }

    fn is_pointer(&self, content: &str) -> bool {
        serde_json::from_str::<XetFileInfo>(content).is_ok()
    }

    async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        XetStorage::clean_bytes(self, data).await
    }

    async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Result<()> {
        XetStorage::smudge_file(self, pointer, output_path).await
    }

    async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>> {
        XetStorage::smudge_bytes(self, pointer).await
    }

    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>> {
        XetStorage::smudge_from_hash(self, hash).await
    }

    async fn smudge_from_hash_to_file(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
    ) -> Result<()> {
        XetStorage::smudge_from_hash_to_file(self, hash, output_path).await
    }
}
