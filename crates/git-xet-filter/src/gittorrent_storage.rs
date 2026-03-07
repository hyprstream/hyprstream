//! GittorrentStorage — StorageBackend implementation for P2P distribution.
//!
//! Fetches XET objects via gittorrent's DHT network with fallback to HTTPS origin.
//! MerkleHash (32-byte SHA256) maps directly to gittorrent's Sha256Hash.

#[cfg(feature = "xet-storage")]
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Result, XetError, XetErrorKind};

/// P2P storage backend using gittorrent DHT.
///
/// Attempts to fetch objects from DHT peers first, falling back to an optional
/// HTTPS origin (typically XetStorage).
pub struct GittorrentStorage {
    service: Arc<gittorrent::service::GitTorrentService>,
    fallback: Option<Box<dyn super::storage::StorageBackend>>,
}

impl GittorrentStorage {
    /// Create a new gittorrent storage backend.
    ///
    /// # Arguments
    /// * `service` - The gittorrent service instance (DHT + object cache)
    /// * `fallback` - Optional fallback storage (e.g., XetStorage for HTTPS origin)
    pub fn new(
        service: Arc<gittorrent::service::GitTorrentService>,
        fallback: Option<Box<dyn super::storage::StorageBackend>>,
    ) -> Self {
        Self { service, fallback }
    }
}

#[cfg(feature = "xet-storage")]
#[async_trait]
impl super::storage::StorageBackend for GittorrentStorage {
    async fn clean_file(&self, path: &Path) -> Result<String> {
        // Read file and delegate to clean_bytes
        let data = tokio::fs::read(path).await.map_err(|e| {
            XetError::new(XetErrorKind::IoError, format!("Failed to read {}: {e}", path.display()))
        })?;
        self.clean_bytes(&data).await
    }

    fn is_pointer(&self, content: &str) -> bool {
        // Only match gittorrent-specific pointers, not standard XET pointers
        content.contains("\"xet\":\"gittorrent\"") || content.contains("\"xet\": \"gittorrent\"")
    }

    async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        // Store in gittorrent DHT
        let hash = self
            .service
            .put_object(data.to_vec())
            .await
            .map_err(|e| XetError::new(XetErrorKind::UploadFailed, e.to_string()))?;

        // Return a pointer referencing the SHA256 hash
        Ok(format!(
            r#"{{"xet":"gittorrent","sha256":"{}","size":{}}}"#,
            hash.as_str(),
            data.len()
        ))
    }

    async fn smudge_file(&self, pointer: &str, output_path: &Path) -> Result<()> {
        let data = self.smudge_bytes(pointer).await?;
        tokio::fs::write(output_path, &data).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to write {}: {e}", output_path.display()),
            )
        })
    }

    async fn smudge_bytes(&self, pointer: &str) -> Result<Vec<u8>> {
        // Parse hash from pointer JSON
        let hash = parse_sha256_from_pointer(pointer)?;
        let merkle = merklehash::MerkleHash::from_hex(&hash).map_err(|e| {
            XetError::new(XetErrorKind::InvalidPointer, format!("Invalid hash: {e}"))
        })?;
        self.smudge_from_hash(&merkle).await
    }

    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>> {
        // MerkleHash is 32-byte SHA256, same as gittorrent::Sha256Hash
        let gt_hash = gittorrent::Sha256Hash::from_bytes(hash.as_bytes()).map_err(|e| {
            XetError::new(XetErrorKind::DownloadFailed, format!("Hash conversion: {e}"))
        })?;

        // Try P2P first
        match self.service.get_object(&gt_hash).await {
            Ok(Some(data)) => return Ok(data),
            Ok(None) => {
                tracing::debug!("Object {} not found in DHT, trying fallback", hash);
            }
            Err(e) => {
                tracing::warn!("DHT fetch failed for {}: {e}", hash);
            }
        }

        // Fallback to HTTPS origin
        if let Some(ref fb) = self.fallback {
            return fb.smudge_from_hash(hash).await;
        }

        Err(XetError::new(
            XetErrorKind::DownloadFailed,
            format!("Object {hash} not found in DHT or fallback"),
        ))
    }

    async fn smudge_from_hash_to_file(
        &self,
        hash: &merklehash::MerkleHash,
        output_path: &Path,
    ) -> Result<()> {
        let data = self.smudge_from_hash(hash).await?;
        tokio::fs::write(output_path, &data).await.map_err(|e| {
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to write {}: {e}", output_path.display()),
            )
        })
    }
}

/// Extract SHA256 hash from a pointer JSON string.
fn parse_sha256_from_pointer(pointer: &str) -> Result<String> {
    // Simple extraction without pulling in a JSON parser dependency
    // Format: {"xet":"gittorrent","sha256":"<hex>","size":<n>}
    let start = pointer.find("\"sha256\":\"").ok_or_else(|| {
        XetError::new(XetErrorKind::InvalidPointer, "Missing sha256 field")
    })? + 10;
    let end = pointer[start..].find('"').ok_or_else(|| {
        XetError::new(XetErrorKind::InvalidPointer, "Unterminated sha256 value")
    })? + start;
    Ok(pointer[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sha256_from_pointer() -> Result<()> {
        let pointer =
            r#"{"xet":"gittorrent","sha256":"a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2","size":1024}"#;
        let hash = parse_sha256_from_pointer(pointer)?;
        assert_eq!(
            hash,
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        );
        Ok(())
    }

    #[test]
    fn test_parse_invalid_pointer() {
        assert!(parse_sha256_from_pointer("not a pointer").is_err());
    }
}
