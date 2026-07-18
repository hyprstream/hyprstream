//! GittorrentStorage — StorageBackend implementation for P2P distribution.
//!
//! Fetches XET objects via gittorrent's DHT network with fallback to HTTPS origin.
//! XET `MerkleHash` values cross into hyprstream-p2p as self-describing CIDs;
//! they are never reinterpreted as SHA-256.

#[cfg(feature = "xet-storage")]
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Result, XetError, XetErrorKind};

const XET_MERKLE_FIELD: &str = "xet-merkle";

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GittorrentPointer {
    xet: String,
    #[serde(rename = "xet-merkle")]
    xet_merkle: String,
    sha256: String,
    size: u64,
}

impl GittorrentPointer {
    fn parse(pointer: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(pointer).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid gittorrent pointer: {e}"),
            )
        })?;
        if parsed.xet != "gittorrent" {
            return Err(XetError::new(
                XetErrorKind::InvalidPointer,
                "Pointer is not a gittorrent XET pointer",
            ));
        }
        parsed.merkle_hash()?;
        hyprstream_p2p::Sha256Hash::new(&parsed.sha256).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid sha256 field: {e}"),
            )
        })?;
        Ok(parsed)
    }

    fn merkle_hash(&self) -> Result<merklehash::MerkleHash> {
        merklehash::MerkleHash::from_hex(&self.xet_merkle).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid {XET_MERKLE_FIELD} field: {e}"),
            )
        })
    }
}

/// P2P storage backend using gittorrent DHT.
///
/// Attempts to fetch objects from DHT peers first, falling back to an optional
/// HTTPS origin (typically XetStorage).
pub struct GittorrentStorage {
    service: Arc<hyprstream_p2p::service::GitTorrentService>,
    fallback: Option<Box<dyn super::storage::StorageBackend>>,
}

impl GittorrentStorage {
    /// Create a new gittorrent storage backend.
    ///
    /// # Arguments
    /// * `service` - The gittorrent service instance (DHT + object cache)
    /// * `fallback` - Optional fallback storage (e.g., XetStorage for HTTPS origin)
    pub fn new(
        service: Arc<hyprstream_p2p::service::GitTorrentService>,
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
            XetError::new(
                XetErrorKind::IoError,
                format!("Failed to read {}: {e}", path.display()),
            )
        })?;
        self.clean_bytes(&data).await
    }

    fn is_pointer(&self, content: &str) -> bool {
        GittorrentPointer::parse(content).is_ok()
    }

    async fn clean_bytes(&self, data: &[u8]) -> Result<String> {
        // XET's keyed-BLAKE3 Merkle root cannot be derived from the raw-file
        // SHA-256. Let the XET origin perform the real clean operation, then
        // carry that root into the P2P key and pointer extension verbatim.
        let fallback = self.fallback.as_ref().ok_or_else(|| {
            XetError::new(
                XetErrorKind::UploadFailed,
                "gittorrent clean requires an XET origin to compute xet-merkle",
            )
        })?;
        let xet_pointer = fallback.clean_bytes(data).await?;
        let xet_info: data::XetFileInfo = serde_json::from_str(&xet_pointer).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("XET origin returned an invalid pointer: {e}"),
            )
        })?;
        let merkle = xet_info.merkle_hash().map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("XET origin returned an invalid Merkle hash: {e}"),
            )
        })?;
        let cid = hyprstream_p2p::ContentCid::xet_merkle(merkle.as_bytes()).map_err(|e| {
            XetError::new(
                XetErrorKind::UploadFailed,
                format!("CID encoding failed: {e}"),
            )
        })?;

        self.service
            .put_object_by_cid(cid, data.to_vec())
            .await
            .map_err(|e| XetError::new(XetErrorKind::UploadFailed, e.to_string()))?;

        let sha256 = hyprstream_p2p::crypto::hash::sha256_git(data)
            .map_err(|e| XetError::new(XetErrorKind::UploadFailed, e.to_string()))?;
        serde_json::to_string(&GittorrentPointer {
            xet: "gittorrent".to_owned(),
            xet_merkle: merkle.to_string(),
            sha256: sha256.to_string(),
            size: data.len() as u64,
        })
        .map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Failed to serialize gittorrent pointer: {e}"),
            )
        })
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
        let pointer = GittorrentPointer::parse(pointer)?;
        let merkle = pointer.merkle_hash()?;
        let data = self.smudge_from_hash(&merkle).await?;
        if data.len() as u64 != pointer.size {
            return Err(XetError::new(
                XetErrorKind::DownloadFailed,
                format!(
                    "Size mismatch: expected {}, got {}",
                    pointer.size,
                    data.len()
                ),
            ));
        }
        let expected = hyprstream_p2p::Sha256Hash::new(pointer.sha256).map_err(|e| {
            XetError::new(
                XetErrorKind::InvalidPointer,
                format!("Invalid sha256 field: {e}"),
            )
        })?;
        if !hyprstream_p2p::crypto::hash::verify_sha256(&data, &expected)
            .map_err(|e| XetError::new(XetErrorKind::DownloadFailed, e.to_string()))?
        {
            return Err(XetError::new(
                XetErrorKind::DownloadFailed,
                "SHA-256 validation failed for gittorrent object",
            ));
        }
        Ok(data)
    }

    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>> {
        let cid = hyprstream_p2p::ContentCid::xet_merkle(hash.as_bytes()).map_err(|e| {
            XetError::new(
                XetErrorKind::DownloadFailed,
                format!("CID encoding failed: {e}"),
            )
        })?;

        // Try P2P first
        match self.service.get_object_by_cid(&cid).await {
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
            let data = fb.smudge_from_hash(hash).await?;
            if let Err(e) = self.service.put_object_by_cid(cid, data.clone()).await {
                tracing::warn!("Failed to cache XET object in gittorrent: {e}");
            }
            return Ok(data);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pointer_carries_distinct_hash_domains() -> Result<()> {
        let pointer = r#"{"xet":"gittorrent","xet-merkle":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789","size":1024}"#;
        let parsed = GittorrentPointer::parse(pointer)?;
        assert_eq!(
            parsed.xet_merkle,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        );
        assert_eq!(
            parsed.sha256,
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        );
        Ok(())
    }

    #[test]
    fn test_parse_invalid_pointer() {
        assert!(GittorrentPointer::parse("not a pointer").is_err());
        let legacy = r#"{"xet":"gittorrent","sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789","size":1}"#;
        assert!(GittorrentPointer::parse(legacy).is_err());
    }
}
