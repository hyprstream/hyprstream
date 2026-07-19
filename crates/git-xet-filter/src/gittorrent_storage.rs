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

    /// Fetch reconstruction bytes for a Merkle hash without committing any
    /// CID binding. The returned flag is `true` when the bytes did not come
    /// from the already-validated local p2p store and should be persisted by
    /// the caller only after end-to-end validation — i.e. when they arrived
    /// via the HTTPS fallback OR via an uncommitted provider hit on a
    /// non-self-verifying (XET Merkle) CID. The flag is `false` only when
    /// the service reports the bytes as committed (local store hit or a
    /// self-verifying fetch that the service committed itself).
    ///
    /// Provenance is taken structurally from
    /// [`GitTorrentService::get_object_by_cid_with_provenance`]; it is never
    /// assumed from the mere fact of a P2P hit, because the service returns
    /// provider-fetched XET Merkle bytes without committing them.
    ///
    /// This is an internal helper rather than a [`StorageBackend`] method:
    /// the trait exposes only the `smudge_*` surface, and the extra
    /// `uncommitted` flag is meaningful solely to this backend's callers.
    async fn fetch_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<(Vec<u8>, bool)> {
        let cid = hyprstream_p2p::ContentCid::xet_merkle(hash.as_bytes()).map_err(|e| {
            XetError::new(
                XetErrorKind::DownloadFailed,
                format!("CID encoding failed: {e}"),
            )
        })?;

        // Try P2P first. Propagate commit provenance rather than assuming
        // every P2P hit is already-validated: a non-self-verifying CID
        // (XET Merkle addresses the reconstruction DAG, not the raw file)
        // is returned uncommitted by the service, and only the caller's
        // end-to-end size + SHA-256 check can license persisting it.
        match self.service.get_object_by_cid_with_provenance(&cid).await {
            Ok(Some(fetch)) => return Ok((fetch.data, !fetch.committed)),
            Ok(None) => {
                tracing::debug!("Object {} not found in DHT, trying fallback", hash);
            }
            Err(e) => {
                tracing::warn!("DHT fetch failed for {}: {e}", hash);
            }
        }

        // Fallback to HTTPS origin; caching is deferred to the caller until
        // the pointer's size and SHA-256 checks pass.
        if let Some(ref fb) = self.fallback {
            let data = fb.smudge_from_hash(hash).await?;
            return Ok((data, true));
        }

        Err(XetError::new(
            XetErrorKind::DownloadFailed,
            format!("Object {hash} not found in DHT or fallback"),
        ))
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
        let (data, uncommitted) = self.fetch_from_hash(&merkle).await?;
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
        if uncommitted {
            // The bytes passed the pointer's end-to-end size + SHA-256 checks;
            // only now is it safe to bind them to the Merkle CID in the p2p
            // object plane (a Merkle CID cannot verify raw bytes on its own).
            let cid = hyprstream_p2p::ContentCid::xet_merkle(merkle.as_bytes()).map_err(|e| {
                XetError::new(
                    XetErrorKind::DownloadFailed,
                    format!("CID encoding failed: {e}"),
                )
            })?;
            if let Err(e) = self.service.put_object_by_cid(cid, data.clone()).await {
                tracing::warn!("Failed to cache XET object in gittorrent: {e}");
            }
        }
        Ok(data)
    }

    async fn smudge_from_hash(&self, hash: &merklehash::MerkleHash) -> Result<Vec<u8>> {
        // Hash-only callers carry no pointer metadata (size, SHA-256) to
        // validate against, so fetched bytes are never committed to the p2p
        // plane on this path — only `smudge_bytes` caches, after validation.
        Ok(self.fetch_from_hash(hash).await?.0)
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

/// Provider-side regression coverage (#1115). These tests drive a real
/// `GitTorrentService` through its public `testing` mocks so the XET smudge
/// path — provenance propagation, end-to-end validation, and the persist
/// gate — flows through production code.
#[cfg(all(test, feature = "gittorrent-transport"))]
mod provider_fetch_tests {
    use super::*;
    use crate::storage::StorageBackend;
    use hyprstream_p2p::locator::PeerContact;
    use hyprstream_p2p::service::testing::{
        new_service_with_mocks, test_config, MockBlobFetcher, MockObjectLocator,
    };
    use hyprstream_p2p::ContentCid;
    use std::net::{Ipv4Addr, SocketAddrV4};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn p2p_err(e: hyprstream_p2p::Error) -> XetError {
        XetError::new(XetErrorKind::DownloadFailed, e.to_string())
    }

    fn io_err(e: std::io::Error) -> XetError {
        XetError::new(XetErrorKind::IoError, e.to_string())
    }

    /// Build a storage backed by a mock object plane that advertises `data`
    /// under `merkle_bytes` from a single local provider.
    async fn storage_with_provider(
        temp: &TempDir,
        merkle_bytes: [u8; 32],
        data: Vec<u8>,
    ) -> Result<(
        GittorrentStorage,
        Arc<MockObjectLocator>,
        Arc<MockBlobFetcher>,
    )> {
        let locator = Arc::new(MockObjectLocator::default());
        let fetcher = Arc::new(MockBlobFetcher::default());
        let service =
            new_service_with_mocks(test_config(temp.path()), locator.clone(), fetcher.clone())
                .await
                .map_err(p2p_err)?;
        let cid = ContentCid::xet_merkle(&merkle_bytes).map_err(p2p_err)?;
        locator
            .add_provider_for_content(
                &cid,
                PeerContact::untrusted(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 6881)),
            )
            .await;
        fetcher.insert_raw(cid, data).await;
        Ok((
            GittorrentStorage::new(Arc::new(service), None),
            locator,
            fetcher,
        ))
    }

    fn pointer_json(merkle_hex: &str, sha256_hex: &str, size: u64) -> String {
        serde_json::json!({
            "xet": "gittorrent",
            "xet-merkle": merkle_hex,
            "sha256": sha256_hex,
            "size": size,
        })
        .to_string()
    }

    /// Regression for #1115: provider-fetched XET Merkle bytes that pass the
    /// pointer's end-to-end size + SHA-256 check MUST be persisted to the
    /// p2p object plane. Before the provenance fix, `fetch_from_hash` mapped
    /// every P2P hit to `uncommitted = false`, so `smudge_bytes` skipped the
    /// persist and the next fetch re-hit the provider.
    #[tokio::test]
    async fn provider_fetched_xet_bytes_are_persisted_after_validation() -> Result<()> {
        let temp = TempDir::new().map_err(io_err)?;
        let data = b"xet payload from provider".to_vec();
        let merkle_bytes = [0x7f_u8; 32];
        let merkle_hex: String = merkle_bytes.iter().map(|b| format!("{b:02x}")).collect();
        let sha256 = hyprstream_p2p::crypto::hash::sha256_git(&data).map_err(p2p_err)?;
        let (storage, _locator, fetcher) =
            storage_with_provider(&temp, merkle_bytes, data.clone()).await?;
        let pointer = pointer_json(&merkle_hex, &sha256.to_string(), data.len() as u64);

        // First smudge: provider fetch + end-to-end validation + persist.
        let smudged = storage.smudge_bytes(&pointer).await?;
        assert_eq!(smudged, data);
        assert_eq!(fetcher.calls().await, 1, "first smudge hits the provider");

        // Second smudge: bytes are now committed locally, so the provider is
        // NOT consulted again. Before the fix this re-fetched (calls == 2)
        // because the unvalidated provider bytes were never persisted.
        let smudged_again = storage.smudge_bytes(&pointer).await?;
        assert_eq!(smudged_again, data);
        assert_eq!(
            fetcher.calls().await,
            1,
            "validated provider bytes are served from the local store on retry"
        );
        Ok(())
    }

    /// The persistence gate must NOT fire when validation fails: a provider
    /// returning the wrong bytes is rejected, not persisted.
    #[tokio::test]
    async fn provider_fetched_xet_bytes_with_wrong_sha256_are_rejected() -> Result<()> {
        let temp = TempDir::new().map_err(io_err)?;
        // Same length as the declared real payload so the size check passes
        // and the SHA-256 check is the gate actually exercised here.
        let bogus = b"not real payload".to_vec();
        let real_sha =
            hyprstream_p2p::crypto::hash::sha256_git(b"the real payload").map_err(p2p_err)?;
        let merkle_bytes = [0x33_u8; 32];
        let merkle_hex: String = merkle_bytes.iter().map(|b| format!("{b:02x}")).collect();
        let (storage, _locator, _fetcher) =
            storage_with_provider(&temp, merkle_bytes, bogus).await?;
        // Pointer describes the real payload; provider serves something else.
        let pointer = pointer_json(&merkle_hex, &real_sha.to_string(), 16);

        match storage.smudge_bytes(&pointer).await {
            Err(err) => assert!(
                matches!(err.kind(), XetErrorKind::DownloadFailed),
                "validation failure must surface as DownloadFailed, got {:?}",
                err.kind()
            ),
            Ok(bytes) => {
                panic!("validation must reject provider bytes with wrong SHA-256; got {bytes:?}")
            }
        }
        Ok(())
    }
}
