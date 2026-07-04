//! Content-addressed store: filesystem-backed XET reconstruction.
//!
//! This is the read/write core extracted from the `cas-serve` binary so it can
//! be reused **in-process** by embedders (e.g. the registry service's `getBlob`)
//! without spawning a subprocess or reimplementing reconstruction.
//!
//! Reconstruction walks the file's `.mdb` reconstruction shard, fetches each
//! referenced xorb, and concatenates the segment bytes — the same algorithm the
//! binary serves over its NDJSON `GetFile` protocol.

use crate::protocol::Request;
use crate::shard::Shard;
use std::path::{Path, PathBuf};

/// Errors from store reconstruction.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("invalid hash: {0}")]
    InvalidHash(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("corrupt shard: {0}")]
    CorruptShard(String),
    #[error("io error: {0}")]
    Io(String),
}

/// Result of ingesting bytes via [`CasStore::put_file_bytes`].
#[derive(Debug, Clone)]
pub struct PutResult {
    /// Server-computed XET merkle root (hex) of the ingested bytes. Authoritative
    /// — derived from the actual content, never supplied by the caller.
    pub merkle: String,
    /// Hex xorb hashes forming the reconstruction set for this content.
    pub xorb_hashes: Vec<String>,
    /// Bytes newly written to the store after content-addressed dedup
    /// (0 if every xorb was already present — fully deduplicated).
    pub bytes_stored: u64,
}

/// Storage path resolution order (matches the `cas-serve` binary):
/// 1. `CAS_STORAGE` environment variable
/// 2. `XET_CACHE_DIR` environment variable
/// 3. `~/.cache/xet/` (default)
pub fn resolve_storage_path() -> PathBuf {
    if let Ok(path) = std::env::var("CAS_STORAGE") {
        return PathBuf::from(shellexpand::tilde(&path).into_owned());
    }
    if let Ok(path) = std::env::var("XET_CACHE_DIR") {
        return PathBuf::from(shellexpand::tilde(&path).into_owned());
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("xet")
}

/// Filesystem-backed content-addressed store.
///
/// Storage layout (identical to the `cas-serve` binary):
/// - `xorbs/default.{xorb_hash}` — raw xorb bytes (concatenated chunks).
/// - `shards/{file_hash}` — reconstruction shard (`.mdb` binary).
#[derive(Debug, Clone)]
pub struct CasStore {
    storage_path: PathBuf,
}

impl CasStore {
    /// Open a store rooted at `storage_path`.
    pub fn new(storage_path: impl AsRef<Path>) -> Self {
        Self {
            storage_path: storage_path.as_ref().to_path_buf(),
        }
    }

    /// Open a store using the standard env/default path resolution.
    pub fn from_env() -> Self {
        Self::new(resolve_storage_path())
    }

    /// Path for a raw xorb, keyed by its hex xorb hash.
    fn xorb_path(&self, xorb_hash_hex: &str) -> PathBuf {
        self.storage_path
            .join("xorbs")
            .join(format!("default.{xorb_hash_hex}"))
    }

    /// Path for a reconstruction shard, keyed by the file's hex merkle hash.
    fn shard_path(&self, file_hash_hex: &str) -> PathBuf {
        self.storage_path.join("shards").join(file_hash_hex)
    }

    /// True if the store can serve this content address (shard or raw blob).
    pub fn exists(&self, hash: &str) -> bool {
        self.shard_path(hash).exists() || self.xorb_path(hash).exists()
    }

    /// Reassemble a file from its reconstruction shard (or raw-blob fallback) and
    /// return the original bytes. This is the in-process equivalent of the
    /// binary's `GetFile` handler.
    pub async fn get_file_bytes(&self, hash: &str) -> Result<Vec<u8>, StoreError> {
        let _file_hash =
            Request::parse_hash(hash).map_err(StoreError::InvalidHash)?;

        // 1. Reconstruction shard (chunked / multi-xorb path).
        let shard_path = self.shard_path(hash);
        if shard_path.exists() {
            return self.reassemble_from_shard(hash, &shard_path).await;
        }

        // 2. Fall back to a directly-stored raw blob (single-chunk file uploaded
        //    via the legacy whole-blob path, where the xorb hash == file hash).
        let xorb_path = self.xorb_path(hash);
        match tokio::fs::read(&xorb_path).await {
            Ok(data) => Ok(data),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(StoreError::NotFound(format!("File not found: {hash}")))
            }
            Err(e) => Err(StoreError::Io(format!("Failed to read file: {e}"))),
        }
    }

    /// Ingest raw bytes: chunk (Gearhash CDC), aggregate into xorbs, build the
    /// reconstruction shard, and store both content-addressed. The XET merkle
    /// root is computed HERE from the actual bytes and returned in
    /// [`PutResult::merkle`] — the caller never supplies it. Existing xorbs are
    /// skipped (global content-addressed dedup), so [`PutResult::bytes_stored`]
    /// counts only newly-written xorb payload.
    ///
    /// This is the in-process, library form of the binary's `UploadFile`
    /// handler; both go through this single implementation.
    pub async fn put_file_bytes(&self, data: &[u8]) -> Result<PutResult, StoreError> {
        use crate::chunker;
        use crate::shard::Shard;

        // Server-side chunk + merkle: the file_hash is derived from the bytes,
        // not asserted by the caller (this is what closes the OID-planting vector
        // for the authenticated write path — see hyprstream registry putBlob).
        let chunks = chunker::chunk_all(data);
        let (shard, xorbs) = Shard::from_chunks(&chunks);

        let xorbs_dir = self.storage_path.join("xorbs");
        let shards_dir = self.storage_path.join("shards");
        tokio::fs::create_dir_all(&xorbs_dir)
            .await
            .map_err(|e| StoreError::Io(format!("create xorbs dir: {e}")))?;
        tokio::fs::create_dir_all(&shards_dir)
            .await
            .map_err(|e| StoreError::Io(format!("create shards dir: {e}")))?;

        // Store each xorb content-addressed. An identical xorb already present is
        // not an error (dedup): skip the write and don't count it.
        let mut xorb_hashes = Vec::with_capacity(xorbs.len());
        let mut bytes_stored: u64 = 0;
        for (xorb_hash, bytes) in &xorbs {
            let hex = xorb_hash.hex();
            let path = self.xorb_path(&hex);
            if !path.exists() {
                tokio::fs::write(&path, bytes)
                    .await
                    .map_err(|e| StoreError::Io(format!("write xorb {hex}: {e}")))?;
                bytes_stored += bytes.len() as u64;
            }
            xorb_hashes.push(hex);
        }

        // Reconstruction shard (`.mdb`), keyed by the file merkle. Small metadata,
        // identical for identical content, so an unconditional (re)write is fine.
        let shard_path = self.shard_path(&shard.file_hash);
        tokio::fs::write(&shard_path, shard.to_bytes())
            .await
            .map_err(|e| StoreError::Io(format!("write shard {}: {e}", shard.file_hash)))?;

        Ok(PutResult {
            merkle: shard.file_hash,
            xorb_hashes,
            bytes_stored,
        })
    }

    /// Read the raw bytes of a single stored xorb, keyed by its hex xorb hash.
    ///
    /// Unlike [`CasStore::get_file_bytes`] this does NOT reconstruct a file: it
    /// returns the exact on-disk `xorbs/default.{hash}` payload. This backs the
    /// HuggingFace-XET `GET /get_xorb/{hash}/` wire route, which fetches an
    /// individual xorb (not a reconstructed file). Callers wanting a byte
    /// sub-range should slice the returned buffer (the store keeps whole xorbs).
    pub async fn read_xorb(&self, xorb_hash_hex: &str) -> Result<Vec<u8>, StoreError> {
        let _ = Request::parse_hash(xorb_hash_hex).map_err(StoreError::InvalidHash)?;
        let path = self.xorb_path(xorb_hash_hex);
        match tokio::fs::read(&path).await {
            Ok(data) => Ok(data),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(StoreError::NotFound(format!("Xorb not found: {xorb_hash_hex}")))
            }
            Err(e) => Err(StoreError::Io(format!("Failed to read xorb: {e}"))),
        }
    }

    /// Load the `.mdb` shard, fetch each referenced xorb, and concatenate the
    /// segment bytes to reconstruct the file.
    async fn reassemble_from_shard(
        &self,
        file_hash: &str,
        shard_path: &Path,
    ) -> Result<Vec<u8>, StoreError> {
        let mdb_bytes = match tokio::fs::read(shard_path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(StoreError::NotFound(format!("Shard not found: {file_hash}")))
            }
            Err(e) => return Err(StoreError::Io(format!("Failed to read shard: {e}"))),
        };

        let file_hash_parsed =
            Request::parse_hash(file_hash).map_err(StoreError::InvalidHash)?;

        let segments = Shard::segments(&mdb_bytes, &file_hash_parsed)
            .map_err(|e| StoreError::CorruptShard(e.to_string()))?;

        let expected_len: u64 = segments.iter().map(|s| s.byte_len).sum();
        let mut out = Vec::with_capacity(expected_len as usize);
        for seg in &segments {
            let xorb_hex = seg.xorb_hash.hex();
            let xorb_path = self.xorb_path(&xorb_hex);
            let xorb_bytes = match tokio::fs::read(&xorb_path).await {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(StoreError::NotFound(format!(
                        "Missing xorb {xorb_hex} for file {file_hash}"
                    )))
                }
                Err(e) => {
                    return Err(StoreError::Io(format!(
                        "Failed to read xorb {xorb_hex}: {e}"
                    )))
                }
            };
            // Raw concatenated xorbs: every segment spans the whole xorb
            // (chunk_start=0). Slice defensively for partial spans.
            let len = (seg.byte_len as usize).min(xorb_bytes.len());
            out.extend_from_slice(&xorb_bytes[..len]);
        }

        if out.len() as u64 != expected_len {
            return Err(StoreError::CorruptShard(format!(
                "Reconstructed {} bytes but shard expected {expected_len} for {file_hash}",
                out.len()
            )));
        }

        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)] // panicking is correct in unit tests
mod tests {
    use super::*;

    // A payload large enough to exercise CDC into multiple chunks/xorbs.
    fn payload(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add((i as u8).wrapping_mul(31))).collect()
    }

    #[tokio::test]
    async fn put_then_get_round_trips_and_computes_merkle() {
        let dir = tempfile::tempdir().unwrap();
        let store = CasStore::new(dir.path());
        let original = payload(7, 512 * 1024);

        let put = store.put_file_bytes(&original).await.unwrap();
        // Merkle is server-computed and non-empty; content is content-addressable.
        assert!(!put.merkle.is_empty(), "server must compute a merkle");
        assert!(!put.xorb_hashes.is_empty());
        assert!(put.bytes_stored > 0, "first upload writes new bytes");

        // The store can now reconstruct the exact original bytes by that merkle.
        let got = store.get_file_bytes(&put.merkle).await.unwrap();
        assert_eq!(got, original);
    }

    #[tokio::test]
    async fn merkle_is_deterministic_and_reupload_dedupes() {
        let dir = tempfile::tempdir().unwrap();
        let store = CasStore::new(dir.path());
        let original = payload(42, 300 * 1024);

        let first = store.put_file_bytes(&original).await.unwrap();
        let second = store.put_file_bytes(&original).await.unwrap();

        // Same bytes ⇒ same merkle (content-addressed, caller-independent).
        assert_eq!(first.merkle, second.merkle);
        assert_eq!(first.xorb_hashes, second.xorb_hashes);
        // Re-upload of identical content writes nothing new (global dedup).
        assert_eq!(second.bytes_stored, 0, "re-upload must fully dedup");
    }

    #[tokio::test]
    async fn distinct_content_yields_distinct_merkle() {
        let dir = tempfile::tempdir().unwrap();
        let store = CasStore::new(dir.path());

        let a = store.put_file_bytes(&payload(1, 200 * 1024)).await.unwrap();
        let b = store.put_file_bytes(&payload(2, 200 * 1024)).await.unwrap();
        assert_ne!(a.merkle, b.merkle);
    }
}
