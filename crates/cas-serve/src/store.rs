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
