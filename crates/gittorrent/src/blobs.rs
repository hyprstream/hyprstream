//! iroh-blobs-backed content-addressed object store (F1, #899).
//!
//! Evaluation + first-integration spike for Track F of the gittorrent →
//! `hyprstream-p2p` convergence (epic #880 §2). Provides a LOCAL (in-process,
//! no network) put/get pair preserving the consumer contract used by
//! git-xet-filter and git2db:
//!
//! ```text
//! put_object(Vec<u8>)      -> Sha256Hash
//! get_object(&Sha256Hash)  -> Option<Vec<u8>>
//! ```
//!
//! # SHA256 ↔ BLAKE3 bridge
//!
//! iroh-blobs addresses content by BLAKE3, while the consumer contract is
//! keyed by SHA256 (XET `MerkleHash` maps 1:1 onto `Sha256Hash`). Rather than
//! re-keying the consumers, this module keeps the SHA256 facade and maintains
//! a `sha256 → blake3` index at put time — the facade-preserving shape the
//! epic ratified. The index is a *locator only*, never a trust root: bytes
//! fetched through it are BLAKE3-verified by iroh-blobs on read and then
//! re-verified here against the requested SHA256, so a corrupted or poisoned
//! index entry can fail a lookup but can never serve wrong bytes.
//!
//! For the persistent (`FsStore`) backend the index is an append-only sidecar
//! file (`sha256-blake3.index`, one `sha256hex blake3hex` line per object)
//! loaded at open. Malformed or unverifiable lines are skipped with a warning
//! (the object merely becomes unreachable via SHA256 until re-put).
//!
//! # Scope
//!
//! Local store only. Networked fetch (provider discovery via the at9p
//! mainline locator + remote transfer) is F2 (#900). Wiring this store under
//! `GitTorrentService::{put,get}_object` in place of the libp2p DHT is also
//! F2; retiring libp2p is F3 (#901).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use iroh_blobs::store::fs::FsStore;
use iroh_blobs::store::mem::MemStore;
use iroh_blobs::Hash as Blake3Hash;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::crypto::hash::{sha256_git, verify_sha256};
use crate::{Error, Result, Sha256Hash};

/// Sidecar index file name (relative to the `FsStore` root).
const INDEX_FILE: &str = "sha256-blake3.index";

enum Backend {
    Mem(MemStore),
    Fs(FsStore),
}

impl Backend {
    fn store(&self) -> &iroh_blobs::api::Store {
        match self {
            Backend::Mem(s) => s,
            Backend::Fs(s) => s,
        }
    }
}

/// Content-addressed object store backed by iroh-blobs, keyed by SHA256.
///
/// See the module docs for the SHA256 ↔ BLAKE3 bridging model.
pub struct IrohBlobStore {
    backend: Backend,
    /// sha256 → blake3 mapping (locator only — reads re-verify SHA256).
    index: Mutex<HashMap<Sha256Hash, Blake3Hash>>,
    /// Append-only index persistence (fs backend only).
    index_path: Option<PathBuf>,
}

impl IrohBlobStore {
    /// Create an ephemeral, in-memory store (tests, throwaway caches).
    pub fn new_memory() -> Self {
        Self {
            backend: Backend::Mem(MemStore::new()),
            index: Mutex::new(HashMap::new()),
            index_path: None,
        }
    }

    /// Open (or create) a persistent store rooted at `root`.
    ///
    /// Blob bytes live in the iroh-blobs `FsStore` under `root`; the
    /// sha256→blake3 index is an append-only sidecar file next to it.
    pub async fn open_fs(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        tokio::fs::create_dir_all(root).await?;
        let store = FsStore::load(root)
            .await
            .map_err(|e| Error::other(format!("failed to open iroh-blobs store: {e}")))?;

        let index_path = root.join(INDEX_FILE);
        let index = load_index(&index_path).await?;

        Ok(Self {
            backend: Backend::Fs(store),
            index: Mutex::new(index),
            index_path: Some(index_path),
        })
    }

    /// Store an object, returning its SHA256 hash (the consumer-facing key).
    ///
    /// The bytes are added to the iroh-blobs store (BLAKE3-addressed, tagged
    /// so garbage collection cannot reclaim them) and the sha256→blake3
    /// mapping is recorded.
    pub async fn put_object(&self, data: Vec<u8>) -> Result<Sha256Hash> {
        let sha256 = sha256_git(&data)?;

        let tag = self
            .backend
            .store()
            .blobs()
            .add_bytes(data)
            .await
            .map_err(|e| Error::other(format!("iroh-blobs add failed: {e}")))?;

        let mut index = self.index.lock().await;
        if index.insert(sha256.clone(), tag.hash).is_none() {
            if let Some(path) = &self.index_path {
                append_index_entry(path, &sha256, &tag.hash).await?;
            }
        }
        drop(index);

        tracing::debug!("stored object sha256={} blake3={}", sha256, tag.hash);
        Ok(sha256)
    }

    /// Fetch an object by its SHA256 hash.
    ///
    /// Returns `Ok(None)` when the hash is unknown. Returns an error if the
    /// index resolves but the retrieved bytes do not hash back to the
    /// requested SHA256 (poisoned/corrupt mapping — never served).
    pub async fn get_object(&self, hash: &Sha256Hash) -> Result<Option<Vec<u8>>> {
        let blake3 = {
            let index = self.index.lock().await;
            match index.get(hash) {
                Some(b3) => *b3,
                None => return Ok(None),
            }
        };

        let bytes = match self.backend.store().blobs().get_bytes(blake3).await {
            Ok(bytes) => bytes,
            Err(e) => {
                // Index entry without a backing blob: unreachable object, not
                // a hard error for a cache-shaped lookup API.
                tracing::warn!("indexed blob {blake3} missing from store for sha256 {hash}: {e}");
                return Ok(None);
            }
        };

        let data = bytes.to_vec();
        if !verify_sha256(&data, hash)? {
            return Err(Error::other(format!(
                "sha256 verification failed for object {hash} (index maps to blake3 {blake3})"
            )));
        }
        Ok(Some(data))
    }

    /// Resolve the BLAKE3 hash backing a SHA256 key, if known.
    ///
    /// This is the seam F2 (#900) uses to announce/locate providers: the
    /// locator plane finds peers, iroh-blobs transfers by BLAKE3.
    pub async fn blake3_of(&self, hash: &Sha256Hash) -> Option<Blake3Hash> {
        self.index.lock().await.get(hash).copied()
    }

    /// Number of SHA256-addressable objects.
    pub async fn len(&self) -> usize {
        self.index.lock().await.len()
    }

    /// Whether the store has no SHA256-addressable objects.
    pub async fn is_empty(&self) -> bool {
        self.index.lock().await.is_empty()
    }

    /// Shut down the underlying iroh-blobs store, flushing pending writes.
    ///
    /// Required before the same `FsStore` root can be reopened (the fs
    /// backend holds a database lock until its actor shuts down); dropping
    /// the store alone does not release it promptly.
    pub async fn shutdown(&self) -> Result<()> {
        self.backend
            .store()
            .shutdown()
            .await
            .map_err(|e| Error::other(format!("iroh-blobs shutdown failed: {e}")))
    }
}

/// Load the sidecar index, skipping (with a warning) any malformed lines.
async fn load_index(path: &Path) -> Result<HashMap<Sha256Hash, Blake3Hash>> {
    let mut index = HashMap::new();
    let contents = match tokio::fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(index),
        Err(e) => return Err(e.into()),
    };

    for (lineno, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed = line.split_once(' ').and_then(|(sha_hex, b3_hex)| {
            let sha = Sha256Hash::new(sha_hex).ok()?;
            let b3 = b3_hex.parse::<Blake3Hash>().ok()?;
            Some((sha, b3))
        });
        match parsed {
            Some((sha, b3)) => {
                index.insert(sha, b3);
            }
            None => {
                tracing::warn!(
                    "skipping malformed index line {} in {}",
                    lineno + 1,
                    path.display()
                );
            }
        }
    }
    Ok(index)
}

/// Append one `sha256hex blake3hex` line to the sidecar index.
async fn append_index_entry(path: &Path, sha256: &Sha256Hash, blake3: &Blake3Hash) -> Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(format!("{sha256} {blake3}\n").as_bytes())
        .await?;
    file.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_memory_roundtrip() -> Result<()> {
        let store = IrohBlobStore::new_memory();
        let data = b"hello iroh-blobs".to_vec();

        let hash = store.put_object(data.clone()).await?;
        // The consumer-facing key is plain SHA256 of the bytes.
        assert_eq!(hash, sha256_git(&data)?);

        let fetched = store.get_object(&hash).await?;
        assert_eq!(fetched.as_deref(), Some(data.as_slice()));
        assert_eq!(store.len().await, 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_unknown_hash_is_none() -> Result<()> {
        let store = IrohBlobStore::new_memory();
        let unknown = sha256_git(b"never stored")?;
        assert!(store.get_object(&unknown).await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_put_is_idempotent() -> Result<()> {
        let store = IrohBlobStore::new_memory();
        let data = b"same bytes twice".to_vec();
        let h1 = store.put_object(data.clone()).await?;
        let h2 = store.put_object(data.clone()).await?;
        assert_eq!(h1, h2);
        assert_eq!(store.len().await, 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_blake3_mapping_exposed() -> Result<()> {
        let store = IrohBlobStore::new_memory();
        let data = b"mapped".to_vec();
        let sha = store.put_object(data.clone()).await?;

        let b3 = store
            .blake3_of(&sha)
            .await
            .ok_or_else(|| Error::not_found("missing mapping"))?;
        // Hash::new computes BLAKE3 of the bytes — the mapping must point at
        // the BLAKE3 of the stored content.
        assert_eq!(b3, Blake3Hash::new(data.as_slice()));
        Ok(())
    }

    #[tokio::test]
    async fn test_fs_persistence_across_reopen() -> Result<()> {
        let dir = TempDir::new()?;
        let data = b"persisted across reopen".to_vec();

        let sha = {
            let store = IrohBlobStore::open_fs(dir.path()).await?;
            let sha = store.put_object(data.clone()).await?;
            store.shutdown().await?;
            sha
        };

        let store = IrohBlobStore::open_fs(dir.path()).await?;
        let fetched = store.get_object(&sha).await?;
        assert_eq!(fetched.as_deref(), Some(data.as_slice()));
        store.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_poisoned_index_never_serves_wrong_bytes() -> Result<()> {
        let dir = TempDir::new()?;
        let a = b"object a".to_vec();
        let b = b"object b".to_vec();

        let (sha_a, sha_b) = {
            let store = IrohBlobStore::open_fs(dir.path()).await?;
            let pair = (store.put_object(a.clone()).await?, store.put_object(b.clone()).await?);
            store.shutdown().await?;
            pair
        };

        // Poison the sidecar: cross-wire a's sha256 to b's blake3.
        let index_path = dir.path().join(INDEX_FILE);
        let contents = tokio::fs::read_to_string(&index_path).await?;
        let mut b3 = HashMap::new();
        for line in contents.lines() {
            if let Some((sha, blake)) = line.split_once(' ') {
                b3.insert(sha.to_owned(), blake.to_owned());
            }
        }
        let poisoned = format!(
            "{} {}\n{} {}\n",
            sha_a,
            b3[sha_b.as_str()],
            sha_b,
            b3[sha_a.as_str()]
        );
        tokio::fs::write(&index_path, poisoned).await?;

        let store = IrohBlobStore::open_fs(dir.path()).await?;
        // Both lookups resolve to real blobs, but the bytes fail SHA256
        // re-verification — the store must error, never serve wrong bytes.
        assert!(store.get_object(&sha_a).await.is_err());
        assert!(store.get_object(&sha_b).await.is_err());
        store.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_malformed_index_lines_skipped() -> Result<()> {
        let dir = TempDir::new()?;
        let data = b"good object".to_vec();

        let sha = {
            let store = IrohBlobStore::open_fs(dir.path()).await?;
            let sha = store.put_object(data.clone()).await?;
            store.shutdown().await?;
            sha
        };

        // Prepend garbage lines to the sidecar.
        let index_path = dir.path().join(INDEX_FILE);
        let contents = tokio::fs::read_to_string(&index_path).await?;
        tokio::fs::write(
            &index_path,
            format!("not an index line\ndeadbeef tooshort\n{contents}"),
        )
        .await?;

        let store = IrohBlobStore::open_fs(dir.path()).await?;
        let fetched = store.get_object(&sha).await?;
        assert_eq!(fetched.as_deref(), Some(data.as_slice()));
        assert_eq!(store.len().await, 1);
        store.shutdown().await?;
        Ok(())
    }
}
