//! iroh-blobs-backed content-addressed object store (F1, #899).
//!
//! Evaluation + first-integration spike for Track F of the gittorrent →
//! `hyprstream-p2p` convergence (epic #880 §2). Provides a LOCAL (in-process,
//! no network) put/get pair preserving the consumer contract used by
//! git-xet-filter and git2db:
//!
//! ```text
//! put_object(Vec<u8>)              -> Sha256Hash
//! get_object(&Sha256Hash)          -> Option<Vec<u8>>
//! put_object_by_cid(ContentCid, ..) -> ()
//! get_object_by_cid(&ContentCid)    -> Option<Vec<u8>>
//! ```
//!
//! # CID → BLAKE3 locator index
//!
//! iroh-blobs addresses raw bytes by BLAKE3. Consumers address several domains:
//! raw Git objects use SHA-256, while XET file reconstruction DAGs use a keyed
//! BLAKE3 `MerkleHash`. Those are carried in a self-describing [`ContentCid`]
//! and indexed to the raw iroh-blobs hash without collapsing algorithm or
//! content-domain identity. SHA-256 methods remain compatibility wrappers that
//! construct a `git-raw + sha2-256` CID.
//!
//! For the persistent (`FsStore`) backend the index is an append-only sidecar
//! file (`sha256-blake3.index`, one `cid blake3hex` line per object) loaded at
//! open. Legacy `sha256hex blake3hex` entries are upgraded in memory to a Git
//! object CID. Malformed entries are skipped with a warning.
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
use crate::{ContentCid, Error, Result, Sha256Hash};

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

/// Content-addressed object store backed by iroh-blobs, keyed by canonical CIDs.
///
/// See the module docs for the CID → BLAKE3 locator model.
pub struct IrohBlobStore {
    backend: Backend,
    /// canonical content CID → raw-byte BLAKE3 mapping (locator only).
    index: Mutex<HashMap<ContentCid, Blake3Hash>>,
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
    /// CID→blake3 index is an append-only sidecar file next to it.
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
    /// so garbage collection cannot reclaim them) and the Git CID→blake3
    /// mapping is recorded.
    pub async fn put_object(&self, data: Vec<u8>) -> Result<Sha256Hash> {
        let sha256 = sha256_git(&data)?;
        self.put_verified_object(sha256.clone(), data).await?;
        Ok(sha256)
    }

    /// Store bytes under an expected SHA256 key after re-verifying them.
    ///
    /// F2 uses this after a remote iroh-blobs transfer: provider hints are
    /// untrusted, so fetched bytes are accepted only when they match the
    /// consumer-facing SHA256 requested by the facade.
    pub async fn put_verified_object(
        &self,
        expected_sha256: Sha256Hash,
        data: Vec<u8>,
    ) -> Result<Sha256Hash> {
        if !verify_sha256(&data, &expected_sha256)? {
            return Err(Error::other(format!(
                "sha256 verification failed for fetched object {expected_sha256}"
            )));
        }

        let cid = ContentCid::git_sha256(&expected_sha256)?;
        self.put_object_by_cid(cid, data).await?;
        Ok(expected_sha256)
    }

    /// Store bytes under an algorithm-preserving content CID.
    ///
    /// Raw Git SHA-256 CIDs are re-verified against the bytes here. An XET
    /// Merkle CID addresses the reconstruction DAG rather than the raw file,
    /// so its binding is established by the XET clean operation and the LFS
    /// SHA-256 remains the end-to-end raw-file integrity check.
    pub async fn put_object_by_cid(&self, cid: ContentCid, data: Vec<u8>) -> Result<()> {
        verify_raw_cid_when_applicable(&cid, &data)?;
        let raw_hash = Blake3Hash::new(&data);
        if let Some(existing) = self.index.lock().await.get(&cid).copied() {
            if existing != raw_hash {
                return Err(Error::other(format!(
                    "content CID {cid} is already bound to a different raw blob"
                )));
            }
        }

        let tag = self
            .backend
            .store()
            .blobs()
            .add_bytes(data)
            .await
            .map_err(|e| Error::other(format!("iroh-blobs add failed: {e}")))?;

        let mut index = self.index.lock().await;
        match index.get(&cid) {
            Some(existing) if *existing != tag.hash => {
                return Err(Error::other(format!(
                    "content CID {cid} is already bound to a different raw blob"
                )));
            }
            Some(_) => {}
            None => {
                index.insert(cid.clone(), tag.hash);
                if let Some(path) = &self.index_path {
                    append_index_entry(path, &cid, &tag.hash).await?;
                }
            }
        }
        drop(index);

        tracing::debug!(content_cid = %cid, blake3 = %tag.hash, "stored object");
        Ok(())
    }

    /// Fetch an object by its SHA256 hash.
    ///
    /// Returns `Ok(None)` when the hash is unknown. Returns an error if the
    /// index resolves but the retrieved bytes do not hash back to the
    /// requested SHA256 (poisoned/corrupt mapping — never served).
    pub async fn get_object(&self, hash: &Sha256Hash) -> Result<Option<Vec<u8>>> {
        let cid = ContentCid::git_sha256(hash)?;
        self.get_object_by_cid(&cid).await
    }

    /// Fetch an object by its self-describing content CID.
    pub async fn get_object_by_cid(&self, cid: &ContentCid) -> Result<Option<Vec<u8>>> {
        let blake3 = {
            let index = self.index.lock().await;
            match index.get(cid) {
                Some(b3) => *b3,
                None => return Ok(None),
            }
        };

        let bytes = match self.backend.store().blobs().get_bytes(blake3).await {
            Ok(bytes) => bytes,
            Err(e) => {
                // Index entry without a backing blob: unreachable object, not
                // a hard error for a cache-shaped lookup API.
                tracing::warn!("indexed blob {blake3} missing from store for CID {cid}: {e}");
                return Ok(None);
            }
        };

        let data = bytes.to_vec();
        verify_raw_cid_when_applicable(cid, &data)?;
        Ok(Some(data))
    }

    /// Resolve the BLAKE3 hash backing a SHA256 key, if known.
    ///
    /// This is the seam F2 (#900) uses to announce/locate providers: the
    /// locator plane finds peers, iroh-blobs transfers by BLAKE3.
    pub async fn blake3_of(&self, hash: &Sha256Hash) -> Option<Blake3Hash> {
        let cid = ContentCid::git_sha256(hash).ok()?;
        self.blake3_of_cid(&cid).await
    }

    /// Resolve the raw-byte BLAKE3 hash backing a content CID, if known.
    pub async fn blake3_of_cid(&self, cid: &ContentCid) -> Option<Blake3Hash> {
        self.index.lock().await.get(cid).copied()
    }

    /// Number of CID-addressable objects.
    pub async fn len(&self) -> usize {
        self.index.lock().await.len()
    }

    /// Whether the store has no CID-addressable objects.
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
async fn load_index(path: &Path) -> Result<HashMap<ContentCid, Blake3Hash>> {
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
        let parsed = line.split_once(' ').and_then(|(key, b3_hex)| {
            let cid = ContentCid::new(key)
                .or_else(|_| {
                    // Backward compatibility for the original SHA256-only index.
                    let sha = Sha256Hash::new(key)?;
                    ContentCid::git_sha256(&sha)
                })
                .ok()?;
            let b3 = b3_hex.parse::<Blake3Hash>().ok()?;
            Some((cid, b3))
        });
        match parsed {
            Some((cid, b3)) => {
                index.insert(cid, b3);
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

/// Append one `cid blake3hex` line to the sidecar index.
async fn append_index_entry(path: &Path, cid: &ContentCid, blake3: &Blake3Hash) -> Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(format!("{cid} {blake3}\n").as_bytes())
        .await?;
    file.flush().await?;
    Ok(())
}

fn verify_raw_cid_when_applicable(cid: &ContentCid, data: &[u8]) -> Result<()> {
    let decoded = cid.decoded()?;
    use hyprstream_rpc::cid::{Codec, HashAlgo};

    match (decoded.codec, decoded.multihash.algo) {
        (Codec::GitRaw, HashAlgo::Sha2_256) => {
            let expected = Sha256Hash::from_bytes(&decoded.multihash.digest)?;
            if !verify_sha256(data, &expected)? {
                return Err(Error::other(format!(
                    "sha256 verification failed for Git object CID {cid}"
                )));
            }
            Ok(())
        }
        (Codec::XetShard, HashAlgo::Blake3) if decoded.multihash.digest.len() == 32 => Ok(()),
        (codec, algo) => Err(Error::other(format!(
            "unsupported object CID domain: codec={codec:?}, algorithm={algo:?}"
        ))),
    }
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
    async fn test_put_verified_object_rejects_wrong_sha256() -> Result<()> {
        let store = IrohBlobStore::new_memory();
        let expected = sha256_git(b"expected")?;

        assert!(store
            .put_verified_object(expected.clone(), b"different".to_vec())
            .await
            .is_err());
        assert!(store.get_object(&expected).await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_xet_cid_cannot_be_rebound_to_different_bytes() -> Result<()> {
        let store = IrohBlobStore::new_memory();
        let cid = ContentCid::xet_merkle(&[0x7b; 32])?;

        store
            .put_object_by_cid(cid.clone(), b"first reconstruction".to_vec())
            .await?;
        assert!(store
            .put_object_by_cid(cid.clone(), b"different reconstruction".to_vec())
            .await
            .is_err());
        assert_eq!(
            store.get_object_by_cid(&cid).await?.as_deref(),
            Some(b"first reconstruction".as_slice())
        );
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
            let pair = (
                store.put_object(a.clone()).await?,
                store.put_object(b.clone()).await?,
            );
            store.shutdown().await?;
            pair
        };

        // Poison the sidecar: cross-wire a's Git CID to b's blake3.
        let index_path = dir.path().join(INDEX_FILE);
        let contents = tokio::fs::read_to_string(&index_path).await?;
        let mut b3 = HashMap::new();
        for line in contents.lines() {
            if let Some((cid, blake)) = line.split_once(' ') {
                b3.insert(cid.to_owned(), blake.to_owned());
            }
        }
        let cid_a = ContentCid::git_sha256(&sha_a)?;
        let cid_b = ContentCid::git_sha256(&sha_b)?;
        let poisoned = format!(
            "{} {}\n{} {}\n",
            cid_a,
            b3[cid_b.as_str()],
            cid_b,
            b3[cid_a.as_str()]
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
