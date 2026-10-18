//! Owned, exact-commit input capture for consumers that cannot trust a worktree.
//!
//! Callers may consume [`PinnedTree::file`] bytes directly or retain a Linux
//! [`SealedTreeProjection`] for legacy path readers. Copying these bytes back
//! into a writable worktree does not preserve the pin. Acquisition currently
//! buffers the full resolved tree before sealing, so memory use is proportional
//! to the model size until projection completes.

use crate::{Git2DBError, Git2DBResult, GitManager, Oid};
use git2::{ObjectType, Repository, Tree};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::fs::{self, File};
#[cfg(target_os = "linux")]
use std::io::Write;
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, FromRawFd};

/// A complete commit tree captured independently of a mutable checkout.
///
/// `Arc<[u8]>` owns every resolved file for the lifetime of this value. No
/// borrowed worktree paths or mutable file descriptors escape acquisition.
#[derive(Debug)]
pub struct PinnedTree {
    commit: Oid,
    files: BTreeMap<PathBuf, PinnedFile>,
}

/// One resolved input with its source Git blob identity.
#[derive(Debug)]
pub struct PinnedFile {
    source_blob: Oid,
    bytes: Arc<[u8]>,
}

/// A private directory whose entries point to sealed, owned Linux memfds.
///
/// All file content is kernel-sealed against writes and size changes. The
/// directory exists only for path-based readers; callers must retain this
/// value through every read, including lazy reads after model startup.
#[derive(Debug)]
pub struct SealedTreeProjection {
    commit: Oid,
    root: tempfile::TempDir,
    files: Vec<std::fs::File>,
}

impl SealedTreeProjection {
    pub fn commit(&self) -> Oid {
        self.commit
    }

    pub fn root(&self) -> &Path {
        self.root.path()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

impl PinnedFile {
    pub fn source_blob(&self) -> Oid {
        self.source_blob
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug)]
struct TreeBlob {
    path: PathBuf,
    source_blob: Oid,
    bytes: Vec<u8>,
}

impl PinnedTree {
    /// Capture the full commit tree, including resolved LFS/XET content.
    /// Missing commits, symlinks, submodules, and unresolved pointers fail.
    /// Worktree cleanliness is a caller policy; this method never reads the
    /// checkout and does not report whether a selected checkout is dirty.
    pub async fn acquire(repo_path: &Path, commit: Oid) -> Git2DBResult<Self> {
        let blobs = capture_blobs(repo_path, commit)?;
        Self::from_blobs(commit, blobs).await
    }

    async fn from_blobs(commit: Oid, blobs: Vec<TreeBlob>) -> Git2DBResult<Self> {
        let mut files = BTreeMap::new();
        #[cfg(feature = "xet-storage")]
        let mut storage = None;
        for blob in blobs {
            #[cfg(feature = "xet-storage")]
            let bytes = resolve_pointer(blob.bytes, &mut storage).await?;
            #[cfg(not(feature = "xet-storage"))]
            let bytes = reject_unresolved_pointer(blob.bytes)?;
            files.insert(
                blob.path,
                PinnedFile {
                    source_blob: blob.source_blob,
                    bytes: Arc::<[u8]>::from(bytes),
                },
            );
        }
        Ok(Self { commit, files })
    }

    pub fn commit(&self) -> Oid {
        self.commit
    }

    pub fn file(&self, path: &Path) -> Option<&[u8]> {
        self.files.get(path).map(PinnedFile::bytes)
    }

    pub fn files(&self) -> impl Iterator<Item = (&Path, &PinnedFile)> {
        self.files.iter().map(|(path, file)| (path.as_path(), file))
    }

    /// Move the captured bytes into kernel-sealed files for existing path
    /// loaders. The temporary directory is private and owns no writable file
    /// content; its lifetime must cover every later path access.
    #[cfg(target_os = "linux")]
    pub fn into_sealed_projection(self) -> Git2DBResult<SealedTreeProjection> {
        let root = tempfile::Builder::new()
            .prefix("git2db-pinned-tree-")
            .tempdir()
            .map_err(|e| Git2DBError::internal(format!("create pinned tree directory: {e}")))?;
        let mut files = Vec::with_capacity(self.files.len());
        for (relative_path, pinned) in self.files {
            let target = root.path().join(&relative_path);
            let parent = target.parent().ok_or_else(|| {
                Git2DBError::internal(format!("pinned file has no parent: {}", target.display()))
            })?;
            fs::create_dir_all(parent).map_err(|e| {
                Git2DBError::internal(format!("create pinned directory {}: {e}", parent.display()))
            })?;
            let mut file = create_sealable_file()?;
            file.write_all(pinned.bytes()).map_err(|e| {
                Git2DBError::internal(format!(
                    "write pinned file {}: {e}",
                    relative_path.display()
                ))
            })?;
            seal_file(&file)?;
            std::os::unix::fs::symlink(format!("/proc/self/fd/{}", file.as_raw_fd()), &target)
                .map_err(|e| {
                    Git2DBError::internal(format!("project pinned file {}: {e}", target.display()))
                })?;
            files.push(file);
        }
        Ok(SealedTreeProjection {
            commit: self.commit,
            root,
            files,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn into_sealed_projection(self) -> Git2DBResult<SealedTreeProjection> {
        Err(Git2DBError::internal(
            "sealed pinned model projection requires Linux memfd seals",
        ))
    }
}

#[cfg(target_os = "linux")]
fn create_sealable_file() -> Git2DBResult<File> {
    const NAME: &[u8] = b"git2db-pinned-input\0";
    // SAFETY: NAME is a NUL-terminated static byte string. A successful
    // memfd_create returns a new owned descriptor, transferred into File.
    let fd = unsafe {
        libc::memfd_create(
            NAME.as_ptr().cast(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    if fd < 0 {
        return Err(Git2DBError::internal(format!(
            "create sealable pinned file: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: fd is newly created and owned exclusively by this function.
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(target_os = "linux")]
fn seal_file(file: &File) -> Git2DBResult<()> {
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    // SAFETY: fcntl operates on a valid owned descriptor; the constant-sized
    // integer third argument is the documented F_ADD_SEALS ABI.
    let result = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) };
    if result < 0 {
        return Err(Git2DBError::internal(format!(
            "seal pinned file: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

fn capture_blobs(repo_path: &Path, commit: Oid) -> Git2DBResult<Vec<TreeBlob>> {
    let repo = GitManager::global().get_repository(repo_path)?.open()?;
    let _ = verified_git_object_bytes(&repo, commit, ObjectType::Commit)?;
    // find_commit, rather than revparse/peel, requires exactly the requested
    // object to be a commit. It cannot silently substitute HEAD or a tag.
    let tree = repo
        .find_commit(commit)
        .map_err(|e| Git2DBError::internal(format!("pinned commit {commit} unavailable: {e}")))?
        .tree()
        .map_err(|e| Git2DBError::internal(format!("pinned tree: {e}")))?;
    let mut blobs = Vec::new();
    capture_tree(&repo, &tree, Path::new(""), &mut blobs)?;
    Ok(blobs)
}

fn capture_tree(
    repo: &Repository,
    tree: &Tree<'_>,
    prefix: &Path,
    blobs: &mut Vec<TreeBlob>,
) -> Git2DBResult<()> {
    let _ = verified_git_object_bytes(repo, tree.id(), ObjectType::Tree)?;
    for entry in tree {
        let name = entry
            .name()
            .ok_or_else(|| Git2DBError::internal("non-UTF-8 tree entry"))?;
        let component = Path::new(name);
        if name.contains('/')
            || name.contains('\\')
            || component.components().count() != 1
            || !matches!(component.components().next(), Some(Component::Normal(_)))
        {
            return Err(Git2DBError::internal(format!(
                "unsafe tree entry: {name:?}"
            )));
        }
        let path = prefix.join(component);
        match (entry.kind(), entry.filemode()) {
            (Some(ObjectType::Tree), 0o040000) => {
                let child = repo
                    .find_tree(entry.id())
                    .map_err(|e| Git2DBError::internal(format!("tree {}: {e}", path.display())))?;
                capture_tree(repo, &child, &path, blobs)?;
            }
            (Some(ObjectType::Blob), 0o100644 | 0o100755) => {
                blobs.push(TreeBlob {
                    path,
                    source_blob: entry.id(),
                    bytes: verified_git_object_bytes(repo, entry.id(), ObjectType::Blob)?,
                });
            }
            _ => {
                return Err(Git2DBError::internal(format!(
                    "unsupported tree entry {} (mode {:o})",
                    path.display(),
                    entry.filemode()
                )))
            }
        }
    }
    Ok(())
}

fn verified_git_object_bytes(
    repo: &Repository,
    oid: Oid,
    expected_kind: ObjectType,
) -> Git2DBResult<Vec<u8>> {
    let odb = repo
        .odb()
        .map_err(|e| Git2DBError::internal(format!("open Git object database: {e}")))?;
    let object = odb
        .read(oid)
        .map_err(|e| Git2DBError::internal(format!("read Git object {oid}: {e}")))?;
    if object.kind() != expected_kind {
        return Err(Git2DBError::internal(format!(
            "Git object {oid} has type {:?}, expected {expected_kind:?}",
            object.kind()
        )));
    }
    let computed = Oid::hash_object(expected_kind, object.data())
        .map_err(|e| Git2DBError::internal(format!("hash Git object {oid}: {e}")))?;
    if computed != oid {
        return Err(Git2DBError::internal(format!(
            "Git object mismatch: requested {oid}, got {computed}"
        )));
    }
    Ok(object.data().to_vec())
}

pub(crate) fn is_pointer(bytes: &[u8]) -> bool {
    let trimmed = bytes.trim_ascii_start();
    bytes.starts_with(b"version https://git-lfs")
        || bytes.starts_with(b"version https://hawser")
        || bytes.starts_with(b"# xet version")
        || (trimmed.starts_with(b"{")
            && serde_json::from_slice::<serde_json::Value>(trimmed).is_ok_and(|v| {
                (v.get("hash").is_some() && v.get("file_size").is_some())
                    || v.get("xet").and_then(serde_json::Value::as_str) == Some("gittorrent")
            }))
}

#[cfg(not(feature = "xet-storage"))]
fn reject_unresolved_pointer(bytes: Vec<u8>) -> Git2DBResult<Vec<u8>> {
    if is_pointer(&bytes) {
        return Err(Git2DBError::internal(
            "pinned tree contains a pointer but xet-storage is disabled",
        ));
    }
    Ok(bytes)
}

#[cfg(feature = "xet-storage")]
async fn resolve_pointer(
    bytes: Vec<u8>,
    storage: &mut Option<crate::LfsStorage>,
) -> Git2DBResult<Vec<u8>> {
    if !is_pointer(&bytes) {
        return Ok(bytes);
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| Git2DBError::internal(format!("pointer is not UTF-8: {e}")))?;
    if text.starts_with("# xet version") {
        return Err(Git2DBError::internal("unsupported legacy XET pointer"));
    }
    if storage.is_none() {
        *storage = Some(crate::LfsStorage::new(&crate::XetConfig::default()).await?);
    }
    let storage = storage
        .as_ref()
        .ok_or_else(|| Git2DBError::internal("LFS storage unavailable"))?;
    if crate::is_lfs_pointer(text) {
        let pointer = crate::LfsPointer::parse(text)?;
        let resolved = storage.smudge_lfs_pointer(&pointer).await?;
        validate_lfs_content(&pointer, &resolved)?;
        return Ok(resolved);
    }
    let info: data::XetFileInfo = serde_json::from_str(text)
        .map_err(|e| Git2DBError::internal(format!("invalid XET pointer: {e}")))?;
    info.merkle_hash()
        .map_err(|e| Git2DBError::internal(format!("invalid XET hash: {e}")))?;
    let resolved = storage.smudge_bytes(text).await?;
    validate_xet_content(&info, &resolved)?;
    Ok(resolved)
}

#[cfg(feature = "xet-storage")]
fn validate_xet_content(info: &data::XetFileInfo, content: &[u8]) -> Git2DBResult<()> {
    if content.len() as u64 != info.file_size() {
        return Err(Git2DBError::internal(format!(
            "XET size mismatch: expected {}, got {}",
            info.file_size(),
            content.len()
        )));
    }
    // XET's file hash is the Merkle aggregation of content-defined chunks.
    // Recompute it from the returned bytes: a CAS response keyed by the
    // captured hash is not itself proof that its payload matches that hash.
    let mut chunker = deduplication::Chunker::default();
    let chunks = chunker.next_block_bytes(&bytes::Bytes::copy_from_slice(content), true);
    let chunk_hashes: Vec<_> = chunks
        .iter()
        .map(|chunk| (chunk.hash, chunk.data.len() as u64))
        .collect();
    let computed = merklehash::file_hash(&chunk_hashes);
    let expected = info
        .merkle_hash()
        .map_err(|e| Git2DBError::internal(format!("invalid XET hash: {e}")))?;
    if computed != expected {
        return Err(Git2DBError::internal(format!(
            "XET Merkle mismatch: expected {}, got {}",
            expected.hex(),
            computed.hex()
        )));
    }
    Ok(())
}

#[cfg(feature = "xet-storage")]
fn validate_lfs_content(pointer: &crate::LfsPointer, content: &[u8]) -> Git2DBResult<()> {
    use sha2::{Digest, Sha256};
    if content.len() as u64 != pointer.size() {
        return Err(Git2DBError::internal(format!(
            "LFS size mismatch: expected {}, got {}",
            pointer.size(),
            content.len()
        )));
    }
    let computed = format!("{:x}", Sha256::digest(content));
    if computed != pointer.oid() {
        return Err(Git2DBError::internal(format!(
            "LFS SHA-256 mismatch: expected {}, got {computed}",
            pointer.oid()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn captures_requested_tree_without_reopening_worktree() -> Git2DBResult<()> {
        let dir = tempfile::tempdir().map_err(|e| Git2DBError::internal(e.to_string()))?;
        let repo =
            Repository::init(dir.path()).map_err(|e| Git2DBError::internal(e.to_string()))?;
        std::fs::write(dir.path().join("config.json"), b"reviewed")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let mut index = repo
            .index()
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        index
            .add_path(Path::new("config.json"))
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let tree_oid = index
            .write_tree()
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let tree = repo
            .find_tree(tree_oid)
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let sig = git2::Signature::now("test", "test@example.invalid")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let oid = repo
            .commit(Some("HEAD"), &sig, &sig, "reviewed", &tree, &[])
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        drop(tree);
        drop(repo);

        // The worktree is already dirty when acquisition begins. The exact
        // commit object, rather than the checkout, supplies the captured file.
        std::fs::write(dir.path().join("config.json"), b"changed")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        std::fs::write(dir.path().join("untracked.safetensors"), b"not reviewed")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let pinned = PinnedTree::acquire(dir.path(), oid).await?;
        assert_eq!(
            pinned.file(Path::new("config.json")),
            Some(&b"reviewed"[..])
        );
        assert!(pinned.file(Path::new("untracked.safetensors")).is_none());
        // Simulate mutation after the Git tree has been captured but before
        // pointer resolution and ownership transfer complete.
        let blobs = capture_blobs(dir.path(), oid)?;
        std::fs::write(dir.path().join("config.json"), b"changed again")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let during_acquire = PinnedTree::from_blobs(oid, blobs).await?;
        assert_eq!(
            during_acquire.file(Path::new("config.json")),
            Some(&b"reviewed"[..])
        );
        assert_eq!(
            pinned.file(Path::new("config.json")),
            Some(&b"reviewed"[..])
        );
        assert!(PinnedTree::acquire(dir.path(), Oid::zero()).await.is_err());
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn sealed_projection_survives_worktree_mutation_and_rejects_writes() -> Git2DBResult<()> {
        let dir = tempfile::tempdir().map_err(|e| Git2DBError::internal(e.to_string()))?;
        let repo =
            Repository::init(dir.path()).map_err(|e| Git2DBError::internal(e.to_string()))?;
        std::fs::write(dir.path().join("tokenizer.json"), b"reviewed")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let mut index = repo
            .index()
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        index
            .add_path(Path::new("tokenizer.json"))
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let tree_oid = index
            .write_tree()
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let tree = repo
            .find_tree(tree_oid)
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let sig = git2::Signature::now("test", "test@example.invalid")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let oid = repo
            .commit(Some("HEAD"), &sig, &sig, "reviewed", &tree, &[])
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        drop(tree);
        drop(repo);

        let projected = PinnedTree::acquire(dir.path(), oid)
            .await?
            .into_sealed_projection()?;
        let input = projected.root().join("tokenizer.json");
        std::fs::write(dir.path().join("tokenizer.json"), b"replaced")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        assert_eq!(
            std::fs::read(&input).map_err(|e| Git2DBError::internal(e.to_string()))?,
            b"reviewed"
        );
        assert!(std::fs::write(&input, b"tampered").is_err());
        assert_eq!(
            std::fs::read(&input).map_err(|e| Git2DBError::internal(e.to_string()))?,
            b"reviewed"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_symlink_in_requested_tree() -> Git2DBResult<()> {
        let dir = tempfile::tempdir().map_err(|e| Git2DBError::internal(e.to_string()))?;
        let repo =
            Repository::init(dir.path()).map_err(|e| Git2DBError::internal(e.to_string()))?;
        std::os::unix::fs::symlink("/etc/passwd", dir.path().join("config.json"))
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let mut index = repo
            .index()
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        index
            .add_path(Path::new("config.json"))
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let tree_oid = index
            .write_tree()
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let tree = repo
            .find_tree(tree_oid)
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let sig = git2::Signature::now("test", "test@example.invalid")
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        let oid = repo
            .commit(Some("HEAD"), &sig, &sig, "symlink", &tree, &[])
            .map_err(|e| Git2DBError::internal(e.to_string()))?;
        drop(tree);
        drop(repo);
        assert!(PinnedTree::acquire(dir.path(), oid).await.is_err());
        Ok(())
    }

    #[cfg(feature = "xet-storage")]
    #[test]
    fn rejects_resolved_pointer_mismatch() -> Git2DBResult<()> {
        use sha2::{Digest, Sha256};
        let expected = format!("{:x}", Sha256::digest(b"reviewed"));
        let pointer = crate::LfsPointer::parse(&format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{expected}\nsize 8\n"
        ))?;
        validate_lfs_content(&pointer, b"reviewed")?;
        assert!(validate_lfs_content(&pointer, b"replaced").is_err());
        assert!(validate_lfs_content(&pointer, b"short").is_err());
        Ok(())
    }

    #[cfg(feature = "xet-storage")]
    #[test]
    fn rejects_xet_payload_with_matching_size_but_wrong_bytes() -> Git2DBResult<()> {
        let expected = merklehash::file_hash(&[(merklehash::compute_data_hash(b"reviewed"), 8)]);
        let info = data::XetFileInfo::new(expected.hex(), 8);
        validate_xet_content(&info, b"reviewed")?;
        assert!(validate_xet_content(&info, b"replaced").is_err());
        assert!(validate_xet_content(&info, b"short").is_err());
        Ok(())
    }
}
