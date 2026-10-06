//! Bounded-memory, exact-commit projection for path-based model loaders.
//!
//! The selected Git tree is enumerated without reading blob bodies. A packaged
//! Git CLI streams packed or loose blobs to private disk files; each body is
//! hashed against its tree OID. Resolved LFS/XET payloads are verified against
//! the captured pointer before any loader path is published. Payload names are
//! then unlinked, leaving only read-only descriptors and private `/proc/self/fd`
//! aliases for the lifetime of this value. No worktree file is read or aliased.

use crate::{Git2DBError, Git2DBResult, GitManager, Oid};
use git2::{ObjectType, Repository, Tree};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "linux")]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const MAX_FILES: usize = 512;
const MAX_TREE_ENTRIES: usize = 4096;
const MAX_TREE_DEPTH: usize = 64;
const MAX_EXPANDED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const FREE_SPACE_RESERVE: u64 = 512 * 1024 * 1024;
const PROJECTION_PREFIX: &str = "git2db-pinned-disk-";
const OWNER_FILE: &str = ".git2db-pinned-disk-owner-v1";
const OWNER_MARKER: &[u8] = b"git2db-pinned-disk-owner-v1\n";
static PROJECTION_CAPACITY: OnceLock<Arc<Semaphore>> = OnceLock::new();

// This absolute path is populated by the runtime image's git-core package.
// The OCI digest, not a mutable host PATH, pins the executable at deployment.
const GIT_CLI: &str = "/usr/bin/git";
const TMPFS_MAGIC: libc::c_long = 0x0102_1994;
const RAMFS_MAGIC: libc::c_long = 0x8584_58f6;

#[derive(Debug)]
struct BlobEntry {
    path: PathBuf,
    oid: Oid,
    source_size: u64,
}

/// Exact-tree files backed by disk, but not by the selected worktree.
///
/// The private parent must be disk-backed and container-private to the loader.
/// In particular, it must not be mounted into Registry or another writer. The
/// only write handles close before verification and publication; unlinked
/// payloads survive solely through these retained read-only descriptors. The
/// 0400 mode is a guard against mistakes, not the security boundary: arbitrary
/// code running as the loader UID or host root can tamper with the process and
/// is outside the Registry-worktree-mutation threat model.
#[derive(Debug)]
pub struct DiskPinnedTreeProjection {
    commit: Oid,
    owner_pid: u32,
    input_root: PathBuf,
    // Rust drops fields in declaration order. Close the unlinked payload FDs
    // before the owner releases the disk-capacity lease.
    files: Vec<File>,
    _root: OwnedProjection,
}

/// Keep the owner marker and its lock until every payload child is gone. A
/// failed or interrupted cleanup leaves the marker for the next scavenger.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone)]
struct OwnedProjection(Arc<OwnedProjectionInner>);

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct OwnedProjectionInner {
    path: PathBuf,
    _owner_lock: File,
    // A cancelled acquire can leave its blocking Git writer running. Keep the
    // one-slot disk lease until that last owner finishes and cleanup completes.
    _capacity: Option<OwnedSemaphorePermit>,
    #[cfg(test)]
    observed_reader_fd: Option<std::os::fd::RawFd>,
}

#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
struct OwnedProjection;

#[cfg(target_os = "linux")]
impl OwnedProjection {
    fn create(parent: &Path, capacity: Option<OwnedSemaphorePermit>) -> Git2DBResult<Self> {
        let root = tempfile::Builder::new()
            .prefix(PROJECTION_PREFIX)
            .tempdir_in(parent)
            .map_err(|e| internal(format!("create private projection: {e}")))?;
        // Before this marker exists the newly created directory is empty.
        let owner_lock = create_projection_owner(root.path())?;
        Ok(Self(Arc::new(OwnedProjectionInner {
            path: root.keep(),
            _owner_lock: owner_lock,
            _capacity: capacity,
            #[cfg(test)]
            observed_reader_fd: None,
        })))
    }

    fn path(&self) -> &Path {
        &self.0.path
    }
}

#[cfg(target_os = "linux")]
impl Drop for OwnedProjectionInner {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(fd) = self.observed_reader_fd {
            // Regression witness: the enclosing artifact must close its
            // unlinked reader before this owner can release the lease.
            if let Ok(target) = fs::read_link(format!("/proc/self/fd/{fd}")) {
                assert!(
                    !target.starts_with(&self.path),
                    "projection reader is still open when its owner drops"
                );
            }
        }
        // The lock field is dropped only after this method returns. If removal
        // fails, the still-marked directory can be retried at next startup.
        if let Err(error) = remove_owned_projection(&self.path) {
            tracing::warn!(path = %self.path.display(), %error, "projection cleanup deferred");
        }
    }
}

impl DiskPinnedTreeProjection {
    pub fn commit(&self) -> Oid {
        self.commit
    }

    pub fn root(&self) -> &Path {
        &self.input_root
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// A `/proc/self/fd` alias is valid only in the acquiring process. An
    /// exec'd inference process must not be handed this projection: its CLOEXEC
    /// descriptors are absent and a reused descriptor number could name other
    /// bytes. The active Model→Inference route is threaded, not subprocess.
    pub fn ensure_current_process(&self) -> Git2DBResult<()> {
        if self.owner_pid != std::process::id() {
            return Err(internal(
                "disk-backed pinned projection cannot cross a process boundary",
            ));
        }
        Ok(())
    }

    /// Capture exactly `commit` into a private disk parent. No partial input
    /// directory is published on any failure.
    #[cfg(target_os = "linux")]
    pub async fn acquire(
        repo_path: &Path,
        commit: Oid,
        private_disk_parent: &Path,
    ) -> Git2DBResult<Self> {
        Self::acquire_with_capacity(repo_path, commit, private_disk_parent, None).await
    }

    /// Staging admission with one global lease retained until the artifact is
    /// dropped. Distinct model refs cannot concurrently consume private disk.
    #[cfg(target_os = "linux")]
    pub async fn acquire_bounded(
        repo_path: &Path,
        commit: Oid,
        private_disk_parent: &Path,
    ) -> Git2DBResult<Self> {
        let capacity = Arc::clone(PROJECTION_CAPACITY.get_or_init(|| Arc::new(Semaphore::new(1))))
            .try_acquire_owned()
            .map_err(|_| internal("bounded model projection capacity is occupied"))?;
        let repo_path = repo_path.to_path_buf();
        let private_disk_parent = private_disk_parent.to_path_buf();
        // The Model service has a current-thread reactor. CAS's bounded writer
        // intentionally does synchronous, backpressured writes; run the entire
        // projection on a dedicated runtime so those writes cannot stall RPCs.
        // The detached worker owns the permit even if this await is cancelled.
        run_on_projection_worker(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .map_err(|e| internal(format!("build projection I/O runtime: {e}")))?;
            runtime.block_on(Self::acquire_with_capacity(
                &repo_path,
                commit,
                &private_disk_parent,
                Some(capacity),
            ))
        })
        .await
    }

    #[cfg(target_os = "linux")]
    async fn acquire_with_capacity(
        repo_path: &Path,
        commit: Oid,
        private_disk_parent: &Path,
        capacity: Option<OwnedSemaphorePermit>,
    ) -> Git2DBResult<Self> {
        verify_private_disk_parent(private_disk_parent)?;
        scavenge_abandoned_projections(private_disk_parent)?;
        let repo = GitManager::global().get_repository(repo_path)?.open()?;
        verify_object(&repo, commit, ObjectType::Commit)?;
        let tree = repo
            .find_commit(commit)
            .map_err(|e| internal(format!("pinned commit {commit} unavailable: {e}")))?
            .tree()
            .map_err(|e| internal(format!("pinned tree: {e}")))?;
        let mut entries = Vec::new();
        let mut visited_entries = 0;
        collect_entries(
            &repo,
            &tree,
            Path::new(""),
            &mut entries,
            &mut visited_entries,
        )?;
        let projected_bytes = preflight_projection(&repo, &mut entries, private_disk_parent)?;
        if entries.len() > MAX_FILES {
            return Err(internal(format!(
                "model tree exceeds {MAX_FILES} file limit"
            )));
        }
        if projected_bytes > MAX_EXPANDED_BYTES {
            return Err(internal(format!(
                "model projection exceeds {MAX_EXPANDED_BYTES}-byte limit"
            )));
        }
        let git_dir = repo.path().to_path_buf();
        drop(tree);
        drop(repo);

        let root = OwnedProjection::create(private_disk_parent, capacity)?;
        let payload_dir = root.path().join("payloads");
        let staged_inputs = root.path().join("staged-inputs");
        fs::create_dir(&payload_dir)
            .map_err(|e| internal(format!("create payload directory: {e}")))?;
        fs::create_dir(&staged_inputs)
            .map_err(|e| internal(format!("create staged inputs: {e}")))?;
        let mut files = Vec::with_capacity(entries.len());
        #[cfg(feature = "xet-storage")]
        let mut storage = None;

        for (index, entry) in entries.into_iter().enumerate() {
            let source_path = payload_dir.join(format!("{index}.git"));
            let source_git_dir = git_dir.clone();
            let source_oid = entry.oid;
            let source_size = entry.source_size;
            let source_copy = source_path.clone();
            // The task can outlive a cancelled acquire future. Retaining the
            // owner prevents cleanup from racing its late payload write.
            let task_owner = root.clone();
            let size = tokio::task::spawn_blocking(move || {
                let _task_owner = task_owner;
                stream_verified_blob(&source_git_dir, source_oid, source_size, &source_copy)
            })
            .await
            .map_err(|e| internal(format!("Git stream task failed: {e}")))??;

            #[cfg(feature = "xet-storage")]
            let mut final_path = source_path;
            #[cfg(not(feature = "xet-storage"))]
            let final_path = source_path;
            if size <= 4096 {
                let bytes = fs::read(&final_path)
                    .map_err(|e| internal(format!("read pointer candidate: {e}")))?;
                if crate::pinned_tree::is_pointer(&bytes) {
                    #[cfg(feature = "xet-storage")]
                    {
                        let resolved = payload_dir.join(format!("{index}.resolved"));
                        resolve_pointer_to_file(&bytes, &resolved, &mut storage, &root).await?;
                        final_path = resolved;
                    }
                    #[cfg(not(feature = "xet-storage"))]
                    return Err(internal("pinned tree contains unresolved pointer"));
                }
            } else {
                // A malformed oversized pointer must not become a loader input.
                let mut prefix = [0u8; 128];
                let count = File::open(&final_path)
                    .and_then(|mut file| file.read(&mut prefix))
                    .map_err(|e| internal(format!("read payload prefix: {e}")))?;
                if prefix[..count].starts_with(b"version https://git-lfs")
                    || prefix[..count].starts_with(b"version https://hawser")
                    || prefix[..count].starts_with(b"# xet version")
                {
                    return Err(internal("oversized model pointer"));
                }
            }

            // Verify after all writers close. Hold only a read-only descriptor;
            // the payload's disk pathname is then removed before publishing the
            // reader-facing symlink.
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&final_path, fs::Permissions::from_mode(0o400))
                .map_err(|e| internal(format!("make private payload read-only: {e}")))?;
            let reader = File::open(&final_path)
                .map_err(|e| internal(format!("open verified payload: {e}")))?;
            fs::remove_file(&final_path)
                .map_err(|e| internal(format!("unlink private payload: {e}")))?;
            let projected_path = staged_inputs.join(&entry.path);
            let parent = projected_path
                .parent()
                .ok_or_else(|| internal("invalid projected path"))?;
            fs::create_dir_all(parent)
                .map_err(|e| internal(format!("create projected directories: {e}")))?;
            std::os::unix::fs::symlink(
                format!("/proc/self/fd/{}", reader.as_raw_fd()),
                &projected_path,
            )
            .map_err(|e| internal(format!("project verified input: {e}")))?;
            files.push(reader);
        }

        let input_root = root.path().join("inputs");
        fs::rename(&staged_inputs, &input_root)
            .map_err(|e| internal(format!("publish verified inputs: {e}")))?;
        Ok(Self {
            commit,
            owner_pid: std::process::id(),
            input_root,
            files,
            _root: root,
        })
    }

    #[cfg(not(target_os = "linux"))]
    async fn acquire_with_capacity(
        _repo_path: &Path,
        _commit: Oid,
        _private_disk_parent: &Path,
        _capacity: Option<OwnedSemaphorePermit>,
    ) -> Git2DBResult<Self> {
        Err(internal(
            "disk-backed pinned projection requires Linux /proc/self/fd",
        ))
    }

    #[cfg(not(target_os = "linux"))]
    pub async fn acquire_bounded(
        _repo_path: &Path,
        _commit: Oid,
        _private_disk_parent: &Path,
    ) -> Git2DBResult<Self> {
        Err(internal(
            "bounded pinned projection requires Linux /proc/self/fd",
        ))
    }

    #[cfg(not(target_os = "linux"))]
    pub async fn acquire(
        _repo_path: &Path,
        _commit: Oid,
        _private_disk_parent: &Path,
    ) -> Git2DBResult<Self> {
        Err(internal(
            "disk-backed pinned projection requires Linux /proc/self/fd",
        ))
    }
}

fn internal(message: impl std::fmt::Display) -> Git2DBError {
    Git2DBError::internal(message.to_string())
}

#[cfg(target_os = "linux")]
async fn run_on_projection_worker<T: Send + 'static>(
    work: impl FnOnce() -> Git2DBResult<T> + Send + 'static,
) -> Git2DBResult<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| internal(format!("projection I/O worker failed: {e}")))?
}

#[cfg(target_os = "linux")]
fn try_lock_projection(file: &File) -> Git2DBResult<bool> {
    // SAFETY: file remains open for the duration of flock; Linux flock locks
    // the open file description and releases it after the last holder closes.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
        Ok(false)
    } else {
        Err(internal(format!("lock private projection: {error}")))
    }
}

#[cfg(target_os = "linux")]
fn create_projection_owner(root: &Path) -> Git2DBResult<File> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(root.join(OWNER_FILE))
        .map_err(|e| internal(format!("create projection owner: {e}")))?;
    if !try_lock_projection(&file)? {
        return Err(internal("new projection owner lock is occupied"));
    }
    file.write_all(OWNER_MARKER)
        .and_then(|_| file.sync_all())
        .map_err(|e| internal(format!("write projection owner: {e}")))?;
    Ok(file)
}

#[cfg(target_os = "linux")]
fn remove_owned_projection(path: &Path) -> Git2DBResult<()> {
    // A recursive removal of the root can unlink OWNER_FILE before reaching
    // a later payload. Remove each child first, so interruption at any point
    // with payload bytes remaining always leaves the reclaim marker intact.
    for entry in fs::read_dir(path).map_err(|e| internal(format!("read projection: {e}")))? {
        let entry = entry.map_err(|e| internal(format!("read projection child: {e}")))?;
        if entry.file_name() == OWNER_FILE {
            continue;
        }
        let child = entry.path();
        let kind = entry
            .file_type()
            .map_err(|e| internal(format!("inspect projection child: {e}")))?;
        if kind.is_dir() {
            fs::remove_dir_all(&child)
                .map_err(|e| internal(format!("remove projection child: {e}")))?;
        } else {
            // remove_file unlinks symlinks instead of following them.
            fs::remove_file(&child)
                .map_err(|e| internal(format!("remove projection child: {e}")))?;
        }
    }
    fs::remove_file(path.join(OWNER_FILE))
        .map_err(|e| internal(format!("remove projection owner: {e}")))?;
    fs::remove_dir(path).map_err(|e| internal(format!("remove empty projection: {e}")))?;
    Ok(())
}

/// Reclaim only this version's abandoned private projections. Older unmarked
/// directories are deliberately left alone: without a lock they cannot be
/// distinguished safely from a projection still in use by another process.
#[cfg(target_os = "linux")]
fn scavenge_abandoned_projections(parent: &Path) -> Git2DBResult<()> {
    for entry in fs::read_dir(parent).map_err(|e| internal(format!("scan projections: {e}")))? {
        let entry = entry.map_err(|e| internal(format!("scan projection entry: {e}")))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(suffix) = name.strip_prefix(PROJECTION_PREFIX) else {
            continue;
        };
        if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_alphanumeric()) {
            continue;
        }
        let path = entry.path();
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if !metadata.is_dir()
            || metadata.file_type().is_symlink()
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            continue;
        }
        let Ok(mut file) = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path.join(OWNER_FILE))
        else {
            continue;
        };
        let Ok(owner) = file.metadata() else { continue };
        if !owner.is_file()
            || owner.uid() != unsafe { libc::geteuid() }
            || owner.permissions().mode() & 0o077 != 0
            || owner.len() != OWNER_MARKER.len() as u64
        {
            continue;
        }
        let mut marker = vec![0; OWNER_MARKER.len()];
        if file.read_exact(&mut marker).is_err() || marker != OWNER_MARKER {
            continue;
        }
        if !try_lock_projection(&file)? {
            continue;
        }
        // Keep the flock through marker-last removal. A failed cleanup leaves
        // the marker for another attempt instead of stranding payload bytes.
        remove_owned_projection(&path)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_private_disk_parent(path: &Path) -> Git2DBResult<()> {
    let metadata =
        fs::symlink_metadata(path).map_err(|e| internal(format!("private parent: {e}")))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(internal("projection parent must be a real directory"));
    }
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o077 != 0 {
        return Err(internal(
            "projection parent must be owned by loader and mode 0700",
        ));
    }
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|e| internal(format!("private parent path: {e}")))?;
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: c_path is NUL-terminated and stat points to writable storage.
    if unsafe { libc::statfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(internal(format!(
            "stat private parent: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: statfs succeeded and initialized the output.
    let stat = unsafe { stat.assume_init() };
    if stat.f_type == TMPFS_MAGIC || stat.f_type == RAMFS_MAGIC {
        return Err(internal("private projection parent is memory-backed"));
    }
    if !Path::new(GIT_CLI).is_file() {
        return Err(internal("packaged /usr/bin/git is unavailable"));
    }
    Ok(())
}

fn verify_object(repo: &Repository, oid: Oid, expected: ObjectType) -> Git2DBResult<()> {
    let odb = repo
        .odb()
        .map_err(|e| internal(format!("open Git ODB: {e}")))?;
    let object = odb
        .read(oid)
        .map_err(|e| internal(format!("read Git object {oid}: {e}")))?;
    if object.kind() != expected
        || Oid::hash_object(expected, object.data())
            .map_err(|e| internal(format!("hash Git object {oid}: {e}")))?
            != oid
    {
        return Err(internal(format!("Git object type/hash mismatch at {oid}")));
    }
    Ok(())
}

/// Compute an upper bound from Git object sizes and the resolved LFS/XET
/// pointer sizes before creating any projection files or streaming large blobs.
fn preflight_projection(
    repo: &Repository,
    entries: &mut [BlobEntry],
    parent: &Path,
) -> Git2DBResult<u64> {
    let odb = repo
        .odb()
        .map_err(|e| internal(format!("open projection ODB: {e}")))?;
    let mut total = 0u64;
    for entry in entries.iter_mut() {
        let (blob_size, kind) = odb
            .read_header(entry.oid)
            .map_err(|e| internal(format!("read model blob size {}: {e}", entry.oid)))?;
        if kind != ObjectType::Blob {
            return Err(internal("model tree object ceased to be a blob"));
        }
        entry.source_size = blob_size as u64;
        let mut size = blob_size as u64;
        if blob_size <= 4096 {
            let blob = odb
                .read(entry.oid)
                .map_err(|e| internal(format!("read pointer candidate: {e}")))?;
            let bytes = blob.data();
            if crate::pinned_tree::is_pointer(bytes) {
                #[cfg(feature = "xet-storage")]
                {
                    let text = std::str::from_utf8(bytes)
                        .map_err(|e| internal(format!("pointer UTF-8: {e}")))?;
                    if crate::is_lfs_pointer(text) {
                        size = crate::LfsPointer::parse(text)?.size();
                    } else if text.starts_with("# xet version") {
                        return Err(internal("unsupported legacy XET pointer"));
                    } else {
                        let info: data::XetFileInfo = serde_json::from_str(text)
                            .map_err(|e| internal(format!("invalid XET pointer: {e}")))?;
                        size = info.file_size();
                    }
                }
            }
        } else {
            total = total
                .checked_add(size)
                .ok_or_else(|| internal("model size overflow"))?;
        }
        total = total
            .checked_add(if blob_size <= 4096 { size } else { 0 })
            .ok_or_else(|| internal("model size overflow"))?;
        if total > MAX_EXPANDED_BYTES {
            return Err(internal(format!(
                "model projection exceeds {MAX_EXPANDED_BYTES}-byte limit"
            )));
        }
    }
    // Pointer blobs and filesystem metadata coexist transiently with resolved
    // payloads; reserve an additional bounded 4 KiB per tree entry.
    let allocation_bound = total.saturating_add((entries.len() as u64).saturating_mul(4096));
    ensure_disk_space(parent, allocation_bound)?;
    Ok(total)
}

#[cfg(target_os = "linux")]
fn ensure_disk_space(parent: &Path, required: u64) -> Git2DBResult<()> {
    let c_path = std::ffi::CString::new(parent.as_os_str().as_encoded_bytes())
        .map_err(|e| internal(format!("projection parent path: {e}")))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: c_path is NUL-terminated and stats is writable.
    if unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(internal(format!(
            "stat projection free space: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: statvfs succeeded and initialized the structure.
    let stats = unsafe { stats.assume_init() };
    let available = stats.f_bavail.saturating_mul(stats.f_frsize);
    if available < required.saturating_add(FREE_SPACE_RESERVE) {
        return Err(internal(format!(
            "insufficient disk space for bounded model projection: need {required} bytes plus reserve"
        )));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn ensure_disk_space(_parent: &Path, _required: u64) -> Git2DBResult<()> {
    Ok(())
}

fn collect_entries(
    repo: &Repository,
    tree: &Tree<'_>,
    prefix: &Path,
    entries: &mut Vec<BlobEntry>,
    visited_entries: &mut usize,
) -> Git2DBResult<()> {
    if prefix.components().count() > MAX_TREE_DEPTH {
        return Err(internal(format!(
            "model tree exceeds {MAX_TREE_DEPTH}-level depth limit"
        )));
    }
    verify_object(repo, tree.id(), ObjectType::Tree)?;
    for entry in tree {
        *visited_entries = visited_entries
            .checked_add(1)
            .ok_or_else(|| internal("model tree entry count overflow"))?;
        if *visited_entries > MAX_TREE_ENTRIES {
            return Err(internal(format!(
                "model tree exceeds {MAX_TREE_ENTRIES}-entry traversal limit"
            )));
        }
        let name = entry
            .name()
            .ok_or_else(|| internal("non-UTF-8 tree entry"))?;
        let component = Path::new(name);
        if name.contains('/')
            || name.contains('\\')
            || component.components().count() != 1
            || !matches!(component.components().next(), Some(Component::Normal(_)))
        {
            return Err(internal(format!("unsafe tree entry: {name:?}")));
        }
        let path = prefix.join(component);
        match (entry.kind(), entry.filemode()) {
            (Some(ObjectType::Tree), 0o040000) => {
                let child = repo
                    .find_tree(entry.id())
                    .map_err(|e| internal(format!("tree {}: {e}", path.display())))?;
                collect_entries(repo, &child, &path, entries, visited_entries)?;
            }
            (Some(ObjectType::Blob), 0o100644 | 0o100755) => {
                if entries.len() == MAX_FILES {
                    return Err(internal(format!(
                        "model tree exceeds {MAX_FILES} file limit"
                    )));
                }
                entries.push(BlobEntry {
                    path,
                    oid: entry.id(),
                    source_size: 0,
                });
            }
            _ => {
                return Err(internal(format!(
                    "unsupported tree entry {}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn stream_verified_blob(
    git_dir: &Path,
    oid: Oid,
    preflight_size: u64,
    destination: &Path,
) -> Git2DBResult<u64> {
    let mut child = Command::new(GIT_CLI)
        .arg("cat-file")
        .arg("--batch")
        .env_clear()
        .env("GIT_DIR", git_dir)
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| internal(format!("spawn packaged Git: {e}")))?;
    let result = (|| {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| internal("Git stdin unavailable"))?;
        writeln!(stdin, "{oid}").map_err(|e| internal(format!("request Git blob: {e}")))?;
        drop(stdin);
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| internal("Git stdout unavailable"))?;
        let mut output = BufReader::new(stdout);
        let mut header = String::new();
        output
            .read_line(&mut header)
            .map_err(|e| internal(format!("read Git blob header: {e}")))?;
        let mut fields = header.split_whitespace();
        let expected_oid = oid.to_string();
        if fields.next() != Some(expected_oid.as_str()) || fields.next() != Some("blob") {
            return Err(internal(format!("Git returned wrong object for {oid}")));
        }
        let size = fields
            .next()
            .ok_or_else(|| internal("Git blob size missing"))?
            .parse::<u64>()
            .map_err(|e| internal(format!("Git blob size invalid: {e}")))?;
        if fields.next().is_some() {
            return Err(internal("Git blob header has extra fields"));
        }
        // The ODB may change between libgit2's preflight and this independent
        // Git CLI read. Reject before creating a payload, and never write more
        // than the preflighted size even if the stream changes underneath us.
        if size != preflight_size {
            return Err(internal(format!(
                "Git blob size changed after preflight: expected {preflight_size}, got {size}"
            )));
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(|e| internal(format!("create private payload: {e}")))?;
        let copied = std::io::copy(&mut output.by_ref().take(preflight_size), &mut file)
            .map_err(|e| internal(format!("stream Git blob: {e}")))?;
        if copied != size {
            return Err(internal(format!(
                "short Git blob: expected {size}, got {copied}"
            )));
        }
        file.sync_all()
            .map_err(|e| internal(format!("sync Git payload: {e}")))?;
        drop(file);
        let mut terminator = [0u8; 1];
        output
            .read_exact(&mut terminator)
            .map_err(|e| internal(format!("Git blob terminator: {e}")))?;
        if terminator != *b"\n" {
            return Err(internal("Git blob terminator mismatch"));
        }
        Ok(size)
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    let status = child
        .wait()
        .map_err(|e| internal(format!("wait for Git blob: {e}")))?;
    let size = result?;
    if !status.success() {
        return Err(internal("Git cat-file failed"));
    }
    verify_blob_file(oid, destination)?;
    Ok(size)
}

fn verify_blob_file(oid: Oid, path: &Path) -> Git2DBResult<()> {
    let computed = Oid::hash_file(ObjectType::Blob, path)
        .map_err(|e| internal(format!("hash streamed Git blob: {e}")))?;
    if computed != oid {
        return Err(internal(format!(
            "streamed Git blob hash mismatch: expected {oid}, got {computed}"
        )));
    }
    Ok(())
}

#[cfg(all(feature = "xet-storage", target_os = "linux"))]
async fn resolve_pointer_to_file(
    bytes: &[u8],
    path: &Path,
    storage: &mut Option<crate::LfsStorage>,
    owner: &OwnedProjection,
) -> Git2DBResult<()> {
    let text = std::str::from_utf8(bytes).map_err(|e| internal(format!("pointer UTF-8: {e}")))?;
    if text.starts_with("# xet version") {
        return Err(internal("unsupported legacy XET pointer"));
    }
    if storage.is_none() {
        *storage = Some(crate::LfsStorage::new(&crate::XetConfig::default()).await?);
    }
    let storage = storage
        .as_ref()
        .ok_or_else(|| internal("LFS storage unavailable"))?;
    if crate::is_lfs_pointer(text) {
        let pointer = crate::LfsPointer::parse(text)?;
        let written = storage
            .smudge_lfs_pointer_to_file_bounded(&pointer, path)
            .await?;
        let expected = pointer.oid().to_owned();
        let expected_size = pointer.size();
        if written != expected_size {
            return Err(internal(
                "LFS downloader reported a size different from its pointer",
            ));
        }
        let path = path.to_path_buf();
        spawn_owned_verifier(owner, move || {
            verify_lfs_file(&path, &expected, expected_size)
        })
        .await
        .map_err(|e| internal(format!("LFS verify task failed: {e}")))??;
        return Ok(());
    }
    let info: data::XetFileInfo =
        serde_json::from_str(text).map_err(|e| internal(format!("invalid XET pointer: {e}")))?;
    let written = storage
        .smudge_xet_pointer_to_file_bounded(text, path, info.file_size().min(MAX_EXPANDED_BYTES))
        .await?;
    if written != info.file_size() {
        return Err(internal(
            "XET downloader reported a size different from its pointer",
        ));
    }
    let path = path.to_path_buf();
    spawn_owned_verifier(owner, move || verify_xet_file(&path, &info))
        .await
        .map_err(|e| internal(format!("XET verify task failed: {e}")))??;
    Ok(())
}

#[cfg(all(target_os = "linux", any(feature = "xet-storage", test)))]
fn spawn_owned_verifier(
    owner: &OwnedProjection,
    verify: impl FnOnce() -> Git2DBResult<()> + Send + 'static,
) -> tokio::task::JoinHandle<Git2DBResult<()>> {
    let task_owner = owner.clone();
    tokio::task::spawn_blocking(move || {
        let _task_owner = task_owner;
        verify()
    })
}

#[cfg(feature = "xet-storage")]
fn verify_lfs_file(path: &Path, expected_sha: &str, expected_size: u64) -> Git2DBResult<()> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path).map_err(|e| internal(format!("open LFS payload: {e}")))?;
    let mut hash = Sha256::new();
    let mut size = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| internal(format!("hash LFS payload: {e}")))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        size += n as u64;
    }
    let actual = format!("{:x}", hash.finalize());
    if size != expected_size || actual != expected_sha {
        return Err(internal("LFS payload SHA-256/size mismatch"));
    }
    Ok(())
}

#[cfg(feature = "xet-storage")]
fn verify_xet_file(path: &Path, info: &data::XetFileInfo) -> Git2DBResult<()> {
    let mut file = File::open(path).map_err(|e| internal(format!("open XET payload: {e}")))?;
    let mut chunker = deduplication::Chunker::default();
    let mut chunks = Vec::new();
    let mut total = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| internal(format!("read XET payload: {e}")))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        let block = bytes::Bytes::copy_from_slice(&buf[..n]);
        chunks.extend(
            chunker
                .next_block_bytes(&block, false)
                .into_iter()
                .map(|chunk| (chunk.hash, chunk.data.len() as u64)),
        );
    }
    chunks.extend(
        chunker
            .next_block_bytes(&bytes::Bytes::new(), true)
            .into_iter()
            .map(|chunk| (chunk.hash, chunk.data.len() as u64)),
    );
    let expected = info
        .merkle_hash()
        .map_err(|e| internal(format!("XET hash invalid: {e}")))?;
    let actual = merklehash::file_hash(&chunks);
    if total != info.file_size() || actual != expected {
        return Err(internal("XET payload Merkle/size mismatch"));
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn scavenger_preserves_live_projection_then_reclaims_abandoned_bytes() -> Git2DBResult<()> {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let owned = parent.path().join("git2db-pinned-disk-owned123");
        fs::create_dir(&owned).map_err(internal)?;
        fs::set_permissions(&owned, fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let owner_lock = create_projection_owner(&owned)?;
        fs::write(owned.join("stale-payload"), b"disk bytes").map_err(internal)?;
        let foreign_target = parent.path().join("foreign-target");
        fs::create_dir(&foreign_target).map_err(internal)?;
        fs::write(foreign_target.join("keep"), b"foreign").map_err(internal)?;
        std::os::unix::fs::symlink(&foreign_target, owned.join("foreign-link"))
            .map_err(internal)?;

        scavenge_abandoned_projections(parent.path())?;
        assert!(
            owned.join("stale-payload").exists(),
            "live owner must not be scavenged"
        );
        drop(owner_lock); // Simulates the lock release after SIGKILL/OOM.
        scavenge_abandoned_projections(parent.path())?;
        assert!(!owned.exists(), "abandoned payload must be reclaimed");
        assert_eq!(
            fs::read(foreign_target.join("keep")).map_err(internal)?,
            b"foreign",
            "scavenging must not follow a symlink inside an owned projection"
        );
        Ok(())
    }

    #[test]
    fn interrupted_cleanup_keeps_marker_until_remaining_payload_is_reclaimed() -> Git2DBResult<()> {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let owned = parent.path().join("git2db-pinned-disk-interrupted123");
        fs::create_dir(&owned).map_err(internal)?;
        fs::set_permissions(&owned, fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let owner_lock = create_projection_owner(&owned)?;
        fs::write(owned.join("already-removed"), b"first").map_err(internal)?;
        fs::write(owned.join("remaining-payload"), b"second").map_err(internal)?;

        // The process dies during child removal, before it can unlink the
        // owner marker. A later process must still recognize this directory.
        fs::remove_file(owned.join("already-removed")).map_err(internal)?;
        assert!(owned.join(OWNER_FILE).exists());
        scavenge_abandoned_projections(parent.path())?;
        assert!(owned.join("remaining-payload").exists(), "live lock wins");
        drop(owner_lock);
        scavenge_abandoned_projections(parent.path())?;
        assert!(!owned.exists(), "interrupted cleanup must be resumable");
        Ok(())
    }

    #[test]
    fn failed_acquire_guard_keeps_lock_until_cleanup_completes() -> Git2DBResult<()> {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let root = OwnedProjection::create(parent.path(), None)?;
        let path = root.path().to_path_buf();
        fs::write(path.join("partial-payload"), b"not published").map_err(internal)?;
        scavenge_abandoned_projections(parent.path())?;
        assert!(path.join("partial-payload").exists());
        drop(root); // The error path drops this guard before returning Err.
        assert!(!path.exists());
        assert_eq!(fs::read_dir(parent.path()).map_err(internal)?.count(), 0);
        Ok(())
    }

    #[test]
    fn cancelled_acquire_cannot_cleanup_before_blocking_writer_releases_owner() -> Git2DBResult<()>
    {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&slots).try_acquire_owned().map_err(internal)?;
        let root = OwnedProjection::create(parent.path(), Some(permit))?;
        let path = root.path().to_path_buf();
        let writer_owner = root.clone();
        drop(root); // Simulates cancellation while spawn_blocking still runs.
        fs::write(path.join("late-payload"), b"writer still owns root").map_err(internal)?;
        scavenge_abandoned_projections(parent.path())?;
        assert!(path.join("late-payload").exists());
        assert!(
            Arc::clone(&slots).try_acquire_owned().is_err(),
            "late writer must retain the disk-capacity permit"
        );
        drop(writer_owner);
        assert!(!path.exists());
        assert!(Arc::clone(&slots).try_acquire_owned().is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn cancelled_verifier_keeps_unlinked_payload_and_lease_until_reader_finishes(
    ) -> Git2DBResult<()> {
        use std::sync::mpsc;
        use std::time::Duration;

        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&slots).try_acquire_owned().map_err(internal)?;
        let root = OwnedProjection::create(parent.path(), Some(permit))?;
        let path = root.path().to_path_buf();
        let payload = path.join("resolved-payload");
        fs::write(&payload, b"verified bytes").map_err(internal)?;
        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let task = spawn_owned_verifier(&root, move || {
            let mut reader = File::open(&payload).map_err(internal)?;
            let mut prefix = [0u8; 4];
            reader.read_exact(&mut prefix).map_err(internal)?;
            started_tx.send(()).map_err(internal)?;
            finish_rx
                .recv_timeout(Duration::from_secs(5))
                .map_err(internal)?;
            let mut bytes = prefix.to_vec();
            reader.read_to_end(&mut bytes).map_err(internal)?;
            if bytes != b"verified bytes" {
                return Err(internal("detached verifier read changed bytes"));
            }
            Ok(())
        });
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(internal)?;
        drop(task); // The acquire future would drop its JoinHandle on cancellation.
        drop(root);
        scavenge_abandoned_projections(parent.path())?;
        assert!(path.exists(), "detached verifier must retain its root");
        assert!(Arc::clone(&slots).try_acquire_owned().is_err());
        finish_tx.send(()).map_err(internal)?;
        for _ in 0..100 {
            if !path.exists() && Arc::clone(&slots).try_acquire_owned().is_ok() {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err(internal(
            "detached verifier did not release the projection lease",
        ))
    }

    #[test]
    fn ordinary_drop_closes_unlinked_reader_before_releasing_capacity() -> Git2DBResult<()> {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&slots).try_acquire_owned().map_err(internal)?;
        let mut root = OwnedProjection::create(parent.path(), Some(permit))?;
        let payload = root.path().join("payload");
        fs::write(&payload, b"owned bytes").map_err(internal)?;
        let reader = File::open(&payload).map_err(internal)?;
        fs::remove_file(&payload).map_err(internal)?;
        let reader_fd = reader.as_raw_fd();
        let owner = Arc::get_mut(&mut root.0)
            .ok_or_else(|| internal("unexpected retained projection owner"))?;
        owner.observed_reader_fd = Some(reader_fd);
        let projection = DiskPinnedTreeProjection {
            commit: Oid::zero(),
            owner_pid: std::process::id(),
            input_root: root.path().to_path_buf(),
            files: vec![reader],
            _root: root,
        };
        assert!(Arc::clone(&slots).try_acquire_owned().is_err());
        drop(projection); // The owner's drop witness asserts reader_fd is closed.
        assert!(Arc::clone(&slots).try_acquire_owned().is_ok());
        assert!(fs::read_dir(parent.path())
            .map_err(internal)?
            .next()
            .is_none());
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocked_projection_write_keeps_service_reactor_live_and_lease_owned(
    ) -> Git2DBResult<()> {
        use std::os::fd::FromRawFd;
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&slots).try_acquire_owned().map_err(internal)?;
        let mut pipe = [-1; 2];
        if unsafe { libc::pipe2(pipe.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            return Err(internal(std::io::Error::last_os_error()));
        }
        // SAFETY: pipe2 returned two distinct, owned descriptors.
        let mut reader = unsafe { File::from_raw_fd(pipe[0]) };
        let mut writer = unsafe { File::from_raw_fd(pipe[1]) };
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let parent_path = parent.path().to_path_buf();
        let task = tokio::spawn(run_on_projection_worker(move || {
            let root = OwnedProjection::create(&parent_path, Some(permit))?;
            let _ = started_tx.send(root.path().to_path_buf());
            // A full pipe forces the same blocking File::write backpressure as
            // the bounded CAS writer, with no reader until the test releases it.
            writer
                .write_all(&vec![0u8; 8 * 1024 * 1024])
                .map_err(internal)?;
            Ok(())
        }));
        let root_path = started_rx
            .await
            .map_err(|e| internal(format!("projection worker did not start: {e}")))?;
        let (release_tx, release_rx) = mpsc::channel();
        let drainer = std::thread::spawn(move || {
            // A watchdog makes an accidental inline blocking write fail the
            // timing assertion instead of hanging the test indefinitely.
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
            std::io::copy(&mut reader, &mut std::io::sink())
        });
        let start = Instant::now();
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "blocking disk backpressure stalled the current-thread service reactor"
        );
        assert!(!task.is_finished(), "the blocked write lost backpressure");
        task.abort();
        match task.await {
            Err(error) if error.is_cancelled() => {}
            outcome => return Err(internal(format!("caller was not cancelled: {outcome:?}"))),
        }
        // The detached blocking worker still owns the write and disk lease.
        assert!(Arc::clone(&slots).try_acquire_owned().is_err());
        assert!(root_path.exists());
        release_tx.send(()).map_err(internal)?;
        tokio::task::spawn_blocking(move || drainer.join())
            .await
            .map_err(internal)?
            .map_err(|_| internal("pipe drainer panicked"))?
            .map_err(internal)?;
        for _ in 0..100 {
            if !root_path.exists() && Arc::clone(&slots).try_acquire_owned().is_ok() {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err(internal(
            "blocked projection worker did not release its lease",
        ))
    }

    #[test]
    fn scavenger_never_follows_symlinks_or_removes_unmarked_foreign_dirs() -> Git2DBResult<()> {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let foreign = parent.path().join("git2db-pinned-disk-foreign123");
        fs::create_dir(&foreign).map_err(internal)?;
        fs::write(foreign.join("keep"), b"foreign").map_err(internal)?;
        let target = parent.path().join("outside-projection");
        fs::create_dir(&target).map_err(internal)?;
        fs::write(target.join("keep"), b"target").map_err(internal)?;
        let link = parent.path().join("git2db-pinned-disk-symlink123");
        std::os::unix::fs::symlink(&target, &link).map_err(internal)?;

        scavenge_abandoned_projections(parent.path())?;
        assert_eq!(
            fs::read(foreign.join("keep")).map_err(internal)?,
            b"foreign"
        );
        assert_eq!(fs::read(target.join("keep")).map_err(internal)?, b"target");
        assert!(link.is_symlink(), "lookalike symlink must not be removed");
        Ok(())
    }

    fn commit_files(root: &Path, files: &[(&str, &[u8])]) -> Git2DBResult<Oid> {
        let repo = Repository::init(root).map_err(internal)?;
        let mut index = repo.index().map_err(internal)?;
        for &(name, content) in files {
            fs::write(root.join(name), content).map_err(internal)?;
            index.add_path(Path::new(name)).map_err(internal)?;
        }
        index.write().map_err(internal)?;
        let tree = repo
            .find_tree(index.write_tree().map_err(internal)?)
            .map_err(internal)?;
        let signature = git2::Signature::now("test", "test@example.invalid").map_err(internal)?;
        repo.commit(Some("HEAD"), &signature, &signature, "pinned", &tree, &[])
            .map_err(internal)
    }

    #[tokio::test]
    async fn bounded_projection_rejects_distinct_ref_burst_and_reclaims_lease() -> Git2DBResult<()>
    {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let first_repo = tempfile::tempdir().map_err(internal)?;
        let first_oid = commit_files(first_repo.path(), &[("weights.bin", b"first")])?;
        let second_repo = tempfile::tempdir().map_err(internal)?;
        let second_oid = commit_files(second_repo.path(), &[("weights.bin", b"second")])?;

        let first =
            DiskPinnedTreeProjection::acquire_bounded(first_repo.path(), first_oid, parent.path())
                .await?;
        assert!(
            DiskPinnedTreeProjection::acquire_bounded(
                second_repo.path(),
                second_oid,
                parent.path()
            )
            .await
            .is_err(),
            "a different ref must not acquire a second private projection while the lease is held"
        );
        assert_eq!(
            fs::read_dir(parent.path()).map_err(internal)?.count(),
            1,
            "rejected burst must not create a partial projection"
        );
        drop(first);
        assert_eq!(
            fs::read_dir(parent.path()).map_err(internal)?.count(),
            0,
            "dropping the artifact must reclaim projection bytes and lease"
        );
        let second = DiskPinnedTreeProjection::acquire_bounded(
            second_repo.path(),
            second_oid,
            parent.path(),
        )
        .await?;
        assert_eq!(second.commit(), second_oid);
        drop(second);
        Ok(())
    }

    #[tokio::test]
    async fn bounded_projection_rejects_file_count_before_temp_copy() -> Git2DBResult<()> {
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let source = tempfile::tempdir().map_err(internal)?;
        let files = (0..=MAX_FILES)
            .map(|index| (format!("model-{index}.bin"), b"small".as_slice()))
            .collect::<Vec<_>>();
        let borrowed = files
            .iter()
            .map(|(name, bytes)| (name.as_str(), *bytes))
            .collect::<Vec<_>>();
        let commit = commit_files(source.path(), &borrowed)?;
        assert!(
            DiskPinnedTreeProjection::acquire_bounded(source.path(), commit, parent.path())
                .await
                .is_err()
        );
        assert_eq!(
            fs::read_dir(parent.path()).map_err(internal)?.count(),
            0,
            "file-cap rejection must precede projection directory creation"
        );
        Ok(())
    }

    #[tokio::test]
    async fn packed_blob_stays_exact_during_checkout_mutation_and_lazy_thread_read(
    ) -> Git2DBResult<()> {
        let source = tempfile::tempdir().map_err(internal)?;
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let reviewed = b"reviewed packed model bytes";
        let commit = commit_files(source.path(), &[("weights.safetensors", reviewed)])?;
        let repo = Repository::open(source.path()).map_err(internal)?;
        let blob = repo
            .find_commit(commit)
            .map_err(internal)?
            .tree()
            .map_err(internal)?
            .get_name("weights.safetensors")
            .ok_or_else(|| internal("missing test blob"))?
            .id();
        drop(repo);
        let status = Command::new(GIT_CLI)
            .arg("-C")
            .arg(source.path())
            .args(["gc", "--prune=now"])
            .status()
            .map_err(internal)?;
        assert!(status.success());
        let loose = source
            .path()
            .join(".git/objects")
            .join(&blob.to_string()[..2])
            .join(&blob.to_string()[2..]);
        assert!(!loose.exists(), "test blob must be packed");
        let linked_parent = tempfile::tempdir().map_err(internal)?;
        let linked = linked_parent.path().join("linked-model");
        let status = Command::new(GIT_CLI)
            .arg("-C")
            .arg(source.path())
            .args(["worktree", "add", "--detach"])
            .arg(&linked)
            .arg(commit.to_string())
            .status()
            .map_err(internal)?;
        assert!(status.success());
        let mut projection =
            DiskPinnedTreeProjection::acquire(&linked, commit, parent.path()).await?;
        projection.ensure_current_process()?;
        assert_eq!(projection.file_count(), 1);
        let input = projection.root().join("weights.safetensors");
        assert!(
            input.is_file(),
            "loader file-type checks must follow the fd alias"
        );
        scavenge_abandoned_projections(parent.path())?;
        assert!(input.is_file(), "active projection must survive scavenging");
        fs::write(linked.join("weights.safetensors"), b"unreviewed mutation").map_err(internal)?;
        assert_eq!(fs::read(&input).map_err(internal)?, reviewed);
        assert_eq!(
            std::thread::spawn({
                let input = input.clone();
                move || fs::read(input)
            })
            .join()
            .map_err(|_| internal("reader thread panicked"))?
            .map_err(internal)?,
            reviewed
        );
        assert!(OpenOptions::new().write(true).open(&input).is_err());
        let child = Command::new("/usr/bin/cat")
            .arg(&input)
            .output()
            .map_err(internal)?;
        assert!(
            !child.status.success(),
            "subprocess must not inherit projection descriptors"
        );
        assert_ne!(child.stdout, reviewed);
        projection.owner_pid = projection.owner_pid.wrapping_add(1);
        assert!(projection.ensure_current_process().is_err());
        projection.owner_pid = std::process::id();
        assert_eq!(
            fs::read_dir(projection._root.path().join("payloads"))
                .map_err(internal)?
                .count(),
            0,
            "no payload pathname may remain"
        );
        drop(projection);
        assert!(
            !input.exists(),
            "dropping the projection removes lazy aliases"
        );
        assert_eq!(fs::read_dir(parent.path()).map_err(internal)?.count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn failed_acquire_removes_partially_streamed_private_tree() -> Git2DBResult<()> {
        let source = tempfile::tempdir().map_err(internal)?;
        let parent = tempfile::tempdir().map_err(internal)?;
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700)).map_err(internal)?;
        let commit = commit_files(
            source.path(),
            &[
                ("a-config.json", b"verified first file"),
                ("z-pointer", b"# xet version 1\nunsupported\n"),
            ],
        )?;
        assert!(
            DiskPinnedTreeProjection::acquire(source.path(), commit, parent.path())
                .await
                .is_err()
        );
        assert_eq!(fs::read_dir(parent.path()).map_err(internal)?.count(), 0);
        Ok(())
    }

    #[test]
    fn streamed_blob_tamper_and_oid_mismatch_fail_closed() -> Git2DBResult<()> {
        let dir = tempfile::tempdir().map_err(internal)?;
        let original = dir.path().join("blob");
        fs::write(&original, b"reviewed").map_err(internal)?;
        let oid = Oid::hash_file(ObjectType::Blob, &original).map_err(internal)?;
        verify_blob_file(oid, &original)?;
        fs::write(&original, b"replaced").map_err(internal)?;
        assert!(verify_blob_file(oid, &original).is_err());
        let repo = Repository::init(dir.path().join("repo")).map_err(internal)?;
        let other = repo.blob(b"other").map_err(internal)?;
        let destination = dir.path().join("wrong-oid");
        assert!(stream_verified_blob(repo.path(), other, 5, &destination).is_ok());
        assert!(
            stream_verified_blob(repo.path(), oid, 8, &dir.path().join("missing-oid")).is_err()
        );
        Ok(())
    }

    #[test]
    fn stream_rejects_larger_blob_than_preflight_before_writing() -> Git2DBResult<()> {
        let dir = tempfile::tempdir().map_err(internal)?;
        let repo = Repository::init(dir.path().join("repo")).map_err(internal)?;
        let oid = repo
            .blob(b"larger-than-the-preflighted-blob")
            .map_err(internal)?;
        let destination = dir.path().join("must-not-exist");
        assert!(stream_verified_blob(repo.path(), oid, 1, &destination).is_err());
        assert!(
            !destination.exists(),
            "oversized Git blob must be rejected before file creation"
        );
        Ok(())
    }

    #[cfg(feature = "xet-storage")]
    #[test]
    fn streamed_lfs_and_xet_digest_checks_reject_matching_size_tamper() -> Git2DBResult<()> {
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().map_err(internal)?;
        let path = dir.path().join("resolved");
        let bytes = vec![b'R'; 256 * 1024 + 17];
        fs::write(&path, &bytes).map_err(internal)?;
        let sha = format!("{:x}", Sha256::digest(&bytes));
        verify_lfs_file(&path, &sha, bytes.len() as u64)?;
        let mut chunker = deduplication::Chunker::default();
        let chunks = chunker.next_block_bytes(&bytes::Bytes::copy_from_slice(&bytes), true);
        let expected = merklehash::file_hash(
            &chunks
                .iter()
                .map(|chunk| (chunk.hash, chunk.data.len() as u64))
                .collect::<Vec<_>>(),
        );
        let info = data::XetFileInfo::new(expected.hex(), bytes.len() as u64);
        verify_xet_file(&path, &info)?;
        fs::write(&path, vec![b'X'; bytes.len()]).map_err(internal)?;
        assert!(verify_lfs_file(&path, &sha, bytes.len() as u64).is_err());
        assert!(verify_xet_file(&path, &info).is_err());
        Ok(())
    }
}
