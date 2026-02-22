//! Linux pathrs-based containment (kernel-enforced via openat2).
//!
//! On Linux 5.6+, pathrs uses `openat2` with `RESOLVE_IN_ROOT` which atomically
//! resolves paths within the root. On older kernels, it falls back to O_PATH
//! walk with `/proc/self/fd` verification.

use crate::error::FsError;
use crate::path::validate_relative_path;
use crate::types::{DirEntry, FsHandle, OpenMode, Stat};
use crate::ContainedFs;
use pathrs::flags::{OpenFlags, RenameFlags};
use std::fs::{self, File};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Kernel-enforced path containment using pathrs + openat2(RESOLVE_IN_ROOT).
pub struct PathrsRoot {
    root: pathrs::Root,
    /// Cached root path for operations that need it.
    root_path: PathBuf,
}

impl PathrsRoot {
    /// Create a new PathrsRoot for the given root path.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(path: &Path) -> Result<Arc<dyn ContainedFs>, FsError> {
        let root_path = fs::canonicalize(path).map_err(FsError::Io)?;
        let root = pathrs::Root::open(path).map_err(pathrs_to_fs_error)?;
        Ok(Arc::new(Self { root, root_path }))
    }
}

fn pathrs_to_fs_error(e: pathrs::error::Error) -> FsError {
    let msg = e.to_string();
    if msg.contains("SafetyViolation") || msg.contains("safety violation") {
        FsError::PathEscape(msg)
    } else if msg.contains("ENOENT") || msg.contains("No such file") {
        FsError::NotFound(msg)
    } else if msg.contains("EACCES") || msg.contains("EPERM") || msg.contains("Permission denied")
    {
        FsError::PermissionDenied(msg)
    } else if msg.contains("EEXIST") || msg.contains("File exists") {
        FsError::AlreadyExists(msg)
    } else {
        FsError::Io(std::io::Error::other(msg))
    }
}

impl ContainedFs for PathrsRoot {
    fn walk(&self, path: &str) -> Result<FsHandle, FsError> {
        let handle = self.root.resolve(path).map_err(pathrs_to_fs_error)?;
        Ok(FsHandle::from_pathrs(handle, path.to_owned()))
    }

    fn open(&self, path: &str, mode: OpenMode) -> Result<File, FsError> {
        let mut flags = OpenFlags::O_CLOEXEC;
        if mode.is_write() {
            flags |= OpenFlags::O_RDWR;
        } else {
            flags |= OpenFlags::O_RDONLY;
        }
        if mode.is_truncate() {
            flags |= OpenFlags::O_TRUNC;
        }
        if mode.is_append() {
            flags |= OpenFlags::O_APPEND;
        }

        self.root
            .open_subpath(path, flags)
            .map_err(pathrs_to_fs_error)
    }

    fn create(&self, path: &str, mode: OpenMode) -> Result<File, FsError> {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o644);

        let mut flags = OpenFlags::O_CLOEXEC;
        if mode.is_write() {
            flags |= OpenFlags::O_RDWR;
        } else {
            flags |= OpenFlags::O_RDONLY;
        }
        if mode.is_truncate() {
            flags |= OpenFlags::O_TRUNC;
        }
        if mode.is_append() {
            flags |= OpenFlags::O_APPEND;
        }
        if mode.is_exclusive() {
            flags |= OpenFlags::O_EXCL;
        }

        self.root
            .create_file(path, flags, &perms)
            .map_err(pathrs_to_fs_error)
    }

    fn mkdir(&self, path: &str) -> Result<(), FsError> {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o755);
        // pathrs only has mkdir_all; for single-level, use it anyway
        self.root
            .mkdir_all(path, &perms)
            .map_err(pathrs_to_fs_error)?;
        Ok(())
    }

    fn mkdir_all(&self, path: &str) -> Result<(), FsError> {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o755);
        self.root
            .mkdir_all(path, &perms)
            .map_err(pathrs_to_fs_error)?;
        Ok(())
    }

    fn stat(&self, path: &str) -> Result<Stat, FsError> {
        let handle = self.root.resolve(path).map_err(pathrs_to_fs_error)?;
        let file: File = handle
            .reopen(OpenFlags::O_RDONLY | OpenFlags::O_CLOEXEC)
            .map_err(pathrs_to_fs_error)?;
        let meta = file.metadata().map_err(FsError::Io)?;
        let name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok(Stat::from_metadata(&meta, name))
    }

    fn rename(&self, src: &str, dst: &str) -> Result<(), FsError> {
        self.root
            .rename(src, dst, RenameFlags::empty())
            .map_err(pathrs_to_fs_error)
    }

    fn remove(&self, path: &str) -> Result<(), FsError> {
        // Try file removal first, then directory
        match self.root.remove_file(path) {
            Ok(()) => Ok(()),
            Err(_) => self.root.remove_dir(path).map_err(pathrs_to_fs_error),
        }
    }

    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, FsError> {
        let handle = self.root.resolve(path).map_err(pathrs_to_fs_error)?;
        let dir_file: File = handle
            .reopen(OpenFlags::O_RDONLY | OpenFlags::O_DIRECTORY | OpenFlags::O_CLOEXEC)
            .map_err(pathrs_to_fs_error)?;

        // Use /proc/self/fd to get the real path for read_dir
        let fd = dir_file.as_raw_fd();
        let real_path = fs::read_link(format!("/proc/self/fd/{}", fd)).map_err(FsError::Io)?;

        let mut entries = Vec::new();
        for entry in fs::read_dir(real_path).map_err(FsError::Io)? {
            let entry = entry.map_err(FsError::Io)?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "." || name == ".." {
                continue;
            }
            let meta = entry.metadata().map_err(FsError::Io)?;
            entries.push(DirEntry {
                name,
                is_dir: meta.is_dir(),
                size: meta.len(),
            });
        }
        Ok(entries)
    }

    fn copy(&self, src: &str, dst: &str) -> Result<(), FsError> {
        // FIX S3: Resolve BOTH src AND dst within root using pathrs
        let src_handle = self.root.resolve(src).map_err(pathrs_to_fs_error)?;
        let src_file: File = src_handle
            .reopen(OpenFlags::O_PATH | OpenFlags::O_CLOEXEC)
            .map_err(pathrs_to_fs_error)?;
        let src_fd = src_file.as_raw_fd();
        let src_path =
            fs::read_link(format!("/proc/self/fd/{}", src_fd)).map_err(FsError::Io)?;

        // Resolve dst parent within root, then construct full dst path
        let dst_validated = validate_relative_path(dst)?;
        let dst_parent = dst_validated.parent().map(|p| p.to_string_lossy().to_string());

        let dst_full = if let Some(parent_str) = dst_parent {
            if !parent_str.is_empty() {
                // Resolve parent directory within root
                let parent_handle = self
                    .root
                    .resolve(&parent_str)
                    .map_err(pathrs_to_fs_error)?;
                let parent_file: File = parent_handle
                    .reopen(OpenFlags::O_PATH | OpenFlags::O_CLOEXEC)
                    .map_err(pathrs_to_fs_error)?;
                let parent_fd = parent_file.as_raw_fd();
                let parent_path = fs::read_link(format!("/proc/self/fd/{}", parent_fd))
                    .map_err(FsError::Io)?;
                parent_path.join(
                    dst_validated
                        .file_name()
                        .unwrap_or_default(),
                )
            } else {
                self.root_path.join(&dst_validated)
            }
        } else {
            self.root_path.join(&dst_validated)
        };

        reflink_copy::reflink_or_copy(&src_path, &dst_full)
            .map(|_| ())
            .map_err(|e| FsError::Io(std::io::Error::other(e)))
    }
}
