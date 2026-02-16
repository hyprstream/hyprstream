//! Path-contained filesystem operations.
//!
//! Provides a `ContainedRoot` trait that abstracts path-safe filesystem operations.
//! All operations are contained within a root directory — paths cannot escape via
//! symlinks, `..` traversal, or other techniques.
//!
//! - **Linux**: Uses `pathrs` with `openat2(RESOLVE_IN_ROOT)` for kernel-enforced
//!   containment. This is atomic and TOCTOU-proof.
//! - **Non-Linux**: Uses `canonicalize` + prefix check. Best-effort (TOCTOU-vulnerable),
//!   acceptable for development/macOS.

use crate::services::generated::registry_client::FsDirEntryInfo;
use std::fs::{self, File, OpenOptions};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

/// Filesystem service error type.
#[derive(Debug, Error)]
pub enum FsServiceError {
    /// Bad file descriptor.
    #[error("Bad file descriptor: {0}")]
    BadFd(u32),
    /// Path or file not found.
    #[error("Not found: {0}")]
    NotFound(String),
    /// Permission denied (FD not owned by caller, or access denied).
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Path escaped containment root (symlink or traversal attack).
    #[error("Path containment violation: {0}")]
    PathEscape(String),
    /// Resource limit exceeded (too many FDs, IO size too large).
    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),
    /// Transport / communication error.
    #[error("Transport error: {0}")]
    Transport(String),
    /// Service is unavailable.
    #[error("Service unavailable")]
    Unavailable,
}

/// Trait abstracting path-contained filesystem operations.
///
/// All paths passed to methods are relative to the contained root.
/// Implementations ensure no path can escape the root directory.
pub trait ContainedRoot: Send + Sync {
    /// Open a file relative to the root.
    fn open_file(
        &self,
        relative: &str,
        write: bool,
        create: bool,
        truncate: bool,
        append: bool,
        exclusive: bool,
    ) -> Result<File, FsServiceError>;

    /// Get file metadata.
    fn stat(&self, relative: &str) -> Result<fs::Metadata, FsServiceError>;

    /// Create a single directory.
    fn mkdir(&self, relative: &str) -> Result<(), FsServiceError>;

    /// Create directories recursively.
    fn mkdir_all(&self, relative: &str) -> Result<(), FsServiceError>;

    /// Remove a file.
    fn remove_file(&self, relative: &str) -> Result<(), FsServiceError>;

    /// Remove an empty directory.
    fn remove_dir(&self, relative: &str) -> Result<(), FsServiceError>;

    /// Rename a file or directory.
    fn rename(&self, src: &str, dst: &str) -> Result<(), FsServiceError>;

    /// List directory entries.
    fn list_dir(&self, relative: &str) -> Result<Vec<FsDirEntryInfo>, FsServiceError>;

    /// Copy a file (uses reflink/COW when available).
    fn copy_file(&self, src: &str, dst: &str) -> Result<(), FsServiceError>;
}

/// Create a ContainedRoot for the given worktree path.
///
/// On Linux: returns PathrsContainedRoot (kernel-enforced via openat2).
/// On other platforms: returns CanonicalContainedRoot (best-effort).
pub fn open_contained_root(root_path: &Path) -> Result<Box<dyn ContainedRoot>, FsServiceError> {
    #[cfg(target_os = "linux")]
    {
        Ok(Box::new(PathrsContainedRoot::new(root_path)?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(Box::new(CanonicalContainedRoot::new(root_path)?))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linux: pathrs-based containment (kernel-enforced via openat2)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use pathrs::flags::{OpenFlags, RenameFlags};
    use std::os::unix::io::AsRawFd;

    /// Kernel-enforced path containment using pathrs + openat2(RESOLVE_IN_ROOT).
    ///
    /// On Linux 5.6+, pathrs uses `openat2` with `RESOLVE_IN_ROOT` which atomically
    /// resolves paths within the root. On older kernels, it falls back to O_PATH
    /// walk with `/proc/self/fd` verification.
    pub struct PathrsContainedRoot {
        root: pathrs::Root,
        /// Cached root path for operations that need it (list_dir, copy).
        root_path: PathBuf,
    }

    impl PathrsContainedRoot {
        pub fn new(path: &Path) -> Result<Self, FsServiceError> {
            let root_path = fs::canonicalize(path).map_err(FsServiceError::Io)?;
            let root = pathrs::Root::open(path).map_err(pathrs_to_fs_error)?;
            Ok(Self { root, root_path })
        }
    }

    fn pathrs_to_fs_error(e: pathrs::error::Error) -> FsServiceError {
        let msg = e.to_string();
        if msg.contains("SafetyViolation") || msg.contains("safety violation") {
            FsServiceError::PathEscape(msg)
        } else if msg.contains("ENOENT") || msg.contains("No such file") {
            FsServiceError::NotFound(msg)
        } else if msg.contains("EACCES") || msg.contains("EPERM") || msg.contains("Permission denied") {
            FsServiceError::PermissionDenied(msg)
        } else {
            FsServiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, msg))
        }
    }

    impl ContainedRoot for PathrsContainedRoot {
        fn open_file(
            &self,
            relative: &str,
            write: bool,
            create: bool,
            truncate: bool,
            append: bool,
            exclusive: bool,
        ) -> Result<File, FsServiceError> {
            if create {
                // openat2(RESOLVE_IN_ROOT) does not support O_CREAT in one-shot mode.
                // Use pathrs::Root::create_file() which safely resolves the parent
                // directory first, then calls openat(O_CREAT) on the final component.
                use std::os::unix::fs::PermissionsExt;
                let perms = std::fs::Permissions::from_mode(0o644);

                let mut flags = OpenFlags::O_CLOEXEC;
                if write {
                    flags |= OpenFlags::O_RDWR;
                } else {
                    flags |= OpenFlags::O_RDONLY;
                }
                if truncate {
                    flags |= OpenFlags::O_TRUNC;
                }
                if append {
                    flags |= OpenFlags::O_APPEND;
                }
                if exclusive {
                    flags |= OpenFlags::O_EXCL;
                }

                self.root
                    .create_file(relative, flags, &perms)
                    .map_err(pathrs_to_fs_error)
            } else {
                // No creation needed — use open_subpath for one-shot resolve + open
                let mut flags = OpenFlags::O_CLOEXEC;
                if write {
                    flags |= OpenFlags::O_RDWR;
                } else {
                    flags |= OpenFlags::O_RDONLY;
                }
                if truncate {
                    flags |= OpenFlags::O_TRUNC;
                }
                if append {
                    flags |= OpenFlags::O_APPEND;
                }

                self.root
                    .open_subpath(relative, flags)
                    .map_err(pathrs_to_fs_error)
            }
        }

        fn stat(&self, relative: &str) -> Result<fs::Metadata, FsServiceError> {
            let handle = self.root.resolve(relative).map_err(pathrs_to_fs_error)?;
            let file: File = handle
                .reopen(OpenFlags::O_RDONLY | OpenFlags::O_CLOEXEC)
                .map_err(pathrs_to_fs_error)?;
            file.metadata().map_err(FsServiceError::Io)
        }

        fn mkdir(&self, relative: &str) -> Result<(), FsServiceError> {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o755);
            // pathrs only has mkdir_all; for single-level, use it anyway
            self.root
                .mkdir_all(relative, &perms)
                .map_err(pathrs_to_fs_error)?;
            Ok(())
        }

        fn mkdir_all(&self, relative: &str) -> Result<(), FsServiceError> {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o755);
            self.root
                .mkdir_all(relative, &perms)
                .map_err(pathrs_to_fs_error)?;
            Ok(())
        }

        fn remove_file(&self, relative: &str) -> Result<(), FsServiceError> {
            self.root
                .remove_file(relative)
                .map_err(pathrs_to_fs_error)
        }

        fn remove_dir(&self, relative: &str) -> Result<(), FsServiceError> {
            self.root
                .remove_dir(relative)
                .map_err(pathrs_to_fs_error)
        }

        fn rename(&self, src: &str, dst: &str) -> Result<(), FsServiceError> {
            self.root
                .rename(src, dst, RenameFlags::empty())
                .map_err(pathrs_to_fs_error)
        }

        fn list_dir(&self, relative: &str) -> Result<Vec<FsDirEntryInfo>, FsServiceError> {
            let handle = self.root.resolve(relative).map_err(pathrs_to_fs_error)?;
            let dir_file: File = handle
                .reopen(OpenFlags::O_RDONLY | OpenFlags::O_DIRECTORY | OpenFlags::O_CLOEXEC)
                .map_err(pathrs_to_fs_error)?;

            // Use /proc/self/fd to get the real path for read_dir
            let fd = dir_file.as_raw_fd();
            let real_path = fs::read_link(format!("/proc/self/fd/{}", fd))
                .map_err(FsServiceError::Io)?;

            let mut entries = Vec::new();
            for entry in fs::read_dir(real_path).map_err(FsServiceError::Io)? {
                let entry = entry.map_err(FsServiceError::Io)?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name == "." || name == ".." {
                    continue;
                }
                let meta = entry.metadata().map_err(FsServiceError::Io)?;
                entries.push(FsDirEntryInfo {
                    name,
                    is_dir: meta.is_dir(),
                    size: meta.len(),
                });
            }
            Ok(entries)
        }

        fn copy_file(&self, src: &str, dst: &str) -> Result<(), FsServiceError> {
            // Resolve src within root to get real path
            let src_handle = self.root.resolve(src).map_err(pathrs_to_fs_error)?;
            let src_file: File = src_handle
                .reopen(OpenFlags::O_PATH | OpenFlags::O_CLOEXEC)
                .map_err(pathrs_to_fs_error)?;
            let src_fd = src_file.as_raw_fd();
            let src_path = fs::read_link(format!("/proc/self/fd/{}", src_fd))
                .map_err(FsServiceError::Io)?;

            // Validate dst within root
            let dst_validated = validate_relative_path(dst)?;
            let dst_full = self.root_path.join(&dst_validated);

            reflink_copy::reflink_or_copy(&src_path, &dst_full)
                .map(|_| ())
                .map_err(|e| FsServiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::PathrsContainedRoot;

// ─────────────────────────────────────────────────────────────────────────────
// Non-Linux: Canonicalize-based containment (best-effort, TOCTOU-vulnerable)
// ─────────────────────────────────────────────────────────────────────────────

/// Canonicalize-based path containment (non-Linux fallback).
///
/// Uses `fs::canonicalize` + prefix check for path containment.
/// This is TOCTOU-vulnerable but functional for development/macOS.
pub struct CanonicalContainedRoot {
    root: PathBuf,
}

impl CanonicalContainedRoot {
    pub fn new(path: &Path) -> Result<Self, FsServiceError> {
        let root = fs::canonicalize(path).map_err(FsServiceError::Io)?;
        Ok(Self { root })
    }

    /// Validate and resolve a relative path within the root.
    fn resolve(&self, relative: &str) -> Result<PathBuf, FsServiceError> {
        let validated = validate_relative_path(relative)?;
        let full = self.root.join(&validated);
        // Try to canonicalize; if the file doesn't exist yet, check the parent
        let resolved = if full.exists() {
            fs::canonicalize(&full).map_err(FsServiceError::Io)?
        } else {
            // For non-existent files, canonicalize the parent and append the filename
            if let Some(parent) = full.parent() {
                if parent.exists() {
                    let canonical_parent = fs::canonicalize(parent).map_err(FsServiceError::Io)?;
                    if !canonical_parent.starts_with(&self.root) {
                        return Err(FsServiceError::PathEscape("escaped root via parent".into()));
                    }
                    canonical_parent.join(full.file_name().unwrap_or_default())
                } else {
                    full
                }
            } else {
                full
            }
        };
        if !resolved.starts_with(&self.root) {
            return Err(FsServiceError::PathEscape("escaped root".into()));
        }
        Ok(resolved)
    }
}

impl ContainedRoot for CanonicalContainedRoot {
    fn open_file(
        &self,
        relative: &str,
        write: bool,
        create: bool,
        truncate: bool,
        append: bool,
        exclusive: bool,
    ) -> Result<File, FsServiceError> {
        let path = self.resolve(relative)?;
        let mut opts = OpenOptions::new();
        opts.read(true);
        if write {
            opts.write(true);
        }
        if create {
            opts.create(true);
        }
        if truncate {
            opts.truncate(true);
        }
        if append {
            opts.append(true);
        }
        if exclusive {
            opts.create_new(true);
        }
        opts.open(&path).map_err(FsServiceError::Io)
    }

    fn stat(&self, relative: &str) -> Result<fs::Metadata, FsServiceError> {
        let path = self.resolve(relative)?;
        fs::metadata(&path).map_err(FsServiceError::Io)
    }

    fn mkdir(&self, relative: &str) -> Result<(), FsServiceError> {
        let path = self.resolve(relative)?;
        fs::create_dir(&path).map_err(FsServiceError::Io)
    }

    fn mkdir_all(&self, relative: &str) -> Result<(), FsServiceError> {
        let path = self.resolve(relative)?;
        fs::create_dir_all(&path).map_err(FsServiceError::Io)
    }

    fn remove_file(&self, relative: &str) -> Result<(), FsServiceError> {
        let path = self.resolve(relative)?;
        fs::remove_file(&path).map_err(FsServiceError::Io)
    }

    fn remove_dir(&self, relative: &str) -> Result<(), FsServiceError> {
        let path = self.resolve(relative)?;
        fs::remove_dir(&path).map_err(FsServiceError::Io)
    }

    fn rename(&self, src: &str, dst: &str) -> Result<(), FsServiceError> {
        let src_path = self.resolve(src)?;
        let dst_path = self.resolve(dst)?;
        fs::rename(&src_path, &dst_path).map_err(FsServiceError::Io)
    }

    fn list_dir(&self, relative: &str) -> Result<Vec<FsDirEntryInfo>, FsServiceError> {
        let path = self.resolve(relative)?;
        let mut entries = Vec::new();
        for entry in fs::read_dir(&path).map_err(FsServiceError::Io)? {
            let entry = entry.map_err(FsServiceError::Io)?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "." || name == ".." {
                continue;
            }
            let meta = entry.metadata().map_err(FsServiceError::Io)?;
            entries.push(FsDirEntryInfo {
                name,
                is_dir: meta.is_dir(),
                size: meta.len(),
            });
        }
        Ok(entries)
    }

    fn copy_file(&self, src: &str, dst: &str) -> Result<(), FsServiceError> {
        let src_path = self.resolve(src)?;
        let dst_path = self.resolve(dst)?;
        reflink_copy::reflink_or_copy(&src_path, &dst_path)
            .map(|_| ())
            .map_err(|e| FsServiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a relative path doesn't contain traversal components.
fn validate_relative_path(relative: &str) -> Result<PathBuf, FsServiceError> {
    let rel = Path::new(relative);
    if rel.is_absolute() {
        return Err(FsServiceError::PathEscape("absolute path rejected".into()));
    }
    for component in rel.components() {
        match component {
            Component::ParentDir => {
                return Err(FsServiceError::PathEscape("path traversal via '..' rejected".into()));
            }
            Component::RootDir => {
                return Err(FsServiceError::PathEscape("root dir component rejected".into()));
            }
            _ => {}
        }
    }
    Ok(rel.to_path_buf())
}
