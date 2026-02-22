//! Scoped path containment for non-Linux Unix (macOS, FreeBSD, etc.).
//!
//! Uses `safe_path::scoped_join` for per-component symlink resolution
//! constrained within the root directory. This is significantly better than
//! `CanonicalRoot` (which uses `fs::canonicalize` — TOCTOU-vulnerable) because
//! it resolves symlinks relative to the contained root, not the real filesystem root.
//!
//! Still not as strong as Linux's `pathrs` (kernel-enforced via `openat2`),
//! but provides much better protection on macOS/FreeBSD where `openat2` is
//! unavailable.

use crate::error::FsError;
use crate::path::validate_relative_path;
use crate::types::{DirEntry, FsHandle, OpenMode, Stat};
use crate::ContainedFs;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Scoped path containment using `safe-path` (Unix non-Linux).
///
/// Uses `safe_path::scoped_join` which walks path components one at a time,
/// resolving symlinks at each step while keeping resolution constrained
/// within the root directory (similar to userspace `chroot`).
///
/// This protects against:
/// - `..` traversal (clamped at root)
/// - Symlinks pointing outside root (resolved relative to root)
/// - Symlink loops (max 255 depth)
///
/// TOCTOU note: While better than `canonicalize`, there is still a per-component
/// race window between resolving and using. For full TOCTOU protection, use
/// `pathrs` on Linux.
pub struct ScopedRoot {
    root: PathBuf,
}

impl ScopedRoot {
    /// Create a new ScopedRoot for the given root path.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(path: &Path) -> Result<Arc<dyn ContainedFs>, FsError> {
        let root = fs::canonicalize(path).map_err(FsError::Io)?;
        Ok(Arc::new(Self { root }))
    }

    /// Resolve a relative path within the root using safe-path's scoped_join.
    ///
    /// For existing paths: uses `scoped_join` which resolves symlinks
    /// per-component, constrained within root.
    ///
    /// For non-existent paths: validates relative components, then joins
    /// with root (symlinks in parent directories are still resolved safely).
    fn resolve(&self, relative: &str) -> Result<PathBuf, FsError> {
        let validated = validate_relative_path(relative)?;

        // Use safe_path::scoped_join for symlink-aware resolution
        // This resolves symlinks at each path component, keeping everything
        // within self.root (like a userspace chroot)
        match safe_path::scoped_join(&self.root, &validated) {
            Ok(resolved) => {
                // Double-check containment (defense-in-depth)
                if !resolved.starts_with(&self.root) {
                    return Err(FsError::path_escape("escaped root via symlink"));
                }
                Ok(resolved)
            }
            Err(e) => {
                // scoped_join fails if root doesn't exist or symlink loop detected.
                // For non-existent leaf paths, the parent must exist for scoped_join
                // to work. Try resolving the parent and appending the filename.
                if let Some(parent) = validated.parent() {
                    if parent.as_os_str().is_empty() {
                        // Top-level file: just join with root
                        Ok(self.root.join(&validated))
                    } else {
                        match safe_path::scoped_join(&self.root, parent) {
                            Ok(resolved_parent) => {
                                if !resolved_parent.starts_with(&self.root) {
                                    return Err(FsError::path_escape("escaped root via parent symlink"));
                                }
                                let filename = validated.file_name().unwrap_or_default();
                                Ok(resolved_parent.join(filename))
                            }
                            Err(_) => {
                                // Parent also doesn't exist — fall back to simple join
                                // with validation already done
                                Ok(self.root.join(&validated))
                            }
                        }
                    }
                } else {
                    Err(FsError::Io(e))
                }
            }
        }
    }
}

impl ContainedFs for ScopedRoot {
    fn walk(&self, path: &str) -> Result<FsHandle, FsError> {
        let resolved = self.resolve(path)?;
        Ok(FsHandle::from_path(resolved, path.to_owned()))
    }

    fn open(&self, path: &str, mode: OpenMode) -> Result<File, FsError> {
        let resolved = self.resolve(path)?;
        mode.to_open_options().open(&resolved).map_err(FsError::Io)
    }

    fn create(&self, path: &str, mode: OpenMode) -> Result<File, FsError> {
        let resolved = self.resolve(path)?;
        mode.to_create_options()
            .open(&resolved)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    FsError::AlreadyExists(path.to_owned())
                } else {
                    FsError::Io(e)
                }
            })
    }

    fn mkdir(&self, path: &str) -> Result<(), FsError> {
        let resolved = self.resolve(path)?;
        fs::create_dir(&resolved).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                FsError::AlreadyExists(path.to_owned())
            } else {
                FsError::Io(e)
            }
        })
    }

    fn mkdir_all(&self, path: &str) -> Result<(), FsError> {
        let resolved = self.resolve(path)?;
        fs::create_dir_all(&resolved).map_err(FsError::Io)
    }

    fn stat(&self, path: &str) -> Result<Stat, FsError> {
        let resolved = self.resolve(path)?;
        let meta = fs::metadata(&resolved).map_err(FsError::Io)?;
        let name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok(Stat::from_metadata(&meta, name))
    }

    fn rename(&self, src: &str, dst: &str) -> Result<(), FsError> {
        let src_path = self.resolve(src)?;
        let dst_path = self.resolve(dst)?;
        fs::rename(&src_path, &dst_path).map_err(FsError::Io)
    }

    fn remove(&self, path: &str) -> Result<(), FsError> {
        let resolved = self.resolve(path)?;
        if resolved.is_dir() {
            fs::remove_dir(&resolved).map_err(FsError::Io)
        } else {
            fs::remove_file(&resolved).map_err(FsError::Io)
        }
    }

    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, FsError> {
        let resolved = self.resolve(path)?;
        let mut entries = Vec::new();
        for entry in fs::read_dir(&resolved).map_err(FsError::Io)? {
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
        let src_path = self.resolve(src)?;
        let dst_path = self.resolve(dst)?;
        reflink_copy::reflink_or_copy(&src_path, &dst_path)
            .map(|_| ())
            .map_err(|e| FsError::Io(std::io::Error::other(e)))
    }
}
