//! Canonicalize-based path containment (cross-platform fallback).
//!
//! Uses `fs::canonicalize` + prefix check for path containment.
//! This is TOCTOU-vulnerable but functional for development/macOS.

use crate::error::FsError;
use crate::path::validate_relative_path;
use crate::types::{DirEntry, FsHandle, OpenMode, Stat};
use crate::ContainedFs;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Canonicalize-based path containment (non-Linux fallback).
///
/// Uses `fs::canonicalize` + prefix check. TOCTOU-vulnerable but
/// acceptable for development/macOS.
pub struct CanonicalRoot {
    root: PathBuf,
}

impl CanonicalRoot {
    /// Create a new CanonicalRoot for the given root path.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(path: &Path) -> Result<Arc<dyn ContainedFs>, FsError> {
        let root = fs::canonicalize(path).map_err(FsError::Io)?;
        Ok(Arc::new(Self { root }))
    }

    /// Validate and resolve a relative path within the root.
    fn resolve(&self, relative: &str) -> Result<PathBuf, FsError> {
        let validated = validate_relative_path(relative)?;
        let full = self.root.join(&validated);
        // Try to canonicalize; if the file doesn't exist yet, check the parent
        let resolved = if full.exists() {
            fs::canonicalize(&full).map_err(FsError::Io)?
        } else {
            // For non-existent files, canonicalize the parent and append the filename
            if let Some(parent) = full.parent() {
                if parent.exists() {
                    let canonical_parent = fs::canonicalize(parent).map_err(FsError::Io)?;
                    if !canonical_parent.starts_with(&self.root) {
                        return Err(FsError::path_escape("escaped root via parent"));
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
            return Err(FsError::path_escape("escaped root"));
        }
        Ok(resolved)
    }
}

impl ContainedFs for CanonicalRoot {
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
