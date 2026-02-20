//! Core types for contained filesystem operations.

use crate::error::FsError;
use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────────────────
// OpenMode — bitflags for 9P-style open modes
// ─────────────────────────────────────────────────────────────────────────────

bitflags::bitflags! {
    /// Open mode flags (9P-aligned).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct OpenMode: u8 {
        /// Open for reading.
        const OREAD   = 0x00;
        /// Open for writing.
        const OWRITE  = 0x01;
        /// Open for read+write.
        const ORDWR   = 0x02;
        /// Truncate file on open.
        const OTRUNC  = 0x10;
        /// Append mode.
        const OAPPEND = 0x20;
        /// Exclusive creation (fail if exists).
        const OEXCL   = 0x40;
    }
}

impl OpenMode {
    /// Check if write access is requested.
    pub fn is_write(&self) -> bool {
        self.contains(Self::OWRITE) || self.contains(Self::ORDWR)
    }

    /// Check if truncation is requested.
    pub fn is_truncate(&self) -> bool {
        self.contains(Self::OTRUNC)
    }

    /// Check if append is requested.
    pub fn is_append(&self) -> bool {
        self.contains(Self::OAPPEND)
    }

    /// Check if exclusive creation is requested.
    pub fn is_exclusive(&self) -> bool {
        self.contains(Self::OEXCL)
    }

    /// Convert to `std::fs::OpenOptions`.
    pub fn to_open_options(&self) -> OpenOptions {
        let mut opts = OpenOptions::new();
        opts.read(true);
        if self.is_write() {
            opts.write(true);
        }
        if self.is_truncate() {
            opts.truncate(true);
        }
        if self.is_append() {
            opts.append(true);
        }
        opts
    }

    /// Convert to `std::fs::OpenOptions` with create enabled.
    pub fn to_create_options(&self) -> OpenOptions {
        let mut opts = self.to_open_options();
        if self.is_exclusive() {
            opts.create_new(true);
        } else {
            opts.create(true);
        }
        opts
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FsHandle — opaque handle from walk operations
// ─────────────────────────────────────────────────────────────────────────────

/// Handle returned by `walk()` for 9P walk state.
///
/// Opaque struct with platform-specific internals.
/// On Linux: wraps a pathrs::Handle (O_PATH fd from openat2).
/// On non-Linux: wraps a validated PathBuf.
pub struct FsHandle {
    rel_path: String,
    inner: FsHandleInner,
}

enum FsHandleInner {
    #[cfg(target_os = "linux")]
    Pathrs(pathrs::Handle),
    Path(PathBuf),
}

impl FsHandle {
    /// Create a handle from a pathrs Handle (Linux).
    #[cfg(target_os = "linux")]
    pub(crate) fn from_pathrs(handle: pathrs::Handle, rel_path: String) -> Self {
        Self {
            rel_path,
            inner: FsHandleInner::Pathrs(handle),
        }
    }

    /// Create a handle from a validated PathBuf.
    pub(crate) fn from_path(path: PathBuf, rel_path: String) -> Self {
        Self {
            rel_path,
            inner: FsHandleInner::Path(path),
        }
    }

    /// Get the relative path from the contained root to this handle.
    pub fn rel_path(&self) -> &str {
        &self.rel_path
    }

    /// Compute the relative path for a child entry under this directory.
    ///
    /// Returns `name` if this handle is at the root (`.` or empty),
    /// otherwise returns `"{rel_path}/{name}"`.
    pub fn child_rel_path(&self, name: &str) -> String {
        let parent = self.rel_path();
        if parent == "." || parent.is_empty() {
            name.to_owned()
        } else {
            format!("{}/{}", parent, name)
        }
    }

    /// Get file metadata from this handle.
    pub fn metadata(&self) -> Result<fs::Metadata, FsError> {
        match &self.inner {
            #[cfg(target_os = "linux")]
            FsHandleInner::Pathrs(handle) => {
                use pathrs::flags::OpenFlags;
                let file: File = handle
                    .reopen(OpenFlags::O_RDONLY | OpenFlags::O_CLOEXEC)
                    .map_err(|e| FsError::Io(std::io::Error::other(e.to_string())))?;
                file.metadata().map_err(FsError::Io)
            }
            FsHandleInner::Path(path) => {
                fs::metadata(path).map_err(FsError::Io)
            }
        }
    }

    /// Open this handle as a real file for I/O.
    pub fn open_file(&self, write: bool) -> Result<File, FsError> {
        match &self.inner {
            #[cfg(target_os = "linux")]
            FsHandleInner::Pathrs(handle) => {
                use pathrs::flags::OpenFlags;
                let mut flags = OpenFlags::O_CLOEXEC;
                if write {
                    flags |= OpenFlags::O_RDWR;
                } else {
                    flags |= OpenFlags::O_RDONLY;
                }
                handle.reopen(flags)
                    .map_err(|e| FsError::Io(std::io::Error::other(e.to_string())))
            }
            FsHandleInner::Path(path) => {
                let mut opts = OpenOptions::new();
                opts.read(true);
                if write {
                    opts.write(true);
                }
                opts.open(path).map_err(FsError::Io)
            }
        }
    }
}

impl std::fmt::Debug for FsHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsHandle")
            .field("rel_path", &self.rel_path)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stat — filesystem metadata
// ─────────────────────────────────────────────────────────────────────────────

/// Filesystem metadata returned by `stat()`.
#[derive(Debug, Clone)]
pub struct Stat {
    /// QID type (QTDIR for directories, QTFILE for files).
    pub qtype: u8,
    /// File mode/permissions.
    pub mode: u32,
    /// Access time (Unix timestamp).
    pub atime: u64,
    /// Modification time (Unix timestamp).
    pub mtime: u64,
    /// File size in bytes.
    pub length: u64,
    /// File name.
    pub name: String,
}

/// QID type constant for directories.
pub const QTDIR: u8 = 0x80;
/// QID type constant for regular files.
pub const QTFILE: u8 = 0x00;

impl Stat {
    /// Create a Stat from `std::fs::Metadata` and a name.
    pub fn from_metadata(meta: &fs::Metadata, name: String) -> Self {
        use std::time::UNIX_EPOCH;

        let qtype = if meta.is_dir() { QTDIR } else { QTFILE };
        let atime = meta
            .accessed()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        #[cfg(unix)]
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            meta.permissions().mode()
        };
        #[cfg(not(unix))]
        let mode = if meta.is_dir() { 0o755 } else { 0o644 };

        Self {
            qtype,
            mode,
            atime,
            mtime,
            length: meta.len(),
            name,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DirEntry — directory listing entry
// ─────────────────────────────────────────────────────────────────────────────

/// Directory entry returned by `readdir()`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DirEntry {
    /// Entry name.
    pub name: String,
    /// Whether this entry is a directory.
    pub is_dir: bool,
    /// File size in bytes.
    pub size: u64,
}
