//! Path-contained filesystem operations with 9P-aligned naming.
//!
//! Provides a [`ContainedFs`] trait that abstracts path-safe filesystem operations.
//! All operations are contained within a root directory — paths cannot escape via
//! symlinks, `..` traversal, or other techniques.
//!
//! - **Linux**: Uses `pathrs` with `openat2(RESOLVE_IN_ROOT)` for kernel-enforced
//!   containment. This is atomic and TOCTOU-proof.
//! - **Unix non-Linux** (macOS, FreeBSD): Uses `safe-path` with per-component symlink
//!   resolution constrained within root. Better than canonicalize but not kernel-enforced.
//! - **Non-Unix** (Windows): Uses `canonicalize` + prefix check. Best-effort
//!   (TOCTOU-vulnerable), acceptable for development.
//!
//! # Path utilities
//!
//! - [`contained_join`]: Clamps `..` traversal (for path construction).
//! - [`validate_relative_path`]: Rejects `..` traversal (for I/O validation).
//! - [`validate_ref_name`]: Git ref name validation (matches `git check-ref-format`).

pub mod canonical_root;
pub mod error;
pub mod path;
#[cfg(target_os = "linux")]
pub mod pathrs_root;
#[cfg(all(unix, not(target_os = "linux")))]
pub mod scoped_root;
pub mod types;

// Re-exports
pub use canonical_root::CanonicalRoot;
pub use error::FsError;
pub use path::{contained_join, validate_ref_name, validate_relative_path};
#[cfg(target_os = "linux")]
pub use pathrs_root::PathrsRoot;
#[cfg(all(unix, not(target_os = "linux")))]
pub use scoped_root::ScopedRoot;
pub use types::{DirEntry, FsHandle, OpenMode, Stat, QTDIR, QTFILE};

use std::path::Path;
use std::sync::Arc;

/// Trait abstracting path-contained filesystem operations.
///
/// All paths passed to methods are relative to the contained root
/// (forward-slash-separated `&str`, 9P convention).
/// Implementations ensure no path can escape the root directory.
pub trait ContainedFs: Send + Sync {
    /// Resolve a relative path and return an FsHandle for 9P walk state.
    fn walk(&self, path: &str) -> Result<FsHandle, FsError>;

    /// Open an existing file relative to the root.
    fn open(&self, path: &str, mode: OpenMode) -> Result<std::fs::File, FsError>;

    /// Create a file relative to the root (implies open).
    fn create(&self, path: &str, mode: OpenMode) -> Result<std::fs::File, FsError>;

    /// Create a single directory.
    fn mkdir(&self, path: &str) -> Result<(), FsError>;

    /// Create directories recursively.
    fn mkdir_all(&self, path: &str) -> Result<(), FsError>;

    /// Get file/directory metadata.
    fn stat(&self, path: &str) -> Result<Stat, FsError>;

    /// Rename a file or directory within the root.
    fn rename(&self, src: &str, dst: &str) -> Result<(), FsError>;

    /// Remove a file or empty directory.
    fn remove(&self, path: &str) -> Result<(), FsError>;

    /// List directory entries.
    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, FsError>;

    /// Copy a file (uses reflink/COW when available).
    fn copy(&self, src: &str, dst: &str) -> Result<(), FsError>;
}

/// Open best available ContainedFs for root path.
///
/// - **Linux**: `PathrsRoot` — kernel-enforced via `openat2(RESOLVE_IN_ROOT)`.
/// - **Unix non-Linux**: `ScopedRoot` — `safe-path` per-component symlink resolution.
/// - **Non-Unix**: `CanonicalRoot` — `canonicalize` + prefix check (best-effort).
pub fn open(root: &Path) -> Result<Arc<dyn ContainedFs>, FsError> {
    #[cfg(target_os = "linux")]
    {
        PathrsRoot::new(root)
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        ScopedRoot::new(root)
    }
    #[cfg(not(unix))]
    {
        CanonicalRoot::new(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_open_creates_contained_fs() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        // Should be able to mkdir and stat
        fs.mkdir_all("test/subdir")
            .unwrap_or_else(|e| panic!("mkdir_all: {e}"));
        let stat = fs.stat("test/subdir").unwrap_or_else(|e| panic!("stat: {e}"));
        assert_eq!(stat.qtype, QTDIR);
    }

    #[test]
    fn test_create_and_read() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        // Create a file
        {
            use std::io::Write;
            let mut file = fs
                .create("test.txt", OpenMode::OWRITE)
                .unwrap_or_else(|e| panic!("create: {e}"));
            file.write_all(b"hello")
                .unwrap_or_else(|e| panic!("write: {e}"));
        }

        // Read it back
        {
            use std::io::Read;
            let mut file = fs
                .open("test.txt", OpenMode::OREAD)
                .unwrap_or_else(|e| panic!("open: {e}"));
            let mut buf = String::new();
            file.read_to_string(&mut buf)
                .unwrap_or_else(|e| panic!("read: {e}"));
            assert_eq!(buf, "hello");
        }
    }

    #[test]
    fn test_walk_and_metadata() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        fs.mkdir("testdir")
            .unwrap_or_else(|e| panic!("mkdir: {e}"));
        let handle = fs.walk("testdir").unwrap_or_else(|e| panic!("walk: {e}"));
        assert_eq!(handle.rel_path(), "testdir");

        let meta = handle.metadata().unwrap_or_else(|e| panic!("metadata: {e}"));
        assert!(meta.is_dir());
    }

    #[test]
    fn test_readdir() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        fs.mkdir("dir").unwrap_or_else(|e| panic!("mkdir: {e}"));
        {
            use std::io::Write;
            let mut f = fs
                .create("dir/a.txt", OpenMode::OWRITE)
                .unwrap_or_else(|e| panic!("create: {e}"));
            f.write_all(b"a").unwrap_or_else(|e| panic!("write: {e}"));
        }
        {
            use std::io::Write;
            let mut f = fs
                .create("dir/b.txt", OpenMode::OWRITE)
                .unwrap_or_else(|e| panic!("create: {e}"));
            f.write_all(b"b").unwrap_or_else(|e| panic!("write: {e}"));
        }

        let entries = fs.readdir("dir").unwrap_or_else(|e| panic!("readdir: {e}"));
        let mut names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["a.txt", "b.txt"]);
    }

    #[test]
    fn test_rename() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        {
            use std::io::Write;
            let mut f = fs
                .create("old.txt", OpenMode::OWRITE)
                .unwrap_or_else(|e| panic!("create: {e}"));
            f.write_all(b"data").unwrap_or_else(|e| panic!("write: {e}"));
        }

        fs.rename("old.txt", "new.txt")
            .unwrap_or_else(|e| panic!("rename: {e}"));

        assert!(fs.stat("old.txt").is_err());
        assert!(fs.stat("new.txt").is_ok());
    }

    #[test]
    fn test_remove() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        {
            use std::io::Write;
            let mut f = fs
                .create("to_remove.txt", OpenMode::OWRITE)
                .unwrap_or_else(|e| panic!("create: {e}"));
            f.write_all(b"data").unwrap_or_else(|e| panic!("write: {e}"));
        }

        fs.remove("to_remove.txt")
            .unwrap_or_else(|e| panic!("remove: {e}"));
        assert!(fs.stat("to_remove.txt").is_err());
    }

    #[test]
    fn test_copy() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        {
            use std::io::Write;
            let mut f = fs
                .create("src.txt", OpenMode::OWRITE)
                .unwrap_or_else(|e| panic!("create: {e}"));
            f.write_all(b"copy me").unwrap_or_else(|e| panic!("write: {e}"));
        }

        fs.copy("src.txt", "dst.txt")
            .unwrap_or_else(|e| panic!("copy: {e}"));

        {
            use std::io::Read;
            let mut f = fs
                .open("dst.txt", OpenMode::OREAD)
                .unwrap_or_else(|e| panic!("open: {e}"));
            let mut buf = String::new();
            f.read_to_string(&mut buf)
                .unwrap_or_else(|e| panic!("read: {e}"));
            assert_eq!(buf, "copy me");
        }
    }

    #[test]
    fn test_child_rel_path() {
        let temp = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let fs = open(temp.path()).unwrap_or_else(|e| panic!("open: {e}"));

        fs.mkdir("parent")
            .unwrap_or_else(|e| panic!("mkdir: {e}"));
        let handle = fs.walk("parent").unwrap_or_else(|e| panic!("walk: {e}"));
        assert_eq!(handle.child_rel_path("child"), "parent/child");

        // Root-level handle
        let root_handle = fs.walk(".").unwrap_or_else(|e| panic!("walk root: {e}"));
        assert_eq!(root_handle.child_rel_path("child"), "child");
    }
}
