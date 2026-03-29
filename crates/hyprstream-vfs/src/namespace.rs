//! Namespace — the client's mount table.
//!
//! Routes 9P operations by longest-prefix match to mount targets.
//! Forkable for per-task sandboxing. Static after initial population.

use std::sync::Arc;

use hyprstream_rpc::Subject;
use crate::mount::{DirEntry, Mount, MountError};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Namespace error.
#[derive(Debug)]
pub enum NamespaceError {
    Mount(MountError),
    ReservedPath(String),
}

impl std::fmt::Display for NamespaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mount(e) => write!(f, "{e}"),
            Self::ReservedPath(p) => write!(f, "reserved path: {p}"),
        }
    }
}

impl std::error::Error for NamespaceError {}

impl From<MountError> for NamespaceError {
    fn from(e: MountError) -> Self {
        Self::Mount(e)
    }
}

/// How a new mount interacts with existing mounts at the same prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindFlag {
    /// Replace any existing mount at this prefix.
    Replace,
    /// Prepend: new mount's entries appear first in union readdir.
    Before,
    /// Append: new mount's entries appear after existing.
    After,
}

/// What a mount point routes to — always an in-process Mount impl.
pub type MountTarget = Arc<dyn Mount>;

struct MountEntry {
    prefix: String,
    target: MountTarget,
}

// ─────────────────────────────────────────────────────────────────────────────
// Namespace
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum read size for cat/ctl loops (16 MB).
const MAX_READ_SIZE: usize = 16 * 1024 * 1024;

/// Client-side namespace. Maps path prefixes to mount targets.
///
/// Populated from service discovery at startup (`/srv/{name}`).
/// Forkable for per-task sandboxing via `fork()` + `unmount()`.
pub struct Namespace {
    mounts: Vec<MountEntry>,
}

impl Namespace {
    /// Create an empty namespace.
    pub fn new() -> Self {
        Self { mounts: Vec::new() }
    }

    /// Mount a target at a path prefix.
    ///
    /// Reserved prefixes (`/config`, `/private`, etc.) are allowed (all mounts are local now).
    pub fn mount(&mut self, prefix: &str, target: MountTarget) -> Result<(), NamespaceError> {
        let prefix = normalize_prefix(prefix);

        // Replace existing mount at this prefix (BindFlag::Replace semantics).
        self.mounts.retain(|m| m.prefix != prefix);
        self.mounts.push(MountEntry { prefix, target });
        // Sort by prefix length descending for longest-prefix match.
        self.mounts.sort_by(|a, b| b.prefix.len().cmp(&a.prefix.len()));
        Ok(())
    }

    /// Remove a mount point. In forked namespaces, this is irreversible.
    pub fn unmount(&mut self, prefix: &str) {
        let prefix = normalize_prefix(prefix);
        self.mounts.retain(|m| m.prefix != prefix);
    }

    /// Create a child namespace (Plan 9 `rfork(RFNAMEG)`).
    ///
    /// The child gets a snapshot of the parent's mounts.
    /// Modifications to the child do not affect the parent.
    pub fn fork(&self) -> Namespace {
        Namespace {
            mounts: self.mounts.iter().map(|m| MountEntry {
                prefix: m.prefix.clone(),
                target: Arc::clone(&m.target),
            }).collect(),
        }
    }

    /// List all mount prefixes (for debugging / `ls /`).
    pub fn mount_prefixes(&self) -> Vec<&str> {
        self.mounts.iter().map(|m| m.prefix.as_str()).collect()
    }

    // ── Convenience methods ─────────────────────────────────────────────────

    /// Read the entire contents of a file. Walk + open + read-all + clunk.
    ///
    /// Loops reads until an empty Vec is returned, up to 16MB cap.
    pub fn cat(&self, path: &str, caller: &Subject) -> Result<Vec<u8>, NamespaceError> {
        let (mount, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        let mut fid = mount.walk(&components, caller)?;
        mount.open(&mut fid, 0, caller)?; // OREAD
        let mut data = Vec::new();
        loop {
            let chunk = mount.read(&fid, data.len() as u64, 64 * 1024, caller)?;
            if chunk.is_empty() {
                break;
            }
            data.extend_from_slice(&chunk);
            if data.len() >= MAX_READ_SIZE {
                break;
            }
        }
        mount.clunk(fid, caller);
        Ok(data)
    }

    /// Write data to a file. Walk + open(write) + write + clunk.
    pub fn echo(&self, path: &str, data: &[u8], caller: &Subject) -> Result<(), NamespaceError> {
        let (mount, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        let mut fid = mount.walk(&components, caller)?;
        mount.open(&mut fid, 1, caller)?; // OWRITE
        mount.write(&fid, 0, data, caller)?;
        mount.clunk(fid, caller);
        Ok(())
    }

    /// Write to a ctl file and read the response. Walk + open(RDWR) + write + read + clunk.
    ///
    /// This is the Plan9 ctl pattern: write a command, read the response, all on
    /// the same fid so the response buffer is available after the write.
    /// Loops reads until an empty Vec is returned, up to 16MB cap.
    pub fn ctl(&self, path: &str, data: &[u8], caller: &Subject) -> Result<Vec<u8>, NamespaceError> {
        let (mount, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        let mut fid = mount.walk(&components, caller)?;
        mount.open(&mut fid, 2, caller)?; // ORDWR
        mount.write(&fid, 0, data, caller)?;
        let mut resp = Vec::new();
        loop {
            let chunk = mount.read(&fid, resp.len() as u64, 64 * 1024, caller)?;
            if chunk.is_empty() {
                break;
            }
            resp.extend_from_slice(&chunk);
            if resp.len() >= MAX_READ_SIZE {
                break;
            }
        }
        mount.clunk(fid, caller);
        Ok(resp)
    }

    /// List directory entries. Walk + open(read) + readdir + clunk.
    pub fn ls(&self, path: &str, caller: &Subject) -> Result<Vec<DirEntry>, NamespaceError> {
        // Special case: ls "/" shows all mount prefixes as directories.
        if path == "/" || path.is_empty() {
            return Ok(self.root_dir_entries());
        }

        let (mount, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        let mut fid = mount.walk(&components, caller)?;
        mount.open(&mut fid, 0, caller)?;
        let entries = mount.readdir(&fid, caller)?;
        mount.clunk(fid, caller);
        Ok(entries)
    }

    // ── Internal ────────────────────────────────────────────────────────────

    /// Resolve a path to (MountTarget, remainder after prefix).
    fn resolve(&self, path: &str) -> Result<(&MountTarget, String), NamespaceError> {
        let path = normalize_path(path);

        for entry in &self.mounts {
            if path == entry.prefix || path.starts_with(&format!("{}/", entry.prefix)) {
                let remainder = path[entry.prefix.len()..].to_owned();
                return Ok((&entry.target, remainder));
            }
        }

        Err(NamespaceError::Mount(MountError::NotFound(path)))
    }

    /// Generate root directory entries from mount prefixes.
    fn root_dir_entries(&self) -> Vec<DirEntry> {
        let mut seen = std::collections::HashSet::new();
        let mut entries = Vec::new();
        for m in &self.mounts {
            // Extract the first path component after '/'.
            let first = m.prefix.trim_start_matches('/')
                .split('/')
                .next()
                .unwrap_or("");
            if !first.is_empty() && seen.insert(first.to_owned()) {
                entries.push(DirEntry {
                    name: first.to_owned(),
                    is_dir: true,
                    size: 0,
                    stat: None,
                });
            }
        }
        entries
    }
}

impl Default for Namespace {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn normalize_prefix(prefix: &str) -> String {
    let p = if prefix.starts_with('/') { prefix.to_owned() } else { format!("/{prefix}") };
    // Remove trailing slash.
    if p.len() > 1 && p.ends_with('/') { p[..p.len() - 1].to_owned() } else { p }
}

/// Normalize a path: ensure leading `/`, resolve `.` and `..` components, remove trailing `/`.
fn normalize_path(path: &str) -> String {
    let path = if path.starts_with('/') { path.to_owned() } else { format!("/{path}") };
    let mut parts: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => { parts.pop(); }
            s => parts.push(s),
        }
    }
    if parts.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", parts.join("/"))
    }
}

fn split_path(path: &str) -> Vec<&str> {
    path.split('/')
        .filter(|s| !s.is_empty() && *s != "." && *s != "..")
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mount::{Fid, Stat, MountError};
    use std::collections::HashMap;

    /// Simple in-memory Mount for testing.
    struct MemMount {
        files: HashMap<String, Vec<u8>>,
    }

    impl MemMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self {
                files: files.into_iter().map(|(k, v)| (k.to_owned(), v.to_vec())).collect(),
            }
        }
    }

    struct MemFid {
        path: String,
        is_open: bool,
    }

    impl Mount for MemMount {
        fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let path = components.join("/");
            Ok(Fid::new(MemFid { path, is_open: false }))
        }

        fn open(&self, fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            let inner = fid.downcast_mut::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            inner.is_open = true;
            Ok(())
        }

        fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() {
                        Ok(vec![])
                    } else {
                        Ok(data[start..].to_vec())
                    }
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        fn write(&self, _fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(data.len() as u32)
        }

        fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let prefix = if inner.path.is_empty() { String::new() } else { format!("{}/", inner.path) };
            let mut entries = Vec::new();
            for key in self.files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry { name: rest.to_owned(), is_dir: false, size: 0, stat: None });
                    }
                }
            }
            Ok(entries)
        }

        fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat { qtype: 0, size: 0, name: inner.path.clone(), mtime: 0 })
        }

        fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn test_subject() -> Subject { Subject::new("test") }

    #[test]
    fn mount_and_cat() {
        let mut ns = Namespace::new();
        let mount: MountTarget = Arc::new(MemMount::new(vec![("temperature", b"0.7\n")]));
        ns.mount("/config", mount).unwrap();

        let data = ns.cat("/config/temperature", &test_subject()).unwrap();
        assert_eq!(data, b"0.7\n");
    }

    #[test]
    fn mount_and_ls_root() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();
        ns.mount("/srv/mcp", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();
        ns.mount("/config", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();

        let entries = ns.ls("/", &test_subject()).unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"srv"));
        assert!(names.contains(&"config"));
    }

    #[test]
    fn longest_prefix_match() {
        let mut ns = Namespace::new();
        let srv: MountTarget = Arc::new(MemMount::new(vec![("status", b"ok")]));
        let nested: MountTarget = Arc::new(MemMount::new(vec![("status", b"loaded")]));
        ns.mount("/srv", srv).unwrap();
        ns.mount("/srv/model", nested).unwrap();

        let data = ns.cat("/srv/model/status", &test_subject()).unwrap();
        assert_eq!(data, b"loaded");
    }

    #[test]
    fn reserved_path_allows_mount() {
        let mut ns = Namespace::new();
        let mount: MountTarget = Arc::new(MemMount::new(vec![]));
        assert!(ns.mount("/config", mount).is_ok());
    }

    #[test]
    fn fork_and_unmount() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![("status", b"loaded")])) as MountTarget).unwrap();
        ns.mount("/srv/worker", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();

        let mut child = ns.fork();
        child.unmount("/srv/worker");

        assert_eq!(ns.mount_prefixes().len(), 2);
        assert_eq!(child.mount_prefixes().len(), 1);
        assert_eq!(child.mount_prefixes()[0], "/srv/model");

        let data = child.cat("/srv/model/status", &test_subject()).unwrap();
        assert_eq!(data, b"loaded");
    }

    #[test]
    fn not_found() {
        let ns = Namespace::new();
        assert!(matches!(ns.cat("/nonexistent/file", &test_subject()), Err(NamespaceError::Mount(MountError::NotFound(_)))));
    }

    #[test]
    fn echo_writes() {
        let mut ns = Namespace::new();
        let mount: MountTarget = Arc::new(MemMount::new(vec![("ctl", b"")]));
        ns.mount("/cmd", mount).unwrap();

        assert!(ns.echo("/cmd/ctl", b"load qwen3:main", &test_subject()).is_ok());
    }

    // ── New tests ──────────────────────────────────────────────────────────

    /// Read loop must handle files larger than 64KB.
    #[test]
    fn read_loop_large_file() {
        // Create a file > 64KB.
        let big_data = vec![0x42u8; 100 * 1024]; // 100 KB
        let mut ns = Namespace::new();

        struct ChunkedMount { data: Vec<u8> }
        struct ChunkedFid { offset_limit: bool }

        impl Mount for ChunkedMount {
            fn walk(&self, _c: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
                Ok(Fid::new(ChunkedFid { offset_limit: false }))
            }
            fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> { Ok(()) }
            fn read(&self, _fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
                let start = offset as usize;
                if start >= self.data.len() { return Ok(vec![]); }
                let end = std::cmp::min(start + count as usize, self.data.len());
                Ok(self.data[start..end].to_vec())
            }
            fn write(&self, _fid: &Fid, _o: u64, d: &[u8], _c: &Subject) -> Result<u32, MountError> { Ok(d.len() as u32) }
            fn readdir(&self, _fid: &Fid, _c: &Subject) -> Result<Vec<DirEntry>, MountError> { Ok(vec![]) }
            fn stat(&self, _fid: &Fid, _c: &Subject) -> Result<crate::mount::Stat, MountError> {
                Ok(crate::mount::Stat { qtype: 0, size: 0, name: String::new(), mtime: 0 })
            }
            fn clunk(&self, _fid: Fid, _c: &Subject) {}
        }

        ns.mount("/big", Arc::new(ChunkedMount { data: big_data.clone() }) as MountTarget).unwrap();
        let result = ns.cat("/big/file", &test_subject()).unwrap();
        assert_eq!(result.len(), 100 * 1024);
        assert_eq!(result, big_data);
    }

    /// OTRUNC mode pass-through.
    #[test]
    fn otrunc_mode_passthrough() {
        use crate::mount::{OWRITE, OTRUNC};
        // Verify the constants combine correctly.
        let mode = OWRITE | OTRUNC;
        assert_eq!(mode & 0x03, OWRITE);
        assert_ne!(mode & OTRUNC, 0);
    }

    /// Path normalization with `..`.
    #[test]
    fn path_normalization_with_dotdot() {
        assert_eq!(normalize_path("/config/../srv/model"), "/srv/model");
        assert_eq!(normalize_path("/config/./status"), "/config/status");
        assert_eq!(normalize_path("//config///status"), "/config/status");
        assert_eq!(normalize_path("/a/b/c/../../d"), "/a/d");
        assert_eq!(normalize_path("/.."), "/");
    }

    /// Normalize path resolves before prefix matching in resolve().
    #[test]
    fn resolve_with_dotdot_path() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![("status", b"ok")])) as MountTarget).unwrap();

        // /config/../srv/model/status should normalize to /srv/model/status
        let data = ns.cat("/config/../srv/model/status", &test_subject()).unwrap();
        assert_eq!(data, b"ok");
    }
}
