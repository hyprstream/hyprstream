//! Namespace — the client's mount table.
//!
//! Routes 9P operations by longest-prefix match to mount targets.
//! Forkable for per-task sandboxing. Static after initial population.

use std::sync::Arc;

use hyprstream_rpc::Subject;
use crate::local_mount::{DirEntry, LocalMount};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// VFS error.
#[derive(Debug)]
pub enum VfsError {
    NotFound(String),
    PermissionDenied(String),
    NotDir(String),
    Io(String),
    ReservedPath(String),
    TransportUnavailable(String),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(p) => write!(f, "not found: {p}"),
            Self::PermissionDenied(p) => write!(f, "permission denied: {p}"),
            Self::NotDir(p) => write!(f, "not a directory: {p}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::ReservedPath(p) => write!(f, "reserved path: {p}"),
            Self::TransportUnavailable(t) => write!(f, "transport unavailable: {t}"),
        }
    }
}

impl std::error::Error for VfsError {}

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

/// Transport abstraction for remote 9P services.
///
/// Implementations handle the actual message serialization and delivery.
/// - Native: ZMQ REQ/REP with signed envelopes
/// - WASM: capnp IPC over Wanix named pipes
pub trait ServiceTransport: Send + Sync {
    /// Send a 9P request and receive a response (serialized bytes).
    fn round_trip(&self, request: &[u8]) -> Result<Vec<u8>, String>;
}

/// What a mount point routes to.
pub enum MountTarget {
    /// Remote service via a transport (ZMQ, IPC, etc.).
    Service(Arc<dyn ServiceTransport>),
    /// In-process mount (no network).
    Local(Arc<dyn LocalMount>),
}

impl Clone for MountTarget {
    fn clone(&self) -> Self {
        match self {
            Self::Service(t) => Self::Service(Arc::clone(t)),
            Self::Local(m) => Self::Local(Arc::clone(m)),
        }
    }
}

struct MountEntry {
    prefix: String,
    target: MountTarget,
}

// ─────────────────────────────────────────────────────────────────────────────
// Namespace
// ─────────────────────────────────────────────────────────────────────────────

/// Reserved prefixes that can only be mounted with LocalMount.
const LOCAL_ONLY: &[&str] = &["/config", "/private", "/env", "/dev", "/proc"];

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
    /// Reserved prefixes (`/config`, `/private`, etc.) reject non-local targets.
    pub fn mount(&mut self, prefix: &str, target: MountTarget) -> Result<(), VfsError> {
        let prefix = normalize_prefix(prefix);

        // Enforce reserved paths.
        if LOCAL_ONLY.iter().any(|p| prefix.starts_with(p))
            && !matches!(target, MountTarget::Local(_))
        {
            return Err(VfsError::ReservedPath(prefix));
        }

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
                target: m.target.clone(),
            }).collect(),
        }
    }

    /// List all mount prefixes (for debugging / `ls /`).
    pub fn mount_prefixes(&self) -> Vec<&str> {
        self.mounts.iter().map(|m| m.prefix.as_str()).collect()
    }

    // ── Convenience methods ─────────────────────────────────────────────────

    /// Read the entire contents of a file. Walk + open + read-all + clunk.
    pub fn cat(&self, path: &str, caller: &Subject) -> Result<Vec<u8>, VfsError> {
        let (target, remainder) = self.resolve(path)?;
        match target {
            MountTarget::Local(mount) => {
                let components: Vec<&str> = split_path(&remainder);
                let mut fid = mount.walk(&components, caller).map_err(VfsError::Io)?;
                mount.open(&mut fid, 0, caller).map_err(VfsError::Io)?; // OREAD
                let data = mount.read(&fid, 0, 64 * 1024, caller).map_err(VfsError::Io)?;
                mount.clunk(fid, caller);
                Ok(data)
            }
            MountTarget::Service(_transport) => {
                Err(VfsError::TransportUnavailable(
                    "service transport not yet implemented".to_owned(),
                ))
            }
        }
    }

    /// Write data to a file. Walk + open(write) + write + clunk.
    pub fn echo(&self, path: &str, data: &[u8], caller: &Subject) -> Result<(), VfsError> {
        let (target, remainder) = self.resolve(path)?;
        match target {
            MountTarget::Local(mount) => {
                let components: Vec<&str> = split_path(&remainder);
                let mut fid = mount.walk(&components, caller).map_err(VfsError::Io)?;
                mount.open(&mut fid, 1, caller).map_err(VfsError::Io)?; // OWRITE
                mount.write(&fid, 0, data, caller).map_err(VfsError::Io)?;
                mount.clunk(fid, caller);
                Ok(())
            }
            MountTarget::Service(_) => {
                Err(VfsError::TransportUnavailable(
                    "service transport not yet implemented".to_owned(),
                ))
            }
        }
    }

    /// Write to a ctl file and read the response. Walk + open(RDWR) + write + read + clunk.
    ///
    /// This is the Plan9 ctl pattern: write a command, read the response, all on
    /// the same fid so the response buffer is available after the write.
    pub fn ctl(&self, path: &str, data: &[u8], caller: &Subject) -> Result<Vec<u8>, VfsError> {
        let (target, remainder) = self.resolve(path)?;
        match target {
            MountTarget::Local(mount) => {
                let components: Vec<&str> = split_path(&remainder);
                let mut fid = mount.walk(&components, caller).map_err(VfsError::Io)?;
                mount.open(&mut fid, 2, caller).map_err(VfsError::Io)?; // ORDWR
                mount.write(&fid, 0, data, caller).map_err(VfsError::Io)?;
                let resp = mount.read(&fid, 0, 64 * 1024, caller).map_err(VfsError::Io)?;
                mount.clunk(fid, caller);
                Ok(resp)
            }
            MountTarget::Service(_) => {
                Err(VfsError::TransportUnavailable(
                    "service transport not yet implemented".to_owned(),
                ))
            }
        }
    }

    /// List directory entries. Walk + open(read) + readdir + clunk.
    pub fn ls(&self, path: &str, caller: &Subject) -> Result<Vec<DirEntry>, VfsError> {
        // Special case: ls "/" shows all mount prefixes as directories.
        if path == "/" || path.is_empty() {
            return Ok(self.root_dir_entries());
        }

        let (target, remainder) = self.resolve(path)?;
        match target {
            MountTarget::Local(mount) => {
                let components: Vec<&str> = split_path(&remainder);
                let mut fid = mount.walk(&components, caller).map_err(VfsError::Io)?;
                mount.open(&mut fid, 0, caller).map_err(VfsError::Io)?;
                let entries = mount.readdir(&fid, caller).map_err(VfsError::Io)?;
                mount.clunk(fid, caller);
                Ok(entries)
            }
            MountTarget::Service(_) => {
                Err(VfsError::TransportUnavailable(
                    "service transport not yet implemented".to_owned(),
                ))
            }
        }
    }

    // ── Internal ────────────────────────────────────────────────────────────

    /// Resolve a path to (MountTarget, remainder after prefix).
    fn resolve(&self, path: &str) -> Result<(&MountTarget, String), VfsError> {
        let path = if path.starts_with('/') { path.to_owned() } else { format!("/{path}") };

        for entry in &self.mounts {
            if path == entry.prefix || path.starts_with(&format!("{}/", entry.prefix)) {
                let remainder = path[entry.prefix.len()..].to_owned();
                return Ok((&entry.target, remainder));
            }
        }

        Err(VfsError::NotFound(path))
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

fn split_path(path: &str) -> Vec<&str> {
    path.split('/').filter(|s| !s.is_empty()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::local_mount::{LocalFid, Stat};
    use std::collections::HashMap;

    /// Simple in-memory LocalMount for testing.
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

    impl LocalMount for MemMount {
        fn walk(&self, components: &[&str], _caller: &Subject) -> Result<LocalFid, String> {
            let path = components.join("/");
            Ok(LocalFid::new(MemFid { path, is_open: false }))
        }

        fn open(&self, fid: &mut LocalFid, _mode: u8, _caller: &Subject) -> Result<(), String> {
            let inner = fid.downcast_mut::<MemFid>().ok_or("bad fid")?;
            inner.is_open = true;
            Ok(())
        }

        fn read(&self, fid: &LocalFid, _offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, String> {
            let inner = fid.downcast_ref::<MemFid>().ok_or("bad fid")?;
            self.files.get(&inner.path).cloned().ok_or_else(|| format!("not found: {}", inner.path))
        }

        fn write(&self, _fid: &LocalFid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, String> {
            Ok(_data.len() as u32)
        }

        fn readdir(&self, fid: &LocalFid, _caller: &Subject) -> Result<Vec<DirEntry>, String> {
            let inner = fid.downcast_ref::<MemFid>().ok_or("bad fid")?;
            let prefix = if inner.path.is_empty() { String::new() } else { format!("{}/", inner.path) };
            let mut entries = Vec::new();
            for key in self.files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry { name: rest.to_owned(), is_dir: false, size: 0 });
                    }
                }
            }
            Ok(entries)
        }

        fn stat(&self, fid: &LocalFid, _caller: &Subject) -> Result<Stat, String> {
            let inner = fid.downcast_ref::<MemFid>().ok_or("bad fid")?;
            Ok(Stat { qtype: 0, size: 0, name: inner.path.clone(), mtime: 0 })
        }

        fn clunk(&self, _fid: LocalFid, _caller: &Subject) {}
    }

    fn test_subject() -> Subject { Subject::new("test") }

    #[test]
    fn mount_and_cat() {
        let mut ns = Namespace::new();
        let mount = Arc::new(MemMount::new(vec![("temperature", b"0.7\n")]));
        ns.mount("/config", MountTarget::Local(mount)).unwrap();

        let data = ns.cat("/config/temperature", &test_subject()).unwrap();
        assert_eq!(data, b"0.7\n");
    }

    #[test]
    fn mount_and_ls_root() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", MountTarget::Local(Arc::new(MemMount::new(vec![])))).unwrap();
        ns.mount("/srv/mcp", MountTarget::Local(Arc::new(MemMount::new(vec![])))).unwrap();
        ns.mount("/config", MountTarget::Local(Arc::new(MemMount::new(vec![])))).unwrap();

        let entries = ns.ls("/", &test_subject()).unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"srv"));
        assert!(names.contains(&"config"));
    }

    #[test]
    fn longest_prefix_match() {
        let mut ns = Namespace::new();
        let srv = Arc::new(MemMount::new(vec![("status", b"ok")]));
        let nested = Arc::new(MemMount::new(vec![("status", b"loaded")]));
        ns.mount("/srv", MountTarget::Local(srv)).unwrap();
        ns.mount("/srv/model", MountTarget::Local(nested)).unwrap();

        let data = ns.cat("/srv/model/status", &test_subject()).unwrap();
        assert_eq!(data, b"loaded");
    }

    #[test]
    fn reserved_path_rejects_remote() {
        let mut ns = Namespace::new();
        struct FakeTransport;
        impl ServiceTransport for FakeTransport {
            fn round_trip(&self, _: &[u8]) -> Result<Vec<u8>, String> { Ok(vec![]) }
        }
        let result = ns.mount("/config", MountTarget::Service(Arc::new(FakeTransport)));
        assert!(matches!(result, Err(VfsError::ReservedPath(_))));
    }

    #[test]
    fn reserved_path_allows_local() {
        let mut ns = Namespace::new();
        let mount = Arc::new(MemMount::new(vec![]));
        assert!(ns.mount("/config", MountTarget::Local(mount)).is_ok());
    }

    #[test]
    fn fork_and_unmount() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", MountTarget::Local(Arc::new(MemMount::new(vec![("status", b"loaded")])))).unwrap();
        ns.mount("/srv/worker", MountTarget::Local(Arc::new(MemMount::new(vec![])))).unwrap();

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
        assert!(matches!(ns.cat("/nonexistent/file", &test_subject()), Err(VfsError::NotFound(_))));
    }

    #[test]
    fn echo_writes() {
        let mut ns = Namespace::new();
        let mount = Arc::new(MemMount::new(vec![("ctl", b"")]));
        ns.mount("/cmd", MountTarget::Local(mount)).unwrap();

        assert!(ns.echo("/cmd/ctl", b"load qwen3:main", &test_subject()).is_ok());
    }
}
