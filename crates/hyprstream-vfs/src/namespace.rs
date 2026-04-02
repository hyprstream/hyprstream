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
    targets: Vec<MountTarget>,
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

    /// Mount a target at a path prefix (replaces any existing mount).
    ///
    /// This is shorthand for `bind_mount(prefix, target, BindFlag::Replace)`.
    pub fn mount(&mut self, prefix: &str, target: MountTarget) -> Result<(), NamespaceError> {
        self.bind_mount(prefix, target, BindFlag::Replace)
    }

    /// Mount a target at a path prefix with union semantics.
    ///
    /// - `Replace`: removes any existing mount at this prefix.
    /// - `Before`: prepends the target — its entries appear first in readdir,
    ///   and walk tries it before existing mounts at the same prefix.
    /// - `After`: appends the target — existing mounts are tried first.
    pub fn bind_mount(&mut self, prefix: &str, target: MountTarget, flag: BindFlag) -> Result<(), NamespaceError> {
        let prefix = normalize_prefix(prefix);

        match flag {
            BindFlag::Replace => {
                self.mounts.retain(|m| m.prefix != prefix);
                self.mounts.push(MountEntry { prefix, targets: vec![target] });
            }
            BindFlag::Before => {
                if let Some(entry) = self.mounts.iter_mut().find(|m| m.prefix == prefix) {
                    entry.targets.insert(0, target);
                } else {
                    self.mounts.push(MountEntry { prefix, targets: vec![target] });
                }
            }
            BindFlag::After => {
                if let Some(entry) = self.mounts.iter_mut().find(|m| m.prefix == prefix) {
                    entry.targets.push(target);
                } else {
                    self.mounts.push(MountEntry { prefix, targets: vec![target] });
                }
            }
        }
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
                targets: m.targets.iter().map(Arc::clone).collect(),
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
    /// With union mounts, tries each target in bind order until one succeeds.
    pub async fn cat(&self, path: &str, caller: &Subject) -> Result<Vec<u8>, NamespaceError> {
        let (targets, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        let mut last_err = None;
        for mount in targets {
            let result: Result<Vec<u8>, MountError> = async {
                let mut fid = mount.walk(&components, caller).await?;
                if let Err(e) = mount.open(&mut fid, 0, caller).await {
                    mount.clunk(fid, caller).await;
                    return Err(e);
                }
                let mut data = Vec::new();
                loop {
                    match mount.read(&fid, data.len() as u64, 64 * 1024, caller).await {
                        Ok(chunk) if chunk.is_empty() => break,
                        Ok(chunk) => {
                            data.extend_from_slice(&chunk);
                            if data.len() >= MAX_READ_SIZE { break; }
                        }
                        Err(e) => { mount.clunk(fid, caller).await; return Err(e); }
                    }
                }
                mount.clunk(fid, caller).await;
                Ok(data)
            }.await;
            match result {
                Ok(data) => return Ok(data),
                Err(e) => { last_err = Some(e); }
            }
        }
        Err(NamespaceError::Mount(last_err.unwrap_or_else(|| MountError::NotFound(path.to_owned()))))
    }

    /// Write data to a file. Walk + open(write) + write + clunk.
    /// With union mounts, tries each target in bind order until one succeeds.
    pub async fn echo(&self, path: &str, data: &[u8], caller: &Subject) -> Result<(), NamespaceError> {
        let (targets, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        let mut last_err = None;
        for mount in targets {
            let result: Result<(), MountError> = async {
                let mut fid = mount.walk(&components, caller).await?;
                if let Err(e) = mount.open(&mut fid, 1, caller).await {
                    mount.clunk(fid, caller).await;
                    return Err(e);
                }
                mount.write(&fid, 0, data, caller).await?;
                mount.clunk(fid, caller).await;
                Ok(())
            }.await;
            match result {
                Ok(()) => return Ok(()),
                Err(e) => { last_err = Some(e); }
            }
        }
        Err(NamespaceError::Mount(last_err.unwrap_or_else(|| MountError::NotFound(path.to_owned()))))
    }

    /// Write to a ctl file and read the response. Walk + open(RDWR) + write + read + clunk.
    ///
    /// This is the Plan9 ctl pattern: write a command, read the response, all on
    /// the same fid so the response buffer is available after the write.
    /// Loops reads until an empty Vec is returned, up to 16MB cap.
    /// With union mounts, tries each target in bind order until one succeeds.
    pub async fn ctl(&self, path: &str, data: &[u8], caller: &Subject) -> Result<Vec<u8>, NamespaceError> {
        let (targets, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        let mut last_err = None;
        for mount in targets {
            let result: Result<Vec<u8>, MountError> = async {
                let mut fid = mount.walk(&components, caller).await?;
                if let Err(e) = mount.open(&mut fid, 2, caller).await {
                    mount.clunk(fid, caller).await;
                    return Err(e);
                }
                if let Err(e) = mount.write(&fid, 0, data, caller).await {
                    mount.clunk(fid, caller).await;
                    return Err(e);
                }
                let mut resp = Vec::new();
                loop {
                    match mount.read(&fid, resp.len() as u64, 64 * 1024, caller).await {
                        Ok(chunk) if chunk.is_empty() => break,
                        Ok(chunk) => {
                            resp.extend_from_slice(&chunk);
                            if resp.len() >= MAX_READ_SIZE { break; }
                        }
                        Err(e) => { mount.clunk(fid, caller).await; return Err(e); }
                    }
                }
                mount.clunk(fid, caller).await;
                Ok(resp)
            }.await;
            match result {
                Ok(resp) => return Ok(resp),
                Err(e) => { last_err = Some(e); }
            }
        }
        Err(NamespaceError::Mount(last_err.unwrap_or_else(|| MountError::NotFound(path.to_owned()))))
    }

    /// List directory entries. Walk + open(read) + readdir + clunk.
    ///
    /// With union mounts, merges readdir results from all targets at the prefix,
    /// deduplicating by name (first occurrence wins, preserving bind order).
    pub async fn ls(&self, path: &str, caller: &Subject) -> Result<Vec<DirEntry>, NamespaceError> {
        // Special case: ls "/" shows all mount prefixes as directories.
        if path == "/" || path.is_empty() {
            return Ok(self.root_dir_entries());
        }

        match self.resolve(path) {
            Ok((targets, remainder)) => {
                let components: Vec<&str> = split_path(&remainder);
                let mut seen = std::collections::HashSet::new();
                let mut merged = Vec::new();
                let mut any_ok = false;
                for mount in targets {
                    if let Ok(mut fid) = mount.walk(&components, caller).await {
                        if mount.open(&mut fid, 0, caller).await.is_ok() {
                            if let Ok(entries) = mount.readdir(&fid, caller).await {
                                any_ok = true;
                                for entry in entries {
                                    if seen.insert(entry.name.clone()) {
                                        merged.push(entry);
                                    }
                                }
                            }
                        }
                        mount.clunk(fid, caller).await;
                    }
                }
                if any_ok {
                    Ok(merged)
                } else {
                    Err(NamespaceError::Mount(MountError::NotFound(path.to_owned())))
                }
            }
            Err(_) => {
                // Path doesn't match any mount prefix directly. Check if it's an
                // intermediate directory — i.e., some mount's prefix starts with
                // "{path}/". Synthesize directory entries from sub-mount components.
                let normalized = normalize_path(path);
                let prefix_with_slash = format!("{}/", normalized);
                let mut seen = std::collections::HashSet::new();
                let mut entries = Vec::new();
                for m in &self.mounts {
                    if let Some(rest) = m.prefix.strip_prefix(&prefix_with_slash[..]) {
                        let next = rest.split('/').next().unwrap_or("");
                        if !next.is_empty() && seen.insert(next.to_owned()) {
                            entries.push(DirEntry {
                                name: next.to_owned(),
                                is_dir: true,
                                size: 0,
                                stat: None,
                            });
                        }
                    }
                }
                if entries.is_empty() {
                    Err(NamespaceError::Mount(MountError::NotFound(path.to_owned())))
                } else {
                    Ok(entries)
                }
            }
        }
    }

    // ── Internal ────────────────────────────────────────────────────────────

    /// Resolve a path to (list of MountTargets, remainder after prefix).
    ///
    /// Returns targets in bind order (Before targets first, After targets last).
    fn resolve(&self, path: &str) -> Result<(&[MountTarget], String), NamespaceError> {
        let path = normalize_path(path);

        for entry in &self.mounts {
            if path == entry.prefix || path.starts_with(&format!("{}/", entry.prefix)) {
                let remainder = path[entry.prefix.len()..].to_owned();
                return Ok((&entry.targets, remainder));
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
    use async_trait::async_trait;
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

    #[async_trait]
    impl Mount for MemMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let path = components.join("/");
            Ok(Fid::new(MemFid { path, is_open: false }))
        }

        async fn open(&self, fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            let inner = fid.downcast_mut::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            inner.is_open = true;
            Ok(())
        }

        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
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

        async fn write(&self, _fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(data.len() as u32)
        }

        async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
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

        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat { qtype: 0, size: 0, name: inner.path.clone(), mtime: 0 })
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn test_subject() -> Subject { Subject::new("test") }

    #[tokio::test]
    async fn mount_and_cat() {
        let mut ns = Namespace::new();
        let mount: MountTarget = Arc::new(MemMount::new(vec![("temperature", b"0.7\n")]));
        ns.mount("/config", mount).unwrap();

        let data = ns.cat("/config/temperature", &test_subject()).await.unwrap();
        assert_eq!(data, b"0.7\n");
    }

    #[tokio::test]
    async fn mount_and_ls_root() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();
        ns.mount("/srv/mcp", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();
        ns.mount("/config", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();

        let entries = ns.ls("/", &test_subject()).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"srv"));
        assert!(names.contains(&"config"));
    }

    #[tokio::test]
    async fn longest_prefix_match() {
        let mut ns = Namespace::new();
        let srv: MountTarget = Arc::new(MemMount::new(vec![("status", b"ok")]));
        let nested: MountTarget = Arc::new(MemMount::new(vec![("status", b"loaded")]));
        ns.mount("/srv", srv).unwrap();
        ns.mount("/srv/model", nested).unwrap();

        let data = ns.cat("/srv/model/status", &test_subject()).await.unwrap();
        assert_eq!(data, b"loaded");
    }

    #[test]
    fn reserved_path_allows_mount() {
        let mut ns = Namespace::new();
        let mount: MountTarget = Arc::new(MemMount::new(vec![]));
        assert!(ns.mount("/config", mount).is_ok());
    }

    #[tokio::test]
    async fn fork_and_unmount() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![("status", b"loaded")])) as MountTarget).unwrap();
        ns.mount("/srv/worker", Arc::new(MemMount::new(vec![])) as MountTarget).unwrap();

        let mut child = ns.fork();
        child.unmount("/srv/worker");

        assert_eq!(ns.mount_prefixes().len(), 2);
        assert_eq!(child.mount_prefixes().len(), 1);
        assert_eq!(child.mount_prefixes()[0], "/srv/model");

        let data = child.cat("/srv/model/status", &test_subject()).await.unwrap();
        assert_eq!(data, b"loaded");
    }

    #[tokio::test]
    async fn not_found() {
        let ns = Namespace::new();
        assert!(matches!(ns.cat("/nonexistent/file", &test_subject()).await, Err(NamespaceError::Mount(MountError::NotFound(_)))));
    }

    #[tokio::test]
    async fn echo_writes() {
        let mut ns = Namespace::new();
        let mount: MountTarget = Arc::new(MemMount::new(vec![("ctl", b"")]));
        ns.mount("/cmd", mount).unwrap();

        assert!(ns.echo("/cmd/ctl", b"load qwen3:main", &test_subject()).await.is_ok());
    }

    // ── New tests ──────────────────────────────────────────────────────────

    /// Read loop must handle files larger than 64KB.
    #[tokio::test]
    async fn read_loop_large_file() {
        // Create a file > 64KB.
        let big_data = vec![0x42u8; 100 * 1024]; // 100 KB
        let mut ns = Namespace::new();

        struct ChunkedMount { data: Vec<u8> }
        struct ChunkedFid { offset_limit: bool }

        #[async_trait]
        impl Mount for ChunkedMount {
            async fn walk(&self, _c: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
                Ok(Fid::new(ChunkedFid { offset_limit: false }))
            }
            async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> { Ok(()) }
            async fn read(&self, _fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
                let start = offset as usize;
                if start >= self.data.len() { return Ok(vec![]); }
                let end = std::cmp::min(start + count as usize, self.data.len());
                Ok(self.data[start..end].to_vec())
            }
            async fn write(&self, _fid: &Fid, _o: u64, d: &[u8], _c: &Subject) -> Result<u32, MountError> { Ok(d.len() as u32) }
            async fn readdir(&self, _fid: &Fid, _c: &Subject) -> Result<Vec<DirEntry>, MountError> { Ok(vec![]) }
            async fn stat(&self, _fid: &Fid, _c: &Subject) -> Result<crate::mount::Stat, MountError> {
                Ok(crate::mount::Stat { qtype: 0, size: 0, name: String::new(), mtime: 0 })
            }
            async fn clunk(&self, _fid: Fid, _c: &Subject) {}
        }

        ns.mount("/big", Arc::new(ChunkedMount { data: big_data.clone() }) as MountTarget).unwrap();
        let result = ns.cat("/big/file", &test_subject()).await.unwrap();
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
    #[tokio::test]
    async fn resolve_with_dotdot_path() {
        let mut ns = Namespace::new();
        ns.mount("/srv/model", Arc::new(MemMount::new(vec![("status", b"ok")])) as MountTarget).unwrap();

        // /config/../srv/model/status should normalize to /srv/model/status
        let data = ns.cat("/config/../srv/model/status", &test_subject()).await.unwrap();
        assert_eq!(data, b"ok");
    }

    // ── Bind / union directory tests ──────────────────────────────────────

    #[tokio::test]
    async fn bind_before_walk_tries_new_mount_first() {
        let mut ns = Namespace::new();
        // Original mount has "status" = "original"
        let orig: MountTarget = Arc::new(MemMount::new(vec![("status", b"original")]));
        ns.mount("/bin", orig).unwrap();

        // Bind Before: new mount has "status" = "override"
        let overlay: MountTarget = Arc::new(MemMount::new(vec![("status", b"override")]));
        ns.bind_mount("/bin", overlay, BindFlag::Before).unwrap();

        let data = ns.cat("/bin/status", &test_subject()).await.unwrap();
        assert_eq!(data, b"override");
    }

    #[tokio::test]
    async fn bind_after_walk_tries_existing_first() {
        let mut ns = Namespace::new();
        let orig: MountTarget = Arc::new(MemMount::new(vec![("status", b"original")]));
        ns.mount("/bin", orig).unwrap();

        // Bind After: new mount also has "status", but original should win.
        let extra: MountTarget = Arc::new(MemMount::new(vec![("status", b"extra")]));
        ns.bind_mount("/bin", extra, BindFlag::After).unwrap();

        let data = ns.cat("/bin/status", &test_subject()).await.unwrap();
        assert_eq!(data, b"original");
    }

    #[tokio::test]
    async fn bind_after_fallback_to_new_mount() {
        let mut ns = Namespace::new();
        // Original has only "cat"
        let orig: MountTarget = Arc::new(MemMount::new(vec![("cat", b"cat-impl")]));
        ns.mount("/bin", orig).unwrap();

        // Bind After: new mount has "mytool" (not in original)
        let extra: MountTarget = Arc::new(MemMount::new(vec![("mytool", b"mytool-impl")]));
        ns.bind_mount("/bin", extra, BindFlag::After).unwrap();

        // cat resolves from original
        let data = ns.cat("/bin/cat", &test_subject()).await.unwrap();
        assert_eq!(data, b"cat-impl");

        // mytool falls through to the After mount
        let data = ns.cat("/bin/mytool", &test_subject()).await.unwrap();
        assert_eq!(data, b"mytool-impl");
    }

    #[tokio::test]
    async fn bind_before_fallback_to_existing() {
        let mut ns = Namespace::new();
        let orig: MountTarget = Arc::new(MemMount::new(vec![("cat", b"cat-impl")]));
        ns.mount("/bin", orig).unwrap();

        // Bind Before: new mount has "newtool" only
        let overlay: MountTarget = Arc::new(MemMount::new(vec![("newtool", b"newtool-impl")]));
        ns.bind_mount("/bin", overlay, BindFlag::Before).unwrap();

        // "cat" falls through to existing mount
        let data = ns.cat("/bin/cat", &test_subject()).await.unwrap();
        assert_eq!(data, b"cat-impl");

        // "newtool" found in Before mount
        let data = ns.cat("/bin/newtool", &test_subject()).await.unwrap();
        assert_eq!(data, b"newtool-impl");
    }

    #[tokio::test]
    async fn union_readdir_merges_entries() {
        let mut ns = Namespace::new();
        let mount_a: MountTarget = Arc::new(MemMount::new(vec![("cat", b""), ("ls", b"")]));
        ns.mount("/bin", mount_a).unwrap();

        let mount_b: MountTarget = Arc::new(MemMount::new(vec![("mytool", b""), ("cat", b"")]));
        ns.bind_mount("/bin", mount_b, BindFlag::After).unwrap();

        let entries = ns.ls("/bin", &test_subject()).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();

        // Should have "cat", "ls", and "mytool" — "cat" should not be duplicated.
        assert!(names.contains(&"cat"));
        assert!(names.contains(&"ls"));
        assert!(names.contains(&"mytool"));
        assert_eq!(names.iter().filter(|&&n| n == "cat").count(), 1);
    }

    #[tokio::test]
    async fn bind_replace_still_works() {
        let mut ns = Namespace::new();
        let orig: MountTarget = Arc::new(MemMount::new(vec![("status", b"original")]));
        ns.mount("/bin", orig).unwrap();

        // Add an After mount
        let extra: MountTarget = Arc::new(MemMount::new(vec![("extra", b"x")]));
        ns.bind_mount("/bin", extra, BindFlag::After).unwrap();

        // Replace should clear all targets at /bin
        let replacement: MountTarget = Arc::new(MemMount::new(vec![("status", b"replaced")]));
        ns.mount("/bin", replacement).unwrap();

        let data = ns.cat("/bin/status", &test_subject()).await.unwrap();
        assert_eq!(data, b"replaced");

        // "extra" should be gone (replaced)
        assert!(ns.cat("/bin/extra", &test_subject()).await.is_err());
    }
}
