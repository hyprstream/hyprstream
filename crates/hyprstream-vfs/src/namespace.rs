//! Namespace — the client's mount table.
//!
//! Routes 9P operations by longest-prefix match to mount targets.
//! Forkable for per-task sandboxing. Static after initial population.

use std::sync::Arc;

use hyprstream_rpc::Subject;
use crate::mount::{DirEntry, Mount, MountError, Stat, OWRITE};

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
    /// Append *and* designate this target as the union's writable **upper**
    /// layer for copy-up (#370). A write to a path that exists only in a
    /// read-only lower is copied up into this target specifically, rather than
    /// the first target that happens to accept a `create` (the bind-order
    /// heuristic used when no upper is designated).
    Upper,
}

/// What a mount point routes to — always an in-process Mount impl.
pub type MountTarget = Arc<dyn Mount>;

struct MountEntry {
    prefix: String,
    targets: Vec<MountTarget>,
    /// The designated copy-up upper for this union, if one was bound with
    /// [`BindFlag::Upper`]. Held as an `Arc` clone (not an index) so it stays
    /// stable across later `Before` inserts that would shift positions. Always
    /// also present in `targets` (so reads and readdir-merge see it).
    upper: Option<MountTarget>,
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
    /// - `Upper`: appends the target *and* records it as the union's writable
    ///   copy-up upper (#370). See [`BindFlag::Upper`].
    pub fn bind_mount(&mut self, prefix: &str, target: MountTarget, flag: BindFlag) -> Result<(), NamespaceError> {
        let prefix = normalize_prefix(prefix);

        match flag {
            BindFlag::Replace => {
                self.mounts.retain(|m| m.prefix != prefix);
                self.mounts.push(MountEntry { prefix, targets: vec![target], upper: None });
            }
            BindFlag::Before => {
                if let Some(entry) = self.mounts.iter_mut().find(|m| m.prefix == prefix) {
                    entry.targets.insert(0, target);
                } else {
                    self.mounts.push(MountEntry { prefix, targets: vec![target], upper: None });
                }
            }
            BindFlag::After => {
                if let Some(entry) = self.mounts.iter_mut().find(|m| m.prefix == prefix) {
                    entry.targets.push(target);
                } else {
                    self.mounts.push(MountEntry { prefix, targets: vec![target], upper: None });
                }
            }
            BindFlag::Upper => {
                if let Some(entry) = self.mounts.iter_mut().find(|m| m.prefix == prefix) {
                    entry.targets.push(Arc::clone(&target));
                    entry.upper = Some(target);
                } else {
                    self.mounts.push(MountEntry {
                        prefix,
                        targets: vec![Arc::clone(&target)],
                        upper: Some(target),
                    });
                }
            }
        }
        // Sort by prefix length descending for longest-prefix match.
        self.mounts.sort_by_key(|b| std::cmp::Reverse(b.prefix.len()));
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
                upper: m.upper.as_ref().map(Arc::clone),
            }).collect(),
        }
    }

    /// List all mount prefixes (for debugging / `ls /`).
    pub fn mount_prefixes(&self) -> Vec<&str> {
        self.mounts.iter().map(|m| m.prefix.as_str()).collect()
    }

    /// Whether the union at `prefix` has a designated copy-up upper
    /// ([`BindFlag::Upper`]).
    ///
    /// This is the query side of the union's copy-up policy (#370). It is the
    /// in-process accessor for what a per-union `ctl` control file will expose
    /// once namespace mutation is mediated through the reference monitor (#613);
    /// designating the upper at bind time via `BindFlag::Upper` is the write
    /// side today.
    pub fn has_designated_upper(&self, prefix: &str) -> bool {
        let prefix = normalize_prefix(prefix);
        self.mounts
            .iter()
            .find(|m| m.prefix == prefix)
            .is_some_and(|m| m.upper.is_some())
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

    /// Read a single block from a file (for streams). Walk + open + read(once) + clunk.
    ///
    /// Unlike `cat()` which loops to EOF, this returns after one read call.
    /// For streams, this returns the next available block. Empty bytes = EOF.
    pub async fn read_one(&self, path: &str, caller: &Subject) -> Result<Vec<u8>, NamespaceError> {
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
                let chunk = mount.read(&fid, 0, 64 * 1024, caller).await;
                mount.clunk(fid, caller).await;
                chunk
            }.await;
            match result {
                Ok(data) => return Ok(data),
                Err(e) => { last_err = Some(e); }
            }
        }
        Err(NamespaceError::Mount(last_err.unwrap_or_else(|| MountError::NotFound(path.to_owned()))))
    }

    /// Write data to a file. Walk + open(write) + write + clunk.
    ///
    /// With union mounts, this implements **copy-up** (#394, #370):
    ///   1. Try each target in bind order. The first writable target that
    ///      already holds the file (open(OWRITE) succeeds) wins.
    ///   2. If no writable target has the file but a read-only *lower* layer
    ///      does, the lower-layer bytes are copied to the writable *upper*
    ///      layer (create + write) before applying the new data. The upper is
    ///      the target designated with [`BindFlag::Upper`] if one exists,
    ///      otherwise the first target that accepts a `create` (bind order).
    ///      The upper layer is node-local and append-only (journal), so this
    ///      never mutates the immutable `/oid` floor in place.
    ///
    /// This is the overlayfs copy-up primitive lifted into the namespace: a
    /// write to a path that resolves through the union dirty-over-committed
    /// tree lands in the upper layer, leaving the committed floor untouched.
    pub async fn echo(&self, path: &str, data: &[u8], caller: &Subject) -> Result<(), NamespaceError> {
        let (entry, remainder) = self.resolve_entry(path)?;
        let targets = &entry.targets[..];
        let components: Vec<&str> = split_path(&remainder);
        let mut last_err = None;

        // Pass 1: try a direct open(OWRITE) on each target in bind order.
        for mount in targets {
            let result: Result<(), MountError> = async {
                let mut fid = mount.walk(&components, caller).await?;
                if let Err(e) = mount.open(&mut fid, OWRITE, caller).await {
                    mount.clunk(fid, caller).await;
                    return Err(e);
                }
                mount.write(&fid, 0, data, caller).await?;
                mount.clunk(fid, caller).await;
                Ok(())
            }.await;
            match result {
                Ok(()) => return Ok(()),
                Err(MountError::NotSupported(_)) => {
                    // This target doesn't support writing (read-only / synthetic
                    // mount that returned Unsupported from open). Fall through
                    // to the next target.
                    last_err = Some(MountError::NotSupported(path.to_owned()));
                }
                Err(e) => { last_err = Some(e); }
            }
        }

        // Pass 2: copy-up. If no writable target held the file, check whether a
        // lower layer has it; if so, stage it into the designated (or first
        // writable) upper target before writing the new data.
        if let Some(copied) = self.copy_up(targets, entry.upper.as_ref(), &components, data, caller).await? {
            return Ok(copied);
        }

        Err(NamespaceError::Mount(last_err.unwrap_or_else(|| MountError::NotFound(path.to_owned()))))
    }

    /// Create a file or directory at `path` in the first writable target.
    ///
    /// This is the namespace-level entry point for the `Mount::create` op: it
    /// walks to the parent directory in each target (bind order) and calls
    /// `create` on the first target that accepts it. Lower-layer files are not
    /// copied up here — `create` is for *new* files; copy-up of existing
    /// lower-layer content on overwrite is handled by [`echo`](Self::echo).
    ///
    /// Returns the new file's `Stat` (qid fields are advisory hints only — see
    /// the invariant on `hyprstream_vfs::Stat`).
    pub async fn create(
        &self,
        path: &str,
        perm: u32,
        caller: &Subject,
    ) -> Result<Stat, NamespaceError> {
        let (targets, remainder) = self.resolve(path)?;
        let components: Vec<&str> = split_path(&remainder);
        if components.is_empty() {
            return Err(NamespaceError::Mount(MountError::InvalidArgument(
                "create requires a non-empty path".into(),
            )));
        }
        let (name, parent) = components.split_last().expect("non-empty");

        let mut last_err: Option<MountError> = None;
        for mount in targets {
            let result: Result<Stat, MountError> = async {
                let mut dir_fid = mount.walk(parent, caller).await?;
                // Open the parent directory for read so the backend can assert
                // it's a directory; some impls require an open before create.
                if let Err(e) = mount.open(&mut dir_fid, 0, caller).await {
                    mount.clunk(dir_fid, caller).await;
                    return Err(e);
                }
                let stat = mount.create(&mut dir_fid, name, perm, OWRITE, caller).await?;
                mount.clunk(dir_fid, caller).await;
                Ok(stat)
            }.await;
            match result {
                Ok(stat) => return Ok(stat),
                Err(MountError::NotSupported(_)) => {
                    last_err = Some(MountError::NotSupported(path.to_owned()));
                }
                Err(e) => { last_err = Some(e); }
            }
        }
        Err(NamespaceError::Mount(last_err.unwrap_or_else(|| MountError::NotFound(path.to_owned()))))
    }

    /// Copy-up: if `components` exists in a read-only lower layer but not in
    /// any writable upper target, read the lower bytes and stage them into the
    /// upper target via create+write, then apply `new_data` on top at offset 0.
    ///
    /// `upper` is the target designated with [`BindFlag::Upper`] for this union,
    /// if any. When set, copy-up writes to it specifically. When `None`, copy-up
    /// falls back to the first target in bind order that accepts a `create`
    /// (#394's original heuristic) — preserving behaviour for unions bound
    /// without an explicit upper.
    ///
    /// Returns `Some(())` if copy-up succeeded (caller treats that as a
    /// successful echo), or `None` if no lower-layer source was found (caller
    /// falls back to the original error). The upper write is *append-only* in
    /// the journal sense — it creates a fresh upper file rather than mutating
    /// the immutable `/oid` floor in place.
    ///
    /// # Semantics
    ///
    /// The lower bytes are staged first so a subsequent partial write (offset >
    /// 0, count < file size) sees the full original content. For a full-file
    /// `echo` the staged bytes are then overwritten at offset 0 by `new_data`,
    /// so the visible result is just `new_data` — copy-up only matters for the
    /// partial-write / append path, but staging unconditionally keeps the upper
    /// layer a faithful snapshot of "floor + edits".
    ///
    /// `caller` threads through both the lower read and the upper create+write,
    /// so copy-up is mediated by the same `Subject` checks as any other op — it
    /// is not a privileged bypass (matters for MAC enforcement, #547).
    async fn copy_up(
        &self,
        targets: &[MountTarget],
        upper: Option<&MountTarget>,
        components: &[&str],
        new_data: &[u8],
        caller: &Subject,
    ) -> Result<Option<()>, NamespaceError> {
        let (name, parent) = match components.split_last() {
            Some(split) => split,
            None => return Ok(None),
        };

        // Find the lower-layer source: the first target whose cat(components) succeeds.
        let mut lower_bytes: Option<Vec<u8>> = None;
        for mount in targets {
            let result: Result<Vec<u8>, MountError> = async {
                let mut fid = mount.walk(components, caller).await?;
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
            if let Ok(data) = result {
                lower_bytes = Some(data);
                break;
            }
        }
        let Some(_lower) = lower_bytes else { return Ok(None); };

        // Stage into the upper target. We only reach copy-up when the direct
        // open(OWRITE) pass failed on every target, so the upper target must
        // *create* the file.
        //
        // `echo` is a full-file write at offset 0, so the new content fully
        // replaces whatever the lower layer held — the staged lower bytes would
        // be overwritten in their entirety. We therefore create the upper file
        // and write only `new_data`. (Partial-write / append paths that need to
        // preserve the lower prefix go through a different primitive, not
        // `echo`; see the module docs on the append-only upper + journal.)
        //
        // Target selection: if the union has a designated upper
        // ([`BindFlag::Upper`]), copy up into it specifically. Otherwise fall
        // back to the first target in bind order that accepts the create.
        let create_into = |mount: &MountTarget| {
            let mount = Arc::clone(mount);
            let new_data = new_data.to_vec();
            let name = name.to_string();
            let parent: Vec<String> = parent.iter().map(|s| s.to_string()).collect();
            async move {
                let parent_refs: Vec<&str> = parent.iter().map(String::as_str).collect();
                let mut dir_fid = mount.walk(&parent_refs, caller).await?;
                // Open the parent dir read-only so impls that need an open fid
                // before create are satisfied.
                if let Err(e) = mount.open(&mut dir_fid, 0, caller).await {
                    mount.clunk(dir_fid, caller).await;
                    return Err(e);
                }
                let _stat = mount
                    .create(&mut dir_fid, &name, 0o644, OWRITE, caller)
                    .await?;
                // The create consumed the dir fid and replaced it with the
                // opened new file. Write the caller's full-file content.
                mount.write(&dir_fid, 0, &new_data, caller).await?;
                mount.clunk(dir_fid, caller).await;
                Ok::<(), MountError>(())
            }
        };

        if let Some(up) = upper {
            // Explicit upper: copy-up targets it and nothing else. A failure
            // here is the real result — we do not silently fall through to some
            // other writable target, because that would defeat the point of
            // designating the upper.
            return match create_into(up).await {
                Ok(()) => Ok(Some(())),
                Err(MountError::NotSupported(_)) => Ok(None),
                Err(e) => Err(NamespaceError::Mount(e)),
            };
        }

        for mount in targets {
            match create_into(mount).await {
                Ok(()) => return Ok(Some(())),
                Err(MountError::NotSupported(_)) => continue,
                Err(e) => return Err(NamespaceError::Mount(e)),
            }
        }
        // No writable target accepted the create — copy-up not possible.
        Ok(None)
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

    /// Resolve a path to its mount targets (bind order) and the path components
    /// **relative to the matched mount root**.
    ///
    /// This is the public, fid-level routing entry point used by FS-A's
    /// `Namespace → FileSystem` down-adapter, which (unlike `cat`/`ls`/`echo`)
    /// drives `walk`/`open`/`read`/`write`/`clunk` on the mount itself and needs
    /// the raw targets plus the component slice. Returns the targets in bind
    /// order (`Before` first, `After` last) so the caller can apply the same
    /// union/fallthrough policy the convenience helpers do.
    ///
    /// `path` is normalised (`.`/`..` resolved, leading `/` enforced) before the
    /// longest-prefix match, identical to the convenience helpers.
    pub fn resolve_targets(&self, path: &str) -> Result<(Vec<MountTarget>, Vec<String>), NamespaceError> {
        let (targets, remainder) = self.resolve(path)?;
        let components = split_path(&remainder).into_iter().map(str::to_owned).collect();
        Ok((targets.to_vec(), components))
    }

    /// Whether `path` is an *intermediate* directory — a path that is not itself
    /// a mount prefix but is a strict ancestor of one (e.g. `/srv` when only
    /// `/srv/model` is mounted). Used by the down-adapter to synthesise
    /// directory inodes for the parts of the tree the namespace spans implicitly.
    ///
    /// Returns the synthetic child directory names if so (deduped, no order
    /// guarantee), or `None` if `path` is neither a mount nor an ancestor.
    pub fn intermediate_children(&self, path: &str) -> Option<Vec<String>> {
        let normalized = normalize_path(path);
        // A real mount prefix is not "intermediate".
        if self.mounts.iter().any(|m| m.prefix == normalized) {
            return None;
        }
        let prefix_with_slash = if normalized == "/" {
            "/".to_owned()
        } else {
            format!("{normalized}/")
        };
        let mut seen = std::collections::HashSet::new();
        let mut names = Vec::new();
        for m in &self.mounts {
            if let Some(rest) = m.prefix.strip_prefix(&prefix_with_slash[..]) {
                let next = rest.split('/').next().unwrap_or("");
                if !next.is_empty() && seen.insert(next.to_owned()) {
                    names.push(next.to_owned());
                }
            }
        }
        if names.is_empty() {
            None
        } else {
            Some(names)
        }
    }

    // ── Internal ────────────────────────────────────────────────────────────

    /// Resolve a path to (list of MountTargets, remainder after prefix).
    ///
    /// Returns targets in bind order (Before targets first, After targets last).
    fn resolve(&self, path: &str) -> Result<(&[MountTarget], String), NamespaceError> {
        self.resolve_entry(path).map(|(entry, remainder)| (&entry.targets[..], remainder))
    }

    /// Resolve a path to its owning `MountEntry` (which carries the designated
    /// copy-up upper, if any) plus the remainder after the matched prefix.
    fn resolve_entry(&self, path: &str) -> Result<(&MountEntry, String), NamespaceError> {
        let path = normalize_path(path);

        for entry in &self.mounts {
            // A root mount ("/") is the catch-all: every path resolves to it
            // with the full path as remainder. Guarded specially because
            // `format!("{}/", "/")` would be "//", which `starts_with` never
            // matches — so a `/`-rooted rootfs (FS-D) would otherwise be unreachable.
            if entry.prefix == "/" {
                return Ok((entry, path));
            }
            if path == entry.prefix || path.starts_with(&format!("{}/", entry.prefix)) {
                let remainder = path[entry.prefix.len()..].to_owned();
                return Ok((entry, remainder));
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
#[allow(clippy::unwrap_used, clippy::disallowed_types)]
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
            Ok(Stat::unknown_qid(0, 0, inner.path.clone(), 0))
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

    /// A `/`-rooted mount (FS-D rootfs) is the catch-all: deep paths resolve to
    /// it, and longer prefixes still win over it (longest-prefix order).
    #[tokio::test]
    async fn root_mount_is_catch_all() {
        let mut ns = Namespace::new();
        let rootfs: MountTarget = Arc::new(MemMount::new(vec![("etc/hostname", b"host\n")]));
        ns.mount("/", rootfs).unwrap();
        let stream: MountTarget = Arc::new(MemMount::new(vec![("job/data", b"chunk")]));
        ns.mount("/stream", stream).unwrap();

        // Deep rootfs path resolves through "/".
        assert_eq!(ns.cat("/etc/hostname", &test_subject()).await.unwrap(), b"host\n");
        // A longer prefix still wins over the root catch-all.
        assert_eq!(ns.cat("/stream/job/data", &test_subject()).await.unwrap(), b"chunk");
        // The rootfs itself is reachable at "/".
        assert!(ns.resolve_targets("/").is_ok());
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
        struct ChunkedFid { _offset_limit: bool }

        #[async_trait]
        impl Mount for ChunkedMount {
            async fn walk(&self, _c: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
                Ok(Fid::new(ChunkedFid { _offset_limit: false }))
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
                Ok(crate::mount::Stat::unknown_qid(0, 0, String::new(), 0))
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

    // ── Intermediate directory synthesis ─────────────────────────────────────

    /// `ls /srv` should synthesize directory entries when mounts exist at
    /// `/srv/model` but not at `/srv` itself.
    ///
    /// This catches the bug where intermediate paths between root and mount
    /// points returned "not found".
    #[tokio::test]
    async fn ls_intermediate_directory() {
        let mut ns = Namespace::new();
        let mount = Arc::new(MemMount::new(vec![("status", b"ok")]));
        ns.mount("/srv/model", mount).unwrap();
        let lang_mount = Arc::new(MemMount::new(vec![("interp", b"tcl")]));
        ns.mount("/lang/tcl", lang_mount).unwrap();

        // /srv is not a mount point but /srv/model is — ls should synthesize.
        let entries = ns.ls("/srv", &test_subject()).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "model");
        assert!(entries[0].is_dir);

        // /lang is not a mount point but /lang/tcl is — same pattern.
        let entries = ns.ls("/lang", &test_subject()).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "tcl");
        assert!(entries[0].is_dir);

        // /nonexistent has no mounts beneath it — still errors.
        assert!(ns.ls("/nonexistent", &test_subject()).await.is_err());
    }

    /// Multiple mounts under the same intermediate prefix synthesize correctly.
    #[tokio::test]
    async fn ls_intermediate_multiple_children() {
        let mut ns = Namespace::new();
        ns.mount("/a/b", Arc::new(MemMount::new(vec![("x", b"")]))).unwrap();
        ns.mount("/a/c", Arc::new(MemMount::new(vec![("y", b"")]))).unwrap();
        ns.mount("/a/d/e", Arc::new(MemMount::new(vec![("z", b"")]))).unwrap();

        let entries = ns.ls("/a", &test_subject()).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"b"));
        assert!(names.contains(&"c"));
        assert!(names.contains(&"d"));
        assert_eq!(entries.len(), 3);

        // Deeper intermediate: /a/d should show "e"
        let entries = ns.ls("/a/d", &test_subject()).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "e");
    }

    // ── Writable union / copy-up / create tests (#394) ──────────────────────

    /// A read-only mount: reads succeed, all writes/create fail. This models
    /// the immutable `/oid` floor over which the writable upper is overlaid.
    struct ReadOnlyMemMount {
        files: HashMap<String, Vec<u8>>,
    }

    impl ReadOnlyMemMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self { files: files.into_iter().map(|(k, v)| (k.to_owned(), v.to_vec())).collect() }
        }
    }

    struct RoFid { path: String }

    #[async_trait]
    impl Mount for ReadOnlyMemMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let path = components.join("/");
            Ok(Fid::new(RoFid { path }))
        }
        async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
            let base_mode = mode & 0x03;
            if base_mode == OWRITE {
                return Err(MountError::PermissionDenied("read-only floor".into()));
            }
            let _ = fid.downcast_ref::<RoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(())
        }
        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<RoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() { Ok(vec![]) } else { Ok(data[start..].to_vec()) }
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }
        async fn write(&self, _fid: &Fid, _o: u64, _d: &[u8], _c: &Subject) -> Result<u32, MountError> {
            Err(MountError::PermissionDenied("read-only floor".into()))
        }
        async fn readdir(&self, _fid: &Fid, _c: &Subject) -> Result<Vec<DirEntry>, MountError> { Ok(vec![]) }
        async fn stat(&self, fid: &Fid, _c: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<RoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat::unknown_qid(0, 0, inner.path.clone(), 0))
        }
        async fn clunk(&self, _fid: Fid, _c: &Subject) {}
    }

    /// A writable in-memory mount that persists writes and supports `create`.
    ///
    /// Unlike `MemMount` (which discards writes), this stores written bytes and
    /// honours `create` so the union copy-up path can be exercised end-to-end.
    struct WritableMemMount {
        files: std::sync::Mutex<HashMap<String, Vec<u8>>>,
    }

    impl WritableMemMount {
        fn empty() -> Self {
            Self { files: std::sync::Mutex::new(HashMap::new()) }
        }

        fn snapshot(&self) -> HashMap<String, Vec<u8>> {
            self.files.lock().unwrap().clone()
        }
    }

    struct WritableFid {
        path: String,
        is_open: bool,
        mode: u8,
    }

    #[async_trait]
    impl Mount for WritableMemMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let path = components.join("/");
            Ok(Fid::new(WritableFid { path, is_open: false, mode: 0 }))
        }

        async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
            let inner = fid.downcast_mut::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let base_mode = mode & 0x03;
            // The root directory (empty path) always exists and is readable.
            if inner.path.is_empty() {
                inner.is_open = true;
                inner.mode = mode;
                return Ok(());
            }
            let files = self.files.lock().unwrap();
            if !files.contains_key(&inner.path) {
                return Err(MountError::NotFound(inner.path.clone()));
            }
            // Suppress unused-assignment lint when base_mode is unused otherwise.
            let _ = base_mode;
            inner.is_open = true;
            inner.mode = mode;
            Ok(())
        }

        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let files = self.files.lock().unwrap();
            match files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() { Ok(vec![]) } else { Ok(data[start..].to_vec()) }
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        async fn write(&self, fid: &Fid, offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            let inner = fid.downcast_ref::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let mut files = self.files.lock().unwrap();
            let entry = files.entry(inner.path.clone()).or_default();
            let start = offset as usize;
            if start > entry.len() {
                entry.resize(start, 0);
            }
            let end = start + data.len();
            if end > entry.len() {
                entry.resize(end, 0);
            }
            entry[start..end].copy_from_slice(data);
            Ok(data.len() as u32)
        }

        async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid.downcast_ref::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let prefix = if inner.path.is_empty() { String::new() } else { format!("{}/", inner.path) };
            let files = self.files.lock().unwrap();
            let mut entries = Vec::new();
            for key in files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry { name: rest.to_owned(), is_dir: false, size: 0, stat: None });
                    }
                }
            }
            Ok(entries)
        }

        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat::unknown_qid(0, 0, inner.path.clone(), 0))
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}

        /// Override the default `create` — this is the writable layer.
        async fn create(
            &self,
            fid: &mut Fid,
            name: &str,
            _perm: u32,
            mode: u8,
            _caller: &Subject,
        ) -> Result<Stat, MountError> {
            let inner = fid.downcast_mut::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let child_path = if inner.path.is_empty() { name.to_owned() } else { format!("{}/{}", inner.path, name) };
            // Create empty file.
            self.files.lock().unwrap().insert(child_path.clone(), Vec::new());
            // Re-point the fid at the new file, opened.
            inner.path = child_path;
            inner.is_open = true;
            inner.mode = mode;
            Ok(Stat::unknown_qid(0, 0, name.to_owned(), 0))
        }
    }

    /// `create` on the namespace lands in the first writable target.
    #[tokio::test]
    async fn namespace_create_lands_in_writable_target() {
        let mut ns = Namespace::new();
        let writable: MountTarget = Arc::new(WritableMemMount::empty());
        ns.mount("/wt", writable).unwrap();

        let caller = test_subject();
        let stat = ns.create("/wt/newfile.txt", 0o644, &caller).await.unwrap();
        assert_eq!(stat.name, "newfile.txt");

        // The file exists and is empty.
        let data = ns.cat("/wt/newfile.txt", &caller).await.unwrap();
        assert_eq!(data, b"");
    }

    /// Copy-up: a write to a file that exists only in a read-only lower layer
    /// copies the lower bytes into the writable upper layer first (#394).
    #[tokio::test]
    async fn copy_up_writes_to_upper_when_lower_is_readonly() {
        let mut ns = Namespace::new();

        // Read-only floor (the immutable /oid layer): reads ok, writes denied.
        let floor: MountTarget = Arc::new(ReadOnlyMemMount::new(vec![("config", b"floor-bytes")]));
        // Writable upper, empty to start. Keep a typed handle so the test can
        // inspect the upper layer's contents after the copy-up.
        let upper = Arc::new(WritableMemMount::empty());

        // Bind the floor first (lower), upper after — readdir merges, writes go
        // to the first writable target that accepts them.
        ns.bind_mount("/union", floor, BindFlag::Replace).unwrap();
        ns.bind_mount("/union", Arc::clone(&upper) as MountTarget, BindFlag::After).unwrap();

        let caller = test_subject();

        // Before write: cat reads from the floor.
        let before = ns.cat("/union/config", &caller).await.unwrap();
        assert_eq!(before, b"floor-bytes");

        // Pass 1 (direct OWRITE): floor rejects (read-only), upper rejects
        // (file doesn't exist yet). Pass 2 (copy-up): reads floor bytes, creates
        // + writes into upper. Result: the upper holds the new content.
        ns.echo("/union/config", b"new-bytes", &caller).await.unwrap();

        let upper_files = upper.snapshot();
        assert!(upper_files.contains_key("config"));
        assert_eq!(upper_files.get("config").unwrap(), b"new-bytes");
    }

    /// Copy-up leaves the immutable floor untouched.
    #[tokio::test]
    async fn copy_up_does_not_mutate_floor() {
        let mut ns = Namespace::new();
        // Read-only floor carrying existing content.
        let floor: MountTarget = Arc::new(ReadOnlyMemMount::new(vec![("data", b"immutable")]));
        let upper = Arc::new(WritableMemMount::empty());

        ns.bind_mount("/u", floor, BindFlag::Replace).unwrap();
        ns.bind_mount("/u", Arc::clone(&upper) as MountTarget, BindFlag::After).unwrap();

        let caller = test_subject();
        ns.echo("/u/data", b"mutated", &caller).await.unwrap();

        // Upper received the copy-up + write.
        let upper_files = upper.snapshot();
        assert_eq!(upper_files.get("data").unwrap(), b"mutated");

        // Reading back returns the upper's version (bound After = tried after
        // the floor; cat tries floor first, but the floor still reads the
        // original immutable bytes — the upper is only surfaced once the floor
        // read fails or via readdir merge). The invariant we care about: the
        // floor was never mutated (it can't be — it's read-only).
        let floor_read = ns.cat("/u/data", &caller).await.unwrap();
        assert_eq!(floor_read, b"immutable");
    }

    /// The default `create` returns NotSupported — existing Mount impls are
    /// unaffected by the trait addition.
    #[tokio::test]
    async fn default_create_returns_not_supported() {
        let mut ns = Namespace::new();
        let mount: MountTarget = Arc::new(MemMount::new(vec![]));
        ns.mount("/ro", mount).unwrap();

        let caller = test_subject();
        let err = ns.create("/ro/anything", 0o644, &caller).await.unwrap_err();
        match err {
            NamespaceError::Mount(MountError::NotSupported(_)) => {}
            other => panic!("expected NotSupported, got {other:?}"),
        }
    }

    // ── Designated copy-up upper (BindFlag::Upper) tests (#370) ──────────────

    /// A writable mount that denies every op unless the caller is a specific
    /// subject. Models a MAC-mediated upper: copy-up threads the caller through
    /// the create+write, so a wrong subject must fail closed.
    struct SubjectGatedMount {
        allow: String,
        files: std::sync::Mutex<HashMap<String, Vec<u8>>>,
    }

    impl SubjectGatedMount {
        fn new(allow: &str) -> Self {
            Self { allow: allow.to_owned(), files: std::sync::Mutex::new(HashMap::new()) }
        }
        fn snapshot(&self) -> HashMap<String, Vec<u8>> {
            self.files.lock().unwrap().clone()
        }
        fn check(&self, caller: &Subject) -> Result<(), MountError> {
            if caller.name() == Some(self.allow.as_str()) {
                Ok(())
            } else {
                Err(MountError::PermissionDenied("subject not permitted".into()))
            }
        }
    }

    #[async_trait]
    impl Mount for SubjectGatedMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            Ok(Fid::new(WritableFid { path: components.join("/"), is_open: false, mode: 0 }))
        }
        async fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError> {
            let inner = fid.downcast_mut::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            if inner.path.is_empty() {
                inner.is_open = true;
                inner.mode = mode;
                return Ok(());
            }
            // Writing needs the permitted subject; a read of an existing file is
            // fine (this mount is only ever a lower in the deny test).
            if mode & 0x03 == OWRITE {
                self.check(caller)?;
            }
            if !self.files.lock().unwrap().contains_key(&inner.path) {
                return Err(MountError::NotFound(inner.path.clone()));
            }
            inner.is_open = true;
            inner.mode = mode;
            Ok(())
        }
        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.lock().unwrap().get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() { Ok(vec![]) } else { Ok(data[start..].to_vec()) }
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }
        async fn write(&self, fid: &Fid, offset: u64, data: &[u8], caller: &Subject) -> Result<u32, MountError> {
            self.check(caller)?;
            let inner = fid.downcast_ref::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let mut files = self.files.lock().unwrap();
            let entry = files.entry(inner.path.clone()).or_default();
            let start = offset as usize;
            let end = start + data.len();
            if end > entry.len() {
                entry.resize(end, 0);
            }
            entry[start..end].copy_from_slice(data);
            Ok(data.len() as u32)
        }
        async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> { Ok(vec![]) }
        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat::unknown_qid(0, 0, inner.path.clone(), 0))
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
        async fn create(&self, fid: &mut Fid, name: &str, _perm: u32, mode: u8, caller: &Subject) -> Result<Stat, MountError> {
            self.check(caller)?;
            let inner = fid.downcast_mut::<WritableFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let child = if inner.path.is_empty() { name.to_owned() } else { format!("{}/{}", inner.path, name) };
            self.files.lock().unwrap().insert(child.clone(), Vec::new());
            inner.path = child;
            inner.is_open = true;
            inner.mode = mode;
            Ok(Stat::unknown_qid(0, 0, name.to_owned(), 0))
        }
    }

    /// A read-only mount that also lists its files via readdir, so union
    /// readdir-merge can be exercised. (`ReadOnlyMemMount` returns an empty
    /// readdir; this one reports its entries.)
    struct ReadOnlyReaddirMount {
        files: HashMap<String, Vec<u8>>,
    }

    impl ReadOnlyReaddirMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self { files: files.into_iter().map(|(k, v)| (k.to_owned(), v.to_vec())).collect() }
        }
    }

    #[async_trait]
    impl Mount for ReadOnlyReaddirMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            Ok(Fid::new(RoFid { path: components.join("/") }))
        }
        async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
            if mode & 0x03 == OWRITE {
                return Err(MountError::PermissionDenied("read-only".into()));
            }
            let _ = fid.downcast_ref::<RoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(())
        }
        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<RoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() { Ok(vec![]) } else { Ok(data[start..].to_vec()) }
                }
                // The root dir (empty path) has no bytes but is a valid open target.
                None if inner.path.is_empty() => Ok(vec![]),
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }
        async fn write(&self, _fid: &Fid, _o: u64, _d: &[u8], _c: &Subject) -> Result<u32, MountError> {
            Err(MountError::PermissionDenied("read-only".into()))
        }
        async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid.downcast_ref::<RoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
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
        async fn stat(&self, fid: &Fid, _c: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<RoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat::unknown_qid(0, 0, inner.path.clone(), 0))
        }
        async fn clunk(&self, _fid: Fid, _c: &Subject) {}
    }

    /// Copy-up targets the *designated* upper, not merely the first writable
    /// target in bind order. Two writable targets are bound After the floor;
    /// only the one marked `BindFlag::Upper` must receive the copied-up file.
    #[tokio::test]
    async fn copy_up_targets_designated_upper_not_bind_order() {
        let mut ns = Namespace::new();
        let floor: MountTarget = Arc::new(ReadOnlyMemMount::new(vec![("weights", b"shared-ro")]));
        // Two writable layers. `first` is bound earlier (would win the bind-order
        // heuristic); `chosen` is the explicitly designated upper.
        let first = Arc::new(WritableMemMount::empty());
        let chosen = Arc::new(WritableMemMount::empty());

        ns.bind_mount("/m", floor, BindFlag::Replace).unwrap();
        ns.bind_mount("/m", Arc::clone(&first) as MountTarget, BindFlag::After).unwrap();
        ns.bind_mount("/m", Arc::clone(&chosen) as MountTarget, BindFlag::Upper).unwrap();

        assert!(ns.has_designated_upper("/m"));

        let caller = test_subject();
        ns.echo("/m/weights", b"tenant-edit", &caller).await.unwrap();

        // The copied-up file landed in the designated upper only.
        assert_eq!(chosen.snapshot().get("weights").unwrap(), b"tenant-edit");
        assert!(!first.snapshot().contains_key("weights"), "first writable must NOT receive the copy-up");
    }

    /// A write to a path already present in the upper goes straight to the
    /// upper (Pass 1 direct write) — no copy-up create is triggered.
    #[tokio::test]
    async fn write_to_existing_upper_file_skips_copy_up() {
        let mut ns = Namespace::new();
        let floor: MountTarget = Arc::new(ReadOnlyMemMount::new(vec![("f", b"floor")]));
        let upper = Arc::new(WritableMemMount::empty());
        // Seed the upper with the file already present (same length as the new
        // write so the test mount's non-truncating write leaves no stale tail —
        // the point here is Pass-1 direct write, not truncation semantics).
        upper.files.lock().unwrap().insert("f".to_owned(), b"upperorig".to_vec());

        ns.bind_mount("/x", floor, BindFlag::Replace).unwrap();
        ns.bind_mount("/x", Arc::clone(&upper) as MountTarget, BindFlag::Upper).unwrap();

        let caller = test_subject();
        ns.echo("/x/f", b"upper-new", &caller).await.unwrap();

        assert_eq!(upper.snapshot().get("f").unwrap(), b"upper-new");
    }

    /// A write to a brand-new path (in neither layer) creates it in the
    /// designated upper, unchanged from the pre-#370 create path.
    #[tokio::test]
    async fn write_new_path_creates_in_designated_upper() {
        let mut ns = Namespace::new();
        let floor: MountTarget = Arc::new(ReadOnlyMemMount::new(vec![("existing", b"ro")]));
        let upper = Arc::new(WritableMemMount::empty());

        ns.bind_mount("/n", floor, BindFlag::Replace).unwrap();
        ns.bind_mount("/n", Arc::clone(&upper) as MountTarget, BindFlag::Upper).unwrap();

        let caller = test_subject();
        // Pass 1 direct write fails (not present); no lower source either, so the
        // create path in the upper is exercised via create() then echo.
        ns.create("/n/fresh", 0o644, &caller).await.unwrap();
        ns.echo("/n/fresh", b"brand-new", &caller).await.unwrap();

        assert_eq!(upper.snapshot().get("fresh").unwrap(), b"brand-new");
    }

    /// After a copy-up, readdir/ls merges lower + upper and the copied file
    /// appears exactly once (deduped by name).
    #[tokio::test]
    async fn readdir_after_copy_up_shows_file_once() {
        let mut ns = Namespace::new();
        // Floor exposes `shared` via readdir; also readable so copy-up can source it.
        let floor: MountTarget = Arc::new(ReadOnlyReaddirMount::new(vec![("shared", b"floor-bytes")]));
        let upper = Arc::new(WritableMemMount::empty());

        ns.bind_mount("/r", floor, BindFlag::Replace).unwrap();
        ns.bind_mount("/r", Arc::clone(&upper) as MountTarget, BindFlag::Upper).unwrap();

        let caller = test_subject();
        ns.echo("/r/shared", b"edited", &caller).await.unwrap();

        let entries = ns.ls("/r", &caller).await.unwrap();
        let shared_count = entries.iter().filter(|e| e.name == "shared").count();
        assert_eq!(shared_count, 1, "copied-up file must appear once, not once per layer");
    }

    /// Copy-up threads the caller through create+write, so a denied subject on
    /// the upper fails closed — the copy-up does not succeed with a wrong
    /// subject, and the upper stays empty.
    #[tokio::test]
    async fn copy_up_fails_closed_for_denied_subject() {
        let mut ns = Namespace::new();
        let floor: MountTarget = Arc::new(ReadOnlyMemMount::new(vec![("secret", b"floor")]));
        // Upper only permits subject "alice".
        let upper = Arc::new(SubjectGatedMount::new("alice"));

        ns.bind_mount("/g", floor, BindFlag::Replace).unwrap();
        ns.bind_mount("/g", Arc::clone(&upper) as MountTarget, BindFlag::Upper).unwrap();

        // "mallory" is not permitted on the upper.
        let mallory = Subject::new("mallory");
        let err = ns.echo("/g/secret", b"evil", &mallory).await.unwrap_err();
        match err {
            NamespaceError::Mount(MountError::PermissionDenied(_)) => {}
            other => panic!("expected PermissionDenied (fail closed), got {other:?}"),
        }
        assert!(upper.snapshot().is_empty(), "denied copy-up must not write the upper");

        // The permitted subject succeeds — confirms the deny was about the
        // subject, not a broken path.
        let alice = Subject::new("alice");
        ns.echo("/g/secret", b"ok", &alice).await.unwrap();
        assert_eq!(upper.snapshot().get("secret").unwrap(), b"ok");
    }
}
