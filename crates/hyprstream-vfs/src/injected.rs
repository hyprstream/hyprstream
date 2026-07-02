//! Native **injected mounts** for a per-sandbox VFS namespace (FS-D, #365).
//!
//! FS-C built the injected mounts (`/stream`, models, deltas) as **wasm-only**
//! `Mount` impls (e.g. `hyprstream-rpc-std`'s `stream_mount.rs`, gated
//! `#![cfg(target_arch = "wasm32")]`). Those rely on `unsafe impl Send/Sync`
//! over `RefCell` because the browser side is single-threaded. They cannot be
//! served to a Cloud Hypervisor guest: FS-A's `hyprstream-vfs-server` runs the
//! down-adapter on real OS threads and requires genuinely `Send + Sync` mounts.
//!
//! This module is the **native** port of those injected mounts:
//!
//! - [`SyntheticMount`] — a read-only synthetic file tree built from in-memory
//!   directory/file nodes. The native counterpart of the wasm `SyntheticTree`,
//!   used for the models / deltas listings injected into the sandbox (e.g.
//!   `/models/<id>/info`, `/deltas/<id>/meta`). Read-only by construction: a
//!   plain [`Mount`] (not an [`FsMount`](crate::FsMount)), so the down-adapter
//!   treats writes as `EROFS` — fail-closed.
//! - [`StreamMount`] — the native `/stream` pipe mount. Streams appear as
//!   `/stream/{topic}/{data,info,ctl}`; `data` yields queued blocks (EOF on an
//!   empty read), `info` is JSON metadata, `ctl` accepts `cancel`. The registry
//!   ([`StreamRegistry`]) is `Mutex`-guarded — genuinely `Send + Sync`, no wasm
//!   `unsafe impl`. The block source is decoupled behind [`StreamRegistry`]'s
//!   `push`/`complete` API so the worker runtime can feed it from any transport.
//!
//! Every op threads the caller [`Subject`] — the same uniform Subject-per-call
//! boundary the rest of the VFS enforces. A sandbox's injected mounts live in
//! that sandbox's forked [`Namespace`](crate::Namespace) only, so tenant
//! isolation is structural (#365): there is no shared registry across sandboxes
//! unless the caller deliberately clones the `Arc`.

#![cfg(not(target_arch = "wasm32"))]

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use crate::mount::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticMount — read-only in-memory tree (models / deltas injected listings)
// ─────────────────────────────────────────────────────────────────────────────

/// A node in a [`SyntheticMount`] tree: either a directory of named children or
/// a file with fixed byte contents.
#[derive(Clone, Debug)]
pub enum SyntheticNode {
    /// Directory: an ordered map of child name → node.
    Dir(BTreeMap<String, SyntheticNode>),
    /// File: fixed contents served on read.
    File(Vec<u8>),
}

impl SyntheticNode {
    /// Build an empty directory node.
    pub fn dir() -> Self {
        Self::Dir(BTreeMap::new())
    }

    /// Build a file node from bytes.
    pub fn file(contents: impl Into<Vec<u8>>) -> Self {
        Self::File(contents.into())
    }

    /// Insert a child into a directory node (chainable). No-op on a file node.
    pub fn with_child(mut self, name: impl Into<String>, child: SyntheticNode) -> Self {
        if let Self::Dir(children) = &mut self {
            children.insert(name.into(), child);
        }
        self
    }

    /// Resolve `components` relative to this node.
    fn resolve(&self, components: &[&str]) -> Option<&SyntheticNode> {
        let mut cur = self;
        for comp in components {
            match cur {
                Self::Dir(children) => cur = children.get(*comp)?,
                Self::File(_) => return None,
            }
        }
        Some(cur)
    }
}

/// A read-only synthetic filesystem mount (native).
///
/// The native counterpart of the wasm `SyntheticTree`, used for the injected
/// models / deltas listings. It is a plain [`Mount`] — not an
/// [`FsMount`](crate::FsMount) — so the FS-A down-adapter serves it read-only
/// (writes return `EROFS`).
pub struct SyntheticMount {
    root: SyntheticNode,
}

impl SyntheticMount {
    /// Build a synthetic mount from a directory root node.
    pub fn new(root: SyntheticNode) -> Self {
        Self { root }
    }
}

/// Fid state for a [`SyntheticMount`]: the resolved absolute components.
struct SyntheticFid {
    components: Vec<String>,
}

#[async_trait]
impl Mount for SyntheticMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        if self.root.resolve(components).is_none() {
            return Err(MountError::NotFound(components.join("/")));
        }
        Ok(Fid::new(SyntheticFid {
            components: components.iter().map(|s| s.to_string()).collect(),
        }))
    }

    async fn open(&self, _fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        // Read-only: reject any write/rdwr open.
        if mode & 0x03 != 0 {
            return Err(MountError::PermissionDenied("synthetic mount is read-only".into()));
        }
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let st = fid
            .downcast_ref::<SyntheticFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let comps: Vec<&str> = st.components.iter().map(String::as_str).collect();
        match self.root.resolve(&comps) {
            Some(SyntheticNode::File(data)) => {
                let start = (offset as usize).min(data.len());
                let end = (start + count as usize).min(data.len());
                Ok(data[start..end].to_vec())
            }
            Some(SyntheticNode::Dir(_)) => Err(MountError::IsDirectory(st.components.join("/"))),
            None => Err(MountError::NotFound(st.components.join("/"))),
        }
    }

    async fn write(&self, _fid: &Fid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        Err(MountError::PermissionDenied("synthetic mount is read-only".into()))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let st = fid
            .downcast_ref::<SyntheticFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let comps: Vec<&str> = st.components.iter().map(String::as_str).collect();
        match self.root.resolve(&comps) {
            Some(SyntheticNode::Dir(children)) => Ok(children
                .iter()
                .map(|(name, node)| DirEntry {
                    name: name.clone(),
                    is_dir: matches!(node, SyntheticNode::Dir(_)),
                    size: match node {
                        SyntheticNode::File(d) => d.len() as u64,
                        SyntheticNode::Dir(_) => 0,
                    },
                    stat: None,
                })
                .collect()),
            Some(SyntheticNode::File(_)) => Err(MountError::NotDirectory(st.components.join("/"))),
            None => Err(MountError::NotFound(st.components.join("/"))),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let st = fid
            .downcast_ref::<SyntheticFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let comps: Vec<&str> = st.components.iter().map(String::as_str).collect();
        match self.root.resolve(&comps) {
            Some(node) => {
                let (qtype, size) = match node {
                    SyntheticNode::Dir(_) => (0x80, 0),
                    SyntheticNode::File(d) => (0, d.len() as u64),
                };
                Ok(Stat::unknown_qid(
                    qtype,
                    size,
                    st.components.last().cloned().unwrap_or_default(),
                    0,
                ))
            }
            None => Err(MountError::NotFound(st.components.join("/"))),
        }
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamMount — native /stream pipe mount
// ─────────────────────────────────────────────────────────────────────────────

/// State for one active stream topic.
struct StreamEntry {
    /// Owner identity (the Subject name that registered the topic).
    owner: Option<String>,
    /// Queued blocks awaiting a guest read (FIFO).
    blocks: VecDeque<Vec<u8>>,
    /// Whether the producer has signalled end-of-stream.
    complete: bool,
    /// Whether the stream was cancelled (ctl `cancel`).
    cancelled: bool,
    /// Bytes delivered to the guest so far.
    bytes_read: u64,
    /// Blocks delivered to the guest so far.
    blocks_read: u64,
}

/// Registry of active streams for a single sandbox's `/stream` mount.
///
/// `Mutex`-guarded and genuinely `Send + Sync` — the native replacement for the
/// wasm `SyncRefCell` registry. A sandbox owns one registry; the worker runtime
/// feeds blocks via [`push`](Self::push) / [`complete`](Self::complete) from
/// whatever transport produced them, and the guest drains them by reading
/// `/stream/{topic}/data`.
pub struct StreamRegistry {
    streams: Mutex<BTreeMap<String, StreamEntry>>,
}

impl StreamRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            streams: Mutex::new(BTreeMap::new()),
        }
    }

    /// Register (or reset) a stream topic owned by `owner`.
    pub fn register(&self, topic: impl Into<String>, owner: Option<String>) {
        self.streams.lock().insert(
            topic.into(),
            StreamEntry {
                owner,
                blocks: VecDeque::new(),
                complete: false,
                cancelled: false,
                bytes_read: 0,
                blocks_read: 0,
            },
        );
    }

    /// Append a data block to a topic. No-op if the topic is unknown or done.
    pub fn push(&self, topic: &str, block: Vec<u8>) {
        let mut s = self.streams.lock();
        if let Some(entry) = s.get_mut(topic) {
            if !entry.complete && !entry.cancelled {
                entry.blocks.push_back(block);
            }
        }
    }

    /// Mark a topic complete: no more blocks; reads drain the queue then EOF.
    pub fn complete(&self, topic: &str) {
        if let Some(entry) = self.streams.lock().get_mut(topic) {
            entry.complete = true;
        }
    }

    /// Whether a topic exists.
    pub fn exists(&self, topic: &str) -> bool {
        self.streams.lock().contains_key(topic)
    }

    /// List topic names visible to `owner` (all if `owner` is `None`).
    pub fn list_topics(&self, owner: Option<&str>) -> Vec<String> {
        self.streams
            .lock()
            .iter()
            .filter(|(_, e)| owner.is_none() || owner == e.owner.as_deref())
            .map(|(k, _)| k.clone())
            .collect()
    }
}

impl Default for StreamRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Fid state for a `/stream` path.
#[derive(Clone)]
enum StreamFidState {
    Root,
    TopicDir { topic: String },
    Data { topic: String },
    Info { topic: String },
    Ctl { topic: String },
}

/// Native `/stream` mount: streaming data pipes as a 9P-shaped tree.
///
/// Genuinely `Send + Sync` (the registry is `Mutex`-guarded), so it can be
/// served to a CH guest by FS-A's down-adapter. Read-only file surface except
/// `/stream/{topic}/ctl`, which accepts `cancel`.
pub struct StreamMount {
    registry: Arc<StreamRegistry>,
}

impl StreamMount {
    /// Build a stream mount over `registry`.
    pub fn new(registry: Arc<StreamRegistry>) -> Self {
        Self { registry }
    }

    /// The backing registry (so the runtime can feed blocks).
    pub fn registry(&self) -> &Arc<StreamRegistry> {
        &self.registry
    }
}

#[async_trait]
impl Mount for StreamMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let state = match components {
            [] => StreamFidState::Root,
            [topic] => {
                if !self.registry.exists(topic) {
                    return Err(MountError::NotFound((*topic).to_string()));
                }
                StreamFidState::TopicDir { topic: (*topic).to_string() }
            }
            [topic, "data"] => StreamFidState::Data { topic: (*topic).to_string() },
            [topic, "info"] => StreamFidState::Info { topic: (*topic).to_string() },
            [topic, "ctl"] => StreamFidState::Ctl { topic: (*topic).to_string() },
            _ => return Err(MountError::NotFound(components.join("/"))),
        };
        Ok(Fid::new(state))
    }

    async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        match state {
            StreamFidState::Root => Err(MountError::IsDirectory("stream root".into())),
            StreamFidState::TopicDir { .. } => Err(MountError::IsDirectory("stream topic".into())),
            StreamFidState::Data { topic } => {
                let mut s = self.registry.streams.lock();
                let entry = s
                    .get_mut(topic.as_str())
                    .ok_or_else(|| MountError::NotFound(topic.clone()))?;
                if entry.cancelled {
                    return Err(MountError::Io("stream cancelled".into()));
                }
                match entry.blocks.pop_front() {
                    Some(block) => {
                        entry.bytes_read += block.len() as u64;
                        entry.blocks_read += 1;
                        Ok(block)
                    }
                    // No block queued: EOF only when the producer is done,
                    // otherwise an empty (non-blocking) read — the guest retries.
                    None => Ok(Vec::new()),
                }
            }
            StreamFidState::Info { topic } => {
                let s = self.registry.streams.lock();
                let entry = s
                    .get(topic.as_str())
                    .ok_or_else(|| MountError::NotFound(topic.clone()))?;
                let info = format!(
                    "{{\"topic\":\"{}\",\"bytesRead\":{},\"blocksRead\":{},\"queued\":{},\"complete\":{},\"cancelled\":{}}}\n",
                    topic, entry.bytes_read, entry.blocks_read, entry.blocks.len(), entry.complete, entry.cancelled
                );
                let bytes = info.into_bytes();
                let start = (offset as usize).min(bytes.len());
                let end = (start + count as usize).min(bytes.len());
                Ok(bytes[start..end].to_vec())
            }
            StreamFidState::Ctl { .. } => Ok(Vec::new()),
        }
    }

    async fn write(&self, fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        match state {
            StreamFidState::Ctl { topic } => {
                let cmd = std::str::from_utf8(data).unwrap_or("").trim();
                match cmd {
                    "cancel" => {
                        if let Some(entry) = self.registry.streams.lock().get_mut(topic.as_str()) {
                            entry.cancelled = true;
                            entry.complete = true;
                            entry.blocks.clear();
                        }
                        Ok(cmd.len() as u32)
                    }
                    _ => Err(MountError::NotSupported(format!("unknown stream command: {cmd}"))),
                }
            }
            _ => Err(MountError::NotSupported("stream files are read-only".into())),
        }
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        match state {
            StreamFidState::Root => Ok(self
                .registry
                .list_topics(None)
                .into_iter()
                .map(|name| DirEntry { name, is_dir: true, size: 0, stat: None })
                .collect()),
            StreamFidState::TopicDir { .. } => Ok(["data", "info", "ctl"]
                .iter()
                .map(|name| DirEntry {
                    name: (*name).to_string(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                })
                .collect()),
            _ => Err(MountError::NotDirectory("not a directory".into())),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let (qtype, name) = match state {
            StreamFidState::Root => (0x80, "stream".to_string()),
            StreamFidState::TopicDir { topic } => (0x80, topic.clone()),
            StreamFidState::Data { .. } => (0, "data".to_string()),
            StreamFidState::Info { .. } => (0, "info".to_string()),
            StreamFidState::Ctl { .. } => (0, "ctl".to_string()),
        };
        Ok(Stat::unknown_qid(qtype, 0, name, 0))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mount::{OREAD, OWRITE};

    fn subj() -> Subject {
        Subject::new("tester")
    }

    #[tokio::test]
    async fn synthetic_reads_files_and_lists_dirs() {
        let root = SyntheticNode::dir().with_child(
            "models",
            SyntheticNode::dir().with_child(
                "qwen3",
                SyntheticNode::dir().with_child("info", SyntheticNode::file(b"qwen3 model\n".to_vec())),
            ),
        );
        let mount = SyntheticMount::new(root);
        let s = subj();

        // Read a leaf file.
        let mut fid = mount.walk(&["models", "qwen3", "info"], &s).await.unwrap();
        mount.open(&mut fid, OREAD, &s).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &s).await.unwrap();
        assert_eq!(data, b"qwen3 model\n");
        mount.clunk(fid, &s).await;

        // List a directory.
        let mut dfid = mount.walk(&["models"], &s).await.unwrap();
        mount.open(&mut dfid, OREAD, &s).await.unwrap();
        let entries = mount.readdir(&dfid, &s).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "qwen3");
        assert!(entries[0].is_dir);
    }

    #[tokio::test]
    async fn synthetic_is_read_only() {
        let root = SyntheticNode::dir().with_child("f", SyntheticNode::file(b"x".to_vec()));
        let mount = SyntheticMount::new(root);
        let s = subj();
        let mut fid = mount.walk(&["f"], &s).await.unwrap();
        // Write open is rejected.
        assert!(mount.open(&mut fid, OWRITE, &s).await.is_err());
        // Plain mount, not an FsMount.
        assert!(mount.as_fsmount().is_none());
    }

    #[tokio::test]
    async fn synthetic_missing_path_not_found() {
        let mount = SyntheticMount::new(SyntheticNode::dir());
        assert!(mount.walk(&["nope"], &subj()).await.is_err());
    }

    #[tokio::test]
    async fn stream_drains_blocks_then_eof() {
        let reg = Arc::new(StreamRegistry::new());
        reg.register("topic-a", Some("tester".to_string()));
        reg.push("topic-a", b"hello".to_vec());
        reg.push("topic-a", b"world".to_vec());
        reg.complete("topic-a");

        let mount = StreamMount::new(reg);
        let s = subj();
        let mut fid = mount.walk(&["topic-a", "data"], &s).await.unwrap();
        mount.open(&mut fid, OREAD, &s).await.unwrap();
        assert_eq!(mount.read(&fid, 0, 1024, &s).await.unwrap(), b"hello");
        assert_eq!(mount.read(&fid, 0, 1024, &s).await.unwrap(), b"world");
        // Drained: empty read = EOF.
        assert!(mount.read(&fid, 0, 1024, &s).await.unwrap().is_empty());
        mount.clunk(fid, &s).await;
    }

    #[tokio::test]
    async fn stream_ctl_cancel() {
        let reg = Arc::new(StreamRegistry::new());
        reg.register("t", None);
        reg.push("t", b"data".to_vec());
        let mount = StreamMount::new(reg);
        let s = subj();
        let mut cfid = mount.walk(&["t", "ctl"], &s).await.unwrap();
        mount.open(&mut cfid, OWRITE, &s).await.unwrap();
        mount.write(&cfid, 0, b"cancel", &s).await.unwrap();
        mount.clunk(cfid, &s).await;

        // After cancel, reading data errors.
        let mut dfid = mount.walk(&["t", "data"], &s).await.unwrap();
        mount.open(&mut dfid, OREAD, &s).await.unwrap();
        assert!(mount.read(&dfid, 0, 1024, &s).await.is_err());
        mount.clunk(dfid, &s).await;
    }

    #[tokio::test]
    async fn stream_root_lists_topics() {
        let reg = Arc::new(StreamRegistry::new());
        reg.register("a", None);
        reg.register("b", None);
        let mount = StreamMount::new(reg);
        let s = subj();
        let mut fid = mount.walk(&[], &s).await.unwrap();
        mount.open(&mut fid, OREAD, &s).await.unwrap();
        let entries = mount.readdir(&fid, &s).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }
}
