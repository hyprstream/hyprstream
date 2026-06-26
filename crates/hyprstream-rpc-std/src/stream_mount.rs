//! VFS Mount for streaming data via named pipes.
//!
//! `/stream/{topic}/data` — read stream data (sequential, blocks until data/EOF)
//! `/stream/{topic}/info` — read stream metadata (JSON)
//! `/stream/{topic}/ctl`  — write control commands (cancel)
//!
//! Follows Plan 9 `/net/tcp/{n}/data` pattern. Streams are created by
//! dispatching streaming RPC methods via `RpcClient::open_stream()` and
//! registered in the StreamRegistry.

#![cfg(target_arch = "wasm32")]

use std::collections::HashMap;

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;
use hyprstream_rpc::stream_consumer::{StreamHandle, StreamPayload};

// ============================================================================
// Send+Sync wrappers for wasm32
// ============================================================================

/// RefCell wrapper that is Send+Sync on wasm32 (single-threaded).
struct SyncRefCell<T>(std::cell::RefCell<T>);
unsafe impl<T> Send for SyncRefCell<T> {}
unsafe impl<T> Sync for SyncRefCell<T> {}

impl<T> SyncRefCell<T> {
    fn new(val: T) -> Self {
        Self(std::cell::RefCell::new(val))
    }
    fn borrow(&self) -> std::cell::Ref<'_, T> {
        self.0.borrow()
    }
    fn borrow_mut(&self) -> std::cell::RefMut<'_, T> {
        self.0.borrow_mut()
    }
}

// ============================================================================
// Stream Registry — tracks active streams
// ============================================================================

/// Metadata and state for an active stream.
pub struct StreamEntry {
    /// Verified stream handle — does HMAC verification internally.
    pub handle: Option<Box<dyn StreamHandle>>,
    /// Owner identity (for access control)
    pub owner: String,
    /// Bytes received so far
    pub bytes_received: u64,
    /// Blocks received so far
    pub blocks_received: u64,
}

/// Registry of active streams, keyed by topic.
pub struct StreamRegistry {
    streams: SyncRefCell<HashMap<String, StreamEntry>>,
}

// SAFETY: wasm32 is single-threaded
unsafe impl Send for StreamRegistry {}
unsafe impl Sync for StreamRegistry {}

impl StreamRegistry {
    pub fn new() -> Self {
        Self {
            streams: SyncRefCell::new(HashMap::new()),
        }
    }

    /// Register a new stream.
    pub fn register(&self, topic: String, entry: StreamEntry) {
        self.streams.borrow_mut().insert(topic, entry);
    }

    /// Check if a topic exists.
    pub fn exists(&self, topic: &str) -> bool {
        self.streams.borrow().contains_key(topic)
    }

    /// List active topic names (optionally filtered by owner).
    pub fn list_topics(&self, owner_filter: Option<&str>) -> Vec<String> {
        self.streams
            .borrow()
            .iter()
            .filter(|(_, e)| owner_filter.is_none() || owner_filter == Some(e.owner.as_str()))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Remove a stream and return it for cleanup.
    pub fn remove(&self, topic: &str) -> Option<StreamEntry> {
        self.streams.borrow_mut().remove(topic)
    }
}

// ============================================================================
// Fid state for stream paths
// ============================================================================

#[derive(Clone, Debug)]
enum StreamFidState {
    /// /stream/ root directory
    Root,
    /// /stream/{topic}/ directory
    TopicDir { topic: String },
    /// /stream/{topic}/data — active data pipe
    Data { topic: String },
    /// /stream/{topic}/info — metadata
    Info { topic: String },
    /// /stream/{topic}/ctl — control
    Ctl { topic: String },
}

// ============================================================================
// StreamMount — serves /stream/ namespace
// ============================================================================

/// VFS mount for streaming data pipes.
///
/// Streams appear as directories under `/stream/{topic}/` with `data`, `info`,
/// and `ctl` pseudo-files. Reading from `data` blocks until the next verified
/// payload arrives via the StreamHandle.
pub struct StreamMount {
    registry: std::sync::Arc<StreamRegistry>,
}

impl StreamMount {
    pub fn new(registry: std::sync::Arc<StreamRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait(?Send)]
impl Mount for StreamMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let state = match components {
            [] => StreamFidState::Root,
            [topic] => {
                if !self.registry.exists(topic) {
                    return Err(MountError::NotFound(topic.to_string()));
                }
                StreamFidState::TopicDir {
                    topic: topic.to_string(),
                }
            }
            [topic, "data"] => StreamFidState::Data {
                topic: topic.to_string(),
            },
            [topic, "info"] => StreamFidState::Info {
                topic: topic.to_string(),
            },
            [topic, "ctl"] => StreamFidState::Ctl {
                topic: topic.to_string(),
            },
            _ => {
                return Err(MountError::NotFound(components.join("/")));
            }
        };
        Ok(Fid::new(state))
    }

    async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match state {
            StreamFidState::Root => {
                Err(MountError::IsDirectory("stream root".into()))
            }
            StreamFidState::TopicDir { .. } => {
                Err(MountError::IsDirectory("stream topic dir".into()))
            }
            StreamFidState::Data { topic } => {
                // Check completion WITHOUT holding borrow across await
                let is_completed = {
                    let streams = self.registry.streams.borrow();
                    let entry = streams
                        .get(topic.as_str())
                        .ok_or_else(|| MountError::NotFound(topic.clone()))?;
                    entry.handle.as_ref().map(|h| h.is_completed()).unwrap_or(true)
                };

                if is_completed {
                    return Ok(vec![]); // EOF
                }

                // Take handle out temporarily (avoids holding RefCell borrow across await)
                let mut handle = {
                    let mut streams = self.registry.streams.borrow_mut();
                    let entry = streams
                        .get_mut(topic.as_str())
                        .ok_or_else(|| MountError::NotFound(topic.clone()))?;
                    entry.handle.take()
                        .ok_or_else(|| MountError::Io("stream read in progress".into()))?
                };

                // Read next verified payload (await without holding borrow)
                let result = handle.next_payload().await;

                // Put handle back
                let is_completed = handle.is_completed();
                {
                    let mut streams = self.registry.streams.borrow_mut();
                    if let Some(entry) = streams.get_mut(topic.as_str()) {
                        entry.handle = Some(handle);
                    }
                }

                match result {
                    Ok(Some(StreamPayload::Data(data))) => {
                        let mut streams = self.registry.streams.borrow_mut();
                        if let Some(entry) = streams.get_mut(topic.as_str()) {
                            entry.bytes_received += data.len() as u64;
                            entry.blocks_received += 1;
                        }
                        Ok(data)
                    }
                    Ok(Some(StreamPayload::Complete(meta))) => {
                        let mut streams = self.registry.streams.borrow_mut();
                        if let Some(entry) = streams.get_mut(topic.as_str()) {
                            entry.blocks_received += 1;
                        }
                        Ok(meta)
                    }
                    Ok(Some(StreamPayload::Error(msg))) => {
                        Err(MountError::Io(format!("stream error: {msg}")))
                    }
                    Ok(Some(StreamPayload::Tagged { payload, .. })) => {
                        let mut streams = self.registry.streams.borrow_mut();
                        if let Some(entry) = streams.get_mut(topic.as_str()) {
                            entry.bytes_received += payload.len() as u64;
                            entry.blocks_received += 1;
                        }
                        Ok(payload)
                    }
                    Ok(None) => Ok(vec![]), // EOF
                    Err(e) => Err(MountError::Io(format!("stream read: {e}"))),
                }
            }
            StreamFidState::Info { topic } => {
                let streams = self.registry.streams.borrow();
                let entry = streams
                    .get(topic.as_str())
                    .ok_or_else(|| MountError::NotFound(topic.clone()))?;

                let complete = entry.handle.as_ref().map(|h| h.is_completed()).unwrap_or(true);
                let info = serde_json::json!({
                    "topic": topic,
                    "bytesReceived": entry.bytes_received,
                    "blocksReceived": entry.blocks_received,
                    "complete": complete,
                });
                let bytes = serde_json::to_string_pretty(&info)
                    .unwrap_or_default()
                    .into_bytes();
                let start = (offset as usize).min(bytes.len());
                let end = (start + count as usize).min(bytes.len());
                Ok(bytes[start..end].to_vec())
            }
            StreamFidState::Ctl { .. } => {
                // Ctl is write-only
                Ok(vec![])
            }
        }
    }

    async fn write(
        &self,
        fid: &Fid,
        _offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match state {
            StreamFidState::Ctl { topic } => {
                let cmd = std::str::from_utf8(data).unwrap_or("").trim();
                match cmd {
                    "cancel" => {
                        if let Some(entry) = self.registry.remove(topic) {
                            if let Some(handle) = entry.handle {
                                let _ = handle.cancel().await;
                            }
                        }
                        Ok(cmd.len() as u32)
                    }
                    _ => Err(MountError::NotSupported(format!(
                        "unknown stream command: {}",
                        cmd
                    ))),
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
            StreamFidState::Root => {
                let topics = self.registry.list_topics(None);
                Ok(topics
                    .into_iter()
                    .map(|name| DirEntry {
                        name,
                        is_dir: true,
                        size: 0,
                        stat: None,
                    })
                    .collect())
            }
            StreamFidState::TopicDir { .. } => Ok(vec![
                DirEntry {
                    name: "data".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "info".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "ctl".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
            ]),
            _ => Err(MountError::NotDirectory("not a directory".into())),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let (qtype, name) = match state {
            StreamFidState::Root => (0x80, "stream"),
            StreamFidState::TopicDir { topic } => (0x80, topic.as_str()),
            StreamFidState::Data { .. } => (0, "data"),
            StreamFidState::Info { .. } => (0, "info"),
            StreamFidState::Ctl { .. } => (0, "ctl"),
        };
        Ok(Stat::unknown_qid(qtype, 0, name.to_string(), 0))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}
