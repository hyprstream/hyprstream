//! VFS Mount for streaming data via named pipes.
//!
//! `/stream/{topic}/data` — read stream data (sequential, blocks until data/EOF)
//! `/stream/{topic}/info` — read stream metadata (JSON)
//! `/stream/{topic}/ctl`  — write control commands (cancel)
//!
//! Follows Plan 9 `/net/tcp/{n}/data` pattern. Streams are created by
//! dispatching streaming RPC methods and registered in the StreamRegistry.

#![cfg(target_arch = "wasm32")]

use std::collections::HashMap;

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;

use crate::wasm_exports::RpcSession;

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
    /// The SubStream providing data blocks
    pub sub_stream: hyprstream_rpc::web_transport::SubStream,
    /// HMAC chain handle for verification (from init_stream_hmac)
    pub hmac_handle: u32,
    /// Owner identity (for access control)
    pub owner: String,
    /// Bytes received so far
    pub bytes_received: u64,
    /// Blocks received so far
    pub blocks_received: u64,
    /// Whether the stream has completed
    pub complete: bool,
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
/// and `ctl` pseudo-files. Reading from `data` blocks until the next StreamBlock
/// arrives via the SUB channel.
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
        _offset: u64,
        _count: u32,
        caller: &Subject,
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
                let mut streams = self.registry.streams.borrow_mut();
                let entry = streams
                    .get_mut(topic.as_str())
                    .ok_or_else(|| MountError::NotFound(topic.clone()))?;

                if entry.complete {
                    return Ok(vec![]); // EOF
                }

                // Read next block from SubStream (blocks until data arrives)
                match entry.sub_stream.next_block().await {
                    Ok(js_val) => {
                        if js_val.is_null() {
                            entry.complete = true;
                            return Ok(vec![]); // EOF
                        }
                        // js_val is an Array of Uint8Array frames
                        // Frame 0 = capnp StreamBlock, Frame 1 = MAC (if present)
                        let arr = js_sys::Array::from(&js_val);
                        if arr.length() == 0 {
                            return Ok(vec![]);
                        }

                        let capnp_frame = js_sys::Uint8Array::from(arr.get(0));
                        let mut capnp_data = vec![0u8; capnp_frame.length() as usize];
                        capnp_frame.copy_to(&mut capnp_data);

                        // HMAC verification if MAC frame present
                        if arr.length() >= 2 {
                            let mac_frame = js_sys::Uint8Array::from(arr.get(1));
                            let mut mac = vec![0u8; mac_frame.length() as usize];
                            mac_frame.copy_to(&mut mac);

                            // Verify using per-stream HMAC handle
                            match hyprstream_rpc::wasm_api::verify_stream_block_step(
                                entry.hmac_handle,
                                &capnp_data,
                                &mac,
                            ) {
                                Ok(true) => {} // Valid
                                Ok(false) => {
                                    return Err(MountError::Io(
                                        "stream block HMAC verification failed".into(),
                                    ));
                                }
                                Err(e) => {
                                    return Err(MountError::Io(format!("HMAC error: {:?}", e)));
                                }
                            }
                        }

                        // Parse StreamBlock and extract data payload
                        let data = parse_stream_block_data(&capnp_data);
                        entry.bytes_received += data.len() as u64;
                        entry.blocks_received += 1;
                        Ok(data)
                    }
                    Err(e) => Err(MountError::Io(format!("stream read: {:?}", e))),
                }
            }
            StreamFidState::Info { topic } => {
                let streams = self.registry.streams.borrow();
                let entry = streams
                    .get(topic.as_str())
                    .ok_or_else(|| MountError::NotFound(topic.clone()))?;

                let info = serde_json::json!({
                    "topic": topic,
                    "bytesReceived": entry.bytes_received,
                    "blocksReceived": entry.blocks_received,
                    "complete": entry.complete,
                });
                Ok(serde_json::to_string_pretty(&info)
                    .unwrap_or_default()
                    .into_bytes())
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
                        if let Some(mut entry) = self.registry.remove(topic) {
                            entry.sub_stream.dispose();
                            hyprstream_rpc::wasm_api::close_stream_hmac(entry.hmac_handle);
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

    async fn readdir(
        &self,
        fid: &Fid,
        caller: &Subject,
    ) -> Result<Vec<DirEntry>, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match state {
            StreamFidState::Root => {
                // List active stream topics (filtered by caller for multi-tenant)
                let owner = caller.name().unwrap_or("anonymous");
                let topics = self.registry.list_topics(Some(owner));
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
            StreamFidState::TopicDir { .. } => {
                Ok(vec![
                    DirEntry { name: "data".into(), is_dir: false, size: 0, stat: None },
                    DirEntry { name: "info".into(), is_dir: false, size: 0, stat: None },
                    DirEntry { name: "ctl".into(), is_dir: false, size: 0, stat: None },
                ])
            }
            _ => Err(MountError::NotDirectory("not a directory".into())),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid
            .downcast_ref::<StreamFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        let (name, is_dir) = match state {
            StreamFidState::Root => ("stream", true),
            StreamFidState::TopicDir { topic } => (topic.as_str(), true),
            StreamFidState::Data { .. } => ("data", false),
            StreamFidState::Info { .. } => ("info", false),
            StreamFidState::Ctl { .. } => ("ctl", false),
        };

        Ok(Stat {
            qtype: if is_dir { 0x80 } else { 0 },
            size: 0,
            name: name.to_string(),
            mtime: 0,
        })
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        // If clunking a Data fid, mark stream for potential cleanup
        // (don't remove immediately — other fids may still reference it)
        if let Some(StreamFidState::Data { topic }) = fid.downcast_ref::<StreamFidState>() {
            let streams = self.registry.streams.borrow();
            if let Some(entry) = streams.get(topic.as_str()) {
                if entry.complete {
                    drop(streams);
                    // Stream finished and reader clunked — clean up
                    if let Some(mut entry) = self.registry.remove(topic) {
                        entry.sub_stream.dispose();
                        hyprstream_rpc::wasm_api::close_stream_hmac(entry.hmac_handle);
                    }
                }
            }
        }
    }
}

// ============================================================================
// StreamBlock parsing
// ============================================================================

/// Parse raw Cap'n Proto StreamBlock bytes and extract concatenated data payload.
fn parse_stream_block_data(capnp_bytes: &[u8]) -> Vec<u8> {
    // Try parsing as Cap'n Proto StreamBlock
    let reader = match capnp::serialize::read_message(
        &mut std::io::Cursor::new(capnp_bytes),
        capnp::message::ReaderOptions::new(),
    ) {
        Ok(r) => r,
        Err(_) => return capnp_bytes.to_vec(), // Fallback: return raw bytes
    };

    let block = match reader
        .get_root::<hyprstream_rpc::streaming_capnp::stream_block::Reader>()
    {
        Ok(b) => b,
        Err(_) => return capnp_bytes.to_vec(),
    };

    // Coalesce all data segments into a single buffer
    let mut result = Vec::new();
    if let Ok(payloads) = block.get_payloads() {
        for i in 0..payloads.len() {
            let payload = payloads.get(i);
            if let Ok(which) = payload.which() {
                use hyprstream_rpc::streaming_capnp::stream_payload::Which;
                match which {
                    Which::Data(Ok(data)) => result.extend_from_slice(data),
                    Which::Complete(Ok(data)) => result.extend_from_slice(data),
                    Which::Error(_) | Which::Heartbeat(()) | Which::Tagged(_) => {}
                    _ => {}
                }
            }
        }
    }

    if result.is_empty() {
        // No payloads — return raw bytes as fallback
        capnp_bytes.to_vec()
    } else {
        result
    }
}
