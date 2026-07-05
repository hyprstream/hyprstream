//! VFS Mount for streaming data via named pipes.
//!
//! `/stream/{topic}/data` — read stream data (sequential, blocks until data/EOF)
//! `/stream/{topic}/info` — read stream metadata (JSON)
//! `/stream/{topic}/ctl`  — write control commands (cancel)
//!
//! Follows Plan 9 `/net/tcp/{n}/data` pattern. Streams are created by
//! dispatching streaming RPC methods via `RpcClient::open_stream()` and
//! registered in the StreamRegistry.

use std::collections::HashMap;

use parking_lot::Mutex;

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;
use hyprstream_rpc::stream_consumer::{StreamHandle, StreamPayload};

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
    streams: Mutex<HashMap<String, StreamEntry>>,
}

impl StreamRegistry {
    pub fn new() -> Self {
        Self {
            streams: Mutex::new(HashMap::new()),
        }
    }

    /// Register a new stream.
    pub fn register(&self, topic: String, entry: StreamEntry) {
        self.lock().insert(topic, entry);
    }

    /// Check if a topic exists.
    pub fn exists(&self, topic: &str) -> bool {
        self.lock().contains_key(topic)
    }

    /// List active topic names (optionally filtered by owner).
    pub fn list_topics(&self, owner_filter: Option<&str>) -> Vec<String> {
        self.lock()
            .iter()
            .filter(|(_, e)| owner_filter.is_none() || owner_filter == Some(e.owner.as_str()))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Remove a stream and return it for cleanup.
    pub fn remove(&self, topic: &str) -> Option<StreamEntry> {
        self.lock().remove(topic)
    }

    /// Lock the stream map (`parking_lot` — no poison).
    fn lock(&self) -> parking_lot::MutexGuard<'_, HashMap<String, StreamEntry>> {
        self.streams.lock()
    }
}

impl Default for StreamRegistry {
    fn default() -> Self {
        Self::new()
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
///
/// Native + wasm (#670): the `Mount` `read`/`write` path awaits
/// [`StreamHandle`] methods, which now return `Send` futures (the
/// `StreamHandle` trait dropped its `?Send`). That lets this mount be a
/// genuinely `Send + Sync` `Mount` on native — served to a Cloud Hypervisor
/// guest or any native namespace — while `#[async_trait(?Send)]` on wasm keeps
/// the browser path working. `StreamRegistry`/`StreamEntry` above are
/// target-agnostic so the generic service mount can register streams on any
/// target.
pub struct StreamMount {
    registry: std::sync::Arc<StreamRegistry>,
}

impl StreamMount {
    pub fn new(registry: std::sync::Arc<StreamRegistry>) -> Self {
        Self { registry }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
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
                    let streams = self.registry.lock();
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
                    let mut streams = self.registry.lock();
                    let entry = streams
                        .get_mut(topic.as_str())
                        .ok_or_else(|| MountError::NotFound(topic.clone()))?;
                    entry.handle.take()
                        .ok_or_else(|| MountError::Io("stream read in progress".into()))?
                };

                // Read next verified payload (await without holding borrow)
                let result = handle.next_payload().await;

                // Put handle back (its completion state is tracked internally).
                {
                    let mut streams = self.registry.lock();
                    if let Some(entry) = streams.get_mut(topic.as_str()) {
                        entry.handle = Some(handle);
                    }
                }

                match result {
                    Ok(Some(StreamPayload::Data(data))) => {
                        let mut streams = self.registry.lock();
                        if let Some(entry) = streams.get_mut(topic.as_str()) {
                            entry.bytes_received += data.len() as u64;
                            entry.blocks_received += 1;
                        }
                        Ok(data)
                    }
                    Ok(Some(StreamPayload::Complete(meta))) => {
                        let mut streams = self.registry.lock();
                        if let Some(entry) = streams.get_mut(topic.as_str()) {
                            entry.blocks_received += 1;
                        }
                        Ok(meta)
                    }
                    Ok(Some(StreamPayload::Error(msg))) => {
                        Err(MountError::Io(format!("stream error: {msg}")))
                    }
                    Ok(Some(StreamPayload::Tagged { payload, .. })) => {
                        let mut streams = self.registry.lock();
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
                let streams = self.registry.lock();
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
                            if let Some(mut handle) = entry.handle {
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

// ============================================================================
// Tests
// ============================================================================

// #670: these assertions are meaningful only on native, where `StreamMount` is
// `Send + Sync` and the `Mount::read` future must be `Send`. On wasm the mount
// is `#[async_trait(?Send)]` by design (JS-backed transport), so gate them out.
#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<F: std::future::Future + Send>(_f: F) {}

    /// #670: `StreamMount` and its registry must be a native `Send + Sync`
    /// `Mount`, so it can live behind `Arc<dyn Mount>` and be served from a
    /// multi-threaded runtime / to a CH guest.
    #[test]
    fn stream_mount_is_native_send_sync() {
        assert_send_sync::<StreamMount>();
        assert_send_sync::<StreamRegistry>();

        // Usable as a shared `Arc<dyn Mount>` — the whole point of un-gating it.
        fn takes_mount(_m: std::sync::Arc<dyn Mount>) {}
        let reg = std::sync::Arc::new(StreamRegistry::new());
        takes_mount(std::sync::Arc::new(StreamMount::new(reg)));
    }

    /// The `read` future itself must be `Send` — this is exactly what #670
    /// unblocks. `read`'s `Data` branch holds a `Box<dyn StreamHandle>` across
    /// `handle.next_payload().await`; if `StreamHandle`'s futures regressed to
    /// `!Send` the whole `read` future would be `!Send` and this stops
    /// compiling, regardless of which match arm runs at runtime.
    #[test]
    fn stream_mount_read_future_is_send() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        let mount = StreamMount::new(reg);
        assert_send(async move {
            let fid = Fid::new(StreamFidState::Data {
                topic: "topic".to_string(),
            });
            let caller = Subject::anonymous();
            let _ = mount.read(&fid, 0, 0, &caller).await;
        });
    }
}
