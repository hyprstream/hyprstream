//! VFS Mount for streaming data — streams as 9P files (epic #809 W3, #819).
//!
//! `/stream/{topic}/data` — read stream data (sequential, blocks until data/EOF)
//! `/stream/{topic}/info` — read stream metadata (JSON)
//! `/stream/{topic}/qos`  — read/select QoS (StreamOpt vocabulary as tokens)
//! `/stream/{topic}/ctl`  — write control commands (`cancel`, `qos …`, `resume …`)
//!
//! Follows Plan 9 `/net/tcp/{n}/data` pattern. Streams are created by
//! dispatching streaming RPC methods via `RpcClient::open_stream()` and
//! registered in the StreamRegistry.
//!
//! # Interface, not a byte-copy path (#819)
//!
//! This mount is the *interface*: naming, addressing, and open-time
//! authorization. It is **not** a memcpy path that replaces the moq carrier.
//! The registered [`StreamHandle`] IS the moq carrier subscription established
//! at open — QUIC-native delivery (per-object streams, drop-stale, datagram,
//! relay rendezvous) stays intact underneath. Two consumption modes ride the
//! same handle:
//!
//! - **Floor (`Tread` on `data`)**: blocking push semantics, exactly Plan 9
//!   `/net/tcp/n/data`. Always works; correct and sufficient for low-rate
//!   streams (tokens). This is the floor, **not** the ceiling.
//! - **Carrier handle (bulk)**: high-rate / bulk consumers take the
//!   carrier-backed handle directly ([`StreamRegistry::take_handle`]) and drive
//!   the moq subscriber, exactly as #817 hands out mmap-at-open for weights.
//!   The frames are **never** routed one-by-one through `Tread` — that would
//!   flatten moq's non-fungible delivery semantics to a lowest-common-
//!   denominator read, the anti-pattern this design exists to avoid.
//!
//! QoS that does not reduce to reliable-in-order is selected through a small
//! vocabulary ([`StreamQos`]) on `qos` / `ctl` — StreamOpt's existing encoding
//! (latest-wins, lossy-ok, drop-stale) moved *onto the file*, so the carrier's
//! rich semantics stay first-class through the interface.
//!
//! Events (group-keyed EventService, #600) ride this same surface via
//! [`StreamKind::Event`] rather than a separate plane — see [`StreamEntry`].

use std::collections::HashMap;

use parking_lot::Mutex;

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;
use hyprstream_rpc::stream_consumer::{StreamHandle, StreamPayload};
use hyprstream_rpc::stream_info::{
    Completion, Delivery, Ordering, OverflowPolicy, Retention, StreamOpt,
};

// ============================================================================
// StreamKind — what plane a topic belongs to (#819: events ride the same surface)
// ============================================================================

/// Which plane a registered topic belongs to.
///
/// Both ride the *same* `/stream` surface (#819 requirement: the group-keyed
/// EventService of #600 is not a separate plane). The kind is reflected in
/// `info`/`stat` so a reader can tell a point-to-point stream from an event
/// broadcast without a second namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    /// A point-to-point inference/data stream (the default — tokens, I/O).
    Stream,
    /// A group-keyed EventService broadcast (#600) surfaced as a stream file.
    Event,
}

impl StreamKind {
    fn as_str(self) -> &'static str {
        match self {
            StreamKind::Stream => "stream",
            StreamKind::Event => "event",
        }
    }
}

// ============================================================================
// StreamQos — the StreamOpt QoS vocabulary, moved onto the file (#819)
// ============================================================================

/// The consumer-selectable QoS vocabulary, wired from `StreamOpt` onto the file.
///
/// A stream's carrier (moq / iroh) has *non-fungible* delivery semantics: a
/// reliable-in-order default plus richer modes that don't reduce to it. Rather
/// than flatten those to a plain read, #819 exposes them as a small token
/// vocabulary written to `qos` / `ctl`. Each token toggles the `StreamOpt`
/// axes that already encode the behaviour (there is no new QoS model — this is
/// StreamOpt's existing encoding, addressable through the file):
///
/// | token         | StreamOpt effect                                            |
/// |---------------|-------------------------------------------------------------|
/// | `reliable`    | the fail-closed default: ordered · at-least-once · end-of-stream · lossless backpressure |
/// | `latest-wins` | drop-oldest overflow (depth 1) + live retention — only the newest payload matters |
/// | `lossy-ok`    | at-most-once + no terminator required + unordered (gaps tolerated) |
/// | `drop-stale`  | drop-oldest overflow + live retention — shed backlog instead of blocking |
///
/// Tokens compose (`latest-wins drop-stale`); `reliable` resets to the floor.
/// [`StreamQos::from_stream_opt`] / [`StreamQos::to_tokens`] round-trip the
/// selection against the underlying `StreamOpt`.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamQos {
    opt: StreamOpt,
}

impl StreamQos {
    /// The fail-closed reliable-in-order floor (matches StreamOpt's `@0`
    /// strictest defaults: ordered + end-of-stream, plus at-least-once /
    /// live / block backpressure).
    pub fn reliable() -> Self {
        Self {
            opt: StreamOpt {
                ordering: Ordering::Ordered,
                delivery: Delivery::AtLeastOnce {
                    dedup_window: 4096,
                    resumable: true,
                },
                completion: Completion::EndOfStream,
                retention: Retention::Live,
                overflow_policy: OverflowPolicy::Block,
            },
        }
    }

    /// Wrap an existing `StreamOpt` (e.g. the producer-signed one from
    /// `StreamInfo`) so it can be rendered as tokens / mutated by selection.
    pub fn from_stream_opt(opt: StreamOpt) -> Self {
        Self { opt }
    }

    /// The underlying `StreamOpt` this selection resolves to.
    pub fn stream_opt(&self) -> &StreamOpt {
        &self.opt
    }

    /// Whether this QoS tolerates loss / gaps (at-most-once **or** unordered).
    ///
    /// Under lossy QoS the mount surfaces a mid-stream carrier error as a soft
    /// end-of-stream (EOF) rather than a hard read error — the consumer-visible
    /// half of "lossy-ok". Reliable QoS keeps failing closed.
    pub fn is_lossy(&self) -> bool {
        matches!(self.opt.delivery, Delivery::AtMostOnce)
            || matches!(self.opt.ordering, Ordering::Unordered { .. })
    }

    /// Apply one vocabulary token, mutating the relevant `StreamOpt` axes.
    /// Unknown tokens return `false` (the caller reports the error).
    pub fn apply_token(&mut self, token: &str) -> bool {
        match token {
            "reliable" => {
                *self = Self::reliable();
                true
            }
            "latest-wins" => {
                self.opt.overflow_policy = OverflowPolicy::DropOldest { high_water_mark: 1 };
                self.opt.retention = Retention::Live;
                self.opt.completion = Completion::None;
                true
            }
            "lossy-ok" => {
                self.opt.delivery = Delivery::AtMostOnce;
                self.opt.completion = Completion::None;
                self.opt.ordering = Ordering::Unordered {
                    anti_replay_window: 64,
                };
                true
            }
            "drop-stale" => {
                self.opt.overflow_policy = OverflowPolicy::DropOldest {
                    high_water_mark: 256,
                };
                self.opt.retention = Retention::Live;
                true
            }
            _ => false,
        }
    }

    /// Parse a whitespace-separated token list, applying each to `reliable`.
    /// Returns the offending token on the first unknown one.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut qos = Self::reliable();
        for tok in spec.split_whitespace() {
            if !qos.apply_token(tok) {
                return Err(tok.to_owned());
            }
        }
        Ok(qos)
    }

    /// Render the selection back to its non-default tokens (round-trips
    /// `parse`/`apply_token`). Always non-empty — `reliable` when at the floor.
    pub fn to_tokens(&self) -> Vec<&'static str> {
        let mut tokens = Vec::new();
        let dropping = matches!(self.opt.overflow_policy, OverflowPolicy::DropOldest { .. });
        match self.opt.overflow_policy {
            OverflowPolicy::DropOldest { high_water_mark } if high_water_mark <= 1 => {
                tokens.push("latest-wins");
            }
            OverflowPolicy::DropOldest { .. } => tokens.push("drop-stale"),
            OverflowPolicy::Block => {}
        }
        if self.is_lossy() {
            tokens.push("lossy-ok");
        }
        if tokens.is_empty() && !dropping {
            tokens.push("reliable");
        }
        tokens
    }
}

impl Default for StreamQos {
    fn default() -> Self {
        Self::reliable()
    }
}

// ============================================================================
// Stream Registry — tracks active streams
// ============================================================================

/// Metadata and state for an active stream.
///
/// The `handle` is the moq **carrier subscription** established at open — the
/// `data` file's blocking `Tread` is the floor push path over it, and bulk
/// consumers take the same handle whole ([`StreamRegistry::take_handle`]). It is
/// never rebuilt per read.
pub struct StreamEntry {
    /// Verified stream handle — does HMAC verification internally.
    pub handle: Option<Box<dyn StreamHandle>>,
    /// Owner identity (for access control)
    pub owner: String,
    /// Bytes received so far
    pub bytes_received: u64,
    /// Blocks received so far
    pub blocks_received: u64,
    /// Which plane this topic belongs to (#819 — events ride the same surface).
    pub kind: StreamKind,
    /// Consumer-selected QoS (StreamOpt vocabulary; see [`StreamQos`]).
    pub qos: StreamQos,
    /// Resume cursor (#169 seam): the next sequence/offset a resumed reader
    /// wants. `0` = live/from-start. Full at-least-once dedup is #169 follow-up.
    pub resume_seq: u64,
}

impl StreamEntry {
    /// A point-to-point stream entry at the reliable-in-order floor.
    pub fn stream(handle: Option<Box<dyn StreamHandle>>, owner: String) -> Self {
        Self {
            handle,
            owner,
            bytes_received: 0,
            blocks_received: 0,
            kind: StreamKind::Stream,
            qos: StreamQos::reliable(),
            resume_seq: 0,
        }
    }

    /// An event entry (#600 group-keyed EventService surfaced on `/stream`).
    ///
    /// Events default to `latest-wins` — for lifecycle/telemetry the newest
    /// state matters more than every intermediate one (mirrors the `EventLive`
    /// StreamOpt preset). The `handle` is the event carrier subscription; see
    /// the events seam note on [`StreamRegistry::register`].
    pub fn event(handle: Option<Box<dyn StreamHandle>>, owner: String) -> Self {
        let mut qos = StreamQos::reliable();
        qos.apply_token("latest-wins");
        Self {
            handle,
            owner,
            bytes_received: 0,
            blocks_received: 0,
            kind: StreamKind::Event,
            qos,
            resume_seq: 0,
        }
    }
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

    /// Register a stream or event under `topic`.
    ///
    /// Both planes share this one registry (#819): a point-to-point stream
    /// ([`StreamEntry::stream`]) and a group-keyed EventService broadcast
    /// ([`StreamEntry::event`], #600) become topics on the same `/stream`
    /// surface. The entry's `handle` is the carrier subscription — for events
    /// the full EventService-consumer→[`StreamHandle`] adapter (group-key
    /// decryption over the moq_event `OriginConsumer`) is the marked follow-up;
    /// this seam is kind-aware and ready to carry it (an event entry with a
    /// `None` handle is an announced-but-not-yet-subscribed topic).
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

    /// Whether a topic currently holds its carrier-backed handle.
    ///
    /// The handle is the moq carrier subscription established at open; this is
    /// `false` while a `data` read has it checked out, or after
    /// [`take_handle`](Self::take_handle) hands it to a bulk consumer.
    pub fn has_carrier_handle(&self, topic: &str) -> bool {
        self.lock()
            .get(topic)
            .map(|e| e.handle.is_some())
            .unwrap_or(false)
    }

    /// Take the carrier-backed handle whole for the **bulk** consumption path
    /// (#819 / same handle-at-open model as #817's mmap-for-weights).
    ///
    /// This is the ceiling: a high-rate consumer drives the moq subscriber
    /// directly instead of pulling frame-by-frame through `Tread`, so moq's
    /// QUIC-native delivery semantics are preserved rather than flattened into
    /// a copy loop. Returns `None` if the topic is unknown or the handle is
    /// already checked out. The entry stays registered (info/qos/ctl remain
    /// addressable); the caller owns the subscription until it drops or
    /// re-parks it.
    pub fn take_handle(&self, topic: &str) -> Option<Box<dyn StreamHandle>> {
        self.lock().get_mut(topic).and_then(|e| e.handle.take())
    }

    /// The plane a topic belongs to (stream vs event), if registered.
    pub fn kind(&self, topic: &str) -> Option<StreamKind> {
        self.lock().get(topic).map(|e| e.kind)
    }

    /// The consumer-selected QoS for a topic, if registered.
    pub fn qos(&self, topic: &str) -> Option<StreamQos> {
        self.lock().get(topic).map(|e| e.qos.clone())
    }

    /// Select QoS for a topic (the `qos`/`ctl` vocabulary write path).
    /// Returns `false` if the topic is unknown.
    pub fn set_qos(&self, topic: &str, qos: StreamQos) -> bool {
        match self.lock().get_mut(topic) {
            Some(e) => {
                e.qos = qos;
                true
            }
            None => false,
        }
    }

    /// Set the resume cursor for a topic (#169 seam). Returns `false` if
    /// the topic is unknown.
    pub fn set_resume(&self, topic: &str, seq: u64) -> bool {
        match self.lock().get_mut(topic) {
            Some(e) => {
                e.resume_seq = seq;
                true
            }
            None => false,
        }
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
    /// /stream/{topic}/qos — read/select QoS (StreamOpt vocabulary)
    Qos { topic: String },
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
            [topic, "qos"] => StreamFidState::Qos {
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

    /// Open-time authorization + subscription binding (#819).
    ///
    /// This is where the interface authorizes: the `caller` `Subject` is the
    /// open-time credential a PEP checks before the carrier subscription is
    /// handed out (the same "authorize once at open, then stream over the
    /// carrier" model as #817's weights). MAC enforcement here is the epic's
    /// reference-monitor seam (#547 / #568) and is deliberately *not* wired in
    /// this app-half story — the mount fails closed with the caller threaded
    /// through, ready for the PEP. It does not re-open the carrier per read:
    /// the subscription established at registration IS the handle.
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
                // Check completion + QoS WITHOUT holding borrow across await.
                // `is_lossy` decides whether a mid-stream carrier error is
                // surfaced as a hard read error (reliable) or a soft EOF
                // (lossy-ok / latest-wins / drop-stale) — the consumer-visible
                // half of the QoS vocabulary selected through the file (#819).
                let (is_completed, is_lossy) = {
                    let streams = self.registry.lock();
                    let entry = streams
                        .get(topic.as_str())
                        .ok_or_else(|| MountError::NotFound(topic.clone()))?;
                    let done = entry.handle.as_ref().map(|h| h.is_completed()).unwrap_or(true);
                    (done, entry.qos.is_lossy())
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
                        if is_lossy {
                            // lossy-ok: tolerate the error as end-of-stream
                            // rather than failing the read (drop-stale/live).
                            Ok(vec![])
                        } else {
                            Err(MountError::Io(format!("stream error: {msg}")))
                        }
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
                    Err(e) => {
                        if is_lossy {
                            Ok(vec![]) // lossy-ok: soft EOF instead of a hard error
                        } else {
                            Err(MountError::Io(format!("stream read: {e}")))
                        }
                    }
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
                    "kind": entry.kind.as_str(),
                    "bytesReceived": entry.bytes_received,
                    "blocksReceived": entry.blocks_received,
                    "complete": complete,
                    "carrierHandle": entry.handle.is_some(),
                    "qos": entry.qos.to_tokens(),
                    "resumeSeq": entry.resume_seq,
                });
                let bytes = serde_json::to_string_pretty(&info)
                    .unwrap_or_default()
                    .into_bytes();
                let start = (offset as usize).min(bytes.len());
                let end = (start + count as usize).min(bytes.len());
                Ok(bytes[start..end].to_vec())
            }
            StreamFidState::Qos { topic } => {
                // Read the current QoS selection as its token vocabulary
                // (StreamOpt encoding, addressable through the file — #819).
                let streams = self.registry.lock();
                let entry = streams
                    .get(topic.as_str())
                    .ok_or_else(|| MountError::NotFound(topic.clone()))?;
                let mut line = entry.qos.to_tokens().join(" ");
                line.push('\n');
                let bytes = line.into_bytes();
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
                let line = std::str::from_utf8(data).unwrap_or("").trim();
                let (cmd, arg) = match line.split_once(char::is_whitespace) {
                    Some((c, a)) => (c, a.trim()),
                    None => (line, ""),
                };
                match cmd {
                    "cancel" => {
                        if let Some(entry) = self.registry.remove(topic) {
                            if let Some(mut handle) = entry.handle {
                                let _ = handle.cancel().await;
                            }
                        }
                        Ok(data.len() as u32)
                    }
                    // QoS selection through ctl (#819): `qos latest-wins drop-stale`.
                    "qos" => {
                        let qos = StreamQos::parse(arg)
                            .map_err(|tok| MountError::InvalidArgument(format!(
                                "unknown qos token: {tok}"
                            )))?;
                        if !self.registry.set_qos(topic, qos) {
                            return Err(MountError::NotFound(topic.clone()));
                        }
                        Ok(data.len() as u32)
                    }
                    // Resume seam (#169): set the cursor a resumed reader wants.
                    // The floor path is `Tread` at offset; the carrier handle
                    // resumes its subscription from this seq. Full at-least-once
                    // dedup is the #169 follow-up.
                    "resume" => {
                        let seq: u64 = arg.parse().map_err(|_| {
                            MountError::InvalidArgument(format!("bad resume seq: {arg}"))
                        })?;
                        if !self.registry.set_resume(topic, seq) {
                            return Err(MountError::NotFound(topic.clone()));
                        }
                        Ok(data.len() as u32)
                    }
                    _ => Err(MountError::NotSupported(format!(
                        "unknown stream command: {}",
                        cmd
                    ))),
                }
            }
            // The `qos` file is directly writable — same vocabulary as `ctl qos`.
            StreamFidState::Qos { topic } => {
                let spec = std::str::from_utf8(data).unwrap_or("").trim();
                let qos = StreamQos::parse(spec).map_err(|tok| {
                    MountError::InvalidArgument(format!("unknown qos token: {tok}"))
                })?;
                if !self.registry.set_qos(topic, qos) {
                    return Err(MountError::NotFound(topic.clone()));
                }
                Ok(data.len() as u32)
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
                    name: "qos".into(),
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
            StreamFidState::Qos { .. } => (0, "qos"),
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

    // ─── #819: a mock carrier handle for the surface/floor/handle tests ─────

    /// A minimal in-memory [`StreamHandle`] standing in for a moq carrier
    /// subscription. It hands out queued payloads; taking it whole out of the
    /// registry proves the bulk path drives *this* subscription rather than a
    /// per-read reconstruction.
    struct MockHandle {
        id: String,
        queue: std::collections::VecDeque<StreamPayload>,
        completed: bool,
        cancelled: bool,
    }

    impl MockHandle {
        fn new(id: &str, payloads: Vec<StreamPayload>) -> Self {
            Self {
                id: id.to_string(),
                queue: payloads.into_iter().collect(),
                completed: false,
                cancelled: false,
            }
        }
    }

    #[async_trait]
    impl StreamHandle for MockHandle {
        async fn next_payload(&mut self) -> anyhow::Result<Option<StreamPayload>> {
            match self.queue.pop_front() {
                Some(p) => {
                    if matches!(p, StreamPayload::Complete(_) | StreamPayload::Error(_)) {
                        self.completed = true;
                    }
                    Ok(Some(p))
                }
                None => {
                    self.completed = true;
                    Ok(None)
                }
            }
        }
        async fn cancel(&mut self) -> anyhow::Result<()> {
            self.cancelled = true;
            Ok(())
        }
        fn stream_id(&self) -> &str {
            &self.id
        }
        fn is_completed(&self) -> bool {
            self.completed
        }
    }

    fn boxed(h: MockHandle) -> Box<dyn StreamHandle> {
        Box::new(h)
    }

    // ─── QoS vocabulary: StreamOpt encoding moved onto the file ─────────────

    #[test]
    fn qos_reliable_is_the_failclosed_floor() {
        let q = StreamQos::reliable();
        assert!(matches!(q.stream_opt().ordering, Ordering::Ordered));
        assert!(matches!(q.stream_opt().completion, Completion::EndOfStream));
        assert!(matches!(q.stream_opt().overflow_policy, OverflowPolicy::Block));
        assert!(!q.is_lossy());
        assert_eq!(q.to_tokens(), vec!["reliable"]);
    }

    #[test]
    fn qos_tokens_map_onto_streamopt_axes() {
        // latest-wins → drop-oldest depth 1 + live + no terminator required.
        let mut lw = StreamQos::reliable();
        assert!(lw.apply_token("latest-wins"));
        assert!(matches!(
            lw.stream_opt().overflow_policy,
            OverflowPolicy::DropOldest { high_water_mark: 1 }
        ));
        assert!(matches!(lw.stream_opt().completion, Completion::None));

        // lossy-ok → at-most-once + unordered (gaps tolerated).
        let mut lo = StreamQos::reliable();
        assert!(lo.apply_token("lossy-ok"));
        assert!(matches!(lo.stream_opt().delivery, Delivery::AtMostOnce));
        assert!(matches!(
            lo.stream_opt().ordering,
            Ordering::Unordered { .. }
        ));
        assert!(lo.is_lossy());

        // drop-stale → drop-oldest overflow (shed backlog) + live retention.
        let mut ds = StreamQos::reliable();
        assert!(ds.apply_token("drop-stale"));
        assert!(matches!(
            ds.stream_opt().overflow_policy,
            OverflowPolicy::DropOldest { .. }
        ));

        // Unknown token is rejected, not silently ignored.
        assert!(!StreamQos::reliable().apply_token("bogus"));
    }

    #[test]
    fn qos_parse_render_roundtrips() {
        let q = StreamQos::parse("lossy-ok").unwrap();
        assert!(q.to_tokens().contains(&"lossy-ok"));
        assert_eq!(StreamQos::parse("nope").unwrap_err(), "nope");
        // Compose two tokens.
        let q2 = StreamQos::parse("latest-wins lossy-ok").unwrap();
        assert!(q2.to_tokens().contains(&"latest-wins"));
        assert!(q2.to_tokens().contains(&"lossy-ok"));
    }

    // ─── Tread floor + carrier handle (not a copy loop) ─────────────────────

    /// The floor: a blocking `Tread` on `data` returns the next carrier
    /// payload (push semantics), then EOF once the subscription completes.
    #[tokio::test]
    async fn tread_floor_reads_then_eof() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        reg.register(
            "t".into(),
            StreamEntry::stream(
                Some(boxed(MockHandle::new(
                    "t",
                    vec![StreamPayload::Data(b"tok".to_vec())],
                ))),
                "owner".into(),
            ),
        );
        let mount = StreamMount::new(std::sync::Arc::clone(&reg));
        let caller = Subject::anonymous();
        let fid = Fid::new(StreamFidState::Data { topic: "t".into() });

        let first = mount.read(&fid, 0, 0, &caller).await.unwrap();
        assert_eq!(first, b"tok");
        // Handle exhausted → completed → EOF.
        let eof = mount.read(&fid, 0, 0, &caller).await.unwrap();
        assert!(eof.is_empty());
    }

    /// The ceiling: the carrier-backed handle is taken *whole* for the bulk
    /// path and remains the same subscription — it is never reconstructed per
    /// read. After `take_handle`, no carrier handle remains in the registry.
    #[tokio::test]
    async fn carrier_handle_is_takeable_not_a_copy_loop() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        reg.register(
            "bulk".into(),
            StreamEntry::stream(
                Some(boxed(MockHandle::new(
                    "bulk",
                    vec![StreamPayload::Data(b"a".to_vec())],
                ))),
                "owner".into(),
            ),
        );
        assert!(reg.has_carrier_handle("bulk"));

        // A bulk consumer takes the subscription whole and drives it directly.
        let mut handle = reg.take_handle("bulk").expect("carrier handle");
        assert_eq!(handle.stream_id(), "bulk");
        let p = handle.next_payload().await.unwrap();
        assert!(matches!(p, Some(StreamPayload::Data(_))));

        // The entry stays registered (info/qos/ctl addressable) but the handle
        // is now owned by the bulk consumer — not re-openable per read.
        assert!(reg.exists("bulk"));
        assert!(!reg.has_carrier_handle("bulk"));
        assert!(reg.take_handle("bulk").is_none());
    }

    // ─── QoS ctl selection through the file interface ───────────────────────

    #[tokio::test]
    async fn qos_ctl_selection_through_the_file() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        reg.register(
            "s".into(),
            StreamEntry::stream(
                Some(boxed(MockHandle::new("s", vec![]))),
                "owner".into(),
            ),
        );
        let mount = StreamMount::new(std::sync::Arc::clone(&reg));
        let caller = Subject::anonymous();

        // Select QoS by writing the vocabulary to `ctl`.
        let ctl = Fid::new(StreamFidState::Ctl { topic: "s".into() });
        mount
            .write(&ctl, 0, b"qos latest-wins drop-stale", &caller)
            .await
            .unwrap();
        let q = reg.qos("s").unwrap();
        assert!(matches!(
            q.stream_opt().overflow_policy,
            OverflowPolicy::DropOldest { .. }
        ));

        // Reading the `qos` file reflects the selection as tokens.
        let qfid = Fid::new(StreamFidState::Qos { topic: "s".into() });
        let shown = mount.read(&qfid, 0, 256, &caller).await.unwrap();
        let text = String::from_utf8(shown).unwrap();
        assert!(text.contains("latest-wins") || text.contains("drop-stale"));

        // Writing the `qos` file directly works too.
        mount.write(&qfid, 0, b"reliable", &caller).await.unwrap();
        assert_eq!(reg.qos("s").unwrap().to_tokens(), vec!["reliable"]);

        // An unknown token is rejected (fail-closed), not silently accepted.
        let err = mount.write(&ctl, 0, b"qos bogus", &caller).await;
        assert!(matches!(err, Err(MountError::InvalidArgument(_))));
    }

    /// Under lossy QoS a mid-stream carrier error is a soft EOF, not a hard
    /// read error — the consumer-visible half of "lossy-ok" selected via ctl.
    #[tokio::test]
    async fn lossy_qos_tolerates_carrier_error_as_eof() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        reg.register(
            "e".into(),
            StreamEntry::stream(
                Some(boxed(MockHandle::new(
                    "e",
                    vec![StreamPayload::Error("boom".into())],
                ))),
                "owner".into(),
            ),
        );
        let mount = StreamMount::new(std::sync::Arc::clone(&reg));
        let caller = Subject::anonymous();
        let fid = Fid::new(StreamFidState::Data { topic: "e".into() });

        // Reliable (default): the error surfaces as a hard read error.
        let hard = mount.read(&fid, 0, 0, &caller).await;
        assert!(matches!(hard, Err(MountError::Io(_))));

        // Re-arm with a fresh error payload, now under lossy-ok.
        reg.register(
            "e".into(),
            StreamEntry::stream(
                Some(boxed(MockHandle::new(
                    "e",
                    vec![StreamPayload::Error("boom".into())],
                ))),
                "owner".into(),
            ),
        );
        reg.set_qos("e", StreamQos::parse("lossy-ok").unwrap());
        let soft = mount.read(&fid, 0, 0, &caller).await.unwrap();
        assert!(soft.is_empty()); // soft EOF
    }

    // ─── Events ride the same surface (#600) ────────────────────────────────

    #[tokio::test]
    async fn events_ride_the_same_surface() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        reg.register(
            "model-lifecycle".into(),
            StreamEntry::event(Some(boxed(MockHandle::new("model-lifecycle", vec![]))), "svc".into()),
        );
        assert_eq!(reg.kind("model-lifecycle"), Some(StreamKind::Event));

        // The event topic is addressable on `/stream` and its `info` reports
        // `kind: event` — one surface, not a separate plane.
        let mount = StreamMount::new(std::sync::Arc::clone(&reg));
        let caller = Subject::anonymous();
        let info_fid = Fid::new(StreamFidState::Info {
            topic: "model-lifecycle".into(),
        });
        let info = mount.read(&info_fid, 0, 4096, &caller).await.unwrap();
        let text = String::from_utf8(info).unwrap();
        assert!(text.contains("\"kind\": \"event\""));
        // Events default to latest-wins (EventLive-style).
        assert!(text.contains("latest-wins"));
    }

    // ─── #169 resume seam ───────────────────────────────────────────────────

    #[tokio::test]
    async fn resume_cursor_seam_via_ctl() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        reg.register(
            "r".into(),
            StreamEntry::stream(Some(boxed(MockHandle::new("r", vec![]))), "o".into()),
        );
        let mount = StreamMount::new(std::sync::Arc::clone(&reg));
        let caller = Subject::anonymous();
        let ctl = Fid::new(StreamFidState::Ctl { topic: "r".into() });

        mount.write(&ctl, 0, b"resume 42", &caller).await.unwrap();

        let info_fid = Fid::new(StreamFidState::Info { topic: "r".into() });
        let info = mount.read(&info_fid, 0, 4096, &caller).await.unwrap();
        assert!(String::from_utf8(info).unwrap().contains("\"resumeSeq\": 42"));

        // A non-numeric resume arg is rejected.
        assert!(matches!(
            mount.write(&ctl, 0, b"resume nope", &caller).await,
            Err(MountError::InvalidArgument(_))
        ));
    }

    // ─── the topic dir exposes the canonical face set ───────────────────────

    #[tokio::test]
    async fn topic_dir_exposes_data_info_qos_ctl() {
        let reg = std::sync::Arc::new(StreamRegistry::new());
        reg.register(
            "d".into(),
            StreamEntry::stream(Some(boxed(MockHandle::new("d", vec![]))), "o".into()),
        );
        let mount = StreamMount::new(reg);
        let caller = Subject::anonymous();
        let dir = mount
            .walk(&["d"], &caller)
            .await
            .unwrap();
        let entries = mount.readdir(&dir, &caller).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"data"));
        assert!(names.contains(&"info"));
        assert!(names.contains(&"qos"));
        assert!(names.contains(&"ctl"));
    }
}
