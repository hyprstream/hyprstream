//! Cross-target streaming consumer types.
//!
//! Extracted from `streaming.rs` (native-only) so that `StreamHandleImpl<T>`
//! can compile on both native and wasm32. All types here are pure logic —
//! no ZMQ, no Tokio, no platform-specific deps.
//!
//! - `StreamPayload` — parsed output from verified stream blocks
//! - `StreamVerifier` — HMAC chain verifier (pure crypto)
//! - `StreamHandleImpl<T>` — unified stream consumer over any `Transport`

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use futures::StreamExt;

use crate::crypto::{derive_stream_keys, keyed_mac_truncated};
use crate::stream_info::StreamInfo;
use crate::streaming_capnp;
use crate::transport_traits::{PublishSink, Transport};

#[cfg(not(feature = "fips"))]
use crate::crypto::ristretto_dh_raw as dh_compute_raw;

// TODO: Add FIPS p256_dh_raw equivalent when needed
// #[cfg(feature = "fips")]
// use crate::crypto::p256_dh_raw as dh_compute_raw;

// ============================================================================
// StreamPayload — parsed output from verified stream blocks
// ============================================================================

/// Output payload from StreamVerifier.
#[derive(Clone, Debug)]
pub enum StreamPayload {
    /// Generic binary data (tokens, I/O, etc.)
    Data(Vec<u8>),
    /// Error during streaming
    Error(String),
    /// Completion with app-specific metadata
    Complete(Vec<u8>),
    /// Encrypted tagged payload with key commitment
    Tagged {
        tag: Vec<u8>,
        payload: Vec<u8>,
        nonce: Vec<u8>,
        key_commitment: Vec<u8>,
    },
}

// ============================================================================
// StreamVerifier — HMAC chain verifier (pure crypto)
// ============================================================================

/// HMAC chain verifier for StreamBlock.
pub struct StreamVerifier {
    key: [u8; 32],
    prev_mac: Option<[u8; 16]>,
    topic: String,
}

impl StreamVerifier {
    /// Create a new verifier.
    pub fn new(key: [u8; 32], topic: String) -> Self {
        Self {
            key,
            prev_mac: None,
            topic,
        }
    }

    /// Verify frames and return parsed payloads.
    ///
    /// Expected frames: [topic, capnp StreamBlock, 16-byte MAC]
    pub fn verify(&mut self, frames: &[Vec<u8>]) -> Result<Vec<StreamPayload>> {
        if frames.len() != 3 {
            anyhow::bail!("Expected 3 frames, got {}", frames.len());
        }

        let received_topic = &frames[0];
        let capnp_data = &frames[1];
        let received_mac = &frames[2];

        if received_mac.len() != 16 {
            anyhow::bail!("Expected 16-byte MAC, got {}", received_mac.len());
        }

        if received_topic != self.topic.as_bytes() {
            anyhow::bail!("Topic mismatch");
        }

        // Compute expected MAC
        let mut input = Vec::with_capacity(64 + capnp_data.len());
        match &self.prev_mac {
            None => input.extend_from_slice(self.topic.as_bytes()),
            Some(prev) => input.extend_from_slice(prev),
        }
        input.extend_from_slice(capnp_data);

        let expected_mac = keyed_mac_truncated(&self.key, &input);

        if !constant_time_eq(received_mac, &expected_mac) {
            anyhow::bail!("MAC verification failed");
        }

        // Update chain state
        let mut new_prev = [0u8; 16];
        new_prev.copy_from_slice(received_mac);
        self.prev_mac = Some(new_prev);

        // Parse StreamBlock
        let reader = capnp::serialize::read_message(
            &mut std::io::Cursor::new(capnp_data),
            capnp::message::ReaderOptions::default(),
        )?;
        let block = reader.get_root::<streaming_capnp::stream_block::Reader>()?;

        let payloads_reader = block.get_payloads()?;
        let mut payloads = Vec::with_capacity(payloads_reader.len() as usize);

        for i in 0..payloads_reader.len() {
            let p = payloads_reader.get(i);

            use streaming_capnp::stream_payload::Which;
            let payload = match p.which()? {
                Which::Data(data_result) => {
                    StreamPayload::Data(data_result?.to_vec())
                }
                Which::Error(err_result) => {
                    let err = err_result?;
                    StreamPayload::Error(err.get_message()?.to_string()?)
                }
                Which::Complete(complete_result) => {
                    StreamPayload::Complete(complete_result?.to_vec())
                }
                Which::Heartbeat(()) => {
                    continue;
                }
                Which::Tagged(tagged_result) => {
                    let tagged = tagged_result?;
                    StreamPayload::Tagged {
                        tag: tagged.get_tag()?.to_vec(),
                        payload: tagged.get_payload()?.to_vec(),
                        nonce: tagged.get_nonce()?.to_vec(),
                        key_commitment: tagged.get_key_commitment()?.to_vec(),
                    }
                }
            };

            payloads.push(payload);
        }

        Ok(payloads)
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        &self.topic
    }
}

/// Constant-time byte slice comparison.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ============================================================================
// StreamHandleImpl<T> — unified stream consumer over any Transport
// ============================================================================

/// E2E authenticated stream consumer, generic over transport.
///
/// Same struct on native (`StreamHandle<ZmqConnection>`) and WASM
/// (`StreamHandle<WtConnection>`). ECDH, key derivation, MAC verification,
/// and Cap'n Proto parsing are identical — only the wire transport differs.
pub struct StreamHandleImpl<T: Transport> {
    subscriber: T::Sub,
    publisher: Option<T::Pub>,
    stream_id: String,
    topic: String,
    verifier: StreamVerifier,
    pending: VecDeque<StreamPayload>,
    completed: bool,
    cancelled: Arc<AtomicBool>,
    ctrl_topic: String,
    ctrl_mac_key: [u8; 32],
}

impl<T: Transport> StreamHandleImpl<T> {
    /// Full streaming setup: ECDH → derive keys → subscribe → open ctrl → create verifier.
    ///
    /// Called by `RpcClient::open_stream()` after the streaming RPC returns `StreamInfo`.
    pub async fn open(
        transport: &T,
        stream_info: StreamInfo,
        client_secret: &[u8; 32],
        client_pubkey: &[u8; 32],
    ) -> Result<Self> {
        // Pure crypto — same code, both targets
        let shared_secret = dh_compute_raw(client_secret, &stream_info.server_pubkey)?;
        let keys = derive_stream_keys(&shared_secret, client_pubkey, &stream_info.server_pubkey)?;

        // Transport-abstracted — ZMQ or WebTransport
        let subscriber = transport.subscribe(keys.topic.as_bytes()).await?;
        let publisher = transport.publish(keys.ctrl_topic.as_bytes()).await.ok();

        // Pure crypto
        let verifier = StreamVerifier::new(*keys.mac_key, keys.topic.clone());

        Ok(Self {
            subscriber,
            publisher,
            stream_id: stream_info.stream_id,
            topic: keys.topic,
            verifier,
            pending: VecDeque::new(),
            completed: false,
            cancelled: Arc::new(AtomicBool::new(false)),
            ctrl_topic: keys.ctrl_topic,
            ctrl_mac_key: *keys.ctrl_mac_key,
        })
    }

    /// Get next verified payload. Returns None on stream end.
    pub async fn next_payload(&mut self) -> Result<Option<StreamPayload>> {
        // Drain buffered payloads first
        if let Some(p) = self.pending.pop_front() {
            return Ok(Some(p));
        }
        if self.completed || self.cancelled.load(Ordering::Acquire) {
            return Ok(None);
        }

        // Poll subscriber (T::Sub implements futures::Stream)
        let frames = match self.subscriber.next().await {
            Some(Ok(frames)) => frames,
            Some(Err(e)) => return Err(e),
            None => {
                self.completed = true;
                return Ok(None);
            }
        };

        // Verify MAC + parse Cap'n Proto (pure crypto, cross-target)
        let payloads = self.verifier.verify(&frames)?;
        for p in payloads {
            if matches!(p, StreamPayload::Complete(..) | StreamPayload::Error(..)) {
                self.completed = true;
            }
            self.pending.push_back(p);
        }

        Ok(self.pending.pop_front())
    }

    /// Cancel the stream via ctrl channel.
    /// Sets local cancelled flag AND sends cancel message to producer.
    pub async fn cancel(&self) -> Result<()> {
        self.cancelled.store(true, Ordering::Release);
        if let Some(ref pub_handle) = self.publisher {
            let msg = build_stream_control_cancel();
            let mac = keyed_mac_truncated(&self.ctrl_mac_key, &msg);
            pub_handle.send_frames(&[
                self.ctrl_topic.as_bytes(),
                &msg,
                &mac,
            ]).await?;
        }
        Ok(())
    }

    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }
}

/// Build a StreamControl::Cancel capnp message.
fn build_stream_control_cancel() -> Vec<u8> {
    let mut builder = capnp::message::Builder::new_default();
    {
        let mut ctrl = builder.init_root::<streaming_capnp::stream_control::Builder>();
        ctrl.set_cancel(());
    }
    let mut buf = Vec::new();
    capnp::serialize::write_message(&mut buf, &builder)
        .expect("StreamControl serialization cannot fail");
    buf
}

// ============================================================================
// StreamHandle — object-safe trait for generated portable clients
// ============================================================================

/// Object-safe stream handle for dynamic dispatch.
///
/// Generated portable clients use `Box<dyn StreamHandle>` so they work
/// with any `StreamHandleImpl<T>` regardless of concrete transport.
#[async_trait::async_trait(?Send)]
pub trait StreamHandle: Send {
    /// Get next verified payload. Returns None on stream end.
    async fn next_payload(&mut self) -> Result<Option<StreamPayload>>;

    /// Cancel the stream via ctrl channel.
    async fn cancel(&self) -> Result<()>;

    /// Get the stream ID.
    fn stream_id(&self) -> &str;

    /// Check if stream is completed.
    fn is_completed(&self) -> bool;
}

/// Blanket impl: any `StreamHandleImpl<T>` satisfies `StreamHandle`.
#[async_trait::async_trait(?Send)]
impl<T: Transport> StreamHandle for StreamHandleImpl<T> {
    async fn next_payload(&mut self) -> Result<Option<StreamPayload>> {
        StreamHandleImpl::next_payload(self).await
    }

    async fn cancel(&self) -> Result<()> {
        StreamHandleImpl::cancel(self).await
    }

    fn stream_id(&self) -> &str {
        StreamHandleImpl::stream_id(self)
    }

    fn is_completed(&self) -> bool {
        StreamHandleImpl::is_completed(self)
    }
}

/// Blanket impl for Box<dyn StreamHandle> so callers can use it directly.
#[async_trait::async_trait(?Send)]
impl StreamHandle for Box<dyn StreamHandle> {
    async fn next_payload(&mut self) -> Result<Option<StreamPayload>> {
        (**self).next_payload().await
    }

    async fn cancel(&self) -> Result<()> {
        (**self).cancel().await
    }

    fn stream_id(&self) -> &str {
        (**self).stream_id()
    }

    fn is_completed(&self) -> bool {
        (**self).is_completed()
    }
}
