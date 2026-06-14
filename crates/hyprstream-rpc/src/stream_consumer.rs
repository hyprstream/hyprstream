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

/// Consumer-side ordering contract (#163/#213). The full `StreamPolicy` (schema) carries
/// producer-side QoS too (delivery/retention/overflow); the consumer only needs ordering
/// + completion to *verify*. Cross-target so native and the wasm/browser consumer share it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamOrdering {
    /// In-order, gap-fatal: each block's `sequenceNumber` must equal the previous + 1.
    Ordered,
    /// Out-of-order media: gaps tolerated (skip-to-live); a block whose `sequenceNumber` is
    /// at/under `last_seen - anti_replay_window` is rejected as a replay.
    Unordered { anti_replay_window: u32 },
}

/// Consumer-side truncation contract (#163/#213). Maps to the schema `Completion` axis
/// (schema `none` → `Open`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completion {
    /// A `Complete`/`Error` payload must be observed before EOF; EOF-without-terminal is a
    /// truncation and must be rejected by the consumer (see [`StreamVerifier::requires_terminal`]).
    /// Matches schema `endOfStream` (gRPC END_STREAM / HTTP/2 DATA+END_STREAM).
    EndOfStream,
    /// EOF is accepted; truncation is not detectable (the explicit choice for inference/live).
    Open,
}

/// The consumer's slice of the stream's API contract (#213), set from the service's
/// `$streamPolicy` via codegen (#217) — NOT from the wire, so it can't be downgraded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPolicy {
    pub ordering: StreamOrdering,
    pub completion: Completion,
}

/// HMAC chain verifier for StreamBlock.
pub struct StreamVerifier {
    key: [u8; 32],
    prev_mac: Option<[u8; 16]>,
    topic: String,
    /// Policy-selected enforcement (#163). `None` ⇒ legacy behaviour (MAC chain only, no
    /// `seq`/completion checks). Set via [`StreamVerifier::with_policy`] by codegen-generated
    /// consumers; default `new()` keeps `None` so existing call sites are unchanged.
    policy: Option<StreamPolicy>,
    /// Next expected `seq` (ordered) / highest `seq` seen (media). `None` until the first
    /// block, so a late-join starting at an arbitrary offset is accepted, then enforced.
    seq_cursor: Option<u64>,
    /// Whether a terminal (`Complete`/`Error`) payload has been observed (for `completion`).
    terminal_seen: bool,
}

impl StreamVerifier {
    /// Create a new verifier with **no** policy enforcement (MAC chain only) — legacy behaviour.
    pub fn new(key: [u8; 32], topic: String) -> Self {
        Self {
            key,
            prev_mac: None,
            topic,
            policy: None,
            seq_cursor: None,
            terminal_seen: false,
        }
    }

    /// Create a verifier that enforces `policy` (#163) — used by codegen-generated consumers,
    /// where the policy is a compile-time constant from the service's API contract.
    pub fn with_policy(key: [u8; 32], topic: String, policy: StreamPolicy) -> Self {
        let mut v = Self::new(key, topic);
        v.policy = Some(policy);
        v
    }

    /// True if this stream's policy requires a terminal payload before EOF; the consumer must
    /// reject an EOF when this is true and [`StreamVerifier::terminal_seen`] is false.
    pub fn requires_terminal(&self) -> bool {
        matches!(self.policy.map(|p| p.completion), Some(Completion::EndOfStream))
    }

    /// True once a terminal (`Complete`/`Error`) payload has been observed.
    pub fn terminal_seen(&self) -> bool {
        self.terminal_seen
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

        // Policy-selected ordering/replay enforcement (#163). The MAC already authenticates
        // `sequenceNumber` (it's inside the block, #219), so a tampered sequenceNumber fails
        // MAC above; here we enforce *position*. `None` policy ⇒ legacy behaviour (MAC chain
        // only, no sequenceNumber/completion checks).
        if let Some(policy) = self.policy {
            let sequence_number = block.get_sequence_number();
            match policy.ordering {
                StreamOrdering::Ordered => {
                    if let Some(expected) = self.seq_cursor {
                        if sequence_number != expected {
                            anyhow::bail!(
                                "stream ordering violation: expected sequenceNumber {expected}, \
                                 got {sequence_number} (gap/reorder on an ordered stream)"
                            );
                        }
                    }
                    // First block (late-join): accept its sequenceNumber, then enforce contiguity.
                    self.seq_cursor = Some(sequence_number.saturating_add(1));
                }
                StreamOrdering::Unordered { .. } => {
                    // Media (out-of-order) needs a per-Group *self-authenticating* MAC, not the
                    // chained prev_mac this verifier uses — the chain assumes contiguous
                    // delivery, which moq's out-of-order/eviction model breaks. Fail closed
                    // until the per-Group MAC scheme lands (producer + consumer); see #163.
                    anyhow::bail!(
                        "unordered/media stream policy not yet supported: needs the per-Group \
                         self-authenticating MAC (the chained MAC assumes in-order delivery) — #163 follow-on"
                    );
                }
            }
        }

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

            // Track terminal observation for the `completion` axis (#163).
            if matches!(payload, StreamPayload::Complete(_) | StreamPayload::Error(_)) {
                self.terminal_seen = true;
            }
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
#[allow(clippy::expect_used)]
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod policy_tests {
    use super::*;

    /// Build one `[topic, capnp StreamBlock, mac]` frame, chaining the MAC exactly as the
    /// producer does (`prev` ‖ capnp, or topic for the first block).
    fn frame(
        key: &[u8; 32],
        topic: &str,
        prev: Option<[u8; 16]>,
        sequence_number: u64,
        terminal: bool,
    ) -> (Vec<Vec<u8>>, [u8; 16]) {
        let mut msg = capnp::message::Builder::new_default();
        {
            let mut b = msg.init_root::<streaming_capnp::stream_block::Builder>();
            let pm: Vec<u8> = match prev {
                Some(p) => p.to_vec(),
                None => topic.as_bytes().iter().take(16).copied().collect(),
            };
            b.set_prev_mac(&pm);
            b.set_sequence_number(sequence_number);
            b.set_epoch(0);
            let mut list = b.init_payloads(1);
            let mut p = list.reborrow().get(0);
            if terminal {
                p.set_complete(b"done");
            } else {
                p.set_data(b"x");
            }
        }
        let mut capnp_bytes = Vec::new();
        capnp::serialize::write_message(&mut capnp_bytes, &msg).unwrap();
        let mut input = Vec::new();
        match prev {
            None => input.extend_from_slice(topic.as_bytes()),
            Some(p) => input.extend_from_slice(&p),
        }
        input.extend_from_slice(&capnp_bytes);
        let mac = keyed_mac_truncated(key, &input);
        (vec![topic.as_bytes().to_vec(), capnp_bytes, mac.to_vec()], mac)
    }

    #[test]
    fn ordered_accepts_contiguous_and_rejects_gap() {
        let key = [7u8; 32];
        let topic = "tok";
        let mut v = StreamVerifier::with_policy(
            key,
            topic.to_owned(),
            StreamPolicy { ordering: StreamOrdering::Ordered, completion: Completion::Open },
        );
        let (f5, m5) = frame(&key, topic, None, 5, false); // late-join at seq 5
        v.verify(&f5).unwrap();
        let (f6, m6) = frame(&key, topic, Some(m5), 6, false);
        v.verify(&f6).unwrap();
        let (f9, _) = frame(&key, topic, Some(m6), 9, false); // gap: expected 7
        let err = v.verify(&f9).unwrap_err().to_string();
        assert!(err.contains("ordering violation"), "got: {err}");
    }

    #[test]
    fn no_policy_keeps_legacy_behaviour() {
        let key = [3u8; 32];
        let topic = "leg";
        let mut v = StreamVerifier::new(key, topic.to_owned()); // no policy
        let (f5, m5) = frame(&key, topic, None, 5, false);
        v.verify(&f5).unwrap();
        // Big seq jump is accepted — no enforcement without a policy (unchanged behaviour).
        let (f9, _) = frame(&key, topic, Some(m5), 9, false);
        v.verify(&f9).unwrap();
    }

    #[test]
    fn unordered_rejected_until_per_group_mac() {
        let key = [1u8; 32];
        let topic = "med";
        let mut v = StreamVerifier::with_policy(
            key,
            topic.to_owned(),
            StreamPolicy {
                ordering: StreamOrdering::Unordered { anti_replay_window: 4 },
                completion: Completion::Open,
            },
        );
        let (f0, _) = frame(&key, topic, None, 0, false);
        assert!(v.verify(&f0).is_err());
    }

    #[test]
    fn completion_tracks_terminal() {
        let key = [5u8; 32];
        let topic = "fin";
        let mut v = StreamVerifier::with_policy(
            key,
            topic.to_owned(),
            StreamPolicy { ordering: StreamOrdering::Ordered, completion: Completion::EndOfStream },
        );
        assert!(v.requires_terminal());
        assert!(!v.terminal_seen());
        let (f0, m0) = frame(&key, topic, None, 0, false);
        v.verify(&f0).unwrap();
        assert!(!v.terminal_seen());
        let (f1, _) = frame(&key, topic, Some(m0), 1, true); // terminal payload
        v.verify(&f1).unwrap();
        assert!(v.terminal_seen());
    }
}
