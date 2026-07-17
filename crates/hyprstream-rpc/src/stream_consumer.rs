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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{ensure, Result};
use futures::StreamExt;

use crate::crypto::{derive_stream_keys, keyed_mac_truncated, keyed_mac_truncated_parts};
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

/// Consumer-side ordering contract (#163/#213). The full `StreamOpt` (schema) carries
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
/// `$streamQos` via codegen (#217) — NOT from the wire, so it can't be downgraded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifierContract {
    pub ordering: StreamOrdering,
    pub completion: Completion,
}

/// HMAC chain verifier for StreamBlock.
pub struct StreamVerifier {
    key: [u8; 32],
    /// Transport-level AEAD key (#321). `Some` ⇒ a `Tagged` payload is opened
    /// (AES-256-GCM, AAD bound to topic+epoch) back into Data/Complete; `None` ⇒
    /// `Tagged` payloads pass through unchanged (the E2E notification path, which
    /// the app layer decrypts itself).
    enc_key: Option<[u8; 32]>,
    prev_mac: Option<[u8; 16]>,
    topic: String,
    /// Verifier contract enforcement (#163). `None` ⇒ legacy behaviour (MAC chain only, no
    /// `seq`/completion checks). Set via [`StreamVerifier::with_policy`] by codegen-generated
    /// consumers; default `new()` keeps `None` so existing call sites are unchanged.
    policy: Option<VerifierContract>,
    /// Next expected `seq` (ordered) / highest `seq` seen (media). `None` until the first
    /// block, so a late-join starting at an arbitrary offset is accepted, then enforced.
    seq_cursor: Option<u64>,
    /// Whether a terminal (`Complete`/`Error`) payload has been observed (for `completion`).
    terminal_seen: bool,
    /// Explicit identified epoch state.  Epoch commits are accepted separately
    /// and reset the chain atomically before any next-epoch Object is processed.
    epoch_ratchet: Option<crate::stream_epoch::StreamEpochRatchet>,
}

impl StreamVerifier {
    /// Create a new verifier with **no** policy enforcement (MAC chain only) — legacy behaviour.
    ///
    /// AEAD opening is OFF (`enc_key = None`); use [`StreamVerifier::with_enc_key`] to
    /// open transport-sealed `Tagged` payloads (#321).
    pub fn new(key: [u8; 32], topic: String) -> Self {
        Self {
            key,
            enc_key: None,
            prev_mac: None,
            topic,
            policy: None,
            seq_cursor: None,
            terminal_seen: false,
            epoch_ratchet: None,
        }
    }

    /// Set the transport AEAD key so sealed `Tagged` payloads are opened (#321).
    ///
    /// Builder-style; pass the `enc_key` from `derive_client_stream_keys`. With it
    /// set, the verifier decrypts each `Tagged` block back into Data/Complete and
    /// fails closed on tamper / wrong key.
    pub fn with_enc_key(mut self, enc_key: [u8; 32]) -> Self {
        self.enc_key = Some(enc_key);
        self
    }

    /// Install the identified epoch profile at its current committed epoch.
    pub fn with_epoch_ratchet(mut self, ratchet: crate::stream_epoch::StreamEpochRatchet) -> Self {
        let keys = ratchet.current_keys();
        self.key = *keys.producer_to_consumer.mac_key;
        self.enc_key = Some(*keys.producer_to_consumer.enc_key);
        self.topic = ratchet.route_topic().to_owned();
        self.prev_mac = None;
        self.seq_cursor = None;
        self.epoch_ratchet = Some(ratchet);
        self
    }

    /// Authenticate and atomically install the next epoch.  Failed, replayed,
    /// skipped, or cross-stream commits leave the current verifier state intact.
    pub fn accept_epoch_commit(
        &mut self,
        commit: &crate::stream_epoch::StreamEpochCommit,
    ) -> Result<()> {
        let ratchet = self
            .epoch_ratchet
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("stream is not using the identified epoch profile"))?;
        let mut pending = ratchet.clone();
        let keys = pending.accept_commit(commit)?;
        self.key = *keys.producer_to_consumer.mac_key;
        self.enc_key = Some(*keys.producer_to_consumer.enc_key);
        self.prev_mac = None;
        self.seq_cursor = None;
        self.epoch_ratchet = Some(pending);
        Ok(())
    }

    /// Create a verifier that enforces `policy` (#163) — used by codegen-generated consumers,
    /// where the contract is a compile-time constant from the service's API contract.
    pub fn with_policy(key: [u8; 32], topic: String, policy: VerifierContract) -> Self {
        let mut v = Self::new(key, topic);
        v.policy = Some(policy);
        v
    }

    /// True if this stream's policy requires a terminal payload before EOF; the consumer must
    /// reject an EOF when this is true and [`StreamVerifier::terminal_seen`] is false.
    pub fn requires_terminal(&self) -> bool {
        matches!(
            self.policy.map(|p| p.completion),
            Some(Completion::EndOfStream)
        )
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
        self.verify_parts(&frames[0], &frames[1], &frames[2])
    }

    /// Zero-copy variant of [`verify`](Self::verify) over borrowed slices.
    ///
    /// `verify` is a thin wrapper over this. The moq path calls it directly with
    /// slices into the received `Bytes` frame (`verify_moq_frame`), so a large
    /// payload (e.g. a pipeline activation tensor) is never copied into owned
    /// per-frame `Vec`s — combined with the multi-part MAC and
    /// `read_message_from_flat_slice` below, the block stays a view into the
    /// receive buffer end-to-end.
    pub fn verify_parts(
        &mut self,
        received_topic: &[u8],
        capnp_data: &[u8],
        received_mac: &[u8],
    ) -> Result<Vec<StreamPayload>> {
        if received_mac.len() != 16 {
            anyhow::bail!("Expected 16-byte MAC, got {}", received_mac.len());
        }

        if received_topic != self.topic.as_bytes() {
            anyhow::bail!("Topic mismatch");
        }

        // Compute the expected MAC over `prev_mac ‖ capnp_data` (or `topic ‖ capnp_data`
        // for the first block) WITHOUT allocating a joined buffer. The multi-part MAC is
        // byte-identical to MAC-of-concatenation, so the payload is never copied into a
        // scratch `Vec` — this matters for large frames (e.g. pipeline activation tensors).
        let prefix: &[u8] = match &self.prev_mac {
            None => self.topic.as_bytes(),
            Some(prev) => &prev[..],
        };
        let expected_mac = keyed_mac_truncated_parts(&self.key, &[prefix, capnp_data]);

        if !constant_time_eq(received_mac, &expected_mac) {
            anyhow::bail!("MAC verification failed");
        }

        // Prepare the chain update, but do not mutate verifier state until the
        // complete authenticated block (including AEAD payloads) is accepted.
        let mut new_prev = [0u8; 16];
        new_prev.copy_from_slice(received_mac);

        // Parse StreamBlock with a borrowing (zero-copy) reader: the capnp `Reader`
        // references `capnp_data` directly instead of copying the whole block into an
        // owned message, so a large payload stays a view into the receive buffer.
        let mut slice: &[u8] = capnp_data;
        let reader = capnp::serialize::read_message_from_flat_slice(
            &mut slice,
            capnp::message::ReaderOptions::default(),
        )?;
        let block = reader.get_root::<streaming_capnp::stream_block::Reader>()?;
        let block_epoch = block.get_epoch();
        if let Some(ratchet) = &self.epoch_ratchet {
            ensure!(
                block_epoch == ratchet.epoch(),
                "stream block epoch {} is not committed epoch {}",
                block_epoch,
                ratchet.epoch()
            );
        }

        // Policy-selected ordering/replay enforcement (#163). The MAC already authenticates
        // `sequenceNumber` (it's inside the block, #219), so a tampered sequenceNumber fails
        // MAC above; here we enforce *position*. `None` policy ⇒ legacy behaviour (MAC chain
        // only, no sequenceNumber/completion checks).
        let mut next_seq_cursor = self.seq_cursor;
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
                    next_seq_cursor = Some(
                        sequence_number
                            .checked_add(1)
                            .ok_or_else(|| anyhow::anyhow!("stream sequenceNumber exhausted"))?,
                    );
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

        // #321: the AEAD AAD binds each sealed payload to (topic, epoch); read the
        // block epoch so a sealed Tagged payload can only open under its own epoch.
        let block_sequence = block.get_sequence_number();

        let payloads_reader = block.get_payloads()?;
        let mut payloads = Vec::with_capacity(payloads_reader.len() as usize);

        let mut terminal_seen = self.terminal_seen;
        for i in 0..payloads_reader.len() {
            let p = payloads_reader.get(i);

            use streaming_capnp::stream_payload::Which;
            let payload = match p.which()? {
                Which::Data(data_result) => StreamPayload::Data(data_result?.to_vec()),
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
                    match self.enc_key {
                        // #321: transport AEAD ON — open the sealed payload back into
                        // Data/Complete (fails closed on tamper / wrong key).
                        Some(ref enc_key) => open_sealed_payload(
                            enc_key,
                            &self.topic,
                            block_epoch,
                            block_sequence,
                            i as usize,
                            tagged.get_tag()?,
                            tagged.get_payload()?,
                            tagged.get_nonce()?,
                            tagged.get_key_commitment()?,
                        )?,
                        // No transport key (E2E notification path): pass Tagged through.
                        None => StreamPayload::Tagged {
                            tag: tagged.get_tag()?.to_vec(),
                            payload: tagged.get_payload()?.to_vec(),
                            nonce: tagged.get_nonce()?.to_vec(),
                            key_commitment: tagged.get_key_commitment()?.to_vec(),
                        },
                    }
                }
            };

            // Track terminal observation for the `completion` axis (#163).
            if matches!(
                payload,
                StreamPayload::Complete(_) | StreamPayload::Error(_)
            ) {
                terminal_seen = true;
            }
            payloads.push(payload);
        }

        self.prev_mac = Some(new_prev);
        self.seq_cursor = next_seq_cursor;
        self.terminal_seen = terminal_seen;

        Ok(payloads)
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        &self.topic
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    a.ct_eq(b).into()
}

// ============================================================================
// #321 — transport-level AEAD framing (cross-target seal/open glue).
//
// The seal side lives in `moq_stream` (native publisher); the open side here is
// cross-target so the wasm/browser consumer opens sealed blocks too. Both sides
// share this AAD + kind-tag framing.
// ============================================================================

/// Plaintext kind tag prepended inside the AEAD-sealed bytes so the open path can
/// restore the original payload variant (`Data` vs `Complete`) without a schema
/// change. Authenticated as part of the AEAD plaintext.
pub(crate) const SEALED_KIND_DATA: u8 = 0x00;
pub(crate) const SEALED_KIND_COMPLETE: u8 = 0x01;

/// Build the AEAD AAD (also the `encrypt_event` "prefix") binding each sealed
/// payload to its `topic` and key-`epoch` (#321/#223). Reused verbatim by the
/// publisher seal path; a block replayed under a different epoch fails AEAD open.
pub(crate) fn stream_aead_aad(
    topic: &str,
    epoch: u64,
    sequence_number: u64,
    payload_index: usize,
) -> String {
    format!(
        "{topic}|identified-stream-object-v1|epoch={epoch}|sequence={sequence_number}|payload={payload_index}"
    )
}

/// Open an AEAD-sealed `Tagged` payload (#321), restoring the original
/// Data/Complete variant. Returns `Err` on tamper / wrong key (fail-closed).
///
/// `epoch` is the StreamBlock's epoch and MUST match the seal-side AAD.
pub(crate) fn open_sealed_payload(
    enc_key: &[u8; 32],
    topic: &str,
    epoch: u64,
    sequence_number: u64,
    payload_index: usize,
    tag: &[u8],
    ciphertext: &[u8],
    nonce: &[u8],
    key_commitment: &[u8],
) -> Result<StreamPayload> {
    use crate::crypto::event_crypto::{check_key_commitment, decrypt_event_full};

    let nonce12: [u8; 12] = nonce
        .try_into()
        .map_err(|_| anyhow::anyhow!("stream AEAD: bad nonce length {}", nonce.len()))?;
    let commitment16: [u8; 16] = key_commitment.try_into().map_err(|_| {
        anyhow::anyhow!(
            "stream AEAD: bad key_commitment length {}",
            key_commitment.len()
        )
    })?;

    // Fast committing-AEAD rejection of a wrong-key block before the GCM open.
    if !check_key_commitment(enc_key, &nonce12, &commitment16) {
        anyhow::bail!("stream AEAD: key commitment mismatch (wrong key or tampered block)");
    }

    let aad = stream_aead_aad(topic, epoch, sequence_number, payload_index);
    let plaintext = decrypt_event_full(enc_key, &nonce12, tag, ciphertext, &aad)
        .map_err(|e| anyhow::anyhow!("stream AEAD open failed: {e}"))?;

    let (&kind, body) = plaintext
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("stream AEAD: empty sealed plaintext (missing kind tag)"))?;
    match kind {
        SEALED_KIND_DATA => Ok(StreamPayload::Data(body.to_vec())),
        SEALED_KIND_COMPLETE => Ok(StreamPayload::Complete(body.to_vec())),
        other => anyhow::bail!("stream AEAD: unknown sealed kind tag {other:#x}"),
    }
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
    ///
    /// #385 INVARIANT (S8 close-out — `dhPublic` authenticity):
    /// The entire AEAD + chained-HMAC integrity envelope bootstraps from
    /// `stream_info.dh_public` — the server's ephemeral Ristretto255 DH public
    /// key. It is trustworthy ONLY because `StreamInfo` arrives over the
    /// authenticated RPC response: carrier target/path checks are insufficient;
    /// the reply is independently wrapped in a `ResponseEnvelope` carrying a
    /// COSE composite signature (EdDSA +
    /// ML-DSA-65 under Hybrid), so `dhPublic` authentication reaches PQ
    /// strength. A `dh_public` sourced from an *unauthenticated* path would let
    /// any MITM substitute its own key, derive the stream keys itself, and forge
    /// frames — defeating the whole envelope. This entry point is the single
    /// library/codegen boundary that turns a `StreamInfo` into a live stream;
    /// `dh_public` MUST therefore arrive here already authenticated. Callers
    /// MUST NOT construct `StreamInfo` from untrusted input (e.g. a forwarded
    /// relay advertisement or an unsigned lookup result). See
    /// [`crate::rpc_client::parse_stream_info`] — the only producer of a
    /// `StreamInfo` from wire bytes, reached solely via the authenticated
    /// `call_streaming` RPC reply path.
    pub async fn open(
        transport: &T,
        stream_info: StreamInfo,
        client_secret: &[u8; 32],
        client_pubkey: &[u8; 32],
    ) -> Result<Self> {
        // Pure crypto — same code, both targets.
        // #385: assert the INVARIANT above at the entry point — the envelope's
        // integrity is a consequence of `dh_public`'s authenticity, so a
        // zero/empty `dh_public` (an obvious "StreamInfo-from-untrusted-input"
        // signature) fails fast here rather than producing a silently-weak
        // handshake. (A *substituted* valid key is not detectable at this layer;
        // it is detected by the consumer's AEAD/MAC verification failing — see
        // the `relay_dhpublic_substitution_is_rejected` test.)
        assert!(
            stream_info.dh_public.iter().any(|&b| b != 0),
            "#385 INVARIANT: StreamInfo.dh_public must be a real (non-zero) DH key — \
             never construct StreamInfo from untrusted input"
        );
        let shared_secret = dh_compute_raw(client_secret, &stream_info.dh_public)?;
        let keys = derive_stream_keys(&shared_secret, client_pubkey, &stream_info.dh_public)?;

        // Transport-abstracted — ZMQ or WebTransport
        let subscriber = transport.subscribe(keys.topic.as_bytes()).await?;
        let publisher = transport.publish(keys.ctrl_topic.as_bytes()).await.ok();

        // QoS contract from the signed StreamInfo handshake — enforced for both native and WASM paths.
        // #321: AEAD ON for this DH-keyed (mesh/networked) stream — open sealed Tagged blocks.
        let verifier =
            StreamVerifier::with_policy(*keys.mac_key, keys.topic.clone(), stream_info.qos.into())
                .with_enc_key(*keys.enc_key);

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
    ///
    /// Takes `&mut self` so the future is `Send` (holds `&mut Self`, which is
    /// `Send` because `Self: Send`) rather than requiring `Self: Sync`. See
    /// #670 / the `StreamHandle::cancel` trait method.
    pub async fn cancel(&mut self) -> Result<()> {
        self.cancelled.store(true, Ordering::Release);
        if let Some(ref pub_handle) = self.publisher {
            let msg = build_stream_control_cancel();
            let mac = keyed_mac_truncated(&self.ctrl_mac_key, &msg);
            pub_handle
                .send_frames(&[self.ctrl_topic.as_bytes(), &msg, &mac])
                .await?;
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
///
/// # `Send` futures on native (#670)
///
/// On native the trait methods return `Send` futures (plain `#[async_trait]`),
/// so a native `Send + Sync` `Mount` (e.g. the `/stream` pipe reader in
/// `hyprstream-rpc-std`) can hold a `StreamHandle` future across `.await`. This
/// is sound because the only implementors are `StreamHandleImpl<T>` and
/// `Box<dyn StreamHandle>`, and `StreamHandleImpl<T>` is built entirely from the
/// `Transport` associated types — which are already `Send`
/// (`Transport::Sub: Send`, `Transport::Pub: Send`), and whose native
/// `PublishSink` impls return `Send` futures.
///
/// On wasm the trait stays `#[async_trait(?Send)]`: the WebTransport
/// `WtPublisher::send_frames` future holds JS values and is genuinely `!Send`,
/// and the browser namespace never needs `Send` (wasm32 is single-threaded).
/// The dual `cfg_attr` mirrors the `hyprstream_vfs::Mount` trait itself.
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait StreamHandle: Send {
    /// Get next verified payload. Returns None on stream end.
    async fn next_payload(&mut self) -> Result<Option<StreamPayload>>;

    /// Cancel the stream via ctrl channel.
    ///
    /// Takes `&mut self` (not `&self`) so the returned future is `Send` without
    /// requiring `Self: Sync` — a `dyn StreamHandle` is `Send` but not `Sync`
    /// (its transport `Sub`/`Pub` are only `Send`-bound), so a `&self` future
    /// could not be sent across threads. `&mut self` also matches the state it
    /// mutates (the cancelled flag). See #670.
    async fn cancel(&mut self) -> Result<()>;

    /// Get the stream ID.
    fn stream_id(&self) -> &str;

    /// Check if stream is completed.
    fn is_completed(&self) -> bool;
}

/// Blanket impl: any `StreamHandleImpl<T>` satisfies `StreamHandle`.
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl<T: Transport> StreamHandle for StreamHandleImpl<T> {
    async fn next_payload(&mut self) -> Result<Option<StreamPayload>> {
        StreamHandleImpl::next_payload(self).await
    }

    async fn cancel(&mut self) -> Result<()> {
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
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl StreamHandle for Box<dyn StreamHandle> {
    async fn next_payload(&mut self) -> Result<Option<StreamPayload>> {
        (**self).next_payload().await
    }

    async fn cancel(&mut self) -> Result<()> {
        (**self).cancel().await
    }

    fn stream_id(&self) -> &str {
        (**self).stream_id()
    }

    fn is_completed(&self) -> bool {
        (**self).is_completed()
    }
}

// ─── Conversion from wire StreamOpt ───────────────────────────────────────────

/// Convert the full wire `StreamOpt` (5 axes) to the consumer's 2-axis slice.
///
/// The delivery, retention, and overflow_policy axes are producer-side concerns;
/// the consumer only enforces ordering + completion.
impl From<crate::stream_info::StreamOpt> for VerifierContract {
    fn from(q: crate::stream_info::StreamOpt) -> Self {
        let ordering = match q.ordering {
            crate::stream_info::Ordering::Ordered => StreamOrdering::Ordered,
            crate::stream_info::Ordering::Unordered { anti_replay_window } => {
                StreamOrdering::Unordered { anti_replay_window }
            }
        };
        let completion = match q.completion {
            crate::stream_info::Completion::EndOfStream => Completion::EndOfStream,
            crate::stream_info::Completion::None => Completion::Open,
        };
        VerifierContract {
            ordering,
            completion,
        }
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
        (
            vec![topic.as_bytes().to_vec(), capnp_bytes, mac.to_vec()],
            mac,
        )
    }

    #[test]
    fn tampered_mac_is_rejected() {
        // Exercises the subtle::ConstantTimeEq path in StreamVerifier::verify.
        let key = [0xAAu8; 32];
        let topic = "tamper";
        let mut v = StreamVerifier::with_policy(
            key,
            topic.to_owned(),
            VerifierContract {
                ordering: StreamOrdering::Ordered,
                completion: Completion::Open,
            },
        );
        let (mut frame, _) = frame(&key, topic, None, 0, false);
        // Corrupt the MAC part (last element of the frame).
        let mac = frame.last_mut().unwrap();
        mac[0] ^= 0xFF; // flip a bit — constant_time_eq must reject this
        let err = v.verify(&frame).unwrap_err().to_string();
        assert!(
            err.contains("MAC")
                || err.contains("mac")
                || err.contains("HMAC")
                || err.contains("invalid"),
            "tampered MAC should be rejected, got: {err}"
        );
    }

    #[test]
    fn ordered_accepts_contiguous_and_rejects_gap() {
        let key = [7u8; 32];
        let topic = "tok";
        let mut v = StreamVerifier::with_policy(
            key,
            topic.to_owned(),
            VerifierContract {
                ordering: StreamOrdering::Ordered,
                completion: Completion::Open,
            },
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
            VerifierContract {
                ordering: StreamOrdering::Unordered {
                    anti_replay_window: 4,
                },
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
            VerifierContract {
                ordering: StreamOrdering::Ordered,
                completion: Completion::EndOfStream,
            },
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

    // ── #385 S8 — relay dhPublic-substitution rejection ──────────────────────
    //
    // Threat (S3 relay + S5 provenance gap): once a relay is interposed, a MITM
    // relay could try to substitute its OWN `dhPublic` in the handshake so the
    // consumer derives keys the relay controls. S5 COSE provenance authenticates
    // each *frame* to the producing host, but the handshake `dhPublic`→producer
    // binding is not separately tested. This test closes that gap: it shows the
    // consumer's AEAD verification FAILS when the keys are derived from a
    // `dhPublic` the relay substituted, because they do not match the keys the
    // producer sealed the frame under. The consumer, holding the *authenticated*
    // `dhPublic` (delivered over the RPC channel — see `StreamHandleImpl::open`
    // INVARIANT), is never fooled: it never re-derives from a relay-supplied
    // `dhPublic`. The test asserts the negative — *if* the relay's substituted
    // key reached the key-derivation, AEAD open would fail — which is exactly
    // the rejection property.
    #[test]
    fn relay_dhpublic_substitution_is_rejected() {
        use crate::crypto::{derive_stream_keys, generate_ephemeral_keypair, ristretto_dh_raw};

        // Consumer generates an ephemeral keypair (the `client_pubkey` it sends
        // in the streaming RPC request).
        let (client_secret, client_pub) = generate_ephemeral_keypair();
        let client_pub_bytes = client_pub.to_bytes();
        let client_secret_bytes = client_secret.scalar().to_bytes();

        // Producer's REAL ephemeral keypair — its public half is the `dhPublic`
        // the authenticated RPC reply carries (the value the consumer trusts).
        let (producer_secret, producer_pub) = generate_ephemeral_keypair();
        let real_dh_public = producer_pub.to_bytes();

        // The relay's substituted keypair — the MITM key the relay would inject
        // in place of `dhPublic` to try to own the derived keys.
        let (_relay_secret, relay_pub) = generate_ephemeral_keypair();
        let relay_dh_public = relay_pub.to_bytes();
        assert_ne!(
            real_dh_public, relay_dh_public,
            "test setup: keys must differ"
        );

        // Producer derives the REAL stream keys from the real DH (ECDH with the
        // client's public) — these are the keys it seals every frame under.
        let producer_shared =
            ristretto_dh_raw(&producer_secret.scalar().to_bytes(), &client_pub_bytes).unwrap();
        let real_keys =
            derive_stream_keys(&producer_shared, &client_pub_bytes, &real_dh_public).unwrap();

        // Build one AEAD-sealed Tagged payload exactly as the producer does
        // (`moq_stream::seal_payload`), using the REAL enc_key. This is the
        // ciphertext the relay forwards verbatim (it is blind to the keys).
        let epoch = 0u64;
        let topic = real_keys.topic.clone();
        let real_enc = *real_keys.enc_key;
        let plaintext = {
            let mut p = Vec::with_capacity(1 + b"secret-token".len());
            p.push(SEALED_KIND_DATA);
            p.extend_from_slice(b"secret-token");
            p
        };
        use crate::crypto::event_crypto::{encrypt_event, EventPrivacy};
        let aad = stream_aead_aad(&topic, epoch, 0, 0);
        let (tag, ciphertext, nonce, key_commitment) =
            encrypt_event(&real_enc, &aad, &plaintext, EventPrivacy::ZeroKnowledge).unwrap();

        // 1) Consumer holding the AUTHENTICATED keys (the real `dhPublic` arrived
        //    over the RPC channel) opens the relay-forwarded frame: SUCCESS.
        let consumer_shared = ristretto_dh_raw(&client_secret_bytes, &real_dh_public).unwrap();
        let consumer_keys =
            derive_stream_keys(&consumer_shared, &client_pub_bytes, &real_dh_public).unwrap();
        assert_eq!(
            *consumer_keys.enc_key, real_enc,
            "consumer (real dhPublic) and producer derive the same enc_key"
        );
        let opened = open_sealed_payload(
            &consumer_keys.enc_key,
            &topic,
            epoch,
            0,
            0,
            &tag,
            &ciphertext,
            &nonce,
            &key_commitment,
        )
        .expect("consumer with the authenticated dhPublic opens the frame");
        assert!(matches!(opened, StreamPayload::Data(d) if d == b"secret-token"));

        // 2) Relay-substitution attack: a consumer that (wrongly) re-derived its
        //    keys from the RELAY's substituted `dhPublic` cannot open the frame —
        //    AEAD fails (wrong key). This is the rejection: the relay cannot make
        //    a substituted `dhPublic` stick, because the keys won't line up with
        //    the producer's sealed ciphertext. (The honest consumer never enters
        //    this branch — it keeps the authenticated `dhPublic`; this case shows
        //    what would happen if the substituted key were used.)
        let sub_shared = ristretto_dh_raw(&client_secret_bytes, &relay_dh_public).unwrap();
        let sub_keys =
            derive_stream_keys(&sub_shared, &client_pub_bytes, &relay_dh_public).unwrap();
        assert_ne!(
            *sub_keys.enc_key, real_enc,
            "substituted dhPublic MUST derive a different enc_key"
        );
        let subst_err = open_sealed_payload(
            &sub_keys.enc_key,
            &topic,
            epoch,
            0,
            0,
            &tag,
            &ciphertext,
            &nonce,
            &key_commitment,
        )
        .unwrap_err();
        let msg = subst_err.to_string();
        assert!(
            msg.contains("AEAD") || msg.contains("commitment") || msg.contains("key"),
            "substituted-dhPublic AEAD open should fail closed, got: {msg}"
        );

        // 3) The MAC chain has the same property: a frame chained under the real
        //    mac_key fails verification under the relay's mac_key. This is the
        //    belt-and-braces check on the chained-HMAC envelope.
        let real_mac = *real_keys.mac_key;
        let sub_mac = *sub_keys.mac_key;
        assert_ne!(
            real_mac, sub_mac,
            "substituted dhPublic MUST derive a different mac_key"
        );
        let (real_frame, _) = frame(&real_mac, &topic, None, 0, false);
        // Real consumer (real mac_key) verifies the producer's frame.
        let mut v_real = StreamVerifier::with_policy(
            real_mac,
            topic.clone(),
            VerifierContract {
                ordering: StreamOrdering::Ordered,
                completion: Completion::Open,
            },
        );
        assert!(
            v_real.verify(&real_frame).is_ok(),
            "real mac_key verifies the frame"
        );
        // Substituted consumer (relay mac_key) CANNOT verify the producer's frame.
        let mut v_sub = StreamVerifier::with_policy(
            sub_mac,
            topic,
            VerifierContract {
                ordering: StreamOrdering::Ordered,
                completion: Completion::Open,
            },
        );
        assert!(
            v_sub.verify(&real_frame).is_err(),
            "a frame chained under the producer's mac_key MUST NOT verify under the \
             relay's substituted-dhPublic mac_key (the consumer rejects the substitution)"
        );
    }
}
