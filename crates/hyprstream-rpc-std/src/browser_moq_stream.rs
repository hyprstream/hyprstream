//! Blind ciphertext consumption after a verified Model RPC response.
//!
//! The RPC, not the relay CONNECT, authorizes inference. This preserves the
//! existing named Ristretto-DH/AES-GCM stream profile; it is not Hybrid KEM.
//! No external URL, credential, or server-authentication fallback is accepted.

use anyhow::{ensure, Result};
use hyprstream_rpc::crypto::{derive_stream_keys, ristretto_dh_raw};
use hyprstream_rpc::stream_consumer::{StreamPayload, StreamVerifier};
use hyprstream_rpc::stream_info::{Completion, Ordering, StreamInfo, TransportConfig};

// WebTransport reports an empty string when no application subprotocol was
// negotiated. The transport trait represents that state as None, which lets
// moq-net negotiate using its existing SETUP exchange. Never rewrite a real
// (including unknown or whitespace-only) nonempty protocol.
pub(crate) fn negotiated_protocol(protocol: Option<&str>) -> Option<&str> {
    protocol.filter(|value| !value.is_empty())
}

pub(crate) struct BrowserStream {
    pub(crate) url: url::Url,
    pub(crate) cert_hashes: Vec<Vec<u8>>,
    pub(crate) broadcast_path: String,
    pub(crate) stream_id: String,
    verifier: StreamVerifier,
}

impl BrowserStream {
    /// Called only with StreamInfo decoded from the verified RPC response and
    /// the SAME ephemeral pair committed in that request.
    pub(crate) fn from_verified_info(
        info: StreamInfo,
        secret: &[u8; 32],
        public: &[u8; 32],
    ) -> Result<Self> {
        ensure!(
            info.kem_ciphertexts.is_empty(),
            "unsupported browser stream KEM profile"
        );
        ensure!(info.dh_public != [0; 32], "missing stream DH public key");
        ensure!(
            matches!(info.qos.ordering, Ordering::Ordered)
                && matches!(info.qos.completion, Completion::EndOfStream),
            "Model stream requires ordered terminal completion"
        );
        let shared = ristretto_dh_raw(secret, &info.dh_public)?;
        let keys = derive_stream_keys(&shared, public, &info.dh_public)?;
        let segments: Vec<_> = info.broadcast_path.split('/').collect();
        ensure!(
            info.broadcast_path.len() <= 1024
                && segments.len() >= 2
                && segments.last().copied() == Some(keys.topic.as_str())
                && segments.iter().all(|s| !s.is_empty()
                    && *s != "."
                    && *s != ".."
                    && s.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')),
            "invalid or mismatched signed broadcast path"
        );
        let reach = info
            .announced_at
            .iter()
            .find_map(|destination| match &destination.transport {
                TransportConfig::Quic(reach) => Some(reach),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("no signed browser-compatible MoQ destination"))?;
        ensure!(
            !reach.addr.is_empty()
                && !reach.addr.contains(['/', '?', '#', '@', '\\'])
                && !reach.addr.bytes().any(|b| b.is_ascii_whitespace()),
            "invalid signed MoQ address"
        );
        let address = url::Url::parse(&format!("https://{}/moq", reach.addr))?;
        let (_, port) = reach
            .addr
            .rsplit_once(':')
            .ok_or_else(|| anyhow::anyhow!("signed MoQ address omitted port"))?;
        let port: u16 = port.parse()?;
        ensure!(
            port != 0 && address.port_or_known_default() == Some(port),
            "invalid signed MoQ port"
        );
        ensure!(
            address.host_str().is_some(),
            "missing signed MoQ address host"
        );
        ensure!(
            reach.cert_hashes.len() <= 16 && reach.cert_hashes.iter().all(|hash| hash.len() == 32),
            "invalid signed MoQ certificate hash"
        );
        // Browsers cannot set SNI separately: use the SIGNED server name and
        // SIGNED explicit port, not the page origin or an invented endpoint.
        ensure!(
            !reach.server_name.is_empty()
                && !reach.server_name.contains(['/', '?', '#', '@', '\\', ':'])
                && !reach.server_name.bytes().any(|b| b.is_ascii_whitespace()),
            "invalid signed MoQ server name"
        );
        let url = url::Url::parse(&format!("https://{}:{port}/moq", reach.server_name))?;
        ensure!(
            url.host_str() == Some(reach.server_name.as_str()),
            "noncanonical signed MoQ server name"
        );
        validate_host(url.host_str())?;
        Ok(Self {
            url,
            cert_hashes: reach.cert_hashes.clone(),
            broadcast_path: info.broadcast_path,
            stream_id: info.stream_id,
            verifier: StreamVerifier::with_policy(
                *keys.mac_key,
                keys.topic.clone(),
                info.qos.into(),
            )
            .with_enc_key(*keys.enc_key),
        })
    }

    pub(crate) fn verify_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamPayload>> {
        ensure!(
            !self.verifier.terminal_seen(),
            "frame after stream terminal"
        );
        ensure!(
            frame.len() >= 16 && frame.len() <= 4 * 1024 * 1024,
            "invalid MoQ frame length"
        );
        let split = frame.len() - 16;
        // The shared verifier supports cleartext legacy payloads; this blind
        // ciphertext consumer deliberately accepts ONLY encrypted Objects.
        let mut slice = &frame[..split];
        let message = capnp::serialize::read_message_from_flat_slice(
            &mut slice,
            capnp::message::ReaderOptions::default(),
        )?;
        ensure!(slice.is_empty(), "trailing MoQ frame bytes");
        let block = message.get_root::<hyprstream_rpc::streaming_capnp::stream_block::Reader>()?;
        let payloads = block.get_payloads()?;
        ensure!(!payloads.is_empty(), "empty encrypted stream block");
        for payload in payloads {
            ensure!(
                matches!(
                    payload.which()?,
                    hyprstream_rpc::streaming_capnp::stream_payload::Tagged(_)
                ),
                "blind MoQ stream rejected unencrypted payload"
            );
        }
        let topic = self.verifier.topic().to_owned();
        let result =
            self.verifier
                .verify_parts(topic.as_bytes(), &frame[..split], &frame[split..])?;
        for (index, payload) in result.iter().enumerate() {
            ensure!(
                !matches!(payload, StreamPayload::Tagged { .. }),
                "unopened encrypted stream payload"
            );
            ensure!(
                !matches!(
                    payload,
                    StreamPayload::Complete(_) | StreamPayload::Error(_)
                ) || index + 1 == result.len(),
                "payload after stream terminal"
            );
        }
        Ok(result)
    }

    pub(crate) fn apply_rpc_tls_policy(&mut self, rpc_url: &str, pins: &[[u8; 32]]) -> Result<()> {
        let rpc = url::Url::parse(rpc_url)?;
        ensure!(
            rpc.scheme() == "https"
                && rpc.username().is_empty()
                && rpc.password().is_none()
                && rpc.query().is_none()
                && rpc.fragment().is_none()
                && rpc.path() == hyprstream_rpc::browser_provisioning::BROWSER_RPC_PATH
                && rpc.host_str() == self.url.host_str()
                && rpc.port_or_known_default() == self.url.port_or_known_default(),
            "signed MoQ destination differs from accepted-current Model TLS authority"
        );
        if pins.is_empty() {
            // The native producer currently emits a leaf hash even for WebPKI.
            // Reuse the accepted-current SAME-authority WebPKI policy, never a
            // retry after pin failure. CA and hostname validation remain on.
            self.cert_hashes.clear();
        } else {
            let mut expected: Vec<_> = pins.iter().map(|pin| pin.to_vec()).collect();
            expected.sort();
            let mut advertised = self.cert_hashes.clone();
            advertised.sort();
            ensure!(
                advertised == expected,
                "signed MoQ pins differ from accepted-current Model TLS policy"
            );
        }
        Ok(())
    }

    pub(crate) fn eof(&self) -> Result<()> {
        ensure!(
            self.verifier.terminal_seen(),
            "MoQ stream truncated before authenticated terminal"
        );
        Ok(())
    }
}

fn validate_host(host: Option<&str>) -> Result<()> {
    let host = host.ok_or_else(|| anyhow::anyhow!("missing MoQ host"))?;
    let host = host.trim_end_matches('.');
    ensure!(
        !host.eq_ignore_ascii_case("localhost")
            && !host.ends_with(".localhost")
            && !host
                .trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback() || ip.is_unspecified()),
        "signed MoQ destination is not externally routable"
    );
    Ok(())
}

// Completion metadata is authenticated control data, never generated text.
// Keep this projection local to MoQ; the legacy stream API is unchanged.
fn application_payload(payload: StreamPayload) -> Result<Option<StreamPayload>> {
    match payload {
        StreamPayload::Data(data) => Ok(Some(StreamPayload::Data(data))),
        StreamPayload::Complete(_) => Ok(None),
        StreamPayload::Error(message) => Err(anyhow::Error::msg(message)),
        StreamPayload::Tagged { .. } => anyhow::bail!("unopened encrypted stream payload"),
    }
}

// Match the native MoQ consumer's startup, group-idle and frame deadlines.
// web-async uses the browser timer on WASM, not a Tokio runtime clock.
const ANNOUNCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const GROUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const FRAME_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

async fn announced_broadcast(
    consumer: &moq_net::OriginConsumer,
    path: &str,
    timeout: std::time::Duration,
) -> Result<moq_net::BroadcastConsumer> {
    web_async::time::timeout(timeout, consumer.announced_broadcast(path))
        .await
        .map_err(|_| anyhow::anyhow!("signed MoQ broadcast announcement timed out"))?
        .ok_or_else(|| anyhow::anyhow!("signed MoQ broadcast not announced"))
}

struct OrderedGroups {
    track: moq_net::TrackConsumer,
    group: Option<moq_net::GroupConsumer>,
    expected_sequence: u64,
}

impl OrderedGroups {
    fn new(track: moq_net::TrackConsumer) -> Self {
        Self {
            track,
            group: None,
            expected_sequence: 0,
        }
    }

    async fn next_frame(&mut self) -> Result<Option<Vec<u8>>> {
        loop {
            if self.group.is_none() {
                // QUIC group streams may arrive out of order. next_group()
                // skips late lower sequences, invalidating the chained MAC.
                self.group = web_async::time::timeout(
                    GROUP_TIMEOUT,
                    self.track.get_group(self.expected_sequence),
                )
                .await
                .map_err(|_| anyhow::anyhow!("signed MoQ group timed out"))??;
                if self.group.is_none() {
                    return Ok(None);
                }
            }
            if let Some(group) = self.group.as_mut() {
                match web_async::time::timeout(FRAME_TIMEOUT, group.read_frame())
                    .await
                    .map_err(|_| anyhow::anyhow!("signed MoQ frame timed out"))??
                {
                    Some(frame) => return Ok(Some(frame.to_vec())),
                    None => {
                        self.group = None;
                        self.expected_sequence = self
                            .expected_sequence
                            .checked_add(1)
                            .ok_or_else(|| anyhow::anyhow!("signed MoQ group sequence overflow"))?;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) struct BrowserMoqHandle {
    stream: BrowserStream,
    session: moq_net::Session,
    groups: OrderedGroups,
    pending: std::collections::VecDeque<StreamPayload>,
    completed: bool,
}

#[cfg(target_arch = "wasm32")]
impl BrowserMoqHandle {
    pub(crate) async fn open(
        info: StreamInfo,
        secret: &[u8; 32],
        public: &[u8; 32],
        rpc_url: &str,
        pins: &[[u8; 32]],
    ) -> Result<Self> {
        let mut stream = BrowserStream::from_verified_info(info, secret, public)?;
        stream.apply_rpc_tls_policy(rpc_url, pins)?;
        let client = if stream.cert_hashes.is_empty() {
            web_transport_wasm::ClientBuilder::new().with_system_roots()
        } else {
            web_transport_wasm::ClientBuilder::new()
                .with_server_certificate_hashes(stream.cert_hashes.clone())
        };
        let transport = client.connect(stream.url.clone()).await?;
        let origin = moq_net::Origin::random().produce();
        let consumer = origin.consume();
        // Consume only. No publisher or SUB/PUB stream is opened on RPC.
        let mut session = moq_net::Client::new()
            .with_consume(origin)
            .connect(crate::moq_wt_session::WtSession(transport))
            .await?;
        let broadcast =
            match announced_broadcast(&consumer, &stream.broadcast_path, ANNOUNCE_TIMEOUT).await {
                Ok(broadcast) => broadcast,
                Err(error) => {
                    session.close(moq_net::Error::Cancel);
                    return Err(error);
                }
            };
        let track = broadcast.subscribe_track(&moq_net::Track::new("stream"))?;
        Ok(Self {
            stream,
            session,
            groups: OrderedGroups::new(track),
            pending: Default::default(),
            completed: false,
        })
    }

    pub(crate) async fn next_payload(&mut self) -> Result<Option<StreamPayload>> {
        let result = self.next_inner().await;
        if result.is_err() {
            self.cancel();
        }
        result
    }

    async fn next_inner(&mut self) -> Result<Option<StreamPayload>> {
        loop {
            if let Some(payload) = self.pending.pop_front() {
                if matches!(
                    payload,
                    StreamPayload::Complete(_) | StreamPayload::Error(_)
                ) {
                    self.completed = true;
                    self.session.close(moq_net::Error::Cancel);
                }
                return application_payload(payload);
            }
            if self.completed {
                return Ok(None);
            }
            match self.groups.next_frame().await? {
                Some(frame) => self.pending.extend(self.stream.verify_frame(&frame)?),
                None => {
                    self.stream.eof()?;
                    self.completed = true;
                    return Ok(None);
                }
            }
        }
    }

    pub(crate) fn cancel(&mut self) {
        self.completed = true;
        self.pending.clear();
        self.groups.group = None;
        self.session.close(moq_net::Error::Cancel);
    }
    pub(crate) fn stream_id(&self) -> &str {
        &self.stream.stream_id
    }
    pub(crate) fn is_completed(&self) -> bool {
        self.completed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::{generate_ephemeral_keypair, keyed_mac_truncated_parts};
    use hyprstream_rpc::stream_info::{Destination, Job, QuicReach, Role, StreamOptPreset};

    #[tokio::test]
    async fn browser_moq_announcement_is_bounded() -> Result<()> {
        assert_eq!(
            ANNOUNCE_TIMEOUT,
            hyprstream_rpc::moq_stream::BROADCAST_ANNOUNCE_TIMEOUT
        );
        assert_eq!(
            GROUP_TIMEOUT,
            hyprstream_rpc::moq_stream::GROUP_IDLE_TIMEOUT
        );
        assert_eq!(
            FRAME_TIMEOUT,
            hyprstream_rpc::moq_stream::FRAME_READ_TIMEOUT
        );
        let origin = moq_net::Origin::random().produce();
        let consumer = origin.consume();
        // A publisher can disappear before the second WebTransport connects.
        let broadcast = moq_net::Broadcast::default().produce();
        origin.publish_broadcast("fixture/gone", broadcast.consume());
        drop(broadcast);
        tokio::task::yield_now().await;
        let result = announced_broadcast(
            &consumer,
            "fixture/gone",
            std::time::Duration::from_millis(5),
        )
        .await;
        assert_eq!(
            result.err().unwrap().to_string(),
            "signed MoQ broadcast announcement timed out"
        );
        let broadcast = moq_net::Broadcast::default().produce();
        origin.publish_broadcast("fixture/present", broadcast.consume());
        announced_broadcast(&consumer, "fixture/present", ANNOUNCE_TIMEOUT).await?;
        Ok(())
    }

    #[tokio::test]
    async fn browser_moq_reordered_groups_preserve_native_mac_chain() -> Result<()> {
        use hyprstream_rpc::moq_stream::MoqStreamOrigin;
        use hyprstream_rpc::streaming::StreamContext;
        let (secret, public) = generate_ephemeral_keypair();
        let ctx = StreamContext::from_third_party_interop_dh(&public.to_bytes())?
            .with_qos_preset::<Job>();
        let mut stream = BrowserStream::from_verified_info(
            info(&ctx),
            &secret.scalar().to_bytes(),
            &public.to_bytes(),
        )?;
        let origin = MoqStreamOrigin::standalone().build();
        let mut publisher = origin.publisher(&ctx)?;
        publisher.publish_data(b"ordered native data").await?;
        publisher.complete_ref(b"terminal metadata").await?;
        let broadcast =
            announced_broadcast(origin.consumer(), &stream.broadcast_path, ANNOUNCE_TIMEOUT)
                .await?;
        let original = broadcast.subscribe_track(&moq_net::Track::new("stream"))?;
        let mut frames = Vec::new();
        for sequence in 0..2 {
            frames.push(
                original
                    .get_group(sequence)
                    .await?
                    .unwrap()
                    .read_frame()
                    .await?
                    .unwrap(),
            );
        }
        let mut producer = moq_net::Track::new("stream").produce();
        let mut legacy = producer.consume();
        let mut ordered = OrderedGroups::new(producer.consume());
        let mut terminal = producer.create_group(moq_net::Group { sequence: 1 })?;
        terminal.write_frame(frames[1].clone())?;
        terminal.finish()?;
        // Demonstrate the old cursor's causal failure: it returns terminal first.
        assert_eq!(legacy.next_group().await?.unwrap().sequence, 1);
        assert!(web_async::time::timeout(
            std::time::Duration::from_millis(5),
            ordered.next_frame()
        )
        .await
        .is_err());
        let mut data = producer.create_group(moq_net::Group { sequence: 0 })?;
        data.write_frame(frames[0].clone())?;
        data.finish()?;
        producer.finish_at(1)?;
        assert!(legacy.next_group().await?.is_none()); // late Data was skipped
        let payloads = stream.verify_frame(&ordered.next_frame().await?.unwrap())?;
        assert!(
            matches!(&payloads[0], StreamPayload::Data(bytes) if bytes == b"ordered native data")
        );
        let payloads = stream.verify_frame(&ordered.next_frame().await?.unwrap())?;
        assert!(matches!(&payloads[0], StreamPayload::Complete(_)));
        assert!(ordered.next_frame().await?.is_none());
        stream.eof()?;
        Ok(())
    }

    #[test]
    fn browser_moq_normalizes_only_absent_webtransport_subprotocol() {
        assert_eq!(negotiated_protocol(None), None);
        assert_eq!(negotiated_protocol(Some("")), None);
        for value in ["moql", "moq-00", "unknown-protocol", " "] {
            assert_eq!(negotiated_protocol(Some(value)), Some(value));
        }
        // The actual browser adapter must apply this tested normalization,
        // rather than preserving Some("") and triggering UnknownAlpn("").
        assert!(
            include_str!("moq_wt_session.rs").contains("negotiated_protocol(self.0.protocol())")
        );
    }

    fn info(ctx: &hyprstream_rpc::streaming::StreamContext) -> StreamInfo {
        StreamInfo {
            stream_id: ctx.stream_id().to_owned(),
            dh_public: *ctx.server_pubkey(),
            qos: Job::stream_opt(),
            broadcast_path: format!("local/streams/{}", ctx.topic()),
            kem_ciphertexts: Vec::new(),
            moql_server_identity: Default::default(),
            announced_at: vec![Destination {
                role: Role::Direct,
                moql_server_identity: Default::default(),
                transport: TransportConfig::Quic(QuicReach {
                    addr: "model.example:443".into(),
                    server_name: "model.example".into(),
                    cert_hashes: vec![],
                }),
            }],
        }
    }

    #[tokio::test]
    async fn browser_moq_native_publisher_roundtrip() -> Result<()> {
        use hyprstream_rpc::moq_stream::MoqStreamOrigin;
        use hyprstream_rpc::streaming::StreamContext;
        let (secret, public) = generate_ephemeral_keypair();
        let ctx = StreamContext::from_third_party_interop_dh(&public.to_bytes())?
            .with_qos_preset::<Job>();
        let mut stream = BrowserStream::from_verified_info(
            info(&ctx),
            &secret.scalar().to_bytes(),
            &public.to_bytes(),
        )?;
        assert_eq!(stream.url.as_str(), "https://model.example/moq");
        assert!(stream.cert_hashes.is_empty());
        assert_eq!(stream.stream_id, ctx.stream_id());
        assert!(stream.eof().is_err());
        let origin = MoqStreamOrigin::standalone().build();
        let mut publisher = origin.publisher(&ctx)?;
        publisher
            .publish_data(b"genuine native encrypted frame")
            .await?;
        publisher.complete_ref(b"finished").await?;
        let broadcast = origin
            .consumer()
            .announced_broadcast(&stream.broadcast_path)
            .await
            .ok_or_else(|| anyhow::anyhow!("fixture broadcast absent"))?;
        let mut track = broadcast.subscribe_track(&moq_net::Track::new("stream"))?;
        for terminal in [false, true] {
            let mut group = track
                .next_group()
                .await?
                .ok_or_else(|| anyhow::anyhow!("fixture group absent"))?;
            let frame = group
                .read_frame()
                .await?
                .ok_or_else(|| anyhow::anyhow!("fixture frame absent"))?;
            let mut payloads = stream.verify_frame(&frame)?;
            assert_eq!(payloads.len(), 1);
            if terminal {
                assert!(
                    matches!(&payloads[0], StreamPayload::Complete(data) if data == b"finished")
                );
                assert!(application_payload(payloads.remove(0))?.is_none());
            } else {
                assert!(
                    matches!(&payloads[0], StreamPayload::Data(data) if data == b"genuine native encrypted frame")
                );
                assert!(matches!(application_payload(payloads.remove(0))?,
                    Some(StreamPayload::Data(data)) if data == b"genuine native encrypted frame"));
                assert!(stream.eof().is_err());
            }
        }
        stream.eof()?;
        assert!(stream.verify_frame(&[0; 32]).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn browser_moq_empty_generation_does_not_emit_completion_metadata() -> Result<()> {
        use hyprstream_rpc::moq_stream::MoqStreamOrigin;
        use hyprstream_rpc::streaming::StreamContext;
        let (secret, public) = generate_ephemeral_keypair();
        let ctx = StreamContext::from_third_party_interop_dh(&public.to_bytes())?
            .with_qos_preset::<Job>();
        let mut stream = BrowserStream::from_verified_info(
            info(&ctx),
            &secret.scalar().to_bytes(),
            &public.to_bytes(),
        )?;
        let origin = MoqStreamOrigin::standalone().build();
        let mut publisher = origin.publisher(&ctx)?;
        publisher.complete_ref(&[0xff, 0, 0x80, 42]).await?;
        let broadcast = origin
            .consumer()
            .announced_broadcast(&stream.broadcast_path)
            .await
            .ok_or_else(|| anyhow::anyhow!("fixture broadcast absent"))?;
        let mut track = broadcast.subscribe_track(&moq_net::Track::new("stream"))?;
        let mut group = track
            .next_group()
            .await?
            .ok_or_else(|| anyhow::anyhow!("fixture group absent"))?;
        let frame = group
            .read_frame()
            .await?
            .ok_or_else(|| anyhow::anyhow!("fixture frame absent"))?;
        let payloads = stream.verify_frame(&frame)?;
        assert_eq!(payloads.len(), 1);
        let mut output_bytes = Vec::new();
        for payload in payloads {
            if let Some(StreamPayload::Data(data)) = application_payload(payload)? {
                output_bytes.extend(data);
            }
        }
        assert!(output_bytes.is_empty());
        stream.eof()?;
        assert!(application_payload(StreamPayload::Error("fixture failure".into())).is_err());
        Ok(())
    }

    fn frame(
        topic: &str,
        mac: &[u8; 32],
        enc: Option<&[u8; 32]>,
        kind: u8,
        sequence: u64,
        prev: Option<&[u8]>,
    ) -> Result<Vec<u8>> {
        let mut message = capnp::message::Builder::new_default();
        let mut block =
            message.init_root::<hyprstream_rpc::streaming_capnp::stream_block::Builder>();
        block.set_sequence_number(sequence);
        let payload = block.init_payloads(1).get(0);
        if let Some(enc) = enc {
            let aad = format!(
                "{topic}|identified-stream-object-v1|epoch=0|sequence={sequence}|payload=0"
            );
            let (tag, data, nonce, commitment) =
                hyprstream_rpc::crypto::event_crypto::encrypt_event(
                    enc,
                    &aad,
                    &[kind, b'x'],
                    hyprstream_rpc::crypto::event_crypto::EventPrivacy::ZeroKnowledge,
                )
                .map_err(anyhow::Error::msg)?;
            let mut tagged = payload.init_tagged();
            tagged.set_tag(&tag);
            tagged.set_payload(&data);
            tagged.set_nonce(&nonce);
            tagged.set_key_commitment(&commitment);
        } else {
            let mut payload = payload;
            payload.set_data(b"cleartext is forbidden even with a valid MAC");
        }
        let mut bytes = capnp::serialize::write_message_to_words(&message);
        let signature = keyed_mac_truncated_parts(mac, &[prev.unwrap_or(topic.as_bytes()), &bytes]);
        bytes.extend_from_slice(&signature);
        Ok(bytes)
    }

    #[test]
    fn browser_moq_rejects_plaintext_tamper_gap_and_truncation() -> Result<()> {
        let (secret, public) = generate_ephemeral_keypair();
        let ctx = hyprstream_rpc::streaming::StreamContext::from_third_party_interop_dh(
            &public.to_bytes(),
        )?;
        let fresh = || {
            BrowserStream::from_verified_info(
                info(&ctx),
                &secret.scalar().to_bytes(),
                &public.to_bytes(),
            )
        };
        let enc = ctx
            .enc_key()
            .ok_or_else(|| anyhow::anyhow!("fixture requires AEAD"))?;
        let first = frame(ctx.topic(), ctx.mac_key(), Some(enc), 0, 0, None)?;
        assert!(
            matches!(&fresh()?.verify_frame(&first)?[0], StreamPayload::Data(data) if data == b"x")
        );
        let mut bad_mac = first.clone();
        if let Some(last) = bad_mac.last_mut() {
            *last ^= 1;
        }
        assert!(fresh()?.verify_frame(&bad_mac).is_err());
        assert!(fresh()?
            .verify_frame(&frame(
                ctx.topic(),
                ctx.mac_key(),
                Some(&[77; 32]),
                0,
                0,
                None
            )?)
            .is_err());
        assert!(fresh()?
            .verify_frame(&frame(ctx.topic(), ctx.mac_key(), None, 0, 0, None)?)
            .is_err());
        assert!(fresh()?.verify_frame(&[0; 15]).is_err());
        let mut stream = fresh()?;
        stream.verify_frame(&first)?;
        assert!(stream.eof().is_err());
        let previous_mac = &first[first.len() - 16..];
        assert!(stream
            .verify_frame(&frame(
                ctx.topic(),
                ctx.mac_key(),
                Some(enc),
                1,
                2,
                Some(previous_mac)
            )?)
            .is_err());
        let error = frame(
            ctx.topic(),
            ctx.mac_key(),
            Some(enc),
            2,
            1,
            Some(previous_mac),
        )?;
        assert!(
            matches!(&stream.verify_frame(&error)?[0], StreamPayload::Error(message) if message == "x")
        );
        stream.eof()?;
        Ok(())
    }

    #[test]
    fn browser_moq_rejects_invalid_or_missing_signed_reaches() -> Result<()> {
        let (secret, public) = generate_ephemeral_keypair();
        let ctx = hyprstream_rpc::streaming::StreamContext::from_third_party_interop_dh(
            &public.to_bytes(),
        )?;
        let check = |value| {
            BrowserStream::from_verified_info(
                value,
                &secret.scalar().to_bytes(),
                &public.to_bytes(),
            )
        };
        let mut missing = info(&ctx);
        missing.announced_at.clear();
        assert!(check(missing).is_err());
        for address in [
            "",
            "model.example",
            "model.example:0",
            "user@model.example",
            "model.example/path",
            "model.example?token=x",
            "model.example\\evil",
        ] {
            let mut bad = info(&ctx);
            if let TransportConfig::Quic(reach) = &mut bad.announced_at[0].transport {
                reach.addr = address.into();
            }
            assert!(check(bad).is_err(), "accepted invalid address");
        }
        let mut bad = info(&ctx);
        bad.broadcast_path = "local/streams/wrong-topic".into();
        assert!(check(bad).is_err());
        let mut bad = info(&ctx);
        bad.broadcast_path = format!("../streams/{}", ctx.topic());
        assert!(check(bad).is_err());
        let mut bad = info(&ctx);
        bad.kem_ciphertexts = vec![1];
        assert!(check(bad).is_err());
        let mut bad = info(&ctx);
        bad.qos.completion = Completion::None;
        assert!(check(bad).is_err());
        let mut bad = info(&ctx);
        if let TransportConfig::Quic(reach) = &mut bad.announced_at[0].transport {
            reach.cert_hashes = vec![vec![1; 31]];
        }
        assert!(check(bad).is_err());
        for server_name in [
            "",
            "model.example/evil",
            "localhost",
            "localhost.",
            "child.localhost.",
            "127.0.0.1",
            "0.0.0.0",
            "[::1]",
            "user@model.example",
        ] {
            let mut bad = info(&ctx);
            if let TransportConfig::Quic(reach) = &mut bad.announced_at[0].transport {
                reach.server_name = server_name.into();
            }
            assert!(check(bad).is_err());
        }
        // A native wildcard listener advertises a loopback socket plus its
        // signed public SNI. Never dial that loopback; use only signed SNI+port.
        for address in ["192.0.2.1:8443", "127.0.0.1:8443", "0.0.0.0:8443"] {
            let mut ip_and_sni = info(&ctx);
            if let TransportConfig::Quic(reach) = &mut ip_and_sni.announced_at[0].transport {
                reach.addr = address.into();
                reach.cert_hashes = vec![vec![3; 32]];
            }
            let selected = check(ip_and_sni)?;
            assert_eq!(selected.url.as_str(), "https://model.example:8443/moq");
            assert_eq!(selected.cert_hashes, vec![vec![3; 32]]);
        }
        Ok(())
    }

    #[test]
    fn browser_moq_reuses_only_same_authority_accepted_tls_mode() -> Result<()> {
        let (secret, public) = generate_ephemeral_keypair();
        let ctx = hyprstream_rpc::streaming::StreamContext::from_third_party_interop_dh(
            &public.to_bytes(),
        )?;
        let fresh = || {
            let mut value = info(&ctx);
            if let TransportConfig::Quic(reach) = &mut value.announced_at[0].transport {
                reach.cert_hashes = vec![vec![7; 32]];
            }
            BrowserStream::from_verified_info(
                value,
                &secret.scalar().to_bytes(),
                &public.to_bytes(),
            )
        };
        let rpc = "https://model.example/hyprstream/browser-rpc";
        let mut webpki = fresh()?;
        webpki.apply_rpc_tls_policy(rpc, &[])?;
        assert!(webpki.cert_hashes.is_empty());
        let mut pinned = fresh()?;
        pinned.apply_rpc_tls_policy(rpc, &[[7; 32]])?;
        assert_eq!(pinned.cert_hashes, vec![vec![7; 32]]);
        assert!(fresh()?.apply_rpc_tls_policy(rpc, &[[8; 32]]).is_err());
        for other in [
            "https://other.example/hyprstream/browser-rpc",
            "https://model.example:8443/hyprstream/browser-rpc",
            "http://model.example/hyprstream/browser-rpc",
            "https://model.example/other",
            "https://user@model.example/hyprstream/browser-rpc",
        ] {
            assert!(fresh()?.apply_rpc_tls_policy(other, &[]).is_err());
        }
        Ok(())
    }

    #[test]
    fn browser_moq_bound_exports_do_not_open_rpc_subscribers() -> Result<()> {
        let source = include_str!("wasm_rpc_client.rs");
        let mut branches = source
            .split("if service_name == \"model\" && method_discriminator == 6 {")
            .skip(1);
        for _ in 0..2 {
            let branch = branches
                .next()
                .ok_or_else(|| anyhow::anyhow!("both bound exports required"))?
                .split("return Ok(")
                .next()
                .ok_or_else(|| anyhow::anyhow!("branch return required"))?;
            assert!(
                branch.contains("BrowserMoqHandle::open"),
                "model stream must consume MoQ, not legacy RPC SUB/PUB"
            );
            assert!(!branch.contains("open_stream_from_info"));
        }
        Ok(())
    }
}
