//! Target-independent StreamInfo + StreamOpt types.
//!
//! Extracted from `streaming` (native-only) so that generated client code
//! on wasm32 can reference `StreamInfo` without pulling in ZMQ/tokio deps.
//!
//! Service codegen modules emit `pub type StreamInfo = hyprstream_rpc::stream_info::StreamInfo;`
//! (and similar aliases for the StreamOpt axis types) rather than generating duplicates.

// ─── StreamOpt axis types ──────────────────────────────────────────────────────

/// Ordered delivery — gaps are fatal (DTLS anti-replay, RFC 9147 §4.2.3).
/// Unordered delivery with an anti-replay window (SRTP RFC 3711 §3.3).
///
/// `@0 = ordered` is the fail-closed default: capnp zero-fills an absent field.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Ordering {
    #[default]
    Ordered,
    Unordered {
        /// Reject block with seq ≤ (highest-seen − antiReplayWindow).
        anti_replay_window: u32,
    },
}

/// Delivery guarantee aligned with MQTT QoS 0/1 (OASIS MQTT 5.0 §4.3).
/// exactlyOnce (QoS 2) is intentionally out of scope.
///
/// `@0 = atMostOnce` is the fail-closed default.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Delivery {
    #[default]
    AtMostOnce,
    AtLeastOnce {
        /// Number of recent sequence numbers remembered for client-side dedup.
        dedup_window: u32,
        /// Resume delivery from last-acked seq after reconnect.
        resumable: bool,
    },
}

/// Stream termination policy. `endOfStream` requires a `complete`/`error`
/// StreamPayload before EOF; absence is a truncation and MUST be rejected.
/// Analogous to gRPC END_STREAM / HTTP/2 DATA+END_STREAM / WebTransport FIN.
///
/// `@0 = endOfStream` is the fail-closed default.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Completion {
    #[default]
    EndOfStream,
    None,
}

/// Relay-side retention / late-join buffer policy.
/// `live` is the smallest attack surface (no buffering beyond the live window).
/// NOTE: `live` is declared by the qos but enforced by the client (via epoch
/// binding), not the relay (which is blind to MAC keys).
///
/// `@0 = live` is the fail-closed default.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Retention {
    #[default]
    Live,
    /// Retain N delivery blocks (~N MoQ Transport Groups).
    Blocks(u32),
    /// Retain for N wall-clock seconds.
    Seconds(u32),
}

/// Application-layer publish-path overflow policy.
/// `block` = lossless EAGAIN-style backpressure on the publisher.
/// NOTE: distinct from QUIC flow control (RFC 9000 §4) which operates at the
/// transport layer independently.
///
/// `@0 = block` is the fail-closed (lossless) default.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum OverflowPolicy {
    #[default]
    Block,
    DropOldest {
        high_water_mark: u32,
    },
}

/// Full per-stream QoS options. Carried in the signed StreamInfo handshake
/// response so the contract is Ed25519-authenticated and wire-visible.
///
/// All axes default to their fail-closed (`@0`) variants per the schema
/// discipline — safe under capnp zero-fill of absent/unknown discriminants.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamOpt {
    pub ordering: Ordering,
    pub delivery: Delivery,
    pub completion: Completion,
    pub retention: Retention,
    pub overflow_policy: OverflowPolicy,
}

/// Marker trait for typed stream QoS presets.
///
/// Each preset is a zero-sized type that encodes a specific `StreamOpt`
/// configuration, enabling type-safe API contracts without runtime dispatch.
/// Implement this trait to define a new named preset.
pub trait StreamOptPreset: Clone + Send + Sync + 'static {
    fn stream_opt() -> StreamOpt;
}

/// Inference / job streams.
///
/// Ordered, at-least-once with 4096-entry dedup window, resumable from
/// last-acked Group, EndOfStream terminator, 256-Group relay retention,
/// lossless backpressure. Use for InferenceService token streams and model
/// lifecycle events (#169).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Job;

/// Log / notification streams.
///
/// Ordered, at-least-once, no terminator sentinel, 300-second relay
/// retention. Use for Metrics, Notification, and mobile clients with
/// intermittent connectivity (#169).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Log;

/// Pipe / console streams.
///
/// Ordered, at-least-once with 256-entry dedup, live retention, no
/// terminator. Use for container I/O attach and the model↔worker callback
/// DEALER replacement (#170).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Pipe;

impl StreamOptPreset for Job {
    fn stream_opt() -> StreamOpt {
        StreamOpt {
            ordering: Ordering::Ordered,
            delivery: Delivery::AtLeastOnce { dedup_window: 4096, resumable: true },
            completion: Completion::EndOfStream,
            retention: Retention::Blocks(256),
            overflow_policy: OverflowPolicy::Block,
        }
    }
}

impl StreamOptPreset for Log {
    fn stream_opt() -> StreamOpt {
        StreamOpt {
            ordering: Ordering::Ordered,
            delivery: Delivery::AtLeastOnce { dedup_window: 4096, resumable: true },
            completion: Completion::None,
            retention: Retention::Seconds(300),
            overflow_policy: OverflowPolicy::Block,
        }
    }
}

impl StreamOptPreset for Pipe {
    fn stream_opt() -> StreamOpt {
        StreamOpt {
            ordering: Ordering::Ordered,
            delivery: Delivery::AtLeastOnce { dedup_window: 256, resumable: false },
            completion: Completion::None,
            retention: Retention::Live,
            overflow_policy: OverflowPolicy::Block,
        }
    }
}

// ─── StreamInfo ───────────────────────────────────────────────────────────────

/// Canonical stream info returned by streaming RPC methods.
///
/// Wrapped in a `SignedEnvelope` (Ed25519) so the `qos` field is authenticated.
/// Service codegen modules alias this type from `hyprstream_rpc::stream_info`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamInfo {
    pub stream_id: String,
    pub endpoint: String,
    pub server_pubkey: [u8; 32],
    pub qos: StreamOpt,
}

impl crate::capnp::ToCapnp for StreamInfo {
    type Builder<'a> = crate::streaming_capnp::stream_info::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_stream_id(&self.stream_id);
        builder.set_endpoint(&self.endpoint);
        builder.set_dh_public(&self.server_pubkey);
        let mut qos_builder = builder.reborrow().init_qos();
        write_stream_qos(&self.qos, &mut qos_builder);
    }
}

impl crate::capnp::FromCapnp for StreamInfo {
    type Reader<'a> = crate::streaming_capnp::stream_info::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> anyhow::Result<Self> {
        let pubkey_data = reader.get_dh_public()?;
        let mut server_pubkey = [0u8; 32];
        if pubkey_data.len() == 32 {
            server_pubkey.copy_from_slice(pubkey_data);
        }
        let qos = if reader.has_qos() {
            read_stream_qos(reader.get_qos()?)?
        } else {
            StreamOpt::default()
        };
        Ok(Self {
            stream_id: reader.get_stream_id()?.to_str()?.to_owned(),
            endpoint: reader.get_endpoint()?.to_str()?.to_owned(),
            server_pubkey,
            qos,
        })
    }
}

// ─── capnp serialization helpers (not pub — internal to this module) ─────────

fn write_stream_qos(
    qos: &StreamOpt,
    builder: &mut crate::streaming_capnp::stream_opt::Builder<'_>,
) {
    // ordering
    {
        let mut b = builder.reborrow().init_ordering();
        match &qos.ordering {
            Ordering::Ordered => b.set_ordered(()),
            Ordering::Unordered { anti_replay_window } => {
                b.init_unordered().set_anti_replay_window(*anti_replay_window);
            }
        }
    }
    // delivery
    {
        let mut b = builder.reborrow().init_delivery();
        match &qos.delivery {
            Delivery::AtMostOnce => b.set_at_most_once(()),
            Delivery::AtLeastOnce { dedup_window, resumable } => {
                let mut g = b.init_at_least_once();
                g.set_dedup_window(*dedup_window);
                g.set_resumable(*resumable);
            }
        }
    }
    // completion
    {
        let mut b = builder.reborrow().init_completion();
        match &qos.completion {
            Completion::EndOfStream => b.set_end_of_stream(()),
            Completion::None => b.set_none(()),
        }
    }
    // retention
    {
        let mut b = builder.reborrow().init_retention();
        match &qos.retention {
            Retention::Live => b.set_live(()),
            Retention::Blocks(n) => b.set_blocks(*n),
            Retention::Seconds(n) => b.set_seconds(*n),
        }
    }
    // overflow_policy
    {
        let mut b = builder.reborrow().init_overflow_policy();
        match &qos.overflow_policy {
            OverflowPolicy::Block => b.set_block(()),
            OverflowPolicy::DropOldest { high_water_mark } => {
                b.init_drop_oldest().set_high_water_mark(*high_water_mark);
            }
        }
    }
}

fn read_stream_qos(
    reader: crate::streaming_capnp::stream_opt::Reader<'_>,
) -> anyhow::Result<StreamOpt> {
    use crate::streaming_capnp::{
        completion, delivery, ordering, overflow_policy, retention,
    };

    let ordering = match reader.get_ordering()?.which()? {
        ordering::Which::Ordered(()) => Ordering::Ordered,
        ordering::Which::Unordered(g) => Ordering::Unordered {
            anti_replay_window: g.get_anti_replay_window(),
        },
    };

    let delivery = match reader.get_delivery()?.which()? {
        delivery::Which::AtMostOnce(()) => Delivery::AtMostOnce,
        delivery::Which::AtLeastOnce(g) => Delivery::AtLeastOnce {
            dedup_window: g.get_dedup_window(),
            resumable: g.get_resumable(),
        },
    };

    let completion = match reader.get_completion()?.which()? {
        completion::Which::EndOfStream(()) => Completion::EndOfStream,
        completion::Which::None(()) => Completion::None,
    };

    let retention = match reader.get_retention()?.which()? {
        retention::Which::Live(()) => Retention::Live,
        retention::Which::Blocks(n) => Retention::Blocks(n),
        retention::Which::Seconds(n) => Retention::Seconds(n),
    };

    let overflow_policy = match reader.get_overflow_policy()?.which()? {
        overflow_policy::Which::Block(()) => OverflowPolicy::Block,
        overflow_policy::Which::DropOldest(g) => OverflowPolicy::DropOldest {
            high_water_mark: g.get_high_water_mark(),
        },
    };

    Ok(StreamOpt {
        ordering,
        delivery,
        completion,
        retention,
        overflow_policy,
    })
}
