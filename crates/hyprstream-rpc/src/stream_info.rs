//! Target-independent StreamInfo + StreamOpt types.
//!
//! Extracted from `streaming` (native-only) so that generated client code
//! on wasm32 can reference `StreamInfo` without pulling in ZMQ/tokio deps.
//!
//! Service codegen modules emit `pub type StreamInfo = hyprstream_rpc::stream_info::StreamInfo;`
//! (and similar aliases for the StreamOpt axis types) rather than generating duplicates.
//!
//! #273: `StreamInfo`, `StreamOpt`, and the five QoS axis unions are now fully
//! code-generated from `streaming.capnp` into [`crate::streaming_types`] (the
//! Cap'n Proto derive macro supports tagged-unions + data-carrying `group` arms).
//! This module re-exports them as the canonical hub and keeps the hand-authored
//! `Job`/`Log`/`Pipe` presets, which encode specific `StreamOpt` combinations.

// ─── StreamOpt axis types ──────────────────────────────────────────────────────
//
// Code-generated (#273). The generated enums/struct match the historic
// hand-written shapes exactly, so the presets and all call sites compile unchanged:
//   Ordering::{Ordered, Unordered{anti_replay_window}}
//   Delivery::{AtMostOnce, AtLeastOnce{dedup_window, resumable}}
//   Completion::{EndOfStream, None}
//   Retention::{Live, Blocks(u32), Seconds(u32)}
//   OverflowPolicy::{Block, DropOldest{high_water_mark}}
//   StreamOpt { ordering, delivery, completion, retention, overflow_policy }
//   StreamInfo { stream_id, dh_public: [u8; 32], qos, broadcast_path, reach }
//
// #274: the moq reach model — `StreamInfo.reach: Vec<Destination>` plus the
// `Role` / `TransportConfig` / `QuicReach` types — is also code-generated here
// from `streaming.capnp` (native capnp, no JSON-in-Text on the wire).
pub use crate::streaming_types::{
    Completion, Delivery, Destination, IrohReach, Ordering, OverflowPolicy, QuicReach, Retention,
    Role, StreamInfo, StreamOpt, TransportConfig,
};

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
#[derive(Debug, Clone, Copy, Default)]
pub struct Job;

/// Log / notification streams.
///
/// Ordered, at-least-once, no terminator sentinel, 300-second relay
/// retention. Use for Metrics, Notification, and mobile clients with
/// intermittent connectivity (#169).
#[derive(Debug, Clone, Copy, Default)]
pub struct Log;

/// Pipe / console streams.
///
/// Ordered, at-least-once with 256-entry dedup, live retention, no
/// terminator. Use for container I/O attach and the model↔worker callback
/// DEALER replacement (#170).
#[derive(Debug, Clone, Copy, Default)]
pub struct Pipe;

impl StreamOptPreset for Job {
    fn stream_opt() -> StreamOpt {
        StreamOpt {
            ordering: Ordering::Ordered,
            delivery: Delivery::AtLeastOnce {
                dedup_window: 4096,
                resumable: true,
            },
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
            delivery: Delivery::AtLeastOnce {
                dedup_window: 4096,
                resumable: true,
            },
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
            delivery: Delivery::AtLeastOnce {
                dedup_window: 256,
                resumable: false,
            },
            completion: Completion::None,
            retention: Retention::Live,
            overflow_policy: OverflowPolicy::Block,
        }
    }
}

/// Default event-bus delivery (#606).
///
/// At-most-once, drop-oldest-on-backpressure, live retention, no terminator.
/// Matches the historic `moq_event` "fan-out, at-most-once, best-effort"
/// behaviour, but BOUNDED: a slow subscriber's local queue evicts its oldest
/// entry instead of blocking the publisher-side read tasks indefinitely. Use
/// for lifecycle/telemetry events where the latest state matters more than
/// every historical transition (e.g. `worker.*.heartbeat`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EventLive;

/// Reliable event-bus delivery (#606).
///
/// At-least-once with a 1024-entry dedup window, resumable from last-acked
/// sequence, retains the last 64 delivery blocks, lossless (block) backpressure.
/// Use for events that must not be silently dropped, e.g. `model.loaded` /
/// `model.loadFailed` and other completion/lifecycle-latch signals (see the
/// EventService consolidation epic, EV7).
#[derive(Debug, Clone, Copy, Default)]
pub struct EventReliable;

impl StreamOptPreset for EventLive {
    fn stream_opt() -> StreamOpt {
        StreamOpt {
            ordering: Ordering::Ordered,
            delivery: Delivery::AtMostOnce,
            completion: Completion::None,
            retention: Retention::Live,
            overflow_policy: OverflowPolicy::DropOldest {
                high_water_mark: 256,
            },
        }
    }
}

impl StreamOptPreset for EventReliable {
    fn stream_opt() -> StreamOpt {
        StreamOpt {
            ordering: Ordering::Ordered,
            delivery: Delivery::AtLeastOnce {
                dedup_window: 1024,
                resumable: true,
            },
            completion: Completion::None,
            retention: Retention::Blocks(64),
            overflow_policy: OverflowPolicy::Block,
        }
    }
}
