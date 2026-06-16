//! Round-trip tests for the code-generated `StreamInfo`/`StreamOpt` codec (#273).
//!
//! The Cap'n Proto derive macro now generates the `ToCapnp`/`FromCapnp` impls for
//! the tagged-union QoS axes and the `StreamInfo`/`StreamOpt` structs that were
//! previously hand-written. These tests prove the generated codec is correct by
//! serializing through a real Cap'n Proto message and reading it back.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use hyprstream_rpc::capnp::{FromCapnp, ToCapnp};
use hyprstream_rpc::stream_info::{
    Completion, Delivery, Job, Log, Ordering, OverflowPolicy, Pipe, Retention, StreamInfo,
    StreamOpt, StreamOptPreset,
};

/// Serialize a `StreamInfo` through a Cap'n Proto message and read it back.
fn roundtrip(info: &StreamInfo) -> StreamInfo {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut builder =
            message.init_root::<hyprstream_rpc::streaming_capnp::stream_info::Builder>();
        info.write_to(&mut builder);
    }

    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message).expect("write_message");

    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(&bytes),
        capnp::message::ReaderOptions::default(),
    )
    .expect("read_message");
    let si_reader = reader
        .get_root::<hyprstream_rpc::streaming_capnp::stream_info::Reader>()
        .expect("get_root");
    StreamInfo::read_from(si_reader).expect("read_from")
}

fn sample_info(qos: StreamOpt) -> StreamInfo {
    StreamInfo {
        stream_id: "stream-273".to_owned(),
        endpoint: "".to_owned(),
        // `dh_public` = server's ephemeral Ristretto255 DH public key.
        dh_public: [7u8; 32],
        qos,
        moq_uds_path: "/tmp/moq.sock".to_owned(),
        moq_broadcast_path: "local/streams/deadbeef".to_owned(),
    }
}

fn assert_roundtrips(qos: StreamOpt) {
    let info = sample_info(qos);
    let back = roundtrip(&info);
    assert_eq!(info.stream_id, back.stream_id);
    assert_eq!(info.endpoint, back.endpoint);
    assert_eq!(info.dh_public, back.dh_public);
    assert_eq!(info.moq_uds_path, back.moq_uds_path);
    assert_eq!(info.moq_broadcast_path, back.moq_broadcast_path);
    assert_eq!(info.qos, back.qos);
}

#[test]
fn default_qos_roundtrips() {
    assert_roundtrips(StreamOpt::default());
}

#[test]
fn presets_roundtrip() {
    assert_roundtrips(Job::stream_opt());
    assert_roundtrips(Log::stream_opt());
    assert_roundtrips(Pipe::stream_opt());
}

#[test]
fn non_default_arms_roundtrip() {
    // Each axis exercised with a non-@0 arm, including all three group arms,
    // both scalar Retention arms, and the Void Completion::None arm.
    let qos = StreamOpt {
        ordering: Ordering::Unordered { anti_replay_window: 7 },
        delivery: Delivery::AtLeastOnce { dedup_window: 4096, resumable: true },
        completion: Completion::None,
        retention: Retention::Blocks(256),
        overflow_policy: OverflowPolicy::DropOldest { high_water_mark: 1024 },
    };
    assert_roundtrips(qos);

    // Retention::Seconds is the other scalar arm.
    assert_roundtrips(StreamOpt {
        retention: Retention::Seconds(300),
        ..StreamOpt::default()
    });
}

#[test]
fn group_arm_field_values_preserved() {
    // Explicitly assert the group-arm leaf values survive the round-trip
    // (not just structural equality), since group arms are the new codegen path.
    let info = sample_info(StreamOpt {
        ordering: Ordering::Unordered { anti_replay_window: 42 },
        delivery: Delivery::AtLeastOnce { dedup_window: 99, resumable: false },
        overflow_policy: OverflowPolicy::DropOldest { high_water_mark: 8 },
        ..StreamOpt::default()
    });
    let back = roundtrip(&info);

    match back.qos.ordering {
        Ordering::Unordered { anti_replay_window } => assert_eq!(anti_replay_window, 42),
        other => panic!("expected Unordered, got {other:?}"),
    }
    match back.qos.delivery {
        Delivery::AtLeastOnce { dedup_window, resumable } => {
            assert_eq!(dedup_window, 99);
            assert!(!resumable);
        }
        other => panic!("expected AtLeastOnce, got {other:?}"),
    }
    match back.qos.overflow_policy {
        OverflowPolicy::DropOldest { high_water_mark } => assert_eq!(high_water_mark, 8),
        other => panic!("expected DropOldest, got {other:?}"),
    }
}
