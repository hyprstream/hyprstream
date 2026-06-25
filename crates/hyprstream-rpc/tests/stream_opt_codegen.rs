//! Round-trip tests for the code-generated `StreamInfo`/`StreamOpt` codec (#273).
//!
//! The Cap'n Proto derive macro now generates the `ToCapnp`/`FromCapnp` impls for
//! the tagged-union QoS axes and the `StreamInfo`/`StreamOpt` structs that were
//! previously hand-written. These tests prove the generated codec is correct by
//! serializing through a real Cap'n Proto message and reading it back.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use hyprstream_rpc::capnp::{FromCapnp, ToCapnp};
use hyprstream_rpc::stream_info::{
    Completion, Delivery, IrohReach, Job, Log, Ordering, OverflowPolicy, Pipe, QuicReach,
    Destination, Retention, Role, StreamInfo, StreamOpt, StreamOptPreset, TransportConfig,
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
        // `dh_public` = server's ephemeral Ristretto255 DH public key.
        dh_public: [7u8; 32],
        qos,
        broadcast_path: "local/streams/deadbeef".to_owned(),
        // #274: exercise the native-capnp reach codec (List(struct) + union + group).
        announced_at: vec![Destination {
            role: Role::Direct,
            transport: TransportConfig::Quic(QuicReach {
                addr: "127.0.0.1:4433".to_owned(),
                server_name: "hyprstream.local".to_owned(),
                cert_hashes: vec![vec![0xABu8; 32], vec![0xCDu8; 32]],
            }),
        }],
    }
}

fn assert_roundtrips(qos: StreamOpt) {
    let info = sample_info(qos);
    let back = roundtrip(&info);
    assert_eq!(info.stream_id, back.stream_id);
    assert_eq!(info.dh_public, back.dh_public);
    assert_eq!(info.broadcast_path, back.broadcast_path);
    assert_eq!(info.announced_at, back.announced_at);
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

/// Round-trip a single wire `TransportConfig` union through capnp and back.
fn roundtrip_transport(t: &TransportConfig) -> TransportConfig {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut builder =
            message.init_root::<hyprstream_rpc::streaming_capnp::transport_config::Builder>();
        t.write_to(&mut builder);
    }
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message).expect("write_message");
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(&bytes),
        capnp::message::ReaderOptions::default(),
    )
    .expect("read_message");
    let r = reader
        .get_root::<hyprstream_rpc::streaming_capnp::transport_config::Reader>()
        .expect("get_root");
    TransportConfig::read_from(r).expect("read_from")
}

#[test]
fn transport_config_iroh_arm_roundtrips() {
    // #320: the iroh arm now carries a real IrohReach (nodeId + alpn + relay).
    let t = TransportConfig::Iroh(IrohReach {
        node_id: [0x5Au8; 32],
        alpn: "hyprstream-rpc/1".to_owned(),
        relay_url: "https://relay.example".to_owned(),
    });
    let back = roundtrip_transport(&t);
    assert_eq!(t, back);
    match back {
        TransportConfig::Iroh(i) => {
            assert_eq!(i.node_id, [0x5Au8; 32]);
            assert_eq!(i.alpn, "hyprstream-rpc/1");
            assert_eq!(i.relay_url, "https://relay.example");
        }
        other => panic!("expected Iroh, got {other:?}"),
    }
}

#[test]
fn transport_config_iroh_empty_relay_roundtrips() {
    // Empty relayUrl (= direct/pkarr, #282) survives the round-trip as empty.
    let t = TransportConfig::Iroh(IrohReach {
        node_id: [1u8; 32],
        alpn: "moql".to_owned(),
        relay_url: String::new(),
    });
    assert_eq!(t, roundtrip_transport(&t));
}

#[test]
fn transport_config_quic_arm_roundtrips() {
    let t = TransportConfig::Quic(QuicReach {
        addr: "10.0.0.1:4433".to_owned(),
        server_name: "mesh.local".to_owned(),
        cert_hashes: vec![vec![9u8; 32]],
    });
    assert_eq!(t, roundtrip_transport(&t));
}

/// #320: a `StreamInfo` whose reach carries the iroh arm round-trips end to end.
#[test]
fn stream_info_iroh_reach_roundtrips() {
    let info = StreamInfo {
        stream_id: "stream-320".to_owned(),
        dh_public: [3u8; 32],
        qos: StreamOpt::default(),
        broadcast_path: "local/streams/iroh".to_owned(),
        announced_at: vec![Destination {
            role: Role::Direct,
            transport: TransportConfig::Iroh(IrohReach {
                node_id: [0x7Bu8; 32],
                alpn: "moql".to_owned(),
                relay_url: "https://r.example".to_owned(),
            }),
        }],
    };
    let back = roundtrip(&info);
    assert_eq!(info.announced_at, back.announced_at);
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
