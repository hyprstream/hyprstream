//! End-to-end MoQ event-plane tests (#421-E5).
//!
//! These tests exercise the full moq origin → announce → subscribe path for the
//! event bus (not just the scoped-consumer logic covered by `moq_event::tests`).
//! Single-host / same-process: one `MoqEventOrigin` is initialized as the
//! process-global event bus, publishers register broadcasts against it, and
//! subscribers resolve the same origin through the public `MoqEventSubscriber`
//! API (exactly as production code does via `global_moq_event_origin()`).
//!
//! Coverage:
//!   1. `EventPublisher` (via `MoqEventOrigin::publisher_with_oid`) publishes to
//!      the per-OID track `local/events/publications/{oid_hash}` (#393).
//!   2. `EventSubscriber::subscribe_oid(oid)` receives ONLY that OID's events —
//!      verified by publishing to two distinct OIDs and asserting a subscriber
//!      scoped to OID-A never sees OID-B's events.
//!   3. Flat-track back-compat: a `publisher_with_oid` publisher mirrors to the
//!      legacy `local/events/{source}` track too.
//!   4. `BackfillMode::FirehoseBackfill` replays history before going live when a
//!      `BackfillSource` reports history; gracefully degrades to `LiveOnly` when
//!      the source has no history.
//!   5. Token-stream e2e: a mock inference publishes a sequence of token events
//!      and a subscriber receives them in order.
//!
//! Libtorch is NOT required — these are pure moq-transport tests.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;

use hyprstream_rpc::moq_event::{
    publication_broadcast_path, BackfillMode, BackfillSource, MoqEventOrigin, MoqEventSubscriber,
    EVENT_PREFIX, EVENT_TRACK, PUBLICATIONS_PREFIX,
};
use moq_net::{Path, Track};

/// One-time initialization of the process-global MoQ event origin.
///
/// `init_global_moq_event_origin` is backed by a `OnceLock`, so the first caller
/// wins and every subsequent call is a no-op. All tests in this binary share the
/// same origin tree — the same posture as a single `event` service in production.
fn shared_origin() -> MoqEventOrigin {
    static ORIGIN: OnceLock<MoqEventOrigin> = OnceLock::new();
    ORIGIN
        .get_or_init(|| {
            let origin = MoqEventOrigin::new();
            hyprstream_rpc::moq_event::init_global_moq_event_origin(origin.clone());
            origin
        })
        .clone()
}

/// Wait for a subscriber event with a generous test timeout (returns the
/// decoded `(topic, payload)` pair).
async fn recv_event(sub: &mut MoqEventSubscriber) -> Result<(String, Vec<u8>)> {
    tokio::time::timeout(Duration::from_secs(10), sub.recv())
        .await
        .map_err(|_| anyhow!("recv timeout"))?
}

/// Decode a raw event frame: `[4 bytes topic_len BE][topic UTF-8][payload bytes]`.
fn decode_event_frame(frame: &[u8]) -> (String, Vec<u8>) {
    assert!(frame.len() >= 4, "frame too short: {} bytes", frame.len());
    let topic_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
    assert!(
        frame.len() >= 4 + topic_len,
        "frame truncated: len={}, topic_len={}",
        frame.len(),
        topic_len
    );
    let topic = std::str::from_utf8(&frame[4..4 + topic_len])
        .expect("topic is valid UTF-8")
        .to_owned();
    let payload = frame[4 + topic_len..].to_vec();
    (topic, payload)
}

/// Read exactly one frame directly from a scoped origin consumer's flat
/// `local/events/{source}` broadcast — used to validate the back-compat mirror
/// path (3) without going through the `MoqEventSubscriber` pattern layer (which
/// has a pre-existing two-level-scope race unrelated to #393).
async fn read_one_flat_frame(scoped: moq_net::OriginConsumer) -> Result<(String, Vec<u8>)> {
    let mut scoped = scoped;
    loop {
        let (_path, broadcast) = match scoped.announced().await {
            None => return Err(anyhow!("origin closed before any flat broadcast")),
            Some((_, None)) => continue, // unannounce
            Some((_p, Some(b))) => (_p, b),
        };
        let mut track = broadcast
            .subscribe_track(&Track::new(EVENT_TRACK))
            .map_err(|e| anyhow!("subscribe_track on flat broadcast: {e}"))?;
        let mut group = match track.next_group().await {
            Ok(Some(g)) => g,
            Ok(None) => continue,
            Err(e) => return Err(anyhow!("next_group: {e}")),
        };
        if let Some(frame) = group.read_frame().await? {
            return Ok(decode_event_frame(&frame));
        }
    }
}

// ============================================================================
// E5.1 + E5.2 — per-OID publish via publisher_with_oid; subscribe_oid selectivity
// ============================================================================

/// A `publisher_with_oid` publisher announces its per-OID publication track at
/// `local/events/publications/{oid_hash}`, and a `subscribe_oid` subscriber
/// receives the published event end-to-end through the moq announce/subscribe
/// path. This is the core #393 / #421-E5 deliverable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn per_oid_publisher_with_oid_reaches_subscribe_oid() -> Result<()> {
    let origin = shared_origin();
    // Unique OID per test: tests share one process-global origin tree and run in
    // parallel, so two tests using the same OID would cross-subscribe.
    let oid = "at://did:web:node.example.com/models/e5-single-oid/v1";

    // Sanity: the broadcast path the publisher will announce matches what the
    // docs promise — `local/events/publications/{oid_hash}`.
    let expected_path = publication_broadcast_path(oid);
    assert!(
        expected_path.starts_with(PUBLICATIONS_PREFIX),
        "publication path must live under {PUBLICATIONS_PREFIX}: got {expected_path}"
    );

    let mut publisher = origin.publisher_with_oid("registry", oid)?;
    assert!(
        publisher.has_oid_mirror(),
        "publisher_with_oid must carry an OID mirror track"
    );

    // Start the OID-scoped subscriber (uses the public API, which resolves the
    // process-global origin at recv() time).
    let mut sub = MoqEventSubscriber::new();
    sub.subscribe_oid(oid)?;

    // Publish after subscribe_oid so the broadcast is guaranteed to be announced
    // before the subscriber's background task begins scanning.
    publisher.publish("qwen3-4b", "published", b"hello-moq")?;

    // Give the origin a moment to propagate the announcement.
    tokio::time::sleep(Duration::from_millis(30)).await;

    let (topic, payload) = recv_event(&mut sub).await?;
    assert_eq!(topic, "registry.qwen3-4b.published");
    assert_eq!(payload, b"hello-moq");

    // Hold the publisher alive so its broadcast stays announced for the
    // subscriber's lifetime; dropping it early can race the read task.
    drop(publisher);
    Ok(())
}

/// #606: `with_resume_from(seq)` skips already-delivered groups on a per-OID
/// (single-broadcast) subscription. Three events are published BEFORE the
/// subscriber starts; resuming from sequence 0 (the first published event's
/// sequence — `MoqEventPublisher` starts counting at 0) must skip exactly that
/// first event and deliver only the second and third.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_from_skips_already_seen_groups() -> Result<()> {
    let origin = shared_origin();
    let oid = "at://did:web:node.example.com/models/e6-resume-from/v1";

    let mut publisher = origin.publisher_oid_only("registry", oid)?;
    publisher.publish("a", "first", b"1")?;
    publisher.publish("a", "second", b"2")?;
    publisher.publish("a", "third", b"3")?;

    let mut sub = MoqEventSubscriber::new();
    sub.subscribe_oid(oid)?;
    sub.with_resume_from(0)?; // skip the first published group (sequence 0)

    let (_, first_seen) = recv_event(&mut sub).await?;
    let (_, second_seen) = recv_event(&mut sub).await?;
    assert_eq!(first_seen, b"2", "sequence-0 group must be skipped");
    assert_eq!(second_seen, b"3");
    assert!(
        sub.last_sequence() >= 2,
        "last_sequence must track the highest delivered group"
    );

    drop(publisher);
    Ok(())
}

/// Selectivity: a `subscribe_oid(OID-A)` subscriber receives ONLY OID-A's
/// events. When events for OID-A and OID-B are both published, the OID-A
/// subscriber sees exactly OID-A's event and never OID-B's. This proves the
/// per-OID scoping fixes the firehose problem at the wire level.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_oid_is_selective_across_two_oids() -> Result<()> {
    let origin = shared_origin();
    // Unique OIDs per test (see note in per_oid_publisher_with_oid_reaches_subscribe_oid).
    let oid_a = "at://did:web:node.example.com/models/e5-selective-a/v1";
    let oid_b = "at://did:web:node.example.com/models/e5-selective-b/v1";
    assert_ne!(oid_a, oid_b, "test OIDs must be distinct");
    assert_ne!(
        hyprstream_rpc::moq_event::oid_hash(oid_a),
        hyprstream_rpc::moq_event::oid_hash(oid_b),
        "OID hashes must be distinct"
    );

    let mut pub_a = origin.publisher_with_oid("registry", oid_a)?;
    let mut pub_b = origin.publisher_with_oid("registry", oid_b)?;

    // Subscriber scoped to OID-A only — through the public API.
    let mut sub_a = MoqEventSubscriber::new();
    sub_a.subscribe_oid(oid_a)?;

    // Publish one event to EACH OID.
    pub_a.publish("qwen3-4b", "published", b"alpha")?;
    pub_b.publish("llama3-8b", "published", b"beta")?;

    tokio::time::sleep(Duration::from_millis(30)).await;

    // The OID-A subscriber must receive alpha...
    let (topic, payload) = recv_event(&mut sub_a).await?;
    assert_eq!(topic, "registry.qwen3-4b.published");
    assert_eq!(payload, b"alpha");

    // ...and must NOT receive beta (scoped to OID-A, not the firehose).
    let next = sub_a.recv_timeout(Duration::from_millis(300)).await?;
    assert!(
        next.is_none(),
        "per-OID subscriber leaked an event for a DIFFERENT OID: {next:?}"
    );

    drop(pub_a);
    drop(pub_b);
    Ok(())
}

// ============================================================================
// E5.3 — flat-track back-compat: legacy local/events/{source} still mirrored
// ============================================================================

/// A `publisher_with_oid` publisher mirrors every event to the legacy flat
/// `local/events/{source}` track too, so legacy subscribers keep working during
/// the #393 transition. Reads the flat broadcast directly (the way the
/// production `EventSubscriber` does via `OriginConsumer::announced()` +
/// `subscribe_track`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flat_track_back_compat_mirror_with_oid_publisher() -> Result<()> {
    let origin = shared_origin();
    let oid = "at://did:web:node.example.com/models/e5-flat-compat/v1";
    // Unique source per test so the flat-track reader does not pick up events
    // from another test's `registry` source (all tests share one origin tree).
    let source = "e5-flat";

    let mut publisher = origin.publisher_with_oid(source, oid)?;

    // Scope to the flat `local/events` subtree, then to this test's source —
    // the path a legacy flat-track subscriber takes.
    let scoped = origin
        .consumer()
        .scope(&[Path::new(EVENT_PREFIX), Path::new(source)])
        .ok_or_else(|| anyhow!("origin has no flat events/{source} scope"))?;

    // Drive publish + read concurrently to defeat the announce-propagation race.
    let recv_task = tokio::spawn(read_one_flat_frame(scoped));

    // Re-publish until the reader wins (or we time out). The first successful
    // publish is enough once the broadcast is announced.
    let publish_loop = async {
        for _ in 0..50u32 {
            let _ = publisher.publish("mistral-7b", "published", b"mirrored");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        std::future::pending::<()>().await;
    };

    let (topic, payload) = tokio::select! {
        r = recv_task => r??,
        _ = publish_loop => return Err(anyhow!("publish loop ended without a reader win")),
        _ = tokio::time::sleep(Duration::from_secs(15)) => {
            return Err(anyhow!("flat-track reader timed out"));
        }
    };

    assert_eq!(topic, format!("{source}.mistral-7b.published"));
    assert_eq!(payload, b"mirrored");

    drop(publisher);
    Ok(())
}

// ============================================================================
// E5.4 — BackfillMode::FirehoseBackfill: replay-then-live, and graceful degrade
// ============================================================================

/// A simple BackfillSource used to test firehose-backfill late-join.
struct MockBackfillSource {
    /// What `has_history` reports for the OID.
    has_history: bool,
    /// History to replay (oldest-first), if `has_history` is true.
    history: Vec<(String, Vec<u8>)>,
}

impl BackfillSource for MockBackfillSource {
    fn has_history(&self, _oid: &str) -> bool {
        self.has_history
    }

    fn replay<'a>(
        &'a self,
        _oid: &'a str,
        tx: &'a mpsc::Sender<(String, Vec<u8>)>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        let history = self.history.clone();
        Box::pin(async move {
            for (topic, payload) in history {
                if tx.send((topic, payload)).await.is_err() {
                    break; // receiver dropped
                }
            }
            Ok(())
        })
    }
}

/// When the backfill source HAS history for the OID, those events are replayed
/// to the subscriber BEFORE live MoQ events arrive (cold-start, then live).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backfill_replays_history_then_live() -> Result<()> {
    let origin = shared_origin();
    // A unique OID per test to avoid colliding with other tests' per-OID tracks.
    let oid = "at://did:web:node.example.com/models/backfill-live/v1";

    let source = Arc::new(MockBackfillSource {
        has_history: true,
        history: vec![
            (
                "registry.backfill.replayed".to_owned(),
                b"backfill-1".to_vec(),
            ),
            (
                "registry.backfill.replayed".to_owned(),
                b"backfill-2".to_vec(),
            ),
        ],
    });

    let mut sub = MoqEventSubscriber::new();
    sub.subscribe_oid(oid)?;
    sub.with_backfill(BackfillMode::FirehoseBackfill {
        oid: oid.to_owned(),
        source,
    })?;

    // `publisher_oid_only` writes to the per-OID track (no flat mirror needed
    // for this backfill test).
    let mut publisher = origin.publisher_oid_only("registry", oid)?;
    publisher.publish("backfill", "live", b"live-token")?;

    // Expected order: backfill-1, backfill-2, then live.
    let e1 = recv_event(&mut sub).await?;
    assert_eq!(e1.0, "registry.backfill.replayed");
    assert_eq!(e1.1, b"backfill-1");

    let e2 = recv_event(&mut sub).await?;
    assert_eq!(e2.0, "registry.backfill.replayed");
    assert_eq!(e2.1, b"backfill-2");

    let e3 = recv_event(&mut sub).await?;
    assert_eq!(e3.0, "registry.backfill.live");
    assert_eq!(e3.1, b"live-token");

    drop(publisher);
    Ok(())
}

/// When the firehose is UNAVAILABLE (has_history=false), backfill mode
/// gracefully degrades to LiveOnly: the subscription does not fail, and live
/// MoQ events still arrive. The unreachable history must NOT leak.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backfill_degrades_to_live_when_source_unavailable() -> Result<()> {
    let origin = shared_origin();
    let oid = "at://did:web:node.example.com/models/backfill-degrade/v1";

    let source = Arc::new(MockBackfillSource {
        has_history: false, // firehose offline / no history for this OID
        history: vec![("never.should.leak".to_owned(), b"never".to_vec())],
    });

    let mut sub = MoqEventSubscriber::new();
    sub.subscribe_oid(oid)?;
    sub.with_backfill(BackfillMode::FirehoseBackfill {
        oid: oid.to_owned(),
        source,
    })?;

    let mut publisher = origin.publisher_oid_only("registry", oid)?;
    publisher.publish("degrade", "live", b"degraded-live")?;

    let got = recv_event(&mut sub).await?;
    assert_eq!(got.0, "registry.degrade.live");
    assert_eq!(got.1, b"degraded-live");
    assert_ne!(
        got.1, b"never",
        "unavailable backfill history must NOT leak"
    );

    drop(publisher);
    Ok(())
}

// ============================================================================
// E5.5 — token-stream end-to-end (mock inference → ordered token events)
// ============================================================================

/// A mock inference publishes a sequence of token events on a per-OID track,
/// and a subscriber receives them in publish order. Models the real inference
/// feedback path: the worker emits `worker.<model>.token` events per generated
/// token and the scheduler (subscribed to that OID) reconstructs the stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn token_stream_published_and_received_in_order() -> Result<()> {
    let origin = shared_origin();
    let oid = "at://did:web:node.example.com/models/token-stream-test/v1";

    // Subscriber first (so the broadcast is announced by the time we publish).
    let mut sub = MoqEventSubscriber::new();
    sub.subscribe_oid(oid)?;

    let mut publisher = origin.publisher_oid_only("worker", oid)?;
    // Let the subscriber's background task attach to the broadcast.
    tokio::time::sleep(Duration::from_millis(30)).await;

    // Publish five token events in order — the order a subscriber must observe.
    let tokens = ["Hello", ",", " world", "!", "<eos>"];
    for (i, tok) in tokens.iter().enumerate() {
        publisher.publish(
            "token-stream-test",
            "token",
            format!("{{\"seq\":{i},\"tok\":{tok:?}}}").as_bytes(),
        )?;
    }

    // Receive all five and assert order + content.
    for (i, expected_tok) in tokens.iter().enumerate() {
        let (topic, payload) = recv_event(&mut sub).await?;
        assert_eq!(
            topic, "worker.token-stream-test.token",
            "token event {i} had unexpected topic"
        );
        let payload_str = std::str::from_utf8(&payload)?;
        let expected = format!("{{\"seq\":{i},\"tok\":{expected_tok:?}}}");
        assert_eq!(
            payload_str, expected,
            "token {i}: expected {expected}, got {payload_str}"
        );
    }

    drop(publisher);
    Ok(())
}
