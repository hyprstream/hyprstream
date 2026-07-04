//! Library weight-fetch reconstruction coverage (CPU only, no libtorch).
//!
//! Ported from the former `e2e_weight_fetch.rs`, which drove the retired
//! `cas-serve` NDJSON-over-stdio *binary* as a subprocess (#654). The wire
//! protocol (Ping/Shutdown/stdio framing) is gone, but the reconstruction
//! correctness it validated is unchanged and now lives entirely in the library:
//! [`cas_serve::CasStore`]. These tests exercise that store **directly, in
//! process** — no subprocess, no NDJSON — preserving the meaningful assertions:
//!
//! 1. Upload → reconstruct round-trip: `put_file_bytes` chunks (Gearhash CDC),
//!    aggregates xorbs, writes the `.mdb` reconstruction shard, and returns the
//!    server-computed XET merkle; `get_file_bytes` walks the shard, fetches each
//!    referenced xorb, and must reassemble the exact original bytes.
//! 2. `exists()` reports the stored file merkle (and each referenced xorb).
//! 3. Multi-xorb reconstruction: a file split across several xorbs (forced via a
//!    small per-xorb cap) still reassembles, and its shard segments sum to the
//!    original length.
//! 4. A bogus / never-stored hash surfaces a clean `StoreError::NotFound` — a
//!    missing weight is never silently mistaken for an empty file.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use cas_serve::store::PutResult;
use cas_serve::{chunker, shard::Shard, CasStore, StoreError};
use merklehash::MerkleHash;

/// Generate `len` bytes of pseudo-random data (xorshift64) so Gearhash CDC
/// actually cuts several boundaries. Constant seed → deterministic.
fn synthetic_weights(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut s: u64 = 0x9e3779b97f4a7c15;
    while out.len() + 8 <= len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&s.to_le_bytes());
    }
    if out.len() < len {
        s ^= s << 13;
        s ^= s >> 7;
        out.extend_from_slice(&s.to_le_bytes()[..len - out.len()]);
    }
    assert_eq!(out.len(), len);
    out
}

/// Steps 1-2 of the old E2E: `put_file_bytes` (CDC → xorbs → mdb_shard →
/// server-side merkle) → `get_file_bytes` round-trip, plus `exists()` for the
/// file merkle and every referenced xorb.
#[tokio::test]
async fn weight_fetch_roundtrip_lib() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = CasStore::new(dir.path());

    // 1 MiB synthetic safetensors-like weight blob (deterministic PRNG).
    const LEN: usize = 1024 * 1024;
    let original = synthetic_weights(LEN);

    // --- Upload: CDC chunks → xorbs → mdb_shard → server-computed merkle -----
    let PutResult {
        merkle,
        xorb_hashes,
        bytes_stored,
    } = store.put_file_bytes(&original).await.expect("put_file_bytes");
    assert!(!xorb_hashes.is_empty(), "upload must yield >=1 xorb");
    assert!(bytes_stored > 0, "first upload writes new payload");
    // The server-computed merkle must be a valid 64-char hex MerkleHash.
    MerkleHash::from_hex(&merkle).expect("merkle must be a valid MerkleHash");

    // --- Reconstruct: shard walk → concatenated xorbs → original bytes -------
    let data = store.get_file_bytes(&merkle).await.expect("get_file_bytes");
    assert_eq!(
        data, original,
        "get_file_bytes must reconstruct the original bytes exactly"
    );

    // exists() must report the shard (by file merkle) and every referenced xorb.
    assert!(store.exists(&merkle), "stored file merkle must exist()");
    for xh in &xorb_hashes {
        assert!(store.exists(xh), "referenced xorb {xh} must exist()");
    }
}

/// Steps 4-6 of the old E2E: multi-xorb reconstruction + the NotFound path.
///
/// `put_file_bytes` uses the spec 64 MiB per-xorb cap, so to exercise multi-xorb
/// aggregation without allocating 64+ MiB we chunk with the real CDC chunker and
/// aggregate with a small forced cap via `Shard::from_chunks_with_cap` (the same
/// builder `put_file_bytes` uses, just with a test-only cap), materialize the
/// xorbs + shard in the store's layout, then reconstruct through the store's
/// public `get_file_bytes` — walking each segment across multiple xorbs.
#[tokio::test]
async fn multi_xorb_reconstruction_lib() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = CasStore::new(dir.path());
    let storage_path = dir.path();

    // ~1 MiB of pseudo-random weights so CDC cuts many boundaries.
    const LEN: usize = 1024 * 1024;
    let original = synthetic_weights(LEN);

    // Chunk + aggregate with a tiny cap so the file spans several xorbs.
    let chunks = chunker::chunk_all(&original);
    assert!(chunks.len() > 1, "expected CDC to cut multiple chunks");
    const FORCED_CAP: usize = 256 * 1024;
    let (shard, xorbs) = Shard::from_chunks_with_cap(&chunks, FORCED_CAP);
    assert!(
        xorbs.len() > 1,
        "expected multiple xorbs with a {FORCED_CAP}-byte cap, got {}",
        xorbs.len()
    );

    // Shard segments must parse and sum to the original length (reconstruction
    // manifest integrity — the assertion the old GetReconstructionInfo made).
    let file_hash_parsed = MerkleHash::from_hex(&shard.file_hash).unwrap();
    let segments = Shard::segments(shard.to_bytes(), &file_hash_parsed)
        .expect("mdb_shard segments must parse for the file hash");
    let total: u64 = segments.iter().map(|s| s.byte_len).sum();
    assert_eq!(total, LEN as u64, "segment byte total must equal file_len");

    // Materialize the xorbs + shard in the store's on-disk layout (the same
    // layout `put_file_bytes` writes), then reconstruct via the public API.
    let xorbs_dir = storage_path.join("xorbs");
    let shards_dir = storage_path.join("shards");
    std::fs::create_dir_all(&xorbs_dir).unwrap();
    std::fs::create_dir_all(&shards_dir).unwrap();
    for (h, bytes) in &xorbs {
        std::fs::write(xorbs_dir.join(format!("default.{}", h.hex())), bytes).unwrap();
    }
    std::fs::write(shards_dir.join(&shard.file_hash), shard.to_bytes()).unwrap();

    let data = store
        .get_file_bytes(&shard.file_hash)
        .await
        .expect("get_file_bytes across multi-xorb shard");
    assert_eq!(
        data, original,
        "reconstructed multi-xorb file must match original bytes exactly"
    );

    // A never-stored hash must surface a clean NotFound (not a silent empty
    // file), so a corrupt/short shard is never mistaken for empty weights.
    let bogus = "0".repeat(64);
    match store.get_file_bytes(&bogus).await {
        Err(StoreError::NotFound(_)) => {}
        other => panic!("expected NotFound for bogus hash, got {other:?}"),
    }
}
