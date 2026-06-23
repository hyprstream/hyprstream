//! Reconstruction shard — the file reconstruction manifest.
//!
//! In the XET data model, a *shard* (the `.mdb` file) carries the File-Info +
//! CAS-Info sections that map a file to the ordered sequence of (xorb hash,
//! chunk-index range, byte length) segments needed to reassemble it. The full
//! `mdb_shard` binary format from xet-core is tightly coupled to
//! `xet_config`/`xet_runtime`; cas-serve intentionally avoids that dependency
//! tree, so this module defines an equivalent, self-describing JSON manifest.
//!
//! The manifest is content-addressed by the file's merkle hash and stored at
//! `shards/{file_hash}`. Reconstruction (`GetFile`) loads the shard, then for
//! each segment fetches the referenced xorb and slices out the chunk range,
//! concatenating the bytes in order. This is the critical-path BYTES layer for
//! multi-xorb (i.e. >64 MiB) file transfer described in #390.

use merklehash::MerkleHash;
use serde::{Deserialize, Serialize};

use crate::chunker::Chunk;

/// One reconstruction segment: the xorb that holds the bytes, the inclusive
/// chunk-index range within that xorb, and the total byte length of those
/// chunks. Reassembling the file means concatenating the chunk bytes for each
/// segment's range, in segment order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Hex-encoded xorb hash (Merkle root over the xorb's chunk hashes).
    pub xorb_hash: String,
    /// Inclusive start chunk index within the xorb.
    pub chunk_start: u32,
    /// Inclusive end chunk index within the xorb.
    pub chunk_end: u32,
    /// Total byte length of the chunk range `[chunk_start, chunk_end]`.
    pub byte_len: u64,
}

/// The reconstruction manifest for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    /// Hex-encoded file merkle hash this shard reconstructs.
    pub file_hash: String,
    /// Total file length in bytes (sum of all `Segment::byte_len`).
    pub file_len: u64,
    /// Ordered segments; concatenating their bytes reproduces the file.
    pub segments: Vec<Segment>,
}

impl Shard {
    /// Build a shard from the file's chunk list, grouping chunks into xorbs
    /// that respect the XET `MAX_XORB_BYTES` (64 MiB) and `MAX_XORB_CHUNKS`
    /// (8192) limits. Returns the shard plus, for each xorb in order, the
    /// xorb hash and the concatenated xorb bytes.
    ///
    /// Chunk hashes/lengths are fed to `merklehash::xorb_hash` to derive the
    /// xorb's Merkle-root hash, exactly as xet-core's `RawXorbData::from_chunks`
    /// does (`deduplication/src/raw_xorb_data.rs`).
    pub fn from_chunks(chunks: &[Chunk]) -> (Shard, Vec<(MerkleHash, Vec<u8>)>) {
        Self::from_chunks_with_cap(chunks, crate::chunker::MAX_XORB_BYTES)
    }

    /// Like [`from_chunks`], but with a custom per-xorb byte cap. Used by tests
    /// to force multi-xorb aggregation without allocating 64 MiB. The chunk
    /// count cap (`MAX_XORB_CHUNKS` = 8192) still applies.
    pub fn from_chunks_with_cap(
        chunks: &[Chunk],
        xorb_byte_cap: usize,
    ) -> (Shard, Vec<(MerkleHash, Vec<u8>)>) {
        use crate::chunker::MAX_XORB_CHUNKS;

        let mut segments: Vec<Segment> = Vec::new();
        let mut xorbs: Vec<(MerkleHash, Vec<u8>)> = Vec::new();
        let mut file_len: u64 = 0;

        let mut i = 0;
        while i < chunks.len() {
            // Greedily pack chunks into one xorb up to the byte/chunk limits.
            let mut xorb_bytes: u64 = 0;
            let mut xorb_chunks: Vec<&Chunk> = Vec::new();
            let start = i;
            while i < chunks.len() {
                let c = &chunks[i];
                let new_bytes = xorb_bytes + c.data.len() as u64;
                if !xorb_chunks.is_empty()
                    && (new_bytes > xorb_byte_cap as u64 || xorb_chunks.len() >= MAX_XORB_CHUNKS)
                {
                    break;
                }
                xorb_bytes = new_bytes;
                file_len += c.data.len() as u64;
                xorb_chunks.push(c);
                i += 1;
            }
            let end = i - 1;

            // Xorb hash = Merkle root over (chunk_hash, chunk_len).
            let hash_inputs: Vec<(MerkleHash, u64)> = xorb_chunks
                .iter()
                .map(|c| (c.hash, c.data.len() as u64))
                .collect();
            let xorb_hash = merklehash::xorb_hash(&hash_inputs);

            // Xorb bytes = chunks concatenated (no compression here; cas-serve
            // stores raw xorbs).
            let mut bytes = Vec::with_capacity(xorb_bytes as usize);
            for c in &xorb_chunks {
                bytes.extend_from_slice(&c.data);
            }

            segments.push(Segment {
                xorb_hash: xorb_hash.hex(),
                chunk_start: start as u32,
                chunk_end: end as u32,
                byte_len: xorb_bytes,
            });
            xorbs.push((xorb_hash, bytes));
        }

        // The file merkle hash is computed over all chunk (hash, len) pairs,
        // matching `merklehash::file_hash`.
        let file_hash_inputs: Vec<(MerkleHash, u64)> = chunks
            .iter()
            .map(|c| (c.hash, c.data.len() as u64))
            .collect();
        let file_hash = merklehash::file_hash(&file_hash_inputs);

        let shard = Shard {
            file_hash: file_hash.hex(),
            file_len,
            segments,
        };
        (shard, xorbs)
    }

    /// Serialize the shard to JSON for storage/transmission.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize a shard from JSON.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::chunker::chunk_all;

    #[test]
    fn single_xorb_shard() {
        let data = vec![0x42u8; 200_000];
        let chunks = chunk_all(&data);
        let (shard, xorbs) = Shard::from_chunks(&chunks);
        assert_eq!(xorbs.len(), 1, "200KB fits in a single 64MiB xorb");
        assert_eq!(shard.segments.len(), 1);
        assert_eq!(shard.file_len, data.len() as u64);
        assert_eq!(shard.segments[0].byte_len, data.len() as u64);
        assert_eq!(shard.segments[0].xorb_hash, xorbs[0].0.hex());
        assert_eq!(shard.segments[0].chunk_start, 0);
        assert_eq!(shard.segments[0].chunk_end, (chunks.len() - 1) as u32);
    }

    #[test]
    fn round_trip_reassembly_matches_original() {
        // Pseudo-random data so CDC cuts multiple chunks.
        let mut data = Vec::with_capacity(300_000);
        let mut s: u64 = 42;
        for _ in 0..37500 {
            s = s.wrapping_mul(0x94d049bb133111eb) ^ (s >> 31);
            data.extend_from_slice(&s.to_le_bytes());
        }
        let chunks = chunk_all(&data);
        assert!(chunks.len() > 1, "expected multiple chunks");
        let (shard, xorbs) = Shard::from_chunks(&chunks);

        // Reassemble from xorbs using the shard.
        let mut rebuilt = Vec::new();
        for seg in &shard.segments {
            let xorb = xorbs
                .iter()
                .find(|(h, _)| h.hex() == seg.xorb_hash)
                .expect("segment references a present xorb");
            rebuilt.extend_from_slice(&xorb.1[..seg.byte_len as usize]);
        }
        assert_eq!(rebuilt, data);
    }
}
