//! Reconstruction shard — the XET `mdb_shard` binary reconstruction format.
//!
//! A *shard* (the `.mdb` file) carries the File-Info + CAS-Info sections that
//! map a file to the ordered sequence of (xorb hash, chunk-index range, byte
//! length) segments needed to reassemble it. This module is the cas-serve
//! integration layer over [`crate::mdb_shard`]: it owns the chunk → xorb →
//! shard pipeline and exposes the helpers the binary's request handlers use.
//!
//! The shard is content-addressed by the file's merkle hash and stored at
//! `shards/{file_hash}` as the raw `.mdb` bytes. Reconstruction (`GetFile`)
//! loads the shard, walks its File-Info segments, fetches each referenced
//! xorb, slices out the chunk range, and concatenates the bytes in order.
//!
//! This is the critical-path BYTES layer for multi-xorb (i.e. >64 MiB) file
//! transfer described in #390. Unlike the earlier bespoke JSON shard, this
//! format is byte-compatible with upstream xet-core (`MDBShardFile` /
//! `FileReconstructor`), so a stock xet-core client can consume the shard
//! returned by `GetReconstructionInfo`.

use merklehash::MerkleHash;

use crate::chunker::Chunk;
use crate::mdb_shard;

/// One reconstruction segment: the xorb that holds the bytes, the chunk-index
/// range within that xorb, and the total byte length of those chunks. This is
/// a deserialized view of one mdb_shard `FileDataSequenceEntry`, exposed for
/// callers (e.g. `GetFile`) that reassemble without keeping a full xorb chunk
/// table in memory.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Xorb hash (Merkle root over the xorb's chunk hashes).
    pub xorb_hash: MerkleHash,
    /// Inclusive start chunk index within the xorb (local to the xorb).
    pub chunk_start: u32,
    /// Exclusive end chunk index within the xorb.
    pub chunk_end: u32,
    /// Total byte length of the chunk range `[chunk_start, chunk_end)`.
    pub byte_len: u64,
}

/// The reconstruction manifest for a single file: the `.mdb` bytes plus the
/// derived metadata the protocol response carries (file hash, length, xorb
/// hashes).
#[derive(Debug, Clone)]
pub struct Shard {
    /// Hex-encoded file merkle hash this shard reconstructs.
    pub file_hash: String,
    /// Total file length in bytes (sum of all segment byte lengths).
    pub file_len: u64,
    /// Hex-encoded xorb hashes the file was split across, in order.
    pub xorb_hashes: Vec<String>,
    /// The raw `.mdb` shard bytes (the XET wire format).
    pub mdb_bytes: Vec<u8>,
}

impl Shard {
    /// Build a shard from the file's chunk list, grouping chunks into xorbs
    /// that respect the XET `MAX_XORB_BYTES` (64 MiB) and `MAX_XORB_CHUNKS`
    /// (8192) limits. Returns the shard plus, for each xorb in order, the
    /// xorb hash and the concatenated xorb bytes.
    ///
    /// Chunk → xorb aggregation, xorb hashing (`merklehash::xorb_hash`), and
    /// file hashing (`merklehash::file_hash`) all match xet-core's
    /// `RawXorbData::from_chunks`, so the hashes embedded in the produced
    /// shard are identical to an upstream client's.
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
        let (builder, file_hash, xorbs) = mdb_shard::build_shard_with_cap(chunks, xorb_byte_cap);
        // `serialize` writes into a `Vec<u8>`; the only failure mode is OOM,
        // which is already a process abort — so this is effectively infallible.
        let mdb_bytes = builder
            .serialize()
            .unwrap_or_else(|e| panic!("mdb_shard serialize failed: {e}"));

        // Derive file_len + xorb_hashes directly from the inputs (they are
        // exactly what the shard encodes, and this avoids a self-parse round
        // trip that would need `expect`).
        let file_len: u64 = chunks.iter().map(|c| c.data.len() as u64).sum();
        let xorb_hashes: Vec<String> = xorbs.iter().map(|(h, _)| h.hex()).collect();
        let shard = Shard {
            file_hash: file_hash.hex(),
            file_len,
            xorb_hashes,
            mdb_bytes,
        };
        (shard, xorbs)
    }

    /// Serialize the shard to its raw `.mdb` bytes for storage/transmission.
    /// This is the XET wire format — a stock xet-core `MDBShardFile` reader can
    /// parse it.
    pub fn to_bytes(&self) -> &[u8] {
        &self.mdb_bytes
    }

    /// Parse the reconstruction segments for this file out of a `.mdb` byte
    /// buffer. Each segment tells the reassembler which xorb to fetch and which
    /// local chunk range to slice out of it.
    pub fn segments(mdb_bytes: &[u8], file_hash: &MerkleHash) -> std::io::Result<Vec<Segment>> {
        let shard = mdb_shard::MdbShard::parse(mdb_bytes)?;
        let info = shard.file_info(mdb_bytes, file_hash)?.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "file hash not in shard")
        })?;
        Ok(info
            .segments
            .into_iter()
            .map(|e| Segment {
                xorb_hash: e.cas_hash,
                chunk_start: e.chunk_index_start,
                chunk_end: e.chunk_index_end,
                byte_len: e.unpacked_segment_bytes as u64,
            })
            .collect())
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
        assert_eq!(shard.file_len, data.len() as u64);
        assert_eq!(shard.xorb_hashes.len(), 1);
        assert_eq!(shard.xorb_hashes[0], xorbs[0].0.hex());

        // The shard must parse as an mdb_shard file and yield one segment
        // spanning the whole xorb.
        let file_hash = MerkleHash::from_hex(&shard.file_hash).unwrap();
        let segs = Shard::segments(&shard.mdb_bytes, &file_hash).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].byte_len, data.len() as u64);
        assert_eq!(segs[0].chunk_start, 0);
        assert_eq!(segs[0].chunk_end as usize, chunks.len());
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

        // Reassemble from xorbs using the shard's segments.
        let file_hash = MerkleHash::from_hex(&shard.file_hash).unwrap();
        let segs = Shard::segments(&shard.mdb_bytes, &file_hash).unwrap();
        let mut rebuilt = Vec::new();
        for seg in &segs {
            let xorb = xorbs
                .iter()
                .find(|(h, _)| *h == seg.xorb_hash)
                .expect("segment references a present xorb");
            // cas-serve stores raw concatenated xorbs; the segment spans the
            // whole xorb (chunk_start=0, chunk_end=xorb chunk count), so
            // byte_len is the full xorb length.
            rebuilt.extend_from_slice(&xorb.1[..seg.byte_len as usize]);
        }
        assert_eq!(rebuilt, data);
    }
}
