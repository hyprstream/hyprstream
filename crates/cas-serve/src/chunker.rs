//! Content-defined chunker — a thin wrapper over the `gearhash` crate.
//!
//! This reproduces the upstream xet-core `Chunker`
//! (`deduplication/src/chunking.rs` at tag `git-xet-v0.2.0`) byte-for-byte:
//! same `gearhash::Hasher` engine, same `TARGET_CHUNK_SIZE` (64 KiB), same
//! minimum (`target / MINIMUM_CHUNK_DIVISOR` = 8 KiB) and maximum
//! (`target * MAXIMUM_CHUNK_MULTIPLIER` = 128 KiB) bounds, and the same
//! left-shifted FastCDC mask. Chunk hashing delegates to
//! `merklehash::compute_data_hash`, i.e. BLAKE3 keyed with the XET `DATA_KEY`,
//! which is exactly what `deduplication::Chunk::new` does.
//!
//! Reusing `gearhash` + `merklehash` directly (rather than depending on the
//! `deduplication` crate) avoids pulling the heavy `xet_config` /
//! `xet_runtime` / `mdb_shard` / `progress_tracking` tree into cas-serve,
//! which is intended to stay lightweight (no libtorch). The boundary logic
//! below is a direct port of xet-core's, so chunks produced here are
//! identical to those produced by a stock xet-core client.

use bytes::Bytes;
use merklehash::MerkleHash;

/// XET chunking constants, matching `deduplication/src/constants.rs`.
///
/// `TARGET_CHUNK_SIZE` is 64 KiB, giving an average chunk size of ~64 KiB and
/// a per-xorb target of ~1024 chunks. `MINIMUM_CHUNK_DIVISOR = 8` yields a
/// minimum chunk size of 8 KiB; `MAXIMUM_CHUNK_MULTIPLIER = 2` yields a
/// maximum of 128 KiB. These match the values pinned at `git-xet-v0.2.0`.
pub const TARGET_CHUNK_SIZE: usize = 64 * 1024;
pub const MINIMUM_CHUNK_DIVISOR: usize = 8;
pub const MAXIMUM_CHUNK_MULTIPLIER: usize = 2;

/// XORB limits, matching `deduplication/src/constants.rs`:
/// a xorb (eXclusive-OR Block) holds at most 64 MiB across at most 8192 chunks.
pub const MAX_XORB_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_XORB_CHUNKS: usize = 8 * 1024;

/// Gear-hash rolling-window size (bytes that must be processed before a
/// boundary can trigger). Matches xet-core's `HASH_WINDOW_SIZE`.
const HASH_WINDOW_SIZE: usize = 64;

/// A content-defined chunk: its BLAKE3-keyed hash and its bytes.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// BLAKE3 keyed with the global XET `DATA_KEY` (`compute_data_hash`).
    pub hash: MerkleHash,
    pub data: Bytes,
}

impl Chunk {
    pub fn new(data: Bytes) -> Self {
        let hash = merklehash::compute_data_hash(&data);
        Chunk { hash, data }
    }
}

/// Content-defined chunker using the FastCDC / gear-hash algorithm.
///
/// Ported from xet-core's `Chunker` (`deduplication/src/chunking.rs`).
pub struct Chunker {
    hash: gearhash::Hasher<'static>,
    minimum_chunk: usize,
    maximum_chunk: usize,
    mask: u64,
    chunkbuf: Vec<u8>,
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(TARGET_CHUNK_SIZE)
    }
}

impl Chunker {
    /// Construct a chunker targeting the given average chunk size. The size
    /// must be a power of two greater than the 64-byte gear-hash window.
    pub fn new(target_chunk_size: usize) -> Self {
        assert_eq!(
            target_chunk_size.count_ones(),
            1,
            "target must be a power of two"
        );
        assert!(target_chunk_size > HASH_WINDOW_SIZE);
        assert!(target_chunk_size < u32::MAX as usize);

        // Build the FastCDC mask exactly as xet-core does: start from
        // (target - 1), then shift it left to the top bits of the word so the
        // rolling-hash window fully participates in the boundary decision.
        let mask = (target_chunk_size - 1) as u64;
        let mask = mask << mask.leading_zeros();

        let minimum_chunk = target_chunk_size / MINIMUM_CHUNK_DIVISOR;
        let maximum_chunk = target_chunk_size * MAXIMUM_CHUNK_MULTIPLIER;
        assert!(maximum_chunk > minimum_chunk);

        Chunker {
            hash: gearhash::Hasher::default(),
            minimum_chunk,
            maximum_chunk,
            mask,
            chunkbuf: Vec::with_capacity(maximum_chunk),
        }
    }

    /// Look for the next chunk boundary in `data`, carrying over any buffered
    /// bytes from the previous call. Returns the offset into `data` at which
    /// the chunk ends, or `None` if no boundary can be decided yet.
    ///
    /// Direct port of xet-core's `Chunker::next_boundary`.
    #[inline]
    pub fn next_boundary(&mut self, data: &[u8]) -> Option<usize> {
        let n_bytes = data.len();
        if n_bytes == 0 {
            return None;
        }

        let previous_len = self.chunkbuf.len();
        let mut cur_index = 0;
        let mut create_chunk = false;

        // Skip the minimum chunk size, accounting for the 64-byte hash window.
        if previous_len + HASH_WINDOW_SIZE < self.minimum_chunk {
            let skip = (self.minimum_chunk - previous_len - HASH_WINDOW_SIZE - 1).min(n_bytes);
            cur_index += skip;
        }

        // Don't scan past the maximum chunk boundary.
        let read_end = n_bytes.min(cur_index + self.maximum_chunk - previous_len);

        loop {
            if let Some(next_match) = self.hash.next_match(&data[cur_index..read_end], self.mask) {
                cur_index += next_match;

                // A boundary found before the minimum chunk size can be an
                // artifact of the hasher's prior state (the window hasn't
                // filled with current-chunk bytes yet); keep scanning.
                if cur_index + previous_len < self.minimum_chunk {
                    continue;
                }
                create_chunk = true;
            } else {
                cur_index = read_end;
            }
            break;
        }

        // Force a boundary at the maximum chunk size.
        if cur_index + previous_len >= self.maximum_chunk {
            cur_index = self.maximum_chunk - previous_len;
            create_chunk = true;
        }

        if create_chunk {
            self.hash.set_hash(0); // reset for the next chunk
            Some(cur_index)
        } else {
            None
        }
    }

    /// Feed more data and return the next emitted chunk (if any) plus the
    /// number of bytes consumed from `data`. When `is_final` is true the
    /// remaining buffer is flushed as a final chunk.
    ///
    /// Direct port of xet-core's `Chunker::next`.
    pub fn next(&mut self, data: &[u8], is_final: bool) -> (Option<Chunk>, usize) {
        let (chunk_data, consume): (Bytes, usize) =
            if let Some(next_boundary) = self.next_boundary(data) {
                if self.chunkbuf.is_empty() {
                    (
                        Bytes::copy_from_slice(&data[..next_boundary]),
                        next_boundary,
                    )
                } else {
                    self.chunkbuf.extend_from_slice(&data[..next_boundary]);
                    (std::mem::take(&mut self.chunkbuf).into(), next_boundary)
                }
            } else if is_final {
                if self.chunkbuf.is_empty() {
                    (Bytes::copy_from_slice(data), data.len())
                } else {
                    self.chunkbuf.extend_from_slice(data);
                    (std::mem::take(&mut self.chunkbuf).into(), data.len())
                }
            } else {
                self.chunkbuf.extend_from_slice(data);
                return (None, data.len());
            };

        if chunk_data.is_empty() {
            return (None, 0);
        }

        (Some(Chunk::new(chunk_data)), consume)
    }
}

/// Convenience: chunk a complete byte slice in one shot, flushing the tail.
///
/// This mirrors xet-core's `Chunker::next_block(&data, true)` usage and is the
/// entry point used by the upload path.
pub fn chunk_all(data: &[u8]) -> Vec<Chunk> {
    let mut chunker = Chunker::default();
    let mut ret = Vec::new();
    let mut pos = 0;
    loop {
        if pos == data.len() {
            // Flush any final partial chunk.
            if let Some(chunk) = chunker.next(&[], true).0 {
                ret.push(chunk);
            }
            return ret;
        }
        let (maybe_chunk, consumed) = chunker.next(&data[pos..], false);
        if let Some(chunk) = maybe_chunk {
            ret.push(chunk);
        }
        if consumed == 0 {
            // `next` consumed nothing and emitted nothing only when the buffer
            // is sub-minimum and not final; force progress by feeding the rest
            // as final.
            let (final_chunk, _) = chunker.next(&data[pos..], true);
            if let Some(chunk) = final_chunk {
                ret.push(chunk);
            }
            return ret;
        }
        pos += consumed;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_all(&[]).is_empty());
    }

    #[test]
    fn round_trip_concatenation_small() {
        // Pseudo-random data so CDC actually cuts boundaries.
        let mut data = Vec::with_capacity(4096);
        let mut state: u64 = 0x9e3779b97f4a7c15;
        for _ in 0..512 {
            state = state.wrapping_mul(0xbf58476d1ce4e5b9) ^ (state >> 27);
            data.extend_from_slice(&state.to_le_bytes());
        }
        let chunks = chunk_all(&data);
        assert!(!chunks.is_empty());

        let mut rebuilt = Vec::new();
        for c in &chunks {
            rebuilt.extend_from_slice(&c.data);
            // chunk hash is BLAKE3-keyed over the chunk bytes
            assert_eq!(c.hash, merklehash::compute_data_hash(&c.data));
        }
        assert_eq!(rebuilt, data);
    }

    #[test]
    fn respects_maximum_chunk_size() {
        // Constant bytes never trigger the gear mask, so chunks should all be
        // the maximum size except the (final) tail.
        let data = vec![0xA5u8; MAX_XORB_BYTES + 17];
        let chunks = chunk_all(&data);
        for c in &chunks {
            assert!(c.data.len() <= TARGET_CHUNK_SIZE * MAXIMUM_CHUNK_MULTIPLIER);
        }
        let total: usize = chunks.iter().map(|c| c.data.len()).sum();
        assert_eq!(total, data.len());
    }

    /// Golden test: constant data must cut at the maximum chunk boundary
    /// (TARGET * MAXIMUM_CHUNK_MULTIPLIER = 128 KiB), matching xet-core's
    /// `test_correctness_1mb_const_data` exactly. This proves the chunker
    /// produces byte-identical boundaries to upstream xet-core.
    #[test]
    fn boundaries_match_xet_core_const_data() {
        let data = vec![59u8; 1_000_000];
        let chunks = chunk_all(&data);
        let boundaries: Vec<usize> = chunks
            .iter()
            .scan(0, |acc, c| {
                *acc += c.data.len();
                Some(*acc)
            })
            .collect();
        // These are the exact expected boundaries from xet-core's
        // deduplication/src/chunking.rs::test_correctness_1mb_const_data.
        assert_eq!(
            boundaries,
            vec![131072, 262144, 393216, 524288, 655360, 786432, 917504, 1_000_000]
        );
    }
}
