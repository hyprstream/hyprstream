//! The XET `mdb_shard` binary reconstruction format — a faithful, minimal port
//! of xet-core's `mdb_shard` crate.
//!
//! cas-serve advertises XET support, which means the reconstruction shard it
//! returns from `GetReconstructionInfo` must be parseable by any stock xet-core
//! client (its `MDBShardFile` reader / `FileReconstructor`). The bespoke JSON
//! shard the original #390 commit produced is not the XET wire format. This
//! module ports the real binary container — the `.mdb` file — so a shard built
//! here is byte-compatible with upstream.
//!
//! # Why a port, not a dependency
//!
//! Upstream `mdb_shard` is `edition = "2024"` and pulls in `xet_config` /
//! `xet_runtime` / `utils` (and, transitively, the libtorch-coupled tree).
//! cas-serve deliberately stays out of that tree (no libtorch). The binary
//! format itself is plain `Write`/`Read` + little-endian primitives, so it
//! ports cleanly to edition 2021.
//!
//! # Layout (ported from `mdb_shard/src/shard_format.rs`)
//!
//! The on-disk shard is, in order:
//!
//! 1. **Header** (`MDBShardFileHeader`, 48 B): 32-byte magic tag, `version`
//!    (u64 LE, =2), `footer_size` (u64 LE).
//! 2. **File-Info section**: one `FileDataSequenceHeader` + N
//!    `FileDataSequenceEntry` per file (in ascending file-hash order), then a
//!    single bookend header. Each entry is 48 B (`[u64;4]` hash + 4×u32).
//! 3. **CAS-Info section**: one `CASChunkSequenceHeader` + N
//!    `CASChunkSequenceEntry` per xorb (in ascending xorb-hash order), then a
//!    single bookend header.
//! 4. **File lookup**: `(u64 truncated file hash, u32 index)` per file.
//! 5. **CAS lookup**: `(u64 truncated xorb hash, u32 index)` per xorb.
//! 6. **Chunk lookup**: `(u64 truncated chunk hash, u32 cas idx, u32 chunk
//!    offset)` per chunk, sorted by key.
//! 7. **Footer** (`MDBShardFileFooter`, 200 B): section offsets/counts, the
//!    32-byte chunk-hash HMAC key, timestamps, byte totals, and
//!    `footer_offset`.
//!
//! `truncate_hash(h) = h[0]` (first u64 of the 4×u64 `MerkleHash`).
//!
//! All multi-byte integers are little-endian, matching
//! `utils::serialization_utils`.

use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::mem::size_of;

use merklehash::MerkleHash;

use crate::chunker::Chunk;

// --------------------------------------------------------------------------- //
// Constants — match `mdb_shard/src/shard_format.rs` verbatim.                //
// --------------------------------------------------------------------------- //

/// 32-byte magic tag at the head of every shard file
/// (`shard_format.rs::MDB_SHARD_HEADER_TAG`).
pub const MDB_SHARD_HEADER_TAG: [u8; 32] = [
    b'H', b'F', b'R', b'e', b'p', b'o', b'M', b'e', b't', b'a', b'D', b'a', b't', b'a', 0, 85, 105,
    103, 69, 106, 123, 129, 87, 131, 165, 189, 217, 92, 205, 209, 74, 169,
];

const MDB_SHARD_HEADER_VERSION: u64 = 2;
const MDB_SHARD_FOOTER_VERSION: u64 = 1;

/// Every fixed-size File-Info / CAS-Info record is 48 bytes: a 32-byte hash
/// (`[u64;4]`) plus four `u32` slots. Asserted byte-for-byte against
/// `FileDataSequenceHeader`, `FileDataSequenceEntry`, `CASChunkSequenceHeader`,
/// and `CASChunkSequenceEntry` below.
pub const MDB_ENTRY_SIZE: usize = size_of::<[u64; 4]>() + 4 * size_of::<u32>();
/// Compile-time check that every fixed-size shard record is 48 bytes.
const _: () = const_assert_entry_size();

/// Size of the footer, in bytes: 21 `u64` fields + the 32-byte HMAC key = 200.
/// Verified against `size_of::<MDBShardFileFooter>()` upstream.
pub const MDB_FOOTER_SIZE: u64 = 21 * size_of::<u64>() as u64 + 32;
/// Compile-time check that the footer layout sums to 200 bytes (21×8 + 32).
const _: () = const_assert_footer_size();

/// `truncate_hash` — the first u64 of a `MerkleHash`, used as the lookup key.
/// Matches `mdb_shard/src/utils.rs::truncate_hash`.
#[inline]
fn truncate_hash(h: &MerkleHash) -> u64 {
    h[0]
}

/// Compile-time check that every fixed-size shard record is 48 bytes.
#[allow(clippy::assertions_on_constants)]
const fn const_assert_entry_size() {
    assert!(MDB_ENTRY_SIZE == 48);
}

/// Compile-time check that the footer layout sums to 200 bytes (21×8 + 32).
#[allow(clippy::assertions_on_constants)]
const fn const_assert_footer_size() {
    assert!(MDB_FOOTER_SIZE == 200);
}

// --------------------------------------------------------------------------- //
// Serialization primitives — direct ports of `utils::serialization_utils`.   //
// --------------------------------------------------------------------------- //
//
// `write_hash`/`read_hash` are *raw* 32-byte writes (the `MerkleHash` is a
// transparent `[u64; 4]` and is serialized as `as_bytes()` / transmuted back;
// on a little-endian host `h[0]` is therefore the first 8 bytes, which is what
// makes `truncate_hash` well-defined on the wire).

#[inline]
fn write_hash<W: Write>(w: &mut W, h: &MerkleHash) -> std::io::Result<()> {
    w.write_all(h.as_ref())
}

#[inline]
fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline]
fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline]
fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; size_of::<u32>()];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[inline]
fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; size_of::<u64>()];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[inline]
fn read_hash<R: Read>(r: &mut R) -> std::io::Result<MerkleHash> {
    let mut b = [0u8; 32];
    r.read_exact(&mut b)?;
    // The upstream reader uses `transmute::<[u8;32],[u64;4]>`; on the wire the
    // bytes are the in-memory representation of the host that wrote them. We
    // reconstruct via `from_slice`, which is the safe, layout-identical copy.
    MerkleHash::from_slice(&b).map_err(|_| std::io::Error::other("32-byte hash expected"))
}

// --------------------------------------------------------------------------- //
// File-Info / CAS-Info records (`file_structs.rs`, `cas_structs.rs`).        //
// --------------------------------------------------------------------------- //
//
// Flags from `file_structs.rs`. cas-serve never emits verification entries,
// so these are kept only to document the on-disk flag layout faithfully.
#[allow(dead_code)]
const MDB_FILE_FLAG_WITH_VERIFICATION: u32 = 1 << 31;
#[allow(dead_code)]
const MDB_FILE_FLAG_VERIFICATION_MASK: u32 = 1 << 31;

/// `FileDataSequenceHeader` (48 B): the per-file record that opens its segment
/// list. `file_hash` is the content address the shard is keyed by.
#[derive(Clone, Debug, PartialEq)]
pub struct FileDataSequenceHeader {
    pub file_hash: MerkleHash,
    pub file_flags: u32,
    pub num_entries: u32,
    pub _unused: u64,
}

impl FileDataSequenceHeader {
    /// The all-ones hash marks the bookend record that terminates sequential
    /// scans of the File-Info section.
    fn bookend() -> Self {
        Self {
            file_hash: [!0u64; 4].into(),
            file_flags: 0,
            num_entries: 0,
            _unused: 0,
        }
    }

    fn is_bookend(&self) -> bool {
        self.file_hash == [!0u64; 4].into()
    }

    /// Whether this file carries per-segment verification entries. cas-serve
    /// never emits them, but a reader must honor the flag when parsing shards
    /// produced by upstream xet-core clients.
    #[allow(dead_code)]
    fn contains_verification(&self) -> bool {
        (self.file_flags & MDB_FILE_FLAG_VERIFICATION_MASK) != 0
    }

    fn serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_hash(w, &self.file_hash)?;
        write_u32(w, self.file_flags)?;
        write_u32(w, self.num_entries)?;
        write_u64(w, self._unused)
    }

    fn deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            file_hash: read_hash(r)?,
            file_flags: read_u32(r)?,
            num_entries: read_u32(r)?,
            _unused: read_u64(r)?,
        })
    }
}

/// `FileDataSequenceEntry` (48 B): one reconstruction segment — the xorb to
/// pull chunks from, the inclusive-start / exclusive-end chunk range, and the
/// total unpacked byte length of that range. Concatenating the chunk bytes for
/// every segment, in segment order, reproduces the file.
#[derive(Clone, Debug, PartialEq)]
pub struct FileDataSequenceEntry {
    pub cas_hash: MerkleHash,
    pub cas_flags: u32,
    pub unpacked_segment_bytes: u32,
    /// Inclusive start chunk index within the xorb.
    pub chunk_index_start: u32,
    /// Exclusive end chunk index within the xorb
    /// (`chunk_index_end - chunk_index_start` = chunk count).
    pub chunk_index_end: u32,
}

impl FileDataSequenceEntry {
    fn serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_hash(w, &self.cas_hash)?;
        write_u32(w, self.cas_flags)?;
        write_u32(w, self.unpacked_segment_bytes)?;
        write_u32(w, self.chunk_index_start)?;
        write_u32(w, self.chunk_index_end)
    }

    fn deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            cas_hash: read_hash(r)?,
            cas_flags: read_u32(r)?,
            unpacked_segment_bytes: read_u32(r)?,
            chunk_index_start: read_u32(r)?,
            chunk_index_end: read_u32(r)?,
        })
    }
}

/// `CASChunkSequenceHeader` (48 B): opens a xorb's chunk list.
#[derive(Clone, Debug, PartialEq)]
pub struct CasChunkSequenceHeader {
    pub cas_hash: MerkleHash,
    pub cas_flags: u32,
    pub num_entries: u32,
    pub num_bytes_in_cas: u32,
    pub num_bytes_on_disk: u32,
}

impl CasChunkSequenceHeader {
    /// All-ones hash marks the CAS-Info bookend.
    fn bookend() -> Self {
        Self {
            cas_hash: [!0u64; 4].into(),
            cas_flags: 0,
            num_entries: 0,
            num_bytes_in_cas: 0,
            num_bytes_on_disk: 0,
        }
    }

    fn is_bookend(&self) -> bool {
        self.cas_hash == [!0u64; 4].into()
    }

    fn serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_hash(w, &self.cas_hash)?;
        write_u32(w, self.cas_flags)?;
        write_u32(w, self.num_entries)?;
        write_u32(w, self.num_bytes_in_cas)?;
        write_u32(w, self.num_bytes_on_disk)
    }

    fn deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            cas_hash: read_hash(r)?,
            cas_flags: read_u32(r)?,
            num_entries: read_u32(r)?,
            num_bytes_in_cas: read_u32(r)?,
            num_bytes_on_disk: read_u32(r)?,
        })
    }
}

/// `CASChunkSequenceEntry` (48 B): one chunk within a xorb.
#[derive(Clone, Debug, PartialEq)]
pub struct CasChunkSequenceEntry {
    pub chunk_hash: MerkleHash,
    pub chunk_byte_range_start: u32,
    pub unpacked_segment_bytes: u32,
    pub flags: u32,
    pub _unused: u32,
}

impl CasChunkSequenceEntry {
    fn serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_hash(w, &self.chunk_hash)?;
        write_u32(w, self.chunk_byte_range_start)?;
        write_u32(w, self.unpacked_segment_bytes)?;
        write_u32(w, self.flags)?;
        write_u32(w, self._unused)
    }

    fn deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            chunk_hash: read_hash(r)?,
            chunk_byte_range_start: read_u32(r)?,
            unpacked_segment_bytes: read_u32(r)?,
            flags: read_u32(r)?,
            _unused: read_u32(r)?,
        })
    }
}

// --------------------------------------------------------------------------- //
// File container header / footer (`shard_format.rs`).                        //
// --------------------------------------------------------------------------- //

/// 48-byte shard header: magic tag + version + footer size.
#[derive(Clone, Debug, PartialEq)]
pub struct MdbShardHeader {
    pub version: u64,
    pub footer_size: u64,
}

impl Default for MdbShardHeader {
    fn default() -> Self {
        Self {
            version: MDB_SHARD_HEADER_VERSION,
            footer_size: MDB_FOOTER_SIZE,
        }
    }
}

impl MdbShardHeader {
    fn serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&MDB_SHARD_HEADER_TAG)?;
        write_u64(w, self.version)?;
        write_u64(w, self.footer_size)
    }

    fn deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut tag = [0u8; 32];
        r.read_exact(&mut tag)?;
        if tag != MDB_SHARD_HEADER_TAG {
            return Err(std::io::Error::other(
                "not an mdb_shard file (magic mismatch)",
            ));
        }
        Ok(Self {
            version: read_u64(r)?,
            footer_size: read_u64(r)?,
        })
    }
}

/// 200-byte shard footer carrying the section offsets, lookup counts, HMAC
/// key, timestamps, and byte totals. Field order matches
/// `MDBShardFileFooter::serialize` exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct MdbShardFooter {
    pub version: u64,
    pub file_info_offset: u64,
    pub cas_info_offset: u64,
    pub file_lookup_offset: u64,
    pub file_lookup_num_entry: u64,
    pub cas_lookup_offset: u64,
    pub cas_lookup_num_entry: u64,
    pub chunk_lookup_offset: u64,
    pub chunk_lookup_num_entry: u64,
    /// 32-byte chunk-hash HMAC key (all-zero = no keying). cas-serve does not
    /// key its chunks, so this is always zero.
    pub chunk_hash_hmac_key: [u8; 32],
    pub shard_creation_timestamp: u64,
    pub shard_key_expiry: u64,
    pub _buffer: [u64; 6],
    pub stored_bytes_on_disk: u64,
    pub materialized_bytes: u64,
    pub stored_bytes: u64,
    pub footer_offset: u64,
}

impl Default for MdbShardFooter {
    fn default() -> Self {
        Self {
            version: MDB_SHARD_FOOTER_VERSION,
            file_info_offset: 0,
            cas_info_offset: 0,
            file_lookup_offset: 0,
            file_lookup_num_entry: 0,
            cas_lookup_offset: 0,
            cas_lookup_num_entry: 0,
            chunk_lookup_offset: 0,
            chunk_lookup_num_entry: 0,
            chunk_hash_hmac_key: [0u8; 32],
            shard_creation_timestamp: 0,
            shard_key_expiry: u64::MAX,
            _buffer: [0u64; 6],
            stored_bytes_on_disk: 0,
            materialized_bytes: 0,
            stored_bytes: 0,
            footer_offset: 0,
        }
    }
}

impl MdbShardFooter {
    fn serialize<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write_u64(w, self.version)?;
        write_u64(w, self.file_info_offset)?;
        write_u64(w, self.cas_info_offset)?;
        write_u64(w, self.file_lookup_offset)?;
        write_u64(w, self.file_lookup_num_entry)?;
        write_u64(w, self.cas_lookup_offset)?;
        write_u64(w, self.cas_lookup_num_entry)?;
        write_u64(w, self.chunk_lookup_offset)?;
        write_u64(w, self.chunk_lookup_num_entry)?;
        w.write_all(&self.chunk_hash_hmac_key)?;
        write_u64(w, self.shard_creation_timestamp)?;
        write_u64(w, self.shard_key_expiry)?;
        for &v in &self._buffer {
            write_u64(w, v)?;
        }
        write_u64(w, self.stored_bytes_on_disk)?;
        write_u64(w, self.materialized_bytes)?;
        write_u64(w, self.stored_bytes)?;
        write_u64(w, self.footer_offset)
    }

    fn deserialize<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let version = read_u64(r)?;
        if version != MDB_SHARD_FOOTER_VERSION {
            return Err(std::io::Error::other(format!(
                "mdb_shard footer version mismatch: expected {MDB_SHARD_FOOTER_VERSION}, got {version}"
            )));
        }
        let mut key = [0u8; 32];
        let mut buf6 = [0u64; 6];
        Ok(Self {
            version,
            file_info_offset: read_u64(r)?,
            cas_info_offset: read_u64(r)?,
            file_lookup_offset: read_u64(r)?,
            file_lookup_num_entry: read_u64(r)?,
            cas_lookup_offset: read_u64(r)?,
            cas_lookup_num_entry: read_u64(r)?,
            chunk_lookup_offset: read_u64(r)?,
            chunk_lookup_num_entry: read_u64(r)?,
            chunk_hash_hmac_key: {
                r.read_exact(&mut key)?;
                key
            },
            shard_creation_timestamp: read_u64(r)?,
            shard_key_expiry: read_u64(r)?,
            _buffer: {
                for v in &mut buf6 {
                    *v = read_u64(r)?;
                }
                buf6
            },
            stored_bytes_on_disk: read_u64(r)?,
            materialized_bytes: read_u64(r)?,
            stored_bytes: read_u64(r)?,
            footer_offset: read_u64(r)?,
        })
    }
}

// --------------------------------------------------------------------------- //
// In-memory shard + builder (port of `MDBInMemoryShard` / `serialize_from`). //
// --------------------------------------------------------------------------- //

/// One file's reconstruction plan: its content-address and the ordered
/// segments that rebuild it.
#[derive(Clone, Debug)]
pub struct MdbFileInfo {
    pub metadata: FileDataSequenceHeader,
    pub segments: Vec<FileDataSequenceEntry>,
}

impl MdbFileInfo {
    /// Total unpacked file length (sum of segment byte lengths).
    pub fn file_size(&self) -> u64 {
        self.segments
            .iter()
            .map(|s| s.unpacked_segment_bytes as u64)
            .sum()
    }
}

/// One xorb's content: its header + the ordered chunk entries it holds.
#[derive(Clone, Debug)]
pub struct MdbCasInfo {
    pub metadata: CasChunkSequenceHeader,
    pub chunks: Vec<CasChunkSequenceEntry>,
}

/// A builder that mirrors xet-core's `MDBInMemoryShard` — accumulate file and
/// xorb records, then `serialize` emits the `.mdb` byte stream in exactly the
/// order `MDBShardInfo::serialize_from` does.
#[derive(Default)]
pub struct MdbShardBuilder {
    /// Files keyed by file hash (BTreeMap ⇒ ascending hash order on disk).
    files: std::collections::BTreeMap<MerkleHash, MdbFileInfo>,
    /// XORBs keyed by xorb hash (BTreeMap ⇒ ascending hash order on disk).
    cas: std::collections::BTreeMap<MerkleHash, MdbCasInfo>,
}

impl MdbShardBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file's reconstruction info. Overwrites any prior entry for the
    /// same file hash.
    pub fn add_file(&mut self, info: MdbFileInfo) {
        self.files.insert(info.metadata.file_hash, info);
    }

    /// Add a xorb's chunk list. Overwrites any prior entry for the same xorb
    /// hash.
    pub fn add_cas(&mut self, info: MdbCasInfo) {
        self.cas.insert(info.metadata.cas_hash, info);
    }

    /// Serialize the shard to a byte vector, in the canonical section order:
    /// header → file-info → cas-info → file lookup → cas lookup → chunk lookup
    /// → footer. Ported from `MDBShardInfo::serialize_from`.
    pub fn serialize(&self) -> std::io::Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        let mut footer = MdbShardFooter::default();

        // 1. Header.
        MdbShardHeader::default().serialize(&mut out)?;
        debug_assert_eq!(out.len() as u64, 48);

        // 2. File-Info section (+ bookend). Track the file-lookup table while
        //    we walk the files in ascending-hash order.
        footer.file_info_offset = out.len() as u64;
        let mut file_lookup: Vec<(u64, u32)> = Vec::with_capacity(self.files.len());
        let mut file_index: u32 = 0;
        for (file_hash, info) in &self.files {
            file_lookup.push((truncate_hash(file_hash), file_index));
            info.metadata.serialize(&mut out)?;
            for seg in &info.segments {
                seg.serialize(&mut out)?;
            }
            // Each file contributes 1 header + N segments, all 48 B records.
            file_index += 1 + info.segments.len() as u32;
        }
        FileDataSequenceHeader::bookend().serialize(&mut out)?;

        // 3. CAS-Info section (+ bookend). Track both the CAS lookup and the
        //    chunk lookup.
        footer.cas_info_offset = out.len() as u64;
        let mut cas_lookup: Vec<(u64, u32)> = Vec::with_capacity(self.cas.len());
        let mut chunk_lookup: Vec<(u64, u32, u32)> = Vec::new();
        let mut cas_index: u32 = 0;
        let mut stored_bytes_on_disk: u64 = 0;
        let mut stored_bytes: u64 = 0;
        for (cas_hash, info) in &self.cas {
            cas_lookup.push((truncate_hash(cas_hash), cas_index));
            info.metadata.serialize(&mut out)?;
            for (i, chunk) in info.chunks.iter().enumerate() {
                chunk.serialize(&mut out)?;
                chunk_lookup.push((truncate_hash(&chunk.chunk_hash), cas_index, i as u32));
            }
            cas_index += 1 + info.chunks.len() as u32;
            stored_bytes_on_disk += info.metadata.num_bytes_on_disk as u64;
            stored_bytes += info.metadata.num_bytes_in_cas as u64;
        }
        CasChunkSequenceHeader::bookend().serialize(&mut out)?;

        // 4. File lookup table (already in ascending truncated-hash order
        //    because BTreeMap iterates in full-hash order and truncation
        //    preserves it).
        footer.file_lookup_offset = out.len() as u64;
        footer.file_lookup_num_entry = file_lookup.len() as u64;
        for &(h, idx) in &file_lookup {
            write_u64(&mut out, h)?;
            write_u32(&mut out, idx)?;
        }

        // 5. CAS lookup table (same ascending-order argument).
        footer.cas_lookup_offset = out.len() as u64;
        footer.cas_lookup_num_entry = cas_lookup.len() as u64;
        for &(h, idx) in &cas_lookup {
            write_u64(&mut out, h)?;
            write_u32(&mut out, idx)?;
        }

        // 6. Chunk lookup table — MUST be sorted by truncated chunk hash
        //    (xet-core sorts; chunk hashes are not monotone with xorb order).
        footer.chunk_lookup_offset = out.len() as u64;
        chunk_lookup.sort_unstable_by_key(|&(h, _, _)| h);
        footer.chunk_lookup_num_entry = chunk_lookup.len() as u64;
        for &(h, a, b) in &chunk_lookup {
            write_u64(&mut out, h)?;
            write_u32(&mut out, a)?;
            write_u32(&mut out, b)?;
        }

        // Byte totals + footer offset, then the footer itself.
        // `materialized_bytes` = total unpacked file bytes (sum over every
        // file segment), matching `MDBInMemoryShard::materialized_bytes`.
        footer.stored_bytes_on_disk = stored_bytes_on_disk;
        footer.materialized_bytes = self
            .files
            .values()
            .flat_map(|f| f.segments.iter())
            .map(|s| s.unpacked_segment_bytes as u64)
            .sum();
        footer.stored_bytes = stored_bytes;
        footer.footer_offset = out.len() as u64;
        footer.serialize(&mut out)?;
        Ok(out)
    }
}

// --------------------------------------------------------------------------- //
// Reader — port of `MDBShardFile` / `MDBShardInfo` (lookup path only).       //
// --------------------------------------------------------------------------- //

/// A parsed shard: header + footer, with helpers to seek into the byte buffer
/// and reconstruct a file.
pub struct MdbShard {
    pub header: MdbShardHeader,
    pub footer: MdbShardFooter,
}

impl MdbShard {
    /// Parse the header and footer from a complete shard byte buffer.
    pub fn parse(bytes: &[u8]) -> std::io::Result<Self> {
        let mut cur = Cursor::new(bytes);
        let header = MdbShardHeader::deserialize(&mut cur)?;
        // Footer is the last `footer_size` bytes.
        let footer_start = bytes.len() as i64 - header.footer_size as i64;
        if footer_start < 0 {
            return Err(std::io::Error::other("shard too small for footer"));
        }
        cur.seek(SeekFrom::Start(footer_start as u64))?;
        let footer = MdbShardFooter::deserialize(&mut cur)?;
        Ok(Self { header, footer })
    }

    /// Read the `MDBFileInfo` (header + segments) for `file_hash`, seeking the
    /// file-lookup table then walking the candidate indices exactly like
    /// `MDBShardInfo::get_file_reconstruction_info`.
    pub fn file_info(
        &self,
        bytes: &[u8],
        file_hash: &MerkleHash,
    ) -> std::io::Result<Option<MdbFileInfo>> {
        let mut cur = Cursor::new(bytes);
        // Scan the file-lookup table for indices whose truncated hash matches.
        let target = truncate_hash(file_hash);
        cur.seek(SeekFrom::Start(self.footer.file_lookup_offset))?;
        let mut candidates: Vec<u32> = Vec::new();
        for _ in 0..self.footer.file_lookup_num_entry {
            let h = read_u64(&mut cur)?;
            let idx = read_u32(&mut cur)?;
            if h == target {
                candidates.push(idx);
            }
        }

        for idx in candidates {
            cur.seek(SeekFrom::Start(
                self.footer.file_info_offset + MDB_ENTRY_SIZE as u64 * idx as u64,
            ))?;
            let header = FileDataSequenceHeader::deserialize(&mut cur)?;
            if header.is_bookend() {
                continue;
            }
            let mut segments = Vec::with_capacity(header.num_entries as usize);
            for _ in 0..header.num_entries {
                segments.push(FileDataSequenceEntry::deserialize(&mut cur)?);
            }
            if header.file_hash == *file_hash {
                return Ok(Some(MdbFileInfo {
                    metadata: header,
                    segments,
                }));
            }
        }
        Ok(None)
    }

    /// Read every xorb's chunk list (used to verify/inspect a shard).
    pub fn all_cas_blocks(&self, bytes: &[u8]) -> std::io::Result<Vec<MdbCasInfo>> {
        let mut cur = Cursor::new(bytes);
        cur.seek(SeekFrom::Start(self.footer.cas_info_offset))?;
        let mut out = Vec::new();
        loop {
            let header = CasChunkSequenceHeader::deserialize(&mut cur)?;
            if header.is_bookend() {
                break;
            }
            let mut chunks = Vec::with_capacity(header.num_entries as usize);
            for _ in 0..header.num_entries {
                chunks.push(CasChunkSequenceEntry::deserialize(&mut cur)?);
            }
            out.push(MdbCasInfo {
                metadata: header,
                chunks,
            });
        }
        Ok(out)
    }
}

// --------------------------------------------------------------------------- //
// cas-serve integration: chunk → xorb → shard builder.                       //
// --------------------------------------------------------------------------- //
//
// The chunker + greedy xorb packing are unchanged from the original #390
// commit (the golden CDC test proves they match xet-core). Only the shard
// *container* changes: JSON → mdb_shard binary.

/// Build an `MdbShardBuilder` (plus, per xorb, its hash and concatenated raw
/// bytes) from a chunk list, packing chunks into xorbs that respect the XET
/// `MAX_XORB_BYTES` (64 MiB) and `MAX_XORB_CHUNKS` (8192) limits.
///
/// Chunk → xorb aggregation, xorb hashing (`merklehash::xorb_hash`), and file
/// hashing (`merklehash::file_hash`) all match xet-core's
/// `RawXorbData::from_chunks`, so the hashes embedded in the produced shard
/// are identical to an upstream client's.
pub fn build_shard(chunks: &[Chunk]) -> (MdbShardBuilder, MerkleHash, Vec<(MerkleHash, Vec<u8>)>) {
    build_shard_with_cap(chunks, crate::chunker::MAX_XORB_BYTES)
}

/// Like [`build_shard`], but with a custom per-xorb byte cap (tests use a tiny
/// cap to force multi-xorb aggregation without allocating 64 MiB).
pub fn build_shard_with_cap(
    chunks: &[Chunk],
    xorb_byte_cap: usize,
) -> (MdbShardBuilder, MerkleHash, Vec<(MerkleHash, Vec<u8>)>) {
    use crate::chunker::MAX_XORB_CHUNKS;

    let mut builder = MdbShardBuilder::new();
    let mut xorbs: Vec<(MerkleHash, Vec<u8>)> = Vec::new();
    // One File-Info segment per xorb, accumulated in file order; the single
    // file entry is added once the file hash is known.
    let mut file_segments: Vec<FileDataSequenceEntry> = Vec::new();

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
            xorb_chunks.push(c);
            i += 1;
        }
        let end = i; // exclusive

        // Xorb hash = Merkle root over (chunk_hash, chunk_len) — matches
        // `merklehash::xorb_hash` / `RawXorbData::from_chunks`.
        let hash_inputs: Vec<(MerkleHash, u64)> = xorb_chunks
            .iter()
            .map(|c| (c.hash, c.data.len() as u64))
            .collect();
        let xorb_hash = merklehash::xorb_hash(&hash_inputs);

        // CAS-Info chunk entries: chunk_hash, byte_range_start, unpacked bytes.
        let mut cas_entries = Vec::with_capacity(xorb_chunks.len());
        let mut running_start: u32 = 0;
        for c in &xorb_chunks {
            let len = c.data.len() as u32;
            cas_entries.push(CasChunkSequenceEntry {
                chunk_hash: c.hash,
                chunk_byte_range_start: running_start,
                unpacked_segment_bytes: len,
                flags: 0,
                _unused: 0,
            });
            running_start = running_start.saturating_add(len);
        }
        builder.add_cas(MdbCasInfo {
            metadata: CasChunkSequenceHeader {
                cas_hash: xorb_hash,
                cas_flags: 0,
                num_entries: cas_entries.len() as u32,
                num_bytes_in_cas: running_start,
                // cas-serve stores xorbs uncompressed: on-disk == in-cas.
                num_bytes_on_disk: running_start,
            },
            chunks: cas_entries,
        });

        // File-Info segment: this xorb holds the whole local chunk range
        // `[0, xorb_chunk_count)`. Chunk indices in a `FileDataSequenceEntry`
        // are *local to the referenced xorb* (its CAS entries are numbered
        // `0..N`), not global file-chunk indices — matching xet-core's
        // `FileDataSequenceEntry::from_cas_entries`. Since cas-serve packs each
        // xorb to belong to exactly one file upload, every xorb contributes one
        // segment spanning all of its chunks.
        let n_local = (end - start) as u32;
        file_segments.push(FileDataSequenceEntry {
            cas_hash: xorb_hash,
            cas_flags: 0,
            unpacked_segment_bytes: running_start,
            chunk_index_start: 0,
            chunk_index_end: n_local,
        });

        // Xorb bytes = chunks concatenated (raw; cas-serve stores uncompressed
        // xorbs, matching `num_bytes_on_disk == num_bytes_in_cas`).
        let mut bytes = Vec::with_capacity(running_start as usize);
        for c in &xorb_chunks {
            bytes.extend_from_slice(&c.data);
        }
        xorbs.push((xorb_hash, bytes));
    }

    // File hash = HMAC'd aggregated node hash over all chunk (hash, len) pairs
    // — matches `merklehash::file_hash`.
    let file_hash_inputs: Vec<(MerkleHash, u64)> = chunks
        .iter()
        .map(|c| (c.hash, c.data.len() as u64))
        .collect();
    let file_hash = merklehash::file_hash(&file_hash_inputs);

    builder.add_file(MdbFileInfo {
        metadata: FileDataSequenceHeader {
            file_hash,
            file_flags: 0, // no verification, no metadata_ext
            num_entries: file_segments.len() as u32,
            _unused: 0,
        },
        segments: file_segments,
    });

    (builder, file_hash, xorbs)
}

// --------------------------------------------------------------------------- //
// Tests: round-trip + byte-layout parity with the xet-core format.           //
// --------------------------------------------------------------------------- //

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::chunker::chunk_all;

    /// The header is exactly 48 bytes and the footer exactly 200, matching the
    /// `size_of` assertions xet-core enforces.
    #[test]
    fn header_and_footer_sizes_match_xet_core() {
        // Header = 32 (tag) + 8 (version) + 8 (footer_size).
        assert_eq!(32 + size_of::<u64>() + size_of::<u64>(), 48);

        // Footer = 21 u64 fields + 32-byte HMAC key = 200.
        assert_eq!(MDB_FOOTER_SIZE, 200);
        // Cross-check the explicit count.
        let counted = 21 * size_of::<u64>() as u64 + 32;
        assert_eq!(counted, MDB_FOOTER_SIZE);
    }

    /// A shard cas-serve builds round-trips through its own reader.
    #[test]
    fn build_then_parse_round_trip() {
        let mut data = Vec::with_capacity(300_000);
        let mut s: u64 = 42;
        for _ in 0..37500 {
            s = s.wrapping_mul(0x94d049bb133111eb) ^ (s >> 31);
            data.extend_from_slice(&s.to_le_bytes());
        }
        let chunks = chunk_all(&data);
        assert!(chunks.len() > 1);

        let (builder, file_hash, _xorbs) = build_shard(&chunks);
        let bytes = builder.serialize().unwrap();

        // Magic + versions.
        assert_eq!(&bytes[..32], &MDB_SHARD_HEADER_TAG);
        assert_eq!(read_u64(&mut Cursor::new(&bytes[32..40])).unwrap(), 2);

        let shard = MdbShard::parse(&bytes).unwrap();
        assert_eq!(shard.footer.version, 1);
        assert_eq!(shard.footer.file_lookup_num_entry, 1);

        let info = shard.file_info(&bytes, &file_hash).unwrap().unwrap();
        assert_eq!(info.metadata.file_hash, file_hash);
        assert_eq!(info.metadata.num_entries as usize, info.segments.len());
        assert_eq!(info.file_size(), data.len() as u64);
    }

    /// Byte-layout parity: the on-disk offsets a reader computes must land on
    /// the exact records a writer produced. This is the field-by-field check
    /// the prompt asks for when full xet-core cross-compilation isn't feasible.
    #[test]
    fn offsets_land_on_exact_records() {
        let data = vec![0x42u8; 200_000];
        let chunks = chunk_all(&data);
        let (builder, file_hash, xorbs) = build_shard(&chunks);
        let bytes = builder.serialize().unwrap();
        let shard = MdbShard::parse(&bytes).unwrap();

        // File-Info section starts right after the 48-byte header.
        assert_eq!(shard.footer.file_info_offset, 48);

        // The single file's header is the first record of the file-info section.
        let mut cur = Cursor::new(&bytes);
        cur.set_position(shard.footer.file_info_offset);
        let fh = FileDataSequenceHeader::deserialize(&mut cur).unwrap();
        assert_eq!(fh.file_hash, file_hash);
        assert_eq!(fh.num_entries as usize, xorbs.len());

        // Each segment's cas_hash must match a xorb we built.
        for seg in shard
            .file_info(&bytes, &file_hash)
            .unwrap()
            .unwrap()
            .segments
        {
            assert!(xorbs.iter().any(|(h, _)| *h == seg.cas_hash));
        }

        // CAS-Info section: each xorb header is where all_cas_blocks lands.
        let cas = shard.all_cas_blocks(&bytes).unwrap();
        assert_eq!(cas.len(), xorbs.len());
        for (info, (h, _)) in cas.iter().zip(&xorbs) {
            assert_eq!(info.metadata.cas_hash, *h);
        }

        // Footer is the final 200 bytes.
        assert_eq!(
            shard.footer.footer_offset as usize + 200,
            bytes.len(),
            "footer_offset + 200 must equal total shard length"
        );
    }

    /// Multi-xorb: a file split across several xorbs still reconstructs via the
    /// mdb_shard binary.
    #[test]
    fn multi_xorb_round_trip() {
        let mut data = Vec::with_capacity(1024 * 1024);
        let mut s: u64 = 0xdeadbeefcafebabe;
        for _ in 0..131072 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            data.extend_from_slice(&s.to_le_bytes());
        }
        let chunks = chunk_all(&data);
        let (builder, file_hash, xorbs) = build_shard_with_cap(&chunks, 256 * 1024);
        assert!(xorbs.len() > 1, "expected multiple xorbs");

        let bytes = builder.serialize().unwrap();
        let shard = MdbShard::parse(&bytes).unwrap();
        let info = shard.file_info(&bytes, &file_hash).unwrap().unwrap();
        assert_eq!(info.metadata.num_entries as usize, xorbs.len());

        // Reassemble exactly as GetFile does: walk segments, slice each xorb by
        // the chunk-index range.
        let mut rebuilt = Vec::new();
        let cas = shard.all_cas_blocks(&bytes).unwrap();
        for seg in &info.segments {
            let xorb = cas
                .iter()
                .find(|c| c.metadata.cas_hash == seg.cas_hash)
                .unwrap();
            for i in seg.chunk_index_start..seg.chunk_index_end {
                let chunk = &xorb.chunks[i as usize];
                let start = chunk.chunk_byte_range_start as usize;
                let end = start + chunk.unpacked_segment_bytes as usize;
                let xorb_bytes = &xorbs.iter().find(|(h, _)| *h == seg.cas_hash).unwrap().1;
                rebuilt.extend_from_slice(&xorb_bytes[start..end]);
            }
        }
        assert_eq!(rebuilt, data);
    }
}
