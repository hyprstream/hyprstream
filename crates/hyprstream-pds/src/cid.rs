//! CIDv1 (Content IDentifier) for DAG-CBOR blocks.
//!
//! A CID is a self-describing content address. This crate uses **CIDv1** with
//! the `dag-cbor` codec (0x71) and `sha2-256` multihash (0x12) — the atproto /
//! IPLD default for repo blocks (commits, MST nodes, records).
//!
//! # Wire format (CIDv1)
//!
//! ```text
//! <version:u8=0x01><codec:varint><multihash>
//! ```
//! where `<multihash> = <hash-code:varint><digest-len:varint><digest-bytes>`.
//!
//! # DAG-CBOR link encoding (tag 42)
//!
//! Inside DAG-CBOR, a CID is carried as a byte string tagged with CBOR tag 42,
//! whose payload is a "CID specifier": a leading 0x00 (identity-multibase
//! marker, per the [DAG-CBOR spec](https://github.com/ipld/specs/blob/master/block-layer/codecs/dag-cbor.md))
//! followed by the raw CID bytes (version+codec+multihash). The 0x00 prefix
//! exists so a CID can never collide with a legitimate CBOR byte string.

use std::fmt;

use anyhow::{anyhow, bail, Result};

/// Codec code for `dag-cbor` (0x71) — used for atproto repo blocks.
pub const CODEC_DAG_CBOR: u64 = 0x71;
/// Codec code for `raw` (0x55) — opaque byte blocks.
pub const CODEC_RAW: u64 = 0x55;

/// Multihash code for SHA-256 (0x12) — the atproto / IPLD default.
const MH_SHA2_256: u64 = 0x12;
const SHA256_LEN: usize = 32;

/// A CIDv1 content identifier.
///
/// Internally stored as a fixed-size byte buffer holding the full CID bytes
/// (`0x01 <codec> <multihash>`), plus a length. CIDv1 with a single-byte codec
/// varint and SHA-256 multihash is exactly 38 bytes, so a 40-byte inline buffer
/// covers all realistic multihashes without a heap allocation. This makes
/// [`Cid`] [`Copy`] (38-ish bytes is cheaper to copy than to refcount), which
/// matters because CIDs flow through tree node fields, map keys, and proof
/// steps that would otherwise drown in `.clone()` noise.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Cid {
    /// Raw CID bytes: `0x01 <codec-varint> <multihash>`.
    bytes: [u8; Self::CAP],
    /// Number of valid bytes in `bytes`.
    len: u8,
}

impl Cid {
    /// Maximum CID byte length supported (CIDv1 + dag-cbor codec + sha2-256
    /// multihash = 38 bytes; the slack covers longer multihashes).
    const CAP: usize = 40;

    /// Compute the CIDv1 `dag-cbor` CID over `block_bytes` (SHA-256 multihash).
    ///
    /// This is the canonical way to address a DAG-CBOR block in atproto: the
    /// caller hands in the already-canonical DAG-CBOR bytes (see [`crate::dag_cbor`])
    /// and gets back the CID that identifies them.
    pub fn from_dag_cbor(block_bytes: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(38);
        bytes.push(0x01); // CIDv1
        write_uvarint(CODEC_DAG_CBOR, &mut bytes);
        // multihash: sha2-256 code + length + digest
        write_uvarint(MH_SHA2_256, &mut bytes);
        write_uvarint(SHA256_LEN as u64, &mut bytes);
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(block_bytes);
        bytes.extend_from_slice(&digest);
        Cid::from_vec(bytes)
    }

    /// CIDv1 `raw` codec over arbitrary bytes (used for external git OIDs
    /// referenced by `currentOid` when the caller has the raw blob handy).
    pub fn from_raw(blob_bytes: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(38);
        bytes.push(0x01);
        write_uvarint(CODEC_RAW, &mut bytes);
        write_uvarint(MH_SHA2_256, &mut bytes);
        write_uvarint(SHA256_LEN as u64, &mut bytes);
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(blob_bytes);
        bytes.extend_from_slice(&digest);
        Cid::from_vec(bytes)
    }

    /// Pack a `Vec<u8>` of CID bytes into the fixed-size inline buffer.
    fn from_vec(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        assert!(
            len <= Self::CAP,
            "CID exceeds inline buffer: {len} > {}",
            Self::CAP
        );
        let mut buf = [0u8; Self::CAP];
        buf[..len].copy_from_slice(&bytes);
        Cid {
            bytes: buf,
            len: len as u8,
        }
    }

    /// Raw CID bytes (version + codec + multihash).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Encode as a DAG-CBOR link: the CBOR-tag-42 payload, i.e. a leading
    /// `0x00` identity-multibase marker followed by the raw CID bytes.
    ///
    /// Returns the bytes that go *inside* the tag-42 byte string (the marker
    /// plus the CID). Used by [`crate::dag_cbor`] when emitting link values.
    pub fn to_link_bytes(&self) -> Vec<u8> {
        let bytes = self.as_bytes();
        let mut out = Vec::with_capacity(1 + bytes.len());
        out.push(0x00);
        out.extend_from_slice(bytes);
        out
    }

    /// Parse a DAG-CBOR tag-42 payload (leading 0x00 marker + CID) back into a [`Cid`].
    pub fn from_link_bytes(link: &[u8]) -> Result<Self> {
        if link.is_empty() {
            bail!("empty DAG-CBOR link payload");
        }
        // The first byte is the multibase identity marker (0x00); strip it.
        if link[0] != 0x00 {
            bail!(
                "DAG-CBOR link payload must start with 0x00 identity marker, got 0x{:02x}",
                link[0]
            );
        }
        Self::from_bytes(&link[1..])
    }

    /// Parse raw CID bytes (version + codec + multihash).
    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        if raw.is_empty() {
            bail!("empty CID bytes");
        }
        if raw[0] != 0x01 {
            bail!(
                "only CIDv1 is supported (got version byte 0x{:02x})",
                raw[0]
            );
        }
        // Validate the varints parse and the digest length matches.
        let rest = &raw[1..];
        let (_codec, rest) = read_uvarint(rest).ok_or_else(|| anyhow!("truncated codec varint"))?;
        let (code, rest) = read_uvarint(rest).ok_or_else(|| anyhow!("truncated multihash code"))?;
        let (len, rest) =
            read_uvarint(rest).ok_or_else(|| anyhow!("truncated multihash length"))?;
        let len = len as usize;
        if rest.len() != len {
            bail!(
                "multihash length {len} does not match remaining {} bytes",
                rest.len()
            );
        }
        if code != MH_SHA2_256 {
            bail!("only sha2-256 multihash is supported (got code {code})");
        }
        if raw.len() > Self::CAP {
            bail!("CID exceeds inline buffer: {} > {}", raw.len(), Self::CAP);
        }
        Ok(Cid::from_vec(raw.to_vec()))
    }

    /// Base32 (lowercase, no padding) `bafy...`/`bafk...`-style encoding — the
    /// canonical string form of a CIDv1.
    ///
    /// Named `encode` rather than `to_string` to avoid shadowing the
    /// `ToString` blanket impl that `Display` would otherwise recurse through.
    ///
    /// This is the [CIDv1 string format](https://github.com/multiformats/multibase):
    /// multibase prefix `b` (base32lower-nopad) followed by the base32 of the
    /// raw CID bytes.
    pub fn encode(&self) -> String {
        let bytes = self.as_bytes();
        let mut out = String::with_capacity(1 + bytes.len() * 2);
        out.push('b');
        out.push_str(&base32_nopad_lower(bytes));
        out
    }
}

impl fmt::Debug for Cid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cid({})", self.encode())
    }
}

impl fmt::Display for Cid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.encode())
    }
}

impl PartialOrd for Cid {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cid {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.bytes.cmp(&other.bytes)
    }
}

// ── varint helpers (LEB128 unsigned, RFC 7541 / multiformats) ────────────────

/// Write a multiformats unsigned varint (7 bits/byte, MSB continuation).
pub(crate) fn write_uvarint(mut val: u64, out: &mut Vec<u8>) {
    while val >= 0x80 {
        out.push(((val & 0x7f) as u8) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

/// Read a multiformats unsigned varint. Returns `(value, remaining_slice)`.
pub(crate) fn read_uvarint(input: &[u8]) -> Option<(u64, &[u8])> {
    let mut val: u64 = 0;
    let mut shift = 0u32;
    for (i, &b) in input.iter().enumerate() {
        if shift >= 64 {
            return None; // varint too long
        }
        val |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((val, &input[i + 1..]));
        }
        shift += 7;
    }
    None // truncated
}

/// RFC 4648 base32, lowercase, no padding.
fn base32_nopad_lower(bytes: &[u8]) -> String {
    const ALPHA: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut out = String::with_capacity((bytes.len() * 8).div_ceil(5));
    let mut buffer: u64 = 0;
    let mut bits: u32 = 0;
    for &b in bytes {
        buffer = (buffer << 8) | (b as u64);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let idx = ((buffer >> bits) & 0x1f) as usize;
            out.push(ALPHA[idx] as char);
        }
    }
    if bits > 0 {
        let idx = ((buffer << (5 - bits)) & 0x1f) as usize;
        out.push(ALPHA[idx] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;

    #[test]
    fn cid_is_deterministic() {
        // Same bytes → same CID (load-bearing for MST + commit sig).
        let a = Cid::from_dag_cbor(b"hello world");
        let b = Cid::from_dag_cbor(b"hello world");
        assert_eq!(a, b);
        assert_eq!(a.as_bytes(), b.as_bytes());
        assert_ne!(
            Cid::from_dag_cbor(b"hello world"),
            Cid::from_dag_cbor(b"hello earth")
        );
    }

    #[test]
    fn cid_round_trip_link_bytes() {
        let cid = Cid::from_dag_cbor(b"some dag-cbor block");
        let link = cid.to_link_bytes();
        assert_eq!(link[0], 0x00);
        let parsed = Cid::from_link_bytes(&link).expect("round-trip");
        assert_eq!(cid, parsed);
    }

    #[test]
    fn cid_round_trip_from_bytes() {
        let cid = Cid::from_dag_cbor(b"x");
        let parsed = Cid::from_bytes(cid.as_bytes()).expect("round-trip");
        assert_eq!(cid, parsed);
    }

    #[test]
    fn cid_string_starts_with_b() {
        let cid = Cid::from_dag_cbor(b"");
        let s = cid.encode();
        assert!(s.starts_with('b'), "CIDv1 base32 must start with 'b': {s}");
    }

    #[test]
    fn varint_round_trip() {
        for val in [
            0u64,
            1,
            0x7f,
            0x80,
            0xff,
            0x71,
            0x12,
            0x4000,
            u32::MAX as u64,
        ] {
            let mut buf = Vec::new();
            write_uvarint(val, &mut buf);
            let (parsed, rest) = read_uvarint(&buf).expect("round-trip");
            assert_eq!(parsed, val, "varint {val}");
            assert!(rest.is_empty(), "varint {val} had trailing bytes");
        }
    }
}
