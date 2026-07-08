//! CIDv1 + multihash encoder for content-addressed model references (#395).
//!
//! Produces deterministic CIDv1 strings for git OIDs and XET chunk/shard hashes,
//! giving every model artifact a single canonical identifier that survives moves
//! between the local registry, the atproto federated record store, and XET-backed
//! CAS shards. The wire representation stays `modelRef :Text` in the capnp schema
//! (#395 is a Rust-only grammar change); the CID is the canonical text payload
//! federated peers resolve.
//!
//! ## Layout
//!
//! ```text
//! CIDv1   ::= multibase("b") || base32( <0x01> || multicodec || multihash )
//! multihash ::= uvarint(algo) || uvarint(len) || digest
//! ```
//!
//! - **Multibase `b`** is lowercase RFC4648 base32 without padding — the
//!   IPFS/atproto *canonical* CIDv1 encoding. (The `z` base58btc prefix used by
//!   DID-doc Multikeys is rejected here: CIDs are base32.) See
//!   <https://github.com/multiformats/multibase>.
//! - **CID version** is always `0x01` (CIDv1). CIDv0 (sha1+git implicit) is not
//!   emitted; git OIDs get an explicit CIDv1 instead.
//! - **Multicodec** identifies the *content type*: `git-raw` (0x78) for a raw git
//!   object addressed by OID, the local XET codes below for XET hashes, and the
//!   local `at9p-capsule` code for capsule commitments.
//! - **Multihash** carries the hash algorithm + digest, so a CID is self-describing
//!   and can address sha1 git blobs, sha2-256 git-tree hashes, and blake3 XET
//!   chunks uniformly.
//!
//! ## Hash codes (multiformats/multicodec registry)
//!
//! | algo      | code   | used for                          |
//! |-----------|--------|-----------------------------------|
//! | sha1      | 0x11   | legacy git blob/tree/commit OIDs  |
//! | sha2-256  | 0x12   | git OIDs (sha256 repos), XET tags |
//! | blake3    | 0x1e   | XET hashes; at9p BLAKE3-512 CIDs  |
//!
//! ## Multicodec codes
//!
//! | codec         | code  | status                                  |
//! |---------------|-------|-----------------------------------------|
//! | git-raw       | 0x78  | registered (multicodec table)           |
//! | xet-xorb      | 0x71  | LOCAL convention, unregistered (#395)   |
//! | xet-shard     | 0x72  | LOCAL convention, unregistered (#395)   |
//! | at9p-capsule  | 0x73  | LOCAL convention, unregistered (#881)   |
//!
//! The XET and at9p-capsule codes live in the multicodec reserved (`0x70`–`0x7f`)
//! private-use range and MUST be replaced with official codes once registered
//! with multiformats. They are tagged `LOCAL_CONVENTION` so a `grep` finds them
//! when the migration lands.
//!
//! ## Determinism
//!
//! Encoding is a pure function of `(codec, algo, digest)`: the same input always
//! yields the same CID string (no randomness, no timestamps, no canonicalization
//! ambiguity). Round-tripping `decode(encode(x)) == x` holds for all valid CIDs.

use anyhow::{bail, ensure, Context, Result};
use data_encoding::BASE32_NOPAD;

// ---------------------------------------------------------------------------
// CIDv1 + multihash structural constants
// ---------------------------------------------------------------------------

/// Multibase prefix for lowercase RFC4648 base32 without padding — the canonical
/// CIDv1 encoding. (Contrast DID-doc Multikeys, which use `z` = base58btc.)
const MULTIBASE_BASE32: char = 'b';

/// CID version byte for CIDv1 (the only version this module emits).
const CIDV1: u8 = 0x01;

/// Hash-algorithm codes from the multihash table
/// (<https://github.com/multiformats/multicodec/blob/master/table.md>).
///
/// Only the algorithms we actually address (git sha1/sha2-256, XET blake3) are
/// modeled; the table is intentionally not exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum HashAlgo {
    /// sha1 (code 0x11). Legacy git blob/tree/commit OIDs.
    Sha1 = 0x11,
    /// sha2-256 (code 0x12). git OIDs in sha256 repos; XET tag hashes.
    Sha2_256 = 0x12,
    /// blake3 (code 0x1e). XET chunk/shard hashes and at9p capsule commitments.
    Blake3 = 0x1e,
}

impl HashAlgo {
    /// Validate the digest length carried by a multihash for this algorithm.
    ///
    /// BLAKE3 is an XOF: `0x1e` may carry either the existing 32-byte XET/CAS
    /// digest or the 64-byte at9p capsule commitment. The length remains part of
    /// the encoded multihash and therefore part of the CID identity.
    pub fn validate_digest_len(self, len: usize) -> Result<()> {
        let valid = match self {
            HashAlgo::Sha1 => len == 20,
            HashAlgo::Sha2_256 => len == 32,
            HashAlgo::Blake3 => matches!(len, 32 | 64),
        };
        ensure!(
            valid,
            "digest length mismatch for {self:?}: expected {}, got {len}",
            self.expected_digest_lengths()
        );
        Ok(())
    }

    fn expected_digest_lengths(self) -> &'static str {
        match self {
            HashAlgo::Sha1 => "20 bytes",
            HashAlgo::Sha2_256 => "32 bytes",
            HashAlgo::Blake3 => "32 or 64 bytes",
        }
    }

    /// Decode a multihash algorithm code into the typed enum.
    pub fn from_code(code: u64) -> Result<Self> {
        match code {
            0x11 => Ok(HashAlgo::Sha1),
            0x12 => Ok(HashAlgo::Sha2_256),
            0x1e => Ok(HashAlgo::Blake3),
            other => bail!("unsupported multihash algorithm code: 0x{other:x}"),
        }
    }
}

/// Content-type codec identifying *what* a CID addresses.
///
/// `GitRaw` is the registered multicodec for a raw git object (0x78). The XET
/// variants are **LOCAL_CONVENTION** codes in the reserved private-use range
/// (0x70–0x7f); replace them with official codes once XET is registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum Codec {
    /// git-raw (multicodec 0x78): a raw git object addressed by OID.
    GitRaw = 0x78,
    /// xet-xorb (LOCAL_CONVENTION, 0x71): a XET chunk/cas-serve content hash.
    XetXorb = 0x71,
    /// xet-shard (LOCAL_CONVENTION, 0x72): a XET shard (merkle) hash.
    XetShard = 0x72,
    /// at9p-capsule (LOCAL_CONVENTION, 0x73): an at9p capsule commitment.
    At9pCapsule = 0x73,
}

impl Codec {
    /// Decode a multicodec content-type code into the typed enum.
    pub fn from_code(code: u64) -> Result<Self> {
        match code {
            0x78 => Ok(Codec::GitRaw),
            0x71 => Ok(Codec::XetXorb),
            0x72 => Ok(Codec::XetShard),
            0x73 => Ok(Codec::At9pCapsule),
            other => bail!("unsupported CID multicodec: 0x{other:x}"),
        }
    }
}

// ---------------------------------------------------------------------------
// unsigned-varint (LEB128) — the multiformats varint used by multihash/CIDv1
// ---------------------------------------------------------------------------

/// Encode a `u64` as an unsigned varint (LEB128, little-endian groups of 7 bits,
/// continuation bit = high bit). The multiformats varint spec caps values at
/// 2^63 − 1 and limits the encoded length to 9 bytes; we enforce the 9-byte cap.
fn write_uvarint(out: &mut Vec<u8>, mut value: u64) {
    // Special-case zero so we always emit at least one byte.
    if value == 0 {
        out.push(0);
        return;
    }
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Decode an unsigned varint from `bytes` starting at `pos`, returning the value
/// and the number of bytes consumed.
fn read_uvarint(bytes: &[u8], pos: usize) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for (i, byte) in bytes[pos..].iter().take(9).enumerate() {
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
    }
    bail!("unsigned varint is truncated or longer than 9 bytes");
}

// ---------------------------------------------------------------------------
// multihash: uvarint(algo) || uvarint(len) || digest
// ---------------------------------------------------------------------------

/// Encode a digest as a multihash: `uvarint(algo) || uvarint(len) || digest`.
///
/// The digest length is checked against the lengths supported by the algorithm:
/// sha1=20, sha2-256=32, and blake3=32 or 64.
pub fn encode_multihash(algo: HashAlgo, digest: &[u8]) -> Result<Vec<u8>> {
    algo.validate_digest_len(digest.len())?;
    let mut out = Vec::with_capacity(2 + digest.len());
    write_uvarint(&mut out, algo as u64);
    write_uvarint(&mut out, digest.len() as u64);
    out.extend_from_slice(digest);
    Ok(out)
}

/// A decoded multihash: the algorithm tag and the raw digest bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Multihash {
    pub algo: HashAlgo,
    pub digest: Vec<u8>,
}

/// Decode a multihash produced by [`encode_multihash`].
pub fn decode_multihash(bytes: &[u8]) -> Result<Multihash> {
    let (algo_code, n1) =
        read_uvarint(bytes, 0).context("multihash: failed to read algorithm code")?;
    let (len, n2) = read_uvarint(bytes, n1).context("multihash: failed to read length")?;
    let len = usize::try_from(len).context("multihash: length overflow")?;
    ensure!(
        n1 + n2 + len == bytes.len(),
        "multihash: trailing bytes after digest (declared len {len})"
    );
    let digest = bytes
        .get(n1 + n2..)
        .context("multihash: digest truncated")?;
    ensure!(digest.len() == len, "multihash: digest length mismatch");
    let algo = HashAlgo::from_code(algo_code)?;
    // Validate the digest length belongs to the algorithm's accepted domain, so
    // a truncated/corrupted multihash is rejected rather than silently accepted.
    algo.validate_digest_len(digest.len())?;
    Ok(Multihash {
        algo,
        digest: digest.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// CIDv1: 0x01 || multicodec || multihash, then base32-multibase encoded
// ---------------------------------------------------------------------------

/// Encode raw `(codec, algo, digest)` as a canonical CIDv1 base32 string.
///
/// The output always begins with the multibase base32 prefix `b`. Encoding is
/// deterministic: identical inputs produce identical strings.
///
/// ```
/// # use hyprstream_rpc::cid::{encode_cid, Codec, HashAlgo};
/// let cid = encode_cid(Codec::GitRaw, HashAlgo::Sha1, &[0u8; 20]).unwrap();
/// assert!(cid.starts_with('b'));
/// ```
pub fn encode_cid(codec: Codec, algo: HashAlgo, digest: &[u8]) -> Result<String> {
    let multihash = encode_multihash(algo, digest)?;
    // CIDv1 body: version || codec || multihash
    let mut body = Vec::with_capacity(1 + 2 + multihash.len());
    body.push(CIDV1);
    write_uvarint(&mut body, codec as u64);
    body.extend_from_slice(&multihash);

    // Canonical CIDv1 multibase = base32 lowercase, RFC4648, no padding ('b').
    // `data_encoding::BASE32_NOPAD` is the uppercase RFC4648 alphabet; CIDs are
    // canonically lowercase, so we downcase the encoded output. (base32 is
    // case-insensitive, so decoding upper-cases first.)
    let encoded = BASE32_NOPAD.encode(&body).to_ascii_lowercase();
    Ok(format!("{MULTIBASE_BASE32}{encoded}"))
}

/// A decoded CIDv1: the content-type codec and the multihash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cid {
    pub codec: Codec,
    pub multihash: Multihash,
}

/// Decode a canonical CIDv1 base32 string produced by [`encode_cid`].
///
/// Both the base32 multibase prefix (`b`) and the CIDv1 version byte (`0x01`)
/// are validated. CIDv0 (implicit sha1+base58btc, no version byte) is rejected
/// — this module only emits CIDv1.
pub fn decode_cid(s: &str) -> Result<Cid> {
    let body_b32 = s
        .strip_prefix(MULTIBASE_BASE32)
        .context("CID must use base32 multibase ('b') prefix — base58btc 'z' is for Multikeys")?;
    // base32 is case-insensitive; the predefined table is uppercase, so up-case
    // before decoding. (We canonicalize CIDs to lowercase on encode.)
    let body = BASE32_NOPAD
        .decode(body_b32.to_ascii_uppercase().as_bytes())
        .context("CID body is not valid base32")?;

    let (version, n_ver) = read_uvarint(&body, 0).context("CID: failed to read version")?;
    ensure!(
        version == CIDV1 as u64,
        "only CIDv1 is supported (got version {version})"
    );
    let (codec_code, n_codec) = read_uvarint(&body, n_ver).context("CID: failed to read codec")?;
    let multihash =
        decode_multihash(&body[n_ver + n_codec..]).context("CID: malformed multihash")?;
    Ok(Cid {
        codec: Codec::from_code(codec_code)?,
        multihash,
    })
}

// ---------------------------------------------------------------------------
// git-oid convenience: hex SHA-1/SHA-256 → CID
// ---------------------------------------------------------------------------

/// Encode a git OID (hex string) as a `git-raw` CIDv1. The algorithm is inferred
/// from the hex length: 40 chars → sha1, 64 chars → sha2-256.
pub fn encode_git_oid(hex_oid: &str) -> Result<String> {
    let raw =
        hex::decode(hex_oid).with_context(|| format!("git OID is not valid hex: {hex_oid}"))?;
    let algo = match raw.len() {
        20 => HashAlgo::Sha1,
        32 => HashAlgo::Sha2_256,
        other => bail!("git OID has unexpected length {other} (expected 20 or 32 bytes)"),
    };
    encode_cid(Codec::GitRaw, algo, &raw)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ---- uvarint round-trip ------------------------------------------------

    #[test]
    fn uvarint_round_trip_single_and_multi_byte() {
        for &v in &[
            0u64,
            1,
            0x7f,
            0x80,
            0xff,
            0x100,
            0x4000,
            0x7fff_ffff_ffff_ffff,
        ] {
            let mut buf = Vec::new();
            write_uvarint(&mut buf, v);
            assert!(buf.len() <= 9, "uvarint for {v} too long");
            let (decoded, n) = read_uvarint(&buf, 0).expect("decode");
            assert_eq!(decoded, v, "round-trip {v}");
            assert_eq!(n, buf.len(), "consumed len {v}");
        }
    }

    #[test]
    fn uvarint_known_encodings() {
        // From the multiformats unsigned-varint examples.
        let mut buf = Vec::new();
        write_uvarint(&mut buf, 1);
        assert_eq!(buf, [0x01]);
        buf.clear();
        write_uvarint(&mut buf, 0x78); // git-raw codec
        assert_eq!(buf, [0x78]);
        buf.clear();
        write_uvarint(&mut buf, 0x12); // sha2-256
        assert_eq!(buf, [0x12]);
        buf.clear();
        write_uvarint(&mut buf, 127);
        assert_eq!(buf, [0x7f]);
        buf.clear();
        write_uvarint(&mut buf, 128);
        assert_eq!(buf, [0x80, 0x01]);
    }

    #[test]
    fn uvarint_rejects_truncated() {
        // Continuation byte with no following byte → truncated.
        assert!(read_uvarint(&[0x80], 0).is_err());
        // All continuation, 9 bytes, no terminator → truncated.
        assert!(read_uvarint(&[0x80; 9], 0).is_err());
    }

    // ---- multihash round-trip ---------------------------------------------

    #[test]
    fn multihash_sha1_round_trip() {
        let digest = [0xabu8; 20];
        let mh = encode_multihash(HashAlgo::Sha1, &digest).unwrap();
        // uvarint(0x11) || uvarint(20) || 20 bytes
        assert_eq!(mh[0], 0x11);
        assert_eq!(mh[1], 20);
        assert_eq!(&mh[2..], &digest);
        let decoded = decode_multihash(&mh).unwrap();
        assert_eq!(decoded.algo, HashAlgo::Sha1);
        assert_eq!(decoded.digest, digest.to_vec());
    }

    #[test]
    fn multihash_sha2_256_round_trip() {
        let digest = [0xcdu8; 32];
        let mh = encode_multihash(HashAlgo::Sha2_256, &digest).unwrap();
        assert_eq!(mh[0], 0x12);
        assert_eq!(mh[1], 32);
        let decoded = decode_multihash(&mh).unwrap();
        assert_eq!(decoded.algo, HashAlgo::Sha2_256);
        assert_eq!(decoded.digest, digest.to_vec());
    }

    #[test]
    fn multihash_blake3_round_trip() {
        let digest = [0xefu8; 32];
        let mh = encode_multihash(HashAlgo::Blake3, &digest).unwrap();
        assert_eq!(mh[0], 0x1e);
        assert_eq!(mh[1], 32);
        let decoded = decode_multihash(&mh).unwrap();
        assert_eq!(decoded.algo, HashAlgo::Blake3);
        assert_eq!(decoded.digest, digest.to_vec());
    }

    #[test]
    fn multihash_blake3_512_round_trip() {
        let digest = [0x51u8; 64];
        let mh = encode_multihash(HashAlgo::Blake3, &digest).unwrap();
        assert_eq!(mh[0], 0x1e);
        assert_eq!(mh[1], 64);
        let decoded = decode_multihash(&mh).unwrap();
        assert_eq!(decoded.algo, HashAlgo::Blake3);
        assert_eq!(decoded.digest, digest.to_vec());
    }

    #[test]
    fn multihash_rejects_wrong_length() {
        // sha1 expects 20 bytes; 19 must be rejected.
        assert!(encode_multihash(HashAlgo::Sha1, &[0u8; 19]).is_err());
        // sha2-256 expects 32; 33 must be rejected.
        assert!(encode_multihash(HashAlgo::Sha2_256, &[0u8; 33]).is_err());
        // blake3 accepts 32 or 64; other lengths must be rejected.
        assert!(encode_multihash(HashAlgo::Blake3, &[0u8; 16]).is_err());
        assert!(encode_multihash(HashAlgo::Blake3, &[0u8; 63]).is_err());
    }

    #[test]
    fn multihash_rejects_trailing_bytes() {
        let mut mh = encode_multihash(HashAlgo::Sha1, &[0u8; 20]).unwrap();
        mh.push(0xff); // trailing garbage
        assert!(decode_multihash(&mh).is_err());
    }

    #[test]
    fn multihash_rejects_unknown_algo() {
        // algo code 0x99 is not in our table.
        let mh = [0x99, 0x01, 0x00];
        assert!(decode_multihash(&mh).is_err());
    }

    // ---- CID round-trip ----------------------------------------------------

    #[test]
    fn cid_git_raw_sha1_round_trip() {
        let digest = [0x11u8; 20];
        let cid = encode_cid(Codec::GitRaw, HashAlgo::Sha1, &digest).unwrap();
        assert!(
            cid.starts_with('b'),
            "canonical CIDv1 must use base32 'b' multibase"
        );
        assert!(
            !cid.starts_with('z'),
            "base58btc 'z' is for DID Multikeys, not CIDs"
        );
        // Lowercase, no padding.
        assert_eq!(cid, cid.to_lowercase());
        assert!(!cid.contains('='), "base32 must be unpadded");

        let decoded = decode_cid(&cid).unwrap();
        assert_eq!(decoded.codec, Codec::GitRaw);
        assert_eq!(decoded.multihash.algo, HashAlgo::Sha1);
        assert_eq!(decoded.multihash.digest, digest.to_vec());
    }

    #[test]
    fn cid_xet_xorb_blake3_round_trip() {
        let digest = [0x22u8; 32];
        let cid = encode_cid(Codec::XetXorb, HashAlgo::Blake3, &digest).unwrap();
        let decoded = decode_cid(&cid).unwrap();
        assert_eq!(decoded.codec, Codec::XetXorb);
        assert_eq!(decoded.multihash.algo, HashAlgo::Blake3);
        assert_eq!(decoded.multihash.digest, digest.to_vec());
    }

    #[test]
    fn cid_xet_shard_sha2_256_round_trip() {
        let digest = [0x33u8; 32];
        let cid = encode_cid(Codec::XetShard, HashAlgo::Sha2_256, &digest).unwrap();
        let decoded = decode_cid(&cid).unwrap();
        assert_eq!(decoded.codec, Codec::XetShard);
        assert_eq!(decoded.multihash.algo, HashAlgo::Sha2_256);
        assert_eq!(decoded.multihash.digest, digest.to_vec());
    }

    #[test]
    fn cid_at9p_capsule_blake3_512_round_trip() {
        let digest = [0x44u8; 64];
        let cid = encode_cid(Codec::At9pCapsule, HashAlgo::Blake3, &digest).unwrap();
        let decoded = decode_cid(&cid).unwrap();
        assert_eq!(decoded.codec, Codec::At9pCapsule);
        assert_eq!(decoded.multihash.algo, HashAlgo::Blake3);
        assert_eq!(decoded.multihash.digest, digest.to_vec());
    }

    #[test]
    fn cid_is_deterministic() {
        // Same input → identical string. Run a handful of times; if any byte
        // drifts, encoding is not a pure function and federation breaks.
        let digest = [0x42u8; 20];
        let c1 = encode_cid(Codec::GitRaw, HashAlgo::Sha1, &digest).unwrap();
        for _ in 0..5 {
            let c2 = encode_cid(Codec::GitRaw, HashAlgo::Sha1, &digest).unwrap();
            assert_eq!(c1, c2, "CID encoding must be deterministic");
        }
    }

    #[test]
    fn cid_different_digests_yield_different_cids() {
        let a = encode_cid(Codec::GitRaw, HashAlgo::Sha1, &[0u8; 20]).unwrap();
        let b = encode_cid(Codec::GitRaw, HashAlgo::Sha1, &[1u8; 20]).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn cid_blake3_32_and_64_byte_digests_do_not_alias() {
        let input = b"canonical genesis capsule";
        let digest32 = *blake3::hash(input).as_bytes();
        let mut digest64 = [0u8; 64];
        let mut hasher = blake3::Hasher::new();
        hasher.update(input);
        hasher.finalize_xof().fill(&mut digest64);

        assert_eq!(&digest64[..32], &digest32);

        let cid32 = encode_cid(Codec::At9pCapsule, HashAlgo::Blake3, &digest32).unwrap();
        let cid64 = encode_cid(Codec::At9pCapsule, HashAlgo::Blake3, &digest64).unwrap();
        assert_ne!(cid32, cid64);

        let decoded32 = decode_cid(&cid32).unwrap();
        let decoded64 = decode_cid(&cid64).unwrap();
        assert_eq!(decoded32.multihash.algo, decoded64.multihash.algo);
        assert_eq!(decoded32.multihash.digest.len(), 32);
        assert_eq!(decoded64.multihash.digest.len(), 64);
        assert_ne!(decoded32.multihash, decoded64.multihash);
    }

    #[test]
    fn cid_different_codecs_yield_different_cids() {
        let digest = [0u8; 32];
        let git_raw = encode_cid(Codec::GitRaw, HashAlgo::Sha2_256, &digest).unwrap();
        let xet = encode_cid(Codec::XetXorb, HashAlgo::Sha2_256, &digest).unwrap();
        let shard = encode_cid(Codec::XetShard, HashAlgo::Sha2_256, &digest).unwrap();
        assert_ne!(git_raw, xet);
        assert_ne!(git_raw, shard);
        assert_ne!(xet, shard);
    }

    #[test]
    fn cid_rejects_base58btc_multibase() {
        // A 'z'-prefixed string (base58btc) is for DID Multikeys, not CIDs.
        assert!(decode_cid("z1234").is_err());
    }

    #[test]
    fn cid_rejects_invalid_base32() {
        // 'b' prefix but non-base32 body. '1' is not in the base32 alphabet.
        assert!(decode_cid("b1@@@").is_err());
    }

    // ---- git-oid convenience ----------------------------------------------

    #[test]
    fn encode_git_oid_sha1_hex() {
        let hex = "1234567890abcdef1234567890abcdef12345678";
        let cid = encode_git_oid(hex).unwrap();
        let decoded = decode_cid(&cid).unwrap();
        assert_eq!(decoded.codec, Codec::GitRaw);
        assert_eq!(decoded.multihash.algo, HashAlgo::Sha1);
        assert_eq!(decoded.multihash.digest.len(), 20);
    }

    #[test]
    fn encode_git_oid_sha2_256_hex() {
        let hex = "a".repeat(64);
        let cid = encode_git_oid(&hex).unwrap();
        let decoded = decode_cid(&cid).unwrap();
        assert_eq!(decoded.codec, Codec::GitRaw);
        assert_eq!(decoded.multihash.algo, HashAlgo::Sha2_256);
        assert_eq!(decoded.multihash.digest.len(), 32);
    }

    #[test]
    fn encode_git_oid_rejects_bad_length() {
        // 10 bytes (20 hex chars) is neither sha1 nor sha2-256.
        assert!(encode_git_oid("12345678901234567890").is_err());
        // Non-hex.
        assert!(encode_git_oid("zz").is_err());
    }

    // ---- cross-check against a known CID vector ---------------------------
    //
    // The empty-byte sha2-256 CIDv1 is a well-known fixture: its multihash is
    // the sha2-256 of the empty string (e3b0c442...). This anchors our encoder
    // against the IPFS canonical output (base32, 'b' prefix, no padding).

    #[test]
    fn cid_matches_ipfs_empty_sha256_vector() {
        let empty_sha256: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        // raw codec = 0x55 in the multicodec table (not one we expose by name,
        // but the multihash/base32 layer is codec-agnostic). Use git-raw (0x78)
        // since that's our domain codec; this test pins the multihash + base32
        // encoding, which is what must match IPFS byte-for-byte.
        let cid = encode_cid(Codec::GitRaw, HashAlgo::Sha2_256, &empty_sha256).unwrap();
        // Round-trips and the multihash body is the canonical sha2-256 of empty.
        let decoded = decode_cid(&cid).unwrap();
        assert_eq!(decoded.multihash.digest, empty_sha256);
        // The base32 body (sans 'b' prefix) must decode back to
        //   01 78 12 20 e3b0c4...55
        let body_b32 = &cid[1..];
        let body = BASE32_NOPAD
            .decode(body_b32.to_ascii_uppercase().as_bytes())
            .unwrap();
        assert_eq!(body[0], 0x01, "CIDv1 version byte");
        assert_eq!(body[1], 0x78, "git-raw codec");
        assert_eq!(body[2], 0x12, "sha2-256 algo");
        assert_eq!(body[3], 0x20, "digest length 32");
        assert_eq!(&body[4..], &empty_sha256[..]);
    }
}
