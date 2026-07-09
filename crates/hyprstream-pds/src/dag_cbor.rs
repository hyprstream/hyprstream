//! Deterministic DAG-CBOR encoding/decoding.
//!
//! DAG-CBOR is a restricted subset of [CBOR (RFC 7049)](https://tools.ietf.org/html/rfc7049)
//! with additional constraints so that a given value always produces the
//! **exact same bytes** — and therefore the same CID. This determinism is
//! load-bearing for the MST (whose internal CIDs are computed over DAG-CBOR
//! node bytes) and for the commit signature (which signs DAG-CBOR bytes).
//!
//! # Constraints enforced here
//!
//! 1. **Map keys are sorted** in pure lexicographic byte order
//!    (RFC 7049 §4.2.1 "core determinism") — the convention atproto's
//!    `@atproto/lex-cbor` (via `cborg`) uses. (NOT length-first-then-lex; that's
//!    the older §3.9 "canonical CBOR" rule, which atproto rejects because it
//!    would produce different bytes for the same record.)
//! 2. **Integers use minimal encoding**: `u8` < 24 → one byte (major 0/1);
//!    otherwise the smallest power-of-two width (1/2/4/8 bytes).
//! 3. **No duplicate keys**: decoding rejects them.
//! 4. **CIDs are CBOR tag 42** byte strings carrying the DAG-CBOR link payload
//!    (leading 0x00 identity marker + raw CID bytes), per the
//!    [DAG-CBOR spec](https://github.com/ipld/specs/blob/master/block-layer/codecs/dag-cbor.md).
//! 5. **Tag 42 byte strings are reserved**: any non-CID byte string is decoded
//!    as [`DagCbor::Bytes`], never confused with a link.
//!
//! # Why a custom enum rather than `serde` + `ciborium` directly
//!
//! `serde`-driven CBOR cannot (a) guarantee map key ordering for arbitrary
//! structs without a manual sorted-map type, nor (b) round-trip CBOR tag 42
//! links as first-class values. By going through a small typed [`DagCbor`]
//! enum we make both invariants explicit and verifiable in tests (same record
//! → same bytes → same CID).

use anyhow::{anyhow, bail, ensure, Result};

use crate::cid::Cid;

/// CBOR major types (RFC 7049 §3.1).
const MT_UNSIGNED: u8 = 0;
const MT_NEGATIVE: u8 = 1;
const MT_BYTES: u8 = 2;
const MT_TEXT: u8 = 3;
const MT_ARRAY: u8 = 4;
const MT_MAP: u8 = 5;
const MT_TAG: u8 = 6;
const MT_SIMPLE: u8 = 7;

/// The CBOR tag identifying a DAG-CBOR link (CID).
const TAG_CID: u64 = 42;

/// A typed DAG-CBOR value.
///
/// Construction is deliberate (no `From<&str>` footguns): callers build the
/// exact shape they need via [`DagCbor::map`] / [`DagCbor::list`] / etc., so
/// the determinism rules (sorted keys, minimal ints, tag-42 links) are enforced
/// at encode time rather than relying on the input order.
#[derive(Clone, Debug, PartialEq)]
pub enum DagCbor {
    Null,
    Bool(bool),
    /// Non-negative integer (CBOR major 0).
    Unsigned(u64),
    /// Negative integer (CBOR major 1); `-1 - n`.
    Negative(i128),
    /// Byte string (major 2).
    Bytes(Vec<u8>),
    /// UTF-8 text string (major 3).
    Text(String),
    /// Array (major 4), in given order.
    List(Vec<DagCbor>),
    /// Map (major 5). Keys are stored in canonical (pure lexicographic byte)
    /// sorted order at construction; encoding emits them as-is.
    Map(Vec<(DagCbor, DagCbor)>),
    /// CID link — encoded as CBOR tag 42.
    Link(Cid),
}

impl DagCbor {
    /// Convenience: empty map.
    pub fn map() -> Self {
        DagCbor::Map(Vec::new())
    }

    /// Convenience: build a map from key/value string pairs, sorted canonically.
    ///
    /// Only string keys are supported here (the common case for records/commits).
    /// For non-string keys build the `Map` variant manually.
    pub fn str_map(pairs: impl IntoIterator<Item = (impl Into<String>, DagCbor)>) -> Self {
        let mut v: Vec<(String, DagCbor)> = pairs.into_iter().map(|(k, v)| (k.into(), v)).collect();
        v.sort_by(|a, b| canonical_key_cmp(a.0.as_bytes(), b.0.as_bytes()));
        // Reject duplicate keys — they'd silently collapse in CBOR and produce
        // a CID-collision risk across implementations.
        if let Some(i) = v.windows(2).position(|w| w[0].0 == w[1].0) {
            // This is a programmer error; surface it loudly via debug assert.
            debug_assert!(false, "duplicate DAG-CBOR map key: {:?}", v[i].0);
        }
        let sorted = v
            .into_iter()
            .map(|(k, val)| (DagCbor::Text(k), val))
            .collect();
        DagCbor::Map(sorted)
    }

    /// Convenience: build a list.
    pub fn list(items: impl IntoIterator<Item = DagCbor>) -> Self {
        DagCbor::List(items.into_iter().collect())
    }

    // ── Encoding ────────────────────────────────────────────────────────────

    /// Encode to canonical DAG-CBOR bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut Vec<u8>) {
        match self {
            DagCbor::Null => write_head(out, MT_SIMPLE, 22), // 0xf6
            DagCbor::Bool(false) => write_head(out, MT_SIMPLE, 20), // 0xf4
            DagCbor::Bool(true) => write_head(out, MT_SIMPLE, 21), // 0xf5
            DagCbor::Unsigned(n) => write_uint(out, MT_UNSIGNED, *n),
            DagCbor::Negative(n) => {
                // CBOR encodes negatives as -1 - n in major type 1 (uint).
                // n is < 0 here.
                let val = (-1_i128 - n) as u64;
                write_uint(out, MT_NEGATIVE, val);
            }
            DagCbor::Bytes(b) => {
                write_uint(out, MT_BYTES, b.len() as u64);
                out.extend_from_slice(b);
            }
            DagCbor::Text(s) => {
                write_uint(out, MT_TEXT, s.len() as u64);
                out.extend_from_slice(s.as_bytes());
            }
            DagCbor::List(items) => {
                write_uint(out, MT_ARRAY, items.len() as u64);
                for it in items {
                    it.write(out);
                }
            }
            DagCbor::Map(pairs) => {
                write_uint(out, MT_MAP, pairs.len() as u64);
                for (k, v) in pairs {
                    k.write(out);
                    v.write(out);
                }
            }
            DagCbor::Link(cid) => {
                // Tag 42 + byte string of the link payload (0x00 marker + CID).
                write_uint(out, MT_TAG, TAG_CID);
                let link = cid.to_link_bytes();
                write_uint(out, MT_BYTES, link.len() as u64);
                out.extend_from_slice(&link);
            }
        }
    }

    // ── Decoding ────────────────────────────────────────────────────────────

    /// Decode canonical DAG-CBOR bytes, enforcing the determinism constraints
    /// (sorted keys, no duplicates, minimal-int width on the wire is accepted
    /// but a re-encode will produce minimal form).
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut cursor = 0usize;
        let val = Self::read(input, &mut cursor)?;
        ensure!(cursor == input.len(), "trailing bytes after DAG-CBOR value");
        Ok(val)
    }

    fn read(input: &[u8], cursor: &mut usize) -> Result<Self> {
        let head = *input
            .get(*cursor)
            .ok_or_else(|| anyhow!("truncated DAG-CBOR (need at least one byte)"))?;
        *cursor += 1;
        let major = head >> 5;
        let info = head & 0x1f;

        // Tag 42 → link.
        if major == MT_TAG {
            let tag = read_arg(info, input, cursor)?;
            ensure!(
                tag == TAG_CID,
                "only CBOR tag 42 (CID link) is supported, got {tag}"
            );
            // Next item must be a byte string carrying the link payload.
            return Self::read_link_payload(input, cursor);
        }

        match major {
            MT_UNSIGNED => Ok(DagCbor::Unsigned(read_arg(info, input, cursor)?)),
            MT_NEGATIVE => {
                let n = read_arg(info, input, cursor)?;
                // Decode major-1 uint into -1 - n as i128.
                Ok(DagCbor::Negative(-1_i128 - n as i128))
            }
            MT_BYTES => {
                let len = read_arg(info, input, cursor)? as usize;
                let b = take(input, cursor, len)?;
                Ok(DagCbor::Bytes(b.to_vec()))
            }
            MT_TEXT => {
                let len = read_arg(info, input, cursor)? as usize;
                let b = take(input, cursor, len)?;
                let s =
                    String::from_utf8(b.to_vec()).map_err(|_| anyhow!("invalid UTF-8 in text"))?;
                Ok(DagCbor::Text(s))
            }
            MT_ARRAY => {
                let len = read_arg(info, input, cursor)? as usize;
                // Cap preallocation by what the remaining input could possibly
                // hold (each element needs at least 1 byte) — the length arg is
                // untrusted and a huge value must not trigger a capacity panic.
                let mut items = Vec::with_capacity(len.min(input.len().saturating_sub(*cursor)));
                for _ in 0..len {
                    items.push(Self::read(input, cursor)?);
                }
                Ok(DagCbor::List(items))
            }
            MT_MAP => {
                let len = read_arg(info, input, cursor)? as usize;
                // Same untrusted-length cap as the array arm (each entry needs
                // at least 2 bytes, so remaining-input is a safe upper bound).
                let mut pairs: Vec<(DagCbor, DagCbor)> =
                    Vec::with_capacity(len.min(input.len().saturating_sub(*cursor)));
                let mut prev_key: Option<Vec<u8>> = None;
                for _ in 0..len {
                    let k = Self::read(input, cursor)?;
                    let v = Self::read(input, cursor)?;
                    // Enforce sorted + unique keys: compare against previous.
                    let key_bytes = canonical_key_of(&k);
                    if let Some(ref prev) = prev_key {
                        ensure!(
                            canonical_key_cmp(prev, &key_bytes) == std::cmp::Ordering::Less,
                            "DAG-CBOR map keys not in canonical order / duplicate"
                        );
                    }
                    prev_key = Some(key_bytes);
                    pairs.push((k, v));
                }
                Ok(DagCbor::Map(pairs))
            }
            MT_TAG => bail!("unexpected CBOR tag (only tag 42/CID is supported)"),
            MT_SIMPLE => match info {
                20 => Ok(DagCbor::Bool(false)),
                21 => Ok(DagCbor::Bool(true)),
                22 => Ok(DagCbor::Null),
                _ => bail!("unsupported CBOR simple value {info}"),
            },
            _ => bail!("unknown CBOR major type {major}"),
        }
    }

    fn read_link_payload(input: &[u8], cursor: &mut usize) -> Result<Self> {
        // Expect a byte-string item: head + payload.
        let head = *input
            .get(*cursor)
            .ok_or_else(|| anyhow!("truncated DAG-CBOR link payload"))?;
        *cursor += 1;
        let major = head >> 5;
        let info = head & 0x1f;
        ensure!(
            major == MT_BYTES,
            "tag-42 must be followed by a byte string"
        );
        let len = read_arg(info, input, cursor)? as usize;
        let payload = take(input, cursor, len)?;
        let cid = Cid::from_link_bytes(payload)?;
        Ok(DagCbor::Link(cid))
    }

    // ── Helpers to extract typed values (used by record/commit decode) ──────

    pub fn as_str(&self) -> Result<&str> {
        match self {
            DagCbor::Text(s) => Ok(s),
            _ => bail!("expected text string, got {:?}", self),
        }
    }
    pub fn as_bytes(&self) -> Result<&[u8]> {
        match self {
            DagCbor::Bytes(b) => Ok(b),
            _ => bail!("expected byte string, got {:?}", self),
        }
    }
    pub fn as_unsigned(&self) -> Result<u64> {
        match self {
            DagCbor::Unsigned(n) => Ok(*n),
            _ => bail!("expected unsigned int, got {:?}", self),
        }
    }
    pub fn as_link(&self) -> Result<&Cid> {
        match self {
            DagCbor::Link(c) => Ok(c),
            _ => bail!("expected CID link, got {:?}", self),
        }
    }
    pub fn as_map(&self) -> Result<&[(DagCbor, DagCbor)]> {
        match self {
            DagCbor::Map(v) => Ok(v),
            _ => bail!("expected map, got {:?}", self),
        }
    }
    pub fn as_list(&self) -> Result<&[DagCbor]> {
        match self {
            DagCbor::List(v) => Ok(v),
            _ => bail!("expected list, got {:?}", self),
        }
    }
    /// Look up a string-keyed entry in a map.
    pub fn get(&self, key: &str) -> Option<&DagCbor> {
        if let DagCbor::Map(v) = self {
            v.iter().find_map(|(k, val)| match k {
                DagCbor::Text(s) if s == key => Some(val),
                _ => None,
            })
        } else {
            None
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, DagCbor::Null)
    }
}

// ── encoding primitives ──────────────────────────────────────────────────────

fn write_head(out: &mut Vec<u8>, major: u8, info: u8) {
    out.push((major << 5) | info);
}

/// Write a CBOR head with a major type and an unsigned argument, choosing the
/// minimal-width encoding (RFC 7049 §3.9 determinism).
fn write_uint(out: &mut Vec<u8>, major: u8, val: u64) {
    let m = major << 5;
    if val < 24 {
        out.push(m | val as u8);
    } else if val < 0x1_00 {
        out.push(m | 24);
        out.push(val as u8);
    } else if val < 0x1_0000 {
        out.push(m | 25);
        out.extend_from_slice(&(val as u16).to_be_bytes());
    } else if val < 0x1_0000_0000 {
        out.push(m | 26);
        out.extend_from_slice(&(val as u32).to_be_bytes());
    } else {
        out.push(m | 27);
        out.extend_from_slice(&val.to_be_bytes());
    }
}

/// Read the argument following a CBOR head byte, consuming the appropriate
/// number of bytes from `input` (the full buffer) and advancing `cursor`.
fn read_arg(info: u8, input: &[u8], cursor: &mut usize) -> Result<u64> {
    match info {
        0..=23 => Ok(info as u64),
        24 => {
            let b = take(input, cursor, 1)?;
            Ok(b[0] as u64)
        }
        25 => {
            let b = take(input, cursor, 2)?;
            Ok(u16::from_be_bytes([b[0], b[1]]) as u64)
        }
        26 => {
            let b = take(input, cursor, 4)?;
            Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64)
        }
        27 => {
            let b = take(input, cursor, 8)?;
            Ok(u64::from_be_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        _ => bail!("invalid CBOR argument info {info}"),
    }
}

/// Take `n` bytes starting at the absolute offset `*cursor` in `input`,
/// advancing `cursor`. Bounds-checked against `input.len()`.
fn take<'a>(input: &'a [u8], cursor: &mut usize, n: usize) -> Result<&'a [u8]> {
    let end = input.len();
    let cur = *cursor;
    let abs_end = cur
        .checked_add(n)
        .ok_or_else(|| anyhow!("CBOR length overflow"))?;
    if abs_end > end {
        bail!(
            "truncated DAG-CBOR: need {n} bytes at offset {cur}, have {}",
            end.saturating_sub(cur)
        );
    }
    let slice = &input[cur..abs_end];
    *cursor = abs_end;
    Ok(slice)
}

// ── canonical key ordering ──────────────────────────────────────────────────

/// Canonical comparison for a map key: returns the "canonical key" byte
/// representation used both for sorting at construction and for verifying order
/// at decode. For text keys this is the UTF-8 bytes; for byte-string keys, the
/// raw bytes. (Mixed-type keys are not supported in DAG-CBOR.)
fn canonical_key_of(key: &DagCbor) -> Vec<u8> {
    match key {
        DagCbor::Text(s) => s.as_bytes().to_vec(),
        DagCbor::Bytes(b) => b.clone(),
        _ => Vec::new(), // non-string keys are rejected at a higher layer if needed
    }
}

/// Compare two canonical keys: **pure lexicographic byte order** (RFC 7049
/// §4.2.1 "core determinism"), which is what atproto's DAG-CBOR (`@atproto/lex-cbor`
/// via `cborg`) uses. (Not length-first-then-lex — that's the older RFC 7049 §3.9
/// "canonical CBOR" convention, which atproto does NOT use.)
fn canonical_key_cmp(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    a.cmp(b)
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
    fn encode_deterministic_text_map() {
        // Order of insertion must NOT affect the encoded bytes.
        let a = DagCbor::str_map([("b", DagCbor::Unsigned(2)), ("a", DagCbor::Unsigned(1))]);
        let b = DagCbor::str_map([("a", DagCbor::Unsigned(1)), ("b", DagCbor::Unsigned(2))]);
        assert_eq!(a.encode(), b.encode(), "str_map must canonicalize order");
    }

    #[test]
    fn round_trip_primitives() {
        for v in [
            DagCbor::Null,
            DagCbor::Bool(true),
            DagCbor::Bool(false),
            DagCbor::Unsigned(0),
            DagCbor::Unsigned(20),
            DagCbor::Unsigned(21),
            DagCbor::Unsigned(22),
            DagCbor::Unsigned(23),
            DagCbor::Unsigned(24),
            DagCbor::Unsigned(0x1_0000),
            DagCbor::Unsigned(u64::MAX),
            DagCbor::Negative(-1),
            DagCbor::Negative(-100),
            DagCbor::Bytes(vec![1, 2, 3]),
            DagCbor::Text("hello".into()),
            DagCbor::list([DagCbor::Unsigned(1), DagCbor::Text("x".into())]),
        ] {
            let enc = v.encode();
            let dec = DagCbor::decode(&enc).expect("round-trip");
            assert_eq!(v, dec, "round-trip {:?}", v);
        }
    }

    #[test]
    fn round_trip_map_with_link() {
        let cid = Cid::from_dag_cbor(b"block");
        let v = DagCbor::str_map([
            ("data", DagCbor::Link(cid)),
            ("did", DagCbor::Text("did:web:example.com".into())),
        ]);
        let enc = v.encode();
        let dec = DagCbor::decode(&enc).expect("round-trip");
        assert_eq!(v, dec);
    }

    #[test]
    fn decode_huge_array_header_errors_not_panics() {
        // 0x9b = array with 8-byte length arg; length claims 2^64-1 elements
        // but the input is only 9 bytes. Must return Err — not panic with a
        // capacity overflow or attempt a huge allocation.
        let input = [0x9b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        assert!(DagCbor::decode(&input).is_err());
    }

    #[test]
    fn decode_huge_map_header_errors_not_panics() {
        // 0xbb = map with 8-byte length arg claiming 2^64-1 entries.
        let input = [0xbb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        assert!(DagCbor::decode(&input).is_err());
    }

    #[test]
    fn decode_rejects_unsorted_keys() {
        // Hand-build a map with keys in WRONG order; decode must reject it.
        let bad = {
            let mut out = Vec::new();
            write_head(&mut out, MT_MAP, 2);
            // emit "b" then "a" — wrong canonical order.
            write_uint(&mut out, MT_TEXT, 1);
            out.push(b'b');
            write_uint(&mut out, MT_UNSIGNED, 1);
            write_uint(&mut out, MT_TEXT, 1);
            out.push(b'a');
            write_uint(&mut out, MT_UNSIGNED, 0);
            out
        };
        assert!(
            DagCbor::decode(&bad).is_err(),
            "unsorted map keys must fail"
        );
    }

    #[test]
    fn encode_minimal_widths() {
        // Determinism: minimal-width integer heads.
        assert_eq!(DagCbor::Unsigned(0).encode(), vec![0x00]);
        assert_eq!(DagCbor::Bool(false).encode(), vec![0xf4]);
        assert_eq!(DagCbor::Bool(true).encode(), vec![0xf5]);
        assert_eq!(DagCbor::Null.encode(), vec![0xf6]);
        assert_eq!(DagCbor::Unsigned(20).encode(), vec![20]);
        assert_eq!(DagCbor::Unsigned(23).encode(), vec![23]);
        assert_eq!(DagCbor::Unsigned(24).encode(), vec![24, 24]);
        assert_eq!(
            DagCbor::Unsigned(256).encode(),
            vec![0x19, 0x01, 0x00],
            "u16 must use head 25"
        );
    }
}
