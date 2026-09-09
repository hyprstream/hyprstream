//! Explicit public AT Protocol CBOR boundary over the existing value type.
//!
//! Public DAG-CBOR orders text map keys by encoded bytes: UTF-8 byte length
//! first, then lexical bytes. The existing native codec has a different order
//! and is retained for already-signed artifacts. Neither decoder falls back to
//! the other's order. This module does not change existing record, commit, MST
//! or identity call sites, and does not establish public repository readiness.
//!
//! Integer-only AT data uses signed 64-bit integers. The public encoder checks
//! this and validates map structure/depth before invoking the existing scalar
//! writer. The decoder retains minimal-width, tag, UTF-8, duplicate-key and
//! bounded-depth checks. Floats remain unsupported by the AT data model.

use std::cmp::Ordering;

use anyhow::{ensure, Result};

use crate::dag_cbor::DagCbor;

const MAX_DEPTH: usize = 128;

fn key_order(a: &[u8], b: &[u8]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// Encode a value in the public AT Protocol format without mutating it.
/// This is canonical serialization, not schema validation or authorization.
pub fn encode(value: &DagCbor) -> Result<Vec<u8>> {
    Ok(normalize(value, 0)?.encode())
}

/// Decode exactly one strictly canonical public AT Protocol value.
/// No legacy/native ordering fallback is permitted.
pub fn decode(bytes: &[u8]) -> Result<DagCbor> {
    let value = DagCbor::decode_with_order(bytes, key_order)?;
    // Enforce signed integer bounds and the same construction constraints as
    // encode; byte-level canonicality has already been checked by the parser.
    normalize(&value, 0)
}

fn normalize(value: &DagCbor, depth: usize) -> Result<DagCbor> {
    ensure!(
        depth <= MAX_DEPTH,
        "ATProto CBOR nesting exceeds {MAX_DEPTH}"
    );
    Ok(match value {
        DagCbor::Unsigned(n) => {
            ensure!(
                *n <= i64::MAX as u64,
                "ATProto integer exceeds signed 64-bit range"
            );
            value.clone()
        }
        DagCbor::Negative(n) => {
            ensure!(
                *n >= i64::MIN as i128 && *n < 0,
                "ATProto negative integer outside signed 64-bit range"
            );
            value.clone()
        }
        DagCbor::List(items) => DagCbor::List(
            items
                .iter()
                .map(|item| normalize(item, depth + 1))
                .collect::<Result<_>>()?,
        ),
        DagCbor::Map(pairs) => {
            let mut normalized = Vec::with_capacity(pairs.len());
            for (key, val) in pairs {
                ensure!(depth < MAX_DEPTH, "ATProto CBOR map exceeds nesting limit");
                let text = key.as_str()?;
                normalized.push((text.to_owned(), normalize(val, depth + 1)?));
            }
            normalized.sort_by(|(a, _), (b, _)| key_order(a.as_bytes(), b.as_bytes()));
            ensure!(
                normalized.windows(2).all(|pair| pair[0].0 != pair[1].0),
                "duplicate ATProto CBOR map key"
            );
            DagCbor::Map(
                normalized
                    .into_iter()
                    .map(|(k, v)| (DagCbor::Text(k), v))
                    .collect(),
            )
        }
        _ => value.clone(),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::Cid;

    fn post() -> DagCbor {
        DagCbor::str_map([
            ("$type", DagCbor::Text("app.bsky.feed.post".into())),
            ("text", DagCbor::Text("Hello from Torment Nexus".into())),
            (
                "createdAt",
                DagCbor::Text("2026-09-09T00:00:00.000Z".into()),
            ),
        ])
    }

    #[test]
    fn public_post_matches_upstream_bytes_and_cid() {
        // Generated independently with @atproto/lex-cbor 0.1.6 encode/cidForLex.
        let upstream = hex::decode("a36474657874781848656c6c6f2066726f6d20546f726d656e74204e65787573652474797065726170702e62736b792e666565642e706f7374696372656174656441747818323032362d30392d30395430303a30303a30302e3030305a").unwrap();
        let bytes = encode(&post()).unwrap();
        assert_eq!(bytes, upstream);
        assert_eq!(
            Cid::from_dag_cbor(&bytes).to_string(),
            "bafyreiaapk47b4fmdslcizkj5aj6ws6bmpmvfvjq3ryhx5qwgug6tldalm"
        );
        assert_eq!(encode(&decode(&bytes).unwrap()).unwrap(), bytes);
    }

    #[test]
    fn native_bytes_and_strict_verifier_remain_separate() {
        let value = DagCbor::str_map([("aa", DagCbor::Unsigned(1)), ("b", DagCbor::Unsigned(2))]);
        let native = hex::decode("a262616101616202").unwrap();
        let public = hex::decode("a261620262616101").unwrap();
        assert_eq!(value.encode(), native);
        assert_eq!(DagCbor::decode(&native).unwrap(), value);
        assert_eq!(encode(&value).unwrap(), public);
        assert!(DagCbor::decode(&public).is_err());
        assert!(decode(&native).is_err());
    }

    #[test]
    fn public_order_applies_to_nested_maps_and_utf8_byte_lengths() {
        let inner = DagCbor::str_map([("é", DagCbor::Bool(true)), ("z", DagCbor::Bool(false))]);
        let value = DagCbor::list([inner]);
        assert_eq!(hex::encode(encode(&value).unwrap()), "81a2617af462c3a9f5");
        assert_eq!(
            encode(&decode(&encode(&value).unwrap()).unwrap()).unwrap(),
            encode(&value).unwrap()
        );
    }

    #[test]
    fn malformed_construction_is_rejected_without_panic() {
        for value in [
            DagCbor::Negative(0),
            DagCbor::Negative(i128::MIN),
            DagCbor::Unsigned(u64::MAX),
            DagCbor::Map(vec![(DagCbor::Bool(true), DagCbor::Null)]),
            DagCbor::Map(vec![(DagCbor::Text("a".into()), DagCbor::Null); 2]),
        ] {
            assert!(encode(&value).is_err());
        }
    }

    #[test]
    fn decoder_retains_strictness_and_integer_bounds() {
        for wire in [
            "a2616101616102",     // duplicate keys
            "1817",               // nonminimal integer
            "00ff",               // trailing data
            "9fff",               // indefinite array
            "c000",               // unsupported tag
            "61ff",               // invalid UTF-8
            "1b8000000000000000", // outside signed integer range
            "3b8000000000000000", // below signed integer range
        ] {
            assert!(decode(&hex::decode(wire).unwrap()).is_err(), "{wire}");
        }
    }

    #[test]
    fn links_and_signed_integer_extremes_round_trip() {
        let value = DagCbor::list([
            DagCbor::Link(Cid::from_raw(b"public blob")),
            DagCbor::Unsigned(i64::MAX as u64),
            DagCbor::Negative(i64::MIN as i128),
        ]);
        assert_eq!(decode(&encode(&value).unwrap()).unwrap(), value);
    }

    #[test]
    fn excessive_nesting_is_rejected() {
        let mut value = DagCbor::Null;
        for _ in 0..130 {
            value = DagCbor::list([value]);
        }
        assert!(encode(&value).is_err());
        assert!(decode(&value.encode()).is_err());
    }
}
