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

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;
use crate::tid::Tid;

const MAX_DEPTH: usize = 128;

/// A generic public AT Protocol repository record.
///
/// The value is retained as a typed DAG-CBOR value and its exact public bytes
/// and CID are derived once at construction. This is deliberately independent
/// of native `ModelRecord`: any supported collection can be stored without
/// converting through a lossy JSON DTO. Construction validates the collection,
/// TID record key, `$type` discriminator and canonical public bytes; it does
/// not grant write authority or perform a full Lexicon validation.
#[derive(Clone, Debug, PartialEq)]
pub struct AtprotoRecord {
    pub collection: String,
    pub rkey: Tid,
    pub value: DagCbor,
    bytes: Vec<u8>,
    cid: Cid,
}

impl AtprotoRecord {
    pub fn new(collection: impl Into<String>, rkey: Tid, value: DagCbor) -> Result<Self> {
        let collection = collection.into();
        validate_nsid(&collection)?;
        let type_value = value
            .get("$type")
            .ok_or_else(|| anyhow::anyhow!("AT record is missing $type"))?
            .as_str()?;
        ensure!(
            type_value == collection,
            "AT record $type does not match collection"
        );
        let bytes = encode(&value)?;
        let cid = Cid::from_dag_cbor(&bytes);
        Ok(Self {
            collection,
            rkey,
            value,
            bytes,
            cid,
        })
    }

    /// Reconstruct a record only when bytes are already strict public
    /// canonical encoding. This check prevents a caller from changing the CID
    /// by normalizing at a later boundary.
    pub fn from_bytes(collection: impl Into<String>, rkey: Tid, bytes: &[u8]) -> Result<Self> {
        let value = decode(bytes)?;
        let record = Self::new(collection, rkey, value)?;
        ensure!(record.bytes == bytes, "AT record bytes are not canonical");
        Ok(record)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn cid(&self) -> Cid {
        self.cid
    }
    pub fn uri(&self, did: &str) -> String {
        format!("at://{did}/{}/{}", self.collection, self.rkey.encode())
    }
}

fn validate_nsid(nsid: &str) -> Result<()> {
    ensure!(
        !nsid.is_empty() && nsid.len() <= 317,
        "invalid AT collection NSID"
    );
    let mut segments = nsid.split('.');
    let first = segments.next().unwrap_or_default();
    ensure!(
        !first.is_empty()
            && first
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
        "invalid NSID authority"
    );
    for segment in segments {
        ensure!(
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "invalid NSID segment"
        );
    }
    ensure!(
        nsid.contains('.'),
        "AT collection NSID needs a domain hierarchy"
    );
    Ok(())
}

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
    use crate::car::{build_public_record_proof_car, parse_car_v1_atproto};
    use crate::commit::{Commit, UnsignedCommit};
    use crate::mst::Node;
    use crate::tid::Tid;
    use crate::Cid;
    use p256::ecdsa::SigningKey;

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
    fn public_record_mst_commit_and_car_proof_round_trip() {
        let rkey = Tid::from_raw(7);
        let record = AtprotoRecord::new("app.bsky.feed.post", rkey, post()).unwrap();
        let tree = Node::from_keyed_records(
            &[(
                format!("app.bsky.feed.post/{}", rkey.encode()),
                record.cid(),
            )]
            .into_iter()
            .collect(),
        );
        let (root_data, node_blocks) = tree.to_node_data_with_blocks_atproto().unwrap();
        let root = root_data.cid_atproto().unwrap();
        let unsigned =
            UnsignedCommit::new("did:web:tormentnexus.social", root, Tid::from_raw(8), None);
        let signing = SigningKey::from_bytes((&[7u8; 32]).into()).unwrap();
        let commit = Commit::sign_atproto(&unsigned, &signing).unwrap();
        commit.verify_atproto(signing.verifying_key()).unwrap();
        let proof = tree.proof_atproto("app.bsky.feed.post", &rkey).unwrap();
        proof.verify_atproto(&root, &record.cid()).unwrap();
        let car = build_public_record_proof_car(&commit, &proof, &node_blocks, &record).unwrap();
        let (roots, blocks) = parse_car_v1_atproto(&car).unwrap();
        assert_eq!(roots, vec![commit.cid_atproto().unwrap()]);
        assert!(blocks
            .iter()
            .any(|(cid, bytes)| *cid == record.cid() && bytes == record.bytes()));
        assert!(blocks
            .iter()
            .any(|(cid, bytes)| *cid == commit.cid_atproto().unwrap()
                && bytes == &commit.to_atproto_dag_cbor().unwrap()));
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
