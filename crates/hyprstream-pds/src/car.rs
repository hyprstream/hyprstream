//! CAR (Content Addressable aRchive) proof production and verification.
//!
//! A CAR file bundles a set of `(CID → block bytes)` pairs with a set of root
//! CIDs, so a recipient can store and re-verify the blocks offline. atproto's
//! `getRepo` / `getRecord` endpoints return CAR files; this module produces
//! minimal CAR **proofs** — just enough blocks to verify one record against the
//! signed commit.
//!
//! # CARv1 wire format (IPLD)
//!
//! ```text
//! CARv1 := header-section block-section*
//!   header-section := varint(len) ++ CBOR({ "version": 1, "roots": [CID,...] })
//!   block-section  := varint(len) ++ CID ++ raw-block-bytes
//! ```
//! `len` is a unsigned-varint byte count of the *rest* of the section.
//!
//! # `verify_record_proof`
//!
//! [`verify_record_proof`] is the D5 untrusted-host check: given a (purported)
//! signed commit, the account's `#atproto` P-256 verifying key, an MST inclusion
//! path, and the record bytes, confirm (1) the commit signature, (2) the MST
//! path leads from the commit's `data` root to the record's entry, and
//! (3) the record's CID matches the entry's value.

use anyhow::{anyhow, bail, ensure, Result};
use p256::ecdsa::VerifyingKey;

use crate::cid::{read_uvarint, write_uvarint, Cid};
use crate::commit::Commit;
use crate::dag_cbor::DagCbor;
use crate::mst::{NodeData, Proof};
use crate::record::ModelRecord;

/// Build a CARv1 byte blob containing the given root CIDs and `(cid → bytes)` blocks.
///
/// Like [`build_car_v1`] but returns each length-framed section as its own
/// `Vec<u8>` — the header section first, then one entry per block — so the
/// caller can stream them incrementally over HTTP without materializing the
/// entire CAR into a single contiguous buffer.
///
/// Each returned `Vec<u8>` is a complete framed CAR section
/// (`varint(len) ++ body`), safe to flush independently.
pub fn build_car_v1_sections(roots: &[Cid], blocks: &[(Cid, Vec<u8>)]) -> Vec<Vec<u8>> {
    let mut sections = Vec::with_capacity(1 + blocks.len());
    let header_value = DagCbor::str_map([
        ("version", DagCbor::Unsigned(1)),
        (
            "roots",
            DagCbor::List(roots.iter().copied().map(DagCbor::Link).collect()),
        ),
    ]);
    let header_bytes = header_value.encode();
    sections.push(frame_section(&header_bytes));
    for (cid, bytes) in blocks {
        let mut section = Vec::with_capacity(cid.as_bytes().len() + bytes.len());
        section.extend_from_slice(cid.as_bytes());
        section.extend_from_slice(bytes);
        sections.push(frame_section(&section));
    }
    sections
}

fn frame_section(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(10 + body.len());
    write_uvarint(body.len() as u64, &mut out);
    out.extend_from_slice(body);
    out
}

/// `blocks` need not be sorted; duplicates are harmless. The header's `roots`
/// is taken verbatim from `roots`.
pub fn build_car_v1(roots: &[Cid], blocks: &[(Cid, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    // Header section: CBOR { "version": 1, "roots": [Link, ...] }
    let header_value = DagCbor::str_map([
        ("version", DagCbor::Unsigned(1)),
        (
            "roots",
            DagCbor::List(roots.iter().copied().map(DagCbor::Link).collect()),
        ),
    ]);
    let header_bytes = header_value.encode();
    write_section(&mut out, &header_bytes);
    // Block sections: varint(len) ++ CID ++ bytes
    for (cid, bytes) in blocks {
        let mut section = Vec::with_capacity(cid.as_bytes().len() + bytes.len());
        section.extend_from_slice(cid.as_bytes());
        section.extend_from_slice(bytes);
        write_section(&mut out, &section);
    }
    out
}

fn write_section(out: &mut Vec<u8>, body: &[u8]) {
    write_uvarint(body.len() as u64, out);
    out.extend_from_slice(body);
}

/// Parse a CARv1 blob into `(roots, blocks)`. Used by tests / offline verifiers
/// that read a fetched CAR.
pub fn parse_car_v1(input: &[u8]) -> Result<(Vec<Cid>, Vec<(Cid, Vec<u8>)>)> {
    let mut cursor = 0usize;
    // Header section.
    let (header_body, after_header) = read_section(input, cursor)?;
    cursor = after_header;
    let header_val = DagCbor::decode(header_body)?;
    let version = header_val
        .get("version")
        .ok_or_else(|| anyhow!("CAR header missing 'version'"))?
        .as_unsigned()?;
    ensure!(
        version == 1,
        "only CARv1 is supported (got version {version})"
    );
    let roots_val = header_val
        .get("roots")
        .ok_or_else(|| anyhow!("CAR header missing 'roots'"))?
        .as_list()?;
    let mut roots = Vec::with_capacity(roots_val.len());
    for r in roots_val {
        roots.push(*r.as_link()?);
    }
    // Block sections.
    let mut blocks = Vec::new();
    while cursor < input.len() {
        let (body, after) = read_section(input, cursor)?;
        cursor = after;
        // body = CID ++ raw-bytes. Parse the CID prefix.
        let (cid, consumed) = parse_cid_prefix(body)?;
        let raw = body[consumed..].to_vec();
        blocks.push((cid, raw));
    }
    Ok((roots, blocks))
}

/// Read one length-prefixed CAR section. Returns `(body, new_cursor)`.
fn read_section(input: &[u8], cursor: usize) -> Result<(&[u8], usize)> {
    let (len, rest) =
        read_uvarint(&input[cursor..]).ok_or_else(|| anyhow!("truncated CAR varint"))?;
    let prefix_len = input.len() - cursor - rest.len();
    let body_start = cursor + prefix_len;
    let body_end = body_start
        .checked_add(len as usize)
        .ok_or_else(|| anyhow!("CAR section length overflow"))?;
    ensure!(body_end <= input.len(), "truncated CAR section body");
    Ok((&input[body_start..body_end], body_end))
}

/// Parse a CID at the start of `body`, returning `(Cid, bytes_consumed)`.
fn parse_cid_prefix(body: &[u8]) -> Result<(Cid, usize)> {
    // CIDv1: 0x01 ++ codec-varint ++ multihash. Multihash = code-varint ++ len-varint ++ bytes.
    // CIDv0: 0x12 ++ 0x20 ++ 32 bytes (base58 in string form; here a raw prefix).
    if body.is_empty() {
        bail!("empty CID prefix");
    }
    if body[0] == 0x01 {
        // CIDv1 — walk the varints to find the total length.
        let mut i = 1usize;
        let (_codec, rest) = read_uvarint(&body[i..]).ok_or_else(|| anyhow!("truncated codec"))?;
        i += (body.len() - i) - rest.len();
        let (_code, rest) = read_uvarint(&body[i..]).ok_or_else(|| anyhow!("truncated mh code"))?;
        i += (body.len() - i) - rest.len();
        let (len, rest) = read_uvarint(&body[i..]).ok_or_else(|| anyhow!("truncated mh len"))?;
        i += (body.len() - i) - rest.len();
        i += len as usize;
        let cid = Cid::from_bytes(&body[..i])?;
        Ok((cid, i))
    } else if body[0] == 0x12 {
        // CIDv0 (sha2-256, 32 bytes): 34 bytes total.
        let cid = Cid::from_bytes(&body[..34])?;
        Ok((cid, 34))
    } else {
        bail!("unsupported CID prefix byte 0x{:02x}", body[0]);
    }
}

/// Build a CAR proof for a single record: the commit block + all MST nodes on
/// the path + the record block, rooted at the commit's CID.
///
/// `record_bytes` is the DAG-CBOR of the [`ModelRecord`] (i.e. `record.to_dag_cbor()`).
pub fn build_record_proof_car(
    commit: &Commit,
    path: &Proof,
    node_blocks: &[(Cid, NodeData)],
    record: &ModelRecord,
) -> Vec<u8> {
    let mut blocks: Vec<(Cid, Vec<u8>)> = Vec::new();
    // Commit block first.
    let commit_cid = commit.cid();
    let commit_bytes = commit.to_dag_cbor();
    blocks.push((commit_cid, commit_bytes));
    // All MST node blocks referenced by the proof (caller passes the full tree's
    // blocks; we filter to those on the path for a minimal proof, but including
    // the whole tree is also valid — verifiers just won't use the extras).
    let path_cids: std::collections::BTreeSet<Cid> = path
        .path
        .iter()
        .map(|step| match step {
            crate::mst::ProofStep::FoundAt(d, _)
            | crate::mst::ProofStep::ThroughEntry(d, _)
            | crate::mst::ProofStep::LeftSubtree(d) => d.cid(),
        })
        .collect();
    for (cid, data) in node_blocks {
        if path_cids.contains(cid) {
            blocks.push((*cid, data.encode()));
        }
    }
    // Record block.
    let record_cid = record.cid();
    let record_bytes = record.to_dag_cbor();
    blocks.push((record_cid, record_bytes));
    build_car_v1(&[commit_cid], &blocks)
}

/// `verifyRecordProof(commit, signingKey, path, record)` — offline D5 verification.
///
/// Checks, in order:
/// 1. **Commit signature**: `commit` is signed by `signing_key` (the account's
///    published `#atproto` P-256 verifying key).
/// 2. **MST path**: `path` is a valid inclusion proof from the commit's `data`
///    (MST root) down to an entry whose value is the record's CID.
/// 3. **Record CID**: the `record`'s computed CID equals the entry value the
///    path addresses.
///
/// If all three pass, the host's claim ("this account's repo contains this
/// record at this MST root, signed by this DID") is trustworthy without needing
/// to trust the host.
pub fn verify_record_proof(
    commit: &Commit,
    signing_key: &VerifyingKey,
    path: &Proof,
    record: &ModelRecord,
) -> Result<()> {
    // (1) Commit signature.
    commit
        .verify(signing_key)
        .map_err(|e| anyhow!("commit signature: {e}"))?;
    // (2) + (3) MST path + record CID.
    let record_cid = record.cid();
    path.verify(&commit.data, &record_cid)
        .map_err(|e| anyhow!("MST proof: {e}"))?;
    Ok(())
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
    use crate::commit::{Commit, UnsignedCommit};
    use crate::mst::Node;
    use crate::record::{self, ModelRecord};
    use crate::tid::Tid;
    use p256::ecdsa::{SigningKey, VerifyingKey};
    use std::collections::BTreeMap;

    fn fixture() -> (
        VerifyingKey,
        Commit,
        Node,
        Tid,
        ModelRecord,
        Vec<(Cid, NodeData)>,
    ) {
        let signing_key = SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = VerifyingKey::from(&signing_key);

        let mut recs = BTreeMap::new();
        let mut records_by_tid: std::collections::BTreeMap<Tid, ModelRecord> = BTreeMap::new();
        for i in 0..6 {
            let tid = Tid::from_micros(1_700_000_000_000_000 + i * 1000, i as u16);
            let rec = ModelRecord::new(
                "at://did:web:alice.example.com",
                format!("bafyreiexampleoid{i:020}"),
                "2026-06-23T12:34:56.789Z",
            )
            .expect("record");
            recs.insert(tid, rec.cid());
            records_by_tid.insert(tid, rec);
        }
        let tree = Node::from_records(record::COLLECTION_NSID, &recs);
        let root = tree.root_cid();
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();

        let unsigned = UnsignedCommit::new("did:web:alice.example.com", root, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing_key);

        let target_tid = recs.keys().nth(2).copied().expect("key");
        let target_record = records_by_tid.get(&target_tid).cloned().expect("record");
        (
            verifying_key,
            commit,
            tree,
            target_tid,
            target_record,
            node_blocks,
        )
    }

    #[test]
    fn verify_record_proof_round_trip() {
        let (vk, commit, tree, tid, record, _node_blocks) = fixture();
        let proof = tree.proof(record::COLLECTION_NSID, &tid).expect("proof");
        verify_record_proof(&commit, &vk, &proof, &record).expect("proof verifies");
    }

    #[test]
    fn verify_record_proof_car_round_trip() {
        let (vk, commit, tree, tid, record, node_blocks) = fixture();
        let proof = tree.proof(record::COLLECTION_NSID, &tid).expect("proof");
        let car = build_record_proof_car(&commit, &proof, &node_blocks, &record);
        // The CAR must parse back, contain the commit + record blocks.
        let (roots, blocks) = parse_car_v1(&car).expect("parse CAR");
        assert_eq!(roots, vec![commit.cid()]);
        let block_cids: std::collections::BTreeSet<Cid> = blocks.iter().map(|(c, _)| *c).collect();
        assert!(
            block_cids.contains(&commit.cid()),
            "CAR must contain commit block"
        );
        assert!(
            block_cids.contains(&record.cid()),
            "CAR must contain record block"
        );

        // And the proof still verifies independently.
        verify_record_proof(&commit, &vk, &proof, &record).expect("proof verifies");
    }

    #[test]
    fn verify_record_proof_detects_wrong_record() {
        let (vk, commit, tree, tid, _record, _node_blocks) = fixture();
        let proof = tree.proof(record::COLLECTION_NSID, &tid).expect("proof");
        // Substitute a different record (wrong CID for the path's terminal entry).
        let wrong = ModelRecord::new(
            "at://did:web:alice.example.com",
            "bafyreiexampleoidDIFFERENT000",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("record");
        assert!(
            verify_record_proof(&commit, &vk, &proof, &wrong).is_err(),
            "must reject a record whose CID isn't addressed by the path"
        );
    }

    #[test]
    fn verify_record_proof_detects_bad_signature() {
        let (_vk, commit, tree, tid, record, _node_blocks) = fixture();
        let proof = tree.proof(record::COLLECTION_NSID, &tid).expect("proof");
        // A different verifying key — commit won't verify.
        let other = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        assert!(
            verify_record_proof(&commit, &other, &proof, &record).is_err(),
            "must reject a commit signed by a different key"
        );
    }

    #[test]
    fn verify_record_proof_detects_tampered_commit_data() {
        let (vk, mut commit, tree, tid, record, _node_blocks) = fixture();
        let proof = tree.proof(record::COLLECTION_NSID, &tid).expect("proof");
        // Point the commit at a different MST root — signature verification will
        // fail because the sig no longer covers the tampered data.
        commit.data = Cid::from_dag_cbor(b"different root");
        assert!(
            verify_record_proof(&commit, &vk, &proof, &record).is_err(),
            "must reject a commit whose data was tampered with"
        );
    }

    #[test]
    fn car_build_parse_round_trip() {
        let cid_a = Cid::from_dag_cbor(b"a");
        let cid_b = Cid::from_dag_cbor(b"b");
        let blocks = vec![(cid_a, b"block-a".to_vec()), (cid_b, b"block-b".to_vec())];
        let car = build_car_v1(&[cid_a], &blocks);
        let (roots, parsed) = parse_car_v1(&car).expect("parse");
        assert_eq!(roots, vec![cid_a]);
        assert_eq!(parsed, blocks);
    }

    #[test]
    fn empty_car_round_trip() {
        // Zero blocks, one root — header only.
        let root = Cid::from_dag_cbor(b"root");
        let car = build_car_v1(&[root], &[]);
        let (roots, blocks) = parse_car_v1(&car).expect("parse");
        assert_eq!(roots, vec![root]);
        assert!(blocks.is_empty());
    }
}
