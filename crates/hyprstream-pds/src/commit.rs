//! Signed atproto repository commits.
//!
//! A commit is the **signed mutable pointer**: it binds an MST root CID (the
//! immutable record-store snapshot) to a DID at a revision, signed by the DID's
//! `#atproto` P-256 key. This is what federation verifies to trust a host's
//! claim about an account's state — the consensus-free versioning primitive.
//!
//! # Structure (atproto v3)
//!
//! ```text
//! Commit {
//!   did:     String,        // account DID (did:web / did:plc)
//!   version: u64 = 3,       // protocol version
//!   data:    Cid (link),    // MST root
//!   rev:     String,        // TID revision
//!   prev:    Option<Cid>,   // previous commit (None for genesis)
//!   sig:     Vec<u8>,       // ES256 signature over the unsigned commit
//! }
//! ```
//!
//! # Signing
//!
//! `sig` is ES256 (P-256 ECDSA over SHA-256) of the DAG-CBOR encoding of the
//! **unsigned commit** — the commit object with the `sig` field *omitted* (not
//! present-and-empty). The verifier re-encodes the unsigned form and checks the
//! signature against the DID's published `#atproto` P-256 verifying key.

use anyhow::{ensure, Result};
use p256::ecdsa::{signature::Signer, Signature, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;
use crate::tid::Tid;

/// The commit version number (atproto v3).
pub const COMMIT_VERSION: u64 = 3;

/// An unsigned commit — the form that gets DAG-CBOR-encoded and signed.
///
/// Field order matches atproto (`did`, `version`, `data`, `rev`, `prev`); the
/// encoder re-sorts canonically (length-first then lex) so the order here is
/// only for readability. `version` is always [`COMMIT_VERSION`] = 3.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsignedCommit {
    pub did: String,
    pub version: u64,
    pub data: Cid,
    pub rev: Tid,
    pub prev: Option<Cid>,
}

impl UnsignedCommit {
    pub fn new(did: impl Into<String>, data: Cid, rev: Tid, prev: Option<Cid>) -> Self {
        UnsignedCommit {
            did: did.into(),
            version: COMMIT_VERSION,
            data,
            rev,
            prev,
        }
    }

    /// DAG-CBOR encode the unsigned commit (no `sig` field). This is what gets signed.
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("did", DagCbor::Text(self.did.clone())),
            ("version", DagCbor::Unsigned(self.version)),
            ("data", DagCbor::Link(self.data)),
            ("rev", DagCbor::Text(self.rev.encode())),
            (
                "prev",
                match &self.prev {
                    Some(c) => DagCbor::Link(*c),
                    None => DagCbor::Null,
                },
            ),
        ])
    }
}

/// A signed commit — the unsigned commit plus its ES256 `sig`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Commit {
    pub did: String,
    pub version: u64,
    pub data: Cid,
    pub rev: Tid,
    pub prev: Option<Cid>,
    pub sig: Vec<u8>,
}

impl Commit {
    /// Sign an unsigned commit with a P-256 (`#atproto`) signing key,
    /// producing a [`Commit`].
    ///
    /// The signature is ES256: ECDSA over SHA-256 of the unsigned commit's
    /// DAG-CBOR bytes. The signature is the fixed-width (64-byte) R‖V‖S
    /// encoding — `Signature::to_vec` in `p256` 0.13 gives this form, which is
    /// the JOSE/COSE/atproto-preferred raw concatenation (not DER).
    pub fn sign(unsigned: &UnsignedCommit, key: &SigningKey) -> Self {
        let unsigned_bytes = unsigned.to_dag_cbor();
        // ES256 = ECDSA with SHA-256. p256's `Signer<Vec<u8>>` impl hashes
        // internally and emits the raw 64-byte R‖V‖S form.
        let sig: Signature = key.sign(&unsigned_bytes);
        Commit {
            did: unsigned.did.clone(),
            version: unsigned.version,
            data: unsigned.data,
            rev: unsigned.rev,
            prev: unsigned.prev,
            sig: sig.to_vec(),
        }
    }

    /// DAG-CBOR encode the (signed) commit. The `sig` field is a byte string.
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("did", DagCbor::Text(self.did.clone())),
            ("version", DagCbor::Unsigned(self.version)),
            ("data", DagCbor::Link(self.data)),
            ("rev", DagCbor::Text(self.rev.encode())),
            (
                "prev",
                match &self.prev {
                    Some(c) => DagCbor::Link(*c),
                    None => DagCbor::Null,
                },
            ),
            ("sig", DagCbor::Bytes(self.sig.clone())),
        ])
    }

    /// The [`UnsignedCommit`] form of this commit (drops `sig`).
    pub fn unsigned(&self) -> UnsignedCommit {
        UnsignedCommit {
            did: self.did.clone(),
            version: self.version,
            data: self.data,
            rev: self.rev,
            prev: self.prev,
        }
    }

    /// Decode a signed commit from DAG-CBOR bytes, validating structure.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let did = value
            .get("did")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'did'"))?
            .as_str()?
            .to_owned();
        let version = value
            .get("version")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'version'"))?
            .as_unsigned()?;
        ensure!(
            version == COMMIT_VERSION,
            "unsupported commit version {version} (expected {COMMIT_VERSION})"
        );
        let data = *value
            .get("data")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'data'"))?
            .as_link()?;
        let rev_str = value
            .get("rev")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'rev'"))?
            .as_str()?;
        let rev = Tid::parse(rev_str)?;
        let prev_val = value
            .get("prev")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'prev'"))?;
        let prev = if prev_val.is_null() {
            None
        } else {
            Some(*prev_val.as_link()?)
        };
        let sig = value
            .get("sig")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'sig'"))?
            .as_bytes()?
            .to_vec();
        Ok(Commit {
            did,
            version,
            data,
            rev,
            prev,
            sig,
        })
    }

    /// The CID of this (signed) commit block.
    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    /// Verify the commit's signature against a `#atproto` P-256 verifying key.
    ///
    /// Re-encodes the unsigned commit, hashes with SHA-256, and verifies the
    /// ES256 signature. Returns `Ok(())` on success.
    ///
    /// This is the core of the D5 untrusted-host posture: a host can serve any
    /// commit it likes, but only commits signed by the account's `#atproto`
    /// key are accepted by verifiers.
    pub fn verify(&self, vk: &VerifyingKey) -> Result<()> {
        use p256::ecdsa::signature::Verifier;
        let unsigned = self.unsigned();
        let unsigned_bytes = unsigned.to_dag_cbor();
        // ES256 verification: parse the signature, then verify over the bytes.
        // p256 0.13's `Signature::from_slice` accepts the 64-byte raw R‖V‖S form.
        let signature = Signature::from_slice(&self.sig)
            .map_err(|e| anyhow::anyhow!("invalid ES256 signature bytes: {e}"))?;
        vk.verify(&unsigned_bytes, &signature)
            .map_err(|e| anyhow::anyhow!("ES256 signature verification failed: {e}"))
    }

    /// Compute the SHA-256 digest of the unsigned commit's DAG-CBOR bytes.
    ///
    /// Exposed for callers that compose their own signing/verification flow
    /// (e.g. using the DID's key material from `hyprstream::auth::key_rotation`
    /// directly). Standard ES256 signers hash internally, so most callers want
    /// [`Commit::sign`] / [`Commit::verify`] instead.
    pub fn unsigned_digest(unsigned: &UnsignedCommit) -> [u8; 32] {
        Sha256::digest(unsigned.to_dag_cbor()).into()
    }
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
    use crate::record::{self, ModelRecord};
    use crate::tid::Tid;
    use std::collections::BTreeMap;

    fn make_signed_commit() -> (Commit, VerifyingKey) {
        // Build a small MST, sign a commit over its root.
        let mut recs = BTreeMap::new();
        for i in 0..3 {
            let tid = Tid::from_micros(1_700_000_000_000_000 + i, i as u16);
            let rec = ModelRecord::new(
                "at://did:web:alice.example.com",
                format!("bafyreiexampleoid{i:020}"),
                "2026-06-23T12:34:56.789Z",
            )
            .expect("record");
            recs.insert(tid, rec.cid());
        }
        let tree = crate::mst::Node::from_records(record::COLLECTION_NSID, &recs);
        let root = tree.root_cid();

        let signing_key = SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = VerifyingKey::from(&signing_key);

        let unsigned = UnsignedCommit::new("did:web:alice.example.com", root, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing_key);
        (commit, verifying_key)
    }

    #[test]
    fn commit_sign_and_verify_round_trip() {
        let (commit, vk) = make_signed_commit();
        commit.verify(&vk).expect("signature must verify");
    }

    #[test]
    fn commit_detects_wrong_key() {
        let (commit, _vk) = make_signed_commit();
        // A different key must fail.
        let other = SigningKey::random(&mut rand::rngs::OsRng);
        let other_vk = VerifyingKey::from(other);
        assert!(
            commit.verify(&other_vk).is_err(),
            "signature must not verify under a different key"
        );
    }

    #[test]
    fn commit_detects_tampered_data() {
        let (mut commit, vk) = make_signed_commit();
        // Tamper with the MST root pointer — signature must fail.
        commit.data = Cid::from_dag_cbor(b"tampered");
        assert!(
            commit.verify(&vk).is_err(),
            "tampered data must fail signature verify"
        );
    }

    #[test]
    fn commit_dag_cbor_round_trip() {
        let (commit, _vk) = make_signed_commit();
        let bytes = commit.to_dag_cbor();
        let back = Commit::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(commit, back);
    }

    #[test]
    fn commit_unsigned_has_no_sig_field() {
        // The unsigned encoding MUST NOT carry a sig field (atproto signs the
        // object without sig, not sig=empty).
        let unsigned = UnsignedCommit::new(
            "did:web:x",
            Cid::from_dag_cbor(b"r"),
            Tid::from_raw(1),
            None,
        );
        let v = unsigned.to_value();
        assert!(
            v.get("sig").is_none(),
            "unsigned commit must not have a sig field"
        );
    }
}
