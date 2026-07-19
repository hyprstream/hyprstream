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

use anyhow::{anyhow, bail, ensure, Result};
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
/// encoder re-sorts canonically (**pure lexicographic byte order**, RFC 7049
/// §4.2.1 "core determinism" — not length-first) so the order here is only for
/// readability. `version` is always [`COMMIT_VERSION`] = 3.
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

    /// Classify this commit's account DID as an accepted repo authority for our
    /// PDS (`did:web`/`did:plc`/`did:at9p`), or `Err` if the method is not one we
    /// host a repo for.
    ///
    /// This is the method-level acceptance gate (#908); it is independent of and
    /// additional to the signature check in [`Commit::verify`]. A caller ingesting
    /// a commit should accept the authority *and* verify the signature.
    pub fn repo_authority(&self) -> Result<crate::repo_authority::RepoAuthority> {
        crate::repo_authority::accept_repo_authority(&self.did)
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
    ///
    /// **Key-rotation caveat (#918):** this single-key form assumes the DID
    /// document still advertises the exact key that signed the commit. The
    /// ES256 `#atproto` store rotates on the order of days, and
    /// `oauth::did_document` publishes only the *active* slot, so a historical
    /// commit signed by a now-rotated-out key fails here even though it was
    /// legitimately signed. Callers that must verify commits across a rotation
    /// boundary should use [`Commit::verify_against_keys`] with the full set
    /// of currently-published `#atproto` slots (active + drain).
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

    /// Rotation-tolerant commit verification — the trust-chain fix for #918.
    ///
    /// **Key-rotation posture (#918, atproto-spec-aligned):** the producer
    /// re-signs the repo head commit on `#atproto` rotation, so the verifier
    /// checks the current head against the SINGLE current `#atproto` key
    /// resolved from the DID document ([`atproto_verifying_key_from_did_document`]).
    /// There is no drain-slot / multi-slot window on the atproto-compat key.
    /// Historical-commit verification, if ever needed, comes from the did:plc
    /// audit log (`/log/audit`), not from the live document — deferred until a
    /// caller needs it (current-head verification is the norm).
    ///
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


/// Decode a `publicKeyMultibase` (multibase `z` + base58btc + multicodec
/// `p256-pub` `0x80 0x24` + 33-byte compressed SEC1) into a P-256 verifying
/// key, with the multicodec prefix checked.
///
/// `label` is the verification-method id suffix (`#atproto` / `#atproto-drain`)
/// used in error messages; the caller already matched it.
fn decode_p256_multibase_vm(vm: &serde_json::Value, label: &str) -> Result<VerifyingKey> {
    const P256_PUB_MULTICODEC: [u8; 2] = [0x80, 0x24];
    let mb = vm
        .get("publicKeyMultibase")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("{label} verification method has no publicKeyMultibase string"))?;
    ensure!(
        mb.starts_with('z'),
        "{label} publicKeyMultibase must be multibase 'z' (base58btc)"
    );
    let payload = bs58::decode(&mb[1..])
        .into_vec()
        .map_err(|e| anyhow!("{label} publicKeyMultibase is not valid base58btc: {e}"))?;
    ensure!(
        payload.len() == 2 + 33,
        "{label} publicKeyMultibase payload is {} bytes, expected 35 (multicodec + compressed SEC1)",
        payload.len()
    );
    ensure!(
        payload[0] == P256_PUB_MULTICODEC[0] && payload[1] == P256_PUB_MULTICODEC[1],
        "{label} publicKeyMultibase has wrong multicodec prefix (not p256-pub 0x1200)"
    );
    VerifyingKey::from_sec1_bytes(&payload[2..])
        .map_err(|e| anyhow!("{label} publicKeyMultibase SEC1 point invalid: {e}"))
}

/// Validate a `#atproto`-family verification method's authority: it MUST be
/// `type: "Multikey"` and `controller: expected_did`. The id match (full
/// `{did}#fragment`) is established by the caller; this checks the rest of the
/// method so a method with the right id but a foreign controller/type is
/// rejected (a document carrying an attacker VM with a spoofed id still fails
/// closed on controller/type).
fn validate_vm_authority(
    vm: &serde_json::Value,
    full_id: &str,
    expected_did: &str,
) -> Result<()> {
    let vm_type = vm
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("`{full_id}` method has no `type`"))?;
    ensure!(
        vm_type == "Multikey",
        "`{full_id}` method type is {vm_type:?}, expected \"Multikey\""
    );
    let controller = vm
        .get("controller")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("`{full_id}` method has no `controller`"))?;
    ensure!(
        controller == expected_did,
        "`{full_id}` method controller is {controller:?}, expected {expected_did:?}"
    );
    Ok(())
}

/// Resolve the SINGLE current `#atproto` P-256 verifying key for `expected_did`
/// from a resolved DID document — the atproto-spec-aligned verification entry
/// point (#918 re-sign-on-rotation).
///
/// atproto DID documents carry exactly one `#atproto` verification method
/// (no `#atproto-drain`, no `nbf`/`exp`). On a `#atproto` rotation the producer
/// re-signs the repo head commit with the new active key, so the verifier only
/// ever needs this one current key to check the current head
/// ([`Commit::verify`]). Historical-commit verification, if a caller ever needs
/// it, comes from the did:plc audit log (`/log/audit`), not the live document.
///
/// Authority hygiene (kept from the earlier review — good regardless of the
/// single-vs-multi question): the document's `id` MUST equal `expected_did`;
/// the `#atproto` method is matched by FULL id (`{did}#atproto`), must be
/// unique (duplicates rejected), and must be `type: "Multikey"` with
/// `controller: expected_did`. Any mismatch fails closed.
pub fn atproto_verifying_key_from_did_document(
    doc: &serde_json::Value,
    expected_did: &str,
) -> Result<VerifyingKey> {
    let doc_id = doc
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("DID document has no `id`"))?;
    ensure!(
        doc_id == expected_did,
        "DID document id {doc_id:?} does not match the DID being resolved {expected_did:?}"
    );
    let vms = doc
        .get("verificationMethod")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("DID document has no verificationMethod array"))?;
    let atproto_id = format!("{expected_did}#atproto");
    let mut atproto_vm: Option<&serde_json::Value> = None;
    for vm in vms {
        if vm.get("id").and_then(serde_json::Value::as_str) == Some(&atproto_id) {
            if atproto_vm.is_some() {
                bail!("DID document has more than one `{atproto_id}` method");
            }
            atproto_vm = Some(vm);
        }
    }
    let atproto = atproto_vm
        .ok_or_else(|| anyhow!("DID document has no `{atproto_id}` verification method"))?;
    validate_vm_authority(atproto, &atproto_id, expected_did)?;
    decode_p256_multibase_vm(atproto, "#atproto")
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

    // ── #918: single-key DID-document verification (re-sign-on-rotation) ──────

    /// Helper: encode a P-256 verifying key as the multibase string the DID
    /// document producer emits (mirrors `did_document::p256_to_multibase`).
    fn p256_to_multibase(vk: &VerifyingKey) -> String {
        use p256::EncodedPoint;
        let point: EncodedPoint = vk.to_encoded_point(true);
        let mut payload = vec![0x80, 0x24];
        payload.extend_from_slice(point.as_bytes());
        format!("z{}", bs58::encode(payload).into_string())
    }

    fn did_doc_with_atproto(did: &str, vk: &VerifyingKey) -> serde_json::Value {
        serde_json::json!({
            "id": did,
            "verificationMethod": [{
                "id": format!("{did}#atproto"),
                "type": "Multikey",
                "controller": did,
                "publicKeyMultibase": p256_to_multibase(vk),
            }],
        })
    }

    /// Resolve the single `#atproto` key from a document, then verify a commit
    /// signed by that key passes and a forged (different-key) commit fails.
    #[test]
    fn atproto_key_from_did_document_verifies_head() {
        let (commit, vk) = make_signed_commit();
        let did = "did:web:alice.example.com";
        let resolved = atproto_verifying_key_from_did_document(&did_doc_with_atproto(did, &vk), did)
            .expect("resolve #atproto key");
        assert_eq!(resolved.to_sec1_bytes(), vk.to_sec1_bytes());
        // Head signed by the resolved key verifies.
        commit.verify(&resolved).expect("head verifies under current #atproto");
        // A head signed by a DIFFERENT key does not verify against this doc's key.
        let (other_commit, _other_vk) = make_signed_commit();
        assert!(
            other_commit.verify(&resolved).is_err(),
            "a commit signed by a different key must fail single-key verification"
        );
    }

    /// The re-sign contract: a head still signed by the OLD key (not re-signed
    /// after rotation) does NOT verify against the current document (=new key).
    /// This is both directions of the #918 re-sign-on-rotation design.
    #[test]
    fn old_key_head_does_not_verify_against_rotated_doc() {
        let (old_commit, old_vk) = make_signed_commit();
        let new_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        let did = "did:web:alice.example.com";
        // Current document publishes the NEW key (post-rotation).
        let resolved = atproto_verifying_key_from_did_document(&did_doc_with_atproto(did, &new_vk), did)
            .expect("resolve");
        // The head (still signed by the old key) fails — the producer's re-sign
        // is what would make it pass; without re-sign, verification correctly fails.
        assert!(
            old_commit.verify(&resolved).is_err(),
            "a head not re-signed after rotation must fail against the rotated-in key"
        );
        let _ = old_vk;
    }

    /// Authority hygiene: a document whose `id` is for a different DID fails.
    #[test]
    fn atproto_key_rejects_wrong_did() {
        let (_c, vk) = make_signed_commit();
        let doc = did_doc_with_atproto("did:web:alice.example.com", &vk);
        assert!(
            atproto_verifying_key_from_did_document(&doc, "did:web:bob.example.com").is_err(),
            "a document for a different DID must fail closed"
        );
    }

    /// Authority hygiene: a wrong controller / type is rejected.
    #[test]
    fn atproto_key_rejects_wrong_controller() {
        let (_c, vk) = make_signed_commit();
        let did = "did:web:alice.example.com";
        let doc = serde_json::json!({
            "id": did,
            "verificationMethod": [{
                "id": format!("{did}#atproto"),
                "type": "Multikey",
                "controller": "did:web:mallory.example.com",
                "publicKeyMultibase": p256_to_multibase(&vk),
            }],
        });
        assert!(
            atproto_verifying_key_from_did_document(&doc, did).is_err(),
            "a #atproto method with a foreign controller must fail closed"
        );
    }

    /// Authority hygiene: duplicate `#atproto` methods are rejected.
    #[test]
    fn atproto_key_rejects_duplicate_method() {
        let (_c, vk) = make_signed_commit();
        let did = "did:web:alice.example.com";
        let vm = serde_json::json!({
            "id": format!("{did}#atproto"), "type": "Multikey",
            "controller": did, "publicKeyMultibase": p256_to_multibase(&vk),
        });
        let doc = serde_json::json!({ "id": did, "verificationMethod": [vm.clone(), vm] });
        let err = atproto_verifying_key_from_did_document(&doc, did)
            .expect_err("duplicate #atproto must fail closed");
        assert!(
            format!("{err}").contains("more than one"),
            "duplicate-method error should be explicit, got: {err}"
        );
    }
}
