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
    /// For a rotated key set, use [`Commit::verify_against_published_keys`].
    /// That path accepts only the active key and bounded, currently-live
    /// `#atproto_drain` / `#atproto_lead` keys published in the DID document.
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

    /// Verify against the bounded `#atproto` slot set currently published by a
    /// DID document.  This is an overlap-availability mechanism: it accepts a
    /// pre-rotation commit under the bounded drain key, but does not establish
    /// that a key was valid at the commit's causal position (that is #1169).
    pub fn verify_against_published_keys(
        &self,
        keys: &PublishedAtprotoKeys,
        now: i64,
    ) -> Result<()> {
        let mut tried = 0usize;
        let mut last_error = None;
        for key in keys.live_keys(now) {
            tried += 1;
            match self.verify(key) {
                Ok(()) => return Ok(()),
                Err(error) => last_error = Some(error),
            }
        }
        match last_error {
            Some(error) => Err(error.context(format!(
                "ES256 signature verified under none of the {tried} live published #atproto slot(s)"
            ))),
            None => bail!("no published #atproto verification slot is live at {now}"),
        }
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

/// The only auxiliary verification-method fragments understood by Hyprstream.
/// Stock atproto implementations select the exact `#atproto` fragment, so the
/// overlap keys deliberately use distinct fragments.
const DRAIN_FRAGMENT: &str = "atproto_drain";
const LEAD_FRAGMENT: &str = "atproto_lead";

/// A P-256 key published for commit verification, with a half-open validity
/// interval. The active slot is unbounded; drain and lead slots must carry
/// explicit document bounds.
#[derive(Clone, Debug)]
pub struct PublishedAtprotoKey {
    key: VerifyingKey,
    nbf: i64,
    exp: i64,
}

impl PublishedAtprotoKey {
    fn active(key: VerifyingKey) -> Self {
        Self {
            key,
            nbf: i64::MIN,
            exp: i64::MAX,
        }
    }

    fn bounded(key: VerifyingKey, nbf: i64, exp: i64) -> Result<Self> {
        ensure!(
            nbf < exp,
            "published #atproto slot has an empty validity interval"
        );
        Ok(Self { key, nbf, exp })
    }

    fn is_live_at(&self, now: i64) -> bool {
        self.nbf <= now && now < self.exp
    }
}

/// A small, authority-checked set of verification keys from a DID document:
/// exactly one active `#atproto` slot and at most one bounded drain and lead
/// slot. Unknown DID verification methods are intentionally ignored.
#[derive(Clone, Debug)]
pub struct PublishedAtprotoKeys {
    slots: Vec<PublishedAtprotoKey>,
}

impl PublishedAtprotoKeys {
    pub const MAX_SLOTS: usize = 3;

    /// Wrap a trusted single current key for existing resolver implementations.
    pub fn single(key: VerifyingKey) -> Self {
        Self {
            slots: vec![PublishedAtprotoKey::active(key)],
        }
    }

    /// Construct the same bounded authority that a root DID document publishes.
    /// This is for local resolvers whose trusted publication source is the
    /// rotation store that feeds that document; remote resolvers must parse the
    /// resolved DID document with [`Self::from_did_document`].
    pub fn from_published_slots(
        active: VerifyingKey,
        drain: Option<(VerifyingKey, i64, i64)>,
        lead: Option<(VerifyingKey, i64, i64)>,
    ) -> Result<Self> {
        let mut slots = vec![PublishedAtprotoKey::active(active)];
        if let Some((key, nbf, exp)) = drain {
            slots.push(PublishedAtprotoKey::bounded(key, nbf, exp)?);
        }
        if let Some((key, nbf, exp)) = lead {
            slots.push(PublishedAtprotoKey::bounded(key, nbf, exp)?);
        }
        Ok(Self { slots })
    }

    pub fn live_keys(&self, now: i64) -> impl Iterator<Item = &VerifyingKey> {
        self.slots
            .iter()
            .filter(move |slot| slot.is_live_at(now))
            .map(|slot| &slot.key)
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Parse the active and bounded overlap slots from a resolved DID document.
    /// Every recognised slot must have the expected full id, controller and
    /// `Multikey` type; duplicates and malformed bounds fail closed.
    pub fn from_did_document(doc: &serde_json::Value, expected_did: &str) -> Result<Self> {
        let doc_id = doc
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow!("DID document has no `id`"))?;
        ensure!(
            doc_id == expected_did,
            "DID document id {doc_id:?} does not match {expected_did:?}"
        );
        let vms = doc
            .get("verificationMethod")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow!("DID document has no verificationMethod array"))?;

        let mut active = None;
        let mut drain = None;
        let mut lead = None;
        for (fragment, destination, bounded) in [
            ("atproto", &mut active, false),
            (DRAIN_FRAGMENT, &mut drain, true),
            (LEAD_FRAGMENT, &mut lead, true),
        ] {
            let full_id = format!("{expected_did}#{fragment}");
            for vm in vms.iter().filter(|vm| {
                vm.get("id").and_then(serde_json::Value::as_str) == Some(full_id.as_str())
            }) {
                ensure!(
                    destination.is_none(),
                    "DID document has more than one `{full_id}` method"
                );
                validate_vm_authority(vm, &full_id, expected_did)?;
                let key = decode_p256_multibase_vm(vm, &format!("#{fragment}"))?;
                *destination = Some(if bounded {
                    let nbf = vm
                        .get("nbf")
                        .and_then(serde_json::Value::as_i64)
                        .ok_or_else(|| anyhow!("`{full_id}` must carry integer `nbf`"))?;
                    let exp = vm
                        .get("exp")
                        .and_then(serde_json::Value::as_i64)
                        .ok_or_else(|| anyhow!("`{full_id}` must carry integer `exp`"))?;
                    PublishedAtprotoKey::bounded(key, nbf, exp)?
                } else {
                    ensure!(
                        vm.get("nbf").is_none() && vm.get("exp").is_none(),
                        "`{full_id}` active slot must not carry `nbf`/`exp`"
                    );
                    PublishedAtprotoKey::active(key)
                });
            }
        }
        let active = active.ok_or_else(|| {
            anyhow!("DID document has no `{expected_did}#atproto` verification method")
        })?;
        let mut slots = vec![active];
        slots.extend(drain);
        slots.extend(lead);
        Ok(Self { slots })
    }
}

/// Decode a `publicKeyMultibase` (multibase `z` + base58btc + multicodec
/// `p256-pub` `0x80 0x24` + 33-byte compressed SEC1) into a P-256 verifying
/// key, with the multicodec prefix checked.
///
/// `label` is the verification-method id suffix (`#atproto`)
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
fn validate_vm_authority(vm: &serde_json::Value, full_id: &str, expected_did: &str) -> Result<()> {
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

/// Resolve the active `#atproto` P-256 verifying key for `expected_did`.
///
/// This compatibility helper intentionally ignores bounded overlap slots. Use
/// [`PublishedAtprotoKeys::from_did_document`] for commit verification.
pub fn atproto_verifying_key_from_did_document(
    doc: &serde_json::Value,
    expected_did: &str,
) -> Result<VerifyingKey> {
    PublishedAtprotoKeys::from_did_document(doc, expected_did)?
        .slots
        .first()
        .map(|slot| slot.key)
        .ok_or_else(|| anyhow!("DID document has no active #atproto verification key"))
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

    fn did_doc_with_overlap(
        did: &str,
        active: &VerifyingKey,
        drain: Option<(&VerifyingKey, i64, i64)>,
        lead: Option<(&VerifyingKey, i64, i64)>,
    ) -> serde_json::Value {
        let mut methods = vec![serde_json::json!({
            "id": format!("{did}#atproto"),
            "type": "Multikey",
            "controller": did,
            "publicKeyMultibase": p256_to_multibase(active),
        })];
        for (fragment, slot) in [(DRAIN_FRAGMENT, drain), (LEAD_FRAGMENT, lead)] {
            if let Some((key, nbf, exp)) = slot {
                methods.push(serde_json::json!({
                    "id": format!("{did}#{fragment}"),
                    "type": "Multikey",
                    "controller": did,
                    "publicKeyMultibase": p256_to_multibase(key),
                    "nbf": nbf,
                    "exp": exp,
                }));
            }
        }
        serde_json::json!({ "id": did, "verificationMethod": methods })
    }

    #[test]
    fn pre_rotation_commit_verifies_during_bounded_drain_overlap() {
        let (commit, old_key) = make_signed_commit();
        let active = VerifyingKey::from(SigningKey::random(&mut rand::rngs::OsRng));
        let doc = did_doc_with_overlap(
            "did:web:alice.example.com",
            &active,
            Some((&old_key, 10, 20)),
            None,
        );
        let keys = PublishedAtprotoKeys::from_did_document(&doc, "did:web:alice.example.com")
            .expect("bounded drain document parses");
        commit
            .verify_against_published_keys(&keys, 15)
            .expect("commit signed before rotation verifies during drain overlap");
    }

    #[test]
    fn key_that_left_document_cannot_verify() {
        let (commit, old_key) = make_signed_commit();
        let active = VerifyingKey::from(SigningKey::random(&mut rand::rngs::OsRng));
        let doc = did_doc_with_overlap("did:web:alice.example.com", &active, None, None);
        let keys = PublishedAtprotoKeys::from_did_document(&doc, "did:web:alice.example.com")
            .expect("active-only document parses");
        assert!(
            commit.verify_against_published_keys(&keys, 15).is_err(),
            "a key absent from the published document must not verify"
        );
        let _ = old_key;
    }

    #[test]
    fn overlap_slot_expires_and_is_not_consulted() {
        let (commit, old_key) = make_signed_commit();
        let active = VerifyingKey::from(SigningKey::random(&mut rand::rngs::OsRng));
        let doc = did_doc_with_overlap(
            "did:web:alice.example.com",
            &active,
            Some((&old_key, 10, 20)),
            None,
        );
        let keys = PublishedAtprotoKeys::from_did_document(&doc, "did:web:alice.example.com")
            .expect("bounded drain document parses");
        assert!(
            commit.verify_against_published_keys(&keys, 20).is_err(),
            "drain key must not be consulted at its exclusive expiry boundary"
        );
    }

    #[test]
    fn currently_published_lead_key_is_accepted_for_verification_only() {
        let lead_signing_key = SigningKey::random(&mut rand::rngs::OsRng);
        let lead = VerifyingKey::from(&lead_signing_key);
        let unsigned = UnsignedCommit::new(
            "did:web:alice.example.com",
            Cid::from_dag_cbor(b"lead-window"),
            Tid::from_raw(42),
            None,
        );
        let commit = Commit::sign(&unsigned, &lead_signing_key);
        let active = VerifyingKey::from(SigningKey::random(&mut rand::rngs::OsRng));
        let doc = did_doc_with_overlap(
            "did:web:alice.example.com",
            &active,
            None,
            Some((&lead, 10, 20)),
        );
        let keys = PublishedAtprotoKeys::from_did_document(&doc, "did:web:alice.example.com")
            .expect("bounded lead document parses");
        commit
            .verify_against_published_keys(&keys, 15)
            .expect("a currently-published lead key is a verification slot");
    }

    /// Resolve the single `#atproto` key from a document, then verify a commit
    /// signed by that key passes and a forged (different-key) commit fails.
    #[test]
    fn atproto_key_from_did_document_verifies_head() {
        let (commit, vk) = make_signed_commit();
        let did = "did:web:alice.example.com";
        let resolved =
            atproto_verifying_key_from_did_document(&did_doc_with_atproto(did, &vk), did)
                .expect("resolve #atproto key");
        assert_eq!(resolved.to_sec1_bytes(), vk.to_sec1_bytes());
        // Head signed by the resolved key verifies.
        commit
            .verify(&resolved)
            .expect("head verifies under current #atproto");
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
        let resolved =
            atproto_verifying_key_from_did_document(&did_doc_with_atproto(did, &new_vk), did)
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

    /// Authority hygiene: a wrong `type` (not "Multikey") is rejected.
    #[test]
    fn atproto_key_rejects_wrong_type() {
        let (_c, vk) = make_signed_commit();
        let did = "did:web:alice.example.com";
        let doc = serde_json::json!({
            "id": did,
            "verificationMethod": [{
                "id": format!("{did}#atproto"),
                "type": "JsonWebKey2020",
                "controller": did,
                "publicKeyMultibase": p256_to_multibase(&vk),
            }],
        });
        assert!(
            atproto_verifying_key_from_did_document(&doc, did).is_err(),
            "a #atproto method with the wrong type must fail closed"
        );
    }
}
