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
    /// Accepts the commit if its ES256 signature verifies under **any one** of
    /// the supplied `#atproto` P-256 verifying keys. `keys` is the set of
    /// currently-published verification methods for the account DID — under the
    /// drain-slot design, that is the active slot plus any drain slot(s) the
    /// DID document still advertises (and optionally the lead slot once it is
    /// `nbf`-valid). Verification fails closed if the slice is empty or no key
    /// verifies.
    ///
    /// # Why this design (drain-slot publication), and not the alternative
    ///
    /// Issue #918 lists two candidate fixes:
    ///
    /// 1. **Drain-slot publication** *(chosen)* — publish the drain (and lead)
    ///    ES256 slots as additional `#atproto`-style verification methods in
    ///    the DID document, and have the verifier accept any currently-
    ///    published slot. This is exactly what this method implements on the
    ///    verifier side. It keeps verification **fail-closed** (an empty slot
    ///    set, or a signature under no published slot, is rejected — never a
    ///    silent fallback), and it preserves commits byte-for-byte: history is
    ///    not rewritten, and a historical commit keeps verifying as long as
    ///    its signing key remains in the bounded drain window, then stops
    ///    verifying loudly once the slot is dropped. The drain window is
    ///    bounded by the ES256 store's `drain_secs` policy, so the accepted
    ///    key set is small and time-limited.
    /// 2. **Re-sign on rotation** *(rejected)* — when the active ES256 key
    ///    rotates, walk every stored commit and re-sign it with the new active
    ///    key. This rewrites persisted history invisibly: a commit CID a
    ///    consumer recorded at time *T* no longer matches the served commit at
    ///    *T+1*, and the rotation task becomes coupled to the PDS store. It
    ///    also silently changes the trust root of historical commits rather
    ///    than letting the drain window expire them. We reject it for the same
    ///    reasons `PdsPublisher` rejects read-side re-signing: the signed
    ///    mutable pointer is written once, at publish time, and serves verbatim.
    ///
    /// # Fail-closed contract
    ///
    /// - `keys` empty → `Err` (no authoritative key was supplied; the verifier
    ///   refuses to accept a signature it cannot bind to a published slot).
    /// - No supplied key verifies → `Err` (the last attempted key's error is
    ///   surfaced; the commit is untrusted).
    /// - At least one verifies → `Ok(())`.
    ///
    /// The caller is responsible for bounding `keys` to the published slots;
    /// this method does not enforce a count or time window itself — the
    /// authoritative bound is "what the resolved DID document advertises."
    pub fn verify_against_keys<'a, I>(&self, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = &'a VerifyingKey>,
    {
        let mut count = 0usize;
        let mut last_err: Option<anyhow::Error> = None;
        for vk in keys {
            count += 1;
            match self.verify(vk) {
                Ok(()) => return Ok(()),
                Err(e) => last_err = Some(e),
            }
        }
        match last_err {
            Some(e) => Err(e.context(format!(
                "ES256 signature verified under none of the {count} published #atproto slot(s)"
            ))),
            // Empty key set: fail closed — never accept a signature we cannot
            // bind to at least one currently-published verifying key.
            None => anyhow::bail!(
                "rotation-tolerant verify called with no #atproto verifying keys \
                 — refusing to verify without an authoritative slot set"
            ),
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

    // ── #918: rotation-survivable verification ───────────────────────────────

    /// A commit signed by a since-rotated-out key must still verify against the
    /// published slot set (active + drain) — the drain-slot design.
    #[test]
    fn commit_verify_survives_key_rotation() {
        let (commit, original_vk) = make_signed_commit();
        // The active key rotates: a brand-new key replaces it.
        let new_signing = SigningKey::random(&mut rand::rngs::OsRng);
        let new_vk = VerifyingKey::from(&new_signing);

        // Single-key verify against the *new* (current DID document) key fails —
        // this is the #918 regression: the historical commit looks untrusted.
        assert!(
            commit.verify(&new_vk).is_err(),
            "historical commit must not verify under the rotated-in key alone"
        );

        // Rotation-tolerant verify against the published slot set {drain=old,
        // active=new} succeeds — drain-slot publication.
        let published_slots = [&new_vk, &original_vk];
        commit
            .verify_against_keys(published_slots.iter().copied())
            .expect("commit signed by a published drain slot must verify");
    }

    /// A genuinely bad signature must still fail under rotation-tolerant verify.
    #[test]
    fn commit_verify_against_keys_rejects_bad_signature() {
        let (commit, _original_vk) = make_signed_commit();
        // Two unrelated keys the commit was never signed by.
        let a = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        let b = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        assert!(
            commit.verify_against_keys([&a, &b]).is_err(),
            "a signature under no published slot must fail, not silently fall back"
        );
    }

    /// An empty published-slot set fails closed — never accept a signature we
    /// cannot bind to at least one authoritative key.
    #[test]
    fn commit_verify_against_keys_fails_closed_on_empty() {
        let (commit, _vk) = make_signed_commit();
        let err = commit
            .verify_against_keys(std::iter::empty::<&VerifyingKey>())
            .expect_err("empty key set must fail closed");
        let msg = format!("{err}");
        assert!(
            msg.contains("no #atproto verifying keys"),
            "empty-set error should explain fail-closed reason, got: {msg}"
        );
    }

    /// Rotation-tolerant verify still detects a tampered commit across a slot set.
    #[test]
    fn commit_verify_against_keys_detects_tampered_data() {
        let (mut commit, original_vk) = make_signed_commit();
        let new_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        commit.data = Cid::from_dag_cbor(b"tampered");
        let published_slots = [&new_vk, &original_vk];
        assert!(
            commit
                .verify_against_keys(published_slots.iter().copied())
                .is_err(),
            "tampered commit must fail under every published slot"
        );
    }
}
