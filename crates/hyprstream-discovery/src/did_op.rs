//! Canonical append-only operation log for hybrid-rooted `did:plc` identities.
//!
//! This module is deliberately only the operation format and verifier. It does
//! not resolve DID documents, consult a directory, or apply anchored-identity
//! policy.
//!
//! # Format
//!
//! Every operation is a canonical DAG-CBOR map:
//!
//! ```text
//! {
//!   "type": "plc_operation",
//!   "sequence": uint,
//!   "prev": null | "<CIDv1 dag-cbor sha2-256>",
//!   "rotationKeys": [{
//!     "ed25519Pub": bytes(32),
//!     "mldsa65Pub": bytes(1952)
//!   }, ...],
//!   "sig": {
//!     "ed25519": bytes(64),
//!     "mldsa65": bytes(3309)
//!   }
//! }
//! ```
//!
//! The signature covers the canonical operation with the entire `sig` field
//! omitted. Both components are mandatory and are bound using hyprstream's
//! nested Ed25519 + ML-DSA-65 composite construction. Genesis has
//! `sequence = 0` and `prev = null`; rotations increment the sequence by one
//! and set `prev` to the CIDv1 of the complete previous signed operation.
//!
//! As in the PLC method, the DID identifier is the first 24 lower-case RFC 4648
//! base32 characters of SHA-256 over the complete signed genesis DAG-CBOR.

use anyhow::{anyhow, bail, ensure, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use hyprstream_crypto::cose_sign::{
    assemble_composite_nested, sign_composite, split_composite, verify_composite,
};
use hyprstream_crypto::pq::{
    ml_dsa_sk_to_vk_bytes, ml_dsa_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey,
    MlDsaVerifyingKey,
};
use hyprstream_pds::{dag_cbor::DagCbor, Cid};

/// Fixed operation discriminator, matching the regular PLC operation name.
pub const DID_OP_TYPE: &str = "plc_operation";
/// Domain separation for both layers of every DID operation signature.
pub const DID_OP_SIGNATURE_CONTEXT: &[u8] = b"hyprstream.did-op/1";
/// PLC-compatible maximum number of active rotation keys.
pub const MAX_ROTATION_KEYS: usize = 5;
/// PQ keys and signatures are larger than classical PLC operations.
pub const MAX_DID_OP_BYTES: usize = 32 * 1024;

const ED25519_PUBLIC_KEY_LEN: usize = 32;
const ED25519_SIGNATURE_LEN: usize = 64;
const ML_DSA65_PUBLIC_KEY_LEN: usize = 1952;
const ML_DSA65_SIGNATURE_LEN: usize = 3309;
const PLC_IDENTIFIER_LEN: usize = 24;
const CIDV1_DAG_CBOR_SHA256_TEXT_LEN: usize = 59;

/// One inseparable authorization position in the rotation-key list.
///
/// The Ed25519 and ML-DSA-65 keys are paired by position. A verifier never
/// accepts one component under a different entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HybridRotationKey {
    pub ed25519_pub: [u8; ED25519_PUBLIC_KEY_LEN],
    pub mldsa65_pub: Vec<u8>,
}

impl HybridRotationKey {
    /// Build and validate a hybrid public-key pair.
    pub fn new(ed25519_pub: [u8; ED25519_PUBLIC_KEY_LEN], mldsa65_pub: Vec<u8>) -> Result<Self> {
        let key = Self {
            ed25519_pub,
            mldsa65_pub,
        };
        key.verifying_keys()?;
        Ok(key)
    }

    /// Derive the public pair corresponding to signing keys.
    pub fn from_signing_keys(ed25519: &SigningKey, mldsa65: &MlDsaSigningKey) -> Self {
        Self {
            ed25519_pub: ed25519.verifying_key().to_bytes(),
            mldsa65_pub: ml_dsa_sk_to_vk_bytes(mldsa65),
        }
    }

    fn to_dag_cbor(&self) -> DagCbor {
        DagCbor::str_map([
            ("ed25519Pub", DagCbor::Bytes(self.ed25519_pub.to_vec())),
            ("mldsa65Pub", DagCbor::Bytes(self.mldsa65_pub.clone())),
        ])
    }

    fn from_dag_cbor(value: &DagCbor) -> Result<Self> {
        require_exact_fields(value, &["ed25519Pub", "mldsa65Pub"], "rotation key")?;
        let ed25519: [u8; ED25519_PUBLIC_KEY_LEN] = required(value, "ed25519Pub")?
            .as_bytes()?
            .try_into()
            .map_err(|_| anyhow!("Ed25519 public key must be {ED25519_PUBLIC_KEY_LEN} bytes"))?;
        Self::new(ed25519, required(value, "mldsa65Pub")?.as_bytes()?.to_vec())
    }

    fn verifying_keys(&self) -> Result<(VerifyingKey, MlDsaVerifyingKey)> {
        ensure!(
            self.mldsa65_pub.len() == ML_DSA65_PUBLIC_KEY_LEN,
            "ML-DSA-65 public key must be {ML_DSA65_PUBLIC_KEY_LEN} bytes"
        );
        let ed = VerifyingKey::from_bytes(&self.ed25519_pub)
            .context("invalid Ed25519 rotation public key")?;
        let pq = ml_dsa_vk_from_bytes(&self.mldsa65_pub)
            .context("invalid ML-DSA-65 rotation public key")?;
        Ok((ed, pq))
    }

    fn matches_signers(&self, ed25519: &SigningKey, mldsa65: &MlDsaSigningKey) -> bool {
        self.ed25519_pub == ed25519.verifying_key().to_bytes()
            && self.mldsa65_pub == ml_dsa_sk_to_vk_bytes(mldsa65)
    }
}

/// Mandatory two-component signature stored on a DID operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HybridDidOpSignature {
    pub ed25519: Vec<u8>,
    pub mldsa65: Vec<u8>,
}

impl HybridDidOpSignature {
    fn placeholder() -> Self {
        Self {
            ed25519: vec![0; ED25519_SIGNATURE_LEN],
            mldsa65: vec![0; ML_DSA65_SIGNATURE_LEN],
        }
    }

    fn validate_shape(&self) -> Result<()> {
        ensure!(
            self.ed25519.len() == ED25519_SIGNATURE_LEN,
            "Ed25519 signature must be {ED25519_SIGNATURE_LEN} bytes"
        );
        ensure!(
            self.mldsa65.len() == ML_DSA65_SIGNATURE_LEN,
            "ML-DSA-65 signature must be {ML_DSA65_SIGNATURE_LEN} bytes"
        );
        Ok(())
    }

    fn to_dag_cbor(&self) -> DagCbor {
        DagCbor::str_map([
            ("ed25519", DagCbor::Bytes(self.ed25519.clone())),
            ("mldsa65", DagCbor::Bytes(self.mldsa65.clone())),
        ])
    }

    fn from_dag_cbor(value: &DagCbor) -> Result<Self> {
        require_exact_fields(value, &["ed25519", "mldsa65"], "DID operation signature")?;
        let sig = Self {
            ed25519: required(value, "ed25519")?.as_bytes()?.to_vec(),
            mldsa65: required(value, "mldsa65")?.as_bytes()?.to_vec(),
        };
        sig.validate_shape()?;
        Ok(sig)
    }
}

/// A complete signed genesis or rotation operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DidOp {
    pub sequence: u64,
    pub prev: Option<String>,
    /// Complete active rotation-key state after this operation.
    pub rotation_keys: Vec<HybridRotationKey>,
    pub signature: HybridDidOpSignature,
}

impl DidOp {
    /// Create and hybrid-sign a genesis operation.
    ///
    /// The signer must be one of the genesis operation's declared keys.
    pub fn signed_genesis(
        rotation_keys: Vec<HybridRotationKey>,
        ed25519: &SigningKey,
        mldsa65: &MlDsaSigningKey,
    ) -> Result<Self> {
        validate_rotation_keys(&rotation_keys)?;
        ensure!(
            rotation_keys
                .iter()
                .any(|key| key.matches_signers(ed25519, mldsa65)),
            "genesis signer is not one of the declared hybrid rotation keys"
        );
        Self::sign(0, None, rotation_keys, ed25519, mldsa65)
    }

    /// Append and hybrid-sign the next rotation.
    ///
    /// Authorization comes only from the immediately previous operation's key
    /// state. The new key state may remove the signer.
    pub fn signed_rotation(
        previous: &Self,
        rotation_keys: Vec<HybridRotationKey>,
        ed25519: &SigningKey,
        mldsa65: &MlDsaSigningKey,
    ) -> Result<Self> {
        previous.validate_format()?;
        validate_rotation_keys(&rotation_keys)?;
        ensure!(
            previous
                .rotation_keys
                .iter()
                .any(|key| key.matches_signers(ed25519, mldsa65)),
            "rotation signer is not authorized by the previous operation"
        );
        let sequence = previous
            .sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("DID operation sequence overflow"))?;
        Self::sign(
            sequence,
            Some(previous.cid().encode()),
            rotation_keys,
            ed25519,
            mldsa65,
        )
    }

    fn sign(
        sequence: u64,
        prev: Option<String>,
        rotation_keys: Vec<HybridRotationKey>,
        ed25519: &SigningKey,
        mldsa65: &MlDsaSigningKey,
    ) -> Result<Self> {
        let mut op = Self {
            sequence,
            prev,
            rotation_keys,
            signature: HybridDidOpSignature::placeholder(),
        };
        let payload = op.signable_bytes();
        let composite = sign_composite(ed25519, Some(mldsa65), &payload, DID_OP_SIGNATURE_CONTEXT)
            .context("hybrid-signing DID operation")?;
        let (ed_sig, pq_sig) =
            split_composite(&composite).context("splitting DID operation composite signature")?;
        op.signature = HybridDidOpSignature {
            ed25519: ed_sig,
            mldsa65: pq_sig.ok_or_else(|| {
                anyhow!("hybrid DID operation signing produced no ML-DSA-65 component")
            })?,
        };
        op.validate_format()?;
        ensure!(
            op.to_dag_cbor().len() <= MAX_DID_OP_BYTES,
            "DID operation exceeds {MAX_DID_OP_BYTES}-byte format limit"
        );
        Ok(op)
    }

    /// Canonical unsigned DAG-CBOR bytes covered by both signature components.
    pub fn signable_bytes(&self) -> Vec<u8> {
        self.to_dag_cbor_value(false).encode()
    }

    /// Canonical complete signed DAG-CBOR bytes used for CID/DID derivation.
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_dag_cbor_value(true).encode()
    }

    /// Strictly decode one canonical operation.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_DID_OP_BYTES,
            "DID operation exceeds {MAX_DID_OP_BYTES}-byte format limit"
        );
        let value = DagCbor::decode(bytes).context("decoding canonical DID operation DAG-CBOR")?;
        require_exact_fields(
            &value,
            &["prev", "rotationKeys", "sequence", "sig", "type"],
            "DID operation",
        )?;
        ensure!(
            required(&value, "type")?.as_str()? == DID_OP_TYPE,
            "unsupported DID operation type"
        );
        let prev = match required(&value, "prev")? {
            DagCbor::Null => None,
            DagCbor::Text(cid) => Some(cid.clone()),
            _ => bail!("DID operation prev must be null or a CID string"),
        };
        let rotation_keys = required(&value, "rotationKeys")?
            .as_list()?
            .iter()
            .map(HybridRotationKey::from_dag_cbor)
            .collect::<Result<Vec<_>>>()?;
        let op = Self {
            sequence: required(&value, "sequence")?.as_unsigned()?,
            prev,
            rotation_keys,
            signature: HybridDidOpSignature::from_dag_cbor(required(&value, "sig")?)?,
        };
        op.validate_format()?;
        ensure!(
            op.to_dag_cbor() == bytes,
            "DID operation is not in canonical DAG-CBOR form"
        );
        Ok(op)
    }

    /// CIDv1/dag-cbor/SHA-256 address of this complete signed operation.
    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    /// Derive the self-certifying PLC DID from a signed genesis operation.
    pub fn genesis_did(&self) -> Result<String> {
        ensure!(
            self.sequence == 0 && self.prev.is_none(),
            "PLC DID can only be derived from a genesis operation"
        );
        self.validate_format()?;
        let digest = Sha256::digest(self.to_dag_cbor());
        let identifier = base32_nopad_lower(&digest);
        Ok(format!("did:plc:{}", &identifier[..PLC_IDENTIFIER_LEN]))
    }

    fn validate_format(&self) -> Result<()> {
        match (&self.prev, self.sequence) {
            (None, 0) => {}
            (Some(_), 0) => bail!("genesis sequence zero must have a null prev"),
            (Some(prev), _) => validate_cid_text(prev)?,
            (None, _) => bail!("only sequence zero may have a null prev"),
        }
        validate_rotation_keys(&self.rotation_keys)?;
        self.signature.validate_shape()
    }

    fn to_dag_cbor_value(&self, include_signature: bool) -> DagCbor {
        let mut fields = vec![
            (
                "prev",
                self.prev
                    .as_ref()
                    .map_or(DagCbor::Null, |cid| DagCbor::Text(cid.clone())),
            ),
            (
                "rotationKeys",
                DagCbor::list(
                    self.rotation_keys
                        .iter()
                        .map(HybridRotationKey::to_dag_cbor),
                ),
            ),
            ("sequence", DagCbor::Unsigned(self.sequence)),
            ("type", DagCbor::Text(DID_OP_TYPE.to_owned())),
        ];
        if include_signature {
            fields.push(("sig", self.signature.to_dag_cbor()));
        }
        DagCbor::str_map(fields)
    }
}

/// Verified head state returned by [`verify_did_op_log`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedDidOpLog {
    pub did: String,
    pub sequence: u64,
    pub head_cid: String,
    pub rotation_keys: Vec<HybridRotationKey>,
}

/// Verify a complete linear operation log, failing closed on every discrepancy.
///
/// Verification pins Hybrid independently of any runtime crypto policy:
///
/// - the genesis DID must match the hash-derived expected DID;
/// - genesis must verify under one of its own declared hybrid key pairs;
/// - every rotation must link to the immediately prior signed operation;
/// - sequence values must increase by exactly one;
/// - every rotation must verify under one complete pair from the prior state.
pub fn verify_did_op_log(expected_did: &str, log: &[DidOp]) -> Result<VerifiedDidOpLog> {
    let genesis = log
        .first()
        .ok_or_else(|| anyhow!("DID operation log is empty"))?;
    genesis
        .validate_format()
        .context("invalid genesis format")?;
    ensure!(
        genesis.sequence == 0 && genesis.prev.is_none(),
        "first DID operation is not genesis"
    );
    let derived_did = genesis.genesis_did()?;
    ensure!(
        expected_did == derived_did,
        "genesis DID mismatch: expected {expected_did:?}, derived {derived_did:?}"
    );
    verify_signature(genesis, &genesis.rotation_keys).context("invalid genesis signature")?;

    for (index, pair) in log.windows(2).enumerate() {
        let previous = &pair[0];
        let current = &pair[1];
        current
            .validate_format()
            .with_context(|| format!("invalid operation {} format", index + 1))?;
        let expected_sequence = previous
            .sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("DID operation sequence overflow"))?;
        ensure!(
            current.sequence == expected_sequence,
            "operation {} sequence is {}, expected {}",
            index + 1,
            current.sequence,
            expected_sequence
        );
        let expected_prev = previous.cid().encode();
        ensure!(
            current.prev.as_deref() == Some(expected_prev.as_str()),
            "operation {} has a broken previous-operation hash link",
            index + 1
        );
        verify_signature(current, &previous.rotation_keys).with_context(|| {
            format!(
                "operation {} signature is unauthorized or invalid",
                index + 1
            )
        })?;
    }

    let head = log
        .last()
        .ok_or_else(|| anyhow!("DID operation log is empty"))?;
    Ok(VerifiedDidOpLog {
        did: derived_did,
        sequence: head.sequence,
        head_cid: head.cid().encode(),
        rotation_keys: head.rotation_keys.clone(),
    })
}

fn verify_signature(op: &DidOp, authorized: &[HybridRotationKey]) -> Result<()> {
    op.signature.validate_shape()?;
    let payload = op.signable_bytes();
    let mut errors = Vec::new();

    for key in authorized {
        let result = (|| {
            let (ed, pq) = key.verifying_keys()?;
            let composite = assemble_composite_nested(
                (ed.to_bytes().to_vec(), op.signature.ed25519.clone()),
                Some((ml_dsa_vk_bytes(&pq), op.signature.mldsa65.clone())),
            )
            .context("assembling hybrid DID operation signature")?;
            let verified = verify_composite(
                &composite,
                &ed,
                Some(&pq),
                &payload,
                DID_OP_SIGNATURE_CONTEXT,
                true,
            )
            .context("verifying hybrid DID operation signature")?;
            ensure!(
                verified.eddsa && verified.ml_dsa,
                "both hybrid signature components were not verified"
            );
            Ok::<(), anyhow::Error>(())
        })();
        match result {
            Ok(()) => return Ok(()),
            Err(error) => errors.push(error.to_string()),
        }
    }

    bail!(
        "no authorized hybrid rotation key verified both signature components: {}",
        errors.join("; ")
    )
}

fn validate_rotation_keys(keys: &[HybridRotationKey]) -> Result<()> {
    ensure!(!keys.is_empty(), "rotation key list must not be empty");
    ensure!(
        keys.len() <= MAX_ROTATION_KEYS,
        "rotation key list exceeds maximum of {MAX_ROTATION_KEYS}"
    );
    for (index, key) in keys.iter().enumerate() {
        key.verifying_keys()
            .with_context(|| format!("invalid rotation key at index {index}"))?;
        ensure!(
            !keys[..index].contains(key),
            "duplicate hybrid rotation key at index {index}"
        );
    }
    Ok(())
}

fn validate_cid_text(cid: &str) -> Result<()> {
    ensure!(
        cid.len() == CIDV1_DAG_CBOR_SHA256_TEXT_LEN,
        "previous-operation CID must be {CIDV1_DAG_CBOR_SHA256_TEXT_LEN} characters"
    );
    ensure!(
        cid.starts_with('b')
            && cid[1..]
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || matches!(byte, b'2'..=b'7')),
        "previous-operation CID must use canonical base32lower multibase"
    );
    Ok(())
}

fn required<'a>(value: &'a DagCbor, field: &str) -> Result<&'a DagCbor> {
    value
        .get(field)
        .ok_or_else(|| anyhow!("missing required field {field:?}"))
}

fn require_exact_fields(value: &DagCbor, expected: &[&str], what: &str) -> Result<()> {
    let map = value
        .as_map()
        .with_context(|| format!("{what} must be a map"))?;
    ensure!(
        map.len() == expected.len(),
        "{what} must contain exactly fields {expected:?}"
    );
    for field in expected {
        ensure!(
            value.get(field).is_some(),
            "{what} is missing required field {field:?}"
        );
    }
    Ok(())
}

fn base32_nopad_lower(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut output = String::with_capacity((bytes.len() * 8).div_ceil(5));
    let mut buffer = 0_u64;
    let mut bits = 0_u32;
    for &byte in bytes {
        buffer = (buffer << 8) | u64::from(byte);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            output.push(ALPHABET[((buffer >> bits) & 0x1f) as usize] as char);
        }
    }
    if bits > 0 {
        output.push(ALPHABET[((buffer << (5 - bits)) & 0x1f) as usize] as char);
    }
    output
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_crypto::pq::{ml_dsa_sk_from_seed, MlDsaSigningKey};

    struct TestKeys {
        ed: SigningKey,
        pq: MlDsaSigningKey,
        public: HybridRotationKey,
    }

    fn keys(seed: u8) -> TestKeys {
        let ed = SigningKey::from_bytes(&[seed; 32]);
        let pq = ml_dsa_sk_from_seed(&[seed.wrapping_add(64); 32]);
        let public = HybridRotationKey::from_signing_keys(&ed, &pq);
        TestKeys { ed, pq, public }
    }

    fn two_op_log() -> (String, Vec<DidOp>) {
        let first = keys(1);
        let second = keys(2);
        let genesis =
            DidOp::signed_genesis(vec![first.public.clone()], &first.ed, &first.pq).unwrap();
        let did = genesis.genesis_did().unwrap();
        let rotation =
            DidOp::signed_rotation(&genesis, vec![second.public], &first.ed, &first.pq).unwrap();
        (did, vec![genesis, rotation])
    }

    #[test]
    fn genesis_derives_plc_identifier_from_signed_canonical_bytes() {
        let key = keys(7);
        let genesis = DidOp::signed_genesis(vec![key.public], &key.ed, &key.pq).unwrap();
        let digest = Sha256::digest(genesis.to_dag_cbor());
        let expected = format!(
            "did:plc:{}",
            &base32_nopad_lower(&digest)[..PLC_IDENTIFIER_LEN]
        );

        assert_eq!(genesis.genesis_did().unwrap(), expected);
        assert_eq!(expected, "did:plc:a3remwhfbs5ozfknkp6krfql");
        assert_eq!(expected.len(), "did:plc:".len() + PLC_IDENTIFIER_LEN);
    }

    #[test]
    fn valid_rotation_chain_verifies_and_returns_head_state() {
        let (did, log) = two_op_log();

        let verified = verify_did_op_log(&did, &log).unwrap();

        assert_eq!(verified.did, did);
        assert_eq!(verified.sequence, 1);
        assert_eq!(verified.head_cid, log[1].cid().encode());
        assert_eq!(verified.rotation_keys, log[1].rotation_keys);
    }

    #[test]
    fn canonical_encoding_round_trips() {
        let (_, log) = two_op_log();
        for op in log {
            let bytes = op.to_dag_cbor();
            let decoded = DidOp::from_dag_cbor(&bytes).unwrap();
            assert_eq!(decoded, op);
            assert_eq!(decoded.to_dag_cbor(), bytes);
        }
    }

    #[test]
    fn broken_hash_link_fails_closed() {
        let (did, mut log) = two_op_log();
        log[1].prev = Some(Cid::from_dag_cbor(b"attacker-selected").encode());

        let error = verify_did_op_log(&did, &log).unwrap_err();
        assert!(error
            .to_string()
            .contains("broken previous-operation hash link"));
    }

    #[test]
    fn bad_signature_fails_closed() {
        let (did, mut log) = two_op_log();
        log[1].signature.ed25519[0] ^= 0x80;

        let error = verify_did_op_log(&did, &log).unwrap_err();
        assert!(error
            .to_string()
            .contains("signature is unauthorized or invalid"));
    }

    #[test]
    fn classical_only_signature_fails_at_hybrid_required_position() {
        let first = keys(11);
        let second = keys(12);
        let genesis =
            DidOp::signed_genesis(vec![first.public.clone()], &first.ed, &first.pq).unwrap();
        let mut rotation =
            DidOp::signed_rotation(&genesis, vec![second.public], &first.ed, &first.pq).unwrap();
        let classical = sign_composite(
            &first.ed,
            None,
            &rotation.signable_bytes(),
            DID_OP_SIGNATURE_CONTEXT,
        )
        .unwrap();
        let (ed_sig, pq_sig) = split_composite(&classical).unwrap();
        assert!(pq_sig.is_none());
        rotation.signature = HybridDidOpSignature {
            ed25519: ed_sig,
            mldsa65: Vec::new(),
        };
        let did = genesis.genesis_did().unwrap();

        let error = verify_did_op_log(&did, &[genesis, rotation]).unwrap_err();
        assert!(format!("{error:#}").contains("ML-DSA-65 signature must be"));
    }

    #[test]
    fn sequence_gap_fails_closed() {
        let (did, mut log) = two_op_log();
        log[1].sequence = 2;

        let error = verify_did_op_log(&did, &log).unwrap_err();
        assert!(error.to_string().contains("sequence is 2, expected 1"));
    }

    #[test]
    fn newly_introduced_key_cannot_authorize_its_own_rotation() {
        let first = keys(21);
        let attacker = keys(22);
        let genesis = DidOp::signed_genesis(vec![first.public], &first.ed, &first.pq).unwrap();

        let error =
            DidOp::signed_rotation(&genesis, vec![attacker.public], &attacker.ed, &attacker.pq)
                .unwrap_err();
        assert!(error.to_string().contains("not authorized"));
    }

    #[test]
    fn wrong_expected_did_fails_closed() {
        let (_, log) = two_op_log();

        let error = verify_did_op_log("did:plc:aaaaaaaaaaaaaaaaaaaaaaaa", &log).unwrap_err();
        assert!(error.to_string().contains("genesis DID mismatch"));
    }
}
