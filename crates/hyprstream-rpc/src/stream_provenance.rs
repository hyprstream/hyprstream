//! Per-host provenance signing/verification for the mesh stream plane (#321).
//!
//! The chained HMAC on `StreamBlock` proves "the producer held the DH key and
//! delivered blocks in order" — it does NOT prove "host-X computed this block"
//! (threat T3 / C-PROV in the authz threat-model). Provenance closes that gap:
//! each block is signed with the producing host's **per-host hybrid COSE
//! identity** (Ed25519 + ML-DSA-65, #328 `derive_mesh_mldsa_key`), and the
//! consumer verifies the signature AND that the signer key is enrolled in the
//! mesh roster. This is a layer ON TOP of AEAD + HMAC.
//!
//! ## Canonical signed region
//!
//! The signature covers the *signed region*: the StreamBlock serialized with the
//! `provenance` field left empty — i.e. exactly [`crate::streaming::encode_stream_block`]
//! over `(prevMac, sequenceNumber, epoch, payloads)`. The publisher signs that
//! region, then attaches the `provenance` field; the consumer reconstructs the
//! same region from the parsed (provenance-cleared) block and verifies. The
//! signature therefore binds the post-AEAD payloads, the sequence number, and the
//! epoch — but is independent of the (circular) provenance bytes themselves.

use anyhow::{anyhow, Result};

use crate::crypto::cose_sign::{sign_composite, verify_composite};
use crate::crypto::pq::{MlDsaSigningKey, MlDsaVerifyingKey};
use crate::envelope::PqTrustStore;
use crate::streaming::StreamPayloadData;
use crate::streaming_capnp;

/// Cap'n Proto file id of `streaming.capnp` — first half of the COSE
/// `external_aad` schema binding for provenance (prevents cross-schema replay).
const STREAMING_SCHEMA_ID: u64 = 0xe7f8_a9b0_c1d2_e3f4;

/// Stable inner-type id binding the provenance COSE composite to the StreamBlock
/// signed region (≠ any envelope type id, so a provenance signature can never be
/// replayed as a request/response envelope signature).
const STREAM_BLOCK_PROVENANCE_TYPE_ID: u64 = 0x5374_726d_4270_3031; // "StrmBp01"

/// COSE `external_aad` for StreamBlock provenance signatures.
fn provenance_external_aad() -> Vec<u8> {
    crate::crypto::cose_sign1::build_external_aad(
        STREAMING_SCHEMA_ID,
        STREAM_BLOCK_PROVENANCE_TYPE_ID,
    )
}

/// The producing host's per-host hybrid signing identity (#328): the node's
/// Ed25519 mesh signer + its deterministically-derived ML-DSA-65 mesh key.
#[derive(Clone)]
pub struct ProvenanceSigner {
    ed_sk: ed25519_dalek::SigningKey,
    pq_sk: MlDsaSigningKey,
}

impl ProvenanceSigner {
    /// Build from the node's Ed25519 mesh signing key. The ML-DSA-65 half is
    /// derived deterministically via [`crate::node_identity::derive_mesh_mldsa_key`]
    /// (#157/#328), so the signer matches the `#mesh-pq` key peers anchor.
    pub fn from_ed25519(ed_sk: ed25519_dalek::SigningKey) -> Self {
        let pq_sk = crate::node_identity::derive_mesh_mldsa_key(&ed_sk);
        Self { ed_sk, pq_sk }
    }

    /// Build from an explicit Ed25519 + ML-DSA-65 keypair.
    pub fn new(ed_sk: ed25519_dalek::SigningKey, pq_sk: MlDsaSigningKey) -> Self {
        Self { ed_sk, pq_sk }
    }

    /// The host's signer kid = its Ed25519 verifying-key bytes (matches the COSE
    /// inner kid and the mesh-roster key).
    pub fn signer_kid(&self) -> [u8; 32] {
        self.ed_sk.verifying_key().to_bytes()
    }

    /// Sign the canonical `signed_region` with the hybrid COSE composite (#321),
    /// returning `(signer_kid, cose_sig_bytes)` for the `provenance` field.
    pub fn sign(&self, signed_region: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        let aad = provenance_external_aad();
        let sig = sign_composite(&self.ed_sk, Some(&self.pq_sk), signed_region, &aad)?;
        Ok((self.signer_kid().to_vec(), sig))
    }
}

/// Verify a StreamBlock's provenance (#321), fail-closed.
///
/// `roster` resolves the signer's anchored ML-DSA-65 key (the mesh
/// `KeyedPqTrustStore`); `is_enrolled` confirms the signer's Ed25519 key is a
/// known/enrolled mesh peer. Rejects on missing/unknown/invalid signer.
///
/// `signed_region` is the provenance-cleared block encoding (see module docs);
/// `signer_kid`/`sig` come from the wire `StreamBlock.provenance` field.
pub fn verify_provenance(
    signer_kid: &[u8],
    sig: &[u8],
    signed_region: &[u8],
    roster: &dyn PqTrustStore,
    is_enrolled: &dyn Fn(&[u8; 32]) -> bool,
) -> Result<()> {
    if sig.is_empty() {
        anyhow::bail!("stream provenance: missing signature (fail-closed)");
    }
    let kid: [u8; 32] = signer_kid
        .try_into()
        .map_err(|_| anyhow!("stream provenance: signer kid must be 32 bytes (Ed25519)"))?;

    // The signer must be an enrolled mesh peer — never accept an unknown signer.
    if !is_enrolled(&kid) {
        anyhow::bail!("stream provenance: signer is not an enrolled mesh peer (unknown signer)");
    }

    let ed_vk = ed25519_dalek::VerifyingKey::from_bytes(&kid)
        .map_err(|e| anyhow!("stream provenance: invalid Ed25519 signer key: {e}"))?;

    // The anchored ML-DSA-65 key for this signer (per-host #mesh-pq). Required:
    // verify under the Hybrid policy (fail-closed on a stripped/absent PQ layer).
    let pq_vk: MlDsaVerifyingKey = roster
        .ml_dsa_key_for(&kid)
        .ok_or_else(|| anyhow!("stream provenance: no anchored ML-DSA-65 key for signer"))?;

    let aad = provenance_external_aad();
    verify_composite(sig, &ed_vk, Some(&pq_vk), signed_region, &aad, /* require_pq */ true)
        .map_err(|e| anyhow!("stream provenance: signature verification failed: {e}"))?;
    Ok(())
}

/// Reconstruct the canonical signed region from a parsed (wire) StreamBlock by
/// re-encoding it with the `provenance` field cleared (#321). Byte-identical to
/// the publisher's pre-provenance [`crate::streaming::encode_stream_block`].
pub fn signed_region_from_block(
    block: &streaming_capnp::stream_block::Reader,
) -> Result<Vec<u8>> {
    use streaming_capnp::stream_payload::Which;

    let prev_mac = block.get_prev_mac()?;
    let sequence_number = block.get_sequence_number();
    let epoch = block.get_epoch();

    let payloads_reader = block.get_payloads()?;
    let mut payloads: Vec<StreamPayloadData> = Vec::with_capacity(payloads_reader.len() as usize);
    for i in 0..payloads_reader.len() {
        let p = payloads_reader.get(i);
        let data = match p.which()? {
            Which::Data(d) => StreamPayloadData::Data(d?.to_vec()),
            Which::Complete(d) => StreamPayloadData::Complete(d?.to_vec()),
            Which::Error(e) => {
                let _ = e?; // error frames are not provenance-signed in practice
                StreamPayloadData::Error(String::new())
            }
            Which::Heartbeat(()) => continue,
            Which::Tagged(t) => {
                let t = t?;
                StreamPayloadData::Tagged {
                    tag: t.get_tag()?.to_vec(),
                    payload: t.get_payload()?.to_vec(),
                    nonce: t.get_nonce()?.to_vec(),
                    key_commitment: t.get_key_commitment()?.to_vec(),
                }
            }
        };
        payloads.push(data);
    }

    crate::streaming::encode_stream_block(prev_mac, sequence_number, epoch, &payloads)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::envelope::KeyedPqTrustStore;
    use ed25519_dalek::SigningKey;

    fn signer() -> ProvenanceSigner {
        ProvenanceSigner::from_ed25519(SigningKey::from_bytes(&[7u8; 32]))
    }

    /// Build a roster anchoring `signer`'s ML-DSA key, plus an enrolled-set closure.
    fn roster_for(signer: &ProvenanceSigner) -> (KeyedPqTrustStore, [u8; 32]) {
        use ml_dsa::Keypair;
        let kid = signer.signer_kid();
        let mut store = KeyedPqTrustStore::new();
        let pq_vk = signer.pq_sk.verifying_key();
        store.bind(kid, &pq_vk);
        (store, kid)
    }

    #[test]
    fn sign_verify_roundtrip() {
        let s = signer();
        let (store, kid) = roster_for(&s);
        let region = b"canonical signed region bytes";
        let (signer_kid, sig) = s.sign(region).unwrap();
        assert_eq!(signer_kid, kid.to_vec());

        let enrolled = |k: &[u8; 32]| *k == kid;
        verify_provenance(&signer_kid, &sig, region, &store, &enrolled)
            .expect("valid provenance must verify");
    }

    #[test]
    fn tampered_region_rejected() {
        let s = signer();
        let (store, kid) = roster_for(&s);
        let (signer_kid, sig) = s.sign(b"original region").unwrap();
        let enrolled = |k: &[u8; 32]| *k == kid;
        let res = verify_provenance(&signer_kid, &sig, b"tampered region", &store, &enrolled);
        assert!(res.is_err(), "tampered region must fail provenance verify");
    }

    #[test]
    fn wrong_signer_rejected() {
        // A different host signs; verifier anchors the EXPECTED host. The kid won't
        // resolve in the roster / enrolled set → rejected.
        let real = signer();
        let attacker = ProvenanceSigner::from_ed25519(SigningKey::from_bytes(&[9u8; 32]));
        let (store, real_kid) = roster_for(&real);
        let region = b"region";
        let (att_kid, att_sig) = attacker.sign(region).unwrap();

        // Only the real host is enrolled; the attacker's kid is unknown.
        let enrolled = |k: &[u8; 32]| *k == real_kid;
        let res = verify_provenance(&att_kid, &att_sig, region, &store, &enrolled);
        assert!(res.is_err(), "wrong (unenrolled) signer must be rejected");
    }

    #[test]
    fn unknown_signer_rejected() {
        // Signer is enrolled in the set but has NO anchored ML-DSA key (roster miss).
        let s = signer();
        let kid = s.signer_kid();
        let region = b"region";
        let (signer_kid, sig) = s.sign(region).unwrap();
        let empty_store = KeyedPqTrustStore::new();
        let enrolled = |k: &[u8; 32]| *k == kid;
        let res = verify_provenance(&signer_kid, &sig, region, &empty_store, &enrolled);
        assert!(res.is_err(), "signer without an anchored PQ key must be rejected");
    }

    #[test]
    fn empty_signature_rejected() {
        let s = signer();
        let (store, kid) = roster_for(&s);
        let enrolled = |k: &[u8; 32]| *k == kid;
        let res = verify_provenance(kid.as_ref(), &[], b"region", &store, &enrolled);
        assert!(res.is_err(), "empty signature must fail closed");
    }
}
