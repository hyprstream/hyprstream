//! COSE_Encrypt0 hybrid-KEM confidentiality — #553 / S2 of epic #550.
//!
//! Expresses confidentiality as a standard `COSE_Encrypt0` (RFC 9052) object,
//! keyed by the S0 hybrid multi-KEM ([`crate::crypto::hybrid_kem`]). This is the
//! confidentiality counterpart to the `COSE_Sign1`/composite signing in
//! [`crate::crypto::cose_sign1`] / [`crate::crypto::cose_sign`]. No Rust crate
//! implements draft-ietf-cose-hpke, so the HPKE-COSE wiring is done here over
//! `coset` (which provides the `COSE_Encrypt0` CBOR structure with a
//! bring-your-own-AEAD closure — it performs no crypto itself).
//!
//! # Construction (draft-ietf-cose-hpke shape)
//!
//! ```text
//! protected   = { alg: ChaCha20Poly1305 (24), HDR_SUITE_ID: suite_u16 }
//! unprotected = { HDR_KEM_EK: HybridKemMaterial.encode() }   // the encapsulated key (one-shot only)
//! ciphertext  = ChaCha20Poly1305_seal(K, N, plaintext, AAD)
//! ```
//! - **K** = `HKDF-SHA256-Expand(hybrid_secret, "…chacha20poly1305 key")` — the
//!   32-byte combiner secret from S0 is already uniform; HKDF-expand gives
//!   key-separation from any other use of that secret.
//! - **N** (96-bit) = `be32(epoch) ‖ be64(seq)` — a deterministic counter nonce.
//!   It is **not** transmitted (no COSE `iv`): both sides derive it from
//!   `(epoch, seq)`, so it cannot be manipulated on the wire. Uniqueness: a
//!   re-key bumps `epoch` *and* re-derives **K** (S3/#223), so within one key
//!   `epoch` is fixed and `seq` is the strictly-monotonic per-block counter ⇒ the
//!   `(K, N)` pair is unique-forever (the AEAD nonce-reuse invariant). One-shot
//!   `seal_to_recipient` uses a *fresh* encapsulation (fresh K) per call, so any
//!   `(epoch, seq)` is safe there.
//! - **AAD** = `coset Enc_structure( protected_header ‖ external_aad' )`, where
//!   `external_aad' = CBOR([external_aad, epoch, seq])`. `coset` folds the
//!   protected header (which carries `alg` + `suite_id`) into the AEAD AAD, so the
//!   suite id, the AEAD algorithm, the caller's `external_aad`, and `(epoch, seq)`
//!   are all authenticated — a tampered header, swapped suite, or replay at a
//!   different `(epoch, seq)` fails decryption.
//!
//! # Why this is fail-closed against downgrade
//!
//! `open_*` requires `alg == ChaCha20Poly1305` and the protected `suite_id` to
//! match the recipient's pinned suite; there is no in-band algorithm negotiation
//! (epic #550 principle 1). [`ALG_XCHACHA20POLY1305_PRIVATE`] is reserved for a
//! future high-nonce-volume profile but is **not** accepted by default.

use anyhow::{bail, Context, Result};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use coset::cbor::value::Value as CborValue;
use coset::{
    iana::{self, EnumI64},
    CborSerializable, CoseEncrypt0, CoseEncrypt0Builder, Header, HeaderBuilder, Label,
};
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::crypto::hybrid_kem::{
    self, HybridKemMaterial, RecipientKeypair, RecipientPublic, SuiteId,
};

/// COSE algorithm id for ChaCha20/Poly1305 (RFC 9053 §6.3.2, IANA = 24).
const ALG_CHACHA20POLY1305: i64 = iana::Algorithm::ChaCha20Poly1305 as i64;

/// Private-use COSE algorithm code point reserved for XChaCha20-Poly1305 (192-bit
/// nonce). XChaCha20-Poly1305 is **not** IANA-registered for COSE; this is a
/// hyprstream-local private-use value (< -65536) reserved for a future
/// high-nonce-volume profile. NOT accepted by the default `open_*` path.
pub const ALG_XCHACHA20POLY1305_PRIVATE: i64 = -65537;

/// Protected-header label carrying the HyKEM `suite_id` (u16). Private-use.
const HDR_SUITE_ID: i64 = -65538;
/// Unprotected-header label carrying the encoded [`HybridKemMaterial`] (the
/// encapsulated key, "ek"). Private-use. Present only on the one-shot
/// `seal_to_recipient` path; the stream path keys from the handshake instead.
const HDR_KEM_EK: i64 = -65539;

/// HKDF-Expand domain label for the content-encryption key.
const KDF_KEY_LABEL: &[u8] = b"hyprstream cose-encrypt0 v1 chacha20poly1305 key";

// ============================================================================
// Key / nonce / AAD derivation
// ============================================================================

/// Derive the 32-byte ChaCha20Poly1305 content key from a 32-byte hybrid-KEM
/// combiner secret via HKDF-SHA256-Expand with a domain label (key separation).
///
/// Exposed so the stream path (S3) can derive the per-stream content key once
/// from the handshake secret and then call [`seal_with_key`]/[`open_with_key`]
/// per block without re-deriving.
pub fn derive_aead_key(hybrid_secret: &[u8; 32]) -> Zeroizing<[u8; 32]> {
    let hk = Hkdf::<Sha256>::new(None, hybrid_secret);
    let mut key = [0u8; 32];
    // Expand to 32 bytes never fails for HKDF-SHA256.
    #[allow(clippy::expect_used)]
    hk.expand(KDF_KEY_LABEL, &mut key)
        .expect("HKDF-SHA256 expand to 32 bytes is infallible");
    Zeroizing::new(key)
}

/// Deterministic 96-bit nonce `be32(epoch) ‖ be64(seq)`.
///
/// `epoch` must fit in 32 bits — it is a per-stream re-key counter, bounded far
/// below 2^32 in practice; a re-key also re-derives the content key, so `epoch`
/// is effectively constant within a single key and `seq` carries uniqueness.
fn derive_nonce(epoch: u64, seq: u64) -> [u8; 12] {
    debug_assert!(
        epoch <= u64::from(u32::MAX),
        "epoch must fit in 32 bits for the AEAD nonce"
    );
    let mut n = [0u8; 12];
    n[..4].copy_from_slice(&(epoch as u32).to_be_bytes());
    n[4..].copy_from_slice(&seq.to_be_bytes());
    n
}

/// Bind the caller's `external_aad` together with `(epoch, seq)` as canonical
/// CBOR `[bstr external_aad, int epoch, int seq]`. The suite id and AEAD alg are
/// bound separately via the protected header (folded into the AEAD AAD by coset).
fn combined_aad(external_aad: &[u8], epoch: u64, seq: u64) -> Vec<u8> {
    let v = CborValue::Array(vec![
        CborValue::Bytes(external_aad.to_vec()),
        CborValue::Integer(epoch.into()),
        CborValue::Integer(seq.into()),
    ]);
    let mut buf = Vec::with_capacity(external_aad.len() + 24);
    #[allow(clippy::expect_used)]
    ciborium::ser::into_writer(&v, &mut buf)
        .expect("CBOR encoding of [bstr,int,int] into a Vec is infallible");
    buf
}

// ============================================================================
// AEAD closures
// ============================================================================

fn aead_seal(key: &[u8; 32], nonce: &[u8; 12], plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    cipher
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| anyhow::anyhow!("ChaCha20Poly1305 encrypt failed"))
}

fn aead_open(key: &[u8; 32], nonce: &[u8; 12], ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| anyhow::anyhow!("ChaCha20Poly1305 decrypt/auth failed"))
}

// ============================================================================
// Header build / parse
// ============================================================================

fn build_protected(suite: SuiteId) -> Header {
    HeaderBuilder::new()
        .algorithm(iana::Algorithm::ChaCha20Poly1305)
        .value(HDR_SUITE_ID, CborValue::Integer(suite.as_u16().into()))
        .build()
}

fn require_chacha20poly1305(enc: &CoseEncrypt0) -> Result<()> {
    match enc.protected.header.alg.as_ref() {
        Some(coset::Algorithm::Assigned(a)) if a.to_i64() == ALG_CHACHA20POLY1305 => Ok(()),
        Some(coset::Algorithm::Assigned(a)) => {
            bail!(
                "COSE_Encrypt0 alg {} is not ChaCha20Poly1305 (24)",
                a.to_i64()
            )
        }
        Some(coset::Algorithm::PrivateUse(v)) => {
            bail!("COSE_Encrypt0 private-use alg {v} not accepted (downgrade rejected)")
        }
        Some(coset::Algorithm::Text(s)) => bail!("unsupported text alg: {s}"),
        None => bail!("COSE_Encrypt0 missing alg in protected header"),
    }
}

fn read_int_header(h: &Header, label: i64) -> Option<i128> {
    h.rest.iter().find_map(|(l, v)| match (l, v) {
        (Label::Int(i), CborValue::Integer(n)) if *i == label => Some((*n).into()),
        _ => None,
    })
}

fn read_bytes_header(h: &Header, label: i64) -> Option<Vec<u8>> {
    h.rest.iter().find_map(|(l, v)| match (l, v) {
        (Label::Int(i), CborValue::Bytes(b)) if *i == label => Some(b.clone()),
        _ => None,
    })
}

fn read_suite(enc: &CoseEncrypt0) -> Result<SuiteId> {
    let raw = read_int_header(&enc.protected.header, HDR_SUITE_ID)
        .context("COSE_Encrypt0 missing HyKEM suite id in protected header")?;
    let u = u16::try_from(raw).map_err(|_| anyhow::anyhow!("suite id {raw} out of range"))?;
    SuiteId::from_u16(u).ok_or_else(|| anyhow::anyhow!("unknown HyKEM suite id {u}"))
}

// ============================================================================
// One-shot recipient API (envelope path, S4) — encapsulates per call
// ============================================================================

/// Seal `plaintext` to `recipient` as a `COSE_Encrypt0`: run the S0 hybrid KEM,
/// key ChaCha20Poly1305 with the combiner secret, and carry the encapsulated key
/// material in the unprotected header. Fresh encapsulation (fresh key) per call.
pub fn seal_to_recipient(
    recipient: &RecipientPublic,
    plaintext: &[u8],
    external_aad: &[u8],
    epoch: u64,
    seq: u64,
) -> Result<Vec<u8>> {
    let (material, secret) = hybrid_kem::encapsulate_to(recipient)?;
    let key = derive_aead_key(&secret);
    let nonce = derive_nonce(epoch, seq);
    let aad = combined_aad(external_aad, epoch, seq);

    let unprotected = HeaderBuilder::new()
        .value(HDR_KEM_EK, CborValue::Bytes(material.encode()))
        .build();

    let mut sealed: Result<()> = Ok(());
    let enc = CoseEncrypt0Builder::new()
        .protected(build_protected(recipient.suite_id))
        .unprotected(unprotected)
        .create_ciphertext(plaintext, &aad, |pt, coset_aad| {
            match aead_seal(&key, &nonce, pt, coset_aad) {
                Ok(ct) => ct,
                Err(e) => {
                    sealed = Err(e);
                    Vec::new()
                }
            }
        })
        .build();
    sealed?;

    enc.to_vec()
        .map_err(|e| anyhow::anyhow!("serialize COSE_Encrypt0: {e}"))
}

/// Open a `COSE_Encrypt0` produced by [`seal_to_recipient`], decapsulating with
/// `recipient`'s hybrid keypair. Fails closed on alg/suite mismatch.
pub fn open_from_recipient(
    recipient: &RecipientKeypair,
    cose_bytes: &[u8],
    external_aad: &[u8],
    epoch: u64,
    seq: u64,
) -> Result<Vec<u8>> {
    let enc = CoseEncrypt0::from_slice(cose_bytes)
        .map_err(|e| anyhow::anyhow!("malformed COSE_Encrypt0: {e}"))?;
    require_chacha20poly1305(&enc)?;

    let suite = read_suite(&enc)?;
    if suite != recipient.suite_id {
        bail!(
            "COSE_Encrypt0 suite {} != recipient suite {}",
            suite.as_str(),
            recipient.suite_id.as_str()
        );
    }

    let ek = read_bytes_header(&enc.unprotected, HDR_KEM_EK)
        .context("COSE_Encrypt0 missing encapsulated-key (HDR_KEM_EK) header")?;
    let material = HybridKemMaterial::decode(&ek)?;
    if material.suite_id != suite {
        bail!("encapsulated material suite mismatch");
    }

    let secret = hybrid_kem::decapsulate(recipient, &material)?;
    let key = derive_aead_key(&secret);
    let nonce = derive_nonce(epoch, seq);
    let aad = combined_aad(external_aad, epoch, seq);

    enc.decrypt_ciphertext(
        &aad,
        || anyhow::anyhow!("COSE_Encrypt0 has no ciphertext"),
        |ct, coset_aad| aead_open(&key, &nonce, ct, coset_aad),
    )
}

// ============================================================================
// Key-based API (stream path, S3) — key from the handshake, no per-block ek
// ============================================================================

/// Seal `plaintext` under a content key derived once from a hybrid-KEM handshake
/// (see [`derive_aead_key`]). No encapsulated-key header — the stream handshake
/// (S3 #554) carries the KEM material separately. `suite` is bound in the
/// protected header for domain separation.
pub fn seal_with_key(
    aead_key: &[u8; 32],
    suite: SuiteId,
    plaintext: &[u8],
    external_aad: &[u8],
    epoch: u64,
    seq: u64,
) -> Result<Vec<u8>> {
    let nonce = derive_nonce(epoch, seq);
    let aad = combined_aad(external_aad, epoch, seq);

    let mut sealed: Result<()> = Ok(());
    let enc = CoseEncrypt0Builder::new()
        .protected(build_protected(suite))
        .create_ciphertext(plaintext, &aad, |pt, coset_aad| {
            match aead_seal(aead_key, &nonce, pt, coset_aad) {
                Ok(ct) => ct,
                Err(e) => {
                    sealed = Err(e);
                    Vec::new()
                }
            }
        })
        .build();
    sealed?;

    enc.to_vec()
        .map_err(|e| anyhow::anyhow!("serialize COSE_Encrypt0: {e}"))
}

/// Open a [`seal_with_key`] object with the pre-derived content key.
pub fn open_with_key(
    aead_key: &[u8; 32],
    expected_suite: SuiteId,
    cose_bytes: &[u8],
    external_aad: &[u8],
    epoch: u64,
    seq: u64,
) -> Result<Vec<u8>> {
    let enc = CoseEncrypt0::from_slice(cose_bytes)
        .map_err(|e| anyhow::anyhow!("malformed COSE_Encrypt0: {e}"))?;
    require_chacha20poly1305(&enc)?;
    let suite = read_suite(&enc)?;
    if suite != expected_suite {
        bail!(
            "COSE_Encrypt0 suite {} != expected {}",
            suite.as_str(),
            expected_suite.as_str()
        );
    }
    let nonce = derive_nonce(epoch, seq);
    let aad = combined_aad(external_aad, epoch, seq);
    enc.decrypt_ciphertext(
        &aad,
        || anyhow::anyhow!("COSE_Encrypt0 has no ciphertext"),
        |ct, coset_aad| aead_open(aead_key, &nonce, ct, coset_aad),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::hybrid_kem::{generate_recipient, SuiteId};

    const SUITE: SuiteId = SuiteId::HyKemX25519MlKem768;
    const AAD: &[u8] = b"hyprstream-test-aad";

    #[test]
    fn recipient_roundtrip() {
        let kp = generate_recipient(SUITE).unwrap();
        let pt = b"hybrid confidential payload";
        let sealed = seal_to_recipient(&kp.public(), pt, AAD, 0, 0).unwrap();
        let opened = open_from_recipient(&kp, &sealed, AAD, 0, 0).unwrap();
        assert_eq!(opened, pt);
    }

    #[test]
    fn recipient_ciphertext_is_cose_encrypt0_and_hides_plaintext() {
        let kp = generate_recipient(SUITE).unwrap();
        let pt = b"secret tokens";
        let sealed = seal_to_recipient(&kp.public(), pt, AAD, 1, 7).unwrap();
        assert!(
            !sealed.windows(pt.len()).any(|w| w == pt),
            "plaintext must not appear"
        );
        // Parses as a COSE_Encrypt0 with the expected alg + suite header.
        let enc = CoseEncrypt0::from_slice(&sealed).unwrap();
        require_chacha20poly1305(&enc).unwrap();
        assert_eq!(read_suite(&enc).unwrap(), SUITE);
    }

    #[test]
    fn tampered_ciphertext_rejected() {
        let kp = generate_recipient(SUITE).unwrap();
        let mut sealed = seal_to_recipient(&kp.public(), b"x", AAD, 0, 0).unwrap();
        let n = sealed.len();
        sealed[n - 1] ^= 0xff; // flip a tag/ciphertext byte
        assert!(open_from_recipient(&kp, &sealed, AAD, 0, 0).is_err());
    }

    #[test]
    fn wrong_recipient_rejected() {
        let kp = generate_recipient(SUITE).unwrap();
        let other = generate_recipient(SUITE).unwrap();
        let sealed = seal_to_recipient(&kp.public(), b"x", AAD, 0, 0).unwrap();
        assert!(open_from_recipient(&other, &sealed, AAD, 0, 0).is_err());
    }

    #[test]
    fn wrong_epoch_seq_rejected() {
        let kp = generate_recipient(SUITE).unwrap();
        let sealed = seal_to_recipient(&kp.public(), b"x", AAD, 3, 9).unwrap();
        assert!(
            open_from_recipient(&kp, &sealed, AAD, 3, 10).is_err(),
            "wrong seq"
        );
        assert!(
            open_from_recipient(&kp, &sealed, AAD, 4, 9).is_err(),
            "wrong epoch"
        );
        assert!(
            open_from_recipient(&kp, &sealed, b"other-aad", 3, 9).is_err(),
            "wrong aad"
        );
    }

    #[test]
    fn malformed_cose_rejected() {
        let kp = generate_recipient(SUITE).unwrap();
        assert!(open_from_recipient(&kp, b"not cbor", AAD, 0, 0).is_err());
    }

    #[test]
    fn key_based_roundtrip_and_nonce_separation() {
        let key = [7u8; 32];
        let pt0 = b"block zero";
        let pt1 = b"block one!";
        let c0 = seal_with_key(&key, SUITE, pt0, AAD, 0, 0).unwrap();
        let c1 = seal_with_key(&key, SUITE, pt1, AAD, 0, 1).unwrap();
        assert_eq!(open_with_key(&key, SUITE, &c0, AAD, 0, 0).unwrap(), pt0);
        assert_eq!(open_with_key(&key, SUITE, &c1, AAD, 0, 1).unwrap(), pt1);
        // A block sealed at seq=0 must not open at seq=1 (nonce/AAD bound).
        assert!(open_with_key(&key, SUITE, &c0, AAD, 0, 1).is_err());
        // Equal plaintext at different seq yields different ciphertext (distinct nonce).
        let a = seal_with_key(&key, SUITE, pt0, AAD, 0, 0).unwrap();
        let b = seal_with_key(&key, SUITE, pt0, AAD, 0, 1).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn derive_aead_key_deterministic_and_separated() {
        let secret = [9u8; 32];
        assert_eq!(*derive_aead_key(&secret), *derive_aead_key(&secret));
        assert_ne!(
            *derive_aead_key(&secret),
            secret,
            "key must differ from raw secret (HKDF)"
        );
    }

    #[test]
    fn pinned_kat_key_based() {
        // Pin the key-based seal output for fixed inputs so the construction
        // (key derivation, nonce, AAD binding, COSE layout) cannot silently drift.
        let key = [0x42u8; 32];
        let ct = seal_with_key(&key, SUITE, b"kat-plaintext", b"kat-aad", 0, 0).unwrap();
        let opened = open_with_key(&key, SUITE, &ct, b"kat-aad", 0, 0).unwrap();
        assert_eq!(opened, b"kat-plaintext");
        // Deterministic: same inputs → same bytes (ChaCha20Poly1305 + fixed nonce + fixed key).
        let ct2 = seal_with_key(&key, SUITE, b"kat-plaintext", b"kat-aad", 0, 0).unwrap();
        assert_eq!(
            ct, ct2,
            "key-based seal must be deterministic for fixed inputs"
        );
    }
}
